# Bonsai — High-Level Design

**Bonsai** is a KV-cache handoff suite for disaggregated LLM inference. A
**control plane** decides where each request runs and how its KV cache is
encoded. A **data plane** compresses the cache on the prefill GPU, moves it
with Perplexity's **fabric-lib** (RDMA, GPUDirect), and rebuilds it on the
decode GPU.

It is the system for the CSE 561 project *"Quantized KV-cache transfer over
fabric-lib"* (see `../proposal.txt`). Model: **Qwen2.5-1.5B-Instruct**, BF16,
through Hugging Face Transformers. Codecs: **BF16** (baseline), **INT8**, and
**packed INT4**. The project's suggested extension, **adaptive precision**, is
part of the core design (§6).

---

## 1. Goals and non-goals

**Goals**

1. Move a prefilled KV cache from a prefill GPU to a decode GPU on another node
   over GPUDirect RDMA, and continue generation correctly.
2. Make the encoding pluggable (BF16 / INT8 / INT4 / per-layer mixed) without
   changing anything else, so the configurations being compared differ *only*
   in the codec.
3. Measure every quantity in the proposal for each handoff: bytes on the wire
   (including metadata), encode and decode overhead, handoff latency, TTFT,
   decode throughput, and temporary buffer usage.
4. Make the knobs we sweep explicit: context length, chunk size, concurrency,
   and available bandwidth.
5. Choose precision per request based on context length and current bandwidth,
   under a quality budget (adaptive precision).

**Non-goals**

- A production serving stack (no batching scheduler, paged attention, or
  multi-model support). Each decode worker runs HF `generate` for one request
  at a time per slot.
- Changing model weights or attention kernels. Only the KV cache *in transit*
  is quantized; the decode worker computes with reconstructed BF16.
- Multi-node prefill/decode parallelism (TP/PP). One GPU per worker.

---

## 2. System overview

```text
                              ┌──────────────────────────────┐
     load generator /  ──────▶│   bonsai-ctl  (control plane) │◀─────── metrics.jsonl
     eval harness       TCP   │ registry · placement · codec  │
                              │ policy · handoff sessions     │
                              └───────┬──────────────┬────────┘
                         Prefill / SlotReady      LeaseSlot / Abort        (TCP, postcard)
                                      │              │
 ┌─────────────── Node A ─────────────▼─┐        ┌───▼───────────── Node B ──────────────┐
 │ prefill worker (Python)              │        │ decode worker (Python)                 │
 │  HF model.forward(prompt)            │        │  HF model.generate(past_key_values=…)  │
 │      │ DynamicCache (BF16, per layer)│        │      ▲ DynamicCache (BF16, rebuilt)     │
 │  ────┼── bonsai-py / data plane ─────│        │  ────┼── bonsai-py / data plane ─────── │
 │      ▼                               │        │      │                                  │
 │  encode kernels ─▶ staging ring      │  RDMA  │  receive arena ─▶ decode kernels         │
 │                    (registered) ─────┼═══════▶│  (registered)    (progressive)           │
 │  fabric-lib TransferEngine           │ WRITE  │  fabric-lib TransferEngine               │
 │  GPU ── PCIe ── NIC (CX-7 / EFA)     │ + imm  │  NIC ── PCIe ── GPU                      │
 └──────────────────────────────────────┘        └─────────────────────────────────────────┘
```

| | Control plane (`bonsai-control`) | Data plane (`bonsai-dataplane`, `bonsai-codec`, `bonsai-transport`) |
|---|---|---|
| Runs | One `bonsai-ctl` process | Inside every worker process, called from Python through `bonsai-py` |
| Carries | Small messages (KB) over TCP | The KV cache (MB to GB) over RDMA |
| State | Workers, load, handoff state machines, bandwidth estimate | Staging ring, receive arena, immediate counters |
| Decides | Placement, codec and chunking, admission, retries and fallback | Nothing about policy; executes a `HandoffParams` |
| On the critical path? | Only to forward the `SlotGrant`, and that overlaps prefill | Yes |

---

## 3. Request lifecycle

```text
client        bonsai-ctl                 prefill P                    decode D
  │ Submit(req)   │                          │                            │
  │──────────────▶│ choose codec (§6),       │                            │
  │               │ place (P, D), build plan │                            │
  │◀──Accepted────│                          │                            │
  │               │──Prefill(params)────────▶│ forward(prompt) ...        │
  │               │──LeaseSlot(params)──────────────────────────────────▶│ lease slot, arm imm
  │               │◀─────────────────────────────────────SlotGranted─────│ counters, then grant
  │               │──SlotReady(grant)───────▶│                            │
  │               │                          │ encode c0 ▶ WRITE c0 ═════▶│ (group counters
  │               │                          │ encode c1 ▶ WRITE c1 ═════▶│  count imms)
  │               │                          │   ...   ▶ WRITE trailer ══▶│ group k complete ▶ decode group k
  │               │◀──PrefillDone(timings)───│                            │ all groups ▶ DynamicCache
  │               │                          │                            │ generate(first_token, cache)
  │               │◀──────────────────────────────────Finished(tokens, report)
  │◀─Completed────│ write HandoffReport                                   │ release slot
```

Key points:

- **Leasing overlaps prefill.** `Prefill` and `LeaseSlot` are sent together.
  The sender blocks on the grant only right before its first RDMA write, so
  the control-plane round trip is normally hidden behind prefill compute.
- **The slot size is known before prefill.** The encoded size is a
  deterministic function of `(layout, precision, seq_len, chunking)`, so the
  coordinator and decode worker size the slot from the prompt length alone.
- **One completion mechanism.** Every frame write carries an RDMA immediate.
  The receiver counts immediates per layer group. The trailer (first token,
  sender timings) is also written with an immediate, so there is no separate
  "done" message on the data path.

---

## 4. Data plane

### 4.1 What is transferred

HF Transformers keeps, per layer, `K, V : [1, num_kv_heads, seq, head_dim]` in
BF16. For Qwen2.5-1.5B that is 28 layers × 2 KV heads (GQA) × 128 dims.

| Codec | Bytes/token (incl. metadata) | vs BF16 | 8K-token prompt | 32K-token prompt |
|---|---:|---:|---:|---:|
| BF16 | 28,672 | 100% | 224 MiB | 896 MiB |
| INT8, symmetric, 1 scale per (layer, head, token) | 14,560 | 50.8% | 113.8 MiB | 455 MiB |
| INT4, asymmetric, group 32 (K per-channel, V per-token) | 8,960 | 31.3% | 70 MiB | 280 MiB |
| Mixed: INT8 on layers 0–3 and 24–27, INT4 elsewhere | 10,560 | 36.8% | 82.5 MiB | 330 MiB |

These numbers come from `CodecSpec::encoded_bytes`, `PrecisionMap::encoded_bytes`
and their unit tests. Frame padding (256 B per section) adds less than 0.1% at
these sizes.

### 4.2 Wire format

A handoff is a series of **chunk frames** plus a 256 B **trailer**, laid out
contiguously in the receiver's slot:

```text
slot ─┬─ frame 0 [K payload | V payload | K scales/zeros | V scales/zeros]
      ├─ frame 1 ...
      └─ trailer [magic | first_token | sender timings | checksum]
```

- A **chunk** is a rectangle of `layers × tokens`. Chunk size is the
  experiment's "transfer chunk size" knob (`ChunkSpec`).
- Each frame contains its own quantization metadata, so it can be decoded as
  soon as it lands.
- A chunk never mixes codecs. With mixed precision, layer groups are split at
  precision boundaries.
- Both ends compute the layout (the `ChunkPlan`) independently and get the same
  result. Only `HandoffParams` travels over the network; there is no manifest
  on the wire.

### 4.3 Sender pipeline (prefill)

```text
CUDA stream:  [prefill] [enc c0][enc c1][enc c2] ...
RDMA:                         [WRITE c0][WRITE c1][WRITE c2] ... [WRITE trailer]
```

- Encode kernels read straight from HF's per-layer tensors (pointer + stride)
  and write into a **staging ring** that was registered once at startup.
  There is no intermediate full-size copy, and BF16 goes through the same path
  (a strided pack), so the baseline pays the same pipeline costs.
- A CUDA event after each chunk's encode gates that chunk's RDMA write.
- A full ring stops the encoder from running ahead of the network. Each write
  runs in its own task that owns its ring entry, which avoids a deadlock
  between waiting for staging space and completing writes.
- The ring depth × frame size is the sender's temporary-buffer footprint,
  independent of context length. A "whole" staging mode, with no pipelining,
  is kept as an ablation and as the milestone 1 baseline.

### 4.4 Receiver (decode)

- The **receive arena** is one large registered GPU buffer carved into slots.
  It is the system's credit pool: concurrent in-flight KV ≤ arena size.
- `prepare` leases a slot and **arms one immediate counter per layer group
  before returning the grant**. fabric-lib resets a counter when it is armed
  (`set_imm_count_expected`), so an immediate that arrives earlier would be
  lost. This ordering is the single most important data-plane invariant.
- `receive` waits for group 0, decodes it into new BF16 tensors, then group 1,
  and so on. Decoding overlaps the arrival of later layers. The last group
  includes the trailer, which provides the first token.
- The new tensors are wrapped as a `DynamicCache`, and generation continues
  from `first_token`.

### 4.5 fabric-lib usage

| Need | fabric-lib API |
|---|---|
| NIC/GPU topology, engine setup | `detect_topology`, `TransferEngineBuilder::add_gpu_domains` |
| Register staging (source) | `register_memory_local` |
| Register arena (remote-writable) and get the descriptor | `register_memory_allow_remote` → `MemoryRegionDescriptor` |
| Write a frame | `TransferRequest::Single{ imm_data, domain: RoundRobinSharded }` |
| Completion counting | `set_imm_count_expected` (armed synchronously), `remove_imm_count` on abort |
| Multi-NIC aggregation / bandwidth limiting | `num_shards` (all NICs vs. 1 NIC) |

fabric-lib's transport is *reliable but unordered* (EFA SRD). Bonsai never
relies on write ordering; counting immediates makes completion
order-independent.

All of this sits behind Bonsai's `Transport` trait. `FabricTransport` is the
real implementation. `LoopbackTransport` (in-process, host memory) has the
same immediate semantics, so pipeline logic can be developed and tested on a
laptop.

---

## 5. Control plane

- **Registry.** Workers register their role, GPU, fabric address, link speed,
  model layout and arena size, then send heartbeats with load. Missing
  heartbeats mark a worker *suspect*, then *lost*.
- **Placement** (`PlacementPolicy`, default `LeastLoaded`). Picks the prefill
  worker with the fewest active requests, and a decode worker with enough
  free arena after subtracting reservations not yet confirmed.
- **Codec policy** (`CodecPolicy`). `StaticCodecPolicy` is used for the sweeps;
  `AdaptivePrecisionPolicy` is described in §6.
- **Sessions.** Each handoff is a state machine:
  `Queued → Leasing → Transferring → Decoding → Done`, with `Failed` reachable
  from any state.
- **Admission and backpressure.** FIFO admission. A request waits while no
  decode worker has arena space, and a full queue rejects new requests. A
  `SlotDenied` reply makes the coordinator try another decode worker while
  the prefill worker keeps its KV.
- **Failure handling.**
  - On a timeout, a lost worker or a reported error, both workers get `Abort`.
  - The receiver disarms its counters and **quarantines** the slot, so a late
    write from the failed sender cannot corrupt the next lease.
  - A failed quantized handoff is retried once as BF16 under a **new**
    `HandoffId`. The new slot epoch means stale immediates can't be miscounted.
- **Design style.** The coordinator is *sans-IO*: it consumes a message and
  returns the messages to send. A thin tokio TCP server wraps it, so every
  protocol path is unit-tested without sockets (`tests/coordinator_flow.rs`).

---

## 6. Adaptive precision

This answers the feedback: *"Adaptive precision could nicely round out the work."*

Adaptive precision works at two levels, both built on the same frame format.

**(a) Per-layer mixed precision (`PrecisionMap`)**

Layers differ in how much quantization hurts. Early layers, and layers with
large K outliers, are usually the sensitive ones. A `PrecisionMap` assigns a
codec to each layer range, for example INT8 on the sensitive layers and INT4
elsewhere. The chunk planner, codecs and receiver handle this natively: a
chunk has exactly one codec, and the receiver uses one codec instance per
distinct spec.

Sensitive layers are chosen offline. We quantize one layer at a time to INT4,
measure F1 loss on a held-out dev split (and, as a cheap proxy, the KV
reconstruction error per layer), and then protect the top-k layers.

**(b) Per-request selection (`AdaptivePrecisionPolicy`)**

For each request the coordinator:

1. Drops candidate configs whose offline quality loss at this context-length
   bucket exceeds the tolerance (the same tolerance as the hypothesis).
2. Predicts handoff latency for each remaining candidate with a calibrated
   `CostModel`. The model accounts for encode and decode throughput, per-write
   and per-kernel fixed costs, pipelining across chunks, and progressive
   decode of all but the last layer group.
3. Uses the bandwidth the request can actually expect: an EWMA of achieved
   goodput from recent reports (`BandwidthEstimator`). Before the first
   report, it uses line rate divided by the number of in-flight handoffs.
4. Picks the fastest candidate only if it beats the least-lossy admissible
   one by at least `min_gain` (hysteresis). Otherwise it keeps higher
   precision.

Expected behavior: short prompts and idle fast links stay at BF16/INT8;
long prompts, contention, or constrained links move to INT4 or mixed
precision. The evaluation compares adaptive against each static config on the
quality-vs-latency plane. The cost model's predictions are also checked
against measured latency, which is a result in its own right.

---

## 7. Measurement

All per-handoff numbers go into a `HandoffReport` (one JSONL line). Each
duration is measured on a single host's monotonic clock; Bonsai never
subtracts timestamps taken on different hosts.

| Proposal metric | How it is measured |
|---|---|
| Transmitted bytes (incl. metadata) | `wire_bytes` = sum of write lengths; `metadata_bytes` and `useful_bytes` reported separately |
| Encode overhead | CUDA events around encode kernels (`encode_gpu_ns`) and time spent waiting for staging |
| Reconstruction overhead | CUDA events around decode kernels (`decode_gpu_ns`), wall time, and `cache_build_ns` (tensors → `DynamicCache`) |
| KV handoff latency | sender `handoff_send_ns` (prefill done → last write completed; an RDMA write completion means the data is in remote memory) + receiver decode + cache build |
| TTFT | `ttft_ns`: submit → first token from prefill. **`ttfdt_ns`**: submit → first token generated on the decode worker, which includes the handoff and is the TTFT that compression can change |
| Throughput | decode tokens/s per request, and aggregate requests/s and tokens/s at each concurrency level (load generator) |
| Temporary buffers | staging footprint, slot bytes, codec scratch, plus `torch.cuda.max_memory_allocated` deltas from Python |
| Task quality | QASPER and HotpotQA F1 (LongBench v1), RULER single-needle accuracy; computed by the eval harness from `output_tokens` |

---

## 8. Deployment and hardware

This answers the question *"How do you plan to get two GPUDirect RDMA nodes?"*

**Benchmark target: two GPUDirect RDMA nodes.** All latency, throughput and
bandwidth results come from two separate nodes:

- Node A runs the prefill worker; node B runs the decode worker.
- Each node has at least one GPU with a dedicated RDMA NIC (ConnectX-7 or
  AWS EFA, both supported by fabric-lib). GPUDirect RDMA uses DMA-BUF.
- Software must meet fabric-lib's requirements: Linux ≥ 5.12, CUDA ≥ 12.8,
  libfabric, libibverbs, GDRCopy, and the `SYS_PTRACE`/`SYS_ADMIN`
  capabilities (the pplx-garden dev Docker image provides these).
- `bonsai-ctl` runs on either node; it is not on the data path.
- Host setup, the Docker image and the build steps are in [BUILD.md](BUILD.md).
- Before any Bonsai run, `fabric-debug` (from pplx-garden) validates the link
  and records the line rate we compare against.

**Using node time efficiently.** The two-node setup is needed only for the
experiments that involve the network. Everything else runs elsewhere:

| Work | Where it runs | Why it doesn't need two nodes |
|---|---|---|
| Quality eval (LongBench, RULER) for every precision config | Any single GPU, using `bonsai.roundtrip_kv` (`Colocated` role) | Encode → decode on one GPU is bit-identical to a network handoff, because the network moves bytes unchanged |
| Codec kernel development and microbenchmarks | Any single GPU | No network involved |
| Pipeline, protocol and failure-path testing | Laptop or CI with `LoopbackTransport` | Same immediate semantics as fabric-lib |
| Handoff latency, TTFT, throughput, bandwidth sweeps | **The two RDMA nodes** | This is what the network measurements require |

**Bandwidth-constrained conditions** (hypothesis 2) are produced on the same
two nodes, without special hardware:

- `write_shards = 1` uses one NIC instead of all of them.
- `throttle_bps` adds a token-bucket cap on sender egress, swept for example
  at 10 / 25 / 50 / 100 Gb/s up to line rate.

Throttled runs are labeled as emulated in the report, separately from
line-rate runs.

---

## 9. Experiment matrix → milestones

| Milestone | Bonsai deliverable | Experiments |
|---|---|---|
| **M1** Baseline (Oct 20 – Nov 2) | `bonsai-core` (done), `FabricTransport`, BF16 pack codec, whole-buffer staging, coordinator happy path, `bonsai-py` send/recv | BF16 handoff on 2 nodes; verify output tokens match colocated HF `generate`; baseline bytes, latency, TTFDT, buffers |
| **M2** Compression (Nov 3 – 16) | INT8 and INT4 kernels + CPU reference oracle, staging ring pipelining, progressive decode, `roundtrip_kv` | Kernel vs reference error stats; INT8 prototype by Nov 9; encode/decode overhead vs context length |
| **M3** Evaluation (Nov 17 – Dec 7) | Mixed precision, `AdaptivePrecisionPolicy`, cost-model calibration, load generator | Full matrix: {BF16, INT8, INT4, mixed, adaptive} × context length {1K … 32K} × chunk size × concurrency {1, 2, 4, 8} × bandwidth {line rate, throttled levels} |

---

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| At 400 Gb/s line rate, compression may never win on latency | That outcome still tests hypothesis 2; throttled and contention runs map the crossover point, and the cost model explains it |
| HF `DynamicCache` tensors are allocated per request, and registering them per request would be slow | Encode from them into a pre-registered staging ring (one extra read, no registration on the critical path) |
| INT4 per-channel K requires chunk token boundaries aligned to group size | Enforced by the planner (`token_alignment`); chunk sizes in the sweep are multiples of 32 |
| Late writes after an abort | Slot epochs in immediates + quarantine before reuse + new `HandoffId` on retry |
| fabric-lib API drift | All fabric-lib calls are confined to `bonsai-transport/src/fabric.rs`; pin the pplx-garden git revision in `Cargo.lock` |
