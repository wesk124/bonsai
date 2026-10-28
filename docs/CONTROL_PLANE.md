# Control Plane User Guide (`bonsai-ctl`)

`bonsai-ctl` is Bonsai's coordinator. Prefill and decode workers register with
it. Clients (the load generator, the eval harness) submit requests to it. For
each request it picks a worker pair and a KV precision, runs the handoff
protocol, and writes one metrics record when the request finishes. It never
touches KV bytes; those move worker-to-worker over fabric-lib.

> **Status.** Implemented: the configuration loader and `--check`, the
> coordinator state machine (registration, placement, slot leasing,
> timeouts, BF16 fallback), the static and adaptive codec policies, and the
> JSONL metrics sink. Not implemented yet: the TCP server (`server::serve`)
> and the worker/client connections. These are milestone 1 items, so for now
> `bonsai-ctl` without `--check` validates the config and exits.

Contents:

1. Quick start
2. Configuration reference
3. Choosing precisions
4. Static sweeps vs. adaptive precision
5. Workers and requests
6. Failure handling
7. Metrics output
8. Troubleshooting

---

## 1. Quick start

```bash
cargo build --release -p bonsai-control
cp config/bonsai-ctl.example.toml config/bonsai-ctl.toml   # edit: listen address, codec, metrics path

./target/release/bonsai-ctl --config config/bonsai-ctl.toml --check   # validate + show wire sizes
./target/release/bonsai-ctl --config config/bonsai-ctl.toml           # run (after M1)
./target/release/bonsai-ctl --config config/bonsai-ctl.toml --listen 10.0.0.1:7000   # override address
```

`--check` resolves every codec and precision in the file, validates chunking,
and prints the wire cost of each precision:

```text
precision           B/token   ratio     1K MiB     8K MiB    32K MiB
bf16                  28672  100.0%       28.0      224.0      896.0
int8                  14560   50.8%       14.2      113.8      455.0
int4                   8960   31.2%        8.8       70.0      280.0
int4-pertoken          8960   31.2%        8.8       70.0      280.0
atom-kv4               7616   26.6%        7.4       59.5      238.0
int8-edges            10560   36.8%       10.3       82.5      330.0
...
```

Exit code 0 means the config is valid, 1 means a config error (the message
names the key), and 2 means bad command-line arguments.

**Where to run it.** On either benchmark node, or a third machine. It only
needs plain TCP reachability from both workers; it is not on the RDMA path.

---

## 2. Configuration reference

The file is TOML. Unknown keys are rejected, so a typo fails `--check`
instead of being silently ignored. The fully commented template is
[`config/bonsai-ctl.example.toml`](../config/bonsai-ctl.example.toml).

### `[server]`

| Key | Default | Meaning |
|---|---|---|
| `listen` | required | TCP `addr:port` for workers and clients |
| `tick_ms` | 50 | How often timeouts and heartbeats are checked |

### `[model]`

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Model id. Workers registering a different layout are rejected. |
| `preset` | — | `"qwen2.5-1.5b-instruct"` (28 layers × 2 KV heads × 128 dims, bf16) |
| `num_layers`, `num_kv_heads`, `head_dim`, `dtype` | — | Explicit layout when there's no preset; `dtype` is `bf16` \| `f16` \| `f32` |

### `[admission]` (optional)

| Key | Default | Meaning |
|---|---|---|
| `max_queue` | 1024 | Requests waiting for placement; beyond this, `Submit` is rejected |
| `max_lease_attempts` | 2 | Decode workers tried per handoff after `SlotDenied`, before re-queueing |

### `[timeouts]` (optional)

| Key | Default | Meaning |
|---|---|---|
| `heartbeat_ms` | 2000 | No heartbeat for 1× → worker is *suspect* (gets no new work); for 3× → *lost* (its handoffs fail) |
| `handoff_ms` | 10000 | Max time from slot lease until all KV is written |
| `decode_ms` | 300000 | Max time from KV received until generation finishes |

### `[failure]` (optional)

| Key | Default | Meaning |
|---|---|---|
| `fallback_to_bf16` | true | Retry a failed quantized handoff as BF16 |
| `max_retries` | 1 | Retries per request |

### `[placement]` (optional)

| Key | Default | Meaning |
|---|---|---|
| `policy` | `"least_loaded"` | Prefill worker with the fewest active + queued requests; decode worker with the fewest active requests that has enough free receive-arena memory |

### `[codecs.<name>]` (optional): custom codecs

| Key | Meaning |
|---|---|
| `bits` | 16 (BF16), 8, or 4 (packed) |
| `key`, `value` | `{ group_size, axis, symmetric }`. `axis = "head_dim"` groups channels within a token (per-token); `axis = "token"` groups tokens within a channel (per-channel), and chunk token sizes must then be multiples of `group_size`. |

### `[precisions.<name>]` (optional): per-layer mixed precision

| Key | Meaning |
|---|---|
| `default` | Codec for layers not covered by an override |
| `overrides` | List of `{ layers = [start, end), codec = "<name>" }`. Later entries win. |

### `[codec]`

| Key | Meaning |
|---|---|
| `policy` | `"static"` or `"adaptive"` |
| `[codec.static]` | `precision`, plus `chunk = { tokens, layers }` (0 means all) |
| `[codec.adaptive]` | `max_quality_loss`, `min_gain` (0.1), `bandwidth_ewma_alpha` (0.2), `[codec.adaptive.cost_model]`, and `[[codec.adaptive.candidates]]` (see §4) |

### `[metrics]` (optional)

| Key | Meaning |
|---|---|
| `jsonl` | Output file, one `HandoffReport` per line. Parent directories are created and the file is truncated at start. Omit it to keep reports in memory only. |

---

## 3. Choosing precisions

Built-in codec names work anywhere a codec or precision is expected:

| Name | Scheme | Bytes vs. BF16 |
|---|---|---:|
| `bf16` | Uncompressed baseline | 100% |
| `int8` | Symmetric, 1 f16 scale per (layer, head, token) over 128 channels | 50.8% |
| `int4` | Asymmetric, group 32. K per-channel (KIVI-style, robust to K outlier channels), V per-token | 31.3% |
| `int4-pertoken` | Asymmetric, group 32, K and V per-token | 31.3% |
| `atom-kv4` | Atom's (MLSys'24) KV scheme: asymmetric, one group per (head, token) | 26.6% |

**Chunking** (`chunk = { tokens, layers }`) controls the unit of one RDMA
write:

- `layers = 1` (per layer) is the default for experiments. It pipelines
  encode with transfer, and the decode side starts dequantizing early layers
  while later ones are still in flight.
- `tokens = N` additionally caps frame size for very long prompts. N must be a
  multiple of 32 for any codec that uses `axis = "token"` with group 32
  (`int4`, for example); `--check` catches this.
- `{ tokens = 0, layers = 0 }` sends the whole cache as one write (the
  no-pipelining ablation).

**Mixed precision.** Protect the layers where quantization hurts most and
compress the rest:

```toml
[precisions.int8-edges]
default = "int4"
overrides = [{ layers = [0, 4], codec = "int8" }, { layers = [24, 28], codec = "int8" }]
```

Pick the protected layers with the layer-sensitivity sweep
(`python -m bonsai_bench.ppl --layer-sensitivity int4`; see
[BENCHMARKS.md](BENCHMARKS.md)). Then check the result with
`bonsai_bench.ppl --config your.toml --precisions int8-edges`, which reads the
same TOML.

---

## 4. Static sweeps vs. adaptive precision

### Static: used for the main experiment matrix

Each run uses one configuration for every request. Use one config file per
point in the sweep, and keep a separate metrics file per run:

```bash
for p in bf16 int8 int4 int8-edges; do
  sed -e "/^\[codec.static\]/,/^\[/ s/^precision = .*/precision = \"$p\"/" \
      -e "s|^jsonl = .*|jsonl = \"results/static-$p.jsonl\"|" \
      config/bonsai-ctl.toml > /tmp/ctl-$p.toml
  bonsai-ctl --config /tmp/ctl-$p.toml --check && bonsai-ctl --config /tmp/ctl-$p.toml
  # ... run the load generator against it, then stop bonsai-ctl
done
```

A request can also override the precision and chunking itself
(`InferenceRequest.precision` / `.chunk`), which is how the eval harness can
mix configs within one run.

### Adaptive: picks a precision per request

For each request, the coordinator:

1. Keeps the candidates whose measured quality loss at the request's context
   length is ≤ `max_quality_loss`. Contexts longer than a candidate's last
   bucket count as unmeasured, so that candidate is skipped.
2. Predicts each candidate's handoff latency with the cost model, at the
   bandwidth the request can expect: the EWMA of recently achieved goodput,
   or line rate divided by in-flight handoffs before the first measurement.
3. Picks the fastest candidate, but only if it beats the least-lossy eligible
   candidate by at least `min_gain`. Otherwise it keeps the least-lossy one.

Filling in the numbers:

| Field | Source |
|---|---|
| `candidates[].quality_loss` | `[[max_ctx, f1_loss], ...]` from the quality sweep on the dev split (LongBench F1 vs. `bf16`), per context bucket |
| `cost_model.{bf16,int8,int4}` | `python -m bonsai_bench.codec_bench` on the benchmark GPU (only meaningful once the CUDA kernels exist) |
| `cost_model.link_efficiency`, `per_write_us` | `fabric-debug` bandwidth ÷ line rate, and small-message latency |
| `per_chunk_kernel_us` | Codec microbenchmark at small context lengths |

The template's numbers are placeholders. Don't report adaptive results until
these values are calibrated.

---

## 5. Workers and requests

**Workers** connect over TCP and send `Register` with:

- role (`prefill`, `decode` or `colocated`);
- GPU index, fabric address, NICs per GPU and link speed;
- model name and KV layout;
- receive-arena size (decode workers);
- max concurrency.

A layout that doesn't match `[model]` gets `Shutdown`. Workers then send a
heartbeat (load and free arena bytes) at least every `heartbeat_ms / 2`.

```python
w = bonsai.Worker(role="decode", coordinator="10.0.0.1:7000", cuda_device=0, arena_bytes=8 << 30)
```

Size the decode arena for your largest context × concurrency, using the
`--check` table. For example, 4 concurrent BF16 handoffs at 32K need
4 × 896 MiB ≈ 3.5 GiB.

**Requests** carry already-tokenized prompts, so every configuration sees
identical input:

- `prompt_tokens`, `max_new_tokens`, `stop_tokens`;
- greedy sampling by default;
- optional `precision` / `chunk` overrides;
- a free-form `tag` (for example `"qasper/123"`) that is copied into the
  metrics record.

Lifecycle (what `bonsai-ctl` logs per handoff):

```text
Queued ─▶ Leasing ─▶ Transferring ─▶ Decoding ─▶ Done
  │  waiting for a      decode worker     KV written,     generation
  │  worker with room   leased a slot;    first token     finished
  │                     prefill writing   reported
  └── any state ─▶ Failed ─▶ (BF16 retry as a new handoff) or Rejected to the client
```

Admission is FIFO. If the request at the head of the queue doesn't fit (no
decode worker has enough free arena), the requests behind it wait too. This
keeps measured latency fair across configurations.

---

## 6. Failure handling

| Event | What the coordinator does |
|---|---|
| Decode worker can't lease a slot (`SlotDenied`) | Tries another decode worker (up to `max_lease_attempts`); the prefill worker keeps its KV meanwhile. Then aborts the prefill and re-queues. |
| Timeout in Leasing/Transferring (`handoff_ms`) or Decoding (`decode_ms`) | Sends `Abort` to both workers, then retries or rejects |
| Worker misses heartbeats (3×) or disconnects | All its handoffs fail and it is removed; it must re-register |
| Worker reports `HandoffFailed` | Same as a timeout |
| Retry | Only for quantized handoffs when `fallback_to_bf16 = true`: new handoff id, precision `bf16`, original submit time kept, so the retry's latency counts against the request |

On `Abort`, the decode worker disarms its completion counters and
quarantines the slot for a short time before reusing it, so a late write
from the failed sender can't corrupt the next handoff. Retries are visible in
the metrics: the report's `codec` is `bf16` for a request submitted with
another precision.

---

## 7. Metrics output

Each line of `[metrics].jsonl` is one finished request (`HandoffReport`):

| Field | Meaning |
|---|---|
| `tag`, `codec`, `seq_len`, `tokens_per_chunk`, `layers_per_chunk`, `concurrency` | Configuration and workload |
| `sender.wire_bytes`, `sender.metadata_bytes`, `sender.useful_bytes` | Transmitted bytes (incl. scales/zeros and padding), metadata alone, and payload + metadata without padding |
| `sender.encode_gpu_ns`, `sender.staging_wait_ns` | Encode cost, and time blocked on staging memory |
| `sender.handoff_send_ns` | Prefill done → last RDMA write completed |
| `receiver.decode_gpu_ns`, `receiver.decode_wall_ns`, `receiver.cache_build_ns` | Reconstruction cost |
| `receiver.slot_bytes`, `receiver.scratch_peak_bytes`, `sender.staging_peak_bytes` | Temporary buffer usage |
| `ttft_ns` | Submit → first token from prefill |
| `ttfdt_ns` | Submit → first token generated by the decode worker (includes the handoff) |
| `e2e_ns`, `receiver.decode_tokens`, `receiver.decode_ns` | End-to-end latency and decode throughput |

KV handoff latency = `sender.handoff_send_ns + receiver.decode_wall_ns + receiver.cache_build_ns`.

```python
import pandas as pd
df = pd.json_normalize([__import__("json").loads(l) for l in open("results/run.jsonl")])
df["handoff_ms"] = (df["sender.handoff_send_ns"] + df["receiver.decode_wall_ns"] + df["receiver.cache_build_ns"]) / 1e6
df.groupby(["codec", "seq_len"])[["handoff_ms", "sender.wire_bytes"]].median()
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `invalid configuration: unknown field ...` | Typo in a key; the message names the key and the valid alternatives |
| `unknown codec "..."` | Not a built-in name and not defined under `[codecs.*]` |
| `tokens_per_chunk N must be a multiple of 32` | You're using a codec with `axis = "token"`; use a multiple of its `group_size`, or `tokens = 0` |
| `precisions.X: layer range [a, b) invalid` | Ranges are half-open and must lie within `num_layers` |
| A worker connects and immediately gets `Shutdown` | Its model layout differs from `[model]` |
| Requests stay `Queued` | No decode worker has enough free arena for the slot: raise the worker's `arena_bytes`, lower concurrency, or use a smaller precision (check sizes with `--check`) |
| Many BF16 records in a quantized run | Handoffs are failing and being retried: look for timeouts (raise `handoff_ms` for very long contexts) and worker errors |
| Adaptive always picks the same candidate | Cost-model rates are uncalibrated, `min_gain` is too high, or the other candidates' `quality_loss` buckets don't cover your context lengths |
