# Bonsai — Low-Level Design (Rust interfaces)

Read [HLD.md](HLD.md) first. This document walks through the Rust interfaces
crate by crate. The code under `crates/` is the source of truth: everything
quoted here compiles (`cargo build`), and the parts marked *implemented* are
covered by tests (`cargo test`, 12 tests).

## 0. Workspace layout

```text
bonsai/
├── Cargo.toml                 workspace; fabric-lib + cuda-lib pulled from pplx-garden by git
├── rust-toolchain.toml        1.91.0 (same as pplx-garden)
├── docs/{HLD,LLD}.md
└── crates/
    ├── bonsai-core/           ids, KvLayout, CodecSpec/PrecisionMap, wire format + ChunkPlan,
    │                          control messages, metrics          (no CUDA, no RDMA, no I/O)
    ├── bonsai-codec/          KvCodec trait, CodecSet, CUDA FFI (feature cuda), CPU reference
    │   └── kernels/           bonsai_codec.cuh (C ABI); .cu implementation lands in M2
    ├── bonsai-transport/      Transport trait; FabricTransport (feature fabric); LoopbackTransport
    ├── bonsai-dataplane/      StagingRing, ReceiveArena, KvSender, KvReceiver, Gpu trait, Throttle
    ├── bonsai-control/        Coordinator (sans-IO), registry, sessions, policies, sink, server,
    │   ├── src/bin/bonsai-ctl.rs
    │   └── tests/coordinator_flow.rs
    └── bonsai-py/             PyO3 module `bonsai` (excluded from workspace; built with maturin)
```

Dependency graph (arrows point to dependencies):

```text
bonsai-py ─▶ bonsai-control ─▶ bonsai-core
    │                              ▲
    └──────▶ bonsai-dataplane ─────┤
                 │    │            │
                 │    └▶ bonsai-codec ─────┘
                 └▶ bonsai-transport ─▶ bonsai-core
                         └▶ fabric-lib, cuda-lib   (feature "fabric", Linux only)
```

**Build features**

| Feature | Crate | Enables | Needs |
|---|---|---|---|
| *(none)* | all | Everything type-checks, core and control tests run | Any OS |
| `fabric` | transport, dataplane, py | `FabricTransport` over `fabric_lib::TransferEngine` | Linux, libfabric, libibverbs, CUDA ≥ 12.8, GDRCopy |
| `cuda` | codec, dataplane, py | `CudaCodec` + nvcc-built kernels | CUDA toolkit |

---

## 1. `bonsai-core`

### 1.1 Identifiers — `ids.rs`

```rust
pub struct RequestId(pub u64);
pub struct HandoffId(pub u64);          // 1 request : ≥1 handoffs (BF16 retry gets a new one)
pub struct WorkerId(pub u32);
pub enum   WorkerRole { Prefill, Decode, Colocated }
pub struct SlotId { pub index: u16, pub epoch: u8 }

/// RDMA immediate: | slot:12 | epoch:8 | group:12 |
pub struct ImmTag(pub u32);
impl ImmTag {
    pub fn new(slot: SlotId, group: u16) -> Self;
    pub fn slot(self) -> SlotId;
    pub fn group(self) -> u16;
}
```

This gives 4096 slots per decode worker and 4096 layer groups per handoff.
`epoch` wraps at 256; with quarantine, a stale write would need 256 reuses of
the same slot within the quarantine window to be miscounted.

### 1.2 Layout and codecs — `layout.rs`, `codec_spec.rs`

```rust
pub struct KvLayout { pub num_layers: u16, pub num_kv_heads: u16, pub head_dim: u16, pub dtype: ElemType }
impl KvLayout { pub const QWEN2_5_1_5B: KvLayout; /* 28 × 2 × 128, BF16 */ }

pub enum QuantBits { Int8, Int4Packed }
pub enum GroupAxis { HeadDim /* per-token */, Token /* per-channel, KIVI-style */ }
pub struct QuantParams { pub group_size: u16, pub axis: GroupAxis, pub symmetric: bool }

pub enum CodecSpec {
    Bf16,
    Quant { bits: QuantBits, key: QuantParams, value: QuantParams },
}
impl CodecSpec {
    pub fn int8_default(layout: &KvLayout) -> Self;   // sym, group = head_dim, per-token
    pub fn int4_default() -> Self;                    // asym g32; K per-channel, V per-token
    pub fn validate(&self, layout: &KvLayout) -> Result<()>;
    pub fn token_alignment(&self) -> u32;             // chunk token boundaries must align
    pub fn section_bytes(&self, l: &KvLayout, layers: u64, tokens: u64) -> (SectionBytes, SectionBytes);
    pub fn encoded_bytes(&self, l: &KvLayout, seq_len: u64) -> u64;
    pub fn name(&self) -> String;                     // e.g. "int4-kc32a-vt32a"
}

/// Adaptive precision, level (a): codec per layer range.
pub struct PrecisionMap { pub default: CodecSpec, pub overrides: Vec<LayerOverride> }
pub struct LayerOverride { pub layers: Range<u16>, pub codec: CodecSpec }
impl PrecisionMap {
    pub fn uniform(c: CodecSpec) -> Self;
    pub fn protect_layers(lo: CodecSpec, hi: CodecSpec, layers: &[Range<u16>]) -> Self;
    pub fn codec_for_layer(&self, layer: u16) -> CodecSpec;
    pub fn segments(&self, num_layers: u16) -> Vec<(Range<u16>, CodecSpec)>;  // maximal same-codec runs
    pub fn encoded_bytes(&self, l: &KvLayout, seq_len: u64) -> u64;
    pub fn token_alignment(&self, l: &KvLayout) -> u32;
    pub fn is_uniform(&self, c: CodecSpec) -> bool;
}
impl From<CodecSpec> for PrecisionMap { /* uniform */ }
```

*Implemented.* Byte accounting lives in core, not in the codec crate, because
the coordinator must size slots before any GPU work starts. Scales and zero
points are stored as f16.

### 1.3 Wire format — `wire.rs` (*implemented*)

```rust
pub const FRAME_ALIGN: u64 = 256;
pub const TRAILER_BYTES: u64 = 256;

pub struct ChunkSpec { pub tokens_per_chunk: u32, pub layers_per_chunk: u16 }   // 0 = all
pub struct Section { pub offset: u64, pub len: u64 }                             // frame-relative
pub struct FrameLayout { pub k_payload: Section, pub v_payload: Section, pub k_meta: Section, pub v_meta: Section }

pub struct ChunkDesc {
    pub index: u32,
    pub layers: Range<u16>,
    pub tokens: Range<u32>,
    pub frame_offset: u64,   // slot-relative
    pub frame_len: u64,      // RDMA write length (incl. padding)
    pub sections: FrameLayout,
    pub codec: CodecSpec,    // one codec per chunk
    pub group: u16,          // completion group (layer group)
}

pub struct ChunkPlan {
    pub layout: KvLayout, pub precision: PrecisionMap, pub seq_len: u32, pub spec: ChunkSpec,
    pub chunks: Vec<ChunkDesc>, pub trailer_offset: u64, pub slot_bytes: u64, pub num_groups: u16,
}
impl ChunkPlan {
    pub fn build(layout: KvLayout, precision: impl Into<PrecisionMap>, seq_len: u32, spec: ChunkSpec) -> Result<Self>;
    pub fn wire_bytes(&self) -> u64;
    pub fn writes_in_group(&self, g: u16) -> u32;   // + 1 for the trailer in the last group
    pub fn imm_tag(&self, slot: SlotId, g: u16) -> ImmTag;
    pub fn largest_frame(&self) -> u64;
}

pub struct HandoffTrailer { pub magic: u32, pub first_token: u32,
                            pub sender_encode_ns: u64, pub sender_transfer_ns: u64, pub checksum: u64 }
impl HandoffTrailer { pub fn to_bytes(&self) -> [u8; 256]; pub fn from_bytes(b: &[u8]) -> Result<Self>; }
```

**Planner algorithm**

1. `segments = precision.segments(L)`, so each segment has a single codec.
2. Split each segment into layer groups of `layers_per_chunk` layers. Groups
   never span two segments.
3. For each group (layer-major), split tokens into ranges of
   `tokens_per_chunk`. Each `(group, token range)` becomes one chunk.
4. Each chunk frame is `K payload | V payload | K meta | V meta`, with every
   section aligned to 256 B. Frames are packed back to back, followed by the
   trailer.

Invariants (tested): frames are contiguous and aligned; the sum of useful
bytes equals `encoded_bytes`; no chunk crosses a precision boundary;
misaligned chunk sizes are rejected.

Layout inside a section: payload is `[layer][head][token][channel]`,
metadata is `[layer][head][group]`. For INT4, element `2i` goes in the low
nibble and element `2i+1` in the high nibble.

### 1.4 Device handles — `device.rs`

```rust
pub struct DevicePtr(pub u64);        // tensor.data_ptr()
pub struct CudaStream(pub u64);       // torch.cuda.current_stream().cuda_stream
pub struct LayerKv { pub k: DevicePtr, pub v: DevicePtr, pub stride_head: u64, pub stride_token: u64, pub stride_dim: u64 }
pub struct KvSource { pub layout: KvLayout, pub seq_len: u32, pub device: u8, pub layers: Vec<LayerKv> }
pub type   KvSink = KvSource;
```

These are plain integers on purpose: they cross the PyO3 boundary without
pulling torch or CUDA types into core.

### 1.5 Control messages — `msg.rs`

```rust
pub struct FabricAddress(pub Bytes);  // serialized fabric_lib DomainAddress (opaque outside transport)
pub struct RemoteRegion(pub Bytes);   // serialized fabric_lib MemoryRegionDescriptor

pub struct HandoffParams { pub handoff: HandoffId, pub request: RequestId, pub layout: KvLayout,
                           pub precision: PrecisionMap, pub seq_len: u32, pub chunk: ChunkSpec }
impl HandoffParams { pub fn plan(&self) -> Result<ChunkPlan>; }

pub struct SlotGrant { pub handoff: HandoffId, pub decode_worker: WorkerId, pub decode_addr: FabricAddress,
                       pub slot: SlotId, pub region: RemoteRegion, pub base_offset: u64, pub len: u64 }

pub enum ToCoordinator {
    Register(WorkerInfo), Heartbeat { worker, load }, Deregister { worker },
    Submit(InferenceRequest),
    SlotGranted(SlotGrant), SlotDenied { handoff, reason },
    PrefillDone { handoff, first_token, timings: SenderTimings },
    Finished { handoff, output_tokens, report: HandoffReport },
    HandoffFailed { handoff, reason },
}
pub enum FromCoordinator {
    Registered { worker },
    LeaseSlot { params, request },   // → decode
    Prefill { params, request },     // → prefill
    SlotReady(SlotGrant),            // → prefill
    Abort { handoff, reason },       // → both
    Accepted { .. }, Completed { .. }, Rejected { .. },   // → client
    Shutdown,
}
pub mod framing { pub fn encode<T: Serialize>(m: &T) -> Result<Vec<u8>>; pub fn decode<'a, T>(b: &'a [u8]) -> Result<T>; }
```

Framing is a u32 little-endian length followed by a postcard body. Control
messages never carry KV bytes.

### 1.6 Metrics — `metrics.rs`

`SenderTimings`, `ReceiverTimings` and `HandoffReport` map one-to-one to the
proposal's metrics (HLD §7):

- `HandoffReport::handoff_latency_ns()` = sender `handoff_send_ns` + receiver
  `decode_wall_ns` + `cache_build_ns`.
- `decode_tokens_per_sec()` gives per-request decode throughput.

---

## 2. `bonsai-codec`

```rust
pub trait KvCodec: Send + Sync {
    fn spec(&self) -> CodecSpec;
    /// Enqueue on `stream`: read chunk.layers × chunk.tokens from `src`, write the frame at `dst`.
    fn encode_chunk(&self, src: &KvSource, chunk: &ChunkDesc, dst: DevicePtr, stream: CudaStream) -> Result<()>;
    /// Enqueue on `stream`: read the frame at `frame`, write BF16 into the chunk's rectangle of `dst`.
    fn decode_chunk(&self, frame: DevicePtr, chunk: &ChunkDesc, dst: &KvSink, stream: CudaStream) -> Result<()>;
    fn scratch_bytes(&self, chunk: &ChunkDesc) -> u64 { 0 }
}
pub enum Backend { Cuda }
pub fn make_codec(spec: CodecSpec, backend: Backend) -> Result<Box<dyn KvCodec>>;

/// One instance per distinct spec in a plan (mixed precision).
pub struct CodecSet { .. }
impl CodecSet {
    pub fn for_plan(plan: &ChunkPlan, b: Backend) -> Result<Self>;
    pub fn get(&self, chunk: &ChunkDesc) -> &dyn KvCodec;
}
```

**Numerics** are defined by `reference.rs`, the CPU oracle; the kernels must
match it bit for bit:

- Symmetric: `s = max|x| / qmax`, `q = round(x / s)`, clamped to `±qmax`.
- Asymmetric: `s = (max − min) / (2^b − 1)`, `z = min`.
- `s` and `z` are rounded to f16 *before* quantizing, so the encoder and
  decoder use the same values.
- Math is done in f32.
- `reference::compare` → `ErrorStats { max_abs, mse, sqnr_db }`.

**CUDA ABI** (`cuda.rs` ↔ `kernels/bonsai_codec.cuh`):

```c
int bonsai_encode_chunk(const BonsaiChunkArgs*, uint64_t stream);
int bonsai_decode_chunk(const BonsaiChunkArgs*, uint64_t stream);
```

`BonsaiChunkArgs` carries device arrays of per-layer K/V pointers, strides,
the chunk rectangle, section offsets, bits, and the per-tensor group size,
axis and symmetry. Kernel mapping:

- HeadDim groups: one warp per (layer, head, token) row; max/min via shuffle;
  single pass.
- Token groups: one thread per channel walking `group_size` tokens through
  shared memory; two passes.
- BF16: strided pack/unpack.

---

## 3. `bonsai-transport`

```rust
pub enum MemDevice { Host, Cuda(u8) }
pub struct WriteOp<L, R> { pub src: L, pub src_offset: u64, pub len: u64, pub dst: R, pub dst_offset: u64,
                           pub imm: Option<ImmTag>, pub shards: Option<NonZeroU8> }

pub trait Transport: Send + Sync + 'static {
    type Local: Clone + Send + Sync + 'static;     // fabric: MemoryRegionHandle
    type Remote: Clone + Send + Sync + 'static;    // fabric: Arc<MemoryRegionDescriptor>
    fn local_address(&self) -> FabricAddress;
    fn nets_per_gpu(&self) -> u8;
    fn link_bps(&self) -> u64;
    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<Self::Local>;
    fn register_sink(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<(Self::Local, RemoteRegion)>;
    fn unregister(&self, r: &Self::Local) -> Result<()>;
    fn resolve_remote(&self, r: &RemoteRegion) -> Result<Self::Remote>;
    fn write(&self, op: WriteOp<Self::Local, Self::Remote>) -> impl Future<Output = Result<()>> + Send;
    /// MUST be called before the peer can send writes with `tag` (fabric-lib resets on arm).
    fn arm_imm(&self, tag: ImmTag, count: NonZeroU32) -> Result<ImmWait>;
    fn disarm_imm(&self, tag: ImmTag);
}
pub struct ImmWait;   // impl Future<Output = Result<()>>; errors if disarmed
```

**`FabricTransport` (feature `fabric`)** maps directly onto fabric-lib:

- Construction: `detect_topology()` → pick the GPU's NICs →
  `TransferEngineBuilder::add_gpu_domains(dev, nics, worker_cpu, uvm_cpu)` →
  `build()`.
- `write` → `submit_transfer_async(TransferRequest::Single{ imm_data, domain: RoundRobinSharded{num_shards} })`.
- `arm_imm` → `set_imm_count_expected(tag, count, callback)`. The callback
  fires a oneshot and returns `Ok(false)` so the counter is one-shot.
  `AsyncTransferEngine::wait_for_imm_count` is deliberately **not** used: it
  arms only when first polled, which is too late.
- The tokio runtime must be multi-threaded, because fabric-lib uses
  `block_in_place`.

**`LoopbackTransport`** uses one `LoopbackFabric` shared by N endpoints.
Writes are `memcpy` into host memory with an optional `LinkModel`
(bandwidth + latency) delay. Immediates are counted with fabric-lib's
semantics, including silently dropping immediates that arrive before arming,
so ordering bugs show up in tests.

---

## 4. `bonsai-dataplane`

### 4.1 Config

```rust
pub enum StagingMode { Ring { depth: u32, frame_cap: u64 }, Whole { capacity: u64 } }
pub struct DataPlaneConfig {
    pub cuda_device: u8, pub staging: StagingMode, pub arena_bytes: u64,
    pub max_inflight_writes: u32,
    pub write_shards: Option<NonZeroU8>,  // Some(1) = single NIC
    pub throttle_bps: Option<u64>,         // emulated constrained link
    pub lease_timeout: Duration, pub quarantine: Duration, pub verify_checksums: bool,
}
```

### 4.2 GPU abstraction

```rust
pub trait Gpu: Send + Sync + 'static {
    type Event: Send + Sync + 'static;
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer>;
    fn free(&self, b: DeviceBuffer) -> Result<()>;
    fn upload(&self, dst: DevicePtr, src: &[u8], s: CudaStream) -> Result<()>;
    fn download(&self, src: DevicePtr, dst: &mut [u8], s: CudaStream) -> Result<()>;
    fn record(&self, s: CudaStream) -> Result<Self::Event>;
    fn completed(&self, ev: &Self::Event) -> impl Future<Output = Result<()>> + Send;  // polls cudaEventQuery
    fn elapsed_ns(&self, a: &Self::Event, b: &Self::Event) -> Result<u64>;
}
```

The real implementation wraps pplx-garden's `cuda-lib` (`event`, `mem`); a
fake implementation backs the tests.

### 4.3 Memory

```rust
pub struct StagingRing<T: Transport>;
impl<T: Transport> StagingRing<T> {
    pub fn new(t: &T, buf: DeviceBuffer, mode: StagingMode) -> Result<Self>;   // register_source once
    pub async fn acquire(&self, len: u64) -> Result<StagingLease>;             // backpressure point
    pub fn region(&self) -> &T::Local;
    pub fn footprint(&self) -> u64;
}
pub struct StagingLease { pub ptr: DevicePtr, pub offset: u64, pub len: u64, /* permit; freed on drop */ }

pub struct ReceiveArena<T: Transport>;
impl<T: Transport> ReceiveArena<T> {
    pub fn new(t: &T, buf: DeviceBuffer) -> Result<Self>;                      // register_sink once
    pub fn lease(&self, h: HandoffId, len: u64) -> Result<SlotLease>;          // bumps epoch; never blocks
    pub fn release(&self, slot: SlotId);
    pub fn release_quarantined(&self, slot: SlotId);
    pub fn remote(&self) -> &RemoteRegion;
    pub fn free_bytes(&self) -> u64;
}
```

The arena allocator is first-fit over a free list sorted by offset, with
256 B alignment and coalescing on release. The slot table holds
`(epoch, lease, quarantine_until)`.

### 4.4 Sender

```rust
pub struct SendJob {
    pub params: HandoffParams, pub source: KvSource, pub stream: CudaStream, pub first_token: u32,
    pub grant: oneshot::Receiver<SlotGrant>, pub prefill_done: Instant, pub prefill_ns: u64,
}
impl<T: Transport, G: Gpu> KvSender<T, G> {
    pub fn new(t: Arc<T>, g: Arc<G>, staging: Arc<StagingRing<T>>, cfg: DataPlaneConfig) -> Self;
    pub async fn send(&self, job: SendJob) -> Result<SenderTimings>;
}
```

`send` (written out in `sender.rs`):

1. `plan = params.plan()` and `codecs = CodecSet::for_plan`.
2. Await the grant. This is the only wait on the control plane. Check
   `grant.len ≥ plan.slot_bytes`.
3. For each chunk:
   - `staging.acquire(frame_len)`;
   - enqueue `encode_chunk`, then record an event;
   - spawn a task: wait for the event, `throttle.acquire`, then
     `transport.write(.., imm = ImmTag(slot, chunk.group))`, then drop the
     lease.
4. Join all write tasks. Spawning is required: polling writes inline while
   blocked in `acquire` would deadlock.
5. Upload the trailer to staging and write it with the last group's tag.
6. Return the timings.

### 4.5 Receiver

```rust
impl<T: Transport, G: Gpu> KvReceiver<T, G> {
    /// Lease a slot and arm every group counter; only then return the grant.
    pub fn prepare(&self, p: &HandoffParams) -> Result<PreparedHandoff>;   // .grant → coordinator
    /// Progressive: await group g → decode its chunks → next group. Timeout = lease_timeout.
    pub async fn receive(&self, p: PreparedHandoff, sink: &KvSink, s: CudaStream) -> Result<ReceivedHandoff>;
    pub fn abort(&self, p: PreparedHandoff);   // disarm + quarantine
}
pub struct ReceivedHandoff { pub trailer: HandoffTrailer, pub timings: ReceiverTimings }
```

On error or timeout, `receive` disarms all counters and quarantines the slot.
If arming fails partway through `prepare`, it rolls back the counters it
already armed and releases the slot.

### 4.6 Throttle

`Throttle::new(bps, burst)` with `async acquire(bytes)` is a token bucket
applied before each write. Combined with `write_shards = Some(1)` it creates
the constrained-bandwidth conditions for hypothesis 2.

### 4.7 Threading model

| Thread | Owner | Work |
|---|---|---|
| Python main thread | HF model | forward / generate; calls `bonsai-py`, which releases the GIL |
| tokio multi-thread runtime | `bonsai-py` | control connection, send/receive futures, write tasks |
| `tx_engine_callback` | fabric-lib | polls completion queues and runs callbacks (oneshot sends only) |
| fabric-lib workers (CPU-pinned) | fabric-lib | post work requests to the NIC |

The CUDA stream is torch's current stream, so encode is ordered after prefill
with no extra synchronization.

---

## 5. `bonsai-control`

### 5.1 Coordinator (sans-IO) — `coordinator.rs`

```rust
pub struct ConnId(pub u64);
pub struct Outbound { pub to: ConnId, pub msg: FromCoordinator }
pub struct CoordinatorConfig { layout, model, heartbeat_timeout, handoff_timeout, decode_timeout,
                               max_lease_attempts, fallback_to_bf16, max_retries, max_queue }
impl Coordinator {
    pub fn new(cfg, placement: Box<dyn PlacementPolicy>, codec: Box<dyn CodecPolicy>, sink: Box<dyn MetricsSink>) -> Self;
    pub fn on_message(&mut self, from: ConnId, msg: ToCoordinator, now: Instant) -> Vec<Outbound>;
    pub fn on_tick(&mut self, now: Instant) -> Vec<Outbound>;        // timeouts, heartbeats
    pub fn on_disconnect(&mut self, c: ConnId, now: Instant) -> Vec<Outbound>;
}
```

*Implemented*: happy path, BF16 fallback on failure, `SlotDenied` retry on
another decode worker and then requeue, timeouts, lost workers, and the
bandwidth EWMA fed by `Finished` reports. *Tested so far*: the happy path and
the BF16 fallback. Tests for denial, timeout and lost workers are still to be
written.

### 5.2 Sessions — `session.rs`

```rust
pub enum HandoffState { Queued, Leasing { prefill, decode, attempts }, Transferring { prefill, decode },
                        Decoding { prefill, decode }, Done, Failed { reason } }
impl SessionManager {
    pub fn create(&mut self, req, params, client: ConnId, now) -> HandoffId;
    pub fn transition(&mut self, id, next: HandoffState, now) -> Result<&mut Session>;  // rejects illegal edges
    pub fn queued(&self) -> impl Iterator<Item = HandoffId>;                             // FIFO
    pub fn active_on(&self, w: WorkerId) -> impl Iterator<Item = HandoffId>;
    pub fn finish(&mut self, id) -> Option<Session>;
}
```

Legal transitions:

- `Queued → Leasing`
- `Leasing → {Leasing, Queued, Transferring}`
- `Transferring → {Decoding, Done}` (`Finished` can arrive before
  `PrefillDone`, because they travel on different TCP connections)
- `Decoding → Done`
- any state → `Failed`

### 5.3 Policies — `policy.rs`

```rust
pub trait PlacementPolicy: Send {
    fn place(&mut self, req: &InferenceRequest, slot_bytes: u64, reg: &WorkerRegistry, exclude: &[WorkerId])
        -> Option<(WorkerId, WorkerId)>;
}
pub struct LeastLoaded;

pub struct PolicyContext { pub nominal_link_bps: u64, pub measured_link_bps: Option<u64>, pub inflight_handoffs: u32 }
pub trait CodecPolicy: Send {
    fn choose(&self, req: &InferenceRequest, layout: &KvLayout, ctx: &PolicyContext) -> (PrecisionMap, ChunkSpec);
}
pub struct StaticCodecPolicy { pub precision: PrecisionMap, pub chunk: ChunkSpec }

pub struct CostModel { pub bf16: CodecRates, pub int8: CodecRates, pub int4: CodecRates,
                       pub per_write_ns: f64, pub per_chunk_kernel_ns: f64, pub link_efficiency: f64 }
impl CostModel { pub fn estimate_ns(&self, plan: &ChunkPlan, link_bps: u64) -> f64; }

pub struct Candidate { pub precision: PrecisionMap, pub chunk: ChunkSpec, pub quality_loss_by_ctx: Vec<(u32, f64)> }
pub struct AdaptivePrecisionPolicy { pub candidates: Vec<Candidate>, pub model: CostModel,
                                     pub max_quality_loss: f64, pub min_gain: f64 }
pub struct BandwidthEstimator;   // EWMA of wire_bytes / handoff_send_ns
```

The cost model is:

```text
encode = Σ_chunks (raw_bytes / encode_bps + per_chunk_kernel_ns)
xfer   = wire_bytes · 8 / (bps · link_efficiency) + (chunks + 1) · per_write_ns
est    = max(encode, xfer) + min(encode, xfer) / chunks + decode(last group)
```

It is calibrated from codec microbenchmarks and `fabric-debug`.

Note: when first tested with made-up rates, the model predicted that INT4
beats BF16 even for a 256-token prompt at 400 Gb/s. Fixed per-chunk costs
only dominate for tiny prompts. Whether that holds on real GPUs is exactly
what calibration in M2/M3 must establish; don't assume BF16 wins on fast
links.

### 5.4 Server, client, sink

- `server::serve(ServerConfig, Coordinator)`: an accept loop, a framed reader
  and writer task per connection, and **one** task that owns the
  `Coordinator` and receives from an mpsc channel and a 50 ms tick. No locks
  around protocol state.
- `client::ControlClient`: `connect`, `send`, `recv`, and `split()` for
  workers that receive assignments while sending reports.
- `sink::MetricsSink`: `JsonlSink` (one `HandoffReport` per line) and
  `MemorySink` (tests).
- Binary: `bonsai-ctl --listen … --codec … --chunk-tokens … --chunk-layers … --metrics out.jsonl`.

---

## 6. `bonsai-py` (PyO3 module `bonsai`)

```python
w = bonsai.Worker(role="prefill"|"decode", coordinator="host:7000", cuda_device=0,
                  nets_per_gpu=None, arena_bytes=None, staging_depth=4, throttle_gbps=None)
job = w.next_job()                                   # Prefill / LeaseSlot (decode: already armed)
report = w.send_kv(job, out.past_key_values, first)  # prefill
cache, first = w.recv_kv(job)                         # decode → DynamicCache
w.finish(job, output_tokens)
cache2 = bonsai.roundtrip_kv(past_key_values, "int4-mixed")   # same-process quality eval
```

`send_kv` reads each `DynamicCache` layer's `data_ptr()` and `stride()`
(non-contiguous views are fine) to build a `KvSource`. `recv_kv` allocates
BF16 tensors through torch, so the PyTorch caching allocator owns that memory.

---

## 7. Testing strategy

| Level | What | Where | Status |
|---|---|---|---|
| Unit | Byte accounting, planner invariants, imm tag packing, mixed-precision segmentation | `bonsai-core` | ✅ 9 tests |
| Protocol | Coordinator happy path, BF16 fallback, adaptive policy regimes | `bonsai-control/tests` | ✅ 3 tests |
| Codec | CUDA kernels vs CPU reference (bit-exact), error stats per codec | `bonsai-codec` (feature `cuda`) | M2 |
| Pipeline | Sender + receiver over `LoopbackTransport` + fake `Gpu`: out-of-order completion, early-imm drop, abort and quarantine, timeout | `bonsai-dataplane` | M1–M2 |
| End-to-end | 2-node BF16 handoff: decode output tokens == colocated `generate` (greedy) | cluster | M1 |
| Quality | `roundtrip_kv` on LongBench QASPER/HotpotQA and RULER | single GPU | M2–M3 |

---

## 8. Implementation checklist (`todo!()` → done)

**M1**

- [ ] `FabricTransport`: register paths compiled on the cluster
- [ ] `Gpu` implementation on `cuda-lib`
- [ ] BF16 pack/unpack kernel
- [ ] `StagingRing` (Whole mode)
- [ ] `ReceiveArena`
- [ ] `server::serve` and `ControlClient`
- [ ] `bonsai-ctl` argument parsing
- [ ] `bonsai-py` `Worker`

**M2**

- [ ] CPU reference codec
- [ ] INT8 and INT4 kernels
- [ ] `StagingRing` (Ring mode)
- [ ] `LoopbackTransport` + pipeline tests
- [ ] `roundtrip_kv`

**M3**

- [ ] Layer-sensitivity sweep → default `PrecisionMap`
- [ ] Cost-model calibration
- [ ] `AdaptivePrecisionPolicy` candidates from dev-split results
- [ ] Load generator and analysis notebooks
