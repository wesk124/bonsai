# Bonsai

Quantized KV-cache handoff for disaggregated LLM inference, built on
Perplexity's [fabric-lib](https://github.com/perplexityai/pplx-garden) (RDMA / GPUDirect).

- **Control plane** (`bonsai-ctl`): worker registry, placement, per-request
  precision choice (static or adaptive), handoff state machines, metrics.
- **Data plane** (inside each worker): encode the KV cache on the prefill GPU
  (BF16 / INT8 / packed INT4 / per-layer mixed), write it over RDMA into the
  decode GPU's receive arena, then decode it progressively and continue
  generation with HF Transformers.

Model: Qwen2.5-1.5B-Instruct. Evaluation: LongBench (QASPER, HotpotQA), RULER.

## Docs

- [High-level design](docs/HLD.md): architecture, request lifecycle, wire
  format, adaptive precision, measurement, two-node deployment
- [Low-level design](docs/LLD.md): Rust interfaces, crate by crate
- [Build guide](docs/BUILD.md): fabric-lib + Bonsai build, Docker image, host requirements for the RDMA nodes

## Status

Design and interface skeleton. `bonsai-core` (byte accounting, wire format /
chunk planner) and the coordinator state machine are implemented and tested.
GPU, RDMA and binding code is still `todo!()` and is tracked in the
[LLD checklist](docs/LLD.md#8-implementation-checklist-todo--done).

## Build

Full requirements and the Docker image are in [docs/BUILD.md](docs/BUILD.md).

```bash
docker build -t bonsai-dev -f docker/Dockerfile .  # on each RDMA node
scripts/run-docker.sh
cargo build && cargo test            # any OS: core + control plane
cargo build -p bonsai-dataplane --features fabric   # Linux RDMA node; see docs/BUILD.md
cd crates/bonsai-py && maturin develop   # Python module `bonsai`
```
