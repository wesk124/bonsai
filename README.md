# Bonsai

Bonsai evaluates quantized KV-cache handoff between a prefill GPU and a decode
GPU using real fabric-lib / GPUDirect RDMA. The frozen core experiment uses
**Qwen/Qwen3-4B-Instruct-2507**, three official RULER retrieval tasks, and
4K/8K/16K context budgets. Model weights and newly generated KV remain BF16.

The benchmark supports seven conditions: local BF16; local INT8/packed INT4
reconstruction; BF16/INT8/INT4 GPU transfers; and text transfer with receiver
prefill. The sequential benchmark uses the auditable PyTorch packed codec
(`pytorch_packed_v1`) and the native Rust `FabricEndpoint` for GPU payloads.
Native CUDA codecs and the coordinator/dataplane implementation have separate
tests; their performance is not substituted for benchmark measurements.

## Completed core experiment

- **900 final prompts, 6,300 successful predictions, 2,700 native transfers**, and
  **3,600 exact local/remote token-ID comparisons** passed the full audit.
- All seven conditions achieved **100% target recovery and all-answer recovery**.
  The two-percentage-point preservation margin remains **inconclusive** under
  conservative simultaneous bounds (−5.72 points per 100-prompt comparison).
- INT8 and INT4 reduced actual application bytes by **49.22% and 68.71%**.
  With the measured PyTorch codec on H200/400 Gbit/s fabric, BF16 transfer was
  faster: at 16K, p50 was **81.89 ms BF16**, **121.88 ms INT8**, **144.51 ms INT4**,
  and **547.95 ms text refill**. Compression did not provide a handoff speedup.
- Three separate timing blocks passed. Actual model-cache packets, metadata and
  reconstructed BF16 caches matched SHA256 across two GPUs for all formats.
- Validation passed **93 Python tests, 52 portable Rust tests and 3 CUDA tests**.

Read the [paper draft](docs/PAPER_DRAFT.md),
[measured tables](results/analysis-final/campaign/campaign.md),
[paired analysis and PNG figures](results/analysis-final/summary.md),
[acceptance manifest](results/analysis-final/artifact_manifest.json), and
[ten-workstream execution plan](docs/EXECUTION_PLAN.md).
The immutable campaign is `runs/final-20261107`; raw failures from earlier
preflight attempts remain separate in the diagnostic record.

## Reproduce

Use the project Python environment `../.venv`, Python 3.11, pinned PyTorch 2.8.0
and Transformers 4.56.2. [Cluster setup](docs/CLUSTER.md) describes native CUDA,
RDMA dependencies, the exact fabric source revision and verified hardware.
Install `bench[analysis]` for reporting; the final figure clarification uses
Seaborn 0.13.2. Inference environment pins remain in the frozen run, with the
analysis-only addition recorded in the accepted artifact bundle.

```bash
source scripts/native_env.sh
PYTHONPATH=bench ../.venv/bin/python -m pytest tests bench/tests -q
cargo +1.91.0 test --workspace
sbatch scripts/cluster/endpoint_smoke.sbatch
sbatch scripts/pilot_transfer.sbatch
# After the job completes, substitute its actual job ID:
../.venv/bin/python scripts/check_handoff_equivalence.py \
  results/pilot-transfer-JOBID.jsonl --expected-prompts 9
```

`native_env.sh` explicitly sets both PyTorch allocator environment aliases to
`expandable_segments:False` before Python starts. Both workers log these settings;
transfer runs reject the known incompatible setting and never fall back to host
staging or TCP. TCP payloads are available only as labeled diagnostics.

After the pilot passes, freeze the actual source and installed native extension
and run the fixed final matrix:

```bash
../.venv/bin/python scripts/freeze_run.py --native --out runs/final-YYYYMMDD
../.venv/bin/python runs/final-YYYYMMDD/source/scripts/verify_frozen_run.py runs/final-YYYYMMDD
BONSAI_RUN_DIR="$PWD/runs/final-YYYYMMDD" sbatch runs/final-YYYYMMDD/source/scripts/final_matrix.sbatch
PYTHONPATH=bench ../.venv/bin/python -m bonsai_bench.analysis \
  runs/final-YYYYMMDD/results/*.jsonl \
  --manifest runs/final-YYYYMMDD/data/ruler/final/*/niah*.jsonl \
  --require-complete-final --out results/analysis-final
```

The matrix has nine task/context cells, seven conditions and 100 prompts per
cell: **6,300 predictions, including 2,700 native transferred continuations**.
Failures remain in JSONL; missing cells cannot satisfy the complete-final check.
All generated figures default to PNG.

## Documentation

- [Execution plan and completed acceptance gates](docs/EXECUTION_PLAN.md)

- [Frozen experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Paired quality analysis, uncertainty and measurement scopes](docs/ANALYSIS.md)
- [Current paper draft and limitations](docs/PAPER_DRAFT.md)
- [Native cluster build and observed validation jobs](docs/CLUSTER.md)
- [High-level architecture](docs/HLD.md), [Rust interfaces](docs/LLD.md),
  [build guide](docs/BUILD.md), [control plane](docs/CONTROL_PLANE.md)

Architecture documents include optional designs such as adaptive precision and
pipelining. These features are outside the frozen sequential core experiment;
only explicitly recorded execution evidence establishes that a path ran.
