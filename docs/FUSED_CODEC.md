# Fused GPU cache codec

The optional `triton_packed_v1` backend reduces the work required to prepare and reconstruct a cache for handoff. It implements the same `bonsai-torch-packed-v1` packet format and quantization rules as the PyTorch reference. The default remains `pytorch_packed_v1`; selecting Triton is explicit and never silently falls back to another implementation.

## Measured pilot result

The completed paired RDMA experiment passed all 378 predictions, 216 local/remote token comparisons and 189 cross-implementation token comparisons. All pilot targets were recovered. Complete handoff medians for the fused implementation were:

| Context budget | BF16 | INT8 | INT4 |
| --- | ---: | ---: | ---: |
| 4K | 24.10 ms | 19.98 ms | 16.17 ms |
| 8K | 40.36 ms | 28.73 ms | 24.87 ms |
| 16K | 79.76 ms | 49.03 ms | 43.62 ms |

At 16K, the matched reference implementation measured 124.11 ms for INT8 and 144.28 ms for INT4. The median per-prompt/block INT4-to-fused-BF16 ratio was 0.5479, corresponding to 45.2% lower measured handoff latency. INT4 sender/receiver additional peak allocation at 16K fell from 1,429/5,322 MiB to 714/3,019 MiB; receiver figures include reconstructed cache residency.

These are descriptive pilot results: three unique prompts per context, each repeated in three blocks. They do not establish population latency or a new 900-prompt quality result. The [strict comparison report](../runs/fused-20261108/analysis/summary.md) includes all percentiles, exact paired comparisons and block variation; the [PNG figure](../runs/fused-20261108/analysis/paired-handoff.png) displays every paired transfer observation. The [versioned manuscript](PAPER_FUSED_DRAFT.md) combines these findings with the original study while preserving their separate evidence scopes.

## Implementation

Encoding allocates one final byte buffer. Each tensor kernel writes quantized values and FP16 metadata directly into its assigned section, avoiding intermediate float tensors, packing tensors and a final concatenation. Numerical validation accumulates into one device flag, read once per quantized cache. Raw BF16 encoding uses direct linear loads for contiguous inputs and retains a stride-aware path for other layouts.

Decoding writes each tensor into fresh storage. Fresh `DynamicLayer` objects adopt those tensors directly, avoiding the full-cache copy performed by `DynamicCache.update` and the `ddp_cache_data` constructor in the pinned Transformers version. The reconstructed cache does not alias the received frame or the sender cache. Continuation still makes its usual private cache copy after the receiver-ready acknowledgement; that common generation preparation remains outside the handoff timer.

The implementation preserves the reference's FP16 scale and offset rounding, ties-to-even quantization, edge padding and low-nibble-first packing. Scale calculation follows PyTorch's FP32 reciprocal multiplication for scalar division; quantization uses correctly rounded tensor division. Decode multiplication and addition remain separate to prevent fused multiply-add from changing reconstructed BF16 bits. Metadata uses byte loads and stores, including when FP16 values are unaligned.

Prompt lengths, strides, section offsets and group counts are runtime kernel arguments. Compile-time choices describe the codec and tensor layout, not individual prompts. The performance protocol warms every selected prompt and condition before recording measurements. The standalone codec benchmark additionally rejects any Triton specialization or cache-loading event during a measured interval.

## Use

The tested environment is PyTorch 2.8.0, Transformers 4.56.2 and Triton 3.4.0 on NVIDIA H200 GPUs. Source `scripts/native_env.sh` before using native RDMA so the validated allocator and native library settings are applied.

Both peers must select the same backend:

```bash
python -m bonsai_bench.worker --listen 0.0.0.0:7007 \
  --transport fabric-lib --codec-backend triton_packed_v1 --once

python -m bonsai_bench.runner --manifest data/ruler/pilot-smoke.jsonl \
  --out results/fused-pilot.jsonl --receiver RECEIVER_HOST:7007 \
  --transport fabric-lib --codec-backend triton_packed_v1 --warmup-samples 9 \
  --condition transfer:bf16 --condition transfer:int8 --condition transfer:int4
```

Backend identity is recorded in run provenance and result rows, checked during connection setup, and checked for every request. Resuming an output file with a different backend is rejected. CUDA is required for the fused codec.

## Validation and measurement scope

The measured kernel revision passed 43 H200 codec tests, including full packet equality, strided tensors, uneven groups, unaligned metadata, rounding boundaries, invalid inputs, detached ownership, cache length and subsequent cache updates. A subsequent live-code fix additionally checks the received buffer's base address before using typed loads, with a byte-load fallback for unaligned views. The live implementation passed 61 H200 codec tests, including 18 new offset-buffer cases. The final CPU suite passed 132 tests; 60 GPU-only cases were skipped there and executed in the H200 codec run.

Performance measurements use the immutable implementation in `runs/fused-20261108/source`, whose `fused.py` SHA256 starts `8c865a7b9fbb`. Its receive buffers come from aligned `torch.empty` allocations. The later alignment fix is archived separately under `runs/fused-20261108/robustness`; its broader buffer-view support was tested separately and is not included in the performance source hash.

The separate actual-model benchmark uses distinct caches produced by the pinned Qwen3-4B model on nine pilot prompts, one per task/context cell. It checks complete packet bytes and canonical headers for BF16, INT8 and INT4, all four combinations of encoder and decoder, and raw BF16 against the original cache. Its codec timings include dispatch, allocation, validation and synchronization, and return the production `DynamicCache` representation.

The paired RDMA pilot uses three fresh process blocks, both implementations, the same nine prompts and seven conditions. Backend order is shuffled per block, and all nine prompts are warmed before measurement. The first block ran alone; the remaining two ran concurrently on disjoint physical node pairs. Nodes were not reserved exclusively from other cluster work. Scheduling changes are recorded in `submission.json`, and analysis retains each block separately. Source/sink memory registration remains per request, so this comparison measures codec and cache construction changes without introducing a registration-pool optimization. Full handoff time runs from sender cache readiness through the receiver-ready acknowledgement.

These repeated pilot observations do not increase the number of unique quality prompts. The completed 900-prompt quality experiment and its original latency measurements remain unchanged in `runs/final-20261107` and `results/analysis-final`. New performance evidence is kept separately in `runs/fused-20261108`.

## Reproduce the comparison

The frozen source and input bytes are verified before each paired block. With an appropriate two-node H200 allocation configuration:

```bash
BONSAI_RUN_DIR=/absolute/path/to/a/new/frozen/run \
  sbatch scripts/optimization_pair.sbatch

python scripts/summarize_optimization.py /absolute/path/to/the/frozen/run
python scripts/plot_optimization.py /absolute/path/to/the/frozen/run
```

Analysis requires all six complete result files, matching source and prompt provenance, no failed rows, exact local/remote continuation equality, and exact cross-backend generated-token equality. It reports matched prompt/block ratios, descriptive percentiles and between-block ranges. The figure is a PNG; its whiskers are empirical percentiles, not confidence intervals.
