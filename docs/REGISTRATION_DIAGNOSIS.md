# Large GPU memory registration failure: observed diagnosis

The first real two-node transfer pilot, job **1294349**, reached a 16K-context
BF16 transfer after local BF16/INT8/INT4 and receiver text-prefill predictions
succeeded. `FabricEndpoint.register_sink` then failed with `ibv_reg_mr`,
`EFAULT` (-14). The failed prediction is retained in
`results/pilot-transfer-1294349.jsonl`; sender and receiver traces are in
`logs/transfer-pilot-1294349.log` and
`logs/transfer-pilot-receiver-1294349.log`.

This was not evidence of unsupported GPU RDMA or a 2 GiB memory-region limit.
The inherited environment enabled `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The following controlled, real H200 experiments isolated allocation mapping and
alignment as the relevant condition.

|Job|Experiment|Observed result|
|---|---|---|
|1294356|Whole allocations: 16 MiB, 512 MiB, 1 GiB, immediately below/above 2 GiB, exact 2,395,865,088-byte cache size, 3 GiB; untouched versus zeroed and synchronized|All 14 registrations passed|
|1294359|16 MiB allocation; 512-byte pointer offset and/or 256-byte length trim; expandable allocator true/false; untouched/zeroed|All registrations passed|
|1294359|2,395,865,088-byte allocation with aligned pointer and length; expandable allocator true/false|All registrations passed|
|1294359|Same large allocation with 512-byte pointer offset and/or 256-byte trim; expandable allocator true|All six offset/trim cases failed with the original `ibv_reg_mr EFAULT`, regardless of initialization|
|1294359|Same large offset/trim allocation cases; expandable allocator false|All registrations passed|

Raw output is `logs/register-probe-1294356.log` and
`logs/register-alignment-1294359.log`. Reproduce with
`scripts/cluster/register_probe.sbatch` and
`scripts/cluster/register_alignment_probe.sbatch`. These jobs only register and
unregister GPU memory; they are not transfer or bandwidth measurements.
The alignment probe records both allocator aliases and CUDA module-loading
configuration. The first probe inherited the expandable allocator setting;
its raw output predates explicit environment logging.

The vendored fabric memory-region implementation attempts CUDA DMA-BUF export
and falls back to `ibv_reg_mr` when export fails. The controlled result is
consistent with large fragmented expandable mappings failing that fallback;
we did not instrument the driver to establish its internal rejection reason.
Zeroing the buffer does not resolve the reproduced failure. Fresh aligned
multi-GiB allocation registration succeeds.

The production environment now explicitly exports both `PYTORCH_ALLOC_CONF`
and `PYTORCH_CUDA_ALLOC_CONF` as `expandable_segments:False` before Python starts.
Runner and receiver log this environment before loading a model; the runner
also includes it in run provenance. Both entrypoints reject an explicitly
enabled expandable allocator for native fabric runs. The backend remains native fabric-lib GPU RDMA.

The corrected-allocator pilot 1294365 passed registration but stalled before
its first large-cache transfer result. A source audit found that upstream's
single-write operation casts the entire 64-bit transfer length into one
`ibv_sge` and submits one work request. The Bonsai adapter now divides each
logical write into requests of at most 256 MiB, preserving the full 64-bit
source/destination offsets. It awaits every native completion and puts the
receiver immediate only on the last request. Cancellation retains the source
registration lease in the pending native callback. The vendor is unchanged.
The native adapter and Python extension compile successfully. Real hardware verification then passed in job **1294370**: position-varying
payloads of 603,979,776 and 2,395,865,088 bytes were compared exactly on the
receiver GPU, including final immediate completion. The two single-write
logical-transfer goodputs were 314.30 and 367.35 Gbit/s; these remain functional
diagnostics, not repeated model-handoff performance results. Raw output is
`logs/endpoint-smoke-1294370.log`.

The corrected warmup/TCP_NODELAY real-cache pilot **1294371** completed all
63 predictions across nine prompts and seven conditions. The strict gate
verified all 36 local/remote generated-token-ID comparisons exactly. Both
registration and large-message fixes therefore passed real cache continuation.
The final 900-prompt experiment and repeated timing blocks remain separate
acceptance steps.

A strict continuation acceptance gate is available after the seven-condition
pilot completes:

```bash
../.venv/bin/python scripts/check_handoff_equivalence.py \
  results/pilot-transfer-JOBID.jsonl --expected-prompts 9
```

It requires all seven conditions for every prompt, matching sample hashes and
run fingerprints within each prompt, native transport labels, and exact output
token ID equality for BF16 transfer/local BF16, INT8 transfer/local INT8, INT4
transfer/local INT4, and text refill/local BF16. It rejects failures, duplicate
rows and incomplete condition sets. This integrity gate does not replace the
manifest/provenance checks or statistical quality analysis.
