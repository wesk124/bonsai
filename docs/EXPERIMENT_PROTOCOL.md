# Frozen core experiment protocol

This protocol implements `../annotated-kv_quantization_retrieval_proposal.pdf`.
It is a specification, not a claim that every experiment has been completed.

## Fixed inputs

- Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Identical BF16 model weights, tokenizer, attention implementation and software
  on both workers; one GPU per worker, batch size one, greedy decoding.
- RULER revision `db43f92ff634bbce1c33d1fb15858aec4bba509a`.
- Tasks: `niah_single_2`, `niah_multikey_1`, `niah_multiquery`.
- Nominal total token budgets 4096, 8192, 16384, with 128 generation tokens
  reserved. Record the actual rendered input length; never truncate a prompt.
- Final: 100 examples per task/budget, 900 unique prompts. Pilot: five per cell,
  with disjoint seeds and hashes. Final data must not be used to tune codecs.
- BF16, symmetric INT8 per head/token, asymmetric INT4 group32: K groups
  along tokens, V groups along channels, f16 scales and offsets. Physically
  pack INT4. Model weights and newly generated KV remain BF16.

## Required conditions

Local BF16; local INT8 and INT4 round trips; BF16, INT8 and INT4 transfers;
text transfer followed by receiver prefill. Only verified fabric-lib transfers
count as the proposal's 2700 transferred continuations. TCP and loopback are
diagnostics and must retain that label in data and figures.

The complete document and question are prefetched. The first output token is
selected from the sender's last-prompt logits and transferred to the receiver;
subsequent tokens attend to the reconstructed cache. The first token is therefore
unaffected by quantization. Every condition uses a fresh cache. Preserve output
IDs, raw predictions and per-answer recovery, including failed runs.

## Measurement boundaries

Encoding, transfer and reconstruction are sequential. Synchronize the producing
CUDA stream before NIC access; signal ready only after reconstruction and cache
wrapping complete. A sender clock measures cache-ready to receiver-ready ACK;
this includes the return ACK and is labeled accordingly. Do not subtract
timestamps from different hosts. CUDA kernel time and host wall time are distinct.

Count actual payload, quantization metadata, alignment padding, trailers and
control messages separately. Describe these as application bytes, not physical
link bytes unless packet counters were collected. Model loading and endpoint
initialization are startup costs. The Python runner currently registers buffers
per handoff, so registration, allocation, initialization and deregistration needed
before receiver readiness remain included in handoff time. A future persistent
registration mode must be labeled separately and report its startup costs.
Measure PyTorch allocated/reserved peaks and native staging/arena/scratch memory;
native allocations do not automatically appear in PyTorch memory statistics.

Warm up all conditions and randomize their order in repeated timing blocks.
Report sample counts, hardware topology, p50/p95, run-to-run uncertainty and
measured link goodput. Do not infer a handoff speedup from compression ratio.
Constrained-link tests are separate from native-link tests and must identify
whether the change is NIC selection, actual network configuration or sender pacing.

## Quality inference

Use upstream case-insensitive target-substring recovery, averaging targets within
each example, plus a separate all-targets-recovered indicator. Pair each quantized
result with the corresponding BF16 result in the same task and length. Keep
partial/missing/duplicate pairs visible and reject them for a complete-study claim.

The preservation margin is two percentage points: mean score difference
`quantized - BF16 >= -0.02`. Report paired confidence intervals and the exact
estimator. A point estimate within the margin is insufficient. Sparse/no-discordance
data must not produce an unjustified certainty claim from a degenerate bootstrap;
report a conservative bound or mark the comparison inconclusive. Do not tune on
the final set or add samples selectively after inspecting favorable results.

Record target positions and report quantization-induced losses separately from
BF16 model failures. Position analyses with few events are exploratory.

## Deliverables and scope

Reproducible source/environment pins, input manifests, raw predictions/timings,
failure records, integrity tests, analysis tables and PNG figures, and a report
that distinguishes observed measurements from hypotheses or calculated sizes.
Adaptive precision, pipelining, concurrency sweeps, other models and HotpotQA
remain optional until every required condition is validated and measured.
