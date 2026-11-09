# Quantized KV-cache handoff for retrieval: quality controls and measured handoff costs

**Manuscript status:** the complete core experiment passed its acceptance audit:
900 final prompts, 6,300 successful predictions, 2,700 native cache transfers,
3,600 exact handoff comparisons, and three complete timing blocks. The
[accepted analysis](../results/analysis-final/campaign/campaign.md) and
[artifact manifest](../results/analysis-final/artifact_manifest.json) contain
measured results and provenance. This is a report draft for review.

## Abstract

Disaggregated inference transfers a prompt's KV cache from a prefill worker to
a decode worker. Quantization reduces bytes but adds encoding and reconstruction
work. We evaluate Qwen3-4B-Instruct-2507 on three RULER retrieval tasks at
approximately 4K, 8K and 16K context budgets, using seven conditions spanning
local inference, local reconstruction, real GPU RDMA transfer and text refill.
All 6,300 predictions on 900 final prompts achieved 100% target recovery and
all-answer recovery. Every same-precision local/remote comparison matched
exactly. INT8 and packed INT4 reduced application handoff bytes by 49.22% and
68.71% on average versus BF16. However, both increased measured handoff latency
on H200 GPUs with 400 Gbit/s NICs: at 16K, median BF16/INT8/INT4 handoffs took
81.89/121.88/144.51 ms, versus 547.95 ms for text refill. Three separately
reported timing blocks supported the same ordering. These results characterize
a reference PyTorch GPU codec with native fabric-lib transport. They show
communication savings without an observed retrieval-score loss, but do not
establish the proposed two-percentage-point population preservation margin:
the simultaneous conservative lower bound is −5.72 percentage points per
100-prompt comparison. Compression reduced bytes and incremental handoff memory;
uncompressed transfer delivered the lowest handoff latency in this experiment.

## 1. Introduction

Separating prefill and decode can improve resource allocation in LLM serving,
but it also introduces a cache handoff. A lower precision representation
reduces the bytes transferred while adding encoding, metadata and reconstruction
work. The relevant question is whether the complete handoff becomes faster
while preserving the downstream task, rather than whether its payload shrinks.

We study retrieval because cache errors may affect access to facts distributed
through a long input. Matched local round trips isolate reconstruction effects;
remote continuation of the same representation tests the transfer path; and
sending the rendered input for receiver prefill measures recomputation. This
design separates quality, communication size and measured handoff cost.

fabric-lib provides one-sided GPU communication with immediate completion
notifications for disaggregated LLM systems [1]. RULER supplies controlled
retrieval tasks with configurable context length and needle complexity [2].
KIVI motivates different quantization axes for keys and values [3]; our group-32
INT4 handoff implementation does not reproduce its complete two-bit inference
system. We use the official Qwen3-4B-Instruct-2507 model and tokenizer [4].
The contribution is a reproducible characterization with explicit correctness
controls and measurement boundaries. We do not claim a new quantization
algorithm or an optimized production serving engine.

## 2. Method

### 2.1 Model and continuation protocol

The model is Qwen/Qwen3-4B-Instruct-2507, pinned at
`cdbee75f17c01a7cc42f958dc650907174af0554`. Both workers use identical BF16
weights and tokenizer, SDPA attention, batch size one and greedy decoding.
Newly generated KV remains BF16.

The sender prefills the complete document and question, then selects the first
output token from its final prompt logits. The receiver receives that token
choice and continues from the reconstructed cache. Every condition uses a
fresh cache. Only tokens after the first output token attend to reconstructed
KV; quantization cannot change the first-token choice in this protocol. This
boundary is shared across conditions and limits conclusions about full-output
sensitivity to cache quantization.

We evaluate seven conditions: local BF16; local INT8 and INT4 reconstruction;
BF16, INT8 and INT4 native transfers; and text transfer followed by receiver
prefill. A transferred representation is compared with its matching local
condition for exact token-ID integrity. Quality comparisons pair quantized
results with BF16 within the same task, context, run and transport.

### 2.2 Cache representation and transport

INT8 is symmetric per-head/token quantization. INT4 is asymmetric group-32
quantization, grouping K along tokens and V along channels, with FP16 scales
and offsets. Two four-bit values are physically packed into each byte.
Quantization parameters, partial-group padding and representation metadata are
included in the actual encoded-size accounting.

For sequence length S, the model's raw BF16 cache uses
`2 × 36 × 8 × S × 128 × 2 = 147,456 × S` bytes. Ignoring the JSON header and
partial-group padding, INT8 uses 74,880 bytes/token, or 50.78125% of BF16;
INT4 uses 46,080 bytes/token, or 31.25%. These are format calculations, not
observed network measurements. INT4 quantization parameters prevent a fourfold
reduction even before application control overhead is counted.

The measured backend is the PyTorch packed reference codec,
`pytorch_packed_v1`, coupled to the native Rust `FabricEndpoint` for GPU RDMA.
The separate native CUDA codec and coordinator/dataplane implementations are
not substituted for this backend's measurements. Encoding, transfer and
reconstruction run sequentially.

The transport uses pinned fabric-lib source with a narrowly documented NIC
discovery export. Bonsai selects an active NIC near the assigned GPU using
PCI ancestry, NUMA and scheduler CPU affinity. Logical writes are divided into
requests of at most 256 MiB; readiness is signaled only after the final request.
Both PyTorch allocator environment aliases explicitly disable expandable
segments, the setting validated for large fragmented registrations on this
fabric. Source provenance and the vendor change are documented in
`crates/bonsai-transport/native-fabric/UPSTREAM.md` and
[the transport diagnosis](REGISTRATION_DIAGNOSIS.md).

### 2.3 Latency, bytes and memory

A sender monotonic clock spans cache-ready through receipt of the
receiver-ready ACK. The interval includes encoding, application control
framing, dynamic memory registration, GPU transfer, reconstruction,
synchronization and the return ACK. Model loading and endpoint construction
are startup work. Readiness means the reconstructed DynamicCache is available
for model execution. The continuation helper then clones that cache to preserve
the supplied state before generating tokens; this common clone and generation
are outside the handoff interval. Thus this metric measures cache readiness
rather than time to the first receiver decoding step. Receiver continuation
starts after readiness, and prediction
response bytes are recorded separately. No timing subtraction crosses hosts.
Per-phase encode/decode timers synchronize CUDA around host wall-clock
measurements; they are not CUDA-event kernel-only timings.

The `wire_bytes` field counts application payload and control frames, including
JSON metadata and framing. It excludes physical network headers and unobserved
fabric-provider connection traffic. Local round-trip bytes describe an encoded
representation and are not transmitted traffic. Registration borrows existing
GPU storage; registered bytes do not represent a second independent allocation.

Memory measurements retain PyTorch allocated/reserved statistics, measured
handoff baselines, peak allocation above those baselines, and native registered
bytes. They distinguish handoff peaks from post-generation memory. Model/cache
residency and allocator reserve must not be presented as exclusive codec
scratch, and unobserved driver/provider allocations are not inferred from
PyTorch statistics.

## 3. Evaluation design

### 3.1 Retrieval inputs

RULER is pinned at `db43f92ff634bbce1c33d1fb15858aec4bba509a`.
`niah_single_2` requests one numeric value associated with one word key;
`niah_multikey_1` requests one value among four key/value facts; and
`niah_multiquery` requests all four values. All use the official essay haystack.
This three-task subset does not represent the full RULER benchmark.

Nominal budgets are 4,096, 8,192 and 16,384 tokens, reserving up to 128 generation
tokens. The generator retains complete rendered inputs without truncation;
nominal lengths are budgets, not exact padded lengths. Final-manifest input
lengths, including the chat prompt and answer prefix, are:

| Nominal budget | Minimum | Mean | Maximum |
|---:|---:|---:|---:|
| 4,096 | 3,324 | 3,508.1 | 3,960 |
| 8,192 | 7,464 | 7,717.6 | 8,056 |
| 16,384 | 15,651 | 15,942.7 | 16,248 |

The pilot contains five prompts per task/context cell, 45 total. Final inputs
contain 100 per cell, 900 total, with disjoint pilot/final seeds and hashes.
The completed seven-condition final matrix has 6,300 successful predictions,
including 2,700 native transferred continuations. Final inputs were not used to
tune codecs or add samples selectively after observing quality.

### 3.2 Hardware, scheduling and timing repetitions

Experiments use two workers, each assigned one H200 GPU and one selected
400 Gbit/s fabric NIC. Scheduler CPU affinity and GPU/NIC topology are logged.
The nodes are shared infrastructure, not isolated hardware. The final campaign
uses nine task/context array cells; its scheduler throttle was raised from
three to six concurrent cells without changing code, inputs or conditions.
That change is recorded in `runs/final-20261107/submission.json` and may affect
resource contention. It is not a serving-concurrency experiment.

Condition order is deterministically shuffled per prompt. Warmup settings and
source identities are retained in provenance. A separate timing campaign uses
three fresh two-worker process launches on the same node pair. Each block
warms up on nine pilot prompts covering all task/context cells, then measures
four remote conditions on those same nine prompts: 36 rows per block and 108
total. Blocks remain separate from final quality data. They repeat processes
and prompts, not independent hardware samples; three-block ranges are
descriptive and are not confidence intervals.

### 3.3 Quality and statistical acceptance

The primary metric is official case-insensitive target-substring recovery,
averaged over requested targets within each prompt. All-targets-recovered is
a separate indicator. Partial multiquery recovery remains fractional. Failures,
missing conditions and unmatched pairs remain visible and are not treated as
correct predictions. Local and native-transfer comparisons are separate strata.

Paired percentile bootstrap intervals resample whole prompts, preserving the
dependence of targets within a prompt: 10,000 resamples, seed 20261107. With no
discordance, this bootstrap can collapse to zero width; that is not proof of
a two-percentage-point preservation margin. We also compute a one-sided exact
upper bound on the probability of a degraded prompt. Since each prompt's loss
is at most one, its negative is a conservative lower bound on mean score
difference. Bonferroni-adjusted bounds support simultaneous primary-score
comparisons across the 18 task/context/precision cells within each local or
native-transfer family, separately; this is not a joint 95% guarantee over both
families. The documented interval rules
and complete native final data are required for preservation acceptance.

With 100 zero-loss pairs, the unadjusted 95% bound alone permits approximately
2.95 percentage points of mean loss. The fixed per-cell design can therefore
remain inconclusive for a two-point margin even when observed scores match.
Position analyses are descriptive when loss events are sparse. Full estimator,
denominator and completeness rules appear in [the analysis documentation](ANALYSIS.md).

## 4. Results

### 4.1 Complete final results

All nine array cells completed successfully. The acceptance audit verified all
900 frozen prompts, all 6,300 predictions with zero execution failures, 2,700
native transferred continuations, and 3,600 exact generated-token-ID pairs.
Each of the 63 task/context/condition cells contains exactly 100 predictions.
The native binary, source and dataset hashes passed their final verification.

**Retrieval.** Every condition achieved 100% target recovery and 100%
all-answer recovery, in every task/context cell. This includes local BF16,
local INT8/INT4 reconstruction, BF16/INT8/INT4 transfer, and text refill.
There were no quantization-induced target losses or restored targets relative
to BF16. The paired score difference is 0.00 percentage points in all 18 native
comparisons and all 18 separate local comparisons.

Each 100-prompt comparison has an empirical paired bootstrap interval of
[0.00, 0.00] percentage points because every observed difference is zero.
The one-sided conservative lower bound is −2.95 points without multiplicity
adjustment and −5.72 points after the within-family 18-comparison adjustment.
Consequently, **the two-percentage-point preservation margin is inconclusive**
for every primary comparison. The zero-width bootstrap reflects the observed
ceiling; it does not override the conservative acceptance rule.

**Complete handoff latency.** The table pools the three tasks at each budget,
with 300 paired prompts per condition. Values are measured p50 / p95 in ms.
They describe within-campaign prompt variation, not confidence intervals over
independent hardware replications.

| Context | BF16 transfer | INT8 transfer | INT4 transfer | Text + prefill |
|---:|---:|---:|---:|---:|
| 4K | 24.39 / 40.39 | 54.84 / 83.10 | 72.49 / 97.49 | 94.46 / 110.20 |
| 8K | 43.65 / 55.75 | 70.96 / 88.48 | 91.39 / 109.91 | 220.09 / 238.02 |
| 16K | 81.89 / 102.01 | 121.88 / 148.66 | 144.51 / 171.04 | 547.95 / 573.90 |

Median per-prompt candidate/BF16 latency ratios for INT8 were 2.269, 1.658
and 1.496 at 4K/8K/16K; INT4 ratios were 3.024, 2.123 and 1.776. These are
medians of paired ratios, not ratios of aggregate medians. BF16 had the lowest
pooled p50 and p95 at each context. All cache formats had lower pooled p50
and p95 than text refill.

**Communication.** Mean application handoff bytes, including framing,
metadata and control exchanges up to readiness ACK, are shown in MiB.
Physical network headers and post-ACK predictions are excluded.

| Context | BF16 transfer | INT8 transfer | INT4 transfer | Text + prefill |
|---:|---:|---:|---:|---:|
| 4K | 493.34 | 250.54 | 154.59 | 0.02 |
| 8K | 1,085.30 | 551.14 | 339.45 | 0.04 |
| 16K | 2,241.95 | 1,138.51 | 700.93 | 0.07 |

Averaging the per-prompt byte reduction over all 900 prompts gives 49.22%
for INT8 and 68.71% for INT4. Text refill sends far fewer bytes but incurs
receiver prefill compute. Fewer bytes alone did not produce lower handoff
latency for either quantized cache format in this implementation.

**Handoff memory.** At the 16K budget, median incremental PyTorch allocation
peaks above each worker's handoff baseline were:

| Condition | Sender peak increment (MiB) | Receiver peak increment (MiB) |
|---|---:|---:|
| BF16 transfer | 4,408.31 | 6,612.47 |
| INT8 transfer | 2,268.19 | 5,527.61 |
| INT4 transfer | 1,378.73 | 5,097.24 |
| Text + prefill | 0.00 | 3,390.90 |

These increments include encoded buffers and reconstructed-cache residency;
they are not pure codec workspace. Sender and receiver baselines differ as
defined in Section 2.3. Post-readiness continuation cloning/generation and
non-PyTorch allocations are outside this metric. Full 4K/8K/16K memory and
phase p50/p95 tables are retained in the accepted JSON and campaign summary.

**Fact positions and output limits.** For each quantized format, the position
analysis contains 610 beginning, 612 middle and 578 end targets. No additional
target failures occurred in any third. With no loss events, this cannot establish
that position has no effect. Targets within a prompt remain correlated.
BF16 returned 128 tokens in 8 of 900 native continuations; INT8 and INT4 did so
in 10 each. All requested targets were recovered, but target recovery does not
measure all aspects of the generated response.

The detailed per-task tables, all uncertainty estimates and measurement counts
are in the [complete analysis](../results/analysis-final/summary.md) and
[campaign tables](../results/analysis-final/campaign/campaign.md). Main PNGs:

- [Quality versus context](../results/analysis-final/00-final-fabric-lib/quality-vs-context.png): all curves coincide at 100%.
- [Quality versus application bytes](../results/analysis-final/00-final-fabric-lib/quality-vs-bytes.png): text refill is labeled separately from BF16 cache transfer.
- [Complete handoff latency](../results/analysis-final/00-final-fabric-lib/latency-vs-context.png): solid p50 and dashed p95, 100 prompts per task/context/precision cell.
- [Paired differences and conservative bounds](../results/analysis-final/00-final-fabric-lib/paired-differences.png): empirical zero-width intervals do not establish the two-point margin.
- [Codec phase medians](../results/analysis-final/00-final-fabric-lib/phase-breakdown.png): stacked phase medians are not a total-latency percentile.
- [Added failures versus fact position](../results/analysis-final/00-final-fabric-lib/failure-vs-position.png): descriptive correlated-target counts, with no observed added failures.

### 4.2 Local pilot and native integrity

The local pilot contains 135 successful predictions on 45 prompts, with 45
predictions each for BF16, INT8 and INT4. Target recovery and all-answer rate
are 100% in all three conditions, and every paired score-difference point
estimate is zero. This records no observed pilot loss and functional local
continuation; it does not establish the two-point preservation margin. The
pilot predates complete frozen-source provenance and remains exploratory.
Earlier failed pilot attempts are retained separately rather than pooled with
successful predictions.

Native endpoints verified position-varying GPU payloads exactly at
603,979,776 and 2,395,865,088 bytes in job 1294370, including immediate
completion. Single logical-transfer goodputs were 314.30 and 367.35 Gbit/s.
These are functional diagnostics, not repeated model-handoff performance
estimates. The corrected seven-condition continuation pilot, job 1294371,
completed 63 predictions over nine prompts. All 36 exact generated-token-ID
comparisons passed: BF16 transfer/local BF16, INT8 transfer/local INT8, INT4
transfer/local INT4, and text refill/local BF16.

An additional untimed actual-model diagnostic, job 1294826, transferred the
16,248-token Qwen cache in all three formats. Full SHA256 hashes of each GPU
packet, its metadata, and its reconstructed BF16 K/V contents matched between
workers. BF16, INT8 and INT4 payloads were respectively 2,395,865,088,
1,216,650,240 and 748,892,160 bytes. The BF16 reconstructed hash also equaled
the original packet hash. Host copies used for hashing belong only to this
diagnostic; the measured benchmark keeps cache payload transfer on GPU RDMA.
The complete hashes and logs are archived under
`runs/final-20261107/integrity/`.

### 4.3 Repeated handoff measurements

Jobs `1294382_0` through `1294382_2` completed all three timing blocks. All 108
measured rows passed the timing completeness/provenance gate. For each context,
the range of the three block medians was:

| Context | BF16 transfer (ms) | INT8 transfer (ms) | INT4 transfer (ms) | Text refill (ms) |
|---:|---:|---:|---:|---:|
| 4K | 24.46–24.71 | 55.33–78.32 | 73.59–76.52 | 110.13–110.37 |
| 8K | 43.54–56.28 | 70.78–73.00 | 90.02–90.62 | 236.79–237.60 |
| 16K | 81.26–91.84 | 123.23–123.64 | 141.18–141.70 | 571.95–576.55 |

Each block/context median uses three distinct pilot prompts. Quantized handoff
had higher median latency than BF16 transfer in every block/context; each
block's median paired quantized/BF16 latency ratio also exceeded one. All three
cache-transfer conditions had lower block medians than text refill. These
observations demonstrate that fewer encoded bytes did not yield faster complete
handoff in this measured reference implementation on this fabric. They do not
establish population speedups, isolate the cost of every implementation choice,
or predict results for an optimized native-codec serving path.

The ranges are not confidence intervals. The completed
[timing summary](../results/analysis-timing-final/campaign.md) retains each
block's p95, per-prompt paired latency ratios and differences, phase measurements,
application bytes, memory fields and hashes. Ratios are computed per paired
prompt before summarization; ratios of aggregate quantiles are not substituted.

## 5. Discussion and limitations

The complete final campaign and repeated timing blocks show the central
distinction between compression and handoff latency: the measured total includes work beyond moving bytes.
They support reporting actual end-to-end costs alongside encoded sizes.
Attribution of the observed ordering to a specific kernel or optimization would
require additional controlled measurements. A faster native codec, different
fabric bandwidth, pipelining or persistent registration could change the tradeoff;
none is inferred from the present results.

External validity is limited to one model, three synthetic retrieval tasks,
sequential batch-one execution, H200 GPUs and the selected 400 Gbit/s NICs.
Shared-node contention and changing array concurrency limit hardware-isolation
claims. Timing repetitions reuse the same nine prompts and node pair. The
reference codec is not an optimized serving backend, and no constrained-link
or multi-request serving-concurrency study is included.

Quality conclusions are limited by ceiling effects, sparse discordance and the
fixed sample size. Target-substring recovery does not fully measure fluency,
unsupported answer text or broader reasoning. The sender-selected first output
token bypasses reconstructed-cache error at that boundary. Correlated targets
within a prompt are not independent samples. A zero-width empirical bootstrap
or a favorable point estimate cannot establish the preservation margin.
Adaptive precision, additional models and the full RULER suite remain outside
the required core experiment.

## 6. Reproducibility and artifact status

The final campaign snapshot is `runs/final-20261107`. It preserves source,
the actual installed native package, pilot/final input bytes, the nine timing
prompts, software versions and freshly hashed native dependencies. Startup
verification checks frozen source, native artifacts, copied inputs and shared
project libraries before execution. Driver and GPU information, topology,
allocator settings, source hashes and run fingerprints accompany the job
records. Frozen runtime sources remain separate from later report edits.

Raw JSONL retains token IDs, predictions, scores and failures. Final analysis
requires compatible protocol identities across array cells and rejects missing
or duplicate cells, failed transfer rows and diagnostic substitutes. Pilot,
final-quality and repeated-timing strata remain separate. All figure exports
are PNG. See [README](../README.md), [cluster setup](CLUSTER.md),
[the protocol](EXPERIMENT_PROTOCOL.md) and [analysis commands](ANALYSIS.md).

The completed acceptance stamp is
[`artifact_manifest.json`](../results/analysis-final/artifact_manifest.json),
with status `complete_and_verified`. The run has no remaining GPU jobs.
The reproducible reporting entry point is:

```bash
../.venv/bin/python scripts/finalize_campaign.py runs/final-20261107 \
  --out results/analysis-final
```

The implementation also passed 93 Python tests, 52 portable Rust tests and
three real-H200 CUDA tests. Test, integrity and scheduler records are retained
with the frozen campaign. The code and analysis satisfy the core proposal;
publication formatting and any extended performance study are separate work.

## Appendix A. Transport validation record

Two integration failures preceded the accepted native pilot: incompatible
large fragmented expandable allocations during registration, and an
unsegmented whole-cache native write. Controlled GPU probes established the
allocator setting; bounded writes then passed exact full-cache GPU byte checks
and local/remote continuation equivalence. The retained failure jobs,
registration sweeps, source changes and raw log paths are documented in
[REGISTRATION_DIAGNOSIS.md](REGISTRATION_DIAGNOSIS.md). These diagnostics
establish correctness of the exercised path and are not substituted for the
final retrieval or repeated handoff measurements.

## References

1. Nandor Licker, Kevin Hu, Vladimir Zaytsev, and Lequn Chen.
   [fabric-lib: RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656), 2026 revision.
2. Cheng-Ping Hsieh et al.
   [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), COLM 2024.
3. Zirui Liu et al.
   [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), ICML 2024.
4. Qwen Team.
   [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), 2025.
