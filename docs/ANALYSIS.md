# Paired retrieval analysis

The completed core campaign is in `results/analysis-final/`: all 900 prompts,
6,300 predictions, 2,700 native transfers and 3,600 exact handoff comparisons
passed. All observed retrieval scores are 100%; the conservative simultaneous
lower bound is −5.72 percentage points, so the two-point margin is inconclusive.
See the [paper](PAPER_DRAFT.md) and
[measured campaign tables](../results/analysis-final/campaign/campaign.md).

For the complete pipeline, including three timing blocks, explicit text-refill
figure labeling and artifact hashes, run:

```bash
../.venv/bin/python scripts/finalize_campaign.py runs/final-20261107 \
  --out results/analysis-final
```

The optional `bench[analysis]` dependency set includes Seaborn 0.13.2 for the
publication figure. The inference snapshot is unchanged; reporting code and
its additional dependency are retained under `analysis_source/` in the output.

Run the analysis from the repository root with the benchmark package installed
or with `PYTHONPATH=bench`. The repository Python environment is `../.venv`.

```bash
PYTHONPATH=bench ../.venv/bin/python -m bonsai_bench.analysis \
  results/pilot-local-v2.jsonl \
  --manifest data/ruler/pilot/*/niah*.jsonl \
  --out results/analysis-pilot-local-v2
```

The output contains `analysis.json`, a readable `summary.md`, and PNG figures
at 300 dpi. Matplotlib is required for figures; seaborn is optional. When
seaborn is unavailable, its bundled Matplotlib white style is used with visible
ticks and the same paper colors. `--no-plots` produces tables only. No other
figure formats are generated.

The completed local pilot has 45 distinct prompts and 135 successful
predictions: local BF16, local INT8 reconstruction, and local INT4
reconstruction. All three have observed target recovery and all-answer rate
of 100%. Each task/context comparison has only five paired prompts. These
observations do not establish the two-percentage-point preservation margin.
The earlier `results/pilot-local.jsonl` contains 135 execution failures; its
separate report is `results/analysis-pilot-local-failures/summary.md`. It is
not pooled with successful pilot predictions. This early local pilot predates
the runner's frozen source hashes and explicit byte-scope metadata; it is
exploratory and cannot validate the final native transfer pipeline.

## Input integrity and pairing

Every result is checked against the complete canonical manifest hash and model
revision. Non-finite numbers anywhere in input or plotted summaries are errors,
including NaN in a metric that would otherwise be omitted from a figure.
Duplicate rows, repeated retry attempts, conflicting prompt hashes, inconsistent
target/all-answer scores, and split mismatches raise errors. Retries must be
resolved explicitly into a reviewed input; failures are never silently replaced.
The original runner JSONL remains the audit record.

Pilot/final, run fingerprints, and local/TCP diagnostic/native fabric transport
are separate strata by default. INT8/INT4 transfer is paired only with BF16
transfer in the same run and transport; local reconstruction is paired only
with local BF16. `text_refill` is reported as its own baseline. A local zero-byte
BF16 row means there was no encoding or network transfer; it is omitted from
the quality-versus-bytes figure rather than used as a BF16 cache-size baseline.

Every table reports attempted and successful prompt counts. JSON additionally
records the expected manifest count, unattempted prompts, unmatched pairs,
execution failures, and the exact source run fingerprints. Quality means use
successful predictions only and are conditional on that denominator. Paired
differences use only successful pairs and show the attempted union. This
conditional analysis cannot support full-study claims when pairs are missing.

## Uncertainty and the two-point margin

The primary score is the official per-prompt fraction of requested targets
recovered, retained as a fractional value. A partially correct multi-query
answer is not converted to a binary failure. The all-targets-recovered rate is
reported separately, including a paired difference and its own interval.

Paired percentile bootstrap 95% intervals resample whole prompts, keeping all
targets in a prompt together (10,000 resamples; seed 20261107). They are
pointwise intervals. With zero observed differences the bootstrap has no
empirical variation; its zero-width interval must not be interpreted as proof
that the population difference is exactly zero.

To address that issue, the analysis also computes a one-sided exact
Clopper–Pearson upper bound on the probability that a prompt's target score
degrades. Since each prompt's score loss is bounded by one, the negative of
this upper bound is a conservative lower confidence bound on the population
mean paired score difference. It remains valid when partial losses and gains
occur, but deliberately ignores their magnitudes and can be quite conservative.
At 100 prompts with zero degraded prompts, the pointwise 95% upper degradation
bound is about 2.95%, so the proposed 2% margin remains inconclusive.

The report also uses Bonferroni-adjusted bounds over all task/context/precision
comparisons within a run/transport stratum. These support simultaneous claims
about the primary target score only; the bootstrap and all-target rate remain
exploratory. The preservation flag requires a complete verified native final
matrix, complete successful pairs, the adjusted conservative lower bound at
least −0.02, and the pointwise bootstrap lower endpoint at least −0.02.
Failure to meet those conditions means inconclusive, not equivalence or
preservation. No current local pilot output makes a preservation claim.

## Final matrix and array jobs

The final study requires nine task/context cells, 100 distinct prompts per
cell, and all three successful native transfer precisions: 900 distinct
prompts and 2,700 transferred continuations. Local and text-refill baselines
can be included in the files and are reported separately.

```bash
PYTHONPATH=bench ../.venv/bin/python -m bonsai_bench.analysis \
  /path/to/frozen-run/results/*.jsonl \
  --manifest data/ruler/final/*/niah*.jsonl \
  --require-complete-final --out results/analysis-final
```

`--require-complete-final` rejects missing cells, failures in transfer rows,
incorrect per-cell counts, mismatched paired prompt sets, pilot inputs, and
TCP diagnostic/local substitutes. The CLI requires source-hash provenance
sidecars. For final array jobs, a protocol fingerprint hashes every config
field except `sample_hashes`: all source hashes, model revision, precision
conditions, software versions, device settings, and transport must agree.
That permits the nine intentionally disjoint frozen sample selections to be
analyzed together while recording their original run fingerprints. Duplicate
prompt/condition rows across combined jobs are errors. Compatible exploratory
arrays can opt in with `--aggregate-compatible-runs`; incompatible protocols
remain separate and cannot satisfy a complete final claim.

## Communication, latency, memory, and positions

Each condition cell reports p50, p95, mean, and sample count for encoding,
decoding, sender-cache-ready to receiver-ready ACK, prefill, continuation,
application frame/payload/metadata/control bytes, and logged sender/receiver
memory fields. Null values mean unavailable measurements and are not converted
to zero. Local `wire_bytes` are encoded frame size only, not transmitted bytes.
Remote bytes exclude network headers. Figures label that scope explicitly.
Phase breakdown bars show phase medians; their sum is not a total-latency
percentile. Memory values follow the runner's allocator/registration scopes
and include model/cache residency; they are not attributed exclusively to
codec temporary storage.

The local pilot produces quality-versus-context, paired differences,
quality-versus-encoded-size, codec phase breakdown, and added failures versus
fact position. Handoff-latency-versus-context is generated only for measured
native `fabric-lib` transfer rows. TCP diagnostics and local codec timings are
never substituted. Fewer bytes alone do not imply faster handoff, and this
analysis does not infer a speedup without measured complete handoff latency.

Fact position uses the manifest's actual input token length and target fact
token start, divided into beginning/middle/end thirds. Outputs are matched to
target facts by value; repeated matching facts are represented by their mean
position. Position tables retain restored targets as well as added failures.
The figure pools descriptive target counts within a stratum and labels their
within-prompt correlation; it does not use targets as independent uncertainty
units or claim a position effect from a small pilot.

Validation:

```bash
PYTHONPATH=bench ../.venv/bin/python -m pytest tests/test_analysis.py -q
```
