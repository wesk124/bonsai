# Proposal execution and acceptance gates

**All core gates completed.** The accepted campaign contains 6,300 successful
predictions, 2,700 native transfers and 3,600 exact handoff comparisons.
All three timing blocks and actual-cache SHA256 checks passed. See the
[paper](PAPER_DRAFT.md) and [acceptance manifest](../results/analysis-final/artifact_manifest.json).

The core paper studies a concrete tradeoff: whether prompt-cache quantization
saves application bytes and complete handoff time while retaining retrieval
quality. Negative latency results and statistically inconclusive quality
comparisons remain valid outcomes. No optimization is selected using final
retrieval scores.

## Ten coordinated workstreams

All ten agents used GPT-6.1 Sol with high reasoning. They ran in staged groups
within the available concurrent-agent limit.

| Workstream | Delivered implementation | Main evidence |
|---|---|---|
| 01 Native codecs | CPU reference and real CUDA BF16/INT8/INT4 codecs | CPU tests and two H200 CUDA tests |
| 02 Python cache handoff | Packed GPU buffers, metadata, fresh-cache continuation | Packing/continuation tests and model checks |
| 03 RULER experiments | Pinned official generation, 45 pilot and 900 final prompts | Dataset hashes, token/fact audits |
| 04 Control plane | TCP coordination, validation, disconnect handling | Unit/socket tests and binary startup/shutdown |
| 05 Transport | Loopback plus real fabric-lib endpoint, topology selection | Eight portable tests and exact GPU RDMA integrity |
| 06 Data plane | Owned buffers, registration lifetimes, staging and receive arenas | Nine portable tests and real CUDA backend smoke |
| 07 Worker runner | Seven paired conditions, measurements, resumable provenance | Worker/runner tests and two-node model pilot |
| 08 Analysis | Paired intervals, strict coverage checks, PNG plots, campaign tables | Analysis and reporting regression tests |
| 09 Native environment | Local CUDA/RDMA dependencies and SLURM launch scripts | Linked native extension and H200 preflights |
| 10 Integration review | Allocator/message-size fixes, immutable snapshots, acceptance checks | Corruption rejection and complete pilot equivalence |

The measured benchmark uses the PyTorch packed GPU codec and native fabric
transport. The Rust CUDA codec and control/data-plane libraries have separate
tests; the paper does not attribute benchmark performance to those components.

## Ordered acceptance gates

1. **Functional cache semantics:** BF16 continuation agrees with ordinary model
   inference, preserving prompt positions, masks and output-token boundaries.
   Quantized local reconstruction uses the same prompts and decoding settings.
2. **Actual transport integrity:** verify position-varying GPU payloads byte for
   byte, including a complete 2,395,865,088-byte cache. Job `1294370` passed.
   The allocator setting and bounded 256 MiB native requests are recorded.
3. **Model handoff integrity:** complete all seven conditions on one pilot from
   each of nine task/context cells. Job `1294371` passed 63 predictions and all
   36 exact local/remote token-ID comparisons.
4. **Freeze and execute:** snapshot source, native binaries, dependencies and
   dataset bytes; verify hashes before every job. Run 900 final prompts across
   seven conditions, plus three separately reported pilot timing blocks.
   Frozen campaign: `runs/final-20261107`; arrays `1294381` and `1294382`.
5. **Complete-study acceptance:** require 6,300 successful predictions, 2,700
   native transferred continuations, 3,600 exact handoff comparisons, and three
   complete 36-row timing blocks. Reject missing cells, duplicate rows,
   transport substitutions and incompatible provenance.
6. **Paper artifacts:** report target recovery and all-answer recovery, paired
   confidence bounds, bytes, complete handoff p50/p95, phase timings, temporary
   memory and fact positions. Generate PNG figures and retain the raw evidence.

Run the final acceptance and analysis after the arrays complete:

```bash
../.venv/bin/python scripts/finalize_campaign.py runs/final-20261107 \
  --out results/analysis-final
```

The acceptance stamp is `results/analysis-final/artifact_manifest.json`, whose
`status` must be `complete_and_verified`. A queued job or partial output is not
completion evidence. The report is [PAPER_DRAFT.md](PAPER_DRAFT.md).

## Interpretation and follow-up priorities

The primary quality comparison is quantized minus BF16 within each task and
context. A zero point difference alone does not establish the two-percentage-
point margin. With 100 prompts per cell, conservative confidence bounds can
remain inconclusive even with no observed losses. Preserve that conclusion
rather than selectively extending favorable cells.

On a fast native link, the reference codec can cost more time than transmission
saves. If the completed results show that pattern, the next engineering study
should integrate the tested CUDA codec into the measured runner, reduce
per-tensor synchronization and copies, and compare persistent registrations.
That would be a separately frozen performance experiment. Actual slower links,
other models, larger contexts, pipelining, adaptive precision and HotpotQA are
follow-up work rather than requirements for the current core matrix.
