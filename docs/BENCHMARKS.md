# Benchmarks and Evaluation

This document describes Bonsai's evaluation suite, including which benchmarks
were migrated from **Atom: Low-Bit Quantization for Efficient and Accurate LLM
Serving** (Zhao et al., MLSys 2024; paper:
https://proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html,
code: https://github.com/efeslab/Atom).

Code lives in [`bench/`](../bench) (Python package `bonsai_bench`).

```bash
pip install -r bench/requirements.txt && pip install -e bench
pytest bench/tests          # 12 tests; tiny random Qwen2 model, CPU is fine
```

---

## 1. What Atom evaluates, and what transfers to Bonsai

Atom quantizes weights, activations **and the KV cache** to 4 bits for
single-GPU serving throughput. Bonsai quantizes **only the KV cache, only
while it is in transit** between a prefill GPU and a decode GPU. That
difference decides what can be migrated:

| # | Atom benchmark | Atom setup | Relevance to Bonsai | Decision | Bonsai implementation |
|---|---|---|---|---|---|
| B1 | **Perplexity**: WikiText2, PTB, C4 | `model/main.py --eval_ppl`; 2048-token windows; fake-quant W, A, and KV | Cheap, continuous quality signal; good for wide sweeps | **Migrated, adapted** to "prefix-handoff" perplexity (§2) | `bonsai_bench.ppl` ✅ |
| B2 | **Zero-shot accuracy**: PIQA, ARC-e, ARC-c, BoolQ, HellaSwag, WinoGrande | lm-eval 0.3.0 | Short contexts; a regression check, not where compression pays off | **Migrated** as an lm-eval ≥ 0.4 model that performs the handoff | `bonsai_bench.zeroshot` ✅ (not run locally: needs lm-eval + GPU) |
| B3 | **Ablation ladder**: base → +reorder → +keep-int8 → +group → +clip → +GPTQ → +KV-int4 | `scripts/run_atom_ablation.sh` | The *method* (add one technique at a time) fits our codec design space | **Migrated as a KV-codec ladder** + per-layer sensitivity sweep (§3) | `bonsai_bench.ppl --precisions …`, `--layer-sensitivity` ✅ |
| B4 | **Kernel microbenchmarks**: INT4 GEMM, reorder, RMSNorm vs. baselines | NVBench, RTX 4090 | Atom's kernels don't apply; the methodology (throughput vs. roofline) does | **Methodology migrated** to the KV codec kernels | `bonsai_bench.codec_bench` ✅ (times the PyTorch reference until CUDA kernels exist) |
| B5 | **End-to-end serving**: throughput and latency vs. batch size, using a synthetic lognormal workload and ShareGPT | Punica, continuous batching, random weights | Workloads and metrics transfer; Punica and random weights don't (we need real outputs) | **Workloads migrated**; load generator pending M1 (§4) | `bonsai_bench.workloads` ✅, load generator ⏳ |
| — | W4A4 weight/activation quantization, GPTQ, outlier reordering, FP4 (bitsandbytes), Llama/Mixtral | | Outside Bonsai's scope (weights are unchanged; model is Qwen2.5-1.5B) | Not migrated | — |

**KV scheme.** Atom's KV quantizer is INT4, asymmetric, with one group per
(head, token) over the 128 channels (`quantize_attn_k/v_wrapper`). It is
available in Bonsai as the codec **`atom-kv4`**, both in `bonsai-ctl` and in
`bonsai_bench`, at 26.6% of BF16 bytes.

Atom uses an integer zero point; Bonsai stores an f16 zero point. A
Python-only `atom-kv4-exact` reproduces Atom's quantizer exactly, so the two
can be compared. A unit test checks that their errors match to within 2%
relative.

**Licensing.** The Atom repository has no license file. Bonsai reimplements
the benchmark methods (dataset splits, window lengths, task list, workload
distribution and its parameters) from the paper and the repository's
description, and cites them. No Atom code is copied.

**Atom's numbers can't be compared directly with ours.** Atom quantizes the
KV cache inside *every* attention call, on top of W4A4 weights and
activations, for Llama models. Bonsai measures KV-only, in-transit
quantization on Qwen2.5-1.5B. The tables have the same shape, but the
numbers mean different things.

---

## 2. B1: Prefix-handoff perplexity (`bonsai_bench.ppl`)

Each 2048-token window (Atom's length) is split at the handoff point P
(default P = 1024):

```text
window[:P]   ─ prefill (BF16) ─▶ KV cache ─ roundtrip(precision) ─▶ cache the decoder sees
window[P:]   ─ teacher-forced decode over that cache (its own new KV stays BF16)
score        = NLL of window[P+1:]   (window[P] comes from prefill logits; the handoff can't change it)
```

This matches what a real handoff does. With `bf16`, the result equals plain
model perplexity on the same positions; a test checks this to a relative
tolerance of 1e-4 on a tiny Qwen2 model.

```bash
python -m bonsai_bench.ppl --datasets wikitext2,c4,ptb \
    --precisions bf16,int8,int4,int4-pertoken,atom-kv4,atom-kv4-exact --limit 0
python -m bonsai_bench.ppl --config config/bonsai-ctl.toml --precisions int8-edges   # custom precisions
python -m bonsai_bench.ppl --prefix-frac 0.875     # longer transferred prefix, shorter scored suffix
```

Output: `results/ppl.jsonl`, with fields `{dataset, precision, seqlen, prefix_len, ppl, windows, scored_tokens}`.

Dataset sources are the same as Atom's (and GPTQ's):

- WikiText2: `wikitext-2-raw-v1` test split.
- PTB: `ptb_text_only` validation split. Its loading script may need an
  older version of `datasets`.
- C4: `allenai/c4` validation shard 0, first 1100 documents.

---

## 3. B3: Codec ablation ladder and layer sensitivity

Following Atom's one-technique-at-a-time ablation, applied to the KV codec
design space:

| Step | Precision | What it adds | Bytes vs. BF16 |
|---|---|---|---:|
| 0 | `bf16` | Baseline | 100% |
| 1 | `atom-kv4` | INT4, per-token, group = head_dim (Atom's KV scheme) | 26.6% |
| 2 | `int4-pertoken` | Finer groups (32) | 31.3% |
| 3 | `int4` | K per-channel (outlier channels in K) | 31.3% |
| 4 | `int8-edges` (or the sweep's choice) | Mixed precision: protect sensitive layers | 36.8% |
| 5 | `int8` | Conservative 8-bit | 50.8% |

```bash
python -m bonsai_bench.ppl --config config/bonsai-ctl.toml \
    --precisions bf16,atom-kv4,int4-pertoken,int4,int8-edges,int8
```

**Layer sensitivity** chooses which layers mixed precision should protect,
which is the input to adaptive precision level (a). Each run quantizes one
layer with the given codec and keeps the rest BF16:

```bash
python -m bonsai_bench.ppl --datasets wikitext2 --layer-sensitivity int4 --limit 40
# -> precision = "int4@L0" ... "int4@L27"; rank by ppl - ppl(bf16), protect the top k
```

Write the chosen layers into `[precisions.*]` in the `bonsai-ctl` config, then
confirm the result on LongBench (§5).

---

## 4. B4 and B5: Systems benchmarks

### B4: Codec microbenchmark (`bonsai_bench.codec_bench`)

This times encode + decode of a full Qwen2.5-1.5B KV cache at each context
length and reports throughput over raw BF16 bytes, next to a memcpy of the
same size (the roofline, as in Atom's kernel plots). It also prints a
`[codec.adaptive.cost_model]` block for `bonsai-ctl`.

```bash
python -m bonsai_bench.codec_bench --seq-lens 1024,8192,32768
```

Until the CUDA kernels land (M2) this times the PyTorch reference, which is
far slower than the kernels will be. Don't use those numbers to calibrate the
cost model.

### B5: Workloads (`bonsai_bench.workloads`)

```bash
python -m bonsai_bench.workloads atom-lognormal --num 1000 --maxlen 2048 --out workloads/atom.jsonl
python -m bonsai_bench.workloads sharegpt --sharegpt-path ShareGPT_V3_unfiltered_cleaned_split.json --num 1000 --out workloads/sharegpt.jsonl
python -m bonsai_bench.workloads fixed --context-lens 1024,2048,4096,8192,16384,32768 --output-len 32 --num 48 --out workloads/fixed.jsonl
```

- **`atom-lognormal`** is Atom's synthetic distribution: prompt length ~
  lognormal(0.8, −1, 18), total length ~ U[0, maxlen), seed `0xabcdabcd987`.
  1000 requests give a median prompt of 16 tokens and a mean output of
  ~990 tokens. The KV cache at handoff is then about 0.45 MB, so the transfer
  is negligible. This workload checks that compression **doesn't hurt** in the
  chat regime; it can't show a benefit.
- **`sharegpt`** uses real chat prompts and reply lengths (Atom's "production"
  trace). It covers the same regime with real text.
- **`fixed`** is a controlled context-length sweep. It is Bonsai's **primary**
  systems workload, because H1 and H2 (HLD) are about long contexts, where the
  KV cache is hundreds of MB.

The **load generator** (milestone 1, needs the control-plane TCP client) will
replay a workload JSONL against `bonsai-ctl` at a fixed concurrency. That
concurrency plays the role of Atom's batch-size sweep (1…16), here
{1, 2, 4, 8}. It will report Atom's metrics (per-request prefill latency,
decode latency, throughput) next to Bonsai's handoff metrics (CONTROL_PLANE.md §7).

---

## 5. Full evaluation matrix

| | Primary | Secondary (migrated from Atom) |
|---|---|---|
| **Quality** | LongBench v1 QASPER and HotpotQA (F1); RULER single-needle at controlled lengths | B1 perplexity (WikiText2/PTB/C4); B2 zero-shot (6 tasks); B3 ablation + layer sensitivity |
| **Systems** | `fixed` context sweep (1K–32K) × chunking × concurrency {1, 2, 4, 8} × bandwidth (line rate, 1 NIC, throttled 10–100 Gb/s), on 2 RDMA nodes | B4 codec microbenchmark; B5 `atom-lognormal` and `sharegpt` replay (no-regression check) |
| **Precisions** | `bf16`, `int8`, `int4`, best mixed precision, adaptive | adds `atom-kv4`, `atom-kv4-exact`, `int4-pertoken` (ablation) |

All quality benchmarks run on a single GPU through `roundtrip`, which is
bit-identical to a two-node handoff. Only the systems benchmarks need the
two RDMA nodes.

---

## 6. Status

| Item | State |
|---|---|
| PyTorch reference codecs (match `bonsai-core` byte accounting and `bonsai-ctl` names) | ✅ tested |
| B1 prefix-handoff perplexity, B3 ladder + layer sensitivity | ✅ tested on a tiny model; full-model runs pending (need GPU + datasets) |
| B2 zero-shot lm-eval wrapper | ✅ written; not yet run (needs lm-eval + GPU) |
| B4 codec microbenchmark | ✅ runs (reference backend) |
| B5 workloads | ✅ tested; load generator ⏳ (M1) |
| LongBench / RULER harness (primary quality) | ⏳ to write: same `roundtrip` hook, greedy generate, official F1 scripts |

Local testing used transformers 4.46 (the `key_cache` cache layout). The
transformers ≥ 4.56 / 5.x `cache.layers` path is implemented in
`bonsai_bench/kv.py`, but it hasn't been exercised, because the local torch
(2.2) is too old for transformers 5.
