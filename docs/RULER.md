# Official RULER retrieval protocol

This integration runs the NVIDIA RULER v1 generator directly, pinned at
`db43f92ff634bbce1c33d1fb15858aec4bba509a`. It does not replace RULER with
handmade retrieval prompts. The upstream README now calls this v1 pipeline
deprecated; the frozen proposal protocol deliberately uses its named tasks.

The model and tokenizer are `Qwen/Qwen3-4B-Instruct-2507` at immutable Hugging
Face revision `cdbee75f17c01a7cc42f958dc650907174af0554` (resolved through the
HF model API). The experiment matrix is:

| Task | Total context caps | Final examples per cap | Pilot examples per cap |
| --- | --- | ---: | ---: |
| `niah_single_2` | 4096, 8192, 16384 | 100 | 5 |
| `niah_multikey_1` | 4096, 8192, 16384 | 100 | 5 |
| `niah_multiquery` | 4096, 8192, 16384 | 100 | 5 |

Final evaluation contains **900 examples**. Pilot contains 45 examples and
uses different seeds. The seed formula is `base + task_index * 100 +
context_index`, using the task/cap order above. Bases are 92701 for final and
71201 for pilot. Pilot data are for tuning and must be excluded from final
metrics. Separate seeds provide separate generated examples, not a guarantee
that no individual random numeric value can recur.

## Generate and preserve the source corpus

Install the bench package (`pip install -e bench`; see `bench/pyproject.toml`). The
upstream generator requires its official `PaulGrahamEssays.json`; run the
unmodified upstream downloader from `scripts/data/synthetic/json/`:

```bash
cd ../third_party/RULER/scripts/data/synthetic/json
/mnt/home/jiechen7/projects/HW/.venv/bin/python download_paulgraham_essay.py
```

The upstream downloader catches individual download failures, so inspect the
log and require **218 essays: 49 GitHub text files and 169 HTML essays**, with
no `Fail download` messages. Preserve the assembled corpus. Live web sources
can change: the frozen corpus and its hash are the reproducible input, rather
than a promise that a later redownload produces identical bytes. The initial
frozen corpus is `data/ruler/PaulGrahamEssays.json`, SHA-256
`8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd`.
`data/ruler/corpus_provenance.json` records the source list hash and completeness.
To reproduce, copy this frozen corpus into the upstream expected path.

The upstream generator uses NLTK `punkt`/`punkt_tab`. Install both resources
into a workspace cache, and expose that directory through `NLTK_DATA`.
Use a local snapshot of the tokenizer at the revision above, never `main`.

```bash
export HF_HOME=/mnt/home/jiechen7/projects/HW/.cache/huggingface
export NLTK_DATA=/mnt/home/jiechen7/projects/HW/.cache/nltk_data
TOKENIZER="$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
../.venv/bin/python -m bonsai_bench.ruler generate \
  --upstream-root ../third_party/RULER --tokenizer-path "$TOKENIZER" \
  --output-dir data/ruler --split pilot
../.venv/bin/python -m bonsai_bench.ruler generate \
  --upstream-root ../third_party/RULER --tokenizer-path "$TOKENIZER" \
  --output-dir data/ruler --split final
```

Use a fresh output directory when regenerating. Existing official files and
manifests cause an error to prevent stale-cache reuse. The final CLI requires
100 examples per cell. Generator errors, timeouts, wrong counts, invalid
indices, missing target facts, and post-chat overlength prompts fail instead
of producing a valid-looking manifest. There is no silent token truncation.
The exact generator argv and its complete stdout/stderr are preserved per cell.
The wrapper bypasses upstream `prepare.py`, which constructs a shell command
and catches generator errors; it invokes the unmodified `niah.py` with an argv
list and `check=True` instead. Task parameters and templates are read from
the verified upstream YAML and constants, not independently recreated here.

## Prompt and length convention

`context_tokens` is the total sequence cap **including 128 output tokens**,
matching official RULER `--max_seq_length`. It is not a promise that every
input has exactly that many tokens: the official generator can shorten its
haystack in 500-word increments. Save and analyze the actual `input_length`.

The official `input` becomes one Qwen user message. Apply the pinned
tokenizer's chat template with `add_generation_prompt=True`, then append the
official `answer_prefix` as assistant prefill. Tokenize the complete rendered
string with `add_special_tokens=False`, `truncation=False`. No extra system
prompt or answer instruction is added. `prompt_protocol` is
`chat-user-input-assistant-answer-prefix-v1`.

The generator receives a cap reduced by the measured chat wrapper overhead
plus an eight-token boundary margin. Every resulting prompt is then measured
exactly; `input_length + max_new_tokens <= context_tokens` is enforced. The
same manifest token IDs must be used across every codec and baseline.

## JSONL interface for runners and analysis

Manifest paths are `data/ruler/{split}/{context_tokens}/{task}.jsonl`.
Each line is an object with `schema_version = "bonsai.ruler.v1"`:

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable `{split}/{task}/{cap}/{seed}/{index:04d}` pairing key |
| `task`, `context_tokens`, `split`, `seed`, `index` | Cell identity and generation seed |
| `model`, `model_revision` | Immutable model/tokenizer identity |
| `input`, `outputs`, `answer_prefix` | Unmodified official prompt, list of answers, prefill |
| `prompt` | Complete rendered Qwen chat prompt |
| `input_ids`, `input_length` | Exact model input IDs and their actual count |
| `max_new_tokens` | 128; greedy generation should use `do_sample=False` |
| `facts` | All planted facts, including unqueried distractor facts |
| `hashes` | SHA-256 of rendered UTF-8 prompt, canonical JSON token IDs, canonical upstream row |
| `upstream` | Pinned commit, generator/scorer source hashes, corpus/source hashes and generator argv |
| `upstream_row` | Complete raw official source object without field deletion |

Every fact has `fact_id`, `key`, `value`, `target`, `character_start`,
`character_end`, `token_start`, `token_end`, `value_character_start`, and
`value_character_end`. **All offsets refer to `prompt`, not raw `input`.**
Intervals are half-open. Token intervals include every token whose offset
overlaps the complete fact sentence. Character offsets count Python Unicode
characters; token indices index `input_ids`. Target indicates the fact's
value is among official reference answers. Facts are numbered in prompt
order, with IDs `{sample_id}/fact/{ordinal}`. The saved whole-fact token span
is suitable for positional analysis; it includes key and value together.

A runner should consume `input_ids` directly, preserve `sample_id`, `outputs`
and raw completion text, and never decode/retemplate/retruncate the prompt.
EOS can end generation before the maximum budget. Preserve an explicit
generation error status instead of silently omitting failed examples.

## Exact scoring

```python
from bonsai_bench.ruler import score_prediction, score_records

detail = score_prediction(raw_completion, outputs)
summary = score_records([
    {"sample_id": sample_id, "raw_output": raw_completion, "outputs": outputs}
])
```

The official evaluator strips the completion, converts ASCII characters
`\x00` through `\x1f` to newlines, strips again, then checks each reference
with case-insensitive substring matching. It accepts a matching substring
inside a longer number. This permissiveness is intentional for exact parity.
Do not normalize numbers, enforce word boundaries, extract a single answer,
or count just one correct answer for multiquery examples.

Despite its upstream name `string_match_all`, official RULER's NIAH metric is
the mean **fraction of target answers recovered per example**, multiplied by
100 and rounded to two decimals. `score_prediction` returns
`per_answer_recovered` (a Boolean list aligned with `outputs`),
`target_recovery` (0–1), and the separate `all_answers_recovered` Boolean.
It also preserves `raw_output` and `normalized_output`. `score_records`
returns `official_ruler_score` (0–100), `target_recovery` (0–1),
`all_answers_rate` (0–1), `count`, and all `per_example` records. Empty
completions score zero; empty datasets raise an error.

The standalone scorer accepts `raw_output` or official `pred` fields and
preserves all original record fields in its output:

```bash
../.venv/bin/python -m bonsai_bench.ruler score predictions.jsonl --output scores.json
../.venv/bin/python -m pytest tests/test_ruler.py -q
```
