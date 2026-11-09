"""Pinned official RULER generation, lossless manifest import, and NIAH scoring.

No substitute data are generated here: upstream ``synthetic/niah.py`` is invoked
with the task arguments and templates read from its pinned source checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

UPSTREAM_SHA = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TASKS = ("niah_single_2", "niah_multikey_1", "niah_multiquery")
CONTEXTS = (4096, 8192, 16384)
SPLIT_SEEDS = {"pilot": 71201, "final": 92701}
SCHEMA_VERSION = "bonsai.ruler.v1"
FACT_RE = re.compile(r"One of the special magic numbers for (?P<key>[^\n]+?) is: (?P<value>\d{7})\.")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def _source_module(path: Path):
    spec = importlib.util.spec_from_file_location("_bonsai_ruler_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_upstream(root: str | Path) -> dict:
    root = Path(root).resolve()
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != UPSTREAM_SHA:
        raise ValueError(f"RULER commit {actual} != pinned {UPSTREAM_SHA}")
    paths = ["scripts/data/synthetic/niah.py", "scripts/data/synthetic/constants.py",
             "scripts/synthetic.yaml", "scripts/data/tokenizer.py",
             "scripts/eval/synthetic/constants.py", "scripts/eval/evaluate.py"]
    files = {}
    for relative in paths:
        content = (root / relative).read_bytes()
        tracked = subprocess.run(["git", "-C", str(root), "show", f"{UPSTREAM_SHA}:{relative}"],
                                 check=True, capture_output=True).stdout
        if content != tracked:
            raise ValueError(f"Pinned RULER source modified: {relative}")
        files[relative] = sha256(content)
    corpus = root / "scripts/data/synthetic/json/PaulGrahamEssays.json"
    if not corpus.is_file():
        raise FileNotFoundError(f"Official essay corpus missing: {corpus}; see docs/RULER.md")
    text = json.loads(corpus.read_text(encoding="utf-8")).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Official essay corpus is empty")
    return {"commit": actual, "files_sha256": files, "corpus_sha256": sha256(corpus.read_bytes())}


def render_prompt(tokenizer, row: dict) -> str:
    """Chat user contains the question; official answer prefix is assistant prefill."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": row["input"]}],
        tokenize=False, add_generation_prompt=True,
    ) + row.get("answer_prefix", "")


def import_rows(rows: list[dict], tokenizer, *, task: str, context_tokens: int,
                split: str, seed: int, expected_count: int, upstream: dict,
                source_file: str = "", max_new_tokens: int = 128) -> list[dict]:
    if task not in TASKS or split not in SPLIT_SEEDS or context_tokens not in CONTEXTS:
        raise ValueError("Unsupported task, split, or context")
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} official examples, found {len(rows)}")
    if [r.get("index") for r in rows] != list(range(expected_count)):
        raise ValueError("Official indices must be unique, contiguous, and ordered")
    imported = []
    for row in rows:
        if not isinstance(row.get("input"), str) or not row.get("outputs"):
            raise ValueError("Missing official input or outputs")
        outputs = row["outputs"]
        if not all(isinstance(x, str) and x for x in outputs):
            raise ValueError("Official outputs must be nonempty strings")
        prompt = render_prompt(tokenizer, row)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False)
        ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
        if len(ids) + max_new_tokens > context_tokens:
            raise ValueError(f"Post-chat example {row['index']} needs {len(ids)} + "
                             f"{max_new_tokens} > {context_tokens}; refusing truncation")
        sid = f"{split}/{task}/{context_tokens}/{seed}/{row['index']:04d}"
        facts = []
        for match in FACT_RE.finditer(prompt):
            touching = [i for i, (a, b) in enumerate(offsets)
                        if b > match.start() and a < match.end()]
            if not touching:
                raise ValueError("Fact lacks tokenizer offsets")
            facts.append({"fact_id": sid + f"/fact/{len(facts)}", "key": match["key"],
                          "value": match["value"], "target": match["value"] in outputs,
                          "character_start": match.start(), "character_end": match.end(),
                          "token_start": min(touching), "token_end": max(touching) + 1,
                          "value_character_start": match.start("value"),
                          "value_character_end": match.end("value")})
        if not all(any(f["value"] == answer and f["target"] for f in facts) for answer in outputs):
            raise ValueError("Official target answer is absent from a fact in the rendered prompt")
        imported.append({"schema_version": SCHEMA_VERSION, "sample_id": sid,
                         "task": task, "context_tokens": context_tokens, "split": split,
                         "seed": seed, "index": row["index"], "input": row["input"],
                         "outputs": outputs, "answer_prefix": row.get("answer_prefix", ""),
                         "prompt": prompt, "input_ids": ids, "input_length": len(ids),
                         "max_new_tokens": max_new_tokens, "model": MODEL,
                         "model_revision": MODEL_REVISION, "facts": facts,
                         "offset_scope": "rendered_prompt", "offset_intervals": "half-open",
                         "prompt_protocol": "chat-user-input-assistant-answer-prefix-v1",
                         "hashes": {"prompt_sha256": sha256(prompt),
                                    "input_ids_sha256": sha256(canonical_json(ids)),
                                    "upstream_row_sha256": sha256(canonical_json(row))},
                         "upstream": {**upstream, "source_file": source_file},
                         "upstream_row": row})
    return imported


def generate_cell(upstream_root: str | Path, tokenizer_path: str | Path,
                  tokenizer, output_dir: str | Path, *, task: str, context_tokens: int,
                  split: str, count: int, python: str = sys.executable,
                  timeout: int = 1800) -> Path:
    """Call the official generator directly; never use upstream prepare.py's shell."""
    import yaml
    root, output = Path(upstream_root).resolve(), Path(output_dir).resolve()
    provenance = verify_upstream(root)
    if task not in TASKS or split not in SPLIT_SEEDS or context_tokens not in CONTEXTS or count <= 0:
        raise ValueError("Invalid RULER cell")
    constants = _source_module(root / "scripts/data/synthetic/constants.py").TASKS["niah"]
    task_config = yaml.safe_load((root / "scripts/synthetic.yaml").read_text())[task]
    budget = constants["tokens_to_generate"]
    # HF raw tokenization used by upstream excludes chat delimiters. Reserve the
    # measured wrapper plus a small boundary margin; exact actual lengths below
    # are authoritative and overlength examples fail, never truncate.
    probe = {"input": "Context probe.", "answer_prefix": " Answer:"}
    plain = probe["input"] + probe["answer_prefix"]
    overhead = max(0, len(tokenizer.encode(render_prompt(tokenizer, probe), add_special_tokens=False))
                   - len(tokenizer.encode(plain, add_special_tokens=False))) + 8
    seed = SPLIT_SEEDS[split] + TASKS.index(task) * 100 + CONTEXTS.index(context_tokens)
    cell_dir = output / split / str(context_tokens)
    source = cell_dir / "official" / task / "validation.jsonl"
    manifest = cell_dir / f"{task}.jsonl"
    if source.exists() or manifest.exists():
        raise FileExistsError(f"Cell exists; use a fresh output directory: {cell_dir}/{task}")
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [python, str(root / "scripts/data/synthetic/niah.py"),
            "--save_dir", str(cell_dir / "official"), "--save_name", task,
            "--subset", "validation", "--tokenizer_path", str(Path(tokenizer_path).resolve()),
            "--tokenizer_type", "hf", "--max_seq_length", str(context_tokens - overhead),
            "--tokens_to_generate", str(budget), "--num_samples", str(count),
            "--random_seed", str(seed), "--template",
            constants["template"] + constants["answer_prefix"]]
    for key, value in task_config["args"].items():
        argv.extend([f"--{key}", str(value)])
    log = cell_dir / f"{task}.generator.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json({"argv": argv}) + "\n")
        handle.flush()
        subprocess.run(argv, check=True, stdout=handle, stderr=subprocess.STDOUT,
                       timeout=timeout, env=os.environ.copy())
    rows = read_jsonl(source)
    provenance.update({"source_sha256": sha256(source.read_bytes()),
                       "generator_argv": argv, "chat_token_reserve": overhead})
    imported = import_rows(rows, tokenizer, task=task, context_tokens=context_tokens,
                           split=split, seed=seed, expected_count=count, upstream=provenance,
                           source_file=str(source), max_new_tokens=budget)
    write_jsonl(manifest, imported)
    return manifest


def score_prediction(prediction: str, outputs: list[str]) -> dict:
    """Exactly match official evaluate.py normalization and NIAH string_match_all.

    The official metric's misleading name means fractional target recovery; the
    separate all_answers_recovered field means every reference was recovered.
    """
    if not isinstance(prediction, str) or not outputs or not all(isinstance(x, str) and x for x in outputs):
        raise ValueError("Prediction must be a string and outputs nonempty strings")
    normalized = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
    recovered = [answer.lower() in normalized.lower() for answer in outputs]
    return {"raw_output": prediction, "normalized_output": normalized,
            "per_answer_recovered": recovered,
            "target_recovery": sum(recovered) / len(recovered),
            "all_answers_recovered": all(recovered)}


def score_records(records: Iterable[dict]) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot score zero predictions")
    scored = [score_prediction(row.get("raw_output", row.get("pred")), row["outputs"]) for row in rows]
    return {"count": len(scored),
            "official_ruler_score": round(100 * sum(x["target_recovery"] for x in scored) / len(scored), 2),
            "target_recovery": sum(x["target_recovery"] for x in scored) / len(scored),
            "all_answers_rate": sum(x["all_answers_recovered"] for x in scored) / len(scored),
            "per_example": [{**row, **score} for row, score in zip(rows, scored)]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--upstream-root", type=Path, required=True)
    gen.add_argument("--tokenizer-path", type=Path, required=True,
                     help="Local tokenizer snapshot at the pinned MODEL_REVISION")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--split", choices=list(SPLIT_SEEDS), required=True)
    gen.add_argument("--count", type=int, default=None)
    gen.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    gen.add_argument("--contexts", type=int, nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    score = sub.add_parser("score")
    score.add_argument("predictions", type=Path)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score_records(read_jsonl(args.predictions))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.tokenizer_path.resolve().name != MODEL_REVISION:
        parser.error("--tokenizer-path must identify the local immutable snapshot directory " + MODEL_REVISION)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    count = args.count if args.count is not None else (100 if args.split == "final" else 5)
    if args.split == "final" and count != 100:
        parser.error("Final protocol requires exactly 100 examples per cell")
    for context in args.contexts:
        for task in args.tasks:
            path = generate_cell(args.upstream_root, args.tokenizer_path, tokenizer,
                                 args.output_dir, task=task, context_tokens=context,
                                 split=args.split, count=count)
            print(path, flush=True)


if __name__ == "__main__":
    main()
