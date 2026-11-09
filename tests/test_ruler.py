"""RULER protocol invariants, scoring equivalence, and subprocess failure handling."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

from bonsai_bench import ruler


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def official_row():
    return {"index": 0, "input": "One of the special magic numbers for red-dog is: 1234567. "
            "One of the special magic numbers for blue-cat is: 7654321. Which is red-dog?",
            "outputs": ["1234567"], "answer_prefix": " Answer:"}


def imported(rows=None, **overrides):
    kwargs = dict(task="niah_multikey_1", context_tokens=4096, split="pilot",
                  seed=71201, expected_count=1, upstream={"commit": ruler.UPSTREAM_SHA})
    kwargs.update(overrides)
    return ruler.import_rows(rows or [official_row()], CharacterTokenizer(), **kwargs)


def test_manifest_retains_exact_tokens_and_fact_offsets():
    row = imported()[0]
    assert row["input_ids"] == list(map(ord, row["prompt"]))
    assert row["input_length"] == len(row["input_ids"])
    assert row["upstream_row"] == official_row()
    assert [fact["target"] for fact in row["facts"]] == [True, False]
    for fact in row["facts"]:
        text = row["prompt"][fact["character_start"]:fact["character_end"]]
        decoded = "".join(map(chr, row["input_ids"][fact["token_start"]:fact["token_end"]]))
        assert decoded == text
        assert fact["value"] in text
    assert imported()[0]["hashes"] == row["hashes"]


def test_import_rejects_wrong_counts_duplicate_indices_and_overlength():
    with pytest.raises(ValueError, match="Expected 2"):
        imported(expected_count=2)
    with pytest.raises(ValueError, match="indices"):
        imported([official_row(), official_row()], expected_count=2)
    row = official_row()
    row["input"] += "x" * 4096
    with pytest.raises(ValueError, match="refusing truncation"):
        imported([row])


def test_import_rejects_missing_target_fact():
    row = official_row()
    row["outputs"] = ["9999999"]
    with pytest.raises(ValueError, match="absent from a fact"):
        imported([row])


def test_multiquery_fraction_and_all_answers_are_distinct():
    score = ruler.score_prediction("Found 1111111 and 2222222", ["1111111", "2222222", "3333333", "4444444"])
    assert score["per_answer_recovered"] == [True, True, False, False]
    assert score["target_recovery"] == .5
    assert score["all_answers_recovered"] is False
    summary = ruler.score_records([{"pred": "1111111", "outputs": ["1111111", "2222222"]},
                                   {"pred": "3333333", "outputs": ["3333333"]}])
    assert summary["official_ruler_score"] == 75
    assert summary["all_answers_rate"] == .5


def test_scoring_matches_pinned_official_metric():
    root = Path(__file__).resolve().parents[2] / "third_party/RULER"
    path = root / "scripts/eval/synthetic/constants.py"
    if not path.exists():
        pytest.skip("Optional upstream checkout absent")
    spec = importlib.util.spec_from_file_location("official_ruler_metric", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    records = [{"pred": "\x001234567\tHELLO\n", "outputs": ["1234567", "hello", "missing"]},
               {"pred": "0001234567000", "outputs": ["1234567"]},
               {"pred": "", "outputs": ["1234567"]}]
    scored = ruler.score_records(records)
    assert scored["official_ruler_score"] == official.string_match_all(
        [x["normalized_output"] for x in scored["per_example"]], [x["outputs"] for x in records])
    assert scored["per_example"][0]["raw_output"] == records[0]["pred"]
    # Official scoring uses substring matching even inside longer strings.
    assert scored["per_example"][1]["target_recovery"] == 1


def test_generator_propagates_failure_and_uses_safe_argv(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    (root / "scripts/data/synthetic").mkdir(parents=True)
    (root / "scripts/data/synthetic/constants.py").write_text(
        "TASKS={'niah': {'tokens_to_generate':128,'template':'{context}', 'answer_prefix':' Answer:'}}")
    (root / "scripts/synthetic.yaml").write_text("niah_single_2:\n  args:\n    type_haystack: essay\n")
    monkeypatch.setattr(ruler, "verify_upstream", lambda path: {"commit": ruler.UPSTREAM_SHA})
    calls = []

    def fail(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(ruler.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        ruler.generate_cell(root, tmp_path / "tokenizer with spaces", CharacterTokenizer(), tmp_path / "out",
                            task="niah_single_2", context_tokens=4096, split="pilot", count=5)
    argv, kwargs = calls[0]
    assert isinstance(argv, list)
    assert kwargs["check"] is True
    assert not kwargs.get("shell", False)
    assert str(tmp_path / "tokenizer with spaces") in argv
    assert not list((tmp_path / "out").rglob("niah_single_2.jsonl"))


def test_empty_predictions_are_counted_and_empty_dataset_rejected():
    assert ruler.score_prediction("", ["1234567"])["target_recovery"] == 0
    with pytest.raises(ValueError):
        ruler.score_records([])


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = imported()
    ruler.write_jsonl(path, rows)
    assert ruler.read_jsonl(path) == rows
    assert not path.with_suffix(".jsonl.tmp").exists()
