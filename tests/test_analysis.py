import json

import numpy as np
import pytest

from bonsai_bench.analysis import (analyze, digest, load_rows, paired_interval,
                                    verify_complete_final)
from bonsai_bench.plots import plot_report


def row(sample="pilot/example/0", condition="local_bf16", precision="bf16", answers=(True,), **kwargs):
    return {"kind": "prediction", "split": "pilot", "task": "niah_single_2", "context_tokens": 4096,
            "sample_id": sample, "condition": condition, "precision": precision, "sample_hash": "samplehash",
            "model_revision": "frozen", "run_fingerprint": "runhash", "transport": "none",
            "encode_ms": 1., "decode_ms": 2., "handoff_ack_ms": None, "wire_bytes": 100,
            "score": {"target_recovery": float(np.mean(answers)), "all_answers_recovered": all(answers),
                      "per_answer_recovered": list(answers)}, **kwargs}


def write_rows(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_missing_pairs_and_failures_are_visible(tmp_path):
    base = row()
    candidate = row(condition="local_roundtrip", precision="int4")
    missing = row(sample="pilot/example/1", condition="local_roundtrip", precision="int4")
    failed = row(sample="pilot/example/2", condition="local_roundtrip", precision="int4", kind="failure")
    failed.pop("score")
    report = analyze(load_rows([write_rows(tmp_path, [base, candidate, missing, failed])]), resamples=100)
    group = report["strata"][0]
    comparison = group["comparisons"][0]
    assert group["failure_rows"] == 1
    assert comparison["n_pairs"] == 1 and comparison["attempted_union"] == 3
    assert comparison["missing_baseline_rows"] == 2
    assert not comparison["complete_pairs"] and not comparison["margin_supported"]


def test_duplicate_rows_and_retries_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_rows([write_rows(tmp_path, [row(), row()])])
    with pytest.raises(ValueError, match="duplicate"):
        load_rows([write_rows(tmp_path, [row(), row(kind="failure")])])


@pytest.mark.parametrize("field,value", [("sample_hash", "other"), ("model_revision", "other")])
def test_mismatched_provenance_rejected(tmp_path, field, value):
    other = row(condition="local_roundtrip", precision="int8", **{field: value})
    with pytest.raises(ValueError, match="mismatched"):
        load_rows([write_rows(tmp_path, [row(), other])])


def test_mismatched_split_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_rows([write_rows(tmp_path, [row(split="final")])])


def test_manifest_hash_must_match(tmp_path):
    from bonsai_bench.analysis import identity
    sample = {"sample_id": "pilot/example/0", "task": "niah_single_2", "context_tokens": 4096,
              "split": "pilot", "model_revision": "frozen"}
    good = row(sample_hash=digest(sample))
    assert len(load_rows([write_rows(tmp_path, [good])], {identity(good): sample})) == 1
    with pytest.raises(ValueError, match="frozen manifest"):
        load_rows([write_rows(tmp_path, [row()])], {identity(good): sample})


def test_nan_rejected_before_plotting(tmp_path):
    bad = row(encode_ms=float("nan"))
    with pytest.raises(ValueError, match="non-finite.*encode_ms"):
        load_rows([write_rows(tmp_path, [bad])])
    with pytest.raises(ValueError, match="non-finite"):
        plot_report({"strata": [], "bad": float("nan")}, tmp_path)
    assert not list(tmp_path.glob("*.png"))


def test_zero_discordance_is_not_evidence_of_two_point_margin():
    result = paired_interval(np.zeros(100), resamples=100)
    assert result["bootstrap_ci95"] == [0., 0.]
    assert result["degradation_probability_upper"] > .029
    assert result["conservative_difference_lower"] < -.02


def test_partial_recovery_is_prompt_paired_not_binarized(tmp_path):
    base = row(answers=(True, True, True, True))
    candidate = row(condition="local_roundtrip", precision="int4", answers=(True, True, False, True))
    report = analyze(load_rows([write_rows(tmp_path, [base, candidate])]), resamples=100)
    cell = report["strata"][0]["comparisons"][0]
    assert cell["difference"] == -.25
    assert cell["all_answers_difference"] == -1
    assert cell["n_pairs"] == 1


def test_run_and_transport_strata_never_pool(tmp_path):
    rows = [row(), row(run_fingerprint="second"),
            row(condition="transfer", transport="tcp_diagnostic"),
            row(condition="transfer", transport="fabric-lib")]
    report = analyze(load_rows([write_rows(tmp_path, rows)]), resamples=100)
    assert len(report["strata"]) == 4


def test_png_output_and_no_local_handoff_plot(tmp_path):
    rows = [row(), row(condition="local_roundtrip", precision="int8")]
    report = analyze(rows, resamples=100)
    paths = plot_report(report, tmp_path)
    assert len(paths) == 4
    assert all(p.endswith(".png") for p in paths)
    for path in paths:
        assert open(path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    assert not list(tmp_path.rglob("latency-vs-context.png"))


def test_complete_final_rejects_partial_and_wrong_transport():
    with pytest.raises(ValueError, match="final split"):
        verify_complete_final([row()])
    with pytest.raises(ValueError, match="native"):
        verify_complete_final([row(split="final", sample="final/example/0", condition="transfer", transport="tcp_diagnostic")])
    with pytest.raises(ValueError, match="2700"):
        verify_complete_final([row(split="final", sample="final/example/0", condition="transfer", transport="fabric-lib")])


def test_complete_final_exact_matrix():
    rows = []
    for task in ("niah_single_2", "niah_multikey_1", "niah_multiquery"):
        for context in (4096, 8192, 16384):
            for i in range(100):
                for precision in ("bf16", "int8", "int4"):
                    rows.append(row(sample=f"final/{task}/{context}/{i}", split="final", task=task,
                                    context_tokens=context, condition="transfer", precision=precision, transport="fabric-lib"))
    assert verify_complete_final(rows)
    rows[-1]["sample_id"] = "final/unpaired"
    with pytest.raises(ValueError, match="900"):
        verify_complete_final(rows)


def test_explicit_array_aggregation_checks_all_non_sample_config(tmp_path):
    paths = []
    for i in range(2):
        config = {"source_sha256": {"runner.py": "frozen-source"}, "model_revision": "frozen",
                  "transport": "none", "sample_hashes": [str(i)]}
        fingerprint = digest(config)
        path = tmp_path / f"cell-{i}.jsonl"
        path.write_text(json.dumps(row(sample=f"pilot/example/{i}", run_fingerprint=fingerprint)) + "\n")
        path.with_suffix(".jsonl.provenance.json").write_text(json.dumps({"config": config, "run_fingerprint": fingerprint}))
        paths.append(path)
    loaded = load_rows(paths)
    assert len(analyze(loaded, resamples=100)["strata"]) == 2
    combined = analyze(loaded, aggregate_compatible_runs=True, resamples=100)["strata"]
    assert len(combined) == 1 and len(combined[0]["source_run_fingerprints"]) == 2
    loaded[1]["protocol_fingerprint"] = "other-source"
    assert len(analyze(loaded, aggregate_compatible_runs=True, resamples=100)["strata"]) == 2
    loaded[1]["protocol_fingerprint"] = loaded[0]["protocol_fingerprint"]
    loaded[1]["sample_id"] = loaded[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate cell"):
        analyze(loaded, aggregate_compatible_runs=True, resamples=100)
