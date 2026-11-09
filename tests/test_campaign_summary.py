"""Check paired arithmetic, handoff memory scope, and block isolation."""
import importlib.util
import copy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_campaign", Path(__file__).resolve().parents[1] / "scripts/summarize_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def row(sample="pilot/example/0", condition="transfer", precision="bf16", latency=10., **kwargs):
    return {"kind": "prediction", "sample_id": sample, "split": "pilot", "task": "niah_single_2",
            "context_tokens": 4096, "condition": condition, "precision": precision, "transport": "fabric-lib",
            "sample_hash": sample, "model_revision": "frozen", "run_fingerprint": "run",
            "protocol_fingerprint": "same-source-protocol", "handoff_ack_ms": latency, "wire_bytes": 1048576,
            "wire_bytes_scope": "application frames", "score": {"target_recovery": 1., "all_answers_recovered": True},
            "handoff_memory": {"temporary_peak_allocated_bytes": 1048576},
            "receiver_memory": {"temporary_peak_allocated_bytes": 2097152},
            "memory": {"peak_allocated_bytes": 9999999999}, **kwargs}


def test_ratio_is_mean_of_pairs_not_ratio_of_means():
    rows = [row(latency=1.), row(precision="int8", latency=2.),
            row(sample="pilot/example/1", latency=100.),
            row(sample="pilot/example/1", precision="int8", latency=50.)]
    paired = campaign.paired_latency(rows, "transfer", "int8", 4096, "fabric-lib")
    assert paired["n_pairs"] == 2
    assert paired["candidate_to_bf16_ratio"]["mean"] == 1.25
    assert paired["candidate_minus_bf16_ms"]["mean"] == -24.5


def test_failed_and_missing_pairs_not_silently_imputed():
    rows = [row(), row(precision="int4", kind="failure"),
            row(sample="pilot/example/1", precision="int4")]
    paired = campaign.paired_latency(rows, "transfer", "int4", 4096, "fabric-lib")
    assert paired["n_pairs"] == 0 and paired["unpaired_or_unavailable"] == 2
    assert paired["candidate_to_bf16_ratio"]["mean"] is None


def test_memory_uses_handoff_scope_and_missing_is_not_zero():
    cell = campaign.cell_summary([row(), row(sample="pilot/example/1", handoff_memory={})])
    sender = cell["sender_handoff_memory"]["temporary_peak_allocated_mib"]
    receiver = cell["receiver_handoff_memory"]["temporary_peak_allocated_mib"]
    assert sender["mean"] == 1. and sender["missing"] == 1
    assert receiver["mean"] == 2. and receiver["n"] == 2
    assert cell["wire_mib"]["mean"] == 1.


def test_timing_blocks_remain_individual_observations():
    blocks = []
    for i, latency in enumerate((10., 20., 40.)):
        rows = [row(latency=latency), row(precision="int8", latency=latency/2)]
        blocks.append(campaign.summarize_rows(rows, f"Timing block {i+1}"))
    variation = campaign.timing_variation(blocks)
    int8 = next(c for c in variation if c["precision"] == "int8")
    assert int8["block_handoff_p50_ms"] == {"n_blocks": 3, "mean": pytest.approx(35/3), "min": 5., "max": 20.}
    assert int8["block_paired_ratio_p50"]["min"] == .5
    assert all(c["strata"][0]["total_rows"] == 2 for c in blocks)


def test_duplicate_aggregated_rows_nan_and_zero_latency_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        campaign.summarize_rows([row(), row(run_fingerprint="another-run")], "test")
    with pytest.raises(ValueError, match="non-finite"):
        campaign.summarize_rows([row(latency=float("nan"))], "test")
    with pytest.raises(ValueError, match="strictly positive"):
        campaign.paired_latency([row(latency=0.), row(precision="int8")], "transfer", "int8", 4096, "fabric-lib")


def test_transport_never_cross_paired():
    paired = campaign.paired_latency([row(), row(precision="int8", transport="tcp_diagnostic")], "transfer", "int8", 4096, "tcp_diagnostic")
    assert paired["n_pairs"] == 0 and paired["unpaired_or_unavailable"] == 1


def test_render_unavailable_values_without_nan():
    block = campaign.summarize_rows([row(handoff_ack_ms=None, receiver_memory={})], "Timing block 1")
    report = {"interpretation": "Descriptive only", "campaigns": [], "timing_blocks": [block], "timing_block_variation": []}
    text = campaign.markdown(report)
    assert "unavailable" in text and "nan" not in text.lower()


def complete_timing_blocks():
    blocks = []
    for block in range(3):
        rows = []
        for task in ("niah_single_2", "niah_multikey_1", "niah_multiquery"):
            for context in (4096, 8192, 16384):
                for condition, precision in sorted(campaign.REMOTE):
                    rows.append(row(sample=f"pilot/{task}/{context}/0", task=task, context_tokens=context,
                                    condition=condition, precision=precision, latency=10.+block))
        blocks.append(rows)
    return blocks


def test_complete_timing_gate_accepts_three_complete_blocks():
    assert campaign.verify_complete_timing(complete_timing_blocks())


def test_complete_timing_gate_rejects_partial_or_failed_blocks():
    blocks = complete_timing_blocks()
    with pytest.raises(ValueError, match="exactly 3"):
        campaign.verify_complete_timing(blocks[:2])
    blocks[1].pop()
    with pytest.raises(ValueError, match="36 successful"):
        campaign.verify_complete_timing(blocks)
    blocks = complete_timing_blocks()
    blocks[1][0]["kind"] = "failure"
    with pytest.raises(ValueError, match="no failures"):
        campaign.verify_complete_timing(blocks)


@pytest.mark.parametrize("change,pattern", [
    ("duplicate", "duplicate"), ("condition", "all 4"), ("identity", "same 9"),
    ("hash", "same 9"), ("revision", "same 9"), ("protocol", "incompatible"),
    ("missing_protocol", "provenance"), ("transport", "native"), ("latency", "measured handoff")])
def test_complete_timing_gate_rejects_incompatible_blocks(change, pattern):
    blocks = complete_timing_blocks()
    if change == "duplicate":
        blocks[1][0] = copy.deepcopy(blocks[1][1])
    elif change == "condition":
        blocks[1][0]["condition"] = "local_roundtrip"
    elif change in ("identity", "hash", "revision"):
        first_id = blocks[1][0]["sample_id"]
        field = {"identity": "sample_id", "hash": "sample_hash", "revision": "model_revision"}[change]
        for r in blocks[1]:
            if r["sample_id"] == first_id:
                r[field] = "pilot/changed" if field == "sample_id" else "changed"
    elif change == "protocol":
        for r in blocks[1]:
            r["protocol_fingerprint"] = "different-source"
    elif change == "missing_protocol":
        blocks[1][0].pop("protocol_fingerprint")
    elif change == "transport":
        blocks[1][0]["transport"] = "tcp_diagnostic"
    elif change == "latency":
        blocks[1][0]["handoff_ack_ms"] = None
    with pytest.raises(ValueError, match=pattern):
        campaign.verify_complete_timing(blocks)


def test_complete_timing_gate_rejects_reused_files_and_identical_blocks(tmp_path):
    sections = [{"paths": [str(tmp_path / f"block-{i}.jsonl")], "source_sha256": {"input": str(i)}} for i in range(3)]
    blocks = complete_timing_blocks()
    assert campaign.verify_complete_timing(blocks, sections)
    sections[1]["paths"] = sections[0]["paths"]
    with pytest.raises(ValueError, match="repeated file"):
        campaign.verify_complete_timing(blocks, sections)
    sections[1]["paths"] = [str(tmp_path / "other.jsonl")]
    sections[1]["source_sha256"] = sections[0]["source_sha256"]
    with pytest.raises(ValueError, match="identical input bytes"):
        campaign.verify_complete_timing(blocks, sections)


def test_cli_strict_timing_rejects_before_writing_outputs(tmp_path, monkeypatch):
    blocks = complete_timing_blocks()
    blocks[2].pop()
    def fake_load(paths, label, expected_split):
        i = int(paths[0])
        return {"paths": paths, "source_sha256": {paths[0]: str(i)}, "strata": []}, blocks[i]
    monkeypatch.setattr(campaign, "load_section", fake_load)
    out = tmp_path / "unwritten"
    with pytest.raises(ValueError, match="36 successful"):
        campaign.main(["--timing-block", "0", "--timing-block", "1", "--timing-block", "2", "--require-complete-timing", "--out", str(out)])
    assert not out.exists()
