"""Synthetic contract tests; no GPU performance values are inferred by these tests."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_optimization", Path(__file__).resolve().parents[1] / "scripts/summarize_optimization.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
from bonsai_bench.analysis import digest


def write_file(path, rows, config):
    fingerprint = digest(config)
    rows = [{**row, "run_fingerprint": fingerprint} for row in rows]
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    path.with_suffix(".jsonl.provenance.json").write_text(json.dumps({"config": config, "run_fingerprint": fingerprint}))


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "comparison"
    directory.mkdir()
    samples = [(task, context, f"pilot/{task}/{context}/0") for task in ("niah_single_2", "niah_multikey_1", "niah_multiquery") for context in (4096,8192,16384)]
    source_dir = tmp_path / "source/bench/bonsai_bench"
    source_dir.mkdir(parents=True)
    sources = {"runner.py": b"# frozen synthetic runner\n", "fused.py": b"# frozen synthetic fused codec\n"}
    source_hashes = {}
    for name, content in sources.items():
        (source_dir / name).write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()
    (tmp_path / "source_manifest.json").write_text(json.dumps({"bench/bonsai_bench/"+name: value for name, value in source_hashes.items()}))
    frozen_samples = [{"split": "pilot", "task": task, "context_tokens": context,
                       "sample_id": sample_id, "model_revision": "frozen",
                       "input_ids": [index+1, 2], "outputs": [str(index+10)]}
                      for index, (task, context, sample_id) in enumerate(samples)]
    pilot_path = tmp_path / "data/ruler/pilot-smoke.jsonl"
    pilot_path.parent.mkdir(parents=True)
    pilot_path.write_text("".join(json.dumps(sample)+"\n" for sample in frozen_samples))
    (tmp_path / "dataset_index.json").write_text(json.dumps({"additional_files": [
        {"path": "data/ruler/pilot-smoke.jsonl", "sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest()}]}))
    memory = {"allocated_bytes": 100, "reserved_bytes": 140, "peak_allocated_bytes": 120,
              "peak_reserved_bytes": 150, "baseline_allocated_bytes": 100,
              "temporary_peak_allocated_bytes": 20, "native_registered_bytes": 0}
    for block in range(3):
        for backend in summary.BACKENDS:
            config = {"codec_backend": backend, "source_sha256": dict(source_hashes),
                      "conditions": [list(c) for c in sorted(summary.EXPECTED)], "model_revision": "frozen",
                      "sample_hashes": [digest(sample) for sample in frozen_samples], "warmup_samples": 9, "transport": "fabric-lib"}
            rows = []
            for i, (task, context, sample_id) in enumerate(samples):
                for condition, precision in sorted(summary.EXPECTED):
                    remote = (condition, precision) in summary.REMOTE
                    latency = (block+1)*100+context/4096*10+i
                    if precision == "int8": latency *= 2
                    if precision == "int4": latency *= 3
                    if backend == summary.BACKENDS[1]: latency *= (2.,.5,1.)[block]
                    codec = condition in ("transfer", "local_roundtrip")
                    rows.append({"kind": "prediction", "sample_id": sample_id, "split": "pilot", "task": task,
                                 "context_tokens": context, "condition": condition, "precision": precision,
                                 "sample_hash": digest(frozen_samples[i]), "model_revision": "frozen", "codec_backend": backend,
                                 "transport": "fabric-lib" if remote else "none", "handoff_ack_ms": latency if remote else None,
                                 "encode_ms": 2. if codec else 0., "decode_ms": 3. if codec else 0.,
                                 "prefill_ms": 20., "continuation_ms": 30., "receiver_prefill_ms": 20. if condition == "text_refill" else 0.,
                                 "wire_bytes": 100 if remote or codec else 0, "payload_bytes": 80 if codec else 0,
                                 "metadata_bytes": 10 if codec else 0, "control_bytes": 10 if remote else 0,
                                 "wire_bytes_scope": "application_frames" if remote else "encoded_size_no_network",
                                 "handoff_memory": dict(memory), "receiver_memory": dict(memory) if remote else {},
                                 "generated_ids": [11,12], "score": {"target_recovery": 1., "all_answers_recovered": True,
                                                                            "per_answer_recovered": [True]}})
            write_file(directory / f"block-{block}-{backend}.jsonl", rows, config)
    return tmp_path


def edit_file(run_dir, mutate_rows=None, mutate_config=None, block=1, backend="triton_packed_v1"):
    path = run_dir / "comparison" / f"block-{block}-{backend}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    config = json.loads(path.with_suffix(".jsonl.provenance.json").read_text())["config"]
    if mutate_rows: mutate_rows(rows)
    if mutate_config: mutate_config(config)
    write_file(path, rows, config)


def test_complete_gate_and_paired_block_aggregation(run_dir):
    files, protocol, cross_pairs = summary.load_and_validate(run_dir)
    report = summary.analyze(files, protocol, cross_pairs)
    assert report["predictions"] == 378 and cross_pairs == 189
    assert all(f["handoff_equivalence"]["exact_token_id_pairs"] == 36 for f in report["inputs"])
    comparison = next(c for c in report["backend_comparisons"] if c["context_tokens"] == 4096 and c["condition"] == "transfer" and c["precision"] == "int8")
    assert comparison["n_pairs"] == 9 and comparison["n_unique_prompts"] == 3
    assert comparison["handoff_candidate_to_baseline_ratio"]["mean"] == pytest.approx(7/6)
    assert comparison["handoff_candidate_to_baseline_ratio"]["p50"] == 1.
    selfbase = next(c for c in report["same_backend_bf16_comparisons"] if c["backend"] == "triton_packed_v1" and c["context_tokens"] == 4096 and c["condition"] == "transfer" and c["precision"] == "int8")
    assert selfbase["handoff_candidate_to_baseline_ratio"]["mean"] == 2.
    variation = next(c for c in report["backend_comparison_block_variation"] if c["context_tokens"] == 4096 and c["condition"] == "transfer" and c["precision"] == "int8")
    assert variation["handoff_candidate_to_baseline_ratio_block_medians"]["values"] == [2.,.5,1.]


def test_ratios_are_computed_per_prompt_before_aggregation():
    def row(latency):
        return {"sample_id": str(latency), "handoff_ack_ms": latency, "encode_ms": 1., "decode_ms": 1.}
    actual = summary.paired_stats([(row(1.),row(2.)),(row(100.),row(50.))])
    assert actual["handoff_candidate_to_baseline_ratio"]["mean"] == 1.25
    assert actual["handoff_candidate_minus_baseline_ms"]["mean"] == -24.5


@pytest.mark.parametrize("change,pattern", [
    ("missing", "63 successful"), ("failure", "no failures"), ("duplicate", "duplicate"),
    ("condition", "seven conditions"), ("backend", "split/backend"), ("nan", "non-finite"),
    ("latency", "positive"), ("memory", "positive"), ("memory_delta", "inconsistent temporary"),
    ("size", "sizes differ"), ("handoff_ids", "generated token IDs differ")])
def test_invalid_inputs_rejected(run_dir, change, pattern):
    def mutate(rows):
        if change == "missing": rows.pop()
        elif change == "failure": rows[0]["kind"] = "failure"
        elif change == "duplicate": rows[0] = dict(rows[1])
        elif change == "condition": rows[0]["condition"] = "unexpected"
        elif change == "backend": rows[0]["codec_backend"] = "wrong"
        elif change == "nan": rows[0]["encode_ms"] = float("nan")
        elif change == "latency": next(r for r in rows if r["condition"] == "transfer")["handoff_ack_ms"] = 0.
        elif change == "memory": rows[0]["handoff_memory"]["baseline_allocated_bytes"] = 0
        elif change == "memory_delta": rows[0]["handoff_memory"]["temporary_peak_allocated_bytes"] = 999
        elif change == "size": next(r for r in rows if r["condition"] == "transfer")["payload_bytes"] += 1
        elif change == "handoff_ids": next(r for r in rows if r["condition"] == "transfer")["generated_ids"] = [999]
    edit_file(run_dir, mutate)
    with pytest.raises(ValueError, match=pattern): summary.load_and_validate(run_dir)


def test_cross_backend_ids_checked_beyond_internal_handoff_pairs(run_dir):
    def mutate(rows):
        sample_id = rows[0]["sample_id"]
        for row in rows:
            if row["sample_id"] == sample_id: row["generated_ids"] = [999]
    edit_file(run_dir, mutate)
    with pytest.raises(ValueError, match="cross-backend generated"):
        summary.load_and_validate(run_dir)


@pytest.mark.parametrize("field,value,pattern", [("codec_backend","pytorch_packed_v1","backend provenance"), ("source_sha256",{},"source_sha256"), ("warmup_samples",1,"incompatible source/config")])
def test_provenance_mismatch_rejected(run_dir, field, value, pattern):
    edit_file(run_dir, mutate_config=lambda c:c.update({field:value}))
    with pytest.raises(ValueError, match=pattern): summary.load_and_validate(run_dir)


def test_same_prompt_ids_and_hashes_required_across_blocks(run_dir):
    def mutate(rows):
        selected = rows[0]["sample_id"]
        for row in rows:
            if row["sample_id"] == selected: row["sample_hash"] = "changed"
    edit_file(run_dir, mutate)
    with pytest.raises(ValueError, match="frozen manifest"):
        summary.load_and_validate(run_dir)


def test_missing_file_and_identical_copied_block_rejected(run_dir):
    directory = run_dir / "comparison"
    first = directory / "block-0-pytorch_packed_v1.jsonl"
    second = directory / "block-1-pytorch_packed_v1.jsonl"
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="duplicate copied block"):
        summary.load_and_validate(run_dir)
    second.unlink()
    with pytest.raises(ValueError, match="missing complete"):
        summary.load_and_validate(run_dir)


def test_cli_writes_json_markdown_only_after_all_checks(run_dir):
    summary.main([str(run_dir)])
    output = run_dir / "analysis"
    assert {p.name for p in output.iterdir()} == {"summary.json","summary.md"}
    report = json.loads((output/"summary.json").read_text())
    assert report["status"] == "PASS" and report["pilot_unique_prompts"] == 9
    text = (output/"summary.md").read_text()
    assert "three unique prompts" in text and "no preservation" in text
    edit_file(run_dir, lambda rows:rows.pop())
    rejected = run_dir / "must-not-exist"
    with pytest.raises(ValueError): summary.main([str(run_dir),"--out",str(rejected)])
    assert not rejected.exists()


@pytest.fixture
def frozen_postprocessors(tmp_path):
    """Load copies outside the repository to exercise snapshot path handling."""
    import shutil
    scripts = tmp_path / "standalone-snapshot/source/scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("summarize_optimization.py", "plot_optimization.py"):
        shutil.copy2(original / name, scripts / name)
    spec = importlib.util.spec_from_file_location("snapshot_summary", scripts / "summarize_optimization.py")
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    plot_spec = importlib.util.spec_from_file_location("snapshot_plot", scripts / "plot_optimization.py")
    plotter = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plotter)
    return copied, plotter


@pytest.mark.parametrize("relative", ["results/analysis-final", "runs/final-20261107"])
@pytest.mark.parametrize("via_symlink", [False, True])
def test_frozen_scripts_reject_actual_accepted_output(run_dir, frozen_postprocessors, relative, via_symlink, monkeypatch):
    copied, plotter = frozen_postprocessors
    repository = Path(__file__).resolve().parents[1]
    protected = repository / relative
    if via_symlink:
        alias = run_dir / "accepted-escape"
        alias.symlink_to(protected, target_is_directory=True)
        protected = alias
    # Neither script may inspect inputs or render before rejecting this output.
    monkeypatch.setattr(copied, "load_and_validate", lambda *_: pytest.fail("validation reached protected output"))
    monkeypatch.setattr(plotter, "load_and_validate", lambda *_: pytest.fail("rendering reached protected output"))
    with pytest.raises(ValueError, match="immutable"):
        copied.main([str(run_dir), "--out", str(protected)])
    with pytest.raises(ValueError, match="immutable"):
        plotter.plot(run_dir, protected / "must-not-write.png")


def test_frozen_scripts_reject_directory_symlink_escape(run_dir, frozen_postprocessors, tmp_path):
    copied, plotter = frozen_postprocessors
    outside = tmp_path.parent / (tmp_path.name + "-outside-artifacts")
    outside.mkdir()
    escape = run_dir / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="within.*run directory"):
        copied.main([str(run_dir), "--out", str(escape / "summary")])
    with pytest.raises(ValueError, match="within.*run directory"):
        plotter.plot(run_dir, escape / "must-not-write.png")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", ["summary.json", "summary.md"])
def test_summary_rejects_individual_file_symlink_escape(run_dir, tmp_path, name):
    outside = tmp_path.parent / (tmp_path.name + "-external-" + name)
    outside.write_text("preserve this artifact")
    destination = run_dir / "analysis"
    destination.mkdir()
    (destination / name).symlink_to(outside)
    with pytest.raises(ValueError, match="within.*run directory"):
        summary.main([str(run_dir)])
    assert outside.read_text() == "preserve this artifact"
    assert {p.name for p in destination.iterdir()} == {name}


def test_guard_keeps_standalone_snapshot_outputs_inside_run(run_dir, frozen_postprocessors, tmp_path):
    copied, _ = frozen_postprocessors
    assert copied.guarded_output(run_dir, run_dir / "analysis/new.json") == (run_dir / "analysis/new.json").resolve()
    with pytest.raises(ValueError, match="within.*run directory"):
        copied.guarded_output(run_dir, tmp_path.parent / (tmp_path.name + "-outside.json"))


def test_coherent_comparison_from_different_source_snapshot_is_rejected(run_dir):
    # All six files agree with each other, but identify a different fused source.
    for block in range(3):
        for backend in summary.BACKENDS:
            def change(config):
                config['source_sha256']['fused.py'] = hashlib.sha256(b'# different implementation\n').hexdigest()
            edit_file(run_dir, mutate_config=change, block=block, backend=backend)
    with pytest.raises(ValueError, match='source_sha256 provenance differs from frozen'):
        summary.load_and_validate(run_dir)


@pytest.mark.parametrize('update_manifest', [False, True])
def test_actual_frozen_source_bytes_are_bound_to_manifest_and_results(run_dir, update_manifest):
    path = run_dir / 'source/bench/bonsai_bench/fused.py'
    path.write_bytes(b'# different frozen implementation\n')
    if update_manifest:
        manifest_path = run_dir / 'source_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['bench/bonsai_bench/fused.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    pattern = 'source_sha256 provenance differs' if update_manifest else 'actual frozen benchmark source differs'
    with pytest.raises(ValueError, match=pattern):
        summary.load_and_validate(run_dir)


@pytest.mark.parametrize('change', ['added_file', 'omitted_provenance_file'])
def test_complete_frozen_benchmark_file_set_is_required(run_dir, change):
    if change == 'added_file':
        (run_dir / 'source/bench/bonsai_bench/untracked.py').write_text('# untracked\n')
        pattern = 'source file set differs'
    else:
        for block in range(3):
            for backend in summary.BACKENDS:
                edit_file(run_dir, mutate_config=lambda config: config['source_sha256'].pop('fused.py'),
                          block=block, backend=backend)
        pattern = 'source_sha256 provenance differs'
    with pytest.raises(ValueError, match=pattern):
        summary.load_and_validate(run_dir)


def test_coherent_comparison_from_different_pilot_inputs_is_rejected(run_dir):
    pilot = [json.loads(line) for line in (run_dir / 'data/ruler/pilot-smoke.jsonl').read_text().splitlines()]
    other_hashes = {sample['sample_id']: digest({**sample, 'outputs': ['different target']}) for sample in pilot}
    for block in range(3):
        for backend in summary.BACKENDS:
            def mutate(rows):
                for row in rows:
                    row['sample_hash'] = other_hashes[row['sample_id']]
            edit_file(run_dir, mutate_rows=mutate,
                      mutate_config=lambda config: config.update(sample_hashes=list(other_hashes.values())),
                      block=block, backend=backend)
    with pytest.raises(ValueError, match='sample hashes differ from the nine frozen pilot inputs'):
        summary.load_and_validate(run_dir)


@pytest.mark.parametrize('update_index', [False, True])
def test_frozen_pilot_bytes_and_canonical_samples_are_verified(run_dir, update_index):
    path = run_dir / 'data/ruler/pilot-smoke.jsonl'
    if update_index:
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        samples[0]['outputs'] = ['different frozen target']
        path.write_text(''.join(json.dumps(sample)+'\n' for sample in samples))
        index_path = run_dir / 'dataset_index.json'
        index = json.loads(index_path.read_text())
        index['additional_files'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        index_path.write_text(json.dumps(index))
    else:
        path.write_bytes(path.read_bytes()+b'\n')
    pattern = 'sample hashes differ' if update_index else 'pilot input bytes differ'
    with pytest.raises(ValueError, match=pattern):
        summary.load_and_validate(run_dir)
