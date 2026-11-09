#!/usr/bin/env python3
"""Strict, descriptive paired analysis of three reference/fused pilot blocks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "bench"))
from bonsai_bench.analysis import digest, finite_tree, identity, load_manifests, load_rows
from check_handoff_equivalence import check, EXPECTED
from summarize_campaign import cell_summary, number, percent, stats

BACKENDS = ("pytorch_packed_v1", "triton_packed_v1")
REMOTE = {("transfer", "bf16"), ("transfer", "int8"), ("transfer", "int4"), ("text_refill", "bf16")}
CELLS = {(task, context) for task in ("niah_single_2", "niah_multikey_1", "niah_multiquery")
         for context in (4096, 8192, 16384)}
MIB = 1024**2


def guarded_output(run_dir, output):
    """Constrain writes even from a standalone snapshot or through symlinks."""
    run = Path(run_dir).resolve()
    destination = Path(output).resolve()
    repositories = set()
    for start in (SCRIPTS, run, destination, Path.cwd().resolve()):
        for ancestor in (start, *start.parents):
            if (ancestor / ".git").exists():
                repositories.add(ancestor)
                break
    for repository in repositories:
        for relative in ("results/analysis-final", "runs/final-20261107"):
            if destination.is_relative_to((repository / relative).resolve()):
                raise ValueError("accepted final artifacts are immutable; choose the new optimization run directory")
    if not destination.is_relative_to(run):
        raise ValueError("optimization outputs must remain within the supplied run directory; symlink escapes are forbidden")
    return destination


def positive(value, label, allow_zero=False):
    if type(value) not in (int, float) or not (value >= 0 if allow_zero else value > 0):
        raise ValueError(f"{label}: require {'nonnegative' if allow_zero else 'positive'} finite measurement")


def validate_measurements(row):
    finite_tree(row)
    remote = (row["condition"], row["precision"]) in REMOTE
    for field in ("encode_ms", "decode_ms"):
        positive(row.get(field), field, allow_zero=row["condition"] != "transfer" and row["condition"] != "local_roundtrip")
    if remote:
        positive(row.get("handoff_ack_ms"), "handoff_ack_ms")
    elif row.get("handoff_ack_ms") is not None:
        raise ValueError("local condition must not claim network handoff timing")
    for field in ("prefill_ms", "continuation_ms"):
        positive(row.get(field), field)
    if row["condition"] == "text_refill":
        positive(row.get("receiver_prefill_ms"), "receiver_prefill_ms")
    for field in ("wire_bytes", "payload_bytes", "metadata_bytes", "control_bytes"):
        positive(row.get(field), field, allow_zero=True)
    if remote:
        positive(row["wire_bytes"], "remote wire_bytes")
    for location in ("handoff_memory", "receiver_memory") if remote else ("handoff_memory",):
        memory = row.get(location, {})
        for field in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "baseline_allocated_bytes"):
            positive(memory.get(field), f"{location}.{field}")
        for field in ("temporary_peak_allocated_bytes", "native_registered_bytes"):
            positive(memory.get(field), f"{location}.{field}", allow_zero=True)
        if memory["peak_allocated_bytes"] < memory["allocated_bytes"]:
            raise ValueError(f"{location}: peak allocation below current allocation")
        if memory["temporary_peak_allocated_bytes"] != max(0, memory["peak_allocated_bytes"]-memory["baseline_allocated_bytes"]):
            raise ValueError(f"{location}: inconsistent temporary peak above baseline")
    if not row.get("wire_bytes_scope"):
        raise ValueError("missing byte scope")


def frozen_inputs(run_dir):
    """Bind analysis to the snapshot's actual source bytes and audited inputs."""
    run = Path(run_dir)
    manifest_path = run / "source_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    prefix = "bench/bonsai_bench/"
    source_hashes = {key[len(prefix):]: value for key, value in source_manifest.items()
                     if key.startswith(prefix) and key.endswith(".py") and "/" not in key[len(prefix):]}
    source_dir = run / "source/bench/bonsai_bench"
    if not source_hashes or {path.name for path in source_dir.glob("*.py")} != set(source_hashes):
        raise ValueError("frozen benchmark source file set differs from source_manifest.json")
    for name, expected in source_hashes.items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"actual frozen benchmark source differs from source_manifest.json: {name}")
    pilot_path = run / "data/ruler/pilot-smoke.jsonl"
    pilot_hash = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    dataset = json.loads((run / "dataset_index.json").read_text())
    entries = [entry for entry in dataset.get("additional_files", [])
               if entry.get("path") == "data/ruler/pilot-smoke.jsonl"]
    if len(entries) != 1 or entries[0].get("sha256") != pilot_hash:
        raise ValueError("frozen pilot input bytes differ from dataset_index.json")
    samples = load_manifests([pilot_path])
    if len(samples) != 9 or {(key[1], key[2]) for key in samples} != CELLS or any(key[0] != "pilot" for key in samples):
        raise ValueError("frozen pilot manifest must contain nine pilot prompts, one per task/context cell")
    sample_hashes = {digest(sample) for sample in samples.values()}
    return source_hashes, samples, sample_hashes, {
        "frozen_source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frozen_pilot_manifest_sha256": pilot_hash}


def load_and_validate(run_dir):
    source_hashes, frozen_samples, frozen_sample_hashes, frozen_binding = frozen_inputs(run_dir)
    files, reference_lineage, reference_protocol = [], None, None
    input_hashes = defaultdict(set)
    for block in range(3):
        for backend in BACKENDS:
            path = Path(run_dir) / "comparison" / f"block-{block}-{backend}.jsonl"
            sidecar = path.with_suffix(".jsonl.provenance.json")
            if not path.is_file() or not sidecar.is_file():
                raise ValueError(f"missing complete comparison input/provenance: {path}")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            if before in input_hashes[backend]:
                raise ValueError("duplicate copied block input; three separately measured blocks are required")
            input_hashes[backend].add(before)
            provenance = json.loads(sidecar.read_text())
            config = provenance.get("config", {})
            if config.get("codec_backend") != backend:
                raise ValueError(f"backend provenance does not match filename: {path}")
            if not config.get("source_sha256"):
                raise ValueError("comparison requires frozen source_sha256 provenance")
            if config["source_sha256"] != source_hashes:
                raise ValueError("source_sha256 provenance differs from frozen source_manifest.json")
            if len(config.get("sample_hashes", [])) != 9 or set(config["sample_hashes"]) != frozen_sample_hashes:
                raise ValueError("provenance sample hashes differ from the nine frozen pilot inputs")
            if {tuple(v) for v in config.get("conditions", [])} != EXPECTED:
                raise ValueError("provenance must select exactly the seven expected conditions")
            protocol = {k: v for k, v in config.items() if k not in ("codec_backend", "sample_hashes")}
            if reference_protocol is not None and protocol != reference_protocol:
                raise ValueError("comparison files have incompatible source/config protocols")
            reference_protocol = protocol
            rows = load_rows([path], manifests=frozen_samples)
            if len(rows) != 63 or any(r["kind"] != "prediction" for r in rows):
                raise ValueError(f"{path}: require 63 successful rows and no failures")
            prompts = defaultdict(list)
            for row in rows:
                if row["split"] != "pilot" or row["codec_backend"] != backend:
                    raise ValueError("prediction split/backend does not match pilot provenance")
                if row["model_revision"] != config.get("model_revision"):
                    raise ValueError("model revision differs from provenance")
                expected_transport = "fabric-lib" if (row["condition"], row["precision"]) in REMOTE else "none"
                if row["transport"] != expected_transport:
                    raise ValueError("comparison requires native fabric transfer and explicitly local baselines")
                validate_measurements(row)
                prompts[identity(row)].append(row)
            if len(prompts) != 9 or {(k[1], k[2]) for k in prompts} != CELLS:
                raise ValueError("comparison requires exactly nine pilot prompts, one per task/context cell")
            lineage = {}
            for key, selected in prompts.items():
                if len(selected) != 7 or {(r["condition"], r["precision"]) for r in selected} != EXPECTED:
                    raise ValueError("every pilot prompt requires all seven conditions exactly once")
                signatures = {(r["sample_hash"], r["model_revision"]) for r in selected}
                if len(signatures) != 1:
                    raise ValueError("prompt lineage mismatch across conditions")
                lineage[key] = next(iter(signatures))
            if reference_lineage is not None and lineage != reference_lineage:
                raise ValueError("all blocks/backends must use the same nine pilot identities, hashes and model revisions")
            reference_lineage = lineage
            if set(config.get("sample_hashes", [])) != {r["sample_hash"] for r in rows} or len(config.get("sample_hashes", [])) != 9:
                raise ValueError("provenance sample hash selection does not match the nine pilot prompts")
            equivalence = check([path], expected_prompts=9)
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise ValueError("comparison input changed while being summarized")
            files.append({"block": block, "backend": backend, "path": str(path), "sha256": before, **frozen_binding,
                          "run_fingerprint": provenance["run_fingerprint"], "handoff_equivalence": equivalence, "rows": rows})
    cross_pairs = 0
    for block in range(3):
        reference, candidate = [next(f for f in files if f["block"] == block and f["backend"] == b) for b in BACKENDS]
        lookup = {(identity(r), r["condition"], r["precision"]): r for r in reference["rows"]}
        for row in candidate["rows"]:
            other = lookup[(identity(row), row["condition"], row["precision"])]
            if row["generated_ids"] != other["generated_ids"]:
                raise ValueError(f"cross-backend generated token IDs differ: block {block}, {row['sample_id']}, {row['condition']}:{row['precision']}")
            if (row["payload_bytes"], row["metadata_bytes"]) != (other["payload_bytes"], other["metadata_bytes"]):
                raise ValueError("encoded payload/metadata sizes differ between backends")
            cross_pairs += 1
    return files, reference_protocol, cross_pairs


def summarize_cell(rows):
    result = cell_summary(rows)
    result["n_unique_prompts"] = len({r["sample_id"] for r in rows})
    result["n_blocks"] = len({r["block"] for r in rows})
    remote = rows[0]["transport"] == "fabric-lib"
    result["remaining_handoff_ms"] = stats((r["handoff_ack_ms"]-r["encode_ms"]-r["decode_ms"] for r in rows), len(rows)) if remote and rows[0]["condition"] == "transfer" else None
    return result


def paired_stats(pairs):
    if not pairs:
        raise ValueError("missing exact paired observations")
    return {"n_pairs": len(pairs), "n_unique_prompts": len({r["sample_id"] for _, r in pairs}),
            "handoff_candidate_to_baseline_ratio": stats(r["handoff_ack_ms"]/b["handoff_ack_ms"] for b, r in pairs),
            "handoff_candidate_minus_baseline_ms": stats(r["handoff_ack_ms"]-b["handoff_ack_ms"] for b, r in pairs),
            "encode_candidate_minus_baseline_ms": stats(r["encode_ms"]-b["encode_ms"] for b, r in pairs),
            "decode_candidate_minus_baseline_ms": stats(r["decode_ms"]-b["decode_ms"] for b, r in pairs)}


def block_ranges(records, keys):
    grouped = defaultdict(list)
    for r in records:
        grouped[tuple(r[k] for k in keys)].append(r)
    output = []
    for key, selected in sorted(grouped.items()):
        item = dict(zip(keys, key))
        item["blocks"] = [r["block"] for r in selected]
        for name in ("handoff_ms", "encode_ms", "decode_ms", "handoff_candidate_to_baseline_ratio", "handoff_candidate_minus_baseline_ms"):
            medians = [r[name]["p50"] for r in selected if name in r and r[name]["p50"] is not None]
            item[name + "_block_medians"] = {"values": medians, "min": min(medians) if medians else None, "max": max(medians) if medians else None}
        output.append(item)
    return output


def analyze(files, protocol, cross_pairs):
    rows = [{**r, "block": f["block"], "backend": f["backend"]} for f in files for r in f["rows"]]
    report = {"schema_version": 1, "status": "PASS", "pilot_unique_prompts": 9, "blocks": 3, "predictions": len(rows),
              "cross_backend_exact_token_pairs": cross_pairs,
              "interpretation": "Exploratory nine-prompt pilot repeated in three blocks; repeated observations are not additional unique prompts. Descriptive p50/p95 and block ranges only; no preservation or population speedup claim.",
              "ratio_definition": "candidate handoff_ms / same-block same-prompt baseline handoff_ms; below one means lower measured latency",
              "difference_definition": "candidate minus baseline milliseconds",
              "byte_compatibility_scope": "Per-pair payload and metadata lengths verified; equality of actual encoded bytes requires the separate cache-integrity validation. Application frame counts can differ with backend labels.",
              "memory_scope": "Peak allocation above each worker's measured handoff baseline; receiver includes reconstructed cache residency, not only transient workspace.",
              "remaining_handoff_scope": "Per-row total minus encode/decode; includes registration, allocation, synchronization, control protocol and transport; not pure network time.",
              "protocol": protocol, "inputs": [{k: v for k, v in f.items() if k != "rows"} for f in files],
              "by_context": [], "by_block_context": [], "backend_comparisons": [], "backend_comparisons_by_block": [],
              "same_backend_bf16_comparisons": []}
    for include_block, output in ((False, "by_context"), (True, "by_block_context")):
        groups = defaultdict(list)
        for r in rows:
            key = (r["backend"], r["context_tokens"], r["condition"], r["precision"])
            groups[(r["block"], *key) if include_block else key].append(r)
        names = (["block"] if include_block else []) + ["backend", "context_tokens", "condition", "precision"]
        for key, selected in sorted(groups.items()):
            report[output].append({**dict(zip(names, key)), **summarize_cell(selected)})
    lookup = {(r["block"], r["backend"], identity(r), r["condition"], r["precision"]): r for r in rows}
    for include_block, output in ((False, "backend_comparisons"), (True, "backend_comparisons_by_block")):
        groups = defaultdict(list)
        for row in rows:
            if row["backend"] != BACKENDS[1] or (row["condition"], row["precision"]) not in REMOTE:
                continue
            baseline = lookup[(row["block"], BACKENDS[0], identity(row), row["condition"], row["precision"])]
            key = (row["context_tokens"], row["condition"], row["precision"])
            groups[(row["block"], *key) if include_block else key].append((baseline, row))
        names = (["block"] if include_block else []) + ["context_tokens", "condition", "precision"]
        for key, pairs in sorted(groups.items()):
            report[output].append({**dict(zip(names, key)), "baseline_backend": BACKENDS[0], "candidate_backend": BACKENDS[1], **paired_stats(pairs)})
    groups = defaultdict(list)
    for row in rows:
        if (row["condition"], row["precision"]) not in REMOTE:
            continue
        baseline = lookup[(row["block"], row["backend"], identity(row), "transfer", "bf16")]
        groups[(row["backend"], row["context_tokens"], row["condition"], row["precision"])].append((baseline, row))
    for key, pairs in sorted(groups.items()):
        report["same_backend_bf16_comparisons"].append({**dict(zip(["backend", "context_tokens", "condition", "precision"], key)), "baseline_condition": "transfer:bf16", **paired_stats(pairs)})
    report["condition_block_variation"] = block_ranges(report["by_block_context"], ["backend", "context_tokens", "condition", "precision"])
    report["backend_comparison_block_variation"] = block_ranges(report["backend_comparisons_by_block"], ["context_tokens", "condition", "precision"])
    finite_tree(report)
    return report


def markdown(report):
    lines = ["# Reference versus fused codec pilot", "", report["interpretation"], "",
             f"Accepted {report['predictions']} predictions across six files; {report['cross_backend_exact_token_pairs']} exact cross-backend token comparisons. Each file also passed all 36 native/local handoff equivalence pairs.", "",
             report["ratio_definition"] + ". Differences are candidate minus baseline.", "", report["byte_compatibility_scope"], "", report["memory_scope"], "", report["remaining_handoff_scope"], "",
             "## Measured conditions", "", "Each context/backend/condition has nine repeated observations from three unique prompts across three blocks. Local handoff latency is unavailable; local bytes mean encoded size only. Remote bytes are application handoff frames, excluding network headers.", "",
             "|Context|Backend|Condition|Handoff p50/p95 ms|Encode p50/p95 ms|Decode p50/p95 ms|Mean frame MiB|Sender/receiver extra peak MiB p50|Target/all-answer %|", "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in report["by_context"]:
        sender = c["sender_handoff_memory"]["temporary_peak_allocated_mib"]["p50"]
        receiver = c["receiver_handoff_memory"]["temporary_peak_allocated_mib"]["p50"]
        pair = lambda metric: f"{number(c[metric]['p50'])}/{number(c[metric]['p95'])}"
        lines.append(f"|{c['context_tokens']}|{c['backend']}|{c['condition']}:{c['precision']}|{pair('handoff_ms')}|{pair('encode_ms')}|{pair('decode_ms')}|{number(c['wire_mib']['mean'])}|{number(sender)}/{number(receiver)}|{percent(c['target_recovery']['mean'])}/{percent(c['all_answers_rate']['mean'])}|")
    for title, field in (("Fused versus matching reference backend", "backend_comparisons"), ("Condition versus BF16 transfer in the same backend", "same_backend_bf16_comparisons")):
        lines += ["", "## " + title, "", "Ratios and differences are computed for each matched block/prompt before aggregation.", "", "|Context|Candidate|Condition|Pairs/unique prompts|Handoff ratio p50/p95|Handoff delta p50/p95 ms|", "|---:|---|---|---:|---:|---:|"]
        for c in report[field]:
            ratio, delta = c["handoff_candidate_to_baseline_ratio"], c["handoff_candidate_minus_baseline_ms"]
            lines.append(f"|{c['context_tokens']}|{c.get('candidate_backend', c.get('backend'))}|{c['condition']}:{c['precision']}|{c['n_pairs']}/{c['n_unique_prompts']}|{number(ratio['p50'],3)}/{number(ratio['p95'],3)}|{number(delta['p50'])}/{number(delta['p95'])}|")
    lines += ["", "## Block variation", "", "Three independent process blocks provide descriptive variation, not a strong confidence interval. This table shows the median paired fused/reference ratio within each block and its range.", "", "|Context|Condition|Block median ratios|Range|", "|---:|---|---|---|"]
    for c in report["backend_comparison_block_variation"]:
        values = c["handoff_candidate_to_baseline_ratio_block_medians"]
        lines.append(f"|{c['context_tokens']}|{c['condition']}:{c['precision']}|{', '.join(number(v,3) for v in values['values'])}|{number(values['min'],3)}–{number(values['max'],3)}|")
    lines += ["", "Complete per-block condition medians/ranges, phase differences, metric counts and memory distributions are retained in summary.json. Old final results were not incorporated into these exploratory comparisons.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="defaults to RUN_DIR/analysis")
    args = parser.parse_args(argv)
    output = guarded_output(args.run_dir, args.out or args.run_dir / "analysis")
    targets = {name: guarded_output(args.run_dir, output / name) for name in ("summary.json", "summary.md")}
    files, protocol, cross_pairs = load_and_validate(args.run_dir)
    report = analyze(files, protocol, cross_pairs)
    output.mkdir(parents=True, exist_ok=True)
    targets["summary.json"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    targets["summary.md"].write_text(markdown(report))
    print(json.dumps({"status": "PASS", "out": str(output), "predictions": report["predictions"], "cross_backend_exact_token_pairs": cross_pairs}))


if __name__ == "__main__":
    main()
