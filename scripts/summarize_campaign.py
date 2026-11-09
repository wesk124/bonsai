#!/usr/bin/env python3
"""Read-only measured campaign tables; final, pilot and timing blocks stay separate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bonsai_bench.analysis import finite_tree, identity, load_rows, verify_complete_final

MIB = 1024 ** 2
REMOTE = {("transfer", "bf16"), ("transfer", "int8"), ("transfer", "int4"), ("text_refill", "bf16")}


def stats(values, expected=None):
    values = list(values)
    if any(v is None for v in values):
        raise ValueError("null metric must be counted as unavailable, not passed to stats")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("non-finite measured series")
    result = {"n": len(array), "missing": None if expected is None else expected-len(array)}
    if len(array):
        result.update(mean=float(array.mean()), p50=float(np.quantile(array, .5)),
                      p95=float(np.quantile(array, .95)), min=float(array.min()), max=float(array.max()))
    else:
        result.update(mean=None, p50=None, p95=None, min=None, max=None)
    return result


def measured(rows, key, scale=1.):
    return stats((r[key]/scale for r in rows if r.get(key) is not None), len(rows))


def memory(rows, key):
    result = {}
    for field in ("temporary_peak_allocated_bytes", "baseline_allocated_bytes", "peak_allocated_bytes", "native_registered_bytes"):
        result[field.removesuffix("_bytes") + "_mib"] = stats(
            (r[key][field]/MIB for r in rows if r.get(key, {}).get(field) is not None), len(rows))
    return result


def cell_summary(rows):
    good = [r for r in rows if r["kind"] == "prediction"]
    result = {"attempted_prompts": len(rows), "predictions": len(good), "execution_failures": len(rows)-len(good),
              "task_counts": dict(sorted((t, sum(r["task"] == t for r in rows)) for t in {r["task"] for r in rows})),
              "target_recovery": stats((r["score"]["target_recovery"] for r in good), len(rows)),
              "all_answers_rate": stats((float(r["score"]["all_answers_recovered"]) for r in good), len(rows)),
              "wire_mib": measured(good, "wire_bytes", MIB),
              "handoff_ms": measured(good, "handoff_ack_ms"),
              "encode_ms": measured(good, "encode_ms"), "decode_ms": measured(good, "decode_ms"),
              "receiver_prefill_ms": measured(good, "receiver_prefill_ms"),
              "sender_handoff_memory": memory(good, "handoff_memory"),
              "receiver_handoff_memory": memory(good, "receiver_memory")}
    result["wire_bytes_scope"] = sorted({r.get("wire_bytes_scope", "unavailable") for r in good})
    return result


def paired_latency(rows, condition, precision, context, transport):
    baseline = {identity(r): r for r in rows if r["context_tokens"] == context and
                (r["condition"], r["precision"], r["transport"]) == ("transfer", "bf16", transport)}
    candidate = {identity(r): r for r in rows if r["context_tokens"] == context and
                 (r["condition"], r["precision"], r["transport"]) == (condition, precision, transport)}
    pairs = [(baseline[k], candidate[k]) for k in sorted(set(baseline) & set(candidate))
             if baseline[k]["kind"] == candidate[k]["kind"] == "prediction"
             and baseline[k].get("handoff_ack_ms") is not None and candidate[k].get("handoff_ack_ms") is not None]
    if any(b["handoff_ack_ms"] <= 0 or c["handoff_ack_ms"] <= 0 for b, c in pairs):
        raise ValueError("paired handoff latency must be strictly positive")
    union = len(set(baseline) | set(candidate))
    return {"baseline": "transfer:bf16", "ratio_definition": "candidate_ms / paired_BF16_transfer_ms; below 1 means lower measured latency",
            "difference_definition": "candidate_ms - paired_BF16_transfer_ms",
            "attempted_union": union, "n_pairs": len(pairs), "unpaired_or_unavailable": union-len(pairs),
            "candidate_to_bf16_ratio": stats((c["handoff_ack_ms"]/b["handoff_ack_ms"] for b, c in pairs), union),
            "candidate_minus_bf16_ms": stats((c["handoff_ack_ms"]-b["handoff_ack_ms"] for b, c in pairs), union)}


def summarize_rows(rows, label, paths=()):
    finite_tree(rows, label)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row.get("protocol_fingerprint", row["run_fingerprint"]))].append(row)
    result = {"label": label, "paths": list(map(str, paths)), "strata": []}
    for (split, protocol), selected in sorted(groups.items()):
        unique = [(identity(r), r["condition"], r["precision"], r["transport"]) for r in selected]
        if len(unique) != len(set(unique)):
            raise ValueError("duplicate prompt/condition across campaign files")
        by_context, by_cell = defaultdict(list), defaultdict(list)
        for row in selected:
            key = (row["context_tokens"], row["condition"], row["precision"], row["transport"])
            by_context[key].append(row)
            by_cell[(row["task"], *key)].append(row)
        group = {"split": split, "protocol_fingerprint": protocol,
                 "run_fingerprints": sorted({r["run_fingerprint"] for r in selected}),
                 "unique_prompts": len({identity(r) for r in selected}),
                 "total_rows": len(selected), "execution_failures": sum(r["kind"] == "failure" for r in selected),
                 "by_context": [], "by_task_context": []}
        for (context, condition, precision, transport), cell in sorted(by_context.items()):
            summary = {"context_tokens": context, "condition": condition, "precision": precision,
                       "transport": transport, **cell_summary(cell)}
            if (condition, precision) in REMOTE and transport != "none":
                summary["paired_latency"] = paired_latency(selected, condition, precision, context, transport)
            group["by_context"].append(summary)
        for (task, context, condition, precision, transport), cell in sorted(by_cell.items()):
            group["by_task_context"].append({"task": task, "context_tokens": context, "condition": condition,
                                            "precision": precision, "transport": transport, **cell_summary(cell)})
        result["strata"].append(group)
    return result


def load_section(paths, label, expected_split=None):
    before_hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    rows = load_rows(paths)
    if expected_split and any(r["split"] != expected_split for r in rows):
        raise ValueError(f"{label} requires {expected_split} split")
    # Multi-file aggregation is permitted only with verified source-hash sidecars.
    if len({r["run_fingerprint"] for r in rows}) > 1 and any("protocol_fingerprint" not in r for r in rows):
        raise ValueError("multiple runs require frozen source_sha256 provenance")
    section = summarize_rows(rows, label, paths)
    section["source_sha256"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    if section["source_sha256"] != before_hashes:
        raise ValueError("input result file changed while being summarized; retry after the writer completes")
    return section, rows


def timing_variation(blocks):
    groups = defaultdict(list)
    for block in blocks:
        for group in block["strata"]:
            for cell in group["by_context"]:
                if (cell["condition"], cell["precision"]) in REMOTE and cell["transport"] != "none":
                    key = (group["protocol_fingerprint"], cell["transport"], cell["context_tokens"], cell["condition"], cell["precision"])
                    groups[key].append((block["label"], cell))
    output = []
    for (protocol, transport, context, condition, precision), cells in sorted(groups.items()):
        def extrema(values):
            values = [v for v in values if v is not None]
            return {"n_blocks": len(values), "mean": float(np.mean(values)) if values else None,
                    "min": min(values) if values else None, "max": max(values) if values else None}
        output.append({"protocol_fingerprint": protocol, "transport": transport, "context_tokens": context,
                       "condition": condition, "precision": precision, "blocks": [label for label, _ in cells],
                       "block_handoff_p50_ms": extrema(c["handoff_ms"]["p50"] for _, c in cells),
                       "block_handoff_p95_ms": extrema(c["handoff_ms"]["p95"] for _, c in cells),
                       "block_paired_ratio_p50": extrema(c["paired_latency"]["candidate_to_bf16_ratio"]["p50"] for _, c in cells)})
    return output


def verify_complete_timing(block_rows, sections=None):
    """Verify three complete, compatible repeats of the same nine pilot prompts."""
    if len(block_rows) != 3:
        raise ValueError("complete timing requires exactly 3 blocks")
    reference, protocols = None, set()
    expected_cells = {(task, context) for task in ("niah_single_2", "niah_multikey_1", "niah_multiquery")
                      for context in (4096, 8192, 16384)}
    for index, rows in enumerate(block_rows, 1):
        finite_tree(rows, f"timing block {index}")
        if len(rows) != 36 or any(r["kind"] != "prediction" for r in rows):
            raise ValueError(f"timing block {index}: require 36 successful rows with no failures")
        if any(r["split"] != "pilot" or r["transport"] != "fabric-lib" for r in rows):
            raise ValueError(f"timing block {index}: require pilot split and native fabric-lib transport")
        if any(not r.get("protocol_fingerprint") for r in rows):
            raise ValueError("complete timing requires frozen source protocol provenance")
        protocols.update(r["protocol_fingerprint"] for r in rows)
        unique = [(identity(r), r["condition"], r["precision"]) for r in rows]
        if len(set(unique)) != 36:
            raise ValueError(f"timing block {index}: duplicate prompt/condition rows")
        prompts = defaultdict(list)
        for row in rows:
            prompts[identity(row)].append(row)
        if len(prompts) != 9 or {(key[1], key[2]) for key in prompts} != expected_cells:
            raise ValueError(f"timing block {index}: require 9 prompts, one in each task/context cell")
        lineage = {}
        for key, prompt_rows in prompts.items():
            if {(r["condition"], r["precision"]) for r in prompt_rows} != REMOTE or len(prompt_rows) != 4:
                raise ValueError(f"timing block {index}: every prompt requires all 4 remote conditions")
            signatures = {(r["sample_hash"], r["model_revision"]) for r in prompt_rows}
            if len(signatures) != 1:
                raise ValueError(f"timing block {index}: conflicting prompt hash/model revision")
            lineage[key] = next(iter(signatures))
        if reference is not None and lineage != reference:
            raise ValueError("timing blocks must use the same 9 pilot identities, hashes and model revisions")
        reference = lineage
        if any(r.get("handoff_ack_ms") is None or r["handoff_ack_ms"] <= 0 for r in rows):
            raise ValueError(f"timing block {index}: every successful row needs positive measured handoff latency")
    if len(protocols) != 1:
        raise ValueError("timing blocks have incompatible frozen source protocols")
    if sections is not None:
        paths = [str(Path(path).resolve()) for section in sections for path in section["paths"]]
        if len(paths) != len(set(paths)):
            raise ValueError("timing blocks must use distinct input files; repeated file path")
        signatures = [tuple(sorted(section["source_sha256"].values())) for section in sections]
        if len(set(signatures)) != len(signatures):
            raise ValueError("timing blocks contain identical input bytes; repeated block")
    return True


def number(value, digits=2):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "unavailable" if value is None else f"{100*value:.2f}"


def markdown(report):
    lines = ["# Measured campaign summary", "", report["interpretation"], "",
             "Ratios are computed per paired prompt before summarizing: candidate handoff / BF16 transfer handoff. Differences are candidate − BF16 transfer. Lower values indicate lower measured latency. No ratio of aggregate quantiles is substituted.", "",
             "Quality percentages condition on successful predictions; execution failures remain visible. Local bytes are encoded size only. Native application-frame bytes exclude network headers. Memory is handoff peak allocation above each worker's measured handoff baseline, not post-generation memory.", ""]
    for section in [*report["campaigns"], *report["timing_blocks"]]:
        lines += [f"## {section['label']}", ""]
        for group in section["strata"]:
            lines += [f"{group['split']}; protocol `{group['protocol_fingerprint'][:12]}`; {group['unique_prompts']} unique prompts; {group['total_rows']} rows; {group['execution_failures']} execution failures.", "",
                      "|Context|Condition|Transport|Predictions/attempts|Target %|All answers %|", "|---:|---|---|---:|---:|---:|"]
            for c in group["by_context"]:
                lines.append(f"|{c['context_tokens']}|{c['condition']}:{c['precision']}|{c['transport']}|{c['predictions']}/{c['attempted_prompts']}|{percent(c['target_recovery']['mean'])}|{percent(c['all_answers_rate']['mean'])}|")
            lines += ["", "Measured remote conditions, aggregated across tasks at each context (300 prompts per complete final context/condition).", "",
                      "|Context|Condition|N handoff|p50/p95 ms|Mean wire MiB|Paired N|Ratio p50/p95|Δ ms p50/p95|Sender/receiver temporary peak MiB p50|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
            for c in group["by_context"]:
                if "paired_latency" not in c:
                    continue
                p = c["paired_latency"]
                ratio, delta = p["candidate_to_bf16_ratio"], p["candidate_minus_bf16_ms"]
                sender = c["sender_handoff_memory"]["temporary_peak_allocated_mib"]["p50"]
                receiver = c["receiver_handoff_memory"]["temporary_peak_allocated_mib"]["p50"]
                lines.append(f"|{c['context_tokens']}|{c['condition']}:{c['precision']}|{c['handoff_ms']['n']}|{number(c['handoff_ms']['p50'])}/{number(c['handoff_ms']['p95'])}|{number(c['wire_mib']['mean'])}|{p['n_pairs']}|{number(ratio['p50'], 3)}/{number(ratio['p95'], 3)}|{number(delta['p50'])}/{number(delta['p95'])}|{number(sender)}/{number(receiver)}|")
            lines += ["", "Per-cell quality (each task/context is reported separately):", "",
                      "|Task|Context|Condition|Predictions/attempts|Target %|All answers %|", "|---|---:|---|---:|---:|---:|"]
            for c in group["by_task_context"]:
                lines.append(f"|{c['task']}|{c['context_tokens']}|{c['condition']}:{c['precision']}|{c['predictions']}/{c['attempted_prompts']}|{percent(c['target_recovery']['mean'])}|{percent(c['all_answers_rate']['mean'])}|")
            lines.append("")
    if report["timing_block_variation"]:
        lines += ["## Timing block variation", "", "Block medians and p95s remain separate observations. With three blocks, ranges are descriptive; no strong confidence interval or independence claim is warranted. Blocks are never pooled with the final quality campaign.", "",
                  "|Context|Condition|Blocks|Block p50 ms range|Block p95 ms range|Block paired ratio p50 range|", "|---:|---|---:|---:|---:|---:|"]
        for c in report["timing_block_variation"]:
            med, tail, ratio = c["block_handoff_p50_ms"], c["block_handoff_p95_ms"], c["block_paired_ratio_p50"]
            lines.append(f"|{c['context_tokens']}|{c['condition']}:{c['precision']}|{med['n_blocks']}|{number(med['min'])}–{number(med['max'])}|{number(tail['min'])}–{number(tail['max'])}|{number(ratio['min'], 3)}–{number(ratio['max'], 3)}|")
    lines += ["", "Full JSON includes metric sample counts and missing counts, phase timings, wire-byte scope, sender and receiver memory p50/p95/max, original run fingerprints, and input SHA256 hashes. These tables do not establish retrieval preservation or a population speedup.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", nargs="+", default=[])
    parser.add_argument("--pilot", nargs="+", default=[])
    parser.add_argument("--timing-block", nargs="+", action="append", default=[], help="repeat once per separate timing block, with that block's JSONL paths")
    parser.add_argument("--require-complete-final", action="store_true")
    parser.add_argument("--require-complete-timing", action="store_true", help="require three complete compatible 36-row native timing blocks over the same nine pilot prompts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not (args.final or args.pilot or args.timing_block):
        parser.error("supply --final, --pilot, or at least one --timing-block")
    report = {"schema_version": 1, "interpretation": "Measured descriptive summaries only; final quality, exploratory pilot, and repeated timing blocks are separate. Incomplete inputs cannot support complete-study claims.",
              "campaigns": [], "timing_blocks": [], "final_matrix_complete": False,
              "complete_timing_verified": False}
    if args.final:
        section, rows = load_section(args.final, "Final campaign", "final")
        try:
            verify_complete_final(rows)
            if any("protocol_fingerprint" not in r for r in rows):
                raise ValueError("complete final needs frozen source hashes")
            report["final_matrix_complete"] = True
        except ValueError as error:
            report["final_matrix_incomplete_reason"] = str(error)
            if args.require_complete_final:
                raise
        report["campaigns"].append(section)
    elif args.require_complete_final:
        parser.error("--require-complete-final requires --final paths")
    if args.pilot:
        section, _ = load_section(args.pilot, "Exploratory pilot", "pilot")
        report["campaigns"].append(section)
    timing_rows = []
    for i, paths in enumerate(args.timing_block, 1):
        section, rows = load_section(paths, f"Timing block {i}", "pilot")
        report["timing_blocks"].append(section)
        timing_rows.append(rows)
    if args.require_complete_timing:
        report["complete_timing_verified"] = verify_complete_timing(timing_rows, report["timing_blocks"])
    report["timing_block_variation"] = timing_variation(report["timing_blocks"])
    finite_tree(report, "campaign summary")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "campaign.md").write_text(markdown(report))
    print(json.dumps({"out": str(out), "final_matrix_complete": report["final_matrix_complete"], "timing_blocks": len(report["timing_blocks"]), "complete_timing_verified": report["complete_timing_verified"]}))


if __name__ == "__main__":
    main()
