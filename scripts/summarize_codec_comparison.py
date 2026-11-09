#!/usr/bin/env python3
"""Validate and summarize the actual-model codec benchmark beside its inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bonsai_bench.analysis import finite_tree


def summarize(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "summary.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    if (metadata.get("gate"), metadata.get("verified_precision_cells"),
        metadata.get("backend_timing_rows"), metadata.get("run_fingerprint")) != ("PASS", 27, 54, fingerprint):
        raise ValueError("require the complete validated nine-prompt codec benchmark")
    checks = [json.loads(s) for s in (directory / "verification.jsonl").read_text().splitlines()]
    timings = [json.loads(s) for s in (directory / "timings.jsonl").read_text().splitlines()]
    finite_tree(checks)
    finite_tree(timings)
    verified = {}
    for row in checks:
        key = row["sample_id"], row["precision"]
        if key in verified or row["run_fingerprint"] != fingerprint or row["gate"] != "PASS":
            raise ValueError("invalid or duplicate exactness record")
        if not all(row[x] is True for x in ("canonical_metadata_exact", "complete_payload_exact", "all_four_cross_decodes_bitwise_equal")):
            raise ValueError("incomplete byte compatibility check")
        if row["reference_payload_sha256"] != row["fused_payload_sha256"]:
            raise ValueError("encoded payload hashes differ")
        if row["precision"] == "bf16" and row["bf16_matches_original"] is not True:
            raise ValueError("BF16 must match the original model cache")
        verified[key] = row
    seen, groups = set(), defaultdict(list)
    for row in timings:
        key = row["sample_id"], row["precision"]
        full_key = *key, row["backend"]
        if key not in verified or full_key in seen or row["run_fingerprint"] != fingerprint:
            raise ValueError("unverified, duplicate or mismatched timing")
        if row["jit_cache_misses_during_timing"] != 0:
            raise ValueError("compilation occurred during timing")
        seen.add(full_key)
        for stage in ("encode", "decode"):
            if row[stage]["median_ms"] <= 0 or row[stage]["repeats"] != 5 or len(row[stage]["measurements"]) != 5:
                raise ValueError("require five positive synchronized timing repeats")
            elapsed = [m["elapsed_ms"] for m in row[stage]["measurements"]]
            if any(t <= 0 for t in elapsed) or statistics.median(elapsed) != row[stage]["median_ms"]:
                raise ValueError("timing median differs from its positive measured repeats")
        if row["sum_stage_medians_ms"] != row["encode"]["median_ms"] + row["decode"]["median_ms"]:
            raise ValueError("combined codec time differs from its measured stages")
        groups[row["context_tokens"], row["precision"], row["backend"]].append(row)
    expected = {(c, p, b) for c in (4096, 8192, 16384) for p in ("bf16", "int8", "int4")
                for b in ("pytorch_packed_v1", "triton_packed_v1")}
    if len(verified) != 27 or len(seen) != 54 or set(groups) != expected or any(len(v) != 3 for v in groups.values()):
        raise ValueError("incomplete context/precision/backend matrix")
    output = {"status": "PASS", "run_fingerprint": fingerprint,
              "scope": "Median across three unique prompts per context of each prompt's five-repeat median. Combined latency is the median of per-prompt encode+decode medians. These are codec-only timings, not complete handoff.",
              "memory_scope": "Median of per-prompt maximum extra allocated bytes over five repeats; decoded cache residency is included.",
              "cells": []}
    for (context, precision, backend), rows in sorted(groups.items()):
        cell = {"context_tokens": context, "precision": precision, "backend": backend, "unique_prompts": 3,
                "combined_ms": statistics.median(r["sum_stage_medians_ms"] for r in rows)}
        for stage in ("encode", "decode"):
            cell[stage + "_ms"] = statistics.median(r[stage]["median_ms"] for r in rows)
            cell[stage + "_peak_extra_mib"] = statistics.median(r[stage]["peak_allocated_increment_bytes"] for r in rows) / 2**20
        output["cells"].append(cell)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codec_directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.codec_directory)
    destination = args.codec_directory / "aggregate.json"
    if destination.resolve().parent != args.codec_directory.resolve():
        raise ValueError("output must remain beside its codec inputs")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(destination)
