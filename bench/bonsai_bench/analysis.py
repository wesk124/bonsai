"""Auditable paired RULER analysis; pilot, run and transport strata stay separate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_tree(value, path="row"):
    """Reject upstream NaN/Inf rather than silently filtering a plotted series."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value (NaN/Inf count >= 1): {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            finite_tree(item, f"{path}[{i}]")


def identity(row):
    return (row["split"], row["task"], int(row["context_tokens"]), row["sample_id"])


def load_manifests(paths):
    result = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            finite_tree(row, str(path))
            key = identity(row)
            if key in result:
                raise ValueError(f"duplicate manifest identity: {key}")
            result[key] = row
    return result


def load_rows(paths, manifests=None):
    rows, seen, lineage = [], set(), {}
    for path in map(Path, paths):
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        provenance = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if provenance and digest(provenance["config"]) != provenance["run_fingerprint"]:
            raise ValueError(f"provenance configuration fingerprint mismatch: {sidecar}")
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            finite_tree(row, f"{path}:{number}")
            for field in ("kind", "split", "task", "context_tokens", "sample_id", "condition",
                          "precision", "sample_hash", "model_revision", "run_fingerprint"):
                if field not in row:
                    raise ValueError(f"missing {field}: {path}:{number}")
            if row["kind"] not in ("prediction", "failure"):
                raise ValueError(f"unknown row kind: {row['kind']}")
            if provenance and row["run_fingerprint"] != provenance["run_fingerprint"]:
                raise ValueError("row/run fingerprint mismatch")
            if row["condition"].startswith("local_"):
                transport = "none"
            else:
                transport = row.get("transport") or (provenance or {}).get("config", {}).get("transport")
                if not transport:
                    raise ValueError("remote failure/result lacks transport provenance")
            row = {**row, "transport": transport}
            if provenance and provenance["config"].get("source_sha256"):
                # Array cells differ only in their frozen sample selections.
                # Every other configuration field, including all source hashes,
                # must agree before the caller explicitly aggregates them.
                protocol = {k: v for k, v in provenance["config"].items() if k != "sample_hashes"}
                row["protocol_fingerprint"] = digest(protocol)
            key = identity(row)
            signature = (row["sample_hash"], row["model_revision"])
            if key in lineage and lineage[key] != signature:
                raise ValueError(f"mismatched sample hash/model revision: {key}")
            lineage[key] = signature
            if row["sample_id"].startswith(("pilot/", "final/")) and not row["sample_id"].startswith(row["split"] + "/"):
                raise ValueError("mismatched split and sample_id")
            if manifests is not None:
                sample = manifests.get(key)
                if sample is None or digest(sample) != row["sample_hash"] or sample["model_revision"] != row["model_revision"]:
                    raise ValueError(f"result does not match frozen manifest: {key}")
            unique = (row["run_fingerprint"], transport, key, row["condition"], row["precision"])
            if unique in seen:
                raise ValueError(f"duplicate result/attempt: {unique}; resolve retries explicitly")
            seen.add(unique)
            if row["kind"] == "prediction":
                score = row["score"]
                if not 0 <= score["target_recovery"] <= 1:
                    raise ValueError("target_recovery outside [0,1]")
                answers = score.get("per_answer_recovered")
                if answers is not None:
                    if not answers or any(type(v) is not bool for v in answers):
                        raise ValueError("per_answer_recovered must be nonempty booleans")
                    if not np.isclose(np.mean(answers), score["target_recovery"]) or bool(all(answers)) != score["all_answers_recovered"]:
                        raise ValueError("inconsistent target/all-answer scores")
            rows.append(row)
    return rows


def paired_interval(differences, *, resamples=10000, seed=20261107, alpha=0.05):
    """Resample whole prompts, preserving all their targets; protect zero discordance.

    The exact binomial upper bound on P(prompt loses any target score) also
    bounds expected score degradation because prompt loss lies in [0,1].
    It is conservative and intentionally does not claim certainty at a ceiling.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) == 0 or not np.isfinite(d).all() or np.any(np.abs(d) > 1):
        raise ValueError("paired differences must be finite, nonempty and in [-1,1]")
    rng = np.random.default_rng(seed)
    # Bounded batches avoid a large resamples x prompts allocation on final data.
    boot = np.concatenate([d[rng.integers(0, len(d), size=(min(500, resamples-i), len(d)))].mean(1)
                           for i in range(0, resamples, 500)])
    losses = int(np.count_nonzero(d < 0))
    upper = 1.0 if losses == len(d) else float(beta.ppf(1-alpha, losses+1, len(d)-losses))
    low, high = np.quantile(boot, [alpha/2, 1-alpha/2])
    return {"n_pairs": len(d), "difference": float(d.mean()), "bootstrap_ci95": [float(low), float(high)],
            "degraded_prompts": losses, "improved_prompts": int(np.count_nonzero(d > 0)),
            "degradation_probability_upper": upper, "conservative_difference_lower": -upper,
            "bound_alpha": alpha}


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError("non-finite metric series")
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.quantile(a, .5)), "p95": float(np.quantile(a, .95))}


def metrics(rows):
    result = {}
    for field in ("encode_ms", "decode_ms", "handoff_ack_ms", "prefill_ms", "receiver_prefill_ms", "continuation_ms",
                  "wire_bytes", "payload_bytes", "metadata_bytes", "control_bytes", "prediction_response_bytes"):
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = quantiles(values)
    for location in ("memory", "handoff_memory", "receiver_memory"):
        result[location] = {field: quantiles([r[location][field] for r in rows if field in r.get(location, {})])
                            for field in sorted({k for r in rows for k in r.get(location, {})})}
    return result


def verify_complete_final(rows, manifests=None):
    """Require the proposal's exact 900-prompt / 2700-successful-transfer matrix."""
    if not rows or any(r["split"] != "final" for r in rows):
        raise ValueError("complete final requires final split only")
    strata = {(r.get("protocol_fingerprint", r["run_fingerprint"]), r["transport"]) for r in rows if r["condition"] == "transfer"}
    if len(strata) != 1 or next(iter(strata))[1] != "fabric-lib":
        raise ValueError("complete final requires one native fabric-lib run")
    transfer = [r for r in rows if r["condition"] == "transfer"]
    if len(transfer) != 2700 or any(r["kind"] != "prediction" for r in transfer):
        raise ValueError("incomplete final: require 2700 successful transferred continuations")
    expected_tasks = {"niah_single_2", "niah_multikey_1", "niah_multiquery"}
    counts = Counter((r["task"], r["context_tokens"], r["precision"]) for r in transfer)
    expected = {(t, c, p): 100 for t in expected_tasks for c in (4096, 8192, 16384) for p in ("bf16", "int8", "int4")}
    if dict(counts) != expected:
        raise ValueError("incomplete final: each task/context/precision cell must contain exactly 100")
    keys = {identity(r) for r in transfer}
    if len(keys) != 900:
        raise ValueError("incomplete final: require 900 unique prompts")
    for precision in ("bf16", "int8", "int4"):
        if {identity(r) for r in transfer if r["precision"] == precision} != keys:
            raise ValueError("incomplete final paired prompt sets")
    if manifests is not None and keys != {key for key in manifests if key[0] == "final"}:
        raise ValueError("final results differ from complete frozen manifest set")
    return True


def position_rows(base, candidate, sample):
    """Descriptive per-target changes; targets are never independent CI units."""
    result = []
    if not sample or "per_answer_recovered" not in base["score"] or "per_answer_recovered" not in candidate["score"]:
        return result
    outputs = sample["outputs"]
    for output, before, after in zip(outputs, base["score"]["per_answer_recovered"], candidate["score"]["per_answer_recovered"]):
        facts = [f for f in sample.get("facts", []) if f.get("target") and f["value"] == output]
        if facts:
            position = float(np.mean([f["token_start"] / sample["input_length"] for f in facts]))
            result.append({"bin": min(2, int(position*3)), "position": position,
                           "added_failure": int(before and not after), "restored": int(after and not before)})
    return result


def analyze(rows, manifests=None, *, require_complete_final=False, aggregate_compatible_runs=False, resamples=10000):
    finite_tree(rows)
    if require_complete_final:
        verify_complete_final(rows, manifests)
    strata = defaultdict(list)
    for row in rows:
        fingerprint = row["run_fingerprint"]
        if require_complete_final or aggregate_compatible_runs:
            fingerprint = row.get("protocol_fingerprint", fingerprint)
        strata[(row["split"], fingerprint, row["transport"])].append(row)
    report = {"schema_version": 1, "margin": .02, "bootstrap_seed": 20261107,
              "resamples": resamples, "complete_final_verified": require_complete_final,
              "interpretation": "Exploratory unless complete final verified; no speedup claim from bytes alone. Confidence bounds are per-family Bonferroni adjusted; bootstrap intervals are pointwise 95%.",
              "strata": []}
    for (split, fingerprint, transport), selected in sorted(strata.items()):
        unique = [(identity(r), r["condition"], r["precision"]) for r in selected]
        if len(set(unique)) != len(unique):
            raise ValueError("duplicate cell/prompt across aggregated runs")
        summary = {"split": split, "run_fingerprint": fingerprint, "transport": transport,
                   "source_run_fingerprints": sorted({r["run_fingerprint"] for r in selected}),
                   "rows": len(selected), "prediction_rows": sum(r["kind"] == "prediction" for r in selected),
                   "failure_rows": sum(r["kind"] == "failure" for r in selected),
                   "unique_prompts": len({identity(r) for r in selected}), "conditions": [], "comparisons": []}
        conditions = defaultdict(list)
        for row in selected:
            conditions[(row["task"], row["context_tokens"], row["condition"], row["precision"])].append(row)
        for (task, context, condition, precision), cell in sorted(conditions.items()):
            good = [r for r in cell if r["kind"] == "prediction"]
            expected = len([k for k in (manifests or {}) if k[:3] == (split, task, context)])
            summary["conditions"].append({"task": task, "context_tokens": context, "condition": condition,
                "precision": precision, "attempted_prompts": len(cell), "n_predictions": len(good),
                "expected_manifest_prompts": expected or None, "unattempted_manifest_prompts": max(0, expected-len(cell)) if expected else None,
                "failure_rows": len(cell)-len(good),
                "target_recovery": float(np.mean([r["score"]["target_recovery"] for r in good])) if good else None,
                "all_answers_rate": float(np.mean([r["score"]["all_answers_recovered"] for r in good])) if good else None,
                "metrics": metrics(good), "bytes_scope": "encoded size; no network transfer" if transport == "none" else "application handoff frames; excludes network headers"})
        comparison_keys = [key for key in conditions if key[2] in ("local_roundtrip", "transfer") and key[3] in ("int8", "int4")]
        alpha = .05 / max(1, len(comparison_keys))
        for task, context, condition, precision in sorted(comparison_keys):
            baseline_condition = "local_bf16" if condition == "local_roundtrip" else "transfer"
            baseline = {identity(r): r for r in conditions.get((task, context, baseline_condition, "bf16"), [])}
            candidate = {identity(r): r for r in conditions[(task, context, condition, precision)]}
            attempted = set(baseline) | set(candidate)
            good_keys = sorted(k for k in set(baseline) & set(candidate) if baseline[k]["kind"] == candidate[k]["kind"] == "prediction")
            d = [candidate[k]["score"]["target_recovery"]-baseline[k]["score"]["target_recovery"] for k in good_keys]
            comparison = {"task": task, "context_tokens": context, "condition": condition, "precision": precision,
                          "baseline_condition": baseline_condition, "attempted_union": len(attempted),
                          "missing_baseline_rows": len(set(candidate)-set(baseline)),
                          "missing_candidate_rows": len(set(baseline)-set(candidate)),
                          "baseline_execution_failures": sum(r["kind"] == "failure" for r in baseline.values()),
                          "candidate_execution_failures": sum(r["kind"] == "failure" for r in candidate.values()),
                          "complete_pairs": len(good_keys) == len(attempted), "n_pairs": len(d),
                          "position_changes": [], "margin_supported": False, "status": "inconclusive"}
            if d:
                interval = paired_interval(d, resamples=resamples)
                bound = paired_interval(d, resamples=resamples, alpha=alpha)
                comparison.update(interval)
                comparison["simultaneous_bound_alpha"] = alpha
                comparison["simultaneous_conservative_difference_lower"] = bound["conservative_difference_lower"]
                comparison["all_answers_difference"] = float(np.mean([int(candidate[k]["score"]["all_answers_recovered"])-int(baseline[k]["score"]["all_answers_recovered"]) for k in good_keys]))
                comparison["all_answers_interval"] = paired_interval([int(candidate[k]["score"]["all_answers_recovered"])-int(baseline[k]["score"]["all_answers_recovered"]) for k in good_keys], resamples=resamples)
                for key in good_keys:
                    comparison["position_changes"].extend(position_rows(baseline[key], candidate[key], (manifests or {}).get(key)))
                comparison["margin_supported"] = bool(require_complete_final and comparison["complete_pairs"] and bound["conservative_difference_lower"] >= -.02 and interval["bootstrap_ci95"][0] >= -.02)
                if comparison["margin_supported"]:
                    comparison["status"] = "two-percentage-point margin supported by conservative simultaneous bound"
            summary["comparisons"].append(comparison)
        report["strata"].append(summary)
    return report


def write_summary(report, path):
    lines = ["# Paired retrieval analysis", "", report["interpretation"], "",
             "Target score and all-answer rate use successful predictions only; execution failures and incomplete pairs are reported separately. No missing result is imputed as a correct answer.", ""]
    for group in report["strata"]:
        lines += [f"## {group['split']} / {group['transport']} / {group['run_fingerprint'][:12]}", "",
                  f"{group['unique_prompts']} distinct prompts, {group['prediction_rows']} predictions, {group['failure_rows']} execution failures.", "",
                  "|Task|Context|Condition|Precision|Predictions/attempts|Target score|All answers|", "|---|---:|---|---|---:|---:|---:|"]
        for cell in group["conditions"]:
            score = "unavailable" if cell["target_recovery"] is None else f"{100*cell['target_recovery']:.2f}%"
            allrate = "unavailable" if cell["all_answers_rate"] is None else f"{100*cell['all_answers_rate']:.2f}%"
            lines.append(f"|{cell['task']}|{cell['context_tokens']}|{cell['condition']}|{cell['precision']}|{cell['n_predictions']}/{cell['attempted_prompts']}|{score}|{allrate}|")
        lines += ["", "|Task|Context|Precision|Pairs/union|Difference (pp)|Paired bootstrap 95% CI (pp)|Conservative lower (pp)|Status|", "|---|---:|---|---:|---:|---|---:|---|"]
        for cell in group["comparisons"]:
            if not cell["n_pairs"]:
                lines.append(f"|{cell['task']}|{cell['context_tokens']}|{cell['precision']}|0/{cell['attempted_union']}|unavailable|unavailable|unavailable|inconclusive|")
                continue
            low, high = cell["bootstrap_ci95"]
            lines.append(f"|{cell['task']}|{cell['context_tokens']}|{cell['precision']}|{cell['n_pairs']}/{cell['attempted_union']}|{100*cell['difference']:.2f}|[{100*low:.2f}, {100*high:.2f}]|{100*cell['simultaneous_conservative_difference_lower']:.2f}|{cell['status']}|")
        lines += ["", "Full phase p50/p95, byte counts, memory measurements, added failures and provenance appear in analysis.json. A zero-width bootstrap interval is descriptive and does not establish the preservation margin.", ""]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-complete-final", action="store_true")
    parser.add_argument("--aggregate-compatible-runs", action="store_true", help="explicitly combine cells only when source hashes and all non-sample configuration agree")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.resamples < 100:
        parser.error("at least 100 bootstrap resamples required")
    manifests = load_manifests(args.manifest)
    rows = load_rows(args.results, manifests)
    if args.require_complete_final and any("protocol_fingerprint" not in r for r in rows):
        raise ValueError("final analysis requires sidecars with frozen source_sha256 provenance")
    report = analyze(rows, manifests, require_complete_final=args.require_complete_final, aggregate_compatible_runs=args.aggregate_compatible_runs, resamples=args.resamples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_summary(report, out / "summary.md")
    if not args.no_plots:
        from .plots import plot_report
        plot_report(report, out)
    print(json.dumps({"out": str(out), "strata": len(report["strata"]), "complete_final_verified": report["complete_final_verified"]}))


if __name__ == "__main__":
    main()
