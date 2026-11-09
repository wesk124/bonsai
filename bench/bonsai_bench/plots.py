"""PNG-only, exploratory figures from validated analysis summaries."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .analysis import finite_tree

COLORS = {"bf16": "#4FB9AF", "int8": "#6686C5", "int4": "#D59AB5"}


def plot_report(report, output):
    # Validate everything before importing pyplot or creating any partial figures.
    finite_tree(report, "plot_report")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
        # Matplotlib bundles the same white seaborn style for lean environments.
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update({"xtick.bottom": True, "ytick.left": True})
    else:
        sns.set_style("white", {"xtick.bottom": True, "ytick.left": True})
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 6})
    output = Path(output)
    paths = []

    def save(fig, path):
        if sns is not None:
            sns.despine(fig=fig)
        else:
            for ax in fig.axes:
                ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    for index, group in enumerate(report["strata"]):
        cells = [r for r in group["conditions"] if r["n_predictions"]]
        if not cells:
            continue
        directory = output / f"{index:02d}-{group['split']}-{group['transport']}"
        directory.mkdir(parents=True, exist_ok=True)
        mode = {"none": "local cache round trips", "fabric-lib": "GPU RDMA",
                "tcp_diagnostic": "TCP diagnostic"}.get(group["transport"], group["transport"])
        label = "Final evaluation" if report.get("complete_final_verified") else f"Exploratory {group['split']}"
        title = f"{label} · {mode}"
        tasks = sorted({r["task"] for r in cells})

        fig, axs = plt.subplots(1, len(tasks), figsize=(4*len(tasks), 3), squeeze=False)
        axes = iter(np.asarray(axs).flatten())
        for task in tasks:
            plt.sca(next(axes))
            for condition, precision in sorted({(c["condition"], c["precision"]) for c in cells}):
                series = sorted([c for c in cells if c["task"] == task and c["condition"] == condition and c["precision"] == precision], key=lambda c: c["context_tokens"])
                if series:
                    marker = {"bf16": "s", "int8": "x", "int4": "o"}[precision]
                    plt.plot([c["context_tokens"] for c in series], [100*c["target_recovery"] for c in series], marker=marker, linestyle="-", markersize=8 if precision == "bf16" else 5, markerfacecolor="none", color=COLORS[precision], label=f"{condition}:{precision}")
                    for c in series:
                        plt.annotate(f"n={c['n_predictions']}", (c["context_tokens"], 100*c["target_recovery"]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=6)
            plt.title(task)
            plt.xlabel("Context budget (tokens)")
            plt.ylabel("Target recovery (%)")
            plt.ylim(-3, 105)
            plt.legend(loc="lower left")
        fig.suptitle(title, fontsize=10, y=1.04)
        save(fig, directory / "quality-vs-context.png")

        comparisons = [r for r in group["comparisons"] if r["n_pairs"]]
        if comparisons:
            fig = plt.figure(figsize=(6, max(3, len(comparisons)*.27)))
            for i, c in enumerate(comparisons):
                low, high = c["bootstrap_ci95"]
                y = c["difference"]*100
                plt.errorbar(y, i, xerr=[[max(0, y-low*100)], [max(0, high*100-y)]], fmt="o", color=COLORS[c["precision"]], capsize=2)
                plt.plot(100*c["simultaneous_conservative_difference_lower"], i, "<", color=COLORS[c["precision"]], markersize=4)
            plt.axvline(0, color="grey", linewidth=.6)
            plt.axvline(-2, color="#08415C", linestyle="--", linewidth=.8, label="−2 pp margin")
            plt.yticks(range(len(comparisons)), [f"{c['task'].replace('niah_', '')} {c['context_tokens']//1024}K {c['precision']} (n={c['n_pairs']})" for c in comparisons], fontsize=6)
            plt.xlabel("Paired target-score difference (pp)")
            plt.title(title + "\nDots: bootstrap 95% CI\nTriangles: conservative simultaneous lower bound")
            plt.legend(loc="lower center")
            save(fig, directory / "paired-differences.png")

        # Local BF16 has no encoded packet. Never depict its zero bytes as a
        # zero-byte BF16 communication baseline.
        bytecells = [c for c in cells if c["condition"] in ("transfer", "local_roundtrip", "text_refill") and c["metrics"]["wire_bytes"]]
        if bytecells:
            fig = plt.figure(figsize=(4, 3))
            for precision in ("bf16", "int8", "int4"):
                series = [c for c in bytecells if c["precision"] == precision]
                if series:
                    plt.scatter([c["metrics"]["wire_bytes"]["mean"]/1e6 for c in series], [100*c["target_recovery"] for c in series], color=COLORS[precision], label=precision)
            plt.xlabel("Encoded frame size (MB; no network)" if group["transport"] == "none" else "Application handoff bytes (MB)")
            plt.ylabel("Target recovery (%)")
            plt.ylim(-3, 105)
            plt.title(title + "\nEach point is a task/context/condition cell")
            plt.legend()
            save(fig, directory / "quality-vs-bytes.png")

        phasecells = [c for c in cells if c["condition"] in ("local_roundtrip", "transfer") and c["metrics"]["encode_ms"] and c["metrics"]["decode_ms"]]
        if phasecells:
            fig = plt.figure(figsize=(max(5, len(phasecells)*.33), 3.3))
            xs = np.arange(len(phasecells))
            encode = np.array([c["metrics"]["encode_ms"]["p50"] for c in phasecells])
            decode = np.array([c["metrics"]["decode_ms"]["p50"] for c in phasecells])
            plt.bar(xs, encode, color="#4FB9AF", label="Encode p50")
            plt.bar(xs, decode, bottom=encode, color="#6686C5", label="Decode p50")
            plt.xticks(xs, [f"{c['task'].replace('niah_', '')}\n{c['context_tokens']//1024}K {c['precision']}" for c in phasecells], rotation=45, ha="right", fontsize=6)
            plt.ylabel("Phase latency (ms)")
            plt.title(title + "\nPhase medians; stacked sum is not a total-latency percentile")
            plt.legend()
            save(fig, directory / "phase-breakdown.png")

        # Fabric-only handoff latency: no substitution of local codec timings,
        # TCP diagnostic timings, or synthetic bandwidth estimates.
        if group["transport"] == "fabric-lib":
            latency = [c for c in cells if c["condition"] == "transfer" and c["metrics"]["handoff_ack_ms"]]
            if latency:
                fig, axs = plt.subplots(1, len(tasks), figsize=(4*len(tasks), 3), squeeze=False)
                axes = iter(np.asarray(axs).flatten())
                for task in tasks:
                    plt.sca(next(axes))
                    for precision in ("bf16", "int8", "int4"):
                        series = sorted([c for c in latency if c["task"] == task and c["precision"] == precision], key=lambda c: c["context_tokens"])
                        if series:
                            x = [c["context_tokens"] for c in series]
                            plt.plot(x, [c["metrics"]["handoff_ack_ms"]["p50"] for c in series], "o-", color=COLORS[precision], label=f"{precision} p50")
                            plt.plot(x, [c["metrics"]["handoff_ack_ms"]["p95"] for c in series], "x--", color=COLORS[precision], label=f"{precision} p95")
                    plt.title(task)
                    plt.xlabel("Context budget (tokens)")
                    plt.ylabel("Complete handoff latency (ms)")
                    plt.legend()
                fig.suptitle(title, fontsize=10, y=1.04)
                save(fig, directory / "latency-vs-context.png")

        positions = [c for c in comparisons if c["position_changes"]]
        if positions:
            fig = plt.figure(figsize=(4, 3))
            for precision, shift in (("int8", -.15), ("int4", .15)):
                selected = [v for c in positions if c["precision"] == precision for v in c["position_changes"]]
                for b in range(3):
                    values = [r for r in selected if r["bin"] == b]
                    if values:
                        rate = 100*np.mean([r["added_failure"] for r in values])
                        plt.bar(b+shift, rate, width=.28, color=COLORS[precision], label=precision if b == 0 else None)
                        plt.annotate(f"n={len(values)}", (b+shift, rate), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
            plt.xticks(range(3), ["Beginning", "Middle", "End"])
            plt.ylabel("Additional target failures vs BF16 (%)")
            plt.xlabel("Requested fact position in actual input tokens")
            plt.ylim(0, max(10, plt.ylim()[1]*1.2))
            plt.title(title + "\nDescriptive targets; correlated within prompts")
            plt.legend()
            save(fig, directory / "failure-vs-position.png")
    return paths
