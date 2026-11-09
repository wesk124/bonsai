#!/usr/bin/env python3
"""Plot real, fully validated optimization pilot pairs as one 300-dpi PNG."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_optimization import BACKENDS, guarded_output, load_and_validate
from bonsai_bench.analysis import finite_tree, identity


def plot(run_dir, output):
    output = guarded_output(run_dir, output)
    if output.suffix.lower() != ".png":
        raise ValueError("optimization figure output must be PNG")
    # Full input/provenance/token checks precede figure creation. Never filter
    # nonfinite or failed observations into a more favorable partial plot.
    files, _, _ = load_and_validate(run_dir)
    lookup = {(f["block"], f["backend"], identity(r), r["precision"]): r
              for f in files for r in f["rows"] if r["condition"] == "transfer"}
    series = {}
    for context in (4096, 8192, 16384):
        for precision in ("bf16", "int8", "int4"):
            pairs = []
            for (block, backend, sample, p), row in sorted(lookup.items()):
                if backend == BACKENDS[0] and sample[2] == context and p == precision:
                    candidate = lookup[(block, BACKENDS[1], sample, precision)]
                    pairs.append([row["handoff_ack_ms"], candidate["handoff_ack_ms"]])
            if len(pairs) != 9:
                raise ValueError(f"plot series {context}/{precision}: expected nine complete paired observations")
            series[f"{context}:{precision}"] = pairs
    finite_tree(series, "paired handoff plot series")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    try:
        import seaborn as sns
    except ImportError:
        sns = None
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update({"xtick.bottom": True, "ytick.left": True})
    else:
        sns.set_style("white", {"xtick.bottom": True, "ytick.left": True})
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    colors = ("#4FB9AF", "#6686C5")
    markers = ("s", "D")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8), sharey=True)
    axes_iter = iter(np.asarray(axes).flatten())
    maximum = max(v for values in series.values() for pair in values for v in pair)
    for context in (4096, 8192, 16384):
        plt.sca(next(axes_iter))
        for x, precision in enumerate(("bf16", "int8", "int4")):
            values = np.asarray(series[f"{context}:{precision}"], dtype=float)
            for before, after in values:
                plt.plot([x-.13, x+.13], [before, after], color="#9596C6", alpha=.3, linewidth=.7, zorder=1)
            for backend in range(2):
                position = x + (-.13 if backend == 0 else .13)
                observations = values[:, backend]
                plt.scatter(np.full(9, position), observations, color=colors[backend], s=10, alpha=.45, zorder=2)
                low, median, high = np.quantile(observations, [.05, .5, .95])
                plt.errorbar(position, median, yerr=[[median-low], [high-median]], color=colors[backend],
                             marker=markers[backend], markersize=5, capsize=3, linewidth=1.4, zorder=3)
        plt.xticks(range(3), ["BF16", "INT8", "INT4"])
        plt.xlim(-.5, 2.5)
        plt.ylim(0, maximum*1.1)
        plt.xlabel("Transfer precision")
        plt.title(f"{context//1024}K context budget", pad=8)
        if context == 4096:
            plt.ylabel("Complete handoff latency (ms)")
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", label=l)
               for c,m,l in zip(colors, markers, ("PyTorch reference", "Fused Triton"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Exploratory nine-prompt pilot · measured native GPU RDMA", fontsize=10, y=.99)
    fig.text(.5, .07, "Thin lines pair the same prompt and block; 3 unique prompts × 3 blocks per context/backend.\nLarge markers: medians; whiskers: empirical 5th–95th percentiles, not confidence intervals.",
             ha="center", va="center", fontsize=7)
    if sns is not None:
        sns.despine(fig=fig)
    else:
        for axis in fig.axes:
            axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0,.13,1,.84))
    output = guarded_output(run_dir, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    print(plot(args.run_dir, args.out or args.run_dir / "analysis/paired-handoff.png"))


if __name__ == "__main__":
    main()
