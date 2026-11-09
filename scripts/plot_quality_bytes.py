#!/usr/bin/env python3
"""Render accepted native results with an explicit text-refill legend entry."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bonsai_bench.analysis import finite_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir', type=Path)
    args = parser.parse_args()
    report = json.loads((args.analysis_dir / 'analysis.json').read_text())
    finite_tree(report)
    if not report['complete_final_verified']:
        raise ValueError('this publication figure requires accepted complete final data')
    sns.set_style('white', {'xtick.bottom': True, 'ytick.left': True})
    groups = [('transfer', 'bf16', 'BF16 cache', '#4FB9AF', 'o'),
              ('transfer', 'int8', 'INT8 cache', '#6686C5', 'o'),
              ('transfer', 'int4', 'INT4 cache', '#D59AB5', 'o'),
              ('text_refill', 'bf16', 'Text + prefill', '#9596C6', 'D')]
    for index, stratum in enumerate(report['strata']):
        if stratum['transport'] != 'fabric-lib':
            continue
        fig = plt.figure(figsize=(5, 3.3))
        for condition, precision, label, color, marker in groups:
            cells = [c for c in stratum['conditions'] if (c['condition'], c['precision']) == (condition, precision)]
            if len(cells) != 9:
                raise ValueError('missing task/context cells for ' + label)
            plt.scatter([c['metrics']['wire_bytes']['mean'] / 1e6 for c in cells],
                        [100 * c['target_recovery'] for c in cells], color=color,
                        marker=marker, s=32, label=label, alpha=.85)
        plt.xlabel('Application handoff bytes (MB)')
        plt.ylabel('Target recovery (%)')
        plt.ylim(-3, 105)
        plt.title('Final evaluation\nEach point is a task/context/condition cell', fontsize=10)
        plt.legend(fontsize=8)
        sns.despine()
        fig.tight_layout()
        path = args.analysis_dir / f'{index:02d}-final-fabric-lib' / 'quality-vs-bytes.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
