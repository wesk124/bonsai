#!/usr/bin/env python3
"""Validate a complete frozen campaign, analyze it, and seal artifact hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = args.out.resolve()
    source = run / 'source'
    results = sorted((run / 'results').glob('*.jsonl'))
    timing = sorted((run / 'timing').glob('*.jsonl'))
    manifests = sorted((run / 'data/ruler/final').glob('*/*.jsonl'))
    if len(results) != 9 or len(manifests) != 9 or len(timing) != 3:
        raise ValueError('require nine final cells and three completed timing blocks')
    integrity = json.loads((run / 'integrity/model-cache-integrity.json').read_text())
    if integrity['status'] != 'PASS' or {r['precision'] for r in integrity['formats']} != {'bf16', 'int8', 'int4'}:
        raise ValueError('actual model-cache integrity diagnostic has not passed')
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(source / 'bench')}
    subprocess.run([sys.executable, str(source / 'scripts/verify_frozen_run.py'), str(run), '--require-native'], check=True)
    gate = subprocess.check_output([sys.executable, str(source / 'scripts/check_handoff_equivalence.py'),
                                    *map(str, results), '--expected-prompts', '900'], text=True)
    (out / 'handoff_equivalence.json').write_text(gate)
    subprocess.run([sys.executable, '-m', 'bonsai_bench.analysis', *map(str, results),
                    '--manifest', *map(str, manifests), '--require-complete-final', '--out', str(out)],
                   env=env, check=True)
    plotter = Path(__file__).resolve().with_name('plot_quality_bytes.py')
    subprocess.run([sys.executable, str(plotter), str(out)], env=env, check=True)
    summarizer = Path(__file__).resolve().with_name('summarize_campaign.py')
    timing_args = [arg for path in timing for arg in ('--timing-block', str(path))]
    subprocess.run([sys.executable, str(summarizer), '--final', *map(str, results),
                    '--require-complete-final', '--require-complete-timing', *timing_args,
                    '--out', str(out / 'campaign')], env=env, check=True)
    analysis_sources = out / 'analysis_source'
    (analysis_sources / 'scripts').mkdir(parents=True, exist_ok=True)
    (analysis_sources / 'tests').mkdir(exist_ok=True)
    for script in (Path(__file__).resolve(), summarizer, plotter):
        shutil.copy2(script, analysis_sources / 'scripts' / script.name)
    test = Path(__file__).resolve().parents[1] / 'tests/test_campaign_summary.py'
    shutil.copy2(test, analysis_sources / 'tests' / test.name)
    files = [*results, *timing, *manifests, Path(__file__).resolve(), summarizer, plotter,
             run / 'source_manifest.json', run / 'dataset_index.json', run / 'native_dependencies.json']
    files += [p.with_suffix(p.suffix + '.provenance.json') for p in (*results, *timing)]
    files += [run / 'submission.json', run / 'requirements.freeze.txt']
    files += [p for p in (run / 'integrity').rglob('*') if p.is_file()]
    files += [p for p in (run / 'logs').rglob('*') if p.is_file()]
    files += [p for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    record = {'status': 'complete_and_verified', 'final_prompts': 900,
              'final_predictions': 6300, 'native_transferred_continuations': 2700,
              'timing_blocks': 3, 'run_dir': str(run),
              'actual_model_cache_integrity': 'PASS',
              'sha256': {str(p): sha(p) for p in sorted(set(files))}}
    (out / 'artifact_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'sha256'}, sort_keys=True))


if __name__ == '__main__':
    main()
