#!/usr/bin/env python3
"""Wait for a submitted final array, then run the complete acceptance pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--array-job', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        counts = {p.stem: p.read_bytes().count(b'\n') for p in sorted((args.run_dir / 'results').glob('*.jsonl'))}
        status = subprocess.check_output(['sacct', '-j', args.array_job, '-X', '--noheader',
                                          '--parsable2', '--format=JobID,State,ExitCode'], text=True)
        jobs = {}
        for line in status.splitlines():
            job, state, code = line.split('|')[:3]
            if job.startswith(args.array_job + '_') and '[' not in job:
                jobs[job] = (state, code)
        print(json.dumps({'predictions_written': sum(counts.values()), 'counts': counts,
                          'completed_jobs': sum(s == 'COMPLETED' for s, _ in jobs.values())}), flush=True)
        failures = {j: (s, c) for j, (s, c) in jobs.items()
                    if s not in {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'COMPLETED'}
                    or (s == 'COMPLETED' and c != '0:0')}
        if failures:
            raise RuntimeError('array failed; retain evidence and resolve explicitly: ' + json.dumps(failures))
        if len(jobs) == 9 and all(s == 'COMPLETED' and c == '0:0' for s, c in jobs.values()):
            subprocess.run([sys.executable, str(Path(__file__).with_name('finalize_campaign.py')),
                            str(args.run_dir), '--out', str(args.out)], check=True)
            return
        time.sleep(45)
    raise TimeoutError('campaign still incomplete after monitor deadline; no completion claimed')


if __name__ == '__main__':
    main()
