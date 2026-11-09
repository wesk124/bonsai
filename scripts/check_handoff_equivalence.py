#!/usr/bin/env python3
"""Verify transport preserves each local continuation; quality inference is separate."""
import argparse
import json
from pathlib import Path

EXPECTED = {('local_bf16', 'bf16'), ('local_roundtrip', 'int8'),
            ('local_roundtrip', 'int4'), ('transfer', 'bf16'),
            ('transfer', 'int8'), ('transfer', 'int4'), ('text_refill', 'bf16')}
PAIRS = [(('transfer', 'bf16'), ('local_bf16', 'bf16')),
         (('transfer', 'int8'), ('local_roundtrip', 'int8')),
         (('transfer', 'int4'), ('local_roundtrip', 'int4')),
         (('text_refill', 'bf16'), ('local_bf16', 'bf16'))]


def check(paths, expected_prompts=None):
    samples = {}
    for path in paths:
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            row = json.loads(line)
            location = f'{path}:{lineno}'
            if row.get('kind') != 'prediction':
                raise ValueError(f'{location}: non-prediction row retained; resolve failures explicitly')
            key = row['sample_id']
            condition = row['condition'], row['precision']
            rows = samples.setdefault(key, {})
            if condition in rows:
                raise ValueError(f'{location}: duplicate {key} {condition}')
            ids = row.get('generated_ids')
            if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
                raise ValueError(f'{location}: missing/invalid generated token IDs')
            rows[condition] = row
    if not samples:
        raise ValueError('no prediction rows')
    if expected_prompts is not None and len(samples) != expected_prompts:
        raise ValueError(f'expected {expected_prompts} prompts, got {len(samples)}')
    for key, rows in samples.items():
        if set(rows) != EXPECTED:
            raise ValueError(f'{key}: missing {EXPECTED-set(rows)}, unexpected {set(rows)-EXPECTED}')
        if len({row['sample_hash'] for row in rows.values()}) != 1:
            raise ValueError(f'{key}: compared rows have different prompt hashes')
        if len({row['run_fingerprint'] for row in rows.values()}) != 1:
            raise ValueError(f'{key}: compared rows have different run fingerprints')
        for remote, local in PAIRS:
            if rows[remote].get('transport') != 'fabric-lib':
                raise ValueError(f'{key}: {remote} is not a native fabric result')
            if rows[remote]['generated_ids'] != rows[local]['generated_ids']:
                raise ValueError(f'{key}: generated token IDs differ for {remote} versus {local}')
    return {'prompts': len(samples), 'predictions': len(samples)*len(EXPECTED),
            'exact_token_id_pairs': len(samples)*len(PAIRS), 'native_handoff_equivalence': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+')
    parser.add_argument('--expected-prompts', type=int)
    args = parser.parse_args()
    print(json.dumps(check(args.results, args.expected_prompts), sort_keys=True))


if __name__ == '__main__':
    main()
