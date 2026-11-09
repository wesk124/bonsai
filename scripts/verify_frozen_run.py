"""Verify frozen source/native package, copied datasets, and shared native libraries."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(run_dir, require_native=False):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / 'source_manifest.json').read_text())
    for relative, expected in manifest.items():
        if sha(run_dir / 'source' / relative) != expected:
            raise ValueError('frozen source/native package changed: ' + relative)
    dataset = json.loads((run_dir / 'dataset_index.json').read_text())
    for split in ('pilot', 'final'):
        for cell in dataset[split]['files']:
            if sha(run_dir / cell['path']) != cell['sha256']:
                raise ValueError('frozen dataset changed: ' + cell['path'])
    for cell in dataset.get('additional_files', []):
        if sha(run_dir / cell['path']) != cell['sha256']:
            raise ValueError('frozen supplemental dataset changed: ' + cell['path'])
    dependencies = json.loads((run_dir / 'native_dependencies.json').read_text())
    if require_native and not dependencies['native']:
        raise ValueError('final fabric execution requires a frozen native extension')
    for native in dependencies['native']:
        for lib in native['resolved_libraries']:
            if lib['shared_project_dependency'] and sha(Path(lib['path'])) != lib['sha256']:
                raise ValueError('shared native dependency changed: ' + lib['path'])
    return {'status': 'PASS', 'source_files': len(manifest), 'final_prompts': dataset['final']['count'],
            'native_extensions': len(dependencies['native'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--require-native', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.require_native), sort_keys=True))
