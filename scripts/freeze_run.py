"""Snapshot source, installed native package, actual dependencies, and input bytes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'refusing to modify frozen run {out}')
    native_files = []
    native_binaries = []
    if args.native:
        spec = importlib.util.find_spec('bonsai')
        if spec is None or not spec.origin:
            raise RuntimeError('validated installed bonsai package is required')
        origin = Path(spec.origin)
        if spec.submodule_search_locations:
            package = origin.parent
            native_files = [(p, Path('bench/bonsai') / p.relative_to(package))
                            for p in package.rglob('*') if p.is_file()
                            and '__pycache__' not in p.parts and p.suffix in {'.py', '.so'}]
            native_binaries = [p for p, _ in native_files if p.suffix == '.so']
        elif origin.suffix == '.so':
            native_files = [(origin, Path('bench') / origin.name)]
            native_binaries = [origin]
        if not native_binaries:
            raise RuntimeError('installed bonsai package contains no native extension')
    dataset = json.loads((root / 'data/ruler/suite_index.json').read_text())
    if not dataset['final']['complete'] or dataset['final']['count'] != 900:
        raise ValueError('final dataset is incomplete')
    for split in ('pilot', 'final'):
        for cell in dataset[split]['files']:
            if sha(root / cell['path']) != cell['sha256']:
                raise ValueError('dataset file changed after audit: ' + cell['path'])
    out.mkdir(parents=True)
    source = out / 'source'
    selected = [root / 'Cargo.toml', root / 'Cargo.lock', root / 'rust-toolchain.toml', root / 'README.md']
    for directory in ('bench', 'crates', 'config', 'scripts', 'docs', 'tests'):
        selected.extend(p for p in (root / directory).rglob('*') if p.is_file()
                        and not any(part in {'target', '__pycache__', '.pytest_cache'} or part.endswith('.egg-info') for part in p.parts)
                        and p.suffix not in {'.so', '.pyc'})
    manifest = {}
    for path in sorted(set(selected)):
        target = source / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(path.relative_to(root))] = sha(target)
    for path, relative in native_files:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = sha(target)
    for split in ('pilot', 'final'):
        for cell in dataset[split]['files']:
            target = out / cell['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / cell['path'], target)
            if sha(target) != cell['sha256']:
                raise ValueError('dataset changed while copying: ' + cell['path'])
    smoke = root / 'data/ruler/pilot-smoke.jsonl'
    if not smoke.is_file():
        raise ValueError('nine-prompt timing/pilot manifest is missing')
    target = out / 'data/ruler/pilot-smoke.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smoke, target)
    dataset['additional_files'] = [{'path': 'data/ruler/pilot-smoke.jsonl', 'sha256': sha(target)}]
    tools = Path(os.environ.get('BONSAI_TOOLS', root.parent / '.tools')).resolve()
    dependencies = {'tools_root': str(tools), 'native': [],
                    'environment': {key: os.environ.get(key) for key in
                                    ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
                                     'CUDA_MODULE_LOADING', 'CUDA_HOME', 'GDRAPI_HOME',
                                     'LIBFABRIC_HOME', 'LD_LIBRARY_PATH')}}
    for binary in native_binaries:
        result = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, check=True)
        libraries = []
        for line in result.stdout.splitlines():
            match = re.search(r'^\s*(\S+)\s+=>\s+(/\S+)', line)
            if match:
                path = Path(match.group(2)).resolve()
                libraries.append({'soname': match.group(1), 'path': str(path),
                                  'sha256': sha(path), 'shared_project_dependency': path.is_relative_to(tools)})
        dependencies['native'].append({'installed_path': str(binary), 'sha256': sha(binary),
                                       'ldd_stdout': result.stdout, 'resolved_libraries': libraries,
                                       'unresolved_on_freeze_host': [line.strip() for line in result.stdout.splitlines() if 'not found' in line]})
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'dataset_index.json').write_text(json.dumps(dataset, indent=2) + '\n')
    (out / 'native_dependencies.json').write_text(json.dumps(dependencies, indent=2) + '\n')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True) if importlib.util.find_spec('pip') else subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True)
    (out / 'requirements.freeze.txt').write_text(freeze)
    print(out)


if __name__ == '__main__':
    main()
