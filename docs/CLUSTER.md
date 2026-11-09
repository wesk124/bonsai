# Native H200 cluster setup

The native path uses the host NVIDIA driver, RDMA stack and `gdrdrv` device. Build
userspace dependencies into `../.tools`; no host package installation, driver
replacement, kernel module installation or container privilege is needed.
The login host cannot execute GPU/RDMA smoke tests.

Prerequisites are Rust1.91 in `.tools/{cargo,rustup}`, the verified CUDA12.8.1
redistributables in `.tools/cuda`, a Python environment with libclang18, a C/C++
compiler and the host `libibverbs.so.1`. Existing installation provenance is
`.tools/cuda-manifest.json`; full NVIDIA manifest is
`.tools/downloads/redistrib_12.8.1.json`.

From the `bonsai` directory:

```bash
bash scripts/bootstrap_native.sh
source scripts/native_env.sh
cargo +1.91.0 check -p bonsai-transport --features fabric
../.venv/bin/maturin build --manifest-path crates/bonsai-py/Cargo.toml --interpreter ../.venv/bin/python --auditwheel skip --out ../.tools/wheels
uv pip install --python ../.venv/bin/python --reinstall ../.tools/wheels/bonsai_py-0.1.0-cp311-cp311-linux_x86_64.whl
sbatch scripts/cluster/fabric_debug.sbatch
sbatch scripts/cluster/endpoint_smoke.sbatch
```

`bootstrap_native.sh` verifies SHA256-pinned libfabric1.22.0, GDRCopy2.5
userspace and CUDA profiler API12.8.90 archives. It compiles `fabric-debug` from
pplx-garden commit `1ed972ed3f0bd5616c997c9507c25616c63394fc`, which is also the
Cargo dependency revision. Its independent source tree is
`../.tools/src/pplx-garden-1ed972e`; the unrelated `third_party/pplx-garden` checkout
is not used. The extracted Debian verbs39 headers are sufficient for this
revision; runtime verbs providers come from each GPU host. libfabric's EFA and
verbs providers are disabled in the local build because the H200 InfiniBand
path uses fabric-lib's direct verbs provider. libfabric still supplies required
API1.22 definitions and symbols for fabric-lib's compiled EFA implementation.

`native_env.sh` exports build prefixes and link paths. CUDA driver stubs appear
only in `LIBRARY_PATH`; adding them to `LD_LIBRARY_PATH` would mask the real
GPU driver. It adds GCC standard headers for the libclang Python wheel, which
otherwise lacks `stddef.h`. CUDA's `lib64` is a compatibility symlink to the
redistributable's `lib` directory.

The Slurm smoke requests two `hpc-high` nodes under account `cw-sup`, one H200
per node, four CPUs, 16GiB RAM and a ten-minute wall limit. It starts the diagnostic patched
pinned debug server on one node and client on the other, selecting CUDA device0
and explicitly selected InfiniBand NIC `ibp0` per GPU. The diagnostic source
patch is preserved in `scripts/cluster/pplx-topology.patch`: it bypasses upstream
PCI discovery with an explicit NIC and allowed scheduler CPUs. Unmodified
pplx-garden discovery failed on these H200 PCI layouts. The client verifies paged GPU writes,
single GPU writes and immediate completions, then runs bounded microbenchmarks.
A pass requires three `VERIFIED` messages, a completed client and
`FABRIC_DEBUG_PASS`. The persistent server is terminated after the client exits.
These microbenchmarks establish network functionality; they are not model
inference quality or goodput results.

Validation records:

- `logs/preflight-1294228.log`: two H200 nodes; NVIDIA595.71.05; PyTorch2.8.0
  CUDA12.8 allocation works; `/dev/gdrdrv`, verbs devices and unlimited memlock.
- `.tools/fabric-debug-build.log`: pinned fabric-debug successfully compiled.
- `logs/fabric-debug-1294320.log`: unmodified upstream topology failed with
  `Failed to read NUMA node`; `1294335` recorded the further empty client
  topology failure. These failed attempts remain preserved.
- `logs/fabric-debug-1294340.log`: actual two-node diagnostic passed on
  `slurm-h200-244-055` and `slurm-h200-244-155`, with explicitly selected `ibp0`
  on each host. Paged and single GPU writes and immediate completions verified.
  The upstream diagnostic reported 272Gbps for 64MiB writes and 9.714us for
  immediate completions. Full server/client logs are in
  `logs/fabric-debug-1294340/`.
- `logs/endpoint-smoke-1294344.log`: production Bonsai FabricEndpoint passed on
  `slurm-h200-238-003` and `slurm-h200-238-057`. Bonsai selected nearest active
  NICs `ibp3` and `ibp1` and CPUs within each task's scheduler affinity. All
  payload bytes and immediate completions verified at 4KiB, 64KiB, 1MiB and
  16MiB. Both peers emitted `ENDPOINT_SMOKE_PASS`. One 16MiB transfer measured
  314.77Gbps, including Python/native completion overhead; this is a single
  functional smoke measurement, not a statistical performance result.

Production uses the small, provenance-documented pinned fabric source vendor in
`crates/bonsai-transport/native-fabric`, exposing visible NIC discovery; Bonsai
performs PCI ancestry, active-link, NUMA and scheduler-affinity selection itself.
It does not rely on the diagnostic hardcoded `ibp0` fallback. Inspect the
vendor's `UPSTREAM.md` and `visible-domains.patch` for the precise source change.
The project Python venv is Python3.11: explicitly pass its interpreter to maturin
and use `--auditwheel skip` to keep host CUDA driver/verbs dependencies external.
The installed validation wheel is
`../.tools/wheels/bonsai_py-0.1.0-cp311-cp311-linux_x86_64.whl`.
Native dependency archive/library hashes are in `../.tools/native-manifest.json`.
