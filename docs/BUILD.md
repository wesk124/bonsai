# Building fabric-lib and Bonsai

This guide covers how to build Perplexity's fabric-lib (pplx-garden) and
Bonsai on the two GPUDirect RDMA benchmark nodes.

| File | Purpose |
|---|---|
| [`docker/Dockerfile`](../docker/Dockerfile) | Dev/benchmark image with CUDA 12.9, libfabric + rdma-core, GDRCopy, Rust 1.91, PyTorch 2.9, HF Transformers, plus pplx-garden built at the pinned commit |
| [`docker/compose.yaml`](../docker/compose.yaml) | The same container as a compose service (devices, capabilities, ulimits, host network) |
| [`scripts/run-docker.sh`](../scripts/run-docker.sh) | `docker run` wrapper that discovers `/dev/infiniband/uverbs*` automatically |
| [`scripts/check-host.sh`](../scripts/check-host.sh) | Preflight check for every requirement below; run it on the host and inside the container |

pplx-garden is pinned to commit `1ed972ed3f0bd5616c997c9507c25616c63394fc`, in
both `Cargo.toml` and the Dockerfile's `PPLX_GARDEN_REV`. Bump both together.

---

## 1. Requirements

The container only supplies **user-space** libraries. Kernel drivers and device
nodes must come from the **host**.

### Host (each node)

| Requirement | Why | Check |
|---|---|---|
| Linux, kernel **≥ 5.12** | DMA-BUF for GPUDirect RDMA without `nvidia-peermem` | `uname -r` |
| NVIDIA driver supporting **CUDA ≥ 12.8** | fabric-lib requirement; the image uses CUDA 12.9 | `nvidia-smi` |
| GPUDirect path: open GPU kernel modules (DMA-BUF) **or** `nvidia_peermem` loaded | NIC reads and writes GPU memory directly | `grep Open /proc/driver/nvidia/version`, `lsmod \| grep peermem` |
| RDMA NIC driver: `mlx5_ib` (ConnectX-7, via DOCA-OFED / MLNX_OFED or inbox rdma-core) **or** `efa` (AWS) | Creates `/dev/infiniband/uverbs*` | `ls /dev/infiniband`, `ibv_devinfo` |
| At least one dedicated RDMA NIC per GPU, ideally under the same PCIe switch | fabric-lib binds NICs to GPUs by topology | `nvidia-smi topo -m` (look for `PIX`/`PXB` between GPU and NIC) |
| **GDRCopy** kernel module `gdrdrv` loaded | fabric-lib's `cuda-lib` uses GDRCopy flags | `ls /dev/gdrdrv` |
| Docker + **NVIDIA Container Toolkit** | `--gpus all` | `docker info \| grep -i nvidia` |
| `memlock` unlimited for the container | Memory registration pins pages | `--ulimit memlock=-1` (set by the scripts) |
| Capabilities `SYS_PTRACE`, `SYS_ADMIN`, `IPC_LOCK` | fabric-lib uses `pidfd_getfd` and pins memory | Set by the scripts |
| The two nodes can reach each other over the RDMA fabric, plus TCP port 7000 for `bonsai-ctl` | Data plane and control plane | `fabric-debug` (§4) |

### Inside the image (provided by the Dockerfile)

| Component | Version | Used by |
|---|---|---|
| CUDA toolkit | 12.9.1 (`nvidia/cuda:12.9.1-devel-ubuntu24.04`) | `cuda-sys`, `cudart-sys`; codec kernels |
| libfabric + rdma-core (libibverbs, mlx5 and efa providers) | AWS EFA installer 1.44.0 (`--skip-kmod`) | `libfabric-sys`, `libibverbs-sys` |
| GDRCopy `libgdrapi` | 2.5.1 | `gdrapi-sys` |
| clang / libclang | 18 | bindgen in every `*-sys` crate |
| Rust | 1.91.0 | pplx-garden + Bonsai toolchain |
| PyTorch | 2.9.0 + cu129 | Workers, `bonsai-py` |
| transformers, datasets, accelerate | latest ≥ 4.45 | Qwen2.5 workers, LongBench/RULER |
| `fabric-debug` | pinned pplx-garden | Link validation and bandwidth baseline |

### Build-time environment variables

Each fabric-lib `*-sys` crate finds its library with
`find_package(ENV, defaults, header)`. The image sets all four. On bare metal,
set them only if your install paths differ from the defaults.

| Variable | Default search path | Must contain |
|---|---|---|
| `LIBFABRIC_HOME` | `/opt/amazon/efa`, `/usr` | `include/rdma/fabric.h`, `lib/libfabric.so` |
| `LIBIBVERBS_HOME` | `/usr` | `include/infiniband/verbs.h` |
| `CUDA_HOME` | `/usr/local/cuda` | `include/cuda.h` |
| `GDRAPI_HOME` | `/usr` | `include/gdrapi.h` |
| `TORCH_CMAKE_PREFIX_PATH` | — | Only for the pplx-garden Python wheel: `python3 -c "import torch; print(torch.utils.cmake_prefix_path)"` |
| `LIBCLANG_PATH` | auto | Only if bindgen can't find libclang |

---

## 2. Host setup (once per node)

Pick the section that matches your NICs, then run `scripts/check-host.sh`.

### NVIDIA ConnectX-7 (InfiniBand or RoCE)

```bash
# 1. RDMA stack: DOCA-OFED (or MLNX_OFED / distro rdma-core) -> mlx5_ib + /dev/infiniband
sudo modprobe mlx5_ib && ibv_devinfo            # ports should be PORT_ACTIVE

# 2. GPUDirect: either use the open GPU kernel modules (DMA-BUF), or
sudo modprobe nvidia_peermem

# 3. GDRCopy kernel module (gdrdrv). Build from source:
git clone -b v2.5.1 https://github.com/NVIDIA/gdrcopy && cd gdrcopy/packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo apt install ./gdrdrv-dkms_*.deb ./libgdrapi_*.deb ./gdrcopy-tests_*.deb
sudo modprobe gdrdrv && ls /dev/gdrdrv

# 4. NVIDIA Container Toolkit
sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### AWS EFA (p4d / p5 / p5e)

```bash
# EFA installer WITH the kernel module on the host (AMIs like the Deep Learning AMI already have it)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.44.0.tar.gz
tar xf aws-efa-installer-1.44.0.tar.gz && cd aws-efa-installer && sudo ./efa_installer.sh -y -g
fi_info -p efa                                   # should list efa devices
# GDRCopy and the NVIDIA Container Toolkit: same as ConnectX steps 3-4.
```

Also launch both instances in the same placement group, with a security group
that allows all traffic to and from itself (EFA requires this).

---

## 3. Build and run the container

```bash
git clone git@github.com:wesk124/bonsai.git && cd bonsai
scripts/check-host.sh                       # fix every FAIL first

# Narrow TORCH_CUDA_ARCH_LIST to your GPU to cut build time:
#   A100 "8.0", H100/H200 "9.0a", B200 "10.0a"
docker build -t bonsai-dev -f docker/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST="9.0a" .

scripts/run-docker.sh                       # or: docker compose -f docker/compose.yaml run --rm bonsai
# inside the container:
scripts/check-host.sh
```

The image build already compiles fabric-lib and `fabric-debug`. If the build
succeeds, the toolchain, headers and libraries are all correct.

---

## 4. Validate the RDMA link with `fabric-debug`

Run this on both nodes inside the container. Arguments are
`<GPU list> <NICs per GPU> [server address]`.

```bash
# node A (server): prints its fabric address
fabric-debug 0 1
# node B (client): paste node A's address
fabric-debug 0 1 fe80...
```

Write down the bandwidth it reports. It is the line rate that Bonsai's numbers
are compared against, and the input for `CostModel::link_efficiency`. With
several NICs per GPU, pass that count as the second argument (for example
`2` on p5).

---

## 5. Build fabric-lib directly (without Bonsai)

```bash
cd /opt/pplx-garden            # already cloned at the pinned commit in the image
cargo build --release -p fabric-lib
cargo build --release -p fabric-debug
cargo test  -p fabric-lib      # unit tests (some need RDMA hardware)

# Optional Python bindings (pplx_garden.fabric_lib), already installed in the image:
export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
python3 -m build --wheel && pip install dist/*.whl
```

### Bare metal (no Docker)

Install the same user-space stack on the host, then build normally:

```bash
sudo apt install build-essential cmake pkg-config libclang-dev clang-18 libssl-dev git curl
# libfabric + rdma-core: AWS EFA installer (works for ConnectX too) or distro packages:
sudo apt install libfabric-dev libibverbs-dev rdma-core ibverbs-utils
# GDRCopy: libgdrapi from §2.  CUDA toolkit 12.8+ at /usr/local/cuda.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0
export LIBFABRIC_HOME=/usr    # or /opt/amazon/efa
cargo build --release -p fabric-lib
```

The distro `libfabric-dev` may be older than the one in the EFA installer. If
bindgen or the linker complains about missing `fi_*` symbols, use the EFA
installer's libfabric.

---

## 6. Build Bonsai

```bash
cd /app                                         # the mounted repo
cargo build && cargo test                       # portable crates, no RDMA needed
cargo build --release -p bonsai-dataplane --features fabric   # links fabric-lib (pinned git rev)
```

- `--features cuda` needs `crates/bonsai-codec/kernels/bonsai_codec.cu` and its
  `build.rs`, which arrive in milestone 2.
- `bonsai-py` is built with `maturin develop` inside `crates/bonsai-py` once
  the `Worker` bindings are implemented.
- The `fabric` code path has been type-checked against the fabric-lib API but
  has not been compiled yet, since that requires Linux with the stack above.
  Expect to fix small API mismatches on the first build there.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `find_package: LIBFABRIC_HOME is not set and include/rdma/fabric.h is not found` | libfabric headers are missing; install them or set `LIBFABRIC_HOME` (same pattern for `LIBIBVERBS_HOME`, `CUDA_HOME`, `GDRAPI_HOME`) |
| `Unable to find libclang` | `apt install libclang-dev`, or `export LIBCLANG_PATH=/usr/lib/llvm-18/lib` |
| Linker: `cannot find -lgdrapi` / `-lfabric` | Library is not in `<prefix>/lib` for the found prefix; check `ldconfig -p \| grep -E 'gdrapi\|fabric'` |
| `pidfd_getfd: Operation not permitted` | Missing `--cap-add SYS_PTRACE --cap-add SYS_ADMIN` (or run as root) |
| `ibv_reg_mr` / memory registration fails | `memlock` not unlimited, or no GPUDirect (no DMA-BUF and `nvidia_peermem` not loaded) |
| No devices inside the container | `/dev/infiniband/*` or `/dev/gdrdrv` were not passed through; use `scripts/run-docker.sh`, which discovers them |
| `fabric-debug` bandwidth far below line rate | GPU and NIC on different PCIe roots or NUMA nodes (`nvidia-smi topo -m`), wrong NICs-per-GPU argument, or EFA instances not in a placement group |
| CUDA error "no kernel image available" | `TORCH_CUDA_ARCH_LIST` doesn't include your GPU; rebuild with the right `--build-arg` |
