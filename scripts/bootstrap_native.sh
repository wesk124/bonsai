#!/usr/bin/env bash
# Build userspace dependencies only. Never installs a kernel module or host package.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools="${BONSAI_TOOLS:-$(dirname "$root")/.tools}"
mkdir -p "$tools"/{downloads,src,gdrcopy,libfabric,rdma}
fetch() {
    local url="$1" name="$2" sha="$3"
    [[ -f "$tools/downloads/$name" ]] || curl -fL --retry 2 "$url" -o "$tools/downloads/$name"
    echo "$sha  $tools/downloads/$name" | sha256sum --check --status
}
fetch https://github.com/ofiwg/libfabric/releases/download/v1.22.0/libfabric-1.22.0.tar.bz2 libfabric-1.22.0.tar.bz2 485e6cafa66c9e4f6aa688d2c9526e274c47fda3a783cf1dd8f7c69a07e2d5fe
fetch https://codeload.github.com/NVIDIA/gdrcopy/tar.gz/refs/tags/v2.5 gdrcopy-2.5.tar.gz 196400877be7e511edcf2a87b21a605cca99522ff217c97429348fd9153b30d7
fetch https://developer.download.nvidia.com/compute/cuda/redist/cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.8.90-archive.tar.xz cuda_profiler_api.tar.xz f97b3779e909caa5edf56606432ec1dd3173a5d3d503ef39f367467dbdb006b7
[[ -f "$tools/cuda/include/cuda_profiler_api.h" ]] || tar -xf "$tools/downloads/cuda_profiler_api.tar.xz" -C "$tools/cuda" --strip-components=1
[[ -d "$tools/src/libfabric-1.22.0" ]] || tar -xf "$tools/downloads/libfabric-1.22.0.tar.bz2" -C "$tools/src"
[[ -d "$tools/src/gdrcopy-2.5" ]] || tar -xf "$tools/downloads/gdrcopy-2.5.tar.gz" -C "$tools/src"
if [[ ! -f "$tools/libfabric/lib/libfabric.so" ]]; then
    (cd "$tools/src/libfabric-1.22.0"
     ./configure --prefix="$tools/libfabric" --disable-verbs --disable-efa --disable-usnic --disable-psm2 --disable-psm3 --disable-opx
     make -j"${JOBS:-8}"
     make install) > "$tools/libfabric-build.log" 2>&1
fi
make -C "$tools/src/gdrcopy-2.5" -j"${JOBS:-8}" lib_install prefix="$tools/gdrcopy" > "$tools/gdrcopy-build.log" 2>&1
# CUDA redistributable layout uses lib; upstream Rust sys crates expect lib64.
[[ -e "$tools/cuda/lib64" ]] || ln -s lib "$tools/cuda/lib64"
# Headers can be provided by the host or the project-local extracted Debian dev package.
if [[ ! -f "$tools/rdma/usr/include/infiniband/verbs.h" ]]; then
    mkdir -p "$tools/debs"
    (cd "$tools/debs"; apt download libibverbs-dev libnl-3-dev libnl-route-3-dev)
    for package in "$tools"/debs/*.deb; do dpkg-deb -x "$package" "$tools/rdma"; done
fi
mkdir -p "$tools/rdma/usr/lib"
ibverbs="$(ldconfig -p | awk '/libibverbs.so.1 / {print $NF; exit}')"
[[ -n "$ibverbs" ]] || { echo 'Host libibverbs.so.1 is required.' >&2; exit 1; }
ln -sfn "$ibverbs" "$tools/rdma/usr/lib/libibverbs.so"
# Rust1.91, CUDA12.8 redistributables and Python libclang must already be installed.
source "$root/scripts/native_env.sh"
cargo +1.91.0 --version
nvcc --version
[[ -f "$LIBCLANG_PATH/libclang.so" ]]
# Compile the exact Cargo-pinned upstream revision in an independent source tree.
rev=1ed972ed3f0bd5616c997c9507c25616c63394fc
upstream="$tools/src/pplx-garden-${rev:0:7}"
if [[ ! -d "$upstream" ]]; then
    git clone https://github.com/perplexityai/pplx-garden "$upstream"
    git -C "$upstream" checkout --detach "$rev"
fi
if [[ -d "$upstream/.git" ]]; then
    [[ "$(git -C "$upstream" rev-parse HEAD)" == "$rev" ]]
fi
# Diagnostic-only compatibility patch; Cargo's production checkout stays untouched.
if ! grep -q 'BONSAI_DEBUG_DIRECT_VERBS' "$upstream/fabric-lib/fabric-debug/src/main.rs"; then
    patch -d "$upstream" -p1 < "$root/scripts/cluster/pplx-topology.patch"
fi
cargo +1.91.0 build --locked --manifest-path "$upstream/Cargo.toml" -p fabric-debug -j"${JOBS:-8}"
echo "Native dependencies and pinned fabric-debug are ready; source $root/scripts/native_env.sh"
