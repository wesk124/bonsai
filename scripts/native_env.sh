#!/usr/bin/env bash
# Source this file from either a login node (build) or an allocated GPU node (run).
_BONSAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BONSAI_TOOLS="${BONSAI_TOOLS:-$(dirname "$_BONSAI_ROOT")/.tools}"
export CARGO_HOME="$BONSAI_TOOLS/cargo"
export RUSTUP_HOME="$BONSAI_TOOLS/rustup"
export CUDA_HOME="$BONSAI_TOOLS/cuda"
export CUDA_PATH="$CUDA_HOME"
export GDRAPI_HOME="$BONSAI_TOOLS/gdrcopy"
export LIBFABRIC_HOME="$BONSAI_TOOLS/libfabric"
export LIBIBVERBS_HOME="$BONSAI_TOOLS/rdma/usr"
export PATH="$CARGO_HOME/bin:$CUDA_HOME/bin:$LIBFABRIC_HOME/bin:$PATH"
if [[ -z "${LIBCLANG_PATH:-}" ]]; then
    for _bonsai_clang in "$(dirname "$_BONSAI_ROOT")"/.venv/lib/python*/site-packages/clang/native; do
        [[ ! -f "$_bonsai_clang/libclang.so" ]] || export LIBCLANG_PATH="$_bonsai_clang"
    done
fi
# The libclang wheel has no resource headers; reuse the installed C compiler headers.
_bonsai_gcc_include="$(gcc -print-file-name=include)"
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $_bonsai_gcc_include${BINDGEN_EXTRA_CLANG_ARGS:+ $BINDGEN_EXTRA_CLANG_ARGS}"
# Real large-cache MR probes (1294359) fail for fragmented expandable VMM
# allocations and pass for conventional cudaMalloc allocations. Set both aliases
# before Python creates a CUDA context; this is part of the measured protocol.
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Link-only CUDA stubs belong in LIBRARY_PATH, never LD_LIBRARY_PATH.
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:$GDRAPI_HOME/lib:$LIBFABRIC_HOME/lib:$LIBIBVERBS_HOME/lib:$LIBIBVERBS_HOME/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Use host RDMA providers/driver; the local RDMA prefix supplies build headers only.
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$GDRAPI_HOME/lib:$LIBFABRIC_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$LIBFABRIC_HOME/lib/pkgconfig:$LIBIBVERBS_HOME/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
unset _BONSAI_ROOT _bonsai_clang _bonsai_gcc_include
