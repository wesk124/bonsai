# Bonsai dev/benchmark image: fabric-lib (pplx-garden) + Bonsai + HF Transformers.
#
# Derived from pplx-garden's docker/dev.Dockerfile. Bonsai changes:
#   * pplx-garden pinned to the same commit as Cargo.toml and built in the image
#     (fabric-lib + fabric-debug), so a broken toolchain shows up at build time
#   * GPU arch is a build arg (default covers A100, H100, B200)
#   * Python deps for the Qwen2.5 workers and the LongBench/RULER eval
#
# Build:  docker build -t bonsai-dev -f docker/Dockerfile .
# Run:    scripts/run-docker.sh     (or: docker compose -f docker/compose.yaml run --rm bonsai)
#
# The image provides user-space libraries only. The HOST must provide the GPU
# driver, the RDMA NIC kernel driver, the gdrdrv module, and DMA-BUF or
# nvidia-peermem. See docs/BUILD.md and scripts/check-host.sh.

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04

# ---------- GDRCopy user-space library (.deb) ----------
FROM ${CUDA_IMAGE} AS gdrcopy-builder
ARG GDRCOPY_VERSION=2.5.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential devscripts debhelper fakeroot pkg-config wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar -xf v${GDRCOPY_VERSION}.tar.gz && \
    cd gdrcopy-${GDRCOPY_VERSION}/packages && \
    CUDA=/usr/local/cuda ./build-deb-packages.sh -t -k && \
    mkdir -p /out && cp /tmp/gdrcopy-${GDRCOPY_VERSION}/packages/libgdrapi_*.deb /out/

# ---------- final image ----------
FROM ${CUDA_IMAGE}

# A100 = 8.0, H100/H200 = 9.0a, B200 = 10.0a. Override with --build-arg to save build time.
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0a+PTX"
ARG TORCH_VERSION=2.9.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129
ARG EFA_INSTALLER_VERSION=1.44.0
ARG RUST_TOOLCHAIN=1.91.0
ARG PPLX_GARDEN_REV=1ed972ed3f0bd5616c997c9507c25616c63394fc

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Toolchain. libclang-dev is required: every fabric-lib *-sys crate runs bindgen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config patchelf git curl wget ca-certificates \
        libclang-dev clang-18 libssl-dev \
        python3-dev python3-pip python3-venv python3-setuptools python3-build \
        # diagnostics used by scripts/check-host.sh inside the container
        pciutils iproute2 numactl \
    && rm -rf /var/lib/apt/lists/*

# libfabric + rdma-core (libibverbs, mlx5 and efa providers) via the AWS EFA installer.
# Works for both AWS EFA and NVIDIA ConnectX (rdma-core includes the mlx5 provider).
# --skip-kmod: kernel modules come from the host.
RUN cd /tmp && \
    curl -sO https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && apt-get update && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /var/lib/apt/lists/* /tmp/aws-efa-installer* && ldconfig
ENV LIBFABRIC_HOME=/opt/amazon/efa \
    LIBIBVERBS_HOME=/usr \
    CUDA_HOME=/usr/local/cuda \
    GDRAPI_HOME=/usr \
    PATH="/opt/amazon/efa/bin:${PATH}" \
    NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent

# GDRCopy user-space library (the gdrdrv kernel module must be loaded on the host).
COPY --from=gdrcopy-builder /out/ /tmp/gdrcopy/
RUN dpkg -i /tmp/gdrcopy/libgdrapi_*.deb && rm -rf /tmp/gdrcopy

# Rust (same toolchain as pplx-garden and Bonsai).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain ${RUST_TOOLCHAIN} --component rustfmt,clippy
ENV PATH="/root/.cargo/bin:${PATH}" CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup

# PyTorch + model / eval dependencies.
RUN python3 -m pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} && \
    python3 -m pip install --no-cache-dir \
        numpy ninja maturin pytest \
        "transformers>=4.45" accelerate datasets sentencepiece \
        pandas matplotlib

# pplx-garden at the pinned commit: build fabric-lib and the fabric-debug benchmark.
RUN git clone https://github.com/perplexityai/pplx-garden.git /opt/pplx-garden && \
    cd /opt/pplx-garden && git checkout ${PPLX_GARDEN_REV} && \
    cargo build --release -p fabric-lib -p fabric-debug && \
    cp target/release/fabric-debug /usr/local/bin/
# Optional: pplx_garden Python wheel (fabric-lib Python bindings), handy for quick experiments.
RUN cd /opt/pplx-garden && \
    TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
    python3 -m build --wheel --no-isolation && \
    python3 -m pip install --no-cache-dir dist/*.whl || \
    echo "WARN: pplx_garden wheel build failed (not needed by Bonsai)"

WORKDIR /app
CMD ["/bin/bash"]
