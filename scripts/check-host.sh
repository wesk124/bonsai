#!/usr/bin/env bash
# Preflight for a fabric-lib / Bonsai node. Run on the HOST (and again inside the
# container). Exits non-zero if a hard requirement is missing.
#   scripts/check-host.sh
set -uo pipefail

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
have() { command -v "$1" >/dev/null 2>&1; }
in_container() { [[ -f /.dockerenv ]] || grep -qa docker /proc/1/cgroup 2>/dev/null; }

echo "== OS / kernel"
[[ $(uname -s) == Linux ]] && ok "Linux" || bad "fabric-lib needs Linux (got $(uname -s))"
kver=$(uname -r | cut -d- -f1)
if [[ $(printf '%s\n5.12\n' "$kver" | sort -V | head -1) == 5.12 ]]; then
    ok "kernel $kver (>= 5.12, DMA-BUF supported)"
else
    warn "kernel $kver < 5.12: no DMA-BUF; GPUDirect needs nvidia-peermem instead"
fi

echo "== GPU"
if have nvidia-smi; then
    ok "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader | sed 's/^/        /'
    cuda_drv=$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | awk '{print $3}')
    if [[ -n $cuda_drv && $(printf '%s\n12.8\n' "$cuda_drv" | sort -V | head -1) == 12.8 ]]; then
        ok "driver supports CUDA $cuda_drv (>= 12.8)"
    else
        bad "driver supports CUDA ${cuda_drv:-?}; need >= 12.8 (update the NVIDIA driver)"
    fi
else
    bad "nvidia-smi not found (NVIDIA driver missing)"
fi
if in_container || have nvcc; then
    have nvcc && ok "nvcc $(nvcc --version | grep -oE 'release [0-9.]+' | awk '{print $2}')" || warn "nvcc not found"
fi

echo "== RDMA NICs"
if ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
    ok "$(ls /dev/infiniband/uverbs* | wc -l) uverbs device(s): $(ls /dev/infiniband/ | tr '\n' ' ')"
else
    bad "/dev/infiniband/uverbs* missing (RDMA driver not loaded: mlx5_ib for ConnectX, efa for AWS)"
fi
if have ibv_devinfo; then
    ibv_devinfo -l 2>/dev/null | sed 's/^/        /'
    ibv_devinfo 2>/dev/null | grep -E 'hca_id|state:|link_layer|active_mtu' | sed 's/^/        /'
else
    warn "ibv_devinfo not installed (apt install ibverbs-utils) - skipping port state check"
fi
if have fi_info; then
    if fi_info -p efa >/dev/null 2>&1; then ok "libfabric EFA provider available"
    else warn "libfabric EFA provider not found (fine on ConnectX; fabric-lib uses verbs there)"; fi
fi
lsmod 2>/dev/null | grep -qE '^(mlx5_ib|efa) ' && ok "RDMA kernel module: $(lsmod | grep -oE '^(mlx5_ib|efa)' | tr '\n' ' ')" \
    || { in_container || warn "neither mlx5_ib nor efa module listed by lsmod"; }

echo "== GPUDirect RDMA"
if lsmod 2>/dev/null | grep -q '^nvidia_peermem'; then
    ok "nvidia_peermem loaded"
elif [[ -e /sys/module/nvidia/parameters ]] && grep -qs . /proc/driver/nvidia/version; then
    warn "nvidia_peermem not loaded; OK if DMA-BUF works (kernel >= 5.12 + open GPU kernel modules)"
fi
if grep -qs "Open Kernel Module" /proc/driver/nvidia/version; then
    ok "NVIDIA open kernel modules (DMA-BUF capable)"
else
    in_container || warn "proprietary NVIDIA kernel module: DMA-BUF may be unavailable -> load nvidia_peermem"
fi

echo "== GDRCopy"
[[ -e /dev/gdrdrv ]] && ok "/dev/gdrdrv present" || bad "/dev/gdrdrv missing (install gdrcopy and load gdrdrv on the host)"
if in_container; then
    ldconfig -p | grep -q libgdrapi && ok "libgdrapi installed" || bad "libgdrapi missing in container"
fi

echo "== Limits / capabilities"
ml=$(ulimit -l)
[[ $ml == unlimited ]] && ok "memlock unlimited" || bad "memlock = $ml KB; need unlimited (--ulimit memlock=-1 or /etc/security/limits.conf)"
if have capsh; then
    caps=$(capsh --print | grep Current)
    for c in cap_sys_admin cap_sys_ptrace cap_ipc_lock; do
        [[ $caps == *$c* || $(id -u) == 0 && ! -f /.dockerenv ]] && ok "$c" || warn "$c not in current capability set (needed for pidfd_getfd)"
    done
fi

echo "== Topology (each GPU should have its own NIC under the same PCIe switch)"
have nvidia-smi && nvidia-smi topo -m 2>/dev/null | sed 's/^/        /'

if ! in_container; then
    echo "== Docker"
    if have docker; then
        ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
        docker info 2>/dev/null | grep -qi nvidia && ok "NVIDIA container runtime registered" \
            || warn "NVIDIA container toolkit not detected (needed for --gpus all)"
    else
        warn "docker not installed (bare-metal build is also possible, see docs/BUILD.md)"
    fi
fi

echo
[[ $fail -eq 0 ]] && echo "All hard requirements met." || echo "Some hard requirements are missing (FAIL above)."
exit $fail
