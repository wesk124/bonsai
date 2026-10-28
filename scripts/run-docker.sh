#!/usr/bin/env bash
# Start the Bonsai dev container with RDMA, GDRCopy and GPU access (mirrors pplx-garden).
# Usage: scripts/run-docker.sh [command...]     e.g. scripts/run-docker.sh fabric-debug 0 1
set -euo pipefail

root_dir=$(realpath "$(dirname "$0")/..")
image=${BONSAI_IMAGE:-bonsai-dev}

devices=()
for path in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
    [[ -e $path ]] && devices+=(--device="$path")
done
if [[ ${#devices[@]} -eq 0 ]]; then
    echo "WARN: no /dev/infiniband devices found; RDMA will not work in the container" >&2
fi
if [[ -e /dev/gdrdrv ]]; then
    devices+=(--device=/dev/gdrdrv)
else
    echo "WARN: /dev/gdrdrv missing; load the gdrdrv module on the host (see docs/BUILD.md)" >&2
fi

tty_flags=(-i)
[[ -t 0 ]] && tty_flags=(-it)

set -x
exec docker run --rm "${tty_flags[@]}" --name="bonsai-$(hostname -s)" \
    -v "$root_dir:/app" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    --init \
    --shm-size=32g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus=all \
    "${devices[@]}" \
    --cap-add=IPC_LOCK \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --network host \
    "$image" "$@"
