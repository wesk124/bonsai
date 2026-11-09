#!/usr/bin/env bash
#SBATCH --job-name=bonsai-transfer-pilot
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/transfer-pilot-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
source scripts/native_env.sh
export HF_HOME=/mnt/home/jiechen7/projects/HW/.cache/huggingface
export PYTHONPATH=$PWD/bench
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
port=$((20000 + SLURM_JOB_ID % 30000))
receiver_log="logs/transfer-pilot-receiver-$SLURM_JOB_ID.log"
srun --exclusive --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" --cpu-bind=none \
    ../.venv/bin/python -m bonsai_bench.worker --listen "0.0.0.0:$port" \
    --device cuda:0 --transport fabric-lib --once > "$receiver_log" 2>&1 &
receiver_pid=$!
cleanup() { kill "$receiver_pid" 2>/dev/null || true; wait "$receiver_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=false
for ((i=0; i<120; i++)); do
    if grep -q '^READY ' "$receiver_log"; then ready=true; break; fi
    kill -0 "$receiver_pid" 2>/dev/null || { cat "$receiver_log"; exit 1; }
    sleep 1
done
[[ "$ready" == true ]] || { cat "$receiver_log"; exit 1; }
srun --exclusive --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" --cpu-bind=none \
    ../.venv/bin/python -m bonsai_bench.runner \
    --manifest data/ruler/pilot-smoke.jsonl --out "results/pilot-transfer-$SLURM_JOB_ID.jsonl" \
    --receiver "${nodes[1]}:$port" --transport fabric-lib --device cuda:0 \
    --condition local_bf16:bf16 --condition local_roundtrip:int8 --condition local_roundtrip:int4 \
    --condition transfer:bf16 --condition transfer:int8 --condition transfer:int4 \
    --condition text_refill:bf16
wait "$receiver_pid"
trap - EXIT
