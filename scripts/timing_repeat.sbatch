#!/usr/bin/env bash
#SBATCH --job-name=bonsai-timing-repeat
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/timing-repeat-%A_%a.log
set -euo pipefail
root=/mnt/home/jiechen7/projects/HW/bonsai
cd "$root"
run_dir=${BONSAI_RUN_DIR:?Set BONSAI_RUN_DIR to the frozen final source}
export BONSAI_TOOLS="$root/../.tools"
source "$run_dir/source/scripts/native_env.sh"
export HF_HOME="$root/../.cache/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONPATH="$run_dir/source/bench"
"$root/../.venv/bin/python" "$run_dir/source/scripts/verify_frozen_run.py" "$run_dir"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
port=$((20000 + SLURM_JOB_ID % 30000))
mkdir -p "$run_dir/timing" "$run_dir/logs"
receiver_log="$run_dir/logs/timing-receiver-$SLURM_ARRAY_TASK_ID-$SLURM_JOB_ID.log"
srun --exclusive --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" --cpu-bind=none \
    "$root/../.venv/bin/python" -m bonsai_bench.worker --listen "0.0.0.0:$port" \
    --device cuda:0 --transport fabric-lib --once > "$receiver_log" 2>&1 &
receiver_pid=$!
cleanup() { kill "$receiver_pid" 2>/dev/null || true; wait "$receiver_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=false
for ((attempt=0; attempt<120; attempt++)); do
    if grep -q '^READY ' "$receiver_log"; then ready=true; break; fi
    kill -0 "$receiver_pid" 2>/dev/null || { cat "$receiver_log"; exit 1; }
    sleep 1
done
[[ "$ready" == true ]] || { cat "$receiver_log"; exit 1; }
srun --exclusive --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" --cpu-bind=none \
    "$root/../.venv/bin/python" -m bonsai_bench.runner \
    --manifest "$run_dir/data/ruler/pilot-smoke.jsonl" \
    --out "$run_dir/timing/block-$SLURM_ARRAY_TASK_ID.jsonl" --warmup-samples 9 \
    --receiver "${nodes[1]}:$port" --transport fabric-lib --device cuda:0 \
    --condition transfer:bf16 --condition transfer:int8 --condition transfer:int4 \
    --condition text_refill:bf16
wait "$receiver_pid"
trap - EXIT
