#!/usr/bin/env bash
# Paired implementation pilot: fresh model processes for each backend/block.
#SBATCH --job-name=bonsai-fused-pilot
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/optimization-pair-%A_%a.log
set -euo pipefail
root=/mnt/home/jiechen7/projects/HW/bonsai
cd "$root"
run_dir=${BONSAI_RUN_DIR:?Set BONSAI_RUN_DIR to a new frozen optimization run}
export BONSAI_TOOLS="$root/../.tools"
source "$run_dir/source/scripts/native_env.sh"
export HF_HOME="$root/../.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONPATH="$run_dir/source/bench"
export TRITON_CACHE_DIR="$run_dir/triton-cache"
python="$root/../.venv/bin/python"
"$python" "$run_dir/source/scripts/verify_frozen_run.py" "$run_dir" --require-native
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
port=$((20000 + SLURM_JOB_ID % 30000))
block=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$run_dir/comparison" "$run_dir/logs"
srun --cpu-bind=none --label nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,power.limit \
    --format=csv > "$run_dir/logs/block-$block-hardware.csv"
mapfile -t backends < <("$python" -c 'import random,sys; b=["pytorch_packed_v1","triton_packed_v1"]; random.Random(20261108+int(sys.argv[1])).shuffle(b); print("\n".join(b))' "$block")
"$python" -c 'import json,os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"block":int(sys.argv[2]),"backend_order":sys.argv[3:],"job_id":os.environ["SLURM_JOB_ID"],"node_list":os.environ["SLURM_JOB_NODELIST"],"warmup_prompts":9,"conditions":7},indent=2)+"\n")' \
    "$run_dir/comparison/block-$block-order.json" "$block" "${backends[@]}"
for backend in "${backends[@]}"; do
    receiver_log="$run_dir/logs/block-$block-$backend-receiver.log"
    srun --exclusive --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" --cpu-bind=none \
        "$python" -m bonsai_bench.worker --listen "0.0.0.0:$port" \
        --codec-backend "$backend" --device cuda:0 --transport fabric-lib --once > "$receiver_log" 2>&1 &
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
        "$python" -m bonsai_bench.runner --codec-backend "$backend" \
        --manifest "$run_dir/data/ruler/pilot-smoke.jsonl" \
        --out "$run_dir/comparison/block-$block-$backend.jsonl" --warmup-samples 9 \
        --receiver "${nodes[1]}:$port" --transport fabric-lib --device cuda:0 \
        --condition local_bf16:bf16 --condition local_roundtrip:int8 --condition local_roundtrip:int4 \
        --condition transfer:bf16 --condition transfer:int8 --condition transfer:int4 \
        --condition text_refill:bf16
    wait "$receiver_pid"
    trap - EXIT
    "$python" "$run_dir/source/scripts/check_handoff_equivalence.py" \
        "$run_dir/comparison/block-$block-$backend.jsonl" --expected-prompts 9
done
