#!/usr/bin/env bash
#SBATCH --job-name=bonsai-model-integrity
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/model-cache-integrity-%j.log
set -euo pipefail
root=/mnt/home/jiechen7/projects/HW/bonsai
cd "$root"
source scripts/native_env.sh
export HF_HOME="$root/../.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/runs/final-20261107/source/bench"
run_dir="$PWD/runs/final-20261107"
../.venv/bin/python "$run_dir/source/scripts/verify_frozen_run.py" "$run_dir"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
port=$((20000 + SLURM_JOB_ID % 30000))
srun --cpu-bind=none --label ../.venv/bin/python scripts/cluster/model_cache_integrity.py \
    --host "${nodes[0]}" --port "$port" --run-dir "$run_dir"
