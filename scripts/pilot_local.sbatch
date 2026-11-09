#!/usr/bin/env bash
#SBATCH --job-name=bonsai-ruler-pilot
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/ruler-pilot-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
export HF_HOME=/mnt/home/jiechen7/projects/HW/.cache/huggingface
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PWD/bench
srun ../.venv/bin/python -m bonsai_bench.runner \
    --manifest data/ruler/pilot/*/*.jsonl \
    --out results/pilot-local-v2.jsonl --device cuda:0
