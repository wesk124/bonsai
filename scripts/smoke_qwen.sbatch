#!/usr/bin/env bash
#SBATCH --job-name=bonsai-qwen-smoke
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/qwen-smoke-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
export HF_HOME=/mnt/home/jiechen7/projects/HW/.cache/huggingface
export OMP_NUM_THREADS=4
srun ../.venv/bin/python scripts/smoke_qwen.py
