#!/usr/bin/env bash
#SBATCH --job-name=bonsai-register-probe
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/register-probe-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
source scripts/native_env.sh
srun --cpu-bind=none env PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../.venv/bin/python scripts/cluster/register_probe.py
