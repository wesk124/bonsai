#!/usr/bin/env bash
#SBATCH --job-name=bonsai-register-alignment
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/register-alignment-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
source scripts/native_env.sh
for config in expandable_segments:True expandable_segments:False; do
    srun --cpu-bind=none env PYTORCH_CUDA_ALLOC_CONF="$config" PYTORCH_ALLOC_CONF="$config" ../.venv/bin/python scripts/cluster/register_alignment_probe.py
done
