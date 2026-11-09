#!/usr/bin/env bash
#SBATCH --job-name=bonsai-preflight
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/preflight-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
srun --label bash -c '
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi topo -m
ls -l /dev/infiniband /dev/gdrdrv 2>/dev/null || true
ulimit -l
command -v nvcc || true
ldconfig -p | grep -E "gdrapi|fabric|ibverbs" || true
../.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.randn(4,device=\"cuda\").cpu())"
'
