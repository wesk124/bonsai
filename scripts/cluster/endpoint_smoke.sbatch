#!/usr/bin/env bash
#SBATCH --job-name=bonsai-endpoint-smoke
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/endpoint-smoke-%j.log
set -euo pipefail
root="${BONSAI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$root/scripts/native_env.sh" ]] || root="$root/bonsai"
cd "$root"
source scripts/native_env.sh
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
read -ra sizes <<< "${BONSAI_SMOKE_SIZES:-4096 65536 1048576 16777216}"
srun --cpu-bind=none --label ../.venv/bin/python scripts/cluster/endpoint_smoke.py --host "${nodes[0]}" --sizes "${sizes[@]}"
