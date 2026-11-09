#!/usr/bin/env bash
#SBATCH --job-name=bonsai-fused-codec
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/fused-codec-bench-%j.log
set -euo pipefail
root=/mnt/home/jiechen7/projects/HW/bonsai
cd "$root"
source scripts/native_env.sh
export HF_HOME="$root/../.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONPATH="$root/bench"
# Keep normal Triton disk-cache behavior. Warmup is outside every interval and
# the benchmark rejects any JIT cache miss/loading request during timing.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$root/../.cache/triton/fused-codec-bench}"
out_dir="${BONSAI_FUSED_BENCH_OUT_DIR:-$root/runs/fused-bench-${SLURM_JOB_ID}}"
srun --cpu-bind=none ../.venv/bin/python scripts/cluster/fused_codec_bench.py \
    --manifest data/ruler/pilot-smoke.jsonl --out-dir "$out_dir" \
    --repeats "${BONSAI_FUSED_BENCH_REPEATS:-5}" --warmup "${BONSAI_FUSED_BENCH_WARMUP:-2}" "$@"
