#!/usr/bin/env bash
#SBATCH --job-name=bonsai-cuda-tests
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/native-cuda-tests-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
source scripts/native_env.sh
export BONSAI_CUDA_ARCH=sm_90a
cargo test -p bonsai-codec --features cuda --test cuda -- --ignored --nocapture
cargo test -p bonsai-dataplane --features cuda --test cuda_gpu -- --ignored --nocapture
