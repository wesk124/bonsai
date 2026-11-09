#!/usr/bin/env bash
#SBATCH --job-name=bonsai-native-codec
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/native-codec-%j.log
set -euo pipefail
cd /mnt/home/jiechen7/projects/HW/bonsai
export PATH="$PWD/../.tools/cargo/bin:$PATH"
export CARGO_HOME="$PWD/../.tools/cargo"
export RUSTUP_HOME="$PWD/../.tools/rustup"
export CUDA_HOME="$PWD/../.tools/cuda"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export BONSAI_CUDA_ARCH=sm_90a
srun cargo test -p bonsai-codec --features cuda --test cuda -- --ignored --nocapture
