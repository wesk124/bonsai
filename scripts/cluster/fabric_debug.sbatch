#!/usr/bin/env bash
#SBATCH --job-name=bonsai-fabric-debug
#SBATCH --account=cw-sup
#SBATCH --partition=hpc-high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fabric-debug-%j.log
set -euo pipefail
root="${BONSAI_ROOT:-$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)}"
[[ -f "$root/scripts/native_env.sh" ]] || root="$root/bonsai"
cd "$root"
source scripts/native_env.sh
export BONSAI_DEBUG_DIRECT_VERBS=1
binary="$BONSAI_TOOLS/src/pplx-garden-1ed972e/target/debug/fabric-debug"
[[ -x "$binary" ]]
run_dir="$root/logs/fabric-debug-$SLURM_JOB_ID"
mkdir -p "$run_dir"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} == 2 ]]
# Each task sees one scheduler-selected physical GPU, exposed as CUDA device0.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none --label bash -c 'hostname; nvidia-smi --query-gpu=name,driver_version --format=csv; test -c /dev/gdrdrv; test -d /dev/infiniband; ulimit -l'
srun --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" --cpu-bind=none "$binary" 0 1 > "$run_dir/server.log" 2>&1 &
server_step=$!
cleanup() { kill "$server_step" 2>/dev/null || true; wait "$server_step" 2>/dev/null || true; }
trap cleanup EXIT
address=''
for ((attempt=0; attempt<60; attempt++)); do
    address="$(sed -n 's/^Main Address: //p' "$run_dir/server.log" | head -1)"
    [[ -z "$address" ]] || break
    kill -0 "$server_step" 2>/dev/null || { cat "$run_dir/server.log"; exit 1; }
    sleep 1
done
[[ -n "$address" ]] || { cat "$run_dir/server.log"; echo 'Server did not advertise address' >&2; exit 1; }
srun --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" --cpu-bind=none timeout 240 "$binary" 0 1 "$address" > "$run_dir/client.log" 2>&1
cat "$run_dir/client.log"
[[ $(grep -c 'VERIFIED' "$run_dir/client.log") == 3 ]]
grep -q '^Done!$' "$run_dir/client.log"
echo "FABRIC_DEBUG_PASS job=$SLURM_JOB_ID nodes=${nodes[*]} logs=$run_dir"
