#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=acg_cuda_1n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.merenda2@studenti.unisa.it

set -euo pipefail

mkdir -p ./logs

export LC_ALL=C
source "$HOME/Projects/thesis/thesis_env_cuda.sh"

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_DISABLE_NCCL=1

export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_btl=^openib
export OMPI_MCA_mpi_cuda_support=1
export UCX_TLS=sm,cuda_copy,cuda_ipc,self

BINARY=${ACG_CUDA_BINARY:-$HOME/Projects/thesis/aCG/build/acg-cuda}
MTXFILE=${ACG_MATRIX:-$HOME/Projects/thesis/dataset/Bump_2911/Bump_2911.mtx}
NTRIALS=${ACG_NTRIALS:-3}
MAX_ITERATIONS=${ACG_MAX_ITERATIONS:-100000}
WARMUP=${ACG_WARMUP:-10}

[ -x "$BINARY" ] || { echo "no executable: $BINARY" >&2; exit 1; }
[ -e "$MTXFILE" ] || { echo "no such file or directory: $MTXFILE" >&2; exit 1; }

echo "job: $SLURM_JOB_NAME/$SLURM_JOB_ID"
echo "node: $(hostname)"
echo "binary: $BINARY"
echo "matrix: $MTXFILE"
echo "trials: $NTRIALS"
echo "max iterations: $MAX_ITERATIONS"
nvidia-smi || true

solve() {
    local ntrials=$1
    local solver=$2
    shift 2

    for trial in $(seq "$ntrials"); do
        local outfile="results/${solver}/suitesparse/Bump_2911/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${trial}-stdout.txt"
        local errfile="results/${solver}/suitesparse/Bump_2911/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${trial}-stderr.txt"

        mkdir -p "$(dirname "$outfile")" "$(dirname "$errfile")"

        echo "${solver} - Trial ${trial} of ${ntrials}"
        echo "stdout: ${outfile}.tmp"
        echo "stderr: ${errfile}.tmp"

        /usr/bin/time -p --verbose \
            mpirun --verbose \
            -np "$SLURM_NTASKS" \
            -map-by node:PE="${SLURM_CPUS_PER_TASK}" \
            -rank-by core \
            -bind-to none \
            "$BINARY" "$MTXFILE" \
            --verbose --verbose --verbose --output-comm-matrix \
            --manufactured-solution \
            --seed 101 \
            --residual-atol 0 \
            --residual-rtol 1e-6 \
            --max-iterations "$MAX_ITERATIONS" \
            --warmup "$WARMUP" \
            "$@" \
            >"${outfile}.tmp" 2>"${errfile}.tmp" \
        && mv --verbose "${outfile}.tmp" "$outfile" \
        && mv --verbose "${errfile}.tmp" "$errfile"
    done
}

solve "$NTRIALS" acg-cg-mpi --solver acg --comm mpi
solve "$NTRIALS" acg-cg-nccl --solver acg --comm nccl
solve "$NTRIALS" acg-cg-nvshmem --solver acg-device --comm nvshmem
