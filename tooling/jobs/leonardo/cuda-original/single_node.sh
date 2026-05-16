#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=acg_cuda_1n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.merenda2@studenti.unisa.it

set -euo pipefail

mkdir -p ./logs

source "$HOME/Projects/thesis/thesis_env_cuda.sh"

export LC_ALL=C
export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_DISABLE_NCCL=1

export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_btl=^openib

BINARY=${ACG_CUDA_BINARY:-$HOME/Projects/thesis/aCG/build/acg-cuda}
MTXFILE=${ACG_MATRIX:-$HOME/Projects/thesis/dataset/Bump_2911/Bump_2911.mtx}
NTRIALS=${ACG_NTRIALS:-3}

[ -x "$BINARY" ] || { echo "no executable: $BINARY" >&2; exit 1; }
[ -e "$MTXFILE" ] || { echo "no such file or directory: $MTXFILE" >&2; exit 1; }

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
            "$BINARY" "$MTXFILE" \
            --verbose --verbose --verbose -q --output-comm-matrix \
            --manufactured-solution \
            --seed 101 \
            --residual-atol 0 \
            --residual-rtol 1e-6 \
            --max-iterations 100000 \
            "$@" \
            >"${outfile}.tmp" 2>"${errfile}.tmp" \
        && mv --verbose "${outfile}.tmp" "$outfile" \
        && mv --verbose "${errfile}.tmp" "$errfile"
    done
}

solve "$NTRIALS" acg-cg-single --solver acg --comm none
