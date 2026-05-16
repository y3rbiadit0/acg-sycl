#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=acg_sycl_2n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.merenda2@studenti.unisa.it

set -euo pipefail

mkdir -p ./logs

export LC_ALL=C
project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "$project_root/tooling/environments/leonardo.sh"

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export OMPI_MCA_mpi_cuda_support=1
export UCX_TLS=${ACG_SYCL_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
export UCX_RNDV_SCHEME=${ACG_SYCL_UCX_RNDV_SCHEME:-get_zcopy}
export UCX_RNDV_THRESH=${ACG_SYCL_UCX_RNDV_THRESH:-16384}

BINARY=${ACG_SYCL_BINARY:-$project_root/build-release/acg}
MTXFILE=${ACG_MATRIX:-$project_root/data/matrices/Bump_2911/Bump_2911.mtx}
NTRIALS=${ACG_NTRIALS:-3}
MAX_ITERATIONS=${ACG_MAX_ITERATIONS:-100000}
EXTRA_ARGS=${ACG_EXTRA_ARGS:-}

[ -x "$BINARY" ] || { echo "no executable: $BINARY" >&2; exit 1; }
[ -e "$MTXFILE" ] || { echo "no such file or directory: $MTXFILE" >&2; exit 1; }

echo "nodes: $SLURM_NODELIST"
echo "tasks: $SLURM_NTASKS"
echo "binary: $BINARY"
echo "matrix: $MTXFILE"
echo "trials: $NTRIALS"
echo "max iterations: $MAX_ITERATIONS"
echo "UCX_TLS: $UCX_TLS"
echo "UCX_RNDV_SCHEME: $UCX_RNDV_SCHEME"
echo "UCX_RNDV_THRESH: $UCX_RNDV_THRESH"
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
            srun --cpu-freq=high \
            -N 2 \
            --ntasks-per-node=4 \
            "$project_root/tooling/jobs/gpu-rank-wrapper.sh" \
            "$BINARY" \
            --matrix "$MTXFILE" \
            --device gpu \
            --mpi-mode gpu-aware \
            --seed 101 \
            --residual-atol 0 \
            --residual-rtol 1e-6 \
            --max-iters "$MAX_ITERATIONS" \
            --manufactured-solution \
            "$@" \
            $EXTRA_ARGS \
            >"${outfile}.tmp" 2>"${errfile}.tmp" \
        && mv --verbose "${outfile}.tmp" "$outfile" \
        && mv --verbose "${errfile}.tmp" "$errfile"
    done
}

solve "$NTRIALS" acg-sycl-mpi

# Future SYCL communicator runs can be enabled here without changing naming/layout.
# solve "$NTRIALS" acg-sycl-nccl <future SYCL NCCL args>
