#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=test_oneccl_nccl_backend
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p ./logs

export LC_ALL=C
project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
oneccl_root=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}
nccl_examples_dir=${ONECCL_NCCL_EXAMPLES_DIR:-$oneccl_root/examples/nccl}
nranks=${ONECCL_TEST_RANKS:-${SLURM_NTASKS:-2}}
run_dir="results/oneccl-nccl/${SLURM_JOB_NAME:-test_oneccl_nccl_backend}-${SLURM_JOB_ID:-manual}"
mpi_lib_path=${ONECCL_MPI_LIBRARY_PATH:-$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12}
mpi_lib_dir=$(dirname "$mpi_lib_path")

source "$project_root/tooling/environments/leonardo.sh"

set +u
source "$oneccl_root/env/vars.sh"
set -u

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
export SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-cuda}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

export CCL_BACKEND=${CCL_BACKEND:-nccl}
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-info}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
export CCL_MPI_LIBRARY_PATH=${CCL_MPI_LIBRARY_PATH:-$mpi_lib_path}
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-auto}

export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL,GRAPH}

export LD_LIBRARY_PATH="$mpi_lib_dir:$GCC12_LIB:$DPCPP_INSTALL/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

[ -x "$nccl_examples_dir/nccl_allreduce_test" ] || { echo "no executable: $nccl_examples_dir/nccl_allreduce_test" >&2; exit 1; }
[ -x "$nccl_examples_dir/nccl_barrier_test" ] || { echo "no executable: $nccl_examples_dir/nccl_barrier_test" >&2; exit 1; }
mkdir -p "$run_dir"

echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
echo "node: $(hostname)"
echo "project_root: $project_root"
echo "oneccl_root: $oneccl_root"
echo "nccl_examples_dir: $nccl_examples_dir"
echo "run_dir: $run_dir"
echo "nranks: $nranks"
echo "CCL_BACKEND: $CCL_BACKEND"
echo "CCL_LOG_LEVEL: $CCL_LOG_LEVEL"
echo "CCL_ATL_TRANSPORT: $CCL_ATL_TRANSPORT"
echo "CCL_MPI_LIBRARY_PATH: $CCL_MPI_LIBRARY_PATH"
echo "I_MPI_HYDRA_BOOTSTRAP: $I_MPI_HYDRA_BOOTSTRAP"
echo "I_MPI_FABRICS: $I_MPI_FABRICS"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "ONEAPI_DEVICE_SELECTOR: $ONEAPI_DEVICE_SELECTOR"
echo "SYCL_DEVICE_FILTER: $SYCL_DEVICE_FILTER"

nvidia-smi || true

echo "compiler and MPI diagnostics:"
which clang++ || true
clang++ --version || true
which mpirun || true
mpirun --version || true
which mpicc || true
mpicc -show || true

echo "SYCL devices:"
sycl-ls || true
mpirun -np "$nranks" sycl-ls || true

echo "selected environment:"
env | grep -E 'NCCL|CUDA|MPI|I_MPI|PMI|DPCPP|SYCL|GCC|CCL' | sort || true

inspect_binary() {
    local binary=$1

    echo "libraries used by $binary:"
    ldd "$binary" | grep -E 'libstdc\+\+|libsycl|libccl|libnccl|libmpi|libcuda|libcudart' || true
}

run_mpi() {
    mpirun -np "$nranks" \
        "$project_root/tooling/jobs/gpu-rank-wrapper.sh" \
        "$@"
}

run_test() {
    local name=$1
    shift
    local stdout="$run_dir/${name}-stdout.txt"
    local stderr="$run_dir/${name}-stderr.txt"

    echo "Running $name"
    echo "stdout: $stdout"
    echo "stderr: $stderr"

    if run_mpi "$@" >"$stdout" 2>"$stderr"; then
        echo "$name completed"
    else
        local status=$?
        echo "$name failed with status $status" >&2
        echo "stdout: $stdout" >&2
        echo "stderr: $stderr" >&2
        return "$status"
    fi
}

inspect_binary "$nccl_examples_dir/nccl_allreduce_test"
run_test nccl-api-allreduce "$nccl_examples_dir/nccl_allreduce_test"

inspect_binary "$nccl_examples_dir/nccl_barrier_test"
run_test nccl-api-barrier "$nccl_examples_dir/nccl_barrier_test"

echo "oneCCL NCCL backend examples completed"
