#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=oneccl_nccl_perf_2n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p ./logs

export LC_ALL=C
project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
oneccl_root=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}
perf_binary=${ONECCL_NCCL_PERF_BINARY:-$oneccl_root/examples/nccl/nccl_backend_allreduce_perf}
nranks=${ONECCL_TEST_RANKS:-${SLURM_NTASKS:-8}}
run_dir="results/oneccl-nccl-perf/${SLURM_JOB_NAME:-oneccl_nccl_perf_2n4g}-${SLURM_JOB_ID:-manual}"
mpi_lib_path=${ONECCL_MPI_LIBRARY_PATH:-$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12}
mpi_lib_dir=$(dirname "$mpi_lib_path")
libfabric_dir=${ONECCL_LIBFABRIC_DIR:-$oneccl_root/opt/mpi/libfabric/lib}
libfabric_provider_dir=${ONECCL_LIBFABRIC_PROVIDER_DIR:-$libfabric_dir/prov-tcp-only}
elem_counts=${ONECCL_TEST_ELEM_COUNTS:-1024,4096,16384,65536,262144,1048576,4194304,16777216}
iters=${ONECCL_TEST_ITERS:-50}
warmup_iters=${ONECCL_TEST_WARMUP_ITERS:-10}
check=${ONECCL_TEST_CHECK:-last}

source "$project_root/tooling/environments/leonardo.sh"

set +u
source "$oneccl_root/env/vars.sh"
set -u

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
export SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-cuda}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export CCL_BACKEND=${CCL_BACKEND:-nccl}
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
export CCL_MPI_LIBRARY_PATH=${CCL_MPI_LIBRARY_PATH:-$mpi_lib_path}
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-auto}

export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
export I_MPI_FABRICS=${I_MPI_FABRICS:-shm:ofi}
export I_MPI_DEBUG=${I_MPI_DEBUG:-0}
export I_MPI_OFI_PROVIDER=${I_MPI_OFI_PROVIDER:-tcp}

if [ ! -e "$libfabric_provider_dir/libtcp-fi.so" ] && [ -e "$libfabric_dir/prov/libtcp-fi.so" ]; then
    mkdir -p "$libfabric_provider_dir"
    ln -sf "$libfabric_dir/prov/libtcp-fi.so" "$libfabric_provider_dir/libtcp-fi.so"
fi

export FI_PROVIDER=${FI_PROVIDER:-tcp}
export FI_PROVIDER_PATH=${FI_PROVIDER_PATH:-$libfabric_provider_dir}
export FI_LOG_LEVEL=${FI_LOG_LEVEL:-error}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL,GRAPH}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export LD_LIBRARY_PATH="$libfabric_dir:$mpi_lib_dir:$GCC12_LIB:$DPCPP_INSTALL/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

[ -x "$perf_binary" ] || {
    echo "no executable: $perf_binary" >&2
    echo "Build it with:" >&2
    echo "  tooling/jobs/leonardo/oneccl-nccl/compile-nccl-backend-perf.sh" >&2
    exit 1
}

mkdir -p "$run_dir"

echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
echo "nodes: ${SLURM_NODELIST:-manual}"
echo "node: $(hostname)"
echo "project_root: $project_root"
echo "oneccl_root: $oneccl_root"
echo "perf_binary: $perf_binary"
echo "run_dir: $run_dir"
echo "nranks: $nranks"
echo "elem_counts: $elem_counts"
echo "iters: $iters"
echo "warmup_iters: $warmup_iters"
echo "check: $check"
echo "CCL_BACKEND: $CCL_BACKEND"
echo "CCL_LOG_LEVEL: $CCL_LOG_LEVEL"
echo "I_MPI_FABRICS: $I_MPI_FABRICS"
echo "I_MPI_OFI_PROVIDER: $I_MPI_OFI_PROVIDER"
echo "FI_PROVIDER: $FI_PROVIDER"
echo "FI_PROVIDER_PATH: $FI_PROVIDER_PATH"
echo "FI_LOG_LEVEL: $FI_LOG_LEVEL"
echo "libfabric_dir: $libfabric_dir"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi || true

echo "MPI diagnostics:"
which mpirun || true
mpirun --version || true
which mpicc || true
mpicc -show || true

echo "rank placement:"
mpirun -np "$nranks" env | grep -E 'MPI_LOCALRANKID|PMI_RANK|PMI_SIZE|CUDA_VISIBLE_DEVICES|NCCL|I_MPI' | sort || true

echo "libraries used by $perf_binary:"
ldd "$perf_binary" | grep -E 'libstdc\+\+|libsycl|libccl|libnccl|libmpi|libcuda|libcudart' || true

stdout="$run_dir/nccl-backend-allreduce-perf-stdout.txt"
stderr="$run_dir/nccl-backend-allreduce-perf-stderr.txt"

echo "Running multi-node oneCCL NCCL backend allreduce perf"
echo "stdout: $stdout"
echo "stderr: $stderr"

mpirun -np "$nranks" \
    "$project_root/tooling/jobs/gpu-rank-wrapper.sh" \
    "$perf_binary" \
    --elem_counts "$elem_counts" \
    --iters "$iters" \
    --warmup_iters "$warmup_iters" \
    --check "$check" \
    >"$stdout" 2>"$stderr"

echo "multi-node oneCCL NCCL backend allreduce perf completed"
