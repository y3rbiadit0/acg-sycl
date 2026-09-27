#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=nccl_test_2n4g
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
nccl_tests_root=${NCCL_TESTS_ROOT:-$HOME/nccl-tests}
all_reduce_perf=${NCCL_TESTS_ALL_REDUCE:-$nccl_tests_root/build/all_reduce_perf_intelmpi}
nranks=${NCCL_TEST_RANKS:-${SLURM_NTASKS:-8}}
run_dir="results/nccl-tests/${SLURM_JOB_NAME:-nccl_test_2n4g}-${SLURM_JOB_ID:-manual}"
mpi_lib_path=${ONECCL_MPI_LIBRARY_PATH:-$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12}
mpi_lib_dir=$(dirname "$mpi_lib_path")
libfabric_dir=${ONECCL_LIBFABRIC_DIR:-$oneccl_root/opt/mpi/libfabric/lib}
libfabric_provider_dir=${ONECCL_LIBFABRIC_PROVIDER_DIR:-$libfabric_dir/prov-tcp-only}

source "$project_root/tooling/environments/leonardo.sh"

set +u
source "$oneccl_root/env/vars.sh"
set -u

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,GRAPH}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export LD_LIBRARY_PATH="$libfabric_dir:$mpi_lib_dir:$oneccl_root/opt/mpi/lib:$CUDA_HOME/lib64:$NCCL_HOME/lib:$GCC12_LIB:${LD_LIBRARY_PATH:-}"

[ -x "$all_reduce_perf" ] || {
    echo "no executable: $all_reduce_perf" >&2
    echo "Build it with something like:" >&2
    echo "  source tooling/environments/leonardo.sh" >&2
    echo "  set +u; source \"$oneccl_root/env/vars.sh\"; set -u" >&2
    echo "  export CC=\"$GCC12_ROOT/bin/gcc\" CXX=\"$GCC12_ROOT/bin/g++\" CUDAHOSTCXX=\"$GCC12_ROOT/bin/g++\"" >&2
    echo "  make -C \"$nccl_tests_root\" -j MPI=1 NAME_SUFFIX=_intelmpi MPI_HOME=\"$oneccl_root/opt/mpi\" CUDA_HOME=\"$CUDA_HOME\" NCCL_HOME=\"$NCCL_HOME\" CXX=\"$GCC12_ROOT/bin/g++\" NVCC_GENCODE=\"-gencode=arch=compute_80,code=sm_80\" NVCCFLAGS=\"-ccbin $GCC12_ROOT/bin/g++\"" >&2
    exit 1
}

mkdir -p "$run_dir"

echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
echo "nodes: ${SLURM_NODELIST:-manual}"
echo "node: $(hostname)"
echo "project_root: $project_root"
echo "oneccl_root: $oneccl_root"
echo "nccl_tests_root: $nccl_tests_root"
echo "all_reduce_perf: $all_reduce_perf"
echo "run_dir: $run_dir"
echo "nranks: $nranks"
echo "I_MPI_HYDRA_BOOTSTRAP: $I_MPI_HYDRA_BOOTSTRAP"
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

echo "libraries used by all_reduce_perf:"
ldd "$all_reduce_perf" | grep -E 'libmpi|libnccl|libcuda|libcudart|libstdc\+\+' || true

stdout="$run_dir/all-reduce-perf-stdout.txt"
stderr="$run_dir/all-reduce-perf-stderr.txt"

echo "Running multi-node NCCL all_reduce_perf"
echo "stdout: $stdout"
echo "stderr: $stderr"

mpirun -np "$nranks" \
    "$all_reduce_perf" \
    -b "${NCCL_TEST_MIN_BYTES:-4K}" \
    -e "${NCCL_TEST_MAX_BYTES:-64M}" \
    -f "${NCCL_TEST_STEP_FACTOR:-4}" \
    -g "${NCCL_TEST_GPUS_PER_RANK:-1}" \
    -w "${NCCL_TEST_WARMUP_ITERS:-10}" \
    -n "${NCCL_TEST_ITERS:-50}" \
    -c "${NCCL_TEST_CHECK_ITERS:-1}" \
    >"$stdout" 2>"$stderr"

echo "multi-node NCCL all_reduce_perf completed"
