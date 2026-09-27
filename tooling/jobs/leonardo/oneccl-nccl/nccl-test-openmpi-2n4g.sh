#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=nccl_ompi_ucx_2n4g
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
nccl_tests_root=${NCCL_TESTS_ROOT:-$HOME/nccl-tests}
all_reduce_perf=${NCCL_TESTS_ALL_REDUCE:-$nccl_tests_root/build/all_reduce_perf_openmpi}
nranks=${NCCL_TEST_RANKS:-${SLURM_NTASKS:-8}}
run_dir="results/nccl-tests/${SLURM_JOB_NAME:-nccl_ompi_ucx_2n4g}-${SLURM_JOB_ID:-manual}"

source "$project_root/tooling/environments/leonardo.sh"

export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=none

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}
export OMPI_MCA_mpi_cuda_support=${OMPI_MCA_mpi_cuda_support:-1}
export OMPI_MCA_opal_warn_on_missing_libcuda=${OMPI_MCA_opal_warn_on_missing_libcuda:-0}
export OMPI_MCA_plm_slurm_args=${OMPI_MCA_plm_slurm_args:---external-launcher}

export UCX_TLS=${UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
export UCX_RNDV_SCHEME=${UCX_RNDV_SCHEME:-get_zcopy}
export UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-16384}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-warn}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,GRAPH}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NCCL_HOME/lib:$GCC12_LIB:${LD_LIBRARY_PATH:-}"

[ -x "$all_reduce_perf" ] || {
    echo "no executable: $all_reduce_perf" >&2
    echo "Build it with:" >&2
    echo "  tooling/jobs/leonardo/oneccl-nccl/compile-nccl-tests-openmpi.sh" >&2
    exit 1
}

mkdir -p "$run_dir"

echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
echo "nodes: ${SLURM_NODELIST:-manual}"
echo "node: $(hostname)"
echo "project_root: $project_root"
echo "nccl_tests_root: $nccl_tests_root"
echo "all_reduce_perf: $all_reduce_perf"
echo "run_dir: $run_dir"
echo "nranks: $nranks"
echo "OPENMPI_HOME: ${OPENMPI_HOME:-unset}"
echo "OMPI_MCA_pml: $OMPI_MCA_pml"
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
echo "OMPI_MCA_coll_hcoll_enable: $OMPI_MCA_coll_hcoll_enable"
echo "OMPI_MCA_coll_ucc_enable: $OMPI_MCA_coll_ucc_enable"
echo "OMPI_MCA_mpi_cuda_support: $OMPI_MCA_mpi_cuda_support"
echo "UCX_TLS: $UCX_TLS"
echo "UCX_RNDV_SCHEME: $UCX_RNDV_SCHEME"
echo "UCX_RNDV_THRESH: $UCX_RNDV_THRESH"
echo "UCX_LOG_LEVEL: $UCX_LOG_LEVEL"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi || true

echo "MPI diagnostics:"
which mpirun || true
mpirun --version || true
which mpicc || true
mpicc --showme:command || true
mpicc --showme:link || true

echo "rank placement:"
mpirun -np "$nranks" env | grep -E 'OMPI|PMIX|UCX|CUDA_VISIBLE_DEVICES|NCCL' | sort || true

echo "libraries used by all_reduce_perf:"
ldd "$all_reduce_perf" | grep -E 'libmpi|libucp|libucs|libuct|libucc|libnccl|libcuda|libcudart|libstdc\+\+' || true

stdout="$run_dir/all-reduce-perf-stdout.txt"
stderr="$run_dir/all-reduce-perf-stderr.txt"

echo "Running multi-node NCCL all_reduce_perf with OpenMPI/UCX"
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

echo "multi-node OpenMPI/UCX NCCL all_reduce_perf completed"
