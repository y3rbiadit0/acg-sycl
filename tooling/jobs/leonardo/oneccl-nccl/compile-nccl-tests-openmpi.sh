#!/bin/bash -l

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-$(pwd)}
nccl_tests_root=${NCCL_TESTS_ROOT:-$HOME/nccl-tests}

source "$project_root/tooling/environments/leonardo.sh"

[ -n "${OPENMPI_HOME:-}" ] || { echo "OPENMPI_HOME is not set" >&2; exit 1; }
[ -d "$nccl_tests_root" ] || { echo "no such directory: $nccl_tests_root" >&2; exit 1; }

echo "OPENMPI_HOME: $OPENMPI_HOME"
echo "NCCL_HOME: $NCCL_HOME"
echo "CUDA_HOME: $CUDA_HOME"
echo "nccl_tests_root: $nccl_tests_root"

which mpirun || true
mpirun --version || true
which mpicc || true
mpicc --showme:command || true
mpicc --showme:link || true

make -C "$nccl_tests_root" -j \
    MPI=1 \
    NAME_SUFFIX=_openmpi \
    MPI_HOME="$OPENMPI_HOME" \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    CXX="$GCC12_ROOT/bin/g++" \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80" \
    NVCCFLAGS="-ccbin $GCC12_ROOT/bin/g++"

binary="$nccl_tests_root/build/all_reduce_perf_openmpi"
[ -x "$binary" ] || { echo "build did not create executable: $binary" >&2; exit 1; }

echo "built: $binary"
ldd "$binary" | grep -E 'libmpi|libucp|libucs|libuct|libucc|libnccl|libcuda|libcudart|libstdc\+\+' || true
