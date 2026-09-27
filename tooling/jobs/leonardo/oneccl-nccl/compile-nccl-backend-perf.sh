#!/bin/bash -l

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-$(pwd)}
oneccl_root=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}
source_file=${ONECCL_NCCL_PERF_SOURCE:-$project_root/tooling/jobs/leonardo/oneccl-nccl/nccl-backend-allreduce-perf.cpp}
output=${ONECCL_NCCL_PERF_BINARY:-$oneccl_root/examples/nccl/nccl_backend_allreduce_perf}
mpi_lib_path=${ONECCL_MPI_LIBRARY_PATH:-$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12}
mpi_lib_dir=$(dirname "$mpi_lib_path")

source "$project_root/tooling/environments/leonardo.sh"

set +u
source "$oneccl_root/env/vars.sh"
set -u

mkdir -p "$(dirname "$output")"

mpi_include="$oneccl_root/opt/mpi/include"
mpi_lib_release="$oneccl_root/opt/mpi/lib/release"
mpi_lib="$oneccl_root/opt/mpi/lib"

"$DPCPP_CLANGXX" \
    -std=c++17 \
    -O3 \
    --gcc-toolchain="$GCC12_ROOT" \
    -fsycl \
    -fsycl-targets="$SYCL_TARGET" \
    -Xsycl-target-backend="$SYCL_TARGET" \
    --cuda-gpu-arch="$NVIDIA_GPU_ARCH" \
    -I"$oneccl_root/include" \
    -I"$mpi_include" \
    "$source_file" \
    -L"$oneccl_root/lib" \
    -L"$mpi_lib_release" \
    -L"$mpi_lib" \
    -L"$mpi_lib_dir" \
    -L"$GCC12_LIB" \
    -Wl,-rpath,"$oneccl_root/lib" \
    -Wl,-rpath,"$mpi_lib_release" \
    -Wl,-rpath,"$mpi_lib" \
    -Wl,-rpath,"$mpi_lib_dir" \
    -Wl,-rpath,"$GCC12_LIB" \
    -lccl \
    -lmpi \
    -ldl \
    -lpthread \
    -o "$output"

echo "built: $output"
ldd "$output" | grep -E 'libstdc\+\+|libsycl|libccl|libnccl|libmpi|libcuda|libcudart' || true
