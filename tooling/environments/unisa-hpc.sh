#!/usr/bin/env bash
module purge
module load gcc/14.1.0
module load cmake/4.1.1
module load cuda12.8/toolkit/12.8.0
module load shared cuda12.8/hpc_sdk/nvhpc-hpcx-cuda12/25.3
module load slurm/slurm/23.11.10

# Ninja
export PATH=$HOME/opt/ninja/build:$PATH

# DPC++ (AMD build)
export DPCPP_INSTALL="$HOME/opt/amd/dpcpp/llvm/build"
export DPCPP_CLANG="$DPCPP_INSTALL/bin/clang"
export DPCPP_CLANGXX="$DPCPP_INSTALL/bin/clang++"
export CC="$DPCPP_CLANG"
export CXX="$DPCPP_CLANGXX"
export PATH="$DPCPP_INSTALL/bin:$PATH"

# CUDA
export CUDA_ROOT="/cm/shared/apps/cuda12.8/toolkit/12.8.0"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"

# SYCL targeting A100 (sm_80)
export SYCL_TARGET="nvptx64-nvidia-cuda"
export NVIDIA_GPU_ARCH="sm_80"
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export SYCL_DEVICE_FILTER="cuda"

export ACG_EXTRA_COMPILE_FLAGS="-march=znver3 -Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH}"
export ACG_EXTRA_LINK_FLAGS="-Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH}"

# oneMath
export ACG_ONEMATH_ROOT="$HOME/opt/amd/oneMath/install"

# MPI: 4 ranks/node bound to NUMA (2x EPYC 7763, 4x A100)
export ACG_MPIRUN_ARGS="--map-by ppr:4:node --bind-to numa"

export LD_LIBRARY_PATH="$DPCPP_INSTALL/lib:$ACG_ONEMATH_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_ROOT/lib64:${LIBRARY_PATH:-}"