#!/usr/bin/env bash

# Modules
module purge
module load gcc/12.2.0
module load cmake/4.1.2
module load python
module load ninja
module load curl
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Toolchain roots
export DPCPP_HOME="$HOME/opt/dpcpp_6.3"
export DPCPP_INSTALL="$DPCPP_HOME/llvm/build/install"
export DPCPP_CLANG="$DPCPP_INSTALL/bin/clang"
export DPCPP_CLANGXX="$DPCPP_INSTALL/bin/clang++"

export CC="$DPCPP_CLANG"
export CXX="$DPCPP_CLANGXX"

# oneMath
export ACG_ONEMATH_ROOT="$HOME/opt/oneMath"

# Extra dependencies
export METIS_DIR="$HOME/thesis/local_gcc"
export METIS_LIB_DIR="${METIS_DIR}/lib64"
export METIS_LIBRARIES="${METIS_LIB_DIR}/libmetis.a;${METIS_LIB_DIR}/libGKlib.a"

# GCC runtime libs
export GCC12_LIB=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/gcc-12.2.0-lkcazt4letxjj4s7nlhzryoyivevsatz/lib64

# CUDA / SYCL
export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export SYCL_TARGET="nvptx64-nvidia-cuda"
export NVIDIA_GPU_ARCH="sm_80"
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export SYCL_DEVICE_FILTER="cuda"

export SYCL_FLAGS="-fsycl -fsycl-targets=${SYCL_TARGET} -Xsycl-target-backend --cuda-gpu-arch=${NVIDIA_GPU_ARCH} -L${GCC12_LIB} -Wl,-rpath,${GCC12_LIB}"

# Search paths
export PATH="$DPCPP_INSTALL/bin:$PATH"
export PATH="$PATH:$HOME/local/hwloc/bin"
export LD_LIBRARY_PATH="$DPCPP_INSTALL/lib:${ACG_ONEMATH_ROOT}/lib:${METIS_LIB_DIR}:${GCC12_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC12_LIB:${LIBRARY_PATH:-}"

# OpenMP
export OpenMP_C_FLAGS="-fopenmp"
export OpenMP_CXX_FLAGS="-fopenmp"

# Project debug vars
export ACG_SYCL_DEBUG_ITERS=5
export ACG_SYCL_MAX_ITERATIONS=20
export ACG_SYCL_WARMUP=0
