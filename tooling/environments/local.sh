#!/usr/bin/env bash

source /opt/intel/oneapi/setvars.sh --force >/dev/null

export DPCPP_INSTALL="/opt/dpcpp/llvm/build"
export DPCPP_CLANG="$DPCPP_INSTALL/bin/clang"
export DPCPP_CLANGXX="$DPCPP_INSTALL/bin/clang++"

export CC="$DPCPP_CLANG"
export CXX="$DPCPP_CLANGXX"

export ACG_ONEMATH_ROOT="/opt/oneMath/install"

export CUDA_ROOT="/usr/local/cuda-12.2"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"

export SYCL_TARGET="nvptx64-nvidia-cuda"
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export SYCL_DEVICE_FILTER="cuda"

export PATH="$DPCPP_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$DPCPP_INSTALL/lib:${ACG_ONEMATH_ROOT}/lib:${LD_LIBRARY_PATH:-}"
