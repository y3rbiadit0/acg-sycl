#!/usr/bin/env bash

# Modules
module purge
module load gcc/12.2.0
module load cmake/4.1.2
module load python
module load ninja
module load curl
module load cuda/12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

# MPI stack. `hpcx` (the default) matches cluster/leonardo/env/modules.sh in
# the native aCG repo, which loads `nvhpc/24.5 hpcx-mpi/2.19`, so the halo
# exchange and the MPI allreduces run on the same stack as native aCG -- a
# comparison between the two must not also be a comparison between MPIs.
# `openmpi` (the CUDA build of Open MPI 4.1.6) remains selectable.
#
# Only the MPI changes. The SYCL compiler stays DPC++ because nvc++ cannot
# compile SYCL, and CUDA stays 12.2, which is what DPC++, NCCL and oneMath here
# were all built against.
ACG_MPI=${ACG_MPI:-hpcx}
case "$ACG_MPI" in
  openmpi)
    module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
    ;;
  hpcx)
    # hpcx-mpi lives under the nvhpc compiler tree on Leonardo, which is why the
    # native repo loads nvhpc first. nvhpc carries its own CUDA 12.4, so
    # CUDA_HOME is captured and put back afterwards rather than left pointing
    # into nvhpc and silently mixing toolkits.
    _acg_cuda_home=${CUDA_HOME:-}
    module load nvhpc/24.5
    module load hpcx-mpi/2.19
    if [ -n "$_acg_cuda_home" ]; then export CUDA_HOME="$_acg_cuda_home"; fi
    unset _acg_cuda_home
    ;;
  *)
    echo "unknown ACG_MPI='$ACG_MPI' (expected openmpi or hpcx)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# Name the MPI explicitly rather than letting find_package(MPI) take whatever
# mpicxx happens to be first on PATH: with two MPI stacks reachable, a silent
# pick is how a build ends up linking one and running under another.
export MPI_CXX_COMPILER=${MPI_CXX_COMPILER:-$(command -v mpicxx 2>/dev/null || true)}
export ACG_MPI

# Toolchain roots
export DPCPP_HOME="$HOME/opt/dpcpp_6.3"
export DPCPP_INSTALL="$DPCPP_HOME/llvm/build/install"
export DPCPP_CLANG="$DPCPP_INSTALL/bin/clang"
export DPCPP_CLANGXX="$DPCPP_INSTALL/bin/clang++"

export CC="$DPCPP_CLANG"
export CXX="$DPCPP_CLANGXX"

# oneMath
export ACG_ONEMATH_ROOT="$HOME/opt/oneMath"
# oneCCL built with its NCCL backend, for --solver-collectives oneccl.
export ONECCL_ROOT="${ONECCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}"

# Extra dependencies
export METIS_HOME="${METIS_HOME:-$HOME/opt/metis_gcc}"
export METIS_DIR="$METIS_HOME"
export METIS_LIB_DIR="${METIS_DIR}/lib64"
export METIS_LIBRARIES="${METIS_LIB_DIR}/libmetis.a;${METIS_LIB_DIR}/libGKlib.a"

# GCC toolchain and runtime libs
export GCC12_ROOT="${GCC_HOME:?gcc/12.2.0 module must define GCC_HOME}"
export GCC12_LIB="${GCC12_ROOT}/lib64"

# CUDA / SYCL
export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export SYCL_TARGET="nvptx64-nvidia-cuda"
export NVIDIA_GPU_ARCH="sm_80"
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export SYCL_DEVICE_FILTER="cuda"

export ACG_EXTRA_COMPILE_FLAGS="--gcc-toolchain=${GCC12_ROOT} -Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH}"
export ACG_EXTRA_LINK_FLAGS="--gcc-toolchain=${GCC12_ROOT} -Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH} -L${GCC12_LIB} -Wl,-rpath,${GCC12_LIB}"
export SYCL_FLAGS="${ACG_EXTRA_COMPILE_FLAGS}"

# Search paths
export PATH="$DPCPP_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$DPCPP_INSTALL/lib:${ACG_ONEMATH_ROOT}/lib:${METIS_LIB_DIR}:${GCC12_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC12_LIB:${LIBRARY_PATH:-}"

# OpenMP
export OpenMP_C_FLAGS="-fopenmp"
export OpenMP_CXX_FLAGS="-fopenmp"

# Project debug vars
export ACG_SYCL_DEBUG_ITERS=5
export ACG_SYCL_MAX_ITERATIONS=20
export ACG_SYCL_WARMUP=0


# hwloc - export HWLOC_ROOT="$HOME/local/hwloc"
export HWLOC_ROOT=$HOME/opt/hwloc
export PKG_CONFIG_PATH=$HWLOC_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HWLOC_ROOT/lib:$LD_LIBRARY_PATH
export PATH=$HWLOC_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/hwloc/lib:$LD_LIBRARY_PATH
