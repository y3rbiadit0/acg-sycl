# Build oneCCL NCCL on Leonardo

This document records the build used for the `unisa-hpc/oneCCL` NCCL-enabled fork on Leonardo.

## Source And Install Layout

Expected source tree:

```bash
$HOME/oneCCL-nccl
```

Build tree on scratch:

```bash
$SCRATCH/oneccl-build
```

Install prefix:

```bash
$HOME/opt/oneccl-nccl-leonardo
```

## Required Source Fixes

Two source fixes were needed for this DPC++ CUDA backend setup.

First, `examples/include/sycl_base.hpp` must accept CUDA in `get_preferred_gpu_platform_name()`:

```cpp
else if (std::strstr(env, "cuda")) {
    filter = "cuda";
}
```

Without this, `ONEAPI_DEVICE_SELECTOR=cuda:*` is rejected and the benchmark fails with:

```text
No devices of requested type available
```

Second, `submit_barrier()` in `examples/include/sycl_base.hpp` must not use an empty `single_task` kernel for open-source DPC++.

Use:

```cpp
inline sycl::event submit_barrier(sycl::queue queue) {
#if defined(CCL_OPEN_SOURCE_DPCPP)
    return queue.ext_oneapi_submit_barrier();
#elif ICPX_VERSION >= 140000
    return queue.ext_oneapi_submit_barrier();
#elif ICPX_VERSION < 140000
    return queue.submit_barrier();
#endif // ICPX_VERSION
}
```

Without this, the SYCL benchmark can execute with `--check off`, but correctness validation fails with:

```text
No kernel named _ZTSZZ14submit_barrier... was found
```

## Why Bundled Intel MPI Is Used

The working build uses oneCCL's bundled Intel MPI path. Do not configure this build against Leonardo OpenMPI.

OpenMPI 4.1.6 failed because this fork references MPI large-count symbols such as `MPI_Irecv_c`, `MPI_Isend_c`, `MPI_Ireduce_c`, and `MPI_Reduce_c`.

OpenMPI 5.0.9 was also not viable in this environment because the same MPI API assumptions and PMIx/OFI source guards caused compile failures.

The bundled Intel MPI path provides the expected `libmpi.so.12` and produced the validated working runtime.

## Build Command

Run from the aCG-SYCL repository:

```bash
tooling/jobs/leonardo/oneccl-nccl/build-oneccl-nccl.sh
```

The script performs a clean configure/build/install in `$SCRATCH/oneccl-build`.

Useful overrides:

```bash
ONECCL_NCCL_SOURCE=$HOME/oneCCL-nccl
ONECCL_NCCL_BUILD_DIR=$SCRATCH/oneccl-build
ONECCL_NCCL_INSTALL=$HOME/opt/oneccl-nccl-leonardo
ONECCL_NCCL_BUILD_JOBS=8
```

## Manual Build Command

The build script expands to this CMake configuration:

```bash
source /leonardo/home/userexternal/fmerenda/aCG-SYCL-V3/tooling/environments/leonardo.sh

export DPCPP_ROOT="$DPCPP_INSTALL"
export ACG_HOST_FLAGS="--gcc-toolchain=${GCC12_ROOT}"
export ACG_SYCL_CXX_FLAGS="${ACG_HOST_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGET} -Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH}"
export ACG_LINK_FLAGS="${ACG_HOST_FLAGS} -L${GCC12_LIB} -Wl,-rpath,${GCC12_LIB}"

rm -rf "$SCRATCH/oneccl-build"
mkdir -p "$SCRATCH/oneccl-build"

cmake -S "$HOME/oneCCL-nccl" \
  -B "$SCRATCH/oneccl-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/oneccl-nccl-leonardo" \
  -DCMAKE_C_COMPILER="$DPCPP_CLANG" \
  -DCMAKE_CXX_COMPILER="$DPCPP_CLANGXX" \
  -DCMAKE_C_FLAGS="$ACG_HOST_FLAGS" \
  -DCMAKE_CXX_FLAGS="$ACG_SYCL_CXX_FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="$ACG_LINK_FLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$ACG_LINK_FLAGS" \
  -DCMAKE_MODULE_LINKER_FLAGS="$ACG_LINK_FLAGS" \
  -DCMAKE_REQUIRED_FLAGS="$ACG_HOST_FLAGS" \
  -DTHREADS_PREFER_PTHREAD_FLAG=ON \
  -DCOMPUTE_BACKEND=dpcpp \
  -DENABLE_MPI=ON \
  -DCCL_ENABLE_NCCL=ON \
  -DCCL_ENABLE_RCCL=OFF \
  -DCCL_ENABLE_ZE=OFF \
  -DENABLE_OFI_HMEM=OFF \
  -DBUILD_FT=OFF \
  -DBUILD_EXAMPLES=ON \
  -DCUDA_ROOT="$CUDA_HOME" \
  -DDPCPP_ROOT="$DPCPP_ROOT"

cmake --build "$SCRATCH/oneccl-build" -j 8 --verbose
cmake --install "$SCRATCH/oneccl-build"
```

## Post-Build Checks

The benchmark should resolve bundled Intel MPI and NCCL:

```bash
ldd "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" | grep -E 'libmpi|libccl|libsycl|libnccl'
```

Expected characteristics:

```text
libmpi.so.12 => .../oneCCL-nccl/deps/mpi/lib/libmpi.so.12
libccl.so.1 => .../opt/oneccl-nccl-leonardo/lib/libccl.so.1
libsycl.so.8 => .../dpcpp_6.3/llvm/build/install/lib/libsycl.so.8
libnccl.so.2 => .../nccl-2.22.3...
```
