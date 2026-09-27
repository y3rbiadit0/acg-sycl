# oneCCL NCCL on Leonardo

This note records the working configuration used to run the `unisa-hpc/oneCCL` NCCL/SYCL implementation on Leonardo with NVIDIA A100 GPUs and the local DPC++ compiler.

## Current Working Path

The working runtime uses the oneCCL bundled Intel MPI, not Leonardo OpenMPI.

Source the Leonardo environment, then source oneCCL without `--ccl-bundled-mpi=no`:

```bash
source tooling/environments/leonardo.sh

set +u
source "$HOME/opt/oneccl-nccl-leonardo/env/vars.sh"
set -u
```

Required runtime variables:

```bash
export LC_ALL=C
export OMP_NUM_THREADS=8
export SLURM_CPU_BIND=none

export ONEAPI_DEVICE_SELECTOR=cuda:*
export SYCL_DEVICE_FILTER=cuda
export CUDA_VISIBLE_DEVICES=0,1

export CCL_ATL_TRANSPORT=mpi
export CCL_MPI_LIBRARY_PATH="$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12"
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY=auto

export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm

export LD_LIBRARY_PATH="$HOME/oneCCL-nccl/deps/mpi/lib:$GCC12_LIB:$DPCPP_INSTALL/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

The host backend smoke test works:

```bash
mpirun -np 2 \
  "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" \
  --backend host \
  --coll allreduce \
  --dtype float32 \
  --elem_counts 1024 \
  --iters 5 \
  --warmup_iters 1 \
  --check last
```

The SYCL/CUDA backend smoke test works with correctness checking disabled:

```bash
mpirun -np 2 \
  "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" \
  --backend sycl \
  --sycl_dev_type gpu \
  --sycl_root_dev 0 \
  --sycl_mem_type usm \
  --sycl_usm_type device \
  --coll allreduce \
  --dtype float32 \
  --elem_counts 1024 \
  --iters 5 \
  --warmup_iters 1 \
  --check off
```

A successful SYCL run prints:

```text
preferred platform: NVIDIA CUDA BACKEND, found: 2 GPU device(s)
# All done
```

## Interactive Run

Use a compute node, not the login node:

```bash
salloc -A IscrC_HIGRAPH_0 \
  -p boost_usr_prod \
  --nodes=1 \
  --ntasks-per-node=2 \
  --gres=gpu:2 \
  --cpus-per-task=8 \
  --time=00:20:00

srun --pty bash -l
```

Then run the environment setup and benchmark commands above.

## Why Bundled Intel MPI Is Used

The goal was initially to use Leonardo OpenMPI with `source env/vars.sh --ccl-bundled-mpi=no`. That path exposed multiple incompatibilities in this `unisa-hpc/oneCCL` fork.

With OpenMPI 4.1.6, enabling oneCCL MPI support failed to compile because the fork references MPI large-count symbols such as:

```text
MPI_Irecv_c
MPI_Isend_c
MPI_Ireduce_c
MPI_Reduce_c
MPI_Ialltoall_c
```

Those symbols are not available in Leonardo OpenMPI 4.1.6 headers.

Leonardo also provides OpenMPI 5.0.9, but the fork still failed to compile against it in this environment. The same `MPI_*_c` symbols were not exposed as expected, and the OFI/PMIx source path also failed with PMIx-related compile errors such as:

```text
unknown type name 'pmix_value_t'
unknown type name 'pmix_data_type_t'
use of undeclared identifier 'PMIX_BYTE_OBJECT'
```

Using the bundled Intel MPI avoids those OpenMPI header/API compatibility problems. The bundled MPI provides the expected `libmpi.so.12`, and the installed benchmark resolves it when `vars.sh` is sourced without `--ccl-bundled-mpi=no` and the bundled MPI library path is present in `LD_LIBRARY_PATH`.

Expected MPI checks for the working setup:

```bash
which mpirun
mpirun --version
ldd "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" | grep libmpi
```

Expected output characteristics:

```text
.../opt/oneccl-nccl-leonardo/opt/mpi/bin/mpirun
Intel(R) MPI Library for Linux* OS, Version 2021.17
libmpi.so.12 => .../oneCCL-nccl/deps/mpi/lib/libmpi.so.12
```

## Why `I_MPI_FABRICS=shm` Is Used

Without forcing shared-memory fabric, Intel MPI failed during OFI initialization on a single Leonardo node:

```text
Fatal error in internal_Init
MPIDI_OFI_mpi_init_hook
Other MPI error
```

For the current smoke test, all ranks are on one node, so shared-memory MPI is sufficient and avoids the OFI provider issue:

```bash
export I_MPI_FABRICS=shm
```

For multi-node runs, this will need to be revisited and a working Intel MPI fabric/provider configuration must be identified.

## Why `--check off` Is Used For SYCL

The SYCL benchmark runs successfully with `--check off`, but `--check last` currently fails in the correctness-check path with:

```text
No kernel named _ZTSZZ14submit_barrier... was found
```

This comes from the benchmark helper `examples/include/sycl_base.hpp`, specifically the open-source DPC++ fallback implementation of `submit_barrier()`. The benchmark can execute the allreduce, but the correctness-check path triggers this missing-kernel issue.

Until that helper is patched to use `queue.ext_oneapi_submit_barrier()` for this DPC++ build, use:

```bash
--check off
```

## CUDA Platform Patch

The benchmark originally rejected `ONEAPI_DEVICE_SELECTOR=cuda:*` because `examples/include/sycl_base.hpp` only accepted `level_zero` and `opencl` in `get_preferred_gpu_platform_name()`.

The required fix is to accept CUDA:

```cpp
else if (std::strstr(env, "cuda")) {
    filter = "cuda";
}
```

After this patch, the benchmark correctly finds the CUDA platform:

```text
preferred platform: NVIDIA CUDA BACKEND, found: 2 GPU device(s)
```

## Useful Diagnostics

Check SYCL visibility outside and inside MPI:

```bash
sycl-ls
mpirun -np 2 sycl-ls
```

Both should list CUDA GPUs:

```text
[cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM-64GB 8.0 [CUDA 12.2]
```

Check runtime linkage:

```bash
ldd "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" | grep -E 'libmpi|libccl|libsycl|libnccl'
```

## OpenMPI Status

OpenMPI is not the current working runtime for this fork on Leonardo. The known blockers are source compatibility with MPI `_c` APIs and PMIx/OFI build guards. To make OpenMPI viable, the fork likely needs patches to:

- Guard `MPI_*_c` declarations and dynamic symbol lookups by actual API availability.
- Fall back to non-`_c` MPI APIs when unavailable.
- Correctly disable or guard PMIx-dependent OFI code when PMIx is disabled.
- Avoid compiling OFI sources when only MPI ATL is desired.
