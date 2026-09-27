# Run oneCCL NCCL on Leonardo

This document records the validated oneCCL and oneCCL NCCL-backend smoke tests on Leonardo.

## Validated Stack

The validated setup is:

```text
DPC++ CUDA backend
unisa-hpc/oneCCL with NCCL enabled
oneCCL bundled Intel MPI 2021.17
NCCL 2.22.3
single Leonardo node
2 MPI ranks / 2 NVIDIA A100 GPUs
```

Validated native-backend tests:

```text
oneCCL host allreduce with --check last
oneCCL SYCL/CUDA allreduce with --check last
```

Validated NCCL-backend tests with `CCL_BACKEND=nccl`:

```text
NCCL API allreduce example
NCCL API barrier example
```

The oneCCL benchmark executable is not used for the NCCL backend. `benchmark --backend sycl` creates a host-style communicator internally and fails under `CCL_BACKEND=nccl` with:

```text
host communicator is only supported for native backend
```

## Submit The Native Runner

From the aCG-SYCL repository:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/run-oneccl-nccl.sh
```

This validates the native oneCCL host and SYCL/CUDA benchmark paths.

## Submit The NCCL-Backend Runner

From the aCG-SYCL repository:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/run-oneccl-nccl-backend.sh
```

This validates the `CCL_BACKEND=nccl` examples path and enables:

```bash
export CCL_BACKEND=nccl
export CCL_LOG_LEVEL=info
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH
```

## Build And Run NCCL-Backend Perf

Build the timed allreduce benchmark from the aCG-SYCL repository on Leonardo:

```bash
tooling/jobs/leonardo/oneccl-nccl/compile-nccl-backend-perf.sh
```

This creates:

```text
$HOME/opt/oneccl-nccl-leonardo/examples/nccl/nccl_backend_allreduce_perf
```

Submit the perf runner:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/run-nccl-backend-perf.sh
```

Submit the 2-node / 4-GPU-per-node perf runner:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/run-nccl-backend-perf-2n4g.sh
```

The perf runner writes results to:

```text
results/oneccl-nccl-perf/oneccl_nccl_perf-<jobid>/nccl-backend-allreduce-perf-stdout.txt
```

Expected output format:

```text
# oneCCL NCCL backend allreduce perf
# count bytes avg_us algbw_GBps busbw_GBps check
1024 4096 ... ... ... ok
```

Useful overrides:

```bash
ONECCL_TEST_ELEM_COUNTS=1024,4096,16384,65536,262144,1048576,4194304,16777216
ONECCL_TEST_ITERS=50
ONECCL_TEST_WARMUP_ITERS=10
ONECCL_TEST_CHECK=last
NCCL_DEBUG=INFO
```

Use `NCCL_DEBUG=INFO` for backend-dispatch proof and the default `NCCL_DEBUG=WARN` for cleaner timing output.

## Multi-Node Raw NCCL Baseline

Detailed validated multi-node notes and results are recorded in:

```text
tooling/jobs/leonardo/oneccl-nccl/README-multinode.md
```

Run the matching 2-node / 4-GPU-per-node raw NCCL baseline:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/nccl-test-2n4g.sh
```

Both multi-node runners default to:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=tcp
export FI_PROVIDER=tcp
export FI_PROVIDER_PATH="$HOME/opt/oneccl-nccl-leonardo/opt/mpi/libfabric/lib/prov-tcp-only"
export FI_LOG_LEVEL=error
export NCCL_SOCKET_IFNAME=ib0
```

Use diagnostic overrides if the first multi-node run fails during MPI or NCCL initialization:

```bash
I_MPI_DEBUG=10 NCCL_DEBUG=INFO FI_LOG_LEVEL=info \
  sbatch tooling/jobs/leonardo/oneccl-nccl/run-nccl-backend-perf-2n4g.sh
```

The runner writes Slurm logs to:

```text
logs/test_oneccl_nccl-<jobid>-stdout.txt
logs/test_oneccl_nccl-<jobid>-stderr.txt
```

Per-test logs are written to:

```text
results/oneccl-nccl/test_oneccl_nccl-<jobid>/
```

Expected native-runner per-test files:

```text
oneccl-host-allreduce-stdout.txt
oneccl-host-allreduce-stderr.txt
oneccl-sycl-allreduce-stdout.txt
oneccl-sycl-allreduce-stderr.txt
nccl-api-allreduce-stdout.txt
nccl-api-allreduce-stderr.txt
nccl-api-barrier-stdout.txt
nccl-api-barrier-stderr.txt
```

Expected NCCL-backend-runner per-test files:

```text
nccl-api-allreduce-stdout.txt
nccl-api-allreduce-stderr.txt
nccl-api-barrier-stdout.txt
nccl-api-barrier-stderr.txt
```

## Expected Success Output

The native runner main Slurm stdout should contain:

```text
oneccl-host-allreduce completed
oneccl-sycl-allreduce completed
nccl-api-allreduce completed
nccl-api-barrier completed
oneCCL NCCL smoke tests completed
```

The NCCL-backend runner main Slurm stdout should contain:

```text
nccl-api-allreduce completed
nccl-api-barrier completed
oneCCL NCCL backend examples completed
```

The host benchmark output should contain:

```text
backend:         host
check:           last
# All done
```

The SYCL benchmark output should contain:

```text
backend:         sycl
check:           last
preferred platform: NVIDIA CUDA BACKEND
# All done
```

The NCCL API examples should print:

```text
PASSED
```

With `NCCL_DEBUG=INFO`, the NCCL-backend stderr should also contain lines like:

```text
CCL_BACKEND changed to be nccl
NCCL INFO ncclCommInitRank ... Init COMPLETE
NCCL COMM: communicator initialized successfully
NCCL INFO AllReduce: opCount ...
```

## Runtime Environment

The runner uses bundled Intel MPI by sourcing:

```bash
source "$HOME/opt/oneccl-nccl-leonardo/env/vars.sh"
```

Do not use this for the validated path:

```bash
source "$HOME/opt/oneccl-nccl-leonardo/env/vars.sh" --ccl-bundled-mpi=no
```

Key runtime variables:

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:*
export SYCL_DEVICE_FILTER=cuda
export CUDA_VISIBLE_DEVICES=0,1

export CCL_ATL_TRANSPORT=mpi
export CCL_MPI_LIBRARY_PATH="$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12"
export CCL_WORKER_COUNT=1
export CCL_WORKER_AFFINITY=auto

export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm
```

`I_MPI_FABRICS=shm` is used because the validated smoke test is single-node. Without it, bundled Intel MPI failed during OFI initialization on Leonardo.

## Manual Interactive Run

Request a node:

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

Set up the runtime:

```bash
cd /leonardo/home/userexternal/fmerenda/aCG-SYCL-V3

source tooling/environments/leonardo.sh

set +u
source "$HOME/opt/oneccl-nccl-leonardo/env/vars.sh"
set -u

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

Run diagnostics:

```bash
which mpirun
mpirun --version
sycl-ls
mpirun -np 2 sycl-ls
ldd "$HOME/opt/oneccl-nccl-leonardo/examples/benchmark/benchmark" | grep -E 'libmpi|libccl|libsycl|libnccl'
```

Run host allreduce:

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

Run SYCL/CUDA allreduce:

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
  --check last
```

Run NCCL API examples:

```bash
export CCL_BACKEND=nccl
export CCL_LOG_LEVEL=info
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH

mpirun -np 2 "$HOME/opt/oneccl-nccl-leonardo/examples/nccl/nccl_allreduce_test"
mpirun -np 2 "$HOME/opt/oneccl-nccl-leonardo/examples/nccl/nccl_barrier_test"
```

## Notes

The GPU rank wrapper maps Intel MPI local ranks to GPUs using `MPI_LOCALRANKID`, so each oneCCL example rank sees one GPU through `CUDA_VISIBLE_DEVICES`.

Do not use the GPU rank wrapper for raw NVIDIA `nccl-tests`. `all_reduce_perf -g 1` handles local GPU selection itself, and wrapping it can make each rank see only one GPU and fail with an invalid GPU count.

NCCL API examples may print warnings about external MPI initialization or MPI finalization order. The smoke-test criterion is that the examples print `PASSED` and exit successfully.

The existing oneCCL benchmark output is a native oneCCL SYCL-vs-host comparison. It is not a oneCCL NCCL-backend-vs-raw-NCCL performance comparison. A real NCCL-backend performance comparison needs a timed benchmark around the `examples/nccl` path or another program that constructs the GPU communicator under `CCL_BACKEND=nccl`.

`nccl_backend_allreduce_perf` is that timed benchmark. It reuses the same GPU communicator creation pattern as `examples/nccl/nccl_allreduce_test`, runs `ccl::allreduce`, and therefore exercises the `CCL_BACKEND=nccl` dispatch path.

OpenMPI is not the validated runtime for this fork on Leonardo. It requires source compatibility work around MPI large-count APIs and PMIx/OFI build guards.
