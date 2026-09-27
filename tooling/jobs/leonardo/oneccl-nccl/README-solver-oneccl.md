# oneCCL Solver Collectives

This experiment keeps the SYCL solver architecture vendor-independent while matching the original CUDA aCG communication design.

The solver keeps MPI as the distributed runtime for process setup, rank discovery, partitioning, halo exchange, host reductions, diagnostics, and final accounting. oneCCL is optional and is used only for solver device collectives. With `CCL_BACKEND=nccl`, oneCCL routes those collectives through NCCL.

## Phase 1: Hot-Loop Device Scalar Allreduces

Use oneCCL for the repeated device-resident scalar allreduces in the CG loop:

```text
rho      = global dot(r, r)
gamma    = global dot(s, A*s)
rho_next = global dot(r, r)
```

The local dot products remain oneMath/SYCL GPU operations. The halo exchange remains MPI point-to-point. Host reductions remain MPI.

Runtime mode:

```bash
--solver-collectives oneccl
```

Expected oneCCL NCCL backend environment:

```bash
export CCL_BACKEND=nccl
export CCL_ATL_TRANSPORT=ofi
```

`ofi`, not `mpi`: the solver binary links the module's CUDA-aware OpenMPI, so the MPI ATL would have `libccl` dlopen oneCCL's bundled Intel MPI into a process that already has an MPI, which hangs in ATL setup. The KVS bootstrap in `init_oneccl()` is what OFI needs and is already there. See `tooling/jobs/leonardo/sycl/campaign-common.sh`.

The SYCL queue must be in-order. oneCCL's NCCL backend extracts a CUDA stream from it and rejects an out-of-order queue (`nccl_comm.cpp:369`), so `make_queue()` in `src/runtime/queue_factory.cpp` sets `sycl::property::queue::in_order` on every queue -- not just the oneCCL ones, so that an mpi run and a oneCCL run differ in the collective and nothing else.

The baseline remains:

```bash
--solver-collectives mpi
```

Build with oneCCL support enabled:

```bash
cmake -S . -B <build-dir> \
  -DACG_ENABLE_ONECCL=ON \
  -DoneCCL_ROOT=$HOME/opt/oneccl-nccl-leonardo
```

Compare correctness, iterations, final residual, `allreduce_s`, and total solve time.

## Phase 2: Device RHS Norm Reduction

Move the setup reduction for `rhs_norm` from host code to the same device scalar path:

```text
local dot(b, b) on GPU -> oneCCL allreduce -> read scalar
```

This more closely matches the original CUDA implementation, which computes `bnrm2sqr` on device and calls its communication abstraction for the allreduce.

## Phase 3: oneCCL Backend/Transport Comparison

Keep the solver code unchanged and compare oneCCL runtime backends/transports when available.

Examples:

```text
CCL_BACKEND=nccl
native oneCCL backends/transports where supported
future SHMEM-like support if oneCCL exposes it
```

The solver should continue to ask only for a device allreduce. It should not depend directly on NCCL, SHMEM, OFI, UCX, or vendor-specific APIs.

## Phase 4: Stream/Overlap Optimization

The first implementation prioritizes correctness and parity with the current solver. Later optimization can reduce synchronization by improving SYCL/oneCCL stream interoperability and by exploring overlap with independent solver work.
