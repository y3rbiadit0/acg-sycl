# Multi-Node oneCCL NCCL Backend on Leonardo

This note records the validated 2-node / 8-GPU workflow for `unisa-hpc/oneCCL` with `CCL_BACKEND=nccl` on Leonardo.

## Validated Setup

```text
Nodes: 2 Leonardo boost nodes
GPUs: 4 NVIDIA A100-SXM-64GB per node, 8 GPUs total
Ranks: 8 MPI ranks, 1 rank per GPU
oneCCL: unisa-hpc/oneCCL with NCCL enabled
MPI: oneCCL bundled Intel MPI 2021.17
NCCL: 2.22.3+cuda12.2
SYCL: DPC++ CUDA backend
```

## Important Runtime Detail

Bundled Intel MPI needs OFI for multi-node startup. The working setup uses Intel MPI's bundled libfabric and forces the TCP OFI provider:

```bash
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=tcp
export FI_PROVIDER=tcp
export FI_PROVIDER_PATH="$HOME/opt/oneccl-nccl-leonardo/opt/mpi/libfabric/lib/prov-tcp-only"
export FI_LOG_LEVEL=error
export LD_LIBRARY_PATH="$HOME/opt/oneccl-nccl-leonardo/opt/mpi/libfabric/lib:${LD_LIBRARY_PATH:-}"
```

The `prov-tcp-only` directory avoids noisy libfabric probing of providers such as `psmx2` that require unavailable libraries. The scripts create the TCP-only provider symlink automatically if needed.

MPI is used for process startup and oneCCL KVS/bootstrap. The allreduce payload is handled by NCCL after `CCL_BACKEND=nccl` constructs the GPU communicator.

## Build The oneCCL NCCL Backend Perf Binary

```bash
tooling/jobs/leonardo/oneccl-nccl/compile-nccl-backend-perf.sh
```

This creates:

```text
$HOME/opt/oneccl-nccl-leonardo/examples/nccl/nccl_backend_allreduce_perf
```

## Run oneCCL NCCL Backend, 2 Nodes / 8 GPUs

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/run-nccl-backend-perf-2n4g.sh
```

Results are written to:

```text
results/oneccl-nccl-perf/oneccl_nccl_perf_2n4g-<jobid>/nccl-backend-allreduce-perf-stdout.txt
```

Validated clean result from job `42346572`:

```text
# ranks 8 iters 50 warmup_iters 10 check last
# count bytes avg_us algbw_GBps busbw_GBps check
1024 4096 32.0419 0.127833 0.223707 ok
4096 16384 33.9514 0.482572 0.844501 ok
16384 65536 39.9601 1.64004 2.87007 ok
65536 262144 56.8344 4.61242 8.07174 ok
262144 1048576 127.631 8.21567 14.3774 ok
1048576 4194304 263.57 15.9134 27.8485 ok
4194304 16777216 732.452 22.9055 40.0847 ok
16777216 67108864 1878.41 35.7265 62.5214 ok
```

## Run Raw NCCL Baseline, 2 Nodes / 8 GPUs

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/nccl-test-2n4g.sh
```

Results are written to:

```text
results/nccl-tests/nccl_test_2n4g-<jobid>/all-reduce-perf-stdout.txt
```

Validated raw NCCL result from job `42346589`:

```text
size(B)   time_us(out-of-place)   algbw_GBps   busbw_GBps   wrong
4096      25.02                   0.16         0.29         0
16384     26.46                   0.62         1.08         0
65536     33.06                   1.98         3.47         0
262144    48.12                   5.45         9.53         0
1048576   121.22                  8.65         15.14        0
4194304   255.72                  16.40        28.70        0
16777216  724.03                  23.17        40.55        0
67108864  1692.25                 39.66        69.40        0
```

## Run Raw NCCL With OpenMPI/UCX

Build a separate `nccl-tests` binary linked against Leonardo OpenMPI instead of oneCCL bundled Intel MPI:

```bash
tooling/jobs/leonardo/oneccl-nccl/compile-nccl-tests-openmpi.sh
```

This creates:

```text
$HOME/nccl-tests/build/all_reduce_perf_openmpi
```

Run the 2-node / 4-GPU-per-node OpenMPI/UCX baseline:

```bash
sbatch tooling/jobs/leonardo/oneccl-nccl/nccl-test-openmpi-2n4g.sh
```

The runner intentionally does not source oneCCL `env/vars.sh`, because that would put bundled Intel MPI first. It uses the OpenMPI module from `tooling/environments/leonardo.sh` and the UCX/HPC-X libraries that Leonardo's OpenMPI module exposes.

Default OpenMPI/UCX settings:

```bash
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_mpi_cuda_support=1
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_plm_slurm_args=--external-launcher

export UCX_TLS=sm,cuda_copy,cuda_ipc,rc,self
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=16384
export UCX_LOG_LEVEL=warn
```

This tests raw NCCL with OpenMPI/UCX as the MPI launch/bootstrap layer. NCCL still owns the collective data path.

## Comparison

```text
Size       oneCCL NCCL backend   raw NCCL out-of-place   delta
4 KiB      32.04 us              25.02 us                +28.1%
16 KiB     33.95 us              26.46 us                +28.3%
64 KiB     39.96 us              33.06 us                +20.9%
256 KiB    56.83 us              48.12 us                +18.1%
1 MiB      127.63 us             121.22 us               +5.3%
4 MiB      263.57 us             255.72 us               +3.1%
16 MiB     732.45 us             724.03 us               +1.2%
64 MiB     1878.41 us            1692.25 us              +11.0%
```

The oneCCL NCCL backend is correct for all tested sizes and is close to raw NCCL for medium/large messages. Small messages show extra oneCCL wrapper/stream/dispatch overhead.

## GPUDirect/RDMA Interpretation

The MPI layer in this validated setup uses OFI/TCP only for startup/bootstrap. It is not the GPU data path.

The allreduce data path is NCCL. Earlier debug logs showed NCCL using IB devices, for example:

```text
NCCL INFO NET/IB : Using mlx5_.../IB
NCCL INFO Using network IB
NCCL INFO AllReduce ... [nranks=8]
```

That strongly indicates NCCL is using its IB transport for inter-node GPU collectives and likely GPUDirect RDMA where supported by the node topology and driver stack. To prove GPUDirect RDMA specifically, run once with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL \
  sbatch tooling/jobs/leonardo/oneccl-nccl/run-nccl-backend-perf-2n4g.sh
```

Then inspect for `NET/IB`, `GDR`, `GPU Direct`, or compare performance with:

```bash
NCCL_NET_GDR_LEVEL=0
```

versus the default/forced higher GDR level.

## Troubleshooting

If MPI fails before NCCL starts with `libfabric.so.1` missing, ensure the Intel MPI bundled libfabric directory is first in `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$HOME/opt/oneccl-nccl-leonardo/opt/mpi/libfabric/lib:${LD_LIBRARY_PATH:-}"
```

If MPI fails with `fi_getinfo() failed`, force TCP provider and the TCP-only provider path:

```bash
export FI_PROVIDER=tcp
export I_MPI_OFI_PROVIDER=tcp
export FI_PROVIDER_PATH="$HOME/opt/oneccl-nccl-leonardo/opt/mpi/libfabric/lib/prov-tcp-only"
```

For diagnostics only, use:

```bash
I_MPI_DEBUG=10 FI_LOG_LEVEL=info NCCL_DEBUG=INFO
```

For clean timing, use:

```bash
I_MPI_DEBUG=0 FI_LOG_LEVEL=error NCCL_DEBUG=WARN CCL_LOG_LEVEL=warn
```
