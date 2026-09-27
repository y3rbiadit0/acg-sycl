# aCG-SYCL

aCG-SYCL is a SYCL implementation of the multi-GPU conjugate gradient solver
from [aCG](https://github.com/ParCoreLab/aCG) (Trotter et al., SC '25). It solves
sparse symmetric positive-definite systems distributed by rows over MPI ranks,
one GPU per rank, and exists to answer one question: **what does it cost to
write this solver in portable SYCL instead of CUDA?**

It is the SYCL side of a CUDA-vs-SYCL comparison. The native reference is the
CUDA aCG in [`aCG-oshmpi`](https://github.com/y3rbiadit0/aCG-oshmpi) (a fork of
aCG), and the benchmark harness here runs both **in the same allocation**, on
the same matrices, partitions, MPI and transport settings, so the comparison
measures the implementation and nothing else.

The solver targets NVIDIA GPUs through DPC++ and oneMath (cuSPARSE/cuBLAS
underneath) and is validated on the Leonardo supercomputer (A100, 4 per node).

## 📊 Results at a glance

Paired against native aCG in the same allocations (SuiteSparse Bump_2911 and
Queen_4147, 3 allocations per cell). Ratio = SYCL time / native time, so below
1 means SYCL is faster.

| Scale | Collectives | SYCL / native | Note |
| --- | --- | --- | --- |
| 1 node, 1–4 GPUs | MPI and NCCL | 0.87–1.02 | parity: within ±3% per iteration |
| 2–8 nodes | oneCCL vs NCCL | 0.74–1.03 | parity or better in every cell |
| 2–8 nodes | CUDA-aware MPI | – | SYCL scales to 220 µs/iteration (Bump, 32 GPUs); native MPI hits a slow mode on the same stack, so no ratio is quoted |

How the solver works, how it got here (a 1.3–1.5× speedup over the first SYCL
version), and the full comparison are in [`sycl.md`](sycl.md).

## ⚙️ How it works

One iteration of CG, entirely queued on the GPU; the host blocks only where it
must:

| Step | Implementation |
| --- | --- |
| Halo exchange | one packing kernel, CUDA-aware MPI into the ghost vector, overlapped with the interior SpMV |
| SpMV | oneMath CSR (cuSPARSE), 32-bit indices when the matrix fits, analysed once |
| Reductions | two dot products per iteration, allreduced in device memory by MPI or oneCCL/NCCL |
| Scalars | alpha and beta computed inside the update kernels; one 16-byte readback per iteration for the convergence test |
| Validation | `‖b − Ax‖ / ‖b‖` recomputed on the host from the original matrix after every solve |

One GPU runs the same code with a one-part partition.

## 🗂️ Repository layout

| Path | Contents |
| --- | --- |
| [`src/solver`](src/solver) | the solver: `cg.cpp` (loop), `device_csr.cpp` (oneMath), `halo.cpp` (exchange), `collectives.cpp` (MPI/oneCCL) |
| [`src/matrix`](src/matrix) | Matrix Market reader, row partitioning (file, METIS, row-block) |
| [`apps/acg_cli`](apps/acg_cli) | the `acg` executable |
| [`tools/onemath_smoke`](tools/onemath_smoke) | oneMath GEMM/SpMV smoke tests |
| [`tooling/environments`](tooling/environments) | per-machine environments (`leonardo.sh`, `local.sh`) |
| [`tooling/jobs/leonardo/sycl`](tooling/jobs/leonardo/sycl/README.md) | the Leonardo harness: submission, SYCL/native campaign, comparison |
| [`tooling/data_analysis`](tooling/data_analysis) | log parsers and report generation |

## 🧰 Prerequisites

Built once under `$HOME/opt`; `tooling/environments/leonardo.sh` looks for
these paths by default.

| Dependency | Default path | Needed for |
| --- | --- | --- |
| DPC++ 6.3 with CUDA support | `$HOME/opt/dpcpp_6.3` | everything |
| oneMath (cuBLAS + cuSPARSE backends) | `$HOME/opt/oneMath` | everything |
| oneCCL with the NCCL backend | `$HOME/opt/oneccl-nccl-leonardo` | `--solver-collectives oneccl` |
| native aCG build | `$HOME/Projects/aCG-oshmpi/build-oshmpi/acg-cuda` | the native side of a comparison |

**DPC++.** An [intel/llvm source build with NVIDIA CUDA
support](https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda);
stock oneAPI DPC++ has no NVPTX target. The compiler must end up at
`$HOME/opt/dpcpp_6.3/llvm/build/install/bin/clang++`. The
[gpu-comm-benchmark README](https://github.com/y3rbiadit0/hpc-comm-playground#-prerequisites)
has the exact Leonardo build commands.

**oneMath.** Built with that compiler, the CUDA BLAS and sparse backends only:

```bash
git clone https://github.com/uxlfoundation/oneMath.git
cd oneMath
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_CXX_COMPILER=~/opt/dpcpp_6.3/llvm/build/bin/clang++ \
  -DCMAKE_C_COMPILER=~/opt/dpcpp_6.3/llvm/build/bin/clang \
  -DENABLE_MKLCPU_BACKEND=False \
  -DENABLE_MKLGPU_BACKEND=False \
  -DENABLE_CUBLAS_BACKEND=True \
  -DENABLE_CUSPARSE_BACKEND=True \
  -DBUILD_FUNCTIONAL_TESTS=False \
  -DBUILD_EXAMPLES=False
cmake --build build -j
cmake --install build
```

**oneCCL.** Optional; built by `cluster/leonardo/deps/oneccl-nccl.sh` in
gpu-comm-benchmark. Without it, build with `ACG_ENABLE_ONECCL=OFF` and leave out
the `oneccl` key.

## ▶️ Run on Leonardo

Run everything from the repository root on a login node. On Leonardo the
Makefile selects `ENV=leonardo` by itself (modules, DPC++, HPC-X, oneMath,
oneCCL).

**1️⃣ Build and check.**

```bash
BUILD_JOBS=4 make build-release                  # build-release/acg, HPC-X, oneCCL on

srun -A <account> -p boost_usr_prod --qos=boost_qos_dbg -N1 --gres=gpu:1 -t 00:10:00 \
  bash -lc 'make smoke-onemath'                  # FP64 SpMV at both index widths vs a CPU reference
```

**2️⃣ Stage the inputs.** Matrices are looked up by name under `data/matrices/`
(or `$HOME/datasets/suitesparse/mtx`); multi-GPU runs read fixed METIS
partitions, the same files the native campaign uses:

```bash
tooling/jobs/leonardo/sycl/stage-partitions.sh   # Bump_2911 and Queen_4147, 2-32 parts
```

**3️⃣ Run the comparison.** Every job runs SYCL and native interleaved, round by
round, on the same nodes:

```bash
export ACG_RESULTS_ROOT="$PWD/results-$(date +%Y%m%d)"

make campaign-plan REPEATS=3                     # what would be submitted, and walltimes
make now-1n4g NTRIALS=1 QOS=boost_qos_dbg        # quick paired check, foreground
make campaign REPEATS=3                          # Bump_2911: every topology, 3 allocations
make campaign REPEATS=3 MATRIX_NAME=Queen_4147
make compare                                     # SYCL / native, per job and per cell
```

| Variable | Default | Set it when |
| --- | --- | --- |
| `SBATCH_ACCOUNT` (make) / `ACG_SLURM_ACCOUNT` | `IscrC_HIGRAPH_0` | using another budget |
| `ACG_MPI` | `hpcx` | building or running against Open MPI 4.1.6 instead (`openmpi`) |
| `ACG_SYCL_BINARY` | `build-release/acg` | comparing another build |
| `ACG_NATIVE_ROOT` / `ACG_NATIVE_BINARY` | `$HOME/Projects/aCG-oshmpi` / `…/build-oshmpi/acg-cuda` | native aCG lives elsewhere |
| `ACG_RESULTS_ROOT` | `results` | always, so campaigns never share a tree |
| `BACKENDS` (make) | `mpi oneccl native-mpi native-nccl` | running a subset |

The [harness guide](tooling/jobs/leonardo/sycl/README.md) covers backend keys,
topologies, what each job writes (run log, provenance, script snapshot),
profiling and diagnostics.

## 🔧 Run one solve directly

```bash
# 4 GPUs on one node; the wrapper pins rank i to GPU i
srun -N1 --ntasks-per-node=4 --gres=gpu:4 tooling/jobs/gpu-rank-wrapper.sh \
  build-release/acg --matrix data/matrices/Bump_2911/Bump_2911.mtx \
  --partition "$HOME/datasets/suitesparse/partitions/Bump_2911_04_parts.mtx" \
  --device gpu --solver-collectives mpi \
  --manufactured-solution --seed 101 --residual-atol 0 --residual-rtol 1e-6 --max-iters 100000
```

| Option | Meaning |
| --- | --- |
| `--matrix <path>` | Matrix Market file (required) |
| `--residual-rtol <x>` / `--residual-atol <x>` | stop when `‖r‖ < rtol·‖r0‖` or `< atol` (rtol required) |
| `--partition <path>` | row-partition vector, one part per rank (else `ACG_PARTITIONER`: row-block or metis) |
| `--solver-collectives mpi\|oneccl` | how the two scalar allreduces are done |
| `--manufactured-solution`, `--seed <n>` | `b = A x_exact` from a random, normalised `x_exact` (native's generator) |
| `--max-iters <n>`, `--warmup <n>` | iteration cap; untimed warmup iterations (default 10, as native) |
| `--log-every <n>` | print the residual every n iterations |

The solver prints `solver:`, `timing:`, `waits:` and `validation:` lines; their
meaning is documented in the [harness guide](tooling/jobs/leonardo/sycl/README.md#what-a-job-writes).
Other machines: add `tooling/environments/<name>.sh` and pass `ENV=<name>`.


## 📝 Citation

The algorithm and the native implementation this project compares against are
from aCG:

> James D. Trotter, Sinan Ekmekçibaşı, Doğan Sağbili, Johannes Langguth, Xing
> Cai, and Didem Unat. 2025. **CPU- and GPU-initiated Communication Strategies
> for Conjugate Gradient Methods on Large GPU Clusters.** In *Proceedings of
> SC '25*, 298–315. https://doi.org/10.1145/3712285.3759774
