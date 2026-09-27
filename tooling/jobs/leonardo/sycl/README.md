# Leonardo SYCL Solver Jobs

These jobs run the SYCL CG solver and native aCG **in the same allocation**, interleaved round by round, so every SYCL/native comparison is paired: same nodes, same fabric neighbours, same clocks. `sycl.md` at the repo root describes the solver and the comparison itself.

A topology is `<nodes>n<gpus_per_node>g` and one rank drives one GPU, so `2n4g` is 8 ranks; `--nodes`, `--ntasks-per-node`, `--gres`, `--time` and the job name all follow from the name.

```text
topology.sh          topology list, parser, backend-key rule, walltime policy
launch.sh            submitter: one topology, --all, --now, --dry-run, --explain
job.sh               the single submitted script
campaign-common.sh   the job body: interleaved rounds, provenance, run log
transport.sh         MPI/UCX settings shared by SYCL and native runs
compare-summary.py   paired SYCL/native summary of a results tree
nsys-acg.sh          Nsight Systems wrapper for short profiling runs
matrix-lib.sh, stage-partitions.sh
```

## Build

HPC-X is the default MPI (`ACG_MPI=hpcx`), the stack native aCG runs on. The build is always CUDA-aware MPI; oneCCL is optional.

```bash
ENV=leonardo ACG_ENABLE_ONECCL=ON ONECCL_ROOT="$HOME/opt/oneccl-nccl-leonardo" \
ACG_ENABLE_METIS=ON BUILD_JOBS=4 make build-release

# on a GPU node: FP64 SpMV at both index widths vs a CPU reference
srun -A IscrC_HIGRAPH_0 -p boost_usr_prod --qos=boost_qos_dbg -N1 --gres=gpu:1 -t 00:10:00 \
  bash -lc 'ENV=leonardo make smoke-onemath'
```

A job refuses to run a SYCL binary built against a different MPI than the one it loaded (both stacks ship a `libmpi.so.40`).

Native aCG is taken from `ACG_NATIVE_ROOT` (default `$HOME/Projects/aCG-oshmpi`, the native campaign's own default) and `ACG_NATIVE_BINARY` (default `$ACG_NATIVE_ROOT/build-oshmpi/acg-cuda`); each native run loads that repo's `cluster/leonardo/env/modules.sh` and `oshmpi.sh` in its own subshell, starting from the job's login environment.

## Partitions

Multi-rank runs read a fixed METIS decomposition from a file, the same files the native campaign reads, so both solve the identical distributed problem. Stage them once:

```bash
tooling/jobs/leonardo/sycl/stage-partitions.sh              # Bump_2911 and Queen_4147
```

## Backend Keys

| key | runs | results label |
| --- | --- | --- |
| `mpi` | SYCL, allreduces through CUDA-aware MPI | `acg-sycl-mpi` |
| `oneccl` | SYCL, allreduces through oneCCL/NCCL (halo still MPI) | `acg-sycl-oneccl-nccl` |
| `native-mpi` | native `--solver acg --comm mpi` | `acg-cg-mpi` |
| `native-nccl` | native `--solver acg --comm nccl` (halo over NCCL too) | `acg-cg-nccl` |
| `none`, `native-none` | one GPU; every key maps to these at 1 rank | `acg-sycl-single`, `acg-cg-single` |

Every run uses the same method: manufactured solution with seed 101, `--residual-atol 0 --residual-rtol 1e-6`, 100000 max iterations, **10 warmup iterations**, the same matrix file, partition file, MPI and `transport.sh` settings.

## Run

```bash
export ACG_SYCL_BINARY="$PWD/build-release/acg"
export ACG_RESULTS_ROOT="$PWD/results-$(date +%Y%m%d)"

make campaign-plan                              # what would be submitted, and walltimes
make campaign                                   # Bump_2911: one job per topology, 4 keys x 3 rounds
make campaign MATRIX_NAME=Queen_4147 REPEATS=3  # 3 independent allocations per topology
make submit-2n4g SOLVERS="mpi native-mpi"       # one topology, chosen keys
make now-1n4g NTRIALS=1 QOS=boost_qos_dbg       # salloc, foreground, debug queue (<=2 nodes, 30 min)
make explain-8n4g                               # how one topology resolves

make compare                                    # paired summary of $ACG_RESULTS_ROOT
```

Each round runs every key once and the order reverses from round to round (A B C D, D C B A, ...), starting reversed on odd job ids. `NTRIALS` is the number of rounds; `REPEATS` the number of independent allocations. Allocation-to-allocation spread is the larger noise (about 20% between 8n4g jobs), so report the **median over allocations of the per-job ratio**, which is what `compare-summary.py` prints last.

## What A Job Writes

- `$ACG_RESULTS_ROOT/<label>/suitesparse/<matrix>/<matrix>-rtol-<rtol>-<NNN>-nodes-<NNNN>-procs-<jobid>-<round>-{stdout,stderr}.txt` -- the native campaign's naming, for SYCL and native alike. Only a converged run (exit 0) leaves `.tmp`.
- `$ACG_RESULTS_ROOT/runs.tsv` -- a `started` and an `ok`/`failed` line per run; a walltime kill leaves only `started`.
- `$ACG_RESULTS_ROOT/provenance/<jobid>/` -- both git revisions and dirty state, binary checksums and CMake settings, `ldd` of both binaries, compiler/CUDA/MPI/UCX versions, modules, transport environment, per-rank host/GPU/CPU mask, GPU clocks, partition checksum (the matrix too with `ACG_CHECKSUM_MATRIX=1`). SYCL stdout also ends with the shared objects actually mapped (`loaded_library:`), dlopen'd ones included.
- `.job-snapshots/<time>/` -- the job scripts as submitted; the job runs from this copy, so editing the scripts never changes a queued job.

SYCL stdout lines the analysis reads:

```text
solver: converged=true iterations=25732 ... solve_time=8.31s spmv_index_bits=32
timing: schema=3 setup_s=… warmup_s=… solver_s=… solver_max_s=… solver_min_s=… per_iter_us=… post_solve_s=… validation_s=…
waits: pack_s=… halo_s=… allreduce_s=… readback_s=…
validation: true_residual=… true_rel_residual=… recurrence_rel_residual=…
```

`solver_max_s` is the same quantity as native's `total solver time` (slowest rank, after warmup, barrier through initial residual and loop). `waits` are the only intervals in which the host blocks; everything else is asynchronous submission, which is why there is no per-kernel breakdown -- use `nsys-acg.sh` for that. `validation` recomputes `||b - A x|| / ||b||` on the host from the original matrix after every solve.

## Diagnostics

```bash
ACG_COLLECTIVE_PROBE=2000 make submit-4n4g SOLVERS="mpi oneccl" NTRIALS=1   # allreduce latency distributions
CCL_LOG_LEVEL=info NCCL_DEBUG=INFO ...                                       # oneCCL/NCCL logging
ACG_EXTRA_ARGS="--log-every 1000"                                            # residual history (SYCL)
```

## Jobs Stuck In Pending

```bash
squeue -u $USER -o "%.10i %.25j %.8T %.10l %.6D %R"
saldo -b    # is IscrC_HIGRAPH_0 out of hours, or expired?
```

`AssocGrpBillingMinutes` or `InvalidAccount` means the budget is gone; `Priority` or `Resources` means a busy queue.
