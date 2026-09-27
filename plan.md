# SYCL aCG Performance Implementation and Leonardo Validation Plan

## Objective and Status

Improve the SYCL solver's performance relative to native CUDA aCG while retaining
numerical validity and measuring each optimization independently.

This is an implementation plan. The optimizations, new options, and harness
extensions below are proposed, not implemented or measured. Existing build and
submission commands are identified separately.

### Implementation status (2026-09-26)

Implemented, **not yet compiled with DPC++ or run on Leonardo**. The changed C++
was checked only with `clang++ -fsyntax-only` against stub SYCL/oneMath/MPI
headers, so the first Leonardo build is the real compile check. No
measurements exist yet.

| Item | Where |
| --- | --- |
| `--opt idx32,pack-flat,mpi-overlap,device-scalars,single-cleanup` (all/none), default none | `solver_options.hpp`, `parse_args.cpp` |
| 32-bit device CSR, overflow guard, corrected byte estimate, halo SpMV included | `onemath_cuda_backend.cpp`, `index_width.hpp`, `DeviceHaloCsrSpmv` |
| Single-GPU cleanup, and the device-scalar loop on one GPU | `cg_single_gpu.cpp` |
| Flat packing | `HaloExchange` in `cg_multi_gpu_mpi.cpp` |
| MPI progress during interior SpMV | `cg_multi_gpu_mpi.cpp` loop |
| Device scalars (5A) + readback overlap (5B) | `cg_kernels.hpp`, both loops |
| Timing schema 2, max/min rank loop time, `--validate` true residual | `solver_result.hpp`, `solver_report.cpp`, `cg_solver.cpp` |
| Collective probe through the selected backend (`ACG_COLLECTIVE_PROBE`) | `run_collective_probe` |
| Variant manifests, alternating rounds, `runs.tsv`, provenance, script snapshots, `make screen-*` | `campaign-common.sh`, `launch.sh`, `job.sh`, `variants/`, `Makefile` |
| FP64 idx32/idx64 SpMV smoke test with irregular and empty rows | `tools/onemath_smoke/spmv_usm.cpp` |

Deviations from the plan text:

- The index width is a runtime switch (`--opt idx32`), not a build option
  `ACG_SPMV_INDEX_BITS`, so baseline and candidate share one binary and one
  allocation. The same applies to every other optimization.
- 5B does not use a second copy queue. The readback copy is queued on the
  in-order queue ahead of the search update, and the host waits only on the
  copy's event, so the update still overlaps the host's convergence decision.
- 5C: oneCCL keeps its waits (`ccl::event::wait` + queue wait) and the
  pre-collective producer wait. Device scalars still apply under oneCCL, but
  the adapter audit has not been done.
- Per-operation times under `mpi-overlap`/`device-scalars` are submission
  times. GPU kernel durations come from Nsight (`nsys-acg.sh`), not from
  SYCL event profiling.
- Section 10 follow-ups are not implemented.

Repositories:

- Implementation: `/Users/stormtrooper/Projects/university/acg-sycl`
- Native reference: `/Users/stormtrooper/Projects/university/aCG-native`
- Archived comparison:
  `/Users/stormtrooper/Projects/university/master-thesis/master-thesis-gpu-device-benchmark/data/main-campaign/acg-sycl-comparison.csv`

Source paths below are relative to the SYCL repository unless otherwise stated.
Leonardo commands run from its SYCL checkout; local macOS paths are not cluster
paths.

## 1. Priorities

| Priority | Change | Main benefit | Complexity | First test |
| --- | --- | --- | --- | --- |
| P0 | Paired runs and trustworthy timing | Establish whether changes actually help | Medium | Baseline |
| P1 | 32-bit device CSR indices | Lower SpMV bandwidth demand | Low–medium | `1n1g`, `1n4g` |
| P1 | One combined halo-packing kernel | Reduce neighbor-dependent launch overhead | Medium | `2n4g`, `8n4g` |
| P1 | MPI progress during interior SpMV | Improve communication/computation overlap | Medium | `2n4g`, `8n4g` |
| P2 | Device scalars and fewer host waits | Reduce per-iteration synchronization | High | `1n4g`, `2n4g` |
| P2 | Single-GPU loop cleanup | Remove an unnecessary reduction and vector pass | Low | `1n1g` |
| Conditional | oneCCL collective anomaly investigation | Resolve the extreme Queen slow mode | Medium | `4n4g` |
| Later | Halo SpMV tuning, persistent MPI, NCCL halo | Additional improvements after attribution | Medium–high | Profile-dependent |

The first two optimizations offer the best balance of likely impact and effort.
The asynchronous solver has the largest architectural upside but needs careful
dependency validation. These are qualitative expectations, not predicted speedups.

### Code-level motivation

- The main SYCL SpMV converts column indices to 64-bit and uses 64-bit row
  pointers. Native defaults to 32-bit indices unless built with `IDXSIZE=64`.
  FP64 values plus column indices therefore occupy 16 rather than 12 bytes per
  nonzero in the SYCL representation. This is not a prediction of a 33% solver
  slowdown; vector traffic, caching, and other work also matter.
- SYCL immediately waits after most GPU operations and reads both reduced CG
  scalars onto the CPU. Native keeps scalar recurrences on-device and overlaps
  residual readback with GPU work.
- SYCL submits one packing kernel per outgoing neighbor on an in-order queue.
  Native packs the complete outgoing buffer in one launch.
- SYCL waits for interior SpMV before entering `MPI_Waitall`. Native enters MPI
  while the GPU computes, allowing CPU-driven MPI progress during that interval.
- SYCL oneCCL/NCCL uses NCCL for scalar reductions but MPI for the halo. Native
  NCCL uses NCCL for both. Their comparison includes a halo-path difference.
- The single-GPU SYCL loop always computes an extra search-direction norm and
  implements the search update as separate `scal` and `axpy` operations.

## 2. P0 — Establish the Experimental Baseline

### 2.1 Freeze builds and execution settings

- [ ] Create independently identifiable baseline and candidate builds.
- [ ] Store source revision, patch/dirty-tree identity, binary checksum, and build
  configuration with every experiment.
- [ ] Record compiler and library versions: DPC++, oneMath, CUDA, cuBLAS,
  cuSPARSE, MPI, UCX, oneCCL, and NCCL.
- [ ] Record actually loaded libraries, including runtime-loaded libraries where
  applicable; environment module names alone are insufficient.
- [ ] Record matrix and partition checksums, node list, rank-to-GPU mapping, CPU
  affinity, GPU state, and effective transport settings.
- [ ] Record enabled optimizations and timing schema version.
- [ ] Record or freeze the job scripts used by an allocation. The current
  campaign sources scripts at job execution time, not at submission time.

Suggested build identities:

```text
baseline
idx32
pack-flat
mpi-overlap
device-scalars
single-cleanup
combined
```

Use the existing `ACG_SYCL_BINARY` and `ACG_RESULTS_ROOT` mechanisms. Keep the
software environment fixed across SYCL variants: initially HPC-X and the current
SYCL CUDA/oneMath installation. Library upgrades are separate experiments.

Retain isolated builds for attribution as well as cumulative builds for the final
solver. A candidate's manifest must identify its exact parent baseline and changes.

### 2.2 Add same-allocation variant comparisons

Files:

- `tooling/jobs/leonardo/sycl/campaign-common.sh`
- `tooling/jobs/leonardo/sycl/launch.sh`
- `Makefile`

- [ ] Add an optional variant manifest with label, absolute binary path, expected
  binary checksum, and additional arguments.
- [ ] Alternate baseline and candidate inside one allocation:

  ```text
  round 1: baseline → candidate
  round 2: candidate → baseline
  round 3: baseline → candidate
  ```

- [ ] Reverse the starting order in subsequent allocations.
- [ ] Separate output paths by variant and preserve backend labels.
- [ ] Include variants in dry-run output and walltime estimates.
- [ ] Record failed, timed-out, and nonconverged attempts explicitly.
- [ ] Preserve the existing single-binary execution mode.
- [ ] Keep profiling runs separate from timing runs.

The current topology campaign pairs backends, not different solver binaries.
The new manifest is an extension to implement, not an existing launch option.

### 2.3 Define timing before introducing asynchronous operations

Files:

- `src/solver/solver_common.hpp`
- `include/acg/solver/perf_breakdown.hpp`
- `include/acg/solver/solver_result.hpp`
- `src/solver/algorithms/cg_single_gpu.cpp`
- `src/solver/algorithms/cg_multi_gpu_mpi.cpp`
- `src/reporting/solver_report.cpp`
- Relevant parsers and tests under `tooling/data_analysis/`

Current `timed_call()` intervals include host waiting. Once operations become
asynchronous, unchanged timers would mostly measure submission cost.

Introduce distinct measurements:

| Metric | Definition |
| --- | --- |
| Initialization time | Allocation, uploads, descriptor setup, and preprocessing |
| Iteration-loop time | Completed CG loop, including required communication and convergence checks |
| Validation time | Download and independent residual/error checking |
| Submission time | CPU time spent issuing GPU operations |
| GPU execution time | Device timing verified against the actual CUDA timeline |
| MPI posting/wait time | Host intervals in communication calls |
| Scalar-readback wait | Host blocking required for convergence |

- [ ] Make the timing boundaries identical for baseline and candidate builds.
- [ ] Establish GPU readiness and a common distributed start boundary outside
  the loop timer.
- [ ] Drain outstanding work before stopping the loop timer.
- [ ] Report maximum rank loop time and retain rank distributions.
- [ ] Perform summary reductions after the measured interval.
- [ ] Do not add overlapping GPU and communication intervals into a total.
- [ ] Preserve legacy fields only where their definitions remain valid; version
  changed definitions and update analysis accordingly.
- [ ] Use short Nsight Systems captures to verify launches, waits, and overlap.
- [ ] Verify oneMath/interop event semantics before treating their profiling
  timestamps as GPU kernel durations.
- [ ] Make `ACG_SOLVER_DIAGNOSTICS` configurable instead of unconditionally on
  in the campaign. Avoid per-iteration diagnostic collectives in performance
  runs; collect summaries after timing.

Initialization, first-use/preprocessing costs, and steady-state work must remain
distinguishable. Do not change warm-up policy between variants silently.

### 2.4 Establish numerical validity

- [ ] Record convergence status, iterations, recurrence residual, and
  manufactured-solution error for the baseline.
- [ ] Independently recompute the true relative residual after timing:

  ```text
  ||b - A*x||_2 / ||b||_2
  ```

- [ ] For initial validation, use the original host matrix and a gathered
  solution to avoid depending on the optimized halo/SpMV path for the check.
- [ ] Establish and document numerical acceptance tolerances from the intended
  stopping criterion and baseline behavior before evaluating candidates.
- [ ] Preserve meaningful handling of a zero right-hand side.
- [ ] Do not require bitwise-identical output or identical iteration counts.

Recurrence residual and true residual may differ on these problems. Record both;
do not silently describe recurrence convergence as an independently verified true
residual. Manufactured-solution error must be compared with the baseline rather
than assumed to be tiny on an ill-conditioned problem.

## 3. Stage 1 — 32-bit Device CSR Indices

### Goal

Reduce main-SpMV memory traffic without changing the solver algorithm or
communication.

### Files

- `include/acg/solver/backends/onemath_cuda_backend.hpp`
- `src/solver/backends/onemath_cuda_backend.cpp`
- `CMakeLists.txt`
- `tooling/scripts/configure.sh`
- `Makefile`
- `tools/onemath_smoke/spmv_usm.cpp`

### Implementation

- [ ] Add a proposed build option `ACG_SPMV_INDEX_BITS=32|64` and forward it
  through the Makefile/configure script.
- [ ] Apply it to the device CSR representation, retaining host/global indexing.
- [ ] Convert both row pointers and column indices to the selected device type.
- [ ] Check dimensions, column indices, and CSR offsets before narrowing.
- [ ] Reject a forced 32-bit configuration clearly when required values exceed
  the representable range.
- [ ] Initialize oneMath CSR handles using matching index types.
- [ ] Correct byte estimates and log effective index width.
- [ ] Retain a 64-bit build as the experimental reference.

Keep the baseline default at 64-bit during attribution. Decide the production
default after validation.

### Validation and first measurements

- [ ] Extend the SpMV smoke test to FP64 and both index widths.
- [ ] Compare against a CPU reference for irregular row lengths and empty rows.
- [ ] Test overflow rejection without allocating an enormous matrix.
- [ ] Run Bump and Queen at `1n1g` and `1n4g`.
- [ ] Measure actual SpMV GPU duration and completed-loop time per iteration.
- [ ] Record preprocessing separately from steady-state execution.

Success means reduced execution time with unchanged numerical validity. A smaller
byte estimate alone is not sufficient evidence.

## 4. Stage 2 — Improve the Halo Path

Implement flat packing and MPI overlap as separate changes with separate A/B
results.

### 4A. One combined packing kernel

Primary file: `src/solver/algorithms/cg_multi_gpu_mpi.cpp`. Extract halo code into
a dedicated component if needed to keep the implementation manageable.

- [ ] Replace per-neighbor allocations and launches with one flattened export
  index array, one contiguous device send buffer, and per-neighbor offsets/counts.
- [ ] Submit one packing kernel covering the complete outgoing buffer.
- [ ] Point each send at its corresponding buffer slice.
- [ ] Retain the direct-to-ghost receive layout; do not introduce an unnecessary
  unpack operation.
- [ ] Retain the pack-completion boundary required before MPI reads the buffer.
- [ ] Keep buffers alive and unreused until the corresponding operations finish.

Tests:

- [ ] Verify ghost values using a deterministic vector indexed by global row.
- [ ] Cover multiple peers, repeated exports, zero-length cases, and unequal
  message sizes.
- [ ] Compare with the existing exchange on identical partitions.
- [ ] Confirm one packing launch per iteration in the GPU timeline.

First measurements:

- Bump: `1n4g`, `2n4g`, `8n4g`.
- Queen confirmation: `2n4g`, `8n4g`.
- Record packing duration, launch count, MPI posting time, and iteration time.

### 4B. MPI progress during interior SpMV

Files:

- `include/acg/solver/backends/onemath_cuda_backend.hpp`
- `src/solver/backends/onemath_cuda_backend.cpp`
- `src/solver/algorithms/cg_multi_gpu_mpi.cpp`

- [ ] Add asynchronous SpMV submission returning a completion event.
- [ ] Retain a synchronous method for the reference path.
- [ ] Implement this schedule:

  ```text
  pack and establish send-buffer readiness
  post receives and sends
  enqueue interior SpMV
  enter MPI_Waitall while the GPU executes SpMV
  enqueue halo SpMV after both prerequisites are satisfied
  ```

- [ ] Keep the existing in-order compute queue initially.
- [ ] Preserve send-buffer lifetime until sends complete.
- [ ] Ensure the previous GPU consumer has finished before MPI overwrites a
  reused ghost buffer.
- [ ] Preserve interior/halo accumulation ordering into the output vector.
- [ ] Ensure MPI writes are complete before GPU ghost reads.
- [ ] Verify the oneMath call actually allows asynchronous execution in the
  installed backend, rather than assuming removing `.wait()` is sufficient.

Posting receives earlier is a later micro-optimization, after buffer-reuse
dependencies are explicit.

Tests and acceptance:

- [ ] Repeated exchanges to expose stale-buffer and reuse races.
- [ ] Numerical checks at `1n4g` and `2n4g`.
- [ ] Timeline evidence of the CPU inside MPI during GPU interior SpMV.
- [ ] Test both MPI and oneCCL solver modes; both share the MPI halo.
- [ ] Measure end-to-end improvement at `2n4g` and `8n4g`.

A smaller `MPI_Waitall` duration alone is not an acceptance criterion: moving its
start changes what the interval measures.

## 5. Stage 3 — Device-resident Scalar Recurrences

### Goal

Remove unnecessary `gamma` host readback, compute `alpha` and `beta` on-device,
and reduce synchronization between dependent operations.

### Files

- `include/acg/solver/backends/cg_backend.hpp`
- `include/acg/solver/backends/onemath_cuda_backend.hpp`
- `src/solver/backends/onemath_cuda_backend.cpp`
- `src/solver/algorithms/cg_multi_gpu_mpi.cpp`
- Eventually `src/solver/algorithms/cg_single_gpu.cpp`
- Timing and reporting components

### 5A. Implement MPI first

- [ ] Maintain device scalars `rho_old`, `gamma`, and `rho_new` with explicit
  producer/consumer lifetimes.
- [ ] Make vector-update kernels consume device scalars.
- [ ] Preserve the existing fused solution/residual update initially.
- [ ] Submit dependent GPU work without immediate host waits where queue ordering
  supplies the required dependency.
- [ ] Wait for the scalar producer before CUDA-aware MPI accesses its buffer.
- [ ] Retain blocking `MPI_Allreduce` in this first implementation.
- [ ] Copy only convergence information required by the CPU.
- [ ] Use reusable pinned host storage for that copy.
- [ ] Preserve nonpositive/nonfinite scalar detection and consistent error
  handling across ranks.

Do not remove waits mechanically. MPI and SYCL do not automatically share
dependency tracking. Scalars must not be overwritten while updates still consume
their previous values.

### 5B. Overlap convergence readback

After the device-scalar path is correct:

- [ ] Use a separate copy queue in the same context for residual/status readback.
- [ ] Establish the producing operation's dependency explicitly.
- [ ] Submit the search-direction update while the copy proceeds.
- [ ] Wait for convergence information before deciding whether to continue.
- [ ] Ensure final solution updates and outstanding operations complete before
  returning or releasing resources.
- [ ] Check convergence every iteration initially.

Changing convergence-check frequency is a separate algorithmic experiment.

### 5C. Asynchronous oneCCL integration

Audit the installed oneCCL/NCCL adapter before changing completion behavior:

- [ ] Verify how producer dependencies reach the native CUDA stream.
- [ ] Verify what `ccl::event` completion guarantees.
- [ ] Establish how subsequent SYCL consumers depend on collective completion.
- [ ] Check for internal host synchronization in submission.
- [ ] Replace immediate collective waits only where a verified dependency path
  exists.

If the adapter cannot expose asynchronous completion safely, retain the required
wait and measure device-scalar improvements separately.

### Validation and acceptance

- [ ] Manufactured-solution tests over multiple iterations and topologies.
- [ ] Breakdown/nonfinite handling, early convergence, and iteration-limit exit.
- [ ] Repeated runs targeting event and buffer lifetime errors.
- [ ] Timeline confirmation that `gamma` readback disappears and host waits fall.
- [ ] Lower completed-loop time with retained numerical validity.

## 6. Single-GPU Cleanup

File: `src/solver/algorithms/cg_single_gpu.cpp`, with backend interface/kernel
support as needed.

- [ ] Compute `dot(s,s)` only when an enabled stopping criterion needs the update
  norm.
- [ ] Preserve requested difference-based stopping behavior.
- [ ] Replace `scal(beta,s)` plus `axpy(1,r,s)` with one fused search update.
- [ ] Update operation counts and work accounting.
- [ ] Test the two changes independently of index width before combining them.
- [ ] Validate both residual-only and enabled difference-based stopping modes.
- [ ] Measure Bump and Queen at `1n1g`.

This is a small independent package and can be implemented early, after the
measurement foundation.

## 7. Conditional Track — Queen oneCCL Slow Mode

The archived comparison contains a Queen `4n4g` oneCCL median of about 905 seconds,
with roughly 844–858 seconds recorded in allreduce across its three trials. Track
this separately from ordinary 1.3–3x performance gaps.

The existing `ACG_REDUCTION_PROBE` in `apps/acg_cli/main.cpp` invokes MPI even when
oneCCL is selected. It is not currently a oneCCL diagnostic.

- [ ] Extend the probe to exercise the selected collective implementation.
- [ ] Measure device allreduce completion.
- [ ] Measure device allreduce plus scalar readback.
- [ ] Measure dot → allreduce → dependent GPU update.
- [ ] Repeat after representative halo/SpMV work.
- [ ] Validate reduced values and downstream consumer results.
- [ ] Record latency distributions and per-rank arrival differences.
- [ ] Start with bounded `4n4g` diagnostics and preserve timeout/failure records.
- [ ] Compare MPI, oneCCL/NCCL, and direct native NCCL in the same allocation where
  practical, with isolated runtime environments.

Enable detailed NCCL/oneCCL logging only in diagnostic runs. Investigate the slow
mode rather than silently excluding it or averaging it into normal optimization
results.

## 8. Leonardo Execution Protocol

### 8.1 Existing baseline build command

Run from the SYCL checkout on Leonardo:

```bash
ENV=leonardo \
ACG_MPI=hpcx \
RELEASE_BUILD_DIR=build-release-baseline-hpcx \
ACG_ENABLE_ONECCL=ON \
ONECCL_ROOT="$HOME/opt/oneccl-nccl-leonardo" \
ACG_ENABLE_GPU_AWARE_MPI=ON \
ACG_ENABLE_METIS=ON \
BUILD_JOBS=4 \
make build-release
```

Build candidates into separate directories with the same environment. The
index-width option and variant manifest require implementation before use.

### 8.2 Existing baseline smoke submission

```bash
ACG_MPI=hpcx \
ACG_SYCL_BINARY="$PWD/build-release-baseline-hpcx/acg" \
ACG_RESULTS_ROOT="$PWD/results-opt/baseline-smoke" \
make submit-2n4g \
  MATRIX_NAME=Bump_2911 \
  SOLVERS="mpi oneccl" \
  NTRIALS=1 \
  QOS=boost_qos_dbg \
  TIME=00:30:00
```

Use `1n1g` for single-GPU tests. Debug QoS covers at most two nodes and 30 minutes;
`4n4g` and `8n4g` require the normal allocation path. Confirm the account and local
cluster policy when submitting.

### 8.3 Screening campaign

| Change | Initial topologies |
| --- | --- |
| Index width | `1n1g`, `1n4g` |
| Single-GPU cleanup | `1n1g` |
| Flat packing | `1n4g`, `2n4g`, `8n4g` |
| MPI overlap | `2n4g`, `8n4g` |
| Device scalars | `1n4g`, `2n4g`, then `8n4g` |

- [ ] Start with Bump and three paired rounds in one allocation per screened cell.
- [ ] Confirm promising changes on Queen and independent allocations.
- [ ] Keep existing campaign parameters:

  ```text
  manufactured solution
  seed 101
  residual-atol 0
  residual-rtol 1e-6
  max-iters 100000
  fixed partition files
  log-every 0
  ```

- [ ] Use a separately identified diagnostic mode or explicitly classify capped
  profiling runs. Lowering `--max-iters` currently causes nonconvergence exit and
  `.tmp` output; it must not be mislabeled as a valid completed solve.
- [ ] Keep diagnostics and profiler overhead out of performance comparisons.
- [ ] Test isolated changes before evaluating their combination.

### 8.4 Final validation campaign

After the combined implementation passes screening:

- Matrices: Bump and Queen.
- Topologies: `1n1g`, `1n2g`, `1n4g`, `2n4g`, `4n4g`, `8n4g`.
- Variants: baseline SYCL, optimized SYCL, native reference.
- Backends: MPI and NCCL/oneCCL where applicable; one single-GPU path at `1n1g`.
- Three independent allocations per matrix/topology.
- Three trials per variant/backend inside each allocation.

This is 36 matrix/topology allocations if each allocation includes all selected
variants and backends. Native and SYCL require isolated runtime environments.
Estimate walltime from screening results and confirm the dry-run expansion before
submission. More sampling may be required for unstable paths.

### 8.5 Reporting and promotion criteria

Primary metrics:

- Completed solver-loop time.
- Time per iteration.
- Iteration count and numerical checks.
- Per-allocation paired speedup.
- Median and spread across allocations.
- Failure and slow-mode frequency.

Use allocation-level summaries rather than treating every trial as an independent
hardware sample. Retain distributions for unstable paths and report both
convergence-normalized and end-to-end performance.

A practical promotion criterion is improvement reproducible beyond measured noise,
targeting approximately 5% or more end-to-end for major changes, without material
regressions elsewhere. Smaller changes can still be worthwhile when inexpensive
and consistently beneficial. A kernel-only improvement is not automatically a
solver improvement.

## 9. Implementation Order and Deliverables

1. [ ] Measurement/harness package: paired binaries, provenance, corrected timing,
   and post-solve validation.
2. [ ] Index-width package: 32/64-bit builds and FP64 SpMV checks.
3. [ ] Single-GPU cleanup: two small isolated ablations.
4. [ ] Flat-packing package: contiguous export buffers and one launch.
5. [ ] MPI-overlap package: asynchronous interior SpMV and verified dependencies.
6. [ ] Device-scalar package: MPI first, then oneCCL integration.
7. [ ] Combined validation: retained improvements and fresh native references.

Run the Queen collective diagnostic track as needed to distinguish its severe
slow mode from the ordinary performance gaps.

Each package must deliver:

- An independently runnable build and configuration manifest.
- Focused correctness results.
- A paired Leonardo comparison.
- A short explanation of whether the expected mechanism appears in the timeline.
- A retain/revise/reject decision.

Start with the measurement foundation, 32-bit CSR, and flat packing. Then address
MPI overlap and device-resident scalar scheduling to recover strong scaling.

## 10. Follow-up Work After the Main Stages

Only prioritize these if the new profiles show a remaining bottleneck:

- Halo SpMV: compare the current one-work-item-per-active-row kernel with tuned
  SYCL or library implementations using the same matrix split.
- Persistent MPI requests: measure separately after flat packing stabilizes the
  buffer layout.
- NCCL halo: add a grouped NCCL path with verified SYCL dependencies to match the
  native NCCL communication configuration more closely.
- CUDA/oneMath library matching: test versions and SpMV algorithms independently
  of solver changes.

Do not assume these changes help without measurements. Preserve the distinction
between implementation evidence, archived results, and newly executed Leonardo
experiments throughout the work.
