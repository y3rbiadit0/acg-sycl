# Native CUDA and SYCL aCG: Implementation Differences and Performance Diagnosis

Analysis date: 2026-09-26.

Implementation and validation follow-up: [plan.md](plan.md).

## 1. Executive Summary

The strongest code-level explanations for the SYCL solver's performance gap are:

1. Larger device CSR indices in the main sparse matrix–vector product (SpMV).
2. Frequent host waits and CPU handling of scalar recurrences.
3. One halo-packing launch per neighbor rather than one combined launch.
4. A CPU schedule that waits for interior SpMV before entering MPI completion,
   potentially reducing MPI progress during computation.
5. Different halo transports in the native NCCL and SYCL oneCCL/NCCL variants.
6. Additional work in the SYCL single-GPU algorithm.

These differences suggest two regimes: local memory traffic and scheduling costs
at small scale, followed by increasingly important packing, communication progress,
and synchronization costs under strong scaling. They identify optimization
candidates, but do not measure the contribution of each candidate.

The archive also contains a severe Queen oneCCL slow mode dominated by recorded
allreduce time. That requires separate diagnosis from the ordinary performance gap.

The defensible conclusion is that the current implementations differ materially
in execution strategy. The comparison does not isolate an intrinsic cost of SYCL
as a programming model.

## 2. Scope and Evidence

### 2.1 Source trees examined

| Repository | Local path | HEAD at inspection |
| --- | --- | --- |
| SYCL | `/Users/stormtrooper/Projects/university/acg-sycl` | `97799525e8fbea335beb3bc40767f84b90398b30` |
| Native | `/Users/stormtrooper/Projects/university/aCG-native` | `d358e55db87a7ce5a8aed1a64b2595d2dd4e2861` |

The SYCL checkout has existing staged/unstaged changes, including campaign,
configuration, and partitioning changes. Findings describe the inspected working
tree, not HEAD alone. The native working tree was clean at inspection. Line
references below identify the inspected files and may move after implementation.

The primary native reference is the standard host-initiated CUDA `acg` solver
using MPI or NCCL. Pipelined and monolithic device-NVSHMEM solvers change more of
the algorithm and execution organization and are not the basis of these findings.

### 2.2 Archived measurements

The quantitative examples use the thesis's generated evidence:

- [Comparison CSV](../master-thesis/master-thesis-gpu-device-benchmark/data/main-campaign/acg-sycl-comparison.csv)
- [SYCL trial CSV](../master-thesis/master-thesis-gpu-device-benchmark/data/main-campaign/acg-sycl-trials.csv)
- [Import manifest](../master-thesis/master-thesis-gpu-device-benchmark/data/main-campaign/acg-sycl-manifest.json)

The manifest specifies medians over residual-valid completed trials, with ratios
defined as SYCL divided by CUDA. It records input repository revision
`986e876cd976960cddbcb78702b84cd70358f59e`; this is provenance for the imported
evidence, not proof that every measured binary matches either current checkout.

The native and SYCL measurements are separate campaigns. They are not paired
executions in the same allocations. Current source evidence and archived timing
evidence must therefore remain distinguishable.

### 2.3 Evidence terminology

- *Observed implementation difference*: directly visible in the inspected source.
- *Archived observation*: recorded in the generated comparison or trial data.
- *Performance hypothesis*: a plausible mechanism requiring profiling or an
  isolated A/B experiment to establish its contribution.

This report is a static source and archive analysis. No new GPU execution,
Leonardo profiling, or optimization benchmark was performed for it.

## 3. Archived Performance Pattern

The following rows use the native NCCL / SYCL oneCCL-NCCL pair for distributed
runs and the single-GPU pair at `1n1g`. Times and ratios are rounded from the
comparison CSV.

| Matrix | Topology | Native solver time (s) | SYCL solver time (s) | SYCL/native |
| --- | --- | ---: | ---: | ---: |
| Bump_2911 | `1n1g` | 38.98 | 50.99 | 1.31 |
| Bump_2911 | `1n2g` | 21.59 | 28.83 | 1.34 |
| Bump_2911 | `1n4g` | 12.30 | 17.11 | 1.39 |
| Bump_2911 | `2n4g` | 7.94 | 13.58 | 1.71 |
| Bump_2911 | `4n4g` | 6.29 | 14.61 | 2.32 |
| Bump_2911 | `8n4g` | 6.37 | 19.78 | 3.11 |
| Queen_4147 | `1n1g` | 88.83 | 118.05 | 1.33 |
| Queen_4147 | `1n2g` | 47.94 | 63.40 | 1.32 |
| Queen_4147 | `1n4g` | 25.48 | 34.82 | 1.37 |
| Queen_4147 | `2n4g` | 15.19 | 21.04 | 1.39 |
| Queen_4147 | `4n4g` | 9.74 | 905.48 | 92.93 |
| Queen_4147 | `8n4g` | 7.51 | 16.67 | 2.22 |

The gap on one GPU demonstrates a substantial noncommunication component. The
Bump distributed ratio grows with scale, which is consistent with increasing
importance of fixed per-iteration and communication-related overheads.

Iteration counts do not explain the large gaps. For example, Bump `8n4g` has a
solver-time ratio of 3.106 and a time-per-iteration ratio of 3.068; Queen `4n4g`
has corresponding ratios of 92.931 and 92.648.

The MPI comparison has a different pattern because native MPI has a slow mode.
At Bump `8n4g`, native MPI has a median of 86.60 s and a 12.55–177.18 s range,
while SYCL MPI has a median of 17.59 s and a 16.83–18.72 s range. The resulting
ratio below one does not contradict the gap against native NCCL and does not
establish faster SYCL computation.

Allocation diversity also differs. For example, the Bump `8n4g` NCCL pair has
nine native trials from three jobs versus nine SYCL trials from one job. The
paired, multi-allocation protocol in `plan.md` addresses this limitation.

## 4. Summary of Implementation Differences

| Area | Native standard CUDA path | SYCL path | Likely affected regime |
| --- | --- | --- | --- |
| Main SpMV indices | 32-bit by default | 64-bit row and column indices | All scales, especially compute-heavy runs |
| Scalar recurrences | Device-side alpha/beta use | CPU computes alpha/beta after readback | All scales; increasing relative cost under strong scaling |
| Operation scheduling | Stream-ordered GPU work | Immediate waits after most operations | Short-kernel and strong-scaling regimes |
| Residual readback | Pinned host destination, separate copy stream | Synchronous copy to host scalar in distributed GPU-aware mode | Per-iteration synchronization |
| Halo packing | One combined kernel | One kernel per outgoing neighbor | Irregular/high-degree partitions |
| MPI requests | Persistent requests reused | New nonblocking requests each iteration | Posting overhead; benefit requires measurement |
| CPU during interior SpMV | Can be inside MPI completion | Waits for GPU before entering MPI completion | Multi-node overlap/progress |
| NCCL variant halo | Grouped NCCL point-to-point | MPI point-to-point | NCCL-pair interpretation |
| Halo SpMV | cuSPARSE off-diagonal operation | One work-item per active row, serial row traversal | Distributed computation |
| Single-GPU search norm | No extra search-direction norm in CUDA loop | Extra `dot(s,s)` every iteration | `1n1g` |
| Single-GPU search update | Fused kernel | Separate `scal` and `axpy` | `1n1g` |
| Phase timing | CUDA event intervals | Host wall intervals including waits | Attribution and profiling methodology |

## 5. Detailed Findings

### D1. Wider CSR indices in the main SYCL SpMV

**Observed difference.** The SYCL backend allocates both device CSR index arrays
as `std::int64_t`, explicitly widening the host column-index array. Native uses
`CUSPARSE_INDEX_32I` unless configured with `IDXSIZE=64`. The supplied native
Leonardo configure script does not select 64-bit indices by default; actual
archived binary index widths still need build-provenance confirmation.

For each FP64 value and column-index pair, the storage is:

| Component | Native default | SYCL |
| --- | ---: | ---: |
| Value | 8 bytes | 8 bytes |
| Column index | 4 bytes | 8 bytes |
| Total | 12 bytes | 16 bytes |

This is 33% more storage, and potentially streamed traffic, for those two arrays.
Row pointers are also wider. Vector accesses, caching, and algorithm selection
prevent translating that directly into a predicted solver slowdown.

**Hypothesis.** Index width is a strong candidate for a substantial part of the
local SpMV gap. It can affect library dispatch as well as memory traffic.

**Confirmation.** Compare otherwise identical 32-bit and 64-bit SYCL device CSR
builds, measuring actual SpMV duration and completed-loop time.

Sources:

- [SYCL CSR allocation and conversion](src/solver/backends/onemath_cuda_backend.cpp#L22-L39)
- [Native index type selection](../aCG-native/acg/config.h#L59-L94)
- [Native cuSPARSE descriptor](../aCG-native/acg/cgcuda.c#L530-L535)
- [Native Leonardo configure arguments](../aCG-native/cluster/leonardo/build/configure-acg.sh#L55-L89)

### D2. Frequent host waits and CPU scalar recurrences

**Observed difference.** SYCL `copy`, `scal`, `axpy`, `dot`, `dot_to_device`, and
`spmv` wait for completion internally. In the distributed GPU-aware path, both
reduced `gamma` and `rho_next` are copied to the CPU. The CPU computes `alpha` and
`beta` and passes their values into vector-update kernels, which also wait.

Simplified SYCL iteration after SpMV:

```text
dot(s,t) → wait
allreduce → completion
gamma readback → wait
CPU alpha calculation
fused x/r update → wait
dot(r,r) → wait
allreduce → completion
rho_next readback → wait
CPU convergence and beta calculation
search update → wait
```

Native uses device-pointer mode and update kernels that derive alpha/beta from
device scalars. It does not read gamma onto the CPU in the normal device-scalar
path. Its residual readback uses pinned host memory and a nonblocking copy
stream. Solution/search updates are enqueued before the convergence-copy wait.

Native MPI still synchronizes its producing stream before blocking allreduce.
Native NCCL enqueues `ncclAllReduce` on the stream without an immediate host wait.
The distinction is therefore not that native execution has no synchronization,
but that it retains more GPU-ordered dependencies and avoids extra readbacks.

**Hypothesis.** Host round trips and submission gaps become increasingly important
as local work shrinks. They can also delay rank arrival at communication calls.

**Confirmation.** Inspect CPU/GPU timelines and test device-resident scalar
recurrences with explicit producer/consumer ordering. Required MPI dependencies
must remain satisfied when host waits are removed.

Sources:

- [SYCL backend waits and scalar reads](src/solver/backends/onemath_cuda_backend.cpp#L95-L145)
- [SYCL distributed scalar/update loop](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L1058-L1156)
- [Native device-pointer and readback setup](../aCG-native/acg/cgcuda.c#L460-L479)
- [Native scalar/update/readback schedule](../aCG-native/acg/cgcuda.c#L901-L1016)
- [Native fused alpha update](../aCG-native/acg/cg-kernels-cuda.cu#L119-L181)
- [Native MPI and NCCL reductions](../aCG-native/acg/comm.c#L395-L425)

### D3. oneCCL is explicitly completed on the host

**Observed difference.** The SYCL collective wrapper calls
`ccl::allreduce(...).wait()` followed by `queue_->wait()`. Its dot producer has
already been waited on, and the scalar is subsequently read back synchronously.

**Hypothesis.** This serial completion protocol can lose benefits available from
stream-ordered NCCL execution and incur adapter/runtime overhead for tiny
reductions. The source alone does not establish whether the second wait adds
significant cost or is redundant in the installed adapter.

**Confirmation.** Audit the adapter's stream and event contracts, then compare
completed reduction and dependent-consumer timelines. An in-order SYCL queue
alone does not prove ordering for externally submitted work.

Source: [SYCL collective wrapper](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L124-L147).

### D4. Per-neighbor packing versus one combined launch

**Observed difference.** SYCL stores a send buffer and export-index array per
neighbor. It submits one packing kernel per nonempty peer and waits for the
complete event list before posting receives and sends. The shared queue is
in-order, so these packing kernels are ordered serially.

Native invokes one packing kernel over the total outgoing buffer. For `d`
nonempty outgoing peers, SYCL submits `d` packing kernels versus one in native.
SYCL export indices are also 64-bit, while the native CUDA packing kernel uses
`int` indices.

**Hypothesis.** Many small launches and their runtime handling amplify packing
cost with neighbor degree. A flattened gather can reduce this overhead even when
the total payload is modest.

**Confirmation.** Keep communication unchanged and replace only the export layout
and packing launch. Verify ghost values and compare launch counts, completed
packing time, and iteration time.

Sources:

- [SYCL send-buffer setup](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L554-L590)
- [SYCL begin, packing, and waits](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L595-L662)
- [SYCL queue ordering](src/runtime/queue_factory.cpp#L11-L29)
- [Native aggregate packing call](../aCG-native/acg/halo.c#L1600-L1608)
- [Native packing kernel](../aCG-native/acg/halo.cu#L45-L95)

### D5. Persistent MPI requests versus per-iteration posting

**Observed difference.** Native creates persistent send/receive requests once and
uses `MPI_Startall` on each exchange. SYCL invokes `MPI_Irecv` and `MPI_Isend` for
each peer on each iteration and rebuilds the active request list.

**Hypothesis.** Persistent requests may reduce repeated setup/posting overhead,
but their benefit is MPI-implementation-dependent. This is a secondary candidate
to test after flattening the send layout.

Sources:

- [SYCL GPU-aware posting](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L778-L823)
- [Native persistent setup](../aCG-native/acg/halo.c#L1155-L1180)
- [Native request activation](../aCG-native/acg/halo.c#L1610-L1616)

### D6. MPI progress opportunity during interior SpMV

**Observed difference.** Both paths begin the halo before interior SpMV, but their
host schedules differ:

```text
SYCL:
  post MPI → submit interior SpMV → CPU waits for GPU → MPI_Waitall

Native MPI:
  start MPI → enqueue interior SpMV → MPI_Waitall while GPU can compute
```

**Hypothesis.** If communication requires CPU-driven progress, SYCL's GPU wait
occupies time in which native can progress MPI. This can turn otherwise hidden
communication into exposed waiting after local computation.

This does not prove that SYCL has no overlap: offload, UCX behavior, and background
progress can still move data. It identifies a schedule that must be measured.

**Confirmation.** Return from SpMV submission without waiting, enter MPI while
the GPU computes, and establish both prerequisites before halo accumulation.
Compare the complete interval, not just a shifted MPI wait timer.

Sources:

- [SYCL iteration ordering](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L1037-L1053)
- [SYCL synchronous SpMV](src/solver/backends/onemath_cuda_backend.cpp#L138-L145)
- [Native iteration ordering](../aCG-native/acg/cgcuda.c#L851-L890)

### D7. The NCCL pair uses different halo transports

**Observed difference.** Selecting oneCCL in SYCL changes scalar reductions, not
the halo. The distributed halo remains MPI. Native's NCCL communicator performs
grouped NCCL sends/receives on a separate communication stream.

| Variant | Scalar reductions | Halo |
| --- | --- | --- |
| Native MPI | MPI | MPI |
| SYCL MPI | MPI | MPI |
| Native NCCL | NCCL | Grouped NCCL point-to-point |
| SYCL oneCCL/NCCL | oneCCL dispatch to NCCL | MPI |

**Interpretation.** The NCCL-pair ratio includes halo transport, progress, and
stream-scheduling differences. It cannot be assigned entirely to oneCCL overhead.

Sources:

- [SYCL halo selection](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L841-L860)
- [SYCL campaign backend contract](tooling/jobs/leonardo/sycl/campaign-common.sh#L14-L17)
- [Native communication stream](../aCG-native/acg/cgcuda.c#L495-L506)
- [Native grouped NCCL exchange](../aCG-native/acg/halo.c#L1397-L1421)
- [Native NCCL halo dispatch](../aCG-native/acg/halo.c#L1617-L1630)

### D8. Different halo SpMV implementations

**Observed difference.** SYCL launches one work-item per active halo row. Each
work-item serially traverses that row's nonzeros and adds its result to the output.
Native uses cuSPARSE for its off-diagonal matrix operation.

**Hypothesis.** Row-length imbalance, memory coalescing, and available parallelism
may penalize the simple SYCL kernel for some partitions. Restricting execution to
active rows can also be beneficial, so a replacement is not automatically faster.

This only affects distributed execution. Identical partition assignments do not
by themselves establish identical matrix layout, row ordering, or locality.

**Confirmation.** Compare halo GPU duration and row-length distributions before
selecting a library operation or tuned SYCL kernel.

Sources:

- [SYCL active-row kernel](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L210-L246)
- [Native off-diagonal descriptor](../aCG-native/acg/cgcuda.c#L554-L577)
- [Native off-diagonal SpMV](../aCG-native/acg/cgcuda.c#L887-L890)

### D9. Additional work in the single-GPU loop

**Observed differences.**

1. SYCL unconditionally computes `dot(s,s)` for the solution-update norm. The
   difference-based stopping tolerances default to zero, so residual-only runs
   do not need this reduction. The native CUDA loop does not compute it.
2. SYCL updates the search direction using separate `scal` and `axpy` operations,
   each waited on. Native's normal fused path performs one search-update kernel.

These differences add a reduction/readback and an extra vector pass/launch in
single-GPU execution. They do not apply unchanged to distributed SYCL, which
already has a fused search-direction update and a fused solution/residual update.

**Confirmation.** Gate the norm on its actual stopping criteria and fuse the
single-GPU search update in separate ablations. Preserve explicitly requested
difference-based stopping behavior.

Sources:

- [SYCL extra norm](src/solver/algorithms/cg_single_gpu.cpp#L116-L120)
- [SYCL two-step search update](src/solver/algorithms/cg_single_gpu.cpp#L157-L161)
- [SYCL default tolerances](include/acg/solver/solver_options.hpp#L39-L46)
- [SYCL distributed fused updates](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L276-L313)
- [Native search update](../aCG-native/acg/cgcuda.c#L995-L1006)

### D10. Toolchain, preprocessing, and transport are not fully matched

**Observed differences.** The supplied SYCL environment selects CUDA 12.2,
DPC++, and its oneMath installation. Native selects NVHPC 24.5 with CUDA 12.4
math libraries. Native conditionally invokes `cusparseSpMV_preprocess` for
supported versions. SYCL calls oneMath optimization, whose actual backend work
depends on its installation and library versions.

Native performs explicit warm-up loops. SYCL caches optimization for the last
input/output pointer pair and lazily optimizes on a new pair. The steady CG SpMV
uses stable vectors, so the code does not imply repeated preprocessing on every
iteration. First-use costs still affect timing boundaries.

Both build scripts default to Release and the Leonardo targets select A100.
A debug build is not the leading explanation from the supplied configuration.
The native CUDA Release flags include `-use_fast_math`; that flag alone does not
establish the cause of an FP64 SpMV or communication gap.

SYCL supports both OpenMPI and HPC-X selections; its default is OpenMPI, while
native uses HPC-X. Even when selecting HPC-X for both, the scripts specify
different UCX transport lists (`cuda` versus explicit CUDA transports). Actual
loaded libraries, effective transport expansion, and run metadata are needed to
establish equivalence.

**Confirmation.** Pin the environment for implementation A/B tests, log actual
libraries, and treat toolkit/algorithm/transport changes as separate experiments.

Sources:

- [SYCL environment](tooling/environments/leonardo.sh#L3-L81)
- [SYCL Release/debug configuration](tooling/scripts/configure.sh#L7-L34)
- [Native environment](../aCG-native/cluster/leonardo/env/modules.sh#L1-L13)
- [Native CUDA flags](../aCG-native/cuda/CMakeLists.txt#L122-L126)
- [Native preprocessing](../aCG-native/acg/cgcuda.c#L530-L577)
- [Native warm-up](../aCG-native/acg/cgcuda.c#L612-L710)
- [SYCL optimization cache](src/solver/backends/onemath_cuda_backend.cpp#L187-L218)
- [SYCL transport selection](tooling/jobs/leonardo/sycl/campaign-common.sh#L73-L92)
- [Native transport selection](../aCG-native/cluster/leonardo/env/transport.sh#L76-L84)

## 6. Timing and Attribution Limitations

### 6.1 Recorded compute time is not pure kernel execution time

SYCL `timed_call()` uses host wall time around calls that wait internally. Its
SpMV, dot, and update categories therefore include submission/runtime and waiting
costs. Native uses CUDA event intervals, some of which span communication
dependencies or host-driven gaps between enqueued work.

The archive's roughly 1.24–1.43 compute-category ratios for the distributed NCCL
pair are not proof that GPU kernels themselves are uniformly that much slower.
They are ratios of differently instrumented operation categories.

Sources:

- [SYCL host timing helper](src/solver/solver_common.hpp#L14-L18)
- [Native SpMV/halo event placement](../aCG-native/acg/cgcuda.c#L851-L893)
- [Native event aggregation](../aCG-native/acg/cgcuda.c#L1032-L1047)

### 6.2 Halo timers have different boundaries

SYCL records wall time in `MPI_Waitall` after its interior SpMV call has completed.
Native's CUDA halo event placement reflects dependencies around the exchange end,
including the wait exposed on the compute stream after overlap. Neither is a
matched measurement of standalone network transfer latency.

SYCL records packing in both its raw packing and host-sync counters; the report
subtracts packing when deriving exclusive host-sync time. Summing raw counters
would double count this interval.

Sources:

- [SYCL pack/wait accounting](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L600-L627)
- [SYCL exclusive host-sync derivation](src/reporting/solver_report.cpp#L34-L55)

### 6.3 Solver timer boundaries need normalization

The archive selects SYCL's CUDA-compatible timer for distributed runs and
`solve_time` for single-GPU runs. In the current source:

- The distributed CUDA-compatible timer includes some initialization before the
  iteration loop and stops before final download/validation.
- The general distributed `solve_time` starts at the loop but includes final
  download, error computation, and summary collectives.
- Single-GPU `solve_time` starts at the loop but also includes final download and
  host-side solution-error evaluation.
- Native's solve timer includes initial residual/norm work and the loop, with
  later profiling aggregation outside it.

Thus the archive manifest's shorthand “loop only” for single-GPU SYCL is not an
exact description of the current source boundary. Resolving historical behavior
requires the measured source snapshot. New A/B measurements should use explicit,
matched initialization, completed-loop, and validation intervals.

Sources:

- [SYCL distributed timer setup](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L953-L1032)
- [SYCL distributed finalization/timers](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L1173-L1205)
- [SYCL single-GPU loop start](src/solver/algorithms/cg_single_gpu.cpp#L101-L104)
- [SYCL single-GPU finalization/timer](src/solver/algorithms/cg_single_gpu.cpp#L167-L190)
- [Native solve start](../aCG-native/acg/cgcuda.c#L723-L744)
- [Native solve end](../aCG-native/acg/cgcuda.c#L1014-L1038)

### 6.4 Diagnostics can change the workload

The campaign currently enables solver diagnostics and defaults to five
per-iteration diagnostic gathers. These occur within the distributed loop timer.
Also, setting `--log-every` above zero in manufactured mode enables solution-error
work every iteration, even though output itself is printed less frequently.

These features should be controlled consistently during profiling and performance
tests. They are not established causes of the archived large-scale gap.

Sources:

- [Campaign diagnostics](tooling/jobs/leonardo/sycl/campaign-common.sh#L94-L97)
- [Distributed diagnostic gather](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L498-L543)
- [Solution-error enablement](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L955-L961)
- [Per-iteration solution-error work](src/solver/algorithms/cg_multi_gpu_mpi.cpp#L1128-L1141)

## 7. Separate Finding: Queen oneCCL Collective Slow Mode

The trial CSV records these residual-valid Queen `4n4g` oneCCL runs:

| Job / trial | Solver time (s) | Allreduce category (s) | Halo category (s) |
| --- | ---: | ---: | ---: |
| 58399809 / 1 | 918.30 | 858.07 | 48.12 |
| 58407728 / 1 | 905.48 | 856.35 | 36.58 |
| 58407728 / 2 | 901.67 | 843.56 | 45.97 |

The corresponding native NCCL median is 9.74 s; SYCL MPI's median for the same
matrix/topology is 15.42 s. The ordinary compute-category ratio remains about
1.36 for the NCCL pair, while the full solver ratio reaches 92.93.

At Queen `8n4g`, the oneCCL median recovers to 16.67 s, but the archive still has
a maximum of 1179.18 s. A representative median alone can hide this slow mode.

The measurements locate most recorded time in the allreduce interval. They do not
identify whether the cause is collective execution, adapter progress, rank skew,
CPU scheduling, or transport configuration. More matrix traffic cannot plausibly
explain the magnitude while the compute category remains near the ordinary ratio.

The current `ACG_REDUCTION_PROBE` always invokes MPI even when oneCCL is selected.
It must be extended or replaced before using it to diagnose oneCCL. Useful probes
include reduction alone, reduction plus readback, and dot → reduction → dependent
GPU consumer, with representative halo work preceding the sequence.

Sources:

- [Archived slow trials](../master-thesis/master-thesis-gpu-device-benchmark/data/main-campaign/acg-sycl-trials.csv#L173-L175)
- [Existing MPI-only reduction probe](apps/acg_cli/main.cpp#L56-L151)

## 8. Explanations Not Established by the Current Evidence

- **An unavoidable SYCL language penalty.** The code changes memory representation,
  synchronization, kernels, and communication organization together.
- **Universal loss of communication overlap.** The MPI progress schedule differs,
  but hardware/offloaded/background progress may still overlap transfers.
- **UVM migration of the current host-result scalar.** Despite its helper name,
  `make_shared_scalar()` now uses `sycl::malloc_host`, not `malloc_shared`.
  The source comment about an older UVM issue is not evidence that the inspected
  implementation still has that mechanism.
- **Host-staged halos in the supplied distributed campaign.** The campaign
  explicitly selects GPU-aware MPI for multi-process runs. Host staging is an
  alternative supported mode, not the selected campaign path.
- **No SpMV preprocessing in SYCL.** The backend calls oneMath optimization and
  caches the vector pair. Its actual implementation must be inspected/profiled.
- **NCCL carries the SYCL halo.** It does not in this implementation.
- **Iteration counts explain the main gap.** Archived time-per-iteration ratios
  retain the major slowdowns.
- **Residual-valid logs prove an independently recomputed true residual.** The
  current solver reports its recurrence residual and manufactured-solution error;
  independent true-residual validation remains part of the follow-up plan.

Additional sources:

- [Current pinned scalar allocation](src/solver/backends/onemath_cuda_backend.cpp#L164-L177)
- [Campaign GPU-aware selection](tooling/jobs/leonardo/sycl/campaign-common.sh#L191-L201)

## 9. Findings-to-Implementation Mapping

| Finding | Priority | Follow-up in plan.md | Key confirming measurement |
| --- | --- | --- | --- |
| Timing/attribution limits | P0 | Experimental baseline | Matched completed-loop timers and CUDA timeline |
| D1: 64-bit device CSR | P1 | Stage 1 | SpMV GPU duration and iteration time, 32-bit vs 64-bit |
| D4: per-neighbor packing | P1 | Stage 2A | Packing launches and completed-loop time |
| D6: MPI progress schedule | P1 | Stage 2B | CPU MPI progress concurrent with GPU SpMV |
| D2/D3: scalar and host waits | P2 | Stage 3 | Readback count, idle gaps, dependency-correct iteration time |
| D9: single-GPU extra work | P2 | Single-GPU cleanup | Removal of extra norm/vector pass and `1n1g` speedup |
| Queen slow mode | Conditional | Collective diagnostic track | Selected-backend latency distributions and rank arrivals |
| D5/D8/D10 and NCCL halo | Later | Follow-up work | Bottleneck-specific isolated A/B results |

Start with the measurement foundation, 32-bit CSR, and combined packing. Then
address MPI progress and device-resident scalar scheduling. The implementation
checklists, correctness criteria, build commands, and Leonardo campaign protocol
are in [plan.md](plan.md).
