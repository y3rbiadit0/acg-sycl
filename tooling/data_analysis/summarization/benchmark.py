from __future__ import annotations

from collections.abc import Iterable

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import BenchmarkRun
from tooling.data_analysis.models.stats import SampleStats
from tooling.data_analysis.models.summaries import (
    BenchmarkSummary,
    CorrectnessStats,
    OperationStats,
    OverheadStats,
    PartitionImbalanceStats,
    RuntimeStats,
    SkewStats,
)


def summarize_runs(runs: Iterable[BenchmarkRun]) -> list[BenchmarkSummary]:
    grouped: dict[tuple[str, str, str, CommunicationBackend, int | None], list[BenchmarkRun]] = {}
    for run in runs:
        key = (run.dataset, run.scale, run.variant, run.comm, run.ranks)
        grouped.setdefault(key, []).append(run)

    summaries: list[BenchmarkSummary] = []
    for (dataset, scale, variant, comm, ranks), group in grouped.items():
        summaries.append(
            BenchmarkSummary(
                dataset=dataset,
                scale=scale,
                variant=variant,
                comm=comm,
                ranks=ranks,
                repeats=len(group),
                runtime=RuntimeStats(
                    solver_s=stats(group, "solver_time_s"),
                    running_solver_s=stats(group, "running_solver_s"),
                    wall_s=stats(group, "wall_s"),
                    iterations=stats(group, "iterations"),
                    gflops=stats(group, "gflops"),
                ),
                correctness=CorrectnessStats(
                    converged_repeats=sum(1 for run in group if run.converged is True),
                    residual_norm=stats(group, "residual_norm"),
                    rel_residual_r0=stats(group, "rel_residual_r0"),
                    rel_residual_rhs=stats(group, "rel_residual_rhs"),
                    error_norm=stats(group, "error_norm"),
                ),
                operations=OperationStats(
                    gemv_s=stats(group, "gemv_s"),
                    spmv_s=stats(group, "spmv_s"),
                    dot_s=stats(group, "dot_s"),
                    nrm2_s=stats(group, "nrm2_s"),
                    axpy_s=stats(group, "axpy_s"),
                    copy_s=stats(group, "copy_s"),
                    allreduce_s=stats(group, "allreduce_s"),
                    allreduce_us=stats(group, "allreduce_us"),
                    allreduce_calls=stats(group, "allreduce_calls"),
                    halo_s=stats(group, "halo_s"),
                    halo_us=stats(group, "halo_us"),
                ),
                overheads=OverheadStats(
                    pack_s=stats(group, "pack_s"),
                    p2p_s=stats(group, "p2p_s"),
                    host_sync_s=stats(group, "host_sync_s"),
                    other_s=stats(group, "other_s"),
                ),
                skew=SkewStats(
                    spmv_s=stats(group, "spmv_skew_s"),
                    allreduce_s=stats(group, "allreduce_skew_s"),
                    p2p_s=stats(group, "p2p_skew_s"),
                    halo_spmv_s=stats(group, "halo_spmv_skew_s"),
                    pack_s=stats(group, "pack_skew_s"),
                    host_sync_s=stats(group, "host_sync_skew_s"),
                ),
                partition=PartitionImbalanceStats(
                    local_nnz=stats(group, "local_nnz_imbalance"),
                    halo_nnz=stats(group, "halo_nnz_imbalance"),
                    ghosts=stats(group, "ghosts_imbalance"),
                    imports=stats(group, "imports_imbalance"),
                    exports=stats(group, "exports_imbalance"),
                ),
            )
        )

    return sorted(summaries, key=lambda item: (item.dataset, item.scale, item.comm, item.variant))


def stats(runs: Iterable[BenchmarkRun], field: str) -> SampleStats:
    return SampleStats.from_values(getattr(run, field) for run in runs)
