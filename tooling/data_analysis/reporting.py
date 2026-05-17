from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import BenchmarkRun
from tooling.data_analysis.summaries import BenchmarkSummary
from tooling.data_analysis.utils import fmt, markdown_table


class ResultsReportRenderer:
    def __init__(
        self,
        has_decomposed_communication: Callable[[BenchmarkSummary], bool],
        report_notes: Callable[[list[BenchmarkSummary]], list[str]],
    ) -> None:
        self.has_decomposed_communication = has_decomposed_communication
        self.report_notes = report_notes

    def render_report(
        self,
        summaries: list[BenchmarkSummary],
        runs: list[BenchmarkRun],
        input_roots: dict[str, Path],
        reference_summaries: list[dict[str, object]] | None = None,
    ) -> str:
        matrices = sorted({run.matrix for run in runs})
        roots = ", ".join(f"{label}={path}" for label, path in sorted(input_roots.items()))
        lines = [
            "# aCG Results Summary",
            "",
            "Deterministic summary generated from verbose benchmark stderr logs. "
            "Values are medians across repeats unless stated otherwise.",
            "",
            f"Input roots: `{roots}`",
            f"Matrices: `{', '.join(matrices)}`",
            "",
            "## Solver And Communication Summary",
            "",
        ]

        lines.append(self.render_solver_summary(summaries))
        lines.extend(["", "## Correctness And Runtime", ""])
        lines.append(self.render_correctness_summary(summaries))
        lines.extend(["", "## Best Solver Time By Scale", ""])
        lines.append(self.render_best_by_scale(summaries))

        breakdown = self.render_operation_breakdown(summaries)
        if breakdown:
            lines.extend(["", "## Operation Breakdown", "", breakdown])

        skew = self.render_rank_skew(summaries)
        if skew:
            lines.extend(["", "## Rank Skew", "", skew])

        imbalance = self.render_partition_imbalance(summaries)
        if imbalance:
            lines.extend(["", "## Partition Imbalance", "", imbalance])

        reference_rows = self.reference_comparison_rows(summaries, reference_summaries or [])
        if reference_rows:
            lines.extend(["", "## CUDA Reference Comparison", ""])
            lines.append(
                markdown_table(
                    [
                        "Scale",
                        "SYCL comm",
                        "Ranks",
                        "SYCL solver s",
                        "SYCL iters",
                        "SYCL ms/iter",
                        "CUDA MPI s",
                        "CUDA MPI iters",
                        "CUDA MPI ms/iter",
                        "CUDA best s",
                        "Solver slowdown vs MPI",
                        "Solver slowdown vs best",
                        "Iter ratio vs MPI",
                        "ms/iter ratio vs MPI",
                    ],
                    reference_rows,
                )
            )

        compare = self.comparison_rows(summaries)
        if compare:
            lines.extend(["", "## Cross-Dataset Comparison", ""])
            lines.append(
                markdown_table(
                    [
                        "Scale",
                        "Comm",
                        "Ranks",
                        "Baseline",
                        "Baseline solver s",
                        "Compared",
                        "Compared solver s",
                        "Baseline/Compared",
                    ],
                    compare,
                )
            )

        notes = self.report_notes(summaries)
        if notes:
            lines.extend(["", "## Notes", "", *notes])

        return "\n".join(lines) + "\n"

    def render_correctness_summary(self, summaries: list[BenchmarkSummary]) -> str:
        return markdown_table(
            [
                "Dataset",
                "Scale",
                "Comm",
                "Ranks",
                "Converged",
                "Iters",
                "Residual",
                "Rel residual r0",
                "Rel residual rhs",
                "Error norm",
                "Solver s",
                "ms/iter",
                "GF/s",
            ],
            [
                [
                    summary.dataset,
                    summary.scale,
                    summary.comm.value,
                    fmt(summary.ranks, 0),
                    f"{summary.correctness.converged_repeats}/{summary.repeats}",
                    fmt(summary.runtime.iterations.median, 0),
                    fmt(summary.correctness.residual_norm.median),
                    fmt(summary.correctness.rel_residual_r0.median),
                    fmt(summary.correctness.rel_residual_rhs.median),
                    fmt(summary.correctness.error_norm.median),
                    fmt(summary.runtime.solver_s.median),
                    fmt(summary.runtime.ms_per_iter),
                    fmt(summary.runtime.gflops.median),
                ]
                for summary in summaries
            ],
        )

    def render_solver_summary(self, summaries: list[BenchmarkSummary]) -> str:
        return markdown_table(
            [
                "Dataset",
                "Scale",
                "Comm",
                "Ranks",
                "Repeats",
                "Solver s",
                "Solver min-max s",
                "Running s",
                "Allreduce s",
                "Allreduce us/op",
                "Allreduce calls",
                "Halo s",
                "Halo us/msg",
                "Iters",
                "GF/s",
                "Wall s",
                "Allreduce %",
                "Halo %",
            ],
            [self.summary_row(summary) for summary in summaries],
        )

    def render_best_by_scale(self, summaries: Iterable[BenchmarkSummary]) -> str:
        return markdown_table(
            ["Dataset", "Scale", "Best comm", "Ranks", "Solver s", "Running s", "Iters", "GF/s"],
            [
                [
                    summary.dataset,
                    summary.scale,
                    summary.comm.value,
                    fmt(summary.ranks, 0),
                    fmt(summary.runtime.solver_s.median),
                    fmt(summary.runtime.running_solver_s.median),
                    fmt(summary.runtime.iterations.median, 0),
                    fmt(summary.runtime.gflops.median),
                ]
                for summary in self.best_by_scale(summaries)
            ],
        )

    def render_operation_breakdown(self, summaries: list[BenchmarkSummary]) -> str:
        rows = []
        for summary in summaries:
            solver = summary.runtime.solver_s.median
            if not solver:
                continue
            spmv = summary.operations.spmv_s.median or summary.operations.gemv_s.median
            if not any(
                value is not None
                for value in [
                    spmv,
                    summary.operations.dot_s.median,
                    summary.operations.nrm2_s.median,
                    summary.operations.allreduce_s.median,
                    summary.operations.halo_s.median,
                    summary.overheads.pack_s.median,
                    summary.overheads.host_sync_s.median,
                    summary.overheads.other_s.median,
                ]
            ):
                continue
            rows.append(
                [
                    summary.dataset,
                    summary.scale,
                    summary.comm.value,
                    fmt(solver),
                    fmt(self.percent(spmv, solver)),
                    fmt(
                        self.percent(
                            self.sum_values(
                                summary.operations.dot_s.median,
                                summary.operations.nrm2_s.median,
                            ),
                            solver,
                        )
                    ),
                    fmt(self.percent(summary.operations.allreduce_s.median, solver)),
                    fmt(
                        self.percent(
                            summary.operations.halo_s.median or summary.overheads.p2p_s.median,
                            solver,
                        )
                    ),
                    fmt(
                        self.percent(
                            self.sum_values(
                                summary.overheads.pack_s.median,
                                summary.overheads.host_sync_s.median,
                            ),
                            solver,
                        )
                    ),
                    fmt(self.percent(summary.overheads.other_s.median, solver)),
                ]
            )
        if not rows:
            return ""
        return markdown_table(
            [
                "Dataset",
                "Scale",
                "Comm",
                "Solver s",
                "SpMV/GEMV %",
                "Dot+nrm2 %",
                "Allreduce %",
                "Halo/P2P %",
                "Pack+host sync %",
                "Other %",
            ],
            rows,
        )

    def render_rank_skew(self, summaries: list[BenchmarkSummary]) -> str:
        rows = []
        for summary in summaries:
            for op, value in [
                ("spmv", summary.skew.spmv_s.median),
                ("allreduce", summary.skew.allreduce_s.median),
                ("p2p", summary.skew.p2p_s.median),
                ("halo_spmv", summary.skew.halo_spmv_s.median),
                ("pack", summary.skew.pack_s.median),
                ("host_sync", summary.skew.host_sync_s.median),
            ]:
                if value is not None:
                    rows.append(
                        [
                            summary.dataset,
                            summary.scale,
                            summary.comm.value,
                            fmt(summary.ranks, 0),
                            op,
                            fmt(value),
                        ]
                    )
        if not rows:
            return ""
        return markdown_table(
            ["Dataset", "Scale", "Comm", "Ranks", "Operation", "Skew s"], rows
        )

    def render_partition_imbalance(self, summaries: list[BenchmarkSummary]) -> str:
        rows = []
        for summary in summaries:
            if not any(
                value is not None
                for value in [
                    summary.partition.local_nnz.median,
                    summary.partition.halo_nnz.median,
                    summary.partition.ghosts.median,
                    summary.partition.imports.median,
                    summary.partition.exports.median,
                ]
            ):
                continue
            rows.append(
                [
                    summary.dataset,
                    summary.scale,
                    summary.comm.value,
                    fmt(summary.ranks, 0),
                    fmt(summary.partition.local_nnz.median),
                    fmt(summary.partition.halo_nnz.median),
                    fmt(summary.partition.ghosts.median),
                    fmt(summary.partition.imports.median),
                    fmt(summary.partition.exports.median),
                ]
            )
        if not rows:
            return ""
        return markdown_table(
            [
                "Dataset",
                "Scale",
                "Comm",
                "Ranks",
                "Local nnz max/min",
                "Halo nnz max/min",
                "Ghosts max/min",
                "Imports max/min",
                "Exports max/min",
            ],
            rows,
        )

    def summary_row(self, summary: BenchmarkSummary) -> list[str]:
        has_decomposed_comm = self.has_decomposed_communication(summary)
        allreduce_percent = None
        solver = summary.runtime.solver_s.median
        allreduce = summary.operations.allreduce_s.median
        halo = summary.operations.halo_s.median
        if has_decomposed_comm and solver and allreduce is not None:
            allreduce_percent = 100 * allreduce / solver
        halo_percent = None
        if has_decomposed_comm and solver and halo is not None:
            halo_percent = 100 * halo / solver

        return [
            summary.dataset,
            summary.scale,
            summary.comm.value,
            fmt(summary.ranks, 0),
            str(summary.repeats),
            fmt(solver),
            f"{fmt(summary.runtime.solver_s.min)}-{fmt(summary.runtime.solver_s.max)}",
            fmt(summary.runtime.running_solver_s.median),
            fmt(allreduce) if has_decomposed_comm else "N/A",
            fmt(summary.operations.allreduce_us.median) if has_decomposed_comm else "N/A",
            fmt(summary.operations.allreduce_calls.median, 0) if has_decomposed_comm else "N/A",
            fmt(halo) if has_decomposed_comm else "N/A",
            fmt(summary.operations.halo_us.median) if has_decomposed_comm else "N/A",
            fmt(summary.runtime.iterations.median, 0),
            fmt(summary.runtime.gflops.median) if summary.runtime.gflops.median else "N/A",
            fmt(summary.runtime.wall_s.median),
            fmt(allreduce_percent),
            fmt(halo_percent),
        ]

    def best_by_scale(self, summaries: Iterable[BenchmarkSummary]) -> list[BenchmarkSummary]:
        best: dict[tuple[str, str], BenchmarkSummary] = {}
        for summary in summaries:
            solver = summary.runtime.solver_s.median
            if solver is None:
                continue
            key = (summary.dataset, summary.scale)
            best_solver = best[key].runtime.solver_s.median if key in best else None
            if key not in best or solver < (best_solver or float("inf")):
                best[key] = summary
        return sorted(best.values(), key=lambda item: (item.dataset, item.scale))

    def comparison_rows(self, summaries: list[BenchmarkSummary]) -> list[list[str]]:
        by_key: dict[tuple[str, CommunicationBackend, int | None], dict[str, BenchmarkSummary]] = {}
        for summary in summaries:
            by_key.setdefault((summary.scale, summary.comm, summary.ranks), {})[
                summary.dataset
            ] = summary

        datasets = sorted({summary.dataset for summary in summaries})
        if len(datasets) < 2:
            return []

        baseline = datasets[0]
        rows: list[list[str]] = []
        for key, entries in sorted(by_key.items()):
            if baseline not in entries:
                continue
            base = entries[baseline]
            for dataset in datasets[1:]:
                other = entries.get(dataset)
                base_solver = base.runtime.solver_s.median
                other_solver = other.runtime.solver_s.median if other else None
                if not other or not base_solver or not other_solver:
                    continue
                rows.append(
                    [
                        key[0],
                        key[1].value,
                        fmt(key[2], 0),
                        baseline,
                        fmt(base_solver),
                        dataset,
                        fmt(other_solver),
                        fmt(base_solver / other_solver),
                    ]
                )
        return rows

    def reference_comparison_rows(
        self, summaries: list[BenchmarkSummary], reference_summaries: list[dict[str, object]]
    ) -> list[list[str]]:
        if not reference_summaries:
            return []

        by_direct = {
            (item["scale"], item["comm"], item["ranks"]): item for item in reference_summaries
        }
        best_by_scale: dict[str, dict[str, object]] = {}
        for item in reference_summaries:
            solver = item.get("solver")
            if solver is None:
                continue
            scale = str(item["scale"])
            if scale not in best_by_scale or solver < best_by_scale[scale]["solver"]:
                best_by_scale[scale] = item

        rows = []
        for summary in summaries:
            solver = summary.runtime.solver_s.median
            if solver is None:
                continue
            direct = by_direct.get((summary.scale, summary.comm.value, summary.ranks))
            best = best_by_scale.get(summary.scale)
            if not direct and not best:
                continue
            direct_solver = direct.get("solver") if direct else None
            direct_iters = direct.get("iterations") if direct else None
            direct_ms_iter = self.ms_per_iter(direct_solver, direct_iters)
            best_solver = best.get("solver") if best else None
            iterations = summary.runtime.iterations.median
            ms_per_iter = summary.runtime.ms_per_iter
            rows.append(
                [
                    summary.scale,
                    summary.comm.value,
                    fmt(summary.ranks, 0),
                    fmt(solver),
                    fmt(iterations, 0),
                    fmt(ms_per_iter),
                    fmt(direct_solver),
                    fmt(direct_iters, 0),
                    fmt(direct_ms_iter),
                    fmt(best_solver),
                    fmt(solver / direct_solver if direct_solver else None),
                    fmt(solver / best_solver if best_solver else None),
                    fmt(iterations / direct_iters if iterations and direct_iters else None),
                    fmt(ms_per_iter / direct_ms_iter if ms_per_iter and direct_ms_iter else None),
                ]
            )
        return rows

    def percent(self, value: float | None, total: float) -> float | None:
        if value is None:
            return None
        return 100 * value / total

    def sum_values(self, *values: float | None) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered)

    def ms_per_iter(self, solver: object, iterations: object) -> float | None:
        if (
            not isinstance(solver, (float, int))
            or not isinstance(iterations, (float, int))
            or not iterations
        ):
            return None
        return 1000 * float(solver) / float(iterations)
