from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import BenchmarkRun, BenchmarkSummary, DatasetRoot
from tooling.data_analysis.utils import fmt, markdown_table, median


class BaseResultsAnalyzer(ABC):
    """Template method for result analyzers with backend-specific parsing/report notes."""

    def collect_runs(self, datasets: Iterable[DatasetRoot]) -> list[BenchmarkRun]:
        runs: list[BenchmarkRun] = []
        for dataset in datasets:
            runs.extend(self.collect_dataset_runs(dataset))
        return runs

    def collect_dataset_runs(self, dataset: DatasetRoot) -> list[BenchmarkRun]:
        return [self.parse_log(path, dataset) for path in self.discover_logs(dataset.path)]

    def discover_logs(self, root: Path) -> list[Path]:
        return sorted(root.glob("**/*-stderr.txt"))

    @abstractmethod
    def parse_log(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        raise NotImplementedError

    def summarize(self, runs: Iterable[BenchmarkRun]) -> list[BenchmarkSummary]:
        grouped: dict[
            tuple[str, str, str, CommunicationBackend, int | None], list[BenchmarkRun]
        ] = {}
        for run in runs:
            key = (run.dataset, run.scale, run.variant, run.comm, run.ranks)
            grouped.setdefault(key, []).append(run)

        summaries: list[BenchmarkSummary] = []
        for (dataset, scale, variant, comm, ranks), group in grouped.items():
            solver_values = [run.solver_time_s for run in group if run.solver_time_s is not None]
            summaries.append(
                BenchmarkSummary(
                    dataset=dataset,
                    scale=scale,
                    variant=variant,
                    comm=comm,
                    ranks=ranks,
                    repeats=len(group),
                    solver_median=median(solver_values),
                    solver_min=min(solver_values) if solver_values else None,
                    solver_max=max(solver_values) if solver_values else None,
                    running_median=median(run.running_solver_s for run in group),
                    allreduce_median=median(run.allreduce_s for run in group),
                    allreduce_us_median=median(run.allreduce_us for run in group),
                    allreduce_calls_median=median(run.allreduce_calls for run in group),
                    halo_median=median(run.halo_s for run in group),
                    halo_us_median=median(run.halo_us for run in group),
                    iterations_median=median(run.iterations for run in group),
                    gflops_median=median(run.gflops for run in group),
                    wall_median=median(run.wall_s for run in group),
                    residual_median=median(run.residual_norm for run in group),
                    error_median=median(run.error_norm for run in group),
                )
            )

        return sorted(summaries, key=lambda item: (item.dataset, item.scale, item.comm, item.variant))

    def render_report(
        self,
        summaries: list[BenchmarkSummary],
        runs: list[BenchmarkRun],
        input_roots: dict[str, Path],
    ) -> str:
        matrices = sorted({run.matrix for run in runs})
        roots = ", ".join(f"{label}={path}" for label, path in sorted(input_roots.items()))
        lines = [
            "# aCG Results Summary",
            "",
            "Deterministic summary generated from verbose benchmark stderr logs. Values are medians across repeats unless stated otherwise.",
            "",
            f"Input roots: `{roots}`",
            f"Matrices: `{', '.join(matrices)}`",
            "",
            "## Solver And Communication Summary",
            "",
        ]

        lines.append(self.render_solver_summary(summaries))
        lines.extend(["", "## Best Solver Time By Scale", ""])
        lines.append(self.render_best_by_scale(summaries))

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
                    fmt(summary.solver_median),
                    fmt(summary.running_median),
                    fmt(summary.iterations_median, 0),
                    fmt(summary.gflops_median),
                ]
                for summary in self.best_by_scale(summaries)
            ],
        )

    def summary_row(self, summary: BenchmarkSummary) -> list[str]:
        has_decomposed_comm = self.has_decomposed_communication(summary)
        allreduce_percent = None
        if has_decomposed_comm and summary.solver_median and summary.allreduce_median is not None:
            allreduce_percent = 100 * summary.allreduce_median / summary.solver_median
        halo_percent = None
        if has_decomposed_comm and summary.solver_median and summary.halo_median is not None:
            halo_percent = 100 * summary.halo_median / summary.solver_median

        return [
            summary.dataset,
            summary.scale,
            summary.comm.value,
            fmt(summary.ranks, 0),
            str(summary.repeats),
            fmt(summary.solver_median),
            f"{fmt(summary.solver_min)}-{fmt(summary.solver_max)}",
            fmt(summary.running_median),
            fmt(summary.allreduce_median) if has_decomposed_comm else "N/A",
            fmt(summary.allreduce_us_median) if has_decomposed_comm else "N/A",
            fmt(summary.allreduce_calls_median, 0) if has_decomposed_comm else "N/A",
            fmt(summary.halo_median) if has_decomposed_comm else "N/A",
            fmt(summary.halo_us_median) if has_decomposed_comm else "N/A",
            fmt(summary.iterations_median, 0),
            fmt(summary.gflops_median) if summary.gflops_median else "N/A",
            fmt(summary.wall_median),
            fmt(allreduce_percent),
            fmt(halo_percent),
        ]

    def has_decomposed_communication(self, summary: BenchmarkSummary) -> bool:
        return True

    def best_by_scale(self, summaries: Iterable[BenchmarkSummary]) -> list[BenchmarkSummary]:
        best: dict[tuple[str, str], BenchmarkSummary] = {}
        for summary in summaries:
            if summary.solver_median is None:
                continue
            key = (summary.dataset, summary.scale)
            if key not in best or summary.solver_median < (best[key].solver_median or float("inf")):
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
                if not other or not base.solver_median or not other.solver_median:
                    continue
                rows.append(
                    [
                        key[0],
                        key[1].value,
                        fmt(key[2], 0),
                        baseline,
                        fmt(base.solver_median),
                        dataset,
                        fmt(other.solver_median),
                        fmt(base.solver_median / other.solver_median),
                    ]
                )
        return rows

    def report_notes(self, summaries: list[BenchmarkSummary]) -> list[str]:
        return []
