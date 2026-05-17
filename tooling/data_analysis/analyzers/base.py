from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from tooling.data_analysis.models import BenchmarkRun, DatasetRoot
from tooling.data_analysis.reporting import ResultsReportRenderer
from tooling.data_analysis.summarizer import summarize_runs
from tooling.data_analysis.summaries import BenchmarkSummary


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
        return summarize_runs(runs)

    def render_report(
        self,
        summaries: list[BenchmarkSummary],
        runs: list[BenchmarkRun],
        input_roots: dict[str, Path],
        reference_summaries: list[dict[str, object]] | None = None,
    ) -> str:
        return ResultsReportRenderer(
            has_decomposed_communication=self.has_decomposed_communication,
            report_notes=self.report_notes,
        ).render_report(summaries, runs, input_roots, reference_summaries)

    def has_decomposed_communication(self, summary: BenchmarkSummary) -> bool:
        return True

    def report_notes(self, summaries: list[BenchmarkSummary]) -> list[str]:
        return []
