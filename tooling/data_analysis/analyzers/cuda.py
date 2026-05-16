from __future__ import annotations

from pathlib import Path

from tooling.data_analysis.analyzers.base import BaseResultsAnalyzer
from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import BenchmarkRun, BenchmarkSummary, DatasetRoot
from tooling.data_analysis.parsers.base import BaseParser
from tooling.data_analysis.parsers.cuda import CudaLogParser


class CudaResultsAnalyzer(BaseResultsAnalyzer):
    def __init__(self, parser: BaseParser | None = None) -> None:
        self.parser = parser or CudaLogParser()

    def parse_log(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        return self.parser.parse(path, dataset)

    def has_decomposed_communication(self, summary: BenchmarkSummary) -> bool:
        return summary.comm != CommunicationBackend.NVSHMEM

    def report_notes(self, summaries: list[BenchmarkSummary]) -> list[str]:
        if not any(summary.comm == CommunicationBackend.NVSHMEM for summary in summaries):
            return []
        return [
            "NVSHMEM/device-side CUDA logs report `allreduce` and `haloexchange` counters as zero; communication is folded into `other` in the solver breakdown. Treat NVSHMEM solver time as comparable, but not its per-operation communication timing.",
            "For SYCL comparison, use the same matrix, seed, residual tolerances, maximum iterations, rank layout, and at least three repeats. Compare MPI-to-MPI first; compare against NCCL/NVSHMEM only as CUDA-specific upper baselines unless SYCL has equivalent communication backends.",
        ]
