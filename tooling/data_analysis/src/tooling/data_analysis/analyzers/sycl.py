from __future__ import annotations

from pathlib import Path

from ..enums import CommunicationBackend
from ..models import BenchmarkRun, DatasetRoot
from ..models.summaries import BenchmarkSummary
from ..parsers.base import BaseParser
from ..parsers.cuda import CudaLogParser
from ..parsers.sycl import SyclLogParser
from .base import BaseResultsAnalyzer


class SyclResultsAnalyzer(BaseResultsAnalyzer):
    def __init__(self, parser: BaseParser | None = None) -> None:
        self.parser = parser or SyclLogParser()
        self.cuda_parser = CudaLogParser()

    def parse_log(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        if dataset.label.upper() == "CUDA":
            return self.cuda_parser.parse(path, dataset)
        return self.parser.parse(path, dataset)

    def has_decomposed_communication(self, summary: BenchmarkSummary) -> bool:
        return not (summary.dataset.upper() == "CUDA" and summary.comm == CommunicationBackend.NVSHMEM)
