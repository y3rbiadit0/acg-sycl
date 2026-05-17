from __future__ import annotations

from pathlib import Path

from tooling.data_analysis.analyzers.base import BaseResultsAnalyzer
from tooling.data_analysis.models import BenchmarkRun, DatasetRoot
from tooling.data_analysis.parsers.base import BaseParser
from tooling.data_analysis.parsers.sycl import SyclLogParser


class SyclResultsAnalyzer(BaseResultsAnalyzer):
    def __init__(self, parser: BaseParser | None = None) -> None:
        self.parser = parser or SyclLogParser()

    def parse_log(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        return self.parser.parse(path, dataset)
