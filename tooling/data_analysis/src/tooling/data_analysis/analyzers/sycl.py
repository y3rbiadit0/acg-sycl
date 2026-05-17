from __future__ import annotations

from pathlib import Path

from ..models import BenchmarkRun, DatasetRoot
from ..parsers.base import BaseParser
from ..parsers.sycl import SyclLogParser
from .base import BaseResultsAnalyzer


class SyclResultsAnalyzer(BaseResultsAnalyzer):
    def __init__(self, parser: BaseParser | None = None) -> None:
        self.parser = parser or SyclLogParser()

    def parse_log(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        return self.parser.parse(path, dataset)
