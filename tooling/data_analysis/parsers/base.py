from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tooling.data_analysis.models import BenchmarkRun, DatasetRoot


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        raise NotImplementedError
