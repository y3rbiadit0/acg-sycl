from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...enums import CommunicationBackend
from .correctness import CorrectnessStats
from .operations import OperationStats
from .overheads import OverheadStats
from .partition import PartitionImbalanceStats
from .runtime import RuntimeStats
from .skew import SkewStats


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    scale: str
    variant: str
    comm: CommunicationBackend
    ranks: int | None
    repeats: int
    runtime: RuntimeStats
    correctness: CorrectnessStats
    operations: OperationStats
    overheads: OverheadStats
    skew: SkewStats
    partition: PartitionImbalanceStats
