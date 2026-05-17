from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models.summaries.correctness import CorrectnessStats
from tooling.data_analysis.models.summaries.operations import OperationStats
from tooling.data_analysis.models.summaries.overheads import OverheadStats
from tooling.data_analysis.models.summaries.partition import PartitionImbalanceStats
from tooling.data_analysis.models.summaries.runtime import RuntimeStats
from tooling.data_analysis.models.summaries.skew import SkewStats


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
