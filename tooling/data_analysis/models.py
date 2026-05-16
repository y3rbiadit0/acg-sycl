from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from tooling.data_analysis.enums import CommunicationBackend


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    file: Path
    scale: str
    variant: str
    matrix: str
    ranks: int | None = None
    omp_threads: int | None = None
    solver_kind: str | None = None
    communicator: str | None = None
    command_communicator: str | None = None
    running_solver_s: float | None = None
    solver_time_s: float | None = None
    allreduce_s: float | None = None
    allreduce_calls: int | None = None
    allreduce_bytes: int | None = None
    allreduce_us: float | None = None
    halo_s: float | None = None
    halo_calls: int | None = None
    halo_bytes: int | None = None
    halo_us: float | None = None
    iterations: int | None = None
    gflops: float | None = None
    residual_norm: float | None = None
    error_norm: float | None = None
    wall_s: float | None = None

    @computed_field
    @property
    def comm(self) -> CommunicationBackend:
        if self.command_communicator:
            return CommunicationBackend.from_value(self.command_communicator)
        if self.communicator:
            return CommunicationBackend.from_value(self.communicator)
        return CommunicationBackend.UNKNOWN


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    scale: str
    variant: str
    comm: CommunicationBackend
    ranks: int | None
    repeats: int
    solver_median: float | None
    solver_min: float | None
    solver_max: float | None
    running_median: float | None
    allreduce_median: float | None
    allreduce_us_median: float | None
    allreduce_calls_median: float | None
    halo_median: float | None
    halo_us_median: float | None
    iterations_median: float | None
    gflops_median: float | None
    wall_median: float | None
    residual_median: float | None
    error_median: float | None


class DatasetRoot(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    path: Path
