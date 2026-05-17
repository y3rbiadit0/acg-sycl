from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.stats import SampleStats


class RuntimeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    solver_s: SampleStats = Field(default_factory=SampleStats)
    running_solver_s: SampleStats = Field(default_factory=SampleStats)
    wall_s: SampleStats = Field(default_factory=SampleStats)
    iterations: SampleStats = Field(default_factory=SampleStats)
    gflops: SampleStats = Field(default_factory=SampleStats)

    @computed_field
    @property
    def ms_per_iter(self) -> float | None:
        if not self.solver_s.median or not self.iterations.median:
            return None
        return 1000 * self.solver_s.median / self.iterations.median


class CorrectnessStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    converged_repeats: int = 0
    residual_norm: SampleStats = Field(default_factory=SampleStats)
    rel_residual_r0: SampleStats = Field(default_factory=SampleStats)
    rel_residual_rhs: SampleStats = Field(default_factory=SampleStats)
    error_norm: SampleStats = Field(default_factory=SampleStats)


class OperationStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    gemv_s: SampleStats = Field(default_factory=SampleStats)
    spmv_s: SampleStats = Field(default_factory=SampleStats)
    dot_s: SampleStats = Field(default_factory=SampleStats)
    nrm2_s: SampleStats = Field(default_factory=SampleStats)
    axpy_s: SampleStats = Field(default_factory=SampleStats)
    copy_s: SampleStats = Field(default_factory=SampleStats)
    allreduce_s: SampleStats = Field(default_factory=SampleStats)
    allreduce_us: SampleStats = Field(default_factory=SampleStats)
    allreduce_calls: SampleStats = Field(default_factory=SampleStats)
    halo_s: SampleStats = Field(default_factory=SampleStats)
    halo_us: SampleStats = Field(default_factory=SampleStats)


class OverheadStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack_s: SampleStats = Field(default_factory=SampleStats)
    p2p_s: SampleStats = Field(default_factory=SampleStats)
    host_sync_s: SampleStats = Field(default_factory=SampleStats)
    other_s: SampleStats = Field(default_factory=SampleStats)


class SkewStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    spmv_s: SampleStats = Field(default_factory=SampleStats)
    allreduce_s: SampleStats = Field(default_factory=SampleStats)
    p2p_s: SampleStats = Field(default_factory=SampleStats)
    halo_spmv_s: SampleStats = Field(default_factory=SampleStats)
    pack_s: SampleStats = Field(default_factory=SampleStats)
    host_sync_s: SampleStats = Field(default_factory=SampleStats)


class PartitionImbalanceStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    local_nnz: SampleStats = Field(default_factory=SampleStats)
    halo_nnz: SampleStats = Field(default_factory=SampleStats)
    ghosts: SampleStats = Field(default_factory=SampleStats)
    imports: SampleStats = Field(default_factory=SampleStats)
    exports: SampleStats = Field(default_factory=SampleStats)


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
