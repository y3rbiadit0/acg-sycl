from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..stats import SampleStats


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
