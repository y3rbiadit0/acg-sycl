from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..stats import SampleStats


class SkewStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    spmv_s: SampleStats = Field(default_factory=SampleStats)
    allreduce_s: SampleStats = Field(default_factory=SampleStats)
    p2p_s: SampleStats = Field(default_factory=SampleStats)
    halo_spmv_s: SampleStats = Field(default_factory=SampleStats)
    pack_s: SampleStats = Field(default_factory=SampleStats)
    host_sync_s: SampleStats = Field(default_factory=SampleStats)
