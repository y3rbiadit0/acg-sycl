from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tooling.data_analysis.models.stats import SampleStats


class OverheadStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack_s: SampleStats = Field(default_factory=SampleStats)
    p2p_s: SampleStats = Field(default_factory=SampleStats)
    host_sync_s: SampleStats = Field(default_factory=SampleStats)
    other_s: SampleStats = Field(default_factory=SampleStats)
