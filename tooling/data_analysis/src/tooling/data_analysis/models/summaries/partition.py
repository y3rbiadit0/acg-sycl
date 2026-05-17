from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..stats import SampleStats


class PartitionImbalanceStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    local_nnz: SampleStats = Field(default_factory=SampleStats)
    halo_nnz: SampleStats = Field(default_factory=SampleStats)
    ghosts: SampleStats = Field(default_factory=SampleStats)
    imports: SampleStats = Field(default_factory=SampleStats)
    exports: SampleStats = Field(default_factory=SampleStats)
