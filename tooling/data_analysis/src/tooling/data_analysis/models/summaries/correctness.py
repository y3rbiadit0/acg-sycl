from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..stats import SampleStats


class CorrectnessStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    converged_repeats: int = 0
    residual_norm: SampleStats = Field(default_factory=SampleStats)
    rel_residual_r0: SampleStats = Field(default_factory=SampleStats)
    rel_residual_rhs: SampleStats = Field(default_factory=SampleStats)
    error_norm: SampleStats = Field(default_factory=SampleStats)
