from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tooling.data_analysis.models.stats import SampleStats


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
