from __future__ import annotations

import statistics
from typing import Iterable

from pydantic import BaseModel, ConfigDict, computed_field


class SampleStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: tuple[float, ...] = ()

    @classmethod
    def from_values(cls, values: Iterable[float | int | None]) -> SampleStats:
        return cls(values=tuple(float(value) for value in values if value is not None))

    @computed_field
    @property
    def count(self) -> int:
        return len(self.values)

    @computed_field
    @property
    def median(self) -> float | None:
        if not self.values:
            return None
        return statistics.median(self.values)

    @computed_field
    @property
    def min(self) -> float | None:
        if not self.values:
            return None
        return min(self.values)

    @computed_field
    @property
    def max(self) -> float | None:
        if not self.values:
            return None
        return max(self.values)

    @computed_field
    @property
    def total(self) -> float | None:
        if not self.values:
            return None
        return sum(self.values)
