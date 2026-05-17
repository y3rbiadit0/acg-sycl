from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DatasetRoot(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    path: Path
