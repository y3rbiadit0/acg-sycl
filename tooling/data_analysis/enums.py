from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    CUDA = "cuda"


class CommunicationBackend(str, Enum):
    MPI = "mpi"
    NCCL = "nccl"
    NVSHMEM = "nvshmem"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: str | None) -> CommunicationBackend:
        if not value:
            return cls.UNKNOWN
        normalized = value.lower()
        for backend in cls:
            if backend.value == normalized:
                return backend
        return cls.UNKNOWN
