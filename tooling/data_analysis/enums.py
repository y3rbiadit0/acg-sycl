from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    CUDA = "cuda"
    SYCL = "sycl"


class CommunicationBackend(str, Enum):
    NONE = "none"
    MPI = "mpi"
    NCCL = "nccl"
    NVSHMEM = "nvshmem"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: str | None) -> CommunicationBackend:
        if not value:
            return cls.UNKNOWN
        normalized = value.lower()
        if normalized in {"host", "none"}:
            return cls.NONE
        if normalized in {"gpu-aware", "gpu_aware", "gpuaware", "mpi"}:
            return cls.MPI
        for backend in cls:
            if backend.value == normalized:
                return backend
        return cls.UNKNOWN
