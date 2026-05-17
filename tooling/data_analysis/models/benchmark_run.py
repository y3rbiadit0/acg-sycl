from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from tooling.data_analysis.enums import CommunicationBackend


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    file: Path
    scale: str
    variant: str
    matrix: str
    ranks: int | None = None
    omp_threads: int | None = None
    solver_kind: str | None = None
    communicator: str | None = None
    command_communicator: str | None = None
    converged: bool | None = None
    running_solver_s: float | None = None
    solver_time_s: float | None = None
    gemv_s: float | None = None
    spmv_s: float | None = None
    dot_s: float | None = None
    nrm2_s: float | None = None
    axpy_s: float | None = None
    copy_s: float | None = None
    allreduce_s: float | None = None
    allreduce_calls: int | None = None
    allreduce_bytes: int | None = None
    allreduce_us: float | None = None
    halo_s: float | None = None
    halo_calls: int | None = None
    halo_bytes: int | None = None
    halo_us: float | None = None
    pack_s: float | None = None
    p2p_s: float | None = None
    host_sync_s: float | None = None
    other_s: float | None = None
    iterations: int | None = None
    gflops: float | None = None
    residual_norm: float | None = None
    rel_residual_r0: float | None = None
    rel_residual_rhs: float | None = None
    error_norm: float | None = None
    wall_s: float | None = None
    spmv_skew_s: float | None = None
    allreduce_skew_s: float | None = None
    p2p_skew_s: float | None = None
    halo_spmv_skew_s: float | None = None
    pack_skew_s: float | None = None
    host_sync_skew_s: float | None = None
    local_nnz_imbalance: float | None = None
    halo_nnz_imbalance: float | None = None
    ghosts_imbalance: float | None = None
    imports_imbalance: float | None = None
    exports_imbalance: float | None = None

    @computed_field
    @property
    def comm(self) -> CommunicationBackend:
        if self.command_communicator:
            return CommunicationBackend.from_value(self.command_communicator)
        if self.communicator:
            return CommunicationBackend.from_value(self.communicator)
        return CommunicationBackend.UNKNOWN
