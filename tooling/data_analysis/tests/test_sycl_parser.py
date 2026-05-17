from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import DatasetRoot
from tooling.data_analysis.parsers.sycl import SyclLogParser


class SyclLogParserTests(unittest.TestCase):
    def test_parse_stdout_stderr_pair_extracts_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "1n4g" / "acg-sycl-mpi" / "suitesparse" / "Bump_2911"
            log_dir.mkdir(parents=True)
            stderr = log_dir / "acg_sycl_1n4g-1-1-stderr.txt"
            stdout = log_dir / "acg_sycl_1n4g-1-1-stdout.txt"
            stderr.write_text(
                'Command being timed: "acg --mpi-mode gpu-aware"\n'
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.50\n",
                encoding="utf-8",
            )
            stdout.write_text(
                "\n".join(
                    [
                        "device-kind: gpu",
                        "ranks: 4",
                        "mpi-mode: gpu-aware",
                        "solver_diag_perf: spmv min_s=1 avg_s=2 max_s=3 skew_s=2",
                        "solver_diag_perf: allreduce min_s=0.1 avg_s=0.2 max_s=0.3 skew_s=0.2",
                        "solver_diag_perf: p2p min_s=0.4 avg_s=0.5 max_s=0.9 skew_s=0.5",
                        "solver_diag_perf: halo_spmv min_s=0.2 avg_s=0.3 max_s=0.5 skew_s=0.3",
                        "solver_diag_perf: pack min_s=0.1 avg_s=0.2 max_s=0.4 skew_s=0.3",
                        "solver_diag_perf: host_sync min_s=0.2 avg_s=0.3 max_s=0.6 skew_s=0.4",
                        "solver_diag_partition: rank local_rows local_nnz interior_nnz "
                        "halo_nnz ghosts imports exports",
                        "solver_diag_partition: 0 10 100 90 10 5 5 4",
                        "solver_diag_partition: 1 10 200 170 30 10 15 8",
                        "solver: converged=true iterations=123 initial_residual=1e+4 "
                        "residual=2.5e-7 "
                        "rel_residual_r0=3.5e-8 rel_residual_rhs=4.5e-8 rhs_norm=9 "
                        "relative_error=0.125 flops=1e+9 gflops=42.5 solve_time=7.5s "
                        "total_time=7.6s",
                        "performance_cuda_compatible:",
                        "   total_flops: 1.0 Gflop",
                        "   flop_rate: 50.0 Gflop/s",
                        "   solver_time: 8.0 seconds",
                        "   gemv: 4.0 seconds/proc 123 times/proc 1000 B/proc 10.0 GB/s/proc",
                        "   dot: 0.5 seconds/proc 123 times/proc 100 B/proc 1.0 GB/s/proc",
                        "   nrm2: 0.25 seconds/proc 124 times/proc 100 B/proc 1.0 GB/s/proc",
                        "   axpy: 0.75 seconds/proc 246 times/proc 100 B/proc 1.0 GB/s/proc",
                        "   copy: 0.125 seconds/proc 1 times/proc 100 B/proc 1.0 GB/s/proc",
                        "   allreduce: 1.25 seconds/proc 246 times/proc 1968 B/proc "
                        "0.0 GB/s/proc 5.081 us/op/proc",
                        "   haloexchange: 0.625 seconds/proc 123 times/proc 2048 B/proc "
                        "3.0 GB/s/proc",
                        "performance_sycl_overheads:",
                        "   pack: 0.375 seconds/proc 123 times/proc 1024 B/proc 2.0 GB/s/proc",
                        "   host_sync: 0.5 seconds/proc 492 times/proc 1968 B/proc 0.0 GB/s/proc",
                        "   other: 0.875 seconds",
                        "performance_sycl_native:",
                        "   spmv: 4.25 seconds/proc 246 times/proc 4096 B/proc 4.0 GB/s/proc",
                        "   p2p: 0.75 seconds/proc 123 times/proc 2048 B/proc 2.7 GB/s/proc",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run = SyclLogParser().parse(stderr, DatasetRoot(label="SYCL", path=root))

        self.assertEqual(run.dataset, "SYCL")
        self.assertEqual(run.scale, "1n4g")
        self.assertEqual(run.variant, "acg-sycl-mpi")
        self.assertEqual(run.matrix, "Bump_2911")
        self.assertEqual(run.ranks, 4)
        self.assertEqual(run.comm, CommunicationBackend.MPI)
        self.assertEqual(run.wall_s, 12.5)
        self.assertTrue(run.converged)
        self.assertEqual(run.iterations, 123)
        self.assertEqual(run.residual_norm, 2.5e-7)
        self.assertEqual(run.rel_residual_r0, 3.5e-8)
        self.assertEqual(run.rel_residual_rhs, 4.5e-8)
        self.assertEqual(run.error_norm, 0.125)
        self.assertEqual(run.running_solver_s, 7.5)
        self.assertEqual(run.solver_time_s, 8.0)
        self.assertEqual(run.gflops, 50.0)
        self.assertEqual(run.gemv_s, 4.0)
        self.assertEqual(run.dot_s, 0.5)
        self.assertEqual(run.nrm2_s, 0.25)
        self.assertEqual(run.axpy_s, 0.75)
        self.assertEqual(run.copy_s, 0.125)
        self.assertEqual(run.allreduce_s, 1.25)
        self.assertEqual(run.allreduce_calls, 246)
        self.assertEqual(run.allreduce_bytes, 1968)
        self.assertEqual(run.allreduce_us, 5.081)
        self.assertEqual(run.halo_s, 0.625)
        self.assertEqual(run.halo_calls, 123)
        self.assertEqual(run.halo_bytes, 2048)
        self.assertEqual(run.pack_s, 0.375)
        self.assertEqual(run.host_sync_s, 0.5)
        self.assertEqual(run.other_s, 0.875)
        self.assertEqual(run.spmv_s, 4.25)
        self.assertEqual(run.p2p_s, 0.75)
        self.assertEqual(run.spmv_skew_s, 2.0)
        self.assertEqual(run.allreduce_skew_s, 0.2)
        self.assertEqual(run.p2p_skew_s, 0.5)
        self.assertEqual(run.halo_spmv_skew_s, 0.3)
        self.assertEqual(run.pack_skew_s, 0.3)
        self.assertEqual(run.host_sync_skew_s, 0.4)
        self.assertEqual(run.local_nnz_imbalance, 2.0)
        self.assertEqual(run.halo_nnz_imbalance, 3.0)
        self.assertEqual(run.ghosts_imbalance, 2.0)
        self.assertEqual(run.imports_imbalance, 3.0)
        self.assertEqual(run.exports_imbalance, 2.0)


if __name__ == "__main__":
    unittest.main()
