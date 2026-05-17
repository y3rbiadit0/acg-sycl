from __future__ import annotations

import unittest
from pathlib import Path

from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import BenchmarkRun
from tooling.data_analysis.summarization import summarize_runs


class SummarizerTests(unittest.TestCase):
    def make_run(self, **values: object) -> BenchmarkRun:
        defaults: dict[str, object] = {
            "dataset": "SYCL",
            "file": Path("run-stderr.txt"),
            "scale": "1n4g",
            "variant": "acg-sycl-mpi",
            "matrix": "Bump_2911",
            "ranks": 4,
            "communicator": "gpu-aware",
        }
        defaults.update(values)
        return BenchmarkRun(**defaults)

    def test_summarize_runs_groups_and_calculates_stats(self) -> None:
        summaries = summarize_runs(
            [
                self.make_run(
                    converged=True,
                    solver_time_s=10.0,
                    running_solver_s=9.5,
                    wall_s=20.0,
                    iterations=100,
                    gflops=1.0,
                    residual_norm=1.0,
                    rel_residual_r0=1e-6,
                    rel_residual_rhs=1e-6,
                    error_norm=0.1,
                    allreduce_s=2.0,
                    allreduce_us=20.0,
                    allreduce_calls=200,
                    halo_s=0.4,
                    pack_s=0.1,
                    host_sync_s=0.2,
                    spmv_skew_s=0.5,
                    local_nnz_imbalance=1.1,
                    halo_nnz_imbalance=2.0,
                ),
                self.make_run(
                    converged=False,
                    solver_time_s=14.0,
                    running_solver_s=13.5,
                    wall_s=24.0,
                    iterations=140,
                    gflops=2.0,
                    residual_norm=3.0,
                    rel_residual_r0=3e-6,
                    rel_residual_rhs=3e-6,
                    error_norm=0.3,
                    allreduce_s=4.0,
                    allreduce_us=40.0,
                    allreduce_calls=400,
                    halo_s=0.8,
                    pack_s=0.3,
                    host_sync_s=0.4,
                    spmv_skew_s=0.7,
                    local_nnz_imbalance=1.3,
                    halo_nnz_imbalance=4.0,
                ),
            ]
        )

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.comm, CommunicationBackend.MPI)
        self.assertEqual(summary.repeats, 2)
        self.assertEqual(summary.runtime.solver_s.median, 12.0)
        self.assertEqual(summary.runtime.solver_s.min, 10.0)
        self.assertEqual(summary.runtime.solver_s.max, 14.0)
        self.assertEqual(summary.runtime.ms_per_iter, 100.0)
        self.assertEqual(summary.correctness.converged_repeats, 1)
        self.assertEqual(summary.correctness.residual_norm.median, 2.0)
        self.assertEqual(summary.correctness.rel_residual_r0.median, 2e-6)
        self.assertEqual(summary.correctness.error_norm.median, 0.2)
        self.assertEqual(summary.operations.allreduce_s.median, 3.0)
        self.assertEqual(summary.operations.allreduce_us.median, 30.0)
        self.assertEqual(summary.operations.allreduce_calls.median, 300.0)
        self.assertAlmostEqual(summary.operations.halo_s.median or 0, 0.6)
        self.assertAlmostEqual(summary.overheads.pack_s.median or 0, 0.2)
        self.assertAlmostEqual(summary.overheads.host_sync_s.median or 0, 0.3)
        self.assertAlmostEqual(summary.skew.spmv_s.median or 0, 0.6)
        self.assertAlmostEqual(summary.partition.local_nnz.median or 0, 1.2)
        self.assertEqual(summary.partition.halo_nnz.median, 3.0)

    def test_summarize_runs_keeps_distinct_groups(self) -> None:
        summaries = summarize_runs(
            [
                self.make_run(scale="1n4g", solver_time_s=1.0),
                self.make_run(scale="2n4g", ranks=8, solver_time_s=2.0),
                self.make_run(scale="1n4g", communicator="host", solver_time_s=3.0),
            ]
        )

        self.assertEqual(len(summaries), 3)
        keys = {(summary.scale, summary.comm, summary.ranks) for summary in summaries}
        self.assertEqual(
            keys,
            {
                ("1n4g", CommunicationBackend.MPI, 4),
                ("1n4g", CommunicationBackend.NONE, 4),
                ("2n4g", CommunicationBackend.MPI, 8),
            },
        )


if __name__ == "__main__":
    unittest.main()
