from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tooling.data_analysis.analyzers.sycl import SyclResultsAnalyzer
from tooling.data_analysis.enums import CommunicationBackend
from tooling.data_analysis.models import DatasetRoot


class MixedAnalyzerTests(unittest.TestCase):
    def test_sycl_analyzer_uses_cuda_parser_for_cuda_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "1n4g" / "acg-cg-mpi" / "suitesparse" / "Bump_2911"
            log_dir.mkdir(parents=True)
            stderr = log_dir / "acg_cuda_1n4g-1-1-stderr.txt"
            stderr.write_text(
                "4 MPI processes\n"
                "Using MPI for communication\n"
                "running solver: 12.5 seconds\n"
                "  total iterations: 25765\n"
                "  total solver time: 10.0 seconds\n"
                "  residual 2-norm: 4.343e+08\n"
                "  allreduce: 1.0 seconds/proc 10 times/proc 80 B/proc "
                "0.0 GB/s/proc 100.0 us/op/proc\n"
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:37.00\n",
                encoding="utf-8",
            )

            run = SyclResultsAnalyzer().parse_log(stderr, DatasetRoot(label="CUDA", path=root))

        self.assertEqual(run.dataset, "CUDA")
        self.assertEqual(run.comm, CommunicationBackend.MPI)
        self.assertEqual(run.ranks, 4)
        self.assertEqual(run.iterations, 25765)
        self.assertEqual(run.solver_time_s, 10.0)
        self.assertEqual(run.running_solver_s, 12.5)
        self.assertEqual(run.allreduce_calls, 10)


if __name__ == "__main__":
    unittest.main()
