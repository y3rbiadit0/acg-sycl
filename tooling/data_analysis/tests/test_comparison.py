from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tooling.data_analysis.comparison import parse_reference_report


class ReferenceReportTests(unittest.TestCase):
    def test_parse_reference_report_extracts_solver_rows(self) -> None:
        report = "\n".join(
            [
                "# Report",
                "",
                "## Solver And Communication Summary",
                "",
                "| Dataset | Scale | Comm | Ranks | Repeats | Solver s | Iters |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| CUDA | 1n1g | none | 1 | 3 | 38.981 | 25634 |",
                "| CUDA | bad | mpi | 4 | 3 | N/A | N/A |",
                "",
                "## Other",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text(report, encoding="utf-8")

            rows = parse_reference_report(path)

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "CUDA",
                    "scale": "1n1g",
                    "comm": "none",
                    "ranks": 1,
                    "solver": 38.981,
                    "iterations": 25634.0,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
