from __future__ import annotations

import re
from pathlib import Path

from tooling.data_analysis.models import BenchmarkRun, DatasetRoot
from tooling.data_analysis.parsers.base import BaseParser
from tooling.data_analysis.utils import parse_wall_clock


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


class CudaLogParser(BaseParser):
    def parse(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        values = self.metadata_from_path(path, dataset)
        pending_running_solver = False

        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                pending_running_solver = self.parse_line(
                    line.rstrip("\n"), values, pending_running_solver
                )

        return BenchmarkRun(**values)

    def parse_line(
        self, line: str, values: dict[str, object], pending_running_solver: bool
    ) -> bool:
        if match := re.match(r"^(\d+) MPI processes", line):
            values["ranks"] = int(match.group(1))
        elif match := re.match(r"^(\d+) OpenMP threads", line):
            values["omp_threads"] = int(match.group(1))
        elif match := re.match(r"^using (.+)$", line):
            values["solver_kind"] = match.group(1)
        elif match := re.match(r"^Using (.+?) for communication", line):
            values["communicator"] = match.group(1)
        elif match := re.match(r"^running solver:\s*(?:" + FLOAT + r" seconds)?", line):
            if match.group(1):
                values["running_solver_s"] = float(match.group(1))
                return False
            return True
        elif pending_running_solver and (match := re.match(r"^" + FLOAT + r" seconds$", line)):
            values["running_solver_s"] = float(match.group(1))
            return False
        elif match := re.match(r"^\s*total iterations: (\d+)", line):
            values["iterations"] = int(match.group(1))
        elif match := re.match(r"^\s*total flop rate: " + FLOAT + r" Gflop/s", line):
            values["gflops"] = float(match.group(1))
        elif match := re.match(r"^\s*total solver time: " + FLOAT + r" seconds", line):
            values["solver_time_s"] = float(match.group(1))
        elif match := re.match(
            r"^\s*allreduce: "
            + FLOAT
            + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
            + FLOAT
            + r" GB/s/proc "
            + FLOAT
            + r" us/op/proc",
            line,
        ):
            values["allreduce_s"] = float(match.group(1))
            values["allreduce_calls"] = int(match.group(2))
            values["allreduce_bytes"] = int(match.group(3))
            values["allreduce_us"] = float(match.group(5))
        elif match := re.match(
            r"^\s*haloexchange: "
            + FLOAT
            + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
            + FLOAT
            + r" GB/s/proc "
            + FLOAT
            + r" msg/proc "
            + FLOAT
            + r" us/msg/proc",
            line,
        ):
            values["halo_s"] = float(match.group(1))
            values["halo_calls"] = int(match.group(2))
            values["halo_bytes"] = int(match.group(3))
            values["halo_us"] = float(match.group(6))
        elif match := re.match(r"^\s*residual 2-norm: ([0-9.eE+-]+)", line):
            values["residual_norm"] = float(match.group(1))
        elif match := re.match(r"^error 2-norm: ([0-9.eE+-]+)", line):
            values["error_norm"] = float(match.group(1))
        elif match := re.match(
            r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)$", line
        ):
            values["wall_s"] = parse_wall_clock(match.group(1))
        elif match := re.match(r'^\s*Command being timed: "(.+)"', line):
            if comm := re.search(r"--comm\s+(\S+)", match.group(1)):
                values["command_communicator"] = comm.group(1)

        return pending_running_solver

    def metadata_from_path(self, path: Path, dataset: DatasetRoot) -> dict[str, object]:
        relative = path.relative_to(dataset.path)
        parts = relative.parts
        return {
            "dataset": dataset.label,
            "file": relative,
            "scale": parts[0] if len(parts) > 0 else "unknown",
            "variant": parts[1] if len(parts) > 1 else "unknown",
            "matrix": self.matrix_from_path(parts),
        }

    def matrix_from_path(self, parts: tuple[str, ...]) -> str:
        if "suitesparse" not in parts:
            return "unknown"
        idx = parts.index("suitesparse")
        if idx + 1 >= len(parts):
            return "unknown"
        return parts[idx + 1]
