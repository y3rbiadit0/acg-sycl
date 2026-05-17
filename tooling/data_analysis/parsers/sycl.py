from __future__ import annotations

import re
from pathlib import Path

from tooling.data_analysis.models import BenchmarkRun, DatasetRoot
from tooling.data_analysis.parsers.base import BaseParser
from tooling.data_analysis.utils import parse_wall_clock


FLOAT = r"([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)"


class SyclLogParser(BaseParser):
    def parse(self, path: Path, dataset: DatasetRoot) -> BenchmarkRun:
        values = self.metadata_from_path(path, dataset)
        self.parse_stderr(path, values)

        stdout = self.stdout_path(path)
        if stdout.is_file():
            self.parse_stdout(stdout, values)

        return BenchmarkRun(**values)

    def parse_stderr(self, path: Path, values: dict[str, object]) -> None:
        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                self.parse_stderr_line(line.rstrip("\n"), values)

    def parse_stderr_line(self, line: str, values: dict[str, object]) -> None:
        if match := re.match(r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)$", line):
            values["wall_s"] = parse_wall_clock(match.group(1))
        elif match := re.match(r'^\s*Command being timed: "(.+)"', line):
            if mpi_mode := re.search(r"--mpi-mode\s+(\S+)", match.group(1)):
                values["command_communicator"] = mpi_mode.group(1)

    def parse_stdout(self, path: Path, values: dict[str, object]) -> None:
        in_cuda_compatible = False
        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                line = line.rstrip("\n")
                if line == "performance_cuda_compatible:":
                    in_cuda_compatible = True
                    continue
                if re.match(r"^\S", line):
                    in_cuda_compatible = False
                self.parse_stdout_line(line, values, in_cuda_compatible)

    def parse_stdout_line(
        self, line: str, values: dict[str, object], in_cuda_compatible: bool
    ) -> None:
        if match := re.match(r"^ranks: (\d+)$", line):
            values["ranks"] = int(match.group(1))
        elif match := re.match(r"^mpi-mode: (\S+)$", line):
            values["communicator"] = match.group(1)
        elif match := re.match(r"^solver: .*$", line):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if iterations := fields.get("iterations"):
                values["iterations"] = int(iterations)
            if residual := fields.get("residual"):
                values["residual_norm"] = float(residual)
            if error := fields.get("relative_error"):
                values["error_norm"] = float(error)
            if gflops := fields.get("gflops"):
                values["gflops"] = float(gflops)
            if solve_time := fields.get("solve_time"):
                values["running_solver_s"] = float(solve_time.rstrip("s"))
                values.setdefault("solver_time_s", float(solve_time.rstrip("s")))
        elif in_cuda_compatible and (
            match := re.match(r"^\s*flop_rate: " + FLOAT + r" Gflop/s", line)
        ):
            values["gflops"] = float(match.group(1))
        elif in_cuda_compatible and (
            match := re.match(r"^\s*solver_time: " + FLOAT + r" seconds", line)
        ):
            values["solver_time_s"] = float(match.group(1))
        elif in_cuda_compatible and (
            match := re.match(
                r"^\s*allreduce: "
                + FLOAT
                + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
                + FLOAT
                + r" GB/s/proc "
                + FLOAT
                + r" us/op/proc",
                line,
            )
        ):
            values["allreduce_s"] = float(match.group(1))
            values["allreduce_calls"] = int(match.group(2))
            values["allreduce_bytes"] = int(match.group(3))
            values["allreduce_us"] = float(match.group(5))
        elif in_cuda_compatible and (
            match := re.match(
                r"^\s*haloexchange: "
                + FLOAT
                + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
                + FLOAT
                + r" GB/s/proc",
                line,
            )
        ):
            values["halo_s"] = float(match.group(1))
            values["halo_calls"] = int(match.group(2))
            values["halo_bytes"] = int(match.group(3))

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

    def stdout_path(self, path: Path) -> Path:
        return path.with_name(path.name.replace("-stderr.txt", "-stdout.txt"))
