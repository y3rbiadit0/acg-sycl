from __future__ import annotations

import re
from pathlib import Path

from ..models import BenchmarkRun, DatasetRoot
from ..utils import parse_wall_clock
from .base import BaseParser


FLOAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"


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
        section: str | None = None
        partitions: list[dict[str, int]] = []
        with path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                line = line.rstrip("\n")
                if line == "performance_cuda_compatible:":
                    section = "cuda"
                    continue
                if line == "performance_sycl_overheads:":
                    section = "overheads"
                    continue
                if line == "performance_sycl_native:":
                    section = "native"
                    continue
                if re.match(r"^\S", line):
                    section = None
                self.parse_stdout_line(line, values, section, partitions)

        self.add_partition_imbalance(values, partitions)
        # Current logs name only the collective; one rank has no communicator.
        if values.get("ranks") == 1 and not values.get("command_communicator"):
            values["communicator"] = "none"

    def parse_stdout_line(
        self,
        line: str,
        values: dict[str, object],
        section: str | None,
        partitions: list[dict[str, int]],
    ) -> None:
        if match := re.match(r"^ranks: (\d+)$", line):
            values["ranks"] = int(match.group(1))
        elif match := re.match(r"^mpi-mode: (\S+)$", line):
            values["communicator"] = match.group(1)
        elif match := re.match(r"^solver-collectives: (\S+)$", line):
            values.setdefault("communicator", match.group(1))
        elif match := re.match(r"^optimizations: (\S+)$", line):
            values["optimizations"] = match.group(1)
        elif line.startswith("timing: "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            schema = int(fields.get("schema", "0"))
            values["timing_schema"] = schema
            # Schema 3 (current): solver_* is native aCG's "total solver time"
            # -- after warmup, from a barrier through the initial residual and
            # the loop; solver_max_s is the slowest rank. Schema 2 used loop_*.
            renames = {"solver_s": "loop_s", "solver_max_s": "loop_max_s", "solver_min_s": "loop_min_s"}
            for key, value in fields.items():
                field = renames.get(key, key)
                if field in ("setup_s", "warmup_s", "loop_s", "loop_max_s", "loop_min_s", "validation_s"):
                    values[field] = float(value)
            if schema >= 3 and "loop_max_s" in values:
                values["solver_time_s"] = values["loop_max_s"]
        elif line.startswith("waits: "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            for key, field in (("pack_s", "pack_s"), ("halo_s", "p2p_s"), ("allreduce_s", "allreduce_s"),
                               ("readback_s", "host_sync_s")):
                if key in fields:
                    values[field] = float(fields[key])
        elif line.startswith("validation: "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if true_rel := fields.get("true_rel_residual"):
                values["true_rel_residual"] = float(true_rel)
        elif match := re.match(r"^solver: .*$", line):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if converged := fields.get("converged"):
                values["converged"] = converged.lower() == "true"
            if iterations := fields.get("iterations"):
                values["iterations"] = int(iterations)
            if residual := fields.get("residual"):
                values["residual_norm"] = float(residual)
            if rel_residual_r0 := fields.get("rel_residual_r0"):
                values["rel_residual_r0"] = float(rel_residual_r0)
            if rel_residual_rhs := fields.get("rel_residual_rhs"):
                values["rel_residual_rhs"] = float(rel_residual_rhs)
            if error := fields.get("relative_error"):
                values["error_norm"] = float(error)
            if gflops := fields.get("gflops"):
                values["gflops"] = float(gflops)
            if solve_time := fields.get("solve_time"):
                values["running_solver_s"] = float(solve_time.rstrip("s"))
                values.setdefault("solver_time_s", float(solve_time.rstrip("s")))
        elif section == "cuda" and (
            match := re.match(r"^\s*flop_rate: " + FLOAT + r" Gflop/s", line)
        ):
            values["gflops"] = float(match.group(1))
        elif section == "cuda" and (
            match := re.match(r"^\s*solver_time: " + FLOAT + r" seconds", line)
        ):
            values["solver_time_s"] = float(match.group(1))
        elif section == "cuda" and (
            match := re.match(
                r"^\s*(gemv|dot|nrm2|axpy|copy): "
                + FLOAT
                + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
                + FLOAT
                + r" GB/s/proc",
                line,
            )
        ):
            values[f"{match.group(1)}_s"] = float(match.group(2))
        elif section == "cuda" and (
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
        elif section == "cuda" and (
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
        elif section == "overheads" and (
            match := re.match(
                r"^\s*(pack|host_sync): "
                + FLOAT
                + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
                + FLOAT
                + r" GB/s/proc",
                line,
            )
        ):
            values[f"{match.group(1)}_s"] = float(match.group(2))
        elif section == "overheads" and (
            match := re.match(r"^\s*other: " + FLOAT + r" seconds", line)
        ):
            values["other_s"] = float(match.group(1))
        elif section == "native" and (
            match := re.match(
                r"^\s*(spmv|p2p): "
                + FLOAT
                + r" seconds/proc (\d+) times/proc (\d+) B/proc\s+"
                + FLOAT
                + r" GB/s/proc",
                line,
            )
        ):
            values[f"{match.group(1)}_s"] = float(match.group(2))
        elif match := re.match(
            r"^solver_diag_perf: (\S+) min_s="
            + FLOAT
            + r" avg_s="
            + FLOAT
            + r" max_s="
            + FLOAT
            + r" skew_s="
            + FLOAT,
            line,
        ):
            field = {
                "spmv": "spmv_skew_s",
                "allreduce": "allreduce_skew_s",
                "p2p": "p2p_skew_s",
                "halo_spmv": "halo_spmv_skew_s",
                "pack": "pack_skew_s",
                "host_sync": "host_sync_skew_s",
            }.get(match.group(1))
            if field:
                values[field] = float(match.group(5))
        elif match := re.match(
            r"^solver_diag_partition: (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)$",
            line,
        ):
            partitions.append(
                {
                    "local_rows": int(match.group(2)),
                    "local_nnz": int(match.group(3)),
                    "interior_nnz": int(match.group(4)),
                    "halo_nnz": int(match.group(5)),
                    "ghosts": int(match.group(6)),
                    "imports": int(match.group(7)),
                    "exports": int(match.group(8)),
                }
            )

    def add_partition_imbalance(
        self, values: dict[str, object], partitions: list[dict[str, int]]
    ) -> None:
        if not partitions:
            return
        for key, field in [
            ("local_nnz", "local_nnz_imbalance"),
            ("halo_nnz", "halo_nnz_imbalance"),
            ("ghosts", "ghosts_imbalance"),
            ("imports", "imports_imbalance"),
            ("exports", "exports_imbalance"),
        ]:
            vals = [partition[key] for partition in partitions]
            if min(vals) > 0:
                values[field] = max(vals) / min(vals)

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
