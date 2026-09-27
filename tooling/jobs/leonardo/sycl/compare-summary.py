#!/usr/bin/env python3
"""SYCL vs native aCG, paired within each allocation.

    tooling/jobs/leonardo/sycl/compare-summary.py [results-root]   (default $ACG_RESULTS_ROOT or results)

Reads the tree campaign-common.sh writes, <root>/<label>/suitesparse/<matrix>/,
and pairs each SYCL label with its native counterpart run in the same job:

    acg-sycl-mpi          vs  acg-cg-mpi       (MPI allreduce + MPI halo)
    acg-sycl-oneccl-nccl  vs  acg-cg-nccl      (NCCL allreduce; native halo is NCCL too)
    acg-sycl-single       vs  acg-cg-single    (one GPU)

Both times are the same quantity: the slowest rank's solver time after warmup,
from a barrier through the initial residual and the loop (SYCL `solver_max_s`,
native `total solver time`). ratio = SYCL / native, so 1.00 is parity and >1
means SYCL is slower. per-iter divides by each side's own iteration count;
iteration counts differ slightly because floating-point summation order does.

Per job: medians over the rounds, and the min..max of the per-round ratios.
Per cell: the median over jobs, which is the figure to report -- an allocation,
not a run, is the independent sample.
"""

from __future__ import annotations

import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAIRS = {
    "acg-sycl-mpi": "acg-cg-mpi",
    "acg-sycl-oneccl-nccl": "acg-cg-nccl",
    "acg-sycl-single": "acg-cg-single",
}
NAME = re.compile(r"-(\d{3})-nodes-(\d{4})-procs-(\d+)-(\d+)-(stdout|stderr)\.txt$")


def number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.M)
    return float(match.group(1).replace(",", "")) if match else None


def parse(path: Path, sycl: bool) -> tuple[float, int] | None:
    text = path.read_text(errors="replace")
    if sycl:
        seconds = number(r"^timing: .*?\bsolver_max_s=(\S+)", text)
        iterations = number(r"^solver: .*?\biterations=(\d+)", text)
    else:
        seconds = number(r"^\s*total solver time: ([\d.,]+) seconds", text)
        iterations = number(r"^\s*total iterations: ([\d,]+)", text)
    if seconds is None or iterations is None or iterations <= 0:
        return None
    return seconds, int(iterations)


def load(root: Path):
    # (matrix, topology, job) -> label -> round -> (seconds, iterations)
    runs: dict[tuple, dict[str, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    labels = set(PAIRS) | set(PAIRS.values())
    for label in labels:
        sycl = label in PAIRS
        stream = "stdout" if sycl else "stderr"
        for path in root.glob(f"{label}/suitesparse/*/*-{stream}.txt"):
            name = NAME.search(path.name)
            parsed = parse(path, sycl) if name else None
            if parsed is None:
                continue
            nodes, procs, job, trial = (int(g) for g in name.groups()[:4])
            runs[(path.parts[-2], f"{nodes}n{procs // nodes}g", job)][label][trial] = parsed
    return runs


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ACG_RESULTS_ROOT", "results"))
    runs = load(root)
    cells: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    print(f"{'matrix':<11}{'topo':<6}{'pair':<22}{'job':>9} n  {'sycl_s':>7} {'native_s':>8}  {'ratio':>5} "
          f"{'rounds':>11}  {'per-it':>6}  {'sycl_us':>8} {'native_us':>9}  iterations s/n")
    for key in sorted(runs):
        matrix, topology, job = key
        for sycl_label, native_label in PAIRS.items():
            sycl, native = runs[key].get(sycl_label), runs[key].get(native_label)
            if not sycl or not native:
                continue
            s_time = statistics.median(v[0] for v in sycl.values())
            n_time = statistics.median(v[0] for v in native.values())
            s_iter = statistics.median(v[0] / v[1] for v in sycl.values())
            n_iter = statistics.median(v[0] / v[1] for v in native.values())
            rounds = [sycl[t][0] / native[t][0] for t in sorted(set(sycl) & set(native))]
            span = f"{min(rounds):.2f}..{max(rounds):.2f}" if rounds else "-"
            its = f"{int(statistics.median(v[1] for v in sycl.values()))}/" \
                  f"{int(statistics.median(v[1] for v in native.values()))}"
            pair = sycl_label[len("acg-sycl-"):]
            print(f"{matrix:<11}{topology:<6}{pair:<22}{job:>9} {len(rounds)}  {s_time:7.2f} {n_time:8.2f}  "
                  f"{s_time / n_time:5.2f} {span:>11}  {s_iter / n_iter:6.2f}  {1e6 * s_iter:8.1f} "
                  f"{1e6 * n_iter:9.1f}  {its}")
            cells[(matrix, topology, pair)].append((s_time / n_time, s_iter / n_iter))
    if not cells:
        print(f"no paired SYCL/native results under {root}", file=sys.stderr)
        return 1
    print()
    print("per cell, median over allocations (SYCL / native; 1.00 = parity, >1 = SYCL slower):")
    for (matrix, topology, pair), values in sorted(cells.items()):
        ratios = [v[0] for v in values]
        per_iter = [v[1] for v in values]
        print(f"  {matrix:<11}{topology:<6}{pair:<14} allocations={len(values)}  "
              f"time {statistics.median(ratios):.2f} [{min(ratios):.2f}..{max(ratios):.2f}]  "
              f"per-iteration {statistics.median(per_iter):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
