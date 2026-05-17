from __future__ import annotations

from pathlib import Path


def parse_number(value: str) -> float | None:
    if value == "N/A":
        return None
    return float(value)


def parse_reference_report(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    in_solver_table = False
    headers: list[str] = []
    for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines():
        if line == "## Solver And Communication Summary":
            in_solver_table = True
            continue
        if in_solver_table and line.startswith("## "):
            break
        if not in_solver_table or not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == "Dataset":
            headers = cells
            continue
        if not headers or cells[0] == "---":
            continue

        item = dict(zip(headers, cells, strict=False))
        solver = parse_number(item["Solver s"])
        iterations = parse_number(item["Iters"])
        if solver is None:
            continue
        rows.append(
            {
                "dataset": item["Dataset"],
                "scale": item["Scale"],
                "comm": item["Comm"],
                "ranks": int(item["Ranks"]),
                "solver": solver,
                "iterations": iterations,
            }
        )
    return rows
