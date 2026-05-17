from __future__ import annotations

import argparse
from pathlib import Path

from .analyzers.base import BaseResultsAnalyzer
from .analyzers.cuda import CudaResultsAnalyzer
from .analyzers.sycl import SyclResultsAnalyzer
from .enums import Backend
from .models import DatasetRoot
from .parsing import parse_reference_report


DESCRIPTION = "Summarize aCG benchmark stderr logs into deterministic Markdown reports."


def parse_dataset(value: str) -> DatasetRoot:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("dataset label must not be empty")
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"dataset path is not a directory: {root}")
    return DatasetRoot(label=label, path=root)


def create_analyzer(backend: str) -> BaseResultsAnalyzer:
    if Backend(backend) == Backend.CUDA:
        return CudaResultsAnalyzer()
    if Backend(backend) == Backend.SYCL:
        return SyclResultsAnalyzer()
    raise ValueError(f"unsupported backend: {backend}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in Backend],
        default=Backend.CUDA.value,
        help="result backend to parse",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        metavar="LABEL=PATH",
        help="result root to parse; can be provided multiple times, e.g. CUDA=cuda_results_summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/acg_results_summary.md"),
        help="Markdown report path",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="existing Markdown summary to compare against, e.g. reports/cuda_results_summary.md",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()

    analyzer = create_analyzer(args.backend)
    datasets: list[DatasetRoot] = args.dataset
    input_roots = {dataset.label: dataset.path for dataset in datasets}
    runs = analyzer.collect_runs(datasets)

    if not runs:
        raise SystemExit("no stderr logs found below the requested dataset roots")

    summaries = analyzer.summarize(runs)
    reference_summaries = (
        parse_reference_report(args.reference_report) if args.reference_report else []
    )
    report = analyzer.render_report(summaries, runs, input_roots, reference_summaries)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"wrote {output}")
    return 0
