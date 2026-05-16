#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)

cd "$project_root"
uv run --project "$project_root/tooling/data_analysis" python -m tooling.data_analysis.analyze_results "$@"

# Usage:
# tooling/scripts/analyze-results.sh   --dataset CUDA=cuda_results_summary   --output reports/cuda_results_summary.md