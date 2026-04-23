#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir=${BUILD_DIR:-build-release}
matrix=${MATRIX:-data/matrices/Bump_2911/Bump_2911.mtx}
device=${DEVICE:-gpu}
residual_rtol=${RESIDUAL_RTOL:-1e-6}
args=${ARGS:-}

exe="$project_root/$build_dir/acg"
if [[ ! -x "$exe" ]]; then
  echo "error: executable not found: $exe" >&2
  exit 1
fi

cd "$project_root"

if [[ -n "$args" ]]; then
  eval '"$exe" --matrix "$matrix" --device "$device" --residual-rtol "$residual_rtol" '"$args"
else
  "$exe" --matrix "$matrix" --device "$device" --residual-rtol "$residual_rtol"
fi
