#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir=${BUILD_DIR:-build-release}
matrix=${MATRIX:-data/matrices/Bump_2911/Bump_2911.mtx}
device=${DEVICE:-gpu}
args=${ARGS:-}

exe="$project_root/$build_dir/acg"
if [[ ! -x "$exe" ]]; then
  echo "error: executable not found: $exe" >&2
  exit 1
fi

cd "$project_root"

if [[ -n "$args" ]]; then
  eval '"$exe" --matrix "$matrix" --device "$device" '"$args"
else
  "$exe" --matrix "$matrix" --device "$device"
fi
