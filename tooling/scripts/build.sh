#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir=${BUILD_DIR:-$(default_build_dir)}
build_jobs=${BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}

build_args=(--build "$project_root/$build_dir")

if [[ -n "$build_jobs" ]]; then
  build_args+=(--parallel "$build_jobs")
fi

cmake "${build_args[@]}"
