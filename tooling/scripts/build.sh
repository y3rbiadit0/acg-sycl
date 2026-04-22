#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir=${BUILD_DIR:-$(default_build_dir)}

cmake --build "$project_root/$build_dir"
