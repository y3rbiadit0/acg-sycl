#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_type=${BUILD_TYPE:-Release}
build_dir=${BUILD_DIR:-$(default_build_dir)}
device_debug=${DEVICE_DEBUG:-OFF}
sycl_target=${SYCL_TARGET_OVERRIDE:-${SYCL_TARGET:-}}
fresh=${FRESH_CONFIGURE:-ON}
extra_compile_flags=${ACG_EXTRA_COMPILE_FLAGS:-}
extra_link_flags=${ACG_EXTRA_LINK_FLAGS:-}

require_var CXX
require_var ACG_ONEMATH_ROOT

cmake_args=(
  -S "$project_root"
  -B "$project_root/$build_dir"
  -DCMAKE_BUILD_TYPE="$build_type"
  -DCMAKE_CXX_COMPILER="$CXX"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DoneMath_ROOT="$ACG_ONEMATH_ROOT"
  -DACG_ENABLE_DEVICE_DEBUG="$device_debug"
)

if [[ -n "$sycl_target" ]]; then
  cmake_args+=(-DACG_SYCL_TARGET="$sycl_target")
fi

if [[ -n "$extra_compile_flags" ]]; then
  cmake_args+=(-DACG_EXTRA_COMPILE_FLAGS="$extra_compile_flags")
fi

if [[ -n "$extra_link_flags" ]]; then
  cmake_args+=(-DACG_EXTRA_LINK_FLAGS="$extra_link_flags")
fi

if [[ "$fresh" == "ON" ]]; then
  cmake --fresh "${cmake_args[@]}"
else
  cmake "${cmake_args[@]}"
fi
