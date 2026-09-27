#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_type=${BUILD_TYPE:-Release}
build_dir=${BUILD_DIR:-$(default_build_dir)}
device_debug=${DEVICE_DEBUG:-OFF}
enable_oneccl=${ACG_ENABLE_ONECCL:-OFF}
enable_metis=${ACG_ENABLE_METIS:-OFF}
sycl_target=${SYCL_TARGET_OVERRIDE:-${SYCL_TARGET:-}}
fresh=${FRESH_CONFIGURE:-ON}
extra_compile_flags=${ACG_EXTRA_COMPILE_FLAGS:-}
extra_link_flags=${ACG_EXTRA_LINK_FLAGS:-}
metis_home=${METIS_HOME:-}
oneccl_root=${oneCCL_ROOT:-${ONECCL_ROOT:-${CCL_ROOT:-}}}

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
  -DACG_ENABLE_ONECCL="$enable_oneccl"
  -DACG_ENABLE_METIS="$enable_metis"
)

if [[ -n "$oneccl_root" ]]; then
  cmake_args+=(-DoneCCL_ROOT="$oneccl_root")
fi

# Pin the MPI the environment selected; see ACG_MPI in tooling/environments.
if [[ -n "${MPI_CXX_COMPILER:-}" ]]; then
  cmake_args+=(-DMPI_CXX_COMPILER="$MPI_CXX_COMPILER")
fi

if [[ -n "$metis_home" ]]; then
  cmake_args+=(-DMETIS_HOME="$metis_home")
fi

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
