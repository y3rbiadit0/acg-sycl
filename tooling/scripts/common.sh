#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)

env_name=${ENV:-local}
env_file="$project_root/tooling/environments/${env_name}.sh"

if [[ ! -f "$env_file" ]]; then
  echo "error: unknown environment '$env_name' (expected $env_file)" >&2
  exit 1
fi

set +u
source "$env_file"
set -u

require_var() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "error: required variable '$name' is not set" >&2
    exit 1
  fi
}

default_build_dir() {
  case "${BUILD_TYPE:-Release}" in
    Debug) printf '%s' build ;;
    Release) printf '%s' build-release ;;
    *) printf '%s' build-${BUILD_TYPE,,} ;;
  esac
}

print_env_summary() {
  echo "ENV       : $env_name"
  echo "Compiler  : ${CXX:-unset}"
  echo "CUDA      : ${CUDA_ROOT:-unset}"
  echo "oneMath   : ${ACG_ONEMATH_ROOT:-unset}"
  echo "SYCL tgt  : ${SYCL_TARGET:-default}"
}
