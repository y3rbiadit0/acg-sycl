#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir=${BUILD_DIR:-build-release}
matrix=${MATRIX:-data/matrices/Bump_2911/Bump_2911.mtx}
device=${DEVICE:-gpu}
mpi_np=${MPI_NP:-0}
residual_rtol=${RESIDUAL_RTOL:-1e-6}
args=${ARGS:-}

exe="$project_root/$build_dir/acg"
if [[ ! -x "$exe" ]]; then
  echo "error: executable not found: $exe" >&2
  exit 1
fi

cd "$project_root"

cmd=("$exe" --matrix "$matrix" --device "$device" --residual-rtol "$residual_rtol")

if [[ "$mpi_np" -gt 0 ]]; then
  mpirun_args=${ACG_MPIRUN_ARGS:-}
  # shellcheck disable=SC2206
  cmd=(mpirun -np "$mpi_np" ${mpirun_args:+$mpirun_args} "${cmd[@]}")
fi

if [[ -n "$args" ]]; then
  eval '"${cmd[@]}" '"$args"
else
  "${cmd[@]}"
fi
