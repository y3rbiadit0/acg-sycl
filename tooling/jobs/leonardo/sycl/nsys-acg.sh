#!/bin/bash
# Nsight Systems wrapper for short profiling runs: use it as the binary.
#
#   ACG_NSYS_TARGET=$PWD/build-release/acg \
#   ACG_SYCL_BINARY=$PWD/tooling/jobs/leonardo/sycl/nsys-acg.sh \
#   ACG_EXTRA_ARGS="--max-iters 200" ACG_SOLVERS=mpi ACG_NTRIALS=1 \
#   ACG_RESULTS_ROOT=$PWD/results/nsys make now-2n4g SOLVERS=mpi QOS=boost_qos_dbg
#
# One report per rank, named by job and rank, in ACG_NSYS_DIR. Capped runs exit
# nonconverged on purpose and stay in .tmp; keep them out of timing results.
set -euo pipefail
: "${ACG_NSYS_TARGET:?set ACG_NSYS_TARGET to the acg binary to profile}"
out=${ACG_NSYS_DIR:-${ACG_RESULTS_ROOT:-results}/nsys}
mkdir -p "$out"
exec nsys profile --trace=cuda,nvtx,mpi,osrt --mpi-impl=openmpi --force-overwrite=true \
    -o "$out/acg-${SLURM_JOB_ID:-0}-rank${SLURM_PROCID:-0}" \
    "$ACG_NSYS_TARGET" "$@"
