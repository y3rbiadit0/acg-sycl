#!/bin/bash
#
# The campaign body: one allocation, every backend key, round by round.
#
# Sourced by job.sh with ACG_JOB_NODES / ACG_NTASKS set. Each round runs every
# key in ACG_SOLVERS once, and the order reverses from one round to the next
# (A B C, C B A, ...), so SYCL and native aCG are compared on the same nodes,
# the same fabric neighbours and the same clocks, with neither always getting
# the warmer slot. Across allocations, compare per-job medians
# (compare-summary.py), not individual runs.
#
# Method, identical for SYCL and native (the native campaign's own settings):
#   --manufactured-solution --seed 101 --residual-atol 0 --residual-rtol 1e-6
#   --max-iters 100000 --warmup 10, the same staged partition files, the same
#   matrix file, the same MPI (HPC-X) and the same transport settings.
#
# Backend keys:
#   mpi, oneccl        SYCL solver; the CG scalar allreduces through CUDA-aware
#                      MPI or oneCCL/NCCL (the halo exchange is MPI in both)
#   none               SYCL on one GPU
#   native-mpi         native aCG, --solver acg --comm mpi
#   native-nccl        native aCG, --solver acg --comm nccl (halo over NCCL too)
#   native-none        native aCG on one GPU
#
# Variables:
#   ACG_SOLVERS        keys to run (job.sh defaults them from the topology)
#   ACG_NTRIALS        rounds (default 3)
#   ACG_MATRIX_NAME    Bump_2911 (default); ACG_MATRIX_DIR
#   ACG_PARTITION_DIR  where <matrix>_<NN>_parts.mtx lives; ACG_PARTITION
#   ACG_MAX_ITERATIONS, ACG_RESIDUAL_RTOL, ACG_WARMUP, ACG_EXTRA_ARGS (SYCL only)
#   ACG_SYCL_BINARY    default build-release/acg
#   ACG_NATIVE_ROOT    native aCG checkout (default $HOME/Projects/aCG-oshmpi)
#   ACG_NATIVE_BINARY  default $ACG_NATIVE_ROOT/build-oshmpi/acg-cuda
#   ACG_RESULTS_ROOT   default results
#   ACG_PROVENANCE     1 (default) writes <results>/provenance/<jobid>/
#   ACG_CHECKSUM_MATRIX 1 also hashes the matrix file (slow on multi-GB files)

set -euo pipefail

: "${ACG_JOB_NODES:?missing ACG_JOB_NODES}"
: "${ACG_NTASKS:?missing ACG_NTASKS}"

mkdir -p ./logs

export LC_ALL=C
project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# launch.sh submits from a snapshot of these scripts taken at submission time
# (ACG_JOBS_DIR), so a queued job runs the scripts it was submitted with.
jobs_dir=${ACG_JOBS_DIR:-$project_root/tooling/jobs/leonardo/sycl}
env_script="$jobs_dir/leonardo.sh"
[ -e "$env_script" ] || env_script="$project_root/tooling/environments/leonardo.sh"
rank_wrapper="$jobs_dir/gpu-rank-wrapper.sh"
[ -e "$rank_wrapper" ] || rank_wrapper="$project_root/tooling/jobs/gpu-rank-wrapper.sh"
# Native aCG runs start from the job's login environment plus its own modules,
# as in the native campaign, not on top of the SYCL toolchain loaded next.
base_path=$PATH
base_ld_library_path=${LD_LIBRARY_PATH:-}
source "$env_script"
source "$jobs_dir/transport.sh"
source "$jobs_dir/matrix-lib.sh"

NTASKS_PER_NODE=${ACG_NTASKS_PER_NODE:-$((ACG_NTASKS / ACG_JOB_NODES))}
MATRIX_NAME=${ACG_MATRIX_NAME:-Bump_2911}
MATRIX_DIR=${ACG_MATRIX_DIR:-$project_root/data/matrices}
MTXFILE=$(acg_find_matrix "$MATRIX_NAME") || { acg_matrix_help "$MATRIX_NAME"; exit 1; }
PARTITION_DIR=${ACG_PARTITION_DIR:-$HOME/datasets/suitesparse/partitions}
PARTFILE=""
if [ "$ACG_NTASKS" -gt 1 ]; then
    PARTFILE=${ACG_PARTITION:-$(printf '%s/%s_%02d_parts.mtx' "$PARTITION_DIR" "$MATRIX_NAME" "$ACG_NTASKS")}
    [ -e "$PARTFILE" ] || {
        echo "no such file: $PARTFILE -- stage it with tooling/jobs/leonardo/sycl/stage-partitions.sh $MATRIX_NAME" >&2
        exit 1
    }
fi

NTRIALS=${ACG_NTRIALS:-3}
MAX_ITERATIONS=${ACG_MAX_ITERATIONS:-100000}
RESIDUAL_RTOL=${ACG_RESIDUAL_RTOL:-1e-6}
WARMUP=${ACG_WARMUP:-10}
EXTRA_ARGS=${ACG_EXTRA_ARGS:-}
RESULTS_ROOT=${ACG_RESULTS_ROOT:-results}
ACG_SOLVERS=${ACG_SOLVERS:-mpi}

SYCL_BINARY=${ACG_SYCL_BINARY:-$project_root/build-release/acg}
NATIVE_ROOT=${ACG_NATIVE_ROOT:-$HOME/Projects/aCG-oshmpi}
NATIVE_BINARY=${ACG_NATIVE_BINARY:-$NATIVE_ROOT/build-oshmpi/acg-cuda}

uses_native() { case " $ACG_SOLVERS " in *" native-"*) return 0 ;; *) return 1 ;; esac; }
uses_sycl() { local k; for k in $ACG_SOLVERS; do case "$k" in native-*) ;; *) return 0 ;; esac; done; return 1; }

if uses_sycl; then
    [ -x "$SYCL_BINARY" ] || { echo "no executable: $SYCL_BINARY" >&2; exit 1; }
    # The binary must run on the MPI it was built with: the environment loads the
    # stack named by ACG_MPI, and both stacks ship a libmpi.so.40.
    built_mpi=$(sed -n 's/^MPI_CXX_COMPILER:FILEPATH=//p' "$(dirname "$SYCL_BINARY")/CMakeCache.txt" 2>/dev/null || true)
    if [ -n "$built_mpi" ] && [ "$built_mpi" != "${MPI_CXX_COMPILER:-}" ]; then
        echo "error: $SYCL_BINARY was built with $built_mpi, this job loaded ${MPI_CXX_COMPILER:-none} (ACG_MPI=$ACG_MPI)" >&2
        exit 1
    fi
    # A oneccl key needs a oneCCL build: fail at job start, before any solve.
    case " $ACG_SOLVERS " in *" oneccl "*)
        grep -q '^ACG_ENABLE_ONECCL:BOOL=ON' "$(dirname "$SYCL_BINARY")/CMakeCache.txt" 2>/dev/null || {
            echo "error: $SYCL_BINARY was built without oneCCL (ACG_ENABLE_ONECCL=ON); drop the oneccl key or rebuild" >&2
            exit 1
        } ;;
    esac
fi
if uses_native; then
    [ -x "$NATIVE_BINARY" ] || { echo "no executable: $NATIVE_BINARY (set ACG_NATIVE_BINARY or ACG_NATIVE_ROOT)" >&2; exit 1; }
fi

# oneCCL's environment is set up inside a subshell per run, so no other run in
# the same job inherits it.
#
# The solver links the module's CUDA-aware MPI and is launched with srun, so
# oneCCL must not pull a second MPI into the process. CCL_ATL_TRANSPORT=mpi
# does exactly that: libccl dlopens the bundled Intel MPI, and the two stacks
# export the same symbols with incompatible handle types. The run then dies in
# ATL setup right after
#
#   |CCL_WARN| could not get local_idx/count from environment variables,
#             trying to get them from ATL
#
# So the ATL is OFI, and oneCCL is bootstrapped by the KVS exchange that
# Collectives (src/solver/collectives.cpp) performs over MPI_Bcast. The CUDA-
# aware MPI stays the only MPI in the process, and NCCL carries every device
# payload.
acg_setup_oneccl_env() {
    local oneccl_root libfabric_dir provider_dir

    oneccl_root=${ONECCL_ROOT:-${CCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}}
    libfabric_dir=${ONECCL_LIBFABRIC_DIR:-$oneccl_root/opt/mpi/libfabric/lib}
    provider_dir=${ONECCL_LIBFABRIC_PROVIDER_DIR:-$libfabric_dir/prov-tcp-only}

    set +u
    source "$oneccl_root/env/vars.sh"
    set -u

    if [ "${CCL_ATL_TRANSPORT:-ofi}" != ofi ]; then
        echo "CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT} would load a second MPI into an OpenMPI binary; see the comment above acg_setup_oneccl_env" >&2
        exit 1
    fi
    export CCL_BACKEND=${CCL_BACKEND:-nccl}
    export CCL_ATL_TRANSPORT=ofi
    # Set by vars.sh or by the submitting environment, it would send libccl
    # straight back to the Intel MPI all of this exists to keep out.
    unset CCL_MPI_LIBRARY_PATH
    export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}
    export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
    export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-auto}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

    # ATL-OFI needs libfabric on every job size, not just off-node the way the
    # bundled Intel MPI did. Only the TCP provider is known to work with this
    # build, and a directory holding just that provider is the only way to stop
    # libfabric probing others (psmx2 and friends, which want libraries that are
    # not installed). The ATL carries bootstrap and small control messages only
    # -- the allreduce payload goes over NCCL on InfiniBand -- so the provider
    # choice costs nothing measurable.
    if [ ! -e "$provider_dir/libtcp-fi.so" ] && [ -e "$libfabric_dir/prov/libtcp-fi.so" ]; then
        mkdir -p "$provider_dir"
        ln -sf "$libfabric_dir/prov/libtcp-fi.so" "$provider_dir/libtcp-fi.so"
    fi
    export FI_PROVIDER=${FI_PROVIDER:-tcp}
    export FI_PROVIDER_PATH=${FI_PROVIDER_PATH:-$provider_dir}
    export FI_LOG_LEVEL=${FI_LOG_LEVEL:-error}

    # No Intel MPI directory belongs on this path at all -- the module's
    # OpenMPI, inherited below, has to stay the only libmpi the loader sees.
    export LD_LIBRARY_PATH="$oneccl_root/lib:$oneccl_root/lib64:$libfabric_dir:$GCC12_LIB:$DPCPP_INSTALL/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    echo "ONECCL_ROOT: $oneccl_root"
    echo "CCL_BACKEND: $CCL_BACKEND"
    echo "CCL_ATL_TRANSPORT: $CCL_ATL_TRANSPORT"
    echo "FI_PROVIDER: $FI_PROVIDER"
    echo "FI_PROVIDER_PATH: $FI_PROVIDER_PATH"
}

# Native aCG's own environment (nvhpc + HPC-X, its cuSPARSE/cuBLAS/NCCL,
# OSHMPI's libraries, which the binary links, and its transport file), in the
# run's subshell, exactly as the native campaign sets it up.
acg_setup_native_env() {
    set +u
    module purge
    export PATH=$base_path LD_LIBRARY_PATH=$base_ld_library_path
    source "$NATIVE_ROOT/cluster/leonardo/env/modules.sh"
    source "$NATIVE_ROOT/cluster/leonardo/env/oshmpi.sh"
    # Native's own transport file too, verbatim, as its campaign does. Its MPI
    # and UCX values equal transport.sh's (already set, so unchanged); it adds
    # what only the native binary reads (OMPI_MCA_osc, NVSHMEM_*, SHMEM_*).
    source "$NATIVE_ROOT/cluster/leonardo/env/transport.sh"
    set -u
}

echo "job: ${SLURM_JOB_NAME:-interactive}/${SLURM_JOB_ID:-none}"
echo "nodes: ${SLURM_NODELIST:-$(hostname)}"
echo "tasks: $ACG_NTASKS ($NTASKS_PER_NODE per node)"
echo "keys: $ACG_SOLVERS, $NTRIALS alternating rounds"
echo "sycl binary: $SYCL_BINARY"
uses_native && echo "native binary: $NATIVE_BINARY"
echo "matrix: $MTXFILE"
echo "partition: ${PARTFILE:-none (single process)}"
echo "residual rtol: $RESIDUAL_RTOL  max iterations: $MAX_ITERATIONS  warmup: $WARMUP"
acg_print_transport_env
nvidia-smi || true

mkdir -p "$RESULTS_ROOT"
RUNS_LOG="$RESULTS_ROOT/runs.tsv"
[ -e "$RUNS_LOG" ] || printf 'time\tjob\ttopology\tmatrix\tlabel\ttrial\tstatus\texit\tstem\n' >"$RUNS_LOG"

# One line when a run starts and one when it ends; a run killed by the walltime
# leaves only its `started` line.
record_run() {  # record_run <label> <trial> <status> <exit> <stem>
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-0}" \
        "${ACG_TOPOLOGY:-unknown}" "$MATRIX_NAME" "$1" "$2" "$3" "$4" "$5" >>"$RUNS_LOG"
}

# run <trial> <label> <command...>: one srun, logs under <label>/, promoted out
# of .tmp only when the solver exits 0 (converged), so a failed or capped run
# can never pass for a result. File names follow the native campaign's.
run() {
    local trial=$1 label=$2
    shift 2
    local dir="$RESULTS_ROOT/$label/suitesparse/$MATRIX_NAME"
    local stem
    stem=$(printf '%s/%s-rtol-%s-%03d-nodes-%04d-procs-%s-%s' "$dir" "$MATRIX_NAME" "$RESIDUAL_RTOL" \
        "$ACG_JOB_NODES" "$ACG_NTASKS" "${SLURM_JOB_ID:-0}" "$trial")
    mkdir -p "$dir"
    echo "$label - round $trial"
    record_run "$label" "$trial" started - "$stem"
    local rc=0
    /usr/bin/time -p --verbose \
        srun --cpu-freq=high -N "$ACG_JOB_NODES" --ntasks-per-node="$NTASKS_PER_NODE" "$@" \
        >"$stem-stdout.txt.tmp" 2>"$stem-stderr.txt.tmp" || rc=$?
    if [ "$rc" -eq 0 ]; then
        mv "$stem-stdout.txt.tmp" "$stem-stdout.txt"
        mv "$stem-stderr.txt.tmp" "$stem-stderr.txt"
        record_run "$label" "$trial" ok 0 "$stem"
    else
        echo "  FAILED rc=$rc (kept $stem-*.tmp)"
        record_run "$label" "$trial" failed "$rc" "$stem"
    fi
}

partition_args=()
[ -n "$PARTFILE" ] && partition_args=(--partition "$PARTFILE")

sycl() {  # sycl <trial> <label> <collectives>
    # One GPU per rank through the wrapper (CUDA_VISIBLE_DEVICES = local rank).
    run "$1" "$2" "$rank_wrapper" "$SYCL_BINARY" \
        --matrix "$MTXFILE" ${partition_args[@]+"${partition_args[@]}"} --device gpu \
        --solver-collectives "$3" --manufactured-solution --seed 101 \
        --residual-atol 0 --residual-rtol "$RESIDUAL_RTOL" --max-iters "$MAX_ITERATIONS" --warmup "$WARMUP" \
        $EXTRA_ARGS
}

native() {  # native <trial> <label> <comm>
    # Arguments as in the native repo's campaign-common.sh.
    run "$1" "$2" "$NATIVE_BINARY" "$MTXFILE" ${partition_args[@]+"${partition_args[@]}"} \
        --verbose --verbose --verbose -q --output-comm-matrix \
        --manufactured-solution --seed 101 --residual-atol 0 --residual-rtol "$RESIDUAL_RTOL" \
        --max-iterations "$MAX_ITERATIONS" --warmup "$WARMUP" --solver acg --comm "$3"
}

run_key() {  # run_key <key> <trial>
    local key=$1 trial=$2
    case "$key" in
        mpi)         sycl "$trial" acg-sycl-mpi mpi ;;
        none)        sycl "$trial" acg-sycl-single mpi ;;
        oneccl)      ( acg_setup_oneccl_env >/dev/null; sycl "$trial" "acg-sycl-oneccl-$CCL_BACKEND" oneccl ) ;;
        native-mpi)  ( acg_setup_native_env; native "$trial" acg-cg-mpi mpi ) ;;
        native-nccl) ( acg_setup_native_env; native "$trial" acg-cg-nccl nccl ) ;;
        native-none) ( acg_setup_native_env; native "$trial" acg-cg-single none ) ;;
        *) echo "unknown backend key: $key" >&2; exit 1 ;;
    esac
}

# Provenance: what ran, from what, where; written once per job, best effort (a
# missing tool must never cost the solves), before any solve.
write_provenance() {
    local out="$RESULTS_ROOT/provenance/${SLURM_JOB_ID:-0}"
    mkdir -p "$out"
    {
        echo "job: ${SLURM_JOB_NAME:-interactive}/${SLURM_JOB_ID:-none}"
        echo "topology: ${ACG_TOPOLOGY:-unknown}  keys: $ACG_SOLVERS  rounds: $NTRIALS"
        echo "nodelist: ${SLURM_NODELIST:-$(hostname)}"
        [ -n "${SLURM_NODELIST:-}" ] && scontrol show hostnames "$SLURM_NODELIST"
        echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "job scripts: $jobs_dir"
    } >"$out/job.txt"
    sha256sum "$jobs_dir"/*.sh >"$out/job-scripts.sha256"
    {
        echo "--- sycl: $project_root"
        git -C "$project_root" rev-parse HEAD
        git -C "$project_root" status --porcelain
        git -C "$project_root" diff HEAD | sha256sum
        echo "--- native: $NATIVE_ROOT"
        git -C "$NATIVE_ROOT" rev-parse HEAD
        git -C "$NATIVE_ROOT" status --porcelain
    } >"$out/source.txt" 2>&1
    {
        uses_sycl && sha256sum "$SYCL_BINARY" && grep -E \
            '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|ACG_[A-Z_]+|oneMath_ROOT|oneCCL_ROOT|MPI_CXX_COMPILER)[:=]' \
            "$(dirname "$SYCL_BINARY")/CMakeCache.txt"
        uses_native && sha256sum "$NATIVE_BINARY"
    } >"$out/binaries.txt" 2>&1
    uses_sycl && ldd "$SYCL_BINARY" >"$out/ldd-sycl.txt" 2>&1
    uses_native && ( acg_setup_native_env >/dev/null 2>&1; ldd "$NATIVE_BINARY" ) >"$out/ldd-native.txt" 2>&1
    {
        echo "--- CXX: ${CXX:-unset}"; "${CXX:-false}" --version | head -3
        echo "--- nvcc"; nvcc --version | tail -2
        echo "--- driver"; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
        echo "--- mpi ($ACG_MPI)"; ompi_info --version | head -2
        echo "--- ucx"; ucx_info -v | head -2
        echo "--- modules"; module list
    } >"$out/versions.txt" 2>&1
    env | grep -E '^(ACG_|UCX_|OMPI_|NCCL_|CCL_|FI_|CUDA_|SLURM_|OMP_|LD_LIBRARY_PATH)' | sort >"$out/env.txt"
    {
        ls -lL --time-style=full-iso "$MTXFILE"
        [ "${ACG_CHECKSUM_MATRIX:-0}" = 1 ] && sha256sum "$MTXFILE"
        [ -n "$PARTFILE" ] && sha256sum "$PARTFILE"
    } >"$out/inputs.txt" 2>&1
    nvidia-smi -q -d CLOCK,PERFORMANCE,POWER >"$out/gpu-state.txt" 2>&1
    # Rank -> host -> GPU -> CPU mask, through the same wrapper the SYCL runs use.
    srun -N "$ACG_JOB_NODES" --ntasks-per-node="$NTASKS_PER_NODE" "$rank_wrapper" \
        bash -c 'echo "rank=${SLURM_PROCID:-0} host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} cpus=$(taskset -cp $$ | cut -d: -f2 | xargs)"' \
        2>&1 | sort -t= -k2 -n >"$out/affinity.txt"
    echo "provenance: $out"
}
if [ "${ACG_PROVENANCE:-1}" = 1 ]; then
    ( set +e +o pipefail; write_provenance ) || echo "warning: provenance incomplete" >&2
fi

# Round-major, alternating order. Odd job ids start reversed, so across
# allocations each key gets each position.
export ACG_LOG_LOADED_LIBS=1
read -r -a keys <<<"$ACG_SOLVERS"
start=$(( ${SLURM_JOB_ID:-0} % 2 ))
for round in $(seq "$NTRIALS"); do
    order=("${keys[@]}")
    if (( (round + start) % 2 == 0 )); then
        order=()
        for ((i = ${#keys[@]} - 1; i >= 0; i--)); do order+=("${keys[$i]}"); done
    fi
    echo "round $round of $NTRIALS: ${order[*]}"
    for key in "${order[@]}"; do
        run_key "$key" "$round"
    done
done
