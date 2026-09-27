#!/usr/bin/env bash
# Submit solver jobs by topology.
#
# One topology:
#   tooling/jobs/leonardo/sycl/launch.sh 2n4g
#   tooling/jobs/leonardo/sycl/launch.sh 2n4g --qos=boost_qos_dbg   # extra sbatch args
#
# Every topology:
#   tooling/jobs/leonardo/sycl/launch.sh --all
#
# Run one now, in the foreground, instead of queueing it:
#   tooling/jobs/leonardo/sycl/launch.sh --now 2n4g
#   ACG_NTRIALS=1 tooling/jobs/leonardo/sycl/launch.sh --now 1n4g --qos=boost_qos_dbg
#
# --now takes the allocation with salloc and runs job.sh inside it, so output
# arrives live and a failure is visible immediately instead of in a log an hour
# later. Same job.sh, same campaign body, same results tree -- only the way the
# allocation is obtained differs. Use it to get one configuration measured while
# the batch queue is busy, and to debug a topology before committing a sweep.
#
# Inspect without submitting:
#   tooling/jobs/leonardo/sycl/launch.sh --dry-run --all
#   tooling/jobs/leonardo/sycl/launch.sh --explain 8n4g
#
# Environment:
#   ACG_BACKENDS      keys each job runs, interleaved round by round (default
#                     "mpi oneccl native-mpi native-nccl"; a 1-rank topology
#                     maps them to `none native-none`)
#   ACG_NTRIALS       rounds per job: every key runs once per round (default 3)
#   ACG_ONLY_TOPOS    space-separated globs, --all only
#   ACG_REPEATS       submit each topology as N independent jobs (default 1)
#   ACG_NTRIALS=1     what you usually want with --now: one solve per backend
#   ACG_TIME          one walltime for every job, overriding what the topology
#                     would derive
#   ACG_SLURM_ACCOUNT, ACG_SLURM_PARTITION
#   ACG_MATRIX_NAME, ACG_MATRIX_DIR, ACG_RESULTS_ROOT, ACG_SYCL_BINARY, ...
#                     exported here and inherited by every job
#   ACG_NATIVE_ROOT, ACG_NATIVE_BINARY   the native aCG build the native-* keys run
#
# Every submission snapshots the job scripts into
# .job-snapshots/, and the job runs from that copy: editing the scripts while a
# job waits in the queue no longer changes what it runs.
#
# On repetition. Trials inside one job share an allocation, so they measure the
# same nodes, the same fabric neighbours and the same clocks; the spread between
# allocations is the larger one. ACG_REPEATS is how you sample that -- each
# repeat is a separate job, and results key on $SLURM_JOB_ID so they accumulate
# side by side. The trade-off is queue slots, which is the whole reason this
# campaign moved to one job per topology; see the README.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
JOBS="$ROOT/tooling/jobs/leonardo/sycl"

source "$JOBS/topology.sh"

ACG_SLURM_ACCOUNT=${ACG_SLURM_ACCOUNT:-IscrC_HIGRAPH_0}
ACG_SLURM_PARTITION=${ACG_SLURM_PARTITION:-boost_usr_prod}
ACG_BACKENDS=${ACG_BACKENDS:-"mpi oneccl native-mpi native-nccl"}
ACG_NTRIALS=${ACG_NTRIALS:-3}

# snapshot_jobs -> prints the snapshot directory: the job scripts, the rank
# wrapper and the environment script, frozen at submission.
snapshot_jobs() {
    local snap
    snap="$ROOT/.job-snapshots/$(date +%Y%m%dT%H%M%S)-$$-$RANDOM"
    mkdir -p "$snap"
    cp "$JOBS"/*.sh "$snap/"
    cp "$ROOT/tooling/jobs/gpu-rank-wrapper.sh" "$snap/"
    cp "$ROOT/tooling/environments/leonardo.sh" "$snap/leonardo.sh"
    git -C "$ROOT" rev-parse HEAD >"$snap/SOURCE_REVISION" 2>/dev/null || true
    echo "$snap"
}

mode=single
dry_run=${ACG_DRYRUN:-0}
explain=0
now=0
args=()
for arg in "$@"; do
    case "$arg" in
        --all)     mode=all ;;
        --dry-run) dry_run=1 ;;
        --now)     now=1 ;;
        --explain) explain=1; dry_run=1 ;;
        -h|--help) sed -n '2,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         args+=("$arg") ;;
    esac
done

require_sbatch() {
    local tool=sbatch
    [[ "$now" == "1" ]] && tool=salloc
    if [[ "$dry_run" != "1" ]] && ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found -- run this on Leonardo (a login node)." >&2
        exit 1
    fi
}

# boost_qos_dbg caps a job at 30 minutes and 2 nodes. The walltime derived from
# the topology is usually longer than that, and Slurm rejects the job outright
# rather than trimming it -- an easy way to lose a --now run to a one-line error
# after waiting for the prompt. Clamp instead, unless a walltime was given.
clamp_for_dbg_qos() {
    local walltime=$1; shift
    local arg
    for arg in "$@"; do
        case "$arg" in
            --qos=boost_qos_dbg|boost_qos_dbg)
                if [[ -z "${ACG_TIME:-}" && "$walltime" > "00:30:00" ]]; then
                    echo "00:30:00"
                    return 0
                fi
                ;;
        esac
    done
    echo "$walltime"
}

# submit <topology> [extra sbatch args...]
submit() {
    local topo="$1"; shift
    acg_topology_fields "$topo"

    local backends walltime runs
    backends=$(acg_topology_backends "$topo" $ACG_BACKENDS)
    runs=$(( $(set -- $backends; echo $#) * ACG_NTRIALS ))
    walltime=${ACG_TIME:-$(acg_walltime_for "$topo" "$runs")}

    if [[ "$explain" == "1" ]]; then
        cat <<EOF
topology   : $topo -- $ACG_NODES node(s) x $ACG_TASKS_PER_NODE GPU(s) = $ACG_NTASKS_TOTAL rank(s)
backends   : $backends
rounds     : $ACG_NTRIALS, every key once per round, order alternating
runs       : $runs ($(acg_minutes_per_run "$ACG_NODES") min budgeted each)
job script : tooling/jobs/leonardo/sycl/job.sh
topologies : tooling/jobs/leonardo/sycl/topology.sh
body       : tooling/jobs/leonardo/sycl/campaign-common.sh
matrix     : ${ACG_MATRIX_NAME:-Bump_2911}
results    : ${ACG_RESULTS_ROOT:-results}/<backend label>/suitesparse/${ACG_MATRIX_NAME:-Bump_2911}
account    : $ACG_SLURM_ACCOUNT   partition: $ACG_SLURM_PARTITION
sbatch     : --nodes=$ACG_NODES --ntasks-per-node=$ACG_TASKS_PER_NODE --gres=gpu:$ACG_TASKS_PER_NODE --time=$walltime
EOF
        return 0
    fi
    walltime=$(clamp_for_dbg_qos "$walltime" "$@")

    if [[ "$dry_run" == "1" ]]; then
        printf 'would %s: %-5s  %-2s node(s) x %s GPU(s)  %s  [%s]\n' \
            "$([[ "$now" == "1" ]] && echo run || echo submit)" \
            "$topo" "$ACG_NODES" "$ACG_TASKS_PER_NODE" "$walltime" "$backends"
        return 0
    fi

    # --now: hold the allocation in this shell and run the job body inside it.
    # salloc blocks until the scheduler grants the nodes, so the wait is visible
    # and interruptible, and everything job.sh prints comes straight to the
    # terminal. campaign-common.sh already launches each solve with its own
    # srun, which becomes a job step inside this allocation rather than a job of
    # its own -- so the body needs no change for this path.
    local snap
    snap=$(snapshot_jobs)

    if [[ "$now" == "1" ]]; then
        echo "allocating $topo ($ACG_NODES node(s) x $ACG_TASKS_PER_NODE GPU(s), $walltime) -- ctrl-c to give up waiting"
        # salloc has no --chdir, and the campaign body writes results/ and logs/
        # relative to the working directory. Subshell so a later iteration is
        # not affected.
        (
        cd "$ROOT" || exit 1
        ACG_PROJECT_ROOT="$ROOT" \
        ACG_JOBS_DIR="$snap" \
        ACG_TOPOLOGY="$topo" \
        ACG_SOLVERS="$backends" \
        ACG_NTRIALS="$ACG_NTRIALS" \
        salloc \
            --account="$ACG_SLURM_ACCOUNT" \
            --partition="$ACG_SLURM_PARTITION" \
            --job-name="acg_sycl_$topo" \
            --nodes="$ACG_NODES" \
            --ntasks-per-node="$ACG_TASKS_PER_NODE" \
            --gres=gpu:"$ACG_TASKS_PER_NODE" \
            --time="$walltime" \
            "$@" \
            bash "$snap/job.sh"
        )
        return $?
    fi

    # The allocation shape is passed here, not baked into a script: sbatch
    # command-line options override #SBATCH directives, so one job.sh serves
    # every topology. ACG_* reach the job through the environment, which sbatch
    # propagates by default.
    ACG_PROJECT_ROOT="$ROOT" \
    ACG_JOBS_DIR="$snap" \
    ACG_TOPOLOGY="$topo" \
    ACG_SOLVERS="$backends" \
    ACG_NTRIALS="$ACG_NTRIALS" \
    sbatch \
        --account="$ACG_SLURM_ACCOUNT" \
        --partition="$ACG_SLURM_PARTITION" \
        --chdir="$ROOT" \
        --job-name="acg_sycl_$topo" \
        --nodes="$ACG_NODES" \
        --ntasks-per-node="$ACG_TASKS_PER_NODE" \
        --gres=gpu:"$ACG_TASKS_PER_NODE" \
        --time="$walltime" \
        "$@" \
        "$snap/job.sh"
}

if [[ "$mode" == "single" ]]; then
    if [[ ${#args[@]} -lt 1 ]]; then
        echo "usage: launch.sh <topology> [sbatch args...]" >&2
        echo "       launch.sh --all [sbatch args...]" >&2
        echo "       launch.sh --explain <topology>" >&2
        echo "topologies: $ACG_ALL_TOPOLOGIES" >&2
        exit 2
    fi
    require_sbatch
    submit "${args[@]}"
    exit 0
fi

matches() {  # matches <value> <space-separated-globs-or-empty>
    local value="$1" filters="$2" f
    [[ -z "$filters" ]] && return 0
    for f in $filters; do [[ "$value" == $f ]] && return 0; done
    return 1
}

require_sbatch

repeats=${ACG_REPEATS:-1}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "ACG_REPEATS must be a positive integer" >&2; exit 2; }

submitted=0
skipped=0
for topo in $ACG_ALL_TOPOLOGIES; do
    matches "$topo" "${ACG_ONLY_TOPOS:-}" || { skipped=$((skipped+1)); continue; }
    for repeat in $(seq "$repeats"); do
        if [[ "$dry_run" != "1" ]]; then
            label="$topo"
            [[ "$repeats" -gt 1 ]] && label="$label (repeat $repeat/$repeats)"
            echo "submitting: $label"
        fi
        submit "$topo" ${args[@]+"${args[@]}"}
        submitted=$((submitted+1))
    done
done

echo "---"
echo "submitted: $submitted   skipped (filtered): $skipped"
echo "watch queue: squeue -u \$USER"
