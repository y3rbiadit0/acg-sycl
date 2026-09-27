#!/usr/bin/env bash
#
# Which node configurations exist, and what each one costs.
#
# A topology is <nodes>n<gpus_per_node>g, and one rank drives one GPU, so the
# string is the whole allocation shape: 2n4g is 2 nodes x 4 GPUs = 8 ranks.
# Everything else -- --nodes, --ntasks-per-node, --gres, --time, the job name,
# the baseline rule -- is derived from it here, which is why adding a
# configuration is a word in ACG_ALL_TOPOLOGIES and not a new file.
#
# The layout follows cluster/harness/matrix.sh in the gpu-comm-benchmark repo,
# for the same reason: six near-identical wrappers drift, one parser cannot.
#
# Sourced from a login node by launch.sh and from inside a job by job.sh, so it
# must work with no modules loaded and no allocation.

# Ordered smallest first, so a cheap job fails before an expensive one does.
ACG_ALL_TOPOLOGIES=${ACG_ALL_TOPOLOGIES:-"1n1g 1n2g 1n4g 2n4g 4n4g 8n4g"}

# acg_topology_fields <topology>
#   -> ACG_NODES, ACG_TASKS_PER_NODE, ACG_NTASKS_TOTAL
acg_topology_fields() {
    local topo=$1
    if [[ ! "$topo" =~ ^([1-9][0-9]*)n([1-9][0-9]*)g$ ]]; then
        echo "error: malformed topology '$topo' (expected <nodes>n<gpus_per_node>g)" >&2
        return 1
    fi
    ACG_NODES=${BASH_REMATCH[1]}
    ACG_TASKS_PER_NODE=${BASH_REMATCH[2]}
    ACG_NTASKS_TOTAL=$(( ACG_NODES * ACG_TASKS_PER_NODE ))
    # Caught here rather than by sbatch, because --gres=gpu:5 is refused at
    # submission with a message about the gres spec and not about the topology
    # string that produced it.
    if (( ACG_TASKS_PER_NODE > 4 )); then
        echo "error: topology '$topo' wants $ACG_TASKS_PER_NODE GPUs per node; a Leonardo booster node has 4" >&2
        return 1
    fi
}

# acg_topology_backends <topology> [keys...]
#
# One rank has no collective to choose, so every SYCL key collapses to `none`
# and every native key to `native-none`; filing a 1-GPU run under a
# communicator would put a number in the table that communicator did not
# produce. Several ranks have no single-GPU run, so `none` keys are dropped.
acg_topology_backends() {
    local topo=$1; shift
    acg_topology_fields "$topo" || return 1
    local keys=() key mapped
    for key in ${*:-mpi}; do
        if (( ACG_NTASKS_TOTAL == 1 )); then
            case "$key" in native-*) mapped=native-none ;; *) mapped=none ;; esac
        else
            case "$key" in none|native-none) continue ;; *) mapped=$key ;; esac
        fi
        [[ " ${keys[*]-} " == *" $mapped "* ]] || keys+=("$mapped")
    done
    echo "${keys[*]}"
}

# Minutes one run costs at a given node count, setup included: reading the
# matrix and the partition file, fabric bring-up, then the solve.
#
# Measured, not guessed. One 4n4g run of Bump_2911 took 44.5s wall -- 22.9s to
# read the matrix, 16.5s to solve, the rest srun startup. Two minutes is roughly
# 3x that, which covers the spread without asking for an hour of nodes the job
# will not use. The earlier 10/20 minute figures came from the walltimes the old
# per-scale wrappers carried, and those were far more conservative than the work
# turned out to need.
#
# Margin matters here in one direction only: Leonardo charges elapsed time, so
# overshooting costs queue position rather than budget -- and queue position is
# the scarce thing. A 15-minute ask backfills in seconds where an hour pends.
#
# A bigger matrix scales the load time, not the node count: Queen_4147 has ~2.6x
# the nonzeros of Bump_2911, so raise this for that campaign rather than editing
# per topology.
#
#   ACG_MINUTES_PER_RUN=5 make campaign-topology MATRIX_NAME=Queen_4147
#
acg_minutes_per_run() {
    local nodes=$1
    if [ -n "${ACG_MINUTES_PER_RUN:-}" ]; then
        echo "$ACG_MINUTES_PER_RUN"
    elif (( nodes >= 8 )); then
        echo 3
    else
        echo 2
    fi
}

# acg_walltime_for <topology> <runs>
#
# <runs> is backends x trials, because a job runs each backend back to back,
# ACG_NTRIALS times each.
#
# Rounded up to a quarter of an hour, not to a whole one. Rounding to the hour
# made every job ask 01:00:00 whatever its size, which is the difference between
# backfilling in seconds and pending behind the queue -- a 15-minute ask was
# granted immediately on a night an hour-long one sat waiting. The floor is one
# quarter, since nothing here is usefully shorter.

acg_walltime_for() {
    local topo=$1 runs=$2 minutes quarters
    acg_topology_fields "$topo" || return 1
    minutes=$(( runs * $(acg_minutes_per_run "$ACG_NODES") ))
    quarters=$(( (minutes + 14) / 15 ))
    (( quarters < 1 )) && quarters=1
    printf '%02d:%02d:00\n' $(( quarters / 4 )) $(( (quarters % 4) * 15 ))
}
