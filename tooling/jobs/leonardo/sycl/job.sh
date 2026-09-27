#!/bin/bash -l
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --cpus-per-task=8
#
# The single submitted script, replacing the six acg-<scale>.sh wrappers.
#
# Only the invariant #SBATCH directives stay above. The account, partition,
# --job-name, --nodes, --ntasks-per-node, --gres and --time all arrive on the
# sbatch command line, which takes precedence over #SBATCH directives -- so one
# file covers every shape of job, and a new node configuration needs no new
# file. Submit through launch.sh, which derives all of them from the topology.
#
# Inputs:
#   ACG_TOPOLOGY   <nodes>n<gpus_per_node>g, e.g. 2n4g   (required)
#   ACG_SOLVERS    backends to run back to back; defaults to the topology's
#                  own rule (`none` at one rank, `mpi` otherwise)
#
# Everything else -- ACG_MATRIX_NAME, ACG_NTRIALS, ACG_RESULTS_ROOT,
# ACG_SYCL_BINARY, ... -- is read by campaign-common.sh and documented there.
#
# Running `sbatch job.sh` by hand with no ACG_TOPOLOGY is an error rather than a
# default, because a silent default would file results under a shape they were
# not measured on.

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# ACG_JOBS_DIR is the snapshot launch.sh took at submission; without it (a
# hand-run sbatch) the working tree's scripts are used, as before.
jobs_dir=${ACG_JOBS_DIR:-$project_root/tooling/jobs/leonardo/sycl}

source "$jobs_dir/topology.sh"

if [ -z "${ACG_TOPOLOGY:-}" ]; then
    echo "error: ACG_TOPOLOGY is not set." >&2
    echo "  submit with: tooling/jobs/leonardo/sycl/launch.sh <topology>" >&2
    echo "  topologies:  $ACG_ALL_TOPOLOGIES" >&2
    exit 2
fi

acg_topology_fields "$ACG_TOPOLOGY"

# Checked against what Slurm actually granted. The allocation shape comes from
# the sbatch command line; if it disagrees with the topology label -- a stale
# --nodes, a hand-edited resubmission -- every result would be filed under a
# configuration it was not measured on, and nothing downstream could tell.
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "$SLURM_JOB_NUM_NODES" != "$ACG_NODES" ]; then
    echo "error: topology $ACG_TOPOLOGY wants $ACG_NODES nodes, allocation has $SLURM_JOB_NUM_NODES" >&2
    exit 3
fi

# The contract campaign-common.sh has always taken. Kept under these names on
# purpose: jobs queued before this refactor source that file from the working
# tree when they finally start, so its inputs must not move under them.
ACG_JOB_NODES=$ACG_NODES
ACG_NTASKS=$ACG_NTASKS_TOTAL
ACG_NTASKS_PER_NODE=$ACG_TASKS_PER_NODE
ACG_SOLVERS=${ACG_SOLVERS:-$(acg_topology_backends "$ACG_TOPOLOGY")}

echo "topology: $ACG_TOPOLOGY ($ACG_NODES node(s) x $ACG_TASKS_PER_NODE GPU(s) = $ACG_NTASKS rank(s))"

source "$jobs_dir/campaign-common.sh"
