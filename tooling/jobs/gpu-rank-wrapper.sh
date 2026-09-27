#!/bin/bash

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-}}}
if [[ -n "$local_rank" ]]; then
    export CUDA_VISIBLE_DEVICES=$local_rank
fi

# oneCCL sizes its per-node state from these, and srun sets neither of the MPI
# variables it would otherwise read them from. Note that they do not silence the
#
#   |CCL_WARN| could not get local_idx/count from environment variables
#
# warning: that lookup reads a different set of variables and falls back to
# asking the ATL, which is fine. The runs that do not use oneCCL ignore these.
local_size=${OMPI_COMM_WORLD_LOCAL_SIZE:-${MPI_LOCALNRANKS:-${SLURM_NTASKS_PER_NODE:-}}}
if [[ -n "$local_rank" && -n "$local_size" ]]; then
    export CCL_LOCAL_RANK=$local_rank
    export CCL_LOCAL_SIZE=$local_size
fi

exec "$@"
