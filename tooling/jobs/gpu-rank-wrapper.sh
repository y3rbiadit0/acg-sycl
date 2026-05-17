#!/bin/bash

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-}}
if [[ -n "$local_rank" ]]; then
    export CUDA_VISIBLE_DEVICES=$local_rank
fi

exec "$@"
