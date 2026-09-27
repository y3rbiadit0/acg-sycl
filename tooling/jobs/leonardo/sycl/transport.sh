#!/bin/bash
# Open MPI + UCX transport settings, shared by every run in a job: SYCL and
# native aCG alike. A comparison between the two must differ in the solver and
# nothing else, so both get this one file.
#
# Ported from cluster/leonardo/env/transport.sh in the native aCG repo (MPI and
# UCX part; its NVSHMEM/OSHMPI knobs do not apply to the communicators run
# here), which cites gpu-comm-benchmark for the measurements behind each value.
# An explicit UCX_* / OMPI_MCA_* in the environment always wins.
#
# Input: ACG_JOB_NODES (node count of the job).

export LC_ALL=${LC_ALL:-C}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# Slurm's default binding pins a rank to one core and starves progress threads.
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}

# UCX owns InfiniBand; hcoll and UCC both claim the collective path, off.
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
# Open MPI 4.x reads opal_cuda_support; the older mpi_cuda_support name is
# accepted and ignored.
export OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support:-1}

export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
export UCX_IB_GPU_DIRECT_RDMA=${UCX_IB_GPU_DIRECT_RDMA:-yes}
export UCX_RNDV_SCHEME=${UCX_RNDV_SCHEME:-get_zcopy}
# aCG's historical threshold; halo messages sit around it.
export UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-16384}
if [ "${ACG_JOB_NODES:-1}" -gt 1 ]; then
    # cuda = cuda_copy,cuda_ipc,gdr_copy; gdr_copy is the small-message path.
    export UCX_TLS=${UCX_TLS:-sm,rc,cuda,self}
    # Adaptive routing on Leonardo's Dragonfly+, and both HCAs.
    export UCX_IB_SL=${UCX_IB_SL:-1}
    export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
else
    export UCX_TLS=${UCX_TLS:-sm,cuda,self}
fi

acg_print_transport_env() {
    local v
    for v in UCX_TLS UCX_RNDV_SCHEME UCX_RNDV_THRESH UCX_NET_DEVICES UCX_IB_GPU_DIRECT_RDMA UCX_IB_SL \
             UCX_MAX_RNDV_RAILS OMPI_MCA_pml OMPI_MCA_opal_cuda_support; do
        printf '%s: %s\n' "$v" "${!v:-unset}"
    done
}
