#!/bin/bash
#SBATCH --job-name=acg_sycl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=gpuq
#SBATCH -A ric_higral_366
#SBATCH --qos=normal
#SBATCH --output=acg_%j.out
#SBATCH --error=acg_%j.err

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "$project_root/tooling/environments/unisa-hpc.sh"

export OMPI_MCA_btl=^openib
# sm   = intra-node host memory (allreduce scalars, host-staged halo)
# cuda_ipc = intra-node device→device via NVLink (gpu-aware halo)
# rc   = inter-node InfiniBand
# self = loopback
# Not setting UCX_MEMTYPE_CACHE=n: with the explicit TLS list UCX detects
# memory types once and caches them; re-checking every call was causing ~4ms/allreduce.
# rc_mlx5 (IB) scores higher than sysv in ucp_context_2, but IB completion
# events on this cluster have ~4ms OS-timer jitter. Exclude rc so UCX is
# forced to use sm (sysv/knem) for host-host — microseconds intra-node.
# For multi-node runs add ,rc back (and remove --mca coll_hcoll/ucc_enable 0).
export UCX_TLS=sm,cuda_copy,cuda_ipc,self

mpirun -np "$SLURM_NTASKS" $ACG_MPIRUN_ARGS \
  --mca coll_hcoll_enable 0 \
  --mca coll_ucc_enable 0 \
  "$project_root/tooling/jobs/gpu-rank-wrapper.sh" \
  "$project_root/build-release/acg" \
    --matrix "$project_root/data/matrices/Bump_2911/Bump_2911.mtx" \
    --device gpu \
    --mpi-mode gpu-aware \
    --residual-rtol 1e-6 \
    --max-iters 32000 \
    --manufactured-solution
