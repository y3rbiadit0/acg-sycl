#!/bin/bash
#SBATCH --job-name=acg_sycl_2n1g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpuq
#SBATCH -A ric_higral_366
#SBATCH --qos=normal
#SBATCH --output=acg_2n1g_%j.out
#SBATCH --error=acg_2n1g_%j.err

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "$project_root/tooling/environments/unisa-hpc.sh"

export OMPI_MCA_btl=^openib
# Inter-node: rc for IB. No cuda_ipc (NVLink only works intra-node).
# cuda_copy required whenever CUDA contexts are active.
export UCX_TLS=cuda_copy,rc,self

mpirun -np "$SLURM_NTASKS" \
  --map-by ppr:1:node --bind-to none \
  --mca coll_hcoll_enable 0 \
  --mca coll_ucc_enable 0 \
  "$project_root/tooling/jobs/gpu-rank-wrapper.sh" \
  "$project_root/build-release/acg" \
    --matrix "$project_root/data/matrices/Bump_2911/Bump_2911.mtx" \
    --device gpu \
    --mpi-mode gpu-aware \
    --residual-rtol 1e-6 \
    --max-iters 100
