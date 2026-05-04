#!/bin/bash
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=acg_sycl_validate_4gpu
#SBATCH --error=./results/acg_sycl_4gpu_%j_err.txt
#SBATCH --output=./results/acg_sycl_4gpu_%j_out.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.merenda2@studenti.unisa.it

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
mkdir -p "$project_root/results"

source "$project_root/tooling/environments/leonardo.sh"

export OMPI_MCA_btl=^openib
export UCX_TLS=sm,cuda_copy,cuda_ipc,self

mpirun -np "$SLURM_NTASKS" \
  --map-by ppr:4:node --bind-to none \
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
