#!/bin/bash
#SBATCH --job-name=acg_sycl_single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpuq
#SBATCH -A ric_higral_366
#SBATCH --qos=normal
#SBATCH --output=acg_single_%j.out

set -euo pipefail

project_root=${ACG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source "$project_root/tooling/environments/unisa-hpc.sh"

export OMPI_MCA_btl=^openib

"$project_root/build-release/acg" \
  --matrix "$project_root/data/matrices/Bump_2911/Bump_2911.mtx" \
  --device gpu \
  --residual-rtol 1e-6 \
  --max-iters 32000 \
  --manufactured-solution
