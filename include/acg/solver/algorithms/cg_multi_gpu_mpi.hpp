#ifndef ACG_SOLVER_ALGORITHMS_CG_MULTI_GPU_MPI_HPP
#define ACG_SOLVER_ALGORITHMS_CG_MULTI_GPU_MPI_HPP

#include <vector>

#include "acg/matrix/distributed_csr_matrix.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver::algorithms {

SolverResult run_cg_multi_gpu_mpi(
    const acg::matrix::DistributedCsrMatrixPartition &partition,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    const std::vector<double> &x_exact_local_host,
    const std::vector<double> &b_local_host);

} // namespace acg::solver::algorithms

#endif
