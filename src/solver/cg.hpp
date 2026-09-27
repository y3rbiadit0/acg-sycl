#ifndef ACG_SOLVER_CG_HPP
#define ACG_SOLVER_CG_HPP

#include <vector>

#include "acg/matrix/distributed_csr_matrix.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver {

// Solves this rank's share of A x = b from x0 = 0. The same code runs on one
// GPU (a one-part partition: no halo, no-op allreduces) and on many.
// x_exact_local may be empty (no manufactured solution).
SolverResult solve_cg(
    const acg::matrix::DistributedCsrMatrixPartition &partition,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    const std::vector<double> &b_local,
    const std::vector<double> &x_exact_local);

} // namespace acg::solver

#endif
