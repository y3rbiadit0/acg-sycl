#ifndef ACG_SOLVER_CG_SOLVER_HPP
#define ACG_SOLVER_CG_SOLVER_HPP

#include "acg/matrix/csr_matrix.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver {

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options);

} // namespace acg::solver

#endif
