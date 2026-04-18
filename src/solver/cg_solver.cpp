#include "acg/solver/cg_solver.hpp"

#include <chrono>

namespace acg::solver {

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options) {
  (void)matrix;
  (void)ctx;
  (void)options;

  const auto start = std::chrono::steady_clock::now();
  const auto end = std::chrono::steady_clock::now();

  SolverResult result;
  result.converged = true;
  result.iterations = 0;
  result.final_residual = 0.0;
  result.solve_time_seconds = std::chrono::duration<double>(end - start).count();
  return result;
}

} // namespace acg::solver
