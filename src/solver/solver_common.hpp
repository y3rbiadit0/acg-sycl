#ifndef ACG_SOLVER_SOLVER_COMMON_HPP
#define ACG_SOLVER_SOLVER_COMMON_HPP

#include <chrono>
#include <iostream>
#include <vector>

#include "acg/runtime/run_context.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver {

inline double timed_call(const auto &fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

inline double host_squared_norm(const std::vector<double> &values) {
  double sum = 0.0;
  for (const double value : values) {
    sum += value * value;
  }
  return sum;
}

inline void maybe_log_progress(
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    int iteration,
    const SolverResult &result) {
  if (ctx.rank != 0 || options.log_every <= 0 || iteration % options.log_every != 0) {
    return;
  }
  std::cout << "iter=" << iteration
            << " residual=" << result.final_residual
            << " rel_residual_r0=" << result.relative_residual_to_initial
            << " rel_residual_rhs=" << result.relative_residual_to_rhs;
  if (options.manufactured_solution) {
    std::cout << " relative_error=" << result.relative_solution_error;
  }
  std::cout << '\n';
}

} // namespace acg::solver

#endif
