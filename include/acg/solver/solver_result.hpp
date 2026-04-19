#ifndef ACG_SOLVER_SOLVER_RESULT_HPP
#define ACG_SOLVER_SOLVER_RESULT_HPP

namespace acg::solver {

struct SolverResult {
  bool converged = false;
  int iterations = 0;
  double final_residual = 0.0;
  double solve_time_seconds = 0.0;
  double relative_solution_error = 0.0;
};

} // namespace acg::solver

#endif
