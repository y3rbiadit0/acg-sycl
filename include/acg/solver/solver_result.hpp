#ifndef ACG_SOLVER_SOLVER_RESULT_HPP
#define ACG_SOLVER_SOLVER_RESULT_HPP

namespace acg::solver {

struct SolverResult {
  bool converged = false;
  int iterations = 0;
  double final_residual = 0.0;
  double solve_time_seconds = 0.0;   // CG iterations only
  double total_time_seconds = 0.0;   // full run_cg including setup
  double relative_solution_error = 0.0;
  double total_flops = 0.0;
  double flop_rate_gflops = 0.0;     // based on solve_time_seconds
};

} // namespace acg::solver

#endif
