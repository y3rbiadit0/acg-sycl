#ifndef ACG_SOLVER_SOLVER_OPTIONS_HPP
#define ACG_SOLVER_SOLVER_OPTIONS_HPP

namespace acg::solver {

struct SolverOptions {
  double tolerance = 1e-8;
  int max_iterations = 1000;
};

} // namespace acg::solver

#endif
