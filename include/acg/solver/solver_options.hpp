#ifndef ACG_SOLVER_SOLVER_OPTIONS_HPP
#define ACG_SOLVER_SOLVER_OPTIONS_HPP

#include <cstdint>

namespace acg::solver {

struct SolverOptions {
  double tolerance = 1e-8;
  int max_iterations = 1000;
  bool manufactured_solution = false;
  std::uint32_t seed = 0;
};

} // namespace acg::solver

#endif
