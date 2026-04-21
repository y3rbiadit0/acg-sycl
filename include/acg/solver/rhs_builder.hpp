#ifndef ACG_SOLVER_RHS_BUILDER_HPP
#define ACG_SOLVER_RHS_BUILDER_HPP

#include <vector>

#include "acg/solver/solver_options.hpp"

namespace acg::solver {

struct RhsBuildResult {
  std::vector<double> x_exact_host;
  std::vector<double> b_host;
};

RhsBuildResult build_rhs_inputs(std::size_t size, const SolverOptions &options);

} // namespace acg::solver

#endif
