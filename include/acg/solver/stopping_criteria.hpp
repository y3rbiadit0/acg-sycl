#ifndef ACG_SOLVER_STOPPING_CRITERIA_HPP
#define ACG_SOLVER_STOPPING_CRITERIA_HPP

#include "acg/solver/solver_options.hpp"

namespace acg::solver {

struct CgThresholds {
  double diff_absolute = 0.0;
  double diff_relative_scaled = 0.0;
  double residual_absolute = 0.0;
  double residual_relative_scaled = 0.0;
};

struct CgIterationMetrics {
  double dx_norm = 0.0;
  double residual_norm = 0.0;
};

CgThresholds make_cg_thresholds(const SolverOptions &options, double x0_norm, double r0_norm);

bool cg_converged(const CgIterationMetrics &metrics, const CgThresholds &thresholds);

} // namespace acg::solver

#endif
