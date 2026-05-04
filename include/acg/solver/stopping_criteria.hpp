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

struct ResidualDiagnostics {
  double relative_to_initial = 0.0;
  double relative_to_rhs = 0.0;
};

CgThresholds make_cg_thresholds(const SolverOptions &options, double x0_norm, double r0_norm);

bool cg_converged(const CgIterationMetrics &metrics, const CgThresholds &thresholds);
bool cg_residual_converged(double residual_norm, const CgThresholds &thresholds);
ResidualDiagnostics make_residual_diagnostics(double rhs_norm, double initial_residual, double final_residual);

} // namespace acg::solver

#endif
