#include "acg/solver/stopping_criteria.hpp"

namespace acg::solver {

CgThresholds make_cg_thresholds(const SolverOptions &options, double x0_norm, double r0_norm) {
  CgThresholds thresholds;
  thresholds.diff_absolute = options.diff_absolute_tolerance;
  thresholds.diff_relative_scaled = options.diff_relative_tolerance > 0.0 ? options.diff_relative_tolerance * x0_norm : 0.0;
  thresholds.residual_absolute = options.residual_absolute_tolerance;
  thresholds.residual_relative_scaled =
      options.residual_relative_tolerance > 0.0 ? options.residual_relative_tolerance * r0_norm : 0.0;
  return thresholds;
}

bool cg_converged(const CgIterationMetrics &metrics, const CgThresholds &thresholds) {
  return (thresholds.diff_absolute > 0.0 && metrics.dx_norm < thresholds.diff_absolute) ||
         (thresholds.diff_relative_scaled > 0.0 && metrics.dx_norm < thresholds.diff_relative_scaled) ||
         (thresholds.residual_absolute > 0.0 && metrics.residual_norm < thresholds.residual_absolute) ||
         (thresholds.residual_relative_scaled > 0.0 && metrics.residual_norm < thresholds.residual_relative_scaled);
}

} // namespace acg::solver
