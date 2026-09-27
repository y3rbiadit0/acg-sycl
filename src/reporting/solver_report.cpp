#include "acg/reporting/solver_report.hpp"

#include <iomanip>
#include <ostream>

namespace acg::reporting {

void print_solver_report(std::ostream &out, const acg::solver::SolverResult &result) {
  const auto &r = result;
  const double per_iteration_us = r.iterations > 0 ? 1.0e6 * r.solver_max_s / r.iterations : 0.0;
  out << std::setprecision(6);
  out << "solver: converged=" << (r.converged ? "true" : "false") << " iterations=" << r.iterations
      << " initial_residual=" << r.initial_residual << " residual=" << r.final_residual
      << " rel_residual_r0=" << r.relative_residual << " rel_residual_rhs=" << r.relative_residual
      << " rhs_norm=" << r.rhs_norm << " relative_error=" << r.relative_solution_error << " flops=" << r.total_flops
      << " gflops=" << r.gflops << " solve_time=" << r.solver_max_s << "s"
      << " spmv_index_bits=" << r.spmv_index_bits << '\n';
  out << std::fixed << "timing: schema=" << acg::solver::SolverResult::kTimingSchema << " setup_s=" << r.setup_s
      << " warmup_s=" << r.warmup_s << " solver_s=" << r.solver_s << " solver_max_s=" << r.solver_max_s
      << " solver_min_s=" << r.solver_min_s << " per_iter_us=" << per_iteration_us << " post_solve_s=" << r.post_solve_s
      << " validation_s=" << r.validation_s << '\n';
  out << "waits: pack_s=" << r.waits.pack_s << " halo_s=" << r.waits.halo_s << " allreduce_s=" << r.waits.allreduce_s
      << " readback_s=" << r.waits.readback_s << '\n';
  out.unsetf(std::ios::floatfield);
  out << std::scientific << "validation: true_residual=" << r.true_residual
      << " true_rel_residual=" << r.true_relative_residual << " recurrence_rel_residual=" << r.relative_residual
      << '\n';
  out.unsetf(std::ios::floatfield);
}

} // namespace acg::reporting
