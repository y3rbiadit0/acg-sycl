#include "acg/reporting/solver_report.hpp"

#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <string>

#include "acg/solver/perf_breakdown.hpp"

namespace acg::reporting {

namespace {

struct DerivedPerfMetrics {
  double cuda_reference_solve_time = 0.0;
  acg::solver::OpStats host_sync_exclusive;
  double other_time = 0.0;
};

void print_perf_op(std::ostream &out, const char *name, const acg::solver::OpStats &stats) {
  out << std::fixed << std::setprecision(6)
      << "   " << name << ": "
      << stats.time_seconds << " seconds/proc "
      << stats.count << " times/proc "
      << stats.bytes << " B/proc "
      << std::setprecision(3) << stats.bandwidth_gbs() << " GB/s/proc";
  if (std::string(name) == "allreduce" && stats.count > 0) {
    out << ' ' << std::setprecision(3) << (1.0e6 * stats.time_seconds / static_cast<double>(stats.count))
        << " us/op/proc";
  }
  out << '\n';
}

DerivedPerfMetrics derive_perf_metrics(const acg::solver::SolverResult &result) {
  DerivedPerfMetrics metrics;
  metrics.cuda_reference_solve_time = result.cuda_reference_solve_time_seconds > 0.0
                                          ? result.cuda_reference_solve_time_seconds
                                          : result.solve_time_seconds;
  metrics.host_sync_exclusive = result.perf.native.host_sync;
  metrics.host_sync_exclusive.time_seconds -= result.perf.native.pack.time_seconds;
  if (metrics.host_sync_exclusive.time_seconds < 0.0) {
    metrics.host_sync_exclusive.time_seconds = 0.0;
  }

  const double known_time = result.perf.cuda.gemv.time_seconds
                            + result.perf.cuda.dot.time_seconds
                            + result.perf.cuda.nrm2.time_seconds
                            + result.perf.cuda.axpy.time_seconds
                            + result.perf.cuda.copy.time_seconds
                            + result.perf.cuda.allreduce.time_seconds
                            + result.perf.cuda.haloexchange.time_seconds
                            + result.perf.native.pack.time_seconds
                            + metrics.host_sync_exclusive.time_seconds;
  metrics.other_time = metrics.cuda_reference_solve_time - known_time;
  return metrics;
}

void print_solver_summary(std::ostream &out, const acg::solver::SolverResult &result) {
  out << "solver: converged=" << (result.converged ? "true" : "false")
      << " iterations=" << result.iterations
      << " initial_residual=" << result.initial_residual
      << " residual=" << result.final_residual
      << " rel_residual_r0=" << result.relative_residual_to_initial
      << " rel_residual_rhs=" << result.relative_residual_to_rhs
      << " rhs_norm=" << result.rhs_norm
      << " relative_error=" << result.relative_solution_error
      << " flops=" << result.total_flops
      << " gflops=" << result.flop_rate_gflops
      << " solve_time=" << result.solve_time_seconds << "s"
      << " total_time=" << result.total_time_seconds << "s\n";
}

void print_cuda_compatible_perf(
    std::ostream &out,
    const acg::solver::SolverResult &result,
    const DerivedPerfMetrics &metrics) {
  out << "performance_cuda_compatible:\n"
      << "   note: names and formulas aligned with ParCoreLab/aCG CUDA logs\n"
      << "   total_flops: " << std::fixed << std::setprecision(3)
      << (result.cuda_reference_total_flops / 1.0e9) << " Gflop\n"
      << "   flop_rate: " << result.cuda_reference_flop_rate_gflops << " Gflop/s\n"
      << "   solver_time: " << std::setprecision(6) << metrics.cuda_reference_solve_time << " seconds\n";
  print_perf_op(out, "gemv", result.perf.cuda.gemv);
  print_perf_op(out, "dot", result.perf.cuda.dot);
  print_perf_op(out, "nrm2", result.perf.cuda.nrm2);
  print_perf_op(out, "axpy", result.perf.cuda.axpy);
  print_perf_op(out, "copy", result.perf.cuda.copy);
  print_perf_op(out, "allreduce", result.perf.cuda.allreduce);
  print_perf_op(out, "haloexchange", result.perf.cuda.haloexchange);
}

void print_sycl_overheads(
    std::ostream &out,
    const acg::solver::SolverResult &result,
    const DerivedPerfMetrics &metrics) {
  out << "performance_sycl_overheads:\n"
      << "   note: SYCL/runtime costs outside CUDA-compatible categories\n";
  print_perf_op(out, "pack", result.perf.native.pack);
  print_perf_op(out, "host_sync", metrics.host_sync_exclusive);
  out << "   other: " << std::fixed << std::setprecision(6) << metrics.other_time << " seconds\n";
}

void print_native_perf(std::ostream &out, const acg::solver::SolverResult &result) {
  out << "performance_sycl_native:\n"
      << "   note: native SYCL counters and byte formulas; not CUDA-comparable\n";
  print_perf_op(out, "spmv", result.perf.native.spmv);
  print_perf_op(out, "dot", result.perf.native.dot);
  print_perf_op(out, "nrm2", result.perf.native.nrm2);
  print_perf_op(out, "axpy", result.perf.native.axpy);
  print_perf_op(out, "copy", result.perf.native.copy);
  print_perf_op(out, "pack", result.perf.native.pack);
  print_perf_op(out, "p2p", result.perf.native.p2p);
  print_perf_op(out, "allreduce", result.perf.native.allreduce);
  print_perf_op(out, "host_sync_total", result.perf.native.host_sync);
}

} // namespace

SolverReportOptions solver_report_options_from_environment() {
  const char *log_native_perf = std::getenv("ACG_LOG_NATIVE_PERF");
  return SolverReportOptions{.log_native_perf = log_native_perf != nullptr && *log_native_perf != '\0'};
}

void print_solver_report(
    std::ostream &out,
    const acg::solver::SolverResult &result,
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::cli::AppConfig &config,
    const SolverReportOptions &options) {
  (void)matrix;
  (void)config;

  print_solver_summary(out, result);
  if (result.perf.cuda.gemv.count == 0) {
    return;
  }

  const DerivedPerfMetrics metrics = derive_perf_metrics(result);
  print_cuda_compatible_perf(out, result, metrics);
  print_sycl_overheads(out, result, metrics);
  if (options.log_native_perf) {
    print_native_perf(out, result);
  }
}

} // namespace acg::reporting
