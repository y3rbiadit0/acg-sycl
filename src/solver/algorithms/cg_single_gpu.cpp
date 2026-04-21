#include "acg/solver/algorithms/cg_single_gpu.hpp"

#include <chrono>
#include <cmath>

#include "acg/solver/stopping_criteria.hpp"

namespace acg::solver::algorithms {

namespace {

double timed_call(const auto &fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct CgSingleGpuState {
  backends::DeviceVector x;
  backends::DeviceVector r;
  backends::DeviceVector s;
  backends::DeviceVector t;
  backends::DeviceVector b;
  double rho = 0.0;
};

} // namespace

SolverResult run_cg_single_gpu(backends::SingleGpuCgBackend &backend,
                               const SolverOptions &options,
                               const std::vector<double> &x_exact_host,
                               const std::vector<double> &b_host) {
  const double matrix_vector_flops = 2.0 * static_cast<double>(backend.nnz());
  const double dot_flops = 2.0 * static_cast<double>(backend.size());
  const double update_x_r_flops = 4.0 * static_cast<double>(backend.size());
  const double update_s_flops = 3.0 * static_cast<double>(backend.size());

  SolverResult result;
  CgSingleGpuState state{
      .x = backend.create_vector(),
      .r = backend.create_vector(),
      .s = backend.create_vector(),
      .t = backend.create_vector(),
      .b = backend.create_vector_from_host(b_host),
      .rho = 0.0,
  };

  if (options.manufactured_solution) {
    const auto x_exact = backend.create_vector_from_host(x_exact_host);
    // 2. GPU SpMV: r0 <- b - A * x0 becomes b <- A * x_exact in manufactured mode.
    result.perf.spmv.record(timed_call([&] { backend.spmv(x_exact, state.b); }), backend.estimated_spmv_bytes());
    result.total_flops += matrix_vector_flops;
  }

  // 5. GPU COPY / initialization: x <- 0, r <- b, s <- r.
  backend.fill_zero(state.x);
  result.perf.copy.record(timed_call([&] { backend.copy(state.b, state.r); }), backend.estimated_copy_bytes());
  result.perf.copy.record(timed_call([&] { backend.copy(state.r, state.s); }), backend.estimated_copy_bytes());

  // 3-4. GPU DOT + CPU SYNC: rho0 <- r0 . r0.
  result.perf.nrm2.record(timed_call([&] { state.rho = backend.dot(state.r, state.r); }), backend.estimated_nrm2_bytes());
  result.total_flops += dot_flops;

  const double r0_norm = std::sqrt(state.rho);
  const CgThresholds thresholds = make_cg_thresholds(options, 0.0, r0_norm);
  result.final_residual = r0_norm;
  result.converged = cg_converged(CgIterationMetrics{.dx_norm = 0.0, .residual_norm = r0_norm}, thresholds);

  const auto solve_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < options.max_iterations && !result.converged; ++iteration) {
    // 7. GPU SpMV: t <- A * s.
    result.perf.spmv.record(timed_call([&] { backend.spmv(state.s, state.t); }), backend.estimated_spmv_bytes());
    result.total_flops += matrix_vector_flops;

    // 8-10. GPU DOT + CPU SYNC + CPU: gamma <- s . t, alpha <- rho / gamma.
    double gamma = 0.0;
    result.perf.dot.record(timed_call([&] { gamma = backend.dot(state.s, state.t); }), backend.estimated_dot_bytes());
    result.total_flops += dot_flops;
    if (gamma <= 0.0) {
      throw std::runtime_error("matrix is not positive definite under CG iteration");
    }
    const double alpha = state.rho / gamma;

    // Update norm for stopping criteria parity with original algorithm.
    double s_norm_squared = 0.0;
    result.perf.nrm2.record(timed_call([&] { s_norm_squared = backend.dot(state.s, state.s); }), backend.estimated_nrm2_bytes());
    result.total_flops += dot_flops;
    const double dx_norm = std::abs(alpha) * std::sqrt(s_norm_squared);

    // 11. GPU AXPY: x_k <- alpha * s + x_{k-1}.
    result.perf.axpy.record(timed_call([&] { backend.axpy(alpha, state.s, state.x); }), backend.estimated_axpy_bytes());

    // 12. GPU AXPY: r_k <- -alpha * t + r_{k-1}.
    result.perf.axpy.record(timed_call([&] { backend.axpy(-alpha, state.t, state.r); }), backend.estimated_axpy_bytes());
    result.total_flops += update_x_r_flops;

    // 13-16. GPU DOT + CPU SYNC + CPU: rho_k <- r_k . r_k, check convergence, beta <- rho_k / rho_{k-1}.
    double rho_next = 0.0;
    result.perf.nrm2.record(timed_call([&] { rho_next = backend.dot(state.r, state.r); }), backend.estimated_nrm2_bytes());
    result.total_flops += dot_flops;

    result.iterations = iteration + 1;
    result.final_residual = std::sqrt(rho_next);
    result.converged = cg_converged(CgIterationMetrics{.dx_norm = dx_norm, .residual_norm = result.final_residual}, thresholds);
    if (result.converged) {
      break;
    }

    const double beta = rho_next / state.rho;

    // 17. GPU AXPY: s <- beta * s + r.
    result.perf.axpy.record(timed_call([&] { backend.scal(beta, state.s); }), backend.estimated_scal_bytes());
    result.perf.axpy.record(timed_call([&] { backend.axpy(1.0, state.r, state.s); }), backend.estimated_axpy_bytes());
    result.total_flops += update_s_flops;

    state.rho = rho_next;
  }

  std::vector<double> x_host;
  backend.download_vector(state.x, x_host);

  if (options.manufactured_solution) {
    double x_exact_norm_squared = 0.0;
    double x_error_norm_squared = 0.0;
    for (std::size_t i = 0; i < x_host.size(); ++i) {
      const double error = x_host[i] - x_exact_host[i];
      x_exact_norm_squared += x_exact_host[i] * x_exact_host[i];
      x_error_norm_squared += error * error;
    }
    const double x_exact_norm = std::sqrt(x_exact_norm_squared);
    result.relative_solution_error = x_exact_norm > 0.0 ? std::sqrt(x_error_norm_squared) / x_exact_norm : 0.0;
  }

  const auto end = std::chrono::steady_clock::now();
  result.solve_time_seconds = std::chrono::duration<double>(end - solve_start).count();
  result.total_time_seconds = result.solve_time_seconds;
  result.flop_rate_gflops = result.solve_time_seconds > 0.0 ? result.total_flops / result.solve_time_seconds / 1.0e9 : 0.0;
  return result;
}

} // namespace acg::solver::algorithms
