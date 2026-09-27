#include "cg.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <mpi.h>

#include "acg/solver/device_csr.hpp"
#include "cg_kernels.hpp"
#include "collectives.hpp"
#include "halo.hpp"

namespace acg::solver {

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

double allreduce_host(double value, MPI_Op op) {
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
  return value;
}

// Rank 0 prints one line per rank; `values` has `count` entries on every rank.
// Counts are printed as integers, which is what the analysis parses.
void print_per_rank(const acg::runtime::RunContext &ctx, const char *tag, const char *header,
                    const double *values, int count, bool integers = false) {
  std::vector<double> all(ctx.rank == 0 ? static_cast<std::size_t>(ctx.size * count) : 0);
  MPI_Gather(values, count, MPI_DOUBLE, all.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (ctx.rank != 0) {
    return;
  }
  std::cout << tag << ": rank " << header << '\n';
  for (int r = 0; r < ctx.size; ++r) {
    std::cout << tag << ": " << r;
    for (int i = 0; i < count; ++i) {
      const double value = all[static_cast<std::size_t>(r * count + i)];
      if (integers) {
        std::cout << ' ' << static_cast<long long>(value);
      }
      else {
        std::cout << ' ' << value;
      }
    }
    std::cout << '\n';
  }
}

// Before timing: the decomposition each rank got, which the analysis turns into
// load-imbalance figures.
void print_partition(const acg::matrix::DistributedCsrMatrixPartition &partition, const acg::runtime::RunContext &ctx) {
  std::int64_t imports = 0;
  std::int64_t exports = 0;
  for (const auto &peer : partition.imports) {
    imports += static_cast<std::int64_t>(peer.global_columns.size());
  }
  for (const auto &peer : partition.exports) {
    exports += static_cast<std::int64_t>(peer.local_indices.size());
  }
  if (ctx.rank == 0) {
    std::cout << "solver_diag_partition_method: " << partition.method << '\n';
  }
  const double stats[] = {
      static_cast<double>(partition.local_rows()),
      static_cast<double>(partition.local_matrix.nnz()),
      static_cast<double>(partition.interior_matrix.nnz()),
      static_cast<double>(partition.halo_matrix.nnz()),
      static_cast<double>(partition.ghost_global_columns.size()),
      static_cast<double>(imports),
      static_cast<double>(exports),
  };
  print_per_rank(ctx, "solver_diag_partition", "local_rows local_nnz interior_nnz halo_nnz ghosts imports exports",
                 stats, 7, true);
}

// After timing: where each rank's host was blocked, and the spread across
// ranks, which shows load imbalance and slow links.
void print_waits(const SolverResult &result, const acg::runtime::RunContext &ctx) {
  const double values[] = {result.solver_s, result.waits.pack_s, result.waits.halo_s, result.waits.allreduce_s,
                           result.waits.readback_s};
  const char *names[] = {"solver", "pack", "p2p", "allreduce", "host_sync"};
  for (int i = 0; i < 5; ++i) {
    const double min = allreduce_host(values[i], MPI_MIN);
    const double max = allreduce_host(values[i], MPI_MAX);
    const double avg = allreduce_host(values[i], MPI_SUM) / ctx.size;
    if (ctx.rank == 0) {
      std::cout << "solver_diag_perf: " << names[i] << " min_s=" << min << " avg_s=" << avg << " max_s=" << max
                << " skew_s=" << (max - min) << '\n';
    }
  }
  print_per_rank(ctx, "solver_diag_rank_waits", "solver_s pack_s halo_s allreduce_s readback_s", values, 5);
}

} // namespace

SolverResult solve_cg(
    const acg::matrix::DistributedCsrMatrixPartition &partition,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    const std::vector<double> &b_local,
    const std::vector<double> &x_exact_local) {
  const auto setup_start = Clock::now();
  if (partition.local_rows() <= 0) {
    throw std::runtime_error("local partition is empty");
  }
  print_partition(partition, ctx);

  sycl::queue queue = ctx.queue;
  SolverResult result;
  DeviceCsr A(queue, partition.interior_matrix);
  result.spmv_index_bits = A.index_bits();
  HaloExchange halo(queue, partition);
  std::optional<HaloSpmv> halo_spmv;
  if (partition.halo_matrix.nnz() > 0) {
    halo_spmv.emplace(queue, partition.halo_matrix);
  }
  Collectives collectives(queue, options.solver_collectives, ctx.rank, ctx.size);
  kernels::ScalarBlock scalars(queue);

  const std::int64_t n = partition.local_rows();
  DeviceVector x = A.vector(n);
  DeviceVector r = A.vector(n);
  DeviceVector s = A.vector(n);
  DeviceVector t = A.vector(n);
  const DeviceVector b = A.vector(b_local);
  const auto ghost_count = static_cast<std::int64_t>(partition.ghost_global_columns.size());
  double *ghosts = ghost_count > 0 ? A.vector(ghost_count).data : nullptr;
  A.prepare_spmv(s, t);

  // MPI and oneCCL do not see SYCL dependencies, so the value's producer is
  // waited for first; the wait is counted as part of the reduction.
  const auto allreduce = [&](double *value, WaitTimes &waits) {
    if (ctx.size <= 1) {
      return;
    }
    const auto start = Clock::now();
    queue.wait_and_throw();
    collectives.allreduce_sum(value);
    waits.allreduce_s += seconds(start, Clock::now());
  };

  // x = 0, r = s = b, rho = (r, r). With x0 = 0, r0 = b, so ||r0|| = ||b||.
  const auto initialize = [&](WaitTimes &waits) {
    scalars.reset();
    A.fill_zero(x);
    A.copy(b, r);
    A.copy(b, s);
    A.dot(r, r, scalars.rho());
    allreduce(scalars.rho(), waits);
    scalars.read_rho().wait_and_throw();
    return scalars.host_rho();
  };

  // One CG iteration, entirely queued: the host blocks only in the halo
  // exchange and the allreduces. The returned event is the copy of gamma and
  // the new rho to the host; the search-direction update is queued behind it,
  // so it runs while the host waits for the copy and tests for convergence
  // (if the solve stops, that update to s is never read).
  const auto iterate = [&](WaitTimes &waits) {
    if (halo.active()) {
      halo.begin(s, ghosts, waits);
    }
    A.spmv(s, t);
    if (halo.active()) {
      halo.finish(waits);
      if (halo_spmv) {
        halo_spmv->add(ghosts, t);
      }
    }
    A.dot(s, t, scalars.gamma());
    allreduce(scalars.gamma(), waits);
    kernels::update_solution_and_residual(queue, scalars.rho(), scalars.gamma(), s, t, x, r);
    A.dot(r, r, scalars.rho_next());
    allreduce(scalars.rho_next(), waits);
    sycl::event readback = scalars.read_gamma_and_rho_next();
    kernels::update_search_direction(queue, scalars.rho_next(), scalars.rho(), r, s);
    return readback;
  };

  // Warmup: real iterations, then the state is reset by initialize() below.
  const auto warmup_start = Clock::now();
  result.setup_s = seconds(setup_start, warmup_start);
  WaitTimes discarded;
  initialize(discarded);
  for (int i = 0; i < options.warmup; ++i) {
    iterate(discarded).wait_and_throw();
    scalars.advance();
  }
  queue.wait_and_throw();
  result.warmup_s = seconds(warmup_start, Clock::now());

  // The timed solve, bounded like native aCG's "total solver time": idle
  // device, barrier, initial residual, loop, drained device.
  MPI_Barrier(MPI_COMM_WORLD);
  const auto solve_start = Clock::now();
  const double rho0 = initialize(result.waits);
  result.initial_residual = std::sqrt(rho0);
  result.rhs_norm = result.initial_residual;
  result.final_residual = result.initial_residual;
  const double atol = options.residual_absolute_tolerance;
  const double rtol = options.residual_relative_tolerance * result.initial_residual;
  const auto converged = [&](double residual) {
    return residual == 0.0 || (atol > 0.0 && residual < atol) || (rtol > 0.0 && residual < rtol);
  };
  result.converged = converged(result.final_residual);

  for (int k = 0; k < options.max_iterations && !result.converged; ++k) {
    sycl::event readback = iterate(result.waits);
    const auto wait_start = Clock::now();
    readback.wait_and_throw();
    result.waits.readback_s += seconds(wait_start, Clock::now());

    // Every rank holds the same reduced values, so every rank takes the same
    // branch: they stop, or fail, together.
    const double gamma = scalars.host_gamma();
    const double rho_next = scalars.host_rho_next();
    if (!std::isfinite(gamma) || !std::isfinite(rho_next)) {
      throw std::runtime_error("non-finite CG scalar (gamma or rho)");
    }
    if (gamma <= 0.0) {
      throw std::runtime_error("matrix is not positive definite under CG iteration");
    }
    result.iterations = k + 1;
    result.final_residual = std::sqrt(rho_next);
    result.converged = converged(result.final_residual);
    if (ctx.rank == 0 && options.log_every > 0 && result.iterations % options.log_every == 0) {
      std::cout << "iter=" << result.iterations << " residual=" << result.final_residual
                << " rel_residual=" << result.final_residual / result.initial_residual << '\n';
    }
    scalars.advance();
  }
  queue.wait_and_throw();
  const auto solve_end = Clock::now();
  result.solver_s = seconds(solve_start, solve_end);

  // Outside the timer from here on.
  result.relative_residual = result.initial_residual > 0.0 ? result.final_residual / result.initial_residual : 0.0;
  A.download(x, result.solution_local);
  if (!x_exact_local.empty()) {
    double error_squared = 0.0;
    double exact_squared = 0.0;
    for (std::size_t i = 0; i < x_exact_local.size(); ++i) {
      const double error = result.solution_local[i] - x_exact_local[i];
      error_squared += error * error;
      exact_squared += x_exact_local[i] * x_exact_local[i];
    }
    error_squared = allreduce_host(error_squared, MPI_SUM);
    exact_squared = allreduce_host(exact_squared, MPI_SUM);
    result.relative_solution_error = exact_squared > 0.0 ? std::sqrt(error_squared / exact_squared) : 0.0;
  }
  // Per iteration: SpMV 2 nnz, two dots 4 n, x/r update 4 n, s update 3 n.
  const double nnz = static_cast<double>(partition.interior_matrix.nnz() + partition.halo_matrix.nnz());
  const double local_flops =
      2.0 * static_cast<double>(n) + result.iterations * (2.0 * nnz + 11.0 * static_cast<double>(n));
  result.total_flops = allreduce_host(local_flops, MPI_SUM);
  result.solver_max_s = allreduce_host(result.solver_s, MPI_MAX);
  result.solver_min_s = allreduce_host(result.solver_s, MPI_MIN);
  result.gflops = result.solver_max_s > 0.0 ? result.total_flops / result.solver_max_s / 1.0e9 : 0.0;
  print_waits(result, ctx);
  result.post_solve_s = seconds(solve_end, Clock::now());
  return result;
}

} // namespace acg::solver
