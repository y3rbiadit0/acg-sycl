#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include "acg/cli/parse_args.hpp"
#include "acg/matrix/matrix_market_reader.hpp"
#include "acg/reporting/solver_report.hpp"
#include "acg/runtime/mpi_runtime.hpp"
#include "acg/runtime/queue_factory.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/cg_solver.hpp"

namespace {

#ifdef ACG_HAVE_MPI
const char *env_or_unset(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr ? value : "unset";
}

long reduction_probe_iterations() {
  const char *value = std::getenv("ACG_REDUCTION_PROBE");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && parsed > 0 ? parsed : 0;
}

void print_probe_stats(const char *name, double local_us_per_op, int rank) {
  double min_us = 0.0;
  double max_us = 0.0;
  double sum_us = 0.0;
  MPI_Reduce(&local_us_per_op, &min_us, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_us_per_op, &max_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_us_per_op, &sum_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "reduction_probe: " << name
              << " min_us_per_op=" << min_us
              << " avg_us_per_op=" << (sum_us / static_cast<double>(size))
              << " max_us_per_op=" << max_us << '\n';
  }
}

void run_reduction_probe(acg::runtime::RunContext &ctx, const acg::cli::AppConfig &config) {
  const long iterations = reduction_probe_iterations();
  if (iterations <= 0 || ctx.size <= 1) {
    return;
  }

  const auto &device = ctx.queue.get_device();
  if (ctx.rank == 0) {
    std::cout << "reduction_probe: iterations=" << iterations
              << " ranks=" << ctx.size
              << " device-kind=" << acg::runtime::to_string(config.device)
              << " mpi-mode=" << acg::solver::to_string(config.solver.mpi_mode) << '\n';
    std::cout << "reduction_probe_env: CUDA_VISIBLE_DEVICES=" << env_or_unset("CUDA_VISIBLE_DEVICES")
              << " OMPI_MCA_pml=" << env_or_unset("OMPI_MCA_pml")
              << " OMPI_MCA_btl=" << env_or_unset("OMPI_MCA_btl")
              << " OMPI_MCA_coll_tuned_allreduce_algorithm=" << env_or_unset("OMPI_MCA_coll_tuned_allreduce_algorithm")
              << " UCX_TLS=" << env_or_unset("UCX_TLS")
              << " UCX_RNDV_THRESH=" << env_or_unset("UCX_RNDV_THRESH")
              << " UCX_MAX_RNDV_RAILS=" << env_or_unset("UCX_MAX_RNDV_RAILS") << '\n';
  }

  char hostname[MPI_MAX_PROCESSOR_NAME + 1] = {};
  int hostname_len = 0;
  MPI_Get_processor_name(hostname, &hostname_len);
  const std::string device_name = device.get_info<sycl::info::device::name>();
  for (int r = 0; r < ctx.size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank == r) {
      std::cout << "reduction_probe_rank: rank=" << ctx.rank
                << " local_rank=" << ctx.local_rank
                << " host=" << hostname
                << " cuda_visible_devices=" << env_or_unset("CUDA_VISIBLE_DEVICES")
                << " ompi_local_rank=" << env_or_unset("OMPI_COMM_WORLD_LOCAL_RANK")
                << " device=" << device_name << '\n';
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int warmup = 100;
  double host_send = static_cast<double>(ctx.rank + 1);
  double host_recv = 0.0;
  for (int i = 0; i < warmup; ++i) {
    MPI_Allreduce(&host_send, &host_recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::steady_clock::now();
  for (long i = 0; i < iterations; ++i) {
    MPI_Allreduce(&host_send, &host_recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  auto end = std::chrono::steady_clock::now();
  print_probe_stats(
      "host_allreduce",
      std::chrono::duration<double>(end - start).count() * 1.0e6 / static_cast<double>(iterations),
      ctx.rank);

  double *device_send = sycl::malloc_device<double>(1, ctx.queue);
  double *device_recv = sycl::malloc_device<double>(1, ctx.queue);
  if (device_send == nullptr || device_recv == nullptr) {
    sycl::free(device_send, ctx.queue);
    sycl::free(device_recv, ctx.queue);
    throw std::runtime_error("failed to allocate reduction probe device scalars");
  }
  ctx.queue.memcpy(device_send, &host_send, sizeof(double)).wait();
  ctx.queue.memcpy(device_recv, &host_recv, sizeof(double)).wait();

  for (int i = 0; i < warmup; ++i) {
    MPI_Allreduce(device_send, device_recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::steady_clock::now();
  for (long i = 0; i < iterations; ++i) {
    MPI_Allreduce(device_send, device_recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  end = std::chrono::steady_clock::now();
  print_probe_stats(
      "device_allreduce",
      std::chrono::duration<double>(end - start).count() * 1.0e6 / static_cast<double>(iterations),
      ctx.rank);

  double readback = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::steady_clock::now();
  for (long i = 0; i < iterations; ++i) {
    MPI_Allreduce(device_send, device_recv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ctx.queue.memcpy(&readback, device_recv, sizeof(double)).wait();
  }
  end = std::chrono::steady_clock::now();
  print_probe_stats(
      "device_allreduce_readback",
      std::chrono::duration<double>(end - start).count() * 1.0e6 / static_cast<double>(iterations),
      ctx.rank);

  sycl::free(device_recv, ctx.queue);
  sycl::free(device_send, ctx.queue);
  MPI_Barrier(MPI_COMM_WORLD);
}
#else
void run_reduction_probe(acg::runtime::RunContext &, const acg::cli::AppConfig &) {}
#endif

} // namespace

int main(int argc, char **argv) {
  int error_rank = 0;
  try {
    acg::runtime::ScopedMpiSession mpi_session(argc, argv);
    const acg::runtime::MpiRuntimeInfo mpi_info = mpi_session.info();
    error_rank = mpi_info.rank;
    const acg::cli::AppConfig config = acg::cli::parse_args(argc, argv);
    if (config.show_help) {
      acg::cli::print_usage(std::cout, argv[0]);
      return 0;
    }

    sycl::queue queue = acg::runtime::make_queue(
        config.device,
        config.enable_profiling,
        config.device == acg::runtime::DeviceKind::GPU ? mpi_info.local_rank : 0);

    acg::runtime::RunContext ctx{
        .queue = std::move(queue),
        .rank = mpi_info.rank,
        .size = mpi_info.size,
        .local_rank = mpi_info.local_rank,
        .profiling_enabled = config.enable_profiling,
    };
    const bool is_root_rank = ctx.rank == 0;

    const auto &device = ctx.queue.get_device();
    if (is_root_rank) {
      std::cout << "device-kind: " << acg::runtime::to_string(config.device) << '\n';
      std::cout << "ranks: " << ctx.size << '\n';
      std::cout << "mpi-mode: " << acg::solver::to_string(config.solver.mpi_mode) << '\n';
    }

#ifdef ACG_HAVE_MPI
    {
      const std::string local_device_name = device.get_info<sycl::info::device::name>();
      char hostname[MPI_MAX_PROCESSOR_NAME + 1] = {};
      int hostname_len = 0;
      MPI_Get_processor_name(hostname, &hostname_len);

      if (is_root_rank) {
        std::cout << "rank " << ctx.rank << ": host=" << hostname
                  << " gpu=" << ctx.local_rank
                  << " device=" << local_device_name << '\n';
        for (int r = 1; r < ctx.size; ++r) {
          int len = 0;
          MPI_Recv(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::string peer_device(static_cast<std::size_t>(len), '\0');
          MPI_Recv(peer_device.data(), len, MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          int peer_local_rank = 0;
          MPI_Recv(&peer_local_rank, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          int hlen = 0;
          MPI_Recv(&hlen, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::string peer_host(static_cast<std::size_t>(hlen), '\0');
          MPI_Recv(peer_host.data(), hlen, MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::cout << "rank " << r << ": host=" << peer_host
                    << " gpu=" << peer_local_rank
                    << " device=" << peer_device << '\n';
        }
      } else {
        int len = static_cast<int>(local_device_name.size());
        MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_device_name.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&ctx.local_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int hlen = hostname_len;
        MPI_Send(&hlen, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(hostname, hlen, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    if (is_root_rank) {
      std::cout << "rank 0: gpu=0 device=" << device.get_info<sycl::info::device::name>() << '\n';
    }
#endif

    run_reduction_probe(ctx, config);

    const auto matrix_load_start = std::chrono::steady_clock::now();
    const auto matrix = acg::matrix::read_matrix_market(config.matrix_path);
    const auto matrix_load_end = std::chrono::steady_clock::now();
    const double matrix_load_time_seconds =
        std::chrono::duration<double>(matrix_load_end - matrix_load_start).count();

    if (is_root_rank) {
      std::cout << "stopping: residual_rtol=" << config.solver.residual_relative_tolerance
                << " residual_atol=" << config.solver.residual_absolute_tolerance
                << " diff_rtol=" << config.solver.diff_relative_tolerance
                << " diff_atol=" << config.solver.diff_absolute_tolerance
                << " solution_rtol=" << config.solver.solution_relative_tolerance
                << " log_every=" << config.solver.log_every
                << " manufactured=" << (config.solver.manufactured_solution ? "true" : "false") << '\n';
      std::cout << "matrix: rows=" << matrix.rows
                << " cols=" << matrix.cols
                << " nnz=" << matrix.nnz()
                << " load_time=" << matrix_load_time_seconds << "s\n";
    }


    

    const acg::solver::SolverResult result = acg::solver::run_cg(matrix, ctx, config.solver);
    if (is_root_rank) {
      acg::reporting::print_solver_report(
          std::cout,
          result,
          matrix,
          config,
          acg::reporting::solver_report_options_from_environment());
    }

    return result.converged ? 0 : 1;
  } catch (const std::exception &e) {
    if (error_rank == 0) {
      std::cerr << "error: " << e.what() << '\n';
      acg::cli::print_usage(std::cerr, argv[0]);
    }
    return 1;
  }
}
