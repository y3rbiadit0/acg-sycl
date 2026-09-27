#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "acg/cli/parse_args.hpp"
#include "acg/matrix/matrix_market_reader.hpp"
#include "acg/reporting/solver_report.hpp"
#include "acg/runtime/mpi_runtime.hpp"
#include "acg/runtime/queue_factory.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/cg_solver.hpp"

namespace {

long positive_env(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && parsed > 0 ? parsed : 0;
}

// Rank 0 prints host and device of every rank, so a log shows the placement.
void print_placement(const acg::runtime::RunContext &ctx) {
  char host[MPI_MAX_PROCESSOR_NAME] = {};
  int host_length = 0;
  MPI_Get_processor_name(host, &host_length);
  const std::string line = "host=" + std::string(host, static_cast<std::size_t>(host_length)) +
                           " gpu=" + std::to_string(ctx.local_rank) +
                           " device=" + ctx.queue.get_device().get_info<sycl::info::device::name>();
  const int length = static_cast<int>(line.size());
  std::vector<int> lengths(ctx.rank == 0 ? static_cast<std::size_t>(ctx.size) : 0);
  MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> displs(lengths.size(), 0);
  for (std::size_t i = 1; i < lengths.size(); ++i) {
    displs[i] = displs[i - 1] + lengths[i - 1];
  }
  std::string all(ctx.rank == 0 ? static_cast<std::size_t>(displs.back() + lengths.back()) : 0, '\0');
  MPI_Gatherv(line.data(), length, MPI_CHAR, all.data(), lengths.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
  for (int r = 0; r < static_cast<int>(lengths.size()); ++r) {
    std::cout << "rank " << r << ": " << all.substr(static_cast<std::size_t>(displs[r]), static_cast<std::size_t>(lengths[r]))
              << '\n';
  }
}

// ACG_LOG_LOADED_LIBS: every shared object mapped into this process, read after
// the solve so that libraries loaded at run time -- cuSPARSE and cuBLAS behind
// oneMath, NCCL behind oneCCL, UCX transports -- are included. Linux only.
void print_loaded_libraries() {
  std::ifstream maps("/proc/self/maps");
  std::set<std::string> libraries;
  std::string line;
  while (std::getline(maps, line)) {
    const std::size_t path = line.find('/');
    if (path != std::string::npos && line.find(".so", path) != std::string::npos) {
      libraries.insert(line.substr(path));
    }
  }
  for (const std::string &library : libraries) {
    std::cout << "loaded_library: " << library << '\n';
  }
}

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

    acg::runtime::RunContext ctx{
        .queue = acg::runtime::make_queue(
            config.device,
            config.enable_profiling,
            config.device == acg::runtime::DeviceKind::GPU ? mpi_info.local_rank : 0),
        .rank = mpi_info.rank,
        .size = mpi_info.size,
        .local_rank = mpi_info.local_rank,
        .profiling_enabled = config.enable_profiling,
    };
    const bool root = ctx.rank == 0;
    if (root) {
      std::cout << "device-kind: " << acg::runtime::to_string(config.device) << '\n'
                << "ranks: " << ctx.size << '\n'
                << "solver-collectives: " << acg::solver::to_string(config.solver.solver_collectives) << '\n';
    }
    print_placement(ctx);

    if (const long probe = positive_env("ACG_COLLECTIVE_PROBE"); probe > 0) {
      acg::solver::run_collective_probe(ctx, config.solver, probe);
    }

    const auto load_start = std::chrono::steady_clock::now();
    const auto matrix = acg::matrix::read_matrix_market(config.matrix_path);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
    if (root) {
      std::cout << "stopping: residual_rtol=" << config.solver.residual_relative_tolerance
                << " residual_atol=" << config.solver.residual_absolute_tolerance
                << " max_iters=" << config.solver.max_iterations << " warmup=" << config.solver.warmup
                << " manufactured=" << (config.solver.manufactured_solution ? "true" : "false") << '\n'
                << "matrix: rows=" << matrix.rows << " cols=" << matrix.cols << " nnz=" << matrix.nnz()
                << " load_time=" << load_s << "s\n";
    }

    const acg::solver::SolverResult result = acg::solver::run_cg(matrix, ctx, config.solver);
    if (root) {
      acg::reporting::print_solver_report(std::cout, result);
      if (positive_env("ACG_LOG_LOADED_LIBS") > 0) {
        print_loaded_libraries();
      }
    }
    return result.converged ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error (rank " << error_rank << "): " << e.what() << '\n';
    // A rank that fails alone would leave the others blocked in MPI until the
    // walltime runs out; take the whole job down instead.
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}
