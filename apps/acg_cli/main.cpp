#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <utility>

#include "acg/cli/parse_args.hpp"
#include "acg/matrix/matrix_market_reader.hpp"
#include "acg/runtime/mpi_runtime.hpp"
#include "acg/runtime/queue_factory.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/cg_solver.hpp"
#include "acg/solver/perf_breakdown.hpp"

int main(int argc, char **argv) {
  try {
    acg::runtime::ScopedMpiSession mpi_session(argc, argv);
    const acg::runtime::MpiRuntimeInfo mpi_info = mpi_session.info();
    const acg::cli::AppConfig config = acg::cli::parse_args(argc, argv);
    if (config.show_help) {
      acg::cli::print_usage(std::cout, argv[0]);
      return 0;
    }

    sycl::queue queue = acg::runtime::make_queue(
        config.device,
        config.enable_profiling,
        config.device == acg::runtime::DeviceKind::GPU ? mpi_info.local_rank : 0);

    const auto matrix_load_start = std::chrono::steady_clock::now();
    const auto matrix = acg::matrix::read_matrix_market(config.matrix_path);
    const auto matrix_load_end = std::chrono::steady_clock::now();
    const double matrix_load_time_seconds =
        std::chrono::duration<double>(matrix_load_end - matrix_load_start).count();

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
      std::cout << "device-name(rank0): " << device.get_info<sycl::info::device::name>() << '\n';
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
      std::cout << "solver: converged=" << (result.converged ? "true" : "false")
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

    const auto print_op = [](const char *name, const acg::solver::OpStats &s) {
      std::cout << std::fixed << std::setprecision(6)
                << name << ": "
                << s.time_seconds << " seconds/proc "
                << s.count << " times/proc "
                << s.bytes << " B/proc "
                << std::setprecision(3) << s.bandwidth_gbs() << " GB/s/proc\n";
    };
    if (is_root_rank) {
      print_op("spmv", result.perf.spmv);
      print_op("dot",  result.perf.dot);
      print_op("nrm2", result.perf.nrm2);
      print_op("axpy", result.perf.axpy);
      print_op("copy", result.perf.copy);
      print_op("pack", result.perf.pack);
      print_op("p2p", result.perf.p2p);
      print_op("allreduce", result.perf.allreduce);
      print_op("host_sync", result.perf.host_sync);
    }

    return result.converged ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    acg::cli::print_usage(std::cerr, argv[0]);
    return 1;
  }
}
