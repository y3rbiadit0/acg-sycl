#include <chrono>
#include <exception>
#include <iostream>
#include <utility>

#include "acg/cli/parse_args.hpp"
#include "acg/matrix/matrix_market_reader.hpp"
#include "acg/runtime/queue_factory.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/cg_solver.hpp"

int main(int argc, char **argv) {
  try {
    const acg::cli::AppConfig config = acg::cli::parse_args(argc, argv);
    if (config.show_help) {
      acg::cli::print_usage(std::cout, argv[0]);
      return 0;
    }

    sycl::queue queue = acg::runtime::make_queue(config.device, config.enable_profiling);

    const auto matrix_load_start = std::chrono::steady_clock::now();
    const auto matrix = acg::matrix::read_matrix_market(config.matrix_path);
    const auto matrix_load_end = std::chrono::steady_clock::now();
    const double matrix_load_time_seconds =
        std::chrono::duration<double>(matrix_load_end - matrix_load_start).count();

    acg::runtime::RunContext ctx{
        .queue = std::move(queue),
        .rank = 0,
        .size = 1,
        .profiling_enabled = config.enable_profiling,
    };

    const auto &device = ctx.queue.get_device();
    std::cout << "device-kind: " << acg::runtime::to_string(config.device) << '\n';
    std::cout << "device-name: " << device.get_info<sycl::info::device::name>() << '\n';
    std::cout << "matrix: rows=" << matrix.rows
              << " cols=" << matrix.cols
              << " nnz=" << matrix.nnz()
              << " load_time=" << matrix_load_time_seconds << "s\n";


    

    const acg::solver::SolverResult result = acg::solver::run_cg(matrix, ctx, config.solver);

    std::cout << "solver: converged=" << (result.converged ? "true" : "false")
              << " iterations=" << result.iterations
              << " residual=" << result.final_residual
              << " relative_error=" << result.relative_solution_error
              << " flops=" << result.total_flops
              << " gflops=" << result.flop_rate_gflops
              << " time=" << result.solve_time_seconds << "s\n";
    return result.converged ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    acg::cli::print_usage(std::cerr, argv[0]);
    return 1;
  }
}
