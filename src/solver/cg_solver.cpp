#include "acg/solver/cg_solver.hpp"

#include <iostream>
#include <stdexcept>

#include "acg/solver/algorithms/cg_single_gpu.hpp"
#include "acg/solver/backends/onemath_cuda_backend.hpp"
#include "acg/solver/rhs_builder.hpp"

namespace acg::solver {

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options) {
  sycl::queue q = ctx.queue;
  std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  if (matrix.rows <= 0 || matrix.cols <= 0) {
    throw std::runtime_error("matrix must be non-empty");
  }
  if (matrix.rows != matrix.cols) {
    throw std::runtime_error("CG requires a square matrix");
  }

  const RhsBuildResult rhs = build_rhs_inputs(static_cast<std::size_t>(matrix.rows), options);
  backends::OnemathCudaBackend backend(q, matrix);
  return algorithms::run_cg_single_gpu(backend, options, rhs.x_exact_host, rhs.b_host);
}

} // namespace acg::solver
