#include "acg/solver/cg_solver.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>

#include "acg/matrix/distributed_csr_matrix.hpp"
#include "acg/solver/algorithms/cg_multi_gpu_mpi.hpp"
#include "acg/solver/algorithms/cg_single_gpu.hpp"
#include "acg/solver/backends/onemath_cuda_backend.hpp"
#include "acg/solver/rhs_builder.hpp"

namespace acg::solver {

namespace {

std::vector<double> slice_vector(
    const std::vector<double> &values,
    std::int64_t begin,
    std::int64_t end) {
  return std::vector<double>(
      values.begin() + static_cast<std::ptrdiff_t>(begin),
      values.begin() + static_cast<std::ptrdiff_t>(end));
}

std::vector<double> build_local_rhs(
    const acg::matrix::CsrMatrix<double> &matrix,
    std::int64_t row_begin,
    std::int64_t row_end,
    const std::vector<double> &x_exact_host,
    const std::vector<double> &default_b_host,
    bool manufactured_solution) {
  if (!manufactured_solution) {
    return slice_vector(default_b_host, row_begin, row_end);
  }

  std::vector<double> b_local(static_cast<std::size_t>(row_end - row_begin), 0.0);
  for (std::int64_t row = row_begin; row < row_end; ++row) {
    double sum = 0.0;
    for (std::int64_t offset = matrix.row_ptr[static_cast<std::size_t>(row)];
         offset < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
         ++offset) {
      sum += matrix.values[static_cast<std::size_t>(offset)]
             * x_exact_host[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(offset)])];
    }
    b_local[static_cast<std::size_t>(row - row_begin)] = sum;
  }
  return b_local;
}

} // namespace

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options) {
  sycl::queue q = ctx.queue;
  if (ctx.rank == 0) {
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  }

  if (matrix.rows <= 0 || matrix.cols <= 0) {
    throw std::runtime_error("matrix must be non-empty");
  }
  if (matrix.rows != matrix.cols) {
    throw std::runtime_error("CG requires a square matrix");
  }

  const RhsBuildResult rhs = build_rhs_inputs(static_cast<std::size_t>(matrix.rows), options);
  if (ctx.size > 1) {
    const acg::matrix::DistributedCsrMatrixPartition partition =
        acg::matrix::build_row_block_partition(matrix, ctx.rank, ctx.size);
    const std::vector<double> x_exact_local = slice_vector(rhs.x_exact_host, partition.local_row_begin, partition.local_row_end);
    const std::vector<double> b_local = build_local_rhs(
        matrix,
        partition.local_row_begin,
        partition.local_row_end,
        rhs.x_exact_host,
        rhs.b_host,
        options.manufactured_solution);
    return algorithms::run_cg_multi_gpu_mpi(partition, ctx, options, x_exact_local, b_local);
  }

  backends::OnemathCudaBackend backend(q, matrix);
  return algorithms::run_cg_single_gpu(backend, ctx, options, rhs.x_exact_host, rhs.b_host);
}

} // namespace acg::solver
