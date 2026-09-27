#include "acg/solver/cg_solver.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include <mpi.h>

#include "acg/matrix/distributed_csr_matrix.hpp"
#include "acg/solver/rhs_builder.hpp"
#include "cg.hpp"

namespace acg::solver {

namespace {

using Clock = std::chrono::steady_clock;

// Every rank reads the whole matrix, so the right-hand side is built on the
// host for this rank's rows: b = A x_exact for a manufactured solution (as
// native aCG does), else the default b of all ones.
std::vector<double> local_rhs(
    const acg::matrix::CsrMatrix<double> &matrix,
    const std::vector<std::int64_t> &rows,
    const RhsBuildResult &rhs,
    bool manufactured) {
  std::vector<double> b(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = static_cast<std::size_t>(rows[i]);
    if (!manufactured) {
      b[i] = rhs.b_host[row];
      continue;
    }
    double sum = 0.0;
    for (std::int64_t k = matrix.row_ptr[row]; k < matrix.row_ptr[row + 1]; ++k) {
      sum += matrix.values[static_cast<std::size_t>(k)] *
             rhs.x_exact_host[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(k)])];
    }
    b[i] = sum;
  }
  return b;
}

// ||b - A x|| / ||b|| from the original host matrix and the gathered solution.
// Shares nothing with the device path -- no partition, no halo, no oneMath --
// so a bug there cannot also hide itself here.
void validate(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const std::vector<std::int64_t> &rows,
    const std::vector<double> &b_local,
    SolverResult &result) {
  const auto start = Clock::now();
  const int local_count = static_cast<int>(rows.size());
  std::vector<int> counts(static_cast<std::size_t>(ctx.size));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> displs(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  const auto total = static_cast<std::size_t>(displs.back() + counts.back());
  std::vector<double> all_values(total);
  std::vector<std::int64_t> all_rows(total);
  MPI_Allgatherv(result.solution_local.data(), local_count, MPI_DOUBLE, all_values.data(), counts.data(),
                 displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgatherv(rows.data(), local_count, MPI_INT64_T, all_rows.data(), counts.data(), displs.data(), MPI_INT64_T,
                 MPI_COMM_WORLD);
  std::vector<double> x(static_cast<std::size_t>(matrix.cols), 0.0);
  for (std::size_t i = 0; i < total; ++i) {
    x[static_cast<std::size_t>(all_rows[i])] = all_values[i];
  }

  double sums[2] = {0.0, 0.0}; // ||b - A x||^2, ||b||^2 over this rank's rows
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = static_cast<std::size_t>(rows[i]);
    double ax = 0.0;
    for (std::int64_t k = matrix.row_ptr[row]; k < matrix.row_ptr[row + 1]; ++k) {
      ax += matrix.values[static_cast<std::size_t>(k)] *
            x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(k)])];
    }
    sums[0] += (b_local[i] - ax) * (b_local[i] - ax);
    sums[1] += b_local[i] * b_local[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  result.true_residual = std::sqrt(sums[0]);
  // A zero right-hand side has x = 0 as its solution: report the absolute
  // residual rather than divide by zero.
  result.true_relative_residual = sums[1] > 0.0 ? result.true_residual / std::sqrt(sums[1]) : result.true_residual;
  result.validation_s = std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options) {
  if (matrix.rows <= 0 || matrix.rows != matrix.cols) {
    throw std::runtime_error("CG requires a non-empty square matrix");
  }
  const auto setup_start = Clock::now();

  // One rank is a one-part partition: all rows interior, no halo.
  acg::matrix::PartitionOptions partition_options = acg::matrix::partition_options_from_environment();
  if (!options.partition_path.empty()) {
    // --partition names a decomposition outright; an ACG_PARTITIONER asking for
    // something else is a contradiction, not a preference.
    if (partition_options.method != acg::matrix::PartitionMethod::RowBlock &&
        partition_options.method != acg::matrix::PartitionMethod::File) {
      throw std::runtime_error("--partition was given but ACG_PARTITIONER asks for a different partitioner");
    }
    partition_options.method = acg::matrix::PartitionMethod::File;
    partition_options.path = options.partition_path;
  }
  if (ctx.size == 1) {
    partition_options = acg::matrix::PartitionOptions{};
  }
  const acg::matrix::DistributedCsrMatrixPartition partition =
      acg::matrix::build_distributed_partition(matrix, ctx.rank, ctx.size, partition_options);

  const RhsBuildResult rhs = build_rhs_inputs(static_cast<std::size_t>(matrix.rows), options);
  const std::vector<double> b_local = local_rhs(matrix, partition.local_global_rows, rhs, options.manufactured_solution);
  std::vector<double> x_exact_local;
  if (options.manufactured_solution) {
    for (const std::int64_t row : partition.local_global_rows) {
      x_exact_local.push_back(rhs.x_exact_host[static_cast<std::size_t>(row)]);
    }
  }
  const double partition_s = std::chrono::duration<double>(Clock::now() - setup_start).count();

  SolverResult result = solve_cg(partition, ctx, options, b_local, x_exact_local);
  result.setup_s += partition_s;
  validate(matrix, ctx, partition.local_global_rows, b_local, result);
  return result;
}

} // namespace acg::solver
