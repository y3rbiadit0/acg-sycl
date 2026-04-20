#include "acg/solver/cg_solver.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <oneapi/math.hpp>
#include <oneapi/math/sparse_blas.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace acg::solver {

namespace {

std::vector<double> generate_exact_solution(std::size_t size, const SolverOptions &options) {
  std::vector<double> x_exact(size, 1.0);

  if (!options.manufactured_solution) {
    return x_exact;
  }

  std::mt19937 generator(options.seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  double norm_squared = 0.0;

  for (double &value : x_exact) {
    value = distribution(generator);
    norm_squared += value * value;
  }

  const double norm = std::sqrt(norm_squared);
  if (norm == 0.0) {
    throw std::runtime_error("manufactured solution generation produced a zero vector");
  }

  for (double &value : x_exact) {
    value /= norm;
  }

  return x_exact;
}

template <typename T>
struct UsmDeleter {
  sycl::queue *queue = nullptr;

  void operator()(T *memory) const {
    if (memory != nullptr) {
      sycl::free(memory, *queue);
    }
  }
};

template <typename T>
using UsmPtr = std::unique_ptr<T, UsmDeleter<T>>;

template <typename T>
UsmPtr<T> make_device_array(sycl::queue &q, std::size_t count) {
  T *ptr = sycl::malloc_device<T>(count, q);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device memory");
  }
  return UsmPtr<T>(ptr, UsmDeleter<T>{&q});
}

template <typename T>
UsmPtr<T> make_shared_scalar(sycl::queue &q, T initial_value = T{}) {
  T *ptr = sycl::malloc_shared<T>(1, q);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate shared scalar");
  }
  *ptr = initial_value;
  return UsmPtr<T>(ptr, UsmDeleter<T>{&q});
}

void copy_to_device(sycl::queue &q, const void *source, void *destination, std::size_t bytes) {
  q.memcpy(destination, source, bytes).wait();
}

void copy_to_host(sycl::queue &q, const void *source, void *destination, std::size_t bytes) {
  q.memcpy(destination, source, bytes).wait();
}

double device_dot(sycl::queue &q, double *lhs, double *rhs, std::int64_t size, double *result) {
  oneapi::math::blas::column_major::dot(q, size, lhs, 1, rhs, 1, result).wait();
  return *result;
}

void initialize_iteration_vectors(sycl::queue &q, double *b, double *x, double *r, double *p, std::int64_t size) {
  q.fill(x, 0.0, static_cast<std::size_t>(size)).wait();
  oneapi::math::blas::column_major::copy(q, size, b, 1, r, 1).wait();
  oneapi::math::blas::column_major::copy(q, size, b, 1, p, 1).wait();
}

void update_x_and_r(sycl::queue &q, double alpha, double *p, double *Ap, double *x, double *r, std::int64_t size) {
  oneapi::math::blas::column_major::axpy(q, size, alpha, p, 1, x, 1).wait();
  oneapi::math::blas::column_major::axpy(q, size, -alpha, Ap, 1, r, 1).wait();
}

void update_p(sycl::queue &q, double beta, double *r, double *p, std::int64_t size) {
  oneapi::math::blas::column_major::scal(q, size, beta, p, 1).wait();
  oneapi::math::blas::column_major::axpy(q, size, 1.0, r, 1, p, 1).wait();
}

void sparse_spmv(
    sycl::queue &q,
    oneapi::math::sparse::matrix_handle_t matrix_handle,
    oneapi::math::sparse::dense_vector_handle_t x_handle,
    oneapi::math::sparse::dense_vector_handle_t y_handle,
    oneapi::math::sparse::spmv_descr_t spmv_descr,
    const oneapi::math::sparse::matrix_view &view,
    const double &alpha,
    const double &beta) {
  std::size_t workspace_size = 0;
  oneapi::math::sparse::spmv_buffer_size(
      q,
      oneapi::math::transpose::nontrans,
      &alpha,
      view,
      matrix_handle,
      x_handle,
      &beta,
      y_handle,
      oneapi::math::sparse::spmv_alg::default_alg,
      spmv_descr,
      workspace_size);

  void *workspace = nullptr;
  if (workspace_size > 0) {
    workspace = sycl::malloc_device<std::uint8_t>(workspace_size, q);
    if (workspace == nullptr) {
      throw std::runtime_error("failed to allocate oneMath SPMV workspace");
    }

    oneapi::math::sparse::spmv_optimize(
        q,
        oneapi::math::transpose::nontrans,
        &alpha,
        view,
        matrix_handle,
        x_handle,
        &beta,
        y_handle,
        oneapi::math::sparse::spmv_alg::default_alg,
        spmv_descr,
        workspace)
        .wait();
  }

  oneapi::math::sparse::spmv(
      q,
      oneapi::math::transpose::nontrans,
      &alpha,
      view,
      matrix_handle,
      x_handle,
      &beta,
      y_handle,
      oneapi::math::sparse::spmv_alg::default_alg,
      spmv_descr)
      .wait();

  if (workspace != nullptr) {
    sycl::free(workspace, q);
  }
}

} // namespace

SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options) {
  const auto start = std::chrono::steady_clock::now();

  sycl::queue q = ctx.queue;
  std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  if (matrix.rows <= 0 || matrix.cols <= 0) {
    throw std::runtime_error("matrix must be non-empty");
  }
  if (matrix.rows != matrix.cols) {
    throw std::runtime_error("CG requires a square matrix");
  }

  constexpr double one = 1.0;
  constexpr double zero = 0.0;
  const std::int64_t size = matrix.rows;
  const std::size_t vector_size = static_cast<std::size_t>(size);
  const double matrix_vector_flops = 2.0 * static_cast<double>(matrix.nnz());
  const double dot_flops = 2.0 * static_cast<double>(vector_size);
  const double update_x_r_flops = 4.0 * static_cast<double>(vector_size);
  const double update_p_flops = 2.0 * static_cast<double>(vector_size);

  const std::vector<double> x_exact_host = generate_exact_solution(vector_size, options);
  std::vector<std::int64_t> col_idx_host(matrix.col_idx.begin(), matrix.col_idx.end());
  std::vector<double> b_host(vector_size, 0.0);
  if (!options.manufactured_solution) {
    std::fill(b_host.begin(), b_host.end(), 1.0);
  }

  auto row_ptr = make_device_array<std::int64_t>(q, matrix.row_ptr.size());
  auto col_idx = make_device_array<std::int64_t>(q, col_idx_host.size());
  auto values = make_device_array<double>(q, matrix.values.size());
  auto x_exact = make_device_array<double>(q, vector_size);
  auto x = make_device_array<double>(q, vector_size);
  auto b = make_device_array<double>(q, vector_size);
  auto r = make_device_array<double>(q, vector_size);
  auto p = make_device_array<double>(q, vector_size);
  auto Ap = make_device_array<double>(q, vector_size);
  auto dot_result = make_shared_scalar<double>(q, 0.0);

  copy_to_device(q, matrix.row_ptr.data(), row_ptr.get(), matrix.row_ptr.size() * sizeof(std::int64_t));
  copy_to_device(q, col_idx_host.data(), col_idx.get(), col_idx_host.size() * sizeof(std::int64_t));
  copy_to_device(q, matrix.values.data(), values.get(), matrix.values.size() * sizeof(double));
  copy_to_device(q, x_exact_host.data(), x_exact.get(), vector_size * sizeof(double));
  copy_to_device(q, b_host.data(), b.get(), vector_size * sizeof(double));

  oneapi::math::sparse::matrix_handle_t matrix_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t x_exact_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t b_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t r_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t p_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t Ap_handle = nullptr;
  oneapi::math::sparse::spmv_descr_t spmv_descr = nullptr;
  const oneapi::math::sparse::matrix_view view(oneapi::math::sparse::matrix_descr::general);

  oneapi::math::sparse::init_csr_matrix(
      q, &matrix_handle, matrix.rows, matrix.cols, matrix.nnz(), oneapi::math::index_base::zero, row_ptr.get(),
      col_idx.get(), values.get());
  oneapi::math::sparse::init_dense_vector(q, &x_exact_handle, size, x_exact.get());
  oneapi::math::sparse::init_dense_vector(q, &x_handle, size, x.get());
  oneapi::math::sparse::init_dense_vector(q, &b_handle, size, b.get());
  oneapi::math::sparse::init_dense_vector(q, &r_handle, size, r.get());
  oneapi::math::sparse::init_dense_vector(q, &p_handle, size, p.get());
  oneapi::math::sparse::init_dense_vector(q, &Ap_handle, size, Ap.get());
  oneapi::math::sparse::init_spmv_descr(q, &spmv_descr);

  if (options.manufactured_solution) {
    sparse_spmv(q, matrix_handle, x_exact_handle, b_handle, spmv_descr, view, one, zero);
  }

  initialize_iteration_vectors(q, b.get(), x.get(), r.get(), p.get(), size);
  double total_flops = options.manufactured_solution ? matrix_vector_flops : 0.0;

  const double b_norm = std::sqrt(device_dot(q, b.get(), b.get(), size, dot_result.get()));
  total_flops += dot_flops;
  double rho = device_dot(q, r.get(), r.get(), size, dot_result.get());
  total_flops += dot_flops;
  const double r0_norm = std::sqrt(rho);
  const double scaled_residual_relative_tolerance = options.residual_relative_tolerance * r0_norm;
  const double scaled_diff_relative_tolerance = options.diff_relative_tolerance * 0.0;

  SolverResult result;
  result.final_residual = r0_norm;
  result.converged =
      (options.residual_absolute_tolerance > 0.0 && result.final_residual < options.residual_absolute_tolerance) ||
      (options.residual_relative_tolerance > 0.0 && result.final_residual < scaled_residual_relative_tolerance);

  for (int iteration = 0; iteration < options.max_iterations && !result.converged; ++iteration) {
    sparse_spmv(q, matrix_handle, p_handle, Ap_handle, spmv_descr, view, one, zero);
    total_flops += matrix_vector_flops;

    const double pAp = device_dot(q, p.get(), Ap.get(), size, dot_result.get());
    total_flops += dot_flops;
    if (pAp <= 0.0) {
      throw std::runtime_error("matrix is not positive definite under CG iteration");
    }

    const double alpha = rho / pAp;
    const double p_norm = std::sqrt(device_dot(q, p.get(), p.get(), size, dot_result.get()));
    total_flops += dot_flops;
    const double dx_norm = std::abs(alpha) * p_norm;
    update_x_and_r(q, alpha, p.get(), Ap.get(), x.get(), r.get(), size);
    total_flops += update_x_r_flops;

    const double rho_next = device_dot(q, r.get(), r.get(), size, dot_result.get());
    total_flops += dot_flops;
    result.iterations = iteration + 1;
    result.final_residual = std::sqrt(rho_next);
    result.converged =
        (options.diff_absolute_tolerance > 0.0 && dx_norm < options.diff_absolute_tolerance) ||
        (options.diff_relative_tolerance > 0.0 && dx_norm < scaled_diff_relative_tolerance) ||
        (options.residual_absolute_tolerance > 0.0 && result.final_residual < options.residual_absolute_tolerance) ||
        (options.residual_relative_tolerance > 0.0 && result.final_residual < scaled_residual_relative_tolerance);
    if (result.converged) {
      break;
    }

    const double beta = rho_next / rho;
    update_p(q, beta, r.get(), p.get(), size);
    total_flops += update_p_flops;
    rho = rho_next;
  }

  std::vector<double> x_host(vector_size, 0.0);
  copy_to_host(q, x.get(), x_host.data(), vector_size * sizeof(double));

  oneapi::math::sparse::release_spmv_descr(q, spmv_descr).wait();
  oneapi::math::sparse::release_dense_vector(q, Ap_handle).wait();
  oneapi::math::sparse::release_dense_vector(q, p_handle).wait();
  oneapi::math::sparse::release_dense_vector(q, r_handle).wait();
  oneapi::math::sparse::release_dense_vector(q, b_handle).wait();
  oneapi::math::sparse::release_dense_vector(q, x_handle).wait();
  oneapi::math::sparse::release_dense_vector(q, x_exact_handle).wait();
  oneapi::math::sparse::release_sparse_matrix(q, matrix_handle).wait();

  if (options.manufactured_solution) {
    std::vector<double> x_error(vector_size, 0.0);
    double x_exact_norm_squared = 0.0;
    double x_error_norm_squared = 0.0;
    for (std::size_t i = 0; i < vector_size; ++i) {
      x_error[i] = x_host[i] - x_exact_host[i];
      x_exact_norm_squared += x_exact_host[i] * x_exact_host[i];
      x_error_norm_squared += x_error[i] * x_error[i];
    }
    const double x_exact_norm = std::sqrt(x_exact_norm_squared);
    result.relative_solution_error = x_exact_norm > 0.0 ? std::sqrt(x_error_norm_squared) / x_exact_norm : 0.0;
  } else {
    result.relative_solution_error = 0.0;
  }

  const auto end = std::chrono::steady_clock::now();
  result.solve_time_seconds = std::chrono::duration<double>(end - start).count();
  result.total_flops = total_flops;
  result.flop_rate_gflops = result.solve_time_seconds > 0.0 ? total_flops / result.solve_time_seconds / 1.0e9 : 0.0;
  return result;
}

} // namespace acg::solver
