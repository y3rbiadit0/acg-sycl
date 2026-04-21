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

namespace acg::solver
{

  namespace
  {

    std::vector<double> generate_exact_solution(std::size_t size, const SolverOptions &options)
    {
      std::vector<double> x_exact(size, 1.0);

      if (!options.manufactured_solution)
      {
        return x_exact;
      }

      std::mt19937 generator(options.seed);
      std::uniform_real_distribution<double> distribution(-1.0, 1.0);
      double norm_squared = 0.0;

      for (double &value : x_exact)
      {
        value = distribution(generator);
        norm_squared += value * value;
      }

      const double norm = std::sqrt(norm_squared);
      if (norm == 0.0)
      {
        throw std::runtime_error("manufactured solution generation produced a zero vector");
      }

      for (double &value : x_exact)
      {
        value /= norm;
      }

      return x_exact;
    }

    template <typename T>
    struct UsmDeleter
    {
      sycl::queue *queue = nullptr;

      void operator()(T *memory) const
      {
        if (memory != nullptr)
        {
          sycl::free(memory, *queue);
        }
      }
    };

    template <typename T>
    using UsmPtr = std::unique_ptr<T, UsmDeleter<T>>;

    template <typename T>
    UsmPtr<T> make_device_array(sycl::queue &q, std::size_t count)
    {
      T *ptr = sycl::malloc_device<T>(count, q);
      if (ptr == nullptr)
      {
        throw std::runtime_error("failed to allocate device memory");
      }
      return UsmPtr<T>(ptr, UsmDeleter<T>{&q});
    }

    template <typename T>
    UsmPtr<T> make_shared_scalar(sycl::queue &q, T initial_value = T{})
    {
      T *ptr = sycl::malloc_shared<T>(1, q);
      if (ptr == nullptr)
      {
        throw std::runtime_error("failed to allocate shared scalar");
      }
      *ptr = initial_value;
      return UsmPtr<T>(ptr, UsmDeleter<T>{&q});
    }

    void copy_to_device(sycl::queue &q, const void *source, void *destination, std::size_t bytes)
    {
      q.memcpy(destination, source, bytes).wait();
    }

    void copy_to_host(sycl::queue &q, const void *source, void *destination, std::size_t bytes)
    {
      q.memcpy(destination, source, bytes).wait();
    }

  } // namespace

  SolverResult run_cg(
      const acg::matrix::CsrMatrix<double> &matrix,
      const acg::runtime::RunContext &ctx,
      const SolverOptions &options)
  {
    const auto start = std::chrono::steady_clock::now();

    sycl::queue q = ctx.queue;
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    if (matrix.rows <= 0 || matrix.cols <= 0)
    {
      throw std::runtime_error("matrix must be non-empty");
    }
    if (matrix.rows != matrix.cols)
    {
      throw std::runtime_error("CG requires a square matrix");
    }

    constexpr double one = 1.0;
    constexpr double zero = 0.0;
    const std::int64_t size = matrix.rows;
    const std::size_t vector_size = static_cast<std::size_t>(size);
    const double matrix_vector_flops = 2.0 * static_cast<double>(matrix.nnz());
    const double dot_flops = 2.0 * static_cast<double>(vector_size);
    const double update_x_r_flops = 4.0 * static_cast<double>(vector_size);
    const double update_p_flops = 3.0 * static_cast<double>(vector_size);

    // Byte transfer estimates per operation (int64 col_idx, double values).
    const std::int64_t n_i64 = size;
    const std::int64_t nnz_i64 = static_cast<std::int64_t>(matrix.nnz());
    // row_ptr (int64, n+1) + col_idx (int64, nnz) + values (f64, nnz) + x (f64, n) + y (f64, n)
    const std::int64_t spmv_bytes = 8LL * (3 * n_i64 + 1 + 2 * nnz_i64);
    const std::int64_t dot_bytes = 16LL * n_i64;  // two f64 vectors
    const std::int64_t nrm2_bytes = 8LL * n_i64;  // one f64 vector
    const std::int64_t axpy_bytes = 24LL * n_i64; // read x + read/write y
    const std::int64_t scal_bytes = 16LL * n_i64; // read/write x
    const std::int64_t copy_bytes = 16LL * n_i64; // read src + write dst

    auto timed_op = [](auto &&f) -> double
    {
      const auto t0 = std::chrono::steady_clock::now();
      f();
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    const std::vector<double> x_exact_host = generate_exact_solution(vector_size, options);
    std::vector<std::int64_t> col_idx_host(matrix.col_idx.begin(), matrix.col_idx.end());
    std::vector<double> b_host(vector_size, 0.0);
    if (!options.manufactured_solution)
    {
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
    // Two descriptors: mfg_descr locked to (x_exact, b), cg_descr locked to (p, Ap).
    // oneMath locks a descriptor to the exact vector handles used in spmv_optimize.
    oneapi::math::sparse::spmv_descr_t mfg_descr = nullptr;
    oneapi::math::sparse::spmv_descr_t cg_descr = nullptr;
    const oneapi::math::sparse::matrix_view view(oneapi::math::sparse::matrix_descr::general);

    oneapi::math::sparse::init_csr_matrix(
        q, &matrix_handle, matrix.rows, matrix.cols, matrix.nnz(), oneapi::math::index_base::zero,
        row_ptr.get(), col_idx.get(), values.get());
    oneapi::math::sparse::init_dense_vector(q, &x_exact_handle, size, x_exact.get());
    oneapi::math::sparse::init_dense_vector(q, &x_handle, size, x.get());
    oneapi::math::sparse::init_dense_vector(q, &b_handle, size, b.get());
    oneapi::math::sparse::init_dense_vector(q, &r_handle, size, r.get());
    oneapi::math::sparse::init_dense_vector(q, &p_handle, size, p.get());
    oneapi::math::sparse::init_dense_vector(q, &Ap_handle, size, Ap.get());
    oneapi::math::sparse::init_spmv_descr(q, &mfg_descr);
    oneapi::math::sparse::init_spmv_descr(q, &cg_descr);

    // buffer_size + optimize once per (descriptor, x, y) triple — workspace kept alive.
    // Mirrors CUDA pattern: cusparseSpMV_bufferSize + cusparseSpMV_preprocess once before loop.
    auto setup_spmv = [&](oneapi::math::sparse::spmv_descr_t descr,
                          oneapi::math::sparse::dense_vector_handle_t x_vec,
                          oneapi::math::sparse::dense_vector_handle_t y_vec) -> UsmPtr<std::uint8_t>
    {
      std::size_t ws_size = 0;
      oneapi::math::sparse::spmv_buffer_size(
          q, oneapi::math::transpose::nontrans, &one, view,
          matrix_handle, x_vec, &zero, y_vec,
          oneapi::math::sparse::spmv_alg::default_alg, descr, ws_size);
      UsmPtr<std::uint8_t> ws(nullptr, UsmDeleter<std::uint8_t>{&q});
      void *raw_ws = nullptr;
      if (ws_size > 0)
      {
        ws = make_device_array<std::uint8_t>(q, ws_size);
        raw_ws = ws.get();
      }
      oneapi::math::sparse::spmv_optimize(
          q, oneapi::math::transpose::nontrans, &one, view,
          matrix_handle, x_vec, &zero, y_vec,
          oneapi::math::sparse::spmv_alg::default_alg, descr, raw_ws)
          .wait();
      return ws;
    };

    auto mfg_workspace = options.manufactured_solution
                             ? setup_spmv(mfg_descr, x_exact_handle, b_handle)
                             : UsmPtr<std::uint8_t>(nullptr, UsmDeleter<std::uint8_t>{&q});
    auto cg_workspace = setup_spmv(cg_descr, p_handle, Ap_handle);

    SolverResult result;
    double total_flops = 0.0;

    if (options.manufactured_solution)
    {
      result.perf.spmv.record(timed_op([&]
                                       { oneapi::math::sparse::spmv(
                                             q, oneapi::math::transpose::nontrans, &one, view,
                                             matrix_handle, x_exact_handle, &zero, b_handle,
                                             oneapi::math::sparse::spmv_alg::default_alg, mfg_descr)
                                             .wait(); }),
                              spmv_bytes);
      total_flops += matrix_vector_flops;
    }

    // Initialize: x = 0, r = b, p = b
    q.fill(x.get(), 0.0, vector_size).wait();
    result.perf.copy.record(timed_op([&]
                                     { oneapi::math::blas::column_major::copy(q, size, b.get(), 1, r.get(), 1).wait(); }),
                            copy_bytes);
    result.perf.copy.record(timed_op([&]
                                     { oneapi::math::blas::column_major::copy(q, size, b.get(), 1, p.get(), 1).wait(); }),
                            copy_bytes);

    // ||b|| (nrm2) — not used in stopping criterion but counted for parity with CUDA
    result.perf.nrm2.record(timed_op([&]
                                     { oneapi::math::blas::column_major::dot(q, size, b.get(), 1, b.get(), 1, dot_result.get()).wait(); }),
                            nrm2_bytes);
    total_flops += dot_flops;

    // rho = ||r0||^2 (nrm2)
    double rho;
    result.perf.nrm2.record(timed_op([&]
                                     {
    oneapi::math::blas::column_major::dot(q, size, r.get(), 1, r.get(), 1, dot_result.get()).wait();
    rho = *dot_result; }),
                            nrm2_bytes);
    total_flops += dot_flops;
    const double r0_norm = std::sqrt(rho);

    // Match CUDA stopping criterion: residualrtol scaled by ||r0|| once.
    const double scaled_residual_rtol = options.residual_relative_tolerance * r0_norm;
    const double scaled_diff_rtol =
        (options.diff_relative_tolerance > 0.0 && r0_norm > 0.0)
            ? options.diff_relative_tolerance * r0_norm
            : 0.0;

    result.final_residual = r0_norm;
    result.converged =
        (options.residual_absolute_tolerance > 0.0 && result.final_residual < options.residual_absolute_tolerance) ||
        (options.residual_relative_tolerance > 0.0 && result.final_residual < scaled_residual_rtol);

    const auto solve_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.max_iterations && !result.converged; ++iteration)
    {
      // Ap = A * p
      result.perf.spmv.record(timed_op([&]
                                       { oneapi::math::sparse::spmv(
                                             q, oneapi::math::transpose::nontrans, &one, view,
                                             matrix_handle, p_handle, &zero, Ap_handle,
                                             oneapi::math::sparse::spmv_alg::default_alg, cg_descr)
                                             .wait(); }),
                              spmv_bytes);
      total_flops += matrix_vector_flops;

      // pAp = p · Ap
      result.perf.dot.record(timed_op([&]
                                      { oneapi::math::blas::column_major::dot(q, size, p.get(), 1, Ap.get(), 1, dot_result.get()).wait(); }),
                             dot_bytes);
      total_flops += dot_flops;
      const double pAp = *dot_result;
      if (pAp <= 0.0)
      {
        throw std::runtime_error("matrix is not positive definite under CG iteration");
      }
      const double alpha = rho / pAp;

      // x += alpha * p
      result.perf.axpy.record(timed_op([&]
                                       { oneapi::math::blas::column_major::axpy(q, size, alpha, p.get(), 1, x.get(), 1).wait(); }),
                              axpy_bytes);

      // r -= alpha * Ap
      result.perf.axpy.record(timed_op([&]
                                       { oneapi::math::blas::column_major::axpy(q, size, -alpha, Ap.get(), 1, r.get(), 1).wait(); }),
                              axpy_bytes);
      total_flops += update_x_r_flops;

      // rho_next = ||r||^2 (nrm2)
      double rho_next;
      result.perf.nrm2.record(timed_op([&]
                                       {
      oneapi::math::blas::column_major::dot(q, size, r.get(), 1, r.get(), 1, dot_result.get()).wait();
      rho_next = *dot_result; }),
                              nrm2_bytes);
      total_flops += dot_flops;

      result.iterations = iteration + 1;
      result.final_residual = std::sqrt(rho_next);
      result.converged =
          (options.diff_absolute_tolerance > 0.0 && std::abs(alpha) * std::sqrt(rho) < options.diff_absolute_tolerance) ||
          (scaled_diff_rtol > 0.0 && std::abs(alpha) * std::sqrt(rho) < scaled_diff_rtol) ||
          (options.residual_absolute_tolerance > 0.0 && result.final_residual < options.residual_absolute_tolerance) ||
          (options.residual_relative_tolerance > 0.0 && result.final_residual < scaled_residual_rtol);
      if (result.converged)
      {
        break;
      }

      // p = beta * p + r  (two separate calls: scal + axpy; both counted under axpy)
      const double beta = rho_next / rho;
      result.perf.axpy.record(timed_op([&]
                                       { oneapi::math::blas::column_major::scal(q, size, beta, p.get(), 1).wait(); }),
                              scal_bytes);
      result.perf.axpy.record(timed_op([&]
                                       { oneapi::math::blas::column_major::axpy(q, size, 1.0, r.get(), 1, p.get(), 1).wait(); }),
                              axpy_bytes);
      total_flops += update_p_flops;

      rho = rho_next;
    }

    std::vector<double> x_host(vector_size, 0.0);
    copy_to_host(q, x.get(), x_host.data(), vector_size * sizeof(double));

    oneapi::math::sparse::release_spmv_descr(q, cg_descr).wait();
    if (mfg_descr)
      oneapi::math::sparse::release_spmv_descr(q, mfg_descr).wait();
    oneapi::math::sparse::release_dense_vector(q, Ap_handle).wait();
    oneapi::math::sparse::release_dense_vector(q, p_handle).wait();
    oneapi::math::sparse::release_dense_vector(q, r_handle).wait();
    oneapi::math::sparse::release_dense_vector(q, b_handle).wait();
    oneapi::math::sparse::release_dense_vector(q, x_handle).wait();
    oneapi::math::sparse::release_dense_vector(q, x_exact_handle).wait();
    oneapi::math::sparse::release_sparse_matrix(q, matrix_handle).wait();

    if (options.manufactured_solution)
    {
      std::vector<double> x_error(vector_size, 0.0);
      double x_exact_norm_squared = 0.0;
      double x_error_norm_squared = 0.0;
      for (std::size_t i = 0; i < vector_size; ++i)
      {
        x_error[i] = x_host[i] - x_exact_host[i];
        x_exact_norm_squared += x_exact_host[i] * x_exact_host[i];
        x_error_norm_squared += x_error[i] * x_error[i];
      }
      const double x_exact_norm = std::sqrt(x_exact_norm_squared);
      result.relative_solution_error = x_exact_norm > 0.0 ? std::sqrt(x_error_norm_squared) / x_exact_norm : 0.0;
    }
    else
    {
      result.relative_solution_error = 0.0;
    }

    const auto end = std::chrono::steady_clock::now();
    result.solve_time_seconds = std::chrono::duration<double>(end - solve_start).count();
    result.total_time_seconds = std::chrono::duration<double>(end - start).count();
    result.total_flops = total_flops;
    result.flop_rate_gflops = result.solve_time_seconds > 0.0 ? total_flops / result.solve_time_seconds / 1.0e9 : 0.0;
    return result;
  }

} // namespace acg::solver
