// oneMath SpMV smoke test, FP64, with both device index widths the solver can
// use (32-bit when the matrix fits, 64-bit otherwise). The matrix has irregular row
// lengths and empty rows, and every result is checked against a CPU reference.
// The 32/64-bit choice is checked on shapes alone, at compile time, so no
// matrix near the limit has to be allocated.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <oneapi/math.hpp>
#include <oneapi/math/sparse_blas.hpp>
#include <sycl/sycl.hpp>

#include "acg/solver/index_width.hpp"

namespace {

struct HostCsr {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::vector<std::int64_t> row_ptr;
  std::vector<std::int64_t> col_idx;
  std::vector<double> values;
};

// Row r holds (r % 7) entries -- so every seventh row is empty -- at columns
// spread across the matrix, plus a long row in the middle.
HostCsr make_irregular_matrix(std::int64_t n) {
  HostCsr m;
  m.rows = n;
  m.cols = n;
  m.row_ptr.push_back(0);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  for (std::int64_t r = 0; r < n; ++r) {
    std::int64_t length = r % 7;
    if (r == n / 2) {
      length = n / 3;
    }
    for (std::int64_t k = 0; k < length; ++k) {
      m.col_idx.push_back((r * 31 + k * 97) % n);
      m.values.push_back(value(rng));
    }
    m.row_ptr.push_back(static_cast<std::int64_t>(m.col_idx.size()));
  }
  return m;
}

std::vector<double> reference_spmv(const HostCsr &m, const std::vector<double> &x) {
  std::vector<double> y(static_cast<std::size_t>(m.rows), 0.0);
  for (std::int64_t r = 0; r < m.rows; ++r) {
    for (std::int64_t k = m.row_ptr[r]; k < m.row_ptr[r + 1]; ++k) {
      y[r] += m.values[k] * x[m.col_idx[k]];
    }
  }
  return y;
}

template <typename Index>
std::vector<double> device_spmv(sycl::queue &queue, const HostCsr &m, const std::vector<double> &x) {
  const std::vector<Index> row_ptr(m.row_ptr.begin(), m.row_ptr.end());
  const std::vector<Index> col_idx(m.col_idx.begin(), m.col_idx.end());
  const std::int64_t nnz = static_cast<std::int64_t>(m.values.size());
  std::vector<double> y(static_cast<std::size_t>(m.rows), -1.0);

  auto *d_row_ptr = sycl::malloc_device<Index>(row_ptr.size(), queue);
  auto *d_col_idx = sycl::malloc_device<Index>(col_idx.size(), queue);
  auto *d_values = sycl::malloc_device<double>(m.values.size(), queue);
  auto *d_x = sycl::malloc_device<double>(x.size(), queue);
  auto *d_y = sycl::malloc_device<double>(y.size(), queue);
  if (d_row_ptr == nullptr || d_col_idx == nullptr || d_values == nullptr || d_x == nullptr || d_y == nullptr) {
    throw std::runtime_error("failed to allocate USM for SpMV test");
  }
  queue.memcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(Index)).wait();
  queue.memcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(Index)).wait();
  queue.memcpy(d_values, m.values.data(), m.values.size() * sizeof(double)).wait();
  queue.memcpy(d_x, x.data(), x.size() * sizeof(double)).wait();
  queue.memcpy(d_y, y.data(), y.size() * sizeof(double)).wait();

  const double alpha = 1.0;
  const double beta = 0.0;
  oneapi::math::sparse::matrix_handle_t matrix = nullptr;
  oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t y_handle = nullptr;
  oneapi::math::sparse::spmv_descr_t descr = nullptr;
  const oneapi::math::sparse::matrix_view view(oneapi::math::sparse::matrix_descr::general);
  oneapi::math::sparse::init_csr_matrix(queue, &matrix, m.rows, m.cols, nnz, oneapi::math::index_base::zero, d_row_ptr,
                                        d_col_idx, d_values);
  oneapi::math::sparse::init_dense_vector(queue, &x_handle, m.cols, d_x);
  oneapi::math::sparse::init_dense_vector(queue, &y_handle, m.rows, d_y);
  oneapi::math::sparse::init_spmv_descr(queue, &descr);

  std::size_t workspace_size = 0;
  oneapi::math::sparse::spmv_buffer_size(queue, oneapi::math::transpose::nontrans, &alpha, view, matrix, x_handle, &beta,
                                         y_handle, oneapi::math::sparse::spmv_alg::default_alg, descr, workspace_size);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    workspace = sycl::malloc_device<std::uint8_t>(workspace_size, queue);
    if (workspace == nullptr) {
      throw std::runtime_error("failed to allocate SpMV workspace");
    }
  }
  oneapi::math::sparse::spmv_optimize(queue, oneapi::math::transpose::nontrans, &alpha, view, matrix, x_handle, &beta,
                                      y_handle, oneapi::math::sparse::spmv_alg::default_alg, descr, workspace)
      .wait();
  // Twice, as the solver does every iteration: the second call must not
  // depend on state the first one left behind.
  for (int repeat = 0; repeat < 2; ++repeat) {
    oneapi::math::sparse::spmv(queue, oneapi::math::transpose::nontrans, &alpha, view, matrix, x_handle, &beta, y_handle,
                               oneapi::math::sparse::spmv_alg::default_alg, descr)
        .wait();
  }
  queue.memcpy(y.data(), d_y, y.size() * sizeof(double)).wait();

  oneapi::math::sparse::release_spmv_descr(queue, descr).wait();
  oneapi::math::sparse::release_dense_vector(queue, y_handle).wait();
  oneapi::math::sparse::release_dense_vector(queue, x_handle).wait();
  oneapi::math::sparse::release_sparse_matrix(queue, matrix).wait();
  if (workspace != nullptr) {
    sycl::free(workspace, queue);
  }
  sycl::free(d_y, queue);
  sycl::free(d_x, queue);
  sycl::free(d_values, queue);
  sycl::free(d_col_idx, queue);
  sycl::free(d_row_ptr, queue);
  return y;
}

bool check(const char *name, const std::vector<double> &expected, const std::vector<double> &actual) {
  double max_error = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
  }
  const bool ok = max_error <= 1e-12;
  std::cout << "spmv " << name << ": max_abs_error=" << max_error << (ok ? " ok" : " FAILED") << '\n';
  return ok;
}

bool check_index_width_choice() {
  using acg::solver::csr_fits_index32;
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  static_assert(csr_fits_index32(100, 100, 1000));
  static_assert(csr_fits_index32(1000, 1000, limit));
  static_assert(!csr_fits_index32(1000, 1000, limit + 1));
  static_assert(!csr_fits_index32(limit + 1, 10, 10));
  static_assert(!csr_fits_index32(10, limit + 1, 10));
  std::cout << "index width choice: ok\n";
  return true;
}

} // namespace

int main() {
  sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
  const HostCsr m = make_irregular_matrix(2000);
  std::vector<double> x(static_cast<std::size_t>(m.cols));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = 1.0 + 0.001 * static_cast<double>(i % 97);
  }
  const std::vector<double> expected = reference_spmv(m, x);

  bool ok = check("fp64 idx64", expected, device_spmv<std::int64_t>(queue, m, x));
  ok = check("fp64 idx32", expected, device_spmv<std::int32_t>(queue, m, x)) && ok;
  ok = check_index_width_choice() && ok;

  if (!ok) {
    std::cerr << "SPMV smoke test FAILED\n";
    return 1;
  }
  std::cout << "SPMV smoke test passed\n";
  return 0;
}
