#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <oneapi/math.hpp>
#include <oneapi/math/sparse_blas.hpp>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue(sycl::gpu_selector_v);

  constexpr std::int64_t rows = 3;
  constexpr std::int64_t cols = 3;
  constexpr std::int64_t nnz = 5;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<std::int64_t> row_ptr = {0, 2, 4, 5};
  std::vector<std::int64_t> col_ind = {0, 1, 1, 2, 2};
  std::vector<float> values = {2.0f, 1.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> x = {1.0f, 2.0f, 3.0f};
  std::vector<float> y(rows, 0.0f);

  auto *dev_row_ptr = sycl::malloc_device<std::int64_t>(row_ptr.size(), queue);
  auto *dev_col_ind = sycl::malloc_device<std::int64_t>(col_ind.size(), queue);
  auto *dev_values = sycl::malloc_device<float>(values.size(), queue);
  auto *dev_x = sycl::malloc_device<float>(x.size(), queue);
  auto *dev_y = sycl::malloc_device<float>(y.size(), queue);
  if (dev_row_ptr == nullptr || dev_col_ind == nullptr || dev_values == nullptr || dev_x == nullptr || dev_y == nullptr) {
    throw std::runtime_error("failed to allocate USM for SPMV test");
  }

  queue.memcpy(dev_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(std::int64_t)).wait();
  queue.memcpy(dev_col_ind, col_ind.data(), col_ind.size() * sizeof(std::int64_t)).wait();
  queue.memcpy(dev_values, values.data(), values.size() * sizeof(float)).wait();
  queue.memcpy(dev_x, x.data(), x.size() * sizeof(float)).wait();
  queue.memcpy(dev_y, y.data(), y.size() * sizeof(float)).wait();

  oneapi::math::sparse::matrix_handle_t matrix_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
  oneapi::math::sparse::dense_vector_handle_t y_handle = nullptr;
  oneapi::math::sparse::spmv_descr_t spmv_descr = nullptr;
  const oneapi::math::sparse::matrix_view view(oneapi::math::sparse::matrix_descr::general);

  oneapi::math::sparse::init_csr_matrix(queue, &matrix_handle, rows, cols, nnz, oneapi::math::index_base::zero,
                                        dev_row_ptr, dev_col_ind, dev_values);
  oneapi::math::sparse::init_dense_vector(queue, &x_handle, rows, dev_x);
  oneapi::math::sparse::init_dense_vector(queue, &y_handle, rows, dev_y);
  oneapi::math::sparse::init_spmv_descr(queue, &spmv_descr);

  std::size_t workspace_size = 0;
  oneapi::math::sparse::spmv_buffer_size(queue, oneapi::math::transpose::nontrans, &alpha,
                                         view, matrix_handle, x_handle, &beta, y_handle,
                                         oneapi::math::sparse::spmv_alg::default_alg, spmv_descr, workspace_size);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    workspace = sycl::malloc_device<std::uint8_t>(workspace_size, queue);
    if (workspace == nullptr) {
      throw std::runtime_error("failed to allocate SPMV workspace");
    }
    oneapi::math::sparse::spmv_optimize(queue, oneapi::math::transpose::nontrans, &alpha,
                                        view, matrix_handle, x_handle, &beta, y_handle,
                                        oneapi::math::sparse::spmv_alg::default_alg, spmv_descr, workspace)
        .wait();
  }

  oneapi::math::sparse::spmv(queue, oneapi::math::transpose::nontrans, &alpha, view, matrix_handle, x_handle, &beta, y_handle,
                             oneapi::math::sparse::spmv_alg::default_alg, spmv_descr)
      .wait();

  queue.memcpy(y.data(), dev_y, y.size() * sizeof(float)).wait();

  if (workspace != nullptr) {
    sycl::free(workspace, queue);
  }
  oneapi::math::sparse::release_spmv_descr(queue, spmv_descr).wait();
  oneapi::math::sparse::release_dense_vector(queue, y_handle).wait();
  oneapi::math::sparse::release_dense_vector(queue, x_handle).wait();
  oneapi::math::sparse::release_sparse_matrix(queue, matrix_handle).wait();

  sycl::free(dev_y, queue);
  sycl::free(dev_x, queue);
  sycl::free(dev_values, queue);
  sycl::free(dev_col_ind, queue);
  sycl::free(dev_row_ptr, queue);

  std::cout << "SPMV result:";
  for (float value : y) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  const std::vector<float> expected = {4.0f, 18.0f, 15.0f};
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (std::abs(y[i] - expected[i]) > 1e-4f) {
      std::cerr << "unexpected SPMV output at index " << i << ": expected " << expected[i] << ", got " << y[i]
                << '\n';
      return 1;
    }
  }

  std::cout << "SPMV smoke test passed\n";
  return 0;
}
