#include "acg/solver/device_csr.hpp"

#include <stdexcept>
#include <string>

#include <oneapi/math.hpp>

#include "acg/solver/index_width.hpp"

namespace acg::solver {

namespace {

template <typename T>
T *device_alloc(sycl::queue &queue, std::size_t count) {
  T *ptr = sycl::malloc_device<T>(count == 0 ? 1 : count, queue);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device memory");
  }
  return ptr;
}

// Uploads the CSR index arrays as Index and wraps them in a oneMath handle.
template <typename Index>
void upload_csr(
    sycl::queue &queue,
    const acg::matrix::CsrMatrix<double> &matrix,
    double *values,
    void *&row_ptr_out,
    void *&col_idx_out,
    oneapi::math::sparse::matrix_handle_t &handle) {
  const std::vector<Index> row_ptr(matrix.row_ptr.begin(), matrix.row_ptr.end());
  const std::vector<Index> col_idx(matrix.col_idx.begin(), matrix.col_idx.end());
  auto *d_row_ptr = device_alloc<Index>(queue, row_ptr.size());
  auto *d_col_idx = device_alloc<Index>(queue, col_idx.size());
  row_ptr_out = d_row_ptr;
  col_idx_out = d_col_idx;
  queue.memcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(Index));
  queue.memcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(Index));
  queue.wait_and_throw();
  oneapi::math::sparse::init_csr_matrix(
      queue, &handle, matrix.rows, matrix.cols, matrix.nnz(), oneapi::math::index_base::zero, d_row_ptr, d_col_idx,
      values);
}

void expect_size(std::int64_t expected, std::int64_t actual, const char *what) {
  if (expected != actual) {
    throw std::runtime_error(std::string(what) + ": size " + std::to_string(actual) + ", expected " +
                             std::to_string(expected));
  }
}

} // namespace

DeviceCsr::DeviceCsr(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix)
    : queue_(&queue), rows_(matrix.rows), cols_(matrix.cols), nnz_(matrix.nnz()),
      index_bits_(csr_fits_index32(matrix.rows, matrix.cols, matrix.nnz()) ? 32 : 64) {
  if (static_cast<std::int64_t>(matrix.row_ptr.size()) != rows_ + 1 ||
      static_cast<std::int64_t>(matrix.col_idx.size()) != nnz_) {
    throw std::runtime_error("malformed CSR matrix: row_ptr/col_idx sizes do not match rows/nnz");
  }
  values_ = device_alloc<double>(queue, matrix.values.size());
  queue.memcpy(values_, matrix.values.data(), matrix.values.size() * sizeof(double));
  if (index_bits_ == 32) {
    upload_csr<std::int32_t>(queue, matrix, values_, row_ptr_, col_idx_, matrix_);
  }
  else {
    upload_csr<std::int64_t>(queue, matrix, values_, row_ptr_, col_idx_, matrix_);
  }
  oneapi::math::sparse::init_spmv_descr(queue, &descr_);
}

DeviceCsr::~DeviceCsr() {
  queue_->wait();
  for (const DeviceVector &v : vectors_) {
    oneapi::math::sparse::release_dense_vector(*queue_, v.handle).wait();
    sycl::free(v.data, *queue_);
  }
  oneapi::math::sparse::release_spmv_descr(*queue_, descr_).wait();
  oneapi::math::sparse::release_sparse_matrix(*queue_, matrix_).wait();
  sycl::free(workspace_, *queue_);
  sycl::free(values_, *queue_);
  sycl::free(col_idx_, *queue_);
  sycl::free(row_ptr_, *queue_);
}

DeviceVector DeviceCsr::vector(std::int64_t size) {
  DeviceVector v{.data = device_alloc<double>(*queue_, static_cast<std::size_t>(size)), .handle = nullptr, .size = size};
  oneapi::math::sparse::init_dense_vector(*queue_, &v.handle, size, v.data);
  vectors_.push_back(v);
  return v;
}

DeviceVector DeviceCsr::vector(const std::vector<double> &host) {
  DeviceVector v = vector(static_cast<std::int64_t>(host.size()));
  queue_->memcpy(v.data, host.data(), host.size() * sizeof(double)).wait_and_throw();
  return v;
}

void DeviceCsr::download(const DeviceVector &src, std::vector<double> &dst) {
  dst.resize(static_cast<std::size_t>(src.size));
  queue_->memcpy(dst.data(), src.data, dst.size() * sizeof(double)).wait_and_throw();
}

void DeviceCsr::prepare_spmv(const DeviceVector &x, const DeviceVector &y) {
  expect_size(cols_, x.size, "spmv input");
  expect_size(rows_, y.size, "spmv output");
  if (spmv_x_ != nullptr) {
    throw std::logic_error("DeviceCsr::prepare_spmv called twice");
  }
  std::size_t workspace_size = 0;
  oneapi::math::sparse::spmv_buffer_size(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, descr_, workspace_size);
  if (workspace_size > 0) {
    workspace_ = device_alloc<std::uint8_t>(*queue_, workspace_size);
  }
  oneapi::math::sparse::spmv_optimize(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, descr_, workspace_)
      .wait_and_throw();
  spmv_x_ = x.data;
  spmv_y_ = y.data;
}

sycl::event DeviceCsr::spmv(const DeviceVector &x, DeviceVector &y) {
  if (x.data != spmv_x_ || y.data != spmv_y_) {
    throw std::logic_error("DeviceCsr::spmv on vectors it was not prepared for");
  }
  return oneapi::math::sparse::spmv(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, descr_);
}

sycl::event DeviceCsr::dot(const DeviceVector &x, const DeviceVector &y, double *device_result) {
  expect_size(x.size, y.size, "dot");
  return oneapi::math::blas::column_major::dot(*queue_, x.size, x.data, 1, y.data, 1, device_result);
}

sycl::event DeviceCsr::copy(const DeviceVector &src, DeviceVector &dst) {
  expect_size(src.size, dst.size, "copy");
  return oneapi::math::blas::column_major::copy(*queue_, src.size, src.data, 1, dst.data, 1);
}

sycl::event DeviceCsr::fill_zero(DeviceVector &x) {
  return queue_->fill(x.data, 0.0, static_cast<std::size_t>(x.size));
}

} // namespace acg::solver
