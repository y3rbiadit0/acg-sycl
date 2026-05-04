#include "acg/solver/backends/onemath_cuda_backend.hpp"

#include <sstream>
#include <stdexcept>

#include <oneapi/math.hpp>

namespace acg::solver::backends {

namespace {

void expect_equal_size(std::int64_t expected, std::int64_t actual, const char *operation) {
  if (expected != actual) {
    std::ostringstream message;
    message << operation << " size mismatch: expected " << expected << ", got " << actual;
    throw std::runtime_error(message.str());
  }
}

} // namespace

OnemathCudaBackend::OnemathCudaBackend(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix)
    : queue_(&queue), rows_(matrix.rows), cols_(matrix.cols), nnz_(static_cast<std::int64_t>(matrix.nnz())),
      spmv_bytes_(8LL * (3 * rows_ + 1 + 2 * nnz_)), dot_bytes_(16LL * rows_), nrm2_bytes_(8LL * rows_),
      axpy_bytes_(24LL * rows_), scal_bytes_(16LL * rows_), copy_bytes_(16LL * rows_),
      row_ptr_(make_device_array<std::int64_t>(matrix.row_ptr.size())),
      col_idx_(make_device_array<std::int64_t>(matrix.col_idx.size())),
      values_(make_device_array<double>(matrix.values.size())), dot_result_(make_shared_scalar<double>(0.0)),
      spmv_workspace_(nullptr), matrix_handle_(nullptr), spmv_descr_(nullptr),
      view_(oneapi::math::sparse::matrix_descr::general), one_(1.0), zero_(0.0), optimized_x_(nullptr), optimized_y_(nullptr) {
  std::vector<std::int64_t> col_idx_host(matrix.col_idx.begin(), matrix.col_idx.end());

  queue_->memcpy(row_ptr_, matrix.row_ptr.data(), matrix.row_ptr.size() * sizeof(std::int64_t)).wait();
  queue_->memcpy(col_idx_, col_idx_host.data(), col_idx_host.size() * sizeof(std::int64_t)).wait();
  queue_->memcpy(values_, matrix.values.data(), matrix.values.size() * sizeof(double)).wait();

  oneapi::math::sparse::init_csr_matrix(
      *queue_, &matrix_handle_, rows_, cols_, nnz_, oneapi::math::index_base::zero, row_ptr_, col_idx_,
      values_);
  oneapi::math::sparse::init_spmv_descr(*queue_, &spmv_descr_);
}

OnemathCudaBackend::~OnemathCudaBackend() {
  for (auto it = vectors_.rbegin(); it != vectors_.rend(); ++it) {
    if (it->handle != nullptr) {
      oneapi::math::sparse::release_dense_vector(*queue_, it->handle).wait();
    }
  }
  if (spmv_descr_ != nullptr) {
    oneapi::math::sparse::release_spmv_descr(*queue_, spmv_descr_).wait();
  }
  if (matrix_handle_ != nullptr) {
    oneapi::math::sparse::release_sparse_matrix(*queue_, matrix_handle_).wait();
  }
  if (spmv_workspace_ != nullptr) {
    sycl::free(spmv_workspace_, *queue_);
  }
  if (dot_result_ != nullptr) {
    sycl::free(dot_result_, *queue_);
  }
  if (values_ != nullptr) {
    sycl::free(values_, *queue_);
  }
  if (col_idx_ != nullptr) {
    sycl::free(col_idx_, *queue_);
  }
  if (row_ptr_ != nullptr) {
    sycl::free(row_ptr_, *queue_);
  }
}

std::int64_t OnemathCudaBackend::size() const noexcept { return rows_; }

std::int64_t OnemathCudaBackend::nnz() const noexcept { return nnz_; }

DeviceVector OnemathCudaBackend::create_vector() { return create_vector(rows_); }

DeviceVector OnemathCudaBackend::create_vector(std::int64_t size) {
  return make_vector_from_pointer(make_device_array<double>(static_cast<std::size_t>(size)), size);
}

DeviceVector OnemathCudaBackend::create_vector_from_host(const std::vector<double> &host) {
  DeviceVector vector = create_vector(static_cast<std::int64_t>(host.size()));
  queue_->memcpy(vector.data, host.data(), host.size() * sizeof(double)).wait();
  return vector;
}

void OnemathCudaBackend::download_vector(const DeviceVector &src, std::vector<double> &dst) {
  dst.resize(static_cast<std::size_t>(src.size));
  queue_->memcpy(dst.data(), src.data, dst.size() * sizeof(double)).wait();
}

void OnemathCudaBackend::fill_zero(DeviceVector &x) { queue_->fill(x.data, 0.0, static_cast<std::size_t>(x.size)).wait(); }

void OnemathCudaBackend::copy(const DeviceVector &src, DeviceVector &dst) {
  expect_equal_size(src.size, dst.size, "copy");
  oneapi::math::blas::column_major::copy(*queue_, src.size, src.data, 1, dst.data, 1).wait();
}

void OnemathCudaBackend::scal(double alpha, DeviceVector &x) {
  oneapi::math::blas::column_major::scal(*queue_, x.size, alpha, x.data, 1).wait();
}

void OnemathCudaBackend::axpy(double alpha, const DeviceVector &x, DeviceVector &y) {
  expect_equal_size(x.size, y.size, "axpy");
  oneapi::math::blas::column_major::axpy(*queue_, x.size, alpha, x.data, 1, y.data, 1).wait();
}

double OnemathCudaBackend::dot(const DeviceVector &x, const DeviceVector &y) {
  expect_equal_size(x.size, y.size, "dot");
  oneapi::math::blas::column_major::dot(*queue_, x.size, x.data, 1, y.data, 1, dot_result_).wait();
  return *dot_result_;
}

double *OnemathCudaBackend::create_device_scalar(double initial_value) {
  double *scalar = make_device_array<double>(1);
  queue_->memcpy(scalar, &initial_value, sizeof(double)).wait();
  return scalar;
}

void OnemathCudaBackend::destroy_device_scalar(double *scalar) noexcept {
  if (scalar != nullptr) {
    sycl::free(scalar, *queue_);
  }
}

void OnemathCudaBackend::dot_to_device(const DeviceVector &x, const DeviceVector &y, double *result) {
  expect_equal_size(x.size, y.size, "dot");
  oneapi::math::blas::column_major::dot(*queue_, x.size, x.data, 1, y.data, 1, result).wait();
}

double OnemathCudaBackend::read_device_scalar(const double *scalar) {
  double value = 0.0;
  queue_->memcpy(&value, scalar, sizeof(double)).wait();
  return value;
}

void OnemathCudaBackend::spmv(const DeviceVector &x, DeviceVector &y) {
  expect_equal_size(cols_, x.size, "spmv input");
  expect_equal_size(rows_, y.size, "spmv output");
  optimize_spmv_for(x, y);
  oneapi::math::sparse::spmv(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_handle_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, spmv_descr_)
      .wait();
}

std::int64_t OnemathCudaBackend::estimated_spmv_bytes() const noexcept { return spmv_bytes_; }
std::int64_t OnemathCudaBackend::estimated_dot_bytes() const noexcept { return dot_bytes_; }
std::int64_t OnemathCudaBackend::estimated_nrm2_bytes() const noexcept { return nrm2_bytes_; }
std::int64_t OnemathCudaBackend::estimated_axpy_bytes() const noexcept { return axpy_bytes_; }
std::int64_t OnemathCudaBackend::estimated_scal_bytes() const noexcept { return scal_bytes_; }
std::int64_t OnemathCudaBackend::estimated_copy_bytes() const noexcept { return copy_bytes_; }

template <typename T>
T *OnemathCudaBackend::make_device_array(std::size_t count) {
  T *ptr = sycl::malloc_device<T>(count, *queue_);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device memory");
  }
  return ptr;
}

template <typename T>
T *OnemathCudaBackend::make_shared_scalar(T initial_value) {
  // malloc_host (pinned host memory) instead of malloc_shared (CUDA UVM):
  // the GPU writes 8 bytes via DMA; CPU reads directly after .wait() with no
  // UVM page-fault serialization. With malloc_shared, 4 concurrent ranks
  // trigger simultaneous GPU→CPU page migrations that serialize through the
  // CUDA driver's global UVM fault handler, each costing ~5ms and showing up
  // as allreduce latency (the faster ranks wait at MPI_Allreduce for the rest).
  T *ptr = sycl::malloc_host<T>(1, *queue_);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate shared scalar");
  }
  *ptr = initial_value;
  return ptr;
}

DeviceVector OnemathCudaBackend::make_vector_from_pointer(double *ptr, std::int64_t size) {
  DeviceVector vector{.data = ptr, .handle = nullptr, .size = size};
  oneapi::math::sparse::init_dense_vector(*queue_, &vector.handle, size, vector.data);
  vectors_.push_back(vector);
  return vector;
}

void OnemathCudaBackend::optimize_spmv_for(const DeviceVector &x, const DeviceVector &y) {
  if (optimized_x_ == x.data && optimized_y_ == y.data) {
    return;
  }

  std::size_t workspace_size = 0;
  oneapi::math::sparse::spmv_buffer_size(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_handle_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, spmv_descr_, workspace_size);

  void *raw_workspace = nullptr;
  if (workspace_size > 0) {
    if (spmv_workspace_ != nullptr) {
      sycl::free(spmv_workspace_, *queue_);
    }
    spmv_workspace_ = make_device_array<std::uint8_t>(workspace_size);
    raw_workspace = spmv_workspace_;
  }
  else {
    if (spmv_workspace_ != nullptr) {
      sycl::free(spmv_workspace_, *queue_);
      spmv_workspace_ = nullptr;
    }
  }

  oneapi::math::sparse::spmv_optimize(
      *queue_, oneapi::math::transpose::nontrans, &one_, view_, matrix_handle_, x.handle, &zero_, y.handle,
      oneapi::math::sparse::spmv_alg::default_alg, spmv_descr_, raw_workspace)
      .wait();

  optimized_x_ = x.data;
  optimized_y_ = y.data;
}

template double *OnemathCudaBackend::make_device_array<double>(std::size_t);
template std::int64_t *OnemathCudaBackend::make_device_array<std::int64_t>(std::size_t);
template std::uint8_t *OnemathCudaBackend::make_device_array<std::uint8_t>(std::size_t);
template double *OnemathCudaBackend::make_shared_scalar<double>(double);

} // namespace acg::solver::backends
