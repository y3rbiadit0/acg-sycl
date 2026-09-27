#ifndef ACG_SOLVER_DEVICE_CSR_HPP
#define ACG_SOLVER_DEVICE_CSR_HPP

#include <cstdint>
#include <vector>

#include <oneapi/math/sparse_blas.hpp>
#include <sycl/sycl.hpp>

#include "acg/matrix/csr_matrix.hpp"

namespace acg::solver {

struct DeviceVector {
  double *data = nullptr;
  oneapi::math::sparse::dense_vector_handle_t handle = nullptr;
  std::int64_t size = 0;
};

// A rank's local CSR matrix on the device, plus the vectors and BLAS-1 calls
// the CG loop uses with it, all through oneMath. Every operation is submitted
// without a host wait and returns its event; the queue is in-order, so later
// submissions already depend on earlier ones and the caller waits only where
// the host needs a result (or MPI, which cannot see SYCL dependencies, is
// about to read one).
class DeviceCsr {
public:
  // Row pointers and column indices go to the device as 32-bit integers when
  // the matrix fits (csr_fits_index32), else 64-bit.
  DeviceCsr(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix);
  ~DeviceCsr();

  DeviceCsr(const DeviceCsr &) = delete;
  DeviceCsr &operator=(const DeviceCsr &) = delete;

  // Vectors are owned here and freed with the matrix.
  DeviceVector vector(std::int64_t size);
  DeviceVector vector(const std::vector<double> &host);
  void download(const DeviceVector &src, std::vector<double> &dst);

  // y = A x. cuSPARSE keeps its analysis in the matrix handle and must not
  // re-analyse one for a second vector pair, so the handle is analysed once, for
  // the pair given here, and spmv() accepts only that pair.
  void prepare_spmv(const DeviceVector &x, const DeviceVector &y);
  sycl::event spmv(const DeviceVector &x, DeviceVector &y);

  sycl::event dot(const DeviceVector &x, const DeviceVector &y, double *device_result);
  sycl::event copy(const DeviceVector &src, DeviceVector &dst);
  sycl::event fill_zero(DeviceVector &x);

  [[nodiscard]] std::int64_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::int64_t nnz() const noexcept { return nnz_; }
  [[nodiscard]] int index_bits() const noexcept { return index_bits_; }
  [[nodiscard]] sycl::queue &queue() noexcept { return *queue_; }

private:
  sycl::queue *queue_;
  std::int64_t rows_;
  std::int64_t cols_;
  std::int64_t nnz_;
  int index_bits_;
  void *row_ptr_ = nullptr;
  void *col_idx_ = nullptr;
  double *values_ = nullptr;

  oneapi::math::sparse::matrix_handle_t matrix_ = nullptr;
  oneapi::math::sparse::spmv_descr_t descr_ = nullptr;
  std::uint8_t *workspace_ = nullptr;
  const double *spmv_x_ = nullptr;
  const double *spmv_y_ = nullptr;
  const oneapi::math::sparse::matrix_view view_{oneapi::math::sparse::matrix_descr::general};
  const double one_ = 1.0;
  const double zero_ = 0.0;

  std::vector<DeviceVector> vectors_;
};

} // namespace acg::solver

#endif
