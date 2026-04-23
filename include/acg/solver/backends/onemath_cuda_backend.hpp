#ifndef ACG_SOLVER_BACKENDS_ONEMATH_CUDA_BACKEND_HPP
#define ACG_SOLVER_BACKENDS_ONEMATH_CUDA_BACKEND_HPP

#include <cstdint>
#include <vector>

#include <oneapi/math/sparse_blas.hpp>
#include <sycl/sycl.hpp>

#include "acg/matrix/csr_matrix.hpp"
#include "acg/solver/backends/cg_backend.hpp"

namespace acg::solver::backends {

class OnemathCudaBackend : public SingleGpuCgBackend {
public:
  OnemathCudaBackend(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix);
  ~OnemathCudaBackend() override;

  OnemathCudaBackend(const OnemathCudaBackend &) = delete;
  OnemathCudaBackend &operator=(const OnemathCudaBackend &) = delete;

  std::int64_t size() const noexcept override;
  std::int64_t nnz() const noexcept override;
  std::int64_t columns() const noexcept;

  DeviceVector create_vector() override;
  DeviceVector create_vector(std::int64_t size);
  DeviceVector create_vector_from_host(const std::vector<double> &host) override;
  void download_vector(const DeviceVector &src, std::vector<double> &dst) override;

  void fill_zero(DeviceVector &x) override;
  void copy(const DeviceVector &src, DeviceVector &dst) override;
  void scal(double alpha, DeviceVector &x) override;
  void axpy(double alpha, const DeviceVector &x, DeviceVector &y) override;
  double dot(const DeviceVector &x, const DeviceVector &y) override;
  void spmv(const DeviceVector &x, DeviceVector &y) override;

  std::int64_t estimated_spmv_bytes() const noexcept override;
  std::int64_t estimated_dot_bytes() const noexcept override;
  std::int64_t estimated_nrm2_bytes() const noexcept override;
  std::int64_t estimated_axpy_bytes() const noexcept override;
  std::int64_t estimated_scal_bytes() const noexcept override;
  std::int64_t estimated_copy_bytes() const noexcept override;

private:
  template <typename T>
  T *make_device_array(std::size_t count);

  template <typename T>
  T *make_shared_scalar(T initial_value = T{});

  DeviceVector make_vector_from_pointer(double *ptr, std::int64_t size);
  void optimize_spmv_for(const DeviceVector &x, const DeviceVector &y);

  sycl::queue *queue_;
  std::int64_t rows_;
  std::int64_t cols_;
  std::int64_t nnz_;
  std::int64_t spmv_bytes_;
  std::int64_t dot_bytes_;
  std::int64_t nrm2_bytes_;
  std::int64_t axpy_bytes_;
  std::int64_t scal_bytes_;
  std::int64_t copy_bytes_;

  std::int64_t *row_ptr_;
  std::int64_t *col_idx_;
  double *values_;
  double *dot_result_;
  std::uint8_t *spmv_workspace_;

  oneapi::math::sparse::matrix_handle_t matrix_handle_;
  oneapi::math::sparse::spmv_descr_t spmv_descr_;
  const oneapi::math::sparse::matrix_view view_;
  double one_;
  double zero_;
  double *optimized_x_;
  double *optimized_y_;

  std::vector<DeviceVector> vectors_;
};

} // namespace acg::solver::backends

#endif
