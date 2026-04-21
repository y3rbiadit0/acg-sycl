#ifndef ACG_SOLVER_BACKENDS_CG_BACKEND_HPP
#define ACG_SOLVER_BACKENDS_CG_BACKEND_HPP

#include <cstdint>
#include <vector>

#include <oneapi/math/sparse_blas.hpp>

namespace acg::solver::backends {

struct DeviceVector {
  double *data = nullptr;
  oneapi::math::sparse::dense_vector_handle_t handle = nullptr;
};

class SingleGpuCgBackend {
public:
  virtual ~SingleGpuCgBackend() = default;

  virtual std::int64_t size() const noexcept = 0;
  virtual std::int64_t nnz() const noexcept = 0;

  virtual DeviceVector create_vector() = 0;
  virtual DeviceVector create_vector_from_host(const std::vector<double> &host) = 0;
  virtual void download_vector(const DeviceVector &src, std::vector<double> &dst) = 0;

  virtual void fill_zero(DeviceVector &x) = 0;
  virtual void copy(const DeviceVector &src, DeviceVector &dst) = 0;
  virtual void scal(double alpha, DeviceVector &x) = 0;
  virtual void axpy(double alpha, const DeviceVector &x, DeviceVector &y) = 0;
  virtual double dot(const DeviceVector &x, const DeviceVector &y) = 0;
  virtual void spmv(const DeviceVector &x, DeviceVector &y) = 0;

  virtual std::int64_t estimated_spmv_bytes() const noexcept = 0;
  virtual std::int64_t estimated_dot_bytes() const noexcept = 0;
  virtual std::int64_t estimated_nrm2_bytes() const noexcept = 0;
  virtual std::int64_t estimated_axpy_bytes() const noexcept = 0;
  virtual std::int64_t estimated_scal_bytes() const noexcept = 0;
  virtual std::int64_t estimated_copy_bytes() const noexcept = 0;
};

} // namespace acg::solver::backends

#endif
