#ifndef ACG_SOLVER_CG_KERNELS_HPP
#define ACG_SOLVER_CG_KERNELS_HPP

#include <cstdint>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "acg/solver/device_csr.hpp"

// The CG vector updates, with alpha and beta formed on the device from scalars
// in device memory, so the host never has to read gamma before the updates can
// be queued. Each work item divides for itself: one extra load and division per
// element is cheaper than a kernel launch to compute the scalar once.
namespace acg::solver::kernels {

// x <- x + alpha s,  r <- r - alpha t,  alpha = rho / gamma.
inline sycl::event update_solution_and_residual(
    sycl::queue &queue,
    const double *rho,
    const double *gamma,
    const DeviceVector &s,
    const DeviceVector &t,
    DeviceVector &x,
    DeviceVector &r) {
  if (s.size != t.size || s.size != x.size || s.size != r.size) {
    throw std::runtime_error("solution/residual update size mismatch");
  }
  const double *s_data = s.data;
  const double *t_data = t.data;
  double *x_data = x.data;
  double *r_data = r.data;
  return queue.parallel_for(sycl::range<1>(static_cast<std::size_t>(s.size)), [=](sycl::id<1> idx) {
    const std::size_t i = idx[0];
    const double alpha = *rho / *gamma;
    r_data[i] -= alpha * t_data[i];
    x_data[i] += alpha * s_data[i];
  });
}

// s <- beta s + r,  beta = rho_next / rho.
inline sycl::event update_search_direction(
    sycl::queue &queue, const double *rho_next, const double *rho, const DeviceVector &r, DeviceVector &s) {
  if (r.size != s.size) {
    throw std::runtime_error("search direction update size mismatch");
  }
  const double *r_data = r.data;
  double *s_data = s.data;
  return queue.parallel_for(sycl::range<1>(static_cast<std::size_t>(r.size)), [=](sycl::id<1> idx) {
    const std::size_t i = idx[0];
    const double beta = *rho_next / *rho;
    s_data[i] = beta * s_data[i] + r_data[i];
  });
}

// The CG scalars in device memory, plus a pinned host mirror. rho alternates
// between slots 0 and 2 from one iteration to the next with gamma in slot 1,
// so the pair the host needs every iteration -- gamma and the new rho -- is
// always two adjacent doubles and comes back in one copy.
class ScalarBlock {
public:
  explicit ScalarBlock(sycl::queue &queue) : queue_(&queue) {
    device_ = sycl::malloc_device<double>(kSlots, queue);
    host_ = sycl::malloc_host<double>(kSlots, queue);
    if (device_ == nullptr || host_ == nullptr) {
      release();
      throw std::runtime_error("failed to allocate CG scalars");
    }
    queue.fill(device_, 0.0, kSlots).wait_and_throw();
  }
  ~ScalarBlock() { release(); }
  ScalarBlock(const ScalarBlock &) = delete;
  ScalarBlock &operator=(const ScalarBlock &) = delete;

  double *rho() noexcept { return device_ + rho_slot_; }
  double *rho_next() noexcept { return device_ + (2 - rho_slot_); }
  double *gamma() noexcept { return device_ + kGamma; }

  sycl::event read_gamma_and_rho_next() {
    const int first = rho_slot_ == 0 ? kGamma : 0;
    return queue_->memcpy(host_ + first, device_ + first, 2 * sizeof(double));
  }
  sycl::event read_rho() { return queue_->memcpy(host_ + rho_slot_, device_ + rho_slot_, sizeof(double)); }

  [[nodiscard]] double host_rho() const noexcept { return host_[rho_slot_]; }
  [[nodiscard]] double host_rho_next() const noexcept { return host_[2 - rho_slot_]; }
  [[nodiscard]] double host_gamma() const noexcept { return host_[kGamma]; }

  // rho_next becomes rho. Only the slot index moves: work already queued keeps
  // the pointers it was given, and the in-order queue runs it before the next
  // producer overwrites the old slot.
  void advance() noexcept { rho_slot_ = 2 - rho_slot_; }
  void reset() noexcept { rho_slot_ = 0; }

private:
  static constexpr int kSlots = 3;
  static constexpr int kGamma = 1;

  void release() noexcept {
    if (device_ != nullptr) {
      sycl::free(device_, *queue_);
    }
    if (host_ != nullptr) {
      sycl::free(host_, *queue_);
    }
    device_ = nullptr;
    host_ = nullptr;
  }

  sycl::queue *queue_;
  double *device_ = nullptr;
  double *host_ = nullptr;
  int rho_slot_ = 0;
};

} // namespace acg::solver::kernels

#endif
