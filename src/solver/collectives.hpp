#ifndef ACG_SOLVER_COLLECTIVES_HPP
#define ACG_SOLVER_COLLECTIVES_HPP

#include <memory>

#include <sycl/sycl.hpp>

#include "acg/solver/solver_options.hpp"

namespace acg::solver {

// In-place sum of one double in device memory across all ranks, through
// CUDA-aware MPI or oneCCL (NCCL backend). When allreduce_sum returns, the
// result is in device memory on every rank and any SYCL work submitted after
// it sees it. The caller must have waited for the value's producer: neither
// MPI nor oneCCL can see SYCL dependencies. A single rank is a no-op.
class Collectives {
public:
  Collectives(sycl::queue &queue, SolverCollectiveMode mode, int rank, int size);
  ~Collectives();
  Collectives(const Collectives &) = delete;
  Collectives &operator=(const Collectives &) = delete;

  void allreduce_sum(double *device_value);
  [[nodiscard]] int size() const noexcept { return size_; }

private:
  struct OneCcl;
  [[maybe_unused]] sycl::queue *queue_; // used by the oneCCL path only
  SolverCollectiveMode mode_;
  int size_;
  std::unique_ptr<OneCcl> oneccl_;
};

} // namespace acg::solver

#endif
