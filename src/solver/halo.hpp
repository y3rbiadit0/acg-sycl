#ifndef ACG_SOLVER_HALO_HPP
#define ACG_SOLVER_HALO_HPP

#include <cstdint>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

#include "acg/matrix/distributed_csr_matrix.hpp"
#include "acg/solver/device_csr.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver {

template <typename T>
class DeviceArray {
public:
  DeviceArray(sycl::queue &queue, const std::vector<T> &host);
  ~DeviceArray() { sycl::free(data_, *queue_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  [[nodiscard]] T *get() const noexcept { return data_; }

private:
  sycl::queue *queue_;
  T *data_ = nullptr;
};

// Exchange of the search direction's border values with neighbouring ranks,
// over CUDA-aware MPI. One kernel packs every outgoing value into one
// contiguous device buffer (each send reads its own slice); receives land
// directly in the ghost vector, so there is no unpack step.
//
//   begin(): pack, wait for the pack (MPI cannot see SYCL dependencies), post
//            every receive and send;
//   finish(): MPI_Waitall.
//
// The caller submits the interior SpMV between the two, so the CPU is inside
// MPI, driving progress, while the GPU computes.
class HaloExchange {
public:
  HaloExchange(sycl::queue &queue, const acg::matrix::DistributedCsrMatrixPartition &partition);

  [[nodiscard]] bool active() const noexcept { return !sends_.empty() || !recvs_.empty(); }
  void begin(const DeviceVector &s, double *ghosts, WaitTimes &waits);
  void finish(WaitTimes &waits);

private:
  struct Peer {
    int rank;
    std::int64_t offset; // into the send buffer, or into the ghost vector
    std::int64_t count;
  };

  sycl::queue *queue_;
  std::vector<Peer> sends_;
  std::vector<Peer> recvs_;
  std::int64_t send_count_ = 0;
  DeviceArray<std::int64_t> send_indices_;
  DeviceArray<double> send_buffer_;
  std::vector<MPI_Request> requests_;
};

// t += A_halo * ghosts for the rows that have entries in ghost columns: one
// work item per such row. Row pointers and the row list are 32-bit when they
// fit, like the main SpMV.
class HaloSpmv {
public:
  HaloSpmv(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix);
  ~HaloSpmv();
  HaloSpmv(const HaloSpmv &) = delete;
  HaloSpmv &operator=(const HaloSpmv &) = delete;

  sycl::event add(const double *ghosts, DeviceVector &y);

private:
  sycl::queue *queue_;
  std::size_t active_rows_ = 0;
  bool index32_;
  void *rows_ = nullptr;
  void *row_ptr_ = nullptr;
  std::int32_t *col_idx_ = nullptr;
  double *values_ = nullptr;
};

} // namespace acg::solver

#endif
