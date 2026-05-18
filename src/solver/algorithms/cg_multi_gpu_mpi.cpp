#include "acg/solver/algorithms/cg_multi_gpu_mpi.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <sycl/sycl.hpp>

#include "../solver_common.hpp"
#include "acg/solver/backends/onemath_cuda_backend.hpp"
#include "acg/solver/stopping_criteria.hpp"

namespace acg::solver::algorithms {

namespace {

bool diagnostics_enabled() {
  const char *value = std::getenv("ACG_SOLVER_DIAGNOSTICS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int diagnostic_iteration_limit() {
  const char *value = std::getenv("ACG_SOLVER_DIAG_ITERS");
  if (value == nullptr || value[0] == '\0') {
    return 5;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : 0;
}

template <typename T>
struct DeviceBufferDeleter {
  sycl::queue *queue = nullptr;

  void operator()(T *ptr) const {
    if (ptr != nullptr && queue != nullptr) {
      sycl::free(ptr, *queue);
    }
  }
};

template <typename T>
using DeviceBuffer = std::unique_ptr<T, DeviceBufferDeleter<T>>;

template <typename T>
DeviceBuffer<T> make_device_buffer(sycl::queue &queue, std::int64_t count) {
  if (count == 0) {
    return DeviceBuffer<T>(nullptr, DeviceBufferDeleter<T>{.queue = &queue});
  }
  T *ptr = sycl::malloc_device<T>(static_cast<std::size_t>(count), queue);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device buffer");
  }
  return DeviceBuffer<T>(ptr, DeviceBufferDeleter<T>{.queue = &queue});
}

template <typename T>
DeviceBuffer<T> make_device_buffer_from(sycl::queue &queue, const std::vector<T> &src) {
  auto buf = make_device_buffer<T>(queue, static_cast<std::int64_t>(src.size()));
  if (!src.empty()) {
    queue.memcpy(buf.get(), src.data(), src.size() * sizeof(T)).wait();
  }
  return buf;
}

double mpi_allreduce_sum(double value) {
#ifdef ACG_HAVE_MPI
  double global_value = 0.0;
  if (MPI_Allreduce(&value, &global_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed");
  }
  return global_value;
#else
  return value;
#endif
}

void mpi_allreduce_sum_device_in_place(double *value) {
#ifdef ACG_HAVE_MPI
  if (MPI_Allreduce(MPI_IN_PLACE, value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed for device scalar");
  }
#else
  (void)value;
#endif
}

class DeviceScalar {
public:
  explicit DeviceScalar(backends::OnemathCudaBackend &backend, double initial_value = 0.0)
      : backend_(&backend), ptr_(backend.create_device_scalar(initial_value)) {}

  ~DeviceScalar() {
    if (backend_ != nullptr) {
      backend_->destroy_device_scalar(ptr_);
    }
  }

  DeviceScalar(const DeviceScalar &) = delete;
  DeviceScalar &operator=(const DeviceScalar &) = delete;

  double *get() const noexcept { return ptr_; }

private:
  backends::OnemathCudaBackend *backend_;
  double *ptr_;
};

class DeviceHaloCsrSpmv {
public:
  DeviceHaloCsrSpmv(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix)
      : queue_(&queue), rows_(matrix.rows), cols_(matrix.cols), nnz_(matrix.nnz()), active_rows_(active_halo_rows(matrix)),
        row_ptr_(make_device_buffer_from(queue, matrix.row_ptr)), col_idx_(make_device_buffer_from(queue, matrix.col_idx)),
        values_(make_device_buffer_from(queue, matrix.values)), device_active_rows_(make_device_buffer_from(queue, active_rows_)),
        bytes_(static_cast<std::int64_t>(active_rows_.size()) * (2LL * sizeof(std::int64_t) + sizeof(double))
               + nnz_ * (sizeof(double) + sizeof(std::int32_t) + sizeof(double))) {}

  void spmv_add(const backends::DeviceVector &x, backends::DeviceVector &y) {
    if (x.size != cols_) {
      throw std::runtime_error("halo spmv input size mismatch");
    }
    if (y.size != rows_) {
      throw std::runtime_error("halo spmv output size mismatch");
    }
    if (active_rows_.empty()) {
      return;
    }

    const std::size_t active_rows = active_rows_.size();
    const std::int64_t *active_row_indices = device_active_rows_.get();
    const std::int64_t *row_ptr = row_ptr_.get();
    const std::int32_t *col_idx = col_idx_.get();
    const double *values = values_.get();
    const double *x_data = x.data;
    double *y_data = y.data;
    queue_->submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range<1>(active_rows), [=](sycl::id<1> idx) {
        const std::int64_t row = active_row_indices[idx[0]];
        double sum = 0.0;
        for (std::int64_t jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) {
          sum += values[jj] * x_data[col_idx[jj]];
        }
        y_data[row] += sum;
      });
    }).wait();
  }

  [[nodiscard]] std::int64_t estimated_spmv_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::int64_t active_row_count() const noexcept { return static_cast<std::int64_t>(active_rows_.size()); }

private:
  static std::vector<std::int64_t> active_halo_rows(const acg::matrix::CsrMatrix<double> &matrix) {
    std::vector<std::int64_t> rows;
    rows.reserve(static_cast<std::size_t>(matrix.rows));
    for (std::int64_t row = 0; row < matrix.rows; ++row) {
      if (matrix.row_ptr[static_cast<std::size_t>(row)] != matrix.row_ptr[static_cast<std::size_t>(row + 1)]) {
        rows.push_back(row);
      }
    }
    return rows;
  }

  sycl::queue *queue_;
  std::int64_t rows_;
  std::int64_t cols_;
  std::int64_t nnz_;
  std::vector<std::int64_t> active_rows_;
  DeviceBuffer<std::int64_t> row_ptr_;
  DeviceBuffer<std::int32_t> col_idx_;
  DeviceBuffer<double> values_;
  DeviceBuffer<std::int64_t> device_active_rows_;
  std::int64_t bytes_;
};

void update_search_direction(sycl::queue &queue, double beta, const backends::DeviceVector &r, backends::DeviceVector &s) {
  if (r.size != s.size) {
    throw std::runtime_error("search direction update size mismatch");
  }
  const std::int64_t n = r.size;
  const double *r_data = r.data;
  double *s_data = s.data;
  queue.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> idx) {
      const std::size_t i = idx[0];
      s_data[i] = beta * s_data[i] + r_data[i];
    });
  }).wait();
}

void update_solution_and_residual(
    sycl::queue &queue,
    double alpha,
    const backends::DeviceVector &s,
    const backends::DeviceVector &t,
    backends::DeviceVector &x,
    backends::DeviceVector &r) {
  if (s.size != t.size || s.size != x.size || s.size != r.size) {
    throw std::runtime_error("solution/residual update size mismatch");
  }
  const std::int64_t n = s.size;
  const double *s_data = s.data;
  const double *t_data = t.data;
  double *x_data = x.data;
  double *r_data = r.data;
  queue.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> idx) {
      const std::size_t i = idx[0];
      r_data[i] -= alpha * t_data[i];
      x_data[i] += alpha * s_data[i];
    });
  }).wait();
}

std::int64_t import_elements(const acg::matrix::DistributedCsrMatrixPartition &partition) {
  std::int64_t total = 0;
  for (const auto &peer : partition.imports) {
    total += static_cast<std::int64_t>(peer.global_columns.size());
  }
  return total;
}

std::int64_t export_elements(const acg::matrix::DistributedCsrMatrixPartition &partition) {
  std::int64_t total = 0;
  for (const auto &peer : partition.exports) {
    total += static_cast<std::int64_t>(peer.local_indices.size());
  }
  return total;
}

void print_partition_diagnostics(const acg::matrix::DistributedCsrMatrixPartition &partition, const acg::runtime::RunContext &ctx) {
#ifdef ACG_HAVE_MPI
  if (ctx.rank == 0) {
    std::cout << "solver_diag_partition_method: " << partition.method;
    if (partition.objective >= 0) {
      std::cout << " objective=" << partition.objective;
    }
    std::cout << '\n';
  }
  const long long local_stats[] = {
      static_cast<long long>(partition.local_rows()),
      static_cast<long long>(partition.local_matrix.nnz()),
      static_cast<long long>(partition.interior_matrix.nnz()),
      static_cast<long long>(partition.halo_matrix.nnz()),
      static_cast<long long>(partition.ghost_global_columns.size()),
      static_cast<long long>(import_elements(partition)),
      static_cast<long long>(export_elements(partition)),
  };
  std::vector<long long> all_stats;
  if (ctx.rank == 0) {
    all_stats.resize(static_cast<std::size_t>(ctx.size) * 7);
  }
  MPI_Gather(local_stats, 7, MPI_LONG_LONG, ctx.rank == 0 ? all_stats.data() : nullptr, 7, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  if (ctx.rank == 0) {
    std::cout << "solver_diag_partition: rank local_rows local_nnz interior_nnz halo_nnz ghosts imports exports\n";
    for (int rank = 0; rank < ctx.size; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(rank) * 7;
      std::cout << "solver_diag_partition: " << rank
                << ' ' << all_stats[offset + 0]
                << ' ' << all_stats[offset + 1]
                << ' ' << all_stats[offset + 2]
                << ' ' << all_stats[offset + 3]
                << ' ' << all_stats[offset + 4]
                << ' ' << all_stats[offset + 5]
                << ' ' << all_stats[offset + 6] << '\n';
    }
  }
#else
  (void)partition;
  (void)ctx;
#endif
}

void print_device_diagnostics(const acg::runtime::RunContext &ctx) {
  const sycl::device device = ctx.queue.get_device();
  const std::string name = device.get_info<sycl::info::device::name>();
  const std::string vendor = device.get_info<sycl::info::device::vendor>();
  const std::string driver = device.get_info<sycl::info::device::driver_version>();
  const int backend = static_cast<int>(device.get_backend());
  const char *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
#ifdef ACG_HAVE_MPI
  for (int rank = 0; rank < ctx.size; ++rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank == rank) {
      std::cout << "solver_diag_device: rank=" << ctx.rank
                << " local_rank=" << ctx.local_rank
                << " backend=" << backend
                << " cuda_visible_devices=" << (cuda_visible_devices != nullptr ? cuda_visible_devices : "")
                << " vendor=\"" << vendor << "\""
                << " name=\"" << name << "\""
                << " driver=\"" << driver << "\"\n";
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  std::cout << "solver_diag_device: rank=" << ctx.rank
            << " local_rank=" << ctx.local_rank
            << " backend=" << backend
            << " cuda_visible_devices=" << (cuda_visible_devices != nullptr ? cuda_visible_devices : "")
            << " vendor=\"" << vendor << "\""
            << " name=\"" << name << "\""
            << " driver=\"" << driver << "\"\n";
#endif
}

void print_op_diagnostic(const char *name, double local_time, int rank, int size) {
#ifdef ACG_HAVE_MPI
  double min_time = 0.0;
  double max_time = 0.0;
  double sum_time = 0.0;
  MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "solver_diag_perf: " << name
              << " min_s=" << min_time
              << " avg_s=" << (sum_time / static_cast<double>(size))
              << " max_s=" << max_time
              << " skew_s=" << (max_time - min_time) << '\n';
  }
#else
  (void)name;
  (void)local_time;
  (void)rank;
  (void)size;
#endif
}

void print_rank_perf_diagnostics(
    const SolverResult &result,
    const acg::runtime::RunContext &ctx,
    double interior_spmv_time,
    double halo_spmv_time) {
#ifdef ACG_HAVE_MPI
  const double local_stats[] = {
      interior_spmv_time,
      halo_spmv_time,
      result.perf.native.spmv.time_seconds,
      result.perf.native.dot.time_seconds,
      result.perf.native.nrm2.time_seconds,
      result.perf.native.axpy.time_seconds,
      result.perf.native.pack.time_seconds,
      result.perf.native.p2p.time_seconds,
      result.perf.native.allreduce.time_seconds,
      result.perf.native.host_sync.time_seconds,
  };
  std::vector<double> all_stats;
  if (ctx.rank == 0) {
    all_stats.resize(static_cast<std::size_t>(ctx.size) * 10);
  }
  MPI_Gather(
      local_stats,
      10,
      MPI_DOUBLE,
      ctx.rank == 0 ? all_stats.data() : nullptr,
      10,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD);
  if (ctx.rank == 0) {
    std::cout << "solver_diag_rank_perf: rank interior_spmv_s halo_spmv_s total_spmv_s dot_s nrm2_s axpy_s pack_s p2p_s allreduce_s host_sync_s\n";
    for (int rank = 0; rank < ctx.size; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(rank) * 10;
      std::cout << "solver_diag_rank_perf: " << rank;
      for (int i = 0; i < 10; ++i) {
        std::cout << ' ' << all_stats[offset + static_cast<std::size_t>(i)];
      }
      std::cout << '\n';
    }
  }
#else
  (void)result;
  (void)ctx;
  (void)interior_spmv_time;
  (void)halo_spmv_time;
#endif
}

void print_perf_diagnostics(
    const SolverResult &result,
    const acg::runtime::RunContext &ctx,
    double interior_spmv_time,
    double halo_spmv_time) {
  print_op_diagnostic("interior_spmv", interior_spmv_time, ctx.rank, ctx.size);
  print_op_diagnostic("halo_spmv", halo_spmv_time, ctx.rank, ctx.size);
  print_op_diagnostic("spmv", result.perf.native.spmv.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("dot", result.perf.native.dot.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("nrm2", result.perf.native.nrm2.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("axpy", result.perf.native.axpy.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("copy", result.perf.native.copy.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("pack", result.perf.native.pack.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("p2p", result.perf.native.p2p.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("allreduce", result.perf.native.allreduce.time_seconds, ctx.rank, ctx.size);
  print_op_diagnostic("host_sync", result.perf.native.host_sync.time_seconds, ctx.rank, ctx.size);
  print_rank_perf_diagnostics(result, ctx, interior_spmv_time, halo_spmv_time);
}

void print_iteration_perf_diagnostics(
    const acg::runtime::RunContext &ctx,
    int iteration,
    const PerfBreakdown &before,
    const PerfBreakdown &after,
    double interior_spmv_time,
    double halo_spmv_time) {
#ifdef ACG_HAVE_MPI
  const double local_stats[] = {
      interior_spmv_time,
      halo_spmv_time,
      after.native.spmv.time_seconds - before.native.spmv.time_seconds,
      after.native.dot.time_seconds - before.native.dot.time_seconds,
      after.native.nrm2.time_seconds - before.native.nrm2.time_seconds,
      after.native.axpy.time_seconds - before.native.axpy.time_seconds,
      after.native.pack.time_seconds - before.native.pack.time_seconds,
      after.native.p2p.time_seconds - before.native.p2p.time_seconds,
      after.native.allreduce.time_seconds - before.native.allreduce.time_seconds,
      after.native.host_sync.time_seconds - before.native.host_sync.time_seconds,
  };
  std::vector<double> all_stats;
  if (ctx.rank == 0) {
    all_stats.resize(static_cast<std::size_t>(ctx.size) * 10);
  }
  MPI_Gather(
      local_stats,
      10,
      MPI_DOUBLE,
      ctx.rank == 0 ? all_stats.data() : nullptr,
      10,
      MPI_DOUBLE,
      0,
      MPI_COMM_WORLD);
  if (ctx.rank == 0) {
    if (iteration == 0) {
      std::cout << "solver_diag_iter_perf: iter rank interior_spmv_s halo_spmv_s total_spmv_s dot_s nrm2_s axpy_s pack_s p2p_s allreduce_s host_sync_s\n";
    }
    for (int rank = 0; rank < ctx.size; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(rank) * 10;
      std::cout << "solver_diag_iter_perf: " << iteration << ' ' << rank;
      for (int i = 0; i < 10; ++i) {
        std::cout << ' ' << all_stats[offset + static_cast<std::size_t>(i)];
      }
      std::cout << '\n';
    }
  }
#else
  (void)ctx;
  (void)iteration;
  (void)before;
  (void)after;
  (void)interior_spmv_time;
  (void)halo_spmv_time;
#endif
}

struct HaloSendState {
  int rank = -1;
  std::int64_t count = 0;
  DeviceBuffer<std::int64_t> device_indices;
  DeviceBuffer<double> buffer;
  std::vector<double> host_buffer;
};

struct HaloRecvState {
  int rank = -1;
  std::int64_t ghost_offset = 0;
  std::vector<double> host_buffer;
};

class HaloExchange {
public:
  HaloExchange(
      sycl::queue &queue,
      const acg::matrix::DistributedCsrMatrixPartition &partition)
      : queue_(queue), partition_(partition) {
    for (const auto &export_peer : partition.exports) {
      const std::int64_t n = static_cast<std::int64_t>(export_peer.local_indices.size());
      send_states_.push_back(HaloSendState{
          .rank = export_peer.rank,
          .count = n,
          .device_indices = make_device_buffer_from(queue_, export_peer.local_indices),
          .buffer = make_device_buffer<double>(queue_, n),
          .host_buffer = std::vector<double>(static_cast<std::size_t>(n)),
      });
    }
    for (const auto &import_peer : partition.imports) {
      recv_states_.push_back(HaloRecvState{
          .rank = import_peer.rank,
          .ghost_offset = import_peer.ghost_offset,
          .host_buffer = std::vector<double>(import_peer.global_columns.size()),
      });
    }
  }

  virtual ~HaloExchange() = default;

  virtual void begin(const backends::DeviceVector &s, backends::DeviceVector &shat, SolverResult &result) {
    if (partition_.ghost_global_columns.empty()) {
      return;
    }

    const double pack_time = timed_call([&] {
      submit_pack_kernels(s);
      wait_for_pack();
      prepare_send_buffers();
    });

    result.perf.native.pack.record(pack_time, 2 * sizeof(double) * packed_elements());
    result.perf.native.host_sync.record(pack_time, 0);
    post_receives(shat);
    post_sends();
  }

  virtual void finish(backends::DeviceVector &shat, SolverResult &result) {
    if (requests_.empty()) {
      return;
    }

    const double wait_time = timed_call([&] {
      if (MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Waitall failed");
      }
    });
    result.perf.native.p2p.record(wait_time, sizeof(double) * exchanged_elements());
    result.perf.cuda.haloexchange.record(wait_time, sizeof(double) * packed_elements());
    requests_.clear();

    const double unpack_time = timed_call([&] { finalize_receives(shat); });
    record_post_wait(unpack_time, result);
  }

  [[nodiscard]] bool has_halo() const noexcept { return !partition_.ghost_global_columns.empty(); }

protected:
  virtual void prepare_send_buffers() = 0;
  virtual void post_receives(backends::DeviceVector &shat) = 0;
  virtual void post_sends() = 0;
  virtual void finalize_receives(backends::DeviceVector &shat) = 0;
  virtual void record_post_wait(double unpack_time, SolverResult &result) = 0;

  void submit_pack_kernels(const backends::DeviceVector &s) {
    pack_events_.clear();
    pack_events_.reserve(send_states_.size());
    for (const auto &send_state : send_states_) {
      if (send_state.count == 0) {
        continue;
      }
      const std::int64_t *indices = send_state.device_indices.get();
      double *dst = send_state.buffer.get();
      const double *src = s.data;
      const std::size_t n = static_cast<std::size_t>(send_state.count);
      pack_events_.push_back(queue_.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
          const std::size_t i = idx[0];
          dst[i] = src[indices[i]];
        });
      }));
    }
  }

  void wait_for_pack() {
    sycl::event::wait_and_throw(pack_events_);
    pack_events_.clear();
  }

  [[nodiscard]] std::int64_t packed_elements() const noexcept {
    std::int64_t packed = 0;
    for (const auto &send_state : send_states_) {
      packed += send_state.count;
    }
    return packed;
  }

  [[nodiscard]] std::int64_t exchanged_elements() const noexcept {
    std::int64_t exchanged = 0;
    for (const auto &recv_state : recv_states_) {
      exchanged += static_cast<std::int64_t>(recv_state.host_buffer.size());
    }
    for (const auto &send_state : send_states_) {
      exchanged += send_state.count;
    }
    return exchanged;
  }

  sycl::queue &queue_;
  const acg::matrix::DistributedCsrMatrixPartition &partition_;
  std::vector<HaloSendState> send_states_;
  std::vector<HaloRecvState> recv_states_;
  std::vector<MPI_Request> requests_;
  std::vector<sycl::event> pack_events_;
};

class HostStagedHaloExchange final : public HaloExchange {
public:
  using HaloExchange::HaloExchange;

private:
  void prepare_send_buffers() override {
    for (auto &send_state : send_states_) {
      if (send_state.host_buffer.empty()) {
        continue;
      }
      queue_.memcpy(
          send_state.host_buffer.data(),
          send_state.buffer.get(),
          send_state.host_buffer.size() * sizeof(double))
          .wait();
    }
  }

  void post_receives(backends::DeviceVector &shat) override {
    (void)shat;
    requests_.clear();
    requests_.reserve(recv_states_.size() + send_states_.size());
    for (auto &recv_state : recv_states_) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (MPI_Irecv(
              recv_state.host_buffer.data(),
              static_cast<int>(recv_state.host_buffer.size()),
              MPI_DOUBLE,
              recv_state.rank,
              0,
              MPI_COMM_WORLD,
              &request)
          != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Irecv failed");
      }
      requests_.push_back(request);
    }
  }

  void post_sends() override {
    for (const auto &send_state : send_states_) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (MPI_Isend(
              send_state.host_buffer.data(),
              static_cast<int>(send_state.host_buffer.size()),
              MPI_DOUBLE,
              send_state.rank,
              0,
              MPI_COMM_WORLD,
              &request)
          != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Isend failed");
      }
      requests_.push_back(request);
    }
  }

  void finalize_receives(backends::DeviceVector &shat) override {
    for (const auto &recv_state : recv_states_) {
      if (recv_state.host_buffer.empty()) {
        continue;
      }
      queue_.memcpy(
          shat.data + recv_state.ghost_offset,
          recv_state.host_buffer.data(),
          recv_state.host_buffer.size() * sizeof(double))
          .wait();
    }
  }

  void record_post_wait(double unpack_time, SolverResult &result) override {
    result.perf.native.copy.record(unpack_time, sizeof(double) * exchanged_elements());
    result.perf.native.host_sync.record(unpack_time, 0);
  }
};

class GpuAwareHaloExchange final : public HaloExchange {
public:
  using HaloExchange::HaloExchange;

private:
  void prepare_send_buffers() override {
#ifndef ACG_HAVE_GPU_AWARE_MPI
    throw std::runtime_error("gpu-aware MPI mode requires a build with ACG_ENABLE_GPU_AWARE_MPI=ON");
#endif
  }

  void post_receives(backends::DeviceVector &shat) override {
#ifndef ACG_HAVE_GPU_AWARE_MPI
    (void)shat;
    throw std::runtime_error("gpu-aware MPI mode requires a build with ACG_ENABLE_GPU_AWARE_MPI=ON");
#else
    requests_.clear();
    requests_.reserve(recv_states_.size() + send_states_.size());
    for (const auto &recv_state : recv_states_) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (MPI_Irecv(
              shat.data + recv_state.ghost_offset,
              static_cast<int>(recv_state.host_buffer.size()),
              MPI_DOUBLE,
              recv_state.rank,
              0,
              MPI_COMM_WORLD,
              &request)
          != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Irecv failed for device buffer");
      }
      requests_.push_back(request);
    }
#endif
  }

  void post_sends() override {
#ifndef ACG_HAVE_GPU_AWARE_MPI
    throw std::runtime_error("gpu-aware MPI mode requires a build with ACG_ENABLE_GPU_AWARE_MPI=ON");
#else
    for (const auto &send_state : send_states_) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (MPI_Isend(
              send_state.buffer.get(),
              static_cast<int>(send_state.count),
              MPI_DOUBLE,
              send_state.rank,
              0,
              MPI_COMM_WORLD,
              &request)
          != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Isend failed for device buffer");
      }
      requests_.push_back(request);
    }
#endif
  }

  void finalize_receives(backends::DeviceVector &shat) override {
#ifndef ACG_HAVE_GPU_AWARE_MPI
    (void)shat;
    throw std::runtime_error("gpu-aware MPI mode requires a build with ACG_ENABLE_GPU_AWARE_MPI=ON");
#else
    (void)shat;
#endif
  }

  void record_post_wait(double unpack_time, SolverResult &result) override {
    (void)unpack_time;
    result.perf.native.host_sync.record(0.0, 0);
  }

};

std::unique_ptr<HaloExchange> make_halo_exchange(
    sycl::queue &queue,
    const acg::matrix::DistributedCsrMatrixPartition &partition,
    const SolverOptions &options) {
  switch (options.mpi_mode) {
  case MpiMode::Host:
    return std::make_unique<HostStagedHaloExchange>(queue, partition);
  case MpiMode::GpuAware: {
#ifndef ACG_HAVE_GPU_AWARE_MPI
    throw std::runtime_error("--mpi-mode gpu-aware requires a build with ACG_ENABLE_GPU_AWARE_MPI=ON");
#else
    if (!queue.get_device().is_gpu()) {
      throw std::runtime_error("--mpi-mode gpu-aware requires a GPU device");
    }
    return std::make_unique<GpuAwareHaloExchange>(queue, partition);
#endif
  }
  }
  throw std::runtime_error("unsupported MPI mode");
}

double compute_relative_solution_error(
    backends::SingleGpuCgBackend &backend,
    const backends::DeviceVector &x,
    const backends::DeviceVector &x_exact,
    backends::DeviceVector &x_error,
    double x_exact_norm,
    SolverResult &result) {
  result.perf.native.copy.record(timed_call([&] { backend.copy(x, x_error); }), backend.estimated_copy_bytes());
  result.perf.native.axpy.record(timed_call([&] { backend.axpy(-1.0, x_exact, x_error); }), backend.estimated_axpy_bytes());
  double local_error_norm_squared = 0.0;
  result.perf.native.nrm2.record(
      timed_call([&] { local_error_norm_squared = backend.dot(x_error, x_error); }),
      backend.estimated_nrm2_bytes());
  const double error_norm_squared = mpi_allreduce_sum(local_error_norm_squared);
  return x_exact_norm > 0.0 ? std::sqrt(error_norm_squared) / x_exact_norm : 0.0;
}

} // namespace

SolverResult run_cg_multi_gpu_mpi(
    const acg::matrix::DistributedCsrMatrixPartition &partition,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    const std::vector<double> &x_exact_local_host,
    const std::vector<double> &b_local_host) {
#ifndef ACG_HAVE_MPI
  (void)partition;
  (void)ctx;
  (void)options;
  (void)x_exact_local_host;
  (void)b_local_host;
  throw std::runtime_error("MPI support is not enabled in this build");
#else
  if (partition.local_rows() <= 0) {
    throw std::runtime_error("local partition is empty");
  }

  sycl::queue queue = ctx.queue;
  backends::OnemathCudaBackend interior_backend(queue, partition.interior_matrix);
  std::unique_ptr<DeviceHaloCsrSpmv> halo_spmv;
  if (partition.halo_matrix.nnz() > 0) {
    halo_spmv = std::make_unique<DeviceHaloCsrSpmv>(queue, partition.halo_matrix);
  }

  const double interior_spmv_flops = 2.0 * static_cast<double>(partition.interior_matrix.nnz());
  const double halo_spmv_flops = 2.0 * static_cast<double>(partition.halo_matrix.nnz());
  const double dot_flops = 2.0 * static_cast<double>(partition.local_rows());
  const double update_x_r_flops = 4.0 * static_cast<double>(partition.local_rows());
  const double update_s_flops = 3.0 * static_cast<double>(partition.local_rows());
  constexpr std::int64_t cuda_value_bytes = sizeof(double);
  constexpr std::int64_t cuda_index_bytes = 4;
  const std::int64_t local_rows = partition.local_rows();
  const std::int64_t ghost_rows = static_cast<std::int64_t>(partition.ghost_global_columns.size());
  const std::int64_t local_nnz = static_cast<std::int64_t>(partition.interior_matrix.nnz() + partition.halo_matrix.nnz());
  const std::int64_t cuda_vector_entries = local_rows + ghost_rows;
  const std::int64_t cuda_gemv_bytes =
      local_nnz * (cuda_value_bytes + cuda_index_bytes)
      + local_rows * (cuda_index_bytes + cuda_value_bytes)
      + (partition.halo_matrix.nnz() > 0 ? local_rows * cuda_index_bytes : 0)
      + cuda_vector_entries * cuda_value_bytes;
  const std::int64_t cuda_dot_bytes = local_rows * 2 * cuda_value_bytes;
  const std::int64_t cuda_nrm2_bytes = local_rows * cuda_value_bytes;
  const std::int64_t cuda_axpy_bytes = local_rows * 2 * cuda_value_bytes;
  const std::int64_t cuda_copy_bytes = local_rows * 2 * cuda_value_bytes;
  const double cuda_gemv_flops = 3.0 * static_cast<double>(local_nnz);
  const double cuda_axpy_flops = 2.0 * static_cast<double>(local_rows);

  SolverResult result;
  double interior_spmv_time = 0.0;
  double halo_spmv_time = 0.0;
  backends::DeviceVector x = interior_backend.create_vector();
  backends::DeviceVector r = interior_backend.create_vector();
  backends::DeviceVector s = interior_backend.create_vector();
  backends::DeviceVector t = interior_backend.create_vector();
  backends::DeviceVector b = interior_backend.create_vector_from_host(b_local_host);
  backends::DeviceVector shat = interior_backend.create_vector(static_cast<std::int64_t>(partition.ghost_global_columns.size()));
  std::optional<backends::DeviceVector> x_exact;
  std::optional<backends::DeviceVector> x_error;
  std::unique_ptr<HaloExchange> halo_exchange = make_halo_exchange(queue, partition, options);
  const bool use_device_scalar_reductions = options.mpi_mode == MpiMode::GpuAware;
  const bool emit_diagnostics = diagnostics_enabled();
  const int diag_iteration_limit = emit_diagnostics ? diagnostic_iteration_limit() : 0;
  if (emit_diagnostics) {
    print_device_diagnostics(ctx);
    print_partition_diagnostics(partition, ctx);
  }
  DeviceScalar rho_scalar(interior_backend);
  DeviceScalar gamma_scalar(interior_backend);
  DeviceScalar rho_next_scalar(interior_backend);
  const auto cuda_reference_solve_start = std::chrono::steady_clock::now();

  const double x_exact_norm = options.manufactured_solution
                                  ? std::sqrt(mpi_allreduce_sum(host_squared_norm(x_exact_local_host)))
                                  : 0.0;
  const bool track_solution_error = options.manufactured_solution && (options.solution_relative_tolerance > 0.0 || options.log_every > 0);
  if (options.manufactured_solution && track_solution_error) {
    x_exact = interior_backend.create_vector_from_host(x_exact_local_host);
    x_error = interior_backend.create_vector();
  }

  interior_backend.fill_zero(x);
  interior_backend.copy(b, r);
  interior_backend.copy(r, s);
  result.perf.native.copy.record(0.0, 2 * interior_backend.estimated_copy_bytes());
  result.perf.cuda.copy.record(0.0, cuda_copy_bytes);
  result.perf.cuda.copy.record(0.0, cuda_copy_bytes);

  const double rhs_norm = std::sqrt(mpi_allreduce_sum(host_squared_norm(b_local_host)));
  result.perf.cuda.nrm2.record(0.0, cuda_nrm2_bytes);
  result.cuda_reference_total_flops += dot_flops;
  if (ctx.size > 1) {
    result.perf.cuda.allreduce.record(0.0, sizeof(double));
  }

  double rho_local = 0.0;
  if (use_device_scalar_reductions) {
    const double nrm2_time = timed_call([&] { interior_backend.dot_to_device(r, r, rho_scalar.get()); });
    result.perf.native.nrm2.record(nrm2_time, interior_backend.estimated_nrm2_bytes());
    result.perf.cuda.nrm2.record(nrm2_time, cuda_nrm2_bytes);
  }
  else {
    const double nrm2_time = timed_call([&] { rho_local = interior_backend.dot(r, r); });
    result.perf.native.nrm2.record(nrm2_time, interior_backend.estimated_nrm2_bytes());
    result.perf.cuda.nrm2.record(nrm2_time, cuda_nrm2_bytes);
  }
  result.total_flops += dot_flops;
  result.cuda_reference_total_flops += dot_flops;

  double rho = 0.0;
  if (use_device_scalar_reductions) {
    const double allreduce_time = timed_call([&] { mpi_allreduce_sum_device_in_place(rho_scalar.get()); });
    result.perf.native.allreduce.record(allreduce_time, sizeof(double));
    result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
    result.perf.native.host_sync.record(timed_call([&] { rho = interior_backend.read_device_scalar(rho_scalar.get()); }), sizeof(double));
  }
  else {
    const double allreduce_time = timed_call([&] { rho = mpi_allreduce_sum(rho_local); });
    result.perf.native.allreduce.record(allreduce_time, sizeof(double));
    result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
  }
  const double rho0 = rho;
  const double r0_norm = std::sqrt(rho0);
  const CgThresholds thresholds = make_cg_thresholds(options, 0.0, r0_norm);
  result.rhs_norm = rhs_norm;
  result.initial_residual = r0_norm;
  result.final_residual = std::sqrt(rho0);
  result.relative_residual_to_initial = 1.0;
  result.relative_residual_to_rhs = rhs_norm > 0.0 ? result.final_residual / rhs_norm : 0.0;
  result.converged = cg_residual_converged(result.final_residual, thresholds);

  const auto solve_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < options.max_iterations && !result.converged; ++iteration) {
    const bool emit_iteration_diagnostics = iteration < diag_iteration_limit;
    const PerfBreakdown perf_before_iteration = result.perf;
    double iteration_interior_spmv_time = 0.0;
    double iteration_halo_spmv_time = 0.0;
    halo_exchange->begin(s, shat, result);

    const double interior_time = timed_call([&] { interior_backend.spmv(s, t); });
    interior_spmv_time += interior_time;
    iteration_interior_spmv_time += interior_time;
    result.perf.native.spmv.record(interior_time, interior_backend.estimated_spmv_bytes());
    result.total_flops += interior_spmv_flops;

    if (halo_exchange->has_halo()) {
      halo_exchange->finish(shat, result);
      if (halo_spmv != nullptr) {
        const double halo_time = timed_call([&] { halo_spmv->spmv_add(shat, t); });
        halo_spmv_time += halo_time;
        iteration_halo_spmv_time += halo_time;
        result.perf.native.spmv.record(halo_time, halo_spmv->estimated_spmv_bytes());
        result.total_flops += halo_spmv_flops + static_cast<double>(halo_spmv->active_row_count());
      }
    }
    result.perf.cuda.gemv.record(iteration_interior_spmv_time + iteration_halo_spmv_time, cuda_gemv_bytes);
    result.cuda_reference_total_flops += cuda_gemv_flops;

    double gamma_local = 0.0;
    if (use_device_scalar_reductions) {
      const double dot_time = timed_call([&] { interior_backend.dot_to_device(s, t, gamma_scalar.get()); });
      result.perf.native.dot.record(dot_time, interior_backend.estimated_dot_bytes());
      result.perf.cuda.dot.record(dot_time, cuda_dot_bytes);
    }
    else {
      const double dot_time = timed_call([&] { gamma_local = interior_backend.dot(s, t); });
      result.perf.native.dot.record(dot_time, interior_backend.estimated_dot_bytes());
      result.perf.cuda.dot.record(dot_time, cuda_dot_bytes);
    }
    result.total_flops += dot_flops;
    result.cuda_reference_total_flops += dot_flops;
    double gamma = 0.0;
    if (use_device_scalar_reductions) {
      const double allreduce_time = timed_call([&] { mpi_allreduce_sum_device_in_place(gamma_scalar.get()); });
      result.perf.native.allreduce.record(allreduce_time, sizeof(double));
      result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
      result.perf.native.host_sync.record(timed_call([&] { gamma = interior_backend.read_device_scalar(gamma_scalar.get()); }), sizeof(double));
    }
    else {
      const double allreduce_time = timed_call([&] { gamma = mpi_allreduce_sum(gamma_local); });
      result.perf.native.allreduce.record(allreduce_time, sizeof(double));
      result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
    }
    if (gamma <= 0.0) {
      throw std::runtime_error("matrix is not positive definite under CG iteration");
    }

    const double alpha = rho / gamma;

    const double update_x_r_time = timed_call([&] { update_solution_and_residual(queue, alpha, s, t, x, r); });
    result.perf.native.axpy.record(update_x_r_time, 2 * interior_backend.estimated_axpy_bytes());
    result.perf.cuda.axpy.record(0.5 * update_x_r_time, cuda_axpy_bytes);
    result.perf.cuda.axpy.record(0.5 * update_x_r_time, cuda_axpy_bytes);
    result.total_flops += update_x_r_flops;
    result.cuda_reference_total_flops += 2.0 * cuda_axpy_flops;

    double rho_next_local = 0.0;
    if (use_device_scalar_reductions) {
      const double nrm2_time = timed_call([&] { interior_backend.dot_to_device(r, r, rho_next_scalar.get()); });
      result.perf.native.nrm2.record(nrm2_time, interior_backend.estimated_nrm2_bytes());
      result.perf.cuda.nrm2.record(nrm2_time, cuda_nrm2_bytes);
    }
    else {
      const double nrm2_time = timed_call([&] { rho_next_local = interior_backend.dot(r, r); });
      result.perf.native.nrm2.record(nrm2_time, interior_backend.estimated_nrm2_bytes());
      result.perf.cuda.nrm2.record(nrm2_time, cuda_nrm2_bytes);
    }
    result.total_flops += dot_flops;
    result.cuda_reference_total_flops += dot_flops;

    double rho_next = 0.0;
    if (use_device_scalar_reductions) {
      const double allreduce_time = timed_call([&] { mpi_allreduce_sum_device_in_place(rho_next_scalar.get()); });
      result.perf.native.allreduce.record(allreduce_time, sizeof(double));
      result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
      result.perf.native.host_sync.record(timed_call([&] { rho_next = interior_backend.read_device_scalar(rho_next_scalar.get()); }), sizeof(double));
    }
    else {
      const double allreduce_time = timed_call([&] { rho_next = mpi_allreduce_sum(rho_next_local); });
      result.perf.native.allreduce.record(allreduce_time, sizeof(double));
      result.perf.cuda.allreduce.record(allreduce_time, sizeof(double));
    }

    result.iterations = iteration + 1;
    result.final_residual = std::sqrt(rho_next);
    const ResidualDiagnostics residual_diagnostics = make_residual_diagnostics(rhs_norm, r0_norm, result.final_residual);
    result.relative_residual_to_initial = residual_diagnostics.relative_to_initial;
    result.relative_residual_to_rhs = residual_diagnostics.relative_to_rhs;
    if (track_solution_error) {
      result.relative_solution_error = compute_relative_solution_error(
          interior_backend,
          x,
          *x_exact,
          *x_error,
          x_exact_norm,
          result);
    }
    result.converged = cg_residual_converged(result.final_residual, thresholds);
    if (options.solution_relative_tolerance > 0.0 && options.manufactured_solution) {
      result.converged = result.converged || result.relative_solution_error <= options.solution_relative_tolerance;
    }
    maybe_log_progress(ctx, options, result.iterations, result);
    if (result.converged) {
      if (emit_iteration_diagnostics) {
        print_iteration_perf_diagnostics(
            ctx,
            iteration,
            perf_before_iteration,
            result.perf,
            iteration_interior_spmv_time,
            iteration_halo_spmv_time);
      }
      break;
    }

    const double beta = rho_next / rho;
    const double update_s_time = timed_call([&] { update_search_direction(queue, beta, r, s); });
    result.perf.native.axpy.record(update_s_time, interior_backend.estimated_axpy_bytes());
    result.perf.cuda.axpy.record(update_s_time, cuda_axpy_bytes);
    result.total_flops += update_s_flops;
    result.cuda_reference_total_flops += cuda_axpy_flops;
    if (emit_iteration_diagnostics) {
      print_iteration_perf_diagnostics(
          ctx,
          iteration,
          perf_before_iteration,
          result.perf,
          iteration_interior_spmv_time,
          iteration_halo_spmv_time);
    }
    rho = rho_next;
  }

  const auto cuda_reference_solve_end = std::chrono::steady_clock::now();
  result.cuda_reference_solve_time_seconds =
      std::chrono::duration<double>(cuda_reference_solve_end - cuda_reference_solve_start).count();

  std::vector<double> x_local_host;
  interior_backend.download_vector(x, x_local_host);

  if (options.manufactured_solution) {
    double local_exact_norm_squared = 0.0;
    double local_error_norm_squared = 0.0;
    for (std::size_t i = 0; i < x_local_host.size(); ++i) {
      const double error = x_local_host[i] - x_exact_local_host[i];
      local_exact_norm_squared += x_exact_local_host[i] * x_exact_local_host[i];
      local_error_norm_squared += error * error;
    }
    const double exact_norm_squared = mpi_allreduce_sum(local_exact_norm_squared);
    const double error_norm_squared = mpi_allreduce_sum(local_error_norm_squared);
    const double exact_norm = std::sqrt(exact_norm_squared);
    result.relative_solution_error = exact_norm > 0.0 ? std::sqrt(error_norm_squared) / exact_norm : 0.0;
  }

  const double global_flops = mpi_allreduce_sum(result.total_flops);
  result.total_flops = global_flops;
  const double cuda_reference_global_flops = mpi_allreduce_sum(result.cuda_reference_total_flops);
  result.cuda_reference_total_flops = cuda_reference_global_flops;

  const auto end = std::chrono::steady_clock::now();
  result.solve_time_seconds = std::chrono::duration<double>(end - solve_start).count();
  result.total_time_seconds = result.solve_time_seconds;
  result.flop_rate_gflops = result.solve_time_seconds > 0.0 ? result.total_flops / result.solve_time_seconds / 1.0e9 : 0.0;
  result.cuda_reference_flop_rate_gflops = result.cuda_reference_solve_time_seconds > 0.0
                                             ? result.cuda_reference_total_flops / result.cuda_reference_solve_time_seconds / 1.0e9
                                             : 0.0;
  if (emit_diagnostics) {
    print_perf_diagnostics(result, ctx, interior_spmv_time, halo_spmv_time);
  }
  return result;
#endif
}

} // namespace acg::solver::algorithms
