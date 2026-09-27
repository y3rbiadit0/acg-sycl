#include "halo.hpp"

#include <chrono>
#include <stdexcept>

#include "acg/solver/index_width.hpp"

namespace acg::solver {

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <typename T>
T *upload(sycl::queue &queue, const std::vector<T> &host) {
  T *ptr = sycl::malloc_device<T>(host.empty() ? 1 : host.size(), queue);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device memory");
  }
  if (!host.empty()) {
    queue.memcpy(ptr, host.data(), host.size() * sizeof(T)).wait_and_throw();
  }
  return ptr;
}

std::vector<std::int64_t> flat_export_indices(const acg::matrix::DistributedCsrMatrixPartition &partition) {
  std::vector<std::int64_t> indices;
  for (const auto &peer : partition.exports) {
    indices.insert(indices.end(), peer.local_indices.begin(), peer.local_indices.end());
  }
  return indices;
}

std::vector<std::int64_t> rows_with_entries(const acg::matrix::CsrMatrix<double> &matrix) {
  std::vector<std::int64_t> rows;
  for (std::int64_t row = 0; row < matrix.rows; ++row) {
    if (matrix.row_ptr[static_cast<std::size_t>(row)] != matrix.row_ptr[static_cast<std::size_t>(row + 1)]) {
      rows.push_back(row);
    }
  }
  return rows;
}

template <typename Index>
void upload_rows(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix,
                 const std::vector<std::int64_t> &rows, void *&rows_out, void *&row_ptr_out) {
  rows_out = upload(queue, std::vector<Index>(rows.begin(), rows.end()));
  row_ptr_out = upload(queue, std::vector<Index>(matrix.row_ptr.begin(), matrix.row_ptr.end()));
}

template <typename Index>
sycl::event halo_kernel(sycl::queue &queue, std::size_t n, const void *rows_ptr, const void *row_ptr_ptr,
                        const std::int32_t *col_idx, const double *values, const double *x, double *y) {
  const auto *rows = static_cast<const Index *>(rows_ptr);
  const auto *row_ptr = static_cast<const Index *>(row_ptr_ptr);
  return queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
    const Index row = rows[idx[0]];
    double sum = 0.0;
    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
      sum += values[k] * x[col_idx[k]];
    }
    y[row] += sum;
  });
}

} // namespace

template <typename T>
DeviceArray<T>::DeviceArray(sycl::queue &queue, const std::vector<T> &host) : queue_(&queue), data_(upload(queue, host)) {}

template class DeviceArray<std::int64_t>;
template class DeviceArray<double>;

HaloExchange::HaloExchange(sycl::queue &queue, const acg::matrix::DistributedCsrMatrixPartition &partition)
    : queue_(&queue),
      send_indices_(queue, flat_export_indices(partition)),
      send_buffer_(queue, std::vector<double>(flat_export_indices(partition).size())) {
  for (const auto &peer : partition.exports) {
    const auto count = static_cast<std::int64_t>(peer.local_indices.size());
    if (count > 0) {
      sends_.push_back(Peer{.rank = peer.rank, .offset = send_count_, .count = count});
      send_count_ += count;
    }
  }
  for (const auto &peer : partition.imports) {
    const auto count = static_cast<std::int64_t>(peer.global_columns.size());
    if (count > 0) {
      recvs_.push_back(Peer{.rank = peer.rank, .offset = peer.ghost_offset, .count = count});
    }
  }
  requests_.reserve(sends_.size() + recvs_.size());
}

void HaloExchange::begin(const DeviceVector &s, double *ghosts, WaitTimes &waits) {
  // Waiting for the pack also drains everything queued before it (the queue is
  // in-order), including the previous iteration's halo SpMV, the last reader
  // of the ghost values the receives below overwrite. A rank that sends
  // nothing skips the wait; its previous halo SpMV finished before that
  // iteration's allreduce, which waits for the device.
  if (send_count_ > 0) {
    const auto start = std::chrono::steady_clock::now();
    const std::int64_t *indices = send_indices_.get();
    double *buffer = send_buffer_.get();
    const double *src = s.data;
    queue_->parallel_for(sycl::range<1>(static_cast<std::size_t>(send_count_)), [=](sycl::id<1> idx) {
      buffer[idx[0]] = src[indices[idx[0]]];
    }).wait_and_throw();
    waits.pack_s += seconds_since(start);
  }

  requests_.clear();
  for (const Peer &peer : recvs_) {
    MPI_Request request;
    if (MPI_Irecv(ghosts + peer.offset, static_cast<int>(peer.count), MPI_DOUBLE, peer.rank, 0, MPI_COMM_WORLD,
                  &request) != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Irecv failed");
    }
    requests_.push_back(request);
  }
  for (const Peer &peer : sends_) {
    MPI_Request request;
    if (MPI_Isend(send_buffer_.get() + peer.offset, static_cast<int>(peer.count), MPI_DOUBLE, peer.rank, 0,
                  MPI_COMM_WORLD, &request) != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Isend failed");
    }
    requests_.push_back(request);
  }
}

void HaloExchange::finish(WaitTimes &waits) {
  const auto start = std::chrono::steady_clock::now();
  if (MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Waitall failed");
  }
  waits.halo_s += seconds_since(start);
}

HaloSpmv::HaloSpmv(sycl::queue &queue, const acg::matrix::CsrMatrix<double> &matrix)
    : queue_(&queue), index32_(csr_fits_index32(matrix.rows, matrix.cols, matrix.nnz())) {
  const std::vector<std::int64_t> rows = rows_with_entries(matrix);
  active_rows_ = rows.size();
  if (index32_) {
    upload_rows<std::int32_t>(queue, matrix, rows, rows_, row_ptr_);
  }
  else {
    upload_rows<std::int64_t>(queue, matrix, rows, rows_, row_ptr_);
  }
  col_idx_ = upload(queue, matrix.col_idx);
  values_ = upload(queue, matrix.values);
}

HaloSpmv::~HaloSpmv() {
  sycl::free(values_, *queue_);
  sycl::free(col_idx_, *queue_);
  sycl::free(row_ptr_, *queue_);
  sycl::free(rows_, *queue_);
}

sycl::event HaloSpmv::add(const double *ghosts, DeviceVector &y) {
  if (active_rows_ == 0) {
    return {};
  }
  return index32_ ? halo_kernel<std::int32_t>(*queue_, active_rows_, rows_, row_ptr_, col_idx_, values_, ghosts, y.data)
                  : halo_kernel<std::int64_t>(*queue_, active_rows_, rows_, row_ptr_, col_idx_, values_, ghosts, y.data);
}

} // namespace acg::solver
