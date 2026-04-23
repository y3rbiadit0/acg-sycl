#include "acg/solver/algorithms/cg_multi_gpu_mpi.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <sycl/sycl.hpp>

#include "acg/solver/backends/onemath_cuda_backend.hpp"
#include "acg/solver/stopping_criteria.hpp"

namespace acg::solver::algorithms {

namespace {

double timed_call(const auto &fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct DeviceBufferDeleter {
  sycl::queue *queue = nullptr;

  void operator()(double *ptr) const {
    if (ptr != nullptr && queue != nullptr) {
      sycl::free(ptr, *queue);
    }
  }
};

using DeviceBuffer = std::unique_ptr<double, DeviceBufferDeleter>;

DeviceBuffer make_device_buffer(sycl::queue &queue, std::int64_t count) {
  if (count == 0) {
    return DeviceBuffer(nullptr, DeviceBufferDeleter{.queue = &queue});
  }
  double *ptr = sycl::malloc_device<double>(static_cast<std::size_t>(count), queue);
  if (ptr == nullptr) {
    throw std::runtime_error("failed to allocate device communication buffer");
  }
  return DeviceBuffer(ptr, DeviceBufferDeleter{.queue = &queue});
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

double host_squared_norm(const std::vector<double> &values) {
  double sum = 0.0;
  for (const double value : values) {
    sum += value * value;
  }
  return sum;
}

struct HaloSendState {
  int rank = -1;
  std::vector<std::int64_t> local_indices;
  DeviceBuffer buffer;
  std::vector<double> host_buffer;
};

struct HaloRecvState {
  int rank = -1;
  std::int64_t ghost_offset = 0;
  std::vector<double> host_buffer;
};

double compute_relative_solution_error(
    backends::SingleGpuCgBackend &backend,
    const backends::DeviceVector &x,
    const backends::DeviceVector &x_exact,
    backends::DeviceVector &x_error,
    double x_exact_norm,
    SolverResult &result) {
  result.perf.copy.record(timed_call([&] { backend.copy(x, x_error); }), backend.estimated_copy_bytes());
  result.perf.axpy.record(timed_call([&] { backend.axpy(-1.0, x_exact, x_error); }), backend.estimated_axpy_bytes());
  double local_error_norm_squared = 0.0;
  result.perf.nrm2.record(
      timed_call([&] { local_error_norm_squared = backend.dot(x_error, x_error); }),
      backend.estimated_nrm2_bytes());
  const double error_norm_squared = mpi_allreduce_sum(local_error_norm_squared);
  return x_exact_norm > 0.0 ? std::sqrt(error_norm_squared) / x_exact_norm : 0.0;
}

void maybe_log_progress(
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options,
    int iteration,
    const SolverResult &result) {
  if (ctx.rank != 0 || options.log_every <= 0 || iteration % options.log_every != 0) {
    return;
  }
  std::cout << "iter=" << iteration
            << " residual=" << result.final_residual
            << " rel_residual_r0=" << result.relative_residual_to_initial
            << " rel_residual_rhs=" << result.relative_residual_to_rhs;
  if (options.manufactured_solution) {
    std::cout << " relative_error=" << result.relative_solution_error;
  }
  std::cout << '\n';
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
  std::unique_ptr<backends::OnemathCudaBackend> halo_backend;
  if (partition.halo_matrix.nnz() > 0) {
    halo_backend = std::make_unique<backends::OnemathCudaBackend>(queue, partition.halo_matrix);
  }

  const double interior_spmv_flops = 2.0 * static_cast<double>(partition.interior_matrix.nnz());
  const double halo_spmv_flops = 2.0 * static_cast<double>(partition.halo_matrix.nnz());
  const double dot_flops = 2.0 * static_cast<double>(partition.local_rows());
  const double update_x_r_flops = 4.0 * static_cast<double>(partition.local_rows());
  const double update_s_flops = 3.0 * static_cast<double>(partition.local_rows());

  SolverResult result;
  backends::DeviceVector x = interior_backend.create_vector();
  backends::DeviceVector r = interior_backend.create_vector();
  backends::DeviceVector s = interior_backend.create_vector();
  backends::DeviceVector t = interior_backend.create_vector();
  backends::DeviceVector b = interior_backend.create_vector_from_host(b_local_host);
  backends::DeviceVector remote = interior_backend.create_vector();
  backends::DeviceVector shat = interior_backend.create_vector(static_cast<std::int64_t>(partition.ghost_global_columns.size()));
  std::optional<backends::DeviceVector> x_exact;
  std::optional<backends::DeviceVector> x_error;

  std::vector<HaloSendState> send_states;
  send_states.reserve(partition.exports.size());
  for (const auto &export_peer : partition.exports) {
    send_states.push_back(HaloSendState{
        .rank = export_peer.rank,
        .local_indices = export_peer.local_indices,
        .buffer = make_device_buffer(queue, static_cast<std::int64_t>(export_peer.local_indices.size())),
        .host_buffer = std::vector<double>(export_peer.local_indices.size()),
    });
  }

  std::vector<HaloRecvState> recv_states;
  recv_states.reserve(partition.imports.size());
  for (const auto &import_peer : partition.imports) {
    recv_states.push_back(HaloRecvState{
        .rank = import_peer.rank,
        .ghost_offset = import_peer.ghost_offset,
        .host_buffer = std::vector<double>(import_peer.global_columns.size()),
    });
  }

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
  result.perf.copy.record(0.0, 2 * interior_backend.estimated_copy_bytes());

  const double rhs_norm = std::sqrt(mpi_allreduce_sum(host_squared_norm(b_local_host)));

  double rho_local = 0.0;
  result.perf.nrm2.record(timed_call([&] { rho_local = interior_backend.dot(r, r); }), interior_backend.estimated_nrm2_bytes());
  result.total_flops += dot_flops;

  double rho = 0.0;
  result.perf.allreduce.record(timed_call([&] { rho = mpi_allreduce_sum(rho_local); }), sizeof(double) * partition.local_rows());
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
    std::vector<MPI_Request> requests;
    requests.reserve(partition.imports.size() + send_states.size());

    if (!partition.ghost_global_columns.empty()) {
      const double pack_time = timed_call([&] {
        for (const auto &send_state : send_states) {
          const std::int64_t count = static_cast<std::int64_t>(send_state.local_indices.size());
          if (count == 0) {
            continue;
          }
          const std::int64_t *indices = send_state.local_indices.data();
          double *dst = send_state.buffer.get();
          const double *src = s.data;
          queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(count)), [=](sycl::id<1> idx) {
              const std::size_t i = idx[0];
              dst[i] = src[indices[i]];
            });
          });
        }
        queue.wait();
        for (auto &send_state : send_states) {
          if (send_state.host_buffer.empty()) {
            continue;
          }
          queue.memcpy(
              send_state.host_buffer.data(),
              send_state.buffer.get(),
              send_state.host_buffer.size() * sizeof(double))
              .wait();
        }
      });

      std::int64_t packed_elements = 0;
      for (const auto &send_state : send_states) {
        packed_elements += static_cast<std::int64_t>(send_state.local_indices.size());
      }
      result.perf.pack.record(pack_time, 2 * sizeof(double) * packed_elements);
      result.perf.host_sync.record(pack_time, 0);

      for (auto &recv_state : recv_states) {
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
        requests.push_back(request);
      }

      for (const auto &send_state : send_states) {
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
        requests.push_back(request);
      }
    }

    result.perf.spmv.record(timed_call([&] { interior_backend.spmv(s, t); }), interior_backend.estimated_spmv_bytes());
    result.total_flops += interior_spmv_flops;

    if (!requests.empty()) {
      std::int64_t exchanged_elements = 0;
      for (const auto &import_peer : partition.imports) {
        exchanged_elements += static_cast<std::int64_t>(import_peer.global_columns.size());
      }
      for (const auto &send_state : send_states) {
        exchanged_elements += static_cast<std::int64_t>(send_state.local_indices.size());
      }
      result.perf.p2p.record(timed_call([&] {
        if (MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
          throw std::runtime_error("MPI_Waitall failed");
        }
      }), sizeof(double) * exchanged_elements);

      const double unpack_time = timed_call([&] {
        for (const auto &recv_state : recv_states) {
          if (recv_state.host_buffer.empty()) {
            continue;
          }
          queue.memcpy(
              shat.data + recv_state.ghost_offset,
              recv_state.host_buffer.data(),
              recv_state.host_buffer.size() * sizeof(double))
              .wait();
        }
      });
      result.perf.copy.record(unpack_time, sizeof(double) * exchanged_elements);
      result.perf.host_sync.record(unpack_time, 0);

      if (halo_backend != nullptr) {
        result.perf.spmv.record(timed_call([&] { halo_backend->spmv(shat, remote); }), halo_backend->estimated_spmv_bytes());
        result.perf.axpy.record(timed_call([&] { interior_backend.axpy(1.0, remote, t); }), interior_backend.estimated_axpy_bytes());
        result.total_flops += halo_spmv_flops + 2.0 * static_cast<double>(partition.local_rows());
      }
    }

    double gamma_local = 0.0;
    result.perf.dot.record(timed_call([&] { gamma_local = interior_backend.dot(s, t); }), interior_backend.estimated_dot_bytes());
    result.total_flops += dot_flops;
    double gamma = 0.0;
    result.perf.allreduce.record(timed_call([&] { gamma = mpi_allreduce_sum(gamma_local); }), sizeof(double) * partition.local_rows());
    if (gamma <= 0.0) {
      throw std::runtime_error("matrix is not positive definite under CG iteration");
    }

    const double alpha = rho / gamma;

    result.perf.axpy.record(timed_call([&] { interior_backend.axpy(-alpha, t, r); }), interior_backend.estimated_axpy_bytes());
    result.total_flops += update_x_r_flops;

    double rho_next_local = 0.0;
    result.perf.nrm2.record(
        timed_call([&] { rho_next_local = interior_backend.dot(r, r); }),
        interior_backend.estimated_nrm2_bytes());
    result.total_flops += dot_flops;

    double rho_next = 0.0;
    result.perf.allreduce.record(timed_call([&] { rho_next = mpi_allreduce_sum(rho_next_local); }), sizeof(double) * partition.local_rows());

    result.perf.axpy.record(timed_call([&] { interior_backend.axpy(alpha, s, x); }), interior_backend.estimated_axpy_bytes());

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
      break;
    }

    const double beta = rho_next / rho;
    result.perf.axpy.record(timed_call([&] { interior_backend.scal(beta, s); }), interior_backend.estimated_scal_bytes());
    result.perf.axpy.record(timed_call([&] { interior_backend.axpy(1.0, r, s); }), interior_backend.estimated_axpy_bytes());
    result.total_flops += update_s_flops;
    rho = rho_next;
  }

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

  const auto end = std::chrono::steady_clock::now();
  result.solve_time_seconds = std::chrono::duration<double>(end - solve_start).count();
  result.total_time_seconds = result.solve_time_seconds;
  result.flop_rate_gflops = result.solve_time_seconds > 0.0 ? result.total_flops / result.solve_time_seconds / 1.0e9 : 0.0;
  return result;
#endif
}

} // namespace acg::solver::algorithms
