#include "collectives.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <oneapi/math.hpp>

#ifdef ACG_HAVE_ONECCL
#include <oneapi/ccl.hpp>
#endif

#include "acg/solver/cg_solver.hpp"

namespace acg::solver {

#ifdef ACG_HAVE_ONECCL
// oneCCL over NCCL, bootstrapped by a KVS address broadcast over MPI. It takes
// its CUDA stream from the (in-order) SYCL queue.
struct Collectives::OneCcl {
  OneCcl(sycl::queue &queue, int rank, int size) {
    ccl::init();
    ccl::kvs::address_type address;
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      address = kvs->get_address();
    }
    MPI_Bcast(address.data(), static_cast<int>(address.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      kvs = ccl::create_kvs(address);
    }
    device.emplace(ccl::create_device(queue.get_device()));
    context.emplace(ccl::create_context(queue.get_context()));
    communicator.emplace(ccl::create_communicator(size, rank, *device, *context, kvs));
    stream.emplace(ccl::create_stream(queue));
  }

  ccl::shared_ptr_class<ccl::kvs> kvs;
  std::optional<ccl::device> device;
  std::optional<ccl::context> context;
  std::optional<ccl::communicator> communicator;
  std::optional<ccl::stream> stream;
};
#else
struct Collectives::OneCcl {};
#endif

Collectives::Collectives(sycl::queue &queue, SolverCollectiveMode mode, int rank, int size)
    : queue_(&queue), mode_(mode), size_(size) {
  if (mode_ != SolverCollectiveMode::OneCcl || size_ <= 1) {
    return;
  }
#ifdef ACG_HAVE_ONECCL
  oneccl_ = std::make_unique<OneCcl>(queue, rank, size);
#else
  (void)rank;
  throw std::runtime_error("--solver-collectives oneccl requires a build with ACG_ENABLE_ONECCL=ON");
#endif
}

Collectives::~Collectives() = default;

void Collectives::allreduce_sum(double *device_value) {
  if (size_ <= 1) {
    return;
  }
  if (mode_ == SolverCollectiveMode::Mpi) {
    if (MPI_Allreduce(MPI_IN_PLACE, device_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Allreduce failed");
    }
    return;
  }
#ifdef ACG_HAVE_ONECCL
  // Waiting on the oneCCL event and then the queue is the conservative choice:
  // how the NCCL adapter orders later SYCL work after the collective has not
  // been audited, so the result is made visible to the host before anything
  // else is queued. This is the one host wait native NCCL does not have.
  ccl::allreduce(device_value, device_value, 1, ccl::datatype::float64, ccl::reduction::sum, *oneccl_->communicator,
                 *oneccl_->stream)
      .wait();
  queue_->wait_and_throw();
#endif
}

namespace {

struct Percentiles {
  double min_us = 0.0, p50_us = 0.0, p99_us = 0.0, max_us = 0.0;
};

Percentiles percentiles(std::vector<double> samples) {
  Percentiles p;
  if (samples.empty()) {
    return p;
  }
  std::sort(samples.begin(), samples.end());
  const auto at = [&](double q) { return samples[static_cast<std::size_t>(q * static_cast<double>(samples.size() - 1))]; };
  p.min_us = samples.front();
  p.p50_us = at(0.50);
  p.p99_us = at(0.99);
  p.max_us = samples.back();
  return p;
}

} // namespace

void run_collective_probe(const acg::runtime::RunContext &ctx, const SolverOptions &options, long iterations) {
  if (ctx.size <= 1 || iterations <= 0) {
    return;
  }
  sycl::queue queue = ctx.queue;
  Collectives collectives(queue, options.solver_collectives, ctx.rank, ctx.size);
  const char *cap_env = std::getenv("ACG_COLLECTIVE_PROBE_MAX_S");
  const double cap_s = cap_env != nullptr && *cap_env != '\0' ? std::atof(cap_env) : 60.0;

  // 4M doubles: the dot and the "work" kernel move SpMV-scale traffic, so the
  // last phase sees the collective right after a busy device, as the solver does.
  constexpr std::size_t n = std::size_t{1} << 22;
  double *vec = sycl::malloc_device<double>(n, queue);
  double *scalar = sycl::malloc_device<double>(1, queue);
  double *host = sycl::malloc_host<double>(1, queue);
  if (vec == nullptr || scalar == nullptr || host == nullptr) {
    throw std::runtime_error("failed to allocate collective probe buffers");
  }
  queue.fill(vec, 1.0, n).wait_and_throw();
  const double rank_value = static_cast<double>(ctx.rank + 1);
  const double rank_sum = 0.5 * static_cast<double>(ctx.size) * static_cast<double>(ctx.size + 1);
  const double dot_sum = static_cast<double>(ctx.size) * static_cast<double>(n);
  const auto read_scalar = [&] {
    queue.memcpy(host, scalar, sizeof(double)).wait_and_throw();
    return *host;
  };
  const auto elapsed_us = [](std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  };

  if (ctx.rank == 0) {
    std::cout << "collective_probe: iterations=" << iterations << " ranks=" << ctx.size
              << " solver-collectives=" << to_string(options.solver_collectives) << " phase_cap_s=" << cap_s << '\n';
  }

  // Each step returns its latency in us, or a negative number if the reduced
  // value was wrong. A phase stops at the time cap, collectively, so a slow
  // mode shows up as fewer samples instead of eating the allocation.
  const auto phase = [&](const char *name, const auto &step) {
    for (int i = 0; i < 20; ++i) {
      step();
    }
    std::vector<double> samples;
    bool ok = true;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (long i = 0; i < iterations; ++i) {
      const double us = step();
      ok = ok && us >= 0.0;
      samples.push_back(us < 0.0 ? -us : us);
      int stop = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > cap_s ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (stop != 0) {
        break;
      }
    }
    const Percentiles p = percentiles(samples);
    const double local[6] = {p.min_us, p.p50_us, p.p99_us, p.max_us, static_cast<double>(samples.size()), ok ? 1.0 : 0.0};
    std::vector<double> all(ctx.rank == 0 ? static_cast<std::size_t>(ctx.size) * 6 : 0);
    MPI_Gather(local, 6, MPI_DOUBLE, all.data(), 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (ctx.rank != 0) {
      return;
    }
    double p50_max = 0.0;
    double p50_min = all[1];
    bool all_ok = true;
    for (int r = 0; r < ctx.size; ++r) {
      const double *v = all.data() + static_cast<std::size_t>(r) * 6;
      std::cout << "collective_probe: phase=" << name << " rank=" << r << " samples=" << static_cast<long>(v[4])
                << " min_us=" << v[0] << " p50_us=" << v[1] << " p99_us=" << v[2] << " max_us=" << v[3]
                << " ok=" << (v[5] > 0.5 ? 1 : 0) << '\n';
      p50_max = std::max(p50_max, v[1]);
      p50_min = std::min(p50_min, v[1]);
      all_ok = all_ok && v[5] > 0.5;
    }
    std::cout << "collective_probe_summary: phase=" << name << " p50_max_us=" << p50_max
              << " p50_spread_us=" << (p50_max - p50_min) << " ok=" << (all_ok ? 1 : 0) << '\n';
  };

  // (a) the collective alone, on a device scalar already in place.
  phase("device_allreduce", [&] {
    queue.memcpy(scalar, &rank_value, sizeof(double)).wait_and_throw();
    const auto start = std::chrono::steady_clock::now();
    collectives.allreduce_sum(scalar);
    const double us = elapsed_us(start);
    return read_scalar() == rank_sum ? us : -us;
  });

  // (b) plus the readback of the result.
  phase("device_allreduce_readback", [&] {
    queue.memcpy(scalar, &rank_value, sizeof(double)).wait_and_throw();
    const auto start = std::chrono::steady_clock::now();
    collectives.allreduce_sum(scalar);
    const bool ok = read_scalar() == rank_sum;
    const double us = elapsed_us(start);
    return ok ? us : -us;
  });

  // (c) dot -> allreduce -> dependent update -> readback: one CG reduction.
  // vec stays all ones, so the dot is exactly n per rank. (d) is the same after
  // a burst of memory-bound device work.
  const auto dot_step = [&](bool with_work) {
    if (with_work) {
      for (int k = 0; k < 4; ++k) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) { vec[idx[0]] = vec[idx[0]] * 1.0; });
      }
    }
    const auto start = std::chrono::steady_clock::now();
    oneapi::math::blas::column_major::dot(queue, static_cast<std::int64_t>(n), vec, 1, vec, 1, scalar);
    queue.wait_and_throw();
    collectives.allreduce_sum(scalar);
    const double expected = dot_sum;
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) { vec[idx[0]] = vec[idx[0]] * (*scalar / expected); });
    const bool ok = read_scalar() == dot_sum;
    const double us = elapsed_us(start);
    return ok ? us : -us;
  };
  phase("dot_allreduce_update", [&] { return dot_step(false); });
  phase("work_dot_allreduce_update", [&] { return dot_step(true); });

  queue.wait();
  sycl::free(host, queue);
  sycl::free(scalar, queue);
  sycl::free(vec, queue);
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace acg::solver
