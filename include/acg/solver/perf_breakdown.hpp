#ifndef ACG_SOLVER_PERF_BREAKDOWN_HPP
#define ACG_SOLVER_PERF_BREAKDOWN_HPP

#include <cstdint>

namespace acg::solver {

struct OpStats {
  double time_seconds = 0.0;
  std::int64_t count = 0;
  std::int64_t bytes = 0;

  void record(double dt, std::int64_t b) noexcept {
    time_seconds += dt;
    ++count;
    bytes += b;
  }

  double bandwidth_gbs() const noexcept {
    return time_seconds > 0.0 ? static_cast<double>(bytes) / time_seconds / 1.0e9 : 0.0;
  }
};

struct PerfBreakdown {
  OpStats spmv;
  OpStats dot;
  OpStats nrm2;
  OpStats axpy;
  OpStats copy;
};

} // namespace acg::solver

#endif
