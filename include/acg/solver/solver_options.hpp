#ifndef ACG_SOLVER_SOLVER_OPTIONS_HPP
#define ACG_SOLVER_SOLVER_OPTIONS_HPP

#include <cstdint>

namespace acg::solver {

enum class MpiMode {
  Host,
  GpuAware,
};

inline const char *to_string(MpiMode mpi_mode) {
  switch (mpi_mode) {
  case MpiMode::Host:
    return "host";
  case MpiMode::GpuAware:
    return "gpu-aware";
  }
  return "unknown";
}

struct SolverOptions {
  int max_iterations = 1000;
  int log_every = 0;
  double diff_absolute_tolerance = 0.0;
  double diff_relative_tolerance = 0.0;
  double residual_absolute_tolerance = 0.0;
  double residual_relative_tolerance = 0.0;
  double solution_relative_tolerance = 0.0;
  MpiMode mpi_mode = MpiMode::Host;
  bool manufactured_solution = false;
  std::uint32_t seed = 0;
};

} // namespace acg::solver

#endif
