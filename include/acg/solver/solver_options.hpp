#ifndef ACG_SOLVER_SOLVER_OPTIONS_HPP
#define ACG_SOLVER_SOLVER_OPTIONS_HPP

#include <cstdint>
#include <string>

namespace acg::solver {

// How the two scalar allreduces of every CG iteration are done. The halo
// exchange is always CUDA-aware MPI.
enum class SolverCollectiveMode {
  Mpi,
  OneCcl,
};

inline const char *to_string(SolverCollectiveMode mode) {
  return mode == SolverCollectiveMode::OneCcl ? "oneccl" : "mpi";
}

struct SolverOptions {
  int max_iterations = 1000;
  // Untimed CG iterations run (and then discarded) before the timed solve, so
  // first-use costs -- SpMV analysis, UCX registration of device buffers, NCCL
  // channel setup -- stay out of the measurement. Same meaning and default as
  // native aCG's --warmup.
  int warmup = 10;
  int log_every = 0;
  double residual_absolute_tolerance = 0.0;
  double residual_relative_tolerance = 0.0;
  SolverCollectiveMode solver_collectives = SolverCollectiveMode::Mpi;
  bool manufactured_solution = false;
  std::uint32_t seed = 0;
  // Empty means partition here, by ACG_PARTITIONER. Set by --partition to the
  // Matrix Market row-partition vector every rank should read instead.
  std::string partition_path;
};

} // namespace acg::solver

#endif
