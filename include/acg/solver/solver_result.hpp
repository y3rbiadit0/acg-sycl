#ifndef ACG_SOLVER_SOLVER_RESULT_HPP
#define ACG_SOLVER_SOLVER_RESULT_HPP

#include <vector>

namespace acg::solver {

// Host time spent blocked inside the timed solve, per rank. Everything else in
// the loop is asynchronous submission, so these are the only intervals where
// the CPU waits and the only per-phase times worth reporting.
struct WaitTimes {
  double pack_s = 0.0;      // packing kernel, before MPI may read the send buffer
  double halo_s = 0.0;      // MPI_Waitall of the halo exchange
  double allreduce_s = 0.0; // producer kernel + scalar allreduce
  double readback_s = 0.0;  // copy of gamma and rho to the host, every iteration
};

struct SolverResult {
  // Timing schema 3, aligned with native aCG's "total solver time": the timer
  // starts after warmup, an idle device and an MPI barrier, covers the initial
  // residual and the CG loop, and stops once the device has drained.
  static constexpr int kTimingSchema = 3;

  bool converged = false;
  int iterations = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double rhs_norm = 0.0;
  double relative_residual = 0.0; // final / initial (x0 = 0, so also / ||b||)
  double relative_solution_error = 0.0;
  double total_flops = 0.0;       // all ranks
  double gflops = 0.0;            // total_flops / solver_max_s

  double setup_s = 0.0;       // uploads, SpMV analysis, communicator setup
  double warmup_s = 0.0;
  double solver_s = 0.0;      // this rank
  double solver_max_s = 0.0;  // slowest rank: the number to compare
  double solver_min_s = 0.0;
  double post_solve_s = 0.0;  // download, error norm, summaries
  double validation_s = 0.0;
  WaitTimes waits;            // this rank

  // ||b - A x|| recomputed on the host from the original matrix after the
  // solve, independently of every device code path.
  double true_residual = 0.0;
  double true_relative_residual = 0.0;

  int spmv_index_bits = 64;
  // This rank's rows of x, in partition order.
  std::vector<double> solution_local;
};

} // namespace acg::solver

#endif
