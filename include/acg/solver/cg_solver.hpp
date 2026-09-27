#ifndef ACG_SOLVER_CG_SOLVER_HPP
#define ACG_SOLVER_CG_SOLVER_HPP

#include "acg/matrix/csr_matrix.hpp"
#include "acg/runtime/run_context.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver {

// Partitions the matrix over the ranks (one part on one rank), solves from
// x0 = 0, and checks the result on the host with ||b - A x|| / ||b||.
SolverResult run_cg(
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::runtime::RunContext &ctx,
    const SolverOptions &options);

// ACG_COLLECTIVE_PROBE: latency distributions of the solver's own device
// allreduce (MPI or oneCCL, per --solver-collectives), alone, with readback,
// and inside a dot -> allreduce -> update step. Diagnostic runs only.
void run_collective_probe(const acg::runtime::RunContext &ctx, const SolverOptions &options, long iterations);

} // namespace acg::solver

#endif
