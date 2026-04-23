#ifndef ACG_SOLVER_ALGORITHMS_CG_SINGLE_GPU_HPP
#define ACG_SOLVER_ALGORITHMS_CG_SINGLE_GPU_HPP

#include <vector>

#include "acg/runtime/run_context.hpp"
#include "acg/solver/backends/cg_backend.hpp"
#include "acg/solver/solver_options.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::solver::algorithms {

SolverResult run_cg_single_gpu(backends::SingleGpuCgBackend &backend,
                               const acg::runtime::RunContext &ctx,
                               const SolverOptions &options,
                               const std::vector<double> &x_exact_host,
                               const std::vector<double> &b_host);

} // namespace acg::solver::algorithms

#endif
