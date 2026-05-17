#include "acg/solver/rhs_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace acg::solver {

namespace {

std::vector<double> generate_exact_solution(std::size_t size, const SolverOptions &options) {
  std::vector<double> x_exact(size);
  double norm_squared = 0.0;

  std::srand(options.seed != 0 ? options.seed : 1);
  for (double &value : x_exact) {
    value = 2.0 * (static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX)) - 1.0;
    norm_squared += value * value;
  }

  const double norm = std::sqrt(norm_squared);
  if (norm == 0.0) {
    throw std::runtime_error("manufactured solution generation produced a zero vector");
  }

  for (double &value : x_exact) {
    value /= norm;
  }

  return x_exact;
}

} // namespace

RhsBuildResult build_rhs_inputs(std::size_t size, const SolverOptions &options) {
  RhsBuildResult result;
  result.b_host.assign(size, 0.0);

  if (options.manufactured_solution) {
    result.x_exact_host = generate_exact_solution(size, options);
  }
  else {
    std::fill(result.b_host.begin(), result.b_host.end(), 1.0);
  }

  return result;
}

} // namespace acg::solver
