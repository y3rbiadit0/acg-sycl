#include "acg/solver/rhs_builder.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace acg::solver {

namespace {

std::vector<double> generate_exact_solution(std::size_t size, const SolverOptions &options) {
  std::vector<double> x_exact(size);
  std::mt19937 generator(options.seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  double norm_squared = 0.0;

  for (double &value : x_exact) {
    value = distribution(generator);
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
