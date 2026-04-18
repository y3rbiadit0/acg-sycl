#ifndef ACG_CLI_APP_CONFIG_HPP
#define ACG_CLI_APP_CONFIG_HPP

#include <string>

#include "acg/runtime/device_kind.hpp"
#include "acg/solver/solver_options.hpp"

namespace acg::cli {

struct AppConfig {
  std::string matrix_path;
  acg::runtime::DeviceKind device = acg::runtime::DeviceKind::Default;
  bool enable_profiling = false;
  bool show_help = false;
  acg::solver::SolverOptions solver;
};

} // namespace acg::cli

#endif
