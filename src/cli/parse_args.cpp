#include "acg/cli/parse_args.hpp"

#include <ostream>
#include <stdexcept>
#include <string>

namespace acg::cli {

namespace {

acg::runtime::DeviceKind parse_device_kind(const std::string &value) {
  if (value == "default") {
    return acg::runtime::DeviceKind::Default;
  }
  if (value == "cpu") {
    return acg::runtime::DeviceKind::CPU;
  }
  if (value == "gpu") {
    return acg::runtime::DeviceKind::GPU;
  }
  throw std::runtime_error("invalid value for --device: " + value);
}

std::string require_value(int argc, char **argv, int &index, const char *option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + option);
  }
  ++index;
  return argv[index];
}

} // namespace

AppConfig parse_args(int argc, char **argv) {
  AppConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      config.show_help = true;
      return config;
    }

    if (arg == "--matrix") {
      config.matrix_path = require_value(argc, argv, i, "--matrix");
      continue;
    }

    if (arg == "--device") {
      config.device = parse_device_kind(require_value(argc, argv, i, "--device"));
      continue;
    }

    if (arg == "--profile") {
      config.enable_profiling = true;
      continue;
    }

    if (arg == "--tol") {
      config.solver.tolerance = std::stod(require_value(argc, argv, i, "--tol"));
      continue;
    }

    if (arg == "--max-iters") {
      config.solver.max_iterations = std::stoi(require_value(argc, argv, i, "--max-iters"));
      continue;
    }

    if (arg == "--manufactured-solution") {
      config.solver.manufactured_solution = true;
      continue;
    }

    if (arg == "--seed") {
      config.solver.seed = static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, i, "--seed")));
      continue;
    }

    throw std::runtime_error("unknown argument: " + arg);
  }

  if (config.matrix_path.empty()) {
    throw std::runtime_error("--matrix is required");
  }

  return config;
}

void print_usage(std::ostream &out, const char *program_name) {
  out << "Usage: " << program_name << " --matrix <path> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --matrix <path>             Matrix Market file to load\n"
      << "  --device <default|cpu|gpu>  SYCL device selector\n"
      << "  --profile                   Enable SYCL queue profiling\n"
      << "  --tol <value>               Solver tolerance\n"
      << "  --max-iters <n>             Maximum solver iterations\n"
      << "  --manufactured-solution     Build rhs from a known solution\n"
      << "  --seed <n>                  Seed for manufactured solution\n"
      << "  --help, -h                  Show this message\n";
}

} // namespace acg::cli
