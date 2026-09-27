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

acg::solver::SolverCollectiveMode parse_solver_collective_mode(const std::string &value) {
  if (value == "mpi") {
    return acg::solver::SolverCollectiveMode::Mpi;
  }
  if (value == "oneccl") {
    return acg::solver::SolverCollectiveMode::OneCcl;
  }
  throw std::runtime_error("invalid value for --solver-collectives: " + value);
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
  bool saw_residual_rtol = false;

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

    if (arg == "--partition") {
      config.solver.partition_path = require_value(argc, argv, i, "--partition");
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

    if (arg == "--solver-collectives") {
      config.solver.solver_collectives =
          parse_solver_collective_mode(require_value(argc, argv, i, "--solver-collectives"));
      continue;
    }

    if (arg == "--log-every") {
      config.solver.log_every = std::stoi(require_value(argc, argv, i, "--log-every"));
      continue;
    }

    if (arg == "--residual-atol") {
      config.solver.residual_absolute_tolerance = std::stod(require_value(argc, argv, i, "--residual-atol"));
      continue;
    }

    if (arg == "--residual-rtol") {
      config.solver.residual_relative_tolerance = std::stod(require_value(argc, argv, i, "--residual-rtol"));
      saw_residual_rtol = true;
      continue;
    }

    if (arg == "--warmup") {
      config.solver.warmup = std::stoi(require_value(argc, argv, i, "--warmup"));
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
  if (!saw_residual_rtol) {
    throw std::runtime_error("--residual-rtol is required");
  }

  return config;
}

void print_usage(std::ostream &out, const char *program_name) {
  out << "Usage: " << program_name << " --matrix <path> --residual-rtol <value> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --matrix <path>             Matrix Market file to load\n"
      << "  --partition <path>          Matrix Market row-partition vector to use\n"
      << "  --device <default|cpu|gpu>  SYCL device selector\n"
      << "  --solver-collectives <mpi|oneccl>  How the CG scalar allreduces are done\n"
      << "  --profile                   Enable SYCL queue profiling\n"
      << "  --residual-atol <value>     Absolute tolerance for the residual norm\n"
      << "  --residual-rtol <value>     Relative tolerance for the residual norm (required)\n"
      << "  --max-iters <n>             Maximum solver iterations\n"
      << "  --warmup <n>                Untimed warmup iterations (default 10, as native aCG)\n"
      << "  --log-every <n>             Print the residual every n iterations\n"
      << "  --manufactured-solution     Build b = A x_exact from a known solution\n"
      << "  --seed <n>                  Seed for the manufactured solution\n"
      << "  --help, -h                  Show this message\n";
}

} // namespace acg::cli
