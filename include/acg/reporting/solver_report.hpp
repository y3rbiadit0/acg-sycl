#ifndef ACG_REPORTING_SOLVER_REPORT_HPP
#define ACG_REPORTING_SOLVER_REPORT_HPP

#include <iosfwd>

#include "acg/cli/app_config.hpp"
#include "acg/matrix/csr_matrix.hpp"
#include "acg/solver/solver_result.hpp"

namespace acg::reporting {

struct SolverReportOptions {
  bool log_native_perf = false;
};

SolverReportOptions solver_report_options_from_environment();

void print_solver_report(
    std::ostream &out,
    const acg::solver::SolverResult &result,
    const acg::matrix::CsrMatrix<double> &matrix,
    const acg::cli::AppConfig &config,
    const SolverReportOptions &options);

} // namespace acg::reporting

#endif
