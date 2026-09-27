#ifndef ACG_REPORTING_SOLVER_REPORT_HPP
#define ACG_REPORTING_SOLVER_REPORT_HPP

#include <iosfwd>

#include "acg/solver/solver_result.hpp"

namespace acg::reporting {

// The lines the analysis parses (tooling/data_analysis/.../parsers/sycl.py).
void print_solver_report(std::ostream &out, const acg::solver::SolverResult &result);

} // namespace acg::reporting

#endif
