#ifndef ACG_CLI_PARSE_ARGS_HPP
#define ACG_CLI_PARSE_ARGS_HPP

#include <iosfwd>

#include "acg/cli/app_config.hpp"

namespace acg::cli {

AppConfig parse_args(int argc, char **argv);
void print_usage(std::ostream &out, const char *program_name);

} // namespace acg::cli

#endif
