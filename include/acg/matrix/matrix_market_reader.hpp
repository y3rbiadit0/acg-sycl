#ifndef ACG_MATRIX_MATRIX_MARKET_READER_HPP
#define ACG_MATRIX_MATRIX_MARKET_READER_HPP

#include <string>

#include "acg/matrix/csr_matrix.hpp"

namespace acg::matrix {

CsrMatrix<double> read_matrix_market(const std::string &path);

} // namespace acg::matrix

#endif
