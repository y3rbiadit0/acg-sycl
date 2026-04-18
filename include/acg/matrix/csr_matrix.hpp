#ifndef ACG_MATRIX_CSR_MATRIX_HPP
#define ACG_MATRIX_CSR_MATRIX_HPP

#include <cstdint>
#include <vector>

namespace acg::matrix {

template <typename T>
struct CsrMatrix {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::vector<std::int64_t> row_ptr;
  std::vector<std::int32_t> col_idx;
  std::vector<T> values;

  [[nodiscard]] std::int64_t nnz() const {
    return static_cast<std::int64_t>(values.size());
  }
};

} // namespace acg::matrix

#endif
