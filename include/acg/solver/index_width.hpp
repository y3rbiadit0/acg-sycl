#ifndef ACG_SOLVER_INDEX_WIDTH_HPP
#define ACG_SOLVER_INDEX_WIDTH_HPP

#include <cstdint>
#include <limits>

namespace acg::solver {

// Whether a CSR matrix of this shape fits 32-bit device row pointers and column
// indices: row pointers run up to nnz, column indices up to cols - 1. The
// device copies use 32 bits whenever this holds (as native aCG does by
// default) and 64 bits otherwise. Shape-only, so it is testable without
// allocating a matrix near the limit.
constexpr bool csr_fits_index32(std::int64_t rows, std::int64_t cols, std::int64_t nnz) {
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  return rows >= 0 && cols >= 0 && nnz >= 0 && rows <= limit && cols <= limit && nnz <= limit;
}

} // namespace acg::solver

#endif
