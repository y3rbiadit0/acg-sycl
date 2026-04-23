#include "acg/matrix/distributed_csr_matrix.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace acg::matrix {

namespace {

std::pair<std::int64_t, std::int64_t> block_bounds(std::int64_t global_size, int rank, int size) {
  const std::int64_t base = global_size / size;
  const std::int64_t remainder = global_size % size;
  const std::int64_t begin = rank * base + std::min<std::int64_t>(rank, remainder);
  const std::int64_t count = base + (rank < remainder ? 1 : 0);
  return {begin, begin + count};
}

int owner_rank(std::int64_t global_index, std::int64_t global_size, int size) {
  const std::int64_t base = global_size / size;
  const std::int64_t remainder = global_size % size;
  const std::int64_t split = (base + 1) * remainder;
  if (global_index < split) {
    return static_cast<int>(global_index / (base + 1));
  }
  if (base == 0) {
    return static_cast<int>(remainder - 1);
  }
  return static_cast<int>(remainder + (global_index - split) / base);
}

CsrMatrix<double> build_local_csr(
    std::int64_t rows,
    std::int64_t cols,
    const std::vector<std::tuple<std::int64_t, std::int32_t, double>> &entries) {
  CsrMatrix<double> matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
  matrix.col_idx.reserve(entries.size());
  matrix.values.reserve(entries.size());

  for (const auto &[row, col, value] : entries) {
    (void)col;
    (void)value;
    ++matrix.row_ptr[static_cast<std::size_t>(row + 1)];
  }
  for (std::size_t i = 1; i < matrix.row_ptr.size(); ++i) {
    matrix.row_ptr[i] += matrix.row_ptr[i - 1];
  }
  for (const auto &[row, col, value] : entries) {
    (void)row;
    matrix.col_idx.push_back(col);
    matrix.values.push_back(value);
  }
  return matrix;
}

} // namespace

DistributedCsrMatrixPartition build_row_block_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size) {
  if (size <= 0) {
    throw std::runtime_error("partition size must be positive");
  }
  if (rank < 0 || rank >= size) {
    throw std::runtime_error("partition rank is out of range");
  }

  DistributedCsrMatrixPartition partition;
  partition.global_rows = matrix.rows;
  partition.global_cols = matrix.cols;
  partition.rank = rank;
  partition.size = size;

  const auto [local_begin, local_end] = block_bounds(matrix.rows, rank, size);
  partition.local_row_begin = local_begin;
  partition.local_row_end = local_end;

  std::vector<std::tuple<std::int64_t, std::int32_t, double>> local_entries;
  std::vector<std::tuple<std::int64_t, std::int32_t, double>> interior_entries;
  std::vector<std::tuple<std::int64_t, std::int32_t, double>> halo_entries;
  std::map<int, std::vector<std::int64_t>> import_columns_by_rank;
  std::vector<std::int64_t> ghost_global_columns;

  for (std::int64_t global_row = local_begin; global_row < local_end; ++global_row) {
    const std::int64_t local_row = global_row - local_begin;
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      const double value = matrix.values[static_cast<std::size_t>(offset)];
      local_entries.emplace_back(local_row, static_cast<std::int32_t>(global_col), value);
      if (global_col >= local_begin && global_col < local_end) {
        interior_entries.emplace_back(local_row, static_cast<std::int32_t>(global_col - local_begin), value);
      } else {
        import_columns_by_rank[owner_rank(global_col, matrix.cols, size)].push_back(global_col);
      }
    }
  }

  std::map<std::int64_t, std::int32_t> ghost_index_by_global_column;
  for (auto &[remote_rank, columns] : import_columns_by_rank) {
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    const std::int64_t ghost_offset = static_cast<std::int64_t>(ghost_global_columns.size());
    partition.imports.push_back(HaloImport{.rank = remote_rank, .ghost_offset = ghost_offset, .global_columns = columns});
    for (std::size_t i = 0; i < columns.size(); ++i) {
      ghost_global_columns.push_back(columns[i]);
      ghost_index_by_global_column[columns[i]] = static_cast<std::int32_t>(ghost_offset + static_cast<std::int64_t>(i));
    }
  }
  partition.ghost_global_columns = ghost_global_columns;

  for (std::int64_t global_row = local_begin; global_row < local_end; ++global_row) {
    const std::int64_t local_row = global_row - local_begin;
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      if (global_col >= local_begin && global_col < local_end) {
        continue;
      }
      const double value = matrix.values[static_cast<std::size_t>(offset)];
      halo_entries.emplace_back(local_row, ghost_index_by_global_column.at(global_col), value);
    }
  }

  std::map<int, std::vector<std::int64_t>> export_columns_by_rank;
  for (std::int64_t global_row = 0; global_row < matrix.rows; ++global_row) {
    const int row_rank = owner_rank(global_row, matrix.rows, size);
    if (row_rank == rank) {
      continue;
    }
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      if (global_col >= local_begin && global_col < local_end) {
        export_columns_by_rank[row_rank].push_back(global_col);
      }
    }
  }

  for (auto &[remote_rank, columns] : export_columns_by_rank) {
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    HaloExport export_peer;
    export_peer.rank = remote_rank;
    export_peer.global_columns = columns;
    export_peer.local_indices.reserve(columns.size());
    for (const std::int64_t global_col : columns) {
      export_peer.local_indices.push_back(global_col - local_begin);
    }
    partition.exports.push_back(std::move(export_peer));
  }

  partition.local_matrix = build_local_csr(partition.local_rows(), matrix.cols, local_entries);
  partition.interior_matrix = build_local_csr(partition.local_rows(), partition.local_rows(), interior_entries);
  partition.halo_matrix = build_local_csr(
      partition.local_rows(),
      static_cast<std::int64_t>(partition.ghost_global_columns.size()),
      halo_entries);
  return partition;
}

} // namespace acg::matrix
