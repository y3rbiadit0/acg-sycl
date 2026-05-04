#include "acg/matrix/distributed_csr_matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#ifdef ACG_HAVE_METIS
#include <metis.h>
#endif

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

std::vector<std::int64_t> local_rows_for_rank(const std::vector<int> &owner_by_row, int rank) {
  std::vector<std::int64_t> rows;
  for (std::int64_t row = 0; row < static_cast<std::int64_t>(owner_by_row.size()); ++row) {
    if (owner_by_row[static_cast<std::size_t>(row)] == rank) {
      rows.push_back(row);
    }
  }
  return rows;
}

DistributedCsrMatrixPartition build_partition_from_owner(
    const CsrMatrix<double> &matrix,
    int rank,
    int size,
    const std::vector<int> &owner_by_row,
    std::string method,
    std::int64_t objective) {
  if (size <= 0) {
    throw std::runtime_error("partition size must be positive");
  }
  if (rank < 0 || rank >= size) {
    throw std::runtime_error("partition rank is out of range");
  }
  if (static_cast<std::int64_t>(owner_by_row.size()) != matrix.rows) {
    throw std::runtime_error("partition owner map size does not match matrix rows");
  }

  DistributedCsrMatrixPartition partition;
  partition.global_rows = matrix.rows;
  partition.global_cols = matrix.cols;
  partition.rank = rank;
  partition.size = size;
  partition.method = std::move(method);
  partition.objective = objective;
  partition.local_global_rows = local_rows_for_rank(owner_by_row, rank);
  if (!partition.local_global_rows.empty()) {
    partition.local_row_begin = partition.local_global_rows.front();
    partition.local_row_end = partition.local_global_rows.back() + 1;
  }

  std::vector<std::int64_t> global_to_local(static_cast<std::size_t>(matrix.rows), -1);
  for (std::int64_t local_row = 0; local_row < partition.local_rows(); ++local_row) {
    const std::int64_t global_row = partition.local_global_rows[static_cast<std::size_t>(local_row)];
    global_to_local[static_cast<std::size_t>(global_row)] = local_row;
  }

  std::vector<std::tuple<std::int64_t, std::int32_t, double>> local_entries;
  std::vector<std::tuple<std::int64_t, std::int32_t, double>> interior_entries;
  std::vector<std::tuple<std::int64_t, std::int32_t, double>> halo_entries;
  std::map<int, std::vector<std::int64_t>> import_columns_by_rank;
  std::vector<std::int64_t> ghost_global_columns;

  for (std::int64_t local_row = 0; local_row < partition.local_rows(); ++local_row) {
    const std::int64_t global_row = partition.local_global_rows[static_cast<std::size_t>(local_row)];
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      const double value = matrix.values[static_cast<std::size_t>(offset)];
      local_entries.emplace_back(local_row, static_cast<std::int32_t>(global_col), value);
      const int col_owner = owner_by_row[static_cast<std::size_t>(global_col)];
      if (col_owner == rank) {
        interior_entries.emplace_back(
            local_row,
            static_cast<std::int32_t>(global_to_local[static_cast<std::size_t>(global_col)]),
            value);
      }
      else {
        import_columns_by_rank[col_owner].push_back(global_col);
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

  for (std::int64_t local_row = 0; local_row < partition.local_rows(); ++local_row) {
    const std::int64_t global_row = partition.local_global_rows[static_cast<std::size_t>(local_row)];
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      if (owner_by_row[static_cast<std::size_t>(global_col)] == rank) {
        continue;
      }
      const double value = matrix.values[static_cast<std::size_t>(offset)];
      halo_entries.emplace_back(local_row, ghost_index_by_global_column.at(global_col), value);
    }
  }

  std::map<int, std::vector<std::int64_t>> export_columns_by_rank;
  for (std::int64_t global_row = 0; global_row < matrix.rows; ++global_row) {
    const int row_rank = owner_by_row[static_cast<std::size_t>(global_row)];
    if (row_rank == rank) {
      continue;
    }
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(global_row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(global_row + 1)];
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t global_col = matrix.col_idx[static_cast<std::size_t>(offset)];
      if (owner_by_row[static_cast<std::size_t>(global_col)] == rank) {
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
      export_peer.local_indices.push_back(global_to_local[static_cast<std::size_t>(global_col)]);
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

std::vector<int> build_row_block_owner(std::int64_t rows, int size) {
  std::vector<int> owner(static_cast<std::size_t>(rows), 0);
  for (std::int64_t row = 0; row < rows; ++row) {
    owner[static_cast<std::size_t>(row)] = owner_rank(row, rows, size);
  }
  return owner;
}

#ifdef ACG_HAVE_METIS
std::vector<int> build_metis_owner(const CsrMatrix<double> &matrix, int size, std::int64_t &objective) {
  if (matrix.rows > static_cast<std::int64_t>(std::numeric_limits<idx_t>::max())) {
    throw std::runtime_error("matrix has too many rows for this METIS idx_t build");
  }
  if (size > static_cast<int>(std::numeric_limits<idx_t>::max())) {
    throw std::runtime_error("partition count is too large for this METIS idx_t build");
  }

  std::vector<idx_t> xadj(static_cast<std::size_t>(matrix.rows + 1), 0);
  std::vector<idx_t> adjncy;
  for (std::int64_t row = 0; row < matrix.rows; ++row) {
    std::vector<idx_t> neighbors;
    const std::int64_t row_start = matrix.row_ptr[static_cast<std::size_t>(row)];
    const std::int64_t row_end = matrix.row_ptr[static_cast<std::size_t>(row + 1)];
    neighbors.reserve(static_cast<std::size_t>(row_end - row_start));
    for (std::int64_t offset = row_start; offset < row_end; ++offset) {
      const std::int64_t col = matrix.col_idx[static_cast<std::size_t>(offset)];
      if (col == row || col < 0 || col >= matrix.rows) {
        continue;
      }
      neighbors.push_back(static_cast<idx_t>(col));
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
    xadj[static_cast<std::size_t>(row + 1)] = static_cast<idx_t>(adjncy.size());
  }

  idx_t nvtxs = static_cast<idx_t>(matrix.rows);
  idx_t ncon = 1;
  idx_t nparts = static_cast<idx_t>(size);
  idx_t objval = 0;
  std::vector<idx_t> part(static_cast<std::size_t>(matrix.rows), 0);
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;

  const int status = METIS_PartGraphKway(
      &nvtxs,
      &ncon,
      xadj.data(),
      adjncy.data(),
      nullptr,
      nullptr,
      nullptr,
      &nparts,
      nullptr,
      nullptr,
      options,
      &objval,
      part.data());
  if (status != METIS_OK) {
    throw std::runtime_error("METIS_PartGraphKway failed");
  }

  objective = static_cast<std::int64_t>(objval);
  std::vector<int> owner(static_cast<std::size_t>(matrix.rows), 0);
  for (std::int64_t row = 0; row < matrix.rows; ++row) {
    owner[static_cast<std::size_t>(row)] = static_cast<int>(part[static_cast<std::size_t>(row)]);
  }
  return owner;
}
#endif

} // namespace

PartitionOptions partition_options_from_environment() {
  const char *value = std::getenv("ACG_PARTITIONER");
  if (value == nullptr || value[0] == '\0' || std::string(value) == "row-block") {
    return PartitionOptions{.method = PartitionMethod::RowBlock};
  }
  if (std::string(value) == "metis") {
    return PartitionOptions{.method = PartitionMethod::Metis};
  }
  throw std::runtime_error("unsupported ACG_PARTITIONER value: " + std::string(value));
}

DistributedCsrMatrixPartition build_distributed_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size,
    const PartitionOptions &options) {
  switch (options.method) {
  case PartitionMethod::RowBlock:
    return build_row_block_partition(matrix, rank, size);
  case PartitionMethod::Metis:
    return build_metis_partition(matrix, rank, size);
  }
  throw std::runtime_error("unsupported partition method");
}

DistributedCsrMatrixPartition build_row_block_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size) {
  DistributedCsrMatrixPartition partition = build_partition_from_owner(
      matrix,
      rank,
      size,
      build_row_block_owner(matrix.rows, size),
      "row-block",
      -1);
  const auto [local_begin, local_end] = block_bounds(matrix.rows, rank, size);
  partition.local_row_begin = local_begin;
  partition.local_row_end = local_end;
  return partition;
}

DistributedCsrMatrixPartition build_metis_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size) {
#ifndef ACG_HAVE_METIS
  (void)matrix;
  (void)rank;
  (void)size;
  throw std::runtime_error("METIS partitioner requested but this build was not configured with ACG_ENABLE_METIS=ON");
#else
  std::int64_t objective = -1;
  std::vector<int> owner = build_metis_owner(matrix, size, objective);
  return build_partition_from_owner(matrix, rank, size, owner, "metis", objective);
#endif
}

} // namespace acg::matrix
