#ifndef ACG_MATRIX_DISTRIBUTED_CSR_MATRIX_HPP
#define ACG_MATRIX_DISTRIBUTED_CSR_MATRIX_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "acg/matrix/csr_matrix.hpp"

namespace acg::matrix {

struct HaloImport {
  int rank = -1;
  std::int64_t ghost_offset = 0;
  std::vector<std::int64_t> global_columns;
};

struct HaloExport {
  int rank = -1;
  std::vector<std::int64_t> global_columns;
  std::vector<std::int64_t> local_indices;
};

struct DistributedCsrMatrixPartition {
  std::int64_t global_rows = 0;
  std::int64_t global_cols = 0;
  int rank = 0;
  int size = 1;
  std::int64_t local_row_begin = 0;
  std::int64_t local_row_end = 0;
  std::string method = "row-block";
  std::int64_t objective = -1;
  std::vector<std::int64_t> local_global_rows;
  CsrMatrix<double> local_matrix;
  CsrMatrix<double> interior_matrix;
  CsrMatrix<double> halo_matrix;
  std::vector<std::int64_t> ghost_global_columns;
  std::vector<HaloImport> imports;
  std::vector<HaloExport> exports;

  [[nodiscard]] std::int64_t local_rows() const noexcept { return static_cast<std::int64_t>(local_global_rows.size()); }
};

enum class PartitionMethod {
  RowBlock,
  Metis,
};

struct PartitionOptions {
  PartitionMethod method = PartitionMethod::RowBlock;
};

PartitionOptions partition_options_from_environment();

DistributedCsrMatrixPartition build_distributed_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size,
    const PartitionOptions &options);

DistributedCsrMatrixPartition build_row_block_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size);

DistributedCsrMatrixPartition build_metis_partition(
    const CsrMatrix<double> &matrix,
    int rank,
    int size);

} // namespace acg::matrix

#endif
