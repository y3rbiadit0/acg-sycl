#include "acg/matrix/matrix_market_reader.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace acg::matrix {

namespace {

struct CoordinateEntry {
  std::int64_t row = 0;
  std::int64_t col = 0;
  double value = 0.0;
};

struct MatrixMarketHeader {
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
};

void skip_spaces(const char *&cursor) {
  while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r') {
    ++cursor;
  }
}

std::string parse_token(const char *&cursor) {
  skip_spaces(cursor);
  const char *start = cursor;
  while (*cursor != '\0' && *cursor != ' ' && *cursor != '\t' &&
         *cursor != '\r' && *cursor != '\n') {
    ++cursor;
  }
  if (start == cursor) {
    throw std::runtime_error("invalid Matrix Market header line");
  }
  return std::string(start, static_cast<std::size_t>(cursor - start));
}

MatrixMarketHeader parse_header_line(const std::string &line) {
  const char *cursor = line.c_str();
  const std::string banner = parse_token(cursor);
  MatrixMarketHeader header;
  header.object = parse_token(cursor);
  header.format = parse_token(cursor);
  header.field = parse_token(cursor);
  header.symmetry = parse_token(cursor);
  skip_spaces(cursor);

  if (banner != "%%MatrixMarket") {
    throw std::runtime_error("invalid Matrix Market banner");
  }
  if (header.object != "matrix") {
    throw std::runtime_error("only Matrix Market object 'matrix' is supported");
  }
  if (header.format != "coordinate") {
    throw std::runtime_error("only Matrix Market format 'coordinate' is supported");
  }
  if (header.field != "real" && header.field != "integer") {
    throw std::runtime_error("only Matrix Market fields 'real' and 'integer' are supported");
  }
  if (header.symmetry != "general" && header.symmetry != "symmetric") {
    throw std::runtime_error("only Matrix Market symmetries 'general' and 'symmetric' are supported");
  }
  if (*cursor != '\0') {
    throw std::runtime_error("invalid Matrix Market header line");
  }

  return header;
}

bool is_comment_or_empty(const std::string &line) {
  return line.empty() || line[0] == '%';
}

void append_entry(std::vector<CoordinateEntry> &entries, std::int64_t row, std::int64_t col, double value) {
  entries.push_back(CoordinateEntry{row, col, value});
}

std::int64_t parse_int64(const char *&cursor) {
  skip_spaces(cursor);
  errno = 0;
  char *end = nullptr;
  const long long value = std::strtoll(cursor, &end, 10);
  if (end == cursor || errno == ERANGE) {
    throw std::runtime_error("invalid integer in Matrix Market file");
  }
  cursor = end;
  return static_cast<std::int64_t>(value);
}

double parse_double(const char *&cursor) {
  skip_spaces(cursor);
  errno = 0;
  char *end = nullptr;
  const double value = std::strtod(cursor, &end);
  if (end == cursor || errno == ERANGE) {
    throw std::runtime_error("invalid floating-point value in Matrix Market file");
  }
  cursor = end;
  return value;
}

void expect_line_end(const char *cursor, const char *context) {
  skip_spaces(cursor);
  if (*cursor != '\0') {
    throw std::runtime_error(context);
  }
}

} // namespace

CsrMatrix<double> read_matrix_market(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open matrix file: " + path);
  }

  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("matrix file is empty: " + path);
  }

  const MatrixMarketHeader header = parse_header_line(line);

  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t file_nnz = 0;
  bool read_size_line = false;

  while (std::getline(input, line)) {
    if (is_comment_or_empty(line)) {
      continue;
    }

    const char *cursor = line.c_str();
    rows = parse_int64(cursor);
    cols = parse_int64(cursor);
    file_nnz = parse_int64(cursor);
    expect_line_end(cursor, "invalid Matrix Market size line");
    if (rows < 0 || cols < 0 || file_nnz < 0) {
      throw std::runtime_error("invalid Matrix Market size line");
    }
    read_size_line = true;
    break;
  }

  if (!read_size_line) {
    throw std::runtime_error("missing Matrix Market size line");
  }

  if (cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("matrix has too many columns for int32 CSR indices");
  }

  std::vector<CoordinateEntry> entries;
  entries.reserve(static_cast<std::size_t>(header.symmetry == "symmetric" ? file_nnz * 2 : file_nnz));

  std::int64_t parsed_entries = 0;
  while (parsed_entries < file_nnz && std::getline(input, line)) {
    if (is_comment_or_empty(line)) {
      continue;
    }

    const char *cursor = line.c_str();
    std::int64_t row = parse_int64(cursor);
    std::int64_t col = parse_int64(cursor);
    const double value = parse_double(cursor);
    expect_line_end(cursor, "invalid Matrix Market entry line");
    if (row <= 0 || col <= 0 || row > rows || col > cols) {
      throw std::runtime_error("Matrix Market indices are out of range");
    }

    --row;
    --col;
    append_entry(entries, row, col, value);
    if (header.symmetry == "symmetric" && row != col) {
      append_entry(entries, col, row, value);
    }

    ++parsed_entries;
  }

  if (parsed_entries != file_nnz) {
    throw std::runtime_error("matrix file ended before all entries were read");
  }

  std::sort(entries.begin(), entries.end(), [](const CoordinateEntry &lhs, const CoordinateEntry &rhs) {
    return std::tie(lhs.row, lhs.col) < std::tie(rhs.row, rhs.col);
  });

  std::vector<CoordinateEntry> merged_entries;
  merged_entries.reserve(entries.size());
  for (const CoordinateEntry &entry : entries) {
    if (!merged_entries.empty() &&
        merged_entries.back().row == entry.row &&
        merged_entries.back().col == entry.col) {
      merged_entries.back().value += entry.value;
      continue;
    }
    merged_entries.push_back(entry);
  }

  CsrMatrix<double> matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
  matrix.col_idx.reserve(merged_entries.size());
  matrix.values.reserve(merged_entries.size());

  for (const CoordinateEntry &entry : merged_entries) {
    ++matrix.row_ptr[static_cast<std::size_t>(entry.row + 1)];
    matrix.col_idx.push_back(static_cast<std::int32_t>(entry.col));
    matrix.values.push_back(entry.value);
  }

  for (std::size_t i = 1; i < matrix.row_ptr.size(); ++i) {
    matrix.row_ptr[i] += matrix.row_ptr[i - 1];
  }

  return matrix;
}

} // namespace acg::matrix
