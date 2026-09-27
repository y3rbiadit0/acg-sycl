#include "acg/runtime/mpi_runtime.hpp"

#include <mpi.h>

#include <exception>
#include <stdexcept>

namespace acg::runtime {

ScopedMpiSession::ScopedMpiSession(int &argc, char **&argv) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Init failed");
    }
    owns_mpi_ = true;
  }

  if (MPI_Comm_rank(MPI_COMM_WORLD, &info_.rank) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (MPI_Comm_size(MPI_COMM_WORLD, &info_.size) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }

  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_split_type failed");
  }
  if (MPI_Comm_rank(local_comm, &info_.local_rank) != MPI_SUCCESS) {
    MPI_Comm_free(&local_comm);
    throw std::runtime_error("MPI_Comm_rank for shared-memory communicator failed");
  }
  MPI_Comm_free(&local_comm);
}

ScopedMpiSession::~ScopedMpiSession() {
  if (std::uncaught_exceptions() > 0) {
    return;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (owns_mpi_ && finalized == 0) {
    MPI_Finalize();
  }
}

MpiRuntimeInfo ScopedMpiSession::info() const noexcept { return info_; }

} // namespace acg::runtime
