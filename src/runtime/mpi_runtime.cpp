#include "acg/runtime/mpi_runtime.hpp"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <stdexcept>

namespace acg::runtime {

ScopedMpiSession::ScopedMpiSession(int &argc, char **&argv) {
#ifdef ACG_HAVE_MPI
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
#else
  (void)argc;
  (void)argv;
#endif
}

ScopedMpiSession::~ScopedMpiSession() {
#ifdef ACG_HAVE_MPI
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (owns_mpi_ && finalized == 0) {
    MPI_Finalize();
  }
#endif
}

MpiRuntimeInfo ScopedMpiSession::info() const noexcept { return info_; }

} // namespace acg::runtime
