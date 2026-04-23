#ifndef ACG_RUNTIME_MPI_RUNTIME_HPP
#define ACG_RUNTIME_MPI_RUNTIME_HPP

namespace acg::runtime {

struct MpiRuntimeInfo {
  int rank = 0;
  int size = 1;
  int local_rank = 0;
};

class ScopedMpiSession {
public:
  ScopedMpiSession(int &argc, char **&argv);
  ~ScopedMpiSession();

  ScopedMpiSession(const ScopedMpiSession &) = delete;
  ScopedMpiSession &operator=(const ScopedMpiSession &) = delete;

  [[nodiscard]] MpiRuntimeInfo info() const noexcept;

private:
  bool owns_mpi_ = false;
  MpiRuntimeInfo info_{};
};

} // namespace acg::runtime

#endif
