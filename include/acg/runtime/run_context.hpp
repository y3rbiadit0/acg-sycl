#ifndef ACG_RUNTIME_RUN_CONTEXT_HPP
#define ACG_RUNTIME_RUN_CONTEXT_HPP

#include <sycl/sycl.hpp>

namespace acg::runtime {

struct RunContext {
  sycl::queue queue;
  int rank = 0;
  int size = 1;
  int local_rank = 0;
  bool profiling_enabled = false;
};

} // namespace acg::runtime

#endif
