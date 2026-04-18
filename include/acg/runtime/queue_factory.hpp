#ifndef ACG_RUNTIME_QUEUE_FACTORY_HPP
#define ACG_RUNTIME_QUEUE_FACTORY_HPP

#include <sycl/sycl.hpp>

#include "acg/runtime/device_kind.hpp"

namespace acg::runtime {

sycl::queue make_queue(DeviceKind device_kind, bool enable_profiling);
const char *to_string(DeviceKind device_kind);

} // namespace acg::runtime

#endif
