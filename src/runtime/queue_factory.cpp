#include "acg/runtime/queue_factory.hpp"

#include <stdexcept>

namespace acg::runtime {

namespace {

template <typename Selector>
sycl::queue make_queue_with_selector(Selector selector, bool enable_profiling) {
  if (enable_profiling) {
    return sycl::queue{selector, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{selector};
}

} // namespace

sycl::queue make_queue(DeviceKind device_kind, bool enable_profiling) {
  switch (device_kind) {
  case DeviceKind::Default:
    return make_queue_with_selector(sycl::default_selector_v, enable_profiling);
  case DeviceKind::CPU:
    return make_queue_with_selector(sycl::cpu_selector_v, enable_profiling);
  case DeviceKind::GPU:
    return make_queue_with_selector(sycl::gpu_selector_v, enable_profiling);
  }

  throw std::runtime_error("unsupported device kind");
}

const char *to_string(DeviceKind device_kind) {
  switch (device_kind) {
  case DeviceKind::Default:
    return "default";
  case DeviceKind::CPU:
    return "cpu";
  case DeviceKind::GPU:
    return "gpu";
  }
  return "unknown";
}

} // namespace acg::runtime
