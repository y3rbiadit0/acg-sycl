#include "acg/runtime/queue_factory.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace acg::runtime {

namespace {

template <typename Selector>
sycl::queue make_queue_with_selector(Selector selector, bool enable_profiling) {
  if (enable_profiling) {
    return sycl::queue{selector, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{selector};
}

sycl::queue make_queue_for_device(const sycl::device &device, bool enable_profiling) {
  if (enable_profiling) {
    return sycl::queue{device, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{device};
}

} // namespace

sycl::queue make_queue(DeviceKind device_kind, bool enable_profiling, int device_ordinal) {
  switch (device_kind) {
  case DeviceKind::Default:
    return make_queue_with_selector(sycl::default_selector_v, enable_profiling);
  case DeviceKind::CPU:
    return make_queue_with_selector(sycl::cpu_selector_v, enable_profiling);
  case DeviceKind::GPU: {
    const std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
      throw std::runtime_error("no GPU devices are available");
    }
    const int normalized_ordinal = device_ordinal >= 0 ? device_ordinal % static_cast<int>(devices.size()) : 0;
    return make_queue_for_device(devices[static_cast<std::size_t>(normalized_ordinal)], enable_profiling);
  }
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
