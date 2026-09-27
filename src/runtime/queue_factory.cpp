#include "acg/runtime/queue_factory.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace acg::runtime {

namespace {

// Every queue is in-order. oneCCL's NCCL backend pulls a CUDA stream out of the
// SYCL queue and rejects an out-of-order one outright:
//
//   nccl_comm.cpp:369 get_cuda_stream: condition
//   sycl_queue.has_property<sycl::property::queue::in_order>() failed
//
// so --solver-collectives oneccl cannot run without it. It is unconditional
// rather than tied to that flag because an mpi run and a oneccl run have to
// differ in the collective and nothing else; making the queue semantics differ
// too would leave the comparison measuring both at once.
//
// The cost is close to nil. Every solver submission is waited on where it is
// issued, apart from the per-neighbour pack kernels in submit_pack_kernels(),
// which an in-order queue serializes rather than overlaps.
sycl::property_list queue_properties(bool enable_profiling) {
  if (enable_profiling) {
    return {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}};
  }
  return {sycl::property::queue::in_order{}};
}

template <typename Selector>
sycl::queue make_queue_with_selector(Selector selector, bool enable_profiling) {
  return sycl::queue{selector, queue_properties(enable_profiling)};
}

sycl::queue make_queue_for_device(const sycl::device &device, bool enable_profiling) {
  return sycl::queue{device, queue_properties(enable_profiling)};
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
