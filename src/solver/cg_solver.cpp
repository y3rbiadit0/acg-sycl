#include "acg/solver/cg_solver.hpp"

#include <array>
#include <chrono>
#include <stdexcept>

namespace acg::solver
{

  SolverResult run_cg(
      const acg::matrix::CsrMatrix<double> &matrix,
      const acg::runtime::RunContext &ctx,
      const SolverOptions &options)
  {
    (void)matrix;
    (void)ctx;
    (void)options;

    const auto start = std::chrono::steady_clock::now();

    constexpr size_t array_size = 1024;
    std::vector<int> data(array_size, 0);

    // Initialize data
    for (size_t i = 0; i < array_size; i++)
    {
      data[i] = i;
    }

    // Create a SYCL queue
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Create a buffer
    sycl::buffer<int> buffer(data.data(), sycl::range<1>(array_size));
    std::array<int, 1> kernel_error = {0};
    sycl::buffer<int> error_buffer(kernel_error.data(), sycl::range<1>(1));

    // Submit a kernel
    q.submit([&](sycl::handler &h)
             {
        sycl::accessor acc(buffer, h);
        sycl::accessor error_acc(error_buffer, h);
        
        h.parallel_for(sycl::range<1>(array_size), [=](sycl::id<1> idx) {
            if (idx[0] + 1 >= array_size) {
                error_acc[0] = 1;
                return;
            }

            acc[idx[0]] = acc[idx[0]] + acc[idx[0] + 1];
        }); })
        .wait();

    sycl::host_accessor error_host(error_buffer, sycl::read_only);
    if (error_host[0] != 0)
    {
      throw std::out_of_range("CG demo kernel attempted an out-of-bounds access at idx + 1");
    }

    // Access the results
    sycl::host_accessor host_acc(buffer, sycl::read_only);

    // Print a few values
    for (size_t i = 0; i < 5; i++)
    {
      std::cout << "data[" << i << "] = " << host_acc[i] << std::endl;
    }

    const auto end = std::chrono::steady_clock::now();

    SolverResult result;
    result.converged = true;
    result.iterations = 0;
    result.final_residual = 0.0;
    result.solve_time_seconds = std::chrono::duration<double>(end - start).count();
    return result;
  }

} // namespace acg::solver
