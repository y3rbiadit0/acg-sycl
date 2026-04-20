#include <iostream>
#include <vector>

#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue(sycl::gpu_selector_v);

  constexpr std::int64_t m = 2;
  constexpr std::int64_t n = 2;
  constexpr std::int64_t k = 2;
  constexpr std::int64_t lda = 2;
  constexpr std::int64_t ldb = 2;
  constexpr std::int64_t ldc = 2;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<float> a = {1.0f, 3.0f, 2.0f, 4.0f};
  std::vector<float> b = {5.0f, 7.0f, 6.0f, 8.0f};
  std::vector<float> c(4, 0.0f);

  float *dev_a = sycl::malloc_device<float>(a.size(), queue);
  float *dev_b = sycl::malloc_device<float>(b.size(), queue);
  float *dev_c = sycl::malloc_device<float>(c.size(), queue);
  if (dev_a == nullptr || dev_b == nullptr || dev_c == nullptr) {
    throw std::runtime_error("failed to allocate USM for GEMM test");
  }

  queue.memcpy(dev_a, a.data(), a.size() * sizeof(float)).wait();
  queue.memcpy(dev_b, b.data(), b.size() * sizeof(float)).wait();
  queue.memcpy(dev_c, c.data(), c.size() * sizeof(float)).wait();

  oneapi::math::blas::column_major::gemm(
      queue,
      oneapi::math::transpose::nontrans,
      oneapi::math::transpose::nontrans,
      m,
      n,
      k,
      alpha,
      dev_a,
      lda,
      dev_b,
      ldb,
      beta,
      dev_c,
      ldc)
      .wait();

  queue.memcpy(c.data(), dev_c, c.size() * sizeof(float)).wait();

  sycl::free(dev_c, queue);
  sycl::free(dev_b, queue);
  sycl::free(dev_a, queue);

  std::cout << "GEMM result:";
  for (float value : c) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  const std::vector<float> expected = {19.0f, 43.0f, 22.0f, 50.0f};
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (std::abs(c[i] - expected[i]) > 1e-4f) {
      std::cerr << "unexpected GEMM output at index " << i << ": expected " << expected[i] << ", got " << c[i]
                << '\n';
      return 1;
    }
  }

  std::cout << "GEMM smoke test passed\n";
  return 0;
}
