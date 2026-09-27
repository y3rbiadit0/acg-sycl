cmake_minimum_required(VERSION 3.20)

project(onemath_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(oneMath_ROOT "/opt/oneMath/install" CACHE PATH "oneMath install root")
find_package(oneMath REQUIRED PATHS ${oneMath_ROOT}/lib/cmake/oneMath NO_DEFAULT_PATH)

# Same flags as the main build (CMakeLists.txt): on Leonardo DPC++ needs
# --gcc-toolchain to find libstdc++, and -fsycl-targets=nvptx64-nvidia-cuda,
# or the kernels are built for SPIR-V only and cannot run on the A100s.
set(ACG_SYCL_TARGET "" CACHE STRING "SYCL device target")
set(ACG_EXTRA_COMPILE_FLAGS "" CACHE STRING "Extra compiler flags")
set(ACG_EXTRA_LINK_FLAGS "" CACHE STRING "Extra linker flags")
separate_arguments(ACG_EXTRA_COMPILE_FLAGS_LIST NATIVE_COMMAND "${ACG_EXTRA_COMPILE_FLAGS}")
separate_arguments(ACG_EXTRA_LINK_FLAGS_LIST NATIVE_COMMAND "${ACG_EXTRA_LINK_FLAGS}")
add_compile_options(-fsycl ${ACG_EXTRA_COMPILE_FLAGS_LIST})
add_link_options(-fsycl ${ACG_EXTRA_LINK_FLAGS_LIST})
if(ACG_SYCL_TARGET)
  add_compile_options(-fsycl-targets=${ACG_SYCL_TARGET})
  add_link_options(-fsycl-targets=${ACG_SYCL_TARGET})
endif()

add_executable(onemath_gemm gemm_usm.cpp)
target_link_libraries(onemath_gemm PRIVATE ONEMATH::onemath)

add_executable(onemath_spmv spmv_usm.cpp)
target_link_libraries(onemath_spmv PRIVATE ONEMATH::onemath)
# index_width.hpp, the solver's 32-bit guard, tested here on shapes alone.
target_include_directories(onemath_spmv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
