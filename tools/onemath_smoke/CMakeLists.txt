cmake_minimum_required(VERSION 3.20)

project(onemath_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(oneMath REQUIRED PATHS /opt/oneMath/install/lib/cmake/oneMath NO_DEFAULT_PATH)

add_executable(onemath_gemm gemm_usm.cpp)
target_link_libraries(onemath_gemm PRIVATE ONEMATH::onemath)

add_executable(onemath_spmv spmv_usm.cpp)
target_link_libraries(onemath_spmv PRIVATE ONEMATH::onemath)
