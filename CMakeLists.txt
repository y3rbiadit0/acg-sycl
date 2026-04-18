cmake_minimum_required(VERSION 3.20)

project(aCG_SYCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ACG_ENABLE_SYCL "Build with SYCL support" ON)
set(ACG_SYCL_TARGET "nvptx64-nvidia-cuda" CACHE STRING "SYCL device target")

add_library(acg_core STATIC
  src/cli/parse_args.cpp
  src/matrix/matrix_market_reader.cpp
  src/runtime/queue_factory.cpp
  src/solver/cg_solver.cpp
)

target_include_directories(acg_core PUBLIC include)

if(ACG_ENABLE_SYCL)
  target_compile_options(acg_core PUBLIC -fsycl -fsycl-targets=${ACG_SYCL_TARGET})
  target_link_options(acg_core PUBLIC -fsycl -fsycl-targets=${ACG_SYCL_TARGET})
endif()

add_executable(acg apps/acg_cli/main.cpp)
target_link_libraries(acg PRIVATE acg_core)
