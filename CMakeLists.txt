cmake_minimum_required(VERSION 3.20)

project(aCG_SYCL LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ACG_ENABLE_SYCL "Build with SYCL support" ON)
option(ACG_ENABLE_MPI "Build with MPI support" ON)
option(ACG_ENABLE_DEVICE_DEBUG "Build SYCL kernels with debug-friendly flags" OFF)
set(ACG_SYCL_TARGET "" CACHE STRING "SYCL device target")
set(ACG_EXTRA_COMPILE_FLAGS "" CACHE STRING "Extra compiler flags for the project")
set(ACG_EXTRA_LINK_FLAGS "" CACHE STRING "Extra linker flags for the project")
set(oneMath_ROOT "/opt/oneMath/install" CACHE PATH "oneMath install root")

if(ACG_EXTRA_COMPILE_FLAGS)
  separate_arguments(ACG_EXTRA_COMPILE_FLAGS_LIST NATIVE_COMMAND "${ACG_EXTRA_COMPILE_FLAGS}")
endif()

if(ACG_EXTRA_LINK_FLAGS)
  separate_arguments(ACG_EXTRA_LINK_FLAGS_LIST NATIVE_COMMAND "${ACG_EXTRA_LINK_FLAGS}")
endif()

add_library(acg_core STATIC
  src/cli/parse_args.cpp
  src/matrix/matrix_market_reader.cpp
  src/matrix/distributed_partition.cpp
  src/runtime/queue_factory.cpp
  src/runtime/mpi_runtime.cpp
  src/solver/algorithms/cg_multi_gpu_mpi.cpp
  src/solver/algorithms/cg_single_gpu.cpp
  src/solver/backends/onemath_cuda_backend.cpp
  src/solver/cg_solver.cpp
  src/solver/rhs_builder.cpp
  src/solver/stopping_criteria.cpp
)

target_include_directories(acg_core PUBLIC include)

if(ACG_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(acg_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(acg_core PUBLIC ACG_HAVE_MPI=1)
endif()

if(ACG_ENABLE_SYCL)
  find_package(oneMath REQUIRED PATHS ${oneMath_ROOT}/lib/cmake/oneMath NO_DEFAULT_PATH)

  target_compile_options(acg_core PUBLIC -fsycl)
  target_link_options(acg_core PUBLIC -fsycl)

  if(ACG_EXTRA_COMPILE_FLAGS)
    target_compile_options(acg_core PUBLIC ${ACG_EXTRA_COMPILE_FLAGS_LIST})
  endif()

  if(ACG_EXTRA_LINK_FLAGS)
    target_link_options(acg_core PUBLIC ${ACG_EXTRA_LINK_FLAGS_LIST})
  endif()

  if(ACG_SYCL_TARGET)
    target_compile_options(acg_core PUBLIC -fsycl-targets=${ACG_SYCL_TARGET})
    target_link_options(acg_core PUBLIC -fsycl-targets=${ACG_SYCL_TARGET})
  endif()
  target_link_libraries(acg_core PUBLIC
    ONEMATH::onemath)

  if(ACG_ENABLE_DEVICE_DEBUG)
    target_compile_options(acg_core PUBLIC -O0 -fno-inline)
    target_link_options(acg_core PUBLIC -O0)
  endif()
endif()

add_executable(acg apps/acg_cli/main.cpp)
target_link_libraries(acg PRIVATE acg_core)
