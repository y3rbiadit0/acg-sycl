cmake_minimum_required(VERSION 3.20)

project(aCG_SYCL LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ACG_ENABLE_SYCL "Build with SYCL support" ON)
option(ACG_ENABLE_ONECCL "Enable oneCCL solver device collectives" OFF)
option(ACG_ENABLE_DEVICE_DEBUG "Build SYCL kernels with debug-friendly flags" OFF)
option(ACG_ENABLE_METIS "Enable METIS partitioning" OFF)
set(ACG_SYCL_TARGET "" CACHE STRING "SYCL device target")
set(ACG_EXTRA_COMPILE_FLAGS "" CACHE STRING "Extra compiler flags for the project")
set(ACG_EXTRA_LINK_FLAGS "" CACHE STRING "Extra linker flags for the project")
set(oneMath_ROOT "/opt/oneMath/install" CACHE PATH "oneMath install root")
set(oneCCL_ROOT "" CACHE PATH "oneCCL installation prefix")
set(METIS_HOME "" CACHE PATH "METIS installation prefix")
set(METIS_LIBRARIES "" CACHE STRING "METIS libraries")

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
  src/reporting/solver_report.cpp
  src/runtime/queue_factory.cpp
  src/runtime/mpi_runtime.cpp
  src/solver/cg.cpp
  src/solver/cg_solver.cpp
  src/solver/collectives.cpp
  src/solver/device_csr.cpp
  src/solver/halo.cpp
  src/solver/rhs_builder.cpp
)

target_include_directories(acg_core PUBLIC include)

# The halo exchange and the MPI allreduces pass device pointers, so the MPI
# must be CUDA-aware (HPC-X or the CUDA build of Open MPI on Leonardo).
find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(acg_core PUBLIC MPI::MPI_CXX)

if(ACG_ENABLE_ONECCL)
  find_path(ONECCL_INCLUDE_DIR oneapi/ccl.hpp
    HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
    PATH_SUFFIXES include)
  find_library(ONECCL_LIBRARY ccl
    HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
    PATH_SUFFIXES lib lib64)
  if(NOT ONECCL_INCLUDE_DIR OR NOT ONECCL_LIBRARY)
    message(FATAL_ERROR "ACG_ENABLE_ONECCL=ON but oneCCL was not found. Set oneCCL_ROOT or CCL_ROOT")
  endif()
  target_include_directories(acg_core PUBLIC ${ONECCL_INCLUDE_DIR})
  target_link_libraries(acg_core PUBLIC ${ONECCL_LIBRARY})
  target_compile_definitions(acg_core PUBLIC ACG_HAVE_ONECCL=1)
endif()

if(ACG_ENABLE_METIS)
  if(NOT METIS_LIBRARIES AND DEFINED ENV{METIS_LIBRARIES})
    set(METIS_LIBRARIES "$ENV{METIS_LIBRARIES}")
  endif()
  find_path(METIS_INCLUDE_DIR metis.h
    HINTS ${METIS_HOME} $ENV{METIS_HOME} $ENV{METIS_DIR}
    PATH_SUFFIXES include)
  if(NOT METIS_LIBRARIES)
    find_library(METIS_LIBRARY metis
      HINTS ${METIS_HOME} $ENV{METIS_HOME} $ENV{METIS_DIR}
      PATH_SUFFIXES lib lib64)
    set(METIS_LIBRARIES ${METIS_LIBRARY})
  endif()
  if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARIES)
    message(FATAL_ERROR "ACG_ENABLE_METIS=ON but METIS was not found. Set METIS_HOME, e.g. -DMETIS_HOME=$ENV{METIS_HOME}")
  endif()
  target_include_directories(acg_core PUBLIC ${METIS_INCLUDE_DIR})
  target_link_libraries(acg_core PUBLIC ${METIS_LIBRARIES})
  target_compile_definitions(acg_core PUBLIC ACG_HAVE_METIS=1)
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
