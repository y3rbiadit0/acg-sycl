ENV ?= local
BUILD_TYPE ?= Release
BUILD_DIR ?= $(if $(filter $(BUILD_TYPE),Debug),build,build-release)
DEVICE ?= gpu
MATRIX ?= data/matrices/Bump_2911/Bump_2911.mtx
ARGS ?=
FRESH_CONFIGURE ?= ON
DEVICE_DEBUG ?= OFF
SYCL_TARGET_OVERRIDE ?=

.PHONY: env-info configure build run \
	configure-debug build-debug run-debug \
	configure-release build-release run-release \
	configure-kernel-debug build-kernel-debug smoke-onemath

env-info:
	ENV=$(ENV) bash tooling/scripts/common.sh >/dev/null && ENV=$(ENV) bash -lc 'source tooling/scripts/common.sh && print_env_summary'

configure:
	ENV=$(ENV) BUILD_TYPE=$(BUILD_TYPE) BUILD_DIR=$(BUILD_DIR) DEVICE_DEBUG=$(DEVICE_DEBUG) FRESH_CONFIGURE=$(FRESH_CONFIGURE) SYCL_TARGET_OVERRIDE=$(SYCL_TARGET_OVERRIDE) bash tooling/scripts/configure.sh

build:
	ENV=$(ENV) BUILD_TYPE=$(BUILD_TYPE) BUILD_DIR=$(BUILD_DIR) bash tooling/scripts/build.sh

run:
	ENV=$(ENV) BUILD_DIR=$(BUILD_DIR) MATRIX=$(MATRIX) DEVICE=$(DEVICE) ARGS='$(ARGS)' bash tooling/scripts/run.sh

configure-debug:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Debug BUILD_DIR=build FRESH_CONFIGURE=OFF

build-debug:
	$(MAKE) configure-debug ENV=$(ENV)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=build BUILD_TYPE=Debug

run-debug:
	$(MAKE) run ENV=$(ENV) BUILD_DIR=build DEVICE=$(DEVICE) MATRIX=$(MATRIX) ARGS='$(ARGS)'

configure-release:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Release BUILD_DIR=build-release FRESH_CONFIGURE=ON

build-release:
	$(MAKE) configure-release ENV=$(ENV)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=build-release BUILD_TYPE=Release

run-release:
	$(MAKE) run ENV=$(ENV) BUILD_DIR=build-release DEVICE=$(DEVICE) MATRIX=$(MATRIX) ARGS='$(ARGS)'

configure-kernel-debug:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Debug BUILD_DIR=build-intel-debug DEVICE_DEBUG=ON SYCL_TARGET_OVERRIDE=spir64 FRESH_CONFIGURE=ON

build-kernel-debug:
	$(MAKE) configure-kernel-debug ENV=$(ENV)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=build-intel-debug BUILD_TYPE=Debug

smoke-onemath:
	ENV=$(ENV) bash -lc 'source tooling/scripts/common.sh && cmake -S tools/onemath_smoke -B tools/onemath_smoke/build -G Ninja -DCMAKE_CXX_COMPILER="$$CXX" -DoneMath_ROOT="$$ACG_ONEMATH_ROOT" && cmake --build tools/onemath_smoke/build && ./tools/onemath_smoke/build/onemath_gemm && ./tools/onemath_smoke/build/onemath_spmv'
