# On Leonardo (/leonardo exists) the environment, modules included, is selected
# automatically; elsewhere pass ENV explicitly.
ENV ?= $(if $(wildcard /leonardo),leonardo,local)
BUILD_TYPE ?= Release
BUILD_DIR ?= $(if $(filter $(BUILD_TYPE),Debug),build,build-release)
DEVICE ?= gpu
MATRIX ?= data/matrices/Bump_2911/Bump_2911.mtx
ARGS ?=
FRESH_CONFIGURE ?= ON
DEVICE_DEBUG ?= OFF
SYCL_TARGET_OVERRIDE ?=
MPI_NP ?= 0
# oneCCL is on by default on Leonardo, where the comparison with native NCCL
# needs it; the install comes from ONECCL_ROOT (tooling/environments/leonardo.sh).
ACG_ENABLE_ONECCL ?= $(if $(wildcard /leonardo),ON,OFF)
ACG_ENABLE_METIS ?= OFF
BUILD_JOBS ?=

# Where `make build-release` puts things. Override to keep variants side by side
# rather than rebuilding over one another -- a second MPI is the reason this
# exists: RELEASE_BUILD_DIR=build-release-hpcx ACG_MPI=hpcx make build-release.
RELEASE_BUILD_DIR ?= build-release

# Leonardo campaign settings. A job runs every key in BACKENDS, interleaved
# round by round in one allocation: SYCL (mpi, oneccl) and native aCG
# (native-mpi, native-nccl) on the same nodes. A 1-GPU topology maps them to
# `none native-none`.
REPO_ROOT            := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
JOBS                 ?= $(REPO_ROOT)/tooling/jobs/leonardo/sycl
BACKENDS             ?= mpi oneccl native-mpi native-nccl
# Keys for a single `make submit-*` / `now-*`; empty means BACKENDS.
SOLVERS              ?=
MATRIX_NAME          ?= Bump_2911
MATRIX_DIR           ?= $(REPO_ROOT)/data/matrices
SCALES               ?= 1n2g 1n4g 2n4g 4n4g 8n4g
SBATCH_ACCOUNT       ?= IscrC_HIGRAPH_0
# QOS=boost_qos_dbg jumps the queue for a smoke test (30 min, max 2 nodes).
# TIME overrides the walltime launch.sh derives from the topology.
TIME                 ?=
QOS                  ?=
# Rounds per job: every key runs once per round, in alternating order.
NTRIALS              ?= 3
# Independent allocations per topology. Allocation-to-allocation spread is the
# larger noise (20% between 8n4g jobs), so a final comparison wants 3.
REPEATS              ?= 1

# Every topology the campaign knows, smallest first. This is the whole list --
# adding 16n4g here is all that is needed, because launch.sh derives --nodes,
# --ntasks-per-node, --gres and --time from the name. SCALES is the multi-GPU
# subset; 1n1g is the single-GPU baseline and is handled separately, since at
# one rank there is no collective to select.
TOPOLOGIES           ?= 1n1g $(SCALES)

LAUNCH = $(JOBS)/launch.sh
# What every launch.sh call inherits. The launcher resolves the allocation shape
# and the walltime itself; these are the campaign's knobs, not Slurm's.
LAUNCH_ENV = ACG_SLURM_ACCOUNT="$(SBATCH_ACCOUNT)" \
	     ACG_PROJECT_ROOT="$(REPO_ROOT)" \
	     ACG_MATRIX_NAME="$(MATRIX_NAME)" \
	     ACG_MATRIX_DIR="$(MATRIX_DIR)" \
	     ACG_ALL_TOPOLOGIES="$(TOPOLOGIES)" \
	     $(if $(TIME),ACG_TIME="$(TIME)",) \
	     $(if $(DRYRUN),ACG_DRYRUN=1,)
QOS_ARG = $(if $(QOS),--qos=$(QOS),)

.PHONY: env-info configure build run \
	configure-debug build-debug run-debug \
	configure-release build-release run-release \
	configure-kernel-debug build-kernel-debug smoke-onemath \
	campaign campaign-plan compare

env-info:
	ENV=$(ENV) bash -lc 'source tooling/scripts/common.sh && print_env_summary'

configure:
	ENV=$(ENV) BUILD_TYPE=$(BUILD_TYPE) BUILD_DIR=$(BUILD_DIR) DEVICE_DEBUG=$(DEVICE_DEBUG) FRESH_CONFIGURE=$(FRESH_CONFIGURE) SYCL_TARGET_OVERRIDE=$(SYCL_TARGET_OVERRIDE) ACG_ENABLE_ONECCL=$(ACG_ENABLE_ONECCL) ACG_ENABLE_METIS=$(ACG_ENABLE_METIS) bash -lc 'source tooling/scripts/configure.sh'

build:
	ENV=$(ENV) BUILD_TYPE=$(BUILD_TYPE) BUILD_DIR=$(BUILD_DIR) BUILD_JOBS=$(BUILD_JOBS) bash -lc 'source tooling/scripts/build.sh'

run:
	ENV=$(ENV) BUILD_DIR=$(BUILD_DIR) MATRIX=$(MATRIX) DEVICE=$(DEVICE) MPI_NP=$(MPI_NP) ARGS='$(ARGS)' bash -lc 'source tooling/scripts/run.sh'

configure-debug:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Debug BUILD_DIR=build FRESH_CONFIGURE=OFF

build-debug:
	$(MAKE) configure-debug ENV=$(ENV)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=build BUILD_TYPE=Debug

run-debug:
	$(MAKE) run ENV=$(ENV) BUILD_DIR=build DEVICE=$(DEVICE) MATRIX=$(MATRIX) ARGS='$(ARGS)'

configure-release:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Release BUILD_DIR=$(RELEASE_BUILD_DIR) FRESH_CONFIGURE=ON ACG_ENABLE_ONECCL=$(ACG_ENABLE_ONECCL) ACG_ENABLE_METIS=$(ACG_ENABLE_METIS)

build-release:
	$(MAKE) configure-release ENV=$(ENV) RELEASE_BUILD_DIR=$(RELEASE_BUILD_DIR)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=$(RELEASE_BUILD_DIR) BUILD_TYPE=Release BUILD_JOBS=$(BUILD_JOBS)

run-release:
	$(MAKE) run ENV=$(ENV) BUILD_DIR=$(RELEASE_BUILD_DIR) DEVICE=$(DEVICE) MATRIX=$(MATRIX) ARGS='$(ARGS)'

configure-kernel-debug:
	$(MAKE) configure ENV=$(ENV) BUILD_TYPE=Debug BUILD_DIR=build-intel-debug DEVICE_DEBUG=ON SYCL_TARGET_OVERRIDE=spir64 FRESH_CONFIGURE=ON

build-kernel-debug:
	$(MAKE) configure-kernel-debug ENV=$(ENV)
	$(MAKE) build ENV=$(ENV) BUILD_DIR=build-intel-debug BUILD_TYPE=Debug

smoke-onemath:
	ENV=$(ENV) bash -lc 'source tooling/scripts/common.sh && cmake --fresh -S tools/onemath_smoke -B tools/onemath_smoke/build -G Ninja -DCMAKE_CXX_COMPILER="$$CXX" -DoneMath_ROOT="$$ACG_ONEMATH_ROOT" -DACG_SYCL_TARGET="$${SYCL_TARGET:-}" -DACG_EXTRA_COMPILE_FLAGS="$${ACG_EXTRA_COMPILE_FLAGS:-}" -DACG_EXTRA_LINK_FLAGS="$${ACG_EXTRA_LINK_FLAGS:-}" && cmake --build tools/onemath_smoke/build && ./tools/onemath_smoke/build/onemath_gemm && ./tools/onemath_smoke/build/onemath_spmv'

# One job per topology (times REPEATS), every key interleaved inside it, so
# each SYCL/native comparison is paired: same nodes, same fabric neighbours.
#
#   make campaign                       # Bump_2911, every topology
#   make campaign MATRIX_NAME=Queen_4147 REPEATS=3
campaign:
	@$(LAUNCH_ENV) ACG_BACKENDS="$(BACKENDS)" ACG_NTRIALS=$(NTRIALS) \
	  ACG_REPEATS=$(REPEATS) $(LAUNCH) --all $(QOS_ARG)

# Any topology by name: `make submit-2n4g`, `make submit-16n4g`. One pattern
# rule rather than a target per shape, for the same reason there is now one job
# script rather than six.
submit-%:
	@$(LAUNCH_ENV) ACG_BACKENDS="$(or $(SOLVERS),$(BACKENDS))" \
	  ACG_NTRIALS=$(NTRIALS) $(LAUNCH) $* $(QOS_ARG)

# Run one topology here and now instead of queueing it: salloc blocks until the
# nodes are granted, then the body runs in the foreground. Named now-% rather
# than run-% because run-debug and run-release already mean something else.
#
#   make now-2n4g NTRIALS=1
#   make now-1n4g NTRIALS=1 SOLVERS=oneccl QOS=boost_qos_dbg
now-%:
	@$(LAUNCH_ENV) ACG_BACKENDS="$(or $(SOLVERS),$(BACKENDS))" \
	  ACG_NTRIALS=$(NTRIALS) $(LAUNCH) --now $* $(QOS_ARG)

# Paired SYCL/native summary of a results tree.
compare:
	@python3 $(JOBS)/compare-summary.py $(or $(ACG_RESULTS_ROOT),results)

# What would be submitted, and how one topology resolves.
campaign-plan:
	@$(LAUNCH_ENV) ACG_BACKENDS="$(BACKENDS)" ACG_NTRIALS=$(NTRIALS) ACG_REPEATS=$(REPEATS) $(LAUNCH) --dry-run --all

explain-%:
	@$(LAUNCH_ENV) ACG_BACKENDS="$(BACKENDS)" ACG_NTRIALS=$(NTRIALS) $(LAUNCH) --explain $*
