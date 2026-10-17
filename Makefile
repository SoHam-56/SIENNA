SHELL := /bin/bash

# =========================================================================
# SIENNA VERIFICATION MAKEFILE CONFIGURATION
# =========================================================================
N    ?= 16
TILE ?= 4
# Number format of every build, regression, test and model run: fp32, bf16 or int8.
FMT  ?= fp32
LANES      ?= 32
COLLAPSE_K ?= 1
# GEN_PKG=0 builds the test_config_pkg.sv already on disk (scripts that write their own); the format guard still runs.
GEN_PKG ?= 1
# Interpreter for the Python flows; tflite and model need the tflite package, e.g. PYTHON=<venv>/bin/python.
PYTHON ?= python3
# model_runner.py has no default model directory, so make model needs one.
MODEL_DIR ?=
# Inferences per model in make model; make regression runs 3.
MODEL_COUNT ?= 1
# make regression: SMALL=1 runs N = 8 and 16 only; REG_DIR holds one result (.log, .json) per point and the table.
SMALL ?= 0
QUICK ?= 0
# Register stages on sienna_top's host, staging, result and output links (L0, L1, L3, L9), a TB_sienna_top parameter; 0 is wires.
LINK_STAGES ?= 0

FORMATS = fp32 bf16 int8
# Exactly one word, and one of FORMATS: anything left after filter-out, or a word count but 1, stops make.
ifneq ($(filter-out $(FORMATS),$(FMT))$(words $(FMT)),1)
$(error Invalid FMT=$(FMT): must be one of fp32, bf16, int8)
endif

# make regression: only SMALL= and FMT= on make's command line narrow it, not exported ones.
REG_SMALL = $(if $(filter command line,$(origin SMALL)),$(SMALL),0)
REG_FMT   = $(if $(filter command line,$(origin FMT)),$(FMT))
ifneq ($(filter-out 0 1,$(REG_SMALL))$(words $(REG_SMALL)),1)
$(error Invalid SMALL=$(SMALL): must be 0 or 1)
endif

# COLLAPSE_K is 0 or 1; TB_sienna_top takes it from the package (make pkg, the guard checks it), sienna_layer's runs and make regression's points do not take it.
ifneq ($(filter-out 0 1,$(COLLAPSE_K))$(words $(COLLAPSE_K)),1)
$(error Invalid COLLAPSE_K=$(COLLAPSE_K): must be 0 or 1)
endif
ifeq ($(COLLAPSE_K),0)
ifneq ($(filter model gemm tflite pack perf-analysis regression regression-summary regression/%,$(MAKECMDGOALS)),)
$(error COLLAPSE_K=0 reaches TB_sienna_top only (pipeline, pkg, verilator, sim, lint and the like): sienna_layer's model, gemm, tflite and pack runs build collapse-k 1, and make regression's points choose their own (<fmt>/N<n>_T<t>/pipeline-ck0))
endif
endif

# EXP_W MAN_W IS_INT ACC_W OUT_W of each FMT, as model_runner.py's _config_items writes them into test_config_pkg.sv.
FMT_FIELDS_fp32 = 8 23 0 32 32
FMT_FIELDS_bf16 = 8 7 0 32 16
FMT_FIELDS_int8 = 0 7 1 32 32

# Project Structure
PRJ_DIR     = $(shell pwd)
SRC_DIR     = $(PRJ_DIR)/src
TB_DIR      = $(PRJ_DIR)/testbenches
SM_DIR      = $(PRJ_DIR)/SystolicMesh/src
SM_LIB_DIR  = $(PRJ_DIR)/SystolicMesh/ArithmeticLibrary
GPNAE_DIR   = $(PRJ_DIR)/GPNAE/src
MAXPOOL_DIR = $(PRJ_DIR)/Maxpool
DROPOUT_DIR = $(PRJ_DIR)/Dropout

# Toolchain
VERILATOR = verilator
VCS       = vcs
WAVE      = surfer

TEST ?=
ACTIVATION ?= tanh
# make pkg (and every build through it) writes this test; it runs in every format.
PKG_TEST = $(or $(TEST),matmul_relu_nopool)

TOP_FILES = \
	sienna_top.sv \
	requant_lanes.sv \
	add_lanes.sv \
	fwft.sv \
	sienna_multi.sv \
	sienna_layer.sv

SM_FILES = \
	top/SystolicMesh.sv \
	top/SystolicArray.sv \
	mem/MeshOutputSram.sv \
	engine/ProcessingElement.sv \
	engine/AccumulationUnit.sv

SM_LIB_FILES = \
	Multipliers/Radix4Booth/src/R4Booth.sv \
	Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
	Multipliers/FP32/src/fp32Multiplier.sv \
	Multipliers/FP/src/fpMultiplier.sv \
	Multipliers/FPWiden/src/fpMulWiden.sv \
	Multipliers/Int/src/intMultiplier.sv \
	Adders/FP32/src/LZC.sv \
	Adders/FP32/src/fp32Adder.sv \
	Adders/FP/src/fpAdder.sv \
	Adders/Int/src/intAdder.sv \
	Converters/FPNarrow/src/fpNarrow.sv \
	Multipliers/Fx/src/fxMac.sv \
	Requant/src/tfliteRequant.sv \
	Add/src/tfliteAdd.sv

# gpnae_poly's lane only: the published gpnae.sv and its InputFIFO / SeLu / sigtan build in GPNAE's own Makefile.
GPNAE_FILES = \
	TYTAN/Memory/CoeffROM.v \
	TYTAN/Memory/ROM.v \
	TYTAN/LZC.v \
	TYTAN/barrel_mac.sv \
	fp32_down.sv \
	gpnae_tail.sv \
	gpnae_poly_int8.sv \
	lane_fifo.sv \
	lane_link.sv \
	gpnae_poly.sv

MAXPOOL_FILES = \
	Maxpool_2D.sv

DROPOUT_FILES = \
	dropout.sv

CREDIT_FILES = \
	credit_link_if.sv \
	credit_counter.sv \
	credit_reg.sv \
	credit_link_checker.sv

DESIGN_FILES = \
	$(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv \
	$(addprefix $(SM_LIB_DIR)/Common/src/,$(CREDIT_FILES)) \
	$(addprefix $(SRC_DIR)/,$(TOP_FILES)) \
	$(addprefix $(SM_DIR)/,$(SM_FILES)) \
	$(addprefix $(SM_LIB_DIR)/,$(SM_LIB_FILES)) \
	$(addprefix $(GPNAE_DIR)/,$(GPNAE_FILES)) \
	$(addprefix $(MAXPOOL_DIR)/,$(MAXPOOL_FILES)) \
	$(addprefix $(DROPOUT_DIR)/,$(DROPOUT_FILES))

# Testbench
# TB_PKG_FILES compiles before TESTBENCH: a package must be declared before it is imported.
TB_PKG_FILES = test_config_pkg.sv
PKG_FILE     = $(TB_DIR)/$(TB_PKG_FILES)
TESTBENCH  = TB_sienna_top.sv
TOP_MODULE = TB_sienna_top

VERILATOR_DIR = $(PRJ_DIR)/Verilator

# Very large meshes put thousands of PEs into a few generated functions that -Os takes hours on; OPT_FAST=-O0 builds in minutes.
OPT_FAST ?= -Os
# Simulation threads; several simulations at once each need fewer (Verilator's threads spin, so oversubscribed runs crawl).
SIM_THREADS ?= $(shell nproc)
VCS_DIR       = $(PRJ_DIR)/VCS

TRACE ?= 0

# ccache 3.7 here served corrupted objects that segfaulted at start-up; USE_CCACHE=1 opts back in.
USE_CCACHE ?= 0
ifeq ($(USE_CCACHE),0)
export CCACHE_DISABLE := 1
endif


ifeq ($(filter $(TRACE),0 vcd fst),)
$(error Invalid TRACE=$(TRACE) — must be one of: 0, vcd, fst)
endif

ifeq ($(TRACE),fst)
TRACE_FILE = TB_sienna_top.fst
else ifeq ($(TRACE),vcd)
TRACE_FILE = TB_sienna_top.vcd
else
TRACE_FILE =
endif

MEM_PATTERNS := *.mem *.hex

MEM_DIRS := \
	$(PRJ_DIR) \
	$(SRC_DIR) \
	$(TB_DIR) \
	$(SM_DIR) \
	$(GPNAE_DIR) \
	$(GPNAE_DIR)/TYTAN/Memory \
	$(MAXPOOL_DIR) \
	$(DROPOUT_DIR)

define copy_mem_files
	@echo "-- Copying memory/config files into $(1)"
	@mkdir -p $(1)
	@copied=0; \
	for d in $(MEM_DIRS); do \
		for p in $(MEM_PATTERNS); do \
			if ls $$d/$$p 1>/dev/null 2>&1; then \
				cp $$d/$$p $(1)/; \
				echo "   Copied $$p from $$d"; \
				copied=1; \
			fi; \
		done; \
	done; \
	if [ $$copied -eq 0 ]; then echo "   (no *.mem/*.hex found)"; fi
endef

VERILATOR_FLAGS = \
	--timing \
	--assert \
	--top-module $(TOP_MODULE) \
	--threads $(SIM_THREADS) \
	-j $(shell nproc) \
	--build-jobs $(shell nproc) \
	--output-split 20000 \
	--output-split-cfuncs 20000 \
	--output-groups 64 \
	-Wno-UNOPTTHREADS \
	-MAKEFLAGS OPT_FAST=$(OPT_FAST) \
	--sv \
	-I$(SRC_DIR) \
	-I$(TB_DIR) \
	-I$(SM_DIR) \
	-I$(SM_LIB_DIR) \
	-I$(GPNAE_DIR) \
	-I$(MAXPOOL_DIR) \
	-I$(DROPOUT_DIR) \
	--Mdir $(VERILATOR_DIR) \
	--Wno-WIDTHTRUNC \
	--Wno-WIDTHEXPAND \
	--Wno-WIDTHCONCAT \
	--Wno-CASEINCOMPLETE \
	--Wno-SELRANGE \
	--Wno-LATCH \
	--Wno-REALCVT \
	--Wno-SHORTREAL \
	--Wno-TIMESCALEMOD \
	--Wno-UNSIGNED

# Hook for one-off defines, e.g. make verilator EXTRA_FLAGS=-DBACK_TO_BACK; SIM_ARGS go to the simulator (e.g. +verilator+rand+reset+2)
VERILATOR_FLAGS += $(EXTRA_FLAGS)
ifneq ($(LINK_STAGES),0)
VERILATOR_FLAGS += -GLINK_STAGES=$(LINK_STAGES)
endif

ifeq ($(TRACE),fst)
VERILATOR_FLAGS += --trace-fst --trace-structs --trace-max-array 2048 --trace-max-width 1024 -DENABLE_TRACE -DTRACE_FST
else ifeq ($(TRACE),vcd)
VERILATOR_FLAGS += --trace --trace-structs --trace-max-array 2048 --trace-max-width 1024 -DENABLE_TRACE
endif

VCS_FLAGS = \
	-full64 \
	-sverilog \
	-timescale=1ns/100ps \
	-Mdir=$(VCS_DIR) \
	+v2k \
	+incdir+$(SRC_DIR) \
	+incdir+$(TB_DIR) \
	+incdir+$(SM_DIR) \
	+incdir+$(SM_LIB_DIR) \
	+incdir+$(GPNAE_DIR) \
	+incdir+$(MAXPOOL_DIR) \
	+incdir+$(DROPOUT_DIR) \
	+define+VCS

ifeq ($(TRACE),vcd)
VCS_FLAGS += -debug_all +define+ENABLE_TRACE
else ifeq ($(TRACE),fst)
VCS_FLAGS += -debug_all +define+ENABLE_TRACE
endif

default: help

help:
	@echo "=== SIENNA Hardware Simulation Makefile ==="
	@echo ""
	@echo "Precision (one variable for every build and run):"
	@echo "  FMT=fp32|bf16|int8      - number format (default fp32); any other value stops make"
	@echo "  N, TILE, LANES          - mesh size, tile size, activation lanes (default 16, 4, 32)"
	@echo "  COLLAPSE_K=1|0          - mesh COLLAPSE_K (default 1), for TB_sienna_top and sm-verilator; sienna_layer's runs build 1"
	@echo "  TEST=<name>             - make pkg / build test (default matmul_relu_nopool); pipeline: name filter"
	@echo "  make pkg                - Write testbenches/test_config_pkg.sv + that test's stimulus for FMT, N, TILE, TEST"
	@echo "  GEN_PKG=0               - Build the package already on disk (after gen-matmul/gen-conv); still guarded"
	@echo "  verilator, lint, debug, perf and vcs run make pkg first, then check its format and N/TILE/LANES"
	@echo "  NOTE: so a default make verilator rewrites the tracked testbenches/test_config_pkg.sv; don't commit it"
	@echo ""
	@echo "Simulation Targets:"
	@echo "  make verilator           - Simulate using Verilator (no waveform)"
	@echo "  make sim                 - Run the last TB_sienna_top build on the stimulus in testbenches/, no build; refuses a build of other inputs or flags"
	@echo "  make build-key           - Print the key make sim compares with the build's (inputs, package without the run-time knobs, flags)"
	@echo "  make verilator TRACE=fst - Simulate + write FST waveform (recommended)"
	@echo "  make verilator TRACE=vcd - Simulate + write VCD waveform (larger)"
	@echo "  make vcs                 - Simulate using Synopsys VCS (no waveform)"
	@echo "  make vcs       TRACE=vcd - Simulate + write VCD waveform"
	@echo "  make vcs       TRACE=fst - VCS has no native FST dump; still writes VCD"
	@echo ""
	@echo "Individual Module Targets:"
	@echo "  make sm-verilator      - Systolic Mesh only: its model checks, then its regression in FMT at N, TILE, COLLAPSE_K (SM_CHECKS=0 skips the checks)"
	@echo "  make gpnae-verilator   - GPNAE only: its regression in FMT on gpnae_poly, the lane sienna_top uses"
	@echo "  make gpnae-verilator GPNAE_MODEL=hw - the same, bit-exact against the lane model instead of accuracy (int8 ignores it)"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  make lint              - Run Verilator lint check"
	@echo "  make debug             - Build with GDB back-trace"
	@echo "  make perf              - Build with performance profiling"
	@echo ""
	@echo "Format-aware Run Targets (all take FMT, N, LANES, and TILE):"
	@echo "  make pipeline FMT=int8            - Full pipeline regression (TEST=<substring> narrows it)"
	@echo "  make model FMT=bf16 MODEL_DIR=<d> - MLPerf Tiny models on the RTL (model_runner.py; no int8)"
	@echo "  make gemm FMT=int8 [QUICK=1]      - GEMM shape sweep on sienna_layer (regression.py --action gemm)"
	@echo "  make perf-analysis FMT=bf16       - Cycle/latency/throughput report (regression.py --action perf)"
	@echo "  make tflite FMT=int8              - Single-layer TFLite int8 models, bit for bit (model_runner.py --action tflite); needs FMT=int8"
	@echo "  make pack FMT=int8                - packed layers vs each job alone (regression.py --action pack)"
	@echo "  make add-lanes                    - add_lanes against ipu.tflite_add (int8 residual ADD)"
	@echo "  make regression MODEL_DIR=<d>     - SIENNA's tests (pipeline, model, pack, gemm, tflite) at N = 8, 16, 32, 64 x every tile size x fp32, bf16, int8, then one table (long: make -jN, N for your memory)"
	@echo "  make regression SMALL=1           - The same at N = 8 and 16 only; FMT=<f> narrows either to one format"
	@echo "  make regression/<point>           - One point, e.g. regression/bf16/N8_T2/pipeline (regression.py --action points lists them)"
	@echo "  make regression-summary           - The table of the results in REG_DIR (default testbenches/results/regression)"
	@echo "                                      SystolicMesh, GPNAE and ArithmeticLibrary each run their own: make -C <repo> regression"
	@echo "  PYTHON=<venv>/bin/python          - Interpreter for these (tflite and model need the tflite package)"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make wave              - View waveforms (requires prior TRACE=vcd|fst run)"
	@echo "  make gen-matmul        - Generate matmul stimulus in FMT (N=16 default); build it with GEN_PKG=0"
	@echo "  make gen-conv          - Generate conv stimulus in FMT   (N=16 default); build it with GEN_PKG=0"
	@echo "  make clean             - Remove all simulation artifacts"
	@echo "  make clean-all         - Clean all including subprojects"
	@echo ""
	@echo "Disk usage note:"
	@echo "  TRACE=0   (default) → no waveform written; Verilator dir stays small."
	@echo "  TRACE=fst           → compact waveform, recommended for this project."
	@echo "  TRACE=vcd           → can be very large for complex designs."
	@echo ""
	@echo "Current Configuration:"
	@echo "  FMT        : $(FMT)  (N=$(N) TILE=$(TILE) LANES=$(LANES) COLLAPSE_K=$(COLLAPSE_K) TEST=$(PKG_TEST))"
	@echo "  TOP_MODULE : $(TOP_MODULE)"
	@echo "  TESTBENCH  : $(TESTBENCH)"
	@echo "  TRACE      : $(TRACE)"
	@echo ""
	@echo "Design Files:"
	@echo "  Systolic Mesh : $(words $(SM_FILES)) files  |  SM Lib: $(words $(SM_LIB_FILES)) files"
	@echo "  GPNAE         : $(words $(GPNAE_FILES)) files"
	@echo "  Maxpool       : $(words $(MAXPOOL_FILES)) files"
	@echo "  Dropout       : $(words $(DROPOUT_FILES)) files"
	@echo "  Total         : $(words $(DESIGN_FILES)) files"

# ─────────────────────────────────────────────────────────────────────────────
# Vector generation convenience targets
# ─────────────────────────────────────────────────────────────────────────────

gen-matmul:
	@echo "=== Generating matmul stimulus (FMT=$(FMT), N=$(N), tile=$(TILE)) ==="
	$(PYTHON) regression.py \
		--action gen \
		--mode matmul \
		--format $(FMT) \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--collapse-k $(COLLAPSE_K)

gen-conv:
	@echo "=== Generating conv stimulus (FMT=$(FMT), N=$(N), tile=$(TILE)) ==="
	$(PYTHON) regression.py \
		--action gen \
		--mode conv \
		--conv-type basic \
		--format $(FMT) \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--collapse-k $(COLLAPSE_K)

# ─────────────────────────────────────────────────────────────────────────────
# Test package: written for FMT, then checked before every build
# ─────────────────────────────────────────────────────────────────────────────
pkg:
	@echo "=== Writing test_config_pkg.sv: FMT=$(FMT) N=$(N) TILE=$(TILE) LANES=$(LANES) COLLAPSE_K=$(COLLAPSE_K) TEST=$(PKG_TEST) ==="
	$(PYTHON) regression.py \
		--action pkg \
		--format $(FMT) \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--collapse-k $(COLLAPSE_K) \
		--test $(PKG_TEST)

# Fails the build unless the package's EXP_W, MAN_W, IS_INT (when present), ACC_W, OUT_W, N, TILE_SIZE and NUM_LANES are FMT's, N, TILE and LANES.
pkg-check: $(if $(filter 0,$(GEN_PKG)),,pkg)
	@[ -r $(PKG_FILE) ] || { echo "ERROR: $(PKG_FILE) is missing or unreadable: run make pkg FMT=$(FMT), or build without GEN_PKG=0."; exit 1; }
	@set -- $(FMT_FIELDS_$(FMT)); \
	field() { awk -v k="$$1" '$$1 == "localparam" && $$3 == k { sub(/;.*/, "", $$5); print $$5 }' $(PKG_FILE) 2>/dev/null; }; \
	e=$$(field EXP_W); m=$$(field MAN_W); i=$$(field IS_INT); n=$$(field N); t=$$(field TILE_SIZE); l=$$(field NUM_LANES); \
	if [ "$$e" != "$$1" ] || [ "$$m" != "$$2" ] || { [ -n "$$i" ] && [ "$$i" != "$$3" ]; }; then \
		echo "ERROR: $(PKG_FILE) has EXP_W=$${e:-missing} MAN_W=$${m:-missing} IS_INT=$${i:-absent}, but FMT=$(FMT) needs EXP_W=$$1 MAN_W=$$2 IS_INT=$$3."; \
		echo "       The package is stale or for another format: run make pkg FMT=$(FMT), or build without GEN_PKG=0."; \
		exit 1; \
	fi; \
	a=$$(field ACC_W); o=$$(field OUT_W); \
	if [ "$$a" != "$$4" ] || [ "$$o" != "$$5" ]; then \
		echo "ERROR: $(PKG_FILE) has ACC_W=$${a:-missing} OUT_W=$${o:-missing}, but FMT=$(FMT) sums in $$4 bits and gives $$5-bit results (ACC_W=$$4 OUT_W=$$5)."; \
		echo "       The package predates fp32 sums for bf16 or is stale: run make pkg FMT=$(FMT), or build without GEN_PKG=0."; \
		exit 1; \
	fi; \
	if [ "$$n" != "$(N)" ] || [ "$$t" != "$(TILE)" ] || [ "$$l" != "$(LANES)" ]; then \
		echo "ERROR: $(PKG_FILE) has N=$${n:-missing} TILE_SIZE=$${t:-missing} NUM_LANES=$${l:-missing}, but the build asks for N=$(N) TILE=$(TILE) LANES=$(LANES)."; \
		echo "       The package was written for another geometry: run make pkg with these N, TILE, LANES, or pass the package's."; \
		exit 1; \
	fi; \
	c=$$(field COLLAPSE_K); \
	if [ "$$c" != "$(COLLAPSE_K)" ]; then \
		echo "ERROR: $(PKG_FILE) has COLLAPSE_K=$${c:-missing}, but the build asks for COLLAPSE_K=$(COLLAPSE_K)."; \
		echo "       The package is for the other mesh mode or predates it: run make pkg with this COLLAPSE_K, or pass the package's."; \
		exit 1; \
	fi; \
	echo "-- test_config_pkg.sv matches FMT=$(FMT) N=$(N) TILE=$(TILE) LANES=$(LANES) COLLAPSE_K=$(COLLAPSE_K): EXP_W=$$e MAN_W=$$m IS_INT=$${i:-absent} ACC_W=$$a OUT_W=$$o"

# ─────────────────────────────────────────────────────────────────────────────
# Verilator — SIENNA Top
# ─────────────────────────────────────────────────────────────────────────────
# TB_sienna_top reads these package items at run time from test_knobs.mem, so a build serves every test that agrees on the rest.
TB_KNOBS = ACTIVATION_CODE NUM_SETS HAS_BIAS WEIGHT_CACHE ACCUM_PASSES MIXED_LEN MIXED_ACTS PACKED TRAINING_MODE DROPOUT_SEED RESIDUAL
# What a build depends on: each Verilator input but the package by checksum, the package without TB_KNOBS, and the flags.
BUILD_KEY = { for f in $$(sed -e 's/^[^:]*://' -e 's/\\$$//' $(VERILATOR_DIR)/V$(TOP_MODULE)__ver.d | tr ' ' '\n' | sort -u); do \
	  [ -z "$$f" ] || [ "$$f" = "$(PKG_FILE)" ] || md5sum "$$f"; done; \
	awk '!($$1 == "localparam" && $$3 ~ /^($(subst $(space),|,$(TB_KNOBS)))$$/)' $(PKG_FILE); \
	echo '$(TOP_MODULE) $(VERILATOR_FLAGS)'; }

verilator: pkg-check
	@echo "=== Verilator simulation: $(TOP_MODULE)  FMT=$(FMT)  TRACE=$(TRACE) ==="
	@mkdir -p $(VERILATOR_DIR)
	@rm -f $(VERILATOR_DIR)/build_key.txt
	$(VERILATOR) --binary \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/$(TESTBENCH) \
		-o $(TOP_MODULE)_sim
	@echo "-- Compiling Verilator C++ model"
	$(MAKE) -C $(VERILATOR_DIR) -f V$(TOP_MODULE).mk
	@$(BUILD_KEY) > $(VERILATOR_DIR)/build_key.txt
	$(call copy_mem_files,$(VERILATOR_DIR))
	@echo "-- Running simulation"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim $(SIM_ARGS)
	@echo "-- Done"
ifneq ($(TRACE),0)
	@echo "-- Trace ($(TRACE)) : $(VERILATOR_DIR)/$(TRACE_FILE)"
endif

# The last build, run on the stimulus now in testbenches/ without building; it refuses a build of other inputs, package items (but TB_KNOBS) or flags.
sim: pkg-check
	@[ -x $(VERILATOR_DIR)/$(TOP_MODULE)_sim ] && [ -r $(VERILATOR_DIR)/build_key.txt ] || { echo "ERROR: no finished build in $(VERILATOR_DIR): make verilator first"; exit 1; }
	@$(BUILD_KEY) | cmp -s - $(VERILATOR_DIR)/build_key.txt || { echo "ERROR: $(VERILATOR_DIR) holds a build of other sources, package items or flags: make verilator"; exit 1; }
	$(call copy_mem_files,$(VERILATOR_DIR))
	@echo "-- Running simulation (no build: $(VERILATOR_DIR)'s inputs, package and flags are this run's)"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim $(SIM_ARGS)
	@echo "-- Done"

# The key make sim compares with the build's (BUILD_KEY), printed.
build-key:
	@$(BUILD_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Sub-module targets
# ─────────────────────────────────────────────────────────────────────────────
# The mesh TB takes its format only from SystolicMesh's regression, which patches EXP_W/MAN_W/COLLAPSE_K into it.
# SM_CHECKS=0: without the model checks (aril-fpu, aril-narrow, model-tests); SystolicMesh's own make regression covers every N, tile size and format.
SM_CHECKS ?= 1
sm-verilator:
	@echo "=== Systolic Mesh only: FMT=$(FMT) N=$(N) TILE=$(TILE) COLLAPSE_K=$(COLLAPSE_K) ==="
	$(MAKE) -C SystolicMesh mesh-regression MATRIX_SIZE=$(N) \
		REGRESSION_OPTS="--format $(FMT) --collapse-k $(COLLAPSE_K) --tiles $(TILE)$(if $(filter 0,$(SM_CHECKS)), --no-checks)"

# GPNAE's Makefile takes no format: its regression writes gpnae_test_config.svh for FMT, then runs make verilator.
# GPNAE_MODEL: exact (accuracy against the functions) or hw (bit-exact against gpnae_poly's model); int8 ignores it and runs both.
GPNAE_MODEL ?= exact
gpnae-verilator:
	@echo "=== GPNAE only: FMT=$(FMT), gpnae_poly lane, model $(GPNAE_MODEL) ==="
	cd GPNAE && $(PYTHON) regression.py --lane poly --format $(FMT) --model $(GPNAE_MODEL)

# ─────────────────────────────────────────────────────────────────────────────
# VCS
# ─────────────────────────────────────────────────────────────────────────────
vcs: pkg-check
	@echo "=== VCS simulation: $(TOP_MODULE)  TRACE=$(TRACE) ==="
ifeq ($(TRACE),fst)
	@echo "-- WARNING: VCS's \$$dumpvars only produces VCD; this run will write"
	@echo "            TB_sienna_top.vcd, not FST (need \$$fsdbDumpvars/Verdi for FST)."
endif
	@mkdir -p $(VCS_DIR)
	$(VCS) $(VCS_FLAGS) \
		-o $(VCS_DIR)/$(TOP_MODULE)_sim \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/$(TESTBENCH)
	$(call copy_mem_files,$(VCS_DIR))
	@echo "-- Running simulation"
	cd $(VCS_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- VCS simulation complete"

# ─────────────────────────────────────────────────────────────────────────────
# Waveform viewer
# ─────────────────────────────────────────────────────────────────────────────
wave:
	@if [ -f $(VERILATOR_DIR)/TB_sienna_top.fst ]; then \
		echo "-- Opening Verilator waveform (FST)"; \
		$(WAVE) $(VERILATOR_DIR)/TB_sienna_top.fst; \
	elif [ -f $(VERILATOR_DIR)/TB_sienna_top.vcd ]; then \
		echo "-- Opening Verilator waveform (VCD)"; \
		$(WAVE) $(VERILATOR_DIR)/TB_sienna_top.vcd; \
	elif compgen -G "$(VCS_DIR)/*.vpd" > /dev/null; then \
		echo "-- Opening VCS waveform"; \
		$(WAVE) $(VCS_DIR)/*.vpd; \
	else \
		echo "-- No waveform found. Run with TRACE=fst or TRACE=vcd first."; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Lint / debug / perf
# ─────────────────────────────────────────────────────────────────────────────
lint: pkg-check
	@echo "=== Linting $(TOP_MODULE)  FMT=$(FMT) ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --lint-only \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/$(TESTBENCH)
	@echo "-- Lint complete"

debug: VERILATOR_FLAGS += --debug --gdbbt
debug: verilator

perf: VERILATOR_FLAGS += --stats --profile-cfuncs
perf: verilator

# ─────────────────────────────────────────────────────────────────────────────
# File listing / checking
# ─────────────────────────────────────────────────────────────────────────────
list-files:
	@echo "=== Design Files ==="
	@echo "Top Level ($(SRC_DIR)):"; \
	for f in $(TOP_FILES); do echo "  - $$f"; done
	@echo "Systolic Mesh ($(SM_DIR)):"; \
	for f in $(SM_FILES); do echo "  - $$f"; done
	@echo "SM ArithmeticLibrary ($(SM_LIB_DIR)):"; \
	for f in $(SM_LIB_FILES); do echo "  - $$f"; done
	@echo "GPNAE ($(GPNAE_DIR)):"; \
	for f in $(GPNAE_FILES); do echo "  - $$f"; done
	@echo "Maxpool ($(MAXPOOL_DIR)):"; \
	for f in $(MAXPOOL_FILES); do echo "  - $$f"; done
	@echo "Dropout ($(DROPOUT_DIR)):"; \
	for f in $(DROPOUT_FILES); do echo "  - $$f"; done
	@echo "Total: $(words $(DESIGN_FILES)) files"

check-files:
	@echo "=== Checking source files exist ==="
	@missing=0; \
	for f in $(DESIGN_FILES); do \
		if [ ! -f "$$f" ]; then \
			echo "  MISSING: $$f"; missing=$$((missing+1)); \
		fi; \
	done; \
	if [ $$missing -eq 0 ]; then \
		echo "  All $(words $(DESIGN_FILES)) files present."; \
	else \
		echo "  $$missing file(s) missing!"; exit 1; \
	fi

pipeline:
	@echo "=== Running Sienna Pipeline Regression: FMT=$(FMT) ==="
	$(PYTHON) regression.py \
		--matrix-size $(N) \
		--tile-size $(TILE) \
		--format $(FMT) \
		--lanes $(LANES) \
		--collapse-k $(COLLAPSE_K) \
		$(if $(TEST),--test $(TEST))

# ─────────────────────────────────────────────────────────────────────────────
# Format-aware runs (each script writes its own package and builds with GEN_PKG=0)
# ─────────────────────────────────────────────────────────────────────────────
model:
	@[ -n "$(MODEL_DIR)" ] || { echo "ERROR: make model needs MODEL_DIR=<directory of the .tflite models>; model_runner.py has no default"; exit 1; }
	$(PYTHON) model_runner.py \
		--model-dir $(MODEL_DIR) \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--format $(FMT) \
		--count $(MODEL_COUNT)

# GEMM shape sweep on sienna_layer, built at N, TILE and LANES.
gemm:
	$(PYTHON) regression.py --action gemm \
		--n $(N) \
		--tile $(TILE) \
		--lanes $(LANES) \
		--format $(FMT) \
		$(if $(filter 1,$(QUICK)),--quick)

perf-analysis:
	$(PYTHON) regression.py --action perf \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--format $(FMT)

tflite:
	@[ "$(FMT)" = int8 ] || { echo "ERROR: make tflite runs the int8 TFLite models only and needs FMT=int8 (FMT=$(FMT))"; exit 1; }
	$(PYTHON) model_runner.py --action tflite \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES)

# Packed layers against each job alone on sienna_layer (regression.py --action pack), in FMT at N, TILE and LANES.
pack:
	$(PYTHON) regression.py --action pack --format $(FMT) --n $(N) --tile $(TILE) --lanes $(LANES)

# add_lanes against ipu.tflite_add (regression.py --action add-vectors); fails unless the log has RESULT: PASSED and no error or failed assertion; its patterns are bracketed, so make's echo of the recipe does not read as a firing
ADDL_DIR = $(VERILATOR_DIR)/add_lanes
add-lanes:
	$(PYTHON) regression.py --action add-vectors $(TB_DIR)/add_lanes.mem
	mkdir -p $(ADDL_DIR)
	$(VERILATOR) --binary --timing --assert --sv -j $(shell nproc) $(filter --Wno-%,$(VERILATOR_FLAGS)) --top-module TB_add_lanes --Mdir $(ADDL_DIR) \
	  $(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/Add/src/tfliteAdd.sv $(SRC_DIR)/add_lanes.sv $(TB_DIR)/TB_add_lanes.sv -o sim
	set -o pipefail; cd $(TB_DIR) && $(ADDL_DIR)/sim | tee $(ADDL_DIR)/sim.log && command grep -q '^RESULT: PASSED' $(ADDL_DIR)/sim.log && \
	  ! command grep -qE '%[E]rror|%[F]atal|Assertion [f]ailed' $(ADDL_DIR)/sim.log || { echo "TB_add_lanes: failed ($(ADDL_DIR)/sim.log)"; exit 1; }

# make regression: SIENNA's points, each its own target (make -jN runs N side by side), then the table; FMT=<f> narrows it to one format.
comma := ,
empty :=
space := $(empty) $(empty)
REG_FORMATS = $(subst $(space),$(comma),$(strip $(or $(REG_FMT),$(FORMATS))))
REG_SELECT  = --formats $(REG_FORMATS)$(if $(filter 1,$(REG_SMALL)), --small)
REG_DIR    ?= $(TB_DIR)/results/regression
REG_SUMMARY = $(PYTHON) regression.py --action summary --results $(REG_DIR) $(REG_SELECT) --model-dir=$(abspath $(MODEL_DIR))
ifneq ($(filter regression,$(MAKECMDGOALS)),)
ifeq ($(origin REG_RUN),undefined)
REG_RUN := $(shell date +%Y%m%d-%H%M%S)-$(shell echo $$PPID)
endif
REG_POINTS := $(shell $(PYTHON) regression.py --action points $(REG_SELECT))
endif

# One point, e.g. make regression/bf16/N8_T2/pipeline: runs in a private copy of the tree, writes $(REG_DIR)/<point>.log and .json.
regression/%:
	@$(PYTHON) regression.py --action point --name $* --results $(REG_DIR) --model-dir=$(abspath $(MODEL_DIR))$(if $(REG_RUN), --run $(REG_RUN))$(if $(filter regression,$(MAKECMDGOALS)), --keep-going)

regression: $(addprefix regression/,$(REG_POINTS))
	@$(REG_SUMMARY) --run $(REG_RUN)

# The table of the results already in REG_DIR (SMALL and FMT pick the points it expects); fails on any FAIL or NOT RUN.
regression-summary:
	@$(REG_SUMMARY)$(if $(REG_RUN), --run $(REG_RUN))

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
clean:
	@echo "-- Cleaning simulation artifacts"
	-rm -rf $(VERILATOR_DIR) $(VCS_DIR)
	-rm -f *.vpd *.vcd *.wlf *.log
	-rm -f csrc simv simv.daidir *.key DVEfiles
	@echo "-- Clean complete"

clean-all: clean
	@echo "-- Cleaning subprojects"
	-$(MAKE) -C SystolicMesh clean 2>/dev/null || true
	-$(MAKE) -C GPNAE clean 2>/dev/null || true
	@echo "-- Deep clean complete"

.PHONY: default help verilator vcs sm-verilator gpnae-verilator \
        wave lint debug perf list-files check-files clean clean-all \
        gen-matmul gen-conv pipeline pkg pkg-check model gemm perf-analysis tflite pack add-lanes regression regression-summary sim build-key
