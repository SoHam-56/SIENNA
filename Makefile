SHELL := /bin/bash

# =========================================================================
# SIENNA VERIFICATION MAKEFILE CONFIGURATION
# =========================================================================
N    ?= 16
TILE ?= 4
# Number format of every build, regression, test and model run: FMT=fp32 (default), bf16 or int8, through ArithmeticLibrary's fmt.mk below.
LANES      ?= 32
COLLAPSE_K ?= 1
# GEN_PKG=0 builds the test_config_pkg.sv already on disk (scripts that write their own); pkg-geometry still checks its N, TILE_SIZE, NUM_LANES and COLLAPSE_K.
GEN_PKG ?= 1
# Interpreter for the Python flows; tflite and model need the tflite package, e.g. PYTHON=<venv>/bin/python.
PYTHON ?= python3
# The MLPerf Tiny models make model and the model points run: models/ (NOTICE: sources, licenses, the inputs not committed); MODEL_DIR=<d> picks another.
MODEL_DIR ?= $(PRJ_DIR)/models
# Inferences per model in make model; make regression runs 3.
MODEL_COUNT ?= 1
# make regression: SMALL=1 runs N = 8 and 16 only; REG_DIR holds one result (.log, .json) per point and the table.
SMALL ?= 0
QUICK ?= 0
# Register stages on sienna_top's host, staging, result and output links (L0, L1, L3, L9), a TB_sienna_top parameter; 0 is wires.
LINK_STAGES ?= 0

# make regression: only SMALL= and FMT= on make's command line narrow it, not exported ones.
REG_SMALL = $(if $(filter command line,$(origin SMALL)),$(SMALL),0)
REG_FMT   = $(if $(filter command line,$(origin FMT)),$(FMT))
ifneq ($(filter-out 0 1,$(REG_SMALL))$(words $(REG_SMALL)),1)
$(error Invalid SMALL=$(SMALL): must be 0 or 1)
endif

# COLLAPSE_K is 0 or 1; TB_sienna_top and the layer contract's model, gemm and tflite runs take it from the package (the guard checks it), pack and make regression's points do not.
ifneq ($(filter-out 0 1,$(COLLAPSE_K))$(words $(COLLAPSE_K)),1)
$(error Invalid COLLAPSE_K=$(COLLAPSE_K): must be 0 or 1)
endif
ifeq ($(COLLAPSE_K),0)
ifneq ($(filter pack perf-analysis regression regression-summary regression/%,$(MAKECMDGOALS)),)
$(error COLLAPSE_K=0 reaches TB_sienna_top and the layer contract's model, gemm and tflite runs: pack refuses it (a_pack_collapsed) and make regression's points choose their own)
endif
endif

# Project Structure
PRJ_DIR     = $(shell pwd)
SRC_DIR     = $(PRJ_DIR)/src
TB_DIR      = $(PRJ_DIR)/testbenches
SM_DIR      = $(PRJ_DIR)/SystolicMesh/src
SM_LIB_DIR  = $(PRJ_DIR)/SystolicMesh/ArithmeticLibrary
GPNAE_DIR   = $(PRJ_DIR)/GPNAE/src
MAXPOOL_DIR = $(SRC_DIR)/pool
DROPOUT_DIR = $(SRC_DIR)/dropout

# FMT ?= fp32; any value but fp32, bf16 or int8 stops make; FMT_DEF is the build's +define+FMT_<FMT>, which sienna_fmt_pkg reads.
# A command-line FMT_DEF= builds with no define, which must elaborate fp32 (make regression's default point).
include $(SM_LIB_DIR)/Common/fmt.mk
# SIM ?= verilator, or vcs (ArithmeticLibrary's sim.mk: plain VCS builds; TRACE=vcd, XPROP=1, RANDINIT=1); every Python flow reads it as SIENNA_SIM.
VCS_LINT_BASELINE = $(PRJ_DIR)/vcs_lint_baseline.txt
include $(SM_LIB_DIR)/Common/sim.mk
export SIENNA_SIM := $(SIM)

# Toolchain
VERILATOR = verilator
VCS       = vcs
WAVE      = surfer

TEST ?=
ACTIVATION ?= tanh
# make pkg (and every build through it) writes this test; it runs in every format.
PKG_TEST = $(or $(TEST),matmul_relu_nopool)

TOP_FILES = \
	top/sienna_top.sv \
	pipeline/sienna_sched.sv \
	pipeline/sienna_pipe.sv \
	lanes/requant_lanes.sv \
	lanes/add_lanes.sv \
	fifo/fwft.sv

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
	Multipliers/Int/src/intMultiplier.sv \
	Adders/FP32/src/LZC.sv \
	Adders/FP32/src/fp32Adder.sv \
	Adders/Int/src/intAdder.sv \
	Requantizer/src/int8Requantizer.sv \
	QuantizedAdder/src/int8QuantizedAdder.sv \
	Fmt/src/fmt_mul.sv \
	Fmt/src/fmt_add.sv \
	Multipliers/BF16/src/bf16Multiplier.sv \
	Multipliers/BF16Wide/src/bf16WideMultiplier.sv \
	Multipliers/Int8/src/int8Multiplier.sv \
	Adders/BF16/src/bf16Adder.sv \
	Adders/Int32/src/int32Adder.sv \
	Converters/FP32ToBF16/src/fp32ToBf16.sv \
	Multipliers/Int16/src/int16Multiplier.sv \
	Multipliers/Int24/src/int24Multiplier.sv

# The gpnae lane (GPNAE's own files; the units it picks are ArithmeticLibrary's, above).
GPNAE_FILES = \
	gpnae_pkg.sv \
	lane_fifo.sv \
	lane_link.sv \
	fixedPointMultiplyAdd.sv \
	gpnae_horner.sv \
	TYTAN/barrel_mac.sv \
	gpnae_exp_split.sv \
	gpnae_coef_mem.sv \
	gpnae_act_in.sv \
	gpnae_post.sv \
	gpnae_tail.sv \
	gpnae.sv

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

# One verilation thread: Verilator 5.035 at -j 8 segfaulted in 3 of 63 TB_sienna_top builds on od-r7a-2xl (0 of 96 at -j 1, same wall time); the C++ build keeps every core.
VERILATOR_FLAGS = \
	--timing \
	--assert \
	--top-module $(TOP_MODULE) \
	--threads $(SIM_THREADS) \
	-j 1 \
	--build-jobs $(shell nproc) \
	--output-split 20000 \
	--output-split-cfuncs 20000 \
	--output-groups 64 \
	-Wno-UNOPTTHREADS \
	-MAKEFLAGS OPT_FAST=$(OPT_FAST) \
	--sv \
	-I$(SRC_DIR)/top \
	-I$(TB_DIR)/top \
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

# The build's precision define, which sienna_fmt_pkg's map and the math modules read.
VERILATOR_FLAGS += $(FMT_DEF)

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

# VCS: the same include paths and define, EXTRA_FLAGS' -G as -pvalue+ and -D as +define+, LINK_STAGES; sim.mk keeps it plain (TRACE=vcd adds waveforms).
VCS_FLAGS = +incdir+$(SRC_DIR)/top +incdir+$(TB_DIR)/top +incdir+$(SM_DIR) +incdir+$(SM_LIB_DIR) +incdir+$(GPNAE_DIR) +incdir+$(MAXPOOL_DIR) +incdir+$(DROPOUT_DIR) \
	$(FMT_DEF) $(call vcs_g,$(TOP_MODULE),$(filter -G%,$(EXTRA_FLAGS))) $(patsubst -D%,+define+%,$(filter -D%,$(EXTRA_FLAGS))) \
	$(filter +define+%,$(EXTRA_FLAGS)) $(if $(filter-out 0,$(LINK_STAGES)),-pvalue+$(TOP_MODULE).LINK_STAGES=$(LINK_STAGES))
# A VCS build is <VCS_DIR>/<TOP_MODULE>_sim, run in VCS_DIR as Verilator's in VERILATOR_DIR; its key hashes every source and include file, as BUILD_KEY's dependency list does.
VCS_BIN = $(VCS_DIR)/$(TOP_MODULE)_sim
VCS_BUILD_KEY = { for f in $(sort $(DESIGN_FILES) $(TB_DIR)/top/$(TESTBENCH) $(wildcard $(TB_DIR)/top/*.svh $(SRC_DIR)/top/*.svh)); do md5sum "$$f"; done; \
	awk '!($$1 == "localparam" && $$3 ~ /^($(subst $(space),|,$(TB_KNOBS)))$$/)' $(PKG_FILE); \
	echo '$(TOP_MODULE) $(VCS_OPTS) $(VCS_FLAGS)'; }

default: help

help:
	@echo "=== SIENNA Hardware Simulation Makefile ==="
	@echo ""
	@echo "Precision (one variable for every build and run):"
	@echo "  FMT=fp32|bf16|int8      - number format (default fp32); any other value stops make"
	@echo "  N, TILE, LANES          - mesh size, tile size, activation lanes (default 16, 4, 32)"
	@echo "  COLLAPSE_K=1|0          - mesh COLLAPSE_K (default 1), for TB_sienna_top, the layer contract's model, gemm and tflite runs, and sm-verilator"
	@echo "  TEST=<name>             - make pkg / build test (default matmul_relu_nopool); pipeline: name filter"
	@echo "  make pkg                - Write testbenches/test_config_pkg.sv + that test's stimulus for FMT, N, TILE, TEST"
	@echo "  GEN_PKG=0               - Build the package already on disk (after gen-matmul/gen-conv)"
	@echo "  verilator, lint, debug, perf and vcs run make pkg first; the format is the define's (+define+FMT_<FMT>), never the package's"
	@echo "  FMT_DEF=                - Build with no precision define, which must elaborate fp32"
	@echo "  NOTE: so a default make verilator rewrites the tracked testbenches/test_config_pkg.sv; don't commit it"
	@echo ""
	@echo "Simulation Targets:"
	@echo "  make verilator           - Simulate using Verilator (no waveform)"
	@echo "  make sim                 - Run the last TB_sienna_top build on the stimulus in testbenches/, no build; refuses a build of other inputs or flags"
	@echo "  make build-key           - Print the key make sim compares with the build's (inputs, package without the run-time knobs, flags)"
	@echo "  make verilator TRACE=fst - Simulate + write FST waveform (recommended)"
	@echo "  make verilator TRACE=vcd - Simulate + write VCD waveform (larger)"
	@echo "  make vcs                 - Simulate using Synopsys VCS (no waveform); SIM=vcs runs every flow below on VCS (README: Running with VCS)"
	@echo "  make vcs       TRACE=vcd - Simulate + write VCD waveform"
	@echo "  make vcs       TRACE=fst - VCS has no native FST dump; still writes VCD"
	@echo ""
	@echo "Individual Module Targets:"
	@echo "  make sm-verilator      - Systolic Mesh only: its model checks, then its regression in FMT at N, TILE, COLLAPSE_K (SM_CHECKS=0 skips the checks)"
	@echo "  make gpnae-verilator   - GPNAE only: the lane sienna_top uses, every input of its gate in FMT against the true function"
	@echo "  make gpnae-verilator GPNAE_TIER=1 - the same at accuracy tier 1 (5%; 2 is 10%, 0 the default 1%)"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  make lint              - Run Verilator lint check"
	@echo "  make check-fmt         - no RTL outside the allowlist names a precision, and a planted EXP_W fails (C4)"
	@echo "  make check-hier        - each define's hierarchy, at SET_PORT 0 and 1, holds only that precision's units, the lanes' exactly"
	@echo "  make debug             - Build with GDB back-trace"
	@echo "  make perf              - Build with performance profiling"
	@echo ""
	@echo "Format-aware Run Targets (all take FMT, N, LANES, and TILE):"
	@echo "  make pipeline FMT=int8            - Full pipeline regression (TEST=<substring> narrows it)"
	@echo "  make model FMT=bf16 [MODEL_DIR=<d>] - MLPerf Tiny models on the RTL (model_runner.py; no int8; default models/)"
	@echo "  make gemm FMT=int8 [QUICK=1]      - GEMM shape sweep on sienna_top's layer contract (regression.py --action gemm)"
	@echo "  make perf-analysis FMT=bf16       - Cycle/latency/throughput report (regression.py --action perf)"
	@echo "  make tflite FMT=int8              - Single-layer TFLite int8 models, each output on the true function (model_runner.py --action tflite); needs FMT=int8"
	@echo "  make residual FMT=int8            - int8 residual layers and the residual-block TFLite models, each stage on the true function (regression.py --action residual); needs FMT=int8"
	@echo "  make pack FMT=int8                - packed layers vs each job alone (regression.py --action pack)"
	@echo "  make multi FMT=bf16 [COPIES=2]    - TB_sienna_multi on sienna_top at COPIES copies (regression.py --action multi)"
	@echo "  make idle-fault SET_PORT=0|1      - a put on the unused contract must fire a_idle_contract (regression.py --action idle-fault)"
	@echo "  make add-lanes                    - add_lanes on the true function of the ADD (int8 residual ADD)"
	@echo "  make rq-lanes                     - requant_lanes on the true function of the requantize (int8)"
	@echo "  make regression [MODEL_DIR=<d>]   - SIENNA's tests (pipeline, model, pack, gemm, tflite, residual) at N = 8, 16, 32, 64 x every tile size x fp32, bf16, int8, then one table (long: make -jN, N for your memory)"
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

# What a build compiles: the package make pkg writes for FMT, N, TILE and TEST, or with GEN_PKG=0 the one on disk; the format is the define's.
# Fails the build unless the package's N, TILE_SIZE, NUM_LANES and COLLAPSE_K are N, TILE, LANES and COLLAPSE_K (a stale package of another size).
pkg-geometry: $(if $(filter 0,$(GEN_PKG)),,pkg)
	@[ -r $(PKG_FILE) ] || { echo "ERROR: $(PKG_FILE) is missing or unreadable: run make pkg, or build without GEN_PKG=0."; exit 1; }
	@field() { awk -v k="$$1" '$$1 == "localparam" && $$3 == k { sub(/;.*/, "", $$5); print $$5 }' $(PKG_FILE) 2>/dev/null; }; \
	n=$$(field N); t=$$(field TILE_SIZE); l=$$(field NUM_LANES); c=$$(field COLLAPSE_K); \
	if [ "$$n" != "$(N)" ] || [ "$$t" != "$(TILE)" ] || [ "$$l" != "$(LANES)" ] || [ "$$c" != "$(COLLAPSE_K)" ]; then \
		echo "ERROR: $(PKG_FILE) has N=$${n:-missing} TILE_SIZE=$${t:-missing} NUM_LANES=$${l:-missing} COLLAPSE_K=$${c:-missing}, but the build asks for N=$(N) TILE=$(TILE) LANES=$(LANES) COLLAPSE_K=$(COLLAPSE_K)."; \
		echo "       The package was written for another geometry: run make pkg with these, or pass the package's."; \
		exit 1; \
	fi; \
	echo "-- test_config_pkg.sv matches N=$(N) TILE=$(TILE) LANES=$(LANES) COLLAPSE_K=$(COLLAPSE_K); the format is the define's ($(or $(FMT_DEF),no define))"
PKG_DEP = pkg-geometry

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

verilator: $(PKG_DEP)
	@echo "=== Verilator simulation: $(TOP_MODULE)  FMT=$(FMT)  TRACE=$(TRACE) ==="
	@mkdir -p $(VERILATOR_DIR)
	@rm -f $(VERILATOR_DIR)/build_key.txt
	$(VERILATOR) --binary \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/top/$(TESTBENCH) \
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
sim: $(PKG_DEP)
ifeq ($(SIM),vcs)
	@[ -x $(VCS_BIN) ] && [ -r $(VCS_BIN).build_key.txt ] || { echo "ERROR: no finished build in $(VCS_DIR): make vcs first"; exit 1; }
	@$(VCS_BUILD_KEY) | cmp -s - $(VCS_BIN).build_key.txt || { echo "ERROR: $(VCS_DIR) holds a build of other sources, package items or flags: make vcs"; exit 1; }
	$(call copy_mem_files,$(VCS_DIR))
	@echo "-- Running simulation (no build: $(VCS_DIR)'s inputs, package and flags are this run's)"
	cd $(VCS_DIR) && ./$(TOP_MODULE)_sim $(VCS_RUN) $(SIM_ARGS)
	@echo "-- Done"
else
	@[ -x $(VERILATOR_DIR)/$(TOP_MODULE)_sim ] && [ -r $(VERILATOR_DIR)/build_key.txt ] || { echo "ERROR: no finished build in $(VERILATOR_DIR): make verilator first"; exit 1; }
	@$(BUILD_KEY) | cmp -s - $(VERILATOR_DIR)/build_key.txt || { echo "ERROR: $(VERILATOR_DIR) holds a build of other sources, package items or flags: make verilator"; exit 1; }
	$(call copy_mem_files,$(VERILATOR_DIR))
	@echo "-- Running simulation (no build: $(VERILATOR_DIR)'s inputs, package and flags are this run's)"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim $(SIM_ARGS)
	@echo "-- Done"
endif

# The key make sim compares with the build's (BUILD_KEY), printed.
build-key:
	@$(if $(filter vcs,$(SIM)),$(VCS_BUILD_KEY),$(BUILD_KEY))

# ─────────────────────────────────────────────────────────────────────────────
# Sub-module targets
# ─────────────────────────────────────────────────────────────────────────────
# The mesh TB takes its format from SystolicMesh's FMT (its define) and its sizes and mesh mode as -G overrides.
# SM_CHECKS=0: without the model checks (aril-fpu, aril-narrow, model-tests); SystolicMesh's own make regression covers every N, tile size and format.
SM_CHECKS ?= 1
sm-verilator:
	@echo "=== Systolic Mesh only: FMT=$(FMT) N=$(N) TILE=$(TILE) COLLAPSE_K=$(COLLAPSE_K) ==="
	$(MAKE) -C SystolicMesh mesh-regression MATRIX_SIZE=$(N) SIM=$(SIM) \
		REGRESSION_OPTS="--format $(FMT) --collapse-k $(COLLAPSE_K) --tiles $(TILE)$(if $(filter 0,$(SM_CHECKS)), --no-checks)"

# GPNAE's lane gate at one tier of its coefficient memory: the RTL's words against the true function at that tier's tolerance.
GPNAE_TIER ?= 0
gpnae-verilator:
	@echo "=== GPNAE only: FMT=$(FMT), tier $(GPNAE_TIER) ==="
	cd GPNAE && $(PYTHON) regression.py --format $(FMT) --run gate --tier $(GPNAE_TIER) --sim $(SIM)

# ─────────────────────────────────────────────────────────────────────────────
# VCS
# ─────────────────────────────────────────────────────────────────────────────
vcs: $(PKG_DEP)
	@echo "=== VCS simulation: $(TOP_MODULE)  FMT=$(FMT)  TRACE=$(TRACE) ==="
ifeq ($(TRACE),fst)
	@echo "-- WARNING: VCS's \$$dumpvars only produces VCD; this run will write"
	@echo "            TB_sienna_top.vcd, not FST (need \$$fsdbDumpvars/Verdi for FST)."
endif
	@mkdir -p $(VCS_DIR)
	@rm -f $(VCS_BIN).build_key.txt
	$(VCS) $(VCS_OPTS) -top $(TOP_MODULE) -Mdir=$(VCS_DIR)/csrc_$(TOP_MODULE) -o $(VCS_BIN) $(VCS_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/top/$(TESTBENCH)
	@$(VCS_BUILD_KEY) > $(VCS_BIN).build_key.txt
	$(call copy_mem_files,$(VCS_DIR))
	@echo "-- Running simulation"
	cd $(VCS_DIR) && ./$(TOP_MODULE)_sim $(VCS_RUN) $(SIM_ARGS)
	@echo "-- Done"

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
lint: $(PKG_DEP)
	@echo "=== Linting $(TOP_MODULE)  FMT=$(FMT) ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --lint-only \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/top/$(TESTBENCH)
	@echo "-- Lint complete"

debug: VERILATOR_FLAGS += --debug --gdbbt
debug: verilator

perf: VERILATOR_FLAGS += --stats --profile-cfuncs
perf: verilator

# check-fmt: no RTL outside the allowlist names a precision (ArithmeticLibrary's check_fmt.sh); GPNAE and SystolicMesh run their own; C4: a planted EXP_W must fail it.
FMT_ALLOW = ^(GPNAE|SystolicMesh)/
check-fmt:
	@bash $(SM_LIB_DIR)/Common/check_fmt.sh $(PRJ_DIR) '$(FMT_ALLOW)' && echo "check-fmt OK: the tree passes" || { echo "check-fmt FAIL: the tree"; exit 1; }
	@t=$$(mktemp -d); mkdir -p $$t/src/pipeline; cp $(SRC_DIR)/pipeline/sienna_pipe.sv $$t/src/pipeline/; \
	sed -i 's/^endmodule/  localparam int EXP_W = 8;\nendmodule/' $$t/src/pipeline/sienna_pipe.sv; \
	if bash $(SM_LIB_DIR)/Common/check_fmt.sh $$t '$(FMT_ALLOW)' > $$t/c4.log; then echo "check-fmt FAIL: C4 planted EXP_W passed"; exit 1; \
	elif ! /usr/bin/grep -q '^check-fmt: 1 RTL files outside the allowlist, 1 naming a precision$$' $$t/c4.log; then echo "check-fmt FAIL: C4 failed for another reason"; cat $$t/c4.log; exit 1; \
	else echo "check-fmt OK: C4 planted EXP_W in sienna_pipe.sv fails"; fi; rm -rf $$t

# check-hier: at fp32, bf16, int8 and no define, SET_PORT 1 (TB_sienna_top) and 0 (TB_model_run), Verilator's elaborated hierarchy holds only that precision's units (check_hier.py), the lanes' set exactly.
check-hier: $(PKG_DEP)
	@$(PYTHON) $(PRJ_DIR)/scripts/check_hier.py --verilator $(VERILATOR) --out $(PRJ_DIR)/Verilator_hier \
	  --inc $(SRC_DIR)/top $(TB_DIR)/top $(SM_DIR) $(SM_LIB_DIR) $(GPNAE_DIR) $(MAXPOOL_DIR) $(DROPOUT_DIR) --files $(DESIGN_FILES) $(PKG_FILE)

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
	@[ -n "$(MODEL_DIR)" ] || { echo "ERROR: make model needs MODEL_DIR=<directory of the .tflite models> (its default is models/)"; exit 1; }
	$(PYTHON) scripts/model_runner.py \
		--model-dir $(MODEL_DIR) \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--format $(FMT) \
		--collapse-k $(COLLAPSE_K) \
		--count $(MODEL_COUNT)

# GEMM shape sweep on sienna_top's layer contract, built at N, TILE and LANES.
gemm:
	$(PYTHON) regression.py --action gemm \
		--n $(N) \
		--tile $(TILE) \
		--lanes $(LANES) \
		--format $(FMT) \
		--collapse-k $(COLLAPSE_K) \
		$(if $(filter 1,$(QUICK)),--quick)

perf-analysis:
	$(PYTHON) regression.py --action perf \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--format $(FMT)

tflite:
	@[ "$(FMT)" = int8 ] || { echo "ERROR: make tflite runs the int8 TFLite models only and needs FMT=int8 (FMT=$(FMT))"; exit 1; }
	$(PYTHON) scripts/model_runner.py --action tflite \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--collapse-k $(COLLAPSE_K)
	@# the packed TFLite layers too, where a block of 8 fits N / 2 (N >= 16); the collapse-k 0 mesh refuses packed sets
	$(if $(shell [ $(N) -ge 16 ] && [ $(COLLAPSE_K) = 1 ] && echo y),$(PYTHON) scripts/model_runner.py --action tflite --pack --n $(N) --tile-size $(TILE) --lanes $(LANES))

# int8 residual layers and the residual-block TFLite models on TB_model_run (regression.py --action residual); int8 only
residual:
	@[ "$(FMT)" = int8 ] || { echo "ERROR: make residual runs the int8 residual layers only and needs FMT=int8 (FMT=$(FMT))"; exit 1; }
	$(PYTHON) regression.py --action residual --n $(N) --tile $(TILE) --lanes $(LANES) --tflite --mid-reset

# A put on the unused contract must fire a_idle_contract: SET_PORT=0 TB_model_run FAULT 4, SET_PORT=1 TB_sienna_top FAULT 8 (regression.py --action idle-fault).
idle-fault:
	@case "$(SET_PORT)" in 0|1) ;; *) echo "ERROR: make idle-fault needs SET_PORT=0 or SET_PORT=1 (SET_PORT=$(SET_PORT))"; exit 1;; esac
	$(PYTHON) regression.py --action idle-fault --set-port $(SET_PORT)

# TB_sienna_multi on sienna_top at COPIES copies (SET_PORT = 1): one pipeline test's sets in FMT at N, TILE and LANES (regression.py --action multi).
COPIES ?= 2
multi:
	$(PYTHON) regression.py --action multi --format $(FMT) --n $(N) --tile $(TILE) --lanes $(LANES) --copies $(COPIES)$(if $(TEST), --test $(TEST))

# Packed layers against each job alone on sienna_top's layer contract (regression.py --action pack), in FMT at N, TILE and LANES.
pack:
	$(PYTHON) regression.py --action pack --format $(FMT) --n $(N) --tile $(TILE) --lanes $(LANES)

# add_lanes on regression.py --action add-vectors: fails unless the log has RESULT: PASSED and no error or failed assertion (patterns bracketed against make's echo), then gates its outputs on the true function
ADDL_DIR = $(if $(filter vcs,$(SIM)),$(VCS_DIR),$(VERILATOR_DIR))/add_lanes
add-lanes:
	mkdir -p $(ADDL_DIR)
	$(PYTHON) regression.py --action add-vectors $(TB_DIR)/add_lanes.mem --truth $(ADDL_DIR)/add_lanes_truth.json
ifeq ($(SIM),vcs)
	$(call vcs_build,TB_add_lanes,$(ADDL_DIR),$(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/QuantizedAdder/src/int8QuantizedAdder.sv $(SRC_DIR)/lanes/add_lanes.sv $(TB_DIR)/units/TB_add_lanes.sv,+define+FMT_INT8)
else
	$(VERILATOR) --binary --timing --assert --sv -j $(shell nproc) $(filter --Wno-%,$(VERILATOR_FLAGS)) -Werror-USERFATAL +define+FMT_INT8 --top-module TB_add_lanes --Mdir $(ADDL_DIR) \
	  $(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/QuantizedAdder/src/int8QuantizedAdder.sv $(SRC_DIR)/lanes/add_lanes.sv \
	  $(TB_DIR)/units/TB_add_lanes.sv -o sim
endif
	set -o pipefail; cd $(TB_DIR) && $(ADDL_DIR)/sim $(SIM_RUN) | tee $(ADDL_DIR)/sim.log && command grep -q '^RESULT: PASSED' $(ADDL_DIR)/sim.log && \
	  ! command grep -qE '%[E]rror|%[F]atal|\[[F]ATAL\]|Assertion [f]ailed|^([E]rror|[F]atal): "|^[E]rror-\[' $(ADDL_DIR)/sim.log || { echo "TB_add_lanes: failed ($(ADDL_DIR)/sim.log)"; exit 1; }
	$(PYTHON) regression.py --action add-vectors --check $(TB_DIR)/add_lanes_out.mem --truth $(ADDL_DIR)/add_lanes_truth.json

# requant_lanes on regression.py --action rq-vectors: fails unless the log has RESULT: PASSED and no error or failed assertion (patterns bracketed against make's echo), then gates its outputs on the true function
RQL_DIR = $(if $(filter vcs,$(SIM)),$(VCS_DIR),$(VERILATOR_DIR))/rq_lanes
rq-lanes:
	mkdir -p $(RQL_DIR)
	$(PYTHON) regression.py --action rq-vectors $(TB_DIR)/rq_lanes.mem --truth $(RQL_DIR)/rq_lanes_truth.json
ifeq ($(SIM),vcs)
	$(call vcs_build,TB_requant_lanes,$(RQL_DIR),$(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/Requantizer/src/int8Requantizer.sv $(SRC_DIR)/lanes/requant_lanes.sv $(TB_DIR)/units/TB_requant_lanes.sv,+define+FMT_INT8)
else
	$(VERILATOR) --binary --timing --assert --sv -j $(shell nproc) $(filter --Wno-%,$(VERILATOR_FLAGS)) -Werror-USERFATAL +define+FMT_INT8 --top-module TB_requant_lanes --Mdir $(RQL_DIR) \
	  $(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/Requantizer/src/int8Requantizer.sv $(SRC_DIR)/lanes/requant_lanes.sv $(TB_DIR)/units/TB_requant_lanes.sv -o sim
endif
	set -o pipefail; cd $(TB_DIR) && $(RQL_DIR)/sim $(SIM_RUN) | tee $(RQL_DIR)/sim.log && command grep -q '^RESULT: PASSED' $(RQL_DIR)/sim.log && \
	  ! command grep -qE '%[E]rror|%[F]atal|\[[F]ATAL\]|Assertion [f]ailed|^([E]rror|[F]atal): "|^[E]rror-\[' $(RQL_DIR)/sim.log || { echo "TB_requant_lanes: failed ($(RQL_DIR)/sim.log)"; exit 1; }
	$(PYTHON) regression.py --action rq-vectors --check $(TB_DIR)/rq_lanes_out.mem --truth $(RQL_DIR)/rq_lanes_truth.json

# make regression: SIENNA's points, each its own target (make -jN runs N side by side), then the table; FMT=<f> narrows it to one format.
comma := ,
empty :=
space := $(empty) $(empty)
REG_FORMATS = $(subst $(space),$(comma),$(strip $(or $(REG_FMT),$(FORMATS))))
REG_SELECT  = --formats $(REG_FORMATS)$(if $(filter 1,$(REG_SMALL)), --small)
REG_DIR    ?= $(TB_DIR)/results/regression
REG_SUMMARY = $(PYTHON) regression.py --action summary --results $(REG_DIR) $(REG_SELECT) --model-dir=$(abspath $(MODEL_DIR)) --sim $(SIM)
ifneq ($(filter regression,$(MAKECMDGOALS)),)
ifeq ($(origin REG_RUN),undefined)
REG_RUN := $(shell date +%Y%m%d-%H%M%S)-$(shell echo $$PPID)
endif
REG_POINTS := $(shell $(PYTHON) regression.py --action points $(REG_SELECT))
endif

# One point, e.g. make regression/bf16/N8_T2/pipeline: runs in a private copy of the tree, writes $(REG_DIR)/<point>.log and .json.
regression/%:
	@$(PYTHON) regression.py --action point --name $* --results $(REG_DIR) --model-dir=$(abspath $(MODEL_DIR)) --sim $(SIM)$(if $(REG_RUN), --run $(REG_RUN))$(if $(filter regression,$(MAKECMDGOALS)), --keep-going)

# The default point compares with its twin's finished result, so its twin runs first.
regression/default/N8_T2/pipeline: | regression/fp32/N8_T2/pipeline

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
        gen-matmul gen-conv pipeline pkg pkg-geometry model gemm perf-analysis tflite residual pack multi idle-fault check-fmt check-hier add-lanes rq-lanes regression regression-summary sim build-key
