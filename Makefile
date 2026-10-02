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
QUICK ?= 0

FORMATS = fp32 bf16 int8
ifneq ($(filter-out $(FORMATS),$(FMT))$(words $(FMT)),1)
$(error Invalid FMT=$(FMT): must be one of fp32, bf16, int8)
endif

# COLLAPSE_K is 0 or 1, and 0 only for sm-verilator: TB_sienna_top has no parameters, so -G cannot reach sienna_top's.
ifneq ($(filter-out 0 1,$(COLLAPSE_K))$(words $(COLLAPSE_K)),1)
$(error Invalid COLLAPSE_K=$(COLLAPSE_K): must be 0 or 1)
endif
ifeq ($(COLLAPSE_K),0)
ifneq ($(filter-out sm-verilator help default,$(or $(MAKECMDGOALS),default)),)
$(error COLLAPSE_K=0: TB_sienna_top builds sienna_top's default collapse-k 1; a collapse-k 0 SIENNA build uses $$J/cmds/sienna_ck0.sh)
endif
endif

# EXP_W MAN_W IS_INT of each FMT, as regression.py's FORMATS writes them into test_config_pkg.sv.
FMT_FIELDS_fp32 = 8 23 0
FMT_FIELDS_bf16 = 8 7 0
FMT_FIELDS_int8 = 0 7 1

# Project Structure
PRJ_DIR     = $(shell pwd)
SRC_DIR     = $(PRJ_DIR)/src
TB_DIR      = $(PRJ_DIR)/testbenches
SM_DIR      = $(PRJ_DIR)/SystolicMesh/src
SM_LIB_DIR  = $(PRJ_DIR)/SystolicMesh/ArithmeticLibrary
GPNAE_DIR   = $(PRJ_DIR)/GPNAE/src
GPNAE_LIB_DIR = $(PRJ_DIR)/GPNAE/ArithmeticLibrary
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
	Multipliers/Int/src/intMultiplier.sv \
	Adders/FP32/src/LZC.sv \
	Adders/FP32/src/fp32Adder.sv \
	Adders/FP/src/fpAdder.sv \
	Adders/Int/src/intAdder.sv \
	Multipliers/Fx/src/fxMac.sv \
	Requant/src/tfliteRequant.sv

GPNAE_FILES = \
	TYTAN/Memory/CoeffROM.v \
	TYTAN/Memory/InputFIFO.v \
	TYTAN/Memory/PE5B.v \
	TYTAN/Memory/RAM.v \
	TYTAN/Memory/ROM.v \
	TYTAN/LZC.v \
	TYTAN/barrel_mac.sv \
	fp32_down.sv \
	fp32_up_down.sv \
	SeLu.sv \
	sigtan.sv \
	gpnae.sv \
	gpnae_tail.sv \
	gpnae_poly_int8.sv \
	gpnae_poly.sv

GPNAE_LIB_FILES = \
	Adders/FP32/src/fp32Adder.sv \
	Adders/FP32/src/LZC.sv \
	Multipliers/Radix4Booth/src/R4Booth.sv \
	Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
	Multipliers/FP32/src/fp32Multiplier.sv \
	Divider/FP32/src/fp32Divider.sv \
	Divider/FP32/src/divu.sv

MAXPOOL_FILES = \
	Maxpool_2D.sv

DROPOUT_FILES = \
	dropout.sv

DESIGN_FILES = \
	$(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv \
	$(addprefix $(SRC_DIR)/,$(TOP_FILES)) \
	$(addprefix $(SM_DIR)/,$(SM_FILES)) \
	$(addprefix $(SM_LIB_DIR)/,$(SM_LIB_FILES)) \
	$(addprefix $(GPNAE_DIR)/,$(GPNAE_FILES)) \
	$(addprefix $(GPNAE_LIB_DIR)/,$(GPNAE_LIB_FILES)) \
	$(addprefix $(MAXPOOL_DIR)/,$(MAXPOOL_FILES)) \
	$(addprefix $(DROPOUT_DIR)/,$(DROPOUT_FILES))

# Testbench
# TB_PKG_FILES compiles before TESTBENCH: a package must be declared before it is imported.
TB_PKG_FILES = test_config_pkg.sv
PKG_FILE     = $(TB_DIR)/test_config_pkg.sv
TESTBENCH  = TB_sienna_top.sv
TOP_MODULE = TB_sienna_top

VERILATOR_DIR = $(PRJ_DIR)/Verilator

# Very large meshes put thousands of PEs into a few generated functions that -Os takes hours on; OPT_FAST=-O0 builds in minutes.
OPT_FAST ?= -Os
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
				cp -u $$d/$$p $(1)/; \
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
	--threads $(shell nproc) \
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
	-I$(GPNAE_LIB_DIR) \
	-I$(MAXPOOL_DIR) \
	-I$(DROPOUT_DIR) \
	--Mdir $(VERILATOR_DIR) \
	--Wno-WIDTHTRUNC \
	--Wno-WIDTHEXPAND \
	--Wno-WIDTHCONCAT \
	--Wno-CASEINCOMPLETE \
	--Wno-MODDUP \
	--Wno-SELRANGE \
	--Wno-LATCH \
	--Wno-REALCVT \
	--Wno-SHORTREAL \
	--Wno-TIMESCALEMOD \
	--Wno-UNSIGNED

# Hook for one-off defines, e.g. make verilator EXTRA_FLAGS=-DBACK_TO_BACK; SIM_ARGS go to the simulator (e.g. +verilator+rand+reset+2)
VERILATOR_FLAGS += $(EXTRA_FLAGS)

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
	+incdir+$(GPNAE_LIB_DIR) \
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
	@echo "  COLLAPSE_K=1|0          - mesh COLLAPSE_K (default 1); 0 only for sm-verilator, as TB_sienna_top builds 1"
	@echo "  TEST=<name>             - make pkg / build test (default matmul_relu_nopool); regression: name filter"
	@echo "  make pkg                - Write testbenches/test_config_pkg.sv + that test's stimulus for FMT, N, TILE, TEST"
	@echo "  GEN_PKG=0               - Build the package already on disk (after gen-matmul/gen-conv); still guarded"
	@echo "  verilator, lint, debug and perf run make pkg first, then check its EXP_W/MAN_W/IS_INT against FMT"
	@echo ""
	@echo "Simulation Targets:"
	@echo "  make verilator           - Simulate using Verilator (no waveform)"
	@echo "  make verilator TRACE=fst - Simulate + write FST waveform (recommended)"
	@echo "  make verilator TRACE=vcd - Simulate + write VCD waveform (larger)"
	@echo "  make vcs                 - Simulate using Synopsys VCS (no waveform)"
	@echo "  make vcs       TRACE=vcd - Simulate + write VCD waveform"
	@echo "  make vcs       TRACE=fst - VCS has no native FST dump; still writes VCD"
	@echo ""
	@echo "Individual Module Targets:"
	@echo "  make sm-verilator      - Systolic Mesh only: its regression in FMT at N, TILE, COLLAPSE_K"
	@echo "  make gpnae-verilator   - GPNAE only: its regression in FMT on gpnae_poly, the lane sienna_top uses"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  make lint              - Run Verilator lint check"
	@echo "  make debug             - Build with GDB back-trace"
	@echo "  make perf              - Build with performance profiling"
	@echo ""
	@echo "Format-aware Run Targets (all take FMT, N, LANES, and TILE except gemm):"
	@echo "  make regression FMT=int8          - Full pipeline regression (TEST=<substring> narrows it)"
	@echo "  make model FMT=bf16 MODEL_DIR=<d> - MLPerf Tiny models on the RTL (model_runner.py; no int8)"
	@echo "  make gemm FMT=int8 [QUICK=1]      - GEMM shape sweep on sienna_layer (gemm_sweep.py; T=4)"
	@echo "  make perf-analysis FMT=bf16       - Cycle/latency/throughput report (perf_analysis.py)"
	@echo "  make tflite FMT=int8              - Single-layer TFLite int8 models, bit for bit; needs FMT=int8"
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
	@echo "  GPNAE         : $(words $(GPNAE_FILES)) files  |  GPNAE Lib: $(words $(GPNAE_LIB_FILES)) files"
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

# Fails the build unless the package's EXP_W, MAN_W, IS_INT (when present), N, TILE_SIZE and NUM_LANES are FMT's, N, TILE and LANES.
pkg-check: $(if $(filter 0,$(GEN_PKG)),,pkg)
	@set -- $(FMT_FIELDS_$(FMT)); \
	field() { awk -v k="$$1" '$$1 == "localparam" && $$3 == k { sub(/;.*/, "", $$5); print $$5 }' $(PKG_FILE) 2>/dev/null; }; \
	e=$$(field EXP_W); m=$$(field MAN_W); i=$$(field IS_INT); n=$$(field N); t=$$(field TILE_SIZE); l=$$(field NUM_LANES); \
	if [ "$$e" != "$$1" ] || [ "$$m" != "$$2" ] || { [ -n "$$i" ] && [ "$$i" != "$$3" ]; }; then \
		echo "ERROR: $(PKG_FILE) has EXP_W=$${e:-missing} MAN_W=$${m:-missing} IS_INT=$${i:-absent}, but FMT=$(FMT) needs EXP_W=$$1 MAN_W=$$2 IS_INT=$$3."; \
		echo "       The package is stale or for another format: run make pkg FMT=$(FMT), or build without GEN_PKG=0."; \
		exit 1; \
	fi; \
	if [ "$$n" != "$(N)" ] || [ "$$t" != "$(TILE)" ] || [ "$$l" != "$(LANES)" ]; then \
		echo "ERROR: $(PKG_FILE) has N=$${n:-missing} TILE_SIZE=$${t:-missing} NUM_LANES=$${l:-missing}, but the build asks for N=$(N) TILE=$(TILE) LANES=$(LANES)."; \
		echo "       The package was written for another geometry: run make pkg with these N, TILE, LANES, or pass the package's."; \
		exit 1; \
	fi; \
	echo "-- test_config_pkg.sv matches FMT=$(FMT) N=$(N) TILE=$(TILE) LANES=$(LANES): EXP_W=$$e MAN_W=$$m IS_INT=$${i:-absent}"

# ─────────────────────────────────────────────────────────────────────────────
# Verilator — SIENNA Top
# ─────────────────────────────────────────────────────────────────────────────
verilator: pkg-check
	@echo "=== Verilator simulation: $(TOP_MODULE)  FMT=$(FMT)  TRACE=$(TRACE) ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --binary \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(addprefix $(TB_DIR)/,$(TB_PKG_FILES)) \
		$(TB_DIR)/$(TESTBENCH) \
		-o $(TOP_MODULE)_sim
	@echo "-- Compiling Verilator C++ model"
	$(MAKE) -C $(VERILATOR_DIR) -f V$(TOP_MODULE).mk
	$(call copy_mem_files,$(VERILATOR_DIR))
	@echo "-- Running simulation"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim $(SIM_ARGS)
	@echo "-- Done"
ifneq ($(TRACE),0)
	@echo "-- Trace ($(TRACE)) : $(VERILATOR_DIR)/$(TRACE_FILE)"
endif

# ─────────────────────────────────────────────────────────────────────────────
# Sub-module targets
# ─────────────────────────────────────────────────────────────────────────────
# The mesh TB takes its format only from SystolicMesh's regression, which patches EXP_W/MAN_W/COLLAPSE_K into it.
sm-verilator:
	@echo "=== Systolic Mesh only: FMT=$(FMT) N=$(N) TILE=$(TILE) COLLAPSE_K=$(COLLAPSE_K) ==="
	$(MAKE) -C SystolicMesh regression MATRIX_SIZE=$(N) \
		REGRESSION_OPTS="--format $(FMT) --collapse-k $(COLLAPSE_K) --tiles $(TILE)"

# GPNAE's Makefile takes no format: its regression writes gpnae_test_config.svh for FMT, then runs make verilator.
gpnae-verilator:
	@echo "=== GPNAE only: FMT=$(FMT), gpnae_poly lane ==="
	cd GPNAE && $(PYTHON) regression.py --lane poly --format $(FMT)

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
	@echo "GPNAE ArithmeticLibrary ($(GPNAE_LIB_DIR)):"; \
	for f in $(GPNAE_LIB_FILES); do echo "  - $$f"; done
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

regression:
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
		--format $(FMT)

# gemm_sweep.py has no --tile-size: it builds sienna_layer at T=4.
gemm:
	$(PYTHON) gemm_sweep.py \
		--n $(N) \
		--lanes $(LANES) \
		--format $(FMT) \
		$(if $(filter 1,$(QUICK)),--quick)

perf-analysis:
	$(PYTHON) perf_analysis.py \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES) \
		--format $(FMT)

tflite:
	@[ "$(FMT)" = int8 ] || { echo "ERROR: make tflite runs the int8 TFLite models only and needs FMT=int8 (FMT=$(FMT))"; exit 1; }
	$(PYTHON) tflite_int8_run.py \
		--n $(N) \
		--tile-size $(TILE) \
		--lanes $(LANES)

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
        gen-matmul gen-conv regression pkg pkg-check model gemm perf-analysis tflite
