SHELL := /bin/bash

# =========================================================================
# SIENNA VERIFICATION MAKEFILE CONFIGURATION
# =========================================================================
N    ?= 16
TILE ?= 4

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

TOP_FILES = \
	sienna_top.sv

SM_FILES = \
	top/SystolicMesh.sv \
	top/SystolicArray.sv \
	mem/RowInputQueue.sv \
	mem/ColumnInputQueue.sv \
	mem/OutputSram.sv \
	mem/MeshOutputSram.sv \
	engine/PEMesh.sv \
	engine/ProcessingElement.sv \
	engine/AccumulationUnit.sv \
	engine/MAC.sv

SM_LIB_FILES = \
	Multipliers/Radix4Booth/src/R4Booth.sv \
	Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
	Multipliers/FP32/src/fp32Multiplier.sv \
	Adders/FP32/src/LZC.sv \
	Adders/FP32/src/fp32Adder.sv

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
TESTBENCH  = TB_sienna_top.sv
TOP_MODULE = TB_sienna_top

VERILATOR_DIR = $(PRJ_DIR)/Verilator
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

# Hook for one-off defines, e.g. make verilator EXTRA_FLAGS=-DBACK_TO_BACK
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
	@echo "Simulation Targets:"
	@echo "  make verilator           - Simulate using Verilator (no waveform)"
	@echo "  make verilator TRACE=fst - Simulate + write FST waveform (recommended)"
	@echo "  make verilator TRACE=vcd - Simulate + write VCD waveform (larger)"
	@echo "  make vcs                 - Simulate using Synopsys VCS (no waveform)"
	@echo "  make vcs       TRACE=vcd - Simulate + write VCD waveform"
	@echo "  make vcs       TRACE=fst - VCS has no native FST dump; still writes VCD"
	@echo ""
	@echo "Individual Module Targets:"
	@echo "  make sm-verilator      - Simulate Systolic Mesh only"
	@echo "  make gpnae-verilator   - Simulate GPNAE only"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  make lint              - Run Verilator lint check"
	@echo "  make debug             - Build with GDB back-trace"
	@echo "  make perf              - Build with performance profiling"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make wave              - View waveforms (requires prior TRACE=vcd|fst run)"
	@echo "  make gen-matmul        - Generate matmul stimulus (N=16 default)"
	@echo "  make gen-conv          - Generate conv stimulus   (N=16 default)"
	@echo "  make clean             - Remove all simulation artifacts"
	@echo "  make clean-all         - Clean all including subprojects"
	@echo ""
	@echo "Disk usage note:"
	@echo "  TRACE=0   (default) → no waveform written; Verilator dir stays small."
	@echo "  TRACE=fst           → compact waveform, recommended for this project."
	@echo "  TRACE=vcd           → can be very large for complex designs."
	@echo ""
	@echo "Current Configuration:"
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
	@echo "=== Generating matmul stimulus (N=$(N), tile=$(TILE)) ==="
	python3 regression.py \
		--action gen \
		--mode matmul \
		--n $(N) \
		--tile-size $(TILE)

gen-conv:
	@echo "=== Generating conv stimulus (N=$(N), tile=$(TILE)) ==="
	python3 regression.py \
		--action gen \
		--mode conv \
		--conv-type basic \
		--n $(N) \
		--tile-size $(TILE)

# ─────────────────────────────────────────────────────────────────────────────
# Verilator — SIENNA Top
# ─────────────────────────────────────────────────────────────────────────────
verilator:
	@echo "=== Verilator simulation: $(TOP_MODULE)  TRACE=$(TRACE) ==="
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
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- Done"
ifneq ($(TRACE),0)
	@echo "-- Trace ($(TRACE)) : $(VERILATOR_DIR)/$(TRACE_FILE)"
endif

# ─────────────────────────────────────────────────────────────────────────────
# Sub-module targets
# ─────────────────────────────────────────────────────────────────────────────
sm-verilator:
	@echo "=== Building Systolic Mesh only ==="
	$(MAKE) -C SystolicMesh verilator

gpnae-verilator:
	@echo "=== Building GPNAE only ==="
	$(MAKE) -C GPNAE verilator

# ─────────────────────────────────────────────────────────────────────────────
# VCS
# ─────────────────────────────────────────────────────────────────────────────
vcs:
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
lint:
	@echo "=== Linting $(TOP_MODULE) ==="
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
	@echo "=== Running Sienna Pipeline Regression ==="
	python3 regression.py \
		--matrix-size $(N) \
		--tile-size $(TILE) \
		$(if $(TEST),--test $(TEST))

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
        gen-matmul gen-conv regression
