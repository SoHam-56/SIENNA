# Project Structure
PRJ_DIR = $(shell pwd)
SRC_DIR = $(PRJ_DIR)/src
TB_DIR = $(PRJ_DIR)/testbenches
SA_DIR = $(PRJ_DIR)/SystolicArray/src
GNAE_DIR = $(PRJ_DIR)/GNAE/src
SUMIT_DIR = $(PRJ_DIR)/Sumit_Anish

# Toolchain
VERILATOR = verilator
VCS = vcs
WAVE = surfer

# SIENNA Top Level
TOP_FILES = \
	sienna_top.sv

# Systolic Array Design Files
SA_FILES = \
	SystolicArray.sv \
	RowInputQueue.sv \
	ColumnInputQueue.sv \
	Mesh.sv \
	OutputSram.sv \
	ProcessingElement.sv \
	MAC/Adder_FP32.sv \
	MAC/LZC.sv \
	MAC/Multiplier_FP32.sv \
	MAC/UnSig_Karatsuba.sv \
	MAC/UnSig_R4Booth.sv \
	MAC/MAC.sv

# GNAE Design Files
GNAE_FILES = \
	gpnae.sv \
	UpDown.sv \
	Down.sv \
	SeLu.sv \
	sigtan.sv \
	Divider/Divider_FP32.sv \
	Divider/divu.sv \
	TYTAN/Adder_FP32.v \
	TYTAN/controller.sv \
	TYTAN/datapath.v \
	TYTAN/LZC.v \
	TYTAN/MAC.sv \
	TYTAN/Multiplier_FP32.sv \
	TYTAN/UnSig_Karatsuba.sv \
	TYTAN/UnSig_R4Booth.v \
	TYTAN/Memory/CoeffROM.v \
	TYTAN/Memory/InputFIFO.v \
	TYTAN/Memory/PE5B.v \
	TYTAN/Memory/RAM.v \
	TYTAN/Memory/ROM.v

# Sumit_Anish Design Files (CNN Components)
SUMIT_FILES = \
	CNN2/dropout_module.v \
	CNN2/Maxpool_2D.v \
	CNN2/SRAM.v

# FP16 Converter/Deconverter Files (if needed)
# Uncomment if these are required by the top module
# FP16_FILES = \
# 	FP16_Converter/topModule.v \
# 	FP16_Deconverter/topModule.v

# Combined Design Files
DESIGN_FILES = \
	$(addprefix $(SRC_DIR)/,$(TOP_FILES)) \
	$(addprefix $(SA_DIR)/,$(SA_FILES)) \
	$(addprefix $(GNAE_DIR)/,$(GNAE_FILES)) \
	$(addprefix $(SUMIT_DIR)/,$(SUMIT_FILES))

# Testbench Configuration
TESTBENCH = TB_sienna_top.sv
TOP_MODULE = TB_sienna_top

# Alternative testbench configurations:
# For Systolic Array only:
# TESTBENCH = TB_SystolicArray.sv
# TOP_MODULE = TB_SystolicArray
# TB_DIR = $(PRJ_DIR)/SystolicArray/testbenches

# For GNAE only:
# TESTBENCH = TB_gpnae.sv
# TOP_MODULE = TB_gpnae
# TB_DIR = $(PRJ_DIR)/GNAE/testbenches

# Directories
VERILATOR_DIR = $(PRJ_DIR)/Verilator
VCS_DIR = $(PRJ_DIR)/VCS

# Verilator Flags
VERILATOR_FLAGS = \
	--trace \
	--timing \
	--top-module $(TOP_MODULE) \
	--threads $(shell nproc) \
	--sv \
	-I$(SRC_DIR) \
	-I$(TB_DIR) \
	-I$(SA_DIR) \
	-I$(GNAE_DIR) \
	-I$(SUMIT_DIR)/CNN2 \
	-I$(SUMIT_DIR)/FP16_Converter \
	-I$(SUMIT_DIR)/FP16_Deconverter \
	--Mdir $(VERILATOR_DIR) \
	--Wno-WIDTHTRUNC \
	--Wno-WIDTHEXPAND \
	--Wno-WIDTHCONCAT \
	--Wno-CASEINCOMPLETE \
	--Wno-UNOPTFLAT \
	--Wno-BLKANDNBLK

# VCS Flags
VCS_FLAGS = \
	-full64 \
	-sverilog \
	-debug_all \
	-timescale=1ns/1ps \
	-Mdir=$(VCS_DIR) \
	+v2k \
	+incdir+$(SRC_DIR) \
	+incdir+$(TB_DIR) \
	+incdir+$(SA_DIR) \
	+incdir+$(GNAE_DIR) \
	+incdir+$(SUMIT_DIR)/CNN2 \
	+incdir+$(SUMIT_DIR)/FP16_Converter \
	+incdir+$(SUMIT_DIR)/FP16_Deconverter \
	+define+VCS

# Default target
default: help

# Help message
help:
	@echo "=== SIENNA Hardware Simulation Makefile ==="
	@echo ""
	@echo "Simulation Targets:"
	@echo "  make verilator    - Simulate using Verilator"
	@echo "  make vcs          - Simulate using Synopsys VCS"
	@echo ""
	@echo "Individual Module Targets:"
	@echo "  make sa-verilator - Simulate Systolic Array only"
	@echo "  make gnae-verilator - Simulate GNAE only"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  make lint         - Run Verilator lint check"
	@echo "  make debug        - Build with debug information"
	@echo "  make perf         - Build with performance profiling"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make wave         - View waveforms"
	@echo "  make clean        - Remove all simulation artifacts"
	@echo "  make clean-all    - Clean all including subprojects"
	@echo ""
	@echo "Current Configuration:"
	@echo "  TOP_MODULE: $(TOP_MODULE)"
	@echo "  TESTBENCH:  $(TESTBENCH)"
	@echo ""
	@echo "Design Files:"
	@echo "  Systolic Array: $(words $(SA_FILES)) files"
	@echo "  GNAE:           $(words $(GNAE_FILES)) files"
	@echo "  CNN (Sumit):    $(words $(SUMIT_FILES)) files"
	@echo "  Total:          $(words $(DESIGN_FILES)) files"

# Verilator Simulation - SIENNA Top
verilator:
	@echo "=== Verilator simulation for $(TOP_MODULE) ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --binary \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(TB_DIR)/$(TESTBENCH) \
		-o $(TOP_MODULE)_sim
	@echo "-- Compiling Verilator simulation"
	make -C $(VERILATOR_DIR) -f V$(TOP_MODULE).mk
	@echo "-- Copying memory files"
	@if ls $(SA_DIR)/*.mem 1> /dev/null 2>&1; then \
		cp $(SA_DIR)/*.mem $(VERILATOR_DIR)/; \
		echo "   Copied Systolic Array .mem files"; \
	fi
	@if ls $(GNAE_DIR)/TYTAN/Memory/*.mem 1> /dev/null 2>&1; then \
		cp $(GNAE_DIR)/TYTAN/Memory/*.mem $(VERILATOR_DIR)/; \
		echo "   Copied GNAE/TYTAN .mem files"; \
	fi
	@if ls $(TB_DIR)/*.mem 1> /dev/null 2>&1; then \
		cp $(TB_DIR)/*.mem $(VERILATOR_DIR)/; \
		echo "   Copied testbench .mem files"; \
	fi
	@echo "-- Running Verilator simulation"
	cd $(VERILATOR_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- Verilator simulation complete"
	@echo "-- Trace file: $(VERILATOR_DIR)/dump.vcd"

# Verilator Simulation - Systolic Array Only
sa-verilator:
	@echo "=== Building Systolic Array only ==="
	$(MAKE) -C SystolicArray verilator

# Verilator Simulation - GNAE Only
gnae-verilator:
	@echo "=== Building GNAE only ==="
	$(MAKE) -C GNAE verilator

# VCS Simulation
vcs:
	@echo "=== VCS simulation for $(TOP_MODULE) ==="
	@mkdir -p $(VCS_DIR)
	$(VCS) $(VCS_FLAGS) \
		-o $(VCS_DIR)/$(TOP_MODULE)_sim \
		$(DESIGN_FILES) \
		$(TB_DIR)/$(TESTBENCH)
	@echo "-- Copying memory files"
	@if ls $(SA_DIR)/*.mem 1> /dev/null 2>&1; then \
		cp $(SA_DIR)/*.mem $(VCS_DIR)/; \
		echo "   Copied Systolic Array .mem files"; \
	fi
	@if ls $(GNAE_DIR)/TYTAN/Memory/*.mem 1> /dev/null 2>&1; then \
		cp $(GNAE_DIR)/TYTAN/Memory/*.mem $(VCS_DIR)/; \
		echo "   Copied GNAE/TYTAN .mem files"; \
	fi
	@echo "-- Running VCS simulation"
	cd $(VCS_DIR) && ./$(TOP_MODULE)_sim
	@echo "-- VCS simulation complete"

# View Waveforms
wave:
	@if [ -f $(VERILATOR_DIR)/dump.vcd ]; then \
		echo "-- Opening Verilator waveform"; \
		$(WAVE) $(VERILATOR_DIR)/dump.vcd; \
	elif [ -f $(VCS_DIR)/*.vpd ]; then \
		echo "-- Opening VCS waveform"; \
		$(WAVE) $(VCS_DIR)/*.vpd; \
	else \
		echo "-- No waveform dumps found"; \
	fi

# Lint check with Verilator (no simulation)
lint:
	@echo "=== Linting $(TOP_MODULE) with Verilator ==="
	@mkdir -p $(VERILATOR_DIR)
	$(VERILATOR) --lint-only \
		$(VERILATOR_FLAGS) \
		$(DESIGN_FILES) \
		$(TB_DIR)/$(TESTBENCH)
	@echo "-- Lint check complete"

# Debug build with extra information
debug: VERILATOR_FLAGS += --debug --gdbbt
debug: verilator

# Performance analysis
perf: VERILATOR_FLAGS += --stats --profile-cfuncs
perf: verilator

# List all design files (useful for verification)
list-files:
	@echo "=== Design Files ==="
	@echo ""
	@echo "Top Level:"
	@for file in $(TOP_FILES); do echo "  - $$file"; done
	@echo ""
	@echo "Systolic Array ($(SA_DIR)):"
	@for file in $(SA_FILES); do echo "  - $$file"; done
	@echo ""
	@echo "GNAE ($(GNAE_DIR)):"
	@for file in $(GNAE_FILES); do echo "  - $$file"; done
	@echo ""
	@echo "CNN Components ($(SUMIT_DIR)):"
	@for file in $(SUMIT_FILES); do echo "  - $$file"; done
	@echo ""
	@echo "Total: $(words $(DESIGN_FILES)) files"

# Check if all source files exist
check-files:
	@echo "=== Checking if all source files exist ==="
	@missing=0; \
	for file in $(DESIGN_FILES); do \
		if [ ! -f "$$file" ]; then \
			echo "  MISSING: $$file"; \
			missing=$$((missing + 1)); \
		fi; \
	done; \
	if [ $$missing -eq 0 ]; then \
		echo "  All $(words $(DESIGN_FILES)) files found!"; \
	else \
		echo "  $$missing file(s) missing!"; \
		exit 1; \
	fi

# Clean all simulation artifacts
clean:
	@echo "-- Cleaning simulation artifacts"
	-rm -rf $(VERILATOR_DIR) $(VCS_DIR)
	-rm -f *.vpd *.vcd *.wlf *.log
	-rm -f csrc simv simv.daidir
	-rm -f *.key DVEfiles
	@echo "-- Clean complete"

# Clean all including subprojects
clean-all: clean
	@echo "-- Cleaning subprojects"
	-$(MAKE) -C SystolicArray clean 2>/dev/null || true
	-$(MAKE) -C GNAE clean 2>/dev/null || true
	@echo "-- Deep clean complete"

# Phony targets
.PHONY: default help verilator vcs sa-verilator gnae-verilator wave lint debug perf list-files check-files clean clean-all
