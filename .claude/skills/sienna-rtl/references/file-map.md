# SIENNA file map

Every file in the design, testbench and tooling scope, and whether it is live.

**Live** = compiled into a build or executed by tooling. **Dead** = present but unreferenced or targeting a module that no longer exists. **Generated** = overwritten by tooling; do not hand-edit.

The repo has two git submodules, `GPNAE` and `SystolicMesh`, each with its own nested `ArithmeticLibrary` submodule. `.gitmodules` also lists a third entry, `SystolicArray`, pointing at the same URL as `SystolicMesh` with a path that is not checked out — a leftover from a rename.

---

## Top level

| File | Lines | Status | What it is |
|---|---|---|---|
| `Makefile` | 406 | live | All build, lint, wave, regression and clean targets. `DESIGN_FILES` is the authoritative source list |
| `regression.py` | 557 | live | Golden model, stimulus generator, orchestrator and scoreboard |
| `run_real_model.py` | 116 | live, untracked | Runs a PyTorch checkpoint's weights through the pipeline |
| `README.md` | 1 | — | Contains only the title |
| `performance_analysis_report.log` | — | untracked | Latency and throughput analysis; mixes measured and modeled figures |

### `src/`

| File | Lines | Status | What it is |
|---|---|---|---|
| `sienna_top.sv` | 740 | live | Top level: outer FSM, fill FSM, window dispatcher, per-lane maxpool feeder |
| `fwft.sv` | 75 | live | First-word-fall-through FIFO. Never backpressures; overwrites when full |

### `Maxpool/`, `Dropout/`

| File | Lines | Status | What it is |
|---|---|---|---|
| `Maxpool/Maxpool_2D.sv` | 197 | live | 2-D max pooling with FP32-aware comparison |
| `Dropout/dropout.sv` | 85 | live | LFSR dropout. `training_mode` is tied low at the top level, so it is a pass-through |

### `testbenches/`

| File | Status | What it is |
|---|---|---|
| `TB_sienna_top.sv` | live | The full-pipeline testbench. 474 lines |
| `test_config_pkg.sv` | **generated** | Written by `regression.py`. Edits are lost on the next run |
| `matrix_west.mem`, `matrix_north.mem`, `expected_output.mem` | generated | Stimulus and golden output, hex |
| `hardware_trace.txt` | output | Stage-by-stage hardware values, parsed by `regression.py` |
| `pipeline_lane_status.txt` | output | Per-lane FSM snapshots. The first place to look for a stall |
| `results/pipeline/*.log` | output | Raw simulation logs, one per test |
| `results/pipeline/*_expected_flow.txt` | output | NumPy golden values per stage |
| `results/pipeline/*_data_flow.txt` | output | Hardware values per stage per lane, laid out to match |

---

## `SystolicMesh/`

### Design — all live, all in `DESIGN_FILES`

| File | Lines | What it is |
|---|---|---|
| `src/top/SystolicMesh.sv` | 258 | Tile grid, broadcast loader, mesh FSM, `MeshOutputSram` |
| `src/top/SystolicArray.sv` | 220 | Handshake tile, used when `SYNC_TILES=0`. Also defines the `NorthInputQueue` / `WestInputQueue` wrappers |
| `src/top/SyncArray.sv` | 170 | Default tile: synchronous N×N output-stationary array for an N×K by K×N product |
| `src/engine/SyncPE.sv` | 160 | SyncArray PE: one product per cycle into six partial sums, pairwise combine at the end |
| `src/mem/RowInputQueue.sv` | 141 | Row-major strided input queue |
| `src/mem/ColumnInputQueue.sv` | 140 | Column-major strided. Identical to the above except the addressing |
| `src/mem/OutputSram.sv` | 116 | Per-tile column-drain collector |
| `src/mem/MeshOutputSram.sv` | 48 | Shared multi-write-port result memory |
| `src/engine/PEMesh.sv` | 156 | PE array, wavefront valid propagation, drain shift register |
| `src/engine/ProcessingElement.sv` | 195 | Four-state PE with muxed east output |
| `src/engine/AccumulationUnit.sv` | 164 | Reduces depth slices, computes global write address |
| `src/engine/MAC.sv` | 142 | Multiply-accumulate FSM around `fp32Multiplier` + `fp32Adder` |

### Testbenches

| File | Lines | Status | Notes |
|---|---|---|---|
| `TB_SystolicMesh.sv` | 421 | live | Main IP testbench. Patched in place by `regression.py`. Tolerance math is wrong — see issue 7 |
| `TB_SystolicArray.sv` | 504 | live | Single-tile testbench. Same tolerance bug |
| `TB_Mesh_2x2.sv` | 382 | **dead** | Targets nonexistent module `Mesh` |
| `TB_Mesh_3x3.sv` | 391 | **dead** | Same, N=3 |
| `TB_Mesh_5x5.sv` | 508 | **dead** | Same, N=5 |
| `TB_Mesh_8x8.sv` | 626 | **dead** | Same, N=8 |

### Tooling

| File | Lines | What it is |
|---|---|---|
| `regression.py` | 498 | IP regression. Patches the TB per config to exploit incremental make |
| `matmul_tests.py` | 255 | 8 matmul generators. Also the shared `write_mem` / `_f2h` / `_ref_matmul` helpers |
| `conv_tests.py` | 469 | 9 conv generators. Its docstring is the authoritative im2col layout spec |
| `Makefile` | 141 | Submodule build targets |
| `README.md` | 113 | Architecture, measured cycle counts, scaling analysis. The best existing prose on the mesh |

---

## `GPNAE/`

### Design — all live

| File | Lines | What it is |
|---|---|---|
| `src/gpnae.sv` | 385 | Top level plus `gpnae_control_unit` |
| `src/SeLu.sv` | 173 | SELU. ~95 lines of commented-out earlier version at the end |
| `src/sigtan.sv` | 43 | Sigmoid/tanh composition. Uses a tri-state mux — see issue 5 |
| `src/fp32_down.sv` | 170 | 6-stage `x − 1.0` |
| `src/fp32_up_down.sv` | 211 | Parallel `x + 1.0` and `x − 1.0`. `fp32_down` duplicated |
| `src/TYTAN/controller.sv` | 192 | Polynomial sequencer with credit counter |
| `src/TYTAN/datapath.v` | 58 | Horner's method: one multiplier, one adder |
| `src/TYTAN/mac.sv` | 81 | Wires controller, datapath and `CoeffROM` together |
| `src/TYTAN/LZC.v` | 44 | `cntlz8`, `cntlz24`. `cntlz8` collides with the ArithmeticLibrary copy |
| `src/TYTAN/Memory/CoeffROM.v` | 27 | Thin wrapper over `rom_block` |
| `src/TYTAN/Memory/ROM.v` | 28 | Registered-read ROM. Uses **`$readmemb`** |
| `src/TYTAN/Memory/RAM.v` | 40 | Dual-port RAM backing the input FIFO |
| `src/TYTAN/Memory/PE5B.v` | 42 | 32-to-5 priority encoder |
| `src/TYTAN/Memory/InputFIFO.v` | 73 | Status-bitmap FIFO. No simultaneous read+write |
| `src/TYTAN/Memory/taylor_coeffs.mem` | 30 | Coefficients, **binary** format. One short for tanh — see issue 2 |

### Testbench

| File | Lines | Status | Notes |
|---|---|---|---|
| `testbenches/TB_gpnae.sv` | 342 | live | No golden checking. Never uses `signal_tanh` — see issue 8 |
| `testbenches/TB_gpnae_activations.sv` | ~185 | live | Numerical check of all three activations against goldens. Caught issues 5b and 5c |
| `testbenches/TB_gpnae_selu_check.sv` | ~175 | live | Focused SELU reproducer written while isolating issue 3 |
| `tb_gpnae.vcd` | — | tracked artifact | ~586k lines of waveform dump committed to git |

---

## `ArithmeticLibrary/` (vendored twice)

Present at both `GPNAE/ArithmeticLibrary/` and `SystolicMesh/ArithmeticLibrary/`. **Both copies are compiled into the top-level build**, so every module below is defined twice — see issue 9. The only differences between the copies are `logic` vs `wire`/`reg` port declarations in `fp32Adder.sv` and `fp32Multiplier.sv`.

| File | Lines | Status | What it is |
|---|---|---|---|
| `Adders/FP32/src/fp32Adder.sv` | 266 | live | 5-stage FP32 adder. Flush-to-zero, truncating |
| `Adders/FP32/src/LZC.sv` | 53 | live | `cntlz28`, `cntlz8` |
| `Multipliers/FP32/src/fp32Multiplier.sv` | 321 | live | Karatsuba mantissa, pipelined exponent adder, truncating |
| `Multipliers/Karatsuba/src/karatsubaUnsigned.sv` | 174 | live | 6-stage, three `R4Booth` cores |
| `Multipliers/Karatsuba/src/karatsubaSigned.sv` | 84 | unit test only | Sign-magnitude wrapper. In the Karatsuba standalone Makefile for `TB_karatsuba`, but **not** in the top-level `DESIGN_FILES` — no SIENNA datapath uses it |
| `Multipliers/Radix4Booth/src/R4Booth.sv` | 111 | live | Contains a dead adder tree — see issue 11 |
| `Divider/FP32/src/fp32Divider.sv` | 216 | live | ~28-stage FP32 divider |
| `Divider/FP32/src/divu.sv` | 95 | live | Project F restoring divider, MIT licensed |

### Library testbenches

| File | Status | Notes |
|---|---|---|
| `Adders/FP32/testbenches/TB_fp32Adder.sv` | live | DPI-C against SoftFloat, ±3 ULP, 2000 random vectors |
| `Multipliers/FP32/testbenches/TB_fp32Multiplier.sv` | live | DPI-C against SoftFloat, ±2 ULP |
| `Adders/FP32/testbenches/TB_fp32AdderVIVADO.sv` | live | Same plan, reads `vectors.mem` instead of DPI-C |
| `Multipliers/FP32/testbenches/TB_fp32MultiplierVIVADO.sv` | live | Same |
| `Multipliers/Karatsuba/testbenches/TB_karatsuba.sv` | live | Signed and unsigned against native `*` |
| `Divider/FP32/testbenches/TB_Divider_FP32.sv` | **dead** | Targets nonexistent `divide_32`; declares itself `TB_Multi_FP32`; no checking |
| `*/testbenches/generate_vectors.sh` | live | Builds SoftFloat, compiles `gen_vectors.cpp`, emits `vectors.mem` |
| `*/testbenches/berkeley-softfloat-3/` | third party | ~1,800 `.c` and ~940 `.cpp` files. The IEEE-754 reference model, not design source |

`ArithmeticLibrary/README.md` is not about the library — it documents the Vivado DPI linker workaround (symlinking Vivado's bundled `ld` to the system one).

---

## Build and tool output (git-ignored)

| Path | What it is |
|---|---|
| `Verilator/` | Verilator object directory, compiled binary, copied `.mem` files |
| `VCS/`, `VIVADO/` | VCS and Vivado working directories |
| `VIVADO/SIENNA.runs/synth_1/runme.log` | Synthesis log. Source of the utilization and critical-warning evidence in issue 13 |
| `SystolicMesh/testbenches/results/readiness/` | 36 IP regression logs plus `readiness_report.md` |
| `__pycache__/` | Python bytecode |

---

## Duplicated data files

`taylor_coeffs.mem` exists in six places, currently all identical (md5 `5f5edfa7...`, 30 lines):

```
GPNAE/src/TYTAN/Memory/taylor_coeffs.mem        <- edit this one
GPNAE/taylor_coeffs.mem
Verilator/taylor_coeffs.mem
VIVADO/SIENNA.ip_user_files/mem_init_files/taylor_coeffs.mem
VIVADO/SIENNA.sim/sim_1/behav/xsim/taylor_coeffs.mem
GPNAE/VIVADO/gpnae_vivado_sim.sim/sim_1/behav/xsim/taylor_coeffs.mem
```

The Makefile's `copy_mem_files` function sweeps `*.mem` and `*.hex` from eight directories into the simulation directory with `cp -u`, so which copy wins depends on modification time and directory order rather than on intent.
