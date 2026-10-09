# SIENNA file map

Every file in the design, testbench and tooling scope, and whether it is live.

**Live** = compiled into a build or executed by tooling. **Dead** = present but unreferenced or targeting a module that no longer exists. **Generated** = overwritten by tooling; do not hand-edit.

The repo has two git submodules, `GPNAE` and `SystolicMesh`, each with its own nested `ArithmeticLibrary` submodule. `.gitmodules` also lists a third entry, `SystolicArray`, pointing at the same URL as `SystolicMesh` with a path that is not checked out — a leftover from a rename.

---

## Top level

| File | Lines | Status | What it is |
|---|---|---|---|
| `Makefile` | 406 | live | All build, lint, wave, regression and clean targets. `DESIGN_FILES` is the authoritative source list |
| `regression.py` | 3,018 | live | Hardware sanity: golden model, stimulus generator, orchestrator and scoreboard, and every check behind `--action` (pack, gemm, perf, oracle, pack-models, gpnae-tflite, rq-vectors, selftest, all = `make regression`); see the `sienna-tooling` skill |
| `model_runner.py` | 1,349 | live | Host software stack in eight sections (contents block at the top): numerics, frontend (TFLite lowering), middle end (tiling, packing), device protocol (`write_layer`, `read_outputs`, `write_sets`), device build package, backends `RtlLayer` / `RtlSets` / `Emulator`, runtime, CLI (`--action model`, `--action tflite [--pack]`) |
| `run_real_model.py` | 116 | live, untracked | Runs a PyTorch checkpoint's weights through the pipeline |
| `README.md` | — | live | Overview, how to run, verification and Verilator performance tables |
| `performance_analysis_report.log` | — | untracked | Latency and throughput analysis; mixes measured and modeled figures |

### `src/`

| File | Lines | Status | What it is |
|---|---|---|---|
| `sienna_top.sv` | 1,270 | live | Top level: entry admission and the L0/L1/L3/L6/L9 links, the activation stage (lane fill on L4, collector on L5), the window dispatcher (L7 per lane), the pooling lanes (FIFO2 → Maxpool_2D → dropout on L8); the per-lane maxpool feeder FSM is gone since 2026-10-07 |
| `fwft.sv` | — | live | First-word-fall-through FIFO on credit links (L7 in, L8 out) since 2026-10-07; no overwrite |
| `sienna_set_side.svh` | 24 | live | `set_side_t`, the L0 host link's per-set sideband, included by sienna_top, sienna_layer, sienna_multi and their TBs |

### `Maxpool/`, `Dropout/`

| File | Lines | Status | What it is |
|---|---|---|---|
| `Maxpool/Maxpool_2D.sv` | — | live | 2-D max pooling with FP32-aware comparison, on credit links since 2026-10-07 |
| `Dropout/dropout.sv` | — | live | LFSR dropout on credit links since 2026-10-07 (passes its output credits through). `training_mode` is tied low at the top level, so it is a pass-through |

### `testbenches/`

| File | Status | What it is |
|---|---|---|
| `TB_sienna_top.sv` | live | The full-pipeline testbench. 474 lines |
| `test_config_pkg.sv` | **generated** | Written by `regression.py` and `model_runner.py` (`write_sv_package()`). Edits are lost on the next run |
| `TB_model_run.sv` | live | Layer-file testbench of `sienna_layer` (was `TB_sienna_layer`); `model_runner.RtlLayer` builds it once and runs one layer per invocation |
| `TB_sienna_model.sv` | live | Host-driven set engine on `sienna_top`; `model_runner.RtlSets` (`--engine sets`) |
| `tb_l9_sink.svh` | live | One lane's L9 consumer for the testbenches (slots, random stalls, a checker and `a_l9_slots`), included by TB_model_run, TB_sienna_model and TB_sienna_multi |
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
| `src/top/SystolicMesh.sv` | 394 | Array grid, staging banks, broadcast loader, mesh FSM, `MeshOutputSram`; `COLLAPSE_K`, `HOST_WORDS`, `WIDE_READ` |
| `src/top/SystolicArray.sv` | 165 | Synchronous T×T output-stationary array for a T×K by K×T product |
| `src/engine/ProcessingElement.sv` | 161 | One product per cycle into six partial sums, pairwise combine at the end |
| `src/engine/AccumulationUnit.sv` | 162 | Adder-tree reduce of the depth slices (a copy with collapse-k), global write address |
| `src/mem/MeshOutputSram.sv` | 65 | Two-bank result memory, one write port per output tile, wide read port |

The previous handshake tile (`PEMesh`, `MAC`, `RowInputQueue`, `ColumnInputQueue`, `OutputSram` and the old `SystolicArray`/`ProcessingElement`) is at git tag `handshake_tile_v1`.

### Testbenches

| File | Lines | Status | Notes |
|---|---|---|---|
| `TB_SystolicMesh.sv` | 421 | live | Main IP testbench. Patched in place by `regression.py`. Tolerance math is wrong — see issue 7 |
| `TB_SystolicArray.sv` | 146 | live | Unit test of the synchronous array over tile sizes and depths (`-GN=`, `-GK=`), against a real-valued model |

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
| `src/TYTAN/Memory/InputFIFO.v` | 73 | Status-bitmap FIFO. No simultaneous read+write. Used by the published `gpnae.sv` only since 2026-10-07 |
| `src/lane_fifo.sv` | 47 | Circular 32-entry FIFO with InputFIFO's read latency; the input FIFO of `gpnae_poly` and `gpnae_poly_int8` since 2026-10-07 (credit-legal puts while popping would reorder InputFIFO) |
| `src/lane_link.sv` | 92 | The lanes' shared credit front end: L4 input credits (32 advertised, one per pop), `last` handling, the L5 output counter and the K-credit group start, `a_in_room` and `a_in_order` |
| `src/TYTAN/Memory/taylor_coeffs.mem` | 30 | Coefficients, **binary** format. One short for tanh — see issue 2 |

### Testbench

| File | Lines | Status | Notes |
|---|---|---|---|
| `testbenches/TB_gpnae.sv` | 342 | live | No golden checking. Never uses `signal_tanh` — see issue 8 |
| `testbenches/TB_gpnae_activations.sv` | ~185 | live | Numerical check of all three activations against goldens. Caught issues 5b and 5c |
| `testbenches/TB_gpnae_selu_check.sv` | ~175 | live | Focused SELU reproducer written while isolating issue 3 |
| `tb_gpnae.vcd` | — | untracked artifact | Waveform written under `+dump`; git-ignored (`*.vcd`), as are `noStart.ron` and `TB_gpnae_behav.wcfg` |

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
| `Common/src/credit_link_if.sv` | 13 | live | The credit link interface (`put`, `data`, `credit`; modports producer, consumer, monitor), since 2026-10-07 |
| `Common/src/credit_counter.sv` | 24 | live | A producer's credit count (`has_credit_o`, `count_o`), `a_no_underflow`, `a_no_overflow` |
| `Common/src/credit_reg.sv` | 29 | live | `STAGES` register stages on a link (put and data forward, credit back); 0 is wires |
| `Common/src/credit_link_checker.sv` | 24 | live (checks only) | Bound to a link's monitor modport: no put without a credit, outstanding credits within SLOTS, every credit back at drain |

### Library testbenches

| File | Status | Notes |
|---|---|---|
| `Adders/FP32/testbenches/TB_fp32Adder.sv` | live | DPI-C against SoftFloat, ±3 ULP, 2000 random vectors |
| `Multipliers/FP32/testbenches/TB_fp32Multiplier.sv` | live | DPI-C against SoftFloat, ±2 ULP |
| `Adders/FP32/testbenches/TB_fp32AdderVIVADO.sv` | live | Same plan, reads `vectors.mem` instead of DPI-C |
| `Multipliers/FP32/testbenches/TB_fp32MultiplierVIVADO.sv` | live | Same |
| `Multipliers/Karatsuba/testbenches/TB_karatsuba.sv` | live | Signed and unsigned against native `*` |
| `*/testbenches/generate_vectors.sh` | live | Builds SoftFloat, compiles `gen_vectors.cpp`, emits `vectors.mem` |
| `Adders/FP32/testbenches/berkeley-softfloat-3/` | third party | The IEEE-754 reference model, not design source; the one copy, used by Adders/FP32, Multipliers/FP32, Adders/FP, Multipliers/FP and Common's `generate_vectors.sh` |

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
