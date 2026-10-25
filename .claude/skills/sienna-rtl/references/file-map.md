# SIENNA file map

Every file in the design, testbench and tooling scope, and whether it is live.

**Live** = compiled into a build or executed by tooling. **Dead** = present but unreferenced or targeting a module that no longer exists. **Generated** = overwritten by tooling; do not hand-edit.

The repo has two git submodules, `GPNAE` and `SystolicMesh`, each with its own nested `ArithmeticLibrary` submodule. `.gitmodules` also lists a third entry, `SystolicArray`, pointing at the same URL as `SystolicMesh` with a path that is not checked out — a leftover from a rename.

---

## Top level

| File | Lines | Status | What it is |
|---|---|---|---|
| `Makefile` | 406 | live | All build, lint, wave, regression and clean targets. `DESIGN_FILES` is the authoritative source list |
| `regression.py` | 3,018 | live | Hardware sanity: golden model, stimulus generator, orchestrator and scoreboard, and every check behind `--action` (pack, gemm, perf, oracle, pack-models, gpnae-tflite, rq-vectors, selftest, and points / point / summary = `make regression`); see the `sienna-tooling` skill |
| `model_runner.py` | 1,349 | live | Host software stack in eight sections (contents block at the top): numerics, frontend (TFLite lowering), middle end (tiling, packing), device protocol (`write_layer`, `read_outputs`, `write_sets`), device build package, backends `RtlLayer` / `RtlSets` / `Emulator`, runtime, CLI (`--action model`, `--action tflite [--pack]`) |
| `run_real_model.py` | 116 | live, untracked | Runs a PyTorch checkpoint's weights through the pipeline |
| `README.md` | — | live | Overview, how to run, verification and Verilator performance tables |
| `performance_analysis_report.log` | — | untracked | Latency and throughput analysis; mixes measured and modeled figures |

### `src/`

| File | Lines | Status | What it is |
|---|---|---|---|
| `sienna_top.sv` | 1,414 | live | Top level: entry admission and the L0/L1/L3/L6/L9 links, the activation stage (lane fill on L4, collector on L5; in int8 the requantize, `hold_mem` (N^2 bytes) and `add_lanes` for residual hold and add sets), the window dispatcher (L7 per lane), the pooling lanes (FIFO2 → Maxpool_2D → dropout on L8); the per-lane maxpool feeder FSM is gone since 2026-10-07 |
| `requant_lanes.sv` | 53 | live (int8) | One `int8Requantizer` per lane on the mesh's int32 sums, with each beat's tag (destination, hold or add) beside it for `req_lat()` cycles |
| `add_lanes.sv` | 54 | live (int8) | The residual ADD: one `int8QuantizedAdder` per lane on an add set's requantized beat and the `hold_mem` beat of the same index, the tag riding beside it for `qadd_lat()` cycles; `busy_o` while a beat is inside |
| `sienna_add_pkg.sv` | 18 | live | `add_words_t`, an add set's 110-bit ADD words (zp1, zp2, m1, m2, shift, zpo, amin, amax), shared by `set_side_t`, `sienna_top` and `add_lanes` |
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
| `TB_add_lanes.sv`, `TB_requant_lanes.sv` | live | Unit benches of `add_lanes` and `requant_lanes` on `regression.py --action add-vectors` / `rq-vectors` (`make add-lanes`, `make rq-lanes`); outputs gated on the true function |
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

The lane since Batch B Stage 2 (2026-10-24): design D (bf16 D', option (i)) of `/proj/work/spramanik/sienna_report/gpnae_fit_study.log`. The published Taylor lane and B's `gpnae_poly` / `gpnae_poly_int8` are gone; the last commit that builds the Taylor lane is tagged `gpnae_taylor_final` (2f2db42).

### Design

| File | Lines | What it is |
|---|---|---|
| `src/gpnae.sv` | 481 | The lane top: input FIFO and credit front end, capture into groups of K = 16, the barrel MAC, the post stage, bf16's tail, in-order emit; ports `tier_i` and the coefficient load path (`coef_wr_i`, `coef_addr_i`, `coef_data_i`) |
| `src/gpnae_pkg.sv` | 85 | Memory map (128 coefficient words; descriptors at 128 + 16 (4 tier + d) + field; tail sets at 320 + 8 tier + field), the descriptor and tail structs, the element kinds |
| `src/gpnae_coef_mem.sv` | 105 | The host-writable coefficient memory; its reset default is the generated section `fit_poly_coeffs.py` writes |
| `src/gpnae_act_in.sv` | 255 | Input stage: piece select from abs(x) against the descriptor's bounds; floats square tanh's operand and split x log2 e into 2^-n and f for negative sigmoid; int8 rescales (int16Multiplier) and centres the operand |
| `src/TYTAN/barrel_mac.sv` | 214 | Format-free barrel Horner: per-slot base, degree and fraction; a slot whose degree is below the group's pads with result 0 |
| `src/gpnae_horner.sv` | 68 | One Horner step: fmt_mul then fmt_add in floats, a register then `fixedPointMultiplyAdd` in int8 |
| `src/fixedPointMultiplyAdd.sv` | 50 | int8 step, sat16(floor(a x / 2^(11 + frac)) + c); fxMac's arithmetic, proven equal before fxMac left ArithmeticLibrary |
| `src/gpnae_exp_split.sv` | 37 | fp32 k into its integer n and exact fraction f (2^-n P(f) range reduction) |
| `src/gpnae_post.sv` | 256 | Post stage: lambda x, x P, P, 1 - P, the 2^-n scale and E(1 - E), the rails; int8's SELU requantize (int24Multiplier, int8Requantizer) |
| `src/gpnae_tail.sv` | 281 | bf16 only: negative sigmoid below -1.5 in fp32 (2^-n P(f), Newton steps for 1/(1 + E)), eight contexts |
| `src/lane_fifo.sv` | 47 | Circular 32-entry input FIFO |
| `src/lane_link.sv` | 92 | The lanes' credit front end: L4 input credits, `last`, the L5 output counter and the K-credit group start |

### Scripts and benches

| File | What it is |
|---|---|
| `gpnae_model.py` | Bit-exact model of the lane, the memory image and its load writes |
| `fit_poly_coeffs.py` | Records the 1% designs, fits the 5% and 10% tiers, writes `gpnae_coef_default.json` and the RTL's reset section |
| `regression.py` | `make regression`: lint, check-fmt, check-hier, the true-function gate per tier, protocol, load, cycles, faults |
| `mutants.py` | One planted fault per new assertion and for the load and tier features |
| `testbenches/TB_gpnae.sv` | The lane bench (plusargs: size, tier, +load, overlap, streaming, stalls, faults) |
| `testbenches/TB_barrel_mac.sv` | barrel_mac bit for bit per define |

---

## `ArithmeticLibrary/` (vendored twice)

Present at both `GPNAE/ArithmeticLibrary/` and `SystolicMesh/ArithmeticLibrary/`, the same commit in both. The top-level build compiles only `SystolicMesh/ArithmeticLibrary` (issue 9, fixed); GPNAE's own `Makefile` uses its checkout.

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
| `Multipliers/Karatsuba/testbenches/TB_karatsuba.sv` | live | Signed and unsigned against native `*` |
| `Adders/FP32/testbenches/berkeley-softfloat-3/` | third party | The IEEE-754 reference model, not design source; the one copy, used by Adders/FP32 and Multipliers/FP32 |

---

## Build and tool output (git-ignored)

| Path | What it is |
|---|---|
| `Verilator/` | Verilator object directory, compiled binary, copied `.mem` files |
| `VCS/` | VCS working directory |
| `SystolicMesh/testbenches/results/readiness/` | 36 IP regression logs plus `readiness_report.md` |
| `__pycache__/` | Python bytecode |

---

## Duplicated data files

None for the lanes since Batch B Stage 2: GPNAE's coefficients are the coefficient memory's reset section in `GPNAE/src/gpnae_coef_mem.sv`, generated with `GPNAE/gpnae_coef_default.json` by `fit_poly_coeffs.py`, and no lane reads a `.mem` file. The published Taylor lane's `taylor_coeffs.mem` copies went with it (tag `gpnae_taylor_final`).
