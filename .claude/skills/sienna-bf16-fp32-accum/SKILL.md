---
name: sienna-bf16-fp32-accum
description: Use when designing, building or verifying SIENNA's bf16 builds with fp32 accumulation - bf16 operands and products widened exactly into fp32 sums in the PEs and reducers, the fp32 -> bf16 narrowing of each result before the activation lanes, the widened bias, the bit-exact models for bf16, and the accuracy and area effect against bf16 accumulation.
---

# bf16 builds accumulate in fp32 (spec approved by Soham 2026-10-08, with the three recommended decisions)

## Why

bf16 builds today accumulate in bf16 (`sienna_fmt_pkg::acc_w` = 1 + EXP_W + MAN_W = 16, ArithmeticLibrary `Common/src/sienna_fmt_pkg.sv:22-23`), a choice the uniform-format work made on purpose ("accepting the accuracy cost", `sienna-uniform-format/SKILL.md:27`). The cost is not small:
- MLPerf Tiny autoencoder (ad01), first layer, 640 inputs per output: 100-180% of the layer's largest value off the float reference; the anomaly score 3x the reference (bf16 30.84 vs 11.09; fp32 11.093858). Every bf16 model run since `three_formats_v4` (runs ms_*_bf16, f_mdl_rn*_bf16, n64m_*_bf16, n64cm_*_bf16).
- bf16 GEMM error grows with depth K: 6% at 256, 23% at 1024, 50% at 3072 (`sienna-uniform-format/SKILL.md`).
- Classifier layers (128 or fewer inputs) lose 1-4%; ResNet-8, DS-CNN and MobileNet keep their class.
Soham chose fp32 accumulation for bf16 (2026-10-08), the industry norm (bf16 multiply, fp32 accumulate).

## What changes, and what does not

| Stays bf16 (16 bits) | Becomes fp32 (32 bits) |
|---|---|
| Host operands, staging banks, weight cache (`mem_A`, `mem_B`, `wcache`) | Each PE's product and accumulator slots (`prod`, `acc[BANKS][U]`) |
| The L1 staging and L2 weight-cache links | The reducer tree and its pass-through registers (`AccumulationUnit`) |
| The result banks and the L3 result link (if the narrowing sits in the mesh, decision 1) | The bias inside the mesh (widened exactly from the bf16 bias row) |
| GPNAE lanes, activation banks, pooling, dropout, the output (L4-L9) | Partial sums across depth passes (they live in the PE slots) |
| fp32 and int8 builds: untouched, outputs must stay identical | |

Cycle counts do not change: `fpMulWiden` takes 3 cycles = today's `mul_lat(8,7)`, the fp32 adder 5 = `add_lat`, so the slot rotation U = 6 is unchanged (survey of 2026-10-08, from the RTL). As built, that holds for the mesh; the polynomial lanes' cycles follow the data (Ruling 3, As built).

## Design (approved 2026-10-08)

1. **Multiplier:** `fpMulWiden` (ArithmeticLibrary `Multipliers/FPWiden/src/fpMulWiden.sv`): bf16 x bf16 to an exact fp32 product, same ports as `fp32Multiplier`, 3 cycles, already tested against `fp32Multiplier` on widened inputs (203,366 bf16 products, 0 mismatches, history/2026-09-28_aril_gate.txt). Added to SM/Makefile, SIENNA Makefile `SM_LIB_FILES`, `synth/sienna_rtl.f`. It resets its datapath flops; bring it to the D-8 rule (valids only) in the same change.
2. **Adder for every sum (PE and reducer):** `fp32Adder`, the unit fp32 builds already use, so the bf16 sum path is the fp32 sum path. (`fpAdder(8,23)` is bit-identical on results, differs only in a D-1 overflow flag nobody reads; using `fp32Adder` avoids a second fp32 adder in the design.)
3. **Parameters:** add `acc_exp_w(exp_w, man_w)` / `acc_man_w(...)` to `sienna_fmt_pkg` (floats: 8, 23; int: 0, 31 style or as today), and `acc_w` = 32 for every format. PE and `AccumulationUnit` choose multiplier by operand format and adder by accumulator format; everything else keeps `EXP_W/MAN_W` as the operand/lane format. Keep the function shapes `perf_analysis` parses (regression.py:1570-1585). Add `ACC_W` to the stale-package guard (Makefile:38-41).
4. **Decision 1 - where the result is narrowed to bf16: in the mesh, at the reducer's write into the result banks** (approved). Every consumer of a mesh result (the lanes, the ReLU/linear bypass, pooling) takes bf16, and nothing after the reducer needs fp32 (bias and residual are added inside the reducer and the PE slots). Narrowing there keeps the result banks, the L3 link (`RES_W`), `sienna_top` and every TB result file at 16 bits: the mesh's external interface does not change. The alternative, narrowing at the lane feed in `sienna_top` (as int8's `requant_lanes`), doubles the result banks (RESULT_BANKS x N^2 x 32 bits: 512 Kbit at N = 64) and the L3 beat (1027 bits at 32 lanes) for no accuracy gain.
5. **Decision 2 - narrowing rounding: round to nearest even** (approved), NaN and infinity preserved, results below bf16's smallest normal flushed to zero. RNE is what `op_round` / `fpu.from_fp32` already use for operands, and what bf16 conversion means everywhere else; truncation (as the arithmetic units do internally) would bias every output toward zero. One new ArithmeticLibrary unit, `fpNarrow` (fp32 -> any narrower float), with a TB against the Python model (corners, every rounding tie class, random fp32 values).
6. **Decision 3 - bias: the bias row stays bf16 in the weight stream and is widened exactly** (bf16 bits << 16) before it reaches the mesh's fp32 bias input (approved). Today `sienna_layer.sv:471` and the TBs zero-extend it (`ACC_W'(...)`), which becomes wrong when ACC_W is 32: replace with the exact widen. The alternative, an fp32 bias row, changes the weight stream format and `format_layer` for no gain at bf16 accuracy.
7. **Silent truncation hazards to fix explicitly** (all builds use `-Wno-WIDTHTRUNC/-WIDTHEXPAND`): PE `fpMultiplier` 16-bit result into a 32-bit `prod`; `AccumulationUnit` choosing `fpAdder(EXP_W,MAN_W)`; `sienna_top.sv:1012` `fill_d = wide_rd_data` (only if narrowing moved to sienna_top); `sienna_layer.sv:471` bias zero-extend; TB `bias_i[c] = ACC_W'(q[c])` in TB_sienna_top, TB_sienna_multi, TB_sienna_model. Each gets an elaboration or time-0 width check so a mismatch stops the build.
8. **Bit-exact models:** `mesh_model.matmul` (and `matmul_packed`, `_reduce`): products by an exact widening multiply, sums in FP32 with `fpu.add`, the bias widened, then `from_fp32`-style RNE narrowing to bf16 at the end. `regression.py` `_golden_bits`, `_packed_float`, `exact_layer`, `lane_inputs`, `stim_format` result handling and `TB_PE_pack`'s bf16 expectation follow. `gpnae_model` unchanged (it takes the narrowed bf16 word).

## Accuracy and cost (estimates until measured)

- Expected (not measured): ad01's first layer error from 100-180% down to bf16's output rounding (about 0.4% of the value); anomaly score within a few percent of the reference; GEMM error no longer growing with K. Measured after the change by the model gate (#43) and `make gemm`.
- Area (my arithmetic from the RTL sizing, no synthesis exists): PE accumulator slots double (BANKS 4 x U 6 x 32 bits per PE instead of 16: 12 KiB to 24 KiB flop-equivalent at N = 16, 48 KiB to 96 KiB at N = 32, 192 KiB to 384 KiB at N = 64); PE and reducer adders become 32-bit class; the PE multiplier becomes an exact 8x8 product with fp32 normalise. Result banks and L3 unchanged (decision 1). bf16 stays well below fp32 because operand storage, the weight cache, the lanes and pooling stay 16-bit.

## Verification (bottom-up, N <= 32 until check-in)

1. ArithmeticLibrary: `fpNarrow` TB against the Python model; `fpMulWiden` reset change re-run (its existing TB); `TB_sienna_fmt_pkg` expectations (`acc_w(8,7)` = 32, the new accumulator-format helpers).
2. SystolicMesh: mesh regression bf16 at N = 8, 16, 32, every tile, bit-exact against the new `mesh_model`; fp32 and int8 identical to before; `TB_PE_pack` bf16 case.
3. SIENNA: `make regression FMT=bf16` (bf16 goldens regenerate from the new model; bit-exact), fp32 and int8 gates unchanged and outputs identical to `stage_credits_v6`; `make gemm FMT=bf16` (bit-exact against the model; error vs K reported); `make pack FMT=bf16`.
4. Model gate (#43): bf16 layers bit-exact against the model, classifiers' top-1, ad01's anomaly score now inside its bound, so ad01 bf16 leaves the REPORTED list and is gated.
5. After check-in: the N = 64 sweep in bf16 (with the gate).

## Order of work (each repo on its own branch, innermost first)

ArithmeticLibrary (`fpNarrow`, package helpers, `fpMulWiden` reset) -> SystolicMesh (PE, reducer, narrowing at the result write, `mesh_model`, stim/regression, TBs) -> SIENNA (bias widen in `sienna_layer` and TBs, goldens, model_runner `ACC_W`, file lists, docs: `sienna-uniform-format`, `sienna-int8` table, README line 30). GPNAE: no change.

## Decisions

All three open questions were answered with the recommendation (Soham, 2026-10-08): narrow in the mesh at the result write; round to nearest even; bf16 bias row widened exactly.

## As built (2026-10-09; branches `bf16_accum`, not merged, not pushed)

Measurements, each with its run: `sienna_report/bf16_accum_compare.log` (outside git). Task record: `.superpowers/sdd/implementation-plan-sienna-bf16-fp32-accum/` (`progress.md`, `task-1..4-report.md`).

### Commits

| Repo | Commits (on `bf16_accum`) |
|---|---|
| ArithmeticLibrary (both clones) | f6a443b `acc_w` 32, `out_w`, `acc_exp_w`/`acc_man_w`, `widen`, `fpNarrow` + TB + `fpu.narrow`, `fpMulWiden` to D-8, `fpu.widen`/`mul_widen`; cce9105 review fixes (two more fpNarrow mutants, `fpu.narrow` masks to 32 bits, fpNarrow lint 0 warnings); a03aa29 `supported()` comment |
| SystolicMesh | 5503185 TB_PE_pack made non-vacuous (on main's RTL); 9f01605 PE `fpMulWiden` + `fp32Adder`, reducer `fp32Adder` with `fpNarrow` at its write, `OUT_W`, fp32 `bias_i`, `mesh_model`, `stim_format`, new tests, AriL bump; a1bc0de review fixes (`mm_range_edge`, `mm_bias_special` wording, expect guard message, file list, `fp32_of` guard); 473285f AriL bump; 3903e03 README; fd0cc98 the regression runs the model checks first |
| GPNAE | ddd8429, a812858: ArithmeticLibrary bumps only; lanes unchanged |
| SIENNA | 9327b61 bias widened in `sienna_layer` and the TBs, `OUT_W` result link in `sienna_top`, width checks, goldens, `bf16_bias_special_linear_nopool`, tooling, bumps; 9e7df01 ad01 bf16 gated (`REPORTED` empty); 0341b75 bf16 layer bound 0.02, bf16 GEMM gated, bias check on the registered capture, `a_bias_widened`; 87d564e bumps (AriL a03aa29 everywhere); 88ef427 TB bias-check strings; 563c9af docs; final-review fixes 3803fec (model-check gate steps, SystolicMesh fd0cc98), d426a95 (`layer_bias_special`), e610ebd (LAYER_BOUND comment), then their docs commit |

### What was built

As designed, with the three approved decisions: narrow in the mesh at the reducer's write, round to nearest even, bf16 bias row widened exactly.
- **Package:** `acc_w` = 32 in every format; `out_w` = 16 in bf16, 32 in fp32 and int8 (the mesh result word: main's `acc_w`); `acc_exp_w`/`acc_man_w` (8/23 floats, 0/31 int); `widen(x, man_w)` = `x << (23 - man_w)`. `mul_lat`/`add_lat` shapes kept.
- **fpNarrow** (`Converters/FPNarrow`): combinational fp32 -> 8-exponent float; RNE, NaN -> 7FC0, exponent 0 -> signed zero, overflow -> infinity, identity at MAN_W 23. 1,196,635 values per format (27 corners, every tie class, a tie sweep, 10^6 random) 0 mismatches against `fpu.narrow`; three mutants (truncate, ties down, saturate) fail.
- **fpMulWiden** resets only its valids (D-8): 203,366 bf16 and 200,546 fp16 products 0 mismatches against fp32Multiplier on widened inputs; the PE reads its reset-free result only with `done_o`.
- **Mesh:** the bf16 PE is `fpMulWiden` + `fp32Adder` with 32-bit `prod`/`acc` (U = 6 unchanged); the reducer adds in `fp32Adder` for every float and narrows with `fpNarrow` (bf16 only, `G_NARROW`); `OUT_W` sizes the result banks and the L3 link; `bias_i` is ACC_W (fp32 bits, the caller widens). `fpMultiplier`/`fpAdder` left the mesh's file list; SIENNA keeps them for the GPNAE lanes and dropout.
- **Top level:** `sienna_top`'s `wide_rd_data`/L3 at `OUT_W`; `sienna_layer` and TB_sienna_top/TB_sienna_multi/TB_sienna_model widen the bias row with `sienna_fmt_pkg::widen`; `model_runner` writes `ACC_W` 32 and `OUT_W`; the Makefile guard requires both.
- **Width checks** for every silent-truncation site of item 7, each shown firing once: `G_BAD_PROD` (PE), `G_BAD_OUT_W` (AccumulationUnit, SystolicMesh, sienna_top), `G_BAD_FILL` (sienna_top's `fill_d`), `G_BAD_BIAS_W` (sienna_layer), `G_BAD_ACC` (three TBs), the package guard (`test_guard_rejects_bf16_sum_package`).
- **New tests:** `test_bf16_k640_within_one_ulp` (model: main's 8.7% of words within 1 ulp, new 100% equal to float64 rounded to bf16); `mm_accum` (one sum over 4 depth passes; no mesh accumulate test existed), `mm_bias_special` (±0 and subnormal bias words), `mm_range_edge` (a sum of 0x7F7F8000 narrows to ±inf; a first tree pair above bf16's largest finite cancels back to ±2^119: only fp32 sums pass); pipeline `bf16_bias_special_linear_nopool`.
- **Direct bias checks** (Review Focus 5): a wrong widen of -0 or a subnormal bias is invisible in the results (the fp32 adder reads both as +0), so TB_sienna_top compares the mesh's registered bias-queue entry with the row widened by an independent expression on every bias put (28-74 puts per bf16 bias test), with a guard that fails a bias test that compared none; `sienna_layer`'s `a_bias_widened` checks `bias_buf` the same way. Mutants: zero-extend (1036 bias-check failures), -0/subnormal flushed to +0 (740 failures, every result word still matched), check removed (the guard fires), layer flush (the assertion fires).
- **TB_PE_pack was vacuous and is fixed** (SystolicMesh 5503185): with Verilator 5.035 `$shortrealtobits(shortreal'(x))` gave 0, so its fp32 and bf16 benches compared zeros with zeros. It now builds words with an integer encoder (`fp32_of`, self-checked, `$fatal` outside [0, 2^24)) and fails unless most slots are nonzero; main's RTL passes it (ba2_pepack_main). Earlier "TB_PE_pack PASSED" runs checked only int8 in effect.

### Rulings (Soham or the controller, `progress.md`)

1. Task 2 started in parallel with Task 1's re-review, since it consumed only the package and units, which the fix did not change in behaviour.
2. No separate tag for the bf16 merge: at close, move `stage_credits_v6` (SIENNA, SystolicMesh) and `credit_lanes_v3` (GPNAE) to the merged commits, with updated messages and sienna-rtl table rows (pushed tags: moving them rewrites the remote tags). Open: the close step.
3. **The cycle constraint, reworded.** "bf16 cycle counts identical" was too strong. The mesh's own cycles are identical (unit latencies unchanged, as above), but the polynomial lanes (tanh, SELU, sigmoid) are data dependent: `gpnae_tail` computes an element past the fits in a number of cycles that depends on its value, and fp32 sums change the mesh's results. As built: mesh, host, pack, GEMM and model-layer cycles identical to main; activation-limited polynomial-lane streams may move, and do in 5 of 32 pipeline tests (1-5 streamed cycles) and 7 of 32 perf configs (largest: `matmul_bias_tanh` 202.7 -> 212.0 cycles per set, one extra 56-cycle tail group in sets 17 and 22). In every one the per-set mesh time moves only after the first activation change (results waiting on full banks). The perf model predicts the pipeline moves of matmul_large_selu and matmul_large_tanh (+4 / +3 cycles against +5 / +3 measured).
4. The bf16 layer bound went 0.25 -> 0.02 (worst layer now 3.77e-03, 5x margin; every model's worst with bf16 sums, 3.35e-02 to 1.83, is above it), so a bug shared by the RTL and its exact model is caught; the bf16 GEMM sweep gates on the same bound.

### Deviations from this spec and the plan

- Cycles: as Ruling 3 (the plan's Global Constraint and Task 4 bullet are reworded to match).
- `sienna_layer`'s check is `$bits(bias_buf[0][0]) == sienna_fmt_pkg::acc_w(EXP_W, MAN_W)`: the plan's `$bits(bias_buf[0]) == ACC_W` was true by construction.
- fp32 and int8 are compared with main c05c332's merged-tree runs (`mc_gate_*`, `crb_*`), the plan's reference, rather than `stage_credits_v6`.
- The bf16 GEMM error is gated (0.02), not only reported.
- Accuracy against the estimates above: ad01's first layer 1.81e-03 to 2.27e-03 (estimate about 0.4%); its anomaly score within 2.24e-03 of the float model's (estimate "a few percent"); GEMM error 2.0e-03 to 3.6e-03 at every K up to 3072.

### Measured (bf16, before = bf16 sums on main, after = fp32 sums)

| | before | after |
|---|---|---|
| ad01 anomaly score error, 4 inferences | 1.69 to 1.84 | 3.65e-05 to 2.24e-03 (bound 1e-2, gated) |
| worst layer error: resnet8 / kws / vww / ad01 | 7.41e-02 / 3.35e-02 / 4.01e-02 / 1.83 | 3.40e-03 / 3.34e-03 / 3.77e-03 / 3.32e-03 |
| classifier top-1 agreement | 9 / 9 | 9 / 9 |
| GEMM error at K 16 / 256 / 1024 / 3072 | 1.6e-02 / 7.9e-02 / 3.0e-01 / 5.0e-01 | 3.2e-03 / 3.2e-03 / 3.6e-03 / 3.1e-03 |
| storage, estimate (no synthesis), N 16 / N 32 | 104.0 / 420.1 KiB | 116.8 / 470.0 KiB (+12%) |

Runs: models mc_models_bf16, ba0_models_N32_bf16 -> ba3f_models_N16_bf16, ba3f_models_N32_bf16 (N 16 and N 32 identical after); GEMM ba0_gemm_bf16 -> ba3f_gemm_bf16; area `sienna_jobs/area_estimate.py` (now with separate sum, result and lane widths). fp32 and int8: words and cycles identical to main in every run compared.

### Final-review fixes (2026-10-09)

- **The checks with their own references gate `make regression`.** Every other bf16 check compares the RTL with its bit-exact model, and the 0.02 layer bound cannot see a 1-ulp rounding bug, so a bug shared by `fpNarrow` and `fpu.narrow` (the same add-and-shift trick) passed the gate. New gated steps in every format: `aril-fpu` (`test_fpu.py`: corner table, integer divmod RNE), `aril-narrow` (FPNarrow `all`: TB corners, RTL against `fpu.narrow` on 1,196,635 values, lint, G_BAD_FORMAT), `sm-model-tests` (`test_mesh_model_packed.py` with K = 640 against float64, `test_stim_format.py`). SystolicMesh's own regression runs the same three first (`--no-checks` skips them; the gate's sm-verilator passes `SM_CHECKS=0`). A mutant that rounds ties away in both `fpNarrow` and `fpu.narrow` (`bff_mut_narrow_tieaway.patch`) passes sm-verilator, pipeline, pack and gemm, and fails the gate through `aril-fpu` (rne_ref: 32,512 tie classes wrong) and `aril-narrow` (corners 3F808000, BF808000).
- **`layer_bias_special`** in `make gemm FMT=bf16` (so in the gate, QUICK too): one bf16 layer on sienna_layer whose bias row holds +0, -0 and subnormals; bit-exact, and `a_bias_widened` must not fire. `model_runner` takes a test-only job key `bias_words` (the layer file would otherwise flush subnormal words). The layer flush mutant fails only this case.
- Collapse-k 0 in bf16, bit-exact at N 16 (84/84, bff_sm16ck0_bf16).

### Deferred minors

- `gen_mm_range_edge` needs N >= 8 (N 4 is weaker, N 2 crashes the generator); no flow runs the mesh below N 8.
- The bf16-sum contrast for `mm_range_edge` is shown on the model only (no RTL mutant with bf16 sums).
- TB_sienna_multi's and TB_sienna_model's bf16 bias widen are linted, not simulated (no bf16 flow builds them; RtlSets is fp32 only).
- `a_bias_widened` reuses the loader's own select terms, so it checks the value, not the half or the timing; TB_sienna_top's bias check names SystolicMesh internals (a rename fails at elaboration, not silently).
- No mutant has fired the bf16 branch of the GEMM error gate (the same line fp32 gates on).
- fpNarrow is verified at MAN_W 7 and 23 only.
- Tail elements were not counted on the perf stimulus (the perf moves are explained per set above, not predicted per config).
- `Common/models/__pycache__/` is untracked in both ArithmeticLibrary clones (left in place).

Open at close (the controller's): the merge on Soham's go-ahead, the N = 64 bf16 sweep (regression, perf, models) with the gate, Ruling 2's tags, the README throughput and latency tables after that sweep, and the push (innermost first; ArithmeticLibrary by its SSH URL).
