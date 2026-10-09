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

Cycle counts do not change: `fpMulWiden` takes 3 cycles = today's `mul_lat(8,7)`, the fp32 adder 5 = `add_lat`, so the slot rotation U = 6 is unchanged (survey of 2026-10-08, from the RTL).

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
