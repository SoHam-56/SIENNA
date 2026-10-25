---
name: sienna-int8-residual
description: Use when designing, building or verifying SIENNA's int8 residual ADD (the generic affine int8QuantizedAdder gated on the true function - add_lanes, the hold/add set pair through the identity pass, hold_mem, the ADD words add_words_t) or the int8 GPNAE sigmoid breakpoint ladder for |x| >= 2.5; the pre-synthesis RTL part of int8 sub-project 2b.
---

> **Names since the restructure (Batch A Stage 4, 2026-10-24).** `sienna_top` is the one top: today's pipeline body is `src/pipeline/sienna_pipe.sv`, `sienna_layer` is `sienna_sched` under `sienna_top` with `SET_PORT = 0`, `sienna_multi` is `sienna_top` with `SET_PORT = 1, COPIES > 1`. The format is the build's define (`make FMT=` gives `+define+FMT_<FMT>`): `EXP_W`, `MAN_W`, `IS_INT`, `ACC_W` and `OUT_W` are gone from SIENNA's RTL and test_config_pkg, in favour of `sienna_fmt_pkg`'s constants (`DATA_WIDTH`, `ACC_DATA_WIDTH`, `RESULT_DATA_WIDTH`, `HAS_REQUANT_EPILOGUE`, `HAS_RESIDUAL_ADD`, the latencies) and bundles (`lane_entry_t`, `requant_scale_t`, `add_params_t`). File:line references below predate it.

# int8 pre-synthesis RTL: residual ADD and the sigmoid ladder (spec; decisions approved by Soham 2026-10-10)

Read from SIENNA main fce13e8 (clean clone `/proj/scratch/spramanik/n8_fix`), GPNAE afd786b, ArithmeticLibrary 1d04a26,
TensorFlow 2.18.1 in `sienna_jobs/venv`. Nothing was built or simulated. The sigmoid numbers below come from evaluating
`gpnae_model` (numpy, bit-exact with the RTL per G2) in this investigation, not from RTL runs.

**Status: implemented and verified 2026-10-18 on branch `int8_residual` in all four repos, at N = 8 and 16 at every tile size (gate G4: PASS, run rs2_g4, `sienna-report/history/2026-10-18_sienna_int8_residual_g4.txt`); N = 32 and 64 run in the final sweep after the restructure.** The spec below is kept as written; "As built" lists where the build departs from it.

**Status (Soham 2026-10-17, Task 2.6b): everything generic, in logic and names.** Decision 1's TFLite bit-exactness is reversed: the ADD is `int8QuantizedAdder` (ArithmeticLibrary `QuantizedAdder/`: y = clamp(sat10(round half up of ((q1 - zp1) m1 + (q2 - zp2) m2) 2^-shift) + zpo), m1 and m2 unsigned 32-bit on one shared shift, 6 stages, qadd_lat() 6) and the requantize is `int8Requantizer` (`Requantizer/`: 24-bit mantissa, 6-bit right shift, 4 stages), both gated on the true function (|hw - t| <= max(1% |t|, one step)); the host encodes real scales with `ipu.encode_scale` and `ipu.encode_add` (which refuses max(s1, s2) / s_out >= 2^23). `add_lanes` takes one `add_words_t` (`src/sienna_add_pkg.sv`: zp1, zp2, m1, m2, shift, zpo, amin, amax; 110 bits) and has `busy_o`, which `sienna_top`'s `rq_room` and `rq_drain` read. G0b, D1's left shift and Prepare, P-11 and P-13 are history; agreement with the TFLite interpreter is a reported count, not a gate. Design: `sienna_report/int8_generic_design.log`; the plan's Task 2.6b and its re-scoped Tasks 2.7-2.12.

## As built (2026-10-18)

Departures from the spec, each with its reason (plan `implementation-plan.md`, decisions P-1 to P-14):

- **Generic units (Task 2.6b, Soham 2026-10-17, forks F1-F5).** Decision 1 is reversed: `int8QuantizedAdder` and `int8Requantizer` replace `tfliteAdd` and `tfliteRequant`, both gated on the true function (|hw - t| <= max(1% |t|, one step); activations at 10%). G0b and the rounding pin are history; agreement with the TFLite interpreter is reported, not gated. G4's criterion became F4: cycles identical to the base, every int8 word gated on the true function, changed int8 words counted per test.
- **P-1** `qadd_lat()` (6), not `add_lat()`: the name was taken by the float adder's latency.
- **P-2** G0b lived in ArithmeticLibrary beside the unit (history since Task 2.6b).
- **P-3** an add set's lanes take `gp_zin_i` = the ADD's zp_o (`g_add_zpo`); mutant `m_zin` fails `truth: activation/tanh` (rerun at G4: rs2_g4_ctl_zin).
- **P-4** the polynomial fit still scores tables on the 2a lane, so `poly_coeffs_int8.mem` is byte-identical.
- **P-5** two rescales per threshold in the ladder edge sets.
- **P-6** (Ruling F26, spec row D5) the device-protocol half of the host interface: the layer file's `A` line, `write_layer` from `job["add"]`, `RtlLayer.run_job`; model lowering of the ADD stays 2b.
- **P-7** set-level tests in `TB_sienna (pipeline)` (nine `int8_residual_*` tests; `_one_pair` added in Task 2.9), layer cases in `sienna_layer` / `TB_sienna (layer)` (`regression.py --action residual`: resnet_relu, deep and shallow cached and uncached, tiles, relu6_conv_linear_add, tanh_after_add).
- **P-8** exhaustive ADD pairs checked from a dump (history with G1).
- **P-9** reversed by Task 2.10b: `--action perf` models the residual tests; validated per test against measured cycles (PERF MODEL: PASS, 805 test-points; control `m_perf_sig` fails).
- **F3** every stage of a residual layer is gated from `TB_sienna (layer)`'s per-stage dumps (the conv's requantize, the identity requantize, hold_mem, the ADD, the lane).
- **D6 (review row)** the set-level tests, the layer cases and the TFLite blocks run in both mesh modes (collapse-k 1 and 0) and at random power-up.
- **Changed words against main 9fa414e** (G4, N = 8 and 16, every tile size; per test in the G4 report): fp32 only 3 words in each of 2 mixed tests with tanh at N = 16 (T_TINY, Task 1.16); bf16 words in 22 tests with a sigmoid, tanh or SELU set (134,112 words over 154 test-points) and their cycles in 118 test-points (Decision 5: the fp32 tail and refits); int8 cycles and results identical everywhere, words changed in 238 test-points: the 6 `req_random` tests 157,649 words (redrawn for the new units), 13 sigmoid and SELU tests 14,053 (the ladder, Task 1.16's SELU, rounding), 20 others 1,030, at most 39 in one test-point (round half up at ties). Every changed word passes the true-function gates.
- **Crosses (spec Risk 3)** hold done with an L3 grant: reached at 6 of 7 points (0 at N16 T16); hold done with a bank done: unreachable by design (`a_hold_done_no_bank`, shown firing by a forced mutant); an add set's last beat with the next accept: excluded by design (`rq_room` reads `busy_o`; `a_add_last_no_accept`, shown firing by `m_roomonly`).
- **Depth (Yosys dchr, topological levels, not timing; information only).** Generic units: `int8Requantizer` 45, `int8QuantizedAdder` 34, `add_lanes` with sienna_top's drivers at N = 32 35 (stage 1 from hold_mem's read mux 16). History: `tfliteAdd` 52 against `tfliteRequant`'s 53; `add_lanes` with drivers 54.

## Goal

(Goal 1 as written is superseded by Task 2.6b, 2026-10-17: the residual ADD is the generic `int8QuantizedAdder`, gated on the true function, |hw - t| <= max(1% |t|, one step); agreement with TFLite's ADD is reported, not gated.)

1. History: int8 residual layers bit-exact against TFLite's int8 ADD (reference kernels), so ResNet-8's three ADDs run on SIENNA.
   `pretrainedResnet.tflite` (float, checked): ops 3, 7 and 11 are ADD with fused ReLU. Each adds a conv with no activation
   (ops 2, 5, 9) to an earlier int8 tensor: the stem's output (block 1), or a 1x1 stride-2 shortcut conv (ops 6, 10). This
   lifts `a_int_no_residual` (`sienna_layer.sv:553`).
2. int8 sigmoid within GPNAE's tolerance (rel <= 6.25% or <= 1 LSB) at every input. Today 3,370 of 65,536 dense Q4.11
   inputs and 34 of 1,280 gated inputs miss it.

Out of scope: `model_runner` lowering (2b, after synthesis), average pool, softmax, depthwise. Unchanged: fp32 and bf16,
the mesh, `tfliteRequant`, and GPNAE's Taylor lane `gpnae.sv`.

## Decisions

**D1, ADD semantics (reversed 2026-10-17: the ADD is the generic int8QuantizedAdder; this is the record)** (TFLite `kernels/add.cc` Prepare, `reference/integer_ops/add.h`). left_shift = 20.
d_i = q_i - zp_i; scaled_i = MBQMSmallerThanOneExp(d_i << 20, m_i, s_i); out = clamp(MBQMSmallerThanOneExp(scaled_1 +
scaled_2, m_o, s_o) + zp_o). Prepare computes twice_max = 2 max(s1, s2), m_1 = s1/twice_max, m_2 = s2/twice_max and
m_o = twice_max/(2^20 s_out), each through QuantizeMultiplierSmallerThanOneExp (shift <= 0). The clamp is the fused
activation's range. The flatbuffer stores none of these: the host must derive them exactly. In DOUBLE,
MBQMSmallerThanOneExp is RDBPOT(SRDHM(x, m), -s), which is `ipu.mbqm(x, m, s<=0, "DOUBLE")`; under TFLITE_SINGLE_ROUNDING
it is MBQM itself. **Not confirmed against installed source:** the TF 2.18.1 wheel ships no TFLite kernel headers. G0b
therefore pins these semantics against the interpreter before any RTL is written, as G0 pinned DOUBLE. Value ranges:
|d| <= 255, |scaled| < 2^28 and |sum| < 2^29, so nothing wraps and SRDHM's INT_MIN saturation cannot occur.

**D2, a separate elementwise op on two int8 tensors.** TFLite requantizes the conv to its own output scale, with its
clamp, before the ADD rescales it. The ADD therefore cannot be folded into the int32 sum bit-exactly.

**D3, placement: option (b).** A new ADD stage sits between `requant_lanes` and the lane feed / bypass write: `add_lanes.sv`,
with one ArithmeticLibrary `tfliteAdd` per lane. Rejected alternatives:
- (a) A mode inside `requant_lanes` / `tfliteRequant`. It changes a G1-verified unit on every int8 set's path, and
  `req_lat()` feeds the drain-overlap chain, which has 0 margin (`a_rq_one_set`).
- (c) The host. It needs no RTL, but ResNet-8 would no longer run end to end on SIENNA, and every ADD layer's output would
  make a round trip through the host. This stays the fallback if area forces it.

**D4, residual source: the existing identity pass, split into two result sets.** `sienna_layer`'s stream order is kept:
the depth tiles, then the residual tile on A, with identity rows on north. In int8, ONE = 1, so the identity pass yields
the residual code q2 exactly in int32. For int8 residual layers, each output tile becomes:
- A **hold set**, the last depth pass. It closes the conv's accumulate group, is requantized with the conv's M_c, sh_c,
  zp and clamp, and its beats go to a hold buffer. It gets no bank and produces no output.
- An **add set**, the residual pass. It is its own group, with no bias and an identity requantize (mult 2^30, shift +1,
  zp 0, clamp [-128, 127], exact in both roundings). Its q2 meets q1 from the hold buffer in `tfliteAdd`, and the result
  continues on the normal ReLU / linear bypass or GPNAE lane path, so a lane activation after an ADD comes free.

Rejected alternatives:
- Residual first in the stream: the stream order would depend on the format, and float must keep it last to keep its
  float sums bit-exact.
- A new host link into the activation stage: it still needs an N^2 reorder buffer, plus a new credit link.
- Read-modify-write in an activation bank: it saves the N^2 bytes, but changes L6 bank ownership and `act_wr` toggling in
  a credit-verified stage. This is the area fallback (Q1).

**D5, latency: only add-set beats take the ADD path.** Every other beat keeps `req_lat()`, so the cycles and words of
every non-ADD int8 set stay unchanged. The set after an add set gets its L3 grant only once `add_lanes` has drained
(`rq_room` and `rq_drain` widened), so the two paths never present a beat in the same cycle. Rejected: a uniform
`req_lat() + add_lat()` for all int8 sets. It costs about 6 cycles on every activation-bound int8 set and breaks
`a_rq_one_set`.

**D6, ADD constants come in the add set's own sideband,** stored per set id like `s_zp`. zp_1 is carried explicitly
(sienna_layer sets it to `cfg_req_zp_i`). It is not read from the hold set's id, because that id can be reused once the
hold set completes. Rejected: overloading the add set's unused per-channel mult / shift words. It would save about
2.6 Kbit but couple unrelated meanings.

**D7, sigmoid: an exact breakpoint ladder for |x| >= 2.5.** Today the lane computes P (degree 3) on |x| for |x| <= 3.5,
takes 1 - P for x < 0, and saturates beyond 3.5. The misses split two ways:
- 3,346 of the dense misses are at x < -3.5, where the lane outputs 0 and the exact output is 2-7 LSB.
- 24 more are inside the fitted range, at -3.39 <= x <= -2.61 (d = 2 LSB on goldens of 8-18).
- Of the 34 gated misses, 31 come from the saturation and 3 from the polynomial. The sienna-int8 skill's "from saturation
  below x = -3.5" is therefore incomplete.

The fix applies for |x| >= 5120 (Q4.11). It computes k = 19 - #{t_i <= |x|} over 19 constants, t = {5228, 5350, 5479, 5616,
5761, 5916, 6082, 6261, 6456, 6669, 6905, 7170, 7471, 7821, 8240, 8763, 9460, 10515, 12773}. These are the codes where
round(256 sigmoid(-|x|)) steps. The output is k - 128 for x < 0 and min(127, 128 - k) for x > 0. This is exact because
sigmoid(x) = 1 - sigmoid(-x) and no Q4.11 input is a rounding tie (the nearest is 2e-5 LSB away).

Model results with the ladder:

| Measure | Result |
|---|---|
| Dense inputs outside tolerance | 0 of 65,536 |
| Gated inputs outside tolerance | 0 of 1,280 |
| Ladder region against `exact_int8` | bit-exact |
| Worst error elsewhere | 2 LSB (passes on the relative bound) |

Where the ladder starts:
- Starting it at 3.5 leaves 24 dense and 3 gated misses.
- Starting at 2.6 reaches 0 with no margin.
- Starting at 2.5 gives 205 codes of margin, so the ladder starts there.

Rejected alternatives:
- Higher degree: unrefined Chebyshev fits on [0, 3.5] at degree 4 and 5 miss 2,832 and 2,097 inputs, even with the
  ladder tail. Q4.11 does not hold the extra terms. This is not conclusive without refinement, but the ladder makes the
  question moot.
- An exp tail: `gpnae_tail` is float-only and is not instantiated in int8.
- A second polynomial segment: `barrel_mac` takes one coefficient base per group of 16, not one per element.

## RTL changes

**ArithmeticLibrary (innermost, pushed first).** `Add/src/tfliteAdd.sv` takes parameter ROUNDING (default
`sienna_fmt_pkg::REQ_ROUNDING`). Its ports:
- Control: `clk_i`, `rstn_i`, `valid_i`.
- Inputs: `q1_i`, `q2_i` [7:0]; `zp1_i`, `zp2_i` signed [7:0]; `m1_i`, `m2_i`, `mo_i` signed [31:0] (Q0.31); `s1_i`,
  `s2_i`, `so_i` signed [7:0] (asserted in [-31, 0]); `zpo_i`, `min_i`, `max_i` signed [7:0].
- Outputs: `result_o` [7:0], `done_o`.

The unit takes its constants at stage 1 with each beat, as `tfliteRequant` does. Stages (DOUBLE):
1. d_i, and SRDHM computed as (d_i m_i + 2^10) >>> 11 (the 2^20 factor cancels, so these are 9 x 32 multipliers).
2. RDBPOT by -s_i.
3. The 30-bit sum.
4. SRDHM(sum, m_o), a 30 x 32 multiply.
5. RDBPOT by -s_o.
6. + zp_o, then max(min), then min(max), TFLite's order.

`add_lat()` = 6 goes in `sienna_fmt_pkg` as an estimate. The same Yosys/ABC depth probe used for the requantize split sets
the final number, with the target that no stage is deeper than `tfliteRequant`'s worst (about 46 levels).

**GPNAE (SIENNA's int8 poly lane only).** In `gpnae_poly_int8.sv`, sigmoid's `sat_buf` comes to mean |x| >= 5120, and
G_POST (line 474) writes `sat ? sig_ladder(x_cur) : quant_sig(y_s)`. The 19 thresholds are localparams, followed by a
thermometer count and k - 128 / min(127, 128 - k). The change adds no state and no cycle. If the depth probe flags the
G_POST cycle, k moves into a 5-bit buffer written beside `x_buf`, still at no cycle cost. `poly_coeffs_int8.mem`,
`barrel_mac`, the float lanes and `gpnae.sv` stay byte-identical.

**SIENNA:**
- `sienna_set_side.svh` gains `res_hold`, `res_add`, `add_zp1`, `add_zp2`, `add_m1`, `add_s1`, `add_m2`, `add_s2`,
  `add_mo`, `add_so`, `add_zpo`, `add_min` and `add_max` (+162 bits on L0).
- `sienna_layer.sv`:
  - Drops `a_int_no_residual`.
  - Adds `cfg_add_{zp2, m1, s1, m2, s2, mo, so, zpo, min, max}_i`, taken with `cfg_load_i`.
  - For int8 residual layers, sets `p_acc = (is_p < dt - 1)` (line 333).
  - Marks pass dt-1 `res_hold`, with act = linear and the layer's requantize zp and clamp.
  - Marks pass dt `res_add`, with identity requantize words, the layer's act and GPNAE words, and the ADD constants.
  - Float layers are unchanged, and a packed residual layer is still rejected.
- `add_lanes.sv` (new) holds NUM_LANES `tfliteAdd` units. Each beat's tags (bypass, bank, idx, pack) ride beside it for
  `add_lat()` cycles.
- `sienna_top.sv`:
  - Adds `hold_mem[PER_LANE][NUM_LANES]` x 8 bits (N^2 bytes; a register array or a 1R1W SRAM). Hold beats write it in
    beat order as they leave requantize, and add beats read it in the same order. Hold beat j always precedes add beat j,
    because the two sets have consecutive ids and `a_rq_one_set` orders them.
  - A hold set does no bypass write, no `bank_done` and no `act_wr` toggle. It releases its L6 reservation once its last
    beat is in `hold_mem` (`l6_rsv`, line 803), adds to `gp_sets` then, and pooling passes it as a null (`p_null`,
    line 748, on `set_accum || set_hold`). It completes like a partial set.
  - Add-set beats go requantize -> `add_lanes` -> a merge mux into the existing `fill_v` / `fill_d`.
  - An add set with ReLU or linear drops to `add_zpo` (`p_zp`, line 905).
  - No new credit link: `hold_mem` is internal to the activation stage, and set order fixes its occupancy.
  - New simulation assertions, each shown to fire by a mutant: `a_hold_then_add`, `a_hold_push_empty`,
    `a_add_pop_full`, `a_fill_one_source`, `a_hold_no_bank`, `a_add_par_stable` and `a_float_no_add`.

**Host interface (2b implements it).**
- int8 residual layer files gain an `A zp2 m1 s1 m2 s2 mo so zpo amin amax` line after `Q`.
- The residual tile carries the residual tensor's int8 codes.
- The conv's clamp is the conv's own range (NONE: -128..127), and the ADD's clamp is the ADD's range (ReLU:
  max(-128, zp_o)..127).
- The ADD is symmetric, so the host maps TFLite's input order; op 3 lists the residual first.
- The float lowering's "depths side by side" fusion of the 1x1 shortcut into the main conv is invalid in int8, because
  each conv requantizes separately. The shortcut becomes its own layer.
- The host derives the ADD words with `ipu.add_params`.

## Bit-exact models

- `ipu.py` (ArithmeticLibrary) adds:
  - `tflite_add(q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, amin, amax, rounding)`, built on `mbqm`.
  - `quantize_multiplier_smaller_than_one_exp`.
  - `add_params(s1, s2, s_out)`, which repeats Prepare in doubles from float32 scales.
- `regression.py`'s int8 golden computes q1 = `ipu.requant(acc, ...)`, then y = `ipu.tflite_add(q1, codes, ...)`, then
  the existing lane, pool and dropout golden.
- `gpnae_model.LaneInt8.run` (code 2) adds the ladder, with `SIG_LAD_X0` and `SIG_LAD_T`. `fit_poly_coeffs.py` computes
  the thresholds from sigmoid, and the int8 model check fails if they differ from the model's or the RTL's literals.

## Verification (bottom-up; N <= 32 while testing, one N = 64 sweep after check-in; every test gates on a reference)

- **G0b, ADD oracle.** Build single-op int8 ADD models with the converter: random s1, s2 and s_out (2^-10 to 2^2, scale
  ratios up to 2^10), every zero point, and fused NONE, RELU and RELU6. Feed each model all 65,536 (q1, q2) pairs. Compare
  the interpreter (BUILTIN_REF) with `ipu.tflite_add` on `add_params`; the gate is 0 mismatches. Record the rounding
  (expected DOUBLE, the conv's pin; otherwise the ADD gets its own pin) and report the optimized resolver. Save the
  outputs as G1 vectors.
- **G1, `tfliteAdd`.** Exhaustive 65,536 pairs at corner constants: every s in [-31, 0]; m = 2^30 and 2^31-1;
  zp = -128 and 127; tight clamps. Then 10^6 random cases and G0b's interpreter outputs replayed, in both roundings.
  Mutants that must fail: left_shift 19, the rounding variant swapped, the clamp applied before + zp_o, s_o's sign
  flipped.
- **G2, sigmoid.**
  - The RTL is bit-exact against the model on the 5 gated cases, on threshold cases (a rescale for each t_i that lands
    codes on t_i - 1 and t_i, both signs) and on a 64-scale sweep.
  - Against exact, the model gives 0 of 65,536 dense and 0 of 1,280 gated outside tolerance. This becomes a **hard gate**
    for sigmoid; tanh is already at 0, and SELU is gated on every int8 set (Task 1.16's block-float post stage; no 487.29 limit).
  - tanh, SELU, the float lanes and `gpnae.sv` give identical outputs (diff empty).
  - Mutants: a threshold off by 1, a ladder starting at 3.5, the positive branch computed as k - 128.
- **G3.** The mesh does not change. A diff of the SystolicMesh tree confirms it, so G3 does not rerun.
- **G4, SIENNA int8.**
  - New tests: `int8_residual_relu` (ResNet shape), `_linear`, `_tanh` (an ADD then a lane), `_deep` (dt >= 3, cached
    and uncached), `_tiles` (rt, ct > 1), `_train` (drops to zp_o), `_then_relu_nopool` (an add set draining into a ReLU
    set) and `_two_layers` (two ADD layers with different constants). They run with `zp_random`, `req_random` and random
    ADD scales at N = 8, 16 and 32, in both mesh modes and at random power-up, and must be bit-exact.
  - A residual-block TFLite model (conv 3x3 with no activation, + ADD(x), ReLU), and its 1x1-shortcut variant as two
    layers, run through `TB_sienna (layer)` and must equal the interpreter bit for bit. They use regression.py's TFLite path,
    not model_runner.
  - Every existing int8 test keeps identical cycles. Its words are identical except in sigmoid tests with some
    |x| >= 2.5; that count is reported.
  - fp32 and bf16 stay identical in results, cycles and words.
  - `TB_sienna (pipeline)` crosses hold / add pairs with partial sets, ReLU drains, lane sets and a full bank. Mutants that
    must fail: a hold read one beat off, zp_1 taken from the add set, the drop value taken from `pe_zp`, and `rq_room`
    left unwidened (`a_fill_one_source` fires).

## Area / latency (estimates, no synthesis)

| Item | Count | Estimate |
|---|---|---|
| `tfliteAdd` | NUM_LANES: 32 at N <= 32, 128 at N = 64 | 2 x (9 x 32) + 1 x (30 x 32) multipliers (about 1,540 partial-product bits, against `tfliteRequant`'s 1,024) and 3 rounding shifters: about 1.5-2x one `tfliteRequant` (estimate). Measured (Yosys dch, generic cells, rs2_yd_t24): `add_lanes` at 32 lanes 441,939 comb + 7,330 FF, about 1.46x `requant_lanes` (302,999 + 6,344 at N = 16), below the estimate; one `tfliteAdd` alone has 473 FF, so without Yosys merging the lanes' identical word registers it would be about 15.1k FF |
| `hold_mem` | 1 | N^2 x 8 bits: 2 Kbit at N = 16, 8 Kbit at N = 32, 32 Kbit at N = 64 |
| ADD constants | 16 ids + 1 copy | about 2.7 Kbit of flops; L0 widens by 162 bits per `LINK_STAGES` register |
| Sigmoid ladder | per lane | 19 constant compares of 15 bits, a 19-to-5 count, an 8-bit subtract and a mux: a few hundred gates, small next to the lane's 16 x 16 and 24 x 24 multipliers |

Latency:
- Non-ADD int8 sets: 0 cycles.
- Add-set beats: `add_lat()` (about 6) more.
- Per output tile of an ADD layer, the activation stage does two wide reads (hold + add) instead of the float residual's
  one, and the next set waits about 8 cycles. The mesh passes per tile are unchanged from the float residual (dt + 1).
- Sigmoid: 0 cycles.

## Risks

1. The ADD semantics are not verified locally. No RTL is written until G0b passes.
2. The int8 ResNet-8 model is not in the repo. Its structure comes from the float model, assuming conversion keeps it;
   G4's TFLite residual-block models check the converter's actual form.
3. The hold set is a new kind of set in a credit-verified stage, the coincidence-invariant bug class. TB_sienna (pipeline)
   targets the same-cycle cases: a hold done with a lane set's `bank_done`, the L6 release with an `l3_grant`, and an add
   set's last beat with the next accept.
4. `hold_mem`'s ordering rests on `a_rq_one_set`, which has 0 margin. If that invariant is relaxed later, `hold_mem` needs
   per-beat set tags.
5. Sigmoid outputs change, so 172 of the 1,280 gated outputs and the report's TFLite logistic agreement move. The goldens
   regenerate from the model.
6. Quota: G0b's vectors (about 4 x 10^7 outputs) are kept compressed, after a quota check.

## Decisions taken by Soham (2026-10-10)

1. ADD area: **full rate**. One `tfliteAdd` per lane (32 at N <= 32, 128 at N = 64) and an N^2-byte `hold_mem`; no throughput cost.
2. Sigmoid ladder: **both signs**. The positive tail becomes exact too; 172 of the 1,280 gated outputs change and their goldens regenerate from the model.
3. Generality (Soham: the hardware must stay open to every model and every input; the TFLite part superseded by Task 2.6b: the ADD is the generic int8QuantizedAdder on the true function): the ADD implements TFLite's int8 ADD for any per-layer scales, zero points and clamp (runtime constants, verified over every input pair, every shift and the extreme multipliers), and adds any two int8 tensors (one through the mesh, one through the identity pass); the ladder's thresholds come from sigmoid itself and are verified over all 65,536 Q4.11 inputs, not model data.
4. bf16 sigmoid (Soham, 2026-10-10, after this spec): fix the bf16 poly lane's sigmoid error near the fitted range's edge (18.1% worst at x = -3.375, bit-exact model) before synthesis, as plan Task 1.14. This supersedes "fp32 and bf16 unchanged" for bf16 sigmoid only; fp32 and every other bf16 path stay unchanged, and `gpnae.sv` stays byte-identical. fp32_down's missing guard bit stays as is (Soham: leave it).
5. bf16 activations (Soham, 2026-10-11, after the Task 1.14 study, sienna_report/bf16_sigmoid_study.log): over every finite bf16 input the lane's sigmoid was 18.06% worst in [-8, 8] and 106.55% beyond (the tail's bf16 e^-|x| and the 8-bit 1 - P), tanh 8.51% (the fit's c0 = 0.9297) and SELU 2.49% (x < 0: c0 1.35% low) and 1.08% (x >= 0: lambda in bf16 plus truncation). Fixes, all in bf16 only: sigmoid for every x < 0 takes the exact tail computed in fp32 with two Newton steps for 1/(1 + e) (study candidate d3: 0.772% worst on every bf16 input; 571-1174 cycles per 16-element group on negative-heavy mixes against 194; about +5.5k gates and +1.4k flops per lane, estimated), with the new tail states behind a parameter so the fp32 and int8 lanes' words and cycles stay identical; tanh is refit (measured 1.325% best, the published bf16 tanh coefficients change, Soham's go-ahead); SELU computes lambda*x as x + x*(lambda - 1) for x >= 0 (0.720%) and its x < 0 side is refit or corrected as far as the table allows, with the measured result reported. The bf16 gates are sigmoid <= 1% wherever |exact| >= 2^-126 (|out| <= 2^-126 below), tanh and SELU at their measured worst. gpnae.sv, barrel_mac, the fp32 and int8 lanes stay unchanged; bf16 tests that use sigmoid, tanh or SELU change outputs, and sigmoid's cycles, where these fixes change them (G4 compares them like int8's sigmoid tests).
6. Accuracy and structure (Soham, 2026-10-11): every activation in every format gates on the true function (numpy float64), never on a model of the algorithm: relative error <= 1% with floors |err| <= 1 LSB for int8 and |out| <= 2^-126 for bf16 and fp32 (the units flush subnormals); matmul and accumulation outputs <= 1% of sum |a b|; every input for int8 and bf16, dense plus random for fp32; a miss fails the run, nothing is reported only; every lane that misses is fixed (bf16 tanh and SELU, the int8 activations), no exceptions; int8 ADD stays bit-exact to TFLite's kernel (Decision 1; superseded 2026-10-17 by Task 2.6b: the ADD is generic and gated on the true function). After this ADD integration, a restructure gives each repo one top module (sienna_top, SystolicMesh, gpnae) with a DATA_WIDTH datapath and every math module selected at compile time by a define (fp32 default), the same mechanism for every precision with each conversion at its own point (identity where a precision does not convert); gpnae.sv becomes barrel_mac plus one coefficient set per activation and precision plus the tail, and the e^x reshaping units (sigtan, SeLu, the divider, fp32_down, fp32_up_down) go. The restructure has its own spec (restructure_inventory.log is its evidence).
