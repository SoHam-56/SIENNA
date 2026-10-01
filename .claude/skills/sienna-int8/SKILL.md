---
name: sienna-int8
description: Use when building or verifying SIENNA's int8 build (sub-project 2 of sienna-uniform-format) - int8 mesh with int32 accumulation, the TFLite-exact requantize stage, fixed-point GPNAE, int8 pooling and dropout, the TFLite oracle, or the int8 golden models. Also use when someone asks how SIENNA's int8 matches TFLite, why GPNAE is fixed point in int8, or what 2a covers against 2b.
---

# SIENNA: the int8 build (sub-project 2a)

**Status: 2a implemented and verified 2026-09-30 on the `int8` branch of all four repos, at N = 8-32; N = 64 is
deferred to one final sweep after check-in (Soham 2026-09-29); 2b not started.**
Gate reports, verbatim in the `sienna-report` skill's `history/`:
- G4 SIENNA: `2026-09-30_sienna_int8_g4.txt` (farm runs `g8*`; GATE G4: PASS; regression 35/35 exact at N = 16 and 32
  in both mesh modes and at random power-up, TFLite int8 layers equal to the interpreter at N16 T4 and N32 T8, fp32 /
  bf16 identical in results, cycles and output words; N = 64 deferred to the final sweep)
- G3 SystolicMesh: `2026-09-29_mesh_gate_int8.txt` (Task 15; VERDICT: PASS at N = 8-32, every tile size, both collapse
  modes, random power-up; fp32 / bf16 cycles identical; N = 64 deferred to the final sweep)
- G2 GPNAE: `2026-09-29_gpnae_gate_int8.txt` (Task 12; VERDICT: PASS for correctness, the lane bit-exact everywhere;
  accuracy against GPNAE's tolerance over the full Q4.11 range: SELU 0/262144 (over the Q4.11 range only) and tanh
  0/65536 outside, sigmoid 3370/65536 outside, from saturation below x = -3.5. Rerun 2026-10-01 after SELU fix A (farm
  run `gb_c3_g2b`, not in `history/`): positive SELU is exact to lambda * x < 512 (x < 487.29) and saturates beyond;
  the not-gated SELU case s_in = 0.21875 is 0 of 256 outside, worst 1 LSB (was 49 of 256, 102 LSB), and 64 positive
  SELU sets per seed, x = 16 to past the limit, are bit-exact. Open accuracy items: sigmoid below x = -3.5, and SELU
  past x = 487.29 (the host guard rejects it); the int8 regression's `matmul_large_selu` (lane inputs 19-68 at
  N = 8-64, default seed) now has 0 saturated inputs and passes against the unsaturated golden)
- G1 ArithmeticLibrary: `2026-09-29_aril_gate_int8.txt` (Task 8; VERDICT: PASS, all units bit-exact; fxMac sweeps
  2.58e10 results, 0 errors)
- G0 oracle: `2026-09-29_g0_oracle_int8.txt` (Task 2; G0: PASS, ROUNDING: DOUBLE pinned; 24,832,000 outputs,
  0 mismatches against BUILTIN_REF with DOUBLE rounding, SINGLE differs in 11,332; TensorFlow 2.18.1)

**REQUIRED BACKGROUND:** the `sienna-uniform-format` skill (one number format per build; this is its sub-project 2)
and the `sienna-rtl` skill.

## Goal (agreed with Soham, 2026-09-29)

An int8 build of SIENNA whose results are **bit-exact against TFLite's own int8 kernels** wherever TFLite defines the
operation: conv and fully connected layers with per-channel symmetric int8 weights, int8 activations with zero points,
int32 bias, TFLite's fixed-point requantize, and ReLU / ReLU6 fused as the requantize clamp. Results are then
comparable with published MLPerf Tiny int8 numbers.

Split in two: **2a** (this spec) is the hardware and its DV, proven on the regressions and on single-layer TFLite
int8 models. **2b** (own spec, after 2a passes) runs the four MLPerf Tiny int8 models end to end.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Reference (oracle) | TensorFlow installed into `sienna_jobs/venv`; its interpreter with the reference kernels (`BUILTIN_REF` op resolver); its converter for single-layer int8 test models | Soham: TFLite's own kernels, not our reading of them |
| Format selection | `EXP_W = 0, MAN_W = 7` is int8; `DATA_WIDTH = 1 + EXP_W + MAN_W = 8` unchanged; `sienna_fmt_pkg::is_int(exp_w)`; `supported()` accepts (0, 7); anything else fails elaboration with `$fatal` | same two knobs as fp32 / bf16 |
| Accumulate width | `ACC_W` = 32 for int8, `DATA_WIDTH` for floats: PE partial sums, reducer tree, bias input, mesh result memory, wide read into requantize | a dot product of int8 needs int32; the uniform principle's one exception |
| Storage | int8 for host operands, staging banks, weight cache; int8 again from requantize on (lanes, activation banks, pooling, dropout, output) | where int8 saves area and bandwidth |
| Mesh arithmetic | int8 x int8 -> int16 sign-extended into an int32 two's-complement accumulate (wraps like TFLite); 1-cycle registered multiply and add, latencies from `sienna_fmt_pkg`, so U follows | no float unit needed; overflow impossible for K <= 133,000 (127 x 127 per product), documented, no saturation logic |
| Zero points | none in the mesh: weights are symmetric (zero point 0); the input zero-point term -z_a * sum(w_c) is folded into the int32 bias by software; conv padding with z_a is done in the software im2col | TFLite's own algebra; keeps the mesh a plain integer array |
| Requantize | new stage, generated only when `is_int`: per int32 output of channel c, `y = MultiplyByQuantizedMultiplier(acc, M_c, shift_c) + z_out`, then `clamp(y, act_min, act_max)` (int8 range, ReLU, ReLU6 in one clamp) | exactly TFLite's conv / FC epilogue |
| Requantize rounding | whichever of TFLite's two variants (single or double rounding) the installed reference kernels use; found and pinned by a test at G0, never assumed | the variants differ in the last bit |
| Requantize placement | at the GPNAE lane feed: one pipelined unit per lane (~3 stages, 32 x 32 multiplier) on the wide read; result memory stays int32 | unit count independent of T (32 lanes at N <= 32, 128 at N = 64); at the reducers it would be (N/T)^2 units (1,024 at N=64 T=2) |
| Requantize parameters | per set, like the bias: `M_c`, `shift_c` for the set's N output channels; layer-wide `z_out`, `act_min`, `act_max` in the set's configuration | sets of one layer share them; channels are the set's columns |
| Activation engine | GPNAE, fixed point (Soham's choice over a 256-entry LUT): the same Horner polynomial and barrel_mac / gpnae_poly structure, with 16-bit integer multiply and add units | one activation engine for every format |
| GPNAE number format | input int8 q -> `x = round(((q - z_in) * M_x) >> sh_x)`, int16 Q4.11 (range +/-16, saturating); 16 x 16 -> 32-bit products shifted back to Q4.11; 32-bit add | 8-bit intermediates cannot hold a polynomial's terms; 16 bits is 8x finer than any int8 output |
| GPNAE coefficients | re-encoded to 16-bit fixed point by `fit_poly_coeffs.py` into a new `poly_coeffs_int8.mem` (degree as the fit needs, as bf16 needed); fp32 / bf16 tables never change | published work; same kind of change as bf16's refit |
| Beyond the fitted range | the float lane's own ranges (tanh abs(x) <= 4, sigmoid abs(x) <= 3.5, SELU x >= -4); beyond them the lane outputs the saturated int8 value (tanh +/-127/128, sigmoid 0 / 255/256, SELU -lambda*alpha); `gpnae_tail` not instantiated in int8 builds | Soham 2026-09-29: same design as the other formats; the saturation error is judged by the tolerance (sigmoid ~2.8%, SELU ~1.9% at the thresholds, estimates) |
| GPNAE output | tanh: y * 128, zero point 0; sigmoid: y * 256, zero point -128 (TFLite's fixed output quantization); SELU: negative branch from the polynomial, positive branch lambda * x exact, per-layer `(M_out, sh_out, z_out)`; ReLU and linear pass through (already clamped) | TFLite's conventions where it has the op; SELU is not a TFLite op |
| GPNAE per-layer parameters | `M_x, sh_x` (input rescale) and SELU's `M_out, sh_out, z_out`, carried with the set's configuration | fixed per layer |
| Max pooling | Maxpool_2D's existing integer compare, pad -128 | exact: max commutes with monotonic requantize and activation |
| Dropout | inference: bypass; training: kept values unchanged, dropped values become the zero point, the 1/keep factor folded into the output scale (the next requantize absorbs it) | Soham 2026-09-29: a literal x2 shift saturates half the int8 range |
| Throughput | one MAC per PE per cycle, as today; int8 alone gives area and power | MAC packing is a later architectural step |

## Verification (bottom-up, a gate per level; the order of the bf16 work)

- **G0, oracle:** TensorFlow installed (quota checked first); requantize and int8 conv / FC Python reference, bit-exact
  against the TFLite interpreter (reference kernels) on thousands of random cases; the rounding variant recorded;
  single-layer int8 TFLite models (3x3 conv, fully connected) with known quantization generated for G4.
- **G1, ArithmeticLibrary:** each new unit gets the float units' DV (DPI reference TB, Vivado vectors, Makefile).
  int8 x int8 multiplier exhaustive (65,536 pairs); int32 adder corners incl. wrap plus random; 16-bit fixed-point
  multiply / add exhaustive where feasible; requantize unit corners (INT_MIN, largest multiplier, every shift) plus
  10^6 random against the G0 reference. fp32 / bf16 units unchanged.
- **G2, GPNAE:** the fixed-point lane bit-exact against `gpnae_model.py`'s int8 extension; every int8 input (256)
  for tanh, sigmoid, SELU at several input scales; accuracy judged by GPNAE's tolerance, as for fp32 and bf16 (Soham
  2026-09-29): relative error <= max(1%, 8 * eps) = 6.25% (eps = 2^-7) or absolute error <= 1 output LSB, reported,
  never a stop (a miss is an open accuracy item, as bf16's was);
  agreement with TFLite's int8 tanh / logistic reported, not gated. fp32 / bf16 lanes unchanged.
- **G3, mesh:** int8 bit-exact against an integer mesh model (numpy int64 matmul, int32 wrap; integer addition is
  associative, so order does not matter) at N = 8..64, every tile size, both collapse modes, random power-up.
  fp32 / bf16 cycles identical.
- **G4, SIENNA:** the regression in int8 with a bit-exact golden (mesh, requantize, GPNAE, pooling, dropout); the
  single-layer TFLite int8 models through the RTL equal the TFLite interpreter's outputs bit for bit; N / T
  performance sweep; the report's section 4 int8 columns (TOPS = 2 x MACs per second at the assumed clock).
  fp32 / bf16 unchanged.

## Working setup

An `int8` branch off `bf16` in all four repos, in a separate checkout at `/proj/work/spramanik/SIENNA_int8`, so fp32
and bf16 reruns keep snapshotting a clean tree (`snap_launch_tree.sh` with `TREE`). Push innermost first, as always.

## As built (departures from the design above)

- Requantize parameters of an accumulate group come from its activated pass, the last, like `activation_function_i`;
  the bias comes with the first pass (the mesh's rule). Partial passes may carry anything: the regression gives them decoys.
- The requantize stage is `src/requant_lanes.sv` (one `tfliteRequant` per lane, channel `(k * PER_LANE + b) % N`), 3 cycles
  at the lane feed; `sienna_top` holds each set's parameters by set id and copies the per-channel words at `g_accept`.
  It rounds with `sienna_fmt_pkg::REQ_ROUNDING`, G0's variant (Task 3); `regression._check_rounding()` checks that the
  package, `ipu.REQ_ROUNDING` and `rounding.txt` agree.
- `sienna_layer` in int8 always takes a bias beat per column block; its int32 bias, multipliers and shifts come on
  `w_bias_i`, `w_req_mult_i`, `w_req_shift_i` beside it. A residual pass adds raw int8 codes (no rescale: 2b); in
  simulation an int8 layer configured with `cfg_residual_i` fails the assertion `a_int_no_residual` (final fix).
- Dropout in training drops to the output zero point of the set's activation (D-5 as corrected: `req_zp_i` after ReLU
  or linear, 0 after tanh, -128 after sigmoid, `gp_zout_i` after SELU).
- Test stimulus is quantized per accumulate group as TFLite PTQ would (zero points by `rint`, not the converter's nudging);
  SELU's output scale is calibrated from each set's data.
- Task 2: DOUBLE is what TensorFlow 2.18.1's reference kernels (`BUILTIN_REF`) do; the default optimized resolver
  matches SINGLE instead, so the pin is tied to that TensorFlow build.
- Task 10: the int8 degrees are `SETS_INT8 = {1: (0, 2), 2: (9, 3), 3: (16, 3)}` (SELU 2, sigmoid 3, tanh 3), from the
  fit on the dense sweep; Task 12's refit over the full Q4.11 range left the table byte-identical.
- Task 10: `gpnae_model`'s SELU rounding reads `REQ_ROUNDING` from GPNAE's own `sienna_fmt_pkg.sv`, not SIENNA's
  `rounding.txt`, so the GPNAE checkout stands alone.
- Task 11: `gpnae_poly_int8`'s G_RUN leaves on barrel_mac's `done_o` for a one-element group (the float lane's G_RUN,
  copied, hung); the published float lanes' same latent hang is fixed the same way since (Soham 2026-09-30):
  `gpnae_poly` in GPNAE 892eb7c, the Taylor lane `gpnae` in e6014b9, each with SINGLE lines in its TB.
- Task 12: `TB_gpnae_poly` drives writes 1 ns after each clock edge in int8 mode (the zero-delay drive tripped the FIFO's
  full assertion at 32 writes per batch); float modes unchanged.
- Task 16: PINMISSING is fatal in this flow, so `sienna_top`'s ten new inputs were tied off (`'0`) in its four
  instantiations until Task 19 connected them.
- Task 20: the regression's `zp_random` key draws distinct per-set zero points and clamps (`req_random` draws only
  multipliers and shifts); int8 has 35 tests, the plan's 33 plus `int8_mixed_act_train` and `int8_pad_negative_tanh`.
- Task 21: the TFLite path uses only the model's own words: the multipliers and shifts from the G0 npz and the model's
  int32 bias tensor (with the -z_in * sum(w) fold), never the regression's `requant_params` or `fold_bias`.
- Task 22: the performance model counts 3 requantize cycles in int8 (exact against measured); it undercounts GPNAE lane
  time by 50-76 cycles per set in every format, which predates the int8 work and was not retuned.
- N = 64 (Soham 2026-09-29): skipped while testing; G3 and G4 ran N = 8-32, and every N = 64 run (int8 and the fp32 /
  bf16 references, both collapse modes) is one final sweep after check-in.
- Final fix: int8 ReLU relies on the host clamp. `sienna_top` (its `!IS_INT &&` ReLU bypass) and `gpnae_poly_int8` both
  pass ReLU through unchanged, so the requantize clamp must be `[max(req_min, req_zp), req_max]`. The TFLite flow and
  `regression.py` always set it; a host that leaves `req_min = -128` with a zero point above -128 gets linear output.
- Final fix: SIENNA commits 1647456, 58171a1, 980bcf7 and 6f329a1 do not build alone (the plan's RTL-first split);
  skip them when bisecting.
- Final fix: the int8 host lowering (`model_runner`) rejects SELU layers whose input range reaches x = 487.29, where
  lambda * x leaves int32 in 2^-22; the regression's SELU tests print "SELU saturation: k of n lane inputs at
  x >= 487.29" and check any such input bit-exact against the saturating golden. The one check is
  `regression.selu_saturates(mx, shx, z_in, q)`, the lane's own unsaturated rescale of code q reaching
  `gpnae_model.SELU_POS_SAT` (997960 in 2^-11), positive side only.
- SELU fix A (Soham 2026-09-30): the SELU post stage works in 2^-22: x * P and -lambda*alpha * 1.0 as they are, and
  lambda * x on the unsaturated 24-bit rescale (POSTM is `intMultiplier` W = 24), floored by 2^3 and saturated to
  int32; `(gp_mout, gp_shout)` = QuantizeMultiplier(2^-22 / s_out), not 2^-25 as D-4 says. The regression calibrates
  SELU's output scale on lane inputs clipped to [-16, 487.29], not [-16, 16]. Lane latency unchanged.
- D-8 in `gpnae_poly_int8` (2026-10-01): FSM state, counters and valid bits are reset; its data registers (buffers,
  unit operands, element tags, `final_result_o`) load in a reset-free `always_ff`.

## Out of scope (2a)

- 2b: the four MLPerf Tiny int8 models end to end (residual ADD rescale, average pooling, depthwise conv, softmax,
  obtaining the official int8 models).
- Packing several int8 MACs per PE per cycle.
- LUT activations (declined 2026-09-29).
- Quantization other than TFLite's standard one: per-channel symmetric weights, asymmetric int8 activations.
  No uint8, no int16 activations.

## Risks

1. The interpreter's optimized kernels can round differently from its reference kernels: the oracle forces
   `BUILTIN_REF`.
2. The per-lane 32 x 32 requantize multiplier is the largest new unit (128 at N = 64); G4 reports its area estimate
   next to fp32 / bf16. Fallback if too large: time-share it across lanes, at a known throughput cost.
3. An activation may miss the tolerance at every degree 16 bits allow (tanh most likely: its polynomial runs on
   u in [0, 16]); G2 then reports it as an open accuracy item for Soham, as bf16's accuracy was, and work continues.
4. SELU has no TFLite op: its output scale in the regression tests is calibrated from each test's data range, as
   TFLite post-training quantization would.
5. fp32 and bf16 must not move: every int8 path is under generate blocks, every gate reruns both suites.
6. `/proj/work` quota: TensorFlow (~600 MB) and a second checkout; a full quota once truncated builds.
