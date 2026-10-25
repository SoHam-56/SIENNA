---
name: sienna-uniform-format
description: Use when changing SIENNA's number format - building it in bf16 (or later int8) end to end, adding the format parameter or format package, writing narrow float units (fpMultiplier, fpAdder) in ArithmeticLibrary, making GPNAE, Maxpool_2D or dropout format-generic, or writing a bit-exact golden model for a narrow format. Also use when someone asks why a bf16 run disagrees with fp32, or how precision is chosen per build.
---

> **Names since the restructure (Batch A Stage 4, 2026-10-24).** `sienna_top` is the one top: today's pipeline body is `src/sienna_pipe.sv`, `sienna_layer` is `sienna_sched` under `sienna_top` with `SET_PORT = 0`, `sienna_multi` is `sienna_top` with `SET_PORT = 1, COPIES > 1`. The format is the build's define (`make FMT=` gives `+define+FMT_<FMT>`): `EXP_W`, `MAN_W`, `IS_INT`, `ACC_W` and `OUT_W` are gone from SIENNA's RTL and test_config_pkg, in favour of `sienna_fmt_pkg`'s constants (`DATA_WIDTH`, `ACC_DATA_WIDTH`, `RESULT_DATA_WIDTH`, `HAS_REQUANT_EPILOGUE`, `HAS_RESIDUAL_ADD`, the latencies) and bundles (`lane_entry_t`, `requant_scale_t`, `add_params_t`). File:line references below predate it.

# SIENNA: one number format per build

**Status: uniform bf16 implemented and verified 2026-09-28; int8 (sub-project 2a) implemented and verified 2026-09-30
at N = 8-32 (N = 64 in the final sweep), see the sienna-int8 skill.** bf16 is on the `bf16`
branch of all four repos; it replaced the earlier mixed bf16. Plan and task record:
`implementation-plan.md`. Gate reports (verbatim, in the `sienna-report` skill's `history/`):
- G1 ArithmeticLibrary: `2026-09-28_aril_gate.txt`
- G2 GPNAE: `2026-09-28_gpnae_gate.txt` (correctness passes; bf16 activation accuracy is an open decision, below)
- G3 SystolicMesh and G4 SIENNA: section 2 and 4 of the SIENNA report (`sienna-report` skill); farm runs `g3*`, `g4p_*`
- fp32 before the work: `2026-09-28_baseline_fp32.txt`; bf16 coefficient fit: `2026-09-28_poly_coeffs_fit.txt`

**Since 2026-10-09 bf16 builds sum in fp32** (skill `sienna-bf16-fp32-accum`, branch `bf16_accum` in all four repos, merge
pending Soham's go-ahead): bf16 x bf16 products widened exactly, fp32 partial sums, reducer and bias, each result rounded to
bf16 (nearest even) as the reducer writes it. The rest of this skill still holds; it says where that change applies.

**REQUIRED BACKGROUND:** the `sienna-rtl` skill.

## The principle (agreed with Soham)

Every module works in the same format, chosen per build by parameters, by swapping its arithmetic
units: storage, mesh multipliers and adders, reducer, GPNAE, pooling, dropout. This is how the
published GPNAE was meant to work ("swap the multipliers and adders, change top-level parameters").

Two exceptions, both the accumulation width, where a format cannot hold a sum of products:
1. int8 accumulates in int32 and requantizes back. That is a property of dot products, not a
   mixed-precision choice.
2. bf16 accumulates in fp32 (Soham, 2026-10-08; built 2026-10-09, `sienna-bf16-fp32-accum`): products,
   partial sums, the reducer and the bias are fp32; operands, storage, the result banks, the lanes, pooling
   and dropout stay bf16. Until then bf16 summed in bf16 (uniform), and the cost was not small: ad01's anomaly
   score 2.8x the float model's, GEMM error 50% of the largest output at K = 3072 (Verification, below).

Two sub-projects, in order:
1. **Uniform bf16** (this design). Also builds the per-format plumbing int8 reuses.
2. **Uniform int8**: principles agreed, design not written yet (see the end).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Format selection | `EXP_W`, `MAN_W` parameters; width `1 + EXP_W + MAN_W`; fp32 = 8/23, bf16 = 8/7 | one knob, same as GPNAE's format framework |
| Rounding in new units | **truncate**, exactly like fp32Multiplier/fp32Adder | Soham's choice: match the existing units |
| Unit latencies | from a shared package `sienna_fmt_pkg`, per format | PE, reducer, barrel MAC stop hard-coding 8 and 5 |
| PE partial-sum slots | U = min(K, adder latency + 1) (from the package); 6 in both formats | the slot interleave exists to cover the adder loop |
| fp32 builds | keep fp32Multiplier/fp32Adder unchanged | published units; fp32 must stay bit- and cycle-identical |
| fpMulWiden (exists, bf16 branch) | stays in ArithmeticLibrary, tested; SIENNA stopped using it, then took it back as the bf16 PE multiplier (fp32 sums, 2026-10-09) | uniform replaced mixed; fp32 sums need an exact product |
| GPNAE lane | only `gpnae_poly` (what SIENNA instantiates) | the published `gpnae.sv` lane is out of scope |
| GPNAE coefficients | refit for bf16 into a new `poly_coeffs_bf16.mem`, fp32 table layout, degrees 3 (SELU), 5 (sigmoid), 4 (tanh) with zero leading coefficients | fp32 files never change (published work); the fp32 degrees (8/6/8) gave bf16 worst errors of 5.3% / 21.7% / 331% |
| Golden model for narrow formats | **bit-exact** emulation, not a tolerance | a worst-case bound on a truncated 144-term bf16 sum exceeds the answer, so tolerances catch nothing |

## "Truncate like fp32" precisely

fp32Adder (exists) aligns the smaller significand right with 3 extra low bits, collapses any shift
of 26 or more into a single 1, adds or subtracts in 28 bits, normalizes with a leading-zero count,
and drops the extra bits. fp32Multiplier (exists) takes the top 23 bits of the 48-bit product.
fpAdder and fpMultiplier are those same algorithms with widths derived from `MAN_W`, same ports,
valid/done handshake, flags and special values (subnormal inputs read as zero, underflow flushes to
signed zero, overflow to infinity, NaN to the format's canonical quiet NaN).

Two departures, both deliberate. fpMultiplier uses `karatsubaUnsigned` (8 cycles) only above 12
significand bits; bf16 takes one registered product (3 cycles), so `mul_lat` is 8 or 3. fpAdder
raises overflow only when the normalized exponent is non-negative (fp32Adder's D-1, below), so at
(8, 23) the two adders agree on every result and differ on the overflow flag of 118 test vectors.

## Module changes (uniform bf16)

- **ArithmeticLibrary**: new `Multipliers/FP/src/fpMultiplier.sv`, `Adders/FP/src/fpAdder.sv`,
  parameterized by `EXP_W`/`MAN_W`, fully pipelined (one op per cycle).
- **Mesh**: PE uses fpMultiplier + fpAdder (fp32 builds keep fp32Multiplier/fp32Adder in a
  `G_FP32` generate branch; any other unsupported format fails elaboration with `$fatal`), bf16
  partial sums, U from the package; reducer tree
  uses fpAdder; bias, result memory and wide read in the format. The `OP_W`/`DATA_WIDTH` split of the
  mixed version collapses back to one width. Since 2026-10-09 the bf16 PE is fpMulWiden + fp32Adder, the
  reducer fp32Adder with fpNarrow at its write, the bias fp32 (`ACC_W` 32); result memory and wide read stay bf16 (`OUT_W` 16).
- **sienna_top / sienna_layer / sienna_multi**: one `DATA_WIDTH` = format width; the bias widening
  and operand/accumulator split go; residual identity rows keep 1.0 in the format (exists).
- **Maxpool_2D**: sign-magnitude compare for any float width (today only width 32 with IS_FP32);
  the padding value -infinity in the format (sienna_top's `32'hFF800000`).
- **dropout**: its multiplier becomes fpMultiplier; `CONST_SCALE` passed in the format.
- **GPNAE gpnae_poly**: barrel MAC multiplier/adder, square and post multipliers, and `fp32_down`
  (an fpAdder with a constant input) use the new units; `gpnae_tail` rewritten with `EXP_W`/`MAN_W`
  and per-format constants; range thresholds encoded per format; `poly_coeffs_bf16.mem`, chosen by
  an if-generate with a literal `INIT_FILE` per branch (both branches named `G_MAC`). The lower
  degrees need no RTL change: zero leading coefficients evaluate bit-identically.

## Verification

- **Each new unit gets the fp32 units' DV infrastructure** (Soham's requirement): a Verilator
  testbench against Berkeley SoftFloat through DPI (fp16 natively; bf16 via f32 then narrowed),
  corner cases plus random pairs, flag checks with the same flush-to-zero handling and ulp rule as
  TB_fp32Multiplier/TB_fp32Adder; a
  Makefile. Plus: at (8, 23) each generic unit must match the fp32 unit bit for bit on all vectors.
  Done: 205,776 vectors identical at (8, 23); bf16 exhaustive over all 2^32 operand pairs, 0 errors
  per unit. The bf16 reference computes in f32 with round-to-odd, then narrows by nearest-even; its
  flags are the round-to-odd operation's, which equal round-toward-zero flags, so they match a
  truncating unit near 2^128.
- **Bit-exact Python unit models** (fpMultiplier, fpAdder) checked against the RTL units, then
  used by the pipeline golden in hardware order: product n to slot n mod U, continuing across depth
  passes; the reducer's pairwise tree with the bias as its last input; pooling max; GPNAE Horner.
  Every bf16 output must match exactly.
- **GPNAE**: its format-generic framework (`number_formats.py`, TB_gpnae) run in bf16, accuracy
  per activation against the TensorFlow-style reference, as for TYTAN.
- **System**: the regression in bf16 (bit-exact golden; 29 tests after adding a zero-rows test
  and a signed-zero test that keeps -0), the four models in bf16 with accuracy against float, the
  GEMM sweep (bf16 checked bit-exact against the mesh model; error against float64 is reported), the random power-up
  regression and the lint.
  GEMM bf16 error, max |hw - ref| / max |ref| against float64 on the same bf16 operands, worst shape per depth K
  (N 16 T 4, 67 shapes; sets and cycles identical in all 67):

  | K | bf16 sums (`ba0_gemm_bf16`, main c05c332) | fp32 sums (`ba3f_gemm_bf16`, bf16_accum 0341b75) |
  |---|---|---|
  | 16 | 1.6e-02 | 3.2e-03 |
  | 64 | 2.7e-02 | 3.1e-03 |
  | 256 | 7.9e-02 | 3.2e-03 |
  | 1024 | 3.0e-01 | 3.6e-03 |
  | 768 (qkv, mlp_up, decode shapes) | 1.6e-01 to 1.9e-01 | 2.0e-03 to 2.9e-03 |
  | 3072 (mlp_down_128tok) | 5.0e-01 | 3.1e-03 |

  With fp32 sums the error no longer grows with K: it stays at bf16's output rounding, at most 2^-8 = 3.9e-03 of a value (round to nearest even). Since then the
  sweep also gates bf16 on the model gate's bf16 layer bound, 0.02. fp32: identical cycle counts; SIENNA-level checks unchanged (tolerance with the fp32 error bound).
  The mesh checks fp32 bit-exact too (Soham, 2026-09-28): a 1% relative limit failed correct N=64 cancellations.

## Open decisions (Soham)

- **bf16 activation accuracy: decided and fixed (Soham 2026-10-10 and 2026-10-11, sienna-int8-residual Decisions 4 and 5;
  Tasks 1.14a and 1.16 of its plan).** Worst relative error of the RTL's words against numpy float64 over every finite bf16
  input (GPNAE `bf16/lane`, pattern `bf16_all`; measured in Tasks 1.14a and 1.16, 2026-10-10 to 2026-10-12, not rerun at G4):
  - sigmoid 106.55% (x = -59.75; 18.06% within [-8, 8], 698 inputs over 1%) -> **0.772%** (x = 0.0156, 0 over 1%): every
    x < 0 takes the tail in fp32 with two Newton steps for 1/(1 + e^-|x|), rounded to bf16 once (runs rs1_b16_* RED,
    rs1_b16f_green2). **Hard gate at 1%** (`SIGMOID GATE`, wherever |exact| >= 2^-126). Cost: all-negative inputs take
    about 6x the cycles of positive ones (13-76 cycles per input; cycles change in bf16 sigmoid tests only); fp32 and int8
    lanes unchanged.
  - tanh 8.51% worst over |x| >= 2^-120 (at |x| = 3.84, the fit's c0), unchanged by the tail's refit (tail-region worst
    0.391% -> 0.063%); below T_TINY = 0.0903 it returns x (Task 1.16), so tiny inputs no longer flush to zero.
  - SELU 2.49% worst (x < 0); x >= 0 within 0.38% with lambda stored as 0x3F87 (Task 1.16; it was 1.08%).
  - tanh and SELU gate at the activation bound in force, 10% (`truth.BOUND`, a waiver; 1% is the target, every run
    prints the count over 1%); reaching 1% needs the fp32-tail candidates of `sienna_report/bf16_tanh_selu_study.log`
    (about 1.6x cycles), not taken.
  History (before 2026-10-10): SELU 2.82%, sigmoid 7.22% (12.68% at range 8; 17.6% on GPNAE's dense negative pattern),
  tanh 8.59% against the regression's 6.25% tolerance, from `2026-09-28_gpnae_gate.txt`; on 2026-10-05 Soham chose to
  gate bf16 bit-exactness and report accuracy, superseded by Decision 6 (every activation gates on the true function).
- **fp32 gpnae_model, negative sigmoid: closed 2026-10-10 (GPNAE 61867b6).** `--model hw` failed 9 of 10 patterns,
  sigmoid only: the model took `fpAdder(P, -1)` where the RTL's `fp32_down` drops P's last bit (no guard bit). The model
  now computes `fp32_down` stage for stage (the RTL is the intended arithmetic, unchanged), so fp32 is bit-exact on every
  activation and pattern; GPNAE's `make regression` gates `fp32/lane` bit-exact and `fp32/lane-accuracy` at 1%.
- **Findings in published units, reported and not changed.** D-1: fp32Adder raises overflow with
  underflow when a cancellation's exponent goes negative (the result is correctly zero; SIENNA does
  not use the flag). D-6: fp32Adder gives -0 for x + (-x) when the first operand is negative (IEEE
  nearest gives +0); fpAdder does the same by design. D-7: fp32Adder's stage 2-4 valid bits have no
  reset, so reset must be held 4+ cycles (the testbenches hold 8). F-GP1: GPNAE's `fp32_down`
  differs from fp32Adder(P, -1) by a few ulp on sigmoid's negative inputs within 3.5 (it has no guard bit; gpnae_model
  now models it, so this is a precision note, not a test gap).

## int8 (sub-project 2): agreed principles, design not written

int8 staging, cache and operands; PE int8 x int8 with int32 adder and partial sums; reducer and bias
in int32; a new requantize stage (scale multiply, shift, round, clamp to int8) after the mesh;
GPNAE in fixed point (integer multiplier and adder in the barrel MAC with shifts; tanh and sigmoid
saturate outside their range, so no large-input unit); integer max pooling (Maxpool_2D's non-FP32
path exists); dropout scale 2 as a shift. Packing several int8 MACs per PE per cycle is a separate,
later, architectural step: it is what gives a throughput gain; int8 alone gives area and power.

## Traps to expect

- A format that is not fp32 must not silently pick an fp32 unit: generate blocks must be exhaustive
  and fail elaboration on an unsupported format.
- Any golden that computes in float64 and compares with a tolerance will pass broken bf16 hardware.
- Never write the fp32 coefficient files; `gpnae_tests.py:write_coeff_rom()` only writes
  `taylor_coeffs_<fmt>.mem` (exists), and the poly lane needs the same guard.
- Verilator demotes an elaboration-time `$error` to a warning (USERERROR); use `$fatal` and lint
  with `-Werror-USERFATAL`, or a bad format elaborates.
- A string parameter passed down to `$readmemb` was not found by Verilator; pick the file with a
  literal in an if-generate.
- Stimulus sets shorter than N x N must write their zero rows explicitly; random power-up leaves
  the rest of the staging bank non-zero.
- `sienna_top` needs N^2 / lanes <= GPNAE_FIFO_DEPTH (32): N=64 needs 128 lanes.
- At N=64 the mesh testbench's VCD fills a farm node's disk (22.8 GB, fp32); run large meshes with
  `TRACE=0`. Collapse-k 0 at N=64 needs a 256 GB node for Verilator.
- Constants still hard-coded as fp32 bit patterns (search `32'h3F800000`, `32'hFF800000`,
  `32'h7FC00000`, `[30:23]`) will be wrong in bf16; the portability audit list in
  `testbenches/results/perf/pre_synthesis_v1_report.log` names them.
