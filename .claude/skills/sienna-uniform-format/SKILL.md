---
name: sienna-uniform-format
description: Use when changing SIENNA's number format - building it in bf16 (or later int8) end to end, adding the format parameter or format package, writing narrow float units (fpMultiplier, fpAdder) in ArithmeticLibrary, making GPNAE, Maxpool_2D or dropout format-generic, or writing a bit-exact golden model for a narrow format. Also use when someone asks why a bf16 run disagrees with fp32, or how precision is chosen per build.
---

# SIENNA: one number format per build

**Status: uniform bf16 implemented and verified 2026-09-28; int8 not started.** On the `bf16`
branch of all four repos; it replaced the earlier mixed bf16. Plan and task record:
`implementation-plan.md`. Gate reports (verbatim, in the `sienna-report` skill's `history/`):
- G1 ArithmeticLibrary: `2026-09-28_aril_gate.txt`
- G2 GPNAE: `2026-09-28_gpnae_gate.txt` (correctness passes; bf16 activation accuracy is an open decision, below)
- G3 SystolicMesh and G4 SIENNA: section 2 and 4 of the SIENNA report (`sienna-report` skill); farm runs `g3*`, `g4p_*`
- fp32 before the work: `2026-09-28_baseline_fp32.txt`; bf16 coefficient fit: `2026-09-28_poly_coeffs_fit.txt`

**REQUIRED BACKGROUND:** the `sienna-rtl` skill.

## The principle (agreed with Soham)

Every module works in the same format, chosen per build by parameters, by swapping its arithmetic
units: storage, mesh multipliers and adders, reducer, GPNAE, pooling, dropout. This is how the
published GPNAE was meant to work ("swap the multipliers and adders, change top-level parameters").

The one inherent exception is accumulation width where a format cannot hold a sum of products:
int8 accumulates in int32 and requantizes back. That is a property of dot products, not a
mixed-precision choice. bf16 accumulates in bf16 (uniform), accepting the accuracy cost.

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
| fpMulWiden (exists, bf16 branch) | stays in ArithmeticLibrary, tested; SIENNA stops using it | uniform replaces mixed |
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
  mixed version collapses back to one width.
- **sienna_top / sienna_layer / sienna_multi**: one `DATA_WIDTH` = format width; the bias widening
  and operand/accumulator split go; residual identity rows keep 1.0 in the format (exists).
- **Maxpool_2D**: sign-magnitude compare for any float width (today only width 32 with IS_FP32);
  the padding value -infinity in the format (sienna_top's `32'hFF800000`).
- **dropout**: its multiplier becomes fpMultiplier; `CONST_ONE`/`CONST_SCALE` passed in the format.
- **GPNAE gpnae_poly**: barrel MAC multiplier/adder, square and post multipliers, and `fp32_down`
  (an fpAdder with a constant input) use the new units; `gpnae_tail` rewritten with `EXP_W`/`MAN_W`
  and per-format constants; range thresholds encoded per format; `poly_coeffs_bf16.mem`, chosen by
  an if-generate with a literal `INIT_FILE` per branch (both branches named `G_MAC`). The lower
  degrees need no RTL change: zero leading coefficients evaluate bit-identically.

## Verification

- **Each new unit gets the fp32 units' DV infrastructure** (Soham's requirement): a Verilator
  testbench against Berkeley SoftFloat through DPI (fp16 natively; bf16 via f32 then narrowed),
  corner cases plus random pairs, flag checks with the same flush-to-zero handling and ulp rule as
  TB_fp32Multiplier/TB_fp32Adder; a Vivado testbench with `gen_vectors.cpp` + `vectors.mem`; a
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
  GEMM sweep (bf16 checked bit-exact against the mesh model; error against float64 is reported,
  not gated, and grows with K: 6% at 256, 23% at 1024, 50% at 3072), the random power-up regression
  and the lint. fp32: identical cycle counts; SIENNA-level checks unchanged (tolerance with the fp32 error bound).
  The mesh checks fp32 bit-exact too (Soham, 2026-09-28): a 1% relative limit failed correct N=64 cancellations.

## Open decisions (Soham)

- **bf16 activation accuracy.** Worst relative error against the exact functions: SELU 2.82%,
  sigmoid 7.22% (12.68% at range 8), tanh 8.59%, against the regression's 6.25% tolerance. The
  hardware matches its bit-exact model; the error is the lane design at 8 significand bits
  (sigmoid's 1 - P near -3.5, power-basis Horner for tanh). `2026-09-28_gpnae_gate.txt` (sienna-report history) has the table.
- **Findings in published units, reported and not changed.** D-1: fp32Adder raises overflow with
  underflow when a cancellation's exponent goes negative (the result is correctly zero; SIENNA does
  not use the flag). D-6: fp32Adder gives -0 for x + (-x) when the first operand is negative (IEEE
  nearest gives +0); fpAdder does the same by design. D-7: fp32Adder's stage 2-4 valid bits have no
  reset, so reset must be held 4+ cycles (the testbenches hold 8). F-GP1: GPNAE's `fp32_down`
  differs from fp32Adder(P, -1) by a few ulp on sigmoid's negative inputs within 3.5.

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
