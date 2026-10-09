# bf16 fp32-accumulation: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** bf16 builds multiply bf16 operands exactly into fp32 products, accumulate in fp32 in the PEs and reducers, and narrow each result to bf16 (round to nearest even) at the reducer's write; fp32 and int8 builds are unchanged.

**Architecture:** one new package split in ArithmeticLibrary: `acc_w` (sums, products, bias) becomes 32 for every format and a new `out_w` (the mesh's result word) keeps today's `acc_w` value (16 bf16, 32 fp32, 32 int8). The PE uses `fpMulWiden` + `fp32Adder` for narrow floats; the reducer adds in fp32 and narrows its output with a new combinational `fpNarrow`, so the result banks, the L3 result link and everything after them keep `out_w`. The bf16 bias row is widened exactly (bits << 16) before the mesh's fp32 bias input.

**Tech Stack:** SystemVerilog (Verilator 5.035, `--binary --timing --assert`), Python 3 bit-exact models (`fpu.py`, `mesh_model.py`), slurm farm.

**Spec:** `.claude/skills/sienna-bf16-fp32-accum/SKILL.md` (same directory; approved by Soham 2026-10-08). Read it first.

## Global Constraints

- fp32 and int8 builds: outputs identical to main `c05c332` (words compared with `$J/cr_msgs/task5a/t5_cmp.py` against the merged-tree runs `mc_gate_{fp32,int8}`; mesh results against `crb_sm16_*`, `crb_sm8g_*`), and cycle counts identical.
- bf16 builds: mesh, host, pack, GEMM and model-layer cycle counts identical to main (fpMulWiden 3 = today's `mul_lat(8,7)`, fp32 adder 5 = `add_lat`, U = 6); activation-limited polynomial-lane streams (tanh, SELU, sigmoid) may move, because `gpnae_tail`'s cycles depend on the values the mesh gives (reworded by Ruling 3, 2026-10-09); outputs change and are checked bit-exactly against the updated models.
- GPNAE (lanes, `gpnae.sv`, GPNAE math) unchanged; the lanes keep taking `DATA_WIDTH` (bf16) words.
- Narrowing: round to nearest even; NaN -> the format's canonical qNaN (`fpu.Fmt.qnan`, 7FC0 in bf16); infinity kept; an fp32 input with exponent field 0 (zero or subnormal) -> signed zero; rounding up past the largest finite gives infinity.
- Bias: the weight stream's bias row stays in the operand format; it is widened exactly (`x << (23 - MAN_W)`) before the mesh's `bias_i`, which is `acc_w` = 32 bits wide in every format.
- Every width assignment the spec lists as a silent-truncation site gets an elaboration or time-0 check (all builds pass `-Wno-WIDTHTRUNC -Wno-WIDTHEXPAND`).
- Each repo on branch `bf16_accum` (SIENNA's already exists with the spec commit). Innermost first: ArithmeticLibrary, then GPNAE (AriL bump only) and SystolicMesh, then SIENNA. A parent bumps a submodule only in the task that adapts to it, so every commit builds and passes.
- One-line comments; assertions under `ifndef SYNTHESIS`; every new check shown firing once on purpose (Verilator rejects `force` once queues exist: use defines/-G/plusargs or saved one-line mutant patches under `$J/cr_msgs/bf16acc/`).
- Every test gates on a reference (memory: every-test-gates-on-a-reference): a run counts as passing only if its outputs were compared and the comparison can fail.
- Farm only: `$J/cr_msgs/cr_launch.sh NAME MEM HOURS cmd...` (J=/proj/work/spramanik/sienna_jobs); partitions od-64-gb-8-cores, od-128-gb-8-cores, od-16-gb-8-cores (small jobs); never spot, never od-256-gb-32-cores, od-768-gb-24-cores, od-512-gb-16-cores; avoid the m7a partitions (od-32-gb-8-cores, od-64-gb-16-cores) and od-128-gb-16-cores, whose nodes fail to boot. Run names `ba<task>_*`. Fast tests (N <= 32) until check-in; one N = 64 bf16 sweep after.
- `git commit -F <file>`; no Co-Authored-By or generated-by lines (Soham's rule). No push until Soham asks.

## Review Focus

1. A sum that rounds past bf16's largest finite (fp32 0x7F7F8000 and above, finite) must narrow to +/-infinity, and NaN sums to 7FC0: pinned in Task 1's fpNarrow corners and in the model.
2. Exact ties (the 16 dropped bits exactly 0x8000) must round to even, both directions: pinned in Task 1 (tie classes with the kept LSB 0 and 1).
3. Accumulate passes (K-tiled layers, partial sums held across passes in the PE slots) must stay fp32 until the final pass and narrow once: pinned in Task 2's accumulate tests (mesh) and Task 3's accumulate pass (TB_sienna_top), bit-exact.
4. Packed sets in bf16 must keep the packed lane order and narrow each word once: pinned in Task 2 (`mm_packed`, `mm_packed_garbage` bf16 at WIDE_READ = N and 32) and Task 3 (`make pack FMT=bf16`).
5. A bias that is a bf16 subnormal or -0 must widen to the matching fp32 bits and be treated by the fp32 adder as the model treats it: pinned in Task 2 (a mesh test with subnormal and -0 bias words) and Task 3 (a pipeline test with them).

---

### Task 0: Branches and baselines

**Files:** none in git. Farm scripts under `$J/cr_msgs/bf16acc/`.

- [ ] Create branch `bf16_accum` in ArithmeticLibrary (SystolicMesh/ArithmeticLibrary and GPNAE/ArithmeticLibrary, the same repo), GPNAE and SystolicMesh, from their current main (AriL 07c6430, GPNAE b1ef12d, SystolicMesh 3fd3ef6). SIENNA's `bf16_accum` exists (spec commit 4772096 on main c05c332).
- [ ] Baselines on main RTL (launch from the current tree before any change; names `ba0_*`): mesh regression bf16 at N 8 (`$J/cmds/mesh_notrace.sh reg 8 bf16 1 --group matmul`), N 16 (`$J/cmds/mk.sh sm-verilator FMT=bf16 N=16 PYTHON=$J/venv/bin/python`), N 32 T 4; `make gemm FMT=bf16` (full, not QUICK, for the error-vs-K table) and `make gemm FMT=fp32 QUICK=1`; model sweep bf16 N 16 (`$J/cmds/model_sweep.sh 16 4 bf16 32`). fp32/int8 baselines: `mc_gate_{fp32,int8}` (main c05c332), `crb_sm16_*`, `crb_sm8g_*`.
- [ ] Record in the ledger: the baseline run names, bf16 ad01 anomaly scores (expected about 2.7-2.8x the reference), GEMM bf16 error vs K.

### Task 1: ArithmeticLibrary: accumulator width split, fpNarrow, fpMulWiden reset, models

**Files:**
- Modify: `Common/src/sienna_fmt_pkg.sv` (acc_w, new out_w, acc_exp_w, acc_man_w, widen)
- Modify: `Common/testbenches/TB_sienna_fmt_pkg.sv`
- Create: `Converters/FPNarrow/src/fpNarrow.sv`, `Converters/FPNarrow/testbenches/TB_fpNarrow.sv`, `Converters/FPNarrow/Makefile`
- Modify: `Multipliers/FPWiden/src/fpMulWiden.sv` (D-8: reset valids only)
- Modify: `Common/models/fpu.py` (narrow, widen, mul_widen), `Common/models/check_fpu.py` (fpNarrow dump check)

**Interfaces (produced, used by Tasks 2-3):**
- `sienna_fmt_pkg::acc_w(exp_w, man_w)` = 32 for every supported format.
- `sienna_fmt_pkg::out_w(exp_w, man_w)` = `is_int(exp_w) ? 32 : 1 + exp_w + man_w` (today's acc_w).
- `sienna_fmt_pkg::acc_exp_w(exp_w, man_w)` = `is_int ? 0 : 8`; `acc_man_w(exp_w, man_w)` = `is_int ? 31 : 23` (the accumulator's own format; int keeps 32-bit two's complement).
- `sienna_fmt_pkg::widen(x, man_w)`: an 8-exponent-bit float's bits to fp32 bits, exact: `x << (23 - man_w)`.
- `module fpNarrow #(EXP_W = 8, MAN_W = 7, W = 1 + EXP_W + MAN_W) (input logic [31:0] x_i, output logic [W-1:0] y_o)`: combinational fp32 -> (EXP_W, MAN_W), EXP_W must be 8 (elaboration `$fatal` otherwise); at MAN_W = 23 it is the identity.
- Python: `fpu.widen(f, x)` -> FP32 bits; `fpu.mul_widen(f, a, b)` = `mul(FP32, widen(f,a), widen(f,b))` (bit-exact to fpMulWiden; returns the 4-tuple); `fpu.narrow(f, x)` -> f bits with the rules in Global Constraints.

- [ ] **Step 1: Failing package test.** In TB_sienna_fmt_pkg add `expect_eq("acc_w(8,7)", acc_w(8,7), 32)`, `expect_eq("out_w(8,7)", out_w(8,7), 16)`, `out_w(8,23)`=32, `out_w(0,7)`=32, `acc_exp_w(8,7)`=8, `acc_man_w(8,7)`=23, `acc_w(0,7)`=32, `widen(32'h3F80, 7)` = 32'h3F800000, `widen(32'h8001,7)` = 32'h80010000; change the existing `acc_w(8,7)` expectation from 16 to 32. Run the AriL package TB on the farm (the target `make` uses for TB_sienna_fmt_pkg; check AriL's Common Makefile): expected FAIL (functions missing / acc_w 16).
- [ ] **Step 2: Package.** In `sienna_fmt_pkg.sv`:

```systemverilog
  // Width of products, partial sums, the reducer and the bias: int32 for int8, fp32 for every float format.
  function automatic int acc_w(input int exp_w, input int man_w);
    return 32;
  endfunction

  // Width of a mesh result word as it leaves the reducer: int32 for int8 (requantized later), the format itself for floats.
  function automatic int out_w(input int exp_w, input int man_w);
    return is_int(exp_w) ? 32 : 1 + exp_w + man_w;
  endfunction

  // The accumulator's own format: fp32 (8, 23) for every float, int32 for int8.
  function automatic int acc_exp_w(input int exp_w, input int man_w);
    return is_int(exp_w) ? 0 : 8;
  endfunction
  function automatic int acc_man_w(input int exp_w, input int man_w);
    return is_int(exp_w) ? 31 : 23;
  endfunction

  // An 8-exponent-bit float widened to fp32 bits, exactly.
  function automatic logic [31:0] widen(input logic [31:0] x, input int man_w);
    return x << (23 - man_w);
  endfunction
```
  Keep `mul_lat`/`add_lat` shapes (regression.py parses them by regex, ~:1570-1585). Re-run Step 1: PASS.
- [ ] **Step 3: Failing fpNarrow test.** Python first: `fpu.narrow(f, x)`:

```python
def narrow(f: Fmt, x):
    """fpNarrow: fp32 bits to f, round to nearest even; NaN -> f.qnan; exponent 0 -> signed zero; overflow -> infinity."""
    x = np.asarray(x, dtype=np.int64)
    sh = 23 - f.m
    if sh == 0:
        return x
    s, e = (x >> 31) & 1, (x >> 23) & 0xFF
    nan = (e == 0xFF) & ((x & 0x7FFFFF) != 0)
    r = (x + (1 << (sh - 1)) - 1 + ((x >> sh) & 1)) >> sh  # RNE on the magnitude bits; a carry into the exponent is right
    r = np.where(e == 0, s << (f.w - 1), r)  # zero and subnormal -> signed zero
    return np.where(nan, f.qnan, r).astype(np.int64)
```
  (The rounding add on the full word carries mantissa overflow into the exponent and, at 0x7F7F8000 and above, into infinity 0x7F80 with the sign intact.) Add `widen` and `mul_widen`. Python unit test in `Common/models/test_fpu.py` (or the existing models test file): corner table (+-0, +-subnormal, smallest normal, 1.0, ties 0x3F808000 -> 0x3F80 and 0x3F818000 -> 0x3F82, 0x7F7F7FFF -> 0x7F7F, 0x7F7F8000 -> 0x7F80, -0x7F7F8000 class -> 0xFF80, +-inf, qNaN, sNaN -> 0x7FC0), plus `narrow(widen(x)) == x` for every bf16 x with exponent != 0 and not NaN.
  TB_fpNarrow (SystemVerilog) walks the same corner table and 10^6 random fp32 values (LFSR), dumps `x y` per line to a file; `check_fpu.py` gets a `narrow` mode comparing the dump with `fpu.narrow`. Run on the farm before writing the RTL: expected build FAIL (fpNarrow missing).
- [ ] **Step 4: fpNarrow RTL** (combinational):

```systemverilog
`timescale 1ns / 100ps

// fp32 to a float with 8 exponent bits: round to nearest even, NaN to the canonical qNaN, exponent 0 to a signed zero, overflow to infinity.
module fpNarrow #(
    parameter int EXP_W = 8,
    parameter int MAN_W = 7,
    parameter int W     = 1 + EXP_W + MAN_W
) (
    input  logic [31:0]  x_i,
    output logic [W-1:0] y_o
);
  localparam int SH = 23 - MAN_W;
  if (EXP_W != 8 || MAN_W > 23 || MAN_W < 1) begin : G_BAD_FORMAT
    $fatal(1, "fpNarrow: EXP_W=%0d MAN_W=%0d: only 8 exponent bits and 1..23 mantissa bits", EXP_W, MAN_W);
  end else if (SH == 0) begin : G_ID
    assign y_o = x_i;
  end else begin : G_NARROW
    logic nan, tiny;
    logic [31:0] r;
    assign nan  = (x_i[30:23] == 8'hFF) && (x_i[22:0] != '0);
    assign tiny = (x_i[30:23] == 8'h00);
    assign r    = (x_i + ((32'd1 << (SH - 1)) - 32'd1) + 32'((x_i >> SH) & 32'd1)) >> SH;  // carries into the exponent, and to infinity
    assign y_o  = nan ? {1'b0, {EXP_W{1'b1}}, 1'b1, {(MAN_W - 1){1'b0}}} : tiny ? {x_i[31], {(W - 1){1'b0}}} : r[W-1:0];
  end
endmodule
```
  Re-run Step 3 on the farm: TB PASS, `check_fpu.py narrow` 0 mismatches over corners + 10^6 random. Lint (`verilator --lint-only -Wall`, 0 errors). Show `G_BAD_FORMAT` firing once (`-GEXP_W=5`).
- [ ] **Step 5: fpMulWiden to D-8.** Move its datapath registers to an `always_ff` without reset; keep reset on the valid/done chain. Re-run its TB (`make -C Multipliers/FPWiden` on the farm, bf16 and fp16): 0 mismatches against fp32Multiplier, as before (history/2026-09-28_aril_gate.txt: 203,366 bf16 products). Add a `check_fpu.py` mode or a Python test that `fpu.mul_widen` equals fpMulWiden's dump if the TB dumps; otherwise `mul_widen` is defined as `mul(FP32, widen, widen)` and fpMulWiden is checked against fp32Multiplier, which `fpu.mul(FP32)` models (existing check).
- [ ] **Step 6: Commit** (AriL `bf16_accum`): "acc_w is 32 in every format and out_w the mesh's result word (16 in bf16): bf16 sums go fp32; fpNarrow (fp32 to an 8-exponent float, round to nearest even, NaN canonical, exponent 0 to signed zero) with its TB and Python model; fpMulWiden resets only its valids (D-8); fpu.widen, mul_widen, narrow."

### Task 2: SystolicMesh: fp32 sums for bf16, narrowing at the reducer's write

**Files:**
- Modify: `src/engine/ProcessingElement.sv` (~:120-145: G_FP branch), `src/engine/AccumulationUnit.sv` (~:105-135 adder select; ~:160-170 write path), `src/top/SystolicArray.sv` (pass-through only if needed), `src/top/SystolicMesh.sv` (ACC_W/OUT_W: ~:9 params, :24 bias_i stays ACC_W, :363-375 result words and the L3 `RES_W` become `out_w` based, :419 MeshOutputSram `.DATA_WIDTH(OUT_W)`)
- Modify: `mesh_model.py` (matmul, _reduce, matmul_packed), `stim_format.py` (bias words for floats are 8-hex-digit widened fp32; result words stay `out_w`), `regression.py` (only if it reads widths), `Makefile` DESIGN_FILES (+ fpMulWiden.sv, fpNarrow.sv)
- Modify: `testbenches/TB_SystolicMesh.sv` (result width `out_w`, bias `acc_w`), `testbenches/TB_PE_pack.sv` (bf16 sums are fp32 bits now: no `>> (23-MAN_W)`), `test_mesh_model_packed.py`
- Bump: `ArithmeticLibrary` to Task 1's commit.

**Interfaces:**
- Consumes Task 1: `acc_w`, `out_w`, `acc_exp_w`, `acc_man_w`, `widen`, `fpNarrow`, `fpMulWiden`, `fpu.mul_widen`, `fpu.narrow`, `fpu.widen`.
- Produces (Task 3): SystolicMesh parameter `OUT_W` (default `out_w(EXP_W, MAN_W)`); `bias_i [MATRIX_SIZE][ACC_W]` (fp32 bits in float builds: the caller widens); result link `DATA_W = WIDE_READ*OUT_W + 3` (unchanged in value for bf16: 16-bit words); `mesh_model.matmul(f, passes, N, T, collapse_k=1, bias=None)` same signature, `bias` given in the operand format f (the model widens), result in format f (narrowed); `matmul_packed` likewise.

- [ ] **Step 1: Model first.** In `mesh_model.py`, for a narrow float f (f.m < 23): products `fpu.mul_widen(f, a, b)[0]`, slot sums `fpu.add(fpu.FP32, ...)`, the reducer tree in FP32 with the bias row `fpu.widen(f, bias)`, then `fpu.narrow(f, result)`; fp32 and int8 paths unchanged. Update the docstring (one line). `test_mesh_model_packed.py`: packed-vs-alone equality still holds in bf16 (both go through the same fp32 sums); add a bf16 case with K = 640 random operands comparing `matmul` (narrowed) against float64 `A @ B` rounded to bf16: error at most 1 bf16 ulp of the largest output on 99% of words (states what fp32 accumulation buys; fails on the old bf16-sum model). Run it before the RTL: it must pass with the new model and fail with main's (show both).
- [ ] **Step 2: TB first.** TB_SystolicMesh: results compared at `OUT_W` bits, bias driven at `ACC_W` from the stimulus (stim_format writes bf16 bias words widened to 8 hex digits for float builds); TB_PE_pack bf16 expectation = fp32 bits. Add mesh test configs with bias words that are bf16 subnormals and -0 (Review Focus 5) and keep `mm_packed`, `mm_packed_garbage` (Review Focus 4) and the accumulate/partial tests (Review Focus 3). Build against the unchanged RTL: expected FAIL (width checks or bit mismatches).
- [ ] **Step 3: RTL.**
  - PE G_FP branch: `fpMulWiden #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (... .result_o(prod) ...)` and `fp32Adder ADD (...)` (prod, acc, add_a, sum are `ACC_W` = 32). Keep `G_FP32` and `G_INT` as they are. Elaboration check: `ACC_W == sienna_fmt_pkg::acc_w(EXP_W, MAN_W)` (exists) plus, in G_FP, `$bits(prod) == 32`.
  - AccumulationUnit: adder by accumulator format: `fp32Adder` whenever `!is_int(EXP_W)` (drop the `fpAdder(EXP_W,MAN_W)` branch); write path: `write_data_o` is `OUT_W` wide; for narrow floats `fpNarrow #(EXP_W, MAN_W) NARROW (.x_i(lvl_d[LEVELS][0]), .y_o(write_data_o))`, else `assign write_data_o = lvl_d[LEVELS][0]`. New parameter `OUT_W = sienna_fmt_pkg::out_w(EXP_W, MAN_W)` with an elaboration check against the package.
  - SystolicMesh: `OUT_W` parameter and check; `MeshOutputSram .DATA_WIDTH(OUT_W)`; `res_words [WIDE_READ][OUT_W]`; `localparam int RES_W = WIDE_READ * OUT_W + 3` (the existing time-0 link-width check then follows); `bias_i` stays `ACC_W`; one-line comment fixes at :9 and wherever the survey listed "the format's own width in floats".
  - Makefile: add `../ArithmeticLibrary/Multipliers/FPWiden/src/fpMulWiden.sv` and `../ArithmeticLibrary/Converters/FPNarrow/src/fpNarrow.sv`.
- [ ] **Step 4: Runs (farm).** bf16 mesh regression at N 8 (matmul group), 16, 32, every tile: every test bit-exact against the new model; fp32 and int8 mesh regressions at N 8 and 16: results and cycles IDENTICAL to `crb_sm8g_*` / `crb_sm16_*` (`$J/cmds/pk_cmp_mesh.py`); bf16 cycles identical to `ba0_*` (cycles may not change). `mm_packed` bf16 at `-GWR=32` (N 16, the width sienna_top uses). Lint the mesh top in all three formats (0 errors; new warnings explained). Show each new elaboration check firing once.
- [ ] **Step 5: Commit** (SystolicMesh `bf16_accum`, AriL bump in the same commit or a separate first commit that builds): "bf16 sums in fp32: the PE multiplies with fpMulWiden into fp32Adder, the reducer adds in fp32 and narrows each result to bf16 with fpNarrow (round to nearest even) at its write, so the result banks and the result link keep bf16 words; the bias input is fp32; mesh_model follows; fp32 and int8 unchanged."
- [ ] **Step 6: GPNAE** (`bf16_accum`): bump `ArithmeticLibrary` to Task 1's commit (GPNAE uses no acc_w); run its lane regression in bf16 and fp32 on the farm: results identical to `crb_gp_*`. Commit: "Bump ArithmeticLibrary (fp32 accumulation for bf16: acc_w, out_w, fpNarrow); the lanes are unchanged."

### Task 3: SIENNA: bias widening, result width, goldens, model gate

**Files:**
- Modify: `src/sienna_top.sv` (wide_rd_data and the L3 link at `OUT_W`, not `ACC_W`: ~:192, :203, :217-223, :290-291; `fill_d = wide_rd_data` (~:1012) gets an elaboration check that the widths match; `bias_i` stays `ACC_W`; pass `OUT_W` to SystolicMesh if it is a parameter there)
- Modify: `src/sienna_layer.sv` (~:471: `bias_buf <= IS_INT ? w_bias_i[c] : sienna_fmt_pkg::widen(32'(w_data_i[c]), MAN_W)`, replacing the zero-extend `ACC_W'(w_data_i[c])`), `src/sienna_multi.sv` (pass-through only)
- Modify: `testbenches/TB_sienna_top.sv` (~:777-778), `TB_sienna_multi.sv` (~:282-283), `TB_sienna_model.sv` (~:358-360): `bias_i[c] = IS_INT ? ACC_W'(q[c]) : sienna_fmt_pkg::widen(q[c], MAN_W)` (float bias files stay in the operand format)
- Modify: `model_runner.py` (`_config_items` ~:834-839: `ACC_W` 32 for bf16, add `OUT_W`; `REPORTED` loses `("ad01","bf16")` once Step 4 shows it inside its bound; CLI help text ~:1163), `regression.py` (only where it computed ACC_W or a bf16 golden outside `mesh_model`; `lane_inputs` and `exact_layer` keep calling `mesh_model` and get narrowed bf16 words), `Makefile` (`SM_LIB_FILES` + fpMulWiden.sv, fpNarrow.sv; the stale-package guard ~:38-41 checks `ACC_W`), `synth/sienna_rtl.f` (+ the two files)
- Bump: `SystolicMesh` and `GPNAE` to Task 2's commits.

**Interfaces:** consumes Task 2's `OUT_W`, the fp32 `bias_i`, and `mesh_model` with unchanged signatures.

- [ ] **Step 1: TBs and goldens first.** TB bias widening as above; a pipeline test config with bf16 subnormal and -0 bias words (Review Focus 5); keep the accumulate pass (Review Focus 3). Build against main's RTL in bf16: expected FAIL (widths / words).
- [ ] **Step 2: RTL and tooling** as listed. Elaboration checks: sienna_top `OUT_W == sienna_fmt_pkg::out_w(EXP_W, MAN_W)` and `$bits(fill_d[0]) == OUT_W` in float builds; sienna_layer `$bits(bias_buf[0]) == ACC_W`. Show each firing once.
- [ ] **Step 3: Runs (farm, N 16 T 4 unless stated).**
  - `make regression FMT=bf16`: all gated steps pass with the regenerated bit-exact goldens; `make regression FMT=fp32` and `FMT=int8`: pass, and pipeline words identical to `mc_gate_{fp32,int8}` (t5_cmp.py); cycles identical in all three formats against `mc_gate_*`.
  - `make gemm FMT=bf16` (full): 0 outputs differ from the bit-exact model; error vs K reported against `ba0_gemm_bf16` (expected: no longer growing with K).
  - `make pack FMT=bf16`: 0 mismatches. `make tflite FMT=int8`: 0 differ.
  - Model sweep bf16 N 16 and N 32 (`$J/cmds/model_sweep.sh 16 4 bf16 32`, `32 4 bf16 32`): MODEL GATE PASS with every layer bit-exact; ad01's anomaly score against its 1e-2 bound. If inside: remove `("ad01","bf16")` from `REPORTED` and rerun (now gated, PASS); if outside: report the measured value and stop for Soham (do not widen the bound).
  - LINK_STAGES=1 bf16 pipeline once (words identical to stage 0).
- [ ] **Step 4: Commit(s)** (SIENNA `bf16_accum`): RTL + bumps first ("bf16 builds sum in fp32: the bias row is widened exactly before the mesh, the mesh's results stay bf16; bumps SystolicMesh and GPNAE"), then tooling/gate ("model gate: ad01 bf16 gated again ...") if separate.

### Task 4: Measurements and docs

**Files:** `sienna_report/bf16_accum_compare.log` (outside git); `.claude/skills/sienna-bf16-fp32-accum/SKILL.md` (As built), `.claude/skills/sienna-uniform-format/SKILL.md` (bf16 accumulation now fp32: a second exception to the uniform rule; the GEMM error-vs-K numbers replaced), `.claude/skills/sienna-int8/SKILL.md` (ACC_W table row), `README.md` (~:30 pipeline format bullet: "fp32 sums for bf16"; the bf16 accuracy per model, task #45), `SystolicMesh/README.md` (~:29), comments the survey listed.

- [ ] `bf16_accum_compare.log`: per model (ResNet-8, DS-CNN, MobileNet, ad01) bf16 vs float reference before (`ba0_*`) and after: top-1 agreement, worst layer error, ad01 anomaly score; GEMM bf16 error vs K before/after; cycles before/after for the pipeline perf configs (mesh equal; any polynomial-lane-bound config that moves listed with the reason, Ruling 3) (`make perf-analysis FMT=bf16 N=16` before and after); area estimate delta (from `$J/area_estimate.py`, updated to the new widths; labelled an estimate, no synthesis). Every number from a named run.
- [ ] Docs as listed; the README states bf16 accuracy per model from gated runs (task #45).
- [ ] Commit docs on SIENNA `bf16_accum` (and SystolicMesh README in its repo).
- [ ] Final whole-branch review across the three changed repos; one fix pass; merge only on Soham's go-ahead; then one N = 64 bf16 sweep (regression, perf, models) with the gate.
