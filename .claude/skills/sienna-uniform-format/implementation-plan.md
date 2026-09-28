# Uniform bf16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SIENNA end to end in bf16, with every module in the build's format, by adding truncating bf16 units to AriL and swapping them in, while fp32 builds stay bit- and cycle-identical.

**Architecture:** Two new AriL units (`fpMultiplier`, `fpAdder`) implement the fp32 units' algorithms at any width. A package, `sienna_fmt_pkg`, says which formats a build supports and gives their unit latencies. Every consumer (PE, reducer, barrel MAC, GPNAE tail and post stage, dropout) picks fp32 units at (8, 23), the new units at (8, 7), and fails elaboration otherwise. One Python bit-exact model of the two units (`fpu.py`) is checked against the RTL, then reused by the GPNAE, mesh and SIENNA goldens, so every bf16 output is compared bit for bit. Verification goes bottom up in four gated levels: AriL, then GPNAE, then SystolicMesh sweeps, then SIENNA sweeps.

**Tech Stack:** SystemVerilog, Verilator (farm, via `snap_launch.sh`), Berkeley SoftFloat 3e through DPI-C, Python 3 with numpy.

**Spec:** `.claude/skills/sienna-uniform-format/SKILL.md` (approved 2026-09-27). Background: the `sienna-rtl` and `sienna-back-to-back` skills.

## Global Constraints

- Format selection: `EXP_W`, `MAN_W` parameters; width `1 + EXP_W + MAN_W`; fp32 = 8/23, bf16 = 8/7.
- Rounding in new units: **truncate**, exactly like fp32Multiplier/fp32Adder.
- Unit latencies: from a shared package `sienna_fmt_pkg`, per format; PE, reducer and barrel MAC stop hard-coding 8 and 5.
- PE partial-sum slots: U = adder latency + 1 (from the package).
- fp32 builds keep fp32Multiplier/fp32Adder unchanged; fp32 must stay bit- and cycle-identical.
- fpMulWiden stays in ArithmeticLibrary, tested; SIENNA stops using it.
- GPNAE lane: only `gpnae_poly`; the published `gpnae.sv` lane is out of scope.
- GPNAE coefficients: refit for bf16 into a new `poly_coeffs_bf16.mem`; fp32 files (`poly_coeffs.mem`, `taylor_coeffs.mem`) never change.
- Golden model for narrow formats: **bit-exact** emulation, not a tolerance.
- A format that is not fp32 must not silently pick an fp32 unit: generate blocks must be exhaustive and fail elaboration on an unsupported format.
- Verification order (Soham, 2026-09-27): AriL first, then GPNAE, then sweeps at SystolicMesh level, then sweeps at SIENNA level. A level's gate passes and is reported before the next level starts.
- Every build and simulation runs on the farm through `/proj/work/spramanik/sienna_jobs/snap_launch.sh NAME MEM_GB HOURS cmd...` (output in `sienna_jobs/runs/NAME/`); never on the login node.
- Commits: one change per commit, RTL before testbench and scripts, and push. Push the innermost repo first: AriL (`git push git@github.com:SoHam-56/ArithmeticLibrary.git <branch>`, since its origin is HTTPS without credentials), then GPNAE and SystolicMesh, then SIENNA. Check each push succeeded. No Co-Authored-By or generated-by lines.
- Comments are one line. Generated reports are `.log`. Never delete a file without Soham's explicit go-ahead (build output directories under `Verilator/` excepted, which only `make clean` touches). GPNAE is published work: report math bugs, do not change them silently.
- Keep the line endings a file already has (dropout.sv is CRLF): edit with tools that preserve them, and check `git diff --stat` shows only the intended lines.

## Branches and file map

All work continues on the existing `bf16` branches (AriL `bf16` at 7ad20b7, SystolicMesh `bf16` at 934d99b, SIENNA `bf16` at 55f3a8c). GPNAE gets a new `bf16` branch from `main` (f1482f4). The mixed-bf16 operand split already on these branches is reworked in place into one format.

| Repo | Create | Modify |
|---|---|---|
| AriL | `Common/src/sienna_fmt_pkg.sv`, `Common/testbenches/{fp_stim.svh, fp_ref_core.h, fp_ref.cpp, gen_vectors.cpp, generate_vectors.sh, TB_sienna_fmt_pkg.sv, TB_fp32Dump.sv}`, `Common/models/{fpu.py, check_fpu.py}`, `Common/Makefile`, `Multipliers/FP/{Makefile, src/fpMultiplier.sv, testbenches/TB_fpMultiplier.sv, TB_fpMultiplierEq.sv, TB_fpMultiplierVIVADO.sv, vectors_bf16.mem}`, `Adders/FP/` (same set for `fpAdder`) | none (fp32 units, fpMulWiden untouched) |
| GPNAE | `fit_poly_coeffs.py`, `gpnae_model.py`, `poly_coeffs_bf16.mem`, `src/TYTAN/Memory/poly_coeffs_bf16.mem` | `ArithmeticLibrary` pointer, `Makefile`, `src/TYTAN/barrel_mac.sv`, `src/gpnae_tail.sv`, `src/gpnae_poly.sv`, `testbenches/TB_gpnae_poly.sv`, `regression.py`, `gpnae_tests.py` |
| SystolicMesh | `mesh_model.py`, `stim_format.py` | `ArithmeticLibrary` pointer, `Makefile`, `src/engine/ProcessingElement.sv`, `src/engine/AccumulationUnit.sv`, `src/top/SystolicArray.sv`, `src/top/SystolicMesh.sv`, `testbenches/TB_SystolicMesh.sv`, `regression.py`, `matmul_tests.py`, `conv_tests.py` |
| SIENNA | `.claude/skills/sienna-uniform-format/implementation-plan.md` (this file) | `SystolicMesh`, `GPNAE` pointers, `Makefile`, `synth/sienna_rtl.f`, `Maxpool/Maxpool_2D.sv`, `Dropout/dropout.sv`, `src/sienna_top.sv`, `src/sienna_layer.sv`, `src/sienna_multi.sv`, `testbenches/TB_sienna_{top,layer,multi,model}.sv`, `regression.py`, `model_runner.py`, `gemm_sweep.py`, the skill's status line |
| sienna_jobs (no repo) | `cmd_aril.sh`, `cmd_aril_exh.sh`, `cmd_gpnae_reg.sh`, `cmd_mesh_fmt.sh`, `cmd_sienna_fmt.sh` | none |

Why these homes: the package and the unit model describe AriL's units, so they live with them, and GPNAE and SystolicMesh both carry AriL as a submodule. `sienna_fmt_pkg` keeps the name the approved spec gives it.

## Decisions this plan makes (flag at review)

- **D-1: fp32Adder raises overflow on some underflows.** By reading `fp32Adder.sv:192-237`: `norm_exp` is a 9-bit unsigned value, so a cancellation whose normalized exponent goes negative wraps to 256 or more. `overflow_o = (norm_exp >= 255) && ...` then fires together with `underflow_o`, while the result itself is correctly a flushed zero. SIENNA never connects the flag. Per the published-work rule, fp32Adder stays unchanged and the bug is reported. `fpAdder` computes overflow only for a non-negative exponent. At (8, 23) the equivalence test requires identical results on every vector and identical flags except on vectors where fp32Adder shows this quirk; it counts those separately and prints them. Task 0 confirms the quirk in simulation before anything depends on it.
- **D-2: fpMultiplier's product.** Significands wider than 12 bits use `karatsubaUnsigned` like fp32Multiplier (8 cycles); narrower ones (bf16) use one registered product (3 cycles, as fpMulWiden). `fpAdder` keeps fp32Adder's 5 stages at every width, so U stays 6.
- **D-3: The SoftFloat reference for bf16** widens exactly to f32, computes with `softfloat_round_odd`, then rounds to nearest-even at 7 mantissa bits. Round-to-odd at 24 bits and then nearest at 8 bits gives the correctly rounded bf16 result. The reference applies the units' input and output policy: inputs that are subnormal read as zero, and results that are tiny before rounding become a signed zero with the underflow flag set. The testbenches then use the fp32 testbenches' acceptance rules unchanged: 2 ulp for the multiplier, 3 for the adder, NaN matches any NaN, and the same flag checks.
- **D-4: Coefficients in bf16** keep the fp32 table's layout (bases 0, 9, 16; degrees 8, 6, 8), so `gpnae_poly`'s ROM map does not change. Task 12 also prints the error of lower degrees as information. Lowering a degree is a later decision for Soham.
- **D-5: bf16 GEMM sweep and models** report accuracy and cycles, as the spec asks. Bit-exactness is claimed for the 27-test regression, GPNAE, and mesh levels, not for the layer engine's multi-layer runs.

## Review Focus

1. **An unsupported format** (EXP_W=5, MAN_W=10) must fail elaboration in every block that picks a unit, not silently build with fp32 units. Pinned in Tasks 9, 10 and 13 (barrel_mac, gpnae_tail, gpnae_poly), Task 15 (mesh), Task 19 (dropout), and Task 20 (top). Each has a lint build that must fail with the block's message.
2. **Signed zeros.** A PE's first add is `0 + p`, so a `-0` product becomes `+0`. `max(-0, +0)` keeps the first operand, and ReLU of `-0` gives `+0`. Every bf16 golden must reproduce these bits, and a float-valued golden would hide them. Pinned in Task 16 with a mesh set whose A holds `-0` rows, in Task 18 with pooling ties, and in Task 21 with a SIENNA test that has zero rows.
3. **Inputs exactly on a GPNAE range threshold** (±4.0 and ±3.5 in bf16). The compare is strict, so exactly 4.0 takes the polynomial and the next value up takes the tail. Pinned in Task 11 by the `act_threshold` pattern, which runs bit-exact in Task 13.
4. **Random power-up state in the new units.** The metadata delay lines and adder stages are not reset. Pinned in Task 7 (unit testbenches with `--x-initial unique`), Task 14 (GPNAE), Task 17 (mesh), and Task 23 (SIENNA random-init regression in bf16).
5. **Stale or wrong-width memory files.** A bf16 build reading 8-digit fp32 words, or `poly_coeffs.mem` instead of `poly_coeffs_bf16.mem`, would truncate silently. Pinned in Task 11 (the TB checks the lane's ROM against the format's file), Task 16 (mesh stimulus width check before every run), and Task 21 (SIENNA file width check before every run).

---

## Level 0: setup and baseline

### Task 0: Branches, job scripts, fp32 baseline

**Files:**
- Create: `/proj/work/spramanik/sienna_jobs/cmd_aril.sh`, `/proj/work/spramanik/sienna_jobs/cmd_gpnae_reg.sh`
- Create (GPNAE branch): `bf16` from `main`

**Interfaces:**
- Produces: `cmd_aril.sh UNIT TARGET [PLUSARGS]` runs `make TARGET` in `SystolicMesh/ArithmeticLibrary/<UNIT>` of a snapshot (UNIT is `Multipliers/FP`, `Adders/FP`, `Common`, `Multipliers/FP32`, `Adders/FP32`) and copies logs to `testbenches/results/uniform/`. `cmd_gpnae_reg.sh ARGS` runs `python3 regression.py ARGS` in `GPNAE/` and copies `results/` and the report.

- [ ] **Step 1: Create the GPNAE branch**

```bash
cd /proj/work/spramanik/SIENNA/GPNAE && git status -s   # expect only the two generated stimulus/config files
git switch -c bf16 && git push -u origin bf16
```

- [ ] **Step 2: Write the AriL job script**

Every command in this plan uses `J=/proj/work/spramanik/sienna_jobs` and runs from `/proj/work/spramanik/SIENNA`. `snap_launch.sh` snapshots the working tree, including uncommitted edits, and runs the command on a farm node from the snapshot root.

`/proj/work/spramanik/sienna_jobs/cmd_aril.sh` (`chmod +x`):

```bash
#!/bin/bash
# AriL unit checks on a snapshot: builds the SoftFloat copies it needs, runs make TARGET in SystolicMesh/ArithmeticLibrary/UNIT; args: UNIT TARGET [PLUSARGS]; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
ROOT=$(pwd); UNIT=$1; TARGET=$2; shift 2
A=SystolicMesh/ArithmeticLibrary
for SF in $A/Adders/FP32/testbenches/berkeley-softfloat-3 $A/$UNIT/testbenches/berkeley-softfloat-3; do
  B=$SF/build/Linux-x86_64-GCC
  if [ -d $B ] && [ ! -f $B/softfloat.a ]; then make -C $B -j16 >> $ROOT/softfloat_build.log 2>&1 || { echo "SoftFloat build failed: $B"; exit 1; }; fi
done
mkdir -p $ROOT/testbenches/results/uniform
LOG=$ROOT/testbenches/results/uniform/$(echo $UNIT | tr / _)_$TARGET.log
(cd $A/$UNIT && make $TARGET PLUSARGS="$*") 2>&1 | tee $LOG
rc=${PIPESTATUS[0]}
grep -q "RESULT: FAILED\|^FAILURE:" $LOG && rc=1  # testbenches finish with status 0 even when they fail
exit $rc
```

The new units' Makefiles use the SoftFloat copy in `Adders/FP32/testbenches`. The fp32 units' own Makefiles use their own copies, which the loop also builds.

- [ ] **Step 3: Write the GPNAE job script**

`/proj/work/spramanik/sienna_jobs/cmd_gpnae_reg.sh`:

```bash
#!/bin/bash
# GPNAE activation regression on a snapshot; args go to GPNAE/regression.py; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
ROOT=$(pwd); mkdir -p $ROOT/testbenches/results/uniform/gpnae
cd GPNAE && python3 regression.py "$@"; rc=$?
cp -r testbenches/results/* $ROOT/testbenches/results/uniform/gpnae/ 2>/dev/null  # per-pattern logs and gpnae_report.log
exit $rc
```

- [ ] **Step 4: Run the fp32 baselines on the farm**

```bash
J=/proj/work/spramanik/sienna_jobs
$J/snap_launch.sh u0_mul32 8 1 $J/cmd_aril.sh Multipliers/FP32 verilator
$J/snap_launch.sh u0_add32 8 1 $J/cmd_aril.sh Adders/FP32 verilator
$J/snap_launch.sh u0_gpnae32 16 2 $J/cmd_gpnae_reg.sh --lane poly --format fp32
$J/snap_launch.sh u0_gpnae32t 16 2 $J/cmd_gpnae_reg.sh --lane gpnae --format fp32
$J/snap_launch.sh u0_sienna32 32 4 python3 regression.py --n 16 --tile-size 4
$J/snap_launch.sh u0_mesh32 32 4 bash -c "cd SystolicMesh && make regression TRACE=0"
```

Expected: the fp32 unit TBs print `SUCCESS: All N vectors passed!`. GPNAE poly prints its verdict and worst error per activation. SIENNA gives 27/27, the mesh 68/68. Record each run's per-test cycle counts and worst errors. They are the fp32 reference for the "bit- and cycle-identical" checks in Tasks 8, 15 and 20. Copy them into `testbenches/results/uniform/baseline_fp32.log`, with one line per test: name, cycles, and pass or fail.

- [ ] **Step 5: Confirm D-1 (fp32Adder overflow on underflow) in simulation**

Copy `SystolicMesh/ArithmeticLibrary/Adders/FP32/testbenches/TB_fp32Adder.sv` to `.claude/scratch/TB_fp32Adder_d1.sv`; the library file stays untouched. In the copy's `[Phase 2]` section, after the massive-cancellation vector, add:

```systemverilog
    drive_bus(32'h01000001, 32'h81000000);  // 2^-125 (1 + 2^-23) - 2^-125: exact 2^-148, exponent goes negative
    @(posedge clk);
```

The job swaps the copy in on the node-local snapshot only:

```bash
$J/snap_launch.sh u0_d1 8 1 bash -c "cp .claude/scratch/TB_fp32Adder_d1.sv SystolicMesh/ArithmeticLibrary/Adders/FP32/testbenches/TB_fp32Adder.sv && $J/cmd_aril.sh Adders/FP32 verilator"
```

Expected: an error line `Expected [Ov=0 Un=1 Inv=0]` / `Got [Ov=1 Un=1 Inv=0]` for that vector, and no other errors. If it does not appear, D-1 is wrong. Stop, remove the quirk accounting from Tasks 2 and 5, and tell Soham.

- [ ] **Step 6: Commit the GPNAE branch point (nothing to commit in repos); report Level 0**

Report to Soham: baselines recorded (path), D-1 confirmed or not. No repo changes in this task.

---

## Level 1: AriL (gate G1 in Task 7)

### Task 1: `sienna_fmt_pkg`

**Files:**
- Create: `SystolicMesh/ArithmeticLibrary/Common/src/sienna_fmt_pkg.sv`
- Create: `SystolicMesh/ArithmeticLibrary/Common/testbenches/TB_sienna_fmt_pkg.sv`
- Create: `SystolicMesh/ArithmeticLibrary/Common/Makefile`

**Interfaces:**
- Produces (all `function automatic`, usable in constant expressions):
  - `bit sienna_fmt_pkg::is_fp32(int exp_w, int man_w)`
  - `bit sienna_fmt_pkg::supported(int exp_w, int man_w)`: true for (8, 23) and (8, 7) only
  - `int sienna_fmt_pkg::mul_lat(int exp_w, int man_w)`: 8 if `man_w + 1 > 12`, else 3
  - `int sienna_fmt_pkg::add_lat(int exp_w, int man_w)`: 5
  - `logic [31:0] sienna_fmt_pkg::from_fp32(logic [31:0] x, int man_w)`: finite fp32 constant rounded nearest-even to `man_w` mantissa bits, right-aligned (8-bit exponent formats only)

- [ ] **Step 1: Write the failing test**

`Common/testbenches/TB_sienna_fmt_pkg.sv`:

```systemverilog
`timescale 1ns / 100ps

// Checks sienna_fmt_pkg: supported formats, unit latencies, and constants narrowed from fp32.
module TB_sienna_fmt_pkg;
  import sienna_fmt_pkg::*;
  int errs = 0;

  task automatic expect_eq(input string what, input longint got, input longint want);
    if (got != want) begin
      errs++;
      $display("[FAIL] %s: got %0h, want %0h", what, got, want);
    end
  endtask

  initial begin
    expect_eq("is_fp32(8,23)", is_fp32(8, 23), 1);
    expect_eq("is_fp32(8,7)", is_fp32(8, 7), 0);
    expect_eq("supported(8,7)", supported(8, 7), 1);
    expect_eq("supported(5,10)", supported(5, 10), 0);
    expect_eq("mul_lat(8,23)", mul_lat(8, 23), 8);
    expect_eq("mul_lat(8,7)", mul_lat(8, 7), 3);
    expect_eq("add_lat(8,23)", add_lat(8, 23), 5);
    expect_eq("add_lat(8,7)", add_lat(8, 7), 5);
    expect_eq("1.0", from_fp32(32'h3F800000, 7), 32'h3F80);
    expect_eq("lambda", from_fp32(32'h3F867D5F, 7), 32'h3F86);  // low half 7D5F rounds down
    expect_eq("1/2!", from_fp32(32'h3E2AAAAB, 7), 32'h3E2B);  // low half AAAB rounds up
    expect_eq("-inf", from_fp32(32'hFF800000, 7), 32'hFF80);
    expect_eq("tie to even down", from_fp32(32'h3F808000, 7), 32'h3F80);
    expect_eq("tie to even up", from_fp32(32'h3F818000, 7), 32'h3F82);
    expect_eq("fp32 unchanged", from_fp32(32'h3F867D5F, 23), 32'h3F867D5F);
    $display("TB_sienna_fmt_pkg: %0d errors", errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Common/Makefile`:

```make
SHELL := /bin/bash
# Common AriL checks: the format package, and the fp32 units dumped for the Python model.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/..
TB_DIR  = $(PRJ_DIR)/testbenches
FLAGS   = --binary --timing --assert --sv -I$(TB_DIR) --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
PLUSARGS ?=

pkg:
	verilator $(FLAGS) --top-module TB_sienna_fmt_pkg --Mdir $(PRJ_DIR)/Verilator/pkg $(PRJ_DIR)/src/sienna_fmt_pkg.sv $(TB_DIR)/TB_sienna_fmt_pkg.sv -o sim
	$(PRJ_DIR)/Verilator/pkg/sim

.PHONY: pkg
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch.sh u1_pkg 4 1 $J/cmd_aril.sh Common pkg`
Expected: build error `Cannot find file containing module/package: 'sienna_fmt_pkg'` (or missing `sienna_fmt_pkg.sv`).

- [ ] **Step 3: Write the package**

`Common/src/sienna_fmt_pkg.sv`:

```systemverilog
`timescale 1ns / 100ps

// Number formats a SIENNA build may use and the latencies of their AriL units; generate blocks reject any format not listed.
package sienna_fmt_pkg;

  // fp32: 8 exponent bits, 23 mantissa bits.
  function automatic bit is_fp32(input int exp_w, input int man_w);
    return (exp_w == 8) && (man_w == 23);
  endfunction

  // fp32 (fp32Multiplier, fp32Adder) and bf16 (fpMultiplier, fpAdder); nothing else is verified.
  function automatic bit supported(input int exp_w, input int man_w);
    return (exp_w == 8) && ((man_w == 23) || (man_w == 7));
  endfunction

  // valid_i to done_o: Karatsuba product (fp32Multiplier, fpMultiplier above 12 significand bits) 8, one-stage product 3.
  function automatic int mul_lat(input int exp_w, input int man_w);
    return (man_w + 1 > 12) ? 8 : 3;
  endfunction

  // valid_i to done_o: fp32Adder and fpAdder, 5 at every width.
  function automatic int add_lat(input int exp_w, input int man_w);
    return 5;
  endfunction

  // A finite fp32 constant in a format with 8 exponent bits, rounded to nearest even, right-aligned.
  function automatic logic [31:0] from_fp32(input logic [31:0] x, input int man_w);
    automatic int sh = 23 - man_w;
    if (sh == 0) return x;
    return (x + ((32'd1 << (sh - 1)) - 32'd1) + ((x >> sh) & 32'd1)) >> sh;
  endfunction

endpackage
```

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch.sh u1_pkg 4 1 $J/cmd_aril.sh Common pkg`
Expected: `TB_sienna_fmt_pkg: 0 errors` and `RESULT: PASSED`.

- [ ] **Step 5: Commit (AriL, branch bf16)**

```bash
cd /proj/work/spramanik/SIENNA/SystolicMesh/ArithmeticLibrary
git add Common/src/sienna_fmt_pkg.sv && git commit -m "sienna_fmt_pkg: supported formats, unit latencies, constants narrowed from fp32"
git add Common/testbenches/TB_sienna_fmt_pkg.sv Common/Makefile && git commit -m "TB_sienna_fmt_pkg: package self-test"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 2: Bit-exact Python model, checked against the fp32 RTL units

**Files:**
- Create: `Common/models/fpu.py`, `Common/models/check_fpu.py`
- Create: `Common/testbenches/fp_stim.svh`, `Common/testbenches/TB_fp32Dump.sv`
- Modify: `Common/Makefile` (add target `dump32`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `fpu.Fmt(name, exp_w, man_w)` with fields `e, m, w, bias, emax, mmask, qnan, sig`; `fpu.FP32`, `fpu.BF16`, `fpu.FORMATS = {"fp32":…, "bf16":…}`.
  - `fpu.mul(f, a, b) -> (res, ov, un, inv)`, where `a` and `b` are int64 arrays or ints of bit patterns, `res` is an int64 array, and the flags are bool arrays.
  - `fpu.add(f, a, b, fp32_overflow_quirk=False) -> (res, ov, un, inv)`, where the operand order matters (A, B as the RTL ports).
  - `fpu.from_fp32(x, man_w) -> int`, the same as the package's.
  - `fp_stim.svh`: `task automatic build_stimulus(input int n_random)` fills the including module's `va`/`vb` queues. It also provides `fp_mk(s, e, m)` and `fix_random_input(v)`. It needs `EXP_W`, `MAN_W`, `W` and `logic [W-1:0] va[$], vb[$]` declared before the include.
  - Dump line format (every TB that dumps): `<a hex> <b hex> <result hex> <ov><un><inv>` with the flags as three binary digits.

- [ ] **Step 1: Write the shared stimulus**

`Common/testbenches/fp_stim.svh`:

```systemverilog
// Shared stimulus for the float unit testbenches; the includer declares EXP_W, MAN_W, W and logic [W-1:0] va[$], vb[$].
// Every special class against every other, multiplier exponent edges, adder cancellations and alignment shifts, then random pairs.

function automatic logic [W-1:0] fp_mk(input bit s, input int e, input longint m);
  return {s, EXP_W'(e), MAN_W'(m)};
endfunction

// As the fp32 testbenches: subnormal exponents to 1, infinity/NaN exponents to the largest finite one.
function automatic logic [W-1:0] fix_random_input(input logic [W-1:0] v);
  if (v[W-2:MAN_W] == '0) return {v[W-1], EXP_W'(1), v[MAN_W-1:0]};
  if (v[W-2:MAN_W] == '1) return {v[W-1], EXP_W'((1 << EXP_W) - 2), v[MAN_W-1:0]};
  return v;
endfunction

task automatic build_stimulus(input int n_random);
  automatic int bias = (1 << (EXP_W - 1)) - 1;
  automatic int emax = (1 << EXP_W) - 1;
  automatic int max_e = emax - 1;
  automatic longint mmax = (longint'(1) << MAN_W) - 1;
  logic [W-1:0] special[$];
  for (int s = 0; s < 2; s++) begin
    special.push_back(fp_mk(s, 0, 0));  // zero
    special.push_back(fp_mk(s, 0, 1));  // subnormal patterns, read as zero
    special.push_back(fp_mk(s, 0, mmax));
    special.push_back(fp_mk(s, 1, 0));  // smallest normals
    special.push_back(fp_mk(s, 1, mmax));
    special.push_back(fp_mk(s, max_e, mmax));  // largest normal
    special.push_back(fp_mk(s, bias, 0));  // 1.0
    special.push_back(fp_mk(s, bias, mmax));  // just under 2.0
    special.push_back(fp_mk(s, emax, 0));  // infinity
    special.push_back(fp_mk(s, emax, longint'(1) << (MAN_W - 1)));  // quiet NaN
    special.push_back(fp_mk(s, emax, 1));  // signaling NaN
  end
  foreach (special[i])
    foreach (special[j]) begin
      va.push_back(special[i]);
      vb.push_back(special[j]);
    end
  // Multiplier edges: exponent sums around the bias (underflow) and around emax + bias (overflow), plain and all-ones mantissas.
  for (int ea = 1; ea <= max_e; ea++)
    for (int k = -4; k <= 4; k++)
      for (int t = 0; t < 2; t++) begin
        automatic int lo = bias - ea + k;
        automatic int hi = emax + bias - 1 - ea + k;
        automatic longint mx = t ? mmax : longint'($urandom);
        if (lo >= 1 && lo <= max_e) begin
          va.push_back(fp_mk($urandom_range(0, 1), ea, mx));
          vb.push_back(fp_mk($urandom_range(0, 1), lo, t ? mmax : longint'($urandom)));
        end
        if (hi >= 1 && hi <= max_e) begin
          va.push_back(fp_mk($urandom_range(0, 1), ea, mx));
          vb.push_back(fp_mk($urandom_range(0, 1), hi, t ? mmax : longint'($urandom)));
        end
      end
  // Adder edges: cancellations at the bottom of the range, exact cancellation, and every alignment shift past the sticky collapse.
  for (int e = 1; e <= MAN_W + 6; e++)
    for (int k = 0; k < 8; k++) begin
      automatic logic [W-1:0] x = fp_mk(0, e, longint'($urandom));
      automatic logic [W-1:0] y = x + W'($urandom_range(0, 3));
      va.push_back(x);
      vb.push_back({1'b1, y[W-2:0]});
      va.push_back({1'b1, x[W-2:0]});
      vb.push_back(x);
    end
  for (int d = 0; d <= MAN_W + 6; d++)
    for (int k = 0; k < 8; k++) begin
      automatic int ea = $urandom_range(d + 1, max_e);
      va.push_back(fp_mk(0, ea, longint'($urandom)));
      vb.push_back(fp_mk($urandom_range(0, 1), ea - d, longint'($urandom)));
    end
  for (int k = 0; k < 16; k++) begin  // sums that overflow
    va.push_back(fp_mk(0, max_e, longint'($urandom)));
    vb.push_back(fp_mk(0, max_e - (k % 2), longint'($urandom)));
  end
  // Random: mostly normal operands as in the fp32 testbenches, one in eight raw so every class appears.
  for (int i = 0; i < n_random; i++) begin
    automatic logic [W-1:0] ra = W'({$urandom, $urandom});
    automatic logic [W-1:0] rb = W'({$urandom, $urandom});
    if (i % 8 != 0) begin
      ra = fix_random_input(ra);
      rb = fix_random_input(rb);
    end
    va.push_back(ra);
    vb.push_back(rb);
  end
endtask
```

- [ ] **Step 2: Write the fp32 dump testbench**

`Common/testbenches/TB_fp32Dump.sv`:

```systemverilog
`timescale 1ns / 100ps

// Drives fp32Multiplier and fp32Adder with the shared stimulus and dumps every result for the Python model: mul32.txt, add32.txt.
module TB_fp32Dump #(
    parameter int RANDOM = 200000
);
  localparam int EXP_W = 8, MAN_W = 23, W = 32;
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0;
  logic [W-1:0] va[$], vb[$];
  `include "fp_stim.svh"

  logic [W-1:0] rm, ra;
  logic dm, da, ovm, ova, unm, una, ivm, iva;
  fp32Multiplier MUL (.clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(rm), .done_o(dm),
                      .overflow_o(ovm), .underflow_o(unm), .invalid_o(ivm));
  fp32Adder ADD (.clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(ra), .done_o(da),
                 .overflow_o(ova), .underflow_o(una), .invalid_o(iva));

  logic [W-1:0] qa[$], qb[$], qa2[$], qb2[$];
  int fm, fa, nm = 0, na = 0;
  always @(posedge clk) begin
    if (dm) begin
      $fwrite(fm, "%h %h %h %b%b%b\n", qa.pop_front(), qb.pop_front(), rm, ovm, unm, ivm);
      nm++;
    end
    if (da) begin
      $fwrite(fa, "%h %h %h %b%b%b\n", qa2.pop_front(), qb2.pop_front(), ra, ova, una, iva);
      na++;
    end
  end

  initial begin
    fm = $fopen("mul32.txt", "w");
    fa = $fopen("add32.txt", "w");
    build_stimulus(RANDOM);
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    foreach (va[i]) begin
      #1 valid = 1;
      a = va[i];
      b = vb[i];
      qa.push_back(a); qb.push_back(b); qa2.push_back(a); qb2.push_back(b);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $fclose(fm);
    $fclose(fa);
    $display("TB_fp32Dump: %0d inputs, %0d products, %0d sums", va.size(), nm, na);
    $display("RESULT: %s", (nm == va.size() && na == va.size()) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

Add the `dump32` target to `Common/Makefile` (below `pkg`, and add it to `.PHONY`):

```make
FP32_UNITS = $(ARIL)/Multipliers/Radix4Booth/src/R4Booth.sv $(ARIL)/Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
             $(ARIL)/Multipliers/FP32/src/fp32Multiplier.sv $(ARIL)/Adders/FP32/src/LZC.sv $(ARIL)/Adders/FP32/src/fp32Adder.sv

dump32:
	verilator $(FLAGS) --top-module TB_fp32Dump --Mdir $(PRJ_DIR)/Verilator/dump32 $(FP32_UNITS) $(TB_DIR)/TB_fp32Dump.sv -o sim
	cd $(PRJ_DIR)/Verilator/dump32 && ./sim $(PLUSARGS)
	python3 $(PRJ_DIR)/models/check_fpu.py $(PRJ_DIR)/Verilator/dump32/mul32.txt --unit mul --format fp32
	python3 $(PRJ_DIR)/models/check_fpu.py $(PRJ_DIR)/Verilator/dump32/add32.txt --unit add --format fp32 --fp32-unit
```

- [ ] **Step 3: Write the checker (the model does not exist yet)**

`Common/models/check_fpu.py`:

```python
#!/usr/bin/env python3
"""Checks a float-unit dump (a b result flags per line; flags are ov un inv as binary digits) against fpu.py, bit for bit."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpu  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump")
    p.add_argument("--unit", choices=["mul", "add"], required=True)
    p.add_argument("--format", choices=sorted(fpu.FORMATS), required=True)
    p.add_argument("--fp32-unit", action="store_true", help="the dump came from fp32Adder (its overflow flag, D-1)")
    a = p.parse_args()
    f = fpu.FORMATS[a.format]
    rows = [ln.split() for ln in open(a.dump) if ln.strip()]
    A = np.array([int(r[0], 16) for r in rows], dtype=np.int64)
    B = np.array([int(r[1], 16) for r in rows], dtype=np.int64)
    R = np.array([int(r[2], 16) for r in rows], dtype=np.int64)
    FL = np.array([int(r[3], 2) for r in rows], dtype=np.int64)
    if a.unit == "mul":
        res, ov, un, inv = fpu.mul(f, A, B)
    else:
        res, ov, un, inv = fpu.add(f, A, B, fp32_overflow_quirk=a.fp32_unit)
    flags = (ov.astype(np.int64) << 2) | (un.astype(np.int64) << 1) | inv.astype(np.int64)
    bad = np.nonzero((res != R) | (flags != FL))[0]
    d = (f.w + 3) // 4
    print(f"{a.unit} {a.format}: {len(rows)} results checked against fpu.py, {len(bad)} mismatches")
    for i in bad[:20]:
        print(f"  {A[i]:0{d}x} {B[i]:0{d}x}: rtl {R[i]:0{d}x} {FL[i]:03b}, model {res[i]:0{d}x} {flags[i]:03b}")
    sys.exit(1 if len(bad) or not rows else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it to see it fail**

Run: `$J/snap_launch.sh u2_dump 8 1 $J/cmd_aril.sh Common dump32`
Expected: `TB_fp32Dump: ... RESULT: PASSED`, then `ModuleNotFoundError: No module named 'fpu'`.

- [ ] **Step 5: Write the model**

`Common/models/fpu.py`:

```python
"""Bit-exact models of AriL's float units, vectorized: fpMultiplier and fpAdder at any (EXP_W, MAN_W), and at (8, 23)
fp32Multiplier and fp32Adder. Operands and results are bit patterns in int64 arrays; each op returns (result, overflow,
underflow, invalid). Truncating; subnormal inputs read as zero; tiny results flush to a signed zero; NaN is canonical."""
import numpy as np


class Fmt:
    def __init__(self, name: str, exp_w: int, man_w: int):
        self.name, self.e, self.m = name, exp_w, man_w
        self.w = 1 + exp_w + man_w
        self.bias = (1 << (exp_w - 1)) - 1
        self.emax = (1 << exp_w) - 1
        self.mmask = (1 << man_w) - 1
        self.qnan = (self.emax << man_w) | (1 << (man_w - 1))
        self.sig = man_w + 1


FP32, BF16 = Fmt("fp32", 8, 23), Fmt("bf16", 8, 7)
FORMATS = {"fp32": FP32, "bf16": BF16}


def from_fp32(x: int, man_w: int) -> int:
    """sienna_fmt_pkg::from_fp32: a finite fp32 constant rounded to nearest even at man_w bits."""
    sh = 23 - man_w
    return x if sh == 0 else (x + (1 << (sh - 1)) - 1 + ((x >> sh) & 1)) >> sh


def _split(f: Fmt, x):
    x = np.asarray(x, dtype=np.int64)
    return (x >> (f.w - 1)) & 1, (x >> f.m) & f.emax, x & f.mmask


def _bitlen(x):
    """Bit length of non-negative int64 values below 2^53."""
    _, e = np.frexp(x.astype(np.float64))
    return np.where(x == 0, 0, e).astype(np.int64)


def _classes(f: Fmt, e, m):
    """zero (subnormals too), infinity, NaN, signaling NaN."""
    return e == 0, (e == f.emax) & (m == 0), (e == f.emax) & (m != 0), (e == f.emax) & (m != 0) & (((m >> (f.m - 1)) & 1) == 0)


def mul(f: Fmt, a, b):
    sa, ea, ma = _split(f, a)
    sb, eb, mb = _split(f, b)
    za, ia, na, sna = _classes(f, ea, ma)
    zb, ib, nb, snb = _classes(f, eb, mb)
    sign = sa ^ sb
    inv = (za & ib) | (ia & zb)
    nan, inf, zero = na | nb | inv, ia | ib, za | zb
    s = ea + eb
    p = ((1 << f.m) | ma) * ((1 << f.m) | mb)
    top = (p >> (2 * f.m + 1)) & 1
    ov_sum, un_sum, pot = s >= f.emax + f.bias, s < f.bias, s == f.bias
    under = un_sum | (pot & (top == 0))
    fexp = ((s - f.bias) + top) & f.emax
    fman = np.where(top == 1, (p >> (f.m + 1)) & f.mmask, (p >> f.m) & f.mmask)
    sz = sign << (f.w - 1)
    infb = sz | (f.emax << f.m)
    res = np.select([nan, inf, zero, ov_sum, under, fexp == f.emax],
                    [np.full_like(s, f.qnan), infb, sz, infb, sz, infb], default=sz | (fexp << f.m) | fman)
    fin = ~nan & ~inf & ~zero
    ov = fin & (ov_sum | (~under & (fexp == f.emax)))
    un = fin & ~ov_sum & under
    return res, ov, un, inv | sna | snb


def add(f: Fmt, a, b, fp32_overflow_quirk: bool = False):
    """fpAdder(A=a, B=b); with fp32_overflow_quirk, fp32Adder, which also raises overflow on a negative exponent (D-1)."""
    sa, ea, ra = _split(f, a)
    sb, eb, rb = _split(f, b)
    za, ia, na, sna = _classes(f, ea, ra)
    zb, ib, nb, snb = _classes(f, eb, rb)
    ma, mb = np.where(za, 0, ra), np.where(zb, 0, rb)
    sub = sa ^ sb
    a_ge = ((ea << f.m) | ma) >= ((eb << f.m) | mb)
    inf_inf = ia & ib & (sa != sb)
    invalid = inf_inf | sna | snb
    nan, inf = na | nb | inf_inf, ia | ib
    zero = za & zb
    bypass = za ^ zb
    siga = ((~za).astype(np.int64) << f.m) | ma
    sigb = ((~zb).astype(np.int64) << f.m) | mb
    big, small = np.where(a_ge, siga, sigb), np.where(a_ge, sigb, siga)
    diff = np.where(a_ge, ea - eb, eb - ea)
    exp, sign = np.where(a_ge, ea, eb), np.where(a_ge, sa, sb)
    sw = f.sig + 4
    swm = (1 << sw) - 1
    big3 = big << 3
    small3 = np.where(bypass, 0, np.where(diff >= f.sig + 2, 1, (small << 3) >> np.minimum(diff, 62)))
    tot = np.where(sub == 1, big3 - small3, big3 + small3) & swm
    zero = zero | (tot == 0)
    lz = sw - _bitlen(tot)
    carry = ((tot >> (sw - 1)) & 1) == 1
    shift = np.maximum(lz - 1, 0)
    conds = [bypass, carry, lz == 1, zero, lz > 1]
    nm = np.select(conds, [tot, tot >> 1, tot, 0, (tot << shift) & swm], default=tot)
    ne = np.select(conds, [exp, exp + 1, exp, 0, exp - shift], default=exp) & ((1 << (f.e + 1)) - 1)
    neg = ((ne >> f.e) & 1) == 1
    under = (neg | (ne == 0)) & ~bypass
    ov_raw = ne >= f.emax
    ov = (ov_raw if fp32_overflow_quirk else (~neg & ov_raw)) & ~inf & ~nan
    sz = sign << (f.w - 1)
    infb = sz | (f.emax << f.m)
    res = np.select([nan, inf, zero, under, ov_raw], [np.full_like(ne, f.qnan), infb, sz, sz, infb],
                    default=sz | ((ne & f.emax) << f.m) | ((nm >> 3) & f.mmask))
    un = ~nan & ~inf & ~zero & under
    return res, ov, un, invalid
```

- [ ] **Step 6: Run it to see it pass**

Run: `$J/snap_launch.sh u2_dump 8 1 $J/cmd_aril.sh Common dump32`
Expected: `mul fp32: N results checked against fpu.py, 0 mismatches` and `add fp32: N results checked against fpu.py, 0 mismatches` (N is about 222,000). A mismatch means the model misreads fp32Multiplier or fp32Adder. Fix the model, never the RTL. For each mismatch, read the stage of the RTL that decides the differing field.

- [ ] **Step 7: Commit (AriL)**

```bash
git add Common/testbenches/fp_stim.svh && git commit -m "fp_stim.svh: shared float-unit stimulus (specials, exponent edges, cancellations, random)"
git add Common/models/fpu.py && git commit -m "fpu.py: bit-exact model of the float units at any width, fp32 units at (8, 23)"
git add Common/models/check_fpu.py Common/testbenches/TB_fp32Dump.sv Common/Makefile && git commit -m "Check fpu.py against fp32Multiplier and fp32Adder: 0 mismatches"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 3: `fpMultiplier` RTL, equivalent to fp32Multiplier at (8, 23)

**Files:**
- Create: `Multipliers/FP/src/fpMultiplier.sv`
- Create: `Multipliers/FP/testbenches/TB_fpMultiplierEq.sv`
- Create: `Multipliers/FP/Makefile`

**Interfaces:**
- Consumes: `sienna_fmt_pkg` (Task 1), `fp_stim.svh` (Task 2), `karatsubaUnsigned`, `R4Booth`, `fp32Multiplier` (exist).
- Produces: `module fpMultiplier #(int EXP_W = 8, int MAN_W = 7, int W = 1 + EXP_W + MAN_W) (clk_i, rstn_i, valid_i, A[W-1:0], B[W-1:0], result_o[W-1:0], done_o, overflow_o, underflow_o, invalid_o)`. The ports and handshake are fp32Multiplier's, and the latency is `sienna_fmt_pkg::mul_lat(EXP_W, MAN_W)`.

- [ ] **Step 1: Write the failing equivalence test**

`Multipliers/FP/testbenches/TB_fpMultiplierEq.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpMultiplier at EXP_W=8, MAN_W=23 against fp32Multiplier on the shared stimulus: results, flags and done cycles bit for bit.
module TB_fpMultiplierEq #(
    parameter int RANDOM = 200000
);
  localparam int EXP_W = 8, MAN_W = 23, W = 32;
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0;
  logic [W-1:0] va[$], vb[$];
  `include "fp_stim.svh"

  logic [W-1:0] rg, rr;
  logic dg, dr, og, orr, ug, ur, ig, ir;
  fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(rg), .done_o(dg), .overflow_o(og), .underflow_o(ug), .invalid_o(ig));
  fp32Multiplier ref_mul (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(rr), .done_o(dr), .overflow_o(orr), .underflow_o(ur), .invalid_o(ir));

  int errs = 0;
  longint checked = 0;
  logic [W-1:0] qa[$], qb[$];
  always @(posedge clk)
    if (rstn && (dg || dr)) begin
      logic [W-1:0] x, y;
      x = qa.pop_front();
      y = qb.pop_front();
      checked++;
      if (dg !== dr || {rg, og, ug, ig} !== {rr, orr, ur, ir}) begin
        errs++;
        if (errs <= 20)
          $display("[FAIL] %h * %h: fpMultiplier %h (done %b ov %b un %b inv %b), fp32Multiplier %h (done %b ov %b un %b inv %b)",
                   x, y, rg, dg, og, ug, ig, rr, dr, orr, ur, ir);
      end
    end

  initial begin
    build_stimulus(RANDOM);
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    foreach (va[i]) begin
      #1 valid = 1;
      a = va[i];
      b = vb[i];
      qa.push_back(a);
      qb.push_back(b);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (checked != va.size()) begin
      errs++;
      $display("[FAIL] %0d results for %0d inputs", checked, va.size());
    end
    $display("fpMultiplier (8, 23) against fp32Multiplier: %0d products, %0d mismatches", checked, errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Multipliers/FP/Makefile`. Task 4 adds the other targets; write the whole file now so both tasks share it:

```make
SHELL := /bin/bash
# fpMultiplier: eq (fp32Multiplier at 8/23), verilator (SoftFloat, bf16), check (bf16 against fpu.py), vivado_tb, vectors.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/../..
COMMON  = $(ARIL)/Common
TB_DIR  = $(PRJ_DIR)/testbenches
SF      = $(ARIL)/Adders/FP32/testbenches/berkeley-softfloat-3
SF_INC  = $(SF)/source/include
SF_LIB  = $(SF)/build/Linux-x86_64-GCC/softfloat.a
PLUSARGS ?=
DESIGN  = $(COMMON)/src/sienna_fmt_pkg.sv $(ARIL)/Multipliers/Radix4Booth/src/R4Booth.sv \
          $(ARIL)/Multipliers/Karatsuba/src/karatsubaUnsigned.sv $(PRJ_DIR)/src/fpMultiplier.sv
FLAGS   = --binary --timing --assert --sv -I$(COMMON)/testbenches -I$(TB_DIR) \
          --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
BF16    = -GEXP_W=8 -GMAN_W=7

eq:
	verilator $(FLAGS) --top-module TB_fpMultiplierEq --Mdir $(PRJ_DIR)/Verilator/eq $(DESIGN) \
	  $(ARIL)/Multipliers/FP32/src/fp32Multiplier.sv $(TB_DIR)/TB_fpMultiplierEq.sv -o sim
	$(PRJ_DIR)/Verilator/eq/sim $(PLUSARGS)

verilator: $(SF_LIB)
	verilator $(FLAGS) $(BF16) --top-module TB_fpMultiplier --Mdir $(PRJ_DIR)/Verilator/sf $(DESIGN) \
	  $(TB_DIR)/TB_fpMultiplier.sv $(COMMON)/testbenches/fp_ref.cpp \
	  -CFLAGS "-I$(SF_INC) -I$(COMMON)/testbenches" -LDFLAGS "$(SF_LIB)" -o sim
	cd $(PRJ_DIR)/Verilator/sf && ./sim $(PLUSARGS)

check: verilator
	cd $(PRJ_DIR)/Verilator/sf && ./sim +DUMP=mul_bf16.txt
	python3 $(COMMON)/models/check_fpu.py $(PRJ_DIR)/Verilator/sf/mul_bf16.txt --unit mul --format bf16

vivado_tb:
	verilator $(FLAGS) $(BF16) --top-module TB_fpMultiplierVIVADO --Mdir $(PRJ_DIR)/Verilator/viv $(DESIGN) \
	  $(TB_DIR)/TB_fpMultiplierVIVADO.sv -o sim
	cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv/sim

vectors: $(SF_LIB)
	$(COMMON)/testbenches/generate_vectors.sh

lint:
	for f in "-GEXP_W=8 -GMAN_W=7" "-GEXP_W=8 -GMAN_W=23"; do \
	  verilator --lint-only -Wall -DSYNTHESIS --top-module fpMultiplier $$f $(DESIGN) || exit 1; done

$(SF_LIB):
	$(MAKE) -C $(SF)/build/Linux-x86_64-GCC

.PHONY: eq verilator check vivado_tb vectors lint
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch.sh u3_eq 8 1 $J/cmd_aril.sh Multipliers/FP eq`
Expected: build error, `fpMultiplier.sv` not found.

- [ ] **Step 3: Write the unit**

`Multipliers/FP/src/fpMultiplier.sv`:

```systemverilog
`timescale 1ns / 100ps

// Float multiplier for EXP_W exponent and MAN_W mantissa bits: fp32Multiplier's algorithm, ports, flags and special values at any width.
// Truncates (the top MAN_W bits of the product); subnormal inputs read as zero, underflow flushes to a signed zero, overflow gives
// infinity, any NaN gives the canonical quiet NaN. At (8, 23) it matches fp32Multiplier bit for bit.
// valid_i at t, done_o at t+8 above 12 significand bits (Karatsuba, as fp32Multiplier), else t+3 (one-stage product).
module fpMultiplier #(
    parameter int EXP_W = 8,
    parameter int MAN_W = 7,
    parameter int W     = 1 + EXP_W + MAN_W
) (
    input  wire         clk_i,
    input  wire         rstn_i,
    input  wire         valid_i,
    input  wire [W-1:0] A,
    input  wire [W-1:0] B,
    output reg  [W-1:0] result_o,
    output reg          done_o,
    output reg          overflow_o,   // finite inputs, infinite result
    output reg          underflow_o,  // result flushed to zero
    output reg          invalid_o     // 0 * Inf, or a signaling NaN input
);
  localparam int SIG_W = MAN_W + 1;
  localparam int PW = 2 * SIG_W;
  localparam int BIAS = (1 << (EXP_W - 1)) - 1;
  localparam int EMAX = (1 << EXP_W) - 1;
  localparam int XW = EXP_W + 2;  // exponent sum
  localparam bit KARATSUBA = (SIG_W > 12);
  localparam int PROD_LAT = KARATSUBA ? 6 : 1;  // karatsubaUnsigned: valid_i at t, valid_o at t+6
  localparam logic [EXP_W-1:0] ONES = '1;
  localparam logic [W-1:0] QNAN = {1'b0, ONES, 1'b1, {(MAN_W - 1) {1'b0}}};

  typedef struct packed {
    logic          nan;
    logic          inv;
    logic          inf;
    logic          zero;
    logic          sign;
    logic [XW-1:0] sum;
  } meta_t;

  // Stage 1: classify, add the exponents, register the significands.
  logic s1_v;
  meta_t s1_m;
  logic [SIG_W-1:0] s1_sa, s1_sb;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      s1_v  <= 1'b0;
      s1_m  <= '0;
      s1_sa <= '0;
      s1_sb <= '0;
    end else begin
      s1_v <= valid_i;
      if (valid_i) begin
        automatic logic [EXP_W-1:0] ea = A[W-2:MAN_W], eb = B[W-2:MAN_W];
        automatic logic zero_a = (ea == '0), zero_b = (eb == '0);
        automatic logic inf_a = (ea == ONES) && (A[MAN_W-1:0] == '0), inf_b = (eb == ONES) && (B[MAN_W-1:0] == '0);
        automatic logic nan_a = (ea == ONES) && (A[MAN_W-1:0] != '0), nan_b = (eb == ONES) && (B[MAN_W-1:0] != '0);
        automatic logic snan = (nan_a && !A[MAN_W-1]) || (nan_b && !B[MAN_W-1]);
        automatic logic inv = (zero_a && inf_b) || (inf_a && zero_b);
        s1_m.nan  <= nan_a || nan_b || inv;
        s1_m.inv  <= inv || snan;
        s1_m.inf  <= inf_a || inf_b;
        s1_m.zero <= zero_a || zero_b;
        s1_m.sign <= A[W-1] ^ B[W-1];
        s1_m.sum  <= XW'(ea) + XW'(eb);
        s1_sa     <= {1'b1, A[MAN_W-1:0]};
        s1_sb     <= {1'b1, B[MAN_W-1:0]};
      end
    end
  end

  // Significand product: Karatsuba for wide significands, one registered multiply for narrow ones.
  logic          prod_v;
  logic [PW-1:0] prod;
  if (KARATSUBA) begin : G_KARATSUBA
    karatsubaUnsigned #(
        .WIDTH(SIG_W)
    ) u_mul (
        .clk_i         (clk_i),
        .rstn_i        (rstn_i),
        .valid_i       (s1_v),
        .multiplicand_i(s1_sa),
        .multiplier_i  (s1_sb),
        .valid_o       (prod_v),
        .product_o     (prod)
    );
  end else begin : G_DIRECT
    always_ff @(posedge clk_i or negedge rstn_i) begin
      if (!rstn_i) begin
        prod_v <= 1'b0;
        prod   <= '0;
      end else begin
        prod_v <= s1_v;
        if (s1_v) prod <= PW'(s1_sa) * PW'(s1_sb);
      end
    end
  end

  // The classification travels beside the product.
  meta_t m_d[PROD_LAT];
  always_ff @(posedge clk_i) begin
    m_d[0] <= s1_m;
    for (int i = 1; i < PROD_LAT; i++) m_d[i] <= m_d[i-1];
  end
  meta_t m;
  assign m = m_d[PROD_LAT-1];

  // Range from the exponent sum, as fp32Multiplier's bias stage; normalize a product in [2, 4) by one place.
  logic top, ov_sum, under;
  logic [EXP_W-1:0] fexp;
  logic [MAN_W-1:0] fman;
  always_comb begin
    top    = prod[PW-1];
    ov_sum = (m.sum >= XW'(EMAX + BIAS));
    under  = (m.sum < XW'(BIAS)) || ((m.sum == XW'(BIAS)) && !top);
    fexp   = EXP_W'(m.sum - XW'(BIAS)) + EXP_W'(top);
    fman   = top ? prod[PW-2-:MAN_W] : prod[PW-3-:MAN_W];
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      result_o    <= '0;
      done_o      <= 1'b0;
      overflow_o  <= 1'b0;
      underflow_o <= 1'b0;
      invalid_o   <= 1'b0;
    end else begin
      done_o      <= prod_v;
      overflow_o  <= 1'b0;
      underflow_o <= 1'b0;
      invalid_o   <= 1'b0;
      if (prod_v) begin
        invalid_o <= m.inv;
        if (m.nan) result_o <= QNAN;
        else if (m.inf) result_o <= {m.sign, ONES, {MAN_W{1'b0}}};
        else if (m.zero) result_o <= {m.sign, {(W - 1) {1'b0}}};
        else if (ov_sum) begin
          result_o   <= {m.sign, ONES, {MAN_W{1'b0}}};
          overflow_o <= 1'b1;
        end else if (under) begin
          result_o    <= {m.sign, {(W - 1) {1'b0}}};
          underflow_o <= 1'b1;
        end else if (fexp == ONES) begin
          result_o   <= {m.sign, ONES, {MAN_W{1'b0}}};
          overflow_o <= 1'b1;
        end else result_o <= {m.sign, fexp, fman};
      end
    end
  end

endmodule
```

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch.sh u3_eq 8 1 $J/cmd_aril.sh Multipliers/FP eq`
Expected: `fpMultiplier (8, 23) against fp32Multiplier: N products, 0 mismatches`, `RESULT: PASSED`. A done-cycle mismatch on every vector means PROD_LAT is wrong. Karatsuba's valid_o comes 6 edges after valid_i (stage 1, two in R4Booth, stages 4, 5 and 6). Count again rather than guess.

- [ ] **Step 5: Commit (AriL)**

```bash
git add Multipliers/FP/src/fpMultiplier.sv && git commit -m "fpMultiplier: fp32Multiplier's algorithm at any EXP_W/MAN_W, truncating"
git add Multipliers/FP/testbenches/TB_fpMultiplierEq.sv Multipliers/FP/Makefile && git commit -m "TB_fpMultiplierEq: fpMultiplier at (8, 23) matches fp32Multiplier bit for bit"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 4: fpMultiplier DV in bf16 (SoftFloat, Vivado vectors, model, exhaustive)

**Files:**
- Create: `Common/testbenches/fp_ref_core.h`, `Common/testbenches/fp_ref.cpp`, `Common/testbenches/gen_vectors.cpp`, `Common/testbenches/generate_vectors.sh`
- Create: `Multipliers/FP/testbenches/TB_fpMultiplier.sv`, `Multipliers/FP/testbenches/TB_fpMultiplierVIVADO.sv`, `Multipliers/FP/testbenches/vectors_bf16.mem` (generated)
- Create: `/proj/work/spramanik/sienna_jobs/cmd_aril_exh.sh`

**Interfaces:**
- Consumes: `fpMultiplier` (Task 3), `fpu.py` and `check_fpu.py` (Task 2), `fp_stim.svh`.
- Produces: `import "DPI-C" function int c_fp_mul(input int a, input int b, input int man_w, output int flags)` and `c_fp_add(...)` with the same signature. Both take right-aligned narrow bit patterns and return the expected result, with SoftFloat flag bits (0 inexact, 1 underflow, 2 overflow, 4 invalid). `gen_vectors mul|add MAN_W COUNT OUT` writes one line per vector, `A B Res Flags` in hex with W/4 digits each and 2 for the flags.

- [ ] **Step 1: Write the reference**

`Common/testbenches/fp_ref_core.h`:

```cpp
// SoftFloat reference for floats with an 8-bit exponent: exact widening to f32, the op rounded to odd at 24 bits, then nearest-even
// at MAN_W bits, which is correctly rounded (MAN_W = 23 is plain f32 nearest-even). The units' policy on top: subnormal inputs read
// as zero, results tiny before rounding flush to a signed zero with underflow, NaN is canonical.
#pragma once
#include <cstdint>
extern "C" {
#include "softfloat.h"
}

static inline uint32_t fpref_widen(uint32_t x, int man_w) {
  uint32_t f = x << (23 - man_w);
  if (((f >> 23) & 0xFFu) == 0) f &= 0x80000000u;  // subnormal reads as zero
  return f;
}

static inline uint32_t fpref_narrow(uint32_t f, int man_w, int *flags) {
  const uint32_t sh = 23 - man_w, sign = f & 0x80000000u, mag = f & 0x7FFFFFFFu;
  if ((mag >> 23) == 0xFFu) return (mag & 0x7FFFFFu) ? (0x7FC00000u >> sh) : ((sign | 0x7F800000u) >> sh);
  if ((*flags & softfloat_flag_underflow) || ((mag >> 23) == 0 && mag != 0)) {  // tiny: flushed
    *flags |= softfloat_flag_underflow | softfloat_flag_inexact;
    return sign >> sh;
  }
  if (sh == 0) return f;
  uint32_t r = (mag + (1u << (sh - 1)) - 1u + ((mag >> sh) & 1u)) >> sh;
  if (mag & ((1u << sh) - 1u)) *flags |= softfloat_flag_inexact;
  if ((r >> man_w) >= 0xFFu) {  // rounded up to infinity
    *flags |= softfloat_flag_overflow | softfloat_flag_inexact;
    r = 0xFFu << man_w;
  }
  return (sign >> sh) | r;
}

// op 0 multiplies, op 1 adds; returns the expected result bits and sets *flags.
static inline uint32_t fpref_op(int op, uint32_t a, uint32_t b, int man_w, int *flags) {
  float32_t fa, fb, fr;
  fa.v = fpref_widen(a, man_w);
  fb.v = fpref_widen(b, man_w);
  softfloat_roundingMode = (man_w == 23) ? softfloat_round_near_even : softfloat_round_odd;
  softfloat_detectTininess = softfloat_tininess_beforeRounding;
  softfloat_exceptionFlags = 0;
  fr = op ? f32_add(fa, fb) : f32_mul(fa, fb);
  *flags = softfloat_exceptionFlags;
  return fpref_narrow(fr.v, man_w, flags);
}
```

`Common/testbenches/fp_ref.cpp`:

```cpp
// DPI wrappers of fp_ref_core.h for the float-unit testbenches.
#include <svdpi.h>
#include "fp_ref_core.h"

extern "C" int c_fp_mul(int a, int b, int man_w, int *flags) {
  return (int)fpref_op(0, (uint32_t)a, (uint32_t)b, man_w, flags);
}

extern "C" int c_fp_add(int a, int b, int man_w, int *flags) {
  return (int)fpref_op(1, (uint32_t)a, (uint32_t)b, man_w, flags);
}
```

`Common/testbenches/gen_vectors.cpp`:

```cpp
// Writes a vectors file for the Vivado testbenches, A B Res Flags in hex per line; args: mul|add MAN_W COUNT OUT.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fp_ref_core.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: gen_vectors mul|add MAN_W COUNT OUT\n");
    return 1;
  }
  const int op = strcmp(argv[1], "add") == 0, man_w = atoi(argv[2]), count = atoi(argv[3]);
  const int w = 9 + man_w, d = (w + 3) / 4;
  const uint32_t mask = (w == 32) ? 0xFFFFFFFFu : ((1u << w) - 1u), sh = 23 - man_w;
  // The fp32 generators' corners, narrowed: zeros, 1.0, infinities, NaN, cancellation, a subnormal input.
  const uint32_t corner[][2] = {{0x00000000u, 0x00000000u}, {0x3F800000u, 0x00000000u}, {0x7F800000u, 0x3F800000u},
                                {0x7F800000u, 0xFF800000u}, {0x7FC00000u, 0x3F800000u}, {0x3FC00000u, 0xBF800000u},
                                {0x3F810000u, 0xBF800000u}, {0x00400000u, 0x3F800000u}, {0x00000000u, 0x7F800000u}};
  const int nc = sizeof(corner) / sizeof(corner[0]);
  FILE *out = fopen(argv[4], "w");
  if (!out) {
    perror(argv[4]);
    return 1;
  }
  srand(42);
  for (int i = 0; i < count; i++) {
    uint32_t a, b;
    if (i < nc) {
      a = corner[i][0] >> sh;
      b = corner[i][1] >> sh;
    } else {
      a = ((uint32_t)rand() ^ ((uint32_t)rand() << 16)) & mask;
      b = ((uint32_t)rand() ^ ((uint32_t)rand() << 16)) & mask;
    }
    int flags = 0;
    const uint32_t r = fpref_op(op, a, b, man_w, &flags);
    fprintf(out, "%0*x%0*x%0*x%02x\n", d, a, d, b, d, r, flags & 0xFF);
  }
  fclose(out);
  printf("%s: %d %s vectors, MAN_W=%d\n", argv[4], count, op ? "add" : "mul", man_w);
  return 0;
}
```

`Common/testbenches/generate_vectors.sh` (`chmod +x`):

```bash
#!/bin/bash
# Builds gen_vectors against SoftFloat and writes the bf16 vectors for fpMultiplier's and fpAdder's Vivado testbenches.
cd "$(dirname "$0")" || exit 1
ARIL=../..
SF=$ARIL/Adders/FP32/testbenches/berkeley-softfloat-3
LIB=$SF/build/Linux-x86_64-GCC/softfloat.a
[ -f $LIB ] || make -C $SF/build/Linux-x86_64-GCC || exit 1
g++ -O2 -o gen_vectors gen_vectors.cpp $LIB -I$SF/source/include || exit 1
./gen_vectors mul 7 10000 $ARIL/Multipliers/FP/testbenches/vectors_bf16.mem || exit 1
[ -d $ARIL/Adders/FP/testbenches ] && { ./gen_vectors add 7 10000 $ARIL/Adders/FP/testbenches/vectors_bf16.mem || exit 1; }
exit 0
```

- [ ] **Step 2: Write the SoftFloat testbench**

`Multipliers/FP/testbenches/TB_fpMultiplier.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpMultiplier against Berkeley SoftFloat through DPI with TB_fp32Multiplier's rules: 2 ulp, any NaN for a NaN, flags checked,
// a flush accepted where the reference is tiny. Shared stimulus by default; +A_LO=+A_HI= runs every b for each a in [A_LO, A_HI).
// +DUMP=<file> writes every result for check_fpu.py.
module TB_fpMultiplier #(
    parameter int EXP_W  = 8,
    parameter int MAN_W  = 7,
    parameter int RANDOM = 200000
);
  localparam int W = 1 + EXP_W + MAN_W;
  localparam int TOL = 2;
  import "DPI-C" function int c_fp_mul(input int a, input int b, input int man_w, output int flags);

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0;
  logic [W-1:0] va[$], vb[$];
  `include "fp_stim.svh"

  logic [W-1:0] res;
  logic done, ov, un, inv;
  fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(res), .done_o(done), .overflow_o(ov), .underflow_o(un), .invalid_o(inv));

  typedef struct {
    logic [W-1:0] a, b, res;
    bit ov, un, inv;
  } tx_t;
  tx_t q[$];
  longint checked = 0, errs = 0;
  int fd = 0, cyc = 0, t_issue = -1, lat = -1;
  string dump;

  function automatic bit is_nan(input logic [W-1:0] x);
    return (x[W-2:MAN_W] == '1) && (x[MAN_W-1:0] != '0);
  endfunction

  task automatic drive(input logic [W-1:0] x, input logic [W-1:0] y);
    tx_t t;
    int fl;
    t.a = x;
    t.b = y;
    t.res = W'(c_fp_mul(int'(x), int'(y), MAN_W, fl));
    t.un = fl[1];
    t.ov = fl[2];
    t.inv = fl[4];
    q.push_back(t);
    #1 valid = 1;
    a = x;
    b = y;
    @(posedge clk);
  endtask

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      bit rm, fm;
      int diff;
      if (lat < 0) lat = cyc - t_issue - 1;  // done_o is sampled one edge after it rises
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        diff = int'(res) - int'(t.res);
        if (diff < 0) diff = -diff;
        rm = (res == t.res) || (is_nan(t.res) && is_nan(res)) || (!is_nan(t.res) && diff <= TOL) ||
             (un && res[W-2:0] == '0 && t.res[W-2:MAN_W] == '0);
        fm = (ov == t.ov) && (inv == t.inv) && ((un == t.un) || (un && res[W-2:0] == '0));
        if (!rm || !fm) begin
          errs++;
          if (errs <= 20)
            $display("[FAIL] %h * %h: got %h (ov %b un %b inv %b), SoftFloat %h (ov %b un %b inv %b)",
                     t.a, t.b, res, ov, un, inv, t.res, t.ov, t.un, t.inv);
        end
        if (fd != 0) $fwrite(fd, "%h %h %h %b%b%b\n", t.a, t.b, res, ov, un, inv);
      end
    end
  end

  initial begin
    int lo, hi;
    if (EXP_W != 8) $fatal(1, "TB_fpMultiplier: the SoftFloat reference needs an 8-bit exponent");
    if ($value$plusargs("DUMP=%s", dump)) fd = $fopen(dump, "w");
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("A_LO=%d", lo) && $value$plusargs("A_HI=%d", hi)) begin
      for (int x = lo; x < hi; x++)
        for (int y = 0; y < (1 << W); y++) drive(W'(x), W'(y));
    end else begin
      build_stimulus(RANDOM);
      foreach (va[i]) drive(va[i], vb[i]);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (fd != 0) $fclose(fd);
    if (q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d results never came out", q.size());
    end
    if (lat != sienna_fmt_pkg::mul_lat(EXP_W, MAN_W)) begin
      errs++;
      $display("[FAIL] latency %0d, sienna_fmt_pkg::mul_lat says %0d", lat, sienna_fmt_pkg::mul_lat(EXP_W, MAN_W));
    end
    $display("fpMultiplier EXP_W=%0d MAN_W=%0d: %0d products against SoftFloat, %0d errors, latency %0d", EXP_W, MAN_W,
             checked, errs, lat);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 3: Write the Vivado testbench**

`Multipliers/FP/testbenches/TB_fpMultiplierVIVADO.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpMultiplier against a gen_vectors.cpp file (A B Res Flags in hex) with TB_fpMultiplier's rules; no DPI, so it runs in Vivado.
module TB_fpMultiplierVIVADO #(
    parameter int    EXP_W       = 8,
    parameter int    MAN_W       = 7,
    parameter int    NUM_VECTORS = 10000,
    parameter string VEC_FILE    = "vectors_bf16.mem"
);
  localparam int W = 1 + EXP_W + MAN_W;
  localparam int TOL = 2;
  logic [3*W+7:0] vec[NUM_VECTORS];
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0, res;
  logic done, ov, un, inv;
  fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(res), .done_o(done), .overflow_o(ov), .underflow_o(un), .invalid_o(inv));

  int errs = 0, checked = 0;
  logic [3*W+7:0] q[$];

  function automatic bit is_nan(input logic [W-1:0] x);
    return (x[W-2:MAN_W] == '1) && (x[MAN_W-1:0] != '0);
  endfunction

  always @(posedge clk)
    if (rstn && done) begin
      logic [W-1:0] ea, eb, er;
      logic [7:0] fl;
      bit rm, fm;
      int diff;
      {ea, eb, er, fl} = q.pop_front();
      checked++;
      diff = int'(res) - int'(er);
      if (diff < 0) diff = -diff;
      rm = (res == er) || (is_nan(er) && is_nan(res)) || (!is_nan(er) && diff <= TOL) ||
           (un && res[W-2:0] == '0 && er[W-2:MAN_W] == '0);
      fm = (ov == fl[2]) && (inv == fl[4]) && ((un == fl[1]) || (un && res[W-2:0] == '0));
      if (!rm || !fm) begin
        errs++;
        if (errs <= 20) $display("[FAIL] %h * %h: got %h (ov %b un %b inv %b), expected %h flags %h", ea, eb, res, ov, un, inv, er, fl);
      end
    end

  initial begin
    $readmemh(VEC_FILE, vec);
    if (^vec[0] === 1'bx) $fatal(1, "TB_fpMultiplierVIVADO: could not read %s", VEC_FILE);
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    for (int i = 0; i < NUM_VECTORS; i++) begin
      #1 valid = 1;
      {a, b} = vec[i][3*W+7:W+8];
      q.push_back(vec[i]);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $display("fpMultiplier vectors %s: %0d checked, %0d errors", VEC_FILE, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NUM_VECTORS) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 4: Run SoftFloat and model checks; expect them to pass or show real bugs**

```bash
$J/snap_launch.sh u4_sf 8 1 $J/cmd_aril.sh Multipliers/FP verilator
$J/snap_launch.sh u4_chk 8 1 $J/cmd_aril.sh Multipliers/FP check
```

Expected: `fpMultiplier EXP_W=8 MAN_W=7: N products against SoftFloat, 0 errors, latency 3` and `mul bf16: N results checked against fpu.py, 0 mismatches`. A SoftFloat error with a result off by more than 2 ulp, or a flag mismatch, is a unit bug. A mismatch against the model is a bug in the unit or the model. Decide which by the fp32 evidence: the model already matches fp32Multiplier (Task 2), and the unit at (8, 23) matches fp32Multiplier (Task 3).

- [ ] **Step 5: Generate the vectors, run the Vivado testbench**

```bash
$J/snap_launch.sh u4_vec 4 1 bash -c 'SystolicMesh/ArithmeticLibrary/Common/testbenches/generate_vectors.sh && mkdir -p testbenches/results/uniform && cp SystolicMesh/ArithmeticLibrary/Multipliers/FP/testbenches/vectors_bf16.mem testbenches/results/uniform/fpMultiplier_vectors_bf16.mem'
cp /proj/work/spramanik/sienna_jobs/runs/u4_vec/results/uniform/fpMultiplier_vectors_bf16.mem SystolicMesh/ArithmeticLibrary/Multipliers/FP/testbenches/vectors_bf16.mem
$J/snap_launch.sh u4_viv 4 1 $J/cmd_aril.sh Multipliers/FP vivado_tb
```

Before the copy, check that the local target does not exist yet. The copy must not overwrite a file you have not looked at. Expected: `wc -l` gives 10000, and the Vivado TB prints `10000 checked, 0 errors`.

- [ ] **Step 6: Exhaustive run over every bf16 pair**

`/proj/work/spramanik/sienna_jobs/cmd_aril_exh.sh`:

```bash
#!/bin/bash
# One slice of an exhaustive bf16 check: every b against a in [LO, HI); args: UNIT LO HI; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
exec /proj/work/spramanik/sienna_jobs/cmd_aril.sh "$1" verilator +A_LO=$2 +A_HI=$3
```

Launch 16 slices of 4096 a-values each:

```bash
for s in $(seq 0 15); do $J/snap_launch.sh u4_exh_$s 8 12 $J/cmd_aril_exh.sh Multipliers/FP $((s*4096)) $(((s+1)*4096)); done
```

Expected: every slice prints `268435456 products against SoftFloat, 0 errors`. Record wall time per slice from `/usr/bin/time` in each `stdout.log`. If a slice exceeds 12 hours, report the count reached and relaunch that slice split in four; do not drop it.

- [ ] **Step 7: Commit (AriL)**

```bash
git add Common/testbenches/fp_ref_core.h Common/testbenches/fp_ref.cpp && git commit -m "fp_ref: SoftFloat reference for 8-bit-exponent formats, round to odd then nearest"
git add Common/testbenches/gen_vectors.cpp Common/testbenches/generate_vectors.sh && git commit -m "gen_vectors: vectors files for the narrow units' Vivado testbenches"
git add Multipliers/FP/testbenches/TB_fpMultiplier.sv && git commit -m "TB_fpMultiplier: bf16 against SoftFloat, shared stimulus or exhaustive, dump for fpu.py"
git add Multipliers/FP/testbenches/TB_fpMultiplierVIVADO.sv Multipliers/FP/testbenches/vectors_bf16.mem && git commit -m "TB_fpMultiplierVIVADO and its bf16 vectors"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 5: `fpAdder` RTL, equivalent to fp32Adder at (8, 23) except D-1

**Files:**
- Create: `Adders/FP/src/fpAdder.sv`, `Adders/FP/testbenches/TB_fpAdderEq.sv`, `Adders/FP/Makefile`

**Interfaces:**
- Consumes: `sienna_fmt_pkg`, `fp_stim.svh`, `fp32Adder` and `LZC.sv` (exist).
- Produces: `module fpAdder #(int EXP_W = 8, int MAN_W = 7, int W = 1 + EXP_W + MAN_W)` with fp32Adder's ports. Latency is `sienna_fmt_pkg::add_lat` (5). Operand order matters for results that are exactly zero (the sign comes from A on a magnitude tie), as in fp32Adder.

- [ ] **Step 1: Write the failing equivalence test**

`Adders/FP/testbenches/TB_fpAdderEq.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpAdder at EXP_W=8, MAN_W=23 against fp32Adder on the shared stimulus: results, underflow, invalid and done cycles bit for bit;
// overflow too, except where fp32Adder raises it on a cancellation that underflows (D-1), which is counted and listed.
module TB_fpAdderEq #(
    parameter int RANDOM = 200000
);
  localparam int EXP_W = 8, MAN_W = 23, W = 32;
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0;
  logic [W-1:0] va[$], vb[$];
  `include "fp_stim.svh"

  logic [W-1:0] rg, rr;
  logic dg, dr, og, orr, ug, ur, ig, ir;
  fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(rg), .done_o(dg), .overflow_o(og), .underflow_o(ug), .invalid_o(ig));
  fp32Adder ref_add (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(rr), .done_o(dr), .overflow_o(orr), .underflow_o(ur), .invalid_o(ir));

  int errs = 0, quirk = 0;
  longint checked = 0;
  logic [W-1:0] qa[$], qb[$];
  always @(posedge clk)
    if (rstn && (dg || dr)) begin
      logic [W-1:0] x, y;
      x = qa.pop_front();
      y = qb.pop_front();
      checked++;
      if (dg !== dr || {rg, ug, ig} !== {rr, ur, ir}) begin
        errs++;
        if (errs <= 20)
          $display("[FAIL] %h + %h: fpAdder %h (done %b ov %b un %b inv %b), fp32Adder %h (done %b ov %b un %b inv %b)",
                   x, y, rg, dg, og, ug, ig, rr, dr, orr, ur, ir);
      end else if (og !== orr) begin
        if (orr && !og && ur && rr[W-2:0] == '0) begin
          quirk++;
          if (quirk <= 5) $display("[D-1] %h + %h: fp32Adder raises overflow with a flushed underflow", x, y);
        end else begin
          errs++;
          if (errs <= 20) $display("[FAIL] %h + %h: overflow fpAdder %b, fp32Adder %b", x, y, og, orr);
        end
      end
    end

  initial begin
    build_stimulus(RANDOM);
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    foreach (va[i]) begin
      #1 valid = 1;
      a = va[i];
      b = vb[i];
      qa.push_back(a);
      qb.push_back(b);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (checked != va.size()) begin
      errs++;
      $display("[FAIL] %0d results for %0d inputs", checked, va.size());
    end
    $display("fpAdder (8, 23) against fp32Adder: %0d sums, %0d mismatches, %0d D-1 overflow flags", checked, errs, quirk);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Adders/FP/Makefile`:

```make
SHELL := /bin/bash
# fpAdder: eq (fp32Adder at 8/23), verilator (SoftFloat, bf16), check (bf16 against fpu.py), vivado_tb, vectors.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/../..
COMMON  = $(ARIL)/Common
TB_DIR  = $(PRJ_DIR)/testbenches
SF      = $(ARIL)/Adders/FP32/testbenches/berkeley-softfloat-3
SF_INC  = $(SF)/source/include
SF_LIB  = $(SF)/build/Linux-x86_64-GCC/softfloat.a
PLUSARGS ?=
DESIGN  = $(COMMON)/src/sienna_fmt_pkg.sv $(PRJ_DIR)/src/fpAdder.sv
FLAGS   = --binary --timing --assert --sv -I$(COMMON)/testbenches -I$(TB_DIR) \
          --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
BF16    = -GEXP_W=8 -GMAN_W=7

eq:
	verilator $(FLAGS) --top-module TB_fpAdderEq --Mdir $(PRJ_DIR)/Verilator/eq $(DESIGN) \
	  $(ARIL)/Adders/FP32/src/LZC.sv $(ARIL)/Adders/FP32/src/fp32Adder.sv $(TB_DIR)/TB_fpAdderEq.sv -o sim
	$(PRJ_DIR)/Verilator/eq/sim $(PLUSARGS)

verilator: $(SF_LIB)
	verilator $(FLAGS) $(BF16) --top-module TB_fpAdder --Mdir $(PRJ_DIR)/Verilator/sf $(DESIGN) \
	  $(TB_DIR)/TB_fpAdder.sv $(COMMON)/testbenches/fp_ref.cpp \
	  -CFLAGS "-I$(SF_INC) -I$(COMMON)/testbenches" -LDFLAGS "$(SF_LIB)" -o sim
	cd $(PRJ_DIR)/Verilator/sf && ./sim $(PLUSARGS)

check: verilator
	cd $(PRJ_DIR)/Verilator/sf && ./sim +DUMP=add_bf16.txt
	python3 $(COMMON)/models/check_fpu.py $(PRJ_DIR)/Verilator/sf/add_bf16.txt --unit add --format bf16

vivado_tb:
	verilator $(FLAGS) $(BF16) --top-module TB_fpAdderVIVADO --Mdir $(PRJ_DIR)/Verilator/viv $(DESIGN) \
	  $(TB_DIR)/TB_fpAdderVIVADO.sv -o sim
	cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv/sim

vectors: $(SF_LIB)
	$(COMMON)/testbenches/generate_vectors.sh

lint:
	for f in "-GEXP_W=8 -GMAN_W=7" "-GEXP_W=8 -GMAN_W=23"; do \
	  verilator --lint-only -Wall -DSYNTHESIS --top-module fpAdder $$f $(DESIGN) || exit 1; done

$(SF_LIB):
	$(MAKE) -C $(SF)/build/Linux-x86_64-GCC

.PHONY: eq verilator check vivado_tb vectors lint
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch.sh u5_eq 8 1 $J/cmd_aril.sh Adders/FP eq`
Expected: build error, `fpAdder.sv` not found.

- [ ] **Step 3: Write the unit**

`Adders/FP/src/fpAdder.sv`:

```systemverilog
`timescale 1ns / 100ps

// Float adder for EXP_W exponent and MAN_W mantissa bits: fp32Adder's algorithm, ports, flags and special values at any width.
// Aligns with three extra low bits, collapses shifts of MAN_W+3 or more into a single 1, normalizes by leading zeros, truncates.
// Subnormal inputs read as zero, underflow flushes to a signed zero, NaN is the canonical quiet NaN. valid_i at t, done_o at t+5.
// At (8, 23) it matches fp32Adder bit for bit, except that overflow_o stays low when a cancellation underflows (plan D-1).
module fpAdder #(
    parameter int EXP_W = 8,
    parameter int MAN_W = 7,
    parameter int W     = 1 + EXP_W + MAN_W
) (
    input  wire         clk_i,
    input  wire         rstn_i,
    input  wire         valid_i,
    input  wire [W-1:0] A,
    input  wire [W-1:0] B,
    output reg  [W-1:0] result_o,
    output reg          done_o,
    output reg          overflow_o,
    output reg          underflow_o,
    output reg          invalid_o
);
  localparam int SIG_W = MAN_W + 1;
  localparam int GW = SIG_W + 3;  // aligned significand with three extra low bits
  localparam int SW = GW + 1;  // sum with its carry
  localparam int LW = $clog2(SW + 1);
  localparam int NW = EXP_W + 1;  // normalized exponent with a sign bit
  localparam int EMAX = (1 << EXP_W) - 1;
  localparam logic [EXP_W-1:0] ONES = '1;
  localparam logic [W-1:0] QNAN = {1'b0, ONES, 1'b1, {(MAN_W - 1) {1'b0}}};

  typedef struct packed {
    logic             inv;
    logic             nan;
    logic             inf;
    logic             zero;
    logic             bypass;  // exactly one operand is zero: the other passes through
    logic             sign;
    logic [EXP_W-1:0] exp;
    logic             sub;
  } meta_t;

  // Leading zeros of the sum; SW when it is zero.
  function automatic logic [LW-1:0] lzc(input logic [SW-1:0] x);
    lzc = LW'(SW);
    for (int i = 0; i < SW; i++) if (x[i]) lzc = LW'(SW - 1 - i);
  endfunction

  // Stage 1: unpack, classify, order the operands by magnitude.
  logic s1_v;
  logic [EXP_W-1:0] s1_diff;
  logic [SIG_W-1:0] s1_big, s1_small;
  meta_t s1_m;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      s1_v     <= 1'b0;
      s1_diff  <= '0;
      s1_big   <= '0;
      s1_small <= '0;
      s1_m     <= '0;
    end else begin
      s1_v <= valid_i;
      if (valid_i) begin
        automatic logic [EXP_W-1:0] ea = A[W-2:MAN_W], eb = B[W-2:MAN_W];
        automatic logic za = (ea == '0), zb = (eb == '0);
        automatic logic [MAN_W-1:0] ma = za ? '0 : A[MAN_W-1:0], mb = zb ? '0 : B[MAN_W-1:0];
        automatic logic ia = (ea == ONES) && (A[MAN_W-1:0] == '0), ib = (eb == ONES) && (B[MAN_W-1:0] == '0);
        automatic logic na = (ea == ONES) && (A[MAN_W-1:0] != '0), nb = (eb == ONES) && (B[MAN_W-1:0] != '0);
        automatic logic snan = (na && !A[MAN_W-1]) || (nb && !B[MAN_W-1]);
        automatic logic inf_inf = ia && ib && (A[W-1] != B[W-1]);
        automatic logic a_ge = {ea, ma} >= {eb, mb};
        s1_m.inv    <= inf_inf || snan;
        s1_m.nan    <= na || nb || inf_inf;
        s1_m.inf    <= ia || ib;
        s1_m.zero   <= za && zb;
        s1_m.bypass <= za ^ zb;
        s1_m.sub    <= A[W-1] ^ B[W-1];
        s1_m.exp    <= a_ge ? ea : eb;
        s1_m.sign   <= a_ge ? A[W-1] : B[W-1];
        s1_big      <= a_ge ? {!za, ma} : {!zb, mb};
        s1_small    <= a_ge ? {!zb, mb} : {!za, ma};
        s1_diff     <= a_ge ? ea - eb : eb - ea;
      end
    end
  end

  // Stage 2: align the smaller significand.
  logic s2_v;
  logic [GW-1:0] s2_big, s2_small;
  meta_t s2_m;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) s2_v <= 1'b0;
    else s2_v <= s1_v;
  always_ff @(posedge clk_i)
    if (s1_v) begin
      s2_m   <= s1_m;
      s2_big <= {s1_big, 3'b000};
      if (s1_m.bypass) s2_small <= '0;
      else if (s1_diff >= EXP_W'(SIG_W + 2)) s2_small <= GW'(1);
      else s2_small <= {s1_small, 3'b000} >> s1_diff;
    end

  // Stage 3: add or subtract.
  logic s3_v;
  logic [SW-1:0] s3_sum;
  meta_t s3_m;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) s3_v <= 1'b0;
    else s3_v <= s2_v;
  always_ff @(posedge clk_i)
    if (s2_v) begin
      s3_m   <= s2_m;
      s3_sum <= s2_m.sub ? {1'b0, s2_big} - {1'b0, s2_small} : {1'b0, s2_big} + {1'b0, s2_small};
    end

  // Stage 4: count leading zeros.
  logic s4_v;
  logic [SW-1:0] s4_sum;
  logic [LW-1:0] s4_lz;
  meta_t s4_m;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) s4_v <= 1'b0;
    else s4_v <= s3_v;
  always_ff @(posedge clk_i)
    if (s3_v) begin
      s4_sum <= s3_sum;
      s4_m   <= s3_m;
      s4_lz  <= lzc(s3_sum);
      if (s3_sum == '0) s4_m.zero <= 1'b1;
    end

  // Normalize, in fp32Adder's order of cases.
  logic [SW-1:0] norm_man;
  logic [NW-1:0] norm_exp;
  always_comb begin
    norm_man = s4_sum;
    norm_exp = NW'(s4_m.exp);
    if (!s4_m.bypass) begin
      if (s4_sum[SW-1]) begin
        norm_man = s4_sum >> 1;
        norm_exp = NW'(s4_m.exp) + NW'(1);
      end else if (s4_lz == LW'(1)) norm_man = s4_sum;
      else if (s4_m.zero) begin
        norm_man = '0;
        norm_exp = '0;
      end else if (s4_lz > LW'(1)) begin
        norm_man = s4_sum << (s4_lz - LW'(1));
        norm_exp = NW'(s4_m.exp) - NW'(s4_lz - LW'(1));
      end
    end
  end

  logic neg, nonpos;
  assign neg    = norm_exp[NW-1];
  assign nonpos = neg || (norm_exp == '0);

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      result_o    <= '0;
      done_o      <= 1'b0;
      overflow_o  <= 1'b0;
      underflow_o <= 1'b0;
      invalid_o   <= 1'b0;
    end else begin
      done_o      <= s4_v;
      overflow_o  <= 1'b0;
      underflow_o <= 1'b0;
      invalid_o   <= 1'b0;
      if (s4_v) begin
        invalid_o  <= s4_m.inv;
        overflow_o <= !neg && (norm_exp >= NW'(EMAX)) && !s4_m.inf && !s4_m.nan;
        if (s4_m.nan) result_o <= QNAN;
        else if (s4_m.inf) result_o <= {s4_m.sign, ONES, {MAN_W{1'b0}}};
        else if (s4_m.zero) result_o <= {s4_m.sign, {(W - 1) {1'b0}}};
        else if (nonpos && !s4_m.bypass) begin
          result_o    <= {s4_m.sign, {(W - 1) {1'b0}}};
          underflow_o <= 1'b1;
        end else if (norm_exp >= NW'(EMAX)) result_o <= {s4_m.sign, ONES, {MAN_W{1'b0}}};
        else result_o <= {s4_m.sign, norm_exp[EXP_W-1:0], norm_man[GW-2-:MAN_W]};
      end
    end
  end

endmodule
```

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch.sh u5_eq 8 1 $J/cmd_aril.sh Adders/FP eq`
Expected: `fpAdder (8, 23) against fp32Adder: N sums, 0 mismatches, K D-1 overflow flags` with K > 0, since the stimulus's bottom-of-range cancellations reach it, and `RESULT: PASSED`. If K is 0, the stimulus missed D-1. Check with the Task 0 vector before accepting.

- [ ] **Step 5: Commit (AriL)**

```bash
git add Adders/FP/src/fpAdder.sv && git commit -m "fpAdder: fp32Adder's algorithm at any EXP_W/MAN_W, truncating"
git add Adders/FP/testbenches/TB_fpAdderEq.sv Adders/FP/Makefile && git commit -m "TB_fpAdderEq: fpAdder at (8, 23) matches fp32Adder, D-1 overflow flags counted"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 6: fpAdder DV in bf16 (SoftFloat, Vivado vectors, model, exhaustive)

**Files:**
- Create: `Adders/FP/testbenches/TB_fpAdder.sv`, `Adders/FP/testbenches/TB_fpAdderVIVADO.sv`, `Adders/FP/testbenches/vectors_bf16.mem` (generated)

**Interfaces:**
- Consumes: `c_fp_add` (Task 4), `fpAdder` (Task 5), `check_fpu.py --unit add`.

- [ ] **Step 1: Write the SoftFloat testbench**

`Adders/FP/testbenches/TB_fpAdder.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpAdder against Berkeley SoftFloat through DPI with TB_fp32Adder's rules: 3 ulp, any NaN for a NaN, flags checked,
// a flush accepted where the reference is tiny. Shared stimulus by default; +A_LO=+A_HI= runs every b for each a in [A_LO, A_HI).
// +DUMP=<file> writes every result for check_fpu.py.
module TB_fpAdder #(
    parameter int EXP_W  = 8,
    parameter int MAN_W  = 7,
    parameter int RANDOM = 200000
);
  localparam int W = 1 + EXP_W + MAN_W;
  localparam int TOL = 3;
  import "DPI-C" function int c_fp_add(input int a, input int b, input int man_w, output int flags);

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0;
  logic [W-1:0] va[$], vb[$];
  `include "fp_stim.svh"

  logic [W-1:0] res;
  logic done, ov, un, inv;
  fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(res), .done_o(done), .overflow_o(ov), .underflow_o(un), .invalid_o(inv));

  typedef struct {
    logic [W-1:0] a, b, res;
    bit ov, un, inv;
  } tx_t;
  tx_t q[$];
  longint checked = 0, errs = 0;
  int fd = 0, cyc = 0, t_issue = -1, lat = -1;
  string dump;

  function automatic bit is_nan(input logic [W-1:0] x);
    return (x[W-2:MAN_W] == '1) && (x[MAN_W-1:0] != '0);
  endfunction

  task automatic drive(input logic [W-1:0] x, input logic [W-1:0] y);
    tx_t t;
    int fl;
    t.a = x;
    t.b = y;
    t.res = W'(c_fp_add(int'(x), int'(y), MAN_W, fl));
    t.un = fl[1];
    t.ov = fl[2];
    t.inv = fl[4];
    q.push_back(t);
    #1 valid = 1;
    a = x;
    b = y;
    @(posedge clk);
  endtask

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      bit rm, fm;
      int diff;
      if (lat < 0) lat = cyc - t_issue - 1;  // done_o is sampled one edge after it rises
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        diff = int'(res) - int'(t.res);
        if (diff < 0) diff = -diff;
        rm = (res == t.res) || (is_nan(t.res) && is_nan(res)) || (!is_nan(t.res) && diff <= TOL) ||
             (un && res[W-2:0] == '0 && t.res[W-2:MAN_W] == '0);
        fm = (ov == t.ov) && (inv == t.inv) && ((un == t.un) || (un && res[W-2:0] == '0));
        if (!rm || !fm) begin
          errs++;
          if (errs <= 20)
            $display("[FAIL] %h + %h: got %h (ov %b un %b inv %b), SoftFloat %h (ov %b un %b inv %b)",
                     t.a, t.b, res, ov, un, inv, t.res, t.ov, t.un, t.inv);
        end
        if (fd != 0) $fwrite(fd, "%h %h %h %b%b%b\n", t.a, t.b, res, ov, un, inv);
      end
    end
  end

  initial begin
    int lo, hi;
    if (EXP_W != 8) $fatal(1, "TB_fpAdder: the SoftFloat reference needs an 8-bit exponent");
    if ($value$plusargs("DUMP=%s", dump)) fd = $fopen(dump, "w");
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("A_LO=%d", lo) && $value$plusargs("A_HI=%d", hi)) begin
      for (int x = lo; x < hi; x++)
        for (int y = 0; y < (1 << W); y++) drive(W'(x), W'(y));
    end else begin
      build_stimulus(RANDOM);
      foreach (va[i]) drive(va[i], vb[i]);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (fd != 0) $fclose(fd);
    if (q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d results never came out", q.size());
    end
    if (lat != sienna_fmt_pkg::add_lat(EXP_W, MAN_W)) begin
      errs++;
      $display("[FAIL] latency %0d, sienna_fmt_pkg::add_lat says %0d", lat, sienna_fmt_pkg::add_lat(EXP_W, MAN_W));
    end
    $display("fpAdder EXP_W=%0d MAN_W=%0d: %0d sums against SoftFloat, %0d errors, latency %0d", EXP_W, MAN_W,
             checked, errs, lat);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 2: Write the Vivado testbench**

`Adders/FP/testbenches/TB_fpAdderVIVADO.sv`:

```systemverilog
`timescale 1ns / 100ps

// fpAdder against a gen_vectors.cpp file (A B Res Flags in hex) with TB_fpAdder's rules; no DPI, so it runs in Vivado.
module TB_fpAdderVIVADO #(
    parameter int    EXP_W       = 8,
    parameter int    MAN_W       = 7,
    parameter int    NUM_VECTORS = 10000,
    parameter string VEC_FILE    = "vectors_bf16.mem"
);
  localparam int W = 1 + EXP_W + MAN_W;
  localparam int TOL = 3;
  logic [3*W+7:0] vec[NUM_VECTORS];
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [W-1:0] a = '0, b = '0, res;
  logic done, ov, un, inv;
  fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b),
      .result_o(res), .done_o(done), .overflow_o(ov), .underflow_o(un), .invalid_o(inv));

  int errs = 0, checked = 0;
  logic [3*W+7:0] q[$];

  function automatic bit is_nan(input logic [W-1:0] x);
    return (x[W-2:MAN_W] == '1) && (x[MAN_W-1:0] != '0);
  endfunction

  always @(posedge clk)
    if (rstn && done) begin
      logic [W-1:0] ea, eb, er;
      logic [7:0] fl;
      bit rm, fm;
      int diff;
      {ea, eb, er, fl} = q.pop_front();
      checked++;
      diff = int'(res) - int'(er);
      if (diff < 0) diff = -diff;
      rm = (res == er) || (is_nan(er) && is_nan(res)) || (!is_nan(er) && diff <= TOL) ||
           (un && res[W-2:0] == '0 && er[W-2:MAN_W] == '0);
      fm = (ov == fl[2]) && (inv == fl[4]) && ((un == fl[1]) || (un && res[W-2:0] == '0));
      if (!rm || !fm) begin
        errs++;
        if (errs <= 20) $display("[FAIL] %h + %h: got %h (ov %b un %b inv %b), expected %h flags %h", ea, eb, res, ov, un, inv, er, fl);
      end
    end

  initial begin
    $readmemh(VEC_FILE, vec);
    if (^vec[0] === 1'bx) $fatal(1, "TB_fpAdderVIVADO: could not read %s", VEC_FILE);
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    for (int i = 0; i < NUM_VECTORS; i++) begin
      #1 valid = 1;
      {a, b} = vec[i][3*W+7:W+8];
      q.push_back(vec[i]);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $display("fpAdder vectors %s: %0d checked, %0d errors", VEC_FILE, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NUM_VECTORS) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 3: Run SoftFloat, model, vectors, Vivado and exhaustive**

```bash
$J/snap_launch.sh u6_sf 8 1 $J/cmd_aril.sh Adders/FP verilator
$J/snap_launch.sh u6_chk 8 1 $J/cmd_aril.sh Adders/FP check
$J/snap_launch.sh u6_vec 4 1 bash -c 'SystolicMesh/ArithmeticLibrary/Common/testbenches/generate_vectors.sh && mkdir -p testbenches/results/uniform && cp SystolicMesh/ArithmeticLibrary/Adders/FP/testbenches/vectors_bf16.mem testbenches/results/uniform/fpAdder_vectors_bf16.mem'
cp /proj/work/spramanik/sienna_jobs/runs/u6_vec/results/uniform/fpAdder_vectors_bf16.mem SystolicMesh/ArithmeticLibrary/Adders/FP/testbenches/vectors_bf16.mem
$J/snap_launch.sh u6_viv 4 1 $J/cmd_aril.sh Adders/FP vivado_tb
for s in $(seq 0 15); do $J/snap_launch.sh u6_exh_$s 8 12 $J/cmd_aril_exh.sh Adders/FP $((s*4096)) $(((s+1)*4096)); done
```

Expected: 0 errors against SoftFloat with latency 5, 0 mismatches against fpu.py, 10000 vectors with 0 errors, and each exhaustive slice with 0 errors. A result beyond 3 ulp in a subtraction whose exponents differ by 2 or more is the truncating alignment, the same algorithm as fp32Adder. Before calling it a bug, compute the fp32Adder-style result by hand for that vector (the Python model gives it). If the model and RTL agree and SoftFloat differs by more than 3 ulp, report it to Soham as a property of the algorithm, with the vector. Do not widen the tolerance.

- [ ] **Step 4: Commit (AriL)**

```bash
git add Adders/FP/testbenches/TB_fpAdder.sv && git commit -m "TB_fpAdder: bf16 against SoftFloat, shared stimulus or exhaustive, dump for fpu.py"
git add Adders/FP/testbenches/TB_fpAdderVIVADO.sv Adders/FP/testbenches/vectors_bf16.mem && git commit -m "TB_fpAdderVIVADO and its bf16 vectors"
git push git@github.com:SoHam-56/ArithmeticLibrary.git bf16
```

### Task 7: Gate G1, AriL

**Files:**
- Create: `testbenches/results/uniform/aril_gate.log` (SIENNA; generated report)

- [ ] **Step 1: Run every AriL check, plus random power-up and lint, from one snapshot**

`/proj/work/spramanik/sienna_jobs/cmd_aril_gate.sh` (`chmod +x`):

```bash
#!/bin/bash
# Gate G1: every AriL check on one snapshot, with random power-up and lint; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/uniform; A=SystolicMesh/ArithmeticLibrary
RI="-DNO_ZERO_INIT --x-initial unique --x-assign unique"
mkdir -p $R
fail() { echo "GATE-FAIL $*"; }
$J/cmd_aril.sh Common pkg || fail pkg
$J/cmd_aril.sh Common dump32 || fail dump32
for u in Multipliers/FP Adders/FP; do
  for t in eq check vivado_tb lint; do $J/cmd_aril.sh $u $t || fail $u $t; done
  L=$R/$(echo $u | tr / _)_randinit.log
  (cd $A/$u && make eq EXTRA_FLAGS="$RI" PLUSARGS=+verilator+rand+reset+2 && make verilator EXTRA_FLAGS="$RI" PLUSARGS=+verilator+rand+reset+2) > $L 2>&1 || fail randinit $u
  grep -q "RESULT: FAILED" $L && fail randinit $u
done
$J/cmd_aril.sh Multipliers/FP32 verilator || fail fp32Multiplier
$J/cmd_aril.sh Adders/FP32 verilator || fail fp32Adder
(cd $A/Multipliers/FPWiden && $J/cmd_fpmul.sh) > $R/fpMulWiden.log 2>&1 || fail fpMulWiden
grep -h "RESULT:" $R/*.log | sort | uniq -c
```

Launch: `$J/snap_launch.sh g1 16 4 $J/cmd_aril_gate.sh`.

Expected:
- no `GATE-FAIL` line in `runs/g1/stdout.log`, and every `RESULT:` line reads `PASSED`;
- the fp32 unit TBs print `SUCCESS`, as in the Task 0 baseline;
- fpMulWiden prints 0 mismatches for bf16 and fp16;
- lint output shows no `LATCH`, `MULTIDRIVEN` or `UNOPTFLAT` warnings.

- [ ] **Step 2: Write the gate report**

Write `testbenches/results/uniform/aril_gate.log` by hand from the job outputs, with these sections:

- the commit hashes (AriL);
- per unit: equivalence counts (and the D-1 count for the adder), SoftFloat counts and latency, model mismatches, Vivado vectors, exhaustive slices (count, errors, wall time), random-init result, lint warning counts by type;
- fp32 units and fpMulWiden unchanged;
- what was not run.

Every number is copied from a log, with the log's path next to it.

- [ ] **Step 3: Report to Soham and wait**

Give the summary and the report's path. Level 2 starts only after Soham has seen G1.

---

## Level 2: GPNAE (gate G2 in Task 14)

All paths are relative to `/proj/work/spramanik/SIENNA/GPNAE` on branch `bf16` unless stated.

### Task 8: GPNAE on the new AriL, fp32 unchanged

**Files:**
- Modify: `ArithmeticLibrary` (submodule pointer to the AriL `bf16` tip from Task 7)
- Modify: `Makefile` (file list, `EXTRA_FLAGS`/`SIM_ARGS` hooks, `lint_fmt` target)

**Interfaces:**
- Produces: `make lint_fmt TOP=<module> FMT="-GEXP_W=.. -GMAN_W=.."`, an elaboration-only build of one block in one format. `make verilator EXTRA_FLAGS=... SIM_ARGS=...` works as in SIENNA's Makefile.

- [ ] **Step 1: Bump the submodule**

```bash
cd ArithmeticLibrary && git fetch origin bf16 && git checkout <AriL bf16 tip from Task 7> && cd ..
git -C ArithmeticLibrary log --oneline -1   # must print the Task 7 tip
```

- [ ] **Step 2: Add the new files and hooks to the Makefile**

In `DESIGN_FILES`, add these first, before `TYTAN/Memory/CoeffROM.v`, because the package must be read before its users:

```make
	../ArithmeticLibrary/Common/src/sienna_fmt_pkg.sv \
```

Add these after `../ArithmeticLibrary/Multipliers/FP32/src/fp32Multiplier.sv \`:

```make
	../ArithmeticLibrary/Multipliers/FP/src/fpMultiplier.sv \
	../ArithmeticLibrary/Adders/FP/src/fpAdder.sv \
```

After the `VERILATOR_FLAGS` block, add:

```make
# Hooks as in SIENNA: one-off defines (EXTRA_FLAGS) and simulator arguments (SIM_ARGS).
VERILATOR_FLAGS += $(EXTRA_FLAGS)
```

In the `verilator` target, change `$(VERILATOR_DIR)/./$(TOP_MODULE)_sim` to `$(VERILATOR_DIR)/./$(TOP_MODULE)_sim $(SIM_ARGS)`. Add the target below (and to `.PHONY` if the file has one):

```make
# Elaboration of one block in one format, e.g. make lint_fmt TOP=barrel_mac FMT="-GEXP_W=8 -GMAN_W=7".
lint_fmt:
	$(VERILATOR) --lint-only -Wno-fatal -DSYNTHESIS --top-module $(TOP) $(FMT) -I$(SRC_DIR) -I$(SRC_DIR)/TYTAN/Memory \
		$(addprefix $(SRC_DIR)/,$(DESIGN_FILES))
```

- [ ] **Step 3: Run the fp32 regressions and compare with the Task 0 baseline**

```bash
$J/snap_launch.sh g8_poly 16 2 $J/cmd_gpnae_reg.sh --lane poly --format fp32
$J/snap_launch.sh g8_taylor 16 2 $J/cmd_gpnae_reg.sh --lane gpnae --format fp32
```

Expected: for every pattern, the `RESULT`, `ERRSTAT` and `CYCLES` lines in `runs/g8_*/results/uniform/gpnae/<pattern>.log` equal the Task 0 baseline's. Check with `diff <(grep -hE "RESULT|ERRSTAT|CYCLES" runs/u0_gpnae32*/results/uniform/gpnae/*.log) <(grep -hE ... runs/g8_*/...)`, one lane at a time. Any difference comes from the submodule move (a8d684b to the Task 7 tip, which includes ea2c09b's port-style change to the fp32 units). Report it; do not go on.

- [ ] **Step 4: Commit (GPNAE) and push**

```bash
git add ArithmeticLibrary && git commit -m "Bump ArithmeticLibrary: format package, fpMultiplier, fpAdder"
git add Makefile && git commit -m "Makefile: format package and narrow units, EXTRA_FLAGS/SIM_ARGS hooks, lint_fmt"
git push origin bf16
```

### Task 9: `barrel_mac` in the build's format

**Files:**
- Modify: `src/TYTAN/barrel_mac.sv`

**Interfaces:**
- Produces: `barrel_mac #(int EXP_W = 8, int MAN_W = 23, int DATA_WIDTH = 1 + EXP_W + MAN_W, ADDR_LINES, K, INIT_FILE)`. The ports are unchanged; the default is fp32, so `gpnae.sv` is unaffected.

- [ ] **Step 1: Write the failing elaboration checks**

```bash
$J/snap_launch.sh g9_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=barrel_mac FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee lint_bad.txt; grep -q "barrel_mac: unsupported format" lint_bad.txt && echo REJECTED || echo NOT-REJECTED'
```

Expected now: `NOT-REJECTED`, since the parameter does not exist yet.

- [ ] **Step 2: Make it format-generic**

Replace the parameter list's first line and the two latency localparams, and wrap the unit instances:

```systemverilog
module barrel_mac #(
    parameter int EXP_W      = 8,
    parameter int MAN_W      = 23,
    parameter int DATA_WIDTH = 1 + EXP_W + MAN_W,
    parameter int ADDR_LINES = 5,
    parameter int K          = 16,
    parameter     INIT_FILE  = "taylor_coeffs.mem"
) (
```

```systemverilog
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i -> done_o of the format's multiplier
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // and adder
```

Replace the `fp32Multiplier MUL (...)` and `fp32Adder ADD (...)` instances with:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("barrel_mac: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid),
        .A(mul_a), .B(mul_b), .result_o(mul_res), .done_o(mul_done),
        .overflow_o(), .underflow_o(), .invalid_o()
    );
    fp32Adder ADD (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(add_valid),
        .A(add_a), .B(add_b), .result_o(add_res), .done_o(add_done),
        .overflow_o(), .underflow_o(), .invalid_o()
    );
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid),
        .A(mul_a), .B(mul_b), .result_o(mul_res), .done_o(mul_done),
        .overflow_o(), .underflow_o(), .invalid_o()
    );
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) ADD (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(add_valid),
        .A(add_a), .B(add_b), .result_o(add_res), .done_o(add_done),
        .overflow_o(), .underflow_o(), .invalid_o()
    );
  end
```

Update the header comment's unit names: `// The format's multiplier and adder accept a new operation every cycle (II=1) ...`. In the loop comment, replace `(8 multiply + 5 add)` with `(MUL_LAT + ADD_LAT: 13 in fp32, 8 in bf16)`. `K >= MIN_PER` still holds for K = 16 in both formats.

- [ ] **Step 3: Run the checks**

```bash
$J/snap_launch.sh g9_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=barrel_mac FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee lint_bad.txt; grep -q "barrel_mac: unsupported format" lint_bad.txt && echo REJECTED || echo NOT-REJECTED'
$J/snap_launch.sh g9_bf16 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=barrel_mac FMT="-GEXP_W=8 -GMAN_W=7"'   # expect no %Error
$J/snap_launch.sh g9_poly 16 2 $J/cmd_gpnae_reg.sh --lane poly --format fp32
$J/snap_launch.sh g9_taylor 16 2 $J/cmd_gpnae_reg.sh --lane gpnae --format fp32
```

Expected: `REJECTED`, then a clean bf16 elaboration, then fp32 `RESULT`/`ERRSTAT`/`CYCLES` lines identical to Task 8's for both lanes. If Verilator does not act on `$error` inside a generate block (the build prints no message), replace the block with `initial $fatal(1, "barrel_mac: unsupported format ...");` under `ifndef SYNTHESIS`, and note in the commit that elaboration-time `$error` was not honoured.

- [ ] **Step 4: Commit (GPNAE)**

```bash
git add src/TYTAN/barrel_mac.sv && git commit -m "barrel_mac: units and latencies from the build's format; unsupported formats fail elaboration"
git push origin bf16
```

### Task 10: `gpnae_tail` in the build's format

**Files:**
- Modify: `src/gpnae_tail.sv`

**Interfaces:**
- Produces: `gpnae_tail #(int EXP_W = 8, int MAN_W = 23, int CONTEXTS = 4, int IW = 4)`, with `x_i` and `result_o` now `[1+EXP_W+MAN_W-1:0]`. The other ports are unchanged.

- [ ] **Step 1: Write the failing elaboration check**

```bash
$J/snap_launch.sh g10_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=gpnae_tail FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee lint_bad.txt; grep -q "gpnae_tail: unsupported format" lint_bad.txt && echo REJECTED || echo NOT-REJECTED'
```

Expected now: `NOT-REJECTED`.

- [ ] **Step 2: Make it format-generic**

Parameters and constants:

```systemverilog
module gpnae_tail #(
    parameter int EXP_W    = 8,
    parameter int MAN_W    = 23,
    parameter int CONTEXTS = 4,
    parameter int IW       = 4   // width of the caller's element index
) (
    input logic clk_i,
    input logic rstn_i,

    input  logic                     start_i,  // taken on a cycle ready_o is high
    input  logic [EXP_W+MAN_W:0]     x_i,
    input  logic [              1:0] func_i,   // 01 SELU, 10 sigmoid, 11 tanh
    input  logic [         IW-1:0]   idx_i,    // returned with the result
    output logic                     ready_o,  // a context is free

    output logic [EXP_W+MAN_W:0]     result_o,
    output logic [         IW-1:0]   idx_o,
    output logic                     done_o,
    output logic                     busy_o
);

  localparam int W = 1 + EXP_W + MAN_W;
  localparam int BIAS = (1 << (EXP_W - 1)) - 1;
  localparam int C = CONTEXTS;
  localparam int CW = (C > 1) ? $clog2(C) : 1;

  function automatic logic [W-1:0] K_(input logic [31:0] fp32);  // an fp32 constant in this format
    return W'(sienna_fmt_pkg::from_fp32(fp32, MAN_W));
  endfunction

  localparam logic [W-1:0] ONE = K_(32'h3F800000);
  localparam logic [W-1:0] TWO = K_(32'h40000000);
  localparam logic [W-1:0] LA = K_(32'h3FE10966);  // lambda * alpha
  localparam logic [W-2:0] BIG = K_(32'h42D00000)[W-2:0];  // |a| > 104: e^a is below every normal value
  localparam logic [W-1:0] SIGN = {1'b1, {(W - 1) {1'b0}}};
```

If Verilator rejects a function call in a localparam that is part-selected (`K_(...)[W-2:0]`), assign it through a full-width localparam first: `localparam logic [W-1:0] BIG_W = K_(32'h42D00000);` and `localparam logic [W-2:0] BIG = BIG_W[W-2:0];`.

`cinv`: `logic [W-1:0] cinv[11];` and each `assign cinv[k] = K_(32'h...);` with the existing fp32 values.

Registers: `logic [W-1:0] x[C], z[C], acc[C], d[C], e[C], s[C], tmp[C], res[C];` and `logic [W-1:0] ma[C], mb[C], aa[C], ab[C];`, plus `mul_a, mul_b, mul_res, add_a, add_b, add_res` as `[W-1:0]`.

`neg`: `function automatic logic [W-1:0] neg(input logic [W-1:0] v); return {~v[W-1], v[W-2:0]}; endfunction`.

Field replacements (every occurrence):

| fp32 code | generic |
|---|---|
| `e[c][30:23] < 8'd64` | `e[c][W-2:MAN_W] < EXP_W'((BIAS + 1) / 2)` |
| `(s[c][30:0] == '0) ? 32'h80000000 : {1'b1, s[c][30:23] + 8'd1, s[c][22:0]}` | `(s[c][W-2:0] == '0) ? SIGN : {1'b1, s[c][W-2:MAN_W] + EXP_W'(1), s[c][MAN_W-1:0]}` |
| `automatic logic [31:0] a;` | `automatic logic [W-1:0] a;` |
| `{1'b1, x_i[30:0]}` | `{1'b1, x_i[W-2:0]}` |
| `(x_i[30:23] >= 8'd133) ? {1'b1, BIG + 31'd1} : {1'b1, x_i[30:23] + 8'd1, x_i[22:0]}` | `(x_i[W-2:MAN_W] >= EXP_W'(BIAS + 6)) ? {1'b1, BIG + (W-1)'(1)} : {1'b1, x_i[W-2:MAN_W] + EXP_W'(1), x_i[MAN_W-1:0]}` |
| `a[30:0] > BIG` | `a[W-2:0] > BIG` |
| `d[c] <= {1'b1, ONE[30:0]};` | `d[c] <= {1'b1, ONE[W-2:0]};` |
| `a[30:23] < 8'd127` | `a[W-2:MAN_W] < EXP_W'(BIAS)` |
| `{a[31], 8'd126, a[22:0]}` | `{a[W-1], EXP_W'(BIAS - 1), a[MAN_W-1:0]}` |
| `4'(a[30:23] - 8'd126)` | `4'(a[W-2:MAN_W] - EXP_W'(BIAS - 1))` |
| `e[c][30:23] < 8'd64` (T_SQ) | `e[c][W-2:MAN_W] < EXP_W'((BIAS + 1) / 2)` |
| `{x[c][31], add_res[30:0]}` | `{x[c][W-1], add_res[W-2:0]}` |

Units: replace `fp32Multiplier TMUL (...)` and `fp32Adder TADD (...)` with:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("gpnae_tail: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier TMUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid), .A(mul_a), .B(mul_b), .result_o(mul_res),
                         .done_o(mul_done), .overflow_o(), .underflow_o(), .invalid_o());
    fp32Adder TADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(add_valid), .A(add_a), .B(add_b), .result_o(add_res),
                    .done_o(add_done), .overflow_o(), .underflow_o(), .invalid_o());
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) TMUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid), .A(mul_a),
        .B(mul_b), .result_o(mul_res), .done_o(mul_done), .overflow_o(), .underflow_o(), .invalid_o());
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) TADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(add_valid), .A(add_a),
        .B(add_b), .result_o(add_res), .done_o(add_done), .overflow_o(), .underflow_o(), .invalid_o());
  end
```

Afterwards, `grep -n "30:\|22:\|31\]\|32'h\|8'd" src/gpnae_tail.sv` may show only the fp32 constants inside `K_(...)` calls.

- [ ] **Step 3: Run the checks**

```bash
$J/snap_launch.sh g10_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=gpnae_tail FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee lint_bad.txt; grep -q "gpnae_tail: unsupported format" lint_bad.txt && echo REJECTED || echo NOT-REJECTED'
$J/snap_launch.sh g10_bf16 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=gpnae_tail FMT="-GEXP_W=8 -GMAN_W=7"'
$J/snap_launch.sh g10_poly 16 2 $J/cmd_gpnae_reg.sh --lane poly --format fp32
$J/snap_launch.sh g10_taylor 16 2 $J/cmd_gpnae_reg.sh --lane gpnae --format fp32
```

Expected: `REJECTED`, a bf16 elaboration with no `%Error`, and fp32 `RESULT`/`ERRSTAT`/`CYCLES` lines identical to Task 8's for both lanes. The Taylor lane does not use gpnae_tail; running it confirms nothing else moved.

- [ ] **Step 4: Commit (GPNAE)**

```bash
git add src/gpnae_tail.sv && git commit -m "gpnae_tail: fields, constants and units from the build's format"
git push origin bf16
```

### Task 11: Bit-exact lane model, `--model hw`, exact compare in TB_gpnae_poly

**Files:**
- Create: `gpnae_model.py`
- Modify: `gpnae_tests.py` (golden for `hw`, threshold pattern), `regression.py` (`--model hw`, `EXACT_MATCH`, `COEFF_FILE` in the header), `testbenches/TB_gpnae_poly.sv` (exact mode)

**Interfaces:**
- Consumes: `fpu.py` via `ArithmeticLibrary/Common/models`.
- Produces:
  - `gpnae_model.Lane(f: fpu.Fmt, rom: list[int])`, with `.element(x_bits: int, code: int) -> int`, `.run(x_bits: np.ndarray, code: int) -> np.ndarray` (element over an array, any shape), `.poly(x_bits: np.ndarray, code: int, rom=None) -> np.ndarray` (polynomial path only), `.tail(x_bits: int, code: int) -> int` and `.in_tail(x_bits, code) -> bool`. Codes are 1 SELU, 2 sigmoid, 3 tanh, 4 ReLU, 5 linear.
  - `gpnae_model.coeff_file(f) -> str` gives `"poly_coeffs.mem"` for fp32 and `"poly_coeffs_<name>.mem"` otherwise. `gpnae_model.read_rom(path) -> list[int]`.
  - `gpnae_model.golden(values: list[float], code: int, fmt) -> list[float]`, where `fmt` is a `number_formats.FloatFormat`; it goes through the bits and back.
  - Header items: `localparam bit EXACT_MATCH`, `localparam string COEFF_FILE`.
  - Test pattern `act_threshold`.

- [ ] **Step 1: Write the model**

`gpnae_model.py`:

```python
#!/usr/bin/env python3
"""Bit-exact model of gpnae_poly and gpnae_tail, op for op and operand for operand, in the lane's format; built on AriL's fpu.py.
fp32's negative sigmoid goes through fp32_down, modelled as fpAdder(P, -1); Task 11 measures whether that holds."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ArithmeticLibrary", "Common", "models"))
import fpu  # noqa: E402

SETS = {1: (0, 8), 2: (9, 6), 3: (16, 8)}  # control word: (ROM base, degree), gpnae_poly's BASE_* and DEG_*
LAMBDA, LA, ONE, TWO, NEG_ONE = 0x3F867D5F, 0x3FE10966, 0x3F800000, 0x40000000, 0xBF800000
FOUR, THREE_HALF, BIG = 0x40800000, 0x40600000, 0x42D00000
CINV = [0x3F800000, 0x3F000000, 0x3E2AAAAB, 0x3D2AAAAB, 0x3C088889, 0x3AB60B61,
        0x39500D01, 0x37D00D01, 0x3638EF1D, 0x3493F27E, 0x32D7322B]  # 1/(k+1)!, gpnae_tail's cinv


def coeff_file(f) -> str:
    return "poly_coeffs.mem" if f.name == "fp32" else f"poly_coeffs_{f.name}.mem"


def read_rom(path: str) -> list:
    return [int(ln.strip(), 2) for ln in open(path) if ln.strip()]


class Lane:
    def __init__(self, f, rom):
        self.f, self.rom = f, rom
        k = lambda v: fpu.from_fp32(v, f.m)
        self.lam, self.la, self.one, self.two, self.neg_one = map(k, (LAMBDA, LA, ONE, TWO, NEG_ONE))
        self.four, self.three_half, self.big = (k(v) & ((1 << (f.w - 1)) - 1) for v in (FOUR, THREE_HALF, BIG))
        self.cinv = [k(v) for v in CINV]
        self.S = 1 << (f.w - 1)
        self.MAG = self.S - 1

    def _mul(self, a, b):
        return fpu.mul(self.f, a, b)[0]

    def _add(self, a, b):
        return fpu.add(self.f, a, b)[0]

    def _e(self, v):
        return (v >> self.f.m) & self.f.emax

    def in_tail(self, x, code) -> bool:
        mag = x & self.MAG
        if code == 1:
            return bool(x & self.S) and mag > self.four
        if code == 2:
            return mag > self.three_half
        if code == 3:
            return mag > self.four
        return False

    def poly(self, x, code, rom=None):
        """Polynomial path for an array of inputs: MAC operand, Horner in the barrel MAC, post stage."""
        rom = self.rom if rom is None else rom
        S, MAG = self.S, self.MAG
        x = np.asarray(x, dtype=np.int64)
        base, deg = SETS[code]
        t = x if code == 1 else (x & MAG) if code == 2 else self._mul(x, x)
        acc = np.zeros_like(x)
        for r in range(deg + 1):  # mul(A=operand, B=acc), then add(A=coefficient, B=product)
            acc = self._add(np.full_like(x, rom[base + deg - r]), self._mul(t, acc))
        pos = ((x & S) == 0) | ((x & MAG) == 0)
        if code == 2:
            return np.where(pos, acc, self._add(acc, np.full_like(x, self.neg_one)) ^ S)
        if code == 1:
            return self._mul(x, np.where(pos, self.lam, acc))
        return self._mul(x, acc)

    def tail(self, x, code) -> int:
        f, S, MAG, E = self.f, self.S, self.MAG, self._e
        mul = lambda a, b: int(self._mul(a, b))
        add = lambda a, b: int(self._add(a, b))
        neg = lambda v: v ^ S
        if code == 1:
            a = x
        elif code == 2:
            a = S | (x & MAG)
        else:
            a = (S | (self.big + 1)) if E(x) >= f.bias + 6 else S | (((E(x) + 1) & f.emax) << f.m) | (x & f.mmask)
        if (a & MAG) > self.big:  # e^a underflows: e^a - 1 = -1, e^a = 0
            d, e = S | self.one, 0
            if code == 1:
                return mul(self.la, d)
        else:
            if E(a) < f.bias:
                z, m = a, 0
            else:
                z, m = (a & S) | ((f.bias - 1) << f.m) | (a & f.mmask), E(a) - (f.bias - 1)
            acc = self.cinv[10]
            for k in range(9, -1, -1):
                acc = add(mul(acc, z), self.cinv[k])
            d = mul(z, acc)
            if code == 1:
                for _ in range(m):
                    d = mul(d, add(d, self.two))
                return mul(self.la, d)
            e = add(self.one, d)
            for _ in range(m):
                if E(e) < (f.bias + 1) // 2:
                    e = 0
                    break
                e = mul(e, e)
        s = mul(e, add(self.one, neg(mul(e, add(self.one, neg(e))))))
        if code == 2:
            return s if (x & S) else add(self.one, neg(s))
        two_s = S if (s & MAG) == 0 else S | (((E(s) + 1) & f.emax) << f.m) | (s & f.mmask)
        return (x & S) | (add(self.one, two_s) & MAG)

    def element(self, x, code) -> int:
        code = code if code in (1, 2, 4, 5) else 3  # gpnae_poly runs tanh for every other control word (is_tanh)
        if code == 4:
            return 0 if (x & self.S) else x  # ReLU: every negative, -0 too, gives +0
        if code == 5:
            return x
        if self.in_tail(x, code):
            return self.tail(x, code)
        return int(self.poly(np.array([x]), code)[0])

    def run(self, x, code):
        """element() over an array: the polynomial path vectorized, tail elements one at a time."""
        x = np.asarray(x, dtype=np.int64)
        code = code if code in (1, 2, 4, 5) else 3
        if code == 4:
            return np.where((x & self.S) != 0, 0, x)
        if code == 5:
            return x.copy()
        out = np.asarray(self.poly(x, code), dtype=np.int64).copy()
        for i in range(x.size):
            v = int(x.flat[i])
            if self.in_tail(v, code):
                out.flat[i] = self.tail(v, code)
        return out


def golden(values, code, fmt):
    """Hardware outputs for FloatFormat values, as floats of the same format."""
    f = fpu.FORMATS[fmt.name]
    lane = Lane(f, read_rom(os.path.join(ROOT, coeff_file(f))))
    out = lane.run(np.array([fmt.encode(v) for v in values], dtype=np.int64), code)
    return [fmt.decode(int(b)) for b in out]
```

Check the tail's last lines against `gpnae_tail.sv`: `u = 1 + (-e)` (T_U), `v = e * u` (T_V), `w = 1 + (-v)` (T_W), `s = e * w` (T_S), with operands in that order. The nested expression above is exactly that.

- [ ] **Step 2: Wire `hw` into golden and the regression**

`gpnae_tests.py`, at the top of `golden()`:

```python
    if model == "hw":
        import gpnae_model
        return gpnae_model.golden(stim, ACTIVATIONS[act]["code"], fmt)
```

Add the threshold pattern after `gen_act_edge`, and register it in `TESTS`:

```python
def gen_act_threshold(act, n, rng, fmt, rs):
    """The lane's range thresholds (+/-3.5, +/-4) exactly and their two neighbours each way, in the format."""
    vals = []
    for t in (3.5, 4.0):
        b = fmt.encode(t)
        for d in (-2, -1, 0, 1, 2):
            v = fmt.decode(b + d)
            vals += [v, -v]
    while len(vals) < n:
        vals.append(float(rs.uniform(-rng, rng)))
    return vals[:n]
```

```python
    dict(name="act_threshold",    description="Range thresholds and their neighbours",    gen_fn=gen_act_threshold),
```

`regression.py`:

- `write_config(..., exact: bool, coeff_file: str)` writes two more lines:

```python
        f.write(f"  localparam bit EXACT_MATCH      = {1 if exact else 0};\n")
        f.write(f'  localparam string COEFF_FILE    = "{coeff_file}";\n')
```

- `--model` gets `choices=["exact", "series", "hw"]`. After `fmt = get_format(...)`:

```python
    exact = args.model == "hw"
    if exact:
        if args.lane != "poly":
            print(err("[ERROR] --model hw models gpnae_poly only; use --lane poly"))
            sys.exit(1)
        rel_tol, args.abs_tol = 0.0, 0.0
    coeff = "poly_coeffs.mem" if fmt.name == "fp32" else f"poly_coeffs_{fmt.name}.mem"
```

  and pass `exact, coeff` to `write_config`.

- The `--per-batch` default stays 30. `act_threshold` needs 20 of them.

`testbenches/TB_gpnae_poly.sv`:

- do not pass `COEFF_FILE` to the DUT: the lane must pick its own table, which is what the check below tests. The DUT's `EXP_W`/`MAN_W` connections are added in Task 13, when gpnae_poly has those parameters;
- `within_tol` first line after the declarations: `if (EXACT_MATCH) begin rel_err = (exp_bits === act_bits) ? 0.0 : 1.0; return exp_bits === act_bits; end`;
- a ROM check, after the existing `initial` block:

```systemverilog
  // The lane must have loaded this format's coefficient table.
  initial begin
    logic [DATA_WIDTH-1:0] want[32];
    #1;
    $readmemb(COEFF_FILE, want);
    for (int i = 0; i < 32; i++)
      if (dut.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i] !== want[i])
        $fatal(1, "coefficient ROM[%0d] is %h, %s has %h", i, dut.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i], COEFF_FILE, want[i]);
  end
```

- in the failure print, raise `shown < 12` to `shown < (EXACT_MATCH ? 200 : 12)` (both places), so an exact run lists every mismatch.

- [ ] **Step 3: Validate the model against today's fp32 lane**

```bash
$J/snap_launch.sh g11_hw32 16 3 $J/cmd_gpnae_reg.sh --lane poly --format fp32 --model hw
$J/snap_launch.sh g11_hw32r 16 3 $J/cmd_gpnae_reg.sh --lane poly --format fp32 --model hw --range 8
```

Expected: SELU and tanh exact on every pattern (`failed 0`), including the tails (`--range 8`). Sigmoid is exact for every non-negative input. Any sigmoid mismatch must be on a negative input inside ±3.5, which means fp32_down's rounding is not fpAdder(P, −1)'s. If so, record the count and list as a finding in the gate report; it does not apply to bf16, which uses fpAdder. Any other mismatch is a model bug. Fix the model against the RTL, never the reverse.

- [ ] **Step 4: Commit (GPNAE)**

```bash
git add testbenches/TB_gpnae_poly.sv && git commit -m "TB_gpnae_poly: the lane's format, an exact-match mode, and a coefficient ROM check"
git add gpnae_model.py && git commit -m "gpnae_model: bit-exact model of gpnae_poly and gpnae_tail"
git add gpnae_tests.py regression.py && git commit -m "regression: --model hw (bit-exact, gpnae_poly), act_threshold pattern"
git push origin bf16
```

### Task 12: bf16 coefficient table

**Files:**
- Create: `fit_poly_coeffs.py`, `poly_coeffs_bf16.mem`, `src/TYTAN/Memory/poly_coeffs_bf16.mem`

**Interfaces:**
- Consumes: `gpnae_model.Lane.poly`, `fpu`.
- Produces: `python3 fit_poly_coeffs.py --format bf16 [--report FILE]` writes both `.mem` copies (32 lines of W-bit binary) and a `.log` with the error per activation, method and degree.

- [ ] **Step 1: Write the guard test first**

```bash
$J/snap_launch.sh g12_guard 2 1 bash -c 'cd GPNAE && python3 fit_poly_coeffs.py --format fp32; echo exit=$?'
```

Expected now: `can't open file ... fit_poly_coeffs.py`. After Step 2 it must print `refusing to write fp32 coefficients` and `exit=1`.

- [ ] **Step 2: Write the fit script**

`fit_poly_coeffs.py`:

```python
#!/usr/bin/env python3
"""Fits gpnae_poly's coefficient table for a narrow format and measures it as the hardware evaluates it (gpnae_model, bit-exact).
Fits as behind poly_coeffs.mem: Chebyshev on 4001 points, SELU (e^x - 1)/x on [-3.5, 0] degree 8, sigmoid on [0, 4] degree 6,
tanh(sqrt u)/sqrt u on [0, 16] degree 8; coefficients rounded to the format, or fixed one at a time lowest first with the rest refit.
Writes poly_coeffs_<fmt>.mem in the GPNAE root and in src/TYTAN/Memory. Never writes the fp32 files."""
import argparse
import os
import sys

import numpy as np

import gpnae_model
from gpnae_model import ROOT, SETS, fpu

LA = 1.7580993408473766  # lambda * alpha
DOMAIN = {1: (-3.5, 0.0), 2: (0.0, 4.0), 3: (0.0, 16.0)}
NAME = {1: "selu", 2: "sigmoid", 3: "tanh"}


def target(code, t):
    t = np.asarray(t, float)
    if code == 1:
        return np.where(np.abs(t) < 1e-8, LA * (1 + t / 2), LA * np.expm1(t) / np.where(t == 0, 1, t))
    if code == 2:
        return 1 / (1 + np.exp(-t))
    s = np.sqrt(np.maximum(t, 0))
    return np.where(t < 1e-12, 1 - t / 3, np.tanh(s) / np.where(s == 0, 1, s))


def exact_act(code, x):
    x = np.asarray(x, float)
    if code == 1:
        return np.where(x >= 0, 1.0507009873554805 * x, LA * np.expm1(x))
    if code == 2:
        return 1 / (1 + np.exp(-x))
    return np.tanh(x)


def to_fmt(f, v):
    """float -> format bits: fp32 nearest, then nearest at the format's width."""
    return fpu.from_fp32(int(np.float32(v).view(np.uint32)), f.m)


def to_float(f, bits):
    return np.asarray((np.asarray(bits, dtype=np.int64) << (23 - f.m)).astype(np.uint32)).view(np.float32).astype(np.float64)


def fit_plain(f, code, deg):
    lo, hi = DOMAIN[code]
    x = np.linspace(lo, hi, 4001)
    c = np.polynomial.chebyshev.Chebyshev.fit(x, target(code, x), deg).convert(kind=np.polynomial.Polynomial).coef
    return [to_fmt(f, v) for v in c]


def fit_greedy(f, code, deg):
    lo, hi = DOMAIN[code]
    x = np.linspace(lo, hi, 4001)
    y = target(code, x)
    w = 1 / np.maximum(np.abs(y), 1e-6)  # relative error
    fixed = []
    for j in range(deg + 1):
        r = y - sum(to_float(f, c) * x**i for i, c in enumerate(fixed))
        V = np.vstack([x**i for i in range(j, deg + 1)]).T
        sol = np.linalg.lstsq(V * w[:, None], r * w, rcond=None)[0]
        fixed.append(to_fmt(f, sol[0]))
    return fixed


def inputs(f, code):
    """Every format value the polynomial path sees: negative SELU inputs, sigmoid and tanh inputs within the thresholds."""
    b = np.arange(1 << f.w, dtype=np.int64)
    e = (b >> f.m) & f.emax
    lane = gpnae_model.Lane(f, [0] * 32)
    finite = (e != f.emax) & (e != 0)
    mag = b & lane.MAG
    keep = finite & (((code == 1) & (b >= lane.S) & (mag <= lane.four)) | ((code == 2) & (mag <= lane.three_half)) |
                     ((code == 3) & (mag <= lane.four)))
    return b[keep]


def measure(f, code, coeffs):
    base, deg = SETS[code]
    rom = [0] * 32
    rom[base:base + deg + 1] = coeffs
    x = inputs(f, code)
    hw = to_float(f, gpnae_model.Lane(f, rom).poly(x, code))
    ref = exact_act(code, to_float(f, x))
    rel = np.abs(hw - ref) / np.maximum(np.abs(ref), 1e-30)
    return float(rel.max()), float(rel.mean()), len(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", required=True, choices=sorted(fpu.FORMATS))
    p.add_argument("--report", default=os.path.join(ROOT, "testbenches", "results", "poly_coeffs_fit.log"))
    a = p.parse_args()
    if a.format == "fp32":
        print("refusing to write fp32 coefficients: poly_coeffs.mem is published and fixed")
        sys.exit(1)
    f = fpu.FORMATS[a.format]
    out = gpnae_model.coeff_file(f)
    assert out not in ("poly_coeffs.mem", "taylor_coeffs.mem")
    L, table = [], [0] * 32
    L.append(f"gpnae_poly coefficients for {a.format}: error of the bit-exact hardware path against the exact function")
    L.append(f"{'activation':<10}{'degree':>7}{'method':>8}{'worst rel':>12}{'mean rel':>12}{'inputs':>8}")
    for code in (1, 2, 3):
        base, deg = SETS[code]
        best = None
        for d in range(2, deg + 1):
            for name, fn in (("plain", fit_plain), ("greedy", fit_greedy)):
                c = fn(f, code, d)
                worst, mean, n = measure(f, code, c + [0] * (deg - d))
                L.append(f"{NAME[code]:<10}{d:>7}{name:>8}{worst:>12.4%}{mean:>12.4%}{n:>8}")
                if d == deg and (best is None or worst < best[0]):
                    best = (worst, name, c)
        table[base:base + deg + 1] = best[2]
        L.append(f"{NAME[code]:<10} chosen: degree {deg} (table layout unchanged), {best[1]}, worst {best[0]:.4%}")
    for path in (os.path.join(ROOT, out), os.path.join(ROOT, "src", "TYTAN", "Memory", out)):
        with open(path, "w") as fh:
            fh.write("".join(format(v, f"0{f.w}b") + "\n" for v in table))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
```

Lower-degree rows pad the table with zero coefficients at the top. This matches evaluating a degree-d polynomial on the degree-8 ROM layout, since the leading zero terms add nothing. They are printed for Soham's later decision (D-4).

- [ ] **Step 3: Run it on the farm, bring the tables back**

```bash
$J/snap_launch.sh g12_fit 8 2 bash -c 'cd GPNAE && python3 fit_poly_coeffs.py --format bf16 --report ../testbenches/results/uniform/poly_coeffs_fit.log && cp poly_coeffs_bf16.mem ../testbenches/results/uniform/'
```

Look at the target paths first; they must not exist yet. Then copy `runs/g12_fit/results/uniform/poly_coeffs_bf16.mem` to `GPNAE/poly_coeffs_bf16.mem` and to `GPNAE/src/TYTAN/Memory/poly_coeffs_bf16.mem`. Check that `git status` shows `poly_coeffs.mem` and `taylor_coeffs.mem` unchanged.

Expected: 32 lines of 16 bits each, and a report with a worst relative error per activation. Hold that number against the bf16 regression tolerance `suggested_rel_tol(bf16)` = 6.25%. If any chosen worst error is above it, stop and give Soham the table (D-4); do not tune further on your own.

- [ ] **Step 4: Commit (GPNAE)**

```bash
git add fit_poly_coeffs.py && git commit -m "fit_poly_coeffs: the poly lane's fits, measured bit-exact, for narrow formats only"
git add poly_coeffs_bf16.mem src/TYTAN/Memory/poly_coeffs_bf16.mem && git commit -m "poly_coeffs_bf16.mem: gpnae_poly's table for bf16"
git push origin bf16
```

### Task 13: `gpnae_poly` in the build's format

**Files:**
- Modify: `src/gpnae_poly.sv`

**Interfaces:**
- Produces: `gpnae_poly #(int EXP_W = 8, int MAN_W = 23, int DATA_WIDTH = 1 + EXP_W + MAN_W, ADDR_LINES, CONTROL_WIDTH, K, TAIL_CONTEXTS, string COEFF_FILE = is_fp32 ? "poly_coeffs.mem" : "poly_coeffs_bf16.mem")`. The ports are unchanged. SIENNA passes EXP_W and MAN_W from Task 20 on.

- [ ] **Step 1: Write the failing checks**

```bash
$J/snap_launch.sh g13_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=gpnae_poly FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee l.txt; grep -q "gpnae_poly: unsupported format" l.txt && echo REJECTED || echo NOT-REJECTED'
$J/snap_launch.sh g13_bf16 16 3 $J/cmd_gpnae_reg.sh --lane poly --format bf16 --model hw
```

Expected now: `NOT-REJECTED`, and the bf16 run fails, either with the ROM `$fatal` (it loads `poly_coeffs.mem`) or with mismatches.

- [ ] **Step 2: Make it format-generic**

Parameters:

```systemverilog
module gpnae_poly #(
    parameter int    EXP_W         = 8,
    parameter int    MAN_W         = 23,
    parameter int    DATA_WIDTH    = 1 + EXP_W + MAN_W,
    parameter int    ADDR_LINES    = 5,
    parameter int    CONTROL_WIDTH = 3,  // 001 SELU, 010 sigmoid, 011 tanh, 100 ReLU, 101 linear
    parameter int    K             = 16,
    parameter int    TAIL_CONTEXTS = 4,  // tail elements gpnae_tail works on at once
    parameter string COEFF_FILE    = sienna_fmt_pkg::is_fp32(EXP_W, MAN_W) ? "poly_coeffs.mem" : "poly_coeffs_bf16.mem"
) (
```

Localparams, replacing `MUL_LAT`, `DN_LAT` and `LAMDA`:

```systemverilog
  localparam bit FP32 = sienna_fmt_pkg::is_fp32(EXP_W, MAN_W);
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // the format's multiplier, valid_i at t, done_o at t+MUL_LAT
  localparam int DN_LAT = FP32 ? 6 : sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // fp32_down: done_o = valid_stage6; else fpAdder
  localparam logic [DATA_WIDTH-1:0] LAMDA = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h3F867D5F, MAN_W));
  localparam logic [DATA_WIDTH-1:0] NEG_ONE = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'hBF800000, MAN_W));
  localparam logic [DATA_WIDTH-1:0] T4_W = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h40800000, MAN_W));  // 4.0
  localparam logic [DATA_WIDTH-1:0] T35_W = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h40600000, MAN_W));  // 3.5
  localparam logic [DATA_WIDTH-2:0] T4 = T4_W[DATA_WIDTH-2:0];
  localparam logic [DATA_WIDTH-2:0] T35 = T35_W[DATA_WIDTH-2:0];

  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("gpnae_poly: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end
```

`sig_in_tail`:

```systemverilog
      3'b001:         sig_in_tail = fifo_data_o[DATA_WIDTH-1] && (fifo_data_o[DATA_WIDTH-2:0] > T4);  // SELU x < -4
      3'b010:         sig_in_tail = (fifo_data_o[DATA_WIDTH-2:0] > T35);  // sigmoid |x| > 3.5
      3'b100, 3'b101: sig_in_tail = 1'b0;  // ReLU and linear are exact
      default:        sig_in_tail = (fifo_data_o[DATA_WIDTH-2:0] > T4);  // tanh |x| > 4
```

`barrel_mac` instance: add `.EXP_W(EXP_W), .MAN_W(MAN_W)` and `.INIT_FILE(COEFF_FILE)`. `gpnae_tail` instance: add `.EXP_W(EXP_W), .MAN_W(MAN_W)`. In `testbenches/TB_gpnae_poly.sv`, the DUT gets `.EXP_W(EXP_BITS), .MAN_W(MAN_BITS)`.

Units: replace `fp32Multiplier SQ`, `fp32Multiplier POST` and `fp32_down POSTD` with:

```systemverilog
  if (FP32) begin : G_FP32
    fp32Multiplier SQ (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(sq_valid), .A(sq_a), .B(sq_a), .result_o(sq_res),
                       .done_o(sq_done), .overflow_o(), .underflow_o(), .invalid_o());
    fp32Multiplier POST (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid), .A(mul_a), .B(mul_b), .result_o(mul_res),
                         .done_o(mul_done), .overflow_o(), .underflow_o(), .invalid_o());
    fp32_down POSTD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(dn_valid), .A(dn_a), .Result(dn_res), .done_o(dn_done));
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) SQ (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(sq_valid), .A(sq_a), .B(sq_a),
        .result_o(sq_res), .done_o(sq_done), .overflow_o(), .underflow_o(), .invalid_o());
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) POST (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(mul_valid), .A(mul_a),
        .B(mul_b), .result_o(mul_res), .done_o(mul_done), .overflow_o(), .underflow_o(), .invalid_o());
    // P - 1 for negative sigmoid inputs: an adder with the constant -1, as fp32_down is in fp32.
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) POSTD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(dn_valid), .A(dn_a), .B(NEG_ONE),
        .result_o(dn_res), .done_o(dn_done), .overflow_o(), .underflow_o(), .invalid_o());
  end
```

The multiplier's and the adder's latency differ in bf16 (3 and 5), so `pm_v`/`pd_v` keep separate lengths, as today. `drain_cnt == MUL_LAT[3:0] + 2` still holds. No hierarchical reference to `SQ`, `POST` or `POSTD` exists in any testbench or script (checked at planning time with `grep -rn '\.SQ\b\|\.POST\b\|\.POSTD\b'`). Re-run that grep before committing.

- [ ] **Step 3: Run the checks**

```bash
$J/snap_launch.sh g13_bad 4 1 bash -c 'cd GPNAE && make lint_fmt TOP=gpnae_poly FMT="-GEXP_W=5 -GMAN_W=10" 2>&1 | tee l.txt; grep -q "gpnae_poly: unsupported format" l.txt && echo REJECTED || echo NOT-REJECTED'
$J/snap_launch.sh g13_bf16 16 3 $J/cmd_gpnae_reg.sh --lane poly --format bf16 --model hw
$J/snap_launch.sh g13_bf16r 16 3 $J/cmd_gpnae_reg.sh --lane poly --format bf16 --model hw --range 8
$J/snap_launch.sh g13_poly32 16 2 $J/cmd_gpnae_reg.sh --lane poly --format fp32
$J/snap_launch.sh g13_taylor32 16 2 $J/cmd_gpnae_reg.sh --lane gpnae --format fp32
```

Expected:
- `REJECTED`;
- bf16 `--model hw`: every pattern, including `act_threshold`, has `failed 0 missing 0` and `exact` equal to `total`, both in range and with `--range 8` (tails);
- fp32, both lanes: `RESULT`/`ERRSTAT`/`CYCLES` identical to Task 8.

A bf16 mismatch is a bug in the RTL or in the model. The unit models are proven against the RTL units (Tasks 4 and 6), so check the op sequence and operand order first.

- [ ] **Step 4: Commit (GPNAE)**

```bash
git add src/gpnae_poly.sv && git commit -m "gpnae_poly: units, constants, thresholds and coefficient table from the build's format"
git add testbenches/TB_gpnae_poly.sv && git commit -m "TB_gpnae_poly: builds the lane in the header's format"
git push origin bf16
```

### Task 14: Gate G2, GPNAE

- [ ] **Step 1: One job for everything at this level**

`/proj/work/spramanik/sienna_jobs/cmd_gpnae_gate.sh`:

```bash
#!/bin/bash
# Gate G2: GPNAE fp32 unchanged, bf16 bit-exact and accurate, random power-up, lint; run from a snapshot root.
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/uniform/g2; mkdir -p $R
run() { tag=$1; shift; $J/cmd_gpnae_reg.sh "$@" > $R/$tag.txt 2>&1 || echo "GATE-FAIL $tag"; cp GPNAE/testbenches/results/gpnae_report.log $R/$tag.report.log 2>/dev/null; }
run poly32 --lane poly --format fp32
run taylor32 --lane gpnae --format fp32
run bf16_hw --lane poly --format bf16 --model hw
run bf16_hw_tails --lane poly --format bf16 --model hw --range 8
run bf16_exact --lane poly --format bf16 --model exact
for s in 1 2 3; do run bf16_hw_seed$s --lane poly --format bf16 --model hw --seed $((s * 101)); done
export EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" SIM_ARGS="+verilator+rand+reset+2"
run bf16_hw_randinit --lane poly --format bf16 --model hw
run poly32_randinit --lane poly --format fp32
unset EXTRA_FLAGS SIM_ARGS
(cd GPNAE && for f in "-GEXP_W=8 -GMAN_W=7" "-GEXP_W=8 -GMAN_W=23"; do
   verilator --lint-only -Wall -DSYNTHESIS --top-module gpnae_poly $f -Isrc -Isrc/TYTAN/Memory $(make -s -p 2>/dev/null | sed -n 's/^DESIGN_FILES := //p' | tr ' ' '\n' | sed 's#^#src/#'); done) > $R/lint.txt 2>&1
grep -c "^%Warning" $R/lint.txt; grep -E "LATCH|MULTIDRIVEN|UNOPTFLAT|^%Error" $R/lint.txt | head
```

If the `make -p` extraction of `DESIGN_FILES` is awkward, list the files explicitly in the script from the Makefile. Launch with `$J/snap_launch.sh g2 32 8 $J/cmd_gpnae_gate.sh`.

Expected:
- no `GATE-FAIL`;
- fp32 lines identical to Task 8;
- every bf16 `hw` run fully exact, random-init included;
- `bf16_exact` passes within 6.25% (`suggested_rel_tol`). If it fails, that is an accuracy result, not a bug. Report it with the worst element per activation;
- lint: no `LATCH`, `MULTIDRIVEN`, `UNOPTFLAT` or `%Error`.

- [ ] **Step 2: Report**

Write `testbenches/results/uniform/gpnae_gate.log` (SIENNA) by hand from `runs/g2/`. It holds:
- the commits;
- the fp32 identity;
- bf16 exact counts per pattern and activation;
- worst and mean error against the exact functions per activation, in bf16 and in fp32 (from the report files);
- cycles per input, fp32 against bf16 (from the `CYCLES` lines);
- the fp32_down finding from Task 11 if any;
- lint counts;
- what was not run: the Taylor lane in bf16 is out of scope.

- [ ] **Step 3: Report to Soham and wait**

Level 3 starts after Soham has seen G2.

---

## Level 3: SystolicMesh (gate G3 in Task 17)

Paths are relative to `/proj/work/spramanik/SIENNA/SystolicMesh` on branch `bf16`.

### Task 15: One format through the mesh

**Files:**
- Modify: `ArithmeticLibrary` pointer (AriL `bf16` tip from Task 7), `Makefile` (`DESIGN_FILES`)
- Modify: `src/engine/ProcessingElement.sv`, `src/engine/AccumulationUnit.sv`, `src/top/SystolicArray.sv`, `src/top/SystolicMesh.sv`

**Interfaces:**
- Produces: `SystolicMesh #(... EXP_W = 8, MAN_W = 23, DATA_WIDTH = 1 + EXP_W + MAN_W ...)`. It has no `OP_EXP_W`, `OP_MAN_W` or `OP_W` parameters any more, and every data port is `DATA_WIDTH` wide. `SystolicArray`, `ProcessingElement` and `AccumulationUnit` take the same `EXP_W`/`MAN_W`. The default `U` is `min(K, add_lat + 1)`.

- [ ] **Step 1: Write the failing checks**

`/proj/work/spramanik/sienna_jobs/cmd_mesh_fmt.sh`:

```bash
#!/bin/bash
# Mesh checks by format; args: lint EXP MAN | reg N FMT COLLAPSE [extra regression args]; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
cd SystolicMesh || exit 1
F="ArithmeticLibrary/Common/src/sienna_fmt_pkg.sv ArithmeticLibrary/Multipliers/Radix4Booth/src/R4Booth.sv
   ArithmeticLibrary/Multipliers/Karatsuba/src/karatsubaUnsigned.sv ArithmeticLibrary/Multipliers/FP32/src/fp32Multiplier.sv
   ArithmeticLibrary/Multipliers/FP/src/fpMultiplier.sv ArithmeticLibrary/Adders/FP32/src/LZC.sv
   ArithmeticLibrary/Adders/FP32/src/fp32Adder.sv ArithmeticLibrary/Adders/FP/src/fpAdder.sv
   src/engine/ProcessingElement.sv src/engine/AccumulationUnit.sv src/mem/MeshOutputSram.sv src/top/SystolicArray.sv src/top/SystolicMesh.sv"
case $1 in
  lint) verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module SystolicMesh -GMATRIX_SIZE=16 -GTILE_SIZE=4 -GEXP_W=$2 -GMAN_W=$3 $F ;;
  reg)  N=$2; FMT=$3; CK=$4; shift 4
        [ $N -ge 32 ] && export EXTRA_FLAGS="$EXTRA_FLAGS --output-split 20000 --output-split-cfuncs 20000 --output-groups 64"
        python3 regression.py --matrix-size $N --format $FMT --collapse-k $CK "$@" ;;
esac
```

Run:

```bash
$J/snap_launch.sh m15_bad 4 1 bash -c "$J/cmd_mesh_fmt.sh lint 5 10 2>&1 | tee l.txt; grep -q 'unsupported format' l.txt && echo REJECTED || echo NOT-REJECTED"
```

Expected now: `NOT-REJECTED`. Until Step 3 the build fails for other reasons: `EXP_W` does not exist yet, and `fpMultiplier` is not in the file list.

- [ ] **Step 2: Bump AriL, update the file list**

```bash
cd ArithmeticLibrary && git fetch origin bf16 && git checkout <Task 7 tip> && cd ..
```

In `Makefile` `DESIGN_FILES`, add `../ArithmeticLibrary/Common/src/sienna_fmt_pkg.sv` as the first entry. Add `../ArithmeticLibrary/Multipliers/FP/src/fpMultiplier.sv` and `../ArithmeticLibrary/Adders/FP/src/fpAdder.sv`. Remove the `fpMulWiden.sv` line; the file stays in AriL.

- [ ] **Step 3: ProcessingElement**

Parameters and ports:

```systemverilog
module ProcessingElement #(
    parameter int EXP_W      = 8,   // the build's format: fp32 8/23, bf16 8/7
    parameter int MAN_W      = 23,
    parameter int DATA_WIDTH = 1 + EXP_W + MAN_W,  // operands, products and sums
    parameter int K          = 4,  // products per set
    parameter int BANKS      = 3,  // sets held at once: one accumulating, the older ones finishing or being read
    parameter int U          = (K < sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1) ? K : sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1,
    parameter int BW         = (BANKS > 1) ? $clog2(BANKS) : 1
) (
    input  logic                            clk_i,
    input  logic                            rstn_i,
    input  logic [          DATA_WIDTH-1:0] a_i,
    input  logic [          DATA_WIDTH-1:0] b_i,
    input  logic                            v_i,
    input  logic                            fresh_i,     // with v_i: this pass starts a set, its first U products add to 0
    input  logic                            more_i,      // with v_i: another pass of the same set follows this one
    output logic [          DATA_WIDTH-1:0] a_o,
    output logic [          DATA_WIDTH-1:0] b_o,
    output logic                            v_o,
    output logic                            fresh_o,
    output logic                            more_o,
    input  logic [                  BW-1:0] rd_bank_i,   // bank the reader looks at
    output logic [U-1:0][DATA_WIDTH-1:0]    partial_o,   // that bank's partial sums
    input  logic                            release_i,   // the reader is done with rd_bank_i
    output logic [               BANKS-1:0] final_o      // per bank: a finished set, all adds written back
);
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+ADD_LAT
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+MUL_LAT
```

Delete the `OP_EXP_W`, `OP_MAN_W`, `OP_W` and `OP_FP32` lines. Replace the `G_MUL32`/`G_MULW` generate with the multiplier half of the block below. Replace the `fp32Adder ADD (...)` instance with the adder half:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("ProcessingElement: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i), .A(a_i), .B(b_i), .result_o(prod), .done_o(prod_v),
                        .overflow_o(), .underflow_o(), .invalid_o());
    fp32Adder ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum), .done_o(sum_v),
                   .overflow_o(), .underflow_o(), .invalid_o());
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i), .A(a_i), .B(b_i),
        .result_o(prod), .done_o(prod_v), .overflow_o(), .underflow_o(), .invalid_o());
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod),
        .result_o(sum), .done_o(sum_v), .overflow_o(), .underflow_o(), .invalid_o());
  end
```

`add_a` must be declared before this block. Move its `logic`/`assign` above the generate. The header comment's second line stays. Remove the `// fp32 operands use the fp32 multiplier; narrower floats multiply exactly into fp32.` comment.

- [ ] **Step 4: AccumulationUnit**

Add `parameter int EXP_W = 8, parameter int MAN_W = 23` before `DATA_WIDTH`. Make `DATA_WIDTH` default to `1 + EXP_W + MAN_W`. Replace `localparam int ADD_LAT = 5;` with `localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // the format's adder`. In `NODE`'s `ADD` branch, replace `fp32Adder adder (...)` with:

```systemverilog
        if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
          fp32Adder adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]), .A(lvl_d[l][2*m]), .B(lvl_d[l][2*m+1]),
                           .result_o(lvl_d[l+1][m]), .done_o(done_bits[m]), .overflow_o(), .underflow_o(), .invalid_o());
        end else begin : G_FP
          fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]),
              .A(lvl_d[l][2*m]), .B(lvl_d[l][2*m+1]), .result_o(lvl_d[l+1][m]), .done_o(done_bits[m]),
              .overflow_o(), .underflow_o(), .invalid_o());
        end
```

Near the top, add the module-level format check:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("AccumulationUnit: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end
```

- [ ] **Step 5: SystolicArray and SystolicMesh**

In both, rename `OP_EXP_W`→`EXP_W` and `OP_MAN_W`→`MAN_W`. Delete the `OP_W` parameter, replace every `OP_W` with `DATA_WIDTH`, and set `DATA_WIDTH`'s default to `1 + EXP_W + MAN_W`. Fix the parameter comments: `// the build's format: fp32 8/23 by default, bf16 8/7` on `EXP_W`, and `// every word: operands, sums, bias, results` on `DATA_WIDTH`. In `SystolicArray`, `U`'s default becomes the ProcessingElement expression from Step 3, with `K`. In `SystolicMesh`:

```systemverilog
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);
  localparam int U = (AK < ADD_LAT + 1) ? AK : ADD_LAT + 1;  // partial sums per array pixel: the adder latency plus one
```

replaces `localparam int U = (AK < 6) ? AK : 6;`. Pass `.EXP_W(EXP_W), .MAN_W(MAN_W)` to every `SystolicArray` and `AccumulationUnit` instance, in place of the `OP_*` ones. Add, beside the existing `ifndef SYNTHESIS` parameter check:

```systemverilog
  initial if (DATA_WIDTH != 1 + EXP_W + MAN_W) $error("SystolicMesh: DATA_WIDTH %0d is not 1 + EXP_W + MAN_W", DATA_WIDTH);
```

Then `grep -rn "OP_\|fpMulWiden" src/` must print nothing.

- [ ] **Step 6: Run the checks**

```bash
$J/snap_launch.sh m15_bad 4 1 bash -c "$J/cmd_mesh_fmt.sh lint 5 10 2>&1 | tee l.txt; grep -q 'unsupported format' l.txt && echo REJECTED || echo NOT-REJECTED"
$J/snap_launch.sh m15_lint 4 1 bash -c "$J/cmd_mesh_fmt.sh lint 8 7; $J/cmd_mesh_fmt.sh lint 8 23"
$J/snap_launch.sh m15_fp32 32 4 bash -c "cd SystolicMesh && make regression TRACE=0"
```

Expected:
- `REJECTED`;
- both lints free of `%Error`, `LATCH`, `MULTIDRIVEN` and `UNOPTFLAT`;
- the fp32 mesh regression 68/68, with every per-test cycle count equal to Task 0's `u0_mesh32`.

The mesh TB still only builds fp32 here; Task 16 adds bf16.

- [ ] **Step 7: Commit (SystolicMesh) and push**

```bash
git add ArithmeticLibrary && git commit -m "Bump ArithmeticLibrary: format package, fpMultiplier, fpAdder"
git add src/engine/ProcessingElement.sv src/engine/AccumulationUnit.sv src/top/SystolicArray.sv src/top/SystolicMesh.sv \
  && git commit -m "One number format through the mesh: EXP_W/MAN_W pick the units, latencies from sienna_fmt_pkg"
git add Makefile && git commit -m "Makefile: format package and narrow units; fpMulWiden no longer used"
git push origin bf16
```

The RTL commit changes four files, because the parameter rename has to land in all of them at once to build.

### Task 16: Bit-exact mesh golden, bf16 in the mesh regression

**Files:**
- Create: `mesh_model.py`, `stim_format.py`
- Modify: `matmul_tests.py` (`_write_set`, one new generator), `conv_tests.py` (`_write_set`), `regression.py` (`--format`, `--collapse-k`, TB patching, width check, report name), `testbenches/TB_SystolicMesh.sv`

**Interfaces:**
- Consumes: `fpu.py` from `ArithmeticLibrary/Common/models`.
- Produces:
  - `mesh_model.matmul(f, passes, N, T, collapse_k=1, bias=None) -> np.ndarray[N, N] of int64 bits`. `passes` is a list of `(A_bits, B_bits)` N×N arrays, summed in order as one set: SIENNA's partial passes, then the final one. `bias` is a length-N row of bits, or None for +0.
  - `stim_format.configure(fmt: str, tile: int, collapse_k: int)`, `stim_format.FORMAT`, `stim_format.digits() -> int`, `stim_format.write_set(A, B, stim_dir, suffix) -> np.ndarray` (C as floats), `stim_format.check_widths(stim_dir) -> None` (raises on a word of the wrong width).
  - Mesh `regression.py --format fp32|bf16 --collapse-k 0|1`.

- [ ] **Step 1: Write the model**

`mesh_model.py`:

```python
#!/usr/bin/env python3
"""Bit-exact model of SystolicMesh's arithmetic in the build's format: each PE sums product n of a set into slot n mod U
(products counted across a set's passes, the first U adding to +0), then the reducer's pairwise tree adds the U partials of every
depth slice with the bias as its last input. Operand order as the RTL: mul(A=a, B=b), PE add(A=slot, B=product), tree add(A=left, B=right)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArithmeticLibrary", "Common", "models"))
import fpu  # noqa: E402

ADD_LAT = 5  # sienna_fmt_pkg::add_lat, every supported format


def matmul(f, passes, N, T, collapse_k=1, bias=None):
    AK = N if collapse_k else T  # depth of each array's product
    RP = 1 if collapse_k else N // T  # depth slices, one array each
    U = min(AK, ADD_LAT + 1)
    acc = np.zeros((RP, U, N, N), dtype=np.int64)
    g = 0
    for A, B in passes:
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        for kk in range(AK):
            u = g % U
            for rp in range(RP):
                k = rp * AK + kk
                a = np.broadcast_to(A[:, k][:, None], (N, N))
                b = np.broadcast_to(B[k, :][None, :], (N, N))
                acc[rp, u] = fpu.add(f, acc[rp, u], fpu.mul(f, a, b)[0])[0]
            g += 1
    bias_row = np.zeros(N, dtype=np.int64) if bias is None else np.asarray(bias, dtype=np.int64)
    level = [acc[rp, u] for rp in range(RP) for u in range(U)] + [np.broadcast_to(bias_row[None, :], (N, N))]
    while len(level) > 1:
        nxt = [fpu.add(f, level[2 * m], level[2 * m + 1])[0] for m in range(len(level) // 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # an odd entry out waits a level, as the RTL's PASS delay
        level = nxt
    return np.asarray(level[0], dtype=np.int64)
```

- [ ] **Step 2: Write the stimulus format module**

`stim_format.py`:

```python
#!/usr/bin/env python3
"""The mesh stimulus's number format. fp32 writes float32 words and a float64 reference exactly as before; bf16 writes bf16
words and the bit-exact expected result from mesh_model, for the TB's exact compare."""
import os
import struct

import numpy as np

import mesh_model
from mesh_model import fpu

FORMAT, TILE, COLLAPSE_K = "fp32", 4, 1


def configure(fmt: str, tile: int, collapse_k: int) -> None:
    global FORMAT, TILE, COLLAPSE_K
    FORMAT, TILE, COLLAPSE_K = fmt, tile, collapse_k


def digits() -> int:
    return (fpu.FORMATS[FORMAT].w + 3) // 4


def to_bits(x) -> np.ndarray:
    """float32 values, rounded to the format (fp32 nearest, then nearest at the format's width); subnormals flush to zero."""
    f = fpu.FORMATS[FORMAT]
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    b = np.vectorize(lambda v: fpu.from_fp32(int(v), f.m))(u) if f.m != 23 else u
    return np.where(((b >> f.m) & f.emax) == 0, b & (1 << (f.w - 1)), b).astype(np.int64)


def to_float(bits) -> np.ndarray:
    f = fpu.FORMATS[FORMAT]
    return (np.asarray(bits, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def _write_words(path, bits) -> None:
    d = digits()
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _f2h(v) -> str:
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(v)))


def write_set(A, B, stim_dir, suffix=""):
    """Write matrixA/B/C<suffix>.mem for one set; returns C as floats."""
    if FORMAT == "fp32":
        C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        for name, M in (("A", A), ("B", B), ("C", C)):
            with open(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), "w") as fh:
                for v in np.asarray(M).flatten():
                    fh.write(_f2h(v) + "\n")
        return C
    N = A.shape[0]
    assert A.shape == (N, N) and B.shape == (N, N), f"mesh sets are N x N; got {A.shape} and {B.shape}"
    Ab, Bb = to_bits(A), to_bits(B)
    Cb = mesh_model.matmul(fpu.FORMATS[FORMAT], [(Ab, Bb)], N, TILE, COLLAPSE_K)
    for name, M in (("A", Ab), ("B", Bb), ("C", Cb)):
        _write_words(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), M)
    return to_float(Cb)


def check_widths(stim_dir) -> None:
    """Every matrix word must be this format's width: a stale fp32 file in a bf16 run would be truncated silently."""
    d = digits()
    for fn in sorted(os.listdir(stim_dir)):
        if fn.startswith("matrix") and fn.endswith(".mem"):
            for i, ln in enumerate(open(os.path.join(stim_dir, fn))):
                if ln.strip() and len(ln.strip()) != d:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {d} hex digits ({FORMAT})")
```

The fp32 branch writes the same bytes as today's `write_mem(_f2h)` and `_ref_matmul`. Step 6 checks that with `cmp`.

- [ ] **Step 3: Route both generators through it, add a signed-zero set**

`matmul_tests.py`: `import stim_format`, and replace `_write_set`'s body:

```python
def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "") -> None:
    """Compute C = A @ B in the stimulus format and write all three .mem files for one set."""
    stim_format.write_set(A, B, stim_dir, suffix)
```

`conv_tests.py`: the same, returning `stim_format.write_set(A, B, stim_dir, suffix)`.

New generator in `matmul_tests.py`, registered in `MATMUL_TESTS` the way its neighbours are:

```python
def gen_mm_signed_zero(stim_dir: str, N: int) -> int:
    """Rows of A all +0 or all -0 against B of mixed signs: every product is a signed zero and the PE's first add is 0 + p."""
    sets = []
    for s in range(MATMUL_NUM_SETS):
        _seed(7100 + s)
        A = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        A[0::4, :] = np.float32(-0.0)
        A[1::4, :] = np.float32(0.0)
        B = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
        sets.append((A, B))
    return _write_all(sets, stim_dir)
```

- [ ] **Step 4: TB and regression options**

`testbenches/TB_SystolicMesh.sv`:

```systemverilog
  localparam int EXP_W = 8;  // patched by regression.py --format
  localparam int MAN_W = 23;
  localparam int COLLAPSE_K = 1;  // patched by regression.py --collapse-k
  localparam DATA_WIDTH = 1 + EXP_W + MAN_W;
```

These replace `localparam DATA_WIDTH = 32;`. Change `ENABLE_TOL` to `localparam logic ENABLE_TOL = (EXP_W == 8 && MAN_W == 23);  // fp32 tolerance; narrow formats compare bit for bit`. Add to the DUT: `.EXP_W(EXP_W), .MAN_W(MAN_W), .COLLAPSE_K(COLLAPSE_K)`. Wrap the checker self-test body in `if (ENABLE_TOL) begin ... end`, because its literals are fp32. In the pass/fail print, show hex only when `!ENABLE_TOL`: the `f32()` decode is fp32-only.

`regression.py`:
- `import stim_format`;
- two options: `--format` (choices `fp32`, `bf16`, default `fp32`) and `--collapse-k` (choices `0`, `1`, `type=int`, default `1`);
- `_patch_tb(tb_path, matrix_size, tile_size, num_test_sets, exp_w, man_w, collapse_k)`, with three more substitutions of the same form:

```python
    for name, val in (("EXP_W", exp_w), ("MAN_W", man_w), ("COLLAPSE_K", collapse_k)):
        patched = re.sub(rf"(localparam\s+int\s+{name}\s*=\s*)\d+", rf"\g<1>{val}", patched)
```

- before each tile's groups run: `stim_format.configure(args.format, T, args.collapse_k)`. Before every `make`: `stim_format.check_widths(STIM_DIR)`;
- `(exp_w, man_w) = (8, 23) if args.format == "fp32" else (8, 7)`;
- the report file name gets `_{args.format}_ck{args.collapse_k}` before `.log`.

- [ ] **Step 5: Run it; expect the bf16 run to pass bit for bit**

```bash
$J/snap_launch.sh m16_bf16 32 4 $J/cmd_mesh_fmt.sh reg 16 bf16 1
$J/snap_launch.sh m16_bf16ck0 32 4 $J/cmd_mesh_fmt.sh reg 16 bf16 0
$J/snap_launch.sh m16_fp32 32 4 $J/cmd_mesh_fmt.sh reg 16 fp32 1
```

Expected: bf16 passes every test at every tile size with collapse-k 1 and 0, all exact (no `PASS-TOL` lines), `gen_mm_signed_zero` included. fp32 passes 68/68 plus the new test at every tile, with cycles equal to Task 15's. In the bf16 run, compare the first mismatching set, if any, against `mesh_model` with a different slot rule before touching the RTL. The model is the claim under test there, and the units underneath it are already proven.

- [ ] **Step 6: fp32 stimulus is byte-identical to before**

Before launching, save the Task 15 generator: `git -C SystolicMesh show <Task 15 commit>:matmul_tests.py > .claude/scratch/matmul_tests_t15.py`. Then, in one job, run `gen_mm_random` from both copies into two directories and `cmp` every file:

```bash
$J/snap_launch.sh m16_same 4 1 bash -c 'cd SystolicMesh && mkdir -p /tmp/s_old /tmp/s_new && cp ../.claude/scratch/matmul_tests_t15.py mt_old.py &&
  python3 -c "import mt_old; mt_old.gen_mm_random(\"/tmp/s_old\", 16)" && python3 -c "import matmul_tests as m; m.gen_mm_random(\"/tmp/s_new\", 16)" &&
  for f in /tmp/s_old/*; do cmp $f /tmp/s_new/$(basename $f) || echo DIFF $f; done; echo checked'
```

Expected: `checked`, with no `DIFF` line.

- [ ] **Step 7: Commit (SystolicMesh)**

```bash
git add mesh_model.py && git commit -m "mesh_model: bit-exact model of the mesh's slot sums and reduce tree"
git add stim_format.py matmul_tests.py conv_tests.py && git commit -m "Mesh stimulus in the build's format; bf16 expected results bit-exact; signed-zero set"
git add testbenches/TB_SystolicMesh.sv regression.py && git commit -m "Mesh regression: --format and --collapse-k, exact compare for narrow formats"
git push origin bf16
```

### Task 17: Gate G3, SystolicMesh sweeps

- [ ] **Step 1: Launch the sweep**

N = 8, 16, 32, 64, each with every tile size (`pow2_tile_sizes(N)`), in fp32 and bf16, with collapse-k 1 and 0. That is 16 jobs:

```bash
for N in 8 16 32 64; do for F in fp32 bf16; do for C in 1 0; do
  H=$([ $N -ge 64 ] && echo 12 || echo 6); M=$([ $N -ge 64 ] && echo 64 || echo 32)
  $J/snap_launch.sh g3_N${N}_${F}_ck$C $M $H $J/cmd_mesh_fmt.sh reg $N $F $C
done; done; done
EI="-DNO_ZERO_INIT --x-initial unique --x-assign unique"
for F in fp32 bf16; do $J/snap_launch.sh g3_ri_$F 32 6 bash -c "export EXTRA_FLAGS='$EI' SIM_ARGS=+verilator+rand+reset+2; $J/cmd_mesh_fmt.sh reg 16 $F 1"; done
```

First check that the mesh Makefile passes `EXTRA_FLAGS` to Verilator and `SIM_ARGS` to the simulator: `grep -n "EXTRA_FLAGS\|SIM_ARGS" SystolicMesh/Makefile`. If `SIM_ARGS` is missing, add it the way SIENNA's Makefile has it, and commit that before launching.

Expected: every job passes every test. bf16 is exact everywhere; fp32 passes within tolerance with cycles equal to the previous mesh sweeps where they exist. For latency, bf16's single-set latency should be fp32's minus 5 cycles, since the mesh latency counts the multiplier once and it drops from 8 to 3 cycles. Steady-state throughput should be equal, since one product per PE per cycle holds in both formats. Any other difference needs an explanation before the gate passes.

- [ ] **Step 2: Report**

Write `testbenches/results/uniform/mesh_gate.log` (SIENNA) from the job reports:
- a table of N, T, format, collapse-k, tests passed, exact count, single-set latency and cycles per set;
- the fp32-to-bf16 latency difference per row;
- random-init results;
- job wall times;
- what was not covered.

- [ ] **Step 3: Report to Soham and wait**

Level 4 starts after Soham has seen G3.

---

## Level 4: SIENNA (gate G4 in Task 23)

Paths are relative to `/proj/work/spramanik/SIENNA` on branch `bf16`.

### Task 18: `Maxpool_2D` compares any float width

**Files:**
- Modify: `Maxpool/Maxpool_2D.sv`
- Create: `testbenches/TB_maxpool_fmt.sv`

**Interfaces:**
- Produces: `Maxpool_2D #(... IS_FP32 = 1, EXP_W = 8, MAN_W = 23)`. With `IS_FP32` set and `DATA_WIDTH == 1 + EXP_W + MAN_W`, it does a sign-magnitude float compare with −∞ padding in that format. The parameter keeps its name; it now means "float compare". fp32 behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

`testbenches/TB_maxpool_fmt.sv`:

```systemverilog
`timescale 1ns / 1ps

// Maxpool_2D as sienna_top uses it (one 2x2 window per start), in bf16: sign-magnitude order, ties between +0 and -0 keep the first.
module TB_maxpool_fmt;
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  logic clk = 0, rst_n = 0, start = 0, valid_in = 0, done, out_valid;
  logic [W-1:0] data_in = '0, out_data;
  always #5 clk = ~clk;
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk(clk), .rst_n(rst_n), .start(start), .done(done), .data_in(data_in), .valid_in(valid_in),
      .out_data(out_data), .out_valid(out_valid));

  int errs = 0;
  task automatic window(input logic [W-1:0] v0, v1, v2, v3, input logic [W-1:0] want);
    logic [W-1:0] v[4] = '{v0, v1, v2, v3};
    logic [W-1:0] got = 'x;
    @(negedge clk) start = 1;
    @(negedge clk);
    for (int i = 0; i < 4; i++) begin
      valid_in = 1;
      data_in = v[i];
      @(posedge clk);
      if (out_valid) got = out_data;
      @(negedge clk);
    end
    valid_in = 0;
    repeat (2) begin
      @(posedge clk);
      if (out_valid) got = out_data;
    end
    @(negedge clk) start = 0;
    @(negedge clk);
    if (got !== want) begin
      errs++;
      $display("[FAIL] max(%h %h %h %h) = %h, want %h", v0, v1, v2, v3, got, want);
    end
  endtask

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    window(16'hBF80, 16'hC000, 16'hBF00, 16'hC040, 16'hBF00);  // all negative: -0.5 is the largest
    window(16'h3F80, 16'hBF80, 16'h4000, 16'h3F00, 16'h4000);  // mixed signs
    window(16'h8000, 16'h0000, 16'hBF80, 16'hBF80, 16'h8000);  // -0 first, +0 ties it: -0 stays
    window(16'h0000, 16'h8000, 16'hBF80, 16'hBF80, 16'h0000);  // +0 first: +0 stays
    window(16'hFF80, 16'hFF80, 16'hFF80, 16'hC2C8, 16'hC2C8);  // -inf inputs lose to -100
    window(16'h7F7F, 16'h0001, 16'h0080, 16'h3F80, 16'h7F7F);  // largest finite wins
    $display("TB_maxpool_fmt: %0d errors", errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

Run it (it must fail to build, since `EXP_W` does not exist yet):

```bash
$J/snap_launch.sh s18 4 1 bash -c 'verilator --binary --timing -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND --top-module TB_maxpool_fmt Maxpool/Maxpool_2D.sv testbenches/TB_maxpool_fmt.sv -o sim --Mdir Verilator_mp && Verilator_mp/sim'
```

- [ ] **Step 2: Generalize the compare and the floor**

Add the parameters after `IS_FP32`, and update its comment:

```systemverilog
    parameter bit IS_FP32     = 1,   // float sign-magnitude compare, any width given EXP_W and MAN_W (the name predates bf16)
    parameter int EXP_W       = 8,
    parameter int MAN_W       = 23
```

Below the localparams:

```systemverilog
  localparam bit FLOAT = IS_FP32 && (DATA_WIDTH == 1 + EXP_W + MAN_W);
  localparam logic [DATA_WIDTH-1:0] FLOOR = FLOAT ? {1'b1, {EXP_W{1'b1}}, {MAN_W{1'b0}}}  // -infinity in the format
                                                  : {1'b1, {(DATA_WIDTH - 1) {1'b0}}};  // most negative integer
```

`is_greater`:

```systemverilog
  function automatic logic is_greater(input logic [DATA_WIDTH-1:0] a, input logic [DATA_WIDTH-1:0] b);
    if (FLOAT) begin
      if ((a[DATA_WIDTH-2:0] == 0) && (b[DATA_WIDTH-2:0] == 0)) return 1'b0;
      if (a[DATA_WIDTH-1] != b[DATA_WIDTH-1]) return !a[DATA_WIDTH-1];
      if (!a[DATA_WIDTH-1]) return a[DATA_WIDTH-2:0] > b[DATA_WIDTH-2:0];
      return a[DATA_WIDTH-2:0] < b[DATA_WIDTH-2:0];
    end else begin
      return $signed(a) > $signed(b);
    end
  endfunction
```

`NEG_FLOOR` in `gen_stream` becomes `localparam logic [DATA_WIDTH-1:0] NEG_FLOOR = FLOOR;`. In the batch path, `if (IS_FP32 && DATA_WIDTH == 32) max_val = 32'hFF800000; else max_val = ...;` becomes `max_val = FLOOR;`. Update the comment above `is_greater` to `// Float (sign-magnitude) or signed-integer compare`.

- [ ] **Step 3: Run it**

Same command as Step 1. Expected: `TB_maxpool_fmt: 0 errors`, `RESULT: PASSED`.

- [ ] **Step 4: Commit (SIENNA)**

```bash
git add Maxpool/Maxpool_2D.sv && git commit -m "Maxpool_2D: float compare and -infinity floor at any width"
git add testbenches/TB_maxpool_fmt.sv && git commit -m "TB_maxpool_fmt: bf16 windows, signed-zero ties"
```

Push after Task 20, with the SystolicMesh and GPNAE pointer bumps; SIENNA is outermost.

### Task 19: `dropout` in the build's format

**Files:**
- Modify: `Dropout/dropout.sv` (CRLF line endings: keep them)
- Create: `testbenches/TB_dropout_fmt.sv`

**Interfaces:**
- Produces: `dropout #(int EXP_W = 8, int MAN_W = 23, int DATA_WIDTH = 1 + EXP_W + MAN_W, DROPOUT_P_PERCENT, LFSR_WIDTH, CONST_ZERO, CONST_ONE = 1.0 in the format, CONST_SCALE = 2.0 in the format)`.

- [ ] **Step 1: Write the failing test**

`testbenches/TB_dropout_fmt.sv`:

```systemverilog
`timescale 1ns / 1ps

// dropout in bf16, training mode: every output is x * 2 or a zero with x's sign; the unit's latency is the format's multiplier's.
module TB_dropout_fmt;
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  logic clk = 0, rst_n = 0, in_valid = 0, reseed = 0, valid_out;
  logic [W-1:0] data_in = '0, data_out;
  always #5 clk = ~clk;
  dropout #(.EXP_W(EXP_W), .MAN_W(MAN_W), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .training_mode(1'b1), .data_in(data_in), .reseed_i(reseed),
      .seed_i(32'h2ACE002A), .data_out(data_out), .valid_out(valid_out));

  logic [W-1:0] q[$];
  int errs = 0, n = 0, kept = 0;
  always @(posedge clk)
    if (valid_out) begin
      logic [W-1:0] x = q.pop_front();
      logic [W-1:0] dbl = {x[W-1], x[W-2:MAN_W] + EXP_W'(1), x[MAN_W-1:0]};  // x * 2 for these normal inputs
      n++;
      if (data_out === dbl) kept++;
      else if (data_out !== {x[W-1], {(W - 1) {1'b0}}}) begin
        errs++;
        $display("[FAIL] %h -> %h, neither %h nor a signed zero", x, data_out, dbl);
      end
    end

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    for (int i = 0; i < 64; i++) begin
      @(negedge clk);
      in_valid = 1;
      data_in = {i[0], EXP_W'(120 + i % 16), MAN_W'(i * 37)};
      q.push_back(data_in);
    end
    @(negedge clk) in_valid = 0;
    repeat (20) @(posedge clk);
    if (n != 64 || kept == 0 || kept == 64) begin
      errs++;
      $display("[FAIL] %0d outputs, %0d kept", n, kept);
    end
    $display("TB_dropout_fmt: %0d outputs, %0d kept, %0d errors", n, kept, errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

Check `dropout.sv`'s port names before running (`clk, rst_n, in_valid, training_mode, data_in, reseed_i, seed_i, data_out, valid_out`, as sienna_top connects them). Run:

```bash
$J/snap_launch.sh s19 4 1 bash -c 'A=SystolicMesh/ArithmeticLibrary; verilator --binary --timing -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND --top-module TB_dropout_fmt $A/Common/src/sienna_fmt_pkg.sv $A/Multipliers/Radix4Booth/src/R4Booth.sv $A/Multipliers/Karatsuba/src/karatsubaUnsigned.sv $A/Multipliers/FP32/src/fp32Multiplier.sv $A/Multipliers/FP/src/fpMultiplier.sv Dropout/dropout.sv testbenches/TB_dropout_fmt.sv -o sim --Mdir Verilator_do && Verilator_do/sim'
```

Expected now: a build failure (no `EXP_W`), or mismatches, since fp32Multiplier is used at width 16.

- [ ] **Step 2: Generalize**

Parameters:

```systemverilog
    parameter int                    EXP_W             = 8,
    parameter int                    MAN_W             = 23,
    parameter int                    DATA_WIDTH        = 1 + EXP_W + MAN_W,
    parameter int                    DROPOUT_P_PERCENT = 50,
    parameter int                    LFSR_WIDTH        = 32,
    parameter logic [DATA_WIDTH-1:0] CONST_ZERO        = '0,
    parameter logic [DATA_WIDTH-1:0] CONST_ONE         = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h3F800000, MAN_W)),
    parameter logic [DATA_WIDTH-1:0] CONST_SCALE       = DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h40000000, MAN_W))
```

The check becomes `if (DROPOUT_P_PERCENT != 50 && CONST_SCALE == DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h40000000, MAN_W)))`. The multiplier:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $error("dropout: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (.clk_i(clk), .rstn_i(rst_n), .valid_i(mult_valid_in), .A(data_in), .B(CONST_SCALE),
                        .result_o(mult_out), .done_o(mult_done), .overflow_o(), .underflow_o(), .invalid_o());
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (.clk_i(clk), .rstn_i(rst_n), .valid_i(mult_valid_in), .A(data_in),
        .B(CONST_SCALE), .result_o(mult_out), .done_o(mult_done), .overflow_o(), .underflow_o(), .invalid_o());
  end
```

`KQ_DEPTH = 16` still covers the multiplier latency in both formats.

Line endings: after editing, `file Dropout/dropout.sv` must say `CRLF`. If the edit wrote LF lines, run `sed -i 's/\r*$/\r/' Dropout/dropout.sv`, then check `git diff --stat Dropout/dropout.sv` shows only the edited lines.

- [ ] **Step 3: Run it, plus the bad-format check**

The Step 1 command must print `RESULT: PASSED`. Then:

```bash
$J/snap_launch.sh s19_bad 4 1 bash -c 'A=SystolicMesh/ArithmeticLibrary; verilator --lint-only -Wno-fatal --top-module dropout -GEXP_W=5 -GMAN_W=10 $A/Common/src/sienna_fmt_pkg.sv $A/Multipliers/FP/src/fpMultiplier.sv $A/Multipliers/FP32/src/fp32Multiplier.sv $A/Multipliers/Karatsuba/src/karatsubaUnsigned.sv $A/Multipliers/Radix4Booth/src/R4Booth.sv Dropout/dropout.sv 2>&1 | grep -q "dropout: unsupported format" && echo REJECTED || echo NOT-REJECTED'
```

Expected: `REJECTED`.

- [ ] **Step 4: Commit (SIENNA)**

```bash
git add Dropout/dropout.sv && git commit -m "dropout: multiplier and constants in the build's format"
git add testbenches/TB_dropout_fmt.sv && git commit -m "TB_dropout_fmt: bf16 training-mode outputs"
```

### Task 20: One format through sienna_top, sienna_layer, sienna_multi

**Files:**
- Modify: `SystolicMesh`, `GPNAE` (pointers to the Task 16 and Task 13 tips), `Makefile`, `synth/sienna_rtl.f`
- Modify: `src/sienna_top.sv`, `src/sienna_layer.sv`, `src/sienna_multi.sv`
- Modify: `testbenches/TB_sienna_top.sv`, `TB_sienna_layer.sv`, `TB_sienna_multi.sv`, `TB_sienna_model.sv`, `regression.py` (package items only), `model_runner.py`, `gemm_sweep.py`, `perf_analysis.py` (option names only)

**Interfaces:**
- Produces: `sienna_top`, `sienna_layer` and `sienna_multi` take `EXP_W` and `MAN_W`, and `DATA_WIDTH` defaults to `1 + EXP_W + MAN_W`. Every data port, host port included, is `DATA_WIDTH` wide. The generated `test_config_pkg` has `EXP_W`, `MAN_W`, `DATA_WIDTH` and `EXACT_GOLDEN`, and no `OP_*`. Every script's `--op-format` becomes `--format`.

- [ ] **Step 1: Write the failing checks**

`/proj/work/spramanik/sienna_jobs/cmd_sienna_fmt.sh`:

```bash
#!/bin/bash
# SIENNA checks by format; args: lint EXP MAN | reg N T FMT [extra regression args]; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
case $1 in
  lint) for top in sienna_layer sienna_top; do
          verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module $top -Isrc -f synth/sienna_rtl.f -GN=16 -GNUM_LANES=32 \
            -GEXP_W=$2 -GMAN_W=$3 > lint_${top}_$2_$3.txt 2>&1
          echo "$top ($2, $3): $(grep -c '^%Warning' lint_${top}_$2_$3.txt) warnings, $(grep -c '^%Error' lint_${top}_$2_$3.txt) errors"
          grep -E "unsupported format|LATCH|MULTIDRIVEN|UNOPTFLAT|^%Error" lint_${top}_$2_$3.txt | head -5
        done
        mkdir -p testbenches/results/uniform && cp lint_*.txt testbenches/results/uniform/ ;;
  reg)  N=$2; T=$3; FMT=$4; shift 4
        [ $N -ge 32 ] && export EXTRA_FLAGS="$EXTRA_FLAGS --output-split 20000 --output-split-cfuncs 20000 --output-groups 64"
        python3 regression.py --n $N --tile-size $T --format $FMT "$@" ;;
esac
```

```bash
$J/snap_launch.sh s20_bad 8 1 $J/cmd_sienna_fmt.sh lint 5 10
```

Expected now: the `-GEXP_W` parameters do not exist yet, so there is no `unsupported format` line.

- [ ] **Step 2: Bump pointers and file lists**

```bash
cd SystolicMesh && git checkout <Task 16 tip> && cd .. && cd GPNAE && git checkout <Task 13 tip> && cd ..
test "$(git -C SystolicMesh/ArithmeticLibrary rev-parse HEAD)" = "$(git -C GPNAE/ArithmeticLibrary rev-parse HEAD)" && echo SAME-ARIL
```

Both AriL copies must be the same commit. The golden imports `fpu.py` through both, and Python loads it once.

`Makefile` `SM_LIB_FILES`: add `Common/src/sienna_fmt_pkg.sv` first, and `Multipliers/FP/src/fpMultiplier.sv` and `Adders/FP/src/fpAdder.sv`. Remove `Multipliers/FPWiden/src/fpMulWiden.sv`. In `synth/sienna_rtl.f`, make the same change with the `SystolicMesh/ArithmeticLibrary/` prefix, the package on the first line.

- [ ] **Step 3: RTL rename and constants**

In `src/sienna_top.sv`, `src/sienna_layer.sv` and `src/sienna_multi.sv`:
- replace the `DATA_WIDTH`, `OP_EXP_W`, `OP_MAN_W` and `OP_W` parameter lines with:

```systemverilog
    parameter int    EXP_W             = 8,   // the build's number format: fp32 8/23, bf16 8/7
    parameter int    MAN_W             = 23,
    parameter int    DATA_WIDTH        = 1 + EXP_W + MAN_W,  // every word: operands, results, activations
```

  with the column alignment of each file;
- replace every `OP_W` with `DATA_WIDTH`, and pass `.EXP_W(EXP_W), .MAN_W(MAN_W)` wherever `.OP_EXP_W`/`.OP_MAN_W` are passed now.

In `sienna_top.sv`:
- `gpnae_poly` gets `.EXP_W(EXP_W), .MAN_W(MAN_W)`, and its `DATA_WIDTH(GPNAE_DATA_WIDTH)` stays. Check that `GPNAE_DATA_WIDTH` is `DATA_WIDTH`;
- `Maxpool_2D` gets `.EXP_W(EXP_W), .MAN_W(MAN_W)`, and `dropout` gets `.EXP_W(EXP_W), .MAN_W(MAN_W)`;
- the pooling pad: add `localparam logic [DATA_WIDTH-1:0] NEG_INF = {1'b1, {EXP_W{1'b1}}, {MAN_W{1'b0}}};  // pooling pad: -infinity in the format`, and on line 417 replace `32'hFF800000` with `NEG_INF`.

In `sienna_layer.sv`:
- `ONE` is `{1'b0, EXP_W'((1 << (EXP_W - 1)) - 1), MAN_W'(0)}` with the comment `// 1.0 in the format`. Delete `OP_BIAS` and the `op_to_fp32` function;
- the bias capture is back to `bias_buf[wl_blk[0]] <= w_data_i;`.

Then `grep -n "OP_\|op_to_fp32\|32'hFF800000\|32'h3[fF]800000\|\[30:23\]" src/*.sv` must print nothing. Also run `grep -n "32'h\|\[31:0\]\|\[31\]" src/*.sv`, and for every hit say in the commit message why it is an integer (seeds, addresses) and not a float.

- [ ] **Step 4: Testbenches and scripts, mechanical rename**

In `testbenches/TB_sienna_{top,layer,multi,model}.sv`, `regression.py`, `model_runner.py`, `gemm_sweep.py` and `perf_analysis.py`, apply exactly these renames: `OP_EXP_W`→`EXP_W`, `OP_MAN_W`→`MAN_W`, `OP_W`→`DATA_WIDTH`, `op_format`→`fmt_name`, `--op-format`→`--format`, `OP_FORMATS`→`FORMATS`. Keep `op_round`, `op_hex` and `write_op_mem`; they are now "in the build's format". Change their docstrings to say so, and change `"fp16": (5, 10)` in `FORMATS` to nothing: remove the entry, since fp16 is not a supported build.

The generated package items become:

```python
        ("EXP_W", FORMATS[fmt][0], "int"),
        ("MAN_W", FORMATS[fmt][1], "int"),
        ("DATA_WIDTH", 1 + sum(FORMATS[fmt]), "int"),
        ("EXACT_GOLDEN", int(fmt != "fp32"), "int"),
```

`TB_sienna_top.sv`:
- pass `.EXP_W(EXP_W), .MAN_W(MAN_W)` to the DUT;
- in `check_tolerance`, first statement: `if (EXACT_GOLDEN) begin info = $sformatf("exp=%h act=%h", expected, actual); return expected === actual; end`;
- `f32(bound[31:0])` becomes `f32(32'(bound))`;
- the self-test runs its fp32 lines only `if (!EXACT_GOLDEN)`, and in exact mode checks `check_tolerance(DATA_WIDTH'(16'h3F80), DATA_WIDTH'(16'h3F80), '0, s)` passes and `...16'h3F81...` fails.

Apply the same exact-mode change to any other TB that compares against a golden (`grep -n "check_tolerance\|REL_TOL" testbenches/TB_sienna_*.sv`).

- [ ] **Step 5: Run the checks**

```bash
$J/snap_launch.sh s20_bad 8 1 $J/cmd_sienna_fmt.sh lint 5 10
$J/snap_launch.sh s20_lint 8 1 bash -c "$J/cmd_sienna_fmt.sh lint 8 7; $J/cmd_sienna_fmt.sh lint 8 23"
$J/snap_launch.sh s20_fp32 32 4 $J/cmd_sienna_fmt.sh reg 16 4 fp32
```

Expected:
- `unsupported format` lines for (5, 10);
- clean bf16 and fp32 lints, with no `%Error`, `LATCH`, `MULTIDRIVEN` or `UNOPTFLAT`. Compare warning counts with `synthesis_readiness.log`'s 156, and explain new ones;
- fp32 regression 27/27, with every per-test cycle count equal to Task 0's `u0_sienna32`.

The bf16 regression needs Task 21's golden.

- [ ] **Step 6: Commit (SIENNA) and push all of SIENNA's work so far**

```bash
git add SystolicMesh GPNAE && git commit -m "Bump SystolicMesh and GPNAE: one number format per build"
git add src/sienna_top.sv src/sienna_layer.sv src/sienna_multi.sv && git commit -m "One number format through sienna_top, sienna_layer, sienna_multi"
git add Makefile synth/sienna_rtl.f && git commit -m "File lists: format package and narrow units; fpMulWiden no longer used"
git add testbenches/TB_sienna_top.sv testbenches/TB_sienna_layer.sv testbenches/TB_sienna_multi.sv testbenches/TB_sienna_model.sv \
  && git commit -m "Testbenches: the build's format, exact compare for narrow formats"
git add regression.py model_runner.py gemm_sweep.py perf_analysis.py && git commit -m "Scripts: --format replaces --op-format"
git push origin bf16
```

### Task 21: Bit-exact SIENNA golden in bf16

**Files:**
- Modify: `regression.py`
- Modify: the `PIPELINE_TESTS` list (one new test)

**Interfaces:**
- Consumes: `mesh_model.matmul` (Task 16), `gpnae_model.Lane` and `read_rom` (Task 11), `fpu`.
- Produces: `regression.py --format bf16` writes bf16 words for every matrix, bias and expected file. Expected results are bit-exact through mesh, activation, pooling and dropout. Every generated `.mem` file is checked for word width.

- [ ] **Step 1: Add the zero-rows test and run bf16 to see it fail**

Add to `PIPELINE_TESTS`, next to its neighbours:

```python
    dict(name="matmul_zero_rows_relu", mode="matmul", act="relu", zero_rows=True,
         description="rows of A all +0 or -0: zero sums start from +0 in the PE, a float golden would give -0"),
```

In `generate_vectors`, after A and B are built: `if cfg.get("zero_rows"): A[0::4, :] = np.float32(-0.0); A[1::4, :] = np.float32(0.0)`.

```bash
$J/snap_launch.sh s21_fail 32 4 $J/cmd_sienna_fmt.sh reg 16 4 bf16
```

Expected: failures on every test, because the expected files are still fp32 floats.

- [ ] **Step 2: Write the bit-level golden**

At the top of `regression.py`:

```python
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh"))
sys.path.insert(0, os.path.join(ROOT, "GPNAE"))
import mesh_model  # noqa: E402
import gpnae_model  # noqa: E402
from mesh_model import fpu  # noqa: E402
```

Helpers, next to `_golden_from_c`:

```python
def fmt_bits(x, fmt: str) -> np.ndarray:
    """Values already rounded to the format (op_round), as its bit patterns."""
    return np.array([int(h, 16) for h in op_hex(np.asarray(x, dtype=np.float32), fmt)], dtype=np.int64).reshape(np.shape(x))


def bits_float(b, fmt: str) -> np.ndarray:
    f = fpu.FORMATS[fmt]
    return (np.asarray(b, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def write_bits(path: str, bits, fmt: str) -> None:
    d = (fpu.FORMATS[fmt].w + 3) // 4
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _is_greater_bits(a: int, b: int, f) -> bool:
    """Maxpool_2D's float compare on bit patterns: +0 and -0 tie."""
    S = 1 << (f.w - 1)
    am, bm = a & (S - 1), b & (S - 1)
    if am == 0 and bm == 0:
        return False
    if (a & S) != (b & S):
        return not (a & S)
    return am > bm if not (a & S) else am < bm


def _maxpool_bits(x, ph: int, pw: int, pad: int, f) -> np.ndarray:
    """sienna_top's dispatcher order: window rows outer, columns inner, -infinity outside, a running max from -infinity."""
    H, W = x.shape
    ninf = (1 << (f.w - 1)) | (f.emax << f.m)
    oh, ow = (H + 2 * pad - ph) // ph + 1, (W + 2 * pad - pw) // pw + 1
    out = np.empty((oh, ow), dtype=np.int64)
    for i in range(oh):
        for j in range(ow):
            run = ninf
            for pr in range(ph):
                for pc in range(pw):
                    r, c = i * ph + pr - pad, j * pw + pc - pad
                    v = int(x[r, c]) if 0 <= r < H and 0 <= c < W else ninf
                    if _is_greater_bits(v, run, f):
                        run = v
            out[i, j] = run
    return out


def dropout_keep(n: int, p: float, seed: int, num_lanes: int) -> np.ndarray:
    """dropout.sv's keep decisions for n outputs, window w on lane w % num_lanes."""
    thr = ((2**32 - 1) * int(round(p * 100))) // 100
    states = [_lane_seed(seed, lane) for lane in range(num_lanes)]
    keep = np.empty(n, dtype=bool)
    for w in range(n):
        lane = w % num_lanes
        states[lane] = _lfsr_next(states[lane])
        keep[w] = states[lane] >= thr
    return keep


def _golden_bits(passes, bias, cfg: dict, act: str, drop_seed: int, fmt: str) -> tuple:
    """Bit-exact mesh, activation, pooling and dropout for one set in a narrow format; passes are (A bits, B bits) in order."""
    f = fpu.FORMATS[fmt]
    N = cfg.get("n", 16)
    C = mesh_model.matmul(f, passes, N, cfg.get("tile_size", 4), cfg.get("collapse_k", 1), bias)
    lane = gpnae_model.Lane(f, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory",
                                                                 gpnae_model.coeff_file(f))))
    code = activation_to_code(act)
    A = lane.run(C, code)
    P = _maxpool_bits(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1), f)
    if not cfg.get("training", False):
        return C, A, P, P.copy()
    flat = P.flatten()
    keep = dropout_keep(flat.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    scale = fpu.from_fp32(int(np.float32(1.0 / (1.0 - cfg.get("dropout_p", 0.5))).view(np.uint32)), f.m)
    prod = fpu.mul(f, flat, np.full_like(flat, scale))[0]
    F = np.where(keep, prod, flat & (1 << (f.w - 1)))  # a dropped beat is a zero with the input's sign
    return C, A, P, F.reshape(P.shape)
```

Make `apply_dropout` use `dropout_keep` for its decisions, so the fp32 and bf16 paths share one LFSR replay. Behaviour must stay the same: `_check_dropout_generator()` runs at the start of every regression and must still pass.

- [ ] **Step 3: Use it in `generate_vectors` for narrow formats**

With `fmt = cfg.get("fmt_name", "fp32")` and `exact = fmt != "fp32"`:
- set 0: when `exact`, `_, _, _, F0 = _golden_bits([(fmt_bits(A, fmt), fmt_bits(B, fmt))], fmt_bits(bias_of(0), fmt) if use_bias else None, cfg, act_type, drop_seed, fmt)`. Write `expected_output.mem` with `write_bits`, and write `bound_output.mem` as `write_bits(..., np.zeros_like(F0), fmt)`;
- the streamed loop: keep a list `grp` of `(fmt_bits(Ak), fmt_bits(Bk))`, reset when `k % passes == 0`, and `gbias = fmt_bits(op_round(bk, fmt), fmt) if use_bias else None` taken at the group's first pass. On the group's last pass, `Fk = _golden_bits(grp, gbias, cfg, act_k, set_dropout_seed(drop_seed, k), fmt)[3]`. Partial sets get an empty expected file as today;
- bias files: `write_op_mem(..., op_round(bk, fmt), fmt)` in place of `write_mem`, so the bias is in the format as the hardware reads it. The bias must also be rounded with `op_round` before either golden sees it, fp32 unchanged;
- `dump_golden_trace` gets the floats: `bits_float(C, fmt)` and so on;
- after every file is written, and before `make`, check the width. Every line of every `matrix_*`, `bias_*`, `expected_output*` and `bound_output*` file must have `(W + 3) // 4` hex digits, and otherwise the run raises `ValueError` naming the file and line.

The fp32 path is unchanged: `exact` is false, and every fp32 file is written as before. Step 5 checks that byte for byte.

- [ ] **Step 4: Run bf16 at N=16**

```bash
$J/snap_launch.sh s21_bf16 32 4 $J/cmd_sienna_fmt.sh reg 16 4 bf16
$J/snap_launch.sh s21_bf16ck0 32 4 bash -c "sed -i -E 's/(parameter int +COLLAPSE_K +=) 1,/\1 0,/' src/sienna_top.sv && grep -qE 'COLLAPSE_K +\= 0,' src/sienna_top.sv && $J/cmd_sienna_fmt.sh reg 16 4 bf16 --collapse-k 0"
```

The second run needs `regression.py` to pass `collapse_k` into `cfg`. Add `--collapse-k` (default 1) to its options for the golden only. The job's `sed` changes the RTL default in `sienna_top` on the snapshot; check that pattern against the file first, because a `sed` that matches nothing would test COLLAPSE_K=1 twice. Expected: 28/28 (27 plus `matmul_zero_rows_relu`), every element exact, in both runs.

- [ ] **Step 5: fp32 stimulus byte-identical**

In one job, generate every test's files with the Task 20 tree and with this one (`regression.py gen` for each test), and `cmp` them. Take the Task 20 copy of `regression.py` from `git show <Task 20 commit>:regression.py > .claude/scratch/regression_t20.py` before launching. Expected: identical, except `matmul_zero_rows_relu`, which is new.

- [ ] **Step 6: Commit (SIENNA)**

```bash
git add regression.py && git commit -m "regression: bit-exact golden for narrow formats (mesh, activation, pooling, dropout); zero-rows test"
git push origin bf16
```

### Task 22: Models and GEMM sweep in uniform bf16

**Files:**
- Modify: `model_runner.py`, `gemm_sweep.py` (reading outputs in the format)

**Interfaces:**
- Consumes: the Task 20 package items, `op_hex`, `bits_float`.
- Produces: `model_runner.py --format bf16` and `gemm_sweep.py --format bf16` run on the layer engine built in bf16, and report error against float64 and against the fp32 run, plus cycles.

- [ ] **Step 1: Make output parsing format-aware**

`model_runner.read_outputs`: decode each word with `regression.bits_float(int(word, 16), fmt)` when `fmt != "fp32"`. The words are `(W+3)//4` hex digits. `write_sets` already goes through `op_hex`. `LayerSim.build` passes `fmt_name` to the package generator. Where `err_vs_float` compares against a float reference, keep the float64 reference and report both errors: vs float64, and vs the fp32 hardware run when one is given.

- [ ] **Step 2: Run the four models and the GEMM sweep, both formats**

Use the same commands and configurations as the pre_synthesis_v1 runs recorded in `testbenches/results/perf/pre_synthesis_v1_report.log` (N=16, 32 lanes), once with `--format fp32` and once with `--format bf16`. Expected: fp32 cycles and errors equal the pre_synthesis_v1 values. bf16 runs to completion with finite outputs, and its cycles and errors get reported. Find the mixed-bf16 results of the same models (search `testbenches/results/perf/` and `sienna_jobs/runs/` for the `--op-format bf16` runs). If they are not on disk, say so in the report rather than quoting remembered numbers.

- [ ] **Step 3: Commit (SIENNA)**

```bash
git add model_runner.py gemm_sweep.py && git commit -m "model_runner, gemm_sweep: outputs read in the build's format"
git push origin bf16
```

### Task 23: Gate G4, SIENNA sweeps

- [ ] **Step 1: Launch**

The (N, T) pairs are those of `testbenches/results/perf/nt_sweep_2026-09-26.log`, which is the sweep of N=8..64 with every valid tile size. Run each in fp32 and bf16:

First give `cmd_nt_sweep.sh` and `cmd_nt_act.sh` a fourth argument `FMT`, passed as `--format $4` to `perf_analysis.py`. In `perf_analysis.py`, `MUL_LAT, ADD_LAT` come from the format (`8, 5` for fp32, `3, 5` for bf16), so its latency model follows the build. Check the pair list the `awk` prints against the log by eye before launching. For N ≥ 32, take the job memory from the earlier sweep's runs (`/usr/bin/time` maximum resident size in `runs/nt_N64_*/stdout.log`), not from a guess.

```bash
PAIRS=$(awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print $1","$2}' testbenches/results/perf/nt_sweep_2026-09-26.log | sort -u)
for NT in $PAIRS; do N=${NT%,*}; T=${NT#*,}
  for F in fp32 bf16; do
    $J/snap_launch.sh g4_N${N}_T${T}_$F 64 12 $J/cmd_sienna_fmt.sh reg $N $T $F
    $J/snap_launch.sh g4_perf_N${N}_T${T}_$F 64 12 bash -c "$J/cmd_nt_sweep.sh $N 32 $T $F && $J/cmd_nt_act.sh $N 32 $T $F"
  done
done
for F in fp32 bf16; do $J/snap_launch.sh g4_ri_$F 32 6 $J/cmd_randinit.sh --n 16 --tile-size 4 --format $F; done
$J/snap_launch.sh g4_lint 16 2 bash -c "$J/cmd_sienna_fmt.sh lint 8 7; $J/cmd_sienna_fmt.sh lint 8 23"
```

Expected:
- every regression passes; bf16 is exact on every element;
- fp32 cycles equal the earlier sweep's;
- the perf model matches measured latency in both formats. bf16 mesh latency should be fp32's minus 5 cycles; activation rounds are shorter by whatever the barrel MAC's shorter loop gives;
- random-init passes; lint is clean.

- [ ] **Step 2: Report**

`testbenches/results/uniform/sienna_uniform_sweep.log`:
- a per (N, T, format) table: tests passed, exact count, cycles per set, single-set latency, model latency;
- fp32 against bf16 cycles and latency;
- the four models and GEMM: accuracy in fp32, mixed bf16 (if found) and uniform bf16, plus cycles;
- storage per format, from `readiness_report.py`'s inventory with every word at the format's width;
- what was not run: synthesis, int8, and the published Taylor lane in bf16.

- [ ] **Step 3: Report to Soham**

### Task 24: Close out

**Files:**
- Modify: `.claude/skills/sienna-uniform-format/SKILL.md` (status line, and anything the work proved different from the design)
- Modify: memory `sienna-repo-layout.md` if a push path changed

- [ ] **Step 1: Update the skill's status**

Replace the status line with `**Status: uniform bf16 implemented and verified <date>; int8 not started.**`, followed by one line per gate with its report path. Where the implementation departs from the design (D-1 to D-5, or anything found on the way), change the design text to match and say so in the commit.

- [ ] **Step 2: Commit and push, innermost first**

AriL (nothing left, or report fixes), GPNAE, SystolicMesh, then SIENNA:

```bash
git add .claude/skills/sienna-uniform-format/SKILL.md && git commit -m "skill: sienna-uniform-format implemented for bf16; status and departures from the design"
git push origin bf16
```

Check each push succeeded with `git status -sb` showing no `ahead`.

## Self-review against the spec

| Spec item | Task |
|---|---|
| EXP_W/MAN_W, width 1+E+M | 1, 3, 5, 9, 10, 13, 15, 18, 19, 20 |
| Truncate like the fp32 units | 3, 5 (equivalence with fp32 units at 8/23) |
| Latencies from `sienna_fmt_pkg` | 1; used in 9, 13, 15; checked in 4, 6 |
| U = adder latency + 1 | 15 |
| fp32 builds keep fp32 units, bit- and cycle-identical | generate branches in 9, 10, 13, 15, 19; identity checks in 8-10, 13, 15, 16, 20, 21 |
| fpMulWiden stays, unused | 7 (still tested), 15, 20 (removed from file lists) |
| gpnae_poly only | 9-13; Taylor lane checked unchanged in fp32 |
| poly_coeffs_bf16.mem, fp32 files unchanged | 12 (guard and `git status` check) |
| Bit-exact golden | 2, 11, 16, 21 |
| DV like the fp32 units (DPI SoftFloat, Vivado vectors, Makefile) plus (8, 23) identity | 3-6 |
| Python unit models checked against RTL | 2 (fp32 units), 4, 6 (bf16 units) |
| GPNAE framework in bf16, accuracy vs reference | 11, 13, 14 |
| System checks: regression, models, GEMM, random-init, lint, fp32 identical | 20-23 |
| Exhaustive generate blocks, fail on unsupported format | 9, 10, 13, 15, 19, 20 |
| Constants re-encoded (1.0, −∞, NaN, λ, λα, thresholds) | 10, 13, 18, 19, 20 |
| Verification order AriL → GPNAE → mesh → SIENNA | gates 7, 14, 17, 23 |
