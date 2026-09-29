# SIENNA int8 (2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An int8 build of SIENNA whose conv and fully connected results are bit-exact against TFLite's own int8 kernels, with fp32 and bf16 builds bit- and cycle-identical to today.

**Architecture:** int8 is one more format of the uniform-format design (`EXP_W = 0, MAN_W = 7`). New AriL integer units (`intMultiplier`, `intAdder`, `fxMac`, `tfliteRequant`) and a numpy model of them (`ipu.py`) replace the float units under generate branches. The mesh multiplies int8 and accumulates int32 (`ACC_W`); a TFLite-exact requantize stage at the GPNAE lane feed turns int32 back into int8; GPNAE evaluates its polynomial in 16-bit fixed point; pooling and dropout stay int8. TensorFlow's interpreter (reference kernels) is the oracle for everything TFLite defines; bit-exact numpy models are the golden for the rest. Verification goes bottom up in gated levels: G0 oracle, G1 AriL, G2 GPNAE, G3 mesh, G4 SIENNA.

**Tech Stack:** SystemVerilog, Verilator (farm, via `snap_launch_tree.sh`), Python 3 with numpy, TensorFlow (TFLite converter and interpreter, reference kernels).

**Spec:** `.claude/skills/sienna-int8/SKILL.md` (approved 2026-09-29). Background: the `sienna-uniform-format`, `sienna-rtl` and `sienna-back-to-back` skills, and the bf16 plan (`.claude/skills/sienna-uniform-format/implementation-plan.md`), whose conventions this plan reuses.

## Global Constraints

- Format selection: `EXP_W = 0, MAN_W = 7` is int8; `DATA_WIDTH = 1 + EXP_W + MAN_W = 8`; `sienna_fmt_pkg::is_int(exp_w)`; `supported()` accepts (0, 7); anything else fails elaboration with `$fatal`.
- Accumulate width `ACC_W` = 32 for int8, `DATA_WIDTH` for floats: PE partial sums, reducer tree, bias input, mesh result memory, wide read into requantize.
- Mesh: int8 x int8 -> int16 sign-extended into an int32 two's-complement accumulate (wraps like TFLite); no zero-point hardware (the input zero-point term is folded into the int32 bias by software; conv padding with z_a in the software im2col).
- Requantize: TFLite's conv / FC epilogue exactly (`MultiplyByQuantizedMultiplier`, + z_out, clamp to [act_min, act_max]); the rounding variant is whichever TFLite's reference kernels use, found and pinned by a test in Task 2, never assumed; one unit per GPNAE lane at the lane feed; result memory stays int32.
- GPNAE in int8: the same Horner polynomial and barrel_mac / gpnae_poly structure on 16-bit fixed point (Q4.11), coefficients re-encoded into `poly_coeffs_int8.mem`, saturation beyond the fitted range, `gpnae_tail` not instantiated; tanh y*128 zp 0, sigmoid y*256 zp -128 (TFLite's fixed output quantization); SELU per-layer output rescale; fp32 / bf16 coefficient files never change.
- Pooling: Maxpool_2D's integer compare, pad -128. Dropout: inference bypass; training keeps values, dropped ones become the output zero point of the set's activation (D-5), 1/keep folded into the output scale.
- One MAC per PE per cycle, as today (MAC packing is out of scope).
- Verification order: G0 oracle, G1 AriL, G2 GPNAE, G3 mesh, G4 SIENNA; a gate passes and is reported before the next level starts; every gate reruns the fp32 and bf16 suites, requiring identical results and cycle counts (G3 reruns the ruled subset, Task 15).
- The oracle is TensorFlow's interpreter with `experimental_op_resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_REF` (reference kernels, never the optimized ones).
- Work in `/proj/work/spramanik/SIENNA_int8` on branch `int8` (off `bf16`) in all four repos; the bf16 checkout `/proj/work/spramanik/SIENNA` stays clean for fp32 / bf16 reruns.
- Every build and simulation runs on the farm: `TREE=/proj/work/spramanik/SIENNA_int8 $J/snap_launch_tree.sh NAME MEM_GB HOURS cmd...` with `J=/proj/work/spramanik/sienna_jobs` (output in `$J/runs/NAME/`); 32 GB minimum; multi-command jobs are scripts in `$J/cmds/`, never `bash -c "..."`. N >= 32 builds use `TRACE=0 OPT_FAST=-O0`.
- Every level's commands assume `J=/proj/work/spramanik/sienna_jobs`, `T=/proj/work/spramanik/SIENNA_int8` and `export TREE=$T` (Level 4 wraps the launch in `L` and `B`). A launch meant for the bf16 tree (a reference run) sets `TREE=/proj/work/spramanik/SIENNA` on its own command line. A launch without `TREE` snapshots the bf16 tree and silently re-tests bf16.
- Nothing a later job reads lives under a `results/` directory: `snap_launch_tree.sh` leaves every path component named `results` out of a snapshot, and `run_snap.sh` copies the snapshot's `testbenches/results/` back to `$J/runs/NAME/results/`. Files a later task needs are copied from there into the tree by hand, after checking the target does not exist.
- Unit latencies come from `sienna_fmt_pkg` (`mul_lat`, `add_lat`, `fx_lat()` = 2, `req_lat()` = 3) and the requantize rounding from `sienna_fmt_pkg::REQ_ROUNDING` in RTL and from `testbenches/tflite_int8/rounding.txt` in Python (`ipu.REQ_ROUNDING`, `tflite_ref.ROUNDING`); outside the package no RTL writes either as a literal.
- Both AriL checkouts, `SystolicMesh/ArithmeticLibrary` and `GPNAE/ArithmeticLibrary`, are on the same commit from Task 9 on; every gate from G2 checks it.
- In shell checks use `command grep` / `command diff` (the interactive shell aliases them to other tools).
- Commits: one change per commit, RTL before testbench and scripts, and push innermost first: AriL (`git push git@github.com:SoHam-56/ArithmeticLibrary.git int8`), then GPNAE and SystolicMesh (`git push origin int8`), then SIENNA; check each push. No Co-Authored-By or generated-by lines.
- Comments are one line. Generated reports are `.log`. Never delete a file without Soham's go-ahead. GPNAE is published work: report math bugs, do not change them silently. Keep line endings (dropout.sv is CRLF).

## Review Focus

1. **Requantize corners TFLite defines and random data never hits:** `acc = INT32_MIN` with `mult = INT32_MIN` (the one saturating case of SaturatingRoundingDoublingHighMul), negative values exactly halfway in RoundingDivideByPOT (TFLite rounds them away from zero), and left shifts (`shift > 0`). A reasonable person expects the RTL to equal TFLite on every one. Pinned in Task 1 (known values), Task 7 (corner vectors) and Task 2 (the oracle confirms the model on them).
2. **A padded conv with a non-zero input zero point.** Padding must be the input zero point, not 0, or border outputs differ from TFLite. Pinned in Task 2 (a 3x3 SAME conv model with z_a != 0) and Task 21 (the same model through the RTL).
3. **Per-channel parameters landing on the wrong channel** (a set's column c using channel c+1's multiplier after a tile or set boundary). Pinned in Task 20 (a test with a distinct random multiplier per channel, and a negative control with every lane one channel off), Task 21, and Task 22 (`gemm_sweep` int8 layers with 40 output channels at N = 16, so the per-channel words cross column blocks).
4. **An unsupported format** (for example `EXP_W = 0, MAN_W = 15`, an int16 attempt) must fail elaboration in every block that picks a unit, not build with int8 or float units. Pinned in Tasks 3, 9, 11, 13, 18 and 19 with a lint build that must fail with the block's message.
5. **Random power-up state in the new units and the requantize pipeline** (valid bits and parameter registers). Pinned in Task 8 (unit TBs with `--x-initial unique`), Task 12 (GPNAE), Task 15 (mesh) and Task 23 (SIENNA random-init regression in int8).

## Decisions this plan makes

Cited as D-1 .. D-8 in the tasks; flag them at review.

- **D-1: fxMac floors.** `fxMac` shifts the product right with floor (an arithmetic shift), matching AriL's truncating
  convention; `ipu.fx_mac` mirrors it (Tasks 1, 6).
- **D-2: requantize and GPNAE parameters travel per set.** `sienna_top` (and `sienna_multi`) take, with each accepted
  start, `req_mult_i [N-1:0][31:0]` and `req_shift_i [N-1:0][7:0]` (per output channel, beside `bias_i`), and the
  layer-wide `req_zp_i, req_min_i, req_max_i [7:0]`, `gp_mx_i [15:0]` (below 2^15), `gp_shx_i [4:0]`,
  `gp_mout_i [31:0]`, `gp_shout_i [7:0]`, `gp_zout_i [7:0]`. `gpnae_poly` gains `gp_mx_i [15:0], gp_shx_i [4:0],
  gp_zin_i [7:0], gp_mout_i [31:0], gp_shout_i [7:0], gp_zout_i [7:0]`; `sienna_top` drives `gp_zin_i` from the set's
  `req_zp` (the lane's input is the requantize output). An accumulate group uses its activated (last) pass's
  parameters and its first pass's bias. The ports exist in every format and are ignored (tied off) outside int8
  (Tasks 11, 16, 19).
- **D-3: requantize sits at the GPNAE lane feed.** `requant_lanes`: one `tfliteRequant` per lane on the mesh's wide
  read; lane k's word at beat b is requantized with channel `(k * PER_LANE + b) % N` (the element's column); latency
  `req_lat()`; rounding `sienna_fmt_pkg::REQ_ROUNDING`; the result memory stays int32 (Task 16).
- **D-4: GPNAE int8 output.** tanh `y * 128`, zero point 0; sigmoid `y * 256`, zero point -128; SELU
  `tfliteRequant(v, gp_mout, gp_shout, gp_zout)` with v in units of 2^-25 and `(gp_mout, gp_shout)` =
  QuantizeMultiplier(2^-25 / s_out); ReLU and linear pass the requantized input through. Beyond the fitted range the
  lane outputs the saturated value (tanh past |x| = 3.125, sigmoid past |x| = 6.25, SELU below x = -7); `gpnae_tail`
  is not instantiated and rejects int8 (Tasks 10, 11).
- **D-5 (corrected): dropout in int8.** Inference: bypass. Training: kept values unchanged; a dropped value becomes the
  zero point of the value dropout sees, the output zero point of the set's activation: ReLU and linear `req_zp`,
  tanh 0, sigmoid -128, SELU `gp_zout`. The 1/keep factor is folded into the next layer's scale (Tasks 18, 20).
- **D-6: the int8 golden.** SIENNA's `regression.py` composes `mesh_model.matmul_int` + `ipu.requant` +
  `gpnae_model.LaneInt8.run` + integer max pooling (pad -128) + dropout (D-5); the mesh TB and the SIENNA TBs compare
  int8 bit for bit (Tasks 14, 20).
- **D-7: float units reject integer formats.** Once `supported(0, 7)` is true, a block without its int8 branch yet must
  not build float units at `EXP_W = 0`: `fpMultiplier` and `fpAdder` `$fatal` at elaboration when `EXP_W < 2`
  (elaboration only; fp32 and bf16 unchanged), and every consumer puts its `is_int` branch before the float branches
  when its task adds it (Task 3, then 9, 11, 13, 18).
- **D-8: data registers are not reset.** The integer units (and `tfliteRequant`) reset only their valid bits; `result_o`
  loads on a valid cycle and holds, random from power-up until the first result, so every consumer qualifies it with
  `done_o`. This saves a reset per data flop in N^2 PEs; the random power-up runs cover it (Tasks 4 to 8, 13).

---

## Level 0: setup, the integer models and the oracle (gate G0 in Task 2)

All paths are relative to `/proj/work/spramanik/SIENNA_int8` (branch `int8` in all four repos, created by Task 0). AriL
paths inside AriL tasks are relative to `SystolicMesh/ArithmeticLibrary`. Every command below starts from:

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/work/spramanik/SIENNA_int8
export TREE=$T   # snap_launch_tree.sh snapshots this tree
```

Two properties of the job scripts matter here. `snap_launch_tree.sh` excludes every path component named `results`
from the snapshot, so nothing a later job reads may live under a `results/` directory. `run_snap.sh` copies the
snapshot's `testbenches/results/` back to `$J/runs/NAME/results/`, so everything a job produces for the tree is written
there and copied into the tree by hand afterwards, after checking the target does not exist yet.

### Task 0: The int8 checkout, TensorFlow, job scripts, reference results

**Files:**
- Create: `/proj/work/spramanik/SIENNA_int8` (clone of the four repos, branch `int8` in each)
- Create: `/proj/work/spramanik/sienna_jobs/cmds/int8_tree.sh`, `int8_aril.sh`, `int8_fpref.sh` (the shared job scripts)
- Modify: `/proj/work/spramanik/sienna_jobs/venv` (TensorFlow installed)

**Interfaces:**
- Produces:
  - `TREE=/proj/work/spramanik/SIENNA_int8`; TensorFlow importable from `$J/venv/bin/python`.
  - `$J/cmds/int8_tree.sh NAME MEM HOURS cmd...`: launches `cmd` on a snapshot of the int8 tree (a one-line wrapper of
    `snap_launch_tree.sh` with TREE set).
  - `$J/cmds/int8_aril.sh UNIT TARGET [PLUSARGS]` (env `TAG` names the log, env `EXTRA_FLAGS` reaches the Makefile):
    runs `make TARGET PLUSARGS=...` in `SystolicMesh/ArithmeticLibrary/UNIT` of a snapshot, tees the output to
    `testbenches/results/int8/<UNIT with / as _>_<TARGET>[_<TAG>].log`, copies the unit's `Verilator/*/*.txt` dumps
    beside it, and fails when the log holds `RESULT: FAILED` or, for a target other than `lint*`, no `RESULT: PASSED`.
  - `$J/cmds/int8_fpref.sh`: the five fp32 / bf16 GPNAE runs (poly, poly range 8, Taylor, bf16 hw, bf16 hw range 8) into
    `testbenches/results/int8/fpref/<tag>/`.
  - The reference runs every gate compares against: `i0_sienna_fp32`, `i0_sienna_bf16` (SIENNA N = 16),
    `i0_reg32_fp32`, `i0_reg32_bf16` (SIENNA N = 32), `i0_multi_fp32` (`sienna_multi`), `i0_fpref` (GPNAE, on the bf16
    tree), and the bf16 branch's gate runs listed in Step 6.

- [ ] **Step 1: Check the /proj/work quota before adding anything**

```bash
quota -s 2>/dev/null | tail -3; du -sh /proj/work/spramanik/SIENNA /proj/work/spramanik/sienna_jobs 2>/dev/null
```

Expected: at least 8 GB free (the clone is ~1.5 GB with submodules, TensorFlow ~600 MB, build directories ~several GB). If less, stop and ask Soham what to clean (a full quota truncates builds and looks like a Verilator segfault).

- [ ] **Step 2: Clone the tree and create the int8 branches**

```bash
cd /proj/work/spramanik
git -c url."git@github.com:".insteadOf="https://github.com/" clone --recurse-submodules -b bf16 git@github.com:SoHam-56/SIENNA.git SIENNA_int8
cd SIENNA_int8 && git switch -c int8
for d in GPNAE SystolicMesh SystolicMesh/ArithmeticLibrary GPNAE/ArithmeticLibrary; do git -C $d switch -c int8; done  # each at the commit bf16 pins
git -C SystolicMesh/ArithmeticLibrary remote set-url --push origin git@github.com:SoHam-56/ArithmeticLibrary.git
git -C GPNAE/ArithmeticLibrary remote set-url --push origin git@github.com:SoHam-56/ArithmeticLibrary.git
git submodule status; for d in . GPNAE SystolicMesh SystolicMesh/ArithmeticLibrary GPNAE/ArithmeticLibrary; do echo "$d: $(git -C $d rev-parse --abbrev-ref HEAD) $(git -C $d rev-parse --short HEAD)"; done
```

Expected: every repo on `int8`, at the commits the `bf16` branch pins (SIENNA cf14921 or later, SystolicMesh e6fa73c, GPNAE 41cad2d, AriL d97e270). Push the new branches: AriL first (`git -C SystolicMesh/ArithmeticLibrary push -u origin int8`), then GPNAE and SystolicMesh, then SIENNA; check each.

- [ ] **Step 3: Install TensorFlow into the job venv**

```bash
/proj/work/spramanik/sienna_jobs/venv/bin/pip install "tensorflow==2.18.*" 2>&1 | tail -2
/proj/work/spramanik/sienna_jobs/venv/bin/python -c "import tensorflow as tf; print(tf.__version__); print(tf.lite.experimental.OpResolverType.BUILTIN_REF)"
```

Expected: a 2.18.x version (matching the installed `tflite` 2.18.0 schema package) and `OpResolverType.BUILTIN_REF`. The import itself is a light check; everything that runs models runs on the farm.

- [ ] **Step 4: Write the shared job scripts**

Each with `chmod +x`. `/proj/work/spramanik/sienna_jobs/cmds/int8_tree.sh`:

```bash
#!/bin/bash
# snap_launch_tree.sh on the int8 checkout; args: NAME MEM_GB HOURS then the command.
TREE=/proj/work/spramanik/SIENNA_int8 exec /proj/work/spramanik/sienna_jobs/snap_launch_tree.sh "$@"
```

`/proj/work/spramanik/sienna_jobs/cmds/int8_aril.sh` is `cmd_aril.sh` without the SoftFloat builds (the integer units do
not use SoftFloat), with its own log directory; it fails a run that never printed its verdict (a crash or a full quota
otherwise passes silently), except for the `lint*` targets, which print none:

```bash
#!/bin/bash
# AriL unit checks on a snapshot for the int8 work: make TARGET in SystolicMesh/ArithmeticLibrary/UNIT, log in testbenches/results/int8; args: UNIT TARGET [PLUSARGS]; env TAG names the log.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
ROOT=$(pwd); UNIT=$1; TARGET=$2; shift 2
A=SystolicMesh/ArithmeticLibrary; R=$ROOT/testbenches/results/int8
mkdir -p $R
LOG=$R/$(echo $UNIT | tr / _)_$TARGET${TAG:+_$TAG}.log
(cd $A/$UNIT && make $TARGET PLUSARGS="$*") 2>&1 | tee $LOG
rc=${PIPESTATUS[0]}
cp $A/$UNIT/Verilator/*/*.txt $R/ 2>/dev/null  # dumps and digests, for a look after the node is gone
grep -q "RESULT: FAILED\|^FAILURE:" $LOG && rc=1  # testbenches finish with status 0 even when they fail
[[ $TARGET == lint* ]] || grep -q "RESULT: PASSED" $LOG || rc=1  # no verdict printed (crash, full quota) is a failure
exit $rc
```

`/proj/work/spramanik/sienna_jobs/cmds/int8_fpref.sh`, the fp32 and bf16 GPNAE runs every int8 GPNAE step compares:

```bash
#!/bin/bash
# The fp32 and bf16 GPNAE runs every int8 GPNAE step compares with the bf16 branch; run from a snapshot root.
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/int8/fpref; mkdir -p $R
run() {  # tag, then regression.py arguments; keeps only this run's logs
  tag=$1; shift; touch $R/.t_$tag
  $J/cmd_gpnae_reg.sh "$@" > $R/$tag.txt 2>&1 || echo "RUN-FAIL $tag"
  mkdir -p $R/$tag && find GPNAE/testbenches/results -maxdepth 1 -name '*.log' -newer $R/.t_$tag -exec cp {} $R/$tag/ \;
}
run poly32 --lane poly --format fp32
run poly32_r8 --lane poly --format fp32 --range 8
run taylor32 --lane gpnae --format fp32
run bf16_hw --lane poly --format bf16 --model hw
run bf16_hw_r8 --lane poly --format bf16 --model hw --range 8
command grep -h "RESULT: " $R/*.txt
```

- [ ] **Step 5: Prove the int8 tree builds as the bf16 tree does (before any change), and run the references**

The int8 tree still holds the bf16 branch's code here, so these runs are the fp32 / bf16 references of Levels 2 and 4.
`i0_fpref` runs on the clean bf16 tree itself.

```bash
J=/proj/work/spramanik/sienna_jobs
$J/cmds/int8_tree.sh i0_sienna_fp32 64 6 bash $J/cmd_sienna_fmt.sh reg 16 4 fp32
$J/cmds/int8_tree.sh i0_sienna_bf16 64 6 bash $J/cmd_sienna_fmt.sh reg 16 4 bf16
$J/cmds/int8_tree.sh i0_reg32_fp32 32 12 $J/cmd_sienna_fmt.sh reg 32 4 fp32
$J/cmds/int8_tree.sh i0_reg32_bf16 32 12 $J/cmd_sienna_fmt.sh reg 32 4 bf16
$J/cmds/int8_tree.sh i0_multi_fp32 32 4 $J/cmd_multi.sh matmul_random_tanh 2 32 1 16
TREE=/proj/work/spramanik/SIENNA $J/snap_launch_tree.sh i0_fpref 32 6 $J/cmds/int8_fpref.sh
```

Expected: `i0_sienna_*` 29/29 each, per-test cycles identical to `fx_suite_fp32` / `fx_suite_bf16` (compare with
`$J/cycles.sh` on both `results/pipeline` directories); `i0_reg32_*` 29/29 each; `i0_multi_fp32` `RESULT: PASSED`;
`i0_fpref`: five `RESULT: PASSED` lines in `runs/i0_fpref/stdout.log` and no `RUN-FAIL`.

- [ ] **Step 6: Record the fp32 / bf16 reference results every gate compares against**

The reference runs, all in `$J/runs/`: AriL, the bf16 G1 gate run `g1b` (one snapshot with every AriL log the int8
G1 gate reruns); GPNAE, `i0_fpref` (Step 5) and the bf16 gate's `g2_*`; mesh, the bf16 branch's G3 runs with the
prefixes `g3_`, `g3m_`, `g3n_`, `g3x_`, `g3y_`, `g3z_`; SIENNA, `fx_suite_*`, `g4r_N32_*`, `g4ri_*`, perf `g4p_*`, GEMM
`s22_gemm_*`, models `ms_N16_T4_*`, lint `g4lint2`, and Step 5's `i0_sienna_*`, `i0_reg32_*`, `i0_multi_fp32`. Write
their names and per-test cycles into `testbenches/results/int8/reference_fp32_bf16.log` in the int8 tree (one line per
test: run, test, cycles, pass/fail), with `$J/cycles.sh`. For the mesh runs also record which finished (`exit=0` in
`$J/runs/<name>.out`) and which points the bf16 branch could not build in the 256 GB node (collapse-k 0 at N = 64:
fp32 T = 2 and T = 4, bf16 T = 2 at the time of writing); Task 15 takes its `NOT_BUILDABLE` list from this record.

- [ ] **Step 7: Report Level 0 setup**

Report to Soham: the int8 tree and branches (commits), TensorFlow version, the reference log path. No code changes in this task.

### Task 1: `ipu.py`, bit-exact models of the integer units and TFLite's requantize

**Files:**
- Create: `Common/models/ipu.py`, `Common/models/test_ipu.py`
- Modify: `Common/Makefile` (add target `ipu`)

**Interfaces:**
- Consumes (Task 0): the `SIENNA_int8` clone; `$J/cmds/int8_aril.sh UNIT TARGET [PLUSARGS]`, which runs
  `make TARGET PLUSARGS=...` in `SystolicMesh/ArithmeticLibrary/UNIT` of a snapshot, tees the output to
  `testbenches/results/int8/<UNIT with / as _>_<TARGET>[_<TAG>].log`, and fails when the log holds `RESULT: FAILED` or,
  for a target other than `lint*`, no `RESULT: PASSED`.
- Produces (all vectorized over int64 numpy arrays, numpy broadcasting; operands are signed values or raw bit patterns,
  the low `w` bits read as two's complement; results are signed values in int64 arrays):
  - `ipu.INT32_MIN`, `ipu.INT32_MAX`, `ipu.ROUNDINGS = ("SINGLE", "DOUBLE")`
  - `ipu.sx(x, w)`: the low `w` bits of `x` as a signed value; `ipu.bits(x, w)`: the low `w` bits as an unsigned pattern
  - `ipu.int_mul(a, b, w=8)`: full signed product (intMultiplier)
  - `ipu.int_add(a, b, w=32)`: `a + b mod 2^w`, signed (intAdder)
  - `ipu.fx_mac(a, x, c, w=16, frac=11)`: `sat_w(((a * x) >> frac) + c)`, floor shift (D-1) (fxMac)
  - `ipu.srdhm(a, b)`: gemmlowp `SaturatingRoundingDoublingHighMul`
  - `ipu.rdbpot(x, exp)`: gemmlowp `RoundingDivideByPOT`, `exp` in [0, 31]
  - `ipu.mbqm(acc, mult, shift, rounding)`: TFLite `MultiplyByQuantizedMultiplier`, `rounding` "DOUBLE" (default TFLite
    build) or "SINGLE" (`TFLITE_SINGLE_ROUNDING`), `shift` in [-31, 30]
  - `ipu.requant(acc, mult, shift, zp, amin, amax, rounding)`: `min(max(wrap32(mbqm + zp), amin), amax)`; `zp`, `amin`,
    `amax` are 8-bit (tfliteRequant)

The C sources these transcribe, so a reviewer can check them line by line:

```cpp
// gemmlowp/fixedpoint/fixedpoint.h
std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t ab_64 = std::int64_t(a) * std::int64_t(b);
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));  // '/' truncates toward zero
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}
IntegerType RoundingDivideByPOT(IntegerType x, int exponent) {  // 0 <= exponent <= 31
  const IntegerType mask = (1ll << exponent) - 1;
  const IntegerType remainder = x & mask;
  const IntegerType threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}
// tensorflow/lite/kernels/internal/common.cc
#if TFLITE_SINGLE_ROUNDING
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
  TFLITE_DCHECK(quantized_multiplier >= 0);
  TFLITE_DCHECK(shift >= -31 && shift <= 30);
  const int64_t total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
  result = result >> total_shift;
  return static_cast<int32_t>(result);
}
#else
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}
#endif
// reference_integer_ops conv.h / fully_connected.h epilogue (acc is int32_t)
acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[c], output_shift[c]);
acc += output_offset;
acc = std::max(acc, output_activation_min);
acc = std::min(acc, output_activation_max);
```

Two consequences the models and the RTL carry. SRDHM rounds ties toward +inf, not away from zero: for a negative
product the nudge `1 - 2^30` and the truncating divide give `floor((a*b + 2^30) / 2^31)`, the same expression as for a
positive one (`-0.5` LSB gives 0, `-1.5` gives -1). RoundingDivideByPOT rounds ties away from zero. So DOUBLE and SINGLE
differ both on double-rounding cases (1.25 gives 2 against 1) and on negative ties (-1.5 gives -2 against -1). The
int32 overflows of `x * (1 << left_shift)` and `acc += output_offset` are undefined in C++; the models wrap them mod
2^32, as the compiled kernels do on x86 (a plan ruling: saturating would not match TFLite). The shift domain is
TFLite's [-31, 30] (single rounding's DCHECK; QuantizeMultiplier yields at most 30), and the tests cover all of it.

- [ ] **Step 1: Write the failing test**

`Common/models/test_ipu.py`:

```python
#!/usr/bin/env python3
"""Checks ipu.py: hand-derived TFLite values, then the vectorized models against scalar big-integer transcriptions of the C."""
import os
import random
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipu  # noqa: E402

MIN, MAX = -(1 << 31), (1 << 31) - 1
H = 1 << 30  # 0.5 in Q0.31
fails = checks = 0


def check(what, got, want):
    global fails, checks
    checks += 1
    g, w = np.asarray(got), np.asarray(want)
    if g.shape != w.shape or np.any(g != w):
        fails += 1
        if fails <= 30:
            print(f"[FAIL] {what}: got {g.ravel()[:8].tolist()}, want {w.ravel()[:8].tolist()}")


# Hand-derived values; each comment is the derivation.
SRDHM = [  # a, b, SaturatingRoundingDoublingHighMul(a, b)
    (MIN, MIN, MAX),  # (-1) * (-1) = +1 does not fit Q0.31: the one saturating case
    (H, H, 1 << 29),  # 0.5 * 0.5 = 0.25
    (MAX, MAX, MAX - 1),  # (2^31 - 1)^2 / 2^31 = 2^31 - 2 + 2^-31, + 0.5 truncates to 2^31 - 2
    (MIN, MAX, MIN + 1),  # -(2^31 - 1) exactly; nudge 1 - 2^30 and truncation leave it
    (MIN, H, -H),  # -1 * 0.5
    (1, H, 1),  # +0.5 LSB rounds up
    (-1, H, 0),  # -0.5 LSB: (-2^30 + 1 - 2^30) / 2^31 truncates to 0, a tie toward +inf
    (3, H, 2),  # 1.5 -> 2
    (-3, H, -1),  # -1.5 -> -1 (toward +inf)
    (5, MAX, 5),  # 4.9999999977 -> 5
    (-5, MAX, -5),  # -4.9999999977 -> -5
]
RDBPOT = [  # x, exponent, RoundingDivideByPOT(x, exponent): nearest, ties away from zero
    (5, 1, 3),  # 2.5 -> 3
    (-5, 1, -3),  # -2.5 -> -3: remainder 1 is not above threshold 0 + 1
    (-3, 1, -2),  # -1.5 -> -2
    (6, 2, 2),  # 1.5 -> 2
    (-6, 2, -2),  # -1.5 -> -2: remainder 2, threshold 1 + 1
    (-7, 2, -2),  # -1.75 -> -2
    (-5, 2, -1),  # -1.25 -> -1: remainder 3 above threshold 2
    (-7, 0, -7),  # exponent 0: unchanged
    (MIN, 31, -1),  # exactly -1
    (MAX, 31, 1),  # 0.99999999953 -> 1
    (H, 31, 1),  # +0.5 -> 1
    (-H, 31, -1),  # -0.5 -> -1
    (-H + 1, 31, 0),  # -0.49999999953 -> 0
]
MBQM = [  # acc, mult, shift, DOUBLE, SINGLE
    (100, H, 0, 50, 50),  # 100 * 0.5
    (3, H, -1, 1, 1),  # 0.75
    (5, H, -1, 2, 1),  # 1.25: DOUBLE rounds 2.5 up to 3, then 1.5 away to 2; SINGLE rounds 1.25 once
    (6, H, -1, 2, 2),  # +1.5 tie: both up
    (-6, H, -1, -2, -1),  # -1.5 tie: DOUBLE away from zero, SINGLE toward +inf
    (-3, H, -1, -1, -1),  # -0.75
    (-1, H, 0, 0, 0),  # -0.5: both toward +inf (SRDHM, and SINGLE's floor)
    (1000, H, -3, 63, 63),  # +62.5
    (-1000, H, -3, -63, -62),  # -62.5
    (MAX, MAX, -31, 1, 1),  # just under 1.0
    (MIN, MAX, -31, -1, -1),  # just above -1.0
    (MIN, MAX, 0, MIN + 1, MIN + 1),
    (MAX, MAX, 0, MAX - 1, MAX - 1),
    (100, H, 8, 12800, 12800),  # 100 * 2^8 * 0.5
    (-(1 << 23), H, 8, -H, -H),  # acc << 8 is exactly INT32_MIN
    (1 << 24, H, 8, 0, MIN),  # outside TFLite's range: DOUBLE's int32 acc << 8 wraps to 0, SINGLE's 2^31 wraps to INT32_MIN
    (1, H, 30, 1 << 29, 1 << 29),  # the largest shift
    (1 << 30, H, -30, 1, 1),  # +0.5: away (DOUBLE) and up (SINGLE) agree
    (1 << 30, H, -31, 0, 0),  # 0.25
]
MBQM += [((1 << (1 - s)) if s <= 0 else 1, H, s, 1 if s <= 0 else 1 << (s - 1), 1 if s <= 0 else 1 << (s - 1))
         for s in range(-29, 9)]  # exact powers of two at every shift -29..8
MBQM += [(-(3 << (-s)), H, s, -2 if s < 0 else -1, -1)
         for s in range(-29, 1)]  # -1.5 at every right shift: DOUBLE -2 (s = 0: SRDHM already gave -1), SINGLE -1
REQUANT = [  # acc, mult, shift, zp, act_min, act_max, DOUBLE, SINGLE
    (1000, H, -3, -5, -128, 127, 58, 58),  # 62.5 -> 63, + zp -5
    (-1000, H, -3, -5, -5, 127, -5, -5),  # ReLU: -68 / -67 clamp at the zero point
    (10 ** 6, MAX, 0, 0, -128, 127, 127, 127),
    (MAX, MAX, 0, 127, -128, 127, -128, -128),  # 2^31 - 2 + 127 wraps in int32, as TFLite's acc += output_offset does
    (-6, H, -1, 0, -128, 127, -2, -1),
    (5, H, -1, 0, -128, 127, 2, 1),
    (0, H, 0, 0, 5, -5, -5, -5),  # act_min > act_max: max then min gives act_max
    (1 << 24, H, 8, 0, -128, 127, 0, -128),
    (0x80, H, 0, 0x80, 0x80, 0x7F, -64, -64),  # 8-bit patterns: zp and act_min read as -128; 64 - 128
]
INT_MUL = [(-128, -128, 16384), (-128, 127, -16256), (127, 127, 16129), (0x80, 0x80, 16384), (0xFF, 1, -1), (0, -128, 0)]
INT_ADD = [(MAX, 1, MIN), (MIN, -1, MAX), (-5, 3, -2), (0xFFFFFFFF, 1, 0), (MIN, MIN, 0)]
FX_MAC = [  # a, x, c, floor((a * x) / 2^11) + c saturated to int16 (Q4.11, 1.0 = 2048)
    (2048, 2048, 0, 2048), (-1, 1, 0, -1), (1, 1, 0, 0), (-2048, 3, 0, -3), (-2049, 1, 0, -2), (1024, 1024, -100, 412),
    (32767, 32767, 0, 32767), (-32768, 32767, 0, -32768), (-32768, -32768, 0, 32767), (2048, 2048, 32767, 32767),
    (-2048, 2048, -32768, -32768),
]


def wrap32(v):
    return ((v + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def c_srdhm(a, b):  # fixedpoint.h, line for line
    overflow = a == b == MIN
    ab = a * b
    n = ab + ((1 << 30) if ab >= 0 else 1 - (1 << 30))
    q = (n >> 31) if n >= 0 else -((-n) >> 31)  # C++ '/' truncates toward zero
    return MAX if overflow else q


def c_rdbpot(x, e):  # fixedpoint.h RoundingDivideByPOT
    mask = (1 << e) - 1
    return (x >> e) + (1 if (x & mask) > (mask >> 1) + (1 if x < 0 else 0) else 0)


def c_mbqm(x, m, s, rounding):  # common.cc, both builds
    if rounding == "DOUBLE":
        return c_rdbpot(c_srdhm(wrap32(x * (1 << max(s, 0))), m), max(-s, 0))
    t = 31 - s
    return wrap32((x * m + (1 << (t - 1))) >> t)


def c_requant(x, m, s, zp, lo, hi, rounding):
    return min(max(wrap32(c_mbqm(x, m, s, rounding) + zp), lo), hi)


def half_away(num, den):  # exact nearest of num / den, ties away from zero
    q, r = divmod(abs(num), den)
    q += 2 * r >= den
    return q if num >= 0 else -q


def main() -> None:
    for a, b, w in SRDHM:
        check(f"srdhm({a}, {b})", ipu.srdhm(a, b), w)
    for x, e, w in RDBPOT:
        check(f"rdbpot({x}, {e})", ipu.rdbpot(x, e), w)
    for x, m, s, d, sg in MBQM:
        check(f"mbqm({x}, {m}, {s}, DOUBLE)", ipu.mbqm(x, m, s, "DOUBLE"), d)
        check(f"mbqm({x}, {m}, {s}, SINGLE)", ipu.mbqm(x, m, s, "SINGLE"), sg)
    for x, m, s, z, lo, hi, d, sg in REQUANT:
        check(f"requant({x}, {m}, {s}, {z}, {lo}, {hi}, DOUBLE)", ipu.requant(x, m, s, z, lo, hi, "DOUBLE"), d)
        check(f"requant({x}, {m}, {s}, {z}, {lo}, {hi}, SINGLE)", ipu.requant(x, m, s, z, lo, hi, "SINGLE"), sg)
    for a, b, w in INT_MUL:
        check(f"int_mul({a}, {b})", ipu.int_mul(a, b), w)
    for a, b, w in INT_ADD:
        check(f"int_add({a}, {b})", ipu.int_add(a, b), w)
    for a, x, c, w in FX_MAC:
        check(f"fx_mac({a}, {x}, {c})", ipu.fx_mac(a, x, c), w)

    # Exhaustive int8 products, as values and as 8-bit patterns.
    g = np.arange(-128, 128, dtype=np.int64)
    ga, gb = np.meshgrid(g, g)
    check("int_mul exhaustive", ipu.int_mul(ga, gb), ga * gb)
    check("int_mul exhaustive, bit patterns", ipu.int_mul(ga & 0xFF, gb & 0xFF), ga * gb)

    # Random: the vectorized models against the scalar transcriptions; every shift appears.
    rng = random.Random(7)
    edge = [MIN, MIN + 1, -H, -1, 0, 1, H, MAX - 1, MAX]
    n = 20000

    def r32():
        return rng.choice(edge) if rng.random() < 0.1 else rng.randint(MIN, MAX)

    A = [r32() for _ in range(n)]
    B = [r32() for _ in range(n)]
    S = list(range(-31, 31)) + [rng.randint(-31, 30) for _ in range(n - 62)]
    E = list(range(32)) + [rng.randint(0, 31) for _ in range(n - 32)]
    Z = [rng.randint(-128, 127) for _ in range(n)]
    LO = [rng.randint(-128, 127) for _ in range(n)]
    HI = [rng.randint(-128, 127) for _ in range(n)]
    npa = [np.array(v, dtype=np.int64) for v in (A, B, S, E, Z, LO, HI)]
    na, nb, ns, ne, nz, nlo, nhi = npa
    check("srdhm random", ipu.srdhm(na, nb), [c_srdhm(a, b) for a, b in zip(A, B)])
    check("srdhm = floor((a*b + 2^30) / 2^31), the RTL's form",
          [c_srdhm(a, b) for a, b in zip(A, B)], [MAX if a == b == MIN else (a * b + H) >> 31 for a, b in zip(A, B)])
    check("rdbpot random", ipu.rdbpot(na, ne), [c_rdbpot(x, e) for x, e in zip(A, E)])
    check("rdbpot = nearest, ties away", [c_rdbpot(x, e) for x, e in zip(A, E)], [half_away(x, 1 << e) for x, e in zip(A, E)])
    for r in ipu.ROUNDINGS:
        check(f"mbqm random {r}", ipu.mbqm(na, nb, ns, r), [c_mbqm(x, m, s, r) for x, m, s in zip(A, B, S)])
        check(f"requant random {r}", ipu.requant(na, nb, ns, nz, nlo, nhi, r),
              [c_requant(*t, r) for t in zip(A, B, S, Z, LO, HI)])
    # Against exact arithmetic where TFLite defines the result: SINGLE within 1/2, DOUBLE within 1.
    bad = {"DOUBLE": 0, "SINGLE": 0}
    for x, m, s in zip(A, B, S):
        xs = x * (1 << max(s, 0))
        if m < 0 or xs > MAX or xs < MIN:
            continue
        exact = Fraction(x * m, 1 << 31) * Fraction(2) ** s
        if abs(exact) >= MAX:
            continue
        bad["SINGLE"] += abs(Fraction(c_mbqm(x, m, s, "SINGLE")) - exact) > Fraction(1, 2)
        bad["DOUBLE"] += abs(Fraction(c_mbqm(x, m, s, "DOUBLE")) - exact) > 1
    check("SINGLE within 1/2 of exact", bad["SINGLE"], 0)
    check("DOUBLE within 1 of exact", bad["DOUBLE"], 0)
    # Broadcasting as the goldens use it: a (rows, channels) accumulator against per-channel multipliers and shifts.
    acc = na[:80].reshape(5, 16)
    mult, shift = (np.abs(nb[:16]) & (H - 1)) | H, ns[:16]  # normalized, as QuantizeMultiplier gives
    for r in ipu.ROUNDINGS:
        check(f"requant broadcast {r}", ipu.requant(acc, mult, shift, 3, -128, 127, r),
              [[c_requant(int(acc[i, c]), int(mult[c]), int(shift[c]), 3, -128, 127, r) for c in range(16)] for i in range(5)])
    X = [rng.randint(-32768, 32767) for _ in range(n)]
    C = [rng.randint(-32768, 32767) for _ in range(n)]
    Y = [rng.randint(-32768, 32767) for _ in range(n)]
    check("fx_mac random", ipu.fx_mac(np.array(X), np.array(Y), np.array(C)),
          [max(-32768, min(32767, ((x * y) >> 11) + c)) for x, y, c in zip(X, Y, C)])
    check("int_add random", ipu.int_add(na, nb), [wrap32(a + b) for a, b in zip(A, B)])

    print(f"test_ipu: {checks} checks, {fails} failures")
    print(f"RESULT: {'PASSED' if fails == 0 else 'FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
```

`Common/Makefile` (whole file; the `ipu` target and the header comment are new):

```make
SHELL := /bin/bash
# Common AriL checks: the format package, the fp32 units dumped for the Python model, and the integer model ipu.py.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/..
TB_DIR  = $(PRJ_DIR)/testbenches
FLAGS   = --binary --timing --assert --sv -I$(TB_DIR) --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
PLUSARGS ?=
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent

pkg:
	verilator $(FLAGS) --top-module TB_sienna_fmt_pkg --Mdir $(PRJ_DIR)/Verilator/pkg $(PRJ_DIR)/src/sienna_fmt_pkg.sv $(TB_DIR)/TB_sienna_fmt_pkg.sv -o sim
	$(PRJ_DIR)/Verilator/pkg/sim

FP32_UNITS = $(ARIL)/Multipliers/Radix4Booth/src/R4Booth.sv $(ARIL)/Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
             $(ARIL)/Multipliers/FP32/src/fp32Multiplier.sv $(ARIL)/Adders/FP32/src/LZC.sv $(ARIL)/Adders/FP32/src/fp32Adder.sv

dump32:
	verilator $(FLAGS) --top-module TB_fp32Dump --Mdir $(PRJ_DIR)/Verilator/dump32 $(FP32_UNITS) $(TB_DIR)/TB_fp32Dump.sv -o sim
	cd $(PRJ_DIR)/Verilator/dump32 && ./sim $(PLUSARGS)
	python3 $(PRJ_DIR)/models/check_fpu.py $(PRJ_DIR)/Verilator/dump32/mul32.txt --unit mul --format fp32
	python3 $(PRJ_DIR)/models/check_fpu.py $(PRJ_DIR)/Verilator/dump32/add32.txt --unit add --format fp32 --fp32-unit

ipu:
	python3 $(PRJ_DIR)/models/test_ipu.py

.PHONY: pkg dump32 ipu
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch_tree.sh i1_ipu 32 1 $J/cmds/int8_aril.sh Common ipu`
Expected: `$J/runs/i1_ipu/stdout.log` ends with `ModuleNotFoundError: No module named 'ipu'` and a make error for
target `ipu`; `$J/runs/i1_ipu.out` shows `exit=1` (the make error, and no verdict printed).

- [ ] **Step 3: Write the model**

`Common/models/ipu.py`:

```python
"""Bit-exact models of AriL's integer units (intMultiplier, intAdder, fxMac, tfliteRequant) and TFLite's int8 requantize,
vectorized over int64 numpy arrays with broadcasting; no TensorFlow. Operands are signed values or raw bit patterns (the low
w bits read as two's complement); results are signed values. int32 intermediates wrap mod 2^32 as TFLite's compiled C does."""
import numpy as np

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
ROUNDINGS = ("SINGLE", "DOUBLE")


def sx(x, w):
    """The low w bits of x as a signed w-bit value (1 <= w <= 62)."""
    x = np.asarray(x, dtype=np.int64)
    h = np.int64(1 << (w - 1))
    return ((x & np.int64((1 << w) - 1)) ^ h) - h


def bits(x, w):
    """x as an unsigned w-bit pattern, for vector files and dumps."""
    return np.asarray(x, dtype=np.int64) & np.int64((1 << w) - 1)


def int_mul(a, b, w=8):
    """intMultiplier: the full signed 2w-bit product."""
    return sx(a, w) * sx(b, w)


def int_add(a, b, w=32):
    """intAdder: a + b mod 2^w, signed."""
    return sx(sx(a, w) + sx(b, w), w)


def fx_mac(a, x, c, w=16, frac=11):
    """fxMac, one Horner step: the 2w-bit product shifted right by frac with floor (D-1), plus c, saturated to w bits."""
    p = (sx(a, w) * sx(x, w)) >> frac
    return np.clip(p + sx(c, w), -(1 << (w - 1)), (1 << (w - 1)) - 1)


def srdhm(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul: (a*b + nudge) / 2^31 with C's truncating divide, nudge 2^30 for a
    non-negative product and 1 - 2^30 otherwise; INT32_MIN * INT32_MIN saturates to INT32_MAX."""
    a, b = sx(a, 32), sx(b, 32)
    ab = a * b
    n = ab + np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    q = np.where(n >= 0, n >> 31, -((-n) >> 31))
    return np.where((a == INT32_MIN) & (b == INT32_MIN), np.int64(INT32_MAX), q)


def rdbpot(x, exp):
    """gemmlowp RoundingDivideByPOT: x / 2^exp to nearest, ties away from zero; exp in [0, 31]."""
    x = sx(x, 32)
    e = np.asarray(exp, dtype=np.int64)
    assert np.all((e >= 0) & (e <= 31)), "RoundingDivideByPOT needs 0 <= exp <= 31"
    mask = (np.int64(1) << e) - 1
    threshold = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> e) + ((x & mask) > threshold).astype(np.int64)


def mbqm(acc, mult, shift, rounding):
    """TFLite MultiplyByQuantizedMultiplier; shift > 0 shifts left, < 0 right, in [-31, 30].
    DOUBLE: RoundingDivideByPOT(SRDHM(acc * 2^left wrapped to int32, mult), right). SINGLE: (acc*mult + 2^(30-shift)) >> (31-shift)."""
    acc, mult = sx(acc, 32), sx(mult, 32)
    shift = np.asarray(shift, dtype=np.int64)
    assert np.all((shift >= -31) & (shift <= 30)), "shift outside [-31, 30]"
    if rounding == "DOUBLE":
        left, right = np.maximum(shift, 0), np.maximum(-shift, 0)
        return rdbpot(srdhm(sx(acc << left, 32), mult), right)
    if rounding == "SINGLE":
        total = 31 - shift
        return sx((acc * mult + (np.int64(1) << (total - 1))) >> total, 32)
    raise ValueError(f"rounding must be SINGLE or DOUBLE, not {rounding!r}")


def requant(acc, mult, shift, zp, amin, amax, rounding):
    """tfliteRequant, TFLite's conv / FC epilogue: y = mbqm + zp in int32, then max(y, amin), then min(y, amax)."""
    y = sx(mbqm(acc, mult, shift, rounding) + sx(zp, 8), 32)
    return np.minimum(np.maximum(y, sx(amin, 8)), sx(amax, 8))
```

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch_tree.sh i1_ipu 32 1 $J/cmds/int8_aril.sh Common ipu`
Expected: `test_ipu: 254 checks, 0 failures` and `RESULT: PASSED` in
`$J/runs/i1_ipu/results/int8/Common_ipu.log`. A failing known value means the model or the derivation in the comment is
wrong: recompute that case by hand from the C above before touching either. A random mismatch against the scalar
transcription is a numpy issue (an int64 overflow or a broadcast), since the transcription is the C line for line.

- [ ] **Step 5: Commit (AriL, branch int8)**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh/ArithmeticLibrary
git add Common/models/ipu.py && git commit -m "ipu.py: bit-exact models of the integer units and TFLite's requantize, both roundings"
git add Common/models/test_ipu.py Common/Makefile && git commit -m "test_ipu.py: hand-derived TFLite values and scalar transcriptions of the C"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
```

Leave the untracked `Common/models/__pycache__/` alone.

### Task 2: `tflite_ref.py`, `tflite_oracle.py`, gate G0

**Files:**
- Create (SIENNA): `tflite_ref.py`, `test_tflite_ref.py`, `tflite_oracle.py`
- Create (SIENNA, generated by G0): `testbenches/tflite_int8/{fc64x16_linear, fc64x16_relu, conv3x3_8x8x16_linear,
  conv3x3_8x8x16_relu6}.{tflite, npz}`, `testbenches/tflite_int8/rounding.txt`
- Create (untracked report): `testbenches/results/int8/g0_oracle.log`
- Modify: `SystolicMesh` (ArithmeticLibrary pointer), SIENNA (SystolicMesh pointer)

**Interfaces:**
- Consumes: `ipu.sx`, `ipu.requant`, `ipu.ROUNDINGS` (Task 1); TensorFlow in `$J/venv` (Task 0: `$J/venv/bin/python3 -c
  "import tensorflow"` works on a farm node).
- Produces:
  - `tflite_ref.ROUNDING`: the pinned variant, `"SINGLE"` or `"DOUBLE"`, read from `testbenches/tflite_int8/rounding.txt`
    (None until G0 has written it)
  - `tflite_ref.round_half_away(v) -> int` (TfLiteRound on a double)
  - `tflite_ref.quantize_multiplier(real, rounding="DOUBLE") -> (mult, shift)` (QuantizeMultiplier; SINGLE caps shift at 30)
  - `tflite_ref.effective_scale(in_scale, w_scale, out_scale, product) -> float`, `product` "double" or "float32"
  - `tflite_ref.layer_multipliers(layer, w_scales, in_scale, out_scale, cout, rounding, scale_product=None) -> (mults, shifts)`
    int64 arrays of length `cout`; `layer` "fc" or "conv"
  - `tflite_ref.activation_range(activation, out_scale, out_zp) -> (amin, amax)`, `activation` "none", "relu", "relu6"
  - `tflite_ref.fold_input_zp(b_q, w_q, in_zp) -> int64 array`: `wrap32(b - in_zp * sum(w over each output channel))`
  - `tflite_ref.im2col_same(x_q, kh, kw, pad_value) -> (B*H*W, kh*kw*Cin)`: NHWC stride-1 SAME patches, columns in
    (kh, kw, cin) order, so `conv = im2col @ w.reshape(cout, -1).T`
  - `tflite_ref.fc_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding, folded=False,
    scale_product=None) -> int8 (B, Cout)`; `w_q` is TFLite's [out, in]
  - `tflite_ref.conv2d_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding,
    folded=False) -> int8 (B, H, W, Cout)`; `x_q` NHWC, `w_q` TFLite's OHWI, stride 1, SAME
  - `folded=True` computes SIENNA's algebra (no zero point in the matmul, `fold_input_zp` bias, conv padding with
    `in_zp` in `im2col_same`), `folded=False` TFLite's reference-kernel algebra; G0 proves they agree.
  - `testbenches/tflite_int8/<name>.npz` keys: `layer` ("fc"/"conv"), `activation`, `in_shape`, `w_q` (int8, TFLite
    layout), `b_q` (int32), `w_scales` (float32), `in_scale`, `in_zp`, `out_scale`, `out_zp`, `act_min`, `act_max`,
    `mults`, `shifts` (int64 per output channel, for the pinned rounding), `x_test` (int8, 64 inputs), `y_test` (int8,
    the BUILTIN_REF interpreter's outputs for them), `rounding`, `tf_version`.
  - `testbenches/tflite_int8/rounding.txt`: one line, `SINGLE` or `DOUBLE`. Task 3 copies it into
    `sienna_fmt_pkg::REQ_ROUNDING`, which every `tfliteRequant` instance takes; `ipu.REQ_ROUNDING` (Task 3) and
    `tflite_ref.ROUNDING` read this file. It lives outside `results/`, so every snapshot carries it.

What `tflite_ref` transcribes, beyond Task 1's epilogue (TFLite `quantization_util.cc`, `kernel_util.cc`, `conv.cc`,
`fully_connected.cc`, reference kernels):

```cpp
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
  if (double_multiplier == 0.) { *quantized_multiplier = 0; *shift = 0; return; }
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1LL << 31)));   // std::round: ties away from zero
  if (q_fixed == (1LL << 31)) { q_fixed /= 2; ++*shift; }
  if (*shift < -31) { *shift = 0; q_fixed = 0; }
#if TFLITE_SINGLE_ROUNDING
  if (*shift > 30) { *shift = 30; q_fixed = (1LL << 31) - 1; }
#endif
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}
// per channel (conv always, FC with per-channel weights): double(input scale) * double(filter scale[c]) / double(output scale)
// per tensor FC, GetQuantizedConvolutionMultipler: static_cast<double>(input->params.scale * filter->params.scale) / double(output scale)
// ReLU / ReLU6 bounds: zero_point + static_cast<int32_t>(TfLiteRound(f / scale)) with f and scale float; clamped to [-128, 127]
// conv / FC accumulate: acc += filter_val * (input_val + input_offset), input_offset = -input zero point; out-of-image taps skipped
```

The per-tensor FC product in float32 is from reading the source, not verified here; G0 checks it: the oracle prints a
diagnostic line when the per-tensor FC models mismatch in both variants. If the installed converter quantizes Dense
layers per channel, the report shows it and the per-tensor path is never exercised.

- [ ] **Step 1: Write the failing test**

`test_tflite_ref.py`:

```python
#!/usr/bin/env python3
"""Known values for tflite_ref.py (numpy only): QuantizeMultiplier, activation ranges, a hand-computed FC and 3x3 conv, the fold."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_ref as ref  # noqa: E402

fails = checks = 0


def check(what, got, want):
    global fails, checks
    checks += 1
    g, w = np.asarray(got), np.asarray(want)
    if g.shape != w.shape or np.any(g != w):
        fails += 1
        print(f"[FAIL] {what}: got {g.tolist()}, want {w.tolist()}")


QM = [  # real -> (mult, shift): frexp, then the mantissa * 2^31 rounded half away from zero
    (0.5, (1 << 30, 0)),
    (1.0, (1 << 30, 1)),
    (0.75, (1610612736, 0)),
    (0.1, (1717986918, -3)),  # 0.8 * 2^31 = 1717986918.4
    (2.0 ** -32, (1 << 30, -31)),  # shift -31 is kept
    (2.0 ** -33, (0, 0)),  # shift -32 flushes to zero
    (0.0, (0, 0)),
    (1.0 - 2.0 ** -40, (1 << 30, 1)),  # the mantissa rounds to 2^31: halved, shift + 1
    (1.0 / 255.0, (1077952576, -7)),
    (0.5 + 2.0 ** -32, ((1 << 30) + 1, 0)),  # 2^30 + 0.5 rounds away from zero; numpy.round would give 2^30
]


def main() -> None:
    for real, want in QM:
        check(f"quantize_multiplier({real!r})", ref.quantize_multiplier(real), want)
    check("quantize_multiplier(2^31, SINGLE)", ref.quantize_multiplier(2.0 ** 31, "SINGLE"), ((1 << 31) - 1, 30))
    check("quantize_multiplier(2^31, DOUBLE)", ref.quantize_multiplier(2.0 ** 31, "DOUBLE"), (1 << 30, 32))
    check("relu6 range", ref.activation_range("relu6", 0.05, -128), (-128, -8))  # 6 / 0.05 = 120 levels
    check("relu range", ref.activation_range("relu", 0.1, 3), (3, 127))
    check("relu6 above int8", ref.activation_range("relu6", 0.02, -10), (-10, 127))  # 300 levels: capped
    check("none range", ref.activation_range("none", 0.1, 3), (-128, 127))

    # FC: acc = 3 * (10 + 2) - 4 * (-20 + 2) + b = 108 + b; scale 0.5 * 0.25 / 1.0 = 2^-3 (QM (2^30, -2)); out zp 3.
    x, w = np.array([[10, -20]]), np.array([[3, -4]])
    fc_cases = [(100, 29, 29),  # 208 / 8 = 26
                (104, 30, 30),  # 212 / 8 = 26.5: a positive tie, both up
                (-176, -6, -5)]  # -68 / 8 = -8.5: DOUBLE away (-9), SINGLE up (-8)
    for b, dbl, sgl in fc_cases:
        for r, want in (("DOUBLE", dbl), ("SINGLE", sgl)):
            for folded in (False, True):
                check(f"fc b={b} {r} folded={folded}",
                      ref.fc_int8(x, w, np.array([b]), -2, [0.25], 0.5, 1.0, 3, -128, 127, r, folded=folded), [[want]])

    # Conv 3x3 SAME on a 2x2 image: in-image taps of (x - 1) = [[0, 1], [2, 3]] give [[49, 43], [31, 25]];
    # scale 0.5 * 1 / 1 (QM (2^30, 0)) makes every one a positive tie, rounded up in both; out zp -3.
    xc = np.array([1, 2, 3, 4]).reshape(1, 2, 2, 1)
    wc = np.arange(1, 10).reshape(1, 3, 3, 1)
    for r in ("DOUBLE", "SINGLE"):
        for folded in (False, True):
            check(f"conv {r} folded={folded}",
                  ref.conv2d_int8(xc, wc, np.array([0]), 1, [1.0], 0.5, 1.0, -3, -128, 127, r, folded=folded),
                  np.array([22, 19, 13, 10]).reshape(1, 2, 2, 1))
    check("fold_input_zp", ref.fold_input_zp(np.array([0]), np.ones((1, 3, 3, 1)), 1), [-9])
    check("im2col_same corner", ref.im2col_same(xc, 3, 3, 1)[0], [1, 1, 1, 1, 1, 2, 1, 3, 4])  # pad value 1 outside

    print(f"test_tflite_ref: {checks} checks, {fails} failures")
    print(f"RESULT: {'PASSED' if fails == 0 else 'FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch_tree.sh i2_ref 32 1 python3 test_tflite_ref.py`
Expected: `ModuleNotFoundError: No module named 'tflite_ref'` in `$J/runs/i2_ref/stdout.log`.

- [ ] **Step 3: Write the reference**

`tflite_ref.py`:

```python
"""numpy-only int8 FC and conv2d as TFLite's reference kernels compute them, on ipu.requant; tflite_oracle.py checks them
against the TFLite interpreter bit for bit. Layouts are TFLite's: FC weights [out, in], conv weights OHWI, data NHWC."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402

ROUNDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbenches", "tflite_int8", "rounding.txt")
ROUNDING = open(ROUNDING_FILE).read().strip() if os.path.exists(ROUNDING_FILE) else None  # G0's pinned variant; None before G0


def round_half_away(v: float) -> int:
    """TfLiteRound (std::round) on a double: nearest, ties away from zero; exact for |v| < 2^52."""
    r = math.floor(abs(v) + 0.5)
    return -r if v < 0 else r


def quantize_multiplier(real: float, rounding: str = "DOUBLE"):
    """TFLite QuantizeMultiplier: real = mult * 2^(shift - 31), mult in [2^30, 2^31); below 2^-32 flushes to (0, 0)."""
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    q_fixed = round_half_away(q * (1 << 31))
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if rounding == "SINGLE" and shift > 30:  # TFLITE_SINGLE_ROUNDING saturates the shift
        return (1 << 31) - 1, 30
    return q_fixed, shift


def effective_scale(in_scale, w_scale, out_scale, product: str) -> float:
    """The real multiplier of TFLite's Prepare: 'double' is the per-channel path, 'float32' the per-tensor FC path."""
    i, w, o = np.float32(in_scale), np.float32(w_scale), np.float32(out_scale)
    if product == "double":
        return float(i) * float(w) / float(o)
    if product == "float32":
        return float(i * w) / float(o)
    raise ValueError(product)


def layer_multipliers(layer, w_scales, in_scale, out_scale, cout, rounding, scale_product=None):
    """Per output channel (mult, shift); a single weight scale is per tensor and broadcast (conv still takes the double path)."""
    ws = np.atleast_1d(np.asarray(w_scales, dtype=np.float32))
    product = scale_product or ("double" if layer == "conv" or ws.size > 1 else "float32")
    pairs = [quantize_multiplier(effective_scale(in_scale, ws[c if ws.size > 1 else 0], out_scale, product), rounding)
             for c in range(cout)]
    return np.array([m for m, _ in pairs], dtype=np.int64), np.array([s for _, s in pairs], dtype=np.int64)


def activation_range(activation: str, out_scale, out_zp: int):
    """TFLite CalculateActivationRangeQuantized for int8: bounds quantized as zp + round(f / scale), f / scale in float32."""
    def quant(f):
        return int(out_zp) + round_half_away(float(np.float32(f) / np.float32(out_scale)))
    if activation == "none":
        return -128, 127
    if activation == "relu":
        return max(-128, quant(0.0)), 127
    if activation == "relu6":
        return max(-128, quant(0.0)), min(127, quant(6.0))
    raise ValueError(activation)


def fold_input_zp(b_q, w_q, in_zp):
    """SIENNA's bias: b - in_zp * sum(w) per output channel, wrapped to int32, so the mesh sees no zero point."""
    w = np.asarray(w_q, dtype=np.int64).reshape(np.shape(w_q)[0], -1)
    return ipu.sx(np.asarray(b_q, dtype=np.int64) - int(in_zp) * w.sum(axis=1), 32)


def im2col_same(x_q, kh, kw, pad_value):
    """NHWC patches of a stride-1 SAME convolution padded with pad_value; rows (b, y, x), columns (kh, kw, cin)."""
    x = np.asarray(x_q, dtype=np.int64)
    b, h, w, c = x.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    xp = np.pad(x, ((0, 0), (ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)), constant_values=pad_value)
    cols = [xp[:, dy:dy + h, dx:dx + w, :] for dy in range(kh) for dx in range(kw)]
    return np.stack(cols, axis=3).reshape(b * h * w, kh * kw * c)


def fc_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding, folded=False,
            scale_product=None):
    """reference_integer_ops::FullyConnected(PerChannel): acc = sum w * (x - in_zp) + b in int32, requantized per channel.
    folded=True: sum w * x + fold_input_zp(b), SIENNA's algebra."""
    x = np.asarray(x_q, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    b = np.zeros(w.shape[0], dtype=np.int64) if b_q is None else np.asarray(b_q, dtype=np.int64)
    acc = x @ w.T + fold_input_zp(b, w, in_zp) if folded else (x - int(in_zp)) @ w.T + b
    mult, shift = layer_multipliers("fc", w_scales, in_scale, out_scale, w.shape[0], rounding, scale_product)
    return ipu.requant(ipu.sx(acc, 32), mult, shift, out_zp, amin, amax, rounding).astype(np.int8)


def conv2d_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding, folded=False):
    """reference_integer_ops::ConvPerChannel, stride 1, SAME: out-of-image taps skipped, i.e. (x - in_zp) padded with 0.
    folded=True: im2col padded with in_zp, sum w * x + fold_input_zp(b), SIENNA's algebra."""
    x = np.asarray(x_q, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    bsz, h, wd, _ = x.shape
    cout, kh, kw, _ = w.shape
    b = np.zeros(cout, dtype=np.int64) if b_q is None else np.asarray(b_q, dtype=np.int64)
    wm = w.reshape(cout, -1)
    if folded:
        acc = im2col_same(x, kh, kw, int(in_zp)) @ wm.T + fold_input_zp(b, w, in_zp)
    else:
        acc = im2col_same(x - int(in_zp), kh, kw, 0) @ wm.T + b
    mult, shift = layer_multipliers("conv", w_scales, in_scale, out_scale, cout, rounding)
    acc = ipu.sx(acc, 32).reshape(bsz, h, wd, cout)
    return ipu.requant(acc, mult, shift, out_zp, amin, amax, rounding).astype(np.int8)
```

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch_tree.sh i2_ref 32 1 python3 test_tflite_ref.py`
Expected: `test_tflite_ref: 34 checks, 0 failures` and `RESULT: PASSED`.

- [ ] **Step 5: Write the oracle**

`tflite_oracle.py`:

```python
#!/usr/bin/env python3
"""Gate G0: single-layer int8 TFLite models (FC, Conv2D 3x3 SAME) from the TF converter, run on the interpreter's reference
kernels (BUILTIN_REF), compared bit for bit with tflite_ref in both roundings; pins the rounding TFLite uses and saves the
G4 test models with their quantization. Needs TensorFlow (sienna_jobs/venv); runs on the farm."""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_ref as ref  # noqa: E402

FC_IN, FC_OUT = 64, 16
CONV_HW, CONV_CIN, CONV_COUT = 8, 16, 16
MODELS = {  # name: (layer, fused activation); seed 0 of each is saved for G4
    "fc64x16_linear": ("fc", "none"),
    "fc64x16_relu": ("fc", "relu"),
    "conv3x3_8x8x16_linear": ("conv", "none"),
    "conv3x3_8x8x16_relu6": ("conv", "relu6"),
}
OP_NAME = {"fc": "FULLY_CONNECTED", "conv": "CONV_2D"}
ACT_FN = {"none": tf.identity, "relu": tf.nn.relu, "relu6": tf.nn.relu6}
ROUNDINGS = ("DOUBLE", "SINGLE")
REF = tf.lite.experimental.OpResolverType.BUILTIN_REF
MIN_DISCRIMINATING = 100  # outputs where the two roundings differ: the pick must rest on evidence
MIN_UNCLAMPED = 0.25  # per model: the comparison must not be dominated by saturated outputs
N_SAVED = 64  # test inputs and interpreter outputs kept per npz for G4


def build_model(layer, act, seed):
    """One layer: per-channel weight magnitudes over two decades; input range with max >= 1.5 |min|, so the zero point is not 0."""
    rng = np.random.default_rng(seed)
    lo = -rng.uniform(0.2, 2.0)
    hi = -lo * rng.uniform(1.5, 4.0)
    cout = FC_OUT if layer == "fc" else CONV_COUT
    sig = np.exp(rng.uniform(np.log(0.01), np.log(1.0), cout))
    if layer == "fc":
        in_shape = (1, FC_IN)
        w = rng.standard_normal((FC_IN, cout)) * sig
    else:
        in_shape = (1, CONV_HW, CONV_HW, CONV_CIN)
        w = rng.standard_normal((3, 3, CONV_CIN, cout)) * sig
    wc = tf.constant(w.astype(np.float32))
    bc = tf.constant((rng.standard_normal(cout) * 0.5).astype(np.float32))
    act_fn = ACT_FN[act]

    @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
    def layer_fn(x):
        y = tf.matmul(x, wc) if layer == "fc" else tf.nn.conv2d(x, wc, strides=1, padding="SAME")
        return act_fn(tf.nn.bias_add(y, bc))

    rep_rng = np.random.default_rng(seed + 1000)

    def representative():
        for _ in range(200):
            yield [rep_rng.uniform(lo, hi, in_shape).astype(np.float32)]

    holder = tf.Module()
    holder.layer_fn = layer_fn
    conv = tf.lite.TFLiteConverter.from_concrete_functions([layer_fn.get_concrete_function()], holder)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert(), lo, hi


def interpreter(model, resolver):
    it = tf.lite.Interpreter(model_content=model, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    return it


def invoke_all(it, xs):
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    ys = []
    for x in xs:
        it.set_tensor(i, x[None].astype(np.int8))
        it.invoke()
        ys.append(it.get_tensor(o)[0].astype(np.int64))
    return np.stack(ys)


def extract(it, layer, act):
    """The op's tensors and quantization; stops unless the graph is one int8 FULLY_CONNECTED or CONV_2D with fused activation."""
    ops = it._get_ops_details()  # private in TF 2.x: op_name, inputs, outputs per op
    names = [o["op_name"] for o in ops]
    if names != [OP_NAME[layer]]:
        sys.exit(f"G0: FAIL, expected one {OP_NAME[layer]}, the converter produced {names}")
    xi, wi, bi = (int(i) for i in ops[0]["inputs"][:3])
    oi = int(ops[0]["outputs"][0])
    td = {t["index"]: t for t in it.get_tensor_details()}
    qp = {i: td[i]["quantization_parameters"] for i in (xi, wi, bi, oi)}
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    if (inp["index"], out["index"]) != (xi, oi) or td[xi]["dtype"] != np.int8 or td[oi]["dtype"] != np.int8:
        sys.exit("G0: FAIL, the layer's input and output are not the graph's int8 input and output")
    if td[wi]["dtype"] != np.int8 or td[bi]["dtype"] != np.int32 or np.any(qp[wi]["zero_points"] != 0):
        sys.exit("G0: FAIL, weights are not symmetric int8 or the bias is not int32")
    p = {"layer": layer, "activation": act, "in_shape": tuple(int(d) for d in inp["shape"]),
         "w_q": it.get_tensor(wi).astype(np.int64), "b_q": it.get_tensor(bi).astype(np.int64),
         "w_scales": qp[wi]["scales"].astype(np.float32),
         "in_scale": np.float32(qp[xi]["scales"][0]), "in_zp": int(qp[xi]["zero_points"][0]),
         "out_scale": np.float32(qp[oi]["scales"][0]), "out_zp": int(qp[oi]["zero_points"][0])}
    p["amin"], p["amax"] = ref.activation_range(act, p["out_scale"], p["out_zp"])
    return p


def reference(p, xs, rounding, folded=False, scale_product=None):
    args = (xs, p["w_q"], p["b_q"], p["in_zp"], p["w_scales"], p["in_scale"], p["out_scale"], p["out_zp"], p["amin"],
            p["amax"], rounding)
    if p["layer"] == "fc":
        return ref.fc_int8(*args, folded=folded, scale_product=scale_product).astype(np.int64)
    return ref.conv2d_int8(*args, folded=folded).astype(np.int64)


def test_inputs(p, lo, hi, n, rng):
    """Mostly the calibration distribution quantized with the model's input parameters; a tenth uniform over all of int8."""
    shape = (n,) + p["in_shape"][1:]
    xq = np.clip(np.round(rng.uniform(lo, hi, shape) / p["in_scale"]) + p["in_zp"], -128, 127).astype(np.int64)
    k = n // 10
    xq[:k] = rng.integers(-128, 128, (k,) + shape[1:])
    return xq


def save(out, name, model, p, xs, ys, rounding):
    with open(os.path.join(out, name + ".tflite"), "wb") as f:
        f.write(model)
    mults, shifts = ref.layer_multipliers(p["layer"], p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], rounding)
    np.savez(os.path.join(out, name + ".npz"), layer=p["layer"], activation=p["activation"], in_shape=np.array(p["in_shape"]),
             w_q=p["w_q"].astype(np.int8), b_q=p["b_q"].astype(np.int32), w_scales=p["w_scales"], in_scale=p["in_scale"],
             in_zp=p["in_zp"], out_scale=p["out_scale"], out_zp=p["out_zp"], act_min=p["amin"], act_max=p["amax"],
             mults=mults, shifts=shifts, x_test=xs.astype(np.int8), y_test=ys.astype(np.int8), rounding=rounding,
             tf_version=tf.__version__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for the G4 models, npz files and rounding files")
    ap.add_argument("--report", required=True, help="G0 report (.log)")
    ap.add_argument("--seeds", type=int, default=8, help="models per kind; seed 0 is saved")
    ap.add_argument("--inputs", type=int, default=1000, help="random test inputs per model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    lines = [f"G0 TFLite oracle: TensorFlow {tf.__version__}, numpy {np.__version__}, op resolver BUILTIN_REF, "
             f"{a.seeds} seeds x {len(MODELS)} models, {a.inputs} inputs each"]
    tot = {"outputs": 0, "DOUBLE": 0, "SINGLE": 0, "disc": 0, "folded": 0}
    problems, keep = [], {}
    for name, (layer, act) in MODELS.items():
        for seed in range(a.seeds):
            model, lo, hi = build_model(layer, act, seed)
            it = interpreter(model, REF)
            p = extract(it, layer, act)
            xs = test_inputs(p, lo, hi, a.inputs, np.random.default_rng(seed + 2000))
            ys = invoke_all(it, xs)
            ys_default = invoke_all(interpreter(model, tf.lite.experimental.OpResolverType.AUTO), xs)
            want = {r: reference(p, xs, r) for r in ROUNDINGS}
            mism = {r: int(np.sum(ys != want[r])) for r in ROUNDINGS}
            dflt = {r: int(np.sum(ys_default != want[r])) for r in ROUNDINGS}
            disc = int(np.sum(want["DOUBLE"] != want["SINGLE"]))
            fold = sum(int(np.sum(reference(p, xs, r, folded=True) != want[r])) for r in ROUNDINGS)
            unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
            _, shifts = ref.layer_multipliers(layer, p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], "DOUBLE")
            per_ch = p["w_scales"].size > 1
            lines.append(f"{name} seed {seed}: in scale {p['in_scale']:.6g} zp {p['in_zp']}, out scale {p['out_scale']:.6g} "
                         f"zp {p['out_zp']}, act [{p['amin']}, {p['amax']}], weights "
                         f"{'per-channel' if per_ch else 'per-tensor'} ({p['w_scales'].size}), shifts [{shifts.min()}, "
                         f"{shifts.max()}], outputs {ys.size}, mismatches DOUBLE {mism['DOUBLE']} SINGLE {mism['SINGLE']}, "
                         f"discriminating {disc}, folded {fold}, unclamped {100 * unclamped:.1f}%, "
                         f"default resolver (info) DOUBLE {dflt['DOUBLE']} SINGLE {dflt['SINGLE']}")
            if layer == "fc" and not per_ch and min(mism.values()) > 0:
                alt = {r: int(np.sum(ys != reference(p, xs, r, scale_product="double"))) for r in ROUNDINGS}
                lines.append(f"  diagnostic: per-tensor FC with a double scale product: DOUBLE {alt['DOUBLE']} SINGLE {alt['SINGLE']}")
            if p["in_zp"] == 0:
                problems.append(f"{name} seed {seed}: input zero point is 0")
            if layer == "conv" and (not per_ch or np.unique(p["w_scales"]).size < 2):
                problems.append(f"{name} seed {seed}: conv weights are not per-channel")
            if unclamped < MIN_UNCLAMPED:
                problems.append(f"{name} seed {seed}: only {100 * unclamped:.1f}% of outputs unclamped")
            tot["outputs"] += ys.size
            tot["disc"] += disc
            tot["folded"] += fold
            for r in ROUNDINGS:
                tot[r] += mism[r]
            if seed == 0:
                keep[name] = (model, p, xs[:N_SAVED], ys[:N_SAVED])
    lines.append(f"totals: outputs {tot['outputs']}, mismatches DOUBLE {tot['DOUBLE']} SINGLE {tot['SINGLE']}, "
                 f"discriminating {tot['disc']}, folded {tot['folded']}")
    winners = [r for r in ROUNDINGS if tot[r] == 0]
    if len(winners) != 1:
        problems.append(f"{len(winners)} roundings match every output; exactly one must")
    if tot["disc"] < MIN_DISCRIMINATING:
        problems.append(f"only {tot['disc']} discriminating outputs (need {MIN_DISCRIMINATING})")
    if tot["folded"]:
        problems.append(f"SIENNA's folded zero-point algebra differs from TFLite's on {tot['folded']} outputs")
    if problems:
        lines += [f"problem: {m}" for m in problems] + ["G0: FAIL"]
    else:
        rounding = winners[0]
        for name, (model, p, xs, ys) in keep.items():
            save(a.out, name, model, p, xs, ys, rounding)
        with open(os.path.join(a.out, "rounding.txt"), "w") as f:
            f.write(rounding + "\n")
        lines += [f"ROUNDING: {rounding}", "G0: PASS"]
    with open(a.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run gate G0 on the farm**

```bash
$J/snap_launch_tree.sh g0_oracle 32 2 $J/venv/bin/python3 tflite_oracle.py \
  --out testbenches/results/int8/tflite_int8 --report testbenches/results/int8/g0_oracle.log
```

**G0 pass criterion** (the oracle computes it; the report's last line is `G0: PASS` or `G0: FAIL`):
1. Every model (8 seeds x 4 kinds) is exactly one int8 `FULLY_CONNECTED` or `CONV_2D` with int8 input and output,
   symmetric int8 weights and int32 bias (else the oracle stops).
2. Over all outputs of all models (about 16.6 million: 1000 inputs x 16 outputs per FC model, x 1024 per conv model),
   exactly one rounding variant of `tflite_ref` equals the BUILTIN_REF interpreter on every output.
3. At least 100 of those outputs discriminate (the two variants differ there), so the pick is evidence.
4. SIENNA's folded algebra (`folded=True`) equals TFLite's on every output, both variants.
5. Every model has a non-zero input zero point; every conv model has per-channel weight scales that are not all equal;
   every model has at least 25% of its outputs off the clamp bounds.

Report: `testbenches/results/int8/g0_oracle.log` (in `$J/runs/g0_oracle/results/int8/`): the versions, one line per model
(scales, zero points, act range, weight quantization, shift range, mismatch counts per variant, discriminating count,
folded mismatches, unclamped share, and the default resolver's mismatches for information), totals, `ROUNDING:` and
the verdict.

Expected: `ROUNDING: DOUBLE` or `ROUNDING: SINGLE` and `G0: PASS`; the losing variant's mismatches are close to the
discriminating count. `$J/runs/g0_oracle.out` shows `exit=0`.

If G0 fails, read the per-model lines before changing anything:
- Both variants mismatch on the per-tensor FC models only, and the `diagnostic` line shows 0 for one variant: the
  per-tensor scale product is double, not float32. Change `layer_multipliers`' default for that case, rerun
  Step 4 and Step 6, and record the finding in the report.
- Both variants mismatch on conv border pixels only: the padding or `im2col_same`.
- Mismatches equal to many outputs at `amin` or `amax`: `activation_range`.
- A zero-point or unclamped-share problem: the model construction (the input range or the representative data), never
  the thresholds.
Never loosen a criterion to pass.

- [ ] **Step 7: Copy the G0 outputs into the tree**

```bash
M=/proj/work/spramanik/SIENNA_int8/testbenches/tflite_int8
ls $M 2>/dev/null && echo "EXISTS: look before copying"   # expect nothing: the copy must not overwrite unseen files
mkdir -p $M && cp $J/runs/g0_oracle/results/int8/tflite_int8/* $M/
ls $M; cat $M/rounding.txt
mkdir -p /proj/work/spramanik/SIENNA_int8/testbenches/results/int8
cp $J/runs/g0_oracle/results/int8/g0_oracle.log /proj/work/spramanik/SIENNA_int8/testbenches/results/int8/
```

Expected: eight model files (four `.tflite`, four `.npz`) and `rounding.txt`; the variant matches the report's
`ROUNDING:` line.

- [ ] **Step 8: Commit (SystolicMesh pointer, then SIENNA)**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh
git add ArithmeticLibrary && git commit -m "ArithmeticLibrary: ipu.py, the integer and requantize models" && git push origin int8
cd /proj/work/spramanik/SIENNA_int8
git add SystolicMesh && git commit -m "SystolicMesh: ArithmeticLibrary with ipu.py"
git add tflite_ref.py && git commit -m "tflite_ref.py: int8 FC and conv as TFLite's reference kernels compute them, on ipu.requant"
git add test_tflite_ref.py && git commit -m "test_tflite_ref.py: QuantizeMultiplier, activation ranges, hand-computed FC and conv"
git add tflite_oracle.py && git commit -m "tflite_oracle.py: gate G0, tflite_ref against the TFLite interpreter's reference kernels"
git add testbenches/tflite_int8 && git commit -m "G0: single-layer int8 test models, their quantization, rounding $(cat testbenches/tflite_int8/rounding.txt)"
git push origin int8
```

`g0_oracle.log` stays untracked (`testbenches/results/` and `*.log` are ignored); Task 24 copies it verbatim into the
sienna-report skill's `history/`.

- [ ] **Step 9: Report G0 to Soham and wait**

Give the verdict, the pinned rounding, the totals line, the TensorFlow version and the report's path. Level 1 starts
only after Soham has seen G0.

---

## Level 1: AriL (gate G1 in Task 8)

Paths in this level are relative to `SystolicMesh/ArithmeticLibrary` in `/proj/work/spramanik/SIENNA_int8` (branch `int8`), unless they start with `$J`, `$T` or `/`. Every command assumes this shell setup:

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/work/spramanik/SIENNA_int8
A=$T/SystolicMesh/ArithmeticLibrary
export TREE=$T
```

Every build and simulation is launched as `TREE=$T $J/snap_launch_tree.sh NAME MEM_GB HOURS cmd...`, with 32 GB (the smallest partition). `TREE` is written on every launch: a launch without it snapshots the bf16 tree, and in Task 8 that would silently re-test bf16.

Task 3 extends the format package; Tasks 4 to 6 build the three integer units, which share one DV pattern, the float units' pattern with the references swapped; Task 7 builds the requantize unit, whose DV reads `ipu.requant`'s vectors directly:

| Float units (bf16 plan) | Integer units (this level) |
|---|---|
| SoftFloat through DPI (`fp_ref_core.h`, `fp_ref.cpp`) | a C reference written from the definitions through DPI (`int_ref_core.h`, `int_ref.cpp`) |
| `+DUMP` checked by `check_fpu.py` against `fpu.py` | `+DUMP` checked by `check_ipu.py` against `ipu.py` (Task 1) |
| `gen_vectors.cpp` + `generate_vectors.sh` -> `vectors_bf16.mem` | `gen_int_vectors.cpp` + `generate_int_vectors.sh` -> `vectors_int8.mem`, `vectors_int32.mem`, `vectors_q4_11.mem` |
| exhaustive slices on the farm (`+A_LO`/`+A_HI`) | intMultiplier exhaustive in one run; fxMac sweep slices with an `ipu.py` digest |
| `TB_*VIVADO.sv` reading the vectors, run in Verilator | the same |

Each TB also checks the C reference on values worked by hand before it judges the unit, so a wrong reference stops the run instead of agreeing with a wrong unit.

Conventions the three units share (D-8 for the reset): ports `clk_i, rstn_i, valid_i, done_o`; valid_i sampled at edge t gives done_o seen at edge t + latency (the TBs measure it as the bf16 TBs do: edges from the one that took valid_i to the one that sees done_o); only the valid bits are reset. `result_o` is loaded only on a valid cycle and holds otherwise; it is random from power-up until the first valid result, so consumers must qualify it with `done_o`.

### Task 3: `sienna_fmt_pkg` int8

**Files:**
- Modify: `Common/src/sienna_fmt_pkg.sv`, `Common/testbenches/TB_sienna_fmt_pkg.sv`, `Common/models/ipu.py`
- Modify: `Multipliers/FP/src/fpMultiplier.sv`, `Adders/FP/src/fpAdder.sv` (the D-7 guard)
- Create (no repo): `$J/cmds/int8_fpguard.sh`

**Interfaces:**
- Consumes: the package and its TB as the bf16 work left them (`Common/Makefile` target `pkg`, unchanged);
  `testbenches/tflite_int8/rounding.txt` (Task 2); `$J/cmds/int8_aril.sh` (Task 0); the bf16 G1 gate run `$J/runs/g1b`
  and `$J/cmd_aril_gate.sh` (the bf16 gate script, unchanged).
- Produces (all `function automatic`, usable in constant expressions):
  - `bit sienna_fmt_pkg::is_int(int exp_w)`: `exp_w == 0`
  - `bit sienna_fmt_pkg::supported(int exp_w, int man_w)`: true for (8, 23), (8, 7), (0, 7) only
  - `int sienna_fmt_pkg::acc_w(int exp_w, int man_w)`: 32 if `is_int(exp_w)`, else `1 + exp_w + man_w`
  - `int sienna_fmt_pkg::mul_lat(int exp_w, int man_w)`: 1 for int8, else 8 above 12 significand bits, else 3
  - `int sienna_fmt_pkg::add_lat(int exp_w, int man_w)`: 1 for int8, else 5
  - `int sienna_fmt_pkg::fx_lat()`: 2 (fxMac, Task 6); `int sienna_fmt_pkg::req_lat()`: 3 (tfliteRequant, Task 7)
  - `localparam string sienna_fmt_pkg::REQ_ROUNDING`: G0's variant, copied from `rounding.txt`; every `tfliteRequant`
    instance takes `.ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)`
  - `is_fp32` and `from_fp32` unchanged; every (8, 23) and (8, 7) result unchanged.
  - `ipu.REQ_ROUNDING`: the same variant, read from the SIENNA tree's `testbenches/tflite_int8/rounding.txt` (four
    directories above `ipu.py` in either AriL checkout); None outside a SIENNA tree.
  - D-7: `fpMultiplier` and `fpAdder` fail elaboration with `<unit>: EXP_W=<n> is an integer format` when `EXP_W < 2`;
    `$J/cmds/int8_fpguard.sh` checks both.
- The GPNAE repo's own AriL checkout (`GPNAE/ArithmeticLibrary`) sees this only when Level 2 moves its pointer; SIENNA's
  Makefile compiles SystolicMesh's copy of the package.

- [ ] **Step 1: Write the failing test**

`Common/testbenches/TB_sienna_fmt_pkg.sv` (whole file; the int8 checks and the constant-expression localparams are new):

```systemverilog
`timescale 1ns / 100ps

// Checks sienna_fmt_pkg: supported formats, accumulator widths, unit latencies, and constants narrowed from fp32.
module TB_sienna_fmt_pkg;
  import sienna_fmt_pkg::*;
  int errs = 0;

  // The functions must work in constant expressions, as generate blocks use them.
  localparam int ACC8 = acc_w(0, 7);
  localparam int MUL8 = mul_lat(0, 7);
  localparam bit INT8 = is_int(0);
  localparam int FX8 = fx_lat();
  localparam int RQ8 = req_lat();

  task automatic expect_eq(input string what, input longint got, input longint want);
    if (got != want) begin
      errs++;
      $display("[FAIL] %s: got %0h, want %0h", what, got, want);
    end
  endtask

  initial begin
    expect_eq("is_fp32(8,23)", is_fp32(8, 23), 1);
    expect_eq("is_fp32(8,7)", is_fp32(8, 7), 0);
    expect_eq("is_fp32(0,7)", is_fp32(0, 7), 0);
    expect_eq("supported(8,23)", supported(8, 23), 1);
    expect_eq("supported(8,7)", supported(8, 7), 1);
    expect_eq("supported(0,7)", supported(0, 7), 1);
    expect_eq("supported(5,10)", supported(5, 10), 0);
    expect_eq("supported(0,15)", supported(0, 15), 0);
    expect_eq("supported(0,3)", supported(0, 3), 0);
    expect_eq("is_int(0)", is_int(0), 1);
    expect_eq("is_int(8)", is_int(8), 0);
    expect_eq("acc_w(0,7)", acc_w(0, 7), 32);
    expect_eq("acc_w(8,23)", acc_w(8, 23), 32);
    expect_eq("acc_w(8,7)", acc_w(8, 7), 16);
    expect_eq("mul_lat(8,23)", mul_lat(8, 23), 8);
    expect_eq("mul_lat(8,7)", mul_lat(8, 7), 3);
    expect_eq("mul_lat(0,7)", mul_lat(0, 7), 1);
    expect_eq("add_lat(8,23)", add_lat(8, 23), 5);
    expect_eq("add_lat(8,7)", add_lat(8, 7), 5);
    expect_eq("add_lat(0,7)", add_lat(0, 7), 1);
    expect_eq("fx_lat()", fx_lat(), 2);
    expect_eq("req_lat()", req_lat(), 3);
    expect_eq("REQ_ROUNDING is SINGLE or DOUBLE", (REQ_ROUNDING == "SINGLE") || (REQ_ROUNDING == "DOUBLE"), 1);
    expect_eq("ACC8 localparam", ACC8, 32);
    expect_eq("MUL8 localparam", MUL8, 1);
    expect_eq("INT8 localparam", INT8, 1);
    expect_eq("FX8 localparam", FX8, 2);
    expect_eq("RQ8 localparam", RQ8, 3);
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

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch_tree.sh i3_pkg 32 1 $J/cmds/int8_aril.sh Common pkg`
Expected: a Verilator build error, `Can't find definition of task/function: 'acc_w'` (or `'is_int'`).

- [ ] **Step 3: Extend the package**

`Common/src/sienna_fmt_pkg.sv` (whole file):

```systemverilog
`timescale 1ns / 100ps

// Number formats a SIENNA build may use and the latencies of their AriL units; generate blocks reject any format not listed.
package sienna_fmt_pkg;

  // fp32: 8 exponent bits, 23 mantissa bits.
  function automatic bit is_fp32(input int exp_w, input int man_w);
    return (exp_w == 8) && (man_w == 23);
  endfunction

  // Integer formats have no exponent: int8 is EXP_W = 0, MAN_W = 7, width 1 + 0 + 7.
  function automatic bit is_int(input int exp_w);
    return exp_w == 0;
  endfunction

  // fp32 (fp32Multiplier, fp32Adder), bf16 (fpMultiplier, fpAdder) and int8 (intMultiplier, intAdder); nothing else is verified.
  function automatic bit supported(input int exp_w, input int man_w);
    return ((exp_w == 8) && ((man_w == 23) || (man_w == 7))) || ((exp_w == 0) && (man_w == 7));
  endfunction

  // Width of partial sums, the reducer, the bias and the mesh result: int32 for int8, the format itself for floats.
  function automatic int acc_w(input int exp_w, input int man_w);
    return is_int(exp_w) ? 32 : 1 + exp_w + man_w;
  endfunction

  // valid_i to done_o: intMultiplier 1; Karatsuba product (fp32Multiplier, fpMultiplier above 12 significand bits) 8, else 3.
  function automatic int mul_lat(input int exp_w, input int man_w);
    if (is_int(exp_w)) return 1;
    return (man_w + 1 > 12) ? 8 : 3;
  endfunction

  // valid_i to done_o: intAdder 1; fp32Adder and fpAdder 5 at every width.
  function automatic int add_lat(input int exp_w, input int man_w);
    return is_int(exp_w) ? 1 : 5;
  endfunction

  // valid_i to done_o of fxMac, the int8 GPNAE lane's Q4.11 multiply-add.
  function automatic int fx_lat();
    return 2;
  endfunction

  // valid_i to done_o of tfliteRequant, TFLite's int8 requantize.
  function automatic int req_lat();
    return 3;
  endfunction

  // The rounding of TFLite's reference kernels, pinned at G0: SIENNA's testbenches/tflite_int8/rounding.txt.
  localparam string REQ_ROUNDING = "DOUBLE";

  // A finite fp32 constant in a format with 8 exponent bits, rounded to nearest even, right-aligned.
  function automatic logic [31:0] from_fp32(input logic [31:0] x, input int man_w);
    automatic int sh = 23 - man_w;
    if (sh == 0) return x;
    return (x + ((32'd1 << (sh - 1)) - 32'd1) + ((x >> sh) & 32'd1)) >> sh;
  endfunction

endpackage
```

The block writes `"DOUBLE"`; this command replaces it with G0's variant, so the package always holds what
`rounding.txt` says:

```bash
cd $T/SystolicMesh/ArithmeticLibrary
sed -i "s/localparam string REQ_ROUNDING = \"[A-Z]*\"/localparam string REQ_ROUNDING = \"$(cat $T/testbenches/tflite_int8/rounding.txt)\"/" Common/src/sienna_fmt_pkg.sv
command grep -n 'REQ_ROUNDING =' Common/src/sienna_fmt_pkg.sv; cat $T/testbenches/tflite_int8/rounding.txt
```

Expected: the package line names the variant `rounding.txt` holds.

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch_tree.sh i3_pkg 32 1 $J/cmds/int8_aril.sh Common pkg`
Expected: `TB_sienna_fmt_pkg: 0 errors` and `RESULT: PASSED`.

- [ ] **Step 5: `ipu.REQ_ROUNDING`**

In `Common/models/ipu.py`, add `import os` above `import numpy as np`, and after the `ROUNDINGS = ("SINGLE", "DOUBLE")`
line:

```python
_ROUNDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "testbenches", "tflite_int8", "rounding.txt")
REQ_ROUNDING = open(_ROUNDING_FILE).read().strip() if os.path.exists(_ROUNDING_FILE) else None  # G0's variant, from the SIENNA tree around this checkout
```

Both AriL checkouts sit four directories below the SIENNA root (`SystolicMesh/ArithmeticLibrary/Common/models` and
`GPNAE/ArithmeticLibrary/Common/models`), in the tree and in every snapshot. Check that the three sources agree (login
node; this imports numpy and reads two small files, no build):

```bash
cd $T
python3 -c "import sys; sys.path.insert(0, 'SystolicMesh/ArithmeticLibrary/Common/models'); import ipu; print('ipu', ipu.REQ_ROUNDING)"
python3 -c "import tflite_ref; print('tflite_ref', tflite_ref.ROUNDING)"
command grep -o 'REQ_ROUNDING = "[A-Z]*"' SystolicMesh/ArithmeticLibrary/Common/src/sienna_fmt_pkg.sv
```

Expected: the same variant three times, the one `rounding.txt` holds.

- [ ] **Step 6: The D-7 guard in `fpMultiplier` and `fpAdder`**

From here on `supported(0, 7)` is true, so every existing consumer that branches `!supported -> $fatal; is_fp32 ->
fp32 units; else -> float units` (`ProcessingElement`, `AccumulationUnit`, `barrel_mac`, `gpnae_poly`, `gpnae_tail`,
`dropout`) would build float units for an int8 elaboration until its own task adds the `is_int` branch. The float
units therefore refuse integer formats themselves. In `Multipliers/FP/src/fpMultiplier.sv`, directly after the line
`localparam logic [W-1:0] QNAN = {1'b0, ONES, 1'b1, {(MAN_W - 1) {1'b0}}};`, insert:

```systemverilog

  // Integer formats have their own units (intMultiplier, intAdder, fxMac): an int8 build must never fall through to this one.
  if (EXP_W < 2) begin : G_BAD_FORMAT
    $fatal(1, "fpMultiplier: EXP_W=%0d is an integer format, not a float one", EXP_W);
  end
```

and in `Adders/FP/src/fpAdder.sv`, after its identical `QNAN` line:

```systemverilog

  // Integer formats have their own units (intMultiplier, intAdder, fxMac): an int8 build must never fall through to this one.
  if (EXP_W < 2) begin : G_BAD_FORMAT
    $fatal(1, "fpAdder: EXP_W=%0d is an integer format, not a float one", EXP_W);
  end
```

At (8, 23) and (8, 7) the condition is false and nothing is generated, so the fp32 and bf16 netlists are unchanged.

`/proj/work/spramanik/sienna_jobs/cmds/int8_fpguard.sh` (`chmod +x`):

```bash
#!/bin/bash
# fpMultiplier and fpAdder at EXP_W = 0 must fail elaboration with their integer-format message (D-7); run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
A=SystolicMesh/ArithmeticLibrary; R=$(pwd)/testbenches/results/int8; mkdir -p $R; rc=0
chk() {  # module, then its sources after the package
  local m=$1; shift
  verilator --lint-only -Wno-fatal -Werror-USERFATAL --top-module $m -GEXP_W=0 -GMAN_W=7 $A/Common/src/sienna_fmt_pkg.sv "$@" \
    > $R/fpguard_$m.txt 2>&1
  if [ $? -ne 0 ] && command grep -q "$m: EXP_W=0 is an integer format" $R/fpguard_$m.txt; then echo "REJECTED $m"
  else echo "NOT-REJECTED $m"; rc=1; fi
}
chk fpMultiplier $A/Multipliers/Radix4Booth/src/R4Booth.sv $A/Multipliers/Karatsuba/src/karatsubaUnsigned.sv $A/Multipliers/FP/src/fpMultiplier.sv
chk fpAdder $A/Adders/FP/src/fpAdder.sv
exit $rc
```

Run the rejection check and the whole bf16 AriL gate (its fp32 / bf16 unit suites) from one snapshot each:

```bash
TREE=$T $J/snap_launch_tree.sh i3_guard 32 1 $J/cmds/int8_fpguard.sh
TREE=$T $J/snap_launch_tree.sh i3_float 32 3 $J/cmd_aril_gate.sh
```

When both have finished, compare the float suites with the bf16 G1 run on the login node (it reads logs only):

```bash
B=$J/runs/g1b/results/uniform; N=$J/runs/i3_float/results/uniform
P="RESULT|SUCCESS|FAILURE|mismatches|errors|latency|checked|products|sums"
for f in $(cd $B && ls *.log); do
  if [[ $f == *_lint.log ]]; then
    command diff <(command grep -oE "^%(Warning|Error)-[A-Z0-9_]+" $B/$f | sort | uniq -c) \
                 <(command grep -oE "^%(Warning|Error)-[A-Z0-9_]+" $N/$f | sort | uniq -c) > /dev/null || echo "DIFFERS: $f"
  else
    command diff <(command grep -hE "$P" $B/$f) <(command grep -hE "$P" $N/$f) > /dev/null || echo "DIFFERS: $f"
  fi
done; echo "compared $(ls $B/*.log | wc -l) logs with runs/g1b"
```

Expected: `i3_guard` prints `REJECTED fpMultiplier` and `REJECTED fpAdder`; the compare prints only its `compared ...`
line: the fp32 / bf16 unit suites are identical to the bf16 gate. If Verilator stops at `EXP_W = 0` on an error
before the `$fatal` (the line then reads `NOT-REJECTED` although the lint failed), read `fpguard_<unit>.txt` and move
the guard above the first declaration that errors; the guard must be what stops the build.

- [ ] **Step 7: Commit (AriL), RTL first**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh/ArithmeticLibrary
git add Common/src/sienna_fmt_pkg.sv && git commit -m "sienna_fmt_pkg: int8 (0, 7) supported, is_int, acc_w, integer unit latencies, fx_lat, req_lat, G0's REQ_ROUNDING"
git add Multipliers/FP/src/fpMultiplier.sv && git commit -m "fpMultiplier: reject integer formats (EXP_W < 2) at elaboration"
git add Adders/FP/src/fpAdder.sv && git commit -m "fpAdder: reject integer formats (EXP_W < 2) at elaboration"
git add Common/testbenches/TB_sienna_fmt_pkg.sv && git commit -m "TB_sienna_fmt_pkg: int8 format, accumulator widths, fx_lat, req_lat, REQ_ROUNDING, constant-expression use"
git add Common/models/ipu.py && git commit -m "ipu.py: REQ_ROUNDING, G0's variant from the SIENNA tree's rounding.txt"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
```

From here on every consumer task (9, 11, 13, 18) puts its `is_int` branch before the float branches (D-7).

### Task 4: `intMultiplier` and the integer-unit DV infrastructure

**Files:**
- Create: `Multipliers/Int/src/intMultiplier.sv`
- Create: `Common/testbenches/int_ref_core.h`, `Common/testbenches/int_ref.cpp`, `Common/testbenches/gen_int_vectors.cpp`, `Common/testbenches/generate_int_vectors.sh`
- Create: `Common/models/check_ipu.py`
- Create: `Multipliers/Int/Makefile`, `Multipliers/Int/testbenches/TB_intMultiplier.sv`, `Multipliers/Int/testbenches/TB_intMultiplierVIVADO.sv`, `Multipliers/Int/testbenches/vectors_int8.mem` (generated)
- Create (no repo): `$J/cmds/int8_aril_ri.sh`, `$J/cmds/int8_vectors.sh`

**Interfaces:**
- Consumes: `ipu.int_mul(a, b, w)`, `ipu.int_add(a, b, w=32)`, `ipu.fx_mac(a, x, c, w=16, frac=11)` (Task 1), called with signed values in int64 arrays (`ipu` reads the low `w` bits of values or bit patterns alike); `sienna_fmt_pkg::mul_lat(0, 7) = 1` (Task 3); `$J/cmds/int8_aril.sh` (Task 0).
- Produces:
  - `module intMultiplier #(parameter int W = 8) (input clk_i, rstn_i, valid_i, input logic signed [W-1:0] A, B, output logic signed [2*W-1:0] result_o, output logic done_o)`: full signed product, latency 1; verified at W = 8 (every pair) and at W = 16, GPNAE's width (every corner pair and 10^6 random pairs).
  - DPI: `import "DPI-C" function int c_int_mul(input int a, input int b, input int w)`, `c_int_add(input int a, input int b, input int w)`, `c_fx_mac(input int a, input int x, input int c, input int w, input int frac)`; each takes signed values (only the low `w` bits are read) and returns the signed result.
  - `gen_int_vectors mul|add|fxmac COUNT OUT`: one line per vector, operands then result in hex, no separators: mul `AABBRRRR`, add `AAAAAAAABBBBBBBBRRRRRRRR`, fxmac `AAAAXXXXCCCCRRRR`.
  - Dump line formats (`+DUMP=`): mul `aa bb rrrr`, add `aaaaaaaa bbbbbbbb rrrrrrrr`, fx `aaaa xxxx cccc rrrr` (hex bit patterns).
  - `check_ipu.py FILE --unit mul|mul16|add|fx [--exhaustive]` and `check_ipu.py FILE --digest`; prints one summary line and `RESULT: PASSED|FAILED`, exits 1 on any mismatch. The W = 16 dump is `aaaa bbbb rrrrrrrr`.
  - `$J/cmds/int8_aril_ri.sh UNIT SEED` (random power-up run of the `verilator` target, log tag `randinit<SEED>`); `$J/cmds/int8_vectors.sh UNIT FILE`.
  - Makefile targets per integer unit: `build`, `verilator`, `check`, `vivado_tb`, `vectors`, `lint` (Int adds `build16`, `verilator16`, `check16`; Fx adds `sweep`).

- [ ] **Step 1: Write the job scripts**

`$J/cmds/int8_aril.sh` is Task 0's. `$J/cmds/int8_aril_ri.sh` (`chmod +x`); environment variables do not survive `blaunch`, so the random power-up flags travel in this script:

```bash
#!/bin/bash
# int8 AriL unit TB with every unreset register powering up random; args: UNIT SEED; run from a snapshot root.
export EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" TAG=randinit$2
exec /proj/work/spramanik/sienna_jobs/cmds/int8_aril.sh "$1" verilator +verilator+rand+reset+2 +verilator+seed+$2
```

`$J/cmds/int8_vectors.sh` (`chmod +x`):

```bash
#!/bin/bash
# Writes the integer units' Vivado vectors on a snapshot and keeps one file; args: UNIT FILE (Multipliers/Int vectors_int8.mem); run from a snapshot root.
A=SystolicMesh/ArithmeticLibrary; R=testbenches/results/int8
mkdir -p $R
$A/Common/testbenches/generate_int_vectors.sh || exit 1
cp $A/$1/testbenches/$2 $R/$2 || exit 1
wc -l $R/$2
```

- [ ] **Step 2: Write the C reference and its DPI wrapper**

`Common/testbenches/int_ref_core.h`:

```cpp
// Reference arithmetic for the integer units, written from the definitions: two's complement, floor shift, saturation.
#pragma once
#include <cstdint>

// The low w bits of x as a signed value.
static inline int64_t iref_sx(int64_t x, int w) {
  const uint64_t m = (w >= 64) ? ~0ull : ((1ull << w) - 1ull);
  uint64_t u = (uint64_t)x & m;
  if (w < 64 && ((u >> (w - 1)) & 1ull)) u |= ~m;
  return (int64_t)u;
}

// floor(x / 2^s), without relying on >> of a negative value.
static inline int64_t iref_floor_shift(int64_t x, int s) { return x >= 0 ? (x >> s) : ~((~x) >> s); }

// x clamped to the signed w-bit range.
static inline int64_t iref_sat(int64_t x, int w) {
  const int64_t hi = (int64_t(1) << (w - 1)) - 1, lo = -(int64_t(1) << (w - 1));
  return x > hi ? hi : (x < lo ? lo : x);
}

// intMultiplier: the full signed product of two w-bit values.
static inline int64_t iref_mul(int64_t a, int64_t b, int w) { return iref_sx(a, w) * iref_sx(b, w); }

// intAdder: a + b modulo 2^w, as a signed value.
static inline int64_t iref_add(int64_t a, int64_t b, int w) { return iref_sx(iref_sx(a, w) + iref_sx(b, w), w); }

// fxMac: sat_w(floor(a * x / 2^frac) + c), computed exactly in 64 bits.
static inline int64_t iref_fx_mac(int64_t a, int64_t x, int64_t c, int w, int frac) {
  return iref_sat(iref_floor_shift(iref_sx(a, w) * iref_sx(x, w), frac) + iref_sx(c, w), w);
}
```

`~((~x) >> s)` is floor for negative x: `~x = -x - 1 >= 0`, and `-(floor((-x - 1) / 2^s) + 1) = -ceil(-x / 2^s) = floor(x / 2^s)`.

`Common/testbenches/int_ref.cpp`:

```cpp
// DPI wrappers of int_ref_core.h for the integer-unit testbenches.
#include <svdpi.h>
#include "int_ref_core.h"

extern "C" int c_int_mul(int a, int b, int w) { return (int)iref_mul(a, b, w); }

extern "C" int c_int_add(int a, int b, int w) { return (int)iref_add(a, b, w); }

extern "C" int c_fx_mac(int a, int x, int c, int w, int frac) { return (int)iref_fx_mac(a, x, c, w, frac); }
```

- [ ] **Step 3: Write the model checker (all three units, so Tasks 5 and 6 only add their units)**

`Common/models/check_ipu.py`:

```python
#!/usr/bin/env python3
"""Checks an integer-unit dump (hex operands then result per line) or a TB_fxMac sweep digest against ipu.py, bit for bit."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipu  # noqa: E402

UNITS = {"mul": (2, 8, 16), "mul16": (2, 16, 32), "add": (2, 32, 32), "fx": (3, 16, 16)}  # operand count, operand width, result width
BLOCK = 256  # outer values per numpy evaluation of a digest (16.7 M results)


def signed(u, w: int):
    """w-bit patterns as signed int64 values."""
    u = np.asarray(u, dtype=np.int64)
    return np.where(u >= (1 << (w - 1)), u - (1 << w), u)


def model(unit: str, ops):
    """ipu.py's result for the unit, as result-width bit patterns."""
    if unit in ("mul", "mul16"):
        r = ipu.int_mul(ops[0], ops[1], w=UNITS[unit][1])
    elif unit == "add":
        r = ipu.int_add(ops[0], ops[1], w=32)
    else:
        r = ipu.fx_mac(ops[0], ops[1], ops[2], w=16, frac=11)
    return np.asarray(r, dtype=np.int64) & ((1 << UNITS[unit][2]) - 1)


def check_dump(path: str, unit: str, exhaustive: bool) -> int:
    n_ops, w, rw = UNITS[unit]
    rows = [ln.split() for ln in open(path) if ln.strip()]
    ops = [signed([int(r[i], 16) for r in rows], w) for i in range(n_ops)]
    got = np.array([int(r[n_ops], 16) for r in rows], dtype=np.int64)
    want = model(unit, ops)
    bad = np.nonzero(got != want)[0]
    ok = len(rows) > 0 and len(bad) == 0
    msg = f"check_ipu {unit}: {len(rows)} results against ipu.py, {len(bad)} mismatches"
    if exhaustive:
        seen = len(set(zip(*(o.tolist() for o in ops))))
        msg += f", {seen} distinct operand combinations of {1 << (n_ops * w)}"
        ok = ok and seen == 1 << (n_ops * w)
    print(msg)
    d, rd = (w + 3) // 4, (rw + 3) // 4
    for i in bad[:20]:
        opnd = " ".join(f"{int(o[i]) & ((1 << w) - 1):0{d}x}" for o in ops)
        print(f"  {opnd}: rtl {got[i]:0{rd}x}, model {want[i]:0{rd}x}")
    return 0 if ok else 1


def check_digest(path: str) -> int:
    """Per outer value, TB_fxMac writes s1 = sum of result patterns and s2 = sum of result pattern * inner pattern."""
    with open(path) as fh:
        head = fh.readline().split()
    if head[:3] != ["#", "fxMac", "digest"]:
        print(f"{path}: not a TB_fxMac digest")
        return 1
    kv = dict(t.split("=", 1) for t in head[3:])
    mode, lo, hi = kv["MODE"], int(kv["LO"]), int(kv["HI"])
    fixed = int(signed(int(kv["FIXED"], 16), 16))
    rows = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    outer = rows[:, 0]
    if len(outer) != hi - lo or not np.array_equal(outer, np.arange(lo, hi)):
        print(f"fx digest MODE={mode} FIXED={fixed}: {len(outer)} outer values, expected every one of [{lo}, {hi})")
        return 1
    inner_u = np.arange(1 << 16, dtype=np.int64)
    inner = signed(inner_u, 16)
    bad = []
    for i0 in range(0, len(outer), BLOCK):
        o = signed(outer[i0:i0 + BLOCK], 16)
        n = len(o)
        big_o, big_i = np.repeat(o, 1 << 16), np.tile(inner, n)
        fix = np.full(big_o.shape, fixed, dtype=np.int64)
        args = (big_o, big_i, fix) if mode == "AX" else (fix, big_o, big_i)
        r = (np.asarray(ipu.fx_mac(*args, w=16, frac=11), dtype=np.int64) & 0xFFFF).reshape(n, 1 << 16)
        m1, m2 = r.sum(axis=1), (r * inner_u).sum(axis=1)
        for k in np.nonzero((m1 != rows[i0:i0 + n, 1]) | (m2 != rows[i0:i0 + n, 2]))[0]:
            bad.append(int(outer[i0 + k]))
    print(f"fx digest MODE={mode} FIXED={fixed}: {len(outer)} outer values x 65536 against ipu.py, "
          f"{len(bad)} mismatching outer values")
    for b in bad[:20]:
        print(f"  outer {b:04x}")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--unit", choices=sorted(UNITS))
    p.add_argument("--digest", action="store_true", help="the file is a TB_fxMac sweep digest")
    p.add_argument("--exhaustive", action="store_true", help="also require every operand combination to appear")
    a = p.parse_args()
    if a.digest == (a.unit is not None):
        p.error("give --unit for a dump or --digest for a digest")
    rc = check_digest(a.file) if a.digest else check_dump(a.file, a.unit, a.exhaustive)
    print(f"RESULT: {'PASSED' if rc == 0 else 'FAILED'}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
```

The digest's sums stay exact in int64: `s2 <= 65535 * 65535 * 65536 < 2^48`. A single wrong result always changes `s1`; two errors that cancel in `s1` would also have to cancel in the position-weighted `s2`. The TB separately compares every result with the C reference, so the digest is the second, independent check against `ipu.py`.

- [ ] **Step 4: Write the failing testbench and the Makefile**

`Multipliers/Int/testbenches/TB_intMultiplier.sv`:

```systemverilog
`timescale 1ns / 100ps

// intMultiplier against the C reference (int_ref_core.h) through DPI, bit for bit: every signed pair at W = 8 (65,536), every corner
// pair and RANDOM random pairs at W = 16 (GPNAE's width), idle cycles between some; latency sienna_fmt_pkg::mul_lat(0, 7). +DUMP=<file> writes every result.
module TB_intMultiplier #(
    parameter int W      = 8,
    parameter int RANDOM = 1000000
);
  import "DPI-C" function int c_int_mul(input int a, input int b, input int w);
  localparam logic [W-1:0] MX = {1'b0, {(W - 1) {1'b1}}};
  localparam logic [W-1:0] MN = {1'b1, {(W - 1) {1'b0}}};
  localparam int NC = 11;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, b = '0;
  logic signed [2*W-1:0] res;
  logic done;
  intMultiplier #(.W(W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(res), .done_o(done));

  typedef struct {
    logic [W-1:0] a, b;
    logic [2*W-1:0] res;
  } tx_t;
  tx_t q[$];
  longint checked = 0, errs = 0, idles = 0, n_expect = 0;
  int fd = 0, cyc = 0, t_issue = -1, lat = -1;
  string dump;

  // The C reference on products worked by hand, before it judges the unit.
  task automatic known_values();
    if (W == 8 && (c_int_mul(-128, -128, 8) != 16384 || c_int_mul(-128, 127, 8) != -16256 || c_int_mul(127, 127, 8) != 16129 ||
                   c_int_mul(-1, -1, 8) != 1 || c_int_mul(-1, 1, 8) != -1 || c_int_mul(0, -128, 8) != 0))
      $fatal(1, "TB_intMultiplier: the C reference fails a known product");
    if (W == 16 && (c_int_mul(-32768, -32768, 16) != 1073741824 || c_int_mul(-32768, 32767, 16) != -1073709056 ||
                    c_int_mul(32767, 32767, 16) != 1073676289 || c_int_mul(-1, -1, 16) != 1))
      $fatal(1, "TB_intMultiplier: the C reference fails a known W = 16 product");
  endtask

  task automatic drive(input logic [W-1:0] x, input logic [W-1:0] y);
    tx_t t;
    t.a = x;
    t.b = y;
    t.res = (2 * W)'(c_int_mul(int'($signed(x)), int'($signed(y)), W));
    q.push_back(t);
    #1 valid = 1;
    a = x;
    b = y;
    @(posedge clk);
  endtask

  // One cycle without valid_i, with junk on the operands.
  task automatic idle();
    #1 valid = 0;
    a = W'($urandom);
    b = W'($urandom);
    idles++;
    @(posedge clk);
  endtask

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      if (lat < 0) lat = cyc - t_issue;  // edges from the one that took valid_i to the one that sees done_o
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        if (res !== t.res) begin
          errs++;
          if (errs <= 20) $display("[FAIL] %h * %h: got %h, reference %h", t.a, t.b, res, t.res);
        end
        if (fd != 0) $fwrite(fd, "%h %h %h\n", t.a, t.b, res);
      end
    end
  end

  initial begin
    if (W > 16) $fatal(1, "TB_intMultiplier: the DPI reference returns a 32-bit int");
    known_values();
    if ($value$plusargs("DUMP=%s", dump)) fd = $fopen(dump, "w");
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    if (W <= 8) begin
      for (int x = 0; x < (1 << W); x++)
        for (int y = 0; y < (1 << W); y++) begin
          drive(W'(x), W'(y));
          if ($urandom_range(0, 7) == 0) idle();
        end
      n_expect = longint'(1) << (2 * W);
    end else begin  // W = 16: every pair of corner values, then RANDOM random pairs
      logic [W-1:0] cv[NC];
      cv = '{'0, W'(1), '1, W'(2), W'(-2), MX, MN, MX - W'(1), MN + W'(1), {(W / 2) {2'b01}}, {(W / 2) {2'b10}}};
      foreach (cv[i])
        foreach (cv[j]) begin
          drive(cv[i], cv[j]);
          if ($urandom_range(0, 7) == 0) idle();
        end
      for (int n = 0; n < RANDOM; n++) begin
        drive(W'($urandom), W'($urandom));
        if ($urandom_range(0, 7) == 0) idle();
      end
      n_expect = NC * NC + RANDOM;
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (fd != 0) $fclose(fd);
    if (q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d results never came out", q.size());
    end
    if (checked != n_expect) begin
      errs++;
      $display("[FAIL] %0d results for %0d pairs", checked, n_expect);
    end
    if (lat != sienna_fmt_pkg::mul_lat(0, 7)) begin
      errs++;
      $display("[FAIL] latency %0d, sienna_fmt_pkg::mul_lat(0, 7) says %0d", lat, sienna_fmt_pkg::mul_lat(0, 7));
    end
    $display("intMultiplier W=%0d: %0d products against the C reference, %0d errors, latency %0d, %0d idle cycles", W, checked,
             errs, lat, idles);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Multipliers/Int/Makefile`. Unlike the float Makefiles, `check` depends on a compile-only `build` target, so it does not run the whole stimulus twice:

```make
SHELL := /bin/bash
# intMultiplier: verilator (C reference through DPI, every int8 pair), check (against ipu.py), verilator16 / check16 (W = 16), vivado_tb, vectors, lint.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/../..
COMMON  = $(ARIL)/Common
TB_DIR  = $(PRJ_DIR)/testbenches
PLUSARGS ?=
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent
RTL     = $(PRJ_DIR)/src/intMultiplier.sv
DESIGN  = $(COMMON)/src/sienna_fmt_pkg.sv $(RTL)
FLAGS   = --binary --timing --assert --sv -I$(COMMON)/testbenches -I$(TB_DIR) \
          --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)

build:
	verilator $(FLAGS) --top-module TB_intMultiplier --Mdir $(PRJ_DIR)/Verilator/ref $(DESIGN) \
	  $(TB_DIR)/TB_intMultiplier.sv $(COMMON)/testbenches/int_ref.cpp -CFLAGS "-I$(COMMON)/testbenches" -o sim

verilator: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim $(PLUSARGS)

check: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim +DUMP=mul_int8.txt $(PLUSARGS)
	python3 $(COMMON)/models/check_ipu.py $(PRJ_DIR)/Verilator/ref/mul_int8.txt --unit mul --exhaustive

build16:
	verilator $(FLAGS) -GW=16 --top-module TB_intMultiplier --Mdir $(PRJ_DIR)/Verilator/ref16 $(DESIGN) \
	  $(TB_DIR)/TB_intMultiplier.sv $(COMMON)/testbenches/int_ref.cpp -CFLAGS "-I$(COMMON)/testbenches" -o sim

verilator16: build16
	cd $(PRJ_DIR)/Verilator/ref16 && ./sim $(PLUSARGS)

check16: build16
	cd $(PRJ_DIR)/Verilator/ref16 && ./sim +DUMP=mul_int16.txt $(PLUSARGS)
	python3 $(COMMON)/models/check_ipu.py $(PRJ_DIR)/Verilator/ref16/mul_int16.txt --unit mul16

vivado_tb:
	verilator $(FLAGS) --top-module TB_intMultiplierVIVADO --Mdir $(PRJ_DIR)/Verilator/viv $(DESIGN) \
	  $(TB_DIR)/TB_intMultiplierVIVADO.sv -o sim
	cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv/sim

vectors:
	$(COMMON)/testbenches/generate_int_vectors.sh

lint:
	for w in 8 16; do verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module intMultiplier -GW=$$w $(RTL) || exit 1; done

.PHONY: build verilator check build16 verilator16 check16 vivado_tb vectors lint
```

Lint reads only the unit, not the package, whose unused function arguments would add UNUSEDSIGNAL noise (bf16 G1 counted 10 such warnings on fpAdder).

- [ ] **Step 5: Run it to see it fail**

Run: `TREE=$T $J/snap_launch_tree.sh i4_ref 32 1 $J/cmds/int8_aril.sh Multipliers/Int verilator`
Expected: Verilator stops with `Cannot find file containing module` naming `src/intMultiplier.sv`; `$J/runs/i4_ref.out` ends in `exit=1`.

- [ ] **Step 6: Write the unit**

`Multipliers/Int/src/intMultiplier.sv`:

```systemverilog
`timescale 1ns / 100ps

// Signed integer multiplier for the int8 build's mesh: result_o = A * B, the full 2W-bit two's-complement product (no rounding, no overflow).
// valid_i at t, done_o at t+1 (one registered multiply); only done_o is reset, result_o loads on valid_i and holds otherwise.
module intMultiplier #(
    parameter int W = 8
) (
    input  logic                  clk_i,
    input  logic                  rstn_i,
    input  logic                  valid_i,
    input  logic signed [W-1:0]   A,
    input  logic signed [W-1:0]   B,
    output logic signed [2*W-1:0] result_o,
    output logic                  done_o
);
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) done_o <= 1'b0;
    else done_o <= valid_i;

  // Both operands sign-extended to 2W bits, so the product is exact.
  always_ff @(posedge clk_i) if (valid_i) result_o <= (2 * W)'(A) * (2 * W)'(B);

endmodule
```

- [ ] **Step 7: Run it to see it pass, and against the model**

```bash
TREE=$T $J/snap_launch_tree.sh i4_ref 32 1 $J/cmds/int8_aril.sh Multipliers/Int verilator
TREE=$T $J/snap_launch_tree.sh i4_chk 32 1 $J/cmds/int8_aril.sh Multipliers/Int check
```

Expected: `intMultiplier W=8: 65536 products against the C reference, 0 errors, latency 1, K idle cycles` (K about 8,200) and `RESULT: PASSED`; `i4_chk` adds `check_ipu mul: 65536 results against ipu.py, 0 mismatches, 65536 distinct operand combinations of 65536` and a second `RESULT: PASSED`.

Then the W = 16 run, since GPNAE's int8 lane instantiates `intMultiplier #(.W(16))` (Task 11):

```bash
TREE=$T $J/snap_launch_tree.sh i4_ref16 32 1 $J/cmds/int8_aril.sh Multipliers/Int verilator16
TREE=$T $J/snap_launch_tree.sh i4_chk16 32 1 $J/cmds/int8_aril.sh Multipliers/Int check16
```

Expected: `intMultiplier W=16: 1000121 products against the C reference, 0 errors, latency 1, K idle cycles` (121 corner pairs and 10^6 random) and `RESULT: PASSED`; `i4_chk16` adds `check_ipu mul16: 1000121 results against ipu.py, 0 mismatches` and a second `RESULT: PASSED`. The exhaustive 2^32 pairs at W = 16 are not run; they are optional as farm slices. A latency of 3 means Task 3's `mul_lat(0, 7)` is missing (the bf16 value). A reference mismatch on products of -128 is a sign-extension bug in the unit. A model mismatch where the C reference agrees with the RTL is a bug in `ipu.py` (Task 1): report it to the Task 1 owner with the pair; do not change the RTL to match.

- [ ] **Step 8: Random power-up**

Run: `TREE=$T $J/snap_launch_tree.sh i4_ri 32 1 $J/cmds/int8_aril_ri.sh Multipliers/Int 2`
Expected: the same summary line as Step 7 and `RESULT: PASSED` in `$J/runs/i4_ri/results/int8/Multipliers_Int_verilator_randinit2.log`.

- [ ] **Step 9: Write the vector generator and the Vivado testbench, generate the vectors**

`Common/testbenches/gen_int_vectors.cpp`:

```cpp
// Writes a vectors file for the integer units' Vivado testbenches, operands then result in hex per line; args: mul|add|fxmac COUNT OUT.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "int_ref_core.h"

// A uniform w-bit pattern (w <= 32) as a signed value.
static int64_t rnd(int w) {
  const uint64_t r = (uint64_t)rand() ^ ((uint64_t)rand() << 16) ^ ((uint64_t)rand() << 32);
  return iref_sx((int64_t)r, w);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: gen_int_vectors mul|add|fxmac COUNT OUT\n");
    return 1;
  }
  const int op = !strcmp(argv[1], "mul") ? 0 : (!strcmp(argv[1], "add") ? 1 : (!strcmp(argv[1], "fxmac") ? 2 : -1));
  const int count = atoi(argv[2]);
  if (op < 0 || count <= 0) {
    fprintf(stderr, "gen_int_vectors: unknown unit %s or bad count %s\n", argv[1], argv[2]);
    return 1;
  }
  const int w = op == 0 ? 8 : (op == 1 ? 32 : 16), frac = 11, rw = op == 0 ? 2 * w : w, d = w / 4, rd = rw / 4;
  const uint64_t m = (1ull << w) - 1ull, rm = (1ull << rw) - 1ull;
  const int64_t mx = (int64_t(1) << (w - 1)) - 1, mn = -(int64_t(1) << (w - 1));
  // Corners first: zero, one, the extremes and their neighbours, and for fxMac 1.0 and its neighbours in Q4.11.
  std::vector<int64_t> cv = {0, 1, -1, 2, -2, mx, mn, mx - 1, mn + 1};
  if (op == 2) cv.insert(cv.end(), {int64_t(2048), int64_t(-2048), int64_t(2047), int64_t(-2049)});
  std::vector<std::array<int64_t, 3>> v;
  for (int64_t a : cv)
    for (int64_t b : cv) {
      if (op == 2) {
        for (int64_t c : {int64_t(0), mx, mn}) v.push_back({a, b, c});
      } else {
        v.push_back({a, b, 0});
      }
    }
  srand(42);
  while ((int)v.size() < count) v.push_back({rnd(w), rnd(w), rnd(w)});
  v.resize(count);
  FILE *out = fopen(argv[3], "w");
  if (!out) {
    perror(argv[3]);
    return 1;
  }
  for (const auto &t : v) {
    const int64_t r = op == 0 ? iref_mul(t[0], t[1], w) : (op == 1 ? iref_add(t[0], t[1], w) : iref_fx_mac(t[0], t[1], t[2], w, frac));
    const unsigned long long a = (uint64_t)t[0] & m, b = (uint64_t)t[1] & m, c = (uint64_t)t[2] & m, res = (uint64_t)r & rm;
    if (op == 2)
      fprintf(out, "%0*llx%0*llx%0*llx%0*llx\n", d, a, d, b, d, c, rd, res);
    else
      fprintf(out, "%0*llx%0*llx%0*llx\n", d, a, d, b, rd, res);
  }
  fclose(out);
  printf("%s: %d %s vectors\n", argv[3], count, argv[1]);
  return 0;
}
```

`Common/testbenches/generate_int_vectors.sh` (`chmod +x`). It writes only the integer units' files; the bf16 `vectors_bf16.mem` files are never touched:

```bash
#!/bin/bash
# Builds gen_int_vectors and writes the vectors for the integer units' Vivado testbenches (intMultiplier, intAdder, fxMac).
cd "$(dirname "$0")" || exit 1
ARIL=../..
g++ -O2 -std=c++17 -o gen_int_vectors gen_int_vectors.cpp || exit 1
[ -d $ARIL/Multipliers/Int/testbenches ] && { ./gen_int_vectors mul 10000 $ARIL/Multipliers/Int/testbenches/vectors_int8.mem || exit 1; }
[ -d $ARIL/Adders/Int/testbenches ] && { ./gen_int_vectors add 10000 $ARIL/Adders/Int/testbenches/vectors_int32.mem || exit 1; }
[ -d $ARIL/Multipliers/Fx/testbenches ] && { ./gen_int_vectors fxmac 10000 $ARIL/Multipliers/Fx/testbenches/vectors_q4_11.mem || exit 1; }
exit 0
```

`Multipliers/Int/testbenches/TB_intMultiplierVIVADO.sv`:

```systemverilog
`timescale 1ns / 100ps

// intMultiplier against a gen_int_vectors.cpp file (A B Res in hex), bit for bit; no DPI, so it runs in Vivado.
module TB_intMultiplierVIVADO #(
    parameter int    W           = 8,
    parameter int    NUM_VECTORS = 10000,
    parameter string VEC_FILE    = "vectors_int8.mem"
);
  logic [4*W-1:0] vec[NUM_VECTORS];
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, b = '0;
  logic signed [2*W-1:0] res;
  logic done;
  intMultiplier #(.W(W)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(res), .done_o(done));

  int errs = 0, checked = 0;
  logic [4*W-1:0] q[$];

  always @(posedge clk)
    if (rstn && done) begin
      logic [W-1:0] ea, eb;
      logic [2*W-1:0] er;
      logic [4*W-1:0] v;
      v = q.pop_front();  // a temporary: Verilator mishandles pop_front() inside a concatenation
      {ea, eb, er} = v;
      checked++;
      if (res !== er) begin
        errs++;
        if (errs <= 20) $display("[FAIL] %h * %h: got %h, expected %h", ea, eb, res, er);
      end
    end

  initial begin
    $readmemh(VEC_FILE, vec);
    if (^vec[0] === 1'bx) $fatal(1, "TB_intMultiplierVIVADO: could not read %s", VEC_FILE);
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    for (int i = 0; i < NUM_VECTORS; i++) begin
      #1 valid = 1;
      {a, b} = vec[i][4*W-1:2*W];
      q.push_back(vec[i]);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $display("intMultiplier vectors %s: %0d checked, %0d errors", VEC_FILE, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NUM_VECTORS) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

Generate on the farm and bring the file back. Check first that the target does not exist, so the copy cannot overwrite a file you have not looked at:

```bash
TREE=$T $J/snap_launch_tree.sh i4_vec 32 1 $J/cmds/int8_vectors.sh Multipliers/Int vectors_int8.mem
test ! -e $A/Multipliers/Int/testbenches/vectors_int8.mem && cp $J/runs/i4_vec/results/int8/vectors_int8.mem $A/Multipliers/Int/testbenches/
wc -l $A/Multipliers/Int/testbenches/vectors_int8.mem; head -3 $A/Multipliers/Int/testbenches/vectors_int8.mem
TREE=$T $J/snap_launch_tree.sh i4_viv 32 1 $J/cmds/int8_aril.sh Multipliers/Int vivado_tb
```

Expected: `10000` lines; the first line `00000000` (0 * 0), the third `00ff0000` (0 * -1); the Vivado TB prints `intMultiplier vectors vectors_int8.mem: 10000 checked, 0 errors` and `RESULT: PASSED`.

- [ ] **Step 10: Lint**

```bash
TREE=$T $J/snap_launch_tree.sh i4_lint 32 1 $J/cmds/int8_aril.sh Multipliers/Int lint
command grep -cE "^%(Warning|Error)" $J/runs/i4_lint/results/int8/Multipliers_Int_lint.log
```

Expected: `0` (W = 8 and W = 16). Any warning is read and fixed in the RTL; none is silenced with `lint_off`.

- [ ] **Step 11: Commit (AriL, branch int8), RTL first**

```bash
cd $A
git add Multipliers/Int/src/intMultiplier.sv && git commit -m "intMultiplier: signed W x W -> 2W product, one registered stage"
git add Common/testbenches/int_ref_core.h Common/testbenches/int_ref.cpp && git commit -m "int_ref: C reference for the integer units (product, wrap sum, floor-shift saturating MAC), DPI wrappers"
git add Common/models/check_ipu.py && git commit -m "check_ipu.py: integer-unit dumps and fxMac sweep digests against ipu.py, bit for bit"
git add Multipliers/Int/testbenches/TB_intMultiplier.sv Multipliers/Int/Makefile && git commit -m "TB_intMultiplier: every int8 pair, and W = 16 corners and random pairs, against the C reference, dump for ipu.py"
git add Common/testbenches/gen_int_vectors.cpp Common/testbenches/generate_int_vectors.sh && git commit -m "gen_int_vectors: vectors files for the integer units' Vivado testbenches"
git add Multipliers/Int/testbenches/TB_intMultiplierVIVADO.sv Multipliers/Int/testbenches/vectors_int8.mem && git commit -m "TB_intMultiplierVIVADO and its int8 vectors"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
test "$(git ls-remote git@github.com:SoHam-56/ArithmeticLibrary.git refs/heads/int8 | cut -f1)" = "$(git rev-parse HEAD)" && echo PUSHED
```

Expected: `PUSHED`.

### Task 5: `intAdder`

**Files:**
- Create: `Adders/Int/src/intAdder.sv`, `Adders/Int/Makefile`, `Adders/Int/testbenches/TB_intAdder.sv`, `Adders/Int/testbenches/TB_intAdderVIVADO.sv`, `Adders/Int/testbenches/vectors_int32.mem` (generated)

**Interfaces:**
- Consumes: `c_int_add`, `check_ipu.py --unit add`, `generate_int_vectors.sh`, `int8_aril_ri.sh`, `int8_vectors.sh` (Task 4); `int8_aril.sh` (Task 0); `ipu.int_add` (Task 1); `sienna_fmt_pkg::add_lat(0, 7) = 1` (Task 3).
- Produces: `module intAdder #(parameter int W = 32) (input clk_i, rstn_i, valid_i, input logic signed [W-1:0] A, B, output logic signed [W-1:0] result_o, output logic done_o)`: `A + B` modulo 2^W, latency 1.

- [ ] **Step 1: Write the failing testbench and the Makefile**

`Adders/Int/testbenches/TB_intAdder.sv`:

```systemverilog
`timescale 1ns / 100ps

// intAdder against the C reference (int_ref_core.h) through DPI, bit for bit: wrap corners worked by hand, every corner pair, carry
// chains of every length, then random pairs (one in four next to a rail), with idle cycles between some. +DUMP=<file> writes every result.
module TB_intAdder #(
    parameter int W      = 32,
    parameter int RANDOM = 1000000
);
  import "DPI-C" function int c_int_add(input int a, input int b, input int w);
  localparam logic [W-1:0] ONE = W'(1);
  localparam logic [W-1:0] ONES = {W{1'b1}};
  localparam logic [W-1:0] MX = {1'b0, {(W - 1) {1'b1}}};
  localparam logic [W-1:0] MN = {1'b1, {(W - 1) {1'b0}}};
  // MAX+1, MIN-1, MIN+MIN, MAX+MAX, -1+1, MIN+MAX, with their sums worked by hand.
  localparam int NK = 6;
  localparam logic [W-1:0] KA[NK] = '{MX, MN, MN, MX, ONES, MN};
  localparam logic [W-1:0] KB[NK] = '{ONE, ONES, MN, MX, ONE, MX};
  localparam logic [W-1:0] KR[NK] = '{MN, MX, '0, ONES - ONE, '0, ONES};
  localparam int NC = 14;
  localparam logic [W-1:0] CORNER[NC] = '{'0, ONE, ONES, ONE + ONE, ONES - ONE, MX, MN, MX - ONE, MN + ONE, {(W / 2) {2'b01}},
                                          {(W / 2) {2'b10}}, ONES >> (W / 2), ONES << (W / 2), ONE << (W / 2)};

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, b = '0, res;
  logic done;
  intAdder #(.W(W)) dut (.clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(res), .done_o(done));

  typedef struct {
    logic [W-1:0] a, b, res;
  } tx_t;
  tx_t q[$];
  logic [W-1:0] va[$], vb[$];
  longint checked = 0, errs = 0, idles = 0;
  int fd = 0, cyc = 0, t_issue = -1, lat = -1;
  string dump;

  function automatic void push(input logic [W-1:0] x, input logic [W-1:0] y);
    va.push_back(x);
    vb.push_back(y);
  endfunction

  task automatic build_stimulus();
    for (int i = 0; i < NK; i++) push(KA[i], KB[i]);
    for (int i = 0; i < NC; i++)
      for (int j = 0; j < NC; j++) push(CORNER[i], CORNER[j]);
    for (int k = 0; k < W; k++) begin  // carries rippling through k bits, borrows through the top, every power of two doubled
      push((ONE << k) - ONE, ONE);
      push(ONES << k, ONES);
      push(ONE << k, ONE << k);
    end
    for (int n = 0; n < RANDOM; n++)
      if (n % 4 == 0)
        push($urandom_range(0, 1) ? MX - W'($urandom_range(0, 255)) : MN + W'($urandom_range(0, 255)),
             W'($signed(9'($urandom_range(0, 511)))));  // within 255 of a rail, a step of -256..255
      else push(W'($urandom), W'($urandom));
  endtask

  task automatic drive(input logic [W-1:0] x, input logic [W-1:0] y);
    tx_t t;
    t.a = x;
    t.b = y;
    t.res = W'(c_int_add(int'(x), int'(y), W));
    q.push_back(t);
    #1 valid = 1;
    a = x;
    b = y;
    @(posedge clk);
  endtask

  // One cycle without valid_i, with junk on the operands.
  task automatic idle();
    #1 valid = 0;
    a = W'($urandom);
    b = W'($urandom);
    idles++;
    @(posedge clk);
  endtask

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      if (lat < 0) lat = cyc - t_issue;  // edges from the one that took valid_i to the one that sees done_o
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        if (res !== t.res) begin
          errs++;
          if (errs <= 20) $display("[FAIL] %h + %h: got %h, reference %h", t.a, t.b, res, t.res);
        end
        if (fd != 0) $fwrite(fd, "%h %h %h\n", t.a, t.b, res);
      end
    end
  end

  initial begin
    if (W > 32) $fatal(1, "TB_intAdder: the DPI reference passes 32-bit ints");
    for (int i = 0; i < NK; i++)
      if (W'(c_int_add(int'(KA[i]), int'(KB[i]), W)) !== KR[i]) $fatal(1, "TB_intAdder: the C reference fails known sum %0d", i);
    if ($value$plusargs("DUMP=%s", dump)) fd = $fopen(dump, "w");
    build_stimulus();
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    foreach (va[i]) begin
      drive(va[i], vb[i]);
      if ($urandom_range(0, 7) == 0) idle();
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (fd != 0) $fclose(fd);
    if (q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d results never came out", q.size());
    end
    if (checked != va.size()) begin
      errs++;
      $display("[FAIL] %0d results for %0d inputs", checked, va.size());
    end
    if (lat != sienna_fmt_pkg::add_lat(0, 7)) begin
      errs++;
      $display("[FAIL] latency %0d, sienna_fmt_pkg::add_lat(0, 7) says %0d", lat, sienna_fmt_pkg::add_lat(0, 7));
    end
    $display("intAdder W=%0d: %0d sums against the C reference, %0d errors, latency %0d, %0d idle cycles", W, checked, errs, lat,
             idles);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Adders/Int/Makefile`:

```make
SHELL := /bin/bash
# intAdder: verilator (C reference through DPI: wrap corners, carry chains, random), check (against ipu.py), vivado_tb, vectors, lint.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/../..
COMMON  = $(ARIL)/Common
TB_DIR  = $(PRJ_DIR)/testbenches
PLUSARGS ?=
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent
RTL     = $(PRJ_DIR)/src/intAdder.sv
DESIGN  = $(COMMON)/src/sienna_fmt_pkg.sv $(RTL)
FLAGS   = --binary --timing --assert --sv -I$(COMMON)/testbenches -I$(TB_DIR) \
          --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)

build:
	verilator $(FLAGS) --top-module TB_intAdder --Mdir $(PRJ_DIR)/Verilator/ref $(DESIGN) \
	  $(TB_DIR)/TB_intAdder.sv $(COMMON)/testbenches/int_ref.cpp -CFLAGS "-I$(COMMON)/testbenches" -o sim

verilator: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim $(PLUSARGS)

check: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim +DUMP=add_int32.txt $(PLUSARGS)
	python3 $(COMMON)/models/check_ipu.py $(PRJ_DIR)/Verilator/ref/add_int32.txt --unit add

vivado_tb:
	verilator $(FLAGS) --top-module TB_intAdderVIVADO --Mdir $(PRJ_DIR)/Verilator/viv $(DESIGN) \
	  $(TB_DIR)/TB_intAdderVIVADO.sv -o sim
	cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv/sim

vectors:
	$(COMMON)/testbenches/generate_int_vectors.sh

lint:
	verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module intAdder $(RTL)

.PHONY: build verilator check vivado_tb vectors lint
```

- [ ] **Step 2: Run it to see it fail**

Run: `TREE=$T $J/snap_launch_tree.sh i5_ref 32 1 $J/cmds/int8_aril.sh Adders/Int verilator`
Expected: Verilator stops with `Cannot find file containing module` naming `src/intAdder.sv`; `exit=1`.

- [ ] **Step 3: Write the unit**

`Adders/Int/src/intAdder.sv`:

```systemverilog
`timescale 1ns / 100ps

// Signed integer adder for the int8 build's int32 accumulation: result_o = A + B modulo 2^W (two's-complement wrap, as TFLite's sums).
// valid_i at t, done_o at t+1 (one registered add); only done_o is reset, result_o loads on valid_i and holds otherwise.
module intAdder #(
    parameter int W = 32
) (
    input  logic                clk_i,
    input  logic                rstn_i,
    input  logic                valid_i,
    input  logic signed [W-1:0] A,
    input  logic signed [W-1:0] B,
    output logic signed [W-1:0] result_o,
    output logic                done_o
);
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) done_o <= 1'b0;
    else done_o <= valid_i;

  // W-bit sum: the carry out of the top bit is dropped, which is the wrap.
  always_ff @(posedge clk_i) if (valid_i) result_o <= A + B;

endmodule
```

- [ ] **Step 4: Run it to see it pass, against the model, and with random power-up**

```bash
TREE=$T $J/snap_launch_tree.sh i5_ref 32 1 $J/cmds/int8_aril.sh Adders/Int verilator
TREE=$T $J/snap_launch_tree.sh i5_chk 32 1 $J/cmds/int8_aril.sh Adders/Int check
TREE=$T $J/snap_launch_tree.sh i5_ri 32 1 $J/cmds/int8_aril_ri.sh Adders/Int 2
```

Expected: `intAdder W=32: 1000298 sums against the C reference, 0 errors, latency 1, K idle cycles` (6 known, 196 corner pairs, 96 carry-chain pairs, 1,000,000 random) and `RESULT: PASSED` in each; `i5_chk` adds `check_ipu add: 1000298 results against ipu.py, 0 mismatches`. A mismatch only on the known wrap vectors (for example `7fffffff + 00000001` giving `80000000` in the reference but something else in the model) is the model saturating or widening instead of wrapping: report it to the Task 1 owner.

- [ ] **Step 5: Vivado testbench and vectors**

`Adders/Int/testbenches/TB_intAdderVIVADO.sv`:

```systemverilog
`timescale 1ns / 100ps

// intAdder against a gen_int_vectors.cpp file (A B Res in hex), bit for bit; no DPI, so it runs in Vivado.
module TB_intAdderVIVADO #(
    parameter int    W           = 32,
    parameter int    NUM_VECTORS = 10000,
    parameter string VEC_FILE    = "vectors_int32.mem"
);
  logic [3*W-1:0] vec[NUM_VECTORS];
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, b = '0, res;
  logic done;
  intAdder #(.W(W)) dut (.clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .B(b), .result_o(res), .done_o(done));

  int errs = 0, checked = 0;
  logic [3*W-1:0] q[$];

  always @(posedge clk)
    if (rstn && done) begin
      logic [W-1:0] ea, eb, er;
      logic [3*W-1:0] v;
      v = q.pop_front();  // a temporary: Verilator mishandles pop_front() inside a concatenation
      {ea, eb, er} = v;
      checked++;
      if (res !== er) begin
        errs++;
        if (errs <= 20) $display("[FAIL] %h + %h: got %h, expected %h", ea, eb, res, er);
      end
    end

  initial begin
    $readmemh(VEC_FILE, vec);
    if (^vec[0] === 1'bx) $fatal(1, "TB_intAdderVIVADO: could not read %s", VEC_FILE);
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    for (int i = 0; i < NUM_VECTORS; i++) begin
      #1 valid = 1;
      {a, b} = vec[i][3*W-1:W];
      q.push_back(vec[i]);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $display("intAdder vectors %s: %0d checked, %0d errors", VEC_FILE, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NUM_VECTORS) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

```bash
TREE=$T $J/snap_launch_tree.sh i5_vec 32 1 $J/cmds/int8_vectors.sh Adders/Int vectors_int32.mem
test ! -e $A/Adders/Int/testbenches/vectors_int32.mem && cp $J/runs/i5_vec/results/int8/vectors_int32.mem $A/Adders/Int/testbenches/
wc -l $A/Adders/Int/testbenches/vectors_int32.mem; command grep -n "^7fffffff00000001" $A/Adders/Int/testbenches/vectors_int32.mem
TREE=$T $J/snap_launch_tree.sh i5_viv 32 1 $J/cmds/int8_aril.sh Adders/Int vivado_tb
```

Expected: `10000` lines; the MAX + 1 corner line reads `7fffffff0000000180000000`; `intAdder vectors vectors_int32.mem: 10000 checked, 0 errors`, `RESULT: PASSED`.

- [ ] **Step 6: Lint**

```bash
TREE=$T $J/snap_launch_tree.sh i5_lint 32 1 $J/cmds/int8_aril.sh Adders/Int lint
command grep -cE "^%(Warning|Error)" $J/runs/i5_lint/results/int8/Adders_Int_lint.log
```

Expected: `0`.

- [ ] **Step 7: Commit (AriL)**

```bash
cd $A
git add Adders/Int/src/intAdder.sv && git commit -m "intAdder: signed W-bit sum wrapping modulo 2^W, one registered stage"
git add Adders/Int/testbenches/TB_intAdder.sv Adders/Int/Makefile && git commit -m "TB_intAdder: int32 wrap corners, carry chains and random pairs against the C reference, dump for ipu.py"
git add Adders/Int/testbenches/TB_intAdderVIVADO.sv Adders/Int/testbenches/vectors_int32.mem && git commit -m "TB_intAdderVIVADO and its int32 vectors"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
test "$(git ls-remote git@github.com:SoHam-56/ArithmeticLibrary.git refs/heads/int8 | cut -f1)" = "$(git rev-parse HEAD)" && echo PUSHED
```

### Task 6: `fxMac` (Q4.11 multiply-add, one Horner step)

**Files:**
- Create: `Multipliers/Fx/src/fxMac.sv`, `Multipliers/Fx/Makefile`, `Multipliers/Fx/testbenches/TB_fxMac.sv`, `Multipliers/Fx/testbenches/TB_fxMacVIVADO.sv`, `Multipliers/Fx/testbenches/vectors_q4_11.mem` (generated)

**Interfaces:**
- Consumes: `c_fx_mac`, `check_ipu.py --unit fx` and `--digest`, `generate_int_vectors.sh`, `int8_aril_ri.sh`, `int8_vectors.sh` (Task 4); `int8_aril.sh` (Task 0); `ipu.fx_mac` (Task 1), which floors (D-1); `sienna_fmt_pkg::fx_lat()` (Task 3).
- Produces: `module fxMac #(parameter int W = 16, parameter int FRAC = 11) (input clk_i, rstn_i, valid_i, input logic signed [W-1:0] A, X, C, output logic signed [W-1:0] result_o, output logic done_o)`: `result_o = sat_W(floor(A * X / 2^FRAC) + C)`, latency `sienna_fmt_pkg::fx_lat()` = 2. Saturation is applied once, after the add. The TB sweep interface `+MODE=AX|XC +FIXED=<hex> +LO= +HI= [+DIGEST=]` and the digest format (header `# fxMac digest MODE= FIXED= LO= HI=`, then `outer s1 s2` per outer pattern, decimal).

**What "exhaustive" means here, and why.** The unit has three 16-bit inputs (2^48 triples), which no simulator covers. The datapath is a 16 x 16 product, a floor shift, a 32-bit add and one saturation. The plan covers each part exhaustively, with 6 sweeps of 2^32 triples each (2.6 x 10^10 in all), plus corners and random triples:

| Sweep | Inputs varied | Fixed | What it exhausts |
|---|---|---|---|
| AX, C = 0 | every A, every X | C = 0 | the product and the floor shift on every pair, and saturation of every product outside int16 |
| AX, C = -1 | every A, every X | C = 0xFFFF | C's sign extension against every product (a zero-extended C gives +65535) |
| AX, C = 32767 | every A, every X | C = 0x7FFF | the upper rail against every product |
| AX, C = -32768 | every A, every X | C = 0x8000 | the lower rail, and every product above 32767 pulled back into range, which catches saturation before the add |
| XC, A = 1.0 | every X, every C | A = 0x0800 | the product equals X exactly, so every int16 + int16 sum and both saturation bounds |
| XC, A = 2.0 | every X, every C | A = 0x1000 | products 2X in [-65536, 65534] (outside int16) against every C |

Each sweep runs as 4 farm slices of 16,384 outer values (2^30 triples per slice). The bf16 exhaustive slices ran 2^28 products in 103 s of simulation (`runs/u4_exh_0`), so a slice should take about 7 minutes plus the digest check. That is an estimate; Step 7 records the measured times. Every result is compared with the C reference in the TB, and every outer value's digest is recomputed from `ipu.fx_mac` by `check_ipu.py --digest`. The per-result dump is not written for sweeps: 2^32 lines per sweep would be about 86 GB, beyond the `/proj/work` quota.

Not covered exhaustively: products outside int16 against middle values of C for |A| other than 2.0, which the random triples and the saturation-boundary triples sample.

- [ ] **Step 1: Write the failing testbench and the Makefile**

`Multipliers/Fx/testbenches/TB_fxMac.sv`:

```systemverilog
`timescale 1ns / 100ps

// fxMac against the C reference (int_ref_core.h) through DPI, bit for bit. Default: triples worked by hand, corner triples, triples whose
// sum lands on a saturation bound or next to it, random triples, idle cycles between some. +MODE=AX|XC +FIXED=<hex> +LO=<n> +HI=<n>:
// every inner pattern for each outer pattern in [LO, HI) (AX: outer A, inner X, C fixed; XC: outer X, inner C, A fixed), and with
// +DIGEST=<file> per outer value the sums check_ipu.py --digest recomputes from ipu.py. +DUMP=<file> writes every result.
module TB_fxMac #(
    parameter int W      = 16,
    parameter int FRAC   = 11,
    parameter int RANDOM = 200000,
    parameter int BOUND  = 5000
);
  localparam int LAT = sienna_fmt_pkg::fx_lat();  // fxMac's latency
  localparam int ONE = 1 << FRAC;
  localparam int MAXV = (1 << (W - 1)) - 1;
  localparam int MINV = -(1 << (W - 1));
  import "DPI-C" function int c_fx_mac(input int a, input int x, input int c, input int w, input int frac);

  // A, X, C and the Q4.11 result worked by hand: floor on negative products, saturation after the add and not before.
  localparam int NK = 10;
  localparam int KV[NK][4] = '{'{-1, 1, 0, -1}, '{3, -683, 0, -2}, '{2048, 2048, 0, 2048}, '{-32768, -32768, 0, 32767},
                               '{32767, -32768, 0, -32768}, '{2048, 32767, 1, 32767}, '{2048, 32766, 1, 32767},
                               '{2048, -32768, -1, -32768}, '{-2048, -32768, -1, 32767}, '{-2048, -32768, -32768, 0}};
  localparam int NC = 19;
  localparam int CORNER[NC] = '{0, 1, -1, 2, -2, ONE - 1, ONE, ONE + 1, -(ONE - 1), -ONE, -(ONE + 1), 2 * ONE, -2 * ONE,
                                MAXV / 2 + 1, MINV / 2, int'({(W / 2) {2'b01}}), int'($signed({(W / 2) {2'b10}})), MAXV, MINV};
  localparam int NCC = 7;
  localparam int CC[NCC] = '{0, 1, -1, ONE / 2, -(ONE / 2), MAXV, MINV};

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, x = '0, c = '0, res;
  logic done;
  fxMac #(.W(W), .FRAC(FRAC)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .X(x), .C(c), .result_o(res), .done_o(done));

  typedef struct {
    logic [W-1:0] a, x, c, res;
    int outer, inner;
  } tx_t;
  tx_t q[$];
  logic [W-1:0] va[$], vx[$], vc[$];
  longint checked = 0, errs = 0, idles = 0, n_expect = 0, s1 = 0, s2 = 0;
  int fd = 0, fdig = 0, cyc = 0, t_issue = -1, lat = -1, cur = -1;
  string dump, dig, mode;

  function automatic int sx(input logic [W-1:0] v);  // a W-bit pattern as a signed int
    return int'($signed(v));
  endfunction

  function automatic void push(input int ia, input int ix, input int ic);
    va.push_back(W'(ia));
    vx.push_back(W'(ix));
    vc.push_back(W'(ic));
  endfunction

  task automatic build_stimulus();
    if (W == 16 && FRAC == 11)
      for (int i = 0; i < NK; i++) push(KV[i][0], KV[i][1], KV[i][2]);
    for (int i = 0; i < NC; i++)
      for (int j = 0; j < NC; j++)
        for (int k = 0; k < NCC; k++) push(CORNER[i], CORNER[j], CC[k]);
    // C chosen so floor(A*X / 2^FRAC) + C is MAX-1, MAX, MAX+1, MIN-1, MIN or MIN+1, where such a C exists.
    for (int n = 0; n < BOUND; n++) begin
      automatic int ra = int'($urandom_range(0, 4 * ONE)) - 2 * ONE;
      automatic int rx = sx(W'($urandom));
      automatic longint p = (longint'(ra) * longint'(rx)) >>> FRAC;
      for (int d = -1; d <= 1; d++) begin
        automatic longint ch = longint'(MAXV) - p + d;
        automatic longint cl = longint'(MINV) - p + d;
        if (ch >= MINV && ch <= MAXV) push(ra, rx, int'(ch));
        if (cl >= MINV && cl <= MAXV) push(ra, rx, int'(cl));
      end
    end
    for (int n = 0; n < RANDOM; n++) push(sx(W'($urandom)), sx(W'($urandom)), sx(W'($urandom)));
  endtask

  task automatic drive(input logic [W-1:0] ia, input logic [W-1:0] ix, input logic [W-1:0] ic, input int outer, input int inner);
    tx_t t;
    t.a = ia;
    t.x = ix;
    t.c = ic;
    t.res = W'(c_fx_mac(sx(ia), sx(ix), sx(ic), W, FRAC));
    t.outer = outer;
    t.inner = inner;
    q.push_back(t);
    #1 valid = 1;
    a = ia;
    x = ix;
    c = ic;
    @(posedge clk);
  endtask

  // One cycle without valid_i, with junk on the operands.
  task automatic idle();
    #1 valid = 0;
    a = W'($urandom);
    x = W'($urandom);
    c = W'($urandom);
    idles++;
    @(posedge clk);
  endtask

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      logic [W-1:0] ru;
      if (lat < 0) lat = cyc - t_issue;  // edges from the one that took valid_i to the one that sees done_o
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        ru = res;
        if (ru !== t.res) begin
          errs++;
          if (errs <= 20) $display("[FAIL] %h * %h + %h: got %h, reference %h", t.a, t.x, t.c, ru, t.res);
        end
        if (fd != 0) $fwrite(fd, "%h %h %h %h\n", t.a, t.x, t.c, ru);
        if (fdig != 0) begin
          if (t.outer != cur) begin
            if (cur >= 0) $fwrite(fdig, "%0d %0d %0d\n", cur, s1, s2);
            cur = t.outer;
            s1 = 0;
            s2 = 0;
          end
          s1 += longint'(ru);
          s2 += longint'(ru) * longint'(t.inner);
        end
      end
    end
  end

  initial begin
    int lo, hi;
    bit ax;
    logic [W-1:0] fixed;
    if (W > 16) $fatal(1, "TB_fxMac: the sweep and the DPI reference assume W <= 16");
    if (W == 16 && FRAC == 11)
      for (int i = 0; i < NK; i++)
        if (c_fx_mac(KV[i][0], KV[i][1], KV[i][2], W, FRAC) != KV[i][3])
          $fatal(1, "TB_fxMac: the C reference fails known triple %0d", i);
    if ($value$plusargs("DUMP=%s", dump)) fd = $fopen(dump, "w");
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("MODE=%s", mode)) begin
      if (!$value$plusargs("FIXED=%h", fixed) || !$value$plusargs("LO=%d", lo) || !$value$plusargs("HI=%d", hi))
        $fatal(1, "TB_fxMac: a sweep needs +FIXED=, +LO= and +HI=");
      if (mode != "AX" && mode != "XC") $fatal(1, "TB_fxMac: MODE is AX or XC, not %s", mode);
      ax = (mode == "AX");
      n_expect = longint'(hi - lo) << W;
      if ($value$plusargs("DIGEST=%s", dig)) begin
        fdig = $fopen(dig, "w");
        $fwrite(fdig, "# fxMac digest MODE=%s FIXED=%h LO=%0d HI=%0d\n", mode, fixed, lo, hi);
      end
      $display("sweep MODE=%s FIXED=%h outer [%0d, %0d)", mode, fixed, lo, hi);
      for (int o = lo; o < hi; o++)
        for (int i = 0; i < (1 << W); i++)
          if (ax) drive(W'(o), W'(i), fixed, o, i);
          else drive(fixed, W'(o), W'(i), o, i);
    end else begin
      build_stimulus();
      n_expect = va.size();
      foreach (va[k]) begin
        drive(va[k], vx[k], vc[k], k, 0);
        if ($urandom_range(0, 7) == 0) idle();
      end
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    if (fd != 0) $fclose(fd);
    if (fdig != 0) begin
      if (cur >= 0) $fwrite(fdig, "%0d %0d %0d\n", cur, s1, s2);
      $fclose(fdig);
    end
    if (q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d results never came out", q.size());
    end
    if (checked != n_expect) begin
      errs++;
      $display("[FAIL] %0d results for %0d inputs", checked, n_expect);
    end
    if (lat != LAT) begin
      errs++;
      $display("[FAIL] latency %0d, fxMac's is %0d", lat, LAT);
    end
    $display("fxMac W=%0d FRAC=%0d: %0d results against the C reference, %0d errors, latency %0d, %0d idle cycles", W, FRAC,
             checked, errs, lat, idles);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

Checked by hand: `-2048 * -32768 = 2^26`, shifted by 11 gives 32768, one past int16. With C = -1 the exact answer is 32767. A unit that saturates the product before adding gives 32767 - 1 = 32766. With C = -32768 the exact answer is 0, and that unit gives -1. `3 * -683 = -2049` floors to -2, where truncation gives -1.

`Multipliers/Fx/Makefile`:

```make
SHELL := /bin/bash
# fxMac: verilator (C reference through DPI), check (against ipu.py), sweep (exhaustive slice, ipu.py digest), vivado_tb, vectors, lint.
PRJ_DIR = $(shell pwd)
ARIL    = $(PRJ_DIR)/../..
COMMON  = $(ARIL)/Common
TB_DIR  = $(PRJ_DIR)/testbenches
PLUSARGS ?=
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent
RTL     = $(PRJ_DIR)/src/fxMac.sv
DESIGN  = $(COMMON)/src/sienna_fmt_pkg.sv $(RTL)
FLAGS   = --binary --timing --assert --sv -I$(COMMON)/testbenches -I$(TB_DIR) \
          --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)

build:
	verilator $(FLAGS) --top-module TB_fxMac --Mdir $(PRJ_DIR)/Verilator/ref $(DESIGN) \
	  $(TB_DIR)/TB_fxMac.sv $(COMMON)/testbenches/int_ref.cpp -CFLAGS "-I$(COMMON)/testbenches" -o sim

verilator: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim $(PLUSARGS)

check: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim +DUMP=fx_q4_11.txt $(PLUSARGS)
	python3 $(COMMON)/models/check_ipu.py $(PRJ_DIR)/Verilator/ref/fx_q4_11.txt --unit fx

sweep: build
	cd $(PRJ_DIR)/Verilator/ref && ./sim $(PLUSARGS) +DIGEST=fx_digest.txt
	python3 $(COMMON)/models/check_ipu.py $(PRJ_DIR)/Verilator/ref/fx_digest.txt --digest

vivado_tb:
	verilator $(FLAGS) --top-module TB_fxMacVIVADO --Mdir $(PRJ_DIR)/Verilator/viv $(DESIGN) \
	  $(TB_DIR)/TB_fxMacVIVADO.sv -o sim
	cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv/sim

vectors:
	$(COMMON)/testbenches/generate_int_vectors.sh

lint:
	verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module fxMac $(RTL)

.PHONY: build verilator check sweep vivado_tb vectors lint
```

- [ ] **Step 2: Run it to see it fail**

Run: `TREE=$T $J/snap_launch_tree.sh i6_ref 32 1 $J/cmds/int8_aril.sh Multipliers/Fx verilator`
Expected: Verilator stops with `Cannot find file containing module` naming `src/fxMac.sv`; `exit=1`.

- [ ] **Step 3: Write the unit**

`Multipliers/Fx/src/fxMac.sv`:

```systemverilog
`timescale 1ns / 100ps

// Fixed-point multiply-add, one Horner step of the int8 GPNAE lane: result_o = sat_W(floor(A * X / 2^FRAC) + C); Q4.11 at the defaults.
// Full 2W-bit signed product, arithmetic shift right by FRAC (floor, D-1), 2W-bit add of C, one saturation to W bits after the add.
// valid_i at t, done_o at t+2 (product registered, then shift, add and saturate registered); only the valid bits are reset.
module fxMac #(
    parameter int W    = 16,
    parameter int FRAC = 11
) (
    input  logic                clk_i,
    input  logic                rstn_i,
    input  logic                valid_i,
    input  logic signed [W-1:0] A,
    input  logic signed [W-1:0] X,
    input  logic signed [W-1:0] C,
    output logic signed [W-1:0] result_o,
    output logic                done_o
);
  localparam int PW = 2 * W;

  // Stage 1: the exact product (|A * X| <= 2^(2W-2)) and the addend.
  logic                 s1_v;
  logic signed [PW-1:0] s1_p;
  logic signed [W-1:0]  s1_c;

  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) s1_v <= 1'b0;
    else s1_v <= valid_i;

  always_ff @(posedge clk_i)
    if (valid_i) begin
      s1_p <= PW'(A) * PW'(X);
      s1_c <= C;
    end

  // Stage 2: the sum fits 2W bits (|p >>> FRAC| <= 2^(2W-2), |C| <= 2^(W-1)); it fits W bits when its top W+1 bits agree.
  logic signed [PW-1:0] sum;
  logic                 fits;
  assign sum  = (s1_p >>> FRAC) + PW'(s1_c);
  assign fits = (sum[PW-1:W-1] == '0) || (sum[PW-1:W-1] == '1);

  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) done_o <= 1'b0;
    else done_o <= s1_v;

  always_ff @(posedge clk_i)
    if (s1_v) result_o <= fits ? sum[W-1:0] : {sum[PW-1], {(W - 1) {~sum[PW-1]}}};

endmodule
```

- [ ] **Step 4: Run it to see it pass, against the model, and with random power-up**

```bash
TREE=$T $J/snap_launch_tree.sh i6_ref 32 1 $J/cmds/int8_aril.sh Multipliers/Fx verilator
TREE=$T $J/snap_launch_tree.sh i6_chk 32 1 $J/cmds/int8_aril.sh Multipliers/Fx check
TREE=$T $J/snap_launch_tree.sh i6_ri 32 1 $J/cmds/int8_aril_ri.sh Multipliers/Fx 2
```

Expected: `fxMac W=16 FRAC=11: N results against the C reference, 0 errors, latency 2, K idle cycles`, with N about 217,500 (10 known, 2,527 corner, about 15,000 boundary, 200,000 random; the boundary count depends on the random draws), and `RESULT: PASSED` in each; `i6_chk` adds `check_ipu fx: N results against ipu.py, 0 mismatches`. If the reference and RTL agree but the model differs on negative products whose low 11 bits are non-zero, the model truncates where D-1 says floor: report it to the Task 1 owner.

- [ ] **Step 5: Vivado testbench and vectors**

`Multipliers/Fx/testbenches/TB_fxMacVIVADO.sv`:

```systemverilog
`timescale 1ns / 100ps

// fxMac against a gen_int_vectors.cpp file (A X C Res in hex), bit for bit; no DPI, so it runs in Vivado.
module TB_fxMacVIVADO #(
    parameter int    W           = 16,
    parameter int    FRAC        = 11,
    parameter int    NUM_VECTORS = 10000,
    parameter string VEC_FILE    = "vectors_q4_11.mem"
);
  logic [4*W-1:0] vec[NUM_VECTORS];
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic signed [W-1:0] a = '0, x = '0, c = '0, res;
  logic done;
  fxMac #(.W(W), .FRAC(FRAC)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .A(a), .X(x), .C(c), .result_o(res), .done_o(done));

  int errs = 0, checked = 0;
  logic [4*W-1:0] q[$];

  always @(posedge clk)
    if (rstn && done) begin
      logic [W-1:0] ea, ex, ec, er;
      logic [4*W-1:0] v;
      v = q.pop_front();  // a temporary: Verilator mishandles pop_front() inside a concatenation
      {ea, ex, ec, er} = v;
      checked++;
      if (res !== er) begin
        errs++;
        if (errs <= 20) $display("[FAIL] %h * %h + %h: got %h, expected %h", ea, ex, ec, res, er);
      end
    end

  initial begin
    $readmemh(VEC_FILE, vec);
    if (^vec[0] === 1'bx) $fatal(1, "TB_fxMacVIVADO: could not read %s", VEC_FILE);
    repeat (8) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    for (int i = 0; i < NUM_VECTORS; i++) begin
      #1 valid = 1;
      {a, x, c} = vec[i][4*W-1:W];
      q.push_back(vec[i]);
      @(posedge clk);
    end
    #1 valid = 0;
    repeat (20) @(posedge clk);
    $display("fxMac vectors %s: %0d checked, %0d errors", VEC_FILE, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NUM_VECTORS) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

```bash
TREE=$T $J/snap_launch_tree.sh i6_vec 32 1 $J/cmds/int8_vectors.sh Multipliers/Fx vectors_q4_11.mem
test ! -e $A/Multipliers/Fx/testbenches/vectors_q4_11.mem && cp $J/runs/i6_vec/results/int8/vectors_q4_11.mem $A/Multipliers/Fx/testbenches/
wc -l $A/Multipliers/Fx/testbenches/vectors_q4_11.mem; command grep -n "^8000800000007fff" $A/Multipliers/Fx/testbenches/vectors_q4_11.mem
TREE=$T $J/snap_launch_tree.sh i6_viv 32 1 $J/cmds/int8_aril.sh Multipliers/Fx vivado_tb
```

Expected: `10000` lines; the (-32768, -32768, 0) corner line reads `8000800000007fff` (2^30 >> 11 saturates to 32767); `fxMac vectors vectors_q4_11.mem: 10000 checked, 0 errors`, `RESULT: PASSED`.

- [ ] **Step 6: Lint, then commit (AriL)**

```bash
TREE=$T $J/snap_launch_tree.sh i6_lint 32 1 $J/cmds/int8_aril.sh Multipliers/Fx lint
command grep -cE "^%(Warning|Error)" $J/runs/i6_lint/results/int8/Multipliers_Fx_lint.log
```

Expected: `0`. Then commit:

```bash
cd $A
git add Multipliers/Fx/src/fxMac.sv && git commit -m "fxMac: Q4.11 multiply-add, floor shift, one saturation after the add, two stages"
git add Multipliers/Fx/testbenches/TB_fxMac.sv Multipliers/Fx/Makefile && git commit -m "TB_fxMac: corners, saturation bounds, random and exhaustive sweeps against the C reference, dump and digest for ipu.py"
git add Multipliers/Fx/testbenches/TB_fxMacVIVADO.sv Multipliers/Fx/testbenches/vectors_q4_11.mem && git commit -m "TB_fxMacVIVADO and its Q4.11 vectors"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
test "$(git ls-remote git@github.com:SoHam-56/ArithmeticLibrary.git refs/heads/int8 | cut -f1)" = "$(git rev-parse HEAD)" && echo PUSHED
git log -1 --format=%h -- Multipliers/Fx/src/fxMac.sv   # the commit the sweeps prove, for the G1 report
```

- [ ] **Step 7: The six sweeps on the farm**

Check the quota first: 24 snapshots of about 25 MB each, and each run keeps a 16,384-line digest.

```bash
quota -s 2>/dev/null | tail -3
for sw in "AX 0000" "AX ffff" "AX 7fff" "AX 8000" "XC 0800" "XC 1000"; do
  set -- $sw
  for s in 0 1 2 3; do
    TREE=$T $J/snap_launch_tree.sh i6_${1}_${2}_$s 32 2 $J/cmds/int8_aril.sh Multipliers/Fx sweep \
      +MODE=$1 +FIXED=$2 +LO=$((s * 16384)) +HI=$(((s + 1) * 16384))
  done
done
```

Collect when all 24 have left the queue (`squeue -u $USER`):

```bash
for r in $J/runs/i6_[AX][XC]_*_[0-3]; do
  echo "$(basename $r): $(command grep -hE '^fxMac W=|^fx digest|Elapsed' $r/stdout.log | tr '\n' ' ') $(command grep -c 'RESULT: PASSED' $r/stdout.log) passed"
done
```

Expected, for each of the 24 runs: `fxMac W=16 FRAC=11: 1073741824 results against the C reference, 0 errors, latency 2, 0 idle cycles`, `fx digest MODE=.. FIXED=..: 16384 outer values x 65536 against ipu.py, 0 mismatching outer values`, and `2 passed`. Record each run's `Elapsed` time for the G1 report. If a slice exceeds 2 hours, relaunch it as four slices of 4,096 outer values; do not drop it. For a mismatching outer value v in a digest, see the triples with `TREE=$T $J/snap_launch_tree.sh i6_dbg 32 1 $J/cmds/int8_aril.sh Multipliers/Fx check +MODE=<mode> +FIXED=<hex> +LO=v +HI=$((v + 1))`, which dumps that outer value's 65,536 results and checks them against `ipu.py` one by one.

- [ ] **Step 8: Report Task 6**

Report the six sweeps' totals (6 x 4,294,967,296 results, errors, digest mismatches), the slice times, and the fxMac commit from Step 6. Nothing to commit. The choice of the six sweeps is in the plan's Open for Soham section.

### Task 7: `tfliteRequant` and its DV

**Files:**
- Create: `Requant/src/tfliteRequant.sv`, `Requant/testbenches/TB_tfliteRequant.sv`,
  `Requant/testbenches/gen_requant_vectors.py`, `Requant/Makefile`
- Create (generated): `Requant/testbenches/vectors_double.mem`, `Requant/testbenches/vectors_single.mem`
- Create (sienna_jobs, no repo): `$J/cmds/int8_req_vec.sh`

**Interfaces:**
- Consumes: `ipu.requant`, `ipu.bits`, `ipu.ROUNDINGS`, `ipu.INT32_MIN/MAX` (Task 1); `$J/cmds/int8_aril.sh` (Task 0);
  `sienna_fmt_pkg::req_lat()` (Task 3), which the testbench checks the latency against.
- Produces:
  - `module tfliteRequant #(parameter string ROUNDING = "DOUBLE") (input clk_i, rstn_i, valid_i, input signed [31:0]
    acc_i, mult_i, input signed [7:0] shift_i, zp_i, act_min_i, act_max_i, output [7:0] result_o, output done_o)`;
    `result_o = clamp(MultiplyByQuantizedMultiplier(acc_i, mult_i, shift_i) + zp_i, act_min_i, act_max_i)` as
    `ipu.requant`; one result per cycle; valid_i at t, done_o at t + `sienna_fmt_pkg::req_lat()` (3); only the valid
    bits are reset (D-8); `shift_i` in [-31, 30] (a simulation assertion checks it); `ROUNDING` other than "SINGLE" or
    "DOUBLE" fails elaboration. Its users instantiate `.ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)`; the DV runs both.
  - Vector line format (TB input and the committed `.mem` files): `acc mult shift zp act_min act_max expected`, hex,
    8 8 2 2 2 2 2 digits, the low bits of each value.
  - `gen_requant_vectors.py OUT --rounding SINGLE|DOUBLE [--random N] [--seed S]`: 33,718 corners, up to 2,000 vectors
    where the roundings differ, then N random (default 10^6).
  - Makefile targets: `verilator` (`ROUNDING=`, `RANDOM=`, `EXTRA_FLAGS=`, `PLUSARGS=`), `both`, `vivado_tb`, `vectors`,
    `lint`, `lint_bad`; G1 (Task 8) runs `both`, `vivado_tb`, `lint` and `lint_bad` through `int8_aril.sh`, and random
    power-up as `both` with `TAG=randinit<seed>`, `EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique"`
    and `+verilator+rand+reset+2 +verilator+seed+<seed>`.

The DV is file-based rather than DPI: `ipu.requant` is itself the G0 reference, so the TB reads its vectors directly and
no second (C) implementation needs its own proof; the same TB with the committed vector files is the Vivado testbench.

- [ ] **Step 1: Write the vector generator, the testbench and the Makefile**

`Requant/testbenches/gen_requant_vectors.py`:

```python
#!/usr/bin/env python3
"""Writes tfliteRequant vectors from ipu.requant, one per line in hex: acc mult shift zp act_min act_max expected.
Corners first (every shift against int32 and multiplier extremes, output ties, zero points and clamps, vectors where the two
roundings differ), then random vectors, three quarters of them aimed at outputs near the int8 range."""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Common", "models"))
import ipu  # noqa: E402

MIN, MAX = ipu.INT32_MIN, ipu.INT32_MAX
ACC = [MIN, MIN + 1, -(1 << 30) - 1, -(1 << 30), -(1 << 24), -65536, -32769, -257, -256, -255, -129, -128, -127, -3, -2,
       -1, 0, 1, 2, 3, 127, 128, 255, 256, 32768, 65536, 1 << 24, 1 << 30, (1 << 30) + 1, MAX - 1, MAX]
MULT = [0, 1, (1 << 30) - 1, 1 << 30, (1 << 30) + 1, 3 << 29, MAX - 1, MAX, MIN, MIN + 1, -1, -(1 << 30)]
SHIFTS = list(range(-31, 31))
ZPS = [-128, -1, 1, 127]


def clamps(zp):
    """Full range, ReLU at the zero point, pinned low, pinned high, a ReLU6-like window, and act_min > act_max."""
    return [(-128, 127), (zp, 127), (-128, -128), (127, 127), (0, 6), (5, -5)]


def corners():
    rows = [(a, m, s, 0, -128, 127) for a, m, s in itertools.product(ACC, MULT, SHIFTS)]
    for a, m, s, z in itertools.product(ACC, [1 << 30, MAX], [-31, -8, -1, 0, 1, 8, 30], ZPS):
        rows += [(a, m, s, z, lo, hi) for lo, hi in clamps(z)]
    for s in range(-31, 1):  # exact halves at the output: +-k << -s times 0.5 * 2^s is +-k/2
        for k in (1, 3, 5, 7):
            if k << (-s) <= MAX:
                rows += [(sg * (k << (-s)), 1 << 30, s, 0, -128, 127) for sg in (1, -1)]
    return [np.array(c, dtype=np.int64) for c in zip(*rows)]


def random_rows(n, rng):
    """Uniform int32 accumulators and shifts, three quarters replaced by ones aimed at |y| <= 160; an eighth with any int32 multiplier."""
    acc = rng.integers(MIN, MAX + 1, n, dtype=np.int64)
    mult = rng.integers(1 << 30, MAX + 1, n, dtype=np.int64)
    shift = rng.integers(-31, 31, n, dtype=np.int64)
    kind = rng.integers(0, 8, n)
    aim = kind < 6
    s_aim = rng.integers(-24, 9, n, dtype=np.int64)
    a_aim = np.clip(np.round(rng.uniform(-160, 160, n) * 2.0 ** 31 / mult * 2.0 ** (-s_aim)), MIN, MAX).astype(np.int64)
    acc, shift = np.where(aim, a_aim, acc), np.where(aim, s_aim, shift)
    mult = np.where(kind == 7, rng.integers(MIN, MAX + 1, n, dtype=np.int64), mult)
    zp = rng.integers(-128, 128, n, dtype=np.int64)
    c = rng.integers(0, 8, n)
    r1, r2 = rng.integers(-128, 128, n, dtype=np.int64), rng.integers(-128, 128, n, dtype=np.int64)
    lo = np.select([c < 5, c == 5], [-128, zp], default=np.minimum(r1, r2))
    hi = np.select([c < 5, c == 5], [127, 127], default=np.maximum(r1, r2))
    return [acc, mult, shift, zp, lo.astype(np.int64), hi.astype(np.int64)]


def discriminators(rng, want=2000):
    """Vectors where DOUBLE and SINGLE disagree, so both RTL variants see their difference."""
    rows = random_rows(200000, rng)
    d = ipu.requant(*rows, "DOUBLE") != ipu.requant(*rows, "SINGLE")
    return [r[d][:want] for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out")
    p.add_argument("--rounding", choices=ipu.ROUNDINGS, required=True)
    p.add_argument("--random", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    c, d, r = corners(), discriminators(rng), random_rows(a.random, rng)
    cols = [np.concatenate([c[i], d[i], r[i]]) for i in range(6)]
    want = ipu.requant(*cols, a.rounding)
    other = ipu.requant(*cols, "SINGLE" if a.rounding == "DOUBLE" else "DOUBLE")
    table = np.stack([ipu.bits(v, w) for v, w in zip(cols + [want], [32, 32, 8, 8, 8, 8, 8])], axis=1)
    np.savetxt(a.out, table, fmt="%08x %08x %02x %02x %02x %02x %02x")
    at_bound = float(np.mean((want == cols[4]) | (want == cols[5])))
    print(f"{a.out}: {len(table)} vectors ({len(c[0])} corners, {len(d[0])} rounding discriminators, {a.random} random), "
          f"rounding {a.rounding}, {int(np.sum(want != other))} differ from the other rounding, "
          f"{100 * at_bound:.1f}% at a clamp bound")


if __name__ == "__main__":
    main()
```

`Requant/testbenches/TB_tfliteRequant.sv`:

```systemverilog
`timescale 1ns / 100ps

// tfliteRequant against ipu.requant's vectors (gen_requant_vectors.py): every result bit for bit, latency 3, one vector per
// cycle with a bubble of random inputs every 64 vectors; no DPI, so it also runs in Vivado. +VEC=<file> picks the vectors.
module TB_tfliteRequant #(
    parameter string ROUNDING = "DOUBLE"
);
  localparam int LAT = sienna_fmt_pkg::req_lat();  // tfliteRequant's latency
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [31:0] acc = '0, mult = '0;
  logic [7:0] shift = '0, zp = '0, amin = '0, amax = '0;
  logic [7:0] res;
  logic done;

  tfliteRequant #(.ROUNDING(ROUNDING)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .acc_i(acc), .mult_i(mult), .shift_i(shift), .zp_i(zp),
      .act_min_i(amin), .act_max_i(amax), .result_o(res), .done_o(done));

  typedef struct {
    logic [31:0] acc, mult;
    logic [7:0] shift, zp, amin, amax, want;
  } tx_t;
  tx_t q[$];
  longint checked = 0, errs = 0, issued = 0;
  int cyc = 0, t_issue = -1, lat = -1;

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      tx_t t;
      if (lat < 0) lat = cyc - t_issue;  // edges from the one that took valid_i to the one that sees done_o
      if (q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        t = q.pop_front();
        checked++;
        if (res !== t.want) begin
          errs++;
          if (errs <= 20)
            $display("[FAIL] acc %h mult %h shift %0d zp %0d clamp [%0d, %0d]: got %0d, ipu.requant %0d", t.acc, t.mult,
                     $signed(t.shift), $signed(t.zp), $signed(t.amin), $signed(t.amax), $signed(res), $signed(t.want));
        end
      end
    end
  end

  initial begin
    string vec;
    int fd;
    logic [31:0] fa, fm;
    logic [7:0] fs, fz, fl, fh, fw;
    tx_t t;
    if (!$value$plusargs("VEC=%s", vec)) vec = (ROUNDING == "SINGLE") ? "vectors_single.mem" : "vectors_double.mem";
    fd = $fopen(vec, "r");
    if (fd == 0) $fatal(1, "TB_tfliteRequant: cannot open %s", vec);
    repeat (4) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    while ($fscanf(fd, "%h %h %h %h %h %h %h\n", fa, fm, fs, fz, fl, fh, fw) == 7) begin
      t.acc = fa;
      t.mult = fm;
      t.shift = fs;
      t.zp = fz;
      t.amin = fl;
      t.amax = fh;
      t.want = fw;
      q.push_back(t);
      #1 valid = 1;
      {acc, mult, shift, zp, amin, amax} = {fa, fm, fs, fz, fl, fh};
      @(posedge clk);
      issued++;
      if (issued % 64 == 0) begin  // a bubble: the inputs change with valid_i low, and nothing may come out for it
        #1 valid = 0;
        {acc, mult} = {$urandom, $urandom};
        {shift, zp, amin, amax} = $urandom;
        @(posedge clk);
      end
    end
    $fclose(fd);
    #1 valid = 0;
    repeat (LAT + 4) @(posedge clk);
    if (issued == 0) begin
      errs++;
      $display("[FAIL] no vectors read from %s", vec);
    end
    if (checked != issued || q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d vectors issued, %0d results, %0d never came out", issued, checked, q.size());
    end
    if (lat != LAT) begin
      errs++;
      $display("[FAIL] latency %0d, expected %0d", lat, LAT);
    end
    $display("tfliteRequant ROUNDING=%s: %0d vectors from %s, %0d errors, latency %0d", ROUNDING, checked, vec, errs, lat);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`Requant/Makefile`:

```make
SHELL := /bin/bash
# tfliteRequant: verilator (corners + RANDOM vectors from ipu.requant, one ROUNDING), both, vivado_tb (committed vectors), vectors, lint.
PRJ_DIR  = $(shell pwd)
TB_DIR   = $(PRJ_DIR)/testbenches
PLUSARGS ?=
ROUNDING ?= DOUBLE
RANDOM   ?= 1000000
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent
RLOW     = $(shell echo $(ROUNDING) | tr A-Z a-z)
DESIGN   = $(PRJ_DIR)/src/tfliteRequant.sv
PKG      = $(PRJ_DIR)/../Common/src/sienna_fmt_pkg.sv
FLAGS    = --binary --timing --assert --sv -I$(TB_DIR) --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
GEN      = python3 $(TB_DIR)/gen_requant_vectors.py

verilator:
	$(GEN) $(PRJ_DIR)/Verilator/vec_$(RLOW).txt --rounding $(ROUNDING) --random $(RANDOM)
	verilator $(FLAGS) -GROUNDING='"$(ROUNDING)"' --top-module TB_tfliteRequant --Mdir $(PRJ_DIR)/Verilator/$(RLOW) \
	  $(PKG) $(DESIGN) $(TB_DIR)/TB_tfliteRequant.sv -o sim
	$(PRJ_DIR)/Verilator/$(RLOW)/sim +VEC=$(PRJ_DIR)/Verilator/vec_$(RLOW).txt $(PLUSARGS)

both:
	$(MAKE) verilator ROUNDING=DOUBLE
	$(MAKE) verilator ROUNDING=SINGLE

vivado_tb:
	for r in DOUBLE SINGLE; do l=$$(echo $$r | tr A-Z a-z); \
	  verilator $(FLAGS) -GROUNDING="\"$$r\"" --top-module TB_tfliteRequant --Mdir $(PRJ_DIR)/Verilator/viv_$$l \
	    $(PKG) $(DESIGN) $(TB_DIR)/TB_tfliteRequant.sv -o sim || exit 1; \
	  (cd $(TB_DIR) && $(PRJ_DIR)/Verilator/viv_$$l/sim $(PLUSARGS)) || exit 1; done

vectors:
	$(GEN) $(TB_DIR)/vectors_double.mem --rounding DOUBLE --random 10000
	$(GEN) $(TB_DIR)/vectors_single.mem --rounding SINGLE --random 10000

lint:
	for r in DOUBLE SINGLE; do \
	  verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module tfliteRequant -GROUNDING="\"$$r\"" $(DESIGN) || exit 1; done

lint_bad:
	verilator --lint-only -Wno-fatal -Werror-USERFATAL --top-module tfliteRequant -GROUNDING='"NEAREST"' $(DESIGN) \
	  > $(PRJ_DIR)/Verilator/lint_bad.txt 2>&1; rc=$$?; cat $(PRJ_DIR)/Verilator/lint_bad.txt; \
	  if grep -q "tfliteRequant: unsupported ROUNDING" $(PRJ_DIR)/Verilator/lint_bad.txt && [ $$rc -ne 0 ]; then echo REJECTED; \
	  else echo NOT-REJECTED; echo "RESULT: FAILED"; exit 1; fi

.PHONY: verilator both vivado_tb vectors lint lint_bad
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch_tree.sh i7_req 32 1 $J/cmds/int8_aril.sh Requant both`
Expected: the generator prints `... 33718 corners, ...`, then a Verilator error that `tfliteRequant.sv` cannot be found.

- [ ] **Step 3: Write the unit**

`Requant/src/tfliteRequant.sv`:

```systemverilog
`timescale 1ns / 100ps

// TFLite's int8 requantize, bit-exact: clamp(MultiplyByQuantizedMultiplier(acc, mult, shift) + zp, act_min, act_max).
// ROUNDING "DOUBLE": gemmlowp SaturatingRoundingDoublingHighMul then RoundingDivideByPOT; "SINGLE": TFLITE_SINGLE_ROUNDING.
// int32 intermediates wrap as TFLite's C does; shift_i in [-31, 30]; valid_i at t, done_o at t+3; only valid bits reset.
module tfliteRequant #(
    parameter string ROUNDING = "DOUBLE"
) (
    input  logic               clk_i,
    input  logic               rstn_i,
    input  logic               valid_i,
    input  logic signed [31:0] acc_i,      // int32 accumulator of output channel c
    input  logic signed [31:0] mult_i,     // Q0.31 multiplier M_c, TFLite's int32_t
    input  logic signed [7:0]  shift_i,    // shift_c: positive shifts left, negative right
    input  logic signed [7:0]  zp_i,       // output zero point
    input  logic signed [7:0]  act_min_i,  // clamp: the int8 range, ReLU or ReLU6
    input  logic signed [7:0]  act_max_i,
    output logic        [7:0]  result_o,
    output logic               done_o
);
  localparam bit SINGLE = (ROUNDING == "SINGLE");
  localparam logic signed [31:0] INT_MIN = 32'sh8000_0000;
  localparam logic signed [31:0] INT_MAX = 32'sh7FFF_FFFF;

  if ((ROUNDING != "SINGLE") && (ROUNDING != "DOUBLE")) begin : G_BAD_ROUNDING
    $fatal(1, "tfliteRequant: unsupported ROUNDING %s, use SINGLE or DOUBLE", ROUNDING);
  end

  logic s1_v, s2_v;
  logic signed [31:0] s1_x, s1_m;  // DOUBLE: acc * 2^left wrapped to 32 bits; SINGLE: acc
  logic [5:0] s1_sh, s2_sh;  // DOUBLE: right shift 0..31; SINGLE: 31 - shift, 1..62
  logic signed [7:0] s1_zp, s1_lo, s1_hi, s2_zp, s2_lo, s2_hi;
  logic signed [63:0] s2_p;  // DOUBLE: SaturatingRoundingDoublingHighMul (fits 32 bits); SINGLE: acc * mult

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      s1_v   <= 1'b0;
      s2_v   <= 1'b0;
      done_o <= 1'b0;
    end else begin
      s1_v   <= valid_i;
      s2_v   <= s1_v;
      done_o <= s2_v;
    end
  end

  // Stage 1: DOUBLE applies TFLite's int32 x * (1 << left_shift); SINGLE forms the total right shift.
  always_ff @(posedge clk_i)
    if (valid_i) begin
      s1_m  <= mult_i;
      s1_zp <= zp_i;
      s1_lo <= act_min_i;
      s1_hi <= act_max_i;
      if (SINGLE) begin
        s1_x  <= acc_i;
        s1_sh <= 6'(8'sd31 - shift_i);
      end else if (shift_i > 8'sd0) begin
        s1_x  <= acc_i <<< shift_i[4:0];
        s1_sh <= 6'd0;
      end else begin
        s1_x  <= acc_i;
        s1_sh <= 6'(-shift_i);
      end
    end

  // Stage 2: the 32 x 32 product; DOUBLE's floor((a*b + 2^30) / 2^31) equals gemmlowp's nudge and truncating divide.
  always_ff @(posedge clk_i)
    if (s1_v) begin
      automatic logic signed [63:0] ab = $signed({{32{s1_x[31]}}, s1_x}) * $signed({{32{s1_m[31]}}, s1_m});
      s2_zp <= s1_zp;
      s2_lo <= s1_lo;
      s2_hi <= s1_hi;
      s2_sh <= s1_sh;
      if (SINGLE) s2_p <= ab;
      else if ((s1_x == INT_MIN) && (s1_m == INT_MIN)) s2_p <= 64'(INT_MAX);  // the one product that overflows saturates
      else s2_p <= (ab + 64'sd1073741824) >>> 31;
    end

  // Stage 3: the rounding right shift, + zp in int32, then max with act_min and min with act_max, TFLite's order.
  always_ff @(posedge clk_i)
    if (s2_v) begin
      automatic logic signed [31:0] q, y;
      if (SINGLE) begin
        automatic logic signed [63:0] r = (s2_p + (64'sd1 <<< (s2_sh - 6'd1))) >>> s2_sh;
        q = r[31:0];  // static_cast<int32_t>: the low 32 bits
      end else begin
        automatic logic signed [31:0] x = s2_p[31:0];
        automatic logic [31:0] mask = (32'd1 << s2_sh) - 32'd1;
        automatic logic [31:0] thr = (mask >> 1) + 32'(x[31]);
        q = (x >>> s2_sh) + (((32'(x) & mask) > thr) ? 32'sd1 : 32'sd0);  // RoundingDivideByPOT: ties away from zero
      end
      y = q + 32'(s2_zp);
      if (y < 32'(s2_lo)) y = 32'(s2_lo);
      if (y > 32'(s2_hi)) y = 32'(s2_hi);
      result_o <= y[7:0];
    end

`ifndef SYNTHESIS
  always @(posedge clk_i)
    if (rstn_i && valid_i)
      assert ((shift_i >= -31) && (shift_i <= 30))
      else $error("tfliteRequant: shift %0d outside [-31, 30]", shift_i);
`endif

endmodule
```

- [ ] **Step 4: Run both roundings, 10^6 random each**

Run: `$J/snap_launch_tree.sh i7_req 32 1 $J/cmds/int8_aril.sh Requant both`
Expected in `$J/runs/i7_req/results/int8/Requant_both.log`, for each of DOUBLE and SINGLE:
- the generator line: `33718 corners`, a discriminator count close to 2000, `1000000 random`, a `differ from the other
  rounding` count at least the discriminator count, and a clamp-bound share near 57% (measured on a 200,000-vector draw while drafting; near 80% means the aimed vectors miss);
- `tfliteRequant ROUNDING=DOUBLE: N vectors from ..., 0 errors, latency 3` (N = 33,718 + discriminators + 1,000,000) and
  `RESULT: PASSED`; the same for SINGLE.

A mismatch is a bug in the RTL or in `ipu.py`; `ipu.py` already matches the C transcriptions (Task 1) and, for DOUBLE or
SINGLE, the interpreter (G0). Decide from the failing vector which stage's value differs: recompute it with
`ipu.srdhm` / `ipu.rdbpot` / `ipu.mbqm`. A latency other than 3 or a count mismatch is the valid pipeline.

- [ ] **Step 5: Lint, and the bad ROUNDING must fail elaboration**

```bash
$J/snap_launch_tree.sh i7_lint 32 1 $J/cmds/int8_aril.sh Requant lint
$J/snap_launch_tree.sh i7_bad 32 1 $J/cmds/int8_aril.sh Requant lint_bad
```

Expected: `lint` finishes for both roundings with no `LATCH`, `MULTIDRIVEN` or `UNOPTFLAT` warning (other warnings are
listed, not failing, as for the float units); `lint_bad` prints the `tfliteRequant: unsupported ROUNDING NEAREST` fatal
and `REJECTED`.

- [ ] **Step 6: Generate the committed vectors, run the Vivado-style testbench**

`/proj/work/spramanik/sienna_jobs/cmds/int8_req_vec.sh` (`chmod +x`):

```bash
#!/bin/bash
# Writes tfliteRequant's committed vector files on a snapshot and copies them to testbenches/results/int8; run from a snapshot root.
R=$(pwd)/testbenches/results/int8; mkdir -p $R
cd SystolicMesh/ArithmeticLibrary/Requant && make vectors && cp testbenches/vectors_double.mem testbenches/vectors_single.mem $R/
```

```bash
$J/snap_launch_tree.sh i7_vec 32 1 $J/cmds/int8_req_vec.sh
V=/proj/work/spramanik/SIENNA_int8/SystolicMesh/ArithmeticLibrary/Requant/testbenches
ls $V/vectors_*.mem 2>/dev/null && echo "EXISTS: look before copying"   # expect nothing
cp $J/runs/i7_vec/results/int8/vectors_double.mem $J/runs/i7_vec/results/int8/vectors_single.mem $V/
wc -l $V/vectors_*.mem
$J/snap_launch_tree.sh i7_viv 32 1 $J/cmds/int8_aril.sh Requant vivado_tb
```

Expected: each file has 33,718 + discriminators + 10,000 lines (about 45,700, about 1.4 MB); the testbench prints
`tfliteRequant ROUNDING=DOUBLE: N vectors from vectors_double.mem, 0 errors, latency 3` and the same for SINGLE, both
`RESULT: PASSED`.

- [ ] **Step 7: Commit (AriL)**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh/ArithmeticLibrary
git add Requant/src/tfliteRequant.sv && git commit -m "tfliteRequant: TFLite's int8 requantize, bit-exact, SINGLE or DOUBLE rounding, 3 stages"
git add Requant/testbenches/gen_requant_vectors.py Requant/testbenches/TB_tfliteRequant.sv Requant/Makefile && \
  git commit -m "TB_tfliteRequant: corners and 10^6 random vectors from ipu.requant, both roundings, lint"
git add Requant/testbenches/vectors_double.mem Requant/testbenches/vectors_single.mem && \
  git commit -m "tfliteRequant vector files for the Vivado testbench"
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
```

GPNAE's and SystolicMesh's ArithmeticLibrary pointers move in Tasks 9 and 13, after G1.

#### Limits of Tasks 1, 2 and 7

- The interpreter checks `ipu.requant` only through whole FC and conv layers with converter-made parameters (positive
  normalized multipliers, shifts about -20 to 0). INT32_MIN accumulators, positive shifts, negative multipliers and the
  int32 wraps are checked against Task 1's hand-derived values and the C transcriptions, not against the interpreter.
- G0 pins the rounding of the TensorFlow build Task 0 installs; another build (TFLite Micro, CMSIS-NN, a
  `TFLITE_SINGLE_ROUNDING` build) may differ, which is why tfliteRequant keeps both variants.
- The requantize unit's timing (a 32 x 32 multiply in stage 2, a 64-bit shift in SINGLE's stage 3) is not measured here.
- `mult_i` is signed, as TFLite's `int32_t` multiplier: QuantizeMultiplier only produces 0 or [2^30, 2^31 - 1], and the
  negative multipliers (needed for SRDHM's `INT32_MIN` saturation) are tested only for agreement with the formulas.

### Task 8: Gate G1, AriL int8

**Files:**
- Create (no repo): `$J/cmds/int8_aril_gate.sh`, `$J/cmds/int8_g1_compare.sh`
- Create: `$T/testbenches/results/int8/aril_gate.log` (generated report; `testbenches/results/` is gitignored in SIENNA, so the report is not committed here; Task 24 copies it into the `sienna-report` history as the bf16 G1 report was)

**Interfaces:**
- Consumes: everything in Tasks 3 to 7: the unit Makefiles (Int: `verilator`, `check`, `verilator16`, `check16`, `vivado_tb`, `lint`; Adders/Int and Fx: `verilator`, `check`, `vivado_tb`, `lint`; Requant: `both`, `vivado_tb`, `lint`, `lint_bad`), `$J/cmds/int8_aril.sh` (Task 0), `int8_aril_ri.sh` (Task 4), `int8_fpguard.sh` (Task 3). The bf16 G1 gate run `$J/runs/g1b` (Task 0's record) is the fp32 / bf16 reference, and `$J/cmd_aril_gate.sh` is the bf16 gate script, reused unchanged.
- Produces: the G1 verdict and report; the AriL `int8` head, pushed, that Level 2 (GPNAE's AriL pointer) and Level 3 (SystolicMesh's) move to.

- [ ] **Step 1: Write the gate script**

`$J/cmds/int8_aril_gate.sh` (`chmod +x`):

```bash
#!/bin/bash
# Gate G1 (int8): the new AriL units' checks, random power-up and lint, then the fp32 / bf16 AriL suites unchanged; run from a snapshot root of the int8 tree.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/int8; A=SystolicMesh/ArithmeticLibrary
mkdir -p $R
fail() { echo "GATE-FAIL $*"; }
[ -f $A/Multipliers/Int/src/intMultiplier.sv ] || { fail "not a snapshot of the int8 tree"; exit 1; }
$J/cmds/int8_aril.sh Common pkg || fail pkg
$J/cmds/int8_fpguard.sh > $R/fpguard.txt 2>&1 || fail fpguard  # D-7: the float units reject EXP_W = 0
cat $R/fpguard.txt
for u in Multipliers/Int Adders/Int Multipliers/Fx; do
  for t in verilator check vivado_tb lint; do $J/cmds/int8_aril.sh $u $t || fail $u $t; done
  for s in 2 7; do $J/cmds/int8_aril_ri.sh $u $s || fail randinit $u seed $s; done
done
for t in verilator16 check16; do $J/cmds/int8_aril.sh Multipliers/Int $t || fail Multipliers/Int $t; done  # W = 16, GPNAE's width
for t in both vivado_tb lint lint_bad; do $J/cmds/int8_aril.sh Requant $t || fail Requant $t; done
for s in 2 7; do
  TAG=randinit$s EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" \
    $J/cmds/int8_aril.sh Requant both +verilator+rand+reset+2 +verilator+seed+$s || fail randinit Requant seed $s
done
for u in Multipliers/Int Adders/Int Multipliers/Fx Requant; do
  L=$R/$(echo $u | tr / _)_lint.log
  grep -E "^%Error|^%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)" $L && fail lint $u
  echo "lint $u: $(grep -c '^%Warning' $L) warnings"
done
$J/cmd_aril_gate.sh > $(pwd)/testbenches/results/float_suite.log 2>&1  # the bf16 gate, unchanged; its logs land in results/uniform
grep "GATE-FAIL" $(pwd)/testbenches/results/float_suite.log && fail float suite
echo "int8 units:"; grep -h "RESULT:" $R/*.log | sort | uniq -c
echo "fp32 / bf16 units:"; grep -h "RESULT:" $(pwd)/testbenches/results/uniform/*.log | sort | uniq -c
```

`float_suite.log` sits outside `results/int8` on purpose: it holds the float runs' `RESULT:` lines, which would blur the int8 count.

- [ ] **Step 2: Run the gate from one snapshot**

Run: `TREE=$T $J/snap_launch_tree.sh i8_g1 32 6 $J/cmds/int8_aril_gate.sh`

Expected in `$J/runs/i8_g1/stdout.log`:
- no `GATE-FAIL` line; `REJECTED fpMultiplier` and `REJECTED fpAdder`;
- `lint <unit>: 0 warnings` for Multipliers/Int, Adders/Int and Multipliers/Fx, and for Requant what Task 7 Step 5 recorded, with no LATCH, MULTIDRIVEN, UNOPTFLAT or error; the Requant `lint_bad` log ends in `REJECTED`;
- int8 units: only `RESULT: PASSED` lines, 30 of them: 1 package; Multipliers/Int 9 (verilator 1, check 2, vivado_tb 1, two random power-up runs 2, verilator16 1, check16 2); Adders/Int and Multipliers/Fx 6 each (the same without W = 16); Requant 8 (`both` 2, `vivado_tb` 2, two random power-up runs of `both` 4);
- fp32 / bf16 units: `16 RESULT: PASSED`, as in the bf16 gate (`runs/g1b`).

If anything fails, stop. Find the cause with the systematic-debugging skill, fix it in the unit's files with a new commit (RTL first), and rerun the whole gate from a new snapshot. Never edit a log, lower a count or weaken a check to pass.

- [ ] **Step 3: fp32 / bf16 identical to the bf16 gate**

`$J/cmds/int8_g1_compare.sh` (`chmod +x`). It only reads logs, so it runs on the login node:

```bash
#!/bin/bash
# Compares the fp32 / bf16 AriL logs of an int8 G1 run with the bf16 G1 run (g1b): summary lines equal, lint warnings equal by type; args: RUN.
J=/proj/work/spramanik/sienna_jobs; B=$J/runs/g1b/results/uniform; N=$J/runs/$1/results/uniform
P="RESULT|SUCCESS|FAILURE|mismatches|errors|latency|checked|products|sums"
n=0; bad=0
for f in $(cd $B && ls *.log); do
  n=$((n + 1))
  if [ ! -f $N/$f ]; then echo "MISSING: $f"; bad=$((bad + 1)); continue; fi
  if [[ $f == *_lint.log ]]; then
    command diff <(command grep -oE "^%(Warning|Error)-[A-Z0-9_]+" $B/$f | sort | uniq -c) \
                 <(command grep -oE "^%(Warning|Error)-[A-Z0-9_]+" $N/$f | sort | uniq -c) > /dev/null || { echo "DIFFERS: $f"; bad=$((bad + 1)); }
  else
    command diff <(command grep -hE "$P" $B/$f) <(command grep -hE "$P" $N/$f) > /dev/null || { echo "DIFFERS: $f"; bad=$((bad + 1)); }
  fi
done
echo "compared $n logs with runs/g1b: $bad differ"
```

Then check that the float RTL and its DV are those of the bf16 branch (AriL `d97e270`, which the bf16 SIENNA pins) apart from the D-7 guards, and that the fxMac RTL is the one the sweeps proved:

```bash
$J/cmds/int8_g1_compare.sh i8_g1
cd $A
git diff --stat d97e270 -- Multipliers/FP Multipliers/FP32 Multipliers/FPWiden Multipliers/Karatsuba Multipliers/Radix4Booth \
  Adders/FP Adders/FP32 Common/testbenches/fp_stim.svh Common/testbenches/fp_ref_core.h Common/testbenches/fp_ref.cpp \
  Common/testbenches/gen_vectors.cpp Common/testbenches/generate_vectors.sh Common/testbenches/TB_fp32Dump.sv \
  Common/models/fpu.py Common/models/check_fpu.py Common/Makefile
git diff d97e270 -- Common/src/sienna_fmt_pkg.sv
for s in $J/snaps/i6_[AX][XC]_*_[0-3].tar; do tar -xOf $s ./SystolicMesh/ArithmeticLibrary/Multipliers/Fx/src/fxMac.sv | command cmp -s - Multipliers/Fx/src/fxMac.sv || echo "fxMac differs from sweep $(basename $s)"; done
```

Expected: `compared 15 logs with runs/g1b: 0 differ`. Both runs use Verilator's default seed, so even the random power-up logs (the D-1 count of the bf16 plan in `Adders_FP_randinit.log`) must match; a difference there is investigated, not waived. The first `git diff --stat` shows only `Multipliers/FP/src/fpMultiplier.sv` and `Adders/FP/src/fpAdder.sv`, 5 insertions each (Task 3's guard and the blank line before it). The package diff shows only Task 3's int8 additions (`is_int`, `acc_w`, the int8 latencies, `fx_lat`, `req_lat`, `REQ_ROUNDING`): the fp32 and bf16 branches of `supported`, `mul_lat` and `add_lat` still return what they did, which `Common_pkg.log` confirms. The sweep loop prints nothing: all 24 sweep snapshots hold the fxMac RTL that is being gated. `Common_pkg.log` is among the compared logs, and its summary line (`TB_sienna_fmt_pkg: 0 errors`) must still match. If it prints a line, relaunch the 24 sweeps of Task 6 Step 7 on the new RTL before writing the report.

- [ ] **Step 4: Write the gate report**

Write `$T/testbenches/results/int8/aril_gate.log` by hand from the job outputs, in the layout of the bf16 G1 report (`sienna-report` history `2026-09-28_aril_gate.txt`). Copy every number from a log and give the log's path next to it:

- header: AriL `int8` commit (pushed), gate run `sienna_jobs/runs/i8_g1`, Verilator version; `VERDICT: PASS` or `FAIL` with the failing checks;
- 1, format package (Task 3): `TB_sienna_fmt_pkg` errors, the int8 cases it checks;
- 1b, D-7: `int8_fpguard.sh` (`fpMultiplier` and `fpAdder` reject `EXP_W = 0`) and the guard's diff against `d97e270`;
- 2, intMultiplier: products against the C reference (65,536), errors, latency; `check_ipu` mismatches and distinct pairs; the W = 16 run (1,000,121 products, `check_ipu mul16`); Vivado vectors checked; random power-up (seeds 2 and 7); lint warnings;
- 3, intAdder: sums, errors, latency, the six wrap corners by name; model mismatches; Vivado; random power-up; lint;
- 4, fxMac: default-stimulus results, errors, latency; model mismatches; Vivado; random power-up; lint; the six sweeps (a table: sweep, results, errors, digest mismatches, slice times from `runs/i6_*`), with the fxMac commit the sweeps proved and the Step 3 check that every sweep snapshot holds the gated RTL; what the sweeps do not cover (Task 6);
- 5, tfliteRequant (Task 7): vectors per rounding (corners, discriminators, random), errors, latency; Vivado; random power-up; lint; `lint_bad` rejected;
- 6, fp32 / bf16 units unchanged: the RTL diff against `d97e270` (the D-7 guards only); `int8_g1_compare.sh` (N logs, 0 differ); the fp32 unit TBs' `SUCCESS` counts (2003 / 2007 in g1b), fpMulWiden;
- 7, findings in ipu.py or the published units, reported and not changed (none expected);
- NOT RUN AT THIS LEVEL: VCS; Vivado itself (the Vivado testbenches ran in Verilator); synthesis of the new units; the other AriL copy, `GPNAE/ArithmeticLibrary`, which stays at `d97e270` until Level 2 moves its pointer.

- [ ] **Step 5: Commit and push (AriL)**

The unit tasks committed their files. Commit here only fixes the gate needed, one per commit, RTL first. Then confirm that the remote head is the tested head:

```bash
cd $A
git status -s          # expect only untracked build output (Verilator/, __pycache__/)
git log --oneline d97e270..HEAD
git push git@github.com:SoHam-56/ArithmeticLibrary.git int8
test "$(git ls-remote git@github.com:SoHam-56/ArithmeticLibrary.git refs/heads/int8 | cut -f1)" = "$(git rev-parse HEAD)" && echo PUSHED
```

Expected: `PUSHED`. `git log` lists Task 3's, Task 4's, Task 5's, Task 6's and Task 7's commits and nothing else. The SystolicMesh and GPNAE pointers to AriL move in Levels 3 and 2, not here.

- [ ] **Step 6: Report to Soham and wait**

Give the verdict, the report's path, and the fxMac sweep coverage (which triples were exhaustive and which were sampled). Level 2 starts only after Soham has seen G1.

---

## Level 2: GPNAE (gate G2 in Task 12)

All paths are relative to `/proj/work/spramanik/SIENNA_int8/GPNAE` on branch `int8` unless stated. Every farm command
runs from `/proj/work/spramanik/SIENNA_int8` with:

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/work/spramanik/SIENNA_int8
export TREE=$T
```

Job scripts live in `$J/cmds/` (no repo), are created with `chmod +x`, and are launched with
`$J/snap_launch_tree.sh NAME MEM_GB HOURS script args...`; output lands in `$J/runs/NAME/` (`stdout.log`, and
`results/` copied from the snapshot's `testbenches/results`). `$J/cmd_gpnae_reg.sh` (exists) runs `GPNAE/regression.py`.
`$J/cmds/int8_fpref.sh` (Task 0) reruns the five fp32 / bf16 GPNAE runs; Task 0's run of it on the bf16 tree,
`$J/runs/i0_fpref`, is the reference every step of this level compares with.

### How the int8 lane computes (what Tasks 9 to 12 build)

Soham (2026-09-29): "for gpnae you can just use different mult and adders, int8 ones and quantize". The int8 lane keeps
the float lane's forms exactly (`gpnae_poly.sv`, `gpnae_model.Lane.poly`) and swaps the units: Horner on fxMac in
Q4.11 (D-1: floor), integer multiplies for the post stage, then int8 quantization. There is no gpnae_tail (D-4).

| Activation | MAC operand (Q4.11) | Polynomial | Post stage (as the float lane) | Past the fitted range | Output (int8) |
|---|---|---|---|---|---|
| SELU (001), x < 0 | x | `P(x) ~ lambda*alpha*(e^x - 1)/x` on [-7, 0] | `x * P` (32-bit product, 2^-22) | x < -7: `-lambda*alpha` | `tfliteRequant(v, gp_mout, gp_shout, gp_zout)`, v in 2^-25 |
| SELU, x >= 0 | (discarded) | none | `x * lambda` (Q1.14, exact product, 2^-25) | none | same requantize |
| sigmoid (010) | `abs(x)` | `P(|x|) ~ sig` on [0, 6.25] | x >= 0: `P`; x < 0: `1 - P` (`2048 - P`, exact) | abs(x) > 6.25: 127 / -128 | `round(256 y) - 128 = ((y + 4) >> 3) - 128`, clamped |
| tanh (011) | `u = sat((x * x) >>> 11)` (an fxMac with C = 0) | `P(u) ~ tanh(sqrt u)/sqrt u` on u in [0, 9.77] | `x * P` (32-bit product, 2^-22) | abs(x) > 3.125: 127 / -128 | `round(128 y) = (x*P + 2^14) >> 15`, clamped |
| ReLU (100), linear (101) | none | none | none | none | the input, unchanged (D-4) |

Input rescale for every activation: `x = sat16(round_half_up((q - z_in) * gp_mx / 2^gp_shx))`, Q4.11.

Fitted ranges. The float lane hands inputs past SELU -4, sigmoid 3.5 and tanh 4 to gpnae_tail; int8 has no tail, so
its polynomial must reach the point where the exact int8 output is already the saturated value. tanh: `128 tanh(x)`
passes 127.5 at 3.1182, so 3.125 (narrower than the float lane's 4). sigmoid: `256 sig(x)` passes 255.5 at 6.2364, so
6.25 (wider than 3.5: saturating at 3.5 would give 255 where the exact value is 248.5). SELU: `lambda*alpha*e^-7 =
0.0016`, 0.19 LSB at the tightest test case (s_out 0.0085), where -4 would leave 0.032, 3.8 LSB. Task 10 also measures
the float lane's ranges, for the record.

Degrees are free (Task 10 picks the lowest that meets the target, 2 to 12 measured). The coefficient ROM keeps the fp32
layout (SELU base 0, sigmoid base 9, tanh base 16) when the chosen degrees fit it (at most 8, 6 and 15), else the three
sets are packed in that order; the layout is `gpnae_model.SETS_INT8` and gpnae_poly_int8's `BASE_*`/`DEG_*`.

Task 10 measures first. If a form cannot reach at most 1 int8 LSB on every input of every gated test case, Task 10
stops with a report for Soham (measured errors and three options) and nothing further in this level is built.

### Decisions this level makes (flag at review)

- **L2-1: the int8 lane is its own module**, `src/gpnae_poly_int8.sv`, instantiated by `gpnae_poly` in a `G_INT8`
  generate branch; the float body of `gpnae_poly.sv` is wrapped, unchanged and not re-indented, in `G_FLOAT`. The float
  lane's hierarchy moves from `dut.G_MAC` to `dut.G_FLOAT.G_MAC` (only TB_gpnae_poly references it; checked with
  `grep -rn "G_MAC\|barrel_mac_inst"`).
- **L2-2: `gp_mx_i < 2^15`**, so the rescale uses `intMultiplier #(.W(16))` like the post multiply; the host picks the
  largest `shx <= 31` with `mx <= 32767` (relative precision at worst 2^-15).
- **L2-3: the post-stage products are full 32-bit products** (intMultiplier), quantized once (tanh) or requantized
  (SELU), rather than shifted back to Q4.11 first; the MAC and the tanh squarer are fxMac (Q4.11, floor).
- **L2-4: the target** is at most 1 int8 LSB after output quantization on every int8 input of every gated case in
  `gpnae_model.INT8_CASES`; the error over every Q4.11 input of the fitted range is reported beside it (SELU in LSB of
  the tightest gated case).
- **L2-5: `fit_poly_coeffs.py` refuses bf16 as well as fp32**, so the published bf16 table cannot be rewritten.

---

### Task 9: `barrel_mac` int8 branch on `fxMac`

**Files:**
- Modify: `ArithmeticLibrary` (submodule pointer to the AriL `int8` tip after G1), `Makefile` (new AriL files)
- Modify: `src/TYTAN/barrel_mac.sv`
- Create: `testbenches/TB_barrel_mac_int8.sv`, `barrel_mac_int8_vectors.py`
- Create (no repo): `$J/cmds/int8_cmp_gpnae.sh`, `$J/cmds/int8_lint.sh`, `$J/cmds/int8_bm.sh`

**Interfaces:**
- Consumes: `sienna_fmt_pkg::is_int`, `supported(0, 7)`, `fx_lat()` = 2 (Task 3); `fxMac #(W, FRAC)` with `clk_i,
  rstn_i, valid_i, A, X, C, result_o, done_o`, latency `fx_lat()`, valid bits reset (Task 6); `ipu.fx_mac(a, x, c, w, frac)` (Task 1); the AriL `int8`
  tip that passed G1 (Task 8); `$J/cmds/int8_fpref.sh` and its reference run `$J/runs/i0_fpref/results/int8/fpref/`
  (Task 0).
- Produces:
  - `barrel_mac #(EXP_W, MAN_W, DATA_WIDTH = is_int(EXP_W) ? 16 : 1 + EXP_W + MAN_W, ADDR_LINES, K, INIT_FILE)`, ports
    unchanged. In int8 it evaluates `acc = sat16(((x * acc) >>> 11) + c)` per round, highest coefficient first, loop 3
    cycles (a register stage, then fxMac), `K >= 4`; `DATA_WIDTH != 16` fails elaboration.
  - `$J/cmds/int8_cmp_gpnae.sh REF_DIR NEW_DIR` compares two `int8_fpref.sh` result sets' `RESULT`, `ERRSTAT` and
    `CYCLES` lines tag by tag. `$J/cmds/int8_lint.sh TOP "FMT" [PATTERN]` elaborates one GPNAE block.
    `$J/cmds/int8_bm.sh [SEED]` runs TB_barrel_mac_int8.
  - `GPNAE/ArithmeticLibrary` on the same AriL commit as `SystolicMesh/ArithmeticLibrary` (the G1 tip), from here on.

- [ ] **Step 1: Write the job scripts**

`$J/cmds/int8_fpref.sh` is Task 0's. `$J/cmds/int8_cmp_gpnae.sh` (runs on the login node: it only reads logs):

```bash
#!/bin/bash
# Compares the RESULT, ERRSTAT and CYCLES lines of two int8_fpref.sh result sets, tag by tag; args: REF_DIR NEW_DIR.
lines() { for f in $(ls $1/*.log 2>/dev/null | command grep -v gpnae_report); do command grep -hE "RESULT [a-z]|ERRSTAT |CYCLES " $f; done; }
rc=0
for d in $1/*/; do
  t=$(basename $d); n=$(lines $1/$t | wc -l)
  if [ "$n" -gt 0 ] && command diff <(lines $1/$t) <(lines $2/$t) > /dev/null; then echo "IDENTICAL $t ($n lines)"
  else echo "DIFFERENT $t"; command diff <(lines $1/$t) <(lines $2/$t) | head -20; rc=1; fi
done
exit $rc
```

`$J/cmds/int8_lint.sh`:

```bash
#!/bin/bash
# Elaborates one GPNAE block with -G overrides; args: TOP FMT [PATTERN]; with PATTERN prints REJECTED when the build fails with it.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
cd GPNAE || exit 1
make lint_fmt TOP=$1 FMT="$2" > lint_fmt.txt 2>&1; rc=$?
cat lint_fmt.txt
if [ -n "$3" ]; then command grep -q "$3" lint_fmt.txt && [ $rc -ne 0 ] && echo REJECTED || echo NOT-REJECTED; fi
echo "make exit=$rc, errors=$(command grep -c '^%Error' lint_fmt.txt)"
```

`$J/cmds/int8_bm.sh`:

```bash
#!/bin/bash
# TB_barrel_mac_int8 on a snapshot: vectors from ipu.fx_mac, then Verilator (EXTRA_FLAGS/SIM_ARGS from the environment); args: [SEED].
export VERILATOR_ROOT="$HOME/.local/share/verilator"
R=$(pwd)/testbenches/results/int8; mkdir -p $R
cd GPNAE && python3 barrel_mac_int8_vectors.py ${1:-1} || exit 1
make verilator TESTBENCH=TB_barrel_mac_int8.sv TOP_MODULE=TB_barrel_mac_int8 2>&1 | tee $R/bm_int8_seed${1:-1}.log
command grep -q "RESULT: PASSED" $R/bm_int8_seed${1:-1}.log
```

- [ ] **Step 2: Check the bf16-branch reference**

Task 0 Step 5 ran it on the clean bf16 tree (`TREE=/proj/work/spramanik/SIENNA $J/snap_launch_tree.sh i0_fpref 32 6
$J/cmds/int8_fpref.sh`). Before Step 3, check `runs/i0_fpref/stdout.log`: five `RESULT: PASSED` lines and no
`RUN-FAIL`.

- [ ] **Step 3: Bump the AriL submodule and add the new units to the Makefile**

```bash
cd /proj/work/spramanik/SIENNA_int8/GPNAE/ArithmeticLibrary
git fetch origin int8 && git checkout FETCH_HEAD
test "$(git rev-parse HEAD)" = "$(git -C ../../SystolicMesh/ArithmeticLibrary rev-parse HEAD)" && echo SAME-TIP || echo DIFFERENT-TIP
cd ..
```

Expected: `SAME-TIP` (the tip Task 8 passed). From this step on both AriL checkouts, `SystolicMesh/ArithmeticLibrary`
and `GPNAE/ArithmeticLibrary`, name the same commit: SIENNA compiles both copies of the units (`--Wno-MODDUP` hides
whichever copy loses) and its goldens import `ipu.py` through both, so a later AriL change moves both pointers
together. Every later gate checks it (G2 Task 12, G3 Task 15, G4 Task 23). In `Makefile`'s `DESIGN_FILES`, after `../ArithmeticLibrary/Adders/FP/src/fpAdder.sv \`, add:

```make
	../ArithmeticLibrary/Multipliers/Int/src/intMultiplier.sv \
	../ArithmeticLibrary/Multipliers/Fx/src/fxMac.sv \
	../ArithmeticLibrary/Requant/src/tfliteRequant.sv \
```

If `fxMac.sv` or `tfliteRequant.sv` instantiate another AriL module, add that module's file too; Task 11's lint names
any missing one (`Cannot find file containing module`).

```bash
$J/snap_launch_tree.sh i9_bump 32 6 $J/cmds/int8_fpref.sh
$J/cmds/int8_cmp_gpnae.sh $J/runs/i0_fpref/results/int8/fpref $J/runs/i9_bump/results/int8/fpref
```

Expected: `IDENTICAL` for all five tags. A difference comes from the submodule move; report it and stop.

```bash
git add ArithmeticLibrary && git commit -m "Bump ArithmeticLibrary: int8 format package, intMultiplier, intAdder, fxMac, tfliteRequant"
git add Makefile && git commit -m "Makefile: intMultiplier, fxMac, tfliteRequant"
git push origin int8
```

- [ ] **Step 4: Write the failing unit test**

`barrel_mac_int8_vectors.py`:

```python
#!/usr/bin/env python3
"""Vectors for TB_barrel_mac_int8: a random Q4.11 coefficient ROM, 256 operand groups, and each slot's Horner result on ipu.fx_mac."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402

K, NG = 16, 256
OUT = os.path.join(ROOT, "testbenches", "stimulus")


def main():
    rs = np.random.RandomState(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    rom = rs.randint(-32768, 32768, 32).astype(np.int64)
    rom[:8] = [32767, -32768, 0, 1, -1, 2048, -2048, 16384]  # extremes, zero, one ULP, +/-1.0, 8.0
    sizes = [1, 2, 3, 4, 5, 8, 15, 16]  # below, at and above the 4-cycle minimum round, and a full barrel
    hdr, opd, exp = [], [], []
    for g in range(NG):
        n = sizes[g] if g < len(sizes) else int(rs.randint(1, K + 1))
        deg = int(rs.randint(0, 9))
        base = int(rs.randint(0, 32 - deg))
        if g % 3 == 0:
            x = rs.randint(-32768, 32768, n)  # full range: products and sums saturate
        elif g % 3 == 1:
            x = rs.randint(-2048, 2049, n)  # abs(t) <= 1, the lane's operands
        else:
            x = rs.choice([32767, -32768, 0, 1, -1, 2048, -2048, 16384, -16384], n)
        x = np.asarray(x, dtype=np.int64)
        acc = np.zeros_like(x)
        for r in range(deg + 1):  # barrel_mac: fxMac(A=operand, X=acc, C=coefficient), highest coefficient first
            acc = ipu.fx_mac(x, acc, np.full_like(x, rom[base + deg - r]), w=16, frac=11)
        hdr.append((n << 16) | (deg << 8) | base)
        opd += [int(v) for v in x] + [0] * (K - n)
        exp += [int(v) for v in acc] + [0] * (K - n)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bm_int8_rom.mem"), "w") as f:
        f.write("".join(format(int(v) & 0xFFFF, "016b") + "\n" for v in rom))
    with open(os.path.join(OUT, "bm_int8_hdr.mem"), "w") as f:
        f.write("".join(f"{h:08x}\n" for h in hdr))
    for name, vals in (("in", opd), ("exp", exp)):
        with open(os.path.join(OUT, f"bm_int8_{name}.mem"), "w") as f:
            f.write("".join(f"{v & 0xFFFF:04x}\n" for v in vals))
    print(f"bm_int8 vectors: {NG} groups, ROM, operands and results in {OUT}")


if __name__ == "__main__":
    main()
```

`testbenches/TB_barrel_mac_int8.sv`:

```systemverilog
`timescale 1ns / 100ps

// barrel_mac's int8 branch against ipu.fx_mac Horner (barrel_mac_int8_vectors.py), bit for bit, with done/busy checks.
module TB_barrel_mac_int8;
  localparam int NG = 256;  // groups, as the vector script writes them
  localparam int K = 16;
  localparam int W = 16;

  logic clk = 1'b0;
  logic rstn = 1'b0;  // reset held from power-up
  logic ld_valid = 1'b0, start = 1'b0;
  logic [W-1:0] ld_data = '0;
  logic [4:0] terms = '0, base = '0;
  logic res_valid, busy, done;
  logic [W-1:0] res_data;

  logic [31:0] hdr[NG];
  logic [W-1:0] opd[NG*K];
  logic [W-1:0] want[NG*K];

  barrel_mac #(
      .EXP_W(0), .MAN_W(7), .DATA_WIDTH(W), .ADDR_LINES(5), .K(K),
      .INIT_FILE("testbenches/stimulus/bm_int8_rom.mem")
  ) dut (
      .clk_i(clk), .rstn_i(rstn), .ld_valid_i(ld_valid), .ld_data_i(ld_data), .start_i(start),
      .terms_i(terms), .coeff_base_i(base), .res_valid_o(res_valid), .res_data_o(res_data),
      .busy_o(busy), .done_o(done)
  );

  always #5 clk = ~clk;

  // One context drives and samples, 1 ns after each edge: Verilator's threaded scheduler is not coherent across contexts.
  task automatic tick();
    @(posedge clk);
    #1;
  endtask

  initial begin
    int n, errs, got, guard, shown;
    time t0;
    errs = 0;
    shown = 0;
    $readmemh("testbenches/stimulus/bm_int8_hdr.mem", hdr);
    $readmemh("testbenches/stimulus/bm_int8_in.mem", opd);
    $readmemh("testbenches/stimulus/bm_int8_exp.mem", want);
    repeat (8) tick();
    rstn = 1'b1;
    tick();
    t0 = $time;
    for (int g = 0; g < NG; g++) begin
      n     = int'(hdr[g][23:16]);
      terms = hdr[g][12:8];
      base  = hdr[g][4:0];
      for (int i = 0; i < n; i++) begin
        ld_valid = 1'b1;
        ld_data  = opd[g*K+i];
        tick();
      end
      ld_valid = 1'b0;
      start    = 1'b1;
      tick();
      start = 1'b0;
      got   = 0;
      guard = 0;
      while (got < n && guard < 1000) begin
        tick();
        guard++;
        if (res_valid) begin
          if (res_data !== want[g*K+got]) begin
            errs++;
            if (shown++ < 20)
              $display("[FAIL] group %0d slot %0d (n %0d degree %0d base %0d operand %h): got %h, want %h", g, got, n,
                       terms, base, opd[g*K+got], res_data, want[g*K+got]);
          end
          if ((got == n - 1) != done) begin
            errs++;
            if (shown++ < 20) $display("[FAIL] group %0d slot %0d: done_o %b", g, got, done);
          end
          got++;
        end
      end
      if (got != n) begin
        errs++;
        $display("[FAIL] group %0d: %0d of %0d results", g, got, n);
      end
      tick();
      if (busy) begin
        errs++;
        $display("[FAIL] group %0d: busy_o high after the last result", g);
      end
    end
    $display("TB_barrel_mac_int8: %0d groups, %0d errors, %0d cycles", NG, errs, ($time - t0) / 10);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 5: Run it to see it fail**

```bash
$J/snap_launch_tree.sh i9_bm 32 1 $J/cmds/int8_bm.sh 1
$J/snap_launch_tree.sh i9_bmw 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=0 -GMAN_W=7 -GDATA_WIDTH=8" "barrel_mac: int8 evaluates in Q4.11"
```

Expected now: `i9_bm` fails (the int8 format takes the `G_FP` branch: `fpMultiplier` at `EXP_W = 0` gives build errors or
`RESULT: FAILED`), and `i9_bmw` prints `NOT-REJECTED`.

- [ ] **Step 6: Add the int8 branch**

In `src/TYTAN/barrel_mac.sv`, add below the header's `// K must be >= 14 ...` line:

```systemverilog
// int8 (EXP_W = 0): one fxMac, a Q4.11 multiply-add in 2 cycles, behind a register stage; the loop is 3 cycles, K >= 4.
```

Replace the `DATA_WIDTH` parameter line with:

```systemverilog
    parameter int DATA_WIDTH = sienna_fmt_pkg::is_int(EXP_W) ? 16 : 1 + EXP_W + MAN_W,  // int8 works in Q4.11
```

Replace the four localparams `MUL_LAT` .. `MIN_PER` with:

```systemverilog
  localparam bit INT     = sienna_fmt_pkg::is_int(EXP_W);  // int8: Q4.11 on fxMac
  localparam int FX_LAT  = sienna_fmt_pkg::fx_lat();  // fxMac valid_i -> done_o
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i -> done_o of the format's multiplier
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // and adder
  localparam int LOOP    = INT ? FX_LAT + 1 : MUL_LAT + ADD_LAT;  // the Horner recurrence: 13 in fp32, 8 in bf16, 3 in int8
  localparam int MIN_PER = LOOP + 1;               // write-back lands a cycle after that
```

Replace the `coeff_addr_full` assignment (`assign coeff_addr_full = ... rnd_dly[MUL_LAT-2];`) with:

```systemverilog
  if (INT) begin : G_ADDR_INT
    // int8: addressed at issue, so the coefficient lands with the registered operands a cycle later.
    assign coeff_addr_full = {1'b0, coeff_base_i} + ncoef - 1 - round;
  end else begin : G_ADDR_FP
    assign coeff_addr_full = {1'b0, coeff_base_i} + ncoef - 1 - rnd_dly[MUL_LAT-2];
  end
```

In the unit block, insert this branch between `G_BAD_FORMAT` and `G_FP32` (the `end else if (sienna_fmt_pkg::is_fp32(...))`
line stays as it is):

```systemverilog
  end else if (INT) begin : G_INT
    if (DATA_WIDTH != 16) begin : G_BAD_WIDTH
      $fatal(1, "barrel_mac: int8 evaluates in Q4.11, DATA_WIDTH must be 16, not %0d", DATA_WIDTH);
    end
    // One Horner step, sat(((x * acc) >>> 11) + c), on the slot's operand and accumulator read at stage 0.
    fxMac #(.W(DATA_WIDTH), .FRAC(11)) MAC (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_dly[0]),
        .A(xop[slot_dly[0][SW-1:0]]), .X(acc[slot_dly[0][SW-1:0]]), .C(coeff_data),
        .result_o(add_res), .done_o(add_done)
    );
    assign mul_res  = '0;
    assign mul_done = 1'b0;
`ifndef SYNTHESIS
    always @(posedge clk_i)
      if (rstn_i && (add_done != v_dly[LOOP-1])) $fatal(1, "barrel_mac: fxMac latency is not %0d", FX_LAT);
`endif
```

Why this timing is right (with `fx_lat()` = 2): a slot issues at cycle t (`mul_valid`, `round`); the ROM, addressed from `round` at t, gives
the coefficient at t+1, where fxMac takes the operand, the accumulator and the coefficient (`v_dly[0]`); the sum is
back at t+3 (`v_dly[2] = v_dly[LOOP-1]`) and written to `acc` at that edge, which is before the slot's next issue at
t+4 or later (`period >= MIN_PER = 4`). The existing write-back line (`if (v_dly[LOOP-1]) acc[...] <= add_res;`),
`DRAIN` (`LOOP + 1`) and `EMIT` serve both branches unchanged. fp32 and bf16 keep the same `LOOP`, the same address
expression and the same units.

- [ ] **Step 7: Run the checks**

```bash
$J/snap_launch_tree.sh i9_bm 32 1 $J/cmds/int8_bm.sh 1
$J/snap_launch_tree.sh i9_bm2 32 1 $J/cmds/int8_bm.sh 2
$J/snap_launch_tree.sh i9_bmw 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=0 -GMAN_W=7 -GDATA_WIDTH=8" "barrel_mac: int8 evaluates in Q4.11"
$J/snap_launch_tree.sh i9_bad 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=5 -GMAN_W=10" "barrel_mac: unsupported format"
$J/snap_launch_tree.sh i9_l8 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=0 -GMAN_W=7"
$J/snap_launch_tree.sh i9_l16 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=8 -GMAN_W=7"
$J/snap_launch_tree.sh i9_l32 32 1 $J/cmds/int8_lint.sh barrel_mac "-GEXP_W=8 -GMAN_W=23"
$J/snap_launch_tree.sh i9_fp 32 6 $J/cmds/int8_fpref.sh
```

Then `$J/cmds/int8_cmp_gpnae.sh $J/runs/i0_fpref/results/int8/fpref $J/runs/i9_fp/results/int8/fpref`.

Expected:
- `i9_bm`, `i9_bm2`: `TB_barrel_mac_int8: 256 groups, 0 errors`, `RESULT: PASSED`, no latency `$fatal`;
- `i9_bmw`, `i9_bad`: `REJECTED`;
- `i9_l8`, `i9_l16`, `i9_l32`: `errors=0`;
- the compare: `IDENTICAL` for all five tags.

A TB mismatch with the latency check silent points at the ROM address timing or the operand read stage; compare the
failing slot's degree and base with `rom[base + deg - r]` in the vector script.

- [ ] **Step 8: Commit (GPNAE)**

```bash
git add src/TYTAN/barrel_mac.sv && git commit -m "barrel_mac: int8 branch, one fxMac Horner step per round in Q4.11"
git add testbenches/TB_barrel_mac_int8.sv barrel_mac_int8_vectors.py && git commit -m "TB_barrel_mac_int8: int8 barrel MAC against ipu.fx_mac, bit for bit"
git push origin int8
```

### Task 10: int8 lane model, measured fit of the float forms, `poly_coeffs_int8.mem`

**Files:**
- Modify: `gpnae_model.py` (int8 lane), `fit_poly_coeffs.py` (`--format int8`, bf16 refused)
- Create: `check_gpnae_model_int8.py`; if the target is met, `poly_coeffs_int8.mem`, `src/TYTAN/Memory/poly_coeffs_int8.mem`
- Create (no repo): `$J/cmds/int8_py.sh`, `$J/cmds/int8_fit.sh`
- Create (untracked, only if the target is missed): `testbenches/results/int8/gpnae_int8_stop.log` (SIENNA root)

**Interfaces:**
- Consumes: `ipu.fx_mac`, `ipu.int_mul(a, b, w)`, `ipu.requant(acc, mult, shift, zp, amin, amax, rounding)`,
  `ipu.REQ_ROUNDING` (Tasks 1 and 3).
- Produces (in `gpnae_model`):
  - `INT8` (name `"int8"`, `w = 8`, `frac = 11`, `iw = 16`); `FORMATS = dict(fpu.FORMATS, int8=INT8)`.
  - `Lane(FORMATS["int8"], rom, sets=None, thresh=None)` returns a `LaneInt8`: `.run(q, code, par) -> np.ndarray`
    (int8 values, any shape), `.element(q, code, par) -> int`, `.poly(x, code)` (P at the float lane's MAC operand),
    `.value(x, code)` (the unsaturated path as a real number before quantizing). Codes 1 SELU, 2 sigmoid, 3 tanh,
    4 ReLU, 5 linear; others run tanh. `sets`/`thresh` override the layout and the saturation thresholds (the fit uses
    them for candidates).
  - `Int8Params(mx, shx, zin, mout, shout, zout)`; `Case(s_in, z_in, s_out, z_out, gated)`; `INT8_CASES` (five cases
    per activation name); `int8_params(case, code)`; `exact_int8(q, code, case)`; `rescale_params(s_in) -> (mx, shx)`;
    `quantize_multiplier(real) -> (mult, shift)` (TFLite's QuantizeMultiplier); `calib_out`, `selu_case`.
  - Functions `rescale`, `mac_operand`, `horner`, `quant_sig`, `quant_tanh`; constants `Q, SETS_INT8, THRESH, T_SELU,
    T_SIG, T_TANH, SELU_SAT, ONE_Q11, LAMBDA_Q14, LAMBDA_F, LA_F`. Task 11's RTL uses the same constants.
  - `python3 fit_poly_coeffs.py --format int8 [--report FILE]`: exit 0, target met and the table written; 3, target
    met but `SETS_INT8` must be updated to the printed layout (table written); 2, target missed, no table, stop report.

- [ ] **Step 1: Write the failing model test**

`check_gpnae_model_int8.py`:

```python
#!/usr/bin/env python3
"""Known values for gpnae_model's int8 lane: rounding and saturation of each step, the fixed output quantizations, SELU through the requantizer."""
import sys

import numpy as np

import gpnae_model as gm

errs = 0


def eq(what, got, want):
    global errs
    got = got if isinstance(got, tuple) else [int(v) for v in np.atleast_1d(got)]
    if got != want:
        errs += 1
        print(f"[FAIL] {what}: got {got}, want {want}")


eq("rescale rounds half up", gm.rescale(np.array([3, -3, 1, -1]), 0, 1, 1), [2, -1, 1, 0])
eq("rescale saturates to Q4.11", gm.rescale(np.array([127, -128]), 0, 32767, 0), [32767, -32768])
eq("rescale subtracts the zero point", gm.rescale(np.array([5]), 5, 1000, 3), [0])
eq("rescale by 2^31", gm.rescale(np.array([127]), -128, 32767, 31), [0])
eq("SELU MAC operand is x", gm.mac_operand(np.array([-5, 7]), 1), [-5, 7])
eq("sigmoid MAC operand is |x|, saturated", gm.mac_operand(np.array([-32768, -5, 5]), 2), [32767, 5, 5])
eq("tanh MAC operand is x^2 on fxMac", gm.mac_operand(np.array([2048, -2048, 6400, 32767]), 3), [2048, 2048, 20000, 32767])
eq("sigmoid output rounding and clamp", gm.quant_sig(np.array([2048, 0, 4, 3, -9])), [127, -128, -127, -128, -128])
eq("tanh output from x*P", gm.quant_tanh(np.array([16384, 16383, -16384, -16385, 1 << 30])), [1, 0, 0, -1, 127])
eq("rescale_params(1/128)", gm.rescale_params(1 / 128), (16384, 10))
eq("rescale_params(0.05)", gm.rescale_params(0.05), (26214, 8))
eq("quantize_multiplier(0.5)", gm.quantize_multiplier(0.5), (1 << 30, 0))
eq("quantize_multiplier(1.0)", gm.quantize_multiplier(1.0), (1 << 30, 1))
eq("quantize_multiplier(0.75)", gm.quantize_multiplier(0.75), (1610612736, 0))
eq("16-bit ROM words read as two's complement", gm.Lane(gm.INT8, [0xFFFF] + [0] * 31).rom[:2], [-1, 0])

zero = gm.Lane(gm.INT8, [0] * 32)  # P = 0 everywhere, so only the non-polynomial paths show
p = gm.Int8Params(mx=26214, shx=8, zin=0, mout=0, shout=0, zout=0)  # s_in = 0.05: x = 102.4 q in Q4.11
eq("tanh, zero table: x * 0 inside +/-3.125, saturated outside", zero.run(np.array([0, 62, 63, -62, -63]), 3, p), [0, 0, 127, 0, -128])
eq("sigmoid, zero table: P = 0, 1 - P = 1 inside +/-6.25, saturated outside", zero.run(np.array([124, 126, -124, -126]), 2, p), [-128, 127, 127, -128])
p2 = gm.Int8Params(mx=26214, shx=8, zin=0, mout=1 << 30, shout=-20, zout=3)  # output multiplier 2^-21
eq("SELU x >= 0: lambda x, requantized", zero.run(np.array([10, 0]), 1, p2), [11, 3])
eq("SELU -7 <= x < 0: x * P = 0 with a zero table", zero.run(np.array([-5]), 1, p2), [3])
p3 = gm.Int8Params(mx=26214, shx=7, zin=0, mout=1 << 30, shout=-20, zout=3)  # s_in = 0.1
eq("SELU x < -7: -lambda*alpha, requantized", zero.run(np.array([-71]), 1, p3), [-25])
eq("ReLU and linear pass through", [int(v) for v in zero.run(np.array([-5, 7]), 4, p)] + [int(v) for v in zero.run(np.array([-5, 7]), 5, p)], [-5, 7, -5, 7])
eq("every case's rescale fits", [int(0 <= gm.int8_params(c, 3).mx <= 32767) for a in ("tanh", "sigmoid", "selu") for c in gm.INT8_CASES[a]], [1] * 15)

print(f"check_gpnae_model_int8: {errs} errors")
print(f"RESULT: {'PASSED' if errs == 0 else 'FAILED'}")
sys.exit(1 if errs else 0)
```

Where the expected values come from: `(3 + 1) >> 1 = 2`, `(-3 + 1) >> 1 = -1`; `(8355585 + 2^30) >> 31 = 0`;
`6400^2 >> 11 = 20000`, `32767^2 >> 11 = 524255 -> 32767`; `(2048 + 4) >> 3 - 128 = 128 -> 127`,
`(-9 + 4) >> 3 - 128 = -129 -> -128`; `(-16385 + 2^14) >> 15 = -1`. tanh at s_in = 0.05:
`(62 * 26214 + 128) >> 8 = 6349` (inside), `(63 * 26214 + 128) >> 8 = 6451` (saturated). Sigmoid:
`(124 * 26214 + 128) >> 8 = 12697` (inside: P = 0 gives -128, 1 - P gives 127), `126` gives 12902 (saturated).
SELU at q = 10: x = `(262140 + 128) >> 8 = 1024`, v = `1024 * 17215 = 17628160`, `v * 2^-21 = 8.41 -> 8`, plus 3
is 11 in both rounding variants. SELU at q = -71, s_in = 0.1: x = `(-1861194 + 64) >> 7 = -14541 < -14336`, so
v = `(-3601 * 2048) << 3 = -58998784`, `v * 2^-21 = -28.13 -> -28` in both variants, plus 3 is -25.

`$J/cmds/int8_py.sh`:

```bash
#!/bin/bash
# Runs a Python script in GPNAE on a snapshot; args: script and its arguments; run from a snapshot root.
R=$(pwd)/testbenches/results/int8; mkdir -p $R
cd GPNAE && python3 "$@" 2>&1 | tee -a $R/py.log; exit ${PIPESTATUS[0]}
```

- [ ] **Step 2: Run it to see it fail**

Run: `$J/snap_launch_tree.sh i10_model 32 1 $J/cmds/int8_py.sh check_gpnae_model_int8.py`
Expected: `AttributeError: module 'gpnae_model' has no attribute 'rescale'`.

- [ ] **Step 3: Write the int8 lane model**

In `gpnae_model.py`, extend the docstring's first line with `; the int8 lane (gpnae_poly_int8) on AriL's ipu.py.`,
add `from collections import namedtuple` to the imports and `import ipu  # noqa: E402` after `import fpu`. Add this
method at the top of `class Lane`:

```python
    def __new__(cls, f, rom, *args, **kwargs):
        if cls is Lane and f is INT8:  # the int8 build's lane
            return super().__new__(LaneInt8)
        return super().__new__(cls)
```

Append to the end of the file:

```python
# int8 lane: the float lane's forms on integer units (fxMac Horner in Q4.11, integer products), then int8 quantization.


class Int8Fmt:
    """The int8 build as the lane sees it: 8-bit ports, Q4.11 inside."""
    name, w, frac, iw = "int8", 8, 11, 16


INT8 = Int8Fmt()
FORMATS = dict(fpu.FORMATS, int8=INT8)

Q = 11  # Q4.11
SETS_INT8 = {1: (0, 8), 2: (9, 6), 3: (16, 8)}  # (ROM base, degree) per control word; Task 10's fit sets it
T_SELU, T_SIG, T_TANH = -14336, 12800, 6400  # saturation in Q4.11: SELU x < -7, sigmoid |x| > 6.25, tanh |x| > 3.125
THRESH = {1: T_SELU, 2: T_SIG, 3: T_TANH}
SELU_SAT, ONE_Q11, LAMBDA_Q14 = -3601, 2048, 17215  # -lambda*alpha and 1.0 in Q4.11, lambda in Q1.14
LAMBDA_F, LA_F = 1.0507009873554805, 1.7580993408473766
Int8Params = namedtuple("Int8Params", "mx shx zin mout shout zout")
Case = namedtuple("Case", "s_in z_in s_out z_out gated")


def rescale(q, zin, mx, s):
    """(q - z_in) * mx rounded half up by 2^s, saturated to Q4.11 (gpnae_poly_int8 rescale())."""
    assert 0 <= mx <= 32767, "gp_mx_i must be below 2^15"
    q = np.asarray(q, dtype=np.int64)
    p = ipu.int_mul(q - zin, np.full_like(q, mx), w=16)
    r = p if s == 0 else (p + (1 << (s - 1))) >> s
    return np.clip(r, -32768, 32767)


def mac_operand(x, code):
    """The float lane's MAC operand in Q4.11: SELU x, sigmoid |x| (saturated), tanh x^2 on an fxMac with C = 0."""
    x = np.asarray(x, dtype=np.int64)
    if code == 1:
        return x
    if code == 2:
        return np.minimum(np.abs(x), 32767)
    return ipu.fx_mac(x, x, np.zeros_like(x), w=16, frac=Q)


def horner(t, rom, base, deg):
    """barrel_mac's int8 rounds: fxMac(A=operand, X=acc, C=coefficient), highest coefficient first, acc from 0."""
    t = np.asarray(t, dtype=np.int64)
    acc = np.zeros_like(t)
    for r in range(deg + 1):
        acc = ipu.fx_mac(t, acc, np.full_like(t, rom[base + deg - r]), w=16, frac=Q)
    return acc


def quant_sig(y):
    """sigmoid output from y in Q4.11: round(256 y) - 128 = ((y + 4) >> 3) - 128, clamped to int8."""
    return np.clip(((np.asarray(y, dtype=np.int64) + 4) >> 3) - 128, -128, 127)


def quant_tanh(p):
    """tanh output from the product x * P in units of 2^-22: round(128 y) = (p + 2^14) >> 15, clamped to int8."""
    return np.clip((np.asarray(p, dtype=np.int64) + (1 << 14)) >> 15, -128, 127)


def rescale_params(s_in):
    """(mx, shx) with mx / 2^shx = s_in * 2^11, the largest shx <= 31 that keeps mx <= 32767."""
    m = s_in * (1 << Q)
    for shx in range(31, -1, -1):
        mx = int(np.floor(m * 2.0 ** shx + 0.5))
        if mx <= 32767:
            return mx, shx
    raise ValueError(f"input scale {s_in} is too large for Q4.11")


def quantize_multiplier(real):
    """TFLite's QuantizeMultiplier: real = mult * 2^shift / 2^31, mult in [2^30, 2^31)."""
    if real == 0.0:
        return 0, 0
    q, shift = np.frexp(real)
    qf = int(np.floor(q * (1 << 31) + 0.5))  # TfLiteRound: half away from zero, q > 0
    if qf == 1 << 31:
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return qf, int(shift)


def calib_out(y_lo, y_hi):
    """TFLite-style int8 output quantization of [y_lo, y_hi], widened to hold 0."""
    lo, hi = min(y_lo, 0.0), max(y_hi, 0.0)
    s = (hi - lo) / 255.0
    return s, int(np.clip(np.floor(-128 - lo / s + 0.5), -128, 127))


def selu_case(s_in, z_in, gated=True):
    """A SELU case whose output scale is calibrated from the inputs' range, as post-training quantization would."""
    r = s_in * (np.array([-128.0, 127.0]) - z_in)
    y = np.where(r >= 0, LAMBDA_F * r, LA_F * np.expm1(r))
    s_out, z_out = calib_out(float(y[0]), float(y[1]))
    return Case(s_in, z_in, s_out, z_out, gated)


# Scales put the fitted range at 128, 32 and 8 int8 steps (SELU: 128, 64, 256, since its x must stay within +/-16).
# The SELU case at 7/32 reaches x = 27.8, past Q4.11; it is checked bit-exact only (not gated for accuracy).
INT8_CASES = {
    "tanh": [Case(3.125 / n, z, 1 / 128, 0, True) for n, z in ((128, 0), (32, 0), (8, 0), (32, -37), (64, 100))],
    "sigmoid": [Case(6.25 / n, z, 1 / 256, -128, True) for n, z in ((128, 0), (32, 0), (8, 0), (32, 25), (64, -100))],
    "selu": [selu_case(7 / 128, 0), selu_case(7 / 64, 0), selu_case(7 / 256, 0), selu_case(7 / 128, 120),
             selu_case(7 / 32, 0, gated=False)],
    "relu": [Case(0.05, z, 0.05, z, False) for z in (0, -20, 20, 100, -100)],
    "linear": [Case(0.05, z, 0.05, z, False) for z in (0, -20, 20, 100, -100)],
}


def int8_params(case, code):
    """The lane's per-layer inputs for a case: rescale always, SELU's output requantize for code 1."""
    mx, shx = rescale_params(case.s_in)
    if code == 1:
        mout, shout = quantize_multiplier(2.0 ** -25 / case.s_out)  # the SELU value is in units of 2^-25
        return Int8Params(mx, shx, case.z_in, mout, shout, case.z_out)
    return Int8Params(mx, shx, case.z_in, 0, 0, 0)


def exact_int8(q, code, case):
    """The exact function at the real input, quantized with round half up and clamped: what the lane approximates."""
    q = np.asarray(q, dtype=np.int64)
    r = case.s_in * (q.astype(float) - case.z_in)
    if code == 3:
        y, s, z = np.tanh(r), 1 / 128, 0
    elif code == 2:
        y, s, z = 1 / (1 + np.exp(-r)), 1 / 256, -128
    elif code == 1:
        y, s, z = np.where(r >= 0, LAMBDA_F * r, LA_F * np.expm1(r)), case.s_out, case.z_out
    else:
        return q.copy()
    return np.clip(np.floor(y / s + 0.5) + z, -128, 127).astype(np.int64)


class LaneInt8(Lane):
    """Bit-exact model of gpnae_poly_int8."""

    def __init__(self, f, rom, sets=None, thresh=None):
        self.f = f
        self.rom = [v - (1 << 16) if v >= (1 << 15) else v for v in rom]
        self.sets = {**SETS_INT8, **(sets or {})}
        self.thresh = {**THRESH, **(thresh or {})}

    def poly(self, x, code):
        """P at the float lane's MAC operand."""
        base, deg = self.sets[code]
        return horner(mac_operand(x, code), self.rom, base, deg)

    def value(self, x, code):
        """The unsaturated path as a real number before quantizing: x * P for SELU (x < 0) and tanh; P or 1 - P for sigmoid."""
        x = np.asarray(x, dtype=np.int64)
        p = self.poly(x, code)
        if code == 2:
            return np.where(x < 0, ONE_Q11 - p, p) / 2048.0
        return ipu.int_mul(x, p, w=16) * 2.0 ** -22

    def run(self, q, code, par):
        q = np.asarray(q, dtype=np.int64)
        code = code if code in (1, 2, 4, 5) else 3  # every other control word runs tanh, as in the float lane
        if code in (4, 5):
            return q.copy()  # ReLU and linear pass through: the requantize clamp applied them
        x = rescale(q, par.zin, par.mx, par.shx)
        neg = x < 0
        p = self.poly(x, code)
        if code == 2:
            sat = np.abs(x) > self.thresh[2]
            return np.where(sat, np.where(neg, -128, 127), quant_sig(np.where(neg, ONE_Q11 - p, p)))
        if code == 3:
            sat = np.abs(x) > self.thresh[3]
            return np.where(sat, np.where(neg, -128, 127), quant_tanh(ipu.int_mul(x, p, w=16)))
        sat = x < self.thresh[1]
        a = np.where(sat, SELU_SAT, x)
        b = np.where(neg, np.where(sat, ONE_Q11, p), LAMBDA_Q14)
        prod = ipu.int_mul(a, b, w=16)
        v = np.where(neg, prod << 3, prod)  # x * P and -lambda*alpha * 1.0 are in 2^-22, x * lambda in 2^-25
        full = lambda c: np.full_like(v, c)
        return ipu.requant(v, full(par.mout), full(par.shout), full(par.zout), full(-128), full(127), ipu.REQ_ROUNDING)

    def element(self, q, code, par) -> int:
        return int(self.run(np.array([q]), code, par)[0])
```

`coeff_file(INT8)` already gives `poly_coeffs_int8.mem`.

- [ ] **Step 4: Run it to see it pass**

Run: `$J/snap_launch_tree.sh i10_model 32 1 $J/cmds/int8_py.sh check_gpnae_model_int8.py`
Expected: `check_gpnae_model_int8: 0 errors`, `RESULT: PASSED`. A failure in the SELU rows with the others passing points
at `ipu.requant`'s argument order or at `REQ_ROUNDING`; the SELU vectors round the same in both variants.

- [ ] **Step 5: Write the failing fit check**

Run: `$J/snap_launch_tree.sh i10_guard 32 1 $J/cmds/int8_py.sh fit_poly_coeffs.py --format int8`
Expected now: `argument --format: invalid choice: 'int8'`.

- [ ] **Step 6: Add the int8 fit: the float forms, measured**

In `fit_poly_coeffs.py`: extend the docstring's last line to `Writes poly_coeffs_<fmt>.mem in the GPNAE root and in
src/TYTAN/Memory. Never writes the fp32 or bf16 files; int8 fits the same forms in Q4.11 and measures them first.`
Change the `--format` choices to `sorted(gpnae_model.FORMATS)`, and replace the fp32 guard in `main()` with:

```python
    if a.format in ("fp32", "bf16"):
        name = gpnae_model.coeff_file(gpnae_model.FORMATS[a.format])
        print(f"refusing to write {a.format} coefficients: {name} is published and fixed")
        sys.exit(1)
    if a.format == "int8":
        sys.exit(main_int8(a))
```

(fp32's message stays `refusing to write fp32 coefficients: poly_coeffs.mem is published and fixed`.) Add before
`main()`:

```python
NAME_INT8 = {1: "selu", 2: "sigmoid", 3: "tanh"}
RANGES_INT8 = {1: (4.0, 7.0), 2: (3.5, 6.25), 3: (3.125,)}  # the float lane's threshold where it differs, then the implemented range
DEGREES_INT8 = range(2, 13)
OPTIONS_INT8 = [
    "",
    "OPTIONS FOR SOHAM (none implemented; the lane keeps the float lanes' forms until Soham decides):",
    "A. Centred, scaled operands. Evaluate each function directly on t = (x - c) / h with |t| < 1: tanh P((|x| - 1.5625) / 2)",
    "   on [0, 3.125] with the sign restored; SELU P((x + 3.5) / 4) on [-7, 0]; sigmoid as tanh at x/2, exact for the int8",
    "   output since round(256 sig(x)) - 128 = round(128 tanh(x/2)). Why: fixed-point Horner multiplies each step's floor",
    "   error by the operand in every later step (bound: the sum of |t|^j ULP), and the float forms' operands reach 9.77",
    "   (tanh u), 6.25 (sigmoid) and 7 (SELU); with |t| < 1 the bound is at most degree + 1 ULP, and the power coefficients",
    "   stay within Q4.11 because each function's nearest singularity is farther from the interval's centre than its",
    "   half-length (estimate, not measured). RTL: a constant subtract and a shift on the MAC operand in place of the",
    "   squarer and abs; sigmoid adds 1 to the rescale shift; the rest of the lane is unchanged.",
    "B. Higher degree. The rows above cover degrees 2 to 12; where the error stops falling with degree, more terms do not",
    "   help at Q4.11. Degrees past the ROM's 32 entries need a 64-entry ROM, whose address width the lane now shares",
    "   with its FIFO depth (ADDR_LINES), so the two would have to be split.",
    "C. Q-format change. Keep the operand in Q4.11 and hold coefficients and accumulator in Q1.14 (range +/-2, enough",
    "   for P of all three forms: at most 1.76): fxMac's FRAC stays 11 (Q4.11 x Q1.14 >> 11 is Q1.14), so this is a",
    "   coefficient and post-stage scaling change only; it gains 3 bits everywhere but not the operand's amplification,",
    "   and needs the partial sums within +/-2 (the max |c| column shows the coefficients' size). Or a 32-bit",
    "   accumulator (fxMac W = 32), which changes Level 1's unit and DV.",
]


def fit_form(code, r, deg):
    """The float lane's fit (target(): (e^x - 1)/x form, sigmoid, tanh(sqrt u)/sqrt u) on the operand's range, rounded to Q4.11."""
    lo, hi = {1: (-r, 0.0), 2: (0.0, r), 3: (0.0, r * r)}[code]
    t = np.linspace(lo, hi, 4001)
    c = np.polynomial.chebyshev.Chebyshev.fit(t, target(code, t), deg).convert(kind=np.polynomial.Polynomial).coef
    ints = [int(np.floor(v * 2048 + 0.5)) for v in c]
    return ints if all(-32768 <= v <= 32767 for v in ints) else None


def lane_for(code, r, coeffs):
    """The bit-exact lane with one coefficient set at base 0 and saturation at the candidate range."""
    rom = [0] * 32
    rom[:len(coeffs)] = coeffs
    th = int(round(r * 2048))
    return gpnae_model.Lane(gpnae_model.INT8, [v & 0xFFFF for v in rom], sets={code: (0, len(coeffs) - 1)},
                            thresh={code: -th if code == 1 else th})


def lsb(code):
    """Output LSB per unit: tanh 128, sigmoid 256, SELU the tightest gated case's 1/s_out."""
    if code == 1:
        return 1.0 / min(c.s_out for c in gpnae_model.INT8_CASES["selu"] if c.gated)
    return 256.0 if code == 2 else 128.0


def measure_cont(code, r, coeffs):
    """Worst and mean error, in output LSB, of the unsaturated path over every Q4.11 input of the fitted range."""
    th = int(round(r * 2048))
    x = np.arange(-th, 1 if code == 1 else th + 1, dtype=np.int64)
    e = np.abs(lane_for(code, r, coeffs).value(x, code) - exact_act(code, x / 2048.0)) * lsb(code)
    return float(e.max()), float(e.mean())


def measure_cases(lane, code):
    """Every int8 input of every gated case: max |lane - exact| in int8 LSB, and how many inputs differ."""
    q = np.arange(-128, 128, dtype=np.int64)
    dmax, ndiff = 0, 0
    for case in gpnae_model.INT8_CASES[NAME_INT8[code]]:
        if case.gated:
            d = np.abs(lane.run(q, code, gpnae_model.int8_params(case, code)) - gpnae_model.exact_int8(q, code, case))
            dmax, ndiff = max(dmax, int(d.max())), ndiff + int((d > 0).sum())
    return dmax, ndiff


def refine_int8(code, r, c, passes=10):
    """Coordinate descent on the integer coefficients against the bit-exact model: absorbs rounding and floor bias."""
    best = measure_cont(code, r, c)[0]
    for _ in range(passes):
        improved = False
        for k in range(len(c)):
            for d in (-2, -1, 1, 2):
                trial = list(c)
                trial[k] += d
                if not -32768 <= trial[k] <= 32767:
                    continue
                w = measure_cont(code, r, trial)[0]
                if w < best:
                    best, c, improved = w, trial, True
        if not improved:
            break
    return c


def layout_int8(deg):
    """The fp32 table's layout when the degrees fit it, else the three sets packed in order; None if they exceed 32 entries."""
    d1, d2, d3 = deg[1], deg[2], deg[3]
    if d1 <= 8 and d2 <= 6 and d3 <= 15:
        return {1: (0, d1), 2: (9, d2), 3: (16, d3)}
    if d1 + d2 + d3 + 3 <= 32:
        return {1: (0, d1), 2: (d1 + 1, d2), 3: (d1 + d2 + 2, d3)}
    return None


def main_int8(a):
    out = gpnae_model.coeff_file(gpnae_model.INT8)
    assert out not in ("poly_coeffs.mem", "poly_coeffs_bf16.mem", "taylor_coeffs.mem")
    L, chosen = [], {}
    L.append("gpnae_poly int8: the float lanes' forms in Q4.11 (fxMac Horner, floor; integer post products), bit-exact model.")
    L.append("cases: max |lane - exact| in int8 LSB over every int8 input of every gated case (target 1); inputs differing.")
    L.append("range: max and mean error of the unsaturated path over every Q4.11 input of the fitted range, in output LSB")
    L.append("(SELU in LSB of the tightest gated case). Candidate ranges: the float lane's threshold, then the implemented one.")
    L.append(f"{'activation':<10}{'range':>7}{'degree':>7}{'cases max':>10}{'differ':>8}{'range max':>10}{'mean':>7}{'max |c|':>9}")
    for code in (1, 2, 3):
        impl, best = RANGES_INT8[code][-1], None
        for r in RANGES_INT8[code]:
            for d in DEGREES_INT8:
                c = fit_form(code, r, d)
                if c is None:
                    L.append(f"{NAME_INT8[code]:<10}{r:>7}{d:>7}  coefficients beyond Q4.11")
                    continue
                if r == impl:
                    c = refine_int8(code, r, c)
                cw, cm = measure_cont(code, r, c)
                dmax, ndiff = measure_cases(lane_for(code, r, c), code)
                L.append(f"{NAME_INT8[code]:<10}{r:>7}{d:>7}{dmax:>10}{ndiff:>8}{cw:>10.2f}{cm:>7.2f}"
                         f"{max(abs(v) for v in c) / 2048:>9.3f}")
                if r == impl:
                    if best is None or (dmax, cw) < (best[1], best[2]):
                        best = (d, dmax, cw)
                    if dmax <= 1 and code not in chosen:
                        chosen[code] = (d, c)
        if code in chosen:
            L.append(f"{NAME_INT8[code]:<10} chosen: range {impl}, degree {chosen[code][0]} (the lowest within 1 LSB): TARGET MET")
        elif best is None:
            L.append(f"{NAME_INT8[code]:<10} no degree fits Q4.11: TARGET MISSED")
        else:
            L.append(f"{NAME_INT8[code]:<10} TARGET MISSED: best degree {best[0]}, {best[1]} LSB on the cases, "
                     f"{best[2]:.2f} LSB over the range")
    lay = layout_int8({k: v[0] for k, v in chosen.items()}) if len(chosen) == 3 else None
    if lay is None:
        if len(chosen) == 3:
            L.append("the chosen degrees do not fit the 32-entry ROM: TARGET MISSED")
        L.append("VERDICT: STOP (no table written)")
        stop = os.path.join(os.path.dirname(os.path.abspath(a.report)), "gpnae_int8_stop.log")
        os.makedirs(os.path.dirname(stop), exist_ok=True)
        open(a.report, "w").write("\n".join(L) + "\n")
        open(stop, "w").write("\n".join(L + OPTIONS_INT8) + "\n")
        print("\n".join(L + OPTIONS_INT8))
        print(f"STOP: report for Soham in {stop}")
        return 2
    table = [0] * 32
    for code, (base, d) in lay.items():
        table[base:base + d + 1] = chosen[code][1]
    lane = gpnae_model.Lane(gpnae_model.INT8, [v & 0xFFFF for v in table], sets=lay)
    L.append("")
    L.append(f"whole table, SETS_INT8 = {lay}: every int8 input of every case (int8 LSB)")
    L.append(f"{'activation':<10}{'s_in':>11}{'z_in':>6}{'s_out':>11}{'z_out':>6}{'max':>5}{'differ':>8}{'gated':>7}")
    q = np.arange(-128, 128, dtype=np.int64)
    ok = True
    for code in (1, 2, 3):
        for case in gpnae_model.INT8_CASES[NAME_INT8[code]]:
            d = np.abs(lane.run(q, code, gpnae_model.int8_params(case, code)) - gpnae_model.exact_int8(q, code, case))
            L.append(f"{NAME_INT8[code]:<10}{case.s_in:>11.6f}{case.z_in:>6}{case.s_out:>11.6f}{case.z_out:>6}"
                     f"{int(d.max()):>5}{int((d > 0).sum()):>8}{'yes' if case.gated else 'no':>7}")
            ok = ok and (int(d.max()) <= 1 or not case.gated)
    assert ok, "a per-activation choice within 1 LSB must stay within 1 LSB in the packed table"
    L.append("VERDICT: PASS")
    for path in (os.path.join(ROOT, out), os.path.join(ROOT, "src", "TYTAN", "Memory", out)):
        with open(path, "w") as fh:
            fh.write("".join(format(v & 0xFFFF, "016b") + "\n" for v in table))
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if lay != gpnae_model.SETS_INT8:
        print(f"UPDATE gpnae_model.SETS_INT8 = {lay} and gpnae_poly_int8's BASE_*/DEG_*, then rerun")
        return 3
    return 0
```

`target` and `exact_act` are the script's existing functions (the float lane's fitted functions and the exact
activations). The float-lane-range rows are fitted without refinement and are there to show what the float lane's
thresholds would cost without a tail; only the implemented range is chosen from.

`$J/cmds/int8_fit.sh`:

```bash
#!/bin/bash
# Fits and measures poly_coeffs_int8.mem, keeps the table, the report and any stop report; run from a snapshot root.
R=$(pwd)/testbenches/results/int8; mkdir -p $R
cd GPNAE && python3 fit_poly_coeffs.py --format int8 --report $R/poly_coeffs_int8_fit.log; rc=$?
cp poly_coeffs_int8.mem $R/ 2>/dev/null
for f in fp32 bf16; do python3 fit_poly_coeffs.py --format $f; echo "guard $f exit=$?"; done
echo "fit exit=$rc"; exit $rc
```

- [ ] **Step 7: Measure on the farm, then act on the verdict**

Run: `$J/snap_launch_tree.sh i10_fit 32 4 $J/cmds/int8_fit.sh`

Expected in `runs/i10_fit/stdout.log`: a row per activation, range and degree; one `chosen:` or `TARGET MISSED` line per
activation; a `VERDICT:` line; `guard fp32 exit=1`, `guard bf16 exit=1`; `fit exit=` 0, 2 or 3. Then:

- **`fit exit=2` (VERDICT: STOP).** This task stops here and Level 2 does not go on. Copy
  `runs/i10_fit/results/int8/gpnae_int8_stop.log` to `testbenches/results/int8/gpnae_int8_stop.log` (SIENNA root) and give
  it to Soham: the measured errors per activation, range and degree, and options A (centred, scaled operands),
  B (higher degree) and C (a Q-format change). Implement none of them. Commit only the model, its check and the fit
  script (Step 8, first three commits), and wait for Soham's decision.
- **`fit exit=3`.** Set `SETS_INT8` in `gpnae_model.py` to the printed layout, rerun `i10_check` below, then rerun
  `i10_fit`; it must now end with `fit exit=0` and the same table. Task 11 uses the same layout.
- **`fit exit=0`.** Bring the table back and check nothing published moved:

```bash
cd /proj/work/spramanik/SIENNA_int8/GPNAE
ls poly_coeffs_int8.mem src/TYTAN/Memory/poly_coeffs_int8.mem   # must not exist yet
cp $J/runs/i10_fit/results/int8/poly_coeffs_int8.mem poly_coeffs_int8.mem
cp $J/runs/i10_fit/results/int8/poly_coeffs_int8.mem src/TYTAN/Memory/poly_coeffs_int8.mem
wc -l poly_coeffs_int8.mem && awk '{ print length($0) }' poly_coeffs_int8.mem | sort -u
git status --short
cmp poly_coeffs_bf16.mem /proj/work/spramanik/SIENNA/GPNAE/poly_coeffs_bf16.mem && cmp poly_coeffs.mem /proj/work/spramanik/SIENNA/GPNAE/poly_coeffs.mem && echo PUBLISHED-UNCHANGED
```

Expected: `32` lines, all of length `16`; `git status` shows the two new table files as untracked and the model and fit
script as modified; `PUBLISHED-UNCHANGED`. After a `SETS_INT8` update, rerun the model check:
`$J/snap_launch_tree.sh i10_check 32 1 $J/cmds/int8_py.sh check_gpnae_model_int8.py` (expected `RESULT: PASSED`; the
checks use a zero table and do not depend on the layout).

- [ ] **Step 8: Commit (GPNAE)**

```bash
git add gpnae_model.py && git commit -m "gpnae_model: bit-exact int8 lane, the float lanes' forms on fx_mac and integer products, int8 quantize"
git add check_gpnae_model_int8.py && git commit -m "check_gpnae_model_int8: known values for the int8 lane model"
git add fit_poly_coeffs.py && git commit -m "fit_poly_coeffs: int8 fit and measurement of the float forms in Q4.11; bf16 table refused like fp32"
git add poly_coeffs_int8.mem src/TYTAN/Memory/poly_coeffs_int8.mem && git commit -m "poly_coeffs_int8.mem: gpnae_poly's Q4.11 table for int8"
git push origin int8
```

The fourth commit only when the verdict is PASS.

### Task 11: `gpnae_poly` int8

Starts only after Task 10's `VERDICT: PASS` (`fit exit=0`).

**Files:**
- Create: `src/gpnae_poly_int8.sv`
- Modify: `src/gpnae_poly.sv` (ports, `G_INT8` / `G_FLOAT`), `src/gpnae_tail.sv` (int8 rejected), `Makefile`,
  `testbenches/TB_gpnae_poly.sv` (the float lane's new hierarchy, new ports tied off)

**Interfaces:**
- Consumes: `barrel_mac` int8 (Task 9); Task 10's constants, layout (`SETS_INT8`) and operation order;
  `intMultiplier #(.W(16))` (Task 4, latency `mul_lat(0, 7)` = 1); `fxMac` (latency `fx_lat()`);
  `tfliteRequant #(.ROUNDING)` (Task 7, latency `req_lat()`), `acc_i, mult_i, shift_i, zp_i, act_min_i, act_max_i,
  result_o`; `sienna_fmt_pkg::REQ_ROUNDING` (Task 3).
- Produces:
  - `gpnae_poly #(EXP_W, MAN_W, DATA_WIDTH, ADDR_LINES, CONTROL_WIDTH, K, TAIL_CONTEXTS)`, existing ports plus
    `gp_mx_i [15:0]` (below 2^15), `gp_shx_i [4:0]`, `gp_zin_i [7:0]`, `gp_mout_i [31:0]`, `gp_shout_i [7:0]`
    (signed), `gp_zout_i [7:0]`; float builds ignore them. Like `control_word_i`, they must be stable from a group's
    first `wr_en_i` to its last `done_o`. In int8, `DATA_WIDTH` is 8 and the lane reads `poly_coeffs_int8.mem` from
    the simulation directory. Hierarchy: `G_INT8.lane_inst` (int8), `G_FLOAT.G_MAC.barrel_mac_inst` (floats).
  - `gpnae_poly_int8 #(DATA_WIDTH = 8, ADDR_LINES, CONTROL_WIDTH, K)`, the same ports.
  - `gpnae_tail` rejects int8 (`gpnae_tail: unsupported format EXP_W=0 MAN_W=7`).

- [ ] **Step 1: Write the failing elaboration checks**

```bash
$J/snap_launch_tree.sh i11_l8 32 1 $J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=0 -GMAN_W=7"
$J/snap_launch_tree.sh i11_tail 32 1 $J/cmds/int8_lint.sh gpnae_tail "-GEXP_W=0 -GMAN_W=7" "gpnae_tail: unsupported format"
```

Expected now: `i11_l8` ends with `errors=` above 0 (the float body elaborates at `EXP_W = 0`: gpnae_tail's field
selects and `fpMultiplier #(.EXP_W(0))`), and `i11_tail` prints `NOT-REJECTED`. Task 12's exhaustive TB is the
functional test for this task (the plan's order); a mismatch there is fixed in this task's files.

- [ ] **Step 2: Write the int8 lane**

`src/gpnae_poly_int8.sv` (set the six `BASE_*`/`DEG_*` values to Task 10's `SETS_INT8`; the values below are the fp32
layout `SETS_INT8` starts from):

```systemverilog
`default_nettype wire
`timescale 1ns / 100ps

// int8 GPNAE lane (gpnae_poly's G_INT8 branch): the float lane's forms on fxMac (Q4.11) and integer products, int8 out; no gpnae_tail.
module gpnae_poly_int8 #(
    parameter int DATA_WIDTH    = 8,
    parameter int ADDR_LINES    = 5,
    parameter int CONTROL_WIDTH = 3,  // 001 SELU, 010 sigmoid, 011 tanh, 100 ReLU, 101 linear
    parameter int K             = 16
) (
    input logic clk_i,
    input logic rstn_i,

    input logic [DATA_WIDTH-1:0] signal_i,
    input logic                  wr_en_i,
    input logic                  last_i,

    input logic [   ADDR_LINES-1:0] terms_i,         // unused: degree comes from the table
    input logic [CONTROL_WIDTH-1:0] control_word_i,

    input logic [15:0] gp_mx_i,     // input rescale: x = round((q - z_in) * mx / 2^shx) in Q4.11, mx below 2^15
    input logic [ 4:0] gp_shx_i,
    input logic [ 7:0] gp_zin_i,    // input zero point
    input logic [31:0] gp_mout_i,   // SELU output: TFLite multiplier and shift of 2^-25 / s_out
    input logic [ 7:0] gp_shout_i,
    input logic [ 7:0] gp_zout_i,   // SELU output zero point

    output logic full_o,
    output logic empty_o,
    output logic idle_o,

    output logic [DATA_WIDTH-1:0] final_result_o,
    output logic                  done_o
);

  localparam int SW = $clog2(K);
  localparam int CAP_LAG = 3;  // as gpnae_poly: a pop's word appears three cycles later
  localparam int W = 16;  // Q4.11
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(0, 7);  // intMultiplier
  localparam int FX_LAT = sienna_fmt_pkg::fx_lat();  // fxMac: the tanh squarer
  localparam int RQ_LAT = sienna_fmt_pkg::req_lat();  // tfliteRequant
  localparam int PS_LAT = MUL_LAT + RQ_LAT;  // SELU post stage: multiply, then requantize

  localparam logic signed [W-1:0] T_SELU = -16'sd14336;  // saturation in Q4.11: SELU x < -7
  localparam logic signed [W-1:0] T_SIG = 16'sd12800;  // sigmoid |x| > 6.25
  localparam logic signed [W-1:0] T_TANH = 16'sd6400;  // tanh |x| > 3.125
  localparam logic signed [W-1:0] SELU_SAT = -16'sd3601;  // -lambda*alpha in Q4.11
  localparam logic signed [W-1:0] ONE_Q11 = 16'sd2048;  // 1.0 in Q4.11
  localparam logic signed [W-1:0] LAMBDA_Q14 = 16'sd17215;  // lambda in Q1.14

  // Coefficient sets: gpnae_model.SETS_INT8 from Task 10's fit.
  localparam logic [ADDR_LINES-1:0] BASE_SELU = 5'd0, DEG_SELU = 5'd8;
  localparam logic [ADDR_LINES-1:0] BASE_SIG = 5'd9, DEG_SIG = 5'd6;
  localparam logic [ADDR_LINES-1:0] BASE_TANH = 5'd16, DEG_TANH = 5'd8;

  if (DATA_WIDTH != 8) begin : G_BAD_WIDTH
    $fatal(1, "gpnae_poly_int8: DATA_WIDTH must be 8, not %0d", DATA_WIDTH);
  end

  // Input rescale: (q - z_in) * mx rounded half up by 2^s, saturated to Q4.11.
  function automatic logic [W-1:0] rescale(input logic [31:0] p, input logic [4:0] s);
    logic signed [39:0] pw, r;
    pw = {{8{p[31]}}, p};
    r  = (s == 5'd0) ? pw : ((pw + (40'sd1 <<< (s - 5'd1))) >>> s);
    if (r > 40'sd32767) return 16'h7FFF;
    if (r < -40'sd32768) return 16'h8000;
    return r[W-1:0];
  endfunction

  // |x|, saturated: sigmoid's MAC operand.
  function automatic logic [W-1:0] abs_sat(input logic [W-1:0] x);
    if (x == 16'h8000) return 16'h7FFF;
    return x[W-1] ? -x : x;
  endfunction

  // sigmoid output: round(256 y) - 128 = ((y + 4) >> 3) - 128 for y in Q4.11, clamped to int8.
  function automatic logic [7:0] quant_sig(input logic signed [W:0] y);
    logic signed [W:0] r;
    r = ((y + 17'sd4) >>> 3) - 17'sd128;
    if (r > 17'sd127) return 8'h7F;
    if (r < -17'sd128) return 8'h80;
    return r[7:0];
  endfunction

  // tanh output from x * P in units of 2^-22: round(128 y) = (p + 2^14) >> 15, clamped to int8.
  function automatic logic [7:0] quant_tanh(input logic [31:0] p);
    logic signed [32:0] r;
    r = ($signed({p[31], p}) + 33'sd16384) >>> 15;
    if (r > 33'sd127) return 8'h7F;
    if (r < -33'sd128) return 8'h80;
    return r[7:0];
  endfunction

  logic [DATA_WIDTH-1:0] fifo_data_o;
  logic                  fifo_rd_en;
  logic [ADDR_LINES:0]   fifo_count;

  InputFIFO #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_LINES(ADDR_LINES)
  ) input_fifo_inst (
      .clk_i  (clk_i),
      .rstn_i (rstn_i),
      .full_o (full_o),
      .empty_o(empty_o),
      .idle_o (idle_o),
      .wr_en_i(wr_en_i),
      .rd_en_i(fifo_rd_en),
      .count_o(fifo_count),
      .data_i (signal_i),
      .data_o (fifo_data_o)
  );

  logic is_selu, is_sig, is_byp, is_tanh;
  assign is_selu = (control_word_i == 3'b001);
  assign is_sig  = (control_word_i == 3'b010);
  assign is_byp  = (control_word_i == 3'b100) || (control_word_i == 3'b101);  // ReLU and linear pass through
  assign is_tanh = !is_selu && !is_sig && !is_byp;  // every other control word runs tanh, as in the float lane

  logic [ADDR_LINES-1:0] poly_base, poly_deg;
  assign poly_base = is_selu ? BASE_SELU : (is_sig ? BASE_SIG : BASE_TANH);
  assign poly_deg  = is_selu ? DEG_SELU : (is_sig ? DEG_SIG : DEG_TANH);

  logic         ld_valid, mac_start;
  logic [W-1:0] mac_in;
  logic         mac_res_valid, mac_busy, mac_done;
  logic [W-1:0] mac_res;

  // The int8 table as a literal: a string parameter passed down to the ROM's $readmemb is not found.
  barrel_mac #(
      .EXP_W     (0),
      .MAN_W     (7),
      .DATA_WIDTH(W),
      .ADDR_LINES(ADDR_LINES),
      .K         (K),
      .INIT_FILE ("poly_coeffs_int8.mem")
  ) barrel_mac_inst (
      .clk_i       (clk_i),
      .rstn_i      (rstn_i),
      .ld_valid_i  (ld_valid),
      .ld_data_i   (mac_in),
      .start_i     (mac_start),
      .terms_i     (poly_deg),
      .coeff_base_i(poly_base),
      .res_valid_o (mac_res_valid),
      .res_data_o  (mac_res),
      .busy_o      (mac_busy),
      .done_o      (mac_done)
  );

  // Rescale multiplier: (q - z_in) * mx.
  logic           rm_valid, rm_done;
  logic [W-1:0]   rm_a, rm_b;
  logic [2*W-1:0] rm_res;
  intMultiplier #(.W(W)) RESC (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(rm_valid), .A(rm_a), .B(rm_b), .result_o(rm_res),
                               .done_o(rm_done));

  // tanh's squaring unit, as the float lane's SQ: u = sat((x * x) >>> 11).
  logic         sq_valid, sq_done;
  logic [W-1:0] sq_a, sq_res;
  fxMac #(.W(W), .FRAC(11)) SQ (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(sq_valid), .A(sq_a), .X(sq_a), .C('0),
                                .result_o(sq_res), .done_o(sq_done));

  // Post multiply, as the float lane's POST: tanh x * P; SELU x * P, x * lambda, or -lambda*alpha * 1.0.
  logic           pm_valid, pm_done, rq_done;
  logic [W-1:0]   pm_a, pm_b;
  logic [2*W-1:0] pm_res, rq_acc;
  logic [7:0]     rq_res;
  intMultiplier #(.W(W)) POSTM (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(pm_valid), .A(pm_a), .B(pm_b), .result_o(pm_res),
                                .done_o(pm_done));

  logic [DATA_WIDTH-1:0] q_buf[K];    // captured int8 inputs
  logic [W-1:0]          x_buf[K];    // rescaled inputs, Q4.11
  logic [K-1:0]          neg_buf, sat_buf;
  logic [W-1:0]          pol_buf[K];
  logic [DATA_WIDTH-1:0] res_buf[K];
  logic [K-1:0]          res_rdy;

  logic [SW:0] n_elems, ld_idx, rx_idx, iss_idx, emit_idx;
  logic [SW:0] pop_idx, grp_n;
  logic [CAP_LAG-1:0] cap_v;
  logic [3:0]  drain_cnt;

  // Result pipelines; bit n lines up with a unit's done_o n cycles after valid_i, and each carries its element index.
  logic [MUL_LAT:0] rs_v;  // rescale
  logic [SW-1:0]    rs_p[MUL_LAT+1];
  logic [FX_LAT:0]  sq_v;  // tanh square, in load order
  logic [MUL_LAT:0] pt_v;  // tanh post multiply
  logic [SW-1:0]    pt_p[MUL_LAT+1];
  logic [PS_LAT:0]  ps_v;  // SELU post multiply and requantize
  logic [SW-1:0]    ps_p[PS_LAT+1];

  // SELU requantize: x * P and -lambda*alpha * 1.0 are in 2^-22, x * lambda in 2^-25.
  assign rq_acc = neg_buf[ps_p[MUL_LAT]] ? {pm_res[2*W-4:0], 3'b000} : pm_res;
  tfliteRequant #(.ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) POSTQ (
      .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(ps_v[MUL_LAT]), .acc_i(rq_acc), .mult_i(gp_mout_i), .shift_i(gp_shout_i),
      .zp_i(gp_zout_i), .act_min_i(8'h80), .act_max_i(8'h7F), .result_o(rq_res), .done_o(rq_done)
  );

  logic [DATA_WIDTH-1:0] q_cur;
  logic [W-1:0]          d_cur, x_new, p_cur, x_cur;
  logic                  x_sat;
  logic signed [W:0]     p_ext, y_s;
  assign q_cur = q_buf[iss_idx[SW-1:0]];
  assign d_cur = {{(W - 8) {q_cur[7]}}, q_cur} - {{(W - 8) {gp_zin_i[7]}}, gp_zin_i};  // q - z_in
  assign x_new = rescale(rm_res, gp_shx_i);
  assign x_sat = is_selu ? ($signed(x_new) < T_SELU)
               : is_sig  ? (($signed(x_new) > T_SIG) || ($signed(x_new) < -T_SIG))
                         : (($signed(x_new) > T_TANH) || ($signed(x_new) < -T_TANH));
  assign p_cur = pol_buf[iss_idx[SW-1:0]];
  assign x_cur = x_buf[iss_idx[SW-1:0]];
  assign p_ext = {p_cur[W-1], p_cur};
  assign y_s   = neg_buf[iss_idx[SW-1:0]] ? (17'sd2048 - p_ext) : p_ext;  // sigmoid: P, or 1 - P for x < 0

  typedef enum logic [3:0] {
    G_IDLE,
    G_CAP,
    G_LOAD,
    G_LDRAIN,
    G_RUN,
    G_RECV,
    G_POST,
    G_EMIT,
    G_NEXT
  } gstate_t;
  gstate_t gstate;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      gstate         <= G_IDLE;
      n_elems        <= '0;
      ld_idx         <= '0;
      rx_idx         <= '0;
      iss_idx        <= '0;
      emit_idx       <= '0;
      pop_idx        <= '0;
      grp_n          <= '0;
      cap_v          <= '0;
      drain_cnt      <= '0;
      ld_valid       <= 1'b0;
      mac_start      <= 1'b0;
      fifo_rd_en     <= 1'b0;
      rm_valid       <= 1'b0;
      sq_valid       <= 1'b0;
      pm_valid       <= 1'b0;
      done_o         <= 1'b0;
      final_result_o <= '0;
      rs_v           <= '0;
      sq_v           <= '0;
      pt_v           <= '0;
      ps_v           <= '0;
      res_rdy        <= '0;
      neg_buf        <= '0;
      sat_buf        <= '0;
      for (int i = 0; i <= MUL_LAT; i++) begin
        rs_p[i] <= '0;
        pt_p[i] <= '0;
      end
      for (int i = 0; i <= PS_LAT; i++) ps_p[i] <= '0;
    end else begin
      ld_valid   <= 1'b0;
      mac_start  <= 1'b0;
      fifo_rd_en <= 1'b0;
      rm_valid   <= 1'b0;
      sq_valid   <= 1'b0;
      pm_valid   <= 1'b0;
      done_o     <= 1'b0;

      rs_v <= {rs_v[MUL_LAT-1:0], 1'b0};
      sq_v <= {sq_v[FX_LAT-1:0], 1'b0};
      pt_v <= {pt_v[MUL_LAT-1:0], 1'b0};
      ps_v <= {ps_v[PS_LAT-1:0], 1'b0};
      for (int i = 1; i <= MUL_LAT; i++) begin
        rs_p[i] <= rs_p[i-1];
        pt_p[i] <= pt_p[i-1];
      end
      for (int i = 1; i <= PS_LAT; i++) ps_p[i] <= ps_p[i-1];

      // A rescaled input: kept for the post stage; its MAC operand is x (SELU), |x| (sigmoid) or x^2 (tanh, via SQ).
      if (rs_v[MUL_LAT]) begin
        x_buf[rs_p[MUL_LAT]]   <= x_new;
        neg_buf[rs_p[MUL_LAT]] <= x_new[W-1];
        sat_buf[rs_p[MUL_LAT]] <= x_sat;
        if (is_tanh) begin
          sq_a     <= x_new;
          sq_valid <= 1'b1;
          sq_v[0]  <= 1'b1;
        end else begin
          mac_in   <= is_sig ? abs_sat(x_new) : x_new;
          ld_valid <= 1'b1;
        end
      end
      if (sq_v[FX_LAT]) begin
        mac_in   <= sq_res;
        ld_valid <= 1'b1;
      end

      // Post results come back tagged with their element: tanh after the multiply, SELU after the requantizer.
      if (pt_v[MUL_LAT]) begin
        res_buf[pt_p[MUL_LAT]] <= quant_tanh(pm_res);
        res_rdy[pt_p[MUL_LAT]] <= 1'b1;
      end
      if (ps_v[PS_LAT]) begin
        res_buf[ps_p[PS_LAT]] <= rq_res;
        res_rdy[ps_p[PS_LAT]] <= 1'b1;
      end

      case (gstate)
        G_IDLE: begin
          ld_idx  <= '0;
          pop_idx <= '0;
          if (last_i) begin
            grp_n  <= (fifo_count > K[ADDR_LINES:0]) ? K[SW:0] : fifo_count[SW:0];
            gstate <= (fifo_count == '0) ? G_IDLE : G_CAP;
          end
        end

        // One pop per cycle; captures trail by CAP_LAG, as in gpnae_poly.
        G_CAP: begin
          if (pop_idx < grp_n) begin
            fifo_rd_en <= 1'b1;
            pop_idx    <= pop_idx + 1;
          end
          cap_v <= {cap_v[CAP_LAG-2:0], (pop_idx < grp_n)};
          if (cap_v[CAP_LAG-1]) begin
            q_buf[ld_idx[SW-1:0]] <= fifo_data_o;
            ld_idx                <= ld_idx + 1;
            if (is_byp) begin
              res_buf[ld_idx[SW-1:0]] <= fifo_data_o;  // the requantize clamp already applied ReLU
              res_rdy[ld_idx[SW-1:0]] <= 1'b1;
            end
            if (ld_idx + 1 == grp_n) begin
              n_elems   <= grp_n;
              iss_idx   <= '0;
              drain_cnt <= '0;
              emit_idx  <= '0;
              gstate    <= is_byp ? G_EMIT : G_LOAD;
            end
          end
        end

        // Rescale one element per cycle.
        G_LOAD: begin
          rm_a     <= d_cur;
          rm_b     <= gp_mx_i;
          rm_valid <= 1'b1;
          rs_v[0]  <= 1'b1;
          rs_p[0]  <= iss_idx[SW-1:0];
          if (iss_idx + 1 == n_elems) begin
            drain_cnt <= '0;
            gstate    <= G_LDRAIN;
          end else begin
            iss_idx <= iss_idx + 1;
          end
        end

        G_LDRAIN: begin
          // Start two cycles after the last operand was loaded; tanh's operands pass the squarer first.
          if (drain_cnt == (is_tanh ? 4'(MUL_LAT + FX_LAT + 3) : 4'(MUL_LAT + 2))) begin
            mac_start <= 1'b1;
            rx_idx    <= '0;
            gstate    <= G_RUN;
          end else begin
            drain_cnt <= drain_cnt + 1;
          end
        end

        G_RUN: begin
          if (mac_res_valid) begin
            pol_buf[rx_idx[SW-1:0]] <= mac_res;
            rx_idx                  <= rx_idx + 1;
            gstate                  <= G_RECV;
          end
        end

        G_RECV: begin
          if (mac_res_valid) begin
            pol_buf[rx_idx[SW-1:0]] <= mac_res;
            rx_idx                  <= rx_idx + 1;
          end
          if (mac_done) begin
            iss_idx  <= '0;
            emit_idx <= '0;
            res_rdy  <= '0;
            gstate   <= G_POST;
          end
        end

        // sigmoid and saturated tanh quantize in place; tanh multiplies; SELU multiplies and requantizes.
        G_POST: begin
          if (is_sig || (is_tanh && sat_buf[iss_idx[SW-1:0]])) begin
            res_buf[iss_idx[SW-1:0]] <= sat_buf[iss_idx[SW-1:0]] ? (neg_buf[iss_idx[SW-1:0]] ? 8'h80 : 8'h7F)
                                                                 : quant_sig(y_s);
            res_rdy[iss_idx[SW-1:0]] <= 1'b1;
          end else begin
            pm_a     <= (is_selu && sat_buf[iss_idx[SW-1:0]]) ? SELU_SAT : x_cur;
            pm_b     <= (is_selu && !neg_buf[iss_idx[SW-1:0]]) ? LAMBDA_Q14
                      : (is_selu && sat_buf[iss_idx[SW-1:0]]) ? ONE_Q11 : p_cur;
            pm_valid <= 1'b1;
            if (is_selu) begin
              ps_v[0] <= 1'b1;
              ps_p[0] <= iss_idx[SW-1:0];
            end else begin
              pt_v[0] <= 1'b1;
              pt_p[0] <= iss_idx[SW-1:0];
            end
          end
          if (iss_idx + 1 == n_elems) gstate <= G_EMIT;
          else iss_idx <= iss_idx + 1;
        end

        // Retire in index order, one done_o per element.
        G_EMIT: begin
          if (res_rdy[emit_idx[SW-1:0]]) begin
            final_result_o <= res_buf[emit_idx[SW-1:0]];
            done_o         <= 1'b1;
            if (emit_idx + 1 == n_elems) gstate <= G_NEXT;
            else emit_idx <= emit_idx + 1;
          end
        end

        G_NEXT: begin
          ld_idx  <= '0;
          pop_idx <= '0;
          cap_v   <= '0;
          grp_n   <= (fifo_count > K[ADDR_LINES:0]) ? K[SW:0] : fifo_count[SW:0];
          gstate  <= (fifo_count == '0) ? G_IDLE : G_CAP;
        end

        default: gstate <= G_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  // The pipelines assume the package's unit latencies; a unit that differs would corrupt results silently.
  always @(posedge clk_i)
    if (rstn_i) begin
      if (rs_v[MUL_LAT] != rm_done) $fatal(1, "gpnae_poly_int8: rescale intMultiplier latency is not %0d", MUL_LAT);
      if (sq_v[FX_LAT] != sq_done) $fatal(1, "gpnae_poly_int8: squaring fxMac latency is not %0d", FX_LAT);
      if ((pt_v[MUL_LAT] | ps_v[MUL_LAT]) != pm_done) $fatal(1, "gpnae_poly_int8: post intMultiplier latency is not %0d", MUL_LAT);
      if (ps_v[PS_LAT] != rq_done) $fatal(1, "gpnae_poly_int8: tfliteRequant latency is not %0d", RQ_LAT);
      if (rm_valid && rm_b[W-1]) $fatal(1, "gpnae_poly_int8: gp_mx_i must be below 2^15");
    end
`endif

endmodule
```

Timing, from a `G_LOAD` issue at cycle c (latencies 1, 2, 3 as in the package): `rm_valid` at c+1, product and
`rs_v[1]` at c+2; SELU and sigmoid operands reach the MAC (`ld_valid`) at c+3; tanh's square is issued at c+3, done at
c+5, loaded at c+6. After the last element `G_LDRAIN` starts the MAC two cycles after the last load (c+5, or c+8 for
tanh). From a `G_POST` multiply at c: `pm_valid` at c+1, product at c+2 (tanh quantizes it there; SELU passes it to
the requantizer's `valid_i`), SELU result at c+5. `G_EMIT` waits on `res_rdy`, so the post latencies need no counting.
The model (Task 10) does the same operations in the same order: rescale, operand, Horner, then the post step.

- [ ] **Step 3: Put it under `gpnae_poly`, reject int8 in `gpnae_tail`, add it to the Makefile**

In `src/gpnae_poly.sv`, change the `EXP_W` parameter comment to `// the build's number format: fp32 8/23, bf16 8/7, int8 0/7`,
and add these ports after `control_word_i`:

```systemverilog
    input logic [15:0] gp_mx_i,     // int8 only (D-2): input rescale multiplier, below 2^15
    input logic [ 4:0] gp_shx_i,    // int8: input rescale shift
    input logic [ 7:0] gp_zin_i,    // int8: input zero point
    input logic [31:0] gp_mout_i,   // int8: SELU output multiplier
    input logic [ 7:0] gp_shout_i,  // int8: SELU output shift
    input logic [ 7:0] gp_zout_i,   // int8: SELU output zero point
```

Directly after the `G_BAD_FORMAT` block (`end` of `if (!sienna_fmt_pkg::supported(...))`), insert:

```systemverilog
  // int8 builds use the fixed-point lane; the float lane below is unchanged, only wrapped in G_FLOAT.
  if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT8
    gpnae_poly_int8 #(
        .DATA_WIDTH   (DATA_WIDTH),
        .ADDR_LINES   (ADDR_LINES),
        .CONTROL_WIDTH(CONTROL_WIDTH),
        .K            (K)
    ) lane_inst (
        .clk_i         (clk_i),
        .rstn_i        (rstn_i),
        .signal_i      (signal_i),
        .wr_en_i       (wr_en_i),
        .last_i        (last_i),
        .terms_i       (terms_i),
        .control_word_i(control_word_i),
        .gp_mx_i       (gp_mx_i),
        .gp_shx_i      (gp_shx_i),
        .gp_zin_i      (gp_zin_i),
        .gp_mout_i     (gp_mout_i),
        .gp_shout_i    (gp_shout_i),
        .gp_zout_i     (gp_zout_i),
        .full_o        (full_o),
        .empty_o       (empty_o),
        .idle_o        (idle_o),
        .final_result_o(final_result_o),
        .done_o        (done_o)
    );
  end else begin : G_FLOAT
```

and directly before `endmodule`:

```systemverilog
  end  // G_FLOAT
```

Do not re-indent the float body: `git diff` of `gpnae_poly.sv` must show only the comment, the six ports, the `G_INT8`
block, the `G_FLOAT` begin line and its `end`. If Verilator rejects the `typedef enum ... gstate_t;` inside the generate
block, move that typedef (only a type) above the `if`, and say so in the commit message.

In `src/gpnae_tail.sv`, change the first generate condition to reject int8, since the tail has no integer datapath and
`supported(0, 7)` is now true:

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W) || sienna_fmt_pkg::is_int(EXP_W)) begin : G_BAD_FORMAT
```

In `Makefile`'s `DESIGN_FILES`, add `gpnae_poly_int8.sv \` on the line before `gpnae_poly.sv`.

In `testbenches/TB_gpnae_poly.sv`, change both `dut.G_MAC.barrel_mac_inst` to `dut.G_FLOAT.G_MAC.barrel_mac_inst`,
and tie the new ports off in the DUT instance, after `.control_word_i(control_word_i),`:

```systemverilog
      .gp_mx_i('0),
      .gp_shx_i('0),
      .gp_zin_i('0),
      .gp_mout_i('0),
      .gp_shout_i('0),
      .gp_zout_i('0),
```

Re-run `grep -rn "G_MAC\|barrel_mac_inst" --include=*.sv --include=*.py /proj/work/spramanik/SIENNA_int8 | command grep -v Verilator`
before committing: only `gpnae_poly.sv`, `gpnae_poly_int8.sv`, `gpnae.sv` (its own instance) and TB_gpnae_poly may appear.

- [ ] **Step 4: Run the checks**

```bash
$J/snap_launch_tree.sh i11_l8 32 1 $J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=0 -GMAN_W=7"
$J/snap_launch_tree.sh i11_l16 32 1 $J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=8 -GMAN_W=7"
$J/snap_launch_tree.sh i11_l32 32 1 $J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=8 -GMAN_W=23"
$J/snap_launch_tree.sh i11_bad 32 1 $J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=5 -GMAN_W=10" "gpnae_poly: unsupported format"
$J/snap_launch_tree.sh i11_w16 32 1 $J/cmds/int8_lint.sh gpnae_poly_int8 "-GDATA_WIDTH=16" "gpnae_poly_int8: DATA_WIDTH must be 8"
$J/snap_launch_tree.sh i11_tail 32 1 $J/cmds/int8_lint.sh gpnae_tail "-GEXP_W=0 -GMAN_W=7" "gpnae_tail: unsupported format"
$J/snap_launch_tree.sh i11_fp 32 6 $J/cmds/int8_fpref.sh
```

Then `$J/cmds/int8_cmp_gpnae.sh $J/runs/i0_fpref/results/int8/fpref $J/runs/i11_fp/results/int8/fpref`.

Expected:
- `i11_l8`, `i11_l16`, `i11_l32`: `errors=0` (a `Cannot find file containing module` names an AriL file missing from
  `DESIGN_FILES`; add it in the Makefile commit);
- `i11_bad`, `i11_w16`, `i11_tail`: `REJECTED`;
- the compare: `IDENTICAL` for all five tags (the wrap and the new ports change nothing in fp32 or bf16).

- [ ] **Step 5: Commit (GPNAE)**

```bash
git add src/gpnae_poly_int8.sv && git commit -m "gpnae_poly_int8: int8 lane, the float forms on fxMac and integer products, saturation, int8 quantize and SELU requantize"
git add src/gpnae_poly.sv && git commit -m "gpnae_poly: int8 builds take gpnae_poly_int8 (G_INT8); per-layer int8 ports; float lane wrapped in G_FLOAT"
git add src/gpnae_tail.sv && git commit -m "gpnae_tail: reject int8, which has no tail"
git add Makefile && git commit -m "Makefile: gpnae_poly_int8"
git add testbenches/TB_gpnae_poly.sv && git commit -m "TB_gpnae_poly: the float lane's G_FLOAT hierarchy, int8 ports tied off"
git push origin int8
```

### Task 12: TB_gpnae_poly int8 mode, TFLite agreement, gate G2

Starts only after Task 10's `VERDICT: PASS` and Task 11.

**Files:**
- Modify: `testbenches/TB_gpnae_poly.sv` (int8 mode), `regression.py` (`--format int8`, `REQ_ROUNDING` in the header)
- Modify (SIENNA): `tflite_oracle.py` (append `activation_int8`)
- Create (SIENNA): `gpnae_int8_tflite.py`
- Create (no repo): `$J/cmds/int8_tfl_act.sh`, `$J/cmds/int8_gpnae_gate.sh`
- Create (SIENNA, untracked): `testbenches/results/int8/gpnae_gate.log`
- Create (SIENNA, committed): `testbenches/int8/gpnae_int8_accuracy.json` (the lane's measured accuracy, which the
  report builder reads in Task 22; outside `results/`)

**Interfaces:**
- Consumes: everything above; `tflite_oracle.py`, `tflite_ref.quantize_multiplier` (Task 2); TensorFlow in
  `$J/venv` (Task 0).
- Produces:
  - `python3 regression.py --lane poly --format int8 [--seed S]`: every int8 input once per case of
    `gpnae_model.INT8_CASES` (5 cases x 256 per activation, SELU, sigmoid, tanh, ReLU, linear), bit-exact; prints and
    reports `ACCURACY: PASS|MISSED`; exit 1 on any mismatch. Report `testbenches/results/gpnae_int8_report.log`, raw TB
    output `testbenches/results/int8_exhaustive.log`, and `testbenches/results/gpnae_int8_accuracy.json`:
    `{"seed", "bitexact", "activations": {selu|sigmoid|tanh: {"worst_lsb", "cases": [{s_in, z_in, s_out, z_out,
    max_lsb, differ, gated}]}}}`, `worst_lsb` the worst int8 LSB over the gated cases.
  - Header item `localparam string REQ_ROUNDING` (every format). Stimulus `<act>_par.mem`: one 80-bit word per batch,
    `{mx[15:0], shx[7:0], zin[7:0], mout[31:0], shout[7:0], zout[7:0]}`.
  - `tflite_oracle.activation_int8(op, in_scale, in_zp) -> (outputs[256] int64, in_scale, in_zp)` for `op` in
    `"tanh"`, `"logistic"`: TFLite's int8 op (reference kernels) on q = -128..127, with the input quantization the
    converter actually chose.
  - `gpnae_int8_tflite.py --report FILE` (SIENNA root, TensorFlow venv).
  - Gate report `testbenches/results/int8/gpnae_gate.log` (SIENNA).

- [ ] **Step 1: See the int8 run fail**

Run: `$J/snap_launch_tree.sh i12_int8 32 2 $J/cmd_gpnae_reg.sh --lane poly --format int8`
Expected: `argument --format: invalid choice: 'int8'`.

- [ ] **Step 2: Add the int8 mode to TB_gpnae_poly**

Replace `localparam int ADDR_LINES = 5;` and `localparam int CONTROL_WIDTH = 2;` with:

```systemverilog
  localparam bit IS_INT = (EXP_BITS == 0);  // int8 build: exact int8 compare, lane parameters per batch
  localparam int ADDR_LINES = 5;
  localparam int CONTROL_WIDTH = IS_INT ? 3 : 2;  // int8 also runs ReLU (100) and linear (101)
```

After `reg [CONTROL_WIDTH-1:0] control_word_i;` add:

```systemverilog
  reg [15:0] gp_mx = '0;  // int8 lane parameters, one set per batch from <act>_par.mem
  reg [7:0] gp_shx = '0, gp_zin = '0, gp_shout = '0, gp_zout = '0;
  reg [31:0] gp_mout = '0;
  reg [79:0] par[0:NUM_BATCHES-1];
```

In the DUT instance, replace the six Task 11 tie-offs with:

```systemverilog
      .gp_mx_i(gp_mx),
      .gp_shx_i(gp_shx[4:0]),
      .gp_zin_i(gp_zin),
      .gp_mout_i(gp_mout),
      .gp_shout_i(gp_shout),
      .gp_zout_i(gp_zout),
```

Replace the `BIAS` and `EXP_MAX` localparams and the `exp = ...` line in `fp_to_real` with:

```systemverilog
  localparam int EW = (EXP_BITS > 0) ? EXP_BITS : 1;  // int8 has no exponent; fp_to_real is not called there
  localparam int BIAS = (1 << (EW - 1)) - 1;
  localparam int EXP_MAX = (1 << EW) - 1;
```

```systemverilog
    exp  = int'(b[DATA_WIDTH-2-:EW]);
```

Replace the coefficient ROM check (`initial begin ... $readmemb(COEFF_FILE, want); ... end`) with:

```systemverilog
  // The lane must have loaded this format's coefficient table; int8's is 16-bit Q4.11.
  if (IS_INT) begin : G_ROM_INT
    initial begin
      logic [15:0] want[32];
      #1;
      $readmemb(COEFF_FILE, want);
      for (int i = 0; i < 32; i++)
        if (dut.G_INT8.lane_inst.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i] !== want[i])
          $fatal(1, "coefficient ROM[%0d] is %h, %s has %h", i, dut.G_INT8.lane_inst.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i],
                 COEFF_FILE, want[i]);
    end
    // The requantizer in the RTL and the model's must round alike (the variant pinned at G0).
    initial if (sienna_fmt_pkg::REQ_ROUNDING != REQ_ROUNDING)
      $fatal(1, "RTL requantize rounds %s, the model %s", sienna_fmt_pkg::REQ_ROUNDING, REQ_ROUNDING);
  end else begin : G_ROM_FP
    initial begin
      logic [DATA_WIDTH-1:0] want[32];
      #1;
      $readmemb(COEFF_FILE, want);
      for (int i = 0; i < 32; i++)
        if (dut.G_FLOAT.G_MAC.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i] !== want[i])
          $fatal(1, "coefficient ROM[%0d] is %h, %s has %h", i, dut.G_FLOAT.G_MAC.barrel_mac_inst.coeff_rom_inst.ROM.ROM[i],
                 COEFF_FILE, want[i]);
    end
  end
```

If Verilator resolves the path of the branch that is not built and fails, keep both checks but guard each path with
`` `ifdef GPNAE_INT8 `` / `` `else ``, and have `run_int8` pass `EXTRA_FLAGS=-DGPNAE_INT8` (append to any existing
`EXTRA_FLAGS`) to `make`.

In `reset_sequence`, change `control_word_i = 2'b00;` to `control_word_i = '0;`. Change the task header to
`task automatic run_activation(input [CONTROL_WIDTH-1:0] ctrl, input string act_name,`. In `run_activation`, after the
two `$readmemh` lines, add:

```systemverilog
      if (IS_INT) $readmemh({STIM_DIR, act_name, "_par.mem"}, par);
```

and after `terms_i        = n_terms;` add:

```systemverilog
        if (IS_INT) {gp_mx, gp_shx, gp_zin, gp_mout, gp_shout, gp_zout} = par[b];
```

After `run_activation(2'b11, "tanh", TANH_TERMS);` add:

```systemverilog
    if (IS_INT) begin
      run_activation(3'b100, "relu", TANH_TERMS);
      run_activation(3'b101, "linear", TANH_TERMS);
    end
```

Float builds see none of this at run time: `IS_INT` is 0, `CONTROL_WIDTH` stays 2, and the display, reset and timing
are unchanged.

- [ ] **Step 3: Add the int8 flow to regression.py**

In `write_config`, add the parameter `req_rounding: str = None` and, as its last write:

```python
        if req_rounding is None:
            import gpnae_model
            req_rounding = gpnae_model.ipu.REQ_ROUNDING
        f.write(f'  localparam string REQ_ROUNDING  = "{req_rounding}";\n')
```

(The float headers gain the line; the float TB does not read it.) Change `--format` to
`choices=sorted(FORMATS) + ["int8"]`, and directly after the `--per-batch` check in `main()` add:

```python
    if args.format == "int8":
        sys.exit(run_int8(args))
```

Add before `main()`:

```python
INT8_ACTS = (("selu", 1), ("sigmoid", 2), ("tanh", 3), ("relu", 4), ("linear", 5))  # TB_gpnae_poly's order in int8


def run_int8(args) -> int:
    """int8 lane: every int8 input at each gpnae_model.INT8_CASES case, bit-exact against the lane model, then accuracy."""
    import json
    import types

    import gpnae_model as gm
    if args.lane != "poly":
        print(err("[ERROR] the int8 lane is gpnae_poly; use --lane poly"))
        return 1
    per = MAX_SIGNALS
    n_cases = len(gm.INT8_CASES["tanh"])
    assert all(len(gm.INT8_CASES[a]) == n_cases for a, _ in INT8_ACTS), "every activation needs the same number of cases"
    batches = n_cases * 256 // per
    fmt = types.SimpleNamespace(name="int8", width=8, exp_bits=0, man_bits=7)
    coeff = gm.coeff_file(gm.INT8)
    lane = gm.Lane(gm.INT8, gm.read_rom(os.path.join(ROOT, coeff)))
    write_config(fmt, batches, per, 0.0, 0.0, "hw", args.timeout, args.seed, True, coeff)
    rs = np.random.RandomState(args.seed)
    os.makedirs(STIM_DIR, exist_ok=True)
    rows = []
    for act, code in INT8_ACTS:
        stim, gold, par = [], [], []
        for case in gm.INT8_CASES[act]:
            p = gm.int8_params(case, code)
            q = rs.permutation(256).astype(np.int64) - 128  # every int8 input once, in a seeded order
            y = lane.run(q, code, p)
            stim += [int(v) for v in q]
            gold += [int(v) for v in y]
            par += [p] * (256 // per)
            if code <= 3:
                d = np.abs(y - gm.exact_int8(q, code, case))
                rows.append((act, case, p, int(d.max()), int((d > 0).sum())))
        with open(os.path.join(STIM_DIR, f"{act}_in.mem"), "w") as fh:
            fh.write("".join(f"{v & 0xFF:02x}\n" for v in stim))
        with open(os.path.join(STIM_DIR, f"{act}_exp.mem"), "w") as fh:
            fh.write("".join(f"{v & 0xFF:02x}\n" for v in gold))
        with open(os.path.join(STIM_DIR, f"{act}_par.mem"), "w") as fh:
            fh.write("".join(f"{p.mx:04x}{p.shx:02x}{p.zin & 0xFF:02x}{p.mout:08x}{p.shout & 0xFF:02x}{p.zout & 0xFF:02x}\n"
                             for p in par))
    print(hdr(f"\n{'='*78}\n  GPNAE int8 lane: {n_cases} cases x 256 inputs per activation, seed {args.seed}\n{'='*78}"))
    raw, wall = run_make("poly")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "int8_exhaustive.log"), "w") as fh:
        fh.write(raw)
    parsed = parse_log(raw, expect_total=batches * per, expect_acts=len(INT8_ACTS))
    for act, a in parsed["per_act"].items():
        print_activation(act, a)
    exact_ok = parsed["status"] == "PASS"
    worst = max(r[3] for r in rows if r[1].gated)
    L = [f"GPNAE int8 lane, seed {args.seed}: bit-exact against gpnae_model, then the lane against the exact functions",
         f"BITEXACT: {'PASS' if exact_ok else parsed['status']}  "
         + "  ".join(f"{a} {v['exact']}/{v['total']}" for a, v in parsed["per_act"].items()),
         f"{'act':<8}{'s_in':>10}{'z_in':>6}{'mx':>7}{'shx':>4}{'s_out':>10}{'z_out':>6}{'mout':>12}{'shout':>6}"
         f"{'max LSB':>8}{'differ':>7}{'gated':>6}"]
    for act, case, p, dmax, ndiff in rows:
        L.append(f"{act:<8}{case.s_in:>10.6f}{case.z_in:>6}{p.mx:>7}{p.shx:>4}{case.s_out:>10.6f}{case.z_out:>6}"
                 f"{p.mout:>12}{p.shout:>6}{dmax:>8}{ndiff:>7}{'yes' if case.gated else 'no':>6}")
    L.append(f"ACCURACY: {'PASS' if worst <= 1 else 'MISSED'} (worst {worst} int8 LSB on gated cases, target 1)")
    with open(os.path.join(RESULTS_DIR, "gpnae_int8_report.log"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    summary = {}  # per activation: the worst int8 LSB over the gated cases, and every case
    for act, case, p, dmax, ndiff in rows:
        s = summary.setdefault(act, {"worst_lsb": 0, "cases": []})
        s["cases"].append({"s_in": float(case.s_in), "z_in": int(case.z_in), "s_out": float(case.s_out),
                           "z_out": int(case.z_out), "max_lsb": dmax, "differ": ndiff, "gated": bool(case.gated)})
        if case.gated:
            s["worst_lsb"] = max(s["worst_lsb"], dmax)
    with open(os.path.join(RESULTS_DIR, "gpnae_int8_accuracy.json"), "w") as fh:
        json.dump({"seed": args.seed, "bitexact": exact_ok, "activations": summary}, fh, indent=1)
    print("\n".join(L[1:]))
    print(f"  {_D}{wall:.1f}s{_X}")
    return 0 if exact_ok else 1
```

The accuracy numbers come from the model; they are the RTL's because the run is bit-exact first.

- [ ] **Step 4: Run the int8 lane**

```bash
$J/snap_launch_tree.sh i12_int8 32 2 $J/cmd_gpnae_reg.sh --lane poly --format int8
```

Expected in `runs/i12_int8/stdout.log`: five activation lines with `fail 0 miss 0` and exact 100%, `BITEXACT: PASS`
with `1280/1280` each, `ACCURACY: PASS (worst 1 int8 LSB ...)` or better, and no `$fatal`.

A mismatch is a bug in `gpnae_poly_int8.sv` (Task 11) or in the model (Task 10); the units under both are proven
(Tasks 4 to 7, TB_barrel_mac_int8). The failure print gives batch `b` (case `b // 8`), the input and both outputs:
recompute that element with `gm.rescale`, `gm.mac_operand`, `gm.horner` and the post step in Python and compare with the RTL
signals (`+dump` scopes the VCD to the DUT). A latency `$fatal` names the unit whose Level 1 latency changed.
`ACCURACY: MISSED` with `BITEXACT: PASS` is an accuracy result, not a bug: report it (Task 10's fit already gated it).

- [ ] **Step 5: Add the TFLite helper and the agreement report**

Append to `tflite_oracle.py` (SIENNA root):

```python
def activation_int8(op, in_scale, in_zp):
    """TFLite's int8 TANH or LOGISTIC (reference kernels) on every int8 input; returns (outputs, input scale, input zero point)."""
    import numpy as np
    import tensorflow as tf
    lo, hi = in_scale * (-128 - in_zp), in_scale * (127 - in_zp)
    grid = np.linspace(lo, hi, 256, dtype=np.float32).reshape(1, 256)
    model = tf.keras.Sequential([tf.keras.Input(shape=(256,)),
                                 tf.keras.layers.Activation({"tanh": "tanh", "logistic": "sigmoid"}[op])])
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: iter([[grid]])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    interp = tf.lite.Interpreter(model_content=conv.convert(),
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    ops = {d["op_name"] for d in interp._get_ops_details()}
    assert ops == {op.upper()}, f"expected one int8 {op.upper()} op, got {ops}"
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    want = {"tanh": (1 / 128, 0), "logistic": (1 / 256, -128)}[op]
    assert abs(o["quantization"][0] - want[0]) < 1e-12 and o["quantization"][1] == want[1], o["quantization"]
    interp.set_tensor(i["index"], np.arange(-128, 128, dtype=np.int8).reshape(1, 256))
    interp.invoke()
    s, z = i["quantization"]
    return interp.get_tensor(o["index"]).reshape(256).astype(np.int64), float(s), int(z)
```

`gpnae_int8_tflite.py` (SIENNA root):

```python
#!/usr/bin/env python3
"""GPNAE's int8 tanh and sigmoid (the lane model, bit-exact with the RTL at G2) against TFLite's int8 TANH and LOGISTIC; reported, not gated."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "GPNAE"))
import gpnae_model as gm  # noqa: E402
import tflite_oracle  # noqa: E402
import tflite_ref  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    lane = gm.Lane(gm.INT8, gm.read_rom(os.path.join(ROOT, "GPNAE", gm.coeff_file(gm.INT8))))
    q = np.arange(-128, 128, dtype=np.int64)
    L = ["GPNAE int8 lane against TFLite int8 (BUILTIN_REF), every int8 input; the input quantization is the converter's",
         f"{'op':<9}{'s_in':>11}{'z_in':>6}{'equal':>7}{'|d|=1':>7}{'max |d|':>8}{'TFLite-exact':>13}{'lane-exact':>11}"]
    for act, code, op in (("tanh", 3, "tanh"), ("sigmoid", 2, "logistic")):
        for case in gm.INT8_CASES[act]:
            tfl, s, z = tflite_oracle.activation_int8(op, case.s_in, case.z_in)
            c = case._replace(s_in=s, z_in=z)
            hw = lane.run(q, code, gm.int8_params(c, code))
            ex = gm.exact_int8(q, code, c)
            d = np.abs(hw - tfl)
            L.append(f"{op:<9}{s:>11.6f}{z:>6}{int((d == 0).sum()):>7}{int((d == 1).sum()):>7}{int(d.max()):>8}"
                     f"{int(np.abs(tfl - ex).max()):>13}{int(np.abs(hw - ex).max()):>11}")
    reals = 10.0 ** np.random.RandomState(1).uniform(-9, 0, 10000)
    bad = [r for r in reals if tuple(gm.quantize_multiplier(r)) != tuple(int(v) for v in tflite_ref.quantize_multiplier(r))]
    L.append(f"quantize_multiplier: gpnae_model against tflite_ref on {len(reals)} reals in [1e-9, 1): {len(bad)} differ")
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

`$J/cmds/int8_tfl_act.sh`:

```bash
#!/bin/bash
# GPNAE int8 tanh and sigmoid against TFLite's int8 TANH and LOGISTIC, reported not gated; run from a snapshot root.
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/int8; mkdir -p $R
$J/venv/bin/python3 gpnae_int8_tflite.py --report $R/gpnae_vs_tflite.log
```

Run: `$J/snap_launch_tree.sh i12_tfl 32 2 $J/cmds/int8_tfl_act.sh`
Expected: `runs/i12_tfl/results/int8/gpnae_vs_tflite.log` with ten rows (5 tanh, 5 logistic) and
`quantize_multiplier: ... 0 differ`, exit 0. The agreement counts are reported as they come; they are not gated. A
non-zero `differ` is a bug in `gm.quantize_multiplier`: make it follow `tflite_ref`'s (Task 2 validated that one against
the interpreter). If the converter does not give a single int8 op (the `assert`), record the op list in the report
instead and tell Soham; do not work around it silently.

- [ ] **Step 6: The gate job**

`$J/cmds/int8_gpnae_gate.sh`:

```bash
#!/bin/bash
# Gate G2 (int8): fp32 and bf16 GPNAE unchanged, int8 lane bit-exact on every int8 input per case, barrel MAC, random power-up, rejections, lint; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs; ROOT=$(pwd); R=$ROOT/testbenches/results/int8/g2; mkdir -p $R
run() {  # tag, then regression.py arguments; keeps only this run's logs and accuracy summary
  tag=$1; shift; touch $R/.t_$tag
  $J/cmd_gpnae_reg.sh "$@" > $R/$tag.txt 2>&1 || echo "RUN-FAIL $tag"
  mkdir -p $R/$tag && find GPNAE/testbenches/results -maxdepth 1 \( -name '*.log' -o -name '*.json' \) -newer $R/.t_$tag -exec cp {} $R/$tag/ \;
}
$J/cmds/int8_fpref.sh
for s in 1 101 202 303; do run int8_seed$s --lane poly --format int8 --seed $s; done
$J/cmds/int8_bm.sh 5 > $R/bm_int8.txt 2>&1 || echo "RUN-FAIL bm_int8"
export EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" SIM_ARGS="+verilator+rand+reset+2"
run int8_randinit --lane poly --format int8 --seed 7
run bf16_hw_randinit --lane poly --format bf16 --model hw
run poly32_randinit --lane poly --format fp32
$J/cmds/int8_bm.sh 11 > $R/bm_int8_randinit.txt 2>&1 || echo "RUN-FAIL bm_int8_randinit"
unset EXTRA_FLAGS SIM_ARGS
$J/cmds/int8_lint.sh barrel_mac "-GEXP_W=5 -GMAN_W=10" "barrel_mac: unsupported format" > $R/rej_barrel_mac_5_10.txt 2>&1
$J/cmds/int8_lint.sh barrel_mac "-GEXP_W=0 -GMAN_W=7 -GDATA_WIDTH=8" "barrel_mac: int8 evaluates in Q4.11" > $R/rej_barrel_mac_w8.txt 2>&1
$J/cmds/int8_lint.sh gpnae_poly "-GEXP_W=5 -GMAN_W=10" "gpnae_poly: unsupported format" > $R/rej_gpnae_poly_5_10.txt 2>&1
$J/cmds/int8_lint.sh gpnae_poly_int8 "-GDATA_WIDTH=16" "gpnae_poly_int8: DATA_WIDTH must be 8" > $R/rej_gpnae_poly_int8_w16.txt 2>&1
$J/cmds/int8_lint.sh gpnae_tail "-GEXP_W=0 -GMAN_W=7" "gpnae_tail: unsupported format" > $R/rej_gpnae_tail_0_7.txt 2>&1
for f in "0 7" "8 7" "8 23"; do set -- $f
  $J/cmds/int8_lint.sh gpnae_poly "-Wall -GEXP_W=$1 -GMAN_W=$2" > $R/lint_$1_$2.txt 2>&1
done
(cd GPNAE && for f in fp32 bf16; do python3 fit_poly_coeffs.py --format $f; echo "fit $f exit=$?"; done) > $R/fit_guard.txt 2>&1
command grep -H "^REJECTED\|^NOT-REJECTED" $R/rej_*.txt
for f in $R/lint_*.txt; do echo "$(basename $f): $(command grep -c '^%Warning' $f) warnings, $(command grep -c '^%Error' $f) errors; $(command grep -hoE '^%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)' $f | sort | uniq -c | tr '\n' ' ')"; done
command grep -h "exit=" $R/fit_guard.txt
command grep -h "RESULT: \|BITEXACT:\|ACCURACY:" $R/*.txt $R/*/gpnae_int8_report.log
```

Launch: `$J/snap_launch_tree.sh i12_g2 32 12 $J/cmds/int8_gpnae_gate.sh`. Then, on the login node:

```bash
$J/cmds/int8_cmp_gpnae.sh $J/runs/i0_fpref/results/int8/fpref $J/runs/i12_g2/results/int8/fpref
a=$(git -C $T/SystolicMesh/ArithmeticLibrary rev-parse HEAD); b=$(git -C $T/GPNAE/ArithmeticLibrary rev-parse HEAD)
[ "$a" = "$b" ] && echo "SAME-ARIL $a" || echo "DIFFERENT-ARIL $a $b"
```

Expected:
- `SAME-ARIL` (both AriL checkouts on the G1 tip, which the snapshot carried);
- no `RUN-FAIL` in `runs/i12_g2/stdout.log`;
- the compare: `IDENTICAL` for poly32, poly32_r8, taylor32, bf16_hw, bf16_hw_r8 (results and cycles);
- the four int8 seeds and `int8_randinit`: `BITEXACT: PASS` and `ACCURACY: PASS`; `bf16_hw_randinit` and
  `poly32_randinit`: `RESULT: PASSED`; both barrel MAC runs `RESULT: PASSED`;
- five `REJECTED` lines, no `NOT-REJECTED`;
- lint: no `LATCH`, `MULTIDRIVEN`, `UNOPTFLAT` or `%Error` in any of the three formats;
- `fit fp32 exit=1`, `fit bf16 exit=1`.

- [ ] **Step 7: Write the gate report**

Write `testbenches/results/int8/gpnae_gate.log` (SIENNA root) by hand from `runs/i12_g2/`, `runs/i12_tfl/`,
`runs/i10_fit/` and `runs/i0_fpref/`, in the layout of `2026-09-28_gpnae_gate.txt` (sienna-report history):
- the commits (GPNAE, its ArithmeticLibrary, SIENNA);
- verdict;
- fp32 and bf16 unchanged: the compare output, and the random power-up runs;
- int8 bit-exact: per activation and seed, exact over total; the barrel MAC TB (groups, errors, cycles);
- int8 accuracy: from `poly_coeffs_int8_fit.log`, per activation and candidate range the cases' worst LSB and the
  range's worst and mean error by degree, the chosen degrees and `SETS_INT8`; the float-lane-range rows as the cost of
  having no tail; and the per-case int8 table (worst LSB, count differing) from `gpnae_int8_report.log`;
- agreement with TFLite (`gpnae_vs_tflite.log`), labelled "reported, not gated";
- cycles per input for int8 against fp32 and bf16 (the `CYCLES` lines);
- rejections and lint counts;
- the level's decisions (L2-1 to L2-5) and what was not run: the Taylor lane (`gpnae.sv`) in int8, SIENNA-level int8
  (Level 4), timing of the lane's rescale path, SELU inputs past Q4.11 for accuracy (bit-exact only).

Every number is copied from a log, with the log's path next to it.

Then bring the accuracy summary into the tree, where the report builder (Task 22) reads it; check first that the target
does not exist:

```bash
D=/proj/work/spramanik/SIENNA_int8/testbenches/int8
ls $D/gpnae_int8_accuracy.json 2>/dev/null && echo "EXISTS: look before copying"   # expect nothing
mkdir -p $D && cp $J/runs/i12_g2/results/int8/g2/int8_seed1/gpnae_int8_accuracy.json $D/
python3 -c "import json; d = json.load(open('$D/gpnae_int8_accuracy.json')); print({k: v['worst_lsb'] for k, v in d['activations'].items()})"
```

Expected: `selu`, `sigmoid` and `tanh` with the worst LSB the seed-1 run's `ACCURACY:` line reports (at most 1).

- [ ] **Step 8: Commit and report**

```bash
cd /proj/work/spramanik/SIENNA_int8/GPNAE
git add testbenches/TB_gpnae_poly.sv && git commit -m "TB_gpnae_poly: int8 mode, per-batch lane parameters, int8 ROM and rounding checks, ReLU and linear"
git add regression.py && git commit -m "regression: --format int8, every int8 input per case bit-exact, accuracy against the exact functions"
git push origin int8
cd /proj/work/spramanik/SIENNA_int8
git add tflite_oracle.py && git commit -m "tflite_oracle: activation_int8, TFLite's int8 TANH and LOGISTIC on every input"
git add gpnae_int8_tflite.py && git commit -m "gpnae_int8_tflite: GPNAE int8 tanh and sigmoid against TFLite, reported"
git add testbenches/int8/gpnae_int8_accuracy.json && git commit -m "G2: the int8 lane's measured accuracy per activation, for the report"
git push origin int8
```

SIENNA's GPNAE submodule pointer moves in Level 4 (Task 16), with the lane's new ports. Give Soham the gate summary
and the report's path; Level 3 starts after Soham has seen G2.

---

## Level 3: SystolicMesh (gate G3 in Task 15)

Paths are relative to `/proj/work/spramanik/SIENNA_int8/SystolicMesh` on branch `int8`, except where a step says SIENNA
(`/proj/work/spramanik/SIENNA_int8`) or `$J`. Every farm command in this level assumes:

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/work/spramanik/SIENNA_int8
export TREE=$T   # snap_launch_tree.sh snapshots this tree, never the bf16 checkout
```

Farm run names: `m13_*` (Task 13), `m14_*` (Task 14), `g3i_*` (Task 15). The bf16 branch's G3 runs, which are the fp32 and
bf16 cycle reference, are the run directories in `$J/runs` with the prefixes `g3_`, `g3m_`, `g3n_`, `g3x_`, `g3y_`, `g3z_`
(Task 0 records which of them finished).

Why the mesh changes look the way they do: int8 needs a second width. Operands stay `DATA_WIDTH` (8) from the host
through the staging banks, the weight cache and the arrays' operand banks. Everything that holds a sum (PE partial sums,
the reducer tree, the bias queue and bias input, the result memory, both read ports) becomes `ACC_W`
(`sienna_fmt_pkg::acc_w`: 32 in int8, `DATA_WIDTH` in fp32 and bf16). In the float formats `ACC_W == DATA_WIDTH`, so
every float netlist keeps the same widths and the same units, and only the int8 branch is new.

### Task 13: int8 through the mesh RTL

**Files:**
- Modify: `ArithmeticLibrary` pointer (AriL `int8` tip from Task 8), `Makefile` (`DESIGN_FILES`)
- Modify: `src/engine/ProcessingElement.sv`, `src/engine/AccumulationUnit.sv`, `src/top/SystolicArray.sv`, `src/top/SystolicMesh.sv`
- No change: `src/mem/MeshOutputSram.sv` (already parameterized by `DATA_WIDTH`; the mesh instantiates it with `ACC_W`)
- Create: `testbenches/TB_PE_int8.sv`
- Create: `$J/cmds/int8_mesh.sh`
- Modify (SIENNA, committed in Task 15 with the SystolicMesh pointer bump): `Makefile` (`SM_LIB_FILES`), `synth/sienna_rtl.f`

**Interfaces:**
- Consumes (Task 3): `sienna_fmt_pkg::is_int(exp_w)`, `supported(0, 7) == 1`, `acc_w(exp_w, man_w)`, `mul_lat(0, 7) == 1`,
  `add_lat(0, 7) == 1`. (Task 4) `intMultiplier #(W = 8)` with `clk_i, rstn_i, valid_i, A, B [W-1:0], result_o [2*W-1:0],
  done_o`, latency 1. (Task 5) `intAdder #(W = 32)` with `clk_i, rstn_i, valid_i, A, B [W-1:0], result_o [W-1:0], done_o`,
  latency 1. (Task 8) the AriL `int8` tip, pushed.
- Produces:
  - `SystolicMesh #(MATRIX_SIZE, TILE_SIZE, EXP_W = 8, MAN_W = 23, DATA_WIDTH = 1 + EXP_W + MAN_W, ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W), ...)`.
    `ACC_W` wide: `bias_i [MATRIX_SIZE-1:0][ACC_W-1:0]`, `read_data_o [ACC_W-1:0]`, `wide_read_data_o [WIDE_READ-1:0][ACC_W-1:0]`.
    `DATA_WIDTH` wide: `north_write_data_i`, `west_write_data_i` (and the weight cache they fill). int8 is `EXP_W = 0, MAN_W = 7`.
  - `SystolicArray`, `ProcessingElement`: a new `ACC_W` parameter (same default); `SystolicArray.read_data_o` and
    `ProcessingElement.partial_o` are `[U-1:0][ACC_W-1:0]`. `AccumulationUnit`: its `DATA_WIDTH` parameter is replaced by
    `ACC_W` (every port and the tree). In int8, `U = min(K, add_lat + 1) = min(K, 2)`.
  - Elaboration fails with `$fatal` for an unsupported format (PE, reducer) and for an `ACC_W` other than `acc_w(EXP_W, MAN_W)` (PE, reducer).
  - `$J/cmds/int8_mesh.sh lintall | lintref | pe | unit | same | slint` (results in `testbenches/results/int8/` of the snapshot, copied to `$J/runs/NAME/results/int8/`).

- [ ] **Step 1: Write the failing test and the job script**

`testbenches/TB_PE_int8.sv`:

```systemverilog
`timescale 1ns / 100ps

// ProcessingElement in int8: every slot of every set against the exact int32 sum, sets back to back, two-pass sets, the extremes.
module TB_PE_int8;
  localparam int EXP_W = 0, MAN_W = 7, DW = 8, ACC_W = 32;
  localparam int K = 4, BANKS = 3, BW = 2;
  localparam int U = 2;  // min(K, add_lat(0, 7) + 1)
  localparam int NSETS = 40;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;

  logic [DW-1:0] a = '0, b = '0;
  logic v = 0, fresh = 0, more = 0, rel = 0;
  logic [BW-1:0] rd_bank = '0;
  logic [U-1:0][ACC_W-1:0] partial;
  logic [BANKS-1:0] fin;

  ProcessingElement #(
      .EXP_W(EXP_W), .MAN_W(MAN_W), .DATA_WIDTH(DW), .ACC_W(ACC_W), .K(K), .BANKS(BANKS), .U(U), .BW(BW)
  ) dut (
      .clk_i(clk), .rstn_i(rstn), .a_i(a), .b_i(b), .v_i(v), .fresh_i(fresh), .more_i(more),
      .a_o(), .b_o(), .v_o(), .fresh_o(), .more_o(),
      .rd_bank_i(rd_bank), .partial_o(partial), .release_i(rel), .final_o(fin)
  );

  // Stimulus and expected slots, all computed before reset is released; product n of a set goes to slot n mod U.
  logic signed [DW-1:0] va[NSETS][2*K], vb[NSETS][2*K];
  int npass[NSETS];
  logic [ACC_W-1:0] want[NSETS][U];
  int errs = 0, checked = 0;

  initial begin
    for (int s = 0; s < NSETS; s++) begin
      npass[s] = (s == 0 || s % 5 == 3) ? 2 : 1;  // set 0: two passes of -128 x -128, 4 products per slot = 65536, beyond int16
      for (int u = 0; u < U; u++) want[s][u] = '0;
      for (int n = 0; n < npass[s] * K; n++) begin
        case (s)
          0: begin va[s][n] = 8'h80; vb[s][n] = 8'h80; end
          1: begin va[s][n] = 8'h80; vb[s][n] = 8'h7F; end
          2: begin va[s][n] = 8'h7F; vb[s][n] = 8'h7F; end
          default: begin va[s][n] = DW'($urandom); vb[s][n] = DW'($urandom); end
        endcase
        want[s][n % U] = want[s][n % U] + ACC_W'(longint'(va[s][n]) * longint'(vb[s][n]));
      end
    end
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    fork
      begin
        fork
          begin : feeder
            for (int s = 0; s < NSETS; s++)
              for (int p = 0; p < npass[s]; p++)
                for (int kk = 0; kk < K; kk++) begin
                  @(posedge clk);
                  #1 v = 1;
                  a = va[s][p*K+kk];
                  b = vb[s][p*K+kk];
                  fresh = (p == 0);
                  more = (p < npass[s] - 1);
                end
            @(posedge clk);
            #1 v = 0;
            fresh = 0;
            more = 0;
          end
          begin : reader
            for (int s = 0; s < NSETS; s++) begin
              do begin
                @(posedge clk);
                #1;
              end while (!fin[rd_bank]);
              for (int u = 0; u < U; u++) begin
                checked++;
                if (partial[u] !== want[s][u]) begin
                  errs++;
                  $display("[FAIL] set %0d slot %0d: got %h, want %h", s, u, partial[u], want[s][u]);
                end
              end
              rel = 1;
              @(posedge clk);
              #1 rel = 0;
              rd_bank = (rd_bank == BW'(BANKS - 1)) ? '0 : rd_bank + 1'b1;
            end
          end
        join
      end
      begin : watchdog
        repeat (20000) @(posedge clk);
        $display("[FATAL] TB_PE_int8 timeout");
        $finish;
      end
    join_any
    disable fork;
    $display("TB_PE_int8: %0d sets, %0d slots checked, %0d mismatches", NSETS, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NSETS * U) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

`$J/cmds/int8_mesh.sh` (`chmod +x`):

```bash
#!/bin/bash
# int8 mesh checks on a snapshot; args: lintall | lintref | pe | unit | same | slint; results in testbenches/results/int8; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs
ROOT=$(pwd); R=$ROOT/testbenches/results/int8; mkdir -p "$R"
cd SystolicMesh || exit 1
A=ArithmeticLibrary
F="$A/Common/src/sienna_fmt_pkg.sv $A/Multipliers/Radix4Booth/src/R4Booth.sv $A/Multipliers/Karatsuba/src/karatsubaUnsigned.sv
   $A/Multipliers/FP32/src/fp32Multiplier.sv $A/Multipliers/FP/src/fpMultiplier.sv $A/Adders/FP32/src/LZC.sv
   $A/Adders/FP32/src/fp32Adder.sv $A/Adders/FP/src/fpAdder.sv"
for u in $A/Multipliers/Int/src/intMultiplier.sv $A/Adders/Int/src/intAdder.sv; do [ -f $u ] && F="$F $u"; done  # absent on the bf16 tree
F="$F src/engine/ProcessingElement.sv src/engine/AccumulationUnit.sv src/mem/MeshOutputSram.sv src/top/SystolicArray.sv src/top/SystolicMesh.sv"

lint_one() {  # TAG EXP MAN CK [extra verilator args]; prints CLEAN, REJECTED or NOT-CLEAN
  local tag=$1 e=$2 m=$3 ck=$4; shift 4
  local L=$R/mesh_lint_$tag.txt v=NOT-CLEAN rc
  verilator --lint-only -Wall -Wno-fatal -Werror-USERFATAL -DSYNTHESIS --top-module SystolicMesh -GMATRIX_SIZE=16 -GTILE_SIZE=4 \
    -GEXP_W=$e -GMAN_W=$m -GCOLLAPSE_K=$ck "$@" $F > $L 2>&1
  rc=$?
  if [ $rc -ne 0 ] && command grep -qE "unsupported format|ACC_W=" $L; then v=REJECTED
  elif [ $rc -eq 0 ] && ! command grep -qE "^%Error|%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)" $L; then v=CLEAN; fi
  echo "lint $tag ($e, $m, ck $ck $*): exit=$rc, $(command grep -c '^%Warning' $L) warnings, $(command grep -c '^%Error' $L) errors: $v"
  command grep -hoE "^%Warning-[A-Z]+" $L | sort | uniq -c
  command grep -hE "^%Error|unsupported format|ACC_W=" $L | head -3 | cut -c1-200
}

case $1 in
  lintall)
    lint_one int8 0 7 1; lint_one int8_ck0 0 7 0; lint_one bf16 8 7 1; lint_one fp32 8 23 1
    lint_one bad_5_10 5 10 1; lint_one bad_0_8 0 8 1; lint_one bad_0_15 0 15 1
    lint_one bad_acc_int8 0 7 1 -GACC_W=16; lint_one bad_acc_bf16 8 7 1 -GACC_W=32 ;;
  lintref)
    lint_one bf16 8 7 1; lint_one fp32 8 23 1 ;;
  pe)
    make verilator TOP_MODULE=TB_PE_int8 TRACE=0 VERILATOR_DIR=$PWD/Verilator_pe > $R/tb_pe_int8.log 2>&1; rc=$?
    command grep -hE "TB_PE_int8:|RESULT|\[FAIL\]|%Error|FATAL" $R/tb_pe_int8.log | head -20
    [ $rc -eq 0 ] && command grep -q "RESULT: PASSED" $R/tb_pe_int8.log ;;
  unit)
    python3 test_stim_format.py ;;
  same)
    OLD=../.claude/scratch/mesh_old; W=$(mktemp -d); rc=0
    ln -sfn "$PWD/ArithmeticLibrary" "$OLD/ArithmeticLibrary"
    for FMT in fp32 bf16; do for N in 16 32; do for CK in 1 0; do
      python3 $J/cmds/int8_gen_all.py $FMT $N 4 $CK $W/new/$FMT$N$CK > /dev/null || rc=1
      (cd $OLD && python3 $J/cmds/int8_gen_all.py $FMT $N 4 $CK $W/old/$FMT$N$CK > /dev/null) || rc=1
      if command diff -rq $W/old/$FMT$N$CK $W/new/$FMT$N$CK; then echo "IDENTICAL $FMT N=$N ck=$CK"; else echo "DIFFERENT $FMT N=$N ck=$CK"; rc=1; fi
    done; done; done
    exit $rc ;;
  slint)
    cd "$ROOT" || exit 1
    make lint > $R/sienna_make_lint.txt 2>&1
    echo "SIENNA make lint: exit=$?, $(command grep -c '^%Error' $R/sienna_make_lint.txt) errors"
    verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module sienna_layer -Isrc -f synth/sienna_rtl.f -GN=16 -GNUM_LANES=32 \
      > $R/sienna_rtl_f_lint.txt 2>&1
    echo "sienna_rtl.f lint: exit=$?, $(command grep -c '^%Error' $R/sienna_rtl_f_lint.txt) errors" ;;
  *) echo "usage: int8_mesh.sh lintall|lintref|pe|unit|same|slint"; exit 2 ;;
esac
```

`unit` and `same` are used in Task 14 (`same` needs `$J/cmds/int8_gen_all.py`, created there).

- [ ] **Step 2: Run the test to see it fail**

```bash
$J/snap_launch_tree.sh m13_pe 32 1 $J/cmds/int8_mesh.sh pe
```

Expected: FAIL. The build fails: Verilator reports `%Error-PINNOTFOUND` for the parameter `ACC_W` (the PE has no
`ACC_W` yet; errors from the float units elaborated at `EXP_W = 0` may follow), and there is no `RESULT:` line.

- [ ] **Step 3: Bump AriL, add the integer units to the file list**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh/ArithmeticLibrary
git log -1 --format=%h   # must be Task 8's gate commit
git ls-remote git@github.com:SoHam-56/ArithmeticLibrary.git int8   # must print the same commit: the pointer names a pushed commit
ls Multipliers/Int/src/intMultiplier.sv Adders/Int/src/intAdder.sv
```

In `Makefile` `DESIGN_FILES`, replace the list's last line

```make
  ../ArithmeticLibrary/Adders/FP/src/fpAdder.sv
```

with

```make
  ../ArithmeticLibrary/Adders/FP/src/fpAdder.sv \
  ../ArithmeticLibrary/Multipliers/Int/src/intMultiplier.sv \
  ../ArithmeticLibrary/Adders/Int/src/intAdder.sv
```

The fp32 and bf16 builds need these files too: Verilator links every instantiated module before it evaluates generate
conditions, so the `G_INT` instances must resolve even where the branch is not taken.

- [ ] **Step 4: ProcessingElement**

Replace `src/engine/ProcessingElement.sv` with:

```systemverilog
`timescale 1ns / 100ps

// Output-stationary PE for the pipelined SystolicArray: one product per cycle, sets back to back with no gap.
// Every K products form a pass; a set is one or more passes, accumulated into its own bank of U partial sums, which the reader combines.
module ProcessingElement #(
    parameter int EXP_W      = 8,   // the build's format: fp32 8/23, bf16 8/7, int8 0/7
    parameter int MAN_W      = 23,
    parameter int DATA_WIDTH = 1 + EXP_W + MAN_W,  // operands
    parameter int ACC_W      = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // products and sums: int32 in int8, DATA_WIDTH in the float formats
    parameter int K          = 4,  // products per set
    parameter int BANKS      = 3,  // sets held at once: one accumulating, the older ones finishing or being read
    parameter int U          = (K < sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1) ? K : sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1,  // partial sums per set: the adder latency plus one
    parameter int BW         = (BANKS > 1) ? $clog2(BANKS) : 1
) (
    input  logic                         clk_i,
    input  logic                         rstn_i,
    input  logic [       DATA_WIDTH-1:0] a_i,
    input  logic [       DATA_WIDTH-1:0] b_i,
    input  logic                         v_i,
    input  logic                         fresh_i,     // with v_i: this pass starts a set, its first U products add to 0
    input  logic                         more_i,      // with v_i: another pass of the same set follows this one
    output logic [       DATA_WIDTH-1:0] a_o,
    output logic [       DATA_WIDTH-1:0] b_o,
    output logic                         v_o,
    output logic                         fresh_o,
    output logic                         more_o,
    input  logic [               BW-1:0] rd_bank_i,   // bank the reader looks at
    output logic [U-1:0][     ACC_W-1:0] partial_o,   // that bank's partial sums
    input  logic                         release_i,   // the reader is done with rd_bank_i
    output logic [            BANKS-1:0] final_o      // per bank: a finished set, all adds written back
);
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+ADD_LAT
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+MUL_LAT
  localparam int S = ADD_LAT + 1;  // a slot is read again S cycles after its add issues, one after the write-back
  localparam int SW = (U > 1) ? $clog2(U) : 1;
  localparam int CW = $clog2(K + 1);

`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (U > S || U > K) $error("ProcessingElement: U=%0d must not exceed min(K=%0d, %0d)", U, K, S);
`endif

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      a_o <= '0;
      b_o <= '0;
      v_o <= 1'b0;
      fresh_o <= 1'b0;
      more_o <= 1'b0;
    end else begin
      a_o <= a_i;
      b_o <= b_i;
      v_o <= v_i;
      fresh_o <= fresh_i;
      more_o <= more_i;
    end
  end

  // The pass flags, delayed to meet their product out of the multiplier.
  logic fresh_d[MUL_LAT], more_d[MUL_LAT], v_d[MUL_LAT];
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < MUL_LAT; i++) begin
        fresh_d[i] <= 1'b0;
        more_d[i] <= 1'b0;
        v_d[i] <= 1'b0;
      end
    end else begin
      fresh_d[0] <= fresh_i;
      more_d[0] <= more_i;
      v_d[0] <= v_i;
      for (int i = 1; i < MUL_LAT; i++) begin
        fresh_d[i] <= fresh_d[i-1];
        more_d[i] <= more_d[i-1];
        v_d[i] <= v_d[i-1];
      end
    end
  end
  logic prod_fresh, prod_more;
  assign prod_fresh = fresh_d[MUL_LAT-1];
  assign prod_more  = more_d[MUL_LAT-1];

  logic [ACC_W-1:0] prod, sum;
  logic prod_v, sum_v;

  logic [ACC_W-1:0] acc[BANKS][U];
  logic [BW-1:0] cur;  // bank the next product joins
  logic [SW-1:0] slot;  // partial sum within it
  logic [CW-1:0] n_prod;  // products taken into cur
  logic [BANKS-1:0] taken;  // all K products of the bank issued, not yet released

  // The first product into a slot of a new set is added to zero, so a reused bank needs no clear; later passes add on.
  logic [ACC_W-1:0] add_a;
  assign add_a = (prod_fresh && n_prod < CW'(U)) ? '0 : acc[cur][slot];

  // The multiplier and adder in the build's format; int8 multiplies exactly into int16 and accumulates in int32, wrapping.
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "ProcessingElement: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC_W
    $fatal(1, "ProcessingElement: ACC_W=%0d is not sienna_fmt_pkg::acc_w(%0d, %0d)", ACC_W, EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i), .A(a_i), .B(b_i), .result_o(prod), .done_o(prod_v),
                        .overflow_o(), .underflow_o(), .invalid_o());
    fp32Adder ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum), .done_o(sum_v),
                   .overflow_o(), .underflow_o(), .invalid_o());
  end else if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT
    logic [2*DATA_WIDTH-1:0] prod_w;  // the full signed product
    intMultiplier #(.W(DATA_WIDTH)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i), .A(a_i), .B(b_i), .result_o(prod_w),
        .done_o(prod_v));
    assign prod = {{(ACC_W - 2 * DATA_WIDTH){prod_w[2*DATA_WIDTH-1]}}, prod_w};  // sign-extended to the accumulator
    intAdder #(.W(ACC_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum),
        .done_o(sum_v));
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i), .A(a_i), .B(b_i),
        .result_o(prod), .done_o(prod_v), .overflow_o(), .underflow_o(), .invalid_o());
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod),
        .result_o(sum), .done_o(sum_v), .overflow_o(), .underflow_o(), .invalid_o());
  end

  // Where each add in flight writes back, and whether it is still in flight.
  logic [BW-1:0] bank_dly[ADD_LAT];
  logic [SW-1:0] slot_dly[ADD_LAT];
  logic          v_dly   [ADD_LAT];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < ADD_LAT; i++) begin
        bank_dly[i] <= '0;
        slot_dly[i] <= '0;
        v_dly[i]    <= 1'b0;
      end
    end else begin
      bank_dly[0] <= cur;
      slot_dly[0] <= slot;
      v_dly[0]    <= prod_v;
      for (int i = 1; i < ADD_LAT; i++) begin
        bank_dly[i] <= bank_dly[i-1];
        slot_dly[i] <= slot_dly[i-1];
        v_dly[i]    <= v_dly[i-1];
      end
    end
  end

  logic [BANKS-1:0] pending;  // an add into the bank is issuing or in flight
  always_comb begin
    pending = '0;
    if (prod_v) pending[cur] = 1'b1;
    for (int i = 0; i < ADD_LAT; i++) if (v_dly[i]) pending[bank_dly[i]] = 1'b1;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      cur     <= '0;
      slot    <= '0;
      n_prod  <= '0;
      taken   <= '0;
      final_o <= '0;
      for (int b = 0; b < BANKS; b++) for (int u = 0; u < U; u++) acc[b][u] <= '0;
    end else begin
      if (sum_v) acc[bank_dly[ADD_LAT-1]][slot_dly[ADD_LAT-1]] <= sum;
      if (prod_v) begin
        if (n_prod == CW'(K - 1) && prod_more) begin
          // The set goes on: same bank, and the slot keeps turning so each slot's add has written back before its next.
          slot   <= (slot == SW'(U - 1)) ? '0 : slot + 1'b1;
          n_prod <= '0;
        end else if (n_prod == CW'(K - 1)) begin
          taken[cur] <= 1'b1;
          cur        <= (cur == BW'(BANKS - 1)) ? '0 : cur + 1'b1;
          slot       <= '0;
          n_prod     <= '0;
        end else begin
          slot   <= (slot == SW'(U - 1)) ? '0 : slot + 1'b1;
          n_prod <= n_prod + 1'b1;
        end
      end
      for (int b = 0; b < BANKS; b++) if (taken[b] && !pending[b]) final_o[b] <= 1'b1;
      if (release_i) begin
        taken[rd_bank_i]   <= 1'b0;
        final_o[rd_bank_i] <= 1'b0;
      end
    end
  end

  always_comb for (int u = 0; u < U; u++) partial_o[u] = acc[rd_bank_i][u];

`ifndef SYNTHESIS
  a_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_v && n_prod == '0) |-> !taken[cur])
    else $error("ProcessingElement: a set started in bank %0d before the reader released it", cur);
  a_flags_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) prod_v == v_d[MUL_LAT-1])
    else $error("ProcessingElement: the pass flags are out of step with the multiplier");
  a_fresh_slot0: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_v && n_prod == '0 && prod_fresh) |-> slot == '0)
    else $error("ProcessingElement: a new set started part way through a bank's slots");
  a_release_final: assert property (@(posedge clk_i) disable iff (!rstn_i) release_i |-> final_o[rd_bank_i])
    else $error("ProcessingElement: bank %0d released before its set was final", rd_bank_i);
`endif

endmodule
```

Against the bf16 file, the changes are: `ACC_W` parameter; `partial_o`, `prod`, `sum`, `acc`, `add_a` at `ACC_W`; the
`G_BAD_ACC_W` and `G_INT` branches; the int8 format in two comments. `G_INT` comes before `G_FP` so an int8 build can
never pick `fpMultiplier`. In the float formats `ACC_W == DATA_WIDTH`, so `G_FP32` and `G_FP` connect exactly as before.
With `ADD_LAT = 1`, `S = 2` and `U = min(K, 2)`: a slot read at t is written back at the end of t+1 and read again at t+2,
so one product per cycle still holds.

- [ ] **Step 5: AccumulationUnit**

Replace the three parameter lines

```systemverilog
    parameter EXP_W = 8,  // the build's format: fp32 8/23, bf16 8/7
    parameter MAN_W = 23,
    parameter DATA_WIDTH = 1 + EXP_W + MAN_W,
```

with

```systemverilog
    parameter EXP_W = 8,  // the build's format: fp32 8/23, bf16 8/7, int8 0/7
    parameter MAN_W = 23,
    parameter ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // partials, bias and results: int32 in int8, the format's width in floats
```

Then rename every remaining use: `sed -i 's/DATA_WIDTH/ACC_W/g' src/engine/AccumulationUnit.sv`. After it,
`command grep -n DATA_WIDTH src/engine/AccumulationUnit.sv` prints nothing (the uses were `tile_data_i`, `bias_i`,
`bias_word_o`, `write_data_o`, `rd_bias`, `lvl_d` and the `PASS` delay `dly`).

Replace the format check

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "AccumulationUnit: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end
```

with

```systemverilog
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "AccumulationUnit: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC_W
    $fatal(1, "AccumulationUnit: ACC_W=%0d is not sienna_fmt_pkg::acc_w(%0d, %0d)", ACC_W, EXP_W, MAN_W);
  end
```

In `NODE`'s `ADD` branch, replace

```systemverilog
        end else begin : G_FP
          fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]),
```

with

```systemverilog
        end else if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT
          intAdder #(.W(ACC_W)) adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]), .A(lvl_d[l][2*m]), .B(lvl_d[l][2*m+1]),
                                       .result_o(lvl_d[l+1][m]), .done_o(done_bits[m]));
        end else begin : G_FP
          fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]),
```

`ADD_LAT` already comes from the package, so the tree's latency `LAT = 1 + LEVELS * ADD_LAT` and the `PASS` delays follow
the int8 adder (1 cycle) with no other change.

- [ ] **Step 6: SystolicArray and SystolicMesh**

`src/top/SystolicArray.sv`:
- `parameter int EXP_W       = 8,   // the build's format: fp32 8/23 by default, bf16 8/7` gets `, int8 0/7` at the end of its comment.
- Replace `parameter int DATA_WIDTH  = 1 + EXP_W + MAN_W,  // every word: operands and partial sums` with two lines:

```systemverilog
    parameter int DATA_WIDTH  = 1 + EXP_W + MAN_W,  // operands
    parameter int ACC_W       = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // partial sums: int32 in int8, the format's width in floats
```

- `output logic [U-1:0][DATA_WIDTH-1:0]       read_data_o,    // the pixel's U partial sums, one cycle later` becomes
  `output logic [U-1:0][ACC_W-1:0]            read_data_o,    // the pixel's U partial sums, one cycle later`.
- `logic [U-1:0][DATA_WIDTH-1:0] part[N][N];` becomes `logic [U-1:0][ACC_W-1:0] part[N][N];`.
- In the `ProcessingElement` instance, after `.DATA_WIDTH(DATA_WIDTH),` add `.ACC_W     (ACC_W),`.

Then `command grep -n "DATA_WIDTH" src/top/SystolicArray.sv` shows only the parameter, the two write-data ports, `a_mem`,
`b_mem`, `a_feed`/`b_feed`, `a_w`/`b_n` and the PE's `.DATA_WIDTH`: operands only.

`src/top/SystolicMesh.sv`, each line replaced as shown (old, then new):

```systemverilog
    parameter EXP_W       = 8,   // the build's format: fp32 8/23 by default, bf16 8/7
    parameter EXP_W       = 8,   // the build's format: fp32 8/23 by default, bf16 8/7, int8 0/7

    parameter DATA_WIDTH  = 1 + EXP_W + MAN_W,  // every word: operands, sums, bias, results
    parameter DATA_WIDTH  = 1 + EXP_W + MAN_W,  // operands: host writes, staging banks, weight cache
    parameter ACC_W       = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // sums, bias and results: int32 in int8, the format's width in floats

    input logic [MATRIX_SIZE-1:0][DATA_WIDTH-1:0] bias_i,
    input logic [MATRIX_SIZE-1:0][ACC_W-1:0]      bias_i,

    output logic [DATA_WIDTH-1:0] read_data_o,
    output logic [     ACC_W-1:0] read_data_o,

    output logic [WIDE_READ-1:0][DATA_WIDTH-1:0] wide_read_data_o,
    output logic [WIDE_READ-1:0][     ACC_W-1:0] wide_read_data_o,

  logic [DATA_WIDTH-1:0] bias_q[BIAS_Q][MATRIX_SIZE];
  logic [ACC_W-1:0] bias_q[BIAS_Q][MATRIX_SIZE];

  logic [MATRIX_SIZE-1:0][DATA_WIDTH-1:0] red_bias;  // bias of the set the reducers are starting
  logic [MATRIX_SIZE-1:0][ACC_W-1:0] red_bias;  // bias of the set the reducers are starting

  logic [NUM_TILES-1:0][DATA_WIDTH-1:0] sram_data_agg;
  logic [NUM_TILES-1:0][     ACC_W-1:0] sram_data_agg;

      .DATA_WIDTH(DATA_WIDTH),        (in MeshOutputSram #(...))
      .DATA_WIDTH(ACC_W),

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][RPU-1:0][DATA_WIDTH-1:0] t_data;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][RPU-1:0][ACC_W-1:0] t_data;

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][DATA_WIDTH-1:0] t_bias;  // the bias of the pixel being summed
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][ACC_W-1:0] t_bias;  // the bias of the pixel being summed

            .DATA_WIDTH(DATA_WIDTH),        (in AccumulationUnit #(...))
            .ACC_W(ACC_W),

            logic [U-1:0][DATA_WIDTH-1:0] rd;
            logic [U-1:0][ACC_W-1:0] rd;
```

and in the `SystolicArray #(...)` instance, after `.DATA_WIDTH(DATA_WIDTH),` add `.ACC_W(ACC_W),`.

Then `command grep -n "DATA_WIDTH" src/top/SystolicMesh.sv` must print exactly 11 lines: the parameter, the
`north_write_data_i` and `west_write_data_i` ports, `mem_A`, `mem_B`, `wcache`, the `DATA_WIDTH != 1 + EXP_W + MAN_W`
check, `b_word`, `load_data_A, load_data_B`, `MeshOutputSram`'s `.DATA_WIDTH(ACC_W)` and the `SystolicArray`'s
`.DATA_WIDTH(DATA_WIDTH)`. Everything else that carries a
word is `ACC_W`: `command grep -c "ACC_W" src/top/SystolicMesh.sv` counts the parameter, three ports, `bias_q`,
`red_bias`, `sram_data_agg`, `t_data`, `t_bias`, `rd` and three instance connections (13).

- [ ] **Step 7: SIENNA file lists (edit now, commit in Task 15)**

In `/proj/work/spramanik/SIENNA_int8/Makefile` `SM_LIB_FILES` (tab-indented), after `	Multipliers/FP/src/fpMultiplier.sv \`
insert `	Multipliers/Int/src/intMultiplier.sv \`, and replace the list's last line `	Adders/FP/src/fpAdder.sv` with
two lines, `	Adders/FP/src/fpAdder.sv \` and `	Adders/Int/src/intAdder.sv`.

In `/proj/work/spramanik/SIENNA_int8/synth/sienna_rtl.f`, insert
`SystolicMesh/ArithmeticLibrary/Multipliers/Int/src/intMultiplier.sv` after the `fpMultiplier.sv` line and
`SystolicMesh/ArithmeticLibrary/Adders/Int/src/intAdder.sv` after the `fpAdder.sv` line.

Without them, every SIENNA build (fp32 and bf16 too) fails to link the PE and reducer, for the reason in Step 3.

- [ ] **Step 8: Run the checks**

```bash
$J/snap_launch_tree.sh m13_pe 32 1 $J/cmds/int8_mesh.sh pe
$J/snap_launch_tree.sh m13_lint 32 1 $J/cmds/int8_mesh.sh lintall
TREE=/proj/work/spramanik/SIENNA $J/snap_launch_tree.sh m13_lintref 32 1 $J/cmds/int8_mesh.sh lintref   # the bf16 tree, read-only
$J/snap_launch_tree.sh m13_slint 32 1 $J/cmds/int8_mesh.sh slint
for F in fp32 bf16; do for C in 1 0; do
  $J/snap_launch_tree.sh m13_${F}_ck$C 64 12 $J/cmds/mesh_notrace.sh reg 16 $F $C
done; done
```

When they finish:

```bash
for t in bf16 fp32; do
  command diff <(command grep -ho '^%Warning-[A-Z]*' $J/runs/m13_lint/results/int8/mesh_lint_$t.txt | sort | uniq -c) \
               <(command grep -ho '^%Warning-[A-Z]*' $J/runs/m13_lintref/results/int8/mesh_lint_$t.txt | sort | uniq -c) \
    && echo "$t lint warnings unchanged"
done
ref() { case $1 in fp32_ck1) echo g3x_N16_fp32_ck1;; fp32_ck0) echo g3x_N16_fp32_ck0;; bf16_ck1) echo g3x_N16_bf16_ck1;; bf16_ck0) echo g3_N16_bf16_ck0;; esac; }
for k in fp32_ck1 fp32_ck0 bf16_ck1 bf16_ck0; do
  command diff <($J/cycles.sh $J/runs/m13_$k/mesh_results/readiness) <($J/cycles.sh $J/runs/$(ref $k)/mesh_results/readiness) > /dev/null \
    && echo "$k: cycles identical" || echo "$k: CYCLES DIFFER"
done
```

(`cycles.sh` and `diff` only read logs; they are not builds, so the login node is fine for them.)

Expected:
- `m13_pe`: `TB_PE_int8: 40 sets, 80 slots checked, 0 mismatches` and `RESULT: PASSED`. Set 0 (4 products of
  -128 x -128 per slot, 65536) passing proves the slots are 32 bits wide, not 16.
- `m13_lint`: `int8`, `int8_ck0`, `bf16`, `fp32` end in `CLEAN`; `bad_5_10`, `bad_0_8`, `bad_0_15` end in `REJECTED`
  with `unsupported format`; `bad_acc_int8` and `bad_acc_bf16` end in `REJECTED` with `ACC_W=16 is not` and
  `ACC_W=32 is not`. The int8 lint has no `WIDTH` warning on a line this task changed (read the `mesh_lint_int8*.txt` files).
- `bf16 lint warnings unchanged` and `fp32 lint warnings unchanged`: the float netlists gained no warning class or count.
- `m13_slint`: `SIENNA make lint: exit=0, 0 errors` and `sienna_rtl.f lint: exit=0, 0 errors`.
- Each `m13_{fp32,bf16}_ck{1,0}` regression 72/72 (18 tests x 4 tiles) and `cycles identical` for all four. The TB is
  still the bf16 branch's here, so these runs compare the RTL alone.

If a float regression's cycles differ, the change leaked into a float branch: diff the PE and reducer generate blocks
first; do not continue to Task 14.

- [ ] **Step 9: Commit (SystolicMesh) and push**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh
git add ArithmeticLibrary && git commit -m "Bump ArithmeticLibrary: int8 format, intMultiplier, intAdder"
git add src/engine/ProcessingElement.sv src/engine/AccumulationUnit.sv src/top/SystolicArray.sv src/top/SystolicMesh.sv \
  && git commit -m "int8 through the mesh: int8 operands on intMultiplier, int32 sums on intAdder; ACC_W for partials, bias, result memory and reads"
git add Makefile && git commit -m "Makefile: the integer units"
git add testbenches/TB_PE_int8.sv && git commit -m "TB_PE_int8: int8 PE slots against exact int32 sums (extremes, two-pass sets, back to back)"
git push origin int8
git status -sb   # no 'ahead'
```

The RTL commit changes four files because `ACC_W` crosses every port between them; it cannot build in parts. The SIENNA
`Makefile` and `sienna_rtl.f` edits stay uncommitted until Task 15 bumps SIENNA's SystolicMesh pointer.

### Task 14: Bit-exact int8 mesh golden, int8 in the mesh regression

**Files:**
- Modify: `mesh_model.py` (`matmul_int`), `stim_format.py` (int8), `test_stim_format.py` (int8 tests)
- Modify: `matmul_tests.py`, `conv_tests.py` (int8 data through `stim_format.rand`, two explicit int8 generators, the
  per-set int8 bias)
- Modify: `testbenches/TB_SystolicMesh.sv` (`ACC_W` result words, per-set bias files), `regression.py` (`--format int8`,
  report notes)
- Create: `$J/cmds/int8_gen_all.py`; `/proj/work/spramanik/SIENNA_int8/.claude/scratch/mesh_old/` (the bf16 branch's generators, for the byte-identity check; not committed)

**Interfaces:**
- Consumes: Task 13's mesh (`ACC_W` ports) and `int8_mesh.sh unit | same`.
- Produces:
  - `mesh_model.matmul_int(passes, N, bias=None) -> np.ndarray[N, N] of int64`: `passes` is a list of `(A, B)` N x N arrays
    of int8 **values** (-128..127, not bit patterns), summed in order as one set; `bias` is N int32 values or None. The
    int64 sum is wrapped to int32 once at the end, which equals the RTL's per-add wrap because addition is associative
    mod 2^32. Returns signed int32 values. Raises `ValueError` on out-of-range operands or bias. D-6's SIENNA golden uses it.
  - `stim_format`: `configure("int8", tile, collapse_k)`, `INT_FORMATS = {"int8": (8, 32)}`, `is_int()`, `digits()`
    (operand hex digits: 2 in int8), `result_digits()` (8 in int8, `digits()` in floats), `rand(lo, hi, shape)`,
    `to_int(x)`, `int8_bias(N, suffix)` (the set's int32 bias: set `_1` within 4,096 below INT32_MAX, the others
    uniform int32), `write_set(A, B, stim_dir, suffix="", bias=None)` (int8: `matrixA/B` words of 2 hex digits,
    `matrixC` of 8, C from `matmul_int` with the bias, and the bias in `matrixBias<suffix>.mem`, N words of 8 hex
    digits; returns C as int64 values), `check_widths` (per file kind; a bias file outside int8 is an error).
  - `TB_SystolicMesh` drives `bias_valid_i` and `bias_i` with each start from the set's `matrixBias<suffix>.mem` when it
    exists (int8 only), so the `ACC_W` bias input, the bias queue and the int32 wrap are tested at G3.
  - Mesh `regression.py --format fp32|bf16|int8`.

- [ ] **Step 1: Write the failing unit tests**

Replace `test_stim_format.py` with:

```python
#!/usr/bin/env python3
"""stim_format's expected C is the bit-exact mesh result in every format: fp32 from mesh_model.matmul, int8 from matmul_int."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matmul_tests as mt  # noqa: E402
import mesh_model as mm  # noqa: E402
import stim_format as sf  # noqa: E402


def _words(d, name):
    return open(os.path.join(d, name)).read().split()


def _wrap32(v):
    return ((v + 2**31) % 2**32) - 2**31


def test_fp32_expected_is_the_mesh_result():
    # N=64 padding set 9001: output 1085 is a near-zero cancellation; the RTL returns 0x38050000 (3.1710e-5), the
    # correctly rounded sum is 0x3803ab66 (3.1392e-5), 1.01% away, so a golden from float64 fails correct hardware
    mt._seed(9001)
    A = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
    B = np.random.uniform(-1, 1, (64, 64)).astype(np.float32)
    sf.configure("fp32", 2, 1)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(A, B, d)
        C = [int(x, 16) for x in open(os.path.join(d, "matrixC.mem")).read().split()]
        A_words = open(os.path.join(d, "matrixA.mem")).read().split()
    assert C[1085] == 0x38050000, hex(C[1085])
    assert A_words[0] == f"{int(A.view(np.uint32).flat[0]):08x}"  # operands still written as their float32 bits


def test_int8_words_and_expected():
    # -1 x 127 over N=8 is -1016 (fffffc08); -128 x 127 is -130048 (fffe0400), beyond int16, so the sums must be int32
    N = 8
    A = np.full((N, N), -128, dtype=np.float32)
    A[0, :] = -1
    B = np.full((N, N), 127, dtype=np.float32)
    sf.configure("int8", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        C = sf.write_set(A, B, d)
        a, b, c = _words(d, "matrixA.mem"), _words(d, "matrixB.mem"), _words(d, "matrixC.mem")
    assert len(a) == len(b) == len(c) == N * N
    assert a[0] == "ff" and a[N] == "80" and b[0] == "7f", (a[0], a[N], b[0])
    assert {len(w) for w in a + b} == {2} and {len(w) for w in c} == {8}
    assert c[0] == "fffffc08" and c[N] == "fffe0400", (c[0], c[N])
    assert int(C[0, 0]) == -1016 and int(C[1, 0]) == -130048


def test_int8_largest_sum():
    # -128 x -128 over K=64 is 64 x 16384 = 2^20, the largest sum one mesh set can produce
    N = 64
    A = np.full((N, N), -128, dtype=np.float32)
    sf.configure("int8", 2, 0)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(A, A, d)
        c = _words(d, "matrixC.mem")
    assert set(c) == {"00100000"}, sorted(set(c))[:3]


def test_matmul_int_wraps_like_the_adders():
    # the RTL wraps every add mod 2^32; wrapping only the final sum is the same because integer addition is associative
    N = 8
    rng = np.random.default_rng(5)
    passes = [(rng.integers(-128, 128, (N, N)), rng.integers(-128, 128, (N, N))) for _ in range(3)]
    bias = rng.integers(-2**31, 2**31, N)
    C = mm.matmul_int(passes, N, bias)
    for i in range(N):
        for j in range(N):
            s = 0
            for A, B in passes:
                for k in range(N):
                    s = _wrap32(s + int(A[i, k]) * int(B[k, j]))
            assert C[i, j] == _wrap32(s + int(bias[j])), (i, j, C[i, j])
    E = mm.matmul_int([(np.eye(N, dtype=np.int64), np.ones((N, N), dtype=np.int64))], N, np.full(N, 2**31 - 1))
    assert (E == -2**31).all()  # 2^31 - 1 + 1 wraps to INT32_MIN


def test_int8_bias_file_and_wrap():
    # set _1's bias sits within 4096 below INT32_MAX, so every positive sum of the set wraps negative, as the int32 adders do
    N = 8
    b = sf.int8_bias(N, "_1")
    assert (b <= 2**31 - 1).all() and (b > 2**31 - 1 - 4096).all()
    A = np.full((N, N), 127, dtype=np.float32)
    sf.configure("int8", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        C = sf.write_set(A, A, d, "_1", bias=b)
        w = _words(d, "matrixBias_1.mem")
        sf.check_widths(d)
    assert len(w) == N and {len(x) for x in w} == {8}
    assert (C == _wrap32(N * 127 * 127 + b[None, :])).all() and (C < 0).all()
    sf.configure("fp32", 4, 1)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "matrixBias_0.mem"), "w").write("00000001\n")
        try:
            sf.check_widths(d)
        except ValueError:
            return
    raise AssertionError("check_widths took a bias file in fp32")


def test_int8_rejects_what_is_not_int8():
    sf.configure("int8", 4, 1)
    for bad in (0.5, 128.0, -129.0):
        A = np.zeros((4, 4), dtype=np.float32)
        A[1, 2] = bad
        with tempfile.TemporaryDirectory() as d:
            try:
                sf.write_set(A, np.zeros((4, 4), dtype=np.float32), d)
            except ValueError:
                continue
        raise AssertionError(f"write_set took the int8 operand {bad}")
    try:
        mm.matmul_int([(np.full((4, 4), 200), np.zeros((4, 4)))], 4)  # a bit pattern, not a value
    except ValueError:
        return
    raise AssertionError("matmul_int took an operand of 200")


def test_int8_check_widths():
    sf.configure("int8", 4, 1)
    ones = np.ones((4, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        sf.write_set(ones, ones, d, "_0")
        sf.check_widths(d)  # 2-digit operands, 8-digit results
        for name, word in (("matrixA_0.mem", "3f800000"), ("matrixC_0.mem", "04")):
            with open(os.path.join(d, name), "w") as fh:
                fh.write(word + "\n")
            try:
                sf.check_widths(d)
            except ValueError:
                sf.write_set(ones, ones, d, "_0")
                continue
            raise AssertionError(f"check_widths took '{word}' in {name}")


def test_rand():
    # int8 draws the whole range (4096 draws: missing -128 or 127 has probability about 2e-7); floats draw exactly as before
    sf.configure("int8", 4, 1)
    np.random.seed(1)
    x = sf.rand(-1, 1, (64, 64))
    assert (x == np.round(x)).all() and x.min() == -128 and x.max() == 127
    sf.configure("fp32", 4, 1)
    np.random.seed(1)
    y = sf.rand(-1, 1, (8, 8))
    np.random.seed(1)
    assert (y == np.random.uniform(-1, 1, (8, 8)).astype(np.float32)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
```

- [ ] **Step 2: Run it to see it fail**

```bash
$J/snap_launch_tree.sh m14_unit 32 1 $J/cmds/int8_mesh.sh unit
```

Expected: `PASS test_fp32_expected_is_the_mesh_result`, then a traceback ending in `KeyError: 'int8'` from
`stim_format.to_bits` (`fpu.FORMATS` has no int8), exit 1.

- [ ] **Step 3: The integer mesh model**

Append to `mesh_model.py`:

```python


def matmul_int(passes, N, bias=None):
    """The int8 mesh: int8 x int8 products summed in int32 two's complement, wrapping, over a set's passes, plus the bias.
    Addition is associative mod 2^32, so the PE slots and the reduce tree's order do not change the result.
    Operands are int8 values (not bit patterns), the bias N int32 values; returns int32 values as int64."""
    acc = np.zeros((N, N), dtype=np.int64)
    for A, B in passes:
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if A.shape != (N, N) or B.shape != (N, N):
            raise ValueError(f"mesh sets are N x N; got {A.shape} and {B.shape}")
        if min(A.min(), B.min()) < -128 or max(A.max(), B.max()) > 127:
            raise ValueError("int8 operands out of range: pass values in [-128, 127], not bit patterns")
        acc += A @ B
    if bias is not None:
        b = np.asarray(bias, dtype=np.int64)
        if b.shape != (N,) or b.min() < -2**31 or b.max() >= 2**31:
            raise ValueError(f"bias must be {N} int32 values")
        acc += b[None, :]
    return ((acc + 2**31) % 2**32) - 2**31
```

int64 cannot overflow here: a pass adds at most 64 x 16384 = 2^20 per element.

- [ ] **Step 4: int8 in the stimulus format**

Replace `stim_format.py` with:

```python
#!/usr/bin/env python3
"""The mesh stimulus's number format: operand words in the format and the bit-exact expected result from mesh_model in every
format (fp32 included), for the TB's exact compare. int8 operands are 2 hex digits, its int32 results 8."""
import os
import struct

import numpy as np

import mesh_model
from mesh_model import fpu

FORMAT, TILE, COLLAPSE_K = "fp32", 4, 1
INT_FORMATS = {"int8": (8, 32)}  # operand bits, accumulator bits (sienna_fmt_pkg::acc_w)


def configure(fmt: str, tile: int, collapse_k: int) -> None:
    global FORMAT, TILE, COLLAPSE_K
    if fmt not in fpu.FORMATS and fmt not in INT_FORMATS:
        raise ValueError(f"unknown mesh format {fmt}")
    FORMAT, TILE, COLLAPSE_K = fmt, tile, collapse_k


def is_int() -> bool:
    return FORMAT in INT_FORMATS


def digits() -> int:
    """Hex digits of an operand word."""
    return INT_FORMATS[FORMAT][0] // 4 if is_int() else (fpu.FORMATS[FORMAT].w + 3) // 4


def result_digits() -> int:
    """Hex digits of a result word: the accumulator's width, which is the format's own in the float formats."""
    return INT_FORMATS[FORMAT][1] // 4 if is_int() else digits()


def rand(lo, hi, shape) -> np.ndarray:
    """Random operands: float formats draw uniform [lo, hi) float32 exactly as before; int8 draws the whole int8 range."""
    if is_int():
        return np.random.randint(-128, 128, shape).astype(np.float32)
    return np.random.uniform(lo, hi, shape).astype(np.float32)


def int8_bias(N, suffix) -> np.ndarray:
    """The int8 mesh TB's int32 bias for set <suffix>: set 1 within 4096 below INT32_MAX, so its positive sums wrap; others uniform."""
    s = int(suffix.lstrip("_") or 0)
    rng = np.random.default_rng(7000 + s)  # its own generator: the operand draws stay as they were
    if s == 1:
        return (2**31 - 1) - rng.integers(0, 4096, N)
    return rng.integers(-2**31, 2**31, N)


def to_int(x) -> np.ndarray:
    """int8 operand values as int64; anything that is not an integer in [-128, 127] is an error, never rounded or clipped."""
    v = np.asarray(x, dtype=np.float64)
    if not (np.all(v == np.round(v)) and v.min() >= -128 and v.max() <= 127):
        raise ValueError(f"int8 operands must be integers in [-128, 127]; got values in [{v.min()}, {v.max()}]")
    return v.astype(np.int64)


def to_bits(x) -> np.ndarray:
    """float32 values, rounded to the format (fp32 nearest, then nearest at the format's width); subnormals flush to zero."""
    f = fpu.FORMATS[FORMAT]
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    b = np.vectorize(lambda v: fpu.from_fp32(int(v), f.m))(u) if f.m != 23 else u
    return np.where(((b >> f.m) & f.emax) == 0, b & (1 << (f.w - 1)), b).astype(np.int64)


def to_float(bits) -> np.ndarray:
    f = fpu.FORMATS[FORMAT]
    return (np.asarray(bits, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def _write_words(path, bits, d=None) -> None:
    d = d or digits()
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _f2h(v) -> str:
    return "".join(f"{b:02x}" for b in struct.pack(">f", float(v)))


def write_set(A, B, stim_dir, suffix="", bias=None):
    """Write matrixA/B/C<suffix>.mem for one set; returns C as floats (int8: as int64 values, with bias, also in matrixBias<suffix>.mem)."""
    N = B.shape[1]
    # A set shorter than N x N is written with its zero rows: the staging bank is not reset, so the test must not rely on it.
    assert A.shape[1] == N and A.shape[0] <= N and B.shape[0] <= N, f"unsupported set shapes {A.shape} and {B.shape}"
    A = np.vstack([A, np.zeros((N - A.shape[0], N), dtype=np.float32)]).astype(np.float32)
    B = np.vstack([B, np.zeros((N - B.shape[0], N), dtype=np.float32)]).astype(np.float32)
    if is_int():
        Ai, Bi = to_int(A), to_int(B)
        Ci = mesh_model.matmul_int([(Ai, Bi)], N, bias)
        _write_words(os.path.join(stim_dir, f"matrixA{suffix}.mem"), Ai & 0xFF)
        _write_words(os.path.join(stim_dir, f"matrixB{suffix}.mem"), Bi & 0xFF)
        _write_words(os.path.join(stim_dir, f"matrixC{suffix}.mem"), Ci & 0xFFFFFFFF, result_digits())
        if bias is not None:
            _write_words(os.path.join(stim_dir, f"matrixBias{suffix}.mem"), np.asarray(bias, dtype=np.int64) & 0xFFFFFFFF,
                         result_digits())
        return Ci
    if bias is not None:
        raise ValueError("a mesh bias file is written in int8 only")
    Ab, Bb = to_bits(A), to_bits(B)
    Cb = mesh_model.matmul(fpu.FORMATS[FORMAT], [(Ab, Bb)], N, TILE, COLLAPSE_K)
    if FORMAT == "fp32":  # operands as their float32 words, exactly as before; C is the mesh's own bit-exact result
        for name, M in (("A", A), ("B", B)):
            with open(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), "w") as fh:
                for v in np.asarray(M).flatten():
                    fh.write(_f2h(v) + "\n")
        _write_words(os.path.join(stim_dir, f"matrixC{suffix}.mem"), Cb)
        return to_float(Cb)
    for name, M in (("A", Ab), ("B", Bb), ("C", Cb)):
        _write_words(os.path.join(stim_dir, f"matrix{name}{suffix}.mem"), M)
    return to_float(Cb)


def check_widths(stim_dir) -> None:
    """Every matrix word must be its kind's width in this format: a stale file from another format would be truncated silently."""
    for fn in sorted(os.listdir(stim_dir)):
        if fn.startswith("matrixBias") and not is_int():
            raise ValueError(f"{fn}: a bias file is int8 only; a stale one would bias this {FORMAT} run")
        if fn.startswith("matrix") and fn.endswith(".mem"):
            d = result_digits() if fn.startswith(("matrixC", "matrixBias")) else digits()
            for i, ln in enumerate(open(os.path.join(stim_dir, fn))):
                if ln.strip() and len(ln.strip()) != d:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {d} hex digits ({FORMAT})")
```

The float paths are the bf16 branch's code unchanged (`to_bits`, `to_float`, the fp32 and bf16 branches of `write_set`;
`_write_words` without its new argument writes the same bytes; `bias` is `None` in the float formats). Step 9 checks
that with `diff`.

- [ ] **Step 5: Run the unit tests to see them pass**

```bash
$J/snap_launch_tree.sh m14_unit 32 1 $J/cmds/int8_mesh.sh unit
```

Expected: eight `PASS test_...` lines (`test_fp32_expected_is_the_mesh_result`, `test_int8_words_and_expected`,
`test_int8_largest_sum`, `test_matmul_int_wraps_like_the_adders`, `test_int8_bias_file_and_wrap`,
`test_int8_rejects_what_is_not_int8`, `test_int8_check_widths`, `test_rand`), exit 0.

- [ ] **Step 6: int8 data from the generators**

Route every random draw through `stim_format.rand`, which draws exactly `np.random.uniform(lo, hi, shape)` in the float
formats, so fp32 and bf16 stimulus stays byte-identical:

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh
sed -i 's/np\.random\.uniform(/stim_format.rand(/g' matmul_tests.py conv_tests.py
command grep -c "np.random.uniform(" matmul_tests.py conv_tests.py   # expect 0 and 0
command grep -c "stim_format.rand(" matmul_tests.py conv_tests.py    # expect 14 and 15
```

The `.astype(np.float32)` that follows each call stays; it is a no-op on `rand`'s float32 result. Both files already
`import stim_format`.

Every int8 set also gets its own int32 bias, so the mesh's `ACC_W` bias input, its bias queue and the int32 wrap are
tested here rather than first at G4 (set 1 of every test sits within 4,096 of INT32_MAX). In `matmul_tests.py`,
`_write_set` becomes

```python
def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "") -> None:
    """Compute C = A @ B in the stimulus format and write its .mem files for one set (int8: with the set's bias)."""
    bias = stim_format.int8_bias(B.shape[1], suffix) if stim_format.is_int() else None
    stim_format.write_set(A, B, stim_dir, suffix, bias)
```

and in `conv_tests.py`

```python
def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "") -> np.ndarray:
    bias = stim_format.int8_bias(B.shape[1], suffix) if stim_format.is_int() else None
    return stim_format.write_set(A, B, stim_dir, suffix, bias)
```

In the float formats `bias` is `None` and nothing changes.

With that, in int8: `mm_random`, `mm_diagonal`, the padding sets and every conv image and kernel are uniform int8 over
-128..127; `mm_identity` (B = I), `mm_zero_b`, `mm_ones`, `mm_alternating` and the zero, ones and impulse kernels are
integers already; `mm_signed_zero`'s `-0.0` and `0.0` rows become plain zero rows. Two generators need their own int8 data.
In `matmul_tests.py`, make these the first lines of `gen_mm_large_values`'s body:

```python
    if stim_format.is_int():  # int8: the signed extremes
        lo, hi = np.float32(-128), np.float32(127)
        sets = [(np.full((N, N), lo), np.full((N, N), lo)),  # every product +16384: the largest sum, N * 2^14
                (np.full((N, N), lo), np.full((N, N), hi))]  # every product -16256: the most negative sum
        for i in range(3):
            _seed(500 + i)
            sets.append((np.random.choice([lo, hi], (N, N)).astype(np.float32),
                         np.random.choice([lo, hi], (N, N)).astype(np.float32)))
        return _write_all(sets, stim_dir)
```

and these the first lines of `gen_mm_small_values`'s body:

```python
    if stim_format.is_int():  # int8: products of -1, 0 and 1
        sets = []
        for i in range(3):
            _seed(600 + i)
            sets.append((np.random.randint(-1, 2, (N, N)).astype(np.float32),
                         np.random.randint(-1, 2, (N, N)).astype(np.float32)))
        return _write_all(_pad_to(sets, MATMUL_NUM_SETS), stim_dir)
```

In `MATMUL_TESTS`, the two descriptions become `"Values ±100; int8 -128/127  (accumulator range stress)"` and
`"Values ±1e-6; int8 -1/0/1  (underflow stress)"`. In `matmul_tests.py`'s module docstring, after the catalogue, add:

```
int8 (stim_format.configure("int8", ...)): stim_format.rand draws the whole int8 range, mm_large_values uses
-128 and 127, mm_small_values -1, 0 and 1, and mm_signed_zero's rows are plain zeros.
```

In `conv_tests.py`'s module docstring, after `Supported matrix sizes`, add:

```
int8: stim_format.rand draws images and kernels over the whole int8 range, so conv_large_kern is conv_random's distribution.
```

- [ ] **Step 7: TB and regression**

`testbenches/TB_SystolicMesh.sv`:
- After `localparam DATA_WIDTH = 1 + EXP_W + MAN_W;` add
  `localparam int ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W);  // result words: int32 in int8, DATA_WIDTH in floats`.
- `wire    [DATA_WIDTH-1:0] r_data;` becomes `wire    [     ACC_W-1:0] r_data;`.
- `reg     [DATA_WIDTH-1:0] expected_mem          [0:SRAM_SIZE-1];` becomes `reg     [     ACC_W-1:0] expected_mem          [0:SRAM_SIZE-1];`.
- In the DUT's parameters, after `.DATA_WIDTH (DATA_WIDTH),` add `.ACC_W      (ACC_W),`.
- In `verify_results`, `reg [DATA_WIDTH-1:0] exp_val, actual_val;` becomes `reg [ACC_W-1:0] exp_val, actual_val;`.
- After the `expected_mem` line add

```systemverilog
  reg                              bias_v = 1'b0;  // the set's bias (int8: matrixBias<suffix>.mem), taken by the mesh with the start
  reg [MATRIX_SIZE-1:0][ACC_W-1:0] bias_d = '0;
```

  and in the DUT's ports `.bias_valid_i(1'b0),` and `.bias_i('0),` become `.bias_valid_i(bias_v),` and `.bias_i(bias_d),`.
- After the `set_files` task add

```systemverilog
  // ── Per-set bias: the mesh samples bias_i with the start; matrixBias<suffix>.mem exists only in int8 ────────
  task automatic drive_bias(input int s);
    string f;
    integer fh, res;
    reg [ACC_W-1:0] tmp;
    f = (NUM_TEST_SETS == 1) ? "matrixBias.mem" : $sformatf("matrixBias_%0d.mem", s);
    bias_d = '0;
    bias_v = 1'b0;
    fh = $fopen(f, "r");
    if (fh) begin
      for (int c = 0; c < MATRIX_SIZE; c++) begin
        res = $fscanf(fh, "%h", tmp);
        if (res != 1) begin
          $display("  [Error] %s: fewer than %0d bias words", f, MATRIX_SIZE);
          $finish;
        end
        bias_d[c] = tmp;
      end
      bias_v = 1'b1;
      $fclose(fh);
    end
  endtask
```

- Call it on the line before each `start_mult = 1;`: `drive_bias(set_id);` in `execute_test_set`, `drive_bias(s);` in
  `stream_all_sets`' producer, and in `staging_overrun_test` `drive_bias(j % NUM_TEST_SETS);` in its first loop and
  `drive_bias(MESH_SETS % NUM_TEST_SETS);` before each of its two later starts. `command grep -c "drive_bias(" testbenches/TB_SystolicMesh.sv`
  prints 6 (the task and five calls). The task waits on no clock, so every set's cycle count is unchanged; in fp32 and
  bf16 no bias file exists, and the mesh sees `bias_valid_i = 0` and `bias_i = 0` as before.
- In the banner, after `$display(" Sets to Run:    %0d", NUM_TEST_SETS);` add
  `$display(" Format:         EXP_W=%0d MAN_W=%0d, %0d-bit operands, %0d-bit results", EXP_W, MAN_W, DATA_WIDTH, ACC_W);`.

The operand loaders keep `DATA_WIDTH` (`w_data`, `n_data`, `tmp`), so an int8 `$fscanf("%h")` reads 2-digit words into
8 bits and the checker reads 8-digit words into 32. The compare is already `!==` on every word, for every format.
`regression.py`'s patterns (`localparam\s+int\s+EXP_W\s*=`, same for `MAN_W`) still match exactly one line each; the new
`ACC_W` line does not match them.

`regression.py`:
- `FORMATS = {"fp32": (8, 23), "bf16": (8, 7)}` becomes `FORMATS = {"fp32": (8, 23), "bf16": (8, 7), "int8": (0, 7)}`.
- The `--format` help becomes `"number format of the build; every format compares bit for bit against mesh_model"`.
- In the module docstring's Usage, add `  python regression.py --matrix-size 16 --format int8 --collapse-k 0`.
- In `_report`, replace the three stale notes

```python
    L.append("* Tolerance: RELATIVE <= 1%")
    L.append("* Reference: float64 matmul cast to float32")
    L.append("* Data type: IEEE-754 Float32")
```

  with

```python
    L.append("* Compare: bit-exact against mesh_model (matmul in the float formats, matmul_int in int8)")
    L.append(f"* Format: {FMT}  |  collapse-k {COLLAPSE}")
```

- [ ] **Step 8: Run the mesh regressions**

```bash
$J/snap_launch_tree.sh m14_int8_ck1 32 6 $J/cmds/mesh_notrace.sh reg 16 int8 1
$J/snap_launch_tree.sh m14_int8_ck0 64 12 $J/cmds/mesh_notrace.sh reg 16 int8 0
$J/snap_launch_tree.sh m14_int8_n8 32 2 $J/cmds/mesh_notrace.sh reg 8 int8 1 --group matmul
$J/snap_launch_tree.sh m14_fp32 64 12 $J/cmds/mesh_notrace.sh reg 16 fp32 1
$J/snap_launch_tree.sh m14_bf16 64 12 $J/cmds/mesh_notrace.sh reg 16 bf16 1
```

When they finish:

```bash
command grep -h "Passed\|Failed\|Simulations" $J/runs/m14_*/stdout.log
for k in fp32 bf16; do
  command diff <($J/cycles.sh $J/runs/m14_$k/mesh_results/readiness) <($J/cycles.sh $J/runs/g3x_N16_${k}_ck1/mesh_results/readiness) > /dev/null \
    && echo "$k: cycles identical" || echo "$k: CYCLES DIFFER"
done
$J/cycles.sh $J/runs/m14_int8_ck1/mesh_results/readiness | command grep "^mm_random"
$J/cycles.sh $J/runs/m14_int8_ck0/mesh_results/readiness | command grep "^mm_random"
```

Expected:
- `m14_int8_ck1` and `m14_int8_ck0`: 72/72 (18 tests x 4 tiles), every test `exact 100.0%`, `Verdict : All tests passed`.
  `m14_int8_n8`: 27/27 (9 matmul tests x 3 tiles). The report is `readiness_report_int8_ck{1,0}.log`.
- `fp32: cycles identical`, `bf16: cycles identical` (72/72 each).
- int8 single-set latency (`Cycles to complete`) for `mm_random` at T = 2, 4, 8, 16: **35, 53, 113, 329** at collapse-k 1
  and **24, 43, 106, 329** at collapse-k 0. These are fp32's 59, 77, 137, 353 and 55, 75, 134, 353 (measured on the
  bf16 branch, `g3x_N16_fp32_ck{1,0}`) minus D, where D = 7 (multiplier 8 -> 1 cycles) + 4 (PE adder 5 -> 1) + 5 x L32 -
  1 x L8, and L is the reduce tree's level count `clog2(RP x U + 1)` with U = min(K, 6) in fp32 and min(K, 2) in int8.
  D is derived from the RTL, not measured. The same model gives bf16 = fp32 - 5 and collapse-k 0 minus collapse-k 1 of
  -4, -2, -3, 0 at N=16, which is what the bf16 branch measured. A different int8 latency is a finding: explain it from
  the PE or reducer timing before continuing.

Every int8 set carries its bias file, so a pass also shows the `ACC_W` bias input, the bias queue in streaming and
overrun order, and the int32 wrap (set 1's positive sums wrap negative). If an int8 set fails, compare the first
mismatching word against `mesh_model.matmul_int` on that set's own files (with its `matrixBias`) before touching the
RTL: the integer model has no order to get wrong, so a mismatch is the RTL, the bias path or the stimulus width.

- [ ] **Step 9: fp32 and bf16 stimulus are byte-identical to the bf16 branch's**

`$J/cmds/int8_gen_all.py`:

```python
#!/usr/bin/env python3
"""Writes every mesh matmul and conv test's stimulus into OUT/<test>/; run with the generators' directory as cwd; args: FMT N TILE CK OUT."""
import os
import sys

sys.path.insert(0, os.getcwd())
import stim_format  # noqa: E402
import matmul_tests  # noqa: E402
import conv_tests  # noqa: E402

fmt, N, T, ck, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
stim_format.configure(fmt, T, ck)
for t in matmul_tests.MATMUL_TESTS + (conv_tests.CONV_TESTS if N >= 16 else []):
    d = os.path.join(out, t["name"])
    os.makedirs(d, exist_ok=True)
    t["gen_fn"](d, N)
print(f"{fmt} N={N} T={T} ck={ck}: {len(os.listdir(out))} tests written to {out}")
```

Copy the bf16 branch's four stimulus files into the int8 tree's scratch area (on the login node; `git archive` only reads),
then run the comparison on the farm:

```bash
cd /proj/work/spramanik/SIENNA_int8 && mkdir -p .claude/scratch/mesh_old
git -C SystolicMesh archive origin/bf16 matmul_tests.py conv_tests.py stim_format.py mesh_model.py | tar -x -C .claude/scratch/mesh_old
$J/snap_launch_tree.sh m14_same 32 2 $J/cmds/int8_mesh.sh same
```

Expected: eight lines `IDENTICAL fp32 N=16 ck=1` ... `IDENTICAL bf16 N=32 ck=0`, no `DIFFERENT`, exit 0. N=32 covers the
conv generators' general 3x3 layout, N=16 their square layout; both collapse modes cover `mesh_model`'s two slot orders.

- [ ] **Step 10: Commit (SystolicMesh) and push**

```bash
cd /proj/work/spramanik/SIENNA_int8/SystolicMesh
git add testbenches/TB_SystolicMesh.sv && git commit -m "TB_SystolicMesh: result words ACC_W wide (int32 in int8), per-set bias files with each start"
git add mesh_model.py && git commit -m "mesh_model: matmul_int, the int8 mesh in int32 two's complement"
git add stim_format.py test_stim_format.py \
  && git commit -m "Mesh stimulus in int8: 2-digit operands, 8-digit int32 expected results and biases from matmul_int; unit tests"
git add matmul_tests.py conv_tests.py \
  && git commit -m "Mesh generators: int8 data (full range, identity, ones, alternating, -128/127 extremes, zero rows) and per-set int32 biases; float draws unchanged"
git add regression.py && git commit -m "Mesh regression: --format int8; the report names the format and the bit-exact compare"
git push origin int8
git status -sb   # no 'ahead'
```

`.claude/scratch/mesh_old/` is not committed.

### Task 15: Gate G3, SystolicMesh sweeps

**Files:**
- Create: `$J/g3i_summary.py`
- Create (SIENNA, generated, not committed): `testbenches/results/int8/mesh_gate.log`
- Create (SIENNA): `.claude/skills/sienna-report/history/<date>_mesh_gate_int8.txt` (verbatim copy of the gate report)
- Modify (SIENNA): `SystolicMesh` pointer, `Makefile`, `synth/sienna_rtl.f` (from Task 13 Step 7)

**Interfaces:**
- Consumes: Tasks 13-14 pushed; the bf16 branch's G3 runs in `$J/runs` (Task 0's record says which finished and which
  points did not build);
  `$J/cycles.sh`; `$J/cmds/mesh_notrace.sh`, `mesh_notrace_O0.sh`, `mesh_randinit.sh` (all call `cmd_mesh_fmt.sh reg`,
  which needs no file list and so works on both trees).
- Produces: the G3 report; SIENNA's pointer at the Task 14 SystolicMesh tip.

`$J/g3_summary.py` is not reused: it is tied to the bf16 gate's run names and compares only fp32 at N=16 collapse-k 1
against the pre-work baseline. `g3i_summary.py` compares every fp32 and bf16 test at every rerun point against the bf16
branch's runs, and every int8 test against fp32 on the same tree.

What runs (a plan ruling): int8 at N = 8..64, every tile size, both collapse modes, with `--tiles` per job, attempted
even where fp32 did not build (its PEs are much smaller). fp32 and bf16 rerun a subset, since their paths are separate
generate branches from int8's and full N = 64 collapse-k 0 reruns would be about 20 jobs of up to 48 h: collapse-k 1
at every N and T, collapse-k 0 at N <= 32 and at N = 64 with T >= 16. A rerun test whose bf16-branch run never finished
is reported "passed, no cycle reference".

- [ ] **Step 1: Preconditions**

```bash
cd /proj/work/spramanik/SIENNA_int8
for r in . SystolicMesh SystolicMesh/ArithmeticLibrary GPNAE; do echo "== $r"; git -C $r status -sb | head -3; done
command du -sh $J/runs $J/snaps
```

Every repo shows its `int8` branch with no `ahead` and no modified tracked file, except SIENNA's uncommitted `Makefile`
and `synth/sienna_rtl.f` (Task 13 Step 7); the only untracked SIENNA path is `.claude/scratch/`. Both AriL checkouts
must be on the same commit, and the SystolicMesh and GPNAE pointers must name it:

```bash
a=$(git -C SystolicMesh/ArithmeticLibrary rev-parse HEAD); b=$(git -C GPNAE/ArithmeticLibrary rev-parse HEAD)
[ "$a" = "$b" ] && echo "SAME-ARIL $a" || echo "DIFFERENT-ARIL $a $b"
git -C SystolicMesh ls-tree HEAD ArithmeticLibrary; git -C GPNAE ls-tree HEAD ArithmeticLibrary   # both name $a
``` The snapshots taken
below are of this state. The sweep takes about 100
snapshots of about 25 MB each (2.5 GB) plus small run directories; if `/proj/work` looks near its quota, stop and ask
Soham before anything is deleted (a full quota truncates Verilator output silently).

Set `NOT_BUILDABLE` in `g3i_summary.py` from Task 0's record of the points the bf16 branch could not build. At the time
of writing they are collapse-k 0 at N = 64: fp32 T = 2 and T = 4, bf16 T = 2 (beyond the 256 GB node). All three lie
inside the fp32 / bf16 subset's skipped points (N = 64 collapse-k 0, T < 16), so the launch below does not need them;
the summary uses them to say why.

- [ ] **Step 2: Write the summary script**

`$J/g3i_summary.py`:

```python
#!/usr/bin/env python3
"""Builds the int8 G3 mesh gate report from the g3i_* farm runs and prints it: tests passed per N, format and collapse-k;
fp32 and bf16 cycles test by test against the bf16 branch's G3 runs; int8 cycles against fp32's with the difference the
unit latencies predict; random power-up; points not run and why; exit, wall time and peak memory per run."""
import glob
import os
import re
import subprocess
from collections import defaultdict

J = "/proj/work/spramanik/sienna_jobs"
R = f"{J}/runs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUN = re.compile(r"^(g3[a-z]?)_N(\d+)b?_(fp32|bf16|int8)_ck([01])(?:_T(\d+))?(?:_(matmul|conv))?$")
RI = re.compile(r"^g3i_ri_(fp32|bf16|int8)_ck([01])$")
REF = ("g3", "g3m", "g3n", "g3x", "g3y", "g3z")  # prefixes of the bf16 branch's G3 runs; this gate's are g3i
OOM = re.compile(r"Killed signal terminated program|Cannot allocate memory|virtual memory exhausted|oom[-_ ]kill|OUT_OF_MEMORY", re.I)
NOT_BUILDABLE = {(64, "fp32", 0, 2), (64, "fp32", 0, 4), (64, "bf16", 0, 2)}  # (N, format, ck, T) beyond 256 GB on the bf16 branch; Task 0's record
FORMATS = ("int8", "fp32", "bf16")
SIZES = (8, 16, 32, 64)


def tiles(N):
    return [2**i for i in range(1, N.bit_length()) if N % 2**i == 0]


def subset_skip(N, F, ck, T):
    """fp32 and bf16 are not rerun at N = 64 collapse-k 0 below T = 16 (the G3 subset ruling); int8 runs everywhere."""
    return F != "int8" and N == 64 and ck == 0 and T < 16


def per_tile(N):
    return 9 if N == 8 else 18  # N=8 runs the matmul group only


def text(path):
    return ANSI.sub("", open(path, errors="replace").read()) if os.path.exists(path) else ""


def info(name):
    out, log = text(f"{R}/{name}.out"), text(f"{R}/{name}/stdout.log")
    e = re.search(r"^exit=(\d+)", out, re.M)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
    return dict(exit=int(e.group(1)) if e else None, cancelled="CANCELLED" in out, oom=bool(OOM.search(out + log)),
                rss=f"{int(rss.group(1)) / 2**20:.1f} GB" if rss else "-", wall=wall.group(1) if wall else "-",
                results=re.findall(r"\[\d+/\d+\]\s+\S+\s+(PASS|FAIL)", log))


def cycles(name):
    """{test_N<N>_T<T>: (per-set latencies, serial total, streamed total)} from one run's per-test logs."""
    d = f"{R}/{name}/mesh_results/readiness"
    if not os.path.isdir(d):
        return {}
    o = subprocess.run([f"{J}/cycles.sh", d], capture_output=True, text=True).stdout
    c = {}
    for ln in o.splitlines():
        if ": " not in ln:
            continue
        t, v = ln.split(": ", 1)
        lat = tuple(int(x) for x in re.findall(r"Cycles to complete : (\d+)", v))
        tot = [int(x) for x in re.findall(r"\d+ sets in (\d+) cycles", v)]
        if lat and len(tot) == 2:
            c[t] = (lat, tot[0], tot[1])
    return c


def merged(names):
    c, conflicts = {}, set()
    for n in names:
        for t, v in cycles(n).items():
            if t in c and c[t] != v:
                conflicts.add(t)
            c[t] = v
    return c, conflicts


def delta(N, T, ck):
    """fp32 latency minus int8's: multiplier 8 -> 1, PE adder 5 -> 1, reduce tree clog2(RP*U + 1) levels of 5 -> of 1."""
    AK, RP = (N, 1) if ck else (T, N // T)
    return (8 - 1) + (5 - 1) + 5 * (RP * min(AK, 6)).bit_length() - (RP * min(AK, 2)).bit_length()


runs, run_tile = defaultdict(list), {}
for d in sorted(glob.glob(f"{R}/g3*/")):
    n = os.path.basename(d.rstrip("/"))
    m = RUN.match(n)
    if not m:
        continue
    br = "int8" if m.group(1) == "g3i" else "bf16" if m.group(1) in REF else None
    if br:
        runs[(br, int(m.group(2)), m.group(3), int(m.group(4)))].append(n)
        run_tile[n] = int(m.group(5)) if m.group(5) else None

L = ["SystolicMesh gate G3, int8 build: TB_SystolicMesh regression, bit-exact against mesh_model (matmul_int in int8),",
     "N = 8 (matmul group only: the conv generators cannot build a 3x3 kernel at N=8), 16, 32, 64, every tile size, collapse-k 1 and 0.",
     "fp32 and bf16 rerun on the int8 tree; their cycles compared test by test with the bf16 branch's G3 runs.", "",
     "1. Tests passed (over the point's tile and group runs) and fp32 / bf16 cycles against the bf16 branch",
     f"{'N':>3} {'format':<6} {'ck':>2} {'runs':>4} {'passed':>16}  cycles vs bf16 branch"]
cyc = {}
for N in SIZES:
    for F in FORMATS:
        for ck in (1, 0):
            names = runs.get(("int8", N, F, ck), [])
            res = [r for n in names for r in info(n)["results"]]
            cyc[(N, F, ck)] = mine = merged(names)[0]
            cmp = "-"
            if F != "int8":
                ref, conf = merged(runs.get(("bf16", N, F, ck), []))
                common = sorted(set(mine) & set(ref))
                diff = [t for t in common if mine[t] != ref[t]]
                cmp = f"{len(common) - len(diff)}/{len(common)} identical"
                cmp += ("; DIFFER: " + " ".join(diff)) if diff else ""
                cmp += f"; {len(set(mine) - set(ref))} passed, no cycle reference" if set(mine) - set(ref) else ""
                cmp += ("; reference runs disagree: " + " ".join(sorted(conf))) if conf else ""
            want = per_tile(N) * len([T for T in tiles(N) if not subset_skip(N, F, ck, T)])
            fails = res.count("FAIL")
            passed = f"{res.count('PASS')}/{want}" + (f" ({fails} FAIL)" if fails else "")
            L.append(f"{N:>3} {F:<6} {ck:>2} {len(names):>4} {passed:>16}  {cmp}")

L += ["", "2. int8 against fp32 on this tree. Expected: latency and streamed total = fp32 - D, serial total = fp32 - sets x D;",
      "   D = 7 (multiplier 8 -> 1) + 4 (PE adder 5 -> 1) + 5 x L32 - L8, L = clog2(RP x U + 1) reduce levels (derived, not measured)",
      f"{'N':>3} {'ck':>2} {'T':>3} {'D':>3} {'fp32 lat':>8} {'int8 lat':>8} {'serial fp32/int8':>17} {'stream fp32/int8':>17}  tests as expected"]
bad = []
for N in SIZES:
    for ck in (1, 0):
        fi, ff = cyc.get((N, "int8", ck), {}), cyc.get((N, "fp32", ck), {})
        for T in tiles(N):
            D = delta(N, T, ck)
            keys = sorted(t for t in set(fi) & set(ff) if t.endswith(f"_N{N}_T{T}"))
            good = [t for t in keys if fi[t][0] == tuple(x - D for x in ff[t][0])
                    and fi[t][1] == ff[t][1] - len(ff[t][0]) * D and fi[t][2] == ff[t][2] - D]
            bad += [f"   N={N} ck={ck} {t}: fp32 {ff[t]}, int8 {fi[t]}" for t in keys if t not in good]
            r = next((t for t in keys if t.startswith("mm_random")), keys[0] if keys else None)
            if r:
                L.append(f"{N:>3} {ck:>2} {T:>3} {D:>3} {ff[r][0][0]:>8} {fi[r][0][0]:>8} {f'{ff[r][1]}/{fi[r][1]}':>17}"
                         f" {f'{ff[r][2]}/{fi[r][2]}':>17}  {len(good)}/{len(keys)}")
            else:
                L.append(f"{N:>3} {ck:>2} {T:>3} {D:>3}  no int8 and fp32 pair at this point")
if bad:
    L += ["   not as expected (each needs an explanation before the gate passes):"] + bad

L += ["", "3. Random power-up (-DNO_ZERO_INIT, --x-initial unique, +verilator+rand+reset+2), N=16: cycles against the zero-init run"]
for d in sorted(glob.glob(f"{R}/g3i_ri_*/")):
    n = os.path.basename(d.rstrip("/"))
    m = RI.match(n)
    if not m:
        continue
    F, ck, i = m.group(1), int(m.group(2)), info(n)
    c, z = merged([n])[0], cyc.get((16, F, ck), {})
    common = set(c) & set(z)
    L.append(f"   {n:<18} {i['results'].count('PASS')}/{len(i['results'])} pass, exit {i['exit']}, "
             f"cycles {sum(1 for t in common if c[t] == z[t])}/{len(common)} identical to the zero-init run")

L += ["", "4. Points not run, or run in part (N, format, collapse-k, T): tests with results / expected, and why"]
for N in SIZES:
    for F in FORMATS:
        for ck in (1, 0):
            have = defaultdict(int)
            for t in cyc.get((N, F, ck), {}):
                have[int(re.search(r"_T(\d+)$", t).group(1))] += 1
            for T in tiles(N):
                if have[T] >= per_tile(N):
                    continue
                if subset_skip(N, F, ck, T):
                    why = "not rerun: the fp32 / bf16 subset at G3" + (
                        "; the bf16 branch could not build it in 256 GB either" if (N, F, ck, T) in NOT_BUILDABLE else "")
                elif F != "int8" and (N, F, ck, T) in NOT_BUILDABLE:
                    why = "not launched: the bf16 branch could not build it in 256 GB (Verilator)"
                else:
                    rs = [n for n in runs.get(("int8", N, F, ck), []) if run_tile[n] in (None, T)]
                    ii = {n: info(n) for n in rs}
                    oom = [n + " (peak " + ii[n]["rss"] + ")" for n in rs if ii[n]["oom"]]
                    cut = [n for n in rs if ii[n]["exit"] is None]
                    if oom:
                        why = "out of memory: " + ", ".join(oom)
                    elif cut:
                        why = "cancelled or still running: " + ", ".join(cut)
                    elif rs:
                        why = "no results in " + ", ".join(rs)
                    else:
                        why = "not launched"
                L.append(f"   N={N:<2} {F:<4} ck={ck} T={T:<2}  {have[T]}/{per_tile(N)}  {why}")

L += ["", "5. Runs: exit, wall time, peak memory"]
for n in sorted(x for k, v in runs.items() if k[0] == "int8" for x in v) + \
        sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{R}/g3i_ri_*/")):
    i = info(n)
    L.append(f"   {n:<30} exit {str(i['exit']):>4}  wall {i['wall']:>10}  peak {i['rss']:>9}")
print("\n".join(L))
```

- [ ] **Step 3: Launch the sweep**

Formats int8, fp32 and bf16; N = 8, 16, 32, 64; every tile size; collapse-k 1 and 0. N = 8 runs `--group matmul`. N >= 32
runs one job per tile (`--tiles`) with `TRACE=0 OPT_FAST=-O0` (`mesh_notrace_O0.sh`); collapse-k 0 at N=64 (every T) and
at N=32 T=2 also splits by group, since one of those tiles ran past 36 hours on the bf16 branch. Memory from the bf16
branch's peaks: 21 GB for fp32 N=32 collapse-k 0 T=4, 23 GB for bf16 N=64 collapse-k 1, and more than 256 GB for fp32
N=64 collapse-k 0 T=2.

```bash
for F in int8 fp32 bf16; do for C in 1 0; do
  $J/snap_launch_tree.sh g3i_N8_${F}_ck$C 32 2 $J/cmds/mesh_notrace.sh reg 8 $F $C --group matmul
  $J/snap_launch_tree.sh g3i_N16_${F}_ck$C 64 12 $J/cmds/mesh_notrace.sh reg 16 $F $C
  for T in 2 4 8 16 32; do
    M=64; [ $C = 0 ] && [ $T -le 8 ] && M=128
    if [ $C = 0 ] && [ $T = 2 ]; then
      for G in matmul conv; do
        $J/snap_launch_tree.sh g3i_N32_${F}_ck0_T2_$G 256 48 $J/cmds/mesh_notrace_O0.sh reg 32 $F 0 --tiles 2 --group $G
      done
    else
      $J/snap_launch_tree.sh g3i_N32_${F}_ck${C}_T$T $M 24 $J/cmds/mesh_notrace_O0.sh reg 32 $F $C --tiles $T
    fi
  done
  for T in 2 4 8 16 32 64; do
    if [ $C = 1 ]; then
      $J/snap_launch_tree.sh g3i_N64_${F}_ck1_T$T 64 24 $J/cmds/mesh_notrace_O0.sh reg 64 $F 1 --tiles $T
    else
      [ $F != int8 ] && [ $T -lt 16 ] && continue   # the fp32 / bf16 subset: N=64 collapse-k 0 only at T >= 16
      M=$([ $T -le 8 ] && echo 256 || echo 128)
      for G in matmul conv; do
        $J/snap_launch_tree.sh g3i_N64_${F}_ck0_T${T}_$G $M 48 $J/cmds/mesh_notrace_O0.sh reg 64 $F 0 --tiles $T --group $G
      done
    fi
  done
done; done
for F in int8 fp32 bf16; do $J/snap_launch_tree.sh g3i_ri_${F}_ck1 32 6 $J/cmds/mesh_randinit.sh 16 $F 1; done
$J/snap_launch_tree.sh g3i_ri_int8_ck0 64 12 $J/cmds/mesh_randinit.sh 16 int8 0
```

int8 is launched at every point, including N=64 collapse-k 0 T=2 and T=4: its PEs are much smaller than fp32's (no
Karatsuba multiplier, no normalizer), so it may fit where fp32 did not. If it does not, that is a not-run point with its
peak memory, not a failure.

Poll with `squeue -u $USER`. Rules while it runs:
- A job cancelled at its time limit: relaunch that point split one level further (by group, or for N=64 collapse-k 1 by
  group too, `..._T<T>_<group>`); the summary merges every run of a point.
- A job killed for memory: do not retry it on the same node size; it is reported as not run.
- A failed test: stop and debug it (superpowers:systematic-debugging); for int8, check the failing set's `matrixC` against
  `mesh_model.matmul_int` of its own `matrixA/B` first. An RTL fix restarts the sweep from Step 3 for every format,
  because the fix can move cycles anywhere.

- [ ] **Step 4: Build the report**

```bash
mkdir -p /proj/work/spramanik/SIENNA_int8/testbenches/results/int8
python3 $J/g3i_summary.py > /proj/work/spramanik/SIENNA_int8/testbenches/results/int8/mesh_gate.log   # reads logs only
```

Then append two hand-written sections to `mesh_gate.log`:

```
6. Findings
   <every row of section 2 that is not as expected, with its explanation from the RTL; every fp32/bf16 DIFFER; any
    reference conflict; "none" if there are none>

7. Not covered at G3
   - Covered, for the record: the ACC_W bias input, the bias queue and the int32 wrap, by every int8 set's bias file
     (Task 14; set 1 of every test within 4,096 of INT32_MAX, so its positive sums wrap).
   - The fp32 / bf16 points outside the subset (N=64 collapse-k 0, T < 16), by ruling.
   - partial_i multi-pass sums and the weight cache in int8: TB_SystolicMesh ties them off. TB_PE_int8 covers two-pass
     sets at PE level; the mesh paths are covered at G4.
   - Lint: Task 13's m13_lint (int8 both collapse modes clean; unsupported formats and wrong ACC_W rejected).
   - <the not-run points of section 4, restated with their reason>
```

Expected, for the gate to pass:
- Section 1: every launched point has `passed` equal to its expected count with no `FAIL` (int8: 27, 72, 90, 108 per
  collapse mode for N = 8, 16, 32, 64, less only at not-run points; fp32 and bf16 the same except 54 at N = 64
  collapse-k 0, the subset's three tile sizes); every fp32 and bf16 row `k/k identical` with no `DIFFER`. Tests whose
  bf16-branch run never finished show as `passed, no cycle reference`; they must pass, and are listed in section 7 as
  compared to nothing.
- Section 2: `tests as expected` equal to the test count at every (N, ck, T), or each exception explained in section 6.
  At N=16 the int8 latencies are the ones in Task 14 Step 8.
- Section 3: every random power-up run all pass, `exit 0`, cycles identical to its zero-init run.
- Section 4: only memory-limit points, the subset's `not rerun` points, and bf16-branch points never finished if Task 0
  recorded any. A point cancelled for time must have been relaunched split, not left in this list.

- [ ] **Step 5: Commit (SIENNA) and push**

```bash
cd /proj/work/spramanik/SIENNA_int8
D=$(date +%F)
cp testbenches/results/int8/mesh_gate.log .claude/skills/sienna-report/history/${D}_mesh_gate_int8.txt
git add SystolicMesh Makefile synth/sienna_rtl.f \
  && git commit -m "Bump SystolicMesh: the int8 mesh (int8 operands, int32 sums and results), matmul_int golden, --format int8; its integer units in the file lists"
git add .claude/skills/sienna-report/history/${D}_mesh_gate_int8.txt && git commit -m "sienna-report history: G3 mesh gate, int8"
git push origin int8
git status -sb   # no 'ahead'
```

The `sienna-int8` skill's status line is Task 24's.

The pointer bump and the file lists are one commit: at any other split, one SIENNA commit would list files its
SystolicMesh does not have, or have a mesh its file lists cannot link.

- [ ] **Step 6: Report to Soham**

Send the gate report's sections 1, 2, 4 and 6 in short form: tests passed per format, fp32/bf16 cycles identical (or
not), int8 latency against the prediction, the not-run points and why. Level 4 starts only when every check above passes.

---

## Level 4: SIENNA (gate G4 in Task 23)

Paths are relative to `/proj/work/spramanik/SIENNA_int8` (branch `int8` in all four repos). `/proj/work/spramanik/SIENNA`
stays on `bf16` and is the fp32 / bf16 reference tree. Every farm command below uses these two shell helpers:

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/work/spramanik/SIENNA_int8
L() { TREE=/proj/work/spramanik/SIENNA_int8 $J/snap_launch_tree.sh "$@"; }  # this tree
B() { TREE=/proj/work/spramanik/SIENNA $J/snap_launch_tree.sh "$@"; }       # the bf16 tree: fp32 / bf16 reruns
```

The fp32 / bf16 references are Task 0's runs: `i0_sienna_fp32`, `i0_sienna_bf16` (N = 16), `i0_reg32_fp32`,
`i0_reg32_bf16` (N = 32) and `i0_multi_fp32`, made on the int8 tree before any change, that is on the bf16 branch's code.

Order: 16, 17, 18, 19, then 20 onward. An int8 `sienna_top` first elaborates in Task 19, since it needs the int8 pooling
pad (17) and dropout (18); until then int8 is checked by unit testbenches, and fp32 / bf16 by their regressions.

### Task 16: Requantize at the GPNAE lane feed

**Files:**
- Create: `src/requant_lanes.sv`, `testbenches/TB_requant_lanes.sv`, `testbenches/gen_rq_lanes.py`
- Create (no repo): `$J/cmds/int8_unit.sh`, `$J/cmds/int8_rq_lanes.sh`, `$J/cmds/int8_cmp_reg.py`
- Modify: `GPNAE` (pointer to the G2 tip; SystolicMesh's moved to the G3 tip in Task 15), `Makefile`, `synth/sienna_rtl.f`,
  `src/sienna_top.sv`

**Interfaces:**
- Consumes: `tfliteRequant` (Task 7), `sienna_fmt_pkg::is_int`, `acc_w`, `req_lat()`, `REQ_ROUNDING` (Task 3); the mesh's
  `bias_i` and `wide_read_data_o` at `sienna_fmt_pkg::acc_w(EXP_W, MAN_W)` bits, derived by the mesh itself (Task 13);
  `gpnae_poly`'s int8 parameter inputs `gp_mx_i [15:0], gp_shx_i [4:0], gp_zin_i [7:0], gp_mout_i [31:0],
  gp_shout_i [7:0], gp_zout_i [7:0]` (Task 11); `ipu.requant`, `tflite_ref.ROUNDING` (Tasks 1, 2); the references
  `i0_sienna_fp32`, `i0_sienna_bf16` (Task 0).
- Produces:
  - `requant_lanes #(NUM_LANES, N, PER_LANE = N*N/NUM_LANES, string ROUNDING = sienna_fmt_pkg::REQ_ROUNDING)` with ports
    `clk_i, rstn_i, clear_i, valid_i, acc_i [NUM_LANES-1:0][31:0], mult_i [N-1:0][31:0], shift_i [N-1:0][7:0], zp_i,
    min_i, max_i [7:0], valid_o, result_o [NUM_LANES-1:0][7:0]`: lane k's word at beat b (beats counted on `valid_i` since
    `clear_i`) is requantized with channel `(k*PER_LANE + b) % N` (D-3); `valid_o` follows `valid_i` by
    `sienna_fmt_pkg::req_lat()` cycles. `sienna_top` instantiates it with `.ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)`.
  - `sienna_top #(... ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W))`: `bias_i [N-1:0][ACC_W-1:0]`; new inputs (D-2), taken
    with each accepted start and ignored outside int8: `req_mult_i [N-1:0][31:0], req_shift_i [N-1:0][7:0], req_zp_i,
    req_min_i, req_max_i [7:0], gp_mx_i [15:0], gp_shx_i [4:0], gp_mout_i [31:0], gp_shout_i [7:0], gp_zout_i [7:0]`.
    The activated set's values are used: for an accumulate group, the last pass's (like `activation_function_i`), while
    the bias is the first pass's (the mesh's rule). `sienna_top` drives each lane's `gp_zin_i` from the set's `req_zp`.
    Module-level nets `p_zp` (the zero point dropout drops to, for the set pooling holds; Task 18 makes it D-5's) and
    `IS_INT` for Tasks 17 and 18. `$fatal` on an unsupported format or a wrong `ACC_W`.
  - `$J/cmds/int8_unit.sh TOP FILES...` builds and runs one unit testbench from `testbenches/`, log in
    `testbenches/results/int8/TOP.log`, exit 0 only on `RESULT: PASSED`.
  - `$J/cmds/int8_cmp_reg.py REF_RUN NEW_RUN`: test-by-test result and cycles (single set and the three streamed passes)
    of two regression runs; exit 1 on any difference. Importable: `compare(ref, new) -> (ok, lines)`.

- [ ] **Step 1: Shared job scripts**

`$J/cmds/int8_unit.sh` (`chmod +x`):

```bash
#!/bin/bash
# Builds one unit testbench and runs it from testbenches/; args: TOP then its sources, the format package first; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
top=$1; shift
mkdir -p testbenches/results/int8
verilator --binary --timing --assert -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-WIDTHCONCAT -Werror-USERFATAL --top-module $top "$@" \
  -o sim --Mdir Verilator_$top > testbenches/results/int8/build_$top.log 2>&1 \
  || { tail -30 testbenches/results/int8/build_$top.log; echo "BUILD FAILED: $top"; exit 1; }
(cd testbenches && ../Verilator_$top/sim) | tee testbenches/results/int8/$top.log
command grep -q "RESULT: PASSED" testbenches/results/int8/$top.log
```

`$J/cmds/int8_cmp_reg.py`:

```python
#!/usr/bin/env python3
"""Compares two SIENNA regression runs test by test: result, single-set cycles and the cycles of each streamed pass;
args: REF_RUN NEW_RUN (names under sienna_jobs/runs); exit 1 if a reference test differs or is missing."""
import glob
import os
import re
import sys

R = "/proj/work/spramanik/sienna_jobs/runs"


def tests(run: str) -> dict:
    """{test: (result, single-set cycles, [cycles of each streamed pass])} from the run's per-test logs."""
    d = {}
    for p in sorted(glob.glob(f"{R}/{run}/results/pipeline/*.log")):
        raw = open(p, errors="ignore").read()
        if "RESULT:" not in raw:
            continue
        m = re.search(r"asserted @ \d+\s+\((\d+) cycles\)", raw)
        d[os.path.basename(p)[:-4]] = ("PASSED" if "RESULT: PASSED" in raw else "FAILED", int(m.group(1)) if m else None,
                                       [int(x) for x in re.findall(r"\[Stream\] \d+ sets in (\d+) cycles", raw)])
    return d


def compare(ref: str, new: str) -> tuple:
    """(identical, report lines) over every test of the reference run."""
    a, b = tests(ref), tests(new)
    diffs = [f"{t}: reference {a[t]}, new {b.get(t, 'missing')}" for t in a if b.get(t) != a[t]]
    extra = sorted(set(b) - set(a))
    summary = (f"{len(a)} reference tests, {len(b)} new, {len(a) - len(diffs)} identical"
               + (f"; new only: {', '.join(extra)}" if extra else ""))
    return bool(a) and not diffs, diffs + [summary]


if __name__ == "__main__":
    ok, lines = compare(sys.argv[1], sys.argv[2])
    print("\n".join(lines))
    print("IDENTICAL" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)
```

The fp32 / bf16 references of this task are Task 0's `i0_sienna_fp32` and `i0_sienna_bf16` (`Passed : 29 / 29` each).

- [ ] **Step 2: Write the failing unit test**

`testbenches/gen_rq_lanes.py`:

```python
#!/usr/bin/env python3
"""Vectors for TB_requant_lanes: random sets of wide-read beats through requant_lanes' channel map, expected values from
ipu.requant in the variant tflite_ref pins; arg: output .mem path (32-bit hex words)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402
import tflite_ref  # noqa: E402

N, LANES, SETS = 16, 32, 6  # TB_requant_lanes' geometry
PER = N * N // LANES


def main(path: str) -> None:
    rng = np.random.RandomState(8)
    out = [SETS]
    for s in range(SETS):
        mult = rng.randint(1 << 30, 1 << 31, N).astype(np.int64)
        if s == 0:
            mult[0], mult[1] = (1 << 31) - 1, 1 << 30  # the largest multiplier and the smallest normalized one
        shift = -((s * N + np.arange(N)) % 32).astype(np.int64)  # every right shift 0..31 across the sets
        zp = int(rng.randint(-128, 128))
        amin, amax = [(-128, 127), (zp, 127), (zp, min(127, zp + 50)), (-128, 127), (-40, 40), (-128, zp)][s]
        amin, amax = min(amin, amax), max(amin, amax)
        small = rng.randint(-(1 << 15), 1 << 15, (PER, LANES))
        full = rng.randint(-(1 << 31), (1 << 31) - 1, (PER, LANES))
        acc = np.where(rng.rand(PER, LANES) < 0.5, small, full).astype(np.int64)
        if s == 0:
            acc[0, :4] = [-(1 << 31), (1 << 31) - 1, 0, -1]
        c = (np.arange(LANES)[None, :] * PER + np.arange(PER)[:, None]) % N  # channel of lane k at beat b
        want = ipu.requant(acc, mult[c], shift[c], zp, amin, amax, tflite_ref.ROUNDING)
        out += [zp, amin, amax] + mult.tolist() + shift.tolist()
        for b in range(PER):
            out += acc[b].tolist() + [int(v) for v in want[b]]
    with open(path, "w") as f:
        f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in out))


if __name__ == "__main__":
    main(sys.argv[1])
```

`testbenches/TB_requant_lanes.sv`:

```systemverilog
`timescale 1ns / 100ps

// requant_lanes against ipu.requant: per-channel parameters, lane k's channel (k*PER_LANE + b) % N, 3-cycle latency.
// rq_lanes.mem (gen_rq_lanes.py): SETS; per set zp, min, max, N multipliers, N shifts, then per beat NUM_LANES sums and NUM_LANES results.
module TB_requant_lanes;
  localparam int N = 16, NUM_LANES = 32, PER_LANE = N * N / NUM_LANES, REQ_LAT = sienna_fmt_pkg::req_lat();
  logic clk_i = 0, rstn_i = 0, clear_i = 1, valid_i = 0, valid_o;
  logic [NUM_LANES-1:0][31:0] acc_i = '0;
  logic [N-1:0][31:0] mult_i = '0;
  logic [N-1:0][7:0] shift_i = '0;
  logic [7:0] zp_i = '0, min_i = '0, max_i = '0;
  logic [NUM_LANES-1:0][7:0] result_o;
  always #5 clk_i = ~clk_i;

  requant_lanes #(.NUM_LANES(NUM_LANES), .N(N), .ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) dut (.*);

  logic [31:0] v[$];
  logic [NUM_LANES-1:0][7:0] want_q[$];  // filled before reset is released, then only popped by the checker
  longint t_in[$];
  longint cyc = 0;
  int errs = 0, lat_errs = 0, n_out = 0;

  // One block owns the checker's state: the cycle count, the input times and the pops.
  always @(posedge clk_i) begin
    cyc <= cyc + 1;
    if (valid_i) t_in.push_back(cyc);
    if (valid_o) begin
      automatic logic [NUM_LANES-1:0][7:0] w = want_q.pop_front();
      automatic longint t = t_in.pop_front();
      n_out++;
      if (cyc - t != REQ_LAT) lat_errs++;
      for (int k = 0; k < NUM_LANES; k++)
        if (result_o[k] !== w[k]) begin
          errs++;
          if (errs <= 20)
            $display("[FAIL] beat %0d lane %0d: got %0d, want %0d", n_out - 1, k, $signed(result_o[k]), $signed(w[k]));
        end
    end
  end

  initial begin
    integer fh;
    logic [31:0] w32;
    int p, sets;
    fh = $fopen("rq_lanes.mem", "r");
    if (fh == 0) begin
      $display("[FATAL] cannot open rq_lanes.mem");
      $finish;
    end
    while ($fscanf(fh, "%h", w32) == 1) v.push_back(w32);
    $fclose(fh);
    sets = int'(v[0]);
    p = 1;
    for (int s = 0; s < sets; s++) begin
      p += 3 + 2 * N;
      for (int b = 0; b < PER_LANE; b++) begin
        logic [NUM_LANES-1:0][7:0] row;
        for (int k = 0; k < NUM_LANES; k++) row[k] = v[p+NUM_LANES+k][7:0];
        want_q.push_back(row);
        p += 2 * NUM_LANES;
      end
    end
    repeat (4) @(posedge clk_i);
    rstn_i = 1;
    repeat (2) @(posedge clk_i);
    p = 1;
    for (int s = 0; s < sets; s++) begin
      @(negedge clk_i);
      clear_i = 1;
      zp_i  = v[p][7:0];
      min_i = v[p+1][7:0];
      max_i = v[p+2][7:0];
      for (int c = 0; c < N; c++) begin
        mult_i[c]  = v[p+3+c];
        shift_i[c] = v[p+3+N+c][7:0];
      end
      p += 3 + 2 * N;
      @(negedge clk_i);
      clear_i = 0;
      for (int b = 0; b < PER_LANE; b++) begin
        if (s % 2 == 1 && b == 3) begin  // a bubble: beats count only with valid_i
          valid_i = 0;
          @(negedge clk_i);
        end
        valid_i = 1;
        for (int k = 0; k < NUM_LANES; k++) acc_i[k] = v[p+k];
        p += 2 * NUM_LANES;
        @(negedge clk_i);
      end
      valid_i = 0;
      repeat (REQ_LAT + 2) @(negedge clk_i);
    end
    repeat (4) @(negedge clk_i);
    if (n_out != sets * PER_LANE || want_q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d beats out, expected %0d", n_out, sets * PER_LANE);
    end
    if (lat_errs != 0) $display("[FAIL] %0d beats did not come %0d cycles after their input", lat_errs, REQ_LAT);
    $display("TB_requant_lanes: %0d beats, %0d errors, %0d latency errors", n_out, errs, lat_errs);
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (errs == 0 && lat_errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
```

`$J/cmds/int8_rq_lanes.sh` (`chmod +x`):

```bash
#!/bin/bash
# requant_lanes unit test: vectors from ipu.requant, then TB_requant_lanes; run from a snapshot root.
python3 testbenches/gen_rq_lanes.py testbenches/rq_lanes.mem || { echo "VECTOR GENERATION FAILED"; exit 1; }
A=SystolicMesh/ArithmeticLibrary
exec bash /proj/work/spramanik/sienna_jobs/cmds/int8_unit.sh TB_requant_lanes $A/Common/src/sienna_fmt_pkg.sv \
  $A/Multipliers/Int/src/intMultiplier.sv $A/Adders/Int/src/intAdder.sv $A/Requant/src/tfliteRequant.sv \
  src/requant_lanes.sv testbenches/TB_requant_lanes.sv
```

- [ ] **Step 3: Run it to see it fail**

```bash
L s16_rq 32 1 $J/cmds/int8_rq_lanes.sh
```

Expected: `BUILD FAILED: TB_requant_lanes`, the build log naming the missing `src/requant_lanes.sv`.

- [ ] **Step 4: Write `requant_lanes`**

`src/requant_lanes.sv`. The rounding is `sienna_fmt_pkg::REQ_ROUNDING`, the variant G0 pinned (Task 3 copied it from
`rounding.txt`); no RTL names the variant as a literal, and Task 20's `_check_rounding()` fails every int8 run if the
package, `ipu.REQ_ROUNDING` and `tflite_ref.ROUNDING` disagree.

```systemverilog
`timescale 1ns / 100ps

// int8: one tfliteRequant per lane on the mesh's wide read. Lane k's word at beat b is element k*PER_LANE + b of the
// row-major N x N result, so its output channel is that element's column, (k*PER_LANE + b) % N.
module requant_lanes #(
    parameter int    NUM_LANES = 32,
    parameter int    N         = 16,
    parameter int    PER_LANE  = N * N / NUM_LANES,
    parameter string ROUNDING  = sienna_fmt_pkg::REQ_ROUNDING  // TFLite reference kernels' variant, pinned at G0 (Task 2)
) (
    input  logic                       clk_i,
    input  logic                       rstn_i,
    input  logic                       clear_i,  // no set in the stage: the next beat is beat 0
    input  logic                       valid_i,  // one wide-read beat
    input  logic [NUM_LANES-1:0][31:0] acc_i,    // int32 sums
    input  logic [N-1:0][31:0]         mult_i,   // per output channel, Q0.31
    input  logic [N-1:0][7:0]          shift_i,  // per output channel, signed
    input  logic [7:0]                 zp_i,     // output zero point
    input  logic [7:0]                 min_i,    // clamp, signed: the int8 range, or the fused ReLU / ReLU6
    input  logic [7:0]                 max_i,
    output logic                       valid_o,  // the beat, sienna_fmt_pkg::req_lat() cycles later
    output logic [NUM_LANES-1:0][7:0]  result_o
);
  localparam int BW = $clog2(PER_LANE + 1);
  localparam int CW = (N > 1) ? $clog2(N) : 1;
  logic [BW-1:0] beat;
  logic [NUM_LANES-1:0] done;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) beat <= '0;
    else if (clear_i) beat <= '0;
    else if (valid_i) beat <= beat + 1'b1;
  end

  for (genvar k = 0; k < NUM_LANES; k++) begin : LANE
    logic [CW-1:0] ch;
    assign ch = CW'((k * PER_LANE + int'(beat)) % N);
    tfliteRequant #(.ROUNDING(ROUNDING)) rq (
        .clk_i    (clk_i),
        .rstn_i   (rstn_i),
        .valid_i  (valid_i),
        .acc_i    (acc_i[k]),
        .mult_i   (mult_i[ch]),
        .shift_i  (shift_i[ch]),
        .zp_i     (zp_i),
        .act_min_i(min_i),
        .act_max_i(max_i),
        .result_o (result_o[k]),
        .done_o   (done[k])
    );
  end
  assign valid_o = done[0];  // every lane takes the same beats

endmodule
```

- [ ] **Step 5: Run it to see it pass**

```bash
L s16_rq 32 1 $J/cmds/int8_rq_lanes.sh
```

Expected: `TB_requant_lanes: 48 beats, 0 errors, 0 latency errors` and `RESULT: PASSED`.

- [ ] **Step 6: Bump GPNAE and the file lists**

The clone's submodules sit on their `int8` tips (Task 0); G3 (Task 15) and G2 (Task 12) must have passed at those tips.
SIENNA's SystolicMesh pointer is already at the G3 tip (Task 15); this step moves GPNAE's.

```bash
cd /proj/work/spramanik/SIENNA_int8
git -C SystolicMesh status -sb | head -1 && git -C GPNAE status -sb | head -1   # both "## int8...origin/int8", no "ahead"
test "$(git -C SystolicMesh/ArithmeticLibrary rev-parse HEAD)" = "$(git -C GPNAE/ArithmeticLibrary rev-parse HEAD)" && echo SAME-ARIL
```

Expected: `SAME-ARIL`. The goldens import `ipu.py` and `fpu.py` through both copies, and Python loads each module once.

Verilator links every instantiated module, including those inside generate branches a format does not take, so the new
units go into every file list now, not when int8 first elaborates. Task 13 already added `intMultiplier` and `intAdder`
(committed in Task 15); this step adds `fxMac` (GPNAE's `barrel_mac`), `tfliteRequant`, `requant_lanes` and GPNAE's
`gpnae_poly_int8` (Task 11). `Makefile`: in `TOP_FILES`, `	sienna_top.sv \` is followed by a new line
`	requant_lanes.sv \`; in `SM_LIB_FILES`, the list's last line `	Adders/Int/src/intAdder.sv` becomes

```make
	Adders/Int/src/intAdder.sv \
	Multipliers/Fx/src/fxMac.sv \
	Requant/src/tfliteRequant.sv
```

and in `GPNAE_FILES` the last line `	gpnae_poly.sv` becomes `	gpnae_poly_int8.sv \` and `	gpnae_poly.sv`.
`synth/sienna_rtl.f`: after `SystolicMesh/ArithmeticLibrary/Adders/Int/src/intAdder.sv` (Task 13's) add

```
SystolicMesh/ArithmeticLibrary/Multipliers/Fx/src/fxMac.sv
SystolicMesh/ArithmeticLibrary/Requant/src/tfliteRequant.sv
```

`GPNAE/src/gpnae_poly_int8.sv` on the line before `GPNAE/src/gpnae_poly.sv`, and `src/requant_lanes.sv` on the line
before `src/sienna_top.sv`. Then:

```bash
command grep -c . synth/sienna_rtl.f   # 34: 28, Task 13's 2, and these 4
make check-files 2>&1 | tail -3        # no missing file
```

- [ ] **Step 7: `sienna_top`: ports, per-set parameters, the requantize stage**

In `src/sienna_top.sv`:

1. After the `DATA_WIDTH` parameter line add:

```systemverilog
    parameter int    ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // sums, bias, mesh results: int32 in int8, DATA_WIDTH otherwise
```

2. Replace the `bias_i` port line with these (other formats ignore every `req_*` and `gp_*` input):

```systemverilog
    input logic [N-1:0][ACC_W-1:0]  bias_i,
    input logic [N-1:0][31:0]       req_mult_i,   // int8, with the start: requantize multiplier (Q0.31) of each output channel (column)
    input logic [N-1:0][7:0]        req_shift_i,  // int8: its shift, signed
    input logic [7:0]               req_zp_i,     // int8, layer-wide: output zero point; dropout drops to it after ReLU or linear (D-5)
    input logic [7:0]               req_min_i,    // int8: clamp, signed; the fused ReLU or ReLU6 lives here
    input logic [7:0]               req_max_i,
    input logic [15:0]              gp_mx_i,      // int8 GPNAE: rescale of the lane input to Q4.11
    input logic [4:0]               gp_shx_i,
    input logic [31:0]              gp_mout_i,    // int8 GPNAE: SELU's output requantize
    input logic [7:0]               gp_shout_i,
    input logic [7:0]               gp_zout_i,
```

3. Above `localparam int GPNAE_DATA_WIDTH = DATA_WIDTH;` add:

```systemverilog
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);  // int8: int32 sums, requantized at the lane feed

  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "sienna_top: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC
    $fatal(1, "sienna_top: ACC_W=%0d, but the format accumulates in %0d bits", ACC_W, sienna_fmt_pkg::acc_w(EXP_W, MAN_W));
  end
```

4. After `logic [CRW-1:0] gp_sets;` (the per-set arrays) add:

```systemverilog
  // int8: the requantize and GPNAE parameters of the set the activation stage holds (g_*); p_zp: dropout's drop value for the pooled set.
  logic [N-1:0][31:0] g_mult;
  logic [N-1:0][7:0]  g_shift;
  logic [7:0]         g_zp, g_min, g_max, g_shout, g_zout, p_zp;
  logic [15:0]        g_mx;
  logic [4:0]         g_shx;
  logic [31:0]        g_mout;
```

5. `logic [NUM_LANES-1:0][DATA_WIDTH-1:0] wide_rd_data;  // element k of lane k's block` becomes

```systemverilog
  logic [NUM_LANES-1:0][ACC_W-1:0] wide_rd_data;  // element k of lane k's block; int32 sums in int8
```

6. In the `gpnae_poly` instantiation, after `.control_word_i(gpnae_ctrl[g]),` add:

```systemverilog
          .gp_mx_i       (g_mx),
          .gp_shx_i      (g_shx),
          .gp_zin_i      (g_zp),     // int8: the lane's input is the requantize output, whose zero point is req_zp_i
          .gp_mout_i     (g_mout),
          .gp_shout_i    (g_shout),
          .gp_zout_i     (g_zout),
```

7. The bypass write keeps ReLU's sign test to float formats; in int8 ReLU is the requantize clamp, and a negative int8
code can be a positive real value. Replace

```systemverilog
        gpnae_out_mem[act_wr_base + k * PER_LANE + fill_count[k]] <= (act_is_relu && fill_d[k][DATA_WIDTH-1]) ? '0 : fill_d[k];
```

with

```systemverilog
        gpnae_out_mem[act_wr_base + k * PER_LANE + fill_count[k]] <= (!IS_INT && act_is_relu && fill_d[k][DATA_WIDTH-1]) ? '0 : fill_d[k];
```

8. After the stage controllers' `always_ff` (the block ending with `p_next_id <= p_next_id + 1'b1;`) add:

```systemverilog
  // int8: requantize and GPNAE parameters travel with each set, indexed by its id like set_act (D-2). At g_accept the
  // activation stage copies its set's per-channel words, so each lane chooses among N words, not NUM_IDS * N.
  if (IS_INT) begin : G_REQ_SETS
    logic [N-1:0][31:0] s_mult [NUM_IDS];
    logic [N-1:0][7:0]  s_shift[NUM_IDS];
    logic [7:0]  s_zp[NUM_IDS], s_min[NUM_IDS], s_max[NUM_IDS], s_shout[NUM_IDS], s_zout[NUM_IDS];
    logic [15:0] s_mx[NUM_IDS];
    logic [4:0]  s_shx[NUM_IDS];
    logic [31:0] s_mout[NUM_IDS];
    always_ff @(posedge clk_i) begin  // no reset: an id's entry is written by its own accept before any stage reads it
      if (host_accept) begin
        s_mult[host_next_id]  <= req_mult_i;
        s_shift[host_next_id] <= req_shift_i;
        s_zp[host_next_id]    <= req_zp_i;
        s_min[host_next_id]   <= req_min_i;
        s_max[host_next_id]   <= req_max_i;
        s_mx[host_next_id]    <= gp_mx_i;
        s_shx[host_next_id]   <= gp_shx_i;
        s_mout[host_next_id]  <= gp_mout_i;
        s_shout[host_next_id] <= gp_shout_i;
        s_zout[host_next_id]  <= gp_zout_i;
      end
      if (g_accept) begin
        g_mult  <= s_mult[g_next_id];
        g_shift <= s_shift[g_next_id];
      end
    end
    assign g_zp    = s_zp[g_set_id];
    assign g_min   = s_min[g_set_id];
    assign g_max   = s_max[g_set_id];
    assign g_mx    = s_mx[g_set_id];
    assign g_shx   = s_shx[g_set_id];
    assign g_mout  = s_mout[g_set_id];
    assign g_shout = s_shout[g_set_id];
    assign g_zout  = s_zout[g_set_id];
    assign p_zp    = s_zp[p_set_id];  // Task 18 replaces this with D-5's choice per activation
  end else begin : G_NO_REQ_SETS
    assign g_mult  = '0;
    assign g_shift = '0;
    assign g_zp    = '0;
    assign g_min   = '0;
    assign g_max   = '0;
    assign g_mx    = '0;
    assign g_shx   = '0;
    assign g_mout  = '0;
    assign g_shout = '0;
    assign g_zout  = '0;
    assign p_zp    = '0;
  end
```

9. Replace

```systemverilog
  assign fill_v = wide_rd_valid;
  assign fill_d = wide_rd_data;
```

with

```systemverilog
  // int8: the lanes, the bypass write and every fill counter see requantized int8 beats, REQ_LAT cycles after the wide read.
  if (IS_INT) begin : G_REQ
    requant_lanes #(
        .NUM_LANES(NUM_LANES),
        .N        (N),
        .PER_LANE (PER_LANE),
        .ROUNDING (sienna_fmt_pkg::REQ_ROUNDING)
    ) rq (
        .clk_i   (clk_i),
        .rstn_i  (rstn_i),
        .clear_i (g_state == G_IDLE),
        .valid_i (wide_rd_valid),
        .acc_i   (wide_rd_data),
        .mult_i  (g_mult),
        .shift_i (g_shift),
        .zp_i    (g_zp),
        .min_i   (g_min),
        .max_i   (g_max),
        .valid_o (fill_v),
        .result_o(fill_d)
    );
  end else begin : G_NO_REQ
    assign fill_v = wide_rd_valid;
    assign fill_d = wide_rd_data;
  end
```

The stage controller needs no new state for the 3-cycle latency, and this is why: every count that ends a fill counts
`fill_v`, the requantizer's output. `fill_count` drives the lanes' start and the bypass `g_done`; `filled_total` and
`all_collected` end the polynomial round. So a set finishes only after its last delayed beat. The mesh bank is released
right after the last read is issued, as before; the beats still in flight are held in the requantizer's own pipeline
registers. The beat counter clears in `G_IDLE`, and `G_IDLE` is reached only after all `PER_LANE` delayed beats have
arrived, so no beat of the previous set is in flight at the clear. `g_mult` and `g_shift` are loaded at `g_accept`, two
cycles before the first read is issued, and `g_set_id` changes at the same edge. Task 22's model adds the `req_lat()`
(3) cycles to the activation stage.

Then check:

```bash
command grep -n "fill_v\|fill_d" src/sienna_top.sv   # only the G_REQ / G_NO_REQ drivers and the existing readers
```

- [ ] **Step 8: fp32 and bf16 unchanged**

```bash
L s16_fp32 32 4 $J/cmd_sienna_fmt.sh reg 16 4 fp32
L s16_bf16 32 4 $J/cmd_sienna_fmt.sh reg 16 4 bf16
python3 $J/cmds/int8_cmp_reg.py i0_sienna_fp32 s16_fp32    # login node: parses logs only
python3 $J/cmds/int8_cmp_reg.py i0_sienna_bf16 s16_bf16
```

Expected: 29/29 in both formats, and `IDENTICAL` for both comparisons (the same result, single-set cycles and streamed
cycles for every test).

- [ ] **Step 9: Commit (SIENNA) and push**

```bash
git add GPNAE && git commit -m "Bump GPNAE: the fixed-point int8 lane (G2)"
git add src/requant_lanes.sv && git commit -m "requant_lanes: one tfliteRequant per lane on the wide read, channel from the element's column"
git add src/sienna_top.sv && git commit -m "sienna_top: int8 requantize at the lane feed, per-set parameters (D-2), int32 bias and results"
git add Makefile synth/sienna_rtl.f && git commit -m "File lists: fxMac, tfliteRequant, requant_lanes, gpnae_poly_int8"
git add testbenches/TB_requant_lanes.sv testbenches/gen_rq_lanes.py && git commit -m "TB_requant_lanes: per-channel requantize against ipu.requant"
git push origin int8 && git status -sb | head -1   # no "ahead"
```

### Task 17: `Maxpool_2D` in int8, pad -128

**Files:**
- Modify: `Maxpool/Maxpool_2D.sv`, `src/sienna_top.sv` (the pooling pad and the `IS_FP32` override)
- Create: `testbenches/TB_maxpool_int8.sv`

**Interfaces:**
- Consumes: `IS_INT` in `sienna_top` (Task 16).
- Produces: `Maxpool_2D` compares signed integers whenever `EXP_W == 0`, whatever `IS_FP32` says; its floor is -128 in
  int8 and -infinity in a float format, built without a zero-width replication. `sienna_top`'s `NEG_INF` is -128 in
  int8 (0x80) and unchanged in fp32 (0xFF800000) and bf16 (0xFF80). fp32 and bf16 behaviour unchanged.

Today, `FLOAT = IS_FP32 && (DATA_WIDTH == 1 + EXP_W + MAN_W)` is true at (0, 7), since 8 = 1 + 0 + 7. The unit would then
compare int8 codes as sign-magnitude floats, and max(-1, -2) would give -2. The floor `{1'b1, {EXP_W{1'b1}}, {MAN_W{1'b0}}}`
also replicates by zero.

- [ ] **Step 1: Write the failing test**

`testbenches/TB_maxpool_int8.sv`:

```systemverilog
`timescale 1ns / 1ps

// Maxpool_2D in int8 (EXP_W = 0): two's-complement order in the streaming window sienna_top uses and in the batch path
// with padding. Both DUTs leave IS_FP32 at 1, so EXP_W = 0 alone must select the integer compare.
module TB_maxpool_int8;
  localparam int W = 8;
  logic clk = 0, rst_n = 0;
  logic start = 0, valid_in = 0, done, out_valid;
  logic [W-1:0] data_in = '0, out_data;
  logic start_b = 0, valid_b = 0, done_b, out_valid_b;
  logic [W-1:0] data_b = '0, out_b;
  always #5 clk = ~clk;

  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(0), .MAN_W(7)) dut (
      .clk(clk), .rst_n(rst_n), .start(start), .done(done), .data_in(data_in), .valid_in(valid_in),
      .out_data(out_data), .out_valid(out_valid));
  // 3 x 3 input, 2 x 2 windows, stride 2, padding 1: every window holds padded positions.
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(3), .IN_COLS(3), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(1), .IS_FP32(1), .EXP_W(0), .MAN_W(7)) dut_b (
      .clk(clk), .rst_n(rst_n), .start(start_b), .done(done_b), .data_in(data_b), .valid_in(valid_b),
      .out_data(out_b), .out_valid(out_valid_b));

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
      $display("[FAIL] max(%0d %0d %0d %0d) = %0d, want %0d", $signed(v0), $signed(v1), $signed(v2), $signed(v3),
               $signed(got), $signed(want));
    end
  endtask

  // v holds the 3 x 3 input row-major, v[8] first; want holds the 2 x 2 output, want[3] first.
  task automatic batch(input logic [9*W-1:0] v, input logic [4*W-1:0] want);
    logic [W-1:0] got[$];
    int waited = 0;
    @(negedge clk) start_b = 1;
    @(negedge clk);
    for (int i = 0; i < 9; i++) begin
      valid_b = 1;
      data_b = v[i*W+:W];
      @(negedge clk);
    end
    valid_b = 0;
    while (got.size() < 4 && waited < 100) begin
      @(posedge clk);
      if (out_valid_b) got.push_back(out_b);
      waited++;
    end
    @(negedge clk) start_b = 0;
    repeat (2) @(negedge clk);
    for (int i = 0; i < 4; i++)
      if (i >= got.size() || got[i] !== want[i*W+:W]) begin
        errs++;
        $display("[FAIL] batch output %0d = %0d, want %0d", i, (i < got.size()) ? $signed(got[i]) : 999,
                 $signed(want[i*W+:W]));
      end
  endtask

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    window(8'hFF, 8'hFE, 8'h80, 8'h81, 8'hFF);  // all negative: -1 is the largest
    window(8'h05, 8'hFB, 8'h7F, 8'h00, 8'h7F);  // mixed signs: 127
    window(8'h80, 8'h80, 8'h80, 8'h81, 8'h81);  // -128 three times loses to -127
    window(8'h80, 8'h80, 8'h80, 8'h80, 8'h80);  // all -128: the floor itself
    window(8'h00, 8'hFF, 8'h01, 8'h80, 8'h01);  // 1 over 0 and the negatives
    // [[-100, -3, -128], [-7, -1, -2], [-128, -50, -128]]: windows see {-100}, {-3, -128}, {-7, -128}, {-1, -2, -50, -128}
    batch({8'h80, 8'hCE, 8'h80, 8'hFE, 8'hFF, 8'hF9, 8'h80, 8'hFD, 8'h9C},
          {8'hFF, 8'hF9, 8'hFD, 8'h9C});
    $display("TB_maxpool_int8: %0d errors", errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
```

The batch input concatenation lists v[8] first, so the row-major input `-100, -3, -128, -7, -1, -2, -128, -50, -128`
reads right to left: `9C` is element 0. The expected outputs likewise: `9C` (-100) is output 0, then `FD` (-3), `F9` (-7),
`FF` (-1).

```bash
L s17_mp 32 1 $J/cmds/int8_unit.sh TB_maxpool_int8 Maxpool/Maxpool_2D.sv testbenches/TB_maxpool_int8.sv
```

Expected: `RESULT: FAILED` with `[FAIL]` lines for windows 1 and 3 and for the batch outputs (sign-magnitude order puts
-1 below -2, and -128 reads as -0), or a build failure on the zero-width replication in `FLOOR`.

- [ ] **Step 2: Integer compare at EXP_W = 0, and a floor without zero-width replication**

In `Maxpool/Maxpool_2D.sv`, the `IS_FP32` parameter's comment becomes
`// float sign-magnitude compare given EXP_W > 0; EXP_W = 0 (int8) always compares signed integers`, and

```systemverilog
  localparam bit FLOAT = IS_FP32 && (DATA_WIDTH == 1 + EXP_W + MAN_W);
  localparam logic [DATA_WIDTH-1:0] FLOOR = FLOAT ? {1'b1, {EXP_W{1'b1}}, {MAN_W{1'b0}}}  // -infinity in the format
                                                  : {1'b1, {(DATA_WIDTH - 1) {1'b0}}};  // most negative integer
```

becomes

```systemverilog
  localparam bit FLOAT = IS_FP32 && (EXP_W > 0) && (DATA_WIDTH == 1 + EXP_W + MAN_W);  // EXP_W = 0 is int8
  localparam logic [DATA_WIDTH-1:0] FLOOR = FLOAT ? DATA_WIDTH'({DATA_WIDTH{1'b1}} << MAN_W)  // -infinity: sign and exponent all ones
                                                  : {1'b1, {(DATA_WIDTH - 1) {1'b0}}};  // most negative integer, -128 in int8
```

At (8, 23) the shift gives 0xFF800000 and at (8, 7) 0xFF80, the values the old concatenation gave.

- [ ] **Step 3: The pad in `sienna_top`**

In `src/sienna_top.sv`, the `NEG_INF` line (below `IS_INT` since Task 16) becomes

```systemverilog
  localparam logic [DATA_WIDTH-1:0] NEG_INF = IS_INT ? {1'b1, {(DATA_WIDTH - 1) {1'b0}}}  // pooling pad: -128 in int8
                                                     : DATA_WIDTH'({DATA_WIDTH{1'b1}} << MAN_W);  // -infinity in a float format
```

and in the `Maxpool_2D` instantiation `.IS_FP32    (1),` becomes `.IS_FP32    (!IS_INT),`. Max pooling pads with -128:
max commutes with the monotonic requantize and activation, and -128 is never above a window's real element.

- [ ] **Step 4: Run it, and bf16 unchanged**

```bash
L s17_mp 32 1 $J/cmds/int8_unit.sh TB_maxpool_int8 Maxpool/Maxpool_2D.sv testbenches/TB_maxpool_int8.sv
L s17_mpbf 32 1 $J/cmds/int8_unit.sh TB_maxpool_fmt Maxpool/Maxpool_2D.sv testbenches/TB_maxpool_fmt.sv
```

Expected: `TB_maxpool_int8: 0 errors`, `RESULT: PASSED`; `TB_maxpool_fmt: 0 errors`, `RESULT: PASSED`. fp32 pooling is
covered by Task 19's regression rerun.

- [ ] **Step 5: Commit (SIENNA) and push**

```bash
git add Maxpool/Maxpool_2D.sv && git commit -m "Maxpool_2D: signed compare whenever EXP_W is 0; floor without a zero-width replication"
git add src/sienna_top.sv && git commit -m "sienna_top: pooling pad -128 in int8, integer compare in Maxpool_2D"
git add testbenches/TB_maxpool_int8.sv && git commit -m "TB_maxpool_int8: two's-complement windows, padded batch windows"
git push origin int8
```

### Task 18: `dropout` in int8 (D-5)

**Files:**
- Modify: `Dropout/dropout.sv` (CRLF line endings: keep them), `src/sienna_top.sv` (`p_zp` per D-5, one connection)
- Create: `testbenches/TB_dropout_int8.sv`

**Interfaces:**
- Consumes: `p_zp`, `G_REQ_SETS`' `s_zp` and `s_zout` in `sienna_top` (Task 16); `set_act[p_set_id]` (existing).
- Produces: `dropout` gains `input [DATA_WIDTH-1:0] zero_point_i`. In int8 (`G_INT`, no multiplier): inference passes
  every beat unchanged; training passes a kept beat unchanged and replaces a dropped beat with `zero_point_i`, in the
  same cycle, with the keep decision the fp32 path uses (the LFSR word the beat advances to). The 1/keep factor is the
  software's: it is folded into the next layer's scale. fp32 and bf16 unchanged; they ignore `zero_point_i`.
  `sienna_top` drives it with `p_zp`, the output zero point of the pooled set's activation (D-5): `req_zp` after ReLU
  or linear, 0 after tanh, -128 after sigmoid, `gp_zout` after SELU.

- [ ] **Step 1: Write the failing test**

`testbenches/TB_dropout_int8.sv`:

```systemverilog
`timescale 1ns / 1ps

// dropout in int8 (D-5): inference passes every beat; training keeps a beat unchanged or drops it to zero_point_i in the
// same cycle, deciding on the LFSR word the beat advances to, replayed here from the seed.
module TB_dropout_int8;
  localparam int W = 8;
  localparam logic [31:0] SEED = 32'h2ACE002A;
  localparam logic [31:0] THR = 32'((64'hFFFFFFFF * 64'd50) / 64'd100);  // dropout.sv's threshold at 50%
  logic clk = 0, rst_n = 0, in_valid = 0, reseed = 0, training = 0, valid_out;
  logic [W-1:0] data_in = '0, data_out;
  logic [W-1:0] zp = 8'hF3;  // zero point -13
  always #5 clk = ~clk;

  dropout #(.EXP_W(0), .MAN_W(7), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .training_mode(training), .data_in(data_in), .reseed_i(reseed),
      .seed_i(SEED), .zero_point_i(zp), .data_out(data_out), .valid_out(valid_out));

  function automatic logic [31:0] lfsr_next(input logic [31:0] s);
    for (int i = 0; i < 32; i++) s = {s[30:0], s[31] ^ s[21] ^ s[1] ^ s[0]};
    return s;
  endfunction

  int errs = 0, kept = 0;
  initial begin
    logic [31:0] s;
    logic [W-1:0] want;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    s = SEED;
    for (int i = 0; i < 96; i++) begin
      @(negedge clk);
      training = (i >= 32);  // 32 inference beats, then 64 training beats
      in_valid = 1;
      data_in = W'(i * 37 - 100);
      s = lfsr_next(s);  // the LFSR advances on every valid beat, in inference too
      want = (!training || s >= THR) ? data_in : zp;
      #1;
      if (!valid_out || data_out !== want) begin
        errs++;
        $display("[FAIL] beat %0d (%s): %0d -> %0d valid %0b, want %0d", i, training ? "training" : "inference",
                 $signed(data_in), $signed(data_out), valid_out, $signed(want));
      end
      if (training && s >= THR) kept++;
    end
    @(negedge clk) in_valid = 0;
    if (kept == 0 || kept == 64) begin
      errs++;
      $display("[FAIL] %0d of 64 training beats kept", kept);
    end
    $display("TB_dropout_int8: %0d training beats kept of 64, %0d errors", kept, errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
```

```bash
A=SystolicMesh/ArithmeticLibrary
DO="$A/Common/src/sienna_fmt_pkg.sv $A/Multipliers/Radix4Booth/src/R4Booth.sv $A/Multipliers/Karatsuba/src/karatsubaUnsigned.sv \
    $A/Multipliers/FP32/src/fp32Multiplier.sv $A/Multipliers/FP/src/fpMultiplier.sv Dropout/dropout.sv"
L s18_do 32 1 $J/cmds/int8_unit.sh TB_dropout_int8 $DO testbenches/TB_dropout_int8.sv
```

`$DO` expands on the login node into the job's argument list, so no quoting reaches the farm. Expected:
`BUILD FAILED: TB_dropout_int8`, the log naming `zero_point_i` (no such port).

- [ ] **Step 2: The int8 branch**

In `Dropout/dropout.sv`:

1. After `input wire [LFSR_WIDTH-1:0] seed_i,  // must be nonzero` add

```systemverilog
    input wire [DATA_WIDTH-1:0] zero_point_i,  // int8 training: a dropped beat becomes this (D-5); other formats ignore it
```

2. Before `localparam logic [63:0] MAX_LFSR_VAL_64 = ...` add

```systemverilog
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);  // int8: no multiplier, the 1/keep factor lives in the next layer's scale
```

3. The parameter check becomes `if (!IS_INT && DROPOUT_P_PERCENT != 50 && CONST_SCALE == DATA_WIDTH'(sienna_fmt_pkg::from_fp32(32'h40000000, MAN_W)))`.

4. In the `always_comb`, between the inference branch and `end else begin`, insert

```systemverilog
    end else if (IS_INT) begin
      // int8 training: in the same cycle, a kept beat unchanged, a dropped beat the zero point (D-5).
      data_out  = (lfsr_next >= DROPOUT_THRESHOLD) ? data_in : zero_point_i;
      valid_out = in_valid;
```

5. In the keep-queue `always_ff`, `if (training_mode && in_valid) begin` becomes `if (!IS_INT && training_mode && in_valid) begin`,
and `if (training_mode && mult_done) kq_rd <= kq_rd + 1'b1;` becomes `if (!IS_INT && training_mode && mult_done) kq_rd <= kq_rd + 1'b1;`.

6. Between the `G_BAD_FORMAT` block and `end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32` insert

```systemverilog
  end else if (IS_INT) begin : G_INT
    assign mult_out  = '0;
    assign mult_done = 1'b0;
```

Line endings: after editing, `file Dropout/dropout.sv` must still say `CRLF`. If the edit wrote LF lines, run
`sed -i 's/\r*$/\r/' Dropout/dropout.sv`, then check `git diff --stat Dropout/dropout.sv` counts only the edited lines
(about 12 insertions, 3 changed lines).

In `src/sienna_top.sv`, `G_REQ_SETS` (Task 16) replaces `assign p_zp    = s_zp[p_set_id];  // Task 18 replaces this ...`
with D-5's choice, from the pooled set's activation code:

```systemverilog
    // D-5: a dropped value is the output zero point of the pooled set's activation (tanh, and every code the lane runs as tanh: 0).
    always_comb
      case (set_act[p_set_id])
        CONTROL_WIDTH'(3'b001): p_zp = s_zout[p_set_id];  // SELU: its requantized output's zero point
        CONTROL_WIDTH'(3'b010): p_zp = 8'h80;  // sigmoid: TFLite's fixed output zero point -128
        CONTROL_WIDTH'(3'b100), CONTROL_WIDTH'(3'b101): p_zp = s_zp[p_set_id];  // ReLU, linear: the requantize output's
        default: p_zp = 8'h00;  // tanh: zero point 0
      endcase
```

and in the `dropout` instantiation, after `.seed_i        (lane_seed),` add `.zero_point_i  (p_zp),`. Task 20's golden
makes the same choice (`drop_zp`).

- [ ] **Step 3: Run it, bf16 unchanged, bad format still rejected**

```bash
L s18_do 32 1 $J/cmds/int8_unit.sh TB_dropout_int8 $DO testbenches/TB_dropout_int8.sv
L s18_dobf 32 1 $J/cmds/int8_unit.sh TB_dropout_fmt $DO testbenches/TB_dropout_fmt.sv
```

Expected: `TB_dropout_int8: <k> training beats kept of 64, 0 errors` with 0 < k < 64, `RESULT: PASSED`; and
`TB_dropout_fmt: 64 outputs, <k> kept, 0 errors`, `RESULT: PASSED`. The bf16 testbench does not connect `zero_point_i`,
which Verilator ties to zero with a PINMISSING warning that is off by default; if the build stops on it, add
`.zero_point_i('0)` to `TB_dropout_fmt.sv`'s instantiation and commit that with the testbench below.

- [ ] **Step 4: Commit (SIENNA) and push**

```bash
git add Dropout/dropout.sv && git commit -m "dropout: int8 without a multiplier; dropped beats become the zero point (D-5)"
git add src/sienna_top.sv && git commit -m "sienna_top: dropout drops to the output zero point of the pooled set's activation (D-5)"
git add testbenches/TB_dropout_int8.sv && git commit -m "TB_dropout_int8: inference bypass, training keep mask replayed from the seed"
git push origin int8
```

### Task 19: int8 through sienna_top, sienna_layer, sienna_multi and their testbenches

**Files:**
- Modify: `src/sienna_layer.sv`, `src/sienna_multi.sv`
- Modify: `testbenches/TB_sienna_top.sv`, `TB_sienna_layer.sv`, `TB_sienna_multi.sv`, `TB_sienna_model.sv`
- Modify: `regression.py` (package items only)
- Modify (no repo): `$J/cmd_multi.sh` (optional format argument); create `$J/cmds/int8_stim_same.sh`, `$J/cmds/int8_cmp_gemm.py`

**Interfaces:**
- Consumes: Task 16's `sienna_top` ports and `ACC_W`; Tasks 17 and 18; `int8_cmp_reg.py` (Task 16); the references
  `i0_sienna_fp32`, `i0_sienna_bf16`, `i0_multi_fp32` (Task 0).
- Produces:
  - `sienna_layer #(... ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W))` with int8 inputs (ignored in other formats):
    `cfg_req_zp_i, cfg_req_min_i, cfg_req_max_i [7:0], cfg_gp_mx_i [15:0], cfg_gp_shx_i [4:0], cfg_gp_mout_i [31:0],
    cfg_gp_shout_i, cfg_gp_zout_i [7:0]`, taken with `cfg_load_i`; and `w_bias_i [N-1:0][ACC_W-1:0], w_req_mult_i
    [N-1:0][31:0], w_req_shift_i [N-1:0][7:0]`, taken beside each column block's bias beat on the weight stream (whose
    `w_data_i` is then ignored). In int8 every block has a bias beat, since its bias carries the folded input zero point.
  - `sienna_multi #(... ACC_W)` with `bias_i` at `ACC_W` and the D-2 inputs, passed to every copy.
  - `test_config_pkg` gains `IS_INT` and `ACC_W` (generated by `regression._config_items`, new).
  - Testbench file formats in int8 (`IS_INT`): `requant_<k>.mem` is 8 layer-wide words (zp, min, max, gp_mx, gp_shx,
    gp_mout, gp_shout, gp_zout), then N multipliers, then N shifts, each an 8-digit two's-complement hex word;
    `bias_<k>.mem` holds 8-digit int32 words. TB_sienna_model's set file gains the same 8 + 2N words after each set's bias
    words. TB_sienna_layer's layer file gains a line `Q zp min max mx shx mout shout zout` (decimal) after the `L` line,
    and after the rows one epilogue per column block: N bias words, N multipliers, N shifts (8-digit hex); its report line
    ends in `epilogues=<taken>/<expected>`.
  - `$J/cmd_multi.sh CONFIG COPIES [LANES] [COLLAPSE_K] [N] [FMT]`.
  - `$J/cmds/int8_stim_same.sh OLD_REGRESSION_COPY FMT [N] [T]`: every test's generated files from this tree and from
    the older copy, compared byte for byte (the package may differ only by `ACC_W` and `IS_INT` lines).
  - `$J/cmds/int8_cmp_gemm.py REF_RUN NEW_RUN [N]`: sets and cycles per GEMM shape; importable `compare()`.

- [ ] **Step 1: Write the failing checks**

```bash
L s19_lint8 32 1 $J/cmd_sienna_fmt.sh lint 0 7
L s19_bad 32 1 $J/cmd_sienna_fmt.sh lint 0 15
```

Expected now: `sienna_layer (0, 7): exit=1` with `%Error` lines in `src/sienna_layer.sv` (`ONE`'s `EXP_W'(...)` cast has
width 0 at `EXP_W = 0`), while `sienna_top (0, 7)` lints (Tasks 16-18); `(0, 15)` prints `sienna_top: unsupported format`
for both tops.

- [ ] **Step 2: `sienna_layer`**

In `src/sienna_layer.sv`:

1. Header comment: after the line `// Activation stream, ...` add

```systemverilog
// int8: every block has the bias beat (its bias carries the folded input zero point); its int32 bias, requantize
// multipliers and shifts come on w_bias_i, w_req_mult_i and w_req_shift_i beside it, and w_data_i is ignored.
```

2. After the `DATA_WIDTH` parameter line add

```systemverilog
    parameter int ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // bias and sums: int32 in int8, DATA_WIDTH otherwise
```

3. After `input logic [LFSR_WIDTH-1:0]    cfg_seed_i,` add

```systemverilog
    // int8 only, taken with cfg_load_i: the layer's requantize and GPNAE parameters (D-2)
    input logic [7:0]               cfg_req_zp_i,
    input logic [7:0]               cfg_req_min_i,
    input logic [7:0]               cfg_req_max_i,
    input logic [15:0]              cfg_gp_mx_i,
    input logic [4:0]               cfg_gp_shx_i,
    input logic [31:0]              cfg_gp_mout_i,
    input logic [7:0]               cfg_gp_shout_i,
    input logic [7:0]               cfg_gp_zout_i,
```

and after `output logic                              w_ready_o,` add

```systemverilog
    // int8 only: beside each block's bias beat on the weight stream
    input  logic [N-1:0][ACC_W-1:0]           w_bias_i,
    input  logic [N-1:0][31:0]                w_req_mult_i,
    input  logic [N-1:0][7:0]                 w_req_shift_i,
```

4. Replace the `ONE` line with

```systemverilog
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);
  localparam logic [DATA_WIDTH-1:0] ONE = IS_INT ? DATA_WIDTH'(1)  // int8: a residual adds its raw codes into the int32 sum
                                                 : DATA_WIDTH'(((1 << (EXP_W > 0 ? EXP_W - 1 : 0)) - 1) << MAN_W);  // 1.0 in the format
```

At (8, 23) this is 127 << 23 = 0x3F800000 and at (8, 7) 0x3F80, the values of the old concatenation.

5. After `logic [LFSR_WIDTH-1:0] seed_q;` add

```systemverilog
  logic [7:0] rq_zp_q, rq_min_q, rq_max_q, gp_shout_q, gp_zout_q;  // int8 layer-wide parameters
  logic [15:0] gp_mx_q;
  logic [4:0] gp_shx_q;
  logic [31:0] gp_mout_q;
```

6. `logic [N-1:0][DATA_WIDTH-1:0]  p_bias;` becomes

```systemverilog
  logic [N-1:0][ACC_W-1:0]       p_bias;
  logic [N-1:0][31:0]            p_mult;   // int8: the requantize words of the set being started
  logic [N-1:0][7:0]             p_shift;
```

7. In the `sienna_top` instantiation, after `.DATA_WIDTH       (DATA_WIDTH),` add `.ACC_W            (ACC_W),`, and
after `.bias_i                     (p_bias),` add

```systemverilog
      .req_mult_i                 (p_mult),
      .req_shift_i                (p_shift),
      .req_zp_i                   (rq_zp_q),
      .req_min_i                  (rq_min_q),
      .req_max_i                  (rq_max_q),
      .gp_mx_i                    (gp_mx_q),
      .gp_shx_i                   (gp_shx_q),
      .gp_mout_i                  (gp_mout_q),
      .gp_shout_i                 (gp_shout_q),
      .gp_zout_i                  (gp_zout_q),
```

8. `logic [N-1:0][DATA_WIDTH-1:0] bias_buf[2];` becomes

```systemverilog
  logic [N-1:0][ACC_W-1:0] bias_buf[2];
  logic [N-1:0][31:0] mult_buf[2];  // int8: the requantize words of the block each half holds
  logic [N-1:0][7:0] shift_buf[2];
```

9. In the driving `always_comb`, after `p_bias     = bias_buf[is_blk[0]];` add

```systemverilog
    p_mult     = mult_buf[is_blk[0]];
    p_shift    = shift_buf[is_blk[0]];
```

10. In the reset branch, after `seed_q <= '0;` add

```systemverilog
      rq_zp_q <= '0;
      rq_min_q <= '0;
      rq_max_q <= '0;
      gp_mx_q <= '0;
      gp_shx_q <= '0;
      gp_mout_q <= '0;
      gp_shout_q <= '0;
      gp_zout_q <= '0;
```

11. In the `cfg_load_i && !active` branch, `bias_q <= cfg_bias_i;` becomes
`bias_q <= cfg_bias_i || IS_INT;  // int8: every block's bias beat carries its requantize words`, and after
`seed_q <= cfg_seed_i;` add

```systemverilog
        rq_zp_q <= cfg_req_zp_i;
        rq_min_q <= cfg_req_min_i;
        rq_max_q <= cfg_req_max_i;
        gp_mx_q <= cfg_gp_mx_i;
        gp_shx_q <= cfg_gp_shx_i;
        gp_mout_q <= cfg_gp_mout_i;
        gp_shout_q <= cfg_gp_shout_i;
        gp_zout_q <= cfg_gp_zout_i;
```

12. The bias capture

```systemverilog
          if (wl_take_bias) begin
            bias_buf[wl_blk[0]] <= w_data_i;
            bias_in[wl_blk[0]] <= 1'b1;
          end
```

becomes

```systemverilog
          if (wl_take_bias) begin
            for (int c = 0; c < N; c++) bias_buf[wl_blk[0]][c] <= IS_INT ? w_bias_i[c] : ACC_W'(w_data_i[c]);
            mult_buf[wl_blk[0]] <= w_req_mult_i;
            shift_buf[wl_blk[0]] <= w_req_shift_i;
            bias_in[wl_blk[0]] <= 1'b1;
          end
```

In fp32 and bf16 `ACC_W == DATA_WIDTH`, so the bias path is the old one bit for bit, and the requantize words are
captured and never read. A residual pass in int8 adds the residual's raw codes into the int32 sum (identity weight 1);
TFLite's ADD rescale is sub-project 2b, so no 2a test uses it.

- [ ] **Step 3: `sienna_multi`**

In `src/sienna_multi.sv`: after the `DATA_WIDTH` parameter add
`parameter int ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),`; the `bias_i` port becomes

```systemverilog
    input logic [N-1:0][ACC_W-1:0]  bias_i,
    input logic [N-1:0][31:0]       req_mult_i,  // int8 (D-2): to the copy that takes the start, as sienna_top
    input logic [N-1:0][7:0]        req_shift_i,
    input logic [7:0]               req_zp_i,
    input logic [7:0]               req_min_i,
    input logic [7:0]               req_max_i,
    input logic [15:0]              gp_mx_i,
    input logic [4:0]               gp_shx_i,
    input logic [31:0]              gp_mout_i,
    input logic [7:0]               gp_shout_i,
    input logic [7:0]               gp_zout_i,
```

and in the `sienna_top` instantiation add `.ACC_W            (ACC_W),` after `.DATA_WIDTH` and, after
`.bias_i                     (bias_i),`:

```systemverilog
        .req_mult_i                 (req_mult_i),
        .req_shift_i                (req_shift_i),
        .req_zp_i                   (req_zp_i),
        .req_min_i                  (req_min_i),
        .req_max_i                  (req_max_i),
        .gp_mx_i                    (gp_mx_i),
        .gp_shx_i                   (gp_shx_i),
        .gp_mout_i                  (gp_mout_i),
        .gp_shout_i                 (gp_shout_i),
        .gp_zout_i                  (gp_zout_i),
```

Each copy captures them only when it accepts the start, as it does the bias.

- [ ] **Step 4: Package items in `regression.py`**

Move the items list of `generate_vectors` into a function, placed after `write_sv_package`, with `IS_INT` and `ACC_W`
added after `EXACT_GOLDEN`:

```python
def _config_items(cfg: dict, fmt: str, act_type: str, num_sets: int, credits: int, passes: int, mixed: list,
                  use_bias: bool, drop_seed: int) -> list:
    """test_config_pkg's items: geometry, the build's format, activation, pooling, dropout and the streamed sets."""
    N = cfg.get("n", 16)
    sram_depth = N * N
    return [
        ("N", N, "int"),
        ("TILE_SIZE", cfg.get("tile_size", 4), "int"),
        ("NUM_LANES", cfg.get("lanes", 32), "int"),
        ("HOST_WORDS", cfg.get("host_words", N), "int"),
        ("EXP_W", FORMATS[fmt][0], "int"),
        ("MAN_W", FORMATS[fmt][1], "int"),
        ("DATA_WIDTH", 1 + sum(FORMATS[fmt]), "int"),
        ("EXACT_GOLDEN", int(fmt != "fp32"), "int"),
        ("IS_INT", int(fmt == "int8"), "int"),
        ("ACC_W", 32 if fmt == "int8" else 1 + sum(FORMATS[fmt]), "int"),
        ("SRAM_DEPTH", sram_depth, "int"),
        ("FIFO_DEPTH", cfg.get("fifo_depth", sram_depth), "int"),
        ("ACTIVATION_CODE", activation_to_code(act_type), "int"),
        ("NUM_TERMS", get_polynomial_terms(act_type), "int"),
        ("IN_ROWS", N, "int"),
        ("IN_COLS", N, "int"),
        ("POOL_H", cfg.get("pool_h", 2), "int"),
        ("POOL_W", cfg.get("pool_w", 2), "int"),
        ("STRIDE_ROWS", cfg.get("pool_h", 2), "int"),
        ("STRIDE_COLS", cfg.get("pool_w", 2), "int"),
        ("PADDING", cfg.get("padding", 1), "int"),
        ("DROPOUT_P_PERCENT", int(round(cfg.get("dropout_p", 0.5) * 100)), "int"),
        ("LFSR_WIDTH", 32, "int"),
        ("CONTROL_WIDTH", 3, "int"),
        ("NUM_SETS", num_sets, "int"),
        ("SETS_IN_FLIGHT", credits, "int"),
        ("HAS_BIAS", int(use_bias), "int"),
        ("WEIGHT_CACHE", int(bool(cfg.get("cached", False))), "int"),
        ("ACCUM_PASSES", passes, "int"),
        ("MIXED_LEN", len(mixed), "int"),
        ("MIXED_ACTS", sum(activation_to_code(a) << (4 * i) for i, a in enumerate(mixed)), "int"),
        ("TRAINING_MODE", int(bool(cfg.get("training", False))), "int"),
        ("DROPOUT_SEED", drop_seed, "int"),
        ("ADDR_LINES", max(1, math.ceil(math.log2(sram_depth))), "int"),
    ]
```

In `generate_vectors`, the block from `sram_depth = N * N` to `write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"), items)`
becomes

```python
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),
                     _config_items(cfg, fmt, act_type, num_sets, credits, passes, mixed, use_bias, drop_seed))
```

`fmt` is already defined earlier in the function (`fmt = cfg.get("fmt_name", "fp32")`). Nothing else changes.

- [ ] **Step 5: `TB_sienna_top`**

1. Replace `logic [N-1:0][DATA_WIDTH-1:0] bias_i;` with

```systemverilog
  logic [N-1:0][ACC_W-1:0] bias_i;  // int32 in int8, where it carries the folded input zero point
  // int8: the requantize and GPNAE parameters of the set being started (D-2); zero in other formats
  logic [N-1:0][31:0] req_mult_i;
  logic [N-1:0][7:0]  req_shift_i;
  logic [7:0]         req_zp_i, req_min_i, req_max_i, gp_shout_i, gp_zout_i;
  logic [15:0]        gp_mx_i;
  logic [4:0]         gp_shx_i;
  logic [31:0]        gp_mout_i;
```

2. In the DUT's port list, after `.bias_i                     (bias_i),` add, one per line in the file's alignment:
`.req_mult_i(req_mult_i)`, `.req_shift_i(req_shift_i)`, `.req_zp_i(req_zp_i)`, `.req_min_i(req_min_i)`,
`.req_max_i(req_max_i)`, `.gp_mx_i(gp_mx_i)`, `.gp_shx_i(gp_shx_i)`, `.gp_mout_i(gp_mout_i)`, `.gp_shout_i(gp_shout_i)`,
`.gp_zout_i(gp_zout_i)`.

3. In `reset()`, after `bias_i = '0;` add

```systemverilog
    req_mult_i = '0;
    req_shift_i = '0;
    req_zp_i = '0;
    req_min_i = '0;
    req_max_i = '0;
    gp_mx_i = '0;
    gp_shx_i = '0;
    gp_mout_i = '0;
    gp_shout_i = '0;
    gp_zout_i = '0;
```

4. After `read_mem_file` add a 32-bit reader:

```systemverilog
  // ── 32-bit word reader: int32 biases and requantize words (8 digits), or narrower words zero-extended ──
  task automatic read_word_file(input string fn, output logic [31:0] q[$]);
    integer fh, rc;
    logic [31:0] w;
    q.delete();
    fh = $fopen(fn, "r");
    if (!fh) begin
      $display("[ERROR] Cannot open: %s", fn);
      $finish;
    end
    while (!$feof(fh)) begin
      rc = $fscanf(fh, "%h", w);
      if (rc == 1) q.push_back(w);
    end
    $fclose(fh);
  endtask
```

5. `apply_bias` reads 32-bit words and narrows them to `ACC_W` (fp32: the same word; bf16: the same 16 bits), and a new
task follows it:

```systemverilog
  task automatic apply_bias(input int k);
    logic [31:0] q[$];
    bias_valid_i = (HAS_BIAS != 0) && ((k % ACCUM_PASSES) == 0);
    bias_i = '0;
    if (bias_valid_i) begin
      read_word_file($sformatf("bias_%0d.mem", k), q);
      for (int c = 0; c < N; c++) bias_i[c] = ACC_W'(q[c]);
    end
  endtask

  // int8: set k's requantize and GPNAE parameters from requant_<k>.mem: 8 layer-wide words, N multipliers, N shifts.
  task automatic apply_requant(input int k);
    logic [31:0] q[$];
    if (!IS_INT) return;
    read_word_file($sformatf("requant_%0d.mem", k), q);
    if (q.size() != 8 + 2 * N) begin
      $display("[FATAL] requant_%0d.mem holds %0d words, expected %0d", k, q.size(), 8 + 2 * N);
      $finish;
    end
    req_zp_i   = q[0][7:0];
    req_min_i  = q[1][7:0];
    req_max_i  = q[2][7:0];
    gp_mx_i    = q[3][15:0];
    gp_shx_i   = q[4][4:0];
    gp_mout_i  = q[5];
    gp_shout_i = q[6][7:0];
    gp_zout_i  = q[7][7:0];
    for (int c = 0; c < N; c++) begin
      req_mult_i[c]  = q[8+c];
      req_shift_i[c] = q[8+N+c][7:0];
    end
  endtask
```

6. After each of the four calls of `apply_bias` (single-set pass `apply_bias(0);`, `BACK_TO_BACK` pass `apply_bias(1);`,
streaming producer and `reset_mid_stream`, both `apply_bias(k);`), add the matching `apply_requant(0);`,
`apply_requant(1);`, `apply_requant(k);`, `apply_requant(k);`. Check:
`command grep -c "apply_requant(" testbenches/TB_sienna_top.sv` prints 5 (the task and four calls).

- [ ] **Step 6: `TB_sienna_model`, `TB_sienna_multi`, `TB_sienna_layer`**

`testbenches/TB_sienna_model.sv`:
- `logic [N-1:0][DATA_WIDTH-1:0] bias_i;` becomes the declarations of `TB_sienna_top` Step 5.1, and the DUT gets the
  same ten connections after `.bias_i`;
- in the `initial` block, add `logic [31:0] w32;` beside `logic [DATA_WIDTH-1:0] w;`, zero the ten D-2 signals next to
  `bias_i = '0;` in the initialization, and replace the bias read

```systemverilog
      bias_i = '0;
      if (has_bias != 0)
        for (int c = 0; c < N; c++) begin
          rc = $fscanf(fin, "%h", w);
          bias_i[c] = w;
        end
```

with

```systemverilog
      bias_i = '0;
      if (has_bias != 0)
        for (int c = 0; c < N; c++) begin
          rc = $fscanf(fin, "%h", w32);
          bias_i[c] = ACC_W'(w32);
        end
      if (IS_INT) begin  // int8: the set's requantize and GPNAE words, laid out as requant_<k>.mem
        logic [31:0] q[8+2*N];
        for (int i = 0; i < 8 + 2 * N; i++) rc = $fscanf(fin, "%h", q[i]);
        req_zp_i   = q[0][7:0];
        req_min_i  = q[1][7:0];
        req_max_i  = q[2][7:0];
        gp_mx_i    = q[3][15:0];
        gp_shx_i   = q[4][4:0];
        gp_mout_i  = q[5];
        gp_shout_i = q[6][7:0];
        gp_zout_i  = q[7][7:0];
        for (int c = 0; c < N; c++) begin
          req_mult_i[c]  = q[8+c];
          req_shift_i[c] = q[8+N+c][7:0];
        end
      end
```

  and its header comment gains `// int8: after each set's bias words, 8 + 2N requantize words as in requant_<k>.mem.`

`testbenches/TB_sienna_multi.sv` (its DUT connects by `.*`, so every new port needs a signal of its name):
- `logic [N-1:0][DATA_WIDTH-1:0] bias_i = '0;` becomes

```systemverilog
  logic [N-1:0][ACC_W-1:0] bias_i = '0;
  logic [N-1:0][31:0] req_mult_i = '0;  // int8 (D-2)
  logic [N-1:0][7:0] req_shift_i = '0;
  logic [7:0] req_zp_i = '0, req_min_i = '0, req_max_i = '0, gp_shout_i = '0, gp_zout_i = '0;
  logic [15:0] gp_mx_i = '0;
  logic [4:0] gp_shx_i = '0;
  logic [31:0] gp_mout_i = '0;
```

- add `read_word_file` (Step 5.4) after `read_mem_file`, and `apply_requant` (the second task of Step 5.5);
- in the set loop, after `dropout_seed_i = set_seed(k);` add

```systemverilog
      bias_valid_i = (HAS_BIAS != 0);  // this TB streams no accumulate groups: every set carries its own bias
      if (bias_valid_i) begin
        logic [31:0] bw[$];
        read_word_file($sformatf("bias_%0d.mem", k), bw);
        for (int c = 0; c < N; c++) bias_i[c] = ACC_W'(bw[c]);
      end
      apply_requant(k);
```

  For the fp32 configurations `cmd_multi.sh` has been run with (none with `bias`), `HAS_BIAS` is 0 and nothing changes.

`testbenches/TB_sienna_layer.sv`:
- header comment: after the `Layer file:` line add
  `// int8 (IS_INT): a line "Q zp min max mx shx mout shout zout" after it, and after the rows one epilogue per column`
  `// block (N biases, N multipliers, N shifts, 8 hex digits each), driven beside the block's bias beat.`
- after `logic [N-1:0][DATA_WIDTH-1:0] a_data_i = '0, w_data_i = '0;` add

```systemverilog
  logic [7:0] cfg_req_zp_i = '0, cfg_req_min_i = '0, cfg_req_max_i = '0, cfg_gp_shout_i = '0, cfg_gp_zout_i = '0;
  logic [15:0] cfg_gp_mx_i = '0;
  logic [4:0] cfg_gp_shx_i = '0;
  logic [31:0] cfg_gp_mout_i = '0;
  logic [N-1:0][ACC_W-1:0] w_bias_i = '0;  // int8: beside each block's bias beat
  logic [N-1:0][31:0] w_req_mult_i = '0;
  logic [N-1:0][7:0] w_req_shift_i = '0;
  logic [31:0] ep[$];  // int8 epilogues, 3N words per column block
```

- in the `initial` block's declarations add `int qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout, ne, ei;`,
  `logic [31:0] w32;` and `bit e_take;` (no initializers: this is a static block);
- after the `if (rc != 11 || kind != "L")` check add

```systemverilog
    {qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout} = '0;  // fp32 and bf16 keep every int8 input at 0
    if (IS_INT) begin
      rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d", kind, qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout);
      if (rc != 9 || kind != "Q") begin
        $display("[FATAL] %s: an int8 layer file needs its Q line", layer_f);
        $finish;
      end
    end
```

- after the row-reading loop, before `$fclose(fin);`:

```systemverilog
    ne = IS_INT ? (n + N - 1) / N : 0;  // one epilogue per column block
    for (int i = 0; i < ne * 3 * N; i++) begin
      rc = $fscanf(fin, "%h", w32);
      ep.push_back(w32);
    end
```

- in the configuration, after `cfg_seed_i = LFSR_WIDTH'(seed);`:

```systemverilog
    cfg_req_zp_i = 8'(qzp);
    cfg_req_min_i = 8'(qmin);
    cfg_req_max_i = 8'(qmax);
    cfg_gp_mx_i = 16'(qmx);
    cfg_gp_shx_i = 5'(qshx);
    cfg_gp_mout_i = 32'(qmout);
    cfg_gp_shout_i = 8'(qshout);
    cfg_gp_zout_i = 8'(qzout);
```

  and `ei = 0;` next to `ai = 0;`;
- in the streaming loop, inside the `for (int c = 0; c < N; c++)` that sets `a_data_i` and `w_data_i`, add

```systemverilog
        w_bias_i[c]      = (ei < ne) ? ACC_W'(ep[(3*ei)*N+c]) : '0;
        w_req_mult_i[c]  = (ei < ne) ? ep[(3*ei+1)*N+c] : '0;
        w_req_shift_i[c] = (ei < ne) ? ep[(3*ei+2)*N+c][7:0] : '0;
```

  after `w_take = w_valid_i && w_ready_o;` add
  `e_take = IS_INT && w_take && dut.wl_take_bias;  // the beat the layer takes as a block's bias`, and after
  `if (w_take) wi++;` add `if (e_take) ei++;`;
- the report line becomes

```systemverilog
    $display("[LAYER] sets=%0d outputs=%0d cycles=%0d a_rows=%0d/%0d w_rows=%0d/%0d epilogues=%0d/%0d", bounds.size(),
             res_q.size(), t_done - t0 + 1, ai, na, wi, nw, ei, ne);
```

  `model_runner.py`'s regex matches the unchanged prefix, so fp32 and bf16 runs parse as before.

- [ ] **Step 7: Job scripts**

`$J/cmd_multi.sh`: the comment's args become `CONFIG COPIES [LANES] [COLLAPSE_K] [N] [FMT]`, and its generator lines
become

```bash
cfg=$1; copies=$2; lanes=${3:-32}; ck=${4:-1}; n=${5:-16}; fmt=${6:-fp32}
python3 - "$cfg" "$lanes" "$n" "$fmt" <<'EOF'
import sys, regression as reg
t = next(x for x in reg.PIPELINE_TESTS if x["name"] == sys.argv[1])
reg.generate_vectors({"n": int(sys.argv[3]), "tile_size": 4, "lanes": int(sys.argv[2]), **t, "num_sets": 48, "fmt_name": sys.argv[4]})
EOF
```

(the `make` line is unchanged). With no sixth argument it generates exactly what it did.

`$J/cmds/int8_stim_same.sh` (`chmod +x`):

```bash
#!/bin/bash
# Every test's generated stimulus from this tree's regression.py and from an older copy, byte for byte; args: OLD_COPY FMT [N] [T]; run from a snapshot root.
cp "$1" regression_old.py
fmt=$2; n=${3:-16}; tsz=${4:-4}; bad=0; count=0
tests=$(python3 -c "import regression_old as r; print(' '.join(x['name'] for x in r.PIPELINE_TESTS))")
mkdir -p stim
for t in $tests; do
  for side in old new; do
    mod=regression; [ $side = old ] && mod=regression_old
    mkdir -p stim/${side}_$t
    python3 -c "
import sys, $mod as r
r.TB_DIR = sys.argv[2]
c = next(x for x in r.PIPELINE_TESTS if x['name'] == sys.argv[1])
r.generate_vectors({'n': $n, 'tile_size': $tsz, 'lanes': 32, 'host_words': $n, 'fmt_name': '$fmt', **c})" "$t" "stim/${side}_$t" \
      || { echo "GENERATION FAILED: $side $t"; bad=1; }
  done
  d=$(command diff -r stim/old_$t stim/new_$t | command grep -E '^[<>]|^Only|^Binary' | command grep -v -E '^> +localparam int (ACC_W|IS_INT) = ')
  if [ -n "$d" ]; then echo "DIFFERS: $t"; echo "$d" | head -5; bad=1; fi
  count=$((count + 1))
done
echo "$count tests compared in $fmt, $([ $bad = 0 ] && echo identical || echo 'DIFFERENCES')"
exit $bad
```

`regression.TB_DIR` is read at call time by every writer, `write_sv_package` and `_check_mem_widths`, so each side
writes into its own new directory and no stale file can hide a difference.

`$J/cmds/int8_cmp_gemm.py`:

```python
#!/usr/bin/env python3
"""Compares two gemm_sweep runs shape by shape (sets and cycles); args: REF_RUN NEW_RUN [N]; exit 1 on any difference."""
import json
import os
import sys

R = "/proj/work/spramanik/sienna_jobs/runs"


def rows(run: str, n: int = 16) -> dict:
    p = f"{R}/{run}/results/gemm/gemm_sweep_N{n}.json"
    return {r["shape"]: (r["sets"], r["cycles"]) for r in json.load(open(p))} if os.path.exists(p) else {}


def compare(ref: str, new: str, n: int = 16) -> tuple:
    """(identical, report lines) over the shapes both runs have."""
    a, b = rows(ref, n), rows(new, n)
    common = [s for s in a if s in b]
    diffs = [f"{s}: reference {a[s]}, new {b[s]}" for s in common if a[s] != b[s]]
    return bool(common) and not diffs, diffs + [f"{len(common)} shapes in both, {len(common) - len(diffs)} identical"]


if __name__ == "__main__":
    ok, lines = compare(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    print("\n".join(lines))
    print("IDENTICAL" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)
```

- [ ] **Step 8: Run the checks**

```bash
mkdir -p .claude/scratch && git -C /proj/work/spramanik/SIENNA show HEAD:regression.py > .claude/scratch/regression_bf16.py
L s19_lint8 32 1 $J/cmd_sienna_fmt.sh lint 0 7
L s19_lint 32 1 $J/cmds/g4lint.sh                    # bf16 and fp32
L s19_bad 32 1 $J/cmd_sienna_fmt.sh lint 0 15
L s19_bad2 32 1 $J/cmd_sienna_fmt.sh lint 5 10
L s19_fp32 32 4 $J/cmd_sienna_fmt.sh reg 16 4 fp32
L s19_bf16 32 4 $J/cmd_sienna_fmt.sh reg 16 4 bf16
L s19_stim32 32 2 $J/cmds/int8_stim_same.sh .claude/scratch/regression_bf16.py fp32
L s19_stim16 32 2 $J/cmds/int8_stim_same.sh .claude/scratch/regression_bf16.py bf16
L s19_gemm32 64 6 $J/venv/bin/python gemm_sweep.py --n 16 --format fp32 --quick
L s19_gemm16 64 6 $J/venv/bin/python gemm_sweep.py --n 16 --format bf16 --quick
L s19_multi 32 4 $J/cmd_multi.sh matmul_random_tanh 2 32 1 16 fp32
L s19_model 32 2 make verilator TOP_MODULE=TB_sienna_model TESTBENCH=TB_sienna_model.sv TRACE=0
```

(`s19_model` builds on the package the snapshot's last regression wrote, an fp32 one; it only has to build and print
`[MODEL] no +sets= and +out= given`.) Then, on the login node (log parsing only):

```bash
python3 $J/cmds/int8_cmp_reg.py i0_sienna_fp32 s19_fp32
python3 $J/cmds/int8_cmp_reg.py i0_sienna_bf16 s19_bf16
python3 $J/cmds/int8_cmp_gemm.py s22_gemm_fp32 s19_gemm32
python3 $J/cmds/int8_cmp_gemm.py s22_gemm_bf16 s19_gemm16
command grep -h "MULTI COPIES\|RESULT" $J/runs/i0_multi_fp32/stdout.log $J/runs/s19_multi/stdout.log
```

Expected:
- `sienna_layer (0, 7)` and `sienna_top (0, 7)`: `exit=0`, no `%Error`, `LATCH`, `MULTIDRIVEN` or `UNOPTFLAT`. The fp32
  and bf16 lints the same, with warning counts equal to the bf16 gate's (`command grep -h "warnings" $J/runs/g4lint2/stdout.log`)
  or each new warning named and explained in the commit message;
- `(0, 15)` and `(5, 10)`: `exit=1` with `sienna_top: unsupported format` (and the submodules' own messages);
- regressions 29/29, both comparisons `IDENTICAL`;
- `29 tests compared in fp32, identical`, the same in bf16 (the package gains only the two new lines);
- both GEMM comparisons `IDENTICAL` (the quick shapes are a subset of the `s22` runs' shapes);
- both multi runs `RESULT: PASSED` with the same `steady ... cycles per set`;
- `s19_model` builds.

The int8 regression, `TB_sienna_model` in int8 and `TB_sienna_layer` in int8 need Task 20's stimulus and Task 21's layer
files; they run there.

- [ ] **Step 9: Commit (SIENNA) and push**

```bash
git add src/sienna_layer.sv && git commit -m "sienna_layer: int8 parameters with the configuration, int32 bias and requantize words beside each block's bias beat"
git add src/sienna_multi.sv && git commit -m "sienna_multi: int32 bias and the int8 requantize inputs to every copy"
git add testbenches/TB_sienna_top.sv testbenches/TB_sienna_model.sv testbenches/TB_sienna_multi.sv testbenches/TB_sienna_layer.sv \
  && git commit -m "Testbenches: int32 bias words, per-set requantize files, int8 layer epilogues"
git add regression.py && git commit -m "regression: package items in one function; IS_INT and ACC_W"
git push origin int8
```

### Task 20: Bit-exact int8 golden in `regression.py` (D-6), int8 regression tests

**Files:**
- Modify: `regression.py`
- Create (no repo): `$J/cmds/int8_seed.sh`, `$J/cmds/int8_neg_channel.sh`, `$J/cmds/int8_model_build.sh`

**Interfaces:**
- Consumes: `mesh_model.matmul_int(passes, N, bias)` (Task 14); `ipu.requant`, `ipu.REQ_ROUNDING` (Tasks 1, 3);
  `tflite_ref.quantize_multiplier`, `tflite_ref.ROUNDING` (Task 2); `sienna_fmt_pkg::REQ_ROUNDING` (Task 3);
  `gpnae_model.FORMATS["int8"]`, `gpnae_model.Lane(FORMATS["int8"], rom).run(q, code, par)` (a `LaneInt8`),
  `gpnae_model.Int8Params(mx, shx, zin, mout, shout, zout)`, `rescale_params(s_in)`, `quantize_multiplier(real)`,
  `read_rom`, `coeff_file` and `poly_coeffs_int8.mem` (Task 10); Task 19's testbench file formats; the references
  `i0_sienna_fp32`, `i0_sienna_bf16` (Task 0).
- Produces, all in `regression.py`:
  - `FORMATS["int8"] = (0, 7)`; `op_round(x, "int8")` (round to the nearest integer, clamp to -128..127) and
    `op_hex(x, "int8")` (2-digit two's complement), so `write_op_mem` writes int8 operands.
  - Quantization helpers, as TFLite post-training quantization chooses parameters: `wrap32(x)`,
    `quant_act(x) -> (q, scale, zp)` (asymmetric, range widened to hold 0), `quant_weights(w) -> (q, per-column scales)`
    (symmetric per output channel, codes -127..127), `fold_bias(bias, s_a, s_w, z_a, B_q)` (TFLite's int32 bias minus
    z_a * sum_k B_q[k, c]), `requant_params(acc, s_a, s_w, act, rng=None) -> dict` (keys `mult, shift, zp, amin, amax,
    mx, shx, mout, shout, zout`, plus `s_out`, `s_selu` for dequantizing; `(mx, shx) = gpnae_model.rescale_params(s_out)`,
    `(mout, shout) = gpnae_model.quantize_multiplier(2^-25 / s_selu)`, D-4).
  - The golden pieces: `requantize(acc, rq)` (channel = column), `activate_int8(R, act, rq)`, `drop_zp(act, rq)` (D-5),
    `_maxpool_int`, `_golden_int8(passes, hw_bias, rq, cfg, act, drop_seed) -> (C, R, A, P, F)`, and
    `int8_layer_exact(A_q, B_q, hw_bias, rq, act)` (sienna_layer's int8 output for one product: no pooling or dropout).
  - `generate_vectors` with `fmt_name = "int8"` writes int8 stimulus, `bias_<k>.mem`, `requant_<k>.mem` and bit-exact
    expected files; every file's word width is checked; `_check_rounding()` stops a run unless
    `sienna_fmt_pkg::REQ_ROUNDING`, `ipu.REQ_ROUNDING` and `tflite_ref.ROUNDING` name the same variant.
  - Test keys: `formats` (a test runs only in the listed formats; default all), `a_range` (the range of A's real
    values), `req_random` (random per-channel multipliers and shifts). Four int8-only tests; int8 runs 33 tests.

How int8 stimulus is quantized. Each test draws real matrices exactly as the float tests do. Each accumulate group
(a set when `accum_passes` is 1) is one layer: its A passes side by side are one activation tensor with one scale and zero
point; its B passes stacked are one weight matrix with one scale per output column. The bias is TFLite's int32 bias
(scale s_a * s_w[c]) with the input zero-point term folded in, so the mesh's plain sum of a * w equals TFLite's sum of
(a - z_a) * w + b. The output scale and zero point come from the range of the group's real outputs (after ReLU for a
ReLU test), and M_c = s_a * s_w[c] / s_out through `QuantizeMultiplier`. ReLU is the clamp's minimum, the zero point.
The lane's input scale is s_out, so `(gp_mx, gp_shx) = gpnae_model.rescale_params(s_out)` and `gp_zin` is the requantize
zero point; SELU's output scale s_selu is calibrated from the range of SELU over the lane's inputs (Risk 4), and
`(gp_mout, gp_shout) = QuantizeMultiplier(2^-25 / s_selu)`, since the lane's SELU value is in units of 2^-25 (D-4). A partial pass carries decoy requantize parameters: the activated pass's must be the ones the hardware uses.
Zero points are rounded with `np.rint`, not the converter's nudging; that choice cannot affect exactness, since the
golden uses exactly the parameters it writes.

- [ ] **Step 1: Run int8 to see it fail**

```bash
git show HEAD:regression.py > .claude/scratch/regression_t19.py   # Task 19's copy, for Step 6
L s20_fail 32 1 $J/cmd_sienna_fmt.sh reg 16 4 int8
```

Expected: `regression.py: error: argument --format: invalid choice: 'int8'`.

- [ ] **Step 2: Format entry, operand encoding, imports**

After `from mesh_model import fpu  # noqa: E402` add

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402
import tflite_ref  # noqa: E402
```

`FORMATS = {"fp32": (8, 23), "bf16": (8, 7)}` becomes

```python
FORMATS = {"fp32": (8, 23), "bf16": (8, 7), "int8": (0, 7)}  # int8: EXP_W = 0, 8-bit two's-complement codes
```

In `op_round`, after `if fmt == "fp32": return x` add

```python
    if fmt == "int8":  # int8 codes, as float32 values holding integers
        return np.clip(np.rint(x), -128, 127).astype(np.float32)
```

In `op_hex`, after `v = op_round(...)` add

```python
    if fmt == "int8":
        return [f"{int(b) & 0xFF:02x}" for b in v.astype(np.int64)]
```

- [ ] **Step 3: Quantization, the golden, the generator**

After `_golden_bits` add:

```python
# =============================================================================
# int8 (D-6): TFLite-style quantization of the float tests' data, and the bit-exact golden
# =============================================================================

REQ_HEAD = 8  # layer-wide words heading requant_<k>.mem: zp, min, max, gp_mx, gp_shx, gp_mout, gp_shout, gp_zout


def wrap32(x) -> np.ndarray:
    """int64 values wrapped to int32, as the mesh's two's-complement accumulate does."""
    return ((np.asarray(x, dtype=np.int64) + (1 << 31)) % (1 << 32)) - (1 << 31)


def imatmul(a, b) -> np.ndarray:
    """Exact integer product: int8 x int8 terms and their sums stay below 2^53, so float64 holds every partial sum."""
    return np.rint(np.asarray(a, np.float64) @ np.asarray(b, np.float64)).astype(np.int64)


def quant_act(x) -> tuple:
    """(q, scale, zero point) of an activation tensor as TFLite post-training quantization chooses them: asymmetric int8
    over the tensor's range widened to hold 0, so real 0 is exactly the zero point."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = min(0.0, float(x.min())), max(0.0, float(x.max()))
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    zp = int(np.clip(np.rint(-128.0 - lo / scale), -128, 127))
    return np.clip(np.rint(x / scale) + zp, -128, 127).astype(np.int64), scale, zp


def quant_weights(w) -> tuple:
    """(q, per-column scales) of a weight matrix as TFLite quantizes conv and FC weights: symmetric per output channel,
    zero point 0, codes -127..127."""
    w = np.asarray(w, dtype=np.float64)
    s = np.max(np.abs(w), axis=0) / 127.0
    s = np.where(s > 0, s, 1.0)
    return np.clip(np.rint(w / s[None, :]), -127, 127).astype(np.int64), s


def fold_bias(bias, s_a: float, s_w, z_a: int, B_q) -> np.ndarray:
    """The int32 bias the mesh adds: TFLite's bias (scale s_a * s_w[c], zero point 0) minus z_a * sum_k B_q[k, c], which
    turns the mesh's sum of a * w into TFLite's sum of (a - z_a) * w."""
    B_q = np.asarray(B_q, np.int64)
    b_q = np.zeros(B_q.shape[1], np.int64) if bias is None else \
        np.rint(np.asarray(bias, np.float64) / (s_a * np.asarray(s_w, np.float64))).astype(np.int64)
    return wrap32(b_q - z_a * B_q.sum(axis=0))


def requantize(acc, rq: dict) -> np.ndarray:
    """ipu.requant of int32 sums (rows x channels) with channel c's multiplier and shift on column c."""
    acc = np.asarray(acc, np.int64)
    m = np.broadcast_to(np.asarray(rq["mult"], np.int64)[None, :acc.shape[1]], acc.shape)
    s = np.broadcast_to(np.asarray(rq["shift"], np.int64)[None, :acc.shape[1]], acc.shape)
    return np.asarray(ipu.requant(acc, m, s, rq["zp"], rq["amin"], rq["amax"], tflite_ref.ROUNDING), np.int64)


def requant_params(acc, s_a: float, s_w, act: str, rng=None) -> dict:
    """Requantize and GPNAE parameters for int32 sums acc (rows x channels), chosen as TFLite PTQ would: output scale and
    zero point from the range of the real outputs (after ReLU for a ReLU layer), M_c = s_a * s_w[c] / s_out through
    QuantizeMultiplier, a fused ReLU as the clamp's minimum. With rng the multipliers and shifts are random per channel.
    The lane's input scale is s_out (gpnae_model.rescale_params); SELU's output scale comes from the range of SELU over the
    lane's real inputs, and its multiplier from QuantizeMultiplier(2^-25 / s_selu)."""
    acc = np.asarray(acc, np.int64)
    s_w = np.asarray(s_w, np.float64)
    real = acc * (s_a * s_w)[None, :]
    if act == "relu":
        real = np.maximum(real, 0.0)
    _, s_out, z_out = quant_act(real)
    qm = [tflite_ref.quantize_multiplier(s_a * float(s) / s_out) for s in s_w]
    mult = np.array([m for m, _ in qm], np.int64)
    shift = np.array([e for _, e in qm], np.int64)
    if rng is not None:  # every normalized multiplier, and right shifts 0..12 (TFLite's left shift can overflow int32)
        mult = rng.randint(1 << 30, 1 << 31, s_w.size).astype(np.int64)
        shift = rng.randint(-12, 1, s_w.size).astype(np.int64)
    mx, shx = gpnae_model.rescale_params(s_out)
    rq = dict(mult=mult, shift=shift, zp=z_out, amin=max(-128, z_out) if act == "relu" else -128, amax=127,
              mx=mx, shx=shx, s_out=s_out)
    x = (requantize(acc, rq) - z_out) * s_out  # the lane's real inputs
    _, s_selu, z_selu = quant_act(apply_activation(np.clip(x, -16.0, 16.0).astype(np.float32), "selu"))
    mout, shout = gpnae_model.quantize_multiplier(2.0 ** -25 / s_selu)  # the lane's SELU value, in units of 2^-25, to its int8 code (D-4)
    rq.update(mout=int(mout), shout=int(shout), zout=z_selu, s_selu=s_selu)
    return rq


def int8_lane():
    f = gpnae_model.FORMATS["int8"]
    return gpnae_model.Lane(f, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f))))


def activate_int8(R, act: str, rq: dict) -> np.ndarray:
    """The lane stage in int8: ReLU and linear pass the requantized value through (the clamp already applied ReLU), the
    others run the fixed-point lane with the set's GPNAE parameters and the requantize zero point as its input's."""
    code = activation_to_code(act)
    R = np.asarray(R, np.int64)
    if code in (4, 5):
        return R.copy()
    par = gpnae_model.Int8Params(mx=rq["mx"], shx=rq["shx"], zin=rq["zp"], mout=rq["mout"], shout=rq["shout"], zout=rq["zout"])
    return np.asarray(int8_lane().run(R, code, par), np.int64)


def drop_zp(act: str, rq: dict) -> int:
    """D-5: dropout drops to the output zero point of the set's activation: SELU gp_zout, sigmoid -128, ReLU and linear the
    requantize zero point, tanh (and every other code, which the lane runs as tanh) 0; sienna_top's p_zp makes the same choice."""
    return {1: rq["zout"], 2: -128, 4: rq["zp"], 5: rq["zp"]}.get(activation_to_code(act), 0)


def _maxpool_int(x, ph: int, pw: int, pad: int) -> np.ndarray:
    """Integer max pooling with -128 outside the input, stride = window, as sienna_top dispatches it."""
    x = np.asarray(x, np.int64)
    H, W = x.shape
    xp = np.full((H + 2 * pad, W + 2 * pad), -128, dtype=np.int64)
    xp[pad:pad + H, pad:pad + W] = x
    oh, ow = (H + 2 * pad - ph) // ph + 1, (W + 2 * pad - pw) // pw + 1
    return np.array([[xp[i * ph:i * ph + ph, j * pw:j * pw + pw].max() for j in range(ow)] for i in range(oh)], np.int64)


def _golden_int8(passes, hw_bias, rq: dict, cfg: dict, act: str, drop_seed: int) -> tuple:
    """Bit-exact int8 set (D-6): the mesh's int32 sums, requantize per column, the lane, integer max pooling, dropout with
    dropped values at the output zero point of the set's activation (D-5); passes are (A codes, B codes) in order."""
    N = cfg.get("n", 16)
    C = mesh_model.matmul_int([(_pad_square(a, N), _pad_square(b, N)) for a, b in passes], N, hw_bias)
    R = requantize(C, rq)
    A = activate_int8(R, act, rq)
    P = _maxpool_int(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1))
    if not cfg.get("training", False):
        return C, R, A, P, P.copy()
    keep = dropout_keep(P.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    return C, R, A, P, np.where(keep, P.flatten(), drop_zp(act, rq)).reshape(P.shape)


def int8_layer_exact(A_q, B_q, hw_bias, rq: dict, act: str) -> np.ndarray:
    """sienna_layer's int8 output for one product: int32 sums (integer addition is associative, so tiles and depth
    passes do not change them), requantize per output column, the lane; the layer engine neither pools nor drops out."""
    acc = wrap32(imatmul(A_q, B_q) + np.asarray(hw_bias, np.int64)[None, :])
    return activate_int8(requantize(acc, rq), act, rq)


def _check_rounding() -> None:
    """sienna_fmt_pkg::REQ_ROUNDING (every tfliteRequant instance) must be the variant G0 pinned, as ipu and tflite_ref read it,
    or golden and RTL round apart in the last bit."""
    pkg = os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "src", "sienna_fmt_pkg.sv")
    m = re.search(r'localparam string REQ_ROUNDING\s*=\s*"(\w+)"', open(pkg).read())
    rtl = m.group(1) if m else "(no REQ_ROUNDING)"
    if tflite_ref.ROUNDING is None or rtl != tflite_ref.ROUNDING or ipu.REQ_ROUNDING != tflite_ref.ROUNDING:
        raise ValueError(f"sienna_fmt_pkg rounds {rtl}, ipu.REQ_ROUNDING is {ipu.REQ_ROUNDING}, "
                         f"tflite_ref.ROUNDING (rounding.txt) is {tflite_ref.ROUNDING}")


def _decoy_params(N: int) -> dict:
    """A partial pass's requantize words: the activated pass's are the ones used, so these must never show in a result."""
    return dict(mult=np.full(N, 1 << 30, np.int64), shift=np.full(N, -1, np.int64), zp=5, amin=-100, amax=100,
                mx=1, shx=0, mout=1 << 30, shout=-1, zout=5)


def _write_s8(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFF:02x}\n" for x in np.asarray(v).flatten()))


def _write_w32(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFFFFFFFF:08x}\n" for x in np.asarray(v).flatten()))


def _requant_words(rq: dict) -> list:
    """requant_<k>.mem's words: the REQ_HEAD layer-wide words, then N multipliers, then N shifts."""
    return [rq["zp"], rq["amin"], rq["amax"], rq["mx"], rq["shx"], rq["mout"], rq["shout"], rq["zout"]] + \
        [int(v) for v in rq["mult"]] + [int(v) for v in rq["shift"]]


def _generate_vectors_int8(cfg: dict) -> None:
    """int8 stimulus and bit-exact golden: the float tests' real matrices, quantized per accumulate group as TFLite PTQ
    would, with each set's requantize and GPNAE parameters in requant_<k>.mem."""
    os.makedirs(TB_DIR, exist_ok=True)
    _check_rounding()
    N, mode = cfg.get("n", 16), cfg.get("mode", "matmul")
    act_type = cfg.get("activation", cfg.get("act", "idle"))
    seed = cfg.get("seed", 42) + int(os.environ.get("SIENNA_SEED", "0"))  # unset keeps the fixed stimulus
    test_name = cfg.get("name", "manual_gen")
    lo, hi = cfg.get("a_range", (-1.0, 1.0))  # a range not centred on 0 gives a non-zero input zero point
    scale = float(cfg.get("scale", 1.0))
    if mode == "conv":
        A0, B0 = build_conv_matrices(N, cfg.get("conv_type", "basic"), seed, cfg.get("conv_stride"))
    else:
        np.random.seed(seed)
        m_type = cfg.get("matrix_type", "random")
        if m_type == "identity":
            A0 = B0 = np.eye(N)
        elif m_type == "ones":
            A0 = B0 = np.ones((N, N))
        elif m_type == "small_exact":
            A0 = B0 = np.random.randint(-3, 4, (N, N))
        else:
            A0, B0 = np.random.uniform(lo, hi, (N, N)), np.random.uniform(-1.0, 1.0, (N, N))
    A0, B0 = np.array(A0, np.float64), np.array(B0, np.float64)  # copies: identity and ones share one array
    if cfg.get("zero_rows"):
        A0[0::4, :] = 0.0
        A0[1::4, :] = 0.0
    drop_seed = 0x2ACE0000 + seed
    passes = cfg.get("accum_passes", 1)
    mixed = cfg.get("mixed_acts", [])
    assert not mixed or mixed[0] == act_type, (test_name, "mixed_acts[0] must be the test's act")
    credits = cfg.get("credits", SETS_IN_FLIGHT)
    num_sets = cfg.get("num_sets", len(mixed) or -(-(credits + 2) // passes) * passes)
    use_bias = bool(cfg.get("bias", False))
    req_rng = np.random.RandomState(seed + 7000) if cfg.get("req_random") else None
    reals = [(A0 * scale, B0 * scale)]
    for k in range(1, num_sets):
        rng = np.random.RandomState(seed + 1000 + k)
        reals.append((rng.uniform(lo, hi, (N, N)) * scale, rng.uniform(-1.0, 1.0, (N, N)) * scale))
    first = None
    for g0 in range(0, num_sets, passes):
        ks = list(range(g0, min(g0 + passes, num_sets)))
        act_g = mixed[ks[-1] % len(mixed)] if mixed else act_type  # the activated pass's activation
        A_q, s_a, z_a = quant_act(np.hstack([reals[k][0] for k in ks]))
        B_q, s_w = quant_weights(np.vstack([reals[k][1] for k in ks]))
        bias_real = np.random.RandomState(seed + 5000 + g0).uniform(-1.0, 1.0, N) * scale if use_bias else None
        hw_bias = fold_bias(bias_real, s_a, s_w, z_a, B_q)
        parts = [(A_q[:, i * N:(i + 1) * N], B_q[i * N:(i + 1) * N, :]) for i in range(len(ks))]
        rq = requant_params(wrap32(sum(imatmul(a, b) for a, b in parts) + hw_bias[None, :]), s_a, s_w, act_g, req_rng)
        complete = len(ks) == passes  # a trailing short group has only partial passes, as in the float tests
        for i, k in enumerate(ks):
            last = complete and i == len(ks) - 1
            rq_k = rq if last else _decoy_params(N)
            write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), parts[i][0], "int8")
            write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), parts[i][1], "int8")
            _write_w32(os.path.join(TB_DIR, f"bias_{k}.mem"), hw_bias if i == 0 else np.zeros(N, np.int64))
            _write_w32(os.path.join(TB_DIR, f"requant_{k}.mem"), _requant_words(rq_k))
            F = _golden_int8(parts, hw_bias, rq, cfg, act_g, set_dropout_seed(drop_seed, k))[4] if last \
                else np.zeros(0, np.int64)
            _write_s8(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F)  # empty for a partial set
            _write_s8(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(F))
            if k == 0:
                first = (parts[0][0], parts[0][1], hw_bias, rq_k)
    # The single-set pass starts set 0 alone, not partial, with set 0's bias and requantize words.
    a0, b0, hb0, rq0 = first
    C0, _, A0q, P0, F0 = _golden_int8([(a0, b0)], hb0, rq0, cfg, act_type, drop_seed)
    write_op_mem(os.path.join(TB_DIR, "matrix_west.mem"), a0, "int8")
    write_op_mem(os.path.join(TB_DIR, "matrix_north.mem"), b0, "int8")
    _write_s8(os.path.join(TB_DIR, "expected_output.mem"), F0)
    _write_s8(os.path.join(TB_DIR, "bound_output.mem"), np.zeros_like(F0))
    dump_golden_trace(test_name, C0.astype(np.float32), A0q.astype(np.float32), P0.astype(np.float32))
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),
                     _config_items(cfg, "int8", act_type, num_sets, credits, passes, mixed, True, drop_seed))
    _check_mem_widths("int8", num_sets)
```

`HAS_BIAS` is 1 in every int8 test: the bias carries the folded zero point even when the test has no bias of its own.
The dropout-mask independence check of the float path is not repeated: int8 uses the same `dropout_keep`.

At the top of `generate_vectors`, before `os.makedirs(TB_DIR, exist_ok=True)`, add

```python
    if cfg.get("fmt_name", "fp32") == "int8":
        return _generate_vectors_int8(cfg)
```

`_check_mem_widths` becomes

```python
def _check_mem_widths(fmt: str, num_sets: int) -> None:
    """Every operand, bias and expected word the TB reads must be the format's width: a stale fp32 file would be truncated.
    int8: operands and results 2 digits, the int32 bias and the requantize words 8."""
    d = 2 if fmt == "int8" else (fpu.FORMATS[fmt].w + 3) // 4
    per_set = ("matrix_west", "matrix_north", "bias", "expected_output", "bound_output") + (("requant",) if fmt == "int8" else ())
    names = [f"{b}.mem" for b in ("matrix_west", "matrix_north", "expected_output", "bound_output")]
    names += [f"{b}_{k}.mem" for k in range(num_sets) for b in per_set]
    for fn in names:
        want = 8 if fmt == "int8" and fn.startswith(("bias_", "requant_")) else d
        if os.path.exists(os.path.join(TB_DIR, fn)):
            for i, ln in enumerate(open(os.path.join(TB_DIR, fn))):
                if ln.strip() and len(ln.strip()) != want:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {want} hex digits ({fmt})")
```

(bf16 checks the same files as before: `requant_<k>.mem` is read only in int8, so a leftover one is not checked there.)

- [ ] **Step 4: Tests and the per-format selection**

At the end of `PIPELINE_TESTS` add

```python
    # int8 only: per-channel random requantize multipliers and shifts, and inputs whose zero point is far from 0.
    {"name": "int8_perchannel_random_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear",
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "formats": ("int8",)},
    {"name": "int8_input_zp_bias_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "bias": True,
     "a_range": (-0.25, 1.0), "pool_h": 1, "pool_w": 1, "padding": 0, "formats": ("int8",)},
    {"name": "int8_input_zp_perchannel_accum3_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh",
     "bias": True, "a_range": (0.0, 1.0), "req_random": True, "accum_passes": 3, "formats": ("int8",)},
    {"name": "int8_input_zp_selu_train", "mode": "matmul", "matrix_type": "random", "act": "selu",
     "a_range": (-0.1, 1.0), "training": True, "formats": ("int8",)},
```

(input zero points near -77, -128 and -105; the exact values follow each set's data range). In `run_regression`,

```python
    tests_to_run = PIPELINE_TESTS

    if target_test:
        tests_to_run = [t for t in PIPELINE_TESTS if target_test in t["name"]]
```

becomes

```python
    tests_to_run = [t for t in PIPELINE_TESTS if fmt_name in t.get("formats", tuple(FORMATS))]  # int8-only tests skip the floats

    if target_test:
        tests_to_run = [t for t in tests_to_run if target_test in t["name"]]
```

- [ ] **Step 5: Job scripts**

`$J/cmds/int8_seed.sh` (`chmod +x`):

```bash
#!/bin/bash
# cmd_sienna_fmt.sh with every stimulus seed shifted; args: SEED then cmd_sienna_fmt.sh's; run from a snapshot root.
export SIENNA_SEED=$1; shift
exec /proj/work/spramanik/sienna_jobs/cmd_sienna_fmt.sh "$@"
```

`$J/cmds/int8_neg_channel.sh` (`chmod +x`):

```bash
#!/bin/bash
# Negative control: requant_lanes with every lane one channel off must fail the per-channel int8 test; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
python3 - <<'EOF'
p = "src/requant_lanes.sv"
s = open(p).read()
old = "(k * PER_LANE + int'(beat)) % N"
assert old in s, "channel map not found"
open(p, "w").write(s.replace(old, "(k * PER_LANE + int'(beat) + 1) % N"))
EOF
python3 regression.py --n 16 --tile-size 4 --format int8 --test int8_perchannel_random_linear_nopool
```

`$J/cmds/int8_model_build.sh` (`chmod +x`):

```bash
#!/bin/bash
# TB_sienna_model built and started in int8 on an int8 package; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
python3 -c "
import regression as r
t = next(x for x in r.PIPELINE_TESTS if x['name'] == 'matmul_relu_nopool')
r.generate_vectors({'n': 16, 'tile_size': 4, 'lanes': 32, 'host_words': 16, 'fmt_name': 'int8', **t})" || exit 1
make verilator TOP_MODULE=TB_sienna_model TESTBENCH=TB_sienna_model.sv TRACE=0
```

- [ ] **Step 6: Run int8, the controls, and fp32 / bf16 unchanged**

```bash
L s20_int8 32 6 $J/cmd_sienna_fmt.sh reg 16 4 int8
L s20_int8s3 32 6 $J/cmds/int8_seed.sh 3 reg 16 4 int8
L s20_neg 32 2 $J/cmds/int8_neg_channel.sh
L s20_model 32 2 $J/cmds/int8_model_build.sh
L s20_fp32 32 4 $J/cmd_sienna_fmt.sh reg 16 4 fp32
L s20_bf16 32 4 $J/cmd_sienna_fmt.sh reg 16 4 bf16
L s20_stim32 32 2 $J/cmds/int8_stim_same.sh .claude/scratch/regression_t19.py fp32
L s20_stim16 32 2 $J/cmds/int8_stim_same.sh .claude/scratch/regression_t19.py bf16
python3 $J/cmds/int8_cmp_reg.py i0_sienna_fp32 s20_fp32     # login node, after the runs
python3 $J/cmds/int8_cmp_reg.py i0_sienna_bf16 s20_bf16
```

Expected:
- `s20_int8` and `s20_int8s3`: `Passed : 33 / 33`, and in every test log `Exact` equals `Total` (`Tol pass : 0`); if a
  test fails, read its stage dumps (`results/pipeline/<test>_expected_flow.txt` against the hardware trace) before
  changing anything, since repeated data can hide stale state;
- `s20_neg`: exits 1 with `[FAIL]` lines, which shows the per-channel test sees a channel error;
- `s20_model`: builds and prints `[MODEL] no +sets= and +out= given, nothing to run`;
- fp32 and bf16 29/29 with both comparisons `IDENTICAL`; both stimulus checks `29 tests compared ..., identical`.

At N = 32 and for collapse-k 0 the int8 regression runs in G4 (Task 23).

- [ ] **Step 7: Commit (SIENNA) and push**

```bash
git add regression.py && git commit -m "regression: bit-exact int8 golden (D-6), TFLite-style quantization of the stimulus, four int8 tests"
git push origin int8
```

### Task 21: Single-layer TFLite int8 models through the RTL, bit-exact against the interpreter

**Files:**
- Create: `tflite_int8_run.py`
- Modify: `model_runner.py` (int8 on the layer engine: `im2col` padding value, `format_layer`, `layer_epilogue`,
  `read_outputs`, `LayerSim.run_job`, a guard in `main`)
- Create (no repo): `$J/cmds/int8_tflite.sh`

**Interfaces:**
- Consumes: Task 2's models, `testbenches/tflite_int8/<name>.tflite` (one `CONV_2D` or `FULLY_CONNECTED` with int8 input,
  filter and output; `conv3x3_8x8x16_*` are SAME-padded per-channel convs with a non-zero input zero point) each with
  `<name>.npz` in Task 2's keys, of which this task reads `x_test` (the 64 saved int8 inputs, NHWC for conv, `[64, 64]`
  for FC) and `y_test` (the `BUILTIN_REF` interpreter's int8 outputs for them); `tflite_ref.quantize_multiplier`;
  Task 19's layer file format; Task 20's `wrap32`, `op_hex(…, "int8")`, `int8_layer_exact`; the `tflite` schema package
  in `$J/venv` (Task 0).
- Produces:
  - `model_runner.im2col(x, kh, kw, stride, same, channel_major=False, pad_value=0.0)`.
  - An int8 job for `LayerSim(…, fmt_name="int8")`: `{"terms": [(X, W)], "bias": int64 folded bias, "act": "linear",
    "shape": …, "req": dict(mult, shift, zp, amin, amax, mx, shx, mout, shout, zout)}`, with X and W float32 arrays
    holding int8 codes. `run_job` returns int64 codes.
  - `model_runner.layer_epilogue(job, N) -> int64 array (column blocks, 3, N)`: bias, multipliers, shifts per block.
  - `tflite_int8_run.py [--n] [--tile-size] [--lanes] [--models DIR] [--work DIR]`: one `MODEL <name>: ...` line per
    model, `TFLITE_INT8: <k> models, <b> failing` and `RESULT: PASSED` or `FAILED`, in
    `testbenches/results/int8/tflite_int8_N<n>_T<t>.log`; exit 0 only if every output of every model equals the
    interpreter's and one model is a SAME-padded per-channel conv with a non-zero input zero point.
  - `$J/cmds/int8_tflite.sh N T LANES`.

The host lowers a layer exactly as TFLite's reference kernels compute it. The operands are the raw int8 codes. SAME
padding pads the im2col with the input zero point, which is real 0, so a padded tap adds (z_in - z_in) * w = 0 as
TFLite's skipped tap does. The term -z_in * sum_k w[k, c] is folded into the int32 bias. Channel c's multiplier comes
from `QuantizeMultiplier(double(s_in) * double(s_w[c]) / double(s_out))`. The fused activation becomes the clamp through
TFLite's `CalculateActivationRangeQuantized`, and the lane stage runs linear, which passes int8 through. The mesh wraps
at 32 bits and TFLite's int32 accumulate does not overflow on these layers, so both give the same sum.

- [ ] **Step 1: Write the runner and see it fail**

`tflite_int8_run.py`:

```python
#!/usr/bin/env python3
"""Runs the single-layer TFLite int8 models of testbenches/tflite_int8/ through sienna_layer built in int8 and compares
every output with the TFLite interpreter's (reference kernels, saved by tflite_oracle.py), bit for bit. The host lowers
each layer with TFLite's own algebra: raw int8 codes, SAME padding with the input zero point, the zero-point term folded
into the int32 bias, TFLite's per-channel multipliers, and the fused activation as the requantize clamp."""
import argparse
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_runner as mr  # noqa: E402
import tflite_ref  # noqa: E402

reg = mr.regression
MODEL_DIR = os.path.join(ROOT, "testbenches", "tflite_int8")


def round_away(x) -> int:
    """TfLiteRound (std::round): halves away from zero."""
    return int(np.sign(x) * np.floor(abs(float(x)) + 0.5))


def load_layer(path: str) -> dict:
    """The one CONV_2D or FULLY_CONNECTED operator of an int8 .tflite, with its tensors' codes and quantization."""
    import tflite
    from tflite.ActivationFunctionType import ActivationFunctionType as AF
    from tflite.BuiltinOperator import BuiltinOperator as BO

    names = {v: k for k, v in BO.__dict__.items() if not k.startswith("_")}
    m = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = m.Subgraphs(0)
    code = lambda op: m.OperatorCodes(op.OpcodeIndex())
    kinds = [names[max(code(g.Operators(i)).BuiltinCode(), code(g.Operators(i)).DeprecatedBuiltinCode())]
             for i in range(g.OperatorsLength())]
    if kinds not in (["CONV_2D"], ["FULLY_CONNECTED"]):
        raise ValueError(f"{path}: operators {kinds}; expected one CONV_2D or FULLY_CONNECTED with int8 input and output")
    op = g.Operators(0)

    def tensor(i):
        if i < 0:
            return None
        t = g.Tensors(i)
        q = t.Quantization()
        buf = m.Buffers(t.Buffer()).DataAsNumpy()
        data = None
        if not isinstance(buf, int) and buf is not None and len(buf):
            data = np.frombuffer(buf.tobytes(), dtype={9: np.int8, 2: np.int32}[t.Type()]).reshape(tuple(t.ShapeAsNumpy()))
        return dict(type=t.Type(), shape=tuple(t.ShapeAsNumpy()), data=data,
                    scale=np.atleast_1d(q.ScaleAsNumpy()).astype(np.float32),
                    zp=np.atleast_1d(q.ZeroPointAsNumpy()).astype(np.int64))

    ins = [op.Inputs(j) for j in range(op.InputsLength())]
    inp, flt = tensor(ins[0]), tensor(ins[1])
    bias = tensor(ins[2]) if len(ins) > 2 else None
    out = tensor(op.Outputs(0))
    if inp["type"] != 9 or flt["type"] != 9 or out["type"] != 9:  # 9 is INT8
        raise ValueError(f"{path}: input, filter and output must be int8")
    if np.any(flt["zp"] != 0):
        raise ValueError(f"{path}: TFLite's int8 filters are symmetric; a non-zero filter zero point is not supported")
    opt = (tflite.Conv2DOptions if kinds[0] == "CONV_2D" else tflite.FullyConnectedOptions)()
    t = op.BuiltinOptions()
    opt.Init(t.Bytes, t.Pos)
    d = dict(kind=kinds[0], input=inp, filter=flt, bias=bias, output=out)
    if kinds[0] == "CONV_2D":
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
            raise ValueError(f"{path}: dilation is not supported")
        d.update(same=opt.Padding() == 0, stride=(opt.StrideH(), opt.StrideW()))
    s_out, z_out = out["scale"][0], int(out["zp"][0])
    qz = lambda f: z_out + round_away(np.float32(f) / s_out)  # CalculateActivationRangeQuantized, float32 division
    fa = opt.FusedActivationFunction()
    ranges = {AF.NONE: (-128, 127), AF.RELU: (max(-128, qz(0.0)), 127),
              AF.RELU6: (max(-128, qz(0.0)), min(127, qz(6.0))), AF.RELU_N1_TO_1: (max(-128, qz(-1.0)), min(127, qz(1.0)))}
    if fa not in ranges:
        raise ValueError(f"{path}: fused activation {fa} is not supported")
    d["act_range"] = ranges[fa]
    return d


def job_of(layer: dict, x: np.ndarray) -> tuple:
    """(int8 job for LayerSim, output shape) of one layer on the interpreter's input codes x."""
    inp, flt, b, out = layer["input"], layer["filter"], layer["bias"], layer["output"]
    z_in, s_in = int(inp["zp"][0]), float(inp["scale"][0])
    s_out, z_out = float(out["scale"][0]), int(out["zp"][0])
    w = flt["data"].astype(np.int64)
    cout = w.shape[0]
    s_w = flt["scale"].astype(np.float64) if flt["scale"].size > 1 else np.full(cout, float(flt["scale"][0]))
    if layer["kind"] == "CONV_2D":
        cols = [mr.im2col(np.asarray(xi, np.float32), w.shape[1], w.shape[2], layer["stride"], layer["same"],
                          pad_value=float(z_in)) for xi in x]  # every saved input image; rows in (image, y, x) order
        X, (oh, ow) = np.vstack([c for c, _ in cols]), cols[0][1]
        W = w.reshape(cout, -1).T  # OHWI filters: depth in (ky, kx, c) order, as im2col lays it out
        shape = (len(x), oh, ow, cout)
    else:
        W = w.T
        X = np.asarray(x, np.float32).reshape(-1, W.shape[0])
        shape = (X.shape[0], cout)
    bq = np.zeros(cout, np.int64) if b is None or b["data"] is None else b["data"].astype(np.int64)
    hw_bias = reg.wrap32(bq - z_in * W.sum(axis=0))
    qm = [tflite_ref.quantize_multiplier(s_in * float(s) / s_out) for s in s_w]
    amin, amax = layer["act_range"]
    req = dict(mult=np.array([m for m, _ in qm], np.int64), shift=np.array([e for _, e in qm], np.int64), zp=z_out,
               amin=amin, amax=amax, mx=0, shx=0, mout=0, shout=0, zout=0)
    job = {"terms": [(X.astype(np.float32), W.astype(np.float32))], "bias": hw_bias, "act": "linear", "shape": shape,
           "req": req}
    return job, shape


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=MODEL_DIR, help="<name>.tflite files, each with <name>.npz holding x_test and y_test")
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "int8"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rep = open(os.path.join(a.work, f"tflite_int8_N{a.n}_T{a.tile_size}.log"), "w")

    def log(s):
        print(s, flush=True)
        rep.write(s + "\n")
        rep.flush()

    paths = sorted(glob.glob(os.path.join(a.models, "*.tflite")))
    if not paths:
        log(f"no .tflite models in {a.models}")
        sys.exit(1)
    sim = mr.LayerSim(a.n, a.lanes, a.work, "int8", a.tile_size)
    t0 = time.time()
    sim.build()
    log(f"TB_sienna_layer built in int8, N={a.n} T={a.tile_size} lanes={a.lanes}, in {time.time() - t0:.0f} s")
    bad, covered = 0, False
    for path in paths:
        name = os.path.basename(path)[:-len(".tflite")]
        ref = np.load(path[:-len(".tflite")] + ".npz")
        x, y = ref["x_test"], ref["y_test"].astype(np.int64)
        layer = load_layer(path)
        job, shape = job_of(layer, x)
        X, W = job["terms"][0]
        low = reg.int8_layer_exact(X.astype(np.int64), W.astype(np.int64), job["bias"], job["req"], "linear").reshape(shape)
        got, sets, cyc = sim.run_job(job, name)
        got = got.reshape(shape)
        want = y.reshape(shape)
        per_channel = layer["filter"]["scale"].size > 1
        z_in = int(layer["input"]["zp"][0])
        padded = layer["kind"] == "CONV_2D" and layer["same"]
        covered |= padded and per_channel and z_in != 0
        m_rtl, m_low = int(np.sum(got != want)), int(np.sum(low != want))
        bad += int(m_rtl != 0)
        log(f"MODEL {name}: {layer['kind']} out {shape} z_in {z_in} {'per-channel' if per_channel else 'per-tensor'} "
            f"{'SAME-padded' if padded else 'unpadded'} clamp {layer['act_range']}: RTL {m_rtl}/{want.size} differ from the "
            f"interpreter, host lowering {m_low}/{want.size}; {sets} sets, {cyc} cycles")
    if not covered:
        log("no SAME-padded per-channel conv with a non-zero input zero point among the models")
        bad += 1
    log(f"TFLITE_INT8: {len(paths)} models, {bad} failing")
    log("RESULT: PASSED" if bad == 0 else "RESULT: FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

The `host lowering` count checks the lowering with the numpy model alone, so an RTL difference and a lowering
difference can be told apart.

`$J/cmds/int8_tflite.sh` (`chmod +x`):

```bash
#!/bin/bash
# The single-layer TFLite int8 models through sienna_layer, bit for bit against the interpreter; args: N T LANES; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
[ $1 -ge 32 ] && export EXTRA_FLAGS="$EXTRA_FLAGS --output-split 20000 --output-split-cfuncs 20000 --output-groups 64"
/proj/work/spramanik/sienna_jobs/venv/bin/python tflite_int8_run.py --n $1 --tile-size $2 --lanes $3
```

```bash
L s21_fail 32 4 $J/cmds/int8_tflite.sh 16 4 32
```

Expected: a failure before any comparison, either `TypeError: im2col() got an unexpected keyword argument 'pad_value'`
(a conv model first) or `RuntimeError: <name>: layer simulation failed` after `[FATAL] ... an int8 layer file needs its
Q line` (a dense model first).

- [ ] **Step 2: int8 on the layer engine in `model_runner.py`**

1. `im2col`:

```python
def im2col(x, kh, kw, stride, same, channel_major=False, pad_value=0.0):
    """x is H x W x C; rows are output pixels, depth is (ky, kx, c), or (c, ky, kx) when channel_major. SAME padding takes
    pad_value: 0 for floats, the input zero point (real 0) for int8 codes."""
```

and its `xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))` becomes
`xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=pad_value)`.

2. `format_layer`: after the docstring's last line add `A job with "req" is int8: every column block gets its bias beat,
whose side words layer_epilogue gives; the beat's weight row is zeros.` to the docstring, and

```python
    bias = job["bias"] if job["bias"] is not None and np.any(job["bias"]) else None
```

becomes

```python
    int8 = job.get("req") is not None  # int8: every block has its bias beat, and the int32 bias rides beside it
    bias = job["bias"] if job["bias"] is not None and (int8 or np.any(job["bias"])) else None
```

`if bias is not None:` before `bp[: bias.size] = bias` becomes `if bias is not None and not int8:`, and

```python
        if bias is not None:
            w_rows.append(bp[c * N : (c + 1) * N][None, :])
```

becomes

```python
        if bias is not None:
            w_rows.append(np.zeros((1, N), np.float32) if int8 else bp[c * N : (c + 1) * N][None, :])
```

The return value keeps its four fields (`layer_stalls.py` unpacks them).

3. After `format_layer` add

```python
def layer_epilogue(job, N):
    """int8: per column block, the words beside its bias beat: N int32 biases, N multipliers, N shifts, zero past the
    layer's columns; an int64 array of shape (column blocks, 3, N)."""
    q, b = job["req"], np.asarray(job["bias"], np.int64)
    C = b.size
    ct = -(-C // N)
    pad = lambda v: np.concatenate([np.asarray(v, np.int64), np.zeros(ct * N - C, np.int64)])
    bb, mm, ss = pad(b), pad(q["mult"]), pad(q["shift"])
    return np.stack([np.stack([v[c * N:(c + 1) * N] for v in (bb, mm, ss)]) for c in range(ct)])
```

4. `read_outputs`: at the start of the function add

```python
    if fmt == "int8":  # two's-complement codes
        sets, cur = [], None
        for line in open(path):
            if line.startswith("S "):
                cur = []
                sets.append(cur)
            else:
                cur.append(int(line, 16))
        return [((np.array(s, np.int64) + 128) % 256) - 128 for s in sets]
```

and its docstring ends `; int8 as signed integer codes.`

5. `LayerSim.run_job`:

```python
    def run_job(self, job, tag):
        N = self.N
        int8 = self.fmt_name == "int8"
        if int8 and job.get("req") is None:
            raise ValueError(f"{tag}: an int8 job needs its requantize parameters (job['req'])")
        cfg, a, w, (M, C, rt, ct) = format_layer(job, N)
        lf = os.path.join(self.work, f"{tag}.layer")
        of = os.path.join(self.work, f"{tag}.out")
        with open(lf, "w") as f:
            f.write(f"L {cfg['m']} {cfg['kb']} {cfg['n']} {cfg['residual']} {cfg['bias']} {cfg['act']} 0 0 {len(a)} {len(w)}\n")
            if int8:
                q = job["req"]
                f.write(f"Q {q['zp']} {q['amin']} {q['amax']} {q['mx']} {q['shx']} {q['mout']} {q['shout']} {q['zout']}\n")
            f.write("\n".join(regression.op_hex(np.concatenate([a.ravel(), w.ravel()]), self.fmt_name)))
            f.write("\n")
            if int8:
                f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in layer_epilogue(job, N).ravel()))
        r = subprocess.run([self.bin, f"+layer={lf}", f"+out={of}"], cwd=os.path.dirname(self.bin), capture_output=True, text=True)
        m = re.search(r"\[LAYER\] sets=(\d+) outputs=(\d+) cycles=(\d+) a_rows=(\d+)/(\d+) w_rows=(\d+)/(\d+)", r.stdout)
        e = re.search(r"epilogues=(\d+)/(\d+)", r.stdout)
        if not m or m.group(4) != m.group(5) or m.group(6) != m.group(7) or (int8 and (not e or e.group(1) != e.group(2))):
            sys.stdout.write(r.stdout[-3000:])
            raise RuntimeError(f"{tag}: layer simulation failed")
        outs = read_outputs(of, self.fmt_name)
        os.remove(lf)
        os.remove(of)
        if len(outs) != rt * ct:
            raise RuntimeError(f"{tag}: {len(outs)} output tiles, expected {rt * ct}")
        Y = np.zeros((rt * N, ct * N), np.int64 if int8 else np.float32)
```

and the rest of the method is unchanged. (The `os.remove` calls delete the job's own temporary layer and output files, as
today.)

6. In `main`, after `a = ap.parse_args()`:

```python
    if a.fmt_name == "int8":
        ap.error("the int8 MLPerf models are sub-project 2b; single-layer TFLite int8 models run with tflite_int8_run.py")
```

- [ ] **Step 3: Run the models, and fp32 / bf16 unchanged**

```bash
L s21_tfl 32 4 $J/cmds/int8_tflite.sh 16 4 32
L s21_gemm32 64 6 $J/venv/bin/python gemm_sweep.py --n 16 --format fp32 --quick
L s21_gemm16 64 6 $J/venv/bin/python gemm_sweep.py --n 16 --format bf16 --quick
python3 $J/cmds/int8_cmp_gemm.py s22_gemm_fp32 s21_gemm32   # login node, after the runs
python3 $J/cmds/int8_cmp_gemm.py s22_gemm_bf16 s21_gemm16
```

Expected:
- one `MODEL` line per model with `RTL 0/<n> differ from the interpreter, host lowering 0/<n>`, at least one of them
  `CONV_2D ... z_in <non-zero> per-channel SAME-padded`, then `TFLITE_INT8: <k> models, 0 failing` and `RESULT: PASSED`;
- both GEMM comparisons `IDENTICAL`.

If the RTL differs and the host lowering does not, the fault is in the hardware or its golden (compare against Task 20's
regression first). If both differ, the lowering or `tflite_ref` disagrees with the interpreter: check against G0 before
touching the RTL.

- [ ] **Step 4: Commit (SIENNA) and push**

```bash
git add model_runner.py && git commit -m "model_runner: int8 layers on sienna_layer (zero-point padding, epilogue words, int8 outputs)"
git add tflite_int8_run.py && git commit -m "tflite_int8_run: single-layer TFLite int8 models through the RTL, bit for bit against the interpreter"
git push origin int8
```

### Task 22: `perf_analysis`, `gemm_sweep` and the report in int8

**Files:**
- Modify: `perf_analysis.py`, `gemm_sweep.py`
- Modify (no repo): `/proj/work/spramanik/sienna_jobs/report/build_report.py`
- Create (no repo): `$J/cmds/int8_cmp_perf.py`

**Interfaces:**
- Consumes: Task 20's quantization helpers and `int8_layer_exact`; Task 21's int8 `LayerSim`; `gpnae_model.SETS_INT8`
  (Task 10, the int8 degrees); barrel_mac's int8 Horner loop, `fx_lat() + 1` = 3 cycles (Task 9); Task 12's
  `testbenches/int8/gpnae_int8_accuracy.json`.
- Produces:
  - `perf_analysis.py --format int8`: `UNIT_LAT["int8"] = (1, 1)`, U = min(K, adder latency + 1) (2 in int8, 6 as
    before in fp32 and bf16), the activation model with the 3-cycle requantize stage, the 3-cycle int8 Horner loop (fxMac
    behind barrel_mac's register stage) and Task 10's degrees; int8 reports count OP and TOPS (2 x MACs per second at the
    assumed clock) where the floats count FLOP and GFLOPS.
    The JSON row layout is unchanged, so `g4_table.py` and the report read it as before. Tests restricted by `formats`
    are dropped from `--configs` in other formats.
  - `gemm_sweep.py --format int8` (layer engine only): every output bit-exact against `int8_layer_exact`, the
    dequantized output's error against float64 reported (`q err`), JSON rows gain `act` and `mism`.
  - `build_report.py`: section 4 fills its int8 columns (latency, cycles per set, TOPS, speedup over fp32) from the runs
    `g8p_N*_T*_int8`, the int8 GEMM accuracy from `g8_gemm_int8`, and the int8 activation accuracy (`INT8_ACT`) from
    Task 12's `gpnae_int8_accuracy.json`; "pending" where a run or the file is missing.
  - `$J/cmds/int8_cmp_perf.py REF_RUN NEW_RUN N T FMT`: latency and cycles per set per config; importable `compare()`.

- [ ] **Step 1: See both scripts fail in int8**

```bash
L s22_pfail 32 1 python3 perf_analysis.py --n 16 --tile-size 4 --lanes 32 --format int8 --configs matmul_relu_nopool
L s22_gfail 64 2 $J/venv/bin/python gemm_sweep.py --n 16 --format int8 --quick
```

Expected: `argument --format: invalid choice: 'int8'` from `perf_analysis.py`; `ValueError: ... an int8 job needs its
requantize parameters` from `gemm_sweep.py`.

- [ ] **Step 2: `perf_analysis.py`**

1. Replace

```python
DEGREE = {"selu": 8, "sigmoid": 6, "tanh": 8}  # gpnae_poly coefficient table
MUL_LAT, ADD_LAT = 8, 5  # valid in to done out: fp32Multiplier and fp32Adder; main() sets the build's format's values
UNIT_LAT = {"fp32": (8, 5), "bf16": (3, 5)}  # sienna_fmt_pkg::mul_lat, add_lat
```

with

```python
DEGREE = {"selu": 8, "sigmoid": 6, "tanh": 8}  # gpnae_poly coefficient table in fp32 and bf16; int8 takes gpnae_model.SETS_INT8's
MUL_LAT, ADD_LAT = 8, 5  # valid in to done out: fp32Multiplier and fp32Adder; main() sets the build's format's values
UNIT_LAT = {"fp32": (8, 5), "bf16": (3, 5), "int8": (1, 1)}  # sienna_fmt_pkg::mul_lat, add_lat
MAC_LAT = {"fp32": 13, "bf16": 8, "int8": 3}  # barrel_mac's Horner loop: multiplier then adder, or fxMac behind a register stage
REQ_LAT = {"fp32": 0, "bf16": 0, "int8": 3}  # tfliteRequant at the lane feed (sienna_fmt_pkg::req_lat()), int8 only
FMT = "fp32"  # the build's format; main() sets it


def degree(act: str) -> int:
    """The polynomial degree of act's coefficient set in the build's format: Task 10's SETS_INT8 in int8."""
    if FMT == "int8":
        return reg.gpnae_model.SETS_INT8[reg.activation_to_code(act)][1]
    return DEGREE[act]
```

2. In `model()`: `U = min(K, 6)  # partial sums per PE pixel` becomes
`U = min(K, ADD_LAT + 1)  # partial sums per PE pixel: the adder loop plus one (6 in fp32 and bf16, 2 in int8)`;
`m["act"] = per_lane + 4  # FEED, LATCH, ...` becomes
`m["act"] = per_lane + 4 + REQ_LAT[FMT]  # FEED, LATCH, one wide beat per cycle plus a cycle of read latency, the requantize stage in int8, then done`;
and `m["act_rounds"] = (DEGREE[act] + 1) * max(per_lane, MUL_LAT + ADD_LAT + 1)  # barrel MAC round: max(n, 14)` becomes
`m["act_rounds"] = (degree(act) + 1) * max(per_lane, MAC_LAT[FMT] + 1)  # barrel MAC round: max(n, Horner loop + 1)`.
In fp32 and bf16 these are the old values (6; 13 + 1 = 14; 8 + 1 = 9, and the fp32 table's degrees); in int8 the round
is `max(per_lane, 4)`, barrel_mac's `MIN_PER` (Task 9).

3. `SUMMARY_COLS = [...]` becomes a function:

```python
def summary_cols() -> list:
    """The summary table's columns; int8 counts integer operations and TOPS where the floats count FLOP and GFLOPS."""
    op, rate = ("OP/cyc", "TOPS*") if FMT == "int8" else ("FLOP/cyc", "GFLOPS*")
    return ["config", "latency", "cycles/set", op, rate, "PE use", "limit", "its cycles",
            "host model", "mesh model", "mesh lat model", "mesh lat", "sim"]
```

and `footer` uses `summary_cols()`. In `footer`, the line
`L += [f" * GFLOPS at an ASSUMED {args.clock_mhz:.0f} MHz clock, not a timing result; they count the set's"` and its
continuation become

```python
    unit = "TOPS (2 x MACs per second)" if FMT == "int8" else "GFLOPS"
    L += [f" * {unit} at an ASSUMED {args.clock_mhz:.0f} MHz clock, not a timing result; they count the set's"
          f" {2 * args.n ** 3}-{'OP' if FMT == 'int8' else 'FLOP'} matmul only.",
          " PE use = multiply-accumulates per cycle over the mesh's N^2. limit = the stage with the largest cost per set."]
```

(in fp32 and bf16 the line reads as before).

4. In `one_config`, after `flop = 2 * args.n ** 3` add

```python
    op = "OP" if FMT == "int8" else "FLOP"
    rate = lambda s: flop / s * args.clock_mhz / (1e6 if FMT == "int8" else 1000)  # TOPS in int8, GFLOPS otherwise
    rate_s = lambda s: f"{rate(s):.3f} TOPS" if FMT == "int8" else f"{rate(s):.1f} GFLOPS"
```

the `Matmul rate` line becomes

```python
          f"  Matmul rate  : {flop} {op} per set -> {flop / steady:.1f} {op}/cycle, {rate_s(steady)}"
          f" at an ASSUMED {args.clock_mhz:.0f} MHz, {100 * flop / 2 / steady / args.n ** 2:.1f}% of the mesh's"
          f" {args.n ** 2} MAC/cycle" if steady else "", ""]
```

and in `row`, `f"{flop / steady * args.clock_mhz / 1000:.1f}"` becomes
`f"{rate(steady):.3f}" if FMT == "int8" else f"{rate(steady):.1f}"`.

5. In `main`: `global MUL_LAT, ADD_LAT` becomes `global MUL_LAT, ADD_LAT, FMT`; after
`MUL_LAT, ADD_LAT = UNIT_LAT[args.fmt_name]` add

```python
    FMT = args.fmt_name
    fmts = {t["name"]: t.get("formats", tuple(UNIT_LAT)) for t in reg.PIPELINE_TESTS}
    args.configs = [c for c in args.configs if args.fmt_name in fmts[c]]  # int8-only tests run only in int8
```

`$J/cmds/int8_cmp_perf.py`:

```python
#!/usr/bin/env python3
"""Compares two perf_analysis point runs config by config (latency, cycles per set, result); args: REF_RUN NEW_RUN N T FMT."""
import glob
import json
import sys

R = "/proj/work/spramanik/sienna_jobs/runs"


def rows(run: str, N: int, T: int, F: str) -> dict:
    d = {}
    for js in glob.glob(f"{R}/{run}/results/perf/nt*_N{N}_T{T}_{F}.json"):
        for r in json.load(open(js))["rows"]:
            d[r[0]] = (r[1], r[2], r[-1])
    return d


def compare(ref: str, new: str, N: int, T: int, F: str) -> tuple:
    """(identical, report lines) over the reference run's configs."""
    a, b = rows(ref, N, T, F), rows(new, N, T, F)
    diffs = [f"{c}: reference {a[c]}, new {b.get(c, 'missing')}" for c in a if b.get(c) != a[c]]
    return bool(a) and not diffs, diffs + [f"{len(a)} configs, {len(a) - len(diffs)} identical"]


if __name__ == "__main__":
    ok, lines = compare(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    print("\n".join(lines))
    print("IDENTICAL" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)
```

- [ ] **Step 3: `gemm_sweep.py` in int8**

After `exact_layer` add:

```python
def run_int8(a, sim, shapes) -> None:
    """int8 on the layer engine: A and B quantized as TFLite PTQ would, each product requantized per output channel; every
    output must equal the model (int32 sums, ipu.requant, the int8 lane). The dequantized output's error against float64
    on the unquantized inputs is the quantization error, reported and not gated."""
    reg = mr.regression
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n
    head = (f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} "
            f"{'slot use':>8} {'mism':>6} {'q err':>8} {'wall s':>6}")
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, int8 operands, int32 sums, requantized output; "
                 f"peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    cases = [(name, m, k, n, "linear") for name, m, k, n in shapes]
    cases += [(f"layer_{act}_bias", 64, 48, 40, act) for act in ("tanh", "sigmoid", "selu")]
    rows, bad = [], 0
    for name, m, k, n, act in cases:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A, B = rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
        bias = None if act == "linear" else rng.uniform(-0.5, 0.5, n)
        A_q, s_a, z_a = reg.quant_act(A)
        B_q, s_w = reg.quant_weights(B)
        hw_bias = reg.fold_bias(bias, s_a, s_w, z_a, B_q)
        req = reg.requant_params(reg.wrap32(reg.imatmul(A_q, B_q) + hw_bias[None, :]), s_a, s_w, act)
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw_bias, "act": act,
               "shape": (m, n), "req": req}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name)
        mism = int(np.sum(y != reg.int8_layer_exact(A_q, B_q, hw_bias, req, act)))
        ref = A @ B + (0.0 if bias is None else bias[None, :])
        if act != "linear":
            ref = reg.apply_activation(ref.astype(np.float32), act).astype(np.float64)
        if act == "tanh":
            deq = y / 128.0  # D-4: tanh y * 128, zero point 0
        elif act == "sigmoid":
            deq = (y + 128) / 256.0  # sigmoid y * 256, zero point -128
        elif act == "selu":
            deq = (y - req["zout"]) * req["s_selu"]
        else:
            deq = (y - req["zp"]) * req["s_out"]
        err = float(np.max(np.abs(deq - ref)) / (np.max(np.abs(ref)) or 1.0))
        macs = m * k * n
        r = {"shape": name, "M": m, "K": k, "N": n, "act": act, "sets": sets, "cycles": cyc, "macs": macs,
             "mac_per_cycle": macs / cyc if cyc else 0, "pe_use": macs / (cyc * peak) if cyc else 0,
             "slot_use": macs / (sets * a.n ** 3), "mism": mism, "err": err, "wall": time.time() - t0}
        rows.append(r)
        line = (f"{name:<22} {m:>5} {k:>5} {n:>5} {sets:>7} {cyc:>10} {r['mac_per_cycle']:>9.1f} {100 * r['pe_use']:>6.1f}% "
                f"{100 * r['slot_use']:>7.1f}% {mism:>6} {err:>8.1e} {r['wall']:>6.0f}")
        print(line, flush=True)
        rep.write(line + "\n")
        rep.flush()
        json.dump(rows, open(os.path.join(a.work, f"gemm_sweep_N{a.n}.json"), "w"), indent=1)
        if mism:
            print(f"FAIL {name}: {mism} outputs differ from the bit-exact model", flush=True)
            bad += 1
    line = f"GEMM int8: {len(rows)} cases, {bad} with outputs that differ from the bit-exact model"
    print(line, flush=True)
    rep.write(line + "\n")
    if bad:
        sys.exit(1)
```

In `main`, after `a = ap.parse_args()` add

```python
    if a.fmt_name == "int8" and (a.emulate or a.engine != "layer"):
        ap.error("int8 runs on the layer engine only")
```

and after `sim.build()` and the `shapes` / `--quick` lines, before `rep = open(...)`, add

```python
    if a.fmt_name == "int8":
        run_int8(a, sim, shapes)
        return
```

The `--format` help becomes `format of A and B on the layer engine; int8 sums in int32 and requantizes`.

- [ ] **Step 4: Run them, and fp32 unchanged**

```bash
L s22p_int8 64 12 $J/cmds/g4_perf.sh 16 4 int8 32
L s22p_fp32 64 12 $J/cmds/g4_perf.sh 16 4 fp32 32
L s22g_int8 64 6 $J/venv/bin/python gemm_sweep.py --n 16 --format int8 --quick
python3 $J/cmds/int8_cmp_perf.py g4p_N16_T4_fp32 s22p_fp32 16 4 fp32   # login node, after the runs
```

Expected:
- `s22p_int8`: every config `pass` in the summaries of `nt_N16_T4_int8.log` and `nta_N16_T4_int8.log`, with TOPS
  columns. For the relu configs the activation model is 15 cycles (`per_lane + 4 + 3`, 8 elements per lane at N=16 with
  32 lanes), 3 above fp32's, and the measured activation stage (`Stage cost per set: ... activation`) is 3 cycles above
  `s22p_fp32`'s: the requantize stage is the only change inside it. The mesh latency model is compared with the
  measured one the same way; a gap is reported with the stage it is in, not absorbed by retuning the model;
- `s22p_fp32`: `IDENTICAL` (7 configs);
- `s22g_int8`: every quick shape and the three `layer_*_bias` cases with `mism 0`, and
  `GEMM int8: 9 cases, 0 with outputs that differ from the bit-exact model`.

- [ ] **Step 5: The report builder**

In `/proj/work/spramanik/sienna_jobs/report/build_report.py`:

1. After `G4 = load_g4()` and its two derived lines add

```python
# ── data: the int8 G4 perf runs g8p_N*_T*_int8; accumulate configs recomputed from the raw trace, as g4_table.py does
def load_int8():
    sys.path.insert(0, "/proj/work/spramanik/SIENNA_int8")  # the int8 tree: its regression lists the int8-only tests
    import perf_analysis as pa
    passes = {t["name"]: t.get("accum_passes", 1) for t in pa.reg.PIPELINE_TESTS}
    d = {}
    for js in glob.glob(f"{R}/g8p_N*_T*_int8/results/perf/nt*_N*_T*_int8.json"):
        m = re.search(r"g8p_N(\d+)_T(\d+)_int8/", js)
        N = int(m.group(1))
        for r in json.load(open(js))["rows"]:
            if r[-1] != "pass":
                continue
            lat, cps, P = int(r[1]), float(r[2]), passes.get(r[0], 1)
            if P > 1:
                ev = pa.events(open(f"{os.path.dirname(js)}/raw/{r[0]}_N{N}.log", errors="ignore").read())
                start, done = [c for c, _, _ in ev["HOST_START"]], [c for c, _, _ in ev["DONE"]]
                full = done[P - 1::P]
                gaps = [b - a for a, b in zip(full, full[1:])]
                steady = gaps[len(gaps) // 2:]
                lat, cps = full[0] - start[0], sum(steady) / len(steady) / P
            d[(N, int(m.group(2)), r[0])] = dict(lat=lat, cps=cps, tops=2 * N ** 3 / cps * CLOCK / 1e6)
    return d


def load_gemm_int8():
    """Rows of the int8 GEMM sweep run g8_gemm_int8 (N = 16), or [] before it has run."""
    p = f"{R}/g8_gemm_int8/results/gemm/gemm_sweep_N16.json"
    return json.load(open(p)) if os.path.exists(p) else []


G8 = load_int8()
GEMM8 = load_gemm_int8()
# The int8 lane's worst error against the exact functions, in int8 LSB, from Task 12's G2 summary; None: pending.
ACC8 = "/proj/work/spramanik/SIENNA_int8/testbenches/int8/gpnae_int8_accuracy.json"
INT8_ACT = {"selu": None, "sigmoid": None, "tanh": None}
if os.path.exists(ACC8):
    INT8_ACT.update({k: v["worst_lsb"] for k, v in json.load(open(ACC8))["activations"].items() if k in INT8_ACT})
```

2. Replace `section_formats` with

```python
def section_formats():
    done8 = bool(G8)
    cfgs5 = ("matmul_relu_nopool", "matmul_accum3_bias_linear_nopool", "matmul_random_selu", "matmul_random_tanh",
             "matmul_large_tanh")
    o = ["<h2>4 · Number formats: fp32, bf16, int8</h2>",
         "<p>One format per build: every unit (mesh multipliers and adders, reducer, GPNAE, pooling, dropout) works in the "
         "build's format, chosen by two parameters. " + (
             "bf16 and int8 are done and verified. int8 multiplies int8 by int8 into int32 sums, requantizes them at the "
             "GPNAE lane feed exactly as TFLite's int8 kernels do, and runs the activation in 16-bit fixed point; single-layer "
             "TFLite int8 models match the interpreter's reference kernels bit for bit." if done8 else
             "bf16 is done and verified; int8 is next and will be added here.") + "</p>"]
    rows, cats = [], [CFG[c] for c in cfgs5]
    best_t = lambda n, c: min(TS[n], key=lambda t: (G4[(n, t, c, "bf16")]["cps"], t))
    groups = [dict(name=f"N={n}", color=NCOL[n],
                   values=[G4[(n, best_t(n, c), c, "fp32")]["cps"] / G4[(n, best_t(n, c), c, "bf16")]["cps"] for c in cfgs5])
              for n in NS]
    o.append(fig(sp.bar_chart(cats, groups, "bf16 throughput over fp32", "speedup (x)", width=760, height=380),
                 "Host-bound work runs at the same rate in both formats; activation-bound work is 1.15-1.48x faster in bf16 "
                 "(3-cycle multiplier against 8)."))
    if done8 and all((n, best_t(n, c), c) in G8 for n in NS for c in cfgs5):
        groups8 = [dict(name=f"N={n}", color=NCOL[n],
                        values=[G4[(n, best_t(n, c), c, "fp32")]["cps"] / G8[(n, best_t(n, c), c)]["cps"] for c in cfgs5])
                   for n in NS]
        o.append(fig(sp.bar_chart(cats, groups8, "int8 throughput over fp32", "speedup (x)", width=760, height=380),
                     "At the same tile sizes as the bf16 comparison. int8 runs one multiply-accumulate per PE per cycle like "
                     "the float builds (no MAC packing), so it gains through its 1-cycle units, not a higher peak."))
    for n in (16, 64):
        for c in ("matmul_relu_nopool", "matmul_random_selu", "matmul_random_tanh"):
            t = best_t(n, c)
            a, b, q = G4[(n, t, c, "fp32")], G4[(n, t, c, "bf16")], G8.get((n, t, c))
            p = (lambda f: f(q) if q else "pending")
            rows.append([f"N={n}, T={t}", CFG[c], f"{a['lat']}", f"{b['lat']}", p(lambda q: f"{q['lat']}"),
                         f"{a['cps']:.0f}", f"{b['cps']:.0f}", p(lambda q: f"{q['cps']:.0f}"),
                         f"{a['gf']:,.0f}", f"{b['gf']:,.0f}", p(lambda q: f"{q['tops']:.2f}"),
                         f"{a['cps'] / b['cps']:.2f}x", p(lambda q: f"{a['cps'] / q['cps']:.2f}x")])
        rows.append(None)
    rows.pop()
    o.append(table([[("", 2), ("latency (cycles)", 3), ("cycles per set", 3), ("GFLOPS", 2), ("TOPS", 1), ("speedup over fp32", 2)],
                    [("mesh", 1), ("workload", 1), ("fp32", 1), ("bf16", 1), ("int8", 1), ("fp32", 1), ("bf16", 1), ("int8", 1),
                     ("fp32", 1), ("bf16", 1), ("int8", 1), ("bf16", 1), ("int8", 1)]], rows, align="llrrrrrrrrrrr",
                   note="Each row at the tile size where bf16 is fastest. TOPS = 2 x MACs per second at the ASSUMED 950 MHz "
                        "clock (no timing run exists), counting the set's N x N x N matmul only."
                        + ("" if done8 else " int8 fills in when its G4 runs finish.")))
    k256 = [r["err"] for r in GEMM8 if r.get("K") == 256]
    k1024 = [r["err"] for r in GEMM8 if r.get("K") == 1024]
    exact8 = bool(GEMM8) and all(r.get("mism", 1) == 0 for r in GEMM8)
    g8 = lambda v: f"exact; {100 * max(v):.1f}% quantization" if v and exact8 else "pending"
    lsb = lambda k: "pending" if INT8_ACT[k] is None else f"{INT8_ACT[k]} LSB"
    o.append("<h3>What bf16 and int8 cost in accuracy</h3>")
    o.append(table([[("measure", 1), ("fp32", 1), ("bf16", 1), ("int8", 1)]], [
        ["GEMM error, depth K = 256", "&le; 1e-6", "6-8%", g8(k256)],
        ["GEMM error, depth K = 1024", "&le; 5e-6", "17-30%", g8(k1024)],
        ["SELU worst relative error (target 6.25%)", "0.006%", "2.8%", lsb("selu")],
        ["sigmoid worst relative error", "0.06%", "7.2%", lsb("sigmoid")],
        ["tanh worst relative error", "0.39%", "8.6%", lsb("tanh")],
    ], align="lrrr", note="Every bf16 output equals the bit-exact model of the hardware, so these are bf16's own errors "
                         "(truncating arithmetic, bf16 accumulation), not hardware bugs. Open decision: the activation error target. "
                         "int8: every output equals the bit-exact model; its GEMM figure is the output's quantization error against "
                         "float64 on the unquantized inputs (int8 inputs, int32 sums, one int8 output), and its activation "
                         "figures are the lane's worst error in int8 LSB against the exact functions (G2; target 1 LSB)."))
    return "\n".join(o)
```

The bf16 chart and table values are computed as before; the fp32 and bf16 cells do not change.

3. Check it builds (login node; pure Python, no simulation):

```bash
python3 /proj/work/spramanik/sienna_jobs/report/build_report.py /proj/work/spramanik/sienna_report/sienna_report.html
```

Expected: the path and size print, and section 4 shows `pending` in every int8 performance and GEMM cell (no `g8p_*`
run exists yet); the three activation rows show G2's LSB values from `gpnae_int8_accuracy.json`.

- [ ] **Step 6: Commit (SIENNA) and push**

```bash
git add perf_analysis.py && git commit -m "perf_analysis: int8 unit latencies, requantize stage and fxMac step in the model, TOPS"
git add gemm_sweep.py && git commit -m "gemm_sweep: int8 on the layer engine, bit-exact against the model, quantization error reported"
git push origin int8
```

`build_report.py` and the `cmds/` scripts live in `sienna_jobs`, which is not a repository.

### Task 23: Gate G4, SIENNA in int8

**Files:**
- Create (no repo): `$J/cmds/int8_ck0_O0.sh`, `$J/cmds/int8_top_lint.sh`, `$J/int8_g4_report.py`
- Create (ignored by git): `testbenches/results/int8/sienna_int8_g4.log`

**Interfaces:**
- Consumes: Tasks 16-22 and their job scripts; the bf16 gate's runs `g4p_N*_T*_{fp32,bf16}`, `s22_gemm_{fp32,bf16}`,
  `ms_N16_T4_{fp32,bf16}`, `g4lint2`; the references `i0_sienna_*`, `i0_reg32_*` and `i0_multi_fp32` (Task 0).
- Produces: the runs `g8*`; `$J/int8_g4_report.py OUT.log`, which prints `GATE G4: PASS` or `GATE G4: FAIL` with the
  failing runs; the gate report.

The (N, T) points are the 18 of the bf16 gate's sweep (`g4p_N*_T*`): N = 8 with T = 2, 4, 8; N = 16 with 2..16; N = 32
with 2..32; N = 64 with 2..64. N = 64 runs 128 lanes (N^2 / lanes must not exceed GPNAE's 32-entry FIFO), the others 32.
Memory from the bf16 gate's measurements: its largest perf point (N = 64) peaked at 17.6 GB and N = 32 at 4.3 GB, so
32 GB covers N <= 32 and 64 GB covers N = 64; the SIENNA collapse-k 0 build at N = 32 has no measurement at this level,
so it gets 64 GB, or 128 GB if Task 15's N = 32 collapse-k 0 mesh runs peaked above 40 GB
(`command grep -h "Maximum resident" $J/runs/<those runs>/stdout.log`).

- [ ] **Step 1: Job scripts**

`$J/cmds/int8_ck0_O0.sh` (`chmod +x`):

```bash
#!/bin/bash
# sienna_ck0.sh with OPT_FAST=-O0, since depth slices put thousands of PEs into a few C++ functions; args: N T FMT; run from a snapshot root.
export OPT_FAST=-O0
exec bash /proj/work/spramanik/sienna_jobs/cmds/sienna_ck0.sh "$@"
```

`$J/cmds/int8_top_lint.sh` (`chmod +x`; not Task 9's `int8_lint.sh`, which elaborates GPNAE blocks):

```bash
#!/bin/bash
# Lint of sienna_layer and sienna_top in int8, bf16 and fp32, and the formats that must be rejected; run from a snapshot root.
S=/proj/work/spramanik/sienna_jobs/cmd_sienna_fmt.sh
for f in "0 7" "8 7" "8 23"; do $S lint $f; done
for f in "0 15" "5 10"; do
  $S lint $f | command grep -q "unsupported format" && echo "REJECTED ($f)" || echo "NOT REJECTED ($f)"
done
```

- [ ] **Step 2: Launch**

```bash
# int8 regressions: both mesh modes at N = 16 and 32, and random power-up
L g8r16_int8 32 6 $J/cmd_sienna_fmt.sh reg 16 4 int8
L g8r16ck0_int8 32 6 $J/cmds/sienna_ck0.sh 16 4 int8
L g8r32_int8 32 12 $J/cmd_sienna_fmt.sh reg 32 4 int8
L g8r32ck0_int8 64 24 $J/cmds/int8_ck0_O0.sh 32 4 int8
L g8ri_int8 32 6 $J/cmd_randinit.sh --n 16 --tile-size 4 --format int8
# fp32 and bf16 on this tree, against Task 0's references (i0_sienna_*, i0_reg32_*)
for F in fp32 bf16; do
  L g8r16_$F 32 4 $J/cmd_sienna_fmt.sh reg 16 4 $F
  L g8r32_$F 32 12 $J/cmd_sienna_fmt.sh reg 32 4 $F
  L g8ri_$F 32 6 $J/cmd_randinit.sh --n 16 --tile-size 4 --format $F
done
# TFLite int8 models end to end, two mesh points
L g8_tfl_N16_T4 32 4 $J/cmds/int8_tflite.sh 16 4 32
L g8_tfl_N32_T8 32 12 $J/cmds/int8_tflite.sh 32 8 32
# the N / T sweep, one job per point and format
PAIRS="8,2 8,4 8,8 16,2 16,4 16,8 16,16 32,2 32,4 32,8 32,16 32,32 64,2 64,4 64,8 64,16 64,32 64,64"
for NT in $PAIRS; do
  N=${NT%,*}; T=${NT#*,}; LN=32; MEM=32
  [ $N = 64 ] && LN=128 && MEM=64
  for F in int8 fp32 bf16; do L g8p_N${N}_T${T}_$F $MEM 12 $J/cmds/g4_perf.sh $N $T $F $LN; done
done
# GEMM, models, sienna_multi, lint
L g8_gemm_int8 64 12 $J/venv/bin/python gemm_sweep.py --n 16 --format int8
for F in fp32 bf16; do
  L g8_gemm_$F 64 12 $J/venv/bin/python gemm_sweep.py --n 16 --format $F
  L g8_ms_N16_T4_$F 64 12 bash $J/cmds/model_sweep.sh 16 4 $F 32
done
L g8_multi_int8 32 4 $J/cmd_multi.sh matmul_random_tanh 2 32 1 16 int8
L g8_multi_fp32 32 4 $J/cmd_multi.sh matmul_random_tanh 2 32 1 16 fp32
L g8_lint 32 2 $J/cmds/int8_top_lint.sh
```

Watch the queue (`squeue -u $USER`) and each `$J/runs/<name>.out` for `exit=`; a run that dies on memory is relaunched
with twice the memory under the same name, and the report says so.

- [ ] **Step 3: The gate report script**

`$J/int8_g4_report.py`:

```python
#!/usr/bin/env python3
"""Gate G4 of SIENNA's int8 build (2a) from the g8* farm runs: int8 regressions, fp32 / bf16 against the bf16 tree, TFLite
end to end, the int8 N / T sweep, GEMM, models, sienna_multi, lint, and a size estimate of the requantize stage;
arg: output .log path. Parses run logs only."""
import glob
import json
import os
import re
import sys

J = "/proj/work/spramanik/sienna_jobs"
R = f"{J}/runs"
sys.path.insert(0, f"{J}/cmds")
import int8_cmp_gemm as cgemm  # noqa: E402
import int8_cmp_perf as cperf  # noqa: E402
import int8_cmp_reg as creg  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CLOCK = 950.0  # MHz, ASSUMED: no timing run exists
PAIRS = [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (16, 16), (32, 2), (32, 4), (32, 8), (32, 16), (32, 32),
         (64, 2), (64, 4), (64, 8), (64, 16), (64, 32), (64, 64)]
LANES = {8: 32, 16: 32, 32: 32, 64: 128}
NUM_IDS = 16  # sienna_top's set ids with the default 15 credits: 2^clog2(16)


def text(run: str) -> str:
    p = f"{R}/{run}/stdout.log"
    return ANSI.sub("", open(p, errors="ignore").read()) if os.path.exists(p) else ""


def status(run: str) -> str:
    p = f"{R}/{run}.out"
    if not os.path.exists(p):
        return "not run"
    m = re.search(r"^exit=(\d+)", open(p, errors="ignore").read(), re.M)
    return "running" if not m else f"exit {m.group(1)}"


def passed(run: str):
    m = re.search(r"Passed\s*:\s*(\d+)\s*/\s*(\d+)", text(run))
    return (int(m.group(1)), int(m.group(2))) if m else None


def exact_counts(run: str) -> tuple:
    """(tests with every element exact, tests) from a run's per-test logs."""
    n = e = 0
    for p in glob.glob(f"{R}/{run}/results/pipeline/*.log"):
        raw = open(p, errors="ignore").read()
        t, x = re.search(r"Total\s+:\s+(\d+)", raw), re.search(r"Exact\s+:\s+(\d+)", raw)
        if t and x:
            n += 1
            e += int(t.group(1) == x.group(1) and int(t.group(1)) > 0)
    return e, n


def model_cycles(run: str) -> dict:
    p = f"{R}/{run}/results/models/model_results_N16.json"
    if not os.path.exists(p):
        return {}
    return {m: [(r.get("cycles"), r.get("hw_top")) for r in v.get("runs", [])] for m, v in json.load(open(p)).items()}


def main(out: str) -> None:
    L = ["=" * 100, " GATE G4: SIENNA int8 (sub-project 2a), measured in Verilator RTL simulation on the farm (g8* runs)",
         "=" * 100, ""]
    fails = []

    L.append("1. Regressions in int8 (bit-exact golden: every element must match)")
    for run, what in [("g8r16_int8", "N=16 T=4, collapse-k 1"), ("g8r16ck0_int8", "N=16 T=4, collapse-k 0"),
                      ("g8r32_int8", "N=32 T=4, collapse-k 1"), ("g8r32ck0_int8", "N=32 T=4, collapse-k 0"),
                      ("g8ri_int8", "N=16 T=4, random power-up")]:
        p, (e, n) = passed(run), exact_counts(run)
        ok = bool(p) and p[0] == p[1] and e == n == p[1]
        fails += [] if ok else [run]
        L.append(f"  {run:<16} {what:<28} {status(run):<8} " + (f"{p[0]}/{p[1]} passed" if p else "no summary")
                 + f", {e}/{n} tests exact on every element")

    L += ["", "2. fp32 and bf16 on the int8 tree against Task 0's references, the bf16 branch's code (result and cycles, test by test)"]
    for ref, new in [("i0_sienna_fp32", "g8r16_fp32"), ("i0_sienna_bf16", "g8r16_bf16"),
                     ("i0_reg32_fp32", "g8r32_fp32"), ("i0_reg32_bf16", "g8r32_bf16")]:
        ok, lines = creg.compare(ref, new)
        fails += [] if ok else [new]
        L.append(f"  {new:<16} vs {ref:<16} {'identical' if ok else 'DIFFERS'}: {lines[-1]}")
        L += ["    " + x for x in lines[:-1][:10]]
    for F in ("fp32", "bf16"):
        p = passed(f"g8ri_{F}")
        ok = bool(p) and p[0] == p[1]
        fails += [] if ok else [f"g8ri_{F}"]
        L.append(f"  g8ri_{F:<11} random power-up: " + (f"{p[0]}/{p[1]} passed" if p else f"no summary ({status(f'g8ri_{F}')})"))

    L += ["", "3. Single-layer TFLite int8 models through the RTL against the interpreter (reference kernels)"]
    for run in ("g8_tfl_N16_T4", "g8_tfl_N32_T8"):
        raw = text(run)
        res = re.search(r"TFLITE_INT8: (\d+) models, (\d+) failing", raw)
        ok = "RESULT: PASSED" in raw
        fails += [] if ok else [run]
        L.append(f"  {run:<16} {status(run):<8} " + (res.group(0) if res else "no summary"))
        L += ["    " + x for x in re.findall(r"^MODEL .*$", raw, re.M)]

    L += ["", "4. int8 N / T sweep (measured cycles; TOPS = 2 x MACs per second at an ASSUMED 950 MHz clock)",
          f"  {'N':>3} {'T':>3}  {'config':<34} {'latency':>8} {'cyc/set':>8} {'TOPS':>6}  {'limit':<11} {'mesh lat':>8} {'model':>6}"]
    for N, T in PAIRS:
        run, rows = f"g8p_N{N}_T{T}_int8", {}
        for js in glob.glob(f"{R}/{run}/results/perf/nt*_N{N}_T{T}_int8.json"):
            for r in json.load(open(js))["rows"]:
                rows[r[0]] = r
        if not rows:
            fails.append(run)
            L.append(f"  {N:>3} {T:>3}  {run}: {status(run)}, no results")
            continue
        for c, r in rows.items():
            if r[-1] != "pass":
                fails.append(f"{run} {c}")
                L.append(f"  {N:>3} {T:>3}  {c:<34} SIMULATION DID NOT PASS")
                continue
            cps = float(r[2])
            L.append(f"  {N:>3} {T:>3}  {c:<34} {r[1]:>8} {cps:>8.1f} {2 * N ** 3 / cps * CLOCK / 1e6:>6.2f}  {r[6]:<11} "
                     f"{r[11]:>8} {r[10]:>6}")
    L.append("  Accumulate configs count every depth pass as a set here; the report builder recomputes them from the raw trace.")

    L += ["", "5. fp32 and bf16 sweep points against the bf16 gate's (latency and cycles per set, config by config)"]
    same = 0
    for N, T in PAIRS:
        for F in ("fp32", "bf16"):
            ok, lines = cperf.compare(f"g4p_N{N}_T{T}_{F}", f"g8p_N{N}_T{T}_{F}", N, T, F)
            same += ok
            if not ok:
                fails.append(f"g8p_N{N}_T{T}_{F}")
                L.append(f"  N={N} T={T} {F}: DIFFERS")
                L += ["    " + x for x in lines]
    L.append(f"  {same} of {2 * len(PAIRS)} points identical")

    L += ["", "6. GEMM on the layer engine (N=16)"]
    m = re.search(r"GEMM int8: (\d+) cases, (\d+) with.*$", text("g8_gemm_int8"), re.M)
    ok = bool(m) and m.group(2) == "0"
    fails += [] if ok else ["g8_gemm_int8"]
    L.append(f"  g8_gemm_int8     {status('g8_gemm_int8'):<8} " + (m.group(0) if m else "no summary"))
    for F in ("fp32", "bf16"):
        ok, lines = cgemm.compare(f"s22_gemm_{F}", f"g8_gemm_{F}")
        fails += [] if ok else [f"g8_gemm_{F}"]
        L.append(f"  g8_gemm_{F:<8} vs s22_gemm_{F}: {'identical' if ok else 'DIFFERS'}: {lines[-1]}")

    L += ["", "7. MLPerf Tiny float models, fp32 and bf16 (cycles and answers per inference against the bf16 tree)"]
    for F in ("fp32", "bf16"):
        a, b = model_cycles(f"ms_N16_T4_{F}"), model_cycles(f"g8_ms_N16_T4_{F}")
        ok = bool(a) and a == b
        fails += [] if ok else [f"g8_ms_N16_T4_{F}"]
        L.append(f"  g8_ms_N16_T4_{F} vs ms_N16_T4_{F}: " + ("identical" if ok else f"DIFFERS: reference {a}, new {b}"))

    L += ["", "8. sienna_multi, two copies, N=16"]
    for run in ("i0_multi_fp32", "g8_multi_fp32", "g8_multi_int8"):
        raw = text(run)
        m = re.search(r"^MULTI .*$", raw, re.M)
        ok = "RESULT: PASSED" in raw
        fails += [] if ok else [run]
        L.append(f"  {run:<16} {'PASSED' if ok else 'FAILED'}  {m.group(0) if m else ''}")

    L += ["", "9. Lint (sienna_layer and sienna_top, -Wall -Werror-USERFATAL)"]
    raw = text("g8_lint")
    L += ["  " + x for x in re.findall(r"^(?:sienna_\w+ \(.*|.*REJECTED.*)$", raw, re.M)]
    fails += ["g8_lint"] if ("NOT REJECTED" in raw or not raw or re.search(r"\(0, 7\): exit=[1-9]", raw)) else []

    L += ["", "10. Requantize stage size: an estimate from the RTL's sizing, not synthesis"]
    for N in (8, 16, 32, 64):
        lanes = LANES[N]
        store = NUM_IDS * (N * 40 + 5 * 8 + 16 + 5 + 32) + N * 40
        L.append(f"  N={N:<3} {lanes:>3} lanes: {lanes} tfliteRequant units (one 32 x 32 multiplier each), {store:,} flops of "
                 f"per-set parameters; for scale, the mesh has {N * N} multipliers (int8 x int8 here, 24 x 24 significands in "
                 f"fp32, 8 x 8 in bf16)")
    L += ["  Per-set parameter storage in sienna_top is NUM_IDS x (40 N + 93) bits plus a 40 N-bit copy at g_accept, about",
          "  45,000 flops at N = 64 (an estimate, not synthesis). A FIFO of SETS_IN_FLIGHT entries would save one entry; sharing",
          "  the storage with set_act would not shrink it. Flagged for Soham's area review with the spec's Risk 2 (the fallback",
          "  there: time-share the requantizers across lanes, at a known throughput cost)."]

    L += ["", "11. Not run: synthesis, timing and area (every TOPS figure assumes 950 MHz); the four MLPerf Tiny int8 models",
          "    (sub-project 2b); agreement of the int8 lane with TFLite's int8 tanh / logistic is G2's report."]
    L += ["", "GATE G4: PASS" if not fails else "GATE G4: FAIL (" + ", ".join(sorted(set(fails))) + ")"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Build the gate report**

When every run shows `exit=` (login node; it parses logs only):

```bash
python3 $J/int8_g4_report.py /proj/work/spramanik/SIENNA_int8/testbenches/results/int8/sienna_int8_g4.log
```

Expected:
- section 1: every run `exit 0`, `33/33 passed, 33/33 tests exact on every element`;
- section 2: four `identical`, and both random power-up runs 29/29;
- section 3: both runs `TFLITE_INT8: <k> models, 0 failing`, every `MODEL` line `RTL 0/<n>`;
- section 4: every config of every point with numbers, none `SIMULATION DID NOT PASS`;
- section 5: `36 of 36 points identical`;
- sections 6-8: `0 with outputs that differ`, both GEMM comparisons and both model comparisons identical, the three
  multi runs `PASSED` with equal `steady` cycles for the two fp32 runs;
- section 9: the int8, bf16 and fp32 lint lines with `exit=0` and `REJECTED (0 15)`, `REJECTED (5 10)`;
- the last line `GATE G4: PASS`.

Then check that both AriL checkouts and the SystolicMesh and GPNAE pointers still name one commit (login node):

```bash
cd /proj/work/spramanik/SIENNA_int8
a=$(git -C SystolicMesh/ArithmeticLibrary rev-parse HEAD); b=$(git -C GPNAE/ArithmeticLibrary rev-parse HEAD)
[ "$a" = "$b" ] && echo "SAME-ARIL $a" || echo "DIFFERENT-ARIL $a $b"
git -C SystolicMesh ls-tree HEAD ArithmeticLibrary; git -C GPNAE ls-tree HEAD ArithmeticLibrary   # both name $a
```

Expected: `SAME-ARIL`, and both pointers name that commit.

If an fp32 or bf16 sweep point differs, rerun that point on the bf16 tree (`B r0p_N<N>_T<T>_<F> ... g4_perf.sh`) and
compare with that before calling it a change: the `g4p` runs predate the bf16 branch's last commits (none of which
touched the RTL, so they are expected to agree). Any real difference stops the gate.

The report builder reads the int8 activation accuracy from Task 12's `gpnae_int8_accuracy.json`; the report is
rebuilt in Task 24 Step 2.

- [ ] **Step 5: Report to Soham**

Send the gate report's path and its last line, the int8 TOPS at the best tile size for N = 16 and 64 (section 4, marked
as at an assumed 950 MHz), the requantize and per-set storage size estimate (section 10, marked as an estimate, for the
area item of the plan's Open for Soham list), and anything relaunched. No repository changes in this task.

### Task 24: Close out

**Files:**
- Modify: `.claude/skills/sienna-int8/SKILL.md` (status line, departures), `.claude/skills/sienna-uniform-format/SKILL.md`
  (its status line names int8), `.claude/skills/sienna-report/SKILL.md` (section 4's data)
- Create: `.claude/skills/sienna-report/history/<date>_sienna_int8_g4.txt` (verbatim copy of the gate report), and the
  verbatim copies of the G0, G1 and G2 reports: `<date>_g0_oracle_int8.txt`, `<date>_aril_gate_int8.txt`,
  `<date>_gpnae_gate_int8.txt` (G3's was committed in Task 15)
- Modify (no repo): the rebuilt `/proj/work/spramanik/sienna_report/sienna_report.html`

- [ ] **Step 1: The skills**

`.claude/skills/sienna-int8/SKILL.md`: replace

```
**Status: spec approved 2026-09-29. NOT implemented.**
Update this line as pieces land.
```

with

```
**Status: 2a implemented and verified <date of G4> on the `int8` branch of all four repos; 2b not started.**
Gate reports, verbatim in the `sienna-report` skill's `history/`:
- G4 SIENNA: `<date>_sienna_int8_g4.txt` (farm runs `g8*`; regression 33/33 exact at N = 16 and 32 in both mesh modes,
  TFLite int8 layers equal to the interpreter, fp32 / bf16 cycles identical)
```

followed by one line per earlier gate, each with its verdict, its headline numbers as the report states them and its
history file: G0 `<date>_g0_oracle_int8.txt` (Task 2: the pinned rounding, the outputs compared), G1
`<date>_aril_gate_int8.txt` (Task 8), G2 `<date>_gpnae_gate_int8.txt` (Task 12: bit-exact, the worst LSB per activation),
G3 `<date>_mesh_gate_int8.txt` (Task 15). Copy the three reports not yet in `history/`, each dated by its gate:

```bash
H=.claude/skills/sienna-report/history
for f in $H/*_g0_oracle_int8.txt $H/*_aril_gate_int8.txt $H/*_gpnae_gate_int8.txt; do [ -e "$f" ] && echo "EXISTS: $f"; done   # expect nothing
cp testbenches/results/int8/g0_oracle.log $H/<G0 date>_g0_oracle_int8.txt
cp testbenches/results/int8/aril_gate.log $H/<G1 date>_aril_gate_int8.txt
cp testbenches/results/int8/gpnae_gate.log $H/<G2 date>_gpnae_gate_int8.txt
```

Then add, before `## Out of scope (2a)`:

```
## As built (departures from the design above)

- Requantize parameters of an accumulate group come from its activated pass, the last, like `activation_function_i`;
  the bias comes with the first pass (the mesh's rule). Partial passes may carry anything: the regression gives them decoys.
- The requantize stage is `src/requant_lanes.sv` (one `tfliteRequant` per lane, channel `(k * PER_LANE + b) % N`), 3 cycles
  at the lane feed; `sienna_top` holds each set's parameters by set id and copies the per-channel words at `g_accept`.
  It rounds with `sienna_fmt_pkg::REQ_ROUNDING`, G0's variant (Task 3); `regression._check_rounding()` checks that the
  package, `ipu.REQ_ROUNDING` and `rounding.txt` agree.
- `sienna_layer` in int8 always takes a bias beat per column block; its int32 bias, multipliers and shifts come on
  `w_bias_i`, `w_req_mult_i`, `w_req_shift_i` beside it. A residual pass adds raw int8 codes (no rescale: 2b).
- Dropout in training drops to the output zero point of the set's activation (D-5 as corrected: `req_zp_i` after ReLU
  or linear, 0 after tanh, -128 after sigmoid, `gp_zout_i` after SELU).
- Test stimulus is quantized per accumulate group as TFLite PTQ would (zero points by `rint`, not the converter's nudging);
  SELU's output scale is calibrated from each set's data.
```

plus anything else the work proved different, each with its task.

`.claude/skills/sienna-uniform-format/SKILL.md`: `**Status: uniform bf16 implemented and verified 2026-09-28; int8 not started.**`
becomes `**Status: uniform bf16 implemented and verified 2026-09-28; int8 (sub-project 2a) implemented and verified <date>, see the sienna-int8 skill.**`

`.claude/skills/sienna-report/SKILL.md`: section 4's row becomes
`| 4 | Number formats: fp32, bf16, int8 | g4 runs, int8 perf runs \`g8p_N*_T*_int8\`, GEMM runs \`s22_gemm_*\` and \`g8_gemm_int8\`, \`history/2026-09-28_gpnae_gate.txt\` and the int8 G2 summary (\`testbenches/int8/gpnae_int8_accuracy.json\` in the int8 tree, read by build_report.py) |`.

Copy the gate report: `cp testbenches/results/int8/sienna_int8_g4.log .claude/skills/sienna-report/history/<date>_sienna_int8_g4.txt`.

- [ ] **Step 2: Rebuild and check the report**

Login node; pure Python and a headless browser render, no simulation:

```bash
python3 /proj/work/spramanik/sienna_jobs/report/build_report.py /proj/work/spramanik/sienna_report/sienna_report.html
google-chrome --headless=new --window-size=1100,9000 --screenshot=/proj/work/spramanik/sienna_report/int8_check.png \
  file:///proj/work/spramanik/sienna_report/sienna_report.html
```

Open `int8_check.png` and check section 4: the int8 columns hold numbers (no `pending` except where Task 23's report
lists a missing run), the int8 speedup chart is present, and the accuracy table's int8 column shows the GEMM figures
and the G2 LSB values.

- [ ] **Step 3: Commit (SIENNA) and push, innermost first**

```bash
git add .claude/skills/sienna-int8/SKILL.md && git commit -m "skill: sienna-int8 2a implemented and verified; status and as-built departures"
git add .claude/skills/sienna-uniform-format/SKILL.md && git commit -m "skill: sienna-uniform-format status names the int8 build"
git add .claude/skills/sienna-report/SKILL.md .claude/skills/sienna-report/history/*int8*.txt \
  && git commit -m "skill: sienna-report, int8 in section 4; the G0, G1, G2 and G4 int8 gate reports"
for d in SystolicMesh/ArithmeticLibrary GPNAE/ArithmeticLibrary GPNAE SystolicMesh; do git -C $d status -sb | head -1; done
```

Each line must read `## int8...origin/int8` with no `ahead`. If AriL is ahead, push it first with
`git -C SystolicMesh/ArithmeticLibrary push git@github.com:SoHam-56/ArithmeticLibrary.git int8`; then GPNAE and
SystolicMesh with `git -C <repo> push origin int8`, committing their pointer bumps in SIENNA after them. Then

```bash
git push origin int8 && git status -sb | head -1   # no "ahead"
```

Merging `int8` into `bf16` or `main` is Soham's decision and not part of this plan.

---

## Open for Soham

1. **fxMac coverage (Task 6).** "Exhaustive where feasible" is six 2^32 sweeps, not the 2^48 triples: C in {0, -1,
   32767, -32768} over every A x X, and A in {1.0, 2.0} over every X x C (24 farm slices), plus corners, saturation
   boundaries and random triples. Approve the set or name other fixed values; each extra sweep is 4 more slices.
2. **The GPNAE stop condition (Task 10).** Task 10 measures the float lanes' forms in Q4.11 before anything else in
   Level 2 is built. If any activation misses 1 int8 LSB on a gated case, it stops with `gpnae_int8_stop.log` (measured
   errors by range and degree, and options A: centred operands, B: higher degree, C: a Q-format change) and nothing
   further in Level 2 is built until Soham decides.
3. **Fitted ranges wider than the float lane's (Tasks 10, 11).** int8 has no tail, so the polynomial must reach the
   point where the exact int8 output saturates: sigmoid to |x| = 6.25 (the float lane hands |x| > 3.5 to the tail;
   saturating there gives 255 where the exact value is 248.5) and SELU to x = -7 (float lane -4; 3.8 LSB off at the tightest test case); tanh is
   narrower, 3.125 against 4. Task 10 also reports what the float lanes' ranges would cost.
4. **Per-set parameter area (Tasks 16, 23).** `sienna_top` holds each set's requantize and GPNAE parameters by set id:
   NUM_IDS x (40 N + 93) bits plus a 40 N-bit copy, about 45,000 flops at N = 64 (estimate), beside the 128 per-lane
   32 x 32 requantize multipliers at N = 64. Task 23 section 10 prints the estimate; synthesis is not run. The spec's
   Risk 2 fallback (time-share the requantizers) is not implemented.

## Self-review against the spec

| Spec item (`sienna-int8` SKILL.md) | Tasks |
|---|---|
| Reference: TensorFlow in `sienna_jobs/venv`, the interpreter with `BUILTIN_REF`, the converter for single-layer int8 models | 0 (install), 2 (oracle, models), 12 (TANH / LOGISTIC helper) |
| Format selection: `EXP_W = 0, MAN_W = 7`, `is_int`, `supported(0, 7)`, anything else `$fatal` | 3 (package; D-7 guards in `fpMultiplier` / `fpAdder`), 9, 11, 13, 16, 18, 19 (rejection lints), 23 (lint) |
| Accumulate width `ACC_W` = 32 in int8 (PE sums, reducer, bias, result memory, wide read) | 3 (`acc_w`), 13 (mesh RTL), 14 (TB, bias files), 16 (`sienna_top`), 19 (`sienna_layer`, `sienna_multi`, TBs) |
| Storage: int8 host operands, staging, weight cache; int8 again from requantize on | 13 (operands at `DATA_WIDTH`), 16 (lanes see int8), 17 (pooling), 18 (dropout) |
| Mesh arithmetic: int8 x int8 -> int16, int32 wrapping accumulate, 1-cycle units, latencies from the package | 3, 4 (`intMultiplier`), 5 (`intAdder`), 13, 14 (`matmul_int`; bias near INT32_MAX wraps at G3), 15 |
| Zero points: none in the mesh; input zero-point term folded into the int32 bias; conv padding with z_a in the im2col | 2 (`fold_input_zp`, `im2col_same`; G0 criterion 4), 20 (`fold_bias`, non-zero input zero point tests), 21 (`im2col(pad_value=z_in)`) |
| Requantize: TFLite's conv / FC epilogue, one clamp for int8 range, ReLU, ReLU6 | 1 (`ipu.requant`), 2 (G0), 7 (`tfliteRequant`), 16, 20, 21 (fused ReLU / ReLU6 models) |
| Requantize rounding: TFLite's variant found and pinned at G0, never assumed | 2 (`rounding.txt`), 3 (`REQ_ROUNDING`, `ipu.REQ_ROUNDING`), 7 (both variants in DV), 20 (`_check_rounding`) |
| Requantize placement: one pipelined unit per lane at the GPNAE lane feed, result memory int32 (D-3) | 16 (`requant_lanes`), 22 (the 3 cycles in the perf model) |
| Requantize parameters per set; layer-wide `z_out`, `act_min`, `act_max` (D-2) | 16, 19, 20 (per-channel random multipliers, decoys on partial passes), 21 |
| Activation engine: GPNAE in fixed point, same Horner / barrel_mac / gpnae_poly structure, 16-bit integer units | 9 (`barrel_mac` on `fxMac`), 10 (model), 11 (`gpnae_poly_int8`), 12 |
| GPNAE number format: input rescale to Q4.11 (saturating), 16 x 16 -> 32-bit products, 32-bit add | 1 (`fx_mac`), 6 (`fxMac`, D-1), 4 (W = 16 multiplier), 10, 11 |
| GPNAE coefficients: `poly_coeffs_int8.mem` from `fit_poly_coeffs.py`; fp32 / bf16 tables never change | 10 (fit, `PUBLISHED-UNCHANGED`, bf16 refused), 12 (fit guard in the gate) |
| Beyond the fitted range: saturated int8 value; `gpnae_tail` not instantiated | 10, 11 (`gpnae_tail` rejects int8), 12 |
| GPNAE output: tanh y*128 zp 0, sigmoid y*256 zp -128, SELU per-layer `(M_out, sh_out, z_out)`, ReLU / linear pass through (D-4) | 10, 11, 12 (every int8 input per case, bit-exact; TFLite agreement reported) |
| GPNAE per-layer parameters `M_x, sh_x` and SELU's with the set's configuration | 11 (ports), 16 (per-set storage, `gp_zin_i` from `req_zp`), 19, 20 (`rescale_params`, `quantize_multiplier`) |
| Max pooling: integer compare, pad -128 | 17, 20 (`_maxpool_int`) |
| Dropout: inference bypass; training keeps values, dropped ones the zero point (D-5 as corrected), 1/keep in the next scale | 18, 20 (`drop_zp`; `int8_input_zp_selu_train`) |
| Throughput: one MAC per PE per cycle | 13 (no packing), 22 (TOPS at the assumed clock) |
| G0: requantize and FC / conv reference bit-exact against the interpreter, rounding recorded, G4 models generated | 1, 2 |
| G1: each new unit with the float units' DV; multiplier exhaustive, adder corners, fixed-point sweeps, requantize corners and 10^6 random; fp32 / bf16 unchanged | 3 to 8 |
| G2: lane bit-exact on all 256 inputs at several scales; <= 1 LSB target; TFLite tanh / logistic reported; fp32 / bf16 unchanged | 9 to 12 |
| G3: int8 bit-exact at N = 8..64, every T, both collapse modes, random power-up; fp32 / bf16 cycles identical | 13 to 15 (fp32 / bf16 rerun as the ruled subset: collapse-k 1 everywhere, collapse-k 0 at N <= 32 and N = 64 T >= 16) |
| G4: regression with the bit-exact golden; single-layer TFLite models through the RTL; N / T sweep; report section 4 in TOPS; fp32 / bf16 unchanged | 16 to 23, 24 (report) |
| Working setup: `int8` branch off `bf16` in all four repos, separate checkout, push innermost first | 0, every commit step |
| Out of scope (2b): MLPerf int8 models, residual rescale, MAC packing, LUT activations, other quantization | 19 (residual adds raw codes), 21 (`model_runner` rejects `--format int8`); not planned otherwise |
| Risk 1: optimized kernels round differently | 2 (the oracle forces `BUILTIN_REF`; the default resolver's mismatches are reported for information) |
| Risk 2: per-lane requantize multiplier area | 23 (section 10 estimate); Open for Soham 4 |
| Risk 3: 1-LSB target may need higher degree or fail | 10 (degrees 2 to 12 measured; stop report); Open for Soham 2 |
| Risk 4: SELU output scale calibrated from the data | 10 (`selu_case`), 20 (`requant_params`) |
| Risk 5: fp32 and bf16 must not move | 3, 8, 9, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23 (every gate reruns and compares) |
| Risk 6: `/proj/work` quota | 0 (quota check), 6 (before the sweeps), 15 (before the G3 sweep) |
