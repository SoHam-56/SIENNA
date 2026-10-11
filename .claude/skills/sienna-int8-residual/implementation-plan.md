# SIENNA int8 Residual ADD and Sigmoid Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** int8 residual layers bit-exact against TFLite's int8 ADD (reference kernels), so ResNet-8's three ADDs run on SIENNA, and the int8 GPNAE sigmoid within GPNAE's tolerance (rel <= 6.25% or <= 1 LSB) at every input; fp32, bf16 and every non-ADD int8 set unchanged.

**Architecture:** Bottom up in gated levels. G0b pins TFLite's int8 ADD against the interpreter (BUILTIN_REF) with `ipu.tflite_add` on `ipu.add_params` before any RTL exists. G1 verifies ArithmeticLibrary's new 6-stage `tfliteAdd` (constants per beat, as `tfliteRequant`). G2 adds the exact 19-breakpoint sigmoid ladder for |x| >= 2.5 to GPNAE's int8 lane (no state, no cycle) and makes sigmoid accuracy a hard gate. G3 is a diff: the mesh does not change. G4: `sienna_layer` turns each int8 residual output tile into a hold set (the last depth pass, conv requantize, beats into an N^2-byte `hold_mem`) and an add set (the identity pass, identity requantize, then `add_lanes`, then the normal bypass or lane path); `sienna_top` routes them without a new credit link; the regression, layer-level cases and residual-block TFLite models gate bit-exact. Phase 1 (Tasks 1.0 to 1.14: ArithmeticLibrary, GPNAE, models, oracle, and the bf16 sigmoid fix) runs now; Phase 2 (Tasks 2.0 to 2.12: every SIENNA file) starts only after the `regression_sweep` branch is merged.

**Tech Stack:** SystemVerilog, Verilator 5.035 on the slurm farm (`cr_launch.sh`), Python 3 with numpy, TensorFlow 2.18.1 (TFLite converter and interpreter) in `/proj/work/spramanik/sienna_jobs/venv`, Yosys/ABC for the logic-depth probe.

**Spec:** `.claude/skills/sienna-int8-residual/SKILL.md` (decisions approved by Soham 2026-10-10; the binding authority). Read with it: `sienna-int8` (2a as built, G0-G4, the requantize split, the drain overlap and `a_rq_one_set`), `sienna-rtl` (and `references/known-issues.md` before debugging), `sienna-credits` (As built: the links, `drained`, D-8, registered assertion terms), `sienna-tooling` (the two scripts, the layer-file protocol), `sienna-back-to-back` (streams of sets).

## Global Constraints

- ADD semantics (D1): left_shift = 20. d_i = q_i - zp_i; scaled_i = MBQMSmallerThanOneExp(d_i << 20, m_i, s_i); out = clamp(MBQMSmallerThanOneExp(scaled_1 + scaled_2, m_o, s_o) + zp_o). Prepare: twice_max = 2 max(s1, s2), m_1 = s1/twice_max, m_2 = s2/twice_max, m_o = twice_max/(2^20 s_out), each through QuantizeMultiplierSmallerThanOneExp (shift <= 0). DOUBLE: RDBPOT(SRDHM(x, m), -s) = `ipu.mbqm(x, m, s<=0, "DOUBLE")`; TFLITE_SINGLE_ROUNDING: MBQM itself. |d| <= 255, |scaled| < 2^28, |sum| < 2^29.
- No RTL is written until G0b passes (spec Risk 1).
- `Add/src/tfliteAdd.sv`: parameter ROUNDING (default `sienna_fmt_pkg::REQ_ROUNDING`); ports `clk_i`, `rstn_i`, `valid_i`; `q1_i`, `q2_i` [7:0]; `zp1_i`, `zp2_i` signed [7:0]; `m1_i`, `m2_i`, `mo_i` signed [31:0] (Q0.31); `s1_i`, `s2_i`, `so_i` signed [7:0] (asserted in [-31, 0]); `zpo_i`, `min_i`, `max_i` signed [7:0]; `result_o` [7:0], `done_o`. Constants taken at stage 1 with each beat. Stages (DOUBLE): 1 d_i and SRDHM as (d_i m_i + 2^10) >>> 11; 2 RDBPOT by -s_i; 3 the 30-bit sum; 4 SRDHM(sum, m_o), 30 x 32; 5 RDBPOT by -s_o; 6 + zp_o, then max(min), then min(max).
- Latency: `sienna_fmt_pkg::qadd_lat()` = 6 (the spec's `add_lat()`, renamed: decision P-1), an estimate until the Yosys/ABC depth probe; target: no stage deeper than `tfliteRequant`'s worst (about 46 levels).
- Sigmoid ladder for |x| >= 5120 (Q4.11): k = 19 - #{t_i <= |x|}, t = {5228, 5350, 5479, 5616, 5761, 5916, 6082, 6261, 6456, 6669, 6905, 7170, 7471, 7821, 8240, 8763, 9460, 10515, 12773}; output k - 128 for x < 0, min(127, 128 - k) for x > 0; no state, no cycle; `poly_coeffs_int8.mem`, `barrel_mac`, the float lanes and `gpnae.sv` byte-identical.
- `sienna_set_side.svh` gains `res_hold`, `res_add`, `add_zp1`, `add_zp2`, `add_m1`, `add_s1`, `add_m2`, `add_s2`, `add_mo`, `add_so`, `add_zpo`, `add_min`, `add_max` (+162 bits on L0). ADD constants travel in the add set's own sideband, stored per set id like `s_zp`; zp_1 explicit (sienna_layer sets it to `cfg_req_zp_i`).
- Add set: its own group, no bias, identity requantize (mult 2^30, shift +1, zp 0, clamp [-128, 127]). Hold set: the last depth pass, closes the conv's group, conv's M_c, sh_c, zp and clamp, act = linear, no bank, no output, completes like a partial set.
- Only add-set beats take the ADD path; every other beat keeps `req_lat()`; the set after an add set gets its L3 grant only once `add_lanes` has drained (`rq_room` and `rq_drain` widened). No new credit link.
- Unchanged: fp32 and bf16 (results, cycles, words), the mesh, `tfliteRequant`, `gpnae.sv`; every existing int8 test keeps identical cycles, and identical words except sigmoid tests with some |x| >= 2.5 (that count is reported). A packed residual layer is still rejected.
- Out of scope: `model_runner` lowering (an ADD op in `lower_op`, the 1x1 shortcut as its own layer, TFLite's input order in the frontend), average pool, softmax, depthwise.
- Verification bottom up: G0b, G1, G2, G3 (diff only), G4; a level passes and is reported before the next starts; N = 8, 16, 32 at every tile size while testing; one N = 64 sweep after check-in. Every test gates on a reference that can fail; "finished cleanly" is never a pass.
- Oracle: TensorFlow 2.18.1 from `$J/venv`, `experimental_op_resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_REF`; the optimized (AUTO) resolver is reported, never gated.
- Farm: every Verilator build or simulation, every Yosys run and every TensorFlow run goes through `TREE=$T $J/cr_msgs/cr_launch.sh NAME MEM_GB HOURS cmd...` (32 GB minimum); never the login node. The login node runs only git, edits, `verilator --lint-only`-free checks and the pure-Python model self-tests named in the steps (no simulator, no TensorFlow, under a minute); every gate reruns those on the farm too. N >= 32 SIENNA builds run under `$J/cmds/int8_O0.sh` (TRACE=0 OPT_FAST=-O0). Multi-command jobs are scripts in `$J/cmds/`, never `bash -c "..."`.
- Tree: `T=/proj/scratch/spramanik/int8_res`, a clean clone on branch `int8_residual` in all four repos (Task 1.0). Never edit or snapshot `/proj/work/spramanik/SIENNA` (another agent's branch).
- Both ArithmeticLibrary checkouts (`SystolicMesh/ArithmeticLibrary`, `GPNAE/ArithmeticLibrary`) are on the same commit from Task 1.8 on; change the library only in `SystolicMesh/ArithmeticLibrary` and fast-forward GPNAE's copy from it.
- Commits: one logical change each, plain `git commit -m`, no Co-Authored-By or generated-by lines (before any push: `git log --format=%B BASE..HEAD | command grep -ci co-authored-by` prints 0), innermost repo first (ArithmeticLibrary, then GPNAE and SystolicMesh, then SIENNA), RTL before testbenches before scripts before docs. Nothing is pushed until Task 2.12, and only with Soham's go-ahead; ArithmeticLibrary pushes to `git@github.com:SoHam-56/ArithmeticLibrary.git`.
- Comments are one line; generated reports are `.log`; never delete a file without Soham's go-ahead; GPNAE is published work (report its math bugs, do not change them); shell checks use `command grep` / `command diff`; data registers are not reset (D-8); new assertions sample registered or internal terms, never a host input driven at the edge, and each is shown firing by a mutant; check free space before writing G0b's ~5 x 10^7 outputs (they go to `/proj/scratch`, compressed).
- Snapshot tars in `$J/snaps` stay until every check that reads them has run.

## Review Focus

Five inputs or conditions the spec implies but no test in its Verification section exercises, most likely to bite first. Each line names the task that pins it.

1. **A GPNAE lane after the ADD (tanh, sigmoid, SELU on an add set).** `sienna_top` drives each lane's `gp_zin_i` from the set's requantize zero point, which for an add set is the identity requantize's 0; the lane's real input zero point is the ADD's zp_o. A reasonable person expects lane outputs equal to `gpnae_model` on the ADD output with z_in = zp_o. Pinned in Task 2.5 (`int8_residual_tanh`, `int8_residual_sigmoid_selu_nopool`, both with zp_o != 0) and Task 2.7 (mutant `m_zin`, `gp_zin_i` from `g_zp`, must fail).
2. **The polynomial-to-ladder hand-over and the Q4.11 rail.** x = 5119 (the last polynomial code), x = 5120 (the first ladder code) on both signs, and a rescale that saturates at x = -32768 (where |x| itself saturates to 32767). Expect bit-exactness against the model and, in the ladder, against `exact_int8`. Pinned in Task 1.9 (known values at +-5119, +-5120, -32768, 32767) and Task 1.11 (ladder edge sets 5119, 5120 and a saturating rescale through the RTL).
3. **A conv with its own fused clamp before an ADD with none** (ReLU6 on the conv, NONE on the ADD). The hold set must apply the conv's clamp, the ADD its own. Pinned in Task 2.8 (layer case `relu6_conv_linear_add`).
4. **Back-to-back ADDs with different constants, and a tile's add set followed at once by the next tile's partial (null) sets and hold set.** `hold_mem` must not be overwritten before it is read, the next hold set's beats must not meet the previous add set's drain, and a null must not pass while add beats drain. Pinned in Task 2.5 (`int8_residual_two_layers_nopool`, `int8_residual_accum2_mixed_nopool`, with the TB's cross counters) and Task 2.8 (`tiles`: rt, ct > 1, dt >= 2).
5. **A mid-stream reset while `hold_mem` holds a hold set's beats and its add set has not started.** After reset the next run must give outputs identical to a run without the reset (no stale `hold_mem` pop). Pinned in Task 2.9 (`TB_model_run +mid_reset=C` at the cycle after the first `hold_done`, with `a_add_pop_full` live).

## Decisions this plan makes (flag at review)

- **P-1 (name):** the spec's `add_lat()` collides with `sienna_fmt_pkg::add_lat(exp_w, man_w)`, the adder latency (SystemVerilog functions do not overload, and `regression.py`'s `rtl_lat()` parses `add_lat`). The `tfliteAdd` latency is `qadd_lat()`; read every `add_lat()` in the spec's ADD sections as `qadd_lat()`.
- **P-2 (G0b's home):** `ArithmeticLibrary/Add/testbenches/add_oracle.py`, beside the unit whose G1 vectors it produces (as `Requant/testbenches/gen_requant_vectors.py` sits beside `tfliteRequant`). Not SIENNA's `regression.py` (being rewritten on `regression_sweep`, and SIENNA keeps exactly two Python scripts), not GPNAE (no ADD). It imports TensorFlow only inside `main()`, so `gen_add_vectors.py` and `check_add.py` import without it. Its ~5 x 10^7 outputs live compressed on `/proj/scratch/spramanik/int8_res_data/g0b`, outside git; G1 replays them through `check_add.py`.
- **P-3 (spec gap):** an add set's lanes take `gp_zin_i` = the set's `add_zpo` (Review Focus 1). The spec names only the dropout drop value (`p_zp`).
- **P-4 (fit reproducibility):** `fit_poly_coeffs.py --format int8` keeps scoring candidate tables on the 2a lane (ladder off), so it reproduces `poly_coeffs_int8.mem` byte for byte, as the G2 gate has required since 2a; with the ladder on, sigmoid might meet the tolerance at a lower degree and the fit would write a different table, which the spec forbids. The fit's report adds the with-ladder figures, and its verdict reads them.
- **P-5 (spec ambiguity):** "a rescale for each t_i that lands codes on t_i - 1 and t_i" is two rescales per t_i (one lands a code on t_i - 1, one on t_i, each also landing the negated code on the negative value): one rescale cannot put two int8 codes one Q4.11 step apart at |x| > 255.
- **P-6 (scope of the host interface):** this plan builds the device-protocol half that G4's TFLite check needs: `TB_model_run` reads the `A` line, `model_runner.write_layer` writes it from `job["add"]`, `RtlLayer.run_job` passes it, plus the numerics helpers (`add_words`, `tflite_add_codes`, `add_lane_params`, `int8_residual_exact`). The frontend lowering stays 2b. `regression.py` builds the residual-block jobs by hand from the interpreter's tensors.
- **P-7 (where each G4 test lives):** set-level tests in `TB_sienna_top` (`int8_residual_relu`, `_linear_nopool`, `_tanh`, `_sigmoid_selu_nopool`, `_train`, `_then_relu_nopool`, `_two_layers_nopool`, `_accum2_mixed_nopool`); layer-level cases in `sienna_layer` / `TB_model_run` through `regression.py --action residual` (`resnet_relu`, `deep_cached`, `deep_uncached`, `tiles`, `relu6_conv_linear_add`, `tanh_after_add`). `_deep` and `_tiles` are `sienna_layer` notions (dt, cache, rt, ct) that `TB_sienna_top` does not have.
- **P-8 (G1 exhaustive):** the exhaustive pairs run in a TB loop whose results are dumped and checked by `check_add.py` against `ipu.tflite_add` or the interpreter, as `check_ipu.py` checks dumps; a vector file of 154 corner sets x 65,536 pairs would be 0.6 GB.
- **P-9 (perf):** `regression.py --action perf` skips residual configs (its analyser groups sets by `accum_passes`), and its report says so; the perf model is not extended.
- **P-10 (gates):** the sigmoid hard gate is GPNAE's 5 gated sigmoid `INT8_CASES` (1,280 outputs) in `GPNAE/regression.py`, plus the dense 65,536 in `check_gpnae_model_int8.py`; tanh and SELU stay reported (spec).
- **P-11 (rounding):** if G0b's winner is not DOUBLE (the conv's pin), stop before Task 1.5 and ask Soham: the ADD would need its own pin (`ADD_ROUNDING` in `sienna_fmt_pkg` and `ipu`).
- **P-12 (hold_mem):** a register array in RTL (`[PER_LANE][NUM_LANES] x 8`); its occupancy bits `hm_v` exist only under `ifndef SYNTHESIS`, for `a_hold_push_empty` and `a_add_pop_full`.
- **P-13 (G0b's evidence):** TFLite's DOUBLE and SINGLE roundings almost never differ on an ADD with realistic random scales (a numpy estimate on the existing primitives: 2 differing outputs over 60 models x 65,536 pairs), so random-scale models alone cannot pin the rounding. G0b adds 16 tie models per activation (s1 = s2 = 2^e, s_out = 2^(e+1) or 2^(e+2): outputs on exact halves), and G1's discriminator vectors draw from tie-structured constants.

## Ordering, branches and the merge point

- **Phase 1 (Tasks 1.0 to 1.13), now.** ArithmeticLibrary (`ipu.py`, `sienna_fmt_pkg`, `Add/`), GPNAE (`gpnae_poly_int8.sv`, `gpnae_model.py`, `fit_poly_coeffs.py`, `check_gpnae_model_int8.py`, `regression.py`'s `run_int8`), SystolicMesh (its ArithmeticLibrary pointer only). No SIENNA file changes: no testbench, no `regression.py`, no `model_runner.py`, no Makefile, no submodule pointer. GPNAE's `regression.py` is also touched by `regression_sweep` (its `make regression` section, appended at the end); this plan edits only `run_int8` and its helpers, so the hunks are apart, and Task 2.0 rebases.
- **Phase 2 (Tasks 2.0 to 2.12), after `regression_sweep` merges** into main in all four repos (Soham's word, checked in Task 2.0). `TB_sienna_top` then reads its TB-only settings from `test_knobs.mem` at run time ("one build per hardware-parameter group"), and `make regression` runs points `regression/<fmt>/N<n>_T<t>/<kind>`; Phase 2 tasks name both. Where a Phase 2 step edits a function that `regression_sweep` rewrote, it gives the new code and the merged function's name to put it in; `python3 regression.py --action points` and `--action selftest` confirm the merged names before editing.
- Commits stay local on `int8_residual` until Task 2.12.

## Conventions used by every command below

```bash
J=/proj/work/spramanik/sienna_jobs
T=/proj/scratch/spramanik/int8_res          # the clean clone (Task 1.0)
A=SystolicMesh/ArithmeticLibrary            # paths inside $T
L() { TREE=$T $J/cr_msgs/cr_launch.sh "$@"; }   # NAME MEM_GB HOURS cmd...: snapshots $T, runs cmd from the snapshot root on the farm
```

- A job's console output lands in `$J/runs/NAME/stdout.log` once it exits (buffered until then), the snapshot's `testbenches/results/` in `$J/runs/NAME/results/`, the slurm log in `$J/runs/NAME.out` (it ends in `exit=<code>`). Poll with `squeue -u $USER -o '%.10i %.30j %.8T %.10M'`; for an ETA, ssh to the node in `NAME.out` and check `ps` for `cc1plus` (compiling) or the `V*` binary (simulating).
- `cr_launch.sh` leaves every path component named `results` and `Verilator*` out of the snapshot: nothing a later job reads may live there.
- Run names start `rs1_` (Phase 1) or `rs2_` (Phase 2). Record each run's name, tree commit (`git -C $T rev-parse --short HEAD` and the four submodule heads) and verdict in `$J/runs/res_ledger.log` (local, outside git).
- Mutants are saved patches in `$J/res_mut/`, written by `$J/cmds/res_mkmut.sh` (the tree is never edited for a mutant) and applied inside the snapshot by `$J/cmds/res_mut.sh`. A mutant passes its check when its run FAILS in the way the step names.

---

# Phase 1: ArithmeticLibrary, GPNAE, models and the oracle

### Task 1.0: Clean clone, branches, job scripts and reference runs

**Files:**
- Create: `/proj/scratch/spramanik/int8_res` (clone; branch `int8_residual` in all four repos)
- Create: `$J/cmds/res_py.sh`, `$J/cmds/res_gpnae.sh`, `$J/cmds/res_mut.sh`, `$J/cmds/res_mkmut.sh`, `$J/cmds/res_lint_gp.sh` (local tooling, outside git)

**Interfaces:**
- Produces: `$T` on `int8_residual` everywhere; `res_py.sh DIR SCRIPT [ARGS]` (venv Python in snapshot subdirectory DIR, `RES` exported as the snapshot's `testbenches/results/res`); `res_gpnae.sh ARGS` (GPNAE's `regression.py ARGS`, results and stimulus copied to `testbenches/results/res/gpnae/`); `res_mut.sh PATCH CMD...`; `res_mkmut.sh NAME FILE OLD NEW [OLD NEW...]`; `res_lint_gp.sh`; reference runs `rs1_base_gp_int8`, `rs1_base_gp_fp32`, `rs1_base_gp_bf16`, `rs1_base_yd_rq_d`, `rs1_base_yd_rq_s`, `rs1_base_yd_gp`, `rs1_base_gp_lint`.

- [ ] **Step 1: Check free space**

```bash
df -h /proj/scratch /proj/work | tail -2; du -sh /proj/scratch/spramanik/* 2>/dev/null | sort -h | tail -5
```

Expected: at least 15 GB free on `/proj/scratch` (clone ~1.5 GB, G0b data ~0.1 GB compressed, G1 dumps on node-local disk, snapshots ~0.3 GB each). If less, stop and ask Soham what may be cleared.

- [ ] **Step 2: Clone and branch**

```bash
cd /proj/scratch/spramanik
git clone /proj/scratch/spramanik/n8_fix int8_res
cd int8_res
git remote set-url origin git@github.com:SoHam-56/SIENNA.git && git fetch origin
git submodule update --init --recursive
git switch -c int8_residual
for d in SystolicMesh/ArithmeticLibrary GPNAE/ArithmeticLibrary GPNAE SystolicMesh; do git -C $d switch -c int8_residual; done
git -C SystolicMesh/ArithmeticLibrary remote set-url --push origin git@github.com:SoHam-56/ArithmeticLibrary.git
git -C GPNAE/ArithmeticLibrary remote set-url --push origin git@github.com:SoHam-56/ArithmeticLibrary.git
git log --oneline -2; git submodule status --recursive
```

Expected: HEAD is this plan's commit on top of `2529f0f`; `SystolicMesh` at `7a913e2`, `GPNAE` at `afd786b`, both ArithmeticLibrary checkouts at `1d04a26`; `git status --short` empty in all four.

- [ ] **Step 3: Write the job scripts**

`$J/cmds/res_py.sh`:

```bash
#!/bin/bash
# Runs a Python script with the venv's interpreter from a snapshot subdirectory; exports RES (the snapshot's testbenches/results/res) for its reports; args: DIR SCRIPT [ARGS...].
export VERILATOR_ROOT="$HOME/.local/share/verilator"
ROOT=$(pwd); export RES=$ROOT/testbenches/results/res; mkdir -p "$RES"
dir=$1; shift
cd "$ROOT/$dir" && /proj/work/spramanik/sienna_jobs/venv/bin/python -u "$@"
```

`$J/cmds/res_gpnae.sh`:

```bash
#!/bin/bash
# GPNAE regression on a snapshot, keeping its results and stimulus for comparisons; args go to GPNAE/regression.py; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
ROOT=$(pwd); R=$ROOT/testbenches/results/res/gpnae; mkdir -p "$R/stimulus"
(cd GPNAE && python3 regression.py "$@"); rc=$?
cp -r GPNAE/testbenches/results/. "$R/" 2>/dev/null
cp GPNAE/testbenches/stimulus/*.mem "$R/stimulus/" 2>/dev/null
echo "res_gpnae: regression.py $* exit=$rc"
exit $rc
```

`$J/cmds/res_mut.sh`:

```bash
#!/bin/bash
# Applies a saved mutant patch (paths relative to the snapshot root), then runs the command; args: PATCH CMD [ARGS...]; exit 90 when the patch does not apply.
p=$1; shift
patch -p1 --forward --batch < "$p" || { echo "res_mut: $p does not apply"; exit 90; }
echo "res_mut: applied $(basename "$p")"
exec "$@"
```

`$J/cmds/res_mkmut.sh`:

```bash
#!/bin/bash
# Writes $J/res_mut/NAME.patch: FILE (relative to $T) with each literal OLD replaced by NEW; the tree is not changed; args: NAME FILE OLD NEW [OLD NEW ...].
J=/proj/work/spramanik/sienna_jobs; T=${T:-/proj/scratch/spramanik/int8_res}
name=$1; f=$2; shift 2
mkdir -p "$J/res_mut"; tmp=$(mktemp)
python3 - "$T/$f" "$tmp" "$@" <<'EOF' || { rm -f "$tmp"; exit 1; }
import sys
src, dst, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
s = open(src).read()
for old, new in zip(pairs[0::2], pairs[1::2]):
    if old not in s:
        sys.exit(f"res_mkmut: {old!r} not in {src}")
    s = s.replace(old, new)
open(dst, "w").write(s)
EOF
command diff -u --label "a/$f" --label "b/$f" "$T/$f" "$tmp" > "$J/res_mut/$name.patch"; rm -f "$tmp"
[ -s "$J/res_mut/$name.patch" ] && echo "res_mkmut: $J/res_mut/$name.patch" || { echo "res_mkmut: $name changed nothing"; exit 1; }
```

`$J/cmds/res_lint_gp.sh` (blaunch flattens multi-word arguments, so the lint's options live in the script):

```bash
#!/bin/bash
# Lints gpnae_poly in int8 with -Wall (gpnae_poly_int8 inside); run from a snapshot root.
FMT="-Wall -GEXP_W=0 -GMAN_W=7" PATTERN= exec /proj/work/spramanik/sienna_jobs/cmds/int8_lint.sh gpnae_poly
```

```bash
chmod +x $J/cmds/res_py.sh $J/cmds/res_gpnae.sh $J/cmds/res_mut.sh $J/cmds/res_mkmut.sh $J/cmds/res_lint_gp.sh
```

- [ ] **Step 4: Launch the reference runs on the untouched tree (before any edit: the snapshot is taken at launch)**

```bash
L rs1_base_gp_int8 32 2 $J/cmds/res_gpnae.sh --lane poly --format int8
L rs1_base_gp_fp32 32 2 $J/cmds/res_gpnae.sh --lane poly --format fp32
L rs1_base_gp_bf16 32 2 $J/cmds/res_gpnae.sh --lane poly --format bf16 --model hw
L rs1_base_yd_rq_d 32 2 bash $J/cmds/yosys_depth.sh tfliteRequant ROUNDING=DOUBLE generic
L rs1_base_yd_rq_s 32 2 bash $J/cmds/yosys_depth.sh tfliteRequant ROUNDING=SINGLE generic
L rs1_base_yd_gp 32 4 bash $J/cmds/yosys_depth.sh gpnae_poly_int8 - generic
L rs1_base_gp_lint 32 1 $J/cmds/res_lint_gp.sh
```

Expected after they exit: `rs1_base_gp_int8/stdout.log` holds `BITEXACT: PASS` and `ACCURACY: MISSED (... outside: sigmoid ...; reported, not gated)` (2a's open item), `res_gpnae: ... exit=0`; fp32 and bf16 runs end `Verdict : All patterns passed` and `exit=0`; each `yd` run leaves `results/yd/<tag>/ltp.txt` with a `length=` line. Record the three ltp lengths in the ledger: they are Task 1.7's and Task 1.13's references.

- [ ] **Step 5: Nothing to commit** (no repository file changed; `git -C $T status --short` and each submodule's are empty).

### Task 1.1: `ipu.py`: QuantizeMultiplierSmallerThanOneExp, add_params, tflite_add

**Files:**
- Modify: `$A/Common/models/ipu.py` (append after `quantize_multiplier`)
- Test: `$A/Common/models/test_ipu.py`

**Interfaces:**
- Consumes: `ipu.sx`, `ipu.mbqm(acc, mult, shift, rounding)`, `ipu.quantize_multiplier(real, rounding)`.
- Produces:
  - `ipu.ADD_LEFT_SHIFT = 20`
  - `ipu.quantize_multiplier_smaller_than_one_exp(real: float) -> tuple[int, int]` (mult, shift <= 0); raises `ValueError` outside 0 < real < 1 or when the rounded shift is positive.
  - `ipu.add_params(s1, s2, s_out) -> dict` with int keys `m1, s1, m2, s2, mo, so` (the tflite_add keyword names), from float32 scales as TFLite's Prepare does in doubles.
  - `ipu.tflite_add(q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, amin, amax, rounding) -> np.ndarray` (int64, broadcasting, every argument int8 patterns or values; shifts in [-31, 0]).

- [ ] **Step 1: Write the failing tests**

In `test_ipu.py`, after the `QM` table add:

```python
QMS = [  # real -> QuantizeMultiplierSmallerThanOneExp: QuantizeMultiplier with shift <= 0
    (0.5, (1 << 30, 0)),
    (0.25, (1 << 30, -1)),
    (0.75, (1610612736, 0)),
    (2.0 ** -31, (1 << 30, -30)),
    (2.0 ** -33, (0, 0)),  # shift -32 flushes to (0, 0), which TFLite accepts (shift 0 <= 0)
]
QMS_BAD = [1.0, 0.0, -0.5, 2.0, 1.0 - 2.0 ** -40]  # the last rounds to 2^31: halved, shift 1 > 0, TFLite's CHECK fails
P_EQ = dict(m1=1 << 30, s1=0, m2=1 << 30, s2=0, mo=1 << 30, so=-19)  # add_params(0.5, 0.5, 1.0): 0.5, 0.5, 2^-20
P_UN = dict(m1=1 << 30, s1=-1, m2=1 << 30, s2=0, mo=1431655765, so=-19)  # add_params(0.25, 0.5, 0.75): 0.25, 0.5, 1/(0.75 * 2^20)
P_WIDE = dict(m1=1 << 30, s1=-10, m2=1 << 30, s2=0, mo=1 << 30, so=-15)  # add_params(2^-10, 1, 2^-3): scale ratio 2^10
ADD = [  # q1, q2, zp1, zp2, words, zpo, amin, amax, DOUBLE, SINGLE; P_EQ gives y = (d1 + d2) / 2
    (10, 6, 0, 0, P_EQ, 0, -128, 127, 8, 8),
    (1, 0, 0, 0, P_EQ, 0, -128, 127, 1, 1),  # +0.5: away (DOUBLE) and up (SINGLE) agree
    (-1, 0, 0, 0, P_EQ, 0, -128, 127, -1, 0),  # -0.5: DOUBLE away from zero, SINGLE toward +inf
    (3, 0, 0, 0, P_EQ, 0, -128, 127, 2, 2),  # +1.5
    (-3, 0, 0, 0, P_EQ, 0, -128, 127, -2, -1),  # -1.5
    (127, -128, -128, 127, P_EQ, 5, -128, 127, 5, 5),  # d1 = 255, d2 = -255: 0 + zpo
    (127, 127, 0, 0, P_EQ, 0, -128, 100, 100, 100),  # 127 clamped to act_max
    (-128, -128, 0, 0, P_EQ, 0, -128, 127, -128, -128),
    (0, 0, 0, 0, P_EQ, 0, 20, 10, 10, 10),  # act_min > act_max: max then min gives act_max
    (100, 50, 0, 0, dict(P_EQ, m1=0), 3, -128, 127, 28, 28),  # a flushed multiplier drops input 1: 50 / 2 + 3
    (12, 3, 0, 0, P_UN, 0, -128, 127, 6, 6),  # (0.25 * 12 + 0.5 * 3) / 0.75 = 6
    (-12, -3, 0, 0, P_UN, 0, -128, 127, -6, -6),
    (0x80, 0x7F, 0, 0, P_EQ, 0, 0x80, 0x7F, -1, 0),  # 8-bit patterns: -128 + 127 = -1, so -0.5; clamp patterns read signed
]
```

After `c_requant` add the transcription:

```python
def c_add(q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, lo, hi, rounding):  # reference integer_ops/add.h AddFunc, line for line
    a1 = c_mbqm(wrap32((q1 - zp1) * (1 << 20)), m1, s1, rounding)
    a2 = c_mbqm(wrap32((q2 - zp2) * (1 << 20)), m2, s2, rounding)
    raw = wrap32(c_mbqm(wrap32(a1 + a2), mo, so, rounding) + zpo)
    return min(hi, max(lo, raw))
```

In `main()`, before the final summary:

```python
    for real, want in QMS:
        check(f"quantize_multiplier_smaller_than_one_exp({real!r})", ipu.quantize_multiplier_smaller_than_one_exp(real), want)
    for real in QMS_BAD:
        try:
            ipu.quantize_multiplier_smaller_than_one_exp(real)
            check(f"quantize_multiplier_smaller_than_one_exp({real!r}) raises", "returned", "ValueError")
        except ValueError:
            check(f"quantize_multiplier_smaller_than_one_exp({real!r}) raises", "ValueError", "ValueError")
    check("add_params(0.5, 0.5, 1.0)", sorted(ipu.add_params(0.5, 0.5, 1.0).items()), sorted(P_EQ.items()))
    check("add_params(0.25, 0.5, 0.75)", sorted(ipu.add_params(0.25, 0.5, 0.75).items()), sorted(P_UN.items()))
    check("add_params(2^-10, 1, 2^-3)", sorted(ipu.add_params(2.0 ** -10, 1.0, 2.0 ** -3).items()), sorted(P_WIDE.items()))
    check("add_params is symmetric", sorted(ipu.add_params(0.5, 0.25, 0.75).items()),
          sorted(dict(P_UN, m1=P_UN["m2"], s1=P_UN["s2"], m2=P_UN["m1"], s2=P_UN["s1"]).items()))
    check("ADD_LEFT_SHIFT", ipu.ADD_LEFT_SHIFT, 20)
    for q1, q2, z1, z2, p, zo, lo, hi, d, sg in ADD:
        for r, w in (("DOUBLE", d), ("SINGLE", sg)):
            check(f"tflite_add({q1}, {q2}, {z1}, {z2}, {p}, {zo}, {lo}, {hi}, {r})",
                  ipu.tflite_add(q1, q2, z1, z2, p["m1"], p["s1"], p["m2"], p["s2"], p["mo"], p["so"], zo, lo, hi, r), w)
    # Every (q1, q2) pair at three constant sets against the transcription, both roundings.
    g1, g2 = (np.arange(1 << 16) >> 8) - 128, (np.arange(1 << 16) & 255) - 128
    sets = [(0, 0, P_EQ, 0, -128, 127), (-7, 33, P_UN, 12, 12, 127), (127, -128, P_WIDE, -100, -128, 6)]
    for z1, z2, p, zo, lo, hi in sets:
        for r in ipu.ROUNDINGS:
            got = ipu.tflite_add(g1, g2, z1, z2, p["m1"], p["s1"], p["m2"], p["s2"], p["mo"], p["so"], zo, lo, hi, r)
            want = [c_add(int(a), int(b), z1, z2, p["m1"], p["s1"], p["m2"], p["s2"], p["mo"], p["so"], zo, lo, hi, r)
                    for a, b in zip(g1, g2)]
            check(f"tflite_add exhaustive {r} zp {z1} {z2} {zo}", got, want)
    # Random constants (any int32 multiplier, every shift in [-31, 0]) against the transcription.
    Q = [rng.randint(-128, 127) for _ in range(n)]
    Q2 = [rng.randint(-128, 127) for _ in range(n)]
    SH = [[rng.randint(-31, 0) for _ in range(n)] for _ in range(3)]
    M3 = [[r32() for _ in range(n)] for _ in range(3)]
    for r in ipu.ROUNDINGS:
        got = ipu.tflite_add(np.array(Q), np.array(Q2), nz, np.array(LO), np.array(M3[0]), np.array(SH[0]), np.array(M3[1]),
                             np.array(SH[1]), np.array(M3[2]), np.array(SH[2]), np.array(HI), np.array(LO), np.array(HI), r)
        want = [c_add(*t, r) for t in zip(Q, Q2, Z, LO, M3[0], SH[0], M3[1], SH[1], M3[2], SH[2], HI, LO, HI)]
        check(f"tflite_add random {r}", got, want)
    # tfliteAdd's stage-1 forms: DOUBLE SRDHM(d << 20, m) = (d m + 2^10) >> 11; SINGLE MBQM(d << 20, m, s) = (d m + 2^(10 - s)) >> (11 - s).
    ms = [0, 1, H - 1, H, H + 1, MAX - 1, MAX, MIN, MIN + 1, -1, -H] + [r32() for _ in range(40)]
    d = np.arange(-255, 256)
    bad = 0
    for m in ms:
        bad += int(np.sum(ipu.srdhm(d << 20, m) != (d * m + 1024) >> 11))
        for s in range(-31, 1):
            bad += int(np.sum(ipu.mbqm(d << 20, m, s, "SINGLE") != (d * m + (1 << (10 - s))) >> (11 - s)))
    check("tfliteAdd stage-1 forms", bad, 0)
    # The value bounds the RTL's widths rest on: |scaled| < 2^28, so |sum| < 2^29.
    worst = max(abs(int(ipu.mbqm(dd << 20, m, s, r))) for dd in (255, -255) for m in (MIN, MAX) for s in (0, -1) for r in ipu.ROUNDINGS)
    check("|scaled| < 2^28", worst < (1 << 28), True)
    # The ADD is symmetric: swapping the inputs with their constants gives the same codes.
    for r in ipu.ROUNDINGS:
        a = ipu.tflite_add(g1, g2, -7, 33, 1 << 30, -1, 1610612736, 0, 1431655765, -19, 12, -128, 127, r)
        b = ipu.tflite_add(g2, g1, 33, -7, 1610612736, 0, 1 << 30, -1, 1431655765, -19, 12, -128, 127, r)
        check(f"tflite_add symmetric {r}", a, b)
```

- [ ] **Step 2: Run the test to see it fail (login node: pure Python, seconds)**

Run: `python3 $T/$A/Common/models/test_ipu.py`
Expected: a traceback ending in `AttributeError: module 'ipu' has no attribute 'quantize_multiplier_smaller_than_one_exp'` (no `RESULT: PASSED`).

- [ ] **Step 3: Implement**

Append to `ipu.py`:

```python
ADD_LEFT_SHIFT = 20  # TFLite add.cc Prepare: the inputs' left shift for int8 (15 for int16)


def quantize_multiplier_smaller_than_one_exp(real: float):
    """TFLite QuantizeMultiplierSmallerThanOneExp: QuantizeMultiplier of 0 < real < 1, whose shift must come out <= 0."""
    if not 0.0 < real < 1.0:
        raise ValueError(f"QuantizeMultiplierSmallerThanOneExp needs 0 < real < 1, got {real!r}")
    mult, shift = quantize_multiplier(real)
    if shift > 0:
        raise ValueError(f"QuantizeMultiplierSmallerThanOneExp: {real!r} rounds to shift {shift} > 0")
    return mult, shift


def add_params(s1, s2, s_out) -> dict:
    """TFLite int8 ADD Prepare from the tensors' float32 scales, in doubles: tflite_add's m1, s1, m2, s2, mo, so."""
    f1, f2, fo = (float(np.float32(s)) for s in (s1, s2, s_out))
    twice_max = float(np.float32(2.0) * np.float32(max(f1, f2)))  # 2 * std::max of floats: a float, exact
    m1, sh1 = quantize_multiplier_smaller_than_one_exp(f1 / twice_max)
    m2, sh2 = quantize_multiplier_smaller_than_one_exp(f2 / twice_max)
    mo, sho = quantize_multiplier_smaller_than_one_exp(twice_max / float(np.float32(1 << ADD_LEFT_SHIFT) * np.float32(fo)))
    return dict(m1=m1, s1=sh1, m2=m2, s2=sh2, mo=mo, so=sho)


def tflite_add(q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, amin, amax, rounding):
    """TFLite reference int8 ADD (integer_ops/add.h): (q - zp) << 20 per input, each scaled, summed, rescaled, + zpo, max then min."""
    for s in (s1, s2, so):
        assert np.all(np.asarray(s) <= 0), "MultiplyByQuantizedMultiplierSmallerThanOneExp needs shift <= 0"
    a1 = mbqm((sx(q1, 8) - sx(zp1, 8)) << ADD_LEFT_SHIFT, m1, s1, rounding)
    a2 = mbqm((sx(q2, 8) - sx(zp2, 8)) << ADD_LEFT_SHIFT, m2, s2, rounding)
    raw = sx(mbqm(sx(a1 + a2, 32), mo, so, rounding) + sx(zpo, 8), 32)
    return np.minimum(np.maximum(raw, sx(amin, 8)), sx(amax, 8))
```

- [ ] **Step 4: Run the test to see it pass, then prove it can fail**

Run: `python3 $T/$A/Common/models/test_ipu.py`
Expected: `test_ipu: <n> checks, 0 failures` and `RESULT: PASSED`.

Negative control: edit `ADD_LEFT_SHIFT = 20` to `19` by hand, rerun, see `[FAIL] tflite_add(10, 6, ...` and `[FAIL] ADD_LEFT_SHIFT` lines and `RESULT: FAILED`, then edit it back to `20` and rerun to `RESULT: PASSED` (do not `git checkout` the file: that would drop the new code).

Farm copy: `L rs1_ipu 32 1 make -C SystolicMesh/ArithmeticLibrary/Common ipu` → `stdout.log` ends `RESULT: PASSED`.

- [ ] **Step 5: Commit (ArithmeticLibrary)**

```bash
git -C $T/$A add Common/models/ipu.py Common/models/test_ipu.py
git -C $T/$A commit -m "ipu: tflite_add, add_params and quantize_multiplier_smaller_than_one_exp, TFLite's int8 ADD as reference integer_ops/add.h AddFunc and add.cc Prepare (left shift 20; each input's multiplier and shift from s_i / (2 max(s1, s2)), the output's from 2 max(s1, s2) / (2^20 s_out), all through QuantizeMultiplierSmallerThanOneExp, in doubles from the float32 scales). test_ipu: hand-derived values (ties in both roundings, zero points, clamps, act_min > act_max, a flushed multiplier, unequal scales), every (q1, q2) pair at three constant sets and 20000 random cases against a line-for-line transcription of AddFunc, the stage-1 forms tfliteAdd will use ((d m + 2^10) >> 11 in DOUBLE, (d m + 2^(10 - s)) >> (11 - s) in SINGLE), |scaled| < 2^28 and input symmetry. Not yet confirmed against the interpreter: G0b does that next."
```

### Task 1.2: G0b, the ADD oracle (`add_oracle.py`), and the shared vector helpers

**Files:**
- Create: `$A/Add/testbenches/gen_add_vectors.py` (shared helpers now; the vector writers in Task 1.4)
- Create: `$A/Add/testbenches/add_oracle.py`
- Output (outside git): `/proj/scratch/spramanik/int8_res_data/g0b/` (`manifest.json`, `rounding_add.txt`, one `g0b_<act>_<j>.npz` per model)

**Interfaces:**
- Consumes: `ipu.tflite_add`, `ipu.add_params`, `ipu.round_half_away`, `ipu.REQ_ROUNDING` (the conv's G0 pin from `testbenches/tflite_int8/rounding.txt`).
- Produces (`gen_add_vectors.py`): `PAIRS = 65536`; `Q1 = (np.arange(PAIRS) >> 8) - 128` and `Q2 = (np.arange(PAIRS) & 255) - 128` (q1 outer, q2 inner: the one pair order of the oracle, the TB's exhaustive mode and the checker); `FIELDS = ("zp1", "zp2", "m1", "s1", "m2", "s2", "mo", "so", "zpo", "amin", "amax")`; `act_range(act, scale, zp) -> (int, int)`; `draw_scales(rng) -> [s1, s2, s_out]`; `oracle_manifest(dir) -> dict`; `oracle_consts(dir) -> list[dict]`; `oracle_outputs(dir) -> np.ndarray (models, PAIRS) int64`.
- Produces (`add_oracle.py`): `python add_oracle.py --out DIR [--models-per-act 256] [--report F] [--selftest]`; report lines `ADD ROUNDING: <R>` and `G0b: PASS|FAIL`.

- [ ] **Step 1: Write the shared helpers and the oracle's self-test first**

`gen_add_vectors.py` (this task writes the header, the shared part and an empty `main` that the next task fills):

```python
#!/usr/bin/env python3
"""tfliteAdd's vectors and constant sets from ipu.tflite_add (and G0b's models): the one (q1, q2) order, corners, discriminators, random."""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Common", "models"))
import ipu  # noqa: E402

PAIRS = 1 << 16
Q1 = (np.arange(PAIRS, dtype=np.int64) >> 8) - 128  # q1 outer
Q2 = (np.arange(PAIRS, dtype=np.int64) & 255) - 128  # q2 inner: the oracle's, the TB's exhaustive mode's and the checker's order
FIELDS = ("zp1", "zp2", "m1", "s1", "m2", "s2", "mo", "so", "zpo", "amin", "amax")  # a constant set, in the consts file's column order
LOG2_LO, LOG2_HI, LOG2_SPAN = -10.0, 2.0, 10.0  # scales 2^-10 to 2^2, ratios up to 2^10 (the spec's G0b ranges)
MIN, MAX = ipu.INT32_MIN, ipu.INT32_MAX


def act_range(act: str, scale, zp: int):
    """TFLite CalculateActivationRangeQuantized, int8: zp + round(f / scale), f / scale in float32."""
    q = lambda f: int(zp) + ipu.round_half_away(float(np.float32(f) / np.float32(scale)))
    return {"none": (-128, 127), "relu": (max(-128, q(0.0)), 127), "relu6": (max(-128, q(0.0)), min(127, q(6.0)))}[act]


def draw_scales(rng):
    """s1, s2, s_out log-uniform in [2^-10, 2^2] with max / min <= 2^10, as float32 values."""
    while True:
        e = rng.uniform(LOG2_LO, LOG2_HI, 3)
        if e.max() - e.min() <= LOG2_SPAN:
            return [float(np.float32(2.0 ** v)) for v in e]


def oracle_manifest(d: str) -> dict:
    return json.load(open(os.path.join(d, "manifest.json")))


def oracle_consts(d: str) -> list:
    """Every G0b model's ADD words, in manifest order: add_params of its scales, its zero points and its clamp."""
    return [m["add"] for m in oracle_manifest(d)["models"]]


def oracle_outputs(d: str) -> np.ndarray:
    """Every G0b model's interpreter outputs (BUILTIN_REF), one row of PAIRS per model, in manifest order."""
    return np.stack([np.load(os.path.join(d, m["file"]))["y"].astype(np.int64) for m in oracle_manifest(d)["models"]])
```

`add_oracle.py`:

```python
#!/usr/bin/env python3
"""Gate G0b: single-op int8 TFLite ADD models on the interpreter's BUILTIN_REF kernels against ipu.tflite_add on ipu.add_params, every (q1, q2); pins the ADD's rounding."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_add_vectors as gv  # noqa: E402
from gen_add_vectors import ipu  # noqa: E402

ACTS = ("none", "relu", "relu6")
ROUNDINGS = ("DOUBLE", "SINGLE")
MIN_DISC = 10000  # pairs where DOUBLE and SINGLE differ, over the run: the pin must rest on evidence
FUSED = {"none": 0, "relu": 1, "relu6": 3}  # tflite ActivationFunctionType: NONE, RELU, RELU6


def fq_range(s: float, z: int):
    """The fake-quant range whose scale is s and whose int8 zero point is z: [s (-128 - z), s (127 - z)]."""
    return s * (-128 - z), s * (127 - z)


N_TIE = 16  # tie models per activation: s1 = s2 = 2^e, s_out = 2^(e+1) or 2^(e+2), so outputs land on exact halves where the roundings differ


def model_specs(per_act: int, seed: int):
    """(act, scales, zero points, tie) per model: per activation, per_act random-scale models with every zero point at each position once, then N_TIE tie models."""
    rng = np.random.default_rng(seed)
    out = []
    for act in ACTS:
        perms = [rng.permutation(256) - 128 for _ in range(3)]
        for j in range(per_act):
            out.append((act, gv.draw_scales(rng), [int(p[j % 256]) for p in perms], False))
        for j in range(N_TIE):
            e = int(rng.integers(-10, 1))
            out.append((act, [2.0 ** e, 2.0 ** e, 2.0 ** (e + 1 + j % 2)], [int(v) for v in rng.integers(-128, 128, 3)], True))
    return out


def build(tf, s, z, act):
    """One ADD of two (1, PAIRS) int8 tensors with the activation fused, quantized by fake-quant ranges (scale s[i], zero point z[i])."""
    r = [fq_range(s[i], z[i]) for i in range(3)]
    fq = lambda x, k: tf.quantization.fake_quant_with_min_max_args(x, r[k][0], r[k][1], num_bits=8)
    f = {"none": tf.identity, "relu": tf.nn.relu, "relu6": tf.nn.relu6}[act]

    @tf.function(input_signature=[tf.TensorSpec((1, gv.PAIRS), tf.float32)] * 2)
    def add_fn(a, b):
        return fq(f(fq(a, 0) + fq(b, 1)), 2)

    rep = np.random.default_rng(0)
    holder = tf.Module()
    holder.f = add_fn
    conv = tf.lite.TFLiteConverter.from_concrete_functions([add_fn.get_concrete_function()], holder)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([rep.uniform(*r[0], (1, gv.PAIRS)).astype(np.float32),
                                            rep.uniform(*r[1], (1, gv.PAIRS)).astype(np.float32)] for _ in range(4))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def fused_activation(model) -> int:
    """The fused activation of the model's only operator, read from the flatbuffer."""
    import tflite
    op = tflite.Model.GetRootAsModel(model, 0).Subgraphs(0).Operators(0)
    opt = tflite.AddOptions()
    t = op.BuiltinOptions()
    opt.Init(t.Bytes, t.Pos)
    return opt.FusedActivationFunction()


def run(tf, model, resolver):
    """(the ADD's input1 codes, input2 codes, output codes, input1/input2/output quantization) for the pair grid."""
    it = tf.lite.Interpreter(model_content=model, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    ops = it._get_ops_details()  # private in TF 2.x: op_name, inputs, outputs per op
    if [o["op_name"] for o in ops] != ["ADD"]:
        raise RuntimeError(f"expected one ADD, the converter produced {[o['op_name'] for o in ops]}")
    i1, i2 = (int(i) for i in ops[0]["inputs"][:2])
    io = int(ops[0]["outputs"][0])
    det = {d["index"]: d for d in it.get_input_details()}
    if set(det) != {i1, i2} or it.get_output_details()[0]["index"] != io:
        raise RuntimeError("the ADD's inputs and output are not the graph's")
    it.set_tensor(i1, gv.Q1[None].astype(np.int8))
    it.set_tensor(i2, gv.Q2[None].astype(np.int8))
    it.invoke()
    td = {t["index"]: t for t in it.get_tensor_details()}
    q = {k: (float(np.float32(td[k]["quantization"][0])), int(td[k]["quantization"][1])) for k in (i1, i2, io)}
    return it.get_tensor(io)[0].astype(np.int64), q[i1], q[i2], q[io]


def reference(a: dict, rounding: str, left_shift_19=False):
    """ipu.tflite_add over the pair grid; left_shift_19 is the negative control, AddFunc with 19 in place of 20, which must mismatch."""
    if not left_shift_19:
        return ipu.tflite_add(gv.Q1, gv.Q2, *[a[k] for k in gv.FIELDS], rounding)
    a1 = ipu.mbqm((gv.Q1 - a["zp1"]) << 19, a["m1"], a["s1"], rounding)
    a2 = ipu.mbqm((gv.Q2 - a["zp2"]) << 19, a["m2"], a["s2"], rounding)
    raw = ipu.sx(ipu.mbqm(ipu.sx(a1 + a2, 32), a["mo"], a["so"], rounding) + a["zpo"], 32)
    return np.minimum(np.maximum(raw, a["amin"]), a["amax"])


def selftest() -> int:
    """No TensorFlow: the fake-quant ranges, the clamp rule, the scale draws and the model list."""
    errs = []
    lo, hi = fq_range(0.5, -3)
    if (round(-128 - lo / ((hi - lo) / 255)), (hi - lo) / 255) != (-3, 0.5):
        errs.append(f"fq_range(0.5, -3) = {lo, hi}")
    if gv.act_range("relu", 0.1, -20) != (-20, 127) or gv.act_range("relu6", 0.1, -20) != (-20, 40) or gv.act_range("none", 1, 5) != (-128, 127):
        errs.append("act_range")
    specs = model_specs(256, 1)
    if len(specs) != 3 * (256 + N_TIE) or sum(sp[3] for sp in specs) != 3 * N_TIE:
        errs.append("model list")
    for act in ACTS:
        zs = [sp[2] for sp in specs if sp[0] == act and not sp[3]]
        if any(len({z[k] for z in zs}) != 256 for k in range(3)):
            errs.append(f"{act}: not every zero point at every position")
    sc = np.array([sp[1] for sp in specs])
    if sc.min() < 2.0 ** -10 or sc.max() > 4.0 or (np.log2(sc.max(axis=1)) - np.log2(sc.min(axis=1))).max() > 10 + 1e-6:
        errs.append("scale draws out of range")
    if not (gv.Q1[0], gv.Q2[0], gv.Q1[1], gv.Q2[1], gv.Q1[-1]) == (-128, -128, -128, -127, 127):
        errs.append("pair order")
    for e in errs:
        print(f"[FAIL] {e}")
    print(f"add_oracle selftest: {len(errs)} failures")
    print(f"RESULT: {'PASSED' if not errs else 'FAILED'}")
    return 1 if errs else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="directory for the models' outputs (npz), manifest.json and rounding_add.txt")
    p.add_argument("--report", default=os.path.join(os.environ.get("RES", "."), "g0b_add_oracle.log"))
    p.add_argument("--models-per-act", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.out:
        p.error("--out is required")
    import tensorflow as tf
    os.makedirs(a.out, exist_ok=True)
    ref, auto = tf.lite.experimental.OpResolverType.BUILTIN_REF, tf.lite.experimental.OpResolverType.AUTO
    L = [f"G0b TFLite ADD oracle: TensorFlow {tf.__version__}, numpy {np.__version__}, resolver BUILTIN_REF, "
         f"{a.models_per_act} random-scale and {N_TIE} tie models per activation ({', '.join(ACTS)}), every (q1, q2) pair, seed {a.seed}"]
    tot = {"outputs": 0, "DOUBLE": 0, "SINGLE": 0, "disc": 0, "neg": 0, "auto_DOUBLE": 0, "auto_SINGLE": 0}
    problems, models, zps = [], [], {act: [set(), set(), set()] for act in ACTS}
    disc_random = 0  # discriminating outputs of the random-scale models alone
    for j, (act, s, z, tie) in enumerate(model_specs(a.models_per_act, a.seed)):
        idx = j % (a.models_per_act + N_TIE)
        tag = f"{act} {idx}{' (tie)' if tie else ''}"
        try:
            model = build(tf, s, z, act)
            y, q1, q2, qo = run(tf, model, ref)
            y_auto = run(tf, model, auto)[0]
            if fused_activation(model) != FUSED[act]:
                raise RuntimeError(f"fused activation {fused_activation(model)}, expected {act}")
            add = dict(zp1=q1[1], zp2=q2[1], **ipu.add_params(q1[0], q2[0], qo[0]), zpo=qo[1])
            add["amin"], add["amax"] = gv.act_range(act, qo[0], qo[1])
        except (RuntimeError, ValueError) as e:
            problems.append(f"{tag}: {e}")
            continue
        want = {r: reference(add, r) for r in ROUNDINGS}
        mism = {r: int(np.sum(y != want[r])) for r in ROUNDINGS}
        disc = int(np.sum(want["DOUBLE"] != want["SINGLE"]))
        neg = int(np.sum(y != reference(add, "DOUBLE", left_shift_19=True)))
        if not tie:
            disc_random += disc
            for k, v in enumerate((q1[1], q2[1], qo[1])):
                zps[act][k].add(v)
        tot["outputs"] += y.size
        tot["disc"] += disc
        tot["neg"] += neg
        for r in ROUNDINGS:
            tot[r] += mism[r]
            tot["auto_" + r] += int(np.sum(y_auto != want[r]))
        fn = f"g0b_{act}_{idx:03d}.npz"
        np.savez_compressed(os.path.join(a.out, fn), y=y.astype(np.int8))
        models.append(dict(file=fn, act=act, scales=[q1[0], q2[0], qo[0]], add={k: int(v) for k, v in add.items()},
                           sha256=hashlib.sha256(y.astype(np.int8).tobytes()).hexdigest()))
        L.append(f"{tag}: s {q1[0]:.6g} {q2[0]:.6g} {qo[0]:.6g} zp {q1[1]} {q2[1]} {qo[1]} clamp [{add['amin']}, {add['amax']}] "
                 f"m {add['m1']} {add['m2']} {add['mo']} s {add['s1']} {add['s2']} {add['so']}: mismatches DOUBLE {mism['DOUBLE']} "
                 f"SINGLE {mism['SINGLE']}, discriminating {disc}, left shift 19 {neg}")
    L.append(f"totals: {len(models)} models, outputs {tot['outputs']}, mismatches DOUBLE {tot['DOUBLE']} SINGLE {tot['SINGLE']}, "
             f"discriminating {tot['disc']} ({disc_random} from the random-scale models), negative control (left shift 19) {tot['neg']}")
    L.append(f"default resolver (AUTO, info): mismatches DOUBLE {tot['auto_DOUBLE']} SINGLE {tot['auto_SINGLE']}")
    for act in ACTS:
        n = [len(x) for x in zps[act]]
        L.append(f"{act}: distinct zero points input1 {n[0]}, input2 {n[1]}, output {n[2]} (of 256)")
        if min(n) < 256:
            problems.append(f"{act}: not every zero point reached every position ({n})")
    winners = [r for r in ROUNDINGS if tot[r] == 0]
    if len(winners) != 1:
        problems.append(f"{len(winners)} roundings match every output; exactly one must")
    if tot["disc"] < MIN_DISC:
        problems.append(f"only {tot['disc']} discriminating outputs (need {MIN_DISC})")
    if tot["neg"] == 0:
        problems.append("the negative control (left shift 19) matched every output: the comparison cannot fail")
    if problems:
        L += [f"problem: {m}" for m in problems] + ["G0b: FAIL"]
    else:
        r = winners[0]
        conv = ipu.REQ_ROUNDING
        L.append(f"ADD ROUNDING: {r}; the conv's G0 pin is {conv}: " + ("the same" if r == conv else "DIFFERENT, the ADD needs its own pin"))
        json.dump(dict(tf_version=tf.__version__, rounding=r, models=models), open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
        open(os.path.join(a.out, "rounding_add.txt"), "w").write(r + "\n")
        L.append("G0b: PASS")
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:1] + L[-8:]))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the self-test (login node: no TensorFlow, under a second)**

Run: `cd $T/$A/Add/testbenches && python3 add_oracle.py --selftest`
Expected: `add_oracle selftest: 0 failures`, `RESULT: PASSED`. Then break `fq_range` by hand (`127 - z` to `126 - z`), rerun, see `[FAIL] fq_range(0.5, -3)` and `RESULT: FAILED`, and restore it.

- [ ] **Step 3: Check the converter path on a small run first**

```bash
mkdir -p /proj/scratch/spramanik/int8_res_data && df -h /proj/scratch | tail -1
L rs1_g0b_smoke 32 1 $J/cmds/res_py.sh SystolicMesh/ArithmeticLibrary/Add/testbenches add_oracle.py --models-per-act 4 --out /proj/scratch/spramanik/int8_res_data/g0b_smoke
```

Expected in `stdout.log`: 60 model lines (4 random-scale and 16 tie models per activation); `problem: ... not every zero point reached every position` (4 models cannot reach 256) and `G0b: FAIL` (correct for a smoke run: the gate is the full run); every model line shows `mismatches DOUBLE 0` or `SINGLE 0` and a non-zero `left shift 19` count. If a model line is missing and a `problem:` names "expected one ADD" or "fused activation", the converter did not make one fused ADD: stop and report the converter's op list (from the problem line) to Soham before changing the model construction.

- [ ] **Step 4: Run the gate**

```bash
L rs1_g0b 32 6 $J/cmds/res_py.sh SystolicMesh/ArithmeticLibrary/Add/testbenches add_oracle.py --out /proj/scratch/spramanik/int8_res_data/g0b
```

Expected: `totals: 816 models, outputs 53477376, mismatches DOUBLE 0 SINGLE <n>` with n in the hundreds of thousands (nearly all from the tie models), `discriminating >= 10000` and its random-scale part small (a numpy estimate gave 2 discriminating outputs in 60 random-scale models: TFLite's two roundings almost never differ on realistic ADD constants, which is why the tie models exist), `negative control (left shift 19) > 0`, every activation `distinct zero points ... 256 ... 256 ... 256`, `ADD ROUNDING: DOUBLE; the conv's G0 pin is DOUBLE: the same`, `G0b: PASS`. The report is `$J/runs/rs1_g0b/results/res/g0b_add_oracle.log` (copied into the SIENNA tree's report history in Task 2.11). If the winner is SINGLE, stop (P-11).

- [ ] **Step 5: Commit (ArithmeticLibrary)**

```bash
git -C $T/$A add Add/testbenches/add_oracle.py Add/testbenches/gen_add_vectors.py
git -C $T/$A commit -m "Add/testbenches: add_oracle.py, gate G0b: 816 single-op int8 TFLite ADD models (per fused activation NONE, RELU, RELU6: 256 with scales 2^-10 to 2^2, ratios up to 2^10 and every zero point at each input and the output, and 16 tie models, s1 = s2 = 2^e and s_out = 2^(e+1) or 2^(e+2), because random scales almost never separate the two roundings) on the interpreter's BUILTIN_REF kernels against ipu.tflite_add on ipu.add_params, every (q1, q2) pair; pins the ADD's rounding, reports the default resolver, and fails when a wrong left shift (19) would also match. gen_add_vectors.py holds the pair order and the helpers the oracle, the TB and the checker share. Run rs1_g0b: <outputs> outputs, DOUBLE 0 mismatches, SINGLE <n>, <d> discriminating, ADD ROUNDING DOUBLE (the conv's pin)."
```

Fill the three numbers from the report before committing.

### Task 1.3: `sienna_fmt_pkg::qadd_lat()`

**Files:**
- Modify: `$A/Common/src/sienna_fmt_pkg.sv` (after `req_lat()`)
- Test: `$A/Common/testbenches/TB_sienna_fmt_pkg.sv`

**Interfaces:**
- Produces: `function automatic int qadd_lat();` returning 6 (`tfliteAdd`'s valid_i to done_o), usable in constant expressions.

- [ ] **Step 1: Write the failing test** — in `TB_sienna_fmt_pkg.sv` add `localparam int QA8 = qadd_lat();` beside `RQ8`, and after the `RQ8 localparam` line:

```systemverilog
    expect_eq("qadd_lat()", qadd_lat(), 6);
    expect_eq("QA8 localparam", QA8, 6);
```

- [ ] **Step 2: Run it to see it fail**

Run: `L rs1_pkg_fail 32 1 $J/cmds/int8_aril.sh Common pkg`
Expected: the build stops with a `%Error` naming `qadd_lat`; no `RESULT: PASSED`; `exit=1`.

- [ ] **Step 3: Implement** — in `sienna_fmt_pkg.sv` after `req_lat()`:

```systemverilog
  // valid_i to done_o of tfliteAdd, TFLite's int8 ADD; an estimate until the logic-depth probe (sienna-int8-residual).
  function automatic int qadd_lat();
    return 6;
  endfunction
```

- [ ] **Step 4: Run it to see it pass**

Run: `L rs1_pkg 32 1 $J/cmds/int8_aril.sh Common pkg` → `TB_sienna_fmt_pkg: 0 errors`, `RESULT: PASSED`. Check that SIENNA's perf parser cannot match the new function: `command grep -c "function automatic int add_lat(" $T/$A/Common/src/sienna_fmt_pkg.sv` prints 1.

- [ ] **Step 5: Commit (ArithmeticLibrary)**

```bash
git -C $T/$A add Common/src/sienna_fmt_pkg.sv Common/testbenches/TB_sienna_fmt_pkg.sv
git -C $T/$A commit -m "sienna_fmt_pkg: qadd_lat() = 6, tfliteAdd's latency (an estimate until the logic-depth probe); named qadd_lat because add_lat(exp_w, man_w) is the adder's and functions do not overload. TB_sienna_fmt_pkg checks it, also as a localparam. Run rs1_pkg."
```

### Task 1.4: G1 vector writers and the dump checker

**Files:**
- Modify: `$A/Add/testbenches/gen_add_vectors.py` (the writers and `main`)
- Create: `$A/Add/testbenches/check_add.py`

**Interfaces:**
- Consumes: Task 1.2's `PAIRS`, `Q1`, `Q2`, `FIELDS`, `act_range`, `draw_scales`, `oracle_manifest`, `oracle_consts`, `oracle_outputs`; `ipu.tflite_add`, `ipu.add_params`.
- Produces:
  - consts file: one constant set per line, 11 hex words in `FIELDS` order (`m1`, `m2`, `mo` 8 digits, the rest 2): `write_consts(path, rows)`, `read_consts(path) -> list[dict]` (signed ints).
  - vector file: one vector per line, 14 hex words `q1 q2 zp1 zp2 m1 s1 m2 s2 mo so zpo amin amax want` (widths 2 2 2 2 8 2 8 2 8 2 2 2 2 2).
  - `corner_consts() -> list[dict]` (154 sets), `corner_rows()`, `random_rows(n, rng)`, `tie_rows(n, rng)`, `discriminators(rng, want)` (2000 rows where the roundings differ), `expected(cols, rounding)`.
  - CLI: `gen_add_vectors.py OUT --rounding R [--random N] [--seed S]` (vector file), `gen_add_vectors.py OUT --consts corners`, `gen_add_vectors.py OUT --consts oracle --oracle DIR`.
  - `check_add.py DUMP --consts F --rounding R [--oracle DIR]` prints `check_add: <n> results of <k> constant sets against <ref>, <m> mismatches` and `RESULT: PASSED|FAILED`; `check_add.py --selftest`.
  - dump file: per constant set, in consts order, 2048 lines of 32 results (64 hex digits, first result first), q1 outer and q2 inner.

- [ ] **Step 1: Write the self-test first** — `check_add.py`:

```python
#!/usr/bin/env python3
"""Checks a TB_tfliteAdd dump (each constant set's 65,536 results, q1 outer and q2 inner, 32 hex bytes a line) against ipu.tflite_add or G0b's interpreter outputs."""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_add_vectors as gv  # noqa: E402
from gen_add_vectors import ipu  # noqa: E402

LINE = 32  # results per dump line


def read_dump(path: str, n_sets: int) -> np.ndarray:
    rows = [ln.strip() for ln in open(path) if ln.strip()]
    want = n_sets * gv.PAIRS // LINE
    if len(rows) != want or any(len(r) != 2 * LINE for r in rows):
        raise ValueError(f"{path}: {len(rows)} lines, expected {want} of {2 * LINE} hex digits")
    return np.frombuffer(bytes.fromhex("".join(rows)), dtype=np.int8).astype(np.int64).reshape(n_sets, gv.PAIRS)


def model(c: dict, rounding: str) -> np.ndarray:
    return np.asarray(ipu.tflite_add(gv.Q1, gv.Q2, *[c[k] for k in gv.FIELDS], rounding), np.int64)


def check(dump: str, consts: list, rounding: str, oracle=None) -> int:
    got = read_dump(dump, len(consts))
    pin = gv.oracle_manifest(oracle)["rounding"] if oracle else None
    if oracle and rounding == pin:
        want, ref = gv.oracle_outputs(oracle), f"G0b's interpreter outputs ({oracle})"
        if want.shape != got.shape:
            raise ValueError(f"{oracle}: {want.shape[0]} models, the dump holds {got.shape[0]} sets")
    else:
        want, ref = np.stack([model(c, rounding) for c in consts]), f"ipu.tflite_add ({rounding})"
    bad = got != want
    print(f"check_add: {got.size} results of {len(consts)} constant sets against {ref}, {int(bad.sum())} mismatches")
    for s, i in list(zip(*np.nonzero(bad)))[:20]:
        print(f"  set {s} q1 {gv.Q1[i]} q2 {gv.Q2[i]}: rtl {got[s, i]}, want {want[s, i]}, constants {consts[s]}")
    if oracle and rounding != pin:
        print(f"  info: {int(np.sum(got != gv.oracle_outputs(oracle)))} results differ from the interpreter ({pin}), as the other rounding should")
    print(f"RESULT: {'PASSED' if not bad.any() else 'FAILED'}")
    return 0 if not bad.any() else 1


def write_dump(path: str, rows: np.ndarray) -> None:
    """A dump as TB_tfliteAdd writes it, for the self-test."""
    b = (np.asarray(rows, np.int64) & 0xFF).astype(np.uint8).tobytes().hex()
    with open(path, "w") as f:
        f.write("".join(b[i:i + 2 * LINE] + "\n" for i in range(0, len(b), 2 * LINE)))


def selftest() -> int:
    errs = []
    cs = gv.corner_consts()
    if len(cs) != 154:
        errs.append(f"{len(cs)} corner sets, expected 154")
    with tempfile.TemporaryDirectory() as d:
        gv.write_consts(os.path.join(d, "c.txt"), cs)
        if gv.read_consts(os.path.join(d, "c.txt")) != [{k: int(c[k]) for k in gv.FIELDS} for c in cs]:
            errs.append("consts do not round-trip")
        two = cs[:1] + cs[-1:]
        good = np.stack([model(c, "DOUBLE") for c in two])
        write_dump(os.path.join(d, "ok.txt"), good)
        if check(os.path.join(d, "ok.txt"), two, "DOUBLE") != 0:
            errs.append("a correct dump failed")
        bad = good.copy()
        bad[1, 12345] ^= 1
        write_dump(os.path.join(d, "bad.txt"), bad)
        if check(os.path.join(d, "bad.txt"), two, "DOUBLE") != 1:
            errs.append("a dump with one wrong result passed")
        try:
            check(os.path.join(d, "ok.txt"), cs[:3], "DOUBLE")
            errs.append("a short dump was read")
        except ValueError:
            pass
        cols = gv.corner_rows()
        gv.write_vectors(os.path.join(d, "v.txt"), cols, "SINGLE")
        tab = np.loadtxt(os.path.join(d, "v.txt"), dtype=str, ndmin=2)
        w = np.array([int(x, 16) for x in tab[:, -1]], np.int64)
        if not np.array_equal(ipu.sx(w, 8), gv.expected(cols, "SINGLE")) or tab.shape != (154 * 49, 14):
            errs.append("vector file does not round-trip")
    for e in errs:
        print(f"[FAIL] {e}")
    print(f"check_add selftest: {len(errs)} failures")
    print(f"RESULT: {'PASSED' if not errs else 'FAILED'}")
    return 1 if errs else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", nargs="?")
    p.add_argument("--consts")
    p.add_argument("--rounding", choices=ipu.ROUNDINGS)
    p.add_argument("--oracle", help="G0b's output directory: in its pinned rounding the reference is the interpreter")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.dump and a.consts and a.rounding):
        p.error("give DUMP, --consts and --rounding")
    sys.exit(check(a.dump, gv.read_consts(a.consts), a.rounding, a.oracle))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to see it fail**

Run (login node, pure Python): `cd $T/$A/Add/testbenches && python3 check_add.py --selftest`
Expected: `AttributeError: module 'gen_add_vectors' has no attribute 'corner_consts'`.

- [ ] **Step 3: Implement the writers** — append to `gen_add_vectors.py`:

```python
COLS = ("q1", "q2") + FIELDS  # a vector file's columns, before the expected result
WIDE = ("m1", "m2", "mo")  # the 32-bit words; every other column is 8 bits
BASE = dict(zp1=0, zp2=0, m1=1 << 30, s1=0, m2=1 << 30, s2=0, mo=1 << 30, so=-20, zpo=0, amin=-128, amax=127)  # y = (d1 + d2) / 4


def write_consts(path: str, rows) -> None:
    with open(path, "w") as f:
        f.write("".join(" ".join(f"{int(r[k]) & 0xFFFFFFFF:08x}" if k in WIDE else f"{int(r[k]) & 0xFF:02x}" for k in FIELDS) + "\n"
                        for r in rows))


def read_consts(path: str) -> list:
    rows = [[int(x, 16) for x in ln.split()] for ln in open(path) if ln.strip()]
    return [{k: int(ipu.sx(v, 32 if k in WIDE else 8)) for k, v in zip(FIELDS, r)} for r in rows]


def corner_consts() -> list:
    """The exhaustive sweep's sets: every shift of each input and the output, the extreme multipliers, the zero-point corners, tight and inverted clamps, flushed multipliers, 32 from add_params."""
    rows = [dict(BASE, **{k: s}) for k in ("s1", "s2", "so") for s in range(-31, 1)]
    rows += [dict(BASE, m1=a, m2=b, mo=c) for a, b, c in itertools.product((1 << 30, MAX), repeat=3)]
    rows += [dict(BASE, zp1=a, zp2=b, zpo=c) for a, b, c in itertools.product((-128, 127), repeat=3)]
    rows += [dict(BASE, amin=lo, amax=hi) for lo, hi in ((0, 0), (-1, 0), (5, 6), (127, 127), (-128, -128), (10, -10), (0, 127))]
    rows += [dict(BASE, **{k: 0}) for k in WIDE]
    rng = np.random.default_rng(11)
    for _ in range(32):
        s = draw_scales(rng)
        z = [int(v) for v in rng.integers(-128, 128, 3)]
        lo, hi = act_range(("none", "relu", "relu6")[int(rng.integers(0, 3))], s[2], z[2])
        rows.append(dict(zp1=z[0], zp2=z[1], **ipu.add_params(*s), zpo=z[2], amin=lo, amax=hi))
    return rows


def corner_rows() -> dict:
    """Every corner set at 49 (q1, q2) pairs of extreme and zero codes."""
    qx = (-128, -127, -1, 0, 1, 126, 127)
    cols = {k: [] for k in COLS}
    for c in corner_consts():
        for a, b in itertools.product(qx, qx):
            cols["q1"].append(a)
            cols["q2"].append(b)
            for k in FIELDS:
                cols[k].append(c[k])
    return {k: np.array(v, dtype=np.int64) for k, v in cols.items()}


def random_rows(n: int, rng) -> dict:
    """Uniform codes and zero points; half the constants from add_params on random scales, half raw (normalized, any int32 or 0 multipliers, any shift in [-31, 0]); mostly full clamps."""
    c = {k: rng.integers(-128, 128, n) for k in ("q1", "q2", "zp1", "zp2", "zpo")}
    for m in WIDE:
        kind = rng.integers(0, 16, n)
        c[m] = np.where(kind < 13, rng.integers(1 << 30, MAX + 1, n), np.where(kind < 15, rng.integers(MIN, MAX + 1, n), 0))
    for s in ("s1", "s2", "so"):
        c[s] = rng.integers(-31, 1, n)
    real = np.flatnonzero(rng.integers(0, 2, n))
    for i in real:
        p = ipu.add_params(*draw_scales(rng))
        for k, v in p.items():
            c[k][i] = v
    sel, r1, r2 = rng.integers(0, 8, n), rng.integers(-128, 128, n), rng.integers(-128, 128, n)
    c["amin"] = np.select([sel < 5, sel == 5], [-128, c["zpo"]], default=np.minimum(r1, r2))
    c["amax"] = np.select([sel < 5, sel == 5], [127, 127], default=np.maximum(r1, r2))
    return {k: np.asarray(c[k], dtype=np.int64) for k in COLS}


def expected(cols: dict, rounding: str) -> np.ndarray:
    return np.asarray(ipu.tflite_add(cols["q1"], cols["q2"], *[cols[k] for k in FIELDS], rounding), np.int64)


def tie_rows(n: int, rng) -> dict:
    """Constants whose outputs land on exact halves (m 2^30, s1 = s2 in {0, -1}, so in {-19, -20, -21}): random constants almost never separate the roundings."""
    c = {k: rng.integers(-128, 128, n) for k in ("q1", "q2", "zp1", "zp2", "zpo")}
    s = rng.integers(-1, 1, n)
    c.update(m1=np.full(n, 1 << 30), m2=np.full(n, 1 << 30), mo=np.full(n, 1 << 30), s1=s, s2=s, so=rng.integers(-21, -18, n),
             amin=np.full(n, -128), amax=np.full(n, 127))
    return {k: np.asarray(c[k], dtype=np.int64) for k in COLS}


def discriminators(rng, want: int = 2000) -> dict:
    """Vectors where DOUBLE and SINGLE disagree, so both RTL variants see their difference: from random and tie-structured constants."""
    pool = {k: np.concatenate([a, b]) for (k, a), b in zip(random_rows(100000, rng).items(), tie_rows(50000, rng).values())}
    d = np.flatnonzero(expected(pool, "DOUBLE") != expected(pool, "SINGLE"))
    d = rng.permutation(d)[:want]
    assert d.size == want, f"only {d.size} discriminating vectors"
    return {k: v[np.sort(d)] for k, v in pool.items()}


def write_vectors(path: str, cols: dict, rounding: str) -> None:
    want = expected(cols, rounding)
    table = np.stack([ipu.bits(cols[k], 32 if k in WIDE else 8) for k in COLS] + [ipu.bits(want, 8)], axis=1)
    np.savetxt(path, table, fmt=" ".join("%08x" if k in WIDE else "%02x" for k in COLS) + " %02x")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out")
    p.add_argument("--rounding", choices=ipu.ROUNDINGS)
    p.add_argument("--random", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--consts", choices=("corners", "oracle"), help="write a constant-set file for the exhaustive or oracle mode")
    p.add_argument("--oracle", help="G0b's output directory (with --consts oracle)")
    a = p.parse_args()
    if a.consts:
        rows = corner_consts() if a.consts == "corners" else oracle_consts(a.oracle)
        write_consts(a.out, rows)
        print(f"{a.out}: {len(rows)} constant sets ({a.consts})")
        return
    if not a.rounding:
        p.error("a vector file needs --rounding")
    rng = np.random.default_rng(a.seed)
    parts = [corner_rows(), discriminators(rng), random_rows(a.random, rng)]
    cols = {k: np.concatenate([x[k] for x in parts]) for k in COLS}
    write_vectors(a.out, cols, a.rounding)
    w, o = expected(cols, a.rounding), expected(cols, "SINGLE" if a.rounding == "DOUBLE" else "DOUBLE")
    print(f"{a.out}: {w.size} vectors ({parts[0]['q1'].size} corners, {parts[1]['q1'].size} rounding discriminators, {a.random} random), "
          f"rounding {a.rounding}, {int(np.sum(w != o))} differ from the other rounding, "
          f"{100 * float(np.mean((w == cols['amin']) | (w == cols['amax']))):.1f}% at a clamp bound")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the self-test to see it pass**

Run: `cd $T/$A/Add/testbenches && python3 check_add.py --selftest`
Expected: two `check_add: ...` lines (0 mismatches, then 1 mismatch with its `set 1 ...` line), `check_add selftest: 0 failures`, `RESULT: PASSED`.

- [ ] **Step 5: No commit yet** — the bench files are committed after the RTL (Task 1.6, RTL first).

### Task 1.5: `TB_tfliteAdd` and the unit Makefile (the failing test)

**Files:**
- Create: `$A/Add/testbenches/TB_tfliteAdd.sv`
- Create: `$A/Add/Makefile`

**Interfaces:**
- Consumes: Task 1.4's file formats; `sienna_fmt_pkg::qadd_lat()`; the `tfliteAdd` ports of the Global Constraints.
- Produces: `make -C $A/Add verilator|exhaustive|oracle|both|vivado_tb|vectors|lint|lint_bad [ROUNDING=DOUBLE|SINGLE] [RANDOM=N] [ORACLE=dir] [PLUSARGS=...] [EXTRA_FLAGS=...]`; the TB prints `tfliteAdd ROUNDING=<R>: <n> ..., <e> errors, latency <l>` then `RESULT: PASSED|FAILED`.

- [ ] **Step 1: Write the TB**

```systemverilog
`timescale 1ns / 100ps

// tfliteAdd against ipu.tflite_add: +VEC= vectors with expected results, or +CONSTS= sets run over every (q1, q2) into +DUMP= for check_add.py; a bubble every 64; latency qadd_lat(); no DPI, so Vivado runs it too.
module TB_tfliteAdd #(
    parameter string ROUNDING = "DOUBLE"
);
  localparam int LAT = sienna_fmt_pkg::qadd_lat();
  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic valid = 0;
  logic [7:0] q1 = '0, q2 = '0, zp1 = '0, zp2 = '0, s1 = '0, s2 = '0, so = '0, zpo = '0, amin = '0, amax = '0;
  logic [31:0] m1 = '0, m2 = '0, mo = '0;
  logic [7:0] res;
  logic done;

  tfliteAdd #(.ROUNDING(ROUNDING)) dut (
      .clk_i(clk), .rstn_i(rstn), .valid_i(valid), .q1_i(q1), .q2_i(q2), .zp1_i(zp1), .zp2_i(zp2), .m1_i(m1), .m2_i(m2),
      .mo_i(mo), .s1_i(s1), .s2_i(s2), .so_i(so), .zpo_i(zpo), .min_i(amin), .max_i(amax), .result_o(res), .done_o(done));

  typedef struct packed {
    logic [7:0] q1, q2, zp1, zp2;
    logic [31:0] m1;
    logic [7:0] s1;
    logic [31:0] m2;
    logic [7:0] s2;
    logic [31:0] mo;
    logic [7:0] so, zpo, amin, amax;
  } vec_t;
  logic [7:0] want_q[$];  // vector mode: the expected results, in issue order
  bit dump_mode = 0;
  int fd_dump = 0;
  longint checked = 0, errs = 0, issued = 0;
  int cyc = 0, t_issue = -1, lat = -1;

  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (valid && t_issue < 0) t_issue = cyc;
    if (rstn && done) begin
      if (lat < 0) lat = cyc - t_issue;  // edges from the one that took valid_i to the one that sees done_o
      checked++;
      if (dump_mode) begin
        $fwrite(fd_dump, "%02x", res);
        if (checked % 32 == 0) $fwrite(fd_dump, "\n");
      end else if (want_q.size() == 0) begin
        errs++;
        $display("[FAIL] a result with nothing issued");
      end else begin
        automatic logic [7:0] w = want_q.pop_front();
        if (res !== w) begin
          errs++;
          if (errs <= 20) $display("[FAIL] vector %0d: got %0d, ipu.tflite_add %0d", checked - 1, $signed(res), $signed(w));
        end
      end
    end
  end

  // One vector on valid_i; every 64th is followed by a bubble whose inputs change with valid_i low, and nothing may come out for it.
  task automatic issue(input vec_t v);
    #1 valid = 1;
    {q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, amin, amax} = v;
    @(posedge clk);
    issued++;
    if (issued % 64 == 0) begin
      #1 valid = 0;
      {q1, q2, zp1, zp2, m1, s1, m2, s2, mo, so, zpo, amin, amax} = {$urandom, $urandom, $urandom, $urandom, $urandom, $urandom};
      @(posedge clk);
    end
  endtask

  // A file line's words in column order (q1 q2, then gen_add_vectors.FIELDS) as a vector.
  function automatic vec_t vec_of(input logic [31:0] f[13]);
    return {f[0][7:0], f[1][7:0], f[2][7:0], f[3][7:0], f[4], f[5][7:0], f[6], f[7][7:0], f[8], f[9][7:0], f[10][7:0], f[11][7:0],
            f[12][7:0]};
  endfunction

  initial begin
    string vec, consts, dump;
    int fd, n_sets = 0;
    logic [31:0] f[13], w;
    repeat (4) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    if ($value$plusargs("CONSTS=%s", consts)) begin
      if (!$value$plusargs("DUMP=%s", dump)) $fatal(1, "TB_tfliteAdd: +CONSTS needs +DUMP");
      dump_mode = 1;
      fd = $fopen(consts, "r");
      fd_dump = $fopen(dump, "w");
      if (fd == 0 || fd_dump == 0) $fatal(1, "TB_tfliteAdd: cannot open %s or %s", consts, dump);
      while ($fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h\n", f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12]) == 11) begin
        n_sets++;
        for (int i = 0; i < 65536; i++) begin
          f[0] = 32'((i >> 8) - 128);  // q1 outer, q2 inner: gen_add_vectors.Q1, Q2
          f[1] = 32'((i & 255) - 128);
          issue(vec_of(f));
        end
      end
    end else begin
      if (!$value$plusargs("VEC=%s", vec)) vec = (ROUNDING == "SINGLE") ? "vectors_single.mem" : "vectors_double.mem";
      fd = $fopen(vec, "r");
      if (fd == 0) $fatal(1, "TB_tfliteAdd: cannot open %s", vec);
      while ($fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10],
                     f[11], f[12], w) == 14) begin
        want_q.push_back(w[7:0]);
        issue(vec_of(f));
      end
    end
    $fclose(fd);
    #1 valid = 0;
    repeat (LAT + 4) @(posedge clk);
    if (dump_mode) $fclose(fd_dump);
    if (issued == 0) begin
      errs++;
      $display("[FAIL] no vectors read");
    end
    if (checked != issued || want_q.size() != 0 || (dump_mode && checked != longint'(n_sets) * 65536)) begin
      errs++;
      $display("[FAIL] %0d vectors issued, %0d results, %0d never came out", issued, checked, want_q.size());
    end
    if (lat != LAT) begin
      errs++;
      $display("[FAIL] latency %0d, expected %0d", lat, LAT);
    end
    if (dump_mode) $display("tfliteAdd ROUNDING=%s: %0d results of %0d constant sets dumped to %s, %0d errors, latency %0d", ROUNDING, checked, n_sets, dump, errs, lat);
    else $display("tfliteAdd ROUNDING=%s: %0d vectors from %s, %0d errors, latency %0d", ROUNDING, checked, vec, errs, lat);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 2: Write the Makefile**

```make
SHELL := /bin/bash
# tfliteAdd: verilator (corners + RANDOM vectors from ipu.tflite_add, one ROUNDING), exhaustive (every (q1, q2) at the corner sets, checked by check_add.py), oracle (G0b's models, ORACLE=dir), both, vivado_tb, vectors, lint, lint_bad.
PRJ_DIR  = $(shell pwd)
TB_DIR   = $(PRJ_DIR)/testbenches
PLUSARGS ?=
ROUNDING ?= DOUBLE
RANDOM   ?= 1000000
ORACLE   ?=
PYTHON   ?= python3
$(shell mkdir -p $(PRJ_DIR)/Verilator)  # Verilator makes --Mdir but not its parent
RLOW     = $(shell echo $(ROUNDING) | tr A-Z a-z)
V        = $(PRJ_DIR)/Verilator
DESIGN   = $(PRJ_DIR)/src/tfliteAdd.sv
PKG      = $(PRJ_DIR)/../Common/src/sienna_fmt_pkg.sv
FLAGS    = --binary --timing --assert --sv -I$(TB_DIR) --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND --Wno-WIDTHCONCAT --Wno-INITIALDLY $(EXTRA_FLAGS)
GEN      = $(PYTHON) $(TB_DIR)/gen_add_vectors.py
CHECK    = $(PYTHON) $(TB_DIR)/check_add.py
BUILD    = verilator $(FLAGS) -GROUNDING='"$(ROUNDING)"' --top-module TB_tfliteAdd --Mdir $(V)/$(RLOW) $(PKG) $(DESIGN) $(TB_DIR)/TB_tfliteAdd.sv -o sim

verilator:
	$(GEN) $(V)/vec_$(RLOW).txt --rounding $(ROUNDING) --random $(RANDOM)
	$(BUILD)
	$(V)/$(RLOW)/sim +VEC=$(V)/vec_$(RLOW).txt $(PLUSARGS)

exhaustive:
	$(GEN) $(V)/consts_corners.txt --consts corners
	$(BUILD)
	$(V)/$(RLOW)/sim +CONSTS=$(V)/consts_corners.txt +DUMP=$(V)/dump_corners_$(RLOW).txt $(PLUSARGS)
	$(CHECK) $(V)/dump_corners_$(RLOW).txt --consts $(V)/consts_corners.txt --rounding $(ROUNDING)

oracle:
	@[ -n "$(ORACLE)" ] || { echo "make oracle needs ORACLE=<G0b output directory>"; echo "RESULT: FAILED"; exit 1; }
	$(GEN) $(V)/consts_oracle.txt --consts oracle --oracle $(ORACLE)
	$(BUILD)
	$(V)/$(RLOW)/sim +CONSTS=$(V)/consts_oracle.txt +DUMP=$(V)/dump_oracle_$(RLOW).txt $(PLUSARGS)
	$(CHECK) $(V)/dump_oracle_$(RLOW).txt --consts $(V)/consts_oracle.txt --rounding $(ROUNDING) --oracle $(ORACLE)

both:
	$(MAKE) verilator ROUNDING=DOUBLE
	$(MAKE) verilator ROUNDING=SINGLE
	$(MAKE) exhaustive ROUNDING=DOUBLE
	$(MAKE) exhaustive ROUNDING=SINGLE

vivado_tb:
	for r in DOUBLE SINGLE; do l=$$(echo $$r | tr A-Z a-z); \
	  verilator $(FLAGS) -GROUNDING="\"$$r\"" --top-module TB_tfliteAdd --Mdir $(V)/viv_$$l $(PKG) $(DESIGN) $(TB_DIR)/TB_tfliteAdd.sv -o sim || exit 1; \
	  (cd $(TB_DIR) && $(V)/viv_$$l/sim $(PLUSARGS)) || exit 1; done

vectors:
	$(GEN) $(TB_DIR)/vectors_double.mem --rounding DOUBLE --random 10000
	$(GEN) $(TB_DIR)/vectors_single.mem --rounding SINGLE --random 10000

lint:
	for r in DOUBLE SINGLE; do \
	  verilator --lint-only -Wall -Wno-fatal -DSYNTHESIS --top-module tfliteAdd -GROUNDING="\"$$r\"" $(PKG) $(DESIGN) || exit 1; done

lint_bad:
	@verilator --lint-only -Wno-fatal -Werror-USERFATAL --top-module tfliteAdd -GROUNDING='"NEAREST"' $(PKG) $(DESIGN) \
	  > $(V)/lint_bad.txt 2>&1; rc=$$?; cat $(V)/lint_bad.txt; \
	  if grep -q "tfliteAdd: unsupported ROUNDING" $(V)/lint_bad.txt && [ $$rc -ne 0 ]; then echo REJECTED; \
	  else echo NOT-REJECTED; echo "RESULT: FAILED"; exit 1; fi

.PHONY: verilator exhaustive oracle both vivado_tb vectors lint lint_bad
```

- [ ] **Step 3: Run it to see it fail** (the RTL does not exist yet)

Run: `L rs1_add_tb_fail 32 1 $J/cmds/int8_aril.sh Add verilator`
Expected: the vector file is written (`vec_double.txt: ... vectors`), then Verilator stops with `%Error: ... Cannot find file containing module: 'tfliteAdd'` or `cannot find ... tfliteAdd.sv`; no `RESULT: PASSED`; `int8_aril.sh` exits 1.

- [ ] **Step 4: No commit yet** (Task 1.6 commits RTL first).

### Task 1.6: `tfliteAdd` RTL

**Files:**
- Create: `$A/Add/src/tfliteAdd.sv`
- Create: `$A/Add/testbenches/vectors_double.mem`, `vectors_single.mem` (by `make vectors`)

**Interfaces:**
- Consumes: `sienna_fmt_pkg::REQ_ROUNDING`, `qadd_lat()`.
- Produces: module `tfliteAdd` with the Global Constraints' ports; latency `qadd_lat()` = 6; `done_o` one cycle per beat; `result_o` valid with `done_o` and held otherwise (D-8: no reset on data).

- [ ] **Step 1: Write the RTL**

```systemverilog
`timescale 1ns / 100ps

// TFLite's int8 ADD (reference integer_ops/add.h), bit-exact, DOUBLE or SINGLE rounding: (q - zp) << 20 per input, scaled, summed, rescaled, + zpo, clamped; 6 stages.
module tfliteAdd #(
    parameter string ROUNDING = sienna_fmt_pkg::REQ_ROUNDING  // G0b's variant; the DV overrides it to run both
) (
    input  logic               clk_i,
    input  logic               rstn_i,
    input  logic               valid_i,
    input  logic        [7:0]  q1_i,   // input 1's code
    input  logic        [7:0]  q2_i,   // input 2's code
    input  logic signed [7:0]  zp1_i,  // the inputs' zero points
    input  logic signed [7:0]  zp2_i,
    input  logic signed [31:0] m1_i,   // Q0.31 multipliers from QuantizeMultiplierSmallerThanOneExp
    input  logic signed [31:0] m2_i,
    input  logic signed [31:0] mo_i,
    input  logic signed [7:0]  s1_i,   // their shifts, each in [-31, 0]
    input  logic signed [7:0]  s2_i,
    input  logic signed [7:0]  so_i,
    input  logic signed [7:0]  zpo_i,  // output zero point
    input  logic signed [7:0]  min_i,  // the fused activation's clamp
    input  logic signed [7:0]  max_i,
    output logic        [7:0]  result_o,
    output logic               done_o
);
  localparam bit SINGLE = (ROUNDING == "SINGLE");

  if ((ROUNDING != "SINGLE") && (ROUNDING != "DOUBLE")) begin : G_BAD_ROUNDING
    $fatal(1, "tfliteAdd: unsupported ROUNDING %s, use SINGLE or DOUBLE", ROUNDING);
  end

  // gemmlowp RoundingDivideByPOT: x / 2^e to nearest, ties away from zero; e in [0, 31].
  function automatic logic signed [31:0] rdbpot(input logic signed [31:0] x, input logic [4:0] e);
    logic [31:0] mask, thr;
    mask = (32'd1 << e) - 32'd1;
    thr  = (mask >> 1) + 32'(x[31]);
    return (x >>> e) + (((32'(x) & mask) > thr) ? 32'sd1 : 32'sd0);
  endfunction

  logic v1, v2, v3, v4, v5;
  logic signed [40:0] s1_p1, s1_p2;  // DOUBLE: SRDHM(d << 20, m), below 2^28; SINGLE: d * m
  logic [5:0] s1_e1, s1_e2, s1_eo, s2_eo, s3_eo, s4_eo;  // right shifts: -s (DOUBLE), 11 - s and 31 - so (SINGLE)
  logic signed [31:0] s1_mo, s2_mo, s3_mo;
  logic signed [7:0] s1_zpo, s1_lo, s1_hi, s2_zpo, s2_lo, s2_hi, s3_zpo, s3_lo, s3_hi, s4_zpo, s4_lo, s4_hi, s5_zpo, s5_lo, s5_hi;
  logic signed [31:0] s2_a1, s2_a2;  // the scaled inputs, below 2^28
  logic signed [29:0] s3_sum;  // their sum, below 2^29
  logic signed [61:0] s4_p;  // DOUBLE: SRDHM(sum, mo) in the low 32 bits; SINGLE: sum * mo
  logic signed [31:0] s5_r;  // the rescaled sum, before + zpo

  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {v1, v2, v3, v4, v5, done_o} <= '0;
    else {v1, v2, v3, v4, v5, done_o} <= {valid_i, v1, v2, v3, v4, v5};

  // Stage 1: d = q - zp and the 9 x 32 product; DOUBLE rounds it as SRDHM(d << 20, m) = (d m + 2^10) >>> 11 (the 2^20 cancels).
  always_ff @(posedge clk_i)
    if (valid_i) begin
      automatic logic signed [8:0] d1 = $signed({q1_i[7], q1_i}) - $signed({zp1_i[7], zp1_i});
      automatic logic signed [8:0] d2 = $signed({q2_i[7], q2_i}) - $signed({zp2_i[7], zp2_i});
      automatic logic signed [40:0] p1 = d1 * m1_i;
      automatic logic signed [40:0] p2 = d2 * m2_i;
      s1_p1  <= SINGLE ? p1 : ((p1 + 41'sd1024) >>> 11);
      s1_p2  <= SINGLE ? p2 : ((p2 + 41'sd1024) >>> 11);
      s1_e1  <= SINGLE ? 6'(8'sd11 - s1_i) : 6'(-s1_i);
      s1_e2  <= SINGLE ? 6'(8'sd11 - s2_i) : 6'(-s2_i);
      s1_eo  <= SINGLE ? 6'(8'sd31 - so_i) : 6'(-so_i);
      s1_mo  <= mo_i;
      s1_zpo <= zpo_i;
      s1_lo  <= min_i;
      s1_hi  <= max_i;
    end

  // Stage 2: each product's rounding right shift: RoundingDivideByPOT by -s (DOUBLE), (p + 2^(e - 1)) >>> e with e = 11 - s (SINGLE).
  always_ff @(posedge clk_i)
    if (v1) begin
      if (SINGLE) begin
        automatic logic signed [43:0] r1 = ($signed({{3{s1_p1[40]}}, s1_p1}) + (44'sd1 <<< (s1_e1 - 6'd1))) >>> s1_e1;
        automatic logic signed [43:0] r2 = ($signed({{3{s1_p2[40]}}, s1_p2}) + (44'sd1 <<< (s1_e2 - 6'd1))) >>> s1_e2;
        s2_a1 <= r1[31:0];
        s2_a2 <= r2[31:0];
      end else begin
        s2_a1 <= rdbpot(s1_p1[31:0], s1_e1[4:0]);
        s2_a2 <= rdbpot(s1_p2[31:0], s1_e2[4:0]);
      end
      s2_eo  <= s1_eo;
      s2_mo  <= s1_mo;
      s2_zpo <= s1_zpo;
      s2_lo  <= s1_lo;
      s2_hi  <= s1_hi;
    end

  // Stage 3: the sum, below 2^29 in magnitude, so 30 bits.
  always_ff @(posedge clk_i)
    if (v2) begin
      s3_sum <= 30'(s2_a1 + s2_a2);
      s3_eo  <= s2_eo;
      s3_mo  <= s2_mo;
      s3_zpo <= s2_zpo;
      s3_lo  <= s2_lo;
      s3_hi  <= s2_hi;
    end

  // Stage 4: the 30 x 32 product; DOUBLE takes SRDHM(sum, mo) = (sum mo + 2^30) >>> 31 (sum is never INT32_MIN, so no saturation).
  always_ff @(posedge clk_i)
    if (v3) begin
      automatic logic signed [61:0] p = s3_sum * s3_mo;
      s4_p   <= SINGLE ? p : ((p + 62'sd1073741824) >>> 31);
      s4_eo  <= s3_eo;
      s4_zpo <= s3_zpo;
      s4_lo  <= s3_lo;
      s4_hi  <= s3_hi;
    end

  // Stage 5: RoundingDivideByPOT by -so (DOUBLE), or (p + 2^(e - 1)) >>> e with e = 31 - so, then int32 (SINGLE).
  always_ff @(posedge clk_i)
    if (v4) begin
      if (SINGLE) begin
        automatic logic signed [63:0] r = ($signed({{2{s4_p[61]}}, s4_p}) + (64'sd1 <<< (s4_eo - 6'd1))) >>> s4_eo;
        s5_r <= r[31:0];
      end else s5_r <= rdbpot(s4_p[31:0], s4_eo[4:0]);
      s5_zpo <= s4_zpo;
      s5_lo  <= s4_lo;
      s5_hi  <= s4_hi;
    end

  // Stage 6: + zpo in int32, then max with min and min with max, TFLite's order.
  always_ff @(posedge clk_i)
    if (v5) begin
      automatic logic signed [31:0] y = s5_r + 32'(s5_zpo);
      if (y < 32'(s5_lo)) y = 32'(s5_lo);
      if (y > 32'(s5_hi)) y = 32'(s5_hi);
      result_o <= y[7:0];
    end

`ifndef SYNTHESIS
  always @(posedge clk_i)
    if (rstn_i && valid_i)
      assert ((s1_i >= -8'sd31) && (s1_i <= 8'sd0) && (s2_i >= -8'sd31) && (s2_i <= 8'sd0) && (so_i >= -8'sd31) && (so_i <= 8'sd0))
      else $error("tfliteAdd: shifts %0d, %0d, %0d outside [-31, 0]", s1_i, s2_i, so_i);
`endif

endmodule
```

- [ ] **Step 2: Run the vector bench in both roundings**

```bash
L rs1_add_v 32 2 $J/cmds/int8_aril.sh Add both
```

Expected in `stdout.log`, in order: `tfliteAdd ROUNDING=DOUBLE: 1009546 vectors from ..., 0 errors, latency 6` (154 x 49 corners + 2000 discriminators + 10^6 random), `RESULT: PASSED`, the same for SINGLE, then for each rounding `tfliteAdd ROUNDING=<R>: 10092544 results of 154 constant sets dumped to ...` with `RESULT: PASSED`, and `check_add: 10092544 results of 154 constant sets against ipu.tflite_add (<R>), 0 mismatches` with `RESULT: PASSED`; no `Assertion failed`; `exit=0`.

- [ ] **Step 3: Write the committed Vivado vectors and run them** (Python only on the login node; the sim on the farm)

```bash
cd $T/$A/Add && make vectors && wc -l testbenches/vectors_double.mem testbenches/vectors_single.mem
L rs1_add_viv 32 1 $J/cmds/int8_aril.sh Add vivado_tb
```

Expected: 19546 lines each (7546 corners, 2000 discriminators, 10000 random); `rs1_add_viv` prints two `RESULT: PASSED` lines.

- [ ] **Step 4: Commit (ArithmeticLibrary), RTL first**

```bash
cd $T/$A
git add Add/src/tfliteAdd.sv
git commit -m "tfliteAdd: TFLite's int8 ADD (reference integer_ops/add.h), bit-exact in DOUBLE or SINGLE rounding, 6 stages, constants taken at stage 1 with each beat: d = q - zp and the 9 x 32 products (DOUBLE rounds them as SRDHM(d << 20, m) = (d m + 2^10) >>> 11), the rounding shifts by -s, the 30-bit sum, the 30 x 32 product with SRDHM(sum, mo), the shift by -so, then + zpo, max(min), min(max). Valid bits reset, data registers not (D-8); a simulation check rejects a shift outside [-31, 0]."
git add Add/Makefile Add/testbenches/TB_tfliteAdd.sv Add/testbenches/gen_add_vectors.py Add/testbenches/check_add.py
git commit -m "Add: TB_tfliteAdd and its Makefile. Vector mode: corners, DOUBLE/SINGLE discriminators and 10^6 random vectors from ipu.tflite_add, a bubble every 64, latency qadd_lat(). Exhaustive mode: every (q1, q2) at 154 constant sets (every shift of each input and the output, the extreme multipliers, the zero-point corners, tight and inverted clamps, flushed multipliers, 32 from add_params), dumped and checked by check_add.py; make oracle replays G0b's 816 models against the interpreter's outputs. gen_add_vectors.py writes the vectors and constant sets; check_add.py --selftest. Runs rs1_add_v: both roundings, 0 errors."
git add Add/testbenches/vectors_double.mem Add/testbenches/vectors_single.mem
git commit -m "tfliteAdd vector files for the Vivado testbench (make vectors: corners, 2000 discriminators, 10000 random, each rounding). Run rs1_add_viv."
```

### Task 1.7: Gate G1 for `tfliteAdd`: replay, power-up, lint, mutants, depth

**Files:**
- Modify (local tooling): `$J/cmds/yosys_depth.sh` (a `tfliteAdd` case)
- Create (local): `$J/cmds/res_g1.sh`; mutant patches `$J/res_mut/m_ls19.patch`, `m_round.patch`, `m_clampfirst.patch`, `m_so_sign.patch`
- Modify (only if the probe says so): `$A/Add/src/tfliteAdd.sv`, `$A/Common/src/sienna_fmt_pkg.sv`

**Interfaces:**
- Consumes: Tasks 1.2 (G0b data), 1.6.
- Produces: the G1 verdict for Phase 2; the final `qadd_lat()`.

- [ ] **Step 1: Write the gate script** `$J/cmds/res_g1.sh`:

```bash
#!/bin/bash
# Gate G1 (tfliteAdd): both roundings (vectors, exhaustive corners), G0b's replay in both, random power-up, lint, lint_bad, the existing units; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs; G0B=/proj/scratch/spramanik/int8_res_data/g0b
fail() { echo "GATE-FAIL $*"; }
$J/cmds/int8_aril.sh Add both || fail both
for r in DOUBLE SINGLE; do (export ROUNDING=$r ORACLE=$G0B; TAG=oracle_$r $J/cmds/int8_aril.sh Add oracle) || fail oracle $r; done  # make takes ROUNDING and ORACLE (?=) from the environment
for s in 2 7; do
  TAG=ri$s EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" $J/cmds/int8_aril.sh Add verilator +verilator+rand+reset+2 +verilator+seed+$s || fail randinit $s
done
$J/cmds/int8_aril.sh Add vivado_tb || fail vivado_tb
$J/cmds/int8_aril.sh Add lint; L=testbenches/results/int8/Add_lint.log
command grep -E "^%Error|^%Warning-(LATCH|MULTIDRIVEN|UNOPTFLAT)" $L && fail lint
echo "lint Add: $(command grep -c '^%Warning' $L) warnings"
$J/cmds/int8_aril.sh Add lint_bad; command grep -q "^REJECTED" testbenches/results/int8/Add_lint_bad.log || fail lint_bad
$J/cmds/int8_aril.sh Common pkg || fail pkg
$J/cmds/int8_aril.sh Common ipu || fail ipu
for t in both vivado_tb; do $J/cmds/int8_aril.sh Requant $t || fail Requant $t; done
echo "units:"; command grep -h "RESULT:" testbenches/results/int8/*.log | sort | uniq -c
```

(`int8_aril.sh` passes every argument after the target to make as `PLUSARGS`, which is why the oracle's settings travel in the environment.)

- [ ] **Step 2: Run the gate**

```bash
chmod +x $J/cmds/res_g1.sh
L rs1_g1 32 8 $J/cmds/res_g1.sh
```

Expected: no `GATE-FAIL` line; `check_add: 53477376 results of 816 constant sets against G0b's interpreter outputs (...), 0 mismatches` for DOUBLE; for SINGLE `... against ipu.tflite_add (SINGLE), 0 mismatches` and `info: <n> results differ from the interpreter (DOUBLE)` with n > 0; both random power-up seeds `RESULT: PASSED`; `lint Add: <w> warnings` (record w; none of LATCH, MULTIDRIVEN, UNOPTFLAT); `REJECTED`; Requant, pkg and ipu unchanged (`RESULT: PASSED`).

- [ ] **Step 3: Write the four mutants and show each fails** (spec G1: left_shift 19, the rounding variant swapped, the clamp before + zp_o, s_o's sign flipped)

```bash
F=SystolicMesh/ArithmeticLibrary/Add/src/tfliteAdd.sv
$J/cmds/res_mkmut.sh m_ls19 $F "41'sd1024) >>> 11" "41'sd2048) >>> 12" "8'sd11 - s" "8'sd12 - s"
$J/cmds/res_mkmut.sh m_round $F 'localparam bit SINGLE = (ROUNDING == "SINGLE");' 'localparam bit SINGLE = (ROUNDING == "DOUBLE");'
$J/cmds/res_mkmut.sh m_clampfirst $F "automatic logic signed [31:0] y = s5_r + 32'(s5_zpo);" "automatic logic signed [31:0] y = s5_r;" \
  "result_o <= y[7:0];" "result_o <= 8'(y + 32'(s5_zpo));"
$J/cmds/res_mkmut.sh m_so_sign $F "6'(8'sd31 - so_i) : 6'(-so_i)" "6'(8'sd31 + so_i) : 6'(so_i)"
for m in m_ls19 m_round m_clampfirst m_so_sign; do L rs1_mut_$m 32 2 $J/cmds/res_mut.sh $J/res_mut/$m.patch $J/cmds/int8_aril.sh Add both; done
```

Expected for each: `res_mut: applied <m>.patch`, then at least one `RESULT: FAILED` with `[FAIL] vector ...` or `check_add: ... <m> mismatches` (m > 0), and `exit=1`. `m_round` must fail in both roundings (each build computes the other variant). A mutant that passes means the bench cannot see that bug: stop and strengthen the vectors before going on.

- [ ] **Step 4: Logic-depth probe**

Add to `$J/cmds/yosys_depth.sh`'s `case $unit in` list: `  tfliteAdd)      files="$PKG $A/Add/src/tfliteAdd.sv" ;;`

```bash
L rs1_yd_add_d 32 2 bash $J/cmds/yosys_depth.sh tfliteAdd ROUNDING=DOUBLE generic
L rs1_yd_add_s 32 2 bash $J/cmds/yosys_depth.sh tfliteAdd ROUNDING=SINGLE generic
command grep -h "length=" $J/runs/rs1_yd_add_[ds]/results/yd/*/ltp.txt $J/runs/rs1_base_yd_rq_[ds]/results/yd/*/ltp.txt
```

Expected: each `tfliteAdd` length (the deepest register-to-register path, i.e. the worst stage) at or below `tfliteRequant`'s in the same rounding (the spec's target: no stage deeper than `tfliteRequant`'s worst, about 46 levels by stage probes, 52-54 as a whole unit). Record all four numbers in the ledger. If `tfliteAdd`'s is deeper: find the deepest stage with stage probes (copy one stage's combinational logic between two registers, as `cmds/yd_probe/req_stage.sv` did for the requantize split), cut it into two stages, set `qadd_lat()` to 7 (Task 1.3's TB expectation too), rerun Steps 2 and 3, and commit the split as its own RTL commit with the probe numbers in its message. Otherwise `qadd_lat()` stays 6.

- [ ] **Step 5: Gate report** (`.log`, local until Task 2.11 copies it into the report history): write `$J/runs/rs1_g1/g1_add_gate.log` with the run names, the pass lines quoted from Step 2, the four mutant verdicts, the four depth numbers and `GATE G1 (tfliteAdd): PASS`. No commit (nothing in the tree changed unless Step 4 split a stage).

### Task 1.8: ArithmeticLibrary in GPNAE and SystolicMesh; G3 by diff

**Files:**
- Modify: `GPNAE/ArithmeticLibrary` (fast-forward), `GPNAE` (pointer commit), `SystolicMesh` (pointer commit)

**Interfaces:**
- Produces: both ArithmeticLibrary checkouts at the same commit (`AH`, the head after Task 1.7); G3's verdict.

- [ ] **Step 1: Fast-forward GPNAE's copy and check both heads**

```bash
AH=$(git -C $T/$A rev-parse HEAD)
git -C $T/GPNAE/ArithmeticLibrary fetch $T/$A int8_residual && git -C $T/GPNAE/ArithmeticLibrary merge --ff-only FETCH_HEAD
[ "$(git -C $T/GPNAE/ArithmeticLibrary rev-parse HEAD)" = "$AH" ] && echo SAME || echo DIFFERENT
```

Expected: `SAME`.

- [ ] **Step 2: G3, the mesh does not change** — the only SystolicMesh change is its library pointer, and the library diff is additive:

```bash
git -C $T/SystolicMesh status --short          # only ' M ArithmeticLibrary'
git -C $T/$A diff --stat 1d04a26 HEAD           # Add/*, Common/models/ipu.py, test_ipu.py, sienna_fmt_pkg.sv, TB_sienna_fmt_pkg.sv only
git -C $T/$A diff 1d04a26 HEAD -- Common/src/sienna_fmt_pkg.sv | command grep '^[-+][^-+]'   # only the qadd_lat lines, all '+'
git -C $T/$A diff --quiet 1d04a26 HEAD -- Adders Multipliers Requant Converters Divider Common/src/credit_*.sv && echo UNITS-UNCHANGED
```

Expected: as the comments say, and `UNITS-UNCHANGED`. G3 does not rerun (spec); its verdict line goes into the Task 1.13 report: `G3: mesh unchanged (SystolicMesh tree: pointer only; ArithmeticLibrary 1d04a26..<AH>: Add/ added, ipu.py and sienna_fmt_pkg.sv extended, no unit the mesh builds changed)`.

- [ ] **Step 3: Commit the pointers**

```bash
git -C $T/GPNAE add ArithmeticLibrary && git -C $T/GPNAE commit -m "Bump ArithmeticLibrary 1d04a26 -> $(git -C $T/$A rev-parse --short HEAD): tfliteAdd (TFLite's int8 ADD) and its bench, ipu.tflite_add / add_params / quantize_multiplier_smaller_than_one_exp, sienna_fmt_pkg::qadd_lat(). Nothing GPNAE builds changes."
git -C $T/SystolicMesh add ArithmeticLibrary && git -C $T/SystolicMesh commit -m "Bump ArithmeticLibrary 1d04a26 -> $(git -C $T/$A rev-parse --short HEAD): tfliteAdd (TFLite's int8 ADD) and its bench, ipu's ADD model, sienna_fmt_pkg::qadd_lat(). No unit the mesh builds changes."
```

### Task 1.9: `gpnae_model`: the sigmoid ladder

**Files:**
- Modify: `GPNAE/gpnae_model.py` (the int8 section only: constants after `THRESH`, `sig_ladder`, `LaneInt8`)
- Test: `GPNAE/check_gpnae_model_int8.py`

**Interfaces:**
- Produces:
  - `gpnae_model.SIG_LAD_X0 = 5120`; `gpnae_model.SIG_LAD_T` (the 19-tuple of the Global Constraints).
  - `gpnae_model.sig_ladder(x, t=SIG_LAD_T) -> np.ndarray` (int64 codes in [-128, 127]).
  - `LaneInt8(f, rom, sets=None, thresh=None, ladder=True, lad_x0=SIG_LAD_X0)`; `ladder=False` is the 2a lane (sigmoid saturates past `thresh[2]` = 7168), which `fit_poly_coeffs` scores tables on (P-4). `gpnae_model.Lane(INT8, rom, ladder=False)` reaches it through `Lane.__new__`.

- [ ] **Step 1: Write the failing checks** — append to `check_gpnae_model_int8.py` before the summary (add `import os` at the top):

```python
ROM = gm.read_rom(os.path.join(gm.ROOT, gm.coeff_file(gm.INT8)))
lad, off = gm.Lane(gm.INT8, ROM), gm.Lane(gm.INT8, ROM, ladder=False)  # the lane as built, and 2a's
eq("ladder constants: 19 rising codes from 5228 to 12773, from x = 5120",
   (gm.SIG_LAD_X0, len(gm.SIG_LAD_T), gm.SIG_LAD_T[0], gm.SIG_LAD_T[-1], int(all(np.diff(gm.SIG_LAD_T) > 0))), (5120, 19, 5228, 12773, 1))
eq("sig_ladder at its edges and the Q4.11 rail",
   gm.sig_ladder(np.array([5120, 5227, 5228, 12772, 12773, 32767, -5120, -5227, -5228, -12772, -12773, -32768])),
   [109, 109, 110, 127, 127, 127, -109, -109, -110, -127, -128, -128])
dense = gm.Case(1 / 2048, 0, 1 / 256, -128, True)  # identity rescale, sigmoid's fixed output quantization
X = np.arange(-32768, 32768, dtype=np.int64)
p_id = gm.int8_params(dense, 2)
eq("the dense sweep's rescale is the identity", int(np.sum(gm.rescale(X, 0, p_id.mx, p_id.shx) != X)), [0])
ax = np.abs(X) >= gm.SIG_LAD_X0
eq("ladder = exact_int8 at every |x| >= 2.5", int(np.sum(gm.sig_ladder(X[ax]) != gm.exact_int8(X[ax], 2, dense))), [0])
y_lad, y_off = lad.run(X, 2, p_id), off.run(X, 2, p_id)
eq("the lane takes the ladder from x = +-5120 (5119 is the polynomial's)",
   [int(y_lad[X == v][0] == gm.sig_ladder(np.array([v]))[0]) for v in (5120, -5120)] + [int(y_lad[X == v][0] == y_off[X == v][0]) for v in (5119, -5119)],
   [1, 1, 1, 1])
eq("below 2.5 the lane is 2a's polynomial, unchanged", int(np.sum(y_lad[~ax] != y_off[~ax])), [0])
a = gm.accuracy_int8(y_lad, X, 2, dense)
eq("sigmoid dense with the ladder: 0 of 65536 outside, worst 2 LSB", (a.fail, a.worst_lsb), (0, 2))
eq("sigmoid dense without it: 2a's 3370 outside", (gm.accuracy_int8(y_off, X, 2, dense).fail,), (3370,))
q = np.arange(-128, 128, dtype=np.int64)
gated = [c for c in gm.INT8_CASES["sigmoid"] if c.gated]
g = gm.merge_acc(gm.accuracy_int8(lad.run(q, 2, gm.int8_params(c, 2)), q, 2, c) for c in gated)
eq("sigmoid gated with the ladder: 0 of 1280 outside", (g.fail, 256 * len(gated)), (0, 1280))
late = gm.Lane(gm.INT8, ROM, lad_x0=7168)  # the spec's rejected start at 3.5
eq("a ladder from 3.5 leaves 24 dense misses (spec)", (gm.accuracy_int8(late.run(X, 2, p_id), X, 2, dense).fail,), (24,))
for code, name in ((1, "selu"), (3, "tanh")):
    for c in gm.INT8_CASES[name]:
        eq(f"{name} unchanged by the ladder ({c.s_in:.4f}, {c.z_in})",
           int(np.sum(lad.run(q, code, gm.int8_params(c, code)) != off.run(q, code, gm.int8_params(c, code)))), [0])
```

- [ ] **Step 2: Run them to see them fail** (login node, numpy, seconds)

Run: `cd $T/GPNAE && python3 check_gpnae_model_int8.py`
Expected: `AttributeError: module 'gpnae_model' has no attribute 'SIG_LAD_X0'` (or `TypeError: ... unexpected keyword argument 'ladder'`).

- [ ] **Step 3: Implement** — in `gpnae_model.py` after `THRESH = ...`:

```python
SIG_LAD_X0 = 5120  # sigmoid from |x| >= 2.5 (Q4.11) is the exact ladder below, not the polynomial
SIG_LAD_T = (5228, 5350, 5479, 5616, 5761, 5916, 6082, 6261, 6456, 6669, 6905, 7170, 7471, 7821, 8240, 8763, 9460, 10515,
             12773)  # where round(256 sigmoid(-|x|)) steps; fit_poly_coeffs.sig_ladder_thresholds() derives them
```

after `quant_tanh`:

```python
def sig_ladder(x, t=SIG_LAD_T):
    """sigmoid for |x| >= SIG_LAD_X0: k = len(t) - #{t_i <= |x|} = round(256 sigmoid(-|x|)); k - 128 for x < 0, min(127, 128 - k) for x > 0."""
    x = np.asarray(x, dtype=np.int64)
    ax = np.minimum(np.abs(x), 32767)
    k = len(t) - (ax[..., None] >= np.asarray(t, dtype=np.int64)).sum(axis=-1)
    return np.where(x < 0, k - 128, np.minimum(127, 128 - k))
```

and in `LaneInt8`:

```python
class LaneInt8(Lane):
    """Bit-exact model of gpnae_poly_int8; ladder=False is the 2a lane (sigmoid saturated past 3.5), which fit_poly_coeffs scores tables on."""

    def __init__(self, f, rom, sets=None, thresh=None, ladder=True, lad_x0=SIG_LAD_X0):
        assert lad_x0 >= SIG_LAD_X0, "the ladder's thresholds start at x = 2.5"
        self.f = f
        self.rom = [v - (1 << 16) if v >= (1 << 15) else v for v in rom]
        self.sets = {**SETS_INT8, **(sets or {})}
        self.thresh = {**THRESH, **(thresh or {})}
        self.ladder, self.lad_x0 = ladder, lad_x0
```

and the sigmoid branch of `run`:

```python
        if code == 2:
            y = quant_sig(np.where(neg, ONE_Q11 - p, p))
            if self.ladder:  # |x| >= 2.5: the exact ladder, gpnae_poly_int8's sat_buf for sigmoid
                return np.where(np.abs(x) >= self.lad_x0, sig_ladder(x), y)
            sat = np.abs(x) > self.thresh[2]
            return np.where(sat, np.where(neg, -128, 127), y)
```

- [ ] **Step 4: Run them to see them pass**

Run: `cd $T/GPNAE && python3 check_gpnae_model_int8.py`
Expected: `check_gpnae_model_int8: 0 errors`, `RESULT: PASSED`. If "a ladder from 3.5 leaves 24" or "3370" does not reproduce, do not change the expectation: report the measured number against the spec's to Soham (the spec's figures came from this model).

- [ ] **Step 5: No commit yet** (Task 1.12 commits RTL, then the model).

### Task 1.10: `fit_poly_coeffs.py`: the thresholds from sigmoid, the table unchanged

**Files:**
- Modify: `GPNAE/fit_poly_coeffs.py`
- Test: `GPNAE/check_gpnae_model_int8.py`
- Create (local): `$J/cmds/res_fit.sh`

**Interfaces:**
- Consumes: Task 1.9's `SIG_LAD_X0`, `SIG_LAD_T`, `Lane(..., ladder=False)`.
- Produces: `fit_poly_coeffs.sig_ladder_thresholds(x0=SIG_LAD_X0) -> tuple[int, ...]` (raises `RuntimeError` on a step larger than 1, a value at x0 other than the count, or a rounding tie); `lane_for(code, coeffs)` builds the 2a lane; `main_int8` reports the ladder and takes its verdict from the lane as built.

- [ ] **Step 1: Write the failing check** — in `check_gpnae_model_int8.py` (add `import fit_poly_coeffs as fpc` beside `import gpnae_model as gm`):

```python
eq("the ladder's thresholds, computed from sigmoid, are gpnae_model's", fpc.sig_ladder_thresholds(), gm.SIG_LAD_T)
```

Run: `cd $T/GPNAE && python3 check_gpnae_model_int8.py` → `AttributeError: module 'fit_poly_coeffs' has no attribute 'sig_ladder_thresholds'`.

- [ ] **Step 2: Implement** — in `fit_poly_coeffs.py` after `NAME_INT8 = ...`:

```python
def sig_ladder_thresholds(x0=gpnae_model.SIG_LAD_X0):
    """The Q4.11 codes above x0 where round(256 sigmoid(-|x|)) (exact_int8's half-up rounding) steps down; each step 1, no input a tie."""
    c = np.arange(x0, 32768, dtype=np.int64)
    y = 256.0 / (1.0 + np.exp(c / 2048.0))
    v = np.floor(y + 0.5).astype(np.int64)
    steps = v[:-1] - v[1:]
    t = tuple(int(x) for x in c[1:][steps == 1])
    tie = float(np.abs(y - np.floor(y) - 0.5).min())
    if steps.max() > 1 or steps.min() < 0 or v[0] != len(t) or v[-1] != 0 or tie < 1e-9:
        raise RuntimeError(f"sigmoid ladder from {x0}: steps {steps.min()}..{steps.max()}, k(x0) {v[0]} for {len(t)} thresholds, "
                           f"k(32767) {v[-1]}, nearest tie {tie:.3g} LSB")
    return t
```

Change `lane_for` to score on the 2a lane:

```python
def lane_for(code, coeffs):
    """The 2a lane (no sigmoid ladder, P-4 of the residual plan) with one coefficient set at base 0 and the model's thresholds."""
    rom = [0] * 32
    rom[:len(coeffs)] = coeffs
    return gpnae_model.Lane(gpnae_model.INT8, [v & 0xFFFF for v in rom], sets={code: (0, len(coeffs) - 1)}, ladder=False)
```

In `main_int8`, make the whole-table lane the 2a lane too (its figures must equal the candidates'): `lane = gpnae_model.Lane(gpnae_model.INT8, [v & 0xFFFF for v in table], sets=lay, ladder=False)`. Then, after the grid section and before the `VERDICT` line, add the lane as built and take the verdict from it:

```python
    built = gpnae_model.Lane(gpnae_model.INT8, [v & 0xFFFF for v in table], sets=lay)  # the lane as built: sigmoid's ladder on
    t = sig_ladder_thresholds()
    L.append("")
    L.append(f"sigmoid for |x| >= {gpnae_model.SIG_LAD_X0 / 2048:.1f}: the exact ladder, thresholds from sigmoid "
             f"({'equal to' if t == gpnae_model.SIG_LAD_T else 'DIFFERENT FROM'} gpnae_model.SIG_LAD_T): {list(t)}")
    dn = measure_dense(built, 2)
    gr = measure_cases(built, 2)
    L.append(f"sigmoid as built: dense {dn.fail} of {n_dense[2]} outside, worst {dn.worst_lsb} LSB; grid {gr.fail} of {n_grid[2]} outside")
    missed = [NAME_INT8[c] for c in (1, 3) if not chosen[c].dense.ok] + (["sigmoid"] if not dn.ok else [])
    if t != gpnae_model.SIG_LAD_T:
        missed.append("sigmoid (ladder thresholds)")
```

(the `missed` line replaces the earlier one, which moves here; the `VERDICT` and `OPTIONS_INT8` lines after it stay as they are).

- [ ] **Step 3: Run the check to see it pass** — `cd $T/GPNAE && python3 check_gpnae_model_int8.py` → `0 errors`, `RESULT: PASSED`.

- [ ] **Step 4: The fit must still write the committed table** — `$J/cmds/res_fit.sh`:

```bash
#!/bin/bash
# Runs fit_poly_coeffs.py --format int8 on a snapshot and compares both table copies with the committed one; run from a snapshot root.
R=$(pwd)/testbenches/results/res/fit; mkdir -p $R
cp GPNAE/poly_coeffs_int8.mem $R/committed.mem
(cd GPNAE && python3 fit_poly_coeffs.py --format int8 --report $R/poly_coeffs_int8_fit.log); rc=$?
for f in GPNAE/poly_coeffs_int8.mem GPNAE/src/TYTAN/Memory/poly_coeffs_int8.mem; do
  command cmp -s $f $R/committed.mem && echo "FIT-TABLE $f: IDENTICAL" || echo "FIT-TABLE $f: DIFFERENT"
done
echo "fit exit=$rc"
```

```bash
chmod +x $J/cmds/res_fit.sh && L rs1_fit 32 6 $J/cmds/res_fit.sh
```

Expected: both `FIT-TABLE ...: IDENTICAL`; `fit exit=0`; the report's ladder lines read `equal to gpnae_model.SIG_LAD_T` and `sigmoid as built: dense 0 of 65536 outside, worst 2 LSB; grid 0 of 1280 outside`; its `VERDICT: TOLERANCE MET for every activation on the dense sweep`. A `DIFFERENT` table is a stop: the fit no longer reproduces the published int8 table; report it to Soham.

- [ ] **Step 5: No commit yet** (Task 1.12).

### Task 1.11: GPNAE regression: ladder edge sets, a scale sweep, sigmoid as a hard gate (the failing test)

**Files:**
- Modify: `GPNAE/regression.py` (`run_int8` and new helpers beside `pos_hit`; nothing in the `make regression` section `regression_sweep` adds at the end)

**Interfaces:**
- Consumes: `gm.SIG_LAD_T`, `gm.SIG_LAD_X0`, `gm.rescale_params`, `gm.Int8Params`.
- Produces: `N_LAD_INT8 = 41`, `N_SWEEP_INT8 = 64`, `GATED_INT8 = ("sigmoid",)`, `ladder_hit(x) -> (mx, shx, d)`, `ladder_sets() -> list[Int8Params]`, `sweep_sets(rng) -> list[Int8Params]`; per activation 206 sets x 256 inputs (5 cases, 32 random, 64 positive SELU, 41 ladder edge, 64 sweep); `ACCURACY:` reads `FAIL` and `run_int8` returns 1 when a `GATED_INT8` activation has a gated output outside the tolerance.

- [ ] **Step 1: Add the helpers** after `pos_hit`:

```python
N_LAD_INT8 = 41  # sigmoid ladder edge sets: t_i - 1 and t_i for the 19 thresholds, the hand-over 5119 and 5120, a rescale saturating at -32768
N_SWEEP_INT8 = 64  # the scale sweep: s_in log-spaced 1/128 to 1/8, every int8 input; bit-exact only
GATED_INT8 = ("sigmoid",)  # activations whose gated accuracy fails the run; tanh and SELU stay reported


def ladder_hit(x):
    """(mx, shx, d): with z_in = 0, input d rescales to x and input -d to -x exactly, d in 1..127."""
    for shx in range(15, -1, -1):
        h = (1 << shx) >> 1
        for d in range(127, 0, -1):
            mx = -(-(x * (1 << shx) - h) // d)  # the smallest mx whose rounded rescale of d reaches x
            if mx <= 32767 and (d * mx + h) >> shx == x and (-d * mx + h) >> shx == -x:
                return mx, shx, d
    raise ValueError(f"no int8 input pair rescales to +-{x}")


def ladder_sets():
    """Two rescales per threshold (codes t_i - 1 and t_i, both signs), the hand-over 5119 and 5120, and q = -128 at x = -32768."""
    import gpnae_model as gm
    xs = [v for t in gm.SIG_LAD_T for v in (t - 1, t)] + [gm.SIG_LAD_X0 - 1, gm.SIG_LAD_X0]
    return [gm.Int8Params(*ladder_hit(x)[:2], 0, 0, 0, 0) for x in xs] + [gm.Int8Params(32767, 0, 0, 0, 0, 0)]


def sweep_sets(rng):
    """The scale sweep: 64 rescales, s_in log-spaced from 1/128 to 1/8, z_in random in [-64, 64]."""
    import gpnae_model as gm
    return [gm.Int8Params(*gm.rescale_params(float(s)), int(rng.integers(-64, 65)), 0, 0, 0) for s in np.geomspace(1 / 128, 1 / 8, N_SWEEP_INT8)]
```

- [ ] **Step 2: Use them in `run_int8`** — replace the batch arithmetic and extend the per-activation loop:

```python
    pos_batches = (n_cases + N_RAND_INT8 + N_POS_INT8) * 256 // per
    lad_batches = pos_batches + N_LAD_INT8 * 256 // per
    batches = lad_batches + N_SWEEP_INT8 * 256 // per  # cases, random, positive SELU, ladder edges, the scale sweep
    lad_sets, sweep = ladder_sets(), sweep_sets(np.random.default_rng([args.seed, 16]))
    assert len(lad_sets) == N_LAD_INT8
    rl = np.random.default_rng([args.seed, 15])  # their input order: a new stream, so the earlier sets' stimulus is unchanged
    extra_q = [rl.permutation(256).astype(np.int64) - 128 for _ in range(N_LAD_INT8 + N_SWEEP_INT8)]
```

and after the `for p, q in zip(pos_sets, pos_q):` block inside `for act, code in INT8_ACTS:`:

```python
        for p, q in zip(lad_sets + sweep, extra_q):
            stim += [int(v) for v in q]
            gold += [int(v) for v in lane.run(q, code, p)]
            par += [p] * (256 // per)
```

Count wrong outputs per segment: `bad = {act: [0] * 5 for act, _ in INT8_ACTS}` and
`bad[m.group(1)][sum(int(m.group(2)) >= b for b in (case_batches, rand_batches, pos_batches, lad_batches))] += int(m.group(3))`.
Add to the report after the positive SELU line:

```python
         f"  sigmoid ladder edge sets ({N_LAD_INT8} per activation: t_i - 1 and t_i for each threshold, 5119 and 5120, both signs, and a "
         f"rescale saturating at -32768; bit-exact only): " + "  ".join(part(a, 3, N_LAD_INT8 * 256) for a, _ in INT8_ACTS),
         f"  scale sweep ({N_SWEEP_INT8} per activation, s_in 1/128 to 1/8; bit-exact only): "
         + "  ".join(part(a, 4, N_SWEEP_INT8 * 256) for a, _ in INT8_ACTS),
```

Replace the `ACCURACY:` line and the return:

```python
    n_gated = {act: 256 * sum(c.gated for c in gm.INT8_CASES[act]) for act in tot}
    gated_miss = [a for a in GATED_INT8 if a in missed]
    L.append(f"ACCURACY: {'FAIL' if gated_miss else 'PASS' if not missed else 'MISSED'} ({tol}; gated: "
             + ", ".join(f"{a} {tot[a].fail} of {n_gated[a]} outside" for a in GATED_INT8) + "; reported, not gated: "
             + ", ".join(f"{a} {tot[a].fail} of {n_gated[a]} outside" for a in tot if a not in GATED_INT8) + ")")
```

`return 0 if exact_ok and not gated_miss else 1`, and add `"gated": list(GATED_INT8)` to the JSON dump.

- [ ] **Step 3: Run it on the unchanged RTL to see it fail**

Run: `L rs1_gp_fail 32 2 $J/cmds/res_gpnae.sh --lane poly --format int8`
Expected: `BITEXACT: FAIL ... sigmoid <n>/52736` with n < 52736 (the RTL saturates or uses the polynomial where the golden now takes the ladder), selu, tanh, relu, linear `52736/52736`; the ladder edge segment shows sigmoid mismatches; `ACCURACY: PASS (...; gated: sigmoid 0 of 1280 outside; ...)` (the golden is the model's); `exit=1`.

- [ ] **Step 4: Confirm the earlier sets' stimulus did not move** — against `rs1_base_gp_int8`:

```bash
B=$J/runs/rs1_base_gp_int8/results/res/gpnae/stimulus; N=$J/runs/rs1_gp_fail/results/res/gpnae/stimulus
for a in selu sigmoid tanh relu linear; do for k in in par; do
  n=$(wc -l < $B/${a}_$k.mem)  # the base file holds exactly the 101 earlier sets
  command cmp -s $B/${a}_$k.mem <(head -n $n $N/${a}_$k.mem) && echo "$a $k SAME ($n lines)" || echo "$a $k MOVED"
done; done
```

Expected: every line `SAME` (the `_in.mem` files 25,856 lines = (5 + 32 + 64) x 256 inputs).

- [ ] **Step 5: No commit yet** (Task 1.12).

### Task 1.12: `gpnae_poly_int8`: the ladder in G_POST

**Files:**
- Modify: `GPNAE/src/gpnae_poly_int8.sv`
- Test: `GPNAE/check_gpnae_model_int8.py` (the RTL literals)

**Interfaces:**
- Consumes: the model's `SIG_LAD_X0`, `SIG_LAD_T`.
- Produces: `localparam logic signed [W-1:0] SIG_LAD_X0 = 16'sd5120;`, `localparam int SIG_LAD_N = 19;`, `localparam logic [W-1:0] SIG_LAD_T[SIG_LAD_N] = '{...};`, `function automatic logic [7:0] sig_ladder(input logic [W-1:0] x);`; sigmoid's `sat_buf` means |x| >= 5120; no new state, no new cycle; `T_SIG` removed (unused).

- [ ] **Step 1: Write the failing literal check** — in `check_gpnae_model_int8.py` (add `import re`):

```python
def rtl_ladder(path=os.path.join(gm.ROOT, "src", "gpnae_poly_int8.sv")):
    """gpnae_poly_int8's SIG_LAD_X0 and SIG_LAD_T literals."""
    s = open(path).read()
    x0 = re.findall(r"localparam\s+logic\s+signed\s+\[W-1:0\]\s+SIG_LAD_X0\s*=\s*16'sd(\d+);", s)
    m = re.search(r"SIG_LAD_T\s*\[SIG_LAD_N\]\s*=\s*'\{([^}]*)\}", s)
    return (int(x0[0]) if len(x0) == 1 else None, tuple(int(v) for v in re.findall(r"16'd(\d+)", m.group(1))) if m else None)


eq("gpnae_poly_int8's ladder literals are gpnae_model's", rtl_ladder(), (gm.SIG_LAD_X0, gm.SIG_LAD_T))
```

Run: `cd $T/GPNAE && python3 check_gpnae_model_int8.py` → `[FAIL] gpnae_poly_int8's ladder literals ...: got (None, None)`, `RESULT: FAILED`.

- [ ] **Step 2: Implement** — in `gpnae_poly_int8.sv`, replace the `T_SIG` line:

```systemverilog
  localparam logic signed [W-1:0] SIG_LAD_X0 = 16'sd5120;  // sigmoid |x| >= 2.5: the exact ladder below, no polynomial (gpnae_model.SIG_LAD_X0)
  localparam int SIG_LAD_N = 19;
  localparam logic [W-1:0] SIG_LAD_T[SIG_LAD_N] = '{16'd5228, 16'd5350, 16'd5479, 16'd5616, 16'd5761, 16'd5916, 16'd6082, 16'd6261,
                                                    16'd6456, 16'd6669, 16'd6905, 16'd7170, 16'd7471, 16'd7821, 16'd8240, 16'd8763,
                                                    16'd9460, 16'd10515, 16'd12773};  // where round(256 sigmoid(-|x|)) steps (fit_poly_coeffs)
```

after `quant_sig`:

```systemverilog
  // sigmoid for |x| >= 2.5: k = 19 - #{t <= |x|} is round(256 sigmoid(-|x|)); k - 128 for x < 0, min(127, 128 - k) for x > 0.
  function automatic logic [7:0] sig_ladder(input logic [W-1:0] x);
    logic [W-1:0] ax;
    logic [4:0] k;
    ax = abs_sat(x);
    k  = 5'(SIG_LAD_N);
    for (int i = 0; i < SIG_LAD_N; i++) if (ax >= SIG_LAD_T[i]) k = k - 5'd1;
    if (x[W-1]) return {3'b000, k} - 8'd128;
    return (k == 5'd0) ? 8'h7F : 8'd128 - {3'b000, k};
  endfunction
```

`x_sat`'s sigmoid term:

```systemverilog
               : is_sig  ? (($signed(x_new) >= SIG_LAD_X0) || ($signed(x_new) <= -SIG_LAD_X0))
```

and G_POST's write (line 475):

```systemverilog
          res_buf[iss_idx[SW-1:0]] <= sat_buf[iss_idx[SW-1:0]] ? (is_sig ? sig_ladder(x_cur) : (neg_buf[iss_idx[SW-1:0]] ? 8'h80 : 8'h7F))
                                                               : quant_sig(y_s);
```

- [ ] **Step 3: Run the checks and the lane regression to see them pass**

```bash
cd $T/GPNAE && python3 check_gpnae_model_int8.py | tail -2      # 0 errors, RESULT: PASSED
L rs1_gp_int8 32 2 $J/cmds/res_gpnae.sh --lane poly --format int8
L rs1_gp_fp32 32 2 $J/cmds/res_gpnae.sh --lane poly --format fp32
L rs1_gp_bf16 32 2 $J/cmds/res_gpnae.sh --lane poly --format bf16 --model hw
```

Expected: `rs1_gp_int8`: `BITEXACT: PASS  selu 52736/52736  sigmoid 52736/52736  tanh 52736/52736  relu 52736/52736  linear 52736/52736  stream <n>/<n>`, every segment `x/x`, `ACCURACY: PASS (rel <= 6.25% or abs <= 1 LSB; gated: sigmoid 0 of 1280 outside; reported, not gated: selu 0 of 1024 outside, tanh 0 of 1280 outside)`, `exit=0`; fp32 and bf16: `Verdict : All patterns passed`, `exit=0`.

- [ ] **Step 4: Lint the lane** — `L rs1_gp_lint 32 1 $J/cmds/res_lint_gp.sh`, then compare with Task 1.0's reference:

```bash
for r in rs1_base_gp_lint rs1_gp_lint; do echo "$r: $(command grep -c '^%Warning' $J/runs/$r/stdout.log) warnings"; done
command grep -E "^%Warning.*(SIG_LAD|sig_ladder)" $J/runs/rs1_gp_lint/stdout.log
```

Expected: the same count or fewer (`T_SIG` is gone) and no warning naming `SIG_LAD` or `sig_ladder`; `make exit=0, errors=0`.

- [ ] **Step 5: Commit (GPNAE), RTL first**

```bash
cd $T/GPNAE
git add src/gpnae_poly_int8.sv
git commit -m "gpnae_poly_int8: sigmoid for |x| >= 2.5 (Q4.11 5120) is an exact 19-breakpoint ladder: k = 19 - #{t_i <= |x|} over the codes where round(256 sigmoid(-|x|)) steps (5228 .. 12773), output k - 128 for x < 0 and min(127, 128 - k) for x > 0, exact because sigmoid(x) = 1 - sigmoid(-x) and no Q4.11 input is a rounding tie (the nearest is 2e-5 LSB). sat_buf means |x| >= 5120 for sigmoid and G_POST writes the ladder there; no state, no cycle; T_SIG goes (unused). The table, barrel_mac, the float lanes and gpnae.sv are unchanged."
git add gpnae_model.py check_gpnae_model_int8.py
git commit -m "gpnae_model: LaneInt8 takes the sigmoid ladder (SIG_LAD_X0, SIG_LAD_T, sig_ladder) as gpnae_poly_int8 does; ladder=False keeps the 2a lane. check_gpnae_model_int8: the ladder's edges and the Q4.11 rail, the ladder equal to exact_int8 at every |x| >= 2.5, the polynomial below unchanged, sigmoid dense 0 of 65536 and gated 0 of 1280 outside (3370 without the ladder, 24 with a ladder from 3.5), SELU and tanh unchanged, and the RTL's literals equal to the model's."
git add fit_poly_coeffs.py
git commit -m "fit_poly_coeffs: sig_ladder_thresholds() derives the ladder's thresholds from sigmoid (each step one code, no rounding tie), and check_gpnae_model_int8 fails if they differ from the model's or the RTL's; the int8 fit keeps scoring tables on the 2a lane, so it still writes poly_coeffs_int8.mem byte for byte (run rs1_fit), and its verdict now reads the lane as built (sigmoid dense 0 of 65536 outside)."
git add regression.py
git commit -m "regression.py (int8): per activation 41 ladder edge sets (codes t_i - 1 and t_i of each threshold and 5119 / 5120 on both signs, a rescale saturating at -32768) and a 64-scale sweep (s_in 1/128 to 1/8), bit-exact against the lane model; sigmoid's gated accuracy is a hard gate (ACCURACY: FAIL and exit 1 when any of its 1280 gated outputs leaves the tolerance), tanh and SELU stay reported. The earlier sets' stimulus is byte-identical. Runs rs1_gp_fail (the 2a RTL fails sigmoid) and rs1_gp_int8 (PASS)."
```

### Task 1.13: Gate G2, mutants, identity checks; Phase 1 closes

**Files:**
- Create (local): `$J/cmds/res_g2.sh`; mutant patches `$J/res_mut/m_thr.patch`, `m_x0.patch`, `m_pos.patch`
- Output (local): `$J/runs/rs1_g2/g2_ladder_gate.log`

**Interfaces:**
- Consumes: Tasks 1.8 to 1.12, `rs1_base_*`.
- Produces: G2's verdict; Phase 1's end state (all four repos committed on `int8_residual`, nothing pushed).

- [ ] **Step 1: The gate script** `$J/cmds/res_g2.sh`:

```bash
#!/bin/bash
# Gate G2 (sigmoid ladder): the int8 lane at four seeds and random power-up, fp32 / bf16 lanes, barrel_mac, the model check, lint; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
J=/proj/work/spramanik/sienna_jobs; R=$(pwd)/testbenches/results/res/g2; mkdir -p $R
run() { tag=$1; shift; $J/cmds/res_gpnae.sh "$@" > $R/$tag.txt 2>&1 || echo "GATE-FAIL $tag"; mkdir -p $R/$tag; cp -r testbenches/results/res/gpnae/. $R/$tag/; }
for s in 1 101 202 303; do run int8_seed$s --lane poly --format int8 --seed $s; done
export EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" SIM_ARGS="+verilator+rand+reset+2"
run int8_randinit --lane poly --format int8 --seed 7
unset EXTRA_FLAGS SIM_ARGS
run fp32 --lane poly --format fp32
run bf16_hw --lane poly --format bf16 --model hw
$J/cmds/int8_bm.sh 5 > $R/bm_int8.txt 2>&1 || echo "GATE-FAIL bm_int8"
(cd GPNAE && python3 check_gpnae_model_int8.py) > $R/model_check.txt 2>&1 || echo "GATE-FAIL model_check"
FMT="-Wall -GEXP_W=0 -GMAN_W=7" PATTERN= $J/cmds/int8_lint.sh gpnae_poly > $R/lint_0_7.txt 2>&1
command grep -h "BITEXACT:\|ACCURACY:\|Verdict" $R/*.txt
```

```bash
chmod +x $J/cmds/res_g2.sh && L rs1_g2 32 8 $J/cmds/res_g2.sh
```

Expected: no `GATE-FAIL`; five `BITEXACT: PASS` with `ACCURACY: PASS (...; gated: sigmoid 0 of 1280 outside; ...)`; fp32 and bf16 `Verdict : All patterns passed`; `model_check.txt` ends `RESULT: PASSED`.

- [ ] **Step 2: The three spec mutants** (a threshold off by 1, a ladder starting at 3.5, the positive branch computed as k - 128) must fail

```bash
F=GPNAE/src/gpnae_poly_int8.sv
$J/cmds/res_mkmut.sh m_thr $F "16'd5761," "16'd5762,"
$J/cmds/res_mkmut.sh m_x0 $F "SIG_LAD_X0 = 16'sd5120;" "SIG_LAD_X0 = 16'sd7168;"
$J/cmds/res_mkmut.sh m_pos $F "return (k == 5'd0) ? 8'h7F : 8'd128 - {3'b000, k};" "return {3'b000, k} - 8'd128;"
for m in m_thr m_x0 m_pos; do L rs1_mut_$m 32 2 $J/cmds/res_mut.sh $J/res_mut/$m.patch $J/cmds/res_gpnae.sh --lane poly --format int8; done
L rs1_mut_m_thr_check 32 1 $J/cmds/res_mut.sh $J/res_mut/m_thr.patch $J/cmds/res_py.sh GPNAE check_gpnae_model_int8.py
```

Expected: each `res_gpnae` run `BITEXACT: FAIL` with sigmoid below 52736 (for `m_thr`, the ladder edge segment's sigmoid count below its total), `exit=1`; `rs1_mut_m_thr_check` prints `[FAIL] gpnae_poly_int8's ladder literals ...` and `RESULT: FAILED`.

- [ ] **Step 3: Identity checks** (spec G2: tanh, SELU, the float lanes and gpnae.sv identical)

```bash
cd $T/GPNAE
git diff --stat afd786b HEAD -- src | command grep -v gpnae_poly_int8.sv | command grep -c '|'   # 0: no other RTL file changed
git diff --quiet afd786b HEAD -- poly_coeffs.mem poly_coeffs_bf16.mem poly_coeffs_int8.mem taylor_coeffs.mem src/TYTAN/Memory && echo TABLES-SAME
git diff -U0 afd786b HEAD -- gpnae_model.py | command grep '^@@'   # every hunk at or after the '# int8 lane' section (line 158)
B=$J/runs/rs1_base_gp_int8/results/res/gpnae/stimulus; N=$J/runs/rs1_g2/results/res/g2/int8_seed1/stimulus
for a in selu tanh relu linear; do command cmp -s $B/${a}_exp.mem <(head -n $(wc -l < $B/${a}_exp.mem) $N/${a}_exp.mem) && echo "$a SAME" || echo "$a MOVED"; done
for n in 1280 25856; do echo "sigmoid outputs changed in the first $n: $(paste -d' ' <(head -n $n $B/sigmoid_exp.mem) <(head -n $n $N/sigmoid_exp.mem) | awk '$1 != $2' | wc -l)"; done
for f in fp32 bf16; do for x in $J/runs/rs1_base_gp_$f/results/res/gpnae/stimulus/*.mem; do
  command cmp -s $x $J/runs/rs1_g2/results/res/g2/${f}*/stimulus/$(basename $x) || echo "$f $(basename $x) MOVED"; done; done
```

Expected: `0`, `TABLES-SAME`, hunks only in the int8 section, `selu/tanh/relu/linear SAME`, the sigmoid counts printed (the spec expects 172 of the 1,280 gated outputs to change; report the measured number either way), no `MOVED` for fp32 or bf16. Because these inputs and goldens are byte-identical and the RTL is bit-exact against them, the RTL's tanh, SELU, ReLU, linear and float outputs are identical to 2a's.

- [ ] **Step 4: Depth of the lane** — `L rs1_yd_gp 32 4 bash $J/cmds/yosys_depth.sh gpnae_poly_int8 - generic`; compare its `ltp.txt` length with `rs1_base_yd_gp`. Expected: not longer. If longer and the path runs through `sig_ladder` (read `ltp.txt`), move k into a 5-bit buffer written beside `x_buf` in the rescale cycle (the spec's fallback, no cycle cost: `k_buf[rs_p[MUL_LAT]] <= ladder_k(x_new)`, and G_POST reads it), rerun Steps 1 to 3, and commit it as its own RTL commit.

- [ ] **Step 5: Gate report and Phase 1 close** — write `$J/runs/rs1_g2/g2_ladder_gate.log` (run names; the Step 1 lines; the three mutant verdicts; the Step 3 results including the sigmoid change counts; the depth numbers; the G3 line from Task 1.8; `GATE G2 (sigmoid ladder): PASS`). Check the four repos:

```bash
for d in $A GPNAE/ArithmeticLibrary GPNAE SystolicMesh .; do echo "$d: $(git -C $T/$d rev-parse --abbrev-ref HEAD) $(git -C $T/$d rev-parse --short HEAD) $(git -C $T/$d status --short | wc -l) changes"; done
for d in $A GPNAE SystolicMesh; do git -C $T/$d log --format=%B origin/main..HEAD | command grep -ci co-authored-by; done
```

Expected: `int8_residual` everywhere, 0 uncommitted changes except SIENNA's ` M GPNAE` and ` M SystolicMesh` (pointers move in Phase 2), and `0` three times. Nothing is pushed. Phase 1 is done; tell Soham G0b, G1 and G2 passed (with the report paths) and that Phase 2 waits for `regression_sweep`.

---
### Task 1.14: bf16 sigmoid accuracy (added by the controller; Soham's decision 2026-10-10)

**Why:** GPNAE's dense negative-input pattern (GPNAE main 61867b6, `act_neg_dense`) shows the bf16 poly lane's sigmoid at 18.1% worst relative error at x = -3.375 (bit-exact model), against the 7.2% recorded before; the region is the fitted range's edge, as for int8 (-3.39 to -2.61). Soham: fix it before synthesis; the hardware must be accurate for every input, not tuned to a model.

**Files:** GPNAE `src/gpnae_poly.sv` (bf16 path only; fp32 words and cycles unchanged), `gpnae_model.py`, `fit_poly_coeffs.py` and the bf16 coefficient table if the fix refits, `regression.py` (bf16 accuracy gate), `check_gpnae_model_*.py`; `gpnae.sv` and its TYTAN units byte-identical.

- [ ] **Step 1: Design study (no RTL).** On the bit-exact bf16 lane model, over EVERY bf16 input in [-8, 8] (and the tail beyond, both signs): the worst relative and absolute error of sigmoid, tanh and SELU against float64, where it occurs, and why (fit degree, fit range, the tail unit's start, bf16 rounding of P or of 1 - P). Candidate fixes for sigmoid, each scored on the model over every bf16 input: (a) the float tail unit (gpnae_tail, exact) starting earlier for sigmoid in bf16; (b) a refit of the bf16 sigmoid table (range, degree within barrel_mac's limits); (c) computing 1 - P differently for x < 0. Report cycles, area (estimate) and the worst error of each. Write the study to `/proj/work/spramanik/sienna_report/bf16_sigmoid_study.log`.
- [ ] **Step 2: Ruling.** The controller picks the fix (the smallest change that meets the target on every bf16 input, no fp32 change, gpnae.sv untouched) and records it; tanh and SELU results go to Soham as findings (fixing them is his call).
- [ ] **Step 3: Failing test first.** A gated bf16 sigmoid accuracy check over every bf16 input: worst relative error <= max(1%, 1 bf16 ulp of the exact value) (target; the study may tighten it). It must fail on today's lane (18.1%).
- [ ] **Step 4: Fix** the model, then the RTL (and the table if refit), keeping RTL bit-exact against the model (GPNAE regression, bf16, hw model, every pattern including act_neg_dense).
- [ ] **Step 5: Verify on the farm:** bf16 bit-exact against the model on every pattern; the new accuracy gate passes; fp32 and int8 lanes unchanged (words and cycles); a mutant (the old fit or tail start) fails the gate; bf16 sigmoid cycles reported before and after.
- [ ] **Step 6: Commit** (GPNAE, branch int8_residual): RTL, then tests, then docs (GPNAE README, sienna-uniform-format's bf16 accuracy lines).

# Phase 2: SIENNA (after `regression_sweep` merges)

Every Phase 2 path is relative to `$T` unless it starts with `$A` (ArithmeticLibrary) or `$J`. `sienna_top.sv` and `sienna_layer.sv` are not touched by `regression_sweep`, so their line numbers below are this clone's (main `2529f0f`); `regression.py`, `Makefile`, `TB_sienna_top.sv` and `model_runner.py` are, so their steps name functions, not lines.

### Task 2.0: The merge point: rebase, reference tree, reference runs

**Files:**
- Modify: the four `int8_residual` branches (rebased)
- Create: `/proj/scratch/spramanik/int8_res_base` (a clean clone of the merged main, for reference runs only)
- Create (local): `$J/cmds/res_g4_pts.sh`

**Interfaces:**
- Produces: `TB=/proj/scratch/spramanik/int8_res_base` (merged main, all four repos); reference runs `rs2_base_<fmt>_small` (N = 8, 16), `rs2_base_<fmt>_n32`, `rs2_base_ck0_int8`, `rs2_base_ri_int8`, `rs2_base_tflite`, `rs2_base_gptfl`; `$J/cmds/res_g4_pts.sh FMT N` (every point of FMT at size N, then the table).

- [ ] **Step 1: Confirm the merge** — Soham says `regression_sweep` is merged and verified, and:

```bash
for d in . GPNAE SystolicMesh $A; do git -C $T/$d fetch origin; done
git -C $T/$A show origin/main:Common/regress.py > /dev/null && echo ARIL-ENGINE
git -C $T show origin/main:regression.py | command grep -c '"points"'          # > 0: the points action is on main
git -C $T/GPNAE show origin/main:regression.py | command grep -c "class LaneSpec"   # 1
```

Expected: `ARIL-ENGINE`, two non-zero counts. Otherwise stop: Phase 2 does not start.

- [ ] **Step 2: Rebase, innermost first**

```bash
OLD_AH=$(git -C $T/$A rev-parse HEAD)
git -C $T/$A rebase origin/main && AH=$(git -C $T/$A rev-parse HEAD)
git -C $T/GPNAE/ArithmeticLibrary fetch $T/$A int8_residual && git -C $T/GPNAE/ArithmeticLibrary checkout -B int8_residual FETCH_HEAD
BUMP=$(git -C $T/GPNAE log --format=%H --grep '^Bump ArithmeticLibrary 1d04a26' origin/main..int8_residual)
git -C $T/GPNAE rebase --onto origin/main $BUMP int8_residual     # drops Phase 1's bump; the ladder commits replay
git -C $T/SystolicMesh checkout -B int8_residual origin/main      # Phase 1's only commit there was the bump
git -C $T rebase origin/main                                      # the spec and this plan's commits on top of the merged main (rebase ignores the moved submodule pointers)
```

If GPNAE's `regression.py` conflicts, keep both sides: `regression_sweep`'s appended `make regression` section and Task 1.11's `run_int8` changes are separate hunks; resolve, `git add regression.py`, `git rebase --continue`. Then redo the two bumps (the GPNAE one last, after the ladder commits):

```bash
git -C $T/GPNAE add ArithmeticLibrary && git -C $T/GPNAE commit -m "Bump ArithmeticLibrary to $(git -C $T/$A rev-parse --short HEAD) (int8_residual on main): tfliteAdd and its bench, ipu's ADD model, sienna_fmt_pkg::qadd_lat(). Nothing GPNAE builds changes."
git -C $T/SystolicMesh add ArithmeticLibrary && git -C $T/SystolicMesh commit -m "Bump ArithmeticLibrary to $(git -C $T/$A rev-parse --short HEAD) (int8_residual on main): tfliteAdd and its bench, ipu's ADD model, sienna_fmt_pkg::qadd_lat(). No unit the mesh builds changes."
```

- [ ] **Step 3: Phase 1's gates on the rebased tree** (the merged ArithmeticLibrary and GPNAE run their own `make regression`)

```bash
L rs2_aril_reg 32 6 make -C SystolicMesh/ArithmeticLibrary regression PYTHON=$J/venv/bin/python
L rs2_gp_reg 32 4 make -C GPNAE regression FMT=int8
```

Expected: the ArithmeticLibrary table all PASS (its points do not include `Add` yet: Task 2.1); GPNAE `int8/lane` PASS (its verdict: `BITEXACT: PASS` and exit 0, which now includes sigmoid's hard gate).

- [ ] **Step 4: The reference tree and runs**

```bash
cd /proj/scratch/spramanik && git clone --recurse-submodules git@github.com:SoHam-56/SIENNA.git int8_res_base
TB=/proj/scratch/spramanik/int8_res_base; git -C $TB submodule status --recursive
```

`$J/cmds/res_g4_pts.sh`:

```bash
#!/bin/bash
# Every make-regression point of one format at one size, each its own make target, then the table; args: FMT N; run from a snapshot root.
export VERILATOR_ROOT="$HOME/.local/share/verilator"
f=$1; n=$2; P=${PYTHON:-/proj/work/spramanik/sienna_jobs/venv/bin/python}
pts=$($P regression.py --action points --formats $f | tr ' ' '\n' | command grep "^$f/N${n}_")
[ -n "$pts" ] || { echo "res_g4_pts: no $f N$n points"; exit 1; }
for p in $pts; do make regression/$p PYTHON=$P || echo "POINT-FAIL $p"; done
make regression-summary FMT=$f PYTHON=$P
```

Before the first launch, check the merged CLI: `python3 $T/regression.py --action points --help` must accept `--formats` and print point names `<fmt>/N<n>_T<t>/<kind>`; adjust the `pts=` line to what it prints if not.

```bash
chmod +x $J/cmds/res_g4_pts.sh
for f in fp32 bf16 int8; do
  TREE=$TB $J/cr_msgs/cr_launch.sh rs2_base_${f}_n8 64 24 $J/cmds/res_g4_pts.sh $f 8
  TREE=$TB $J/cr_msgs/cr_launch.sh rs2_base_${f}_n16 64 24 $J/cmds/res_g4_pts.sh $f 16
  TREE=$TB $J/cr_msgs/cr_launch.sh rs2_base_${f}_n32 64 48 $J/cmds/int8_O0.sh $J/cmds/res_g4_pts.sh $f 32
done
TREE=$TB $J/cr_msgs/cr_launch.sh rs2_base_ck0_int8 64 24 $J/cmds/sienna_ck0.sh 16 4 int8
TREE=$TB $J/cr_msgs/cr_launch.sh rs2_base_gptfl 32 2 $J/cmds/res_py.sh . regression.py --action gpnae-tflite --report gpnae_tflite.log
```

Expected: every table PASS (the merged main's own state); keep each run's per-test logs (words and cycles) as the reference for Task 2.11. `rs2_base_gptfl`'s report is the 2a TFLite logistic agreement (spec Risk 5).

- [ ] **Step 5: No commit beyond Step 2's bumps.**

### Task 2.1: ArithmeticLibrary's regression points for `Add`

**Files:**
- Modify: `$A/regression.py` (`POINTS`, and any self-test that counts them)

**Interfaces:**
- Produces: points `Add/both`, `Add/vivado_tb`, `Add/lint`, `Add/lint_bad` in `make regression` (the oracle replay needs G0b's data outside the repo, so it stays a farm-only gate step: Task 1.7's `res_g1.sh`).

- [ ] **Step 1: Write the failing self-test** — in `$A/regression.py`'s self-tests, a test that every unit directory with a Makefile has at least one point:

```python
@selftest
def test_every_unit_has_points():
    """Every unit directory with a Makefile appears in POINTS, so a new unit cannot be left out of make regression."""
    units = sorted(os.path.relpath(os.path.dirname(p), ROOT) for p in glob.glob(os.path.join(ROOT, "*", "*", "Makefile")) + glob.glob(os.path.join(ROOT, "*", "Makefile"))
                   if os.path.dirname(p) != ROOT)
    listed = {u for u, *_ in POINTS}
    missing = [u for u in units if u not in listed and u not in NO_POINTS]
    assert not missing, f"units without a make regression point: {missing}"
```

with `NO_POINTS = frozenset({"Multipliers/Radix4Booth", "Divider/FP32"})` (no Makefile there today; the set names any directory the glob finds that has no bench) and `import glob` at the top. Run on the login node: `cd $T/$A && python3 regression.py --action selftest` → `AssertionError: units without a make regression point: ['Add']`.

- [ ] **Step 2: Implement** — add to `POINTS` after the Requant entries:

```python
    ("Add", "both", TB, TB_BAD, True), ("Add", "vivado_tb", TB, TB_BAD, True), ("Add", "lint", (), LINT_BAD, True),
    ("Add", "lint_bad", (r"^REJECTED$",), (r"^NOT-REJECTED$",), False),  # its log shows the %Error it provokes
```

and update any self-test that pins the point count (38 before) to 42.

- [ ] **Step 3: Run** — `python3 regression.py --action selftest` (login node) → `ALL <n> PASSED`; `L rs2_aril_reg2 32 6 make -C SystolicMesh/ArithmeticLibrary regression PYTHON=$J/venv/bin/python` → the table has the four `Add/*` points PASS.

- [ ] **Step 4: Commit (ArithmeticLibrary), then the pointers**

```bash
git -C $T/$A add regression.py && git -C $T/$A commit -m "make regression runs tfliteAdd's bench: Add/both (vectors and exhaustive pairs in DOUBLE and SINGLE), vivado_tb, lint and lint_bad; a self-test fails when a unit directory with a Makefile has no point (it failed before: Add). 42 points. G0b's replay (make -C Add oracle ORACLE=...) needs the oracle's data outside the repo and stays a gate step."
git -C $T/GPNAE/ArithmeticLibrary fetch $T/$A int8_residual && git -C $T/GPNAE/ArithmeticLibrary merge --ff-only FETCH_HEAD
for r in GPNAE SystolicMesh; do git -C $T/$r add ArithmeticLibrary && git -C $T/$r commit -m "Bump ArithmeticLibrary to $(git -C $T/$A rev-parse --short HEAD): its make regression runs tfliteAdd's bench."; done
```

### Task 2.2: SIENNA: the submodule pointers, the file lists, G3

**Files:**
- Modify: `GPNAE`, `SystolicMesh` (pointers), `Makefile` (`SM_LIB_FILES`), `synth/sienna_rtl.f`

**Interfaces:**
- Produces: SIENNA builds with `tfliteAdd.sv` (unused until Task 2.6) and GPNAE's ladder; the existing int8 sigmoid tests' goldens follow the ladder through `gpnae_model` (SIENNA's golden imports it).

- [ ] **Step 1: The failing check** — `make check-files` lists every design file; add the unit first:

In the `Makefile`, after `Requant/src/tfliteRequant.sv` in `SM_LIB_FILES` add `	Add/src/tfliteAdd.sv` (the line before gets its ` \`). In `synth/sienna_rtl.f` after `SystolicMesh/ArithmeticLibrary/Requant/src/tfliteRequant.sv` add `SystolicMesh/ArithmeticLibrary/Add/src/tfliteAdd.sv`. Before the pointer bump, `make check-files` (login node, file existence only) prints `MISSING: .../Add/src/tfliteAdd.sv` because SIENNA's pointer still names the old SystolicMesh.

- [ ] **Step 2: Bump the pointers** — `git -C $T add GPNAE SystolicMesh` and rerun `make check-files` → no `MISSING`.

- [ ] **Step 3: G3 and the first SIENNA run**

```bash
git -C $T diff --cached --submodule=log SystolicMesh | head -20     # only ArithmeticLibrary's bump commits
L rs2_bump_int8_n16 64 24 $J/cmds/res_g4_pts.sh int8 16
L rs2_bump_fp32_n16 64 24 $J/cmds/res_g4_pts.sh fp32 16
```

Expected: both tables PASS; against `rs2_base_*_n16`, fp32 words and cycles identical; int8 cycles identical, words identical except in sigmoid tests (`int8_mixed_act_train`, `int8_accum2_mixed_bypass_nopool` and any with sigmoid), whose changed words are counted and listed in the ledger (they are new goldens from the model, the run passes against them).

- [ ] **Step 4: Commit (SIENNA)**

```bash
git -C $T add GPNAE SystolicMesh Makefile synth/sienna_rtl.f
git -C $T commit -m "Bump GPNAE and SystolicMesh (int8_residual): GPNAE's int8 sigmoid takes an exact ladder for |x| >= 2.5 (0 of 65536 dense and 0 of 1280 gated inputs outside GPNAE's tolerance, from 3370 and 34; sigmoid tests' goldens follow gpnae_model, <k> words changed at N = 16, cycles unchanged), ArithmeticLibrary adds tfliteAdd (TFLite's int8 ADD) and qadd_lat(). The Makefile and synth/sienna_rtl.f list tfliteAdd.sv. The mesh is unchanged (G3: its tree differs only in the library pointer). Runs rs2_bump_int8_n16, rs2_bump_fp32_n16."
```

### Task 2.3: The residual sideband, per-id storage and `a_float_no_add` (no behaviour change)

**Files:**
- Modify: `src/sienna_set_side.svh`, `src/sienna_top.sv` (per-id flags and ADD words, the registered accept terms, `a_float_no_add`), `testbenches/TB_sienna_top.sv` (`side_now()` and a fault define)

**Interfaces:**
- Produces:
  - `set_side_t` fields `res_hold`, `res_add` (1 bit), `add_zp1`, `add_zp2`, `add_s1`, `add_s2`, `add_so`, `add_zpo`, `add_min`, `add_max` ([7:0]), `add_m1`, `add_m2`, `add_mo` ([31:0]).
  - In `sienna_top`: `set_hold[NUM_IDS]`, `set_add[NUM_IDS]` (reset), `g_hold`, `g_add` (the activation stage's copies, reset); module-level `g_add_w` (160 bits `{zp1, zp2, m1, s1, m2, s2, mo, so, zpo, min, max}`, loaded at `g_accept` in `G_REQ_SETS`, `'0` outside int8) and `g_add_zpo = g_add_w[23:16]`; in `G_REQ_SETS` `s_add[NUM_IDS]`, `pe_add` (reset), `pe_add_zpo`; module-level `hold_done` and `fill_add`, tied to 0 here (Task 2.6 drives them).

- [ ] **Step 1: The failing check: `a_float_no_add` must fire** — add to `TB_sienna_top.sv`'s `side_now()` (before `return s;`), inside `ifdef TB_FLOAT_ADD`: `s.res_add = 1'b1;` (a fault build: every set claims to be an add set). Build fp32 with it: `L rs2_fadd_fail 64 4 $J/cmds/mk.sh pipeline FMT=fp32 N=16 TILE=4 TEST=matmul_relu_nopool EXTRA_FLAGS=-DTB_FLOAT_ADD` → fails to elaborate (`res_add` is not a member of `set_side_t`).

- [ ] **Step 2: Implement** — `sienna_set_side.svh`, after `shift`:

```systemverilog
  logic                                       res_hold;       // int8 residual: the last depth pass; its beats go to hold_mem, no bank, no output
  logic                                       res_add;        // int8 residual: the identity pass; its beats meet hold_mem in add_lanes
  logic [7:0]                                 add_zp1;        // the ADD's words (tfliteAdd): input zero points, multipliers and shifts, output zero point and clamp
  logic [7:0]                                 add_zp2;
  logic [31:0]                                add_m1;
  logic [7:0]                                 add_s1;
  logic [31:0]                                add_m2;
  logic [7:0]                                 add_s2;
  logic [31:0]                                add_mo;
  logic [7:0]                                 add_so;
  logic [7:0]                                 add_zpo;
  logic [7:0]                                 add_min;
  logic [7:0]                                 add_max;
```

`sienna_top.sv`: beside `set_accum` (line 164) declare `logic set_hold[NUM_IDS], set_add[NUM_IDS];  // int8 residual: a hold set, an add set`, `logic g_hold, g_add;  // the activation stage's set is a hold set, an add set`, `logic [159:0] g_add_w;  // its ADD words, {zp1, zp2, m1, s1, m2, s2, mo, so, zpo, min, max}`, `logic [7:0] g_add_zpo;  // its ADD's output zero point, the lanes' input zero point (P-3)` with `assign g_add_zpo = g_add_w[23:16];`, and after `logic [FCNT_W-1:0] byp_idx;` (line 152) `logic hold_done;  // int8: a hold set's last beat is in hold_mem: its L6 reservation goes back, pooling may pass it` and `logic fill_add;  // int8: the fill beat this cycle came through add_lanes`, each with `assign ... = 1'b0;` in both `G_REQ` and `G_NO_REQ` for now (Task 2.6 drives them in `G_REQ`); in the reset branch of the main `always_ff` add `g_hold <= 1'b0; g_add <= 1'b0;` and in its per-id loop `set_hold[k] <= 1'b0; set_add[k] <= 1'b0;`; under `if (host_accept)` add `set_hold[host_next_id] <= IS_INT && side.res_hold; set_add[host_next_id] <= IS_INT && side.res_add;`; under `if (g_accept) begin ... end` add `g_hold <= set_hold[g_next_id]; g_add <= set_add[g_next_id];`.

In `G_REQ_SETS` (after the `s_mout` declaration):

```systemverilog
    logic [159:0] s_add[NUM_IDS];  // int8 residual: an add set's ADD words, {zp1, zp2, m1, s1, m2, s2, mo, so, zpo, min, max}
    logic [7:0]   pe_add_zpo;  // the pooled set's ADD output zero point, for the drop value
    logic         pe_add;  // the pooled set is an add set
```

in its no-reset `always_ff`: under `if (host_accept)` `s_add[host_next_id] <= {side.add_zp1, side.add_zp2, side.add_m1, side.add_s1, side.add_m2, side.add_s2, side.add_mo, side.add_so, side.add_zpo, side.add_min, side.add_max};`, under `if (g_accept)` `g_add_w <= s_add[g_next_id];` (the activation stage's copy, like `g_mult`), under `if (p_accept || p_null)` `pe_add_zpo <= s_add[p_next_id][23:16];`; and a reset block beside `pe_act`'s: `always_ff @(posedge clk_i or negedge rstn_i) if (!rstn_i) pe_add <= 1'b0; else if (p_accept || p_null) pe_add <= set_add[p_next_id];`. In `G_NO_REQ_SETS`: `assign g_add_w = '0;`.

In the `ifndef SYNTHESIS` accept terms: add `acc_hold_q, acc_add_q` to the registered tuple (`<= {..., side.res_hold, side.res_add}`) and:

```systemverilog
  if (!IS_INT) begin : G_FLOAT_NO_ADD
    a_float_no_add: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_q |-> !(acc_hold_q || acc_add_q))
      else $error("sienna_top: a_float_no_add: a float set was put as a hold or add set; the residual ADD is int8 only");
  end
```

`sienna_layer.sv` needs nothing yet (`p_side = '0` leaves the new fields 0). `TB_sienna_top`, `TB_sienna_model`, `TB_sienna_multi` and `TB_model_run` build `set_side_t` from `'0` (check: `command grep -n "= '0;" testbenches/TB_sienna_*.sv | command grep -i side`), so the new fields are 0.

- [ ] **Step 3: Show the assertion fires, then that nothing else moved**

```bash
L rs2_fadd 64 4 $J/cmds/mk.sh pipeline FMT=fp32 N=16 TILE=4 TEST=matmul_relu_nopool EXTRA_FLAGS=-DTB_FLOAT_ADD
for f in fp32 bf16 int8; do L rs2_side_${f}_n16 64 24 $J/cmds/res_g4_pts.sh $f 16; done
L rs2_side_int8_n8 64 24 $J/cmds/res_g4_pts.sh int8 8
```

Expected: `rs2_fadd` log holds `Assertion failed` with `a_float_no_add` (and the run FAILS: the parsers fail any log with `Assertion failed`); the three N = 16 tables and int8 N = 8 PASS with words and cycles identical to `rs2_bump_*` / `rs2_base_int8_n8` (the sideband widens L0 by 162 bits, which no stage reads yet).

- [ ] **Step 4: Commit (SIENNA), RTL first**

```bash
git -C $T add src/sienna_set_side.svh src/sienna_top.sv
git -C $T commit -m "Set sideband: res_hold, res_add and the ADD's words (add_zp1, add_zp2, add_m1, add_s1, add_m2, add_s2, add_mo, add_so, add_zpo, add_min, add_max; +162 bits on L0). sienna_top keeps the flags per set id (reset) and the words per id like s_zp, and copies them at g_accept and, for the drop value, at the pooling accept; a_float_no_add fails a float set put as a hold or add set. No stage reads them yet: every test's words and cycles are unchanged (rs2_side_*)."
git -C $T add testbenches/TB_sienna_top.sv
git -C $T commit -m "TB_sienna_top: -DTB_FLOAT_ADD puts every set as an add set, so a float build shows a_float_no_add firing (rs2_fadd)."
```

### Task 2.4: `add_lanes` and its bench

**Files:**
- Create: `src/add_lanes.sv`, `testbenches/TB_add_lanes.sv`
- Modify: `regression.py` (`--action add-vectors`), `Makefile` (`TOP_FILES`, target `add-lanes`), `synth/sienna_rtl.f`

**Interfaces:**
- Consumes: `tfliteAdd`, `qadd_lat()`, `ipu.tflite_add`, `ipu.add_params`.
- Produces: module `add_lanes #(NUM_LANES, TAG_W, ROUNDING)` with ports `clk_i, rstn_i, valid_i, q1_i[NUM_LANES][7:0], q2_i[NUM_LANES][7:0], tag_i[TAG_W], zp1_i, zp2_i, s1_i, s2_i, so_i, zpo_i, min_i, max_i [7:0], m1_i, m2_i, mo_i [31:0], valid_o, result_o[NUM_LANES][7:0], tag_o[TAG_W]`; `regression.py --action add-vectors OUT` (vectors for `TB_add_lanes`: 6 sets of `ADD_PER` beats, each set its own ADD words, expected from `ipu.tflite_add` in model_runner's rounding); `make add-lanes`.

- [ ] **Step 1: Write the vector writer and the TB first**

In `regression.py`, beside `rq_vectors_main` (its section, `# ── rq-vectors`), add a section and register `"add-vectors": add_vectors_main` in `MAINS` and `ACTIONS`:

```python
# ── add-vectors: TB_add_lanes' vectors ────────────────────────────────────────

ADD_LANES, ADD_SETS, ADD_PER = 32, 6, 8  # TB_add_lanes' geometry: lanes, sets, beats per set


def add_vectors_main(argv=None) -> None:
    """Vectors for TB_add_lanes: per set its ADD words, then per beat a tag and each lane's q1, q2 and expected code; arg: output .mem path."""
    ap = argparse.ArgumentParser(description=add_vectors_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="add-vectors")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(29)
    lines = []
    for s in range(ADD_SETS):
        sc = [float(np.float32(2.0 ** e)) for e in rng.uniform(-8, 1, 3)]
        z = [int(v) for v in rng.integers(-128, 128, 3)]
        add = mr.add_words(sc[0], z[0], sc[1], z[1], sc[2], z[2], ("none", "relu", "relu6")[s % 3])
        lines.append(" ".join(f"{add[k] & 0xFFFFFFFF:08x}" for k in mr.ADD_FIELDS))
        for b in range(ADD_PER):
            q1, q2 = rng.integers(-128, 128, ADD_LANES), rng.integers(-128, 128, ADD_LANES)
            want = mr.tflite_add_codes(q1, q2, add)
            lines.append(f"{(s * ADD_PER + b) & 0xFFFF:04x} " + " ".join(f"{int(x) & 0xFF:02x}{int(y) & 0xFF:02x}{int(w) & 0xFF:02x}"
                                                                     for x, y, w in zip(q1, q2, want)))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {ADD_SETS} sets x {ADD_PER} beats x {ADD_LANES} lanes, rounding {mr.ROUNDING}")
```

and, in `model_runner.py`'s Numerics section after `drop_zp`, the three helpers it uses:

```python
ADD_FIELDS = ("zp1", "zp2", "m1", "s1", "m2", "s2", "mo", "so", "zpo", "amin", "amax")  # a layer's ADD words, tfliteAdd's order


def add_words(s1, z1, s2, z2, s_out, z_out, act: str) -> dict:
    """A layer's ADD words from its inputs' and output's quantization (TFLite's Prepare, ipu.add_params) and the fused activation's clamp."""
    amin, amax = activation_range(act if act in ("relu", "relu6") else "none", s_out, int(z_out))
    return dict(zp1=int(z1), zp2=int(z2), **ipu.add_params(s1, s2, s_out), zpo=int(z_out), amin=int(amin), amax=int(amax))


def tflite_add_codes(q1, q2, add: dict) -> np.ndarray:
    """ipu.tflite_add of two int8 code arrays with a layer's ADD words, in the pinned rounding."""
    return np.asarray(ipu.tflite_add(q1, q2, *[add[k] for k in ADD_FIELDS], ROUNDING), np.int64)
```

`testbenches/TB_add_lanes.sv`:

```systemverilog
`timescale 1ns / 100ps

// add_lanes against ipu.tflite_add (add_lanes.mem): per set its ADD words, per beat each lane's q1 and q2; the tag rides qadd_lat() cycles.
module TB_add_lanes;
  localparam int NUM_LANES = 32, SETS = 6, PER = 8, TAG_W = 16, LAT = sienna_fmt_pkg::qadd_lat();
  logic clk_i = 0, rstn_i = 0, valid_i = 0, valid_o;
  logic [NUM_LANES-1:0][7:0] q1_i = '0, q2_i = '0, result_o;
  logic [TAG_W-1:0] tag_i = '0, tag_o;
  logic [7:0] zp1_i = '0, zp2_i = '0, s1_i = '0, s2_i = '0, so_i = '0, zpo_i = '0, min_i = '0, max_i = '0;
  logic [31:0] m1_i = '0, m2_i = '0, mo_i = '0;
  always #5 clk_i = ~clk_i;

  add_lanes #(.NUM_LANES(NUM_LANES), .TAG_W(TAG_W), .ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) dut (.*);

  logic [NUM_LANES-1:0][7:0] want_q[$];
  logic [TAG_W-1:0] tag_q[$];
  longint t_in[$];
  longint cyc = 0;
  int errs = 0, lat_errs = 0, tag_errs = 0, n_out = 0;

  always @(posedge clk_i) begin
    cyc <= cyc + 1;
    if (valid_i) t_in.push_back(cyc);
    if (valid_o) begin
      automatic logic [NUM_LANES-1:0][7:0] w = want_q.pop_front();
      automatic logic [TAG_W-1:0] tg = tag_q.pop_front();
      automatic longint t = t_in.pop_front();
      n_out++;
      if (cyc - t != LAT) lat_errs++;
      if (tag_o !== tg) tag_errs++;
      for (int k = 0; k < NUM_LANES; k++)
        if (result_o[k] !== w[k]) begin
          errs++;
          if (errs <= 20) $display("[FAIL] beat %0d lane %0d: got %0d, want %0d", n_out - 1, k, $signed(result_o[k]), $signed(w[k]));
        end
    end
  end

  initial begin
    int fd;
    logic [31:0] c[11];
    logic [15:0] tg;
    logic [23:0] e;
    fd = $fopen("add_lanes.mem", "r");
    if (fd == 0) $fatal(1, "TB_add_lanes: no add_lanes.mem (regression.py --action add-vectors writes it)");
    repeat (3) @(posedge clk_i);
    #1 rstn_i = 1;
    for (int s = 0; s < SETS; s++) begin
      void'($fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10]));
      for (int b = 0; b < PER; b++) begin
        automatic logic [NUM_LANES-1:0][7:0] w;
        void'($fscanf(fd, "%h", tg));
        @(posedge clk_i);
        #1;
        {zp1_i, zp2_i, m1_i, s1_i, m2_i, s2_i, mo_i, so_i, zpo_i, min_i, max_i} = {c[0][7:0], c[1][7:0], c[2], c[3][7:0], c[4], c[5][7:0], c[6], c[7][7:0],
                                                                                   c[8][7:0], c[9][7:0], c[10][7:0]};
        for (int k = 0; k < NUM_LANES; k++) begin
          void'($fscanf(fd, "%h", e));
          {q1_i[k], q2_i[k], w[k]} = e;
        end
        tag_i = tg;
        want_q.push_back(w);
        tag_q.push_back(tg);
        valid_i = 1;
        if (b == PER / 2) begin  // a bubble inside a set: valid low for a cycle
          @(posedge clk_i);
          #1 valid_i = 0;
          {q1_i, q2_i, tag_i} = {$urandom, $urandom};
        end
      end
      @(posedge clk_i);
      #1 valid_i = 0;
    end
    repeat (LAT + 4) @(posedge clk_i);
    if (n_out != SETS * PER || want_q.size() != 0) errs++;
    $display("add_lanes: %0d beats, %0d errors, %0d latency errors, %0d tag errors", n_out, errs, lat_errs, tag_errs);
    $display("RESULT: %s", (errs == 0 && lat_errs == 0 && tag_errs == 0) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
```

The beat stream holds `valid_i` high across the beats of a set and changes the ADD words between sets (`tfliteAdd` takes them per beat, so a word change is safe once the set's last beat is in).

`Makefile`: add `	add_lanes.sv \` to `TOP_FILES` after `requant_lanes.sv`, and a target beside the other unit benches:

```make
# add_lanes against ipu.tflite_add; vectors from regression.py --action add-vectors (testbenches/*.mem is ignored)
add-lanes:
	$(PYTHON) regression.py --action add-vectors $(TB_DIR)/add_lanes.mem
	verilator --binary --timing --assert --sv --top-module TB_add_lanes --Mdir $(VERILATOR_DIR)/add_lanes \
	  $(SM_LIB_DIR)/Common/src/sienna_fmt_pkg.sv $(SM_LIB_DIR)/Add/src/tfliteAdd.sv $(SRC_DIR)/add_lanes.sv $(TB_DIR)/TB_add_lanes.sv -o sim
	cd $(TB_DIR) && $(VERILATOR_DIR)/add_lanes/sim
```

and `add-lanes` to the `.PHONY` list and the help text. Add `src/add_lanes.sv` to `synth/sienna_rtl.f` after `src/requant_lanes.sv`.

- [ ] **Step 2: Run it to see it fail** — `L rs2_addl_fail 32 1 $J/cmds/mk.sh add-lanes PYTHON=$J/venv/bin/python` → `Cannot find file containing module: 'add_lanes'` (or the missing `src/add_lanes.sv`); no `RESULT: PASSED`.

- [ ] **Step 3: Implement** `src/add_lanes.sv`:

```systemverilog
`timescale 1ns / 100ps

// int8 residual ADD: one tfliteAdd per lane on an add set's requantized beat and its hold_mem beat; the beat's tag rides beside it for qadd_lat() cycles.
module add_lanes #(
    parameter int    NUM_LANES = 32,
    parameter int    TAG_W     = 1,  // the beat's destination as sienna_top packs it: {bypass, bank, packed, element index}
    parameter string ROUNDING  = sienna_fmt_pkg::REQ_ROUNDING
) (
    input  logic                      clk_i,
    input  logic                      rstn_i,
    input  logic                      valid_i,
    input  logic [NUM_LANES-1:0][7:0] q1_i,  // hold_mem's beat: the conv's requantized codes
    input  logic [NUM_LANES-1:0][7:0] q2_i,  // the add set's beat: the residual's codes through the identity requantize
    input  logic [TAG_W-1:0]          tag_i,
    input  logic [7:0]                zp1_i, zp2_i, s1_i, s2_i, so_i, zpo_i, min_i, max_i,  // the set's ADD words, every lane
    input  logic [31:0]               m1_i, m2_i, mo_i,
    output logic                      valid_o,
    output logic [NUM_LANES-1:0][7:0] result_o,
    output logic [TAG_W-1:0]          tag_o
);
  localparam int QAL = sienna_fmt_pkg::qadd_lat();
  logic [NUM_LANES-1:0] done;
  logic [TAG_W-1:0] tag_q[QAL];

  for (genvar k = 0; k < NUM_LANES; k++) begin : LANE
    tfliteAdd #(.ROUNDING(ROUNDING)) u (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(valid_i), .q1_i(q1_i[k]), .q2_i(q2_i[k]), .zp1_i(zp1_i), .zp2_i(zp2_i), .m1_i(m1_i),
        .m2_i(m2_i), .mo_i(mo_i), .s1_i(s1_i), .s2_i(s2_i), .so_i(so_i), .zpo_i(zpo_i), .min_i(min_i), .max_i(max_i),
        .result_o(result_o[k]), .done_o(done[k]));
  end

  always_ff @(posedge clk_i) begin  // no reset: read only beside valid_o (D-8)
    tag_q[0] <= tag_i;
    for (int i = 1; i < QAL; i++) tag_q[i] <= tag_q[i-1];
  end
  assign tag_o   = tag_q[QAL-1];
  assign valid_o = done[0];  // every lane takes the same beats

`ifndef SYNTHESIS
  a_lanes_in_step: assert property (@(posedge clk_i) disable iff (!rstn_i) done == {NUM_LANES{done[0]}})
    else $error("add_lanes: the lanes' tfliteAdd units are out of step");
`endif

endmodule
```

- [ ] **Step 4: Run it to see it pass, and prove the tag check** — `L rs2_addl 32 1 $J/cmds/mk.sh add-lanes PYTHON=$J/venv/bin/python` → `add_lanes: 48 beats, 0 errors, 0 latency errors, 0 tag errors`, `RESULT: PASSED`. Mutant: `$J/cmds/res_mkmut.sh m_tag src/add_lanes.sv "assign tag_o   = tag_q[QAL-1];" "assign tag_o   = tag_q[QAL-2];"` and `L rs2_mut_tag 32 1 $J/cmds/res_mut.sh $J/res_mut/m_tag.patch $J/cmds/mk.sh add-lanes PYTHON=$J/venv/bin/python` → `tag errors` > 0, `RESULT: FAILED`.

- [ ] **Step 5: Commit (SIENNA), RTL first**

```bash
git -C $T add src/add_lanes.sv synth/sienna_rtl.f
git -C $T commit -m "add_lanes: one tfliteAdd per lane for the int8 residual ADD, the set's ADD words to every lane, each beat's tag (its destination) carried beside it for qadd_lat() cycles; a_lanes_in_step. Not instantiated yet."
git -C $T add testbenches/TB_add_lanes.sv regression.py model_runner.py Makefile
git -C $T commit -m "TB_add_lanes (make add-lanes): add_lanes against ipu.tflite_add on regression.py --action add-vectors (6 sets of 8 beats x 32 lanes, each set its own ADD words from add_params, a bubble inside each set): results, latency qadd_lat() and tags, which a tag one stage early fails (rs2_mut_tag). model_runner's numerics gain ADD_FIELDS, add_words and tflite_add_codes. Run rs2_addl."
```

### Task 2.5: Residual pipeline tests: golden, stimulus, `TB_sienna_top` (the failing tests)

**Files:**
- Modify: `model_runner.py` (Numerics: `add_lane_params`, `int8_residual_exact`), `regression.py` (`PIPELINE_TESTS`, the int8 stimulus generator, `_golden_int8_residual`, the per-test log check, self-tests), `testbenches/TB_sienna_top.sv`

**Interfaces:**
- Consumes: Task 2.4's `mr.ADD_FIELDS`, `mr.add_words`, `mr.tflite_add_codes`.
- Produces:
  - `mr.add_lane_params(add, s_out, y) -> dict` (`zp` = zp_o, `amin`/`amax` = the ADD's clamp, `mx`, `shx`, `mout`, `shout`, `zout`, `s_out`): the lane's words after an ADD.
  - `mr.int8_residual_exact(A_q, B_q, hw_bias, rq, R_q, add, ra, act) -> np.ndarray`.
  - Per set `add_<k>.mem`: 12 words of 8 hex digits: `flags` (bit 0 partial, bit 1 bias, bit 2 hold, bit 3 add), then `zp1 zp2 m1 s1 m2 s2 mo so zpo amin amax`; written for every set of a residual test.
  - `test_knobs.mem` gains `residual 0|1`.
  - Config keys: `"residual": True` (every group: `accum_passes` depth passes, the last a hold set, then an add set), `"residual_every": 2` (every second group residual, the others plain), `"add_layers": 2` (two ADD settings alternate across groups).
  - `TB_sienna_top` prints `[RESIDUAL] hold/add pairs <p>, hold release with an L3 grant <a>, hold done with a bank done <b>, an add set's last beat with the next accept <c>, grants held for an add drain <d>` for a residual test; the per-test check fails a residual test without it or with `p` below the stream's pair count.

- [ ] **Step 1: The numerics** — in `model_runner.py`'s Numerics section, after Task 2.4's `tflite_add_codes`:

```python
def add_lane_params(add: dict, s_out: float, y) -> dict:
    """The lane's words after an ADD: input zero point the ADD's zp_o, the rescale of s_out, SELU's output calibrated on the ADD's outputs."""
    mx, shx = gpnae_model.rescale_params(s_out)
    x = (np.asarray(y, np.int64) - add["zpo"]) * s_out
    _, s_selu, z_selu = quant_act(apply_activation(np.clip(x, -16.0, gpnae_model.SELU_POS_SAT / 2048.0).astype(np.float32), "selu"))
    mout, shout = gpnae_model.quantize_multiplier(2.0 ** -22 / s_selu)
    return dict(zp=add["zpo"], amin=add["amin"], amax=add["amax"], mx=mx, shx=shx, mout=int(mout), shout=int(shout), zout=z_selu,
                s_out=s_out)


def int8_residual_exact(A_q, B_q, hw_bias, rq: dict, R_q, add: dict, ra: dict, act: str) -> np.ndarray:
    """sienna_layer's int8 residual output for one product: the conv's requantize (q1), TFLite's ADD with the residual codes, the lane."""
    acc = wrap32(imatmul(A_q, B_q) + np.asarray(hw_bias, np.int64)[None, :])
    return activate_int8(tflite_add_codes(requantize(acc, rq), R_q, add), act, ra)
```

- [ ] **Step 2: The tests' entries and the golden, test-first** — `PIPELINE_TESTS`, after the int8 entries:

```python
    # int8 residual ADD (sienna-int8-residual): a residual group's last depth pass is a hold set, then an add set (the identity pass) meets it in add_lanes.
    {"name": "int8_residual_relu", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "residual": True,
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "residual": True, "accum_passes": 2,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_sigmoid_selu_nopool", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "residual": True,
     "mixed_acts": ["sigmoid", "selu"], "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "num_sets": 16,
     "formats": ("int8",)},
    {"name": "int8_residual_train", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True, "training": True,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_then_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual_every": 2,
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_two_layers_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "residual": True,
     "add_layers": 2, "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_accum2_mixed_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "accum_passes": 2,
     "residual_every": 2, "mixed_acts": ["relu", "tanh", "linear", "selu", "sigmoid"], "pool_h": 1, "pool_w": 1, "padding": 0,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
```

The golden, beside `_golden_int8`:

```python
def _golden_int8_residual(parts, hw_bias, rq: dict, R, add: dict, ra: dict, cfg: dict, act: str, drop_seed: int) -> tuple:
    """Bit-exact int8 residual set: the depth passes' sums, the conv's requantize (q1), TFLite's ADD with R, the lane with the ADD's words, pooling, dropout (D-5: zp_o)."""
    N = cfg.get("n", 16)
    C = mesh_model.matmul_int([(_pad_square(a, N), _pad_square(b, N)) for a, b in parts], N, hw_bias)
    Y = mr.tflite_add_codes(requantize(C, rq), R, add)
    A = activate_int8(Y, act, ra)
    P = _maxpool_int(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1))
    if not cfg.get("training", False):
        return C, Y, A, P, P.copy()
    keep = dropout_keep(P.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    return C, Y, A, P, np.where(keep, P.flatten(), drop_zp(act, ra)).reshape(P.shape)


def _add_draw(rng, q1, s1: float, z1: int, act: str, zp_random: bool, setting=None) -> tuple:
    """An add set's inputs for a conv output q1 (scale s1, zero point z1): the residual codes, the ADD's words and the lane's; setting fixes (s2 / s1, z2, s_out factor)."""
    N = q1.shape[0]
    f2, z2, fo = setting or (2.0 ** rng.uniform(-3.0, 3.0), int(rng.randint(-100, 101)) if zp_random else 0, 2.0 ** rng.uniform(-1.0, 1.0))
    s2 = s1 * f2
    R = rng.randint(-128, 128, (N, N)).astype(np.int64)
    y = s1 * (q1 - z1) + s2 * (R - z2)
    _, s_out, z_out = quant_act(np.maximum(y, 0.0) if act == "relu" else y)
    s_out *= fo if zp_random else 1.0  # random ADD scales: the output scale off its calibration, so some outputs clamp
    add = mr.add_words(s1, z1, s2, z2, s_out, z_out, act)
    return R, add, mr.add_lane_params(add, s_out, mr.tflite_add_codes(q1, R, add))


def _identity_requant(N: int, ra: dict) -> dict:
    """An add set's requantize words: identity (mult 2^30, shift +1, zp 0, full clamp), with the lane's words after the ADD."""
    return dict(mult=np.full(N, 1 << 30, np.int64), shift=np.full(N, 1, np.int64), zp=0, amin=-128, amax=127,
                mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])
```

Self-tests (in `regression.py`'s registry):

```python
@selftest
def test_residual_golden():
    """_golden_int8_residual against an element-by-element composition (scalar ipu on each code), and it moves with zp1, the ADD and zp_o."""
    rng = np.random.RandomState(3)
    N = 8
    A = rng.randint(-128, 128, (N, N)).astype(np.int64)
    B = rng.randint(-127, 128, (N, N)).astype(np.int64)
    bias = rng.randint(-500, 500, N).astype(np.int64)
    acc = wrap32(imatmul(A, B) + bias[None, :])
    rq = requant_params(acc, 0.02, np.full(N, 0.01), "linear", np.random.RandomState(4), (-9, -128, 127))
    q1 = requantize(acc, rq)
    R, add, ra = _add_draw(rng, q1, rq["s_out"], rq["zp"], "tanh", True)
    assert add["zpo"] != 0, "the fixture needs zp_o != 0, or the lane's zero point cannot show; pick another seed"
    cfg = {"n": N, "pool_h": 1, "pool_w": 1, "padding": 0}
    Y = _golden_int8_residual([(A, B)], bias, rq, R, add, ra, cfg, "tanh", 1)[1]
    for i in range(N):
        for j in range(N):
            assert int(Y[i, j]) == int(ipu.tflite_add(int(q1[i, j]), int(R[i, j]), *[add[k] for k in mr.ADD_FIELDS], mr.ROUNDING))
    for k, v in (("zp1", 0), ("zpo", add["zpo"] + 7)):
        assert np.any(_golden_int8_residual([(A, B)], bias, rq, R, dict(add, **{k: v}), ra, cfg, "tanh", 1)[1] != Y), k
    assert np.any(_golden_int8_residual([(A, B)], bias, rq, R, add, dict(ra, zp=0), cfg, "tanh", 1)[2] !=
                  _golden_int8_residual([(A, B)], bias, rq, R, add, ra, cfg, "tanh", 1)[2]), "the lane must see zp_o"


@selftest
def test_residual_stimulus():
    """A residual test writes add_<k>.mem for every set: per group its partial and hold flags, then the add set; hold and partial sets expect no output."""
    cfg = dict(next(t for t in PIPELINE_TESTS if t["name"] == "int8_residual_tanh"), n=8, lanes=32)
    with _scratch_tb_dir():  # the merged regression.py's fixture that points TB_DIR at a temporary directory
        _generate_vectors_int8(cfg)
        flags = [int(open(os.path.join(TB_DIR, f"add_{k}.mem")).readline(), 16) for k in range(_num_sets(cfg))]
        assert flags[:3] == [0b0011, 0b0100, 0b1000], flags[:3]  # partial with bias, hold, add
        assert all(os.path.getsize(os.path.join(TB_DIR, f"expected_output_{k}.mem")) == 0 for k in (0, 1))
        assert "residual 1" in open(os.path.join(TB_DIR, "test_knobs.mem")).read()
```

(`_scratch_tb_dir` and `_num_sets`: use the merged file's own fixture for a temporary `TB_DIR` and its own count of a test's sets; if it has none, add a three-line context manager that swaps the module global `TB_DIR` and restores it.)

Run (login node): `python3 regression.py --action selftest` → `test_residual_golden` and `test_residual_stimulus` fail (`_golden_int8_residual`'s helpers exist after this step; the stimulus does not write `add_<k>.mem` yet).

- [ ] **Step 3: The stimulus** — in the int8 stimulus generator (`_generate_vectors_int8` on this clone; find its merged name with `command grep -n "def _generate_vectors_int8\|requant_<k>\|requant_{k}" regression.py`):

1. Group layout. Replace `starts = list(range(0, num_sets, passes))` and the `num_sets` arithmetic with a layout of groups `(first set, depth passes, residual)`: plain groups take `passes` sets; residual groups (`residual`, or every `residual_every`-th group starting with group 0) take `passes + 1`. With neither key the layout is today's. The number of groups is `len(mixed)` when `mixed_acts` is set, else the smallest count whose sets exceed `credits + 2`, else `num_sets // passes` when `num_sets` is given (then `num_sets` must equal the layout's total; raise `ValueError` as today when it does not).
2. Per residual group: draw the conv's zero-point triple with act `"linear"` (`_zp_draws(..., ["linear"])` when `zp_random`), compute `rq = requant_params(acc, s_a, s_w, "linear", req_rng, zq)`, `q1 = requantize(acc, rq)`, then `R, add, ra = _add_draw(np.random.RandomState(seed + 9000 + g0), q1, rq["s_out"], rq["zp"], act_g, cfg.get("zp_random", False), settings[g % 2] if cfg.get("add_layers") else None)`, where `settings` holds two `(f2, z2, fo)` draws made once per test from `RandomState(seed + 9500)`.
3. Files per residual group, depth pass `i` (set `k`): as today (`matrix_west_k`, `matrix_north_k`, `bias_k` on `i == 0`, `requant_k` = `rq` on the last depth pass and the decoys before it), `expected_output_k` empty for every depth pass, and `add_k.mem` = `[flags, 0 x 11]` with `flags = (i < passes - 1) | (i == 0) << 1 | (i == passes - 1) << 2`. The add set (set `k_add`): `matrix_west` = `R`, `matrix_north` = the identity (`np.eye(N, dtype=np.int64)`, int8 code 1 on the diagonal), `bias` zeros, `requant` = `_identity_requant(N, ra)`, `add_<k_add>.mem` = `[0b1000] + [add[f] for f in mr.ADD_FIELDS]` (8-digit words, `_write_w32`), `expected_output` = `_golden_int8_residual(parts, hw_bias, rq, R, add, ra, cfg, act_g, set_dropout_seed(drop_seed, k_add))[4]`.
4. Plain groups in a residual test write `add_k.mem` = `[flags, 0 x 11]` with `flags = partial | bias`.
5. `test_knobs.mem` gets `residual 1` for a test with `residual` or `residual_every`, else `residual 0` (every test writes the key). `_check_mem_widths` checks `add_<k>.mem` at 8 digits when the test is residual.
6. The single-set pass (set 0 alone) stays as today: set 0's own words with its activation, its golden from `_golden_int8([(a0, b0)], ...)`; the TB clears set 0's residual flags for it (Step 4).

`perf`: skip residual configs (P-9): in `perf_main`, after the format filter, `args.configs = [c for c in args.configs if not (cfgs[c].get("residual") or cfgs[c].get("residual_every"))]` with `cfgs = {t["name"]: t for t in PIPELINE_TESTS}`, and add the header line `"residual tests (int8 ADD) are not modelled and not run here"`.

- [ ] **Step 4: `TB_sienna_top`** — on the merged TB:

1. `read_knobs()`: accept `"residual": begin k_residual = v; seen |= 1 << 10; end` and require the widened mask (`11'h7FF`).
2. Per-set ADD table, read once after the knobs: `logic [31:0] add_tab[int][12];` and

```systemverilog
  // A residual test's add_<k>.mem: flags (partial, bias, hold, add), then the ADD's words; other tests leave the table empty.
  task automatic read_add_tables();
    logic [31:0] q[$];
    if (k_residual == 0) return;
    for (int k = 0; k < k_num_sets; k++) begin
      read_word_file($sformatf("add_%0d.mem", k), q);
      if (q.size() != 12) $fatal(1, "TB_sienna_top: add_%0d.mem holds %0d words, not 12", k, q.size());
      for (int i = 0; i < 12; i++) add_tab[k][i] = q[i];
    end
  endtask
  function automatic bit is_hold(input int k);
    return k_residual != 0 && add_tab[k][0][2];
  endfunction
  function automatic bit is_add(input int k);
    return k_residual != 0 && add_tab[k][0][3];
  endfunction
```

3. `is_partial(k)`: `if (k_residual != 0) return add_tab[k][0][0];` first. `apply_bias(k)`: `bias_valid_i = (k_residual != 0) ? add_tab[k][0][1] : <today's rule>;`.
4. New per-set signals `logic res_hold_i, res_add_i; logic [7:0] add_zp1_i, add_zp2_i, add_s1_i, add_s2_i, add_so_i, add_zpo_i, add_min_i, add_max_i; logic [31:0] add_m1_i, add_m2_i, add_mo_i;` set by `task automatic apply_add(input int k, input bit single = 0);` (zeros unless `k_residual` and not `single`; else `res_hold_i = add_tab[k][0][2]`, `res_add_i = add_tab[k][0][3]`, the eleven words from `add_tab[k][1..11]`), called wherever `apply_bias(k)` is; the single-set pass calls `apply_add(0, 1)`. `side_now()` maps them: `s.res_hold = res_hold_i; s.res_add = res_add_i; s.add_zp1 = add_zp1_i; ... s.add_max = add_max_i;`.
5. A set with no outputs is `is_partial(k) || is_hold(k)` wherever the TB decides a set's expected words or counts the sets that produce output (the stream's `stream_bounds`, `verify_slice`, the overrun and mid-reset passes).
6. The accumulate pass does not run for a residual test: `if (k_residual != 0) $display("[SKIP] accumulate pass: the residual stream's groups come from add_<k>.mem"); else accum_null_pass();`.
7. The int8 requantize monitor (`rq_id`, `rq_bank`, `rq_byp`): push on `dut.wide_rd_valid && !dut.g_hold` (hold beats never reach `fill_v`), and push `rq_add` (`dut.g_add`); exclude add beats from `rq_wait` and `rq_hold` (an add drain holds the next grant by design) and count them in `add_drain_hold` instead.
8. Cross counters (int8, `@(negedge clk_i)`, while `stream_on`): `res_pairs += dut.hold_done;` `cross_rel_grant += dut.hold_done && dut.l3_grant;` `cross_hold_bank += dut.hold_done && dut.bank_done;` `cross_add_accept += dut.fill_add && (int'(dut.byp_idx) == PER_LANE - 1) && dut.g_accept;` and at the end of each stream pass of a residual test:

```systemverilog
    if (k_residual != 0)
      $display("[RESIDUAL] hold/add pairs %0d, hold release with an L3 grant %0d, hold done with a bank done %0d, an add set's last beat with the next accept %0d, grants held for an add drain %0d",
               res_pairs, cross_rel_grant, cross_hold_bank, cross_add_accept, add_drain_hold);
```

In `regression.py`'s per-test log check (the function that turns a pipeline test's log into its verdict), a residual test fails without a `[RESIDUAL]` line or when its pair count is below the stream's residual groups (`[FAIL] residual: <p> hold/add pairs, expected <g>`).

- [ ] **Step 5: Run the residual tests on today's RTL to see them fail**

```bash
python3 $T/regression.py --action selftest      # login node: ALL <n> PASSED, the two new tests included
L rs2_res_fail 64 4 $J/cmds/mk.sh pipeline FMT=int8 N=16 TILE=4 TEST=int8_residual PYTHON=$J/venv/bin/python
```

Expected: the eight `int8_residual_*` tests FAIL: the hold sets produce outputs nobody expects, the add sets return the residual codes through the identity requantize instead of the ADD, and every `[RESIDUAL]` line reads `hold/add pairs 0` (`hold_done` is tied to 0 since Task 2.3). Every other int8 test still passes.

- [ ] **Step 6: No commit yet** — Task 2.6 commits the RTL, then these.

### Task 2.6: `sienna_top`: the hold path, the add path, the widened drain

**Files:**
- Modify: `src/sienna_top.sv`

**Interfaces:**
- Consumes: Task 2.3's flags and words, Task 2.4's `add_lanes`.
- Produces (module level, read by `TB_sienna_top` and Task 2.7's assertions): `hold_done` (a hold set's last beat is in `hold_mem`), `fill_add` (this cycle's fill beat came from `add_lanes`); in `G_REQ`: `rq_v`, `rq_d` (requantize's output), `tg_hold`, `tg_add`, `hold_mem`, `hold_wr`, `add_in`, `hold_rd_idx`, `at_v`, `add_v`, `add_d`, `add_tag`, `rq_out`.

- [ ] **Step 1: Drop the tie-offs** — remove Task 2.3's `assign hold_done = 1'b0;` and `assign fill_add = 1'b0;` from `G_REQ` (Step 4 drives them; `G_NO_REQ` keeps its tie-offs).

- [ ] **Step 2: The main controller** (the `always_ff` from line 752):

```systemverilog
      l6_rsv  <= l6_rsv + 2'(l3_grant) - 2'(l6.put) - 2'(hold_done);  // a hold set reserved a bank at its grant and fills none
```

```systemverilog
      if (g_done && !g_hold) act_wr <= ~act_wr;  // a hold set fills no bank
```

```systemverilog
      gp_sets <= gp_sets + CRW'(bank_done || g_null_done) + CRW'(hold_done) - CRW'(set_complete);
```

and the pooling accept (lines 748, 749):

```systemverilog
  assign p_null = (p_state == P_IDLE) && (gp_sets != 0) && (set_accum[p_next_id] || set_hold[p_next_id]) && !complete_q;
  assign p_accept = (p_state == P_IDLE) && (gp_sets != 0) && !set_accum[p_next_id] && !set_hold[p_next_id] && act_full[act_rd];
```

with `a_real_has_bank`'s antecedent gaining `&& !set_hold[p_next_id]`.

- [ ] **Step 3: The drop value and the lanes' zero point** — line 905: `CONTROL_WIDTH'(3'b100), CONTROL_WIDTH'(3'b101): p_zp[k] = pe_add ? pe_add_zpo : pe_zp[p_lane_ent[k]];  // ReLU, linear: the ADD's or the requantize output's`; line 472: `.gp_zin_i      (g_add ? g_add_zpo : g_zp[g]),  // int8: the lane's input zero point: the ADD's zp_o after an ADD, else its requantize entry's (P-3)`.

- [ ] **Step 4: `G_REQ`** — rename `requant_lanes`' outputs `.valid_o(rq_v)`, `.result_o(rq_d)` (declare `logic rq_v; logic [NUM_LANES-1:0][7:0] rq_d;`), add `tg_hold`, `tg_add` beside `tg_byp` (`logic [RQL-1:0] tg_byp, tg_bank, tg_hold, tg_add;`, shifted in the no-reset block with `g_hold`, `g_add`), and replace the block from `assign byp_all_in` to `assign bank_sel` with:

```systemverilog
    assign byp_all_in = (rq_in == PER_LANE[FCNT_W-1:0]);  // the last beat is in: the stage may leave while it drains
    // int8 residual: hold beats wait in hold_mem by beat index; add beats meet them there and go through add_lanes.
    localparam int QAL = sienna_fmt_pkg::qadd_lat();
    localparam int ATW = 3 + FCNT_W;  // add_lanes' tag: {bypass, bank, packed, element index}
    logic [NUM_LANES-1:0][7:0] hold_mem[PER_LANE];  // the hold set's requantized beats (N^2 bytes); no reset, read only after its own write
    logic hold_wr, add_in, rq_out, add_v;
    logic [FCNT_W-1:0] hold_rd_idx;
    logic [NUM_LANES-1:0][7:0] add_d;
    logic [ATW-1:0] add_tag;
    logic [QAL-1:0] at_v;  // an add beat at each add_lanes stage
    assign hold_wr     = rq_v && tg_hold[RQL-1];
    assign add_in      = rq_v && tg_add[RQL-1];
    assign rq_out      = rq_v && !tg_hold[RQL-1] && !tg_add[RQL-1];
    assign hold_rd_idx = tg_idx[RQL-1];
    assign hold_done   = hold_wr && (tg_idx[RQL-1] == PER_LANE[FCNT_W-1:0] - 1'b1);
    always_ff @(posedge clk_i) if (hold_wr) hold_mem[tg_idx[RQL-1]] <= rq_d;
    add_lanes #(.NUM_LANES(NUM_LANES), .TAG_W(ATW), .ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) addl (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(add_in), .q1_i(hold_mem[hold_rd_idx]), .q2_i(rq_d),
        .tag_i({tg_byp[RQL-1], tg_bank[RQL-1], tg_pack[RQL-1], tg_idx[RQL-1]}),
        .zp1_i(g_add_w[159:152]), .zp2_i(g_add_w[151:144]), .m1_i(g_add_w[143:112]), .s1_i(g_add_w[111:104]), .m2_i(g_add_w[103:72]),
        .s2_i(g_add_w[71:64]), .mo_i(g_add_w[63:32]), .so_i(g_add_w[31:24]), .zpo_i(g_add_w[23:16]), .min_i(g_add_w[15:8]),
        .max_i(g_add_w[7:0]),
        .valid_o(add_v), .result_o(add_d), .tag_o(add_tag));
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) at_v <= '0;
      else at_v <= {at_v[QAL-2:0], add_in};
    // The fill sees one source a cycle (a_fill_one_source): requantize's non-ADD beats, or add_lanes' beats with their own tags.
    logic f_byp, f_bank, f_pack;
    logic [FCNT_W-1:0] f_idx;
    assign fill_v   = rq_out || add_v;
    assign fill_d   = add_v ? add_d : rq_d;
    assign fill_add = add_v;
    assign {f_byp, f_bank, f_pack, f_idx} = add_v ? add_tag : {tg_byp[RQL-1], tg_bank[RQL-1], tg_pack[RQL-1], tg_idx[RQL-1]};
    assign rq_drain = |tg_v || |at_v;
    // Beats granted now arrive 2 or more cycles later; after an add set, grant only once no add beat is in requantize or add_lanes.
    localparam int RQ_HOLD = (RQL > 2) ? RQL - 2 : 0;
    always_comb begin
      rq_room = !(|(tg_v & tg_add)) && !(|at_v);
      for (int i = 0; i < RQ_HOLD; i++) if (tg_v[i]) rq_room = 1'b0;
    end
    assign lane_v     = fill_v && !f_byp;
    assign byp_wr     = fill_v && f_byp;
    assign byp_bank   = f_bank;
    assign byp_pack   = f_pack;
    assign byp_idx    = f_idx;
    // The bank is full when a ReLU or linear set's last beat leaves (from requantize or add_lanes), or when the lanes return a set.
    assign bank_done  = (byp_wr && (f_idx == PER_LANE[FCNT_W-1:0] - 1'b1)) || (g_done && !act_bypass && !g_hold);
    assign bank_sel   = byp_wr ? f_bank : act_wr;
```

(Delete the old `rq_drain`, `RQ_HOLD`/`rq_room`, `lane_v` .. `bank_sel` lines this replaces; `a_rq_tag_aligned` now compares `rq_v == tg_v[RQL-1]`.)

- [ ] **Step 5: Run the residual tests and the existing ones**

```bash
L rs2_res16 64 6 $J/cmds/mk.sh pipeline FMT=int8 N=16 TILE=4 TEST=int8_residual PYTHON=$J/venv/bin/python
L rs2_res8 64 6 $J/cmds/mk.sh pipeline FMT=int8 N=8 TILE=2 TEST=int8_residual PYTHON=$J/venv/bin/python
L rs2_res32 64 12 $J/cmds/int8_O0.sh $J/cmds/mk.sh pipeline FMT=int8 N=32 TILE=8 TEST=int8_residual PYTHON=$J/venv/bin/python
L rs2_r_int8_n16 64 24 $J/cmds/res_g4_pts.sh int8 16
L rs2_r_fp32_n16 64 24 $J/cmds/res_g4_pts.sh fp32 16
```

Expected: the eight residual tests `RESULT: PASSED` at N = 8, 16, 32, each with a `[RESIDUAL]` line (pairs equal to the stream's groups); across the eight tests at least one non-zero count of each cross (`hold release with an L3 grant`, `hold done with a bank done`, `an add set's last beat with the next accept`) and `grants held for an add drain` > 0 in `int8_residual_then_relu_nopool` (record which crosses stay 0: a 0 is reported as not reachable at that point, as the credits work did, not hidden); no `Assertion failed`. The int8 and fp32 tables PASS, with every pre-existing test's words and cycles identical to `rs2_side_*_n16` (the ADD path carries only add-set beats).

- [ ] **Step 6: Commit (SIENNA), RTL first**

```bash
git -C $T add src/sienna_top.sv
git -C $T commit -m "sienna_top: the int8 residual ADD. A hold set's requantized beats go to hold_mem (PER_LANE x NUM_LANES bytes, N^2) by beat index, with no bypass write, no bank_done and no act_wr toggle; once its last beat is in it returns its L6 reservation, joins gp_sets and pooling passes it as a null, like a partial set. An add set's beats leave requantize into add_lanes with hold_mem's beat of the same index and the add set's own ADD words, and come back into the fill (fill_v, fill_d) with their tags; only add beats take the extra qadd_lat() cycles. The grant after an add set waits until no add beat is in requantize or add_lanes (rq_room, and rq_drain covers add_lanes), so the two sources never present a beat together. An add set's lanes take the ADD's zp_o as their input zero point, and its ReLU or linear drop value is zp_o. Runs rs2_res8, rs2_res16, rs2_res32: the eight residual tests bit-exact; every other test's words and cycles unchanged (rs2_r_*)."
git -C $T add regression.py testbenches/TB_sienna_top.sv
git -C $T commit -m "regression.py and TB_sienna_top: eight int8 residual tests (relu, linear_nopool, tanh with 2 depth passes, sigmoid_selu_nopool, train, then_relu_nopool, two_layers_nopool, accum2_mixed_nopool), zp_random and req_random with random ADD scales; per set add_<k>.mem (flags partial, bias, hold, add, then the ADD's words) and the knob residual. The golden is the conv's requantize, ipu.tflite_add, the lane with z_in = zp_o, pooling and dropout to zp_o; its self-test checks it element by element against ipu and that zp1, zp_o and the lane's zero point each move it. The TB counts hold/add pairs and the same-cycle crosses (a hold release with an L3 grant, a hold done with a bank done, an add set's last beat with the next accept, grants held for an add drain); a residual test without the pair count fails. perf skips residual tests (not modelled)."
git -C $T add model_runner.py && git -C $T commit -m "model_runner numerics: add_lane_params (the lane's words after an ADD: z_in = zp_o, the rescale of s_out, SELU's output calibrated on the ADD's output) and int8_residual_exact (conv requantize, TFLite's ADD with the residual codes, the lane)."
```

### Task 2.7: The new assertions, and every mutant the spec names

**Files:**
- Modify: `src/sienna_top.sv` (assertions in `G_REQ`'s `ifndef SYNTHESIS` block and in the accept-term section)
- Create (local): `$J/res_mut/m_pair.patch`, `m_push.patch`, `m_off.patch`, `m_room.patch`, `m_par.patch`, `m_bank.patch`, `m_zp1.patch`, `m_drop.patch`, `m_zin.patch`

**Interfaces:**
- Produces: `a_hold_then_add`, `a_hold_push_empty`, `a_add_pop_full`, `a_fill_one_source`, `a_hold_no_bank`, `a_add_par_stable` (Task 2.3 added `a_float_no_add`), each on internal or registered terms (no host input sampled at its edge).

- [ ] **Step 1: Write the assertions** — in `G_REQ`'s `ifndef SYNTHESIS` block, after `a_rq_one_set`:

```systemverilog
    logic [PER_LANE-1:0] hm_v;  // hold_mem's beats written by a hold set and not yet read by its add set
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) hm_v <= '0;
      else begin
        if (hold_wr) hm_v[tg_idx[RQL-1]] <= 1'b1;
        if (add_in && int'(hold_rd_idx) < PER_LANE) hm_v[hold_rd_idx] <= 1'b0;
      end
    a_hold_push_empty: assert property (@(posedge clk_i) disable iff (!rstn_i) hold_wr |-> !hm_v[tg_idx[RQL-1]])
      else $error("sienna_top: a_hold_push_empty: hold beat %0d overwrote one its add set had not read", tg_idx[RQL-1]);
    a_add_pop_full: assert property (@(posedge clk_i) disable iff (!rstn_i) add_in |-> (int'(hold_rd_idx) < PER_LANE) && hm_v[hold_rd_idx])
      else $error("sienna_top: a_add_pop_full: add beat %0d read hold_mem where no hold beat waits", hold_rd_idx);
    a_fill_one_source: assert property (@(posedge clk_i) disable iff (!rstn_i) !(rq_out && add_v))
      else $error("sienna_top: a_fill_one_source: a requantized beat and an add beat reached the fill in the same cycle");
    logic hold_left_q, act_wr_q;  // a hold set left the stage at the last edge; act_wr before that edge
    logic [159:0] g_add_w_q;  // the ADD words before the last edge
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) {hold_left_q, act_wr_q} <= '0;
      else {hold_left_q, act_wr_q} <= {g_done && g_hold, act_wr};
    always_ff @(posedge clk_i) g_add_w_q <= g_add_w;
    a_hold_no_bank: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                     !((g_state != G_IDLE) && g_hold && !act_bypass) && !(hold_wr && byp_wr) && !(hold_left_q && (act_wr != act_wr_q)))
      else $error("sienna_top: a_hold_no_bank: a hold set runs a lane activation, writes a bank or toggles act_wr");
    a_add_par_stable: assert property (@(posedge clk_i) disable iff (!rstn_i) (|(tg_v & tg_add)) |-> (g_add_w == g_add_w_q))
      else $error("sienna_top: a_add_par_stable: the ADD words changed under an add beat still in requantize");
```

In the accept-term section (`ifndef SYNTHESIS` near line 1170), beside `last_accum`: `logic acc_prev_hold_q, last_hold;  // the accept before this one was a hold set`, reset with the others, `acc_prev_hold_q <= last_hold;` in the registered tuple and `if (host_accept) last_hold <= side.res_hold;`, then:

```systemverilog
  if (IS_INT) begin : G_HOLD_PAIRS
    a_hold_then_add: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                      acc_q |-> (acc_add_q == acc_prev_hold_q) && !(acc_hold_q && acc_add_q) &&
                                                !((acc_hold_q || acc_add_q) && (acc_accum_q || acc_shift_q != '0)))
      else $error("sienna_top: a_hold_then_add: hold and add sets come in pairs, hold first, neither a partial sum nor packed");
  end
```

- [ ] **Step 2: The mutants** (spec G4: a hold read one beat off, zp_1 taken from the add set, the drop value from `pe_zp`, `rq_room` unwidened; and one per assertion)

```bash
S=src/sienna_top.sv
$J/cmds/res_mkmut.sh m_pair regression.py "[0b1000] + [add[f] for f in mr.ADD_FIELDS]" "[0b0100] + [add[f] for f in mr.ADD_FIELDS]"
$J/cmds/res_mkmut.sh m_push $S "assign hold_wr     = rq_v && tg_hold[RQL-1];" "assign hold_wr     = rq_v && (tg_hold[RQL-1] || tg_add[RQL-1]);"
$J/cmds/res_mkmut.sh m_off $S "assign hold_rd_idx = tg_idx[RQL-1];" "assign hold_rd_idx = tg_idx[RQL-1] + 1'b1;"
$J/cmds/res_mkmut.sh m_room $S "rq_room = !(|(tg_v & tg_add)) && !(|at_v);" "rq_room = 1'b1;"
$J/cmds/res_mkmut.sh m_par $S "if (g_accept) g_add_w <= s_add[g_next_id];" "if (g_accept || wide_rd_valid) g_add_w <= s_add[g_next_id];"
$J/cmds/res_mkmut.sh m_bank $S "if (g_done && !g_hold) act_wr <= ~act_wr;" "if (g_done) act_wr <= ~act_wr;"
$J/cmds/res_mkmut.sh m_zp1 $S ".zp1_i(g_add_w[159:152])" ".zp1_i(g_zp[0])"
$J/cmds/res_mkmut.sh m_drop $S "p_zp[k] = pe_add ? pe_add_zpo : pe_zp[p_lane_ent[k]];" "p_zp[k] = pe_zp[p_lane_ent[k]];"
$J/cmds/res_mkmut.sh m_zin $S ".gp_zin_i      (g_add ? g_add_zpo : g_zp[g])," ".gp_zin_i      (g_zp[g]),"
for m in m_pair m_push m_off m_room m_par m_bank m_zp1 m_drop m_zin; do
  L rs2_mut_$m 64 6 $J/cmds/res_mut.sh $J/res_mut/$m.patch $J/cmds/mk.sh pipeline FMT=int8 N=16 TILE=4 TEST=int8_residual PYTHON=$J/venv/bin/python
done
```

(The `m_pair` patch's OLD string is Task 2.5's add-set flags expression; if it was written differently, use that line.)

- [ ] **Step 3: Read each verdict**

| Mutant | Must show |
|---|---|
| `m_pair` | `a_hold_then_add` in `Assertion failed`, residual tests FAILED |
| `m_push` | `a_hold_push_empty` |
| `m_off` | `a_add_pop_full`, and wrong outputs (FAILED) |
| `m_room` | `a_fill_one_source` (in `int8_residual_then_relu_nopool` or `_accum2_mixed_nopool`) |
| `m_par` | `a_add_par_stable`, wrong outputs |
| `m_bank` | `a_hold_no_bank` |
| `m_zp1` | FAILED outputs in the `zp_random` tests (zp_1 is never 0 there) |
| `m_drop` | `int8_residual_train` FAILED (drops to 0, not zp_o) |
| `m_zin` | `int8_residual_tanh` and `_sigmoid_selu_nopool` FAILED (Review Focus 1) |

A mutant whose run passes means the tests cannot see that bug: rerun it at `N=8 TILE=2`; if it still passes, strengthen the stream (more sets, a test whose schedule reaches the cross) before going on, and say which mutant needed it in the commit message. Record each run and verdict in the ledger.

- [ ] **Step 4: The unmutated tree stays clean** — `L rs2_asrt16 64 6 $J/cmds/mk.sh pipeline FMT=int8 N=16 TILE=4 PYTHON=$J/venv/bin/python` (the whole int8 pipeline suite) → every test PASSED, no `Assertion failed`.

- [ ] **Step 5: Commit (SIENNA)**

```bash
git -C $T add src/sienna_top.sv
git -C $T commit -m "sienna_top: simulation assertions for the int8 residual path, each shown firing by a mutant (rs2_mut_*): a_hold_then_add (hold and add sets in pairs, hold first, neither partial nor packed; registered accept terms), a_hold_push_empty and a_add_pop_full (hold_mem's per-beat occupancy), a_fill_one_source (requantize and add_lanes never fill in one cycle; fires when rq_room is not widened), a_hold_no_bank (a hold set runs no lane activation, writes no bank, keeps act_wr), a_add_par_stable (the ADD words hold while add beats are in requantize). The spec's mutants also fail the tests: a hold read one beat off, zp_1 from the add set, the drop value from pe_zp, and the lanes' zero point from the identity requantize."
```

### Task 2.8: `sienna_layer`, the `A` line, and the layer-level cases

**Files:**
- Modify: `src/sienna_layer.sv`, `testbenches/TB_model_run.sv`, `model_runner.py` (Device protocol: `write_layer`; Backends: `RtlLayer.run_job`), `regression.py` (`--action residual`, self-tests)

**Interfaces:**
- Consumes: Tasks 2.4 to 2.6; `mr.add_words`, `mr.add_lane_params`, `mr.int8_residual_exact`, `mr.requant_params`, `mr.requantize`, `mr.quant_act`, `mr.activation_range`.
- Produces:
  - `sienna_layer` inputs `cfg_add_zp2_i`, `cfg_add_s1_i`, `cfg_add_s2_i`, `cfg_add_so_i`, `cfg_add_zpo_i`, `cfg_add_min_i`, `cfg_add_max_i` [7:0], `cfg_add_m1_i`, `cfg_add_m2_i`, `cfg_add_mo_i` [31:0], taken with `cfg_load_i`; for an int8 residual layer pass dt-1 is a hold set and pass dt an add set; `a_int_no_residual` is gone.
  - Layer file: `A zp2 m1 s1 m2 s2 mo so zpo amin amax` (decimal) right after `Q` for an int8 residual layer; zp1 is `Q`'s zp.
  - `model_runner.write_layer(path, cfg, a, w, fmt, req=None, pack=None, epilogue=None, bias_words=None, add=None)`; `RtlLayer.run_job(job, tag, plusargs=())` passes `job["add"]` and records `self.first_hold` (the TB's `[RES] first hold_done at cycle C`, or None).
  - `regression.py --action residual [--n N] [--tile T] [--lanes L] [--cases ...] [--tflite] [--mid-reset]` printing per case `residual <case> N=<n> T=<t>: ...` and finally `RESIDUAL: PASS|FAIL`.

- [ ] **Step 1: Ports only, then the bench and the protocol (the failing test)**

`sienna_layer.sv`: after `cfg_gp_zout_i` add

```systemverilog
    // int8 residual only, taken with cfg_load_i: the ADD's words; input 1 is the conv's output (zero point cfg_req_zp_i), input 2 the residual
    input logic [7:0]               cfg_add_zp2_i,
    input logic [31:0]              cfg_add_m1_i,
    input logic [7:0]               cfg_add_s1_i,
    input logic [31:0]              cfg_add_m2_i,
    input logic [7:0]               cfg_add_s2_i,
    input logic [31:0]              cfg_add_mo_i,
    input logic [7:0]               cfg_add_so_i,
    input logic [7:0]               cfg_add_zpo_i,
    input logic [7:0]               cfg_add_min_i,
    input logic [7:0]               cfg_add_max_i,
```

with registers `logic [7:0] add_zp2_q, add_s1_q, add_s2_q, add_so_q, add_zpo_q, add_min_q, add_max_q; logic [31:0] add_m1_q, add_m2_q, add_mo_q;` beside `rq_zp_q`, reset to 0 and loaded under `if (cfg_load_i && !active)` like `rq_zp_q`.

`TB_model_run.sv`: `logic [7:0] cfg_add_zp2_i = '0, cfg_add_s1_i = '0, cfg_add_s2_i = '0, cfg_add_so_i = '0, cfg_add_zpo_i = '0, cfg_add_min_i = '0, cfg_add_max_i = '0; logic [31:0] cfg_add_m1_i = '0, cfg_add_m2_i = '0, cfg_add_mo_i = '0;` connected by name in the `sienna_layer` instance; the header comment gains `// int8 residual: an "A zp2 m1 s1 m2 s2 mo so zpo amin amax" line after Q (zp1 is Q's zp).`; after the `Q` parse:

```systemverilog
    {azp2, am1, as1, am2, as2, amo, aso, azpo, aamin, aamax} = '0;
    if (IS_INT && res) begin
      rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d %d", kind, azp2, am1, as1, am2, as2, amo, aso, azpo, aamin, aamax);
      if (rc != 11 || kind != "A") begin
        $display("[FATAL] %s: an int8 residual layer file needs its A line", layer_f);
        $finish;
      end
    end
```

(`int azp2, am1, as1, am2, as2, amo, aso, azpo, aamin, aamax;` with the other parse variables), and in the configure block beside `cfg_gp_zout_i = 8'(qzout);`: `cfg_add_zp2_i = 8'(azp2); cfg_add_m1_i = 32'(am1); cfg_add_s1_i = 8'(as1); cfg_add_m2_i = 32'(am2); cfg_add_s2_i = 8'(as2); cfg_add_mo_i = 32'(amo); cfg_add_so_i = 8'(aso); cfg_add_zpo_i = 8'(azpo); cfg_add_min_i = 8'(aamin); cfg_add_max_i = 8'(aamax);`. A monitor for Review Focus 5:

```systemverilog
  int first_hold = -1;  // the cycle of the first hold set's last beat into hold_mem, for +mid_reset
  always @(posedge clk_i) if (IS_INT && dut.pipe.hold_done && first_hold < 0) first_hold = cycle;
```

printed beside the `[LAYER]` line: `if (first_hold >= 0) $display("[RES] first hold_done at cycle %0d", first_hold);` (use the TB's own cycle counter and `sienna_layer` instance name).

`model_runner.py` (Device protocol), `write_layer` gains `add=None` and after the `Q` line:

```python
        if fmt == "int8" and cfg["residual"]:  # zp1 is Q's zp, the conv output's
            if add is None or int(add["zp1"]) != int(req["zp"]):
                raise ValueError("an int8 residual layer needs its ADD words (job['add']) with zp1 equal to the requantize zp")
            f.write("A " + " ".join(str(int(add[k])) for k in ADD_FIELDS[1:]) + "\n")
```

`RtlLayer.run_job(self, job, tag, plusargs=())`: `write_layer(..., job.get("bias_words"), add=job.get("add"))`, run `[self.bin, f"+layer={lf}", f"+out={of}", *plusargs]`, and `h = re.search(r"\[RES\] first hold_done at cycle (\d+)", r.stdout); self.first_hold = int(h.group(1)) if h else None`.

`regression.py`, a new section and action (`"residual": residual_main` in `MAINS`, `"residual"` in `ACTIONS`):

```python
# ── residual: int8 ADD layers on sienna_layer (sienna-int8-residual) ─────────

RES_CASES = {  # name: N -> (rows M, depth K, columns C, conv clamp, ADD's fused activation, the lane after the ADD)
    "resnet_relu": lambda N: (64, 144, 16, "none", "relu", "relu"),  # ResNet-8's shape: 8x8 rows, 3x3x16 depth, 16 channels
    "deep_cached": lambda N: (2 * N, 3 * N + 1, N, "none", "relu", "relu"),  # dt 4, rt 2: cached
    "deep_uncached": lambda N: (N, 3 * N + 1, N, "none", "none", "linear"),  # dt 4, rt 1: uncached
    "tiles": lambda N: (2 * N + 3, N + 2, 2 * N + 5, "none", "relu", "relu"),  # rt 3, ct 3, dt 2
    "relu6_conv_linear_add": lambda N: (2 * N, 2 * N, N, "relu6", "none", "linear"),  # Review Focus 3
    "tanh_after_add": lambda N: (N, 2 * N, N, "none", "none", "tanh"),
}


def residual_case(name: str, N: int, rng) -> tuple:
    """One int8 residual layer job and its bit-exact output: random codes and per-channel words, random ADD scales, and two outputs a wrong ADD would give."""
    M, K, C, conv_act, add_act, lane_act = RES_CASES[name](N)
    A_q = rng.randint(-128, 128, (M, K)).astype(np.int64)
    W_q = rng.randint(-127, 128, (K, C)).astype(np.int64)
    s_w = np.exp(rng.uniform(np.log(1e-3), np.log(2e-2), C))
    hw_bias = rng.randint(-2000, 2000, C).astype(np.int64)
    acc = wrap32(imatmul(A_q, W_q) + hw_bias[None, :])
    rq = requant_params(acc, 0.02, s_w, "linear" if conv_act == "none" else "relu", np.random.RandomState(rng.randint(1 << 30)))
    rq["amin"], rq["amax"] = mr.activation_range(conv_act, rq["s_out"], rq["zp"])
    q1 = requantize(acc, rq)
    s2, z2 = rq["s_out"] * 2.0 ** rng.uniform(-3.0, 3.0), int(rng.randint(-100, 101))
    R_q = rng.randint(-128, 128, (M, C)).astype(np.int64)
    y = rq["s_out"] * (q1 - rq["zp"]) + s2 * (R_q - z2)
    _, s_out, z_out = quant_act(np.maximum(y, 0.0) if add_act == "relu" else y)
    s_out *= 2.0 ** rng.uniform(-0.5, 0.5)
    add = mr.add_words(rq["s_out"], rq["zp"], s2, z2, s_out, z_out, add_act)
    ra = mr.add_lane_params(add, s_out, mr.tflite_add_codes(q1, R_q, add))
    want = mr.int8_residual_exact(A_q, W_q, hw_bias, rq, R_q, add, ra, lane_act)
    wrong = (mr.int8_residual_exact(A_q, W_q, hw_bias, rq, R_q, dict(add, zp1=0), ra, lane_act),  # zp_1 taken as the add set's 0
             activate_int8(q1, lane_act, ra))  # no ADD at all
    req = dict(rq, mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])  # Q: the conv's zp and clamp, the lane's words
    job = {"terms": [(A_q.astype(np.float32), W_q.astype(np.float32)), (R_q.astype(np.float32), np.eye(C, dtype=np.float32))],
           "bias": hw_bias, "act": lane_act, "shape": (M, C), "req": req, "add": add}
    return job, want, wrong


def residual_main(argv=None) -> None:
    """int8 residual layers on sienna_layer (TB_model_run), bit-exact against int8_residual_exact; --tflite adds the residual-block TFLite models, --mid-reset a reset with hold_mem full."""
    ap = argparse.ArgumentParser(description=residual_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="residual")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--cases", nargs="*", default=list(RES_CASES))
    ap.add_argument("--tflite", action="store_true")
    ap.add_argument("--mid-reset", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "residual"))
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    sim = mr.RtlLayer(a.n, a.lanes, a.work, "int8", a.tile)
    sim.build()
    bad = []
    for name in a.cases:
        job, want, wrong = residual_case(name, a.n, np.random.RandomState(a.seed + sorted(RES_CASES).index(name)))
        M, K, C = RES_CASES[name](a.n)[:3]
        rt, ct, dt = -(-M // a.n), -(-C // a.n), -(-K // a.n)
        Y, sets, cyc = sim.run_job(job, name)
        mism, w1, w2 = int(np.sum(Y != want)), int(np.sum(want != wrong[0])), int(np.sum(want != wrong[1]))
        print(f"residual {name} N={a.n} T={a.tile}: {M} x {K} x {C}, {Y.size} outputs, {mism} mismatches, sets {sets} "
              f"(expected {rt * ct * (dt + 1)}), cycles {cyc}; the golden differs from zp_1 = 0 on {w1}, from no ADD on {w2}", flush=True)
        if mism or sets != rt * ct * (dt + 1):
            bad.append(name)
        if (job["add"]["zp1"] != 0 and w1 == 0) or w2 == 0:
            bad.append(f"{name} (its golden cannot tell a wrong ADD)")
        if a.mid_reset and name == "tiles":
            bad += residual_mid_reset(sim, job, Y, want)
    if a.tflite:
        bad += residual_tflite(sim, a.seed)
    print(f"RESIDUAL: {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
    sys.exit(1 if bad else 0)
```

(`residual_mid_reset` and `residual_tflite` come in Task 2.9; until then guard their calls with the flags, which stay off.)

Self-test (`import tempfile` at the top if the merged file lacks it):

```python
@selftest
def test_layer_file_add_line():
    """An int8 residual layer file carries A after Q (zp2 .. amax); write_layer refuses one without ADD words or with zp1 off the Q line's zp."""
    job, _, _ = residual_case("deep_uncached", 8, np.random.RandomState(1))
    cfg, a_rows, w_rows, _ = mr.format_layer(job, 8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.layer")
        mr.write_layer(p, cfg, a_rows, w_rows, "int8", job["req"], None, mr.layer_epilogue(job, 8), add=job["add"])
        lines = open(p).read().splitlines()
        assert lines[1].startswith("Q ") and lines[2].split()[0] == "A" and len(lines[2].split()) == 11, lines[:3]
        assert [int(v) for v in lines[2].split()[1:]] == [job["add"][k] for k in mr.ADD_FIELDS[1:]]
        for add in (None, dict(job["add"], zp1=job["req"]["zp"] + 1)):
            try:
                mr.write_layer(p, cfg, a_rows, w_rows, "int8", job["req"], None, mr.layer_epilogue(job, 8), add=add)
                raise AssertionError("write_layer accepted a residual layer with bad ADD words")
            except ValueError:
                pass
```

Run: `python3 $T/regression.py --action selftest` (login node) → passes (the protocol exists); then `L rs2_lay_fail 64 4 $J/cmds/mk.sh residual FMT=int8 N=16 TILE=4 PYTHON=$J/venv/bin/python` (Makefile target from Task 2.10 Step 2; until then `$J/cmds/res_py.sh . regression.py --action residual --n 16 --tile 4`) → every case raises `an assertion or a testbench check failed in the layer simulation` (`a_int_no_residual` fires), `RESIDUAL: FAIL ...`.

- [ ] **Step 2: Implement the marking** — `sienna_layer.sv`:

```systemverilog
  logic int_res;  // an int8 residual layer: pass dt-1 holds, pass dt adds
  assign int_res = IS_INT && res_q;
```

line 333: `p_acc      = int_res ? (is_p < dt - 1) : (is_p != np - 1);  // int8 residual: the hold set closes the conv's sum, the add set is its own`; at the end of the `p_side` `always_comb`:

```systemverilog
    p_side.res_hold = int_res && (is_p == dt - 1);  // the conv's words and clamp, linear: its beats wait in hold_mem
    p_side.res_add  = int_res && is_res_pass;  // identity requantize, the ADD's words, the layer's activation
    if (p_side.res_hold) p_side.act[0] = CONTROL_WIDTH'(3'b101);
    if (p_side.res_add) begin
      for (int c = 0; c < N; c++) begin
        p_side.mult[c]  = 32'h4000_0000;
        p_side.shift[c] = 8'd1;
      end
      p_side.zp[0]   = '0;
      p_side.amin[0] = 8'h80;
      p_side.amax[0] = 8'h7F;
      {p_side.add_zp1, p_side.add_zp2, p_side.add_m1, p_side.add_s1, p_side.add_m2, p_side.add_s2, p_side.add_mo, p_side.add_so,
       p_side.add_zpo, p_side.add_min, p_side.add_max} = {rq_zp_q, add_zp2_q, add_m1_q, add_s1_q, add_m2_q, add_s2_q, add_mo_q,
                                                          add_so_q, add_zpo_q, add_min_q, add_max_q};
    end
```

Delete the `G_INT_NO_RESIDUAL` block (lines 553-556). Comments: line 9 gains "int8: the depth passes' last is a hold set and the residual pass an add set (sienna-int8-residual)"; `ONE`'s comment becomes `// int8: the identity pass gives the residual's codes exactly in int32`.

- [ ] **Step 3: Run the cases to see them pass**

```bash
for n in 8 16; do for t in 2 4; do L rs2_lay_n${n}_t$t 64 6 $J/cmds/res_py.sh . regression.py --action residual --n $n --tile $t; done; done
L rs2_lay_n32_t8 64 12 $J/cmds/int8_O0.sh $J/cmds/res_py.sh . regression.py --action residual --n 32 --tile 8 --lanes 32
```

Expected: every case `0 mismatches`, `sets` equal to expected, both "the golden differs" counts above 0 (above 0 for zp_1 wherever zp_1 != 0), `RESIDUAL: PASS`. Also the float residual layer is unchanged: `L rs2_lay_fp32 64 6 $J/cmds/mk.sh gemm FMT=fp32 N=16 QUICK=1 PYTHON=$J/venv/bin/python` and `pack` match `rs2_base_fp32_n16`'s `gemm`/`pack` points (they include the cached residual layer).

- [ ] **Step 4: Commit (SIENNA), RTL first**

```bash
git -C $T add src/sienna_layer.sv
git -C $T commit -m "sienna_layer: int8 residual layers. The layer takes the ADD's words (cfg_add_zp2, m1, s1, m2, s2, mo, so, zpo, min, max; zp_1 is cfg_req_zp) with cfg_load_i; its last depth pass becomes a hold set (closes the conv's sum, the conv's words and clamp, linear) and the residual pass an add set (its own group, no bias, identity requantize: mult 2^30, shift 1, zp 0, clamp -128..127; the ADD's words; the layer's activation and GPNAE words). a_int_no_residual goes; a packed residual layer is still refused; float layers are unchanged. TB_model_run connects the new inputs in the next commit."
git -C $T add testbenches/TB_model_run.sv
git -C $T commit -m "TB_model_run: an int8 residual layer file carries an A line after Q (zp2 m1 s1 m2 s2 mo so zpo amin amax), driven onto cfg_add_*; the TB prints the cycle of the first hold set's last beat into hold_mem, for +mid_reset."
git -C $T add model_runner.py
git -C $T commit -m "model_runner device protocol: write_layer writes the A line from job['add'] for an int8 residual layer and refuses one without it or with zp1 off the Q line's zp; RtlLayer.run_job passes it, takes plusargs, and reads the first hold_done cycle."
git -C $T add regression.py
git -C $T commit -m "regression.py --action residual: int8 residual layers on sienna_layer bit-exact against int8_residual_exact (resnet_relu 64 x 144 x 16, deep_cached and deep_uncached dt 4, tiles rt 3 ct 3, relu6_conv_linear_add, tanh_after_add), random per-channel words and ADD scales, set counts rt ct (dt + 1); each case fails if its golden cannot tell zp_1 = 0 or a missing ADD. Self-test test_layer_file_add_line. Runs rs2_lay_*."
```

### Task 2.9: Residual-block TFLite models, and a reset with `hold_mem` full

**Files:**
- Modify: `regression.py` (`residual_tflite`, `residual_mid_reset`)

**Interfaces:**
- Consumes: `mr.im2col`, `mr.layer_multipliers`, `mr.activation_range`, `mr.add_words`, `mr.RtlLayer`; TensorFlow from the venv (`--tflite` only).
- Produces: `residual_tflite(sim, seed) -> list[str]` (failures), lines `tflite <model>: ...` with `ADD input 1 is the residual: yes|no`; `residual_mid_reset(sim, job, Y, want) -> list[str]`.

- [ ] **Step 1: Write them**

```python
def _tf_res_models(tf, seed: int) -> dict:
    """Two int8 residual blocks on 8x8x16 from the converter: conv 3x3 + ADD(x) + ReLU, and conv 3x3 / 2 + ADD(conv 1x1 / 2) + ReLU."""
    rng = np.random.default_rng(seed)
    w3 = tf.constant((rng.standard_normal((3, 3, 16, 16)) * 0.08).astype(np.float32))
    w1 = tf.constant((rng.standard_normal((1, 1, 16, 16)) * 0.25).astype(np.float32))
    b3 = tf.constant((rng.standard_normal(16) * 0.1).astype(np.float32))
    b1 = tf.constant((rng.standard_normal(16) * 0.1).astype(np.float32))
    spec = [tf.TensorSpec((1, 8, 8, 16), tf.float32)]

    @tf.function(input_signature=spec)
    def block(x):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w3, 1, "SAME"), b3) + x)

    @tf.function(input_signature=spec)
    def block_s2(x):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w3, 2, "SAME"), b3) + tf.nn.bias_add(tf.nn.conv2d(x, w1, 2, "VALID"), b1))

    rep = np.random.default_rng(seed + 1)
    out = {}
    for name, fn in (("block", block), ("block_s2", block_s2)):
        holder = tf.Module()
        holder.f = fn
        conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], holder)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([rep.uniform(-1.0, 2.0, (1, 8, 8, 16)).astype(np.float32)] for _ in range(100))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        out[name] = conv.convert()
    return out


def _conv_job(it, td, op, x, k: int, stride: int, same: bool) -> tuple:
    """One CONV_2D op of the interpreter's graph as an int8 job on input codes x (images, h, w, c): im2col with the input zero point, the folded bias, its words."""
    xi, wi, bi = (int(i) for i in op["inputs"][:3])
    oi = int(op["outputs"][0])
    q = lambda i: td[i]["quantization_parameters"]
    z_in, z_out, s_in, s_out = int(q(xi)["zero_points"][0]), int(q(oi)["zero_points"][0]), float(q(xi)["scales"][0]), float(q(oi)["scales"][0])
    w = it.get_tensor(wi).astype(np.int64)  # OHWI
    cols = [mr.im2col(np.asarray(xx, np.float32), k, k, (stride, stride), same, pad_value=float(z_in)) for xx in x]
    W = w.reshape(w.shape[0], -1).T
    hw_bias = wrap32(it.get_tensor(bi).astype(np.int64) - z_in * W.sum(axis=0))
    mult, shift = mr.layer_multipliers("conv", q(wi)["scales"].astype(np.float32), np.float32(s_in), np.float32(s_out), w.shape[0], mr.ROUNDING)
    amin, amax = mr.activation_range("none", s_out, z_out)
    req = dict(mult=mult, shift=shift, zp=z_out, amin=amin, amax=amax, mx=0, shx=0, mout=0, shout=0, zout=0)
    X = np.vstack([c for c, _ in cols]).astype(np.float32)
    return X, W.astype(np.float32), hw_bias, req, (s_out, z_out), oi


def residual_tflite(sim, seed: int) -> list:
    """The residual-block TFLite models through TB_model_run, every output equal to the interpreter's (BUILTIN_REF); the 1x1 shortcut runs as its own layer."""
    _tf_imports()
    bad = []
    x = np.random.default_rng(seed + 2).integers(-128, 128, (4, 8, 8, 16)).astype(np.int64)
    for name, model in _tf_res_models(tf, seed).items():
        it = tf.lite.Interpreter(model_content=model, experimental_op_resolver_type=REF, experimental_preserve_all_tensors=True)
        it.allocate_tensors()
        ops, td = it._get_ops_details(), {t["index"]: t for t in it.get_tensor_details()}
        kinds = sorted(o["op_name"] for o in ops)
        if kinds != (["ADD", "CONV_2D"] if name == "block" else ["ADD", "CONV_2D", "CONV_2D"]):
            bad.append(f"{name}: the converter made {kinds}")
            continue
        i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        tens, ys = {}, []
        for img in x:  # one image at a time, keeping every tensor
            it.set_tensor(i_in, img[None].astype(np.int8))
            it.invoke()
            ys.append(it.get_tensor(i_out)[0].astype(np.int64))
            for o in ops:
                tens.setdefault(int(o["outputs"][0]), []).append(it.get_tensor(int(o["outputs"][0]))[0].astype(np.int64))
        add_op = next(o for o in ops if o["op_name"] == "ADD")
        convs = {int(o["outputs"][0]): o for o in ops if o["op_name"] == "CONV_2D"}
        qa = lambda i: td[i]["quantization_parameters"]
        s_o, z_o = float(qa(i_out)["scales"][0]), int(qa(i_out)["zero_points"][0])
        a1, a2 = (int(i) for i in add_op["inputs"][:2])
        if name == "block":
            main = convs[a1] if a1 in convs else convs[a2]
            res_i = a2 if a1 in convs else a1  # the graph's input x
            X, W, hb, req, (s_c, z_c), oc = _conv_job(it, td, main, x, 3, 1, True)
            R = x.reshape(-1, 16).astype(np.float32)
            s_r, z_r = float(qa(res_i)["scales"][0]), int(qa(res_i)["zero_points"][0])
        else:
            short = next(o for o in convs.values() if td[int(o["inputs"][1])]["shape"][1] == 1)
            main = next(o for o in convs.values() if o is not short)
            Xs, Ws, hbs, reqs, (s_r, z_r), os_ = _conv_job(it, td, short, x, 1, 2, False)
            S, _, _ = sim.run_job({"terms": [(Xs, Ws)], "bias": hbs, "act": "linear", "shape": (Xs.shape[0], 16), "req": reqs}, f"{name}_short")
            ws = np.stack(tens[os_]).reshape(-1, 16)
            print(f"tflite {name} shortcut layer: {S.size} outputs, {int(np.sum(S != ws))} differ from the interpreter's tensor")
            if np.any(S != ws):
                bad.append(f"{name} shortcut")
            X, W, hb, req, (s_c, z_c), oc = _conv_job(it, td, main, x, 3, 2, True)
            R = S.astype(np.float32)
            res_i = os_
        add = mr.add_words(s_c, z_c, s_r, z_r, s_o, z_o, "relu")
        job = {"terms": [(X, W), (R, np.eye(16, dtype=np.float32))], "bias": hb, "act": "relu", "shape": (X.shape[0], 16), "req": req, "add": add}
        Y, sets, cyc = sim.run_job(job, name)
        want = np.stack(ys).reshape(-1, 16)
        wc = np.stack(tens[oc]).reshape(-1, 16)
        print(f"tflite {name}: {Y.size} outputs, {int(np.sum(Y != want))} differ from the interpreter (BUILTIN_REF), sets {sets}, cycles {cyc}; "
              f"ADD input 1 is the residual: {'yes' if a1 == res_i else 'no'}; the conv's own codes (q1) checked against its tensor: "
              f"{int(np.sum(mr.requantize(wrap32(imatmul(X.astype(np.int64), W.astype(np.int64)) + hb[None, :]), req) != wc))} differ")
        if np.any(Y != want):
            bad.append(f"tflite {name}")
    return bad


def residual_mid_reset(sim, job, Y0, want) -> list:
    """Review Focus 5: reset one cycle after the first hold set's last beat is in hold_mem; the run after the reset must equal the plain one."""
    if sim.first_hold is None:
        return ["mid-reset: TB_model_run printed no first hold_done"]
    Y, _, _ = sim.run_job(job, "tiles_mid_reset", plusargs=(f"+mid_reset={sim.first_hold + 1}",))
    n = int(np.sum(Y != Y0))
    print(f"residual tiles mid-reset at cycle {sim.first_hold + 1}: {Y.size} outputs, {n} differ from the run without it, "
          f"{int(np.sum(Y != want))} from the golden")
    return [] if n == 0 and np.array_equal(Y, want) else ["mid-reset"]
```

(Check `_tf_imports`, `tf`, `REF`, `mr.im2col`'s and `mr.layer_multipliers`' signatures against the merged file before the run; they are the ones the oracle and `job_of` use on this clone.)

- [ ] **Step 2: Run** (TensorFlow: the venv, on the farm)

```bash
L rs2_tfl16 64 6 $J/cmds/res_py.sh . regression.py --action residual --n 16 --tile 4 --tflite --mid-reset --cases tiles
L rs2_tfl8 64 6 $J/cmds/res_py.sh . regression.py --action residual --n 8 --tile 2 --tflite --mid-reset --cases tiles
L rs2_tfl32 64 12 $J/cmds/int8_O0.sh $J/cmds/res_py.sh . regression.py --action residual --n 32 --tile 8 --tflite --mid-reset --cases tiles
```

Expected: `tflite block: 256 outputs, 0 differ ...` and `tflite block_s2: 64 outputs, 0 differ ...` (4 images: 4 x 64 and 4 x 16 rows of 16), the shortcut layer `0 differ`, the conv's own codes `0 differ`, the `ADD input 1 is the residual` answer recorded (spec: ResNet-8's op 3 lists the residual first); `residual tiles mid-reset ...: 0 differ from the run without it, 0 from the golden`; `RESIDUAL: PASS`; no `Assertion failed` (with `a_add_pop_full` live, a stale `hold_mem` read after the reset would fire). If the converter does not produce the expected ops (`the converter made ...`), report its op list to Soham (spec Risk 2: the residual block's int8 form is assumed from the float model) before changing the model.

- [ ] **Step 3: Commit (SIENNA)**

```bash
git -C $T add regression.py
git -C $T commit -m "regression.py --action residual --tflite: two converter-made int8 residual blocks on 8x8x16 (conv 3x3 + ADD(x) + ReLU; conv 3x3 / 2 + ADD(conv 1x1 / 2) + ReLU, the shortcut as its own layer, since each conv requantizes separately) through TB_model_run, every output equal to the interpreter's BUILTIN_REF output and the shortcut's and the conv's codes equal to its tensors; reports which ADD input is the residual. --mid-reset resets TB_model_run one cycle after the first hold set's last beat is in hold_mem and requires the outputs of the run without the reset. Runs rs2_tfl8, rs2_tfl16, rs2_tfl32."
```

### Task 2.10: `make residual` and the `residual` regression point

**Files:**
- Modify: `Makefile` (target `residual`, help), `regression.py` (`SiennaSpec`: points, commands, verdicts; self-tests that count points)

**Interfaces:**
- Produces: `make residual FMT=int8 N= TILE= LANES= PYTHON=`; point `int8/N<n>_T<t>/residual` at every int8 (N, T), command `regression.py --action residual --n N --tile T --lanes L --tflite --mid-reset`, verdict `^RESIDUAL: PASS` and exit 0 and no firing assertion or `%Error`.

- [ ] **Step 1: The failing self-test** — extend the merged `test_regression_points` (or add `test_residual_points`):

```python
@selftest
def test_residual_points():
    """Every int8 (N, T) has a residual point; no float format has one; its command runs --action residual with --tflite and --mid-reset."""
    pts = SPEC.points(False, ("fp32", "bf16", "int8"))
    nt = {p.rsplit("/", 1)[0] for p in pts if p.startswith("int8/N") and p.endswith("/pipeline")}
    res = {p.rsplit("/", 1)[0] for p in pts if p.endswith("/residual")}
    assert res == nt and all(p.startswith("int8/") for p in pts if p.endswith("/residual")), (sorted(res), sorted(nt))
    argv = SPEC.argv(sorted(p for p in pts if p.endswith("/residual"))[0], {})
    assert "--action" in argv and argv[argv.index("--action") + 1] == "residual" and "--tflite" in argv and "--mid-reset" in argv, argv
```

Run (login node) `python3 regression.py --action selftest` → fails (`res` empty).

- [ ] **Step 2: Implement** — in `SiennaSpec.points`, beside the int8 `tflite` point per (N, T), add `f"int8/N{n}_T{t}/residual"`; in `argv`, for `residual`, `[PY, "regression.py", "--action", "residual", "--n", str(n), "--tile", str(t), "--lanes", str(lanes), "--tflite", "--mid-reset"]` (the venv Python the `tflite` point uses, and its lanes rule for N = 64); in `verdict`, `regress.marker_verdict(text, rc, (r"^RESIDUAL: PASS",))`. Update every self-test that pins the point count (85 with SMALL=1 and 217 in full at `4ccd47b`) by the number of int8 (N, T) pairs. Makefile:

```make
# int8 residual layers and the residual-block TFLite models on TB_model_run (regression.py --action residual); int8 only
residual:
	@[ "$(FMT)" = int8 ] || { echo "make residual is int8 only (FMT=$(FMT))"; exit 1; }
	$(PYTHON) regression.py --action residual --n $(N) --tile $(TILE) --lanes $(LANES) --tflite --mid-reset
```

with `residual` in `.PHONY` and the help text beside `tflite`.

- [ ] **Step 3: Run** — `python3 regression.py --action selftest` → `ALL <n> PASSED`; `python3 regression.py --action points --formats int8 | tr ' ' '\n' | command grep -c residual` → the int8 (N, T) count; `L rs2_pt 64 6 make regression/int8/N16_T4/residual PYTHON=$J/venv/bin/python` → PASS.

- [ ] **Step 4: Commit (SIENNA)**

```bash
git -C $T add regression.py Makefile
git -C $T commit -m "make regression runs int8's residual point at every N and tile size (regression.py --action residual --tflite --mid-reset: the layer cases, the residual-block TFLite models, the reset with hold_mem full), passing on 'RESIDUAL: PASS'; make residual runs it alone. Self-test test_residual_points (failed before: no residual point)."
```

### Task 2.11: Gate G4, the reports and the docs

**Files:**
- Create (local): `$J/cmds/res_cmp_points.py` (adapted from `int8_cmp_reg.py`), `$J/runs/rs2_g4/g4_residual_gate.log`
- Create: `.claude/skills/sienna-report/history/2026-10-DD_g0b_add_oracle_int8.txt`, `..._g1_tfliteadd_int8.txt`, `..._g2_sigmoid_ladder_int8.txt`, `..._sienna_int8_residual_g4.txt` (DD: each report's date)
- Modify: `.claude/skills/sienna-int8-residual/SKILL.md` (status, as built), `.claude/skills/sienna-int8/SKILL.md` (status line, the sigmoid open item closed), `.claude/skills/sienna-rtl/SKILL.md` and `references/file-map.md` (`add_lanes.sv`, `hold_mem`, the hold/add sets), `.claude/skills/sienna-tooling/SKILL.md` (`--action residual`, `add-vectors`, the `A` line), `.claude/skills/sienna-credits/SKILL.md` (one line: the hold set lives inside the activation stage, no new link), `README.md`, `$A/README.md`, `GPNAE/README.md`

**Interfaces:**
- Consumes: every earlier task; the Task 2.0 reference runs.
- Produces: G4's verdict and report; docs that say what is built.

- [ ] **Step 1: The sweep** (N = 8, 16, 32 at every tile size, every format; both mesh modes; random power-up)

```bash
for f in fp32 bf16 int8; do
  L rs2_g4_${f}_n8 64 24 $J/cmds/res_g4_pts.sh $f 8
  L rs2_g4_${f}_n16 64 24 $J/cmds/res_g4_pts.sh $f 16
  L rs2_g4_${f}_n32 64 48 $J/cmds/int8_O0.sh $J/cmds/res_g4_pts.sh $f 32
done
L rs2_g4_ck0_int8_16 64 24 $J/cmds/sienna_ck0.sh 16 4 int8
L rs2_g4_ck0_int8_8 64 24 $J/cmds/sienna_ck0.sh 8 2 int8
L rs2_g4_ck0_int8_32 64 48 $J/cmds/int8_O0.sh $J/cmds/sienna_ck0.sh 32 8 int8
```

and a random power-up run of the residual tests and the layer cases (a script, since the flags are multi-word): `$J/cmds/res_ri.sh`:

```bash
#!/bin/bash
# int8 residual tests and layer cases at random power-up; run from a snapshot root.
export EXTRA_FLAGS="-DNO_ZERO_INIT --x-initial unique --x-assign unique" SIM_ARGS="+verilator+rand+reset+2"
P=/proj/work/spramanik/sienna_jobs/venv/bin/python
make pipeline FMT=int8 N=16 TILE=4 TEST=int8_residual PYTHON=$P; a=$?
make residual FMT=int8 N=16 TILE=4 PYTHON=$P; b=$?
echo "res_ri: pipeline exit=$a residual exit=$b"; [ $a -eq 0 ] && [ $b -eq 0 ]
```

`chmod +x $J/cmds/res_ri.sh && L rs2_g4_ri 64 8 $J/cmds/res_ri.sh`, and the TFLite logistic agreement after the ladder: `L rs2_g4_gptfl 32 2 $J/cmds/res_py.sh . regression.py --action gpnae-tflite --report gpnae_tflite.log`.

Expected: every table PASS; the collapse-k 0 runs PASS (the residual tests included); `res_ri: pipeline exit=0 residual exit=0`.

- [ ] **Step 2: Compare with the references** — write `$J/cmds/res_cmp_points.py` from `int8_cmp_reg.py`, its glob pointed at where the merged pipeline points keep their per-test logs (read one `rs2_base_int8_n16` run to find them), and run it for each (format, N): `python3 $J/cmds/res_cmp_points.py rs2_base_<f>_n<n> rs2_g4_<f>_n<n>`. Required: fp32 and bf16 identical in results, cycles and words (spec); int8: every pre-existing test identical in cycles; words identical except tests with sigmoid sets, whose differing words are counted per test (spec: "that count is reported"); the eight residual tests new. Any other difference is a stop.

- [ ] **Step 3: The G4 report** — `$J/runs/rs2_g4/g4_residual_gate.log`: the run names and trees; the tables' verdicts per (format, N); the comparison results with the sigmoid word counts; the residual tests' `[RESIDUAL]` cross counts per point (which crosses were reached, which not); the layer cases and TFLite lines (including which ADD input is the residual); the mutant verdicts of Task 2.7; the TFLite logistic agreement before (`rs2_base_gptfl`) and after; the area and latency of the spec's estimate table unchanged and labelled estimates (no synthesis ran); `GATE G4 (int8 residual): PASS`. Copy it and the G0b, G1 and G2 reports (Tasks 1.2, 1.7, 1.13) into `.claude/skills/sienna-report/history/` verbatim, with the date prefix.

- [ ] **Step 4: The docs** (README style: what it is and why it is fast, no file tours; skills: what is true now)
  - `sienna-int8-residual/SKILL.md`: a status line at the top ("Implemented and verified <date> on branch `int8_residual` in all four repos, at N = 8-32; N = 64 in the final sweep") and an "As built" section listing every departure from the spec with its reason: `qadd_lat()` (P-1), G0b in ArithmeticLibrary (P-2), the lanes' z_in = zp_o (P-3), the fit scoring on the 2a lane (P-4), two rescales per threshold (P-5), the device-protocol half of the host interface (P-6), the test split (P-7), the dump checker (P-8), perf skipping residual tests (P-9), the measured sigmoid change counts, the cross counts reached, the depth numbers.
  - `sienna-int8/SKILL.md`: status line (2b's pre-synthesis RTL part done: the residual ADD and the sigmoid ladder); the G2 line's "sigmoid 3370/65536 outside" becomes history with the new figure (0 of 65,536, a hard gate).
  - `sienna-rtl`: `add_lanes.sv` in the file map; the hold and add sets in the activation stage; the trap that a hold set needs an L3 grant (and so a free activation bank) though it fills none.
  - `sienna-tooling`: `--action residual`, `--action add-vectors`, `make residual`, `make add-lanes`, the `A` line in the device protocol.
  - `sienna-credits`: one line under "Rules the RTL relies on": `hold_mem` is internal to the activation stage, set order (`a_rq_one_set`) fixes its occupancy, no new link.
  - `README.md` (SIENNA): one sentence in the int8 part: residual layers add TFLite's int8 ADD bit-exactly, at full rate, in the activation stage. `$A/README.md`: the `tfliteAdd` unit in the unit list. `GPNAE/README.md`: int8 sigmoid is exact past |x| = 2.5.

- [ ] **Step 5: Commit** (innermost first: ArithmeticLibrary README, then GPNAE README, pointer bumps in GPNAE and SystolicMesh for the README commits, then SIENNA)

```bash
git -C $T/$A add README.md && git -C $T/$A commit -m "README: tfliteAdd, TFLite's int8 ADD, bit-exact in both roundings."
git -C $T/GPNAE/ArithmeticLibrary fetch $T/$A int8_residual && git -C $T/GPNAE/ArithmeticLibrary merge --ff-only FETCH_HEAD
git -C $T/GPNAE add README.md ArithmeticLibrary && git -C $T/GPNAE commit -m "README: the int8 sigmoid is an exact ladder past |x| = 2.5 (0 of 65536 inputs outside GPNAE's tolerance); ArithmeticLibrary README bump."
git -C $T/SystolicMesh add ArithmeticLibrary && git -C $T/SystolicMesh commit -m "Bump ArithmeticLibrary: README names tfliteAdd."
git -C $T add GPNAE SystolicMesh README.md .claude/skills
git -C $T commit -m "Skills, README and report history for the int8 residual ADD and the sigmoid ladder (sienna-int8-residual as built; sienna-int8, sienna-rtl, sienna-tooling, sienna-credits; the G0b, G1, G2 and G4 reports); submodule bumps for their README commits."
```

### Task 2.12: Check-in, then the one N = 64 sweep

**Files:**
- No source changes (unless the N = 64 sweep finds a defect, which then gets its own fix commit, test first, and is reported)

**Interfaces:**
- Produces: `int8_residual` pushed in all four repos (with Soham's go-ahead) and merged into main; the N = 64 sweep's report.

- [ ] **Step 1: Ask Soham to approve the push** (visible to others), with the four gate reports. Before pushing, in each repo: `git log --format=%B origin/main..int8_residual | command grep -ci co-authored-by` → 0.

- [ ] **Step 2: Push, innermost first, checking each**

```bash
git -C $T/$A push git@github.com:SoHam-56/ArithmeticLibrary.git int8_residual
git -C $T/GPNAE push origin int8_residual && git -C $T/SystolicMesh push origin int8_residual
git -C $T push origin int8_residual
for d in $A GPNAE SystolicMesh .; do git -C $T/$d ls-remote --heads origin int8_residual; done
```

Expected: four heads equal to the local ones. Merging into main and deleting the branches follow Soham's word (memory: branches only while developing; branch deletion is his to approve).

- [ ] **Step 3: The N = 64 sweep** (all formats, both mesh modes; after check-in only)

```bash
for f in fp32 bf16 int8; do L rs2_n64_$f 64 72 $J/cmds/int8_O0.sh $J/cmds/res_g4_pts.sh $f 64; done
L rs2_n64_ck0_int8 64 72 $J/cmds/int8_O0.sh $J/cmds/sienna_ck0.sh 64 16 int8
```

Before giving an ETA, ssh to each node and check whether it compiles or simulates. Expected: every table PASS (the residual points at N = 64 use 128 lanes); compare with the last N = 64 references as in Task 2.11 Step 2; write `.claude/skills/sienna-report/history/<date>_sienna_int8_residual_n64.txt` and commit it (`git commit -m "Report history: the N = 64 sweep of the int8 residual work (<runs>): <verdict>."`), then push it the same way.

---

## Self-review against the spec (done while writing; recorded for the reviewer)

| Spec item | Task |
|---|---|
| Goal 1, D1 ADD semantics, G0b (every zero point, three activations, scale ranges, 65,536 pairs, rounding pin, optimized resolver reported, outputs kept as G1 vectors) | 1.1, 1.2, 1.7 |
| D2 separate elementwise op; D3 option (b) `add_lanes` after `requant_lanes` | 2.4, 2.6 |
| D4 hold/add sets, identity requantize, stream order kept | 2.8 (`sienna_layer`), 2.6 (`sienna_top`) |
| D5 only add beats take the ADD path; grant after drain; `rq_room` / `rq_drain` widened | 2.6, 2.7 (`a_fill_one_source`, `m_room`) |
| D6 ADD constants in the add set's sideband, per id; zp_1 explicit | 2.3, 2.8, 2.7 (`m_zp1`) |
| D7 sigmoid ladder (thresholds, both signs, exactness, start at 2.5, G_POST, no state or cycle) | 1.9, 1.10, 1.12 |
| `tfliteAdd` ports, stages, `add_lat()` estimate and the depth probe | 1.3, 1.6, 1.7 |
| `set_side` +162 bits; `sienna_layer` changes; `add_lanes`; `hold_mem`; hold set's L6 release, `gp_sets`, `p_null`; drop to `add_zpo`; no new credit link | 2.3, 2.4, 2.6, 2.8 |
| The seven named assertions, each fired by a mutant | 2.3, 2.7 |
| Host interface (A line, residual tile codes, clamps, input order, shortcut as its own layer, `ipu.add_params`) | 2.8, 2.9 (P-6) |
| Bit-exact models (`ipu`, `regression.py`'s golden, `gpnae_model`, `fit_poly_coeffs`) | 1.1, 2.5, 1.9, 1.10 |
| G1 (exhaustive corners, 10^6 random, G0b replay, both roundings, four mutants) | 1.6, 1.7 |
| G2 (bit-exact on gated, threshold and 64-scale cases; dense and gated hard gate; tanh/SELU/float/gpnae.sv identical; three mutants) | 1.11, 1.12, 1.13 |
| G3 by diff | 1.8, 2.2 |
| G4 tests (relu, linear, tanh, deep cached/uncached, tiles, train, then_relu_nopool, two_layers) at N = 8, 16, 32, both mesh modes, random power-up; TFLite residual blocks; existing int8 cycles identical and sigmoid word count; fp32/bf16 identical; TB crosses; the four mutants | 2.5 to 2.11 |
| One N = 64 sweep after check-in | 2.12 |
| Risks: semantics unverified (G0b first), model not in repo (converter check), coincidence class (cross counters, assertions), `a_rq_one_set` margin (`a_hold_push_empty` / `a_add_pop_full` would fire), sigmoid outputs move (counted, goldens from the model), quota (scratch, compressed) | 1.2, 2.9, 2.5, 2.7, 1.13 / 2.11, 1.0 |
