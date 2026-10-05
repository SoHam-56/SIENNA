# SIENNA two-script tooling: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SIENNA's Python tooling becomes exactly two scripts — `regression.py` (hardware sanity) and `model_runner.py`
(the host software stack) — with every flow producing the same results as before and the old scripts deleted.

**Architecture:** a behaviour-preserving move. Numerics and the device-build package writer move into model_runner
(the host needs them); regression.py imports them. Each other script's code moves verbatim into one of the two, behind
a new `--action`, and the Makefile recipe that called it is repointed in the same task, so `make` never breaks between
tasks. Old files are deleted in the task that absorbs them.

**Tech Stack:** Python 3 + numpy (pure-Python checks on the login node), Verilator on the slurm farm.

**Spec:** `.claude/skills/sienna-tooling/SKILL.md` (same directory). Read it first.

## Global Constraints

- Branch `tooling` (SIENNA only). SystolicMesh, GPNAE and ArithmeticLibrary are not touched; their scripts stay.
- After the move, the only Python files tracked in SIENNA outside the submodules are `regression.py` and
  `model_runner.py` (check: `git ls-files '*.py' | command grep -v '^GPNAE/\|^SystolicMesh/'`).
- Dependency direction: regression.py may import model_runner; model_runner never imports regression.
- Behaviour-preserving: moved code is moved verbatim (rename only where a task says so); every flow's results must be
  identical to the final packing gate (`pkf_*` runs) — regression words and cycles via `$J/cmds/int8_cmp_reg.py`,
  mesh-free flows by comparing their result lines (wall times stripped). A difference is a bug in the move.
- The generated `testbenches/test_config_pkg.sv` must be byte-identical to what the old code wrote for the same
  arguments (compare in a scratch dir before and after; the TB build depends on it).
- Every build/sim on the farm: `J=/proj/work/spramanik/sienna_jobs; $J/cmds/int8_tree.sh NAME 32 HOURS cmd words...`,
  prefix `tl_`, one word per argument, logged to `$J/runs/pk_launch.log`; any `Assertion failed` / `%Error` is a failure.
- `grep`, `diff`, `du` are aliased: use `command grep` etc. Commit messages via `git commit -F <file>`; never
  Co-Authored-By or generated-by lines; no pushes. One-line comments and docstrings. Reports are .log.
- Farm tooling (outside git) that the current gate and report use is updated to the new entry points in the task that
  moves its target (list in each task); historical one-off scripts under sienna_jobs are left as history.

## Review Focus

1. A flow that silently changes behaviour because a moved function picked up a different module global (COLLAPSE_K,
   FORMATS, ROOT, TB_DIR, SETS_IN_FLIGHT) — every global has one owner after Task 1.
2. `make pkg` / the pkg-check guard and `make verilator` rebuilding from a package written by the new code path.
3. The report builder (`sienna_jobs/report/build_report.py`) and gate scripts that import `perf_analysis` or
   `gemm_sweep` — they must keep working against the new entry points.
4. `--action all` exiting zero when a sub-step failed (it must propagate every failure).
5. A selftest that no longer runs anything after the move (count the tests before and after; same number).

---

### Task 1: Numerics and the build package move into model_runner

**Files:** Modify `model_runner.py`, `regression.py`; delete `tflite_ref.py` (its code moves into model_runner).

- [ ] Inventory first: `command grep -o "regression\.[A-Za-z_]*" model_runner.py | sort -u` and the same for
  `mr.regression.` / `reg.` users in the other scripts; record the list in the report.
- [ ] Move verbatim from regression.py into a `# ── Numerics ──` section of model_runner.py: `FORMATS`, `op_round`,
  `op_hex`, `fmt_bits`, `bits_float`, `activation_to_code`, `apply_activation`, `wrap32`, `imatmul`, `quant_act`,
  `quant_weights`, `fold_bias`, `requantize`, `requant_params`, `selu_saturates`, `int8_lane`, `activate_int8`,
  `drop_zp`, `int8_layer_exact`, `_check_rounding`, plus any helper they call; and all of tflite_ref.py's code (its
  `ROUNDING`, `quantize_multiplier`, `layer_multipliers`, `activation_range` and what they need). Keep names.
- [ ] Move into a `# ── Device build ──` section: `write_sv_package`, `_config_items`, `SETS_IN_FLIGHT`,
  `COLLAPSE_K` (the one owner; regression.py's `--collapse-k` sets `model_runner.COLLAPSE_K`), and a
  `write_build_pkg(N, T, lanes, fmt)` that writes exactly the package `LayerSim.build` / `Sim.build` get today
  (today they call `regression.generate_vectors` with the `matmul_relu_nopool` test). Byte-identical package, checked.
- [ ] regression.py: `from model_runner import (...)` every moved name (and `tflite_ref` users switch to model_runner);
  delete the moved definitions. model_runner: drop `import regression` and every `regression.` reference.
- [ ] Checks: the five test_*.py files pass unchanged (they import the old names through regression/model_runner);
  `python3 regression.py --action pkg ...` for one test per format writes a byte-identical package and stimulus to the
  pre-move code (run both in scratch copies); farm: `tl1_reg16_<fmt>` (`bash $J/cmds/mk.sh regression FMT=<fmt> N=16 TILE=4`, three formats) IDENTICAL to `pkf_reg16_<fmt>_T4`, and `tl1_pack_bf16` / `tl1_tfl16` identical to the pkf runs.
- [ ] Commit: "model_runner owns the numerics (format rounding, TFLite quantization, requantize, the int8 lane and layer golden, TFLite's reference kernels from tflite_ref.py) and the device build package; regression.py imports them; tflite_ref.py is folded in."

### Task 2: TFLite runs become `model_runner.py --action tflite`

**Files:** Modify `model_runner.py`, `Makefile` (`tflite` target); delete `tflite_int8_run.py`, `tflite_pack_run.py`.

- [ ] Add `--action` to model_runner's CLI: `model` (default, today's CLI unchanged) and `tflite` (today's
  tflite_int8_run main; with `--pack`, tflite_pack_run's main), their code moved verbatim into a `# ── TFLite runs ──`
  section. The log file names they write stay the same.
- [ ] Rename the backends as the spec says: `LayerSim` -> `RtlLayer`, `Sim` -> `RtlSets`, `EmuSim` -> `Emulator`, every
  user updated (regression's moved code arrives in Task 3 already using the new names).
- [ ] Makefile `tflite`: `$(PYTHON) model_runner.py --action tflite ...` (same arguments); help text updated.
- [ ] Farm tooling: `$J/cmds/int8_tflite.sh` and any pkf gate script that runs tflite_int8_run.py / tflite_pack_run.py.
- [ ] Checks: `tl2_tfl16` (make tflite int8 N=16) and `tl2_tfp32` (`--action tflite --pack --n 32 --tile-size 4`)
  identical to `pkf_tfl16` / `pkf_tfp_N32_T4`; `tl2_model_bf16` and `tl2_model_sets_fp32` identical to
  `pkf_model_bf16` / `pkf_model_sets_fp32` (find the exact commands in `$J/runs/pk_launch.log`).
- [ ] Commit: "model_runner.py --action tflite runs the single-layer int8 TFLite models and, with --pack, the packed TFLite layers (was tflite_int8_run.py and tflite_pack_run.py); the backends are RtlLayer, RtlSets and Emulator."

### Task 3: Hardware checks become regression.py actions

**Files:** Modify `regression.py`, `Makefile` (`gemm`, `perf-analysis`, `pack`); delete `pack_regression.py`,
`gemm_sweep.py`, `perf_analysis.py`, `tflite_oracle.py`, `tflite_pack_models.py`, `gpnae_int8_tflite.py`,
`testbenches/gen_rq_lanes.py`.

- [ ] Move each script's code verbatim into its own `# ── <name> ──` section of regression.py behind `--action`:
  `pack` (pack_regression), `gemm` (gemm_sweep; it gains `--tile`, passed to `exact_layer` and the build), `perf`
  (perf_analysis), `oracle` (tflite_oracle), `pack-models` (tflite_pack_models), `gpnae-tflite` (gpnae_int8_tflite),
  `rq-vectors` (gen_rq_lanes). Each action keeps its old options and output files. TensorFlow-only imports stay local
  to the actions that need them (regression.py must still import without TensorFlow).
- [ ] Makefile: `gemm` -> `regression.py --action gemm` (drop the "ignores TILE" warning; pass `--tile $(TILE)`),
  `perf-analysis` -> `--action perf`, `pack` -> `--action pack`; help text updated.
- [ ] Farm tooling: `report/build_report.py` (imports perf_analysis), `cmds/int8_rq_lanes.sh` (gen_rq_lanes),
  `cmds/int8_cmp_perf.py`, `cmds/int8_cmp_gemm.py`, and the pkf gate scripts that call these; point them at the new
  entry points / `import regression`.
- [ ] Checks: farm `tl3_pack_<fmt>` (make pack N=16 T=4, three formats) identical to `pkf_pr_N16_T4_<fmt>`;
  `tl3_gemm_int8` (make gemm int8 QUICK=1) identical to `pkf_gemm_int8`; `tl3_perf` identical to a pkf perf run;
  `tl3_rq` (int8_rq_lanes.sh) PASSED; the report builder rebuilds the report HTML identically (diff against the
  current `sienna_report/sienna_report.html` except its date line).
- [ ] Commit: "regression.py gains the hardware-check actions pack, gemm, perf, oracle, pack-models, gpnae-tflite and rq-vectors (was pack_regression.py, gemm_sweep.py, perf_analysis.py, tflite_oracle.py, tflite_pack_models.py, gpnae_int8_tflite.py and testbenches/gen_rq_lanes.py); make gemm now takes TILE."

### Task 4: Self-tests and the one-verdict gate

**Files:** Modify `regression.py`, `Makefile` (new `check`); delete the five `test_*.py`.

- [ ] Count the self-tests first (`python3 test_X.py` for each; record pass counts). Move every `test_*` function
  verbatim into a `# ── Self-tests ──` section; `--action selftest` runs them all with the same reporting
  (PASS/FAIL per test, non-zero exit on any failure). Same number of tests, all pass.
- [ ] `--action all [--format] [--n] [--tile] [--lanes]`: selftest, then via make the SystolicMesh regression
  (`sm-verilator`), GPNAE (`gpnae-verilator`), the SIENNA regression, pack, gemm quick and (int8) tflite; a table of
  step, verdict and log; exit non-zero if any step failed (Review Focus 4: test it with a deliberately failing step in
  a scratch copy).
- [ ] Makefile `check` -> `regression.py --action all`, with help text.
- [ ] Checks: `python3 regression.py --action selftest` ALL PASS with the recorded count; farm `tl4_check_int8`
  (`make check FMT=int8 N=16 TILE=4`) passes every step.
- [ ] Commit: "regression.py --action selftest runs the tool self-tests (was test_makefile_fmt, test_pack_jobs, test_perf_analysis, test_regression_parse, test_tflite_ref) and --action all gives one verdict; make check runs it."

### Task 5: Docs and the final gate

**Files:** `.claude/skills/sienna-tooling/SKILL.md` (status, as built), `sienna-rtl` (commands, file map),
`sienna-packing` and `sienna-int8` current-usage lines that name deleted scripts (history sections keep old names),
`sienna-report` if it names perf_analysis.

- [ ] Run the full packing gate again on the final tree with prefix `tlf_` (same list as `pkf_`, via the updated farm
  scripts) and compare every run with its `pkf_` counterpart: IDENTICAL. Append a dated section to
  `sienna_report/packing_gate.log`.
- [ ] Global check: only `regression.py` and `model_runner.py` remain (command in Global Constraints).
- [ ] Commit docs: "skills: the two-script tooling as built (regression.py, model_runner.py) and the commands that replace the deleted scripts."
