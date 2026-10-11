---
name: sienna-tooling
description: Use when running, changing or extending SIENNA's Python tooling - the two scripts regression.py (hardware sanity: regressions, unit benches, packed layers, GEMM sweep, perf, tool self-tests, make regression's points and table) and model_runner.py (the host software stack: TFLite loading, lowering, tiling, packing, the device protocol to TB_model_run, backends, reference numerics). Also use when someone asks how a software developer should build on SIENNA, what the layer-file protocol is, or where a former script (pack_regression.py, gemm_sweep.py, perf_analysis.py, tflite_*.py, test_*.py) went.
---

# SIENNA tooling: two scripts

**Status: implemented and verified 2026-10-05 on branch `tooling` (SIENNA only; not pushed), gated tree 7b711bd
(b396aeb..7b711bd; the commit after it changes only this skill). Final gate `tlf_*` (`sienna_report/tooling_gate.log`):
all 106 packing-gate configurations identical to `pkf_*`. Follow-ups the same day (Soham: "go with option 2, and do the
layering now"): model_runner laid out in eight layered sections (2a88978) and `make check` passing in fp32, bf16 and
int8 at 972464c, bf16 accuracy reported, not gated (see Gate).** Design approved 2026-10-05 (Soham:
"keep submodule scripts, delete old ones, go ahead"). Soham asked for exactly two
Python scripts in SIENNA: one for hardware sanity, one that runs models and stands as the reference host software a
software team builds its own layer from ("this model runner is supposed to be the compiler").

## Before the move (main b1cfacb)

17 tracked Python files at the SIENNA root and one under testbenches/, about 4,700 lines:

| Today | Lines | Role |
|---|---|---|
| regression.py | 1,427 | pipeline tests, golden models, stimulus and test_config_pkg.sv writer, numerics (format rounding, int8 quantization, requantize) |
| model_runner.py | 787 | TFLite loading, lowering (im2col, fused add, global average pool), tiling, pack_jobs / unpack, simulator drivers (LayerSim, Sim, EmuSim), CLI |
| perf_analysis.py | 412 | cycle / throughput analysis of streamed sets |
| tflite_oracle.py | 283 | gate G0: TFLite interpreter vs reference kernels, pins rounding, saves models |
| gemm_sweep.py | 213 | GEMM shape sweep on the layer engine |
| test_*.py (5 files) | 505 | tool self-tests (pack_jobs, parsers, Makefile FMT flow, perf analysis, tflite_ref) |
| tflite_int8_run.py | 161 | single-layer int8 TFLite models on the RTL, bit-exact |
| pack_regression.py | 153 | packed layers vs each job alone |
| tflite_ref.py | 88 | TFLite reference integer kernels (QuantizeMultiplier, activation range) |
| tflite_pack_run.py, tflite_pack_models.py, gpnae_int8_tflite.py | 134 | packed TFLite runs, packed test models, GPNAE int8 vs TFLite |
| testbenches/gen_rq_lanes.py | ~70 | vectors for TB_requant_lanes |

The submodules keep their own scripts (SystolicMesh: regression.py, matmul_tests.py, conv_tests.py, mesh_model.py,
stim_format.py; GPNAE: regression.py, gpnae_model.py, ...). They are separate repos with their own regressions; this
design does not move them. Both new scripts call them through make (`sm-verilator`, `gpnae-verilator`) or import
their models (mesh_model, gpnae_model, fpu, ipu), as today.

## The two scripts

### model_runner.py — the host software stack (what a software developer reads and rebuilds)

Keeps today's name; it grows from today's model_runner.py.

Organised top to bottom as the layers a driver/compiler has, each a labelled section with a short public API:

1. **Numerics** (reference arithmetic the host must reproduce): format rounding and bit patterns (fp32, bf16, int8),
   TFLite post-training quantization (`quant_act`, `quant_weights`, `fold_bias`), `requant_params`, `requantize`,
   the int8 lane and layer golden (`int8_layer_exact`), and the TFLite reference kernels now in tflite_ref.py.
   Moved here from regression.py and tflite_ref.py because the compiler needs them, and regression.py imports them.
2. **Frontend**: TFLite loading (`load_tflite`, `load_layer`), lowering each op to GEMM jobs (`lower_op`, `im2col`,
   `fuse_add`, global average pool), the int8 job builder (`job_of`).
3. **Middle end**: tiling into N x N sets (`format_layer`, `layer_epilogue`), packing (`pack_jobs`, `unpack`,
   `pack_precheck`), with the rules stated where they live: block width, at most 8 settings, N | lanes, a set runs at
   its slowest activation.
4. **Device protocol**: the one boundary between host and hardware — the layer file TB_model_run reads (`L`, `Q`,
   `P`, `E` lines, rows of N hex words, per-block epilogue) and the result file it writes. Written as one documented
   encoder / decoder pair, so a driver for real silicon replaces only what is below it.
5. **Backends**: `RtlLayer` (TB_model_run: build once, run a layer per invocation; was LayerSim), `RtlSets`
   (TB_sienna_model, the host-driven set engine; was Sim), `Emulator` (numpy stand-in; was EmuSim). Same interface:
   `build()`, `run_job(job, tag) -> (outputs, sets, cycles)`.
6. **CLI**:
   - `model_runner.py --action model` (default) `--model-dir D [--models ...] [--format fp32|bf16|int8] [--n] [--tile] [--lanes] [--engine layer|sets|emulate] [--count] [--no-sim]` (today's CLI)
   - `model_runner.py --action tflite --models D [--pack] [--n] [--tile] [--lanes]` (was tflite_int8_run.py and, with `--pack`, tflite_pack_run.py)

### regression.py — hardware sanity (one place to prove the RTL is right)

Named as SystolicMesh's and GPNAE's own `regression.py` (Soham, 2026-10-05), so every repo has its hardware check under
the same name. It keeps today's `--action` interface and grows it; it imports model_runner for numerics and backends,
never the other way round.

- `--action regression` (default) `[--format] [--n] [--tile] [--lanes] [--test SUBSTR] [--collapse-k]` — the pipeline regression, as today
- `--action pkg --test NAME` — test_config_pkg.sv and one test's stimulus (`make pkg` and the guard use it), as today; `gen`, `analyze` as today
- `--action pack [--act] [--rows ...]` — packed layers vs each job alone (was pack_regression.py)
- `--action gemm [--quick] [--engine]` — GEMM sweep (was gemm_sweep.py)
- `--action perf ...` — streamed-set performance analysis (was perf_analysis.py)
- `--action oracle --out D --report F` — gate G0 (was tflite_oracle.py); `--action pack-models D` (was tflite_pack_models.py); `--action gpnae-tflite` (was gpnae_int8_tflite.py)
- `--action rq-vectors OUT` — TB_requant_lanes vectors (was testbenches/gen_rq_lanes.py)
- `--action selftest` — every tool self-test now in test_*.py, pure Python, no simulator
- `--action points [--small] [--formats f,...]`, `--action point --name P`, `--action summary` — make regression (below); they replaced `--action all`, the single-point gate, on 2026-10-10

### Makefile

Every target keeps its name and arguments and calls the new script: `regression`, `pkg`, `pack`, `gemm`,
`perf-analysis`, `model`, `tflite` (and a new `check` for `regression.py --action all`; since 2026-10-05 the gate is `make regression` and the pipeline suite `make pipeline`). test_makefile_fmt's checks move into
`selftest` and follow the new recipes.

## make regression (2026-10-10)

"Regression means running all the tests we have in a sweep and seeing whether the design is still healthy: N = 8 through
64 and every valid tile size for each N" (Soham, 2026-10-10). Until 2026-10-10 `make regression` was a one-verdict gate
at N = 16, T = 4 (`--action all`); records dated before that (skills, reports) mean that gate.

Each repo owns its regression (Soham, 2026-10-10): SIENNA's runs SIENNA's tests only. SystolicMesh (`make regression
[SMALL=1] [FMT=]`: model-tests, aril-fpu, aril-narrow once, the mesh at every N, tile size and format, collapse-k 1; the
one-N run is `make mesh-regression MATRIX_SIZE=<n>`, which SIENNA's `make sm-verilator` calls), GPNAE (`make regression
[FMT=]`: its two model checks once, the lane bit-exact against gpnae_model in each format, and its accuracy against
the functions: fp32 gated at 1%, bf16 reported, not gated) and ArithmeticLibrary (`make regression`: every
unit bench and model check once) have their own, on the same engine.

- **Engine.** `SystolicMesh/ArithmeticLibrary/Common/regress.py` (loaded by path as `aril_regress`, since every repo has
  a `regression.py`): a repo's `Spec` gives the points and each point's command, environment, verdict and reason not to
  run; the engine runs a point in a private copy of the tree (`copy_tree`: git's tracked and untracked, not ignored,
  files of the repo and its submodules, so ignored stale stimulus and old builds never reach a run), with TRACE=0,
  OPT_FAST=-O0 from N = 32 (-Os compiles a large mesh for hours), SIM_THREADS = cores / N under make -jN (1 under an
  unlimited -j; every Verilator Makefile passes it to `--threads`: Verilator's threads spin, and four 8-thread
  simulations on 8 cores took a 30 s GPNAE run past 55 min) and none of the caller's make variables, writes
  `<REG_DIR>/<point>.log` and `.json` (rc, seconds, run, host), keeps a failing point's `testbenches/results` in
  `<point>.artifacts/`, and writes the table. Its own self-tests are ArithmeticLibrary's `regression.py --action selftest`.
- **SIENNA's points** (`SiennaSpec` in regression.py): `selftest` once, and per N in {8, 16, 32, 64} (SMALL=1: {8, 16})
  x every power-of-2 tile size x format: `pipeline` (`make pipeline`), `model` in fp32 and bf16 (`make model
  MODEL_COUNT=3`, the four MLPerf Tiny models), `pack`, `gemm` (QUICK=1) and `tflite` in int8. LANES is 32 up to
  N = 32 and 128 at N = 64. Names are `selftest` and `fmt/N<n>_T<t>/kind`: 85 points with SMALL=1, 217 in full;
  SMALL= and FMT= narrow it only on make's command line (`$(origin)`), never from the environment; MODEL_DIR reaches the
  points absolute (`$(abspath)`, and the engine's `--model-dir`), since a point runs in a private copy.
- **Targets.** Each point is `make regression/<point>`, so `make -jN` runs N side by side (N for the memory, never a bare -j); `make regression` runs
  them all with `--keep-going` (a failing point does not stop the others), then the summary for its own run.
  `make regression-summary` prints the table of what is in REG_DIR (default `testbenches/results/regression`) without
  running anything. REG_RUN names the run: `make regression` sets one, each result records it, and the summary turns a
  result of another run into NOT RUN (stale). Without REG_RUN the summary takes any run.
- **Verdicts**, never "finished cleanly": pipeline `Passed : X / X` with X = every pipeline test of the format (32 fp32,
  33 bf16, 41 int8) and `Sanity Clean`; model and tflite `MODEL GATE: PASS`; pack `PACK REGRESSION PASS: 0 mismatching
  outputs`; gemm its `grid_*` rows, each check's own line (fp32 the three `layer_<act>_bias` errors, bf16 `layer_<act>: 0
  outputs differ` and layer_bias_special's 0, int8 `GEMM int8: N cases, 0 with outputs that differ`) and no `FAIL` line;
  every point exit 0 and no `Assertion failed` / `%Error` line. Model points without a usable MODEL_DIR (unset, missing,
  or without the four .tflite files) are NOT RUN with how to set it. The summary exits 1 on any gated FAIL or NOT RUN,
  on no point selected, or when no point ran.
- Self-tests: `test_regression_points`, `test_point_commands`, `test_point_verdicts`, `test_model_points_need_model_dir`,
  `test_regression_target` (make -n).
- **Run times** (measured 2026-10-10, 8-core machines, one build per group): SIENNA SMALL=1 (85 points) 3.4 h one point
  at a time, 20 min with every point in parallel, model points the largest share (108 of 203 min); a fresh clone's
  `make -j4 regression SMALL=1 FMT=int8` on one machine 12 min. N = 32 T = 8: pipeline fp32 29 min, bf16 19, int8 7;
  model fp32 11, bf16 8. N = 64 T = 16: pipeline fp32 3 h 57 min, bf16 1 h 59 min, int8 55 min; model fp32 48 min,
  bf16 29; pack fp32 21 min; gemm fp32 10 min. The full sweep was not run after these changes: about 2.7 days one
  point at a time, estimated from one tile size per N. SystolicMesh SMALL=1 22 min one point at a time (fp32 N = 64
  T = 16 mesh 57 min), ArithmeticLibrary 10 min, GPNAE 2 min.

### One build per hardware group (2026-10-10)

The pipeline regression builds TB_sienna_top once per group of tests whose packages agree on everything but the
TB-only knobs (`TB_KNOBS` in model_runner.py and the Makefile: ACTIVATION_CODE, NUM_SETS, HAS_BIAS, WEIGHT_CACHE,
ACCUM_PASSES, MIXED_LEN, MIXED_ACTS, PACKED, TRAINING_MODE, DROPOUT_SEED), and runs the other tests of the group on
that build with `make sim`. Measured: a build is 70 s at N = 16 against 1-2 s of simulation, and about 8.6 min at
N = 64 fp32.
- Every generator writes `test_config_pkg.sv` with every item (TB_sienna_multi still reads the knobs from it) and, beside
  it, `test_knobs.mem` (`write_test_config`: lowercase name and value per line). TB_sienna_top reads the knobs from that
  file at time 0 (`read_knobs`: a missing, unknown or repeated setting is `$fatal`), and no file the build compiles or
  can include names one but the package's own declarations (`knob_references` over `build_sources`), so no knob value
  can reach the elaboration or the DUT. `copy_mem_files` copies every stimulus file (plain `cp`: `cp -u` could skip a
  test's files written in the same second as the last one's on a coarse-timestamp filesystem).
- `run_regression` orders the tests by group (`build_order`, from each test's configuration) and builds when the
  package the test just wrote differs, outside the knobs, from the build on disk (`build_key`); a failed build is never
  reused. `make sim` runs the binary only when `BUILD_KEY` (every Verilator input but the package by checksum, the
  package without the knobs, and the Verilator flags) equals the one `make verilator` stored after a whole build, so a
  stale or mismatched binary stops the run instead of answering for another configuration.
- Self-tests: `test_tb_knobs_are_run_time` (TB_sienna_top names no knob; read_knobs covers each; the Makefile's list is
  model_runner's; the knobs file), `test_build_groups` (knob-only differences share a build, any other package item
  does not; every test once, each group contiguous), `test_sim_reuses_only_its_own_build` (the key ignores knob values
  and changes with any other item, an input file or a flag).

## Rules

- Behaviour-preserving: every flow must produce the same files and results as before the move (regression words and
  cycles, model outputs, GEMM rows, pack and TFLite verdicts), proven on the farm against the pkf_ gate runs.
- The old scripts are deleted once the new ones are proven (Soham approved; git keeps their history); nothing else
  changes in RTL or testbenches. The submodules keep their own scripts, untouched.
- One-line comments and docstrings; sections marked with one-line headers.
- Reports stay .log; farm tooling stays in sienna_jobs (outside git).

## Size and trade-off (Soham chose two files)

regression.py will be about 3,000 lines and model_runner.py about 1,500. To keep them readable: a contents block at the
top of each, sections in the order above, no cross-section reach-ins (sections call each other through their public
functions), and model_runner's sections ordered so a reader can stop at the device protocol and know everything a
driver needs.

## Out of scope (possible next steps)

- Automatic packing in `model`: a scheduler that finds small layers, groups them by activation class and calls
  `pack_jobs`. The packer exists; the policy does not.
- A real-silicon backend (a driver speaking the same per-set interface as the device protocol).
- Moving the submodules' scripts.

## As built (2026-10-05)

Commits on `tooling` (SIENNA; the submodules are untouched: SystolicMesh b1a2c31, GPNAE 0d407a7, AriL cb46ced):
d49a829 (TB_sienna_layer -> TB_model_run), 3cc5675 (Task 1), 21fbc8f (Task 2), 3217ed7 and 3584a1f (Task 3), ff91381
(Task 4), fb68726 (comments) and the skill commits. The only tracked Python outside the submodules is `model_runner.py`
(1,349 lines) and `regression.py` (3,017 lines); the check is the global-constraints command in the implementation plan.
The plan ledger (`.superpowers/sdd/implementation-plan-sienna-tooling/progress.md`) has the full record.

### Command map (old -> new)

| Before | Now | make |
|---|---|---|
| `pack_regression.py [--n --tile --lanes --rows --format --act]` | `regression.py --action pack` (same options) | `make pack` |
| `gemm_sweep.py [--n --lanes --work --emulate --quick --engine --format --host-gaps]` | `regression.py --action gemm` (same, plus `--tile`, default 4) | `make gemm [TILE=]` |
| `perf_analysis.py [--sets --configs --n --tile-size --lanes --slices --merge --format ...]` | `regression.py --action perf` (same options) | `make perf-analysis` |
| `tflite_oracle.py --out D --report F [--seeds --inputs]` | `regression.py --action oracle` (same options) | |
| `tflite_pack_models.py D` | `regression.py --action pack-models D` | |
| `gpnae_int8_tflite.py --report F` | `regression.py --action gpnae-tflite --report F` | |
| `testbenches/gen_rq_lanes.py OUT` | `regression.py --action rq-vectors OUT` | |
| `python3 test_makefile_fmt.py` (and the other four `test_*.py`) | `regression.py --action selftest` (38 tests) | |
| | `regression.py --action all [--format --n --tile --lanes]` (until 2026-10-10) | `make regression` (was `make check`) |
| `regression.py --action all` | `--action points`, `point`, `summary`: every N, tile size and format (2026-10-10) | `make regression`, `make regression/<point>`, `make regression-summary` |
| `make regression` (the pipeline suite) | `regression.py --action regression` (unchanged) | `make pipeline` |
| `tflite_int8_run.py [--n --tile-size --lanes --models --work]` | `model_runner.py --action tflite` (same options) | `make tflite` |
| `tflite_pack_run.py [--n --tile-size --lanes --models]` | `model_runner.py --action tflite --pack` (same options) | |
| `model_runner.py --model-dir D ...` | unchanged (`--action model` is the default) | `make model` |
| `import tflite_ref`; `regression.FORMATS`, `op_round`, `quant_act`, `int8_layer_exact`, `write_sv_package`, ... | `model_runner.X` (regression.py imports them by name, so `regression.X` still resolves) | |
| `model_runner.LayerSim`, `Sim`, `EmuSim` | `RtlLayer`, `RtlSets`, `Emulator` (no aliases) | |
| `TB_sienna_layer` | `TB_model_run` | |

Log and result file names did not change (`pack_regression_*.log`, `gemm_sweep_N*.log`, `tflite_int8_N*_T*.log`,
`pipeline_performance_report.log`), so the farm comparison scripts read them as before.

### Departures from the design above, each with its reason

1. **model_runner has eight sections, not six** (2a88978, after the plan's tasks): a contents block, then 1 Numerics,
   2 Frontend, 3 Middle end, 4 Device protocol, 5 Device build, 6 Backends, 7 Runtime (`execute`, `macs_of`), 8 CLI.
   The build package and the graph executor fit none of the six layers. The device protocol is `write_layer` (moved
   out of `RtlLayer.run_job`) and `read_outputs` for TB_model_run, plus `write_sets` for TB_sienna_model.
   `RtlSets.run_job` replaces the free function `run_job_hw`, so all three backends are used as `build()`,
   `run_job(job, tag)`. Every other definition is AST-identical to before. The 9 layer files compared offline are
   byte-identical, and the `tlr_` farm runs (pack in three formats, TFLite single-layer and packed, gemm, model bf16,
   model sets fp32) are identical to `pkf_`. `test_layer_file_format` pins the file layout.
2. **`# ── Device build ──` in model_runner** (`write_sv_package`, `_config_items`, `SETS_IN_FLIGHT`, `COLLAPSE_K`, and a
   new `write_build_pkg(N, T, lanes, fmt)`) so every global has one owner; `--collapse-k` sets `model_runner.COLLAPSE_K`.
   `write_build_pkg` writes only `test_config_pkg.sv` (byte-identical, checked in scratch copies); the old build path
   also wrote stimulus `.mem` files that neither TB_model_run nor TB_sienna_model reads.
3. **model_runner appends the submodule directories to `sys.path`** (not insert): GPNAE and SystolicMesh each have a
   `regression.py`, and with insert `import regression` after `import model_runner` loaded SystolicMesh's.
4. **perf reads the RTL in `perf_init()`, not at import** (ruling, Task 3 review): verbatim, `import regression` parsed
   sienna_fmt_pkg.sv, gpnae_poly.sv and sienna_top.sv, so an RTL change perf cannot read broke every action. perf's
   results are identical; its errors start "regression.py --action perf:".
5. **Self-tests are an explicit registry** (`@selftest`, `@perf_selftest` runs `perf_init()` first), not collected by
   name: the oracle's helper `test_inputs` is a global `test_*` too. Clashing fixture names were renamed (ENV ->
   MAKE_ENV, write_pkg -> fixture_pkg, ...); 33 moved tests plus one new `test_check_target` = 34; output follows
   source order.
6. **Option spellings were kept per script**, so `--tile` (pack, gemm, all) and `--tile-size` (regression, perf,
   model, tflite) both exist; the Makefile passes the right one. The spec's `--tile` for `--action tflite` is
   `--tile-size`.
7. **`make gemm` takes TILE** (the "ignores TILE" warning is gone): `--tile` goes to RtlLayer, both `exact_layer` calls,
   and RtlSets / Emulator; default 4 keeps the old results.
8. **Renames inside regression.py to avoid clashes in one namespace:** the pack-models list is `PACK_MODELS`, the
   rq-vectors constants `RQ_N`, `RQ_LANES`, `RQ_SETS`, `RQ_PER`; the oracle's TensorFlow-built constants are made in
   `_tf_imports()` after argument parsing, so regression.py imports without TensorFlow.
9. **`--action all`'s verdict per step** is the exit status plus a scan for `Assertion failed` / `%Error` in the step's
   log; logs go to `testbenches/results/check/`. It follows GPNAE's own contract: the int8 sigmoid accuracy shortfall
   is printed "reported, not gated", so `make check` passes with it (an open int8 item, not a tooling one). In bf16
   (f10f49d, fixed in 972464c) GPNAE is two steps: `gpnae-verilator` with `GPNAE_MODEL=hw` (bit-exact against the
   lane model, gated) and `gpnae-accuracy` (against the functions, in `REPORTED`: shown, not counted). fp32 keeps
   one gated accuracy step because gpnae_model's fp32 negative sigmoid does not match the RTL (`--model hw` fails 9 of
   10 patterns, sigmoid only, as `g11_hw32` did on 2026-09-28; a GPNAE item). The steps run without make check's own
   variables (`STEP_ENV_DROP`), so `make check TEST=x` cannot narrow a gated step.
10. **regression.py keeps an unused `get_polynomial_terms` import** for the untracked `run_real_model.py`.

### Known gaps (deferred minors, from the ledger)

- `--action selftest` does not parse its arguments (`--action selftest --bogus` runs the tests).
- (`--action all`, now gone, did not flush per line.)
- perf helpers called without `perf_init()` meet None globals (only `perf_main` and the perf self-tests call it).
- model_runner's `--action` / `--pack` arguments are declared in the sub-parsers for help only.
- `pack_precheck` (Middle end) reads `COLLAPSE_K`, owned by Device build: a reach-in across sections.
- No `make -n` self-test for the gemm recipe's `--tile`.

### Gate

`make check` below is today's `make regression` (renamed 2026-10-05, Soham); the pipeline suite it runs is `make pipeline`.

Each task compared its flows on the farm with the final packing gate (`pkf_*`): regression words and cycles
(`int8_cmp_reg.py`), pack, gemm, TFLite and model result lines, the perf report, rq-vectors, the oracle and pack-models
files; all identical (task reports in the ledger directory). `tl4_check_int8` (`make check FMT=int8 N=16 TILE=4`) passed
every step. The final gate reran the packing gate's list with prefix `tlf_` on 7b711bd (snapshot from clean tracked trees) and
is recorded in `sienna_report/tooling_gate.log` (a new file beside `packing_gate.log`, which stays the packing record;
summary by `sienna_jobs/tlf_gate_summary.py`, launcher `sienna_jobs/tlf_msgs/`). 108 of 109 runs pass:
- All 106 `pkf_` configurations pass and are identical to their `pkf_` run: regression words and cycles at N = 8, 16
  (T = 2-16) and 32 and collapse-k 0, the mesh sweeps, the unit benches, `make pack` at every N / T and format and the
  8 `--act` sweeps, the TFLite single-layer and packed runs, gemm, the four lints, `make model`, `--engine sets` and
  TB_sienna_multi. The only text difference is the int8 pack log's info line naming `model_runner.py --action tflite
  --pack` instead of `tflite_pack_run.py` (21fbc8f).
- `make check` passes in fp32 (6 of 6 steps) and int8 (7 of 7); its regression, pack and tflite match the pkf runs.
- `make check FMT=bf16` failed at `gpnae-verilator` on 7b711bd: GPNAE's regression gates bf16 accuracy against the
  exact functions at 6.25% and gets tanh 8.59% worst (sigmoid 7.22%), the open item in `sienna-uniform-format`.
  The lane is bit-exact against its model (`tlf_x_gpnae_bf16_hw`, `--model hw`: 7200 of 7200). Soham chose to report
  it, not gate it (2026-10-05); departure 9 has how.
- Follow-up gate on 972464c (`tlh_check_*`, `make check N=16 TILE=4`): fp32 6 of 6, bf16 6 of 6 gated with
  `gpnae-accuracy` reported (worst 8.59% against 6.25%, unchanged; bit-exact step 0.0000%), int8 7 of 7. Each run's
  regression words and cycles, pack log and (int8) TFLite lines are identical to the `pkf_` runs.
- /proj/work was at its quota during the gate, so the run outputs live on `/proj/scratch/spramanik/sienna_tlf`,
  symlinked into `sienna_jobs/runs` and `snaps`.
