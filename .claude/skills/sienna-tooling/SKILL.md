---
name: sienna-tooling
description: Use when running, changing or extending SIENNA's Python tooling - the two scripts regression.py (hardware sanity: regressions, unit benches, packed layers, GEMM sweep, perf, tool self-tests, one-verdict gate) and model_runner.py (the host software stack: TFLite loading, lowering, tiling, packing, the device protocol to TB_model_run, backends, reference numerics). Also use when someone asks how a software developer should build on SIENNA, what the layer-file protocol is, or where a former script (pack_regression.py, gemm_sweep.py, perf_analysis.py, tflite_*.py, test_*.py) went.
---

# SIENNA tooling: two scripts

**Status: implemented and verified 2026-10-05 on branch `tooling` (SIENNA only; not pushed), gated tree 7b711bd
(b396aeb..7b711bd; the commit after it changes only this skill). Final gate `tlf_*` (`sienna_report/tooling_gate.log`):
all 106 packing-gate configurations identical to `pkf_*`, `make check` passes in fp32 and int8; `make check FMT=bf16`
fails only on GPNAE's bf16 activation accuracy, the open decision of 2026-09-28 (see Gate).** Design approved 2026-10-05 (Soham:
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
- `--action all` — one verdict: selftest, the SystolicMesh and GPNAE regressions (via make), the SIENNA regression, pack, gemm quick, and tflite (int8); prints a table and exits non-zero on any failure (farm launching stays in sienna_jobs, outside git)

### Makefile

Every target keeps its name and arguments and calls the new script: `regression`, `pkg`, `pack`, `gemm`,
`perf-analysis`, `model`, `tflite` (and a new `check` for `regression.py --action all`). test_makefile_fmt's checks move into
`selftest` and follow the new recipes.

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
(1,353 lines) and `regression.py` (2,923 lines); the check is the global-constraints command in the implementation plan.
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
| `python3 test_makefile_fmt.py` (and the other four `test_*.py`) | `regression.py --action selftest` (34 tests) | |
| | `regression.py --action all [--format --n --tile --lanes]` | `make check` |
| `tflite_int8_run.py [--n --tile-size --lanes --models --work]` | `model_runner.py --action tflite` (same options) | `make tflite` |
| `tflite_pack_run.py [--n --tile-size --lanes --models]` | `model_runner.py --action tflite --pack` (same options) | |
| `model_runner.py --model-dir D ...` | unchanged (`--action model` is the default) | `make model` |
| `import tflite_ref`; `regression.FORMATS`, `op_round`, `quant_act`, `int8_layer_exact`, `write_sv_package`, ... | `model_runner.X` (regression.py imports them by name, so `regression.X` still resolves) | |
| `model_runner.LayerSim`, `Sim`, `EmuSim` | `RtlLayer`, `RtlSets`, `Emulator` (no aliases) | |
| `TB_sienna_layer` | `TB_model_run` | |

Log and result file names did not change (`pack_regression_*.log`, `gemm_sweep_N*.log`, `tflite_int8_N*_T*.log`,
`pipeline_performance_report.log`), so the farm comparison scripts read them as before.

### Departures from the design above, each with its reason

1. **model_runner gained sections but was not reorganised.** It has `# ── Numerics ──`, `# ── Device build ──` and
   `# ── TFLite runs ──`; the frontend, tiling and packing, the layer-file writer (inside `RtlLayer`), the backends and
   the CLI stay in their old order without new headers, and there is no contents block or separate device-protocol
   encoder / decoder. The plan was a behaviour-preserving move (code moved verbatim); the layered reorganisation is
   left as a next step.
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
   is printed "reported, not gated", so `make check` passes with it (an open int8 item, not a tooling one).
10. **regression.py keeps an unused `get_polynomial_terms` import** for the untracked `run_real_model.py`.

### Known gaps (deferred minors, from the ledger)

- `--action selftest` does not parse its arguments (`--action selftest --bogus` runs the tests).
- `--action all` does not flush per line, so a piped console lags; its table names log paths inside the run's tree.
- perf helpers called without `perf_init()` meet None globals (only `perf_main` and the perf self-tests call it).
- model_runner's `ACTIONS` sits at the bottom of the file; the `--action` / `--pack` arguments are declared in the
  sub-parsers for help only.
- No `make -n` self-test for the gemm recipe's `--tile`.

### Gate

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
- `make check FMT=bf16` fails at `gpnae-verilator`: GPNAE's regression gates bf16 accuracy against the exact functions
  at 6.25% and gets sigmoid 7.22% and tanh 8.59% worst, exactly the open item in the `sienna-uniform-format` skill.
  The lane is bit-exact against its model on this tree (diagnostic `tlf_x_gpnae_bf16_hw`, `--model hw`: 7200 of
  7200). Not a tooling fault, and the check was not loosened; bf16 `make check` cannot pass until that decision.
- /proj/work was at its quota during the gate, so the run outputs live on `/proj/scratch/spramanik/sienna_tlf`,
  symlinked into `sienna_jobs/runs` and `snaps`.
