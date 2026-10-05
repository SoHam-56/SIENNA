---
name: sienna-tooling
description: Use when running, changing or extending SIENNA's Python tooling - the two scripts regression.py (hardware sanity: regressions, unit benches, packed layers, GEMM sweep, perf, tool self-tests, one-verdict gate) and model_runner.py (the host software stack: TFLite loading, lowering, tiling, packing, the device protocol to TB_model_run, backends, reference numerics). Also use when someone asks how a software developer should build on SIENNA, what the layer-file protocol is, or where a former script (pack_regression.py, gemm_sweep.py, perf_analysis.py, tflite_*.py, test_*.py) went.
---

# SIENNA tooling: two scripts

**Status: design approved 2026-10-05 (Soham: "keep submodule scripts, delete old ones, go ahead"); implementation in progress.** Branch `tooling` (SIENNA). Soham asked for exactly two
Python scripts in SIENNA: one for hardware sanity, one that runs models and stands as the reference host software a
software team builds its own layer from ("this model runner is supposed to be the compiler").

## What exists today (main b1cfacb)

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

## Done

- TB_sienna_layer renamed TB_model_run (file, module, binary, make arguments, references; it had no dump block).
