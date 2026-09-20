# SIENNA verification

How the testbenches, the Python tooling and the logs fit together.

## Three independent verification layers

| Layer | DUT | Golden model | Checks results? |
|---|---|---|---|
| Full pipeline | `sienna_top` | NumPy, in `regression.py` | yes — but see the tolerance caveat below |
| SystolicMesh IP | `SystolicMesh` | float64 NumPy matmul | yes, with the same caveat |
| Arithmetic units | `fp32Adder`, `fp32Multiplier` | Berkeley SoftFloat via DPI-C | yes, ±2–3 ULP |
| GPNAE activations | `gpnae` | `gpnae_tests.py` golden model | yes, 1% relative or 1e-6 absolute — `TB_gpnae.sv` |

`TB_gpnae.sv` is file-driven: `GPNAE/regression.py` generates stimulus and
golden values into `testbenches/stimulus/`, and the testbench scores every
element. Run it with `./run_regression.sh` from the `GPNAE` directory, or
`make verilator` for a single pass over the committed stimulus. It keeps the
three original protocol assertions (no write when full, no write when not
idle, result stable at `done_o`).

It replaced a version that printed results but never compared them, and which
drove `signal_sigmoid` in all three cases — strictly increasing from -5 to +5,
so it contained no positive-followed-by-negative transition and could not
expose the SELU operand race (known-issues 5d).

Nine stimulus patterns run per activation. `act_edge` places ±range literally
and is the deterministic reproducer; `act_log_uniform` spreads across
exponents and is what reaches the near-zero cancellation region.

`TB_gpnae_activations.sv` fills that gap: all three activations over a vector
mixing positive, negative and zero inputs, every result compared against a
golden value. It is what caught the SELU λ·α constant bug and `tanh(0)`.

**Tolerance caveat.** None of the tolerance checks measure float-relative error.
`TB_SystolicMesh.sv` and `TB_SystolicArray.sv` use `$signed()` on FP32 bit
patterns; `TB_sienna_top.sv` uses `$bitstoshortreal`, which Verilator maps to
the 64-bit `$bitstoreal`. Both end up comparing integer bit patterns rather
than values. See known-issues #7 — including a portable `f32()` replacement.

## Full-pipeline flow

`regression.py` is generator, orchestrator and scoreboard in one file.

1. `generate_vectors(cfg)` builds A and B, runs the NumPy golden model end to end — `_ref_matmul` → `apply_activation` → `apply_maxpool_2d` → `apply_dropout` — and writes `matrix_west.mem`, `matrix_north.mem`, `expected_output.mem` into `testbenches/`.
2. It also writes `testbenches/test_config_pkg.sv` and a human-readable `<test>_expected_flow.txt` showing the value of every intermediate stage.
3. `_run_make_live()` shells out to `make verilator`, streaming `[STAGE]`/`[STATUS]`/`[FATAL]` lines to the console and the whole log to `testbenches/results/pipeline/<test>.log`.
4. `_parse_log()` scrapes the `Total`/`Exact`/`Tol pass`/`Failed` counters and the completion cycle count.
5. `dump_hardware_trace()` parses `testbenches/hardware_trace.txt` — written by the testbench — into `<test>_data_flow.txt`, laid out to line up with the expected-flow file.
6. On failure the suite aborts immediately (`sys.exit(1)`) and dumps the hardware trace to the console.

The seven tests in `PIPELINE_TESTS`: `matmul_ones_idle`, `matmul_small_exact`, `matmul_ident_selu`, `matmul_random_sigm`, `matmul_random_tanh`, `conv_basic_selu`, `conv_basic_tanh`.

### Debugging a mismatch

Diff the two flow files side by side:

```
testbenches/results/pipeline/<test>_expected_flow.txt   # NumPy, per stage
testbenches/results/pipeline/<test>_data_flow.txt       # hardware, per stage per lane
```

Both print `value (0xhex)` in matrix layout, so the stage where they diverge tells you which block to look at: Stage 1 is the mesh, Stage 2 is GPNAE, Stage 3 is Maxpool. The hardware file warns when lane element counts are uneven, which is the usual symptom of a stalled lane.

For FSM-level stalls use `testbenches/pipeline_lane_status.txt`. The testbench writes a per-lane snapshot on every outer-FSM transition, on every `mp_state` change, and every 5000 cycles. Each line carries `fill_count`, `done_count`, `load_finalized`, `lane_collected`, `mp_state`, `mp_window_fed`, `mp_windows_done`, `lane_windows_total` and `dropout_out_count`. A lane with `load_finalized=1` and `lane_collected=0` that never advances is a GPNAE that never asserted `done_o`.

### Testbench mechanics

`TB_sienna_top.sv` (474 lines) loads the three `.mem` files into queues, resets, drives the west and north write ports in parallel `fork`/`join` threads, pulses `start_pipeline_i`, then waits on `pipeline_complete_o` with a 5000-cycle heartbeat and a 200,000-cycle timeout.

Results are captured **on the fly** by an `always_ff` block that pushes `dut.dropout_data_out[lane]` into `actual_results` whenever `dut.dropout_valid_out[lane]` is high — not read back from a buffer at the end. Order across lanes within a cycle is lane-index order.

`check_tolerance()` converts with `$bitstoshortreal`, which is correct. Modes are `ABSOLUTE`, `RELATIVE` or `BOTH`; the configured mode is `RELATIVE` with `REL_TOL = 0.01`.

Be aware the `[PASS-TOL]`/`[FAIL]` lines print `exp=0.000000 act=0.000000` for every element — the `%f` fields in that `$sformatf` do not render under Verilator. The `abs=` and `rel=` fields and the hex values are correct, so read those. Do not chase this as a numerical bug.

## SystolicMesh IP regression

`SystolicMesh/regression.py` is a separate suite with a different trick: it patches `MATRIX_SIZE`, `TILE_SIZE` and `NUM_TEST_SETS` directly into `TB_SystolicMesh.sv` by regex, once per (tile × group), then calls `make` once per test. Because only `.mem` files change between tests in a group, make skips recompilation and reuses the binary. The testbench is always restored in a `finally` block — including on `KeyboardInterrupt`.

```bash
python regression.py --matrix-size 16                  # all groups, all tiles
python regression.py --matrix-size 16 --group matmul
python regression.py --matrix-size 64 --fast           # one middle tile size
```

Output goes to `SystolicMesh/testbenches/results/readiness/`, with a Markdown summary in `readiness_report.md`.

Stimulus comes from `matmul_tests.py` (8 generators) and `conv_tests.py` (9 generators). Both pad to a fixed set count — `MATMUL_NUM_SETS = 5`, `CONV_NUM_SETS = 3` — by repeating the last real set, which is what lets one compiled binary serve every test in a group.

Conv is pure data layout: the hardware only ever does `A × B`. `conv_tests.py`'s module docstring is the authoritative im2col spec, covering basic non-overlapping stride, overlapping stride via multiple batches, multi-output-channel, and multi-input-channel. Conv tests require `N` to be a perfect square.

## Arithmetic unit tests

`ArithmeticLibrary/{Adders,Multipliers}/FP32/testbenches/` drive 2000 random vectors plus hand-picked corner cases against Berkeley SoftFloat through DPI-C, checking both the result and the `overflow`/`underflow`/`invalid` flags.

Tolerance is ±3 ULP for the adder, ±2 ULP for the multiplier, with explicit escape hatches for NaN equivalence and for flush-to-zero versus SoftFloat's denormal results. Random inputs are sanitized by `fix_random_input()`, which rewrites denormals to the smallest normal and Inf/NaN to the largest normal, so the random phase never exercises those — corner cases cover them separately.

The `*VIVADO.sv` variants read pre-generated vectors from `vectors.mem` instead of calling DPI-C, because linking SoftFloat into xsim is fragile. `generate_vectors.sh` builds SoftFloat if needed, compiles `gen_vectors.cpp` against it and emits that file. `ArithmeticLibrary/README.md` documents the Vivado DPI linker workaround (symlinking Vivado's bundled `ld` to the system one).

## Current status

**7 of 7 passing** on a clean build (Verilator 5.035, N=16, TILE=4):

| Test | Result | Exact | Tol | Cycles |
|---|---|---|---|---|
| `matmul_ones_idle` | PASS | 100.0% | 0% | 1838 |
| `matmul_small_exact` | PASS | 100.0% | 0% | 1838 |
| `matmul_ident_selu` | PASS | 100.0% | 0% | 2190 |
| `matmul_random_sigm` | PASS | 16.0% | 84.0% | 10609 |
| `matmul_random_tanh` | PASS | 2.5% | 97.5% | 18769 |
| `conv_basic_selu` | PASS | 33.3% | 66.7% | 10023 |
| `conv_basic_tanh` | PASS | 0.0% | 100.0% | 18769 |

The three pass-through / identity tests are 100% exact, as they should be:
`idle` applies no arithmetic, and `selu(1.0)` and `selu(0.0)` are a single exact
multiply. The transcendental tests sit almost entirely in the tolerance band
because the FP units truncate.

`matmul_random_tanh` at 18769 cycles reproduces the figure quoted in
`performance_analysis_report.log` exactly.

Before this round the suite was 4 of 7, with `matmul_ones_idle` and
`matmul_small_exact` timing out and `matmul_ident_selu` failing on one element.
`conv_basic_selu` had never been reached at all, because the suite aborts on
first failure — which is how the SELU λ·α constant bug survived.

SystolicMesh IP: 36/36 conv configurations pass (N=16, tiles 2/4/8/16). No
matmul-group logs are present from that run.

GPNAE activations: PASS, 9/9 patterns, 0 failures across 6480 element checks.
Measured cost per input: SELU 154 cycles, sigmoid 347, tanh 602.

## `run_real_model.py`

Untracked helper that loads a PyTorch checkpoint, slices an `N×N` block out of the first suitable weight tensor, pairs it with a seeded random activation matrix, and pushes it through the same golden-model and simulation path as the regression suite under the name `real_model_test`. It forces `DROPOUT_P_PERCENT = 0` for determinism. The default `--model` path (`/home/admin/Downloads/Model/end2end.pt`) is from a different machine and needs overriding.
