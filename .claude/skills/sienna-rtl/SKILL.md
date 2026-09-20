---
name: sienna-rtl
description: Working knowledge of the SIENNA SystemVerilog accelerator repo — a Float32 systolic-mesh matmul engine feeding 8 polynomial-activation lanes, then maxpool and dropout. Use this whenever touching SIENNA RTL, its testbenches, its Python regression/stimulus tooling, or its Makefiles; whenever reading SIENNA simulation logs or waveform traces; whenever a SIENNA test fails, times out, or reports mismatches; and whenever the question involves the SystolicMesh, GPNAE/TYTAN, ArithmeticLibrary, Maxpool_2D, dropout, or fwft modules. Use it even when the request sounds routine ("why is this test failing", "add a module", "bump the matrix size") — this repo has several non-obvious traps that silently produce wrong answers.
---

# SIENNA

A Float32 neural-network accelerator pipeline in SystemVerilog, simulated with Verilator (primary) and VCS, synthesized with Vivado. One matrix flows end to end: matmul, then an activation function, then 2-D max pooling, then dropout.

Read `references/architecture.md` before changing dataflow or FSMs, `references/verification.md` before touching a testbench or the regression scripts, and **`references/known-issues.md` before debugging anything** — several live defects are catalogued there with evidence, and re-deriving them costs hours. `references/file-map.md` says what every file is and, importantly, which ones are dead.

## The pipeline in one pass

```
matrix_west.mem (A) ─┐
                     ├─> SystolicMesh ──> fwft FIFO1 ──> 8 × GPNAE lanes ──> gpnae_out_mem
matrix_north.mem (B)─┘   (C = A×B)        (256 deep)     (activation)        (N×N buffer)
                                                                                   │
                     final_result_o <── dropout <── Maxpool_2D <── fwft FIFO2 <─ window
                                        (8 lanes)   (8 lanes)      (8 × 16)    dispatcher
```

`src/sienna_top.sv` owns this. Its outer FSM runs IDLE → SYSTOLIC_START_PULSE → SYSTOLIC_PROCESSING → FEED_GPNAE_FIFO → LATCH_GPNAE_COUNT → GPNAE_ROUND → DISPATCH_WINDOWS → WAIT_DOWNSTREAM → PIPELINE_COMPLETE. Two inner FSMs matter as much as the outer one: the `fill_state` machine that hands FIFO1 data round-robin to the 8 GPNAE lanes, and the per-lane `mp_state` machine that feeds pre-packaged pooling windows into each Maxpool.

The key structural idea: the mesh produces results in one flat stream, but activation is the slow part, so the design fans out to 8 parallel GPNAE lanes and reassembles them in `gpnae_out_mem` before pooling. The dispatcher then re-reads that buffer in pooling-window order and scatters windows back across the 8 lanes.

## Toolchain

`--timing` needs C++20 coroutines, so the default RHEL 8 `g++` 8.5 cannot build
the simulation. On a machine where Verilator is installed under `~/.local` and
was configured against the system compiler, three things have to be set before
`make verilator` will work:

```bash
export VERILATOR_ROOT=$HOME/.local/share/verilator   # else it looks in /usr/share
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
# Verilator configured against g++ 8 does not emit -fcoroutines, which
# GCC 10..13 require for <coroutine>. Put a wrapper early on PATH:
#   #!/bin/bash
#   exec /opt/rh/gcc-toolset-13/root/usr/bin/g++ -fcoroutines "$@"
```

`GPNAE/run_regression.sh` already does all three and execs `regression.py`, so use it
rather than repeating the setup. `make lint` works without any of this, because it never
compiles C++.

**Verilator 5.035 here is unstable in specific, repeatable ways.** These cost hours if you
meet them cold:

- A variable written by a clocked `always` block and read from a timing coroutine (an
  `initial` block, a task) is **not coherent** under `--threads`. A cycle counter read back
  correctly on a 352-cycle run and as 0 on a 37,000-cycle one. Measure with `$time` inside
  the coroutine instead.
- Writing the same variable from two contexts crashes the threaded scheduler. Keep one writer.
- `$dumpvars` over a whole testbench crashes `trace_init` when the TB has `real` or `string`
  variables. Scope the dump to the DUT.
- `$bitstoshortreal` silently maps to the 64-bit `$bitstoreal`, decoding every 32-bit pattern
  as a denormal near zero, so every comparison built on it passes. Decode from the exponent
  and mantissa widths by hand.
- `--threads 1` with `--trace --timing` crashes in the stimulus coroutine, while the
  Makefile's `--threads $(nproc)` default runs. `GPNAE/regression.py` has a `GPNAE_PIN=1`
  escape hatch for the opposite failure (thread-pool assertion), off by default.

If the build dies with `No rule to make target '/usr/share/verilator/...'` or
`undefined reference to main`, the `Verilator/` directory holds stale dependency
files or a stale archive from a previous toolchain. Remove or move that
directory and rebuild — **do not** use `make clean` for this, because its
`rm -f *.log` also deletes `performance_analysis_report.log`, which is untracked.

## Build and run

Everything runs from the repo root through the top-level `Makefile`.

```bash
make verilator                  # build + simulate, no waveform
make verilator TRACE=fst        # + compact waveform (preferred)
make verilator TRACE=vcd        # + VCD (large)
make lint                       # Verilator lint only
make regression                 # full 5-test suite via regression.py
make regression TEST=sigm       # one test, matched by name substring
make regression N=16 TILE=4     # override matrix and tile size
make wave                       # open the last trace in surfer
make check-files                # verify every file in DESIGN_FILES exists
```

`make` with no target prints help, including the current configuration and file counts. Submodules build standalone with `make sm-verilator` and `make gpnae-verilator`, or directly via `make -C SystolicMesh` / `make -C GPNAE`.

## State as of 2026-09-20 — read this first

GPNAE was overhauled and its regression is green: **PASS, 9/9 patterns, 0 failures across
6480 element checks**, from 10 failures before. The fixes are in `known-issues.md` 5d and 9.

**The SIENNA top-level pipeline has NOT been re-run since those GPNAE changes.** Verify it
before trusting anything end to end. The changes altered GPNAE's control interface in ways
the integration can feel:

- The GPNAE FSM is now the sole owner of the input-FIFO read pointer — the MAC no longer pops.
- The MAC starts only on an explicit per-element request; its credit counter is gone, so it
  no longer free-runs off `wr_en_i`.
- `InputFIFO` read latency changed: `regceb` is tied high, so `data_o` tracks `rd_ptr` with a
  flat two-cycle latency instead of reloading only on a pop.
- `done_delay` widened 3 -> 4 stages, so each element retires one cycle later.

`sienna_top`'s `fill_state` machine pushes into lanes while they process, which is exactly the
concurrent push/pop case that the old design got wrong, so this is worth actual measurement
rather than assumption. Start with `make regression`.

**Also open:** `src/sienna_top.sv` had the `SystolicMesh` parameter overrides
(`MATRIX_SIZE`/`TILE_SIZE`/`DATA_WIDTH`) commented out, with `MATRIX_SIZE` hard-coded to 16
in the module to compensate. Both were reverted to HEAD on 2026-09-19 as an unjustified
workaround rather than a fix. If they were commented out to dodge a real problem, it will
come back — find out what it was.

Everything is committed and pushed across all three remotes. `GPNAE/tb_gpnae.vcd` is still a
tracked 5 MB generated waveform that `make clean` deletes; `.gitignore` covers `*.vcd` but
that does not apply to an already-tracked file, so it wants `git rm --cached` — ask first.

## Things that will bite you

**`testbenches/test_config_pkg.sv` is generated, not source.** `regression.py`'s `write_sv_package()` overwrites it on every `--action gen` and on every regression test. Hand-edits survive exactly until the next run. To change a test's configuration, change the entry in `PIPELINE_TESTS` or the `generate_vectors()` defaults in `regression.py`.

**The pass criterion is a tolerance, not bit-exactness.** The FP32 adder and multiplier truncate rather than round (`fp32Adder.sv` takes `norm_man[25:3]`, `fp32Multiplier.sv` takes `raw_product[46:24]`), so hardware results sit a few ULP below the NumPy reference. In a typical random-data run 0–15% of elements match exactly and the rest pass on tolerance. **A low exact-match rate is normal and is not the bug you are looking for.** Real failures show up as `[FAIL]`, as `act=MISSING`, or as a timeout. GPNAE's own suite passes on either a relative or an absolute bound (`numpy.isclose` semantics): near zero, the fp32 evaluation of `e^x - 1` cancels, so a correct result with a negligible absolute error still shows a large relative error. `matmul_ident_selu` is the exception at 100% exact: an identity matrix gives only `selu(1.0)` and `selu(0.0)`, each a single exact multiply.

Be aware the tolerance check does not measure what it claims: every testbench in the repo compares integer bit patterns rather than float values, by two different mechanisms. It still discriminates correctly for the errors this design produces, but do not read the "Tol%" figures literally, and never build a new check on `$bitstoshortreal`. Known-issues #7 has the detail and a portable replacement.

**GPNAE has exactly three activations: SELU (`01`), sigmoid (`10`), tanh (`11`).** Control word `00` is not a pass-through and not a mode — `gpnae_control_unit`'s `OP` state sends it to `default: next_state = IDLE`, the lane never asserts `done_o`, and the pipeline stalls in `GPNAE_ROUND` until the testbench times out. `regression.py` used to offer an `"idle"` activation that mapped to `0`; it now raises instead. If you see a stall with 0 words captured, check `ACTIVATION_CODE` before suspecting the RTL.

**`--Wno-MODDUP` is hiding duplicate module definitions.** Both `GPNAE/ArithmeticLibrary` and `SystolicMesh/ArithmeticLibrary` are compiled into the same Verilator build, and both define `fp32Adder`, `fp32Multiplier`, `R4Booth` and `karatsubaUnsigned`. `cntlz8` is defined twice as well, now by the two `ArithmeticLibrary` checkouts rather than by GPNAE's own copy, which has been removed — a GPNAE-only build is clean. Whichever definition the tool takes silently wins. Today the copies are functionally identical — the diffs are only `logic` vs `wire`/`reg` port declarations — so nothing is broken, but editing one copy will produce changes that appear to do nothing. Edit both, or consolidate. Vivado flags this as `CRITICAL WARNING [Synth 8-9873]`.

**Coefficient memory is binary, everything else is hex.** `taylor_coeffs.mem` is loaded with `$readmemb` (`GPNAE/src/TYTAN/Memory/ROM.v`) and holds 32-character binary lines. Every matrix `.mem` is `$readmemh`. Writing a coefficient table in hex fails silently, producing zeros or X.

**The GPNAE FIFO is not first-word-fall-through, and its read pointer has two owners.** `InputFIFO` presents the *previously popped* word, so a consumer must pop before it samples. The TYTAN controller does; the SELU FSM did not, which cost it the first element of every batch. Both the TYTAN controller and the SELU FSM can issue pops, arbitrated by decoding `current_state` in `fifo_rd_en_o`. Any change to element consumption in GPNAE has to account for both. See known-issues #3.

**Six copies of `taylor_coeffs.mem` exist** across the working tree and build directories, currently all identical. The Makefile's `copy_mem_files` sweeps several directories into the simulation directory, so an inconsistent copy will be picked up by position rather than intent. Change `GPNAE/src/TYTAN/Memory/taylor_coeffs.mem` and let the copy rules propagate it.

## Conventions

Ports are suffixed `_i` / `_o`; the reset is `rstn_i`, active low and asynchronous. Most new RTL uses `logic` with `always_ff` / `always_comb`; the older TYTAN and ProcessingElement code uses `reg`/`wire` with `always @`. Both styles are present and neither is being migrated — match the file you are in.

FSM states are `typedef enum logic [n:0]` with a `current_state`/`next_state` pair: a combinational block computes `next_state`, a sequential block registers it. Follow that shape.

`timescale` is `1ns / 100ps` nearly everywhere (`src/fwft.sv`, `Dropout/dropout.sv` and `Maxpool/Maxpool_2D.sv` use `1ns / 1ps`). The testbench clock is 10 ns.

Adding a design file means adding it to the appropriate `*_FILES` list in the root `Makefile` — there is no glob. `make check-files` verifies the list.

## Where the numbers come from

Cycle counts and utilization figures quoted around this project have two very different provenances, and they get conflated easily:

- **Measured**: SystolicMesh cycle counts (399/849/1797/3885 for N=16 at T=2/4/8/16), the 18,769-cycle end-to-end tanh latency, and the Vivado synthesis numbers (98.26% LUT, 18.05% FF, 4 BRAM tiles, 0 DSPs on `xc7a200tfbg484-3`). These come from real runs and are reproducible.
- **Modeled**: every GOPS and GB/s figure in `performance_analysis_report.log`, and all INT8 projections. They assume a 950 MHz FP32 clock and a 1.5 GHz INT8 clock that no timing run has demonstrated, plus an assumed 16× area ratio. The report labels these as back-calculations; keep that label when quoting them.

`performance_analysis_report.log` is untracked and its RTL breakdown was written by inspection, so treat its per-stage cycle attribution as an argument rather than an instrumented measurement.
