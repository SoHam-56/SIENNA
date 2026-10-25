---
name: sienna-rtl
description: Working knowledge of the SIENNA SystemVerilog accelerator repo — a Float32 systolic-mesh matmul engine feeding 8 polynomial-activation lanes, then maxpool and dropout. Use this whenever touching SIENNA RTL, its testbenches, its Python regression/stimulus tooling, or its Makefiles; whenever reading SIENNA simulation logs or waveform traces; whenever a SIENNA test fails, times out, or reports mismatches; and whenever the question involves the SystolicMesh, GPNAE/TYTAN, ArithmeticLibrary, Maxpool_2D, dropout, or fwft modules. Use it even when the request sounds routine ("why is this test failing", "add a module", "bump the matrix size") — this repo has several non-obvious traps that silently produce wrong answers.
---

# SIENNA

> **Names since the restructure (Batch A Stage 4, 2026-10-24).** `sienna_top` is the one top: today's pipeline body is `src/pipeline/sienna_pipe.sv`, `sienna_layer` is `sienna_sched` under `sienna_top` with `SET_PORT = 0`, `sienna_multi` is `sienna_top` with `SET_PORT = 1, COPIES > 1`. The format is the build's define (`make FMT=` gives `+define+FMT_<FMT>`): `EXP_W`, `MAN_W`, `IS_INT`, `ACC_W` and `OUT_W` are gone from SIENNA's RTL and test_config_pkg, in favour of `sienna_fmt_pkg`'s constants (`DATA_WIDTH`, `ACC_DATA_WIDTH`, `RESULT_DATA_WIDTH`, `HAS_REQUANT_EPILOGUE`, `HAS_RESIDUAL_ADD`, the latencies) and bundles (`lane_entry_t`, `requant_scale_t`, `add_params_t`). File:line references below predate it.

A Float32 neural-network accelerator pipeline in SystemVerilog, simulated with Verilator (primary) and VCS. One matrix flows end to end: matmul, then an activation function, then 2-D max pooling, then dropout.

Read `references/architecture.md` before changing dataflow or FSMs, `references/verification.md` before touching a testbench or the regression scripts, and **`references/known-issues.md` before debugging anything** — several live defects are catalogued there with evidence, and re-deriving them costs hours. `references/file-map.md` says what every file is and, importantly, which ones are dead.
For anything involving a second matrix or a stream of sets, read the `sienna-back-to-back`
skill and known-issues #15 first — five completion flags in the mesh are cleared only by
reset, and the obvious test passes anyway because it feeds the same matrix twice.

## The pipeline in one pass

```
matrix_west.mem (A) ─┐
                     ├─> SystolicMesh ══16-wide══> 16 × GPNAE lanes ──> gpnae_out_mem
matrix_north.mem (B)─┘   (C = A×B)        (256 deep)     (activation)        (N×N buffer)
                                                                                   │
                     out[lanes] (L9) <── dropout <── Maxpool_2D <── fwft FIFO2 <─ window
                                        (8 lanes)   (8 lanes)      (8 × 16)    dispatcher
```

`src/pipeline/sienna_pipe.sv` owns this, as one copy inside `src/top/sienna_top.sv`, the only top (since 2026-10-24): `SET_PORT = 0` drives it from `src/pipeline/sienna_sched.sv` (a layer configuration and two row streams), `SET_PORT = 1` from the ports (a put per set), and `COPIES` puts several copies behind one set-contract host; the choice is wires only. Internal paths are `dut.G_LAYER.pipe` / `dut.G_LAYER.sched` and `dut.G_SET.G_COPY[c].pipe`. Since 2026-09-22 there is no single outer FSM: the mesh queues its own sets, an activation stage (`g_state`: G_IDLE → G_FEED → G_LATCH → G_ROUND) turns one mesh result into `gpnae_out_mem`, and a pooling stage (`p_state`: P_IDLE → P_DISPATCH → P_WAIT) drains it through maxpool and dropout. Up to `SETS_IN_FLIGHT` (15) sets are in flight behind the host link; `pipeline_complete_o` is a one-cycle pulse per set. Since 2026-10-07 (branch `credits`) every module boundary is a credit link (`credit_link_if`): the host puts a set on L0, the mesh pushes results as wide beats on L3, the lanes, FIFO2, maxpool and dropout take and give words on credits, and the output is one L9 link per lane that a downstream consumer can stall. The `sienna-back-to-back` skill tables every boundary as built; the `sienna-credits` skill (section "As built") has the contract, the deviations and the measured cost. The activation lanes are `gpnae_poly` (fitted polynomials) plus `gpnae_tail` for inputs past the fits, not the published `gpnae`; see known-issues #17. Since 2026-09-23 the activation stage fills all lanes in parallel: each wide result beat gives every lane its next element, then every lane starts together; there is no FIFO1 and no lane-by-lane fill FSM. Since the credit links there is no per-lane maxpool feeder (`mp_state` is gone): FIFO2 streams each window into Maxpool_2D on credits.

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

**Never build or simulate on the login node (wspd0).** It is shared by 400+ users and an
admin flagged a Verilator build there on 2026-09-22. Submit through the Slurm farm with
`blaunch`; farm nodes have no gcc-toolset-13, so take gcc 13.2 from `launch` instead:

```bash
blaunch --cpus 16 --mem 32 --time 01:00:00 --jobname vtb_mesh \
  --output mesh.log --error mesh.log \
  launch -c gnu/gcc@13.2 -- ./sim.sh TRACE=0      # sim.sh: VERILATOR_ROOT + -fcoroutines wrapper, then make
squeue -j <jobid>                                   # poll until it leaves the queue
```

Keep job scripts and logs on `/proj`, not `/tmp`, which is node-local. Two concurrent jobs
on the same tree must not share a build directory: pass `VERILATOR_DIR=<dir>` to the top
Makefile. `launch -l` lists the available tool versions.

`GPNAE/scripts/run_regression.sh` already does all three and execs `regression.py`, so use it
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
files or a stale archive from a previous toolchain. A stale build directory can also
build cleanly and then segfault before the testbench prints anything (seen 2026-09-23 on every
T=16 matmul test); a clean directory passed the same configuration. Remove or move that
directory and rebuild — **do not** use `make clean` for this, because its
`rm -f *.log` also deletes `performance_analysis_report.log`, which is untracked.

## Build and run

Everything runs from the repo root through the top-level `Makefile`.

```bash
make verilator                  # build + simulate, no waveform
make verilator TRACE=fst        # + compact waveform (preferred)
make verilator TRACE=vcd        # + VCD (large)
make lint                       # Verilator lint only
make pipeline                   # the pipeline suite via regression.py (FMT below)
make pipeline TEST=sigm         # one test, matched by name substring
make pipeline N=16 TILE=4       # override matrix and tile size
make wave                       # open the last trace in surfer
make check-files                # verify every file in DESIGN_FILES exists
```

`make` with no target prints help, including the current configuration and file counts. Submodules build standalone with `make sm-verilator` and `make gpnae-verilator`, or directly via `make -C SystolicMesh` / `make -C GPNAE`.

### Selecting the precision

`FMT=fp32|bf16|int8` (default `fp32`) sets the format of every build and run; any other value stops make. `N`, `TILE`, `LANES` (32), `COLLAPSE_K` (1) and `TEST` go with it, one word each.

```bash
make pkg FMT=bf16 TEST=matmul_random_tanh       # test_config_pkg.sv + that test's stimulus (default matmul_relu_nopool)
make verilator FMT=int8                         # make pkg, then the package guard, then build and run
make verilator TB_MODE=layer GEN_PKG=0         # TB_sienna's other modes (layer, sets, copies; pipeline is the default), each built as TB_sienna_<mode>_sim
make lint FMT=bf16                              # also through pkg and the guard; so are debug, perf and vcs
make pipeline FMT=int8 N=16 TILE=4              # 41 tests in int8, 32 in fp32 and bf16 (each with its packed tests)
make gemm FMT=int8 QUICK=1 PYTHON=<venv>/bin/python   # regression.py --action gemm; takes TILE
make model FMT=bf16 MODEL_DIR=<dir>             # model_runner.py has no default model dir; it refuses int8
make perf-analysis FMT=bf16                     # regression.py --action perf; make perf is still the profiling build
make tflite FMT=int8 PYTHON=<venv>/bin/python   # model_runner.py --action tflite; refuses any FMT but int8
make pack FMT=int8 N=32 PYTHON=<venv>/bin/python  # packed layers vs each job alone on the layer contract (regression.py --action pack)
make multi FMT=bf16                             # TB_sienna (copies) on sienna_top at COPIES = 2 (regression.py --action multi)
make check-fmt check-hier                       # no RTL outside the package names a format; each define's hierarchy holds only its units
make sm-verilator FMT=int8                      # SystolicMesh's model checks (aril-fpu, aril-narrow, model-tests), then its mesh-regression at N, TILE, COLLAPSE_K; SM_CHECKS=0 skips the checks
make gpnae-verilator FMT=int8                   # GPNAE's lane gate, the lane sienna_top uses, on every input against the true function; GPNAE_TIER=1 or 2 for the 5% and 10% tiers
make regression MODEL_DIR=<dir> PYTHON=<venv>/bin/python  # SIENNA's tests (pipeline, model, pack, gemm, tflite) at N = 8, 16, 32, 64 x every tile size x fp32, bf16, int8, one point per target (make -jN, N for the memory), then one table; SMALL=1: N = 8, 16; FMT=<f>: one format (sienna-tooling)
make -C SystolicMesh regression SMALL=1                 # each submodule has its own (GPNAE, SystolicMesh/ArithmeticLibrary too)
make regression/bf16/N8_T2/pipeline              # one point; make regression-summary reprints the table from testbenches/results/regression
make sim GEN_PKG=0                               # the last TB_sienna (pipeline) build on the stimulus now in testbenches/; refuses a build of other inputs, package items (but the run-time knobs) or flags
make verilator SIM_THREADS=2                     # Verilator --threads (default nproc; SystolicMesh's 8); make -jN regression sets it per point
```

SIENNA's Python is two scripts: `regression.py` (hardware sanity, every check above behind `--action`) and `scripts/model_runner.py` (the host stack: numerics, lowering, tiling, packing, the layer file, backends `RtlLayer` / `RtlSets` / `Emulator`). The `sienna-tooling` skill has their actions and the map from the deleted scripts (`pack_regression.py`, `gemm_sweep.py`, `perf_analysis.py`, `tflite_*.py`, `test_*.py`).

- **`make verilator` rewrites the package and stimulus.** After `make gen-matmul` / `gen-conv`, or anything else that wrote its own, build with `GEN_PKG=0` and that package's `FMT`, `N`, `TILE`, `LANES`. `regression.py` (every action, perf included) and `model_runner.py` pass all of them to the make they run.
- **The format is the define.** `make FMT=<f>` adds `+define+FMT_<F>` (ArithmeticLibrary's `fmt.mk`; `FMT_DEF=` builds with none, which is fp32), and only `sienna_fmt_pkg` and the arithmetic units read it. test_config_pkg carries no format. **The guard (`pkg-geometry`)** stops the build unless the package's `N`, `TILE_SIZE`, `NUM_LANES` and `COLLAPSE_K` are the build's; `regression.py --action selftest` covers it without a build.
- `COLLAPSE_K` must be 0 or 1, and 0 is refused for every target but `sm-verilator` and `help`: `TB_sienna (pipeline)` has no parameters, so it always builds `sienna_top`'s default, 1. A collapse-k 0 SIENNA build needs a testbench that sets `sienna_top`'s `COLLAPSE_K`; the collapse-k 0 mesh is checked by SystolicMesh's own regression.
- **TB_sienna (pipeline) reads ten settings at run time** from `testbenches/test_knobs.mem` (activation, set count, bias, cache, accumulate passes, mixed activations, packing, training, dropout seed), written with the package by every generator; the package still holds them for the other testbenches. Without the file TB_sienna (pipeline) stops at time 0. So one build runs every pipeline test of its hardware group (sienna-tooling).
- The tracked `testbenches/test_config_pkg.sv` is the fp32 default (`make pkg FMT=fp32 N=16 TILE=4`). Every build overwrites it, so don't commit it from a run.
- `PYTHON` (default `python3`) runs the scripts; `tflite`, `model` and make regression's tflite and model points need the `tflite` package (`PYTHON=<venv>/bin/python`).
- **Packing ports.** On the set contract each start's sideband carries `pack_shift`, `pack_map`, the activation per entry and, in int8, each entry's `lane_entry_t` (requantize zero point and clamp, lane rescale and SELU words); tied to 0 they give today's results and cycles. On the layer contract the same come as configuration (`cfg_pack_shift_i`, `cfg_pack_map_i`, `cfg_pack_act_i`, `cfg_pack_lane_i`; the layer file's L, P and E lines). Packing needs `LANES` = N or a multiple: the default 32 cannot pack at N = 64 (use `LANES=64` or `128`). The `sienna-packing` skill has the rules and refusals.
- `gpnae-verilator` runs GPNAE's lane gate in FMT at `GPNAE_TIER` (default 0); it reads no coefficient file.

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

`sienna_top` used to push into lanes while they processed, which is exactly the
concurrent push/pop case that the old design got wrong, so this is worth actual measurement
rather than assumption. Start with `make pipeline`.

**Also open:** `src/top/sienna_top.sv` had the `SystolicMesh` parameter overrides
(`MATRIX_SIZE`/`TILE_SIZE`/`DATA_WIDTH`) commented out, with `MATRIX_SIZE` hard-coded to 16
in the module to compensate. Both were reverted to HEAD on 2026-09-19 as an unjustified
workaround rather than a fix. If they were commented out to dodge a real problem, it will
come back — find out what it was.

Everything is committed and pushed across all four remotes. `GPNAE/tb_gpnae.vcd` (a 5 MB `+dump`
waveform), `noStart.ron` and `TB_gpnae_behav.wcfg` are no longer tracked; they stay on disk and
GPNAE's `.gitignore` covers them.

## Things that will bite you

**`testbenches/test_config_pkg.sv` is generated, not source.** `write_sv_package()` (model_runner.py's Device build section, called by regression.py) overwrites it on every `--action gen` and on every regression test. Hand-edits survive exactly until the next run. To change a test's configuration, change the entry in `PIPELINE_TESTS` or the `generate_vectors()` defaults in `regression.py`.

**The pass criterion is a tolerance, not bit-exactness.** The FP32 adder and multiplier truncate rather than round (`fp32Adder.sv` takes `norm_man[25:3]`, `fp32Multiplier.sv` takes `raw_product[46:24]`), so hardware results sit a few ULP below the NumPy reference. In a typical random-data run 0–15% of elements match exactly and the rest pass on tolerance. **A low exact-match rate is normal and is not the bug you are looking for.** Real failures show up as `[FAIL]`, as `act=MISSING`, or as a timeout. GPNAE's own suite passes on either a relative or an absolute bound (`numpy.isclose` semantics): near zero, the fp32 evaluation of `e^x - 1` cancels, so a correct result with a negligible absolute error still shows a large relative error. `matmul_ident_selu` is the exception at 100% exact: an identity matrix gives only `selu(1.0)` and `selu(0.0)`, each a single exact multiply.

Be aware the tolerance check does not measure what it claims: every testbench in the repo compares integer bit patterns rather than float values, by two different mechanisms. It still discriminates correctly for the errors this design produces, but do not read the "Tol%" figures literally, and never build a new check on `$bitstoshortreal`. Known-issues #7 has the detail and a portable replacement.

**The pipeline's lanes (`gpnae`, since Batch B Stage 2) take a 3-bit control word: SELU (`001`), sigmoid (`010`), tanh (`011`), ReLU (`100`), linear (`101`), and a 2-bit accuracy tier.** ReLU and linear skip the polynomial and are exact. `sienna_pipe` latches each accepted set's activation and tier from its L0 sideband (`sienna_set_side.svh`; on the layer contract the tier is `cfg_tier_i`), so sets in flight keep their own; the degree, pieces and coefficients are data in each lane's coefficient memory, which `coef_wr_i`/`coef_addr_i`/`coef_data_i` rewrite while the lanes are idle (model_runner's `write_coef`, TB `+coef=`).

**The top-level build has one `ArithmeticLibrary`.** The root `Makefile` and `src/sienna_rtl.f` compile only `SystolicMesh/ArithmeticLibrary` and only the GPNAE files the `gpnae` lanes use, and `--Wno-MODDUP` is gone, so a module defined twice stops the Verilator build (MODDUP). `GPNAE/ArithmeticLibrary` is a second checkout of the same commit, for GPNAE's standalone build: change the library in `SystolicMesh/ArithmeticLibrary`, fast-forward GPNAE's checkout from it, and bump both pointers.

**An assertion on a host-driven input samples it one cycle late.** `TB_sienna (pipeline)`, `TB_SystolicMesh`, `TB_sienna (sets)` and `TB_sienna (copies)` drive inputs right after the clock edge. Verilator's flops see the new value at that edge, but a concurrent assertion samples the value from before it, so a term combinational from such an input meets state the edge has already changed: `host_accept`-based checks never fired, and the old `a_read_outstanding` missed a read taken on the edge its result was written. Assertions that mix a host input with state therefore check registered copies taken at the edge (`sa_*` and `wcw_q`/`rd_q` in `SystolicMesh.sv`, `acc_*` in `sienna_top.sv`). `TB_sienna (layer)` and the unit benches drive 1 ns after the edge, so they are aligned either way. With `+verilator+error+limit` a firing changes neither the exit code nor the TB's counts: `regression.py`, the mesh's `regression.py` and model_runner's `RtlLayer` and `RtlSets` fail a log that holds `Assertion failed` (and SIENNA's on any `%Error`).

**An int8 residual layer is two sets, and the first fills no bank yet needs one.** `sienna_sched` turns each output tile of an int8 ADD layer into a *hold set* (the conv's last depth pass, requantized into `hold_mem`, no output) and an *add set* (the residual through the identity pass, requantized, then `add_lanes` with the `hold_mem` beat of the same index, then the normal bypass or lane path). Both live inside the activation stage: no new credit link. A hold set still takes an L3 grant, so it waits for a free activation bank (`l6_rsv` counts its reservation from the grant until `hold_done`) although it writes none; a hang with a hold set at the head of the stage is a bank shortage, not an ADD bug. `hold_mem` keeps no set tags: its order rests on `a_rq_one_set` (one set in the requantize at a time), and `a_hold_push_empty` / `a_add_pop_full` catch an overwrite or an early read. The set after an add set waits for `add_lanes` to drain (`rq_room`, `rq_drain` read `busy_o`). Floats refuse the residual sideband (`a_float_no_add`).

**The lanes read no coefficient file.** `gpnae_coef_mem` resets to the literal tables `fit_poly_coeffs.py` writes into it from `GPNAE/scripts/gpnae_coef_default.json`, and the host changes them only through the load path. Every matrix `.mem` is `$readmemh`.

**The GPNAE FIFO is not first-word-fall-through, and its read pointer has two owners.** `InputFIFO` presents the *previously popped* word, so a consumer must pop before it samples. The TYTAN controller does; the SELU FSM did not, which cost it the first element of every batch. Both the TYTAN controller and the SELU FSM can issue pops, arbitrated by decoding `current_state` in `fifo_rd_en_o`. Any change to element consumption in GPNAE has to account for both. See known-issues #3.

## Milestone tags

Each repo tags its own milestones: the name says what that version contains, the number counts that repo's milestones, and the tag message gives the date and a one-line description. SIENNA and SystolicMesh share names because every SIENNA milestone so far was also a mesh change (SystolicMesh is tagged at the commit SIENNA pins); GPNAE has its own; ArithmeticLibrary is a library and is not tagged. SIENNA's submodule pointers record which GPNAE and ArithmeticLibrary commit each SIENNA tag used.

| Repo | Tag | Commit | Date | What it contains | Was |
|---|---|---|---|---|---|
| SIENNA, SystolicMesh | `handshake_tile_v1` | 5fdeaeb, 98b429b | 2026-09-24 | the valid/ready handshake PE tile beside the synchronous one | `legacy_tile_v1` |
| SIENNA, SystolicMesh | `serial_mesh_v2` | bb998cb, a4cea39 | 2026-09-25 | synchronous tiles only; the mesh takes one set at a time | `serial_mesh_v1` |
| SIENNA, SystolicMesh | `pipelined_mesh_v3` | 012b013, 407b2bc | 2026-09-26 | pipelined mesh, sets back to back; fp32 only | `pre_synthesis_v1` |
| SIENNA, SystolicMesh | `three_formats_v4` | 1a6b081, a32f237 | 2026-10-04 | one format per build: fp32, bf16 or int8 (TFLite requantize) | `pre_packing_v1` |
| SIENNA, SystolicMesh | `job_packing_v5` | 3ff67e9, c569e1f | 2026-10-07 | several small jobs packed on one mesh; credits only at the entry | new |
| SIENNA, SystolicMesh | `stage_credits_v6` | 09c4dc4, 10f6831 | 2026-10-10 | every module boundary on one credit link, output back-pressure, optional register stages; bf16 sums in fp32, gated model runs and model checks, dead code removed (moved from 68625d6, 3fd3ef6) | new |
| GPNAE | `tytan_vdat2025` | 59c0aa6 | 2025-10-29 | the Taylor-series lane as published (VDAT 2025) | same |
| GPNAE | `poly_lane_v1` | f1482f4 | 2026-09-25 | the fitted-polynomial lane, exact past the fits, ReLU and linear; fp32 | `pre_synthesis_v1` |
| GPNAE | `three_formats_v2` | 0d407a7 | 2026-10-01 | the polynomial lane in fp32 or bf16, and a separate int8 lane | `pre_packing_v1` |
| GPNAE | `credit_lanes_v3` | afd786b | 2026-10-10 | the polynomial lanes take inputs and give results on credit links; a circular input FIFO keeps word order; terms_i and dead signals removed (moved from b1ef12d) | new |

## Conventions

Ports are suffixed `_i` / `_o`; the reset is `rstn_i`, active low and asynchronous. Most new RTL uses `logic` with `always_ff` / `always_comb`; the older TYTAN and ProcessingElement code uses `reg`/`wire` with `always @`. Both styles are present and neither is being migrated — match the file you are in.

FSM states are `typedef enum logic [n:0]` with a `current_state`/`next_state` pair: a combinational block computes `next_state`, a sequential block registers it. Follow that shape.

`timescale` is `1ns / 100ps` nearly everywhere (`src/fifo/fwft.sv`, `src/dropout/dropout.sv` and `src/pool/Maxpool_2D.sv` use `1ns / 1ps`). The testbench clock is 10 ns.

Adding a design file means adding it to the appropriate `*_FILES` list in the root `Makefile` — there is no glob. `make check-files` verifies the list.

## Where the numbers come from

Cycle counts and utilization figures quoted around this project have two very different provenances, and they get conflated easily:

- **Measured**: SystolicMesh cycle counts (399/849/1797/3885 for N=16 at T=2/4/8/16), the 18,769-cycle end-to-end tanh latency, and the FPGA synthesis numbers (98.26% LUT, 18.05% FF, 4 BRAM tiles, 0 DSPs on `xc7a200tfbg484-3`). These come from real runs and are reproducible.
- **Modeled**: every GOPS and GB/s figure in `performance_analysis_report.log`, and all INT8 projections. They assume a 950 MHz FP32 clock and a 1.5 GHz INT8 clock that no timing run has demonstrated, plus an assumed 16× area ratio. The report labels these as back-calculations; keep that label when quoting them.

`performance_analysis_report.log` is untracked and its RTL breakdown was written by inspection, so treat its per-stage cycle attribution as an argument rather than an instrumented measurement.
