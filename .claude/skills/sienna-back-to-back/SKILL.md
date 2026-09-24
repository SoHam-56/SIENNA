---
name: sienna-back-to-back
description: Use when making SIENNA process back-to-back or streamed matrix sets - pipelining stages, overlapping the systolic mesh with activation, re-arming control state after a matmul, or debugging a second set that hangs, returns stale results, or reports completion without computing.
---

# SIENNA back-to-back sets

Design for running a continuous stream of matrix sets through SIENNA with all
stages overlapped. Status as of 2026-09-22: **all four phases done.** Phase 1 fixed
the six re-arm defects below. Phase 2 made the mesh stream: two staging banks and
two result banks. Phase 3 replaced `sienna_top`'s single FSM with an activation
stage and a pooling stage, banked `gpnae_out_mem`, and put a 3-credit interface in
front. Phase 4 is the regression: every test streams 4 distinct sets.
Plans: `implementation-plan.md`, `phase2-plan.md`, `phase3-plan.md`.

## Top-level interface after phase 3

| Port | Meaning |
|---|---|
| `pipeline_ready_o` | a credit and a mesh staging bank are free; host may load a set and pulse start |
| `start_pipeline_i` | pulse after loading; ignored while `pipeline_ready_o` is low |
| `pipeline_complete_o` | **one-cycle pulse** when a set's last output leaves dropout (was a held level) |
| `done_set_id_o` | 2-bit id of that set: accepted starts, counted mod 4 |

Measured by `perf_analysis.py` (12 streamed sets, N=16, T=4, farm, 2026-09-24) on the current
defaults: 32 lanes, one-row host writes (`HOST_WORDS = N`), SyncArray mesh tiles. GFLOPS count the
8192-FLOP matmul at an **assumed** 950 MHz; no timing run exists.

| Config | Single set | Per set, steady | GFLOPS @950 | Bottleneck |
|---|---|---|---|---|
| matmul_ident_selu / conv_basic_selu | 295 | 221 | 35.2 | activation (195) |
| matmul_random_tanh | 313 | 271 | 28.8 | activation (213, 298 with two tails in a lane) |
| matmul_random_sigm | 378 | 263 | 29.6 | activation (278) |
| matmul_random_tanh_train | 321 | 271 | 28.8 | activation (213) |
| matmul_large_{selu,sigm,tanh} | 627-691 | 530-591 | 13.2-14.7 | activation (tails) |

What each change bought, all measured the same way:

| Change | Stage it moved | Steady state per set (SELU / tanh / sigmoid) |
|---|---|---|
| tail overlap (GPNAE `b972abd`) | activation: a lane's first tail element hides under the polynomial | 336 / 459 / 472 -> 266 / 303 / 325 |
| one-row host writes | host load 257 -> 17 | 266 / 303 / 325 -> 264 / 303 / 325 (activation now the limit) |
| 32 lanes | activation 251 -> 195 (SELU), 277 -> 213 (tanh) | -> 221 / 271 / 263 |
| SyncArray tiles | mesh 129 -> 77 (tile phase 94 -> 42) | unchanged; single-set latency -52 |
| four tail contexts (GPNAE `e68882c`) | tail elements in a lane run together | normal data unchanged; large-value 1510-2078 -> 530-591 |

The mesh stage is 77 cycles per set (broadcast 4, tiles 42, reduce 29). It is no longer on the
critical path; activation is. gpnae_tail now works on four out-of-range elements at once
(`TAIL_CONTEXTS`), sharing one multiplier and one adder, bit-identical to running them in turn. `python3 perf_analysis.py`
regenerates `testbenches/results/perf/pipeline_performance_report.log`; `--lanes`, `--n` and
`--tile-size` pick another geometry.

### Larger N (measured 2026-09-24)

Full pipeline at N=32, T=4, one-row host writes, SyncArray tiles: all 8 matmul tests pass (conv needs N to
be a perfect square in the generator). Mesh stage 82 cycles, host load 33, so activation sets the rate:

| Lanes | Tail | SELU steady | tanh steady | GFLOPS @950 (SELU / tanh) | Lanes busy |
|---|---|---|---|---|---|
| 32 | one at a time | 1036 | 1561 | 60.1 / 39.9 | 51-58% |
| 64 | one at a time | 772 | 1122 | 80.7 / 55.5 | 35-41% |
| 64 | four contexts | 288 | 380 | 216.6 / 163.8 | 84-87% |

N=32 sums 32 products per output, so far more activation inputs land past the fitted range; with
the tail run one element at a time, lanes sat idle while one lane worked through its tail elements. Mesh only,
set 0 at N=32, T=4: 136 cycles with SystolicArray, 84 with SyncArray (8192 PEs each), 105 with
collapse-k (1024 PEs). At N=64, T=8 collapse-k takes 197 cycles with 4096 PEs, about 32% PE
utilisation, against 840 cycles and 32768 PEs for the old serial-reduce mesh.

### Several pipelines: `sienna_multi`

`src/sienna_multi.sv` puts COPIES `sienna_top`s behind one host port; each accepted start goes to
the next copy in turn and each copy keeps its own output port (no reorder buffer).
`testbenches/TB_sienna_multi.sv` streams 48 distinct sets and checks every one. Measured with 16
lanes per copy: tanh 307.5 cycles per set with one copy, 149.0 with two, 73.4 with four; SELU 264.0
and 66.0. Every copy uses the collapse-k mesh (`COLLAPSE_K=1` is sienna_multi's default): at N=16
with 32 lanes, four copies give 62.6 (tanh) and 55.2 (SELU) cycles per set with or without it, so
it saves three quarters of the mesh PEs for 11 cycles of latency. At N=32 with 64 lanes and four
tail contexts, four copies give 79.0 (tanh) and 71.8 (SELU) cycles per set, 829 and 913 FLOP per
cycle, about 790 and 870 GFLOPS at the assumed 950 MHz; all 48 sets pass. The shared host port costs 17 cycles per set, so it becomes the limit only past about 15
copies (estimate). Under Verilator a one-element fixed array of queues lost its updates, so the TB
keys its per-copy queues by int.

### Collapse-k

`COLLAPSE_K=1` in SystolicMesh gives each output tile one SyncArray of depth N, so N^2 PEs instead
of N^3/T and no reduce. At N=16, T=4: 256 PEs instead of 1024, mesh stage 87 cycles instead of 77,
same steady state (activation-bound). Mesh 68/68, SIENNA 11/11 and back-to-back 12 clean with it
selected. It is off by default.

## Mesh interface after phase 2

| Port | Meaning |
|---|---|
| `input_ready_o` | a staging bank is free; host may write a set and pulse start. Writes while low are dropped |
| `start_matrix_mult_i` | pulse: the set just written is complete. Queued, not launched; ignored while `input_ready_o` is low |
| `collection_complete_o` | the oldest unreleased result is readable. A bank-full level, cleared by release |
| `result_release_i` | pulse after the consumer's last read. `sienna_top` sends it when its read stream ends |

The mesh launches a set when a staging bank is full and a result bank is free. A
queued start costs one cycle: 204 cycles per set at N=16, T=4, instead of 203.
A consumer that never releases stalls the mesh after two results.

**REQUIRED BACKGROUND:** read the `sienna-rtl` skill first. This one assumes its
pipeline description, its Verilator traps and its tolerance-not-bit-exactness
pass criterion.

## The one rule

**No completion or progress flag may be a sticky level cleared only by `rstn_i`.**

Completion is a one-cycle pulse, or a level qualified by both its owning FSM's
terminal state and the absence of a new start. Every terminal state needs an
exit transition driven by a re-arm signal.

Every back-to-back bug found in this design so far is a violation of that one
rule. When a second set misbehaves, look for a flag that is still high from the
first set before looking anywhere else.

## Confirmed defects (fixed in phase 1, known-issues #15 has the commits)

Measured on N=16, TILE=4, tanh, via `make verilator EXTRA_FLAGS=-DBACK_TO_BACK`.

| Site | Defect | Symptom it produces |
|---|---|---|
| `SystolicMesh.sv:40,59` | `ptr_A`/`ptr_B` never rewound; 9-bit counter wraps 512 to 0 at the second 256-word load | `queue_empty_o` falsely reads 1, `sienna_top.sv:445` rejects the start pulse, pipeline sits in IDLE until timeout |
| `OutputSram.sv:90` | `COMPLETE` is terminal - no exit transition exists at all | `tile_col_done` stuck high, mesh `WAIT_TILES` falls through instantly |
| `AccumulationUnit.sv:107` | `done_o = (r_curr == RDONE)`, parks until its next `start_i` | `all_reducers_done` stuck high, `WAIT_REDUCE` falls through |
| `Row`/`ColumnInputQueue.sv` | `pe_data_count`, `read_addr` cleared only under `rstn_i` | on a second `start_i`, `pe_data_count[i] < N` never fires, read pointers never advance |
| `SystolicMesh.sv:189` | `collection_complete_o = all_reducers_done`, ungated | `sienna_top` sees "complete" before the mesh has run |
| `PEMesh.sv` done logic | `last_element_seen` and `done_o` are one-shot, cleared only by reset | the last-element detect never fires again, `done_o` never falls, `drain_pending` never sees another rising edge, so the drain wave never runs |

Combined effect: the second matmul traverses the whole mesh FSM in ~23 cycles
without computing, and `sienna_top` reads the previous set's output SRAM.

**`ctrl_reset_all` already exists and already means "re-arm for a new matmul".**
It is already wired to the input queues' `write_reset_i` and was simply never
routed to the output side. Most of the fix is plumbing it the rest of the way.

## Two traps found while fixing this

**A re-arm fed combinationally into a status output closes a loop.** Driving
`OutputSram.rearm_i` straight from `ctrl_reset_all` produced a Verilator `UNOPTFLAT`
error: `ctrl_reset_all` -> `rearm_i` -> `collection_complete_o` (combinational) ->
`tile_col_done` -> `all_tiles_collected` -> the FSM that produces `ctrl_reset_all`.
Register the re-arm pulse. `RESET_SEQ` is followed by `BROADCAST`, so a one-cycle-late
clear still lands long before `WAIT_TILES` samples anything. Expect the same problem for
every re-arm added in later phases.

**A sticky completion level corrupts the measurement, not just the logic.**
`matrix_mult_complete_o` is held through `DONE`, so it is still high from the previous
set when start is pulsed. `wait(complete)` returned immediately and the TB read results
before the new matmul ran; the cycle counter stopped on the same stale level and reported
**0 cycles** for every set after the first. Wait for the flag to fall first, and stop
counters on the completion *edge*. A back-to-back number of 0, or one identical to a
serial baseline, is the symptom.

## Architecture

Three stage controllers. Dropout gets no controller of its own - it is a
per-element transform with no buffering (combinational bypass in inference), so
it folds into the pooling stage's output path.

| Stage | Owns | Consumes | Produces |
|---|---|---|---|
| `MESH` | SystolicMesh | input staging bank | mesh-output bank |
| `GPNAE` | FIFO1 + activation lanes | mesh-output bank | activation bank |
| `POOL` | window dispatch + FIFO2 + Maxpool + dropout | activation bank | `final_result_o` |

Three ping-pong buffers, 2 banks each: `mem_A`/`mem_B` (input staging, inside
SystolicMesh), `MeshOutputSram`, and `gpnae_out_mem`. `FIFO1` and `FIFO2` stay
unbanked - they are intra-stage, not handoff points.

Double-buffering the *input staging* is what makes the credit contract clean.
Without it, "credits available" would not imply "safe to write `mem_A`" and the
host would need a second readiness signal on the side.

Interface is credit-based: a counter initialized to `MAX_SETS_IN_FLIGHT` (3),
decremented on accept, incremented when a set's last result leaves. Each set
carries a 2-bit `set_id` held by the stage controller processing it, not
attached to every data word.

Ordering is an invariant, not a mechanism: stages are in-order and each holds one
set, so results emerge in issue order by construction. No reorder buffer.

The mesh releases its input staging bank as soon as `BROADCAST` has copied
`mem_A`/`mem_B` into the per-tile queues, not when the multiply finishes. That is
the overlap win on the input side.

## Trap: `current_state == IDLE` stops being a clear point

`sienna_top` clears per-set state at six places keyed on the global FSM being in
`IDLE` (lines 181, 375, 481, 520, 712, 736). With three independent controllers
there is no global `IDLE`. Each becomes a clear on *that stage accepting a set*.
Missing one reintroduces the stale-flag bug at the top level.

## Phasing

| Phase | Green when |
|---|---|
| 1 Mesh re-arm | `TB_SystolicMesh` K sets, no reset between, all pass — **done** |
| 2 Mesh streaming | mesh accepts set i+1 during set i's broadcast — **done**: `TB_SystolicMesh` streams K sets, host and consumer both overlap the mesh |
| 3 Top overlap | `TB_sienna_top` K sets + overlap assertion — **done** |
| 4 Full | `make regression` green, all 5 tests x K sets — **done** |

Submodule changes land first, per repo convention.

## Testing

`TB_SystolicMesh` already has K-set plumbing (`matrixA_<k>.mem`), but
`execute_test_set()` calls `apply_reset()` before **every** set - which is exactly
why these bugs were invisible at IP level. Phase 1's test change is a back-to-back
mode that resets once at time zero and never again.

Three checks matter beyond element compare:

- **Overlap actually happened** - assert some cycle has two stages busy on
  different `set_id`s. Without it a correct-but-serial implementation passes
  completely green and you get no signal the feature works. Most important
  new assertion.
- **A stage never completes without having been busy since its last accept.**
  Catches the stale-flag class directly; would have failed on the first
  back-to-back run.
- Results leave in issue order.

Per-set data must be random and distinct, never identity - identity masks
failures, and identical data cannot distinguish a correct second matmul from a
reused first one. That distinction is what exposed the stale-output bug: feeding
a different second matrix produced byte-identical results.

`NUM_SETS` must come from `regression.py`'s `write_sv_package()`.
`test_config_pkg.sv` is generated and hand edits survive until the next run.

## Known hazard, not solved

The dropout LFSR advances per valid beat and is shared across sets. Inference
bypasses it combinationally so regression stays deterministic. If training mode
is ever verified, overlapped sets share one LFSR stream and a per-set software
model will not match unless the LFSR is per-set seeded. No test exercises this
today.

## Traps found in phase 2

**TB inputs set after a rising edge are taken at that same edge under Verilator.**
A monitor that reads `start_mult` combinationally sees a start the mesh has already
consumed. Count launches from registered mesh state (`current_state == RESET_SEQ`).

**The mesh stimulus directory holds whatever the last regression test wrote.** After a
regression it ends on conv sets whose A files are 64 words, mixed with leftover 256-word
matmul sets. A streamed run then reads stale staging rows and fails rows 4-15. Regenerate
before a standalone run:
`python3 -c "import matmul_tests as m; m.gen_mm_random('testbenches/stimulus', 16)"`.
The conv tests also load fewer than N*N words and rely on the rest of the staging bank
being zero, which holds only because each regression test is a fresh simulation.

## What the tests prove, and what they do not

Added 2026-09-23. Every build now compiles with `--assert`. Handshake assertions sit at the
end of `SystolicMesh.sv` and `sienna_top.sv` under `ifndef SYNTHESIS`, and
`-DASSERT_SELFTEST` adds one that always fails, to prove they are live. Directed tests:

- **Mesh staging overrun** (`staging_overrun_test`): no releases, so both result banks and
  both staging banks fill. Then 256 writes and a start with `input_ready_o` low must be
  ignored. The four queued sets must come out intact, and a later proper load must land cleanly.
- **Top credit overrun** (second stream pass): with no credit free, the host loads 16 words
  and pulses start. The mesh write pointer must stay at 16, the load resumes, and set 3 must
  still be correct.
- **Reset mid-stream** (`reset_mid_stream`): reset with two sets in flight. The pipeline must
  go idle with all credits free and no output for 2000 cycles, and a clean 4-set stream must pass.
- **Output port**: both captures read `final_result_o` qualified by the new `result_valid_o`,
  never the internal dropout signals.

Back-pressure: the mesh stalls on full result banks only in the credit-overrun pass
(159-217 cycles). **The activation stage never stalls on full activation banks, in any
test.** Pooling takes far less time than activation and has no output back-pressure, so
that path cannot be reached in this configuration. Its handshake is covered only by the
`a_act_bank_free` and `a_act_bank_full` assertions.

Since 2026-09-23 also covered: GPNAE's own assertions (fixed, known-issues #16), varied seeds
(`SIENNA_SEED` shifts every stimulus seed in the mesh and SIENNA generators, `--seed` for GPNAE;
unset reproduces the fixed stimulus), dropout training mode (`_train` tests, known-issues #18),
and activation inputs far outside the fitted range (`matmul_large_*`, known-issues #17). The
back-to-back pass now uses a distinct random second matrix with its own golden.

Still not covered: functional coverage, formal, four-state (VCS) simulation, any N other than
16 at the top level, and synthesis after the buffers doubled.

## Traps found in phase 3

**`checker` is a SystemVerilog keyword.** A named block `begin : checker` is a syntax error.

**A pass that starts in the previous pass's completion cycle records a phantom set
boundary.** Every slice then shifts by one set. Wait for `pipeline_complete_o` to fall
before arming a capture.

**Set ids continue across passes.** The single-set and `BACK_TO_BACK` passes consume ids
before the stream starts, so the stream's first id is 1 or 2, not 0.
