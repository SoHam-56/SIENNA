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

**Update 2026-09-25.** The mesh itself is now pipelined (see `sienna-rtl` architecture), the
credit count is the `SETS_IN_FLIGHT` parameter (default 8, ids `$clog2(SETS_IN_FLIGHT+1)` bits),
ReLU and linear sets bypass the lanes, and 1x1 pooling with no padding bypasses FIFO2 and
Maxpool (`POOL_BYPASS`). Every regression test streams `SETS_IN_FLIGHT + 2` sets. Measured
steady state for ReLU sets through TB_sienna_top: 19-21 cycles per set at N=16 (was 88) and 35 at N=32
(was 104), set by that testbench's host (N rows plus a start cycle and two handshake cycles). With
TB_sienna_model's streaming host (start with the last row) a GEMM runs at 17.4 cycles per set at N=16
and exactly 32.0 at N=32, the mesh's own limit of N cycles per set. `TB_sienna_model +trace` logs every
stage event and wait, which is how to find the next limit. Three
credits had capped the rate at latency / 3, about 50 cycles per set, once the stages were fast.
Plans: `implementation-plan.md`, `phase2-plan.md`, `phase3-plan.md`.

## Top-level interface after phase 3 (history: before the credit links)

**Superseded on 2026-10-07 by the credit links; see "Module boundaries as built" below for the current interface.**
`pipeline_ready_o`, `start_pipeline_i` and every port "sampled with the start" are gone: the host puts a set on the L0 link `host`, whose data is the set's sideband (`set_side_t` in `src/sienna_set_side.svh`, the same fields as the old ports).
`pipeline_complete_o` and `done_set_id_o` are kept; outputs leave on the L9 links `out[NUM_LANES]` instead of `final_result_o` / `result_valid_o`.
The table below is the interface as it was from phase 3 until then.

| Port | Meaning |
|---|---|
| `pipeline_ready_o` | a credit and a mesh staging bank are free; host may load a set and pulse start |
| `start_pipeline_i` | pulse after loading; ignored while `pipeline_ready_o` is low |
| `pipeline_complete_o` | **one-cycle pulse** when a set's last output leaves dropout (was a held level) |
| `done_set_id_o` | 2-bit id of that set: accepted starts, counted mod 4 |
| `accumulate_i` | sampled with the start: 1 makes the set a partial sum (added into `acc_mem`, no output, still completes in order); the next set with it low is added to the sum and activated |
| `pack_shift_i` | sampled with the start: a packed set of blocks b = N >> pack_shift_i wide (sienna-packing); 0 is unpacked and behaves exactly as before. Refused with accumulate or when it continues a partial sum, pooling, collapse-k 0, N not dividing the lanes, or b < 2 |
| `pack_map_i` | sampled with the start: the table entry (0..7) of each column block; entry 0 is the per-set ports below |
| `pack_act_i`, int8 `pack_zp_i` `pack_min_i` `pack_max_i` `pack_mx_i` `pack_shx_i` `pack_mout_i` `pack_shout_i` `pack_zout_i` | sampled with the start: table entries 1..7 (activation; requantize zero point and clamp; GPNAE words), held per set id like `activation_function_i` |

Measured by `perf_analysis.py`, now `regression.py --action perf` (12 streamed sets, N=16, T=4, farm, 2026-09-24) on the current
defaults: 32 lanes, one-row host writes (`HOST_WORDS = N`), synchronous `SystolicArray` mesh tiles. GFLOPS count the
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
| synchronous tiles | mesh 129 -> 77 (tile phase 94 -> 42) | unchanged; single-set latency -52 |
| four tail contexts (GPNAE `e68882c`) | tail elements in a lane run together | normal data unchanged; large-value 1510-2078 -> 530-591 |

The mesh stage is 77 cycles per set (broadcast 4, tiles 42, reduce 29). It is no longer on the
critical path; activation is. gpnae_tail now works on four out-of-range elements at once
(`TAIL_CONTEXTS`), sharing one multiplier and one adder, bit-identical to running them in turn. `python3 regression.py --action perf`
(`make perf-analysis`) regenerates `testbenches/results/perf/pipeline_performance_report.log`; `--lanes`, `--n` and
`--tile-size` pick another geometry.

### Larger N (measured 2026-09-24)

Full pipeline at N=32, T=4, one-row host writes, synchronous tiles: all 8 matmul tests pass (conv needs N to
be a perfect square in the generator). Mesh stage 82 cycles, host load 33, so activation sets the rate:

| Lanes | Tail | SELU steady | tanh steady | GFLOPS @950 (SELU / tanh) | Lanes busy |
|---|---|---|---|---|---|
| 32 | one at a time | 1036 | 1561 | 60.1 / 39.9 | 51-58% |
| 64 | one at a time | 772 | 1122 | 80.7 / 55.5 | 35-41% |
| 64 | four contexts | 288 | 380 | 216.6 / 163.8 | 84-87% |

N=32 sums 32 products per output, so far more activation inputs land past the fitted range; with
the tail run one element at a time, lanes sat idle while one lane worked through its tail elements. Mesh only,
set 0 at N=32, T=4: 136 cycles with the old handshake tile, 84 with the synchronous tile (8192 PEs each), 105 with
collapse-k (1024 PEs). At N=64, T=8 collapse-k takes 197 cycles with 4096 PEs, about 32% PE
utilisation, against 840 cycles and 32768 PEs for the old serial-reduce mesh.

### Accumulate mode (2026-09-24)

For products deeper than N the host sends the depth in passes: every pass but the last with
`accumulate_i` high. The activation stage sums partial mesh results in `acc_mem` (NUM_LANES fp32
adders, the first partial copied exactly) in states G_ACC_RD/G_ACC_WAIT, then fills the lanes from
the sum (G_AFEED) for the final pass. A partial set takes an activation bank marked null, which
pooling passes in one cycle, so completions stay in issue order and credits return. A set with no
pending sum skips all of this, so normal behaviour is unchanged. Tests: `matmul_accum2_tanh`,
`matmul_accum3_selu`, `matmul_accum2_tanh_train` (ACCUM_PASSES groups the streamed sets).

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

`COLLAPSE_K=1` in SystolicMesh gives each output tile one SystolicArray of depth N, so N^2 PEs instead
of N^3/T and no reduce. At N=16, T=4: 256 PEs instead of 1024, mesh stage 87 cycles instead of 77,
same steady state (activation-bound). Mesh 68/68, SIENNA 11/11 and back-to-back 12 clean with it
selected. It is the default since 2026-09-24 (sienna_top, sienna_multi and the mesh).

## Mesh interface after phase 2 (history: before the credit links)

**Superseded on 2026-10-07 by the credit links; see "Module boundaries as built" below.**
`input_ready_o` / `start_matrix_mult_i` became the L1 staging link (a put per set, the pack shift and the other per-set bits in its data), `collection_complete_o` / `result_release_i` / the wide read became the L3 result push, and `wide_read_packed_i` is gone (the packed flag travels with each sum).
The table below is the mesh interface as it was from phase 2 until then.

| Port | Meaning |
|---|---|
| `input_ready_o` | a staging bank is free; host may write a set and pulse start. Writes while low are dropped |
| `start_matrix_mult_i` | pulse: the set just written is complete. Queued, not launched; ignored while `input_ready_o` is low |
| `collection_complete_o` | the oldest unreleased result is readable. A bank-full level, cleared by release |
| `result_release_i` | pulse after the consumer's last read. `sienna_top` sends it when its read stream ends |
| `pack_shift_i` | sampled with the start: the set's pack shift, held per staging and operand bank and passed east with A; each PE skips products outside its column block |
| `wide_read_packed_i` | the wide read takes the oldest result column-wise (word k is column k % N), so a lane holds one column block |

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

The queue, PEMesh and OutputSram rows below refer to the handshake tile removed on 2026-09-24 (git tag `legacy_tile_v1`).

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
| `POOL` | window dispatch + FIFO2 + Maxpool + dropout | activation bank | the L9 output links (`out[NUM_LANES]`) |

Three ping-pong buffers, 2 banks each: `mem_A`/`mem_B` (input staging, inside
SystolicMesh), `MeshOutputSram`, and `gpnae_out_mem`. `FIFO1` and `FIFO2` stay
unbanked - they are intra-stage, not handoff points.

Double-buffering the *input staging* is what makes the credit contract clean.
Without it, "credits available" would not imply "safe to write `mem_A`" and the
host would need a second readiness signal on the side.

Interface is credit-based: entry admission allows `SETS_IN_FLIGHT` sets (default 15 in
sienna_top) in flight, and since the credit links (2026-10-07) every boundary
inside is a credit link too (table below). Each set carries a `set_id` of `$clog2(SETS_IN_FLIGHT+1)` bits held by the
stage controller processing it, not attached to every data word.

Ordering is an invariant, not a mechanism: stages are in-order and each holds one
set, so results emerge in issue order by construction. No reorder buffer.

### Module boundaries as built (credit links, SIENNA branch `credits` at 3d66c57, 2026-10-07)

Every module boundary is now one credit link (`credit_link_if`, from ArithmeticLibrary `Common/src`); no boundary uses ready/valid or valid-only handshakes any more.
What still travels without its own credit: the row, cache-write and bias buses and `sienna_layer`'s int8 epilogue words, which ride beside a put that holds the credit, and `requant_lanes`' fixed-latency valid pipe inside the activation stage, which cannot stall.
The consumer owns the buffer and grants one credit per free slot (after reset its slot count, sent as ordinary credits); the producer puts only while it holds a credit.
The full contract, the deviations from the design and the measured cost are in the `sienna-credits` skill, section "As built".
Entry admission is still `SETS_IN_FLIGHT` (15), but it now lives inside L0: the host gets a staging credit only while fewer than SETS_IN_FLIGHT sets are in flight or granted.
`pipeline_complete_o` is still a one-cycle pulse per set; it no longer returns a host credit by itself.

| Link | Boundary | Protocol as built | Slots (consumer) | Back-pressure |
|---|---|---|---|---|
| L0 | host -> `sienna_top` | one put per set, `data` = the set's sideband (`set_side_t`, `src/sienna_set_side.svh`), rows on the write buses before the put | 2 staging banks, withheld while the entry is full | yes |
| L1 | `sienna_top` -> SystolicMesh | one put per set (the start), `data` = `{wc_last, tile, cached, pack_shift, bias_valid, accumulate}`; credit back on `bcast_release` | 2 staging banks | yes |
| L2 | weight-cache writer -> SystolicMesh (x2 regions) | one put opens a fill; the region's credit returns when the fill's last set (marked `wc_last`) is broadcast | 1 per region | yes |
| L3 | SystolicMesh -> activation stage | the mesh pushes a result as PER_LANE wide beats `{packed, last, first, one word per lane}`; the stage grants a set's PER_LANE beats at once from G_IDLE | PER_LANE beats (8 at N 16, 32 lanes) | yes |
| L4 | activation stage -> GPNAE lane (per lane) | one word per put, `last` rides the set's final put | 32 (the lane's `lane_fifo`) | yes |
| L5 | GPNAE lane -> collector (per lane) | one result per put; a lane starts a barrel group of K only with K credits | 16 (`LANE_OUT_SLOTS`), credit the cycle after the bank write | yes |
| L6 | activation stage -> pooling stage (inside `sienna_top`) | put on `bank_done`, `data` = the bank; credit on `p_release` | 2 activation banks | yes |
| L7 | dispatcher -> FIFO2 (`fwft`, per lane) | one word per put; overwrite-on-full removed | 16 | yes |
| L8 | FIFO2 -> `Maxpool_2D` -> `dropout` (per lane) | maxpool grants a window's 4 element credits at once, only with an output credit reserved; dropout passes the output credits through | one window per reserved output credit, up to 2 ahead | yes |
| L9 | `sienna_top` -> downstream (per lane) | one word per put | advertised by the downstream consumer (TBs: `+out_slots`, default 64) | yes: a consumer can now stall SIENNA |
| L10 | host -> `sienna_layer` A and W row streams | one row per put; a set's N A rows (and N B rows if uncached) granted at once per L0 credit held | N rows per set, one set granted ahead | yes |

`sienna_multi` exposes L0 (`host[COPIES]`) and L9 (`out[COPIES*NUM_LANES]`) per copy and fans L2 out to every copy (a region credit goes back once every copy has returned its own).
`LINK_STAGES` (Makefile, TB_sienna_top parameter) puts a `credit_reg` on L0, L1, L3 and L9 only; the row buses, cache write bus and bias are delayed 2*LINK_STAGES with the put.

The host is hardware (testbench in simulation; a DMA or `sienna_layer`'s credit streams in a system), never software: the driver (`model_runner.py`'s role) works at layer granularity.

Before the credit links (main, tag job_packing_v5) only the entry was credited: host -> mesh was `input_ready_o` + `start_matrix_mult_i`, mesh -> activation was `collection_complete_o` + wide read + `result_release_i`, the lanes reported an unread `full_o`, FIFO2 overwrote when full behind an almost-full threshold, and maxpool, dropout and the output port were valid-only.

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

`NUM_SETS` must come from `write_sv_package()` (model_runner.py, called by regression.py).
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
`-DASSERT_SELFTEST` adds one that always fails, to prove they are live. Directed tests (as written 2026-09-23; the
bullet after them gives what changed with the credit links on 2026-10-07):

- **Mesh staging overrun** (`staging_overrun_test`): no releases, so both result banks and
  both staging banks fill. Then 256 writes and a start with `input_ready_o` low must be
  ignored. The four queued sets must come out intact, and a later proper load must land cleanly.
- **Top credit overrun** (second stream pass): with no credit free, the host loads 16 words
  and pulses start. The mesh write pointer must stay at 16, the load resumes, and set 3 must
  still be correct.
- **Reset mid-stream** (`reset_mid_stream`): reset with two sets in flight. The pipeline must
  go idle with all credits free and no output for 2000 cycles, and a clean 4-set stream must pass.
- **Output port** (history): both captures read `final_result_o` qualified by the new `result_valid_o`,
  never the internal dropout signals. Since 2026-10-07 the TB reads the L9 links instead (next bullet).
- Since the credit links (2026-10-07) the two overrun tests are no-credit tests: the mesh's became "producer waits" (the host stages
  until no credit returns, then everything drains intact) and the top's became "entry full" (the host holds no credit while the entry
  is full); the output is read from the L9 links, and a 500-cycle L9 hold checks that nothing is lost, duplicated or reordered.

Back-pressure before the credit links (history, 2026-09-23): the mesh stalled on full result banks only in the
credit-overrun pass (159-217 cycles), and the activation stage never stalled on full activation banks, because
pooling was faster than activation and the output could not be stalled.

Back-pressure since the credit links (2026-10-07): a downstream consumer stalls SIENNA through L9.
- TB_sienna_top's consumer advertises `+out_slots` per lane (default 64) and withholds credit returns at `+out_stall_pct`.
- After a second reset it advertises one L9 slot per lane and withholds every credit for 500 cycles mid-set: no word leaves in those cycles, and afterwards the words and set boundaries equal the plain pass and every link count is home (Task 5b, every test in all three formats).
- In the credits-3 tests that hold reaches the host: the entry is full for 485 to 500 of the 500 cycles, so the stalled output backs sets up all the way to the entry.
- With 2 slots and 50% stalls (`cr5bxg_o2s50_int8_T4`) the producer is starved of L9 credits in every pass and the words still equal main's.
- `a_act_bank_free` and `a_act_bank_full` are still in `sienna_top`; the L6 credit link and its checker now carry the activation-to-pooling handshake.

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
