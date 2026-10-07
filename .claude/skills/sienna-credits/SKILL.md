---
name: sienna-credits
description: Use when designing, building or verifying credit-based interfaces at SIENNA's module boundaries - the credit_link_if SystemVerilog interface, the producer credit counter and the protocol checker in ArithmeticLibrary, credit links between the host interface, SystolicMesh, the activation stage, the GPNAE lanes, pooling, maxpool, dropout and SIENNA's output, output back-pressure, or the before/after performance comparison of the credit RTL.
---

# SIENNA credit links

**Status: built, 2026-10-07, on branch `credits` in all four repos (not merged, not pushed).** Approach A, approved by
Soham 2026-10-07: one credit link at every module boundary, today's buffers kept where they are, ports as SV
interfaces. The sections up to "As built" are the approved design; "As built" at the end records what was built,
where it differs and what it costs. The links as built are also tabled in the `sienna-back-to-back` skill,
"Module boundaries as built".

## Why

Soham, 2026-10-07: every module boundary should be credit based, for all of: timing closure (registers allowed
between blocks placed apart, 950 MHz at large N), reusable IP (one contract per repo, one checker), and output
back-pressure (a downstream consumer can stall SIENNA). Today only the entry is credited, and three places are
correct only by sizing: `sienna_top` never reads a GPNAE lane's `full_o`; the pooling FIFOs (`fwft`) overwrite when
full, guarded by an almost-full threshold (12 of 16); lane outputs, `Maxpool_2D`, `dropout` and the output port
have no ready at all.

## The credit link

- The **consumer** owns the buffer. After reset it grants one credit per free slot; afterwards it returns one credit
  for every slot it frees.
- The **producer** may `put` one item only while it holds a credit; the credit goes back to the consumer with the
  item. An item is one data word (beat link) or one whole set (set link).
- Credits travel only consumer to producer; there is no piggybacking (every SIENNA link carries data one way).
- Either direction may carry any number of register stages. Full rate needs `slots >= round trip` (cycles from a
  put to the credit for that slot being usable again); fewer slots throttle the link, never break it.
- Several slots may free in one cycle: `credit` is a count `CRW` bits wide (1 for single-slot returns).
- Reset: producer count 0, then the consumer's advertisement (its slot count) arrives as ordinary credit returns.

```systemverilog
// One credit link: the consumer grants credits (one per free slot), the producer spends one per put.
interface credit_link_if #(parameter int DATA_W = 32, parameter int CRW = 1);
  logic              put;     // producer: one item this cycle
  logic [DATA_W-1:0] data;    // producer: the item (beat) or the set's sideband (set link)
  logic [CRW-1:0]    credit;  // consumer: slots freed this cycle; after reset, the slots it advertises
  modport producer (output put, data, input  credit);
  modport consumer (input  put, data, output credit);
  modport monitor  (input  put, data, credit);
endinterface
```

Shared pieces, in ArithmeticLibrary `Common/src` beside `sienna_fmt_pkg.sv` (both submodules carry AriL, so every
repo gets them; AriL pushes first):

- `credit_link_if.sv` (above).
- `credit_counter.sv`: the producer's count, `cnt <= cnt + credit - put`, `has_credit_o = cnt != 0`; parameter
  `MAX` (the most slots any consumer may advertise); `a_no_underflow` (put only with a credit), `a_no_overflow`
  (count never above MAX).
- `credit_link_checker.sv`: bound to the `monitor` modport with the consumer's `SLOTS`; tracks granted minus used
  and asserts: no put without a credit, outstanding credits never above SLOTS, every slot credited back once the link
  drains (a `drained_i` input from the testbench). Under `ifndef SYNTHESIS`, like every assertion in the repos.
- `credit_reg.sv`: N register stages on a link (data/put forward, credit back), `STAGES` default 0, to prove
  latency tolerance.

## Links (approach A)

| # | Link | Producer → consumer | Unit | Consumer's slots | Replaces |
|---|---|---|---|---|---|
| L0 | host | host interface → `sienna_top` | set | 2 staging banks, withheld while `SETS_IN_FLIGHT` sets are in flight | `pipeline_ready_o` / `start_pipeline_i` |
| L1 | staging | `sienna_top` entry → SystolicMesh | set | 2 staging banks (credit on `bcast_release`) | `input_ready_o` / `start_matrix_mult_i` |
| L2 | weight cache | host interface → SystolicMesh | region | 1 per region, 2 links | `wc_region_busy_o` |
| L3 | results | SystolicMesh → activation stage | wide beat | `PER_LANE` beats per set the stage can take (lane FIFO space and a free activation bank) | `collection_complete_o`, wide read, `result_release_i` |
| L4 | lane in | activation stage → GPNAE lane (×`NUM_LANES`) | word | 32 (lane FIFO) | `wr_en_i`, `last_i`, unread `full_o` |
| L5 | lane out | GPNAE lane → activation collector (×lanes) | word | 16 (one barrel group; the collector writes the bank at once) | `done_o` / `final_result_o` |
| L6 | activation bank | activation stage → pooling stage | set | 2 activation banks (credit on `p_release`) | `act_full` / `g_accept` / `p_accept` |
| L7 | pool in | dispatcher → pooling lane FIFO (×lanes) | word | 16 (`fwft` FIFO2) | `disp_can_write` threshold, overwrite on full |
| L8 | pool lane | FIFO2 → `Maxpool_2D` → `dropout` | word | a window enters maxpool only with an L9 credit reserved for its result | `start`/`done`, `valid_in` → `valid_out` |
| L9 | output | `sienna_top` → downstream (×lanes) | word | advertised by the downstream consumer | `final_result_o` / `result_valid_o` |
| L10 | layer streams | host → `sienna_layer` A and W rows | row | N rows (one staging bank's worth) granted per L0 credit | `a_ready_o` / `w_ready_o` |

Notes per link:

- **Data stays in today's buffers.** Staging banks, result banks, lane FIFOs, activation banks and pooling FIFOs keep
  their sizes; credits count their free slots. New storage is only where nothing can stall today: a 16-word
  collector slot per lane (L5) if a lane may not write the bank at once, and nothing else.
- **L0/L1:** a staging credit authorises writing one set's rows and its start; rows stay on today's write buses (the
  bank is reserved by the credit, so rows need no per-row credit). The set's sideband (partial, bias valid, pack,
  cached tile, activation, int8 parameters, training, seed) travels as the L1/L0 `data`.
- **L3 becomes push:** the mesh streams a finished result as `PER_LANE` wide beats (first/last and set id in
  `data`), from its 4 result banks (kept: they let reducers finish while the stage is busy); it frees a result bank
  after the last beat is put. The 1-cycle wide read latency moves inside the mesh.
- **L4:** `last` becomes a data bit on the final put of a set; the lane advertises 32.
- **L5/L8/L9:** a lane starts a barrel group of K only with K output credits, and a pooling window enters maxpool
  only with an output credit, so nothing in flight (barrel MAC, dropout's training multiplier) ever needs to stall
  mid-pipeline. Lanes and maxpool stall at their inputs only.
- **L9 is the new back-pressure:** testbenches act as the downstream consumer and advertise slots; the default
  (`OUT_SLOTS` large, credits returned every cycle) reproduces today's never-stalling output.
- `sienna_multi` routes L0 per copy as today; `TB_sienna_multi` follows.
- `gpnae.sv` (the published Taylor lane) is not changed; `gpnae_poly` and `gpnae_poly_int8` get L4/L5.

## Work (branch `credits` in AriL, GPNAE, SystolicMesh, SIENNA; innermost first)

0. **Early checks, before converting anything (throwaway, outside the repos):** a producer/consumer pair over
   `credit_link_if`, an array of 128 links in a generate loop, `credit_reg` with 0 and 2 stages, the checker; built
   and simulated with our Verilator 5.035 on the farm, plus `--lint-only -DSYNTHESIS -Wall`. Pass: compiles, every
   put matched by a credit, the checker fires on a deliberate violation.
1. AriL: the four shared files and a unit bench (counter, checker firing on purpose, `credit_reg` 0..3 stages).
2. GPNAE: L4/L5 in `gpnae_poly` and `gpnae_poly_int8`; `TB_gpnae_poly` drives links with the checker bound.
3. SystolicMesh: L1, L2, L3; `TB_SystolicMesh` drives links (start-while-not-ready and overrun tests become
   no-credit tests).
4. SIENNA: `sienna_top` (L0, L3-L9), `sienna_layer` (L10), `sienna_multi`, `Maxpool_2D`, `dropout`, `fwft`; every
   testbench; `regression.py` / `model_runner.py` only where they read changed signals.

## Verification and the before/after comparison

- Fast regressions first, each against today's `main`: GPNAE int8 (under a minute) and fp32; SystolicMesh N = 16
  all tiles in fp32 / bf16 / int8; SIENNA pipeline N = 16, T = 4 in int8 and bf16 (then fp32), `make pack`,
  `make tflite`; then `make regression` in all three formats.
- **Correctness:** every output identical to today's (words compared with `int8_cmp_reg.py`; mesh and GPNAE
  bit-exact as today). Cycles may differ; they are reported, not required equal.
- **Performance, before vs after:** `make perf-analysis` at N = 16, T = 4 in fp32, bf16, int8 (baselines exist:
  `tl3b_perf`, `mr_perf_N16_bf16`, `mr_perf_N16_int8`), and `make model` ResNet-8 at N = 16. Report latency, cycles
  per set and GFLOPS / TOPS per configuration, side by side, with `credit_reg` STAGES = 0 and again with 1 stage on
  every link (the latency-tolerance proof). Expected: equal throughput at STAGES = 0; a few cycles of latency per
  registered link.
- **Back-pressure:** a new test that withholds L9 credits mid-set and checks nothing is lost, duplicated or
  reordered, and that the pipeline resumes.
- N = 64 only after the change is checked in (one report sweep).

## As built (2026-10-07)

### Commits (branch `credits`, not pushed, not merged)

- ArithmeticLibrary 5255999: `credit_link_if`, `credit_counter` (with `count_o`), `credit_reg`, `credit_link_checker`, `TB_credit_link` (`make credit`); 07c6430 (final review fixes: the counter's after-update check, elaboration and width checks, `credit_reg` data not reset, TB MODES 5 to 8).
- GPNAE 4896cd7 (AriL bump), bd4d919 (L4/L5 in `gpnae_poly` and `gpnae_poly_int8`), bbd96b4 (`lane_fifo`), 8334d4d (`lane_link`, the shared credit front end), b1ef12d (AriL bump to 07c6430).
- SystolicMesh 1e7f484 (AriL bump), 0f18888 (L1, L2, L3), 8076166 (review fixes), 3fd3ef6 (AriL bump to 07c6430).
- SIENNA 8adf841 (L0, L1, L3, L6, L2 passed through; mesh bump), 2615bb2 (fix), 8824c31 (L4, L5, L7, L8, L9; GPNAE bump), a575057 (TB fix), b7dec1a (L10 in `sienna_layer`, L0/L9 per copy in `sienna_multi`), 3d66c57 (fix), dcf03fa (the unused `l9_sink.sv` removed); final review fixes 724e28f (GPNAE and SystolicMesh bumps), 757a58b (`sienna_top`: `SRAM_DEPTH` check, write-bus data not reset), 3acdda0 (`LINK_STAGES`, `OUT_MAX`, `OUT_CRW` through `sienna_layer` and `sienna_multi`), 1d24509 (mid-stream resets in TB_model_run and TB_sienna_multi), e913a25 (TB_sienna_top's hold).
- `gpnae.sv` (the published Taylor lane) and the GPNAE math are unchanged; nothing existing in ArithmeticLibrary changed.

### The links as built

| # | Producer → consumer | Unit, data | Consumer's slots | CRW | Register stage with `LINK_STAGES` |
|---|---|---|---|---|---|
| L0 | host → `sienna_top` (`host`) | set; `data` = `set_side_t` (`src/sienna_set_side.svh`, 1488 bits at N 16) | 2 staging banks, withheld while `SETS_IN_FLIGHT` sets are in flight or granted | 1 | yes |
| L1 | `sienna_top` → SystolicMesh (`staging`) | set; `{wc_last, weight_tile, weight_cached, pack_shift, bias_valid, accumulate}` | 2 staging banks, credit on `bcast_release` | 1 | yes |
| L2 | `sienna_layer` / `sienna_multi`'s host → SystolicMesh (`wc_region[2]`) | region; a put opens a fill | 1 per region, back when the fill's last set is broadcast | 1 | no |
| L3 | SystolicMesh → activation stage (`result`) | wide beat; `{packed, last, first, one word per lane}` | PER_LANE beats (8 at N 16, 32 lanes), granted at once | `$clog2(PER_LANE+1)` | yes |
| L4 | activation stage → GPNAE lane (per lane) | word; `{last, word}` | 32 (`lane_fifo`) | 1 | no |
| L5 | GPNAE lane → collector (per lane) | word | 16 (`LANE_OUT_SLOTS`) | 1 | no |
| L6 | activation stage → pooling stage (inside `sienna_top`) | set; the bank | 2 activation banks, credit on `p_release` | 1 | no |
| L7 | dispatcher → FIFO2 (`fwft`, per lane) | word | 16 | 1 | no |
| L8 | FIFO2 → `Maxpool_2D` → `dropout` (per lane) | word | a window's 4 elements at once, only with an output credit reserved, up to 2 windows ahead | 3 into maxpool | no |
| L9 | `sienna_top` → downstream (`out[NUM_LANES]`) | word | the downstream consumer's advertisement (`OUT_MAX` 64) | 1 (`OUT_CRW`) | yes |
| L10 | host → `sienna_layer` (`a_rows`, `w_rows`) | row of N words | a set's N A rows (and N B rows if uncached) per L0 credit held, one set ahead; bias row 1, cache tile N | `$clog2(N+1)` | no |

- `sienna_multi` exposes `host[COPIES]` (L0 per copy) and `out[COPIES*NUM_LANES]` (L9 per copy), and fans L2 out to every copy.
- Every link has a `credit_link_checker` in its testbench or inside the consumer, with `a_all_back` judged at each drain.
- Each new assertion and `$fatal` was shown firing once on purpose (task reports 1 to 6 and the final fix report name the runs); a few fired only on saved one-line mutants because Verilator 5.035 rejects `force` in a model that uses queue methods.

### Rules the RTL relies on that the design did not state

- `credit_counter`'s contract (Ruling 19, replacing Ruling 18): the count after the update, `cnt + credit - put`, never exceeds MAX, and `a_no_overflow` checks exactly that, as `credit_link_checker` does. A consumer may therefore return a slot's credit in the cycle of the put that fills it (TB_credit_link MODE 5: 5884 same-cycle puts at count MAX, 0 assertions; the old pre-put check fires on it). MAX < 1 and a CRW wider than the count stop elaboration; `credit_reg` checks its DATA_W and CRW against its links at time 0.
- A producer whose consumer may grant N credits in the cycle of the last put of a set needs counter MAX >= N (`sienna_layer`'s A rows: 1 + N - 1 after the update; N+1 under the old check); one that may get two cache tiles' grants needs MAX 2N (W rows). Stated on the ports, not checked inside.
- L10 puts must come from registered state: inside `sienna_layer` a put reaches `a_rows.credit` and `w_rows.credit` combinationally (loop-free while every producer puts from a registered count).
- Both ends of a link share one reset: an advertisement made while the other end is in reset is lost (`credit_link_if` header).
- L9 lanes drift apart by up to FIFO2's 4 windows + maxpool's 2 windows ahead + the L9 slots, so a consumer must not make one lane's credits wait on another lane's later windows (`sienna_top`'s L9 port).
- `SRAM_DEPTH` must equal N*N (`G_BAD_SRAM_DEPTH`): L3 grants SRAM_DEPTH/NUM_LANES beats per set while the mesh pushes N*N/NUM_LANES.
- Datapath flops are not reset (D-8): `credit_reg`'s data registers and the data of `sienna_top`'s 2*LINK_STAGES write-bus delay line; put, credit and the bus enables and resets are.
- `credit` is compared as `int` everywhere, never truncated to CRW: a `CRW'(2)` at `SETS_IN_FLIGHT` 1 read 0 and deadlocked (found in Task 5a review, fixed in 2615bb2).
- No credit leaves during reset: a `live` flop (one cycle after reset) gates every advertisement in the mesh, `sienna_top`, `sienna_layer` and `sienna_multi`.

### Where the build differs from the design above

- **L2 needs a `wc_last` bit in the L0/L1 sideband (Ruling 5).** With one slot per region the mesh can return "region free" only when it knows the fill's last set has been broadcast. Producer contract: one L2 put per fill, the fill's sets, the last one marked; `wc_last` low on uncached sets (`a_wc_last_cached`).
- **The L2 put only feeds checks in synthesis (Ruling 6).** The region's credit is driven by the marked set's broadcast.
- **L2's producer is the cache writer (Ruling 10):** `sienna_layer`'s weight loader (one counter per region), and in `sienna_multi` the host, whose put goes to every copy; a region credit goes back only when every copy has returned its own.
- **`sienna_multi`'s L2 contract is checked, not enforced:** a fill must reach at least COPIES sets, each copy's last one marked. `a_wc_fill_short`, `a_wc_after_last`, `a_put_on_turn` and, at drain, `a_wc_drained` name a host that breaks it.
- **The poly lanes use a new circular `lane_fifo` (Ruling 3).** InputFIFO writes the lowest free slot, so a credit-legal put while the lane pops reordered words. `lane_fifo` keeps InputFIFO's 32 entries and read latency; `gpnae.sv` still uses InputFIFO.
- **L3 carries no set id;** set ids stay in `sienna_top`. The activation stage grants a set's PER_LANE beats ahead of the result, from G_IDLE with an activation bank reserved, every lane holding PER_LANE L4 credits, and no partial set waiting.
- **`LINK_STAGES` stages L0, L1, L3 and L9 only (Ruling 15),** the links that cross a module or repo boundary with a register stage; `sienna_layer` and `sienna_multi` pass `LINK_STAGES`, `OUT_MAX` and `OUT_CRW` through to `sienna_top`. L4, L5, L7 and L8 sit inside a lane's tile, L6 inside `sienna_top`; L2 and L10 are not staged. The north/west row buses, the cache write bus and `bias_i` are delayed 2*LINK_STAGES with the put, so a set's rows never land after its put or in the previous bank. `pipeline_complete_o` and `done_set_id_o` are delayed by LINK_STAGES so they still arrive with the set's last word.
- **dropout passes its output credits through** (`in.credit = out.credit`): every beat leaves after a fixed delay, and maxpool (or the dispatcher with a 1x1 pool) reserves the L9 credit before a beat enters.
- **Entry admission stays inside L0:** a host credit is granted only while an undelegated staging credit exists and `sets_out + l0_out < SETS_IN_FLIGHT`.
- **`src/l9_sink.sv`** (a never-stalling L9 consumer used between Tasks 5b and 6) lost its last user in Task 6 and was removed with Soham's approval (SIENNA dcf03fa, with its Makefile and `synth/sienna_rtl.f` lines).
- **The int8 epilogue (bias, multiplier, shift) rides side buses beside the W bias row's put,** not the L10 data, so L10 is not latency-tolerant if it is ever staged.

### Performance and correctness (`/proj/work/spramanik/sienna_report/credits_compare.log`)

All at N 16, T 4, from the runs named in the comparison log; GFLOPS, GOPS and µs are derived at an assumed 950 MHz (no timing run exists).
- Outputs are identical to main: `make regression` passes in fp32, bf16 and int8 (bf16 GPNAE accuracy is the known reported item); pipeline words, mesh results, GPNAE results, GEMM, pack, TFLite and ResNet-8 match.
- At `LINK_STAGES` 0 every perf-analysis config (TB_sienna_top streams, 32/32/41 in fp32/bf16/int8) is equal or better: first-output latency is 1 cycle lower without pooling and 5 lower with pooling; activation-bound streams run 1 cycle per set faster (fp32 tanh 263.0 → 262.0); pooled ReLU/linear streams run 21.0 → 19.0 cycles per set (+10.5%).
- ResNet-8 bf16 on the layer engine runs 51959 → 51501 cycles (−0.88%, 54.69 → 54.21 µs).
- Why better: the mesh pushes a result as soon as its bank is full (no collect, feed, read sequence), and FIFO2 → maxpool on credits takes 4 cycles per 2x2 window where the old feeder took 6.
- At `LINK_STAGES` 1 the latency is 4 cycles above `LINK_STAGES` 0 (one per staged link) and activation-bound streams pay +2 cycles per set (the L3 round trip); host-bound streams are unchanged except two: the credits-3 ReLU stream (fp32 39.3 → 40.7, bf16 37.7 → 39.0, int8 32.3 → 33.7 cycles per set) and matmul_accum2_mixed_nopool (fp32 53.5 → 54.7, bf16 39.4 → 40.6, int8 23.0 → 24.0).
- `sienna_layer`'s grant-then-put round trip (Task 6: a set's row credits are granted before the producer may put) adds 1 to 3 cycles per layer run against the design before Task 6: 1 per GEMM QUICK grid case, 3 per cached activation layer, per ResNet-8 conv or add layer and per pack job.
- On short `sienna_layer` runs that is more than the earlier tasks saved, so these end above main: pack int8 packed 315 → 316 and its single jobs 445/884/1589 → 448/890/1602 (2/4/8 jobs); the cached residual layer 2782 → 2783; ResNet-8's first conv (L00) 2162 → 2164.
- Longer runs stay below main: every GEMM QUICK case (−1 per grid case, −19 per activation layer), every other ResNet-8 layer, and the whole network.
- Mesh latency (77 cycles at N 16, T 4, fp32) and GPNAE lane timing are unchanged; the GPNAE and mesh testbench cycle counts fell only because those testbenches now put and read faster.

### Final review fixes: runs (N 16, T 4)

- `make credit` (crf_aril): every mode passes; MODE 5's same-cycle credits fire nothing, the old check on them fires `a_no_overflow` (crf_oldovf.patch); MODES 6, 7, 8 stop at `G_BAD_CRW`, `G_BAD_MAX` and `credit_reg`'s width check.
- The 500-cycle hold in TB_sienna_top now starts early enough that the host is blocked with a set to put in every test with more sets than `SETS_IN_FLIGHT`: by staging credits in the 15-credit no-pool tests (14 sets in flight), by the entry in the pooled tests (the pooled activations wait on staging briefly first, pooled ReLU/linear only on the entry) and where fewer credits or partial sums make it bind first; the check passes on one blocked cycle, so it proves the stall reaches the host, not how long it holds it (matmul_mixed_act's 6 sets and matmul_accum2_mixed_nopool's 8 all fit in flight and are reported as not reachable); outputs identical to the plain pass, pipeline words identical to `crb_` in int8, bf16 and fp32 (crf_hold2_reg16_*_T4); the old start fires the new check (crf_hold2_mut).
- `LINK_STAGES` 1: ResNet-8 bf16 on the layer engine gives hw top 5 = ref top 5, label 2, max |hw-ref| 1.08e-02 as main, 51538 cycles (51501 at stage 0; crf_resnet8_ls1_bf16); TB_sienna_multi int8 cached passes with the stage-0 output hash, 54.5 cycles per set (53.5 at stage 0; crf_multi_ls1_int8).
- Mid-stream reset (`+mid_reset=C`): TB_model_run on the fp32 cached residual layer (also with 2 L9 slots and 50% stalls, and at `LINK_STAGES` 1) and TB_sienna_multi int8 cached (a fill open at the reset) come home (row counts 0, staging and region credits back, the turn at copy 0, every L9 slot re-advertised, no stray output), then give outputs identical to a run without the reset (crf_layer, crf_layer_ls1, crf_multi_int8, crf_multi_ls1_int8); `-DTB_STALE_HOME` fires the home check (crf_layer_stale, crf_multi_stale).
- `make regression FMT=int8` passes on the final SIENNA head (crf_gate2_int8); GPNAE's lane regression (crf_gp_int8b) and the SystolicMesh N 16 int8 regression (crf_sm16_int8b) pass on the AriL bump.

### Known limits

- The mesh's push leaves one idle beat between two result banks (rate BEATS/(BEATS+1)); not exposed in SIENNA at N 16 T 4 because L3 is granted one set at a time.
- With `LINK_STAGES` 1 every activation-bound set pays the L3 grant-to-first-beat round trip (2*LINK_STAGES). Removing it needs a grant a round trip before the lanes finish, with a second bank reserved.
- A back-pressuring consumer desynchronises the lanes; consumers rebuild each set per lane in window order (TB_sienna_top, TB_model_run, TB_sienna_model, TB_sienna_multi).
- Not covered: `OUT_CRW` > 1; `LINK_STAGES` > 1; N = 64 (after check-in only); a mid-stream reset at the instant a region credit is half returned across `sienna_multi`'s copies; a mid-stream reset while `sienna_layer` holds W row credits or a region is mid-fill (every TB_model_run reset landed with W 0 credits held); the hold starting after sets have completed (only int8_mixed_act_train starts it at set 5).
