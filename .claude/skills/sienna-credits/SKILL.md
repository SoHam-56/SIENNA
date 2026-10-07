---
name: sienna-credits
description: Use when designing, building or verifying credit-based interfaces at SIENNA's module boundaries - the credit_link_if SystemVerilog interface, the producer credit counter and the protocol checker in ArithmeticLibrary, credit links between the host interface, SystolicMesh, the activation stage, the GPNAE lanes, pooling, maxpool, dropout and SIENNA's output, output back-pressure, or the before/after performance comparison of the credit RTL.
---

# SIENNA credit links

**Status: design, 2026-10-07, awaiting Soham's review.** Approach A chosen in discussion: one credit link at every
module boundary, today's buffers kept where they are, ports as SV interfaces. No branch or RTL change before the
spec is approved. Today's protocols (what this replaces) are tabled in the `sienna-back-to-back` skill, "Module
boundaries as built".

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
