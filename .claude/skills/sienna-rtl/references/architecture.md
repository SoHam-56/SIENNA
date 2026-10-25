# SIENNA architecture

> **Partly stale since 2026-09-22.** The top-level sections describe the single outer FSM, 8 lanes and FIFO1. The current
> design has separate activation and pooling stages, 32 lanes (default since 2026-09-24) filled in parallel from a wide
> mesh read, banked buffers and credits; see `SKILL.md` and the `sienna-back-to-back` skill before trusting any FSM
> detail here. The SystolicMesh section is current as of 2026-09-24 (after the module renames).
> **Handshakes changed 2026-10-07 (branch `credits`):** every module boundary is a credit link (`credit_link_if`); the
> `sienna-back-to-back` skill tables them as built and the `sienna-credits` skill ("As built") has the contract. Ports named
> below such as `start_pipeline_i`, `last_i`, `done_o` on the poly lanes, `wc_region_busy_o` and fwft's `wr_ready_o` are gone.

Module-by-module reference. Line numbers are from the working tree at the time of writing; treat them as pointers, not guarantees.

## Contents

- [Top level](#top-level-srcsienna_topsv)
- [SystolicMesh](#systolicmesh)
- [GPNAE — the activation engine](#gpnae--the-activation-engine)
- [ArithmeticLibrary](#arithmeticlibrary)
- [Maxpool and dropout](#maxpool-and-dropout)
- [fwft FIFO](#fwft-fifo)
- [Parameter propagation](#parameter-propagation)

---

## Top level (`src/sienna_top.sv`)

740 lines. Instantiates one `SystolicMesh`, one `fwft` (FIFO1), and a `NUM_LANES`-wide generate block containing a `gpnae`, an `fwft` (FIFO2), a `Maxpool_2D` and a `dropout` per lane. `NUM_LANES` is 8.

### Outer FSM

| State | Leaves when |
|---|---|
| `IDLE` | `start_pipeline_i` and both input queues report non-empty |
| `SYSTOLIC_START_PULSE` | unconditionally next cycle (one-cycle start pulse) |
| `SYSTOLIC_PROCESSING` | `systolic_collection_complete` |
| `FEED_GPNAE_FIFO` | unconditionally (kicks off the mesh→FIFO1 read walk) |
| `LATCH_GPNAE_COUNT` | unconditionally; latches `total_elements = SRAM_DEPTH` |
| `GPNAE_ROUND` | `all_collected` |
| `DISPATCH_WINDOWS` | `disp_done` |
| `WAIT_DOWNSTREAM` | `streaming_complete` |
| `PIPELINE_COMPLETE` | `!start_pipeline_i` |

`all_collected` = every element written AND `round_done` AND `total_elements > 0`. `round_done` is true when no lane is still finalized-but-uncollected. `streaming_complete` is true when every lane's `dropout_out_count` equals its `lane_windows_total`.

### The fill FSM

Inside `GPNAE_ROUND`, a four-state machine (`F_WRITE` → `F_GAP` → `F_PULSE` → `F_ROUND_IDLE`) drains FIFO1 into the lanes. `F_WRITE` pushes one word into lane `fill_ptr`. `F_GAP` decides whether that lane is full (32 entries), or the data ran out, or the round capacity is reached — if so it goes to `F_PULSE`, which asserts `last_i` to that lane's GPNAE and advances `fill_ptr`. Otherwise it returns to `F_WRITE` for another word into the same lane.

So lanes fill **sequentially, 32 words each**, not interleaved word by word. Lane 0 gets elements 0–31, lane 1 gets 32–63, and so on.

`ROUND_CAPACITY = NUM_LANES × GPNAE_FIFO_DEPTH = 8 × 32 = 256`. At the default `N=16`, `SRAM_DEPTH = 256`, so the whole matrix fits in exactly one round and `round_number` stays 0. At `N=32` there would be four rounds. The write index into the result buffer is

```systemverilog
gpnae_out_mem[round_number*ROUND_CAPACITY + i*GPNAE_FIFO_DEPTH + done_count[i]]
```

which, given the sequential fill order above, is exactly the flat row-major index. The dispatcher reads it back as `gpnae_out_mem[in_row*IN_COLS + in_col]`. The two agree — this looks like a layout mismatch on first reading but is not one. It holds only because each lane receives exactly `GPNAE_FIFO_DEPTH` words; a change to the fill policy would break it silently.

### Window dispatcher

Walks output pooling positions `(disp_r, disp_c)` and, within each, the `POOL_H × POOL_W` source pixels `(disp_pr, disp_pc)`. Computes the padded source coordinate, substitutes `32'hFF800000` (−infinity) when out of bounds, and pushes the value into `fifo2` of lane `window_idx % NUM_LANES`. It backpressures on `fifo2_count[target_lane] <= FIFO2_DEPTH - 4`.

Each lane's `Maxpool_2D` is therefore instantiated as a degenerate `POOL_H × POOL_W` pooler with `PADDING=0` — padding is resolved upstream by the dispatcher, not inside the pooler. This is why the lane-level Maxpool parameters (`IN_ROWS(POOL_H)`, `IN_COLS(POOL_W)`) look wrong at a glance; they are deliberate.

Work is spread round-robin, so `lane_windows_total[i] = MAXPOOL_OUT_COUNT/NUM_LANES` plus one for the first `MAXPOOL_OUT_COUNT % NUM_LANES` lanes. At the default 16×16 with 2×2 pooling, stride 2 and padding 1, there are 81 output windows: lane 0 gets 11, lanes 1–7 get 10.

### Status outputs

`systolic_busy_o`, `gpnae_busy_o`, `maxpool_busy_o`, `dropout_busy_o` are decoded from the outer FSM state (plus lane activity for the last two). `intermediate_buffer_full_o` / `_empty_o` report on FIFO1.

---

## SystolicMesh

`SystolicMesh/src/`, brought in as a git submodule. Computes `C = A × B + bias` in the build's format: fp32, bf16 (products and sums in fp32, results rounded to bf16) or int8 (int32 sums).

### Hierarchy

```
SystolicMesh            staging banks, broadcaster, reduce dispatcher, result bank states, MeshOutputSram
 ├─ AccumulationUnit    one per (i,j) output tile: sums every pixel's partials (U per array, per depth slice)
 └─ SystolicArray       one per (i,j) with collapse-k, one per (i,j,k) without: a pipelined TILE_SIZE² array
     └─ ProcessingElement ─ multiplier + adder by format (fp32: fp32Multiplier + fp32Adder; bf16: fpMulWiden + fp32Adder; int8: intMultiplier + intAdder), ACC_BANKS (4) banks of U partial sums of ACC_W (32) bits
```

`TILES_PER_DIM = MATRIX_SIZE / TILE_SIZE`. With `COLLAPSE_K=1` (default) each output tile has one `SystolicArray` of depth `K = MATRIX_SIZE`: N² PEs. With `COLLAPSE_K=0` the problem is `TILES_PER_DIM³` arrays of depth T: N³/T PEs. `U = min(K, 6)` partial sums per pixel (the adder latency plus one).

Since 2026-09-25 the mesh is pipelined: sets flow through it back to back. The one-set-at-a-time mesh (a state machine per set, PEs combining their own partials, two result banks) is at git tag `serial_mesh_v2` in SystolicMesh and the parent; the older handshake tile (`PEMesh`, `MAC`, input queues, `OutputSram`) is at `handshake_tile_v1`.

### SystolicArray

Output-stationary and synchronous. Two operand banks hold A (T×K) and B (K×T), so the next set is written while one feeds; `commit_i` queues the bank just written, `load_ready_o` says one is free. The feeder walks k over a set on row 0 and a queued set follows k=K-1 with no gap; row r and column c see the same command r and c cycles later. Each `ProcessingElement` takes one product per cycle into one of U partial sums and switches to the next of `ACC_BANKS` (default 4) banks every K products. A bank is final when its last add is written back; `set_final_o` means every PE's oldest bank is final. `next_final_o` says the set after it is final too. The reader reads a pixel's U partials through the read port and pulses `release_i` after the last pixel, which frees the bank. A bank comes back about 3K cycles after its set starts, which is why three banks held N=16 to ~18.6 cycles per set and four reach the reducer's rate. Unit test: `SystolicMesh/testbenches/TB_SystolicArray.sv` (`-GN= -GK= -GREAD_GAP=`), 14 configurations; steady state is K+2 cycles per set for K ≥ 16.

### Mesh control

Three concurrent parts, no mesh-wide state machine. The host's last row may arrive in the start cycle; it lands before the staging bank switches. The broadcaster (`B_IDLE → B_LOAD → B_COMMIT`) waits for a full staging bank and every array's free operand bank, writes one tile row per cycle for T cycles, then commits. Each array feeds by itself. The reduce dispatcher starts every `AccumulationUnit` when all arrays hold a final set and the next result bank is free, or, in the cycle the reducers read their last pixel, when the next set is final (`arrays_next_final`), so reducers run back to back. A set's bias row, sampled with the start, waits in an in-order queue and enters the reducer tree as one more input. Result banks (`RESULT_BANKS`, default 4) go FREE → WRITING (reduce start) → FULL (last pixel written) → FREE (consumer release). `mesh_busy`, `set_launch`, `reduce_start` and `set_done` are the signals testbenches probe. With nothing released the mesh holds 12 sets: 2 staging, 2 operand and 4 partial-sum banks per array, 4 result banks.

### Output path

`AccumulationUnit` reads one pixel per cycle from its arrays, sums the `RP × U` partials and the bias in a log2 tree (`fp32Adder` in every float format, `intAdder` in int8; ACC_W = 32 bits) and writes one result per cycle of `OUT_W` bits to `MeshOutputSram` (bf16: `fpNarrow` rounds each fp32 sum to bf16, nearest even, at this write; the bias arrives widened to fp32 by the caller, `sienna_fmt_pkg::widen`); the pixel index and result bank travel beside the data, so a new set can be read while the previous one is still in the tree. `MeshOutputSram` has `RESULT_BANKS` banks, one write port per output tile and a wide port of `WIDE_READ` words; since the credit links the mesh reads it itself and pushes each result as `PER_LANE` beats on the L3 link, while it holds the consumer's credits, freeing the bank on the last beat. The reducer reads T² pixels per set, so the mesh needs K ≥ T² to run at K cycles per set (true for T=4 at N ≥ 16).

### sienna_layer: a layer scheduled in hardware

`src/sienna_layer.sv` (2026-09-25) wraps `sienna_top`. Software's only jobs are the configuration, the input streams and the result check: no per-set control crosses the boundary.
- Configuration (`cfg_load_i` while idle): M, K per column block, output columns, residual, bias, activation, dropout mode and seed.
- Weight stream, per column block: the bias row, then the block's weight tiles, once if cached (more than one row tile and at most `WC_TILES/2` tiles), else once per row tile.
- Activation stream, per block, per row tile: the depth tiles of A, then the residual tile.
- Results per output tile, column blocks outer and row tiles inner; `done_o` after the layer's last set.
- The set issuer derives the loops, the accumulate flags, the bias pass, the identity pass of a residual and the dropout seeds; the weight loader fills half `c%2` of the mesh's weight cache only while it holds that region's L2 credit (one put opens the fill; the credit returns when the fill's last set, marked `wc_last`, is broadcast).
- Since 2026-10-07 the two streams are credit links (`a_rows`, `w_rows`, L10): a set's N A rows (and N B rows if uncached) are granted at once per staging credit held, one set ahead; results leave on the L9 links `out[NUM_LANES]`.
- Testbench `TB_model_run.sv`; `model_runner.py --engine layer` (default, backend `RtlLayer`) and `regression.py --action gemm` format the streams with `format_layer()`, which only rearranges data.

---

## GPNAE — the activation engine

`GPNAE/src/`, a submodule; each of the pipeline's lanes is one `gpnae` (since Batch B Stage 2, 2026-10-24; `src/gpnae_lane.sv` is gone). Design D of `/proj/work/spramanik/sienna_report/gpnae_fit_study.log` (bf16: D', option (i)): short polynomial pieces with saturation past them, no lookup tables. The published Taylor lane is tagged `gpnae_taylor_final` (2f2db42).

### Control word, tier and load path

| `control_word_i` | Activation |
|---|---|
| `3'b001` | SELU |
| `3'b010` | sigmoid |
| `3'b100` / `3'b101` | ReLU / linear, exact (bypass) |
| any other | tanh |

`tier_i` (0, 1, 2) picks the accuracy tier, the sets fitted to 1%, 5% and 10%; `sienna_pipe` takes it per set from the set side (`set_side_t.tier`, the set contract) or from `cfg_tier_i` (the layer contract), with the activation. `coef_wr_i`, `coef_addr_i` (9 bits) and `coef_data_i` (32 bits) write the coefficient memory; `sienna_top` fans one load port out to every lane behind an interlock: `coef_ready_o` is high only while no set is in any pipeline, every lane's `idle_o` is high and no layer runs, and a write while it is low is refused (`a_load_ready` and each lane's `a_load_idle` fire if one lands anyway). `coef_ready_o` is combinational: it falls in the cycle of `cfg_load_i` or a host put. `a_coef_frozen` (sienna_top, both contracts) checks the lanes' memory write strobes against an in-flight count of its own (configure to `done_o`; sets put less sets completed), apart from the interlock's terms. The tests write the running activation's own words while it streams (`busy_writes`, `lanes_refused` on both contracts), so a write that lands moves outputs. The host loads between layers. The memory's reset contents are the fitted image (`GPNAE/gpnae_coef_default.json`), so a host that never loads gets the fitted lane.

### Structure

- `gpnae_act_in`: per element, the piece from abs(x) against the descriptor's bounds (count of bounds passed; past the last, the function's limit), the coefficient base, degree and fraction; floats square tanh's operand and, for negative sigmoid past the reflect range, split x log2 e into 2^-n and f; int8 rescales the code (int16Multiplier) and centres it on the fit.
- `barrel_mac`: K = 16 slots take one Horner step each in turn (`gpnae_horner`: fmt_mul then fmt_add, or fixedPointMultiplyAdd in int8), so the step unit starts one step a cycle; each slot runs its own base and degree, and a slot past its degree pads with 0.
- `gpnae_post`: lambda x (SELU x >= 0), x P (SELU x < 0, tanh), P or 1 - P (sigmoid), 2^-n P then E(1 - E) (fp32 negative sigmoid), the rails, and int8's SELU requantize.
- `gpnae_tail` (bf16 only, `HAS_FLOAT_TAIL`): negative sigmoid below -1.5 in fp32, 2^-n P(f) and Newton steps for 1/(1 + E), rounded to bf16; eight contexts.
- Results leave in input order once K output credits are held (`lane_link`).

---

## ArithmeticLibrary

A nested submodule, vendored twice (once under `GPNAE/`, once under `SystolicMesh/`). Both copies are compiled into the top-level build.

| Module | Latency | Notes |
|---|---|---|
| `fp32Adder` | 5 stages | flush-to-zero, no denormal support, truncates to `norm_man[25:3]` |
| `fp32Multiplier` | ~7 stages | mantissa via `karatsubaUnsigned`, exponent via a 4-slice pipelined adder, truncates |
| `karatsubaUnsigned` | 6 stages | three `R4Booth` multipliers plus the classic subtract-and-shift recombination |
| `karatsubaSigned` | +2 | absolute value, unsigned core, sign restore. Not in any build list |
| `R4Booth` | 2 stages | radix-4 Booth partial products, combinational adder tree |
| `fp32Divider` | ~28 stages | wraps Project F's `divu` (MIT licensed, restoring division) |
| `cntlz8` / `cntlz28` | combinational | leading-zero counters |

Both the adder and the multiplier truncate rather than round to nearest. That single decision is why the whole project verifies on relative tolerance.

The adder and multiplier handle NaN, infinity and signaling-NaN explicitly and expose `overflow_o` / `underflow_o` / `invalid_o`, all of which the SIENNA integration leaves unconnected.

`berkeley-softfloat-3` is vendored under the adder's and multiplier's `testbenches/` directories. It is the IEEE-754 reference used through DPI-C as the golden model for those unit tests — it is not part of the design.

---

## Maxpool and dropout

`Maxpool/Maxpool_2D.sv` — since 2026-10-07 on credit links (`in`, `out`; `start`/`done` and the valid ports are gone): the streaming path SIENNA uses grants a 2x2 window's 4 element credits at once, only with an output credit reserved, up to 2 windows ahead. The batch path (padded multi-output windows, `TB_maxpool_int8` only) keeps the five states (`IDLE → COLLECT_INPUT → PROCESS → OUTPUT_RESULTS → FINISH`). Collects `IN_ROWS × IN_COLS` values into a 2-D buffer, computes one output pixel per cycle in `PROCESS`, then streams the buffer out. The `is_greater()` function does sign-magnitude comparison when `IS_FP32` is set, because a raw `$signed` compare is wrong for FP32 negatives; `−infinity` (`32'hFF800000`) is the identity element.

`Dropout/dropout.sv` — a 32-bit LFSR (taps 31/21/1/0) compared against a threshold derived from `DROPOUT_P_PERCENT`. Three paths: inference (`training_mode` low) passes data straight through combinationally; drop emits zero; keep routes through an `fp32Multiplier` by the scale constant. **The top level hardwires `training_mode` to `1'b0`**, so in the assembled pipeline dropout is always a pass-through and the multiplier is never exercised.

---

## fwft FIFO

`src/fwft.sv` — first-word-fall-through FIFO on credit links since 2026-10-07: `credit_link_if.consumer in` advertises `FIFO_DEPTH` (16 for FIFO2) after reset and returns one credit per pop; `credit_link_if.producer out` puts the oldest word while it holds the consumer's credit. Overwrite-on-full is removed and `a_fifo_room` fires on a put into a full FIFO. The dispatcher's almost-full threshold is replaced by one credit counter per lane (L7). Before 2026-10-07 it never back-pressured: a write into a full FIFO overwrote the oldest entry (known-issues #14).

---

## Parameter propagation

`regression.py` writes `testbenches/test_config_pkg.sv` (with model_runner.py's `write_sv_package()`); `TB_sienna_top.sv` imports it and passes the values down to `sienna_top`, which derives its local parameters and passes them to the leaf modules. The chain is:

```
regression.py  →  test_config_pkg.sv  →  TB_sienna_top  →  sienna_top  →  leaf modules
```

`SRAM_DEPTH` and `FIFO_DEPTH` default to `N*N`. `IN_ROWS`/`IN_COLS` are set to `N`. `ADDR_LINES` is `ceil(log2(SRAM_DEPTH))`. GPNAE's own widths are fixed locally in `sienna_top` (`GPNAE_ADDR_LINES = 5`, so 32-deep lanes) and are *not* driven from the package.

Note that `sienna_top.sv` currently instantiates `SystolicMesh` with its parameter overrides commented out, so the mesh falls back to its own defaults (`MATRIX_SIZE = 16`, `TILE_SIZE = 4`). Changing `N` in the config package alone will not resize the mesh.
