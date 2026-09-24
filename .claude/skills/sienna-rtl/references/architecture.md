# SIENNA architecture

> **Partly stale since 2026-09-22.** The top-level sections describe the single outer FSM, 8 lanes and FIFO1. The current
> design has separate activation and pooling stages, 32 lanes (default since 2026-09-24) filled in parallel from a wide
> mesh read, banked buffers and credits; see `SKILL.md` and the `sienna-back-to-back` skill before trusting any FSM
> detail here. The SystolicMesh section is current as of 2026-09-24.

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

`SystolicMesh/src/`, brought in as a git submodule. Computes `C = A × B` over FP32.

### Hierarchy

```
SystolicMesh          tile grid, broadcast loader, MeshOutputSram
 ├─ AccumulationUnit  one per (i,j) tile position — reduces the depth slices
 ├─ SyncArray         one per (i,j,k) when SYNC_TILES=1 (default) — a TILE_SIZE² synchronous PE array
 │   └─ SyncPE ─ fp32Multiplier, fp32Adder
 └─ SystolicArray     one per (i,j,k) when SYNC_TILES=0 — the older handshake tile
     ├─ NorthInputQueue ─ ColumnInputQueue   (column-major striding)
     ├─ WestInputQueue  ─ RowInputQueue      (row-major striding)
     ├─ PEMesh ─ ProcessingElement ─ MAC ─ fp32Multiplier, fp32Adder
     └─ OutputSram
```

`TILES_PER_DIM = MATRIX_SIZE / TILE_SIZE`. The problem decomposes into `TILES_PER_DIM³` tiles: `TILES_PER_DIM²` spatial positions each with `TILES_PER_DIM` depth slices. With `COLLAPSE_K=1` there is one tile per position, `SyncArray` with depth `K = MATRIX_SIZE`, so there are `N²` PEs instead of `N³/T` and the `AccumulationUnit` (P=1) only copies the result out.

### SyncArray (default tile since 2026-09-24)

Output-stationary and fully synchronous. A (N×K) and B (K×N) sit in local registers, written one row per broadcast cycle. On start, row r of A and column c of B enter r and c cycles late and every operand moves one PE per cycle, so there are no valid joins, no per-PE FSM and no drain wave. Each `SyncPE` takes a product every cycle into one of six partial sums (the adder latency plus one), then adds the partials pairwise with the same adder. Results are read straight from the PE registers through the tile read port. Summation order differs from a serial sum, which the 1% tolerance absorbs. Unit test: `SystolicMesh/testbenches/TB_SyncArray.sv`, parameterized by N and K.

Host writes: `HOST_WORDS` words per write (the regression uses one row, N words). Mesh write pointer steps by `HOST_WORDS`, which must divide N².

### Mesh FSM

`IDLE → RESET_SEQ → BROADCAST → FIRE_PULSE → WAIT_TILES → REDUCE_PULSE → WAIT_REDUCE → DONE`. `BROADCAST` walks `load_idx` over the `TILE_SIZE` tile rows, computing per-tile addresses into the flat `mem_A`/`mem_B` arrays and writing one whole tile row into every tile's input queue in parallel (the queues take `WRITE_WORDS = TILE_SIZE` words per write since 2026-09-23), so it costs T cycles, not T². All tiles then fire simultaneously, so total latency tracks one tile's depth rather than the tile count.

### Row vs Column input queue

`RowInputQueue.sv` and `ColumnInputQueue.sv` are otherwise identical files. The entire difference is the addressing:

| | RowInputQueue | ColumnInputQueue |
|---|---|---|
| reset address | `read_addr[i] <= i * N` | `read_addr[i] <= i` |
| advance | `read_addr[i] + 1` | `read_addr[i] + N` |

Row-major walk versus column-major walk over the same `N×N` SRAM. If you fix a bug in one, fix it in the other.

Both delay `passthrough_valid_i` by two cycles (`_d1`, `_d2`) before using it to advance, matching the PE's response latency.

### Processing element

Four-state FSM: `IDLE → LOAD_DATA → MAC_COMPUTE → OUTPUT`, one pass per input element. `MAC.sv` is itself a four-state machine (`IDLE → MULTIPLY → ACCUMULATE → DONE`) wrapping one `fp32Multiplier` and one `fp32Adder`, with single-cycle valid pulses derived from state transitions.

Data flows south, weights flow east. `east_o` is muxed: during normal operation it passes the buffered west input; during the accumulator drain (`select_accumulator_i` while in `IDLE`) it emits the accumulated value instead. `PEMesh` drains column by column using a shift register (`col_shift`) that walks a single `1` from MSB down, triggered one cycle after `done_o` rises.

`inputs_valid` propagates as a wavefront: PE(0,0) takes the external valid, the top row and left column inherit from one neighbor, and interior PEs AND their north and west neighbors' `passthrough_valid_o`.

### Output path

`OutputSram` (one per tile) collects drained columns right to left, starting at `current_column = N-1`. `AccumulationUnit` then reads the same pixel from all P depth slices each cycle, sums them in a log2(P) `fp32Adder` tree (since 2026-09-23; it was one serial adder over P·T² adds), and writes one result per cycle the reduced value to the shared `MeshOutputSram` at a globally-offset address computed from `TILE_ROW_OFFSET` / `TILE_COL_OFFSET`. `MeshOutputSram` is multi-write-port (one per tile) and single-read-port.

---

## GPNAE — the activation engine

`GPNAE/src/`, also a submodule. Evaluates SELU, sigmoid or tanh on one FP32 value at a time. "TYTAN" is the polynomial core.

### Control word

| `control_word_i` | Activation | `NUM_TERMS` | Path |
|---|---|---|---|
| `2'b00` | idle / pass-through | 1 | **not implemented** — see known-issues |
| `2'b01` | SELU | 14 | sign check, then SeLu block |
| `2'b10` | sigmoid | 15 | MAC → sigtan, `select_sub = 2'b00` |
| `2'b11` | tanh | 30 | input doubled, MAC → sigtan, `select_sub = 2'b01` |

These encodings match `activation_to_code()` and `get_polynomial_terms()` in `regression.py`.

### Structure

`gpnae.sv` holds an `InputFIFO`, a `mac` (the Taylor/Maclaurin evaluator), a `SeLu` block, a `sigtan` block, and `gpnae_control_unit`.

The control unit runs `IDLE → WAIT_LAST → OP → {SELU_CHECK→SELU_POS|SELU_NEG, SIGMOID, TANH} → OP`. It stays in `OP` as the per-element dispatch point, looping until the FIFO drains. `done_o` is delayed through a 3-bit shift register so downstream logic sees a settled result.

It also gates clocks combinationally: `selu_clk_o = clk_i & selu_enable`, and similarly for `mac_clk_o` and `sigtan_clk_o`. `SeLu.sv` does the same to its `fp32_down` instance. This is power-motivated but is not synthesis-clean — see known-issues.

### Polynomial evaluation (TYTAN)

`datapath.v` is Horner's method in hardware: one multiplier and one adder wired so that each pass computes `coeff + x·previous`. `controller.sv` sequences it: `IDLE → RESET_DATAPATH → LOAD_SIGNAL → LOAD_COEFF → MULTIPLY → WAIT_MUL → ADD → WAIT_ADD → CHECK_TERMS → STORE_RESULT`, looping back to `LOAD_COEFF` until `term_count` reaches `terms_i - 1`.

Coefficients come from `CoeffROM` → `rom_block`, addressed **descending** (`coeff_addr_o = terms_i - 1 - term_count`), which is what Horner requires.

`controller.sv` tracks a credit counter clocked on `mac_credit_clk_i`, incremented on `fifo_wr_i` and decremented on entering `LOAD_SIGNAL`, so it knows how many elements are waiting without polling the FIFO. Note that `start_i` (driven by the top level's `last_i` pulse) acts as a **stop** condition in `IDLE`, not a go signal.

### Transcendental composition

Sigmoid and tanh share one identity. The MAC evaluates `e^x` (with the input pre-doubled for tanh, via a manual exponent increment in `gpnae.sv`). Then `fp32_up_down.sv` computes `e^x + 1` and `e^x − 1` in parallel — it is `fp32_down.sv` duplicated with two constant paths. Finally `fp32Divider` produces:

- sigmoid: `e^x / (e^x + 1)` — `select_sub = 2'b00` selects `mac_result` as numerator
- tanh: `(e^2x − 1) / (e^2x + 1)` — `select_sub = 2'b01` selects `sub_result`

SELU takes a different route: positive inputs are multiplied by λ (`32'h3F867D5F`) directly; negative inputs go through `fp32_down` (which computes `e^x − 1`) and are then multiplied by λα (`32'h3FD62D7D`).

### InputFIFO

`InputFIFO.v` is not a pointer FIFO. It keeps a `2^ADDR_LINES` status bitmap and uses two priority encoders (`PE5B`) to find the first free slot for writing and the first occupied slot for reading, backed by a `dual_port_ram`. Simultaneous read and write is **not** supported — the `always` block is `if (wr_en_i) ... else if (rd_en_i) ...`, so a read coinciding with a write is dropped.

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

`Maxpool/Maxpool_2D.sv` — five states (`IDLE → COLLECT_INPUT → PROCESS → OUTPUT_RESULTS → FINISH`). Collects `IN_ROWS × IN_COLS` values into a 2-D buffer, computes one output pixel per cycle in `PROCESS`, then streams the buffer out. The `is_greater()` function does sign-magnitude comparison when `IS_FP32` is set, because a raw `$signed` compare is wrong for FP32 negatives; `−infinity` (`32'hFF800000`) is the identity element.

`Dropout/dropout.sv` — a 32-bit LFSR (taps 31/21/1/0) compared against a threshold derived from `DROPOUT_P_PERCENT`. Three paths: inference (`training_mode` low) passes data straight through combinationally; drop emits zero; keep routes through an `fp32Multiplier` by the scale constant. **The top level hardwires `training_mode` to `1'b0`**, so in the assembled pipeline dropout is always a pass-through and the multiplier is never exercised.

---

## fwft FIFO

`src/fwft.sv` — first-word-fall-through, `rd_data_o` is combinational from `mem[rd_ptr]`. It **never backpressures**: `wr_ready_o` is tied high and a write into a full FIFO overwrites the oldest entry, advancing the read pointer to match. Producers must be rate-limited externally. FIFO1 is sized `SRAM_DEPTH` (256) so this does not trigger at the default configuration; FIFO2 is only 16 deep and relies on the dispatcher's explicit `<= FIFO2_DEPTH - 4` check.

---

## Parameter propagation

`regression.py` writes `testbenches/test_config_pkg.sv`; `TB_sienna_top.sv` imports it and passes the values down to `sienna_top`, which derives its local parameters and passes them to the leaf modules. The chain is:

```
regression.py  →  test_config_pkg.sv  →  TB_sienna_top  →  sienna_top  →  leaf modules
```

`SRAM_DEPTH` and `FIFO_DEPTH` default to `N*N`. `IN_ROWS`/`IN_COLS` are set to `N`. `ADDR_LINES` is `ceil(log2(SRAM_DEPTH))`. GPNAE's own widths are fixed locally in `sienna_top` (`GPNAE_ADDR_LINES = 5`, so 32-deep lanes) and are *not* driven from the package.

Note that `sienna_top.sv` currently instantiates `SystolicMesh` with its parameter overrides commented out, so the mesh falls back to its own defaults (`MATRIX_SIZE = 16`, `TILE_SIZE = 4`). Changing `N` in the config package alone will not resize the mesh.
