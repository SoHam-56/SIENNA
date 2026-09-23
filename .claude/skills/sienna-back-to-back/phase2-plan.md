# SIENNA back-to-back — Phase 2: mesh streaming

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` syntax.

**Goal:** `SystolicMesh` overlaps its own work with its neighbours: the host loads set i+1
while the mesh computes set i, and the consumer reads set i while the mesh computes set i+1.

**Architecture:** Two ping-pong buffers inside the mesh. The input staging `mem_A`/`mem_B`
gets two banks, so a start pulse queues a loaded set instead of launching it, and
`BROADCAST` frees its bank the moment it has copied it into the tile queues. The result
memory `MeshOutputSram` gets two banks, released by a new consumer pulse
`result_release_i`. The mesh FSM itself stays serial (`BROADCAST` .. `DONE`); only its
boundaries overlap. `sienna_top` stays serial in this phase and only learns to release.

**Tech stack:** SystemVerilog, Verilator 5.035, farm runs via `blaunch`.

**Spec:** `SKILL.md` in this directory (Architecture, Phasing row 2). Phase 1 plan:
`implementation-plan.md`.

## Global constraints

- Never build or simulate on the login node. Farm only, as in `sienna-rtl` SKILL.md:
  `blaunch --cpus 16 --mem 32 ... launch -c gnu/gcc@13.2 -- <script>`. Scripts and logs
  live in `/proj/work/spramanik/sienna_jobs` (`mesh_sim.sh`, `run_in.sh`, `top_sim.sh`).
- Two jobs must never build in the same `Verilator/` directory at once.
- No completion or progress flag may be sticky; each new flag is set by one event and
  cleared by another (the skill's one rule).
- `regression.py` rewrites `MATRIX_SIZE`, `TILE_SIZE`, `NUM_TEST_SETS` in the TB by regex.
  Never reuse those names.
- Comments are one line. Commit messages carry the explanation. No attribution lines.
- One change per commit; SystolicMesh commits push before the parent pointer moves.
- Pass criterion is the float tolerance check with its self-test (known-issues #7).

## Interface contract (new)

| Port | Dir | Meaning |
|---|---|---|
| `input_ready_o` | out | a staging bank is free; the host may write a set and pulse start |
| `start_matrix_mult_i` | in | one-cycle pulse: the set just written is complete; ignored if `input_ready_o` is low |
| `collection_complete_o` | out | the oldest unreleased result is readable (level, cleared by release) |
| `result_release_i` | in | one-cycle pulse after the consumer's last read of that result |

Host writes while `input_ready_o` is low are dropped. The host must not write on the
cycle it pulses start. Reads use addresses `0 .. N*N-1` and always see the oldest
unreleased result.

## Review focus

- Release with no result outstanding: must be ignored, `collection_complete_o` stays low. (Task 3 test)
- Start while both staging banks are full: must be ignored, not corrupt a queued set. (Task 3 TB protocol check flags the TB ever doing it; RTL drops it)
- A result completing without a matching launch: the stale-flag class. (Task 1 monitor: completions never exceed launches, both equal K)
- TB inputs set after a rising edge are taken at that edge under Verilator, so monitors read registered state, never a combinational view of start.
- The serial top level, which never queues ahead: must still get the right set's results. (Task 4, top back-to-back + perturbed run)
- Tile sizes other than 4: bank offsets must hold for T=2/8/16. (Task 4, full mesh regression)

---

### Task 1: Handshake ports, serial release, and the failing streaming test

**Files:**
- Modify: `SystolicMesh/src/top/SystolicMesh.sv` (ports, stub `input_ready_o`)
- Modify: `SystolicMesh/testbenches/TB_SystolicMesh.sv` (release in serial pass, streaming pass)
- Modify: `src/sienna_top.sv` (connect release, parent repo)

**Interfaces:** Produces `SystolicMesh.result_release_i`, `SystolicMesh.input_ready_o`,
TB task `stream_all_sets()`.

- [ ] **Step 1: Ports with stub behaviour**

In the `SystolicMesh` port list after `collection_active_o`:

```systemverilog
    input  logic result_release_i,  // consumer finished reading the oldest result
    output logic input_ready_o,     // a staging bank is free for the host
```

After `state_t current_state, next_state;`:

```systemverilog
  assign input_ready_o = (current_state == IDLE) || (current_state == DONE);  // stub until banks exist
```

- [ ] **Step 2: Release from `sienna_top`**

Declare `logic systolic_release;` beside `systolic_reading`. In the registered block that
drives `systolic_start`, reset it to 0 and otherwise:

```systemverilog
      systolic_release     <= systolic_read_enable && !systolic_read_enable_next;  // after the last read
```

On the `SystolicMesh` instance:

```systemverilog
      .result_release_i      (systolic_release),
      .input_ready_o         (),
```

- [ ] **Step 3: TB wiring and serial release**

Declarations near `n_empty`: `wire in_ready, coll_complete; reg rel;`. Connect
`.collection_complete_o(coll_complete)`, `.result_release_i(rel)`, `.input_ready_o(in_ready)`.
In `apply_reset` set `rel = 0;`. In `execute_test_set`, right after `verify_results(...)`:

```systemverilog
      rel = 1;  // hand the result bank back
      @(posedge clk);
      rel = 0;
      @(posedge clk);
```

Factor the file naming out of `execute_test_set` into:

```systemverilog
  task automatic set_files(input int s, output string f_a, output string f_b, output string f_c);
    if (NUM_TEST_SETS == 1) begin
      f_a = "matrixA.mem"; f_b = "matrixB.mem"; f_c = "matrixC.mem";
    end else begin
      f_a = $sformatf("matrixA_%0d.mem", s);
      f_b = $sformatf("matrixB_%0d.mem", s);
      f_c = $sformatf("matrixC_%0d.mem", s);
    end
  endtask
```

- [ ] **Step 4: Streaming pass and overlap monitor**

```systemverilog
  // ── Streaming: host, mesh and consumer run concurrently ───────────────────
  // Sampled on the falling edge so no coroutine races the registers it watches.
  int  in_overlap = 0, out_overlap = 0, n_launched = 0, n_completed = 0;
  bit  streaming = 0, count_bad = 0;
  wire mesh_busy = (int'(dut.current_state) != 0) && (int'(dut.current_state) != 7);  // not IDLE, not DONE
  initial forever begin
    @(negedge clk);
    if (streaming) begin
      if ((w_we || n_we) && mesh_busy) in_overlap++;
      if (r_en && mesh_busy) out_overlap++;
      if (int'(dut.current_state) == 1) n_launched++;  // RESET_SEQ lasts one cycle per set
      if (dut.set_done) n_completed++;
      if (n_completed > n_launched) count_bad = 1;
    end
  end

  task automatic stream_all_sets();
    longint t0;
    $display("\n[STAGE] STREAMING: %0d sets, host / mesh / consumer concurrent", NUM_TEST_SETS);
    streaming = 1;
    t0 = $time;
    fork
      begin
        fork
          begin : producer
            string f_a, f_b, f_c;
            for (int s = 0; s < NUM_TEST_SETS; s++) begin
              set_files(s, f_a, f_b, f_c);
              while (!in_ready) @(posedge clk);
              fork
                load_west_queue(f_a);
                load_north_queue(f_b);
              join
              if (!in_ready) $display("  [FAIL] Start pulsed while input_ready_o is low");
              start_mult = 1;
              @(posedge clk);
              start_mult = 0;
              @(posedge clk);  // let the bank flip land before sampling in_ready again
            end
          end
          begin : consumer
            string f_a, f_b, f_c;
            int errs;
            for (int s = 0; s < NUM_TEST_SETS; s++) begin
              set_files(s, f_a, f_b, f_c);
              while (!coll_complete) @(posedge clk);
              $display("  [Stream] set %0d readable @%0t", s, $time);
              verify_results(f_c, errs);
              total_sets_run++;
              if (errs == 0) sets_passed++;
              else sets_failed++;
              rel = 1;
              @(posedge clk);
              rel = 0;
              @(posedge clk);
            end
          end
        join
      end
      begin : watchdog
        repeat (TIMEOUT_CYCLES * NUM_TEST_SETS) @(posedge clk);
        $display("  [FATAL] Timeout in the streaming pass");
        $finish;
      end
    join_any
    disable fork;
    streaming = 0;
    $display("  [Stream] %0d sets in %0d cycles", NUM_TEST_SETS, ($time - t0) / CLK_PERIOD);
    $display("  [Stream] host loading while mesh busy: %0d cycles", in_overlap);
    $display("  [Stream] consumer reading while mesh busy: %0d cycles", out_overlap);
    if (in_overlap == 0) $display("  [FAIL] Overlap: host never loaded a set while the mesh was busy");
    if (out_overlap == 0) $display("  [FAIL] Overlap: consumer never read a result while the mesh was busy");
    if (count_bad || n_completed != NUM_TEST_SETS || n_launched != NUM_TEST_SETS)
      $display("  [FAIL] %0d sets launched and %0d completed, expected %0d each", n_launched,
               n_completed, NUM_TEST_SETS);
  endtask
```

`dut.set_done` does not exist until Task 2. For this task add
`logic set_done; assign set_done = (current_state == WAIT_REDUCE) && all_reducers_done;`
to `SystolicMesh.sv` next to the stub; Task 2 keeps it.

In the top-level `initial`, time the serial loop and run the stream after it:

```systemverilog
    begin
      longint t_serial;
      t_serial = $time;
      for (int i = 0; i < NUM_TEST_SETS; i++) execute_test_set(i);
      $display("  [Serial] %0d sets in %0d cycles", NUM_TEST_SETS, ($time - t_serial) / CLK_PERIOD);
    end
    stream_all_sets();
```

- [ ] **Step 5: Run — expect RED**

Farm: `mesh_sim.sh TRACE=0`. Expected: serial sets all pass; streaming fails, with both
overlap counts 0 or element `[FAIL]`s (the next set overwrites the only result bank while
the consumer reads). Record which. If the stream passes, STOP — the test is not
exercising overlap.

- [ ] **Step 6: Commit (SystolicMesh only; the parent change waits for Task 4)**

```bash
git add src/top/SystolicMesh.sv testbenches/TB_SystolicMesh.sv
git commit -m "Add the mesh streaming handshake and a failing streaming test"
```

---

### Task 2: Two result banks

**Files:** Modify `SystolicMesh/src/top/SystolicMesh.sv`

- [ ] **Step 1: Bank state**

Declare next to `ctrl_reset_all` (they are read by the FSM):

```systemverilog
  logic [1:0] out_full;  // per result bank: holds a finished, unreleased result
  logic out_wr, out_rd;  // bank the reducers write, bank the consumer reads
```

After the FSM, replacing nothing:

```systemverilog
  // Result banks: set by a finished reduce, cleared by the consumer's release.
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      out_full <= '0;
      out_wr   <= 1'b0;
      out_rd   <= 1'b0;
    end else begin
      if (set_done) begin
        out_full[out_wr] <= 1'b1;
        out_wr <= ~out_wr;
      end
      if (result_release_i && out_full[out_rd]) begin
        out_full[out_rd] <= 1'b0;
        out_rd <= ~out_rd;
      end
    end
  end
```

- [ ] **Step 2: Gate starts on a free result bank**

In the FSM: `IDLE: if (start_matrix_mult_i && !out_full[out_wr]) next_state = RESET_SEQ;`
and in `DONE` the same condition. Stub becomes
`assign input_ready_o = ((current_state == IDLE) || (current_state == DONE)) && !out_full[out_wr];`

- [ ] **Step 3: Bank the memory**

`MeshOutputSram` `.DEPTH(2 * GLOBAL_ELEMENTS)`. Writes and reads get the bank offset:

```systemverilog
  logic [NUM_TILES-1:0][31:0] sram_addr_bank;
  always_comb
    for (int p = 0; p < NUM_TILES; p++)
      sram_addr_bank[p] = sram_addr_agg[p] + (out_wr ? GLOBAL_ELEMENTS : 0);
```

`.waddr_i(sram_addr_bank)`, `.read_enable_i(read_enable_i && read_addr_i < GLOBAL_ELEMENTS)`,
`.read_addr_i(read_addr_i + (out_rd ? GLOBAL_ELEMENTS : 0))`.

Replace the old `collection_complete_o` assign and its comment with:

```systemverilog
  assign collection_complete_o = out_full[out_rd];  // cleared by release, never sticky
```

- [ ] **Step 4: Run** — expected: all elements pass, `out_overlap > 0`, `in_overlap == 0`,
so the stream still fails on the input overlap check only.

- [ ] **Step 5: Commit** — "Double-buffer the mesh result memory".

---

### Task 3: Two staging banks and queued starts

**Files:** Modify `SystolicMesh/src/top/SystolicMesh.sv`, `TB_SystolicMesh.sv`

- [ ] **Step 1: Staging banks**

```systemverilog
  logic [DATA_WIDTH-1:0] mem_A[0:2*GLOBAL_ELEMENTS-1];
  logic [DATA_WIDTH-1:0] mem_B[0:2*GLOBAL_ELEMENTS-1];
  logic [$clog2(GLOBAL_ELEMENTS):0] ptr_A, ptr_B;
  logic [1:0] in_full;  // per staging bank: a started set not yet broadcast
  logic in_wr, in_rd;   // bank the host writes, bank BROADCAST reads
  logic start_accept, bcast_release;
  assign input_ready_o = !in_full[in_wr];
  assign start_accept  = start_matrix_mult_i && input_ready_o;
```

The write block: rewind on `west_write_reset_i || start_accept` (drop `ctrl_reset_all`),
write only when `input_ready_o`, at `mem_A[int'(in_wr) * GLOBAL_ELEMENTS + int'(ptr_A)]`
(same for B). In the same block:

```systemverilog
      if (start_accept) begin
        in_full[in_wr] <= 1'b1;
        in_wr <= ~in_wr;
      end
      if (bcast_release) begin
        in_full[in_rd] <= 1'b0;
        in_rd <= ~in_rd;
      end
```

`bcast_release` is `(current_state == BROADCAST) && loading_done`, assigned after
`loading_done`. Remove the Task 1/2 stub assign of `input_ready_o`.

- [ ] **Step 2: FSM runs queued sets**

`IDLE: if (in_full[in_rd] && !out_full[out_wr]) next_state = RESET_SEQ;` and the same in
`DONE`. `BROADCAST` reads `mem_A[int'(in_rd) * GLOBAL_ELEMENTS + addr_calc]` (same for B).

- [ ] **Step 3: Spurious-release test**

At the end of `stream_all_sets`, after the overlap checks:

```systemverilog
    rel = 1;  // nothing outstanding: must be ignored
    @(posedge clk);
    rel = 0;
    repeat (2) @(posedge clk);
    if (coll_complete) $display("  [FAIL] Release with no result outstanding raised collection_complete_o");
```

- [ ] **Step 4: Run** — expected: all sets pass in both passes, both overlap counts > 0,
completions == accepted starts. Record serial vs stream cycles.

- [ ] **Step 5: Commit** — "Double-buffer the mesh staging memory and queue starts".

---

### Task 4: Regressions, top level, docs, push

- [ ] **Step 1:** Mesh `make regression` on the farm. Green at T=2/4/8/16, 68 tests.
- [ ] **Step 2:** SIENNA `make regression` (5/5), then `-DBACK_TO_BACK` and the perturbed
  second matrix (`top_perturb_edit.py`, pass 2 must fail), in separate build dirs.
- [ ] **Step 3:** Update `SKILL.md` status and phasing (phase 2 done, measured cycles).
- [ ] **Step 4:** Push SystolicMesh, then one parent commit: bump plus the `sienna_top`
  release (they only build together), then the skill commit, and push.

## Phase 2 done when

- `TB_SystolicMesh` streams K distinct sets with both overlap counts > 0 and all pass
- Mesh regression green at every tile size; SIENNA regression and back-to-back green
- The perturbed back-to-back run fails pass 2
