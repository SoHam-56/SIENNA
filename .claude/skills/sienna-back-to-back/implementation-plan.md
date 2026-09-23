# SIENNA back-to-back — Phase 1: mesh re-arm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` syntax.

**Goal:** Make `SystolicMesh` compute correctly for a stream of matrix sets with no
reset between them.

**Architecture:** Five completion/progress flags in the mesh are cleared only by
`rstn_i`. `ctrl_reset_all` already exists, already means "re-arm for a new matmul", and
is already wired to the input queues' `write_reset_i`. This phase routes it to the
output side and the read side, and rewinds the staging write pointers.

**Tech stack:** SystemVerilog, Verilator 5.035, Python 3 stimulus generators.

**Spec:** `SKILL.md` in this directory. Known-issues #15 in the `sienna-rtl` skill
carries the measured evidence.

**Already in the working tree:** the fifth defect site, `SystolicMesh.sv:189`, was gated
during diagnosis to `all_reducers_done && (current_state == DONE) && !start_matrix_mult_i`.
It has no task here. Commit it with Task 2, which is the first task to touch that file.

**Note on stimulus:** padding with random pairs changes sets 3-5 of the uniform tests
(`gen_mm_ones`, `gen_mm_identity`, `gen_mm_zero_b`) from uniform to random data. That is
strictly more coverage — `_write_set` computes C from the actual A and B — but it does
mean those tests no longer exercise their named pattern in every set.

**Scope:** Phase 1 only. Phases 2-4 (mesh streaming, top-level overlap, full
regression) get their own plans once this one is green — their detail depends on what
this phase actually measures.

## Global constraints

- Build env: `SystolicMesh/../GPNAE/run_regression.sh` shows the required toolchain —
  `VERILATOR_ROOT=$HOME/.local/share/verilator`, gcc-toolset-13 on PATH, and a `g++`
  wrapper that injects `-fcoroutines`. Nothing builds without all three.
- `regression.py` rewrites `localparam MATRIX_SIZE`, `localparam TILE_SIZE` and
  `localparam int NUM_TEST_SETS` in the TB by regex. Never reuse those names.
- `--Wno-MODDUP` hides duplicate `fp32Adder`/`fp32Multiplier` across the two
  `ArithmeticLibrary` copies. This phase touches neither; if that changes, edit both.
- Pass criterion is tolerance, not bit-exactness. A low exact-match rate is normal.
- Commit one change per commit. Submodule commits land and push before the superproject
  pointer moves.
- No attribution or Co-Authored-By lines in commit messages.

---

### Task 1: Make the failing test

`TB_SystolicMesh` already runs `NUM_TEST_SETS` sets, but `execute_test_set()` calls
`apply_reset()` before **every** set, which is why these defects never surfaced at IP
level. It also pads sets by repeating the last pair, which cannot detect staleness.

**Files:**
- Modify: `SystolicMesh/matmul_tests.py:78-86` (`_pad_to`)
- Modify: `SystolicMesh/testbenches/TB_SystolicMesh.sv:12` and `:284-305`

- [ ] **Step 1: Pad with distinct data instead of repeats**

Replace `_pad_to` entirely:

```python
def _pad_to(sets: list, target: int) -> list:
    """
    Pad a list of (A, B) pairs to exactly `target` entries with fresh random pairs.

    Padding by repeating the last pair (the previous behaviour) cannot detect a stale
    result: a mesh that replays set i-1 produces byte-identical output and passes.
    Every set must differ from its neighbour for back-to-back runs to mean anything.
    """
    n = len(sets)
    if n == 0:
        raise ValueError("_pad_to needs at least one real set")
    dim = sets[0][0].shape[0]
    while len(sets) < target:
        np.random.seed(9000 + len(sets))
        A = np.random.uniform(-1, 1, (dim, dim)).astype(np.float32)
        B = np.random.uniform(-1, 1, (dim, dim)).astype(np.float32)
        sets.append((A, B))
    return sets[:target]
```

- [ ] **Step 2: Add back-to-back mode to the TB**

At `TB_SystolicMesh.sv:12`, beside `NUM_TEST_SETS`:

```systemverilog
  // Reset once at time zero only. Resetting per set hides every re-arm defect.
  localparam bit B2B_MODE = 1'b1;
```

In `execute_test_set`, replace the unconditional `apply_reset();` with:

```systemverilog
      if (!B2B_MODE || set_id == 0) apply_reset();
```

- [ ] **Step 3: Regenerate stimulus and run**

```bash
cd /proj/work/spramanik/SIENNA/SystolicMesh
python3 -c "import matmul_tests as m, os; os.makedirs('testbenches/stimulus', exist_ok=True); m.gen_mm_random('testbenches/stimulus', 16)"
```

Then build and run with the toolchain env (see Global constraints):

```bash
make verilator
```

Expected: **set 0 passes, set 1 fails.** Record the exact failure — mismatch counts and
whether `complete` asserts early. If every set passes, STOP: the test is not exercising
the defect and the rest of this plan is unverifiable.

- [ ] **Step 4: Commit the failing test**

```bash
git add matmul_tests.py testbenches/TB_SystolicMesh.sv
git commit -m "Test back-to-back sets in the mesh, with distinct data per set

Pad with fresh random pairs rather than repeats; repeated data cannot
detect a stale result. Reset once at time zero under B2B_MODE.

"
```

---

### Task 2: Re-arm `OutputSram`

`COMPLETE` is a terminal state with no exit transition, so `collection_complete_o`
sticks high until reset and the mesh's `WAIT_TILES` falls straight through.

**Files:**
- Modify: `SystolicMesh/src/mem/OutputSram.sv`
- Modify: `SystolicMesh/src/top/SystolicArray.sv`
- Modify: `SystolicMesh/src/top/SystolicMesh.sv`

**Interfaces:**
- Produces: `OutputSram.rearm_i`, `SystolicArray.rearm_i` — both `input logic`, active
  high, single cycle. Tasks 3 and 5 drive their own ports from the same source.

- [ ] **Step 1: Add the port and the exit transition**

In `OutputSram.sv`, add to the port list after `matrix_mult_complete_i`:

```systemverilog
    input logic rearm_i,
```

In the `always_comb` FSM, replace the `COMPLETE` arm:

```systemverilog
            COMPLETE: begin
                collection_complete_o = 1'b1;
                collection_active_o = 1'b0;
                if (rearm_i) next_collection_state = IDLE;
            end
```

- [ ] **Step 2: Plumb it through `SystolicArray`**

Add to the `SystolicArray` port list:

```systemverilog
    input logic rearm_i,
```

and on the `output_sram_inst` instantiation:

```systemverilog
      .rearm_i(rearm_i),
```

- [ ] **Step 3: Drive it from the mesh**

In `SystolicMesh.sv`, on each `SystolicArray` instantiation in the generate block
(near the existing `.north_write_reset_i(ctrl_reset_all)`):

```systemverilog
              .rearm_i(ctrl_reset_all),
```

- [ ] **Step 4: Run**

```bash
make verilator
```

Expected: still fails, but differently — the mesh should now spend real cycles in
`WAIT_TILES` instead of falling through. Record the new per-set cycle counts. Set 1 will
still fail because `AccumulationUnit` is still latched.

- [ ] **Step 5: Commit**

```bash
git add src/mem/OutputSram.sv src/top/SystolicArray.sv src/top/SystolicMesh.sv
git commit -m "Re-arm OutputSram collection state on a new matmul

COMPLETE had no exit transition, so collection_complete_o stayed high
until reset and WAIT_TILES fell through on every set after the first.

"
```

---

### Task 3: Re-arm `AccumulationUnit`

`RDONE: ;` is annotated "latched, as before: the mesh resets the unit between matmuls".
The mesh does not — there is no reset port on the unit at all, and `start_i` is only
checked in the `RIDLE` arm. `RDONE` is unreachable-from except by `rstn_i`.

**Files:**
- Modify: `SystolicMesh/src/engine/AccumulationUnit.sv`
- Modify: `SystolicMesh/src/top/SystolicMesh.sv`

- [ ] **Step 1: Add the port**

In the `AccumulationUnit` port list, after `start_i`:

```systemverilog
    input logic rearm_i,
```

- [ ] **Step 2: Give `RDONE` an exit**

Replace line 173:

```systemverilog
        RDONE: if (rearm_i) r_curr <= RIDLE;
```

- [ ] **Step 3: Drive it**

On the `acc_unit` instantiation in `SystolicMesh.sv`, beside `.start_i(ctrl_reduce_pulse)`:

```systemverilog
            .rearm_i(ctrl_reset_all),
```

- [ ] **Step 4: Run**

```bash
make verilator
```

Expected: still fails. `WAIT_REDUCE` should now take real cycles. The remaining failure
is the input queues never advancing their read pointers, so expect set 1 output to be
wrong rather than stale-identical.

- [ ] **Step 5: Commit**

```bash
git add src/engine/AccumulationUnit.sv src/top/SystolicMesh.sv
git commit -m "Re-arm AccumulationUnit on a new matmul

RDONE had no exit transition; the comment claiming the mesh resets the
unit between matmuls was never true - there was no reset port.

"
```

---

### Task 4: Re-arm the input queue read side

Both queues clear `pe_data_count` and `read_addr` only under `rstn_i`. On a second
`start_i` they set `queue_active` and `first_data_sent` but leave `pe_data_count` at N,
so `pe_data_count[i] < N` never fires and the read pointers never advance.

Note the two initialisations differ: row is `i * N`, column is `i`. Match each file.

**Files:**
- Modify: `SystolicMesh/src/mem/RowInputQueue.sv`
- Modify: `SystolicMesh/src/mem/ColumnInputQueue.sv`

- [ ] **Step 1: Re-arm the row queue**

In `RowInputQueue.sv`, replace the start block:

```systemverilog
      if (start_i && !queue_active) begin
        queue_active <= 1'b1;
        first_data_sent <= 1'b0;
        // Read side must return to its reset position, or pe_data_count stays at N
        // and no element is ever read again.
        for (int i = 0; i < N; i++) begin
          pe_data_count[i] <= '0;
          read_addr[i] <= i * N;
        end
      end
```

- [ ] **Step 2: Re-arm the column queue**

Identical in `ColumnInputQueue.sv`, except `read_addr[i] <= i;`

- [ ] **Step 3: Run**

```bash
make verilator
```

Expected: sets 0 and 1 pass; later sets may still fail if the staging pointer has run
past `GLOBAL_ELEMENTS` (Task 5).

- [ ] **Step 4: Commit**

```bash
git add src/mem/RowInputQueue.sv src/mem/ColumnInputQueue.sv
git commit -m "Re-arm input queue read pointers on a new matmul

pe_data_count and read_addr were cleared only on reset, so a second
start left pe_data_count at N and no element was ever read again.

"
```

---

### Task 5: Rewind and bound the staging write pointers

`ptr_A`/`ptr_B` are `[$clog2(GLOBAL_ELEMENTS):0]` — 9 bits for N=16. A second 256-word
load runs 256 to 512 and wraps to 0, so `queue_empty_o` falsely reads "empty".

Clearing on `ctrl_reset_all` is safe: `BROADCAST` addresses `mem_A`/`mem_B` directly via
`addr_calc` and never uses `ptr_A`/`ptr_B`, so the pointers are write-side only.

**Files:**
- Modify: `SystolicMesh/src/top/SystolicMesh.sv:42-57`

- [ ] **Step 1: Rewind on re-arm and saturate the write**

```systemverilog
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ptr_A <= '0;
      ptr_B <= '0;
    end else begin
      // Rewind for the next set once this one has been accepted. Without this the
      // pointer runs past GLOBAL_ELEMENTS and wraps, reading back as "empty".
      if (west_write_reset_i || ctrl_reset_all) ptr_A <= '0;
      else if (west_write_enable_i && ptr_A < GLOBAL_ELEMENTS) begin
        mem_A[ptr_A] <= west_write_data_i;
        ptr_A <= ptr_A + 1;
      end
      if (north_write_reset_i || ctrl_reset_all) ptr_B <= '0;
      else if (north_write_enable_i && ptr_B < GLOBAL_ELEMENTS) begin
        mem_B[ptr_B] <= north_write_data_i;
        ptr_B <= ptr_B + 1;
      end
    end
  end
```

- [ ] **Step 2: Run**

```bash
make verilator
```

Expected: **all 5 sets pass.** This is the phase-1 acceptance gate.

- [ ] **Step 3: Commit**

```bash
git add src/top/SystolicMesh.sv
git commit -m "Rewind and bound the mesh staging write pointers

ptr_A/ptr_B are 9-bit for N=16, so a second 256-word load wrapped to 0
and queue_empty_o falsely read empty, rejecting the next start.

"
```

---

### Task 6: Confirm no regression, then drop the top-level stopgap

**Files:**
- Modify: `testbenches/TB_sienna_top.sv` (superproject — the `load_inputs` write_reset pulse)

- [ ] **Step 1: Run the mesh's own regression**

```bash
cd /proj/work/spramanik/SIENNA/SystolicMesh && make regression
```

Expected: green. This suite resets between sets, so it exercises the fixes without
relying on B2B_MODE. Any failure here is a real regression — fix before continuing.

- [ ] **Step 2: Push the submodule**

```bash
git push
```

- [ ] **Step 3: Remove the superproject stopgap**

The `write_reset_i` pulse added to `load_inputs()` during diagnosis is now redundant —
Task 5 rewinds the pointer in RTL. Delete the pulse block and its comment, leaving
`load_inputs` as it was.

- [ ] **Step 4: Run the full pipeline back-to-back test**

```bash
cd /proj/work/spramanik/SIENNA && make verilator EXTRA_FLAGS=-DBACK_TO_BACK TRACE=0
```

Expected: passes **and** is no longer hollow. Verify by perturbing the second matrix
(set every `west_data_queue` entry to `32'h3f800000`) and confirming the results now
**differ** from pass 1. Revert the perturbation afterward. If results are still
identical, phase 1 did not fix the staleness and the failure must be diagnosed before
phase 2.

- [ ] **Step 5: Commit and push the superproject**

```bash
git add testbenches/TB_sienna_top.sv SystolicMesh
git commit -m "Drop the back-to-back write_reset stopgap, bump SystolicMesh

The mesh now rewinds its staging pointers itself.

"
git push
```

## Phase 1 done when

- `TB_SystolicMesh` passes 5 distinct sets with a single reset at time zero
- `make regression` in SystolicMesh is green
- the pipeline back-to-back test passes *and* fails when the second matrix is perturbed
