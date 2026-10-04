# SIENNA multi-job packing: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a synthesized SIENNA of any N runs N/b small jobs (b = N >> pack_shift, set per set at runtime) in one N x N set,
each job's result bit-identical to the same job run alone, with its own activation and int8 output parameters.

**Architecture:** the host packs jobs as block-diagonal weights. A per-set `pack_shift` travels with the set through the
mesh; each PE counts only products in its column block (skipping the rest), so a packed block sums exactly as the job
alone. sienna_top reads a packed result into the lanes column-wise (lane k takes column k % N), so each lane's batch is
one model and takes one parameter-table entry; requantize zero point / clamp and dropout's drop value become per lane.

**Tech Stack:** SystemVerilog (Verilator 5, `--assert`), Python 3 + numpy (bit-exact models), slurm farm via
`$J/cmds/int8_tree.sh`.

**Spec:** `.claude/skills/sienna-packing/SKILL.md` (same directory). Read it before any task.

## Global Constraints

- Repos and branches: SystolicMesh branch `packing` (created in Task 1), SIENNA branch `packing` (exists). GPNAE and
  ArithmeticLibrary are not changed; if a task seems to need them, stop and report.
- Push the submodule before the parent; SIENNA's pointer bump is its own commit. No `Co-Authored-By` or generated-by
  lines in any commit: plain `git commit -m` only; before every push, `git log --format=%B origin/packing..HEAD | command grep -ci co-authored-by` must print 0.
- One logical change per commit, RTL commits before test commits. Source comments are one line, never paragraphs.
- Every build or simulation runs on the farm, never the login node: `J=/proj/work/spramanik/sienna_jobs;
  $J/cmds/int8_tree.sh NAME MEM_GB HOURS cmd words...` snapshots `/proj/work/spramanik/SIENNA` (working tree, including
  uncommitted edits and submodules) and runs the command there. Every argument is one word. Results: `$J/runs/NAME.out`
  (`exit=` line) and `$J/runs/NAME/stdout.log`, `$J/runs/NAME/results/`, `$J/runs/NAME/mesh_results/`. Append
  `LAUNCH NAME ... at <date>` to `$J/runs/pk_launch.log`. Wait with `squeue -u spramanik`. Pure-Python unit tests (no
  simulator) may run on the login node.
- Test and gate at N <= 32 only (N = 8, 16, 32 with every valid T). One N = 64 sweep after check-in, not in this plan.
- `grep`, `diff`, `du` are aliased to non-GNU tools: use `command grep` etc. in checks.
- Reports are `.log` files, never `.md`.
- `pack_shift` encoding: b = N >> pack_shift; 0 = unpacked, valid 0 .. log2(N) - 1; 3 bits everywhere.
- `PACK_ENTRIES` = 8 (a power of two >= 2); entry 0 is the existing per-set ports; map has N/2 words of 3 bits.
- A set with pack_shift = 0 and map[0] = 0 must be cycle- and bit-identical to `pre_packing_v1`.
- Packing is refused (assertion) when: N does not divide NUM_LANES; the build pools (not POOL_BYPASS); COLLAPSE_K = 0;
  the set accumulates (accumulate_i / partial_i, or continues a partial); pack_shift >= log2(N).
- Bit-exact everywhere: int8, bf16 and fp32 packed blocks equal the job alone bit for bit at the mesh and layer level.
  The fp32 top-level regression keeps its tolerance compare (its GPNAE fp32 model is not bit-exact, F-GP1).

## Review Focus

1. A packed set followed by an unpacked one (and the reverse) in one stream, and through reset mid-stream: no map,
   entry, lane order or pack shift may leak between set ids. Pinned by Task 5's packed tests (shifts 1, 0, b=2, ...).
2. Empty column blocks (fewer models than N/b): their outputs must be the job-less result (bias only, through its
   entry) and must not disturb other blocks. Pinned in Task 7 (`partial_packing` case).
3. Signed-zero inputs (rows of +0 and -0): packed and alone must still agree bit for bit, including the sign of zero
   results. Pinned in Task 6 (`zero_rows` case in `pack_regression.py`).
4. A build where N does not divide NUM_LANES (N = 16, LANES = 8): a packed start must fire `a_pack_lanes`, not produce
   silently wrong output. Pinned in Task 5, Step 9.
5. An int8 SELU entry whose input range saturates (x >= 487.29): `pack_jobs` must refuse it per entry. Pinned in Task 7.

---

## File map

| File | Repo | Change |
|---|---|---|
| `mesh_model.py` | SystolicMesh | `matmul_packed`, tree factored into `_reduce` |
| `test_mesh_model_packed.py` | SystolicMesh | new: packed block == job alone |
| `src/engine/ProcessingElement.sv` | SystolicMesh | `COL` parameter, `pack_i`/`pack_o`, out-of-block skip, unwritten slots read +0 |
| `testbenches/TB_PE_pack.sv` | SystolicMesh | new: PE slots under packing, fp32 / bf16 / int8 |
| `src/top/SystolicArray.sv` | SystolicMesh | `COL0`, `commit_pack_i`, pack travels east with A |
| `src/top/SystolicMesh.sv` | SystolicMesh | `pack_shift_i` per staging bank, `wide_read_packed_i`, assertions |
| `stim_format.py`, `matmul_tests.py`, `regression.py`, `testbenches/TB_SystolicMesh.sv` | SystolicMesh | packed sets in the mesh regression |
| `src/requant_lanes.sv`, `testbenches/TB_requant_lanes.sv`, `testbenches/gen_rq_lanes.py` | SIENNA | per-lane zp/min/max, packed channel map |
| `src/sienna_top.sv` | SIENNA | ports, per-id pack state and entries, lane order, per-lane parameters, D-5 per lane, refusals |
| `src/sienna_multi.sv`, `testbenches/TB_sienna_model.sv` | SIENNA | tie the new ports to 0 |
| `regression.py`, `testbenches/TB_sienna_top.sv` | SIENNA | packed pipeline tests |
| `src/sienna_layer.sv`, `testbenches/TB_sienna_layer.sv`, `model_runner.py` | SIENNA | packed layer configuration |
| `pack_regression.py` | SIENNA | new: packed layers vs each job alone, all formats |
| `model_runner.py`, `test_pack_jobs.py`, `gemm_sweep.py` | SIENNA | `pack_jobs` / `unpack`, `exact_layer` takes T |
| `tflite_pack_models.py`, `tflite_pack_run.py` | SIENNA | new: two packed TFLite runs bit-exact |
| `Makefile`, skills | SIENNA | `make pack`, docs |
| `$J/cmds/pk_cmp_mesh.py` | outside git | mesh readiness reports compared |

---

### Task 1: Bit-exact model of a packed mesh set

**Files:**
- Modify: `SystolicMesh/mesh_model.py:16-40`
- Create: `SystolicMesh/test_mesh_model_packed.py`

**Interfaces:**
- Produces: `mesh_model.matmul_packed(f, A, B, N, shift, bias=None) -> np.ndarray` (int64 bit patterns, N x N), one pass,
  collapse-k only; `mesh_model._reduce(f, parts, N, bias) -> np.ndarray`.

- [ ] **Step 1: Create the SystolicMesh branch**

```bash
cd /proj/work/spramanik/SIENNA/SystolicMesh && git checkout -b packing main && git push -u origin packing
```

- [ ] **Step 2: Write the failing test** (`SystolicMesh/test_mesh_model_packed.py`)

```python
#!/usr/bin/env python3
"""mesh_model.matmul_packed: every block of a packed set equals its job run alone as an unpacked set, bit for bit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_model as mm  # noqa: E402
from mesh_model import fpu  # noqa: E402


def _bits(x, fmt):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    return u if fmt == "fp32" else u >> (23 - fpu.FORMATS[fmt].m)


def _case(fmt, N, shift, seed, garbage=False):
    rng = np.random.RandomState(seed)
    b = N >> shift
    A = _bits(rng.uniform(-1, 1, (N, N)), fmt)
    B = _bits(rng.uniform(-1, 1, (N, N)) if garbage else np.zeros((N, N)), fmt)
    for c in range(N // b):
        B[c * b:(c + 1) * b, c * b:(c + 1) * b] = _bits(rng.uniform(-1, 1, (b, b)), fmt)
    return A, B, _bits(rng.uniform(-0.5, 0.5, N), fmt), b


def _alone(f, A, B, bias, N, c, b):
    """Block c's job as an unpacked set: its inputs and weights at the top left, zeros elsewhere."""
    Ac = np.zeros((N, N), np.int64)
    Ac[:, :b] = A[:, c * b:(c + 1) * b]
    Wc = np.zeros((N, N), np.int64)
    Wc[:b, :b] = B[c * b:(c + 1) * b, c * b:(c + 1) * b]
    bc = np.zeros(N, np.int64)
    bc[:b] = bias[c * b:(c + 1) * b]
    return mm.matmul(f, [(Ac, Wc)], N, 4, 1, bc)[:, :b]


def test_shift0_is_matmul():
    for fmt in ("fp32", "bf16"):
        f = fpu.FORMATS[fmt]
        for N in (8, 16):
            A, B, bias, _ = _case(fmt, N, 1, 11 * N, garbage=True)
            assert np.array_equal(mm.matmul_packed(f, A, B, N, 0, bias), mm.matmul(f, [(A, B)], N, 4, 1, bias)), (fmt, N)


def test_packed_block_equals_alone():
    for fmt in ("fp32", "bf16"):
        f = fpu.FORMATS[fmt]
        for N in (8, 16, 32):
            for shift in range(1, N.bit_length() - 1):
                for garbage in (False, True):
                    A, B, bias, b = _case(fmt, N, shift, 100 * N + 10 * shift + garbage, garbage)
                    P = mm.matmul_packed(f, A, B, N, shift, bias)
                    for c in range(N // b):
                        got = P[:, c * b:(c + 1) * b]
                        assert np.array_equal(got, _alone(f, A, B, bias, N, c, b)), (fmt, N, shift, garbage, c)


def test_unskipped_mesh_differs():
    # The probe's finding: without the skip, a block starting mid-rotation rounds differently from its job alone.
    f = fpu.FORMATS["fp32"]
    A, B, bias, b = _case("fp32", 16, 2, 7)
    plain = mm.matmul(f, [(A, B)], 16, 4, 1, bias)
    alone = np.hstack([_alone(f, A, B, bias, 16, c, b) for c in range(16 // b)])
    assert int(np.sum(plain != alone)) > 0


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {name}: {e!r}")
    print(f"{'ALL PASS' if bad == 0 else f'{bad} FAILED'}")
    sys.exit(1 if bad else 0)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd /proj/work/spramanik/SIENNA/SystolicMesh && python3 test_mesh_model_packed.py`
Expected: `FAIL test_packed_block_equals_alone: AttributeError(... 'matmul_packed')`, `FAIL test_shift0_is_matmul`,
`PASS test_unskipped_mesh_differs`, `2 FAILED`.

- [ ] **Step 4: Implement** (in `SystolicMesh/mesh_model.py`, replace the tree at the end of `matmul` and add the new function after it)

Replace lines 33-40 of `matmul` (from `bias_row = ...` to `return np.asarray(level[0], dtype=np.int64)`) with:

```python
    return _reduce(f, [acc[rp, u] for rp in range(RP) for u in range(U)], N, bias)


def _reduce(f, parts, N, bias):
    """The reducer: a pairwise tree over the partials with the bias as its last input; an odd entry waits a level."""
    bias_row = np.zeros(N, dtype=np.int64) if bias is None else np.asarray(bias, dtype=np.int64)
    level = list(parts) + [np.broadcast_to(bias_row[None, :], (N, N))]
    while len(level) > 1:
        nxt = [fpu.add(f, level[2 * m], level[2 * m + 1])[0] for m in range(len(level) // 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # an odd entry out waits a level, as the RTL's PASS delay
        level = nxt
    return np.asarray(level[0], dtype=np.int64)


def matmul_packed(f, A, B, N, shift, bias=None):
    """A packed set on the collapse-k mesh (one pass): PE (i, j) adds only products k in column j's block of b = N >> shift,
    the r-th of them into slot r mod U, so each block sums exactly as its job alone; unwritten slots stay +0."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    b = N >> shift
    U = min(N, ADD_LAT + 1)
    acc = np.zeros((U, N, N), dtype=np.int64)
    for c in range(N // b):
        cols = slice(c * b, (c + 1) * b)
        for r in range(b):
            k = c * b + r
            a = np.broadcast_to(A[:, k][:, None], (N, b))
            w = np.broadcast_to(B[k, cols][None, :], (N, b))
            acc[r % U][:, cols] = fpu.add(f, acc[r % U][:, cols], fpu.mul(f, a, w)[0])[0]
    return _reduce(f, [acc[u] for u in range(U)], N, bias)
```

- [ ] **Step 5: Run the test and the existing model test**

Run: `python3 test_mesh_model_packed.py && python3 test_stim_format.py`
Expected: `ALL PASS` for the new file; `test_stim_format.py` passes as before (the refactor changed no result).

- [ ] **Step 6: Commit and push**

```bash
git add mesh_model.py test_mesh_model_packed.py
git commit -m "mesh_model: matmul_packed, the collapse-k mesh on a packed set (each PE adds only its column block's products, the r-th into slot r mod U), with the reducer's tree factored into _reduce; test_mesh_model_packed.py shows every block equals its job alone in fp32 and bf16 at N = 8-32, and that the unskipped mesh does not."
git push origin packing
```

---

### Task 2: Processing element skips products outside its block

**Files:**
- Modify: `SystolicMesh/src/engine/ProcessingElement.sv` (whole file below)
- Create: `SystolicMesh/testbenches/TB_PE_pack.sv`

**Interfaces:**
- Produces: `ProcessingElement` parameter `COL` (int, default 0); ports `input logic [2:0] pack_i`, `output logic [2:0] pack_o`
  (registered pass-through like `fresh_o`); internal `prod_v` keeps its name (the TB counts it).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test** (`SystolicMesh/testbenches/TB_PE_pack.sv`)

```systemverilog
`timescale 1ns / 100ps

// One format's packed-PE bench: NP PEs at spread columns take one stream; each slot must hold only its block's products.
module pe_pack_bench #(
    parameter int EXP_W = 8,
    parameter int MAN_W = 23,
    parameter int K = 16,
    parameter int NP = 4,
    parameter int NSETS = 30
) (
    output logic done_o,
    output int   errs_o,
    output int   checked_o
);
  localparam int DW = 1 + EXP_W + MAN_W;
  localparam int ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W);
  localparam int ADD1 = sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1;
  localparam int U = (K < ADD1) ? K : ADD1;
  localparam int BANKS = 3, BW = 2, LGK = $clog2(K);
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic [DW-1:0] a = '0, b = '0;
  logic v = 0, fresh = 0, rel = 0;
  logic [2:0] pack = '0, pack_q = '0;
  logic [BW-1:0] rd_bank = '0;
  logic [U-1:0][ACC_W-1:0] part[NP];
  logic [BANKS-1:0] fin[NP];
  logic [2:0] pack_o[NP];
  int muls[NP], want_muls[NP];
  int pass_errs = 0;
  always @(posedge clk) pack_q <= pack;

  for (genvar p = 0; p < NP; p++) begin : PE
    ProcessingElement #(.EXP_W(EXP_W), .MAN_W(MAN_W), .DATA_WIDTH(DW), .ACC_W(ACC_W), .K(K), .BANKS(BANKS), .U(U), .BW(BW),
                        .COL((p * (K - 1)) / (NP - 1))) dut (
        .clk_i(clk), .rstn_i(rstn), .a_i(a), .b_i(b), .v_i(v), .fresh_i(fresh), .more_i(1'b0), .pack_i(pack),
        .a_o(), .b_o(), .v_o(), .fresh_o(), .more_o(), .pack_o(pack_o[p]),
        .rd_bank_i(rd_bank), .partial_o(part[p]), .release_i(rel), .final_o(fin[p]));
    initial muls[p] = 0;
    always @(posedge clk) if (dut.prod_v) muls[p]++;
    always @(negedge clk) if (rstn && pack_o[p] !== pack_q) pass_errs++;  // pack_o is pack_i one cycle later
  end

  // Small positive integers are exact in every float format and their sums never cancel; int8 takes the whole range.
  function automatic logic [DW-1:0] word(input int x);
    if (IS_INT) return DW'(x);
    return DW'($shortrealtobits(shortreal'(x)) >> (23 - MAN_W));
  endfunction

  int xa[NSETS][K], xb[NSETS][K], sh[NSETS];
  logic [ACC_W-1:0] want[NSETS][NP][U];

  initial begin
    int lst[4];
    done_o = 0;
    errs_o = 0;
    checked_o = 0;
    lst = '{0, 1, 2, LGK - 1};
    for (int p = 0; p < NP; p++) want_muls[p] = 0;
    for (int s = 0; s < NSETS; s++) begin
      sh[s] = (lst[s % 4] < LGK) ? lst[s % 4] : LGK - 1;
      for (int k = 0; k < K; k++) begin
        xa[s][k] = IS_INT ? int'($urandom_range(0, 255)) - 128 : int'($urandom_range(1, 4));
        xb[s][k] = IS_INT ? int'($urandom_range(0, 255)) - 128 : int'($urandom_range(1, 4));
      end
      for (int p = 0; p < NP; p++) begin
        automatic int col = (p * (K - 1)) / (NP - 1);
        automatic int bw = K >> sh[s];
        automatic int blk = col / bw;
        automatic longint sums[U];
        automatic bit hit[U];
        for (int u = 0; u < U; u++) begin
          sums[u] = 0;
          hit[u] = 0;
        end
        for (int r = 0; r < bw; r++) begin  // the r-th in-block product goes to slot r mod U
          sums[r%U] += longint'(xa[s][blk*bw+r]) * longint'(xb[s][blk*bw+r]);
          hit[r%U] = 1;
        end
        want_muls[p] += bw;
        for (int u = 0; u < U; u++)
          want[s][p][u] = !hit[u] ? '0 : IS_INT ? ACC_W'(sums[u]) : ACC_W'($shortrealtobits(shortreal'(sums[u])) >> (23 - MAN_W));
      end
    end
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    fork
      begin
        fork
          begin : feeder
            for (int s = 0; s < NSETS; s++)
              for (int k = 0; k < K; k++) begin
                @(posedge clk);
                #1 v = 1;
                a = word(xa[s][k]);
                b = word(xb[s][k]);
                fresh = 1;
                pack = 3'(sh[s]);
              end
            @(posedge clk);
            #1 v = 0;
            fresh = 0;
            pack = '0;
          end
          begin : reader
            for (int s = 0; s < NSETS; s++) begin
              automatic bit all_fin;
              do begin
                @(posedge clk);
                #1;
                all_fin = 1;
                for (int p = 0; p < NP; p++) all_fin &= fin[p][rd_bank];
              end while (!all_fin);
              for (int p = 0; p < NP; p++)
                for (int u = 0; u < U; u++) begin
                  checked_o++;
                  if (part[p][u] !== want[s][p][u]) begin
                    errs_o++;
                    if (errs_o <= 20) $display("[FAIL] EXP_W=%0d set %0d shift %0d PE %0d slot %0d: got %h, want %h", EXP_W, s,
                                               sh[s], p, u, part[p][u], want[s][p][u]);
                  end
                end
              rel = 1;
              @(posedge clk);
              #1 rel = 0;
              rd_bank = (rd_bank == BW'(BANKS - 1)) ? '0 : rd_bank + 1'b1;
            end
          end
        join
      end
      begin : watchdog
        repeat (40000) @(posedge clk);
        $display("[FATAL] pe_pack_bench EXP_W=%0d MAN_W=%0d timeout", EXP_W, MAN_W);
        $finish;
      end
    join_any
    disable fork;
    for (int p = 0; p < NP; p++)
      if (muls[p] != want_muls[p]) begin
        errs_o++;
        $display("[FAIL] EXP_W=%0d PE %0d issued %0d multiplies, want %0d", EXP_W, p, muls[p], want_muls[p]);
      end
    if (pass_errs != 0) begin
      errs_o++;
      $display("[FAIL] EXP_W=%0d pack_o was not pack_i delayed one cycle (%0d cycles)", EXP_W, pass_errs);
    end
    done_o = 1;
  end
endmodule

// ProcessingElement under packing in fp32 (U = 6), bf16 (U = 6) and int8 (U = 2), K = 16 / 16 / 8, every shift.
module TB_PE_pack;
  logic d_f, d_b, d_i;
  int e_f, e_b, e_i, c_f, c_b, c_i;
  pe_pack_bench #(.EXP_W(8), .MAN_W(23), .K(16), .NP(4)) F (.done_o(d_f), .errs_o(e_f), .checked_o(c_f));
  pe_pack_bench #(.EXP_W(8), .MAN_W(7), .K(16), .NP(4)) B (.done_o(d_b), .errs_o(e_b), .checked_o(c_b));
  pe_pack_bench #(.EXP_W(0), .MAN_W(7), .K(8), .NP(3)) I (.done_o(d_i), .errs_o(e_i), .checked_o(c_i));
  initial begin
    wait (d_f && d_b && d_i);
    $display("TB_PE_pack: fp32 %0d slots %0d errors, bf16 %0d slots %0d errors, int8 %0d slots %0d errors", c_f, e_f, c_b,
             e_b, c_i, e_i);
    if (e_f + e_b + e_i == 0 && c_f == 30 * 4 * 6 && c_b == 30 * 4 * 6 && c_i == 30 * 3 * 2) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
```

- [ ] **Step 2: Run it to verify it fails**

```bash
J=/proj/work/spramanik/sienna_jobs
$J/cmds/int8_tree.sh pk_pe_red 16 1 make -C SystolicMesh verilator TOP_MODULE=TB_PE_pack
```
Expected (`$J/runs/pk_pe_red/stdout.log`): a Verilator error that `ProcessingElement` has no parameter `COL` / no port
`pack_i` (the feature is missing).

- [ ] **Step 3: Implement** (replace `SystolicMesh/src/engine/ProcessingElement.sv` with)

```systemverilog
`timescale 1ns / 100ps

// Output-stationary PE for the pipelined SystolicArray: one product per cycle, sets back to back with no gap.
// Every K products form a pass; a set is one or more passes, accumulated into its own bank of U partial sums, which the reader combines.
// A packed set (pack_i != 0) counts only the products in this PE's column block of K >> pack_i, so each block sums as its job alone.
module ProcessingElement #(
    parameter int EXP_W      = 8,   // the build's format: fp32 8/23, bf16 8/7, int8 0/7
    parameter int MAN_W      = 23,
    parameter int DATA_WIDTH = 1 + EXP_W + MAN_W,  // operands
    parameter int ACC_W      = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // products and sums: int32 in int8, DATA_WIDTH in the float formats
    parameter int K          = 4,  // products per set
    parameter int BANKS      = 3,  // sets held at once: one accumulating, the older ones finishing or being read
    parameter int U          = (K < sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1) ? K : sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1,  // partial sums per set: the adder latency plus one
    parameter int BW         = (BANKS > 1) ? $clog2(BANKS) : 1,
    parameter int COL        = 0  // this PE's column in the whole mesh: its packed block is COL / (K >> pack)
) (
    input  logic                         clk_i,
    input  logic                         rstn_i,
    input  logic [       DATA_WIDTH-1:0] a_i,
    input  logic [       DATA_WIDTH-1:0] b_i,
    input  logic                         v_i,
    input  logic                         fresh_i,     // with v_i: this pass starts a set, its first U counted products add to 0
    input  logic                         more_i,      // with v_i: another pass of the same set follows this one
    input  logic [                  2:0] pack_i,      // with v_i: the set's pack shift; 0 is an unpacked set
    output logic [       DATA_WIDTH-1:0] a_o,
    output logic [       DATA_WIDTH-1:0] b_o,
    output logic                         v_o,
    output logic                         fresh_o,
    output logic                         more_o,
    output logic [                  2:0] pack_o,
    input  logic [               BW-1:0] rd_bank_i,   // bank the reader looks at
    output logic [U-1:0][     ACC_W-1:0] partial_o,   // that bank's partial sums; a slot no product reached reads +0
    input  logic                         release_i,   // the reader is done with rd_bank_i
    output logic [            BANKS-1:0] final_o      // per bank: a finished set, all adds written back
);
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+ADD_LAT
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+MUL_LAT
  localparam int S = ADD_LAT + 1;  // a slot is read again S cycles after its add issues, one after the write-back
  localparam int SW = (U > 1) ? $clog2(U) : 1;
  localparam int CW = $clog2(K + 1);
  localparam int LGK = (K > 1) ? $clog2(K) : 1;

`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (U > S || U > K) $error("ProcessingElement: U=%0d must not exceed min(K=%0d, %0d)", U, K, S);
  initial if (COL >= K) $error("ProcessingElement: COL=%0d must be below K=%0d", COL, K);
`endif

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      a_o <= '0;
      b_o <= '0;
      v_o <= 1'b0;
      fresh_o <= 1'b0;
      more_o <= 1'b0;
      pack_o <= '0;
    end else begin
      a_o <= a_i;
      b_o <= b_i;
      v_o <= v_i;
      fresh_o <= fresh_i;
      more_o <= more_i;
      pack_o <= pack_i;
    end
  end

  // The input's place in its pass, and whether it lies in this PE's block (always, unpacked).
  logic [CW-1:0] k_in;
  logic in_blk;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) k_in <= '0;
    else if (v_i) k_in <= (k_in == CW'(K - 1)) ? '0 : k_in + 1'b1;
  assign in_blk = (pack_i == '0) || (((int'(k_in) ^ COL) >> (LGK - int'(pack_i))) == 0);

  // The pass flags, delayed to meet their product out of the multiplier; v_d ticks for every product, counted or not.
  logic fresh_d[MUL_LAT], more_d[MUL_LAT], v_d[MUL_LAT], inb_d[MUL_LAT];
  logic [2:0] pack_d[MUL_LAT];
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < MUL_LAT; i++) begin
        fresh_d[i] <= 1'b0;
        more_d[i] <= 1'b0;
        v_d[i] <= 1'b0;
        inb_d[i] <= 1'b0;
        pack_d[i] <= '0;
      end
    end else begin
      fresh_d[0] <= fresh_i;
      more_d[0] <= more_i;
      v_d[0] <= v_i;
      inb_d[0] <= in_blk;
      pack_d[0] <= pack_i;
      for (int i = 1; i < MUL_LAT; i++) begin
        fresh_d[i] <= fresh_d[i-1];
        more_d[i] <= more_d[i-1];
        v_d[i] <= v_d[i-1];
        inb_d[i] <= inb_d[i-1];
        pack_d[i] <= pack_d[i-1];
      end
    end
  end
  logic prod_fresh, prod_more, prod_tick;
  logic [2:0] prod_pack;
  assign prod_fresh = fresh_d[MUL_LAT-1];
  assign prod_more  = more_d[MUL_LAT-1];
  assign prod_tick  = v_d[MUL_LAT-1];
  assign prod_pack  = pack_d[MUL_LAT-1];

  logic [ACC_W-1:0] prod, sum;
  logic prod_v, sum_v;  // prod_v: a counted product, only those reach the multiplier

  logic [ACC_W-1:0] acc[BANKS][U];
  logic [U-1:0] wr_mask[BANKS];  // per bank: slots a counted product reached this set
  logic [BW-1:0] cur;  // bank the next product joins
  logic [SW-1:0] slot;  // partial sum within it
  logic [CW-1:0] n_prod;  // products of the pass seen, counted or not
  logic [CW-1:0] u_cnt;  // counted products of the pass
  logic [BANKS-1:0] taken;  // all K products of the bank issued, not yet released

  // The first counted product into a slot of a new set is added to zero, so a reused bank needs no clear; later passes add on.
  logic [ACC_W-1:0] add_a;
  assign add_a = (prod_fresh && u_cnt < CW'(U)) ? '0 : acc[cur][slot];

  // The multiplier and adder in the build's format; int8 multiplies exactly into int16 and accumulates in int32, wrapping.
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "ProcessingElement: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC_W
    $fatal(1, "ProcessingElement: ACC_W=%0d is not sienna_fmt_pkg::acc_w(%0d, %0d)", ACC_W, EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i), .result_o(prod), .done_o(prod_v),
                        .overflow_o(), .underflow_o(), .invalid_o());
    fp32Adder ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum), .done_o(sum_v),
                   .overflow_o(), .underflow_o(), .invalid_o());
  end else if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT
    logic [2*DATA_WIDTH-1:0] prod_w;  // the full signed product
    intMultiplier #(.W(DATA_WIDTH)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i), .result_o(prod_w),
        .done_o(prod_v));
    assign prod = {{(ACC_W - 2 * DATA_WIDTH){prod_w[2*DATA_WIDTH-1]}}, prod_w};  // sign-extended to the accumulator
    intAdder #(.W(ACC_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum),
        .done_o(sum_v));
  end else begin : G_FP
    fpMultiplier #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i),
        .result_o(prod), .done_o(prod_v), .overflow_o(), .underflow_o(), .invalid_o());
    fpAdder #(.EXP_W(EXP_W), .MAN_W(MAN_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod),
        .result_o(sum), .done_o(sum_v), .overflow_o(), .underflow_o(), .invalid_o());
  end

  // Where each add in flight writes back, and whether it is still in flight.
  logic [BW-1:0] bank_dly[ADD_LAT];
  logic [SW-1:0] slot_dly[ADD_LAT];
  logic          v_dly   [ADD_LAT];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < ADD_LAT; i++) begin
        bank_dly[i] <= '0;
        slot_dly[i] <= '0;
        v_dly[i]    <= 1'b0;
      end
    end else begin
      bank_dly[0] <= cur;
      slot_dly[0] <= slot;
      v_dly[0]    <= prod_v;
      for (int i = 1; i < ADD_LAT; i++) begin
        bank_dly[i] <= bank_dly[i-1];
        slot_dly[i] <= slot_dly[i-1];
        v_dly[i]    <= v_dly[i-1];
      end
    end
  end

  logic [BANKS-1:0] pending;  // an add into the bank is issuing or in flight
  always_comb begin
    pending = '0;
    if (prod_v) pending[cur] = 1'b1;
    for (int i = 0; i < ADD_LAT; i++) if (v_dly[i]) pending[bank_dly[i]] = 1'b1;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      cur     <= '0;
      slot    <= '0;
      n_prod  <= '0;
      u_cnt   <= '0;
      taken   <= '0;
      final_o <= '0;
      for (int b = 0; b < BANKS; b++) begin
        wr_mask[b] <= '0;
        for (int u = 0; u < U; u++) acc[b][u] <= '0;
      end
    end else begin
      if (sum_v) acc[bank_dly[ADD_LAT-1]][slot_dly[ADD_LAT-1]] <= sum;
      // A set's first product clears its bank's mask; each slot a counted product of its first pass reaches is marked.
      if (prod_tick && n_prod == '0 && prod_fresh) wr_mask[cur] <= prod_v ? (U'(1) << slot) : '0;
      else if (prod_v && prod_fresh && u_cnt < CW'(U)) wr_mask[cur][slot] <= 1'b1;
      if (prod_v) begin
        slot  <= (slot == SW'(U - 1)) ? '0 : slot + 1'b1;
        u_cnt <= u_cnt + 1'b1;
      end
      if (prod_tick) begin
        if (n_prod == CW'(K - 1)) begin
          n_prod <= '0;
          u_cnt  <= '0;
          if (!prod_more) begin  // the set's last pass; a continuing set keeps its bank and the slot keeps turning
            taken[cur] <= 1'b1;
            cur        <= (cur == BW'(BANKS - 1)) ? '0 : cur + 1'b1;
            slot       <= '0;
          end
        end else n_prod <= n_prod + 1'b1;
      end
      for (int b = 0; b < BANKS; b++) if (taken[b] && !pending[b]) final_o[b] <= 1'b1;
      if (release_i) begin
        taken[rd_bank_i]   <= 1'b0;
        final_o[rd_bank_i] <= 1'b0;
      end
    end
  end

  always_comb for (int u = 0; u < U; u++) partial_o[u] = wr_mask[rd_bank_i][u] ? acc[rd_bank_i][u] : '0;

`ifndef SYNTHESIS
  a_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_tick && n_prod == '0) |-> !taken[cur])
    else $error("ProcessingElement: a set started in bank %0d before the reader released it", cur);
  a_flags_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) prod_v == (v_d[MUL_LAT-1] && inb_d[MUL_LAT-1]))
    else $error("ProcessingElement: the pass flags are out of step with the multiplier");
  a_fresh_slot0: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_v && u_cnt == '0 && prod_fresh) |-> slot == '0)
    else $error("ProcessingElement: a new set started part way through a bank's slots");
  a_release_final: assert property (@(posedge clk_i) disable iff (!rstn_i) release_i |-> final_o[rd_bank_i])
    else $error("ProcessingElement: bank %0d released before its set was final", rd_bank_i);
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) v_i |-> int'(pack_i) < LGK)
    else $error("ProcessingElement: pack shift %0d leaves blocks narrower than 2 of K=%0d", pack_i, K);
  a_pack_count: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                 (prod_tick && n_prod == CW'(K - 1)) |-> (int'(u_cnt) + int'(prod_v)) == (K >> int'(prod_pack)))
    else $error("ProcessingElement: a pass counted %0d products, its block holds %0d", int'(u_cnt) + int'(prod_v),
                K >> int'(prod_pack));
`endif

endmodule
```

- [ ] **Step 4: Run the PE bench and the existing PE and array benches**

```bash
$J/cmds/int8_tree.sh pk_pe 16 1 make -C SystolicMesh verilator TOP_MODULE=TB_PE_pack
$J/cmds/int8_tree.sh pk_pe_int8 16 1 make -C SystolicMesh verilator TOP_MODULE=TB_PE_int8
```
Expected: `pk_pe` prints `TB_PE_pack: fp32 720 slots 0 errors, bf16 720 slots 0 errors, int8 180 slots 0 errors` and
`RESULT: PASSED`, no assertion message; `pk_pe_int8` prints `RESULT: PASSED` (TB_PE_int8 leaves `pack_i` unconnected:
if Verilator rejects that, connect `.pack_i(3'b0), .pack_o()` in TB_PE_int8 and say so in the report). `TB_SystolicArray`
is updated in Task 3, where SystolicArray gains the port.

- [ ] **Step 5: Commit** (do not push yet: SystolicArray does not build until Task 3)

```bash
git add src/engine/ProcessingElement.sv && git commit -m "ProcessingElement: packed sets. pack_i travels with the pass; a product counts only when it lies in this PE's column block of K >> pack (COL parameter), so out-of-block products feed neither multiplier nor adder, the slot rotation advances on counted products only, and a slot no counted product reached reads +0 (wr_mask). pack 0 counts every product, as before. Assertions: pack range, and a pass counts exactly its block's products."
git add testbenches/TB_PE_pack.sv && git commit -m "TB_PE_pack: packed PEs in fp32, bf16 (U = 6) and int8 (U = 2) at spread columns, every shift: each slot holds only its block's products, the r-th in slot r mod U, slots past the block +0, multiplies issued equal the block width, pack_o follows pack_i."
```

---

### Task 3: The pack shift through the array and the mesh; packed wide read; mesh regression

**Files:**
- Modify: `SystolicMesh/src/top/SystolicArray.sv`, `SystolicMesh/src/top/SystolicMesh.sv`
- Modify: `SystolicMesh/stim_format.py:84-121`, `SystolicMesh/matmul_tests.py`, `SystolicMesh/regression.py:404-416`,
  `SystolicMesh/testbenches/TB_SystolicMesh.sv`, `SystolicMesh/testbenches/TB_SystolicArray.sv`
- Create (outside git): `$J/cmds/pk_cmp_mesh.py`

**Interfaces:**
- Consumes: Task 2's PE ports; Task 1's `matmul_packed`.
- Produces: `SystolicMesh` ports `input logic [2:0] pack_shift_i` (sampled with the start, like `bias_i`) and
  `input logic wide_read_packed_i` (word k of a wide read is element `((k / N) * (N*N/WIDE_READ) + index) * N + k % N`);
  `SystolicArray` parameter `COL0`, port `input logic [2:0] commit_pack_i`; `stim_format.write_set(..., pack=0)` writes
  `packShift<suffix>.mem` (one hex word) when pack != 0 and removes a stale one otherwise.

- [ ] **Step 0: Check the packed read against the result memory's synthesis plan**

`.claude/skills/sienna-report/history/2026-09-27_synthesis_readiness.txt` (items 2b and the memory table) records
MeshOutputSram as flops with one NUM_LANES-word wide read of arbitrary addresses (32 read muxes at N = 16), and proposes
banking it per tile with a small read crossbar later. The packed mode adds a second address formula into the same muxes,
so it fits the current memory. Write one ledger line saying so, and that a future per-tile banking must route the packed
pattern (per beat: NUM_LANES/N runs of N consecutive words, one row each) through its crossbar. If the file says
otherwise when you read it, stop and report before Step 3.

- [ ] **Step 1: Write the failing mesh tests**

In `SystolicMesh/stim_format.py`, change `write_set`'s signature and body:

```python
def write_set(A, B, stim_dir, suffix="", bias=None, pack=0):
    """Write matrixA/B/C<suffix>.mem for one set; returns C as floats (int8: as int64 values, with bias, also in matrixBias<suffix>.mem).
    pack != 0: a packed set (b = N >> pack columns per job), packShift<suffix>.mem holds the shift and C sums each block alone."""
```
and, right after the two `np.vstack` padding lines, add:

```python
    pf = os.path.join(stim_dir, f"packShift{suffix}.mem")
    if pack:
        with open(pf, "w") as fh:
            fh.write(f"{int(pack):x}\n")
    elif os.path.exists(pf):
        os.remove(pf)  # a stale shift would pack this set
```
In the int8 branch replace `Ci = mesh_model.matmul_int([(Ai, Bi)], N, bias)` with:

```python
        b = N >> pack
        Bm = Bi * np.kron(np.eye(N // b, dtype=np.int64), np.ones((b, b), np.int64)) if pack else Bi  # the skip ignores off-block weights
        Ci = mesh_model.matmul_int([(Ai, Bm)], N, bias)
```
and in the float branch replace `Cb = mesh_model.matmul(fpu.FORMATS[FORMAT], [(Ab, Bb)], N, TILE, COLLAPSE_K)` with:

```python
    f = fpu.FORMATS[FORMAT]
    Cb = mesh_model.matmul_packed(f, Ab, Bb, N, pack) if pack else mesh_model.matmul(f, [(Ab, Bb)], N, TILE, COLLAPSE_K)
```

In `SystolicMesh/matmul_tests.py`, change `_write_set` to pass a pack shift, and add the generators and catalogue rows:

```python
def _write_set(A: np.ndarray, B: np.ndarray,
               stim_dir: str, suffix: str = "", pack: int = 0) -> None:
    """Compute C = A @ B in the stimulus format and write its .mem files for one set (int8: with the set's bias; pack: a packed set)."""
    bias = stim_format.int8_bias(B.shape[1], suffix) if stim_format.is_int() else None
    stim_format.write_set(A, B, stim_dir, suffix, bias, pack)


def _pack_shifts(N: int) -> list:
    """Per set: packed, unpacked between packed sets, b = 2 (fewer columns than the PE's 6 slots), b = N / 4, packed again."""
    lg = N.bit_length() - 1
    return [1, 0, lg - 1, min(2, lg - 1), 1]


def _packed_sets(stim_dir: str, N: int, garbage: bool, seed0: int) -> int:
    for s, sh in enumerate(_pack_shifts(N)):
        _seed(seed0 + s)
        b = N >> sh
        A = stim_format.rand(-1, 1, (N, N)).astype(np.float32)
        B = stim_format.rand(-1, 1, (N, N)).astype(np.float32)
        if sh and not garbage:
            B = np.where(np.kron(np.eye(N // b), np.ones((b, b))).astype(bool), B, np.float32(0.0))
        _write_set(A, B, stim_dir, f"_{s}", sh)
    return MATMUL_NUM_SETS


def gen_mm_packed(stim_dir: str, N: int) -> int:
    """Block-diagonal B, a pack shift per set with an unpacked set between: every block sums exactly as its job alone."""
    return _packed_sets(stim_dir, N, False, 7300)


def gen_mm_packed_garbage(stim_dir: str, N: int) -> int:
    """Packed sets whose off-block weights are random, not zero: the PEs must ignore them."""
    return _packed_sets(stim_dir, N, True, 7400)
```
Append to `MATMUL_TESTS`:

```python
    dict(name="mm_packed",       description="Packed sets, a shift per set  (block == job alone)", gen_fn=gen_mm_packed, packed=True),
    dict(name="mm_packed_garbage", description="Packed sets, random off-block weights  (skipped)", gen_fn=gen_mm_packed_garbage,
         packed=True),
```

In `SystolicMesh/regression.py` `main()`, after `run_mm, run_conv` are chosen, add:

```python
    run_mm = [t for t in run_mm if COLLAPSE or not t.get("packed")]  # packing is collapse-k only (the mesh asserts it)
```

In `SystolicMesh/testbenches/TB_SystolicMesh.sv`: add `reg [2:0] pack_d = '0;  // the set's pack shift (packShift<suffix>.mem), taken with the start`
next to `bias_d`; connect `.pack_shift_i(pack_d),` after `.bias_i(bias_d),` and `.wide_read_packed_i(1'b0),` after
`.wide_read_enable_i(1'b0),`; add the task below after `drive_bias`, and call `drive_pack(<same index>);` on the line
after every `drive_bias(...)` call (execute_test_set, the stream producer, and the three in staging_overrun_test):

```systemverilog
  // ── Per-set pack shift: packShift<suffix>.mem exists only for packed sets ─────────────────────────────────
  task automatic drive_pack(input int s);
    string f;
    integer fh, res;
    reg [31:0] tmp;
    f = (NUM_TEST_SETS == 1) ? "packShift.mem" : $sformatf("packShift_%0d.mem", s);
    pack_d = '0;
    fh = $fopen(f, "r");
    if (fh) begin
      res = $fscanf(fh, "%h", tmp);
      pack_d = tmp[2:0];
      $fclose(fh);
    end
  endtask
```

- [ ] **Step 2: Run the mesh regression to verify the packed tests fail**

```bash
$J/cmds/int8_tree.sh pk_sm_red 32 4 bash $J/cmds/mesh_notrace.sh reg 16 fp32 1 --tiles 4
```
Expected: a Verilator error that `SystolicMesh` has no port `pack_shift_i` (or, if the RTL steps were done first by
mistake, `mm_packed` FAIL). Every pre-existing test's row is what matters later.

- [ ] **Step 3: Implement the array plumbing** (`SystolicMesh/src/top/SystolicArray.sv`)

Add the parameter after `U`: `    parameter int COL0        = 0   // mesh column of this tile's column 0, for packed blocks`
(and the comma on the `U` line). Add the port after `commit_more_i`:
`    input logic [2:0]                             commit_pack_i,   // with commit_i: the pass's pack shift`.
Declare `logic [2:0] ob_pack[2];` beside `ob_fresh, ob_more`, `logic [2:0] cmd_p[N];` beside `cmd_m`,
`logic [2:0] p_feed[N];` beside `m_feed`, and `logic [2:0] p_w[N][N+1];  // pack shift travels east with A` beside `v_w`.
In the reset branch add `ob_pack[0] <= '0; ob_pack[1] <= '0;` and `cmd_p[r] <= '0;`; in `if (commit_i)` add
`ob_pack[lb] <= commit_pack_i;`; after the `cmd_m[0]` line add `cmd_p[0] <= launch ? ob_pack[fb] : ob_pack[ob_cur];`; in the
shift loop add `cmd_p[r] <= cmd_p[r-1];`; in the feed register block add `p_feed[r] <= '0;` (reset) and
`p_feed[r] <= cmd_p[r];`; in `FEED` add `assign p_w[r][0] = p_feed[r];`; and in the PE instance add `.COL(COL0 + c)` to the
parameters and `.pack_i(p_w[r][c]),` / `.pack_o(p_w[r][c+1]),` to the ports.

- [ ] **Step 4: Implement the mesh plumbing** (`SystolicMesh/src/top/SystolicMesh.sv`)

Ports, after `bias_i`:
```systemverilog
    input logic [2:0]                            pack_shift_i,  // with the start: a packed set, b = N >> pack_shift_i columns per job; 0 unpacked
```
and after `wide_read_index_i`:
```systemverilog
    input  logic                                 wide_read_packed_i,  // the oldest result is packed: word k is column k % N, rows (k / N) * stride + index
```
State: `logic [2:0] in_pack[2];  // per staging bank: the set's pack shift` beside `in_tile`; reset `in_pack[0] <= '0; in_pack[1] <= '0;`;
in `if (start_accept)` add `in_pack[in_wr] <= pack_shift_i;`. Beside `commit_more_q`: `logic [2:0] commit_pack_q;`, reset
`commit_pack_q <= '0;`, and `commit_pack_q <= in_pack[in_rd];` next to `commit_more_q <= in_more[in_rd];`.
SystolicArray instance: add `.COL0(j * TILE_SIZE)` to the parameters and `.commit_pack_i(commit_pack_q),` after
`.commit_more_i(commit_more_q),`. Wide address, replacing the existing `always_comb` loop body:

```systemverilog
  always_comb
    for (int k = 0; k < WIDE_READ; k++)
      wide_addr[k] = int'(out_rd) * GLOBAL_ELEMENTS + (wide_read_packed_i
                     ? ((k / MATRIX_SIZE) * WIDE_STRIDE + int'(wide_read_index_i)) * MATRIX_SIZE + k % MATRIX_SIZE
                     : k * WIDE_STRIDE + int'(wide_read_index_i));
```
Assertions, after `a_wide_read_outstanding`:

```systemverilog
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) start_accept |-> int'(pack_shift_i) < $clog2(MATRIX_SIZE))
    else $error("SystolicMesh: pack shift %0d leaves blocks narrower than 2 of N=%0d", pack_shift_i, MATRIX_SIZE);
  a_pack_collapsed: assert property (@(posedge clk_i) disable iff (!rstn_i) (start_accept && pack_shift_i != 0) |-> COLLAPSE_K != 0)
    else $error("SystolicMesh: a packed set on the collapse-k 0 mesh");
  a_pack_one_pass: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                    (start_accept && pack_shift_i != 0) |-> (!partial_i && !last_partial))
    else $error("SystolicMesh: a packed set is part of an accumulated sum");
  a_wide_packed: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                  (wide_read_enable_i && wide_read_packed_i) |-> (WIDE_READ % MATRIX_SIZE == 0))
    else $error("SystolicMesh: a packed wide read needs N (%0d) to divide WIDE_READ (%0d)", MATRIX_SIZE, WIDE_READ);
```
`TB_SystolicArray.sv`: connect `.commit_pack_i(3'b0),` beside `commit_more_i`.

- [ ] **Step 5: Write the comparison tool** (`/proj/work/spramanik/sienna_jobs/cmds/pk_cmp_mesh.py`, outside git)

```python
#!/usr/bin/env python3
"""Mesh readiness reports of two runs, row by row: every reference row's result and average cycles must match; args: REF NEW."""
import glob
import re
import sys

R = "/proj/work/spramanik/sienna_jobs/runs"


def rows(run):
    out = {}
    for f in glob.glob(f"{R}/{run}/mesh_results/readiness/readiness_report_*.log"):
        for ln in open(f):
            m = re.match(r"(\S+)\s+T=(\d+)\s+(\S+)\s+(\d+/\d+)\s+[\d.]+%\s+[\d.]+%\s+(\d+)\s+(\d+)", ln)
            if m:
                out[(m[1], int(m[2]))] = (m[3], int(m[6]))
    return out


a, b = rows(sys.argv[1]), rows(sys.argv[2])
bad = [f"{k}: {a[k]} -> {b.get(k)}" for k in sorted(a) if b.get(k) != a[k]]
new = sorted(k for k in b if k not in a)
print(f"{len(a)} reference rows, {len(b)} new rows, {len(bad)} differ; only in the new run: {new}")
for x in bad:
    print("DIFF", x)
print("IDENTICAL" if a and not bad else "DIFFERENT")
sys.exit(0 if a and not bad else 1)
```

- [ ] **Step 6: Run the mesh regression at every N <= 32, every T, all formats, both mesh modes at N = 16**

Baselines: `ls $J/runs | command grep -E '^(mg_mesh|g3i_N(8|16|32)_)'` lists the pre-packing runs (same RTL as
`pre_packing_v1`). Launch (one job per line; log each to `pk_launch.log`):

```bash
for f in fp32 bf16 int8; do
  $J/cmds/int8_tree.sh pk_sm8_${f} 32 4 bash $J/cmds/mesh_notrace.sh reg 8 $f 1
  $J/cmds/int8_tree.sh pk_sm16_${f}_ck1 32 4 bash $J/cmds/mesh_notrace.sh reg 16 $f 1
  $J/cmds/int8_tree.sh pk_sm16_${f}_ck0 32 4 bash $J/cmds/mesh_notrace.sh reg 16 $f 0
  for t in 2 4 8 16 32; do $J/cmds/int8_tree.sh pk_sm32_${f}_T$t 64 8 bash $J/cmds/mesh_notrace_O0.sh reg 32 $f 1 --tiles $t; done
done
```
Expected: every run's readiness report `RESULT : READY`; `mm_packed` and `mm_packed_garbage` PASS at every collapse-k 1
tile; absent from the ck0 runs. `python3 $J/cmds/pk_cmp_mesh.py <baseline> <new>` prints `IDENTICAL` for every pair
that has a baseline (mg_mesh16_* for N = 16, mg_mesh32_*_ck1_T4 for N = 32 T = 4, g3i_* for the rest). A run without a
baseline is reported as such, never skipped silently.

- [ ] **Step 7: Commit and push SystolicMesh**

```bash
git add src/top/SystolicArray.sv src/top/SystolicMesh.sv && git commit -m "SystolicArray, SystolicMesh: the pack shift is taken with the start, held per staging bank and per operand bank, and travels east with A to every PE, whose column is COL0 + c; wide_read_packed_i reads a packed result column-wise (word k takes column k % N, so a consumer lane gets one column block); assertions refuse a packed set on the collapse-k 0 mesh, in an accumulated sum, narrower than 2 columns, or a packed wide read unless N divides WIDE_READ."
git add stim_format.py matmul_tests.py regression.py testbenches/TB_SystolicMesh.sv testbenches/TB_SystolicArray.sv && git commit -m "Mesh regression: mm_packed and mm_packed_garbage stream sets with a pack shift each (one unpacked between, b = 2 and b = N / 4 included) and expect each block exactly as its job alone (mesh_model.matmul_packed; int8 masks the off-block weights); packShift<suffix>.mem carries the shift and is removed by every unpacked writer; collapse-k 0 runs skip them."
git log --format=%B origin/packing..HEAD | command grep -ci co-authored-by   # must print 0
git push origin packing
```
Then in SIENNA: `git add SystolicMesh && git commit -m "SystolicMesh: packing branch (PE out-of-block skip, pack shift per set, packed wide read, packed mesh tests)." && git push origin packing`.

---

### Task 4: Requantize per lane, with the packed channel map

**Files:**
- Modify: `src/requant_lanes.sv`, `testbenches/TB_requant_lanes.sv`, `testbenches/gen_rq_lanes.py`

**Interfaces:**
- Produces: `requant_lanes` ports `input logic packed_i` (lane k's channel is `k % N`, else `(k*PER_LANE + beat) % N`),
  `input logic [NUM_LANES-1:0][7:0] zp_i, min_i, max_i` (per lane).

- [ ] **Step 1: Write the failing test.** In `gen_rq_lanes.py`, make odd sets packed with per-lane parameters. Replace the
  per-set header and expectation (the lines from `c = (np.arange...` to `out += [zp, amin, amax] + ...`) with:

```python
        packed = s % 2 == 1  # odd sets: lane k takes column k % N, and its block's zero point and clamp
        c = (np.arange(LANES)[None, :] % N + 0 * np.arange(PER)[:, None]) if packed else \
            (np.arange(LANES)[None, :] * PER + np.arange(PER)[:, None]) % N  # channel of lane k at beat b
        if packed:
            ents = []
            for _ in range(4):  # zero point, then a clamp low <= high
                z = int(rng.randint(-128, 128))
                lo_, hi_ = sorted(int(v) for v in rng.randint(-128, 128, 2))
                ents.append((z, lo_, hi_))
            blk = (np.arange(LANES) % N) // (N // 4)  # four blocks of N / 4 columns
            zpL = np.array([ents[e][0] for e in blk]); aminL = np.array([ents[e][1] for e in blk]); amaxL = np.array([ents[e][2] for e in blk])
        else:
            zpL, aminL, amaxL = np.full(LANES, zp), np.full(LANES, amin), np.full(LANES, amax)
        srng, found = np.random.RandomState(1900 + s), []
        for ch in (np.arange(N) + 5 * s) % N:
            k0 = int(ch)  # packed: lane ch reads channel ch at every beat
            a = separating(srng, int(mult[ch]), int(shift[ch]), int(zpL[k0]), int(aminL[k0]), int(amaxL[k0]))
            if a is None:
                continue
            if packed:
                b, k = len(found) % PER, int(ch) + N * (len(found) % (LANES // N))
            else:
                b, k = int(ch) % PER, 8 + 2 * len(found) + int(ch) // PER
            assert c[b, k] == ch
            acc[b, k] = a
            found.append(int(ch))
            if len(found) == SEP_SLOTS:
                break
        if not found:
            raise RuntimeError(f"set {s}: no channel has a small accumulator that separates DOUBLE from SINGLE")
        print(f"set {s}{' (packed)' if packed else ''}: DOUBLE/SINGLE separating accumulators on channels {found}")
        want = np.stack([ipu.requant(acc[:, k], mult[c[:, k]], shift[c[:, k]], int(zpL[k]), int(aminL[k]), int(amaxL[k]),
                                     tflite_ref.ROUNDING) for k in range(LANES)], axis=1)
        out += [int(packed)] + zpL.tolist() + aminL.tolist() + amaxL.tolist() + mult.tolist() + shift.tolist()
```
In `TB_requant_lanes.sv`: add `logic packed_i = 0;`, make `zp_i, min_i, max_i` `logic [NUM_LANES-1:0][7:0]`; the header of a
set is now `1 + 3 * NUM_LANES + 2 * N` words: replace both `p += 3 + 2 * N;` with `p += 1 + 3 * NUM_LANES + 2 * N;`, and the
header read with

```systemverilog
      packed_i = v[p][0];
      for (int k = 0; k < NUM_LANES; k++) begin
        zp_i[k]  = v[p+1+k][7:0];
        min_i[k] = v[p+1+NUM_LANES+k][7:0];
        max_i[k] = v[p+1+2*NUM_LANES+k][7:0];
      end
      for (int c = 0; c < N; c++) begin
        mult_i[c]  = v[p+1+3*NUM_LANES+c];
        shift_i[c] = v[p+1+3*NUM_LANES+N+c][7:0];
      end
```
and update its header comment to `per-lane zp/min/max; odd sets packed (lane k reads channel k % N)`.

- [ ] **Step 2: Run to verify it fails**

`$J/cmds/int8_tree.sh pk_rq_red 16 1 bash $J/cmds/int8_rq_lanes.sh` → Verilator error (`packed_i` not a port), or
`RESULT: FAILED` on the packed sets.

- [ ] **Step 3: Implement** (`src/requant_lanes.sv`): header comment `// int8: one tfliteRequant per lane on the wide read; lane k's channel is (k*PER_LANE + b) % N, or k % N for a packed set; zero point and clamp per lane.`;
  ports `input logic packed_i,  // the stage's set is packed: lane k reads column k % N at every beat` and
  `input logic [NUM_LANES-1:0][7:0] zp_i, min_i, max_i` (one comment each, as now); channel
  `assign ch = packed_i ? CW'(k % N) : CW'((k * PER_LANE + int'(beat)) % N);`; instance `.zp_i(zp_i[k]), .act_min_i(min_i[k]), .act_max_i(max_i[k])`.

- [ ] **Step 4: Run** `pk_rq` (same command) → `TB_requant_lanes: 48 beats, 0 errors, 0 latency errors`, `RESULT: PASSED`.
  Run the three mutation checks `bash $J/cmds/int8_rq_mut.sh beat|clear|single` the same way: each must report the unit
  test FAILED; if a mutation's sed pattern no longer matches the new line, fix the pattern in that script (outside git)
  and report it.

- [ ] **Step 5: Commit** (sienna_top does not build until Task 5: do not push yet)

```bash
git add src/requant_lanes.sv && git commit -m "requant_lanes: zero point and clamp per lane, and packed_i, under which lane k reads channel k % N at every beat (a packed set's column order)."
git add testbenches/TB_requant_lanes.sv testbenches/gen_rq_lanes.py && git commit -m "TB_requant_lanes: per-lane zero points and clamps; odd sets packed, with four column blocks of their own parameters and the DOUBLE/SINGLE separating accumulators placed on the packed channel map."
```

---

### Task 5: sienna_top runs packed sets

**Files:**
- Modify: `src/sienna_top.sv`, `src/sienna_multi.sv`, `testbenches/TB_sienna_model.sv`, `testbenches/TB_sienna_top.sv`,
  `src/sienna_layer.sv` (tie-off only; Task 6 wires it), `regression.py`

**Interfaces:**
- Consumes: Task 3's mesh ports, Task 4's requant_lanes ports.
- Produces: sienna_top parameter `PACK_ENTRIES = 8` and ports `pack_shift_i[2:0]`, `pack_map_i[N/2-1:0][PEW-1:0]`,
  `pack_act_i[PACK_ENTRIES-1:1]`, `pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i [PACK_ENTRIES-1:1][7:0]`,
  `pack_mx_i[...][15:0]`, `pack_shx_i[...][4:0]`, `pack_mout_i[...][31:0]`; `regression.PACK_ENTRIES`, `pack_shift_of(N, k)`,
  `pack_map_of(N, sh, k)`; per-set file `pack_<k>.mem` (shift, N/2 map words, then entries 1..7 as act, zp, min, max, mx,
  shx, mout, shout, zout); package item `PACKED`.

- [ ] **Step 1: Write the failing tests** (`regression.py`)

Add after `_golden_bits` (and make `_golden_bits` end with `return C, A, P, _dropout_bits(P, cfg, drop_seed, f)`, deleting
its own dropout lines; behavior is unchanged):

```python
def _dropout_bits(P, cfg: dict, drop_seed: int, f) -> np.ndarray:
    """Dropout of a set's pooled bits as the lanes apply it: scaled by 1/(1-p) where kept, a zero with the input's sign where dropped."""
    if not cfg.get("training", False):
        return P.copy()
    flat = P.flatten()
    keep = dropout_keep(flat.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    scale = fpu.from_fp32(int(np.float32(1.0 / (1.0 - cfg.get("dropout_p", 0.5))).view(np.uint32)), f.m)
    prod = fpu.mul(f, flat, np.full_like(flat, scale))[0]
    return np.where(keep, prod, flat & (1 << (f.w - 1))).reshape(P.shape)
```
Add the packed generator before `generate_vectors`, and make `generate_vectors` start with
`if cfg.get("packed"): return _generate_vectors_packed(cfg)`:

```python
PACK_ENTRIES = 8  # sienna_top's PACK_ENTRIES; entry 0 is the per-set ports


def pack_shift_of(N: int, k: int) -> int:
    """Set k's pack shift in the packed tests: packed, unpacked, b = 2 (fewer columns than the PE's slots), b = N / 4."""
    lg = N.bit_length() - 1
    return [1, 0, lg - 1, min(2, lg - 1)][k % 4]


def pack_map_of(N: int, sh: int, k: int) -> list:
    """Entry of each of the N/2 block slots: the set's 2^sh blocks rotate through the entries; an unpacked set uses entry 0."""
    return [((c + k) % PACK_ENTRIES if sh and c < (1 << sh) else 0) for c in range(N // 2)]


def _write_pack(path: str, sh: int, mp: list, ents: list) -> None:
    """pack_<k>.mem: the shift, N/2 map words, then entries 1..7 as act, zp, min, max, mx, shx, mout, shout, zout."""
    words = [sh] + list(mp)
    for act, rq in ents[1:]:
        q = rq or {}
        words += [activation_to_code(act)] + [int(q.get(x, 0)) for x in ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")]
    _write_w32(path, np.array(words, np.int64))


def _packed_float(cfg, k, A, B, bias, sh, col_ent, acts, drop):
    """One packed float set's files; the golden is each column's block through its entry's activation, then dropout."""
    N, fmt = cfg.get("n", 16), cfg.get("fmt_name", "fp32")
    A, B = op_round(A, fmt), op_round(B, fmt)
    b = op_round(bias, fmt) if bias is not None else np.zeros(N, np.float32)
    write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), A, fmt)
    write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), B, fmt)
    if fmt == "fp32":
        write_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), b)
        C = (_ref_matmul(A, B) + b).astype(np.float32)
        S = np.abs(A).astype(np.float64) @ np.abs(B).astype(np.float64) + np.abs(b)
        Ca, Bnd = np.zeros_like(C), np.zeros_like(C)
        for e in sorted(set(col_ent.tolist())):
            cols = col_ent == e
            Ca[:, cols] = apply_activation(C, acts[e])[:, cols]
            Bnd[:, cols] = fp32_error_bound(S, N, cfg, acts[e], C)[:, cols]
        F = apply_dropout(Ca, cfg.get("dropout_p", 0.5), cfg.get("training", False), drop, cfg.get("lanes", 32))
        write_mem(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F)
        write_mem(os.path.join(TB_DIR, f"bound_output_{k}.mem"), Bnd)
        return
    f = fpu.FORMATS[fmt]
    write_op_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), b, fmt)
    Ab, Bb = fmt_bits(A, fmt), fmt_bits(B, fmt)
    bb = fmt_bits(b, fmt) if bias is not None else None
    C = mesh_model.matmul_packed(f, Ab, Bb, N, sh, bb) if sh else mesh_model.matmul(f, [(Ab, Bb)], N, cfg.get("tile_size", 4), 1, bb)
    lane = gpnae_model.Lane(f, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f))))
    Aout = np.zeros_like(C)
    for e in sorted(set(col_ent.tolist())):
        cols = col_ent == e
        Aout[:, cols] = lane.run(C, activation_to_code(acts[e]))[:, cols]
    F = _dropout_bits(Aout, cfg, drop, f)
    write_bits(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F, fmt)
    write_bits(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(F), fmt)


def _packed_int8(cfg, k, A, B, bias, sh, col_ent, acts, drop, req_rng, zqs):
    """One packed int8 set: one input scale for the set, weights per column; each entry's requantize and lane parameters
    come from the accumulators of the columns that use it; dropout drops each column to its entry's zero point (D-5)."""
    N = cfg.get("n", 16)
    A_q, s_a, z_a = quant_act(A)
    B_q, s_w = quant_weights(B)
    hw_bias = fold_bias(bias, s_a, s_w, z_a, B_q)
    acc = wrap32(imatmul(A_q, B_q) + hw_bias[None, :])
    mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
    ents = [(acts[e], None) for e in range(PACK_ENTRIES)]
    Y, dzp = np.zeros_like(acc), np.zeros(N, np.int64)
    for e in sorted(set(col_ent.tolist())):
        cols = col_ent == e
        rq = requant_params(acc[:, cols], s_a, s_w[cols], acts[e], req_rng, zqs[e])
        mult[cols], shift[cols] = rq["mult"], rq["shift"]
        ents[e] = (acts[e], rq)
        Y[:, cols] = activate_int8(requantize(acc[:, cols], rq), acts[e], rq)
        dzp[cols] = drop_zp(acts[e], rq)
    if cfg.get("training", False):
        keep = dropout_keep(N * N, cfg.get("dropout_p", 0.5), drop, cfg.get("lanes", 32)).reshape(N, N)
        Y = np.where(keep, Y, dzp[None, :])
    head = ents[0][1] or requant_params(acc, s_a, s_w, acts[0], req_rng, zqs[0])  # entry 0 rides on the per-set ports
    write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), A_q, "int8")
    write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), B_q, "int8")
    _write_w32(os.path.join(TB_DIR, f"bias_{k}.mem"), hw_bias)
    _write_w32(os.path.join(TB_DIR, f"requant_{k}.mem"), _requant_words(dict(head, mult=mult, shift=shift)))
    _write_s8(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Y)
    _write_s8(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(Y))
    return ents


def _generate_vectors_packed(cfg: dict) -> None:
    """Packed sets (sienna-packing): block-diagonal B, a pack shift per set, an activation (int8: and output parameters) per
    column block from an 8-entry table; the golden is each block's job alone, assembled. Needs a 1x1 pool."""
    os.makedirs(TB_DIR, exist_ok=True)
    N, fmt, act_type, acts = cfg.get("n", 16), cfg.get("fmt_name", "fp32"), cfg["act"], cfg["pack_acts"]
    assert len(acts) == PACK_ENTRIES and acts[0] == act_type, (cfg["name"], "pack_acts[0] must be the test's act")
    assert (cfg.get("pool_h"), cfg.get("pool_w"), cfg.get("padding")) == (1, 1, 0), (cfg["name"], "packed sets need a 1x1 pool")
    if fmt == "int8":
        _check_rounding()
    seed = cfg.get("seed", 42) + int(os.environ.get("SIENNA_SEED", "0"))
    credits = cfg.get("credits", SETS_IN_FLIGHT)
    num_sets = cfg.get("num_sets", credits + 2)
    drop_seed = 0x2ACE0000 + seed
    use_bias = bool(cfg.get("bias", False))
    req_rng = np.random.RandomState(seed + 7000) if cfg.get("req_random") else None
    zqs = _zp_draws(np.random.RandomState(seed + 8000), PACK_ENTRIES, acts) if cfg.get("zp_random") else [None] * PACK_ENTRIES
    for k in range(num_sets):
        sh, rng = pack_shift_of(N, k), np.random.RandomState(seed + 1000 + k)
        mp = pack_map_of(N, sh, k)
        col_ent = np.array([mp[j // (N >> sh)] if sh else mp[0] for j in range(N)])
        b = N >> sh
        mask = np.kron(np.eye(N // b), np.ones((b, b))).astype(bool) if sh else np.ones((N, N), bool)
        A = rng.uniform(-1.0, 1.0, (N, N))
        B = np.where(mask, rng.uniform(-1.0, 1.0, (N, N)), 0.0)
        bias = rng.uniform(-1.0, 1.0, N) if use_bias else None
        drop = set_dropout_seed(drop_seed, k)
        if fmt == "int8":
            ents = _packed_int8(cfg, k, A, B, bias, sh, col_ent, acts, drop, req_rng, zqs)
        else:
            _packed_float(cfg, k, A, B, bias, sh, col_ent, acts, drop)
            ents = [(a, None) for a in acts]
        _write_pack(os.path.join(TB_DIR, f"pack_{k}.mem"), sh, mp, ents)
    for name in ("matrix_west", "matrix_north", "expected_output", "bound_output"):  # the single-set pass runs set 0
        shutil.copy(os.path.join(TB_DIR, f"{name}_0.mem"), os.path.join(TB_DIR, f"{name}.mem"))
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),
                     _config_items(cfg, fmt, act_type, num_sets, credits, 1, [], use_bias, drop_seed))
    if fmt != "fp32":
        _check_mem_widths(fmt, num_sets)
```
(add `import shutil` at the top if absent.) In `_config_items` add `("PACKED", int(bool(cfg.get("packed", False))), "int"),`
after `MIXED_ACTS`. Append to `PIPELINE_TESTS`:

```python
    # sienna-packing: shifts 1, 0, b = 2, b = N / 4 per set, an 8-entry table of activations rotating over the blocks
    {"name": "packed_mixed_act_nopool", "mode": "matmul", "matrix_type": "random", "act": "tanh", "pool_h": 1, "pool_w": 1,
     "padding": 0, "packed": True, "pack_acts": ["tanh", "relu", "selu", "linear", "sigmoid", "tanh", "relu", "selu"]},
    {"name": "packed_bias_cached_train_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "pool_h": 1,
     "pool_w": 1, "padding": 0, "packed": True, "bias": True, "cached": True, "training": True,
     "pack_acts": ["relu", "linear", "tanh", "sigmoid", "relu", "selu", "linear", "tanh"]},
    {"name": "packed_all_bypass_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "pool_h": 1, "pool_w": 1,
     "padding": 0, "packed": True, "pack_acts": ["relu", "linear"] * 4},  # every lane ReLU or linear: the bypass path, per lane
    {"name": "int8_packed_zp_random_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "pool_h": 1,
     "pool_w": 1, "padding": 0, "packed": True, "req_random": True, "zp_random": True, "formats": ("int8",),
     "pack_acts": ["linear", "relu", "tanh", "selu", "sigmoid", "linear", "relu", "tanh"]},
```

In `TB_sienna_top.sv`: declare beside the int8 inputs

```systemverilog
  localparam int PACK_ENTRIES = 8, PEW = 3;
  logic [2:0] pack_shift_i;  // the set's pack shift and parameter table (pack_<k>.mem), with the start
  logic [N/2-1:0][PEW-1:0] pack_map_i;
  logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] pack_act_i;
  logic [PACK_ENTRIES-1:1][7:0] pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i;
  logic [PACK_ENTRIES-1:1][15:0] pack_mx_i;
  logic [PACK_ENTRIES-1:1][4:0] pack_shx_i;
  logic [PACK_ENTRIES-1:1][31:0] pack_mout_i;
```
connect each by name in the `dut` instance (`.pack_shift_i(pack_shift_i),` ... after `.gp_zout_i`), set all to `'0` in
`reset()`, add this task after `apply_bias`, and call `apply_pack(k);` (with the same k) on the line after every
`apply_requant(...)` call (five places: stream producer, reset_mid_stream, single-set pass, back-to-back pass):

```systemverilog
  // Set k's pack shift, block map and table entries 1..7 (pack_<k>.mem); unpacked tests drive zeros.
  task automatic apply_pack(input int k);
    logic [31:0] q[$];
    pack_shift_i = '0;
    pack_map_i = '0;
    {pack_act_i, pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i, pack_mx_i, pack_shx_i, pack_mout_i} = '0;
    if (PACKED == 0) return;
    read_word_file($sformatf("pack_%0d.mem", k), q);
    pack_shift_i = q[0][2:0];
    for (int c = 0; c < N / 2; c++) pack_map_i[c] = q[1+c][PEW-1:0];
    for (int e = 1; e < PACK_ENTRIES; e++) begin
      automatic int o = 1 + N / 2 + 9 * (e - 1);
      pack_act_i[e]   = q[o][CONTROL_WIDTH-1:0];
      pack_zp_i[e]    = q[o+1][7:0];
      pack_min_i[e]   = q[o+2][7:0];
      pack_max_i[e]   = q[o+3][7:0];
      pack_mx_i[e]    = q[o+4][15:0];
      pack_shx_i[e]   = q[o+5][4:0];
      pack_mout_i[e]  = q[o+6];
      pack_shout_i[e] = q[o+7][7:0];
      pack_zout_i[e]  = q[o+8][7:0];
    end
  endtask
```

- [ ] **Step 2: Run to verify the packed tests fail**

`$J/cmds/int8_tree.sh pk_top_red 32 4 bash $J/cmds/mk.sh regression FMT=bf16 N=16 TILE=4 TEST=packed_mixed`
Expected: Verilator error, `sienna_top` has no port `pack_shift_i`.

- [ ] **Step 3: Implement sienna_top** (`src/sienna_top.sv`). Every edit below is one change; keep comments one line.

3a. Parameter, after `PADDING`'s line block (before `DROPOUT_P_PERCENT` is fine):
`    parameter int    PACK_ENTRIES      = 8,  // distinct activation / int8 output settings one packed set may mix; entry 0 is the per-set ports`.
Ports, after `gp_zout_i`:

```systemverilog
    input logic [2:0]                       pack_shift_i,  // with the start: a packed set of N >> pack_shift_i columns per job; 0 unpacked
    input logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0] pack_map_i,  // entry of each column block; an unpacked set uses block 0's
    input logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] pack_act_i,  // entries 1..: activation; entry 0 is activation_function_i
    input logic [PACK_ENTRIES-1:1][7:0]     pack_zp_i,     // int8 entries 1..: as req_zp_i, req_min_i, req_max_i
    input logic [PACK_ENTRIES-1:1][7:0]     pack_min_i,
    input logic [PACK_ENTRIES-1:1][7:0]     pack_max_i,
    input logic [PACK_ENTRIES-1:1][15:0]    pack_mx_i,     // int8 entries 1..: as gp_mx_i, gp_shx_i, gp_mout_i, gp_shout_i, gp_zout_i
    input logic [PACK_ENTRIES-1:1][4:0]     pack_shx_i,
    input logic [PACK_ENTRIES-1:1][31:0]    pack_mout_i,
    input logic [PACK_ENTRIES-1:1][7:0]     pack_shout_i,
    input logic [PACK_ENTRIES-1:1][7:0]     pack_zout_i,
```

3b. Locals after `localparam int CRW = ...`:

```systemverilog
  localparam int PEW = $clog2(PACK_ENTRIES);
  localparam int LGN = $clog2(N);
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (PACK_ENTRIES < 2 || (PACK_ENTRIES & (PACK_ENTRIES - 1)) != 0) $error("sienna_top: PACK_ENTRIES (%0d) must be a power of two >= 2", PACK_ENTRIES);
`endif
  // A packed set's lanes each take one column: N must divide the lanes; the mesh packs only collapsed; pooling must be the identity.
  function automatic logic [PEW-1:0] ent_of(input logic [N/2-1:0][PEW-1:0] map, input logic [2:0] sh, input int col);
    return (sh == '0) ? map[0] : map[col >> (LGN - int'(sh))];
  endfunction
  function automatic logic is_byp_code(input logic [CONTROL_WIDTH-1:0] c);
    return (c == CONTROL_WIDTH'(3'b100)) || (c == CONTROL_WIDTH'(3'b101));
  endfunction
  // Lane k's element i of the set the activation stage holds: a run along a row, or for a packed set down column k % N.
  function automatic int elem(input int k, input int i, input logic pk);
    return pk ? ((k / N) * PER_LANE + i) * N + (k % N) : k * PER_LANE + i;
  endfunction
```

3c. Per-id state beside `set_terms`:

```systemverilog
  logic [2:0] set_pack[NUM_IDS];  // each set's pack shift, block map and table activations (entry 0 = set_act)
  logic [N/2-1:0][PEW-1:0] set_map[NUM_IDS];
  logic [PACK_ENTRIES-1:0][CONTROL_WIDTH-1:0] set_ents[NUM_IDS];
  logic g_pack;  // the activation stage's set is packed
  logic [PEW-1:0] lane_ent[NUM_LANES];  // the entry each lane uses for that set
  logic [PEW-1:0] p_lane_ent[NUM_LANES];  // and each pooling lane for the pooled set
  logic [CONTROL_WIDTH-1:0] lane_act[NUM_LANES];
  logic act_bypass;  // every lane's code is ReLU or linear; declared here because the lane-control block above its old place reads it
```
In the stage controller's reset add `g_pack <= 1'b0;`, `set_pack[k] <= '0; set_map[k] <= '0; set_ents[k] <= '0;` in the id
loop, and `for (int k = 0; k < NUM_LANES; k++) begin lane_ent[k] <= '0; p_lane_ent[k] <= '0; end`. In `if (host_accept)` add

```systemverilog
        set_pack[host_next_id] <= pack_shift_i;
        set_map[host_next_id]  <= pack_map_i;
        set_ents[host_next_id][0] <= activation_function_i;
        for (int e = 1; e < PACK_ENTRIES; e++) set_ents[host_next_id][e] <= pack_act_i[e];
```
and after `if (g_accept) g_set_id <= g_next_id;`:

```systemverilog
      if (g_accept) begin
        g_pack <= (set_pack[g_next_id] != '0);
        for (int k = 0; k < NUM_LANES; k++) lane_ent[k] <= ent_of(set_map[g_next_id], set_pack[g_next_id], k % N);
      end
      if (p_accept)
        for (int k = 0; k < NUM_LANES; k++) p_lane_ent[k] <= ent_of(set_map[p_next_id], set_pack[p_next_id], k % N);
```

3d. Activation codes per lane, replacing the `gpnae_ctrl[i] = set_act[g_set_id];` line and the `act_bypass` / `act_is_relu`
assigns:

```systemverilog
      gpnae_ctrl[i] = (IS_INT && lane_act[i] == CONTROL_WIDTH'(3'b100) && !act_bypass) ? CONTROL_WIDTH'(3'b101) : lane_act[i];  // int8 ReLU is the clamp's
```
```systemverilog
  // ReLU and linear need no polynomial: when every lane's code is one of them, each beat goes straight into its bank and the lanes stay idle.
  always_comb begin
    act_bypass = 1'b1;
    for (int k = 0; k < NUM_LANES; k++) begin
      lane_act[k] = set_ents[g_set_id][lane_ent[k]];
      act_bypass &= is_byp_code(lane_act[k]);
    end
  end
```
(`act_is_relu` is replaced by `lane_act[k] == 3'b100` per lane below; delete its declaration.)

3e. Write-back, in the `GPNAE TO CENTRAL BUFFER WRITE LOGIC` block:

```systemverilog
    if (!IS_INT && g_state != G_IDLE && act_bypass && fill_v)
      for (int k = 0; k < NUM_LANES; k++)
        gpnae_out_mem[act_wr_base + elem(k, int'(fill_count[k]), g_pack)] <=
            (lane_act[k] == CONTROL_WIDTH'(3'b100) && fill_d[k][DATA_WIDTH-1]) ? '0 : fill_d[k];
    if (IS_INT && byp_wr)  // int8: a beat lands where its own tag says, so it may leave the requantize pipeline after its set left the stage
      for (int k = 0; k < NUM_LANES; k++) gpnae_out_mem[(byp_bank ? SRAM_DEPTH : 0) + elem(k, int'(byp_idx), byp_pack)] <= fill_d[k];
```
and in the lane-result loop `gpnae_out_mem[act_wr_base + elem(i, int'(done_count[i]), g_pack)] <= gpnae_result[i];`.
Declare `logic byp_pack;  // int8: the draining beat's set was packed` beside `byp_bank`. In `G_REQ`: add `logic [RQL-1:0] tg_pack;`,
`tg_pack <= {tg_pack[RQL-2:0], g_pack};` beside `tg_bank`, `assign byp_pack = tg_pack[RQL-1];`; in `G_NO_REQ`: `assign byp_pack = 1'b0;`.

3f. Mesh instance: `.pack_shift_i(pack_shift_i),` beside `.bias_i(...)` (the mesh samples it with `systolic_start`, which is
`host_accept`), and `.wide_read_packed_i(g_pack),` beside `.wide_read_index_i(...)`.

3g. int8 entries per id and per lane, in `G_REQ_SETS` (replace the scalar `s_zp ... s_zout` arrays and their assigns):

```systemverilog
    logic [PACK_ENTRIES-1:0][7:0]  s_zp[NUM_IDS], s_min[NUM_IDS], s_max[NUM_IDS], s_shout[NUM_IDS], s_zout[NUM_IDS];
    logic [PACK_ENTRIES-1:0][15:0] s_mx[NUM_IDS];
    logic [PACK_ENTRIES-1:0][4:0]  s_shx[NUM_IDS];
    logic [PACK_ENTRIES-1:0][31:0] s_mout[NUM_IDS];
```
On `host_accept`: entry 0 from `req_zp_i, req_min_i, req_max_i, gp_mx_i, gp_shx_i, gp_mout_i, gp_shout_i, gp_zout_i` as now
(`s_zp[host_next_id][0] <= req_zp_i;` etc.) and `for (int e = 1; e < PACK_ENTRIES; e++)` from `pack_*_i[e]`. Per lane:

```systemverilog
    for (genvar k = 0; k < NUM_LANES; k++) begin : G_LANE_PAR
      assign g_zp[k]    = s_zp[g_set_id][lane_ent[k]];
      assign g_min[k]   = s_min[g_set_id][lane_ent[k]];
      assign g_max[k]   = s_max[g_set_id][lane_ent[k]];
      assign g_mx[k]    = s_mx[g_set_id][lane_ent[k]];
      assign g_shx[k]   = s_shx[g_set_id][lane_ent[k]];
      assign g_mout[k]  = s_mout[g_set_id][lane_ent[k]];
      assign g_shout[k] = s_shout[g_set_id][lane_ent[k]];
      assign g_zout[k]  = s_zout[g_set_id][lane_ent[k]];
      // D-5: a dropped value is the output zero point of the pooled column's activation (tanh, and every code the lane runs as tanh: 0).
      always_comb
        case (set_ents[p_set_id][p_lane_ent[k]])
          CONTROL_WIDTH'(3'b001): p_zp[k] = s_zout[p_set_id][p_lane_ent[k]];  // SELU: its requantized output's zero point
          CONTROL_WIDTH'(3'b010): p_zp[k] = 8'h80;  // sigmoid: TFLite's fixed output zero point -128
          CONTROL_WIDTH'(3'b100), CONTROL_WIDTH'(3'b101): p_zp[k] = s_zp[p_set_id][p_lane_ent[k]];  // ReLU, linear: the requantize output's
          default: p_zp[k] = 8'h00;  // tanh: zero point 0
        endcase
    end
```
and change the declarations of `g_zp, g_min, g_max, g_shout, g_zout, p_zp, g_mx, g_shx, g_mout` to per-lane arrays
(`logic [NUM_LANES-1:0][7:0] g_zp, ...`); `G_NO_REQ_SETS` assigns `'0` to the arrays. Connect `requant_lanes` with
`.packed_i(g_pack), .zp_i(g_zp), .min_i(g_min), .max_i(g_max)`, each `gpnae_poly` with `.gp_mx_i(g_mx[g]) ... .gp_zout_i(g_zout[g])`
and `.gp_zin_i(g_zp[g])`, and each `dropout` with `.zero_point_i(p_zp[g])`.

3h. Refusals, after `a_complete_dispatched`:

```systemverilog
  localparam bit PACK_OK = (NUM_LANES % N == 0) && (COLLAPSE_K != 0) && POOL_BYPASS;
  a_pack_lanes: assert property (@(posedge clk_i) disable iff (!rstn_i) (host_accept && pack_shift_i != '0) |-> PACK_OK)
    else $error("sienna_top: a packed set needs N (%0d) to divide NUM_LANES (%0d), collapse-k 1 and a 1x1 pool", N, NUM_LANES);
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) host_accept |-> int'(pack_shift_i) < LGN)
    else $error("sienna_top: pack shift %0d leaves blocks narrower than 2 of N=%0d", pack_shift_i, N);
  a_pack_one_pass: assert property (@(posedge clk_i) disable iff (!rstn_i) (host_accept && pack_shift_i != '0) |-> !accumulate_i)
    else $error("sienna_top: a packed set cannot be a partial sum");
```

3i. Tie-offs: in `src/sienna_multi.sv`, `testbenches/TB_sienna_model.sv` and `src/sienna_layer.sv` connect every new port
to `'0` (`.pack_shift_i('0), .pack_map_i('0), .pack_act_i('0), ...`).

- [ ] **Step 4: Run the packed tests**

```bash
for f in fp32 bf16 int8; do $J/cmds/int8_tree.sh pk_top16_${f}_packed 32 6 bash $J/cmds/mk.sh regression FMT=$f N=16 TILE=4 TEST=packed; done
```
Expected: every `packed_*` test (fp32 / bf16: 3, int8: 4 including `int8_packed_zp_random_nopool`) PASS, 0 failed, no
assertion messages, the stream's overlap checks pass.

- [ ] **Step 5: Run the full regression and compare with the pre-packing runs**

```bash
for f in fp32 bf16 int8; do $J/cmds/int8_tree.sh pk_reg16_$f 32 6 bash $J/cmds/mk.sh regression FMT=$f N=16 TILE=4; done
```
Expected: `Passed: 32 / 32` (fp32, bf16) and `41 / 41` (int8). `python3 $J/cmds/int8_cmp_reg.py mg_reg_$f pk_reg16_$f`
reports every reference test IDENTICAL (result, single-set and streamed cycles, every act= and output word); the new
packed tests appear only in the new run.

- [ ] **Step 6: N = 8 and 32, and lint**

```bash
for f in fp32 bf16 int8; do $J/cmds/int8_tree.sh pk_reg8_$f 32 6 bash $J/cmds/mk.sh regression FMT=$f N=8 TILE=4; done
$J/cmds/int8_tree.sh pk_reg32_int8 32 8 bash $J/cmds/int8_O0.sh bash $J/cmds/cmd_sienna_fmt.sh reg 32 4 int8
for f in fp32 bf16 int8; do $J/cmds/int8_tree.sh pk_lint_$f 32 2 bash $J/cmds/mk.sh lint FMT=$f; done
$J/cmds/int8_tree.sh pk_lint_tops 32 2 bash $J/cmds/int8_top_lint.sh
```
Expected: all regressions pass (compare N = 32 int8 with `mg_r32_int8` via `int8_cmp_reg.py`: IDENTICAL for reference
tests); lint exit 0 with 0 errors and 0 warnings; `int8_top_lint.sh` `LINT VERDICT PASS`.

- [ ] **Step 7: Refusal of a lane count N does not divide** (Review Focus 4)

`$J/cmds/int8_tree.sh pk_refuse 32 4 bash $J/cmds/mk.sh regression FMT=bf16 N=16 TILE=4 LANES=8 TEST=packed_mixed`
Expected: the log contains `sienna_top: a packed set needs N (16) to divide NUM_LANES (8)` and the test FAILs. (This run
is expected to fail; record it as the refusal's evidence.)

- [ ] **Step 8: Commit** (RTL first, then the tie-offs, then the tests; push after Task 6 builds)

```bash
git add src/sienna_top.sv && git commit -m "sienna_top: packed sets. pack_shift_i, a block map and table entries 1..7 (activation; int8 requantize zero point and clamp, GPNAE words) come with each start and are held per set id; the mesh gets the shift; a packed result is read column-wise so lane k takes column k % N and the entry of its block, for its activation code, requantize zero point and clamp, GPNAE words and dropout drop value (D-5); the bypass path runs when every lane's code is ReLU or linear; int8 ReLU lanes in a mixed set run as linear behind the clamp. Unpacked sets with map[0] = 0 behave exactly as before. Assertions refuse packing without N | NUM_LANES, collapse-k 1 and a 1x1 pool, beyond log2(N) - 1, or with accumulate."
git add src/sienna_multi.sv testbenches/TB_sienna_model.sv src/sienna_layer.sv && git commit -m "sienna_multi, TB_sienna_model, sienna_layer: tie sienna_top's packing ports to 0."
git add regression.py testbenches/TB_sienna_top.sv && git commit -m "Packed pipeline tests: packed_mixed_act_nopool, packed_bias_cached_train_nopool, packed_all_bypass_nopool and int8_packed_zp_random_nopool stream sets with shifts 1, 0, b = 2 and b = N / 4 and an 8-entry table rotating over the blocks; the golden is each block's job alone (matmul_packed in bf16, the float64 product in fp32, per-entry requantize in int8) through its entry's activation, then dropout to each column's drop value; TB_sienna_top drives pack_<k>.mem with every set."
```

---

### Task 6: The layer engine takes a packed layer

**Files:**
- Modify: `src/sienna_layer.sv`, `testbenches/TB_sienna_layer.sv`, `model_runner.py:415-476`
- Create: `pack_regression.py`

**Interfaces:**
- Consumes: Task 5's sienna_top ports.
- Produces: sienna_layer config ports `cfg_pack_shift_i[2:0]`, `cfg_pack_map_i[N/2-1:0][2:0]`, `cfg_pack_act_i`,
  `cfg_pack_zp_i ... cfg_pack_zout_i` (shapes as sienna_top's, entries 1..7), taken with `cfg_load_i`; layer file: `L`
  line gains a 12th field `packed` (0/1); when 1, a `P sh m0 .. m(N/2-1)` line and seven `E act zp min max mx shx mout
  shout zout` lines follow `L`/`Q`; `LayerSim.run_job(job, tag)` writes them from `job["pack"] = {"shift": int, "map":
  list, "ents": [(act, rq_or_None)] * 8}`; a packed job must be N rows-multiple x N x N.

- [ ] **Step 1: Write the failing test** (`pack_regression.py`, repo root)

```python
#!/usr/bin/env python3
"""Packed layers on sienna_layer: every job's block equals the same job run alone on the RTL, bit for bit, and its golden;
cycles packed against alone. Writes testbenches/results/pack/pack_regression_<fmt>_N<n>_T<t>.log."""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import gemm_sweep as gs  # noqa: E402
import model_runner as mr  # noqa: E402

reg = mr.regression
ACTS = ["linear", "tanh", "relu", "selu", "sigmoid", "linear", "relu", "tanh"]


def layer(N, sh, R, rng, fmt, zero_rows=False):
    """A packed layer of R row tiles: block c's job is A[:, block] @ W_c + bias, through ACTS[(c + 1) % 8]."""
    b = N >> sh
    A = rng.uniform(-1, 1, (R * N, N))
    if zero_rows:
        A[0::4, :], A[1::4, :] = -0.0, 0.0
    B = np.zeros((N, N))
    for c in range(N // b):
        B[c * b:(c + 1) * b, c * b:(c + 1) * b] = rng.uniform(-1, 1, (b, b))
    bias = rng.uniform(-0.5, 0.5, N)
    mp = [((c + 1) % 8 if c < N // b else 0) for c in range(N // 2)]
    return A, B, bias, mp, b


def run_case(a, sim, N, sh, R, seed, log, zero_rows=False):
    rng = np.random.RandomState(seed)
    A, B, bias, mp, b = layer(N, sh, R, rng, a.fmt, zero_rows)
    col_ent = [mp[j // b] for j in range(N)]
    ents = [(ACTS[e], None) for e in range(8)]
    if a.fmt == "int8":
        A_q, s_a, z_a = reg.quant_act(A)
        B_q, s_w = reg.quant_weights(B)
        hw = reg.fold_bias(bias, s_a, s_w, z_a, B_q)
        acc = reg.wrap32(reg.imatmul(A_q, B_q) + hw[None, :])
        mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
        for e in sorted(set(col_ent)):
            cols = np.array(col_ent) == e
            act = "linear" if ACTS[e] == "selu" else ACTS[e]  # SELU's int8 saturation is pack_jobs' to refuse (Task 7)
            rq = reg.requant_params(acc[:, cols], s_a, s_w[cols], act)
            mult[cols], shift[cols] = rq["mult"], rq["shift"]
            ents[e] = (act, rq)
        head = ents[0][1] or reg.requant_params(acc, s_a, s_w, "linear")
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw, "act": ents[0][0], "shape": (R * N, N),
               "req": dict(head, mult=mult, shift=shift), "pack": {"shift": sh, "map": mp, "ents": ents}}
    else:
        A, B, bias = (reg.op_round(v, a.fmt) for v in (A, B, bias))
        job = {"terms": [(A, B)], "bias": bias, "act": ACTS[0], "shape": (R * N, N), "pack": {"shift": sh, "map": mp, "ents": ents}}
    t0 = time.time()
    Yp, sets_p, cyc_p = sim.run_job(job, f"pk_s{sh}")
    bad = cyc_a = 0
    for c in range(N // b):
        cols = slice(c * b, (c + 1) * b)
        e = mp[c]
        if a.fmt == "int8":
            rq = dict(ents[e][1], mult=mult[cols], shift=shift[cols])
            alone = {"terms": [(A_q[:, cols].astype(np.float32), B_q[cols, cols].astype(np.float32))], "bias": hw[cols],
                     "act": ents[e][0], "shape": (R * N, b), "req": rq}
            gold = reg.int8_layer_exact(A_q[:, cols], B_q[cols, cols], hw[cols], rq, ents[e][0])
        else:
            alone = {"terms": [(A[:, cols], B[cols, cols])], "bias": bias[cols], "act": ACTS[e], "shape": (R * N, b)}
            gold = gs.exact_layer(A[:, cols], B[cols, cols], bias[cols], ACTS[e], N, a.fmt) if a.fmt == "bf16" else None  # fp32: RTL against RTL (F-GP1)
        Ya, _, cyc = sim.run_job(alone, f"al_s{sh}_{c}")
        cyc_a += cyc
        bits = (lambda y: y) if a.fmt == "int8" else (lambda y: reg.fmt_bits(y, a.fmt))
        bad += int(np.sum(bits(Yp[:, cols]) != bits(Ya)))
        if gold is not None:
            bad += int(np.sum(bits(Yp[:, cols]) != gold))
    line = (f"{a.fmt} N={N} T={a.tile} b={b:<3} rows={R * N:<4} {'zero rows ' if zero_rows else ''}packed: {sets_p} sets "
            f"{cyc_p} cycles | {N // b} jobs alone: {cyc_a} cycles | speedup {cyc_a / cyc_p:5.2f}x | mismatches {bad} | "
            f"wall {time.time() - t0:.0f}s")
    print(line, flush=True)
    log.write(line + "\n")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2, help="row tiles per packed layer")
    ap.add_argument("--format", dest="fmt", default="int8", choices=sorted(reg.FORMATS))
    a = ap.parse_args()
    work = os.path.join(ROOT, "testbenches", "results", "pack")
    os.makedirs(work, exist_ok=True)
    sim = mr.LayerSim(a.n, a.lanes, work, a.fmt, a.tile)
    sim.build()
    log = open(os.path.join(work, f"pack_regression_{a.fmt}_N{a.n}_T{a.tile}.log"), "w")
    bad = 0
    for sh in range(1, a.n.bit_length() - 1):
        bad += run_case(a, sim, a.n, sh, a.rows, 900 + sh, log)
    if a.fmt != "int8":
        bad += run_case(a, sim, a.n, 2, a.rows, 990, log, zero_rows=True)  # Review Focus 3: signed zeros
    tail = f"PACK REGRESSION {'PASS' if bad == 0 else 'FAIL'}: {bad} mismatching outputs"
    print(tail, flush=True)
    log.write(tail + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```
(`gs.exact_layer` fixes T = 4 until Task 7; with collapse-k 1 the mesh model does not use T, so the bf16 goldens hold at
every T already. Task 7 passes `a.tile` anyway.)

- [ ] **Step 2: Run to verify it fails**

`$J/cmds/int8_tree.sh pk_lr_red 32 4 $J/venv/bin/python pack_regression.py --format bf16 --n 16 --tile 4`
Expected: `KeyError`/`RuntimeError` from LayerSim (no packing support) or a layer simulation failure.

- [ ] **Step 3: Implement the RTL** (`src/sienna_layer.sv`)

Parameter `parameter int PACK_ENTRIES = 8,` after `DIM_W`'s line (comma on DIM_W). Ports after `cfg_gp_zout_i`:

```systemverilog
    // Packing, taken with cfg_load_i: every set of the layer is packed this way (sienna-packing)
    input logic [2:0]                          cfg_pack_shift_i,
    input logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0] cfg_pack_map_i,
    input logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] cfg_pack_act_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_zp_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_min_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_max_i,
    input logic [PACK_ENTRIES-1:1][15:0]       cfg_pack_mx_i,
    input logic [PACK_ENTRIES-1:1][4:0]        cfg_pack_shx_i,
    input logic [PACK_ENTRIES-1:1][31:0]       cfg_pack_mout_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_shout_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_zout_i,
```
Registers `pk_*_q` of the same shapes, loaded beside `gp_zout_q <= cfg_gp_zout_i;` and reset to `'0`; replace Task 5's
tie-offs in the `pipe` instance with `.pack_shift_i(pk_shift_q), .pack_map_i(pk_map_q), .pack_act_i(pk_act_q), ...`, and
pass `.PACK_ENTRIES(PACK_ENTRIES)`. Assertion beside `a_start_taken`:

```systemverilog
  a_pack_shape: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                 (cfg_load_i && !active && cfg_pack_shift_i != '0) |-> (cfg_n_i == DIM_W'(N) && cfg_kb_i == DIM_W'(N) && !cfg_residual_i))
    else $error("sienna_layer: a packed layer is N columns and N deep, without a residual");
```

- [ ] **Step 4: Implement the TB and LayerSim**

`TB_sienna_layer.sv`: header comment line `// "L ... w_rows packed"; packed: a "P sh m0 .. m(N/2-1)" line and seven "E act zp min max mx shx mout shout zout" lines after L (and Q).`;
declare `logic [2:0] cfg_pack_shift_i = '0; logic [N/2-1:0][2:0] cfg_pack_map_i = '0;` and the entry arrays (as TB_sienna_top's,
`cfg_` prefix, all `'0`); read `L` with 12 fields (`rc != 12` is the error), `pk` the 12th; after the `Q` block:

```systemverilog
    if (pk) begin
      int sh, mv, ev[9];
      rc = $fscanf(fin, "%s %d", kind, sh);
      if (rc != 2 || kind != "P") begin
        $display("[FATAL] %s: a packed layer needs its P line", layer_f);
        $finish;
      end
      cfg_pack_shift_i = 3'(sh);
      for (int c = 0; c < N / 2; c++) begin
        rc = $fscanf(fin, "%d", mv);
        cfg_pack_map_i[c] = 3'(mv);
      end
      for (int e = 1; e < 8; e++) begin
        rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d", kind, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8]);
        if (rc != 10 || kind != "E") begin
          $display("[FATAL] %s: entry %0d needs its E line", layer_f, e);
          $finish;
        end
        cfg_pack_act_i[e] = CONTROL_WIDTH'(ev[0]);
        cfg_pack_zp_i[e] = 8'(ev[1]);
        cfg_pack_min_i[e] = 8'(ev[2]);
        cfg_pack_max_i[e] = 8'(ev[3]);
        cfg_pack_mx_i[e] = 16'(ev[4]);
        cfg_pack_shx_i[e] = 5'(ev[5]);
        cfg_pack_mout_i[e] = 32'(ev[6]);
        cfg_pack_shout_i[e] = 8'(ev[7]);
        cfg_pack_zout_i[e] = 8'(ev[8]);
      end
    end
```
(`int pk;` beside the other fields; the L read becomes
`rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d %d %d", kind, m, kb, n, res, bias, act, train, seed, na, nw, pk);` with
`rc != 12` as the error.) `model_runner.py` `LayerSim.run_job`, replacing the `f.write(f"L ...")` line and
adding the packing lines after the `Q` line:

```python
            pk = job.get("pack")
            if pk and (C != N or cfg["kb"] != N or rt * N != M):
                raise ValueError(f"{tag}: a packed layer is a whole number of row tiles, N columns and N deep")
            f.write(f"L {cfg['m']} {cfg['kb']} {cfg['n']} {cfg['residual']} {cfg['bias']} {cfg['act']} 0 0 {len(a)} {len(w)} {int(bool(pk))}\n")
            if int8:
                q = job["req"]
                f.write(f"Q {q['zp']} {q['amin']} {q['amax']} {q['mx']} {q['shx']} {q['mout']} {q['shout']} {q['zout']}\n")
            if pk:
                f.write("P " + " ".join(str(int(v)) for v in [pk["shift"]] + list(pk["map"])) + "\n")
                for act, rq in pk["ents"][1:]:
                    q = rq or {}
                    f.write(f"E {ACT_CODES[act]} " + " ".join(str(int(q.get(x, 0))) for x in
                                                            ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")) + "\n")
```
(delete the old `L` and `Q` writes these replace.) A job with `"pack"` must not take the residual path: `format_layer`
already treats a non-identity B as dense.

- [ ] **Step 5: Run every N <= 32, every T, all formats**

```bash
for f in fp32 bf16 int8; do for nt in 8:2 8:4 8:8 16:2 16:4 16:8 16:16 32:2 32:4 32:8 32:16 32:32; do
  N=${nt%:*}; T=${nt#*:}; $J/cmds/int8_tree.sh pk_lr_N${N}_T${T}_$f 32 12 $J/venv/bin/python pack_regression.py --format $f --n $N --tile $T
done; done
```
Expected: every log ends `PACK REGRESSION PASS: 0 mismatching outputs`; speedups about N/b (packed vs the jobs alone).
Also rerun `make gemm FMT=int8 QUICK=1` and `make tflite FMT=int8` (farm, via `$J/cmds/mk.sh` and
`$J/cmds/int8_tflite.sh 16 4 32`): unchanged results (the `L` line format change must not break them).

- [ ] **Step 6: Commit and push**

```bash
git add src/sienna_layer.sv && git commit -m "sienna_layer: a packed layer. The pack shift, block map and table entries are layer configuration, given to every set; an assertion requires a packed layer to be N columns and N deep without a residual."
git add testbenches/TB_sienna_layer.sv model_runner.py && git commit -m "TB_sienna_layer, LayerSim: the layer file's L line carries a packed flag, followed for a packed layer by its P (shift, map) and E (entries 1..7) lines; LayerSim writes them from job['pack']."
git add pack_regression.py && git commit -m "pack_regression.py: packed layers of every shift, one entry per block, against each job alone on the RTL and its golden, bit for bit in every format (signed-zero rows too), with packed-vs-alone cycles."
git log --format=%B origin/packing..HEAD | command grep -ci co-authored-by   # 0
git push origin packing
```

---

### Task 7: Packer and unpacker for arbitrary small jobs

**Files:**
- Modify: `model_runner.py` (new functions after `layer_epilogue`), `gemm_sweep.py:30-58`, `pack_regression.py`
- Create: `test_pack_jobs.py`

**Interfaces:**
- Produces: `model_runner.pack_jobs(models: list, N: int, int8: bool) -> (job: dict, recipe: list)` where each model is
  `{"W": np.ndarray (K x C), "bias": np.ndarray or None, "act": str, "req": dict or None, "inputs": [np.ndarray (m_i x K)]}`,
  K, C <= N/2; `model_runner.unpack(Y: np.ndarray, recipe) -> list[list[np.ndarray]]` (per model, per input, m_i x C);
  `gemm_sweep.exact_layer(A, B, bias, act, N, fmt, T=4)`.

- [ ] **Step 1: Write the failing test** (`test_pack_jobs.py`)

```python
#!/usr/bin/env python3
"""model_runner.pack_jobs / unpack, without a simulator: layout, refusals, and every job recovered from Y = A @ B + bias."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_runner as mr  # noqa: E402


def _models(rng, shapes, acts):
    return [{"W": rng.uniform(-1, 1, (K, C)), "bias": rng.uniform(-1, 1, C), "act": a, "req": None,
             "inputs": [rng.uniform(-1, 1, (m, K)) for m in ms]} for (K, C, ms), a in zip(shapes, acts)]


def test_round_trip_float():
    rng = np.random.RandomState(3)
    models = _models(rng, [(5, 3, [2, 7]), (8, 8, [1]), (2, 6, [4, 4, 3])], ["tanh", "relu", "linear"])
    job, recipe = mr.pack_jobs(models, 32, int8=False)
    (A, B), = job["terms"]
    assert job["pack"]["shift"] == 2 and A.shape == (32, 32) and B.shape == (32, 32)  # b = 8, rows padded to N
    Y = A @ B + job["bias"][None, :]
    for m, outs in zip(models, mr.unpack(Y, recipe)):
        for x, y in zip(m["inputs"], outs):
            assert np.allclose(y, x @ m["W"] + m["bias"]), "a job's rows and columns came back wrong"


def test_partial_packing_and_entries():
    rng = np.random.RandomState(4)
    models = _models(rng, [(3, 3, [1]), (4, 2, [2])], ["selu", "selu"])
    job, recipe = mr.pack_jobs(models, 16, int8=False)
    pk = job["pack"]
    assert pk["shift"] == 2 and pk["map"][:4] == [0, 0, 0, 0] and pk["ents"][0][0] == "selu"  # same setting, one entry; blocks 2, 3 empty
    Y = job["terms"][0][0] @ job["terms"][0][1]
    assert np.all(Y[:, 8:] == 0), "an empty block's columns must stay zero before bias"


def test_refusals():
    rng = np.random.RandomState(5)
    big = _models(rng, [(9, 4, [1])], ["linear"])
    for bad, why in ((big, "K > N/2"), (_models(rng, [(2, 2, [1])] * 9, ["tanh", "relu", "linear", "selu", "sigmoid", "tanh",
                                                                          "relu", "linear", "selu"]), "more models than blocks")):
        try:
            mr.pack_jobs(bad, 16, int8=False)
        except ValueError:
            continue
        raise AssertionError(f"pack_jobs accepted {why}")


def test_selu_saturation_refused():
    # Review Focus 5: an int8 SELU entry whose lane input reaches x >= 487.29 must be refused, not packed.
    req = dict(mult=np.full(2, 1 << 30), shift=np.zeros(2, np.int64), zp=-128, amin=-128, amax=127, mx=(1 << 15) - 1, shx=0,
               mout=1, shout=0, zout=0)
    m = {"W": np.ones((2, 2)), "bias": None, "act": "selu", "req": req, "inputs": [np.ones((1, 2))]}
    try:
        mr.pack_jobs([m], 16, int8=True)
    except ValueError:
        return
    raise AssertionError("pack_jobs accepted a saturating SELU entry")


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {name}: {e!r}")
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    sys.exit(1 if bad else 0)
```

- [ ] **Step 2: Run to verify it fails**

`python3 test_pack_jobs.py` (login node; numpy only) → `FAIL ... AttributeError ... 'pack_jobs'` for all four.

- [ ] **Step 3: Implement** (`model_runner.py`, after `layer_epilogue`)

```python
PACK_ENTRIES = 8  # sienna_top's parameter table


def pack_jobs(models: list, N: int, int8: bool) -> tuple:
    """One packed layer from small models (sienna-packing): model c gets column block c of width b, the smallest power of two
    >= every K, C and 2; its inputs go down its block's rows; models with the same activation and int8 output words share a
    table entry. Returns (LayerSim job, recipe for unpack). Refuses K or C > N/2, more models than blocks, more than 8
    settings, and an int8 SELU entry whose input range saturates."""
    if not models:
        raise ValueError("nothing to pack")
    K = max(m["W"].shape[0] for m in models)
    C = max(m["W"].shape[1] for m in models)
    b = 2
    while b < max(K, C):
        b *= 2
    if b > N // 2:
        raise ValueError(f"a job of depth {K} and width {C} needs blocks of {b}; packing needs at most N/2 = {N // 2}")
    if len(models) > N // b:
        raise ValueError(f"{len(models)} models need {len(models)} blocks of {b}; N = {N} holds {N // b}")
    sh = (N // b).bit_length() - 1
    rows = max(sum(x.shape[0] for x in m["inputs"]) for m in models)
    M = -(-rows // N) * N
    A, B = np.zeros((M, N), np.float32), np.zeros((N, N), np.float32)
    bias = np.zeros(N, np.int64 if int8 else np.float32)
    mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
    keys, ents, mp, recipe = [], [], [0] * (N // 2), []
    for c, m in enumerate(models):
        k, cc = m["W"].shape
        X = np.vstack(m["inputs"]).astype(np.float32)
        A[:X.shape[0], c * b:c * b + k] = X
        B[c * b:c * b + k, c * b:c * b + cc] = m["W"]
        if m["bias"] is not None:
            bias[c * b:c * b + cc] = m["bias"]
        q = m.get("req")
        if int8:
            mult[c * b:c * b + cc], shift[c * b:c * b + cc] = q["mult"], q["shift"]
            if m["act"] == "selu" and np.any(regression.selu_saturates(q["mx"], q["shx"], q["zp"], np.arange(q["amin"], q["amax"] + 1))):
                raise ValueError(f"model {c}: its SELU input range reaches x = 487.29, where the int8 lane saturates")
        key = (m["act"],) + (tuple(int(q[x]) for x in ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")) if int8 else ())
        if key not in keys:
            keys.append(key)
            ents.append((m["act"], {x: v for x, v in q.items() if x not in ("mult", "shift")} if int8 else None))
        mp[c] = keys.index(key)
        r0, spans = 0, []
        for x in m["inputs"]:
            spans.append((r0, x.shape[0]))
            r0 += x.shape[0]
        recipe.append((c * b, cc, spans))
    if len(ents) > PACK_ENTRIES:
        raise ValueError(f"{len(ents)} distinct activation / output settings; a packed set holds {PACK_ENTRIES}")
    ents += [("linear", None)] * (PACK_ENTRIES - len(ents))
    job = {"terms": [(A, B)], "bias": bias, "act": ents[0][0], "shape": (M, N), "pack": {"shift": sh, "map": mp, "ents": ents}}
    if int8:
        job["req"] = dict(ents[0][1], mult=mult, shift=shift)
    return job, recipe


def unpack(Y: np.ndarray, recipe: list) -> list:
    """Each model's outputs, one array per input, from a packed layer's result."""
    return [[Y[r0:r0 + m, c0:c0 + cc] for r0, m in spans] for c0, cc, spans in recipe]
```

`gemm_sweep.py`: `def exact_layer(A, B, bias, act, N, fmt, T=4):` and `Ct = mesh_model.matmul(f, passes, N, T, 1, b)`
(docstring: `... the bias with the first, then the lane; T is the build's tile size.`). In `pack_regression.py`, pass
`a.tile` to `gs.exact_layer`, and add a pack_jobs case after the shift loop:

```python
    bad += run_models(a, sim, log)
```
with

```python
def run_models(a, sim, log):
    """Review Focus 2: heterogeneous small jobs through pack_jobs, fewer models than blocks; each against itself alone."""
    rng = np.random.RandomState(77)
    N = a.n
    shapes = [(3, 2, [5, 9]), (2, 2, [N]), (2, 1, [3])]  # blocks of 4: N = 8 holds two models, 16 and 32 leave blocks empty
    acts = ["tanh", "relu", "linear"]
    models = [{"W": rng.uniform(-1, 1, (K, C)), "bias": rng.uniform(-0.5, 0.5, C), "act": act, "req": None,
               "inputs": [rng.uniform(-1, 1, (m, K)) for m in ms]} for (K, C, ms), act in zip(shapes, acts)][:N // 4]
    if a.fmt != "fp32":
        for m in models:
            m["W"], m["bias"] = reg.op_round(m["W"], a.fmt), reg.op_round(m["bias"], a.fmt)
            m["inputs"] = [reg.op_round(x, a.fmt) for x in m["inputs"]]
    if a.fmt == "int8":
        log.write("int8 pack_jobs case: covered by tflite_pack_run.py (Task 8)\n")
        return 0
    job, recipe = mr.pack_jobs(models, N, int8=False)
    Y, _, _ = sim.run_job(job, "pj")
    bad = 0
    for i, (m, outs) in enumerate(zip(models, mr.unpack(Y, recipe))):
        X = np.vstack(m["inputs"]).astype(np.float32)
        alone = {"terms": [(X, m["W"].astype(np.float32))], "bias": m["bias"].astype(np.float32), "act": m["act"],
                 "shape": (X.shape[0], m["W"].shape[1])}
        Ya, _, _ = sim.run_job(alone, f"pj_al{i}")
        got = np.vstack(outs)
        bad += int(np.sum(reg.fmt_bits(got, a.fmt) != reg.fmt_bits(Ya, a.fmt)))
    line = f"{a.fmt} N={N} T={a.tile} pack_jobs: {len(models)} models in {N // (N >> job['pack']['shift'])} blocks, mismatches {bad}"
    print(line, flush=True)
    log.write(line + "\n")
    return bad
```

- [ ] **Step 4: Run** `python3 test_pack_jobs.py` → `ALL PASS`; then the Task 6 Step 5 farm sweep again (names `pk_lr2_*`)
→ every log `PACK REGRESSION PASS` with the `pack_jobs` line at 0 mismatches; `make gemm FMT=bf16 QUICK=1` (farm)
unchanged.

- [ ] **Step 5: Commit and push**

```bash
git add model_runner.py && git commit -m "model_runner: pack_jobs packs small models (K, C <= N/2) into one layer, each in its own column block with its inputs down the rows and a table entry per distinct activation / int8 output setting, refusing what does not fit or an int8 SELU entry that saturates; unpack returns each input's outputs."
git add gemm_sweep.py && git commit -m "gemm_sweep: exact_layer takes the build's tile size instead of fixing T = 4."
git add test_pack_jobs.py pack_regression.py && git commit -m "test_pack_jobs.py covers pack_jobs' layout, partial packing, shared entries and refusals without a simulator; pack_regression.py runs heterogeneous models through pack_jobs against each alone and compares goldens at the build's T."
git push origin packing
```

---

### Task 8: Two packed TFLite runs, bit-exact against the interpreter

**Files:**
- Create: `tflite_pack_models.py`, `tflite_pack_run.py`; models in `testbenches/tflite_int8_pack/` (committed, as
  `testbenches/tflite_int8/` is)

**Interfaces:**
- Consumes: `tflite_oracle.build_model`, `extract`, `interpreter`, `invoke_all`, `test_inputs`, `save`, `REF`;
  `tflite_int8_run.load_layer`, `job_of`; Task 7's `pack_jobs`, `unpack`.

- [ ] **Step 1: Generate the models** (`tflite_pack_models.py`)

```python
#!/usr/bin/env python3
"""Small int8 TFLite layers for packing: fc 8x8 linear, fc 8x4 ReLU, fc 6x8 ReLU6 (blocks of 8), conv 3x3 x1 -> 8 linear and
fc 16x16 ReLU (blocks of 16), saved as tflite_oracle saves its G4 models; arg: output directory."""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import tflite_oracle as to  # noqa: E402

MODELS = {  # name: (layer, act, inputs, outputs)
    "fc8x8_linear": ("fc", "none", 8, 8),
    "fc8x4_relu": ("fc", "relu", 8, 4),
    "fc6x8_relu6": ("fc", "relu6", 6, 8),
    "conv3x3_6x6x1x8_linear": ("conv", "none", 1, 8),
    "fc16x16_relu": ("fc", "relu", 16, 16),
}


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    rounding = open(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt")).read().strip()
    for name, (layer, act, cin, cout) in MODELS.items():
        to.FC_IN, to.FC_OUT, to.CONV_CIN, to.CONV_COUT, to.CONV_HW = cin, cout, cin, cout, 6
        model, lo, hi = to.build_model(layer, act, 0)
        it = to.interpreter(model, to.REF)
        p = to.extract(it, model, layer, act)
        xs = to.test_inputs(p, lo, hi, 32, np.random.default_rng(2000))
        to.save(out, name, model, p, xs, to.invoke_all(it, xs), rounding)
        print(f"{name}: in zp {p['in_zp']} out zp {p['out_zp']} act [{p['amin']}, {p['amax']}]")
    shutil.copy(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt"), out)


if __name__ == "__main__":
    main(sys.argv[1])
```
Run on the farm, writing outside the snapshot so the files survive it (TensorFlow is in the venv; if
`import tensorflow` fails there, stop and report):
`$J/cmds/int8_tree.sh pk_tfgen 16 1 $J/venv/bin/python tflite_pack_models.py $J/runs/pk_tfmodels`, then
`cp $J/runs/pk_tfmodels/* testbenches/tflite_int8_pack/`. The activation names are `tflite_oracle.ACT_FN`'s keys
(`none`, `relu`, `relu6`).

- [ ] **Step 2: Write the packed run** (`tflite_pack_run.py`)

```python
#!/usr/bin/env python3
"""Packed TFLite layers on sienna_layer: groups of models share one packed layer; every model's outputs must equal the
interpreter's bit for bit. Writes testbenches/results/int8/tflite_pack_N<n>_T<t>.log."""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_runner as mr  # noqa: E402
import tflite_int8_run as tr  # noqa: E402

GROUPS = [["fc8x8_linear", "fc8x4_relu", "fc6x8_relu6"], ["conv3x3_6x6x1x8_linear", "fc16x16_relu"]]


def model_of(path):
    ref = np.load(path[:-len(".tflite")] + ".npz")
    layer = tr.load_layer(path)
    job, shape = tr.job_of(layer, ref["x_test"], ref)
    (X, W), = job["terms"]
    return {"W": W, "bias": job["bias"], "act": "linear", "req": job["req"], "inputs": [X]}, shape, ref["y_test"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=os.path.join(ROOT, "testbenches", "tflite_int8_pack"))
    a = ap.parse_args()
    work = os.path.join(ROOT, "testbenches", "results", "int8")
    os.makedirs(work, exist_ok=True)
    rep = open(os.path.join(work, f"tflite_pack_N{a.n}_T{a.tile_size}.log"), "w")
    sim = mr.LayerSim(a.n, a.lanes, work, "int8", a.tile_size)
    sim.build()
    bad = 0
    for g, names in enumerate(GROUPS):
        got = [model_of(os.path.join(a.models, f"{n}.tflite")) for n in names]
        job, recipe = mr.pack_jobs([m for m, _, _ in got], a.n, int8=True)
        Y, sets, cyc = sim.run_job(job, f"tflp{g}")
        for (m, shape, y), outs, n in zip(got, mr.unpack(Y, recipe), names):
            mism = int(np.sum(np.vstack(outs).reshape(shape) != y.reshape(shape)))
            bad += int(mism != 0)
            line = f"PACKED {n} (group {g}, b = {a.n >> job['pack']['shift']}): {mism}/{y.size} differ from the interpreter; {sets} sets, {cyc} cycles"
            print(line, flush=True)
            rep.write(line + "\n")
    tail = f"TFLITE_PACK: {sum(len(x) for x in GROUPS)} models in {len(GROUPS)} packed layers, {bad} failing"
    print(tail)
    rep.write(tail + "\nRESULT: " + ("PASSED" if bad == 0 else "FAILED") + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it** (it fails before Steps 1-2's files exist; after them it must pass)

```bash
for t in 2 4 8 16; do $J/cmds/int8_tree.sh pk_tfp_N32_T$t 32 6 $J/venv/bin/python tflite_pack_run.py --n 32 --tile-size $t; done
```
Expected: both groups pack (b = 8: three models in four blocks; b = 16: two models), every model `0/... differ`,
`RESULT: PASSED`. N = 32 only: at N = 16 blocks are at most 8 wide and hold two models, so these groups do not fit
(pack_jobs refuses them, as Task 7 tests).

- [ ] **Step 4: Commit and push**

```bash
git add tflite_pack_models.py testbenches/tflite_int8_pack && git commit -m "tflite_pack_models.py: five small int8 TFLite layers (fc 8x8, 8x4 ReLU, 6x8 ReLU6, conv 3x3 x1 -> 8, fc 16x16 ReLU) built and saved as tflite_oracle's G4 models, for packing."
git add tflite_pack_run.py && git commit -m "tflite_pack_run.py: packs groups of TFLite layers into one sienna_layer layer through pack_jobs and checks every model's outputs against the interpreter, bit for bit."
git push origin packing
```

---

### Task 9: Documentation, Makefile target, gates and report record

**Files:**
- Modify: `Makefile` (target `pack`), `.claude/skills/sienna-packing/SKILL.md` (status, as-built, results),
  `.claude/skills/sienna-rtl/SKILL.md` (precision section: `make pack`; interface note), `.claude/skills/sienna-back-to-back/SKILL.md`
  (top-level interface table: the packing ports), `test_makefile_fmt.py` (the new target in its `make -n` checks)
- Create: `/proj/work/spramanik/sienna_report/packing_gate.log`

- [ ] **Step 1: Makefile target** (after `tflite`):

```make
# Packed layers against each job alone on sienna_layer (pack_regression.py), in FMT at N, TILE and LANES.
pack:
	$(PYTHON) pack_regression.py --format $(FMT) --n $(N) --tile $(TILE) --lanes $(LANES)
```
and a help line `@echo "  make pack FMT=int8                 - packed layers vs each job alone (pack_regression.py)"`.
Add `pack` to `test_makefile_fmt.py`'s target list the way `gemm` is handled; run `python3 test_makefile_fmt.py` (login
node, `make -n` only) → all pass.

- [ ] **Step 2: Full gate on the final tree** (fresh runs, all on the farm, prefix `pkg_`): the mesh sweep of Task 3
  Step 6; `TB_PE_pack`; `TB_requant_lanes`; the SIENNA regression at N = 8, 16, 32 (T = 4) in three formats and N = 16
  at T = 2, 8, 16; the collapse-k 0 SIENNA run (`$J/cmds/sienna_ck0.sh 16 4 int8`); `pack_regression.py` at every N/T
  and format; `tflite_pack_run.py`; `make tflite FMT=int8`; `make gemm FMT=int8 QUICK=1`; lint both ways; and the area
  estimate flow behind `2026-09-27_synthesis_readiness.txt` (its script is named in that report or in `$J/cmds`) rerun on
  this tree at N = 16 and 32 in fp32, bf16 and int8, reporting the parameter table's and the lane-order muxes' share.
  Expected: all pass; every reference regression IDENTICAL to its `mg_*` / `g3i_*` baseline; the area delta recorded
  as an estimate (no timing run exists).

- [ ] **Step 3: Record** `packing_gate.log` in `/proj/work/spramanik/sienna_report/`: one line per run (name, command,
  verdict, comparison result), the packed-vs-alone speedups from `pack_regression` logs per N/T/format, and the multiply
  count per set (`b/N` of today's, from the PE assertion and `TB_PE_pack`), labelled as an energy proxy, not a power figure.

- [ ] **Step 4: Skills.** In `sienna-packing/SKILL.md` change the status line to
  `Status: implemented and verified <date> on the packing branch (SIENNA <hash>, SystolicMesh <hash>) at N = 8-32; N = 64 after check-in.`,
  add an "As built" section listing any deviation (each with its reason) and the gate summary; update `sienna-rtl` and
  `sienna-back-to-back` as listed above.

- [ ] **Step 5: Commit and push** (docs last)

```bash
git add Makefile test_makefile_fmt.py && git commit -m "Makefile: make pack runs pack_regression.py in FMT at N, TILE and LANES; test_makefile_fmt.py covers it."
git add .claude/skills && git commit -m "skill: sienna-packing implemented and verified at N = 8-32 (status, as built, gate); sienna-rtl and sienna-back-to-back gain the packing ports and make pack."
git log --format=%B origin/packing..HEAD | command grep -ci co-authored-by   # 0
git push origin packing
```
Then stop: merging `packing` to `main` in both repos, deleting the branches, and the N = 64 sweep are Soham's call.
