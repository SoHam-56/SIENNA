---
name: sienna-packing
description: Use when designing, building or verifying SIENNA's multi-job packing - several small jobs (images, layers or whole small models smaller than the mesh) sharing one N x N set through block-diagonal weights, the per-set block size, the PE's out-of-block skip, per-block activation and int8 output parameters, the packed lane order, the packer and unpacker in model_runner. Also use when someone asks how a large synthesized mesh (say N = 64) runs 2x2 / 8x8 / 16x16 workloads without idling, or why packed float results equal unpacked ones bit for bit.
---

# SIENNA: multi-job packing

**Status: implemented and verified 2026-10-04 on the packing branch (gated tree SIENNA 8d7a4b2, SystolicMesh b1a2c31; the commit after it changes only this skill) at N = 8-32; N = 64 after check-in.**
Spec approved 2026-10-04; "As built" at the end lists every deviation. The design before this work (2026-10-04) is tagged
`three_formats_v4` in SIENNA and SystolicMesh and `three_formats_v2` in GPNAE.

## The problem

A set is always N x N x N: N^2 words of A, N^2 of B, an N-deep product in every PE, N^2 elements through activation.
Nothing in the hardware knows a size below N. A user who synthesized N = 64 and has 8 x 8 work pads every job to
64 x 64. Today's best practice, `sienna_layer` stacking a model's inputs along M, fills the rows but leaves K and the
output columns at b / N use. Soham's requirement (2026-10-03): both cases, (a) one model and many small inputs and
(b) different small models at the same time, at every synthesized N, without per-job RTL parameters ("those are fixed
after synthesis"): everything per job is a runtime input.

## The idea: block-diagonal packing

Put model c's weights W_c (K_c x C_c, both <= b) on the diagonal of B, B = diag(W_0, ..., W_{N/b-1}), and model c's
inputs down column block c of A. Block (rows, c) of C = A B is then that model's output, X_c W_c. A set carries N/b
models across and as many inputs per model as there are rows, so case (a) goes down the rows and case (b) goes across
the blocks. b is a power of two, 2 <= b <= N; b = N is packing off.

## Evidence (no-RTL probe, 2026-10-04)

`sienna_jobs/pack_probe/pack_probe.py` (throwaway, outside git) on the unchanged RTL through `sienna_layer`: every
N <= 32 at every valid T (N = 8: T 2/4/8, 16: T 2-16, 32: T 2-32), b = 2 .. N/2, linear and tanh, fp32 / bf16 / int8;
farm runs `pk_probe_*` and `pkt_N*_T*_*` (`sienna_jobs/runs`, launch list `pk_launch.log`).

- Throughput: sets drop by exactly N/b against today's per-model M-batched layers, at every N, T and format (fp32
  tanh 1.96-1.97x where 2x is due: its activation time depends on the data). A set costs the same cycles packed or not.
- int8: 0 mismatches anywhere (packed = unpacked RTL = packed golden = each model's own golden).
- fp32 / bf16: the RTL equals the packed golden on every output (0 in 24 configurations), but from b = 4 up about 20%
  of outputs differ from the same job run alone, mostly by 1 ulp, with the same accuracy against float64; identical
  counts at every T. Cause (model and RTL agree): each PE accumulates into U = ADD_LAT + 1 = 6 rotating partial sums
  (`ProcessingElement.sv:34`, `SystolicMesh.sv:149`, `mesh_model.matmul`), product k into slot k mod 6, combined by a
  fixed pairwise tree with the bias. A packed job starting at K offset c * b starts mid-rotation, so its products group
  differently and float rounding differs. b = 2 always lands in an aligned pair; its 2 bf16 differences at N = 32 are
  the sign of an exact-zero result, the adder's known D-6 (x + (-x) = -0 when the first operand is negative).

## Decisions (Soham, 2026-10-04)

1. Packing first; mesh partitioning only if measurements later show its latency or energy are worth the rework.
2. Packed float results must equal the job run alone, bit for bit: the PE skips the products outside its job's block
   (option 2 of three; it also removes the adds on zeros, packing's energy cost).
3. Per-block activation by giving each lane a fixed column in packed sets (option 1 of two); GPNAE RTL is not changed.
4. Out of scope: jobs deeper than one pass (K > N); packed sets on the collapse-k 0 mesh; hardware pooling of packed sets.
5. Up to 8 distinct activation / int8 output parameter entries per set (a build-time capacity, not a per-job value).

## Runtime interface (per set, captured at accept like `activation_function_i`)

| Input | Meaning |
|---|---|
| `pack_shift_i` | b = N >> pack_shift_i; 0 is packing off and must behave exactly as today. Valid 0 .. log2(N) - 1 |
| table entry e = 1 .. P-1 | activation code; int8: requantize zero point, min, max, and the lane's gp_mx, gp_shx, gp_mout, gp_shout, gp_zout |
| `pack_map_i[N/2]` | entry (0 .. P-1) of column block c; blocks past N/b are ignored |

Entry 0 is today's per-set ports (`activation_function_i`, `req_zp_i`, `req_min_i`, `req_max_i`, `gp_*_i`), so a host
that never drives the new ports gets entry 0 everywhere and today's behavior. P = `PACK_ENTRIES` (default 8) is a
build capacity. Per column, unchanged: bias (`bias_i`), requantize multiplier and shift (`req_mult_i`, `req_shift_i`);
per set, unchanged: training mode, dropout seed, accumulate, weight cache select.

Storage estimate (arithmetic, not synthesized): an int8 entry is about 96 bits, so 8 entries plus a 3-bit map for up
to N/2 blocks is about 0.9 kbit per set id, about 14 kbit for 16 ids at N = 64. Per-column copies of every int8
parameter would be about 98 kbit at N = 64, which is why the table.

`sienna_layer` takes the same as layer configuration; `TB_model_run`'s layer file gains them. Packing is refused
(elaboration error, or an assertion on accept for runtime values) when: N does not divide NUM_LANES; the build pools
(POOL_H * POOL_W > 1) and pack_shift_i != 0; COLLAPSE_K = 0 and pack_shift_i != 0; a map entry is >= P; accumulate is
set with pack_shift_i != 0 (no multi-pass packing).

**A packing build needs NUM_LANES = N or a multiple of it.** The Makefile's default `LANES = 32` packs at N = 8, 16 and
32, but an N = 64 build needs `LANES=64` or `128`. In simulation a packed start on any other lane count fires
`a_pack_lanes`; silicon has no assertions, so the set would read past the result bank and come back silently wrong. The
host must not issue it: `model_runner.pack_precheck` (RtlLayer, before the layer file is written) refuses with ValueError
a lane count N does not divide, collapse-k 0 (`model_runner.COLLAPSE_K`), a residual input, a shift outside
1 .. log2(N) - 1, a map that is not N/2 entries in 0 .. 7, and a table that is not 8 entries.

## Mesh: the out-of-block skip

- pack_shift travels with its set through the mesh beside `fresh` and `more` (sets of different b run back to back).
- A PE knows its global column j at elaboration (tile column * T + local column). Product k (k = 0 .. N-1, the PE's
  count within the pass) counts only when k / b == j / b. Out-of-block products are not added; the multiplier is not
  fed either (`v` low into it), so neither unit toggles for them.
- The slot rotation advances only on counted products, and a slot's first counted product of a set adds to zero, so a
  packed job's slots hold exactly what they would hold alone. When b < U, slots b .. U-1 get no counted product and
  must read as +0 (what the job alone, padded with zero products, puts there).
- pack_shift = 0: every product counts; cycle- and bit-identical to today.
- The skip costs no cycles: products still arrive one per cycle; only the add is suppressed.
- Energy proxy: the TB counts issued multiplies and adds per set (packed: about b/N of today's). A proxy, not a power
  number; label it so in every report.

## Lanes: requantize, activation, dropout

- Requantize (`requant_lanes.sv`): zero point, min and max become per element, from the entry of the element's
  column block; multiplier and shift are already per column (`requant_lanes.sv:36`).
- Activation: a lane applies one activation code to its whole batch (`gpnae_poly.sv:133-141`, read live), and today
  lane k takes elements k * PER_LANE + i, a run along one row (`SystolicMesh.sv:343`, `sienna_top.sv:413`). In a
  packed set lane k instead takes column k mod N, rows g * PER_LANE + i with g = k / N (N | NUM_LANES gives
  NUM_LANES / N groups of PER_LANE rows = N rows). Each lane's batch is then one column, one model: it gets its block's
  entry (code and gp words) for the whole batch. The wide-read address (SystolicMesh) and the write-back index into
  `gpnae_out_mem` (sienna_top) switch formula for packed sets only; pack_shift = 0 keeps today's order exactly.
- `gpnae_out_mem` is written at each element's true index, so pooling and the output order see the usual row-major set.
- Dropout's drop value (D-5) follows each element's entry (activation and zero point). Training mode and seed stay per set.
- Synthesis risk: the packed wide read gathers a different address pattern from MeshOutputSram; the plan must check it
  against the output memory's banking before the RTL is committed.

## Software

- `model_runner`: `pack_jobs(models, N, int8)` picks the smallest power-of-two b >= max(K_c, C_c, 2) over the jobs it packs,
  puts each model in a column block (jobs of one model share the block, inputs stacked down its rows), assigns table
  entries (at most P distinct activation / int8 output settings per set), builds A and the block-diagonal B, bias,
  per-column multiplier and shift, and the map; `unpack` cuts each job's rows and columns back out. Jobs with
  K or C > N/2 are refused (ValueError). int8 input quantization stays per model: each block's A codes use that model's scale,
  and its zero point is already folded into its columns' bias.
- Golden: a packed job's golden is the job computed alone (`int8_layer_exact`, `exact_layer` with the build's T);
  with the skip they are identical by construction, so no packed golden is needed beyond the existing ones.
- `exact_layer` (regression.py's gemm section, was `gemm_sweep.exact_layer`) takes T; `--action pack` and
  `--action gemm` pass the build's tile (`make gemm TILE=`; gemm_sweep.py hard-coded 4).

## Verification (bottom up, N <= 32; one N = 64 sweep after check-in)

1. ArithmeticLibrary and GPNAE: no RTL change, so no new gate; their regressions must stay green on the packing tree.
2. SystolicMesh, every N <= 32 and every valid T, fp32, bf16 and int8: packed sets at every b with random distinct
   data per block (never identity, never repeated blocks): every output equals the job alone; pack_shift = 0 sets equal
   today's results and cycles; streams that alternate b, including 0, back to back; b = 2 and 4 (slots past b); the
   weight cache with packed B; the add counter.
3. SIENNA top, three formats: today's regression unchanged (29 fp32 / bf16, 37 int8 tests) with identical cycles, plus
   packed tests on a 1 x 1 pool build: mixed activations across blocks, mixed int8 entries, all 8 entries used,
   alternating packed and unpacked sets, training dropout with per-block drop values, and the refusals above.
4. Layer engine: the probe becomes a regression check (packed = alone, bit for bit, all formats, all N/T).
5. End to end: at least two different int8 TFLite models packed in one set, each bit-exact against TFLite.
6. Lint both tops; rerun the area estimate for the table and the lane-order muxes.

## Review focus (failure modes no single test above names)

- Per-set state keyed on set id: a packed set followed by an unpacked one must not inherit its map, entries or order.
- A column block whose entry is SELU with an int8 input range that saturates (x >= 487.29): refuse per entry, as today per set.
- Partial packing: fewer models than N/b blocks; empty blocks must produce zeros and cost nothing extra.
- Weight-cached packed B reused across sets with a different pack_shift.
- Float signed zero (D-6) on exact cancellations: allowed to differ only between +0 and -0, never in value.

## As built (2026-10-04)

Script and class names in this section are those of 2026-10-04; the `sienna-tooling` skill maps them to today's
commands (`pack_regression.py` is `regression.py --action pack`, `tflite_pack_run.py` is
`model_runner.py --action tflite --pack`, LayerSim is `RtlLayer`).

Verified tree: SIENNA `packing` 8d7a4b2, SystolicMesh `packing` b1a2c31 (not pushed; GPNAE and ArithmeticLibrary
unchanged). The final gate (`pkf_*`, `packing_gate.log` section 8) ran Task 9's full list on exactly those commits, after
the final review's fixes; the commit after 8d7a4b2 changes only this skill. Earlier gates: Task 9 on efe2b68 (+ 935221f
for collapse-k 0, `pkg_*`), Task 10 on 1d2cac8 (`pk10_*`). Run results in `sienna_jobs/runs`.

Deviations from the plan, each with its reason (the plan ledger has the full rulings):

1. **PE column origin is `COL0 = COLLAPSE_K ? j * T : 0`** (Task 3), not unconditional: under collapse-k 0 a PE's depth is
   T, and the PE's `COL < K` check fired. Collapse-k 0 never packs, so local column c is right there.
2. **The Verilator build deletes stale `packShift*.mem`** before copying stimulus (Task 3): the copy never deletes, and a
   packed test's file packed the next unpacked conv set.
3. **N = 8 runs skip conv** (mesh: `--group matmul`; SIENNA: no conv, and two `_train` tests whose generator rejects
   N = 8), as the pre-packing baselines did: the conv kernel depth 9 does not fit N = 8.
4. **The packed wide read is a second address formula into MeshOutputSram's existing read muxes** (Task 3, against
   `2026-09-27_synthesis_readiness.txt` item 3b); a future per-tile banking must route that pattern through its crossbar.
5. **`TB_requant_lanes` packs sets 1, 2 and 5** (Task 4), so an unpacked set starts at an odd multiple of the period and
   a `clear_i` mutant dies.
6. **Refusals and start-time checks are assertions on registered accept terms** (Task 5 and its fix round; Task 9 for
   two more). A testbench that drives an input at the clock edge is seen by the flops at that edge but by a concurrent
   assertion one edge later, against state the edge already changed; `host_accept` checks never fired. `sienna_top`
   keeps `acc_*` copies, the mesh `sa_*`, `wcw_q`/`rd_q`. The refusal of a packed set that continues a partial sum was
   added (the accumulate flag of the previous accept).
7. **`set_act` became `set_ents[id][0]`**; the int8 packed generator needed `HAS_BIAS` (Task 5).
8. **Regressions fail on any assertion firing** (Task 6 LayerSim, Task 9 both `regression.py` parsers): with
   `+verilator+error+limit` a firing changes neither the exit code nor the testbench's counts.
9. **`pack_jobs` does not reorder jobs** (Task 7): grouping is the caller's. A set costs its slowest activation, so a
   set mixing ReLU/linear with a polynomial activation loses the bypass path for its fast blocks.
10. **The TFLite packed run is N = 32 only** (Task 8): its groups need b <= N/2 = 16.
11. **SIENNA's collapse-k 0 regression skips the packed tests** (Task 9): that mesh refuses them (`a_pack_collapsed`
    fired in the first gate run).
12. **`make pack`** runs `pack_regression.py`, whose `--act` (one activation in every entry) and `--rows` (several
    row-tile counts) give the same-activation cycle sweep.
13. **The host refuses what the RTL only asserts** (final review): `pack_precheck` in `model_runner` raises ValueError
    for NUM_LANES not N or a multiple (an N = 64 build needs `LANES=64` or `128`), collapse-k 0, a residual, a shift out
    of range, and a map or table of the wrong size, so a packed layer never reaches a build that would compute it wrong.

**Gate (all on the farm, N <= 32):** Task 9 launched 94 gate runs, 93 configurations plus `pkg_ck0b_16_int8`, the rerun
of the one failure (`pkg_ck0_16_int8`, packed tests on collapse-k 0, deviation 11); 93 pass. Not counted there: the 8
`pkg_perf_*` measurement runs and the 11 audit runs (`pkg_vacprobe`, `pkg_neg_*`). Task 10 added 12 `pk10_*` runs
(`packing_gate.log` section 7 lists the three lints on one line). The final gate (section 8) is 106 runs on 8d7a4b2 +
b1a2c31, all pass: Task 9's 93 configurations, the 8 perf sweeps, `make model FMT=bf16`, `model_runner --engine sets`
(TB_sienna_model) and TB_sienna_multi in three formats; each identical to Task 9's baseline and to Task 9's own run,
the sets-engine and multi runs identical to the three_formats_v4 tree, `make model`'s first inference per model identical
to a 2026-09-30 run, and no assertion fired in any of them. Mesh sweep N = 8 (matmul), 16 (collapse-k 1 and 0) and 32 (T = 2-32) in
fp32, bf16 and int8: READY, every row of `mg_mesh16_*`, `mg_mesh32_*_T4`, `g3i_N32_*`, `g3i_N8_*` identical in result
and cycles. `TB_PE_pack`, `TB_PE_int8`, `TB_SystolicArray`, `TB_requant_lanes`: passed. SIENNA regression N = 16 T = 4
identical to `mg_reg_fp32/bf16/int8` (29/29/37 tests, plus 3/3/4 packed); N = 32 int8 identical to `mg_r32_int8`;
collapse-k 0 identical to `mg_ck0_16_int8`; N = 8 identical to Task 5's runs; N = 16 T = 2, 8, 16 and N = 32
fp32/bf16 pass (no earlier reference). `make pack` at every N/T and format: 0 mismatches, lines identical to Task 7's
`pk_lr2_*`. `tflite_pack_run.py` (N = 32, T = 2-16), `make tflite`, `make gemm QUICK=1` identical to their references;
lint clean both tops in three formats with the four rejections. No assertion fired in any run.

**Throughput (measured, `pack_regression.py --act`, N = 16 and 32, T = 4, bf16 and int8, linear and tanh, 2, 8, 32 row
tiles):** the baseline is each job alone as its own unpacked layer on the same build, padded to full N x N sets
(K = C = b rounded up to one N tile). A packed set costs exactly what one of its jobs costs alone, so packing N/b jobs
is exactly N/b times faster at every row count. Fits are exact (residual 0): cycles = fixed + R x per-set, identical packed and alone, e.g.
N = 32 bf16 tanh 190 + 530 R, bf16 linear 190 + 36 R, int8 tanh 171 + 366 R, int8 linear 173 + 36.5 R;
N = 16 bf16 tanh 118 + 157 R, int8 linear 98 + 16 R. The mixed-activation table of the default run shows less
(N = 32 bf16 b = 2: 9.2x for 16) only because a packed set costs its slowest entry.

**Energy proxy (not a power figure):** a packed set issues b/N of today's multiplies and adds: each PE counts only its
block's b products per pass (`a_pack_count`; `TB_PE_pack` counts the multiplier's valid strobes, fp32 / bf16 / int8).

**Area (estimate from the RTL's sizing, `sienna_jobs/pk_area_estimate.py`; no synthesis):** packing adds 1.2-3.0% of
storage bits (N = 16 / 32, all formats; most in the PE's pack pipeline). The lane-order muxes (packed wide-read address,
write-back index, block lookup) cost 0.1-0.3% of flop-storage area as address selects, up to 5-11% if synthesis must
double every word's sources. Per-lane parameters (Task 10): the stages copy their set's 8 entries at `g_accept`
(`g_ents`, and in int8 `ge_*`) and, in int8, on `p_accept`/`p_null` (`pe_*`: code, zp, zout for the drop value), as `g_mult`
already was, so each lane selects 8:1 behind one shared 16:1 per stage: about 84 kGE at N = 16 and 135 kGE at N = 32
in int8, 2 / 4 kGE in the floats (code only). Before Task 10 the lanes indexed {set id, entry}, 128:1 per field per
lane: about 0.9 MGE at N = 16 (47% of flop-storage area) and 1.9 MGE at N = 32 (30%) in int8, 24 / 49 kGE in the
floats, unless synthesis decomposed the index set id first. Assumptions: a mux2 bit = 2 GE, a flop bit = 5 GE, an
n:1 mux = n - 1 mux2 per bit, no sharing across lanes; estimates, not synthesis. The copy changed no result or cycle
(regression, `make pack` and TFLite pack runs identical to Task 9's, `pk10_*` in `packing_gate.log` section 7).

**Known gaps:**

- `sienna_layer`'s `a_pack_shape` has never been seen to fire: `pack_precheck` refuses its cases first, so only a
  hand-written layer file reaches it.
- `sienna_layer`'s assertions that mix its inputs (`cfg_load_i`, `a_valid_i`, `w_valid_i`) with state, `a_pack_shape`
  among them, are aligned only because `TB_model_run` drives 1 ns after the edge; an edge-driving host would be
  checked one edge late (not converted to registered terms, Task 9 audit).
- The packed TFLite models have full-range clamps only (TFLite folds their ReLU into the zero point); a non-trivial
  per-entry clamp is covered by `TB_requant_lanes` and `int8_packed_zp_random_nopool`, not end to end.
- Per-entry int8 SELU saturation is refused in `pack_jobs` only; the RTL has no per-entry check, and RtlLayer checks
  entry 0 (the set's activation).
- Untested: an unpacked set with `pack_map_i[0] != 0`; a weight-cached packed B reused with a different shift; packed
  sets interleaved with unpacked accumulate pairs; the outputs of empty blocks (partial packing) are not compared.
- `PACK_ENTRIES` is a parameter of `sienna_top` and `sienna_layer`, but 8 is hard-coded in `TB_model_run`,
  `model_runner.PACK_ENTRIES` and `regression.py`.
- Without a pre-packing reference (pass/fail only): SIENNA regression N = 16 T = 2, 8, 16 and N = 32 fp32 / bf16;
  N = 8 is compared only to Task 5's packing tree.
