---
name: sienna-packing
description: Use when designing, building or verifying SIENNA's multi-job packing - several small jobs (images, layers or whole small models smaller than the mesh) sharing one N x N set through block-diagonal weights, the per-set block size, the PE's out-of-block skip, per-block activation and int8 output parameters, the packed lane order, the packer and unpacker in model_runner. Also use when someone asks how a large synthesized mesh (say N = 64) runs 2x2 / 8x8 / 16x16 workloads without idling, or why packed float results equal unpacked ones bit for bit.
---

# SIENNA: multi-job packing

**Status: spec, approved in conversation 2026-10-04; not implemented.** Branch `packing` (SIENNA; SystolicMesh gets
one when its RTL changes). Tag `pre_packing_v1` (all four repos, 2026-10-04) is the design before this work.

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

`sienna_layer` takes the same as layer configuration; `TB_sienna_layer`'s layer file gains them. Packing is refused
(elaboration error, or an assertion on accept for runtime values) when: N does not divide NUM_LANES; the build pools
(POOL_H * POOL_W > 1) and pack_shift_i != 0; COLLAPSE_K = 0 and pack_shift_i != 0; a map entry is >= P; accumulate is
set with pack_shift_i != 0 (no multi-pass packing).

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

- `model_runner`: `pack_jobs(jobs, N)` picks the smallest power-of-two b >= max(K_c, C_c, 2) over the jobs it packs,
  puts each model in a column block (jobs of one model share the block, inputs stacked down its rows), assigns table
  entries (at most P distinct activation / int8 output settings per set), builds A and the block-diagonal B, bias,
  per-column multiplier and shift, and the map; `unpack` cuts each job's rows and columns back out. Jobs with
  K or C > N/2 are not packed. int8 input quantization stays per model: each block's A codes use that model's scale,
  and its zero point is already folded into its columns' bias.
- Golden: a packed job's golden is the job computed alone (`int8_layer_exact`, `exact_layer` with the build's T);
  with the skip they are identical by construction, so no packed golden is needed beyond the existing ones.
- `gemm_sweep.exact_layer` hard-codes T = 4 (`mesh_model.matmul(f, passes, N, 4, 1, b)`); it must take T.

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
