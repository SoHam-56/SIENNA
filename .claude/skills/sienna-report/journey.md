# SIENNA journey notes, 2026-09-19 to 2026-09-28

Source notes for the journey section of the report. Every number carries a source; "not measured" means no
source gives one. All cycle counts are Verilator RTL simulation. GFLOPS figures in the sources assume 950 MHz,
which no timing run supports.

Conventions:
- Repos: S = SIENNA, SM = SystolicMesh, G = GPNAE, AL = ArithmeticLibrary (SM's copy).
- Dates are commit author dates (UTC-7). History file dates are local write dates and can differ by one day.
- `H:<name>` = `history/2026-09-xx_<name>.txt`. `B2B` = `sienna-back-to-back/SKILL.md`. `UF` = `sienna-uniform-format/SKILL.md`.
  `VER` = `sienna-rtl/references/verification.md`. `TODAY` = the caller's 2026-09-28 measurements.
- Config shorthand: N matrix size, T tile size, L activation lanes. "serial" = one set at a time, cycles from
  start to completion. "steady" = mean completion gap of a streamed run. "lat" = one set through an empty pipeline.
- Unless stated, N=16, T=4, fp32. Lanes: 8 until 1e67cb0 (2026-09-21), 16 until 4929054 (2026-09-24), then 32
  (64 at N=32).

## A. Architecture changes by theme

### A1. Day one and the fixes that made it work at all

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| baseline | S | 0eeac7e (dated 2026-07-27) + e988875 | Day-one tree, rebuilt with only the build-order fix | Reference point | N=16 T=4, one set at a time, load + compute cycles: ident_selu 271 + 2453 (9 wrong outputs); random_sigm 271 + 10578 (pass); random_tanh 271 + 18738 (pass); conv_basic_selu never completes (200000-cycle timeout) | TODAY |
| 09-19 | G | 975820a (S bump c3200e4) | SELU operand race in TYTAN MAC fixed: GPNAE FSM owns the FIFO read pointer, MAC starts on explicit request, reads a captured operand. Same diff changes LAMDA_ALPHA 3FD62D7D (alpha) -> 3FE10966 (lambda*alpha) | MAC ran 75 computations for 90 elements with mixed operands; negative SELU ~4.8% low | 90 pops for 90 elements; GPNAE regression failures 10 -> 1 | commit 975820a; known-issues 5b |
| 09-19 | G | 6927558, ddf1bb5, 88589c4, 515abb9, 0cd0302 | FIFO pop lost on a coincident push; exact cancellation gave -1.0 / -0 (now +0); tri-state numerator mux in sigtan; duplicate cntlz8; dead SELU code | Silent wrong results, synthesis hazards | Together with 975820a: GPNAE regression PASS 9/9 patterns, 0 failures in 6480 checks (was 10 failures) | commit c3200e4 |
| 09-19 | G | 7c5ca62, 02dd2fd, 7ab835c, 734e439 | Format-generic float encoding, golden model, 9 stimulus patterns, regression driver, file-driven tolerance TB | No golden values existed; `$bitstoshortreal` made every compare pass | Enabler; not a perf change | commits |
| 09-19 | SM | e1ff46a (S eefdcc6) | north_queue_empty_o compared ptr_B against 1 (west used 0) | North queue read empty with one element held | not measured | commit e1ff46a |
| 09-19 | S | e988875, 7ca7bec | Compile TB package before TB; drop two tests using nonexistent activation code 0 | Build order not guaranteed; those tests could only time out | not measured | commits |
| 09-20 | S/SM/G | 40a9d2b, 071d4b6, e5b18e7; 39cf1e9, e99220d | Parallel Verilator C++ compile; "did not run" reported separately from FAIL | Single-threaded compile; stale builds read as RTL failures | Build time not measured | commits |

### A2. GPNAE activation engine and the stages around it

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-20 | G | e35dca8 (S d3c0bc2, 49c2cdb) | Barrel MAC: K=16 elements rotate through one multiplier and adder in lockstep on the term index; replaces mac/controller/datapath | Controller issued one op per 15 cycles, multiplier ~6% busy; Horner has a 13-cycle loop per element | Per element (GPNAE TB): SELU 139 -> 41, sigmoid 315 -> 87, tanh 540 -> 102. Full pipeline tanh 18,605 -> 4,594 serial. matmul_ident_selu 1,966 -> 2,546 (worse: SELU positives now use the MAC). Results bit-identical (worst 1.2658%) | commit e35dca8 |
| 09-20 | S | adfd2d8 | Lane fill FSM writes one element per cycle (F_GAP removed) | Lane 7 waited 7*32*2 = 448 cycles | GPNAE_ROUND fixed startup 397 -> 141 cycles (tanh and conv_basic_selu); end-to-end not stated | commit adfd2d8 |
| 09-20 | G | 0559f3a (S 20a311e) | gpnae_poly lane: activation fitted directly as a polynomial; no exponential, no divider, no e^x - 1 cancellation | Divider 52 cycles and unpipelined; cancellation caused the 1.2658% worst case | Worst error 1.2658% -> 0.3769%; per element SELU 41 -> 30, sigmoid 87 -> 23, tanh 102 -> 39 (GPNAE TB). Full-pipeline effect not measured on its own (see C9) | commit 0559f3a |
| 09-21 | G | f9c40f4, 567d986 (S 45bfde9) | fp32_down stage values carried per stage; poly lane streams its squaring and post stage | fp32_down corrupted back-to-back operands (act_negative 218/240 failed, errors to 537%); G_SQW 256 + G_POSTW 288 of 1126 busy cycles were waits | Serial, L=8: tanh 2074 -> 1584, SELU 1786 -> 1532, sigmoid 1580 -> 1468 | commit 45bfde9; H:latency_analysis §4 |
| 09-21 | G | 13062c6, 790a33e (S a9bf77c) | FIFO occupancy output; lane pops its whole group back to back | Capture cost 5 cycles per element, 4 of them waiting | Per element SELU 22 -> 18, sigmoid 20 -> 16, tanh 24 -> 20. Serial, L=8: tanh 1232 -> 1108, SELU 1180 -> 1056, sigmoid 1116 -> 992 | commits 790a33e, a9bf77c |
| 09-21 | S | 1e67cb0, 3f9818c | Per-lane share PER_LANE = SRAM_DEPTH / NUM_LANES (was GPNAE_FIFO_DEPTH, equal only at 8 lanes); lanes 8 -> 16; constraints checked at elaboration | Latent coincidence bug; one barrel-MAC group per lane at 16 | Serial: tanh 1108 -> 805, SELU 1056 -> 779, sigmoid 992 -> 747 | commit 1e67cb0 |
| 09-21 | S | 6f4b991 | Window dispatcher writes all lanes per cycle (was one lane per cycle); also the commit that instantiates gpnae_poly in sienna_top | Dispatch was width-limited: 324 writes in 325 cycles | Serial: tanh 1584 -> 1374, SELU 1532 -> 1322, sigmoid 1468 -> 1258 | commit 6f4b991 |
| 09-21 | S | fee5d9c | Maxpool streams (running max) when the input is one segment | Batch FSM cost ~8 cycles per 2x2 window | Serial: tanh 805 -> 776, SELU 779 -> 750, sigmoid 747 -> 718 | commit fee5d9c |
| 09-23 | S | 558e57f (uses SM bd05045 wide read port) | Parallel lane fill: 16 wide reads, every lane starts together; FIFO1 and lane-by-lane fill removed | Activation stage 97% occupied, lanes 36-48% busy; result read one word per cycle | L=16: lat tanh 777 -> 519, SELU 751 -> 493; steady tanh 645 -> 459, SELU 565 -> 336; lanes 93% busy. Model beforehand: ~240 cycles per set, 1.7-2.2x | commits 558e57f, 5903550 |
| 09-23 | G | 385b9ca (S 8db9eb2) | gpnae_tail: exact activation for inputs past the fits (halve, short Taylor, square back) | Fits diverge past abs(x) ~4; a random seed gave tanh = 19.32 (true 0.99983) | Poly lane passes at +/-100, worst 0.37%. A tail element costs up to ~250 cycles against ~20 | commit 385b9ca |
| 09-23 | G | b972abd (S 4197f5d) | Tail element starts at capture and runs beside the polynomial | One tail element doubled its lane's round | Steady, L=16: SELU 336 -> 266, tanh 459 -> 303, sigmoid 472 -> 325; large-value configs 2449-3896 -> 2235-3659 | commit 4197f5d; H:pipeline_performance_report |
| 09-24 | S | 4929054 | Default 32 lanes (8 elements per lane) | Activation was the limit after one-row host writes | Activation SELU 251 -> 195, tanh 277 -> 213. Steady SELU 264 -> 221, tanh 303 -> 271, sigmoid 325 -> 263; large configs 1.5-1.8x. (1e67cb0 had projected ~10% for 32 lanes) | commit 4929054; H:architecture_changes §1 |
| 09-24 | G | e68882c (S bc8c199) | gpnae_tail runs 4 contexts sharing one multiplier and one adder | Tail ~270 cycles per element, one at a time | Large-value, L=32: steady 1510/1981/2078 -> 530/566/591 (SELU/sigmoid/tanh); lat 1431/2230/2345 -> 627/666/691. N=32 L=64: SELU 772 -> 288, tanh 1122 -> 380. GPNAE regression totals 454,222 -> 193,633 (+/-12), 911,699 -> 303,480 (+/-100), bit-identical | commits e68882c, bc8c199; B2B "Larger N" |
| 09-25 | G | f1482f4 (S c0deb53) | ReLU (100) and linear (101) modes; 3-bit control word; activation latched per set | Networks need them; layers in flight need their own activation | Functional; GPNAE regression 36/36, same worst error | commits |
| 09-25 | S | 96a1a0f | ReLU/linear sets bypass the lanes | Exact modes need no polynomial | Activation per set at N=16: 33 -> 11 cycles | commit 96a1a0f |
| 09-25 | S | 8ac0f0e | POOL_BYPASS: 1x1 pooling skips FIFO2 and Maxpool | 1x1 stride 1 is the identity | Pooling per set at N=16: 26 -> 10 cycles | commit 8ac0f0e |

No "pipelined divider" change exists in any repo (see C8).

### A3. Streaming and back-to-back sets

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-21 | SM, S | c083ee7, 8313027 | PE accumulator cleared per pass (from the drain pulse); BACK_TO_BACK test added | MAC cleared only on reset | Serial cycles unchanged (776/750/718). Second matrix never starts: 200000-cycle timeout | commits |
| 09-22 | SM | 01e5c0c, 223eb61 | Mesh TB: one reset, distinct random data per set; wait for completion edge | Per-set reset hid re-arm defects; held completion level made later sets read 0 cycles | Red as intended: set 0 passes in 117 cycles, sets 1-4 "complete" in 0 cycles | commits |
| 09-22 | SM | bdc8850, c451717, 9b9b17a, 7e1e9ac, 1b1b176, 36b2730 (S 9d8ebae) | Six re-arm defects fixed: OutputSram COMPLETE terminal, AccumulationUnit RDONE, queue read pointers, PEMesh done latches, staging write pointers (9-bit wrap 512 -> 0), ungated collection_complete_o | Every completion flag was cleared only by reset | Second matmul had traversed the mesh FSM in ~23 cycles without computing. After: TB_SystolicMesh 5 distinct sets with one reset pass | commits; H:back_to_back_findings; B2B |
| 09-22 | SM | 5139b15, e620533, b5295bf (S eac34a7) | Mesh streaming: input_ready_o, result_release_i, double-buffered staging and result memory, queued start | Host and consumer could not overlap the mesh | 5 sets N=16 T=4: streamed 1765 cycles vs serial 3769 (serial includes TB gaps); host loads while mesh busy 796 cycles. Queued start +1 cycle per set (203 -> 204); each SIENNA test +1 cycle | commits b5295bf, eac34a7 |
| 09-22 | S | 44af72d | TB streams K=4 distinct sets; timeout fails the regression | Serial top could not be tested for overlap | Red: 4 sets correct in 4150 cycles, no overlap, every set id 0 | commit |
| 09-22 | S | 3a8c7ed | Single outer FSM -> activation and pooling stage controllers (with the mesh: three); gpnae_out_mem two banks; per-stage clears; credit interface (A4) | Stages ran strictly in sequence: 486 + 203 + 84 + 32 = 805 exactly | tanh: single set 776 (unchanged); 4 streamed sets 2642 vs 4150 serial (1.57x; projected 1.6x in H:latency_analysis §6b); mesh busy under activation 606 cycles, act/pool overlap 93, up to 3 in flight | commit 3a8c7ed |
| 09-23 | S, SM | 8b62fd6, fa8cab0, 12464f9; SM 9513852, b43e4d8 | result_valid_o; 9 top + 4 mesh handshake assertions, --assert on; no-credit start, mid-stream reset, overrun tests | Nothing proved the handshake; no assertion had ever run | No assertion fires. Activation never stalls on full banks; mesh stalls only in the overrun pass (159-217 cycles) | commits; B2B |
| 09-24 | S | 4e634c3 | Act/pool overlap required only when reachable | At T=16 the mesh (365) is slower than activation + pooling | T=16 11/11 (0 overlap cycles), T=4 11/11 (48 overlap cycles) | commit |

### A4. Handshake to credit-based interface (most important theme)

What it was before. Day one had no ready output and no flow control. `start_pipeline_i` was a pulse taken only
in the outer FSM's IDLE, gated on both mesh queues being non-empty; IDLE had no start latch, so a pulse at any other
time was lost. `pipeline_complete_o = (current_state == PIPELINE_COMPLETE)`, a held level (0eeac7e
src/sienna_top.sv line 716; H:back_to_back_findings). The mesh had the same shape: a start pulse and a completion
level held in DONE. It was not a ready/valid interface (see C1). One set at a time.

What it is now. `pipeline_ready_o = (credits != 0) && mesh_input_ready`; a start is accepted only then (and with
the staging queues non-empty); a credit is spent per accepted start and returned when the set's last output leaves
pooling/dropout; `pipeline_complete_o` is a one-cycle pulse with `done_set_id_o`; results leave in issue order by
construction (in-order stages, no reorder buffer). Credits: `MAX_SETS_IN_FLIGHT = 3` localparam (3a8c7ed) ->
`SETS_IN_FLIGHT` parameter 7 (c53234e) -> 8 (382dcad) -> derived 2 staging + 2 operand + ACC_BANKS + RESULT_BANKS +
2 activation + 1 pooling = 15 with ACC_BANKS = RESULT_BANKS = 4 (78014e4). Set id width `$clog2(SETS_IN_FLIGHT+1)`:
2, 3, 4, 4 bits. sienna_layer (e19c249) puts ready/valid input streams in front of this interface.

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-22 | SM | b5295bf | Mesh-level ready: input_ready_o = a staging bank is free; writes while low are dropped | Double-buffered staging makes "credit free" imply "safe to write" | See A3 | commit; B2B |
| 09-22 | S | 44af72d | pipeline_ready_o, done_set_id_o as stubs (ready = state == IDLE) | Failing test first | 4150 cycles for 4 serial sets | commit |
| 09-22 | S | 3a8c7ed | 3-credit interface, completion pulse, 2-bit set id | Overlap mesh, activation and pooling on different sets | 2642 vs 4150 cycles for 4 tanh sets (1.57x); up to 3 in flight | commit |
| 09-23 | S | fa8cab0, 12464f9 | Credit-counter assertions; start with no credit must be ignored (16 words loaded, mesh write pointer stays at 16) | Prove the contract | No assertion fires; set 3 still correct | commits |
| 09-23 | S | 5903550 | First per-stage perf trace on the credit design | Find the limit | L=16: lat 751-1003, steady 565-645 cycles per set; activation 97% occupied | commit |
| 09-23 | S | 558e57f | Credit-overrun check reworked | Fast data no longer puts 3 sets in flight: "the pipeline drains as fast as the host loads" | Check skipped when unreachable; matmul_large_* still exercise it | commit |
| 09-25 | S | c53234e | SETS_IN_FLIGHT parameter, default 7; per-set tables sized to it | With the pipelined mesh a set spends ~130 cycles in flight, so 3 credits capped the rate at ~50 cycles per set (latency / 3; derived, not an isolated run) | Isolated effect not measured. Combined with the pipelined mesh and bypasses (46e07de): ReLU steady 88 -> 19-21 (N=16), 104 -> 35 (N=32) | commits c53234e, 46e07de; B2B |
| 09-25 | S | 382dcad | Default 8 credits | Set ~104 cycles in flight at N=16 with the 4-bank mesh | Isolated effect not measured | commit |
| 09-25 | S | 501ea33 | TB_sienna_model streaming host: last row with the start, next set the cycle after (as a DMA); +host_gaps keeps TB_sienna_top's handshake | Handshake host costs N+3 cycles per set (enable drop, start, credit settle) | TB_sienna_top host: 19 cycles per set at N=16, 35 at N=32 (measured = model, 7ae1e7d). Streaming host: 256x256x64 GEMM 17.4 (N=16), 32.0 (N=32) cycles per set. ResNet-8 N=16 61,344 -> 56,382, N=32 23,141 -> 21,284 (same batch as the 4-bank mesh and 8 credits, not isolated) | commits 501ea33, e1ad49d, 5a46444; H:nt_sweep; GPNAE_review §4 |
| 09-26 | S | 5ca57e8 | Stall-cause counter in TB_sienna_layer (PERF) | Find N=16 per-set overhead | With 8 credits: credit stalls 934 of 10,336 cycles on a ResNet-8-like layer, 2,622 of 39,816 on a K=576 layer; 0 at N=32 | commit; H:SIENNA_GPNAE_review §6 |
| 09-26 | S | 78014e4 | SETS_IN_FLIGHT derived from bank counts (15) | Credits must never be the tighter limit | No speedup: layer 10,336 cycles unchanged, ResNet-8 56,902 unchanged; stalls moved to staging banks (881 at N=16 T=4 conv3x3_16ch_bias_relu). Credit stalls 0 at every N (8-64) and T in the sweep | commit; H:nt_sweep; H:SIENNA_GPNAE_review §12 |
| 09-26 | S | 9877256 | Downstream-hold counter | Credits were a symptom | Activation stage busy 9,538 of 10,336 cycles with a result waiting (N=16); nothing held at N=32. Led to e8d0da6 (A6) | commit |
| 09-26 | S | 275d7d8 | Tests with 3 credits (matmul_relu_nopool_credits3, matmul_tanh_credits3) | With 15 credits no stock test hit a no-credit start | Only isolated measurement of credit count: ReLU no-pool 39.3 cycles per set with 3 credits vs 19.0 with 15 (N=16), 55.3 vs 35.0 (N=32); tanh 263.0 with either (activation-bound) | commit; H:pre_synthesis_v1 |

### A5. Mesh

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-21 | SM | 3939d40 (S f28d103) | C-slow AccumulationUnit: pixels rotate through the reduction adder | 9 cycles per (pixel, k); reduction 70% of the mesh | Mesh N=16 serial: T2 399 -> 167, T4 849 -> 345, T8 1797 -> 781, T16 3885 -> 1845. Pipeline serial: tanh 2578 -> 2074, SELU 2290 -> 1786, sigmoid 2084 -> 1580 | commit 3939d40 |
| 09-21 | SM | c4dc2f6 (S 229273d) | PE releases passthrough operands before its MAC retires (FORWARD state) | Wavefront advanced one PE per MAC (~20 cycles) | Mesh T4 345 -> 238 (T2 167 -> 132, T8 781 -> 530, T16 1845 -> 1306). Pipeline: tanh 1374 -> 1267, SELU 1322 -> 1215, sigmoid 1258 -> 1151 | commit c4dc2f6 |
| 09-21 | SM | 3f695b7 (S a37107f) | PE multiply overlaps its accumulate | Only the accumulate is loop-carried; op cost ~15 cycles | Mesh T4 238 -> 203 (T2 132 -> 117, T8 530 -> 455, T16 1306 -> 1151). Pipeline: tanh 1267 -> 1232, SELU 1215 -> 1180, sigmoid 1151 -> 1116 | commit 3f695b7 |
| 09-23 | SM | bd05045 | Wide result read port (WIDE_READ words per read) | Enables the parallel lane fill (A2) | Mesh 68/68 | commit |
| 09-23 | SM | 1521df9 | Parallel adder-tree reduce of the partial tiles | Serial reduce took P x T^2 adds | Mesh set 0: T2 118 -> 78, T4 204 -> 143, T8 456 -> 326, T16 1152 -> 889 | commit |
| 09-23 | SM | 5e27feb (S 5b1d684 with 1521df9) | Broadcast one tile row per cycle | Broadcast cost T^2 cycles | Mesh set 0: T4 143 -> 131 (T2 78 -> 76, T8 326 -> 270, T16 889 -> 649). In pipeline: mesh stage 202 -> 129, every lat -73, steady unchanged | commits |
| 09-24 | SM | 1d1397e (S c1e6224) | HOST_WORDS per host write (one row) | Loading a set cost N*N = 256 cycles | Host load 257 -> ~17; SELU steady 266 -> 264 (activation now the limit) | commit c1e6224 |
| 09-24 | SM | 2ce28cb, 36275ba (S 5e39d51) | SyncArray: synchronous tile, operands move one PE per cycle, 6 partial sums per PE; no valid joins, per-PE FSM or drain wave | Handshake tile overhead | Tile phase 94 -> 42; mesh set 0 T2/4/8/16: 76/131/270/649 -> 56/79/146/365; pipeline mesh stage 129 -> 75 (commit) or 77 (report, C12); lat SELU 421 -> 295, tanh 447 -> 313 (C11); steady unchanged 221/271/263 | commits; H:architecture_changes |
| 09-24 | SM | 105a62b (S 38e2ec2, 157e40f; SM da1a6ef) | Collapse-k: (N/T)^2 tiles of depth N, N^2 PEs, no reduce; default since 09-24 | Area: N^3/T PEs with depth slices | N=16 T=4: 256 PEs instead of 1024; set 0 89 vs 79; pipeline lat +10; steady unchanged. N=64 T=8: 197 cycles with 4096 PEs vs 840 with 32768 (old serial reduce) | commits; B2B |
| 09-24 | SM | db01911 (S c599d50) | Only the synchronous tile kept (handshake tile at tag legacy_tile_v1) | Cleanup | Mesh 68/68 both modes | commit |
| 09-25 | SM | c85b431 (S 2f57700) | Pipelined mesh: broadcaster, arrays, reducers concurrent; two operand banks; 3 partial-sum banks per PE; reducers combine partials. Serial mesh at tag serial_mesh_v1 | A set cost 87 cycles at N=16, 16 of them products | 4x4 array: 18 cycles per set at depth 16, 66 at depth 64 | commit c85b431 |
| 09-25 | SM | e7cb172 | Third result bank | Two banks limited N=16 to ~22 cycles per set | 19 cycles per set (1x1 pooling, full pipeline, N=16) | commit |
| 09-25 | S | 46e07de (combined result of 2f57700, c53234e, 96a1a0f, 8ac0f0e) | Testbenches follow the pipelined mesh | | ReLU steady 88 -> 19-21 (N=16), 104 -> 35 (N=32). GEMM peak 46.5 -> 215 MAC/cycle (18% -> 84%) N=16, 315 -> 936 (31% -> 91%) N=32. ResNet-8 298,383 -> 65,619 (N=16), 67,103 -> 23,456 (N=32) | commit 46e07de |
| 09-25 | SM | e8119be (S c63ccac, 5a46444) | Per-set bias added as one more reducer-tree input | A bias row of ones cost a whole pass when depth is a multiple of N | 1-16% fewer sets on MLPerf Tiny. N=16: ResNet-8 65,619 -> 61,344, KWS 40,469 -> 35,586, VWW 130,266 -> 116,777, ad01 22,825 -> 20,982. N=32: 23,456 -> 23,141, 17,210 -> 14,935, 67,294 -> 60,526, 12,481 -> 10,766. Cost: +5 cycles lat where RP*U is a power of two (one more tree level) (depth slices 50/70/75 -> 55/75/80) | commits; SM 8809c21; H:SIENNA_GPNAE_review §4 |
| 09-25 | SM | fa61ddc, f3952ca, e1ad49d (S 382dcad) | Last row taken with the start; reducers restart with no gap (T^2+1 -> T^2 per set); 4 partial-sum and 4 result banks | Partial-sum bank returns ~3K cycles after start: 3 banks allowed a set every ~18 cycles | 256x256x64 GEMM, streaming host: 18.6 -> 17.4 cycles per set (N=16), 32.0 at N=32 (every PE busy) | commit e1ad49d |
| 09-25 | SM | a601a95 (S 679379c) | Weight cache, WC_TILES = 128 N x N tiles, two regions | Operands had to come from the host every set | Via sienna_layer: host words 32-45% fewer; cycles within 0.2-3.5% of host-driven | commit e19c249 |
| 09-26 | SM | 8809c21 | Mesh latency closed form 3T + K + T^2 + LAT + 17, LAT = 1 + 5*ceil(log2(RP*U + 1)) | Documentation | Matches every measured entry (e.g. 77 at N=16 T=4, 93 at N=32 T=4, 4385 at N=64 T=64) | commit; H:sienna_full_comparison §2 |
| 09-26 | SM | 407b2bc (S d026de4, 012b013; tag pre_synthesis_v1) | partial_i: a deep product's sums stay in the PEs across depth passes; acc_mem, its NUM_LANES fp32 adders and three activation states removed | Partial sums left the mesh and were added downstream | ResNet-8 52,652 -> 52,004 (N=16), 21,764 -> 21,572 (N=32); VWW 101,092 -> 98,684; GEMM via sienna_layer N=16 256.0 MAC/cycle, 100% of peak (was 93.6%) | commit d026de4; H:sienna_full_comparison §5-6 |

### A6. Layer engine, multiple pipelines, model support

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-24 | S | 3d6c4d3, 77dfe77, SM 250b1da | Lane count and tile size configurable; conv tests for any N | N=32 had no conv tests | N=32 L=64 conv 3/3; mesh N=32 conv 45/45 | commits |
| 09-24 | S | fea6c5d, 38e2ec2 | sienna_multi: COPIES pipelines behind one host port, round-robin; collapse-k in each copy | Activation-bound single pipeline | L=16: tanh 307.5 / 149.0 / 73.4 cycles per set (1/2/4 copies); SELU 264.0 / 66.0 (1/4). 4 copies L=32: tanh 62.6, SELU 55.2 with or without collapse-k; lat 295 -> 306. N=32 L=64 4 copies: tanh 79.0, SELU 71.8. Host port limit ~15 copies (estimate) | commits; B2B |
| 09-24 | S | 8ab0751 | Accumulate mode: partial sets summed in acc_mem before activation | Layers deeper than N | Functional; existing 11 tests same latencies | commit |
| 09-25 | S | c0deb53, d57553d | Activation and terms latched per set; ReLU, linear, 1x1 pooling, mixed-activation tests | Overlapping layers need per-set activation | Regression 19/19; four copies 62.6 cycles per set unchanged | commits |
| 09-25 | S | 98454b8 | model_runner: tflite models end to end on the RTL (im2col in software) | First real workloads | Cycles per inference N=16 / N=32: ResNet-8 298,383 / 67,103, KWS 181,709 / 47,909, VWW 586,075 / 191,635, ad01 100,990 / 34,198 per slice; top-1 equal to float reference | commit; H:model_runs_report |
| 09-25 | S | bb998cb (tag serial_mesh_v1) | gemm_sweep, 67 shapes | Model-agnostic benchmark | Baseline: N=16 peak 46.5 MAC/cycle (18%), N=32 315 (31%); 88 / 104 cycles per set | commit; H:gemm_baseline_N16 |
| 09-25 | S | e19c249, 74ff3a7 | sienna_layer: layer scheduled in hardware (tile loops, partial/last pass, bias, residual identity pass, weight caching, per-set seed); ready/valid input streams | Software only configures and streams | ResNet-8 56,902 (N=16), 21,764 (N=32); GEMM within 0.1% of host-driven; N=16 large GEMM 239.5 MAC/cycle (93.6%) | commits; H:layer_engine_performance |
| 09-26 | S | 7ae1e7d | perf_analysis design model (host N+3, mesh interval max(T+2, K, T^2), bypassed activation N^2/L + 4) | Explain measurements | Measured equals model on all 24 tests at N=16 and 32 | commit |
| 09-26 | S | e8d0da6 | Partial set leaves the activation stage once its beats are taken (ACC_OVERLAP) | Partial set held activation N^2/L + 9 = 17 cycles > 16-cycle set | Layer 10,336 -> 9,391; staging stalls 881 -> 0. N=16: ResNet-8 56,902 -> 52,652, KWS 33,831 -> 30,522, VWW 109,988 -> 101,092, ad01 19,363 -> 17,909. N=32 unchanged | commit e8d0da6 |
| 09-26 | S | d026de4 | Partial sums moved into the mesh (A5); acc_mem removed | | See A5 | |
| 09-28 | S | f1cc285 | perf_analysis counts an accumulate group as one output | accum3 read a 2-cycle latency and over 100% PE use | bf16 N=16 T=4 accum3: 133 cycles lat, 19 per depth pass. Accumulate figures before this commit are unreliable (C20) | commit |

### A7. Number format (bf16)

| Date | Repo | Commit(s) | What changed | Why | Measured effect | Source |
|---|---|---|---|---|---|---|
| 09-26 | AL, SM, S | AL 49e484b; SM 1302351; S d843c84, fe0ce00, 032ae24, 12136d5, 3af9f5c (bf16 branch) | Mixed bf16: A/B operands in bf16 (fpMulWiden, exact product into fp32); sums, GPNAE, pooling, results fp32 | First narrow format | fpMulWiden 0 mismatches (203,366 bf16 / 200,546 fp16 products). fp32 cycles identical on 27 tests; bf16 27/27 at N=16, each set 5 cycles shorter (multiplier 8 -> 3). Storage estimate N=16 207 -> 133 KiB (estimate, not synthesis) | commits; H:synthesis_readiness §5 |
| 09-27 | S | 55f3a8c | Uniform-format design approved: one format per build (EXP_W/MAN_W), truncating units, format package, bit-exact golden | Replace mixed precision | Design only | commit; UF |
| 09-28 | AL | 77550e1 ... d97e270 | sienna_fmt_pkg, fpu.py model, fpMultiplier and fpAdder at any width (truncating), SoftFloat and Vivado TBs | Narrow units with fp32-grade DV | G1 PASS: at (8,23) 205,776 vectors identical to fp32 units; bf16 exhaustive 2^32 pairs, 0 errors per unit; mul_lat 3 (bf16) vs 8, add_lat 5 | H:aril_gate |
| 09-28 | G | b2d6f10 ... 41cad2d | barrel_mac, gpnae_tail, gpnae_poly in the build's format; bf16 coefficient refit (degrees 3/5/4 vs fp32 8/6/8) | fp32 degrees gave bf16 worst errors 5.3% / 21.7% / 331% | G2: bf16 bit-exact against model; worst error SELU 2.82%, sigmoid 7.22% (12.68% range 8), tanh 8.59% vs 6.25% tolerance (open decision). GPNAE TB cycles per input fp32 -> bf16: 18 -> 18, 16 -> 16, 20 -> 19 | H:gpnae_gate; H:poly_coeffs_fit |
| 09-28 | SM, S | SM 7b925b9 ... 1cdfd49; S 1203cc6, 157d13a, 5f0a2ab, 04d5768, bee150d, 44c3e6f, b07ec38, a27f96e, 85e4150, 9f604d2, f46b75c, 663443b, f487ef0, 40a5015, b8ec8cb | Mesh, Maxpool (float compare, -inf pad at any width), dropout, top levels, testbenches, regression, model_runner, gemm_sweep, perf_analysis in one format; bit-exact mesh model; zero-row and signed-zero tests | Uniform bf16 end to end | N=16 T=4 cycles per set fp32 -> bf16: ident_selu 215 -> 157, random_sigm 273 -> 198, random_tanh 263 -> 190, conv_basic_selu 215 -> 157. bf16 GEMM error vs float64 grows with K: 6% at 256, 23% at 1024, 50% at 3072 | TODAY; UF |

### A8. Verification and synthesis readiness after day one (short)

| Date | Repo | Commit(s) | What changed | Measured effect | Source |
|---|---|---|---|---|---|
| 09-22 | SM, S | e20b168, 5d26a19 | Float compare decoded by hand | Old checks accepted 0.43x to 2.5x of expected; mesh 68/68, SIENNA 5/5 after | commits |
| 09-23 | S, SM, G | d0b6c58, a016e6e, c20ca51 | ccache off by default | 11 of 14 cached full regressions crashed vs 0 of 4 uncached | commit a016e6e |
| 09-23 | S | 2c16673, 54e0fde | Dropout training mode: 4 defects fixed, per-set seed table by set id | Both-kept 0.621 vs 0.559 before, 0.247 vs 0.249 after; 64 -> 300 of 300 distinct masks; 11 tests pass on six seeds | commits |
| 09-26 | S | 747d03b | Pass criterion adds a per-output fp32 error bound | N=64 T=8 nopool 6/6 (3 false failures before) | commit |
| 09-27 | S, SM | main 13df7d9, 7ff1880, 66ed4e5; bf16 63e9e30, aa3a9f1, 2cab850; SM b45e5b3 / 8b2dd12 | Closed-form lane advance (was a 32-deep compare-subtract chain), no reset net in result-memory writes, synthesis file list | Same cycle counts on 27/27; 27/27 with random power-up state | commits; H:synthesis_readiness |

## B. Plottable series (one metric and config per series)

B1. matmul_random_tanh, N=16 T=4, fp32, serial cycles (start to completion, one set at a time). L=8 until 1e67cb0, then 16.

| Date | Step | Value | Source |
|---|---|---|---|
| 2026-09-19 | Day one (0eeac7e + e988875) | 18,738 (+271 load) | TODAY |
| 2026-09-20 | Barrel MAC | 4,594 | e35dca8 (test name not stated) |
| 2026-09-21 | Start of day (adds lane fill fix and gpnae_poly) | 2,578 | H:latency_analysis §1 |
| 2026-09-21 | C-slow reduction | 2,074 | f28d103 |
| 2026-09-21 | Poly lane streamed | 1,584 | 45bfde9 |
| 2026-09-21 | All lanes dispatched per cycle | 1,374 | 6f4b991 |
| 2026-09-21 | PE forwarding | 1,267 | 229273d |
| 2026-09-21 | PE multiply/accumulate overlap | 1,232 | a37107f |
| 2026-09-21 | Poly lane FIFO streaming | 1,108 | a9bf77c |
| 2026-09-21 | Even split, 16 lanes | 805 | 1e67cb0 |
| 2026-09-21 | Maxpool streaming | 776 | fee5d9c |

B2. matmul_random_tanh, N=16 T=4, fp32, steady-state cycles per set (streamed). Day one ran one set at a time, so its
point is load + latency, a lower bound (sienna-report rule).

| Date | Step | Value | Source |
|---|---|---|---|
| 2026-09-19 | Day one (load + latency) | 19,009 | TODAY |
| 2026-09-23 | Credit interface, first perf trace (L=16) | 645 | 5903550, 558e57f |
| 2026-09-23 | Parallel lane fill | 459 | 558e57f |
| 2026-09-23 | Tree reduce + row broadcast | 459 | H:architecture_changes §1 |
| 2026-09-23 | Tail beside polynomial | 303 | 4197f5d |
| 2026-09-24 | One-row host writes | 303 | H:architecture_changes §1 |
| 2026-09-24 | 32 lanes | 271 | 4929054 |
| 2026-09-24 | SyncArray tiles | 271 | 5e39d51 |
| 2026-09-26 | Pipelined mesh, 15 credits (24 sets) | 263.0 | H:pipeline_performance_N16 SUMMARY |
| 2026-09-28 | Today, fp32 | 263 | TODAY |

bf16 is a different config: 190 cycles per set on 2026-09-28 (TODAY); plot as its own point, not in B2.
Not included: 2642 / 4 = 660.5 from 3a8c7ed is a 4-set mean including fill, not a steady state.

B3. SELU (matmul_ident_selu = conv_basic_selu, identical in every source), N=16 T=4, fp32, serial cycles. No valid
day-one point (ident_selu 9 wrong outputs, conv_basic_selu timeout; TODAY).

| Date | Step | Value | Source |
|---|---|---|---|
| 2026-09-21 | Start of day | 2,290 | H:latency_analysis §1 |
| 2026-09-21 | C-slow | 1,786 | f28d103 |
| 2026-09-21 | Poly lane streamed | 1,532 | 45bfde9 |
| 2026-09-21 | Dispatch | 1,322 | 6f4b991 |
| 2026-09-21 | Forwarding | 1,215 | 229273d |
| 2026-09-21 | MAC overlap | 1,180 | a37107f |
| 2026-09-21 | FIFO streaming | 1,056 | a9bf77c |
| 2026-09-21 | 16 lanes | 779 | 1e67cb0 |
| 2026-09-21 | Maxpool streaming | 750 | fee5d9c |

B4. SELU, N=16 T=4, fp32, steady-state cycles per set.

| Date | Step | Value | Source |
|---|---|---|---|
| 2026-09-23 | Credit interface, first perf trace | 565 | 558e57f |
| 2026-09-23 | Parallel lane fill | 336 | 558e57f |
| 2026-09-23 | Tree reduce + row broadcast | 336 | H:architecture_changes §1 |
| 2026-09-23 | Tail beside polynomial | 266 | 4197f5d |
| 2026-09-24 | One-row host writes | 264 | c1e6224 |
| 2026-09-24 | 32 lanes | 221 | 4929054 |
| 2026-09-24 | SyncArray tiles | 221 | 5e39d51 |
| 2026-09-26 | Pipelined mesh, 15 credits | 214.8 | H:pipeline_performance_N16 |
| 2026-09-28 | Today, fp32 | 215 | TODAY |

bf16 on 2026-09-28: 157 (TODAY), separate point.

B5. matmul_random_sigm, N=16 T=4, fp32. Serial cycles: day one 10,578 (+271 load, TODAY); 09-21 start 2,084; C-slow
1,580; poly streamed 1,468; dispatch 1,258; forwarding 1,151; MAC overlap 1,116; FIFO streaming 992; 16 lanes 747;
Maxpool streaming 718 (H:latency_analysis §1 and the same commits as B1). Steady cycles per set: day one 10,849
(load + latency, TODAY); after lane fill 472; tree reduce 472; tail overlap 325; one-row host 325; 32 lanes 263;
SyncArray 263 (H:architecture_changes §1); 2026-09-26 273.3 (H:pipeline_performance_N16); 2026-09-28 fp32 273,
bf16 198 (TODAY). The 263 -> 273.3 rise is unexplained (C13).

B6. Mesh alone, N=16 T=4, depth slices (COLLAPSE_K=0), set-0 latency, cycles: 849 (09-21 morning), 345 (3939d40),
238 (c4dc2f6), 203 (3f695b7), 204 (b5295bf, queued start), 143 (1521df9), 131 (5e27feb), 79 (36275ba), 70
(pipelined mesh, before bias; H:SIENNA_GPNAE_review §4), 75 (after bias; SM 8809c21, H:sienna_full_comparison §2).
Collapse-k is a separate series: 89 (105a62b, 09-24), 77 (H:sienna_full_comparison §2; baseline_fp32 09-28).

B7. ResNet-8 cycles per inference, fp32. Host-driven sets engine, N=16: 298,383 (98454b8) -> 65,619 (46e07de) ->
61,344 (5a46444) -> 56,382 (501ea33). sienna_layer engine, N=16: 56,902 (e19c249) -> 56,902 (78014e4, 15 credits)
-> 52,652 (e8d0da6) -> 52,004 (d026de4). N=32 host-driven: 67,103 -> 23,456 -> 23,141 -> 21,284; N=32 layer engine:
21,764 -> 21,764 -> 21,764 -> 21,572 (same commits; H:sienna_full_comparison §6). Do not join the two engines into
one line.

B8. Best GEMM MAC/cycle, N=16 (peak 256). Host-driven: 46.5 (bb998cb) -> 215 (46e07de) -> 254.4 (501ea33;
H:pipelined_mesh_report). sienna_layer: 239.5 (74ff3a7; H:layer_engine_performance §4) -> 256.0 (d026de4).

## C. Conflicts and unclear points

1. The pre-credit interface was not ready/valid. Day one had a start pulse (lost unless the FSM was in IDLE) and a
   held completion level, with no ready output (0eeac7e sienna_top.sv:716). Ready/valid appears only on sienna_layer's
   input streams (e19c249). A heading "ready/valid to credits" would misdescribe it; "start/complete level to credits"
   is accurate.
2. Day-one tanh has three values: 18,605 (e35dca8, d3c0bc2), 18,769 (sienna-rtl SKILL.md and VER), 18,738 + 271 load
   (TODAY).
3. Day-one sigmoid: 12,685 in H:SIENNA_GPNAE_review §5 (labelled "session notes, not a checked-in report") vs
   10,609 (VER) vs 10,578 + 271 (TODAY).
4. Day-one SELU: ident_selu 1,966 (e35dca8 "before") vs 2,190 PASS (VER) vs 2,453 with 9 wrong (TODAY); conv_basic_selu
   10,023 PASS (VER) vs timeout (TODAY). VER's 7/7 table was taken after some fixes (lambda*alpha, and a pass-through
   mode later found wrong, known-issues cross-reference), so it is not pure 0eeac7e.
5. Per-element GPNAE cost before the barrel MAC: 139 / 315 / 540 (e35dca8) vs 154 / 347 / 602 (VER).
6. The lambda*alpha constant fix is inside 975820a's diff, but its message mentions only the operand race.
7. Barrel MAC "~4.6x faster" (H:SIENNA_GPNAE_review §9) has no traceable source: full pipeline 18,605 -> 4,594 is 4.05x;
   per element 3.4x / 3.6x / 5.3x.
8. No pipelined-divider change was found in any repo. The divider (52 cycles, unpipelined) was removed from SIENNA's
   path by gpnae_poly (0559f3a); the published gpnae lane keeps it.
9. gpnae_poly was instantiated in sienna_top only in 6f4b991 (09-21 02:45) and added to the build list in 85b826f, yet
   f28d103 and 45bfde9 (09-21 00:05, 00:33) report pipeline numbers that depend on it: those came from an uncommitted
   working tree. The 4,594 -> 2,578 step (poly lane in the pipeline, plus possibly adfd2d8) has no measured split.
10. Whether 4,594 includes adfd2d8's fill fix is unclear; both were bumped together in d3c0bc2.
11. SELU single-set latency after 32 lanes: 347 (H:architecture_changes §1) vs 5e39d51's "421 -> 295", which implies
    421 before SyncArray. Also SELU after lane fill: 493 (558e57f) vs 494 (H:architecture_changes).
12. Pipeline mesh stage after SyncArray: 75 (5e39d51) vs 77 (H:architecture_changes, B2B).
13. Normal-range steady state after SyncArray: tanh 271 / sigmoid 263 (5e39d51, H:architecture_changes, B2B) vs
    "221/270/262" (bc8c199, 157e40f), whose stated order makes sigmoid 270 and tanh 262. Later (09-26) tanh 263.0 and
    sigmoid 273.3. The tanh/sigmoid labels look swapped in one of these.
14. H:layer_engine_performance contradicts itself: §5 gives tanh 266, sigmoid 268, ident_selu 205 (mean of last 8 gaps),
    §7c gives 263.0, 273.3, 214.8 for the same tests.
15. sienna_multi, 4 copies, 32 lanes: 62.7 / 55.3 (H:architecture_changes §2) vs 62.6 / 55.2 (38e2ec2).
16. The N=16 layer-engine overhead was first attributed to credits (5ca57e8; H:layer_engine_performance §6;
    H:SIENNA_GPNAE_review §3 and §6, "6% lower because of the credit limit"). 78014e4 showed 15 credits give no gain;
    the limit was the activation stage's partial-sum path (9877256, e8d0da6). The earlier statements are superseded.
17. The "3 credits capped the rate at ~50 cycles per set" figure (c53234e, B2B) is derived (latency / 3), not an
    isolated run. The only isolated credit measurement is the 3-credit tests: 39.3 vs 19.0 (N=16), 55.3 vs 35.0 (N=32).
    No run isolates 7 or 8 credits against 3.
18. Mesh latency formula constant: +17 (SM 8809c21, H:sienna_full_comparison, H:SIENNA_GPNAE_review) vs +16
    (perf_analysis 7ae1e7d, H:pipeline_performance_N16). Each states its start point (start vs launch), so they are
    probably consistent but differently defined.
19. H:pre_synthesis_v1 labels model times as microseconds but the values are nanoseconds (ResNet-8 N=16 "54,741.1 us";
    52,004 cycles at 950 MHz is 54.7 us), and its "faster" column reads "0x". H:sienna_full_comparison has 54.7 us and
    93x vs Syntiant.
20. Accumulate-test cycles per set differ across reports (164.0 in H:sienna_full_comparison vs 154.3 in
    H:pre_synthesis_v1, because d026de4 landed between), and f1cc285 says perf_analysis mis-reported accumulate configs
    until 2026-09-28. Treat every accumulate cycles-per-set figure before f1cc285 as unreliable.
21. bf16: GPNAE TB cycles per input barely change (18 -> 18, 16 -> 16, 20 -> 19; H:gpnae_gate) while pipeline cycles
    per set drop 26-28% (TODAY). No source explains the difference. My unverified guess: at 8 elements per lane the
    barrel-MAC round sits at its LOOP+1 floor, which shrinks with the 3-cycle bf16 multiplier, and the bf16 degrees
    are lower; the GPNAE TB runs larger groups. Needs a measurement before it goes in the report.
22. H:SIENNA_GPNAE_review §5 "best GEMM N=16 day one ~0.6 GFLOPS" and "mesh alone day one 849" come from session notes;
    849 was measured on 09-21 morning after the 09-19 fixes, and neither was re-measured from 0eeac7e.
23. The day-one commit 0eeac7e is dated 2026-07-27; "day one" (2026-09-19) is when work on it started. History file
    dates can lag or lead commit dates by a day (H:layer_engine_performance is filed 09-26, its header says 09-25).
24. Synthesis-readiness commits exist twice with identical messages: 13df7d9 / 7ff1880 / 66ed4e5 on main and
    63e9e30 / aa3a9f1 / 2cab850 on bf16. Mixed bf16 (09-26) and uniform bf16 (09-28) are on the bf16 branch only.
25. 1e67cb0 projected about 10% for 32 lanes; 4929054 measured 11-19% (SELU 264 -> 221, tanh 303 -> 271, sigmoid 325
    -> 263). Projection, not a contradiction, but do not quote the 10%.
