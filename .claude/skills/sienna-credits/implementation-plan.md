# SIENNA credit links: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every SIENNA module boundary uses one credit link (`credit_link_if`), with today's buffers kept, outputs identical, and latency / throughput / GFLOPS / TOPS measured before and after.

**Architecture:** shared interface, producer counter, register stage and checker in ArithmeticLibrary `Common/src` (new files only). Then, innermost first: GPNAE lanes (L4/L5), SystolicMesh (L1/L2/L3), SIENNA top (L0, L3 consumer, L6), SIENNA lanes / pooling / maxpool / dropout / output (L4-L9), layer engine and wrappers (L10). Each repo works on branch `credits`.

**Tech Stack:** SystemVerilog interfaces, Verilator 5.035 (`--binary --timing --assert`), Python 3 regressions, slurm farm.

**Spec:** `.claude/skills/sienna-credits/SKILL.md` (same directory; approved by Soham 2026-10-07). Read it first. Today's protocols: `sienna-back-to-back` skill, "Module boundaries as built".

## Global Constraints

- The shared files live in ArithmeticLibrary `Common/src` (Soham, 2026-10-07: "keep it in AriL, we can rename AriL repo later"); nothing existing in AriL changes. AriL's arithmetic units get no credits.
- Branch `credits` in ArithmeticLibrary, GPNAE, SystolicMesh, SIENNA. Innermost repo commits first; a parent bumps a submodule only in the task that adapts to it, so every commit builds.
- Credit semantics exactly as the spec: the consumer grants (after reset it advertises its slot count as ordinary credit returns), the producer spends one credit per put, credits travel consumer to producer only, `credit` is a count `CRW` bits wide.
- Ports are `credit_link_if` modports (`producer`, `consumer`, `monitor`). No new ready/valid pairs.
- `gpnae.sv` (the published Taylor lane) is not changed. GPNAE math (coefficients, term counts, ranges) is not changed.
- Outputs must stay identical to `main` (bit-exact where they are bit-exact today; words compared with `$J/cmds/int8_cmp_reg.py` for the pipeline). Cycles may change and are reported.
- Every build and simulation runs on the farm, on-demand only: `$J/cr_msgs/cr_launch.sh NAME MEM HOURS cmd...` (Task 0) on `od-64-gb-8-cores,od-128-gb-8-cores`, 8 CPUs; never spot, never `od-256-gb-32-cores`, `od-768-gb-24-cores`, `od-512-gb-16-cores`. fp32 N >= 64 needs >= 60 GB.
- Fast regressions only until Task 7: N = 16, T = 4 (and N = 8 for the mesh). N = 64 only after check-in.
- One-line comments; assertions under `ifndef SYNTHESIS`; every new assertion is shown to fire once on purpose (memory: edge-driven assertions are vacuous unless proven). Reports are .log. `grep`/`diff`/`du` are aliased: use `command grep` / `/usr/bin/grep`.
- Commit messages via `git commit -F <file>`; no Co-Authored-By or generated-by lines (Soham's rule). Pushes only when Soham asks.

## Review Focus

1. Reset in the middle of a stream (`TB_sienna_top`'s reset-mid-stream test): every producer counter returns to 0 and every consumer re-advertises; a lost advertisement would hang the pipeline forever. Pinned in Task 5a.
2. Several slots freed in one cycle (the mesh frees a result bank while the activation stage frees an activation bank; pooling lanes pop together): `credit` must be wide enough, never truncated. Pinned in Task 1 (CRW > 1 test) and Task 5b.
3. Output back-pressure mid-set (L9 credits withheld): nothing lost, duplicated or reordered; `pipeline_complete_o` and the entry credits still count right. Pinned in Task 5b.
4. Sets that produce no result (partial / accumulate sets, `g_null_done`, `p_null`): no L3/L6 credit spent or leaked for them. Pinned in Task 5a.
5. int8 bypass (ReLU / linear skip the lanes through `requant_lanes`) and packed sets: lane credits (L4/L5) must not be spent for lanes that receive nothing, and the packed wide-read order must survive the L3 push. Pinned in Task 5b.

---

### Task 0: Branches, launcher, baselines

**Files:** none in git. Create `$J/cr_msgs/cr_launch.sh` (outside git; `J=/proj/work/spramanik/sienna_jobs`).

- [ ] Create branch `credits` from `main` in ArithmeticLibrary (`SystolicMesh/ArithmeticLibrary` and `GPNAE/ArithmeticLibrary` are two checkouts of one repo: create it in one, fetch in the other), GPNAE, SystolicMesh, SIENNA. Check each is at its current `main`.
- [ ] `cr_launch.sh`: copy `$J/n64_msgs/n64_launch.sh`; scratch root `/proj/scratch/spramanik/sienna_cr`; snapshot the current SIENNA working tree into `$S/snaps/<NAME>.tar` on every launch (the same tar exclusions as `snap_launch_tree.sh`); defaults `CPUS=8`, `PARTITION=od-64-gb-8-cores,od-128-gb-8-cores`; log every launch to `$J/runs/pk_launch.log`.
- [ ] Baseline runs from the `main` tree, prefix `crb_`: GPNAE `python3 regression.py --lane poly --format {fp32,bf16,int8}` (run in `GPNAE/`); mesh `make sm-verilator FMT={fp32,bf16,int8} N=16` and `N=8`; `make pipeline FMT={int8,bf16,fp32} N=16 TILE=4`; `make perf-analysis FMT={fp32,bf16,int8} N=16`; `python3 model_runner.py --model-dir $J/models --models resnet8 --n 16 --tile-size 4 --format bf16` (ResNet-8 on the layer engine). Expected: all pass (they are today's gate).
- [ ] Ledger line per baseline run (name, command, verdict) in `.superpowers/sdd/<plan>/progress.md`.

### Task 1: Shared credit files in ArithmeticLibrary

**Files:** Create `Common/src/credit_link_if.sv`, `credit_counter.sv`, `credit_reg.sv`, `credit_link_checker.sv`, `Common/testbenches/TB_credit_link.sv`; Modify `Common/Makefile` (target `credit`). All in ArithmeticLibrary on branch `credits`.

**Interfaces (produces, used by every later task):**
- `interface credit_link_if #(int DATA_W = 32, int CRW = 1)`: `put`, `data[DATA_W-1:0]`, `credit[CRW-1:0]`; modports `producer (output put, data, input credit)`, `consumer (input put, data, output credit)`, `monitor (input put, data, credit)`.
- `module credit_counter #(int MAX = 32, int CRW = 1) (input clk_i, rstn_i, put_i, input [CRW-1:0] credit_i, output has_credit_o, output [$clog2(MAX+1)-1:0] count_o)`; `cnt <= cnt + credit_i - put_i`, reset 0; assertions `a_no_underflow` (put only with a credit), `a_no_overflow` (cnt + credit <= MAX).
- `module credit_reg #(int STAGES = 0, int DATA_W = 32, int CRW = 1) (input clk_i, rstn_i, credit_link_if.consumer up, credit_link_if.producer dn)`; STAGES register stages forward (put, data) and back (credit); 0 = wires. `/* verilator lint_off UNUSEDSIGNAL */` on clk/rstn for STAGES = 0.
- `module credit_link_checker #(int SLOTS = 32) (input clk_i, rstn_i, drained_i, credit_link_if.monitor lnk)`; tracks granted - used; `a_put_has_credit`, `a_within_slots` (outstanding <= SLOTS), `a_all_back` (drained_i -> outstanding == SLOTS); body under `ifndef SYNTHESIS` with `lint_off UNUSED*`.

Start from the passing prototype in `/proj/scratch/spramanik/credit_proto/` (credit_link_if.sv, credit_counter.sv, credit_reg.sv, credit_link_checker.sv, tb_proto.sv); add `count_o` to the counter.

- [ ] Write `TB_credit_link.sv` first: 128 links, random producer / consumer rates, consumer advertises SLOTS after reset with `CRW = $clog2(SLOTS+1)`, `credit_reg` STAGES from a `+stages` plusarg-selected generate (build once per STAGES via `-GSTAGES`), plus a test that frees 3 slots in one cycle (CRW > 1) and checks the count. Modes via `-GMODE`: 0 normal; 1 put without credit (expect `a_no_underflow`); 2 consumer over-grants (expect `a_within_slots` and `a_no_overflow`); 3 consumer keeps one credit at drain (expect `a_all_back`); 4 checker-only violation with the counter bypassed (expect `a_put_has_credit`). Prints `CREDIT_LINK STAGES=s MODE=m: ...` and `RESULT: PASSED/FAILED`; in modes 1-4 PASSED means the named assertion fired (run with `+verilator+error+limit+100` and count the lines).
- [ ] Makefile target `credit`: builds and runs STAGES 0, 1, 3 in mode 0 and modes 1-4 at STAGES 0; fails unless every run prints its expected result.
- [ ] Run on the farm (`cr1_credit`, `make -C SystolicMesh/ArithmeticLibrary/Common credit`). Expected: mode 0 PASSED at every STAGES with identical items delivered; each of modes 1-4 shows its assertion firing; lint `verilator --lint-only -Wall -DSYNTHESIS` on the four files: 0 warnings, 0 errors.
- [ ] Commit (AriL `credits`): "Common: credit_link_if, credit_counter, credit_reg and credit_link_checker for SIENNA's module boundaries, with TB_credit_link (make credit): 128 links at 0/1/3 register stages deliver every item in order, and each assertion is shown to fire on purpose."

### Task 2: GPNAE lanes on credit links (L4, L5)

**Files:** Modify GPNAE `src/gpnae_poly.sv`, `src/gpnae_poly_int8.sv`, `testbenches/TB_gpnae_poly.sv`, `Makefile` (add the four AriL credit files to the file list), `regression.py` only if it reads changed signals. Bump GPNAE's `ArithmeticLibrary` to Task 1's commit. `src/gpnae.sv` untouched.

**Interfaces:**
- Consumes: Task 1's interface, counter, checker.
- Produces (sienna_top uses these in Task 5b): `gpnae_poly` ports `credit_link_if.consumer in` (`DATA_W = DATA_WIDTH + 1`: `{last, signal}`; the lane advertises its FIFO depth `2**ADDR_LINES` = 32) and `credit_link_if.producer out` (`DATA_W = DATA_WIDTH`, one result per put; output slots are granted by the downstream collector). Removed: `signal_i`, `wr_en_i`, `last_i`, `full_o`, `empty_o`, `idle_o`, `final_result_o`, `done_o`. Kept: `clk_i`, `rstn_i`, `terms_i`, `control_word_i`, the int8 `gp_*` inputs. Same for `gpnae_poly_int8`.
- Rule: a lane starts a barrel group of K elements only while its `out` counter holds >= K credits (`count_o >= K`), so the barrel MAC never stalls mid-group; the `last` bit replaces `last_i` (the group size is still taken from the FIFO count at `last`). The input FIFO returns one `in` credit per pop.

- [ ] Change `TB_gpnae_poly.sv` first: the stimulus driver becomes a producer (`credit_counter`, MAX 32), the result capture a consumer advertising `+out_slots` (default 64) that returns one credit per captured result, optionally withholding credits at random (`+stall_pct`, default 0); bind `credit_link_checker` on both links. Build against the old lane: expect it not to compile (ports missing).
- [ ] Implement the ports and the K-credit group rule in `gpnae_poly` and `gpnae_poly_int8`.
- [ ] Farm (`cr2_gp_{fp32,bf16,int8}`): `python3 regression.py --lane poly --format F` (and `--model hw` for bf16). Expected: every result identical to `crb_` (bit-exact checks pass, accuracy figures unchanged); cycles per input reported. Then the same with `+stall_pct=30` (a TB plusarg passed through `SIM_ARGS`): every result still identical, no checker firing.
- [ ] Commit (GPNAE `credits`, after the AriL bump commit): "gpnae_poly and gpnae_poly_int8 take inputs and give results on credit links (L4/L5): the lane advertises its 32-entry FIFO and starts a barrel group only with K output credits; TB_gpnae_poly drives both links with the checker bound; every result identical to main, also with 30% random output stalls."

### Task 3: SystolicMesh on credit links (L1, L2, L3)

**Files:** Modify SystolicMesh `src/top/SystolicMesh.sv`, `src/mem/MeshOutputSram.sv` (only if the push needs it), `testbenches/TB_SystolicMesh.sv`, `Makefile` (credit files), `regression.py` only if it reads changed signals. Bump SystolicMesh's `ArithmeticLibrary` to Task 1's commit.

**Interfaces:**
- Produces (sienna_top uses these in Task 5a): `credit_link_if.consumer staging` (`put` = a set's start; `data` = the set sideband today sampled at accept: `partial`, `bias_valid`, `pack_shift`, `weight_cached`, `weight_tile`; the mesh advertises 2 and returns one credit per `bcast_release`); `credit_link_if.consumer wc_region[2]` (credit = region free to fill; `put` = a fill of that region starts; one slot each); `credit_link_if.producer result` (`DATA_W = WIDE_READ*ACC_W + 2 + PACKFLAG`: one wide beat of the oldest result plus `first`/`last` bits; `N*N/WIDE_READ` beats per set, element order exactly as today's wide read for index 0.. and the packed order when packed). Removed: `start_matrix_mult_i`, `input_ready_o`, `wc_region_busy_o`, `collection_complete_o`, `result_release_i`, `wide_read_*`. Write buses (`north/west_write_*`, `wc_write_*`, `bias_i`) stay: rows need no per-row credit (the staging credit reserves the bank).
- The mesh pushes a result only while it holds `result` credits; it frees the result bank after the set's last beat is put. The legacy single-word read port (`read_enable_i` / `read_addr_i`) stays for `TB_SystolicMesh`'s checks if still used; otherwise removed.

- [ ] Change `TB_SystolicMesh.sv` first: the host side becomes an L1/L2 producer (counters, checker bound), the consumer an L3 consumer advertising `+res_slots` beats (default `N*N/WIDE_READ`, i.e. one set) and collecting the pushed result; the start-while-not-ready and staging-overrun tests become "a put without a credit is impossible: the producer waits" tests; the release-with-no-result test is removed (no release port). Keep every data check and the `[Perf]` / `[Serial]` / `[Stream]` prints.
- [ ] Implement L1/L2/L3 in `SystolicMesh.sv`; keep every existing mesh assertion that still applies, rewrite the ones on removed ports as link assertions.
- [ ] Farm (`cr3_sm{8,16}_{fp32,bf16,int8}`): `make regression MATRIX_SIZE=N REGRESSION_OPTS="--format F"` (every tile). Expected: every test bit-exact against `mesh_model`; latency and stream cycles reported against `crb_` (`$J/cmds/pk_cmp_mesh.py`).
- [ ] Commit (SystolicMesh `credits`): "SystolicMesh takes sets and weight-cache fills on credit links and pushes results on one (L1/L2/L3): staging and result banks unchanged, the result is streamed as wide beats while the consumer holds credits; TB_SystolicMesh drives the links with the checker bound; bit-exact at N = 8 and 16 in fp32, bf16, int8."

### Task 4: (folded into 5b) `Maxpool_2D`, `dropout`, `fwft` change ports in the same commit as `sienna_top`, so SIENNA builds at every commit.

### Task 5a: SIENNA top, mesh side (L0, L1, L3 consumer, L6)

**Files:** Modify `src/sienna_top.sv`, `testbenches/TB_sienna_top.sv`, `Makefile` (credit files in `DESIGN_FILES` from `$(SM_LIB_DIR)/Common/src`), `regression.py` (only signal names it reads). Bump SIENNA's `SystolicMesh` to Task 3 (GPNAE stays at `main` here).

**Interfaces:**
- `sienna_top` port `credit_link_if.consumer host` (L0): `put` = start with today's per-set sideband in `data` (activation, terms, train, seed, pack, int8 parameters, accumulate flag, cached tile); credits = staging banks from the mesh's L1, withheld while `SETS_IN_FLIGHT` sets are in flight (the entry counter stays, internal). Removed: `pipeline_ready_o`, `start_pipeline_i`. Kept: write buses, `pipeline_complete_o`, `done_set_id_o` (status).
- Internal: `sienna_top` is the L1 producer and L3 consumer of the mesh; it advertises `PER_LANE` beats when it holds a free activation bank and its lanes are idle (one set at a time, as today); L6 (activation -> pooling) becomes a set link inside `sienna_top` (counter + checker), credit on `p_release`, put on `bank_done`; `g_null_done` / `p_null` spend and return nothing.
- Parameter `LINK_STAGES` (default 0) puts `credit_reg` on L0, L1, L3.

- [ ] Change `TB_sienna_top.sv` first: the host becomes an L0 producer (counter + checker); every internal `dut.credits` / `dut.mesh_input_ready` / `dut.systolic_*` reference moves to the new signal names; the credit-overrun pass becomes "the host cannot put without a credit"; reset-mid-stream unchanged (Review Focus 1); add a test of 4 partial sets followed by their accumulate set with the checkers bound (Review Focus 4).
- [ ] Implement; keep every top assertion, rewriting those on removed ports as link assertions.
- [ ] Farm (`cr5a_reg16_{int8,bf16}_T4`, then fp32): `make pipeline FMT=F N=16 TILE=4`. Expected: 41/41 int8, 32/32 bf16 and fp32; words identical to `crb_` (`int8_cmp_reg.py`); cycles per test reported. Same with `LINK_STAGES=1` (a `make` variable passed to Verilator as `-GLINK_STAGES`): all pass, words identical.
- [ ] Commit (SIENNA `credits`): "sienna_top: the host, the mesh and the activation-to-pooling hand-off on credit links (L0, L1, L3, L6), entry admission kept inside L0; TB_sienna_top drives L0; outputs identical to main in int8, bf16, fp32 at N = 16, also with a register stage on every link."

### Task 5b: SIENNA lanes, pooling, maxpool, dropout, output (L4, L5, L7, L8, L9)

**Files:** Modify `src/sienna_top.sv`, `src/fwft.sv`, `Maxpool/Maxpool_2D.sv`, `Dropout/dropout.sv`, `src/requant_lanes.sv` (only if the int8 bypass needs a link), `testbenches/TB_sienna_top.sv`, `TB_maxpool_fmt.sv`, `TB_maxpool_int8.sv`, `TB_dropout_fmt.sv`, `TB_dropout_int8.sv`. Bump SIENNA's `GPNAE` to Task 2.

**Interfaces:**
- Lanes: `sienna_top` is the L4 producer (one counter per lane) and the L5 consumer (a collector per lane advertising 16 and returning a credit as each result is written to the activation bank, which it does at once).
- `fwft`: `credit_link_if.consumer in` (advertises DEPTH, credit per pop; overwrite-when-full removed) and `credit_link_if.producer out` toward maxpool. The dispatcher's `disp_can_write` threshold is replaced by per-lane L7 counters.
- `Maxpool_2D`: `credit_link_if.consumer in`, `credit_link_if.producer out`; `start`/`done` removed; a window enters only when the lane holds an L9 credit for its result.
- `dropout`: `credit_link_if.consumer in`, `credit_link_if.producer out`; training pipeline depth covered by the reserved credit.
- `sienna_top` output: `credit_link_if.producer out[NUM_LANES]` (L9, one result word per put); replaces `final_result_o` / `result_valid_o`. `pipeline_complete_o` still pulses when a set's last word is put.

- [ ] Change the four unit TBs first (drive and collect through links, checker bound, random stalls); then `TB_sienna_top`: the output capture becomes L9 consumers advertising `+out_slots` (default 64) with `+out_stall_pct` (default 0); add the back-pressure test (Review Focus 3): withhold all L9 credits for 500 cycles in the middle of a stream, then release; every output and completion identical to the no-stall run.
- [ ] Implement; int8 bypass and packed sets keep their paths (Review Focus 5: lanes that receive nothing get no L4 put).
- [ ] Farm (`cr5b_*`): unit TBs; `make pipeline` N = 16, T = 4 in int8 / bf16 / fp32 with `+out_stall_pct=0` and `30`; `make pack FMT=int8`; `make tflite FMT=int8`. Expected: all pass, words identical to `crb_`, TFLite 0 differ; cycles reported.
- [ ] Commit (SIENNA `credits`): "Lanes, pooling FIFOs, maxpool, dropout and the output on credit links (L4, L5, L7, L8, L9): nothing can overrun or be dropped, and a downstream consumer can stall SIENNA; outputs identical to main, also with 30% random output stalls and a 500-cycle full stall."

### Task 6: Layer engine and wrappers (L10)

**Files:** Modify `src/sienna_layer.sv`, `src/sienna_multi.sv`, `testbenches/TB_model_run.sv`, `TB_sienna_model.sv`, `TB_sienna_multi.sv`, `model_runner.py` / `regression.py` only where they read changed signals.

**Interfaces:** `sienna_layer` ports `credit_link_if.consumer a_rows`, `credit_link_if.consumer w_rows` (one N-word row per put; N rows granted per L0 credit it holds, i.e. one staging bank's worth) replacing `a_valid_i/a_ready_o/w_valid_i/w_ready_o`; its outputs become L9 producers like `sienna_top`'s. `sienna_multi` exposes L0 per copy (the round-robin `sel` stays) and L9 per copy.

- [ ] Change the three TBs first (drive rows and collect results through links, checker bound).
- [ ] Implement.
- [ ] Farm (`cr6_*`): `make gemm FMT=int8 QUICK=1`, `make tflite FMT=int8`, `python3 model_runner.py --model-dir $J/models --models resnet8 --n 16 --tile-size 4 --format bf16` (ResNet-8 on the layer engine), `model_runner.py --engine sets --format fp32` (ResNet-8), TB_sienna_multi in three formats. Expected: GEMM 0 differ, TFLite 0 differ, model outputs identical to `crb_`/`tlf_` (same class, same max |hw-ref|), multi passes.
- [ ] Commit: "sienna_layer takes A and W rows on credit links (L10) and gives results on L9; sienna_multi exposes L0 and L9 per copy; TB_model_run, TB_sienna_model and TB_sienna_multi drive the links; GEMM, TFLite and model outputs identical to main."

### Task 7: Full check, before/after comparison, docs

**Files:** `sienna_report/credits_compare.log` (outside git), `.claude/skills/sienna-credits/SKILL.md` (As built), `sienna-back-to-back` and `sienna-rtl` (boundaries), README architecture bullet and diagram.

- [ ] Farm: `make regression FMT={fp32,bf16,int8}` (the gate) with `LINK_STAGES=0`; `make perf-analysis FMT={fp32,bf16,int8} N=16` with `LINK_STAGES=0` and `1` (`cr7_perf_*`); `python3 model_runner.py --model-dir $J/models --models resnet8 --n 16 --tile-size 4 --format bf16` (ResNet-8 on the layer engine).
- [ ] `credits_compare.log`: per configuration (the 7 perf configs × 3 formats) latency, cycles per set and GFLOPS (TOPS in int8) for `crb_` (main), `LINK_STAGES=0` and `LINK_STAGES=1`, with the difference; ResNet-8 cycles and µs; mesh latency per tile; GPNAE cycles per input. Every number from a run log.
- [ ] Docs: the spec's As built; diagram arrows become credit links; README pipeline bullet.
- [ ] Commit docs; report to Soham: what got better, what got worse, by how much, and why. Merge only on his go-ahead.
