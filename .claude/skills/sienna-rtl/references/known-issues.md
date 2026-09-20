# SIENNA known issues

Defects found by reading the full source tree and cross-checking against simulation and synthesis logs. Each entry states how strong the evidence is, because that changes how much you should trust it:

- **Confirmed (reproduced)** — a log in the repo shows the failure.
- **Confirmed (inspection)** — the code plainly says so; no run has been done to exercise it.
- **Latent** — real defect, currently masked by configuration or tool behaviour.

Entries 3, 4, 5, 5b, 5c and 5d have been **fixed** and are kept here because the
reasoning is worth having. Entries 1 and 2 are **corrections** — claims that
turned out to be wrong, kept so the same wrong conclusion is not reached twice.
Entry 9 is partly fixed. Everything from 6 onward is otherwise still open.

Read the relevant entry before spending time on a diagnosis.

Two things this file got wrong the first time, both worth internalising:
counting a file with `wc -l` when it has no trailing newline (entry 2), and
assuming that because the tooling offers a mode, the RTL must implement it
(entry 1). A missing mode is a spec gap for the designer to decide on, not a
bug to fix by inventing hardware.

---

## 1. Activation code 0 is not a mode — MISDIAGNOSED, tests removed

**This was originally written up as an RTL bug. It was not.** GPNAE implements
three activations and only three. The committed `gpnae_control_unit` is explicit:

```systemverilog
OP: case (control_word_i)
      2'b01: next_state = SELU_CHECK;
      2'b10: next_state = SIGMOID;
      2'b11: next_state = TANH;
      default: next_state = IDLE;    // 2'b00 was never a mode
    endcase
```

The "idle" concept existed only in the Python tooling: `activation_to_code()`
mapped `"idle" -> 0`, `apply_activation()` fell through to `return x.copy()`,
and two entries in `PIPELINE_TESTS` set `"act": "idle"`. Those two tests were
exercising a pass-through the hardware does not have, so they could only ever
stall in `GPNAE_ROUND` until the 200,000-cycle testbench timeout:

```
testbenches/results/pipeline/matmul_ones_idle.log
  Activation code : 0  Num terms : 1
  [FATAL] Timeout after 200000 cycles - pipeline_complete_o never asserted.
  [STAGE] Verifying - 81 expected, 0 captured
```

**Resolution:** `matmul_ones_idle` and `matmul_small_exact` were removed from
`PIPELINE_TESTS`, and `activation_to_code()` now raises on an unsupported
activation instead of silently returning 0. A mode the RTL does not implement
is no longer reachable from a test definition. The matrix types those tests
covered are still exercised against the mesh by the SystolicMesh IP suite
(`mm_ones`, `mm_small_values`).

**The lesson worth keeping:** a stall in `GPNAE_ROUND` with 0 words captured
means the lane never asserted `done_o`. Before assuming the RTL is wrong, check
that the configured `ACTIVATION_CODE` is one the hardware actually decodes.
Adding a bypass would be a design change, not a bug fix.

---

## 2. tanh coefficient range — NOT a bug (retracted)

An earlier revision of this file claimed tanh read one coefficient past the end
of `taylor_coeffs.mem`. **That was wrong, and the arithmetic is worth recording
so nobody re-derives it.**

The file has **31** entries, not 30. It has no trailing newline, so `wc -l`
reports 30 — count with `grep -c .` instead. The entries are the Maclaurin
coefficients of `e^x`, i.e. `1/k!` for k = 0…30, each within ~1 ULP of the
correctly rounded FP32 value.

The address walk asks for exactly 31 coefficients:

- `gpnae.sv` passes `terms_i = 30` to `mac`
- `mac.sv` instantiates the controller with `.terms_i(terms_i + 1'd1)` → 31
- `controller.sv`: `coeff_addr_o = terms_i - 1 - term_count` → starts at 30
- `CHECK_TERMS` loops while `term_count < terms_i - 1` → `term_count` runs 0…30

Addresses 30 down to 0: 31 reads against 31 entries. Nothing is out of range.
Sigmoid (16 reads, addresses 15…0) and SELU (15 reads, addresses 14…0) sit
comfortably inside the table.

The `+1` in `mac.sv` is load-bearing: `NUM_TERMS` names the polynomial degree,
and degree *n* needs *n+1* coefficients. Do not "simplify" it away.

---

## 3. SELU dropped the first element of every batch — FIXED

**Root cause confirmed by isolation; fix applied in `InputFIFO.v`.**

Symptom was one wrong value in `matmul_ident_selu`:

```
[FAIL] [0] exp=0x3f867d5f act=0x00000000 | rel=100.000%
Total: 81   Exact: 80   Failed: 1
```

`0x3f867d5f` is λ = `selu(1.0)`. Everything else was bit-exact, which made it
look like an indexing problem. It was not.

**`InputFIFO` is not first-word-fall-through.** `RAM.v`'s output register
`doutb_reg` only reloads when `regceb` (= `rd_en_i`) is asserted, so `data_o`
presents the value fetched by the *previous* pop. Before the first pop it holds
its reset value. A standalone probe of the FIFO confirms it:

```
after 4 writes, before any read: data_o=00000000   <-- not the head
pop 0: before rd_en = 00000000   after rd_en = aaaa0000
pop 1: before rd_en = aaaa0000   after rd_en = aaaa0001
```

The GPNAE control unit samples `fifo_data_o` in `OP` (`capture_data_o`), and
for the SELU-positive path `selu_input = captured_signal`. On the first element
of a batch no pop has happened yet, so it captured 0 and computed `selu(0) = 0`.

Alignment survived because the first element is popped *twice* — once by the
free-running TYTAN controller and once by `fsm_rd_en` — so from element 1
onward the capture lines up again. Two defects that cancelled, leaving exactly
one wrong value per lane. Sigmoid and tanh were unaffected because they take
`mac_input` from `fifo_data_o` only after the MAC's own pop has landed.

Isolated with `GPNAE/testbenches/TB_gpnae_activations.sv`, which showed
index 0 = `00000000` against an expected λ while indices 1–7 were all correct
and correctly positioned.

**Fix:** hold the RAM output register enabled until the first pop, so `data_o`
tracks the head from the moment the FIFO becomes non-empty:

```verilog
reg read_started;
always @(posedge clk_i or negedge rstn_i)
    if (~rstn_i)      read_started <= 1'b0;
    else if (rd_en_i) read_started <= 1'b1;

wire out_reg_en = rd_en_i | ~read_started;   // -> .regceb(out_reg_en)
```

After the first pop the behaviour is byte-for-byte what it was, so the pop
accounting the rest of the design depends on is untouched.

The underlying design smell remains: **the GPNAE read pointer has two owners**
(the TYTAN controller and the SELU FSM), and `fifo_rd_en_o` arbitrates between
them by decoding `current_state`. The double-pop on element 0 is a symptom of
that. A single read-pointer owner would be the real cleanup.

---

## 4. `north_queue_empty_o` compared against the wrong value — FIXED

**Was latent.** `SystolicMesh/src/top/SystolicMesh.sv`:

```systemverilog
assign west_queue_empty_o  = (ptr_A == 0);
assign north_queue_empty_o = (ptr_B == 1);   // <-- should be 0
```

The two lines are otherwise symmetric. As written, the north queue reports "empty" after exactly one word has been written, and reports "not empty" when nothing has been written at all.

`sienna_top` gates its `IDLE → SYSTOLIC_START_PULSE` transition on `!north_queue_empty && !west_queue_empty`, so an unloaded north queue would not block the start. It was masked because the testbench always loads all 256 words before starting, making `ptr_B = 256`.

**Fix:** changed to `(ptr_B == 0)`, matching the west line. No behavioural
change for any current test; it closes the degenerate cases.

---

## 5. `sigtan` used a tri-state mux — FIXED

**Was confirmed by inspection.** `GPNAE/src/sigtan.sv`:

```systemverilog
assign mux_output = (select_sub == 2'b00) ? mac_result : 'bz;
assign mux_output = (select_sub == 2'b01) ? sub_result : 'bz;
```

Two continuous assignments driving one net, relying on high-impedance resolution. When `select_sub` is neither `2'b00` nor `2'b01` — and the control unit's *default* value is `2'b11` — both drivers output `z`, so `mux_output` floats into `fp32Divider`.

It worked in simulation because `select_sub` is only `2'b11` outside the `SIGMOID`/`TANH` states, when the divider result is not sampled. It is not synthesizable as internal logic on an ASIC and is poor practice on FPGA.

**Fix:** replaced with a single `always_comb` `unique case` whose default
returns `mac_result`, so the divider never sees a floating input. The
`mux_output_reg` declaration, which was never used, was removed with it.

---

## 5b. SELU's negative-branch constant was α, not λ·α — FIXED

**Confirmed numerically.** Found only after the GPNAE testbench was given real
golden values; no existing test could have caught it.

SELU is `λ·x` for `x ≥ 0` and `λ·α·(e^x − 1)` for `x < 0`. `SeLu.sv` performs
exactly **one** multiply, selecting the constant by sign, so the negative-branch
constant has to carry both factors. It held α alone:

| | value | hex |
|---|---|---|
| `LAMDA_ALPHA`, as written | 1.6732632 | `3FD62D7D` — this is α |
| required λ·α | 1.7580993 | `3FE10966` |

Every negative SELU result was low by a factor of λ, about 4.8%. Measured
against the golden model before the fix:

```
selu(-1.0)  expected -1.11133  got -1.05770
selu(-0.5)  expected -0.69176  got -0.65838
selu(-2.0)  expected -1.52017  got -1.44681
```

The ratio is 0.951746 in all three cases, which is exactly α/(λ·α).

Why nothing caught it: the only SELU test in the pipeline suite is
`matmul_ident_selu`, and an identity matrix contains only 1.0 and 0.0. Every
input is non-negative, so the negative branch never ran. `conv_basic_selu`
would have exercised it, but the suite aborts on first failure and never
reached it.

**Fix:** `LAMDA_ALPHA = 32'h3FE10966`.

---

## 5c. `tanh(0)` returned −0.5 — FIXED (exact cancellation)

**Root cause: exact cancellation in `fp32_up_down` returned ±1.0 instead of
zero.** Fixed there and in `fp32_down.sv`; `tanh(0)` now returns 0. What
remains near zero is not this bug -- see the note at the end of this entry.

`tanh(0)` should be `(e^0 − 1)/(e^0 + 1) = 0/2 = 0`. The hardware returns
`0xBF000000` = −0.5. Every other tanh input in the test vector is correct to
within a few ULP.

The divider is **not** at fault — driven directly it behaves correctly:

```
0 / 2  -> 00000000   (done after 4 cycles)
1 / 2  -> 3f000000   (done after 51 cycles)
2 / 2  -> 3f800000   (done after 51 cycles)
0 / 1  -> 00000000   (done after 4 cycles)
```

Note the latency asymmetry: `fp32Divider` short-circuits a zero numerator via
its divide-by-zero path in 4 cycles, against 51 for a normal divide. That
variable latency, arriving at a `sigtan`/control-unit handshake built around
the long case, is the most likely mechanism — −0.5 is what you get from
`−1 / 2`, i.e. a numerator and denominator taken from different pipeline
stages. Not yet proven; a probe of `sigtan`'s internals segfaulted Verilator
and the trail was not pursued further.

Low practical impact: random stimulus never produces an exact zero into tanh,
so the pipeline suite cannot see it. It would matter for any real activation
tensor containing exact zeros, which is common after ReLU or padding.

---

## 5d. SELU: MAC operand mixed across the Horner recursion — FIXED

**The major defect in this IP.** Not a precision issue: the MAC consumed
operands belonging to several different elements within one computation.

Two independent causes, both required for the fix:

1. `controller.sv` started a computation whenever its credit counter said data
   had been written, so the MAC free-ran, out of step with the GPNAE FSM.
2. Both the MAC (`LOAD_SIGNAL`) and the FSM (`SELU_POS`) popped the input FIFO,
   so a positive SELU element was popped twice and the two drifted apart.

Instrumented trace, `act_edge` 1x30: **75 MAC computations for 90 elements**,
and nearly every one walked through a changing operand. The computation that
produced the reported failure used elements 4, 5, 6 and 7 on its first four
Horner rounds and element 8 for the remaining eleven.

At x = -3.5 the block computed e^x = 0.017022 against 0.030288. Replaying the
traced operand sequence in software reproduces 0.01702130 against the RTL's
0.01702189 — 2 ULP, the difference being round-to-nearest model versus
truncating hardware. That is the proof; nothing else matched.

**Why it looked small.** Only 1 element in 6480 exceeded the 1% bar. The
corrupted rounds carry the tiny leading coefficients (1/14!, 1/13!, ...), so
the error stayed inside tolerance for most inputs. The whole negative-SELU
range was wrong, not just the element that tripped the check.

**Fix.**

- FSM owns the read pointer: one pop per element, on its own done pulse, in
  every terminal state.
- MAC starts on the FSM's explicit request; credit counter removed.
- MAC reads `captured_signal`, not the live `fifo_data_o`.
- `InputFIFO` holds `regceb` high. Gating it on `rd_en_i` meant the output
  register only reloaded on a pop, so the first pop re-presented the word
  already showing and every element after it arrived one pop late.
- `done_delay` widened 3 -> 4 stages; three met the new read latency with no
  margin at all.

**Verified.** 90 pops for 90 elements, 74 computations for the 74 elements
that need one, zero computations with a changing operand. Regression went from
10 element failures to 0 across 6480 checks.

**A claim this entry previously made that was wrong:** "the MAC does NOT
free-run — computation count is exactly sigmoid(30) + tanh(30) +
SELU-negatives(15) = 75". The 75 was real but the inference was backwards: 75
computations for 90 elements is the *evidence of* the bug, not evidence
against it. The count coincided with a plausible figure and was taken as
confirmation instead of being checked. Count against the number of elements,
not against what you expect the number to be.

**Fix attempts that did NOT work** (all measured, all reverted), kept because
each one is a plausible-looking dead end:

| attempt | result |
|---|---|
| reset `datapath.v` pipeline registers on `dp_reset_o` | no change |
| source `mac_input` from `captured_signal` alone | SELU 14 -> 5 but sigmoid 0 -> 8 |
| operand hold in `mac.sv`, frozen at first MULTIPLY | SELU 14 -> 55 |

The second is instructive: it is part of the real fix, but on its own it makes
things worse, because the snapshot is only meaningful once the FSM is the sole
popper. Fixing one of the two causes without the other does not converge.

**Why the original testbench never saw it.** `signal_sigmoid` is strictly
increasing from -5 to +5: one sign change, zero positive-followed-by-negative
transitions. It also had no golden comparison.

---

## 6. Combinational clock gating

**Confirmed (inspection).** `gpnae_control_unit` gates three clocks with plain AND gates:

```systemverilog
assign selu_clk_o   = clk_i & selu_enable;
assign mac_clk_o    = clk_i & mac_enable;
assign sigtan_clk_o = clk_i & sigtan_enable;
```

`SeLu.sv` does the same to its `fp32_down` instance: `.clk_i(clk_i & !is_positive_selu)`.

The enables are registered, which avoids the worst glitching, but this still produces derived clocks with no clock-tree treatment, creates cross-domain paths into blocks whose resets are on the ungated clock, and will not pass CDC or timing signoff. An ASIC flow needs proper integrated clock-gating cells; an FPGA flow generally wants clock enables on the flops instead. It also complicates any waveform debugging inside GPNAE, since those blocks simply stop advancing.

This may well interact with issue 3.

---

## 7. Every testbench's tolerance check measures the wrong thing

**Confirmed (inspection).** `TB_SystolicMesh.sv` and `TB_SystolicArray.sv` both contain:

```systemverilog
expected_real = $signed(expected);
actual_real   = $signed(actual);
```

`expected` and `actual` are IEEE-754 bit patterns. Reinterpreting them as signed integers makes the resulting `abs_diff` and `rel_diff` meaningless as floating-point error measures.

**`TB_sienna_top.sv` has the same defect by a different route.** It uses
`$bitstoshortreal`, which looks correct — but Verilator maps it to the 64-bit
`$bitstoreal`:

```
%Warning-WIDTHEXPAND: Operator BITSTOREALD expects 64 bits on the LHS,
                      but LHS's VARREF 'b' generates 32 bits.
```

The 32-bit pattern is zero-extended and reinterpreted as a *double*, so every
binary32 value decodes to a denormal near 1e-315. That is why every log line
reads `exp=0.000000 act=0.000000` — the values really are ~0, not just badly
formatted. This was previously written off as a display quirk; it is not.

The pass/fail decision still discriminates, because the ratio of two such
denormals equals the ratio of their integer bit patterns. So the check is an
*integer-ULP-relative* test, not the 1%-of-value test it claims to be. For the
few-ULP truncation error the FP units produce it passes, and for a grossly
wrong value it fails — which is why the suite has been useful despite this. But
the reported "Tol%" figures do not mean what they say, and the check will not
behave sensibly across a sign change or an exponent boundary.

A portable replacement that works under Verilator is a manual decode; see
`f32()` in `GPNAE/testbenches/TB_gpnae_activations.sv`:

```systemverilog
function automatic real f32(input logic [31:0] b);
  real m, v; int e;
  if (b[30:23] == 8'h00) return 0.0;
  m = 1.0 + (real'(b[22:0]) / 8388608.0);
  e = int'({24'b0, b[30:23]}) - 127;
  v = m * (2.0 ** e);
  return b[31] ? -v : v;
endfunction
```

This matters beyond cosmetics: a comparator built on the broken conversion
reports success for *everything* when the expected value is zero, because both
sides decode to ~0 and the absolute-difference branch is taken. That is how the
`tanh(0)` defect (5c) stayed invisible until the decode was fixed.

The reported pass rates are still *directionally* usable — for two nearby floats of the same sign, the integer distance is monotonic in the float distance, which is why the ULP-scale differences produced by the truncating FP units happen to land inside the 1% integer-relative band. But the numbers printed in `readiness_report.md` under "Tol%" do not mean what they say, and the check will not behave correctly across a sign change, across an exponent boundary, or for large errors.

Worth fixing before anyone treats the readiness report as signoff evidence.

---

## 8. GPNAE had no numerical verification — PARTLY ADDRESSED

**Confirmed (inspection).** The testbench declares two 30-element input arrays, `signal_sigmoid` and `signal_tanh`, and runs three tests:

```systemverilog
perform_test(2'b01, "SELU",    14);
perform_test(2'b10, "Sigmoid", 15);
perform_test(2'b11, "Tanh",    30);
```

But `perform_test` unconditionally drives `write_signal(signal_sigmoid[i], ...)`. `signal_tanh` is never read. All three tests run the sigmoid input range (roughly ±5); the tanh array covers roughly ±10.

Combined with the fact that this testbench checks no expected values at all (see `verification.md`), GPNAE had effectively no standalone numerical verification.

**Added:** `GPNAE/testbenches/TB_gpnae_activations.sv` runs all three
activations over a vector that mixes positive, negative and zero inputs and
compares every result against a golden value at 1% relative tolerance. It is
what exposed defects 5b and 5c, both of which had been invisible for the life
of the project. `TB_gpnae.sv` itself is unchanged and still has the dead
`signal_tanh` array.

---

## 9. Duplicate module definitions across the build

**Latent**, and flagged by Vivado.

Both `ArithmeticLibrary` copies are listed in the root `Makefile` (`SM_LIB_FILES` and `GPNAE_LIB_FILES`), so `fp32Adder`, `fp32Multiplier`, `R4Booth` and `karatsubaUnsigned` are each compiled twice.

**Partly fixed.** The `cntlz8` copy in `GPNAE/src/TYTAN/LZC.v` has been removed, so a GPNAE-only build now has exactly one definition. Two remain at SIENNA top level, because the two `ArithmeticLibrary` checkouts each carry one — that needs the submodules consolidating, not an RTL edit.

Verilator is silenced with `--Wno-MODDUP`. Vivado is not:

```
CRITICAL WARNING: [Synth 8-9873] overwriting previous definition of module 'cntlz8'
  [GPNAE/src/TYTAN/LZC.v:20]
```

The two `ArithmeticLibrary` copies currently differ only in port declaration style (`logic` versus `wire`/`reg`) and are functionally identical, so nothing is wrong today. The hazard is that an edit to one copy appears to have no effect, and the two can silently drift apart.

Either drop one copy from the Makefile lists, or promote `ArithmeticLibrary` to a single shared submodule at the top level.

---

## 10. Dead testbenches referencing modules that no longer exist

**Confirmed (inspection).** These compile against nothing and will fail if anyone adds them to a build:

| File | Instantiates | Status |
|---|---|---|
| `SystolicMesh/testbenches/TB_Mesh_2x2.sv` | `Mesh` | no such module |
| `SystolicMesh/testbenches/TB_Mesh_3x3.sv` | `Mesh` | no such module |
| `SystolicMesh/testbenches/TB_Mesh_5x5.sv` | `Mesh` | no such module |
| `SystolicMesh/testbenches/TB_Mesh_8x8.sv` | `Mesh` | no such module |
| `ArithmeticLibrary/Divider/FP32/testbenches/TB_Divider_FP32.sv` | `divide_32` | no such module; still present — it lives in the shared submodule with two checkouts |

The four `TB_Mesh_*` files are the same 380–630 line testbench regenerated per size, differing only in `N`, the hardcoded FP32 constant tables and the `wait_for_pe_idle` case arms. They target an older mesh interface that had a `select_accumulator_i` input; the current `PEMesh` drives accumulator draining internally.

`TB_Divider_FP32.sv` also declares itself as `module TB_Multi_FP32` and performs no checking — it just drives 13 vector pairs.

The old names `divide_32`, `multiply_32` and `Adder_32` also survive in a commented-out block in `SystolicMesh/src/engine/MAC.sv`. The equivalent block in `GPNAE/src/SeLu.sv` has been removed.

---

## 11. `R4Booth` contains a dead adder tree

**Confirmed (reproduced by synthesis).** `R4Booth.sv` has two summation blocks. The sequential one is dead:

```systemverilog
sum_final <= '0;
for (int k = 0; k < NUM_PP; k++) begin
  if (k == 0) sum_final <= pp_reg[k];
  else sum_final <= sum_final + pp_reg[k];
end
```

Non-blocking assignments in a loop mean only the last iteration takes effect, and `sum_final` is never read. The real output comes from the `adder_tree_comb` `always_comb` block below it. Vivado confirms:

```
WARNING: [Synth 8-6014] Unused sequential element sum_final_reg was removed.
WARNING: [Synth 8-6014] Unused sequential element valid_s2_reg was removed.
```

Harmless, but it reads as the primary datapath and will mislead the next person. Delete it.

---

## 12. `dropout` is inert as integrated

**Confirmed (inspection).** `sienna_top.sv` ties `.training_mode(1'b0)` on every lane, and the inference path in `dropout.sv` is a combinational pass-through. So the LFSR, the threshold comparison and the `fp32Multiplier` scale path are never exercised in any full-pipeline test, regardless of `DROPOUT_P_PERCENT`.

The golden model matches this — `apply_dropout(..., training=False)` also returns a copy — so the tests agree with the hardware. But `DROPOUT_P_PERCENT` flows from `regression.py` all the way into the RTL while having no observable effect, which reads as though it were being tested.

---

## 13. Synthesis: memories inferred as registers

**Confirmed (reproduced).** From `VIVADO/SIENNA.runs/synth_1/`, targeting `xc7a200tfbg484-3`:

| Resource | Used | Available | Util |
|---|---|---|---|
| Slice LUTs | 132,262 | 134,600 | **98.26%** |
| Slice registers | 48,598 | 269,200 | 18.05% |
| Block RAM tiles | 4 | 365 | 1.10% |
| DSPs | 0 | 740 | 0.00% |

Synthesis completes with 0 errors, but the design essentially fills the part with almost no BRAM and no DSP usage. The cause is visible in the warnings:

```
WARNING: [Synth 8-4767] Trying to implement RAM 'mem_reg' in registers.
  Block RAM or DRAM implementation is not possible  [src/fwft.sv:50]
WARNING: [Synth 8-4767] Trying to implement RAM 'gpnae_out_mem_reg' in registers.
  [src/sienna_top.sv:343]
```

`fwft`'s asynchronous reset over the whole memory array and `gpnae_out_mem`'s access pattern both block BRAM inference. Converting those to BRAM-friendly forms — synchronous-reset or no-reset memory, single write port, registered read — is the highest-leverage area reduction available.

0 DSPs is expected rather than a defect: the FP32 units are custom logic built from `R4Booth` and Karatsuba stages, not `A*B` expressions that Vivado would map onto DSP48 slices.

There is also a cluster of `[Synth 8-7137] ... has both Set and reset with same priority ... may cause simulation mismatches` warnings on `fwft`'s `mem_reg` and on `sienna_top`'s `fifo2_wr_data_reg`, which are worth resolving on their own.

---

## 14. `fwft` silently drops data when full

**Confirmed (inspection), by design.** `src/fwft.sv` ties `wr_ready_o` high and, on a write into a full FIFO, overwrites the oldest entry and advances `rd_ptr`. There is no overflow flag.

This is deliberate — the comment says so — but it means any upstream rate mismatch corrupts data with no indication. FIFO1 is 256 deep against 256 elements, so it cannot overflow at `N=16`; FIFO2 is 16 deep and is protected only by the dispatcher's `fifo2_count[target_lane] <= FIFO2_DEPTH - 4` check. Changing `NUM_LANES`, `POOL_H`/`POOL_W`, or the dispatcher's backpressure margin can silently break this.

Adding an overflow sticky bit would make the whole class of problems observable.

---

## Cross-reference: which issue explains which failure

| Symptom | Issue |
|---|---|
| Stall in `GPNAE_ROUND`, 0 words captured | 1 — check `ACTIVATION_CODE` is 1, 2 or 3 |
| `matmul_ident_selu` element [0] = 0 | 3 (fixed) |
| SELU negative results ~4.8% low | 5b (fixed) |
| `tanh(0)` returns −0.5 | 5c (fixed) |
| Low exact-match percentages | not a bug — truncating FP units |
| `exp=0.000000 act=0.000000` in log lines | 7 — the conversion really is broken, not the formatting |
| SELU negatives wrong above \|x\| ≈ 2.5 | 5d (fixed) |
| Vivado `cntlz8` critical warning | 9 — GPNAE-only builds are clean now |
| Vivado "unused sequential element" warnings | 11 |
| 98% LUT utilization | 13 |

A note on test strength, from an episode worth not repeating. While a
pass-through mode was briefly (and wrongly) added to GPNAE, `matmul_ones_idle`
passed at 100% exact on RTL that was duplicating every element, because
`ones @ ones` makes every output identical. `matmul_small_exact`, with distinct
values, failed 37 of 81 on that same RTL. Both tests have since been removed as
invalid, but the principle stands: a stimulus whose expected output is uniform
cannot detect duplication, reordering or off-by-one. When validating anything
that touches element ordering, use distinct values.
