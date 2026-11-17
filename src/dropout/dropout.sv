`timescale 1ns / 1ps

// Dropout in the build's format, from sienna_fmt_pkg: in training a kept word is x * CONST_SCALE on fmt_mul (DROPOUT_SCALE_LATENCY cycles), or x where
// that latency is 0 (int8: the 1/keep factor is in the next layer's scale); a dropped word is the package's fmt_drop. Inference passes every word.
module dropout #(
    parameter int                    DATA_WIDTH        = sienna_fmt_pkg::DATA_WIDTH,
    parameter int                    DROPOUT_P_PERCENT = 50,
    parameter int                    LFSR_WIDTH        = 32,
    parameter logic [DATA_WIDTH-1:0] CONST_SCALE       = sienna_fmt_pkg::DROPOUT_KEEP_SCALE  // 2.0, 1/(1-p) at p = 50%
) (
    input wire clk,
    input wire rst_n,

    credit_link_if.consumer in,   // one beat per put; its credits are out's, passed through, since every beat leaves after a fixed delay
    credit_link_if.producer out,  // one beat per put
    input wire                  training_mode,
    input wire                  reseed_i,  // load seed_i into the LFSR; only between sets
    input wire [LFSR_WIDTH-1:0] seed_i,  // must be nonzero
    input wire [LFSR_WIDTH-1:0] seed_next_i,  // the seed_i of the next cycle (sienna_pipe's lane_seed before its register): the reseed's flags form a cycle early
    input wire [DATA_WIDTH-1:0] zero_point_i  // the dropped word where the format has a zero point (int8, D-5); floats ignore it
);
  localparam bit SCALES = (sienna_fmt_pkg::DROPOUT_SCALE_LATENCY > 0);  // a kept word is multiplied; else it leaves as it came, in the same cycle
  // A positive power of two as the scale (2.0 at p = 50%) is an exponent increment: one register, the same words as fmt_mul (subnormal inputs
  // and underflow flush to a signed zero, overflow to infinity, NaN to the canonical quiet NaN). Both float formats carry 8 exponent bits; the
  // tests below use shifts, not part-selects, so an int8 build (DATA_WIDTH 8, SCALES 0) elaborates them too.
  localparam logic [7:0] SCALE_EXP = 8'(CONST_SCALE >> (DATA_WIDTH - 9));
  localparam bit POW2 = SCALES && (DATA_WIDTH > 9) && !CONST_SCALE[DATA_WIDTH-1] && (SCALE_EXP != 8'd0) && (SCALE_EXP != 8'hFF)
                        && ((CONST_SCALE & ((DATA_WIDTH'(1) << (DATA_WIDTH - 9)) - DATA_WIDTH'(1))) == '0);
  localparam logic [63:0] MAX_LFSR_VAL_64 = 64'((64'(1) << LFSR_WIDTH) - 64'(1));
  localparam logic [63:0] THRESHOLD_CALC_64 = (MAX_LFSR_VAL_64 * 64'(DROPOUT_P_PERCENT)) / 64'(100);
  localparam logic [LFSR_WIDTH-1:0] DROPOUT_THRESHOLD = THRESHOLD_CALC_64[LFSR_WIDTH-1:0];
  localparam int KQ_DEPTH = 16;  // keep/drop decisions waiting for their product
  typedef logic [LFSR_WIDTH-1:0][LFSR_WIDTH-1:0] adv_t;  // an advance of k words as XOR masks: row j selects the state bits that make bit j
  localparam adv_t ADV1 = adv_masks(1), ADV2 = adv_masks(2), ADV3 = adv_masks(3);  // one balanced XOR tree per bit (the 32-step loops left a chain: 3 words were 4.9 ns at 45 nm, 0.4 ns as masks)
  localparam int HW = LFSR_WIDTH / 2;  // the threshold compare runs in two halves, kept as flags beside the state (the 32-bit compare on the next word was 1.03 ns at 45 nm)

  // The beat in and the beat out.
  wire                   in_valid;
  wire  [DATA_WIDTH-1:0] data_in;
  logic [DATA_WIDTH-1:0] data_out;
  logic                  valid_out;
  // The LFSR: a fresh word per beat.
  logic [LFSR_WIDTH-1:0] lfsr_state, lfsr_next, lfsr_next2, seed_next;  // the state, one and two beats on, and the seed one beat on
  logic [LFSR_WIDTH-1:0] lfsr_n1, lfsr_n2;  // lfsr_next and lfsr_next2 as registers: each takes one advance per beat, and the flags compare a register (two advances into the compare were 1.12-1.28 ns)
  logic hi_gt, hi_eq, lo_ge;  // lfsr_next against DROPOUT_THRESHOLD by halves: the next beat's decision, kept with the state
  logic keep_next, keep_seed;  // the beat arriving now is kept; the seed's first draw is kept
  logic [2:0] sd1_q, sd2_q;  // the flags of seed_i one and two words on, formed from seed_next_i a cycle early (seed_i into keep_q was 1.13-1.25 ns)
  // Training: every beat takes the multiplier, and its keep/drop decision waits here for the product.
  logic keep_q[KQ_DEPTH];
  logic [$clog2(KQ_DEPTH)-1:0] kq_wr, kq_rd;
  logic                  mult_valid_in;
  logic [DATA_WIDTH-1:0] mult_out;
  logic                  mult_done;

`ifndef SYNTHESIS
  // Interface widths are not elaboration constants in Verilator, so the link widths are checked at time 0.
  initial
    if ($bits(in.data) != DATA_WIDTH || $bits(out.data) != DATA_WIDTH || $bits(in.credit) != $bits(out.credit))
      $fatal(1, "dropout: links need data %0d bits and equal credit widths, found %0d/%0d and %0d/%0d", DATA_WIDTH, $bits(in.data),
             $bits(in.credit), $bits(out.data), $bits(out.credit));
  // The scale is fixed at elaboration; a different drop rate needs its own 1/(1-p).
  initial begin
    if (SCALES && DROPOUT_P_PERCENT != 50 && CONST_SCALE == sienna_fmt_pkg::DROPOUT_KEEP_SCALE)
      $error("dropout: CONST_SCALE is 2.0, which is only 1/(1-p) for DROPOUT_P_PERCENT = 50");
  end
`endif

  // The downstream credit is the upstream's: a beat is put here only on a slot already reserved past this unit's pipeline.
  assign in_valid  = in.put;
  assign data_in   = in.data;
  assign out.put   = valid_out;
  assign out.data  = data_out;
  assign in.credit = out.credit;

  // A fresh 32-bit word per beat, so neighbouring keep/drop decisions are independent.
  // One beat's advance of the LFSR: 32 steps per beat.
  function automatic adv_t adv_masks(input int k);  // advance each unit vector k words: column i of the k-word map
    logic [LFSR_WIDTH-1:0] v;
    adv_masks = '0;
    for (int i = 0; i < LFSR_WIDTH; i++) begin
      v = LFSR_WIDTH'(1) << i;
      for (int r = 0; r < k; r++)  // k beats of 32 single-bit steps, written out here: DC will not evaluate a function call in this constant function (ELAB-425)
        for (int b = 0; b < LFSR_WIDTH; b++)
          if (LFSR_WIDTH == 32) v = {v[30:0], v[31] ^ v[21] ^ v[1] ^ v[0]};
          else v = {v[LFSR_WIDTH-2:0], v[LFSR_WIDTH-1] ^ v[1]};
      for (int j = 0; j < LFSR_WIDTH; j++) adv_masks[j][i] = v[j];
    end
  endfunction
  function automatic logic [LFSR_WIDTH-1:0] lfsr_adv(input logic [LFSR_WIDTH-1:0] s, input adv_t m);  // the advance m holds, as XOR trees
    for (int j = 0; j < LFSR_WIDTH; j++) lfsr_adv[j] = ^(s & m[j]);
  endfunction
  function automatic logic [LFSR_WIDTH-1:0] lfsr_step(input logic [LFSR_WIDTH-1:0] s);  // one beat: 32 LFSR steps
    return lfsr_adv(s, ADV1);
  endfunction
  function automatic logic [2:0] above_by_halves(input logic [LFSR_WIDTH-1:0] v);  // {high half greater, high half equal, low half not below}
    return {v[LFSR_WIDTH-1:HW] > DROPOUT_THRESHOLD[LFSR_WIDTH-1:HW], v[LFSR_WIDTH-1:HW] == DROPOUT_THRESHOLD[LFSR_WIDTH-1:HW],
            v[HW-1:0] >= DROPOUT_THRESHOLD[HW-1:0]};
  endfunction

  // The words one and two beats on (from their registers), and the seed's first word (the testbench reads these nets).
  always_comb begin
    lfsr_next  = lfsr_n1;
    lfsr_next2 = lfsr_n2;
    seed_next  = lfsr_adv(seed_i, ADV1);
    // The decision for the beat arriving now: the flags hold lfsr_next >= DROPOUT_THRESHOLD by halves; a beat arriving as the LFSR reseeds
    // is the seed's first draw (the state advances from the seed for it).
    keep_seed = sd1_q[2] || (sd1_q[1] && sd1_q[0]);
    keep_next = reseed_i ? keep_seed : (hi_gt || (hi_eq && lo_ge));
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lfsr_state <= '1;
      lfsr_n1    <= lfsr_step('1);
      lfsr_n2    <= lfsr_step(lfsr_step('1));
      {hi_gt, hi_eq, lo_ge} <= above_by_halves(lfsr_step('1));  // the word after the reset state
    end else if (reseed_i) begin
      lfsr_state <= in_valid ? seed_next : seed_i;
      lfsr_n1    <= in_valid ? lfsr_adv(seed_i, ADV2) : seed_next;  // each straight from the seed: no advance waits on another
      lfsr_n2    <= in_valid ? lfsr_adv(seed_i, ADV3) : lfsr_adv(seed_i, ADV2);
      {hi_gt, hi_eq, lo_ge} <= in_valid ? sd2_q : sd1_q;
    end else if (in_valid) begin
      lfsr_state <= lfsr_n1;
      lfsr_n1    <= lfsr_n2;
      lfsr_n2    <= lfsr_step(lfsr_n2);
      {hi_gt, hi_eq, lo_ge} <= above_by_halves(lfsr_n2);
    end
  end

  always_ff @(posedge clk) begin  // no reset, as lane_seed: read only at a reseed
    sd1_q <= above_by_halves(lfsr_adv(seed_next_i, ADV1));
    sd2_q <= above_by_halves(lfsr_adv(seed_next_i, ADV2));
  end
`ifndef SYNTHESIS
  a_seed_next: assert property (@(posedge clk) disable iff (!rst_n) reseed_i |-> seed_i == $past(seed_next_i))
    else $error("dropout: a_seed_next: seed_i %h at a reseed is not the cycle before's seed_next_i %h", seed_i, $past(seed_next_i));
`endif

  always_comb begin
    mult_valid_in = 1'b0;
    data_out      = '0;
    valid_out     = 1'b0;

    if (!training_mode) begin
      // INFERENCE: Combinational bypass, starve the multiplier
      data_out  = data_in;
      valid_out = in_valid;
    end else if (!SCALES) begin
      // Training with no scale: in the same cycle, a kept beat unchanged, a dropped beat the format's dropped word (int8: the zero point, D-5).
      data_out  = keep_next ? data_in : sienna_fmt_pkg::fmt_drop(data_in, zero_point_i);
      valid_out = in_valid;
    end else begin
      mult_valid_in = in_valid;
      valid_out     = mult_done;
      // A dropped beat is the format's dropped word of the product: a zero with the input's sign in floats, as x * 0 gives in IEEE arithmetic.
      data_out      = keep_q[kq_rd] ? mult_out : sienna_fmt_pkg::fmt_drop(mult_out, zero_point_i);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      kq_wr <= '0;
      kq_rd <= '0;
    end else begin
      if (SCALES && training_mode && in_valid) begin
        keep_q[kq_wr] <= keep_next;  // the word this beat advances to
        kq_wr <= kq_wr + 1'b1;
      end
      if (SCALES && training_mode && mult_done) kq_rd <= kq_rd + 1'b1;
    end
  end

  if (DATA_WIDTH != sienna_fmt_pkg::DATA_WIDTH) begin : G_BAD_DATA_WIDTH  // fmt_mul and fmt_drop work at the build's width
    $fatal(1, "dropout: DATA_WIDTH %0d, but the build's words are %0d bits", DATA_WIDTH, sienna_fmt_pkg::DATA_WIDTH);
  end else if (!SCALES) begin : G_KEEP
    assign mult_out  = '0;
    assign mult_done = 1'b0;
  end else if (POW2 && sienna_fmt_pkg::DROPOUT_SCALE_LATENCY != 1) begin : G_BAD_POW2_LATENCY
    $fatal(1, "dropout: a power-of-two scale takes one cycle, but DROPOUT_SCALE_LATENCY is %0d", sienna_fmt_pkg::DROPOUT_SCALE_LATENCY);
  end else if (POW2) begin : G_POW2  // x * 2^k by the exponent: the fp32Multiplier's product of x and the constant was the fp32 top's path
    localparam int MAN_W = DATA_WIDTH - 9;
    localparam logic signed [9:0] SCALE_K = 10'($signed({2'b00, SCALE_EXP})) - 10'sd127;  // the increment
    localparam logic [DATA_WIDTH-1:0] CANONICAL_QNAN = {1'b0, 8'hFF, 1'b1, {(MAN_W - 1) {1'b0}}};
    function automatic logic [DATA_WIDTH-1:0] pow2_scale(input logic [DATA_WIDTH-1:0] x);
      logic sign;
      logic [7:0] e;
      logic [MAN_W-1:0] man;
      logic signed [9:0] ne;
      sign = x[DATA_WIDTH-1];
      e    = x[DATA_WIDTH-2-:8];
      man  = x[MAN_W-1:0];
      ne   = 10'($signed({2'b00, e})) + SCALE_K;
      if (e == 8'hFF) return (man != '0) ? CANONICAL_QNAN : x;  // NaN, infinity
      if (e == 8'd0 || ne <= 10'sd0) return {sign, {(DATA_WIDTH - 1) {1'b0}}};  // a subnormal or zero input, or an underflow
      if (ne >= 10'sd255) return {sign, 8'hFF, {MAN_W{1'b0}}};
      return {sign, ne[7:0], man};
    endfunction
    always_ff @(posedge clk or negedge rst_n)
      if (!rst_n) mult_done <= 1'b0;
      else mult_done <= mult_valid_in;
    always_ff @(posedge clk)
      if (mult_valid_in) mult_out <= pow2_scale(data_in);
  end else if (sienna_fmt_pkg::mul_latency(DATA_WIDTH, DATA_WIDTH) != sienna_fmt_pkg::DROPOUT_SCALE_LATENCY) begin : G_BAD_LATENCY
    $fatal(1, "dropout: fmt_mul takes %0d cycles at %0d bits, but DROPOUT_SCALE_LATENCY is %0d", sienna_fmt_pkg::mul_latency(DATA_WIDTH, DATA_WIDTH),
           DATA_WIDTH, sienna_fmt_pkg::DROPOUT_SCALE_LATENCY);
  end else begin : G_SCALE  // x * CONST_SCALE in the format: fp32Multiplier, bf16Multiplier
    fmt_mul #(.DATA_WIDTH(DATA_WIDTH), .EXT_DATA_WIDTH(DATA_WIDTH)) MUL (.clk_i(clk), .rstn_i(rst_n), .valid_i(mult_valid_in), .a_i(data_in),
        .b_i(CONST_SCALE), .product_o(mult_out), .done_o(mult_done));
  end

endmodule
