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
    input wire [DATA_WIDTH-1:0] zero_point_i  // the dropped word where the format has a zero point (int8, D-5); floats ignore it
);

  // The downstream credit is the upstream's: a beat is put here only on a slot already reserved past this unit's pipeline.
  wire                  in_valid = in.put;
  wire [DATA_WIDTH-1:0] data_in  = in.data;
  logic [DATA_WIDTH-1:0] data_out;
  logic                  valid_out;
  assign out.put   = valid_out;
  assign out.data  = data_out;
  assign in.credit = out.credit;
`ifndef SYNTHESIS
  // Interface widths are not elaboration constants in Verilator, so the link widths are checked at time 0.
  initial
    if ($bits(in.data) != DATA_WIDTH || $bits(out.data) != DATA_WIDTH || $bits(in.credit) != $bits(out.credit))
      $fatal(1, "dropout: links need data %0d bits and equal credit widths, found %0d/%0d and %0d/%0d", DATA_WIDTH, $bits(in.data),
             $bits(in.credit), $bits(out.data), $bits(out.credit));
`endif

  localparam bit SCALES = (sienna_fmt_pkg::DROPOUT_SCALE_LATENCY > 0);  // a kept word is multiplied; else it leaves as it came, in the same cycle
  localparam logic [63:0] MAX_LFSR_VAL_64 = 64'((64'(1) << LFSR_WIDTH) - 64'(1));
  localparam logic [63:0] THRESHOLD_CALC_64 = (MAX_LFSR_VAL_64 * 64'(DROPOUT_P_PERCENT)) / 64'(100);
  localparam logic [LFSR_WIDTH-1:0] DROPOUT_THRESHOLD = THRESHOLD_CALC_64[LFSR_WIDTH-1:0];

  logic [LFSR_WIDTH-1:0] lfsr_state, lfsr_next;

  // A fresh 32-bit word per beat, so neighbouring keep/drop decisions are independent.
  always_comb begin
    lfsr_next = lfsr_state;
    for (int i = 0; i < LFSR_WIDTH; i++) begin
      if (LFSR_WIDTH == 32)
        lfsr_next = {lfsr_next[30:0], lfsr_next[31] ^ lfsr_next[21] ^ lfsr_next[1] ^ lfsr_next[0]};
      else lfsr_next = {lfsr_next[LFSR_WIDTH-2:0], lfsr_next[LFSR_WIDTH-1] ^ lfsr_next[1]};
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr_state <= '1;
    else if (reseed_i) lfsr_state <= seed_i;
    else if (in_valid) lfsr_state <= lfsr_next;
  end

  // The scale is fixed at elaboration; a different drop rate needs its own 1/(1-p).
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial begin
    if (SCALES && DROPOUT_P_PERCENT != 50 && CONST_SCALE == sienna_fmt_pkg::DROPOUT_KEEP_SCALE)
      $error("dropout: CONST_SCALE is 2.0, which is only 1/(1-p) for DROPOUT_P_PERCENT = 50");
  end
`endif

  // Training: every beat takes the multiplier, and its keep/drop decision waits here for the product.
  localparam int KQ_DEPTH = 16;
  logic keep_q[KQ_DEPTH];
  logic [$clog2(KQ_DEPTH)-1:0] kq_wr, kq_rd;

  // ---------------------------------------------------------
  // MUX Bypass Logic
  // ---------------------------------------------------------
  logic                  mult_valid_in;
  logic [DATA_WIDTH-1:0] mult_out;
  logic                  mult_done;

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
      data_out  = (lfsr_next >= DROPOUT_THRESHOLD) ? data_in : sienna_fmt_pkg::fmt_drop(data_in, zero_point_i);
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
        keep_q[kq_wr] <= (lfsr_next >= DROPOUT_THRESHOLD);  // the word this beat advances to
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
  end else if (sienna_fmt_pkg::mul_latency(DATA_WIDTH, DATA_WIDTH) != sienna_fmt_pkg::DROPOUT_SCALE_LATENCY) begin : G_BAD_LATENCY
    $fatal(1, "dropout: fmt_mul takes %0d cycles at %0d bits, but DROPOUT_SCALE_LATENCY is %0d", sienna_fmt_pkg::mul_latency(DATA_WIDTH, DATA_WIDTH),
           DATA_WIDTH, sienna_fmt_pkg::DROPOUT_SCALE_LATENCY);
  end else begin : G_SCALE  // x * CONST_SCALE in the format: fp32Multiplier, bf16Multiplier
    fmt_mul #(.DATA_WIDTH(DATA_WIDTH), .EXT_DATA_WIDTH(DATA_WIDTH)) MUL (.clk_i(clk), .rstn_i(rst_n), .valid_i(mult_valid_in), .a_i(data_in),
        .b_i(CONST_SCALE), .product_o(mult_out), .done_o(mult_done));
  end

endmodule
