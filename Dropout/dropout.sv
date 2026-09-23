`timescale 1ns / 1ps

module dropout #(
    parameter int                    DATA_WIDTH        = 32,
    parameter int                    DROPOUT_P_PERCENT = 50,
    parameter int                    LFSR_WIDTH        = 32,
    parameter logic [DATA_WIDTH-1:0] CONST_ZERO        = '0,
    parameter logic [DATA_WIDTH-1:0] CONST_ONE         = 32'h3F800000,
    parameter logic [DATA_WIDTH-1:0] CONST_SCALE       = 32'h40000000
) (
    input wire clk,
    input wire rst_n,

    input wire                  in_valid,
    input wire                  training_mode,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire                  reseed_i,  // load seed_i into the LFSR; only between sets
    input wire [LFSR_WIDTH-1:0] seed_i,  // must be nonzero

    output logic [DATA_WIDTH-1:0] data_out,
    output logic                  valid_out
);

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
  initial begin
    if (DROPOUT_P_PERCENT != 50 && CONST_SCALE == 32'h40000000)
      $error("dropout: CONST_SCALE is 2.0, which is only 1/(1-p) for DROPOUT_P_PERCENT = 50");
  end

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
    end else begin
      mult_valid_in = in_valid;
      valid_out     = mult_done;
      // A dropped beat is a zero with the input's sign, as x * 0 gives in IEEE arithmetic.
      data_out      = keep_q[kq_rd] ? mult_out : {mult_out[DATA_WIDTH-1], {(DATA_WIDTH - 1) {1'b0}}};
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      kq_wr <= '0;
      kq_rd <= '0;
    end else begin
      if (training_mode && in_valid) begin
        keep_q[kq_wr] <= (lfsr_next >= DROPOUT_THRESHOLD);  // the word this beat advances to
        kq_wr <= kq_wr + 1'b1;
      end
      if (training_mode && mult_done) kq_rd <= kq_rd + 1'b1;
    end
  end

  fp32Multiplier MUL (
      .clk_i      (clk),
      .rstn_i     (rst_n),
      .valid_i    (mult_valid_in),
      .A          (data_in),
      .B          (CONST_SCALE),
      .result_o   (mult_out),
      .done_o     (mult_done),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

endmodule
