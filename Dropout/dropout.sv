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

    output logic [DATA_WIDTH-1:0] data_out,
    output logic                  valid_out
);

  localparam logic [63:0] MAX_LFSR_VAL_64 = 64'((64'(1) << LFSR_WIDTH) - 64'(1));
  localparam logic [63:0] THRESHOLD_CALC_64 = (MAX_LFSR_VAL_64 * 64'(DROPOUT_P_PERCENT)) / 64'(100);
  localparam logic [LFSR_WIDTH-1:0] DROPOUT_THRESHOLD = THRESHOLD_CALC_64[LFSR_WIDTH-1:0];

  logic [LFSR_WIDTH-1:0] lfsr_state, lfsr_next;

  always_comb begin
    lfsr_next = lfsr_state;
    if (LFSR_WIDTH == 32) begin
      lfsr_next = {
        lfsr_state[30:0], lfsr_state[31] ^ lfsr_state[21] ^ lfsr_state[1] ^ lfsr_state[0]
      };
    end else lfsr_next = {lfsr_state[LFSR_WIDTH-2:0], lfsr_state[LFSR_WIDTH-1] ^ lfsr_state[1]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr_state <= '1;
    else if (in_valid) lfsr_state <= lfsr_next;
  end

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
      if (lfsr_state < DROPOUT_THRESHOLD) begin
        // DROP: Combinational bypass, starvation of DSP
        data_out  = CONST_ZERO;
        valid_out = in_valid;
      end else begin
        // KEEP (Scale): Feed DSP Block
        mult_valid_in = in_valid;
        data_out      = mult_out;
        valid_out     = mult_done;
      end
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
