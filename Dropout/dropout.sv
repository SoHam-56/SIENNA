`timescale 1ns / 1ps

module dropout #(
    // -----------------------------------------------------------------
    // 1. FLEXIBILITY PARAMETERS
    // -----------------------------------------------------------------
    parameter int DATA_WIDTH        = 32,  // e.g., 32 for FP32, 16 for BF16, 8 for INT8
    parameter int DROPOUT_P_PERCENT = 50,  // Probability (0-100)
    parameter int LFSR_WIDTH        = 32,

    // -----------------------------------------------------------------
    // 2. FORMAT-SPECIFIC CONSTANTS (Must be provided by User)
    // -----------------------------------------------------------------
    // Value representing "0" (Drop). Usually '0, but customizable.
    parameter logic [DATA_WIDTH-1:0] CONST_ZERO = '0,

    // Value representing "1.0" (Identity for Inference). 
    // FP32: 32'h3F800000, INT8 (Q4.4): 8'h10
    parameter logic [DATA_WIDTH-1:0] CONST_ONE = 32'h3F800000,

    // Value representing "1 / (1-p)" (Scale Factor).
    // FP32 (50%): 32'h40000000
    parameter logic [DATA_WIDTH-1:0] CONST_SCALE = 32'h40000000
) (
    input logic clk,
    input logic rst_n,
    input logic en,
    input logic training_mode,
    input logic [DATA_WIDTH-1:0] data_in,

    output logic [DATA_WIDTH-1:0] data_out,
    output logic valid_out
);

  // ---------------------------------------------------------
  // 1. Probability Logic (Format Agnostic)
  // ---------------------------------------------------------
  localparam logic [63:0] MAX_LFSR_VAL_64 = 64'((64'(1) << LFSR_WIDTH) - 64'(1));
  localparam logic [63:0] THRESHOLD_CALC_64 = (MAX_LFSR_VAL_64 * 64'(DROPOUT_P_PERCENT)) / 64'(100);
  localparam logic [LFSR_WIDTH-1:0] DROPOUT_THRESHOLD = THRESHOLD_CALC_64[LFSR_WIDTH-1:0];

  logic [LFSR_WIDTH-1:0] lfsr_state, lfsr_next;

  // Generic Galois LFSR
  always_comb begin
    lfsr_next = lfsr_state;
    // Simple tap selection based on width (Add more cases if needed)
    if (LFSR_WIDTH == 32)
      lfsr_next = {
        lfsr_state[30:0], lfsr_state[31] ^ lfsr_state[21] ^ lfsr_state[1] ^ lfsr_state[0]
      };
    else lfsr_next = {lfsr_state[LFSR_WIDTH-2:0], lfsr_state[LFSR_WIDTH-1] ^ lfsr_state[1]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr_state <= '1;
    else if (en) lfsr_state <= lfsr_next;
  end

  // ---------------------------------------------------------
  // 2. Control Logic (Selects A and B inputs)
  // ---------------------------------------------------------
  logic [DATA_WIDTH-1:0] op_a;
  logic [DATA_WIDTH-1:0] op_b;

  always_comb begin
    // Default: Keep Data, Multiply by Scale
    op_a = data_in;
    op_b = CONST_SCALE;

    if (training_mode) begin
      // TRAINING: Check Random Drop
      if (lfsr_state < DROPOUT_THRESHOLD) begin
        // Drop: Multiply 0 * Scale -> 0
        op_a = CONST_ZERO;
      end
    end else begin
      // INFERENCE: Multiply Data * 1.0 -> Data
      op_b = CONST_ONE;
    end
  end

  // ---------------------------------------------------------
  // 3. THE "SOCKET" (Instantiate your multiplier here)
  // ---------------------------------------------------------
  // REPLACE 'multiply_32' below with your specific module 
  // (multiply_16, multiply_int8, etc.)
  // Ensure the port names match!
  // ---------------------------------------------------------

  multiply_32 #(
  // If your multiplier module supports parameters, pass them here
  // .WIDTH(DATA_WIDTH) 
  ) u_math_core (
      .clk_i  (clk),
      .rstn_i (rst_n),
      .valid_i(en),
      .A      (op_a),
      .B      (op_b),
      .Result (data_out),
      .done_o (valid_out)
  );

endmodule
