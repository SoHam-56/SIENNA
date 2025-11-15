`timescale 1ns / 1ps

module dropout #(
    parameter DATA_WIDTH = 8,
    parameter DROPOUT_P  = 0.5,
    parameter LFSR_WIDTH = 32
) (
    input logic clk,
    input logic rst_n,
    input logic en,
    input logic training_mode,
    input logic [DATA_WIDTH-1:0] data_in,
    output logic [DATA_WIDTH-1:0] data_out,
    output logic valid_out
);
  logic [LFSR_WIDTH-1:0] lfsr_state;
  logic [LFSR_WIDTH-1:0] lfsr_next_state;
  localparam int FRACTIONAL_BITS = DATA_WIDTH;
  localparam int SCALING_FACTOR_FIXED_POINT = ((1.0 - DROPOUT_P) < 1e-6) ? 0 : $rtoi(
      (1.0 / (1.0 - DROPOUT_P)) * (1 << FRACTIONAL_BITS)
  );
  localparam int INTERMEDIATE_WIDTH = 2 * DATA_WIDTH;

  always_comb begin
    unique case (LFSR_WIDTH)
      8:
      lfsr_next_state = {
        lfsr_state[LFSR_WIDTH-2:0], (lfsr_state[7] ^ lfsr_state[3] ^ lfsr_state[2] ^ lfsr_state[1])
      };
      16:
      lfsr_next_state = {
        lfsr_state[15] ^ lfsr_state[14] ^ lfsr_state[12] ^ lfsr_state[3], lfsr_state[15:1]
      };
      32:
      lfsr_next_state = {
        lfsr_state[31] ^ lfsr_state[21] ^ lfsr_state[1] ^ lfsr_state[0], lfsr_state[31:1]
      };
      default:
      lfsr_next_state = {lfsr_state[LFSR_WIDTH-1] ^ lfsr_state[0], lfsr_state[LFSR_WIDTH-1:1]};
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr_state <= {{(LFSR_WIDTH - 1) {1'b0}}, 1'b1};
    else if (en) lfsr_state <= lfsr_next_state;
  end

  always_comb begin
    if (training_mode) begin
      data_out = (lfsr_state < $rtoi(DROPOUT_P * (1 << LFSR_WIDTH))) ? '0 : data_in;
    end else begin
      logic [2*DATA_WIDTH-1:0] intermediate_product;
      intermediate_product = data_in * SCALING_FACTOR_FIXED_POINT;
      data_out = intermediate_product[2*DATA_WIDTH-1:DATA_WIDTH];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) valid_out <= 1'b0;
    else valid_out <= en;
  end
endmodule

