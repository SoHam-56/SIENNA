`timescale 1ns / 1ps

module dropout #(
    parameter DATA_WIDTH = 8,
    parameter real DROPOUT_P = 0.5,
    parameter LFSR_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    input  logic training_mode,
    input  logic [DATA_WIDTH-1:0] data_in,
    output logic [DATA_WIDTH-1:0] data_out,
    output logic valid_out
);

    // LFSR state registers
    logic [LFSR_WIDTH-1:0] lfsr_state;
    logic [LFSR_WIDTH-1:0] lfsr_next_state;

    // Fixed-point parameters
    localparam int FRACTIONAL_BITS = DATA_WIDTH;
    localparam int SCALING_FACTOR_FIXED_POINT = ((1.0 - DROPOUT_P) < 1e-6) ? 0 :
                                                 int'((1.0 / (1.0 - DROPOUT_P)) * (1 << FRACTIONAL_BITS));
    localparam int INTERMEDIATE_WIDTH = 2 * DATA_WIDTH;

    // Dropout threshold for LFSR comparison
    localparam logic [LFSR_WIDTH-1:0] DROPOUT_THRESHOLD = LFSR_WIDTH'(DROPOUT_P * real'((1 << LFSR_WIDTH) - 1));

    // LFSR next state logic (combinational)
    always_comb begin
        unique case (LFSR_WIDTH)
            8: begin
                lfsr_next_state = {lfsr_state[6:0],
                                   lfsr_state[7] ^ lfsr_state[5] ^ lfsr_state[4] ^ lfsr_state[3]};
            end
            16: begin
                lfsr_next_state = {lfsr_state[0] ^ lfsr_state[2] ^ lfsr_state[3] ^ lfsr_state[5],
                                   lfsr_state[15:1]};
            end
            32: begin
                lfsr_next_state = {lfsr_state[0] ^ lfsr_state[1] ^ lfsr_state[21] ^ lfsr_state[31],
                                   lfsr_state[31:1]};
            end
            default: begin
                lfsr_next_state = {lfsr_state[LFSR_WIDTH-2:0],
                                   lfsr_state[LFSR_WIDTH-1] ^ lfsr_state[0]};
            end
        endcase
    end

    // LFSR state register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            lfsr_state <= {{(LFSR_WIDTH-1){1'b0}}, 1'b1};
        else if (en)
            lfsr_state <= lfsr_next_state;
    end

    // Dropout logic
    logic [DATA_WIDTH-1:0] data_out_comb;
    logic [2*DATA_WIDTH-1:0] intermediate_product;

    always_comb begin
        if (training_mode) begin
            // During training: randomly drop activations
            if (lfsr_state < DROPOUT_THRESHOLD)
                data_out_comb = '0;
            else
                data_out_comb = data_in;
        end else begin
            // During inference: scale by (1 / (1 - DROPOUT_P))
            intermediate_product = data_in * SCALING_FACTOR_FIXED_POINT;
            data_out_comb = intermediate_product[2*DATA_WIDTH-1:DATA_WIDTH];
        end
    end

    // Register outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;
            valid_out <= 1'b0;
        end else begin
            data_out <= data_out_comb;
            valid_out <= en;
        end
    end

endmodule
