// Code your design here
`include "SRAM.sv"
`include "Maxpool_2D.sv"
`include "dropout_module.sv"
`timescale 1ns/1ps
module Neural_Processing_System #(
    parameter DATA_WIDTH = 8,
    parameter IN_ROWS = 6,
    parameter IN_COLS = 6,
    parameter SEG_ROWS = 2,     // New parameter for maxpool
    parameter SEG_COLS = 2,     // New parameter for maxpool
    parameter STRIDE_ROWS = 2,  // New parameter for maxpool
    parameter STRIDE_COLS = 2,  // New parameter for maxpool
    parameter DROPOUT_P = 0.5,
    parameter PADDING = 1
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done
);
    // Memory dimensions
    localparam IN_SIZE = IN_ROWS * IN_COLS;
    // Updated output size calculation to use stride and padding logic from new Maxpool module
    localparam OUT_ROWS = (PADDING == 1) ? ((IN_ROWS - 1) / STRIDE_ROWS) + 1 : ((IN_ROWS - SEG_ROWS) / STRIDE_ROWS) + 1;
    localparam OUT_COLS = (PADDING == 1) ? ((IN_COLS - 1) / STRIDE_COLS) + 1 : ((IN_COLS - SEG_COLS) / STRIDE_COLS) + 1;
    localparam OUT_SIZE = OUT_ROWS * OUT_COLS;

    // SRAM connections
    logic [$clog2(IN_SIZE)-1:0] in_addr;
    logic [DATA_WIDTH-1:0] in_data;
    logic in_rd_en;
    
    logic [$clog2(OUT_SIZE)-1:0] out_addr;
    logic [DATA_WIDTH-1:0] out_data;
    logic out_wr_en;

    // Processing pipeline
    logic [DATA_WIDTH-1:0] maxpool_out;
    logic maxpool_done;
    logic maxpool_valid;
    
    logic [DATA_WIDTH-1:0] dropout_out;
    logic dropout_valid;

    // SRAM Instances
    SRAM #(
        .ADDR_WIDTH($clog2(IN_SIZE)),
        .DATA_WIDTH(DATA_WIDTH)
    ) input_sram (
        .clk(clk), .rst_n(rst_n),
        .addr(in_addr), .rd_en(in_rd_en),
        .wdata('0), .wr_en('0),
        .rdata(in_data)
    );

    SRAM #(
        .ADDR_WIDTH($clog2(OUT_SIZE)),
        .DATA_WIDTH(DATA_WIDTH)
    ) output_sram (
        .clk(clk), .rst_n(rst_n),
        .addr(out_addr), .wr_en(out_wr_en),
        .wdata(out_data), .rd_en('0),
        .rdata()
    );

    // MaxPool Instance
    Maxpool_2D #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_ROWS(IN_ROWS), .IN_COLS(IN_COLS),
        .SEG_ROWS(SEG_ROWS), .SEG_COLS(SEG_COLS),
        .STRIDE_ROWS(STRIDE_ROWS), .STRIDE_COLS(STRIDE_COLS),
        .PADDING(PADDING)
    ) maxpool (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .done(maxpool_done),
        .in_addr(in_addr),
        .in_data(in_data),
        .in_rd_en(in_rd_en),
        .out_data(maxpool_out),
        .out_valid(maxpool_valid)
    );

    // Dropout Instance
    dropout_module #(
        .DATA_WIDTH(DATA_WIDTH),
        .DROPOUT_P(DROPOUT_P)
    ) dropout (
        .clk(clk), .rst_n(rst_n),
        .en(maxpool_valid),
        .training_mode(1'b0),
        .data_in(maxpool_out),
        .data_out(dropout_out),
        .valid_out(dropout_valid)
    );

    // Output SRAM Controller
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_addr <= 0;
            out_wr_en <= 0;
            out_data <= 0;
        end else if (dropout_valid) begin
            out_data <= dropout_out;
            if (out_addr < OUT_SIZE) begin
                 out_wr_en <= 1;
                 out_addr <= out_addr + 1;
            end else begin
                 out_wr_en <= 0;
            end
        end else begin
            out_wr_en <= 0;
        end
    end

    // Global Done Signal
    assign done = maxpool_done;
endmodule