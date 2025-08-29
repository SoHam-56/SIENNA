`timescale 1ns/1ps
module SRAM #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 8
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [ADDR_WIDTH-1:0]  addr,
    input  logic                   wr_en,
    input  logic                   rd_en,
    input  logic [DATA_WIDTH-1:0]  wdata,
    output logic [DATA_WIDTH-1:0]  rdata
);
    logic [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (int i = 0; i < (1<<ADDR_WIDTH); i++) mem[i] <= '0;
        else if (wr_en)
            mem[addr] <= wdata;
    end

    always_ff @(posedge clk) begin
        if (rd_en)
            rdata <= mem[addr];
    end
endmodule