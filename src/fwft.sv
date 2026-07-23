`timescale 1ns / 1ps

module fwft #(
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 32,
    parameter ADDR_WIDTH = (FIFO_DEPTH > 1) ? $clog2(FIFO_DEPTH) : 1
) (
    input wire clk_i,
    input wire rstn_i,

    // Write Interface: Never backpressures. Will overwrite if full.
    input  wire                  wr_valid_i,
    input  wire [DATA_WIDTH-1:0] wr_data_i,
    output wire                  wr_ready_o,

    // Read Interface -- FWFT
    output wire                  rd_valid_o,
    output wire [DATA_WIDTH-1:0] rd_data_o,
    input  wire                  rd_ready_i,

    // Status
    output wire [ADDR_WIDTH:0] count_o
);

  logic [DATA_WIDTH-1:0] mem[0:FIFO_DEPTH-1];

  logic [ADDR_WIDTH-1:0] wr_ptr;
  logic [ADDR_WIDTH-1:0] rd_ptr;
  logic [ADDR_WIDTH:0] count;

  // The FIFO is always ready to accept data, because it can overwrite
  assign wr_ready_o = 1'b1;

  wire wr_fire = wr_valid_i && wr_ready_o;  // Effectively just wr_valid_i
  wire rd_fire = rd_ready_i && rd_valid_o;

  assign rd_valid_o = (count > 0);
  assign count_o    = count;
  assign rd_data_o  = mem[rd_ptr];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
      count  <= '0;
    end else begin

      // Write Pointer & Data Logic
      if (wr_fire) begin
        mem[wr_ptr] <= wr_data_i;
        if (wr_ptr == FIFO_DEPTH - 1) wr_ptr <= '0;
        else wr_ptr <= wr_ptr + 1'b1;
      end

      // Read Pointer Logic
      // The read pointer advances if the consumer reads (rd_fire), OR 
      // if a write tramples the oldest data in a full FIFO without a simultaneous read.
      if (rd_fire || (wr_fire && (count == FIFO_DEPTH))) begin
        if (rd_ptr == FIFO_DEPTH - 1) rd_ptr <= '0;
        else rd_ptr <= rd_ptr + 1'b1;
      end

      // Count Update Logic
      if (wr_fire && !rd_fire) begin
        // Cap the count at FIFO_DEPTH. If it's full, it stays full.
        if (count < FIFO_DEPTH) count <= count + 1'b1;
      end else if (rd_fire && !wr_fire) begin
        // Only decrement if we read without a new write coming in
        count <= count - 1'b1;
      end
      // If both fire (or neither fire), count remains unchanged
    end
  end

endmodule
