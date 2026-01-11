`timescale 1ns / 1ps

module fwft #(
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 32,
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH)
) (
    input wire clk_i,
    input wire rstn_i,

    // Write Interface
    input  wire                  wr_en_i,
    input  wire [DATA_WIDTH-1:0] data_i,
    output wire                  full_o,

    // Read Interface
    input  wire                  rd_en_i,  // "Ack" / "Consume" signal
    output wire [DATA_WIDTH-1:0] data_o,   // Valid data sits here when !empty
    output wire                  empty_o,

    // Status
    output wire [ADDR_WIDTH:0] count_o  // Exact number of items in FIFO
);

  // Memory
  logic [DATA_WIDTH-1:0] mem[0:FIFO_DEPTH-1];

  // Pointers
  logic [ADDR_WIDTH:0] wr_ptr;
  logic [ADDR_WIDTH:0] rd_ptr;
  logic [ADDR_WIDTH:0] count;

  // Output assignments
  assign full_o  = (count == FIFO_DEPTH);
  assign empty_o = (count == 0);
  assign count_o = count;

  // FWFT Logic: Data is always pointing to the next item to be read
  assign data_o  = mem[rd_ptr[ADDR_WIDTH-1:0]];

  // FIFO Logic
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wr_ptr <= 0;
      rd_ptr <= 0;
      count  <= 0;
    end else begin
      // Write
      if (wr_en_i && !full_o) begin
        mem[wr_ptr[ADDR_WIDTH-1:0]] <= data_i;
        wr_ptr <= wr_ptr + 1;
      end

      // Read (Consume)
      if (rd_en_i && !empty_o) begin
        rd_ptr <= rd_ptr + 1;
      end

      // Count Update
      if (wr_en_i && !full_o && !(rd_en_i && !empty_o)) begin
        count <= count + 1;
      end else if (rd_en_i && !empty_o && !(wr_en_i && !full_o)) begin
        count <= count - 1;
      end
    end
  end

endmodule
