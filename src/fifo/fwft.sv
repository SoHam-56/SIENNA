`timescale 1ns / 1ps

// First-word-fall-through FIFO on credit links: the writer holds one credit per free slot, the oldest word is put while the reader's credit is held.
module fwft #(
    parameter DATA_WIDTH = 32,
    parameter FIFO_DEPTH = 32,
    parameter ADDR_WIDTH = (FIFO_DEPTH > 1) ? $clog2(FIFO_DEPTH) : 1,
    parameter int OUT_MAX = 8,  // the most credits the reader may grant
    parameter int OUT_CRW = 1   // out.credit width
) (
    input wire clk_i,
    input wire rstn_i,

    credit_link_if.consumer in,   // advertises FIFO_DEPTH after reset, one per cycle, then one credit per pop
    credit_link_if.producer out,  // the oldest word, one per put; a put is a pop

    output wire [ADDR_WIDTH:0] count_o
);
  localparam int OCW = $clog2(OUT_MAX + 1);

  // The words, the pointers and the count.
  logic [DATA_WIDTH-1:0] mem[0:FIFO_DEPTH-1];
  logic [ADDR_WIDTH-1:0] wr_ptr;
  logic [ADDR_WIDTH-1:0] rd_ptr;
  logic [ADDR_WIDTH:0] count;
  // The oldest word as a register (mem[rd_ptr] read through the depth's select was the reader's input path: FIFO2 -> maxpool's compare):
  // each edge it loads the slot rd_ptr moves to, or the word written into that slot at the same edge.
  logic [DATA_WIDTH-1:0] head;
  logic [ADDR_WIDTH-1:0] rd_ptr_n;
  wire wr_take;  // the write lands this edge
  // The reader's credits, and a write or a pop this cycle.
  logic [OCW-1:0] out_cnt;
  wire wr_fire;
  wire rd_fire;
  // Input credits: freed slots not yet credited, and the credit.
  logic [ADDR_WIDTH:0] in_owed;
  logic in_cr;

`ifndef SYNTHESIS
  // Interface widths are not elaboration constants in Verilator, so the link widths are checked at time 0.
  initial
    if ($bits(in.data) != DATA_WIDTH || $bits(in.credit) != 1 || $bits(out.data) != DATA_WIDTH || $bits(out.credit) != OUT_CRW)
      $fatal(1, "fwft: links need data %0d bits, in.credit 1 and out.credit %0d, found %0d/%0d and %0d/%0d", DATA_WIDTH, OUT_CRW,
             $bits(in.data), $bits(in.credit), $bits(out.data), $bits(out.credit));
`endif

  credit_counter #(.MAX(OUT_MAX), .CRW(OUT_CRW)) out_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(out.put), .credit_i(out.credit),
                                                        .has_credit_o(), .count_o(out_cnt));

  assign wr_fire = in.put;
  assign rd_fire = (count != 0) && (out_cnt != 0);

  assign out.put  = rd_fire;
  assign out.data = head;
  assign wr_take  = wr_fire && (count < FIFO_DEPTH || rd_fire);
  assign rd_ptr_n = !rd_fire ? rd_ptr : (rd_ptr == ADDR_WIDTH'(FIFO_DEPTH - 1)) ? '0 : rd_ptr + 1'b1;
  always_ff @(posedge clk_i) head <= (wr_take && wr_ptr == rd_ptr_n) ? in.data : mem[rd_ptr_n];  // no reset: read only with a put
  assign count_o  = count;

  // Input credits: one per free slot, FIFO_DEPTH after reset, then one per pop, at most one a cycle.
  assign in.credit = in_cr;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wr_ptr  <= '0;
      rd_ptr  <= '0;
      count   <= '0;
      in_owed <= (ADDR_WIDTH + 1)'(FIFO_DEPTH);
      in_cr   <= 1'b0;
    end else begin
      in_owed <= in_owed + (ADDR_WIDTH + 1)'(rd_fire) - (ADDR_WIDTH + 1)'(in_owed != '0);
      in_cr   <= in_owed != '0;
      if (wr_take) begin
        mem[wr_ptr] <= in.data;
        if (wr_ptr == FIFO_DEPTH - 1) wr_ptr <= '0;
        else wr_ptr <= wr_ptr + 1'b1;
      end
      if (rd_fire) begin
        if (rd_ptr == FIFO_DEPTH - 1) rd_ptr <= '0;
        else rd_ptr <= rd_ptr + 1'b1;
      end
      if (wr_fire && !rd_fire && count < FIFO_DEPTH) count <= count + 1'b1;
      else if (rd_fire && !wr_fire) count <= count - 1'b1;
    end
  end

`ifndef SYNTHESIS
  // A put needs a free slot: with FIFO_DEPTH words held no credit can be outstanding; nothing is overwritten any more.
  a_fifo_room: assert property (@(posedge clk_i) disable iff (!rstn_i) in.put |-> int'(count) < FIFO_DEPTH)
    else $error("fwft: a_fifo_room: put into a full FIFO (%0d words)", count);
  a_head: assert property (@(posedge clk_i) disable iff (!rstn_i) out.put |-> head == mem[rd_ptr])
    else $error("fwft: a_head: the registered head %h is not the oldest word %h (slot %0d)", head, mem[rd_ptr], rd_ptr);
`endif

endmodule
