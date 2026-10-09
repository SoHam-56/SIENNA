// Testbench L9 consumer for one lane, included once per testbench file (no include guard).
// It advertises slots_i after reset and credits each freed slot from the next cycle on, unless a random stall withholds it.
// Link signals change on the falling edge; valid_o and data_o show the put, for the testbench to sample at the rising edge.
// +l9_overgrant advertises one slot more than slots_i, to show a_l9_slots firing.
module tb_l9_sink #(
    parameter int DATA_W    = 32,
    parameter int MAX_SLOTS = 64  // the producer's OUT_MAX; the checker's a_all_back runs only when slots_i equals it
) (
    input  logic              clk_i,
    input  logic              rstn_i,
    input  int                slots_i,      // slots advertised after reset (1..MAX_SLOTS)
    input  int                stall_pct_i,  // percent of credit returns withheld at random
    input  logic              drain_i,      // the test is quiet: every slot must be credited back
    credit_link_if            lnk,  // this side is the consumer; the checker watches it
    output logic              valid_o,
    output logic [DATA_W-1:0] data_o,
    output logic              home_o        // every slot credited back and no credit on the wire
);
  logic cr = 1'b0;
  int over = 0;  // extra slots advertised by the over-grant fault
  initial over = $test$plusargs("l9_overgrant") ? 1 : 0;
  int owed = 0;  // freed or advertised slots not yet credited
  int held = 0;
  assign lnk.credit = cr;
  assign valid_o = lnk.put;
  assign data_o  = lnk.data;
  assign home_o  = (owed == 0) && !cr && (held == slots_i);
  always @(negedge clk_i) begin
    if (!rstn_i) begin
      cr   = 1'b0;
      owed = slots_i + over;
      held = 0;
    end else begin
      held += int'(cr);
      cr = 1'b0;
      if (owed > 0 && !(stall_pct_i > 0 && $urandom_range(99) < stall_pct_i)) begin
        cr = 1'b1;
        owed--;
      end
      if (lnk.put) begin
        owed++;
        held--;
      end
    end
  end
  credit_link_checker #(.SLOTS(MAX_SLOTS)) chk (.clk_i(clk_i), .rstn_i(rstn_i),
                                                .drained_i(drain_i && slots_i == MAX_SLOTS && owed == 0 && !cr), .lnk(lnk));
  // The checker's SLOTS is the producer's MAX; this bounds the credits outstanding by the slots configured at run time.
  a_l9_slots: assert property (@(posedge clk_i) disable iff (!rstn_i) chk.granted <= slots_i)
    else $error("tb_l9_sink: %0d L9 credits outstanding, above the %0d slots configured", chk.granted, slots_i);
endmodule
