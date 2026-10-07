`timescale 1ns / 100ps

// An L9 consumer that never stalls: it advertises SLOTS after reset, one per cycle, and returns each word's credit the cycle after it arrives.
module l9_sink #(
    parameter int DATA_W = 32,
    parameter int SLOTS  = 64  // at most the producer's OUT_MAX
) (
    input  logic              clk_i,
    input  logic              rstn_i,
    credit_link_if.consumer   lnk,
    output logic              valid_o,  // a word this cycle, as result_valid_o was
    output logic [DATA_W-1:0] data_o
);
  localparam int SW = $clog2(SLOTS + 1);
  logic live;  // out of reset for a cycle: nothing is advertised before it
  logic [SW-1:0] owed;  // slots freed or advertised, not yet credited
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) begin
      live <= 1'b0;
      owed <= SW'(SLOTS);
    end else begin
      live <= 1'b1;
      owed <= owed + SW'(lnk.put) - SW'(lnk.credit != '0);
    end
  assign lnk.credit = live && (owed != '0);
  assign valid_o    = lnk.put;
  assign data_o     = lnk.data;
endmodule
