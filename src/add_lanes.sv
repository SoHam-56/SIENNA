`timescale 1ns / 100ps

// int8 residual ADD: one tfliteAdd per lane on an add set's requantized beat and its hold_mem beat; the beat's tag rides beside it for qadd_lat() cycles.
module add_lanes #(
    parameter int    NUM_LANES = 32,
    parameter int    TAG_W     = 1,  // the beat's destination as sienna_top packs it: {bypass, bank, packed, element index}
    parameter string ROUNDING  = sienna_fmt_pkg::REQ_ROUNDING
) (
    input  logic                      clk_i,
    input  logic                      rstn_i,
    input  logic                      valid_i,
    input  logic [NUM_LANES-1:0][7:0] q1_i,  // hold_mem's beat: the conv's requantized codes
    input  logic [NUM_LANES-1:0][7:0] q2_i,  // the add set's beat: the residual's codes through the identity requantize
    input  logic [TAG_W-1:0]          tag_i,
    input  logic [7:0]                zp1_i, zp2_i, s1_i, s2_i, so_i, zpo_i, min_i, max_i,  // the set's ADD words, every lane
    input  logic [31:0]               m1_i, m2_i, mo_i,
    output logic                      valid_o,
    output logic [NUM_LANES-1:0][7:0] result_o,
    output logic [TAG_W-1:0]          tag_o
);
  localparam int QAL = sienna_fmt_pkg::qadd_lat();
  logic [NUM_LANES-1:0] done;
  logic [TAG_W-1:0] tag_q[QAL];

  if (QAL < 1) begin : G_BAD_LAT
    $fatal(1, "add_lanes: qadd_lat() is %0d; the tag line needs at least one stage", QAL);
  end

  for (genvar k = 0; k < NUM_LANES; k++) begin : LANE
    tfliteAdd #(.ROUNDING(ROUNDING)) u (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(valid_i), .q1_i(q1_i[k]), .q2_i(q2_i[k]), .zp1_i(zp1_i), .zp2_i(zp2_i), .m1_i(m1_i),
        .m2_i(m2_i), .mo_i(mo_i), .s1_i(s1_i), .s2_i(s2_i), .so_i(so_i), .zpo_i(zpo_i), .min_i(min_i), .max_i(max_i),
        .result_o(result_o[k]), .done_o(done[k]));
  end

  always_ff @(posedge clk_i) begin  // no reset: read only beside valid_o (D-8)
    tag_q[0] <= tag_i;
    for (int i = 1; i < QAL; i++) tag_q[i] <= tag_q[i-1];
  end
  assign tag_o   = tag_q[QAL-1];
  assign valid_o = done[0];  // every lane takes the same beats

`ifndef SYNTHESIS
  a_lanes_in_step: assert property (@(posedge clk_i) disable iff (!rstn_i) done == {NUM_LANES{done[0]}})
    else $error("add_lanes: the lanes' tfliteAdd units are out of step");
`endif

endmodule
