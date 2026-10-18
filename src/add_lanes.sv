`timescale 1ns / 100ps

// int8 residual ADD: one int8QuantizedAdder per lane on an add set's requantized beat and its hold_mem beat; the beat's tag rides beside it for qadd_lat() cycles; busy_o while a beat is inside.
module add_lanes #(
    parameter int NUM_LANES = 32,
    parameter int TAG_W     = 1  // the beat's destination as sienna_top packs it: {bypass, bank, packed, element index}
) (
    input  logic                       clk_i,
    input  logic                       rstn_i,
    input  logic                       valid_i,
    input  logic [NUM_LANES-1:0][7:0]  q1_i,     // hold_mem's beat: the conv's requantized codes
    input  logic [NUM_LANES-1:0][7:0]  q2_i,     // the add set's beat: the residual's codes through the identity requantize
    input  logic [TAG_W-1:0]           tag_i,
    input  sienna_add_pkg::add_words_t words_i,  // the set's ADD words, every lane
    output logic                       valid_o,
    output logic [NUM_LANES-1:0][7:0]  result_o,
    output logic [TAG_W-1:0]           tag_o,
    output logic                       busy_o    // a beat entered in one of the last qadd_lat() cycles
);
  localparam int QAL = sienna_fmt_pkg::qadd_lat();
  logic [NUM_LANES-1:0] done;
  logic [TAG_W-1:0] tag_q[QAL];
  logic [QAL-1:0] vq;  // valid_i delayed 1..QAL cycles, beside the tag line

  if (QAL < 1) begin : G_BAD_LAT
    $fatal(1, "add_lanes: qadd_lat() is %0d; the tag line needs at least one stage", QAL);
  end

  for (genvar k = 0; k < NUM_LANES; k++) begin : LANE
    int8QuantizedAdder u (
        .clk_i(clk_i), .rstn_i(rstn_i), .valid_i(valid_i), .q1_i(q1_i[k]), .q2_i(q2_i[k]), .zp1_i(words_i.zp1), .zp2_i(words_i.zp2),
        .m1_i(words_i.m1), .m2_i(words_i.m2), .shift_i(words_i.shift), .zpo_i(words_i.zpo), .act_min_i(words_i.amin),
        .act_max_i(words_i.amax), .result_o(result_o[k]), .done_o(done[k]));
  end

  always_ff @(posedge clk_i) begin  // no reset: read only beside valid_o (D-8)
    tag_q[0] <= tag_i;
    for (int i = 1; i < QAL; i++) tag_q[i] <= tag_q[i-1];
  end
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) vq <= '0;
    else vq <= QAL'({vq, valid_i});  // elaborates at any qadd_lat() >= 1
  assign tag_o   = tag_q[QAL-1];
  assign valid_o = done[0];  // every lane takes the same beats
  assign busy_o  = |vq;

`ifndef SYNTHESIS
  a_lanes_in_step: assert property (@(posedge clk_i) disable iff (!rstn_i) done == {NUM_LANES{done[0]}})
    else $error("add_lanes: the lanes' int8QuantizedAdder units are out of step");
  a_tag_lat: assert property (@(posedge clk_i) disable iff (!rstn_i) done[0] == vq[QAL-1])
    else $error("add_lanes: int8QuantizedAdder's latency is not qadd_lat() = %0d, so tag_o is off its beat", QAL);
`endif

endmodule
