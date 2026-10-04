`timescale 1ns / 100ps

// int8: one tfliteRequant per lane on the wide read; lane k's channel is (k*PER_LANE + b) % N, or k % N for a packed set; zero point and clamp per lane.
module requant_lanes #(
    parameter int    NUM_LANES = 32,
    parameter int    N         = 16,
    parameter int    PER_LANE  = N * N / NUM_LANES,
    parameter string ROUNDING  = sienna_fmt_pkg::REQ_ROUNDING  // TFLite reference kernels' variant, pinned at G0 (Task 2)
) (
    input  logic                       clk_i,
    input  logic                       rstn_i,
    input  logic                       clear_i,  // no set in the stage: the next beat is beat 0
    input  logic                       valid_i,  // one wide-read beat
    input  logic [NUM_LANES-1:0][31:0] acc_i,    // int32 sums
    input  logic [N-1:0][31:0]         mult_i,   // per output channel, Q0.31
    input  logic [N-1:0][7:0]          shift_i,  // per output channel, signed
    input  logic                       packed_i, // the stage's set is packed: lane k reads column k % N at every beat
    input  logic [NUM_LANES-1:0][7:0]  zp_i,     // output zero point, per lane
    input  logic [NUM_LANES-1:0][7:0]  min_i,    // clamp, signed, per lane: the int8 range, or the fused ReLU / ReLU6
    input  logic [NUM_LANES-1:0][7:0]  max_i,
    output logic                       valid_o,  // the beat, sienna_fmt_pkg::req_lat() cycles later
    output logic [NUM_LANES-1:0][7:0]  result_o
);
  localparam int BW = $clog2(PER_LANE + 1);
  localparam int CW = (N > 1) ? $clog2(N) : 1;
  logic [BW-1:0] beat;
  logic [NUM_LANES-1:0] done;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) beat <= '0;
    else if (clear_i) beat <= '0;
    else if (valid_i) beat <= beat + 1'b1;
  end

  for (genvar k = 0; k < NUM_LANES; k++) begin : LANE
    logic [CW-1:0] ch;
    assign ch = packed_i ? CW'(k % N) : CW'((k * PER_LANE + int'(beat)) % N);
    tfliteRequant #(.ROUNDING(ROUNDING)) rq (
        .clk_i    (clk_i),
        .rstn_i   (rstn_i),
        .valid_i  (valid_i),
        .acc_i    (acc_i[k]),
        .mult_i   (mult_i[ch]),
        .shift_i  (shift_i[ch]),
        .zp_i     (zp_i[k]),
        .act_min_i(min_i[k]),
        .act_max_i(max_i[k]),
        .result_o (result_o[k]),
        .done_o   (done[k])
    );
  end
  assign valid_o = done[0];  // every lane takes the same beats

endmodule
