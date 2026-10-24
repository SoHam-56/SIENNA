`timescale 1ns / 100ps

// The lane feed's requantize: one int8Requantizer per lane where the build requantizes (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE), else the words pass unchanged in the same cycle.
// Lane k's channel is (k*PER_LANE + b) % N, or k % N for a packed set; zero point and clamp per lane.
module requant_lanes #(
    parameter int NUM_LANES = 32,
    parameter int N         = 16,
    parameter int PER_LANE  = N * N / NUM_LANES
) (
    input  logic                                                     clk_i,
    input  logic                                                     rstn_i,
    input  logic                                                     clear_i,  // no set in the stage: the next beat is beat 0
    input  logic                                                     valid_i,  // one wide-read beat
    input  logic [NUM_LANES-1:0][sienna_fmt_pkg::RESULT_DATA_WIDTH-1:0] acc_i,  // the mesh's result words: int32 sums where it requantizes
    input  sienna_fmt_pkg::requant_scale_t [N-1:0]                   scale_i,  // per output channel: the scale's mantissa and right shift
    input  logic                                                     packed_i, // the stage's set is packed: lane k reads column k % N at every beat
    input  sienna_fmt_pkg::lane_entry_t [NUM_LANES-1:0]              entry_i,  // per lane: its entry's output zero point and clamp (the int8 range, or the fused ReLU / ReLU6)
    output logic                                                     valid_o,  // the beat, REQUANT_LATENCY cycles later
    output logic [NUM_LANES-1:0][sienna_fmt_pkg::DATA_WIDTH-1:0]     result_o
);
  if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE) begin : G_REQUANT
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
      int8Requantizer rq (
          .clk_i    (clk_i),
          .rstn_i   (rstn_i),
          .valid_i  (valid_i),
          .acc_i    (acc_i[k]),
          .mult_i   (scale_i[ch].mantissa),
          .shift_i  (scale_i[ch].shift),
          .zp_i     (entry_i[k].zero_point),
          .act_min_i(entry_i[k].act_min),
          .act_max_i(entry_i[k].act_max),
          .result_o (result_o[k]),
          .done_o   (done[k])
      );
    end
    assign valid_o = done[0];  // every lane takes the same beats
    wire unused_lane_words = ^entry_i;  // the entries' lane words are GPNAE's
  end else begin : G_PASS
    if (sienna_fmt_pkg::RESULT_DATA_WIDTH != sienna_fmt_pkg::DATA_WIDTH) begin : G_BAD_WIDTH  // a pass-through cannot narrow
      $fatal(1, "requant_lanes: the mesh gives %0d-bit results, the lanes take %0d-bit words", sienna_fmt_pkg::RESULT_DATA_WIDTH,
             sienna_fmt_pkg::DATA_WIDTH);
    end
    assign valid_o  = valid_i;
    assign result_o = acc_i;
    wire unused_requant_inputs = ^{clk_i, rstn_i, clear_i, scale_i, packed_i, entry_i};  // only a requantizing build reads them
  end

endmodule
