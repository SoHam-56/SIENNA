`timescale 1ns / 1ps

// Maxpool_2D in int8: two's-complement order in the streaming window and the padded batch path; IS_FP32 stays 1, so EXP_W = 0 must decide.
module TB_maxpool_int8;
  localparam int W = 8;
  logic clk = 0, rst_n = 0;
  logic start = 0, valid_in = 0, done, out_valid;
  logic [W-1:0] data_in = '0, out_data;
  logic start_b = 0, valid_b = 0, done_b, out_valid_b;
  logic [W-1:0] data_b = '0, out_b;
  always #5 clk = ~clk;

  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(0), .MAN_W(7)) dut (
      .clk(clk), .rst_n(rst_n), .start(start), .done(done), .data_in(data_in), .valid_in(valid_in),
      .out_data(out_data), .out_valid(out_valid));
  // 3 x 3 input, 2 x 2 windows, stride 2, padding 1: every window holds padded positions.
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(3), .IN_COLS(3), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(1), .IS_FP32(1), .EXP_W(0), .MAN_W(7)) dut_b (
      .clk(clk), .rst_n(rst_n), .start(start_b), .done(done_b), .data_in(data_b), .valid_in(valid_b),
      .out_data(out_b), .out_valid(out_valid_b));

  int errs = 0;
  task automatic window(input logic [W-1:0] v0, v1, v2, v3, input logic [W-1:0] want);
    logic [W-1:0] v[4] = '{v0, v1, v2, v3};
    logic [W-1:0] got = 'x;
    @(negedge clk) start = 1;
    @(negedge clk);
    for (int i = 0; i < 4; i++) begin
      valid_in = 1;
      data_in = v[i];
      @(posedge clk);
      if (out_valid) got = out_data;
      @(negedge clk);
    end
    valid_in = 0;
    repeat (2) begin
      @(posedge clk);
      if (out_valid) got = out_data;
    end
    @(negedge clk) start = 0;
    @(negedge clk);
    if (got !== want) begin
      errs++;
      $display("[FAIL] max(%0d %0d %0d %0d) = %0d, want %0d", $signed(v0), $signed(v1), $signed(v2), $signed(v3),
               $signed(got), $signed(want));
    end
  endtask

  // v holds the 3 x 3 input row-major, v[8] first; want holds the 2 x 2 output, want[3] first.
  task automatic batch(input logic [9*W-1:0] v, input logic [4*W-1:0] want);
    logic [W-1:0] got[$];
    int waited = 0;
    @(negedge clk) start_b = 1;
    @(negedge clk);
    for (int i = 0; i < 9; i++) begin
      valid_b = 1;
      data_b = v[i*W+:W];
      @(negedge clk);
    end
    valid_b = 0;
    while (got.size() < 4 && waited < 100) begin
      @(posedge clk);
      if (out_valid_b) got.push_back(out_b);
      waited++;
    end
    @(negedge clk) start_b = 0;
    repeat (2) @(negedge clk);
    for (int i = 0; i < 4; i++)
      if (i >= got.size() || got[i] !== want[i*W+:W]) begin
        errs++;
        $display("[FAIL] batch output %0d = %0d, want %0d", i, (i < got.size()) ? $signed(got[i]) : 999,
                 $signed(want[i*W+:W]));
      end
  endtask

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    window(8'hFF, 8'hFE, 8'h80, 8'h81, 8'hFF);  // all negative: -1 is the largest
    window(8'h05, 8'hFB, 8'h7F, 8'h00, 8'h7F);  // mixed signs: 127
    window(8'h80, 8'h80, 8'h80, 8'h81, 8'h81);  // -128 three times loses to -127
    window(8'h80, 8'h80, 8'h80, 8'h80, 8'h80);  // all -128: the floor itself
    window(8'h00, 8'hFF, 8'h01, 8'h80, 8'h01);  // 1 over 0 and the negatives
    // [[-100, -3, -128], [-7, -1, -2], [-128, -50, -128]]: windows see {-100}, {-3, -128}, {-7, -128}, {-1, -2, -50, -128}
    batch({8'h80, 8'hCE, 8'h80, 8'hFE, 8'hFF, 8'hF9, 8'h80, 8'hFD, 8'h9C},
          {8'hFF, 8'hF9, 8'hFD, 8'h9C});
    $display("TB_maxpool_int8: %0d errors", errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
