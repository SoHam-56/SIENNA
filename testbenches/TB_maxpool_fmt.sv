`timescale 1ns / 1ps

// Maxpool_2D as sienna_top uses it (one 2x2 window per start), in bf16: sign-magnitude order, ties between +0 and -0 keep the first.
module TB_maxpool_fmt;
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  logic clk = 0, rst_n = 0, start = 0, valid_in = 0, done, out_valid;
  logic [W-1:0] data_in = '0, out_data;
  always #5 clk = ~clk;
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(EXP_W), .MAN_W(MAN_W)) dut (
      .clk(clk), .rst_n(rst_n), .start(start), .done(done), .data_in(data_in), .valid_in(valid_in),
      .out_data(out_data), .out_valid(out_valid));

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
      $display("[FAIL] max(%h %h %h %h) = %h, want %h", v0, v1, v2, v3, got, want);
    end
  endtask

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    window(16'hBF80, 16'hC000, 16'hBF00, 16'hC040, 16'hBF00);  // all negative: -0.5 is the largest
    window(16'h3F80, 16'hBF80, 16'h4000, 16'h3F00, 16'h4000);  // mixed signs
    window(16'h8000, 16'h0000, 16'hBF80, 16'hBF80, 16'h8000);  // -0 first, +0 ties it: -0 stays
    window(16'h0000, 16'h8000, 16'hBF80, 16'hBF80, 16'h0000);  // +0 first: +0 stays
    window(16'hFF80, 16'hFF80, 16'hFF80, 16'hC2C8, 16'hC2C8);  // -inf inputs lose to -100
    window(16'h7F7F, 16'h0001, 16'h0080, 16'h3F80, 16'h7F7F);  // largest finite wins
    $display("TB_maxpool_fmt: %0d errors", errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
