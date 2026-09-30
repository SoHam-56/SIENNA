`timescale 1ns / 1ps

// dropout in bf16, training mode: every output is x * 2 or a zero with x's sign; the unit's latency is the format's multiplier's.
module TB_dropout_fmt;
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  logic clk = 0, rst_n = 0, in_valid = 0, reseed = 0, valid_out;
  logic [W-1:0] data_in = '0, data_out;
  always #5 clk = ~clk;
  dropout #(.EXP_W(EXP_W), .MAN_W(MAN_W), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .training_mode(1'b1), .data_in(data_in), .reseed_i(reseed),
      .seed_i(32'h2ACE002A), .zero_point_i('0), .data_out(data_out), .valid_out(valid_out));

  logic [W-1:0] q[$];
  int errs = 0, n = 0, kept = 0;
  always @(posedge clk)
    if (valid_out) begin
      logic [W-1:0] x = q.pop_front();
      logic [W-1:0] dbl = {x[W-1], x[W-2:MAN_W] + EXP_W'(1), x[MAN_W-1:0]};  // x * 2 for these normal inputs
      n++;
      if (data_out === dbl) kept++;
      else if (data_out !== {x[W-1], {(W - 1) {1'b0}}}) begin
        errs++;
        $display("[FAIL] %h -> %h, neither %h nor a signed zero", x, data_out, dbl);
      end
    end

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    for (int i = 0; i < 64; i++) begin
      @(negedge clk);
      in_valid = 1;
      data_in = {i[0], EXP_W'(120 + i % 16), MAN_W'(i * 37)};
      q.push_back(data_in);
    end
    @(negedge clk) in_valid = 0;
    repeat (20) @(posedge clk);
    if (n != 64 || kept == 0 || kept == 64) begin
      errs++;
      $display("[FAIL] %0d outputs, %0d kept", n, kept);
    end
    $display("TB_dropout_fmt: %0d outputs, %0d kept, %0d errors", n, kept, errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
