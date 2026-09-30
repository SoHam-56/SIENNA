`timescale 1ns / 1ps

// dropout in int8 (D-5): inference passes every beat; training keeps a beat or drops it to zero_point_i, on the LFSR word replayed here.
module TB_dropout_int8;
  localparam int W = 8;
  localparam logic [31:0] SEED = 32'h2ACE002A;
  localparam logic [31:0] THR = 32'((64'hFFFFFFFF * 64'd50) / 64'd100);  // dropout.sv's threshold at 50%
  logic clk = 0, rst_n = 0, in_valid = 0, reseed = 0, training = 0, valid_out;
  logic [W-1:0] data_in = '0, data_out;
  logic [W-1:0] zp = 8'hF3;  // zero point -13
  always #5 clk = ~clk;

  dropout #(.EXP_W(0), .MAN_W(7), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .training_mode(training), .data_in(data_in), .reseed_i(reseed),
      .seed_i(SEED), .zero_point_i(zp), .data_out(data_out), .valid_out(valid_out));

  function automatic logic [31:0] lfsr_next(input logic [31:0] s);
    for (int i = 0; i < 32; i++) s = {s[30:0], s[31] ^ s[21] ^ s[1] ^ s[0]};
    return s;
  endfunction

  int errs = 0, kept = 0;
  initial begin
    logic [31:0] s;
    logic [W-1:0] want;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    s = SEED;
    for (int i = 0; i < 96; i++) begin
      @(negedge clk);
      training = (i >= 32);  // 32 inference beats, then 64 training beats
      in_valid = 1;
      data_in = W'(i * 37 - 100);
      s = lfsr_next(s);  // the LFSR advances on every valid beat, in inference too
      want = (!training || s >= THR) ? data_in : zp;
      #1;
      if (!valid_out || data_out !== want) begin
        errs++;
        $display("[FAIL] beat %0d (%s): %0d -> %0d valid %0b, want %0d", i, training ? "training" : "inference",
                 $signed(data_in), $signed(data_out), valid_out, $signed(want));
      end
      if (training && s >= THR) kept++;
    end
    @(negedge clk) in_valid = 0;
    if (kept == 0 || kept == 64) begin
      errs++;
      $display("[FAIL] %0d of 64 training beats kept", kept);
    end
    $display("TB_dropout_int8: %0d training beats kept of 64, %0d errors", kept, errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
