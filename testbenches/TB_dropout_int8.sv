`timescale 1ns / 1ps

// dropout in int8 (D-5) on credit links with checkers bound and 0% then 30% random stalls: inference passes beats, training keeps or drops to zero_point_i.
module TB_dropout_int8;
  localparam int W = 8;
  localparam int SLOTS = 3;
  localparam logic [31:0] SEED = 32'h2ACE002A;
  localparam logic [31:0] THR = 32'((64'hFFFFFFFF * 64'd50) / 64'd100);  // dropout.sv's threshold at 50%
  logic clk = 0, rst_n = 0, reseed = 0, training = 0;
  logic [W-1:0] zp = 8'hF3;  // zero point -13
  always #5 clk = ~clk;

  credit_link_if #(.DATA_W(W), .CRW(1)) in_l ();
  credit_link_if #(.DATA_W(W), .CRW(1)) out_l ();
  dropout #(.EXP_W(0), .MAN_W(7), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in(in_l), .out(out_l), .training_mode(training), .reseed_i(reseed), .seed_i(SEED),
      .zero_point_i(zp));

  logic drained = 0;
  logic [1:0] in_cnt;
  credit_counter #(.MAX(SLOTS), .CRW(1)) in_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(in_l.put), .credit_i(in_l.credit), .has_credit_o(),
                                               .count_o(in_cnt));
  credit_link_checker #(.SLOTS(SLOTS)) chk_in (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(in_l));
  credit_link_checker #(.SLOTS(SLOTS)) chk_out (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(out_l));

  function automatic logic [31:0] lfsr_next(input logic [31:0] s);
    for (int i = 0; i < 32; i++) s = {s[30:0], s[31] ^ s[21] ^ s[1] ^ s[0]};
    return s;
  endfunction

  // Each beat's expected output, in put order; the consumer compares as beats arrive.
  logic [W-1:0] want[$];
  string what[$];
  int errs = 0, kept = 0, owed, pct = 0, got = 0;
  // dropout passes int8 beats through in the cycle they are put, so its output is read at the rising edge, after the falling-edge put.
  always @(posedge clk)
    if (rst_n && out_l.put) begin
      got++;
      owed++;
      if (want.size() == 0 || out_l.data !== want[0]) begin
        errs++;
        $display("[FAIL] beat %0d (%s): %0d, want %0d", got - 1, (what.size() != 0) ? what[0] : "none", $signed(out_l.data),
                 (want.size() != 0) ? $signed(want[0]) : 999);
      end
      if (want.size() != 0) begin
        void'(want.pop_front());
        void'(what.pop_front());
      end
    end
  always @(negedge clk) begin
    if (!rst_n) begin
      owed = SLOTS;
      out_l.credit = 0;
    end else begin
      out_l.credit = 0;
      if (owed > 0 && !(pct > 0 && $urandom_range(99) < pct)) begin
        out_l.credit = 1;
        owed--;
      end
    end
  end

  initial begin
    in_l.put  = 0;
    in_l.data = '0;
  end

  // 32 inference beats, then 64 training beats; the LFSR advances on every beat, in inference too.
  task automatic run_pass(input string name, input int stall);
    logic [31:0] s = SEED;
    int k0 = kept, g0 = got, waited = 0;
    pct = stall;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    for (int i = 0; i < 96; i++) begin
      @(negedge clk);
      while (in_cnt == 0 || (pct > 0 && $urandom_range(99) < pct)) begin
        in_l.put = 0;
        @(negedge clk);
      end
      training  = (i >= 32);
      in_l.put  = 1;
      in_l.data = W'(i * 37 - 100);
      s = lfsr_next(s);
      want.push_back((!training || s >= THR) ? in_l.data : zp);
      what.push_back(training ? "training" : "inference");
      if (training && s >= THR) kept++;
    end
    @(negedge clk) in_l.put = 0;
    while (got - g0 < 96 && waited < 500) begin
      @(negedge clk);
      waited++;
    end
    pct = 0;
    repeat (10) @(negedge clk);
    drained = 1;
    @(negedge clk) drained = 0;
    if (got - g0 != 96 || kept - k0 == 0 || kept - k0 == 64) begin
      errs++;
      $display("[FAIL] %s: %0d outputs of 96, %0d of 64 training beats kept", name, got - g0, kept - k0);
    end
    $display("%s: %0d outputs, %0d of 64 training beats kept", name, got - g0, kept - k0);
  endtask

  initial begin
    repeat (2) @(posedge clk);
    @(negedge clk) rst_n = 1;
    repeat (6) @(negedge clk);
    run_pass("no stalls", 0);
    run_pass("30% stalls", 30);
    $display("TB_dropout_int8: %0d training beats kept of 128, %0d errors", kept, errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
