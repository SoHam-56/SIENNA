`timescale 1ns / 1ps

// dropout in bf16, training mode, on credit links with checkers bound and 0% then 30% random stalls: every output is x * 2 or a signed zero.
module TB_dropout_fmt #(
    parameter int FAULT = 0  // 6: the input link's credit two bits wide against the output's one
);
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  localparam int SLOTS = 4;  // the TB's output slots: beats in flight through the multiplier never exceed them
  logic clk = 0, rst_n = 0, reseed = 0;
  always #5 clk = ~clk;
  credit_link_if #(.DATA_W(W), .CRW((FAULT == 6) ? 2 : 1)) in_l ();
  credit_link_if #(.DATA_W(W), .CRW(1)) out_l ();
  dropout #(.EXP_W(EXP_W), .MAN_W(MAN_W), .LFSR_WIDTH(32)) dut (
      .clk(clk), .rst_n(rst_n), .in(in_l), .out(out_l), .training_mode(1'b1), .reseed_i(reseed), .seed_i(32'h2ACE002A),
      .zero_point_i('0));

  logic drained = 0;
  logic [2:0] in_cnt;
  credit_counter #(.MAX(SLOTS), .CRW(1)) in_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(in_l.put), .credit_i(in_l.credit[0]),
                                               .has_credit_o(), .count_o(in_cnt));
  credit_link_checker #(.SLOTS(SLOTS)) chk_in (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(in_l));
  credit_link_checker #(.SLOTS(SLOTS)) chk_out (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(out_l));

  logic [W-1:0] q[$];
  int errs = 0, n = 0, kept = 0, owed, pct = 0;
  always @(negedge clk) begin
    if (!rst_n) begin
      owed = SLOTS;
      out_l.credit = 0;
    end else begin
      // A beat's slot is credited from the next cycle on, never with the put that frees it.
      out_l.credit = 0;
      if (owed > 0 && !(pct > 0 && $urandom_range(99) < pct)) begin
        out_l.credit = 1;
        owed--;
      end
      if (out_l.put) begin
        logic [W-1:0] x = q.pop_front();
        logic [W-1:0] dbl = {x[W-1], x[W-2:MAN_W] + EXP_W'(1), x[MAN_W-1:0]};  // x * 2 for these normal inputs
        n++;
        owed++;
        if (out_l.data === dbl) kept++;
        else if (out_l.data !== {x[W-1], {(W - 1) {1'b0}}}) begin
          errs++;
          $display("[FAIL] %h -> %h, neither %h nor a signed zero", x, out_l.data, dbl);
        end
      end
    end
  end

  initial begin
    in_l.put  = 0;
    in_l.data = '0;
  end

  task automatic run_pass(input string name, input int stall);
    int n0 = n, k0 = kept, waited = 0;
    pct = stall;
    for (int i = 0; i < 64; i++) begin
      @(negedge clk);
      while (in_cnt == 0 || (pct > 0 && $urandom_range(99) < pct)) begin
        in_l.put = 0;
        @(negedge clk);
      end
      in_l.put  = 1;
      in_l.data = {i[0], EXP_W'(120 + i % 16), MAN_W'(i * 37)};
      q.push_back(in_l.data);
    end
    @(negedge clk) in_l.put = 0;
    while (n - n0 < 64 && waited < 500) begin
      @(negedge clk);
      waited++;
    end
    pct = 0;
    repeat (20) @(negedge clk);
    drained = 1;
    @(negedge clk) drained = 0;
    if (n - n0 != 64 || kept - k0 == 0 || kept - k0 == 64) begin
      errs++;
      $display("[FAIL] %s: %0d outputs, %0d kept", name, n - n0, kept - k0);
    end
    $display("%s: %0d outputs, %0d kept", name, n - n0, kept - k0);
  endtask

  initial begin
    repeat (2) @(posedge clk);
    @(negedge clk) rst_n = 1;
    @(negedge clk) reseed = 1;
    @(negedge clk) reseed = 0;
    repeat (8) @(negedge clk);
    run_pass("no stalls", 0);
    run_pass("30% stalls", 30);
    $display("TB_dropout_fmt: %0d outputs, %0d kept, %0d errors", n, kept, errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
