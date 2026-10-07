`timescale 1ns / 1ps

// Maxpool_2D behind a 16-word FIFO2, as sienna_top uses it, in bf16 on credit links with checkers bound: sign-magnitude order, +0/-0 ties keep the first.
module TB_maxpool_fmt #(
    parameter int FAULT = 0  // 1: puts with no credit, results withheld (a_fifo_room); 3: FIFO2's input link one bit wider; 4: maxpool's output link
);
  localparam int EXP_W = 8, MAN_W = 7, W = 16;
  localparam int OUT_SLOTS = 2;  // results the TB takes ahead; maxpool reserves one per window it grants
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  credit_link_if #(.DATA_W(W + ((FAULT == 3) ? 1 : 0)), .CRW(1)) in_l ();  // TB -> FIFO2
  credit_link_if #(.DATA_W(W), .CRW(3)) mid ();  // FIFO2 -> maxpool: a window's 4 credits at once
  credit_link_if #(.DATA_W(W + ((FAULT == 4) ? 1 : 0)), .CRW(1)) out_l ();  // maxpool -> TB
  fwft #(.DATA_WIDTH(W), .FIFO_DEPTH(16), .OUT_MAX(8), .OUT_CRW(3)) fifo (.clk_i(clk), .rstn_i(rst_n), .in(in_l), .out(mid), .count_o());
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(EXP_W), .MAN_W(MAN_W), .AHEAD(2), .OUT_MAX(4), .OUT_CRW(1), .IN_CRW(3)) dut (
      .clk(clk), .rst_n(rst_n), .in(mid), .out(out_l));

  // Producer side of FIFO2's link, consumer side of maxpool's; every link signal changes on the falling edge.
  logic [4:0] in_cnt;
  logic drained = 0;
  credit_counter #(.MAX(16), .CRW(1)) in_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(in_l.put), .credit_i(in_l.credit), .has_credit_o(),
                                             .count_o(in_cnt));
  credit_link_checker #(.SLOTS(16)) chk_in (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(in_l));
  credit_link_checker #(.SLOTS(8)) chk_mid (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(mid));
  credit_link_checker #(.SLOTS(OUT_SLOTS)) chk_out (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(out_l));

  int owed, pct = 0;
  bit hold = 0;
  logic [W-1:0] got[$];
  initial begin
    in_l.put = 0;
    in_l.data = '0;
  end
  always @(negedge clk) begin
    if (!rst_n) begin
      owed = OUT_SLOTS;
      out_l.credit = 0;
    end else begin
      // A result's slot is credited from the next cycle on, never with the put that frees it.
      out_l.credit = 0;
      if (owed > 0 && !hold && !(pct > 0 && $urandom_range(99) < pct)) begin
        out_l.credit = 1;
        owed--;
      end
      if (out_l.put) begin
        got.push_back(W'(out_l.data));
        owed++;
      end
    end
  end

  // Puts each word once a credit is held, skipping cycles at random with pct.
  task automatic produce(input logic [W-1:0] v[$]);
    foreach (v[i]) begin
      @(negedge clk);
      while (in_cnt == 0 || (pct > 0 && $urandom_range(99) < pct)) begin
        in_l.put = 0;
        @(negedge clk);
      end
      in_l.put  = 1;
      in_l.data = v[i];
    end
    @(negedge clk) in_l.put = 0;
  endtask

  logic [W-1:0] ins[$], want[$];
  task automatic window(input logic [W-1:0] v0, v1, v2, v3, input logic [W-1:0] w);
    ins.push_back(v0);
    ins.push_back(v1);
    ins.push_back(v2);
    ins.push_back(v3);
    want.push_back(w);
  endtask

  int errs = 0;
  task automatic run_pass(input string name, input int stall, input int hold_cyc);
    int waited = 0;
    got.delete();
    pct = stall;
    if (hold_cyc > 0) hold = 1;
    fork
      produce(ins);
      if (hold_cyc > 0) begin
        repeat (hold_cyc) @(negedge clk);
        if (got.size() > OUT_SLOTS) begin  // only the credits maxpool already held
          errs++;
          $display("[FAIL] %s: %0d results with every credit withheld, %0d credits held", name, got.size(), OUT_SLOTS);
        end else $display("%s: %0d results while withheld, at most the %0d credits maxpool held", name, got.size(), OUT_SLOTS);
        hold = 0;
      end
    join
    while (got.size() < want.size() && waited < 1000) begin
      @(negedge clk);
      waited++;
    end
    pct = 0;
    foreach (want[i])
      if (i >= got.size() || got[i] !== want[i]) begin
        errs++;
        $display("[FAIL] %s window %0d: max(%h %h %h %h) = %h, want %h", name, i, ins[4*i], ins[4*i+1], ins[4*i+2], ins[4*i+3],
                 (i < got.size()) ? got[i] : 'x, want[i]);
      end
    if (got.size() != want.size()) begin
      errs++;
      $display("[FAIL] %s: %0d results, want %0d", name, got.size(), want.size());
    end
    // Drained: every link holds all its slots again, checked by each checker's a_all_back.
    repeat (40) @(negedge clk);
    drained = 1;
    @(negedge clk) drained = 0;
    $display("%s: %0d windows, %0d results", name, want.size(), got.size());
  endtask

  initial begin
    window(16'hBF80, 16'hC000, 16'hBF00, 16'hC040, 16'hBF00);  // all negative: -0.5 is the largest
    window(16'h3F80, 16'hBF80, 16'h4000, 16'h3F00, 16'h4000);  // mixed signs
    window(16'h8000, 16'h0000, 16'hBF80, 16'hBF80, 16'h8000);  // -0 first, +0 ties it: -0 stays
    window(16'h0000, 16'h8000, 16'hBF80, 16'hBF80, 16'h0000);  // +0 first: +0 stays
    window(16'hFF80, 16'hFF80, 16'hFF80, 16'hC2C8, 16'hC2C8);  // -inf inputs lose to -100
    window(16'h7F7F, 16'h0001, 16'h0080, 16'h3F80, 16'h7F7F);  // largest finite wins
    repeat (2) @(posedge clk);
    @(negedge clk) rst_n = 1;
    if (FAULT == 1) begin  // 40 puts, one a cycle, credit or not, while no result credit returns: FIFO2 overflows
      hold = 1;
      for (int i = 0; i < 40; i++) begin
        @(negedge clk);
        in_l.put  = 1;
        in_l.data = W'(i);
      end
      @(negedge clk) in_l.put = 0;
      $display("[FAULT 1] 40 puts with no credit check, results withheld");
      $finish;
    end
    repeat (40) @(negedge clk);
    run_pass("no stalls", 0, 0);
    run_pass("30% stalls", 30, 0);
    run_pass("results withheld 60 cycles", 0, 60);
    $display("TB_maxpool_fmt: %0d errors", errs);
    $display("RESULT: %s", errs == 0 ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
