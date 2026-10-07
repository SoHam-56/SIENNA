`timescale 1ns / 1ps

// Maxpool_2D in int8, streaming window and padded batch path, on credit links with checkers bound and 0% then 30% stalls; EXP_W = 0 must decide the order.
module TB_maxpool_int8 #(
    parameter int FAULT = 0  // 2: elements put with no credit granted (a_mp_in_credit); 5: the batch path's OUT_MAX below its 4 results
);
  localparam int W = 8;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  // Streaming: one 2x2 window per result, one output slot. Batch: 3 x 3 input, 2 x 2 windows, stride 2, padding 1, four results.
  credit_link_if #(.DATA_W(W), .CRW(3)) s_in ();
  credit_link_if #(.DATA_W(W), .CRW(1)) s_out ();
  credit_link_if #(.DATA_W(W), .CRW(4)) b_in ();
  credit_link_if #(.DATA_W(W), .CRW(1)) b_out ();
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(2), .IN_COLS(2), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(0), .IS_FP32(1), .EXP_W(0), .MAN_W(7), .AHEAD(2), .OUT_MAX(1), .OUT_CRW(1), .IN_CRW(3)) dut (
      .clk(clk), .rst_n(rst_n), .in(s_in), .out(s_out));
  Maxpool_2D #(.DATA_WIDTH(W), .IN_ROWS(3), .IN_COLS(3), .SEG_ROWS(2), .SEG_COLS(2), .STRIDE_ROWS(2), .STRIDE_COLS(2),
               .PADDING(1), .IS_FP32(1), .EXP_W(0), .MAN_W(7), .OUT_MAX((FAULT == 5) ? 2 : 4), .OUT_CRW(1), .IN_CRW(4)) dut_b (
      .clk(clk), .rst_n(rst_n), .in(b_in), .out(b_out));

  logic drained = 0;
  logic [3:0] s_cnt, b_cnt;
  credit_counter #(.MAX(8), .CRW(3)) s_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(s_in.put), .credit_i(s_in.credit), .has_credit_o(),
                                          .count_o(s_cnt));
  credit_counter #(.MAX(9), .CRW(4)) b_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(b_in.put), .credit_i(b_in.credit), .has_credit_o(),
                                          .count_o(b_cnt));
  // With one output slot the streaming path reserves one window, so 4 input credits are home when drained.
  credit_link_checker #(.SLOTS(4)) chk_s_in (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(s_in));
  credit_link_checker #(.SLOTS(1)) chk_s_out (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(s_out));
  credit_link_checker #(.SLOTS(9)) chk_b_in (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(b_in));
  credit_link_checker #(.SLOTS(4)) chk_b_out (.clk_i(clk), .rstn_i(rst_n), .drained_i(drained), .lnk(b_out));

  // Consumers: one slot (streaming) and four (batch); a result frees its slot at once; credits change on the falling edge.
  int s_owed, b_owed, pct = 0;
  logic [W-1:0] s_got[$], b_got[$];
  always @(negedge clk) begin
    if (!rst_n) begin
      s_owed = 1;
      b_owed = 4;
      s_out.credit = 0;
      b_out.credit = 0;
    end else begin
      // A result's slot is credited from the next cycle on, never with the put that frees it.
      s_out.credit = 0;
      b_out.credit = 0;
      if (s_owed > 0 && !(pct > 0 && $urandom_range(99) < pct)) begin
        s_out.credit = 1;
        s_owed--;
      end
      if (b_owed > 0 && !(pct > 0 && $urandom_range(99) < pct)) begin
        b_out.credit = 1;
        b_owed--;
      end
      if (s_out.put) begin
        s_got.push_back(s_out.data);
        s_owed++;
      end
      if (b_out.put) begin
        b_got.push_back(b_out.data);
        b_owed++;
      end
    end
  end

  initial begin
    s_in.put  = 0;
    s_in.data = '0;
    b_in.put  = 0;
    b_in.data = '0;
  end

  int errs = 0;
  logic [W-1:0] s_ins[$], s_want[$];
  task automatic window(input logic [W-1:0] v0, v1, v2, v3, input logic [W-1:0] w);
    s_ins.push_back(v0);
    s_ins.push_back(v1);
    s_ins.push_back(v2);
    s_ins.push_back(v3);
    s_want.push_back(w);
  endtask

  task automatic s_produce();
    foreach (s_ins[i]) begin
      @(negedge clk);
      while (s_cnt == 0 || (pct > 0 && $urandom_range(99) < pct)) begin
        s_in.put = 0;
        @(negedge clk);
      end
      s_in.put  = 1;
      s_in.data = s_ins[i];
    end
    @(negedge clk) s_in.put = 0;
  endtask

  // v holds the 3 x 3 input row-major, v[8] first; want holds the 2 x 2 output, want[3] first.
  task automatic batch(input logic [9*W-1:0] v);
    for (int i = 0; i < 9; i++) begin
      @(negedge clk);
      while (b_cnt == 0 || (pct > 0 && $urandom_range(99) < pct)) begin
        b_in.put = 0;
        @(negedge clk);
      end
      b_in.put  = 1;
      b_in.data = v[i*W+:W];
    end
    @(negedge clk) b_in.put = 0;
  endtask

  task automatic run_pass(input string name, input int stall);
    int waited = 0;
    logic [4*W-1:0] bw = {8'hFF, 8'hF9, 8'hFD, 8'h9C};
    s_got.delete();
    b_got.delete();
    pct = stall;
    // [[-100, -3, -128], [-7, -1, -2], [-128, -50, -128]]: windows see {-100}, {-3, -128}, {-7, -128}, {-1, -2, -50, -128}
    fork
      s_produce();
      batch({8'h80, 8'hCE, 8'h80, 8'hFE, 8'hFF, 8'hF9, 8'h80, 8'hFD, 8'h9C});
    join
    while ((s_got.size() < s_want.size() || b_got.size() < 4) && waited < 1000) begin
      @(negedge clk);
      waited++;
    end
    pct = 0;
    foreach (s_want[i])
      if (i >= s_got.size() || s_got[i] !== s_want[i]) begin
        errs++;
        $display("[FAIL] %s window %0d: max(%0d %0d %0d %0d) = %0d, want %0d", name, i, $signed(s_ins[4*i]), $signed(s_ins[4*i+1]),
                 $signed(s_ins[4*i+2]), $signed(s_ins[4*i+3]), (i < s_got.size()) ? $signed(s_got[i]) : 999, $signed(s_want[i]));
      end
    for (int i = 0; i < 4; i++)
      if (i >= b_got.size() || b_got[i] !== bw[i*W+:W]) begin
        errs++;
        $display("[FAIL] %s batch output %0d = %0d, want %0d", name, i, (i < b_got.size()) ? $signed(b_got[i]) : 999, $signed(bw[i*W+:W]));
      end
    if (s_got.size() != s_want.size() || b_got.size() != 4) begin
      errs++;
      $display("[FAIL] %s: %0d window and %0d batch results, want %0d and 4", name, s_got.size(), b_got.size(), s_want.size());
    end
    repeat (40) @(negedge clk);
    drained = 1;
    @(negedge clk) drained = 0;
    $display("%s: %0d windows and one batch, %0d and %0d results", name, s_want.size(), s_got.size(), b_got.size());
  endtask

  initial begin
    window(8'hFF, 8'hFE, 8'h80, 8'h81, 8'hFF);  // all negative: -1 is the largest
    window(8'h05, 8'hFB, 8'h7F, 8'h00, 8'h7F);  // mixed signs: 127
    window(8'h80, 8'h80, 8'h80, 8'h81, 8'h81);  // -128 three times loses to -127
    window(8'h80, 8'h80, 8'h80, 8'h80, 8'h80);  // all -128: the floor itself
    window(8'h00, 8'hFF, 8'h01, 8'h80, 8'h01);  // 1 over 0 and the negatives
    repeat (2) @(posedge clk);
    @(negedge clk) rst_n = 1;
    if (FAULT == 2) begin  // four elements on the cycles after reset, before any input credit is granted
      for (int i = 0; i < 4; i++) begin
        @(negedge clk);
        s_in.put  = 1;
        s_in.data = W'(i);
      end
      @(negedge clk) s_in.put = 0;
      $display("[FAULT 2] four elements put with no credit granted");
      $finish;
    end
    repeat (10) @(negedge clk);
    run_pass("no stalls", 0);
    run_pass("30% stalls", 30);
    $display("TB_maxpool_int8: %0d errors", errs);
    if (errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
