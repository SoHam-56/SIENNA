`timescale 1ns / 100ps

import test_config_pkg::*;

`include "tb_l9_sink.svh"

// Streams NUM_SETS distinct sets through sienna_multi, checks every set against its golden output, and measures throughput.
// The host is the producer of every copy's L0 link and of both cache regions (L2), and the consumer of every copy's L9 links.
// +cached: sets go in fills of each region's cache (COPIES to COPIES+2 sets, B from the cache, each copy's last set of a fill marked), an uncached set after every second fill.
// +out_slots=S (1..64) and +out_stall_pct=P on L9; +fault=1 a fill of one set, 2 a copy's first set of a fill marked, 3 a put to a copy whose turn it is not,
// 4 the last set alone in a fill of one, then the host stops.
module TB_sienna_multi #(
    parameter int COPIES     = 2,
    parameter int COLLAPSE_K = 1
);
  localparam ADDR_LINES = $clog2(FIFO_DEPTH);
  localparam int TIMEOUT_CYCLES = 400_000;
  localparam int WAIT_CYCLES = 20_000;  // a credit this long in coming is a hang
  localparam real REL_TOL = 0.01;
  localparam int WC_TILES = 128;
  localparam int HALF = WC_TILES / 2;  // tiles per cache region
  localparam int WCTW = $clog2(WC_TILES);
  localparam int WCAW = $clog2(WC_TILES * N * N);
  localparam int PACK_ENTRIES = 8;
  localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX
  localparam int SW = $clog2(COPIES + 1);
  `include "sienna_set_side.svh"

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;

  logic [N-1:0][ACC_W-1:0] bias_i = '0;
  logic [N-1:0][31:0] req_mult_i = '0;  // int8 (D-2)
  logic [N-1:0][7:0] req_shift_i = '0;
  logic [7:0] req_zp_i = '0, req_min_i = '0, req_max_i = '0, gp_shout_i = '0, gp_zout_i = '0;
  logic [15:0] gp_mx_i = '0;
  logic [4:0] gp_shx_i = '0;
  logic [31:0] gp_mout_i = '0;
  logic wc_write_enable_i = 0;
  logic [WCAW-1:0] wc_write_addr_i = '0;
  logic north_write_enable_i = 0, west_write_enable_i = 0, north_write_reset_i = 0, west_write_reset_i = 0;
  logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i = '0, west_write_data_i = '0;  // operands in the package's format
  logic [SW-1:0] copy_sel_o;
  logic [COPIES-1:0] pipeline_complete_o;
  logic [COPIES-1:0][$clog2(SETS_IN_FLIGHT+1)-1:0] done_set_id_o;
  logic drain_chk = 1'b0;

  // ── L0 per copy: a counter and a checker each; link signals change on the falling edge ──
  credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) host[COPIES] ();
  set_side_t side = '0;
  logic [COPIES-1:0] hput = '0;
  logic [1:0] hcnt[COPIES];
  for (genvar c = 0; c < COPIES; c++) begin : G_HOST
    assign host[c].put = hput[c];
    assign host[c].data = side;
    credit_counter #(.MAX(2), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host[c].put), .credit_i(host[c].credit),
                                           .has_credit_o(), .count_o(hcnt[c]));
    credit_link_checker #(.SLOTS(2)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drain_chk && hcnt[c] == 2'd2), .lnk(host[c]));
  end

  // ── L2: this host opens each fill of a region with one put while it holds that region's credit ──
  credit_link_if #(.DATA_W(1), .CRW(1)) wc_region[2] ();
  logic [1:0] wput = '0;
  logic wcnt[2];
  for (genvar r = 0; r < 2; r++) begin : G_WC
    assign wc_region[r].put = wput[r];
    assign wc_region[r].data = 1'b0;
    credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(wc_region[r].put), .credit_i(wc_region[r].credit),
                                           .has_credit_o(), .count_o(wcnt[r]));
    credit_link_checker #(.SLOTS(1)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drain_chk && wcnt[r]), .lnk(wc_region[r]));
  end

  // ── L9 per copy and lane: copy c's lane l is out_lnk[c*NUM_LANES + l] ──
  localparam int NL = COPIES * NUM_LANES;
  credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) out_lnk[NL] ();
  int out_slots = OUT_CAP, out_stall_pct = 0;
  logic [NL-1:0] ov, lane_home;
  logic [DATA_WIDTH-1:0] od[NL];
  for (genvar i = 0; i < NL; i++) begin : G_OUT
    tb_l9_sink #(.DATA_W(DATA_WIDTH), .MAX_SLOTS(OUT_CAP)) snk (.clk_i(clk_i), .rstn_i(rstn_i), .slots_i(out_slots),
                                                                .stall_pct_i(out_stall_pct), .hold_i(1'b0), .drain_i(drain_chk),
                                                                .lnk(out_lnk[i]), .valid_o(ov[i]), .data_o(od[i]), .held_o(),
                                                                .home_o(lane_home[i]));
  end

  sienna_multi #(
      .COPIES           (COPIES),
      .NUM_LANES        (NUM_LANES),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (HOST_WORDS),
      .COLLAPSE_K       (COLLAPSE_K),
      .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
      .DATA_WIDTH       (DATA_WIDTH),
      .EXP_W            (EXP_W),
      .MAN_W            (MAN_W),
      .SRAM_DEPTH       (SRAM_DEPTH),
      .CONTROL_WIDTH    (CONTROL_WIDTH),
      .IN_ROWS          (IN_ROWS),
      .IN_COLS          (IN_COLS),
      .POOL_H           (POOL_H),
      .POOL_W           (POOL_W),
      .STRIDE_ROWS      (STRIDE_ROWS),
      .STRIDE_COLS      (STRIDE_COLS),
      .PADDING          (PADDING),
      .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
      .LFSR_WIDTH       (LFSR_WIDTH),
      .FIFO_DEPTH       (FIFO_DEPTH)
  ) dut (
      .clk_i               (clk_i),
      .rstn_i              (rstn_i),
      .host                (host),
      .bias_i              (bias_i),
      .wc_region           (wc_region),
      .wc_write_enable_i   (wc_write_enable_i),
      .wc_write_addr_i     (wc_write_addr_i),
      .north_write_enable_i(north_write_enable_i),
      .north_write_data_i  (north_write_data_i),
      .north_write_reset_i (north_write_reset_i),
      .west_write_enable_i (west_write_enable_i),
      .west_write_data_i   (west_write_data_i),
      .west_write_reset_i  (west_write_reset_i),
      .drained_i           (drain_chk),
      .copy_sel_o          (copy_sel_o),
      .out                 (out_lnk),
      .pipeline_complete_o (pipeline_complete_o),
      .done_set_id_o       (done_set_id_o)
  );

  function automatic real f32(input logic [31:0] b);
    int e;
    real m, v;
    e = int'(b[30:23]);
    m = real'(longint'(b[22:0])) / 8388608.0;
    if (e == 255) v = 1.0e38;
    else if (e == 0) v = 0.0;
    else v = (1.0 + m) * (2.0 ** (e - 127));
    return b[31] ? -v : v;
  endfunction

  // Within REL_TOL, or within the golden model's fp32 error bound for this output.
  function automatic bit close(input logic [31:0] e, input logic [31:0] a, input logic [31:0] bnd);
    real er = f32(e), ar = f32(a), br = f32(bnd), d;
    if (EXACT_GOLDEN) return e === a;  // narrow formats: bit-exact golden
    d = (er > ar) ? er - ar : ar - er;
    if (br > 0.0 && d <= br) return 1;
    if (er == 0.0) return ar == 0.0;
    return d <= REL_TOL * ((er > 0.0) ? er : -er);
  endfunction

  task automatic read_mem_file(input string fn, output logic [DATA_WIDTH-1:0] q[$]);
    integer fh, rc;
    logic [DATA_WIDTH-1:0] w;
    q.delete();
    fh = $fopen(fn, "r");
    if (!fh) begin
      $display("[ERROR] Cannot open: %s", fn);
      $finish;
    end
    while (!$feof(fh)) begin
      rc = $fscanf(fh, "%h", w);
      if (rc == 1) q.push_back(w);
    end
    $fclose(fh);
  endtask

  `include "int8_tb_util.svh"  // read_word_file, unpack_requant, apply_requant

  function automatic logic [LFSR_WIDTH-1:0] set_seed(input int k);
    return LFSR_WIDTH'(DROPOUT_SEED ^ (32'h85EBCA6B * k));
  endfunction

  longint cyc = 0;
  always_ff @(posedge clk_i) cyc <= cyc + 1;  // nonblocking, as TB_model_run: a blocking ++ raced its readers under --threads

  // Per copy: the sets it was given, in order; per lane of each copy, the words of its current set.
  // Keyed by int: Verilator dropped updates to a one-element fixed array of queues.
  int copy_sets[int][$];
  logic [DATA_WIDTH-1:0] lq[int][$];
  logic [DATA_WIDTH-1:0] got[int][$];
  longint t_start[int], t_done[int];
  int n_done = 0;

  always @(posedge clk_i) begin
    for (int i = 0; i < NL; i++) if (ov[i]) lq[i].push_back(od[i]);
    for (int c = 0; c < COPIES; c++) begin
      if (pipeline_complete_o[c]) begin
        automatic int k;
        automatic logic [DATA_WIDTH-1:0] w[$];
        automatic bit any;
        w.delete();
        any = 1;
        k = -1;
        if (copy_sets[c].size() > 0) begin
          k = copy_sets[c][0];
          copy_sets[c] = copy_sets[c][1:$];
        end
        // Lane l's j-th word is window j*NUM_LANES + l, so popping the lanes in turn rebuilds window order whatever each lane's stalls.
        while (any) begin
          any = 0;
          for (int l = 0; l < NUM_LANES; l++)
            if (lq[c*NUM_LANES+l].size() != 0) begin
              w.push_back(lq[c*NUM_LANES+l].pop_front());
              any = 1;
            end
        end
        $display("  [done] copy %0d set %0d: %0d outputs at cycle %0d", c, k, w.size(), cyc);
        got[k] = w;
        t_done[k] = cyc;
        n_done++;
      end
    end
  end

  logic [DATA_WIDTH-1:0] wq[$], nq[$], eq[$], bq[$];
  int failed = 0, fault = 0, tsel = 0;
  bit cached_mode = 0;

  // Wait on falling edges for a credit; this long without one is a hang.
  task automatic wait_copy_credit(input int c);
    int waited;
    waited = 0;
    while (hcnt[c] == 0) begin
      @(negedge clk_i);
      waited++;
      if (waited > WAIT_CYCLES) begin
        $display("[FATAL] copy %0d held no staging credit for %0d cycles, %0d sets done", c, waited, n_done);
        $finish;
      end
    end
  endtask
  task automatic wait_region_credit(input int r);
    int waited;
    waited = 0;
    while (!wcnt[r]) begin
      @(negedge clk_i);
      waited++;
      if (waited > WAIT_CYCLES) begin
        $display("[FATAL] region %0d credit not back for %0d cycles, %0d sets done", r, waited, n_done);
        $finish;
      end
    end
  endtask

  // One set to the copy whose turn it is: its A rows (and B unless cached) on the write buses, then its put with the sideband.
  task automatic put_set(input int k, input bit cached, input int tile, input bit mark);
    automatic int c;
    c = tsel;
    read_mem_file($sformatf("matrix_west_%0d.mem", k), wq);
    read_mem_file($sformatf("matrix_north_%0d.mem", k), nq);
    wait_copy_credit(c);
    if (int'(copy_sel_o) != c) begin
      failed++;
      $display("  [FAIL] set %0d: copy_sel_o %0d, the host's turn %0d", k, copy_sel_o, c);
    end
    for (int i = 0; i < wq.size(); i += HOST_WORDS) begin
      west_write_enable_i  = 1;
      north_write_enable_i = !cached;
      for (int w = 0; w < HOST_WORDS; w++) begin
        west_write_data_i[w]  = (i + w < wq.size()) ? wq[i+w] : '0;
        north_write_data_i[w] = (i + w < nq.size()) ? nq[i+w] : '0;
      end
      @(negedge clk_i);
    end
    west_write_enable_i  = 0;
    north_write_enable_i = 0;
    bias_i = '0;
    if (HAS_BIAS != 0) begin  // this TB streams no accumulate groups: every set carries its own bias
      logic [31:0] bw[$];
      read_word_file($sformatf("bias_%0d.mem", k), bw);
      for (int j = 0; j < N; j++) bias_i[j] = ACC_W'(bw[j]);
    end
    apply_requant(k);
    side               = '0;
    side.wc_last       = mark;
    side.weight_tile   = WCTW'(tile);
    side.weight_cached = cached;
    side.bias_valid    = (HAS_BIAS != 0);
    side.train         = TRAINING_MODE;
    side.seed          = set_seed(k);
    side.terms         = NUM_TERMS;
    side.act[0]        = ACTIVATION_CODE;
    side.zp[0]         = req_zp_i;
    side.amin[0]       = req_min_i;
    side.amax[0]       = req_max_i;
    side.mx[0]         = gp_mx_i;
    side.shx[0]        = gp_shx_i;
    side.mout[0]       = gp_mout_i;
    side.shout[0]      = gp_shout_i;
    side.zout[0]       = gp_zout_i;
    side.mult          = req_mult_i;
    side.shift         = req_shift_i;
    if (fault == 3 && k == 0) begin
      $display("  [FAULT 3] set 0 put to copy %0d, whose turn it is not", (c + 1) % COPIES);
      c = (c + 1) % COPIES;
    end
    copy_sets[c].push_back(k);
    if (!cached) $display("  [start] set %0d to copy %0d at cycle %0d", k, c, cyc);
    else if (!mark) $display("  [start] set %0d to copy %0d at cycle %0d, cached tile %0d", k, c, cyc, tile);
    else $display("  [start] set %0d to copy %0d at cycle %0d, cached tile %0d, its copy's last of the fill", k, c, cyc, tile);
    t_start[k] = cyc;
    hput[c] = 1'b1;
    @(negedge clk_i);
    hput = '0;
    tsel = (tsel + 1) % COPIES;
  endtask

  // Opens a fill of region r and writes sets k..k+s-1's B into its tiles 0..s-1.
  task automatic fill_region(input int r, input int k, input int s);
    wait_region_credit(r);
    wput[r] = 1'b1;
    @(negedge clk_i);
    wput[r] = 1'b0;
    for (int j = 0; j < s; j++) begin
      read_mem_file($sformatf("matrix_north_%0d.mem", k + j), nq);
      for (int i = 0; i < nq.size(); i += HOST_WORDS) begin
        wc_write_enable_i = 1;
        wc_write_addr_i = WCAW'((r * HALF + j) * N * N + i);
        for (int w = 0; w < HOST_WORDS; w++) north_write_data_i[w] = (i + w < nq.size()) ? nq[i+w] : '0;
        @(negedge clk_i);
      end
      wc_write_enable_i = 0;
    end
    $display("  [fill] region %0d: sets %0d..%0d at cycle %0d", r, k, k + s - 1, cyc);
  endtask

  initial begin
    void'($value$plusargs("out_slots=%d", out_slots));
    void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
    void'($value$plusargs("fault=%d", fault));
    cached_mode = $test$plusargs("cached");
    if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90) begin
      $display("[ERROR] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct);
      $finish;
    end
    $display(" Multi: %0d copies, cached sets %0d, %0d L9 slots per lane, %0d%% of credit returns stalled, fault %0d", COPIES,
             cached_mode, out_slots, out_stall_pct, fault);
    repeat (4) @(posedge clk_i);
    rstn_i = 1;
    repeat (4) @(negedge clk_i);
    if (!cached_mode) begin
      for (int k = 0; k < NUM_SETS; k++) put_set(k, 0, 0, 0);
    end else begin
      automatic int k = 0, f = 0;
      while (k < NUM_SETS) begin
        automatic int s;
        s = COPIES + (f % 3);
        if (fault == 1 && f == 0) s = 1;  // fewer sets than copies: a copy never sees a marked set
        if (fault == 2 && f == 0) s = COPIES + 1;
        if (fault == 4 && k == NUM_SETS - 1) s = 1;  // a short last fill: the host stops with a copy unmarked
        if (fault == 4 && k < NUM_SETS - 1 && k + s > NUM_SETS - 1) s = NUM_SETS - 1 - k;
        if (k + s > NUM_SETS) s = NUM_SETS - k;
        if (s < COPIES && !(fault == 1 && f == 0) && !(fault == 4 && k == NUM_SETS - 1)) begin
          put_set(k, 0, 0, 0);
          k++;
          continue;
        end
        fill_region(f % 2, k, s);
        for (int j = 0; j < s; j++) begin
          automatic bit mark;
          mark = (j >= s - COPIES);  // each copy's last set of the fill, in round-robin order
          if ((fault == 1 && f == 0) || (fault == 4 && k == NUM_SETS - 1)) mark = 1;
          if (fault == 2 && f == 0 && j == 0) begin
            mark = 1;
            $display("  [FAULT 2] set %0d marked although its copy takes set %0d of the same fill", k, k + COPIES);
          end
          put_set(k + j, 1, (f % 2) * HALF + j, mark);
        end
        k += s;
        f++;
        if (f % 2 == 0 && k < NUM_SETS - (fault == 4 ? 1 : 0)) begin
          put_set(k, 0, 0, 0);
          k++;
        end
      end
    end
    while (n_done < NUM_SETS && cyc < TIMEOUT_CYCLES) @(posedge clk_i);
    @(posedge clk_i);
    if (n_done < NUM_SETS) begin
      failed++;
      $display("  [FAIL] only %0d of %0d sets completed", n_done, NUM_SETS);
    end
    for (int k = 0; k < NUM_SETS; k++) begin
      automatic int errs;
      errs = 0;
      read_mem_file($sformatf("expected_output_%0d.mem", k), eq);
      read_mem_file($sformatf("bound_output_%0d.mem", k), bq);
      if (!got.exists(k) || got[k].size() != eq.size()) begin
        failed++;
        $display("  [FAIL] set %0d: %0d outputs, expected %0d", k, got.exists(k) ? got[k].size() : -1, eq.size());
        continue;
      end
      for (int i = 0; i < eq.size(); i++) if (!close(eq[i], got[k][i], (i < bq.size()) ? bq[i] : '0)) errs++;
      if (errs) begin
        failed++;
        $display("  [FAIL] set %0d: %0d of %0d outputs out of tolerance", k, errs, eq.size());
      end
      $display("  set %0d: latency %0d cycles, done at %0d, %0d outputs, %0d mismatches", k,
               t_done.exists(k) ? t_done[k] - t_start[k] : -1, t_done.exists(k) ? t_done[k] : -1, eq.size(), errs);
    end
    // Every credit comes home once the output stalls stop: staging 2 per copy, each region 1, every L9 slot.
    out_stall_pct = 0;
    repeat (OUT_CAP + 4) @(posedge clk_i);
    for (int c = 0; c < COPIES; c++)
      if (hcnt[c] != 2'd2) begin
        failed++;
        $display("  [FAIL] copy %0d holds %0d of its 2 staging credits after the last set", c, hcnt[c]);
      end
    if (!wcnt[0] || !wcnt[1]) begin
      failed++;
      $display("  [FAIL] region credits not back after the last set: %0d %0d", wcnt[0], wcnt[1]);
    end
    if (lane_home != '1) begin
      failed++;
      $display("  [FAIL] L9 slots not all credited back: lanes home %b", lane_home);
    end
    @(negedge clk_i);
    drain_chk = 1'b1;
    @(negedge clk_i);
    drain_chk = 1'b0;
    repeat (2) @(posedge clk_i);  // the registered drain checks judge a cycle later
    // Steady state over the second half, spanning whole rounds of COPIES sets so bursts do not skew it.
    begin
      automatic longint tl[$];
      automatic real gap_sum = 0.0;
      automatic int ng, lo;
      for (int k = 0; k < NUM_SETS; k++) if (t_done.exists(k)) tl.push_back(t_done[k]);
      tl.sort();
      ng = ((tl.size() / 2) / COPIES) * COPIES;
      lo = tl.size() - 1 - ng;
      if (ng > 0 && lo >= 0) gap_sum = real'(tl[tl.size()-1] - tl[lo]);
      if (ng > 0 && lo >= 0)
        $display("MULTI COPIES=%0d N=%0d lanes=%0d host_words=%0d collapse_k=%0d: steady %.1f cycles per set, %.1f FLOP/cycle",
                 COPIES, N, NUM_LANES, HOST_WORDS, COLLAPSE_K, gap_sum / ng, 2.0 * N * N * N * ng / gap_sum);
    end
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (failed == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
