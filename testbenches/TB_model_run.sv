`timescale 1ns / 100ps

import test_config_pkg::*;

`include "tb_l9_sink.svh"

// Runs one layer on sienna_layer: writes the configuration, streams the two inputs from +layer=<file> on the row links (L10)
// as the layer grants credits, and writes every result taken from the output links (L9) to +out=<file>.
// Layer file: "L m kb n residual bias act train seed a_rows w_rows packed", then a_rows and w_rows rows of N hex words.
// packed: a "P sh m0 .. m(N/2-1)" line and seven "E act zp min max mx shx mout shout zout" lines after L (and Q).
// int8 (IS_INT): a "Q zp min max mx shx mout shout zout" line after it; after the rows, per column block N biases, N multipliers, N shifts (hex).
// +out_slots=S (1..64) L9 slots per lane, +out_stall_pct=P and +in_stall_pct=P random stalls; -DTB_OUT_STALL_PCT / -DTB_IN_STALL_PCT set the defaults.
// FAULT 1: an A row put with no credit; 2: a W row put with no credit; 3: the A link one bit wide too many.
module TB_model_run #(
    parameter int FAULT       = 0,
    parameter int LINK_STAGES = 0  // sienna_top's register stages on L0, L1, L3 and L9
);

  localparam int STALL_CYCLES = 50_000;  // this long with nothing moving is a hang
  localparam int RCW = $clog2(N + 1);  // a set's N row credits in one cycle
  localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;

  logic cfg_load_i = 0, cfg_residual_i = 0, cfg_bias_i = 0, cfg_train_i = 0;
  logic [15:0] cfg_m_i = '0, cfg_kb_i = '0, cfg_n_i = '0;
  logic [CONTROL_WIDTH-1:0] cfg_act_i = '0;
  logic [LFSR_WIDTH-1:0] cfg_seed_i = '0;
  logic busy_o, done_o, set_done_o;
  logic [7:0] cfg_req_zp_i = '0, cfg_req_min_i = '0, cfg_req_max_i = '0, cfg_gp_shout_i = '0, cfg_gp_zout_i = '0;
  logic [15:0] cfg_gp_mx_i = '0;
  logic [4:0] cfg_gp_shx_i = '0;
  logic [31:0] cfg_gp_mout_i = '0;
  logic [N-1:0][ACC_W-1:0] w_bias_i = '0;  // int8: with each block's bias row
  logic [N-1:0][31:0] w_req_mult_i = '0;
  logic [N-1:0][7:0] w_req_shift_i = '0;
  logic [2:0] cfg_pack_shift_i = '0;  // packing: the P and E lines
  logic [N/2-1:0][2:0] cfg_pack_map_i = '0;
  logic [7:1][CONTROL_WIDTH-1:0] cfg_pack_act_i = '0;
  logic [7:1][7:0] cfg_pack_zp_i = '0, cfg_pack_min_i = '0, cfg_pack_max_i = '0, cfg_pack_shout_i = '0, cfg_pack_zout_i = '0;
  logic [7:1][15:0] cfg_pack_mx_i = '0;
  logic [7:1][4:0] cfg_pack_shx_i = '0;
  logic [7:1][31:0] cfg_pack_mout_i = '0;
  logic [31:0] ep[$];  // int8 epilogues, 3N words per column block

  // ── Row links (L10): this side is the producer of both streams, a counter and a checker on each ──
  credit_link_if #(.DATA_W(N * DATA_WIDTH + (FAULT == 3 ? 1 : 0)), .CRW(RCW)) a_lnk ();
  credit_link_if #(.DATA_W(N * DATA_WIDTH), .CRW(RCW)) w_lnk ();
  logic [$clog2(2*N+1)-1:0] a_cnt, w_cnt;
  credit_counter #(.MAX(2 * N), .CRW(RCW)) a_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(a_lnk.put), .credit_i(a_lnk.credit),
                                                .has_credit_o(), .count_o(a_cnt));
  credit_counter #(.MAX(2 * N), .CRW(RCW)) w_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(w_lnk.put), .credit_i(w_lnk.credit),
                                                .has_credit_o(), .count_o(w_cnt));
  // The layer grants per set, bias row or tile, so nothing is outstanding when idle: a_all_back is not armed; the end checks the counts are 0.
  credit_link_checker #(.SLOTS(N)) a_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(1'b0), .lnk(a_lnk));
  credit_link_checker #(.SLOTS(2 * N)) w_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(1'b0), .lnk(w_lnk));
  initial begin
    a_lnk.put = 1'b0;
    a_lnk.data = '0;
    w_lnk.put = 1'b0;
    w_lnk.data = '0;
  end

  // ── Output links (L9): one consumer per lane ──
  credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) out_lnk[NUM_LANES] ();
`ifdef TB_OUT_STALL_PCT
  int out_stall_pct = `TB_OUT_STALL_PCT;
`else
  int out_stall_pct = 0;
`endif
`ifdef TB_IN_STALL_PCT
  int in_stall_pct = `TB_IN_STALL_PCT;
`else
  int in_stall_pct = 0;
`endif
  int out_slots = OUT_CAP;
  logic drain_chk = 1'b0;
  logic [NUM_LANES-1:0] res_v, lane_home;
  logic [DATA_WIDTH-1:0] res_d[NUM_LANES];
  for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
    tb_l9_sink #(.DATA_W(DATA_WIDTH), .MAX_SLOTS(OUT_CAP)) snk (.clk_i(clk_i), .rstn_i(rstn_i), .slots_i(out_slots),
                                                                .stall_pct_i(out_stall_pct), .hold_i(1'b0), .drain_i(drain_chk),
                                                                .lnk(out_lnk[l]), .valid_o(res_v[l]), .data_o(res_d[l]), .held_o(),
                                                                .home_o(lane_home[l]));
  end

  sienna_layer #(
      .NUM_LANES     (NUM_LANES),
      .N             (N),
      .TILE_SIZE     (TILE_SIZE),
      .SETS_IN_FLIGHT(SETS_IN_FLIGHT),
      .DATA_WIDTH    (DATA_WIDTH),
      .EXP_W         (EXP_W),
      .MAN_W         (MAN_W),
      .CONTROL_WIDTH (CONTROL_WIDTH),
      .LFSR_WIDTH    (LFSR_WIDTH),
      .LINK_STAGES   (LINK_STAGES),
      .OUT_MAX       (OUT_CAP),
      .OUT_CRW       (1)
  ) dut (
      .clk_i           (clk_i),
      .rstn_i          (rstn_i),
      .cfg_load_i      (cfg_load_i),
      .cfg_m_i         (cfg_m_i),
      .cfg_kb_i        (cfg_kb_i),
      .cfg_n_i         (cfg_n_i),
      .cfg_residual_i  (cfg_residual_i),
      .cfg_bias_i      (cfg_bias_i),
      .cfg_act_i       (cfg_act_i),
      .cfg_train_i     (cfg_train_i),
      .cfg_seed_i      (cfg_seed_i),
      .cfg_req_zp_i    (cfg_req_zp_i),
      .cfg_req_min_i   (cfg_req_min_i),
      .cfg_req_max_i   (cfg_req_max_i),
      .cfg_gp_mx_i     (cfg_gp_mx_i),
      .cfg_gp_shx_i    (cfg_gp_shx_i),
      .cfg_gp_mout_i   (cfg_gp_mout_i),
      .cfg_gp_shout_i  (cfg_gp_shout_i),
      .cfg_gp_zout_i   (cfg_gp_zout_i),
      .cfg_pack_shift_i(cfg_pack_shift_i),
      .cfg_pack_map_i  (cfg_pack_map_i),
      .cfg_pack_act_i  (cfg_pack_act_i),
      .cfg_pack_zp_i   (cfg_pack_zp_i),
      .cfg_pack_min_i  (cfg_pack_min_i),
      .cfg_pack_max_i  (cfg_pack_max_i),
      .cfg_pack_mx_i   (cfg_pack_mx_i),
      .cfg_pack_shx_i  (cfg_pack_shx_i),
      .cfg_pack_mout_i (cfg_pack_mout_i),
      .cfg_pack_shout_i(cfg_pack_shout_i),
      .cfg_pack_zout_i (cfg_pack_zout_i),
      .busy_o          (busy_o),
      .done_o          (done_o),
      .a_rows          (a_lnk),
      .w_rows          (w_lnk),
      .w_bias_i        (w_bias_i),
      .w_req_mult_i    (w_req_mult_i),
      .w_req_shift_i   (w_req_shift_i),
      .out             (out_lnk),
      .set_done_o      (set_done_o)
  );

  longint cycle = 0;
  always_ff @(posedge clk_i) cycle <= cycle + 1;

  // Results per set in window order: lane L's j-th word is window j*NUM_LANES + L, whatever each lane's stalls; empty for partial sums.
  logic [DATA_WIDTH-1:0] lane_q[NUM_LANES][$];
  logic [DATA_WIDTH-1:0] res_q[$];
  int bounds[$];
  longint t_done = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i) begin
      for (int l = 0; l < NUM_LANES; l++) if (res_v[l]) lane_q[l].push_back(res_d[l]);
      if (set_done_o) begin
        automatic bit any;
        any = 1;
        while (any) begin
          any = 0;
          for (int l = 0; l < NUM_LANES; l++)
            if (lane_q[l].size() != 0) begin
              res_q.push_back(lane_q[l].pop_front());
              any = 1;
            end
        end
        bounds.push_back(res_q.size());
      end
      if (done_o) t_done <= cycle;
    end
  end

  // int8: the epilogue beside the next bias row; the layer says which W row it took as a block's bias.
  int ei = 0;
  always_ff @(posedge clk_i) if (rstn_i && IS_INT && dut.wl_take_bias) ei <= ei + 1;

`ifdef PERF
  // Cycles with sets left but no A row taken, by cause, printed at the end.
  int st_credit = 0, st_staging = 0, st_bias = 0, st_tile = 0, st_input = 0, st_rows = 0, st_idle = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i && dut.active && !a_lnk.put && dut.g_blk < dut.ct) begin
      if (dut.pend) st_input++;  // the set's rows are granted and not all sent
      else if (!dut.g_l0_ok && dut.pipe.entry_full) st_credit++;  // the entry withholds the staging credit
      else if (!dut.g_l0_ok) st_staging++;
      else if (!dut.g_bias_ok) st_bias++;
      else if (!dut.g_tile_ok) st_tile++;
      else st_idle++;
    end
    if (rstn_i && a_lnk.put) st_rows++;
  end
  // Downstream: cycles a stage holds a ready set but cannot pass it on, and why.
  int dn_accbank = 0, dn_resbank = 0, dn_actbusy = 0, dn_actbank = 0, dn_poolbusy = 0, dn_bcast = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i && dut.active) begin
      if (dut.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ob_full[dut.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.fb] &&
          dut.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ab_busy[dut.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ab_next])
        dn_accbank++;
      if (dut.pipe.systolic_array_inst.in_full[dut.pipe.systolic_array_inst.in_rd] && !dut.pipe.systolic_array_inst.arrays_load_ready &&
          dut.pipe.systolic_array_inst.bstate == 0)
        dn_bcast++;
      if (dut.pipe.systolic_array_inst.arrays_final && !dut.pipe.systolic_array_inst.reduce_start &&
          dut.pipe.systolic_array_inst.out_state[dut.pipe.systolic_array_inst.out_wr] != 0)
        dn_resbank++;
      if (dut.pipe.systolic_array_inst.out_full[dut.pipe.systolic_array_inst.out_rd] && dut.pipe.g_fed) dn_actbusy++;
      if (dut.pipe.systolic_array_inst.out_full[dut.pipe.systolic_array_inst.out_rd] && dut.pipe.g_state == 0 &&
          dut.pipe.act_full[dut.pipe.act_wr]) dn_actbank++;
      if (dut.pipe.act_full[dut.pipe.act_rd] && dut.pipe.p_state != 0) dn_poolbusy++;
    end
  end
`endif

  logic [DATA_WIDTH-1:0] a_rows[$], w_rows[$];  // N words per row, back to back
  int ai = 0, wi = 0, na = 0, nw = 0, ne = 0;
  bit streaming = 0, faulted = 0;

  // The producers: a row goes on the falling edge while a credit is held and the random input stall allows it.
  always @(negedge clk_i) begin
    a_lnk.put = 1'b0;
    w_lnk.put = 1'b0;
    if (streaming) begin
      if (FAULT == 1 && !faulted) begin
        $display("  [FAULT 1] an A row put with %0d A credits held", a_cnt);
        a_lnk.put = 1'b1;
        faulted = 1;
      end else if (FAULT == 2 && !faulted) begin
        $display("  [FAULT 2] a W row put with %0d W credits held", w_cnt);
        w_lnk.put = 1'b1;
        faulted = 1;
      end
      if (ai < na && a_cnt != 0 && !a_lnk.put && !(in_stall_pct > 0 && $urandom_range(99) < in_stall_pct)) begin
        for (int c = 0; c < N; c++) a_lnk.data[c*DATA_WIDTH +: DATA_WIDTH] = a_rows[ai*N+c];
        a_lnk.put = 1'b1;
        ai++;
      end
      if (wi < nw && w_cnt != 0 && !w_lnk.put && !(in_stall_pct > 0 && $urandom_range(99) < in_stall_pct)) begin
        for (int c = 0; c < N; c++) w_lnk.data[c*DATA_WIDTH +: DATA_WIDTH] = w_rows[wi*N+c];
        w_lnk.put = 1'b1;
        wi++;
      end
      for (int c = 0; c < N; c++) begin
        w_bias_i[c]      = (ei < ne) ? ACC_W'(ep[(3*ei)*N+c]) : '0;
        w_req_mult_i[c]  = (ei < ne) ? ep[(3*ei+1)*N+c] : '0;
        w_req_shift_i[c] = (ei < ne) ? ep[(3*ei+2)*N+c][7:0] : '0;
      end
    end
  end

  initial begin
    string layer_f, out_f, kind;
    integer fin, fout, rc;
    int m, kb, n, res, bias, act, train, seed, pk, idle, last_ai, last_wi, last_sets;
    int qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout;
    logic [DATA_WIDTH-1:0] v;
    logic [31:0] w32;
    longint t0;
    bit finished;
    int failed = 0, osp;

    void'($value$plusargs("out_slots=%d", out_slots));
    void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
    void'($value$plusargs("in_stall_pct=%d", in_stall_pct));
    if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90 || in_stall_pct < 0 || in_stall_pct > 90) begin
      $display("[FATAL] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d, +in_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct,
               in_stall_pct);
      $finish;
    end
    if (!$value$plusargs("layer=%s", layer_f) || !$value$plusargs("out=%s", out_f)) begin
      $display("[LAYER] no +layer= and +out= given, nothing to run (%0d L9 slots per lane, %0d%% output and %0d%% input stalls)",
               out_slots, out_stall_pct, in_stall_pct);
      $finish;
    end
    fin = $fopen(layer_f, "r");
    if (fin == 0) begin
      $display("[FATAL] cannot open %s", layer_f);
      $finish;
    end
    rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d %d %d", kind, m, kb, n, res, bias, act, train, seed, na, nw, pk);
    if (rc != 12 || kind != "L") begin
      $display("[FATAL] %s is not a layer file", layer_f);
      $finish;
    end
    {qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout} = '0;  // fp32 and bf16 keep every int8 input at 0
    if (IS_INT) begin
      rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d", kind, qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout);
      if (rc != 9 || kind != "Q") begin
        $display("[FATAL] %s: an int8 layer file needs its Q line", layer_f);
        $finish;
      end
    end
    if (pk) begin
      int sh, mv, ev[9];
      rc = $fscanf(fin, "%s %d", kind, sh);
      if (rc != 2 || kind != "P") begin
        $display("[FATAL] %s: a packed layer needs its P line", layer_f);
        $finish;
      end
      cfg_pack_shift_i = 3'(sh);
      for (int c = 0; c < N / 2; c++) begin
        rc = $fscanf(fin, "%d", mv);
        cfg_pack_map_i[c] = 3'(mv);
      end
      for (int e = 1; e < 8; e++) begin
        rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d", kind, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8]);
        if (rc != 10 || kind != "E") begin
          $display("[FATAL] %s: entry %0d needs its E line", layer_f, e);
          $finish;
        end
        cfg_pack_act_i[e] = CONTROL_WIDTH'(ev[0]);
        cfg_pack_zp_i[e] = 8'(ev[1]);
        cfg_pack_min_i[e] = 8'(ev[2]);
        cfg_pack_max_i[e] = 8'(ev[3]);
        cfg_pack_mx_i[e] = 16'(ev[4]);
        cfg_pack_shx_i[e] = 5'(ev[5]);
        cfg_pack_mout_i[e] = 32'(ev[6]);
        cfg_pack_shout_i[e] = 8'(ev[7]);
        cfg_pack_zout_i[e] = 8'(ev[8]);
      end
    end
    for (int i = 0; i < (na + nw) * N; i++) begin
      rc = $fscanf(fin, "%h", v);
      if (i < na * N) a_rows.push_back(v);
      else w_rows.push_back(v);
    end
    ne = IS_INT ? (n + N - 1) / N : 0;  // one epilogue per column block
    for (int i = 0; i < ne * 3 * N; i++) begin
      rc = $fscanf(fin, "%h", w32);
      ep.push_back(w32);
    end
    $fclose(fin);

    repeat (5) @(posedge clk_i);
    #1 rstn_i = 1;
    repeat (3) @(posedge clk_i);

    // Configure: the only control the software gives.
    #1;
    cfg_m_i = 16'(m);
    cfg_kb_i = 16'(kb);
    cfg_n_i = 16'(n);
    cfg_residual_i = res[0];
    cfg_bias_i = bias[0];
    cfg_act_i = CONTROL_WIDTH'(act);
    cfg_train_i = train[0];
    cfg_seed_i = LFSR_WIDTH'(seed);
    cfg_req_zp_i = 8'(qzp);
    cfg_req_min_i = 8'(qmin);
    cfg_req_max_i = 8'(qmax);
    cfg_gp_mx_i = 16'(qmx);
    cfg_gp_shx_i = 5'(qshx);
    cfg_gp_mout_i = 32'(qmout);
    cfg_gp_shout_i = 8'(qshout);
    cfg_gp_zout_i = 8'(qzout);
    cfg_load_i = 1;
    @(posedge clk_i);
    #1 cfg_load_i = 0;
    t0 = cycle;
    streaming = 1;

    // Stream both inputs as credits arrive; a hang is this long with no row and no set leaving.
    idle = 0;
    finished = 0;
    last_ai = 0;
    last_wi = 0;
    last_sets = 0;
    while (!finished) begin
      @(posedge clk_i);
      finished = done_o;
      #1;
      idle = (ai != last_ai || wi != last_wi || bounds.size() != last_sets) ? 0 : idle + 1;
      last_ai = ai;
      last_wi = wi;
      last_sets = bounds.size();
      if (idle > STALL_CYCLES) begin
        $display("[FATAL] nothing moved for %0d cycles: %0d/%0d activation rows, %0d/%0d weight rows, %0d sets done", idle, ai, na, wi,
                 nw, bounds.size());
        $finish;
      end
    end
    streaming = 0;
    // The layer granted exactly the rows the streams hold, and every L9 slot comes home once the stalls stop.
    if (a_cnt != 0 || w_cnt != 0) begin
      failed++;
      $display("[FAIL] the layer is done with row credits still held: A %0d, W %0d", a_cnt, w_cnt);
    end
    osp = out_stall_pct;
    out_stall_pct = 0;
    repeat (OUT_CAP + 4) @(posedge clk_i);
    if (lane_home != '1) begin
      failed++;
      $display("[FAIL] L9 slots not all credited back after the layer: lanes home %b", lane_home);
    end
    @(negedge clk_i);
    drain_chk = 1'b1;
    @(negedge clk_i);
    drain_chk = 1'b0;

    fout = $fopen(out_f, "w");
    for (int k = 0; k < bounds.size(); k++) begin
      automatic int lo = (k == 0) ? 0 : bounds[k-1];
      if (bounds[k] == lo) continue;  // a partial sum
      $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
      for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
    end
    $fclose(fout);
    if (failed == 0)
      $display("[LAYER] sets=%0d outputs=%0d cycles=%0d a_rows=%0d/%0d w_rows=%0d/%0d epilogues=%0d/%0d out_stall=%0d in_stall=%0d",
               bounds.size(), res_q.size(), t_done - t0 + 1, ai, na, wi, nw, ei, ne, osp, in_stall_pct);
`ifdef PERF
    $display("[PERF] rows=%0d stalls: mid-set=%0d credits=%0d staging=%0d bias=%0d tile=%0d other=%0d", st_rows, st_input,
             st_credit, st_staging, st_bias, st_tile, st_idle);
    $display("[PERF] held: operand-bank-full=%0d partial-sum-bank=%0d result-bank=%0d activation-busy=%0d activation-bank=%0d pooling-busy=%0d",
             dn_bcast, dn_accbank, dn_resbank, dn_actbusy, dn_actbank, dn_poolbusy);
`endif
    $finish;
  end

endmodule
