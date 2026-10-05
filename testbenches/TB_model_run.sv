`timescale 1ns / 100ps

import test_config_pkg::*;

// Runs one layer on sienna_layer: writes the configuration, streams the two inputs from +layer=<file>
// whenever the hardware is ready, and writes every result to +out=<file>. Nothing else crosses the boundary.
// Layer file: "L m kb n residual bias act train seed a_rows w_rows packed", then a_rows and w_rows rows of N hex words.
// packed: a "P sh m0 .. m(N/2-1)" line and seven "E act zp min max mx shx mout shout zout" lines after L (and Q).
// int8 (IS_INT): a "Q zp min max mx shx mout shout zout" line after it; after the rows, per column block N biases, N multipliers, N shifts (hex).
module TB_model_run;

  localparam int STALL_CYCLES = 50_000;  // this long with nothing moving is a hang

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;

  logic cfg_load_i = 0, cfg_residual_i = 0, cfg_bias_i = 0, cfg_train_i = 0;
  logic [15:0] cfg_m_i = '0, cfg_kb_i = '0, cfg_n_i = '0;
  logic [CONTROL_WIDTH-1:0] cfg_act_i = '0;
  logic [LFSR_WIDTH-1:0] cfg_seed_i = '0;
  logic busy_o, done_o, set_done_o;
  logic a_valid_i = 0, w_valid_i = 0, a_ready_o, w_ready_o;
  logic [N-1:0][DATA_WIDTH-1:0] a_data_i = '0, w_data_i = '0;  // both streams in the package's operand format
  logic [7:0] cfg_req_zp_i = '0, cfg_req_min_i = '0, cfg_req_max_i = '0, cfg_gp_shout_i = '0, cfg_gp_zout_i = '0;
  logic [15:0] cfg_gp_mx_i = '0;
  logic [4:0] cfg_gp_shx_i = '0;
  logic [31:0] cfg_gp_mout_i = '0;
  logic [N-1:0][ACC_W-1:0] w_bias_i = '0;  // int8: beside each block's bias beat
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
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
  logic [NUM_LANES-1:0] result_valid_o;

  sienna_layer #(
      .NUM_LANES     (NUM_LANES),
      .N             (N),
      .TILE_SIZE     (TILE_SIZE),
      .SETS_IN_FLIGHT(SETS_IN_FLIGHT),
      .DATA_WIDTH    (DATA_WIDTH),
      .EXP_W      (EXP_W),
      .MAN_W      (MAN_W),
      .CONTROL_WIDTH (CONTROL_WIDTH),
      .LFSR_WIDTH    (LFSR_WIDTH)
  ) dut (.*);

  longint cycle = 0;
  always_ff @(posedge clk_i) cycle <= cycle + 1;

  // Results in arrival order; a set boundary after each completed set, empty for partial sums.
  logic [DATA_WIDTH-1:0] res_q[$];
  int bounds[$];
  longint t_done = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i) begin
      for (int l = 0; l < NUM_LANES; l++) if (result_valid_o[l]) res_q.push_back(final_result_o[l]);
      if (set_done_o) bounds.push_back(res_q.size());
      if (done_o) t_done <= cycle;
    end
  end

`ifdef PERF
  // Cycles with work left but no row taken, by cause, printed at the end.
  int st_credit = 0, st_staging = 0, st_bias = 0, st_tile = 0, st_input = 0, st_rows = 0, st_idle = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i && dut.active && !dut.is_row_ok && dut.is_blk < dut.ct) begin
      if (dut.is_loading) st_input++;
      else if (dut.pipe.credits == 0) st_credit++;
      else if (!dut.pipe.mesh_input_ready) st_staging++;
      else if (dut.bias_q && !dut.bias_in[dut.is_blk[0]]) st_bias++;
      else if (dut.is_cached_pass && !(dut.tiles_in[dut.is_blk[0]] > dut.is_p)) st_tile++;
      else st_idle++;
    end
    if (rstn_i && dut.is_row_ok) st_rows++;
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
      if (dut.pipe.systolic_collection_complete && dut.pipe.g_state != 0) dn_actbusy++;
      if (dut.pipe.systolic_collection_complete && dut.pipe.g_state == 0 && dut.pipe.act_full[dut.pipe.act_wr]) dn_actbank++;
      if (dut.pipe.act_full[dut.pipe.act_rd] && dut.pipe.p_state != 0) dn_poolbusy++;
    end
  end
`endif

  logic [DATA_WIDTH-1:0] a_rows[$], w_rows[$];  // N words per row, back to back

  initial begin
    string layer_f, out_f, kind;
    integer fin, fout, rc;
    int m, kb, n, res, bias, act, train, seed, na, nw, pk, ai, wi, idle;
    int qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout, ne, ei;
    logic [DATA_WIDTH-1:0] v;
    logic [31:0] w32;
    longint t0;
    bit a_take, w_take, finished;
    bit e_take;

    if (!$value$plusargs("layer=%s", layer_f) || !$value$plusargs("out=%s", out_f)) begin
      $display("[LAYER] no +layer= and +out= given, nothing to run");
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

    // Stream both inputs; a row moves on an edge where valid and ready are both high.
    ai = 0;
    ei = 0;
    wi = 0;
    idle = 0;
    finished = 0;
    while (!finished) begin
      a_valid_i = (ai < na);
      w_valid_i = (wi < nw);
      for (int c = 0; c < N; c++) begin
        a_data_i[c] = (ai < na) ? a_rows[ai*N+c] : '0;
        w_data_i[c] = (wi < nw) ? w_rows[wi*N+c] : '0;
        w_bias_i[c]      = (ei < ne) ? ACC_W'(ep[(3*ei)*N+c]) : '0;
        w_req_mult_i[c]  = (ei < ne) ? ep[(3*ei+1)*N+c] : '0;
        w_req_shift_i[c] = (ei < ne) ? ep[(3*ei+2)*N+c][7:0] : '0;
      end
      @(negedge clk_i);  // ready has settled
      a_take = a_valid_i && a_ready_o;
      w_take = w_valid_i && w_ready_o;
      e_take = IS_INT && w_take && dut.wl_take_bias;  // the beat the layer takes as a block's bias
      @(posedge clk_i);
      finished = done_o;
      #1;
      if (a_take) ai++;
      if (w_take) wi++;
      if (e_take) ei++;
      idle = (a_take || w_take || set_done_o) ? 0 : idle + 1;
      if (idle > STALL_CYCLES) begin
        $display("[FATAL] nothing moved for %0d cycles: %0d/%0d activation rows, %0d/%0d weight rows, %0d sets done",
                 idle, ai, na, wi, nw, bounds.size());
        $finish;
      end
    end
    a_valid_i = 0;
    w_valid_i = 0;
    repeat (2) @(posedge clk_i);

    fout = $fopen(out_f, "w");
    for (int k = 0; k < bounds.size(); k++) begin
      automatic int lo = (k == 0) ? 0 : bounds[k-1];
      if (bounds[k] == lo) continue;  // a partial sum
      $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
      for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
    end
    $fclose(fout);
    $display("[LAYER] sets=%0d outputs=%0d cycles=%0d a_rows=%0d/%0d w_rows=%0d/%0d epilogues=%0d/%0d", bounds.size(),
             res_q.size(), t_done - t0 + 1, ai, na, wi, nw, ei, ne);
`ifdef PERF
    $display("[PERF] rows=%0d stalls: mid-set=%0d credits=%0d staging=%0d bias=%0d tile=%0d other=%0d", st_rows, st_input,
             st_credit, st_staging, st_bias, st_tile, st_idle);
    $display("[PERF] held: operand-bank-full=%0d partial-sum-bank=%0d result-bank=%0d activation-busy=%0d activation-bank=%0d pooling-busy=%0d",
             dn_bcast, dn_accbank, dn_resbank, dn_actbusy, dn_actbank, dn_poolbusy);
`endif
    $finish;
  end

endmodule
