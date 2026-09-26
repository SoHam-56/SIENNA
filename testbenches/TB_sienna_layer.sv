`timescale 1ns / 100ps

import test_config_pkg::*;

// Runs one layer on sienna_layer: writes the configuration, streams the two inputs from +layer=<file>
// whenever the hardware is ready, and writes every result to +out=<file>. Nothing else crosses the boundary.
// Layer file: "L m kb n residual bias act train seed a_rows w_rows", then a_rows and w_rows rows of N hex words.
module TB_sienna_layer;

  localparam int STALL_CYCLES = 50_000;  // this long with nothing moving is a hang

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;

  logic cfg_load_i = 0, cfg_residual_i = 0, cfg_bias_i = 0, cfg_train_i = 0;
  logic [15:0] cfg_m_i = '0, cfg_kb_i = '0, cfg_n_i = '0;
  logic [CONTROL_WIDTH-1:0] cfg_act_i = '0;
  logic [LFSR_WIDTH-1:0] cfg_seed_i = '0;
  logic busy_o, done_o, set_done_o;
  logic a_valid_i = 0, w_valid_i = 0, a_ready_o, w_ready_o;
  logic [N-1:0][DATA_WIDTH-1:0] a_data_i = '0, w_data_i = '0;
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
  logic [NUM_LANES-1:0] result_valid_o;

  sienna_layer #(
      .NUM_LANES     (NUM_LANES),
      .N             (N),
      .TILE_SIZE     (TILE_SIZE),
      .SETS_IN_FLIGHT(SETS_IN_FLIGHT),
      .DATA_WIDTH    (DATA_WIDTH),
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
`endif

  logic [DATA_WIDTH-1:0] a_rows[$], w_rows[$];  // N words per row, back to back

  initial begin
    string layer_f, out_f, kind;
    integer fin, fout, rc;
    int m, kb, n, res, bias, act, train, seed, na, nw, ai, wi, idle;
    logic [DATA_WIDTH-1:0] v;
    longint t0;
    bit a_take, w_take, finished;

    if (!$value$plusargs("layer=%s", layer_f) || !$value$plusargs("out=%s", out_f)) begin
      $display("[LAYER] no +layer= and +out= given, nothing to run");
      $finish;
    end
    fin = $fopen(layer_f, "r");
    if (fin == 0) begin
      $display("[FATAL] cannot open %s", layer_f);
      $finish;
    end
    rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d %d", kind, m, kb, n, res, bias, act, train, seed, na, nw);
    if (rc != 11 || kind != "L") begin
      $display("[FATAL] %s is not a layer file", layer_f);
      $finish;
    end
    for (int i = 0; i < (na + nw) * N; i++) begin
      rc = $fscanf(fin, "%h", v);
      if (i < na * N) a_rows.push_back(v);
      else w_rows.push_back(v);
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
    cfg_load_i = 1;
    @(posedge clk_i);
    #1 cfg_load_i = 0;
    t0 = cycle;

    // Stream both inputs; a row moves on an edge where valid and ready are both high.
    ai = 0;
    wi = 0;
    idle = 0;
    finished = 0;
    while (!finished) begin
      a_valid_i = (ai < na);
      w_valid_i = (wi < nw);
      for (int c = 0; c < N; c++) begin
        a_data_i[c] = (ai < na) ? a_rows[ai*N+c] : '0;
        w_data_i[c] = (wi < nw) ? w_rows[wi*N+c] : '0;
      end
      @(negedge clk_i);  // ready has settled
      a_take = a_valid_i && a_ready_o;
      w_take = w_valid_i && w_ready_o;
      @(posedge clk_i);
      finished = done_o;
      #1;
      if (a_take) ai++;
      if (w_take) wi++;
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
    $display("[LAYER] sets=%0d outputs=%0d cycles=%0d a_rows=%0d/%0d w_rows=%0d/%0d", bounds.size(), res_q.size(),
             t_done - t0 + 1, ai, na, wi, nw);
`ifdef PERF
    $display("[PERF] rows=%0d stalls: mid-set=%0d credits=%0d staging=%0d bias=%0d tile=%0d other=%0d", st_rows, st_input,
             st_credit, st_staging, st_bias, st_tile, st_idle);
`endif
    $finish;
  end

endmodule
