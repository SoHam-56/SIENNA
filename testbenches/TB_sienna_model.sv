`timescale 1ns / 100ps

import test_config_pkg::*;

// Streams one network layer's sets from +sets=<file> through sienna_top back to back and writes every set's outputs to +out=<file>.
module TB_sienna_model;

  localparam ADDR_LINES = $clog2(FIFO_DEPTH);
  localparam int STALL_CYCLES = 50_000;  // this long with no completion is a hang
  localparam int ID_W = $clog2(SETS_IN_FLIGHT + 1);

  // ── DUT I/O ───────────────────────────────────────────────────────────
  logic clk_i, rstn_i;
  logic                     start_pipeline_i;
  logic                     training_mode_i;
  logic                     accumulate_i;
  logic [LFSR_WIDTH-1:0]    dropout_seed_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [     ADDR_LINES:0] num_terms_i;

  logic north_write_enable_i, north_write_reset_i;
  logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i;
  logic west_write_enable_i, west_write_reset_i;
  logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i;

  logic [ NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
  logic [ NUM_LANES-1:0]                 result_valid_o;
  logic                                  pipeline_complete_o;
  logic                                  pipeline_ready_o;
  logic                        [ID_W-1:0] done_set_id_o;
  logic systolic_busy_tb, gpnae_busy_tb, maxpool_busy_tb, dropout_busy_tb;
  logic intermediate_buffer_full_tb, intermediate_buffer_empty_tb;

  initial begin
    clk_i = 0;
    forever #5 clk_i = ~clk_i;
  end

  sienna_top #(
      .NUM_LANES        (NUM_LANES),
      .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (HOST_WORDS),
      .DATA_WIDTH       (DATA_WIDTH),
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
      .clk_i                      (clk_i),
      .rstn_i                     (rstn_i),
      .start_pipeline_i           (start_pipeline_i),
      .training_mode_i            (training_mode_i),
      .accumulate_i               (accumulate_i),
      .dropout_seed_i             (dropout_seed_i),
      .activation_function_i      (activation_function_i),
      .num_terms_i                (num_terms_i),
      .north_write_enable_i       (north_write_enable_i),
      .north_write_data_i         (north_write_data_i),
      .north_write_reset_i        (north_write_reset_i),
      .west_write_enable_i        (west_write_enable_i),
      .west_write_data_i          (west_write_data_i),
      .west_write_reset_i         (west_write_reset_i),
      .final_result_o             (final_result_o),
      .result_valid_o             (result_valid_o),
      .pipeline_complete_o        (pipeline_complete_o),
      .pipeline_ready_o           (pipeline_ready_o),
      .done_set_id_o              (done_set_id_o),
      .systolic_busy_o            (systolic_busy_tb),
      .gpnae_busy_o               (gpnae_busy_tb),
      .maxpool_busy_o             (maxpool_busy_tb),
      .dropout_busy_o             (dropout_busy_tb),
      .intermediate_buffer_full_o (intermediate_buffer_full_tb),
      .intermediate_buffer_empty_o(intermediate_buffer_empty_tb)
  );

  longint cycle = 0;
  always_ff @(posedge clk_i) cycle <= cycle + 1;

  // ── Output capture: every beat from the output port, one boundary per completed set ──
  logic [DATA_WIDTH-1:0] res_q[$];
  int bounds[$];
  int done_ids[$];
  longint last_done = 0;
  longint busy_mesh = 0, busy_act = 0;

  always_ff @(posedge clk_i) begin
    if (rstn_i) begin
      for (int lane = 0; lane < NUM_LANES; lane++) if (result_valid_o[lane]) res_q.push_back(final_result_o[lane]);
      if (pipeline_complete_o) begin
        bounds.push_back(res_q.size());
        done_ids.push_back(int'(done_set_id_o));
        last_done <= cycle;
      end
      if (systolic_busy_tb) busy_mesh <= busy_mesh + 1;
      if (gpnae_busy_tb) busy_act <= busy_act + 1;
    end
  end

  logic [DATA_WIDTH-1:0] west_q[$], north_q[$];

  // HOST_WORDS words of each operand per cycle, both operands at once.
  task automatic load_set();
    fork
      begin
        for (int i = 0; i < west_q.size(); i += HOST_WORDS) begin
          west_write_enable_i = 1;
          for (int c = 0; c < HOST_WORDS; c++) west_write_data_i[c] = (i + c < west_q.size()) ? west_q[i+c] : '0;
          @(posedge clk_i);
        end
        west_write_enable_i = 0;
        @(posedge clk_i);
      end
      begin
        for (int i = 0; i < north_q.size(); i += HOST_WORDS) begin
          north_write_enable_i = 1;
          for (int c = 0; c < HOST_WORDS; c++) north_write_data_i[c] = (i + c < north_q.size()) ? north_q[i+c] : '0;
          @(posedge clk_i);
        end
        north_write_enable_i = 0;
        @(posedge clk_i);
      end
    join
  endtask

  initial begin
    string sets_f, out_f;
    integer fin, fout, rc;
    int n_sets, acc, act, terms, bad_ids;
    logic [DATA_WIDTH-1:0] w;
    longint t0, waited;

    rstn_i = 0;
    start_pipeline_i = 0;
    training_mode_i = 0;
    accumulate_i = 0;
    dropout_seed_i = '1;
    activation_function_i = '0;
    num_terms_i = '0;
    north_write_reset_i = 1;
    west_write_reset_i = 1;
    north_write_enable_i = 0;
    west_write_enable_i = 0;
    north_write_data_i = '0;
    west_write_data_i = '0;

    if (!$value$plusargs("sets=%s", sets_f) || !$value$plusargs("out=%s", out_f)) begin
      $display("[MODEL] no +sets= and +out= given, nothing to run");
      $finish;
    end
    fin = $fopen(sets_f, "r");
    if (fin == 0) begin
      $display("[FATAL] cannot open %s", sets_f);
      $finish;
    end
    rc = $fscanf(fin, "%d", n_sets);

    repeat (10) @(posedge clk_i);
    rstn_i = 1;
    north_write_reset_i = 0;
    west_write_reset_i = 0;
    repeat (5) @(posedge clk_i);

    t0 = cycle;
    for (int k = 0; k < n_sets; k++) begin
      rc = $fscanf(fin, "%d %d %d", acc, act, terms);
      if (rc != 3) begin
        $display("[FATAL] set %0d header unreadable in %s", k, sets_f);
        $finish;
      end
      west_q.delete();
      north_q.delete();
      for (int i = 0; i < N * N; i++) begin
        rc = $fscanf(fin, "%h", w);
        west_q.push_back(w);
      end
      for (int i = 0; i < N * N; i++) begin
        rc = $fscanf(fin, "%h", w);
        north_q.push_back(w);
      end
      waited = 0;
      while (!pipeline_ready_o) begin
        @(posedge clk_i);
        waited++;
        if (waited > STALL_CYCLES) begin
          $display("[FATAL] pipeline_ready_o low for %0d cycles at set %0d, %0d sets done", waited, k, bounds.size());
          $finish;
        end
      end
      accumulate_i = acc[0];
      activation_function_i = CONTROL_WIDTH'(act);
      num_terms_i = terms[ADDR_LINES:0];
      load_set();
      start_pipeline_i = 1;
      @(posedge clk_i);
      start_pipeline_i = 0;
      @(posedge clk_i);  // let the credit land before sampling ready again
    end
    $fclose(fin);

    waited = 0;
    while (bounds.size() < n_sets) begin
      @(posedge clk_i);
      waited++;
      if (waited > STALL_CYCLES) begin
        $display("[FATAL] %0d of %0d sets completed, then nothing for %0d cycles", bounds.size(), n_sets, waited);
        $finish;
      end
    end
    @(posedge clk_i);

    // Sets complete in issue order, so set k's id is k mod 2^ID_W.
    bad_ids = 0;
    for (int k = 0; k < n_sets; k++) if (done_ids[k] != (k % (1 << ID_W))) bad_ids++;

    fout = $fopen(out_f, "w");
    for (int k = 0; k < n_sets; k++) begin
      automatic int lo = (k == 0) ? 0 : bounds[k-1];
      $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
      for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
    end
    $fclose(fout);
    $display("[MODEL] sets=%0d outputs=%0d cycles=%0d mesh_busy=%0d act_busy=%0d order_errors=%0d", n_sets,
             res_q.size(), last_done - t0 + 1, busy_mesh, busy_act, bad_ids);
    $finish;
  end

endmodule
