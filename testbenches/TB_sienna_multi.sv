`timescale 1ns / 100ps

import test_config_pkg::*;

// Streams NUM_SETS distinct sets through sienna_multi, checks every set against its golden output, and measures throughput.
module TB_sienna_multi #(
    parameter int COPIES     = 2,
    parameter int COLLAPSE_K = 1
);
  localparam ADDR_LINES = $clog2(FIFO_DEPTH);
  localparam int TIMEOUT_CYCLES = 400_000;
  localparam real REL_TOL = 0.01;

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;

  logic start_pipeline_i = 0, training_mode_i = TRAINING_MODE;
  logic [LFSR_WIDTH-1:0] dropout_seed_i = '0;
  logic [CONTROL_WIDTH-1:0] activation_function_i = ACTIVATION_CODE;
  logic [ADDR_LINES:0] num_terms_i = NUM_TERMS;
  logic north_write_enable_i = 0, west_write_enable_i = 0, north_write_reset_i = 0, west_write_reset_i = 0;
  logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i = '0, west_write_data_i = '0;

  logic pipeline_ready_o;
  logic [$clog2(COPIES+1)-1:0] copy_sel_o;
  logic [COPIES-1:0][NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
  logic [COPIES-1:0][NUM_LANES-1:0] result_valid_o;
  logic [COPIES-1:0] pipeline_complete_o;
  logic [COPIES-1:0][1:0] done_set_id_o;

  sienna_multi #(
      .COPIES           (COPIES),
      .NUM_LANES        (NUM_LANES),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (HOST_WORDS),
      .COLLAPSE_K       (COLLAPSE_K),
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
  ) dut (.*);

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

  function automatic bit close(input logic [31:0] e, input logic [31:0] a);
    real er = f32(e), ar = f32(a), d;
    d = (er > ar) ? er - ar : ar - er;
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

  function automatic logic [LFSR_WIDTH-1:0] set_seed(input int k);
    return LFSR_WIDTH'(DROPOUT_SEED ^ (32'h85EBCA6B * k));
  endfunction

  longint cyc = 0;
  always @(posedge clk_i) cyc++;

  // Per copy: the sets it was given, in order, and what it has produced for the current one.
  // Keyed by copy: Verilator dropped updates to a one-element fixed array of queues.
  int copy_sets[int][$];
  logic [DATA_WIDTH-1:0] cur[int][$];
  logic [DATA_WIDTH-1:0] got[int][$];
  longint t_start[int], t_done[int];
  int n_done = 0;

  always @(posedge clk_i) begin
    for (int c = 0; c < COPIES; c++) begin
      for (int l = 0; l < NUM_LANES; l++) if (result_valid_o[c][l]) cur[c].push_back(final_result_o[c][l]);
      if (pipeline_complete_o[c]) begin
        automatic int k = -1;
        if (copy_sets[c].size() > 0) begin
          k = copy_sets[c][0];
          copy_sets[c] = copy_sets[c][1:$];
        end
        $display("  [done] copy %0d set %0d: %0d outputs at cycle %0d", c, k, cur[c].size(), cyc);
        got[k] = cur[c];
        cur[c] = {};
        t_done[k] = cyc;
        n_done++;
      end
    end
  end

  logic [DATA_WIDTH-1:0] wq[$], nq[$], eq[$];
  int failed = 0;

  initial begin
    repeat (4) @(posedge clk_i);
    rstn_i = 1;
    repeat (4) @(posedge clk_i);
    for (int k = 0; k < NUM_SETS; k++) begin
      read_mem_file($sformatf("matrix_west_%0d.mem", k), wq);
      read_mem_file($sformatf("matrix_north_%0d.mem", k), nq);
      while (!pipeline_ready_o) @(posedge clk_i);
      fork
        begin
          for (int i = 0; i < wq.size(); i += HOST_WORDS) begin
            west_write_enable_i = 1;
            for (int w = 0; w < HOST_WORDS; w++) west_write_data_i[w] = (i + w < wq.size()) ? wq[i+w] : '0;
            @(posedge clk_i);
          end
          west_write_enable_i = 0;
        end
        begin
          for (int i = 0; i < nq.size(); i += HOST_WORDS) begin
            north_write_enable_i = 1;
            for (int w = 0; w < HOST_WORDS; w++) north_write_data_i[w] = (i + w < nq.size()) ? nq[i+w] : '0;
            @(posedge clk_i);
          end
          north_write_enable_i = 0;
        end
      join
      dropout_seed_i = set_seed(k);
      copy_sets[int'(copy_sel_o)].push_back(k);
      $display("  [start] set %0d to copy %0d at cycle %0d", k, copy_sel_o, cyc);
      t_start[k] = cyc;
      if (!pipeline_ready_o) begin
        failed++;
        $display("  [FAIL] set %0d: ready fell during its own load", k);
      end
      start_pipeline_i = 1;
      @(posedge clk_i);
      start_pipeline_i = 0;
    end
    while (n_done < NUM_SETS && cyc < TIMEOUT_CYCLES) @(posedge clk_i);
    @(posedge clk_i);
    if (n_done < NUM_SETS) begin
      failed++;
      $display("  [FAIL] only %0d of %0d sets completed", n_done, NUM_SETS);
    end
    for (int k = 0; k < NUM_SETS; k++) begin
      automatic int errs = 0;
      read_mem_file($sformatf("expected_output_%0d.mem", k), eq);
      if (!got.exists(k) || got[k].size() != eq.size()) begin
        failed++;
        $display("  [FAIL] set %0d: %0d outputs, expected %0d", k, got.exists(k) ? got[k].size() : -1, eq.size());
        continue;
      end
      for (int i = 0; i < eq.size(); i++) if (!close(eq[i], got[k][i])) errs++;
      if (errs) begin
        failed++;
        $display("  [FAIL] set %0d: %0d of %0d outputs out of tolerance", k, errs, eq.size());
      end
      $display("  set %0d: latency %0d cycles, done at %0d, %0d outputs, %0d mismatches", k,
               t_done.exists(k) ? t_done[k] - t_start[k] : -1, t_done.exists(k) ? t_done[k] : -1, eq.size(), errs);
    end
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
    $display(failed == 0 ? "RESULT: PASSED" : "RESULT: FAILED");
    $finish;
  end
endmodule
