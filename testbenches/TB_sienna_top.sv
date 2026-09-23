`timescale 1ns / 100ps

import test_config_pkg::*;

module TB_sienna_top;

  // ── Localparams from generated SV package ─────────────────────────────
  localparam string WEST_INPUT_FILE = "matrix_west.mem";
  localparam string NORTH_INPUT_FILE = "matrix_north.mem";
  localparam string EXPECTED_OUTPUT_FILE = "expected_output.mem";

  localparam ADDR_LINES = $clog2(FIFO_DEPTH);
  localparam int NUM_LANES = 16;

  // ── Timeout / heartbeat ───────────────────────────────────────────────
  localparam int TIMEOUT_CYCLES = 200_000;
  localparam int HEARTBEAT_CYCLES = 5_000;

  // ── Tolerance ─────────────────────────────────────────────────────────
  localparam string TOLERANCE_MODE = "RELATIVE";
  localparam real ABS_TOL = 0.001;
  localparam real REL_TOL = 0.01;

  // ── DUT I/O ───────────────────────────────────────────────────────────
  logic clk_i, rstn_i;
  logic                     start_pipeline_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [     ADDR_LINES:0] num_terms_i;

  logic north_write_enable_i, north_write_reset_i;
  logic [DATA_WIDTH-1:0] north_write_data_i;
  logic west_write_enable_i, west_write_reset_i;
  logic [DATA_WIDTH-1:0]                 west_write_data_i;

  logic [ NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
  logic                                  pipeline_complete_o;
  logic                                  pipeline_ready_o;
  logic                        [    1:0] done_set_id_o;
  logic systolic_busy_tb, gpnae_busy_tb;
  logic maxpool_busy_tb, dropout_busy_tb;
  logic intermediate_buffer_full_tb, intermediate_buffer_empty_tb;

  // ── Data queues ───────────────────────────────────────────────────────
  logic [DATA_WIDTH-1:0] north_data_queue[$];
  logic [DATA_WIDTH-1:0] west_data_queue [$];
  logic [DATA_WIDTH-1:0] expected_results[$];
  logic [DATA_WIDTH-1:0] actual_results  [$];

  // ── Verification counters ─────────────────────────────────────────────
  int total_elements = 0, exact_passed = 0, tol_passed = 0, failed = 0;

  // ── Cycle counter (free-running, for heartbeat timestamps) ────────────
  longint cycle_count = 0;
  always_ff @(posedge clk_i) cycle_count <= cycle_count + 1;

  // ── Clock ─────────────────────────────────────────────────────────────
  initial begin
    clk_i = 0;
    forever #5 clk_i = ~clk_i;
  end

  // ── DUT ───────────────────────────────────────────────────────────────
  sienna_top #(
      .NUM_LANES        (NUM_LANES),
      .N                (N),
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
      .activation_function_i      (activation_function_i),
      .num_terms_i                (num_terms_i),
      .north_write_enable_i       (north_write_enable_i),
      .north_write_data_i         (north_write_data_i),
      .north_write_reset_i        (north_write_reset_i),
      .west_write_enable_i        (west_write_enable_i),
      .west_write_data_i          (west_write_data_i),
      .west_write_reset_i         (west_write_reset_i),
      .final_result_o             (final_result_o),
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

  // Manual binary32 decode. $signed() and $bitstoshortreal both leave the bit pattern as an integer
  // under Verilator, which turns a 1% bound into roughly a factor of two.
  function automatic real f32(input logic [31:0] b);
    int  e;
    real m, v;
    e = int'(b[30:23]);
    m = real'(longint'(b[22:0])) / 8388608.0;
    if (e == 255) v = 1.0e38;                         // Inf / NaN, clamped so any finite compare fails
    else if (e == 0) v = 0.0;                         // zero / flushed subnormal
    else v = (1.0 + m) * (2.0 ** (e - 127));
    return b[31] ? -v : v;
  endfunction

  // ── Tolerance check ───────────────────────────────────────────────────
  function automatic logic check_tolerance(input [DATA_WIDTH-1:0] expected, actual,
                                           output string info);
    real exp_r, act_r, abs_d, rel_d;

    exp_r = f32(expected[31:0]);
    act_r = f32(actual[31:0]);
    abs_d = (exp_r > act_r) ? (exp_r - act_r) : (act_r - exp_r);

    if (exp_r != 0.0) rel_d = abs_d / ((exp_r > 0.0) ? exp_r : -exp_r);
    else rel_d = (act_r == 0.0) ? 0.0 : 1.0;

    info = $sformatf(
        "exp=%g act=%g abs=%.3g (lim %.4f) rel=%.4f%% (lim %.1f%%)",
        exp_r,
        act_r,
        abs_d,
        ABS_TOL,
        rel_d * 100.0,
        REL_TOL * 100.0
    );

    case (TOLERANCE_MODE)
      "ABSOLUTE": return (abs_d <= ABS_TOL);
      "RELATIVE": return (rel_d <= REL_TOL);
      "BOTH":     return (abs_d <= ABS_TOL) && (rel_d <= REL_TOL);
      default:    return (abs_d <= ABS_TOL);
    endcase
  endfunction

  // Checker self-test: a loose or broken compare must fail the run before any result is trusted.
  initial begin
    string st_info;
    if (!check_tolerance(32'h3f800000, 32'h3f800003, st_info) ||   // 1.0 vs 1.0 + 3 ulp: pass
        check_tolerance(32'h3f800000, 32'h40000000, st_info) ||    // 1.0 vs 2.0: fail
        check_tolerance(32'h3f800000, 32'h3f7ae148, st_info) ||    // 1.0 vs 0.98: fail
        check_tolerance(32'h3f800000, 32'hbf800000, st_info) ||    // 1.0 vs -1.0: fail
        check_tolerance(32'hbf000000, 32'hbd4ccccd, st_info)) begin // -0.5 vs -0.05: fail
      $display("[FAIL] Tolerance checker self-test failed; results cannot be trusted");
      $finish;
    end
  end

  // ── .mem reader ───────────────────────────────────────────────────────
  task automatic read_mem_file(input string fn, output logic [DATA_WIDTH-1:0] q[$]);
    integer fh, rc;
    logic [DATA_WIDTH-1:0] w;
    q.delete();
    fh = $fopen(fn, "r");
    if (!fh) begin
      $display("[ERROR] Cannot open: %s", fn);
      $finish;
    end
    while (!$feof(
        fh
    )) begin
      rc = $fscanf(fh, "%h", w);
      if (rc == 1) q.push_back(w);
    end
    $fclose(fh);
  endtask

  // ── Helper: print current DUT status signals ──────────────────────────
  // Armed for the back-to-back pass only: every outer-FSM transition with its timestamp.
  logic trace_states = 0;
  int   prev_state = -1;
  always @(posedge clk_i) begin
    if (trace_states && dut.current_state !== prev_state) begin
      $display("  [FSM] @%0t state %0d -> %0d  (coll_complete=%0b mult_complete=%0b)",
               $time, prev_state, dut.current_state,
               dut.systolic_collection_complete, dut.systolic_mult_complete);
      prev_state = dut.current_state;
    end
  end

  task automatic print_status(input string stage_label);
    $display(
        "[STATUS @ %0t] sys_busy=%0b gpnae_busy=%0b max_busy=%0b drop_busy=%0b | full=%0b empty=%0b",
        $time, systolic_busy_tb, gpnae_busy_tb, maxpool_busy_tb, dropout_busy_tb,
        intermediate_buffer_full_tb, intermediate_buffer_empty_tb);
  endtask

  // ── Reset ─────────────────────────────────────────────────────────────
  task automatic reset();
    $display("\n[STAGE] Reset");
    rstn_i = 0;
    start_pipeline_i = 0;
    north_write_reset_i = 1;
    west_write_reset_i = 1;
    north_write_enable_i = 0;
    west_write_enable_i = 0;
    north_write_data_i = '0;
    west_write_data_i = '0;
    activation_function_i = '0;
    num_terms_i = '0;
    repeat (10) @(posedge clk_i);
    rstn_i = 1;
    north_write_reset_i = 0;
    west_write_reset_i = 0;
    repeat (5) @(posedge clk_i);
    $display("  Reset complete @ %0t", $time);
  endtask

  // ── Load inputs ───────────────────────────────────────────────────────
  task automatic load_inputs();
    $display("\n[STAGE] Loading inputs");
    $display("  West  queue : %0d words", west_data_queue.size());
    $display("  North queue : %0d words", north_data_queue.size());

    fork
      begin
        foreach (west_data_queue[i]) begin
          west_write_enable_i = 1;
          west_write_data_i   = west_data_queue[i];
          @(posedge clk_i);
        end
        west_write_enable_i = 0;
        @(posedge clk_i);
      end
      begin
        foreach (north_data_queue[i]) begin
          north_write_enable_i = 1;
          north_write_data_i   = north_data_queue[i];
          @(posedge clk_i);
        end
        north_write_enable_i = 0;
        @(posedge clk_i);
      end
    join
    $display("  Load complete @ %0t  (%0d cycles)", $time, cycle_count);
  endtask

  // =========================================================================
  // ON-THE-FLY STREAMING CAPTURE
  // =========================================================================
  always_ff @(posedge clk_i) begin
    if (rstn_i && !pipeline_complete_o) begin
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.dropout_valid_out[lane]) begin
          actual_results.push_back(dut.dropout_data_out[lane]);
        end
      end
    end
  end

  // ── Collect outputs — with heartbeat ─────────────────────────────────
  task automatic collect_outputs();
    automatic int waited = 0;

    $display("\n[STAGE] Waiting for pipeline_complete_o");
    $display("  (heartbeat every %0d cycles, timeout at %0d cycles)", HEARTBEAT_CYCLES,
             TIMEOUT_CYCLES);

    // Wait loop with heartbeat
    while (!pipeline_complete_o) begin
      @(posedge clk_i);
      waited++;

      if (waited % HEARTBEAT_CYCLES == 0)
        print_status($sformatf("still waiting (%0d / %0d cycles)", waited, TIMEOUT_CYCLES));

      if (waited >= TIMEOUT_CYCLES) begin
        $display("[FATAL] Timeout after %0d cycles - pipeline_complete_o never asserted.", waited);
        $display("  Last known state:");
        print_status("at timeout");
        // Which stage is stuck is the whole diagnosis; sys_busy alone does not say.
        $display("  outer FSM current_state = %0d", dut.current_state);
        $display("  fill_state=%0d  filled_total=%0d  total_elements=%0d  all_collected=%0b",
                 dut.fill_state, dut.filled_total, dut.total_elements, dut.all_collected);
        $display("  systolic_collection_complete=%0b  fifo1_count=%0d  disp_done=%0b",
                 dut.systolic_collection_complete, dut.fifo1_count, dut.disp_done);
        // IDLE only exits on start && both mesh queues non-empty; print that gate verbatim.
        $display("  start gate: start_pipeline_i=%0b north_queue_empty=%0b west_queue_empty=%0b",
                 start_pipeline_i, dut.north_queue_empty, dut.west_queue_empty);
        for (int i = 0; i < 4; i++)
          $display("  lane %0d: fill_count=%0d done_count=%0d load_finalized=%0b collected=%0b",
                   i, dut.fill_count[i], dut.done_count[i], dut.load_finalized[i],
                   dut.lane_collected[i]);
        if (lane_fd) $fclose(lane_fd);
        $finish;
      end
    end

    $display("  pipeline_complete_o asserted @ %0t  (%0d cycles)", $time, waited);
    print_status("at completion");
    $display("  Captured %0d words out of %0d expected", actual_results.size(),
             expected_results.size());
  endtask

  // ── Verify ───────────────────────────────────────────────────────────
  task automatic verify_outputs();
    automatic int n_exp = expected_results.size();
    automatic int n_act = actual_results.size();
    automatic logic [DATA_WIDTH-1:0] ev, av;
    string info;

    $display("\n[STAGE] Verifying - %0d expected, %0d captured", n_exp, n_act);

    for (int i = 0; i < n_exp && i < n_act; i++) begin
      ev = expected_results[i];
      av = actual_results[i];
      total_elements++;
      if (ev === av) begin
        exact_passed++;
      end else if (check_tolerance(ev, av, info)) begin
        tol_passed++;
        $display("  [PASS-TOL] [%0d] exp=0x%h act=0x%h | %s", i, ev, av, info);
      end else begin
        failed++;
        $display("  [FAIL]     [%0d] exp=0x%h act=0x%h | %s", i, ev, av, info);
      end
    end

    // Elements that were expected but never captured
    for (int i = n_act; i < n_exp; i++) begin
      total_elements++;
      failed++;
      $display("  [FAIL]     [%0d] exp=0x%h  act=MISSING", i, expected_results[i]);
    end
  endtask

  // ── File and Console Output Monitors ──────────────────────────────────
  integer trace_fd;

  initial begin
    trace_fd = $fopen("../testbenches/hardware_trace.txt", "w");
    if (!trace_fd) begin
      $display("[ERROR] Could not open hardware_trace.txt for writing.");
    end else begin
      $fdisplay(trace_fd, "==============================================");
      $fdisplay(trace_fd, " SIENNA Hardware Intermediate Trace");
      $fdisplay(trace_fd, "==============================================\n");
    end
  end

  always @(posedge clk_i) begin
    if (rstn_i) begin
      // Console Print: Dispatcher data entering Maxpool
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.fifo2_wr_valid[lane]) begin
          $display("[DEBUG %0t] Dispatcher -> Maxpool (Lane %0d) : dec=%.4f  hex=%08x", $time, lane,
                   real'($bitstoshortreal(dut.fifo2_wr_data[lane])), dut.fifo2_wr_data[lane]);
        end
      end
      // Console Print: Data exiting Dropout to final Output
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.dropout_valid_out[lane]) begin
          $display("[DEBUG %0t] Dropout -> Output (Lane %0d)   : dec=%.4f  hex=%08x", $time, lane,
                   real'($bitstoshortreal(dut.dropout_data_out[lane])), dut.dropout_data_out[lane]);
        end
      end
    end
  end

  always @(posedge clk_i) begin
    if (rstn_i && trace_fd) begin
      if (dut.fifo1_rd_ready && dut.fifo1_rd_valid) begin
        $fdisplay(trace_fd, "[%0t] Systolic -> GPNAE   : dec=%.6f  hex=%08x", $time,
                  real'($bitstoshortreal(dut.fifo1_rd_data)), dut.fifo1_rd_data);
      end
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.fifo2_rd_ready[lane] && dut.fifo2_rd_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] GPNAE -> Maxpool    lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.fifo2_rd_data[lane])),
                    dut.fifo2_rd_data[lane]);
        end
      end
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.dropout_in_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] Maxpool -> Dropout  lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.dropout_data_in[lane])),
                    dut.dropout_data_in[lane]);
        end
      end
    end
  end

  integer       lane_fd;
  logic   [1:0] prev_mp_state      [NUM_LANES];
  logic   [3:0] prev_current_state;
  logic         lane_log_init_done;

  initial begin
    lane_fd = $fopen("../testbenches/pipeline_lane_status.txt", "w");
    if (!lane_fd) begin
      $display("[ERROR] Could not open pipeline_lane_status.txt for writing.");
    end else begin
      $fdisplay(lane_fd, "==============================================");
      $fdisplay(lane_fd, " SIENNA Per-Lane Streaming Status");
      $fdisplay(lane_fd, " mp_state key: 0=MP_IDLE 1=MP_FEED 2=MP_WAIT_DONE 3=MP_DONE");
      $fdisplay(lane_fd, "==============================================\n");
    end
    lane_log_init_done = 1'b0;
  end

  task automatic print_lane_status(input string reason);
    if (!lane_fd) return;
    $fdisplay(lane_fd,
              "---- %0t  (%s)  current_state=%0d  all_collected=%0b  streaming_complete=%0b ----",
              $time, reason, dut.current_state, dut.all_collected, dut.streaming_complete);
    // Note: mp_real_consumed explicitly removed to sync with current top-level architecture
    for (int lane = 0; lane < NUM_LANES; lane++) begin
      $fdisplay(
          lane_fd,
          "  lane=%0d fill_count=%0d done_count=%0d load_finalized=%0b lane_collected=%0b | mp_state=%0d mp_window_fed=%0d mp_windows_done=%0d lane_windows_total=%0d | dropout_out_count=%0d",
          lane, dut.fill_count[lane], dut.done_count[lane], dut.load_finalized[lane],
          dut.lane_collected[lane], dut.mp_state[lane], dut.mp_window_fed[lane],
          dut.mp_windows_done[lane], dut.lane_windows_total[lane], dut.dropout_out_count[lane]);
    end
  endtask

  always @(posedge clk_i) begin
    if (rstn_i && lane_fd) begin
      if (!lane_log_init_done) begin
        for (int lane = 0; lane < NUM_LANES; lane++) prev_mp_state[lane] <= dut.mp_state[lane];
        prev_current_state <= dut.current_state;
        lane_log_init_done <= 1'b1;
      end else begin
        automatic logic any_mp_state_changed = 1'b0;
        for (int lane = 0; lane < NUM_LANES; lane++) begin
          if (dut.mp_state[lane] != prev_mp_state[lane]) any_mp_state_changed = 1'b1;
          prev_mp_state[lane] <= dut.mp_state[lane];
        end

        if (dut.current_state != prev_current_state) begin
          print_lane_status("current_state changed");
        end else if (any_mp_state_changed) begin
          print_lane_status("mp_state changed");
        end else if (cycle_count % HEARTBEAT_CYCLES == 0) begin
          print_lane_status("heartbeat");
        end

        prev_current_state <= dut.current_state;
      end
    end
  end

  // ── Streaming: K distinct sets through overlapped stages ──────────────
  // The monitor reads registered state on the falling edge: TB inputs set after a rising
  // edge are taken at that edge under Verilator, so a combinational view is a cycle late.
  bit stream_on = 0;
  int ov_mesh_g = 0, ov_g_p = 0, max_in_flight = 0, n_started = 0;
  logic [DATA_WIDTH-1:0] stream_results[$];
  int stream_bounds[$];
  int stream_ids[$];
  wire mesh_computing = (int'(dut.systolic_array_inst.current_state) != 0) &&
                        (int'(dut.systolic_array_inst.current_state) != 7);  // not IDLE, not DONE

  always_ff @(posedge clk_i) begin
    if (stream_on) begin
      for (int lane = 0; lane < NUM_LANES; lane++)
        if (dut.dropout_valid_out[lane]) stream_results.push_back(dut.dropout_data_out[lane]);
      if (pipeline_complete_o) begin
        stream_bounds.push_back(stream_results.size());
        stream_ids.push_back(int'(done_set_id_o));
      end
    end
  end

  initial forever begin
    @(negedge clk_i);
    if (stream_on) begin
      if (mesh_computing && gpnae_busy_tb) ov_mesh_g++;
      if (gpnae_busy_tb && maxpool_busy_tb) ov_g_p++;
      if (n_started - stream_bounds.size() > max_in_flight)
        max_in_flight = n_started - stream_bounds.size();
    end
  end

  task automatic verify_slice(input int k, input logic [DATA_WIDTH-1:0] exp_q[$]);
    automatic int lo = (k == 0) ? 0 : stream_bounds[k-1];
    automatic int n_act = stream_bounds[k] - lo;
    automatic int errs = 0;
    logic [DATA_WIDTH-1:0] av;
    string info;
    if (stream_ids[k] != (k % 4)) begin
      failed++;
      $display("  [FAIL] Stream set %0d completed as set id %0d", k, stream_ids[k]);
    end
    if (n_act != exp_q.size()) begin
      failed++;
      $display("  [FAIL] Stream set %0d produced %0d outputs, expected %0d", k, n_act, exp_q.size());
    end
    for (int i = 0; i < exp_q.size(); i++) begin
      total_elements++;
      if (i >= n_act) begin
        failed++;
        errs++;
        $display("  [FAIL] Stream set %0d [%0d] exp=0x%h act=MISSING", k, i, exp_q[i]);
      end else begin
        av = stream_results[lo+i];
        if (av === exp_q[i]) exact_passed++;
        else if (check_tolerance(exp_q[i], av, info)) tol_passed++;
        else begin
          failed++;
          errs++;
          $display("  [FAIL] Stream set %0d [%0d] exp=0x%h act=0x%h | %s", k, i, exp_q[i], av, info);
        end
      end
    end
    $display("  [Stream] set %0d: %0d outputs, %0d mismatches", k, n_act, errs);
  endtask

  task automatic stream_all_sets();
    automatic longint t0 = $time;
    $display("\n[STAGE] STREAMING: %0d sets through overlapped stages", NUM_SETS);
    stream_results.delete();
    stream_bounds.delete();
    stream_ids.delete();
    n_started = 0;
    while (pipeline_complete_o) @(posedge clk_i);  // the previous pass's pulse is not a set boundary
    @(posedge clk_i);
    stream_on = 1;
    fork
      begin
        fork
          begin : producer
            for (int k = 0; k < NUM_SETS; k++) begin
              read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
              read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
              while (!pipeline_ready_o) @(posedge clk_i);
              load_inputs();
              if (!pipeline_ready_o) begin
                failed++;
                $display("  [FAIL] Start pulsed while pipeline_ready_o is low");
              end
              start_pipeline_i = 1;
              @(posedge clk_i);
              start_pipeline_i = 0;
              n_started++;
              @(posedge clk_i);  // let the credit land before sampling ready again
            end
          end
          begin : verifier
            logic [DATA_WIDTH-1:0] exp_q[$];
            for (int k = 0; k < NUM_SETS; k++) begin
              while (stream_bounds.size() <= k) @(posedge clk_i);
              read_mem_file($sformatf("expected_output_%0d.mem", k), exp_q);
              verify_slice(k, exp_q);
            end
          end
        join
      end
      begin : watchdog
        repeat (TIMEOUT_CYCLES) @(posedge clk_i);
        $display("[FATAL] Timeout in the streaming pass: %0d of %0d sets completed",
                 stream_bounds.size(), NUM_SETS);
        $finish;
      end
    join_any
    disable fork;
    stream_on = 0;
    // $time is in the 1 ns timeunit, so a 10 ns clock is 10 units per cycle.
    $display("  [Stream] %0d sets in %0d cycles", NUM_SETS, ($time - t0) / 10);
    $display("  [Stream] mesh computing while activation busy: %0d cycles", ov_mesh_g);
    $display("  [Stream] activation and pooling busy together: %0d cycles", ov_g_p);
    $display("  [Stream] most sets in flight: %0d", max_in_flight);
    if (ov_mesh_g == 0) begin
      failed++;
      $display("  [FAIL] Overlap: the mesh never computed while the activation stage held a set");
    end
    if (ov_g_p == 0) begin
      failed++;
      $display("  [FAIL] Overlap: the activation and pooling stages never held sets at once");
    end
    if (max_in_flight > 3) begin
      failed++;
      $display("  [FAIL] %0d sets in flight, the credit limit is 3", max_in_flight);
    end
  endtask

  // ── Top-level stimulus ────────────────────────────────────────────────
  initial begin

    $display("==============================================");
    $display(" SIENNA PIPELINE VERIFICATION");
    $display("==============================================");
    $display(" N=%-0d  TILE_SIZE=%-0d  FIFO_DEPTH=%-0d", N, TILE_SIZE, FIFO_DEPTH);
    $display(" Activation code : %0b  Num terms : %0d", ACTIVATION_CODE, NUM_TERMS);
    $display(" Tolerance       : %s  rel<=%.1f%%  abs<=%.4f", TOLERANCE_MODE, REL_TOL * 100.0,
             ABS_TOL);
    $display(" Timeout         : %0d cycles  Heartbeat: %0d cycles", TIMEOUT_CYCLES,
             HEARTBEAT_CYCLES);
    $display("==============================================");

    $display("\n[STAGE] Reading .mem files");
    read_mem_file(WEST_INPUT_FILE, west_data_queue);
    read_mem_file(NORTH_INPUT_FILE, north_data_queue);
    read_mem_file(EXPECTED_OUTPUT_FILE, expected_results);

    $display("  West=%0d  North=%0d  Expected=%0d words", west_data_queue.size(),
             north_data_queue.size(), expected_results.size());

    reset();

    // Clear array right before run so streaming block cleanly builds it
    // actual_results.delete();

    load_inputs();
    repeat (5) @(posedge clk_i);

    $display("\n[STAGE] Starting pipeline");
    activation_function_i = ACTIVATION_CODE[CONTROL_WIDTH-1:0];
    num_terms_i           = NUM_TERMS[ADDR_LINES:0];
    @(posedge clk_i);
    start_pipeline_i = 1;
    @(posedge clk_i);
    start_pipeline_i = 0;
    $display("  start pulse sent @ %0t", $time);
    print_status("immediately after start");
    print_lane_status("immediately after start");

    collect_outputs();
    verify_outputs();

`ifdef BACK_TO_BACK
    // Run a second, identical matrix WITHOUT asserting reset. Anything that carries state
    // between matmuls shows up as a different result the second time, and the cycle delta is
    // the real steady-state cost per matrix rather than an estimate.
    begin
      int unsigned t_second_start;
      int unsigned pass1_failed;
      pass1_failed = failed;

      $display("\n[STAGE] BACK TO BACK: second matrix, no reset");
      trace_states = 1;
      actual_results.delete();
      total_elements = 0;
      exact_passed   = 0;
      tol_passed     = 0;
      failed         = 0;

      load_inputs();
      repeat (5) @(posedge clk_i);
      t_second_start = $time;

      @(posedge clk_i);
      start_pipeline_i = 1;
      $display("  [B2B] gate before pulse: state=%0d north_empty=%0b west_empty=%0b",
               dut.current_state, dut.north_queue_empty, dut.west_queue_empty);
      @(posedge clk_i);
      $display("  [B2B] gate at pulse   : state=%0d north_empty=%0b west_empty=%0b",
               dut.current_state, dut.north_queue_empty, dut.west_queue_empty);
      start_pipeline_i = 0;
      @(posedge clk_i);
      $display("  [B2B] state after pulse: %0d", dut.current_state);

      collect_outputs();
      verify_outputs();

      // $time evaluates in the 1ns timeunit (a 10ns clock is 10 units), even though %0t prints ps.
      $display(" BACK_TO_BACK second-pass cycles : %0d", ($time - t_second_start) / 10);
      if (failed == 0 && pass1_failed == 0)
        $display(" BACK_TO_BACK: PASSED (both matrices correct without an intervening reset)");
      else
        $display(" BACK_TO_BACK: FAILED (pass1 %0d, pass2 %0d mismatches)", pass1_failed, failed);
      failed = failed + pass1_failed;
    end
`endif

    stream_all_sets();

    $display("\n==============================================");
    $display(" RESULT SUMMARY");
    $display("==============================================");
    $display(" Total    : %0d", total_elements);
    $display(" Exact    : %0d", exact_passed);
    $display(" Tol pass : %0d  (rel <= %.1f%%)", tol_passed, REL_TOL * 100.0);
    $display(" Failed   : %0d", failed);
    $display("----------------------------------------------");
    if (failed == 0) $display(" RESULT: PASSED");
    else $display(" RESULT: FAILED (%0d mismatches)", failed);
    $display("==============================================");

    if (trace_fd) $fclose(trace_fd);
    if (lane_fd) $fclose(lane_fd);

    #100 $finish;
  end

  initial begin
`ifdef ENABLE_TRACE
`ifdef TRACE_FST
    $dumpfile("TB_sienna_top.fst");
`else
    $dumpfile("TB_sienna_top.vcd");
`endif
    $dumpvars(0, TB_sienna_top);
`endif
  end

endmodule
