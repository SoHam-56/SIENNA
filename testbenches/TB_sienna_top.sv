`timescale 1ns / 100ps

import test_config_pkg::*;

// ─────────────────────────────────────────────────────────────────────────────
// TB_sienna_top
// ─────────────────────────────────────────────────────────────────────────────
// Progress reporting added:
//   • Stage banners  : reset, load, start, wait, verify
//   • Heartbeat      : prints every HEARTBEAT_CYCLES while waiting for complete
//   • Status signals : systolic_busy, gpnae_busy, maxpool_busy, dropout_busy
//                      printed on every heartbeat so you can see which stage
//                      the DUT is stuck in without needing a waveform
//   • Timeout        : TIMEOUT_CYCLES (default 2M) with a clear fatal message
//
// Tracing opt-in:
//   make verilator             → no waveform
//   make verilator TRACE=fst   → FST waveform (compact, recommended)
//   make verilator TRACE=vcd   → VCD waveform (larger)
// ─────────────────────────────────────────────────────────────────────────────

module TB_sienna_top;

  // ── Localparams from generated SV package ─────────────────────────────
  localparam string WEST_INPUT_FILE = "matrix_west.mem";
  localparam string NORTH_INPUT_FILE = "matrix_north.mem";
  localparam string EXPECTED_OUTPUT_FILE = "expected_output.mem";

  localparam ADDR_LINES = $clog2(FIFO_DEPTH);
  localparam int NUM_LANES = 8;

  // ── Timeout / heartbeat ───────────────────────────────────────────────
  // Both expressed in clock cycles (10 ns each).
  // N=8 should finish in well under 10k cycles; 2M is a very safe ceiling.
  localparam int TIMEOUT_CYCLES = 200_000;
  localparam int HEARTBEAT_CYCLES = 5_000;  // print status every 50k cycles

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
      // .INPUT_A_FILE     (),
      // .INPUT_B_FILE     (),
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
      .systolic_busy_o            (systolic_busy_tb),
      .gpnae_busy_o               (gpnae_busy_tb),
      .maxpool_busy_o             (maxpool_busy_tb),
      .dropout_busy_o             (dropout_busy_tb),
      .intermediate_buffer_full_o (intermediate_buffer_full_tb),
      .intermediate_buffer_empty_o(intermediate_buffer_empty_tb)
  );

  // ── Tolerance check ───────────────────────────────────────────────────
  function automatic logic check_tolerance(input [DATA_WIDTH-1:0] expected, actual,
                                           output string info);
    // Store in shortreal first so $bitstoshortreal reinterprets 32 bits
    // correctly without 64-bit promotion swallowing the mantissa.
    shortreal exp_sr, act_sr, abs_sr;
    real abs_d, rel_d;

    exp_sr = $bitstoshortreal(expected[31:0]);
    act_sr = $bitstoshortreal(actual[31:0]);
    abs_sr = (exp_sr > act_sr) ? (exp_sr - act_sr) : (act_sr - exp_sr);
    abs_d  = real'(abs_sr);

    if (exp_sr != shortreal'(0.0))
      rel_d = abs_d / real'((exp_sr > shortreal'(0.0)) ? exp_sr : -exp_sr);
    else rel_d = (act_sr == shortreal'(0.0)) ? 0.0 : 1.0;

    // Show decoded float values so mismatches are human-readable
    info = $sformatf(
        "exp=%f act=%f abs=%.5f (lim %.4f) rel=%.3f%% (lim %.1f%%)",
        real'(exp_sr),
        real'(act_sr),
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

  // ── Collect outputs — with heartbeat ─────────────────────────────────
  task automatic collect_outputs();
    automatic int waited = 0;
    actual_results.delete();

    $display("\n[STAGE] Waiting for pipeline_complete_o");
    $display("  (heartbeat every %0d cycles, timeout at %0d cycles)", HEARTBEAT_CYCLES,
             TIMEOUT_CYCLES);

    // ── Wait loop with heartbeat ──────────────────────────────────────
    while (!pipeline_complete_o) begin
      @(posedge clk_i);
      waited++;

      if (waited % HEARTBEAT_CYCLES == 0)
        print_status($sformatf("still waiting (%0d / %0d cycles)", waited, TIMEOUT_CYCLES));

      if (waited >= TIMEOUT_CYCLES) begin
        $display("[FATAL] Timeout after %0d cycles - pipeline_complete_o never asserted.", waited);
        $display("  Last known state:");
        print_status("at timeout");
        $display("  Likely causes:");
        $display("    * FSM stuck - check which busy signal is still high");
        $display("    * num_terms_i or activation_function_i mismatch");
        $display("    * FIFO never drained (check buf_empty above)");
        $finish;
      end
    end

    $display("  pipeline_complete_o asserted @ %0t  (%0d cycles)", $time, waited);
    print_status("at completion");

    // ── Capture output words ──────────────────────────────────────────
    $display("\n[STAGE] Capturing output (%0d expected words)", expected_results.size());
    begin
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        actual_results.push_back(final_result_o[lane]);

        if (actual_results.size() >= expected_results.size()) break;
      end
    end
    $display("  Captured %0d words", actual_results.size());
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

  // ── Intermediate Value Monitor (File Output) ──────────────────────────
  integer trace_fd;

  initial begin
    // Open the file in the testbenches directory
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
    if (rstn_i && trace_fd) begin
      // Monitor FIFO 1 (Systolic Array -> GPNAE)
      // NOTE: fifo1 now exposes a ready/valid interface (fwft rework), so
      // the old dut.fifo1_rd_en / !dut.fifo1_empty pairing no longer
      // exists on the DUT. A transfer fires exactly when rd_ready and
      // rd_valid are both high on the same edge, same as the old
      // rd_en && !empty condition.
      if (dut.fifo1_rd_ready && dut.fifo1_rd_valid) begin
        $fdisplay(trace_fd, "[%0t] Systolic -> GPNAE   : dec=%.6f  hex=%08x", $time,
                  real'($bitstoshortreal(dut.fifo1_rd_data)), dut.fifo1_rd_data);
      end

      // Monitor FIFO2 (GPNAE -> Maxpool), one instance per lane. A transfer
      // fires exactly when rd_ready and rd_valid are both high on the same
      // edge, same convention as the FIFO1 monitor above. Each line is
      // tagged with lane=<idx> so the Python trace dumper can split the
      // capture back out per lane.
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.fifo2_rd_ready[lane] && dut.fifo2_rd_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] GPNAE -> Maxpool    lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.fifo2_rd_data[lane])),
                    dut.fifo2_rd_data[lane]);
        end
      end

      // Monitor FIFO3 (Maxpool -> Dropout), one instance per lane.
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.fifo3_rd_ready[lane] && dut.fifo3_rd_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] Maxpool -> Dropout  lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.fifo3_rd_data[lane])),
                    dut.fifo3_rd_data[lane]);
        end
      end
    end
  end

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

    // ── Load files ───────────────────────────────────────────────────
    $display("\n[STAGE] Reading .mem files");
    read_mem_file(WEST_INPUT_FILE, west_data_queue);
    read_mem_file(NORTH_INPUT_FILE, north_data_queue);
    read_mem_file(EXPECTED_OUTPUT_FILE, expected_results);

    $display("  West=%0d  North=%0d  Expected=%0d words", west_data_queue.size(),
             north_data_queue.size(), expected_results.size());

    // ── Reset → Load → Start ─────────────────────────────────────────
    reset();
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

    // ── Wait → Capture → Verify ──────────────────────────────────────
    collect_outputs();
    verify_outputs();

    // ── Final report ─────────────────────────────────────────────────
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

    // Add this to close the trace file safely
    if (trace_fd) $fclose(trace_fd);

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
