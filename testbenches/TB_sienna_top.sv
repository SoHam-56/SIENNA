`timescale 1ns / 1ps

module TB_sienna_top;

  // Parameters matching DUT
  parameter N = 32;
  parameter DATA_WIDTH = 32;
  parameter SRAM_DEPTH = N * N;
  parameter ADDR_LINES = 5;
  parameter CONTROL_WIDTH = 2;
  parameter IN_ROWS = 5;
  parameter IN_COLS = 5;
  parameter POOL_H = 2;
  parameter POOL_W = 2;
  parameter STRIDE_ROWS = 2;
  parameter STRIDE_COLS = 2;
  parameter PADDING = 1;
  parameter real DROPOUT_P = 0.5;
  parameter LFSR_WIDTH = 32;
  parameter INPUT_A_FILE = "matrixA.mem";
  parameter INPUT_B_FILE = "matrixB.mem";
  parameter INTERMEDIATE_BUFFER_DEPTH = SRAM_DEPTH * 2;
  parameter FIFO_DEPTH = 16;

  // Test file parameters
  parameter NORTH_INPUT_FILE = "matrix_north.txt";
  parameter WEST_INPUT_FILE = "matrix_west.txt";
  parameter EXPECTED_OUTPUT_FILE = "expected_output.txt";
  parameter ACTUAL_OUTPUT_FILE = "actual_outputs.txt";
  parameter TEST_CONFIG_FILE = "test_config.txt";

  // Clock and Reset
  logic clk_i;
  logic rstn_i;

  // Top-level Control Interface
  logic start_pipeline_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [ADDR_LINES-1:0] num_terms_i;

  // Systolic Array Input Interface
  logic north_write_enable_i;
  logic [DATA_WIDTH-1:0] north_write_data_i;
  logic north_write_reset_i;
  logic west_write_enable_i;
  logic [DATA_WIDTH-1:0] west_write_data_i;
  logic west_write_reset_i;

  // Maxpool SRAM Interface (backward compatibility)
  logic [DATA_WIDTH-1:0] maxpool_input_data_i;
  logic maxpool_input_valid_i;

  // Final Output Interface
  logic [DATA_WIDTH-1:0] final_result_o;
  logic pipeline_complete_o;
  logic gpnae_done_o;

  // Status Outputs
  logic systolic_busy_o;
  logic gpnae_busy_o;
  logic maxpool_busy_o;
  logic dropout_busy_o;
  logic intermediate_buffer_full_o;
  logic intermediate_buffer_empty_o;

  // Debug/Monitoring Outputs
  logic [DATA_WIDTH-1:0] systolic_result_debug_o;
  logic systolic_complete_debug_o;
  logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer_count_debug_o;

  // Testbench variables
  integer north_file, west_file, expected_file, actual_file, config_file;
  integer scan_result, i, j;
  logic [DATA_WIDTH-1:0] north_data_queue[$];
  logic [DATA_WIDTH-1:0] west_data_queue[$];
  logic [DATA_WIDTH-1:0] expected_results[$];
  logic [DATA_WIDTH-1:0] actual_results[$];
  integer test_num;
  integer num_tests;
  integer errors;
  integer total_errors;
  logic test_passed;

  // Performance monitoring
  longint start_time, end_time, total_cycles;

  // State tracking for better debug visibility
  string current_phase;

  // State enum for monitoring (matching DUT)
  typedef enum logic [3:0] {
    IDLE,
    SYSTOLIC_PROCESSING,
    FEED_GPNAE,
    GPNAE_PROCESSING,
    COLLECT_GPNAE,
    FILL_MAXPOOL_SRAM,
    MAXPOOL_PROCESSING,
    COLLECT_MAXPOOL,
    DROPOUT_PROCESSING,
    PIPELINE_COMPLETE
  } pipeline_state_t;

  // Clock generation
  initial begin
    clk_i = 0;
    forever #5 clk_i = ~clk_i;  // 100MHz clock (10ns period)
  end

  // DUT instantiation
  sienna_top #(
      .N(N),
      .DATA_WIDTH(DATA_WIDTH),
      .SRAM_DEPTH(SRAM_DEPTH),
      .ADDR_LINES(ADDR_LINES),
      .CONTROL_WIDTH(CONTROL_WIDTH),
      .IN_ROWS(IN_ROWS),
      .IN_COLS(IN_COLS),
      .POOL_H(POOL_H),
      .POOL_W(POOL_W),
      .STRIDE_ROWS(STRIDE_ROWS),
      .STRIDE_COLS(STRIDE_COLS),
      .PADDING(PADDING),
      .DROPOUT_P(DROPOUT_P),
      .LFSR_WIDTH(LFSR_WIDTH),
      .INPUT_A_FILE(INPUT_A_FILE),
      .INPUT_B_FILE(INPUT_B_FILE),
      .INTERMEDIATE_BUFFER_DEPTH(INTERMEDIATE_BUFFER_DEPTH),
      .FIFO_DEPTH(FIFO_DEPTH)
  ) dut (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .start_pipeline_i(start_pipeline_i),
      .activation_function_i(activation_function_i),
      .num_terms_i(num_terms_i),
      .north_write_enable_i(north_write_enable_i),
      .north_write_data_i(north_write_data_i),
      .north_write_reset_i(north_write_reset_i),
      .west_write_enable_i(west_write_enable_i),
      .west_write_data_i(west_write_data_i),
      .west_write_reset_i(west_write_reset_i),
      .maxpool_input_data_i(maxpool_input_data_i),
      .maxpool_input_valid_i(maxpool_input_valid_i),
      .final_result_o(final_result_o),
      .pipeline_complete_o(pipeline_complete_o),
      .gpnae_done_o(gpnae_done_o),
      .systolic_busy_o(systolic_busy_o),
      .gpnae_busy_o(gpnae_busy_o),
      .maxpool_busy_o(maxpool_busy_o),
      .dropout_busy_o(dropout_busy_o),
      .intermediate_buffer_full_o(intermediate_buffer_full_o),
      .intermediate_buffer_empty_o(intermediate_buffer_empty_o),
      .systolic_result_debug_o(systolic_result_debug_o),
      .systolic_complete_debug_o(systolic_complete_debug_o),
      .buffer_count_debug_o(buffer_count_debug_o)
  );

  // Task to generate simple test data if files don't exist
  task automatic generate_test_data();
    logic [DATA_WIDTH-1:0] data;

    $display("Generating simple test data...");

    // Generate north inputs (column-wise for matrix B)
    north_data_queue.delete();
    for (int idx = 0; idx < N * N; idx++) begin
      data = idx + 1;  // Simple incrementing pattern
      north_data_queue.push_back(data);
    end
    $display("Generated %0d north inputs", north_data_queue.size());

    // Generate west inputs (row-wise for matrix A)
    west_data_queue.delete();
    for (int idx = 0; idx < N * N; idx++) begin
      data = (idx + 1) * 2;  // Different pattern
      west_data_queue.push_back(data);
    end
    $display("Generated %0d west inputs", west_data_queue.size());

    // Generate expected output (placeholder)
    expected_results.delete();
    data = 32'hDEADBEEF;
    expected_results.push_back(data);
    $display("Generated %0d expected results", expected_results.size());
  endtask

  // Task to load input data from files
  task automatic load_input_files();
    logic [DATA_WIDTH-1:0] data;
    logic files_exist;

    files_exist = 1;

    // Try to load north inputs
    north_file  = $fopen(NORTH_INPUT_FILE, "r");
    if (north_file == 0) begin
      $display("WARNING: Could not open %s", NORTH_INPUT_FILE);
      files_exist = 0;
    end else begin
      north_data_queue.delete();
      while (!$feof(
          north_file
      )) begin
        scan_result = $fscanf(north_file, "%h\n", data);
        if (scan_result == 1) begin
          north_data_queue.push_back(data);
        end
      end
      $fclose(north_file);
      $display("Loaded %0d north inputs from %s", north_data_queue.size(), NORTH_INPUT_FILE);
    end

    // Try to load west inputs
    west_file = $fopen(WEST_INPUT_FILE, "r");
    if (west_file == 0) begin
      $display("WARNING: Could not open %s", WEST_INPUT_FILE);
      files_exist = 0;
    end else begin
      west_data_queue.delete();
      while (!$feof(
          west_file
      )) begin
        scan_result = $fscanf(west_file, "%h\n", data);
        if (scan_result == 1) begin
          west_data_queue.push_back(data);
        end
      end
      $fclose(west_file);
      $display("Loaded %0d west inputs from %s", west_data_queue.size(), WEST_INPUT_FILE);
    end

    // Try to load expected outputs
    expected_file = $fopen(EXPECTED_OUTPUT_FILE, "r");
    if (expected_file == 0) begin
      $display("WARNING: Could not open %s", EXPECTED_OUTPUT_FILE);
      files_exist = 0;
    end else begin
      expected_results.delete();
      while (!$feof(
          expected_file
      )) begin
        scan_result = $fscanf(expected_file, "%h\n", data);
        if (scan_result == 1) begin
          expected_results.push_back(data);
        end
      end
      $fclose(expected_file);
      $display("Loaded %0d expected outputs from %s", expected_results.size(),
               EXPECTED_OUTPUT_FILE);
    end

    // If files don't exist, generate test data
    if (!files_exist) begin
      generate_test_data();
    end
  endtask

  // Task to load test configuration
  task automatic load_test_config();
    config_file = $fopen(TEST_CONFIG_FILE, "r");
    if (config_file == 0) begin
      $display("WARNING: Could not open %s, using default config", TEST_CONFIG_FILE);
      num_tests = 1;
      activation_function_i = 2'b10;  // Sigmoid
      num_terms_i = 5'd16;  // 16 terms for GPNAE
    end else begin
      scan_result = $fscanf(config_file, "%d\n", num_tests);
      scan_result = $fscanf(config_file, "%b\n", activation_function_i);
      scan_result = $fscanf(config_file, "%h\n", num_terms_i);
      $fclose(config_file);
      $display("Loaded test configuration: %0d tests, activation=%b, terms=%0d", num_tests,
               activation_function_i, num_terms_i);
    end
  endtask

  // Task to write north inputs
  task automatic write_north_inputs();
    $display("[%0t] Writing %0d north inputs...", $time, north_data_queue.size());
    current_phase = "LOADING_NORTH";

    north_write_reset_i = 1;
    @(posedge clk_i);
    north_write_reset_i = 0;
    @(posedge clk_i);

    foreach (north_data_queue[idx]) begin
      north_write_enable_i = 1;
      north_write_data_i   = north_data_queue[idx];
      @(posedge clk_i);

      // Display first few and last few for verification
      if (idx < 5 || idx >= north_data_queue.size() - 5) begin
        $display("  North[%0d] = 0x%08h", idx, north_data_queue[idx]);
      end else if (idx == 5) begin
        $display("  ... (%0d more entries) ...", north_data_queue.size() - 10);
      end
    end

    north_write_enable_i = 0;
    $display("[%0t] North inputs written successfully", $time);
  endtask

  // Task to write west inputs
  task automatic write_west_inputs();
    $display("[%0t] Writing %0d west inputs...", $time, west_data_queue.size());
    current_phase = "LOADING_WEST";

    west_write_reset_i = 1;
    @(posedge clk_i);
    west_write_reset_i = 0;
    @(posedge clk_i);

    foreach (west_data_queue[idx]) begin
      west_write_enable_i = 1;
      west_write_data_i   = west_data_queue[idx];
      @(posedge clk_i);

      // Display first few and last few for verification
      if (idx < 5 || idx >= west_data_queue.size() - 5) begin
        $display("  West[%0d] = 0x%08h", idx, west_data_queue[idx]);
      end else if (idx == 5) begin
        $display("  ... (%0d more entries) ...", west_data_queue.size() - 10);
      end
    end

    west_write_enable_i = 0;
    $display("[%0t] West inputs written successfully", $time);
  endtask

  // Task to collect outputs
  task automatic collect_outputs();
    $display("[%0t] Waiting for pipeline completion...", $time);
    current_phase = "PROCESSING";
    actual_results.delete();

    start_time = $time;

    // Wait for pipeline to complete with timeout
    fork
      begin
        wait (pipeline_complete_o);
        end_time = $time;
        total_cycles = (end_time - start_time) / 10;  // 10ns period
        $display("[%0t] Pipeline completed in %0d cycles!", $time, total_cycles);
      end
      begin
        #50000000;  // 50ms timeout
        $display("[%0t] ERROR: Pipeline completion timeout!", $time);
        $finish;
      end
    join_any
    disable fork;

    // Collect final result
    @(posedge clk_i);
    actual_results.push_back(final_result_o);
    $display("[%0t] Collected final output: 0x%08h (%0d)", $time, final_result_o,
             $signed(final_result_o));
  endtask

  // Task to verify outputs
  task automatic verify_outputs();
    integer local_errors;
    local_errors = 0;

    $display("\n========================================");
    $display("       OUTPUT VERIFICATION");
    $display("========================================");

    if (actual_results.size() != expected_results.size()) begin
      $display("WARNING: Size mismatch! Expected %0d results, got %0d", expected_results.size(),
               actual_results.size());
      $display("(This may be expected if using generated test data)");
    end

    for (int idx = 0; idx < actual_results.size(); idx++) begin
      if (idx < expected_results.size()) begin
        if (actual_results[idx] !== expected_results[idx]) begin
          $display("MISMATCH: Result[%0d]", idx);
          $display("  Expected: 0x%08h (%0d)", expected_results[idx], $signed(
                                                                          expected_results[idx]));
          $display("  Got:      0x%08h (%0d)", actual_results[idx], $signed(actual_results[idx]));
          local_errors++;
        end else begin
          $display("PASS: Result[%0d] = 0x%08h", idx, actual_results[idx]);
        end
      end else begin
        $display("Result[%0d] = 0x%08h (no expected value to compare)", idx, actual_results[idx]);
      end
    end

    errors = local_errors;

    if (local_errors == 0 && expected_results.size() > 0) begin
      $display("\n*** ALL OUTPUTS MATCH EXPECTED VALUES! ***");
      test_passed = 1;
    end else if (expected_results.size() == 0) begin
      $display("\n*** TEST COMPLETED (No expected values for comparison) ***");
      test_passed = 1;
    end else begin
      $display("\n*** %0d ERROR(S) FOUND! ***", local_errors);
      test_passed = 0;
    end

    total_errors += local_errors;
  endtask

  // Task to save actual outputs to file
  task automatic save_actual_outputs();
    actual_file = $fopen(ACTUAL_OUTPUT_FILE, "w");
    if (actual_file == 0) begin
      $display("ERROR: Could not open %s for writing", ACTUAL_OUTPUT_FILE);
    end else begin
      foreach (actual_results[idx]) begin
        $fwrite(actual_file, "%h\n", actual_results[idx]);
      end
      $fclose(actual_file);
      $display("Saved %0d actual outputs to %s", actual_results.size(), ACTUAL_OUTPUT_FILE);
    end
  endtask

  // Task to run a single test
  task automatic run_test();
    $display("\n========================================");
    $display("       STARTING TEST %0d", test_num);
    $display("========================================");
    $display("Configuration:");
    $display("  Activation Function: %0d", activation_function_i);
    $display("  Number of Terms: %0d", num_terms_i);
    $display("  Matrix Size: %0dx%0d", N, N);
    $display("========================================\n");

    // Reset
    current_phase = "RESET";
    rstn_i = 0;
    start_pipeline_i = 0;
    north_write_enable_i = 0;
    north_write_data_i = 0;
    north_write_reset_i = 0;
    west_write_enable_i = 0;
    west_write_data_i = 0;
    west_write_reset_i = 0;
    maxpool_input_data_i = 0;
    maxpool_input_valid_i = 0;

    repeat (10) @(posedge clk_i);
    rstn_i = 1;
    repeat (5) @(posedge clk_i);
    $display("[%0t] Reset complete", $time);

    // Write inputs in parallel using fork-join
    fork
      write_north_inputs();
      write_west_inputs();
    join

    // Wait for queues to stabilize
    repeat (10) @(posedge clk_i);

    // Start pipeline
    current_phase = "STARTING_PIPELINE";
    $display("\n[%0t] Starting pipeline...", $time);
    $display("  North queue empty: %0b", dut.north_queue_empty);
    $display("  West queue empty:  %0b", dut.west_queue_empty);

    @(posedge clk_i);
    start_pipeline_i = 1;
    @(posedge clk_i);
    @(posedge clk_i);

    // Collect outputs
    collect_outputs();

    // Deassert start
    start_pipeline_i = 0;

    // Wait for system to stabilize
    repeat (10) @(posedge clk_i);

    // Verify and save results
    verify_outputs();
    save_actual_outputs();
  endtask

  // Main test sequence
  initial begin
    $display("========================================");
    $display("   SIENNA TOP MODULE TESTBENCH");
    $display("   (FIFO-Based Interface Design)");
    $display("========================================");
    $display("Simulation started at time %0t", $time);
    $display("Parameters:");
    $display("  N = %0d", N);
    $display("  DATA_WIDTH = %0d", DATA_WIDTH);
    $display("  SRAM_DEPTH = %0d", SRAM_DEPTH);
    $display("  IN_ROWS x IN_COLS = %0dx%0d", IN_ROWS, IN_COLS);
    $display("  POOL_H x POOL_W = %0dx%0d", POOL_H, POOL_W);
    $display("  FIFO_DEPTH = %0d", FIFO_DEPTH);
    $display("========================================\n");

    total_errors = 0;
    test_num = 0;

    // Load test configuration
    load_test_config();

    // Load input files
    load_input_files();

    // Run tests
    for (test_num = 1; test_num <= num_tests; test_num++) begin
      run_test();
    end

    // Final summary
    $display("\n========================================");
    $display("       FINAL TEST SUMMARY");
    $display("========================================");
    $display("Total tests run: %0d", num_tests);
    $display("Total errors: %0d", total_errors);
    if (num_tests > 0) begin
      $display("Average cycles per test: %0d", total_cycles / num_tests);
    end

    if (total_errors == 0) begin
      $display("\n*** ALL TESTS PASSED! ***\n");
    end else begin
      $display("\n*** SOME TESTS FAILED! ***\n");
    end

    $display("========================================");
    $display("Simulation finished at time %0t", $time);
    $display("========================================");
    $finish;
  end

  // Timeout watchdog
  initial begin
    #100000000;  // 100ms timeout
    $display("\n========================================");
    $display("ERROR: Simulation timeout!");
    $display("Current phase: %s", current_phase);
    $display("========================================");
    $finish;
  end

  // Waveform dumping
  initial begin
    $dumpfile("TB_sienna_top.vcd");
    $dumpvars(0, TB_sienna_top);
  end

  // Enhanced state monitoring - function to convert state to string
  function string state_to_string(logic [3:0] state);
    case (state)
      4'd0: return "IDLE";
      4'd1: return "SYSTOLIC_PROCESSING";
      4'd2: return "FEED_GPNAE";
      4'd3: return "GPNAE_PROCESSING";
      4'd4: return "COLLECT_GPNAE";
      4'd5: return "FILL_MAXPOOL_SRAM";
      4'd6: return "MAXPOOL_PROCESSING";
      4'd7: return "COLLECT_MAXPOOL";
      4'd8: return "DROPOUT_PROCESSING";
      4'd9: return "PIPELINE_COMPLETE";
      default: return "UNKNOWN";
    endcase
  endfunction

  // State transition monitoring
  logic [3:0] prev_state;

  always_ff @(posedge clk_i) begin
    if (!rstn_i) begin
      prev_state <= 4'd0;
    end else begin
      if (dut.current_state != prev_state) begin
        $display("[%0t] STATE TRANSITION: %s -> %s", $time, state_to_string(prev_state),
                 state_to_string(dut.current_state));
        prev_state <= dut.current_state;
      end
    end
  end

  // Monitor FIFO levels
  always @(posedge clk_i) begin
    if (rstn_i) begin
      if (buffer_count_debug_o > (FIFO_DEPTH * 3 / 4)) begin
        $display("[%0t] WARNING: FIFO1 nearly full (%0d/%0d)", $time, buffer_count_debug_o,
                 FIFO_DEPTH);
      end

      // Monitor completion signals
      if (systolic_complete_debug_o && !dut.systolic_collection_complete) begin
        $display("[%0t] Systolic collection starting", $time);
      end

      if (gpnae_done_o && (dut.current_state == 4'd4)) begin  // COLLECT_GPNAE
        $display("[%0t] GPNAE processing done, result = 0x%08h", $time, dut.gpnae_final_result);
      end
    end
  end

  // Monitor FIFO operations (reduced verbosity)
  logic fifo_debug_enable;
  initial fifo_debug_enable = 0;  // Set to 1 for detailed FIFO debugging

  always @(posedge clk_i) begin
    if (rstn_i && fifo_debug_enable) begin
      // FIFO1 operations
      if (dut.fifo1_wr_en && !dut.fifo1_full) begin
        $display("[%0t] FIFO1 Write: 0x%08h (count: %0d)", $time, dut.fifo1_wr_data,
                 dut.fifo1_count);
      end
      if (dut.fifo1_rd_en && !dut.fifo1_empty) begin
        $display("[%0t] FIFO1 Read: 0x%08h (count: %0d)", $time, dut.fifo1_rd_data,
                 dut.fifo1_count);
      end

      // FIFO2 operations
      if (dut.fifo2_wr_en && !dut.fifo2_full) begin
        $display("[%0t] FIFO2 Write: 0x%08h (count: %0d)", $time, dut.fifo2_wr_data,
                 dut.fifo2_count);
      end
      if (dut.fifo2_rd_en && !dut.fifo2_empty) begin
        $display("[%0t] FIFO2 Read: 0x%08h (count: %0d)", $time, dut.fifo2_rd_data,
                 dut.fifo2_count);
      end

      // FIFO3 operations
      if (dut.fifo3_wr_en && !dut.fifo3_full) begin
        $display("[%0t] FIFO3 Write: 0x%08h (count: %0d)", $time, dut.fifo3_wr_data,
                 dut.fifo3_count);
      end
      if (dut.fifo3_rd_en && !dut.fifo3_empty) begin
        $display("[%0t] FIFO3 Read: 0x%08h (count: %0d)", $time, dut.fifo3_rd_data,
                 dut.fifo3_count);
      end
    end
  end

  // Error detection
  always @(posedge clk_i) begin
    if (rstn_i) begin
      if (intermediate_buffer_full_o && dut.fifo1_wr_en) begin
        $error("[%0t] FIFO1 write overflow!", $time);
      end

      if (dut.fifo1_empty && dut.fifo1_rd_en) begin
        $error("[%0t] FIFO1 read underflow!", $time);
      end

      if (dut.fifo2_full && dut.fifo2_wr_en) begin
        $error("[%0t] FIFO2 write overflow!", $time);
      end

      if (dut.fifo2_empty && dut.fifo2_rd_en) begin
        $error("[%0t] FIFO2 read underflow!", $time);
      end

      if (dut.fifo3_full && dut.fifo3_wr_en) begin
        $error("[%0t] FIFO3 write overflow!", $time);
      end

      if (dut.fifo3_empty && dut.fifo3_rd_en) begin
        $error("[%0t] FIFO3 read underflow!", $time);
      end
    end
  end

endmodule
