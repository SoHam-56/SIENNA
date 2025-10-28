`timescale 1ns / 100ps

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
  parameter PADDING = 1;
  parameter DROPOUT_P = 0.5;
  parameter LFSR_WIDTH = 32;
  parameter INPUT_A_FILE = "matrixA.mem";
  parameter INPUT_B_FILE = "matrixB.mem";
  parameter INTERMEDIATE_BUFFER_DEPTH = SRAM_DEPTH * 2;

  // Test file parameters
  parameter NORTH_INPUT_FILE = "north_inputs.txt";
  parameter WEST_INPUT_FILE = "west_inputs.txt";
  parameter MAXPOOL_INPUT_FILE = "maxpool_inputs.txt";
  parameter EXPECTED_OUTPUT_FILE = "expected_outputs.txt";
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

  // Maxpool SRAM Interface
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
  integer north_file, west_file, maxpool_file, expected_file, actual_file, config_file;
  integer scan_result, i, j;
  logic [DATA_WIDTH-1:0] north_data_queue[$];
  logic [DATA_WIDTH-1:0] west_data_queue[$];
  logic [DATA_WIDTH-1:0] maxpool_data_queue[$];
  logic [DATA_WIDTH-1:0] expected_results[$];
  logic [DATA_WIDTH-1:0] actual_results[$];
  integer test_num;
  integer num_tests;
  integer errors;
  integer total_errors;
  logic test_passed;

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
      .PADDING(PADDING),
      .DROPOUT_P(DROPOUT_P),
      .LFSR_WIDTH(LFSR_WIDTH),
      .INPUT_A_FILE(INPUT_A_FILE),
      .INPUT_B_FILE(INPUT_B_FILE),
      .INTERMEDIATE_BUFFER_DEPTH(INTERMEDIATE_BUFFER_DEPTH)
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

  // Task to load input data from files
  task load_input_files();
    logic [DATA_WIDTH-1:0] data;

    // Load north inputs
    north_file = $fopen(NORTH_INPUT_FILE, "r");
    if (north_file == 0) begin
      $display("ERROR: Could not open %s", NORTH_INPUT_FILE);
      $finish;
    end

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

    // Load west inputs
    west_file = $fopen(WEST_INPUT_FILE, "r");
    if (west_file == 0) begin
      $display("ERROR: Could not open %s", WEST_INPUT_FILE);
      $finish;
    end

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

    // Load maxpool inputs
    maxpool_file = $fopen(MAXPOOL_INPUT_FILE, "r");
    if (maxpool_file == 0) begin
      $display("WARNING: Could not open %s", MAXPOOL_INPUT_FILE);
    end else begin
      maxpool_data_queue.delete();
      while (!$feof(
          maxpool_file
      )) begin
        scan_result = $fscanf(maxpool_file, "%h\n", data);
        if (scan_result == 1) begin
          maxpool_data_queue.push_back(data);
        end
      end
      $fclose(maxpool_file);
      $display("Loaded %0d maxpool inputs from %s", maxpool_data_queue.size(), MAXPOOL_INPUT_FILE);
    end

    // Load expected outputs
    expected_file = $fopen(EXPECTED_OUTPUT_FILE, "r");
    if (expected_file == 0) begin
      $display("ERROR: Could not open %s", EXPECTED_OUTPUT_FILE);
      $finish;
    end

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
    $display("Loaded %0d expected outputs from %s", expected_results.size(), EXPECTED_OUTPUT_FILE);

  endtask

  // Task to load test configuration
  task load_test_config();
    config_file = $fopen(TEST_CONFIG_FILE, "r");
    if (config_file == 0) begin
      $display("WARNING: Could not open %s, using default config", TEST_CONFIG_FILE);
      num_tests = 1;
      activation_function_i = 2'b00;
      num_terms_i = 5'h10;
    end else begin
      scan_result = $fscanf(config_file, "%d\n", num_tests);
      scan_result = $fscanf(config_file, "%b\n", activation_function_i);
      scan_result = $fscanf(config_file, "%h\n", num_terms_i);
      $fclose(config_file);
      $display("Loaded test configuration: %0d tests, activation=%b, terms=%h", num_tests,
               activation_function_i, num_terms_i);
    end
  endtask

  // Task to write north inputs
  task write_north_inputs();
    $display("[%0t] Writing %0d north inputs...", $time, north_data_queue.size());
    north_write_reset_i = 1;
    @(posedge clk_i);
    north_write_reset_i = 0;
    @(posedge clk_i);

    foreach (north_data_queue[i]) begin
      north_write_enable_i = 1;
      north_write_data_i   = north_data_queue[i];
      @(posedge clk_i);
    end

    north_write_enable_i = 0;
    $display("[%0t] North inputs written", $time);
  endtask

  // Task to write west inputs
  task write_west_inputs();
    $display("[%0t] Writing %0d west inputs...", $time, west_data_queue.size());
    west_write_reset_i = 1;
    @(posedge clk_i);
    west_write_reset_i = 0;
    @(posedge clk_i);

    foreach (west_data_queue[i]) begin
      west_write_enable_i = 1;
      west_write_data_i   = west_data_queue[i];
      @(posedge clk_i);
    end

    west_write_enable_i = 0;
    $display("[%0t] West inputs written", $time);
  endtask

  // Task to provide maxpool inputs (if needed)
  task provide_maxpool_inputs();
    fork
      begin
        foreach (maxpool_data_queue[i]) begin
          wait (maxpool_busy_o);
          @(posedge clk_i);
          maxpool_input_data_i  = maxpool_data_queue[i];
          maxpool_input_valid_i = 1;
          @(posedge clk_i);
          maxpool_input_valid_i = 0;
        end
      end
    join_none
  endtask

  // Task to collect outputs
  task collect_outputs();
    $display("[%0t] Waiting for pipeline completion...", $time);
    actual_results.delete();

    // Wait for pipeline to complete
    wait (pipeline_complete_o);
    $display("[%0t] Pipeline completed!", $time);

    // Collect final result
    @(posedge clk_i);
    actual_results.push_back(final_result_o);
    $display("[%0t] Collected output: %h", $time, final_result_o);

  endtask

  // Task to verify outputs
  task verify_outputs();
    errors = 0;

    $display("\n========================================");
    $display("       OUTPUT VERIFICATION");
    $display("========================================");

    if (actual_results.size() != expected_results.size()) begin
      $display("ERROR: Size mismatch! Expected %0d results, got %0d", expected_results.size(),
               actual_results.size());
      errors++;
    end

    for (i = 0; i < actual_results.size(); i++) begin
      if (i < expected_results.size()) begin
        if (actual_results[i] !== expected_results[i]) begin
          $display("ERROR: Result[%0d] mismatch! Expected: %h, Got: %h", i, expected_results[i],
                   actual_results[i]);
          errors++;
        end else begin
          $display("PASS: Result[%0d] = %h", i, actual_results[i]);
        end
      end
    end

    if (errors == 0) begin
      $display("\n*** ALL TESTS PASSED! ***");
      test_passed = 1;
    end else begin
      $display("\n*** %0d ERROR(S) FOUND! ***", errors);
      test_passed = 0;
    end

    total_errors += errors;

  endtask

  // Task to save actual outputs to file
  task save_actual_outputs();
    actual_file = $fopen(ACTUAL_OUTPUT_FILE, "w");
    if (actual_file == 0) begin
      $display("ERROR: Could not open %s for writing", ACTUAL_OUTPUT_FILE);
    end else begin
      foreach (actual_results[i]) begin
        $fwrite(actual_file, "%h\n", actual_results[i]);
      end
      $fclose(actual_file);
      $display("Saved %0d actual outputs to %s", actual_results.size(), ACTUAL_OUTPUT_FILE);
    end
  endtask

  // Task to run a single test
  task run_test();
    // Reset
    $display("\n========================================");
    $display("       STARTING TEST %0d", test_num);
    $display("========================================");

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

    // Write inputs
    write_north_inputs();
    write_west_inputs();

    // Start pipeline
    $display("[%0t] Starting pipeline...", $time);
    @(posedge clk_i);
    start_pipeline_i = 1;
    @(posedge clk_i);

    // Provide maxpool inputs if needed
    if (maxpool_data_queue.size() > 0) begin
      provide_maxpool_inputs();
    end

    // Collect outputs
    collect_outputs();

    // Deassert start
    start_pipeline_i = 0;

    // Wait a bit
    repeat (10) @(posedge clk_i);

    // Verify and save results
    verify_outputs();
    save_actual_outputs();

  endtask

  // Main test sequence
  initial begin
    // Initialize
    $display("========================================");
    $display("   SIENNA TOP MODULE TESTBENCH");
    $display("========================================");
    $display("Simulation started at time %0t", $time);

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

    if (total_errors == 0) begin
      $display("\n*** ALL TESTS PASSED! ***\n");
    end else begin
      $display("\n*** SOME TESTS FAILED! ***\n");
    end

    $display("Simulation finished at time %0t", $time);
    $finish;
  end

  // Timeout watchdog
  initial begin
    #100000000;  // 100ms timeout
    $display("\n========================================");
    $display("ERROR: Simulation timeout!");
    $display("========================================");
    $finish;
  end

  // Optional: Waveform dumping
  initial begin
    $dumpfile("TB_sienna_top.vcd");
    $dumpvars(0, TB_sienna_top);
  end

  // Monitor key signals
  always @(posedge clk_i) begin
    if (systolic_busy_o) begin
      $display("[%0t] Systolic Array is busy", $time);
    end
    if (gpnae_busy_o) begin
      $display("[%0t] GPNAE is busy", $time);
    end
    if (maxpool_busy_o) begin
      $display("[%0t] Maxpool is busy", $time);
    end
    if (dropout_busy_o) begin
      $display("[%0t] Dropout is busy", $time);
    end
    if (intermediate_buffer_full_o) begin
      $display("[%0t] WARNING: Intermediate buffer is full!", $time);
    end
  end

endmodule
