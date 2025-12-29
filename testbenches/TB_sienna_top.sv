`timescale 1ns / 1ps

module TB_sienna_top;

  // ================================
  // Compile-time parameters (DUT)
  // ================================
  parameter N                 = 32;
  parameter DATA_WIDTH        = 32;
  parameter SRAM_DEPTH        = N * N;
  parameter ADDR_LINES        = 5;
  parameter CONTROL_WIDTH     = 2;
  parameter IN_ROWS           = 5;
  parameter IN_COLS           = 5;
  parameter POOL_H            = 2;
  parameter POOL_W            = 2;
  parameter STRIDE_ROWS       = 2;
  parameter STRIDE_COLS       = 2;
  parameter PADDING           = 1;
  parameter DROPOUT_P_PERCENT = 50;
  parameter LFSR_WIDTH        = 32;
  parameter INPUT_A_FILE      = "matrixA.mem";
  parameter INPUT_B_FILE      = "matrixB.mem";
  parameter INTERMEDIATE_BUFFER_DEPTH = SRAM_DEPTH * 2;
  parameter FIFO_DEPTH        = 16;

  // Test file parameters
  parameter NORTH_INPUT_FILE   = "matrix_north.mif";
  parameter WEST_INPUT_FILE    = "matrix_west.mif";
  parameter EXPECTED_OUTPUT_FILE = "expected_output.mif";
  parameter ACTUAL_OUTPUT_FILE = "actual_outputs.txt";
  parameter TEST_CONFIG_FILE   = "test_config.mif";

  // ================================
  // Clock and Reset
  // ================================
  logic clk_i;
  logic rstn_i;

  initial begin
    clk_i = 0;
    forever #5 clk_i = ~clk_i;  // 100MHz
  end

  // ================================
  // DUT I/O
  // ================================
  logic start_pipeline_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [ADDR_LINES-1:0]    num_terms_i;

  logic north_write_enable_i;
  logic [DATA_WIDTH-1:0] north_write_data_i;
  logic north_write_reset_i;

  logic west_write_enable_i;
  logic [DATA_WIDTH-1:0] west_write_data_i;
  logic west_write_reset_i;

  logic [DATA_WIDTH-1:0] final_result_o;
  logic pipeline_complete_o;
  logic gpnae_done_o;

  logic systolic_busy_o;
  logic gpnae_busy_o;
  logic maxpool_busy_o;
  logic dropout_busy_o;
  logic intermediate_buffer_full_o;
  logic intermediate_buffer_empty_o;

  logic [DATA_WIDTH-1:0] systolic_result_debug_o;
  logic systolic_complete_debug_o;
  logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer_count_debug_o;

  // ================================
  // Testbench variables
  // ================================
  integer north_file, west_file, expected_file, actual_file;
  integer scan_result, i, j;
  logic [DATA_WIDTH-1:0] north_data_queue[$];
  logic [DATA_WIDTH-1:0] west_data_queue[$];
  logic [DATA_WIDTH-1:0] expected_results[$];
  logic [DATA_WIDTH-1:0] actual_results[$];

  integer test_num;
  integer num_tests;
  integer errors;
  integer total_errors;
  logic   test_passed;

  longint start_time, end_time, total_cycles;
  string current_phase;

  // ================================
  // Config values loaded from MIF
  // ================================
  // Raw 32-bit words from file
  logic [31:0] cfg_word   [0:22];
  integer      cfg_count;

  // Decoded/typed views
  int    cfg_NUM_TESTS;
  logic [CONTROL_WIDTH-1:0] cfg_ACTIVATION_CODE;
  logic [ADDR_LINES-1:0]    cfg_NUM_TERMS;

  int    cfg_N, cfg_DATA_WIDTH, cfg_SRAM_DEPTH, cfg_ADDR_LINES, cfg_CONTROL_WIDTH;
  int    cfg_IN_ROWS, cfg_IN_COLS, cfg_POOL_H, cfg_POOL_W;
  int    cfg_STRIDE_ROWS, cfg_STRIDE_COLS, cfg_PADDING;
  int    cfg_DROPOUT_P_PERCENT, cfg_LFSR_WIDTH, cfg_INTERMEDIATE_BUFFER_DEPTH;
  int    cfg_FIFO_DEPTH, cfg_SEED, cfg_MATRIX_TYPE;
  shortreal cfg_MIN_VAL, cfg_MAX_VAL;

  // ================================
  // Helpers
  // ================================
  function string strip_whitespace(string s);
    integer k, start_idx, end_idx;
    start_idx = 0; end_idx = s.len() ? s.len()-1 : 0;
    for (k = 0; k < s.len(); k++) begin
      if (s[k] != " " && s[k] != "\t" && s[k] != "\n" && s[k] != "\r") begin
        start_idx = k; break;
      end
    end
    for (k = s.len()-1; k >= 0; k--) begin
      if (s[k] != " " && s[k] != "\t" && s[k] != "\n" && s[k] != "\r") begin
        end_idx = k; break;
      end
    end
    if (s.len() == 0) return "";
    if (start_idx <= end_idx) return s.substr(start_idx, end_idx);
    else return "";
  endfunction

  function string activation_name(logic [CONTROL_WIDTH-1:0] code);
    case (code)
      2'b00:   return "None/Linear";
      2'b01:   return "ReLU";
      2'b10:   return "Sigmoid";
      2'b11:   return "Tanh";
      default: return "Unknown";
    endcase
  endfunction

  function string matrix_type_name(int code);
    case (code)
      0: return "random";
      1: return "identity";
      2: return "ones";
      3: return "small_int";
      default: return $sformatf("unknown(%0d)", code);
    endcase
  endfunction

  function void check_param_match(string name, int tb_param, int file_val);
    if (tb_param !== file_val) begin
      $display("WARNING: %s mismatch: TB=%0d, FILE=%0d. (TB uses compile-time value.)",
               name, tb_param, file_val);
    end
  endfunction

  // ================================
  // Parse generic MIF into queue
  // ================================
  task automatic parse_mif_file(input string filename, output logic [DATA_WIDTH-1:0] data_queue[$]);
    integer file_handle;
    string line;
    integer address;
    logic [DATA_WIDTH-1:0] data;
    integer scan_result;
    logic in_content_section;

    data_queue.delete();
    file_handle = $fopen(filename, "r");

    if (file_handle == 0) begin
      $display("ERROR: Could not open MIF file %s", filename);
      return;
    end

    in_content_section = 0;

    while (!$feof(file_handle)) begin
      scan_result = $fgets(line, file_handle);
      if (scan_result == 0) break;

      line = strip_whitespace(line);
      if (line.len() == 0) continue;
      if (line.len() >= 2 && line.substr(0,1) == "--") continue;

      if (line.len() >= 12 && line.substr(0, 12) == "CONTENT BEGIN") begin
        in_content_section = 1;
        continue;
      end
      if (line.len() >= 3 && line.substr(0, 3) == "END") begin
        break;
      end

      if (in_content_section) begin
        scan_result = $sscanf(line, "%d : %h", address, data);
        if (scan_result == 2) begin
          data_queue.push_back(data);
        end
      end
    end

    $fclose(file_handle);
  endtask

  // ================================
  // Load CONFIG from test_config.mif
  // ================================
  task automatic load_test_config();
    integer file_handle;
    string line;
    integer address;
    logic [31:0] data;
    integer sr;
    logic in_content_section;

    cfg_count = 0;
    for (int k = 0; k <= 22; k++) cfg_word[k] = '0;

    file_handle = $fopen(TEST_CONFIG_FILE, "r");
    if (file_handle == 0) begin
      $display("WARNING: Could not open %s, using defaults", TEST_CONFIG_FILE);
      // Minimal defaults:
      cfg_NUM_TESTS       = 1;
      cfg_ACTIVATION_CODE = 2'b01;
      cfg_NUM_TERMS       = N*N;
      num_tests           = cfg_NUM_TESTS;
      activation_function_i = cfg_ACTIVATION_CODE;
      num_terms_i         = cfg_NUM_TERMS[ADDR_LINES-1:0];
      return;
    end

    $display("Loading test configuration from %s...", TEST_CONFIG_FILE);

    in_content_section = 0;
    while (!$feof(file_handle)) begin
      sr = $fgets(line, file_handle);
      if (sr == 0) break;

      line = strip_whitespace(line);
      if (line.len() == 0) continue;
      if (line.len() >= 2 && line.substr(0,1) == "--") continue;

      if (!in_content_section && line.len() >= 12 && line.substr(0,12) == "CONTENT BEGIN") begin
        in_content_section = 1;
        continue;
      end
      if (in_content_section && line.len() >= 3 && line.substr(0,3) == "END") begin
        break;
      end

      if (in_content_section) begin
        sr = $sscanf(line, "%d : %h", address, data);
        if (sr == 2 && address >= 0 && address <= 22) begin
          cfg_word[address] = data;
          cfg_count++;
        end
      end
    end
    $fclose(file_handle);

    // Decode typed values
    cfg_NUM_TESTS        = cfg_word[0];
    cfg_ACTIVATION_CODE  = cfg_word[1][CONTROL_WIDTH-1:0];
    cfg_NUM_TERMS        = cfg_word[2][ADDR_LINES-1:0];

    cfg_N                = cfg_word[3];
    cfg_DATA_WIDTH       = cfg_word[4];
    cfg_SRAM_DEPTH       = cfg_word[5];
    cfg_ADDR_LINES       = cfg_word[6];
    cfg_CONTROL_WIDTH    = cfg_word[7];
    cfg_IN_ROWS          = cfg_word[8];
    cfg_IN_COLS          = cfg_word[9];
    cfg_POOL_H           = cfg_word[10];
    cfg_POOL_W           = cfg_word[11];
    cfg_STRIDE_ROWS      = cfg_word[12];
    cfg_STRIDE_COLS      = cfg_word[13];
    cfg_PADDING          = cfg_word[14];
    cfg_DROPOUT_P_PERCENT= cfg_word[15];
    cfg_LFSR_WIDTH       = cfg_word[16];
    cfg_INTERMEDIATE_BUFFER_DEPTH = cfg_word[17];
    cfg_FIFO_DEPTH       = cfg_word[18];
    cfg_SEED             = cfg_word[19];
    cfg_MATRIX_TYPE      = cfg_word[20];
    cfg_MIN_VAL          = $bitstoshortreal(cfg_word[21]);
    cfg_MAX_VAL          = $bitstoshortreal(cfg_word[22]);

    // Drive DUT inputs from config
    num_tests            = cfg_NUM_TESTS;
    activation_function_i= cfg_ACTIVATION_CODE;
    num_terms_i          = cfg_NUM_TERMS[ADDR_LINES-1:0];

    // Report + sanity checks vs compile-time params
    $display("Loaded test configuration (addresses 0..22):");
    $display("  [0] NUM_TESTS               = %0d", cfg_NUM_TESTS);
    $display("  [1] ACTIVATION_CODE         = 0b%02b (%s)",
             cfg_ACTIVATION_CODE, activation_name(cfg_ACTIVATION_CODE));
    $display("  [2] NUM_TERMS               = %0d", cfg_NUM_TERMS);
    $display("  [3] N                       = %0d", cfg_N);
    $display("  [4] DATA_WIDTH              = %0d", cfg_DATA_WIDTH);
    $display("  [5] SRAM_DEPTH              = %0d", cfg_SRAM_DEPTH);
    $display("  [6] ADDR_LINES              = %0d", cfg_ADDR_LINES);
    $display("  [7] CONTROL_WIDTH           = %0d", cfg_CONTROL_WIDTH);
    $display("  [8] IN_ROWS                 = %0d", cfg_IN_ROWS);
    $display("  [9] IN_COLS                 = %0d", cfg_IN_COLS);
    $display(" [10] POOL_H                  = %0d", cfg_POOL_H);
    $display(" [11] POOL_W                  = %0d", cfg_POOL_W);
    $display(" [12] STRIDE_ROWS             = %0d", cfg_STRIDE_ROWS);
    $display(" [13] STRIDE_COLS             = %0d", cfg_STRIDE_COLS);
    $display(" [14] PADDING                 = %0d", cfg_PADDING);
    $display(" [15] DROPOUT_P_PERCENT       = %0d", cfg_DROPOUT_P_PERCENT);
    $display(" [16] LFSR_WIDTH              = %0d", cfg_LFSR_WIDTH);
    $display(" [17] INTERMEDIATE_BUF_DEPTH  = %0d", cfg_INTERMEDIATE_BUFFER_DEPTH);
    $display(" [18] FIFO_DEPTH              = %0d", cfg_FIFO_DEPTH);
    $display(" [19] SEED                    = %0d", cfg_SEED);
    $display(" [20] MATRIX_TYPE             = %0d (%s)", cfg_MATRIX_TYPE, matrix_type_name(cfg_MATRIX_TYPE));
    $display(" [21] MIN_VAL                 = %f", cfg_MIN_VAL);
    $display(" [22] MAX_VAL                 = %f", cfg_MAX_VAL);

    // Hard checks / warnings vs compile-time params
    check_param_match("N", N, cfg_N);
    check_param_match("DATA_WIDTH", DATA_WIDTH, cfg_DATA_WIDTH);
    check_param_match("SRAM_DEPTH", SRAM_DEPTH, cfg_SRAM_DEPTH);
    check_param_match("ADDR_LINES", ADDR_LINES, cfg_ADDR_LINES);
    check_param_match("CONTROL_WIDTH", CONTROL_WIDTH, cfg_CONTROL_WIDTH);
    check_param_match("IN_ROWS", IN_ROWS, cfg_IN_ROWS);
    check_param_match("IN_COLS", IN_COLS, cfg_IN_COLS);
    check_param_match("POOL_H", POOL_H, cfg_POOL_H);
    check_param_match("POOL_W", POOL_W, cfg_POOL_W);
    check_param_match("STRIDE_ROWS", STRIDE_ROWS, cfg_STRIDE_ROWS);
    check_param_match("STRIDE_COLS", STRIDE_COLS, cfg_STRIDE_COLS);
    check_param_match("PADDING", PADDING, cfg_PADDING);
    check_param_match("DROPOUT_P_PERCENT", DROPOUT_P_PERCENT, cfg_DROPOUT_P_PERCENT);
    check_param_match("LFSR_WIDTH", LFSR_WIDTH, cfg_LFSR_WIDTH);
    check_param_match("INTERMEDIATE_BUFFER_DEPTH", INTERMEDIATE_BUFFER_DEPTH, cfg_INTERMEDIATE_BUFFER_DEPTH);
    check_param_match("FIFO_DEPTH", FIFO_DEPTH, cfg_FIFO_DEPTH);
  endtask

  // ================================
  // DUT instantiation
  // ================================
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
      .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
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

  // ================================
  // Data loading (north/west/expected)
  // ================================
  task automatic generate_test_data();
    logic [DATA_WIDTH-1:0] data;
    $display("Generating simple test data...");

    north_data_queue.delete();
    for (int idx = 0; idx < N * N; idx++) begin
      data = idx + 1;
      north_data_queue.push_back(data);
    end

    west_data_queue.delete();
    for (int idx = 0; idx < N * N; idx++) begin
      data = (idx + 1) * 2;
      west_data_queue.push_back(data);
    end

    expected_results.delete();
    expected_results.push_back(32'hDEADBEEF);
  endtask

  task automatic load_input_files();
    logic files_ok;
    files_ok = 1;

    $display("Loading north inputs from %s...", NORTH_INPUT_FILE);
    parse_mif_file(NORTH_INPUT_FILE, north_data_queue);
    if (north_data_queue.size() == 0) begin
      $display("WARNING: No data loaded from %s", NORTH_INPUT_FILE);
      files_ok = 0;
    end else $display("Loaded %0d north inputs", north_data_queue.size());

    $display("Loading west inputs from %s...", WEST_INPUT_FILE);
    parse_mif_file(WEST_INPUT_FILE, west_data_queue);
    if (west_data_queue.size() == 0) begin
      $display("WARNING: No data loaded from %s", WEST_INPUT_FILE);
      files_ok = 0;
    end else $display("Loaded %0d west inputs", west_data_queue.size());

    $display("Loading expected outputs from %s...", EXPECTED_OUTPUT_FILE);
    parse_mif_file(EXPECTED_OUTPUT_FILE, expected_results);
    if (expected_results.size() == 0) begin
      $display("WARNING: No data loaded from %s", EXPECTED_OUTPUT_FILE);
      files_ok = 0;
    end else $display("Loaded %0d expected outputs", expected_results.size());

    if (!files_ok) generate_test_data();
  endtask

  // ================================
  // I/O driving + run control
  // ================================
  task automatic write_north_inputs();
    $display("[%0t] Writing %0d north inputs...", $time, north_data_queue.size());
    current_phase = "LOADING_NORTH";
    north_write_reset_i = 1; @(posedge clk_i); north_write_reset_i = 0; @(posedge clk_i);

    foreach (north_data_queue[idx]) begin
      north_write_enable_i = 1;
      north_write_data_i   = north_data_queue[idx];
      @(posedge clk_i);
      if (idx < 5 || idx >= north_data_queue.size()-5)
        $display("  North[%0d] = 0x%08h", idx, north_data_queue[idx]);
      else if (idx == 5)
        $display("  ... (%0d more entries) ...", north_data_queue.size() - 10);
    end
    north_write_enable_i = 0;
    $display("[%0t] North inputs written", $time);
  endtask

  task automatic write_west_inputs();
    $display("[%0t] Writing %0d west inputs...", $time, west_data_queue.size());
    current_phase = "LOADING_WEST";
    west_write_reset_i = 1; @(posedge clk_i); west_write_reset_i = 0; @(posedge clk_i);

    foreach (west_data_queue[idx]) begin
      west_write_enable_i = 1;
      west_write_data_i   = west_data_queue[idx];
      @(posedge clk_i);
      if (idx < 5 || idx >= west_data_queue.size()-5)
        $display("  West[%0d] = 0x%08h", idx, west_data_queue[idx]);
      else if (idx == 5)
        $display("  ... (%0d more entries) ...", west_data_queue.size() - 10);
    end
    west_write_enable_i = 0;
    $display("[%0t] West inputs written", $time);
  endtask

  task automatic collect_outputs();
    $display("[%0t] Waiting for pipeline completion...", $time);
    current_phase = "PROCESSING";
    actual_results.delete();
    start_time = $time;

    fork
      begin
        wait (pipeline_complete_o);
        end_time = $time;
        total_cycles = (end_time - start_time) / 10; // 10ns period
        $display("[%0t] Pipeline completed in %0d cycles!", $time, total_cycles);
      end
      begin
        #50000000;
        $display("[%0t] ERROR: Pipeline completion timeout!", $time);
        $finish;
      end
    join_any
    disable fork;

    @(posedge clk_i);
    actual_results.push_back(final_result_o);
    $display("[%0t] Collected final output: 0x%08h (%0d)", $time, final_result_o,
             $signed(final_result_o));
  endtask

  task automatic verify_outputs();
    integer local_errors = 0;
    $display("\n========================================");
    $display("       OUTPUT VERIFICATION");
    $display("========================================");

    if (actual_results.size() != expected_results.size()) begin
      $display("WARNING: Size mismatch! Expected %0d, got %0d",
               expected_results.size(), actual_results.size());
    end

    for (int idx = 0; idx < actual_results.size(); idx++) begin
      if (idx < expected_results.size()) begin
        if (actual_results[idx] !== expected_results[idx]) begin
          $display("MISMATCH[%0d]: exp=0x%08h got=0x%08h", idx,
                   expected_results[idx], actual_results[idx]);
          local_errors++;
        end else begin
          $display("PASS[%0d]: 0x%08h", idx, actual_results[idx]);
        end
      end else begin
        $display("Got extra result[%0d]=0x%08h", idx, actual_results[idx]);
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

  task automatic save_actual_outputs();
    actual_file = $fopen(ACTUAL_OUTPUT_FILE, "w");
    if (actual_file == 0) begin
      $display("ERROR: Could not open %s for writing", ACTUAL_OUTPUT_FILE);
    end else begin
      foreach (actual_results[idx]) $fwrite(actual_file, "%h\n", actual_results[idx]);
      $fclose(actual_file);
      $display("Saved %0d actual outputs to %s", actual_results.size(), ACTUAL_OUTPUT_FILE);
    end
  endtask

  // ================================
  // Pretty names for DUT state (optional debug)
  // ================================
  typedef enum logic [3:0] {
    IDLE,
    SYSTOLIC_PROCESSING,
    FEED_GPNAE,
    GPNAE_PROCESSING,
    COLLECT_GPNAE,
    FEED_MAXPOOL,
    MAXPOOL_PROCESSING,
    COLLECT_MAXPOOL,
    DROPOUT_PROCESSING,
    PIPELINE_COMPLETE
  } pipeline_state_t;

  function string state_to_string(logic [3:0] state);
    case (state)
      4'd0: return "IDLE";
      4'd1: return "SYSTOLIC_PROCESSING";
      4'd2: return "FEED_GPNAE";
      4'd3: return "GPNAE_PROCESSING";
      4'd4: return "COLLECT_GPNAE";
      4'd5: return "FEED_MAXPOOL";
      4'd6: return "MAXPOOL_PROCESSING";
      4'd7: return "COLLECT_MAXPOOL";
      4'd8: return "DROPOUT_PROCESSING";
      4'd9: return "PIPELINE_COMPLETE";
      default: return "UNKNOWN";
    endcase
  endfunction

  logic [3:0] prev_state;
  always_ff @(posedge clk_i) begin
    if (!rstn_i) begin
      prev_state <= 4'd0;
    end else begin
      if (dut.current_state != prev_state) begin
        $display("[%0t] STATE: %s -> %s", $time, state_to_string(prev_state),
                 state_to_string(dut.current_state));
        prev_state <= dut.current_state;
      end
    end
  end

  // (Optional) FIFO + Maxpool debug blocks left as-is from your original TB...

  // ================================
  // Main test sequence
  // ================================
  initial begin
    $display("========================================");
    $display("   SIENNA TOP MODULE TESTBENCH");
    $display("   (Streaming Pipeline Design)");
    $display("========================================");
    $display("Simulation started at time %0t", $time);
    $display("Parameters (compile-time):");
    $display("  N=%0d DATA_WIDTH=%0d SRAM_DEPTH=%0d", N, DATA_WIDTH, SRAM_DEPTH);
    $display("  IN_ROWSxIN_COLS=%0dx%0d  POOL_HxPOOL_W=%0dx%0d",
             IN_ROWS, IN_COLS, POOL_H, POOL_W);
    $display("  STRIDE_ROWS=%0d STRIDE_COLS=%0d PADDING=%0d", STRIDE_ROWS, STRIDE_COLS, PADDING);
    $display("  FIFO_DEPTH=%0d", FIFO_DEPTH);
    $display("========================================\n");

    total_errors = 0;
    test_num = 0;

    // 1) Load full config (populates num_tests/activation/num_terms and logs all others)
    load_test_config();

    // 2) Load input/output vectors
    load_input_files();

    // 3) Run tests
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
    if (total_errors == 0) $display("\n*** ALL TESTS PASSED! ***\n");
    else $display("\n*** SOME TESTS FAILED! ***\n");

    $display("========================================");
    $display("Simulation finished at time %0t", $time);
    $display("========================================");
    $finish;
  end

  task automatic run_test();
    $display("\n========================================");
    $display("       STARTING TEST %0d", test_num);
    $display("========================================");
    $display("Configuration (from file):");
    $display("  Activation: %s (0b%02b)", activation_name(activation_function_i), activation_function_i);
    $display("  Num Terms:  %0d", num_terms_i);
    $display("  Seed:       %0d", cfg_SEED);
    $display("  MatrixType: %s", matrix_type_name(cfg_MATRIX_TYPE));
    $display("  ValueRange: [%f, %f]", cfg_MIN_VAL, cfg_MAX_VAL);
    $display("  Pool: %0dx%0d  Stride: %0dx%0d  Pad:%0d  Dropout:%0d%%",
             cfg_POOL_H, cfg_POOL_W, cfg_STRIDE_ROWS, cfg_STRIDE_COLS, cfg_PADDING, cfg_DROPOUT_P_PERCENT);
    $display("========================================\n");

    // Reset
    current_phase = "RESET";
    rstn_i = 0;
    start_pipeline_i = 0;
    north_write_enable_i = 0; north_write_data_i = '0; north_write_reset_i = 0;
    west_write_enable_i  = 0; west_write_data_i  = '0; west_write_reset_i  = 0;

    repeat (10) @(posedge clk_i);
    rstn_i = 1;
    repeat (5) @(posedge clk_i);
    $display("[%0t] Reset complete", $time);

    // Load inputs in parallel
    fork
      write_north_inputs();
      write_west_inputs();
    join

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

    // Collect & verify
    collect_outputs();
    start_pipeline_i = 0;
    repeat (10) @(posedge clk_i);

    verify_outputs();
    save_actual_outputs();
  endtask

  // Timeout watchdog & VCD
  initial begin
    #100000000;  // 100ms
    $display("\n========================================");
    $display("ERROR: Simulation timeout!");
    $display("Current phase: %s", current_phase);
    $display("========================================");
    $finish;
  end

  initial begin
    $dumpfile("TB_sienna_top.vcd");
    $dumpvars(0, TB_sienna_top);
  end

endmodule
