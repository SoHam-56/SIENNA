`timescale 1ns / 1ps

import test_config_pkg::*;

module TB_sienna_top;

  localparam string NORTH_INPUT_FILE = "matrix_north.mif";
  localparam string WEST_INPUT_FILE = "matrix_west.mif";
  localparam string EXPECTED_OUTPUT_FILE = "expected_output.mif";
  localparam ADDR_LINES = $clog2(FIFO_DEPTH);

  logic clk_i;
  logic rstn_i;

  logic start_pipeline_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [ADDR_LINES:0]    num_terms_i;

  logic north_write_enable_i;
  logic [DATA_WIDTH-1:0] north_write_data_i;
  logic north_write_reset_i;
  logic west_write_enable_i;
  logic [DATA_WIDTH-1:0] west_write_data_i;
  logic west_write_reset_i;
  logic [DATA_WIDTH-1:0] final_result_o;
  logic pipeline_complete_o;

  logic [DATA_WIDTH-1:0] north_data_queue[$];
  logic [DATA_WIDTH-1:0] west_data_queue[$];
  logic [DATA_WIDTH-1:0] expected_results[$];
  logic [DATA_WIDTH-1:0] actual_results[$];

  initial begin
    clk_i = 0;
    forever #5 clk_i = ~clk_i;
  end

  sienna_top #(
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

      .INPUT_A_FILE             ("matrixA.mem"),
      .INPUT_B_FILE             ("matrixB.mem"),
      .INTERMEDIATE_BUFFER_DEPTH(INTERMEDIATE_BUFFER_DEPTH),
      .FIFO_DEPTH               (FIFO_DEPTH)
  ) dut (
      .clk_i (clk_i),
      .rstn_i(rstn_i),

      .start_pipeline_i     (start_pipeline_i),
      .activation_function_i(activation_function_i),
      .num_terms_i          (num_terms_i),
      .north_write_enable_i (north_write_enable_i),
      .north_write_data_i   (north_write_data_i),
      .north_write_reset_i  (north_write_reset_i),
      .west_write_enable_i  (west_write_enable_i),
      .west_write_data_i    (west_write_data_i),
      .west_write_reset_i   (west_write_reset_i),
      .final_result_o       (final_result_o),
      .pipeline_complete_o  (pipeline_complete_o)

  );

  // ========================================================================
  // 5. HELPER TASKS
  // ========================================================================
  task automatic parse_mif_file(input string filename, output logic [DATA_WIDTH-1:0] data_queue[$]);
    integer file_handle, scan_result, address;
    logic [DATA_WIDTH-1:0] data;
    string line;
    logic in_content;

    data_queue.delete();
    file_handle = $fopen(filename, "r");
    if (!file_handle) begin
      $display("Error opening %s", filename);
      return;
    end

    in_content = 0;
    while (!$feof(
        file_handle
    )) begin
      scan_result = $fgets(line, file_handle);
      if (line.len() >= 12 && line.substr(0, 12) == "CONTENT BEGIN") in_content = 1;
      if (in_content) begin
        scan_result = $sscanf(line, "%d : %h", address, data);
        if (scan_result == 2) data_queue.push_back(data);
      end
    end
    $fclose(file_handle);
  endtask

  initial begin

    // Load Data
    parse_mif_file(NORTH_INPUT_FILE, north_data_queue);
    parse_mif_file(WEST_INPUT_FILE, west_data_queue);
    parse_mif_file(EXPECTED_OUTPUT_FILE, expected_results);

    // Apply Reset
    rstn_i = 0;
    north_write_reset_i = 1;
    west_write_reset_i = 1;
    repeat (10) @(posedge clk_i);
    rstn_i = 1;
    north_write_reset_i = 0;
    west_write_reset_i = 0;
    repeat (5) @(posedge clk_i);

    // Load Memories
    fork
      begin
        foreach (north_data_queue[i]) begin
          north_write_enable_i = 1;
          north_write_data_i   = north_data_queue[i];
          @(posedge clk_i);
        end
        north_write_enable_i = 0;
      end
      begin
        foreach (west_data_queue[i]) begin
          west_write_enable_i = 1;
          west_write_data_i   = west_data_queue[i];
          @(posedge clk_i);
        end
        west_write_enable_i = 0;
      end
    join

    // Start DUT
    activation_function_i = ACTIVATION_CODE[CONTROL_WIDTH-1:0];
    num_terms_i           = NUM_TERMS;

    @(posedge clk_i);
    start_pipeline_i = 1;
    @(posedge clk_i);
    start_pipeline_i = 0;

    // Wait & Verify
    wait (pipeline_complete_o);
    @(posedge clk_i);

    if (final_result_o === expected_results[0]) $display("PASSED: 0x%h", final_result_o);
    else $display("FAILED: Exp 0x%h, Got 0x%h", expected_results[0], final_result_o);

    $finish;
  end

  initial begin
    $dumpfile("TB_sienna_top.vcd");
    $dumpvars(0, TB_sienna_top);
  end

endmodule

