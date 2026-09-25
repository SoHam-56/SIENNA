`timescale 1ns / 100ps

// COPIES independent SIENNA pipelines behind one host port: sets go to the copies in turn, one set per accepted start.
// Each copy keeps its own output port and set ids, since sienna_top has no output back-pressure to merge them in order.
module sienna_multi #(
    parameter int COPIES            = 2,
    parameter int NUM_LANES         = 32,
    parameter int N                 = 16,
    parameter int TILE_SIZE         = 4,
    parameter int HOST_WORDS        = N,
    parameter int COLLAPSE_K        = 1,  // collapse-k mesh in every copy, as in sienna_top
    parameter int SETS_IN_FLIGHT    = 7,  // credits per copy
    parameter int DATA_WIDTH        = 32,
    parameter int SRAM_DEPTH        = N * N,
    parameter int FIFO_DEPTH        = N * N,
    parameter int ADDR_LINES        = $clog2(FIFO_DEPTH),
    parameter int CONTROL_WIDTH     = 3,
    parameter int IN_ROWS           = 16,
    parameter int IN_COLS           = 16,
    parameter int POOL_H            = 2,
    parameter int POOL_W            = 2,
    parameter int STRIDE_ROWS       = 2,
    parameter int STRIDE_COLS       = 2,
    parameter int PADDING           = 1,
    parameter int DROPOUT_P_PERCENT = 50,
    parameter int LFSR_WIDTH        = 32
) (
    input logic clk_i,
    input logic rstn_i,

    input logic                     start_pipeline_i,
    input logic                     training_mode_i,
    input logic                     accumulate_i,
    input logic                     bias_valid_i,
    input logic [N-1:0][DATA_WIDTH-1:0] bias_i,
    input logic [   LFSR_WIDTH-1:0] dropout_seed_i,
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [     ADDR_LINES:0] num_terms_i,
    input logic                     north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                     west_write_reset_i,

    output logic                                 pipeline_ready_o,  // the copy whose turn it is can take a set
    output logic [$clog2(COPIES+1)-1:0]          copy_sel_o,        // copy the host is loading now
    output logic [COPIES-1:0][NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o,
    output logic [COPIES-1:0][NUM_LANES-1:0]                 result_valid_o,
    output logic [COPIES-1:0]                                pipeline_complete_o,
    output logic [COPIES-1:0][$clog2(SETS_IN_FLIGHT+1)-1:0]  done_set_id_o
);
  localparam int SW = $clog2(COPIES + 1);
  logic [SW-1:0] sel;
  logic [COPIES-1:0] ready;
  assign copy_sel_o = sel;
  assign pipeline_ready_o = ready[sel];

  // The host's start moves the turn on only when the selected copy accepts it.
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) sel <= '0;
    else if (start_pipeline_i && ready[sel]) sel <= (sel == SW'(COPIES - 1)) ? '0 : sel + 1'b1;
  end

  for (genvar c = 0; c < COPIES; c++) begin : COPY
    logic mine;
    assign mine = (sel == SW'(c));
    sienna_top #(
        .NUM_LANES        (NUM_LANES),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (HOST_WORDS),
        .COLLAPSE_K       (COLLAPSE_K),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
        .DATA_WIDTH       (DATA_WIDTH),
        .SRAM_DEPTH       (SRAM_DEPTH),
        .FIFO_DEPTH       (FIFO_DEPTH),
        .ADDR_LINES       (ADDR_LINES),
        .CONTROL_WIDTH    (CONTROL_WIDTH),
        .IN_ROWS          (IN_ROWS),
        .IN_COLS          (IN_COLS),
        .POOL_H           (POOL_H),
        .POOL_W           (POOL_W),
        .STRIDE_ROWS      (STRIDE_ROWS),
        .STRIDE_COLS      (STRIDE_COLS),
        .PADDING          (PADDING),
        .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
        .LFSR_WIDTH       (LFSR_WIDTH)
    ) pipe (
        .clk_i                      (clk_i),
        .rstn_i                     (rstn_i),
        .start_pipeline_i           (start_pipeline_i && mine),
        .training_mode_i            (training_mode_i),
        .accumulate_i               (accumulate_i),
        .bias_valid_i               (bias_valid_i),
        .bias_i                     (bias_i),
        .dropout_seed_i             (dropout_seed_i),
        .activation_function_i      (activation_function_i),
        .num_terms_i                (num_terms_i),
        .north_write_enable_i       (north_write_enable_i && mine),
        .north_write_data_i         (north_write_data_i),
        .north_write_reset_i        (north_write_reset_i && mine),
        .west_write_enable_i        (west_write_enable_i && mine),
        .west_write_data_i          (west_write_data_i),
        .west_write_reset_i         (west_write_reset_i && mine),
        .final_result_o             (final_result_o[c]),
        .result_valid_o             (result_valid_o[c]),
        .pipeline_complete_o        (pipeline_complete_o[c]),
        .pipeline_ready_o           (ready[c]),
        .done_set_id_o              (done_set_id_o[c]),
        .systolic_busy_o            (),
        .gpnae_busy_o               (),
        .maxpool_busy_o             (),
        .dropout_busy_o             (),
        .intermediate_buffer_full_o (),
        .intermediate_buffer_empty_o()
    );
  end

endmodule
