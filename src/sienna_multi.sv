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
    parameter int ACC_BANKS         = 4,
    parameter int RESULT_BANKS      = 4,
    parameter int SETS_IN_FLIGHT    = 2 + 2 + ACC_BANKS + RESULT_BANKS + 2 + 1,  // credits per copy, as sienna_top
    parameter int WC_TILES          = 128,  // weight cache tiles per copy; every copy holds the same weights
    parameter int EXP_W             = 8,   // the build's number format, as sienna_top
    parameter int MAN_W             = 23,
    parameter int DATA_WIDTH        = 1 + EXP_W + MAN_W,
    parameter int ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),
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
    input logic [N-1:0][ACC_W-1:0]  bias_i,
    input logic [N-1:0][31:0]       req_mult_i,  // int8 (D-2): to the copy that takes the start, as sienna_top
    input logic [N-1:0][7:0]        req_shift_i,
    input logic [7:0]               req_zp_i,
    input logic [7:0]               req_min_i,
    input logic [7:0]               req_max_i,
    input logic [15:0]              gp_mx_i,
    input logic [4:0]               gp_shx_i,
    input logic [31:0]              gp_mout_i,
    input logic [7:0]               gp_shout_i,
    input logic [7:0]               gp_zout_i,
    input logic                     weight_cached_i,
    input logic [$clog2(WC_TILES)-1:0] weight_tile_i,
    input logic                     wc_last_i,  // with the start: the last set this copy takes from its cache region's fill (one per copy)
    credit_link_if.consumer         wc_region[2],  // L2: a put opens a fill of that region in every copy; its credit returns once every copy's has
    input logic                     wc_write_enable_i,  // written into every copy's cache
    input logic [$clog2(WC_TILES*N*N)-1:0] wc_write_addr_i,
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
  localparam int PACK_ENTRIES = 8;  // sienna_top's default; sienna_multi never packs
  `include "sienna_set_side.svh"
  logic [SW-1:0] sel;
  logic [COPIES-1:0] ready;  // per copy: this side holds a staging credit of that copy
  assign copy_sel_o = sel;
  assign pipeline_ready_o = ready[sel];

  // The host's start moves the turn on only when the selected copy accepts it.
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) sel <= '0;
    else if (start_pipeline_i && ready[sel]) sel <= (sel == SW'(COPIES - 1)) ? '0 : sel + 1'b1;
  end

  // The set's sideband from the start-time ports, put to the copy whose turn it is.
  set_side_t side;
  always_comb begin
    side               = '0;
    side.wc_last       = wc_last_i;
    side.weight_tile   = weight_tile_i;
    side.weight_cached = weight_cached_i;
    side.accumulate    = accumulate_i;
    side.bias_valid    = bias_valid_i;
    side.train         = training_mode_i;
    side.seed          = dropout_seed_i;
    side.terms         = num_terms_i;
    side.act[0]        = activation_function_i;
    side.zp[0]         = req_zp_i;
    side.amin[0]       = req_min_i;
    side.amax[0]       = req_max_i;
    side.mx[0]         = gp_mx_i;
    side.shx[0]        = gp_shx_i;
    side.mout[0]       = gp_mout_i;
    side.shout[0]      = gp_shout_i;
    side.zout[0]       = gp_zout_i;
    side.mult          = req_mult_i;
    side.shift         = req_shift_i;
  end

  // L2: a fill goes to every copy at once; a region's credit goes up only when every copy has returned its own.
  logic [COPIES-1:0] wc_cred[2];  // this cycle's region credits from each copy
  logic [COPIES-1:0] wc_got[2];  // copies whose region credit came back and has not gone up yet
  for (genvar r = 0; r < 2; r++) begin : G_WC
    logic all_back;
    assign all_back = &(wc_got[r] | wc_cred[r]);
    assign wc_region[r].credit = all_back;
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) wc_got[r] <= '0;
      else wc_got[r] <= all_back ? '0 : (wc_got[r] | wc_cred[r]);
  end

  for (genvar c = 0; c < COPIES; c++) begin : COPY
    logic mine;
    logic [1:0] cnt;  // staging credits of this copy held here
    assign mine = (sel == SW'(c));
    credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) host ();
    credit_link_if #(.DATA_W(1), .CRW(1)) wcc[2] ();
    credit_counter #(.MAX(2), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host.put), .credit_i(host.credit), .has_credit_o(),
                                           .count_o(cnt));
    assign ready[c] = (cnt != 0);
    assign host.put = start_pipeline_i && mine && ready[c];  // a start the copy cannot take is ignored, as before
    assign host.data = side;
    for (genvar r = 0; r < 2; r++) begin : G_WCC
      assign wcc[r].put = wc_region[r].put;
      assign wcc[r].data = wc_region[r].data;
      assign wc_cred[r][c] = wcc[r].credit;
    end
    sienna_top #(
        .NUM_LANES        (NUM_LANES),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (HOST_WORDS),
        .COLLAPSE_K       (COLLAPSE_K),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
        .ACC_BANKS        (ACC_BANKS),
        .RESULT_BANKS     (RESULT_BANKS),
        .WC_TILES         (WC_TILES),
        .EXP_W            (EXP_W),
        .MAN_W            (MAN_W),
        .DATA_WIDTH       (DATA_WIDTH),
        .ACC_W            (ACC_W),
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
        .host                       (host),
        .bias_i                     (bias_i),
        .wc_region                  (wcc),
        .wc_write_enable_i          (wc_write_enable_i),
        .wc_write_addr_i            (wc_write_addr_i),
        .north_write_enable_i       (north_write_enable_i && mine),
        .north_write_data_i         (north_write_data_i),
        .north_write_reset_i        (north_write_reset_i && mine),
        .west_write_enable_i        (west_write_enable_i && mine),
        .west_write_data_i          (west_write_data_i),
        .west_write_reset_i         (west_write_reset_i && mine),
        .final_result_o             (final_result_o[c]),
        .result_valid_o             (result_valid_o[c]),
        .pipeline_complete_o        (pipeline_complete_o[c]),
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
