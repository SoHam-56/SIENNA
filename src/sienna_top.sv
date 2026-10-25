`timescale 1ns / 100ps

// SIENNA, the one top: COPIES pipelines (sienna_pipe) behind one of two host contracts, chosen by SET_PORT at the wires (no register, so neither contract's cycles move).
// SET_PORT = 0, the layer contract: a configuration and the layer's rows on L10, scheduled into the pipeline by sienna_sched; COPIES must be 1.
// SET_PORT = 1, the set contract: a put per set on L0 with its sideband, the bias, the cache regions (L2) and the write buses; COPIES > 1 takes sets in turn.
// The other contract's outputs are 0, and a_idle_contract fails if its inputs ever put or write; results leave on L9 in both, copy c's lane l on out[c*NUM_LANES + l].
module sienna_top #(
    parameter int SET_PORT          = 0,   // 0: the layer contract (sienna_sched), 1: the set contract
    parameter int COPIES            = 1,   // pipelines; above 1 only on the set contract, which puts sets to them in turn
    parameter int NUM_LANES         = 32,
    parameter int N                 = 16,
    parameter int TILE_SIZE         = 4,
    parameter int HOST_WORDS        = N,  // words per host write, one matrix row; must divide N*N (N on the layer contract)
    parameter int COLLAPSE_K        = 1,  // 1: one full-depth mesh tile per output tile, N^2 PEs and no reduce
    parameter int ACC_BANKS         = 4,  // mesh partial-sum banks per PE
    parameter int RESULT_BANKS      = 4,  // mesh result banks
    parameter int SETS_IN_FLIGHT    = 2 + 2 + ACC_BANKS + RESULT_BANKS + 2 + 1,  // credits per copy: every set the banks can hold
    parameter int ID_W              = $clog2(SETS_IN_FLIGHT + 1),  // set id width; ids count accepted starts
    parameter int WC_TILES          = 128,  // weight cache tiles per copy; every copy holds the same weights
    parameter int WCTW              = $clog2(WC_TILES),
    parameter int WCAW              = $clog2(WC_TILES * N * N),
    parameter int SRAM_DEPTH        = N * N,
    parameter int FIFO_DEPTH        = N * N,
    parameter int ADDR_LINES        = $clog2(FIFO_DEPTH),
    parameter int CONTROL_WIDTH     = 3,  // activation: 001 SELU, 010 sigmoid, 011 tanh, 100 ReLU, 101 linear
    parameter int IN_ROWS           = N,  // the pooling input; N x N on the layer contract
    parameter int IN_COLS           = N,
    parameter int POOL_H            = 1,  // 1x1, stride 1, no padding: pooling is the identity
    parameter int POOL_W            = 1,
    parameter int STRIDE_ROWS       = 1,
    parameter int STRIDE_COLS       = 1,
    parameter int PADDING           = 0,
    parameter int PACK_ENTRIES      = 8,  // distinct activation / int8 output settings one packed set may mix; entry 0 is the set's own
    parameter int DROPOUT_P_PERCENT = 50,
    parameter int LFSR_WIDTH        = 32,
    parameter int LINK_STAGES       = 0,  // register stages on each pipeline's L0, L1, L3 and L9; L10 is not staged
    parameter int OUT_MAX           = 64,  // the most L9 credits a downstream consumer may grant per lane
    parameter int OUT_CRW           = 1,   // L9 credit width
    parameter int LANE_OUT_SLOTS    = 16,  // the collector's L5 slots per lane
    parameter int DIM_W             = 16   // width of the layer contract's configured sizes
) (
    input logic clk_i,
    input logic rstn_i,

    // ── The layer contract (SET_PORT = 0): configuration, taken while idle; the layer starts at once ──
    input logic                     cfg_load_i,
    input logic [DIM_W-1:0]         cfg_m_i,         // rows of A and C
    input logic [DIM_W-1:0]         cfg_kb_i,        // depth of the product for each column block
    input logic [DIM_W-1:0]         cfg_n_i,         // columns of C
    input logic                     cfg_residual_i,  // add an M x N residual input, streamed with A
    input logic                     cfg_bias_i,      // add a bias row, streamed with the weights
    input logic [CONTROL_WIDTH-1:0] cfg_act_i,
    input logic [1:0]               cfg_tier_i,  // the lanes' accuracy tier for the layer: 0 within 1%, 1 within 5%, 2 within 10%
    input logic                     cfg_train_i,
    input logic [LFSR_WIDTH-1:0]    cfg_seed_i,
    // Where the build has them (int8; 1-bit dummies in floats): the layer's requantize and lane words (D-2), and the residual ADD's
    // words, whose in1_zero_point is ignored for cfg_lane_i's zero_point, the conv output's
    input sienna_fmt_pkg::lane_entry_t cfg_lane_i,
    input sienna_fmt_pkg::add_params_t cfg_add_i,
    // Packing, taken with cfg_load_i: every set of the layer is packed this way (sienna-packing)
    input logic [2:0]                                 cfg_pack_shift_i,
    input logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0]   cfg_pack_map_i,
    input logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] cfg_pack_act_i,
    input sienna_fmt_pkg::lane_entry_t [PACK_ENTRIES-1:1] cfg_pack_lane_i,  // entries 1..: requantize and lane words, as cfg_lane_i
    output logic                    busy_o,
    output logic                    done_o,  // one cycle: the layer's last result has left
    credit_link_if.consumer         a_rows,  // L10: one row of N words per put (sienna_sched's contract)
    credit_link_if.consumer         w_rows,  // L10: the weight stream, one row of N words per put
    input logic [N-1:0][sienna_fmt_pkg::ACC_DATA_WIDTH-1:0] w_bias_i,  // int8 only: with each block's bias row on the weight stream
    input sienna_fmt_pkg::requant_scale_t [N-1:0] w_req_scale_i,  // int8 only: per channel, the requantize scale, with each block's bias row
    output logic                    set_done_o,  // a set left the pipeline with its last word (partial sums leave with no results)

    // ── The set contract (SET_PORT = 1), per copy where noted ──
    credit_link_if.consumer         host[COPIES],  // L0 per copy: a put per set, data its sideband (sienna_set_side.svh); put only the copy whose turn it is (copy_sel_o)
    input logic [N-1:0][sienna_fmt_pkg::ACC_DATA_WIDTH-1:0] bias_i,  // with the put: added to column c when the sideband's bias_valid is set; floats: the row widened
    credit_link_if.consumer         wc_region[2],  // L2: a put opens a fill of that cache region in every copy; its credit returns once every copy's has
    input logic                     wc_write_enable_i,  // weight cache write of north_write_data_i at word wc_write_addr_i, into every copy's cache
    input logic [WCAW-1:0]          wc_write_addr_i,
    input logic                     north_write_enable_i,  // the row buses reach only the copy whose turn it is
    input logic [HOST_WORDS-1:0][sienna_fmt_pkg::DATA_WIDTH-1:0] north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][sienna_fmt_pkg::DATA_WIDTH-1:0] west_write_data_i,
    input logic                     west_write_reset_i,
    input logic                     drained_i,  // checks only (COPIES > 1): the host has put its last set; no fill may be left reaching only some copies
    output logic [$clog2(COPIES+1)-1:0] copy_sel_o,  // copy whose turn it is: the host loads its rows and puts it next
    output logic [COPIES-1:0]       pipeline_complete_o,  // per copy: with the put of a set's last word
    output logic [COPIES-1:0][ID_W-1:0] done_set_id_o,  // id of the set pipeline_complete_o reports
    output logic [COPIES-1:0]       systolic_busy_o,
    output logic [COPIES-1:0]       gpnae_busy_o,
    output logic [COPIES-1:0]       maxpool_busy_o,
    output logic [COPIES-1:0]       dropout_busy_o,

    // ── Both contracts: L9, copy c's lane l is out[c*NUM_LANES + l]; a word per put while the consumer's credit is held ──
    credit_link_if.producer         out[COPIES*NUM_LANES]
);
  localparam int DATA_WIDTH     = sienna_fmt_pkg::DATA_WIDTH;
  localparam int ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
  localparam int SW = $clog2(COPIES + 1);
  `include "sienna_set_side.svh"
  localparam int SIDE_W = $bits(set_side_t);

  if (SET_PORT != 0 && SET_PORT != 1) begin : G_BAD_SET_PORT
    $fatal(1, "sienna_top: SET_PORT %0d: 0 (the layer contract) or 1 (the set contract)", SET_PORT);
  end else if (COPIES < 1 || (COPIES > 1 && SET_PORT != 1)) begin : G_BAD_COPIES  // no flow schedules layers over several copies
    $fatal(1, "sienna_top: COPIES %0d with SET_PORT %0d: COPIES > 1 needs SET_PORT = 1", COPIES, SET_PORT);
  end else if (SET_PORT == 0 && (HOST_WORDS != N || IN_ROWS != N || IN_COLS != N)) begin : G_BAD_LAYER  // sienna_sched writes N-word rows of N x N tiles
    $fatal(1, "sienna_top: the layer contract needs HOST_WORDS, IN_ROWS and IN_COLS = N (%0d), not %0d, %0d, %0d", N, HOST_WORDS, IN_ROWS, IN_COLS);
  end

  // I3: every plain-vector boundary port is as wide as the define's package says; WIDTH warnings are off in the Makefile, so a port written at another width would truncate silently.
  localparam int ENTRY_BITS = $bits(sienna_fmt_pkg::lane_entry_t);
  if ($bits(bias_i) != N * ACC_DATA_WIDTH || $bits(w_bias_i) != N * ACC_DATA_WIDTH ||
      $bits(north_write_data_i) != HOST_WORDS * DATA_WIDTH || $bits(west_write_data_i) != HOST_WORDS * DATA_WIDTH ||
      $bits(w_req_scale_i) != N * $bits(sienna_fmt_pkg::requant_scale_t) || $bits(cfg_lane_i) != ENTRY_BITS ||
      $bits(cfg_pack_lane_i) != (PACK_ENTRIES - 1) * ENTRY_BITS || $bits(cfg_add_i) != $bits(sienna_fmt_pkg::add_params_t)) begin : G_BAD_PORT_WIDTH
    $fatal(1, "sienna_top: a port's width is not the %s define's: bias_i %0d w_bias_i %0d (want %0d), north/west_write_data_i %0d/%0d (want %0d), w_req_scale_i %0d (want %0d), cfg_lane_i %0d cfg_pack_lane_i %0d (want %0d, %0d), cfg_add_i %0d (want %0d)",
           sienna_fmt_pkg::FORMAT_NAME, $bits(bias_i), $bits(w_bias_i), N * ACC_DATA_WIDTH, $bits(north_write_data_i), $bits(west_write_data_i),
           HOST_WORDS * DATA_WIDTH, $bits(w_req_scale_i), N * $bits(sienna_fmt_pkg::requant_scale_t), $bits(cfg_lane_i), $bits(cfg_pack_lane_i),
           ENTRY_BITS, (PACK_ENTRIES - 1) * ENTRY_BITS, $bits(cfg_add_i), $bits(sienna_fmt_pkg::add_params_t));
  end

`ifndef SYNTHESIS  // interface widths are not elaboration constants in Verilator, so the host and output links are checked at time 0
  if (SET_PORT == 1) begin : G_HOST_W
    initial
      if ($bits(host[0].data) != SIDE_W || $bits(host[0].credit) != 1)
        $fatal(1, "sienna_top: the host link needs data %0d bits and credit 1, found %0d and %0d", SIDE_W, $bits(host[0].data), $bits(host[0].credit));
  end
  initial
    if ($bits(out[0].data) != DATA_WIDTH || $bits(out[0].credit) != OUT_CRW)
      $fatal(1, "sienna_top: the output links need data %0d bits and credit %0d, found %0d and %0d", DATA_WIDTH, OUT_CRW, $bits(out[0].data),
             $bits(out[0].credit));
`endif

  // Whether the unused contract's inputs put or write this cycle; a_idle_contract checks it registered, since a host may drive at the edge.
  logic idle_put;

  if (SET_PORT == 0) begin : G_LAYER
    credit_link_if #(.DATA_W(SIDE_W), .CRW(1)) l0 ();  // sienna_sched's sets into the pipeline
    credit_link_if #(.DATA_W(1), .CRW(1)) wc[2] ();  // its cache regions
    logic [N-1:0][ACC_DATA_WIDTH-1:0] bias;
    logic wc_we, north_we, west_we, set_done;
    logic [WCAW-1:0] wc_addr;
    logic [N-1:0][DATA_WIDTH-1:0] north, west;
    sienna_sched #(
        .N            (N),
        .WC_TILES     (WC_TILES),
        .CONTROL_WIDTH(CONTROL_WIDTH),
        .LFSR_WIDTH   (LFSR_WIDTH),
        .DIM_W        (DIM_W),
        .PACK_ENTRIES (PACK_ENTRIES)
    ) sched (
        .clk_i               (clk_i),
        .rstn_i              (rstn_i),
        .cfg_load_i          (cfg_load_i),
        .cfg_m_i             (cfg_m_i),
        .cfg_kb_i            (cfg_kb_i),
        .cfg_n_i             (cfg_n_i),
        .cfg_residual_i      (cfg_residual_i),
        .cfg_bias_i          (cfg_bias_i),
        .cfg_act_i           (cfg_act_i),
        .cfg_tier_i          (cfg_tier_i),
        .cfg_train_i         (cfg_train_i),
        .cfg_seed_i          (cfg_seed_i),
        .cfg_lane_i          (cfg_lane_i),
        .cfg_add_i           (cfg_add_i),
        .cfg_pack_shift_i    (cfg_pack_shift_i),
        .cfg_pack_map_i      (cfg_pack_map_i),
        .cfg_pack_act_i      (cfg_pack_act_i),
        .cfg_pack_lane_i     (cfg_pack_lane_i),
        .busy_o              (busy_o),
        .done_o              (done_o),
        .a_rows              (a_rows),
        .w_rows              (w_rows),
        .w_bias_i            (w_bias_i),
        .w_req_scale_i       (w_req_scale_i),
        .set_done_o          (set_done_o),
        .host                (l0),
        .wc_region           (wc),
        .bias_o              (bias),
        .wc_write_enable_o   (wc_we),
        .wc_write_addr_o     (wc_addr),
        .north_write_enable_o(north_we),
        .north_write_data_o  (north),
        .west_write_enable_o (west_we),
        .west_write_data_o   (west),
        .set_done_i          (set_done)
    );
    logic unused_done_id, unused_busy;  // the layer contract reports completion through sienna_sched only
    logic [ID_W-1:0] unused_id;
    logic [3:0] unused_busys;
    sienna_pipe #(
        .NUM_LANES        (NUM_LANES),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (N),
        .COLLAPSE_K       (COLLAPSE_K),
        .ACC_BANKS        (ACC_BANKS),
        .RESULT_BANKS     (RESULT_BANKS),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
        .ID_W             (ID_W),
        .WC_TILES         (WC_TILES),
        .WCTW             (WCTW),
        .WCAW             (WCAW),
        .SRAM_DEPTH       (SRAM_DEPTH),
        .FIFO_DEPTH       (FIFO_DEPTH),
        .ADDR_LINES       (ADDR_LINES),
        .CONTROL_WIDTH    (CONTROL_WIDTH),
        .IN_ROWS          (N),
        .IN_COLS          (N),
        .POOL_H           (POOL_H),
        .POOL_W           (POOL_W),
        .STRIDE_ROWS      (STRIDE_ROWS),
        .STRIDE_COLS      (STRIDE_COLS),
        .PADDING          (PADDING),
        .PACK_ENTRIES     (PACK_ENTRIES),
        .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
        .LFSR_WIDTH       (LFSR_WIDTH),
        .LINK_STAGES      (LINK_STAGES),
        .OUT_MAX          (OUT_MAX),
        .OUT_CRW          (OUT_CRW),
        .LANE_OUT_SLOTS   (LANE_OUT_SLOTS)
    ) pipe (
        .clk_i               (clk_i),
        .rstn_i              (rstn_i),
        .host                (l0),
        .bias_i              (bias),
        .wc_region           (wc),
        .wc_write_enable_i   (wc_we),
        .wc_write_addr_i     (wc_addr),
        .north_write_enable_i(north_we),
        .north_write_data_i  (north),
        .north_write_reset_i (1'b0),
        .west_write_enable_i (west_we),
        .west_write_data_i   (west),
        .west_write_reset_i  (1'b0),
        .out                 (out),
        .pipeline_complete_o (set_done),
        .done_set_id_o       (unused_id),
        .systolic_busy_o     (unused_busys[0]),
        .gpnae_busy_o        (unused_busys[1]),
        .maxpool_busy_o      (unused_busys[2]),
        .dropout_busy_o      (unused_busys[3])
    );
    assign unused_done_id = ^unused_id;
    assign unused_busy    = ^unused_busys;
    // The set contract, unused: no credit, no completion, nothing busy.
    assign host[0].credit = 1'b0;
    for (genvar r = 0; r < 2; r++) begin : G_WC_IDLE
      assign wc_region[r].credit = 1'b0;
    end
    assign copy_sel_o          = '0;
    assign pipeline_complete_o = '0;
    assign done_set_id_o       = '0;
    assign systolic_busy_o     = '0;
    assign gpnae_busy_o        = '0;
    assign maxpool_busy_o      = '0;
    assign dropout_busy_o      = '0;
    assign idle_put = host[0].put || wc_region[0].put || wc_region[1].put || wc_write_enable_i || north_write_enable_i || north_write_reset_i ||
                      west_write_enable_i || west_write_reset_i;
    wire unused_set_inputs = ^{host[0].data, wc_region[0].data, wc_region[1].data, bias_i, wc_write_addr_i, north_write_data_i,
                               west_write_data_i, drained_i};  // only the set contract reads them

  end else begin : G_SET
    logic [SW-1:0] sel;  // the copy whose turn it is
    logic [COPIES-1:0] put;  // per copy: the host's put this cycle
    logic [COPIES-1:0] wc_cred[2];  // per region: this cycle's credit from each copy
    for (genvar c = 0; c < COPIES; c++) begin : G_COPY
      logic mine;  // the turn is this copy's: the row buses reach it
      assign mine   = (COPIES == 1) || (sel == SW'(c));
      assign put[c] = host[c].put;
      credit_link_if #(.DATA_W(1), .CRW(1)) wcc[2] ();  // the copy's cache regions
      for (genvar r = 0; r < 2; r++) begin : G_WCC
        assign wcc[r].put    = wc_region[r].put;
        assign wcc[r].data   = wc_region[r].data;
        assign wc_cred[r][c] = wcc[r].credit;
      end
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) outl[NUM_LANES] ();  // the copy's L9
      for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
        assign out[c*NUM_LANES+l].put  = outl[l].put;
        assign out[c*NUM_LANES+l].data = outl[l].data;
        assign outl[l].credit          = out[c*NUM_LANES+l].credit;
      end
      sienna_pipe #(
          .NUM_LANES        (NUM_LANES),
          .N                (N),
          .TILE_SIZE        (TILE_SIZE),
          .HOST_WORDS       (HOST_WORDS),
          .COLLAPSE_K       (COLLAPSE_K),
          .ACC_BANKS        (ACC_BANKS),
          .RESULT_BANKS     (RESULT_BANKS),
          .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
          .ID_W             (ID_W),
          .WC_TILES         (WC_TILES),
          .WCTW             (WCTW),
          .WCAW             (WCAW),
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
          .PACK_ENTRIES     (PACK_ENTRIES),
          .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
          .LFSR_WIDTH       (LFSR_WIDTH),
          .LINK_STAGES      (LINK_STAGES),
          .OUT_MAX          (OUT_MAX),
          .OUT_CRW          (OUT_CRW),
          .LANE_OUT_SLOTS   (LANE_OUT_SLOTS)
      ) pipe (
          .clk_i               (clk_i),
          .rstn_i              (rstn_i),
          .host                (host[c]),
          .bias_i              (bias_i),
          .wc_region           (wcc),
          .wc_write_enable_i   (wc_write_enable_i),
          .wc_write_addr_i     (wc_write_addr_i),
          .north_write_enable_i(north_write_enable_i && mine),
          .north_write_data_i  (north_write_data_i),
          .north_write_reset_i (north_write_reset_i && mine),
          .west_write_enable_i (west_write_enable_i && mine),
          .west_write_data_i   (west_write_data_i),
          .west_write_reset_i  (west_write_reset_i && mine),
          .out                 (outl),
          .pipeline_complete_o (pipeline_complete_o[c]),
          .done_set_id_o       (done_set_id_o[c]),
          .systolic_busy_o     (systolic_busy_o[c]),
          .gpnae_busy_o        (gpnae_busy_o[c]),
          .maxpool_busy_o      (maxpool_busy_o[c]),
          .dropout_busy_o      (dropout_busy_o[c])
      );
    end

    if (COPIES == 1) begin : G_ONE  // one pipeline: L2 is its own, and there is no turn
      assign sel = '0;
      for (genvar r = 0; r < 2; r++) begin : G_WC
        assign wc_region[r].credit = wc_cred[r][0];
      end
      wire unused_one = ^{put, drained_i};  // COPIES > 1 only: the turn and the drain checks
    end else begin : G_MULTI  // COPIES pipelines behind one host: sets go to the copies in turn, one set per put
      // The turn moves on with each put to the copy whose turn it is.
      always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) sel <= '0;
        else if (put[sel]) sel <= (sel == SW'(COPIES - 1)) ? '0 : sel + 1'b1;
      end

      logic live;  // out of reset for a cycle: no region credit before it
      always_ff @(posedge clk_i or negedge rstn_i)
        if (!rstn_i) live <= 1'b0;
        else live <= 1'b1;

      // L2: a fill goes to every copy at once; a region's credit goes up only when every copy has returned its own.
      logic [COPIES-1:0] wc_got[2];  // copies whose region credit came back and has not gone up yet
      for (genvar r = 0; r < 2; r++) begin : G_WC
        logic all_back;
        assign all_back = live && &(wc_got[r] | wc_cred[r]);
        assign wc_region[r].credit = all_back;
        always_ff @(posedge clk_i or negedge rstn_i)
          if (!rstn_i) wc_got[r] <= '0;
          else wc_got[r] <= all_back ? '0 : (wc_got[r] | wc_cred[r]);
      end

`ifndef SYNTHESIS
      logic [COPIES-1:0] p_cached, p_last, p_region;  // per copy: the put set reads the cache, is marked its fill's last, and its region
      for (genvar c = 0; c < COPIES; c++) begin : G_SIDE
        set_side_t side;
        assign side = host[c].data;
        assign p_cached[c] = side.weight_cached;
        assign p_last[c] = side.wc_last;
        assign p_region[c] = side.weight_tile[WCTW-1];
      end
      logic [1:0] wc_put;  // the host opens a fill of the region this cycle
      assign wc_put = {wc_region[1].put, wc_region[0].put};
      // The L2 contract, tracked per region: a fill is open from its put until every copy has taken its marked set.
      logic [1:0] f_open, f_started;  // the fill is open; a set of it has been put
      logic [COPIES-1:0] f_marked[2];  // copies that took their marked set of the open fill
      logic any_put, set_cached, set_last, set_region;
      logic [SW-1:0] set_copy;
      always_comb begin
        any_put = |put;
        set_copy = sel;
        set_cached = 1'b0;
        set_last = 1'b0;
        set_region = 1'b0;
        for (int c = 0; c < COPIES; c++)
          if (put[c]) begin
            set_copy = SW'(c);
            set_cached = p_cached[c];
            set_last = p_last[c];
            set_region = p_region[c];
          end
      end
      always_ff @(posedge clk_i or negedge rstn_i)
        if (!rstn_i) begin
          f_open <= '0;
          f_started <= '0;
          f_marked[0] <= '0;
          f_marked[1] <= '0;
        end else
          for (int r = 0; r < 2; r++)
            if (wc_put[r]) begin
              f_open[r] <= 1'b1;
              f_started[r] <= 1'b0;
              f_marked[r] <= '0;
            end else if (any_put && set_cached && set_region == 1'(r)) begin
              f_started[r] <= 1'b1;
              if (set_last) begin
                f_marked[r] <= f_marked[r] | (COPIES'(1) << set_copy);
                if ((f_marked[r] | (COPIES'(1) << set_copy)) == '1) f_open[r] <= 1'b0;
              end
            end
      // A put and its terms, registered: a host may drive its puts at the edge.
      logic pq, pq_turn, pq_after_last, pq_short;
      logic [SW-1:0] pq_copy;
      logic pq_region;
      always_ff @(posedge clk_i or negedge rstn_i)
        if (!rstn_i) {pq, pq_turn, pq_after_last, pq_short, pq_copy, pq_region} <= '0;
        else begin
          pq            <= any_put;
          pq_turn       <= (put == (COPIES'(1) << sel));
          pq_after_last <= set_cached && (!f_open[set_region] || f_marked[set_region][set_copy]);
          pq_short      <= 1'b0;
          pq_copy       <= set_copy;
          pq_region     <= set_region;
          // Another kind of set while a fill has reached only some copies with their marked sets: those copies' regions never close.
          for (int r = 0; r < 2; r++)
            if (f_open[r] && f_started[r] && !(set_cached && set_region == 1'(r))) begin
              pq_short  <= 1'b1;
              pq_region <= 1'(r);
            end
        end
      a_put_on_turn: assert property (@(posedge clk_i) disable iff (!rstn_i) pq |-> pq_turn)
        else $error("sienna_top: a put to a copy whose turn it is not (copy %0d put, turn %0d): its rows went to the copy whose turn it was",
                    pq_copy, $past(sel));
      a_wc_after_last: assert property (@(posedge clk_i) disable iff (!rstn_i) pq |-> !pq_after_last)
        else $error("sienna_top: copy %0d took a cached set of region %0d after its marked one, or with no fill open: its region may be refilled under it",
                    pq_copy, pq_region);
      a_wc_fill_short: assert property (@(posedge clk_i) disable iff (!rstn_i) pq |-> !pq_short)
        else $error("sienna_top: region %0d's fill reached only some copies with their marked sets when another set was put: the rest never close it",
                    pq_region);
      // At drain, a fill with sets put must have reached every copy's marked set, else its region credit never returns.
      logic dq, dq_short;
      logic [COPIES-1:0] dq_marked;
      logic dq_region;
      always_ff @(posedge clk_i or negedge rstn_i)
        if (!rstn_i) {dq, dq_short, dq_marked, dq_region} <= '0;
        else begin
          dq <= drained_i;
          dq_short <= 1'b0;
          dq_marked <= '0;
          dq_region <= 1'b0;
          for (int r = 0; r < 2; r++)
            if (f_open[r] && f_started[r]) begin
              dq_short <= 1'b1;
              dq_marked <= f_marked[r];
              dq_region <= 1'(r);
            end
        end
      a_wc_drained: assert property (@(posedge clk_i) disable iff (!rstn_i) dq |-> !dq_short)
        else $error("sienna_top: drained with region %0d's fill reaching only copies %b with their marked sets: its credit never returns",
                    dq_region, dq_marked);
`else
      wire unused_drained = drained_i;  // checks only
`endif
    end
    assign copy_sel_o = sel;

    // The layer contract, unused: no row credit, never busy or done.
    assign a_rows.credit = '0;
    assign w_rows.credit = '0;
    assign busy_o        = 1'b0;
    assign done_o        = 1'b0;
    assign set_done_o    = 1'b0;
    assign idle_put = cfg_load_i || a_rows.put || w_rows.put;
    wire unused_layer_inputs = ^{cfg_m_i, cfg_kb_i, cfg_n_i, cfg_residual_i, cfg_bias_i, cfg_act_i, cfg_tier_i, cfg_train_i, cfg_seed_i, cfg_lane_i,
                                 cfg_add_i, cfg_pack_shift_i, cfg_pack_map_i, cfg_pack_act_i, cfg_pack_lane_i, a_rows.data, w_rows.data,
                                 w_bias_i, w_req_scale_i};  // only the layer contract reads them
  end

`ifndef SYNTHESIS
  // The unused contract's inputs, registered: a host that drives at the edge is still seen.
  logic idle_put_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) idle_put_q <= 1'b0;
    else idle_put_q <= idle_put;
  a_idle_contract: assert property (@(posedge clk_i) disable iff (!rstn_i) !idle_put_q)
    else $error("sienna_top: a_idle_contract: a put or write on the %s contract, which SET_PORT = %0d leaves unused", (SET_PORT == 0) ? "set" : "layer",
                SET_PORT);
`else
  wire unused_idle_put = idle_put;  // checks only
`endif

endmodule
