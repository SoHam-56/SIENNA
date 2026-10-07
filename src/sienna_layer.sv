`timescale 1ns / 100ps

// One network layer as C = A x B (+ bias, + residual), scheduled entirely in hardware.
// The host writes a configuration, then streams data in a fixed order; every per-set decision is made here.
//
// Weight stream, per column block c of N output columns: the bias row (if cfg_bias_i), then the block's
// ceil(cfg_kb_i/N) weight tiles of N rows each, once if the block is cached, else once per row tile.
// A block is cached when it has more than one row tile and at most WC_TILES/2 weight tiles.
// Activation stream, per column block, per row tile: the depth tiles of A (N rows each), then the residual tile if cfg_residual_i.
// int8: every block has a bias beat, its int32 bias and requantize words on w_bias_i, w_req_mult_i, w_req_shift_i (w_data_i ignored).
// Results leave per output tile, column blocks outer and row tiles inner, N x N row-major.
module sienna_layer #(
    parameter int NUM_LANES         = 32,
    parameter int N                 = 16,
    parameter int TILE_SIZE         = 4,
    parameter int COLLAPSE_K        = 1,
    parameter int ACC_BANKS         = 4,
    parameter int RESULT_BANKS      = 4,
    parameter int SETS_IN_FLIGHT    = 2 + 2 + ACC_BANKS + RESULT_BANKS + 2 + 1,  // as sienna_top: every set the banks can hold
    parameter int WC_TILES          = 128,
    parameter int EXP_W             = 8,   // the build's number format: fp32 8/23, bf16 8/7
    parameter int MAN_W             = 23,
    parameter int DATA_WIDTH        = 1 + EXP_W + MAN_W,  // every word: operands, bias, results
    parameter int ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // bias and sums: int32 in int8, DATA_WIDTH otherwise
    parameter int CONTROL_WIDTH     = 3,
    parameter int LFSR_WIDTH        = 32,
    parameter int POOL_H            = 1,
    parameter int POOL_W            = 1,
    parameter int STRIDE_ROWS       = 1,
    parameter int STRIDE_COLS       = 1,
    parameter int PADDING           = 0,
    parameter int DROPOUT_P_PERCENT = 50,
    parameter int DIM_W             = 16,  // width of the configured sizes
    parameter int PACK_ENTRIES      = 8
) (
    input logic clk_i,
    input logic rstn_i,

    // Configuration, taken while idle; the layer starts at once.
    input logic                     cfg_load_i,
    input logic [DIM_W-1:0]         cfg_m_i,         // rows of A and C
    input logic [DIM_W-1:0]         cfg_kb_i,        // depth of the product for each column block
    input logic [DIM_W-1:0]         cfg_n_i,         // columns of C
    input logic                     cfg_residual_i,  // add an M x N residual input, streamed with A
    input logic                     cfg_bias_i,      // add a bias row, streamed with the weights
    input logic [CONTROL_WIDTH-1:0] cfg_act_i,
    input logic                     cfg_train_i,
    input logic [LFSR_WIDTH-1:0]    cfg_seed_i,
    // int8 only, taken with cfg_load_i: the layer's requantize and GPNAE parameters (D-2)
    input logic [7:0]               cfg_req_zp_i,
    input logic [7:0]               cfg_req_min_i,
    input logic [7:0]               cfg_req_max_i,
    input logic [15:0]              cfg_gp_mx_i,
    input logic [4:0]               cfg_gp_shx_i,
    input logic [31:0]              cfg_gp_mout_i,
    input logic [7:0]               cfg_gp_shout_i,
    input logic [7:0]               cfg_gp_zout_i,
    // Packing, taken with cfg_load_i: every set of the layer is packed this way (sienna-packing)
    input logic [2:0]                          cfg_pack_shift_i,
    input logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0] cfg_pack_map_i,
    input logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] cfg_pack_act_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_zp_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_min_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_max_i,
    input logic [PACK_ENTRIES-1:1][15:0]       cfg_pack_mx_i,
    input logic [PACK_ENTRIES-1:1][4:0]        cfg_pack_shx_i,
    input logic [PACK_ENTRIES-1:1][31:0]       cfg_pack_mout_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_shout_i,
    input logic [PACK_ENTRIES-1:1][7:0]        cfg_pack_zout_i,
    output logic                    busy_o,
    output logic                    done_o,  // one cycle: the layer's last result has left

    input  logic                              a_valid_i,
    input  logic [N-1:0][DATA_WIDTH-1:0]            a_data_i,
    output logic                              a_ready_o,
    input  logic                              w_valid_i,
    input  logic [N-1:0][DATA_WIDTH-1:0]            w_data_i,
    output logic                              w_ready_o,
    // int8 only: beside each block's bias beat on the weight stream
    input  logic [N-1:0][ACC_W-1:0]           w_bias_i,
    input  logic [N-1:0][31:0]                w_req_mult_i,
    input  logic [N-1:0][7:0]                 w_req_shift_i,

    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o,
    output logic [NUM_LANES-1:0]                 result_valid_o,
    output logic                                 set_done_o  // a set left the pipeline (partial sums leave with no results)
);
  localparam int HALF = WC_TILES / 2;  // cache tiles per column block; the other half fills with the next block
  localparam int WCTW = $clog2(WC_TILES);
  localparam int WCAW = $clog2(WC_TILES * N * N);
  localparam int ID_W = $clog2(SETS_IN_FLIGHT + 1);
  localparam int ADDR_LINES = $clog2(N * N);
  localparam int RW = (N > 1) ? $clog2(N) : 1;  // row within a tile
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);
  localparam logic [DATA_WIDTH-1:0] ONE = IS_INT ? DATA_WIDTH'(1)  // int8: a residual adds its raw codes into the int32 sum
                                                 : DATA_WIDTH'(((1 << (EXP_W > 0 ? EXP_W - 1 : 0)) - 1) << MAN_W);  // 1.0 in the format

  // ── Configuration and what follows from it ────────────────────────────
  logic [DIM_W-1:0] rt, ct, dt, np;  // row tiles, column blocks, weight tiles per block, passes per output tile
  logic res_q, bias_q, cached, train_q;
  logic [CONTROL_WIDTH-1:0] act_q;
  logic [LFSR_WIDTH-1:0] seed_q;
  logic [7:0] rq_zp_q, rq_min_q, rq_max_q, gp_shout_q, gp_zout_q;  // int8 layer-wide parameters
  logic [15:0] gp_mx_q;
  logic [4:0] gp_shx_q;
  logic [31:0] gp_mout_q;
  logic [2:0] pk_shift_q;  // packing configuration, as the cfg_pack_* ports
  logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0] pk_map_q;
  logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] pk_act_q;
  logic [PACK_ENTRIES-1:1][7:0] pk_zp_q, pk_min_q, pk_max_q, pk_shout_q, pk_zout_q;
  logic [PACK_ENTRIES-1:1][15:0] pk_mx_q;
  logic [PACK_ENTRIES-1:1][4:0] pk_shx_q;
  logic [PACK_ENTRIES-1:1][31:0] pk_mout_q;
  logic [31:0] total_sets;
  logic active;

  function automatic logic [DIM_W-1:0] tiles(input logic [DIM_W-1:0] x);
    return (x + DIM_W'(N - 1)) / DIM_W'(N);
  endfunction

  // Polynomial terms per activation, the same table as model_runner.py's ACTIVATION_TERMS.
  function automatic logic [ADDR_LINES:0] terms(input logic [CONTROL_WIDTH-1:0] a);
    case (a)
      3'b001:  return 14;
      3'b010:  return 15;
      3'b011:  return 30;
      default: return 0;
    endcase
  endfunction

  // ── The pipeline this layer runs on ───────────────────────────────────
  logic                          p_start, p_acc, p_bias_v, p_cached, p_ready, p_complete, p_wc_we;
  logic [N-1:0][ACC_W-1:0]       p_bias;
  logic [N-1:0][31:0]            p_mult;   // int8: the requantize words of the set being started
  logic [N-1:0][7:0]             p_shift;
  logic [N-1:0][DATA_WIDTH-1:0]        p_west, p_north;
  logic [WCTW-1:0]               p_tile;
  logic [WCAW-1:0]               p_wc_addr;
  logic                          p_wc_last;  // the set is its block's last cached pass: the region's fill closes with it
  logic                          p_west_we, p_north_we;
  logic [LFSR_WIDTH-1:0]         p_seed;
  logic [ID_W-1:0]               p_done_id;

  // The pipeline's host link (L0): this layer holds up to two staging credits and puts each set with its last row.
  `include "sienna_set_side.svh"
  set_side_t p_side;
  logic [1:0] l0_cnt;
  credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) p_host ();
  credit_counter #(.MAX(2), .CRW(1)) l0_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(p_host.put), .credit_i(p_host.credit),
                                            .has_credit_o(), .count_o(l0_cnt));
  assign p_ready = (l0_cnt != 0);
  assign p_host.put = p_start;
  assign p_host.data = p_side;
  always_comb begin
    p_side               = '0;
    p_side.wc_last       = p_wc_last;
    p_side.weight_tile   = p_tile;
    p_side.weight_cached = p_cached;
    p_side.accumulate    = p_acc;
    p_side.bias_valid    = p_bias_v;
    p_side.train         = train_q;
    p_side.seed          = p_seed;
    p_side.terms         = terms(act_q);
    p_side.pack_shift    = pk_shift_q;
    p_side.pack_map      = pk_map_q;
    p_side.act           = {pk_act_q, act_q};
    p_side.zp            = {pk_zp_q, rq_zp_q};
    p_side.amin          = {pk_min_q, rq_min_q};
    p_side.amax          = {pk_max_q, rq_max_q};
    p_side.mx            = {pk_mx_q, gp_mx_q};
    p_side.shx           = {pk_shx_q, gp_shx_q};
    p_side.mout          = {pk_mout_q, gp_mout_q};
    p_side.shout         = {pk_shout_q, gp_shout_q};
    p_side.zout          = {pk_zout_q, gp_zout_q};
    p_side.mult          = p_mult;
    p_side.shift         = p_shift;
  end

  // The cache regions (L2): the loader puts once per cached block's fill, on the region of its half, when it holds that region's credit.
  credit_link_if #(.DATA_W(1), .CRW(1)) p_wc[2] ();
  logic wc_has[2];  // per region: its credit is held
  logic wl_put;  // the loader opens its block's fill this cycle
  for (genvar r = 0; r < 2; r++) begin : G_WC
    credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(p_wc[r].put), .credit_i(p_wc[r].credit),
                                           .has_credit_o(), .count_o(wc_has[r]));
    assign p_wc[r].put = wl_put && (wl_blk[0] == 1'(r));
    assign p_wc[r].data = 1'b0;
  end

  // L9: a sink per lane that never stalls, so the outputs leave as they did before the output had credits (Task 6 adds back-pressure).
  credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) p_out[NUM_LANES] ();
  for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
    l9_sink #(.DATA_W(DATA_WIDTH)) sink (.clk_i(clk_i), .rstn_i(rstn_i), .lnk(p_out[l]), .valid_o(result_valid_o[l]),
                                         .data_o(final_result_o[l]));
  end

  sienna_top #(
      .NUM_LANES        (NUM_LANES),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (N),
      .COLLAPSE_K       (COLLAPSE_K),
      .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
      .ACC_BANKS        (ACC_BANKS),
      .RESULT_BANKS     (RESULT_BANKS),
      .WC_TILES         (WC_TILES),
      .EXP_W            (EXP_W),
      .MAN_W            (MAN_W),
      .DATA_WIDTH       (DATA_WIDTH),
      .ACC_W            (ACC_W),
      .CONTROL_WIDTH    (CONTROL_WIDTH),
      .IN_ROWS          (N),
      .IN_COLS          (N),
      .POOL_H           (POOL_H),
      .POOL_W           (POOL_W),
      .STRIDE_ROWS      (STRIDE_ROWS),
      .STRIDE_COLS      (STRIDE_COLS),
      .PADDING          (PADDING),
      .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
      .LFSR_WIDTH       (LFSR_WIDTH),
      .PACK_ENTRIES     (PACK_ENTRIES)
  ) pipe (
      .clk_i                      (clk_i),
      .rstn_i                     (rstn_i),
      .host                       (p_host),
      .bias_i                     (p_bias),
      .wc_region                  (p_wc),
      .wc_write_enable_i          (p_wc_we),
      .wc_write_addr_i            (p_wc_addr),
      .north_write_enable_i       (p_north_we),
      .north_write_data_i         (p_north),
      .north_write_reset_i        (1'b0),
      .west_write_enable_i        (p_west_we),
      .west_write_data_i          (p_west),
      .west_write_reset_i         (1'b0),
      .out                        (p_out),
      .pipeline_complete_o        (p_complete),
      .done_set_id_o              (p_done_id),
      .systolic_busy_o            (),
      .gpnae_busy_o               (),
      .maxpool_busy_o             (),
      .dropout_busy_o             (),
      .intermediate_buffer_full_o (),
      .intermediate_buffer_empty_o()
  );
  assign set_done_o = p_complete;

  // ── Weight loader: bias rows and, for cached layers, each block's tiles into half c%2 of the cache ──
  logic [N-1:0][ACC_W-1:0] bias_buf[2];
  logic [N-1:0][31:0] mult_buf[2];  // int8: the requantize words of the block each half holds
  logic [N-1:0][7:0] shift_buf[2];
  logic [DIM_W-1:0] wl_blk;  // block the loader works on
  logic wl_bias_done;  // its bias row is in
  logic [DIM_W-1:0] wl_tile;  // tiles of it complete
  logic [RW-1:0] wl_row;
  logic [DIM_W-1:0] is_blk;  // block the issuer works on
  logic [DIM_W-1:0] tiles_in[2];  // complete tiles in each half, for the block that half holds
  logic [1:0] bias_in;  // the bias row for the block each half holds is in

  // The loader may start block b once the issuer has finished block b-2, whose half it overwrites, and (cached) has opened the half's fill on its region's credit.
  logic wl_pre, wl_may, wl_take_bias, wl_take_tile, is_needs_north;
  logic wl_opened;  // the loader's block has its region's fill open
  assign wl_pre = active && (wl_blk < ct) && (wl_blk <= is_blk + 1) && (cached || wl_blk == is_blk);
  assign wl_put = wl_pre && cached && !wl_opened && wc_has[wl_blk[0]];
  assign wl_may = wl_pre && (!cached || wl_opened);
  assign wl_take_bias = wl_may && bias_q && !wl_bias_done && w_valid_i;
  assign wl_take_tile = wl_may && cached && (!bias_q || wl_bias_done) && (wl_tile < dt) && w_valid_i && !is_needs_north;

  // ── Set issuer: for each block, row tile and pass, load N rows of A (and B if not cached) and start ──
  logic [DIM_W-1:0] is_r, is_p;
  logic [RW-1:0] is_row;
  logic is_loading;  // part of a set is in; the rest follows
  logic is_eff;  // loading, or a set may begin this cycle
  logic [31:0] n_issued, n_done;
  logic is_res_pass, is_cached_pass, is_need_w, is_row_ok, is_can_start;

  assign is_res_pass    = res_q && (is_p == dt);
  assign is_cached_pass = cached && !is_res_pass;
  assign is_need_w      = !cached && !is_res_pass;  // an uncached layer's weight rows arrive with the set
  assign is_needs_north = is_eff && !is_cached_pass;
  // A set may begin once its bias and (if cached) its weight tile are in and the pipeline has room.
  assign is_can_start = active && (is_blk < ct) && p_ready && (!bias_q || bias_in[is_blk[0]]) &&
                        (!is_cached_pass || tiles_in[is_blk[0]] > is_p);
  assign is_eff = is_loading || is_can_start;
  assign is_row_ok = is_eff && a_valid_i && (!is_need_w || w_valid_i);

  assign a_ready_o = is_row_ok;
  assign w_ready_o = wl_take_bias || wl_take_tile || (is_row_ok && is_need_w);

  // Drive the pipeline: A rows on west; B rows, identity rows or cache fills on north.
  always_comb begin
    p_west_we  = is_row_ok;
    p_west     = a_data_i;
    p_north_we = is_row_ok && !is_cached_pass;
    p_north    = w_data_i;
    if (is_res_pass)
      for (int c = 0; c < N; c++) p_north[c] = (c == int'(is_row)) ? ONE : '0;
    p_wc_we    = wl_take_tile;
    p_wc_addr  = WCAW'((int'(wl_blk[0]) * HALF + int'(wl_tile)) * N * N + int'(wl_row) * N);
    if (wl_take_tile) p_north = w_data_i;
    p_start    = is_row_ok && (is_row == RW'(N - 1));
    p_acc      = (is_p != np - 1);
    p_bias_v   = bias_q && (is_p == 0);
    p_bias     = bias_buf[is_blk[0]];
    p_mult     = mult_buf[is_blk[0]];
    p_shift    = shift_buf[is_blk[0]];
    p_cached   = is_cached_pass;
    p_wc_last  = is_cached_pass && (is_r == rt - 1) && (is_p == dt - 1);
    p_tile     = WCTW'(int'(is_blk[0]) * HALF + int'(is_p));
    p_seed     = LFSR_WIDTH'(seed_q ^ (32'h85EBCA6B * n_issued));
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      active <= 1'b0;
      rt <= '0;
      ct <= '0;
      dt <= '0;
      np <= '0;
      res_q <= 1'b0;
      bias_q <= 1'b0;
      cached <= 1'b0;
      train_q <= 1'b0;
      act_q <= '0;
      seed_q <= '0;
      rq_zp_q <= '0;
      rq_min_q <= '0;
      rq_max_q <= '0;
      gp_mx_q <= '0;
      gp_shx_q <= '0;
      gp_mout_q <= '0;
      gp_shout_q <= '0;
      gp_zout_q <= '0;
      pk_shift_q <= '0;
      pk_map_q <= '0;
      pk_act_q <= '0;
      pk_zp_q <= '0;
      pk_min_q <= '0;
      pk_max_q <= '0;
      pk_mx_q <= '0;
      pk_shx_q <= '0;
      pk_mout_q <= '0;
      pk_shout_q <= '0;
      pk_zout_q <= '0;
      total_sets <= '0;
      wl_blk <= '0;
      wl_bias_done <= 1'b0;
      wl_tile <= '0;
      wl_row <= '0;
      wl_opened <= 1'b0;
      tiles_in[0] <= '0;
      tiles_in[1] <= '0;
      bias_in <= '0;
      is_blk <= '0;
      is_r <= '0;
      is_p <= '0;
      is_row <= '0;
      is_loading <= 1'b0;
      n_issued <= '0;
      n_done <= '0;
      done_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      if (cfg_load_i && !active) begin
        automatic logic [DIM_W-1:0] r = tiles(cfg_m_i), c = tiles(cfg_n_i), d = tiles(cfg_kb_i);
        active <= 1'b1;
        rt <= r;
        ct <= c;
        dt <= d;
        np <= d + DIM_W'(cfg_residual_i);
        res_q <= cfg_residual_i;
        bias_q <= cfg_bias_i || IS_INT;  // int8: every block's bias beat carries its requantize words
        cached <= (r > 1) && (d <= DIM_W'(HALF));
        train_q <= cfg_train_i;
        act_q <= cfg_act_i;
        seed_q <= cfg_seed_i;
        rq_zp_q <= cfg_req_zp_i;
        rq_min_q <= cfg_req_min_i;
        rq_max_q <= cfg_req_max_i;
        gp_mx_q <= cfg_gp_mx_i;
        gp_shx_q <= cfg_gp_shx_i;
        gp_mout_q <= cfg_gp_mout_i;
        gp_shout_q <= cfg_gp_shout_i;
        gp_zout_q <= cfg_gp_zout_i;
        pk_shift_q <= cfg_pack_shift_i;
        pk_map_q <= cfg_pack_map_i;
        pk_act_q <= cfg_pack_act_i;
        pk_zp_q <= cfg_pack_zp_i;
        pk_min_q <= cfg_pack_min_i;
        pk_max_q <= cfg_pack_max_i;
        pk_mx_q <= cfg_pack_mx_i;
        pk_shx_q <= cfg_pack_shx_i;
        pk_mout_q <= cfg_pack_mout_i;
        pk_shout_q <= cfg_pack_shout_i;
        pk_zout_q <= cfg_pack_zout_i;
        total_sets <= 32'(r) * 32'(c) * 32'(d + DIM_W'(cfg_residual_i));
        wl_blk <= '0;
        wl_bias_done <= 1'b0;
        wl_tile <= '0;
        wl_row <= '0;
        wl_opened <= 1'b0;
        tiles_in[0] <= '0;
        tiles_in[1] <= '0;
        bias_in <= '0;
        is_blk <= '0;
        is_r <= '0;
        is_p <= '0;
        is_row <= '0;
        is_loading <= 1'b0;
        n_issued <= '0;
        n_done <= '0;
      end else if (active) begin
        // Loader: a block is loaded once its bias (if any) and, for a cached layer, its tiles are in.
        begin
          automatic logic bias_now = wl_bias_done || wl_take_bias;
          automatic logic last_row = wl_take_tile && (wl_row == RW'(N - 1));
          automatic logic [DIM_W-1:0] tiles_now = wl_tile + DIM_W'(last_row);
          if (wl_take_bias) begin
            for (int c = 0; c < N; c++) bias_buf[wl_blk[0]][c] <= IS_INT ? w_bias_i[c] : ACC_W'(w_data_i[c]);
            bias_in[wl_blk[0]] <= 1'b1;
          end
          if (last_row) tiles_in[wl_blk[0]] <= tiles_now;
          if (wl_put) wl_opened <= 1'b1;
          if (wl_may && (!bias_q || bias_now) && (!cached || tiles_now == dt)) begin
            wl_blk <= wl_blk + 1'b1;
            wl_bias_done <= 1'b0;
            wl_tile <= '0;
            wl_row <= '0;
            wl_opened <= 1'b0;
          end else begin
            if (wl_take_bias) wl_bias_done <= 1'b1;
            if (wl_take_tile) begin
              wl_row <= last_row ? '0 : wl_row + 1'b1;
              wl_tile <= tiles_now;
            end
          end
        end
        // Issuer: a set's first row may come in the cycle it is allowed to begin.
        if (is_row_ok && is_row == RW'(N - 1)) is_loading <= 1'b0;
        else if (is_eff) is_loading <= 1'b1;
        if (is_row_ok) begin
          if (is_row == RW'(N - 1)) begin
            is_row <= '0;
            n_issued <= n_issued + 1;
            if (is_p == np - 1) begin
              is_p <= '0;
              if (is_r == rt - 1) begin
                is_r <= '0;
                is_blk <= is_blk + 1'b1;
                // This half is free for block is_blk + 2 once its sets are broadcast.
                tiles_in[is_blk[0]] <= '0;
                bias_in[is_blk[0]] <= 1'b0;
              end else is_r <= is_r + 1'b1;
            end else is_p <= is_p + 1'b1;
          end else is_row <= is_row + 1'b1;
        end
        if (p_complete) n_done <= n_done + 1;
        if (n_issued == total_sets && n_done == total_sets && is_blk == ct) begin
          active <= 1'b0;
          done_o <= 1'b1;
        end
      end
    end
  end

  // D-8: the requantize words carry no reset, so rstn_i is no hold enable on them; readers wait for bias_in, set by the same beat.
  always_ff @(posedge clk_i)
    if (wl_take_bias) begin
      mult_buf[wl_blk[0]]  <= w_req_mult_i;
      shift_buf[wl_blk[0]] <= w_req_shift_i;
    end

  assign busy_o = active;

`ifndef SYNTHESIS
  a_pack_shape: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                 (cfg_load_i && !active && cfg_pack_shift_i != '0) |-> (cfg_n_i == DIM_W'(N) && cfg_kb_i == DIM_W'(N) && !cfg_residual_i))
    else $error("sienna_layer: a packed layer is N columns and N deep, without a residual");
  a_one_north: assert property (@(posedge clk_i) disable iff (!rstn_i) !(p_wc_we && p_north_we))
    else $error("sienna_layer: a cache fill and a set's B row on the north bus in one cycle");
  if (IS_INT) begin : G_INT_NO_RESIDUAL  // int8 residual adds raw codes: its rescale is 2b
    a_int_no_residual: assert property (@(posedge clk_i) disable iff (!rstn_i) !(cfg_load_i && !active && cfg_residual_i))
      else $error("sienna_layer: int8 residual is not supported until 2b");
  end
`endif

endmodule
