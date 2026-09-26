`timescale 1ns / 100ps

// One network layer as C = A x B (+ bias, + residual), scheduled entirely in hardware.
// The host writes a configuration, then streams data in a fixed order; every per-set decision is made here.
//
// Weight stream, per column block c of N output columns: the bias row (if cfg_bias_i), then the block's
// ceil(cfg_kb_i/N) weight tiles of N rows each, once if the block is cached, else once per row tile.
// A block is cached when it has more than one row tile and at most WC_TILES/2 weight tiles.
// Activation stream, per column block, per row tile: the depth tiles of A (N rows each), then the residual tile if cfg_residual_i.
// Results leave per output tile, column blocks outer and row tiles inner, N x N row-major.
module sienna_layer #(
    parameter int NUM_LANES         = 32,
    parameter int N                 = 16,
    parameter int TILE_SIZE         = 4,
    parameter int COLLAPSE_K        = 1,
    parameter int SETS_IN_FLIGHT    = 8,
    parameter int WC_TILES          = 128,
    parameter int DATA_WIDTH        = 32,
    parameter int CONTROL_WIDTH     = 3,
    parameter int LFSR_WIDTH        = 32,
    parameter int POOL_H            = 1,
    parameter int POOL_W            = 1,
    parameter int STRIDE_ROWS       = 1,
    parameter int STRIDE_COLS       = 1,
    parameter int PADDING           = 0,
    parameter int DROPOUT_P_PERCENT = 50,
    parameter int DIM_W             = 16  // width of the configured sizes
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
    output logic                    busy_o,
    output logic                    done_o,  // one cycle: the layer's last result has left

    input  logic                              a_valid_i,
    input  logic [N-1:0][DATA_WIDTH-1:0]      a_data_i,
    output logic                              a_ready_o,
    input  logic                              w_valid_i,
    input  logic [N-1:0][DATA_WIDTH-1:0]      w_data_i,
    output logic                              w_ready_o,

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
  localparam logic [DATA_WIDTH-1:0] ONE = 32'h3f800000;

  // ── Configuration and what follows from it ────────────────────────────
  logic [DIM_W-1:0] rt, ct, dt, np;  // row tiles, column blocks, weight tiles per block, passes per output tile
  logic res_q, bias_q, cached, train_q;
  logic [CONTROL_WIDTH-1:0] act_q;
  logic [LFSR_WIDTH-1:0] seed_q;
  logic [31:0] total_sets;
  logic active;

  function automatic logic [DIM_W-1:0] tiles(input logic [DIM_W-1:0] x);
    return (x + DIM_W'(N - 1)) / DIM_W'(N);
  endfunction

  // Polynomial terms per activation, the same table regression.py uses.
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
  logic [N-1:0][DATA_WIDTH-1:0]  p_bias, p_west, p_north;
  logic [WCTW-1:0]               p_tile;
  logic [WCAW-1:0]               p_wc_addr;
  logic [1:0]                    p_region_busy;
  logic                          p_west_we, p_north_we;
  logic [LFSR_WIDTH-1:0]         p_seed;
  logic [ID_W-1:0]               p_done_id;

  sienna_top #(
      .NUM_LANES        (NUM_LANES),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (N),
      .COLLAPSE_K       (COLLAPSE_K),
      .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
      .WC_TILES         (WC_TILES),
      .DATA_WIDTH       (DATA_WIDTH),
      .CONTROL_WIDTH    (CONTROL_WIDTH),
      .IN_ROWS          (N),
      .IN_COLS          (N),
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
      .start_pipeline_i           (p_start),
      .training_mode_i            (train_q),
      .accumulate_i               (p_acc),
      .bias_valid_i               (p_bias_v),
      .bias_i                     (p_bias),
      .weight_cached_i            (p_cached),
      .weight_tile_i              (p_tile),
      .wc_write_enable_i          (p_wc_we),
      .wc_write_addr_i            (p_wc_addr),
      .wc_region_busy_o           (p_region_busy),
      .dropout_seed_i             (p_seed),
      .activation_function_i      (act_q),
      .num_terms_i                (terms(act_q)),
      .north_write_enable_i       (p_north_we),
      .north_write_data_i         (p_north),
      .north_write_reset_i        (1'b0),
      .west_write_enable_i        (p_west_we),
      .west_write_data_i          (p_west),
      .west_write_reset_i         (1'b0),
      .final_result_o             (final_result_o),
      .result_valid_o             (result_valid_o),
      .pipeline_complete_o        (p_complete),
      .pipeline_ready_o           (p_ready),
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
  logic [N-1:0][DATA_WIDTH-1:0] bias_buf[2];
  logic [DIM_W-1:0] wl_blk;  // block the loader works on
  logic wl_bias_done;  // its bias row is in
  logic [DIM_W-1:0] wl_tile;  // tiles of it complete
  logic [RW-1:0] wl_row;
  logic [DIM_W-1:0] is_blk;  // block the issuer works on
  logic [DIM_W-1:0] tiles_in[2];  // complete tiles in each half, for the block that half holds
  logic [1:0] bias_in;  // the bias row for the block each half holds is in

  // The loader may start block b once the issuer has finished block b-2, whose half it overwrites,
  // and the mesh no longer has a staged set reading that half; an uncached layer's weights come with its sets.
  logic wl_may, wl_take_bias, wl_take_tile, is_needs_north;
  assign wl_may = active && (wl_blk < ct) && (wl_blk <= is_blk + 1) && !p_region_busy[wl_blk[0]] &&
                  (cached || wl_blk == is_blk);
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
    p_cached   = is_cached_pass;
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
      total_sets <= '0;
      wl_blk <= '0;
      wl_bias_done <= 1'b0;
      wl_tile <= '0;
      wl_row <= '0;
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
        bias_q <= cfg_bias_i;
        cached <= (r > 1) && (d <= DIM_W'(HALF));
        train_q <= cfg_train_i;
        act_q <= cfg_act_i;
        seed_q <= cfg_seed_i;
        total_sets <= 32'(r) * 32'(c) * 32'(d + DIM_W'(cfg_residual_i));
        wl_blk <= '0;
        wl_bias_done <= 1'b0;
        wl_tile <= '0;
        wl_row <= '0;
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
            bias_buf[wl_blk[0]] <= w_data_i;
            bias_in[wl_blk[0]] <= 1'b1;
          end
          if (last_row) tiles_in[wl_blk[0]] <= tiles_now;
          if (wl_may && (!bias_q || bias_now) && (!cached || tiles_now == dt)) begin
            wl_blk <= wl_blk + 1'b1;
            wl_bias_done <= 1'b0;
            wl_tile <= '0;
            wl_row <= '0;
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

  assign busy_o = active;

`ifndef SYNTHESIS
  a_start_taken: assert property (@(posedge clk_i) disable iff (!rstn_i) p_start |-> p_ready)
    else $error("sienna_layer: a set's last row came when the pipeline could not take its start");
  a_one_north: assert property (@(posedge clk_i) disable iff (!rstn_i) !(p_wc_we && p_north_we))
    else $error("sienna_layer: a cache fill and a set's B row on the north bus in one cycle");
`endif

endmodule
