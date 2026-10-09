`timescale 1ns / 100ps

// One network layer as C = A x B (+ bias, + residual), scheduled entirely in hardware.
// The host writes a configuration, then streams data in a fixed order on two row links (L10); every per-set decision is made here.
//
// Weight stream (w_rows), per column block c of N output columns: the bias row (if cfg_bias_i), then the block's
// ceil(cfg_kb_i/N) weight tiles of N rows each, once if the block is cached, else once per row tile.
// A block is cached when it has more than one row tile and at most WC_TILES/2 weight tiles.
// Activation stream (a_rows), per column block, per row tile: the depth tiles of A (N rows each), then the residual tile if cfg_residual_i.
// Row credits, in stream order: a set's N A rows (and N B rows if uncached) while an L0 credit is held, a bias row's one, a cache tile's N.
// int8: every block has a bias beat, its int32 bias and requantize words on w_bias_i, w_req_mult_i, w_req_shift_i with its put (its row ignored).
// Results leave per output tile on the output links (L9), column blocks outer and row tiles inner; lane L's j-th word of a set is word j*NUM_LANES + L.
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
    parameter int ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // bias and sums: int32 in int8, fp32 in every float format (the bias row widened exactly)
    parameter int CONTROL_WIDTH     = 3,
    parameter int LFSR_WIDTH        = 32,
    parameter int POOL_H            = 1,
    parameter int POOL_W            = 1,
    parameter int STRIDE_ROWS       = 1,
    parameter int STRIDE_COLS       = 1,
    parameter int PADDING           = 0,
    parameter int DROPOUT_P_PERCENT = 50,
    parameter int DIM_W             = 16,  // width of the configured sizes
    parameter int PACK_ENTRIES      = 8,
    parameter int LINK_STAGES       = 0,   // sienna_top's register stages on L0, L1, L3 and L9; L10 is not staged
    parameter int OUT_MAX           = 64,  // the most L9 credits the consumer may grant per lane
    parameter int OUT_CRW           = 1    // L9 credit width
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

    // L10 puts must come from registered state: a put reaches a_rows.credit and w_rows.credit combinationally inside this module.
    credit_link_if.consumer          a_rows,  // L10: one row of N words per put; a set's N credits may come with the last row's put (count 1 + N - 1 = N after it), so the producer counter needs MAX >= N
    credit_link_if.consumer          w_rows,  // L10: one row of N words per put, in the weight stream's order; two cache tiles may be granted, so the producer counter needs MAX 2N
    // int8 only: with each block's bias row on the weight stream
    input  logic [N-1:0][ACC_W-1:0]  w_bias_i,
    input  logic [N-1:0][31:0]       w_req_mult_i,
    input  logic [N-1:0][7:0]        w_req_shift_i,

    credit_link_if.producer          out[NUM_LANES],  // L9: lane k's results, one word per put while the consumer's credit is held
    output logic                     set_done_o  // a set left the pipeline with its last word (partial sums leave with no results)
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
  localparam int RCW = $clog2(N + 1);  // row credits: a set's or a tile's N in one cycle
  localparam int TOW = $clog2(2 * N + 1);  // cache tile rows outstanding: at most two tiles

`ifndef SYNTHESIS  // interface widths are not elaboration constants in Verilator, so the row links are checked at time 0
  initial
    if ($bits(a_rows.data) != N * DATA_WIDTH || $bits(a_rows.credit) != RCW || $bits(w_rows.data) != N * DATA_WIDTH ||
        $bits(w_rows.credit) != RCW)
      $fatal(1, "sienna_layer: the row links need data %0d bits and credit %0d, found A %0d and %0d, W %0d and %0d", N * DATA_WIDTH, RCW,
             $bits(a_rows.data), $bits(a_rows.credit), $bits(w_rows.data), $bits(w_rows.credit));
`endif
  logic [N-1:0][DATA_WIDTH-1:0] a_data_i, w_data_i;  // the rows put this cycle
  assign a_data_i = a_rows.data;
  assign w_data_i = w_rows.data;
  logic live;  // out of reset for a cycle: no row credit before it
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) live <= 1'b0;
    else live <= 1'b1;

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
  logic                          p_start, p_acc, p_bias_v, p_cached, p_complete, p_wc_we;
  logic [N-1:0][ACC_W-1:0]       p_bias;
  logic [N-1:0][31:0]            p_mult;   // int8: the requantize words of the set being started
  logic [N-1:0][7:0]             p_shift;
  logic [N-1:0][DATA_WIDTH-1:0]        p_west, p_north;
  logic [WCTW-1:0]               p_tile;
  logic [WCAW-1:0]               p_wc_addr;
  logic                          p_wc_last;  // the set is its block's last cached pass: the region's fill closes with it
  logic                          p_west_we, p_north_we;
  logic [LFSR_WIDTH-1:0]         p_seed;

  // The pipeline's host link (L0): this layer holds up to two staging credits and puts each set with its last row.
  `include "sienna_set_side.svh"
  set_side_t p_side;
  logic [1:0] l0_cnt;
  credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) p_host ();
  credit_counter #(.MAX(2), .CRW(1)) l0_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(p_host.put), .credit_i(p_host.credit),
                                            .has_credit_o(), .count_o(l0_cnt));
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
      .PACK_ENTRIES     (PACK_ENTRIES),
      .LINK_STAGES      (LINK_STAGES),
      .OUT_MAX          (OUT_MAX),
      .OUT_CRW          (OUT_CRW)
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
      .out                        (out),
      .pipeline_complete_o        (p_complete),
      .done_set_id_o              (),
      .systolic_busy_o            (),
      .gpnae_busy_o               (),
      .maxpool_busy_o             (),
      .dropout_busy_o             ()
  );
  assign set_done_o = p_complete;

  // ── Weight loader: bias rows and, for cached layers, each block's tiles into half c%2 of the cache ──
  logic [N-1:0][ACC_W-1:0] bias_buf[2];
  if ($bits(bias_buf[0][0]) != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_BIAS_W  // the widened bias would truncate or pad silently
    $fatal(1, "sienna_layer: bias words are %0d bits, the mesh's bias input %0d", $bits(bias_buf[0][0]), sienna_fmt_pkg::acc_w(EXP_W, MAN_W));
  end
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
  // Uncached, block b's bias row follows block b-1's last B row in the stream, so its credit waits until that set's are granted.
  logic wl_pre, wl_may, wl_take_bias, wl_take_tile, wl_bias_due;
  logic wl_opened;  // the loader's block has its region's fill open
  logic wb_gr;  // the loader's bias row is granted, not yet in
  logic [31:0] wt_gr;  // the loader's tile rows granted so far
  logic [TOW-1:0] wt_out;  // tile rows granted, not yet in
  logic wb_grant, wt_grant;
  logic [DIM_W-1:0] g_blk, g_p;  // the set the next grant is for
  assign wl_pre = active && (wl_blk < ct) && (wl_blk <= is_blk + 1) && (cached || wl_blk <= g_blk);
  assign wl_put = wl_pre && cached && !wl_opened && wc_has[wl_blk[0]];
  assign wl_may = wl_pre && (!cached || wl_opened);
  assign wl_bias_due = wl_pre && bias_q && !wl_bias_done;  // the next weight row the loader expects is its block's bias

  // ── Set issuer: for each block, row tile and pass, N rows of A (and B if not cached) on the credits it grants, then the start ──
  logic [DIM_W-1:0] is_r, is_p;  // the set being received, or the next to grant when none is
  logic [31:0] n_issued, n_done;
  logic is_res_pass, is_cached_pass;
  logic pend;  // a set is granted and not yet started
  logic [RCW-1:0] a_need, w_need;  // its rows still to come
  logic [RW-1:0] a_row;  // its A rows in, for the residual pass's identity row
  logic a_take, w_set_row;  // an A row; a weight row that is the set's B row
  logic s_grant;  // a set's row credits go out this cycle

  assign is_res_pass    = res_q && (is_p == dt);
  assign is_cached_pass = cached && !is_res_pass;
  assign a_take         = a_rows.put;
  // A weight row is the pending set's B row while it needs any (they precede the next bias row in the stream), else the loader's.
  logic w_is_b;
  assign w_is_b         = !cached && pend && (w_need != '0);
  assign wl_take_bias   = w_rows.put && !w_is_b && wl_bias_due;
  assign wl_take_tile   = w_rows.put && cached && !wl_bias_due;
  assign w_set_row      = w_rows.put && w_is_b;
  assign p_start        = pend && (a_need == RCW'(int'(a_take))) && (w_need == RCW'(int'(w_set_row)));

  // The set the next grant is for: the one after the pending set, else the issuer's.
  always_comb begin
    g_blk = is_blk;
    g_p   = is_p;
    if (pend) begin
      if (is_p == np - 1) begin
        g_p = '0;
        if (is_r == rt - 1) g_blk = is_blk + 1'b1;
      end else g_p = is_p + 1'b1;
    end
  end
  logic g_res, g_need_w, g_l0_ok, g_bias_ok, g_tile_ok, g_ok;
  assign g_res     = res_q && (g_p == dt);
  assign g_need_w  = !cached && !g_res;  // an uncached layer's B rows come with the set
  assign g_l0_ok   = int'(l0_cnt) > int'(pend);  // an L0 credit held beyond the pending set's
  assign g_bias_ok = !bias_q || bias_in[g_blk[0]] || (wl_take_bias && wl_blk == g_blk);  // its bias row in, or arriving now
  assign g_tile_ok = !(cached && !g_res) || tiles_in[g_blk[0]] > g_p;
  assign g_ok      = live && active && (g_blk < ct) && (!pend || p_start) && g_l0_ok && g_bias_ok && g_tile_ok;
  // A residual pass writes identity rows on the north bus, so it waits for no cache tile row to be outstanding, and holds back new tile grants.
  assign s_grant   = g_ok && (!g_res || wt_out == '0);
  assign wb_grant  = live && wl_may && bias_q && !wl_bias_done && !wb_gr;
  assign wt_grant  = live && wl_may && cached && (!bias_q || wl_bias_done) && (wt_gr < 32'(dt) * 32'(N)) && (int'(wt_out) <= N) &&
                     !(g_ok && g_res) && !(pend && is_res_pass);
  // One grant per stream per cycle: a set's, a bias row's or a tile's are never due together, so the counts never add.
  assign a_rows.credit = s_grant ? RCW'(N) : '0;
  assign w_rows.credit = (s_grant && g_need_w) || wt_grant ? RCW'(N) : RCW'(int'(wb_grant));

  // Drive the pipeline: A rows on west; B rows, identity rows or cache fills on north.
  always_comb begin
    p_west_we  = a_take;
    p_west     = a_data_i;
    p_north_we = (a_take && is_res_pass) || w_set_row;
    p_north    = w_data_i;
    if (a_take && is_res_pass)
      for (int c = 0; c < N; c++) p_north[c] = (c == int'(a_row)) ? ONE : '0;
    p_wc_we    = wl_take_tile;
    p_wc_addr  = WCAW'((int'(wl_blk[0]) * HALF + int'(wl_tile)) * N * N + int'(wl_row) * N);
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
      wb_gr <= 1'b0;
      wt_gr <= '0;
      wt_out <= '0;
      pend <= 1'b0;
      a_need <= '0;
      w_need <= '0;
      a_row <= '0;
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
        wb_gr <= 1'b0;
        wt_gr <= '0;
        wt_out <= '0;
        pend <= 1'b0;
        a_need <= '0;
        w_need <= '0;
        a_row <= '0;
        n_issued <= '0;
        n_done <= '0;
      end else if (active) begin
        // Loader: a block is loaded once its bias (if any) and, for a cached layer, its tiles are in.
        begin
          automatic logic bias_now = wl_bias_done || wl_take_bias;
          automatic logic last_row = wl_take_tile && (wl_row == RW'(N - 1));
          automatic logic [DIM_W-1:0] tiles_now = wl_tile + DIM_W'(last_row);
          if (wl_take_bias) begin
            for (int c = 0; c < N; c++) bias_buf[wl_blk[0]][c] <= IS_INT ? w_bias_i[c] : sienna_fmt_pkg::widen(32'(w_data_i[c]), MAN_W);
            bias_in[wl_blk[0]] <= 1'b1;
          end
          if (last_row) tiles_in[wl_blk[0]] <= tiles_now;
          if (wl_put) wl_opened <= 1'b1;
          wt_out <= wt_out + (wt_grant ? TOW'(N) : '0) - TOW'(wl_take_tile);
          if (wl_may && (!bias_q || bias_now) && (!cached || tiles_now == dt)) begin
            wl_blk <= wl_blk + 1'b1;
            wl_bias_done <= 1'b0;
            wl_tile <= '0;
            wl_row <= '0;
            wl_opened <= 1'b0;
            wb_gr <= 1'b0;
            wt_gr <= '0;
          end else begin
            if (wl_take_bias) wl_bias_done <= 1'b1;
            if (wb_grant) wb_gr <= 1'b1;
            else if (wl_take_bias) wb_gr <= 1'b0;
            if (wt_grant) wt_gr <= wt_gr + 32'(N);
            if (wl_take_tile) begin
              wl_row <= last_row ? '0 : wl_row + 1'b1;
              wl_tile <= tiles_now;
            end
          end
        end
        // Issuer: a grant opens a set; its rows count down; it starts with the last of them.
        if (s_grant) begin
          pend <= 1'b1;
          a_need <= RCW'(N);
          w_need <= g_need_w ? RCW'(N) : '0;
          a_row <= '0;
        end else begin
          if (p_start) pend <= 1'b0;
          if (a_take) begin
            a_need <= a_need - 1'b1;
            a_row <= a_row + 1'b1;
          end
          if (w_set_row) w_need <= w_need - 1'b1;
        end
        if (p_start) begin
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
  // A row's put and whether its stream held a credit for it, registered: a host may drive its puts at the edge.
  logic ap_q, ap_ok_q, wp_q, wp_ok_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {ap_q, ap_ok_q, wp_q, wp_ok_q} <= '0;
    else {ap_q, ap_ok_q, wp_q, wp_ok_q} <= {a_take, pend && a_need != '0, w_rows.put,
                                            w_is_b || (cached ? (wl_bias_due ? wb_gr : wt_out != '0) : (wl_bias_due && wb_gr))};
  a_a_granted: assert property (@(posedge clk_i) disable iff (!rstn_i) ap_q |-> ap_ok_q)
    else $error("sienna_layer: an A row arrived with no A row credit granted (no set granted, or its N rows already in)");
  a_w_granted: assert property (@(posedge clk_i) disable iff (!rstn_i) wp_q |-> wp_ok_q)
    else $error("sienna_layer: a weight row arrived with no credit granted for the row the stream is at (bias, cache tile or a set's B row)");
  // The weight credit mux sends one grant: a bias row's must never come with a set's B rows or a tile's.
  logic wgb_q, wgn_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {wgb_q, wgn_q} <= '0;
    else {wgb_q, wgn_q} <= {wb_grant, (s_grant && g_need_w) || wt_grant};
  a_w_one_grant: assert property (@(posedge clk_i) disable iff (!rstn_i) !(wgb_q && wgn_q))
    else $error("sienna_layer: a bias row's weight credit and a set's or tile's N in one cycle: the credit mux sent only N");
  if (IS_INT) begin : G_INT_NO_RESIDUAL  // int8 residual adds raw codes: its rescale is 2b
    a_int_no_residual: assert property (@(posedge clk_i) disable iff (!rstn_i) !(cfg_load_i && !active && cfg_residual_i))
      else $error("sienna_layer: int8 residual is not supported until 2b");
  end
  // A float bias row word must land in bias_buf as the word followed by zeros, -0 and subnormals included.
  logic chk_bias_v;  // registered copy of the loader's bias write (active && wl_take_bias), float builds
  logic chk_bias_h;
  logic [N-1:0][DATA_WIDTH-1:0] chk_bias_row;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) chk_bias_v <= 1'b0;
    else chk_bias_v <= active && wl_take_bias && !IS_INT;
  always_ff @(posedge clk_i) begin  // no reset: read only with chk_bias_v
    chk_bias_h <= wl_blk[0];
    chk_bias_row <= w_data_i;
  end
  for (genvar c = 0; c < N; c++) begin : G_A_BIAS
    a_bias_widened: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                     chk_bias_v |-> bias_buf[chk_bias_h][c] == (ACC_W'(chk_bias_row[c]) << (ACC_W - DATA_WIDTH)))
      else $error("sienna_layer: bias column %0d holds %h, the row word %h widened is %h", c, bias_buf[chk_bias_h][c], chk_bias_row[c],
                  ACC_W'(chk_bias_row[c]) << (ACC_W - DATA_WIDTH));
  end
`endif

endmodule
