`timescale 1ns / 100ps

// COPIES independent SIENNA pipelines behind one host: sets go to the copies in turn, one set per put.
// Each copy has its own host link (L0), output links (L9) and set ids; nothing merges the copies' outputs in order.
// L2 contract: a fill goes to every copy, so its sets are put one after another, at least COPIES of them, and each copy's last one is marked wc_last.
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
    parameter int OUT_W             = sienna_fmt_pkg::out_w(EXP_W, MAN_W),  // each copy's mesh result words, as sienna_top
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
    parameter int LFSR_WIDTH        = 32,
    parameter int LINK_STAGES       = 0,   // each copy's register stages on L0, L1, L3 and L9, as sienna_top
    parameter int OUT_MAX           = 64,  // the most L9 credits the consumer may grant per lane
    parameter int OUT_CRW           = 1    // L9 credit width
) (
    input logic clk_i,
    input logic rstn_i,

    credit_link_if.consumer         host[COPIES],  // L0 per copy, as sienna_top's: put only the copy whose turn it is (copy_sel_o), its rows went there
    input logic [N-1:0][ACC_W-1:0]  bias_i,  // with a put, to the copy put; float builds: fp32 bits, as sienna_top
    credit_link_if.consumer         wc_region[2],  // L2: a put opens a fill of that region in every copy; its credit returns once every copy's has
    input logic                     wc_write_enable_i,  // written into every copy's cache
    input logic [$clog2(WC_TILES*N*N)-1:0] wc_write_addr_i,
    input logic                     north_write_enable_i,  // the row buses reach only the copy whose turn it is
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                     west_write_reset_i,

    input logic                     drained_i,  // checks only: the host has put its last set; no fill may be left reaching only some copies
    output logic [$clog2(COPIES+1)-1:0]          copy_sel_o,  // copy whose turn it is: the host loads its rows and puts it next
    credit_link_if.producer         out[COPIES*NUM_LANES],  // L9 per copy: copy c's lane l is out[c*NUM_LANES + l]
    output logic [COPIES-1:0]                                pipeline_complete_o,
    output logic [COPIES-1:0][$clog2(SETS_IN_FLIGHT+1)-1:0]  done_set_id_o
);
  localparam int SW = $clog2(COPIES + 1);
  localparam int PACK_ENTRIES = 8;  // sienna_top's default; sienna_multi never packs
  localparam int WCTW = $clog2(WC_TILES);
  `include "sienna_set_side.svh"
  logic [SW-1:0] sel;
  logic [COPIES-1:0] put;  // per copy: the host's put this cycle
  assign copy_sel_o = sel;

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
  logic [COPIES-1:0] wc_cred[2];  // this cycle's region credits from each copy
  logic [COPIES-1:0] wc_got[2];  // copies whose region credit came back and has not gone up yet
  logic [1:0] wc_put;  // the host opens a fill of the region this cycle
  for (genvar r = 0; r < 2; r++) begin : G_WC
    logic all_back;
    assign wc_put[r] = wc_region[r].put;
    assign all_back = live && &(wc_got[r] | wc_cred[r]);
    assign wc_region[r].credit = all_back;
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) wc_got[r] <= '0;
      else wc_got[r] <= all_back ? '0 : (wc_got[r] | wc_cred[r]);
  end

`ifndef SYNTHESIS
  logic [COPIES-1:0] p_cached, p_last, p_region;  // per copy: the put set reads the cache, is marked its fill's last, and its region
`endif
  for (genvar c = 0; c < COPIES; c++) begin : COPY
    logic mine;
    assign mine = (sel == SW'(c));
    assign put[c] = host[c].put;
`ifndef SYNTHESIS
    set_side_t side;
    assign side = host[c].data;
    assign p_cached[c] = side.weight_cached;
    assign p_last[c] = side.wc_last;
    assign p_region[c] = side.weight_tile[WCTW-1];
`endif
    credit_link_if #(.DATA_W(1), .CRW(1)) wcc[2] ();
    for (genvar r = 0; r < 2; r++) begin : G_WCC
      assign wcc[r].put = wc_region[r].put;
      assign wcc[r].data = wc_region[r].data;
      assign wc_cred[r][c] = wcc[r].credit;
    end
    credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) outl[NUM_LANES] ();
    for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
      assign out[c*NUM_LANES+l].put = outl[l].put;
      assign out[c*NUM_LANES+l].data = outl[l].data;
      assign outl[l].credit = out[c*NUM_LANES+l].credit;
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
        .OUT_W            (OUT_W),
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
        .LFSR_WIDTH       (LFSR_WIDTH),
        .LINK_STAGES      (LINK_STAGES),
        .OUT_MAX          (OUT_MAX),
        .OUT_CRW          (OUT_CRW)
    ) pipe (
        .clk_i                      (clk_i),
        .rstn_i                     (rstn_i),
        .host                       (host[c]),
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
        .out                        (outl),
        .pipeline_complete_o        (pipeline_complete_o[c]),
        .done_set_id_o              (done_set_id_o[c]),
        .systolic_busy_o            (),
        .gpnae_busy_o               (),
        .maxpool_busy_o             (),
        .dropout_busy_o             ()
    );
  end

`ifndef SYNTHESIS
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
    else $error("sienna_multi: a put to a copy whose turn it is not (copy %0d put, turn %0d): its rows went to the copy whose turn it was",
                pq_copy, $past(sel));
  a_wc_after_last: assert property (@(posedge clk_i) disable iff (!rstn_i) pq |-> !pq_after_last)
    else $error("sienna_multi: copy %0d took a cached set of region %0d after its marked one, or with no fill open: its region may be refilled under it",
                pq_copy, pq_region);
  a_wc_fill_short: assert property (@(posedge clk_i) disable iff (!rstn_i) pq |-> !pq_short)
    else $error("sienna_multi: region %0d's fill reached only some copies with their marked sets when another set was put: the rest never close it",
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
    else $error("sienna_multi: drained with region %0d's fill reaching only copies %b with their marked sets: its credit never returns",
                dq_region, dq_marked);
`endif

endmodule
