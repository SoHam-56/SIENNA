`timescale 1ns / 100ps

import test_config_pkg::*;

`include "tb_l9_sink.svh"

// Runs one layer on sienna_top's layer contract (SET_PORT = 0): writes the configuration, streams the two inputs from +layer=<file> on the row links (L10)
// as the layer grants credits, and writes every result taken from the output links (L9) to +out=<file>.
// Layer file: "L m kb n residual bias act train seed a_rows w_rows packed", then a_rows and w_rows rows of N hex words.
// packed: a "P sh m0 .. m(N/2-1)" line and seven "E act zp min max mx shx mout shout zout" lines after L (and Q).
// Requantizing builds (int8): a "Q zp min max mx shx mout shout zout" line after it; after the rows, per column block N biases, N multipliers, N shifts (hex).
// int8 residual: an "A zp2 m1 m2 shift zpo amin amax" line after Q (zp1 is Q's zp).
// int8 +dump=<file>: per stage "X k e code" (activation input), "R k e code" (hold beat), "Q k e q1 q2 y" (ADD); k the set in issue order, e its tile element.
// +out_slots=S (1..64) L9 slots per lane, +out_stall_pct=P and +in_stall_pct=P random stalls; -DTB_OUT_STALL_PCT / -DTB_IN_STALL_PCT set the defaults.
// FAULT 1: an A row put with no credit; 2: a W row put with no credit; 3: the A link one bit wide too many; 4: a put on the unused set contract's L0 (a_idle_contract).
// +mid_reset=C: a first pass is reset C cycles into its stream (once the A producer holds credits); every count must come home with no stray output, then the clean pass runs.
// int8 +hold_wait: the reset waits, from C, for a cycle at which hold_mem holds unread hold beats; +hold_check: a reset outside that window fails.
// -DTB_STALE_HOME keeps the A counter's count through that reset, to show the home check firing.
// +tier=T: the layer's lane tier (cfg_tier_i); +coef=<file>: "aaa dddddddd" (hex address, data) lines written through the lanes' load path
// after every reset, before the configure, each once coef_ready_o is high.
// +between=<file> (with +out_first=<file>): the layer runs to completion, its results go to out_first, then <file> is loaded with no reset
// and the layer runs again into +out. +coef_busy: four corrupting writes while the layer streams; the interlock must refuse every one.
module TB_model_run #(
    parameter int FAULT       = 0,
    parameter int LINK_STAGES = 0  // sienna_top's register stages on L0, L1, L3 and L9
);
  // The build's format, from the define through sienna_fmt_pkg; EXP_W, MAN_W and IS_INT only decode words; protocol keys on the package's HAS_ flags.
  localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH, ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
  localparam int RESULT_DATA_WIDTH = sienna_fmt_pkg::RESULT_DATA_WIDTH;
  localparam int EXP_W = sienna_fmt_pkg::EXPONENT_WIDTH, MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH, IS_INT = (sienna_fmt_pkg::EXPONENT_WIDTH == 0);
  localparam int PACK_ENTRIES = 8, WC_TILES = 128;
  `include "sienna_set_side.svh"
  `include "tb_side_pack.svh"

  localparam int STALL_CYCLES = 50_000;  // this long with nothing moving is a hang
  localparam int RCW = $clog2(N + 1);  // a set's N row credits in one cycle
  localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX

  logic clk_i = 0, rstn_i = 0;
  always #5 clk_i = ~clk_i;
  logic por_n = 0;  // the power-on reset only: TB_STALE_HOME's A counter ignores the mid-stream reset

  logic cfg_load_i = 0, cfg_residual_i = 0, cfg_bias_i = 0, cfg_train_i = 0;
  logic [15:0] cfg_m_i = '0, cfg_kb_i = '0, cfg_n_i = '0;
  logic [CONTROL_WIDTH-1:0] cfg_act_i = '0;
  logic [1:0] cfg_tier_i = '0;
  logic coef_wr_i = 1'b0;
  logic [8:0] coef_addr_i = '0;
  logic [31:0] coef_data_i = '0;
  logic coef_ready_o;  // sienna_top's load interlock
  bit restream = 0;  // +between: the second run streams and collects from the start again, with no reset
  logic [LFSR_WIDTH-1:0] cfg_seed_i = '0;
  logic busy_o, done_o, set_done_o;
  logic [7:0] cfg_req_zp_i = '0, cfg_req_min_i = '0, cfg_req_max_i = '0, cfg_gp_zout_i = '0;
  logic [5:0] cfg_gp_shout_i = '0;
  logic [15:0] cfg_gp_mx_i = '0;
  logic [4:0] cfg_gp_shx_i = '0;
  logic [23:0] cfg_gp_mout_i = '0;
  logic [7:0] cfg_add_zp2_i = '0, cfg_add_zpo_i = '0, cfg_add_amin_i = '0, cfg_add_amax_i = '0;  // int8 residual: the A line
  logic [31:0] cfg_add_m1_i = '0, cfg_add_m2_i = '0;
  logic [5:0] cfg_add_shift_i = '0;
  logic [N-1:0][ACC_DATA_WIDTH-1:0] w_bias_i = '0;  // int8: with each block's bias row
  logic [N-1:0][23:0] w_req_mult_i = '0;
  logic [N-1:0][5:0] w_req_shift_i = '0;
  logic [2:0] cfg_pack_shift_i = '0;  // packing: the P and E lines
  logic [N/2-1:0][2:0] cfg_pack_map_i = '0;
  logic [7:1][CONTROL_WIDTH-1:0] cfg_pack_act_i = '0;
  logic [7:1][7:0] cfg_pack_zp_i = '0, cfg_pack_min_i = '0, cfg_pack_max_i = '0, cfg_pack_zout_i = '0;
  logic [7:1][5:0] cfg_pack_shout_i = '0;
  logic [7:1][15:0] cfg_pack_mx_i = '0;
  logic [7:1][4:0] cfg_pack_shx_i = '0;
  logic [7:1][23:0] cfg_pack_mout_i = '0;
  logic [31:0] ep[$];  // int8 epilogues, 3N words per column block

  // ── Row links (L10): this side is the producer of both streams, a counter and a checker on each ──
  credit_link_if #(.DATA_W(N * DATA_WIDTH + (FAULT == 3 ? 1 : 0)), .CRW(RCW)) a_lnk ();
  credit_link_if #(.DATA_W(N * DATA_WIDTH), .CRW(RCW)) w_lnk ();
  logic [$clog2(2*N+1)-1:0] a_cnt, w_cnt;
`ifdef TB_STALE_HOME
  credit_counter #(.MAX(2 * N), .CRW(RCW)) a_cc (.clk_i(clk_i), .rstn_i(por_n), .put_i(a_lnk.put), .credit_i(a_lnk.credit),
                                                .has_credit_o(), .count_o(a_cnt));
`else
  credit_counter #(.MAX(2 * N), .CRW(RCW)) a_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(a_lnk.put), .credit_i(a_lnk.credit),
                                                .has_credit_o(), .count_o(a_cnt));
`endif
  credit_counter #(.MAX(2 * N), .CRW(RCW)) w_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(w_lnk.put), .credit_i(w_lnk.credit),
                                                .has_credit_o(), .count_o(w_cnt));
  // The layer grants per set, bias row or tile, so nothing is outstanding when idle: a_all_back is not armed; the end checks the counts are 0.
  credit_link_checker #(.SLOTS(N)) a_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(1'b0), .lnk(a_lnk));
  credit_link_checker #(.SLOTS(2 * N)) w_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(1'b0), .lnk(w_lnk));
  initial begin
    a_lnk.put = 1'b0;
    a_lnk.data = '0;
    w_lnk.put = 1'b0;
    w_lnk.data = '0;
  end

  // ── Output links (L9): one consumer per lane ──
  credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) out_lnk[NUM_LANES] ();
`ifdef TB_OUT_STALL_PCT
  int out_stall_pct = `TB_OUT_STALL_PCT;
`else
  int out_stall_pct = 0;
`endif
`ifdef TB_IN_STALL_PCT
  int in_stall_pct = `TB_IN_STALL_PCT;
`else
  int in_stall_pct = 0;
`endif
  int out_slots = OUT_CAP;
  logic drain_chk = 1'b0;
  logic [NUM_LANES-1:0] res_v, lane_home;
  logic [DATA_WIDTH-1:0] res_d[NUM_LANES];
  for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
    tb_l9_sink #(.DATA_W(DATA_WIDTH), .MAX_SLOTS(OUT_CAP)) snk (.clk_i(clk_i), .rstn_i(rstn_i), .slots_i(out_slots),
                                                                .stall_pct_i(out_stall_pct), .drain_i(drain_chk),
                                                                .lnk(out_lnk[l]), .valid_o(res_v[l]), .data_o(res_d[l]),
                                                                .home_o(lane_home[l]));
  end

  // The set contract, unused at SET_PORT = 0: its L0 and cache regions are idle but for FAULT 4's one put.
  credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) idle_host[1] ();
  credit_link_if #(.DATA_W(1), .CRW(1)) idle_wc[2] ();
  logic idle_put = 1'b0;
  assign idle_host[0].put  = idle_put;
  assign idle_host[0].data = '0;
  for (genvar r = 0; r < 2; r++) begin : G_IDLE_WC
    assign idle_wc[r].put  = 1'b0;
    assign idle_wc[r].data = 1'b0;
  end
  if (FAULT == 4) begin : G_FAULT_IDLE  // one put on the unused L0 out of reset: sienna_top's a_idle_contract must fire
    initial begin
      wait (rstn_i);
      repeat (4) @(negedge clk_i);
      idle_put = 1'b1;
      $display("[FAULT 4] a put on the unused set contract's L0");
      @(negedge clk_i) idle_put = 1'b0;
    end
  end
  // The int8 words in the package's bundles (zeros in floats); cfg_add_i's in1_zero_point is ignored for the requantize's.
  logic [N-1:0][TB_SCALE_W-1:0] w_req_raw;
  logic [PACK_ENTRIES-1:1][TB_ENTRY_W-1:0] pack_raw;
  for (genvar c = 0; c < N; c++) begin : G_SCALE
    assign w_req_raw[c] = {w_req_mult_i[c], w_req_shift_i[c]};
  end
  for (genvar e = 1; e < PACK_ENTRIES; e++) begin : G_PACK_ENTRY
    assign pack_raw[e] = tb_entry(cfg_pack_zp_i[e], cfg_pack_min_i[e], cfg_pack_max_i[e], cfg_pack_mx_i[e], cfg_pack_shx_i[e], cfg_pack_mout_i[e],
                                  cfg_pack_shout_i[e], cfg_pack_zout_i[e]);
  end
  tb_add_words_t cfg_add_raw;
  assign cfg_add_raw = '{zp1: cfg_req_zp_i, zp2: cfg_add_zp2_i, m1: cfg_add_m1_i, m2: cfg_add_m2_i, shift: cfg_add_shift_i, zpo: cfg_add_zpo_i,
                         amin: cfg_add_amin_i, amax: cfg_add_amax_i};

  sienna_top #(
      .SET_PORT      (0),  // the layer contract: configuration and rows on L10
      .NUM_LANES     (NUM_LANES),
      .N             (N),
      .TILE_SIZE     (TILE_SIZE),
      .COLLAPSE_K    (COLLAPSE_K),  // the package's mesh mode, a build item as in TB_sienna_top
      .SETS_IN_FLIGHT(SETS_IN_FLIGHT),
      .CONTROL_WIDTH (CONTROL_WIDTH),
      .LFSR_WIDTH    (LFSR_WIDTH),
      .LINK_STAGES   (LINK_STAGES),
      .OUT_MAX       (OUT_CAP),
      .OUT_CRW       (1)
  ) dut (
      .clk_i           (clk_i),
      .rstn_i          (rstn_i),
      .cfg_load_i      (cfg_load_i),
      .cfg_m_i         (cfg_m_i),
      .cfg_kb_i        (cfg_kb_i),
      .cfg_n_i         (cfg_n_i),
      .cfg_residual_i  (cfg_residual_i),
      .cfg_bias_i      (cfg_bias_i),
      .cfg_act_i       (cfg_act_i),
      .cfg_tier_i      (cfg_tier_i),
      .cfg_train_i     (cfg_train_i),
      .cfg_seed_i      (cfg_seed_i),
      .cfg_lane_i      (tb_lane_entry(tb_entry(cfg_req_zp_i, cfg_req_min_i, cfg_req_max_i, cfg_gp_mx_i, cfg_gp_shx_i, cfg_gp_mout_i,
                                               cfg_gp_shout_i, cfg_gp_zout_i))),
      .cfg_add_i       (tb_side_add(cfg_add_raw)),
      .cfg_pack_shift_i(cfg_pack_shift_i),
      .cfg_pack_map_i  (cfg_pack_map_i),
      .cfg_pack_act_i  (cfg_pack_act_i),
      .cfg_pack_lane_i (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? ((PACK_ENTRIES - 1) * ENTRY_BITS)'(pack_raw) : '0),
      .busy_o          (busy_o),
      .done_o          (done_o),
      .a_rows          (a_lnk),
      .w_rows          (w_lnk),
      .w_bias_i        (w_bias_i),
      .w_req_scale_i   (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? (N * SCALE_BITS)'(w_req_raw) : '0),
      .set_done_o      (set_done_o),
      // The set contract, unused at SET_PORT = 0: tied off
      .host                (idle_host),
      .bias_i              ('0),
      .wc_region           (idle_wc),
      .wc_write_enable_i   (1'b0),
      .wc_write_addr_i     ('0),
      .north_write_enable_i(1'b0),
      .north_write_data_i  ('0),
      .north_write_reset_i (1'b0),
      .west_write_enable_i (1'b0),
      .west_write_data_i   ('0),
      .west_write_reset_i  (1'b0),
      .drained_i           (1'b0),
      .copy_sel_o          (),
      .pipeline_complete_o (),
      .done_set_id_o       (),
      .systolic_busy_o     (),
      .gpnae_busy_o        (),
      .maxpool_busy_o      (),
      .dropout_busy_o      (),
      .coef_wr_i           (coef_wr_i),
      .coef_addr_i         (coef_addr_i),
      .coef_data_i         (coef_data_i),
      .coef_ready_o        (coef_ready_o),
      .out                 (out_lnk)
  );

  longint cycle = 0;
  always_ff @(posedge clk_i) cycle <= cycle + 1;
  longint t0 = 0;  // the pass's configure: +mid_reset and first_hold count from it
  bit streaming = 0;
  bit meas_clr = 0;  // set over each pass's configure edge: the measurements below restart with the pass
  longint first_hold = -1;  // cycles from t0 to the first hold set's last beat into hold_mem
  always @(posedge clk_i)
    if (meas_clr) first_hold <= -1;
    else if (sienna_fmt_pkg::HAS_RESIDUAL_ADD && streaming && dut.G_LAYER.pipe.hold_done && first_hold < 0) first_hold <= cycle - t0;
  longint first_free = -1;  // cycles from t0 to the first cycle after first_hold with hold_mem read out: the first hold window has closed
  always @(posedge clk_i)
    if (meas_clr) first_free <= -1;
    else if (sienna_fmt_pkg::HAS_RESIDUAL_ADD && streaming && first_hold >= 0 && first_free < 0 && !dut.G_LAYER.pipe.hm_any) first_free <= cycle - t0;
  int hb_wr = 0, hb_rd = 0;  // this pass's hold beats written into and read from hold_mem
  always @(posedge clk_i)
    if (meas_clr) {hb_wr, hb_rd} <= '0;
    else if (streaming) {hb_wr, hb_rd} <= {hb_wr + int'(dut.G_LAYER.pipe.hold_wr_s), hb_rd + int'(dut.G_LAYER.pipe.add_in_s)};

  // int8 per-stage dump: each set's number in issue order by its id, beats in order from the mesh, requantize and add_lanes.
  localparam int PER_LANE = N * N / NUM_LANES;
  localparam int TR_IDS = 1 << $clog2(SETS_IN_FLIGHT + 1);  // sienna_top's set ids
  integer dump_fd = 0;
  function automatic int tr_elem(input int k, input int i, input bit pk);  // sienna_top's elem(): lane k's element i
    return pk ? ((k / N) * PER_LANE + i) * N + (k % N) : k * PER_LANE + i;
  endfunction
  // The dump's inputs, taken at the rising edge as the DUT's flops take them, so its order against the edge-driven producer cannot matter.
  localparam int SDW = $clog2(TR_IDS);
  logic sd_rst = 1'b1, sd_hacc = 1'b0, sd_gacc = 1'b0, sd_rd = 1'b0, sd_ghold = 1'b0, sd_gpack = 1'b0, sd_hwr = 1'b0, sd_fv = 1'b0;
  logic sd_fadd = 1'b0, sd_ain = 1'b0;
  logic [SDW-1:0] sd_hid = '0, sd_gnid = '0, sd_gsid = '0;
  logic [NUM_LANES-1:0][7:0] sd_hd = '0, sd_fd = '0, sd_q1 = '0, sd_q2 = '0;
  always_ff @(posedge clk_i) begin
    sd_rst <= !rstn_i || restream;  // +between's second run numbers its sets from 0 again
    {sd_hacc, sd_gacc, sd_rd, sd_ghold, sd_gpack, sd_hwr, sd_fv, sd_fadd, sd_ain} <=
        {dut.G_LAYER.pipe.host_accept, dut.G_LAYER.pipe.g_accept, dut.G_LAYER.pipe.wide_rd_valid, dut.G_LAYER.pipe.g_hold, dut.G_LAYER.pipe.g_pack, dut.G_LAYER.pipe.hold_wr_s,
         dut.G_LAYER.pipe.fill_v, dut.G_LAYER.pipe.fill_add, dut.G_LAYER.pipe.add_in_s};
    {sd_hid, sd_gnid, sd_gsid} <= {dut.G_LAYER.pipe.host_next_id, dut.G_LAYER.pipe.g_next_id, dut.G_LAYER.pipe.g_set_id};
    for (int k = 0; k < NUM_LANES; k++) sd_fd[k] <= dut.G_LAYER.pipe.fill_d[k][7:0];
    {sd_hd, sd_q1, sd_q2} <= {dut.G_LAYER.pipe.hold_d_s, dut.G_LAYER.pipe.add_q1_s, dut.G_LAYER.pipe.add_q2_s};
  end
  initial begin : stage_dump  // reads the rising edge's copies at the falling edge, one cycle behind the DUT, in order
    int tr_k[TR_IDS];  // per set id: the set's number in issue order
    int beat[TR_IDS];  // beats of each set id read from the mesh so far
    int put_seq;
    logic [$clog2(TR_IDS)-1:0] q_id[$], h_id[$];  // beats inside requantize: their set id and beat index
    int q_i[$], h_i[$];
    bit q_pk[$];
    logic [NUM_LANES-1:0][7:0] a_q1[$], a_q2[$];  // add beats inside add_lanes, their two inputs
    put_seq = 0;
    forever begin
      @(negedge clk_i);
      if (sd_rst) begin  // the cut pass's beats are dropped; its lines stay before the next H line
        put_seq = 0;
        q_id.delete();
        q_i.delete();
        q_pk.delete();
        h_id.delete();
        h_i.delete();
        a_q1.delete();
        a_q2.delete();
      end else if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE && dump_fd != 0) begin
        if (sd_hacc) tr_k[int'(sd_hid)] = put_seq++;
        if (sd_gacc) beat[int'(sd_gnid)] = 0;
        if (sd_rd) begin  // a beat leaves the mesh into requantize
          if (!sd_ghold) begin
            q_id.push_back(sd_gsid);
            q_i.push_back(beat[int'(sd_gsid)]);
            q_pk.push_back(sd_gpack);
          end else begin
            h_id.push_back(sd_gsid);
            h_i.push_back(beat[int'(sd_gsid)]);
          end
          beat[int'(sd_gsid)]++;
        end
        if (sd_hwr) begin  // a hold beat into hold_mem: the conv's requantized codes
          if (h_id.size() == 0) $display("[FAIL] stage dump: a hold_mem write with no hold beat in the requantize pipeline");
          else begin
            automatic logic [$clog2(TR_IDS)-1:0] id = h_id.pop_front();
            automatic int i = h_i.pop_front();
            for (int k = 0; k < NUM_LANES; k++) $fdisplay(dump_fd, "R %0d %0d %0d", tr_k[id], tr_elem(k, i, 1'b0), $signed(sd_hd[k]));
          end
        end
        if (sd_fv) begin  // a beat into the activation stage, from requantize or add_lanes
          if (q_id.size() == 0) $display("[FAIL] stage dump: a fill beat no requantize or add_lanes beat accounts for");
          else begin
            automatic logic [$clog2(TR_IDS)-1:0] id = q_id.pop_front();
            automatic int i = q_i.pop_front();
            automatic bit pk = q_pk.pop_front();
            if (sd_fadd && a_q1.size() == 0) $display("[FAIL] stage dump: an add_lanes fill with no add beat before it");
            else if (sd_fadd) begin
              automatic logic [NUM_LANES-1:0][7:0] q1 = a_q1.pop_front(), q2 = a_q2.pop_front();
              for (int k = 0; k < NUM_LANES; k++)
                $fdisplay(dump_fd, "Q %0d %0d %0d %0d %0d", tr_k[id], tr_elem(k, i, pk), $signed(q1[k]), $signed(q2[k]),
                          $signed(sd_fd[k]));
            end
            for (int k = 0; k < NUM_LANES; k++) $fdisplay(dump_fd, "X %0d %0d %0d", tr_k[id], tr_elem(k, i, pk), $signed(sd_fd[k]));
          end
        end
        if (sd_ain) begin  // an add beat enters add_lanes with hold_mem's beat (q1) and requantize's (q2)
          a_q1.push_back(sd_q1);
          a_q2.push_back(sd_q2);
        end
      end
    end
  end

  // Results per set in window order: lane L's j-th word is window j*NUM_LANES + L, whatever each lane's stalls; empty for partial sums.
  logic [DATA_WIDTH-1:0] lane_q[NUM_LANES][$];
  logic [DATA_WIDTH-1:0] res_q[$];
  int bounds[$];
  longint t_done = 0;
  always_ff @(posedge clk_i) begin
    if (!rstn_i || restream) begin  // a mid-stream reset drops the cut pass's results; +between's first run was written out
      for (int l = 0; l < NUM_LANES; l++) lane_q[l].delete();
      res_q.delete();
      bounds.delete();
      t_done <= 0;
    end else begin
      for (int l = 0; l < NUM_LANES; l++) if (res_v[l]) lane_q[l].push_back(res_d[l]);
      if (set_done_o) begin
        automatic bit any;
        any = 1;
        while (any) begin
          any = 0;
          for (int l = 0; l < NUM_LANES; l++)
            if (lane_q[l].size() != 0) begin
              res_q.push_back(lane_q[l].pop_front());
              any = 1;
            end
        end
        bounds.push_back(res_q.size());
      end
      if (done_o) t_done <= cycle;
    end
  end

  // int8: the epilogue beside the next bias row; the layer says which W row it took as a block's bias.
  int ei = 0;
  always_ff @(posedge clk_i)
    if (!rstn_i || restream) ei <= 0;
    else if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE && dut.G_LAYER.sched.wl_take_bias) ei <= ei + 1;

  // Output of any kind while the mid-stream reset's home check watches is stray.
  bit mid_check = 0;
  int stray = 0;
  always @(posedge clk_i) if (mid_check && rstn_i && (res_v != '0 || set_done_o || done_o)) stray++;

`ifdef PERF
  // Cycles with sets left but no A row taken, by cause, printed at the end.
  int st_credit = 0, st_staging = 0, st_bias = 0, st_tile = 0, st_input = 0, st_rows = 0, st_idle = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i && dut.G_LAYER.sched.active && !a_lnk.put && dut.G_LAYER.sched.g_blk < dut.G_LAYER.sched.ct) begin
      if (dut.G_LAYER.sched.pend) st_input++;  // the set's rows are granted and not all sent
      else if (!dut.G_LAYER.sched.g_l0_ok && dut.G_LAYER.pipe.entry_full) st_credit++;  // the entry withholds the staging credit
      else if (!dut.G_LAYER.sched.g_l0_ok) st_staging++;
      else if (!dut.G_LAYER.sched.g_bias_ok) st_bias++;
      else if (!dut.G_LAYER.sched.g_tile_ok) st_tile++;
      else st_idle++;
    end
    if (rstn_i && a_lnk.put) st_rows++;
  end
  // Downstream: cycles a stage holds a ready set but cannot pass it on, and why.
  int dn_accbank = 0, dn_resbank = 0, dn_actbusy = 0, dn_actbank = 0, dn_poolbusy = 0, dn_bcast = 0;
  always_ff @(posedge clk_i) begin
    if (rstn_i && dut.G_LAYER.sched.active) begin
      if (dut.G_LAYER.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ob_full[dut.G_LAYER.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.fb] &&
          dut.G_LAYER.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ab_busy[dut.G_LAYER.pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.ab_next])
        dn_accbank++;
      if (dut.G_LAYER.pipe.systolic_array_inst.in_full[dut.G_LAYER.pipe.systolic_array_inst.in_rd] && !dut.G_LAYER.pipe.systolic_array_inst.arrays_load_ready &&
          dut.G_LAYER.pipe.systolic_array_inst.bstate == 0)
        dn_bcast++;
      if (dut.G_LAYER.pipe.systolic_array_inst.arrays_final && !dut.G_LAYER.pipe.systolic_array_inst.reduce_start &&
          dut.G_LAYER.pipe.systolic_array_inst.out_state[dut.G_LAYER.pipe.systolic_array_inst.out_wr] != 0)
        dn_resbank++;
      if (dut.G_LAYER.pipe.systolic_array_inst.out_full[dut.G_LAYER.pipe.systolic_array_inst.out_rd] && dut.G_LAYER.pipe.g_fed) dn_actbusy++;
      if (dut.G_LAYER.pipe.systolic_array_inst.out_full[dut.G_LAYER.pipe.systolic_array_inst.out_rd] && dut.G_LAYER.pipe.g_state == 0 &&
          dut.G_LAYER.pipe.act_full[dut.G_LAYER.pipe.act_wr]) dn_actbank++;
      if (dut.G_LAYER.pipe.act_full[dut.G_LAYER.pipe.act_rd] && dut.G_LAYER.pipe.p_state != 0) dn_poolbusy++;
    end
  end
`endif

  logic [DATA_WIDTH-1:0] a_rows[$], w_rows[$];  // N words per row, back to back
  int ai = 0, wi = 0, na = 0, nw = 0, ne = 0;
  bit faulted = 0;

  // The producers: a row goes on the falling edge while a credit is held and the random input stall allows it.
  always @(negedge clk_i) begin
    a_lnk.put = 1'b0;
    w_lnk.put = 1'b0;
    if (!rstn_i || restream) begin  // the clean pass after a mid-stream reset (or +between's second run) streams from the first row
      ai = 0;
      wi = 0;
    end
    if (streaming) begin
      if (FAULT == 1 && !faulted) begin
        $display("  [FAULT 1] an A row put with %0d A credits held", a_cnt);
        a_lnk.put = 1'b1;
        faulted = 1;
      end else if (FAULT == 2 && !faulted) begin
        $display("  [FAULT 2] a W row put with %0d W credits held", w_cnt);
        w_lnk.put = 1'b1;
        faulted = 1;
      end
      if (ai < na && a_cnt != 0 && !a_lnk.put && !(in_stall_pct > 0 && $urandom_range(99) < in_stall_pct)) begin
        for (int c = 0; c < N; c++) a_lnk.data[c*DATA_WIDTH +: DATA_WIDTH] = a_rows[ai*N+c];
        a_lnk.put = 1'b1;
        ai++;
      end
      if (wi < nw && w_cnt != 0 && !w_lnk.put && !(in_stall_pct > 0 && $urandom_range(99) < in_stall_pct)) begin
        for (int c = 0; c < N; c++) w_lnk.data[c*DATA_WIDTH +: DATA_WIDTH] = w_rows[wi*N+c];
        w_lnk.put = 1'b1;
        wi++;
      end
      for (int c = 0; c < N; c++) begin
        w_bias_i[c]      = (ei < ne) ? ACC_DATA_WIDTH'(ep[(3*ei)*N+c]) : '0;
        if (ei < ne && (ep[(3*ei+1)*N+c][31:24] != 0 || ep[(3*ei+2)*N+c][31:6] != 0))
          $fatal(1, "TB_model_run: channel %0d's requantize words %h %h are too wide", c, ep[(3*ei+1)*N+c], ep[(3*ei+2)*N+c]);
        w_req_mult_i[c]  = (ei < ne) ? ep[(3*ei+1)*N+c][23:0] : '0;
        w_req_shift_i[c] = (ei < ne) ? ep[(3*ei+2)*N+c][5:0] : '0;
      end
    end
  end

  // +coef=<file>: one load-path write per cycle once coef_ready_o, each driven 1 ns after an edge as every other input here.
  task automatic coef_load();
    string cf;
    if (!$value$plusargs("coef=%s", cf)) return;
    coef_load_file(cf);
  endtask
  task automatic coef_load_file(input string cf);
    integer fc, nw;
    logic [11:0] a;
    logic [31:0] d;
    fc = $fopen(cf, "r");
    if (fc == 0) $fatal(1, "TB_model_run: cannot open %s", cf);
    nw = 0;
    while ($fscanf(fc, "%h %h", a, d) == 2) begin
      #1;
      while (!coef_ready_o) begin
        coef_wr_i = 1'b0;
        @(posedge clk_i);
        #1;
      end
      coef_wr_i   = 1'b1;
      coef_addr_i = a[8:0];
      coef_data_i = d;
      @(posedge clk_i);
      nw++;
    end
    #1 coef_wr_i = 1'b0;
    $fclose(fc);
    $display("[LAYER] coef: %0d load-path writes from %s", nw, cf);
    @(posedge clk_i);
  endtask

  initial begin
    string layer_f, out_f, kind;
    integer fin, fout, rc;
    int m, kb, n, res, bias, act, train, seed, pk, idle, last_ai, last_wi, last_sets;
    int qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout;
    int azp2, ash, azpo, aamin, aamax;
    longint am1, am2;  // unsigned 32-bit mantissas
    string dump_f;
    logic [DATA_WIDTH-1:0] v;
    logic [31:0] w32;
    bit finished;
    int failed = 0, osp, mid_reset = 0, msp, tier = 0;
    string between_f, first_f;
    int between = 0, busy_load = 0, busy_tries = 0, busy_refused = 0;
    int hold_wait = 0, hold_check = 0;  // +hold_wait: the mid-stream reset waits for the hold window; +hold_check: it must land in it

    void'($value$plusargs("out_slots=%d", out_slots));
    void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
    void'($value$plusargs("in_stall_pct=%d", in_stall_pct));
    void'($value$plusargs("mid_reset=%d", mid_reset));
    void'($value$plusargs("tier=%d", tier));
    between = $value$plusargs("between=%s", between_f);
    if (between && (!$value$plusargs("out_first=%s", first_f) || mid_reset > 0)) $fatal(1, "TB_model_run: +between needs +out_first and no +mid_reset");
    busy_load = $test$plusargs("coef_busy");
    if (tier < 0 || tier > 2) $fatal(1, "TB_model_run: +tier=%0d: 0, 1 or 2", tier);
    hold_wait = $test$plusargs("hold_wait");
    hold_check = $test$plusargs("hold_check");
    if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90 || in_stall_pct < 0 || in_stall_pct > 90) begin
      $display("[FATAL] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d, +in_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct,
               in_stall_pct);
      $finish;
    end
    if (!$value$plusargs("layer=%s", layer_f) || !$value$plusargs("out=%s", out_f)) begin
      $display("[LAYER] no +layer= and +out= given, nothing to run (%0d L9 slots per lane, %0d%% output and %0d%% input stalls)",
               out_slots, out_stall_pct, in_stall_pct);
      $finish;
    end
    fin = $fopen(layer_f, "r");
    if (fin == 0) begin
      $display("[FATAL] cannot open %s", layer_f);
      $finish;
    end
    rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d %d %d", kind, m, kb, n, res, bias, act, train, seed, na, nw, pk);
    if (rc != 12 || kind != "L") begin
      $display("[FATAL] %s is not a layer file", layer_f);
      $finish;
    end
    {qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout} = '0;  // fp32 and bf16 keep every int8 input at 0
    if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE) begin
      rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d", kind, qzp, qmin, qmax, qmx, qshx, qmout, qshout, qzout);
      if (rc != 9 || kind != "Q") begin
        $display("[FATAL] %s: an int8 layer file needs its Q line", layer_f);
        $finish;
      end
    end
    {azp2, am1, am2, ash, azpo, aamin, aamax} = '0;
    if (sienna_fmt_pkg::HAS_RESIDUAL_ADD && res != 0) begin
      rc = $fscanf(fin, "%s %d %d %d %d %d %d %d", kind, azp2, am1, am2, ash, azpo, aamin, aamax);
      if (rc != 8 || kind != "A") begin
        $display("[FATAL] %s: an int8 residual layer file needs its A line", layer_f);
        $finish;
      end
      if (ash < 0 || ash >= 64 || azp2 < -128 || azp2 > 127 || azpo < -128 || azpo > 127 || aamin < -128 || aamin > 127 || aamax < -128 || aamax > 127)
        $fatal(1, "TB_model_run: the A line's words %0d %0d %0d %0d %0d are out of range", azp2, ash, azpo, aamin, aamax);
    end
    if (pk) begin
      int sh, mv, ev[9];
      rc = $fscanf(fin, "%s %d", kind, sh);
      if (rc != 2 || kind != "P") begin
        $display("[FATAL] %s: a packed layer needs its P line", layer_f);
        $finish;
      end
      cfg_pack_shift_i = 3'(sh);
      for (int c = 0; c < N / 2; c++) begin
        rc = $fscanf(fin, "%d", mv);
        cfg_pack_map_i[c] = 3'(mv);
      end
      for (int e = 1; e < 8; e++) begin
        rc = $fscanf(fin, "%s %d %d %d %d %d %d %d %d %d", kind, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8]);
        if (rc != 10 || kind != "E") begin
          $display("[FATAL] %s: entry %0d needs its E line", layer_f, e);
          $finish;
        end
        cfg_pack_act_i[e] = CONTROL_WIDTH'(ev[0]);
        cfg_pack_zp_i[e] = 8'(ev[1]);
        cfg_pack_min_i[e] = 8'(ev[2]);
        cfg_pack_max_i[e] = 8'(ev[3]);
        cfg_pack_mx_i[e] = 16'(ev[4]);
        cfg_pack_shx_i[e] = 5'(ev[5]);
        if (ev[6] >= (1 << 24) || ev[6] < 0 || ev[7] >= 64 || ev[7] < 0) $fatal(1, "TB_model_run: pack entry %0d's SELU words %0d %0d are out of range", e, ev[6], ev[7]);
        cfg_pack_mout_i[e] = 24'(ev[6]);
        cfg_pack_shout_i[e] = 6'(ev[7]);
        cfg_pack_zout_i[e] = 8'(ev[8]);
      end
    end
    for (int i = 0; i < (na + nw) * N; i++) begin
      rc = $fscanf(fin, "%h", v);
      if (i < na * N) a_rows.push_back(v);
      else w_rows.push_back(v);
    end
    ne = sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? (n + N - 1) / N : 0;  // one epilogue per column block
    for (int i = 0; i < ne * 3 * N; i++) begin
      rc = $fscanf(fin, "%h", w32);
      ep.push_back(w32);
    end
    $fclose(fin);
    if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE && $value$plusargs("dump=%s", dump_f)) begin
      dump_fd = $fopen(dump_f, "w");
      if (dump_fd == 0) $fatal(1, "TB_model_run: cannot open %s", dump_f);
    end

    repeat (5) @(posedge clk_i);
    #1 rstn_i = 1;
    por_n = 1;
    repeat (3) @(posedge clk_i);

    for (int pass = (mid_reset > 0 || between) ? 0 : 1; pass < 2; pass++) begin
      if (!(between && pass == 1)) coef_load();  // the lanes' coefficient memory, reset to its default with the rest (+between: loaded above)
      // Configure: the only control the software gives.
      #1;
      cfg_tier_i = 2'(tier);
      cfg_m_i = 16'(m);
      cfg_kb_i = 16'(kb);
      cfg_n_i = 16'(n);
      cfg_residual_i = res[0];
      cfg_bias_i = bias[0];
      cfg_act_i = CONTROL_WIDTH'(act);
      cfg_train_i = train[0];
      cfg_seed_i = LFSR_WIDTH'(seed);
      cfg_req_zp_i = 8'(qzp);
      cfg_req_min_i = 8'(qmin);
      cfg_req_max_i = 8'(qmax);
      cfg_gp_mx_i = 16'(qmx);
      cfg_gp_shx_i = 5'(qshx);
      if (qmout >= (1 << 24) || qmout < 0 || qshout >= 64 || qshout < 0) $fatal(1, "TB_model_run: SELU words %0d %0d are out of range", qmout, qshout);
      cfg_gp_mout_i = 24'(qmout);
      cfg_gp_shout_i = 6'(qshout);
      cfg_gp_zout_i = 8'(qzout);
      cfg_add_zp2_i = 8'(azp2);
      if (am1 < 0 || am1 >= 64'h1_0000_0000 || am2 < 0 || am2 >= 64'h1_0000_0000) $fatal(1, "TB_model_run: the ADD's mantissas %0d %0d are out of range", am1, am2);
      cfg_add_m1_i = 32'(am1);
      cfg_add_m2_i = 32'(am2);
      cfg_add_shift_i = 6'(ash);
      cfg_add_zpo_i = 8'(azpo);
      cfg_add_amin_i = 8'(aamin);
      cfg_add_amax_i = 8'(aamax);
      if (dump_fd != 0) $fdisplay(dump_fd, "H pass %0d", pass);  // the reader keeps the lines after the last H
      cfg_load_i = 1;
      meas_clr = 1;
      @(posedge clk_i);
      #1 cfg_load_i = 0;
      meas_clr = 0;
      t0 = cycle;
      streaming = 1;

      // Stream both inputs as credits arrive; a hang is this long with no row and no set leaving.
      idle = 0;
      finished = 0;
      last_ai = 0;
      last_wi = 0;
      last_sets = 0;
      while (!finished) begin
        @(posedge clk_i);
        finished = done_o;
        #1;
        idle = (ai != last_ai || wi != last_wi || bounds.size() != last_sets) ? 0 : idle + 1;
        last_ai = ai;
        last_wi = wi;
        last_sets = bounds.size();
        if (pass == 1 && busy_load && busy_tries < 4 && cycle - t0 >= 40) begin  // +coef_busy: words 0, 1 and two descriptor fields
          coef_wr_i   = 1'b1;
          coef_addr_i = (busy_tries < 2) ? 9'(busy_tries) : 9'(128 + busy_tries);
          coef_data_i = 32'h4120_0000;
          if (!coef_ready_o) busy_refused++;
          busy_tries++;
        end else if (busy_load) coef_wr_i = 1'b0;
        if (pass == 0 && mid_reset > 0 && cycle - t0 >= mid_reset && a_cnt != 0 && (hold_wait == 0 || dut.G_LAYER.pipe.hm_any)) break;  // mid-stream, A credits held (+hold_wait: hold_mem waiting)
        if (idle > STALL_CYCLES) begin
          $display("[FATAL] nothing moved for %0d cycles: %0d/%0d activation rows, %0d/%0d weight rows, %0d sets done", idle, ai, na, wi,
                   nw, bounds.size());
          $finish;
        end
      end
      streaming = 0;
      if (pass == 0 && between) begin  // the first run is done: write it out, load the next image with no reset, run again
        repeat (OUT_CAP + 4) @(posedge clk_i);
        fout = $fopen(first_f, "w");
        for (int k = 0; k < bounds.size(); k++) begin
          automatic int lo = (k == 0) ? 0 : bounds[k-1];
          if (bounds[k] == lo) continue;
          $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
          for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
        end
        $fclose(fout);
        $display("  [Between] first run: %0d sets, %0d outputs; loading %s with no reset", bounds.size(), res_q.size(), between_f);
        coef_load_file(between_f);
        #1 restream = 1;
        @(posedge clk_i);
        #1 restream = 0;
        repeat (3) @(posedge clk_i);
        continue;
      end
      if (pass == 0) begin
        if (finished) begin
          $display("[FATAL] the layer finished before +mid_reset=%0d cycles: no mid-stream reset", mid_reset);
          $finish;
        end
        $display("  [Reset] mid-stream, %0d cycles into the layer: %0d/%0d A rows, %0d/%0d W rows, %0d sets done, A %0d and W %0d credits held",
                 cycle - t0, ai, na, wi, nw, bounds.size(), a_cnt, w_cnt);
        if (sienna_fmt_pkg::HAS_RESIDUAL_ADD)
          $display("[RES] reset %0d cycles into the layer, %0s the hold window, %0d of %0d hold beats unread", cycle - t0,
                   dut.G_LAYER.pipe.hm_any ? "inside" : "outside", hb_wr - hb_rd, PER_LANE);
        if (hold_check != 0 && !dut.G_LAYER.pipe.hm_any) begin
          $display("[FAIL] +hold_check: the mid-stream reset at %0d cycles is outside the hold window", cycle - t0);
          $finish;
        end
        rstn_i = 0;
        repeat (5) @(posedge clk_i);
        msp = out_stall_pct;
        out_stall_pct = 0;
        #1 rstn_i = 1;
        mid_check = 1;
        repeat (OUT_CAP + 64 + 8 * LINK_STAGES) @(posedge clk_i);
        mid_check = 0;
        // Home: no row credit held (the layer grants per set), both staging and region credits back, every L9 slot re-advertised.
        if (a_cnt != 0 || w_cnt != 0 || int'(dut.G_LAYER.sched.l0_cnt) != 2 || !dut.G_LAYER.sched.wc_has[0] || !dut.G_LAYER.sched.wc_has[1] || lane_home != '1 || stray != 0 ||
            busy_o) begin
          $display("[FAIL] not home after the mid-stream reset: A %0d (0), W %0d (0), staging %0d (2), regions %0d %0d (1 1), lanes home %0d of %0d, %0d stray output cycles, busy %0d",
                   a_cnt, w_cnt, dut.G_LAYER.sched.l0_cnt, dut.G_LAYER.sched.wc_has[0], dut.G_LAYER.sched.wc_has[1], $countones(lane_home), NUM_LANES, stray, busy_o);
          $finish;
        end else
          $display("  [Reset] home: row credits 0 0, staging 2, regions 1 1, %0d lanes with %0d L9 slots each, no stray output", NUM_LANES,
                   out_slots);
        out_stall_pct = msp;
        repeat (3) @(posedge clk_i);
        continue;
      end
    end
    // The layer granted exactly the rows the streams hold, and every L9 slot comes home once the stalls stop.
    if (a_cnt != 0 || w_cnt != 0) begin
      failed++;
      $display("[FAIL] the layer is done with row credits still held: A %0d, W %0d", a_cnt, w_cnt);
    end
    osp = out_stall_pct;
    out_stall_pct = 0;
    repeat (OUT_CAP + 4) @(posedge clk_i);
    if (lane_home != '1) begin
      failed++;
      $display("[FAIL] L9 slots not all credited back after the layer: lanes home %b", lane_home);
    end
    // sienna_top's own link checkers judge only when it is drained: it must drain after the layer's last set.
    for (int w = 0; w < 200 + 8 * LINK_STAGES && !dut.G_LAYER.pipe.drained; w++) @(posedge clk_i);
    if (!dut.G_LAYER.pipe.drained) begin
      failed++;
      $display("[FAIL] sienna_top never drained after the layer's last set (%0d drains in the run): its link checkers never judged it",
               dut.G_LAYER.pipe.drain_n);
    end else $display("  [Drain] sienna_top drained after the last set, %0d drains in the run", dut.G_LAYER.pipe.drain_n);
    if (dut.G_LAYER.pipe.hm_any) begin  // a hold set left without its add set (Task 2.7 review M5)
      failed++;
      $display("[FAIL] hold_mem still holds unread hold beats after the layer's last set: a hold set without its add set");
    end
    @(negedge clk_i);
    drain_chk = 1'b1;
    @(negedge clk_i);
    drain_chk = 1'b0;

    if (busy_load) begin
      $display("[LAYER] coef busy: %0d writes while the layer streamed, %0d refused", busy_tries, busy_refused);
      if (busy_tries != 4 || busy_refused != busy_tries) begin
        failed++;
        $display("[FAIL] +coef_busy: %0d of %0d writes taken while the layer streamed", busy_tries - busy_refused, busy_tries);
      end
    end
    fout = $fopen(out_f, "w");
    for (int k = 0; k < bounds.size(); k++) begin
      automatic int lo = (k == 0) ? 0 : bounds[k-1];
      if (bounds[k] == lo) continue;  // a partial sum
      $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
      for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
    end
    $fclose(fout);
    if (failed == 0)
      $display("[LAYER] sets=%0d outputs=%0d cycles=%0d a_rows=%0d/%0d w_rows=%0d/%0d epilogues=%0d/%0d out_stall=%0d in_stall=%0d collapse_k=%0d",
               bounds.size(), res_q.size(), t_done - t0 + 1, ai, na, wi, nw, ei, ne, osp, in_stall_pct,
               dut.G_LAYER.pipe.systolic_array_inst.COLLAPSE_K);  // the elaborated mesh's mode, not the package's: run_job checks it
    if (first_hold >= 0) $display("[RES] first hold_done %0d cycles into the layer", first_hold);
    if (first_free >= 0) $display("[RES] first hold window closed %0d cycles into the layer", first_free);
    if (dump_fd != 0) $fclose(dump_fd);
`ifdef PERF
    $display("[PERF] rows=%0d stalls: mid-set=%0d credits=%0d staging=%0d bias=%0d tile=%0d other=%0d", st_rows, st_input,
             st_credit, st_staging, st_bias, st_tile, st_idle);
    $display("[PERF] held: operand-bank-full=%0d partial-sum-bank=%0d result-bank=%0d activation-busy=%0d activation-bank=%0d pooling-busy=%0d",
             dn_bcast, dn_accbank, dn_resbank, dn_actbusy, dn_actbank, dn_poolbusy);
`endif
    $finish;
  end

endmodule
