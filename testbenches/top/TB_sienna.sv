`timescale 1ns / 100ps

import test_config_pkg::*;

`include "tb_l9_sink.svh"

// TB_sienna: SIENNA's one top bench. MODE (set with -GMODE, as the mesh bench takes MATRIX_SIZE) picks the contract, the copies and the run:
//   0 pipeline: the pipeline regression's streamed sets on the set contract (SET_PORT = 1), one copy;
//   1 layer: one layer on the layer contract (SET_PORT = 0), as model_runner's layer engine drives it;
//   2 sets: one network layer's sets on the set contract (SET_PORT = 1), as model_runner's set engine drives it;
//   3 copies: COPIES copies of sienna_top on the set contract (SET_PORT = 1).
// Each mode's branch is the body of the bench it replaces, unchanged; its notes and plusargs are at the branch's head.
// Every mode checks each lane's coefficient memory after every reset and load against the host's image (tb_coef_check.svh; coef_default.mem, <file>.img).
module TB_sienna #(
    parameter int MODE        = 0,
    parameter int LINK_STAGES = 0,  // sienna_top's register stages on L0, L1, L3 and L9 (each copy's in mode 3)
    parameter int FAULT       = 0,  // modes 0 and 1: the planted fault their notes list
    parameter int COPIES      = 2,  // mode 3: sienna_top's copies
    parameter int COLLAPSE_K  = test_config_pkg::COLLAPSE_K  // the mesh mode: the package's unless -GCOLLAPSE_K sets it (mode 3 is run at 1)
);
  if (MODE == 0) begin : g_pipeline
    // LINK_STAGES: register stages on sienna_top's links; FAULT 1: a put with no credit and no rows, 3: a host link one bit wide.
    // FAULT (TB_sienna's parameter): also 6: 8 L5 slots per lane (a_l5_starved); 7: output links one bit wider; 8: a put on the unused A row link (a_idle_contract)
    // FAULT 9: the reset image's word 7 flipped after it is read, so the coefficient-image check must fail
    // The build's format, from the define through sienna_fmt_pkg; EXP_W, MAN_W and IS_INT only decode words; protocol keys on the package's HAS_ flags.
    localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH, ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
    localparam int RESULT_DATA_WIDTH = sienna_fmt_pkg::RESULT_DATA_WIDTH;
    localparam int EXP_W = sienna_fmt_pkg::EXPONENT_WIDTH, MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH, IS_INT = (sienna_fmt_pkg::EXPONENT_WIDTH == 0);

    // ── Localparams from generated SV package ─────────────────────────────
    localparam string WEST_INPUT_FILE = "matrix_west.mem";
    localparam string NORTH_INPUT_FILE = "matrix_north.mem";
    localparam string EXPECTED_OUTPUT_FILE = "expected_output.mem";

    localparam ADDR_LINES = $clog2(FIFO_DEPTH);
    localparam int ID_W = $clog2(SETS_IN_FLIGHT + 1);  // set ids count accepted starts modulo 2^ID_W

    // ── Timeout / heartbeat ───────────────────────────────────────────────
    localparam int TIMEOUT_CYCLES = 200_000;
    localparam int HEARTBEAT_CYCLES = 5_000;

    // ── Tolerance ─────────────────────────────────────────────────────────
    localparam string TOLERANCE_MODE = "RELATIVE";
    localparam real ABS_TOL = 0.001;
    localparam real REL_TOL = 0.01;

    // ── DUT I/O ───────────────────────────────────────────────────────────
    logic clk_i, rstn_i;
    logic                     training_mode_i;
    logic                     accumulate_i;  // this set is a partial sum
    logic                     bias_valid_i;  // this set carries a bias row
    logic [N-1:0][ACC_DATA_WIDTH-1:0] bias_i;  // int32 in int8, where it carries the folded input zero point; fp32 bits in floats
    logic [N-1:0][ACC_DATA_WIDTH-1:0] bias_want;  // the bias the mesh must see with this set's put: the file's row, widened exactly
    int bias_checked = 0;  // puts whose bias reached the mesh's bias queue and was compared
    // int8: the requantize and GPNAE parameters of the set being started (D-2); zero in other formats
    logic [N-1:0][23:0] req_mult_i;
    logic [N-1:0][5:0]  req_shift_i;
    logic [7:0]         req_zp_i, req_min_i, req_max_i, gp_zout_i;
    logic [15:0]        gp_mx_i;
    logic [4:0]         gp_shx_i;
    logic [23:0]        gp_mout_i;
    logic [5:0]         gp_shout_i;
    localparam int PACK_ENTRIES = 8, PEW = 3;
    logic [2:0] pack_shift_i;  // the set's pack shift and parameter table (pack_<k>.mem), with the start
    logic [N/2-1:0][PEW-1:0] pack_map_i;
    logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] pack_act_i;
    logic [PACK_ENTRIES-1:1][7:0] pack_zp_i, pack_min_i, pack_max_i, pack_zout_i;
    logic [PACK_ENTRIES-1:1][15:0] pack_mx_i;
    logic [PACK_ENTRIES-1:1][4:0] pack_shx_i;
    logic [PACK_ENTRIES-1:1][23:0] pack_mout_i;
    logic [PACK_ENTRIES-1:1][5:0] pack_shout_i;
    localparam int WC_TILES = 128;
    logic                     weight_cached_i;  // this set takes B from cache tile weight_tile_i
    logic [$clog2(WC_TILES)-1:0] weight_tile_i;
    logic                     wc_write_enable_i;
    logic [$clog2(WC_TILES*N*N)-1:0] wc_write_addr_i;
    logic                     wc_last_i;  // this set is the last of its cache region's fill
    logic [LFSR_WIDTH-1:0]    dropout_seed_i;
    logic [CONTROL_WIDTH-1:0] activation_function_i;
    logic res_hold_i, res_add_i;  // int8 residual: the set's hold and add flags and its ADD's words (apply_add)
    localparam int PER_LANE = SRAM_DEPTH / NUM_LANES;  // result beats per set

    // The test's TB-only settings, read from test_knobs.mem at time 0 (written with its stimulus): no build depends on them, so one build runs every test of its group.
    int k_activation_code, k_num_sets, k_has_bias, k_weight_cache, k_accum_passes, k_mixed_len, k_mixed_acts, k_packed, k_training_mode, k_dropout_seed;
    int k_residual;  // int8 residual: each set's partial, bias, hold, add and activation come from add_<k>.mem
    int hold_set;  // the hold pass's set: SETS_IN_FLIGHT+4 sets to put after it where k_num_sets allows, else set 0
    task automatic read_knobs();
      int fd, v, seen, i;
      string key;
      fd = $fopen("test_knobs.mem", "r");
      if (fd == 0) $fatal(1, "TB_sienna pipeline: no test_knobs.mem: regression.py writes it with each test's stimulus (make pkg)");
      seen = 0;
      while ($fscanf(fd, "%s %d\n", key, v) == 2) begin
        case (key)
          "activation_code": begin i = 0; k_activation_code = v; end
          "num_sets":        begin i = 1; k_num_sets = v;        end
          "has_bias":        begin i = 2; k_has_bias = v;        end
          "weight_cache":    begin i = 3; k_weight_cache = v;    end
          "accum_passes":    begin i = 4; k_accum_passes = v;    end
          "mixed_len":       begin i = 5; k_mixed_len = v;       end
          "mixed_acts":      begin i = 6; k_mixed_acts = v;      end
          "packed":          begin i = 7; k_packed = v;          end
          "training_mode":   begin i = 8; k_training_mode = v;   end
          "dropout_seed":    begin i = 9; k_dropout_seed = v;    end
          "residual":        begin i = 10; k_residual = v;       end
          default: $fatal(1, "TB_sienna pipeline: test_knobs.mem names an unknown setting '%s'", key);
        endcase
        if (seen[i]) $fatal(1, "TB_sienna pipeline: test_knobs.mem sets '%s' twice", key);
        seen[i] = 1'b1;
      end
      $fclose(fd);
      if (seen != 11'h7FF) $fatal(1, "TB_sienna pipeline: test_knobs.mem lacks settings (mask %b of 11)", seen[10:0]);
      if (k_num_sets < 1 || k_accum_passes < 1) $fatal(1, "TB_sienna pipeline: test_knobs.mem: num_sets %0d, accum_passes %0d", k_num_sets, k_accum_passes);
      hold_set = (k_num_sets / 2 < k_num_sets - SETS_IN_FLIGHT - 4) ? k_num_sets / 2
               : ((k_num_sets - SETS_IN_FLIGHT - 4 > 0) ? k_num_sets - SETS_IN_FLIGHT - 4 : 0);
    endtask

    // ── Links: this TB is the host (L0) and the weight-cache writer (L2), each with a counter and a checker ──
    `include "sienna_set_side.svh"
    `include "tb_side_pack.svh"
`define TB_CC_LANES NUM_LANES
`define TB_CC_MEM(i) dut.G_SET.G_COPY[0].pipe.backend_lanes[i].gpnae.COEF
    `include "tb_coef_check.svh"
`undef TB_CC_LANES
`undef TB_CC_MEM
    tb_add_words_t add_words_i;  // the set's ADD words (apply_add)
    localparam int SIDE_W = $bits(set_side_t);
    credit_link_if #(.DATA_W(SIDE_W + ((FAULT == 3) ? 1 : 0)), .CRW(1)) host_lnk[1] ();  // L0 of sienna_top's one copy
    // The layer contract's row links (L10), unused at SET_PORT = 1: nothing is ever put on them.
    credit_link_if #(.DATA_W(N * DATA_WIDTH), .CRW($clog2(N + 1))) idle_rows[2] ();
    logic idle_fault_put = 1'b0;  // FAULT 8: one put on the unused A row link out of reset
    for (genvar r = 0; r < 2; r++) begin : G_IDLE_ROWS
      assign idle_rows[r].put  = (r == 0) && idle_fault_put;
      assign idle_rows[r].data = '0;
    end
    credit_link_if #(.DATA_W(1), .CRW(1)) wc_lnk[2] ();
    logic [1:0] host_cnt;  // staging credits this host holds
    logic       wc_cnt[2];  // per cache region: its credit is held
    bit         wc_open_tb[2];  // per region: a fill is open (put, its last set not yet put)
    logic       drained;  // nothing in flight for a while: every credit must be home
    int         tb_inflight, quiet_cyc;  // sets put and not complete; cycles with none
    int         tb_puts;  // sets put, counted on the same edge as tb_inflight
    localparam int QUIET_TB = 16 + 4 * LINK_STAGES;
    credit_counter #(.MAX(2), .CRW(1)) host_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host_lnk[0].put), .credit_i(host_lnk[0].credit),
                                                .has_credit_o(), .count_o(host_cnt));
    localparam int HOST_SLOTS = (SETS_IN_FLIGHT < 2) ? SETS_IN_FLIGHT : 2;  // the host holds both staging credits unless the entry admits fewer sets
    credit_link_checker #(.SLOTS(HOST_SLOTS)) chk_host (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(host_lnk[0]));
    for (genvar r = 0; r < 2; r++) begin : G_WC
      credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(wc_lnk[r].put), .credit_i(wc_lnk[r].credit),
                                             .has_credit_o(), .count_o(wc_cnt[r]));
      credit_link_checker #(.SLOTS(1)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained && !wc_open_tb[r]), .lnk(wc_lnk[r]));
    end
    logic sum_open;  // the last set put was a partial sum: it completes before the mesh frees its staging bank, so no drain until the sum ends
    set_side_t side_put;
    assign side_put = set_side_t'(host_lnk[0].data[SIDE_W-1:0]);
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) begin
        tb_inflight <= 0;
        tb_puts     <= 0;
        quiet_cyc   <= 0;
        sum_open    <= 1'b0;
      end else begin
        tb_inflight <= tb_inflight + int'(host_lnk[0].put) - int'(pipeline_complete_o);
        tb_puts     <= tb_puts + int'(host_lnk[0].put);
        quiet_cyc   <= (tb_inflight != 0 || host_lnk[0].put || sum_open) ? 0 : (quiet_cyc < QUIET_TB) ? quiet_cyc + 1 : quiet_cyc;
        if (host_lnk[0].put) sum_open <= side_put.accumulate;
      end
    assign drained = (quiet_cyc == QUIET_TB);
    bit acc_on = 0;  // the accumulate pass is running
    // Truth dump: every stage's words of the single-set pass (P0), the back-to-back pass (B2B, when built), each streaming pass (S1, S2, ...)
    // and the accumulate pass (A), for regression.py's true-function gates.
    localparam int TR_IDS = 1 << ID_W;
    integer tr_fd;
    bit tr_on = 0;  // the pass running is dumped
    string tr_pass = "-";
    int tr_seq = 0;  // passes dumped so far: a new one restarts the monitor's per-element result counts
    int tr_streams = 0;  // streaming passes so far
    int tr_cur_k = -1;  // the pass's number of the set the host puts next (-1: a set with no result)
    int tr_k[TR_IDS];  // per set id: the pass's number of the set holding it
    int put_seq = 0;  // puts since reset: set ids count them modulo TR_IDS
    bit stream_on = 0;  // a streaming pass is running
    logic [DATA_WIDTH-1:0] stream_results[$];
    int stream_bounds[$];
    int stream_ids[$];
    int put_sets = 0;  // sets of the streaming pass that have put a word on L9, counted by the L9 consumer
    logic [DATA_WIDTH-1:0] acc_results[$];
    int acc_bounds[$];
    // Admission seen from the host: sets in flight plus credits held never exceed SETS_IN_FLIGHT.
    int over_admit = 0;
    always @(negedge clk_i) if (rstn_i && tb_inflight + int'(host_cnt) > SETS_IN_FLIGHT) over_admit++;

    logic north_write_enable_i, north_write_reset_i;
    logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i;  // operands in the package's format
    logic west_write_enable_i, west_write_reset_i;
    logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i;

    logic                                  pipeline_complete_o;
    logic                        [ID_W-1:0] done_set_id_o;
    logic systolic_busy_tb, gpnae_busy_tb;
    logic maxpool_busy_tb, dropout_busy_tb;

    // ── Data queues ───────────────────────────────────────────────────────
    logic [DATA_WIDTH-1:0] north_data_queue[$];
    logic [DATA_WIDTH-1:0] west_data_queue [$];
    logic [DATA_WIDTH-1:0] expected_results[$];
    logic [DATA_WIDTH-1:0] bound_results[$];  // per-output fp32 error bounds from the golden model
    logic [DATA_WIDTH-1:0] actual_results  [$];

    // ── Verification counters ─────────────────────────────────────────────
    int total_elements = 0, exact_passed = 0, tol_passed = 0, failed = 0;

    // ── Cycle counter (free-running, for heartbeat timestamps) ────────────
    longint cycle_count = 0;
    always_ff @(posedge clk_i) cycle_count <= cycle_count + 1;

    // ── Clock ─────────────────────────────────────────────────────────────
    initial begin
      clk_i = 0;
      forever #5 clk_i = ~clk_i;
    end

    // ── L9: this TB is the downstream consumer of every lane's output link ──
    localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX: the most slots a lane's consumer may advertise
    credit_link_if #(.DATA_W(DATA_WIDTH + ((FAULT == 7) ? 1 : 0)), .CRW(1)) out_lnk[NUM_LANES] ();

    if (FAULT == 8) begin : G_FAULT_IDLE  // sienna_top's a_idle_contract must fire on it
      initial begin
        wait (rstn_i === 1'b1);
        repeat (4) @(negedge clk_i);
        idle_fault_put = 1'b1;
        $display("[FAULT 8] a put on the unused layer contract's A row link");
        @(negedge clk_i) idle_fault_put = 1'b0;
      end
    end

    // ── DUT ───────────────────────────────────────────────────────────────
    sienna_top #(
        .SET_PORT         (1),  // the set contract: this bench is the host on L0
        .COLLAPSE_K       (COLLAPSE_K),  // the package's mesh mode: 1 full-depth tiles, 0 depth slices and the reduce tree
        .NUM_LANES        (NUM_LANES),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (HOST_WORDS),
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
        .FIFO_DEPTH       (FIFO_DEPTH),
        .WC_TILES         (WC_TILES),
        .PACK_ENTRIES     (PACK_ENTRIES),
        .LINK_STAGES      (LINK_STAGES),
        .OUT_MAX          (OUT_CAP),
        .LANE_OUT_SLOTS   ((FAULT == 6) ? 8 : 16)
    ) dut (
        .clk_i                      (clk_i),
        .rstn_i                     (rstn_i),
        // The layer contract, unused at SET_PORT = 1: tied off
        .cfg_load_i          (1'b0),
        .cfg_m_i             ('0),
        .cfg_kb_i            ('0),
        .cfg_n_i             ('0),
        .cfg_residual_i      (1'b0),
        .cfg_bias_i          (1'b0),
        .cfg_act_i           ('0),
        .cfg_tier_i          ('0),
        .cfg_train_i         (1'b0),
        .cfg_seed_i          ('0),
        .cfg_lane_i          ('0),
        .cfg_add_i           ('0),
        .cfg_pack_shift_i    ('0),
        .cfg_pack_map_i      ('0),
        .cfg_pack_act_i      ('0),
        .cfg_pack_lane_i     ('0),
        .busy_o              (),
        .done_o              (),
        .a_rows              (idle_rows[0]),
        .w_rows              (idle_rows[1]),
        .w_bias_i            ('0),
        .w_req_scale_i       ('0),
        .set_done_o          (),
        .host                       (host_lnk),
        .bias_i                     (bias_i),
        .wc_region                  (wc_lnk),
        .wc_write_enable_i          (wc_write_enable_i),
        .wc_write_addr_i            (wc_write_addr_i),
        .north_write_enable_i       (north_write_enable_i),
        .north_write_data_i         (north_write_data_i),
        .north_write_reset_i        (north_write_reset_i),
        .west_write_enable_i        (west_write_enable_i),
        .west_write_data_i          (west_write_data_i),
        .west_write_reset_i         (west_write_reset_i),
        .drained_i                  (1'b0),
        .copy_sel_o                 (),
        .coef_wr_i           (1'b0),  // the lanes keep their reset coefficient memory
        .coef_addr_i         ('0),
        .coef_data_i         ('0),
        .coef_ready_o        (),
        .out                        (out_lnk),
        .pipeline_complete_o        (pipeline_complete_o),
        .done_set_id_o              (done_set_id_o),
        .systolic_busy_o            (systolic_busy_tb),
        .gpnae_busy_o               (gpnae_busy_tb),
        .maxpool_busy_o             (maxpool_busy_tb),
        .dropout_busy_o             (dropout_busy_tb)
    );

    wire [4:0] dut_stage = {dut.G_SET.G_COPY[0].pipe.g_state, dut.G_SET.G_COPY[0].pipe.p_state};  // activation and pooling stage states
    wire res_ready = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.out_full[dut.G_SET.G_COPY[0].pipe.systolic_array_inst.out_rd];  // the mesh holds a finished result

    // ── L9 consumer: out_adv slots per lane after each reset; a word frees its slot at once, its credit returned unless stalled or held ──
    int out_slots = OUT_CAP, out_stall_pct = 0;  // +out_slots=N (1..64) L9 slots per lane; +out_stall_pct=P of credit returns withheld at random
    int out_adv = OUT_CAP;  // slots advertised after the next reset: out_slots, or 1 for the hold pass after the mid-stream reset
    bit out_hold = 0;  // Review Focus 3: every L9 credit withheld
    logic [NUM_LANES-1:0] out_put, out_cr;
    logic [DATA_WIDTH-1:0] out_data[NUM_LANES];
    int out_owed[NUM_LANES];  // freed slots not yet credited
    int out_held[NUM_LANES];  // credits the DUT holds: returned minus words put
    int hold_puts = 0;  // words put while every credit is withheld
    bit hold_on = 0;  // the hold pass runs: its words are checked against the plain pass, not printed
    longint starved = 0;  // lane-cycles of a streaming pass with no L9 credit held by the DUT
    logic [DATA_WIDTH-1:0] lane_q[NUM_LANES][$];  // each lane's words of the set being output
    localparam bit TB_POOL_BYPASS = (POOL_H == 1) && (POOL_W == 1) && (STRIDE_ROWS == 1) && (STRIDE_COLS == 1) && (PADDING == 0);
    localparam int TB_L5 = (FAULT == 6) ? 8 : 16;  // the collector's L5 slots per lane
    initial begin
      void'($value$plusargs("out_slots=%d", out_slots));
      void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
      if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90) begin
        $display("[ERROR] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct);
        $finish;
      end
      out_adv = out_slots;
      $display(" Output links    : %0d L9 slots per lane, %0d%% of credit returns stalled", out_slots, out_stall_pct);
    end
    for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
      assign out_put[l]  = out_lnk[l].put;
      assign out_data[l] = DATA_WIDTH'(out_lnk[l].data);
      assign out_lnk[l].credit = out_cr[l];
      // Drained only once every slot is credited and no credit is still on the wire this cycle.
      credit_link_checker #(.SLOTS(OUT_CAP)) chk (.clk_i(clk_i), .rstn_i(rstn_i),
                                                  .drained_i(drained && out_adv == OUT_CAP && out_owed[l] == 0 && !out_cr[l]), .lnk(out_lnk[l]));
    end

    // Lane L's j-th word of a set is window j*NUM_LANES + L, so popping the lanes in turn rebuilds window order whatever each lane's stalls.
    task automatic take_set(output logic [DATA_WIDTH-1:0] w[$], output int src[$]);
      bit any = 1;
      w.delete();
      src.delete();
      while (any) begin
        any = 0;
        for (int l = 0; l < NUM_LANES; l++)
          if (lane_q[l].size() != 0) begin
            w.push_back(lane_q[l].pop_front());
            src.push_back(l);
            any = 1;
          end
      end
    endtask

    // Link signals change on the falling edge; the words of a set are taken in window order with its completion.
    always @(negedge clk_i) begin
      if (!rstn_i) begin
        for (int l = 0; l < NUM_LANES; l++) begin
          out_cr[l]   = 1'b0;
          out_owed[l] = out_adv;
          out_held[l] = 0;
          lane_q[l].delete();
        end
      end else begin
        // A word's slot is credited from the next cycle on: the producer's counter checks no credit comes with the put that frees it.
        if (stream_on && (|out_put) && put_sets <= stream_bounds.size()) put_sets = stream_bounds.size() + 1;  // before this edge's completion: a set's words can all come with it
        for (int l = 0; l < NUM_LANES; l++) begin
          out_held[l] += int'(out_cr[l]);
          if (stream_on && out_held[l] == 0) starved++;
          out_cr[l] = 1'b0;
          if (out_owed[l] > 0 && !out_hold && !(out_stall_pct > 0 && $urandom_range(99) < out_stall_pct)) begin
            out_cr[l] = 1'b1;
            out_owed[l]--;
          end
          if (out_put[l]) begin
            lane_q[l].push_back(out_data[l]);
            out_owed[l]++;
            out_held[l]--;
            if (out_hold) hold_puts++;
          end
        end
        if (pipeline_complete_o) begin
          logic [DATA_WIDTH-1:0] w[$];
          int src[$];
          take_set(w, src);
          foreach (w[i]) begin
            actual_results.push_back(w[i]);
            if (stream_on) stream_results.push_back(w[i]);
            if (acc_on) acc_results.push_back(w[i]);
            // Data leaving dropout for the output, dec in the build's format (int8: the signed code); not for the accumulate pass, which main never ran
            if (!acc_on && !hold_on) begin
              if (EXP_W == 0)
                $display("[DEBUG %0t] Dropout -> Output (Lane %0d)   : dec=%0d  hex=%08x", $time, src[i], $signed(w[i][7:0]), w[i]);
              else
                $display("[DEBUG %0t] Dropout -> Output (Lane %0d)   : dec=%.4f  hex=%08x", $time, src[i],
                         f32((EXP_W == 8 && MAN_W == 7) ? {16'(w[i]), 16'h0} : 32'(w[i])), w[i]);
            end
          end
          if (stream_on) begin
            stream_bounds.push_back(stream_results.size());
            stream_ids.push_back($isunknown(done_set_id_o) ? -1 : int'(done_set_id_o));  // an unknown id never matches
          end
          if (acc_on) acc_bounds.push_back(acc_results.size());
          if (tr_on && tr_fd) foreach (w[i]) $fdisplay(tr_fd, "D %s %0d %0d %h", tr_pass, tr_k[int'(done_set_id_o)], i, w[i]);
        end
      end
    end

    // Every lane link holds all its slots: L4 32, L5 the collector's, the pooling input (FIFO2, or L9 with a 1x1 pool) and L9 out_adv.
    function automatic bit lanes_home(output string why);
      automatic int pin = TB_POOL_BYPASS ? out_adv : 16;
      for (int l = 0; l < NUM_LANES; l++)
        if (g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cnt[l] !== 32 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_out[l] !== TB_L5 || g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cnt[l] !== pin || out_held[l] !== out_adv ||
            out_owed[l] !== 0) begin
          why = $sformatf("lane %0d: L4 %0d (32), L5 %0d (%0d), pool input %0d (%0d), L9 %0d (%0d), L9 owed %0d (0)", l, g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cnt[l],
                          g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_out[l], TB_L5, g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cnt[l], pin, out_held[l], out_adv, out_owed[l]);
          return 0;
        end
      why = $sformatf("lanes: L4 32, L5 %0d, pool input %0d, L9 %0d each", TB_L5, pin, out_adv);
      return 1;
    endfunction

    // The sideband of the set being put, from the per-set signals above; entry 0 of each table is the set's own.
    function automatic set_side_t side_now();
      set_side_t s;
      logic [PACK_ENTRIES-1:0][TB_ENTRY_W-1:0] ents;
      ents[0] = tb_entry(req_zp_i, req_min_i, req_max_i, gp_mx_i, gp_shx_i, gp_mout_i, gp_shout_i, gp_zout_i);
      for (int e = 1; e < PACK_ENTRIES; e++)
        ents[e] = tb_entry(pack_zp_i[e], pack_min_i[e], pack_max_i[e], pack_mx_i[e], pack_shx_i[e], pack_mout_i[e], pack_shout_i[e], pack_zout_i[e]);
      s               = '0;
      s.wc_last       = wc_last_i;
      s.weight_tile   = weight_tile_i;
      s.weight_cached = weight_cached_i;
      s.accumulate    = accumulate_i;
      s.bias_valid    = bias_valid_i;
      s.train         = training_mode_i;
      s.seed          = dropout_seed_i;
      s.pack_shift    = pack_shift_i;
      s.pack_map      = pack_map_i;
      s.act           = {pack_act_i, activation_function_i};
      s.entry         = tb_side_entries(ents);
      s.scale         = tb_side_scales(req_mult_i, req_shift_i);
      s.res_hold      = res_hold_i;
      s.res_add       = res_add_i;
      s.add           = tb_side_add(add_words_i);
`ifdef TB_FLOAT_ADD
      s.res_add       = 1'b1;  // fault build: every set claims to be an add set, so a build without the residual ADD fires a_no_residual_add
`endif
      return s;
    endfunction

    // Waits for a staging credit; the edge after a put has not counted it yet, so callers leave a cycle after each put.
    task automatic wait_credit();
      while (host_cnt == 0) @(posedge clk_i);
    endtask

    // Puts the set whose rows were just written, its sideband in the data and bias_i beside it; link signals change on the falling edge, so counters, checkers and the DUT all sample the same put.
    task automatic host_put();
      @(negedge clk_i);
      if (host_cnt == 0) begin
        failed++;
        $display("  [FAIL] The host put a set with no staging credit");
      end
      g_pipeline.host_lnk[0].data = side_now();
      g_pipeline.host_lnk[0].put  = 1'b1;
      tr_k[put_seq % TR_IDS] = tr_cur_k;
      put_seq++;
      if (bias_valid_i) check_mesh_bias(bias_want);
      @(negedge clk_i);
      g_pipeline.host_lnk[0].put  = 1'b0;
      if (wc_last_i) wc_open_tb[weight_tile_i[$clog2(WC_TILES)-1]] = 1'b0;
    endtask

    // The bias the mesh captures (bias_push into its bias queue, L0 and L1 stages after the host put) must be want: a zero-extended or flushed bf16 row reads wrong here even where the sums cannot show it.
    task automatic check_mesh_bias(input logic [N-1:0][ACC_DATA_WIDTH-1:0] want);
      fork
        begin
          automatic int q;
          repeat (2 * LINK_STAGES) @(negedge clk_i);
          #1;  // after the put's combinational paths settle
          if (!g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bias_push) begin
            failed++;
            $display("  [FAIL] Bias check: the mesh captured no bias %0d cycles after the host put", 2 * LINK_STAGES);
          end else begin
            q = int'(g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bq_wr);  // the queue entry this capture writes
            @(posedge clk_i);
            #1;  // the registered entry
            bias_checked++;
            if (!g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bias_qv[q]) begin
              failed++;
              $display("  [FAIL] Bias check: bias queue entry %0d was captured without its bias_valid", q);
            end
            for (int c = 0; c < N; c++)
              if (g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bias_q[q][c] !== want[c]) begin
                failed++;
                $display("  [FAIL] Bias check: the mesh's bias queue entry %0d column %0d is %h, the bias row widened is %h", q, c,
                         g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bias_q[q][c], want[c]);
              end
          end
        end
      join_none
    endtask

    // Opens a fill of cache region r: waits for its credit and puts; the rows follow.
    logic [1:0] wc_put_tb = '0;  // per region: this TB's put; interface arrays take constant indices only
    assign wc_lnk[0].put = wc_put_tb[0];
    assign wc_lnk[1].put = wc_put_tb[1];
    assign wc_lnk[0].data = 1'b0;
    assign wc_lnk[1].data = 1'b0;
    task automatic wc_open(input int r);
      @(negedge clk_i);
      while (wc_cnt[r] == 1'b0) @(negedge clk_i);
      wc_put_tb[r] = 1'b1;
      @(negedge clk_i);
      wc_put_tb[r] = 1'b0;
      wc_open_tb[r] = 1'b1;
      @(posedge clk_i);
    endtask

    // A w-bit word's value for debug prints: int8 codes and int32 sums as signed integers, floats widened to fp32 and decoded by f32.
    function automatic real word_val(input logic [31:0] b, input int w);
      if (IS_INT) return real'($signed(b << (32 - w)) >>> (32 - w));
      return f32(b << (32 - w));
    endfunction

    // Manual binary32 decode; $bitstoshortreal leaves the bit pattern as an integer under Verilator.
    function automatic real f32(input logic [31:0] b);
      int  e;
      real m, v;
      e = int'(b[30:23]);
      m = real'(longint'(b[22:0])) / 8388608.0;
      if (e == 255) v = 1.0e38;                         // Inf / NaN, clamped so any finite compare fails
      else if (e == 0) v = 0.0;                         // zero / flushed subnormal
      else v = (1.0 + m) * (2.0 ** (e - 127));
      return b[31] ? -v : v;
    endfunction

    // ── Tolerance check ───────────────────────────────────────────────────
    // Passes within REL_TOL, or within bound: the golden model's fp32 error bound for this output, for sums that cancel.
    function automatic logic check_tolerance(input [DATA_WIDTH-1:0] expected, actual, bound,
                                             output string info);
      real exp_r, act_r, abs_d, rel_d, bnd_r;

      if (EXACT_GOLDEN) begin  // narrow formats: the golden is bit-exact, so only identical bits pass
        info = $sformatf("exp=%h act=%h", expected, actual);
        return expected === actual;
      end
      if ($isunknown(actual)) begin  // an x or z word fails, whatever its real value would be
        info = $sformatf("exp=%h act=%h (unknown bits)", expected, actual);
        return 1'b0;
      end
      exp_r = f32(32'(expected));
      act_r = f32(32'(actual));
      bnd_r = f32(32'(bound));
      abs_d = (exp_r > act_r) ? (exp_r - act_r) : (act_r - exp_r);

      if (exp_r != 0.0) rel_d = abs_d / ((exp_r > 0.0) ? exp_r : -exp_r);
      else rel_d = (act_r == 0.0) ? 0.0 : 1.0;

      info = $sformatf(
          "exp=%g act=%g abs=%.3g (lim %.4f, fp32 bound %.3g) rel=%.4f%% (lim %.1f%%)",
          exp_r,
          act_r,
          abs_d,
          ABS_TOL,
          bnd_r,
          rel_d * 100.0,
          REL_TOL * 100.0
      );
      if (bnd_r > 0.0 && abs_d <= bnd_r) return 1'b1;

      case (TOLERANCE_MODE)
        "ABSOLUTE": return (abs_d <= ABS_TOL);
        "RELATIVE": return (rel_d <= REL_TOL);
        "BOTH":     return (abs_d <= ABS_TOL) && (rel_d <= REL_TOL);
        default:    return (abs_d <= ABS_TOL);
      endcase
    endfunction

    // Checker self-test: a loose or broken compare must fail the run before any result is trusted.
    initial begin
      string st_info;
      if (EXACT_GOLDEN && (!check_tolerance(DATA_WIDTH'(16'h3F80), DATA_WIDTH'(16'h3F80), '0, st_info) ||
                           check_tolerance(DATA_WIDTH'(16'h3F80), DATA_WIDTH'(16'h3F81), '0, st_info))) begin
        $display("[FAIL] Exact checker self-test failed; results cannot be trusted");
        $finish;
      end
      if (!EXACT_GOLDEN && (!check_tolerance(32'h3f800000, 32'h3f800003, 0, st_info) ||   // 1.0 vs 1.0 + 3 ulp: pass
          check_tolerance(32'h3f800000, 32'h40000000, 0, st_info) ||    // 1.0 vs 2.0: fail
          check_tolerance(32'h3f800000, 32'h3f7ae148, 0, st_info) ||    // 1.0 vs 0.98: fail
          check_tolerance(32'h3f800000, 32'hbf800000, 0, st_info) ||    // 1.0 vs -1.0: fail
          check_tolerance(32'hbf000000, 32'hbd4ccccd, 0, st_info) ||    // -0.5 vs -0.05: fail
          !check_tolerance(32'h370e8795, 32'h37020000, 32'h3727c5ac, st_info) ||  // 8.5e-6 vs 7.7e-6, bound 1e-5: pass
          check_tolerance(32'h370e8795, 32'h37020000, 32'h350637bd, st_info) ||   // same (off by 7.5e-7), bound 5e-7: fail
          check_tolerance(32'h3f800000, 32'h40000000, 32'h3a83126f, st_info))) begin // 1.0 vs 2.0, bound 1e-3: fail
        $display("[FAIL] Tolerance checker self-test failed; results cannot be trusted");
        $finish;
      end
    end

    // ── .mem reader ───────────────────────────────────────────────────────
    task automatic read_mem_file(input string fn, output logic [DATA_WIDTH-1:0] q[$]);
      integer fh, rc;
      logic [DATA_WIDTH-1:0] w;
      q.delete();
      fh = $fopen(fn, "r");
      if (!fh) begin
        $display("[ERROR] Cannot open: %s", fn);
        $finish;
      end
      while (!$feof(
          fh
      )) begin
        rc = $fscanf(fh, "%h", w);
        if (rc == 1) q.push_back(w);
      end
      $fclose(fh);
    endtask

    `include "int8_tb_util.svh"  // read_word_file, unpack_requant, apply_requant

    // ── Helper: print current DUT status signals ──────────────────────────
    // Armed for the back-to-back pass only: every outer-FSM transition with its timestamp.
    logic trace_states = 0;
    int   prev_state = -1;
    always @(posedge clk_i) begin
      if (trace_states && dut_stage !== prev_state) begin
        $display("  [FSM] @%0t state %0d -> %0d  (res_ready=%0b mult_complete=%0b)",
                 $time, prev_state, dut_stage,
                 res_ready, dut.G_SET.G_COPY[0].pipe.systolic_mult_complete);
        prev_state = dut_stage;
      end
    end

    task automatic print_status(input string stage_label);
      $display(
          "[STATUS @ %0t] sys_busy=%0b gpnae_busy=%0b max_busy=%0b drop_busy=%0b",
          $time, systolic_busy_tb, gpnae_busy_tb, maxpool_busy_tb, dropout_busy_tb);
    endtask

    // ── Reset ─────────────────────────────────────────────────────────────
    task automatic reset();
      $display("\n[STAGE] Reset");
      rstn_i = 0;
      put_seq = 0;
      out_cr = '0;  // the downstream consumer drops its L9 credits with the reset, as it does every other link signal
      g_pipeline.host_lnk[0].put = 1'b0;
      g_pipeline.host_lnk[0].data = '0;
      wc_put_tb = '0;
      wc_open_tb[0] = 1'b0;
      wc_open_tb[1] = 1'b0;
      wc_last_i = 1'b0;
      training_mode_i = 1'b0;
      accumulate_i = 1'b0;
      bias_valid_i = 1'b0;
      bias_i = '0;
      req_mult_i = '0;
      req_shift_i = '0;
      req_zp_i = '0;
      req_min_i = '0;
      req_max_i = '0;
      gp_mx_i = '0;
      gp_shx_i = '0;
      gp_mout_i = '0;
      gp_shout_i = '0;
      gp_zout_i = '0;
      pack_shift_i = '0;
      pack_map_i = '0;
      {pack_act_i, pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i, pack_mx_i, pack_shx_i, pack_mout_i} = '0;
      apply_add(0, 1);
      weight_cached_i = 1'b0;
      weight_tile_i = '0;
      wc_write_enable_i = 1'b0;
      wc_write_addr_i = '0;
      dropout_seed_i = '1;
      north_write_reset_i = 1;
      west_write_reset_i = 1;
      north_write_enable_i = 0;
      west_write_enable_i = 0;
      north_write_data_i = '0;
      west_write_data_i = '0;
      activation_function_i = '0;
      repeat (10) begin
        @(negedge clk_i);
        // Review Focus 1: while reset is held every producer count is 0 and no consumer advertises a credit.
        if (host_cnt !== 0 || wc_cnt[0] || wc_cnt[1] || g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out !== 0 ||
            g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt !== 0 || g_pipeline.host_lnk[0].credit || g_pipeline.wc_lnk[0].credit || g_pipeline.wc_lnk[1].credit || g_pipeline.dut.G_SET.G_COPY[0].pipe.l3_grant ||
            g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_credit || g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.stg_credit) begin
          failed++;
          $display("  [FAIL] In reset: host %0d, regions %0d %0d, staging %0d, act banks %0d, sets %0d, granted %0d, results %0d; credits out %0b%0b%0b%0b%0b%0b",
                   host_cnt, wc_cnt[0], wc_cnt[1], g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt,
                   g_pipeline.host_lnk[0].credit, g_pipeline.wc_lnk[0].credit, g_pipeline.wc_lnk[1].credit, g_pipeline.dut.G_SET.G_COPY[0].pipe.l3_grant, g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_credit, g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.stg_credit);
        end
        // The lane links too: no L4, L5, pooling-input or L9 count and no credit while reset is held.
        for (int l = 0; l < NUM_LANES; l++)
          if (g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cnt[l] !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_out[l] !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cnt[l] !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cr[l] || g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_cr[l] || g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cr[l]) begin
            failed++;
            $display("  [FAIL] In reset: lane %0d L4 %0d, L5 %0d, pool input %0d; credits out %0b%0b%0b", l, g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cnt[l], g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_out[l],
                     g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cnt[l], g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cr[l], g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_cr[l], g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cr[l]);
            break;
          end
      end
      @(posedge clk_i);
      rstn_i = 1;
      north_write_reset_i = 0;
      west_write_reset_i = 0;
      if (FAULT == 1) begin  // a put with no credit and no rows, the cycle reset ends
        @(negedge clk_i);
        g_pipeline.host_lnk[0].data = side_now();
        g_pipeline.host_lnk[0].put = 1'b1;
        @(negedge clk_i);
        g_pipeline.host_lnk[0].put = 1'b0;
        $display("  [FAULT 1] put with no credit and no rows @ %0t", $time);
      end
      repeat (5) @(posedge clk_i);
      cc_image("coef_default.mem");
      if (FAULT == 9) begin
        cc_img[7] ^= 32'd1;
        $display("  [FAULT 9] the reset image's word 7 flipped");
      end
      cc_check("reset");
      $display("  Reset complete @ %0t", $time);
    endtask

    // ── Load inputs ───────────────────────────────────────────────────────
    task automatic load_inputs();
      $display("\n[STAGE] Loading inputs");
      $display("  West  queue : %0d words", west_data_queue.size());
      $display("  North queue : %0d words", north_data_queue.size());

      load_range(0, 32'h7FFF_FFFF);
      $display("  Load complete @ %0t  (%0d cycles)", $time, cycle_count);
    endtask

    // The output words reach actual_results from the L9 consumer, a set at a time in window order.

    // ── Collect outputs — with heartbeat ─────────────────────────────────
    task automatic collect_outputs();
      automatic int waited = 0;

      $display("\n[STAGE] Waiting for pipeline_complete_o");
      $display("  (heartbeat every %0d cycles, timeout at %0d cycles)", HEARTBEAT_CYCLES,
               TIMEOUT_CYCLES);

      // Wait loop with heartbeat
      while (!pipeline_complete_o) begin
        @(posedge clk_i);
        waited++;

        if (waited % HEARTBEAT_CYCLES == 0)
          print_status($sformatf("still waiting (%0d / %0d cycles)", waited, TIMEOUT_CYCLES));

        if (waited >= TIMEOUT_CYCLES) begin
          $display("[FATAL] Timeout after %0d cycles - pipeline_complete_o never asserted.", waited);
          $display("  Last known state:");
          print_status("at timeout");
          // Which stage is stuck is the whole diagnosis; sys_busy alone does not say.
          $display("  stage states {g,p} = %0d", dut_stage);
          $display("  filled_total=%0d  total_elements=%0d  all_collected=%0b", g_pipeline.dut.G_SET.G_COPY[0].pipe.filled_total,
                   g_pipeline.dut.G_SET.G_COPY[0].pipe.total_elements, g_pipeline.dut.G_SET.G_COPY[0].pipe.all_collected);
          $display("  res_ready=%0b  result beat=%0b  l3_armed=%0b  disp_done=%0b",
                   res_ready, g_pipeline.dut.G_SET.G_COPY[0].pipe.wide_rd_valid, g_pipeline.dut.G_SET.G_COPY[0].pipe.l3_armed, g_pipeline.dut.G_SET.G_COPY[0].pipe.disp_done);
          // The host side of L0 and the mesh's staging side of L1.
          $display("  host credits=%0d sets_out=%0d granted=%0d staging=%0d north_queue_empty=%0b west_queue_empty=%0b",
                   host_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.north_queue_empty, g_pipeline.dut.G_SET.G_COPY[0].pipe.west_queue_empty);
          for (int i = 0; i < 4; i++)
            $display("  lane %0d: fill_count=%0d done_count=%0d load_finalized=%0b collected=%0b",
                     i, g_pipeline.dut.G_SET.G_COPY[0].pipe.fill_count[i], g_pipeline.dut.G_SET.G_COPY[0].pipe.done_count[i], g_pipeline.dut.G_SET.G_COPY[0].pipe.load_finalized[i],
                     g_pipeline.dut.G_SET.G_COPY[0].pipe.lane_collected[i]);
          if (lane_fd) $fclose(lane_fd);
          $finish;
        end
      end

      $display("  pipeline_complete_o asserted @ %0t  (%0d cycles)", $time, waited);
      print_status("at completion");
      @(posedge clk_i);  // the capture block takes the last beat on this edge
      $display("  Captured %0d words out of %0d expected", actual_results.size(),
               expected_results.size());
    endtask

    // ── Verify ───────────────────────────────────────────────────────────
    task automatic verify_outputs();
      automatic int n_exp = expected_results.size();
      automatic int n_act = actual_results.size();
      automatic logic [DATA_WIDTH-1:0] ev, av;
      string info;

      $display("\n[STAGE] Verifying - %0d expected, %0d captured", n_exp, n_act);

      for (int i = 0; i < n_exp && i < n_act; i++) begin
        ev = expected_results[i];
        av = actual_results[i];
        total_elements++;
        if (ev === av) begin
          exact_passed++;
        end else if (check_tolerance(ev, av, (i < bound_results.size()) ? bound_results[i] : '0, info)) begin
          tol_passed++;
          $display("  [PASS-TOL] [%0d] exp=0x%h act=0x%h | %s", i, ev, av, info);
        end else begin
          failed++;
          $display("  [FAIL]     [%0d] exp=0x%h act=0x%h | %s", i, ev, av, info);
        end
      end

      // Elements that were expected but never captured
      for (int i = n_act; i < n_exp; i++) begin
        total_elements++;
        failed++;
        $display("  [FAIL]     [%0d] exp=0x%h  act=MISSING", i, expected_results[i]);
      end

      // Words captured beyond the expected ones: an empty or short golden must not pass
      for (int i = n_exp; i < n_act; i++) begin
        total_elements++;
        failed++;
        $display("  [FAIL]     [%0d] exp=NONE  act=0x%h", i, actual_results[i]);
      end

      // An empty golden compares nothing, so the pass fails even with nothing captured
      if (n_exp == 0) begin
        failed++;
        $display("  [FAIL]     no expected words: an empty golden is no reference");
      end
    endtask

    // ── File and Console Output Monitors ──────────────────────────────────
    integer trace_fd;

    initial begin
      trace_fd = $fopen("../testbenches/hardware_trace.txt", "w");
      if (!trace_fd) begin
        $display("[ERROR] Could not open hardware_trace.txt for writing.");
      end else begin
        $fdisplay(trace_fd, "==============================================");
        $fdisplay(trace_fd, " SIENNA Hardware Intermediate Trace");
        $fdisplay(trace_fd, "==============================================\n");
      end
    end

    always @(posedge clk_i) begin
      if (rstn_i) begin
        // Console Print: Dispatcher data entering Maxpool
        for (int lane = 0; lane < NUM_LANES; lane++) begin
          if (dut.G_SET.G_COPY[0].pipe.fifo2_wr_valid[lane]) begin
            $display("[DEBUG %0t] Dispatcher -> Maxpool (Lane %0d) : dec=%.4f  hex=%08x", $time, lane,
                     word_val(32'(dut.G_SET.G_COPY[0].pipe.fifo2_wr_data[lane]), DATA_WIDTH), dut.G_SET.G_COPY[0].pipe.fifo2_wr_data[lane]);
          end
        end
        // Dropout -> Output lines are printed by the L9 consumer, a set at a time in window order, so stalls cannot reorder them.
      end
    end

    always @(posedge clk_i) begin
      if (rstn_i && trace_fd) begin
        if (dut.G_SET.G_COPY[0].pipe.wide_rd_valid) begin
          for (int lane = 0; lane < NUM_LANES; lane++)
            $fdisplay(trace_fd, "[%0t] Systolic -> GPNAE   : dec=%.6f  hex=%08x", $time,
                      word_val(32'(dut.G_SET.G_COPY[0].pipe.wide_rd_data[lane]), RESULT_DATA_WIDTH), dut.G_SET.G_COPY[0].pipe.wide_rd_data[lane]);
        end
        for (int lane = 0; lane < NUM_LANES; lane++) begin
          if (dut.G_SET.G_COPY[0].pipe.fifo2_rd_ready[lane] && dut.G_SET.G_COPY[0].pipe.fifo2_rd_valid[lane]) begin
            $fdisplay(trace_fd, "[%0t] GPNAE -> Maxpool    lane=%0d : dec=%.6f  hex=%08x", $time,
                      lane, word_val(32'(dut.G_SET.G_COPY[0].pipe.fifo2_rd_data[lane]), DATA_WIDTH),
                      dut.G_SET.G_COPY[0].pipe.fifo2_rd_data[lane]);
          end
        end
        for (int lane = 0; lane < NUM_LANES; lane++) begin
          if (dut.G_SET.G_COPY[0].pipe.dropout_in_valid[lane]) begin
            $fdisplay(trace_fd, "[%0t] Maxpool -> Dropout  lane=%0d : dec=%.6f  hex=%08x", $time,
                      lane, word_val(32'(dut.G_SET.G_COPY[0].pipe.dropout_data_in[lane]), DATA_WIDTH),
                      dut.G_SET.G_COPY[0].pipe.dropout_data_in[lane]);
          end
        end
      end
    end

    // Truth dump lines, each tagged with the pass and the pass's set number k (M: the r-th result of the pass):
    //   M pass r e sum word   a reducer's write of element e: the sum before fpNarrow (fp32; int32 in int8) and the word written
    //   X pass k e word       the activation stage's input of element e (the mesh's word; int8: the requantize output)
    //   A pass k e word       the activation bank as pooling takes it
    //   P pass k lane j word keep   lane's j-th dropout input (window j*NUM_LANES + lane) and dropout's keep decision for it
    //   D pass k i word       the i-th output word (window order)
    localparam int TR_TPD = N / TILE_SIZE;  // output tiles per dimension, one reducer each
    localparam logic [63:0] TR_THR = (((64'(1) << LFSR_WIDTH) - 64'(1)) * 64'(DROPOUT_P_PERCENT)) / 64'(100);  // dropout.sv's threshold
    logic [TR_TPD*TR_TPD-1:0] tr_we;
    logic [TR_TPD*TR_TPD-1:0][31:0] tr_addr;
    logic [TR_TPD*TR_TPD-1:0][ACC_DATA_WIDTH-1:0] tr_sum;
    logic [TR_TPD*TR_TPD-1:0][RESULT_DATA_WIDTH-1:0] tr_word;
    for (genvar ti = 0; ti < TR_TPD; ti++) begin : TR_ROW
      for (genvar tj = 0; tj < TR_TPD; tj++) begin : TR_COL
        assign tr_we[ti*TR_TPD+tj]   = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.sram_we_agg[ti*TR_TPD+tj];
        assign tr_addr[ti*TR_TPD+tj] = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.sram_addr_agg[ti*TR_TPD+tj];
        assign tr_word[ti*TR_TPD+tj] = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.sram_data_agg[ti*TR_TPD+tj];
        if (RESULT_DATA_WIDTH != ACC_DATA_WIDTH) begin : G_NARROW  // the sum going into the reducer's fp32ToBf16, as AccumulationUnit picks it
          assign tr_sum[ti*TR_TPD+tj] = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.ROW[ti].COL[tj].acc_unit.G_NARROW.NARROW.fp32_i;
        end else begin : G_SAME  // fp32 and int8 write the sum itself
          assign tr_sum[ti*TR_TPD+tj] = ACC_DATA_WIDTH'(dut.G_SET.G_COPY[0].pipe.systolic_array_inst.sram_data_agg[ti*TR_TPD+tj]);
        end
      end
    end
    logic [LFSR_WIDTH-1:0] tr_lfsr[NUM_LANES];
    for (genvar g = 0; g < NUM_LANES; g++) begin : TR_LANE
      assign tr_lfsr[g] = dut.G_SET.G_COPY[0].pipe.backend_lanes[g].dropout_inst.lfsr_next;  // the word this beat advances to decides keep
    end
    function automatic int tr_elem(input int k, input int i, input bit pk);  // sienna_top's elem(): lane k's element i
      return pk ? ((k / N) * PER_LANE + i) * N + (k % N) : k * PER_LANE + i;
    endfunction
    initial begin
      tr_fd = $fopen("../testbenches/truth_dump.txt", "w");
`ifdef BACK_TO_BACK
      if (tr_fd) $fdisplay(tr_fd, "H b2b 1");  // the passes this build runs: regression.py requires each one's lines
`else
      if (tr_fd) $fdisplay(tr_fd, "H b2b 0");
`endif
    end
    initial begin : tr_monitor  // one process writes every monitor variable, on the falling edge
      int wn[SRAM_DEPTH];  // results of the pass written so far, per element
      int beat[TR_IDS];  // beats of each set id read from the mesh so far
      int pj[NUM_LANES];  // dropout inputs of the pooled set so far, per lane
      int q_id[$], q_i[$];
      bit q_pk[$];
      logic [$clog2(TR_IDS)-1:0] h_id[$];  // int8 residual: hold beats inside requantize, their set id and beat index
      int h_i[$];
      logic [NUM_LANES-1:0][7:0] a_q1[$], a_q2[$];  // add beats inside add_lanes, their two inputs
      int seen;
      seen = 0;
      forever begin
        @(negedge clk_i);
        if (tr_seq != seen) wn = '{default: 0};
        seen = tr_seq;
        if (!rstn_i) begin
          q_id.delete();
          q_i.delete();
          q_pk.delete();
          h_id.delete();
          h_i.delete();
          a_q1.delete();
          a_q2.delete();
        end else begin
          for (int t = 0; t < TR_TPD * TR_TPD; t++)
            if (tr_we[t]) begin
              automatic int e = int'(tr_addr[t]) % SRAM_DEPTH;
              if (tr_on && tr_fd) $fdisplay(tr_fd, "M %s %0d %0d %h %h", tr_pass, wn[e], e, tr_sum[t], tr_word[t]);
              wn[e]++;
            end
          if (dut.G_SET.G_COPY[0].pipe.g_accept) beat[int'(dut.G_SET.G_COPY[0].pipe.g_next_id)] = 0;
          if (dut.G_SET.G_COPY[0].pipe.wide_rd_valid) begin  // a beat leaves the mesh; floats fill with it, int8 after the requantize pipeline
            if (!dut.G_SET.G_COPY[0].pipe.g_hold) begin  // a hold set's beats go to hold_mem and never fill
              q_id.push_back(int'(dut.G_SET.G_COPY[0].pipe.g_set_id));
              q_i.push_back(beat[int'(dut.G_SET.G_COPY[0].pipe.g_set_id)]);
              q_pk.push_back(dut.G_SET.G_COPY[0].pipe.g_pack);
            end else begin
              h_id.push_back(dut.G_SET.G_COPY[0].pipe.g_set_id);
              h_i.push_back(beat[int'(dut.G_SET.G_COPY[0].pipe.g_set_id)]);
            end
            beat[int'(dut.G_SET.G_COPY[0].pipe.g_set_id)]++;
          end
          if (dut.G_SET.G_COPY[0].pipe.hold_wr_s) begin  // a hold beat into hold_mem: its requantized codes (R)
            if (h_id.size() == 0) begin
              failed++;
              $display("  [FAIL] truth dump: a hold_mem write with no hold beat in the requantize pipeline");
            end else begin
              automatic logic [$clog2(TR_IDS)-1:0] hid = h_id.pop_front();
              automatic int i = h_i.pop_front();
              if (tr_on && tr_fd != 0)
                for (int k = 0; k < NUM_LANES; k++) $fdisplay(tr_fd, "R %s %0d %0d %h", tr_pass, tr_k[hid], tr_elem(k, i, 1'b0), dut.G_SET.G_COPY[0].pipe.hold_d_s[k]);
            end
          end
          if (dut.G_SET.G_COPY[0].pipe.fill_v && q_id.size() != 0) begin
            automatic int id = q_id.pop_front(), i = q_i.pop_front();
            automatic bit pk = q_pk.pop_front();
            if (dut.G_SET.G_COPY[0].pipe.fill_add && a_q1.size() == 0) begin
              failed++;
              $display("  [FAIL] truth dump: an add_lanes beat filled with no add beat entering add_lanes before it");
            end else if (dut.G_SET.G_COPY[0].pipe.fill_add) begin  // the ADD's inputs and output (Q)
              automatic logic [NUM_LANES-1:0][7:0] q1 = a_q1.pop_front(), q2 = a_q2.pop_front();
              if (tr_on && tr_fd != 0)
                for (int k = 0; k < NUM_LANES; k++)
                  $fdisplay(tr_fd, "Q %s %0d %0d %h %h %h", tr_pass, tr_k[id], tr_elem(k, i, pk), q1[k], q2[k], dut.G_SET.G_COPY[0].pipe.fill_d[k][7:0]);
            end
            if (tr_on && tr_fd)
              for (int k = 0; k < NUM_LANES; k++) $fdisplay(tr_fd, "X %s %0d %0d %h", tr_pass, tr_k[id], tr_elem(k, i, pk), dut.G_SET.G_COPY[0].pipe.fill_d[k]);
          end
          if (dut.G_SET.G_COPY[0].pipe.add_in_s) begin  // an add beat enters add_lanes with hold_mem's beat (q1) and requantize's (q2)
            a_q1.push_back(dut.G_SET.G_COPY[0].pipe.add_q1_s);
            a_q2.push_back(dut.G_SET.G_COPY[0].pipe.add_q2_s);
          end
          if (dut.G_SET.G_COPY[0].pipe.p_accept) begin
            for (int l = 0; l < NUM_LANES; l++) pj[l] = 0;
            if (tr_on && tr_fd)
              for (int e = 0; e < SRAM_DEPTH; e++)
                $fdisplay(tr_fd, "A %s %0d %0d %h", tr_pass, tr_k[int'(dut.G_SET.G_COPY[0].pipe.p_next_id)], e, dut.G_SET.G_COPY[0].pipe.gpnae_out_mem[dut.G_SET.G_COPY[0].pipe.act_rd_base+e]);
          end
          for (int l = 0; l < NUM_LANES; l++)
            if (dut.G_SET.G_COPY[0].pipe.dropout_in_valid[l]) begin
              if (tr_on && tr_fd)
                $fdisplay(tr_fd, "P %s %0d %0d %0d %h %0d", tr_pass, tr_k[int'(dut.G_SET.G_COPY[0].pipe.p_set_id)], l, pj[l], dut.G_SET.G_COPY[0].pipe.dropout_data_in[l],
                          64'(tr_lfsr[l]) >= TR_THR);
              pj[l]++;
            end
        end
      end
    end

    integer       lane_fd;
    logic   [4:0] prev_current_state;
    logic         lane_log_init_done;

    initial begin
      lane_fd = $fopen("../testbenches/pipeline_lane_status.txt", "w");
      if (!lane_fd) begin
        $display("[ERROR] Could not open pipeline_lane_status.txt for writing.");
      end else begin
        $fdisplay(lane_fd, "==============================================");
        $fdisplay(lane_fd, " SIENNA Per-Lane Streaming Status");
        $fdisplay(lane_fd, " credits held per lane: L4 into the GPNAE lane, L5 by the lane, pool input (FIFO2, or L9 with a 1x1 pool)");
        $fdisplay(lane_fd, "==============================================\n");
      end
      lane_log_init_done = 1'b0;
    end

    task automatic print_lane_status(input string reason);
      if (!lane_fd) return;
      $fdisplay(lane_fd,
                "---- %0t  (%s)  current_state=%0d  all_collected=%0b  streaming_complete=%0b ----",
                $time, reason, dut_stage, g_pipeline.dut.G_SET.G_COPY[0].pipe.all_collected, g_pipeline.dut.G_SET.G_COPY[0].pipe.streaming_complete);
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        $fdisplay(
            lane_fd,
            "  lane=%0d fill_count=%0d done_count=%0d load_finalized=%0b lane_collected=%0b | L4 %0d L5 %0d pool-in %0d | maxpool out %0d of %0d | dropout_out_count=%0d",
            lane, g_pipeline.dut.G_SET.G_COPY[0].pipe.fill_count[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.done_count[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.load_finalized[lane],
            g_pipeline.dut.G_SET.G_COPY[0].pipe.lane_collected[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.l4_cnt[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.l5_out[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.pin_cnt[lane],
            g_pipeline.dut.G_SET.G_COPY[0].pipe.mp_out_count[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.lane_windows_total[lane], g_pipeline.dut.G_SET.G_COPY[0].pipe.dropout_out_count[lane]);
      end
    endtask

    always @(posedge clk_i) begin
      if (rstn_i && lane_fd) begin
        if (!lane_log_init_done) begin
          prev_current_state <= dut_stage;
          lane_log_init_done <= 1'b1;
        end else begin
          if (dut_stage != prev_current_state) begin
            print_lane_status("current_state changed");
          end else if (cycle_count % HEARTBEAT_CYCLES == 0) begin
            print_lane_status("heartbeat");
          end
          prev_current_state <= dut_stage;
        end
      end
    end

    // A residual test's add_<k>.mem: flags (partial, bias, hold, add, activation in 7:4), then the ADD's words; other tests leave the table empty.
    localparam int ADD_MEM_WORDS = 9;  // flags, zp1, zp2, m1, m2, shift, zpo, amin, amax
    logic [31:0] add_tab[int][ADD_MEM_WORDS];
    task automatic read_add_tables();
      logic [31:0] q[$];
      if (k_residual == 0) return;
      for (int k = 0; k < k_num_sets; k++) begin
        read_word_file($sformatf("add_%0d.mem", k), q);
        if (q.size() != ADD_MEM_WORDS) $fatal(1, "TB_sienna pipeline: add_%0d.mem holds %0d words, not %0d", k, q.size(), ADD_MEM_WORDS);
        for (int i = 0; i < ADD_MEM_WORDS; i++) add_tab[k][i] = q[i];
      end
    endtask

    // With k_accum_passes = P, sets come in groups of P: the first P-1 are partial sums, the last is activated; a residual test's flags say.
    function automatic bit is_partial(input int k);
      if (k_residual != 0) return add_tab[k][0][0];
      return (k_accum_passes > 1) && ((k % k_accum_passes) != k_accum_passes - 1);
    endfunction

    // Set k's activation: with k_mixed_len > 0 the codes cycle through k_mixed_acts, 4 bits per set; a residual test's from its flags.
    function automatic logic [CONTROL_WIDTH-1:0] act_of(input int k);
      if (k_residual != 0) return CONTROL_WIDTH'(add_tab[k][0][7:4]);  // hold sets linear, each group's activation from the generator
      return (k_mixed_len > 0) ? CONTROL_WIDTH'((k_mixed_acts >> (4 * (k % k_mixed_len))) & 15) : CONTROL_WIDTH'(k_activation_code);
    endfunction

    // Set k's hold and add flags and ADD words (add_<k>.mem); zeros outside a residual test and in a pass that puts the set alone.
    task automatic apply_add(input int k, input bit single = 0);
      {res_hold_i, res_add_i, add_words_i} = '0;
      if (k_residual == 0 || single) return;
      res_hold_i        = add_tab[k][0][2];
      res_add_i         = add_tab[k][0][3];
      add_words_i.zp1   = add_tab[k][1][7:0];
      add_words_i.zp2   = add_tab[k][2][7:0];
      add_words_i.m1    = add_tab[k][3];
      add_words_i.m2    = add_tab[k][4];
      add_words_i.shift = add_tab[k][5][5:0];
      add_words_i.zpo   = add_tab[k][6][7:0];
      add_words_i.amin  = add_tab[k][7][7:0];
      add_words_i.amax  = add_tab[k][8][7:0];
    endtask

    // Set k's bias: with k_has_bias, the first pass of each group reads bias_<k>.mem; a residual test's flags say.
    task automatic apply_bias(input int k);
      logic [31:0] q[$];
      bias_valid_i = (k_residual != 0) ? add_tab[k][0][1] : ((k_has_bias != 0) && ((k % k_accum_passes) == 0));
      bias_i = '0;
      if (bias_valid_i) begin
        read_word_file($sformatf("bias_%0d.mem", k), q);
        for (int c = 0; c < N; c++) bias_i[c] = !(sienna_fmt_pkg::BIAS_DATA_WIDTH == sienna_fmt_pkg::DATA_WIDTH) ? ACC_DATA_WIDTH'(q[c]) : (ACC_DATA_WIDTH'(q[c]) << (ACC_DATA_WIDTH - DATA_WIDTH));
        for (int c = 0; c < N; c++) bias_want[c] = !(sienna_fmt_pkg::BIAS_DATA_WIDTH == sienna_fmt_pkg::DATA_WIDTH) ? q[c] : ACC_DATA_WIDTH'(q[c][DATA_WIDTH-1:0]) << (ACC_DATA_WIDTH - DATA_WIDTH);
      end
    endtask

    // Set k's pack shift, block map and table entries 1..7 (pack_<k>.mem); unpacked tests drive zeros.
    task automatic apply_pack(input int k);
      logic [31:0] q[$];
      pack_shift_i = '0;
      pack_map_i = '0;
      {pack_act_i, pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i, pack_mx_i, pack_shx_i, pack_mout_i} = '0;
      if (k_packed == 0) return;
      read_word_file($sformatf("pack_%0d.mem", k), q);
      pack_shift_i = q[0][2:0];
      for (int c = 0; c < N / 2; c++) pack_map_i[c] = q[1+c][PEW-1:0];
      for (int e = 1; e < PACK_ENTRIES; e++) begin
        automatic int o = 1 + N / 2 + 9 * (e - 1);
        pack_act_i[e]   = q[o][CONTROL_WIDTH-1:0];
        pack_zp_i[e]    = q[o+1][7:0];
        pack_min_i[e]   = q[o+2][7:0];
        pack_max_i[e]   = q[o+3][7:0];
        pack_mx_i[e]    = q[o+4][15:0];
        pack_shx_i[e]   = q[o+5][4:0];
        if (q[o+6][31:24] != 0 || q[o+7][31:6] != 0) $fatal(1, "TB_sienna pipeline: pack entry %0d's SELU words %h %h are too wide", e, q[o+6], q[o+7]);
        pack_mout_i[e]  = q[o+6][23:0];
        pack_shout_i[e] = q[o+7][5:0];
        pack_zout_i[e]  = q[o+8][7:0];
      end
    endtask

    // With k_weight_cache, set k's B is written once into cache tile k of region 0, one fill, and the set sends only A.
    task automatic write_cache();
      logic [DATA_WIDTH-1:0] q[$];
      if (k_weight_cache == 0) return;
      if (k_num_sets > WC_TILES / 2) begin
        $display("[FATAL] %0d cached sets do not fit cache region 0 (%0d tiles)", k_num_sets, WC_TILES / 2);
        $finish;
      end
      wc_open(0);
      for (int k = 0; k < k_num_sets; k++) begin
        read_mem_file($sformatf("matrix_north_%0d.mem", k), q);
        for (int i = 0; i < N * N; i += HOST_WORDS) begin
          wc_write_enable_i = 1'b1;
          wc_write_addr_i = ($clog2(WC_TILES*N*N))'(k * N * N + i);
          for (int c = 0; c < HOST_WORDS; c++) north_write_data_i[c] = q[i+c];
          @(posedge clk_i);
        end
      end
      wc_write_enable_i = 1'b0;
      north_write_data_i = '0;
      $display("  [Cache] %0d weight tiles written", k_num_sets);
    endtask

    task automatic apply_weight(input int k);
      weight_cached_i = (k_weight_cache != 0);
      weight_tile_i = ($clog2(WC_TILES))'(k);
      if (weight_cached_i) north_data_queue.delete();
    endtask

    // Set k's dropout seed; regression.py's set_dropout_seed() mirrors it.
    function automatic logic [LFSR_WIDTH-1:0] set_seed(input int k);
      return LFSR_WIDTH'(k_dropout_seed ^ (32'h85EBCA6B * k));
    endfunction

    // ── Partial load: words [lo, hi) of both queues, HOST_WORDS per write ─
    task automatic load_range(input int lo, input int hi);
      fork
        begin
          for (int i = lo; i < hi && i < west_data_queue.size(); i += HOST_WORDS) begin
            west_write_enable_i = 1;
            for (int c = 0; c < HOST_WORDS; c++)
              west_write_data_i[c] = (i + c < hi && i + c < west_data_queue.size()) ? west_data_queue[i+c] : '0;
            @(posedge clk_i);
          end
          west_write_enable_i = 0;
          @(posedge clk_i);
        end
        begin
          for (int i = lo; i < hi && i < north_data_queue.size(); i += HOST_WORDS) begin
            north_write_enable_i = 1;
            for (int c = 0; c < HOST_WORDS; c++)
              north_write_data_i[c] = (i + c < hi && i + c < north_data_queue.size()) ? north_data_queue[i+c] : '0;
            @(posedge clk_i);
          end
          north_write_enable_i = 0;
          @(posedge clk_i);
        end
      join
    endtask

    // ── Streaming: K distinct sets through overlapped stages ──────────────
    // The monitor reads registered state on the falling edge, never a combinational view of start.
    int ov_mesh_g = 0, ov_g_p = 0, max_in_flight = 0, n_started = 0, bp_mesh = 0, bp_act = 0;
    int pool_with_result = 0;  // pooling busy while a mesh result waited on an idle activation stage, so activation could have overlapped it
    int stream_id_base;  // sets started before the stream, which consumed ids
    wire mesh_computing = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.mesh_busy;  // a set between staging and a written result
    // A finished set waits for a result bank: the mesh is held up by the consumer.
    wire mesh_blocked = dut.G_SET.G_COPY[0].pipe.systolic_array_inst.arrays_final && dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reducers_ready &&
                        !dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reduce_start;
    int withheld = 0;  // cycles the full entry kept a free staging bank from the host

    initial forever begin
      @(negedge clk_i);
      if (stream_on) begin
        if (mesh_computing && gpnae_busy_tb) ov_mesh_g++;
        if (gpnae_busy_tb && maxpool_busy_tb) ov_g_p++;
        // A finished result sits in the mesh, not being pushed, while pooling runs and the activation stage has neither taken it nor granted its beats.
        if (maxpool_busy_tb && res_ready && !dut.G_SET.G_COPY[0].pipe.wide_rd_valid && int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && !dut.G_SET.G_COPY[0].pipe.l3_armed) pool_with_result++;
        if (mesh_blocked) bp_mesh++;  // a finished set waits because no result bank is free
        if (int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && res_ready && dut.G_SET.G_COPY[0].pipe.act_full[dut.G_SET.G_COPY[0].pipe.act_wr])
          bp_act++;  // a mesh result waits because both activation banks are full
        if (n_started - stream_bounds.size() > max_in_flight)
          max_in_flight = n_started - stream_bounds.size();
      end
    end

    // int8: set id, activation bank, bypass and add flags of each beat inside the requantize pipeline (and add_lanes), oldest first.
    int rq_id[$], rq_bank[$], rq_byp[$], rq_add[$];
    int add_drain_hold = 0;  // cycles an add set's drain held the next grant, which it does by design
    int ov_rq = 0, rq_wait = 0;
    int ov_lane = 0, rq_hold = 0, rq_null_wait = 0, rq_null_pass = 0, rd_in = 0;  // rd_in: wide reads of the stage's set so far
    wire next_ready = res_ready && dut.G_SET.G_COPY[0].pipe.mesh_sets != 0 && !dut.G_SET.G_COPY[0].pipe.set_accum[dut.G_SET.G_COPY[0].pipe.g_next_id];  // the next set's result is ready to push
    initial forever begin
      @(negedge clk_i);
      if (EXP_W == 0) begin
        if (!rstn_i) begin
          rq_id.delete();
          rq_bank.delete();
          rq_byp.delete();
          rq_add.delete();
          rd_in = 0;
        end else begin
          if (stream_on && rq_id.size() != 0) begin
            // Overlap: set k's beats still in the requantize pipeline while the stage holds set k+1 or has granted its beats.
            if ((int'(dut.G_SET.G_COPY[0].pipe.g_state) != 0 && rq_id[0] != int'(dut.G_SET.G_COPY[0].pipe.g_set_id)) || (int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && dut.G_SET.G_COPY[0].pipe.l3_armed)) begin
              ov_rq++;
              if (rq_byp[0] != 0 && int'(dut.G_SET.G_COPY[0].pipe.g_state) != 0 && !dut.G_SET.G_COPY[0].pipe.act_bypass) ov_lane++;  // a ReLU or linear beat drains while a lane set holds the stage
            end
            // A ReLU or linear set's beats are all in, the next result is ready with a free bank, yet no beats granted: the drain holds it.
            else if (rq_add[0] == 0 && rq_byp[0] != 0 && int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && !dut.G_SET.G_COPY[0].pipe.l3_armed && !dut.G_SET.G_COPY[0].pipe.l3_grant && next_ready &&
                     !dut.G_SET.G_COPY[0].pipe.act_full[rq_bank[0] == 0])
              rq_wait++;
            else if (rq_add[0] != 0 && int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && !dut.G_SET.G_COPY[0].pipe.l3_armed && !dut.G_SET.G_COPY[0].pipe.l3_grant && next_ready && !dut.G_SET.G_COPY[0].pipe.act_full[rq_bank[0] == 0])
              add_drain_hold++;  // an add drain holds the next grant by design
            // The draining ReLU or linear set still holds the stage, not leaving this cycle, though the next set could start.
            if (rq_add[0] == 0 && int'(dut.G_SET.G_COPY[0].pipe.g_state) != 0 && rq_id[0] == int'(dut.G_SET.G_COPY[0].pipe.g_set_id) && rq_byp[0] != 0 && dut.G_SET.G_COPY[0].pipe.act_bypass &&
                rd_in == SRAM_DEPTH / NUM_LANES && !dut.G_SET.G_COPY[0].pipe.g_done && next_ready && !dut.G_SET.G_COPY[0].pipe.act_full[!dut.G_SET.G_COPY[0].pipe.act_wr])
              rq_hold++;
            // A partial set that could pass the stage but for the drain, and one that passed during it.
            if (int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0 && dut.G_SET.G_COPY[0].pipe.mesh_sets != 0 && dut.G_SET.G_COPY[0].pipe.set_accum[dut.G_SET.G_COPY[0].pipe.g_next_id]) rq_null_wait++;
            if (dut.G_SET.G_COPY[0].pipe.g_null_done) rq_null_pass++;
          end
          if (int'(dut.G_SET.G_COPY[0].pipe.g_state) == 0) rd_in = 0;
          else if (dut.G_SET.G_COPY[0].pipe.wide_rd_valid) rd_in++;
          if (dut.G_SET.G_COPY[0].pipe.fill_v && dut.G_SET.G_COPY[0].pipe.fill_add) begin  // add_lanes' beats, per add set, and any that came through it for a set not flagged add
            res_add_beats++;
            if (rq_add.size() == 0 || rq_add[0] == 0) res_add_stray++;
            else res_add_set[tr_k[rq_id[0]]]++;
          end
          if (dut.G_SET.G_COPY[0].pipe.fill_v && rq_id.size() == 0) begin  // a fill beat that no beat entering requantize accounts for
            failed++;
            $display("  [FAIL] a fill beat with no beat in the requantize pipeline or add_lanes");
          end
          if (dut.G_SET.G_COPY[0].pipe.fill_v && rq_id.size() != 0) begin
            void'(rq_id.pop_front());
            void'(rq_bank.pop_front());
            void'(rq_byp.pop_front());
            void'(rq_add.pop_front());
          end
          if (dut.G_SET.G_COPY[0].pipe.wide_rd_valid && !dut.G_SET.G_COPY[0].pipe.g_hold) begin  // hold beats go to hold_mem and never reach fill_v
            rq_id.push_back(int'(dut.G_SET.G_COPY[0].pipe.g_set_id));
            rq_bank.push_back(int'(dut.G_SET.G_COPY[0].pipe.act_wr));
            rq_byp.push_back(int'(dut.G_SET.G_COPY[0].pipe.act_bypass));
            rq_add.push_back(int'(dut.G_SET.G_COPY[0].pipe.g_add));
          end
        end
      end
    end

    // int8 residual (spec Risk 3): hold sets done, add_lanes' fill beats, and the same-cycle crosses of a hold set's done with an L3 grant and with a bank done.
    int res_holds = 0, res_add_beats = 0, res_add_stray = 0, cross_rel_grant = 0, cross_hold_bank = 0;
    int res_add_set[int];  // add_lanes' fill beats per add set of the pass, by the set's number
    initial forever begin
      @(negedge clk_i);
      if (EXP_W == 0 && stream_on && rstn_i) begin
        res_holds += int'(dut.G_SET.G_COPY[0].pipe.hold_done);
        cross_rel_grant += int'(dut.G_SET.G_COPY[0].pipe.hold_done && dut.G_SET.G_COPY[0].pipe.l3_grant);
        cross_hold_bank += int'(dut.G_SET.G_COPY[0].pipe.hold_done && dut.G_SET.G_COPY[0].pipe.bank_done);
      end
    end

    task automatic verify_slice(input int k, input logic [DATA_WIDTH-1:0] exp_q[$]);
      automatic logic [DATA_WIDTH-1:0] bnd_q[$];
      automatic int lo = (k == 0) ? 0 : stream_bounds[k-1];
      automatic int n_act = stream_bounds[k] - lo;
      automatic int errs = 0;
      logic [DATA_WIDTH-1:0] av;
      string info;
      read_mem_file($sformatf("bound_output_%0d.mem", k), bnd_q);
      if (stream_ids[k] != ((stream_id_base + k) % (1 << ID_W))) begin
        failed++;
        $display("  [FAIL] Stream set %0d completed as set id %0d, expected %0d", k, stream_ids[k],
                 (stream_id_base + k) % (1 << ID_W));
      end
      if (n_act != exp_q.size()) begin
        failed++;
        $display("  [FAIL] Stream set %0d produced %0d outputs, expected %0d", k, n_act, exp_q.size());
      end
      for (int i = 0; i < exp_q.size(); i++) begin
        total_elements++;
        if (i >= n_act) begin
          failed++;
          errs++;
          $display("  [FAIL] Stream set %0d [%0d] exp=0x%h act=MISSING", k, i, exp_q[i]);
        end else begin
          av = stream_results[lo+i];
          if (av === exp_q[i]) exact_passed++;
          else if (check_tolerance(exp_q[i], av, (i < bnd_q.size()) ? bnd_q[i] : '0, info)) tol_passed++;
          else begin
            failed++;
            errs++;
            $display("  [FAIL] Stream set %0d [%0d] exp=0x%h act=0x%h | %s", k, i, exp_q[i], av, info);
          end
        end
      end
      $display("  [Stream] set %0d: %0d outputs, %0d mismatches", k, n_act, errs);
    endtask

    // overrun: before set SETS_IN_FLIGHT the host waits until every admitted set is in flight and shows it gets no credit; close_fill: the last set closes the cache fill.
    // hold (Review Focus 3): from the first words of set hold_set, every L9 credit is withheld for HOLD_CYC cycles; the outputs must equal the first plain pass's.
    // hold_set leaves SETS_IN_FLIGHT+4 sets to put after it where k_num_sets allows (else set 0), so the buffers fill and the host is blocked.
    // The host must end up blocked when it has more sets left than the entry admits: by staging credits once the banks are full, or by the entry first (fewer credits, or partial sums that hold no result bank).
    // HOLD_CYC: at least 500, and SETS_IN_FLIGHT + 2 sets at the slower of the host's put and the mesh's set interval max(T+2, K, T^2) (N = 64: 1156 at T = 8, 17408 at T = 32).
    localparam int HOST_SET_CYC = (N * N + HOST_WORDS - 1) / HOST_WORDS + 4;
    localparam int MESH_K = (COLLAPSE_K != 0) ? N : TILE_SIZE;  // the depth each array multiplies
    localparam int MESH_SET_CYC = (TILE_SIZE * TILE_SIZE > MESH_K) ? ((TILE_SIZE * TILE_SIZE > TILE_SIZE + 2) ? TILE_SIZE * TILE_SIZE : TILE_SIZE + 2)
                                                                   : ((MESH_K > TILE_SIZE + 2) ? MESH_K : TILE_SIZE + 2);
    localparam int SET_CYC = (MESH_SET_CYC > HOST_SET_CYC) ? MESH_SET_CYC : HOST_SET_CYC;
    localparam int HOLD_CYC = ((SETS_IN_FLIGHT + 2) * SET_CYC > 500) ? (SETS_IN_FLIGHT + 2) * SET_CYC : 500;
    logic [DATA_WIDTH-1:0] ref_results[$];  // the first plain pass's words and set boundaries
    int ref_bounds[$];
    task automatic stream_all_sets(input int id_base, input bit overrun, input bit close_fill, input bit hold = 0);
      automatic longint t0 = $time;
      automatic int hold_stalled = 0, hold_completions = 0, hold_full = 0, hold_inflight = 0, hold_blocked = 0, hold_capped = 0;
      automatic int hold_put0 = 0, hold_in0 = 0, hold_room = 0;  // sets put and in flight at the hold's start; sets the entry admits after it
      automatic int puts_s0 = 0;  // tb_puts at the pass's start
      automatic int hold_extra = 0;  // cycles the hold ran on past HOLD_CYC until the host was blocked
      automatic bit hold_end_blocked = 0;  // the host was blocked with a set to put as the hold ended
      $display("\n[STAGE] STREAMING: %0d sets through overlapped stages%s%s", k_num_sets,
               overrun ? $sformatf(", the host waiting for a credit on set %0d with every admitted set in flight", SETS_IN_FLIGHT) : "",
               hold ? $sformatf(", %0d L9 slot per lane and every L9 credit withheld %0d cycles from set %0d's first words", out_adv, HOLD_CYC, hold_set)
                    : "");
      stream_results.delete();
      stream_bounds.delete();
      stream_ids.delete();
      put_sets = 0;
      n_started = 0;
      stream_id_base = id_base;
      ov_mesh_g = 0;
      ov_g_p = 0;
      pool_with_result = 0;
      max_in_flight = 0;
      bp_mesh = 0;
      bp_act = 0;
      ov_rq = 0;
      starved = 0;
      hold_on = hold;
      rq_wait = 0;
      ov_lane = 0;
      rq_hold = 0;
      rq_null_wait = 0;
      rq_null_pass = 0;
      add_drain_hold = 0;
      res_holds = 0;
      res_add_beats = 0;
      res_add_stray = 0;
      res_add_set.delete();
      cross_rel_grant = 0;
      cross_hold_bank = 0;
      while (pipeline_complete_o) @(posedge clk_i);  // the previous pass's pulse is not a set boundary
      @(posedge clk_i);
      stream_on = 1;
      puts_s0 = tb_puts;
      tr_streams++;
      tr_pass = $sformatf("S%0d", tr_streams);
      tr_seq++;
      tr_on = 1;
`ifdef PERF
      $display("PERF %0d PASS %0d %0d", int'($time / 10), id_base, overrun);
`endif
      fork
        begin
          fork
            begin : producer
              for (int k = 0; k < k_num_sets; k++) begin
                read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
                read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
                dropout_seed_i = set_seed(k);
                accumulate_i = is_partial(k);
                activation_function_i = act_of(k);
                apply_bias(k);
                apply_add(k);
                apply_requant(k);
                apply_pack(k);
                apply_weight(k);
                wc_last_i = close_fill && (k_weight_cache != 0) && (k == k_num_sets - 1);
                tr_cur_k = k;
`ifdef PERF
                if (!overrun) wait_credit();
                $display("PERF %0d HOST_LOAD %0d", int'($time / 10), k);
`endif
                if (overrun && k == SETS_IN_FLIGHT) begin
                  automatic int waited = 0, held = 0, free_bank = 0;
                  // The host cannot put without a credit: wait until every admitted set is in flight, then count the cycles it gets none.
                  while (!(host_cnt == 0 && g_pipeline.dut.G_SET.G_COPY[0].pipe.entry_full) && waited < 2000) begin
                    @(posedge clk_i);
                    waited++;
                  end
                  if (host_cnt == 0 && g_pipeline.dut.G_SET.G_COPY[0].pipe.entry_full) begin
                    while (host_cnt == 0) begin
                      held++;
                      if (g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt > g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out) free_bank++;  // a staging bank the full entry keeps from the host
                      @(posedge clk_i);
                    end
                    $display("  [Stream] entry full: the host held no credit for %0d cycles, %0d of them with a staging bank free", held,
                             free_bank);
                  end else $display("  [Stream] entry never full: the pipeline drains faster than the host loads");
                end
                wait_credit();
                load_inputs();
`ifdef PERF
                $display("PERF %0d HOST_START %0d", int'($time / 10), k);
`endif
                host_put();
                n_started++;
                @(posedge clk_i);  // the spent credit is counted on this edge
              end
            end
            begin : verifier
              logic [DATA_WIDTH-1:0] exp_q[$];
              for (int k = 0; k < k_num_sets; k++) begin
                while (stream_bounds.size() <= k) @(posedge clk_i);
                read_mem_file($sformatf("expected_output_%0d.mem", k), exp_q);
                verify_slice(k, exp_q);
              end
            end
            begin : holder
              if (hold) begin
                automatic bit started = 0;
                while (!started && stream_bounds.size() < k_num_sets) begin
                  @(negedge clk_i);
                  started = (put_sets > hold_set);  // a put, not lane_q: lane_q is empty again once a set's last words come with its completion
                end
                if (!started) begin
                  failed++;
                  $display("  [FAIL] Hold: set %0d never started its output", hold_set);
                end else begin
                  out_hold  = 1;
                  hold_puts = 0;
                  hold_put0 = tb_puts - puts_s0;  // the holder counts puts as tb_inflight does: n_started lags a put in progress by a cycle
                  hold_in0  = tb_inflight;
                  repeat (HOLD_CYC) begin
                    @(negedge clk_i);
                    if (out_put == '0) hold_stalled++;
                    if (pipeline_complete_o) hold_completions++;
                    if (host_cnt == 0 && g_pipeline.dut.G_SET.G_COPY[0].pipe.entry_full) hold_full++;
                    if (host_cnt == 0 && !g_pipeline.dut.G_SET.G_COPY[0].pipe.entry_full && tb_puts - puts_s0 < k_num_sets) hold_blocked++;  // a set to put, no staging credit
                    if (host_cnt == 0 && g_pipeline.dut.G_SET.G_COPY[0].pipe.entry_full && tb_puts - puts_s0 < k_num_sets) hold_capped++;  // a set to put, the entry at SETS_IN_FLIGHT
                    if (tb_inflight > hold_inflight) hold_inflight = tb_inflight;
                  end
                  // With the output frozen no real set completes, so the entry admits SETS_IN_FLIGHT less those in flight at the start (plus any completion): more sets left must block the host.
                  hold_room = SETS_IN_FLIGHT - hold_in0 + hold_completions;
                  // Required, the host must still be blocked as the hold ends; one between a staging credit and its put gets one set's load to block again.
                  while (k_num_sets - hold_put0 > hold_room && !(host_cnt == 0 && tb_puts - puts_s0 < k_num_sets) && hold_extra < HOST_SET_CYC + 8) begin
                    @(negedge clk_i);
                    if (pipeline_complete_o) hold_completions++;
                    hold_extra++;
                  end
                  hold_end_blocked = (host_cnt == 0 && tb_puts - puts_s0 < k_num_sets);
                  out_hold = 0;
                  $display("  [Hold] every L9 credit withheld %0d cycles from set %0d's first words: %0d words put, %0d cycles with no word, %0d completions, %0d sets most in flight, %0d cycles with the entry full, the host blocked with a set to put %0d cycles by staging credits and %0d by the entry, %0d sets put (%0d at the start, %0d in flight); %0d cycles more to the end, blocked at the end %0b",
                           HOLD_CYC, hold_set, hold_puts, hold_stalled, hold_completions, hold_inflight, hold_full, hold_blocked, hold_capped, n_started,
                           hold_put0, hold_in0, hold_extra, hold_end_blocked);
                  if (k_num_sets - hold_put0 <= hold_room)
                    $display("  [Hold] host blocking not reachable: %0d sets left to put at the hold's start, the entry admits %0d more (SETS_IN_FLIGHT %0d, %0d in flight, %0d completions)",
                             k_num_sets - hold_put0, hold_room, SETS_IN_FLIGHT, hold_in0, hold_completions);
                  else if (hold_blocked + hold_capped == 0) begin
                    failed++;
                    $display("  [FAIL] Hold: the host was never blocked with a set to put, so the stall never reached it");
                  end else if (!hold_end_blocked) begin
                    failed++;
                    $display("  [FAIL] Hold: the host was not blocked when the hold ended (%0d cycles blocked in all), so the stall did not hold it back",
                             hold_blocked + hold_capped);
                  end
                  if (hold_stalled < HOLD_CYC * 4 / 5 || hold_puts > out_adv * NUM_LANES) begin
                    failed++;
                    $display("  [FAIL] Hold: the output did not stall (%0d of %0d cycles with no word, %0d words on %0d credits per lane)", hold_stalled,
                             HOLD_CYC, hold_puts, out_adv);
                  end
                end
              end
            end
          join
        end
        begin : watchdog
          repeat (TIMEOUT_CYCLES) @(posedge clk_i);
          $display("[FATAL] Timeout in the streaming pass: %0d of %0d sets completed",
                   stream_bounds.size(), k_num_sets);
          $finish;
        end
      join_any
      disable fork;
      stream_on = 0;
      tr_on = 0;
      out_hold = 0;
      // An add set's last beat cannot meet the next accept: rq_room holds the grant until add_lanes drains (a_add_last_no_accept).
      if (k_residual != 0) begin
        automatic int whole = 0;  // add sets whose add_lanes beats are exactly PER_LANE
        foreach (res_add_set[k]) whole += int'(res_add_set[k] == PER_LANE);
        $display("[HOLD/ADD] hold sets done %0d, add beats %0d, add sets of %0d beats %0d, add beats of other sets %0d, hold done with an L3 grant %0d, hold done with a bank done %0d, grants held for an add drain %0d",
                 res_holds, res_add_beats, PER_LANE, whole, res_add_stray, cross_rel_grant, cross_hold_bank, add_drain_hold);
        if (g_pipeline.dut.G_SET.G_COPY[0].pipe.hm_any) begin  // a hold set left without its add set (Task 2.7 review M5)
          failed++;
          $display("  [FAIL] Stream: hold_mem still holds unread hold beats after the last set: a hold set without its add set");
        end
      end
      // $time is in the 1 ns timeunit, so a 10 ns clock is 10 units per cycle.
      $display("  [%s] %0d sets in %0d cycles", hold ? "Hold" : "Stream", k_num_sets, ($time - t0) / 10);
      $display("  [Stream] lane-cycles with no L9 credit held: %0d", starved);
      hold_on = 0;
      if (!overrun && !hold && ref_bounds.size() == 0) begin
        ref_results = stream_results;
        ref_bounds  = stream_bounds;
      end
      if (hold) begin  // Review Focus 3: nothing lost, duplicated or reordered, and every count back home once drained
        automatic int first = -1, waited = 0;
        string why;
        for (int i = 0; i < ref_results.size() && i < stream_results.size(); i++)
          if (first < 0 && stream_results[i] !== ref_results[i]) first = i;
        if (stream_results.size() != ref_results.size() || stream_bounds != ref_bounds || first >= 0) begin
          failed++;
          $display("  [FAIL] Hold: %0d words in %0d sets against the plain pass's %0d in %0d, first difference at word %0d", stream_results.size(),
                   stream_bounds.size(), ref_results.size(), ref_bounds.size(), first);
        end else $display("  [Hold] outputs identical to the plain pass: %0d words, the same %0d set boundaries", stream_results.size(),
                          stream_bounds.size());
        while (!drained && waited < 2000) begin
          @(posedge clk_i);
          waited++;
        end
        repeat (40) @(posedge clk_i);
        if (!drained || tb_inflight !== 0 || host_cnt !== HOST_SLOTS || g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out !== HOST_SLOTS || !lanes_home(why)) begin
          failed++;
          $display("  [FAIL] Hold: not home after the drain: drained %0b, in flight %0d, host %0d (%0d), sets %0d, granted %0d; %s", drained,
                   tb_inflight, host_cnt, HOST_SLOTS, g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out, why);
        end else $display("  [Hold] drained: every completion counted (in flight 0), host credits %0d, %s", host_cnt, why);
      end
      $display("  [Stream] mesh computing while activation busy: %0d cycles", ov_mesh_g);
      $display("  [Stream] activation and pooling busy together: %0d cycles", ov_g_p);
      $display("  [Stream] most sets in flight: %0d", max_in_flight);
      $display("  [Stream] mesh stalled on full result banks: %0d cycles", bp_mesh);
      $display("  [Stream] activation stalled on full activation banks: %0d cycles", bp_act);
      if (hold) $display("  [Stream] overlap requirements not applied: the hold pass's output stalls by design");
      else if (ov_mesh_g == 0 && SETS_IN_FLIGHT > 1) begin
        failed++;
        $display("  [FAIL] Overlap: the mesh never computed while the activation stage held a set");
      end else if (ov_mesh_g == 0)
        $display("  [Stream] mesh/activation overlap not reachable: SETS_IN_FLIGHT 1 admits one set at a time");
      // Only a failure if a mesh result was waiting while pooling ran; with the mesh slowest there is nothing to overlap.
      if (!hold && ov_g_p == 0 && pool_with_result > 0) begin
        failed++;
        $display("  [FAIL] Overlap: a mesh result waited %0d cycles while pooling ran, yet activation never overlapped it",
                 pool_with_result);
      end else if (ov_g_p == 0)
        $display("  [Stream] activation/pooling overlap not reachable: no mesh result was ready while pooling ran");
      if (EXP_W == 0) begin
        $display("  [Stream] requantize pipeline held set k while set k+1 was in the activation stage: %0d cycles", ov_rq);
        $display("  [Stream] of those, a ReLU or linear set draining while a GPNAE lane set held the stage: %0d cycles", ov_lane);
        if (!hold && rq_wait > 0 && ov_rq == 0) begin
          failed++;
          $display("  [FAIL] Overlap: a mesh result waited %0d cycles on a ReLU or linear set's requantize drain, yet no drain overlapped the next set",
                   rq_wait);
        end else if (rq_wait == 0)
          $display("  [Stream] requantize drain overlap not reachable: no mesh result waited on a ReLU or linear set's drain");
        $display("  [Stream] stage held by a ReLU or linear set with every read in, the next result ready and a bank free: %0d cycles",
                 rq_hold);
        if (!hold && rq_hold > 0) begin
          failed++;
          $display("  [FAIL] Overlap: a ReLU or linear set held the activation stage for %0d cycles of its requantize drain", rq_hold);
        end
        $display("  [Stream] partial set held back by the requantize drain: %0d cycles", rq_null_wait);
        if (rq_null_pass > 0) begin
          failed++;
          $display("  [FAIL] A partial set passed the activation stage in %0d cycles while the requantize pipeline held beats",
                   rq_null_pass);
        end
      end
      if (max_in_flight > SETS_IN_FLIGHT) begin
        failed++;
        $display("  [FAIL] %0d sets in flight, the credit limit is %0d", max_in_flight, SETS_IN_FLIGHT);
      end
      if (over_admit > 0) begin
        failed++;
        $display("  [FAIL] %0d cycles with sets in flight plus credits held above %0d", over_admit, SETS_IN_FLIGHT);
      end
    endtask

    // ── Reset with sets in flight ─────────────────────────────────────────
    task automatic reset_mid_stream();
      automatic int waited = 0;
      automatic int stray = 0;
      string why;
      $display("\n[STAGE] RESET MID-STREAM");
      fork
        begin
          for (int k = 0; k < k_num_sets; k++) begin
            read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
            read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
            accumulate_i = is_partial(k);
            activation_function_i = act_of(k);
            apply_bias(k);
            apply_add(k);
            apply_requant(k);
            apply_pack(k);
            apply_weight(k);
            wait_credit();
            load_inputs();
            host_put();
            @(posedge clk_i);
          end
        end
      join_none
      while (!(gpnae_busy_tb && g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out >= HOST_SLOTS) && waited < TIMEOUT_CYCLES) begin
        @(posedge clk_i);
        waited++;
      end
      if (!(gpnae_busy_tb && g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out >= HOST_SLOTS)) begin
        failed++;
        $display("  [FAIL] Never reached two sets in flight before the reset");
      end
      disable fork;
      $display("  [Reset] resetting with %0d sets in flight, host credits %0d, stages {g,p}=%0d", g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out, host_cnt, dut_stage);
      reset();
      activation_function_i = k_activation_code[CONTROL_WIDTH-1:0];
      training_mode_i       = k_training_mode[0];
      repeat (2000) begin
        @(posedge clk_i);
        if (pipeline_complete_o || (|out_put)) stray++;
      end
      if (stray != 0) begin
        failed++;
        $display("  [FAIL] %0d cycles of output after reset with nothing started", stray);
      end
      // Review Focus 1: every consumer advertised its slots again and every producer counted them from 0.
      if (host_cnt !== HOST_SLOTS || g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt !== 2 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out !== HOST_SLOTS || g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out !== 0 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt !== 2 || !g_pipeline.dut.G_SET.G_COPY[0].pipe.l3_armed ||
          g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt !== PER_LANE || !wc_cnt[0] || !wc_cnt[1] || dut_stage !== 0) begin
        failed++;
        $display("  [FAIL] Not idle after reset: host %0d (%0d), staging %0d (2), granted %0d (%0d), sets %0d (0), act banks %0d (2), results %0d (%0d), regions %0d %0d (1 1), stages %0d",
                 host_cnt, HOST_SLOTS, g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.l0_out, HOST_SLOTS, g_pipeline.dut.G_SET.G_COPY[0].pipe.sets_out, g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt, PER_LANE,
                 wc_cnt[0], wc_cnt[1], dut_stage);
      end else if (!lanes_home(why)) begin
        failed++;
        $display("  [FAIL] Lane links not re-advertised after reset: %s", why);
      end else $display("  [Reset] idle after reset: every link re-advertised (host %0d, staging 2, act banks 2, results %0d, regions 1 1; %s), no stray output",
                        HOST_SLOTS, PER_LANE, why);
      write_cache();  // the reset closed the cache fill: open it again
      stream_all_sets(0, 0, 0);
      // Review Focus 3: a second reset re-advertises one L9 slot per lane, so the hold pass stalls mid-set.
      out_adv = 1;
      reset();
      activation_function_i = k_activation_code[CONTROL_WIDTH-1:0];
      training_mode_i       = k_training_mode[0];
      repeat (100) @(posedge clk_i);
      if (!lanes_home(why)) begin
        failed++;
        $display("  [FAIL] Lane links not re-advertised after the second reset: %s", why);
      end else $display("  [Reset] idle after the second reset: %s", why);
      write_cache();
      stream_all_sets(0, 0, k_packed != 0, 1);  // with the HOLD_CYC hold; the accumulate pass closes the fill when it runs
    endtask

    // ── Review Focus 4: four partial sets of zeros, then set k_accum_passes-1's own passes ─────
    // Only the last set has a result: one L3 set of beats and one L6 bank; the partial sets spend no L3 or L6 credit.
    int acc_beats = 0, acc_banks = 0;
    always_ff @(posedge clk_i) begin
      if (acc_on) begin  // its words and boundaries come from the L9 consumer
        if (dut.G_SET.G_COPY[0].pipe.wide_rd_valid) acc_beats <= acc_beats + 1;
        if (dut.G_SET.G_COPY[0].pipe.bank_done) acc_banks <= acc_banks + 1;
      end
    end

    task automatic accum_null_pass();
      automatic int G = k_accum_passes - 1;  // the first set the stream activates; its golden is the sum of sets 0..G
      automatic int n = 4 + k_accum_passes, waited = 0, errs = 0, n_exact = 0;
      automatic logic [DATA_WIDTH-1:0] exp_q[$], bnd_q[$];
      string info;
      $display("\n[STAGE] ACCUMULATE: 4 partial sets of zeros, then sets 0..%0d as one sum; checkers bound", G);
      if (k_packed != 0) begin
        $display("  [Accum] skipped: a packed set cannot be a partial sum");
        return;
      end
      while (pipeline_complete_o) @(posedge clk_i);
      acc_results.delete();
      acc_bounds.delete();
      acc_beats = 0;
      acc_banks = 0;
      acc_on = 1;
      tr_pass = "A";
      tr_seq++;
      tr_on = 1;
      for (int j = 0; j < n; j++) begin
        automatic int k = (j < 4) ? 0 : j - 4;  // the stream set whose rows this pass sends
        read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
        read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
        if (j < 4) foreach (west_data_queue[i]) west_data_queue[i] = '0;
        accumulate_i = (j < n - 1) || $test$plusargs("acc_open");  // on purpose with +acc_open: the sum never closes
        activation_function_i = act_of(G);
        dropout_seed_i = set_seed(G);
        apply_bias(0);  // the sum's bias rides its first pass only
        apply_add(0, 1);
        if (j != 0) begin
          bias_valid_i = 1'b0;
          bias_i = '0;
        end
        apply_requant(G);
        apply_pack(0);
        apply_weight(k);
        wc_last_i = (k_weight_cache != 0) && (j == n - 1);
        tr_cur_k = (j == n - 1) ? G : -1;  // only the last pass has a result: set G's, the sum of group 0
        wait_credit();
        load_inputs();
        host_put();
        @(posedge clk_i);
      end
      while (acc_bounds.size() < n && waited < TIMEOUT_CYCLES) begin
        @(posedge clk_i);
        waited++;
      end
      waited = 0;
      while (!drained && waited < 1000) begin
        @(posedge clk_i);
        waited++;
      end
      acc_on = 0;
      tr_on = 0;
      if ($test$plusargs("acc_fault") && acc_results.size() > 0) begin  // on purpose: the summed set's first output corrupted
        acc_results[0] = acc_results[0] ^ DATA_WIDTH'(1 << (DATA_WIDTH - 2));
        $display("  [FAULT acc] the accumulate pass's first output corrupted");
      end
      // Narrow floats: regression.py's bit-exact golden of this pass (the zero partials move the mesh's accumulator slots); int8 and fp32: set G's own.
      read_mem_file((EXACT_GOLDEN != 0 && IS_INT == 0) ? "expected_accum.mem" : $sformatf("expected_output_%0d.mem", G), exp_q);
      read_mem_file($sformatf("bound_output_%0d.mem", G), bnd_q);
      if (acc_bounds.size() != n || !drained) begin
        failed++;
        $display("  [FAIL] Accumulate pass: %0d of %0d sets completed, drained=%0b", acc_bounds.size(), n, drained);
      end else begin
        for (int j = 0; j < n - 1; j++)
          if (acc_bounds[j] != 0) begin
            failed++;
            $display("  [FAIL] Accumulate pass: partial set %0d left %0d outputs", j, acc_bounds[j]);
          end
        if (acc_results.size() != exp_q.size()) begin
          failed++;
          $display("  [FAIL] Accumulate pass: %0d outputs, expected %0d", acc_results.size(), exp_q.size());
        end
        // int8 and narrow floats bit-exact (check_tolerance takes only identical bits there); fp32 within its tolerance and bound.
        for (int i = 0; i < exp_q.size() && i < acc_results.size(); i++) begin
          total_elements++;
          if (acc_results[i] === exp_q[i]) begin
            exact_passed++;
            n_exact++;
          end else if (check_tolerance(exp_q[i], acc_results[i], (i < bnd_q.size()) ? bnd_q[i] : '0, info)) tol_passed++;
          else begin
            failed++;
            errs++;
            $display("  [FAIL] Accumulate pass [%0d] exp=0x%h got=0x%h | %s", i, exp_q[i], acc_results[i], info);
          end
        end
      end
      // Credits: one set of beats and one bank spent; drained, every link holds all its slots again.
      if (acc_beats !== PER_LANE || acc_banks !== 1 || g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt !== 2 || !g_pipeline.dut.G_SET.G_COPY[0].pipe.l3_armed ||
          g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt !== PER_LANE || g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt !== 2 || host_cnt !== HOST_SLOTS) begin
        failed++;
        $display("  [FAIL] Accumulate pass credits: beats %0d (%0d), banks %0d (1), act banks held %0d (2), results held %0d (%0d), staging %0d (2), host %0d (%0d)",
                 acc_beats, PER_LANE, acc_banks, g_pipeline.dut.G_SET.G_COPY[0].pipe.l6_cnt, g_pipeline.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.res_cnt, PER_LANE, g_pipeline.dut.G_SET.G_COPY[0].pipe.l1_cnt, host_cnt, HOST_SLOTS);
      end
      $display("  [Accum] %0d sets, %0d with outputs (%0d outputs, %0d exact, %0d mismatches); %0d result beats and %0d activation bank spent",
               acc_bounds.size(), (acc_bounds.size() > 0 && acc_results.size() > 0) ? 1 : 0, acc_results.size(), n_exact, errs, acc_beats,
               acc_banks);
    endtask

    // ── PERF trace: stage transitions per cycle, read by regression.py --action perf ──
`ifdef PERF
    int perf_mesh_st = -1, perf_g_st = -1, perf_p_st = -1, perf_cred = -1, perf_mread = 0;
    int perf_lane_busy = 0, perf_round_cyc = 0;
    int g_eff, entry_free;
    assign g_eff = (dut.G_SET.G_COPY[0].pipe.g_fed || dut.G_SET.G_COPY[0].pipe.wide_rd_valid) ? int'(dut.G_SET.G_COPY[0].pipe.g_state) : 0;  // the stage's state once its set's beats arrive
    assign entry_free = SETS_IN_FLIGHT - int'(dut.G_SET.G_COPY[0].pipe.sets_out) - int'(dut.G_SET.G_COPY[0].pipe.l0_out);  // sets the entry may still admit
    initial forever begin
      @(negedge clk_i);
      if (stream_on) begin
        automatic int c = int'($time / 10);
        automatic int busy = 0;
        // Mesh events, one of each per set and in set order: 2 broadcast start, 3 broadcast end, 4 feed start, 5 reduce start, 7 written.
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.set_launch) begin
          $display("PERF %0d MESH 1", c);
          $display("PERF %0d MESH 2", c);
        end
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bcast_release) $display("PERF %0d MESH 3", c);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.launch) $display("PERF %0d MESH 4", c);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reduce_start) $display("PERF %0d MESH 5", c);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.set_done) $display("PERF %0d MESH 7", c);
        // The stage takes its credits ahead of the result; G counts from the set's first beat, as main's read did.
        if (g_eff != perf_g_st) begin
          $display("PERF %0d G %0d", c, g_eff);
          if (perf_g_st == 3) $display("PERF %0d LANES %0d %0d", c, perf_lane_busy, perf_round_cyc);
          if (perf_g_st == 0) begin
            perf_lane_busy = 0;
            perf_round_cyc = 0;
          end
        end
        if (int'(dut.G_SET.G_COPY[0].pipe.p_state) != perf_p_st) $display("PERF %0d P %0d", c, dut.G_SET.G_COPY[0].pipe.p_state);
        if (entry_free != perf_cred) $display("PERF %0d CREDITS %0d", c, entry_free);
        if (int'(dut.G_SET.G_COPY[0].pipe.wide_rd_valid) != perf_mread) $display("PERF %0d MREAD %0d", c, dut.G_SET.G_COPY[0].pipe.wide_rd_valid);
        if (pipeline_complete_o) $display("PERF %0d DONE %0d", c, done_set_id_o);
        if (g_eff == 3) begin
          for (int i = 0; i < NUM_LANES; i++) if (dut.G_SET.G_COPY[0].pipe.load_finalized[i] && !dut.G_SET.G_COPY[0].pipe.lane_collected[i]) busy++;
          perf_lane_busy += busy;
          perf_round_cyc++;
        end
      end
      perf_g_st    = g_eff;
      perf_p_st    = int'(dut.G_SET.G_COPY[0].pipe.p_state);
      perf_cred    = entry_free;
      perf_mread   = int'(dut.G_SET.G_COPY[0].pipe.wide_rd_valid);
    end
`endif

    // ── Top-level stimulus ────────────────────────────────────────────────
    initial begin
      read_knobs();
      read_add_tables();

      $display("==============================================");
      $display(" SIENNA PIPELINE VERIFICATION");
      $display("==============================================");
      $display(" N=%-0d  TILE_SIZE=%-0d  FIFO_DEPTH=%-0d", N, TILE_SIZE, FIFO_DEPTH);
      $display(" Activation code : %0b", k_activation_code);
      $display(" Dropout         : %s  seed 0x%08h", k_training_mode ? "training" : "inference", k_dropout_seed);
      $display(" Tolerance       : %s  rel<=%.1f%%  abs<=%.4f", TOLERANCE_MODE, REL_TOL * 100.0,
               ABS_TOL);
      $display(" Timeout         : %0d cycles  Heartbeat: %0d cycles", TIMEOUT_CYCLES,
               HEARTBEAT_CYCLES);
      $display("==============================================");

      $display("\n[STAGE] Reading .mem files");
      read_mem_file(WEST_INPUT_FILE, west_data_queue);
      read_mem_file(NORTH_INPUT_FILE, north_data_queue);
      read_mem_file(EXPECTED_OUTPUT_FILE, expected_results);
      read_mem_file("bound_output.mem", bound_results);

      $display("  West=%0d  North=%0d  Expected=%0d words", west_data_queue.size(),
               north_data_queue.size(), expected_results.size());

      reset();
      write_cache();
      apply_weight(0);

      // Clear array right before run so streaming block cleanly builds it
      // actual_results.delete();

      wait_credit();
      load_inputs();
      repeat (5) @(posedge clk_i);

      $display("\n[STAGE] Starting pipeline");
      activation_function_i = act_of(0);
      apply_bias(0);
      apply_add(0, 1);  // set 0 alone: no hold, no add
      apply_requant(0);
      apply_pack(0);
      training_mode_i       = k_training_mode[0];
      dropout_seed_i        = set_seed(0);
      @(posedge clk_i);
      tr_cur_k = 0;  // set 0 alone
      tr_pass = "P0";
      tr_seq++;
      tr_on = 1;
      host_put();
      $display("  start pulse sent @ %0t", $time);
      print_status("immediately after start");
      print_lane_status("immediately after start");

      collect_outputs();
      verify_outputs();
      tr_on = 0;

`ifdef BACK_TO_BACK
      // Run a second, distinct matrix WITHOUT asserting reset. Anything that carries state
      // between matmuls shows up as a different result the second time, and the cycle delta is
      // the real steady-state cost per matrix rather than an estimate.
      begin
        int unsigned t_second_start;
        int unsigned pass1_failed;
        pass1_failed = failed;

        $display("\n[STAGE] BACK TO BACK: second matrix, no reset");
        // A different matrix with its own golden, so a replayed first result cannot pass.
        read_mem_file("matrix_west_1.mem", west_data_queue);
        read_mem_file("matrix_north_1.mem", north_data_queue);
        read_mem_file("expected_b2b.mem", expected_results);  // set 1 alone, not its accumulate group's sum
        read_mem_file("bound_b2b.mem", bound_results);
        dropout_seed_i = set_seed(1);
        activation_function_i = act_of(1);
        apply_bias(1);
        apply_add(1, 1);  // set 1 alone: no hold, no add
        apply_requant(1);
        apply_pack(1);
        apply_weight(1);
        trace_states = 1;
        actual_results.delete();
        total_elements = 0;
        exact_passed   = 0;
        tol_passed     = 0;
        failed         = 0;

        wait_credit();
        load_inputs();
        repeat (5) @(posedge clk_i);
        t_second_start = $time;

        @(posedge clk_i);
        $display("  [B2B] gate before put: state=%0d north_empty=%0b west_empty=%0b credits=%0d",
                 dut_stage, dut.G_SET.G_COPY[0].pipe.north_queue_empty, dut.G_SET.G_COPY[0].pipe.west_queue_empty, host_cnt);
        tr_cur_k = 1;  // set 1 alone
        tr_pass = "B2B";
        tr_seq++;
        tr_on = 1;
        host_put();
        $display("  [B2B] gate at put    : state=%0d north_empty=%0b west_empty=%0b",
                 dut_stage, dut.G_SET.G_COPY[0].pipe.north_queue_empty, dut.G_SET.G_COPY[0].pipe.west_queue_empty);
        @(posedge clk_i);
        $display("  [B2B] state after pulse: %0d", dut_stage);

        collect_outputs();
        verify_outputs();
        tr_on = 0;

        // $time evaluates in the 1ns timeunit (a 10ns clock is 10 units), even though %0t prints ps.
        $display(" BACK_TO_BACK second-pass cycles : %0d", ($time - t_second_start) / 10);
        if (failed == 0 && pass1_failed == 0)
          $display(" BACK_TO_BACK: PASSED (both matrices correct without an intervening reset)");
        else
          $display(" BACK_TO_BACK: FAILED (pass1 %0d, pass2 %0d mismatches)", pass1_failed, failed);
        failed = failed + pass1_failed;
      end
`endif

      begin
        automatic int ids_used = 1;  // the single-set pass
`ifdef BACK_TO_BACK
        ids_used = 2;  // plus the back-to-back pass
`endif
        stream_all_sets(ids_used, 0, 0);
        stream_all_sets(ids_used + k_num_sets, 1, 0);
      end
      reset_mid_stream();
      if (k_residual != 0) $display("[SKIP] accumulate pass: the residual stream's groups come from add_<k>.mem");
      else accum_null_pass();
      // sienna_top's own link checkers judge only when it is drained: it must drain after the last set.
      for (int w = 0; w < 200 + 8 * LINK_STAGES && !dut.G_SET.G_COPY[0].pipe.drained; w++) @(posedge clk_i);
      if (!dut.G_SET.G_COPY[0].pipe.drained) begin
        failed++;
        $display("  [FAIL] sienna_top never drained after the last set (%0d drains in the run): its link checkers never judged it", dut.G_SET.G_COPY[0].pipe.drain_n);
      end else $display("  [Drain] sienna_top drained after the last set, %0d drains in the run", dut.G_SET.G_COPY[0].pipe.drain_n);

      if (k_has_bias != 0 && bias_checked == 0) begin
        failed++;
        $display("  [FAIL] Bias check: a bias test compared no set's bias in the mesh's bias queue");
      end else $display("  [Bias] %0d puts: the mesh's bias queue entry was compared with the bias row widened", bias_checked);

      $display("\n==============================================");
      $display(" RESULT SUMMARY");
      $display("==============================================");
      $display(" Total    : %0d", total_elements);
      $display(" Exact    : %0d", exact_passed);
      $display(" Tol pass : %0d  (rel <= %.1f%%)", tol_passed, REL_TOL * 100.0);
      failed += cc_bad;  // a lane memory that differs from its image fails the run
      $display(" COEF IMAGE : checks %0d  differ %0d", cc_checks, cc_bad);
      $display(" Failed   : %0d", failed);
      $display("----------------------------------------------");
      if (failed == 0) $display(" RESULT: PASSED");
      else $display(" RESULT: FAILED (%0d mismatches)", failed);
      $display("==============================================");

      if (trace_fd) $fclose(trace_fd);
      if (tr_fd) $fclose(tr_fd);
      if (lane_fd) $fclose(lane_fd);

      #100 $finish;
    end

    initial begin
`ifdef ENABLE_TRACE
`ifdef TRACE_FST
      $dumpfile("TB_sienna.fst");
`else
      $dumpfile("TB_sienna.vcd");
`endif
      $dumpvars(0, TB_sienna);
`endif
    end

  end else if (MODE == 1) begin : g_layer
    // Runs one layer on sienna_top's layer contract (SET_PORT = 0): writes the configuration, streams the two inputs from +layer=<file> on the row links (L10)
    // as the layer grants credits, and writes every result taken from the output links (L9) to +out=<file>.
    // Layer file: "L m kb n residual bias act train seed a_rows w_rows packed", then a_rows and w_rows rows of N hex words.
    // packed: a "P sh m0 .. m(N/2-1)" line and seven "E act zp min max mx shx mout shout zout" lines after L (and Q).
    // Requantizing builds (int8): a "Q zp min max mx shx mout shout zout" line after it; after the rows, per column block N biases, N multipliers, N shifts (hex).
    // int8 residual: an "A zp2 m1 m2 shift zpo amin amax" line after Q (zp1 is Q's zp).
    // int8 +dump=<file>: per stage "X k e code" (activation input), "R k e code" (hold beat), "Q k e q1 q2 y" (ADD); k the set in issue order, e its tile element.
    // +out_slots=S (1..64) L9 slots per lane, +out_stall_pct=P and +in_stall_pct=P random stalls; -DTB_OUT_STALL_PCT / -DTB_IN_STALL_PCT set the defaults.
    // FAULT 1: an A row put with no credit; 2: a W row put with no credit; 3: the A link one bit wide too many; 4: a put on the unused set contract's L0 (a_idle_contract).
    // FAULT 5: the sixth write of every +coef / +between load driven with bit 0 flipped, so the coefficient-image check must fail.
    // +mid_reset=C: a first pass is reset C cycles into its stream (once the A producer holds credits); every count must come home with no stray output, then the clean pass runs.
    // int8 +hold_wait: the reset waits, from C, for a cycle at which hold_mem holds unread hold beats; +hold_check: a reset outside that window fails.
    // -DTB_STALE_HOME keeps the A counter's count through that reset, to show the home check firing.
    // +tier=T: the layer's lane tier (cfg_tier_i); +coef=<file>: "aaa dddddddd" (hex address, data) lines written through the lanes' load path
    // after every reset, before the configure, each once coef_ready_o is high.
    // +between=<file> (with +out_first=<file>): the layer runs to completion, <file> is loaded with no reset from the first cycle coef_ready_o
    // is high, the first run's results go to out_first and the layer runs again into +out. +coef_busy=<file>: its "aaa dddddddd" writes, one a cycle from 40 cycles into the stream; the interlock must
    // refuse every one.
    // LINK_STAGES (TB_sienna's parameter): sienna_top's register stages on L0, L1, L3 and L9
    // The build's format, from the define through sienna_fmt_pkg; EXP_W, MAN_W and IS_INT only decode words; protocol keys on the package's HAS_ flags.
    localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH, ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
    localparam int RESULT_DATA_WIDTH = sienna_fmt_pkg::RESULT_DATA_WIDTH;
    localparam int EXP_W = sienna_fmt_pkg::EXPONENT_WIDTH, MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH, IS_INT = (sienna_fmt_pkg::EXPONENT_WIDTH == 0);
    localparam int PACK_ENTRIES = 8, WC_TILES = 128;
    `include "sienna_set_side.svh"
    `include "tb_side_pack.svh"
`define TB_CC_LANES NUM_LANES
`define TB_CC_MEM(i) dut.G_LAYER.pipe.backend_lanes[i].gpnae.COEF
    `include "tb_coef_check.svh"
`undef TB_CC_LANES
`undef TB_CC_MEM

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
    int t_between_done = 0, t_first_wr = 0;  // +between: the cycle done_o was seen and the cycle of the load's first write
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
        .COLLAPSE_K    (COLLAPSE_K),  // the package's mesh mode, a build item as in the pipeline mode
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
    always @(posedge clk_i) if (mid_check && rstn_i && (res_v !== '0 || set_done_o || done_o)) stray++;

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
            $fatal(1, "TB_sienna layer: channel %0d's requantize words %h %h are too wide", c, ep[(3*ei+1)*N+c], ep[(3*ei+2)*N+c]);
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
      if (fc == 0) $fatal(1, "TB_sienna layer: cannot open %s", cf);
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
        coef_data_i = (FAULT == 5 && nw == 5) ? d ^ 32'd1 : d;  // FAULT 5: one word corrupted after packing
        if (nw == 0) t_first_wr = cycle;  // +between reports how soon after done_o the load began
        @(posedge clk_i);
        nw++;
      end
      #1 coef_wr_i = 1'b0;
      $fclose(fc);
      $display("[LAYER] coef: %0d load-path writes from %s", nw, cf);
      if (FAULT == 5) $display("  [FAULT 5] write 5 of %s driven with bit 0 flipped", cf);
      @(posedge clk_i);
      cc_image({cf, ".img"});  // the image the host packed into cf
      cc_check({"load ", cf});
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
      static int failed = 0, osp, mid_reset = 0, msp, tier = 0;
      string between_f, first_f;
      static int between = 0, busy_load = 0, busy_tries = 0, busy_refused = 0, busy_n = 0;
      logic [8:0] busy_a[8];  // +coef_busy: the writes' addresses and data
      logic [31:0] busy_d[8];
      string busy_f;
      static int hold_wait = 0, hold_check = 0;  // +hold_wait: the mid-stream reset waits for the hold window; +hold_check: it must land in it

      void'($value$plusargs("out_slots=%d", out_slots));
      void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
      void'($value$plusargs("in_stall_pct=%d", in_stall_pct));
      void'($value$plusargs("mid_reset=%d", mid_reset));
      void'($value$plusargs("tier=%d", tier));
      between = $value$plusargs("between=%s", between_f);
      if (between && (!$value$plusargs("out_first=%s", first_f) || mid_reset > 0)) $fatal(1, "TB_sienna layer: +between needs +out_first and no +mid_reset");
      busy_load = $value$plusargs("coef_busy=%s", busy_f);
      if (busy_load) begin
        integer fb;
        logic [11:0] a;
        logic [31:0] d;
        fb = $fopen(busy_f, "r");
        if (fb == 0) $fatal(1, "TB_sienna layer: cannot open +coef_busy=%s", busy_f);
        while (busy_n < 8 && $fscanf(fb, "%h %h", a, d) == 2) begin
          busy_a[busy_n] = a[8:0];
          busy_d[busy_n] = d;
          busy_n++;
        end
        $fclose(fb);
        if (busy_n == 0) $fatal(1, "TB_sienna layer: +coef_busy=%s holds no write", busy_f);
      end
      if (tier < 0 || tier > 2) $fatal(1, "TB_sienna layer: +tier=%0d: 0, 1 or 2", tier);
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
          $fatal(1, "TB_sienna layer: the A line's words %0d %0d %0d %0d %0d are out of range", azp2, ash, azpo, aamin, aamax);
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
          if (ev[6] >= (1 << 24) || ev[6] < 0 || ev[7] >= 64 || ev[7] < 0) $fatal(1, "TB_sienna layer: pack entry %0d's SELU words %0d %0d are out of range", e, ev[6], ev[7]);
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
        if (dump_fd == 0) $fatal(1, "TB_sienna layer: cannot open %s", dump_f);
      end

      repeat (5) @(posedge clk_i);
      #1 rstn_i = 1;
      por_n = 1;
      repeat (3) @(posedge clk_i);
      cc_image("coef_default.mem");
      cc_check("reset");

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
        if (qmout >= (1 << 24) || qmout < 0 || qshout >= 64 || qshout < 0) $fatal(1, "TB_sienna layer: SELU words %0d %0d are out of range", qmout, qshout);
        cfg_gp_mout_i = 24'(qmout);
        cfg_gp_shout_i = 6'(qshout);
        cfg_gp_zout_i = 8'(qzout);
        cfg_add_zp2_i = 8'(azp2);
        if (am1 < 0 || am1 >= 64'h1_0000_0000 || am2 < 0 || am2 >= 64'h1_0000_0000) $fatal(1, "TB_sienna layer: the ADD's mantissas %0d %0d are out of range", am1, am2);
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
          if (pass == 1 && busy_load && busy_tries < busy_n && cycle - t0 >= 40) begin  // +coef_busy: the file's writes, one a cycle
            coef_wr_i   = 1'b1;
            coef_addr_i = busy_a[busy_tries];
            coef_data_i = busy_d[busy_tries];
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
        if (pass == 0 && between) begin  // the first run is done: load the next image from the first ready cycle, write the run out, run again
          t_between_done = cycle;
          coef_load_file(between_f);
          $display("  [Between] loaded %s with no reset, its first write %0d cycles after done_o", between_f, t_first_wr - t_between_done);
          repeat (OUT_CAP + 4) @(posedge clk_i);
          fout = $fopen(first_f, "w");
          for (int k = 0; k < bounds.size(); k++) begin
            automatic int lo = (k == 0) ? 0 : bounds[k-1];
            if (bounds[k] == lo) continue;
            $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
            for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
          end
          $fclose(fout);
          $display("  [Between] first run: %0d sets, %0d outputs", bounds.size(), res_q.size());
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
          cc_image("coef_default.mem");
          cc_check("mid-stream reset");
          // Home: no row credit held (the layer grants per set), both staging and region credits back, every L9 slot re-advertised.
          if (a_cnt !== 0 || w_cnt !== 0 || dut.G_LAYER.sched.l0_cnt !== 2 || !dut.G_LAYER.sched.wc_has[0] || !dut.G_LAYER.sched.wc_has[1] || lane_home !== '1 || stray !== 0 ||
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
      if (a_cnt !== 0 || w_cnt !== 0) begin
        failed++;
        $display("[FAIL] the layer is done with row credits still held: A %0d, W %0d", a_cnt, w_cnt);
      end
      osp = out_stall_pct;
      out_stall_pct = 0;
      repeat (OUT_CAP + 4) @(posedge clk_i);
      if (lane_home !== '1) begin
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
        if (busy_tries != busy_n || busy_refused != busy_tries) begin
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
      failed += cc_bad;  // a lane memory that differs from its image fails the layer
      $display("[LAYER] COEF IMAGE : checks %0d  differ %0d", cc_checks, cc_bad);
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

  end else if (MODE == 2) begin : g_sets
    // Streams one network layer's sets from +sets=<file> through sienna_top back to back and writes every set's outputs to +out=<file>.
    // int8: after each set's bias words, 8 + 2N requantize words as in requant_<k>.mem.
    // +out_slots=S (1..64) L9 slots per lane and +out_stall_pct=P random output stalls (-DTB_OUT_STALL_PCT sets the default); +fault_put puts with no credit once.
    // +tier=T: every set's lane tier; +coef=<file>: "aaa dddddddd" (hex address, data) lines written through the lanes' load path after reset.
    // +coef_busy=<file>: those writes, one a cycle from 40 cycles after the first set's start, while sets stream; the interlock must refuse every one.
    // The build's format, from the define through sienna_fmt_pkg; EXP_W, MAN_W and IS_INT only decode words; protocol keys on the package's HAS_ flags.
    localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH, ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
    localparam int RESULT_DATA_WIDTH = sienna_fmt_pkg::RESULT_DATA_WIDTH;
    localparam int EXP_W = sienna_fmt_pkg::EXPONENT_WIDTH, MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH, IS_INT = (sienna_fmt_pkg::EXPONENT_WIDTH == 0);

    localparam ADDR_LINES = $clog2(FIFO_DEPTH);
    localparam int STALL_CYCLES = 50_000;  // this long with no completion is a hang
    localparam int ID_W = $clog2(SETS_IN_FLIGHT + 1);

    // ── DUT I/O ───────────────────────────────────────────────────────────
    logic clk_i, rstn_i;
    logic                     start_pipeline_i;
    logic                     training_mode_i;
    logic                     accumulate_i;
    logic                     bias_valid_i;
    logic [N-1:0][ACC_DATA_WIDTH-1:0] bias_i;  // int32 in int8, where it carries the folded input zero point; fp32 bits in floats
    // int8: the requantize and GPNAE parameters of the set being started (D-2); zero in other formats
    logic [N-1:0][23:0] req_mult_i;
    logic [N-1:0][5:0]  req_shift_i;
    logic [7:0]         req_zp_i, req_min_i, req_max_i, gp_zout_i;
    logic [5:0]         gp_shout_i;
    logic [15:0]        gp_mx_i;
    logic [4:0]         gp_shx_i;
    logic [23:0]        gp_mout_i;
    `include "int8_tb_util.svh"  // unpack_requant
    localparam int WC_TILES = 128;
    logic weight_cached_i, wc_write_enable_i;
    logic [$clog2(WC_TILES)-1:0] weight_tile_i;
    logic [$clog2(WC_TILES*N*N)-1:0] wc_write_addr_i;
    logic [LFSR_WIDTH-1:0]    dropout_seed_i;
    logic [CONTROL_WIDTH-1:0] activation_function_i;
    logic [1:0] tier = '0;  // +tier: the sets' lane tier
    logic coef_wr = 1'b0;  // +coef: the lanes' load path
    logic [8:0] coef_addr = '0;
    logic [31:0] coef_data = '0;
    logic coef_ready;  // sienna_top's load interlock: each write waits for it

    // The host link (L0): start_pipeline_i is the put, pipeline_ready_o says this host holds a staging credit; no cached sets, so no fills (L2).
    localparam int PACK_ENTRIES = 8;
    `include "sienna_set_side.svh"
    `include "tb_side_pack.svh"
`define TB_CC_LANES NUM_LANES
`define TB_CC_MEM(i) dut.G_SET.G_COPY[0].pipe.backend_lanes[i].gpnae.COEF
    `include "tb_coef_check.svh"
`undef TB_CC_LANES
`undef TB_CC_MEM
    credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) host_lnk[1] ();  // L0 of sienna_top's one copy
    // The layer contract's row links (L10), unused at SET_PORT = 1: nothing is ever put on them.
    credit_link_if #(.DATA_W(N * DATA_WIDTH), .CRW($clog2(N + 1))) idle_rows[2] ();
    for (genvar r = 0; r < 2; r++) begin : G_IDLE_ROWS
      assign idle_rows[r].put  = 1'b0;
      assign idle_rows[r].data = '0;
    end
    credit_link_if #(.DATA_W(1), .CRW(1)) wc_lnk[2] ();
    logic [1:0] host_cnt;
    set_side_t side;
    credit_counter #(.MAX(2), .CRW(1)) host_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host_lnk[0].put), .credit_i(host_lnk[0].credit),
                                                .has_credit_o(), .count_o(host_cnt));
    always_comb begin
      side               = '0;
      side.weight_tile   = weight_tile_i;
      side.weight_cached = weight_cached_i;
      side.accumulate    = accumulate_i;
      side.bias_valid    = bias_valid_i;
      side.train         = training_mode_i;
      side.seed          = dropout_seed_i;
      side.act[0]        = activation_function_i;
      side.tier          = tier;
      side.entry         = tb_side_entries({{(PACK_ENTRIES - 1) * TB_ENTRY_W{1'b0}},
                                            tb_entry(req_zp_i, req_min_i, req_max_i, gp_mx_i, gp_shx_i, gp_mout_i, gp_shout_i, gp_zout_i)});
      side.scale         = tb_side_scales(req_mult_i, req_shift_i);
    end
    // The host drives its rows, sideband and put on the falling edge, so the counter, checker and DUT sample them alike; it starts a set only with a credit held.
    logic drain_chk = 1'b0;
    bit fault_put = 0;
    longint cycle = 0;
    assign host_lnk[0].put = start_pipeline_i;
    assign host_lnk[0].data = side;
    credit_link_checker #(.SLOTS(2)) host_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drain_chk), .lnk(host_lnk[0]));
    always @(negedge clk_i)  // just before the edge that samples the put, so it reports ahead of the DUT's own checks
      #4 if (rstn_i && start_pipeline_i && host_cnt == 0) begin
        $display("[FATAL] a host put with no staging credit held at cycle %0d", cycle);
        $finish;
      end
    assign wc_lnk[0].put = 1'b0;
    assign wc_lnk[0].data = 1'b0;
    assign wc_lnk[1].put = 1'b0;
    assign wc_lnk[1].data = 1'b0;

    logic north_write_enable_i, north_write_reset_i;
    logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i;  // operands in the package's format
    logic west_write_enable_i, west_write_reset_i;
    logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i;

    logic [ NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o;
    logic [ NUM_LANES-1:0]                 result_valid_o;
    logic                                  pipeline_complete_o;
    logic                                  pipeline_ready_o;  // this host holds a staging credit
    logic                        [ID_W-1:0] done_set_id_o;
    assign pipeline_ready_o = (host_cnt != 0);
    logic systolic_busy_tb, gpnae_busy_tb, maxpool_busy_tb, dropout_busy_tb;

    initial begin
      clk_i = 0;
      forever #5 clk_i = ~clk_i;
    end

    // L9: a consumer per lane, out_slots slots each, out_stall_pct of its credit returns withheld at random.
    localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX
    credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) out_lnk[NUM_LANES] ();
`ifdef TB_OUT_STALL_PCT
    int out_stall_pct = `TB_OUT_STALL_PCT;
`else
    int out_stall_pct = 0;
`endif
    int out_slots = OUT_CAP;
    logic [NUM_LANES-1:0] lane_home;
    for (genvar l = 0; l < NUM_LANES; l++) begin : G_OUT
      tb_l9_sink #(.DATA_W(DATA_WIDTH), .MAX_SLOTS(OUT_CAP)) snk (.clk_i(clk_i), .rstn_i(rstn_i), .slots_i(out_slots),
                                                                  .stall_pct_i(out_stall_pct), .drain_i(drain_chk),
                                                                  .lnk(out_lnk[l]), .valid_o(result_valid_o[l]), .data_o(final_result_o[l]),
                                                                  .home_o(lane_home[l]));
    end

    sienna_top #(
        .SET_PORT         (1),  // the set contract: this bench is the host on L0
        .NUM_LANES        (NUM_LANES),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (HOST_WORDS),
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
        .FIFO_DEPTH       (FIFO_DEPTH)
    ) dut (
        .clk_i                      (clk_i),
        .rstn_i                     (rstn_i),
        // The layer contract, unused at SET_PORT = 1: tied off
        .cfg_load_i          (1'b0),
        .cfg_m_i             ('0),
        .cfg_kb_i            ('0),
        .cfg_n_i             ('0),
        .cfg_residual_i      (1'b0),
        .cfg_bias_i          (1'b0),
        .cfg_act_i           ('0),
        .cfg_tier_i          ('0),
        .cfg_train_i         (1'b0),
        .cfg_seed_i          ('0),
        .cfg_lane_i          ('0),
        .cfg_add_i           ('0),
        .cfg_pack_shift_i    ('0),
        .cfg_pack_map_i      ('0),
        .cfg_pack_act_i      ('0),
        .cfg_pack_lane_i     ('0),
        .busy_o              (),
        .done_o              (),
        .a_rows              (idle_rows[0]),
        .w_rows              (idle_rows[1]),
        .w_bias_i            ('0),
        .w_req_scale_i       ('0),
        .set_done_o          (),
        .host                       (host_lnk),  // unpacked sets: the sideband's pack fields and entries 1.. stay zero
        .bias_i                     (bias_i),
        .wc_region                  (wc_lnk),
        .wc_write_enable_i          (wc_write_enable_i),
        .wc_write_addr_i            (wc_write_addr_i),
        .north_write_enable_i       (north_write_enable_i),
        .north_write_data_i         (north_write_data_i),
        .north_write_reset_i        (north_write_reset_i),
        .west_write_enable_i        (west_write_enable_i),
        .west_write_data_i          (west_write_data_i),
        .west_write_reset_i         (west_write_reset_i),
        .drained_i                  (1'b0),
        .copy_sel_o                 (),
        .coef_wr_i                  (coef_wr),  // +coef=<file>: the lanes' coefficient memory, loaded after reset
        .coef_addr_i                (coef_addr),
        .coef_data_i                (coef_data),
        .coef_ready_o               (coef_ready),
        .out                        (out_lnk),
        .pipeline_complete_o        (pipeline_complete_o),
        .done_set_id_o              (done_set_id_o),
        .systolic_busy_o            (systolic_busy_tb),
        .gpnae_busy_o               (gpnae_busy_tb),
        .maxpool_busy_o             (maxpool_busy_tb),
        .dropout_busy_o             (dropout_busy_tb)
    );

    always_ff @(posedge clk_i) cycle <= cycle + 1;

    // ── Output capture: per lane, then a set at a time in window order (lane L's j-th word is window j*NUM_LANES + L) ──
    logic [DATA_WIDTH-1:0] lane_q[NUM_LANES][$];
    logic [DATA_WIDTH-1:0] res_q[$];
    int bounds[$];
    int done_ids[$];
    longint last_done = 0;
    longint busy_mesh = 0, busy_act = 0;

    always_ff @(posedge clk_i) begin
      if (rstn_i) begin
        for (int lane = 0; lane < NUM_LANES; lane++) if (result_valid_o[lane]) lane_q[lane].push_back(final_result_o[lane]);
        if (pipeline_complete_o) begin
          automatic bit any;
          any = 1;
          while (any) begin
            any = 0;
            for (int lane = 0; lane < NUM_LANES; lane++)
              if (lane_q[lane].size() != 0) begin
                res_q.push_back(lane_q[lane].pop_front());
                any = 1;
              end
          end
          bounds.push_back(res_q.size());
          done_ids.push_back($isunknown(done_set_id_o) ? -1 : int'(done_set_id_o));  // an unknown id never matches
          last_done <= cycle;
        end
        if (systolic_busy_tb) busy_mesh <= busy_mesh + 1;
        if (gpnae_busy_tb) busy_act <= busy_act + 1;
      end
    end

    // +trace: one line per event, read by a script to find the stage that sets the pace.
    bit trace_on;
    initial trace_on = $test$plusargs("trace");
    always_ff @(posedge clk_i) begin
      if (trace_on && rstn_i) begin
        if (dut.G_SET.G_COPY[0].pipe.host_accept) $display("EV %0d host", cycle);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.set_launch) $display("EV %0d bcast", cycle);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.launch) $display("EV %0d feed", cycle);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.arrays_final && !$past(g_sets.dut.G_SET.G_COPY[0].pipe.systolic_array_inst.arrays_final)) $display("EV %0d final", cycle);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reduce_start) $display("EV %0d reduce", cycle);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.set_done) $display("EV %0d written", cycle);
        if (dut.G_SET.G_COPY[0].pipe.g_accept) $display("EV %0d act", cycle);
        if (dut.G_SET.G_COPY[0].pipe.p_accept || dut.G_SET.G_COPY[0].pipe.p_null) $display("EV %0d pool", cycle);
        if (pipeline_complete_o) $display("EV %0d done", cycle);
        if (!pipeline_ready_o) $display("EV %0d notready c%0d m%0d", cycle, !dut.G_SET.G_COPY[0].pipe.entry_full, dut.G_SET.G_COPY[0].pipe.l1_cnt > dut.G_SET.G_COPY[0].pipe.l0_out);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.arrays_final && !dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reduce_start)
          $display("EV %0d blocked red%0d bank%0d", cycle, dut.G_SET.G_COPY[0].pipe.systolic_array_inst.reducers_ready,
                   dut.G_SET.G_COPY[0].pipe.systolic_array_inst.out_state[dut.G_SET.G_COPY[0].pipe.systolic_array_inst.out_wr]);
        if (dut.G_SET.G_COPY[0].pipe.systolic_array_inst.in_full[dut.G_SET.G_COPY[0].pipe.systolic_array_inst.in_rd] && !dut.G_SET.G_COPY[0].pipe.systolic_array_inst.set_launch &&
            dut.G_SET.G_COPY[0].pipe.systolic_array_inst.bstate == 0)
          $display("EV %0d bwait ready%0d", cycle, dut.G_SET.G_COPY[0].pipe.systolic_array_inst.arrays_load_ready);
      end
    end

    logic [DATA_WIDTH-1:0] west_q[$], north_q[$];

    // A streaming host: HOST_WORDS words of each operand per cycle, the start with the last write, the next set right after.
    task automatic load_and_start();
      for (int i = 0; i < west_q.size(); i += HOST_WORDS) begin
        west_write_enable_i  = 1;
        north_write_enable_i = 1;
        for (int c = 0; c < HOST_WORDS; c++) begin
          west_write_data_i[c]  = (i + c < west_q.size()) ? west_q[i+c] : '0;
          north_write_data_i[c] = (i + c < north_q.size()) ? north_q[i+c] : '0;
        end
        start_pipeline_i = (i + HOST_WORDS >= west_q.size());
        @(negedge clk_i);
      end
      west_write_enable_i  = 0;
      north_write_enable_i = 0;
      start_pipeline_i     = 0;
    endtask

    // The host the pipeline mode models (+host_gaps): an idle cycle after the load, a start cycle, then a cycle for the credit.
    task automatic load_set();
      fork
        begin
          for (int i = 0; i < west_q.size(); i += HOST_WORDS) begin
            west_write_enable_i = 1;
            for (int c = 0; c < HOST_WORDS; c++) west_write_data_i[c] = (i + c < west_q.size()) ? west_q[i+c] : '0;
            @(negedge clk_i);
          end
          west_write_enable_i = 0;
          @(negedge clk_i);
        end
        begin
          for (int i = 0; i < north_q.size(); i += HOST_WORDS) begin
            north_write_enable_i = 1;
            for (int c = 0; c < HOST_WORDS; c++) north_write_data_i[c] = (i + c < north_q.size()) ? north_q[i+c] : '0;
            @(negedge clk_i);
          end
          north_write_enable_i = 0;
          @(negedge clk_i);
        end
      join
    endtask

    // +coef=<file>: the lanes' coefficient memory, one (address, data) write per cycle on the falling edge, before any set.
    task automatic coef_load();
      string cf;
      integer fc, n;
      logic [11:0] a;
      logic [31:0] d;
      if (!$value$plusargs("coef=%s", cf)) return;
      fc = $fopen(cf, "r");
      if (fc == 0) begin
        $display("[FATAL] cannot open %s", cf);
        $finish;
      end
      n = 0;
      while ($fscanf(fc, "%h %h", a, d) == 2) begin
        coef_wr = 1'b0;
        while (!coef_ready) @(negedge clk_i);
        coef_wr   = 1'b1;
        coef_addr = a[8:0];
        coef_data = d;
        @(negedge clk_i);
        n++;
      end
      coef_wr = 1'b0;
      $fclose(fc);
      $display("[MODEL] coef: %0d load-path writes from %s", n, cf);
      @(posedge clk_i);
      #1;
      cc_image({cf, ".img"});  // the image the host packed into cf
      cc_check({"load ", cf});
    endtask

    // +coef_busy=<file>: the writes, one a cycle on the falling edge, each counted refused when coef_ready is low as it is driven.
    int busy_tries = 0, busy_refused = 0, busy_n = 0;
    task automatic coef_busy_writes(input string bf);
      integer fb;
      logic [11:0] a;
      logic [31:0] d;
      fb = $fopen(bf, "r");
      if (fb == 0) begin
        $display("[FATAL] cannot open %s", bf);
        $finish;
      end
      repeat (40) @(negedge clk_i);
      while ($fscanf(fb, "%h %h", a, d) == 2) begin
        busy_n++;
        coef_wr   = 1'b1;
        coef_addr = a[8:0];
        coef_data = d;
        #1;
        if (!coef_ready) busy_refused++;
        busy_tries++;
        @(negedge clk_i);
      end
      coef_wr = 1'b0;
      $fclose(fb);
    endtask

    initial begin
      string sets_f, out_f;
      integer fin, fout, rc;
      int n_sets, acc, act, has_bias, bad_ids, osp;
      bit host_gaps;
      logic [DATA_WIDTH-1:0] w;
      logic [31:0] w32;
      longint t0, waited;

      rstn_i = 0;
      start_pipeline_i = 0;
      training_mode_i = 0;
      accumulate_i = 0;
      bias_valid_i = 0;
      bias_i = '0;
      req_mult_i = '0;
      req_shift_i = '0;
      req_zp_i = '0;
      req_min_i = '0;
      req_max_i = '0;
      gp_mx_i = '0;
      gp_shx_i = '0;
      gp_mout_i = '0;
      gp_shout_i = '0;
      gp_zout_i = '0;
      weight_cached_i = 0;
      weight_tile_i = '0;
      wc_write_enable_i = 0;
      wc_write_addr_i = '0;
      dropout_seed_i = '1;
      activation_function_i = '0;
      north_write_reset_i = 1;
      west_write_reset_i = 1;
      north_write_enable_i = 0;
      west_write_enable_i = 0;
      north_write_data_i = '0;
      west_write_data_i = '0;

      if (!$value$plusargs("sets=%s", sets_f) || !$value$plusargs("out=%s", out_f)) begin
        $display("[MODEL] no +sets= and +out= given, nothing to run");
        $finish;
      end
      fin = $fopen(sets_f, "r");
      if (fin == 0) begin
        $display("[FATAL] cannot open %s", sets_f);
        $finish;
      end
      rc = $fscanf(fin, "%d", n_sets);
      host_gaps = $test$plusargs("host_gaps");
      fault_put = $test$plusargs("fault_put");
      void'($value$plusargs("out_slots=%d", out_slots));
      void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
      if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90) begin
        $display("[FATAL] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct);
        $finish;
      end

      begin
        int t;
        if ($value$plusargs("tier=%d", t)) tier = 2'(t);
      end
      repeat (10) @(posedge clk_i);
      rstn_i = 1;
      north_write_reset_i = 0;
      west_write_reset_i = 0;
      @(negedge clk_i);
      cc_image("coef_default.mem");
      cc_check("reset");
      coef_load();
      if (fault_put) begin  // before the staging credits are advertised
        $display("  [FAULT put] a host put with %0d credits held", host_cnt);
        start_pipeline_i = 1;
        @(negedge clk_i);
        start_pipeline_i = 0;
      end
      repeat (5) @(negedge clk_i);

      t0 = cycle;
      for (int k = 0; k < n_sets; k++) begin
        rc = $fscanf(fin, "%d %d %d", acc, act, has_bias);
        if (rc != 3) begin
          $display("[FATAL] set %0d header unreadable in %s", k, sets_f);
          $finish;
        end
        west_q.delete();
        north_q.delete();
        for (int i = 0; i < N * N; i++) begin
          rc = $fscanf(fin, "%h", w);
          west_q.push_back(w);
        end
        for (int i = 0; i < N * N; i++) begin
          rc = $fscanf(fin, "%h", w);
          north_q.push_back(w);
        end
        bias_i = '0;
        if (has_bias != 0)
          for (int c = 0; c < N; c++) begin
            rc = $fscanf(fin, "%h", w32);
            bias_i[c] = !(sienna_fmt_pkg::BIAS_DATA_WIDTH == sienna_fmt_pkg::DATA_WIDTH) ? ACC_DATA_WIDTH'(w32) : (ACC_DATA_WIDTH'(w32) << (ACC_DATA_WIDTH - DATA_WIDTH));
          end
        if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE) begin  // int8: the set's requantize and GPNAE words, laid out as requant_<k>.mem
          logic [31:0] q[$];
          q.delete();  // a static block: the queue would otherwise keep the previous set's words
          for (int i = 0; i < 8 + 2 * N; i++) begin
            rc = $fscanf(fin, "%h", w32);
            q.push_back(w32);
          end
          unpack_requant(q);
        end
        waited = 0;
        while (!pipeline_ready_o) begin
          @(negedge clk_i);
          waited++;
          if (waited > STALL_CYCLES) begin
            $display("[FATAL] pipeline_ready_o low for %0d cycles at set %0d, %0d sets done", waited, k, bounds.size());
            $finish;
          end
        end
        accumulate_i = acc[0];
        bias_valid_i = (has_bias != 0);
        activation_function_i = CONTROL_WIDTH'(act);
        if (host_gaps) begin
          load_set();
          start_pipeline_i = 1;
          @(negedge clk_i);
          start_pipeline_i = 0;
          @(negedge clk_i);  // let the credit land before sampling ready again
        end else load_and_start();
        if (k == 0) begin  // +coef_busy: writes while the first sets stream
          string bf;
          if ($value$plusargs("coef_busy=%s", bf)) fork coef_busy_writes(bf); join_none
        end
      end
      $fclose(fin);

      waited = 0;
      while (bounds.size() < n_sets) begin
        @(posedge clk_i);
        waited++;
        if (waited > STALL_CYCLES) begin
          $display("[FATAL] %0d of %0d sets completed, then nothing for %0d cycles", bounds.size(), n_sets, waited);
          $finish;
        end
      end
      @(posedge clk_i);
      // Every L0 and L9 slot comes home once the output stalls stop.
      osp = out_stall_pct;
      out_stall_pct = 0;
      repeat (OUT_CAP + 4) @(posedge clk_i);
      @(negedge clk_i);
      drain_chk = 1'b1;  // the link checkers judge before the count check below can stop the run
      @(negedge clk_i);
      drain_chk = 1'b0;
      repeat (2) @(posedge clk_i);
      if (lane_home !== '1 || host_cnt !== 2'd2) begin
        $display("[FATAL] credits not all back after the last set: host %0d of 2, lanes home %b", host_cnt, lane_home);
        $finish;
      end
      // sienna_top's own link checkers judge only when it is drained: it must drain after the last set.
      for (int w = 0; w < 200 && !dut.G_SET.G_COPY[0].pipe.drained; w++) @(posedge clk_i);
      if (!dut.G_SET.G_COPY[0].pipe.drained) begin
        $display("[FATAL] sienna_top never drained after the last set (%0d drains in the run): its link checkers never judged it", dut.G_SET.G_COPY[0].pipe.drain_n);
        $finish;
      end
      $display("  [Drain] sienna_top drained after the last set, %0d drains in the run", dut.G_SET.G_COPY[0].pipe.drain_n);

      // Sets complete in issue order, so set k's id is k mod 2^ID_W.
      bad_ids = 0;
      for (int k = 0; k < n_sets; k++) if (done_ids[k] != (k % (1 << ID_W))) bad_ids++;

      fout = $fopen(out_f, "w");
      for (int k = 0; k < n_sets; k++) begin
        automatic int lo = (k == 0) ? 0 : bounds[k-1];
        $fwrite(fout, "S %0d %0d\n", k, bounds[k] - lo);
        for (int i = lo; i < bounds[k]; i++) $fwrite(fout, "%h\n", res_q[i]);
      end
      $fclose(fout);
      if (busy_n > 0) begin
        $display("[MODEL] coef busy: %0d writes while sets streamed, %0d refused", busy_tries, busy_refused);
        if (busy_refused != busy_tries) $display("[FAIL] +coef_busy: %0d of %0d writes taken while sets streamed", busy_tries - busy_refused, busy_tries);
      end
      $display("[MODEL] COEF IMAGE : checks %0d  differ %0d", cc_checks, cc_bad);
      $display("[MODEL] sets=%0d outputs=%0d cycles=%0d mesh_busy=%0d act_busy=%0d order_errors=%0d out_stall=%0d", n_sets,
               res_q.size(), last_done - t0 + 1, busy_mesh, busy_act, bad_ids, osp);
      $finish;
    end

  end else if (MODE == 3) begin : g_copies
    // Streams NUM_SETS distinct sets through sienna_top at COPIES copies (SET_PORT = 1), checks every set against its golden output, and measures throughput.
    // The host is the producer of every copy's L0 link and of both cache regions (L2), and the consumer of every copy's L9 links.
    // +cached: sets go in fills of each region's cache (COPIES to COPIES+2 sets, B from the cache, each copy's last set of a fill marked), an uncached set after every second fill.
    // +out_slots=S (1..64) and +out_stall_pct=P on L9; +fault=1 a fill of one set, 2 a copy's first set of a fill marked, 3 a put to a copy whose turn it is not,
    // 4 the last set alone in a fill of one, then the host stops.
    // +mid_reset=C: a cut pass (cached: region 0's fill whole, region 1's opened with one set) is reset C cycles after it starts, once copy 0 holds a staging credit;
    // every count must come home with no stray output before the stream runs. -DTB_STALE_HOME keeps copy 0's staging count through that reset.
    // LINK_STAGES (TB_sienna's parameter): each copy's register stages on L0, L1, L3 and L9
    // The build's format, from the define through sienna_fmt_pkg; EXP_W, MAN_W and IS_INT only decode words; protocol keys on the package's HAS_ flags.
    localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH, ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;
    localparam int RESULT_DATA_WIDTH = sienna_fmt_pkg::RESULT_DATA_WIDTH;
    localparam int EXP_W = sienna_fmt_pkg::EXPONENT_WIDTH, MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH, IS_INT = (sienna_fmt_pkg::EXPONENT_WIDTH == 0);
    localparam ADDR_LINES = $clog2(FIFO_DEPTH);
    localparam int TIMEOUT_CYCLES = 400_000;
    localparam int WAIT_CYCLES = 20_000;  // a credit this long in coming is a hang
    localparam real REL_TOL = 0.01;
    localparam int WC_TILES = 128;
    localparam int HALF = WC_TILES / 2;  // tiles per cache region
    localparam int WCTW = $clog2(WC_TILES);
    localparam int WCAW = $clog2(WC_TILES * N * N);
    localparam int PACK_ENTRIES = 8;
    localparam int OUT_CAP = 64;  // sienna_top's OUT_MAX
    localparam int SW = $clog2(COPIES + 1);
    `include "sienna_set_side.svh"
    `include "tb_side_pack.svh"
`define TB_CC_LANES COPIES * NUM_LANES
`define TB_CC_MEM(i) dut.G_SET.G_COPY[(i) / NUM_LANES].pipe.backend_lanes[(i) % NUM_LANES].gpnae.COEF
    `include "tb_coef_check.svh"
`undef TB_CC_LANES
`undef TB_CC_MEM
    // The layer contract's row links (L10), unused at SET_PORT = 1: nothing is ever put on them.
    credit_link_if #(.DATA_W(N * DATA_WIDTH), .CRW($clog2(N + 1))) idle_rows[2] ();
    for (genvar r = 0; r < 2; r++) begin : G_IDLE_ROWS
      assign idle_rows[r].put  = 1'b0;
      assign idle_rows[r].data = '0;
    end

    logic clk_i = 0, rstn_i = 0;
    always #5 clk_i = ~clk_i;
    logic por_n = 0;  // the power-on reset only: TB_STALE_HOME's copy 0 staging counter ignores the mid-stream reset

    logic [N-1:0][ACC_DATA_WIDTH-1:0] bias_i = '0;  // int32 in int8; fp32 bits in floats, the bias row widened exactly
    logic [N-1:0][23:0] req_mult_i = '0;  // int8 (D-2)
    logic [N-1:0][5:0] req_shift_i = '0;
    logic [7:0] req_zp_i = '0, req_min_i = '0, req_max_i = '0, gp_zout_i = '0;
    logic [5:0] gp_shout_i = '0;
    logic [15:0] gp_mx_i = '0;
    logic [4:0] gp_shx_i = '0;
    logic [23:0] gp_mout_i = '0;
    logic wc_write_enable_i = 0;
    logic [WCAW-1:0] wc_write_addr_i = '0;
    logic north_write_enable_i = 0, west_write_enable_i = 0, north_write_reset_i = 0, west_write_reset_i = 0;
    logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i = '0, west_write_data_i = '0;  // operands in the package's format
    logic [SW-1:0] copy_sel_o;
    logic [COPIES-1:0] pipeline_complete_o;
    logic [COPIES-1:0][$clog2(SETS_IN_FLIGHT+1)-1:0] done_set_id_o;
    logic drain_chk = 1'b0;

    // ── L0 per copy: a counter and a checker each; link signals change on the falling edge ──
    credit_link_if #(.DATA_W($bits(set_side_t)), .CRW(1)) host[COPIES] ();
    set_side_t side = '0;
    logic [COPIES-1:0] hput = '0;
    logic [1:0] hcnt[COPIES];
    for (genvar c = 0; c < COPIES; c++) begin : G_HOST
      assign host[c].put = hput[c];
      assign host[c].data = side;
`ifdef TB_STALE_HOME
      credit_counter #(.MAX(2), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(c == 0 ? por_n : rstn_i), .put_i(host[c].put), .credit_i(host[c].credit),
                                             .has_credit_o(), .count_o(hcnt[c]));
`else
      credit_counter #(.MAX(2), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host[c].put), .credit_i(host[c].credit),
                                             .has_credit_o(), .count_o(hcnt[c]));
`endif
      credit_link_checker #(.SLOTS(2)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drain_chk), .lnk(host[c]));
    end

    // ── L2: this host opens each fill of a region with one put while it holds that region's credit ──
    credit_link_if #(.DATA_W(1), .CRW(1)) wc_region[2] ();
    logic [1:0] wput = '0;
    logic wcnt[2];
    for (genvar r = 0; r < 2; r++) begin : G_WC
      assign wc_region[r].put = wput[r];
      assign wc_region[r].data = 1'b0;
      credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(wc_region[r].put), .credit_i(wc_region[r].credit),
                                             .has_credit_o(), .count_o(wcnt[r]));
      credit_link_checker #(.SLOTS(1)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drain_chk), .lnk(wc_region[r]));
    end

    // ── L9 per copy and lane: copy c's lane l is out_lnk[c*NUM_LANES + l] ──
    localparam int NL = COPIES * NUM_LANES;
    credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) out_lnk[NL] ();
    int out_slots = OUT_CAP, out_stall_pct = 0;
    logic [NL-1:0] ov, lane_home;
    logic [DATA_WIDTH-1:0] od[NL];
    for (genvar i = 0; i < NL; i++) begin : G_OUT
      tb_l9_sink #(.DATA_W(DATA_WIDTH), .MAX_SLOTS(OUT_CAP)) snk (.clk_i(clk_i), .rstn_i(rstn_i), .slots_i(out_slots),
                                                                  .stall_pct_i(out_stall_pct), .drain_i(drain_chk),
                                                                  .lnk(out_lnk[i]), .valid_o(ov[i]), .data_o(od[i]),
                                                                  .home_o(lane_home[i]));
    end

    sienna_top #(
        .SET_PORT         (1),  // the set contract, COPIES pipelines in turn
        .COPIES           (COPIES),
        .NUM_LANES        (NUM_LANES),
        .N                (N),
        .TILE_SIZE        (TILE_SIZE),
        .HOST_WORDS       (HOST_WORDS),
        .COLLAPSE_K       (COLLAPSE_K),
        .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
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
        .FIFO_DEPTH       (FIFO_DEPTH),
        .LINK_STAGES      (LINK_STAGES),
        .OUT_MAX          (OUT_CAP),
        .OUT_CRW          (1)
    ) dut (
        .clk_i               (clk_i),
        .rstn_i              (rstn_i),
        // The layer contract, unused at SET_PORT = 1: tied off
        .cfg_load_i          (1'b0),
        .cfg_m_i             ('0),
        .cfg_kb_i            ('0),
        .cfg_n_i             ('0),
        .cfg_residual_i      (1'b0),
        .cfg_bias_i          (1'b0),
        .cfg_act_i           ('0),
        .cfg_tier_i          ('0),
        .cfg_train_i         (1'b0),
        .cfg_seed_i          ('0),
        .cfg_lane_i          ('0),
        .cfg_add_i           ('0),
        .cfg_pack_shift_i    ('0),
        .cfg_pack_map_i      ('0),
        .cfg_pack_act_i      ('0),
        .cfg_pack_lane_i     ('0),
        .busy_o              (),
        .done_o              (),
        .a_rows              (idle_rows[0]),
        .w_rows              (idle_rows[1]),
        .w_bias_i            ('0),
        .w_req_scale_i       ('0),
        .set_done_o          (),
        .host                (host),
        .bias_i              (bias_i),
        .wc_region           (wc_region),
        .wc_write_enable_i   (wc_write_enable_i),
        .wc_write_addr_i     (wc_write_addr_i),
        .north_write_enable_i(north_write_enable_i),
        .north_write_data_i  (north_write_data_i),
        .north_write_reset_i (north_write_reset_i),
        .west_write_enable_i (west_write_enable_i),
        .west_write_data_i   (west_write_data_i),
        .west_write_reset_i  (west_write_reset_i),
        .drained_i           (drain_chk),
        .copy_sel_o          (copy_sel_o),
        .coef_wr_i           (1'b0),  // the lanes keep their reset coefficient memory
        .coef_addr_i         ('0),
        .coef_data_i         ('0),
        .coef_ready_o        (),
        .out                 (out_lnk),
        .pipeline_complete_o (pipeline_complete_o),
        .done_set_id_o       (done_set_id_o),
        .systolic_busy_o     (),
        .gpnae_busy_o        (),
        .maxpool_busy_o      (),
        .dropout_busy_o      ()
    );

    function automatic real f32(input logic [31:0] b);
      int e;
      real m, v;
      e = int'(b[30:23]);
      m = real'(longint'(b[22:0])) / 8388608.0;
      if (e == 255) v = 1.0e38;
      else if (e == 0) v = 0.0;
      else v = (1.0 + m) * (2.0 ** (e - 127));
      return b[31] ? -v : v;
    endfunction

    // Within REL_TOL, or within the golden model's fp32 error bound for this output.
    function automatic bit close(input logic [31:0] e, input logic [31:0] a, input logic [31:0] bnd);
      real er = f32(e), ar = f32(a), br = f32(bnd), d;
      if ($isunknown(a)) return 0;  // an x or z word fails, whatever its real value would be
      if (EXACT_GOLDEN) return e === a;  // narrow formats: bit-exact golden
      d = (er > ar) ? er - ar : ar - er;
      if (br > 0.0 && d <= br) return 1;
      if (er == 0.0) return ar == 0.0;
      return d <= REL_TOL * ((er > 0.0) ? er : -er);
    endfunction

    task automatic read_mem_file(input string fn, output logic [DATA_WIDTH-1:0] q[$]);
      integer fh, rc;
      logic [DATA_WIDTH-1:0] w;
      q.delete();
      fh = $fopen(fn, "r");
      if (!fh) begin
        $display("[ERROR] Cannot open: %s", fn);
        $finish;
      end
      while (!$feof(fh)) begin
        rc = $fscanf(fh, "%h", w);
        if (rc == 1) q.push_back(w);
      end
      $fclose(fh);
    endtask

    `include "int8_tb_util.svh"  // read_word_file, unpack_requant, apply_requant

    function automatic logic [LFSR_WIDTH-1:0] set_seed(input int k);
      return LFSR_WIDTH'(DROPOUT_SEED ^ (32'h85EBCA6B * k));
    endfunction

    longint cyc = 0;
    always_ff @(posedge clk_i) cyc <= cyc + 1;  // nonblocking, as in the layer mode: a blocking ++ raced its readers under --threads

    // Per copy: the sets it was given, in order; per lane of each copy, the words of its current set.
    // Keyed by int: Verilator dropped updates to a one-element fixed array of queues.
    int copy_sets[int][$];
    logic [DATA_WIDTH-1:0] lq[int][$];
    logic [DATA_WIDTH-1:0] got[int][$];
    longint t_start[int], t_done[int];
    int n_done = 0;

    always @(posedge clk_i) begin
      for (int i = 0; i < NL; i++) if (ov[i]) lq[i].push_back(od[i]);
      for (int c = 0; c < COPIES; c++) begin
        if (pipeline_complete_o[c]) begin
          automatic int k;
          automatic logic [DATA_WIDTH-1:0] w[$];
          automatic bit any;
          w.delete();
          any = 1;
          k = -1;
          if (copy_sets[c].size() > 0) begin
            k = copy_sets[c][0];
            copy_sets[c] = copy_sets[c][1:$];
          end
          // Lane l's j-th word is window j*NUM_LANES + l, so popping the lanes in turn rebuilds window order whatever each lane's stalls.
          while (any) begin
            any = 0;
            for (int l = 0; l < NUM_LANES; l++)
              if (lq[c*NUM_LANES+l].size() != 0) begin
                w.push_back(lq[c*NUM_LANES+l].pop_front());
                any = 1;
              end
          end
          $display("  [done] copy %0d set %0d: %0d outputs at cycle %0d", c, k, w.size(), cyc);
          got[k] = w;
          t_done[k] = cyc;
          n_done++;
        end
      end
    end

    logic [DATA_WIDTH-1:0] wq[$], nq[$], eq[$], bq[$];
    int failed = 0, fault = 0, tsel = 0;
    bit cached_mode = 0;

    // Wait on falling edges for a credit; this long without one is a hang.
    task automatic wait_copy_credit(input int c);
      int waited;
      waited = 0;
      while (hcnt[c] == 0) begin
        @(negedge clk_i);
        waited++;
        if (waited > WAIT_CYCLES) begin
          $display("[FATAL] copy %0d held no staging credit for %0d cycles, %0d sets done", c, waited, n_done);
          $finish;
        end
      end
    endtask
    task automatic wait_region_credit(input int r);
      int waited;
      waited = 0;
      while (!wcnt[r]) begin
        @(negedge clk_i);
        waited++;
        if (waited > WAIT_CYCLES) begin
          $display("[FATAL] region %0d credit not back for %0d cycles, %0d sets done", r, waited, n_done);
          $finish;
        end
      end
    endtask

    // One set to the copy whose turn it is: its A rows (and B unless cached) on the write buses, then its put with the sideband.
    task automatic put_set(input int k, input bit cached, input int tile, input bit mark);
      automatic int c;
      c = tsel;
      read_mem_file($sformatf("matrix_west_%0d.mem", k), wq);
      read_mem_file($sformatf("matrix_north_%0d.mem", k), nq);
      wait_copy_credit(c);
      if (copy_sel_o !== c) begin
        failed++;
        $display("  [FAIL] set %0d: copy_sel_o %0d, the host's turn %0d", k, copy_sel_o, c);
      end
      for (int i = 0; i < wq.size(); i += HOST_WORDS) begin
        west_write_enable_i  = 1;
        north_write_enable_i = !cached;
        for (int w = 0; w < HOST_WORDS; w++) begin
          west_write_data_i[w]  = (i + w < wq.size()) ? wq[i+w] : '0;
          north_write_data_i[w] = (i + w < nq.size()) ? nq[i+w] : '0;
        end
        @(negedge clk_i);
      end
      west_write_enable_i  = 0;
      north_write_enable_i = 0;
      bias_i = '0;
      if (HAS_BIAS != 0) begin  // this TB streams no accumulate groups: every set carries its own bias
        logic [31:0] bw[$];
        read_word_file($sformatf("bias_%0d.mem", k), bw);
        for (int j = 0; j < N; j++) bias_i[j] = !(sienna_fmt_pkg::BIAS_DATA_WIDTH == sienna_fmt_pkg::DATA_WIDTH) ? ACC_DATA_WIDTH'(bw[j]) : (ACC_DATA_WIDTH'(bw[j]) << (ACC_DATA_WIDTH - DATA_WIDTH));
      end
      apply_requant(k);
      side               = '0;
      side.wc_last       = mark;
      side.weight_tile   = WCTW'(tile);
      side.weight_cached = cached;
      side.bias_valid    = (HAS_BIAS != 0);
      side.train         = TRAINING_MODE;
      side.seed          = set_seed(k);
      side.act[0]        = ACTIVATION_CODE;
      side.entry         = tb_side_entries({{(PACK_ENTRIES - 1) * TB_ENTRY_W{1'b0}},
                                            tb_entry(req_zp_i, req_min_i, req_max_i, gp_mx_i, gp_shx_i, gp_mout_i, gp_shout_i, gp_zout_i)});
      side.scale         = tb_side_scales(req_mult_i, req_shift_i);
      if (fault == 3 && k == 0) begin
        $display("  [FAULT 3] set 0 put to copy %0d, whose turn it is not", (c + 1) % COPIES);
        c = (c + 1) % COPIES;
      end
      copy_sets[c].push_back(k);
      if (!cached) $display("  [start] set %0d to copy %0d at cycle %0d", k, c, cyc);
      else if (!mark) $display("  [start] set %0d to copy %0d at cycle %0d, cached tile %0d", k, c, cyc, tile);
      else $display("  [start] set %0d to copy %0d at cycle %0d, cached tile %0d, its copy's last of the fill", k, c, cyc, tile);
      t_start[k] = cyc;
      hput[c] = 1'b1;
      @(negedge clk_i);
      hput = '0;
      tsel = (tsel + 1) % COPIES;
    endtask

    // Opens a fill of region r and writes sets k..k+s-1's B into its tiles 0..s-1.
    task automatic fill_region(input int r, input int k, input int s);
      wait_region_credit(r);
      wput[r] = 1'b1;
      @(negedge clk_i);
      wput[r] = 1'b0;
      for (int j = 0; j < s; j++) begin
        read_mem_file($sformatf("matrix_north_%0d.mem", k + j), nq);
        for (int i = 0; i < nq.size(); i += HOST_WORDS) begin
          wc_write_enable_i = 1;
          wc_write_addr_i = WCAW'((r * HALF + j) * N * N + i);
          for (int w = 0; w < HOST_WORDS; w++) north_write_data_i[w] = (i + w < nq.size()) ? nq[i+w] : '0;
          @(negedge clk_i);
        end
        wc_write_enable_i = 0;
      end
      $display("  [fill] region %0d: sets %0d..%0d at cycle %0d", r, k, k + s - 1, cyc);
    endtask

    // The cut pass: sets in every copy, a fill open, then a reset mid-stream; the counts must come home with no stray output.
    task automatic cut_pass(input int after);
      automatic longint t0 = cyc;
      automatic int stray = 0, waited = 0;
      if (cached_mode) begin
        fill_region(0, 0, COPIES);
        for (int j = 0; j < COPIES; j++) put_set(j, 1, j, 1);  // a whole fill, each copy's set its last
        fill_region(1, COPIES, COPIES);
        put_set(COPIES, 1, HALF, 0);  // region 1's fill opened, one copy's set put
      end else begin
        for (int k = 0; k < 2 * COPIES; k++) put_set(k, 0, 0, 0);
      end
      while (cyc - t0 < after || hcnt[0] == 0) begin
        @(negedge clk_i);
        waited++;
        if (waited > WAIT_CYCLES) begin
          $display("[FATAL] cut pass: copy 0 held no staging credit for %0d cycles", waited);
          $finish;
        end
      end
      $display("  [Reset] mid-stream, %0d cycles after the cut pass began: %0d sets done, copy 0 holds %0d staging credits, regions %0d %0d",
               cyc - t0, n_done, hcnt[0], wcnt[0], wcnt[1]);
      rstn_i = 0;
      repeat (5) @(negedge clk_i);
      copy_sets.delete();
      lq.delete();
      got.delete();
      t_start.delete();
      t_done.delete();
      n_done = 0;
      tsel = 0;
      rstn_i = 1;
      repeat (OUT_CAP + 64 + 8 * LINK_STAGES) begin
        @(posedge clk_i);
        if (ov !== '0 || pipeline_complete_o !== '0) stray++;
      end
      cc_image("coef_default.mem");
      cc_check("mid-stream reset");
      begin
        automatic bit home = (lane_home === '1) && wcnt[0] && wcnt[1] && copy_sel_o === '0 && g_copies.dut.G_SET.G_MULTI.wc_got[0] === '0 && g_copies.dut.G_SET.G_MULTI.wc_got[1] === '0 && stray == 0;
        for (int c = 0; c < COPIES; c++) if (hcnt[c] !== 2'd2) home = 0;
        if (!home) begin
          $display("[FAIL] not home after the mid-stream reset: copy 0 staging %0d (2), copy %0d staging %0d (2), regions %0d %0d (1 1), turn %0d (0), region returns %b %b (0 0), lanes home %0d of %0d, %0d stray output cycles",
                   hcnt[0], COPIES - 1, hcnt[COPIES-1], wcnt[0], wcnt[1], copy_sel_o, g_copies.dut.G_SET.G_MULTI.wc_got[0], g_copies.dut.G_SET.G_MULTI.wc_got[1], $countones(lane_home), NL,
                   stray);
          $display("RESULT: FAILED");
          $finish;
        end else
          $display("  [Reset] home: staging 2 in each of %0d copies, regions 1 1, turn 0, no region return pending, %0d L9 lanes with %0d slots, no stray output",
                   COPIES, NL, out_slots);
      end
      lq.delete();
    endtask

    initial begin
      automatic int mid_reset = 0;
      void'($value$plusargs("mid_reset=%d", mid_reset));
      void'($value$plusargs("out_slots=%d", out_slots));
      void'($value$plusargs("out_stall_pct=%d", out_stall_pct));
      void'($value$plusargs("fault=%d", fault));
      cached_mode = $test$plusargs("cached");
      if (out_slots < 1 || out_slots > OUT_CAP || out_stall_pct < 0 || out_stall_pct > 90) begin
        $display("[ERROR] +out_slots=%0d must be 1..%0d and +out_stall_pct=%0d 0..90", out_slots, OUT_CAP, out_stall_pct);
        $finish;
      end
      $display(" Multi: %0d copies, cached sets %0d, %0d L9 slots per lane, %0d%% of credit returns stalled, fault %0d", COPIES,
               cached_mode, out_slots, out_stall_pct, fault);
      repeat (4) @(posedge clk_i);
      rstn_i = 1;
      por_n = 1;
      repeat (4) @(negedge clk_i);
      cc_image("coef_default.mem");
      cc_check("reset");
      if (mid_reset > 0) cut_pass(mid_reset);
      if (!cached_mode) begin
        for (int k = 0; k < NUM_SETS; k++) put_set(k, 0, 0, 0);
      end else begin
        automatic int k = 0, f = 0;
        while (k < NUM_SETS) begin
          automatic int s;
          s = COPIES + (f % 3);
          if (fault == 1 && f == 0) s = 1;  // fewer sets than copies: a copy never sees a marked set
          if (fault == 2 && f == 0) s = COPIES + 1;
          if (fault == 4 && k == NUM_SETS - 1) s = 1;  // a short last fill: the host stops with a copy unmarked
          if (fault == 4 && k < NUM_SETS - 1 && k + s > NUM_SETS - 1) s = NUM_SETS - 1 - k;
          if (k + s > NUM_SETS) s = NUM_SETS - k;
          if (s < COPIES && !(fault == 1 && f == 0) && !(fault == 4 && k == NUM_SETS - 1)) begin
            put_set(k, 0, 0, 0);
            k++;
            continue;
          end
          fill_region(f % 2, k, s);
          for (int j = 0; j < s; j++) begin
            automatic bit mark;
            mark = (j >= s - COPIES);  // each copy's last set of the fill, in round-robin order
            if ((fault == 1 && f == 0) || (fault == 4 && k == NUM_SETS - 1)) mark = 1;
            if (fault == 2 && f == 0 && j == 0) begin
              mark = 1;
              $display("  [FAULT 2] set %0d marked although its copy takes set %0d of the same fill", k, k + COPIES);
            end
            put_set(k + j, 1, (f % 2) * HALF + j, mark);
          end
          k += s;
          f++;
          if (f % 2 == 0 && k < NUM_SETS - (fault == 4 ? 1 : 0)) begin
            put_set(k, 0, 0, 0);
            k++;
          end
        end
      end
      while (n_done < NUM_SETS && cyc < TIMEOUT_CYCLES) @(posedge clk_i);
      @(posedge clk_i);
      if (n_done < NUM_SETS) begin
        failed++;
        $display("  [FAIL] only %0d of %0d sets completed", n_done, NUM_SETS);
      end
      for (int k = 0; k < NUM_SETS; k++) begin
        automatic int errs;
        errs = 0;
        read_mem_file($sformatf("expected_output_%0d.mem", k), eq);
        read_mem_file($sformatf("bound_output_%0d.mem", k), bq);
        if (!got.exists(k) || got[k].size() != eq.size()) begin
          failed++;
          $display("  [FAIL] set %0d: %0d outputs, expected %0d", k, got.exists(k) ? got[k].size() : -1, eq.size());
          continue;
        end
        for (int i = 0; i < eq.size(); i++) if (!close(eq[i], got[k][i], (i < bq.size()) ? bq[i] : '0)) errs++;
        if (errs) begin
          failed++;
          $display("  [FAIL] set %0d: %0d of %0d outputs out of tolerance", k, errs, eq.size());
        end
        $display("  set %0d: latency %0d cycles, done at %0d, %0d outputs, %0d mismatches", k,
                 t_done.exists(k) ? t_done[k] - t_start[k] : -1, t_done.exists(k) ? t_done[k] : -1, eq.size(), errs);
      end
      // Every credit comes home once the output stalls stop: staging 2 per copy, each region 1, every L9 slot.
      out_stall_pct = 0;
      repeat (OUT_CAP + 4) @(posedge clk_i);
      for (int c = 0; c < COPIES; c++)
        if (hcnt[c] !== 2'd2) begin
          failed++;
          $display("  [FAIL] copy %0d holds %0d of its 2 staging credits after the last set", c, hcnt[c]);
        end
      if (!wcnt[0] || !wcnt[1]) begin
        failed++;
        $display("  [FAIL] region credits not back after the last set: %0d %0d", wcnt[0], wcnt[1]);
      end
      if (lane_home !== '1) begin
        failed++;
        $display("  [FAIL] L9 slots not all credited back: lanes home %b", lane_home);
      end
      @(negedge clk_i);
      drain_chk = 1'b1;
      @(negedge clk_i);
      drain_chk = 1'b0;
      repeat (2) @(posedge clk_i);  // the registered drain checks judge a cycle later
      // Steady state over the second half, spanning whole rounds of COPIES sets so bursts do not skew it.
      begin
        automatic longint tl[$];
        automatic real gap_sum = 0.0;
        automatic int ng, lo;
        for (int k = 0; k < NUM_SETS; k++) if (t_done.exists(k)) tl.push_back(t_done[k]);
        tl.sort();
        ng = ((tl.size() / 2) / COPIES) * COPIES;
        lo = tl.size() - 1 - ng;
        if (ng > 0 && lo >= 0) gap_sum = real'(tl[tl.size()-1] - tl[lo]);
        if (ng > 0 && lo >= 0)
          $display("MULTI COPIES=%0d N=%0d lanes=%0d host_words=%0d collapse_k=%0d: steady %.1f cycles per set, %.1f FLOP/cycle",
                   COPIES, N, NUM_LANES, HOST_WORDS, COLLAPSE_K, gap_sum / ng, 2.0 * N * N * N * ng / gap_sum);
      end
      // A hash of every set's words in set order, for comparing runs.
      begin
        automatic longint unsigned h = 64'd1469598103934665603;
        automatic int words = 0;
        for (int k = 0; k < NUM_SETS; k++)
          if (got.exists(k))
            for (int i = 0; i < got[k].size(); i++) begin
              h = (h ^ 64'(got[k][i])) * 64'd1099511628211;
              words++;
            end
        $display("MULTI_HASH %016h over %0d words", h, words);
      end
      failed += cc_bad;  // a lane memory that differs from its image fails the run
      $display("MULTI COEF IMAGE : checks %0d  differ %0d", cc_checks, cc_bad);
      // A ternary of two strings prints as a number under Verilator, so branch instead.
      if (failed == 0) $display("RESULT: PASSED");
      else $display("RESULT: FAILED");
      $finish;
    end
  end else begin : g_bad_mode
    initial $fatal(1, "TB_sienna: MODE %0d: 0 pipeline, 1 layer, 2 sets or 3 copies", MODE);
  end

endmodule
