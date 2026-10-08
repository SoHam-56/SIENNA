`timescale 1ns / 100ps

import test_config_pkg::*;

// LINK_STAGES: register stages on sienna_top's links; FAULT 1: a put with no credit and no rows, 3: a host link one bit wide.
module TB_sienna_top #(
    parameter int LINK_STAGES = 0,
    parameter int FAULT       = 0  // also 6: 8 L5 slots per lane (a_l5_starved); 7: output links one bit wider
);

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
  logic [N-1:0][ACC_W-1:0] bias_i;  // int32 in int8, where it carries the folded input zero point
  // int8: the requantize and GPNAE parameters of the set being started (D-2); zero in other formats
  logic [N-1:0][31:0] req_mult_i;
  logic [N-1:0][7:0]  req_shift_i;
  logic [7:0]         req_zp_i, req_min_i, req_max_i, gp_shout_i, gp_zout_i;
  logic [15:0]        gp_mx_i;
  logic [4:0]         gp_shx_i;
  logic [31:0]        gp_mout_i;
  localparam int PACK_ENTRIES = 8, PEW = 3;
  logic [2:0] pack_shift_i;  // the set's pack shift and parameter table (pack_<k>.mem), with the start
  logic [N/2-1:0][PEW-1:0] pack_map_i;
  logic [PACK_ENTRIES-1:1][CONTROL_WIDTH-1:0] pack_act_i;
  logic [PACK_ENTRIES-1:1][7:0] pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i;
  logic [PACK_ENTRIES-1:1][15:0] pack_mx_i;
  logic [PACK_ENTRIES-1:1][4:0] pack_shx_i;
  logic [PACK_ENTRIES-1:1][31:0] pack_mout_i;
  localparam int WC_TILES = 128;
  logic                     weight_cached_i;  // this set takes B from cache tile weight_tile_i
  logic [$clog2(WC_TILES)-1:0] weight_tile_i;
  logic                     wc_write_enable_i;
  logic [$clog2(WC_TILES*N*N)-1:0] wc_write_addr_i;
  logic                     wc_last_i;  // this set is the last of its cache region's fill
  logic [LFSR_WIDTH-1:0]    dropout_seed_i;
  logic [CONTROL_WIDTH-1:0] activation_function_i;
  logic [     ADDR_LINES:0] num_terms_i;
  localparam int PER_LANE = SRAM_DEPTH / NUM_LANES;  // result beats per set

  // ── Links: this TB is the host (L0) and the weight-cache writer (L2), each with a counter and a checker ──
  `include "sienna_set_side.svh"
  localparam int SIDE_W = $bits(set_side_t);
  credit_link_if #(.DATA_W(SIDE_W + ((FAULT == 3) ? 1 : 0)), .CRW(1)) host_lnk ();
  credit_link_if #(.DATA_W(1), .CRW(1)) wc_lnk[2] ();
  logic [1:0] host_cnt;  // staging credits this host holds
  logic       wc_cnt[2];  // per cache region: its credit is held
  bit         wc_open_tb[2];  // per region: a fill is open (put, its last set not yet put)
  logic       drained;  // nothing in flight for a while: every credit must be home
  int         tb_inflight, quiet_cyc;  // sets put and not complete; cycles with none
  localparam int QUIET_TB = 16 + 4 * LINK_STAGES;
  credit_counter #(.MAX(2), .CRW(1)) host_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(host_lnk.put), .credit_i(host_lnk.credit),
                                              .has_credit_o(), .count_o(host_cnt));
  localparam int HOST_SLOTS = (SETS_IN_FLIGHT < 2) ? SETS_IN_FLIGHT : 2;  // the host holds both staging credits unless the entry admits fewer sets
  credit_link_checker #(.SLOTS(HOST_SLOTS)) chk_host (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(host_lnk));
  for (genvar r = 0; r < 2; r++) begin : G_WC
    credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(wc_lnk[r].put), .credit_i(wc_lnk[r].credit),
                                           .has_credit_o(), .count_o(wc_cnt[r]));
    credit_link_checker #(.SLOTS(1)) chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained && !wc_open_tb[r]), .lnk(wc_lnk[r]));
  end
  logic sum_open;  // the last set put was a partial sum: it completes before the mesh frees its staging bank, so no drain until the sum ends
  set_side_t side_put;
  assign side_put = set_side_t'(host_lnk.data[SIDE_W-1:0]);
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) begin
      tb_inflight <= 0;
      quiet_cyc   <= 0;
      sum_open    <= 1'b0;
    end else begin
      tb_inflight <= tb_inflight + int'(host_lnk.put) - int'(pipeline_complete_o);
      quiet_cyc   <= (tb_inflight != 0 || host_lnk.put || sum_open) ? 0 : (quiet_cyc < QUIET_TB) ? quiet_cyc + 1 : quiet_cyc;
      if (host_lnk.put) sum_open <= side_put.accumulate;
    end
  assign drained = (quiet_cyc == QUIET_TB);
  bit acc_on = 0;  // the accumulate pass is running
  bit stream_on = 0;  // a streaming pass is running
  logic [DATA_WIDTH-1:0] stream_results[$];
  int stream_bounds[$];
  int stream_ids[$];
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
  logic intermediate_buffer_full_tb, intermediate_buffer_empty_tb;

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

  // ── DUT ───────────────────────────────────────────────────────────────
  sienna_top #(
      .NUM_LANES        (NUM_LANES),
      .SETS_IN_FLIGHT   (SETS_IN_FLIGHT),
      .N                (N),
      .TILE_SIZE        (TILE_SIZE),
      .HOST_WORDS       (HOST_WORDS),
      .DATA_WIDTH       (DATA_WIDTH),
      .EXP_W         (EXP_W),
      .MAN_W         (MAN_W),
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
      .out                        (out_lnk),
      .pipeline_complete_o        (pipeline_complete_o),
      .done_set_id_o              (done_set_id_o),
      .systolic_busy_o            (systolic_busy_tb),
      .gpnae_busy_o               (gpnae_busy_tb),
      .maxpool_busy_o             (maxpool_busy_tb),
      .dropout_busy_o             (dropout_busy_tb),
      .intermediate_buffer_full_o (intermediate_buffer_full_tb),
      .intermediate_buffer_empty_o(intermediate_buffer_empty_tb)
  );

  wire [4:0] dut_stage = {dut.g_state, dut.p_state};  // activation and pooling stage states
  wire res_ready = dut.systolic_array_inst.out_full[dut.systolic_array_inst.out_rd];  // the mesh holds a finished result

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
          stream_ids.push_back(int'(done_set_id_o));
        end
        if (acc_on) acc_bounds.push_back(acc_results.size());
      end
    end
  end

  // Every lane link holds all its slots: L4 32, L5 the collector's, the pooling input (FIFO2, or L9 with a 1x1 pool) and L9 out_adv.
  function automatic bit lanes_home(output string why);
    automatic int pin = TB_POOL_BYPASS ? out_adv : 16;
    for (int l = 0; l < NUM_LANES; l++)
      if (int'(dut.l4_cnt[l]) != 32 || int'(dut.l5_out[l]) != TB_L5 || int'(dut.pin_cnt[l]) != pin || out_held[l] != out_adv ||
          out_owed[l] != 0) begin
        why = $sformatf("lane %0d: L4 %0d (32), L5 %0d (%0d), pool input %0d (%0d), L9 %0d (%0d), L9 owed %0d (0)", l, dut.l4_cnt[l],
                        dut.l5_out[l], TB_L5, dut.pin_cnt[l], pin, out_held[l], out_adv, out_owed[l]);
        return 0;
      end
    why = $sformatf("lanes: L4 32, L5 %0d, pool input %0d, L9 %0d each", TB_L5, pin, out_adv);
    return 1;
  endfunction

  // The sideband of the set being put, from the per-set signals above; entry 0 of each table is the set's own.
  function automatic set_side_t side_now();
    set_side_t s;
    s               = '0;
    s.wc_last       = wc_last_i;
    s.weight_tile   = weight_tile_i;
    s.weight_cached = weight_cached_i;
    s.accumulate    = accumulate_i;
    s.bias_valid    = bias_valid_i;
    s.train         = training_mode_i;
    s.seed          = dropout_seed_i;
    s.terms         = num_terms_i;
    s.pack_shift    = pack_shift_i;
    s.pack_map      = pack_map_i;
    s.act           = {pack_act_i, activation_function_i};
    s.zp            = {pack_zp_i, req_zp_i};
    s.amin          = {pack_min_i, req_min_i};
    s.amax          = {pack_max_i, req_max_i};
    s.mx            = {pack_mx_i, gp_mx_i};
    s.shx           = {pack_shx_i, gp_shx_i};
    s.mout          = {pack_mout_i, gp_mout_i};
    s.shout         = {pack_shout_i, gp_shout_i};
    s.zout          = {pack_zout_i, gp_zout_i};
    s.mult          = req_mult_i;
    s.shift         = req_shift_i;
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
    host_lnk.data = side_now();
    host_lnk.put  = 1'b1;
    @(negedge clk_i);
    host_lnk.put  = 1'b0;
    if (wc_last_i) wc_open_tb[weight_tile_i[$clog2(WC_TILES)-1]] = 1'b0;
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
               res_ready, dut.systolic_mult_complete);
      prev_state = dut_stage;
    end
  end

  task automatic print_status(input string stage_label);
    $display(
        "[STATUS @ %0t] sys_busy=%0b gpnae_busy=%0b max_busy=%0b drop_busy=%0b | full=%0b empty=%0b",
        $time, systolic_busy_tb, gpnae_busy_tb, maxpool_busy_tb, dropout_busy_tb,
        intermediate_buffer_full_tb, intermediate_buffer_empty_tb);
  endtask

  // ── Reset ─────────────────────────────────────────────────────────────
  task automatic reset();
    $display("\n[STAGE] Reset");
    rstn_i = 0;
    out_cr = '0;  // the downstream consumer drops its L9 credits with the reset, as it does every other link signal
    host_lnk.put = 1'b0;
    host_lnk.data = '0;
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
    num_terms_i = '0;
    repeat (10) begin
      @(negedge clk_i);
      // Review Focus 1: while reset is held every producer count is 0 and no consumer advertises a credit.
      if (host_cnt != 0 || wc_cnt[0] || wc_cnt[1] || dut.l1_cnt != 0 || dut.l6_cnt != 0 || dut.sets_out != 0 || dut.l0_out != 0 ||
          dut.systolic_array_inst.res_cnt != 0 || host_lnk.credit || wc_lnk[0].credit || wc_lnk[1].credit || dut.l3_grant ||
          dut.l6_credit || dut.systolic_array_inst.stg_credit) begin
        failed++;
        $display("  [FAIL] In reset: host %0d, regions %0d %0d, staging %0d, act banks %0d, sets %0d, granted %0d, results %0d; credits out %0b%0b%0b%0b%0b%0b",
                 host_cnt, wc_cnt[0], wc_cnt[1], dut.l1_cnt, dut.l6_cnt, dut.sets_out, dut.l0_out, dut.systolic_array_inst.res_cnt,
                 host_lnk.credit, wc_lnk[0].credit, wc_lnk[1].credit, dut.l3_grant, dut.l6_credit, dut.systolic_array_inst.stg_credit);
      end
      // The lane links too: no L4, L5, pooling-input or L9 count and no credit while reset is held.
      for (int l = 0; l < NUM_LANES; l++)
        if (dut.l4_cnt[l] != 0 || dut.l5_out[l] != 0 || dut.pin_cnt[l] != 0 || dut.l4_cr[l] || dut.l5_cr[l] || dut.pin_cr[l]) begin
          failed++;
          $display("  [FAIL] In reset: lane %0d L4 %0d, L5 %0d, pool input %0d; credits out %0b%0b%0b", l, dut.l4_cnt[l], dut.l5_out[l],
                   dut.pin_cnt[l], dut.l4_cr[l], dut.l5_cr[l], dut.pin_cr[l]);
          break;
        end
    end
    @(posedge clk_i);
    rstn_i = 1;
    north_write_reset_i = 0;
    west_write_reset_i = 0;
    if (FAULT == 1) begin  // a put with no credit and no rows, the cycle reset ends
      @(negedge clk_i);
      host_lnk.data = side_now();
      host_lnk.put = 1'b1;
      @(negedge clk_i);
      host_lnk.put = 1'b0;
      $display("  [FAULT 1] put with no credit and no rows @ %0t", $time);
    end
    repeat (5) @(posedge clk_i);
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
        $display("  filled_total=%0d  total_elements=%0d  all_collected=%0b", dut.filled_total,
                 dut.total_elements, dut.all_collected);
        $display("  res_ready=%0b  result beat=%0b  l3_armed=%0b  disp_done=%0b",
                 res_ready, dut.wide_rd_valid, dut.l3_armed, dut.disp_done);
        // The host side of L0 and the mesh's staging side of L1.
        $display("  host credits=%0d sets_out=%0d granted=%0d staging=%0d north_queue_empty=%0b west_queue_empty=%0b",
                 host_cnt, dut.sets_out, dut.l0_out, dut.l1_cnt, dut.north_queue_empty, dut.west_queue_empty);
        for (int i = 0; i < 4; i++)
          $display("  lane %0d: fill_count=%0d done_count=%0d load_finalized=%0b collected=%0b",
                   i, dut.fill_count[i], dut.done_count[i], dut.load_finalized[i],
                   dut.lane_collected[i]);
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
        if (dut.fifo2_wr_valid[lane]) begin
          $display("[DEBUG %0t] Dispatcher -> Maxpool (Lane %0d) : dec=%.4f  hex=%08x", $time, lane,
                   real'($bitstoshortreal(dut.fifo2_wr_data[lane])), dut.fifo2_wr_data[lane]);
        end
      end
      // Dropout -> Output lines are printed by the L9 consumer, a set at a time in window order, so stalls cannot reorder them.
    end
  end

  always @(posedge clk_i) begin
    if (rstn_i && trace_fd) begin
      if (dut.wide_rd_valid) begin
        for (int lane = 0; lane < NUM_LANES; lane++)
          $fdisplay(trace_fd, "[%0t] Systolic -> GPNAE   : dec=%.6f  hex=%08x", $time,
                    real'($bitstoshortreal(dut.wide_rd_data[lane])), dut.wide_rd_data[lane]);
      end
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.fifo2_rd_ready[lane] && dut.fifo2_rd_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] GPNAE -> Maxpool    lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.fifo2_rd_data[lane])),
                    dut.fifo2_rd_data[lane]);
        end
      end
      for (int lane = 0; lane < NUM_LANES; lane++) begin
        if (dut.dropout_in_valid[lane]) begin
          $fdisplay(trace_fd, "[%0t] Maxpool -> Dropout  lane=%0d : dec=%.6f  hex=%08x", $time,
                    lane, real'($bitstoshortreal(dut.dropout_data_in[lane])),
                    dut.dropout_data_in[lane]);
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
              $time, reason, dut_stage, dut.all_collected, dut.streaming_complete);
    for (int lane = 0; lane < NUM_LANES; lane++) begin
      $fdisplay(
          lane_fd,
          "  lane=%0d fill_count=%0d done_count=%0d load_finalized=%0b lane_collected=%0b | L4 %0d L5 %0d pool-in %0d | maxpool out %0d of %0d | dropout_out_count=%0d",
          lane, dut.fill_count[lane], dut.done_count[lane], dut.load_finalized[lane],
          dut.lane_collected[lane], dut.l4_cnt[lane], dut.l5_out[lane], dut.pin_cnt[lane],
          dut.mp_out_count[lane], dut.lane_windows_total[lane], dut.dropout_out_count[lane]);
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

  // With ACCUM_PASSES = P, sets come in groups of P: the first P-1 are partial sums, the last is activated.
  function automatic bit is_partial(input int k);
    return (ACCUM_PASSES > 1) && ((k % ACCUM_PASSES) != ACCUM_PASSES - 1);
  endfunction

  // Set k's activation: with MIXED_LEN > 0 the codes cycle through MIXED_ACTS, 4 bits per set.
  function automatic logic [CONTROL_WIDTH-1:0] act_of(input int k);
    return (MIXED_LEN > 0) ? CONTROL_WIDTH'((MIXED_ACTS >> (4 * (k % MIXED_LEN))) & 15) : CONTROL_WIDTH'(ACTIVATION_CODE);
  endfunction

  // Polynomial terms for set k's code, the same table as model_runner.py's ACTIVATION_TERMS.
  function automatic logic [ADDR_LINES:0] terms_of(input int k);
    if (MIXED_LEN == 0) return NUM_TERMS[ADDR_LINES:0];
    case (act_of(k))
      1: return 14;
      2: return 15;
      3: return 30;
      default: return 0;
    endcase
  endfunction

  // Set k's bias: with HAS_BIAS, the first pass of each group reads bias_<k>.mem.
  task automatic apply_bias(input int k);
    logic [31:0] q[$];
    bias_valid_i = (HAS_BIAS != 0) && ((k % ACCUM_PASSES) == 0);
    bias_i = '0;
    if (bias_valid_i) begin
      read_word_file($sformatf("bias_%0d.mem", k), q);
      for (int c = 0; c < N; c++) bias_i[c] = ACC_W'(q[c]);
    end
  endtask

  // Set k's pack shift, block map and table entries 1..7 (pack_<k>.mem); unpacked tests drive zeros.
  task automatic apply_pack(input int k);
    logic [31:0] q[$];
    pack_shift_i = '0;
    pack_map_i = '0;
    {pack_act_i, pack_zp_i, pack_min_i, pack_max_i, pack_shout_i, pack_zout_i, pack_mx_i, pack_shx_i, pack_mout_i} = '0;
    if (PACKED == 0) return;
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
      pack_mout_i[e]  = q[o+6];
      pack_shout_i[e] = q[o+7][7:0];
      pack_zout_i[e]  = q[o+8][7:0];
    end
  endtask

  // With WEIGHT_CACHE, set k's B is written once into cache tile k of region 0, one fill, and the set sends only A.
  task automatic write_cache();
    logic [DATA_WIDTH-1:0] q[$];
    if (WEIGHT_CACHE == 0) return;
    if (NUM_SETS > WC_TILES / 2) begin
      $display("[FATAL] %0d cached sets do not fit cache region 0 (%0d tiles)", NUM_SETS, WC_TILES / 2);
      $finish;
    end
    wc_open(0);
    for (int k = 0; k < NUM_SETS; k++) begin
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
    $display("  [Cache] %0d weight tiles written", NUM_SETS);
  endtask

  task automatic apply_weight(input int k);
    weight_cached_i = (WEIGHT_CACHE != 0);
    weight_tile_i = ($clog2(WC_TILES))'(k);
    if (weight_cached_i) north_data_queue.delete();
  endtask

  // Set k's dropout seed; regression.py's set_dropout_seed() mirrors it.
  function automatic logic [LFSR_WIDTH-1:0] set_seed(input int k);
    return LFSR_WIDTH'(DROPOUT_SEED ^ (32'h85EBCA6B * k));
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
  int pool_with_result = 0;  // pooling busy while a mesh result waited, so activation could have overlapped it
  int stream_id_base;  // sets started before the stream, which consumed ids
  wire mesh_computing = dut.systolic_array_inst.mesh_busy;  // a set between staging and a written result
  // A finished set waits for a result bank: the mesh is held up by the consumer.
  wire mesh_blocked = dut.systolic_array_inst.arrays_final && dut.systolic_array_inst.reducers_ready &&
                      !dut.systolic_array_inst.reduce_start;
  int withheld = 0;  // cycles the full entry kept a free staging bank from the host

  initial forever begin
    @(negedge clk_i);
    if (stream_on) begin
      if (mesh_computing && gpnae_busy_tb) ov_mesh_g++;
      if (gpnae_busy_tb && maxpool_busy_tb) ov_g_p++;
      // A finished result sits in the mesh, not being pushed, while pooling runs.
      if (maxpool_busy_tb && res_ready && !dut.wide_rd_valid) pool_with_result++;
      if (mesh_blocked) bp_mesh++;  // a finished set waits because no result bank is free
      if (int'(dut.g_state) == 0 && res_ready && dut.act_full[dut.act_wr])
        bp_act++;  // a mesh result waits because both activation banks are full
      if (n_started - stream_bounds.size() > max_in_flight)
        max_in_flight = n_started - stream_bounds.size();
    end
  end

  // int8: set id, activation bank and bypass flag of each beat inside the requantize pipeline, oldest first.
  int rq_id[$], rq_bank[$], rq_byp[$];
  int ov_rq = 0, rq_wait = 0;
  int ov_lane = 0, rq_hold = 0, rq_null_wait = 0, rq_null_pass = 0, rd_in = 0;  // rd_in: wide reads of the stage's set so far
  wire next_ready = res_ready && dut.mesh_sets != 0 && !dut.set_accum[dut.g_next_id];  // the next set's result is ready to push
  initial forever begin
    @(negedge clk_i);
    if (EXP_W == 0) begin
      if (!rstn_i) begin
        rq_id.delete();
        rq_bank.delete();
        rq_byp.delete();
        rd_in = 0;
      end else begin
        if (stream_on && rq_id.size() != 0) begin
          // Overlap: set k's beats still in the requantize pipeline while the stage holds set k+1 or has granted its beats.
          if ((int'(dut.g_state) != 0 && rq_id[0] != int'(dut.g_set_id)) || (int'(dut.g_state) == 0 && dut.l3_armed)) begin
            ov_rq++;
            if (rq_byp[0] != 0 && int'(dut.g_state) != 0 && !dut.act_bypass) ov_lane++;  // a ReLU or linear beat drains while a lane set holds the stage
          end
          // A ReLU or linear set's beats are all in, the next result is ready with a free bank, yet no beats granted: the drain holds it.
          else if (rq_byp[0] != 0 && int'(dut.g_state) == 0 && !dut.l3_armed && !dut.l3_grant && next_ready &&
                   !dut.act_full[rq_bank[0] == 0])
            rq_wait++;
          // The draining ReLU or linear set still holds the stage, not leaving this cycle, though the next set could start.
          if (int'(dut.g_state) != 0 && rq_id[0] == int'(dut.g_set_id) && rq_byp[0] != 0 && dut.act_bypass &&
              rd_in == SRAM_DEPTH / NUM_LANES && !dut.g_done && next_ready && !dut.act_full[!dut.act_wr])
            rq_hold++;
          // A partial set that could pass the stage but for the drain, and one that passed during it.
          if (int'(dut.g_state) == 0 && dut.mesh_sets != 0 && dut.set_accum[dut.g_next_id]) rq_null_wait++;
          if (dut.g_null_done) rq_null_pass++;
        end
        if (int'(dut.g_state) == 0) rd_in = 0;
        else if (dut.wide_rd_valid) rd_in++;
        if (dut.fill_v) begin
          void'(rq_id.pop_front());
          void'(rq_bank.pop_front());
          void'(rq_byp.pop_front());
        end
        if (dut.wide_rd_valid) begin
          rq_id.push_back(int'(dut.g_set_id));
          rq_bank.push_back(int'(dut.act_wr));
          rq_byp.push_back(int'(dut.act_bypass));
        end
      end
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
  // hold (Review Focus 3): from the first words of set HOLD_SET, every L9 credit is withheld for 500 cycles; the outputs must equal the first plain pass's.
  // HOLD_SET leaves SETS_IN_FLIGHT+4 sets to put after it where NUM_SETS allows (else set 0), so the buffers fill and the host is blocked.
  localparam int HOLD_SET = (NUM_SETS / 2 < NUM_SETS - SETS_IN_FLIGHT - 4) ? NUM_SETS / 2
                          : ((NUM_SETS - SETS_IN_FLIGHT - 4 > 0) ? NUM_SETS - SETS_IN_FLIGHT - 4 : 0);
  // The host must end up blocked when the test has more sets than SETS_IN_FLIGHT: by staging credits once the banks are full, or by the entry first (fewer credits, or partial sums that hold no result bank).
  logic [DATA_WIDTH-1:0] ref_results[$];  // the first plain pass's words and set boundaries
  int ref_bounds[$];
  task automatic stream_all_sets(input int id_base, input bit overrun, input bit close_fill, input bit hold = 0);
    automatic longint t0 = $time;
    automatic int hold_stalled = 0, hold_done = 0, hold_full = 0, hold_inflight = 0, hold_blocked = 0, hold_capped = 0;
    $display("\n[STAGE] STREAMING: %0d sets through overlapped stages%s%s", NUM_SETS,
             overrun ? $sformatf(", the host waiting for a credit on set %0d with every admitted set in flight", SETS_IN_FLIGHT) : "",
             hold ? $sformatf(", %0d L9 slot per lane and every L9 credit withheld 500 cycles from set %0d's first words", out_adv, HOLD_SET)
                  : "");
    stream_results.delete();
    stream_bounds.delete();
    stream_ids.delete();
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
    while (pipeline_complete_o) @(posedge clk_i);  // the previous pass's pulse is not a set boundary
    @(posedge clk_i);
    stream_on = 1;
`ifdef PERF
    $display("PERF %0d PASS %0d %0d", int'($time / 10), id_base, overrun);
`endif
    fork
      begin
        fork
          begin : producer
            for (int k = 0; k < NUM_SETS; k++) begin
              read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
              read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
              dropout_seed_i = set_seed(k);
              accumulate_i = is_partial(k);
              activation_function_i = act_of(k);
              num_terms_i = terms_of(k);
              apply_bias(k);
              apply_requant(k);
              apply_pack(k);
              apply_weight(k);
              wc_last_i = close_fill && (WEIGHT_CACHE != 0) && (k == NUM_SETS - 1);
`ifdef PERF
              if (!overrun) wait_credit();
              $display("PERF %0d HOST_LOAD %0d", int'($time / 10), k);
`endif
              if (overrun && k == SETS_IN_FLIGHT) begin
                automatic int waited = 0, held = 0, free_bank = 0;
                // The host cannot put without a credit: wait until every admitted set is in flight, then count the cycles it gets none.
                while (!(host_cnt == 0 && dut.entry_full) && waited < 2000) begin
                  @(posedge clk_i);
                  waited++;
                end
                if (host_cnt == 0 && dut.entry_full) begin
                  while (host_cnt == 0) begin
                    held++;
                    if (dut.l1_cnt > dut.l0_out) free_bank++;  // a staging bank the full entry keeps from the host
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
            for (int k = 0; k < NUM_SETS; k++) begin
              while (stream_bounds.size() <= k) @(posedge clk_i);
              read_mem_file($sformatf("expected_output_%0d.mem", k), exp_q);
              verify_slice(k, exp_q);
            end
          end
          begin : holder
            if (hold) begin
              automatic bit started = 0;
              while (!started && stream_bounds.size() < NUM_SETS) begin
                @(negedge clk_i);
                if (stream_bounds.size() >= HOLD_SET)
                  for (int l = 0; l < NUM_LANES; l++) if (lane_q[l].size() != 0) started = 1;
              end
              if (!started) begin
                failed++;
                $display("  [FAIL] Hold: set %0d never started its output", HOLD_SET);
              end else begin
                out_hold  = 1;
                hold_puts = 0;
                repeat (500) begin
                  @(negedge clk_i);
                  if (out_put == '0) hold_stalled++;
                  if (pipeline_complete_o) hold_done++;
                  if (host_cnt == 0 && dut.entry_full) hold_full++;
                  if (host_cnt == 0 && !dut.entry_full && n_started < NUM_SETS) hold_blocked++;  // a set to put, no staging credit
                  if (host_cnt == 0 && dut.entry_full && n_started < NUM_SETS) hold_capped++;  // a set to put, the entry at SETS_IN_FLIGHT
                  if (tb_inflight > hold_inflight) hold_inflight = tb_inflight;
                end
                out_hold = 0;
                $display("  [Hold] every L9 credit withheld 500 cycles from set %0d's first words: %0d words put, %0d cycles with no word, %0d completions, %0d sets most in flight, %0d cycles with the entry full, the host blocked with a set to put %0d cycles by staging credits and %0d by the entry, %0d sets put",
                         HOLD_SET, hold_puts, hold_stalled, hold_done, hold_inflight, hold_full, hold_blocked, hold_capped, n_started);
                if (NUM_SETS <= SETS_IN_FLIGHT)
                  $display("  [Hold] host blocking not reachable: the test's %0d sets fit in SETS_IN_FLIGHT %0d", NUM_SETS, SETS_IN_FLIGHT);
                else if (hold_blocked + hold_capped == 0) begin
                  failed++;
                  $display("  [FAIL] Hold: the host was never blocked with a set to put, so the stall never reached it");
                end
                if (hold_stalled < 400 || hold_puts > out_adv * NUM_LANES) begin
                  failed++;
                  $display("  [FAIL] Hold: the output did not stall (%0d of 500 cycles with no word, %0d words on %0d credits per lane)", hold_stalled,
                           hold_puts, out_adv);
                end
              end
            end
          end
        join
      end
      begin : watchdog
        repeat (TIMEOUT_CYCLES) @(posedge clk_i);
        $display("[FATAL] Timeout in the streaming pass: %0d of %0d sets completed",
                 stream_bounds.size(), NUM_SETS);
        $finish;
      end
    join_any
    disable fork;
    stream_on = 0;
    out_hold = 0;
    // $time is in the 1 ns timeunit, so a 10 ns clock is 10 units per cycle.
    $display("  [%s] %0d sets in %0d cycles", hold ? "Hold" : "Stream", NUM_SETS, ($time - t0) / 10);
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
      if (!drained || tb_inflight != 0 || host_cnt != HOST_SLOTS || dut.sets_out != 0 || dut.l0_out != HOST_SLOTS || !lanes_home(why)) begin
        failed++;
        $display("  [FAIL] Hold: not home after the drain: drained %0b, in flight %0d, host %0d (%0d), sets %0d, granted %0d; %s", drained,
                 tb_inflight, host_cnt, HOST_SLOTS, dut.sets_out, dut.l0_out, why);
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
        for (int k = 0; k < NUM_SETS; k++) begin
          read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
          read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
          accumulate_i = is_partial(k);
          activation_function_i = act_of(k);
          num_terms_i = terms_of(k);
          apply_bias(k);
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
    while (!(gpnae_busy_tb && dut.sets_out >= HOST_SLOTS) && waited < TIMEOUT_CYCLES) begin
      @(posedge clk_i);
      waited++;
    end
    if (!(gpnae_busy_tb && dut.sets_out >= HOST_SLOTS)) begin
      failed++;
      $display("  [FAIL] Never reached two sets in flight before the reset");
    end
    disable fork;
    $display("  [Reset] resetting with %0d sets in flight, host credits %0d, stages {g,p}=%0d", dut.sets_out, host_cnt, dut_stage);
    reset();
    activation_function_i = ACTIVATION_CODE[CONTROL_WIDTH-1:0];
    num_terms_i           = NUM_TERMS[ADDR_LINES:0];
    training_mode_i       = TRAINING_MODE[0];
    repeat (2000) begin
      @(posedge clk_i);
      if (pipeline_complete_o || (|out_put)) stray++;
    end
    if (stray != 0) begin
      failed++;
      $display("  [FAIL] %0d cycles of output after reset with nothing started", stray);
    end
    // Review Focus 1: every consumer advertised its slots again and every producer counted them from 0.
    if (host_cnt != HOST_SLOTS || dut.l1_cnt != 2 || dut.l0_out != HOST_SLOTS || dut.sets_out != 0 || dut.l6_cnt != 2 || !dut.l3_armed ||
        int'(dut.systolic_array_inst.res_cnt) != PER_LANE || !wc_cnt[0] || !wc_cnt[1] || dut_stage != 0) begin
      failed++;
      $display("  [FAIL] Not idle after reset: host %0d (%0d), staging %0d (2), granted %0d (%0d), sets %0d (0), act banks %0d (2), results %0d (%0d), regions %0d %0d (1 1), stages %0d",
               host_cnt, HOST_SLOTS, dut.l1_cnt, dut.l0_out, HOST_SLOTS, dut.sets_out, dut.l6_cnt, dut.systolic_array_inst.res_cnt, PER_LANE,
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
    activation_function_i = ACTIVATION_CODE[CONTROL_WIDTH-1:0];
    num_terms_i           = NUM_TERMS[ADDR_LINES:0];
    training_mode_i       = TRAINING_MODE[0];
    repeat (100) @(posedge clk_i);
    if (!lanes_home(why)) begin
      failed++;
      $display("  [FAIL] Lane links not re-advertised after the second reset: %s", why);
    end else $display("  [Reset] idle after the second reset: %s", why);
    write_cache();
    stream_all_sets(0, 0, PACKED != 0, 1);  // with the 500-cycle hold; the accumulate pass closes the fill when it runs
  endtask

  // ── Review Focus 4: four partial sets of zeros, then set ACCUM_PASSES-1's own passes ─────
  // Only the last set has a result: one L3 set of beats and one L6 bank; the partial sets spend no L3 or L6 credit.
  int acc_beats = 0, acc_banks = 0;
  always_ff @(posedge clk_i) begin
    if (acc_on) begin  // its words and boundaries come from the L9 consumer
      if (dut.wide_rd_valid) acc_beats <= acc_beats + 1;
      if (dut.bank_done) acc_banks <= acc_banks + 1;
    end
  end

  task automatic accum_null_pass();
    automatic int G = ACCUM_PASSES - 1;  // the first set the stream activates; its golden is the sum of sets 0..G
    automatic int n = 4 + ACCUM_PASSES, waited = 0, errs = 0, n_exact = 0;
    automatic logic [DATA_WIDTH-1:0] exp_q[$], bnd_q[$];
    string info;
    $display("\n[STAGE] ACCUMULATE: 4 partial sets of zeros, then sets 0..%0d as one sum; checkers bound", G);
    if (PACKED != 0) begin
      $display("  [Accum] skipped: a packed set cannot be a partial sum");
      return;
    end
    while (pipeline_complete_o) @(posedge clk_i);
    acc_results.delete();
    acc_bounds.delete();
    acc_beats = 0;
    acc_banks = 0;
    acc_on = 1;
    for (int j = 0; j < n; j++) begin
      automatic int k = (j < 4) ? 0 : j - 4;  // the stream set whose rows this pass sends
      read_mem_file($sformatf("matrix_west_%0d.mem", k), west_data_queue);
      read_mem_file($sformatf("matrix_north_%0d.mem", k), north_data_queue);
      if (j < 4) foreach (west_data_queue[i]) west_data_queue[i] = '0;
      accumulate_i = (j < n - 1);
      activation_function_i = act_of(G);
      num_terms_i = terms_of(G);
      dropout_seed_i = set_seed(G);
      apply_bias(0);  // the sum's bias rides its first pass only
      if (j != 0) begin
        bias_valid_i = 1'b0;
        bias_i = '0;
      end
      apply_requant(G);
      apply_pack(0);
      apply_weight(k);
      wc_last_i = (WEIGHT_CACHE != 0) && (j == n - 1);
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
    if (acc_beats != PER_LANE || acc_banks != 1 || dut.l6_cnt != 2 || !dut.l3_armed ||
        int'(dut.systolic_array_inst.res_cnt) != PER_LANE || dut.l1_cnt != 2 || host_cnt != HOST_SLOTS) begin
      failed++;
      $display("  [FAIL] Accumulate pass credits: beats %0d (%0d), banks %0d (1), act banks held %0d (2), results held %0d (%0d), staging %0d (2), host %0d (%0d)",
               acc_beats, PER_LANE, acc_banks, dut.l6_cnt, dut.systolic_array_inst.res_cnt, PER_LANE, dut.l1_cnt, host_cnt, HOST_SLOTS);
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
  assign g_eff = (dut.g_fed || dut.wide_rd_valid) ? int'(dut.g_state) : 0;  // the stage's state once its set's beats arrive
  assign entry_free = SETS_IN_FLIGHT - int'(dut.sets_out) - int'(dut.l0_out);  // sets the entry may still admit
  initial forever begin
    @(negedge clk_i);
    if (stream_on) begin
      automatic int c = int'($time / 10);
      automatic int busy = 0;
      // Mesh events, one of each per set and in set order: 2 broadcast start, 3 broadcast end, 4 feed start, 5 reduce start, 7 written.
      if (dut.systolic_array_inst.set_launch) begin
        $display("PERF %0d MESH 1", c);
        $display("PERF %0d MESH 2", c);
      end
      if (dut.systolic_array_inst.bcast_release) $display("PERF %0d MESH 3", c);
      if (dut.systolic_array_inst.ROW[0].COL[0].DEPTH[0].S.tile.launch) $display("PERF %0d MESH 4", c);
      if (dut.systolic_array_inst.reduce_start) $display("PERF %0d MESH 5", c);
      if (dut.systolic_array_inst.set_done) $display("PERF %0d MESH 7", c);
      // The stage takes its credits ahead of the result; G counts from the set's first beat, as main's read did.
      if (g_eff != perf_g_st) begin
        $display("PERF %0d G %0d", c, g_eff);
        if (perf_g_st == 3) $display("PERF %0d LANES %0d %0d", c, perf_lane_busy, perf_round_cyc);
        if (perf_g_st == 0) begin
          perf_lane_busy = 0;
          perf_round_cyc = 0;
        end
      end
      if (int'(dut.p_state) != perf_p_st) $display("PERF %0d P %0d", c, dut.p_state);
      if (entry_free != perf_cred) $display("PERF %0d CREDITS %0d", c, entry_free);
      if (int'(dut.wide_rd_valid) != perf_mread) $display("PERF %0d MREAD %0d", c, dut.wide_rd_valid);
      if (pipeline_complete_o) $display("PERF %0d DONE %0d", c, done_set_id_o);
      if (g_eff == 3) begin
        for (int i = 0; i < NUM_LANES; i++) if (dut.load_finalized[i] && !dut.lane_collected[i]) busy++;
        perf_lane_busy += busy;
        perf_round_cyc++;
      end
    end
    perf_g_st    = g_eff;
    perf_p_st    = int'(dut.p_state);
    perf_cred    = entry_free;
    perf_mread   = int'(dut.wide_rd_valid);
  end
`endif

  // ── Top-level stimulus ────────────────────────────────────────────────
  initial begin

    $display("==============================================");
    $display(" SIENNA PIPELINE VERIFICATION");
    $display("==============================================");
    $display(" N=%-0d  TILE_SIZE=%-0d  FIFO_DEPTH=%-0d", N, TILE_SIZE, FIFO_DEPTH);
    $display(" Activation code : %0b  Num terms : %0d", ACTIVATION_CODE, NUM_TERMS);
    $display(" Dropout         : %s  seed 0x%08h", TRAINING_MODE ? "training" : "inference", DROPOUT_SEED);
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
    num_terms_i           = terms_of(0);
    apply_bias(0);
    apply_requant(0);
    apply_pack(0);
    training_mode_i       = TRAINING_MODE[0];
    dropout_seed_i        = set_seed(0);
    @(posedge clk_i);
    host_put();
    $display("  start pulse sent @ %0t", $time);
    print_status("immediately after start");
    print_lane_status("immediately after start");

    collect_outputs();
    verify_outputs();

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
      read_mem_file("expected_output_1.mem", expected_results);
      read_mem_file("bound_output_1.mem", bound_results);
      dropout_seed_i = set_seed(1);
      activation_function_i = act_of(1);
      num_terms_i = terms_of(1);
      apply_bias(1);
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
               dut_stage, dut.north_queue_empty, dut.west_queue_empty, host_cnt);
      host_put();
      $display("  [B2B] gate at put    : state=%0d north_empty=%0b west_empty=%0b",
               dut_stage, dut.north_queue_empty, dut.west_queue_empty);
      @(posedge clk_i);
      $display("  [B2B] state after pulse: %0d", dut_stage);

      collect_outputs();
      verify_outputs();

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
      stream_all_sets(ids_used + NUM_SETS, 1, 0);
    end
    reset_mid_stream();
    accum_null_pass();

    $display("\n==============================================");
    $display(" RESULT SUMMARY");
    $display("==============================================");
    $display(" Total    : %0d", total_elements);
    $display(" Exact    : %0d", exact_passed);
    $display(" Tol pass : %0d  (rel <= %.1f%%)", tol_passed, REL_TOL * 100.0);
    $display(" Failed   : %0d", failed);
    $display("----------------------------------------------");
    if (failed == 0) $display(" RESULT: PASSED");
    else $display(" RESULT: FAILED (%0d mismatches)", failed);
    $display("==============================================");

    if (trace_fd) $fclose(trace_fd);
    if (lane_fd) $fclose(lane_fd);

    #100 $finish;
  end

  initial begin
`ifdef ENABLE_TRACE
`ifdef TRACE_FST
    $dumpfile("TB_sienna_top.fst");
`else
    $dumpfile("TB_sienna_top.vcd");
`endif
    $dumpvars(0, TB_sienna_top);
`endif
  end

endmodule
