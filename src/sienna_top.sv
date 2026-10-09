`timescale 1ns / 100ps

module sienna_top #(
    parameter int    NUM_LANES         = 32,
    parameter int    N                 = 16,
    parameter int    TILE_SIZE         = 4,
    parameter int    HOST_WORDS        = N,  // words per host write, one matrix row; must divide N*N
    parameter int    COLLAPSE_K        = 1,  // 1: one full-depth mesh tile per output tile, N^2 PEs and no reduce
    parameter int    ACC_BANKS         = 4,  // mesh partial-sum banks per PE
    parameter int    RESULT_BANKS      = 4,  // mesh result banks
    parameter int    SETS_IN_FLIGHT    = 2 + 2 + ACC_BANKS + RESULT_BANKS + 2 + 1,  // credits: every set the banks can hold (staging, operand, partial-sum, result, activation, pooling)
    parameter int    ID_W              = $clog2(SETS_IN_FLIGHT + 1),  // set id width; ids count accepted starts
    parameter int    WC_TILES          = 128,  // weight cache tiles in the mesh
    parameter int    WCTW              = $clog2(WC_TILES),
    parameter int    WCAW              = $clog2(WC_TILES * N * N),
    parameter int    EXP_W             = 8,   // the build's number format: fp32 8/23, bf16 8/7
    parameter int    MAN_W             = 23,
    parameter int    DATA_WIDTH        = 1 + EXP_W + MAN_W,  // every word: operands, results, activations
    parameter int    ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // sums and bias: int32 in int8, fp32 in every float format
    parameter int    OUT_W             = sienna_fmt_pkg::out_w(EXP_W, MAN_W),  // mesh result words: int32 in int8, DATA_WIDTH in floats
    parameter int    SRAM_DEPTH        = N * N,
    parameter int    FIFO_DEPTH        = N * N,
    parameter int    ADDR_LINES        = $clog2(FIFO_DEPTH),
    parameter int    CONTROL_WIDTH     = 3,  // activation: 001 SELU, 010 sigmoid, 011 tanh, 100 ReLU, 101 linear
    parameter int    IN_ROWS           = 16,
    parameter int    IN_COLS           = 16,
    parameter int    POOL_H            = 2,
    parameter int    POOL_W            = 2,
    parameter int    STRIDE_ROWS       = 2,
    parameter int    STRIDE_COLS       = 2,
    parameter int    PADDING           = 1,
    parameter int    PACK_ENTRIES      = 8,  // distinct activation / int8 output settings one packed set may mix; entry 0 is the per-set ports
    parameter int    DROPOUT_P_PERCENT = 50,
    parameter int    LFSR_WIDTH        = 32,
    parameter int    LINK_STAGES       = 0,  // register stages on the host, staging, result and output links (L0, L1, L3, L9); the write buses and the completion get as many as the put
    parameter int    OUT_MAX           = 64,  // the most L9 credits a downstream consumer may grant per lane
    parameter int    OUT_CRW           = 1,   // L9 credit width
    parameter int    LANE_OUT_SLOTS    = 16   // the collector's L5 slots per lane: one barrel group; below 16 a lane can never start (a_l5_starved)
) (
    input logic clk_i,
    input logic rstn_i,

    credit_link_if.consumer         host,  // L0: a put per set, data its sideband (sienna_set_side.svh); credits are free staging banks, withheld while SETS_IN_FLIGHT sets are in flight
    input logic [N-1:0][ACC_W-1:0]  bias_i,  // with the put: added to column c when the sideband's bias_valid is set; float builds: the row widened to fp32 bits
    credit_link_if.consumer         wc_region[2],  // L2, to the mesh: a put opens a fill of that cache region; its credit returns once the fill's last set is broadcast
    input logic                     wc_write_enable_i,  // weight cache write of north_write_data_i at word wc_write_addr_i
    input logic [WCAW-1:0]          wc_write_addr_i,
    input logic                     north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0]       north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0]       west_write_data_i,
    input logic                     west_write_reset_i,

    credit_link_if.producer         out[NUM_LANES],  // L9: lane k's pooled, dropped-out results, one word per put, while the consumer's credit is held
    // L9 lanes drift apart by up to FIFO2's 4 windows + maxpool's 2 windows ahead + the L9 slots: a consumer must not make one lane's credits wait on another lane's later windows.

    output logic pipeline_complete_o,  // with the put of a set's last word
    output logic [ID_W-1:0] done_set_id_o,  // id of the set pipeline_complete_o reports
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic maxpool_busy_o,
    output logic dropout_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o
);

  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);  // int8: int32 sums, requantized at the lane feed

  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "sienna_top: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC
    $fatal(1, "sienna_top: ACC_W=%0d, but the format accumulates in %0d bits", ACC_W, sienna_fmt_pkg::acc_w(EXP_W, MAN_W));
  end else if (OUT_W != sienna_fmt_pkg::out_w(EXP_W, MAN_W)) begin : G_BAD_OUT_W  // L3 and the lane feed would truncate or pad silently
    $fatal(1, "sienna_top: OUT_W=%0d, but the format's mesh results are %0d bits", OUT_W, sienna_fmt_pkg::out_w(EXP_W, MAN_W));
  end else if (SRAM_DEPTH != N * N) begin : G_BAD_SRAM_DEPTH  // L3 grants SRAM_DEPTH/NUM_LANES beats, the mesh pushes N*N/NUM_LANES
    $fatal(1, "sienna_top: SRAM_DEPTH %0d must be N*N (%0d)", SRAM_DEPTH, N * N);
  end

  localparam int GPNAE_DATA_WIDTH = DATA_WIDTH;
  localparam logic [DATA_WIDTH-1:0] NEG_INF = IS_INT ? {1'b1, {(DATA_WIDTH - 1) {1'b0}}}  // pooling pad: -128 in int8
                                                     : DATA_WIDTH'({DATA_WIDTH{1'b1}} << MAN_W);  // -infinity in a float format
  localparam int GPNAE_ADDR_LINES = 5;
  localparam int GPNAE_CTRL_WIDTH = 3;
  localparam int GPNAE_FIFO_DEPTH = 2 ** GPNAE_ADDR_LINES;
  // Work is split evenly across the lanes rather than filling each to its FIFO depth. With
  // 8 lanes those coincide (256/8 = 32 = GPNAE_FIFO_DEPTH), which is why the old code could use
  // the depth directly; at any other lane count it would leave the later lanes empty and index
  // gpnae_out_mem past its end.
  localparam int PER_LANE = SRAM_DEPTH / NUM_LANES;

  // Lanes take contiguous blocks of PER_LANE elements, so the division has to be exact: a
  // remainder would be dropped silently, and PER_LANE must fit a lane's input FIFO.
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial begin
    if ((SRAM_DEPTH % NUM_LANES) != 0)
      $error("sienna_top: NUM_LANES (%0d) must divide SRAM_DEPTH (%0d)", NUM_LANES, SRAM_DEPTH);
    if (PER_LANE > GPNAE_FIFO_DEPTH)
      $error("sienna_top: PER_LANE (%0d) exceeds GPNAE_FIFO_DEPTH (%0d)", PER_LANE, GPNAE_FIFO_DEPTH);
  end
`endif

  localparam int FCNT_W = $clog2(PER_LANE + 1);  // what fill_count/done_count actually range over
  localparam int TOT_W = $clog2(SRAM_DEPTH + 1);

  localparam int FIFO2_DEPTH = 16;
  // A 1x1 window with stride 1 and no padding is the identity: windows skip FIFO2 and maxpool and go straight to dropout.
  localparam bit POOL_BYPASS = (POOL_H == 1) && (POOL_W == 1) && (STRIDE_ROWS == 1) && (STRIDE_COLS == 1) && (PADDING == 0);
  localparam int MP_WIN = POOL_H * POOL_W;  // a maxpool window's elements, granted at once on L8
  localparam int MP_CRW = $clog2(MP_WIN + 1);
  localparam int MP_AHEAD = 2;  // windows maxpool grants ahead, so FIFO2 streams one element a cycle
  localparam int LANE_K = 16;  // gpnae_poly's K: a lane starts a group only with this many L5 credits

  localparam int POOL_OUT_ROWS = (IN_ROWS + 2 * PADDING - POOL_H) / STRIDE_ROWS + 1;
  localparam int POOL_OUT_COLS = (IN_COLS + 2 * PADDING - POOL_W) / STRIDE_COLS + 1;
  localparam int MAXPOOL_OUT_COUNT = POOL_OUT_ROWS * POOL_OUT_COLS;

  localparam int NUM_IDS = 1 << ID_W;  // more ids than sets in flight, so ids in flight never repeat
  localparam int CRW = $clog2(SETS_IN_FLIGHT + 1);
  localparam int PEW = $clog2(PACK_ENTRIES);
  localparam int LGN = $clog2(N);
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (PACK_ENTRIES < 2 || (PACK_ENTRIES & (PACK_ENTRIES - 1)) != 0) $error("sienna_top: PACK_ENTRIES (%0d) must be a power of two >= 2", PACK_ENTRIES);
`endif
  // A packed set's lanes each take one column: N must divide the lanes; the mesh packs only collapsed; pooling must be the identity.
  function automatic logic [PEW-1:0] ent_of(input logic [N/2-1:0][PEW-1:0] map, input logic [2:0] sh, input int col);
    return (sh == '0) ? map[0] : map[col >> (LGN - int'(sh))];
  endfunction
  function automatic logic is_byp_code(input logic [CONTROL_WIDTH-1:0] c);
    return (c == CONTROL_WIDTH'(3'b100)) || (c == CONTROL_WIDTH'(3'b101));
  endfunction
  // Lane k's element i of the set the activation stage holds: a run along a row, or for a packed set down column k % N.
  function automatic int elem(input int k, input int i, input logic pk);
    return pk ? ((k / N) * PER_LANE + i) * N + (k % N) : k * PER_LANE + i;
  endfunction
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (NUM_IDS < SETS_IN_FLIGHT) $error("sienna_top: ID_W=%0d is too narrow for %0d sets in flight", ID_W, SETS_IN_FLIGHT);
`endif

  // Activation stage: read a mesh result into FIFO1, fill the lanes, write gpnae_out_mem.
  // A partial set is summed in the mesh's PEs and leaves no result; it passes this stage in G_IDLE as a null.
  typedef enum logic [1:0] {G_IDLE, G_FEED, G_LATCH, G_ROUND} g_state_t;
  // Pooling stage: dispatch windows into FIFO2, then wait for maxpool and dropout to drain.
  typedef enum logic [1:0] {P_IDLE, P_DISPATCH, P_WAIT} p_state_t;
  g_state_t g_state;
  p_state_t p_state;

  logic host_accept, g_accept, g_done, p_accept, p_release, pool_done;
  // g_done: the stage may take the next set; bank_done: bank bank_sel holds a whole set (later than g_done for an int8 ReLU or linear set).
  logic bank_done, bank_sel, byp_all_in, rq_drain;
  logic lane_v;  // a beat for the fill counters and the lanes
  logic byp_wr, byp_bank;  // int8: a ReLU or linear beat to write, and its bank
  logic byp_pack;  // int8: the draining beat's set was packed
  logic [FCNT_W-1:0] byp_idx;  // its element within each lane's block
  logic [1:0] act_full;  // per activation bank: a finished activation not yet dispatched
  logic act_wr, act_rd;  // bank the lanes write, bank the dispatcher reads
  int act_wr_base, act_rd_base;
  logic [CRW-1:0] sets_out;  // sets put and not yet complete
  logic [CRW-1:0] l0_out;  // host credits granted and not yet put
  logic entry_full;  // SETS_IN_FLIGHT sets are in flight or granted: the entry withholds credits
  logic [CRW-1:0] mesh_sets;  // accepted sets the activation stage has not taken yet
  logic [ID_W-1:0] g_next_id, g_set_id, p_next_id, p_set_id, host_next_id;
  // Dropout mode and seed travel with each set, indexed by its id, so sets in flight keep their own.
  logic                  set_train[NUM_IDS];
  logic [LFSR_WIDTH-1:0] set_seed [NUM_IDS];
  logic                  set_accum[NUM_IDS];  // the set is a partial sum: accumulate it, output nothing
  logic [   ADDR_LINES:0] set_terms[NUM_IDS];  // polynomial terms that go with each set's activation
  logic [2:0] set_pack[NUM_IDS];  // each set's pack shift, block map and table activations (entry 0 = the set's own)
  logic [N/2-1:0][PEW-1:0] set_map[NUM_IDS];
  logic [PACK_ENTRIES-1:0][CONTROL_WIDTH-1:0] set_ents[NUM_IDS];
  logic [PACK_ENTRIES-1:0][CONTROL_WIDTH-1:0] g_ents;  // the activation stage's copy of its set's entry codes, taken at g_accept
  logic g_pack;  // the activation stage's set is packed
  logic [PEW-1:0] lane_ent[NUM_LANES];  // the entry each lane uses for that set
  logic [PEW-1:0] p_lane_ent[NUM_LANES];  // and each pooling lane for the pooled set
  logic [CONTROL_WIDTH-1:0] lane_act[NUM_LANES];
  logic act_bypass;  // every lane's code is ReLU or linear; declared here because the lane-control block above its old place reads it
  logic [CRW-1:0] gp_sets;  // sets past the activation stage and not yet complete; pooling takes them in id order
  // int8: the requantize and GPNAE parameters of the set the activation stage holds (g_*); p_zp: dropout's drop value for the pooled set.
  logic [N-1:0][31:0] g_mult;
  logic [N-1:0][7:0]  g_shift;
  logic [NUM_LANES-1:0][7:0]  g_zp, g_min, g_max, g_shout, g_zout, p_zp;  // per lane: its column block's entry
  logic [NUM_LANES-1:0][15:0] g_mx;
  logic [NUM_LANES-1:0][4:0]  g_shx;
  logic [NUM_LANES-1:0][31:0] g_mout;
  assign act_wr_base = act_wr ? SRAM_DEPTH : 0;
  assign act_rd_base = act_rd ? SRAM_DEPTH : 0;

  // Intermediate Memory Buffer
  logic [DATA_WIDTH-1:0] gpnae_out_mem  [0:2*SRAM_DEPTH-1];


  logic [NUM_LANES-1:0][OUT_W-1:0] wide_rd_data;  // a result beat: element k of lane k's block; int32 sums in int8, narrowed sums in floats
  logic                                 wide_rd_valid;  // a result beat arrives on L3
  logic                  systolic_mult_complete;
  logic north_queue_empty, west_queue_empty;

  // =========================================================================
  // LINKS: L0 host -> here, L1 here -> mesh, L3 mesh -> activation, L6 activation -> pooling; L2 passes to the mesh
  // =========================================================================
  `include "sienna_set_side.svh"
  localparam int SIDE_W = $bits(set_side_t);
  localparam int STG_W  = WCTW + 7;  // the mesh's staging sideband
  localparam int L3_W   = NUM_LANES * OUT_W + 3;  // a result beat: {packed, last, first, one word per lane}
  localparam int L3_CRW = $clog2(PER_LANE + 1);  // a set's beats granted in one cycle
`ifndef SYNTHESIS  // interface widths are not elaboration constants in Verilator, so the host and output links are checked at time 0
  initial
    if ($bits(host.data) != SIDE_W || $bits(host.credit) != 1)
      $fatal(1, "sienna_top: the host link needs data %0d bits and credit 1, found %0d and %0d", SIDE_W, $bits(host.data), $bits(host.credit));
  initial
    if ($bits(out[0].data) != DATA_WIDTH || $bits(out[0].credit) != OUT_CRW)
      $fatal(1, "sienna_top: the output links need data %0d bits and credit %0d, found %0d and %0d", DATA_WIDTH, OUT_CRW, $bits(out[0].data),
             $bits(out[0].credit));
`endif
  credit_link_if #(.DATA_W(SIDE_W), .CRW(1)) l0 ();  // the host link after its register stages
  credit_link_if #(.DATA_W(STG_W), .CRW(1)) l1p ();  // staging, this side of its register stages
  credit_link_if #(.DATA_W(STG_W), .CRW(1)) l1m ();  // staging, the mesh's side
  credit_link_if #(.DATA_W(L3_W), .CRW(L3_CRW)) l3m ();  // results, the mesh's side
  credit_link_if #(.DATA_W(L3_W), .CRW(L3_CRW)) l3c ();  // results, this side
  credit_link_if #(.DATA_W(1), .CRW(1)) l6 ();  // activation bank: a put when a bank holds a whole set, data the bank
  credit_link_if #(.DATA_W(1), .CRW(1)) wcm[2] ();  // the cache regions, to the mesh
  credit_reg #(.STAGES(LINK_STAGES), .DATA_W(SIDE_W), .CRW(1)) l0_reg (.clk_i(clk_i), .rstn_i(rstn_i), .up(host), .dn(l0));
  credit_reg #(.STAGES(LINK_STAGES), .DATA_W(STG_W), .CRW(1)) l1_reg (.clk_i(clk_i), .rstn_i(rstn_i), .up(l1p), .dn(l1m));
  credit_reg #(.STAGES(LINK_STAGES), .DATA_W(L3_W), .CRW(L3_CRW)) l3_reg (.clk_i(clk_i), .rstn_i(rstn_i), .up(l3m), .dn(l3c));
  for (genvar r = 0; r < 2; r++) begin : G_WC
    assign wcm[r].put = wc_region[r].put;
    assign wcm[r].data = wc_region[r].data;
    assign wc_region[r].credit = wcm[r].credit;
  end

  logic live;  // out of reset for a cycle: no consumer here advertises before it
  logic drained;  // the pipeline has been empty a while: every internal link must hold all its credits (set below)
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) live <= 1'b0;
    else live <= 1'b1;

  // L0 and L1: a host put is the set's start and goes to the mesh at once; a staging credit passes to the host while the entry admits.
  set_side_t side;
  logic [1:0] l1_cnt;  // staging credits held here or passed to the host and not yet spent
  logic l0_grant;
  assign side = l0.data;
  assign host_accept = l0.put;
  assign l1p.put = l0.put;
  assign l1p.data = {side.wc_last, side.weight_tile, side.weight_cached, side.pack_shift, side.bias_valid, side.accumulate};
  credit_counter #(.MAX(2), .CRW(1)) l1_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(l1p.put), .credit_i(l1p.credit), .has_credit_o(),
                                            .count_o(l1_cnt));
  assign entry_full = (int'(sets_out) + int'(l0_out)) >= SETS_IN_FLIGHT;
  assign l0_grant = live && (int'(l1_cnt) > int'(l0_out)) && !entry_full;  // uncast: CRW is 1 bit when SETS_IN_FLIGHT is 1
  assign l0.credit = l0_grant;

  // Rows, cache writes and bias pass as many stages as the put (L0 then L1), so a set's rows reach its bank before its put.
  localparam int BUS_STAGES = 2 * LINK_STAGES;
  localparam int BUS_W = 5 + 2 * HOST_WORDS * DATA_WIDTH + WCAW + N * ACC_W;
  logic [BUS_W-1:0] bus_in, bus_out;
  logic m_north_we, m_west_we, m_north_rst, m_west_rst, m_wc_we;
  logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] m_north, m_west;
  logic [WCAW-1:0] m_wc_addr;
  logic [N-1:0][ACC_W-1:0] m_bias;
  assign bus_in = {north_write_enable_i, west_write_enable_i, north_write_reset_i, west_write_reset_i, wc_write_enable_i,
                   north_write_data_i, west_write_data_i, wc_write_addr_i, bias_i};
  if (BUS_STAGES == 0) begin : G_BUS_WIRE
    assign bus_out = bus_in;
  end else begin : G_BUS_REGS
    localparam int CTL_W = 5;  // the enables and resets lead bus_in and are reset; the rest is datapath, not reset (D-8)
    logic [CTL_W-1:0] ctl_q[BUS_STAGES];
    logic [BUS_W-CTL_W-1:0] dat_q[BUS_STAGES];
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) for (int i = 0; i < BUS_STAGES; i++) ctl_q[i] <= '0;
      else begin
        ctl_q[0] <= bus_in[BUS_W-1-:CTL_W];
        for (int i = 1; i < BUS_STAGES; i++) ctl_q[i] <= ctl_q[i-1];
      end
    always_ff @(posedge clk_i) begin
      dat_q[0] <= bus_in[BUS_W-CTL_W-1:0];
      for (int i = 1; i < BUS_STAGES; i++) dat_q[i] <= dat_q[i-1];
    end
    assign bus_out = {ctl_q[BUS_STAGES-1], dat_q[BUS_STAGES-1]};
  end
  assign {m_north_we, m_west_we, m_north_rst, m_west_rst, m_wc_we, m_north, m_west, m_wc_addr, m_bias} = bus_out;

  // L3: the stage grants a set's PER_LANE beats while idle with a bank reserved; it takes the next result set when its beats are granted.
  logic l3_armed;  // beats granted for a set the stage has not taken yet
  logic l3_grant, l6_room, rq_room;
  logic l4_room;  // every lane holds a whole set's words of L4 credits
  logic l3_first, l3_last, l3_pk;
  logic g_fed;  // a beat of the stage's set has arrived
  // Not while a partial set waits to pass: the next result's beats could then arrive before the stage takes its set.
  assign l3_grant = live && (g_state == G_IDLE) && !l3_armed && l6_room && rq_room && l4_room && !(mesh_sets != 0 && set_accum[g_next_id]);
  assign l3c.credit = l3_grant ? L3_CRW'(PER_LANE) : '0;
  assign wide_rd_valid = l3c.put;
  assign wide_rd_data = l3c.data[NUM_LANES*OUT_W-1:0];
  assign {l3_pk, l3_last, l3_first} = l3c.data[L3_W-1-:3];

  // L6: activation bank link inside this module; the activation stage spends a credit per bank it fills, pooling returns one per release.
  logic [1:0] l6_cnt;  // banks the activation stage may fill
  logic [1:0] l6_rsv;  // of those, reserved by a grant whose set has not filled its bank yet
  logic [1:0] l6_owed;  // pooling's advertisement still to send
  logic l6_credit;
  credit_counter #(.MAX(2), .CRW(1)) l6_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(l6.put), .credit_i(l6.credit), .has_credit_o(),
                                            .count_o(l6_cnt));
  assign l6_room = l6_cnt > l6_rsv;
  assign l6.put = bank_done;
  assign l6.data = bank_sel;
  assign l6_credit = p_release || (live && l6_owed != 0);
  assign l6.credit = l6_credit;

  // Backend Arrays
  logic [      DATA_WIDTH-1:0] gpnae_signal_i     [NUM_LANES];  // L4: the word put to lane k
  logic                        gpnae_wr_en        [NUM_LANES];  // L4 put
  logic                        gpnae_last         [NUM_LANES];  // L4: the set's last word for the lane
  logic [GPNAE_ADDR_LINES-1:0] gpnae_terms        [NUM_LANES];
  logic [GPNAE_CTRL_WIDTH-1:0] gpnae_ctrl         [NUM_LANES];
  logic [      DATA_WIDTH-1:0] gpnae_result       [NUM_LANES];  // L5 data
  logic                        gpnae_done         [NUM_LANES];  // L5 put

  logic                        fifo2_wr_valid     [NUM_LANES];  // L7 put (with a 1x1 pool, dropout's input)
  logic                        fifo2_rd_ready     [NUM_LANES];  // FIFO2 pops: an L8 put
  logic [      DATA_WIDTH-1:0] fifo2_wr_data      [NUM_LANES];
  logic [      DATA_WIDTH-1:0] byp_data           [NUM_LANES];  // with a 1x1 pool: the dispatched word, straight to dropout
  logic [       NUM_LANES-1:0] byp_valid;
  logic [      DATA_WIDTH-1:0] fifo2_rd_data      [NUM_LANES];
  logic                        fifo2_rd_valid     [NUM_LANES];

  logic                        maxpool_out_valid  [NUM_LANES];  // a window's result into dropout

  logic                        dropout_in_valid   [NUM_LANES];
  logic [      DATA_WIDTH-1:0] dropout_data_in    [NUM_LANES];
  logic [      DATA_WIDTH-1:0] dropout_data_out   [NUM_LANES];
  logic                        dropout_valid_out  [NUM_LANES];

  logic [          FCNT_W-1:0] fill_count         [NUM_LANES];
  logic [          FCNT_W-1:0] done_count         [NUM_LANES];
  logic                        load_finalized     [NUM_LANES];
  logic                        lane_collected     [NUM_LANES];
  logic                        lane_collected_n   [NUM_LANES];

  logic [           TOT_W-1:0] total_elements;
  logic [           TOT_W-1:0] filled_total;

  logic [          FCNT_W-1:0] fill_count_n       [NUM_LANES];
  logic [          FCNT_W-1:0] done_count_n       [NUM_LANES];
  logic                        load_finalized_n   [NUM_LANES];
  logic                        gpnae_last_n       [NUM_LANES];
  logic                        gpnae_wr_en_n      [NUM_LANES];
  logic [      DATA_WIDTH-1:0] gpnae_signal_n     [NUM_LANES];
  logic [           TOT_W-1:0] filled_total_n;

  logic [$clog2(FIFO2_DEPTH):0] fifo2_count[NUM_LANES];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      total_elements <= 0;
    end else begin
      if (g_state == G_IDLE) total_elements <= 0;
      else if (g_state == G_LATCH) total_elements <= SRAM_DEPTH[TOT_W-1:0];
    end
  end

  logic all_collected;
  // Every lane gets PER_LANE elements, so a set is done only when every lane has returned all of them.
  logic all_lanes_collected;
  always_comb begin
    all_lanes_collected = 1'b1;
    for (int i = 0; i < NUM_LANES; i++) if (!lane_collected[i]) all_lanes_collected = 1'b0;
  end
  assign all_collected = (filled_total >= total_elements) && all_lanes_collected && (total_elements > 0);

  SystolicMesh #(
      .MATRIX_SIZE(N),
      .TILE_SIZE  (TILE_SIZE),
      .EXP_W      (EXP_W),
      .MAN_W      (MAN_W),
      .DATA_WIDTH (DATA_WIDTH),
      .WIDE_READ  (NUM_LANES),
      .HOST_WORDS (HOST_WORDS),
      .COLLAPSE_K (COLLAPSE_K),
      .WC_TILES   (WC_TILES),
      .OUT_W      (OUT_W),
      .ACC_BANKS  (ACC_BANKS),
      .RESULT_BANKS(RESULT_BANKS),
      .RES_MAX    (PER_LANE),
      .RES_CRW    (L3_CRW)
  ) systolic_array_inst (
      .clk_i                 (clk_i),
      .rstn_i                (rstn_i),
      .staging               (l1m),
      .bias_i                (m_bias),
      .wc_region             (wcm),
      .wc_write_enable_i     (m_wc_we),
      .wc_write_addr_i       (m_wc_addr),
      .north_write_enable_i  (m_north_we),
      .north_write_data_i    (m_north),
      .north_write_reset_i   (m_north_rst),
      .west_write_enable_i   (m_west_we),
      .west_write_data_i     (m_west),
      .west_write_reset_i    (m_west_rst),
      .north_queue_empty_o   (north_queue_empty),
      .west_queue_empty_o    (west_queue_empty),
      .matrix_mult_complete_o(systolic_mult_complete),
      .collection_active_o   (),
      .result                (l3m)
  );

  // L4, L5 per lane: the stage spends an L4 credit per word; the collector grants LANE_OUT_SLOTS and returns each the cycle after its result.
  localparam int L5W = $clog2(LANE_OUT_SLOTS + 1);
  localparam int L5_MAX = (LANE_OUT_SLOTS > LANE_K) ? LANE_OUT_SLOTS : LANE_K;  // the lane's out counter; never below K, which it checks
  logic [GPNAE_ADDR_LINES:0] l4_cnt[NUM_LANES];  // L4 credits the stage holds per lane
  logic                      l4_cr [NUM_LANES];  // L4 credit, lane to stage
  logic                      l5_cr [NUM_LANES];  // L5 credit, collector to lane
  logic [L5W-1:0]            l5_owed[NUM_LANES];  // the collector's advertisement still to send
  logic [L5W-1:0]            l5_out[NUM_LANES];  // L5 credits the lane holds
  logic [NUM_LANES-1:0][GPNAE_ADDR_LINES:0] lane_outst;  // words put to a lane whose results are not back
  logic [NUM_LANES-1:0][FCNT_W-1:0] l4_n;  // L4 puts per lane for the stage's set
  logic [NUM_LANES-1:0] l5_want;  // the lane's result is due: the stage holds its set and not all results are in
  // L7 (L9 through dropout with a 1x1 pool): the dispatcher's words into each pooling lane, a credit counter per lane.
  logic [15:0]               pin_cnt[NUM_LANES];
  logic                      pin_cr [NUM_LANES];
  logic [NUM_LANES-1:0]      pin_room;  // the lane takes the next dispatched word: a credit beyond the put now in flight

  always_comb begin
    l4_room = 1'b1;
    for (int i = 0; i < NUM_LANES; i++) if (int'(l4_cnt[i]) < PER_LANE) l4_room = 1'b0;
    for (int i = 0; i < NUM_LANES; i++) begin
      l5_cr[i]   = live && (l5_owed[i] != '0);  // the advertisement, then each result's slot the cycle after it is written
      l5_want[i] = (g_state != G_IDLE) && (done_count[i] < fill_count[i]);
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      lane_outst <= '0;
      l4_n       <= '0;
      for (int i = 0; i < NUM_LANES; i++) begin
        l5_owed[i] <= L5W'(LANE_OUT_SLOTS);
        l5_out[i]  <= '0;
      end
    end else begin
      for (int i = 0; i < NUM_LANES; i++) begin
        l5_owed[i]    <= l5_owed[i] + L5W'(gpnae_done[i]) - L5W'(l5_cr[i]);
        l5_out[i]     <= l5_out[i] + L5W'(l5_cr[i]) - L5W'(gpnae_done[i]);
        lane_outst[i] <= lane_outst[i] + (GPNAE_ADDR_LINES + 1)'(gpnae_wr_en[i]) - (GPNAE_ADDR_LINES + 1)'(gpnae_done[i]);
        if (g_state == G_IDLE) l4_n[i] <= '0;
        else if (gpnae_wr_en[i]) l4_n[i] <= l4_n[i] + 1'b1;
      end
    end
  end

  generate
    genvar g;
    for (g = 0; g < NUM_LANES; g++) begin : backend_lanes

      credit_link_if #(.DATA_W(DATA_WIDTH + 1), .CRW(1)) l4 ();  // L4: {last, word}
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(1)) l5 ();  // L5: one result per put
      assign l4.put  = gpnae_wr_en[g];
      assign l4.data = {gpnae_last[g], gpnae_signal_i[g]};
      assign l4_cr[g] = l4.credit;
      credit_counter #(.MAX(GPNAE_FIFO_DEPTH), .CRW(1)) l4_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(l4.put), .credit_i(l4.credit),
                                                               .has_credit_o(), .count_o(l4_cnt[g]));
      assign gpnae_done[g]   = l5.put;
      assign gpnae_result[g] = l5.data;
      assign l5.credit       = l5_cr[g];

      gpnae_poly #(
          .EXP_W        (EXP_W),
          .MAN_W        (MAN_W),
          .DATA_WIDTH   (GPNAE_DATA_WIDTH),
          .ADDR_LINES   (GPNAE_ADDR_LINES),
          .CONTROL_WIDTH(GPNAE_CTRL_WIDTH),
          .OUT_MAX      (L5_MAX),
          .OUT_CRW      (1)
      ) gpnae_inst (
          .clk_i         (clk_i),
          .rstn_i        (rstn_i),
          .in            (l4),
          .out           (l5),
          .terms_i       (gpnae_terms[g]),
          .control_word_i(gpnae_ctrl[g]),
          .gp_mx_i       (g_mx[g]),
          .gp_shx_i      (g_shx[g]),
          .gp_zin_i      (g_zp[g]),  // int8: the lane's input is the requantize output, whose zero point is its entry's
          .gp_mout_i     (g_mout[g]),
          .gp_shout_i    (g_shout[g]),
          .gp_zout_i     (g_zout[g])
      );

      // L7 into FIFO2, or with a 1x1 pool straight into dropout on L9's credits; L8 FIFO2 -> maxpool -> dropout; L9 out.
      localparam int PIN_MAX = POOL_BYPASS ? OUT_MAX : FIFO2_DEPTH;
      localparam int PIN_CRW = POOL_BYPASS ? OUT_CRW : 1;
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(PIN_CRW)) pin ();  // the dispatcher's words
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) drp ();  // into dropout: window results, or pin itself
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) l9p ();  // out of dropout, into the output register
      credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) l9q ();  // out of the output register, into L9's register stages
      logic [$clog2(PIN_MAX + 1)-1:0] pin_n;
      assign pin.put  = POOL_BYPASS ? byp_valid[g] : fifo2_wr_valid[g];
      assign pin.data = POOL_BYPASS ? byp_data[g] : fifo2_wr_data[g];
      assign pin_cr[g] = pin.credit != '0;
      credit_counter #(.MAX(PIN_MAX), .CRW(PIN_CRW)) pin_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(pin.put), .credit_i(pin.credit),
                                                             .has_credit_o(), .count_o(pin_n));
      assign pin_cnt[g]  = 16'(pin_n);
      assign pin_room[g] = int'(pin_n) > int'(pin.put);  // the dispatcher's put is registered: one credit is already spoken for

      if (POOL_BYPASS) begin : G_NOPOOL
        assign drp.put    = pin.put;
        assign drp.data   = pin.data;
        assign pin.credit = drp.credit;
        assign fifo2_rd_valid[g]    = 1'b0;
        assign fifo2_rd_ready[g]    = 1'b0;
        assign fifo2_rd_data[g]     = '0;
        assign fifo2_count[g]       = '0;
        assign maxpool_out_valid[g] = 1'b0;
      end else begin : G_POOL
        credit_link_if #(.DATA_W(DATA_WIDTH), .CRW(MP_CRW)) l8 ();  // FIFO2 -> maxpool, a window's credits at once
        fwft #(
            .DATA_WIDTH(DATA_WIDTH),
            .FIFO_DEPTH(FIFO2_DEPTH),
            .OUT_MAX   (MP_AHEAD * MP_WIN),
            .OUT_CRW   (MP_CRW)
        ) fifo2_inst (
            .clk_i  (clk_i),
            .rstn_i (rstn_i),
            .in     (pin),
            .out    (l8),
            .count_o(fifo2_count[g])
        );
        assign fifo2_rd_valid[g] = l8.put;
        assign fifo2_rd_ready[g] = l8.put;
        assign fifo2_rd_data[g]  = l8.data;

        Maxpool_2D #(
            .DATA_WIDTH (DATA_WIDTH),
            .IN_ROWS    (POOL_H),
            .IN_COLS    (POOL_W),
            .SEG_ROWS   (POOL_H),
            .SEG_COLS   (POOL_W),
            .STRIDE_ROWS(POOL_H),
            .STRIDE_COLS(POOL_W),
            .PADDING    (0),
            .IS_FP32    (!IS_INT),
            .EXP_W      (EXP_W),
            .MAN_W      (MAN_W),
            .AHEAD      (MP_AHEAD),
            .OUT_MAX    (OUT_MAX),
            .OUT_CRW    (OUT_CRW),
            .IN_CRW     (MP_CRW)
        ) maxpool_inst (
            .clk  (clk_i),
            .rst_n(rstn_i),
            .in   (l8),
            .out  (drp)
        );
        assign maxpool_out_valid[g] = drp.put;
      end

      // A nonlinear mix, since an XOR-only one makes the 16 lanes' masks linearly tied.
      logic [31:0] lane_mul;
      logic [LFSR_WIDTH-1:0] lane_mix, lane_seed;
      assign lane_mul  = (32'(set_seed[p_next_id]) ^ (32'h9E3779B9 * (g + 1))) * 32'h85EBCA6B;
      assign lane_mix  = LFSR_WIDTH'(lane_mul ^ (lane_mul >> 16));
      assign lane_seed = (lane_mix == '0) ? '1 : lane_mix;  // an all-zero LFSR state would lock up

      assign dropout_in_valid[g] = drp.put;
      assign dropout_data_in[g]  = drp.data;
      dropout #(
          .EXP_W            (EXP_W),
          .MAN_W            (MAN_W),
          .DATA_WIDTH       (DATA_WIDTH),
          .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
          .LFSR_WIDTH       (LFSR_WIDTH)
      ) dropout_inst (
          .clk          (clk_i),
          .rst_n        (rstn_i),
          .in           (drp),
          .out          (l9p),
          .training_mode(set_train[p_set_id]),
          .reseed_i     (p_accept),
          .seed_i       (lane_seed),
          .zero_point_i (p_zp[g])
      );
      assign dropout_valid_out[g] = l9p.put;
      assign dropout_data_out[g]  = l9p.data;

      // The output register, as final_result_o was; L9's credits pass it unregistered, which only spends them earlier.
      logic                  oq_put;
      logic [DATA_WIDTH-1:0] oq_data;
      always_ff @(posedge clk_i or negedge rstn_i)
        if (!rstn_i) begin
          oq_put  <= 1'b0;
          oq_data <= '0;
        end else begin
          oq_put <= l9p.put;
          if (l9p.put) oq_data <= l9p.data;
        end
      assign l9q.put    = oq_put;
      assign l9q.data   = oq_data;
      assign l9p.credit = l9q.credit;
      credit_reg #(.STAGES(LINK_STAGES), .DATA_W(DATA_WIDTH), .CRW(OUT_CRW)) l9_reg (.clk_i(clk_i), .rstn_i(rstn_i), .up(l9q), .dn(out[g]));

`ifndef SYNTHESIS
      credit_link_checker #(.SLOTS(GPNAE_FIFO_DEPTH)) l4_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(l4));
      credit_link_checker #(.SLOTS(LANE_OUT_SLOTS)) l5_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(l5));
      if (!POOL_BYPASS) begin : G_L7_CHK
        credit_link_checker #(.SLOTS(FIFO2_DEPTH)) l7_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(pin));
      end
`endif
    end
  endgenerate

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      gpnae_terms[i] = set_terms[g_set_id][GPNAE_ADDR_LINES-1:0];
      gpnae_ctrl[i] = (IS_INT && lane_act[i] == CONTROL_WIDTH'(3'b100) && !act_bypass) ? CONTROL_WIDTH'(3'b101) : lane_act[i];  // int8 ReLU is the clamp's
    end
  end

  // ReLU and linear need no polynomial: when every lane's code is one of them, each beat goes straight into its bank and the lanes stay idle.
  always_comb begin
    act_bypass = 1'b1;
    for (int k = 0; k < NUM_LANES; k++) begin
      lane_act[k] = g_ents[lane_ent[k]];
      act_bypass &= is_byp_code(lane_act[k]);
    end
  end

  // =========================================================================
  // GPNAE TO CENTRAL BUFFER WRITE LOGIC
  // =========================================================================
  always_ff @(posedge clk_i) begin
    // Beat b of the wide read holds element elem(k, b) in word k, the element lane k would have taken.
    if (!IS_INT && g_state != G_IDLE && act_bypass && fill_v)
      for (int k = 0; k < NUM_LANES; k++)
        gpnae_out_mem[act_wr_base + elem(k, int'(fill_count[k]), g_pack)] <=
            (lane_act[k] == CONTROL_WIDTH'(3'b100) && fill_d[k][DATA_WIDTH-1]) ? '0 : fill_d[k];
    if (IS_INT && byp_wr)  // int8: a beat lands where its own tag says, so it may leave the requantize pipeline after its set left the stage
      for (int k = 0; k < NUM_LANES; k++) gpnae_out_mem[(byp_bank ? SRAM_DEPTH : 0) + elem(k, int'(byp_idx), byp_pack)] <= fill_d[k];
    // L5 collector: a lane's result is written to the bank the cycle it arrives, so its credit goes back the next cycle.
    for (int i = 0; i < NUM_LANES; i++)
      if (gpnae_done[i] && l5_want[i]) gpnae_out_mem[act_wr_base + elem(i, int'(done_count[i]), g_pack)] <= gpnae_result[i];
  end

  // =========================================================================
  // WINDOW DISPATCHER FSM
  // =========================================================================
  // Windows go to lanes round-robin, so NUM_LANES of them can be dispatched at once: lane L
  // takes windows L, L+NUM_LANES, ... exactly as before, but all lanes are written in the same
  // cycle instead of one per cycle. gpnae_out_mem is a register array, so the parallel reads
  // are free. Row and column are carried per lane rather than divided out of a window index,
  // which would cost NUM_LANES dividers by a non-power-of-two.
  localparam int NUM_GROUPS = (MAXPOOL_OUT_COUNT + NUM_LANES - 1) / NUM_LANES;
  localparam int ADV_R = NUM_LANES / POOL_OUT_COLS;  // rows and columns a lane moves by per group of windows
  localparam int ADV_C = NUM_LANES % POOL_OUT_COLS;

  logic [        $clog2(NUM_GROUPS+1)-1:0] disp_g;
  logic [           $clog2(POOL_H+1)-1:0] disp_pr;
  logic [           $clog2(POOL_W+1)-1:0] disp_pc;
  logic                                   disp_done;

  logic [    $clog2(POOL_OUT_ROWS+1)-1:0] lane_r     [NUM_LANES];
  logic [    $clog2(POOL_OUT_COLS+1)-1:0] lane_c     [NUM_LANES];
  logic [$clog2(MAXPOOL_OUT_COUNT+1)-1:0] lane_win   [NUM_LANES];
  logic [                  NUM_LANES-1:0] lane_active;

  logic [DATA_WIDTH-1:0] lane_val[NUM_LANES];
  logic                  disp_can_write;

  always_comb begin
    for (int L = 0; L < NUM_LANES; L++) begin
      automatic logic signed [31:0] ir = signed'(lane_r[L] * STRIDE_ROWS + disp_pr) - signed'(PADDING);
      automatic logic signed [31:0] ic = signed'(lane_c[L] * STRIDE_COLS + disp_pc) - signed'(PADDING);
      lane_active[L] = (lane_win[L] < MAXPOOL_OUT_COUNT);
      lane_val[L] = ((ir >= 0) && (ir < IN_ROWS) && (ic >= 0) && (ic < IN_COLS))
                    ? gpnae_out_mem[act_rd_base+ir*IN_COLS+ic] : NEG_INF;
    end
  end

  // One active lane without an L7 credit (L9 with a 1x1 pool) stalls the whole group, which keeps every lane on the same element.
  always_comb begin
    disp_can_write = 1'b1;
    for (int L = 0; L < NUM_LANES; L++)
      if (lane_active[L] && !pin_room[L]) disp_can_write = 1'b0;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i || p_state == P_IDLE) begin
      disp_g   <= '0;
      disp_pr  <= '0;
      disp_pc  <= '0;
      disp_done <= '0;
      for (int L = 0; L < NUM_LANES; L++) begin
        lane_win[L] <= L[$clog2(MAXPOOL_OUT_COUNT+1)-1:0];
        lane_r[L]   <= (L / POOL_OUT_COLS);
        lane_c[L]   <= (L % POOL_OUT_COLS);
      end
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;
      byp_valid <= '0;
    end else if (p_state == P_DISPATCH) begin
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;
      byp_valid <= '0;

      if (!disp_done && disp_can_write) begin
        for (int L = 0; L < NUM_LANES; L++) begin
          if (lane_active[L]) begin
            if (POOL_BYPASS) begin
              byp_data[L]  <= lane_val[L];
              byp_valid[L] <= 1'b1;
            end else begin
              fifo2_wr_data[L]  <= lane_val[L];
              fifo2_wr_valid[L] <= 1'b1;
            end
          end
        end

        if (disp_pc < POOL_W - 1) begin
          disp_pc <= disp_pc + 1;
        end else begin
          disp_pc <= '0;
          if (disp_pr < POOL_H - 1) begin
            disp_pr <= disp_pr + 1;
          end else begin
            disp_pr <= '0;
            if (disp_g < NUM_GROUPS - 1) begin
              disp_g <= disp_g + 1;
              // Advance every lane by NUM_LANES windows: whole rows plus a column step, wrapping at most once
              // since a lane's column is below POOL_OUT_COLS; one add and compare, not a loop of subtractions.
              for (int L = 0; L < NUM_LANES; L++) begin
                automatic int c_tmp = lane_c[L] + ADV_C;
                automatic int r_tmp = lane_r[L] + ADV_R + ((c_tmp >= POOL_OUT_COLS) ? 1 : 0);
                if (c_tmp >= POOL_OUT_COLS) c_tmp = c_tmp - POOL_OUT_COLS;
                lane_c[L]   <= c_tmp[$clog2(POOL_OUT_COLS+1)-1:0];
                lane_r[L]   <= r_tmp[$clog2(POOL_OUT_ROWS+1)-1:0];
                lane_win[L] <= lane_win[L] + NUM_LANES;
              end
            end else begin
              disp_done <= 1'b1;
            end
          end
        end
      end
    end else begin
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;
      byp_valid <= '0;
    end
  end

  // =========================================================================
  // STAGE CONTROLLERS: the mesh, activation (G) and pooling (P) each hold one set
  // =========================================================================
  logic streaming_complete;

  // The stage takes the next real set once its beats are granted; the beats follow when the mesh has the result.
  assign g_accept = (g_state == G_IDLE) && l3_armed && (mesh_sets != 0) && !set_accum[g_next_id];
  assign g_done = (g_state == G_ROUND) && (act_bypass ? byp_all_in : all_collected);
  logic g_null_done;  // a partial set, summed in the PEs, passes with no result and no activation bank
  assign g_null_done = (g_state == G_IDLE) && (mesh_sets != 0) && set_accum[g_next_id] && !rq_drain;
  // Pooling completes sets in id order: a partial in one cycle, never right after another completion, so pulses stay one cycle.
  logic complete_q, p_null;
  logic set_complete;  // a set completes this cycle, with its last word into the output register
  logic [ID_W-1:0] set_done_id;
  assign p_null = (p_state == P_IDLE) && (gp_sets != 0) && set_accum[p_next_id] && !complete_q;
  assign p_accept = (p_state == P_IDLE) && (gp_sets != 0) && !set_accum[p_next_id] && act_full[act_rd];
  assign p_release = (p_state == P_DISPATCH) && disp_done;  // the bank is copied into FIFO2
  assign pool_done = (p_state == P_WAIT) && streaming_complete;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      g_state   <= G_IDLE;
      p_state   <= P_IDLE;
      act_full  <= '0;
      gp_sets   <= '0;
      complete_q <= 1'b0;
      act_wr    <= 1'b0;
      act_rd    <= 1'b0;
      sets_out  <= '0;
      l0_out    <= '0;
      l3_armed  <= 1'b0;
      l6_rsv    <= '0;
      l6_owed   <= 2'd2;
      g_fed     <= 1'b0;
      mesh_sets <= '0;
      g_next_id <= '0;
      g_set_id  <= '0;
      p_next_id <= '0;
      p_set_id  <= '0;
      host_next_id <= '0;
      g_pack    <= 1'b0;
      g_ents    <= '0;
      for (int k = 0; k < NUM_IDS; k++) begin
        set_train[k] <= 1'b0;
        set_seed[k]  <= '1;
        set_accum[k] <= 1'b0;
        set_terms[k] <= '0;
        set_pack[k]  <= '0;
        set_map[k]   <= '0;
        set_ents[k]  <= '0;
      end
      for (int k = 0; k < NUM_LANES; k++) begin
        lane_ent[k]   <= '0;
        p_lane_ent[k] <= '0;
      end
    end else begin
      complete_q <= set_complete;
      if (host_accept) begin
        set_accum[host_next_id] <= side.accumulate;
        set_terms[host_next_id] <= side.terms;
        set_pack[host_next_id] <= side.pack_shift;
        set_map[host_next_id]  <= side.pack_map;
        set_ents[host_next_id] <= side.act;
        set_train[host_next_id] <= side.train;
        set_seed[host_next_id]  <= side.seed;
        host_next_id <= host_next_id + 1'b1;
      end
      if (l3_grant) l3_armed <= 1'b1;
      else if (g_accept) l3_armed <= 1'b0;
      if (g_state == G_IDLE) g_fed <= 1'b0;
      else if (wide_rd_valid) g_fed <= 1'b1;
      l6_rsv  <= l6_rsv + 2'(l3_grant) - 2'(l6.put);
      l6_owed <= l6_owed + 2'(p_release) - 2'(l6_credit);
      case (g_state)
        G_IDLE:     if (g_accept) g_state <= G_FEED;
        G_FEED:     g_state <= G_LATCH;
        G_LATCH:    g_state <= G_ROUND;
        G_ROUND:    if (g_done) g_state <= G_IDLE;
        default:    g_state <= G_IDLE;
      endcase
      case (p_state)
        P_IDLE:     if (p_accept) p_state <= P_DISPATCH;
        P_DISPATCH: if (disp_done) p_state <= P_WAIT;
        P_WAIT:     if (streaming_complete) p_state <= P_IDLE;
        default:    p_state <= P_IDLE;
      endcase
      if (l6.put) act_full[l6.data] <= 1'b1;  // pooling, the L6 consumer, owns the bank flags
      if (g_done) act_wr <= ~act_wr;
      if (p_release) begin
        act_full[act_rd] <= 1'b0;
        act_rd <= ~act_rd;
      end
      gp_sets <= gp_sets + CRW'(bank_done || g_null_done) - CRW'(set_complete);
      sets_out  <= sets_out + CRW'(host_accept) - CRW'(set_complete);
      l0_out    <= l0_out + CRW'(l0_grant) - CRW'(host_accept);
      mesh_sets <= mesh_sets + CRW'(host_accept) - CRW'(g_accept || g_null_done);
      if (g_accept) g_set_id <= g_next_id;
      if (g_accept) begin
        g_pack <= (set_pack[g_next_id] != '0);
        g_ents <= set_ents[g_next_id];
        for (int k = 0; k < NUM_LANES; k++) lane_ent[k] <= ent_of(set_map[g_next_id], set_pack[g_next_id], k % N);
      end
      if (p_accept)
        for (int k = 0; k < NUM_LANES; k++) p_lane_ent[k] <= ent_of(set_map[p_next_id], set_pack[p_next_id], k % N);
      if (g_accept || g_null_done) g_next_id <= g_next_id + 1'b1;
      if (p_accept || p_null) begin
        p_set_id  <= p_next_id;
        p_next_id <= p_next_id + 1'b1;
      end
    end
  end

  // int8 (D-2): parameters per set id; the stages copy their set's words and entries at accept, so a lane picks among N or 8, not NUM_IDS times that.
  if (IS_INT) begin : G_REQ_SETS
    logic [N-1:0][31:0] s_mult [NUM_IDS];
    logic [N-1:0][7:0]  s_shift[NUM_IDS];
    logic [PACK_ENTRIES-1:0][7:0]  s_zp[NUM_IDS], s_min[NUM_IDS], s_max[NUM_IDS], s_shout[NUM_IDS], s_zout[NUM_IDS];
    logic [PACK_ENTRIES-1:0][15:0] s_mx[NUM_IDS];
    logic [PACK_ENTRIES-1:0][4:0]  s_shx[NUM_IDS];
    logic [PACK_ENTRIES-1:0][31:0] s_mout[NUM_IDS];
    logic [PACK_ENTRIES-1:0][7:0]  ge_zp, ge_min, ge_max, ge_shout, ge_zout;  // the activation stage's copy of its set's entries
    logic [PACK_ENTRIES-1:0][15:0] ge_mx;
    logic [PACK_ENTRIES-1:0][4:0]  ge_shx;
    logic [PACK_ENTRIES-1:0][31:0] ge_mout;
    logic [PACK_ENTRIES-1:0][CONTROL_WIDTH-1:0] pe_act;  // the pooling stage's copy of its set's codes, zp and zout, for the drop value
    logic [PACK_ENTRIES-1:0][7:0]  pe_zp, pe_zout;
    always_ff @(posedge clk_i) begin  // no reset: an id's entry is written by its own accept before any stage reads it
      if (host_accept) begin
        s_mult[host_next_id]  <= side.mult;
        s_shift[host_next_id] <= side.shift;
        s_zp[host_next_id]    <= side.zp;
        s_min[host_next_id]   <= side.amin;
        s_max[host_next_id]   <= side.amax;
        s_mx[host_next_id]    <= side.mx;
        s_shx[host_next_id]   <= side.shx;
        s_mout[host_next_id]  <= side.mout;
        s_shout[host_next_id] <= side.shout;
        s_zout[host_next_id]  <= side.zout;
      end
      if (g_accept) begin
        g_mult   <= s_mult[g_next_id];
        g_shift  <= s_shift[g_next_id];
        ge_zp    <= s_zp[g_next_id];
        ge_min   <= s_min[g_next_id];
        ge_max   <= s_max[g_next_id];
        ge_mx    <= s_mx[g_next_id];
        ge_shx   <= s_shx[g_next_id];
        ge_mout  <= s_mout[g_next_id];
        ge_shout <= s_shout[g_next_id];
        ge_zout  <= s_zout[g_next_id];
      end
      if (p_accept || p_null) begin  // the edge p_set_id loads on
        pe_zp   <= s_zp[p_next_id];
        pe_zout <= s_zout[p_next_id];
      end
    end
    always_ff @(posedge clk_i or negedge rstn_i)  // reset, as set_ents is, so p_zp reads tanh's 0 before the first pooled set
      if (!rstn_i) pe_act <= '0;
      else if (p_accept || p_null) pe_act <= set_ents[p_next_id];
    for (genvar k = 0; k < NUM_LANES; k++) begin : G_LANE_PAR
      assign g_zp[k]    = ge_zp[lane_ent[k]];
      assign g_min[k]   = ge_min[lane_ent[k]];
      assign g_max[k]   = ge_max[lane_ent[k]];
      assign g_mx[k]    = ge_mx[lane_ent[k]];
      assign g_shx[k]   = ge_shx[lane_ent[k]];
      assign g_mout[k]  = ge_mout[lane_ent[k]];
      assign g_shout[k] = ge_shout[lane_ent[k]];
      assign g_zout[k]  = ge_zout[lane_ent[k]];
      // D-5: a dropped value is the output zero point of the pooled column's activation (tanh, and every code the lane runs as tanh: 0).
      always_comb
        case (pe_act[p_lane_ent[k]])
          CONTROL_WIDTH'(3'b001): p_zp[k] = pe_zout[p_lane_ent[k]];  // SELU: its requantized output's zero point
          CONTROL_WIDTH'(3'b010): p_zp[k] = 8'h80;  // sigmoid: TFLite's fixed output zero point -128
          CONTROL_WIDTH'(3'b100), CONTROL_WIDTH'(3'b101): p_zp[k] = pe_zp[p_lane_ent[k]];  // ReLU, linear: the requantize output's
          default: p_zp[k] = 8'h00;  // tanh: zero point 0
        endcase
    end
  end else begin : G_NO_REQ_SETS
    assign g_mult  = '0;
    assign g_shift = '0;
    assign g_zp    = '0;
    assign g_min   = '0;
    assign g_max   = '0;
    assign g_mx    = '0;
    assign g_shx   = '0;
    assign g_mout  = '0;
    assign g_shout = '0;
    assign g_zout  = '0;
    assign p_zp    = '0;
  end

  // =========================================================================
  // PARALLEL LANE FILL: each result beat gives every lane its next element at once
  // =========================================================================
  logic fill_v;
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] fill_d;
  // int8: the lanes, the bypass write and every fill counter see requantized int8 beats, REQ_LAT cycles after the wide read.
  if (IS_INT) begin : G_REQ
    requant_lanes #(
        .NUM_LANES(NUM_LANES),
        .N        (N),
        .PER_LANE (PER_LANE),
        .ROUNDING (sienna_fmt_pkg::REQ_ROUNDING)
    ) rq (
        .clk_i   (clk_i),
        .rstn_i  (rstn_i),
        .clear_i (g_state == G_IDLE),
        .valid_i (wide_rd_valid),
        .acc_i   (wide_rd_data),
        .mult_i  (g_mult),
        .shift_i (g_shift),
        .packed_i(g_pack),
        .zp_i    (g_zp),
        .min_i   (g_min),
        .max_i   (g_max),
        .valid_o (fill_v),
        .result_o(fill_d)
    );
    // Each beat's destination rides beside tfliteRequant, which takes its own mult, shift, zp, min and max at stage 1.
    localparam int RQL = sienna_fmt_pkg::req_lat();
    logic [FCNT_W-1:0] rq_in;  // beats of the stage's set that entered the pipeline
    logic [RQL-1:0] tg_v;  // a beat at each pipeline stage
    logic [RQL-1:0] tg_byp, tg_bank;
    logic [RQL-1:0] tg_pack;
    logic [FCNT_W-1:0] tg_idx[RQL];
    always_ff @(posedge clk_i or negedge rstn_i) begin
      if (!rstn_i) begin
        rq_in <= '0;
        tg_v  <= '0;
      end else begin
        if (g_state == G_IDLE) rq_in <= '0;
        else if (wide_rd_valid) rq_in <= rq_in + 1'b1;
        tg_v <= {tg_v[RQL-2:0], wide_rd_valid};
      end
    end
    always_ff @(posedge clk_i) begin  // no reset: read only beside tg_v / fill_v
      tg_byp  <= {tg_byp[RQL-2:0], act_bypass};
      tg_bank <= {tg_bank[RQL-2:0], act_wr};
      tg_pack <= {tg_pack[RQL-2:0], g_pack};
      tg_idx[0] <= rq_in;
      for (int i = 1; i < RQL; i++) tg_idx[i] <= tg_idx[i-1];
    end
    assign byp_all_in = (rq_in == PER_LANE[FCNT_W-1:0]);  // the last beat is in: the stage may leave while it drains
    assign rq_drain   = |tg_v;
    // Beats granted now arrive 2 or more cycles later (counter, then the mesh's registered read): grant once no beat would still be inside then.
    localparam int RQ_HOLD = (RQL > 2) ? RQL - 2 : 0;
    always_comb begin
      rq_room = 1'b1;
      for (int i = 0; i < RQ_HOLD; i++) if (tg_v[i]) rq_room = 1'b0;
    end
    assign lane_v     = fill_v && !tg_byp[RQL-1];
    assign byp_wr     = fill_v && tg_byp[RQL-1];
    assign byp_bank   = tg_bank[RQL-1];
    assign byp_pack   = tg_pack[RQL-1];
    assign byp_idx    = tg_idx[RQL-1];
    // The bank is full when a ReLU or linear set's last beat leaves the pipeline, or when the lanes return a set.
    assign bank_done  = (byp_wr && (tg_idx[RQL-1] == PER_LANE[FCNT_W-1:0] - 1'b1)) || (g_done && !act_bypass);
    assign bank_sel   = byp_wr ? tg_bank[RQL-1] : act_wr;
`ifndef SYNTHESIS
    a_rq_tag_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) fill_v == tg_v[RQL-1])
      else $error("sienna_top: the requantize sideband is out of step with requant_lanes");
    a_rq_bank_once: assert property (@(posedge clk_i) disable iff (!rstn_i) !(byp_wr && g_done && !act_bypass))
      else $error("sienna_top: a drained ReLU or linear set and a lane set completed banks in the same cycle");
    a_rq_bank_empty: assert property (@(posedge clk_i) disable iff (!rstn_i) bank_done |-> !act_full[bank_sel])
      else $error("sienna_top: a set completed into a full activation bank");
    a_rq_one_set: assert property (@(posedge clk_i) disable iff (!rstn_i) (wide_rd_valid && rq_in == '0) |-> !rq_drain)
      else $error("sienna_top: a set's first beat entered the requantize pipeline while an earlier beat was still in it");
`endif
  end else begin : G_NO_REQ
    if ($bits(fill_d[0]) != OUT_W) begin : G_BAD_FILL  // float lanes take the mesh's result words as they are
      $fatal(1, "sienna_top: the lanes take %0d-bit words, the mesh gives %0d-bit results", $bits(fill_d[0]), OUT_W);
    end
    assign fill_v     = wide_rd_valid;
    assign fill_d     = wide_rd_data;
    assign lane_v     = fill_v;
    assign byp_wr     = 1'b0;
    assign byp_bank   = 1'b0;
    assign byp_pack   = 1'b0;
    assign byp_idx    = '0;
    assign byp_all_in = (fill_count[0] == PER_LANE[FCNT_W-1:0]);
    assign rq_drain   = 1'b0;
    assign rq_room    = 1'b1;
    assign bank_done  = g_done;
    assign bank_sel   = act_wr;
  end

  always_comb begin
    filled_total_n = filled_total;
    for (int i = 0; i < NUM_LANES; i++) begin
      fill_count_n[i]     = fill_count[i];
      done_count_n[i]     = done_count[i];
      load_finalized_n[i] = load_finalized[i];
      gpnae_last_n[i]     = 1'b0;
      gpnae_wr_en_n[i]    = 1'b0;
      gpnae_signal_n[i]   = gpnae_signal_i[i];
      lane_collected_n[i] = lane_collected[i];
    end

    if (g_state == G_IDLE) begin
      filled_total_n = '0;
      for (int i = 0; i < NUM_LANES; i++) begin
        fill_count_n[i]     = '0;
        done_count_n[i]     = '0;
        load_finalized_n[i] = 1'b0;
        lane_collected_n[i] = 1'b0;
      end
    end else begin
      if (lane_v) begin
        filled_total_n = filled_total + NUM_LANES[TOT_W-1:0];
        for (int i = 0; i < NUM_LANES; i++) begin
          gpnae_signal_n[i] = fill_d[i];
          gpnae_wr_en_n[i]  = !act_bypass;
          gpnae_last_n[i]   = (fill_count[i] + 1'b1) == PER_LANE[FCNT_W-1:0];  // L4: last rides the set's final put
          fill_count_n[i]   = fill_count[i] + 1'b1;
        end
      end
      // Every lane holds its whole set from the cycle after its last word is put.
      for (int i = 0; i < NUM_LANES; i++)
        if ((fill_count[i] == PER_LANE[FCNT_W-1:0]) && !load_finalized[i] && !act_bypass) load_finalized_n[i] = 1'b1;
      for (int i = 0; i < NUM_LANES; i++) begin
        if (gpnae_done[i] && l5_want[i]) begin
          done_count_n[i] = done_count[i] + 1'b1;
          if ((done_count[i] + 1'b1) == fill_count[i]) lane_collected_n[i] = 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      filled_total <= '0;
      for (int i = 0; i < NUM_LANES; i++) begin
        fill_count[i]     <= '0;
        done_count[i]     <= '0;
        load_finalized[i] <= 1'b0;
        gpnae_wr_en[i]    <= 1'b0;
        gpnae_last[i]     <= 1'b0;
        gpnae_signal_i[i] <= '0;
        lane_collected[i] <= 1'b0;
      end
    end else begin
      filled_total <= filled_total_n;
      for (int i = 0; i < NUM_LANES; i++) begin
        fill_count[i]     <= fill_count_n[i];
        done_count[i]     <= done_count_n[i];
        load_finalized[i] <= load_finalized_n[i];
        gpnae_wr_en[i]    <= gpnae_wr_en_n[i];
        gpnae_last[i]     <= gpnae_last_n[i];
        gpnae_signal_i[i] <= gpnae_signal_n[i];
        lane_collected[i] <= lane_collected_n[i];
      end
    end
  end

  // =========================================================================
  // POOLING LANES: FIFO2 -> maxpool -> dropout on credit links (L8); each lane counts its set's windows out of maxpool and dropout
  // =========================================================================
  logic [TOT_W-1:0] lane_windows_total[NUM_LANES];
  logic [TOT_W-1:0] mp_out_count[NUM_LANES];  // this set's windows out of maxpool
  logic [TOT_W-1:0] dropout_out_count[NUM_LANES], dropout_out_count_n[NUM_LANES];

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      lane_windows_total[i] = (MAXPOOL_OUT_COUNT[TOT_W-1:0] / NUM_LANES) +
                              ((i < (MAXPOOL_OUT_COUNT % NUM_LANES)) ? 1'b1 : 1'b0);
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < NUM_LANES; i++) mp_out_count[i] <= '0;
    end else if (p_state == P_IDLE) begin
      for (int i = 0; i < NUM_LANES; i++) mp_out_count[i] <= '0;
    end else begin
      for (int i = 0; i < NUM_LANES; i++) if (maxpool_out_valid[i]) mp_out_count[i] <= mp_out_count[i] + 1'b1;
    end
  end

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      dropout_out_count_n[i] = dropout_valid_out[i] ? dropout_out_count[i] + 1'b1 : dropout_out_count[i];
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < NUM_LANES; i++) dropout_out_count[i] <= '0;
    end else if (p_state == P_IDLE) begin
      for (int i = 0; i < NUM_LANES; i++) dropout_out_count[i] <= '0;
    end else begin
      for (int i = 0; i < NUM_LANES; i++) dropout_out_count[i] <= dropout_out_count_n[i];
    end
  end

  always_comb begin
    streaming_complete = 1'b1;
    for (int i = 0; i < NUM_LANES; i++) begin
      if (dropout_out_count[i] != lane_windows_total[i]) streaming_complete = 1'b0;
    end
  end

  // One cycle per set, in issue order, with the put of its last word on L9; a partial set completes with no outputs as it passes pooling.
  assign set_complete = pool_done || p_null;
  assign set_done_id  = p_null ? p_next_id : p_set_id;
  if (LINK_STAGES == 0) begin : G_DONE_WIRE
    assign pipeline_complete_o = set_complete;
    assign done_set_id_o       = set_done_id;
  end else begin : G_DONE_REGS  // as many stages as L9, so the completion still arrives with the last word
    logic            done_q[LINK_STAGES];
    logic [ID_W-1:0] id_q  [LINK_STAGES];
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) for (int i = 0; i < LINK_STAGES; i++) begin done_q[i] <= 1'b0; id_q[i] <= '0; end
      else begin
        done_q[0] <= set_complete;
        id_q[0]   <= set_done_id;
        for (int i = 1; i < LINK_STAGES; i++) begin done_q[i] <= done_q[i-1]; id_q[i] <= id_q[i-1]; end
      end
    assign pipeline_complete_o = done_q[LINK_STAGES-1];
    assign done_set_id_o       = id_q[LINK_STAGES-1];
  end
  assign intermediate_buffer_full_o = 1'b0;  // no buffer between the mesh and the lanes since parallel fill
  assign intermediate_buffer_empty_o = 1'b1;

  assign systolic_busy_o = (mesh_sets != 0) || ((g_state != G_IDLE) && !g_fed);  // a set not yet pushed to the activation stage
  assign gpnae_busy_o = (g_state != G_IDLE) && g_fed;  // the stage takes its set before the result: busy once a beat is in

  logic any_mp_active;  // a lane has windows of the pooled set still to leave maxpool; never with a 1x1 pool, which skips it
  always_comb begin
    any_mp_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++) if (!POOL_BYPASS && mp_out_count[i] < lane_windows_total[i]) any_mp_active = 1'b1;
  end
  assign maxpool_busy_o = (p_state != P_IDLE) && any_mp_active;

  logic any_dropout_active;
  always_comb begin
    any_dropout_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++) if (dropout_valid_out[i]) any_dropout_active = 1'b1;
  end
  assign dropout_busy_o = any_dropout_active;

`ifndef SYNTHESIS
  // The accept's terms, registered: sampled assertion values miss a combinational host_accept when the host drives the start at the edge.
  logic acc_q, acc_accum_q, acc_prev_accum_q, last_accum;  // acc_prev_accum_q: the accept before this one was a partial sum
  logic acc_credit_q, acc_mready_q;  // a host credit was granted and a staging credit held, at the put
  logic [2:0] acc_shift_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {acc_q, acc_accum_q, acc_shift_q, acc_prev_accum_q, last_accum, acc_credit_q, acc_mready_q} <= '0;
    else begin
      {acc_q, acc_accum_q, acc_shift_q, acc_prev_accum_q} <= {host_accept, side.accumulate, side.pack_shift, last_accum};
      {acc_credit_q, acc_mready_q} <= {l0_out != 0, l1_cnt != 0};
      if (host_accept) last_accum <= side.accumulate;
    end
  // The mesh-side put's terms, registered: the set's rows are in its bank (B only when uncached).
  logic mput_q, mrows_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {mput_q, mrows_q} <= '0;
    else {mput_q, mrows_q} <= {l1m.put, !west_queue_empty && (l1m.data[5] || !north_queue_empty)};  // data[5]: weight_cached
  // A result beat's frame: beat index within the stage's set.
  logic [L3_CRW-1:0] l3_rx;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) l3_rx <= '0;
    else if (g_state == G_IDLE) l3_rx <= '0;
    else if (wide_rd_valid) l3_rx <= l3_rx + 1'b1;
  // Stage handshake invariants; live only with --assert.
  a_credit_range: assert property (@(posedge clk_i) disable iff (!rstn_i) int'(sets_out) + int'(l0_out) <= SETS_IN_FLIGHT)
    else $error("sienna_top: more sets in flight and host credits granted than SETS_IN_FLIGHT");
  a_credit_accept: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_q |-> acc_credit_q)
    else $error("sienna_top: a host put with no host credit granted");
  a_credit_return: assert property (@(posedge clk_i) disable iff (!rstn_i) pool_done |-> sets_out != 0)
    else $error("sienna_top: a set finished with no set in flight");
  a_mesh_takes_start: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_q |-> acc_mready_q)  // the staging put is the host's
    else $error("sienna_top: a staging put to the mesh with no staging credit");
  a_put_has_rows: assert property (@(posedge clk_i) disable iff (!rstn_i) mput_q |-> mrows_q)
    else $error("sienna_top: a set put to the mesh before its rows (A, and B unless cached) were written");
  a_l3_frame: assert property (@(posedge clk_i) disable iff (!rstn_i)
                               wide_rd_valid |-> (g_state != G_IDLE) && (l3_first == (l3_rx == '0)) &&
                                                 (l3_last == (int'(l3_rx) == PER_LANE - 1)) && (l3_pk == g_pack))
    else $error("sienna_top: result beat %0d out of frame (first %0b, last %0b, packed %0b against the stage's %0b)", l3_rx, l3_first,
                l3_last, l3_pk, g_pack);
  // Protocol checkers on the mesh side of L1 and L3, and on L6, checked whenever the pipeline has been empty a while.
  localparam int QUIET = GPNAE_FIFO_DEPTH + 8 + 4 * LINK_STAGES;  // past the lanes' 32-cycle advertisement after reset
  int quiet_n;
  // last_accum: a partial set completes as it passes the stage, before the mesh broadcasts its staging bank, so an open sum is never drained.
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) quiet_n <= 0;
    else if (host_accept || sets_out != 0 || mesh_sets != 0 || gp_sets != 0 || g_state != G_IDLE || p_state != P_IDLE || last_accum)
      quiet_n <= 0;
    else if (quiet_n < QUIET) quiet_n <= quiet_n + 1;
  assign drained = (quiet_n == QUIET);
  int drain_n;  // drain rises since reset; testbenches print it and require drained after their last set
  logic drained_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) begin
      drain_n   <= 0;
      drained_q <= 1'b0;
    end else begin
      drained_q <= drained;
      if (drained && !drained_q) drain_n <= drain_n + 1;
    end
  credit_link_checker #(.SLOTS(2)) l1_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(l1m));
  credit_link_checker #(.SLOTS(PER_LANE)) l3_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(l3m));
  credit_link_checker #(.SLOTS(2)) l6_chk (.clk_i(clk_i), .rstn_i(rstn_i), .drained_i(drained), .lnk(l6));
  // L4, L5 (Task 2 carries, Review Focus 5): a lane's words and results, checked on registered terms.
  logic l4_count_ok;  // every lane got PER_LANE words of a lane set, none of a ReLU or linear one
  logic [NUM_LANES-1:0] l5_put_v, l5_starved;
  always_comb begin
    l4_count_ok = 1'b1;
    for (int i = 0; i < NUM_LANES; i++) begin
      if (int'(l4_n[i]) != (act_bypass ? 0 : PER_LANE)) l4_count_ok = 1'b0;
      l5_put_v[i]   = gpnae_done[i];
      l5_starved[i] = (lane_outst[i] != '0) && (int'(l5_out[i]) == LANE_OUT_SLOTS) && (LANE_OUT_SLOTS < LANE_K);
    end
  end
  a_l4_count: assert property (@(posedge clk_i) disable iff (!rstn_i) g_done |-> l4_count_ok)
    else $error("sienna_top: a_l4_count: a lane's L4 puts for the set (lane 0: %0d) are not %0d, or not 0 for a ReLU or linear set (bypass %0b)",
                l4_n[0], PER_LANE, act_bypass);
  a_lane_hold: assert property (@(posedge clk_i) disable iff (!rstn_i) g_accept |-> lane_outst == '0)
    else $error("sienna_top: a_lane_hold: the next set's control word and parameters load while a lane still holds words (lane 0: %0d)",
                lane_outst[0]);
  a_l5_in_set: assert property (@(posedge clk_i) disable iff (!rstn_i) (l5_put_v & ~l5_want) == '0)
    else $error("sienna_top: a_l5_in_set: a lane result arrived with none due (results %b, due %b)", l5_put_v, l5_want);
  a_l5_starved: assert property (@(posedge clk_i) disable iff (!rstn_i) l5_starved == '0)
    else $error("sienna_top: a_l5_starved: a lane waits for %0d L5 credits and holds all %0d the collector advertised", LANE_K,
                LANE_OUT_SLOTS);
  a_g_from_mesh: assert property (@(posedge clk_i) disable iff (!rstn_i) g_accept |-> mesh_sets != 0)
    else $error("sienna_top: the activation stage took a result the host never started");
  a_act_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) g_done |-> !act_full[act_wr])
    else $error("sienna_top: activation finished into a full bank");
  a_act_bank_full: assert property (@(posedge clk_i) disable iff (!rstn_i) p_release |-> act_full[act_rd])
    else $error("sienna_top: pooling released an empty bank");
  a_real_has_bank: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                    ((p_state == P_IDLE) && (gp_sets != 0) && !set_accum[p_next_id]) |-> act_full[act_rd])
    else $error("sienna_top: pooling's next set is a real result with no activation bank holding it");
  a_null_no_result: assert property (@(posedge clk_i) disable iff (!rstn_i) g_null_done |-> !g_accept)
    else $error("sienna_top: a partial set and a mesh result were taken in the same cycle");
  a_complete_pulse: assert property (@(posedge clk_i) disable iff (!rstn_i) set_complete |=> !set_complete)
    else $error("sienna_top: pipeline_complete_o held for more than one cycle");
  a_complete_dispatched: assert property (@(posedge clk_i) disable iff (!rstn_i) pool_done |-> disp_done)
    else $error("sienna_top: pooling completed a set it never dispatched");
  localparam bit PACK_OK = (NUM_LANES % N == 0) && (COLLAPSE_K != 0) && POOL_BYPASS;
  a_pack_lanes: assert property (@(posedge clk_i) disable iff (!rstn_i) (acc_q && acc_shift_q != '0) |-> PACK_OK)
    else $error("sienna_top: a packed set needs N (%0d) to divide NUM_LANES (%0d), collapse-k 1 and a 1x1 pool", N, NUM_LANES);
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_q |-> int'(acc_shift_q) < LGN)
    else $error("sienna_top: pack shift %0d leaves blocks narrower than 2 of N=%0d", acc_shift_q, N);
  a_pack_one_pass: assert property (@(posedge clk_i) disable iff (!rstn_i) (acc_q && acc_shift_q != '0) |-> !acc_accum_q && !acc_prev_accum_q)
    else $error("sienna_top: a packed set cannot be a partial sum or continue one");
`ifdef ASSERT_SELFTEST
  a_selftest: assert property (@(posedge clk_i) disable iff (!rstn_i) 1'b0)
    else $error("sienna_top: assertion self-test fired, so assertions are live");
`endif
`endif

endmodule
