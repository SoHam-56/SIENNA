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
    parameter int    ACC_W             = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // sums, bias, mesh results: int32 in int8, DATA_WIDTH otherwise
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
    parameter int    DROPOUT_P_PERCENT = 50,
    parameter int    LFSR_WIDTH        = 32,
    parameter string INPUT_A_FILE      = "matrixA.mem",
    parameter string INPUT_B_FILE      = "matrixB.mem"
) (
    input logic clk_i,
    input logic rstn_i,

    input logic                     start_pipeline_i,
    input logic                     training_mode_i,  // dropout mode for the set being started
    input logic                     accumulate_i,     // 1: add this set's product to the running sum and output nothing
    input logic                     bias_valid_i,     // with the start: add bias_i[c] to column c of this set's product
    input logic [N-1:0][ACC_W-1:0]  bias_i,
    input logic [N-1:0][31:0]       req_mult_i,   // int8, with the start: requantize multiplier (Q0.31) of each output channel (column)
    input logic [N-1:0][7:0]        req_shift_i,  // int8: its shift, signed
    input logic [7:0]               req_zp_i,     // int8, layer-wide: output zero point; dropout drops to it after ReLU or linear (D-5)
    input logic [7:0]               req_min_i,    // int8: clamp, signed; the fused ReLU or ReLU6 lives here
    input logic [7:0]               req_max_i,
    input logic [15:0]              gp_mx_i,      // int8 GPNAE: rescale of the lane input to Q4.11
    input logic [4:0]               gp_shx_i,
    input logic [31:0]              gp_mout_i,    // int8 GPNAE: SELU's output requantize
    input logic [7:0]               gp_shout_i,
    input logic [7:0]               gp_zout_i,
    input logic                     weight_cached_i,  // with the start: B is cache tile weight_tile_i, only A is written
    input logic [WCTW-1:0]          weight_tile_i,
    input logic                     wc_write_enable_i,  // weight cache write of north_write_data_i at word wc_write_addr_i
    input logic [WCAW-1:0]          wc_write_addr_i,
    output logic [1:0]              wc_region_busy_o,   // a started set not yet broadcast reads this cache region
    input logic [   LFSR_WIDTH-1:0] dropout_seed_i,   // dropout seed for the set being started
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [     ADDR_LINES:0] num_terms_i,
    input logic                     north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0]       north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0]       west_write_data_i,
    input logic                     west_write_reset_i,

    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o,
    output logic [NUM_LANES-1:0] result_valid_o,  // lane's final_result_o is new this cycle

    output logic pipeline_complete_o,
    output logic pipeline_ready_o,  // a credit and a staging bank are free
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

  localparam int MAXPOOL_IN_COUNT = IN_ROWS * IN_COLS;
  localparam int POOL_OUT_ROWS = (IN_ROWS + 2 * PADDING - POOL_H) / STRIDE_ROWS + 1;
  localparam int POOL_OUT_COLS = (IN_COLS + 2 * PADDING - POOL_W) / STRIDE_COLS + 1;
  localparam int MAXPOOL_OUT_COUNT = POOL_OUT_ROWS * POOL_OUT_COLS;

  localparam int NUM_IDS = 1 << ID_W;  // more ids than sets in flight, so ids in flight never repeat
  localparam int CRW = $clog2(SETS_IN_FLIGHT + 1);
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

  logic mesh_input_ready, host_accept, g_accept, g_done, p_accept, p_release, pool_done;
  // g_done: the stage may take the next set; bank_done: bank bank_sel holds a whole set (later than g_done for an int8 ReLU or linear set).
  logic bank_done, bank_sel, byp_all_in, rq_drain;
  logic lane_v;  // a beat for the fill counters and the lanes
  logic byp_wr, byp_bank;  // int8: a ReLU or linear beat to write, and its bank
  logic [FCNT_W-1:0] byp_idx;  // its element within each lane's block
  logic [1:0] act_full;  // per activation bank: a finished activation not yet dispatched
  logic act_wr, act_rd;  // bank the lanes write, bank the dispatcher reads
  int act_wr_base, act_rd_base;
  logic [CRW-1:0] credits;  // sets the host may still start
  logic [CRW-1:0] mesh_sets;  // accepted sets the activation stage has not taken yet
  logic [ID_W-1:0] g_next_id, g_set_id, p_next_id, p_set_id, host_next_id;
  // Dropout mode and seed travel with each set, indexed by its id, so sets in flight keep their own.
  logic                  set_train[NUM_IDS];
  logic [LFSR_WIDTH-1:0] set_seed [NUM_IDS];
  logic                  set_accum[NUM_IDS];  // the set is a partial sum: accumulate it, output nothing
  logic [CONTROL_WIDTH-1:0] set_act[NUM_IDS];  // activation each set asked for, so layers in flight keep their own
  logic [   ADDR_LINES:0] set_terms[NUM_IDS];  // polynomial terms that go with set_act
  logic [CRW-1:0] gp_sets;  // sets past the activation stage and not yet complete; pooling takes them in id order
  // int8: the requantize and GPNAE parameters of the set the activation stage holds (g_*); p_zp: dropout's drop value for the pooled set.
  logic [N-1:0][31:0] g_mult;
  logic [N-1:0][7:0]  g_shift;
  logic [7:0]         g_zp, g_min, g_max, g_shout, g_zout, p_zp;
  logic [15:0]        g_mx;
  logic [4:0]         g_shx;
  logic [31:0]        g_mout;
  assign act_wr_base = act_wr ? SRAM_DEPTH : 0;
  assign act_rd_base = act_rd ? SRAM_DEPTH : 0;

  // Intermediate Memory Buffer
  logic [DATA_WIDTH-1:0] gpnae_out_mem  [0:2*SRAM_DEPTH-1];


  logic                  systolic_start;
  logic systolic_read_enable;
  logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr;  // wide read index, 0 .. PER_LANE-1
  logic [NUM_LANES-1:0][ACC_W-1:0] wide_rd_data;  // element k of lane k's block; int32 sums in int8
  logic                                 wide_rd_valid;
  logic                  systolic_mult_complete;
  logic                  systolic_collection_complete;
  logic systolic_reading;
  logic systolic_release;
  assign systolic_start = host_accept;  // the mesh queues it in a staging bank
  logic north_queue_empty, west_queue_empty;

  // Backend Arrays
  logic [      DATA_WIDTH-1:0] gpnae_signal_i     [NUM_LANES];
  logic                        gpnae_wr_en        [NUM_LANES];
  logic                        gpnae_start        [NUM_LANES];
  logic [GPNAE_ADDR_LINES-1:0] gpnae_terms        [NUM_LANES];
  logic [GPNAE_CTRL_WIDTH-1:0] gpnae_ctrl         [NUM_LANES];
  logic                        gpnae_full         [NUM_LANES];
  logic                        gpnae_empty_o      [NUM_LANES];
  logic                        gpnae_idle         [NUM_LANES];
  logic [      DATA_WIDTH-1:0] gpnae_result       [NUM_LANES];
  logic                        gpnae_done         [NUM_LANES];

  logic                        fifo2_wr_valid     [NUM_LANES];
  logic                        fifo2_rd_ready     [NUM_LANES];
  logic [      DATA_WIDTH-1:0] fifo2_wr_data      [NUM_LANES];
  logic [      DATA_WIDTH-1:0] fifo2_rd_data      [NUM_LANES];
  logic                        fifo2_rd_valid     [NUM_LANES];

  logic                        maxpool_start      [NUM_LANES];
  logic [      DATA_WIDTH-1:0] maxpool_data_in    [NUM_LANES];
  logic                        maxpool_valid_in   [NUM_LANES];
  logic [      DATA_WIDTH-1:0] maxpool_out_data   [NUM_LANES];
  logic                        maxpool_out_valid  [NUM_LANES];
  logic                        maxpool_done_signal[NUM_LANES];

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
  logic                        gpnae_start_n      [NUM_LANES];
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
      .ACC_BANKS  (ACC_BANKS),
      .RESULT_BANKS(RESULT_BANKS)
  ) systolic_array_inst (
      .clk_i                 (clk_i),
      .rstn_i                (rstn_i),
      .start_matrix_mult_i   (systolic_start),
      .partial_i             (accumulate_i),
      .bias_valid_i          (bias_valid_i),
      .bias_i                (bias_i),
      .weight_cached_i       (weight_cached_i),
      .weight_tile_i         (weight_tile_i),
      .wc_write_enable_i     (wc_write_enable_i),
      .wc_write_addr_i       (wc_write_addr_i),
      .wc_region_busy_o      (wc_region_busy_o),
      .north_write_enable_i  (north_write_enable_i),
      .north_write_data_i    (north_write_data_i),
      .north_write_reset_i   (north_write_reset_i),
      .west_write_enable_i   (west_write_enable_i),
      .west_write_data_i     (west_write_data_i),
      .west_write_reset_i    (west_write_reset_i),
      .north_queue_empty_o   (north_queue_empty),
      .west_queue_empty_o    (west_queue_empty),
      .matrix_mult_complete_o(systolic_mult_complete),
      .read_enable_i         (1'b0),
      .read_addr_i           ('0),
      .read_data_o           (),
      .read_valid_o          (),
      .wide_read_enable_i    (systolic_read_enable),
      .wide_read_index_i     (32'(systolic_read_addr)),
      .wide_read_data_o      (wide_rd_data),
      .wide_read_valid_o     (wide_rd_valid),
      .collection_complete_o (systolic_collection_complete),
      .collection_active_o   (),
      .result_release_i      (systolic_release),
      .input_ready_o         (mesh_input_ready)
  );

  generate
    genvar g;
    for (g = 0; g < NUM_LANES; g++) begin : backend_lanes

      gpnae_poly #(
          .EXP_W        (EXP_W),
          .MAN_W        (MAN_W),
          .DATA_WIDTH   (GPNAE_DATA_WIDTH),
          .ADDR_LINES   (GPNAE_ADDR_LINES),
          .CONTROL_WIDTH(GPNAE_CTRL_WIDTH)
      ) gpnae_inst (
          .clk_i         (clk_i),
          .rstn_i        (rstn_i),
          .signal_i      (gpnae_signal_i[g]),
          .wr_en_i       (gpnae_wr_en[g]),
          .last_i        (gpnae_start[g]),
          .terms_i       (gpnae_terms[g]),
          .control_word_i(gpnae_ctrl[g]),
          .gp_mx_i       (g_mx),
          .gp_shx_i      (g_shx),
          .gp_zin_i      (g_zp),     // int8: the lane's input is the requantize output, whose zero point is req_zp_i
          .gp_mout_i     (g_mout),
          .gp_shout_i    (g_shout),
          .gp_zout_i     (g_zout),
          .full_o        (gpnae_full[g]),
          .empty_o       (gpnae_empty_o[g]),
          .idle_o        (gpnae_idle[g]),
          .final_result_o(gpnae_result[g]),
          .done_o        (gpnae_done[g])
      );

      fwft #(
          .DATA_WIDTH(DATA_WIDTH),
          .FIFO_DEPTH(FIFO2_DEPTH)
      ) fifo2_inst (
          .clk_i     (clk_i),
          .rstn_i    (rstn_i),
          .wr_valid_i(fifo2_wr_valid[g]),
          .wr_data_i (fifo2_wr_data[g]),
          .wr_ready_o(),
          .rd_ready_i(fifo2_rd_ready[g]),
          .rd_data_o (fifo2_rd_data[g]),
          .rd_valid_o(fifo2_rd_valid[g]),
          .count_o   (fifo2_count[g])
      );

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
          .MAN_W      (MAN_W)
      ) maxpool_inst (
          .clk      (clk_i),
          .rst_n    (rstn_i),
          .start    (maxpool_start[g]),
          .done     (maxpool_done_signal[g]),
          .data_in  (maxpool_data_in[g]),
          .valid_in (maxpool_valid_in[g]),
          .out_data (maxpool_out_data[g]),
          .out_valid(maxpool_out_valid[g])
      );

      // A nonlinear mix, since an XOR-only one makes the 16 lanes' masks linearly tied.
      logic [31:0] lane_mul;
      logic [LFSR_WIDTH-1:0] lane_mix, lane_seed;
      assign lane_mul  = (32'(set_seed[p_next_id]) ^ (32'h9E3779B9 * (g + 1))) * 32'h85EBCA6B;
      assign lane_mix  = LFSR_WIDTH'(lane_mul ^ (lane_mul >> 16));
      assign lane_seed = (lane_mix == '0) ? '1 : lane_mix;  // an all-zero LFSR state would lock up

      dropout #(
          .EXP_W            (EXP_W),
          .MAN_W            (MAN_W),
          .DATA_WIDTH       (DATA_WIDTH),
          .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
          .LFSR_WIDTH       (LFSR_WIDTH)
      ) dropout_inst (
          .clk          (clk_i),
          .rst_n        (rstn_i),
          .in_valid     (dropout_in_valid[g]),
          .training_mode(set_train[p_set_id]),
          .data_in      (dropout_data_in[g]),
          .reseed_i     (p_accept),
          .seed_i       (lane_seed),
          .zero_point_i (p_zp),
          .data_out     (dropout_data_out[g]),
          .valid_out    (dropout_valid_out[g])
      );
    end
  endgenerate

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      dropout_in_valid[i] = POOL_BYPASS ? byp_valid[i] : maxpool_out_valid[i];
      dropout_data_in[i] = POOL_BYPASS ? byp_data[i] : maxpool_out_data[i];
      gpnae_terms[i] = set_terms[g_set_id][GPNAE_ADDR_LINES-1:0];
      gpnae_ctrl[i] = set_act[g_set_id];  // the set the activation stage holds
    end
  end

  // ReLU and linear need no polynomial: the activation stage writes each beat straight into its bank and the lanes stay idle.
  logic act_bypass;
  assign act_bypass = (set_act[g_set_id] == CONTROL_WIDTH'(3'b100)) || (set_act[g_set_id] == CONTROL_WIDTH'(3'b101));
  logic act_is_relu;
  assign act_is_relu = (set_act[g_set_id] == CONTROL_WIDTH'(3'b100));

  // =========================================================================
  // GPNAE TO CENTRAL BUFFER WRITE LOGIC
  // =========================================================================
  always_ff @(posedge clk_i) begin
    // Beat b of the wide read holds element k*PER_LANE + b in word k, the element lane k would have taken.
    if (!IS_INT && g_state != G_IDLE && act_bypass && fill_v)
      for (int k = 0; k < NUM_LANES; k++)
        gpnae_out_mem[act_wr_base + k * PER_LANE + fill_count[k]] <= (act_is_relu && fill_d[k][DATA_WIDTH-1]) ? '0 : fill_d[k];
    if (IS_INT && byp_wr)  // int8: a beat lands where its own tag says, so it may leave the requantize pipeline after its set left the stage
      for (int k = 0; k < NUM_LANES; k++) gpnae_out_mem[(byp_bank ? SRAM_DEPTH : 0) + k * PER_LANE + int'(byp_idx)] <= fill_d[k];
    if (g_state == G_ROUND && !act_bypass) begin
      for (int i = 0; i < NUM_LANES; i++) begin
        if (load_finalized[i] && gpnae_done[i] && (done_count[i] < fill_count[i])) begin
          // RESTORED: This is the mathematically perfect chunked indexing!
          gpnae_out_mem[act_wr_base + i * PER_LANE + done_count[i]] <= gpnae_result[i];
        end
      end
    end
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
  // A 1x1 window with stride 1 and no padding is the identity: windows skip FIFO2 and maxpool and go straight to dropout.
  localparam bit POOL_BYPASS = (POOL_H == 1) && (POOL_W == 1) && (STRIDE_ROWS == 1) && (STRIDE_COLS == 1) && (PADDING == 0);
  logic [DATA_WIDTH-1:0] byp_data [NUM_LANES];
  logic [NUM_LANES-1:0]  byp_valid;

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

  // One lane short of room stalls the whole group, which keeps every lane on the same element.
  always_comb begin
    disp_can_write = 1'b1;
    for (int L = 0; L < NUM_LANES; L++)
      if (!POOL_BYPASS && lane_active[L] && (fifo2_count[L] > (FIFO2_DEPTH - 4))) disp_can_write = 1'b0;
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

  assign pipeline_ready_o = (credits != 0) && mesh_input_ready;
  assign host_accept = start_pipeline_i && pipeline_ready_o && (weight_cached_i || !north_queue_empty) && !west_queue_empty;
  // Not while the previous result's read or release is in flight: its bank flag may still read full.
  assign g_accept = (g_state == G_IDLE) && !set_accum[g_next_id] && systolic_collection_complete && !act_full[act_wr] &&
                    !systolic_read_enable && !systolic_release;
  assign g_done = (g_state == G_ROUND) && (act_bypass ? byp_all_in : all_collected);
  logic g_null_done;  // a partial set, summed in the PEs, passes with no result and no activation bank
  assign g_null_done = (g_state == G_IDLE) && (mesh_sets != 0) && set_accum[g_next_id] && !rq_drain;
  // Pooling completes sets in id order: a partial in one cycle, never right after another completion, so pulses stay one cycle.
  logic complete_q, p_null;
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
      credits   <= CRW'(SETS_IN_FLIGHT);
      mesh_sets <= '0;
      g_next_id <= '0;
      g_set_id  <= '0;
      p_next_id <= '0;
      p_set_id  <= '0;
      host_next_id <= '0;
      for (int k = 0; k < NUM_IDS; k++) begin
        set_train[k] <= 1'b0;
        set_seed[k]  <= '1;
        set_accum[k] <= 1'b0;
        set_act[k]   <= '0;
        set_terms[k] <= '0;
      end
    end else begin
      complete_q <= pipeline_complete_o;
      if (host_accept) begin
        set_accum[host_next_id] <= accumulate_i;
        set_act[host_next_id]   <= activation_function_i;
        set_terms[host_next_id] <= num_terms_i;
        set_train[host_next_id] <= training_mode_i;
        set_seed[host_next_id]  <= dropout_seed_i;
        host_next_id <= host_next_id + 1'b1;
      end
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
      if (bank_done) act_full[bank_sel] <= 1'b1;
      if (g_done) act_wr <= ~act_wr;
      if (p_release) begin
        act_full[act_rd] <= 1'b0;
        act_rd <= ~act_rd;
      end
      gp_sets <= gp_sets + CRW'(bank_done || g_null_done) - CRW'(pipeline_complete_o);
      credits   <= credits - CRW'(host_accept) + CRW'(pipeline_complete_o);
      mesh_sets <= mesh_sets + CRW'(host_accept) - CRW'(g_accept || g_null_done);
      if (g_accept) g_set_id <= g_next_id;
      if (g_accept || g_null_done) g_next_id <= g_next_id + 1'b1;
      if (p_accept || p_null) begin
        p_set_id  <= p_next_id;
        p_next_id <= p_next_id + 1'b1;
      end
    end
  end

  // int8 (D-2): parameters per set id; at g_accept the stage copies its set's per-channel words, so a lane picks among N, not NUM_IDS * N.
  if (IS_INT) begin : G_REQ_SETS
    logic [N-1:0][31:0] s_mult [NUM_IDS];
    logic [N-1:0][7:0]  s_shift[NUM_IDS];
    logic [7:0]  s_zp[NUM_IDS], s_min[NUM_IDS], s_max[NUM_IDS], s_shout[NUM_IDS], s_zout[NUM_IDS];
    logic [15:0] s_mx[NUM_IDS];
    logic [4:0]  s_shx[NUM_IDS];
    logic [31:0] s_mout[NUM_IDS];
    always_ff @(posedge clk_i) begin  // no reset: an id's entry is written by its own accept before any stage reads it
      if (host_accept) begin
        s_mult[host_next_id]  <= req_mult_i;
        s_shift[host_next_id] <= req_shift_i;
        s_zp[host_next_id]    <= req_zp_i;
        s_min[host_next_id]   <= req_min_i;
        s_max[host_next_id]   <= req_max_i;
        s_mx[host_next_id]    <= gp_mx_i;
        s_shx[host_next_id]   <= gp_shx_i;
        s_mout[host_next_id]  <= gp_mout_i;
        s_shout[host_next_id] <= gp_shout_i;
        s_zout[host_next_id]  <= gp_zout_i;
      end
      if (g_accept) begin
        g_mult  <= s_mult[g_next_id];
        g_shift <= s_shift[g_next_id];
      end
    end
    assign g_zp    = s_zp[g_set_id];
    assign g_min   = s_min[g_set_id];
    assign g_max   = s_max[g_set_id];
    assign g_mx    = s_mx[g_set_id];
    assign g_shx   = s_shx[g_set_id];
    assign g_mout  = s_mout[g_set_id];
    assign g_shout = s_shout[g_set_id];
    assign g_zout  = s_zout[g_set_id];
    // D-5: a dropped value is the output zero point of the pooled set's activation (tanh, and every code the lane runs as tanh: 0).
    always_comb
      case (set_act[p_set_id])
        CONTROL_WIDTH'(3'b001): p_zp = s_zout[p_set_id];  // SELU: its requantized output's zero point
        CONTROL_WIDTH'(3'b010): p_zp = 8'h80;  // sigmoid: TFLite's fixed output zero point -128
        CONTROL_WIDTH'(3'b100), CONTROL_WIDTH'(3'b101): p_zp = s_zp[p_set_id];  // ReLU, linear: the requantize output's
        default: p_zp = 8'h00;  // tanh: zero point 0
      endcase
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
  // PARALLEL LANE FILL: each wide read gives every lane its next element at once
  // =========================================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      systolic_read_enable <= 1'b0;
      systolic_read_addr   <= '0;
      systolic_reading     <= 1'b0;
      systolic_release     <= 1'b0;
    end else begin
      systolic_release <= 1'b0;
      if (g_state == G_FEED) begin
        systolic_reading     <= 1'b1;
        systolic_read_enable <= 1'b1;
        systolic_read_addr   <= '0;
      end else if (systolic_read_enable) begin
        if (systolic_read_addr == PER_LANE[$clog2(SRAM_DEPTH)-1:0] - 1'b1) begin
          systolic_read_enable <= 1'b0;
          systolic_reading     <= 1'b0;
          systolic_release     <= 1'b1;  // after the last read
        end else systolic_read_addr <= systolic_read_addr + 1'b1;
      end
    end
  end

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
      tg_idx[0] <= rq_in;
      for (int i = 1; i < RQL; i++) tg_idx[i] <= tg_idx[i-1];
    end
    assign byp_all_in = (rq_in == PER_LANE[FCNT_W-1:0]);  // the last read is in: the stage may leave while it drains
    assign rq_drain   = |tg_v;
    assign lane_v     = fill_v && !tg_byp[RQL-1];
    assign byp_wr     = fill_v && tg_byp[RQL-1];
    assign byp_bank   = tg_bank[RQL-1];
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
    assign fill_v     = wide_rd_valid;
    assign fill_d     = wide_rd_data;
    assign lane_v     = fill_v;
    assign byp_wr     = 1'b0;
    assign byp_bank   = 1'b0;
    assign byp_idx    = '0;
    assign byp_all_in = (fill_count[0] == PER_LANE[FCNT_W-1:0]);
    assign rq_drain   = 1'b0;
    assign bank_done  = g_done;
    assign bank_sel   = act_wr;
  end

  always_comb begin
    filled_total_n = filled_total;
    for (int i = 0; i < NUM_LANES; i++) begin
      fill_count_n[i]     = fill_count[i];
      done_count_n[i]     = done_count[i];
      load_finalized_n[i] = load_finalized[i];
      gpnae_start_n[i]    = 1'b0;
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
          fill_count_n[i]   = fill_count[i] + 1'b1;
        end
      end
      // Start every lane together, the cycle after its last element is written.
      for (int i = 0; i < NUM_LANES; i++) begin
        if ((fill_count[i] == PER_LANE[FCNT_W-1:0]) && !load_finalized[i] && !act_bypass) begin
          gpnae_start_n[i]    = 1'b1;
          load_finalized_n[i] = 1'b1;
        end
      end
      if (g_state == G_ROUND) begin
        for (int i = 0; i < NUM_LANES; i++) begin
          if (load_finalized[i] && gpnae_done[i] && (done_count[i] < fill_count[i])) begin
            done_count_n[i] = done_count[i] + 1'b1;
            if ((done_count[i] + 1'b1) == fill_count[i]) lane_collected_n[i] = 1'b1;
          end
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
        gpnae_start[i]    <= 1'b0;
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
        gpnae_start[i]    <= gpnae_start_n[i];
        gpnae_signal_i[i] <= gpnae_signal_n[i];
        lane_collected[i] <= lane_collected_n[i];
      end
    end
  end

  // =========================================================================
  // STREAMING MAXPOOL FEEDER (Pre-Packaged Window Receiver)
  // =========================================================================
  localparam int MPW_W = $clog2(POOL_H * POOL_W + 1);

  typedef enum logic [1:0] {
    MP_IDLE,
    MP_FEED,
    MP_WAIT_DONE,
    MP_DONE
  } mp_state_t;

  mp_state_t mp_state[NUM_LANES], mp_state_n[NUM_LANES];

  logic [MPW_W-1:0] mp_window_fed[NUM_LANES], mp_window_fed_n[NUM_LANES];
  logic [TOT_W-1:0] mp_windows_done[NUM_LANES], mp_windows_done_n[NUM_LANES];
  logic [TOT_W-1:0] lane_windows_total[NUM_LANES];
  logic [TOT_W-1:0] dropout_out_count[NUM_LANES], dropout_out_count_n[NUM_LANES];

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      lane_windows_total[i] = (MAXPOOL_OUT_COUNT[TOT_W-1:0] / NUM_LANES) + 
                              ((i < (MAXPOOL_OUT_COUNT % NUM_LANES)) ? 1'b1 : 1'b0);
    end
  end

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      mp_state_n[i]        = mp_state[i];
      mp_window_fed_n[i]   = mp_window_fed[i];
      mp_windows_done_n[i] = mp_windows_done[i];

      maxpool_start[i]     = 1'b0;
      maxpool_valid_in[i]  = 1'b0;
      maxpool_data_in[i]   = '0;
      fifo2_rd_ready[i]    = 1'b0;

      case (mp_state[i])
        MP_IDLE: begin
          if (!POOL_BYPASS && mp_windows_done[i] < lane_windows_total[i]) begin
            mp_state_n[i] = MP_FEED;
            maxpool_start[i] = 1'b1;
          end else if (p_state == P_WAIT) begin
            mp_state_n[i] = MP_DONE;
          end
        end

        MP_FEED: begin
          maxpool_start[i] = 1'b1;
          if (mp_window_fed[i] < (POOL_H * POOL_W)) begin
            if (fifo2_rd_valid[i]) begin
              maxpool_valid_in[i] = 1'b1;
              maxpool_data_in[i]  = fifo2_rd_data[i];
              fifo2_rd_ready[i]   = 1'b1;
              mp_window_fed_n[i]  = mp_window_fed[i] + 1'b1;
            end
          end
          if (mp_window_fed_n[i] == (POOL_H * POOL_W)) begin
            mp_state_n[i] = MP_WAIT_DONE;
          end
        end

        MP_WAIT_DONE: begin
          maxpool_start[i] = 1'b1;
          if (maxpool_done_signal[i]) begin
            maxpool_start[i] = 1'b0;
            mp_window_fed_n[i] = '0;
            mp_windows_done_n[i] = mp_windows_done[i] + 1'b1;
            mp_state_n[i] = MP_IDLE;
          end
        end

        MP_DONE: maxpool_start[i] = 1'b0;
        default: mp_state_n[i] = MP_IDLE;
      endcase
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < NUM_LANES; i++) begin
        mp_state[i]        <= MP_IDLE;
        mp_window_fed[i]   <= '0;
        mp_windows_done[i] <= '0;
      end
    end else if (p_state == P_IDLE) begin
      for (int i = 0; i < NUM_LANES; i++) begin
        mp_state[i]        <= MP_IDLE;
        mp_window_fed[i]   <= '0;
        mp_windows_done[i] <= '0;
      end
    end else begin
      for (int i = 0; i < NUM_LANES; i++) begin
        mp_state[i]        <= mp_state_n[i];
        mp_window_fed[i]   <= mp_window_fed_n[i];
        mp_windows_done[i] <= mp_windows_done_n[i];
      end
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

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < NUM_LANES; i++) final_result_o[i] <= '0;
      result_valid_o <= '0;
    end else begin
      for (int i = 0; i < NUM_LANES; i++) begin
        if (dropout_valid_out[i]) final_result_o[i] <= dropout_data_out[i];
        result_valid_o[i] <= dropout_valid_out[i];
      end
    end
  end

  // One cycle per set, in issue order; a partial set completes with no outputs as it passes pooling.
  assign pipeline_complete_o = pool_done || p_null;
  assign done_set_id_o = p_null ? p_next_id : p_set_id;
  assign intermediate_buffer_full_o = 1'b0;  // no buffer between the mesh and the lanes since parallel fill
  assign intermediate_buffer_empty_o = 1'b1;

  assign systolic_busy_o = (mesh_sets != 0);
  assign gpnae_busy_o = (g_state != G_IDLE);

  logic any_mp_active;
  always_comb begin
    any_mp_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++)
    if (mp_state[i] != MP_DONE && mp_state[i] != MP_IDLE) any_mp_active = 1'b1;
  end
  assign maxpool_busy_o = (p_state != P_IDLE) && any_mp_active;

  logic any_dropout_active;
  always_comb begin
    any_dropout_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++) if (dropout_valid_out[i]) any_dropout_active = 1'b1;
  end
  assign dropout_busy_o = any_dropout_active;

`ifndef SYNTHESIS
  // Stage handshake invariants; live only with --assert.
  a_credit_range: assert property (@(posedge clk_i) disable iff (!rstn_i) credits <= SETS_IN_FLIGHT)
    else $error("sienna_top: more credits than SETS_IN_FLIGHT");
  a_credit_accept: assert property (@(posedge clk_i) disable iff (!rstn_i) host_accept |-> credits != 0)
    else $error("sienna_top: a start was accepted without a credit");
  a_credit_return: assert property (@(posedge clk_i) disable iff (!rstn_i) pool_done |-> credits < SETS_IN_FLIGHT)
    else $error("sienna_top: a set finished with every credit already free");
  a_mesh_takes_start: assert property (@(posedge clk_i) disable iff (!rstn_i) systolic_start |-> mesh_input_ready)
    else $error("sienna_top: a start was forwarded to a mesh with no free staging bank");
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
  a_complete_pulse: assert property (@(posedge clk_i) disable iff (!rstn_i) pipeline_complete_o |=> !pipeline_complete_o)
    else $error("sienna_top: pipeline_complete_o held for more than one cycle");
  a_complete_dispatched: assert property (@(posedge clk_i) disable iff (!rstn_i) pool_done |-> disp_done)
    else $error("sienna_top: pooling completed a set it never dispatched");
`ifdef ASSERT_SELFTEST
  a_selftest: assert property (@(posedge clk_i) disable iff (!rstn_i) 1'b0)
    else $error("sienna_top: assertion self-test fired, so assertions are live");
`endif
`endif

endmodule
