`timescale 1ns / 100ps

module sienna_top #(
    parameter int    NUM_LANES         = 32,
    parameter int    N                 = 16,
    parameter int    TILE_SIZE         = 4,
    parameter int    HOST_WORDS        = 1,  // words per host write; must divide N*N
    parameter int    COLLAPSE_K        = 0,  // 1: one full-depth mesh tile per output tile, N^2 PEs and no reduce
    parameter int    DATA_WIDTH        = 32,
    parameter int    SRAM_DEPTH        = N * N,
    parameter int    FIFO_DEPTH        = N * N,
    parameter int    ADDR_LINES        = $clog2(FIFO_DEPTH),
    parameter int    CONTROL_WIDTH     = 2,
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
    input logic [   LFSR_WIDTH-1:0] dropout_seed_i,   // dropout seed for the set being started
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [     ADDR_LINES:0] num_terms_i,
    input logic                     north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                     west_write_reset_i,

    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o,
    output logic [NUM_LANES-1:0] result_valid_o,  // lane's final_result_o is new this cycle

    output logic pipeline_complete_o,
    output logic pipeline_ready_o,  // a credit and a staging bank are free
    output logic [1:0] done_set_id_o,  // id of the set pipeline_complete_o reports
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic maxpool_busy_o,
    output logic dropout_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o
);

  localparam int GPNAE_DATA_WIDTH = 32;
  localparam int GPNAE_ADDR_LINES = 5;
  localparam int GPNAE_CTRL_WIDTH = 2;
  localparam int GPNAE_FIFO_DEPTH = 2 ** GPNAE_ADDR_LINES;
  // Work is split evenly across the lanes rather than filling each to its FIFO depth. With
  // 8 lanes those coincide (256/8 = 32 = GPNAE_FIFO_DEPTH), which is why the old code could use
  // the depth directly; at any other lane count it would leave the later lanes empty and index
  // gpnae_out_mem past its end.
  localparam int PER_LANE = SRAM_DEPTH / NUM_LANES;

  // Lanes take contiguous blocks of PER_LANE elements, so the division has to be exact: a
  // remainder would be dropped silently, and PER_LANE must fit a lane's input FIFO.
  initial begin
    if ((SRAM_DEPTH % NUM_LANES) != 0)
      $error("sienna_top: NUM_LANES (%0d) must divide SRAM_DEPTH (%0d)", NUM_LANES, SRAM_DEPTH);
    if (PER_LANE > GPNAE_FIFO_DEPTH)
      $error("sienna_top: PER_LANE (%0d) exceeds GPNAE_FIFO_DEPTH (%0d)", PER_LANE, GPNAE_FIFO_DEPTH);
  end

  localparam int FCNT_W = $clog2(PER_LANE + 1);  // what fill_count/done_count actually range over
  localparam int TOT_W = $clog2(SRAM_DEPTH + 1);

  localparam int FIFO2_DEPTH = 16;

  localparam int MAXPOOL_IN_COUNT = IN_ROWS * IN_COLS;
  localparam int POOL_OUT_ROWS = (IN_ROWS + 2 * PADDING - POOL_H) / STRIDE_ROWS + 1;
  localparam int POOL_OUT_COLS = (IN_COLS + 2 * PADDING - POOL_W) / STRIDE_COLS + 1;
  localparam int MAXPOOL_OUT_COUNT = POOL_OUT_ROWS * POOL_OUT_COLS;

  localparam int MAX_SETS_IN_FLIGHT = 3;

  // Activation stage: read a mesh result into FIFO1, fill the lanes, write gpnae_out_mem.
  // G_ACC_RD/G_ACC_WAIT add a mesh result into acc_mem; G_AFEED fills the lanes from acc_mem instead of the mesh.
  typedef enum logic [2:0] {G_IDLE, G_FEED, G_LATCH, G_ROUND, G_ACC_RD, G_ACC_WAIT, G_AFEED} g_state_t;
  // Pooling stage: dispatch windows into FIFO2, then wait for maxpool and dropout to drain.
  typedef enum logic [1:0] {P_IDLE, P_DISPATCH, P_WAIT} p_state_t;
  g_state_t g_state;
  p_state_t p_state;

  logic mesh_input_ready, host_accept, g_accept, g_done, p_accept, p_release, pool_done;
  logic [1:0] act_full;  // per activation bank: a finished activation not yet dispatched
  logic act_wr, act_rd;  // bank the lanes write, bank the dispatcher reads
  int act_wr_base, act_rd_base;
  logic [1:0] credits;  // sets the host may still start
  logic [2:0] mesh_sets;  // accepted sets the activation stage has not taken yet
  logic [1:0] g_next_id, g_set_id, p_next_id, p_set_id, host_next_id;
  // Dropout mode and seed travel with each set, indexed by its id, so sets in flight keep their own.
  logic                  set_train[4];
  logic [LFSR_WIDTH-1:0] set_seed [4];
  logic                  set_accum[4];  // the set is a partial sum: accumulate it, output nothing
  logic [1:0] act_null;  // per activation bank: holds a partial set, which pooling passes through without output
  assign act_wr_base = act_wr ? SRAM_DEPTH : 0;
  assign act_rd_base = act_rd ? SRAM_DEPTH : 0;

  // Intermediate Memory Buffer
  logic [DATA_WIDTH-1:0] gpnae_out_mem  [0:2*SRAM_DEPTH-1];


  logic                  systolic_start;
  logic systolic_read_enable;
  logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr;  // wide read index, 0 .. PER_LANE-1
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] wide_rd_data;  // element k of lane k's block
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
      .DATA_WIDTH (DATA_WIDTH),
      .WIDE_READ  (NUM_LANES),
      .HOST_WORDS (HOST_WORDS),
      .COLLAPSE_K (COLLAPSE_K)
  ) systolic_array_inst (
      .clk_i                 (clk_i),
      .rstn_i                (rstn_i),
      .start_matrix_mult_i   (systolic_start),
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
          .IS_FP32    (1)
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
          .data_out     (dropout_data_out[g]),
          .valid_out    (dropout_valid_out[g])
      );
    end
  endgenerate

  always_comb begin
    for (int i = 0; i < NUM_LANES; i++) begin
      dropout_in_valid[i] = maxpool_out_valid[i];
      dropout_data_in[i] = maxpool_out_data[i];
      gpnae_terms[i] = num_terms_i[GPNAE_ADDR_LINES-1:0];
      gpnae_ctrl[i] = activation_function_i;
    end
  end

  // =========================================================================
  // GPNAE TO CENTRAL BUFFER WRITE LOGIC
  // =========================================================================
  always_ff @(posedge clk_i) begin
    if (g_state == G_ROUND) begin
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
                    ? gpnae_out_mem[act_rd_base+ir*IN_COLS+ic] : 32'hFF800000;
    end
  end

  // One lane short of room stalls the whole group, which keeps every lane on the same element.
  always_comb begin
    disp_can_write = 1'b1;
    for (int L = 0; L < NUM_LANES; L++)
      if (lane_active[L] && (fifo2_count[L] > (FIFO2_DEPTH - 4))) disp_can_write = 1'b0;
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
    end else if (p_state == P_DISPATCH) begin
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;

      if (!disp_done && disp_can_write) begin
        for (int L = 0; L < NUM_LANES; L++) begin
          if (lane_active[L]) begin
            fifo2_wr_data[L]  <= lane_val[L];
            fifo2_wr_valid[L] <= 1'b1;
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
              // Advance every lane by NUM_LANES windows, carrying into the row.
              for (int L = 0; L < NUM_LANES; L++) begin
                automatic int c_tmp = lane_c[L] + NUM_LANES;
                automatic int r_tmp = lane_r[L];
                for (int k = 0; k < NUM_LANES; k++) begin
                  if (c_tmp >= POOL_OUT_COLS) begin
                    c_tmp = c_tmp - POOL_OUT_COLS;
                    r_tmp = r_tmp + 1;
                  end
                end
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
    end
  end

  // =========================================================================
  // ACCUMULATION: partial sets are summed in acc_mem, laid out like the wide read (beat i, word k)
  // =========================================================================
  localparam int ACC_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int AW = $clog2(PER_LANE + 1);
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] acc_mem[PER_LANE];
  logic acc_valid;  // acc_mem holds a sum; clear means the next partial is copied, not added
  logic [AW-1:0] acc_beat;  // wide-read beats taken this accumulation
  logic [3:0] acc_pend;  // adds issued and not yet written back
  logic acc_phase, acc_issue;
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] acc_sum;
  logic [NUM_LANES-1:0] acc_sum_v;
  logic [AW-1:0] acc_tag[ACC_LAT];
  assign acc_phase = (g_state == G_ACC_RD) || (g_state == G_ACC_WAIT);
  assign acc_issue = acc_phase && wide_rd_valid && acc_valid;
  // Every beat taken and every add written back: the sum for this set is final.
  logic acc_done_w;
  assign acc_done_w = (acc_beat == AW'(PER_LANE)) && (acc_pend == '0) && !acc_issue && !systolic_read_enable;

  for (genvar k = 0; k < NUM_LANES; k++) begin : ACC_ADD
    fp32Adder add (
        .clk_i      (clk_i),
        .rstn_i     (rstn_i),
        .valid_i    (acc_issue),
        .A          (acc_mem[acc_beat][k]),
        .B          (wide_rd_data[k]),
        .result_o   (acc_sum[k]),
        .done_o     (acc_sum_v[k]),
        .overflow_o (),
        .underflow_o(),
        .invalid_o  ()
    );
  end

  // Reading acc_mem back into the lanes: one beat per cycle, one cycle of latency like the mesh read.
  logic acc_rd_en, acc_rd_valid;
  logic [AW-1:0] acc_rd_addr;
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] acc_rd_data;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      acc_valid    <= 1'b0;
      acc_beat     <= '0;
      acc_pend     <= '0;
      acc_rd_en    <= 1'b0;
      acc_rd_addr  <= '0;
      acc_rd_valid <= 1'b0;
      acc_rd_data  <= '0;
      for (int i = 0; i < ACC_LAT; i++) acc_tag[i] <= '0;
    end else begin
      acc_tag[0] <= acc_beat;
      for (int i = 1; i < ACC_LAT; i++) acc_tag[i] <= acc_tag[i-1];
      acc_pend <= acc_pend + {3'b0, acc_issue} - {3'b0, acc_sum_v[0]};
      if (acc_sum_v[0]) acc_mem[acc_tag[ACC_LAT-1]] <= acc_sum;
      if (g_state == G_ACC_RD) acc_beat <= '0;
      else if (acc_phase && wide_rd_valid) begin
        if (!acc_valid) acc_mem[acc_beat] <= wide_rd_data;  // the first partial is copied exactly
        acc_beat <= acc_beat + 1'b1;
      end
      // A partial leaves a sum behind; the final set's read of it empties it.
      if (g_state == G_ACC_WAIT && acc_done_w && set_accum[g_set_id]) acc_valid <= 1'b1;
      if (g_state == G_AFEED) acc_valid <= 1'b0;
      acc_rd_valid <= acc_rd_en;
      if (acc_rd_en) acc_rd_data <= acc_mem[acc_rd_addr];
      if (g_state == G_AFEED) begin
        acc_rd_en   <= 1'b1;
        acc_rd_addr <= '0;
      end else if (acc_rd_en) begin
        if (acc_rd_addr == AW'(PER_LANE - 1)) acc_rd_en <= 1'b0;
        else acc_rd_addr <= acc_rd_addr + 1'b1;
      end
    end
  end

  // =========================================================================
  // STAGE CONTROLLERS: the mesh, activation (G) and pooling (P) each hold one set
  // =========================================================================
  logic streaming_complete;

  assign pipeline_ready_o = (credits != 0) && mesh_input_ready;
  assign host_accept = start_pipeline_i && pipeline_ready_o && !north_queue_empty && !west_queue_empty;
  // Not while the previous result's read or release is in flight: its bank flag may still read full.
  assign g_accept = (g_state == G_IDLE) && systolic_collection_complete && !act_full[act_wr] &&
                    !systolic_read_enable && !systolic_release;
  assign g_done = (g_state == G_ROUND) && all_collected;
  logic g_null_done;  // a partial set has been summed; it takes an activation bank only to keep sets in order
  assign g_null_done = (g_state == G_ACC_WAIT) && acc_done_w && set_accum[g_set_id];
  // A partial set passes pooling in one cycle, never right after another completion, so pulses stay one cycle.
  logic complete_q, p_null;
  assign p_null = (p_state == P_IDLE) && act_full[act_rd] && act_null[act_rd] && !complete_q;
  assign p_accept = (p_state == P_IDLE) && act_full[act_rd] && !act_null[act_rd];
  assign p_release = (p_state == P_DISPATCH) && disp_done;  // the bank is copied into FIFO2
  assign pool_done = (p_state == P_WAIT) && streaming_complete;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      g_state   <= G_IDLE;
      p_state   <= P_IDLE;
      act_full  <= '0;
      act_null  <= '0;
      complete_q <= 1'b0;
      act_wr    <= 1'b0;
      act_rd    <= 1'b0;
      credits   <= MAX_SETS_IN_FLIGHT[1:0];
      mesh_sets <= '0;
      g_next_id <= '0;
      g_set_id  <= '0;
      p_next_id <= '0;
      p_set_id  <= '0;
      host_next_id <= '0;
      for (int k = 0; k < 4; k++) begin
        set_train[k] <= 1'b0;
        set_seed[k]  <= '1;
        set_accum[k] <= 1'b0;
      end
    end else begin
      complete_q <= pipeline_complete_o;
      if (host_accept) begin
        set_accum[host_next_id] <= accumulate_i;
        set_train[host_next_id] <= training_mode_i;
        set_seed[host_next_id]  <= dropout_seed_i;
        host_next_id <= host_next_id + 1'b1;
      end
      case (g_state)
        // A set goes through acc_mem if it is a partial or a sum is pending; otherwise straight to the lanes.
        G_IDLE:     if (g_accept) g_state <= (set_accum[g_next_id] || acc_valid) ? G_ACC_RD : G_FEED;
        G_FEED:     g_state <= G_LATCH;
        G_LATCH:    g_state <= G_ROUND;
        G_ROUND:    if (all_collected) g_state <= G_IDLE;
        G_ACC_RD:   g_state <= G_ACC_WAIT;
        G_ACC_WAIT: if (acc_done_w) g_state <= set_accum[g_set_id] ? G_IDLE : G_AFEED;
        G_AFEED:    g_state <= G_LATCH;
        default:    g_state <= G_IDLE;
      endcase
      case (p_state)
        P_IDLE:     if (p_accept) p_state <= P_DISPATCH;
        P_DISPATCH: if (disp_done) p_state <= P_WAIT;
        P_WAIT:     if (streaming_complete) p_state <= P_IDLE;
        default:    p_state <= P_IDLE;
      endcase
      if (g_done || g_null_done) begin
        act_full[act_wr] <= 1'b1;
        act_null[act_wr] <= g_null_done;
        act_wr <= ~act_wr;
      end
      if (p_release || p_null) begin
        act_full[act_rd] <= 1'b0;
        act_rd <= ~act_rd;
      end
      credits   <= credits - {1'b0, host_accept} + {1'b0, pipeline_complete_o};
      mesh_sets <= mesh_sets + {2'b0, host_accept} - {2'b0, g_accept};
      if (g_accept) begin
        g_set_id  <= g_next_id;
        g_next_id <= g_next_id + 1'b1;
      end
      if (p_accept || p_null) begin
        p_set_id  <= p_next_id;
        p_next_id <= p_next_id + 1'b1;
      end
    end
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
      if (g_state == G_FEED || g_state == G_ACC_RD) begin
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
  assign fill_v = (wide_rd_valid && !acc_phase) || acc_rd_valid;
  assign fill_d = acc_rd_valid ? acc_rd_data : wide_rd_data;

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
      if (fill_v) begin
        filled_total_n = filled_total + NUM_LANES[TOT_W-1:0];
        for (int i = 0; i < NUM_LANES; i++) begin
          gpnae_signal_n[i] = fill_d[i];
          gpnae_wr_en_n[i]  = 1'b1;
          fill_count_n[i]   = fill_count[i] + 1'b1;
        end
      end
      // Start every lane together, the cycle after its last element is written.
      for (int i = 0; i < NUM_LANES; i++) begin
        if ((fill_count[i] == PER_LANE[FCNT_W-1:0]) && !load_finalized[i]) begin
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
          if (mp_windows_done[i] < lane_windows_total[i]) begin
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
  a_credit_range: assert property (@(posedge clk_i) disable iff (!rstn_i) credits <= MAX_SETS_IN_FLIGHT)
    else $error("sienna_top: more credits than MAX_SETS_IN_FLIGHT");
  a_credit_accept: assert property (@(posedge clk_i) disable iff (!rstn_i) host_accept |-> credits != 0)
    else $error("sienna_top: a start was accepted without a credit");
  a_credit_return: assert property (@(posedge clk_i) disable iff (!rstn_i) pool_done |-> credits < MAX_SETS_IN_FLIGHT)
    else $error("sienna_top: a set finished with every credit already free");
  a_mesh_takes_start: assert property (@(posedge clk_i) disable iff (!rstn_i) systolic_start |-> mesh_input_ready)
    else $error("sienna_top: a start was forwarded to a mesh with no free staging bank");
  a_g_from_mesh: assert property (@(posedge clk_i) disable iff (!rstn_i) g_accept |-> mesh_sets != 0)
    else $error("sienna_top: the activation stage took a result the host never started");
  a_act_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) g_done |-> !act_full[act_wr])
    else $error("sienna_top: activation finished into a full bank");
  a_act_bank_full: assert property (@(posedge clk_i) disable iff (!rstn_i) p_release |-> act_full[act_rd])
    else $error("sienna_top: pooling released an empty bank");
  a_acc_add_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_sum_v[0] |-> (acc_pend != '0))
    else $error("sienna_top: an accumulation add returned with none outstanding");
  a_acc_no_lane_fill: assert property (@(posedge clk_i) disable iff (!rstn_i) acc_phase |-> !acc_rd_valid)
    else $error("sienna_top: the sum was read back while a mesh result was being summed");
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
