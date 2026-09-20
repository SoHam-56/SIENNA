`timescale 1ns / 100ps

module sienna_top #(
    parameter int    NUM_LANES         = 8,
    parameter int    N                 = 16,
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
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [     ADDR_LINES:0] num_terms_i,
    input logic                     north_write_enable_i,
    input logic [   DATA_WIDTH-1:0] north_write_data_i,
    input logic                     north_write_reset_i,
    input logic                     west_write_enable_i,
    input logic [   DATA_WIDTH-1:0] west_write_data_i,
    input logic                     west_write_reset_i,

    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] final_result_o,

    output logic pipeline_complete_o,
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
  localparam int ROUND_CAPACITY = NUM_LANES * GPNAE_FIFO_DEPTH;

  localparam int FCNT_W = $clog2(GPNAE_FIFO_DEPTH + 1);
  localparam int PTR_W = $clog2(NUM_LANES);
  localparam int TOT_W = $clog2(SRAM_DEPTH + 1);
  localparam int RND_W = $clog2(ROUND_CAPACITY + 1);

  localparam int FIFO1_DEPTH = SRAM_DEPTH;
  localparam int FIFO2_DEPTH = 16;

  localparam int MAXPOOL_IN_COUNT = IN_ROWS * IN_COLS;
  localparam int POOL_OUT_ROWS = (IN_ROWS + 2 * PADDING - POOL_H) / STRIDE_ROWS + 1;
  localparam int POOL_OUT_COLS = (IN_COLS + 2 * PADDING - POOL_W) / STRIDE_COLS + 1;
  localparam int MAXPOOL_OUT_COUNT = POOL_OUT_ROWS * POOL_OUT_COLS;

  typedef enum logic [3:0] {
    IDLE,
    SYSTOLIC_START_PULSE,
    SYSTOLIC_PROCESSING,
    FEED_GPNAE_FIFO,
    LATCH_GPNAE_COUNT,
    GPNAE_ROUND,
    DISPATCH_WINDOWS,
    WAIT_DOWNSTREAM,
    PIPELINE_COMPLETE
  } pipeline_state_t;

  pipeline_state_t current_state, next_state;

  typedef enum logic [1:0] {
    F_WRITE,
    F_PULSE,
    F_ROUND_IDLE
  } fill_state_t;

  fill_state_t fill_state, fill_state_n;

  // Intermediate Memory Buffer
  logic [DATA_WIDTH-1:0] gpnae_out_mem  [0:SRAM_DEPTH-1];

  logic                  fifo1_rd_ready;
  logic [DATA_WIDTH-1:0] fifo1_rd_data;
  logic                  fifo1_rd_valid;
  logic [     TOT_W-1:0] fifo1_count;

  logic                  systolic_start;
  logic systolic_read_enable, systolic_read_enable_next;
  logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr, systolic_read_addr_next;
  logic [DATA_WIDTH-1:0] systolic_read_data;
  logic                  systolic_read_valid;
  logic                  systolic_mult_complete;
  logic                  systolic_collection_complete;
  logic systolic_reading, systolic_reading_next;
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

  logic [           PTR_W-1:0] fill_ptr;
  logic [           TOT_W-1:0] total_elements;
  logic [           TOT_W-1:0] filled_total;
  logic [           RND_W-1:0] fill_round_total;

  logic [          FCNT_W-1:0] fill_count_n       [NUM_LANES];
  logic [          FCNT_W-1:0] done_count_n       [NUM_LANES];
  logic                        load_finalized_n   [NUM_LANES];
  logic                        gpnae_start_n      [NUM_LANES];
  logic                        gpnae_wr_en_n      [NUM_LANES];
  logic [      DATA_WIDTH-1:0] gpnae_signal_n     [NUM_LANES];
  logic [           PTR_W-1:0] fill_ptr_n;
  logic [           TOT_W-1:0] filled_total_n;
  logic [           RND_W-1:0] fill_round_total_n;

  logic                        round_done;
  logic [15:0] round_number, round_number_n;
  logic [$clog2(FIFO2_DEPTH):0] fifo2_count[NUM_LANES];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      total_elements <= 0;
      round_number   <= 0;
    end else begin
      round_number <= round_number_n;
      if (current_state == IDLE) total_elements <= 0;
      else if (current_state == LATCH_GPNAE_COUNT) total_elements <= SRAM_DEPTH[TOT_W-1:0];
    end
  end

  always_comb begin
    round_done = 1'b1;
    for (int i = 0; i < NUM_LANES; i++) begin
      if (load_finalized[i] && !lane_collected[i]) round_done = 0;
    end
  end

  logic all_collected;
  assign all_collected = (filled_total >= total_elements) && round_done && (total_elements > 0);

  SystolicMesh #(
      .MATRIX_SIZE(N),
      .TILE_SIZE  (4),
      .DATA_WIDTH (DATA_WIDTH)
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
      .read_enable_i         (systolic_read_enable),
      .read_addr_i           (32'(systolic_read_addr)),
      .read_data_o           (systolic_read_data),
      .read_valid_o          (systolic_read_valid),
      .collection_complete_o (systolic_collection_complete),
      .collection_active_o   ()
  );

  fwft #(
      .DATA_WIDTH(DATA_WIDTH),
      .FIFO_DEPTH(FIFO1_DEPTH)
  ) fifo1_inst (
      .clk_i     (clk_i),
      .rstn_i    (rstn_i),
      .wr_valid_i(systolic_read_valid),
      .wr_data_i (systolic_read_data),
      .wr_ready_o(),
      .rd_ready_i(fifo1_rd_ready),
      .rd_data_o (fifo1_rd_data),
      .rd_valid_o(fifo1_rd_valid),
      .count_o   (fifo1_count)
  );

  generate
    genvar g;
    for (g = 0; g < NUM_LANES; g++) begin : backend_lanes

      gpnae #(
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

      dropout #(
          .DATA_WIDTH       (DATA_WIDTH),
          .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
          .LFSR_WIDTH       (LFSR_WIDTH)
      ) dropout_inst (
          .clk          (clk_i),
          .rst_n        (rstn_i),
          .in_valid     (dropout_in_valid[g]),
          .training_mode(1'b0),
          .data_in      (dropout_data_in[g]),
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
    if (current_state == GPNAE_ROUND) begin
      for (int i = 0; i < NUM_LANES; i++) begin
        if (load_finalized[i] && gpnae_done[i] && (done_count[i] < fill_count[i])) begin
          // RESTORED: This is the mathematically perfect chunked indexing!
          gpnae_out_mem[round_number * ROUND_CAPACITY + i * GPNAE_FIFO_DEPTH + done_count[i]] <= gpnae_result[i];
        end
      end
    end
  end

  // =========================================================================
  // WINDOW DISPATCHER FSM
  // =========================================================================
  logic [    $clog2(POOL_OUT_ROWS+1)-1:0] disp_r;
  logic [    $clog2(POOL_OUT_COLS+1)-1:0] disp_c;
  logic [           $clog2(POOL_H+1)-1:0] disp_pr;
  logic [           $clog2(POOL_W+1)-1:0] disp_pc;
  logic                                   disp_done;
  logic [$clog2(MAXPOOL_OUT_COUNT+1)-1:0] window_idx;

  logic [          $clog2(NUM_LANES)-1:0] target_lane;
  assign target_lane = window_idx % NUM_LANES;

  logic signed [31:0] in_row, in_col;
  logic in_bounds;
  logic [DATA_WIDTH-1:0] disp_val;

  assign in_row = signed'(disp_r * STRIDE_ROWS + disp_pr) - signed'(PADDING);
  assign in_col = signed'(disp_c * STRIDE_COLS + disp_pc) - signed'(PADDING);
  assign in_bounds = (in_row >= 0) && (in_row < IN_ROWS) && (in_col >= 0) && (in_col < IN_COLS);

  // RESTORED: Standard Row-Major indexing
  assign disp_val = in_bounds ? gpnae_out_mem[in_row*IN_COLS+in_col] : 32'hFF800000;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i || current_state == IDLE) begin
      disp_r <= '0;
      disp_c <= '0;
      disp_pr <= '0;
      disp_pc <= '0;
      window_idx <= '0;
      disp_done <= '0;
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;
    end else if (current_state == DISPATCH_WINDOWS) begin
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;

      if (!disp_done) begin
        if (fifo2_count[target_lane] <= (FIFO2_DEPTH - 4)) begin
          fifo2_wr_data[target_lane]  <= disp_val;
          fifo2_wr_valid[target_lane] <= 1'b1;

          if (disp_pc < POOL_W - 1) begin
            disp_pc <= disp_pc + 1;
          end else begin
            disp_pc <= '0;
            if (disp_pr < POOL_H - 1) begin
              disp_pr <= disp_pr + 1;
            end else begin
              disp_pr <= '0;
              if (window_idx < MAXPOOL_OUT_COUNT - 1) window_idx <= window_idx + 1;
              if (disp_c < POOL_OUT_COLS - 1) begin
                disp_c <= disp_c + 1;
              end else begin
                disp_c <= '0;
                if (disp_r < POOL_OUT_ROWS - 1) begin
                  disp_r <= disp_r + 1;
                end else begin
                  disp_done <= 1'b1;
                end
              end
            end
          end
        end
      end
    end else begin
      for (int i = 0; i < NUM_LANES; i++) fifo2_wr_valid[i] <= 1'b0;
    end
  end

  // =========================================================================
  // OUTER PIPELINE FSM
  // =========================================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) current_state <= IDLE;
    else current_state <= next_state;
  end

  logic streaming_complete;
  always_comb begin
    next_state = current_state;
    case (current_state)
      IDLE:
      if (start_pipeline_i && !north_queue_empty && !west_queue_empty)
        next_state = SYSTOLIC_START_PULSE;
      SYSTOLIC_START_PULSE: next_state = SYSTOLIC_PROCESSING;
      SYSTOLIC_PROCESSING: if (systolic_collection_complete) next_state = FEED_GPNAE_FIFO;
      FEED_GPNAE_FIFO: next_state = LATCH_GPNAE_COUNT;
      LATCH_GPNAE_COUNT: next_state = GPNAE_ROUND;
      GPNAE_ROUND: if (all_collected) next_state = DISPATCH_WINDOWS;
      DISPATCH_WINDOWS: if (disp_done) next_state = WAIT_DOWNSTREAM;
      WAIT_DOWNSTREAM: if (streaming_complete) next_state = PIPELINE_COMPLETE;
      PIPELINE_COMPLETE: if (!start_pipeline_i) next_state = IDLE;
      default: next_state = IDLE;
    endcase
  end

  // =========================================================================
  // SYSTOLIC READ-INTO-FIFO1
  // =========================================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      systolic_start       <= 0;
      systolic_read_enable <= 0;
      systolic_read_addr   <= 0;
      systolic_reading     <= 0;
    end else begin
      systolic_start       <= (current_state == SYSTOLIC_START_PULSE);
      systolic_read_enable <= systolic_read_enable_next;
      systolic_read_addr   <= systolic_read_addr_next;
      systolic_reading     <= systolic_reading_next;
    end
  end

  always_comb begin
    systolic_read_enable_next = systolic_read_enable;
    systolic_read_addr_next   = systolic_read_addr;
    systolic_reading_next     = systolic_reading;

    if (current_state == IDLE) begin
      systolic_read_enable_next = 0;
      systolic_read_addr_next   = 0;
      systolic_reading_next     = 0;
    end else begin
      if (current_state == FEED_GPNAE_FIFO) systolic_reading_next = 1;
      if (systolic_reading) begin
        systolic_read_enable_next = 1;
        if (systolic_read_enable) begin
          if (systolic_read_addr < N * N - 1) systolic_read_addr_next = systolic_read_addr + 1;
          else begin
            systolic_read_enable_next = 0;
            systolic_reading_next     = 0;
          end
        end
      end else systolic_read_enable_next = 0;
    end
  end

  // =========================================================================
  // GPNAE_ROUND — Independent Fill FSM
  // =========================================================================
  always_comb begin
    fill_state_n       = fill_state;
    fill_ptr_n         = fill_ptr;
    filled_total_n     = filled_total;
    fill_round_total_n = fill_round_total;
    round_number_n     = round_number;

    for (int i = 0; i < NUM_LANES; i++) begin
      fill_count_n[i]     = fill_count[i];
      done_count_n[i]     = done_count[i];
      load_finalized_n[i] = load_finalized[i];
      gpnae_start_n[i]    = 1'b0;
      gpnae_wr_en_n[i]    = 1'b0;
      gpnae_signal_n[i]   = gpnae_signal_i[i];
      lane_collected_n[i] = lane_collected[i];
    end

    if (current_state == IDLE) begin
      fill_state_n       = F_WRITE;
      fill_ptr_n         = '0;
      filled_total_n     = '0;
      fill_round_total_n = '0;
      round_number_n     = '0;
      for (int i = 0; i < NUM_LANES; i++) begin
        fill_count_n[i]     = '0;
        done_count_n[i]     = '0;
        load_finalized_n[i] = 1'b0;
        gpnae_start_n[i]    = 1'b0;
        lane_collected_n[i] = 1'b0;
      end
    end else if (current_state == GPNAE_ROUND) begin

      case (fill_state)
        F_WRITE: begin
          if (fifo1_rd_valid && !gpnae_full[fill_ptr] && (filled_total < total_elements) && (fill_round_total < ROUND_CAPACITY[RND_W-1:0])) begin
            gpnae_signal_n[fill_ptr] = fifo1_rd_data;
            gpnae_wr_en_n[fill_ptr]  = 1'b1;
            fill_count_n[fill_ptr]   = fill_count[fill_ptr] + 1'b1;
            filled_total_n           = filled_total + 1'b1;
            fill_round_total_n       = fill_round_total + 1'b1;
            // Stop conditions evaluated from the post-write values, so no extra cycle.
            if (((fill_count[fill_ptr] + 1'b1) >= GPNAE_FIFO_DEPTH[FCNT_W-1:0]) ||
                ((filled_total + 1'b1) >= total_elements) ||
                ((fill_round_total + 1'b1) >= ROUND_CAPACITY[RND_W-1:0]))
              fill_state_n = F_PULSE;
            else fill_state_n = F_WRITE;
          end else if (fill_count[fill_ptr] != '0) begin
            // Cannot write and the lane already holds data: release it, as F_GAP did.
            fill_state_n = F_PULSE;
          end
        end
        F_PULSE: begin
          gpnae_start_n[fill_ptr]    = 1'b1;
          load_finalized_n[fill_ptr] = 1'b1;
          if (fifo1_rd_valid && (filled_total < total_elements) &&
             (fill_round_total < ROUND_CAPACITY[RND_W-1:0]) && (fill_ptr < NUM_LANES[PTR_W:0] - 1)) begin
            fill_ptr_n   = fill_ptr + 1'b1;
            fill_state_n = F_WRITE;
          end else fill_state_n = F_ROUND_IDLE;
        end
        F_ROUND_IDLE: ;
        default: fill_state_n = F_WRITE;
      endcase

      for (int i = 0; i < NUM_LANES; i++) begin
        if (load_finalized[i] && gpnae_done[i] && (done_count[i] < fill_count[i])) begin
          done_count_n[i] = done_count[i] + 1'b1;
          if ((done_count[i] + 1'b1) == fill_count[i]) lane_collected_n[i] = 1'b1;
        end
      end

      if (round_done && (fill_state == F_ROUND_IDLE) && !all_collected) begin
        fill_state_n       = F_WRITE;
        fill_ptr_n         = '0;
        fill_round_total_n = '0;
        round_number_n     = round_number + 1'b1;
        for (int i = 0; i < NUM_LANES; i++) begin
          fill_count_n[i]     = '0;
          done_count_n[i]     = '0;
          load_finalized_n[i] = 1'b0;
          gpnae_start_n[i]    = 1'b0;
          lane_collected_n[i] = 1'b0;
        end
      end
    end else begin
      for (int i = 0; i < NUM_LANES; i++) gpnae_start_n[i] = 1'b0;
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      fill_state       <= F_WRITE;
      fill_ptr         <= '0;
      filled_total     <= '0;
      fill_round_total <= '0;
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
      fill_state       <= fill_state_n;
      fill_ptr         <= fill_ptr_n;
      filled_total     <= filled_total_n;
      fill_round_total <= fill_round_total_n;
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

  assign fifo1_rd_ready = (current_state == GPNAE_ROUND) && (fill_state == F_WRITE) && fifo1_rd_valid &&
                          !gpnae_full[fill_ptr] && (filled_total < total_elements) && (fill_round_total < ROUND_CAPACITY[RND_W-1:0]);

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
          end else if (current_state == WAIT_DOWNSTREAM || current_state == PIPELINE_COMPLETE) begin
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
    end else if (current_state == IDLE) begin
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
    end else if (current_state == IDLE) begin
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
    end else begin
      for (int i = 0; i < NUM_LANES; i++) begin
        if (dropout_valid_out[i]) final_result_o[i] <= dropout_data_out[i];
      end
    end
  end

  assign pipeline_complete_o = (current_state == PIPELINE_COMPLETE);
  assign intermediate_buffer_full_o = (fifo1_count == SRAM_DEPTH[TOT_W-1:0]);
  assign intermediate_buffer_empty_o = (fifo1_count == 0);

  assign systolic_busy_o = (current_state == SYSTOLIC_START_PULSE ||
                            current_state == SYSTOLIC_PROCESSING  ||
                            current_state == FEED_GPNAE_FIFO);
  assign gpnae_busy_o = (current_state == LATCH_GPNAE_COUNT || current_state == GPNAE_ROUND);

  logic any_mp_active;
  always_comb begin
    any_mp_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++)
    if (mp_state[i] != MP_DONE && mp_state[i] != MP_IDLE) any_mp_active = 1'b1;
  end
  assign maxpool_busy_o = (current_state == DISPATCH_WINDOWS || current_state == WAIT_DOWNSTREAM) && any_mp_active;

  logic any_dropout_active;
  always_comb begin
    any_dropout_active = 1'b0;
    for (int i = 0; i < NUM_LANES; i++) if (dropout_valid_out[i]) any_dropout_active = 1'b1;
  end
  assign dropout_busy_o = any_dropout_active;

endmodule
