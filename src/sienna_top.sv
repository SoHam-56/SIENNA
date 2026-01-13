`timescale 1ns / 1ps

module sienna_top #(
    parameter N                 = 32,
    parameter DATA_WIDTH        = 32,
    parameter SRAM_DEPTH        = N * N,
    parameter FIFO_DEPTH        = 32,
    parameter ADDR_LINES        = $clog2(FIFO_DEPTH),
    parameter CONTROL_WIDTH     = 2,
    parameter IN_ROWS           = 5,
    parameter IN_COLS           = 5,
    parameter POOL_H            = 2,
    parameter POOL_W            = 2,
    parameter STRIDE_ROWS       = 2,
    parameter STRIDE_COLS       = 2,
    parameter PADDING           = 1,
    parameter DROPOUT_P_PERCENT = 50,
    parameter LFSR_WIDTH        = 32,
    parameter INPUT_A_FILE      = "matrixA.mem",
    parameter INPUT_B_FILE      = "matrixB.mem"
) (
    input logic clk_i,
    input logic rstn_i,

    // Inputs
    input logic start_pipeline_i,
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [ADDR_LINES:0] num_terms_i,
    input logic north_write_enable_i,
    input logic [DATA_WIDTH-1:0] north_write_data_i,
    input logic north_write_reset_i,
    input logic west_write_enable_i,
    input logic [DATA_WIDTH-1:0] west_write_data_i,
    input logic west_write_reset_i,

    // Outputs
    output logic [DATA_WIDTH-1:0] final_result_o,
    output logic pipeline_complete_o,
    output logic gpnae_done_o,
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic maxpool_busy_o,
    output logic dropout_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o
);

  // ============================================================
  // Derived Parameters & Internal Wires
  // ============================================================
  localparam int MAXPOOL_IN_SIZE = IN_ROWS * IN_COLS;

  // -- FSM States --
  typedef enum logic [3:0] {
    IDLE,
    SYSTOLIC_START_PULSE,
    SYSTOLIC_PROCESSING,
    FEED_GPNAE_FIFO,
    LATCH_GPNAE_COUNT,
    DRAIN_FIFO_TO_GPNAE,
    COLLECT_GPNAE,
    PREP_MAXPOOL,
    FEED_MAXPOOL,
    MAXPOOL_PROCESSING,
    COLLECT_MAXPOOL,
    DROPOUT_PROCESSING,
    PIPELINE_COMPLETE
  } pipeline_state_t;

  pipeline_state_t current_state, next_state;

  // -- FIFO Counters (Status inputs to FSM) --
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo1_count;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo2_count;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo3_count;

  // -- Internal Data Wires --
  logic [DATA_WIDTH-1:0] systolic_read_data;
  logic systolic_read_valid;
  logic systolic_mult_complete;
  logic systolic_collection_complete;
  logic north_queue_empty, west_queue_empty;

  logic [DATA_WIDTH-1:0] fifo1_rd_data;
  logic fifo1_full, fifo1_empty;

  logic [DATA_WIDTH-1:0] gpnae_final_result;
  logic gpnae_full, gpnae_empty, gpnae_idle;
  logic gpnae_done_signal;

  logic [DATA_WIDTH-1:0] fifo2_rd_data;
  logic fifo2_full, fifo2_empty;

  logic [DATA_WIDTH-1:0] maxpool_out_data;
  logic maxpool_out_valid;
  logic maxpool_done_signal;

  logic [DATA_WIDTH-1:0] fifo3_rd_data;
  logic fifo3_full, fifo3_empty;

  logic [DATA_WIDTH-1:0] dropout_data_out;
  logic dropout_valid_out;

  // ============================================================
  // REGISTERS & SIGNALS
  // ============================================================

  // 1. Combinational Control Signals (FIX 2: Driven directly by logic)
  logic fifo1_wr_en;
  logic fifo3_wr_en;

  // 2. Registered Control Signals (Driven by FSM State)
  logic systolic_start, systolic_start_next;
  logic systolic_read_enable, systolic_read_enable_next;
  logic fifo1_rd_en, fifo1_rd_en_next;
  logic gpnae_wr_en, gpnae_wr_en_next;
  logic gpnae_last, gpnae_last_next;
  logic fifo2_wr_en, fifo2_wr_en_next;
  logic fifo2_rd_en, fifo2_rd_en_next;
  logic maxpool_start, maxpool_start_next;
  logic maxpool_valid_in, maxpool_valid_in_next;
  logic fifo3_rd_en, fifo3_rd_en_next;
  logic dropout_en, dropout_en_next;

  // 3. Data/Count Registers
  logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr, systolic_read_addr_next;
  logic [$clog2(FIFO_DEPTH+1)-1:0] items_left_to_process, items_left_to_process_next;
  logic [$clog2(FIFO_DEPTH+1)-1:0] items_expected_back, items_expected_back_next;
  logic [DATA_WIDTH-1:0] maxpool_data_in, maxpool_data_in_next;
  logic [DATA_WIDTH-1:0] dropout_data_in, dropout_data_in_next;
  logic [DATA_WIDTH-1:0] final_output_reg, final_output_reg_next;


  // ============================================================
  // Module Instantiations
  // ============================================================

  SystolicArray #(
      .N(N),
      .DATA_WIDTH(DATA_WIDTH),
      .ROWS(INPUT_A_FILE),
      .COLS(INPUT_B_FILE)
  ) systolic_array_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .start_matrix_mult_i(systolic_start),
      .north_write_enable_i(north_write_enable_i),
      .north_write_data_i(north_write_data_i),
      .north_write_reset_i(north_write_reset_i),
      .west_write_enable_i(west_write_enable_i),
      .west_write_data_i(west_write_data_i),
      .west_write_reset_i(west_write_reset_i),
      .north_queue_empty_o(north_queue_empty),
      .west_queue_empty_o(west_queue_empty),
      .matrix_mult_complete_o(systolic_mult_complete),
      .read_enable_i(systolic_read_enable),
      .read_addr_i(systolic_read_addr),
      .read_data_o(systolic_read_data),
      .read_valid_o(systolic_read_valid),
      .collection_complete_o(systolic_collection_complete),
      .collection_active_o()
  );

  fwft #(
      .DATA_WIDTH(DATA_WIDTH),
      .FIFO_DEPTH(FIFO_DEPTH)
  ) fifo1_inst (
      .clk_i  (clk_i),
      .rstn_i (rstn_i),
      .wr_en_i(fifo1_wr_en),
      .data_i (systolic_read_data),
      .full_o (fifo1_full),
      .rd_en_i(fifo1_rd_en),
      .data_o (fifo1_rd_data),
      .empty_o(fifo1_empty),
      .count_o(fifo1_count)
  );

  gpnae #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_LINES(ADDR_LINES + 1),
      .CONTROL_WIDTH(CONTROL_WIDTH)
  ) gpnae_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .signal_i(fifo1_rd_data),
      .wr_en_i(gpnae_wr_en),
      .last_i(gpnae_last),
      .terms_i(num_terms_i),
      .control_word_i(activation_function_i),
      .full_o(gpnae_full),
      .empty_o(gpnae_empty),
      .idle_o(gpnae_idle),
      .final_result_o(gpnae_final_result),
      .done_o(gpnae_done_signal)
  );

  fwft #(
      .DATA_WIDTH(DATA_WIDTH),
      .FIFO_DEPTH(FIFO_DEPTH)
  ) fifo2_inst (
      .clk_i  (clk_i),
      .rstn_i (rstn_i),
      .wr_en_i(fifo2_wr_en),
      .data_i (gpnae_final_result),
      .full_o (fifo2_full),
      .rd_en_i(fifo2_rd_en),
      .data_o (fifo2_rd_data),
      .empty_o(fifo2_empty),
      .count_o(fifo2_count)
  );

  Maxpool_2D #(
      .DATA_WIDTH(DATA_WIDTH),
      .IN_ROWS(IN_ROWS),
      .IN_COLS(IN_COLS),
      .SEG_ROWS(POOL_H),
      .SEG_COLS(POOL_W),
      .STRIDE_ROWS(STRIDE_ROWS),
      .STRIDE_COLS(STRIDE_COLS),
      .PADDING(PADDING)
  ) maxpool_inst (
      .clk(clk_i),
      .rst_n(rstn_i),
      .start(maxpool_start),
      .done(maxpool_done_signal),
      .data_in(maxpool_data_in),
      .valid_in(maxpool_valid_in),
      .out_data(maxpool_out_data),
      .out_valid(maxpool_out_valid)
  );

  fwft #(
      .DATA_WIDTH(DATA_WIDTH),
      .FIFO_DEPTH(FIFO_DEPTH)
  ) fifo3_inst (
      .clk_i  (clk_i),
      .rstn_i (rstn_i),
      .wr_en_i(fifo3_wr_en),
      .data_i (maxpool_out_data),
      .full_o (fifo3_full),
      .rd_en_i(fifo3_rd_en),
      .data_o (fifo3_rd_data),
      .empty_o(fifo3_empty),
      .count_o(fifo3_count)
  );

  dropout #(
      .DATA_WIDTH(DATA_WIDTH),
      .DROPOUT_P_PERCENT(DROPOUT_P_PERCENT),
      .LFSR_WIDTH(LFSR_WIDTH)
  ) dropout_inst (
      .clk(clk_i),
      .rst_n(rstn_i),
      .en(dropout_en),
      .training_mode(1'b0),
      .data_in(dropout_data_in),
      .data_out(dropout_data_out),
      .valid_out(dropout_valid_out)
  );

  // ============================================================
  // FSM Process 1: State Register
  // ============================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) current_state <= IDLE;
    else current_state <= next_state;
  end

  // ============================================================
  // FSM Process 2: Next State Logic
  // ============================================================
  always_comb begin
    next_state = current_state;
    case (current_state)
      IDLE:
      if (start_pipeline_i && !north_queue_empty && !west_queue_empty)
        next_state = SYSTOLIC_START_PULSE;
      SYSTOLIC_START_PULSE: next_state = SYSTOLIC_PROCESSING;
      SYSTOLIC_PROCESSING: if (systolic_collection_complete) next_state = FEED_GPNAE_FIFO;

      FEED_GPNAE_FIFO: begin
        // If we are reading the last address, we move on
        if (systolic_read_enable && systolic_read_addr >= N * N - 1) next_state = LATCH_GPNAE_COUNT;
      end

      LATCH_GPNAE_COUNT:   next_state = DRAIN_FIFO_TO_GPNAE;
      DRAIN_FIFO_TO_GPNAE: if (items_left_to_process == 0) next_state = COLLECT_GPNAE;
      COLLECT_GPNAE:       if (items_expected_back == 0) next_state = PREP_MAXPOOL;
      PREP_MAXPOOL:        next_state = FEED_MAXPOOL;
      FEED_MAXPOOL:        if (items_left_to_process == 0) next_state = MAXPOOL_PROCESSING;
      MAXPOOL_PROCESSING:  if (maxpool_done_signal) next_state = COLLECT_MAXPOOL;
      COLLECT_MAXPOOL:     if (fifo3_count >= 1) next_state = DROPOUT_PROCESSING;
      DROPOUT_PROCESSING:  if (fifo3_empty) next_state = PIPELINE_COMPLETE;
      PIPELINE_COMPLETE:   if (!start_pipeline_i) next_state = IDLE;
      default:             next_state = IDLE;
    endcase
  end

  // ============================================================
  // FSM Process 3a: DATAPATH NEXT-VALUE LOGIC (Combinational)
  // ============================================================
  always_comb begin
    // --- 1. Pure Combinational Handshaking (FIX 2) ---
    // These respond immediately to inputs. They are NOT registered.
    fifo1_wr_en = 0;
    if (systolic_read_valid && !fifo1_full) fifo1_wr_en = 1;

    fifo3_wr_en = 0;
    if (maxpool_out_valid && !fifo3_full) fifo3_wr_en = 1;

    // --- 2. Registered Logic Defaults ---
    systolic_start_next        = 0;
    systolic_read_enable_next  = 0;
    fifo1_rd_en_next           = 0;
    gpnae_wr_en_next           = 0;
    gpnae_last_next            = 0;
    fifo2_wr_en_next           = 0;
    fifo2_rd_en_next           = 0;
    maxpool_start_next         = 0;
    maxpool_valid_in_next      = 0;
    fifo3_rd_en_next           = 0;
    dropout_en_next            = 0;

    // Hold previous values for data registers
    systolic_read_addr_next    = systolic_read_addr;
    items_left_to_process_next = items_left_to_process;
    items_expected_back_next   = items_expected_back;
    maxpool_data_in_next       = maxpool_data_in;
    dropout_data_in_next       = dropout_data_in;
    final_output_reg_next      = final_output_reg;

    // --- 3. Independent Logic (Data Capture) ---
    if (dropout_valid_out) begin
      final_output_reg_next = dropout_data_out;
    end

    // --- 4. State-Dependent Logic ---
    case (current_state)
      IDLE: begin
        systolic_read_addr_next = 0;
      end

      SYSTOLIC_START_PULSE: begin
        systolic_start_next = 1;
      end

      FEED_GPNAE_FIFO: begin
        if (!fifo1_full) begin
          // Assert enable
          systolic_read_enable_next = 1;

          // FIX 1: Only increment if we are ALREADY enabled.
          // This ensures Addr 0 is held for 1 cycle while enable goes high.
          if (systolic_read_enable) begin
            if (systolic_read_addr < N * N - 1) begin
              systolic_read_addr_next = systolic_read_addr + 1;
            end else begin
              systolic_read_enable_next = 0;
            end
          end
        end
      end

      LATCH_GPNAE_COUNT: begin
        items_left_to_process_next = fifo1_count;
        items_expected_back_next   = fifo1_count;
      end

      DRAIN_FIFO_TO_GPNAE: begin
        if (!fifo1_empty && !gpnae_full && items_left_to_process > 0) begin
          fifo1_rd_en_next = 1;
          gpnae_wr_en_next = 1;
          if (gpnae_wr_en) items_left_to_process_next = items_left_to_process - 1;

          if (items_left_to_process == 1) begin
            gpnae_last_next = 1;
          end
        end
      end

      COLLECT_GPNAE: begin
        if (gpnae_done_signal) begin
          fifo2_wr_en_next = 1;
          if ((fifo2_wr_en == 1'b1) && (items_expected_back > 0))
            items_expected_back_next = items_expected_back - 1;
        end
      end

      PREP_MAXPOOL: begin
        items_left_to_process_next = fifo2_count;
        maxpool_start_next = 1;
      end

      FEED_MAXPOOL: begin
        if (!fifo2_empty && items_left_to_process > 0) begin
          fifo2_rd_en_next = 1;
          maxpool_valid_in_next = 1;
          maxpool_data_in_next = fifo2_rd_data;
          items_left_to_process_next = items_left_to_process - 1;
        end
      end

      DROPOUT_PROCESSING: begin
        if (!fifo3_empty) begin
          fifo3_rd_en_next = 1;
          dropout_en_next = 1;
          dropout_data_in_next = fifo3_rd_data;
        end
      end
    endcase
  end


  // ============================================================
  // FSM Process 3b: DATAPATH REGISTERS (Sequential)
  // ============================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      systolic_start        <= 0;
      systolic_read_enable  <= 0;
      systolic_read_addr    <= 0;

      // FIFO Write Enables are NOT here anymore (pure combinational)
      fifo1_rd_en           <= 0;

      gpnae_wr_en           <= 0;
      gpnae_last            <= 0;

      fifo2_wr_en           <= 0;
      fifo2_rd_en           <= 0;

      maxpool_start         <= 0;
      maxpool_valid_in      <= 0;
      maxpool_data_in       <= 0;

      fifo3_rd_en           <= 0;

      dropout_en            <= 0;
      dropout_data_in       <= 0;
      final_output_reg      <= 0;

      items_left_to_process <= 0;
      items_expected_back   <= 0;
    end else begin
      // Update registers
      systolic_start        <= systolic_start_next;
      systolic_read_enable  <= systolic_read_enable_next;
      systolic_read_addr    <= systolic_read_addr_next;

      fifo1_rd_en           <= fifo1_rd_en_next;

      gpnae_wr_en           <= gpnae_wr_en_next;
      gpnae_last            <= gpnae_last_next;

      fifo2_wr_en           <= fifo2_wr_en_next;
      fifo2_rd_en           <= fifo2_rd_en_next;

      maxpool_start         <= maxpool_start_next;
      maxpool_valid_in      <= maxpool_valid_in_next;
      maxpool_data_in       <= maxpool_data_in_next;

      fifo3_rd_en           <= fifo3_rd_en_next;

      dropout_en            <= dropout_en_next;
      dropout_data_in       <= dropout_data_in_next;
      final_output_reg      <= final_output_reg_next;

      items_left_to_process <= items_left_to_process_next;
      items_expected_back   <= items_expected_back_next;
    end
  end

  // ============================================================
  // Status Outputs
  // ============================================================
  assign final_result_o = final_output_reg;
  assign gpnae_done_o = (current_state == COLLECT_GPNAE) && gpnae_done_signal;
  assign pipeline_complete_o = (current_state == PIPELINE_COMPLETE);
  assign intermediate_buffer_full_o = fifo1_full;
  assign intermediate_buffer_empty_o = fifo1_empty;

  // Busy Logic
  assign systolic_busy_o = (current_state == SYSTOLIC_START_PULSE || 
                            current_state == SYSTOLIC_PROCESSING || 
                            current_state == FEED_GPNAE_FIFO);

  assign gpnae_busy_o    = (current_state == LATCH_GPNAE_COUNT || 
                            current_state == DRAIN_FIFO_TO_GPNAE || 
                            current_state == COLLECT_GPNAE);

  assign maxpool_busy_o  = (current_state == PREP_MAXPOOL || 
                            current_state == FEED_MAXPOOL || 
                            current_state == MAXPOOL_PROCESSING || 
                            current_state == COLLECT_MAXPOOL);

  assign dropout_busy_o = (current_state == DROPOUT_PROCESSING);

endmodule
