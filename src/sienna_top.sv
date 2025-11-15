`timescale 1ns / 1ps

module sienna_top #(
    // Systolic Array Parameters
    parameter N          = 32,
    parameter DATA_WIDTH = 32,
    parameter SRAM_DEPTH = N * N,

    // GPNAE Parameters
    parameter ADDR_LINES    = 5,
    parameter CONTROL_WIDTH = 2,

    // Maxpool Parameters
    parameter IN_ROWS = 5,
    parameter IN_COLS = 5,
    parameter POOL_H = 2,
    parameter POOL_W = 2,
    parameter STRIDE_ROWS = 2,
    parameter STRIDE_COLS = 2,
    parameter PADDING = 1,

    // Dropout Parameters
    parameter real DROPOUT_P = 0.5,
    parameter LFSR_WIDTH = 32,

    // Memory Parameters
    parameter INPUT_A_FILE = "matrixA.mem",
    parameter INPUT_B_FILE = "matrixB.mem",
    parameter INTERMEDIATE_BUFFER_DEPTH = SRAM_DEPTH * 2,
    parameter FIFO_DEPTH = 16
) (
    input logic clk_i,
    input logic rstn_i,

    // Top-level Control
    input logic                     start_pipeline_i,
    input logic [CONTROL_WIDTH-1:0] activation_function_i,
    input logic [   ADDR_LINES-1:0] num_terms_i,

    // Systolic Array Input Interface
    input logic north_write_enable_i,
    input logic [DATA_WIDTH-1:0] north_write_data_i,
    input logic north_write_reset_i,
    input logic west_write_enable_i,
    input logic [DATA_WIDTH-1:0] west_write_data_i,
    input logic west_write_reset_i,

    // Maxpool SRAM Interface (backward compatibility)
    input logic [DATA_WIDTH-1:0] maxpool_input_data_i,
    input logic maxpool_input_valid_i,

    // Final Output Interface
    output logic [DATA_WIDTH-1:0] final_result_o,
    output logic pipeline_complete_o,
    output logic gpnae_done_o,

    // Status Outputs
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic maxpool_busy_o,
    output logic dropout_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o,

    // Debug/Monitoring Outputs
    output logic [DATA_WIDTH-1:0] systolic_result_debug_o,
    output logic systolic_complete_debug_o,
    output logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer_count_debug_o
);

  // ============================================================
  // Calculate Maxpool Output Dimensions
  // ============================================================
  localparam int MAXPOOL_IN_SIZE = IN_ROWS * IN_COLS;
  localparam int MAXPOOL_OUT_ROWS = (PADDING == 1) ?
        ((IN_ROWS + 2*PADDING - POOL_H) / STRIDE_ROWS) + 1 :
        ((IN_ROWS - POOL_H) / STRIDE_ROWS) + 1;
  localparam int MAXPOOL_OUT_COLS = (PADDING == 1) ?
        ((IN_COLS + 2*PADDING - POOL_W) / STRIDE_COLS) + 1 :
        ((IN_COLS - POOL_W) / STRIDE_COLS) + 1;
  localparam int MAXPOOL_OUT_SIZE = MAXPOOL_OUT_ROWS * MAXPOOL_OUT_COLS;

  // ============================================================
  // State Machine
  // ============================================================
  typedef enum logic [3:0] {
    IDLE,
    SYSTOLIC_PROCESSING,
    FEED_GPNAE,
    GPNAE_PROCESSING,
    COLLECT_GPNAE,
    FILL_MAXPOOL_SRAM,
    MAXPOOL_PROCESSING,
    COLLECT_MAXPOOL,
    DROPOUT_PROCESSING,
    PIPELINE_COMPLETE
  } pipeline_state_t;

  pipeline_state_t current_state, next_state;

  // ============================================================
  // Systolic Array Signals
  // ============================================================
  logic systolic_start;
  logic systolic_mult_complete;
  logic systolic_collection_complete;
  logic systolic_collection_active;
  logic systolic_read_enable;
  logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr;
  logic [DATA_WIDTH-1:0] systolic_read_data;
  logic systolic_read_valid;
  logic north_queue_empty, west_queue_empty;

  // ============================================================
  // FIFO 1: Systolic → GPNAE Interface
  // ============================================================
  logic fifo1_wr_en, fifo1_rd_en;
  logic [DATA_WIDTH-1:0] fifo1_wr_data, fifo1_rd_data;
  logic fifo1_full, fifo1_empty;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo1_count;

  // ============================================================
  // GPNAE Signals & Control
  // ============================================================
  logic gpnae_wr_en;
  logic gpnae_last;
  logic [DATA_WIDTH-1:0] gpnae_signal_data;
  logic gpnae_full, gpnae_empty, gpnae_idle;
  logic [DATA_WIDTH-1:0] gpnae_final_result;
  logic gpnae_done;
  logic [$clog2(SRAM_DEPTH+1)-1:0] gpnae_input_counter;
  logic gpnae_processing_done;

  // ============================================================
  // FIFO 2: GPNAE → Maxpool SRAM Interface
  // ============================================================
  logic fifo2_wr_en, fifo2_rd_en;
  logic [DATA_WIDTH-1:0] fifo2_wr_data, fifo2_rd_data;
  logic fifo2_full, fifo2_empty;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo2_count;

  // ============================================================
  // Maxpool SRAM & Signals
  // ============================================================
  logic [$clog2(MAXPOOL_IN_SIZE)-1:0] maxpool_sram_addr;
  logic [DATA_WIDTH-1:0] maxpool_sram_wdata;
  logic maxpool_sram_wr_en;
  logic [DATA_WIDTH-1:0] maxpool_sram_rdata;
  logic maxpool_sram_rd_en;

  logic [$clog2(MAXPOOL_IN_SIZE)-1:0] maxpool_in_addr;
  logic [DATA_WIDTH-1:0] maxpool_in_data;
  logic maxpool_in_rd_en;
  logic maxpool_start;
  logic maxpool_done;
  logic [DATA_WIDTH-1:0] maxpool_out_data;
  logic maxpool_out_valid;

  logic [$clog2(MAXPOOL_IN_SIZE+1)-1:0] sram_write_counter;
  logic sram_fill_complete;

  // ============================================================
  // FIFO 3: Maxpool → Dropout Interface
  // ============================================================
  logic fifo3_wr_en, fifo3_rd_en;
  logic [DATA_WIDTH-1:0] fifo3_wr_data, fifo3_rd_data;
  logic fifo3_full, fifo3_empty;
  logic [$clog2(FIFO_DEPTH+1)-1:0] fifo3_count;

  // ============================================================
  // Dropout Signals & Output Collection
  // ============================================================
  logic dropout_en;
  logic dropout_training_mode;
  logic [DATA_WIDTH-1:0] dropout_data_in;
  logic [DATA_WIDTH-1:0] dropout_data_out;
  logic dropout_valid_out;

  logic [$clog2(MAXPOOL_OUT_SIZE+1)-1:0] dropout_output_counter;
  logic all_outputs_collected;

  // Output storage
  logic [DATA_WIDTH-1:0] final_output_reg;

  // Control counters
  logic [$clog2(SRAM_DEPTH+1)-1:0] systolic_transfer_counter;
  logic systolic_transfer_complete;

  // State tracking
  logic state_entered;
  pipeline_state_t state_latch;

  // ============================================================
  // Systolic Array Instantiation
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
      .collection_active_o(systolic_collection_active)
  );

  // ============================================================
  // FIFO 1: Systolic → GPNAE
  // ============================================================
  fifo_buffer #(
      .DATA_WIDTH(DATA_WIDTH),
      .DEPTH(FIFO_DEPTH)
  ) fifo1_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .wr_en_i(fifo1_wr_en),
      .wr_data_i(fifo1_wr_data),
      .rd_en_i(fifo1_rd_en),
      .rd_data_o(fifo1_rd_data),
      .full_o(fifo1_full),
      .empty_o(fifo1_empty),
      .count_o(fifo1_count)
  );

  // ============================================================
  // GPNAE Instantiation
  // ============================================================
  gpnae #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_LINES(ADDR_LINES),
      .CONTROL_WIDTH(CONTROL_WIDTH)
  ) gpnae_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .signal_i(gpnae_signal_data),
      .wr_en_i(gpnae_wr_en),
      .last_i(gpnae_last),
      .terms_i(num_terms_i),
      .control_word_i(activation_function_i),
      .full_o(gpnae_full),
      .empty_o(gpnae_empty),
      .idle_o(gpnae_idle),
      .final_result_o(gpnae_final_result),
      .done_o(gpnae_done)
  );

  // ============================================================
  // FIFO 2: GPNAE → Maxpool SRAM
  // ============================================================
  fifo_buffer #(
      .DATA_WIDTH(DATA_WIDTH),
      .DEPTH(FIFO_DEPTH)
  ) fifo2_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .wr_en_i(fifo2_wr_en),
      .wr_data_i(fifo2_wr_data),
      .rd_en_i(fifo2_rd_en),
      .rd_data_o(fifo2_rd_data),
      .full_o(fifo2_full),
      .empty_o(fifo2_empty),
      .count_o(fifo2_count)
  );

  // ============================================================
  // Maxpool Input SRAM
  // ============================================================
  SRAM #(
      .ADDR_WIDTH($clog2(MAXPOOL_IN_SIZE)),
      .DATA_WIDTH(DATA_WIDTH)
  ) maxpool_input_sram (
      .clk  (clk_i),
      .rst_n(rstn_i),
      .addr (maxpool_sram_addr),
      .wdata(maxpool_sram_wdata),
      .wr_en(maxpool_sram_wr_en),
      .rd_en(maxpool_sram_rd_en),
      .rdata(maxpool_sram_rdata)
  );

  // Maxpool SRAM arbitration
  always_comb begin
    if (current_state == MAXPOOL_PROCESSING) begin
      maxpool_sram_addr = maxpool_in_addr;
      maxpool_sram_rd_en = maxpool_in_rd_en;
      maxpool_in_data = maxpool_sram_rdata;
      maxpool_sram_wr_en = 1'b0;
      maxpool_sram_wdata = '0;
    end else begin
      maxpool_sram_addr = sram_write_counter[$clog2(MAXPOOL_IN_SIZE)-1:0];
      maxpool_sram_rd_en = 1'b0;
      maxpool_in_data = '0;
      maxpool_sram_wr_en = (current_state == FILL_MAXPOOL_SRAM) && !fifo2_empty;
      maxpool_sram_wdata = fifo2_rd_data;
    end
  end

  // ============================================================
  // Maxpool Instantiation (using fixed version)
  // ============================================================
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
      .done(maxpool_done),
      .in_addr(maxpool_in_addr),
      .in_data(maxpool_in_data),
      .in_rd_en(maxpool_in_rd_en),
      .out_data(maxpool_out_data),
      .out_valid(maxpool_out_valid)
  );

  // ============================================================
  // FIFO 3: Maxpool → Dropout
  // ============================================================
  fifo_buffer #(
      .DATA_WIDTH(DATA_WIDTH),
      .DEPTH(FIFO_DEPTH)
  ) fifo3_inst (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .wr_en_i(fifo3_wr_en),
      .wr_data_i(fifo3_wr_data),
      .rd_en_i(fifo3_rd_en),
      .rd_data_o(fifo3_rd_data),
      .full_o(fifo3_full),
      .empty_o(fifo3_empty),
      .count_o(fifo3_count)
  );

  // ============================================================
  // Dropout Instantiation (using fixed version)
  // ============================================================
  dropout #(
      .DATA_WIDTH(DATA_WIDTH),
      .DROPOUT_P (DROPOUT_P),
      .LFSR_WIDTH(LFSR_WIDTH)
  ) dropout_inst (
      .clk(clk_i),
      .rst_n(rstn_i),
      .en(dropout_en),
      .training_mode(dropout_training_mode),
      .data_in(dropout_data_in),
      .data_out(dropout_data_out),
      .valid_out(dropout_valid_out)
  );

  // ============================================================
  // State Machine
  // ============================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      current_state <= IDLE;
      state_latch   <= IDLE;
    end else begin
      current_state <= next_state;
      state_latch   <= current_state;
    end
  end

  // Detect state entry
  assign state_entered = (current_state != state_latch);

  // State transition logic
  always_comb begin
    next_state = current_state;

    case (current_state)
      IDLE: begin
        if (start_pipeline_i && !north_queue_empty && !west_queue_empty) begin
          next_state = SYSTOLIC_PROCESSING;
        end
      end

      SYSTOLIC_PROCESSING: begin
        if (systolic_collection_complete) begin
          next_state = FEED_GPNAE;
        end
      end

      FEED_GPNAE: begin
        if (systolic_transfer_complete) begin
          next_state = GPNAE_PROCESSING;
        end
      end

      GPNAE_PROCESSING: begin
        if (gpnae_processing_done) begin
          next_state = COLLECT_GPNAE;
        end
      end

      COLLECT_GPNAE: begin
        if (!fifo2_empty) begin
          next_state = FILL_MAXPOOL_SRAM;
        end
      end

      FILL_MAXPOOL_SRAM: begin
        if (sram_fill_complete) begin
          next_state = MAXPOOL_PROCESSING;
        end
      end

      MAXPOOL_PROCESSING: begin
        if (maxpool_done) begin
          next_state = COLLECT_MAXPOOL;
        end
      end

      COLLECT_MAXPOOL: begin
        if (fifo3_empty && maxpool_done) begin
          next_state = DROPOUT_PROCESSING;
        end
      end

      DROPOUT_PROCESSING: begin
        if (all_outputs_collected) begin
          next_state = PIPELINE_COMPLETE;
        end
      end

      PIPELINE_COMPLETE: begin
        if (!start_pipeline_i) begin
          next_state = IDLE;
        end
      end

      default: next_state = IDLE;
    endcase
  end

  // ============================================================
  // Control Logic
  // ============================================================
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      // Systolic control
      systolic_start <= 1'b0;
      systolic_read_enable <= 1'b0;
      systolic_read_addr <= '0;
      systolic_transfer_counter <= '0;
      systolic_transfer_complete <= 1'b0;

      // FIFO controls
      fifo1_wr_en <= 1'b0;
      fifo1_rd_en <= 1'b0;
      fifo1_wr_data <= '0;
      fifo2_wr_en <= 1'b0;
      fifo2_rd_en <= 1'b0;
      fifo2_wr_data <= '0;
      fifo3_wr_en <= 1'b0;
      fifo3_rd_en <= 1'b0;
      fifo3_wr_data <= '0;

      // GPNAE control
      gpnae_wr_en <= 1'b0;
      gpnae_last <= 1'b0;
      gpnae_signal_data <= '0;
      gpnae_input_counter <= '0;
      gpnae_processing_done <= 1'b0;

      // Maxpool control
      sram_write_counter <= '0;
      sram_fill_complete <= 1'b0;
      maxpool_start <= 1'b0;

      // Dropout control
      dropout_en <= 1'b0;
      dropout_training_mode <= 1'b0;
      dropout_data_in <= '0;
      dropout_output_counter <= '0;
      all_outputs_collected <= 1'b0;

      // Output
      final_output_reg <= '0;

    end else begin
      // Default: pulse signals low
      systolic_start <= 1'b0;
      systolic_read_enable <= 1'b0;
      fifo1_wr_en <= 1'b0;
      fifo1_rd_en <= 1'b0;
      fifo2_wr_en <= 1'b0;
      fifo2_rd_en <= 1'b0;
      fifo3_wr_en <= 1'b0;
      fifo3_rd_en <= 1'b0;
      gpnae_wr_en <= 1'b0;
      gpnae_last <= 1'b0;
      maxpool_start <= 1'b0;
      dropout_en <= 1'b0;

      case (current_state)
        IDLE: begin
          // Reset all counters and flags
          systolic_transfer_counter <= '0;
          systolic_transfer_complete <= 1'b0;
          gpnae_input_counter <= '0;
          gpnae_processing_done <= 1'b0;
          sram_write_counter <= '0;
          sram_fill_complete <= 1'b0;
          dropout_output_counter <= '0;
          all_outputs_collected <= 1'b0;
          dropout_training_mode <= 1'b0;
          final_output_reg <= '0;
        end

        SYSTOLIC_PROCESSING: begin
          if (state_entered) begin
            systolic_start <= 1'b1;
          end
        end

        FEED_GPNAE: begin
          if (systolic_transfer_counter < num_terms_i && !fifo1_full) begin
            systolic_read_enable <= 1'b1;
            systolic_read_addr   <= systolic_transfer_counter[$clog2(SRAM_DEPTH)-1:0];

            if (systolic_read_valid) begin
              fifo1_wr_en <= 1'b1;
              fifo1_wr_data <= systolic_read_data;
              systolic_transfer_counter <= systolic_transfer_counter + 1'b1;

              if (systolic_transfer_counter + 1'b1 >= num_terms_i) begin
                systolic_transfer_complete <= 1'b1;
              end
            end
          end
        end

        GPNAE_PROCESSING: begin
          if (!fifo1_empty && gpnae_idle && !gpnae_full) begin
            fifo1_rd_en <= 1'b1;
            gpnae_wr_en <= 1'b1;
            gpnae_signal_data <= fifo1_rd_data;

            if (gpnae_input_counter >= num_terms_i - 1'b1) begin
              gpnae_last <= 1'b1;
            end

            gpnae_input_counter <= gpnae_input_counter + 1'b1;
          end

          if (gpnae_done) begin
            gpnae_processing_done <= 1'b1;
          end
        end

        COLLECT_GPNAE: begin
          if (gpnae_done && !fifo2_full) begin
            fifo2_wr_en   <= 1'b1;
            fifo2_wr_data <= gpnae_final_result;
          end
        end

        FILL_MAXPOOL_SRAM: begin
          if (!fifo2_empty && sram_write_counter < MAXPOOL_IN_SIZE) begin
            fifo2_rd_en <= 1'b1;
            sram_write_counter <= sram_write_counter + 1'b1;

            if (sram_write_counter + 1'b1 >= MAXPOOL_IN_SIZE) begin
              sram_fill_complete <= 1'b1;
            end
          end
        end

        MAXPOOL_PROCESSING: begin
          if (state_entered) begin
            maxpool_start <= 1'b1;
          end
        end

        COLLECT_MAXPOOL: begin
          if (maxpool_out_valid && !fifo3_full) begin
            fifo3_wr_en   <= 1'b1;
            fifo3_wr_data <= maxpool_out_data;
          end
        end

        DROPOUT_PROCESSING: begin
          if (!fifo3_empty) begin
            fifo3_rd_en <= 1'b1;
            dropout_en <= 1'b1;
            dropout_data_in <= fifo3_rd_data;
          end

          if (dropout_valid_out) begin
            final_output_reg <= dropout_data_out;
            dropout_output_counter <= dropout_output_counter + 1'b1;

            if (dropout_output_counter + 1'b1 >= MAXPOOL_OUT_SIZE) begin
              all_outputs_collected <= 1'b1;
            end
          end
        end

        PIPELINE_COMPLETE: begin
          // Hold final result
        end

        default: ;
      endcase
    end
  end

  // ============================================================
  // Output Assignments
  // ============================================================
  assign final_result_o = final_output_reg;
  assign pipeline_complete_o = (current_state == PIPELINE_COMPLETE);
  assign gpnae_done_o = gpnae_done;

  assign systolic_busy_o = (current_state == SYSTOLIC_PROCESSING) || (current_state == FEED_GPNAE);
  assign gpnae_busy_o = (current_state == GPNAE_PROCESSING) || (current_state == COLLECT_GPNAE);
  assign maxpool_busy_o = (current_state == FILL_MAXPOOL_SRAM) ||
                          (current_state == MAXPOOL_PROCESSING) ||
                          (current_state == COLLECT_MAXPOOL);
  assign dropout_busy_o = (current_state == DROPOUT_PROCESSING);

  assign intermediate_buffer_full_o = fifo1_full;
  assign intermediate_buffer_empty_o = fifo1_empty;
  assign systolic_result_debug_o = systolic_read_data;
  assign systolic_complete_debug_o = systolic_collection_complete;
  assign buffer_count_debug_o = {
    {($clog2(INTERMEDIATE_BUFFER_DEPTH) - $clog2(FIFO_DEPTH + 1)) {1'b0}}, fifo1_count
  };

endmodule

// ============================================================
// FIFO Buffer Module
// ============================================================
module fifo_buffer #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 16,
    parameter ADDR_WIDTH = $clog2(DEPTH)
) (
    input logic clk_i,
    input logic rstn_i,

    input logic                  wr_en_i,
    input logic [DATA_WIDTH-1:0] wr_data_i,

    input  logic                  rd_en_i,
    output logic [DATA_WIDTH-1:0] rd_data_o,

    output logic                full_o,
    output logic                empty_o,
    output logic [ADDR_WIDTH:0] count_o
);

  logic [DATA_WIDTH-1:0] mem[0:DEPTH-1];
  logic [ADDR_WIDTH:0] wr_ptr;
  logic [ADDR_WIDTH:0] rd_ptr;
  logic [ADDR_WIDTH:0] count;

  // Write pointer
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wr_ptr <= '0;
    end else if (wr_en_i && !full_o) begin
      wr_ptr <= wr_ptr + 1'b1;
    end
  end

  // Read pointer
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      rd_ptr <= '0;
    end else if (rd_en_i && !empty_o) begin
      rd_ptr <= rd_ptr + 1'b1;
    end
  end

  // Memory write
  always_ff @(posedge clk_i) begin
    if (wr_en_i && !full_o) begin
      mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data_i;
    end
  end

  // Memory read
  assign rd_data_o = mem[rd_ptr[ADDR_WIDTH-1:0]];

  // Count logic
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      count <= '0;
    end else begin
      case ({
        wr_en_i && !full_o, rd_en_i && !empty_o
      })
        2'b00:   count <= count;
        2'b01:   count <= count - 1'b1;
        2'b10:   count <= count + 1'b1;
        2'b11:   count <= count;
        default: count <= count;
      endcase
    end
  end

  // Status outputs
  assign empty_o = (count == '0);
  assign full_o  = (count == DEPTH[ADDR_WIDTH:0]);
  assign count_o = count;

  // Synthesis-off assertions
  // synthesis translate_off
  // always_ff @(posedge clk_i) begin
  //   if (rstn_i) begin
  //     if (wr_en_i && full_o) begin
  //       $error("Time %0t: FIFO write overflow!", $time);
  //     end
  //     if (rd_en_i && empty_o) begin
  //       $error("Time %0t: FIFO read underflow!", $time);
  //     end
  //   end
  // end
  // synthesis translate_on

endmodule
