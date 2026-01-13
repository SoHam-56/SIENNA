`timescale 1ns / 1ps

module Maxpool_2D #(
    parameter DATA_WIDTH  = 8,
    parameter IN_ROWS     = 5,
    parameter IN_COLS     = 5,
    parameter SEG_ROWS    = 2,
    parameter SEG_COLS    = 2,
    parameter STRIDE_ROWS = 2,
    parameter STRIDE_COLS = 2,
    parameter PADDING     = 1   // 0: no padding, 1: zero padding
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    // Streaming input interface
    input logic [DATA_WIDTH-1:0] data_in,
    input logic valid_in,

    // Streaming output interface
    output logic [DATA_WIDTH-1:0] out_data,
    output logic out_valid
);

  // Calculate output dimensions based on padding and stride
  localparam int OUT_ROWS = (PADDING == 1) ?
                              ((IN_ROWS + 2*PADDING - SEG_ROWS) / STRIDE_ROWS) + 1 :
                              ((IN_ROWS - SEG_ROWS) / STRIDE_ROWS) + 1;
  localparam int OUT_COLS = (PADDING == 1) ?
                              ((IN_COLS + 2*PADDING - SEG_COLS) / STRIDE_COLS) + 1 :
                              ((IN_COLS - SEG_COLS) / STRIDE_COLS) + 1;
  localparam int OUT_SIZE = OUT_ROWS * OUT_COLS;
  localparam int IN_SIZE = IN_ROWS * IN_COLS;

  // FSM states
  typedef enum logic [2:0] {
    IDLE,
    COLLECT_INPUT,
    PROCESS,
    OUTPUT_RESULTS,
    FINISH
  } state_t;

  state_t state, next_state;

  // Input buffer to store entire input feature map
  logic [DATA_WIDTH-1:0] input_buffer[0:IN_ROWS-1][0:IN_COLS-1];
  logic [$clog2(IN_SIZE+1)-1:0] input_count;
  logic input_collection_done;

  // Output buffer to store pooling results
  logic [DATA_WIDTH-1:0] output_buffer[0:OUT_SIZE-1];
  logic [$clog2(OUT_SIZE+1)-1:0] output_count;
  logic processing_done;

  // Processing counters
  logic [$clog2(OUT_ROWS+1)-1:0] out_r;
  logic [$clog2(OUT_COLS+1)-1:0] out_c;

  // FSM State register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      state <= next_state;
    end
  end

  // FSM next state logic
  always_comb begin
    next_state = state;

    case (state)
      IDLE: begin
        if (start) next_state = COLLECT_INPUT;
      end

      COLLECT_INPUT: begin
        if (input_collection_done) next_state = PROCESS;
      end

      PROCESS: begin
        if (processing_done) next_state = OUTPUT_RESULTS;
      end

      OUTPUT_RESULTS: begin
        if (output_count >= OUT_SIZE) next_state = FINISH;
      end

      FINISH: begin
        if (!start) next_state = IDLE;
      end

      default: next_state = IDLE;
    endcase
  end

  // Input collection logic
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_count <= '0;
      input_collection_done <= 1'b0;
      for (int i = 0; i < IN_ROWS; i++) begin
        for (int j = 0; j < IN_COLS; j++) begin
          input_buffer[i][j] <= '0;
        end
      end
    end else begin
      case (state)
        IDLE: begin
          input_count <= '0;
          input_collection_done <= 1'b0;
        end

        COLLECT_INPUT: begin
          if (valid_in && input_count < IN_SIZE) begin
            // Store input in row-major order
            input_buffer[input_count/IN_COLS][input_count%IN_COLS] <= data_in;
            input_count <= input_count + 1'b1;

            if (input_count + 1'b1 >= IN_SIZE) begin
              input_collection_done <= 1'b1;
            end
          end
        end

        default: ;
      endcase
    end
  end

  // Processing logic
  // --- THIS IS THE SECTION WITH THE CRITICAL FIXES ---
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_r <= '0;
      out_c <= '0;
      processing_done <= 1'b0;
      for (int i = 0; i < OUT_SIZE; i++) begin
        output_buffer[i] <= '0;
      end
    end else begin
      case (state)
        IDLE: begin
          out_r <= '0;
          out_c <= '0;
          processing_done <= 1'b0;
        end

        PROCESS: begin
          // 1. Logic variable declarations inside the block
          logic [DATA_WIDTH-1:0] max_val;
          logic signed [31:0] in_row;  // Use 32-bit signed for safe index math
          logic signed [31:0] in_col;
          logic in_bounds;
          logic [DATA_WIDTH-1:0] current_val;

          // 2. Initialize max to 0 (Unsigned Min) always
          max_val = '0;

          // 3. Find max in the pooling window
          for (int sr = 0; sr < SEG_ROWS; sr++) begin
            for (int sc = 0; sc < SEG_COLS; sc++) begin

              // Calculate coordinates (using signed types to handle negatives)
              if (PADDING == 1) begin
                in_row = signed'(out_r * STRIDE_ROWS + sr) - 1;
                in_col = signed'(out_c * STRIDE_COLS + sc) - 1;
              end else begin
                in_row = signed'(out_r * STRIDE_ROWS + sr);
                in_col = signed'(out_c * STRIDE_COLS + sc);
              end

              // Check bounds
              in_bounds = (in_row >= 0) && (in_row < IN_ROWS) &&
                                        (in_col >= 0) && (in_col < IN_COLS);

              // 4. Update Max logic (Unsigned only)
              if (in_bounds) begin
                current_val = input_buffer[in_row][in_col];
                if (current_val > max_val) begin
                  max_val = current_val;
                end
              end
              // If !in_bounds (padding area), value is implicitly 0.
              // Since max_val starts at 0, no update is needed.
            end
          end

          // Store result
          output_buffer[out_r*OUT_COLS+out_c] <= max_val;

          // Move to next output position
          if (out_c < OUT_COLS - 1) begin
            out_c <= out_c + 1'b1;
          end else begin
            out_c <= '0;
            if (out_r < OUT_ROWS - 1) begin
              out_r <= out_r + 1'b1;
            end else begin
              processing_done <= 1'b1;
            end
          end
        end

        default: ;
      endcase
    end
  end

  // Output logic
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_data <= '0;
      out_valid <= 1'b0;
      output_count <= '0;
    end else begin
      case (state)
        IDLE: begin
          out_valid <= 1'b0;
          output_count <= '0;
        end

        OUTPUT_RESULTS: begin
          if (output_count < OUT_SIZE) begin
            out_data <= output_buffer[output_count];
            out_valid <= 1'b1;
            output_count <= output_count + 1'b1;
          end else begin
            out_valid <= 1'b0;
          end
        end

        default: begin
          out_valid <= 1'b0;
        end
      endcase
    end
  end

  // Done signal
  assign done = (state == FINISH);

endmodule
