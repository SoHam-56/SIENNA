`timescale 1ns / 1ps

module Maxpool_2D #(
    parameter     DATA_WIDTH  = 32,
    parameter     IN_ROWS     = 5,
    parameter     IN_COLS     = 5,
    parameter     SEG_ROWS    = 2,
    parameter     SEG_COLS    = 2,
    parameter     STRIDE_ROWS = 2,
    parameter     STRIDE_COLS = 2,
    parameter     PADDING     = 1,
    parameter bit IS_FP32     = 1    // Added flag for FP32 sign-magnitude compare
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input logic [DATA_WIDTH-1:0] data_in,
    input logic valid_in,

    output logic [DATA_WIDTH-1:0] out_data,
    output logic out_valid
);

  localparam int OUT_ROWS = (PADDING == 1) ? ((IN_ROWS + 2*PADDING - SEG_ROWS) / STRIDE_ROWS) + 1 :
                                             ((IN_ROWS - SEG_ROWS) / STRIDE_ROWS) + 1;
  localparam int OUT_COLS = (PADDING == 1) ? ((IN_COLS + 2*PADDING - SEG_COLS) / STRIDE_COLS) + 1 :
                                             ((IN_COLS - SEG_COLS) / STRIDE_COLS) + 1;
  localparam int OUT_SIZE = OUT_ROWS * OUT_COLS;
  localparam int IN_SIZE = IN_ROWS * IN_COLS;

  typedef enum logic [2:0] {
    IDLE,
    COLLECT_INPUT,
    PROCESS,
    OUTPUT_RESULTS,
    FINISH
  } state_t;

  state_t state, next_state;

  logic [DATA_WIDTH-1:0] input_buffer[0:IN_ROWS-1][0:IN_COLS-1];
  logic [$clog2(IN_SIZE+1)-1:0] input_count;
  logic input_collection_done;

  logic [DATA_WIDTH-1:0] output_buffer[0:OUT_SIZE-1];
  logic [$clog2(OUT_SIZE+1)-1:0] output_count;
  logic processing_done;

  logic [$clog2(OUT_ROWS+1)-1:0] out_r;
  logic [$clog2(OUT_COLS+1)-1:0] out_c;

  // ---------------------------------------------------------
  // Safe FP32 / Signed Int Comparator
  // ---------------------------------------------------------
  function automatic logic is_greater(input logic [DATA_WIDTH-1:0] a,
                                      input logic [DATA_WIDTH-1:0] b);
    if (IS_FP32 && DATA_WIDTH == 32) begin
      logic a_sign = a[31];
      logic b_sign = b[31];
      if ((a[30:0] == 0) && (b[30:0] == 0)) return 1'b0;
      if (a_sign != b_sign) return !a_sign;
      if (!a_sign) return a[30:0] > b[30:0];
      return a[30:0] < b[30:0];
    end else begin
      return $signed(a) > $signed(b);
    end
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else state <= next_state;
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE:           if (start) next_state = COLLECT_INPUT;
      COLLECT_INPUT:  if (input_collection_done) next_state = PROCESS;
      PROCESS:        if (processing_done) next_state = OUTPUT_RESULTS;
      OUTPUT_RESULTS: if (output_count >= OUT_SIZE) next_state = FINISH;
      FINISH:         if (!start) next_state = IDLE;
      default:        next_state = IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_count <= '0;
      input_collection_done <= 1'b0;
      for (int i = 0; i < IN_ROWS; i++) begin
        for (int j = 0; j < IN_COLS; j++) input_buffer[i][j] <= '0;
      end
    end else begin
      case (state)
        IDLE: begin
          input_count <= '0;
          input_collection_done <= 1'b0;
        end
        COLLECT_INPUT: begin
          if (valid_in && input_count < IN_SIZE) begin
            input_buffer[input_count/IN_COLS][input_count%IN_COLS] <= data_in;
            input_count <= input_count + 1'b1;
            if (input_count + 1'b1 >= IN_SIZE) input_collection_done <= 1'b1;
          end
        end
        default: ;
      endcase
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_r <= '0;
      out_c <= '0;
      processing_done <= 1'b0;
      for (int i = 0; i < OUT_SIZE; i++) output_buffer[i] <= '0;
    end else begin
      case (state)
        IDLE: begin
          out_r <= '0;
          out_c <= '0;
          processing_done <= 1'b0;
        end
        PROCESS: begin
          logic [DATA_WIDTH-1:0] max_val;
          logic signed [31:0] in_row;
          logic signed [31:0] in_col;
          logic in_bounds;
          logic [DATA_WIDTH-1:0] current_val;

          // Initialize with correct minimum floor
          if (IS_FP32 && DATA_WIDTH == 32) max_val = 32'hFF800000;  // -Infinity
          else max_val = {1'b1, {(DATA_WIDTH - 1) {1'b0}}};  // Max Neg 2's Complement

          for (int sr = 0; sr < SEG_ROWS; sr++) begin
            for (int sc = 0; sc < SEG_COLS; sc++) begin
              if (PADDING == 1) begin
                in_row = signed'(out_r * STRIDE_ROWS + sr) - 1;
                in_col = signed'(out_c * STRIDE_COLS + sc) - 1;
              end else begin
                in_row = signed'(out_r * STRIDE_ROWS + sr);
                in_col = signed'(out_c * STRIDE_COLS + sc);
              end

              in_bounds = (in_row >= 0) && (in_row < IN_ROWS) &&
                          (in_col >= 0) && (in_col < IN_COLS);

              if (in_bounds) begin
                current_val = input_buffer[in_row][in_col];
                if (is_greater(current_val, max_val)) max_val = current_val;
              end
            end
          end

          output_buffer[out_r*OUT_COLS+out_c] <= max_val;

          if (out_c < OUT_COLS - 1) begin
            out_c <= out_c + 1'b1;
          end else begin
            out_c <= '0;
            if (out_r < OUT_ROWS - 1) out_r <= out_r + 1'b1;
            else processing_done <= 1'b1;
          end
        end
        default: ;
      endcase
    end
  end

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
          end else out_valid <= 1'b0;
        end
        default: out_valid <= 1'b0;
      endcase
    end
  end

  assign done = (state == FINISH);

endmodule
