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
    input logic clk,
    input logic rst_n,
    input logic start,  // Start signal to begin processing
    output logic done,  // Signal to indicate processing is complete
    // Input SRAM interface
    output logic [$clog2(IN_ROWS*IN_COLS)-1:0] in_addr,
    input logic [DATA_WIDTH-1:0] in_data,
    output logic in_rd_en,
    // Output interface (now goes to Dropout)
    output logic [DATA_WIDTH-1:0] out_data,
    output logic out_valid  // Data valid flag for the output
);
  // Calculate output dimensions based on padding and stride
  localparam OUT_ROWS = (PADDING == 1) ? ((IN_ROWS - 1) / STRIDE_ROWS) + 1 : ((IN_ROWS - SEG_ROWS) / STRIDE_ROWS) + 1;
  localparam OUT_COLS = (PADDING == 1) ? ((IN_COLS - 1) / STRIDE_COLS) + 1 : ((IN_COLS - SEG_COLS) / STRIDE_COLS) + 1;
  localparam OUT_SIZE = OUT_ROWS * OUT_COLS;

  typedef enum logic [2:0] {
    IDLE,
    READ,
    SEG,
    WRITE,
    NEXT,
    FINISH
  } state_t;
  state_t state, next_state;

  // Output position counters
  logic [$clog2(OUT_ROWS):0] out_r;
  logic [$clog2(OUT_COLS):0] out_c;

  // Segmentation window position counters
  logic [$clog2(SEG_ROWS):0] seg_r;
  logic [$clog2(SEG_COLS):0] seg_c;

  // Buffer for pooling window
  logic [DATA_WIDTH-1:0] window[0:SEG_ROWS-1][0:SEG_COLS-1];

  // Max value in window
  logic [DATA_WIDTH-1:0] max_val_reg;
  logic [DATA_WIDTH-1:0] max_val_next;

  // Internal signals for input address calculation
  logic [$clog2(IN_ROWS):0] in_row;
  logic [$clog2(IN_COLS):0] in_col;

  // FSM State and counter updates
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      out_r <= 0;
      out_c <= 0;
      seg_r <= 0;
      seg_c <= 0;
      max_val_reg <= '0;
    end else begin
      state <= next_state;

      if (state == SEG) begin
        if (seg_c < SEG_COLS - 1) seg_c <= seg_c + 1;
        else begin
          seg_c <= 0;
          if (seg_r < SEG_ROWS - 1) seg_r <= seg_r + 1;
          else seg_r <= 0;
        end
      end else if (state == NEXT) begin
        if (out_c < OUT_COLS - 1) out_c <= out_c + 1;
        else begin
          out_c <= 0;
          if (out_r < OUT_ROWS - 1) out_r <= out_r + 1;
          else out_r <= 0;
        end
      end else if (state == IDLE) begin
        out_r <= 0;
        out_c <= 0;
        seg_r <= 0;
        seg_c <= 0;
      end
    end
  end

  // FSM next state and output logic
  always_comb begin
    next_state = state;
    in_rd_en = 0;
    out_valid = 0;
    out_data = max_val_reg;
    done = 0;

    case (state)
      IDLE: begin
        if (start) next_state = READ;
        else next_state = IDLE;
      end
      READ: begin
        in_rd_en   = 1;
        next_state = SEG;
      end
      SEG: begin
        if ((seg_r == SEG_ROWS - 1) && (seg_c == SEG_COLS - 1)) next_state = WRITE;
        else next_state = READ;
      end
      WRITE: begin
        out_valid  = 1;
        next_state = NEXT;
      end
      NEXT: begin
        if ((out_c == OUT_COLS - 1) && (out_r == OUT_ROWS - 1)) next_state = FINISH;
        else next_state = READ;
      end
      FINISH: begin
        done = 1;
        if (!start) begin
          next_state = IDLE;
        end
      end
    endcase
  end

  // Pool window and address management
  always_comb begin
    in_row = out_r * STRIDE_ROWS + seg_r;
    in_col = out_c * STRIDE_COLS + seg_c;
    if (state == READ) begin
      if (in_row < IN_ROWS && in_col < IN_COLS) begin
        in_addr = in_row * IN_COLS + in_col;
      end else begin
        in_addr = 0;
      end
    end else begin
      in_addr = 0;
    end
  end

  // Latch input data into window
  always_ff @(posedge clk) begin
    if (state == SEG) begin
      if (in_row < IN_ROWS && in_col < IN_COLS) window[seg_r][seg_c] <= in_data;
      else window[seg_r][seg_c] <= (PADDING ? '0 : 8'h0);
    end
  end

  // Combinational block to find the max value.
  // This is the synthesizable way to find the max in a matrix.
  always_comb begin
    max_val_next = window[0][0];
    for (int i = 0; i < SEG_ROWS; i++) begin
      for (int j = 0; j < SEG_COLS; j++) begin
        if (window[i][j] > max_val_next) begin
          max_val_next = window[i][j];
        end
      end
    end
  end

  // Register the combinational result
  always_ff @(posedge clk) begin
    if (state == WRITE) begin
      max_val_reg <= max_val_next;
    end
  end

endmodule

