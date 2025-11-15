`timescale 1ns / 1ps

module Maxpool_2D #(
    parameter DATA_WIDTH  = 8,
    parameter IN_ROWS     = 5,
    parameter IN_COLS     = 5,
    parameter SEG_ROWS    = 2,
    parameter SEG_COLS    = 2,
    parameter STRIDE_ROWS = 2,
    parameter STRIDE_COLS = 2,
    parameter PADDING     = 1  // 0: no padding, 1: zero padding
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    // Input SRAM interface
    output logic [$clog2(IN_ROWS*IN_COLS)-1:0] in_addr,
    input  logic [DATA_WIDTH-1:0] in_data,
    output logic in_rd_en,

    // Output interface (now goes to Dropout)
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

    // FSM states
    typedef enum logic [2:0] {
        IDLE,
        READ,
        SEG,
        COMPUTE,
        WRITE,
        NEXT,
        FINISH
    } state_t;

    state_t state, next_state;

    // Output position counters
    logic [$clog2(OUT_ROWS+1)-1:0] out_r;
    logic [$clog2(OUT_COLS+1)-1:0] out_c;

    // Segmentation window position counters
    logic [$clog2(SEG_ROWS+1)-1:0] seg_r;
    logic [$clog2(SEG_COLS+1)-1:0] seg_c;

    // Buffer for pooling window
    logic [DATA_WIDTH-1:0] window [0:SEG_ROWS-1][0:SEG_COLS-1];

    // Max value in window
    logic [DATA_WIDTH-1:0] max_val_reg;
    logic [DATA_WIDTH-1:0] max_val_next;

    // Internal signals for input address calculation
    logic signed [$clog2(IN_ROWS+1):0] in_row;
    logic signed [$clog2(IN_COLS+1):0] in_col;
    logic in_bounds;

    // FSM State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // Counter updates
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_r <= '0;
            out_c <= '0;
            seg_r <= '0;
            seg_c <= '0;
        end else begin
            case (state)
                IDLE: begin
                    out_r <= '0;
                    out_c <= '0;
                    seg_r <= '0;
                    seg_c <= '0;
                end

                SEG: begin
                    if (seg_c < SEG_COLS-1) begin
                        seg_c <= seg_c + 1'b1;
                    end else begin
                        seg_c <= '0;
                        if (seg_r < SEG_ROWS-1) begin
                            seg_r <= seg_r + 1'b1;
                        end else begin
                            seg_r <= '0;
                        end
                    end
                end

                NEXT: begin
                    seg_r <= '0;
                    seg_c <= '0;
                    if (out_c < OUT_COLS-1) begin
                        out_c <= out_c + 1'b1;
                    end else begin
                        out_c <= '0;
                        if (out_r < OUT_ROWS-1) begin
                            out_r <= out_r + 1'b1;
                        end
                    end
                end

                default: ;
            endcase
        end
    end

    // Max value register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_val_reg <= '0;
        end else if (state == COMPUTE) begin
            max_val_reg <= max_val_next;
        end
    end

    // FSM next state logic
    always_comb begin
        next_state = state;

        case (state)
            IDLE: begin
                if (start)
                    next_state = READ;
            end

            READ: begin
                next_state = SEG;
            end

            SEG: begin
                if ((seg_r == SEG_ROWS-1) && (seg_c == SEG_COLS-1))
                    next_state = COMPUTE;
                else
                    next_state = READ;
            end

            COMPUTE: begin
                next_state = WRITE;
            end

            WRITE: begin
                next_state = NEXT;
            end

            NEXT: begin
                if ((out_c == OUT_COLS-1) && (out_r == OUT_ROWS-1))
                    next_state = FINISH;
                else
                    next_state = READ;
            end

            FINISH: begin
                if (!start)
                    next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // Output control signals
    always_comb begin
        in_rd_en  = (state == READ);
        out_valid = (state == WRITE);
        out_data  = max_val_reg;
        done      = (state == FINISH);
    end

    // Input address calculation with padding support
    always_comb begin
        // Calculate position in input feature map (accounting for padding)
        if (PADDING == 1) begin
            in_row = signed'(out_r * STRIDE_ROWS + seg_r) - 1;
            in_col = signed'(out_c * STRIDE_COLS + seg_c) - 1;
        end else begin
            in_row = signed'(out_r * STRIDE_ROWS + seg_r);
            in_col = signed'(out_c * STRIDE_COLS + seg_c);
        end

        // Check if position is within bounds
        in_bounds = (in_row >= 0) && (in_row < IN_ROWS) &&
                    (in_col >= 0) && (in_col < IN_COLS);

        // Generate address
        if (state == READ && in_bounds) begin
            in_addr = in_row[$clog2(IN_ROWS*IN_COLS)-1:0] * IN_COLS +
                      in_col[$clog2(IN_ROWS*IN_COLS)-1:0];
        end else begin
            in_addr = '0;
        end
    end

    // Latch input data into window buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < SEG_ROWS; i++) begin
                for (int j = 0; j < SEG_COLS; j++) begin
                    window[i][j] <= '0;
                end
            end
        end else if (state == SEG) begin
            if (in_bounds) begin
                window[seg_r][seg_c] <= in_data;
            end else begin
                // Padding: fill with zeros or minimum value
                window[seg_r][seg_c] <= (PADDING == 1) ? '0 : {DATA_WIDTH{1'b1}};
            end
        end
    end

    // Find maximum value in window (combinational)
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

endmodule
