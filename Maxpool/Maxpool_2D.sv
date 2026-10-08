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
    parameter bit IS_FP32     = 1,   // float sign-magnitude compare given EXP_W > 0; EXP_W = 0 (int8) always compares signed integers
    parameter int EXP_W       = 8,
    parameter int MAN_W       = 23,
    parameter int AHEAD       = 2,   // streaming: windows whose input credits are granted before their results are out
    parameter int OUT_MAX     = 64,  // the most output credits the downstream consumer may grant
    parameter int OUT_CRW     = 1,   // out.credit width
    parameter int IN_CRW      = $clog2(IN_ROWS * IN_COLS + 1)  // in.credit width: a whole input granted in one cycle
) (
    input  logic clk,
    input  logic rst_n,

    credit_link_if.consumer in,   // IN_ROWS*IN_COLS elements per input, granted at once only when output credits for its results are reserved
    credit_link_if.producer out   // one result per put
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

  localparam bit FLOAT = IS_FP32 && (EXP_W > 0) && (DATA_WIDTH == 1 + EXP_W + MAN_W);  // EXP_W = 0 is int8
  localparam logic [DATA_WIDTH-1:0] FLOOR = FLOAT ? DATA_WIDTH'({DATA_WIDTH{1'b1}} << MAN_W)  // -infinity: sign and exponent all ones
                                                  : {1'b1, {(DATA_WIDTH - 1) {1'b0}}};  // most negative integer, -128 in int8

  logic [DATA_WIDTH-1:0] input_buffer[0:IN_ROWS-1][0:IN_COLS-1];
  logic [$clog2(IN_SIZE+1)-1:0] input_count;
  logic input_collection_done;

  logic [DATA_WIDTH-1:0] output_buffer[0:OUT_SIZE-1];
  logic [$clog2(OUT_SIZE+1)-1:0] output_count;
  logic processing_done;

  logic [$clog2(OUT_ROWS+1)-1:0] out_r;
  logic [$clog2(OUT_COLS+1)-1:0] out_c;

  // ---------------------------------------------------------
  // Float (sign-magnitude) or signed-integer compare
  // ---------------------------------------------------------
  function automatic logic is_greater(input logic [DATA_WIDTH-1:0] a,
                                      input logic [DATA_WIDTH-1:0] b);
    if (FLOAT) begin
      if ((a[DATA_WIDTH-2:0] == 0) && (b[DATA_WIDTH-2:0] == 0)) return 1'b0;
      if (a[DATA_WIDTH-1] != b[DATA_WIDTH-1]) return !a[DATA_WIDTH-1];
      if (!a[DATA_WIDTH-1]) return a[DATA_WIDTH-2:0] > b[DATA_WIDTH-2:0];
      return a[DATA_WIDTH-2:0] < b[DATA_WIDTH-2:0];
    end else begin
      return $signed(a) > $signed(b);
    end
  endfunction

  // Links: elements arrive on in, results leave on out; an input is granted only with output credits reserved for all its results.
  localparam bit SINGLE_SEG = (OUT_SIZE == 1);
  logic [DATA_WIDTH-1:0] data_in;
  logic valid_in;
  logic [DATA_WIDTH-1:0] out_data;
  logic out_valid;
  assign data_in  = in.data;
  assign valid_in = in.put;
  assign out.put  = out_valid;
  assign out.data = out_data;

  if (OUT_MAX < OUT_SIZE || AHEAD < 1) begin : G_BAD_OUT_MAX
    $fatal(1, "Maxpool_2D: OUT_MAX %0d is below the %0d results of one input (or AHEAD %0d < 1), so no input could be granted", OUT_MAX,
           OUT_SIZE, AHEAD);
  end
`ifndef SYNTHESIS
  // Interface widths are not elaboration constants in Verilator, so the link widths are checked at time 0.
  initial
    if ($bits(in.data) != DATA_WIDTH || $bits(in.credit) != IN_CRW || $bits(out.data) != DATA_WIDTH || $bits(out.credit) != OUT_CRW)
      $fatal(1, "Maxpool_2D: links need data %0d bits, in.credit %0d and out.credit %0d, found %0d/%0d and %0d/%0d", DATA_WIDTH, IN_CRW,
             OUT_CRW, $bits(in.data), $bits(in.credit), $bits(out.data), $bits(out.credit));
`endif

  logic live;  // out of reset for a cycle: no input is granted before it
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) live <= 1'b0;
    else live <= 1'b1;

  localparam int OCW = $clog2(OUT_MAX + 1);
  logic [OCW-1:0] out_cnt;  // output credits held, reserved or not
  credit_counter #(.MAX(OUT_MAX), .CRW(OUT_CRW)) out_cc (.clk_i(clk), .rstn_i(rst_n), .put_i(out.put), .credit_i(out.credit),
                                                        .has_credit_o(), .count_o(out_cnt));
  logic grant;  // the IN_SIZE input credits of one more input go out this cycle
  assign in.credit = grant ? IN_CRW'(IN_SIZE) : '0;

`ifndef SYNTHESIS
  // Every element arrives on a credit this module granted: credits granted minus elements received.
  int in_open;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) in_open <= 0;
    else in_open <= in_open + (grant ? IN_SIZE : 0) - int'(in.put);
  a_mp_in_credit: assert property (@(posedge clk) disable iff (!rst_n) in.put |-> in_open > 0)
    else $error("Maxpool_2D: a_mp_in_credit: an element arrived with no input credit granted");
`endif

  // SIENNA instantiates this with SEG == IN and PADDING == 0, so the whole input is one segment
  // and OUT_SIZE is 1: the window dispatcher already does the tiling and padding. That case is a
  // running max, which accepts an element every cycle, whereas the batch FSM below collects,
  // then processes, then emits, and cannot overlap consecutive windows.
  generate
    if (SINGLE_SEG) begin : gen_stream
      localparam logic [DATA_WIDTH-1:0] NEG_FLOOR = FLOOR;

      logic [DATA_WIDTH-1:0]        run_max;
      logic [$clog2(IN_SIZE+1)-1:0] in_cnt;
      logic [$clog2(AHEAD+1)-1:0]   rsv;  // windows granted whose result is not out yet, each holding one output credit

      // A window is granted when it can reserve an output credit no other granted window holds; the result never waits.
      assign grant = live && (int'(rsv) < AHEAD) && (int'(out_cnt) > int'(rsv));

      // Folding the incoming element in combinationally lets the last one be emitted on the
      // cycle it arrives rather than one later.
      wire [DATA_WIDTH-1:0] nxt_max = is_greater(data_in, run_max) ? data_in : run_max;

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          run_max   <= NEG_FLOOR;
          in_cnt    <= '0;
          rsv       <= '0;
          out_data  <= '0;
          out_valid <= 1'b0;
        end else begin
          out_valid <= 1'b0;
          rsv <= rsv + ($clog2(AHEAD+1))'(grant) - ($clog2(AHEAD+1))'(out_valid);
          if (valid_in) begin
            if ((in_cnt + 1'b1) == IN_SIZE[$clog2(IN_SIZE+1)-1:0]) begin
              out_data  <= nxt_max;
              out_valid <= 1'b1;
              in_cnt    <= '0;
              run_max   <= NEG_FLOOR;
            end else begin
              run_max <= nxt_max;
              in_cnt  <= in_cnt + 1'b1;
            end
          end
        end
      end
    end else begin : gen_batch


  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else state <= next_state;
  end

  // One input at a time: granted from IDLE once the output credits of all its results are held.
  assign grant = live && (state == IDLE) && (int'(out_cnt) >= OUT_SIZE);

  always_comb begin
    next_state = state;
    case (state)
      IDLE:           if (grant) next_state = COLLECT_INPUT;
      COLLECT_INPUT:  if (input_collection_done) next_state = PROCESS;
      PROCESS:        if (processing_done) next_state = OUTPUT_RESULTS;
      OUTPUT_RESULTS: if (output_count >= OUT_SIZE) next_state = FINISH;
      FINISH:         next_state = IDLE;
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
          max_val = FLOOR;  // -infinity for floats, the most negative integer otherwise

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


    end
  endgenerate

endmodule
