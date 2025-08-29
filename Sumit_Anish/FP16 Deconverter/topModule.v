`timescale 1ns/1ps
module fp16_to_int (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [15:0] fp16_in,
    output logic        done_out,
    output logic [15:0] integer_out
);

  localparam [15:0] MAX_POS = 16'h7FFF;
  localparam [15:0] MAX_NEG = 16'h8000;
  localparam [4:0]  BIAS    = 5'd15;
  localparam [4:0]  MAX_EXP = 5'd30;

  logic        sign_r;
  logic [4:0]  exponent_r;
  logic [9:0]  mantissa_r;
  logic        is_zero_r;
  logic        is_inf_r;
  logic        is_nan_r;
  logic        is_denormal_r;
  logic signed [6:0]  exp_unbiased_r;
  logic [10:0]        full_mantissa_11_r;
  logic [15:0]        shifted_value_r;
  logic               overflow_r;
  logic               underflow_r;
  logic [15:0]        abs_result_r;
  logic               result_zero_r;
  logic [5:0]   s_amt_abs;
  logic [15:0]   mantissa_aligned;
  logic signed [6:0] s_amt_signed;

  typedef enum logic [1:0] { IDLE, CALCULATE, OUTPUT } state_t;
  state_t state, next_state;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state            <= IDLE;
      sign_r           <= 1'b0;
      exponent_r       <= 5'd0;
      mantissa_r       <= 10'd0;
      is_zero_r        <= 1'b1;
      is_inf_r         <= 1'b0;
      is_nan_r         <= 1'b0;
      is_denormal_r    <= 1'b0;
      exp_unbiased_r   <= '0;
      full_mantissa_11_r <= '0;
      shifted_value_r  <= '0;
      overflow_r       <= 1'b0;
      underflow_r      <= 1'b0;
      abs_result_r     <= '0;
      result_zero_r    <= 1'b1;
      integer_out      <= '0;
      done_out         <= 1'b0;
    end else begin
      state <= next_state;
      if (next_state != OUTPUT)
        done_out <= 1'b0;
      if (state == IDLE && next_state == CALCULATE) begin
        sign_r     <= fp16_in[15];
        exponent_r <= fp16_in[14:10];
        mantissa_r <= fp16_in[9:0];
        is_zero_r     <= (fp16_in[14:10] == 5'b00000) && (fp16_in[9:0] == 10'b0);
        is_inf_r      <= (fp16_in[14:10] == 5'b11111) && (fp16_in[9:0] == 10'b0);
        is_nan_r      <= (fp16_in[14:10] == 5'b11111) && (fp16_in[9:0] != 10'b0);
        is_denormal_r <= (fp16_in[14:10] == 5'b00000) && (fp16_in[9:0] != 10'b0);
        if ( (fp16_in[14:10] == 5'b00000) ) begin
          exp_unbiased_r <= $signed(7'sd1) - $signed({2'b00, BIAS});
        end else begin
          exp_unbiased_r <= $signed({2'b00, fp16_in[14:10]}) - $signed({2'b00, BIAS});
        end
        if ((fp16_in[14:10] == 5'b00000))
          full_mantissa_11_r <= {1'b0, fp16_in[9:0]};
        else
          full_mantissa_11_r <= {1'b1, fp16_in[9:0]};
      end
      if (state == IDLE && next_state == CALCULATE) begin
        if ($signed(exp_unbiased_r) >= 7'sd10)
          s_amt_signed = $signed(exp_unbiased_r) - 7'sd10;
        else
          s_amt_signed = 7'sd10 - $signed(exp_unbiased_r);
        s_amt_abs = (s_amt_signed < 0) ? ( -s_amt_signed[5:0] ) : s_amt_signed[5:0];
        mantissa_aligned = {5'b0, full_mantissa_11_r};
        if ($signed(exp_unbiased_r) >= 7'sd10) begin
          if (s_amt_abs >= 16) begin
            shifted_value_r <= {16{1'b0}};
          end else begin
            shifted_value_r <= mantissa_aligned << s_amt_abs;
          end
        end else begin
          if (s_amt_abs >= 16) begin
            shifted_value_r <= 16'd0;
          end else begin
            shifted_value_r <= mantissa_aligned >> s_amt_abs;
          end
        end
        if ($signed(exp_unbiased_r) >= 7'sd15 || $signed(exp_unbiased_r) - 7'sd10 > 4)
          overflow_r <= 1'b1;
        else
          overflow_r <= 1'b0;
        if ($signed(exp_unbiased_r) < 7'sd0) begin
          underflow_r <= ( (mantissa_aligned >> ( (7'sd10 - $signed(exp_unbiased_r)) > 6'd31 ? 6'd31 : (7'sd10 - $signed(exp_unbiased_r)) )) == 16'd0 );
        end else begin
          underflow_r <= 1'b0;
        end
        result_zero_r <= is_zero_r || underflow_r;
        if (overflow_r) begin
          abs_result_r <= MAX_POS;
        end else begin
          abs_result_r <= shifted_value_r;
        end
      end
      if (state == CALCULATE && next_state == OUTPUT) begin
        done_out <= 1'b1;
        if (is_nan_r || result_zero_r) begin
          integer_out <= 16'd0;
        end else if (is_inf_r || overflow_r) begin
          integer_out <= sign_r ? MAX_NEG : MAX_POS;
        end else begin
          if (sign_r) begin
            integer_out <= (~abs_result_r) + 16'd1;
          end else begin 
            integer_out <= abs_result_r;
          end
        end
      end
    end
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE: begin
        if(valid_in) next_state = CALCULATE;
        else next_state = IDLE;
      end
      CALCULATE: next_state = OUTPUT;
      OUTPUT:    next_state = IDLE;
      default:   next_state = IDLE;
    endcase
  end

endmodule
