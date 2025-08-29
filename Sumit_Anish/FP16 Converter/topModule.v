`timescale 1ns/1ps

module int_to_fp16 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         valid_in,
    input  logic [19:0]  integer_in,
    output logic         done_out,
 	output logic [15:0] fp16_out
);

    typedef enum logic [3:0] { IDLE, CALC_ABS, FIND_LZ_STAGE1, FIND_LZ_STAGE2, NORMALIZE, CALC_EXP_ROUND, ASSEMBLE_FP16, DONE } state_e;
    state_e current_state, next_state;

    logic          sign_bit_reg;
    logic [19:0]   abs_val_reg;
    logic [19:0]   abs_val_reg_d1;
    integer        leading_zeros_stage1_reg;
    integer        leading_zeros_reg;
    logic [19:0]   normalized_val_reg;
    logic [9:0]    mantissa_raw_reg;
    logic          guard_bit_reg;
    logic          sticky_bit_reg;
    logic [4:0]    biased_exponent_reg;
    logic          round_up_reg;
    logic [15:0]   fp16_out_reg;

    logic          sign_bit_comb;
    logic [19:0]   abs_val_comb;
    integer        leading_zeros_stage1_comb;
    logic          abs_val_is_zero_comb;
    integer        leading_zeros_stage2_comb;
    logic [19:0]   normalized_val_comb;
    logic [9:0]    mantissa_raw_comb;
    logic          guard_bit_comb;
    logic          sticky_bit_comb;
    logic [4:0]    biased_exponent_comb;
    logic          round_up_comb;
    logic [4:0]    final_exponent_comb;
    logic [9:0]    final_mantissa_comb;

    localparam FP16_EXP_BIAS     = 5'd15;
    localparam FP16_MAX_EXP_NORM = 5'd30;
    localparam FP16_EXP_INF_NAN  = 5'd31;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            fp16_out_reg  <= 16'b0;
            done_out      <= 1'b0;
        end else begin
            current_state <= next_state;
            if (current_state == ASSEMBLE_FP16) begin
                fp16_out_reg <= {sign_bit_reg, final_exponent_comb, final_mantissa_comb};
            end
            done_out <= (current_state == DONE);
        end
    end

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE:           if (valid_in) next_state = CALC_ABS;
            CALC_ABS:       next_state = FIND_LZ_STAGE1;
            FIND_LZ_STAGE1: next_state = FIND_LZ_STAGE2;
            FIND_LZ_STAGE2: next_state = NORMALIZE;
            NORMALIZE:      next_state = CALC_EXP_ROUND;
            CALC_EXP_ROUND: next_state = ASSEMBLE_FP16;
            ASSEMBLE_FP16:  next_state = DONE;
            DONE: begin
                if (!valid_in) next_state = IDLE;
                else           next_state = CALC_ABS;
            end
            default: next_state = IDLE;
        endcase
    end

    always_comb begin
        sign_bit_comb = integer_in[19];
        if (integer_in[19]) begin
            abs_val_comb = ~integer_in + 1;
        end else begin
            abs_val_comb = integer_in;
        end
    end

    always_comb begin
        leading_zeros_stage1_comb = 0;

        abs_val_is_zero_comb = (abs_val_reg == 20'b0);

        if (abs_val_is_zero_comb) begin
            leading_zeros_stage1_comb = 20;
        end else begin
            if (abs_val_reg[19]) leading_zeros_stage1_comb = 0;
            else if (abs_val_reg[18]) leading_zeros_stage1_comb = 1;
            else if (abs_val_reg[17]) leading_zeros_stage1_comb = 2;
            else if (abs_val_reg[16]) leading_zeros_stage1_comb = 3;
            else if (abs_val_reg[15]) leading_zeros_stage1_comb = 4;
            else if (abs_val_reg[14]) leading_zeros_stage1_comb = 5;
            else if (abs_val_reg[13]) leading_zeros_stage1_comb = 6;
            else if (abs_val_reg[12]) leading_zeros_stage1_comb = 7;
            else if (abs_val_reg[11]) leading_zeros_stage1_comb = 8;
            else if (abs_val_reg[10]) leading_zeros_stage1_comb = 9;
            else leading_zeros_stage1_comb = 10;
        end
    end

    always_comb begin
        leading_zeros_stage2_comb = leading_zeros_stage1_reg;

        if (leading_zeros_stage1_reg == 10) begin
            if (abs_val_reg_d1[9]) leading_zeros_stage2_comb = 10;
            else if (abs_val_reg_d1[8]) leading_zeros_stage2_comb = 11;
            else if (abs_val_reg_d1[7]) leading_zeros_stage2_comb = 12;
            else if (abs_val_reg_d1[6]) leading_zeros_stage2_comb = 13;
            else if (abs_val_reg_d1[5]) leading_zeros_stage2_comb = 14;
            else if (abs_val_reg_d1[4]) leading_zeros_stage2_comb = 15;
            else if (abs_val_reg_d1[3]) leading_zeros_stage2_comb = 16;
            else if (abs_val_reg_d1[2]) leading_zeros_stage2_comb = 17;
            else if (abs_val_reg_d1[1]) leading_zeros_stage2_comb = 18;
            else if (abs_val_reg_d1[0]) leading_zeros_stage2_comb = 19;
            else leading_zeros_stage2_comb = 20;
        end
    end

    assign normalized_val_comb = abs_val_reg_d1 << leading_zeros_reg;
    assign mantissa_raw_comb   = normalized_val_comb[18:9];
    assign guard_bit_comb      = normalized_val_comb[8];
    assign sticky_bit_comb     = (normalized_val_comb[7:0] != 8'b0);

    always_comb begin
        if (abs_val_reg_d1 == 20'b0) begin
            biased_exponent_comb = 5'b0;
        end else begin
            biased_exponent_comb = (19 - leading_zeros_reg) + FP16_EXP_BIAS;
        end

        round_up_comb = 1'b0;
        if (guard_bit_reg) begin
            if (sticky_bit_reg) begin
                round_up_comb = 1'b1;
            end else begin
                if (mantissa_raw_reg[0]) begin
                    round_up_comb = 1'b1;
                end
            end
        end
    end

    always_comb begin
        final_mantissa_comb = mantissa_raw_reg;
        final_exponent_comb = biased_exponent_reg;

        if (round_up_reg) begin
            final_mantissa_comb = mantissa_raw_reg + 1;
            if (final_mantissa_comb == 10'b0) begin
                final_exponent_comb = biased_exponent_reg + 1;
            end
        end

        if (abs_val_reg_d1 == 20'b0) begin
            final_exponent_comb = 5'b0;
            final_mantissa_comb = 10'b0;
        end

        if (final_exponent_comb > FP16_MAX_EXP_NORM) begin
            final_exponent_comb = FP16_EXP_INF_NAN;
            final_mantissa_comb = 10'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sign_bit_reg          <= 1'b0;
            abs_val_reg           <= 20'b0;
            abs_val_reg_d1        <= 20'b0;
            leading_zeros_stage1_reg <= 0;
            leading_zeros_reg     <= 0;
            normalized_val_reg    <= 20'b0;
            mantissa_raw_reg      <= 10'b0;
            guard_bit_reg         <= 1'b0;
            sticky_bit_reg        <= 1'b0;
            biased_exponent_reg   <= 5'b0;
            round_up_reg          <= 1'b0;
        end else begin
            case (current_state)
                IDLE: begin
                end
                CALC_ABS: begin
                    sign_bit_reg <= sign_bit_comb;
                    abs_val_reg  <= abs_val_comb;
                end
                FIND_LZ_STAGE1: begin
                    leading_zeros_stage1_reg <= leading_zeros_stage1_comb;
                    abs_val_reg_d1           <= abs_val_reg;
                end
                FIND_LZ_STAGE2: begin
                    leading_zeros_reg <= leading_zeros_stage2_comb;
                end
                NORMALIZE: begin
                    normalized_val_reg <= normalized_val_comb;
                    mantissa_raw_reg   <= mantissa_raw_comb;
                    guard_bit_reg      <= guard_bit_comb;
                    sticky_bit_reg     <= sticky_bit_comb;
                end
                CALC_EXP_ROUND: begin
                    biased_exponent_reg <= biased_exponent_comb;
                    round_up_reg        <= round_up_comb;
                end
                ASSEMBLE_FP16: begin
                end
                DONE: begin
                end
            endcase
        end
    end

    assign fp16_out = fp16_out_reg;

endmodule