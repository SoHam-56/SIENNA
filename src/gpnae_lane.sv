`timescale 1ns / 100ps

// One GPNAE lane at the build's widths: gpnae_poly, which still takes its format as EXP_W and MAN_W, with both from sienna_fmt_pkg (check-fmt allows this file until GPNAE's single gpnae replaces it).
module gpnae_lane #(
    parameter int ADDR_LINES    = 5,
    parameter int CONTROL_WIDTH = 3,   // 001 SELU, 010 sigmoid, 011 tanh, 100 ReLU, 101 linear
    parameter int OUT_MAX       = 64,  // the most output credits the downstream consumer may advertise
    parameter int OUT_CRW       = 1    // out.credit width
) (
    input logic clk_i,
    input logic rstn_i,

    credit_link_if.consumer in,   // {last, signal}, last on a set's final element
    credit_link_if.producer out,  // one result per put

    input logic [CONTROL_WIDTH-1:0] control_word_i,

    input logic [15:0] gp_mx_i,     // int8 only (D-2): input rescale multiplier, below 2^15
    input logic [ 4:0] gp_shx_i,    // int8: input rescale shift
    input logic [ 7:0] gp_zin_i,    // int8: input zero point
    input logic [23:0] gp_mout_i,   // int8: SELU output scale's mantissa
    input logic [ 5:0] gp_shout_i,  // int8: SELU output scale's right shift
    input logic [ 7:0] gp_zout_i    // int8: SELU output zero point
);
  if (1 + sienna_fmt_pkg::EXPONENT_WIDTH + sienna_fmt_pkg::MANTISSA_WIDTH != sienna_fmt_pkg::DATA_WIDTH) begin : G_BAD_WIDTH  // the lane's own DATA_WIDTH
    $fatal(1, "gpnae_lane: 1 + EXPONENT_WIDTH + MANTISSA_WIDTH is not DATA_WIDTH %0d", sienna_fmt_pkg::DATA_WIDTH);
  end

  gpnae_poly #(
      .EXP_W        (sienna_fmt_pkg::EXPONENT_WIDTH),
      .MAN_W        (sienna_fmt_pkg::MANTISSA_WIDTH),
      .DATA_WIDTH   (sienna_fmt_pkg::DATA_WIDTH),
      .ADDR_LINES   (ADDR_LINES),
      .CONTROL_WIDTH(CONTROL_WIDTH),
      .OUT_MAX      (OUT_MAX),
      .OUT_CRW      (OUT_CRW)
  ) lane (
      .clk_i         (clk_i),
      .rstn_i        (rstn_i),
      .in            (in),
      .out           (out),
      .control_word_i(control_word_i),
      .gp_mx_i       (gp_mx_i),
      .gp_shx_i      (gp_shx_i),
      .gp_zin_i      (gp_zin_i),
      .gp_mout_i     (gp_mout_i),
      .gp_shout_i    (gp_shout_i),
      .gp_zout_i     (gp_zout_i)
  );

endmodule
