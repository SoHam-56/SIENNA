`timescale 1ns / 100ps

// The int8 residual ADD's words, shared by set_side_t, sienna_top and add_lanes' one port.
package sienna_add_pkg;

  // One add set's ADD words as int8QuantizedAdder takes them (110 bits).
  typedef struct packed {
    logic [7:0]  zp1;    // input zero points
    logic [7:0]  zp2;
    logic [31:0] m1;     // s1 / s_out and s2 / s_out as unsigned mantissas on one shared right shift (ipu.encode_add)
    logic [31:0] m2;
    logic [5:0]  shift;
    logic [7:0]  zpo;    // output zero point
    logic [7:0]  amin;   // the fused activation's clamp
    logic [7:0]  amax;
  } add_words_t;

endpackage
