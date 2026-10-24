// Packs a bench's raw int8 words into sienna_fmt_pkg's bundles (set_side_t's entry, scale and add; sienna_top's layer ports), in the package's field order.
// A build without them (floats) gets zeros in their 1-bit dummies. Included inside a TB module after sienna_set_side.svh, so N and PACK_ENTRIES are defined.

  localparam int TB_ENTRY_W = 8 + 8 + 8 + 16 + 5 + 24 + 6 + 8;  // lane_entry_t in int8
  localparam int TB_SCALE_W = 24 + 6;  // requant_scale_t in int8
  localparam int TB_ADD_W = 8 + 8 + 32 + 32 + 6 + 8 + 8 + 8;  // add_params_t in int8
  localparam int ENTRY_BITS = $bits(sienna_fmt_pkg::lane_entry_t);
  localparam int SCALE_BITS = $bits(sienna_fmt_pkg::requant_scale_t);
  localparam int ADD_BITS = $bits(sienna_fmt_pkg::add_params_t);
  if (sienna_fmt_pkg::HAS_REQUANT_EPILOGUE && (ENTRY_BITS != TB_ENTRY_W || SCALE_BITS != TB_SCALE_W)) begin : G_BAD_TB_BUNDLE
    $fatal(1, "tb_side_pack: the package's lane_entry_t is %0d bits and requant_scale_t %0d, the bench packs %0d and %0d", ENTRY_BITS, SCALE_BITS,
           TB_ENTRY_W, TB_SCALE_W);
  end
  if (sienna_fmt_pkg::HAS_RESIDUAL_ADD && ADD_BITS != TB_ADD_W) begin : G_BAD_TB_ADD
    $fatal(1, "tb_side_pack: the package's add_params_t is %0d bits, the bench packs %0d", ADD_BITS, TB_ADD_W);
  end

  // The ADD's words as the bench fills them; add_params_t's int8 layout and order.
  typedef struct packed {
    logic [7:0]  zp1;
    logic [7:0]  zp2;
    logic [31:0] m1;
    logic [31:0] m2;
    logic [5:0]  shift;
    logic [7:0]  zpo;
    logic [7:0]  amin;
    logic [7:0]  amax;
  } tb_add_words_t;

  // One entry's raw words in lane_entry_t's order: requantize zero point and clamp, lane rescale, SELU requantize.
  function automatic logic [TB_ENTRY_W-1:0] tb_entry(input logic [7:0] zp, input logic [7:0] amin, input logic [7:0] amax, input logic [15:0] mx,
                                                     input logic [4:0] shx, input logic [23:0] mout, input logic [5:0] shout, input logic [7:0] zout);
    return {zp, amin, amax, mx, shx, mout, shout, zout};
  endfunction

  function automatic logic [PACK_ENTRIES*ENTRY_BITS-1:0] tb_side_entries(input logic [PACK_ENTRIES-1:0][TB_ENTRY_W-1:0] raw);
    return sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? (PACK_ENTRIES * ENTRY_BITS)'(raw) : '0;
  endfunction

  function automatic logic [ENTRY_BITS-1:0] tb_lane_entry(input logic [TB_ENTRY_W-1:0] raw);
    return sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? ENTRY_BITS'(raw) : '0;
  endfunction

  function automatic logic [N*SCALE_BITS-1:0] tb_side_scales(input logic [N-1:0][23:0] mult, input logic [N-1:0][5:0] shift);
    logic [N-1:0][TB_SCALE_W-1:0] raw;
    for (int c = 0; c < N; c++) raw[c] = {mult[c], shift[c]};
    return sienna_fmt_pkg::HAS_REQUANT_EPILOGUE ? (N * SCALE_BITS)'(raw) : '0;
  endfunction

  function automatic logic [ADD_BITS-1:0] tb_side_add(input tb_add_words_t w);
    return sienna_fmt_pkg::HAS_RESIDUAL_ADD ? ADD_BITS'(w) : '0;
  endfunction
