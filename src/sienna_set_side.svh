// One set's sideband, the data of the host link (L0) put with its start; the including scope defines N, CONTROL_WIDTH, LFSR_WIDTH, WC_TILES, PACK_ENTRIES.
typedef struct packed {
  logic                                       wc_last;        // the last set of its cache region's fill: the region's credit returns once it is broadcast
  logic [$clog2(WC_TILES)-1:0]                weight_tile;    // B is this cache tile; only A is written
  logic                                       weight_cached;
  logic                                       accumulate;     // a partial sum: added to the running sum, no output
  logic                                       bias_valid;     // add bias_i[c] to column c
  logic                                       train;          // dropout mode
  logic [LFSR_WIDTH-1:0]                      seed;           // dropout seed
  logic [2:0]                                 pack_shift;     // a packed set of N >> pack_shift columns per job; 0 unpacked
  logic [N/2-1:0][$clog2(PACK_ENTRIES)-1:0]   pack_map;       // entry of each column block
  logic [PACK_ENTRIES-1:0][CONTROL_WIDTH-1:0] act;            // activation per entry; entry 0 is the set's own, 1.. a packed set's table
  logic [PACK_ENTRIES-1:0][7:0]               zp;             // int8 per entry: requantize zero point and clamp
  logic [PACK_ENTRIES-1:0][7:0]               amin;
  logic [PACK_ENTRIES-1:0][7:0]               amax;
  logic [PACK_ENTRIES-1:0][15:0]              mx;             // int8 per entry: GPNAE input rescale and output requantize
  logic [PACK_ENTRIES-1:0][4:0]               shx;
  logic [PACK_ENTRIES-1:0][31:0]              mout;
  logic [PACK_ENTRIES-1:0][7:0]               shout;
  logic [PACK_ENTRIES-1:0][7:0]               zout;
  logic [N-1:0][31:0]                         mult;           // int8 per output channel: requantize multiplier (Q0.31) and signed shift
  logic [N-1:0][7:0]                          shift;
  logic                                       res_hold;       // int8 residual: the last depth pass; its beats go to hold_mem, no bank, no output
  logic                                       res_add;        // int8 residual: the identity pass; its beats meet hold_mem in add_lanes
  logic [7:0]                                 add_zp1;        // the ADD's words (tfliteAdd): input zero points, multipliers and shifts, output zero point and clamp
  logic [7:0]                                 add_zp2;
  logic [31:0]                                add_m1;
  logic [7:0]                                 add_s1;
  logic [31:0]                                add_m2;
  logic [7:0]                                 add_s2;
  logic [31:0]                                add_mo;
  logic [7:0]                                 add_so;
  logic [7:0]                                 add_zpo;
  logic [7:0]                                 add_min;
  logic [7:0]                                 add_max;
} set_side_t;

// One add set's ADD words as tfliteAdd takes them; every reader names a field, so no bit position is written twice.
typedef struct packed {
  logic [7:0]  zp1;
  logic [7:0]  zp2;
  logic [31:0] m1;
  logic [7:0]  s1;
  logic [31:0] m2;
  logic [7:0]  s2;
  logic [31:0] mo;
  logic [7:0]  so;
  logic [7:0]  zpo;
  logic [7:0]  amin;
  logic [7:0]  amax;
} add_words_t;
