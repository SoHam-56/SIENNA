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
  sienna_fmt_pkg::lane_entry_t [PACK_ENTRIES-1:0] entry;     // per entry: requantize zero point and clamp, lane rescale and SELU words (1-bit dummies in floats)
  sienna_fmt_pkg::requant_scale_t [N-1:0]      scale;          // per output channel: the requantize scale (1-bit dummies in floats)
  logic                                       res_hold;       // residual: the last depth pass; its beats go to hold_mem, no bank, no output
  logic                                       res_add;        // residual: the identity pass; its beats meet hold_mem in add_lanes
  sienna_fmt_pkg::add_params_t                add;            // residual: the ADD's words (a 1-bit dummy in floats)
} set_side_t;
