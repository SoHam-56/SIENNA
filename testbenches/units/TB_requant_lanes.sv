`timescale 1ns / 100ps

// requant_lanes on rq_lanes.mem: per-lane zp/min/max; sets 1, 2, 5 packed (lane k reads channel k % N); REQUANT_LATENCY latency; outputs to rq_lanes_out.mem for the true-function gate.
module TB_requant_lanes;
  localparam int N = 16, NUM_LANES = 32, PER_LANE = N * N / NUM_LANES, REQ_LAT = sienna_fmt_pkg::REQUANT_LATENCY;
  logic clk_i = 0, rstn_i = 0, clear_i = 1, valid_i = 0, valid_o;
  logic [NUM_LANES-1:0][31:0] acc_i = '0;
  logic [N-1:0][23:0] mult_i = '0;
  logic [N-1:0][5:0] shift_i = '0;
  logic packed_i = 0;
  logic [NUM_LANES-1:0][7:0] zp_i = '0, min_i = '0, max_i = '0;
  logic [NUM_LANES-1:0][7:0] result_o;
  always #5 clk_i = ~clk_i;

  if (!(sienna_fmt_pkg::HAS_REQUANT_EPILOGUE)) begin : G_BAD_FORMAT  // the bench's vectors are int8 words
    $fatal(1, "TB_requant_lanes: build with FMT=int8 (+define+FMT_INT8), not %s", sienna_fmt_pkg::FORMAT_NAME);
  end
  // The raw words in the package's bundles: a channel's scale, a lane's entry (its lane words zero).
  sienna_fmt_pkg::requant_scale_t [N-1:0] scale_i;
  sienna_fmt_pkg::lane_entry_t [NUM_LANES-1:0] entry_i;
  for (genvar c = 0; c < N; c++) begin : G_SCALE
    assign scale_i[c] = '{mantissa: mult_i[c], shift: shift_i[c]};
  end
  for (genvar k = 0; k < NUM_LANES; k++) begin : G_ENTRY
    assign entry_i[k] = '{zero_point: zp_i[k], act_min: min_i[k], act_max: max_i[k], default: '0};
  end
  requant_lanes #(.NUM_LANES(NUM_LANES), .N(N)) dut (.clk_i(clk_i), .rstn_i(rstn_i), .clear_i(clear_i), .valid_i(valid_i), .acc_i(acc_i),
      .scale_i(scale_i), .packed_i(packed_i), .entry_i(entry_i), .valid_o(valid_o), .result_o(result_o));

  logic [31:0] v[$];
  logic [NUM_LANES-1:0][7:0] want_q[$];  // filled before reset is released, then only popped by the checker
  longint t_in[$];
  longint cyc = 0;
  int errs = 0, lat_errs = 0, n_out = 0, diffs = 0;  // diffs: outputs off the bit-exact model, reported only
  integer fo;

  // One block owns the checker's state: the cycle count, the input times and the pops.
  always @(posedge clk_i) begin
    cyc <= cyc + 1;
    if (valid_i) t_in.push_back(cyc);
    if (valid_o) begin
      automatic logic [NUM_LANES-1:0][7:0] w = want_q.pop_front();
      automatic longint t = t_in.pop_front();
      n_out++;
      if (cyc - t != REQ_LAT) lat_errs++;
      for (int k = 0; k < NUM_LANES; k++) begin
        $fwrite(fo, "%02x ", result_o[k]);
        if (result_o[k] !== w[k]) begin
          diffs++;
          if (diffs <= 20)
            $display("[EXACT] beat %0d lane %0d: got %0d, bit-exact model %0d (reported)", n_out - 1, k, $signed(result_o[k]), $signed(w[k]));
        end
      end
      $fwrite(fo, "\n");  // one line per output beat, lane 0 first
    end
  end

  initial begin
    integer fh;
    logic [31:0] w32;
    int p, sets;
    fh = $fopen("rq_lanes.mem", "r");
    if (fh == 0) begin
      $display("[FATAL] cannot open rq_lanes.mem");
      $finish;
    end
    while ($fscanf(fh, "%h", w32) == 1) v.push_back(w32);
    $fclose(fh);
    fo = $fopen("rq_lanes_out.mem", "w");
    if (fo == 0) $fatal(1, "TB_requant_lanes: cannot write rq_lanes_out.mem");
    sets = int'(v[0]);
    p = 1;
    for (int s = 0; s < sets; s++) begin
      p += 1 + 3 * NUM_LANES + 2 * N;
      for (int b = 0; b < PER_LANE; b++) begin
        logic [NUM_LANES-1:0][7:0] row;
        for (int k = 0; k < NUM_LANES; k++) row[k] = v[p+NUM_LANES+k][7:0];
        want_q.push_back(row);
        p += 2 * NUM_LANES;
      end
    end
    repeat (4) @(posedge clk_i);
    rstn_i = 1;
    repeat (2) @(posedge clk_i);
    p = 1;
    for (int s = 0; s < sets; s++) begin
      @(negedge clk_i);
      clear_i = 1;
      packed_i = v[p][0];
      for (int k = 0; k < NUM_LANES; k++) begin
        zp_i[k]  = v[p+1+k][7:0];
        min_i[k] = v[p+1+NUM_LANES+k][7:0];
        max_i[k] = v[p+1+2*NUM_LANES+k][7:0];
      end
      for (int c = 0; c < N; c++) begin
        if (v[p+1+3*NUM_LANES+c][31:24] != 0 || v[p+1+3*NUM_LANES+N+c][31:6] != 0) $fatal(1, "TB_requant_lanes: channel %0d's words are too wide", c);
        mult_i[c]  = v[p+1+3*NUM_LANES+c][23:0];
        shift_i[c] = v[p+1+3*NUM_LANES+N+c][5:0];
      end
      p += 1 + 3 * NUM_LANES + 2 * N;
      @(negedge clk_i);
      clear_i = 0;
      for (int b = 0; b < PER_LANE; b++) begin
        if (s % 2 == 1 && b == 3) begin  // a bubble: beats count only with valid_i
          valid_i = 0;
          @(negedge clk_i);
        end
        valid_i = 1;
        for (int k = 0; k < NUM_LANES; k++) acc_i[k] = v[p+k];
        p += 2 * NUM_LANES;
        @(negedge clk_i);
      end
      valid_i = 0;
      repeat (REQ_LAT + 2) @(negedge clk_i);
    end
    repeat (4) @(negedge clk_i);
    if (n_out != sets * PER_LANE || want_q.size() != 0) begin
      errs++;
      $display("[FAIL] %0d beats out, expected %0d", n_out, sets * PER_LANE);
    end
    if (lat_errs != 0) $display("[FAIL] %0d beats did not come %0d cycles after their input", lat_errs, REQ_LAT);
    $fclose(fo);
    $display("TB_requant_lanes: %0d beats, %0d errors, %0d latency errors; %0d outputs differ from the bit-exact model (reported)", n_out, errs, lat_errs, diffs);
    $display("values: not judged here; regression.py --action rq-vectors --check gates rq_lanes_out.mem on the true function");
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (errs == 0 && lat_errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
