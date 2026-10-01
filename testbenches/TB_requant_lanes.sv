`timescale 1ns / 100ps

// requant_lanes against ipu.requant (rq_lanes.mem): per-channel words, lane k's channel (k*PER_LANE + b) % N, req_lat() latency.
module TB_requant_lanes;
  localparam int N = 16, NUM_LANES = 32, PER_LANE = N * N / NUM_LANES, REQ_LAT = sienna_fmt_pkg::req_lat();
  logic clk_i = 0, rstn_i = 0, clear_i = 1, valid_i = 0, valid_o;
  logic [NUM_LANES-1:0][31:0] acc_i = '0;
  logic [N-1:0][31:0] mult_i = '0;
  logic [N-1:0][7:0] shift_i = '0;
  logic [7:0] zp_i = '0, min_i = '0, max_i = '0;
  logic [NUM_LANES-1:0][7:0] result_o;
  always #5 clk_i = ~clk_i;

  requant_lanes #(.NUM_LANES(NUM_LANES), .N(N), .ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) dut (.*);

  logic [31:0] v[$];
  logic [NUM_LANES-1:0][7:0] want_q[$];  // filled before reset is released, then only popped by the checker
  longint t_in[$];
  longint cyc = 0;
  int errs = 0, lat_errs = 0, n_out = 0;

  // One block owns the checker's state: the cycle count, the input times and the pops.
  always @(posedge clk_i) begin
    cyc <= cyc + 1;
    if (valid_i) t_in.push_back(cyc);
    if (valid_o) begin
      automatic logic [NUM_LANES-1:0][7:0] w = want_q.pop_front();
      automatic longint t = t_in.pop_front();
      n_out++;
      if (cyc - t != REQ_LAT) lat_errs++;
      for (int k = 0; k < NUM_LANES; k++)
        if (result_o[k] !== w[k]) begin
          errs++;
          if (errs <= 20)
            $display("[FAIL] beat %0d lane %0d: got %0d, want %0d", n_out - 1, k, $signed(result_o[k]), $signed(w[k]));
        end
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
    sets = int'(v[0]);
    p = 1;
    for (int s = 0; s < sets; s++) begin
      p += 3 + 2 * N;
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
      zp_i  = v[p][7:0];
      min_i = v[p+1][7:0];
      max_i = v[p+2][7:0];
      for (int c = 0; c < N; c++) begin
        mult_i[c]  = v[p+3+c];
        shift_i[c] = v[p+3+N+c][7:0];
      end
      p += 3 + 2 * N;
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
    $display("TB_requant_lanes: %0d beats, %0d errors, %0d latency errors", n_out, errs, lat_errs);
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (errs == 0 && lat_errs == 0) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
