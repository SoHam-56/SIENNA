`timescale 1ns / 100ps

// add_lanes against ipu.tflite_add (add_lanes.mem): per set its ADD words, per beat each lane's q1 and q2; the tag rides qadd_lat() cycles.
module TB_add_lanes;
  localparam int NUM_LANES = 32, SETS = 6, PER = 8, TAG_W = 16, LAT = sienna_fmt_pkg::qadd_lat();
  logic clk_i = 0, rstn_i = 0, valid_i = 0, valid_o;
  logic [NUM_LANES-1:0][7:0] q1_i = '0, q2_i = '0, result_o;
  logic [TAG_W-1:0] tag_i = '0, tag_o;
  logic [7:0] zp1_i = '0, zp2_i = '0, s1_i = '0, s2_i = '0, so_i = '0, zpo_i = '0, min_i = '0, max_i = '0;
  logic [31:0] m1_i = '0, m2_i = '0, mo_i = '0;
  always #5 clk_i = ~clk_i;

  add_lanes #(.NUM_LANES(NUM_LANES), .TAG_W(TAG_W), .ROUNDING(sienna_fmt_pkg::REQ_ROUNDING)) dut (.*);

  logic [NUM_LANES-1:0][7:0] want_q[$];
  logic [TAG_W-1:0] tag_q[$];
  longint t_in[$];
  longint cyc = 0;
  int errs = 0, lat_errs = 0, tag_errs = 0, n_out = 0;

  // One block owns the checker's state: the cycle count, the input times and the pops.
  always @(posedge clk_i) begin
    cyc <= cyc + 1;
    if (valid_i) t_in.push_back(cyc);
    if (valid_o && want_q.size() == 0) begin
      errs++;
      $display("[FAIL] an output beat with no input beat left to match it");
    end else if (valid_o) begin
      automatic logic [NUM_LANES-1:0][7:0] w = want_q.pop_front();
      automatic logic [TAG_W-1:0] tg = tag_q.pop_front();
      automatic longint t = t_in.pop_front();
      n_out++;
      if (cyc - t != LAT) lat_errs++;
      if (tag_o !== tg) tag_errs++;
      for (int k = 0; k < NUM_LANES; k++)
        if (result_o[k] !== w[k]) begin
          errs++;
          if (errs <= 20) $display("[FAIL] beat %0d lane %0d: got %0d, want %0d", n_out - 1, k, $signed(result_o[k]), $signed(w[k]));
        end
    end
  end

  // Valid low: random codes, tag and ADD words, so a stage that takes them off a beat shows up as wrong results.
  task automatic scramble();
    for (int k = 0; k < NUM_LANES; k++) {q1_i[k], q2_i[k]} = 16'($urandom);
    tag_i = TAG_W'($urandom);
    {zp1_i, zp2_i, s1_i, s2_i, so_i, zpo_i, min_i, max_i} = {$urandom, $urandom};
    {m1_i, m2_i, mo_i} = {$urandom, $urandom, $urandom};
  endtask

  initial begin
    int fd;
    logic [31:0] c[11];
    logic [15:0] tg;
    logic [23:0] e;
    fd = $fopen("add_lanes.mem", "r");
    if (fd == 0) $fatal(1, "TB_add_lanes: no add_lanes.mem (regression.py --action add-vectors writes it)");
    repeat (3) @(posedge clk_i);
    #1 rstn_i = 1;
    for (int s = 0; s < SETS; s++) begin
      if ($fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10]) != 11)
        $fatal(1, "TB_add_lanes: add_lanes.mem: set %0d has no line of 11 ADD words", s);
      for (int b = 0; b < PER; b++) begin
        automatic logic [NUM_LANES-1:0][7:0] w;
        if ($fscanf(fd, "%h", tg) != 1) $fatal(1, "TB_add_lanes: add_lanes.mem: set %0d beat %0d has no tag", s, b);
        @(posedge clk_i);
        #1;
        {zp1_i, zp2_i, m1_i, s1_i, m2_i, s2_i, mo_i, so_i, zpo_i, min_i, max_i} = {c[0][7:0], c[1][7:0], c[2], c[3][7:0], c[4], c[5][7:0], c[6], c[7][7:0],
                                                                                   c[8][7:0], c[9][7:0], c[10][7:0]};
        for (int k = 0; k < NUM_LANES; k++) begin
          if ($fscanf(fd, "%h", e) != 1) $fatal(1, "TB_add_lanes: add_lanes.mem: set %0d beat %0d lane %0d is missing", s, b, k);
          {q1_i[k], q2_i[k], w[k]} = e;
        end
        tag_i = tg;
        want_q.push_back(w);
        tag_q.push_back(tg);
        valid_i = 1;
        if (b == PER / 2) begin  // a bubble inside a set: valid low for a cycle
          @(posedge clk_i);
          #1 valid_i = 0;
          scramble();
        end
      end
      if (s % 2 == 1 || s == SETS - 1) begin  // a gap after odd sets; an even set's last beat is followed at once by the next set's first, new words and all
        @(posedge clk_i);
        #1 valid_i = 0;
        scramble();
      end
    end
    if ($fscanf(fd, "%h", e) == 1) $fatal(1, "TB_add_lanes: add_lanes.mem holds more than %0d sets of %0d beats", SETS, PER);  // at the end of the file this simulator's $fscanf gives 0, not -1
    repeat (LAT + 4) @(posedge clk_i);
    if (n_out != SETS * PER || want_q.size() != 0) errs++;
    $display("add_lanes: %0d beats, %0d errors, %0d latency errors, %0d tag errors", n_out, errs, lat_errs, tag_errs);
    $display("RESULT: %s", (errs == 0 && lat_errs == 0 && tag_errs == 0) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
