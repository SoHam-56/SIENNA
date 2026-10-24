// int8 testbench helpers, included inside a TB module after its D-2 signals (no include guard: one copy per module).

  // 32-bit word reader: int32 biases and requantize words (8 digits), or narrower words zero-extended.
  task automatic read_word_file(input string fn, output logic [31:0] q[$]);
    integer fh, rc;
    logic [31:0] w;
    q.delete();
    fh = $fopen(fn, "r");
    if (!fh) begin
      $display("[ERROR] Cannot open: %s", fn);
      $finish;
    end
    while (!$feof(fh)) begin
      rc = $fscanf(fh, "%h", w);
      if (rc == 1) q.push_back(w);
    end
    $fclose(fh);
  endtask

  // One set's 8 + 2N words onto the D-2 signals: zp, min, max, gp_mx, gp_shx, gp_mout, gp_shout, gp_zout, N multipliers, N shifts.
  task automatic unpack_requant(input logic [31:0] q[$]);
    if (q[5][31:24] != 0 || q[6][31:6] != 0) $fatal(1, "int8_tb_util: SELU words %h %h exceed a 24-bit mantissa and a 6-bit shift", q[5], q[6]);
    for (int c = 0; c < N; c++)
      if (q[8+c][31:24] != 0 || q[8+N+c][31:6] != 0)
        $fatal(1, "int8_tb_util: channel %0d's words %h %h exceed a 24-bit mantissa and a 6-bit shift", c, q[8+c], q[8+N+c]);
    req_zp_i   = q[0][7:0];
    req_min_i  = q[1][7:0];
    req_max_i  = q[2][7:0];
    gp_mx_i    = q[3][15:0];
    gp_shx_i   = q[4][4:0];
    gp_mout_i  = q[5][23:0];  // the scale's 24-bit mantissa and right shift in 32-bit words
    gp_shout_i = q[6][5:0];
    gp_zout_i  = q[7][7:0];
    for (int c = 0; c < N; c++) begin
      req_mult_i[c]  = q[8+c][23:0];
      req_shift_i[c] = q[8+N+c][5:0];
    end
  endtask

  // Requantizing builds (int8): set k's requantize and GPNAE parameters from requant_<k>.mem; nothing in others.
  task automatic apply_requant(input int k);
    logic [31:0] q[$];
    if (!sienna_fmt_pkg::HAS_REQUANT_EPILOGUE) return;
    read_word_file($sformatf("requant_%0d.mem", k), q);
    if (q.size() != 8 + 2 * N) begin
      $display("[FATAL] requant_%0d.mem holds %0d words, expected %0d", k, q.size(), 8 + 2 * N);
      $finish;
    end
    unpack_requant(q);
  endtask
