// The lanes' coefficient-image check: every lane's memory, read hierarchically, against the image the host meant it to hold.
// Include in a mode's scope after defining TB_CC_LANES (lanes to check) and TB_CC_MEM(i) (lane i's gpnae_coef_mem instance).
localparam int CC_LANES = `TB_CC_LANES;
localparam int CC_W = gpnae_pkg::ACT_W;
localparam int CC_FD = gpnae_pkg::PMAX + 8;  // a descriptor's fields in Mem.contents() order: bounds, np, deg, base, reflect, exp_base, exp_deg, center, frac
localparam int CC_FT = gpnae_pkg::TAIL_MAXC + 2;  // a tail set's: coef, deg, newton
localparam int CC_N = gpnae_pkg::COEF_WORDS + gpnae_pkg::NUM_TIERS * (gpnae_pkg::NUM_DESC * CC_FD + CC_FT);
localparam logic [31:0] CC_END = 32'hC0EF_1A5E;  // gpnae_model.IMAGE_END, the line after the last field
logic [CC_W-1:0] cc_words[CC_LANES][gpnae_pkg::COEF_WORDS];  // each lane's registered memory state
gpnae_pkg::desc_t cc_desc[CC_LANES][gpnae_pkg::NUM_TIERS][gpnae_pkg::NUM_DESC];
gpnae_pkg::tail_t cc_tail[CC_LANES][gpnae_pkg::NUM_TIERS];
for (genvar i = 0; i < CC_LANES; i++) begin : G_CC
  for (genvar a = 0; a < gpnae_pkg::COEF_WORDS; a++) begin : G_W
    assign cc_words[i][a] = `TB_CC_MEM(i).words[a];
  end
  for (genvar t = 0; t < gpnae_pkg::NUM_TIERS; t++) begin : G_T
    for (genvar d = 0; d < gpnae_pkg::NUM_DESC; d++) begin : G_D
      assign cc_desc[i][t][d] = `TB_CC_MEM(i).desc[t][d];
    end
    if (sienna_fmt_pkg::HAS_FLOAT_TAIL) begin : G_TAIL
      assign cc_tail[i][t] = `TB_CC_MEM(i).G_TAIL.tail[t];
    end else begin : G_NO_TAIL
      assign cc_tail[i][t] = '0;
    end
  end
end

logic [31:0] cc_img[0:CC_N];  // the image the lanes must hold now, Mem.contents() order, then CC_END
int cc_checks = 0, cc_bad = 0;  // checks run, and checks that found a difference

// The image file f (gpnae_model.Mem.write_contents) becomes the one the lanes must hold; a short or missing file is fatal.
task automatic cc_image(input string f);
  for (int k = 0; k <= CC_N; k++) cc_img[k] = '0;
  $readmemh(f, cc_img);
  if (cc_img[CC_N] !== CC_END) $fatal(1, "coefficient image %s: not %0d fields and the end word", f, CC_N);
endtask

// One field of one lane: a difference is counted and named with its load address.
task automatic cc_field(input string where, input int l, input string what, input int addr, input logic [31:0] got, input logic [31:0] want,
                        inout int bad);
  if (got !== want) begin
    if (bad < 20) $display("[FAIL] COEF IMAGE %s: lane %0d %s (load address %0d) holds %h, the image has %h", where, l, what, addr, got, want);
    bad++;
  end
endtask

// Every lane against cc_img: each word, each descriptor field and, with the float tail, each tail field.
task automatic cc_check(input string where);
  int bad, k, ab;
  gpnae_pkg::desc_t q;
  bad = 0;
  for (int l = 0; l < CC_LANES; l++) begin
    for (int a = 0; a < gpnae_pkg::COEF_WORDS; a++) cc_field(where, l, $sformatf("word %0d", a), a, 32'(cc_words[l][a]), cc_img[a], bad);
    k = gpnae_pkg::COEF_WORDS;
    for (int t = 0; t < gpnae_pkg::NUM_TIERS; t++)
      for (int d = 0; d < gpnae_pkg::NUM_DESC; d++) begin
        q  = cc_desc[l][t][d];
        ab = gpnae_pkg::A_DESC + gpnae_pkg::DESC_STRIDE * (t * gpnae_pkg::NUM_DESC + d);
        for (int j = 0; j < gpnae_pkg::PMAX; j++) cc_field(where, l, $sformatf("tier %0d descriptor %0d bound %0d", t, d, j), ab + j, 32'(q.bound[j]), cc_img[k+j], bad);
        k += gpnae_pkg::PMAX;
        cc_field(where, l, $sformatf("tier %0d descriptor %0d np", t, d), ab + gpnae_pkg::F_SET, 32'(q.np), cc_img[k], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d deg", t, d), ab + gpnae_pkg::F_SET, 32'(q.deg), cc_img[k+1], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d base", t, d), ab + gpnae_pkg::F_SET, 32'(q.base), cc_img[k+2], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d reflect", t, d), ab + gpnae_pkg::F_SET, 32'(q.reflect), cc_img[k+3], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d exp_base", t, d), ab + gpnae_pkg::F_EXP, 32'(q.exp_base), cc_img[k+4], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d exp_deg", t, d), ab + gpnae_pkg::F_EXP, 32'(q.exp_deg), cc_img[k+5], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d center", t, d), ab + gpnae_pkg::F_CENTER, 32'(q.center), cc_img[k+6], bad);
        cc_field(where, l, $sformatf("tier %0d descriptor %0d frac", t, d), ab + gpnae_pkg::F_FRAC, 32'(q.frac), cc_img[k+7], bad);
        k += 8;
      end
    if (sienna_fmt_pkg::HAS_FLOAT_TAIL)
      for (int t = 0; t < gpnae_pkg::NUM_TIERS; t++) begin
        ab = gpnae_pkg::A_TAIL + gpnae_pkg::TAIL_STRIDE * t;
        for (int j = 0; j < gpnae_pkg::TAIL_MAXC; j++) cc_field(where, l, $sformatf("tail %0d coef %0d", t, j), ab + j, cc_tail[l][t].coef[j], cc_img[k+j], bad);
        cc_field(where, l, $sformatf("tail %0d deg", t), ab + gpnae_pkg::TAIL_MAXC, 32'(cc_tail[l][t].deg), cc_img[k+6], bad);
        cc_field(where, l, $sformatf("tail %0d newton", t), ab + gpnae_pkg::TAIL_MAXC, 32'(cc_tail[l][t].newton), cc_img[k+7], bad);
        k += CC_FT;
      end
  end
  cc_checks++;
  if (bad > 0) begin
    $display("[FAIL] COEF IMAGE %s: %0d fields differ over %0d lanes", where, bad, CC_LANES);
    cc_bad++;
  end
endtask
