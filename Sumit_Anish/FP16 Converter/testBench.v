`timescale 1ns/1ps

module tb_int_to_fp16();
    logic        clk;
    logic        rst_n;
    logic        valid_in;
    logic [19:0] integer_in;
    logic       done_out;
    logic [15:0] fp16_out;

    // Instantiate the DUT
    int_to_fp16 dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .integer_in(integer_in),
        .done_out(done_out),
        .fp16_out(fp16_out)
    );

    // Clock generation
    always #2 clk = ~clk;

    // Test cases
    logic [19:0] test_integers [0:12];
    logic [15:0] expected_fp16 [0:12];
    logic [127:0] test_descriptions [0:12];
    
    integer i;
    integer random_tests;

    initial begin
        // Initialize test cases
        test_integers[0] = 20'd0;       expected_fp16[0] = 16'h0000; test_descriptions[0] = "Zero";
        test_integers[1] = 20'd1;       expected_fp16[1] = 16'h3C00; test_descriptions[1] = "One";
        test_integers[2] = 20'd2;       expected_fp16[2] = 16'h4000; test_descriptions[2] = "Two";
      test_integers[3] = 20'd1023;    expected_fp16[3] = 16'h63FE; test_descriptions[3] = "1023";
      test_integers[4] = 20'd1024;    expected_fp16[4] = 16'h6400; test_descriptions[4] = "1024";
      test_integers[5] = 20'd2047;    expected_fp16[5] = 16'h67FF; test_descriptions[5] = "2047";
      test_integers[6] = 20'd2048;    expected_fp16[6] = 16'h6800; test_descriptions[6] = "2048";
        test_integers[7] = 20'd32767;   expected_fp16[7] = 16'h7800; test_descriptions[7] = "32767";
        test_integers[8] = 20'd65504;   expected_fp16[8] = 16'h7BFF; test_descriptions[8] = "65504";
      test_integers[9] = 20'd1111;   expected_fp16[9] = 16'h6457; test_descriptions[9] = "65505";
        test_integers[10] = -20'sd1;    expected_fp16[10] = 16'hBC00; test_descriptions[10] = "Negative one";
      test_integers[11] = -20'sd1023; expected_fp16[11] = 16'hE3FE; test_descriptions[11] = "Negative 1023";
        test_integers[12] = -20'sd32768;expected_fp16[12] = 16'hF800; test_descriptions[12] = "Negative 32768";

        // Initialize signals
        clk = 1;
        rst_n = 1;
        valid_in = 0;
       // integer_in = 0;

        // Reset
        //#10 rst_n = 1;
        #1;

        // Run test cases
      for (i = 0; i < 13; i = i + 1) begin
            $display("Testing case: %s", test_descriptions[i]);
          	valid_in = 1;
            integer_in = test_integers[i];
            
            // Wait for conversion to complete
            @(posedge done_out);
            #1; // Small delay for output to stabilize
            
            // Check result
        if (fp16_out === expected_fp16[i-1]) begin
                $display("PASS: Input %0d -> Output %04h", 
                        test_integers[i], fp16_out);
            end else begin
                $display("FAIL: Input %0d -> Output %04h", 
                        test_integers[i], fp16_out);
            end
            
            valid_in = 0;
            #20; // Wait a couple cycles between tests
        end
        $finish;
    end
endmodule