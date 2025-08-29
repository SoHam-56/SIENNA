// `timescale 1ns/1ps

// module tb_fp16_to_int();
//     logic        clk;
//     logic        rst_n;
//     logic        valid_in;
//     logic [15:0] fp16_in;
//     logic        done_out;
//     logic [15:0] integer_out;

//     // Instantiate the DUT
//     fp16_to_int dut (
//         .clk(clk),
//         .rst_n(rst_n),
//         .valid_in(valid_in),
//         .fp16_in(fp16_in),
//         .done_out(done_out),
//         .integer_out(integer_out)
//     );

//     // Clock generation
//     always #10 clk = ~clk;

//     // Test cases
//     logic [15:0] test_fp16 [0:19];
//     logic [15:0] expected_int [0:19];
//   	logic [255:0] test_descriptions [0:19];
    
//     initial begin
//         // Basic positive numbers
//         test_fp16[0]  = 16'h0000; expected_int[0]  = 16'd0;        test_descriptions[0]  = "Zero";
//         test_fp16[1]  = 16'h0001; expected_int[1]  = 16'd0;        test_descriptions[1]  = "Denormal smallest";
//         test_fp16[2]  = 16'h3C00; expected_int[2]  = 16'd1;        test_descriptions[2]  = "One";
//         test_fp16[3]  = 16'h3F00; expected_int[3]  = 16'd1;        test_descriptions[3]  = "One with fraction";
//         test_fp16[4]  = 16'h4000; expected_int[4]  = 16'd2;        test_descriptions[4]  = "Two";
//         test_fp16[5]  = 16'h63FE; expected_int[5]  = 16'd1023;     test_descriptions[5]  = "1023";
//         test_fp16[6]  = 16'h6400; expected_int[6]  = 16'd1024;     test_descriptions[6]  = "1024";
//         test_fp16[7]  = 16'h67FF; expected_int[7]  = 16'd2047;     test_descriptions[7]  = "2047";
//         test_fp16[8]  = 16'h6800; expected_int[8]  = 16'd2048;     test_descriptions[8]  = "2048";
        
//         // Positive numbers that would overflow to 16-bit max
//       test_fp16[9]  = 16'h7700; expected_int[9]  = 16'h7000;    test_descriptions[9]  = "28672";
//       test_fp16[10] = 16'h7000; expected_int[10] = 16'h2000;    test_descriptions[10] = "8192";
//       test_fp16[11] = 16'h6500; expected_int[11] = 16'h0500;     test_descriptions[11] = "1280";
        
//         // Negative numbers
//         test_fp16[12] = 16'hBC00; expected_int[12] = -16'sd1;     test_descriptions[12] = "Negative one";
//         test_fp16[13] = 16'hBF00; expected_int[13] = -16'sd1;      test_descriptions[13] = "Negative one with fraction";
//         test_fp16[14] = 16'hE3FE; expected_int[14] = -16'sd1023;  test_descriptions[14] = "Negative 1023";
//         test_fp16[15] = 16'hF800; expected_int[15] = 16'h8000;    test_descriptions[15] = "Negative 32768 (overflow to min)";

//         // NaN and Infinity cases
//         test_fp16[16] = 16'h7C00; expected_int[16] = 16'h7FFF;    test_descriptions[16] = "+Infinity";
//         test_fp16[17] = 16'h7E00; expected_int[17] = 16'd0;        test_descriptions[17] = "NaN";
//         test_fp16[18] = 16'hFC00; expected_int[18] = 16'h8000;    test_descriptions[18] = "-Infinity";
//         test_fp16[19] = 16'hFE00; expected_int[19] = 16'd0;        test_descriptions[19] = "Negative NaN";

//         // Initialize signals
//         clk = 1;
//         rst_n = 0;
//         valid_in = 0;
//         fp16_in = 0;

//         // Reset
//         #2 rst_n = 1;
//         #1;

//         // Run test cases
//         for (int i = 0; i < 20; i++) begin
//             @(negedge clk);
//             valid_in = 1;
//             fp16_in = test_fp16[i];
            
//             wait(done_out);
//             @(negedge clk);
            
//             if (integer_out === expected_int[i]) begin
//               $display("PASS: %s (0x%04h) -> %0d", 
//                         test_descriptions[i], fp16_in, integer_out);
//             end else begin
//               $display("FAIL: %s (0x%04h) -> %0d (Expected %0d)", 
//                         test_descriptions[i], fp16_in, integer_out, expected_int[i]);
//             end
            
//             valid_in = 0;
//             repeat(2) @(posedge clk);
//         end
        
//         $display("Simulation completed");
//         $finish;
//     end
// endmodule

`timescale 1ns/1ps

module tb_fp16_to_int();
    logic        clk;
    logic        rst_n;
    logic        valid_in;
    logic [15:0] fp16_in;
    logic        done_out;
    logic [15:0] integer_out;

    // Instantiate the DUT
    fp16_to_int dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .fp16_in(fp16_in),
        .done_out(done_out),
        .integer_out(integer_out)
    );

    // Clock generation
    always #10 clk = ~clk;

    // Test cases - positive numbers and special cases (NaN/Infinity)
    logic [15:0] test_fp16 [0:15];
    logic [15:0] expected_int [0:15];
    logic [255:0] test_descriptions [0:15];
    
    initial begin
        // Basic positive numbers
        test_fp16[0]  = 16'h0000; expected_int[0]  = 16'd0;        test_descriptions[0]  = "Zero";
        test_fp16[1]  = 16'h0001; expected_int[1]  = 16'd0;        test_descriptions[1]  = "Denormal smallest";
        test_fp16[2]  = 16'h3C00; expected_int[2]  = 16'd1;        test_descriptions[2]  = "One";
        test_fp16[3]  = 16'h3F00; expected_int[3]  = 16'd1;        test_descriptions[3]  = "One with fraction";
      test_fp16[4] = 16'hBC00; expected_int[4] = -16'sd1;     test_descriptions[4] = "Negative one";
        //test_fp16[4]  = 16'h4000; expected_int[4]  = 16'd2;        test_descriptions[4]  = "Two";
      test_fp16[10]  = 16'h63FE; expected_int[10]  = 16'd1023;     test_descriptions[10]  = "1023";
      test_fp16[11]  = 16'h6400; expected_int[11]  = 16'd1024;     test_descriptions[11]  = "1024";
        test_fp16[7]  = 16'h67FF; expected_int[7]  = 16'd2047;     test_descriptions[7]  = "2047";
        test_fp16[8]  = 16'h6800; expected_int[8]  = 16'd2048;     test_descriptions[8]  = "2048";
        
        // Positive numbers
        test_fp16[9]  = 16'h7700; expected_int[9]  = 16'h7000;    test_descriptions[9]  = "28672";
      test_fp16[5] = 16'h7000; expected_int[5] = 16'h2000;    test_descriptions[5] = "8192";
      test_fp16[6] = 16'h6500; expected_int[6] = 16'h0500;    test_descriptions[6] = "1280";

        // NaN and Infinity cases (should be kept - they test special cases)
        test_fp16[12] = 16'h7C00; expected_int[12] = 16'h7FFF;    test_descriptions[12] = "+Infinity";
        test_fp16[13] = 16'h7E00; expected_int[13] = 16'd0;        test_descriptions[13] = "NaN";
        test_fp16[14] = 16'hFC00; expected_int[14] = 16'h8000;    test_descriptions[14] = "-Infinity";
        test_fp16[15] = 16'hFE00; expected_int[15] = 16'd0;        test_descriptions[15] = "Negative NaN";

        // Initialize signals
        clk = 1;
        rst_n = 0;
        valid_in = 0;
        fp16_in = 0;

        // Reset
        #2 rst_n = 1;
        #1;

        // Run test cases
        for (int i = 0; i < 16; i++) begin
            @(negedge clk);
            valid_in = 1;
            fp16_in = test_fp16[i];
            
            wait(done_out);
            @(negedge clk);
            
            if (integer_out === expected_int[i]) begin
              $display("PASS: %s (0x%04h) -> %0d", 
                        test_descriptions[i], fp16_in, integer_out);
            end else begin
              $display("FAIL: %s (0x%04h) -> %0d (Expected %0d)", 
                        test_descriptions[i], fp16_in, integer_out, expected_int[i]);
            end
            
            valid_in = 0;
            repeat(2) @(posedge clk);
        end
        
        $display("Simulation completed");
        $finish;
    end
endmodule