`timescale 1ns/1ps
module Neural_Processing_System_tb;

    parameter DATA_WIDTH  = 8;
    parameter IN_ROWS     = 10;
    parameter IN_COLS     = 10;
    parameter SEG_ROWS    = 3;     // Corresponds to POOL_H
    parameter SEG_COLS    = 3;     // Corresponds to POOL_W
    parameter STRIDE_ROWS = 3;     // Added parameter for stride
    parameter STRIDE_COLS = 4;     // Added parameter for stride
    parameter DROPOUT_P   = 0.5;
    parameter PADDING     = 1;     // Added parameter for padding

    // --- Clock and reset ---
    logic clk = 0;
    logic rst_n = 0;

    // --- DUT signals ---
    logic start;
    logic done;
  
  	localparam OUT_ROWS = (PADDING == 1) ? ((IN_ROWS - 1) / STRIDE_ROWS) + 1 : ((IN_ROWS - SEG_ROWS) / STRIDE_ROWS) + 1;
    localparam OUT_COLS = (PADDING == 1) ? ((IN_COLS - 1) / STRIDE_COLS) + 1 : ((IN_COLS - SEG_COLS) / STRIDE_COLS) + 1;

    always #5 clk = ~clk;
  
    Neural_Processing_System #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_ROWS(IN_ROWS),
        .IN_COLS(IN_COLS),
        .SEG_ROWS(SEG_ROWS),
        .SEG_COLS(SEG_COLS),
        .STRIDE_ROWS(STRIDE_ROWS),
        .STRIDE_COLS(STRIDE_COLS),
        .DROPOUT_P(DROPOUT_P),
        .PADDING(PADDING)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done)
    );

    // --- Test stimulus and data generation ---
    initial begin
        // Reset sequence
        rst_n = 0;
        start = 0;
        #20;
        rst_n = 1;
        
        for (int i = 0; i < IN_ROWS; i++) begin
            for (int j = 0; j < IN_COLS; j++) begin
                dut.input_sram.mem[i * IN_COLS + j] = i + j;
            end
        end
        
        #10;
        start = 1;
        #10;
        start = 0;
        
        // Wait for processing to complete.
        wait(done);
        #100; // Wait a few extra cycles for signals to settle
        
        // Display results for verification
        $display("Input SRAM contents:");
        for (int i = 0; i < IN_ROWS; i++) begin
            for (int j = 0; j < IN_COLS; j++) begin
                $write("%3d ", dut.input_sram.mem[i * IN_COLS + j]);
            end
            $display();
        end
        
        // Calculate output size using DUT's internal localparams
        
        $display("\nOutput SRAM contents after maxpool and dropout:");
        for (int i = 0; i < OUT_ROWS; i++) begin
            for (int j = 0; j < OUT_COLS; j++) begin
                $write("%3d ", dut.output_sram.mem[i * OUT_COLS + j]);
            end
            $display();
        end
        
        $display("\nSimulation completed.");
        $finish;
    end

    // --- Monitor progress ---
    // This block displays real-time updates of the pipeline's progress.
    always @(posedge clk) begin
        if (dut.maxpool_valid)
            $display("Time %0t: MaxPool output = %d", $time, dut.maxpool_out);
        if (dut.dropout_valid)
            $display("Time %0t: Dropout output = %d", $time, dut.dropout_out);
        if (done)
            $display("Time %0t: Processing completed", $time);
    end

endmodule