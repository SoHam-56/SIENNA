`timescale 1ns / 100ps

module sienna_top #(
    // Systolic Array Parameters
    parameter N = 32,                    // Matrix size (N x N)
    parameter DATA_WIDTH = 32,           // Data width for both modules
    parameter SRAM_DEPTH = N * N,        // Output SRAM depth
    
    // GPNAE Parameters  
    parameter ADDR_LINES = 5,            // Address lines for GPNAE (2^5 = 32 max terms)
    parameter CONTROL_WIDTH = 2,         // Control word width for activation functions
    
    // Memory Parameters
    parameter INPUT_A_FILE = "matrixA.mem",
    parameter INPUT_B_FILE = "matrixB.mem",
    parameter INTERMEDIATE_BUFFER_DEPTH = SRAM_DEPTH * 2  // Double buffer for pipeline
)(
    // Clock and Reset
    input  logic clk_i,
    input  logic rstn_i,
    
    // Top-level Control Interface
    input  logic start_pipeline_i,              // Start the entire pipeline
    input  logic [CONTROL_WIDTH-1:0] activation_function_i,  // Activation function select
    input  logic [ADDR_LINES-1:0] num_terms_i,              // Number of terms for GPNAE
    
    // Systolic Array Input Interface (for loading matrices)
    input  logic north_write_enable_i,
    input  logic [DATA_WIDTH-1:0] north_write_data_i,
    input  logic north_write_reset_i,
    
    input  logic west_write_enable_i,
    input  logic [DATA_WIDTH-1:0] west_write_data_i,
    input  logic west_write_reset_i,
    
    // Final Output Interface
    output logic [DATA_WIDTH-1:0] final_result_o,
    output logic pipeline_complete_o,
    output logic gpnae_done_o,
    
    // Status Outputs
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o,
    
    // Debug/Monitoring Outputs
    output logic [DATA_WIDTH-1:0] systolic_result_debug_o,
    output logic systolic_complete_debug_o,
    output logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer_count_debug_o
);

    // Internal State Machine
    typedef enum logic [2:0] {
        IDLE,
        SYSTOLIC_PROCESSING,
        BUFFER_TRANSFER,
        GPNAE_PROCESSING,
        PIPELINE_COMPLETE
    } pipeline_state_t;
    
    pipeline_state_t current_state, next_state;
    
    // Systolic Array Signals
    logic systolic_start_matrix_mult;
    logic systolic_matrix_mult_complete;
    logic systolic_collection_complete;
    logic systolic_collection_active;
    logic systolic_read_enable;
    logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_addr;
    logic [DATA_WIDTH-1:0] systolic_read_data;
    logic systolic_read_valid;
    logic north_queue_empty, west_queue_empty;
    
    // Intermediate Buffer Signals
    logic intermediate_wr_en;
    logic intermediate_rd_en;
    logic [DATA_WIDTH-1:0] intermediate_wr_data;
    logic [DATA_WIDTH-1:0] intermediate_rd_data;
    logic intermediate_full;
    logic intermediate_empty;
    logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] intermediate_count;
    
    // GPNAE Signals
    logic gpnae_signal_wr_en;
    logic [DATA_WIDTH-1:0] gpnae_signal_data;
    logic gpnae_last;
    logic gpnae_start;
    logic gpnae_full;
    logic gpnae_empty;
    logic gpnae_idle;
    logic gpnae_done;
    logic [DATA_WIDTH-1:0] gpnae_final_result;
    
    // Internal Counters and Control
    logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_counter;
    logic [$clog2(SRAM_DEPTH)-1:0] gpnae_write_counter;
    logic [$clog2(SRAM_DEPTH)-1:0] total_elements;
    logic transfer_active;
    logic all_data_transferred;
    logic all_data_processed;
    
    // Instantiate Systolic Array
    SystolicArray #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ROWS(INPUT_A_FILE),
        .COLS(INPUT_B_FILE)
    ) systolic_array_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .start_matrix_mult_i(systolic_start_matrix_mult),
        
        // North Queue Write interface
        .north_write_enable_i(north_write_enable_i),
        .north_write_data_i(north_write_data_i),
        .north_write_reset_i(north_write_reset_i),
        
        // West Queue Write interface  
        .west_write_enable_i(west_write_enable_i),
        .west_write_data_i(west_write_data_i),
        .west_write_reset_i(west_write_reset_i),
        
        // Queue status
        .north_queue_empty_o(north_queue_empty),
        .west_queue_empty_o(west_queue_empty),
        .matrix_mult_complete_o(systolic_matrix_mult_complete),
        
        // OutputSram read interface
        .read_enable_i(systolic_read_enable),
        .read_addr_i(systolic_read_addr),
        .read_data_o(systolic_read_data),
        .read_valid_o(systolic_read_valid),
        
        // OutputSram status signals
        .collection_complete_o(systolic_collection_complete),
        .collection_active_o(systolic_collection_active)
    );
    
    // Intermediate Buffer (FIFO) - Stores results from Systolic Array for GPNAE
    fifo_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(INTERMEDIATE_BUFFER_DEPTH)
    ) intermediate_buffer_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .wr_en_i(intermediate_wr_en),
        .wr_data_i(intermediate_wr_data),
        .rd_en_i(intermediate_rd_en),
        .rd_data_o(intermediate_rd_data),
        .full_o(intermediate_full),
        .empty_o(intermediate_empty),
        .count_o(intermediate_count)
    );
    
    // Instantiate GPNAE
    gpnae #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_LINES(ADDR_LINES),
        .CONTROL_WIDTH(CONTROL_WIDTH)
    ) gpnae_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .signal_i(gpnae_signal_data),
        .wr_en_i(gpnae_signal_wr_en),
        .last_i(gpnae_start),  // Use start signal as last signal trigger
        .terms_i(num_terms_i),
        .control_word_i(activation_function_i),
        .full_o(gpnae_full),
        .empty_o(gpnae_empty),
        .idle_o(gpnae_idle),
        .final_result_o(gpnae_final_result),
        .done_o(gpnae_done)
    );
    
    // State Machine - Sequential Logic
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // State Machine - Combinational Logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start_pipeline_i && !north_queue_empty && !west_queue_empty) begin
                    next_state = SYSTOLIC_PROCESSING;
                end
            end
            
            SYSTOLIC_PROCESSING: begin
                if (systolic_collection_complete) begin
                    next_state = BUFFER_TRANSFER;
                end
            end
            
            BUFFER_TRANSFER: begin
                if (all_data_transferred) begin
                    next_state = GPNAE_PROCESSING;
                end
            end
            
            GPNAE_PROCESSING: begin
                if (all_data_processed) begin
                    next_state = PIPELINE_COMPLETE;
                end
            end
            
            PIPELINE_COMPLETE: begin
                if (!start_pipeline_i) begin
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Control Logic
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            systolic_start_matrix_mult <= 1'b0;
            systolic_read_enable <= 1'b0;
            systolic_read_addr <= '0;
            systolic_read_counter <= '0;
            
            intermediate_wr_en <= 1'b0;
            intermediate_rd_en <= 1'b0;
            
            gpnae_signal_wr_en <= 1'b0;
            gpnae_last <= 1'b0;
            gpnae_start <= 1'b0;
            gpnae_write_counter <= '0;
            
            total_elements <= N * N;
            transfer_active <= 1'b0;
            all_data_transferred <= 1'b0;
            all_data_processed <= 1'b0;
            
        end else begin
            // Default values
            systolic_start_matrix_mult <= 1'b0;
            systolic_read_enable <= 1'b0;
            intermediate_wr_en <= 1'b0;
            intermediate_rd_en <= 1'b0;
            gpnae_signal_wr_en <= 1'b0;
            gpnae_last <= 1'b0;
            gpnae_start <= 1'b0;
            
            case (current_state)
                IDLE: begin
                    systolic_read_counter <= '0;
                    gpnae_write_counter <= '0;
                    transfer_active <= 1'b0;
                    all_data_transferred <= 1'b0;
                    all_data_processed <= 1'b0;
                end
                
                SYSTOLIC_PROCESSING: begin
                    if (current_state != next_state) begin // Just entered this state
                        systolic_start_matrix_mult <= 1'b1;
                    end
                end
                
                BUFFER_TRANSFER: begin
                    if (!transfer_active) begin
                        transfer_active <= 1'b1;
                    end
                    
                    // Read from systolic array and write to intermediate buffer
                    if (systolic_read_counter < total_elements && !intermediate_full) begin
                        systolic_read_enable <= 1'b1;
                        systolic_read_addr <= systolic_read_counter;
                        
                        if (systolic_read_valid) begin
                            intermediate_wr_en <= 1'b1;
                            systolic_read_counter <= systolic_read_counter + 1;
                            
                            if (systolic_read_counter == total_elements - 1) begin
                                all_data_transferred <= 1'b1;
                            end
                        end
                    end
                end
                
                GPNAE_PROCESSING: begin
                    // Read from intermediate buffer and write to GPNAE
                    if (!intermediate_empty && gpnae_idle && !gpnae_full) begin
                        intermediate_rd_en <= 1'b1;
                        gpnae_signal_wr_en <= 1'b1;
                        gpnae_write_counter <= gpnae_write_counter + 1;
                        
                        // Check if this is the last element
                        if (gpnae_write_counter == num_terms_i - 1) begin
                            gpnae_last <= 1'b1;
                            gpnae_start <= 1'b1;
                            all_data_processed <= 1'b1;
                        end
                    end
                end
                
                PIPELINE_COMPLETE: begin
                    // Hold in this state until start signal is deasserted
                end
                
            endcase
        end
    end
    
    // Data Path Connections
    always_comb begin
        // Intermediate buffer write data comes from systolic array
        intermediate_wr_data = systolic_read_data;
        
        // GPNAE signal data comes from intermediate buffer
        gpnae_signal_data = intermediate_rd_data;
        
        // Output assignments
        final_result_o = gpnae_final_result;
        pipeline_complete_o = (current_state == PIPELINE_COMPLETE);
        gpnae_done_o = gpnae_done;
        
        // Status outputs
        systolic_busy_o = (current_state == SYSTOLIC_PROCESSING) || 
                         (current_state == BUFFER_TRANSFER);
        gpnae_busy_o = (current_state == GPNAE_PROCESSING);
        intermediate_buffer_full_o = intermediate_full;
        intermediate_buffer_empty_o = intermediate_empty;
        
        // Debug outputs
        systolic_result_debug_o = systolic_read_data;
        systolic_complete_debug_o = systolic_collection_complete;
        buffer_count_debug_o = intermediate_count;
    end

endmodule

// Simple FIFO Buffer Module
module fifo_buffer #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024
)(
    input  logic clk_i,
    input  logic rstn_i,
    input  logic wr_en_i,
    input  logic [DATA_WIDTH-1:0] wr_data_i,
    input  logic rd_en_i,
    output logic [DATA_WIDTH-1:0] rd_data_o,
    output logic full_o,
    output logic empty_o,
    output logic [$clog2(DEPTH)-1:0] count_o
);

    logic [DATA_WIDTH-1:0] buffer_mem [0:DEPTH-1];
    logic [$clog2(DEPTH)-1:0] wr_ptr, rd_ptr;
    logic [$clog2(DEPTH)-1:0] count;
    
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count <= '0;
        end else begin
            // Write operation
            if (wr_en_i && !full_o) begin
                buffer_mem[wr_ptr] <= wr_data_i;
                wr_ptr <= (wr_ptr == DEPTH-1) ? '0 : wr_ptr + 1;
            end
            
            // Read operation  
            if (rd_en_i && !empty_o) begin
                rd_ptr <= (rd_ptr == DEPTH-1) ? '0 : rd_ptr + 1;
            end
            
            // Count update
            if (wr_en_i && !rd_en_i && !full_o) begin
                count <= count + 1;
            end else if (!wr_en_i && rd_en_i && !empty_o) begin
                count <= count - 1;
            end
        end
    end
    
    // Output assignments
    assign rd_data_o = buffer_mem[rd_ptr];
    assign full_o = (count == DEPTH);
    assign empty_o = (count == 0);
    assign count_o = count;

endmodule
