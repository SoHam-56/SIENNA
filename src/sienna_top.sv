`timescale 1ns / 100ps

module sienna_top #(
    // Systolic Array Parameters
    parameter N = 32,                   // Matrix size (N x N)
    parameter DATA_WIDTH = 32,          // Data width for all modules
    parameter SRAM_DEPTH = N * N,       // Output SRAM depth
    
    // GPNAE Parameters  
    parameter ADDR_LINES = 5,           // Address lines for GPNAE (2^5 = 32 max terms)
    parameter CONTROL_WIDTH = 2,        // Control word width for activation functions
    
    // Maxpool Parameters
    parameter IN_ROWS = 5,
    parameter IN_COLS = 5,
    parameter POOL_H = 2,
    parameter POOL_W = 2,
    parameter PADDING = 1,
    
    // Dropout Parameters
    parameter DROPOUT_P = 0.5,
    parameter LFSR_WIDTH = 32,
    
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
    
    // Maxpool SRAM Interface (if needed for input)
    input  logic [DATA_WIDTH-1:0] maxpool_input_data_i,
    input  logic maxpool_input_valid_i,
    
    // Final Output Interface
    output logic [DATA_WIDTH-1:0] final_result_o,
    output logic pipeline_complete_o,
    output logic gpnae_done_o,
    
    // Status Outputs
    output logic systolic_busy_o,
    output logic gpnae_busy_o,
    output logic maxpool_busy_o,
    output logic dropout_busy_o,
    output logic intermediate_buffer_full_o,
    output logic intermediate_buffer_empty_o,
    
    // Debug/Monitoring Outputs
    output logic [DATA_WIDTH-1:0] systolic_result_debug_o,
    output logic systolic_complete_debug_o,
    output logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer_count_debug_o
);

    // Internal State Machine
    typedef enum logic [3:0] {
        IDLE,
        SYSTOLIC_PROCESSING,
        BUFFER_TRANSFER_1,
        GPNAE_PROCESSING,
        BUFFER_TRANSFER_2,
        MAXPOOL_PROCESSING,
        BUFFER_TRANSFER_3,
        DROPOUT_PROCESSING,
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
    
    // Intermediate Buffer Signals (FIFO between Systolic and GPNAE)
    logic buffer1_wr_en;
    logic buffer1_rd_en;
    logic [DATA_WIDTH-1:0] buffer1_wr_data;
    logic [DATA_WIDTH-1:0] buffer1_rd_data;
    logic buffer1_full;
    logic buffer1_empty;
    logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer1_count;
    
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
    
    // Buffer between GPNAE and Maxpool
    logic buffer2_wr_en;
    logic buffer2_rd_en;
    logic [DATA_WIDTH-1:0] buffer2_wr_data;
    logic [DATA_WIDTH-1:0] buffer2_rd_data;
    logic buffer2_full;
    logic buffer2_empty;
    logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer2_count;
    
    // Maxpool Signals
    logic maxpool_start;
    logic maxpool_done;
    logic maxpool_out_valid;
    logic [DATA_WIDTH-1:0] maxpool_out_data;
    logic maxpool_in_rd_en;
    logic [$clog2(IN_ROWS*IN_COLS)-1:0] maxpool_in_addr;
    
    // Buffer between Maxpool and Dropout
    logic buffer3_wr_en;
    logic buffer3_rd_en;
    logic [DATA_WIDTH-1:0] buffer3_wr_data;
    logic [DATA_WIDTH-1:0] buffer3_rd_data;
    logic buffer3_full;
    logic buffer3_empty;
    logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer3_count;
    
    // Dropout Signals
    logic dropout_en;
    logic dropout_training_mode;
    logic dropout_valid_out;
    logic [(2*DATA_WIDTH)-1:0] dropout_data_out;
    
    // Final Output Buffer
    logic buffer4_wr_en;
    logic buffer4_rd_en;
    logic [(2*DATA_WIDTH)-1:0] buffer4_wr_data;
    logic [(2*DATA_WIDTH)-1:0] buffer4_rd_data;
    logic buffer4_full;
    logic buffer4_empty;
    logic [$clog2(INTERMEDIATE_BUFFER_DEPTH)-1:0] buffer4_count;
    
    // Internal Counters and Control
    logic [$clog2(SRAM_DEPTH)-1:0] systolic_read_counter;
    logic [$clog2(SRAM_DEPTH)-1:0] gpnae_write_counter;
    logic [$clog2(SRAM_DEPTH)-1:0] total_elements;
    logic transfer_active;
    logic all_data_transferred_1;
    logic all_data_processed_gpnae;
    logic all_data_transferred_2;
    logic maxpool_processing_complete;
    logic all_data_transferred_3;
    logic dropout_processing_complete;
    
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
    
    // Buffer 1: Systolic Array -> GPNAE
    fifo_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(INTERMEDIATE_BUFFER_DEPTH)
    ) buffer1_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .wr_en_i(buffer1_wr_en),
        .wr_data_i(buffer1_wr_data),
        .rd_en_i(buffer1_rd_en),
        .rd_data_o(buffer1_rd_data),
        .full_o(buffer1_full),
        .empty_o(buffer1_empty),
        .count_o(buffer1_count)
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
        .last_i(gpnae_start),
        .terms_i(num_terms_i),
        .control_word_i(activation_function_i),
        .full_o(gpnae_full),
        .empty_o(gpnae_empty),
        .idle_o(gpnae_idle),
        .final_result_o(gpnae_final_result),
        .done_o(gpnae_done)
    );
    
    // Buffer 2: GPNAE -> Maxpool
    fifo_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(INTERMEDIATE_BUFFER_DEPTH)
    ) buffer2_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .wr_en_i(buffer2_wr_en),
        .wr_data_i(buffer2_wr_data),
        .rd_en_i(buffer2_rd_en),
        .rd_data_o(buffer2_rd_data),
        .full_o(buffer2_full),
        .empty_o(buffer2_empty),
        .count_o(buffer2_count)
    );
    
    // Instantiate Maxpool
    Maxpool_2D #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_ROWS(IN_ROWS),
        .IN_COLS(IN_COLS),
        .POOL_H(POOL_H),
        .POOL_W(POOL_W),
        .PADDING(PADDING)
    ) maxpool_inst (
        .clk(clk_i),
        .rst_n(rstn_i),
        .start(maxpool_start),
        .done(maxpool_done),
        .in_addr(maxpool_in_addr),
        .in_data(maxpool_input_data_i),  // External input or from buffer
        .in_rd_en(maxpool_in_rd_en),
        .out_data(maxpool_out_data),
        .out_valid(maxpool_out_valid)
    );
    
    // Buffer 3: Maxpool -> Dropout
    fifo_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(INTERMEDIATE_BUFFER_DEPTH)
    ) buffer3_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .wr_en_i(buffer3_wr_en),
        .wr_data_i(buffer3_wr_data),
        .rd_en_i(buffer3_rd_en),
        .rd_data_o(buffer3_rd_data),
        .full_o(buffer3_full),
        .empty_o(buffer3_empty),
        .count_o(buffer3_count)
    );
    
    // Instantiate Dropout
    dropout_module #(
        .DATA_WIDTH(DATA_WIDTH),
        .DROPOUT_P(DROPOUT_P),
        .LFSR_WIDTH(LFSR_WIDTH)
    ) dropout_inst (
        .clk(clk_i),
        .rst_n(rstn_i),
        .en(dropout_en),
        .training_mode(dropout_training_mode),
        .data_in(buffer3_rd_data),
        .data_out(dropout_data_out),
        .valid_out(dropout_valid_out)
    );
    
    // Buffer 4: Dropout -> Final Output
    fifo_buffer #(
        .DATA_WIDTH(2*DATA_WIDTH),  // Dropout output is 2x width
        .DEPTH(INTERMEDIATE_BUFFER_DEPTH)
    ) buffer4_inst (
        .clk_i(clk_i),
        .rstn_i(rstn_i),
        .wr_en_i(buffer4_wr_en),
        .wr_data_i(buffer4_wr_data),
        .rd_en_i(buffer4_rd_en),
        .rd_data_o(buffer4_rd_data),
        .full_o(buffer4_full),
        .empty_o(buffer4_empty),
        .count_o(buffer4_count)
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
                    next_state = BUFFER_TRANSFER_1;
                end
            end
            
            BUFFER_TRANSFER_1: begin
                if (all_data_transferred_1) begin
                    next_state = GPNAE_PROCESSING;
                end
            end
            
            GPNAE_PROCESSING: begin
                if (all_data_processed_gpnae) begin
                    next_state = BUFFER_TRANSFER_2;
                end
            end
            
            BUFFER_TRANSFER_2: begin
                if (all_data_transferred_2) begin
                    next_state = MAXPOOL_PROCESSING;
                end
            end
            
            MAXPOOL_PROCESSING: begin
                if (maxpool_processing_complete) begin
                    next_state = BUFFER_TRANSFER_3;
                end
            end
            
            BUFFER_TRANSFER_3: begin
                if (all_data_transferred_3) begin
                    next_state = DROPOUT_PROCESSING;
                end
            end
            
            DROPOUT_PROCESSING: begin
                if (dropout_processing_complete) begin
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
            // Reset all control signals
            systolic_start_matrix_mult <= 1'b0;
            systolic_read_enable <= 1'b0;
            systolic_read_addr <= '0;
            systolic_read_counter <= '0;
            
            buffer1_wr_en <= 1'b0;
            buffer1_rd_en <= 1'b0;
            buffer2_wr_en <= 1'b0;
            buffer2_rd_en <= 1'b0;
            buffer3_wr_en <= 1'b0;
            buffer3_rd_en <= 1'b0;
            buffer4_wr_en <= 1'b0;
            buffer4_rd_en <= 1'b0;
            
            gpnae_signal_wr_en <= 1'b0;
            gpnae_last <= 1'b0;
            gpnae_start <= 1'b0;
            gpnae_write_counter <= '0;
            
            maxpool_start <= 1'b0;
            dropout_en <= 1'b0;
            dropout_training_mode <= 1'b0;
            
            total_elements <= N * N;
            transfer_active <= 1'b0;
            all_data_transferred_1 <= 1'b0;
            all_data_processed_gpnae <= 1'b0;
            all_data_transferred_2 <= 1'b0;
            maxpool_processing_complete <= 1'b0;
            all_data_transferred_3 <= 1'b0;
            dropout_processing_complete <= 1'b0;
            
        end else begin
            // Default values
            systolic_start_matrix_mult <= 1'b0;
            systolic_read_enable <= 1'b0;
            buffer1_wr_en <= 1'b0;
            buffer1_rd_en <= 1'b0;
            buffer2_wr_en <= 1'b0;
            buffer2_rd_en <= 1'b0;
            buffer3_wr_en <= 1'b0;
            buffer3_rd_en <= 1'b0;
            buffer4_wr_en <= 1'b0;
            buffer4_rd_en <= 1'b0;
            gpnae_signal_wr_en <= 1'b0;
            gpnae_last <= 1'b0;
            gpnae_start <= 1'b0;
            maxpool_start <= 1'b0;
            dropout_en <= 1'b0;
            
            case (current_state)
                IDLE: begin
                    systolic_read_counter <= '0;
                    gpnae_write_counter <= '0;
                    transfer_active <= 1'b0;
                    all_data_transferred_1 <= 1'b0;
                    all_data_processed_gpnae <= 1'b0;
                    all_data_transferred_2 <= 1'b0;
                    maxpool_processing_complete <= 1'b0;
                    all_data_transferred_3 <= 1'b0;
                    dropout_processing_complete <= 1'b0;
                }
                
                SYSTOLIC_PROCESSING: begin
                    if (current_state != next_state) begin
                        systolic_start_matrix_mult <= 1'b1;
                    end
                }
                
                BUFFER_TRANSFER_1: begin
                    if (!transfer_active) begin
                        transfer_active <= 1'b1;
                    end
                    
                    // Read from systolic array and write to buffer1
                    if (systolic_read_counter < total_elements && !buffer1_full) begin
                        systolic_read_enable <= 1'b1;
                        systolic_read_addr <= systolic_read_counter;
                        
                        if (systolic_read_valid) begin
                            buffer1_wr_en <= 1'b1;
                            systolic_read_counter <= systolic_read_counter + 1;
                            
                            if (systolic_read_counter == total_elements - 1) begin
                                all_data_transferred_1 <= 1'b1;
                            end
                        end
                    end
                }
                
                GPNAE_PROCESSING: begin
                    // Read from buffer1 and write to GPNAE
                    if (!buffer1_empty && gpnae_idle && !gpnae_full) begin
                        buffer1_rd_en <= 1'b1;
                        gpnae_signal_wr_en <= 1'b1;
                        gpnae_write_counter <= gpnae_write_counter + 1;
                        
                        if (gpnae_write_counter == num_terms_i - 1) begin
                            gpnae_last <= 1'b1;
                            gpnae_start <= 1'b1;
                            all_data_processed_gpnae <= 1'b1;
                        end
                    end
                }
                
                BUFFER_TRANSFER_2: begin
                    // Write GPNAE results to buffer2
                    if (gpnae_done && !buffer2_full) begin
                        buffer2_wr_en <= 1'b1;
                        all_data_transferred_2 <= 1'b1;
                    end
                }
                
                MAXPOOL_PROCESSING: begin
                    if (current_state != next_state) begin
                        maxpool_start <= 1'b1;
                    end
                    
                    if (maxpool_done) begin
                        maxpool_processing_complete <= 1'b1;
                    end
                }
                
                BUFFER_TRANSFER_3: begin
                    // Write maxpool results to buffer3
                    if (maxpool_out_valid && !buffer3_full) begin
                        buffer3_wr_en <= 1'b1;
                        all_data_transferred_3 <= 1'b1;
                    end
                }
                
                DROPOUT_PROCESSING: begin
                    // Read from buffer3, process through dropout, write to buffer4
                    if (!buffer3_empty && !buffer4_full) begin
                        buffer3_rd_en <= 1'b1;
                        dropout_en <= 1'b1;
                        dropout_training_mode <= 1'b0;  // Set to 0 for inference mode
                        
                        if (dropout_valid_out) begin
                            buffer4_wr_en <= 1'b1;
                            dropout_processing_complete <= 1'b1;
                        end
                    end
                }
                
                PIPELINE_COMPLETE: begin
                    // Hold final result in buffer4
                }
                
            endcase
        end
    end
    
    // Data Path Connections
    always_comb begin
        // Buffer connections
        buffer1_wr_data = systolic_read_data;
        gpnae_signal_data = buffer1_rd_data;
        buffer2_wr_data = gpnae_final_result;
        buffer3_wr_data = maxpool_out_data;
        buffer4_wr_data = dropout_data_out;
        
        // Output assignments
        final_result_o = buffer4_rd_data;
        pipeline_complete_o = (current_state == PIPELINE_COMPLETE);
        gpnae_done_o = gpnae_done;
        
        // Status outputs
        systolic_busy_o = (current_state == SYSTOLIC_PROCESSING) || 
                         (current_state == BUFFER_TRANSFER_1);
        gpnae_busy_o = (current_state == GPNAE_PROCESSING);
        maxpool_busy_o = (current_state == MAXPOOL_PROCESSING);
        dropout_busy_o = (current_state == DROPOUT_PROCESSING);
        intermediate_buffer_full_o = buffer1_full;
        intermediate_buffer_empty_o = buffer1_empty;
        
        // Debug outputs
        systolic_result_debug_o = systolic_read_data;
        systolic_complete_debug_o = systolic_collection_complete;
        buffer_count_debug_o = buffer1_count;
    end

endmodule
