/*
 * input_buffer.v — 128-Sample x 2-Channel I/Q Input Shift Register
 * ==================================================================
 * Serial-in, parallel-out buffer for loading I/Q radar pulse data.
 *
 * Operation:
 *   - Assert `wr_en` and provide {I[7:0], Q[7:0]} on `din` each clock.
 *   - After 128 clocks, `full` goes high and the entire 128x2 buffer
 *     is available via `data_out`.
 *   - Assert `clear` to reset for next pulse.
 *
 * Parameters:
 *   NUM_SAMPLES = 128   — time-steps per pulse
 *   DATA_WIDTH  = 8     — bits per I or Q sample (INT8)
 */

module input_buffer #(
    parameter NUM_SAMPLES = 128,
    parameter DATA_WIDTH  = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          wr_en,
    input  wire                          clear,
    input  wire [2*DATA_WIDTH-1:0]       din,      // {I[7:0], Q[7:0]}
    output reg                           full,
    output wire [NUM_SAMPLES*2*DATA_WIDTH-1:0] data_out  // flattened buffer
);

    // Internal storage: 128 entries x 16 bits (I+Q)
    reg [2*DATA_WIDTH-1:0] buffer [0:NUM_SAMPLES-1];
    reg [$clog2(NUM_SAMPLES):0] wr_ptr;

    integer i;

    // Flatten buffer to output bus
    genvar g;
    generate
        for (g = 0; g < NUM_SAMPLES; g = g + 1) begin : gen_flatten
            assign data_out[g*2*DATA_WIDTH +: 2*DATA_WIDTH] = buffer[g];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            full   <= 1'b0;
            for (i = 0; i < NUM_SAMPLES; i = i + 1)
                buffer[i] <= {(2*DATA_WIDTH){1'b0}};
        end else if (clear) begin
            wr_ptr <= 0;
            full   <= 1'b0;
        end else if (wr_en && !full) begin
            buffer[wr_ptr] <= din;
            if (wr_ptr == NUM_SAMPLES - 1) begin
                full <= 1'b1;
            end
            wr_ptr <= wr_ptr + 1;
        end
    end

endmodule
