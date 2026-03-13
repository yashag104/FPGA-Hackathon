/*
 * gap_unit.v — Global Average Pooling (INT8)
 * ============================================
 * Accumulates across TIME_STEPS time-steps per channel, then divides
 * by right-shifting (integer divide by TIME_STEPS, which must be a power of 2).
 *
 * For RadarNet Block 2 output: 64 channels × 32 time-steps.
 * Division by 32 = right-shift by 5.
 *
 * FSM: IDLE -> ACCUMULATE -> DIVIDE -> DONE
 */

module gap_unit #(
    parameter NUM_CHANNELS = 64,
    parameter TIME_STEPS   = 32,
    parameter DATA_WIDTH   = 8,
    parameter SHIFT_BITS   = 5      // log2(TIME_STEPS) = log2(32) = 5
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input: flattened feature map (NUM_CHANNELS * TIME_STEPS * 8 bits)
    input  wire [NUM_CHANNELS*TIME_STEPS*DATA_WIDTH-1:0] feature_in,

    // Output: one INT8 value per channel
    output reg  [NUM_CHANNELS*DATA_WIDTH-1:0] gap_out,
    output reg         out_valid
);

    localparam S_IDLE = 2'd0;
    localparam S_ACC  = 2'd1;
    localparam S_DIV  = 2'd2;
    localparam S_DONE = 2'd3;

    reg [1:0] state;
    reg [$clog2(NUM_CHANNELS):0] ch;
    reg [$clog2(TIME_STEPS):0]   ts;

    // 16-bit accumulator per channel (8-bit * 32 max = 13 bits needed)
    reg signed [15:0] acc;

    // Extract one input sample
    wire signed [DATA_WIDTH-1:0] sample;
    wire [$clog2(NUM_CHANNELS*TIME_STEPS*DATA_WIDTH)-1:0] bit_idx;
    assign bit_idx = (ch * TIME_STEPS + ts) * DATA_WIDTH;
    assign sample = $signed(feature_in[bit_idx +: DATA_WIDTH]);

    // Saturation helper
    function [DATA_WIDTH-1:0] saturate;
        input signed [15:0] val;
        begin
            if (val > 127)
                saturate = 8'sd127;
            else if (val < -128)
                saturate = -8'sd128;
            else
                saturate = val[DATA_WIDTH-1:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            done      <= 1'b0;
            out_valid <= 1'b0;
            ch <= 0; ts <= 0;
            acc <= 16'd0;
            gap_out <= {(NUM_CHANNELS*DATA_WIDTH){1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done      <= 1'b0;
                    out_valid <= 1'b0;
                    if (start) begin
                        state <= S_ACC;
                        ch <= 0; ts <= 0;
                        acc <= 16'd0;
                    end
                end

                S_ACC: begin
                    acc <= acc + {{8{sample[DATA_WIDTH-1]}}, sample};
                    if (ts < TIME_STEPS - 1) begin
                        ts <= ts + 1;
                    end else begin
                        // Done accumulating this channel
                        state <= S_DIV;
                    end
                end

                S_DIV: begin
                    // Arithmetic right shift = divide by 2^SHIFT_BITS
                    gap_out[ch*DATA_WIDTH +: DATA_WIDTH] <=
                        saturate(acc >>> SHIFT_BITS);
                    acc <= 16'd0;
                    ts  <= 0;
                    if (ch < NUM_CHANNELS - 1) begin
                        ch <= ch + 1;
                        state <= S_ACC;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    out_valid <= 1'b1;
                    state     <= S_IDLE;
                end
            endcase
        end
    end

endmodule
