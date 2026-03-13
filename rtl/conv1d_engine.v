/*
 * conv1d_engine.v — Parameterised 1D Convolution Engine (INT8)
 * =============================================================
 * Performs a single Conv1D operation:
 *   - Slides a kernel of size KERNEL_SIZE across IN_LENGTH time-steps
 *   - Accumulates INT8 x INT8 MACs into 24-bit accumulators
 *   - Truncates with saturation to 8-bit signed output
 *
 * Parameters:
 *   IN_CHANNELS  — number of input channels
 *   OUT_CHANNELS — number of output channels (filters)
 *   IN_LENGTH    — input time-series length
 *   KERNEL_SIZE  — convolution kernel width
 *   STRIDE       — stride of the convolution
 *   WEIGHT_DEPTH — total weight ROM entries for this layer
 *
 * The engine reads weights sequentially from weight_addr / weight_data
 * interface (connected to weight_rom.v).
 *
 * FSM: IDLE -> COMPUTE -> DONE
 *   COMPUTE iterates over output positions, channels, kernel taps.
 */

module conv1d_engine #(
    parameter IN_CHANNELS  = 2,
    parameter OUT_CHANNELS = 16,
    parameter IN_LENGTH    = 128,
    parameter KERNEL_SIZE  = 7,
    parameter STRIDE       = 1,
    parameter PAD          = 3,
    parameter BIAS_EN      = 1       // 1 = add bias after accumulation
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input feature map (flattened: IN_CHANNELS * IN_LENGTH * 8 bits)
    input  wire [IN_CHANNELS*IN_LENGTH*8-1:0] feature_in,

    // Weight ROM interface
    output reg  [15:0] weight_addr,
    input  wire [ 7:0] weight_data,   // INT8 signed weight

    // Bias ROM interface
    output reg  [15:0] bias_addr,
    input  wire [ 7:0] bias_data,     // INT8 signed bias

    // Output feature map
    output reg  [OUT_CHANNELS*((IN_LENGTH+2*PAD-KERNEL_SIZE)/STRIDE+1)*8-1:0] feature_out,
    output reg         out_valid
);

    // Derived constants
    localparam OUT_LENGTH = (IN_LENGTH + 2*PAD - KERNEL_SIZE) / STRIDE + 1;
    localparam TOTAL_WEIGHTS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE;

    // FSM states
    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_BIAS    = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;

    // Loop counters
    reg [$clog2(OUT_CHANNELS):0] oc;   // output channel
    reg [$clog2(OUT_LENGTH):0]   op;   // output position
    reg [$clog2(IN_CHANNELS):0]  ic;   // input channel
    reg [$clog2(KERNEL_SIZE):0]  kk;   // kernel tap

    // Accumulator (24-bit to prevent overflow)
    reg signed [23:0] acc;

    // Temporary products
    wire signed [7:0] w_val;
    wire signed [7:0] x_val;
    wire signed [15:0] product;

    assign w_val = $signed(weight_data);

    // Index into flattened feature_in
    wire [$clog2(IN_CHANNELS*IN_LENGTH)-1:0] feat_idx;
    wire signed [$clog2(IN_LENGTH)+1:0] in_pos;  // may be negative (padding)
    assign in_pos = $signed({1'b0, op}) * STRIDE + $signed({1'b0, kk}) - PAD;

    // Zero-padding: if in_pos < 0 or >= IN_LENGTH, input is 0
    wire in_bounds;
    assign in_bounds = (in_pos >= 0) && (in_pos < IN_LENGTH);

    wire [7:0] feat_byte;
    wire [$clog2(IN_CHANNELS*IN_LENGTH*8)-1:0] bit_idx;
    assign bit_idx = (ic * IN_LENGTH + in_pos) * 8;
    assign feat_byte = in_bounds ? feature_in[bit_idx +: 8] : 8'd0;
    assign x_val = $signed(feat_byte);

    assign product = w_val * x_val;

    // Saturation to INT8
    function [7:0] saturate_int8;
        input signed [23:0] val;
        begin
            if (val > 24'sd127)
                saturate_int8 = 8'sd127;
            else if (val < -24'sd128)
                saturate_int8 = -8'sd128;
            else
                saturate_int8 = val[7:0];
        end
    endfunction

    // Weight address calculation
    wire [15:0] w_addr_calc;
    assign w_addr_calc = oc * IN_CHANNELS * KERNEL_SIZE + ic * KERNEL_SIZE + kk;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            out_valid   <= 1'b0;
            oc <= 0; op <= 0; ic <= 0; kk <= 0;
            acc         <= 24'd0;
            weight_addr <= 16'd0;
            bias_addr   <= 16'd0;
            feature_out <= {(OUT_CHANNELS*OUT_LENGTH*8){1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done      <= 1'b0;
                    out_valid <= 1'b0;
                    if (start) begin
                        state <= S_COMPUTE;
                        oc <= 0; op <= 0; ic <= 0; kk <= 0;
                        acc <= 24'd0;
                        weight_addr <= 16'd0;
                    end
                end

                S_COMPUTE: begin
                    // MAC: acc += weight * input
                    weight_addr <= w_addr_calc;
                    acc <= acc + {{8{product[15]}}, product};  // sign-extend to 24-bit

                    // Advance kernel tap
                    if (kk < KERNEL_SIZE - 1) begin
                        kk <= kk + 1;
                    end else begin
                        kk <= 0;
                        if (ic < IN_CHANNELS - 1) begin
                            ic <= ic + 1;
                        end else begin
                            ic <= 0;
                            // All input channels done for this (oc, op)
                            if (BIAS_EN) begin
                                state <= S_BIAS;
                                bias_addr <= oc;
                            end else begin
                                // Store saturated result
                                feature_out[(oc*OUT_LENGTH+op)*8 +: 8] <= saturate_int8(acc);
                                acc <= 24'd0;
                                // Advance output position
                                if (op < OUT_LENGTH - 1) begin
                                    op <= op + 1;
                                end else begin
                                    op <= 0;
                                    if (oc < OUT_CHANNELS - 1) begin
                                        oc <= oc + 1;
                                    end else begin
                                        state <= S_DONE;
                                    end
                                end
                            end
                        end
                    end
                end

                S_BIAS: begin
                    // Add bias and store
                    feature_out[(oc*OUT_LENGTH+op)*8 +: 8] <=
                        saturate_int8(acc + {{16{bias_data[7]}}, bias_data});
                    acc <= 24'd0;
                    // Advance output position
                    if (op < OUT_LENGTH - 1) begin
                        op <= op + 1;
                        state <= S_COMPUTE;
                    end else begin
                        op <= 0;
                        if (oc < OUT_CHANNELS - 1) begin
                            oc <= oc + 1;
                            state <= S_COMPUTE;
                        end else begin
                            state <= S_DONE;
                        end
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
