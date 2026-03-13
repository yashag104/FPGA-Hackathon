/*
 * proj_conv.v — 1×1 Projection Convolution (INT8)
 * =================================================
 * Used in Residual Block 2 skip path to match dimensions:
 *   Input:  32 channels × 64 time-steps
 *   Output: 64 channels × 32 time-steps (stride=2)
 *
 * 1×1 convolution = per-time-step matrix multiply (no kernel sliding).
 * For each output time-step t: out[oc, t] = sum_ic( W[oc, ic] * in[ic, 2t] ) + b[oc]
 */

module proj_conv #(
    parameter IN_CHANNELS  = 32,
    parameter OUT_CHANNELS = 64,
    parameter IN_LENGTH    = 64,
    parameter STRIDE       = 2,
    parameter DATA_WIDTH   = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input feature map
    input  wire [IN_CHANNELS*IN_LENGTH*DATA_WIDTH-1:0] feature_in,

    // Weight ROM interface (1x1 weights: OUT_CHANNELS × IN_CHANNELS)
    output reg  [15:0] weight_addr,
    input  wire [DATA_WIDTH-1:0] weight_data,

    // Bias ROM interface
    output reg  [15:0] bias_addr,
    input  wire [DATA_WIDTH-1:0] bias_data,

    // Output feature map
    output reg  [OUT_CHANNELS*(IN_LENGTH/STRIDE)*DATA_WIDTH-1:0] feature_out,
    output reg         out_valid
);

    localparam OUT_LENGTH = IN_LENGTH / STRIDE;

    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_BIAS    = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;
    reg [$clog2(OUT_CHANNELS):0] oc;
    reg [$clog2(OUT_LENGTH):0]   ot;   // output time index
    reg [$clog2(IN_CHANNELS):0]  ic;

    reg signed [23:0] acc;

    // Input sample at strided position
    wire [$clog2(IN_CHANNELS*IN_LENGTH*DATA_WIDTH)-1:0] in_bit_idx;
    assign in_bit_idx = (ic * IN_LENGTH + ot * STRIDE) * DATA_WIDTH;
    wire signed [DATA_WIDTH-1:0] x_val;
    assign x_val = $signed(feature_in[in_bit_idx +: DATA_WIDTH]);

    wire signed [DATA_WIDTH-1:0] w_val;
    assign w_val = $signed(weight_data);

    wire signed [15:0] product;
    assign product = w_val * x_val;

    function [DATA_WIDTH-1:0] saturate;
        input signed [23:0] val;
        begin
            if (val > 24'sd127)       saturate = 8'sd127;
            else if (val < -24'sd128) saturate = -8'sd128;
            else                      saturate = val[DATA_WIDTH-1:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0; out_valid <= 0;
            oc <= 0; ot <= 0; ic <= 0; acc <= 0;
            weight_addr <= 0; bias_addr <= 0;
            feature_out <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0; out_valid <= 0;
                    if (start) begin
                        state <= S_COMPUTE;
                        oc <= 0; ot <= 0; ic <= 0; acc <= 0;
                    end
                end
                S_COMPUTE: begin
                    weight_addr <= oc * IN_CHANNELS + ic;
                    acc <= acc + {{8{product[15]}}, product};
                    if (ic < IN_CHANNELS - 1) begin
                        ic <= ic + 1;
                    end else begin
                        ic <= 0;
                        state <= S_BIAS;
                        bias_addr <= oc;
                    end
                end
                S_BIAS: begin
                    feature_out[(oc*OUT_LENGTH+ot)*DATA_WIDTH +: DATA_WIDTH] <=
                        saturate(acc + {{16{bias_data[DATA_WIDTH-1]}}, bias_data});
                    acc <= 0;
                    if (ot < OUT_LENGTH - 1) begin
                        ot <= ot + 1; state <= S_COMPUTE;
                    end else begin
                        ot <= 0;
                        if (oc < OUT_CHANNELS - 1) begin
                            oc <= oc + 1; state <= S_COMPUTE;
                        end else
                            state <= S_DONE;
                    end
                end
                S_DONE: begin
                    done <= 1; out_valid <= 1; state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
