/*
 * fc_layer.v — Fully Connected Layer (64 → 4, INT8)
 * ====================================================
 * Computes Y = W·X + b where:
 *   X is a 64-element INT8 vector (from GAP)
 *   W is a 4×64 INT8 weight matrix (256 MACs total)
 *   b is a 4-element INT8 bias vector
 *   Y is a 4-element output (logits, kept in wider precision for argmax)
 *
 * The 256 MACs are computed sequentially per output neuron.
 * FSM: IDLE -> COMPUTE -> BIAS -> NEXT/DONE
 */

module fc_layer #(
    parameter IN_FEATURES  = 64,
    parameter OUT_FEATURES = 4,
    parameter DATA_WIDTH   = 8
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input vector: 64 x 8-bit
    input  wire [IN_FEATURES*DATA_WIDTH-1:0] fc_in,

    // Weight ROM interface
    output reg  [15:0] weight_addr,
    input  wire [DATA_WIDTH-1:0] weight_data,

    // Bias ROM interface
    output reg  [15:0] bias_addr,
    input  wire [DATA_WIDTH-1:0] bias_data,

    // Output: 4 logits (kept as signed 8-bit for argmax)
    output reg  [OUT_FEATURES*DATA_WIDTH-1:0] fc_out,
    output reg         out_valid
);

    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_BIAS    = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;
    reg [$clog2(OUT_FEATURES):0] neuron;   // output neuron index
    reg [$clog2(IN_FEATURES):0]  feat;     // input feature index

    // 24-bit accumulator
    reg signed [23:0] acc;

    // Weight and input extraction
    wire signed [DATA_WIDTH-1:0] w_val;
    wire signed [DATA_WIDTH-1:0] x_val;
    wire signed [15:0] product;

    assign w_val = $signed(weight_data);
    assign x_val = $signed(fc_in[feat*DATA_WIDTH +: DATA_WIDTH]);
    assign product = w_val * x_val;

    // Saturation
    function [DATA_WIDTH-1:0] saturate;
        input signed [23:0] val;
        begin
            if (val > 24'sd127)
                saturate = 8'sd127;
            else if (val < -24'sd128)
                saturate = -8'sd128;
            else
                saturate = val[DATA_WIDTH-1:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            out_valid   <= 1'b0;
            neuron <= 0; feat <= 0;
            acc         <= 24'd0;
            weight_addr <= 16'd0;
            bias_addr   <= 16'd0;
            fc_out      <= {(OUT_FEATURES*DATA_WIDTH){1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done      <= 1'b0;
                    out_valid <= 1'b0;
                    if (start) begin
                        state  <= S_COMPUTE;
                        neuron <= 0;
                        feat   <= 0;
                        acc    <= 24'd0;
                        weight_addr <= 16'd0;
                    end
                end

                S_COMPUTE: begin
                    // MAC: acc += W[neuron][feat] * X[feat]
                    weight_addr <= neuron * IN_FEATURES + feat;
                    acc <= acc + {{8{product[15]}}, product};

                    if (feat < IN_FEATURES - 1) begin
                        feat <= feat + 1;
                    end else begin
                        // Done with this neuron's dot product
                        feat  <= 0;
                        state <= S_BIAS;
                        bias_addr <= neuron;
                    end
                end

                S_BIAS: begin
                    // Add bias and store
                    fc_out[neuron*DATA_WIDTH +: DATA_WIDTH] <=
                        saturate(acc + {{16{bias_data[DATA_WIDTH-1]}}, bias_data});
                    acc <= 24'd0;
                    if (neuron < OUT_FEATURES - 1) begin
                        neuron <= neuron + 1;
                        state  <= S_COMPUTE;
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
