/*
 * argmax_out.v — Argmax of 4 Logits with Anomaly Flag
 * =====================================================
 * Compares four INT8 logit values and outputs:
 *   - class_id   [1:0] : index of the maximum logit (0-3)
 *   - anomaly_flag     : 1 if class_id != 0 (i.e., not "Normal")
 *
 * Combinational logic, single-cycle evaluation.
 */

module argmax_out (
    input  wire signed [7:0] logit_0,   // Normal
    input  wire signed [7:0] logit_1,   // Jammer
    input  wire signed [7:0] logit_2,   // Spoofer
    input  wire signed [7:0] logit_3,   // Interference
    input  wire              valid_in,
    output reg  [1:0]        class_id,
    output wire              anomaly_flag,
    output reg               valid_out
);

    reg signed [7:0] max_val;

    always @(*) begin
        // Default: class 0
        max_val  = logit_0;
        class_id = 2'd0;

        if (logit_1 > max_val) begin
            max_val  = logit_1;
            class_id = 2'd1;
        end
        if (logit_2 > max_val) begin
            max_val  = logit_2;
            class_id = 2'd2;
        end
        if (logit_3 > max_val) begin
            max_val  = logit_3;
            class_id = 2'd3;
        end

        valid_out = valid_in;
    end

    // Anomaly flag: 1 when class is NOT Normal (class 0)
    assign anomaly_flag = (class_id != 2'd0);

endmodule
