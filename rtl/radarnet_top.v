/*
 * radarnet_top.v — Top-Level Wrapper with FSM Controller
 * ========================================================
 * RadarNet-1D inference accelerator for INT8-quantised ResNet-1D.
 *
 * 8-State FSM:
 *   S0 IDLE      — Wait for start pulse
 *   S1 LOAD      — Shift in 128 I/Q pairs to input_buffer
 *   S2 CONV_IN   — Layer 1: Conv1D(2→16,k7) + Layer 2: Conv1D(16→32,k3,s2)
 *   S3 RES_BLK1  — Layers 3-4: Residual Block 1 (32→32)
 *   S4 RES_BLK2  — Layers 5-6: Residual Block 2 (32→64, with projection)
 *   S5 GAP       — Layer 7: Global Average Pooling (64ch × 32t → 64-vec)
 *   S6 FC_OUT    — Layer 8: FC(64→4) + argmax
 *   S7 DONE      — Assert done, output valid
 *
 * External interface:
 *   - Serial I/Q input (16-bit per clock: 8-bit I + 8-bit Q)
 *   - 2-bit class_id output + anomaly_flag + done signal
 *   - Weight ROM is internal (BRAM initialised from .hex)
 *
 * Target: 100 MHz on Xilinx 7-series (ZedBoard / Zybo / PYNQ)
 */

module radarnet_top #(
    parameter DATA_WIDTH   = 8,
    parameter NUM_SAMPLES  = 128,
    parameter NUM_CLASSES  = 4
)(
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire        start,         // pulse to begin inference
    output reg         done,          // asserted when result is valid
    output reg         busy,

    // Serial I/Q input (one sample per clock during LOAD state)
    input  wire        iq_valid,
    input  wire [15:0] iq_data,       // {I[7:0], Q[7:0]}

    // Classification output
    output wire [1:0]  class_id,
    output wire        anomaly_flag,
    output reg         result_valid,

    // Debug / status
    output reg  [2:0]  fsm_state,
    output reg  [31:0] cycle_count    // inference latency counter
);

    // ── FSM States ──────────────────────────────────────────────────
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;
    localparam S_CONV_IN  = 3'd2;
    localparam S_RES_BLK1 = 3'd3;
    localparam S_RES_BLK2 = 3'd4;
    localparam S_GAP      = 3'd5;
    localparam S_FC_OUT   = 3'd6;
    localparam S_DONE     = 3'd7;

    reg [2:0] state, next_state;

    // ── Internal wires ──────────────────────────────────────────────
    // Input buffer
    wire ib_full;
    wire [NUM_SAMPLES*2*DATA_WIDTH-1:0] ib_data;
    reg  ib_clear;

    // Weight ROM
    reg  [15:0] wrom_addr;
    wire [DATA_WIDTH-1:0] wrom_data;

    // Sub-module start/done signals
    reg  layer_start;
    wire layer_done;

    // Intermediate feature maps (simplified — flattened registers)
    // L1 output: 16 ch × 128 t = 16384 bits
    reg [16*128*DATA_WIDTH-1:0] feat_l1;
    // L2 output: 32 ch × 64 t = 16384 bits
    reg [32*64*DATA_WIDTH-1:0] feat_l2;
    // Block 1 output: 32 ch × 64 t
    reg [32*64*DATA_WIDTH-1:0] feat_b1;
    // Block 2 output: 64 ch × 32 t = 16384 bits
    reg [64*32*DATA_WIDTH-1:0] feat_b2;
    // GAP output: 64 × 8 = 512 bits
    reg [64*DATA_WIDTH-1:0] feat_gap;
    // FC output: 4 × 8 = 32 bits
    reg [4*DATA_WIDTH-1:0] feat_fc;

    // Sub-state counter for multi-cycle operations
    reg [15:0] sub_counter;
    reg sub_phase;   // 0 = first conv, 1 = second conv within a state

    // ── Input Buffer Instance ───────────────────────────────────────
    input_buffer #(
        .NUM_SAMPLES(NUM_SAMPLES),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_input_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(iq_valid && (state == S_LOAD)),
        .clear(ib_clear),
        .din(iq_data),
        .full(ib_full),
        .data_out(ib_data)
    );

    // ── Weight ROM Instance ─────────────────────────────────────────
    weight_rom #(
        .DATA_WIDTH(DATA_WIDTH),
        .TOTAL_DEPTH(32768),
        .INIT_FILE("weights_all.hex")
    ) u_weight_rom (
        .clk(clk),
        .addr(wrom_addr[14:0]),
        .data_out(wrom_data)
    );

    // ── Argmax Instance ─────────────────────────────────────────────
    argmax_out u_argmax (
        .logit_0($signed(feat_fc[0*DATA_WIDTH +: DATA_WIDTH])),
        .logit_1($signed(feat_fc[1*DATA_WIDTH +: DATA_WIDTH])),
        .logit_2($signed(feat_fc[2*DATA_WIDTH +: DATA_WIDTH])),
        .logit_3($signed(feat_fc[3*DATA_WIDTH +: DATA_WIDTH])),
        .valid_in(result_valid),
        .class_id(class_id),
        .anomaly_flag(anomaly_flag),
        .valid_out()
    );

    // ── FSM ─────────────────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            busy         <= 1'b0;
            result_valid <= 1'b0;
            fsm_state    <= 3'd0;
            cycle_count  <= 32'd0;
            ib_clear     <= 1'b0;
            wrom_addr    <= 16'd0;
            layer_start  <= 1'b0;
            sub_counter  <= 16'd0;
            sub_phase    <= 1'b0;
            feat_l1 <= 0; feat_l2 <= 0;
            feat_b1 <= 0; feat_b2 <= 0;
            feat_gap <= 0; feat_fc <= 0;
        end else begin
            fsm_state <= state;

            case (state)
                // ── S0: IDLE ────────────────────────────────────────
                S_IDLE: begin
                    done         <= 1'b0;
                    result_valid <= 1'b0;
                    if (start) begin
                        state       <= S_LOAD;
                        busy        <= 1'b1;
                        ib_clear    <= 1'b1;
                        cycle_count <= 32'd0;
                    end
                end

                // ── S1: LOAD ────────────────────────────────────────
                S_LOAD: begin
                    ib_clear    <= 1'b0;
                    cycle_count <= cycle_count + 1;
                    if (ib_full) begin
                        state       <= S_CONV_IN;
                        sub_counter <= 16'd0;
                        sub_phase   <= 1'b0;
                    end
                end

                // ── S2: CONV_IN (Layers 1+2) ────────────────────────
                S_CONV_IN: begin
                    cycle_count <= cycle_count + 1;
                    // Simulate computation time for two conv layers
                    // L1: Conv1D(2→16, k=7, s=1) on 128 samples
                    //     MACs = 16 × 2 × 7 × 128 = 28,672
                    // L2: Conv1D(16→32, k=3, s=2) on 128 samples
                    //     MACs = 32 × 16 × 3 × 64 = 98,304
                    sub_counter <= sub_counter + 1;
                    // Placeholder: advance after sufficient cycles
                    if (sub_counter >= 16'd500) begin
                        state       <= S_RES_BLK1;
                        sub_counter <= 16'd0;
                    end
                end

                // ── S3: RES_BLK1 (Layers 3-4) ──────────────────────
                S_RES_BLK1: begin
                    cycle_count <= cycle_count + 1;
                    sub_counter <= sub_counter + 1;
                    // Block 1: 2 × Conv1D(32→32, k=3) × 64 + skip add
                    if (sub_counter >= 16'd400) begin
                        state       <= S_RES_BLK2;
                        sub_counter <= 16'd0;
                    end
                end

                // ── S4: RES_BLK2 (Layers 5-6) ──────────────────────
                S_RES_BLK2: begin
                    cycle_count <= cycle_count + 1;
                    sub_counter <= sub_counter + 1;
                    // Block 2: 2 × Conv1D(32→64, k=3, s=2) + proj + skip
                    if (sub_counter >= 16'd500) begin
                        state       <= S_GAP;
                        sub_counter <= 16'd0;
                    end
                end

                // ── S5: GAP (Layer 7) ───────────────────────────────
                S_GAP: begin
                    cycle_count <= cycle_count + 1;
                    sub_counter <= sub_counter + 1;
                    // GAP: 64 channels × 32 accumulations
                    if (sub_counter >= 16'd100) begin
                        state       <= S_FC_OUT;
                        sub_counter <= 16'd0;
                    end
                end

                // ── S6: FC_OUT (Layer 8) + Argmax ───────────────────
                S_FC_OUT: begin
                    cycle_count <= cycle_count + 1;
                    sub_counter <= sub_counter + 1;
                    // FC: 64 × 4 = 256 MACs
                    if (sub_counter >= 16'd64) begin
                        state       <= S_DONE;
                        sub_counter <= 16'd0;
                    end
                end

                // ── S7: DONE ────────────────────────────────────────
                S_DONE: begin
                    done         <= 1'b1;
                    busy         <= 1'b0;
                    result_valid <= 1'b1;
                    // Return to IDLE on next start or automatically
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
