/*
 * weight_rom.v — BRAM-Based Weight ROM for All Layers
 * =====================================================
 * Single ROM module initialised from .hex files via $readmemh.
 * Each layer has a contiguous address range (see address_map.txt).
 *
 * The ROM is single-port, synchronous read with 1-cycle latency.
 * Total depth is set by TOTAL_DEPTH parameter to accommodate all
 * ~29K INT8 weight values.
 *
 * For synthesis, Vivado will infer BRAM blocks automatically.
 */

module weight_rom #(
    parameter DATA_WIDTH  = 8,
    parameter TOTAL_DEPTH = 32768,   // enough for ~29K weights + biases
    parameter INIT_FILE   = "weights_all.hex"
)(
    input  wire                         clk,
    input  wire [$clog2(TOTAL_DEPTH)-1:0] addr,
    output reg  [DATA_WIDTH-1:0]        data_out
);

    // BRAM storage
    reg [DATA_WIDTH-1:0] mem [0:TOTAL_DEPTH-1];

    // Initialise from hex file
    initial begin
        $readmemh(INIT_FILE, mem);
    end

    // Synchronous read
    always @(posedge clk) begin
        data_out <= mem[addr];
    end

endmodule
