module tanh_shared_N #(
    parameter N = 12
) (
    input clk,
    input rst,
    input start,
    input  [8*N-1:0] b_in,
    output reg [8*N-1:0] z_out,
    output reg done
);

    wire [7:0] b [0:N-1];
    wire [3:0] seg [0:N-1];
    wire [7:0] core_z [0:N-1];
    
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : unflatten
            assign b[i] = b_in[i*8 +: 8];
        end
    endgenerate

    reg [$clog2(N+1)-1:0] state;
    
    reg [3:0] lut_seg_in;
    wire [31:0] lut_m_out;
    wire [31:0] lut_c_out;
    
    tanh_lut_ext lut_inst (
        .seg(lut_seg_in),
        .m_lut(lut_m_out),
        .c_lut(lut_c_out)
    );

    always @(*) begin
        if (state < N) begin
            lut_seg_in = seg[state];
        end else begin
            lut_seg_in = 4'd0;
        end
    end

    generate
        for (i = 0; i < N; i = i + 1) begin : cores
            tanh_core_ext core_inst (
                .b(b[i]),
                .m_lut(lut_m_out),
                .c_lut(lut_c_out),
                .seg(seg[i]),
                .z(core_z[i])
            );
        end
    endgenerate

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= N;
            done <= 0;
            z_out <= 0;
        end else begin
            if (start) begin
                state <= 0;
                done <= 0;
            end else if (state < N) begin
                z_out[state*8 +: 8] <= core_z[state];
                if (state == N - 1) begin
                    done <= 1;
                    state <= N;
                end else begin
                    state <= state + 1;
                end
            end else begin
                done <= 0;
            end
        end
    end
endmodule
