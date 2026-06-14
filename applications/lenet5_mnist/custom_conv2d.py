import torch
import cupy as cp
from custom_mul import CUSTOM_MUL_CODE

conv_code = CUSTOM_MUL_CODE + r'''
extern "C" __global__
void conv2d_kernel(const float* x, const float* weight, const float* bias, float* out,
                   int N, int Cin, int Cout, int H, int W, int Kh, int Kw, 
                   int Hout, int Wout, int stride, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int w_o = idx % Wout, h_o = (idx / Wout) % Hout;
    int co = (idx / (Wout * Hout)) % Cout, n = idx / (Wout * Hout * Cout);

    float acc = bias[co];
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                int cur_h = h_o * stride + kh, cur_w = w_o * stride + kw;
                float val = x[((n * Cin + ci) * H + cur_h) * W + cur_w];
                float w_val = weight[((co * Cin + ci) * Kh + kh) * Kw + kw];
                acc += custom_mul_approx_fp32(val, w_val, level);
            }
        }
    }
    out[idx] = acc;
}
'''
conv_kernel = cp.RawKernel(conv_code, 'conv2d_kernel')

def custom_conv2d(x, weight, bias=None, stride=1, level=0):
    N, Cin, H, W = x.shape
    Cout, _, Kh, Kw = weight.shape
    Hout, Wout = (H - Kh) // stride + 1, (W - Kw) // stride + 1
    if bias is None: bias = torch.zeros(Cout, device=x.device)
    
    cp_x = cp.from_dlpack(x.contiguous())
    cp_w = cp.from_dlpack(weight.contiguous())
    cp_b = cp.from_dlpack(bias.contiguous())
    cp_out = cp.empty((N, Cout, Hout, Wout), dtype=cp.float32)

    total = N * Cout * Hout * Wout
    grid = (total + 255) // 256
    conv_kernel((grid,), (256,), (cp_x, cp_w, cp_b, cp_out, N, Cin, Cout, H, W, 
                                  Kh, Kw, Hout, Wout, stride, level))
    return torch.from_dlpack(cp_out)
