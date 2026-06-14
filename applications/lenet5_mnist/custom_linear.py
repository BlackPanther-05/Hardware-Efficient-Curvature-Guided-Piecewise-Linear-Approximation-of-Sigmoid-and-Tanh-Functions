import torch
import cupy as cp
from custom_mul import CUSTOM_MUL_CODE

linear_code = CUSTOM_MUL_CODE + r'''
extern "C" __global__
void linear_kernel(const float* x, const float* w, const float* b, float* out, 
                   int N, int InF, int OutF, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OutF;
    if (idx >= total) return;

    int o = idx % OutF, n = idx / OutF;
    float acc = b[o];
    for (int i = 0; i < InF; ++i) {
        acc += custom_mul_approx_fp32(x[n * InF + i], w[o * InF + i], level);
    }
    out[idx] = acc;
}
'''
linear_kernel = cp.RawKernel(linear_code, 'linear_kernel')

def custom_linear(x, weight, bias=None, level=0):
    N, InF = x.shape
    OutF = weight.shape[0]
    if bias is None: bias = torch.zeros(OutF, device=x.device)
    
    cp_x = cp.from_dlpack(x.contiguous())
    cp_w = cp.from_dlpack(weight.contiguous())
    cp_b = cp.from_dlpack(bias.contiguous())
    cp_out = cp.empty((N, OutF), dtype=cp.float32)

    grid = (N * OutF + 255) // 256
    linear_kernel((grid,), (256,), (cp_x, cp_w, cp_b, cp_out, N, InF, OutF, level))
    return torch.from_dlpack(cp_out)
