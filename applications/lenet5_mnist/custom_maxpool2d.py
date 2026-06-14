import torch
import cupy as cp

maxpool_code = r'''
extern "C" __global__
void maxpool_kernel(const float* x, float* out,
                    int N, int C, int H, int W,
                    int Hout, int Wout,
                    int k, int s) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * Hout * Wout) return;

    int w = idx % Wout;
    int h = (idx / Wout) % Hout;
    int c = (idx / (Wout * Hout)) % C;
    int n = idx / (Wout * Hout * C);

    float m = -1e20f;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            int hi = h * s + i;
            int wi = w * s + j;
            float v = x[((n * C + c) * H + hi) * W + wi];
            if (v > m) m = v;
        }

    out[idx] = m;
}
'''

maxpool_kernel = cp.RawKernel(maxpool_code, "maxpool_kernel")

def custom_maxpool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    Hout = (H - kernel_size) // stride + 1
    Wout = (W - kernel_size) // stride + 1

    cp_x = cp.from_dlpack(x.contiguous())
    cp_out = cp.empty((N, C, Hout, Wout), dtype=cp.float32)

    grid = (N * C * Hout * Wout + 255) // 256
    maxpool_kernel(
        (grid,), (256,),
        (cp_x, cp_out, N, C, H, W,
         Hout, Wout, kernel_size, stride)
    )

    return torch.from_dlpack(cp_out)

