import torch
import cupy as cp

relu_kernel = cp.ElementwiseKernel(
    'float32 x',
    'float32 y',
    'y = x > 0 ? x : 0',
    'relu_kernel'
)

def custom_relu(x):
    cp_x = cp.from_dlpack(x.contiguous())
    cp_out = cp.empty_like(cp_x)
    relu_kernel(cp_x, cp_out)
    return torch.from_dlpack(cp_out)

