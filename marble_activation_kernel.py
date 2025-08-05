import torch
from functools import lru_cache
from torch.utils.cpp_extension import load_inline

__all__ = ["marble_activation"]


@lru_cache(maxsize=None)
def _load_kernel():
    """Compile and return the MARBLE activation CUDA extension.

    Returns ``None`` when CUDA is unavailable so that CPU execution is still
    possible without raising an error.
    """
    if not torch.cuda.is_available():
        return None
    return load_inline(
        name="marble_activation_cuda",
        cpp_sources="""
#include <torch/extension.h>
#include <algorithm>

void marble_activation_launcher(const at::Tensor& x, at::Tensor& y,
                                float threshold, float a, float b, float c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marble_activation_launcher", &marble_activation_launcher,
          "MARBLE custom activation forward launcher");
}
""",
        cuda_sources="""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Optimized MARBLE activation kernel leveraging vectorized loads,
// grid-stride loops and branchless computation for maximum throughput.
__global__ void marble_activation_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         float threshold, float a, float b, float c,
                                         long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    long n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (long i = idx; i < n4; i += stride) {
        float4 xv = x4[i];
        float4 out;

        float m = xv.x > threshold;
        out.x = __fmaf_rn(xv.x, m * a + (1.f - m) * c, m * b);
        m = xv.y > threshold;
        out.y = __fmaf_rn(xv.y, m * a + (1.f - m) * c, m * b);
        m = xv.z > threshold;
        out.z = __fmaf_rn(xv.z, m * a + (1.f - m) * c, m * b);
        m = xv.w > threshold;
        out.w = __fmaf_rn(xv.w, m * a + (1.f - m) * c, m * b);

        y4[i] = out;
    }

    for (long i = n4 * 4 + idx; i < n; i += stride) {
        float v = x[i];
        float m = v > threshold;
        y[i] = __fmaf_rn(v, m * a + (1.f - m) * c, m * b);
    }
}

void marble_activation_launcher(const at::Tensor& x, at::Tensor& y,
                                float threshold, float a, float b, float c) {
    const long n = x.numel();
    const int threads = 256;
    const int blocks = std::min<int>((n + threads - 1) / threads, 65535);
    marble_activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                                  y.data_ptr<float>(),
                                                  threshold, a, b, c, n);
}
""",
        functions=["marble_activation_launcher"],
        verbose=False,
    )


def _marble_activation_forward(x: torch.Tensor, threshold: float, a: float, b: float, c: float) -> torch.Tensor:
    """Forward computation of the MARBLE activation.

    Uses a custom CUDA kernel when running on GPU, otherwise falls back to a
    pure PyTorch implementation. The operation applies a piecewise linear
    transformation tailored to MARBLE's neuromorphic processing style.
    """
    if x.device.type == "cuda":
        module = _load_kernel()
        if module is None:
            raise RuntimeError("CUDA not available for MARBLE activation kernel")
        y = torch.empty_like(x)
        module.marble_activation_launcher(x.contiguous(), y,
                                          float(threshold), float(a), float(b), float(c))
        return y
    else:
        return torch.where(x > threshold, a * x + b, c * x)


class _MarbleActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float, a: float, b: float, c: float):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.a = a
        ctx.c = c
        return _marble_activation_forward(x, threshold, a, b, c)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_x = torch.where(x > ctx.threshold,
                             torch.tensor(ctx.a, device=x.device, dtype=x.dtype),
                             torch.tensor(ctx.c, device=x.device, dtype=x.dtype))
        return grad_output * grad_x, None, None, None, None


def marble_activation(x: torch.Tensor, threshold: float, a: float, b: float, c: float) -> torch.Tensor:
    """Piecewise linear activation used across MARBLE components.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    threshold : float
        Boundary at which the activation switches behavior.
    a : float
        Slope above the threshold.
    b : float
        Intercept above the threshold.
    c : float
        Slope below the threshold.

    Returns
    -------
    torch.Tensor
        Activated tensor.
    """
    return _MarbleActivationFunction.apply(x, float(threshold), float(a), float(b), float(c))
