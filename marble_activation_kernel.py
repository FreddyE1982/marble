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

__global__ void marble_activation_kernel(const float* x, float* y,
                                         float threshold, float a, float b, float c,
                                         long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = v > threshold ? a * v + b : c * v;
    }
}

void marble_activation_launcher(const at::Tensor& x, at::Tensor& y,
                                float threshold, float a, float b, float c) {
    const long n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
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
