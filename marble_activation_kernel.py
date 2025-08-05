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
#include <cuda_fp16.h>
#include <torch/extension.h>

// High-performance MARBLE activation kernels for float32 and float16.
// The float kernel uses vectorized float4 operations while the half kernel
// leverages half2 intrinsics for maximum throughput.

__global__ void marble_activation_kernel_float(const float* __restrict__ x,
                                               float* __restrict__ y,
                                               float threshold, float a,
                                               float b, float c, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    float diff = a - c;
    long n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (long i = idx; i < n4; i += stride) {
        float4 xv = x4[i];
        float4 out;

        float m = xv.x > threshold;
        out.x = __fmaf_rn(xv.x, m * diff + c, m * b);
        m = xv.y > threshold;
        out.y = __fmaf_rn(xv.y, m * diff + c, m * b);
        m = xv.z > threshold;
        out.z = __fmaf_rn(xv.z, m * diff + c, m * b);
        m = xv.w > threshold;
        out.w = __fmaf_rn(xv.w, m * diff + c, m * b);

        y4[i] = out;
    }

    for (long i = n4 * 4 + idx; i < n; i += stride) {
        float v = x[i];
        float m = v > threshold;
        y[i] = __fmaf_rn(v, m * diff + c, m * b);
    }
}

__global__ void marble_activation_kernel_half(const __half* __restrict__ x,
                                              __half* __restrict__ y,
                                              __half threshold, __half a,
                                              __half b, __half c, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    __half diff = __hsub(a, c);
    long n2 = n / 2;
    const __half2* x2 = reinterpret_cast<const __half2*>(x);
    __half2* y2 = reinterpret_cast<__half2*>(y);

    __half2 thresh2 = __half2half2(threshold);
    __half2 diff2 = __half2half2(diff);
    __half2 c2 = __half2half2(c);
    __half2 b2 = __half2half2(b);

    for (long i = idx; i < n2; i += stride) {
        __half2 xv = x2[i];
        __half2 m = __hgt2(xv, thresh2);
        __half2 t = __hfma2(diff2, m, c2);
        y2[i] = __hfma2(xv, t, __hmul2(m, b2));
    }

    for (long i = n2 * 2 + idx; i < n; i += stride) {
        __half xv = x[i];
        __half m = __hgt(xv, threshold);
        __half t = __hfma(diff, m, c);
        y[i] = __hfma(xv, t, __hmul(m, b));
    }
}

void marble_activation_launcher(const at::Tensor& x, at::Tensor& y,
                                float threshold, float a, float b, float c) {
    const long n = x.numel();
    const int threads = 256;
    const int blocks = std::min<int>((n + threads - 1) / threads, 65535);

    if (x.scalar_type() == at::kFloat) {
        marble_activation_kernel_float<<<blocks, threads>>>(x.data_ptr<float>(),
                                                           y.data_ptr<float>(),
                                                           threshold, a, b, c, n);
    } else if (x.scalar_type() == at::kHalf) {
        __half th = __float2half(threshold);
        __half ah = __float2half(a);
        __half bh = __float2half(b);
        __half ch = __float2half(c);
        marble_activation_kernel_half<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            th, ah, bh, ch, n);
    } else {
        AT_ERROR("marble_activation: unsupported tensor dtype");
    }
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
