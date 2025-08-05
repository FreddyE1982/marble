from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline

__all__ = ["apply_weight_updates"]


@lru_cache(maxsize=None)
def _load_kernel():
    if not torch.cuda.is_available():
        return None
    return load_inline(
        name="neuronenblitz_cuda",
        cpp_sources="""
#include <torch/extension.h>
void nb_apply_launcher(const at::Tensor& source,
                       at::Tensor& weights,
                       at::Tensor& potentials,
                       at::Tensor& momentum,
                       at::Tensor& grad_sq,
                       at::Tensor& prev_grad,
                       const at::Tensor& eligibility,
                       const at::Tensor& mem_gate,
                       at::Tensor& scores,
                       float error,
                       float learning_rate,
                       float momentum_coeff,
                       float rms_beta,
                       float grad_epsilon,
                       float cap,
                       float weight_limit,
                       float gradient_score_scale,
                       float synapse_potential_cap,
                       int path_len);
""",
        cuda_sources="""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// Kernel for float32
__global__ void nb_apply_kernel_float(const float* __restrict__ source,
                                      float* __restrict__ weights,
                                      float* __restrict__ potentials,
                                      float* __restrict__ momentum,
                                      float* __restrict__ grad_sq,
                                      float* __restrict__ prev_grad,
                                      const float* __restrict__ eligibility,
                                      const float* __restrict__ mem_gate,
                                      float* __restrict__ scores,
                                      float error,
                                      float learning_rate,
                                      float momentum_coeff,
                                      float rms_beta,
                                      float grad_epsilon,
                                      float cap,
                                      float weight_limit,
                                      float gradient_score_scale,
                                      float synapse_potential_cap,
                                      int path_len,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float src = source[i];
        float delta = (error * src) / (path_len + 1.0f);
        float prev = prev_grad[i];
        if (prev * delta < 0.f) delta *= 0.5f;
        prev_grad[i] = delta;
        delta *= eligibility[i];
        float prev_v = grad_sq[i];
        float v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        grad_sq[i] = v;
        float scaled_delta = delta / sqrtf(v + grad_epsilon);
        float mom_prev = momentum[i];
        float mom = momentum_coeff * mom_prev + scaled_delta;
        momentum[i] = mom;
        float update = learning_rate * (momentum_coeff * mom + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        float w = weights[i] + update;
        w = fminf(weight_limit, fmaxf(-weight_limit, w));
        weights[i] = w;
        float pot = potentials[i] + fabsf(scaled_delta) * gradient_score_scale;
        potentials[i] = fminf(synapse_potential_cap, pot);
        float mem_factor = 1.0f + mem_gate[i];
        scores[i] = fabsf(error) * fabsf(w) / (float)path_len * mem_factor;
    }
}

// Kernel for float16 using float math internally
__global__ void nb_apply_kernel_half(const __half* __restrict__ source,
                                     __half* __restrict__ weights,
                                     __half* __restrict__ potentials,
                                     __half* __restrict__ momentum,
                                     __half* __restrict__ grad_sq,
                                     __half* __restrict__ prev_grad,
                                     const __half* __restrict__ eligibility,
                                     const __half* __restrict__ mem_gate,
                                     __half* __restrict__ scores,
                                     float error,
                                     float learning_rate,
                                     float momentum_coeff,
                                     float rms_beta,
                                     float grad_epsilon,
                                     float cap,
                                     float weight_limit,
                                     float gradient_score_scale,
                                     float synapse_potential_cap,
                                     int path_len,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float src = __half2float(source[i]);
        float delta = (error * src) / (path_len + 1.0f);
        float prev = __half2float(prev_grad[i]);
        if (prev * delta < 0.f) delta *= 0.5f;
        prev_grad[i] = __float2half(delta);
        delta *= __half2float(eligibility[i]);
        float prev_v = __half2float(grad_sq[i]);
        float v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        grad_sq[i] = __float2half(v);
        float scaled_delta = delta / sqrtf(v + grad_epsilon);
        float mom_prev = __half2float(momentum[i]);
        float mom = momentum_coeff * mom_prev + scaled_delta;
        momentum[i] = __float2half(mom);
        float update = learning_rate * (momentum_coeff * mom + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        float w = __half2float(weights[i]) + update;
        w = fminf(weight_limit, fmaxf(-weight_limit, w));
        weights[i] = __float2half(w);
        float pot = __half2float(potentials[i]) + fabsf(scaled_delta) * gradient_score_scale;
        potentials[i] = __float2half(fminf(synapse_potential_cap, pot));
        float mem_factor = 1.0f + __half2float(mem_gate[i]);
        scores[i] = __float2half(fabsf(error) * fabsf(w) / (float)path_len * mem_factor);
    }
}

void nb_apply_launcher(const at::Tensor& source,
                       at::Tensor& weights,
                       at::Tensor& potentials,
                       at::Tensor& momentum,
                       at::Tensor& grad_sq,
                       at::Tensor& prev_grad,
                       const at::Tensor& eligibility,
                       const at::Tensor& mem_gate,
                       at::Tensor& scores,
                       float error,
                       float learning_rate,
                       float momentum_coeff,
                       float rms_beta,
                       float grad_epsilon,
                       float cap,
                       float weight_limit,
                       float gradient_score_scale,
                       float synapse_potential_cap,
                       int path_len) {
    int n = weights.numel();
    int threads = 256;
    int blocks = std::min((n + threads - 1) / threads, 65535);
    if (weights.scalar_type() == at::kFloat) {
        nb_apply_kernel_float<<<blocks, threads>>>(
            source.data_ptr<float>(),
            weights.data_ptr<float>(),
            potentials.data_ptr<float>(),
            momentum.data_ptr<float>(),
            grad_sq.data_ptr<float>(),
            prev_grad.data_ptr<float>(),
            eligibility.data_ptr<float>(),
            mem_gate.data_ptr<float>(),
            scores.data_ptr<float>(),
            error, learning_rate, momentum_coeff, rms_beta,
            grad_epsilon, cap, weight_limit, gradient_score_scale,
            synapse_potential_cap, path_len, n);
    } else if (weights.scalar_type() == at::kHalf) {
        nb_apply_kernel_half<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(source.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(weights.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(potentials.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(momentum.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_sq.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(prev_grad.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(eligibility.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(mem_gate.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
            error, learning_rate, momentum_coeff, rms_beta,
            grad_epsilon, cap, weight_limit, gradient_score_scale,
            synapse_potential_cap, path_len, n);
    } else {
        AT_ERROR("nb_apply_launcher: unsupported tensor dtype");
    }
}
""",
        functions=["nb_apply_launcher"],
        verbose=False,
    )


def apply_weight_updates(source: torch.Tensor,
                         weights: torch.Tensor,
                         potentials: torch.Tensor,
                         momentum: torch.Tensor,
                         grad_sq: torch.Tensor,
                         prev_grad: torch.Tensor,
                         eligibility: torch.Tensor,
                         mem_gate: torch.Tensor,
                         error: float,
                         learning_rate: float,
                         momentum_coeff: float,
                         rms_beta: float,
                         grad_epsilon: float,
                         cap: float,
                         weight_limit: float,
                         gradient_score_scale: float,
                         synapse_potential_cap: float,
                         path_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_kernel()
    if module is None:
        raise RuntimeError("CUDA not available for Neuronenblitz kernel")
    scores = torch.empty_like(weights)
    module.nb_apply_launcher(source.contiguous(),
                             weights, potentials, momentum, grad_sq, prev_grad,
                             eligibility, mem_gate, scores,
                             float(error), float(learning_rate), float(momentum_coeff),
                             float(rms_beta), float(grad_epsilon), float(cap),
                             float(weight_limit), float(gradient_score_scale),
                             float(synapse_potential_cap), int(path_len))
    return weights, potentials, momentum, grad_sq, prev_grad, scores
