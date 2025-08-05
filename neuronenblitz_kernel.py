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

// Kernel for float32 leveraging vectorized float4 operations
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
    int n4 = n / 4;
    const float4* source4 = reinterpret_cast<const float4*>(source);
    float4* weights4 = reinterpret_cast<float4*>(weights);
    float4* potentials4 = reinterpret_cast<float4*>(potentials);
    float4* momentum4 = reinterpret_cast<float4*>(momentum);
    float4* grad_sq4 = reinterpret_cast<float4*>(grad_sq);
    float4* prev_grad4 = reinterpret_cast<float4*>(prev_grad);
    const float4* eligibility4 = reinterpret_cast<const float4*>(eligibility);
    const float4* mem_gate4 = reinterpret_cast<const float4*>(mem_gate);
    float4* scores4 = reinterpret_cast<float4*>(scores);
    for (int i = idx; i < n4; i += stride) {
        float4 src = source4[i];
        float4 w = weights4[i];
        float4 pot = potentials4[i];
        float4 mom = momentum4[i];
        float4 gs = grad_sq4[i];
        float4 pg = prev_grad4[i];
        float4 elig = eligibility4[i];
        float4 mem = mem_gate4[i];
        float4 sc;

        // Lane 0
        float delta = (error * src.x) / (path_len + 1.0f);
        float prev = pg.x;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.x = delta;
        delta *= elig.x;
        float prev_v = gs.x;
        float v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.x = v;
        float scaled_delta = delta / sqrtf(v + grad_epsilon);
        float mom_prev = mom.x;
        float m = momentum_coeff * mom_prev + scaled_delta;
        mom.x = m;
        float update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        float wv = w.x + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.x = wv;
        float potv = pot.x + fabsf(scaled_delta) * gradient_score_scale;
        pot.x = fminf(synapse_potential_cap, potv);
        float mem_factor = 1.0f + mem.x;
        sc.x = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        // Lane 1
        delta = (error * src.y) / (path_len + 1.0f);
        prev = pg.y;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.y = delta;
        delta *= elig.y;
        prev_v = gs.y;
        v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.y = v;
        scaled_delta = delta / sqrtf(v + grad_epsilon);
        mom_prev = mom.y;
        m = momentum_coeff * mom_prev + scaled_delta;
        mom.y = m;
        update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        wv = w.y + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.y = wv;
        potv = pot.y + fabsf(scaled_delta) * gradient_score_scale;
        pot.y = fminf(synapse_potential_cap, potv);
        mem_factor = 1.0f + mem.y;
        sc.y = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        // Lane 2
        delta = (error * src.z) / (path_len + 1.0f);
        prev = pg.z;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.z = delta;
        delta *= elig.z;
        prev_v = gs.z;
        v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.z = v;
        scaled_delta = delta / sqrtf(v + grad_epsilon);
        mom_prev = mom.z;
        m = momentum_coeff * mom_prev + scaled_delta;
        mom.z = m;
        update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        wv = w.z + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.z = wv;
        potv = pot.z + fabsf(scaled_delta) * gradient_score_scale;
        pot.z = fminf(synapse_potential_cap, potv);
        mem_factor = 1.0f + mem.z;
        sc.z = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        // Lane 3
        delta = (error * src.w) / (path_len + 1.0f);
        prev = pg.w;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.w = delta;
        delta *= elig.w;
        prev_v = gs.w;
        v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.w = v;
        scaled_delta = delta / sqrtf(v + grad_epsilon);
        mom_prev = mom.w;
        m = momentum_coeff * mom_prev + scaled_delta;
        mom.w = m;
        update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        wv = w.w + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.w = wv;
        potv = pot.w + fabsf(scaled_delta) * gradient_score_scale;
        pot.w = fminf(synapse_potential_cap, potv);
        mem_factor = 1.0f + mem.w;
        sc.w = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        weights4[i] = w;
        potentials4[i] = pot;
        momentum4[i] = mom;
        grad_sq4[i] = gs;
        prev_grad4[i] = pg;
        scores4[i] = sc;
    }
    for (int i = n4 * 4 + idx; i < n; i += stride) {
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

// Kernel for float16 using vectorized half2 operations
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
    int n2 = n / 2;
    const __half2* source2 = reinterpret_cast<const __half2*>(source);
    __half2* weights2 = reinterpret_cast<__half2*>(weights);
    __half2* potentials2 = reinterpret_cast<__half2*>(potentials);
    __half2* momentum2 = reinterpret_cast<__half2*>(momentum);
    __half2* grad_sq2 = reinterpret_cast<__half2*>(grad_sq);
    __half2* prev_grad2 = reinterpret_cast<__half2*>(prev_grad);
    const __half2* eligibility2 = reinterpret_cast<const __half2*>(eligibility);
    const __half2* mem_gate2 = reinterpret_cast<const __half2*>(mem_gate);
    __half2* scores2 = reinterpret_cast<__half2*>(scores);
    for (int i = idx; i < n2; i += stride) {
        float2 src = __half22float2(source2[i]);
        float2 w = __half22float2(weights2[i]);
        float2 pot = __half22float2(potentials2[i]);
        float2 mom = __half22float2(momentum2[i]);
        float2 gs = __half22float2(grad_sq2[i]);
        float2 pg = __half22float2(prev_grad2[i]);
        float2 elig = __half22float2(eligibility2[i]);
        float2 mem = __half22float2(mem_gate2[i]);
        float2 sc;

        // Lane 0
        float delta = (error * src.x) / (path_len + 1.0f);
        float prev = pg.x;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.x = delta;
        delta *= elig.x;
        float prev_v = gs.x;
        float v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.x = v;
        float scaled_delta = delta / sqrtf(v + grad_epsilon);
        float mom_prev = mom.x;
        float m = momentum_coeff * mom_prev + scaled_delta;
        mom.x = m;
        float update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        float wv = w.x + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.x = wv;
        float potv = pot.x + fabsf(scaled_delta) * gradient_score_scale;
        pot.x = fminf(synapse_potential_cap, potv);
        float mem_factor = 1.0f + mem.x;
        sc.x = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        // Lane 1
        delta = (error * src.y) / (path_len + 1.0f);
        prev = pg.y;
        if (prev * delta < 0.f) delta *= 0.5f;
        pg.y = delta;
        delta *= elig.y;
        prev_v = gs.y;
        v = rms_beta * prev_v + (1.f - rms_beta) * (delta * delta);
        gs.y = v;
        scaled_delta = delta / sqrtf(v + grad_epsilon);
        mom_prev = mom.y;
        m = momentum_coeff * mom_prev + scaled_delta;
        mom.y = m;
        update = learning_rate * (momentum_coeff * m + scaled_delta);
        if (update > cap) update = cap;
        else if (update < -cap) update = -cap;
        wv = w.y + update;
        wv = fminf(weight_limit, fmaxf(-weight_limit, wv));
        w.y = wv;
        potv = pot.y + fabsf(scaled_delta) * gradient_score_scale;
        pot.y = fminf(synapse_potential_cap, potv);
        mem_factor = 1.0f + mem.y;
        sc.y = fabsf(error) * fabsf(wv) / (float)path_len * mem_factor;

        weights2[i] = __floats2half2_rn(w.x, w.y);
        potentials2[i] = __floats2half2_rn(pot.x, pot.y);
        momentum2[i] = __floats2half2_rn(mom.x, mom.y);
        grad_sq2[i] = __floats2half2_rn(gs.x, gs.y);
        prev_grad2[i] = __floats2half2_rn(pg.x, pg.y);
        scores2[i] = __floats2half2_rn(sc.x, sc.y);
    }
    for (int i = n2 * 2 + idx; i < n; i += stride) {
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
