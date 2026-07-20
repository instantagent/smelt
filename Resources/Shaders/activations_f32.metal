#include <metal_stdlib>
using namespace metal;

/// erf approximation — Metal has no erf(). Abramowitz & Stegun 7.1.26, max abs error
/// ~1.5e-7, far inside the cosine>=0.999/relL2<=0.02 gate vs the CPU's erff.
inline float erf_approx(float x) {
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                      - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

/// GELU (erf form), fp32: y = 0.5*x*(1 + erf(x/sqrt(2))). Matches Qwen3TTSCodec.gelu
/// (ConvNeXt pointwise activation). One thread per element. Constant 2: count.
kernel void gelu_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float x = input[tid];
    output[tid] = 0.5f * x * (1.0f + erf_approx(x * 0.70710678f));  // 1/sqrt(2)
}

/// SiLU (swish), fp32: y = x / (1 + exp(-x)). The pre_transformer MLP gate activation
/// (silu(gate)*up is then an elementwise multiply). One thread per element.
kernel void silu_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float x = input[tid];
    output[tid] = x / (1.0f + exp(-x));
}

/// SwiGLU, fp32: out = silu(gate) * up (fused). The pre_transformer MLP nonlinearity.
kernel void swiglu_f32(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float g = gate[tid];
    output[tid] = (g / (1.0f + exp(-g))) * up[tid];
}
