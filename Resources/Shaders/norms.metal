#include <metal_stdlib>
using namespace metal;

// ─── RMSNorm (1 + weight) variant ───
// Used for input_layernorm and post_attention_layernorm.
// x * rsqrt(mean(x^2) + eps) * (1 + weight)
// Dispatch: 1 threadgroup, threads = min(dim, 1024)

kernel void rms_norm_1pw(
    device const half* input   [[buffer(0)]],  // [dim]
    device const half* weight  [[buffer(1)]],  // [dim]
    device half*       output  [[buffer(2)]],  // [dim]
    constant uint&     dim     [[buffer(3)]],  // hidden dimension
    constant float&    eps     [[buffer(4)]],  // 1e-6
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Match MLX rms_looped: four contiguous reads per lane, then advance by
    // one whole threadgroup tile. A lane-strided walk changes the reduction
    // tree for dimensions above MLX's 4096-element single-row threshold.
    constexpr uint reads = 4;
    float sumSq = 0.0f;
    const uint localBase = tid * reads;
    for (uint tile = 0; tile < dim; tile += tgs * reads) {
        for (uint read = 0; read < reads; ++read) {
            const uint i = tile + localBase + read;
            if (i < dim) {
                const float v = float(input[i]);
                sumSq += v * v;
            }
        }
    }

    // SIMD reduction
    sumSq = simd_sum(sumSq);

    // Cross-SIMD reduction
    threadgroup float partial[32];
    if (simd_group == 0) { partial[simd_lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (simd_group == 0) {
        const float total = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            shared_rsqrt = metal::precise::rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply: x * rsqrt * (1 + weight)
    for (uint tile = 0; tile < dim; tile += tgs * reads) {
        for (uint read = 0; read < reads; ++read) {
            const uint i = tile + localBase + read;
            if (i < dim) {
                const half normalized = half(float(input[i]) * shared_rsqrt);
                const half directWeight = half(1.0f + float(weight[i]));
                output[i] = directWeight * normalized;
            }
        }
    }

}

// Decode hot-path specialization for hiddenSize=2048, eps=1e-6.
// Uses 256 threads and vectorized half4 loads/stores, while caching the input
// once in threadgroup memory so the second pass avoids another device read.
kernel void rms_norm_1pw_d2048(
    device const half* input   [[buffer(0)]],  // [2048]
    device const half* weight  [[buffer(1)]],  // [2048]
    device half*       output  [[buffer(2)]],  // [2048]
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2048;
    constexpr uint D4 = D / 4;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output);

    threadgroup half4 cached[D4];
    threadgroup float partial[8];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + 256];
    cached[tid] = xh0;
    cached[tid + 256] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < 8; s++) {
            total += partial[s];
        }
        shared_rsqrt = metal::precise::rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + 256]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + 256] = half4(float4(cached[tid + 256]) * scale1);
}

kernel void rms_norm_1pw_d1024(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1024;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = metal::precise::rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d256_add(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 256;
    constexpr uint SIMD_GROUPS = 8;
    constexpr float eps = 1e-6f;

    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output);
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    float x = float(input[tid]);
    float sumSq = x * x;
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < SIMD_GROUPS; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    half residualValue = residual[tid];
    half normValue = half(x * shared_rsqrt * (1.0f + float(weight[tid])));
    outputScalar[tid] = normValue;
    outputScalar[tid] = half(outputScalar[tid] + residualValue);
}

kernel void rms_norm_1pw_d256_add_scalar_weight(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device const half* scalar   [[buffer(3)]],
    device half*       output   [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 256;
    constexpr uint SIMD_GROUPS = 8;
    constexpr float eps = 1e-6f;

    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output);
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    float x = float(input[tid]);
    float sumSq = x * x;
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < SIMD_GROUPS; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    half residualValue = residual[tid];
    half scalarValue = scalar[0];
    half normValue = half(x * shared_rsqrt * (1.0f + float(weight[tid])));
    outputScalar[tid] = normValue;
    outputScalar[tid] = half(outputScalar[tid] + residualValue);
    outputScalar[tid] = half(float(outputScalar[tid]) * float(scalarValue));
}

kernel void rms_norm_1pw_d1024_batched(
    device const half* input   [[buffer(0)]],  // [B, 1024]
    device const half* weight  [[buffer(1)]],  // [1024]
    device half*       output  [[buffer(2)]],  // [B, 1024]
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1024;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output + batch * D);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d1536(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1536;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d1536_add(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1536;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output);
    const device half4* residual4 = reinterpret_cast<const device half4*>(residual);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));
    half4 norm0 = half4(float4(cached[tid]) * scale0);
    half4 norm1 = half4(float4(cached[tid + VEC_THREADS]) * scale1);
    half4 residual0 = residual4[tid];
    half4 residual1 = residual4[tid + VEC_THREADS];
    uint base0 = tid * 4;
    uint base1 = (tid + VEC_THREADS) * 4;

    // Match the unfused path exactly: write half lanes to device memory first,
    // then reload those half lanes and apply scalar half adds. Cache residuals
    // first so residual/output aliasing matches staged norm + add.
    outputScalar[base0] = norm0.x;
    outputScalar[base0 + 1] = norm0.y;
    outputScalar[base0 + 2] = norm0.z;
    outputScalar[base0 + 3] = norm0.w;
    outputScalar[base1] = norm1.x;
    outputScalar[base1 + 1] = norm1.y;
    outputScalar[base1 + 2] = norm1.z;
    outputScalar[base1 + 3] = norm1.w;

    outputScalar[base0] = half(outputScalar[base0] + residual0.x);
    outputScalar[base0 + 1] = half(outputScalar[base0 + 1] + residual0.y);
    outputScalar[base0 + 2] = half(outputScalar[base0 + 2] + residual0.z);
    outputScalar[base0 + 3] = half(outputScalar[base0 + 3] + residual0.w);
    outputScalar[base1] = half(outputScalar[base1] + residual1.x);
    outputScalar[base1 + 1] = half(outputScalar[base1 + 1] + residual1.y);
    outputScalar[base1 + 2] = half(outputScalar[base1 + 2] + residual1.z);
    outputScalar[base1 + 3] = half(outputScalar[base1 + 3] + residual1.w);
}

kernel void rms_norm_1pw_d1536_batched(
    device const half* input   [[buffer(0)]],  // [B, 1536]
    device const half* weight  [[buffer(1)]],  // [1536]
    device half*       output  [[buffer(2)]],  // [B, 1536]
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1536;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output + batch * D);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d1536_add_batched(
    device const half* input    [[buffer(0)]],  // [B, 1536]
    device const half* weight   [[buffer(1)]],  // [1536]
    device const half* residual [[buffer(2)]],  // [B, 1536]
    device half*       output   [[buffer(3)]],  // [B, 1536]
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1536;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    const device half4* residual4 = reinterpret_cast<const device half4*>(residual + batch * D);
    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output + batch * D);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));
    half4 norm0 = half4(float4(cached[tid]) * scale0);
    half4 norm1 = half4(float4(cached[tid + VEC_THREADS]) * scale1);
    half4 residual0 = residual4[tid];
    half4 residual1 = residual4[tid + VEC_THREADS];
    uint base0 = tid * 4;
    uint base1 = (tid + VEC_THREADS) * 4;

    outputScalar[base0] = norm0.x;
    outputScalar[base0 + 1] = norm0.y;
    outputScalar[base0 + 2] = norm0.z;
    outputScalar[base0 + 3] = norm0.w;
    outputScalar[base1] = norm1.x;
    outputScalar[base1 + 1] = norm1.y;
    outputScalar[base1 + 2] = norm1.z;
    outputScalar[base1 + 3] = norm1.w;

    outputScalar[base0] = half(outputScalar[base0] + residual0.x);
    outputScalar[base0 + 1] = half(outputScalar[base0 + 1] + residual0.y);
    outputScalar[base0 + 2] = half(outputScalar[base0 + 2] + residual0.z);
    outputScalar[base0 + 3] = half(outputScalar[base0 + 3] + residual0.w);
    outputScalar[base1] = half(outputScalar[base1] + residual1.x);
    outputScalar[base1 + 1] = half(outputScalar[base1 + 1] + residual1.y);
    outputScalar[base1 + 2] = half(outputScalar[base1 + 2] + residual1.z);
    outputScalar[base1 + 3] = half(outputScalar[base1 + 3] + residual1.w);
}

kernel void rms_norm_scale_only_d1536(
    device const half* input    [[buffer(0)]],  // [1536]
    device float*      scaleOut [[buffer(1)]],  // [1]
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 1536;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    threadgroup float partial[32];

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        scaleOut[0] = rsqrt(total / float(D) + eps);
    }
}

kernel void rms_norm_scale_only_d2560(
    device const half* input    [[buffer(0)]],  // [2560]
    device float*      scaleOut [[buffer(1)]],  // [1]
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2560;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    threadgroup float partial[32];

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        scaleOut[0] = rsqrt(total / float(D) + eps);
    }
}

#define AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED(NAME, D, VEC_THREADS) \
kernel void NAME( \
    device const half* input    [[buffer(0)]], \
    device float*      scaleOut [[buffer(1)]], \
    uint batch      [[threadgroup_position_in_grid]], \
    uint tid        [[thread_index_in_threadgroup]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    constexpr float eps = 1e-6f; \
    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D); \
    threadgroup float partial[32]; \
    half4 xh0 = input4[tid]; \
    half4 xh1 = input4[tid + VEC_THREADS]; \
    float4 x0 = float4(xh0); \
    float4 x1 = float4(xh1); \
    float sumSq = dot(x0, x0) + dot(x1, x1); \
    sumSq = simd_sum(sumSq); \
    if (simd_lane == 0) { \
        partial[simd_group] = sumSq; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float total = 0.0f; \
        for (uint s = 0; s < VEC_THREADS / 32; s++) { \
            total += partial[s]; \
        } \
        scaleOut[batch] = rsqrt(total / float(D) + eps); \
    } \
}

AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED(rms_norm_scale_only_d1024_batched, 1024, 128)

#undef AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED

#define AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR(NAME, D, EPS) \
kernel void NAME( \
    device const half* input    [[buffer(0)]], \
    device float*      scaleOut [[buffer(1)]], \
    uint batch      [[threadgroup_position_in_grid]], \
    uint tid        [[thread_index_in_threadgroup]], \
    uint tgs        [[threads_per_threadgroup]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    constexpr float eps = EPS; \
    uint offset = batch * D; \
    float sumSq = 0.0f; \
    for (uint i = tid; i < D; i += tgs) { \
        float v = float(input[offset + i]); \
        sumSq += v * v; \
    } \
    sumSq = simd_sum(sumSq); \
    threadgroup float partial[32]; \
    if (simd_lane == 0) { \
        partial[simd_group] = sumSq; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float total = 0.0f; \
        for (uint s = 0; s < tgs / 32; s++) { \
            total += partial[s]; \
        } \
        scaleOut[batch] = rsqrt(total / float(D) + eps); \
    } \
}

AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR(rms_norm_scale_only_d2048_batched, 2048, 1e-6f)
AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR(rms_norm_scale_only_d2048_eps1e5_batched, 2048, 1e-5f)
AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR(rms_norm_scale_only_d2560_batched, 2560, 1e-6f)
AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR(rms_norm_scale_only_d3072_eps1e5_batched, 3072, 1e-5f)

#undef AGENT_DECLARE_RMS_NORM_SCALE_ONLY_1PW_BATCHED_SCALAR

kernel void rms_norm_1pw_d2560(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2560;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d2560_add(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2560;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    const device half4* residual4 = reinterpret_cast<const device half4*>(residual);
    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));
    half4 norm0 = half4(float4(cached[tid]) * scale0);
    half4 norm1 = half4(float4(cached[tid + VEC_THREADS]) * scale1);
    half4 residual0 = residual4[tid];
    half4 residual1 = residual4[tid + VEC_THREADS];
    uint base0 = tid * 4;
    uint base1 = (tid + VEC_THREADS) * 4;

    outputScalar[base0] = norm0.x;
    outputScalar[base0 + 1] = norm0.y;
    outputScalar[base0 + 2] = norm0.z;
    outputScalar[base0 + 3] = norm0.w;
    outputScalar[base1] = norm1.x;
    outputScalar[base1 + 1] = norm1.y;
    outputScalar[base1 + 2] = norm1.z;
    outputScalar[base1 + 3] = norm1.w;

    outputScalar[base0] = half(outputScalar[base0] + residual0.x);
    outputScalar[base0 + 1] = half(outputScalar[base0 + 1] + residual0.y);
    outputScalar[base0 + 2] = half(outputScalar[base0 + 2] + residual0.z);
    outputScalar[base0 + 3] = half(outputScalar[base0 + 3] + residual0.w);
    outputScalar[base1] = half(outputScalar[base1] + residual1.x);
    outputScalar[base1 + 1] = half(outputScalar[base1 + 1] + residual1.y);
    outputScalar[base1 + 2] = half(outputScalar[base1 + 2] + residual1.z);
    outputScalar[base1 + 3] = half(outputScalar[base1 + 3] + residual1.w);
}

kernel void rms_norm_1pw_d2560_batched(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2560;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    device half4* output4 = reinterpret_cast<device half4*>(output + batch * D);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));

    output4[tid] = half4(float4(cached[tid]) * scale0);
    output4[tid + VEC_THREADS] = half4(float4(cached[tid + VEC_THREADS]) * scale1);
}

kernel void rms_norm_1pw_d2560_add_batched(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half*       output   [[buffer(3)]],
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 2560;
    constexpr uint D4 = D / 4;
    constexpr uint VEC_THREADS = D4 / 2;
    constexpr float eps = 1e-6f;

    const device half4* input4 = reinterpret_cast<const device half4*>(input + batch * D);
    const device half4* weight4 = reinterpret_cast<const device half4*>(weight);
    const device half4* residual4 = reinterpret_cast<const device half4*>(residual + batch * D);
    volatile device half* outputScalar = reinterpret_cast<volatile device half*>(output + batch * D);

    threadgroup half4 cached[D4];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;

    half4 xh0 = input4[tid];
    half4 xh1 = input4[tid + VEC_THREADS];
    cached[tid] = xh0;
    cached[tid + VEC_THREADS] = xh1;

    float4 x0 = float4(xh0);
    float4 x1 = float4(xh1);
    float sumSq = dot(x0, x0) + dot(x1, x1);
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < VEC_THREADS / 32; s++) {
            total += partial[s];
        }
        shared_rsqrt = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    float4 scale0 = rs * (1.0f + float4(weight4[tid]));
    float4 scale1 = rs * (1.0f + float4(weight4[tid + VEC_THREADS]));
    half4 norm0 = half4(float4(cached[tid]) * scale0);
    half4 norm1 = half4(float4(cached[tid + VEC_THREADS]) * scale1);
    half4 residual0 = residual4[tid];
    half4 residual1 = residual4[tid + VEC_THREADS];
    uint base0 = tid * 4;
    uint base1 = (tid + VEC_THREADS) * 4;

    outputScalar[base0] = norm0.x;
    outputScalar[base0 + 1] = norm0.y;
    outputScalar[base0 + 2] = norm0.z;
    outputScalar[base0 + 3] = norm0.w;
    outputScalar[base1] = norm1.x;
    outputScalar[base1 + 1] = norm1.y;
    outputScalar[base1 + 2] = norm1.z;
    outputScalar[base1 + 3] = norm1.w;

    outputScalar[base0] = half(outputScalar[base0] + residual0.x);
    outputScalar[base0 + 1] = half(outputScalar[base0 + 1] + residual0.y);
    outputScalar[base0 + 2] = half(outputScalar[base0 + 2] + residual0.z);
    outputScalar[base0 + 3] = half(outputScalar[base0 + 3] + residual0.w);
    outputScalar[base1] = half(outputScalar[base1] + residual1.x);
    outputScalar[base1 + 1] = half(outputScalar[base1 + 1] + residual1.y);
    outputScalar[base1 + 2] = half(outputScalar[base1 + 2] + residual1.z);
    outputScalar[base1 + 3] = half(outputScalar[base1 + 3] + residual1.w);
}

// ─── RMSNorm scale only ───
// Computes rsqrt(mean(x^2) + eps) once and writes a single FP32 scale.
// Used by cooperative norm fusion so consumer kernels can normalize inline.

kernel void rms_norm_scale_only(
    device const half* input    [[buffer(0)]],  // [dim]
    device float*      scaleOut [[buffer(1)]],  // [1]
    constant uint&     dim      [[buffer(2)]],
    constant float&    eps      [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = float(input[i]);
        sumSq += v * v;
    }

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < tgs / 32; s++) {
            total += partial[s];
        }
        scaleOut[0] = rsqrt(total / float(dim) + eps);
    }
}

// Exact scale-only producer for graph rewrites that must match rms_norm_1pw's
// precise rsqrt before a consumer reproduces its fp16 normalization boundary.
kernel void rms_norm_scale_only_precise(
    device const half* input    [[buffer(0)]],
    device float*      scaleOut [[buffer(1)]],
    constant uint&     dim      [[buffer(2)]],
    constant float&    eps      [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        const float value = float(input[i]);
        sumSq += value * value;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_group] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint simd = 0; simd < tgs / 32; ++simd) {
            total += partial[simd];
        }
        scaleOut[0] = metal::precise::rsqrt(total / float(dim) + eps);
    }
}

// Generic residual boundary + precise RMS-scale producer. This deliberately
// keeps the matvec in its proven kernel and reproduces elementwise_add's fp16
// store before the exact scale reduction observes the updated hidden state.
kernel void residual_add_rms_norm_scale_only_precise(
    device const half* inputA   [[buffer(0)]],
    device const half* inputB   [[buffer(1)]],
    device half*       output   [[buffer(2)]],
    device float*      scaleOut [[buffer(3)]],
    constant uint&     dim      [[buffer(4)]],
    constant float&    eps      [[buffer(5)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        const half value = inputA[i] + inputB[i];
        output[i] = value;
        const float precise_value = float(value);
        sumSq += precise_value * precise_value;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) partial[simd_group] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint simd = 0; simd < tgs / 32; ++simd) {
            total += partial[simd];
        }
        scaleOut[0] = metal::precise::rsqrt(total / float(dim) + eps);
    }
}

kernel void rms_norm_gated(
    device const half* input   [[buffer(0)]],  // [H * D] flattened
    device const half* gate    [[buffer(1)]],  // [H * D] flattened (z projection)
    device const half* weight  [[buffer(2)]],  // [D] (shared across heads)
    device half*       output  [[buffer(3)]],  // [H * D] flattened
    constant uint&     headDim [[buffer(4)]],  // D (128)
    constant float&    eps     [[buffer(5)]],  // 1e-6
    uint head       [[threadgroup_position_in_grid]],  // head index
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    // Cache input and gate in threadgroup memory to avoid double device read.
    // headDim is at most 256 for known models.
    threadgroup float cached_input[256];
    threadgroup float cached_gate[256];

    // Pass 1: load into cache + accumulate sum of squares
    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(input[offset + i]);
        float g = float(gate[offset + i]);
        cached_input[i] = v;
        cached_gate[i] = g;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = metal::precise::rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;

    // Pass 2: apply from cached values (no second device read)
    for (uint i = tid; i < headDim; i += tgs) {
        float x = cached_input[i];
        float g = cached_gate[i];
        float w = float(weight[i]);
        float silu_g = g / (1.0f + exp(-g));  // silu = x * sigmoid(x)
        output[offset + i] = half(w * (x * rs) * silu_g);
    }
}

// ─── L2 Normalize (clamp style) ───
// x / max(||x||_2, eps) — per head, dim=-1
// Dispatch: numHeads threadgroups

kernel void l2_normalize_d128(
    device half* data [[buffer(0)]],  // [H * 128] in-place
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    constexpr float eps = 1e-6f;
    uint offset = head * D;

    threadgroup half cached[D];
    threadgroup float partial[4];
    threadgroup float shared_scale = 0.0f;

    half vh = data[offset + tid];
    cached[tid] = vh;
    float v = float(vh);
    float sumSq = v * v;
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        shared_scale = 1.0f / max(sqrt(total), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    data[offset + tid] = half(float(cached[tid]) * shared_scale);
}

// Prompt-prefill specializations for Qwen DeltaNet Q/K slices inside qkv[T, 6144].
// Dispatch with threadgroups(width: 16, height: seqLen, depth: 1), tg=(128,1,1).
kernel void l2_normalize_q_d128_c6144_h16_prefill(
    device half* qkv [[buffer(0)]],
    uint2 group    [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    constexpr uint C = 6144;
    constexpr float eps = 1e-6f;

    uint head = group.x;
    uint pos = group.y;
    uint offset = pos * C + head * D;

    threadgroup half cached[D];
    threadgroup float partial[4];
    threadgroup float shared_scale = 0.0f;

    half vh = qkv[offset + tid];
    cached[tid] = vh;
    float v = float(vh);
    float sumSq = v * v;
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        shared_scale = 1.0f / max(sqrt(total), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qkv[offset + tid] = half(float(cached[tid]) * shared_scale);
}

kernel void l2_normalize_k_d128_c6144_h16_prefill(
    device half* qkv [[buffer(0)]],
    uint2 group    [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    constexpr uint H = 16;
    constexpr uint C = 6144;
    constexpr uint K_BASE = H * D;
    constexpr float eps = 1e-6f;

    uint head = group.x;
    uint pos = group.y;
    uint offset = pos * C + K_BASE + head * D;

    threadgroup half cached[D];
    threadgroup float partial[4];
    threadgroup float shared_scale = 0.0f;

    half vh = qkv[offset + tid];
    cached[tid] = vh;
    float v = float(vh);
    float sumSq = v * v;
    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        shared_scale = 1.0f / max(sqrt(total), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qkv[offset + tid] = half(float(cached[tid]) * shared_scale);
}

kernel void l2_normalize_q_prefill(
    device half*       qkv     [[buffer(0)]],
    constant uint&     headDim [[buffer(1)]],
    constant float&    eps     [[buffer(2)]],
    constant uint&     qkvDim  [[buffer(3)]],
    constant uint&     qkHeads [[buffer(4)]],
    uint2 group       [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint2 tgs_v       [[threads_per_threadgroup]],
    uint simd_lane    [[thread_index_in_simdgroup]],
    uint simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint tgs = tgs_v.x;
    uint head = group.x;
    uint pos = group.y;
    if (head >= qkHeads) return;

    uint offset = pos * qkvDim + head * headDim;

    threadgroup float cached[512];
    threadgroup float partial[32];
    threadgroup float shared_scale = 0.0f;

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(qkv[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < tgs / 32; s++) {
            total += partial[s];
        }
        shared_scale = 1.0f / max(sqrt(total), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < headDim; i += tgs) {
        qkv[offset + i] = half(cached[i] * shared_scale);
    }
}

kernel void l2_normalize_k_prefill(
    device half*       qkv     [[buffer(0)]],
    constant uint&     headDim [[buffer(1)]],
    constant float&    eps     [[buffer(2)]],
    constant uint&     qkvDim  [[buffer(3)]],
    constant uint&     qkHeads [[buffer(4)]],
    uint2 group       [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint2 tgs_v       [[threads_per_threadgroup]],
    uint simd_lane    [[thread_index_in_simdgroup]],
    uint simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint tgs = tgs_v.x;
    uint head = group.x;
    uint pos = group.y;
    if (head >= qkHeads) return;

    uint offset = pos * qkvDim + qkHeads * headDim + head * headDim;

    threadgroup float cached[512];
    threadgroup float partial[32];
    threadgroup float shared_scale = 0.0f;

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(qkv[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < tgs / 32; s++) {
            total += partial[s];
        }
        shared_scale = 1.0f / max(sqrt(total), eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < headDim; i += tgs) {
        qkv[offset + i] = half(cached[i] * shared_scale);
    }
}

// Q/K RMS scaling brick shared by decode and prompt prefill. One dispatch
// owns both adjacent Q and K regions, preserving their distinct graph-edge
// scales without relying on an optimizer to coalesce two nominally identical
// L2 normalizations.
kernel void rms_scale_qk(
    device half*       qkv     [[buffer(0)]],
    constant uint&     headDim [[buffer(1)]],
    constant float&    eps     [[buffer(2)]],
    constant uint&     qkvDim  [[buffer(3)]],
    constant uint&     qkHeads [[buffer(4)]],
    uint2 group       [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint2 tgs_v       [[threads_per_threadgroup]],
    uint simd_lane    [[thread_index_in_simdgroup]],
    uint simd_group   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint N_READS = 4;
    const bool isK = group.x >= qkHeads;
    const uint head = isK ? group.x - qkHeads : group.x;
    if (head >= qkHeads) return;

    const uint regionBase = isK ? qkHeads * headDim : 0;
    const uint offset = group.y * qkvDim + regionBase + head * headDim;
    const uint readBase = tid * N_READS;

    threadgroup half cached[512];
    threadgroup float partial[16];
    threadgroup float sharedScale = 0.0f;

    float sumSq = 0.0f;
    for (uint i = 0; i < N_READS && readBase + i < headDim; ++i) {
        const half value = qkv[offset + readBase + i];
        cached[readBase + i] = value;
        const float x = float(value);
        sumSq += x * x;
    }
    sumSq = simd_sum(sumSq);
    if (simd_lane == 0) {
        partial[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint groupIndex = 0; groupIndex < tgs_v.x / 32; ++groupIndex) {
            total += partial[groupIndex];
        }
        sharedScale = metal::precise::rsqrt(total / float(headDim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const half edgeScale = isK
        ? half(metal::precise::rsqrt(float(headDim)))
        : half(1.0f / float(headDim));
    for (uint i = 0; i < N_READS && readBase + i < headDim; ++i) {
        const half normalized = half(float(cached[readBase + i]) * sharedScale);
        qkv[offset + readBase + i] = normalized * edgeScale;
    }
}

kernel void l2_normalize(
    device half*       data    [[buffer(0)]],  // [H * D] in-place
    constant uint&     headDim [[buffer(1)]],  // D (128)
    constant float&    eps     [[buffer(2)]],  // 1e-6
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    // Cache data in threadgroup memory to avoid double device read.
    threadgroup float cached[512];

    // Pass 1: load + compute L2 norm in FP32
    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(data[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_scale = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float norm = sqrt(total);
        shared_scale = 1.0f / max(norm, eps);  // clamp, not add-inside-sqrt
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: normalize from cached values (no second device read)
    float scale = shared_scale;
    for (uint i = tid; i < headDim; i += tgs) {
        data[offset + i] = half(cached[i] * scale);
    }
}

// ─── Batched RMSNorm (1 + weight) — prefill ───
// Applies rms_norm_1pw independently to B vectors.
// Input [B, dim] row-major, weight [dim] shared, output [B, dim].
// Dispatch: B threadgroups, min(dim, 1024) threads each.
// threadgroup_position_in_grid = batch index.

kernel void rms_norm_1pw_batched(
    device const half* input   [[buffer(0)]],  // [B, dim]
    device const half* weight  [[buffer(1)]],  // [dim]
    device half*       output  [[buffer(2)]],  // [B, dim]
    constant uint&     dim     [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = batch * dim;

    // Keep the same four-contiguous-values-per-lane reduction tree as the
    // single-row/MLX kernel. A lane-strided batched walk is mathematically
    // equivalent but changes FP32 rounding once dense hidden states arrive.
    constexpr uint reads = 4;
    float sumSq = 0.0f;
    const uint localBase = tid * reads;
    for (uint tile = 0; tile < dim; tile += tgs * reads) {
        for (uint read = 0; read < reads; ++read) {
            const uint i = tile + localBase + read;
            if (i < dim) {
                const float v = float(input[offset + i]);
                sumSq += v * v;
            }
        }
    }

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_group == 0) { partial[simd_lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (simd_group == 0) {
        const float total = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            shared_rsqrt = metal::precise::rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;

    for (uint tile = 0; tile < dim; tile += tgs * reads) {
        for (uint read = 0; read < reads; ++read) {
            const uint i = tile + localBase + read;
            if (i < dim) {
                const half normalized = half(float(input[offset + i]) * rs);
                const half directWeight = half(1.0f + float(weight[i]));
                output[offset + i] = directWeight * normalized;
            }
        }
    }
}
