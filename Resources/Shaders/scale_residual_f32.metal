#include <metal_stdlib>
using namespace metal;

/// Per-channel scale + residual, fp32, on [channels, frames] (C,T): out = residual + scale[c]*x.
/// Used for the ConvNeXt gamma+residual and the pre_transformer layer_scale+residual.
/// One thread per element.
///
/// Buffers: 0 x [C,T], 1 residual [C,T], 2 scale [C], 3 out [C,T]
/// Constants: 4 channels, 5 frames
kernel void scale_residual_f32(
    device const float* x        [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* scale    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      channels [[buffer(4)]],
    constant uint&      frames   [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint c = tid.y;
    uint t = tid.x;
    if (c >= channels || t >= frames) return;
    uint idx = c * frames + t;
    out[idx] = residual[idx] + scale[c] * x[idx];
}

/// Per-channel scale + residual on [frames, channels] (T,C, sequence-major): out[t,i] =
/// residual[t,i] + scale[i]*x[t,i]. The pre_transformer layer_scale + residual. One thread
/// per element. Buffers 0 x, 1 residual, 2 scale[channels], 3 out. Constants 4 channels, 5 frames,
/// 6 has_scale — when 0, `out = residual + x` (the talker/MTP residual-adds scale by all-ones, so the
/// scale read + multiply are skipped; bit-identical since 1.0*x == x in fp32).
kernel void scale_residual_tc_f32(
    device const float* x        [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* scale    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      channels [[buffer(4)]],
    constant uint&      frames   [[buffer(5)]],
    constant uint&      has_scale [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.y;  // channel
    uint t = tid.x;  // frame
    if (i >= channels || t >= frames) return;
    uint idx = t * channels + i;
    out[idx] = residual[idx] + (has_scale != 0 ? scale[i] * x[idx] : x[idx]);
}
