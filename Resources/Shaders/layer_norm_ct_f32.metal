#include <metal_stdlib>
using namespace metal;

/// LayerNorm over the channel dim per frame, fp32, on [channels, frames] (C,T) storage.
/// Matches Qwen3TTSCodec.layerNormCT (the ConvNeXt LN): for each frame t, normalize across
/// channels then scale/shift by normW[c]/normB[c]. One thread per frame.
///
/// Buffers:
///   0: input   [channels, frames]  float
///   1: normW   [channels]          float
///   2: normB   [channels]          float
///   3: output  [channels, frames]  float
/// Constants:
///   4: channels  uint
///   5: frames    uint
///   6: eps       float
kernel void layer_norm_ct_f32(
    device const float* input    [[buffer(0)]],
    device const float* normW    [[buffer(1)]],
    device const float* normB    [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint&      channels [[buffer(4)]],
    constant uint&      frames   [[buffer(5)]],
    constant float&     eps      [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint t = tid;
    if (t >= frames) return;

    float mean = 0.0f;
    for (uint c = 0; c < channels; c++) mean += input[c * frames + t];
    mean /= float(channels);

    float varr = 0.0f;
    for (uint c = 0; c < channels; c++) {
        float d = input[c * frames + t] - mean;
        varr += d * d;
    }
    varr /= float(channels);

    float inv = 1.0f / sqrt(varr + eps);
    for (uint c = 0; c < channels; c++) {
        output[c * frames + t] = (input[c * frames + t] - mean) * inv * normW[c] + normB[c];
    }
}
