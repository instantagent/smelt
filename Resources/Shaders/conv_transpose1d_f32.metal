#include <metal_stdlib>
using namespace metal;

/// 1D transposed convolution (fp32 variant of conv_transpose1d, for the fp32 codec).
/// Produces the FULL transpose output (no causal right-trim — the caller trims by
/// (kernel - stride), matching Qwen3TTSCodec.causalTransConv1d). One thread per output.
///
/// Buffers:
///   0: input   [C_in, L_in]         float
///   1: weight  [C_in, C_out, K]     float (C_in is first dim for transpose)
///   2: bias    [C_out]              float
///   3: output  [C_out, L_out]       float
/// Constants:
///   4: C_in, 5: C_out, 6: K, 7: stride, 8: padding, 9: L_in
kernel void conv_transpose1d_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint&      C_in    [[buffer(4)]],
    constant uint&      C_out   [[buffer(5)]],
    constant uint&      K       [[buffer(6)]],
    constant uint&      stride  [[buffer(7)]],
    constant uint&      padding [[buffer(8)]],
    constant uint&      L_in    [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint oc = tid.y;
    uint ol = tid.x;

    uint L_out = (L_in - 1) * stride - 2 * padding + K;
    if (oc >= C_out || ol >= L_out) return;

    float acc = bias[oc];

    // Only taps with (ol+padding-k) ≡ 0 (mod stride) contribute — the rest are the zeros the transpose
    // interleaves. Step k by `stride` from k0 = (ol+padding) mod stride instead of testing every tap
    // (the DAC upsamples at stride 8/5/4/3 → ~stride× fewer iterations). numerator decreases by stride
    // each step; break when negative. Same nonzero terms in the same order → bit-exact vs the old loop.
    int op = int(ol) + int(padding);
    int k0 = op % int(stride);
    // The valid (k, il) tap pairs are IDENTICAL for every ic — hoist them out of the ic loop so
    // the inner body is two loads + one FMA instead of div/branch per tap (the codec transconvs
    // have K/stride == 2 live taps). Pair order is k-ascending, matching the original inner loop
    // → same terms in the same order per output, bit-exact. Fallback covers >8 live taps.
    uint kIdx[8]; uint ilIdx[8]; uint nTaps = 0;
    bool overflow = false;
    for (int k = k0; k < int(K); k += int(stride)) {
        int numerator = op - k;
        if (numerator < 0) break;
        uint il = uint(numerator) / stride;
        if (il < L_in) {
            if (nTaps >= 8) { overflow = true; break; }
            kIdx[nTaps] = uint(k); ilIdx[nTaps] = il; nTaps++;
        }
    }
    if (!overflow) {
        device const float* wrow = weight + oc * K;
        for (uint ic = 0; ic < C_in; ic++) {
            device const float* wic = wrow + ic * C_out * K;
            device const float* xic = input + ic * L_in;
            for (uint t = 0; t < nTaps; t++) acc += wic[kIdx[t]] * xic[ilIdx[t]];
        }
        output[oc * L_out + ol] = acc;
        return;
    }

    for (uint ic = 0; ic < C_in; ic++) {
        for (int k = k0; k < int(K); k += int(stride)) {
            int numerator = op - k;
            if (numerator < 0) break;
            uint il = uint(numerator) / stride;
            if (il < L_in) {
                acc += weight[ic * C_out * K + oc * K + uint(k)] * input[ic * L_in + il];
            }
        }
    }

    output[oc * L_out + ol] = acc;
}
