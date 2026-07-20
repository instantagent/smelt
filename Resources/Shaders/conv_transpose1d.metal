#include <metal_stdlib>
using namespace metal;

/// 1D transposed convolution (upsampling).
/// One thread per output element.
///
/// Buffers:
///   0: input   [C_in, L_in]         FP16
///   1: weight  [C_in, C_out, K]     FP16 (note: C_in is first dim for transpose)
///   2: bias    [C_out]              FP16
///   3: output  [C_out, L_out]       FP16
/// Constants:
///   0: C_in    uint
///   1: C_out   uint
///   2: K       uint (kernel size)
///   3: stride  uint
///   4: padding uint
///   5: L_in    uint
kernel void conv_transpose1d(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half*       output  [[buffer(3)]],
    constant uint&     C_in    [[buffer(4)]],
    constant uint&     C_out   [[buffer(5)]],
    constant uint&     K       [[buffer(6)]],
    constant uint&     stride  [[buffer(7)]],
    constant uint&     padding [[buffer(8)]],
    constant uint&     L_in    [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint oc = tid.y;   // output channel
    uint ol = tid.x;   // output position

    // L_out = (L_in - 1) * stride - 2 * padding + K
    uint L_out = (L_in - 1) * stride - 2 * padding + K;
    if (oc >= C_out || ol >= L_out) return;

    float acc = float(bias[oc]);

    for (uint ic = 0; ic < C_in; ic++) {
        for (uint k = 0; k < K; k++) {
            // Transposed conv: output[ol] accumulates from input positions
            // where (ol + padding - k) is divisible by stride
            int numerator = int(ol) + int(padding) - int(k);
            if (numerator >= 0 && numerator % int(stride) == 0) {
                uint il = uint(numerator) / stride;
                if (il < L_in) {
                    float w = float(weight[ic * C_out * K + oc * K + k]);
                    float x = float(input[ic * L_in + il]);
                    acc += w * x;
                }
            }
        }
    }

    output[oc * L_out + ol] = half(acc);
}
