#include <metal_stdlib>
using namespace metal;

/// General 1D convolution (spatial, not the DeltaNet stateful shift-and-dot).
/// Supports padding, stride, dilation, and groups. One thread per output element.
/// Contiguous [C, L] buffers: input row stride = L_in, output row stride = L_out
/// (L_out != L_in for strided convs, so they must differ — a shared stride would
/// write a padded layout the next contiguous conv mis-reads).
///
/// Buffers:
///   0: input   [C_in, L_in]        FP16
///   1: weight  [C_out, C_in/G, K]  FP16
///   2: bias    [C_out]             FP16
///   3: output  [C_out, L_out]      FP16
/// Constants:
///   4: C_in, 5: C_out, 6: K, 7: stride, 8: padding, 9: dilation, 10: groups,
///   11: L_in_packed (low 31 bits = L_in; high bit set = has_bias)
kernel void conv1d_forward(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half*       output  [[buffer(3)]],
    constant uint&     C_in    [[buffer(4)]],
    constant uint&     C_out   [[buffer(5)]],
    constant uint&     K       [[buffer(6)]],
    constant uint&     stride  [[buffer(7)]],
    constant uint&     padding [[buffer(8)]],
    constant uint&     dilation [[buffer(9)]],
    constant uint&     groups  [[buffer(10)]],
    constant uint&     L_in_packed [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint L_in = L_in_packed & 0x7FFFFFFFu;
    bool has_bias = (L_in_packed >> 31) != 0;

    uint oc = tid.y;   // output channel
    uint ol = tid.x;   // output position

    uint L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (oc >= C_out || ol >= L_out) return;

    uint in_stride = L_in;    // contiguous input rows
    uint out_stride = L_out;  // contiguous output rows

    uint group_size_in = C_in / groups;
    uint group_size_out = C_out / groups;
    uint g = oc / group_size_out;

    float acc = has_bias ? float(bias[oc]) : 0.0f;

    for (uint ic = 0; ic < group_size_in; ic++) {
        uint ic_global = g * group_size_in + ic;
        for (uint k = 0; k < K; k++) {
            int il = int(ol * stride) - int(padding) + int(k * dilation);
            if (il >= 0 && uint(il) < L_in) {
                float w = float(weight[oc * group_size_in * K + ic * K + k]);
                float x = float(input[ic_global * in_stride + uint(il)]);
                acc += w * x;
            }
        }
    }

    output[oc * out_stride + ol] = half(acc);
}
