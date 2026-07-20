#include <metal_stdlib>
using namespace metal;

/// General 1D convolution (fp32 variant of conv1d_forward, for the fp32 codec).
/// Supports padding, stride, dilation, and groups. One thread per output element.
/// Contiguous [C, L] buffers: input row stride = L_in, output row stride = L_out.
///
/// Buffers:
///   0: input   [C_in, L_in]        float
///   1: weight  [C_out, C_in/G, K]  float
///   2: bias    [C_out]             float
///   3: output  [C_out, L_out]      float
/// Constants:
///   4: C_in, 5: C_out, 6: K, 7: stride, 8: padding, 9: dilation, 10: groups,
///   11: L_in_packed (low 31 bits = L_in; high bit set = has_bias)
kernel void conv1d_forward_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint&      C_in    [[buffer(4)]],
    constant uint&      C_out   [[buffer(5)]],
    constant uint&      K       [[buffer(6)]],
    constant uint&      stride  [[buffer(7)]],
    constant uint&      padding [[buffer(8)]],
    constant uint&      dilation [[buffer(9)]],
    constant uint&      groups  [[buffer(10)]],
    constant uint&      L_in_packed [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint L_in = L_in_packed & 0x7FFFFFFFu;
    bool has_bias = (L_in_packed >> 31) != 0;

    uint oc = tid.y;   // output channel
    uint ol = tid.x;   // output position

    uint L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (oc >= C_out || ol >= L_out) return;

    uint in_stride = L_in;
    uint out_stride = L_out;

    uint group_size_in = C_in / groups;
    uint group_size_out = C_out / groups;
    uint g = oc / group_size_out;

    float acc = has_bias ? bias[oc] : 0.0f;

    // Interior fast path: when every tap is in-bounds (always true for padding=0 "valid" convs —
    // the whole codec-stream path), drop the per-tap bounds check + index arithmetic that
    // dominated the inner loop (~10 ALU ops per FMA, the measured codec-chunk wall). The
    // accumulation SEQUENCE is unchanged (bias first, ic ascending, k ascending, same operands)
    // → bit-identical outputs; only address computation differs.
    int il0 = int(ol * stride) - int(padding);
    if (il0 >= 0 && il0 + int((K - 1) * dilation) < int(L_in)) {
        device const float* wrow = weight + oc * group_size_in * K;
        device const float* xbase = input + g * group_size_in * in_stride + uint(il0);
        if (K == 7) {
            for (uint ic = 0; ic < group_size_in; ic++) {
                device const float* xr = xbase + ic * in_stride;
                #pragma unroll
                for (uint k = 0; k < 7; k++) acc += wrow[ic * 7 + k] * xr[k * dilation];
            }
        } else if (K == 3) {
            for (uint ic = 0; ic < group_size_in; ic++) {
                device const float* xr = xbase + ic * in_stride;
                #pragma unroll
                for (uint k = 0; k < 3; k++) acc += wrow[ic * 3 + k] * xr[k * dilation];
            }
        } else if (K == 1) {
            for (uint ic = 0; ic < group_size_in; ic++) acc += wrow[ic] * xbase[ic * in_stride];
        } else {
            for (uint ic = 0; ic < group_size_in; ic++) {
                device const float* xr = xbase + ic * in_stride;
                for (uint k = 0; k < K; k++) acc += wrow[ic * K + k] * xr[k * dilation];
            }
        }
        output[oc * out_stride + ol] = acc;
        return;
    }

    for (uint ic = 0; ic < group_size_in; ic++) {
        uint ic_global = g * group_size_in + ic;
        for (uint k = 0; k < K; k++) {
            int il = int(ol * stride) - int(padding) + int(k * dilation);
            if (il >= 0 && uint(il) < L_in) {
                float w = weight[oc * group_size_in * K + ic * K + k];
                float x = input[ic_global * in_stride + uint(il)];
                acc += w * x;
            }
        }
    }

    output[oc * out_stride + ol] = acc;
}
