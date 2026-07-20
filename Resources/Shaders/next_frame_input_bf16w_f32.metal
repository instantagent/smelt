#include <metal_stdlib>
using namespace metal;

/// bf16-MTP-table variant of next_frame_input_f32: the 15 MTP codec_embedding tables (m0..m14) are
/// bfloat (BF16-source, stored bf16 to halve footprint), widened to fp32 on read (`float(bfloat)` =
/// bits<<16, EXACT for a bf16 source) so the gather-sum is BIT-IDENTICAL to the f32 path. `talkerEmb`
/// (cb0 table) stays f32 — it is also blitted raw elsewhere (the seed row), so it can't be bf16.
/// out[d] = talkerEmb[codes[0]·dim+d] + ttsPad[d] + Σ_{i=0..14} float(m_i[codes[i+1]·dim+d]).
///
/// Buffers: 0 codes [16] uint, 1 ttsPad [dim] float, 2 talkerEmb [vocab0,dim] float, 3 out [dim] float,
///          4..18 the 15 MTP codec_embedding tables [vocab,dim] bfloat (cb1..cb15)
/// Constants: 19 dim
kernel void next_frame_input_bf16w_f32(
    device const uint*   codes     [[buffer(0)]],
    device const float*  ttsPad    [[buffer(1)]],
    device const float*  talkerEmb [[buffer(2)]],
    device float*        out       [[buffer(3)]],
    device const bfloat* m0  [[buffer(4)]],  device const bfloat* m1  [[buffer(5)]],
    device const bfloat* m2  [[buffer(6)]],  device const bfloat* m3  [[buffer(7)]],
    device const bfloat* m4  [[buffer(8)]],  device const bfloat* m5  [[buffer(9)]],
    device const bfloat* m6  [[buffer(10)]], device const bfloat* m7  [[buffer(11)]],
    device const bfloat* m8  [[buffer(12)]], device const bfloat* m9  [[buffer(13)]],
    device const bfloat* m10 [[buffer(14)]], device const bfloat* m11 [[buffer(15)]],
    device const bfloat* m12 [[buffer(16)]], device const bfloat* m13 [[buffer(17)]],
    device const bfloat* m14 [[buffer(18)]],
    constant uint&       dim       [[buffer(19)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) return;
    float acc = talkerEmb[codes[0] * dim + d] + ttsPad[d];
    acc += float(m0[codes[1]  * dim + d]);  acc += float(m1[codes[2]  * dim + d]);
    acc += float(m2[codes[3]  * dim + d]);  acc += float(m3[codes[4]  * dim + d]);
    acc += float(m4[codes[5]  * dim + d]);  acc += float(m5[codes[6]  * dim + d]);
    acc += float(m6[codes[7]  * dim + d]);  acc += float(m7[codes[8]  * dim + d]);
    acc += float(m8[codes[9]  * dim + d]);  acc += float(m9[codes[10] * dim + d]);
    acc += float(m10[codes[11] * dim + d]); acc += float(m11[codes[12] * dim + d]);
    acc += float(m12[codes[13] * dim + d]); acc += float(m13[codes[14] * dim + d]);
    acc += float(m14[codes[15] * dim + d]);
    out[d] = acc;
}
