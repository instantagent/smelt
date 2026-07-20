#include <metal_stdlib>
using namespace metal;

/// Next-frame talker input = sum of the frame's 16 codebook embeddings + tts_pad, on-GPU.
/// out[d] = talkerEmb[codes[0]·dim + d] + ttsPad[d] + Σ_{i=0..14} mtp_i[codes[i+1]·dim + d].
///
/// Replaces the CPU gather-sum (Qwen3TTSGenerator.nextFrameInput) so the frame's codes never round-trip
/// to the host and the next decode reads this buffer directly — and it lets the driver drop the 251 MB
/// `mtpEmbs` [[Float]] materialization (the 15 MTP codec_embedding tables are already resident GPU
/// buffers). The accumulation order — talkerEmb + ttsPad, then mtp_0 … mtp_14 — is IDENTICAL to the CPU
/// reference, so this is bit-exact (not merely fp32-equivalent). One thread per output element.
///
/// Buffers: 0 codes [16] uint, 1 ttsPad [dim], 2 talkerEmb [vocab0,dim], 3 out [dim],
///          4..18 the 15 MTP codec_embedding tables [vocab,dim] (cb1..cb15)
/// Constants: 19 dim
kernel void next_frame_input_f32(
    device const uint*  codes     [[buffer(0)]],
    device const float* ttsPad    [[buffer(1)]],
    device const float* talkerEmb [[buffer(2)]],
    device float*       out       [[buffer(3)]],
    device const float* m0  [[buffer(4)]],  device const float* m1  [[buffer(5)]],
    device const float* m2  [[buffer(6)]],  device const float* m3  [[buffer(7)]],
    device const float* m4  [[buffer(8)]],  device const float* m5  [[buffer(9)]],
    device const float* m6  [[buffer(10)]], device const float* m7  [[buffer(11)]],
    device const float* m8  [[buffer(12)]], device const float* m9  [[buffer(13)]],
    device const float* m10 [[buffer(14)]], device const float* m11 [[buffer(15)]],
    device const float* m12 [[buffer(16)]], device const float* m13 [[buffer(17)]],
    device const float* m14 [[buffer(18)]],
    constant uint&      dim       [[buffer(19)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) return;
    float acc = talkerEmb[codes[0] * dim + d] + ttsPad[d];
    acc += m0[codes[1]  * dim + d];  acc += m1[codes[2]  * dim + d];
    acc += m2[codes[3]  * dim + d];  acc += m3[codes[4]  * dim + d];
    acc += m4[codes[5]  * dim + d];  acc += m5[codes[6]  * dim + d];
    acc += m6[codes[7]  * dim + d];  acc += m7[codes[8]  * dim + d];
    acc += m8[codes[9]  * dim + d];  acc += m9[codes[10] * dim + d];
    acc += m10[codes[11] * dim + d]; acc += m11[codes[12] * dim + d];
    acc += m12[codes[13] * dim + d]; acc += m13[codes[14] * dim + d];
    acc += m14[codes[15] * dim + d];
    out[d] = acc;
}
