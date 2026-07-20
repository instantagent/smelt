#include <metal_stdlib>
using namespace metal;

/// RVQ dequant gather+sum (fp32). Produces the stacked latent q = [qFirst ; qRest] of shape
/// [2*dim, frames] (channel-major): qFirst[d,t] = firstEmb[code0[t]*dim+d] (semantic, 1
/// codebook), qRest[d,t] = sum over the `restCount` acoustic codebooks of
/// restEmb[k][code(k+1)[t]*dim+d]. The output_proj that follows ([firstProj|restProj] @ q)
/// is a separate matmul. Matches the gather/sum in Qwen3TTSCodec.rvqDequantize.
/// One thread per (d, t).
///
/// Buffers:
///   0: codes    [K, frames]  int   (K = 1 + restCount; row 0 semantic, rows 1.. acoustic)
///   1: firstEmb [firstN, dim] float
///   2: restEmb  [restCount, restN, dim] float (contiguous, stride restN*dim per codebook)
///   3: q        [2*dim, frames] float (out)
/// Constants:
///   4: dim, 5: frames, 6: restCount, 7: restN
kernel void rvq_gather_sum_f32(
    device const int*   codes     [[buffer(0)]],
    device const float* firstEmb  [[buffer(1)]],
    device const float* restEmb   [[buffer(2)]],
    device float*       q         [[buffer(3)]],
    constant uint&      dim       [[buffer(4)]],
    constant uint&      frames    [[buffer(5)]],
    constant uint&      restCount [[buffer(6)]],
    constant uint&      restN     [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint d = tid.y;
    uint t = tid.x;
    if (d >= dim || t >= frames) return;

    // Semantic: row 0 of codes.
    uint c0 = uint(codes[0 * frames + t]);
    q[d * frames + t] = firstEmb[c0 * dim + d];

    // Acoustic: sum over the restCount codebooks (rows 1..restCount), into row (dim+d).
    float acc = 0.0f;
    for (uint k = 0; k < restCount; k++) {
        uint ck = uint(codes[(k + 1) * frames + t]);
        acc += restEmb[k * restN * dim + ck * dim + d];
    }
    q[(dim + d) * frames + t] = acc;
}
