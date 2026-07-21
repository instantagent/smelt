#include <metal_stdlib>
using namespace metal;

/// Applies the authored eight-neighbor blend and maintains the exact top four
/// joints in one pass. One thread owns a vertex, so every joint accumulation
/// and the final normalization retain their source order.
kernel void skin_transfer_top4_f32(
    device const float* sampledWeights [[buffer(0)]],
    device const uint* neighborOffsets [[buffer(1)]],
    device const uint* neighborQueryRows [[buffer(2)]],
    device const float* neighborBlends [[buffer(3)]],
    device ushort* outputJoints [[buffer(4)]],
    device float* outputWeights [[buffer(5)]],
    constant uint& vertexCount [[buffer(6)]],
    constant uint& jointCount [[buffer(7)]],
    constant uint& queryCount [[buffer(8)]],
    constant uint& jointMajor [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vertexCount) return;
    float bestWeights[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint bestJoints[4] = {0u, 0u, 0u, 0u};
    uint start = neighborOffsets[gid];
    uint end = neighborOffsets[gid + 1u];
    for (uint joint = 0u; joint < jointCount; ++joint) {
        float accumulated = 0.0f;
        for (uint neighbor = start; neighbor < end; ++neighbor) {
            uint query = neighborQueryRows[neighbor];
            uint sampledIndex = jointMajor != 0u
                ? joint * queryCount + query
                : query * jointCount + joint;
            float product = sampledWeights[sampledIndex]
                * neighborBlends[neighbor];
            accumulated += product;
        }
        uint insertion = 4u;
        for (uint lane = 0u; lane < 4u; ++lane) {
            if (accumulated > bestWeights[lane]
                || (accumulated == bestWeights[lane] && joint < bestJoints[lane])) {
                insertion = lane;
                break;
            }
        }
        if (insertion < 4u) {
            for (uint lane = 3u; lane > insertion; --lane) {
                bestWeights[lane] = bestWeights[lane - 1u];
                bestJoints[lane] = bestJoints[lane - 1u];
            }
            bestWeights[insertion] = accumulated;
            bestJoints[insertion] = joint;
        }
    }

    float sum = 0.0f;
    for (uint lane = 0u; lane < 4u; ++lane) {
        bestWeights[lane] = max(bestWeights[lane], 0.0f);
        sum += bestWeights[lane];
    }
    uint output = gid * 4u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        outputJoints[output + lane] = ushort(bestJoints[lane]);
        outputWeights[output + lane] = sum > 1e-8f
            ? bestWeights[lane] / sum
            : (lane == 0u ? 1.0f : 0.0f);
    }
}
