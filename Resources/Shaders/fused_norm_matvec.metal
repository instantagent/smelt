#include <metal_stdlib>
using namespace metal;

// ─── Fused norm + consumer via device-scope atomic signaling ───
// TG 0 computes the RMS norm scale, writes it, atomically signals "ready".
// All other TGs spin on the atomic flag, then proceed with inline normalization.
// ONE dispatch. No redundant per-TG norm. No intermediate buffer.

constant uint ROWS_PER_SG = 4;
constant uint ROWS_PER_TG = 8;
constant uint FC_COLS       [[function_constant(0)]];
constant uint FC_GROUP_SIZE [[function_constant(1)]];

// ─── Fused norm + affine gate+up+SwiGLU (atomic signaling) ───
// Post-attention norm is the sole consumer case: no side-effect write needed.
// TG 0 computes scale, signals via atomic. Other TGs normalize inline.
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void atomic_norm_affine_gate_up_swiglu(
    device const half*    norm_input   [[buffer(0)]],   // [dim] raw hidden state
    device const half*    norm_weight  [[buffer(1)]],   // [dim] RMS norm (1+w)
    device const uint8_t* gate_weights [[buffer(2)]],
    device const half*    gate_scales  [[buffer(3)]],
    device const half*    gate_biases  [[buffer(4)]],
    device const uint8_t* up_weights   [[buffer(5)]],
    device const half*    up_scales    [[buffer(6)]],
    device const half*    up_biases    [[buffer(7)]],
    device half*          output       [[buffer(8)]],
    device float*         scratch      [[buffer(9)]],   // [2]: scratch[0]=scale, scratch[1]=ready flag (as float)
    constant uint&        numRows      [[buffer(10)]],
    constant float&       eps          [[buffer(11)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // ═══ TG 0: compute norm scale and signal ═══
    device atomic_uint* ready_flag = (device atomic_uint*)(scratch + 1);

    if (tgid == 0) {
        // Reset flag first (in case of previous use)
        if (tid == 0) atomic_store_explicit(ready_flag, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_device);

        float sum_sq = 0.0f;
        for (uint i = tid; i < FC_COLS; i += 64) {
            float v = float(norm_input[i]);
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        threadgroup float sg_sums[8];
        if (tid % 32 == 0) sg_sums[tid / 32] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0;
            for (uint s = 0; s < 2; s++) total += sg_sums[s]; // 64 threads = 2 SGs
            scratch[0] = rsqrt(total / float(FC_COLS) + eps);
            // Signal: scale is ready
            threadgroup_barrier(mem_flags::mem_device);
            atomic_store_explicit(ready_flag, 1u, memory_order_relaxed);
        }
    }

    // ═══ All other TGs: spin until scale is ready ═══
    if (tgid != 0 && tid == 0) {
        while (atomic_load_explicit(ready_flag, memory_order_relaxed) == 0u) {
            // spin
        }
    }
    // Broadcast within TG via barrier
    threadgroup float tg_scale = 0.0f;
    if (tid == 0) tg_scale = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = tg_scale;

    // ═══ Gate+up+SwiGLU with inline normalization ═══
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;
    uint u16PerRow = halfCols / 2;

    float g0=0,g1=0,g2=0,g3=0;
    float u0=0,u1=0,u2=0,u3=0;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * halfCols);
    device const uint16_t* gr1 = (sgValidRows > 1) ? gr0 + u16PerRow : gr0;
    device const uint16_t* gr2 = (sgValidRows > 2) ? gr0 + 2 * u16PerRow : gr0;
    device const uint16_t* gr3 = (sgValidRows > 3) ? gr0 + 3 * u16PerRow : gr0;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * halfCols);
    device const uint16_t* ur1 = (sgValidRows > 1) ? ur0 + u16PerRow : ur0;
    device const uint16_t* ur2 = (sgValidRows > 2) ? ur0 + 2 * u16PerRow : ur0;
    device const uint16_t* ur3 = (sgValidRows > 3) ? ur0 + 3 * u16PerRow : ur0;

    device const half* gs0=gate_scales+sgBaseRow*numColGroups; device const half* gs1=gs0+numColGroups; device const half* gs2=gs0+2*numColGroups; device const half* gs3=gs0+3*numColGroups;
    device const half* gb0=gate_biases+sgBaseRow*numColGroups; device const half* gb1=gb0+numColGroups; device const half* gb2=gb0+2*numColGroups; device const half* gb3=gb0+3*numColGroups;
    device const half* us0=up_scales+sgBaseRow*numColGroups;   device const half* us1=us0+numColGroups; device const half* us2=us0+2*numColGroups; device const half* us3=us0+3*numColGroups;
    device const half* ub0=up_biases+sgBaseRow*numColGroups;   device const half* ub1=ub0+numColGroups; device const half* ub2=ub0+2*numColGroups; device const half* ub3=ub0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint gidx = col / FC_GROUP_SIZE;

        // Normalize inline: x = raw * scale * (1 + weight)
        device const half* xp = norm_input + col;
        device const half* wp = norm_weight + col;
        float x0 =float(xp[0]) *scale*(1.f+float(wp[0]));
        float x1 =float(xp[1]) *scale*(1.f+float(wp[1]));
        float x2 =float(xp[2]) *scale*(1.f+float(wp[2]));
        float x3 =float(xp[3]) *scale*(1.f+float(wp[3]));
        float x4 =float(xp[4]) *scale*(1.f+float(wp[4]));
        float x5 =float(xp[5]) *scale*(1.f+float(wp[5]));
        float x6 =float(xp[6]) *scale*(1.f+float(wp[6]));
        float x7 =float(xp[7]) *scale*(1.f+float(wp[7]));
        float x8 =float(xp[8]) *scale*(1.f+float(wp[8]));
        float x9 =float(xp[9]) *scale*(1.f+float(wp[9]));
        float x10=float(xp[10])*scale*(1.f+float(wp[10]));
        float x11=float(xp[11])*scale*(1.f+float(wp[11]));
        float x12=float(xp[12])*scale*(1.f+float(wp[12]));
        float x13=float(xp[13])*scale*(1.f+float(wp[13]));
        float x14=float(xp[14])*scale*(1.f+float(wp[14]));
        float x15=float(xp[15])*scale*(1.f+float(wp[15]));

        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { g0+=float(gs0[gidx])*D16(gr0+j*4)+float(gb0[gidx])*xsum; u0+=float(us0[gidx])*D16(ur0+j*4)+float(ub0[gidx])*xsum; }
        if (sgValidRows>1) { g1+=float(gs1[gidx])*D16(gr1+j*4)+float(gb1[gidx])*xsum; u1+=float(us1[gidx])*D16(ur1+j*4)+float(ub1[gidx])*xsum; }
        if (sgValidRows>2) { g2+=float(gs2[gidx])*D16(gr2+j*4)+float(gb2[gidx])*xsum; u2+=float(us2[gidx])*D16(ur2+j*4)+float(ub2[gidx])*xsum; }
        if (sgValidRows>3) { g3+=float(gs3[gidx])*D16(gr3+j*4)+float(gb3[gidx])*xsum; u3+=float(us3[gidx])*D16(ur3+j*4)+float(ub3[gidx])*xsum; }
#undef D16
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + exp(-g0)) * u0);
        if (sgValidRows>1) output[sgBaseRow+1] = half(g1 / (1.0f + exp(-g1)) * u1);
        if (sgValidRows>2) output[sgBaseRow+2] = half(g2 / (1.0f + exp(-g2)) * u2);
        if (sgValidRows>3) output[sgBaseRow+3] = half(g3 / (1.0f + exp(-g3)) * u3);
    }
}

// ─── Fused norm + affine matvec (atomic signaling, writes norm output) ───
// For input_norm → first matvec: other matvecs also need the norm output.
// TG 0 computes scale + writes full normalized output. Other TGs spin, then matvec.

kernel void atomic_norm_affine_matvec(
    device const half*    norm_input   [[buffer(0)]],
    device const half*    norm_weight  [[buffer(1)]],
    device half*          norm_output  [[buffer(2)]],   // side-effect: full normalized output
    device const uint8_t* weights      [[buffer(3)]],
    device const half*    scales       [[buffer(4)]],
    device const half*    biases       [[buffer(5)]],
    device half*          output       [[buffer(6)]],
    device float*         scratch      [[buffer(7)]],   // [2]: scale + ready flag
    constant uint&        numRows      [[buffer(8)]],
    constant float&       eps          [[buffer(9)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    device atomic_uint* ready_flag = (device atomic_uint*)(scratch + 1);

    if (tgid == 0) {
        if (tid == 0) atomic_store_explicit(ready_flag, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_device);

        float sum_sq = 0.0f;
        for (uint i = tid; i < FC_COLS; i += 64) {
            float v = float(norm_input[i]);
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        threadgroup float sg_sums[8];
        if (tid % 32 == 0) sg_sums[tid / 32] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale;
        if (tid == 0) {
            float total = 0;
            for (uint s = 0; s < 2; s++) total += sg_sums[s];
            scale = rsqrt(total / float(FC_COLS) + eps);
            scratch[0] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scale = scratch[0];

        // Write full normalized output (other matvecs need it)
        for (uint i = tid; i < FC_COLS; i += 64) {
            norm_output[i] = half(float(norm_input[i]) * scale * (1.0f + float(norm_weight[i])));
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) threadgroup_barrier(mem_flags::mem_device);
            atomic_store_explicit(ready_flag, 1u, memory_order_relaxed);
    }

    // All other TGs: spin until ready
    if (tgid != 0 && tid == 0) {
        while (atomic_load_explicit(ready_flag, memory_order_relaxed) == 0u) {}
    }
    threadgroup float tg_scale = 0.0f;
    if (tid == 0) tg_scale = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = tg_scale;

    // Matvec with inline normalization
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;
    uint u16PerRow = halfCols / 2;

    float acc0=0, acc1=0, acc2=0, acc3=0;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * halfCols);
    device const uint16_t* r1 = (sgValidRows > 1) ? r0 + u16PerRow : r0;
    device const uint16_t* r2 = (sgValidRows > 2) ? r0 + 2 * u16PerRow : r0;
    device const uint16_t* r3 = (sgValidRows > 3) ? r0 + 3 * u16PerRow : r0;

    device const half* sc0=scales+sgBaseRow*numColGroups; device const half* sc1=sc0+numColGroups; device const half* sc2=sc0+2*numColGroups; device const half* sc3=sc0+3*numColGroups;
    device const half* bi0=biases+sgBaseRow*numColGroups; device const half* bi1=bi0+numColGroups; device const half* bi2=bi0+2*numColGroups; device const half* bi3=bi0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint g = col / FC_GROUP_SIZE;

        device const half* xp = norm_input + col;
        device const half* wp = norm_weight + col;
        float x0 =float(xp[0]) *scale*(1.f+float(wp[0]));
        float x1 =float(xp[1]) *scale*(1.f+float(wp[1]));
        float x2 =float(xp[2]) *scale*(1.f+float(wp[2]));
        float x3 =float(xp[3]) *scale*(1.f+float(wp[3]));
        float x4 =float(xp[4]) *scale*(1.f+float(wp[4]));
        float x5 =float(xp[5]) *scale*(1.f+float(wp[5]));
        float x6 =float(xp[6]) *scale*(1.f+float(wp[6]));
        float x7 =float(xp[7]) *scale*(1.f+float(wp[7]));
        float x8 =float(xp[8]) *scale*(1.f+float(wp[8]));
        float x9 =float(xp[9]) *scale*(1.f+float(wp[9]));
        float x10=float(xp[10])*scale*(1.f+float(wp[10]));
        float x11=float(xp[11])*scale*(1.f+float(wp[11]));
        float x12=float(xp[12])*scale*(1.f+float(wp[12]));
        float x13=float(xp[13])*scale*(1.f+float(wp[13]));
        float x14=float(xp[14])*scale*(1.f+float(wp[14]));
        float x15=float(xp[15])*scale*(1.f+float(wp[15]));

        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        acc0 += float(sc0[g]) * D16(r0+j*4) + float(bi0[g]) * xsum;
        if (sgValidRows>1) acc1 += float(sc1[g]) * D16(r1+j*4) + float(bi1[g]) * xsum;
        if (sgValidRows>2) acc2 += float(sc2[g]) * D16(r2+j*4) + float(bi2[g]) * xsum;
        if (sgValidRows>3) acc3 += float(sc3[g]) * D16(r3+j*4) + float(bi3[g]) * xsum;
#undef D16
    }

    acc0=simd_sum(acc0); acc1=simd_sum(acc1); acc2=simd_sum(acc2); acc3=simd_sum(acc3);
    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        if (sgValidRows>1) output[sgBaseRow+1] = half(acc1);
        if (sgValidRows>2) output[sgBaseRow+2] = half(acc2);
        if (sgValidRows>3) output[sgBaseRow+3] = half(acc3);
    }
}
