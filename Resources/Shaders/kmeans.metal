// kmeans.metal — GPU KMeans quantization for PAL-4 weight compression.
//
// Two kernels:
// 1. kmeans_quantize: Fused Lloyd loop. One threadgroup per group of rows.
//    Entire KMeans (init + iterate + sort + assign) runs in one dispatch.
//    Zero CPU readbacks between iterations.
//
// 2. kmeans_pack_u4: Pack uint8 assignments into packed bytes (2 per byte).

#include <metal_stdlib>
using namespace metal;

// ─── Constants ───
constant uint K = 16;               // Number of centroids (4-bit = 16 values)
constant uint MAX_ITER = 100;       // Max Lloyd iterations
constant float EPSILON = 1e-4;      // Convergence threshold (FP16 inputs: ~3 digits precision)
constant uint N_INIT = 3;           // Number of random initializations

// Barrier that fences both threadgroup and device memory.
// Required because groupAssign lives in device memory and is written/read
// across SIMD groups within the same threadgroup.
#define BARRIER_ALL threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)

// ─── Fused KMeans quantize ───
// One threadgroup per group. 256 threads.
// Input:  values[nGroups, groupElements] (float16)
// Output: assignments[nGroups, groupElements] (uint8, 0-15)
//         lut[nGroups, 16] (float16, sorted centroids)
// Constant: groupElements = groupSize * cols
//           seed = random seed for initialization

kernel void kmeans_quantize(
    device const half*   values       [[buffer(0)]],    // source weights
    device uint8_t*      assignments  [[buffer(1)]],    // output indices
    device half*         lut          [[buffer(2)]],    // output LUT [nGroups, 16]
    constant uint&       groupElements [[buffer(3)]],   // 16 * cols
    constant uint&       seed         [[buffer(4)]],    // random seed
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tgs     [[threads_per_threadgroup]]
) {
    // Pointers to this group's data
    device const half* groupVals = values + tgid * groupElements;
    device uint8_t* groupAssign = assignments + tgid * groupElements;
    device half* groupLUT = lut + tgid * K;

    // Threadgroup memory
    threadgroup float centroids[K];
    threadgroup float best_centroids[K];
    threadgroup float sums[K];
    threadgroup atomic_uint counts[K];
    threadgroup float best_sse;
    threadgroup float current_sse;

    if (tid == 0) best_sse = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiple initializations — keep best SSE
    for (uint init_round = 0; init_round < N_INIT; init_round++) {

        // --- Initialize centroids: parallel min/max, then linear spread ---
        float localMin = INFINITY;
        float localMax = -INFINITY;
        for (uint i = tid; i < groupElements; i += tgs) {
            float val = float(groupVals[i]);
            localMin = min(localMin, val);
            localMax = max(localMax, val);
        }
        // SIMD reduce
        localMin = simd_min(localMin);
        localMax = simd_max(localMax);
        // Cross-simdgroup reduce via threadgroup memory
        threadgroup float tgMin[32];  // up to 1024 threads / 32 SIMD width
        threadgroup float tgMax[32];
        uint simdGroup = tid / 32;
        uint simdLane = tid % 32;
        if (simdLane == 0) {
            tgMin[simdGroup] = localMin;
            tgMax[simdGroup] = localMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float vmin = tgMin[0];
            float vmax = tgMax[0];
            for (uint sg = 1; sg < tgs / 32; sg++) {
                vmin = min(vmin, tgMin[sg]);
                vmax = max(vmax, tgMax[sg]);
            }
            // Perturb with init_round for diversity across restarts
            float offset = float(init_round) * (vmax - vmin) / float(N_INIT * K);
            for (uint c = 0; c < K; c++) {
                centroids[c] = vmin + offset + float(c) * (vmax - vmin - offset) / float(K - 1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Lloyd iterations ---
        for (uint iter = 0; iter < MAX_ITER; iter++) {

            // Zero accumulators
            if (tid < K) {
                sums[tid] = 0.0;
                atomic_store_explicit(&counts[tid], 0, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Assign each element to nearest centroid + accumulate
            for (uint i = tid; i < groupElements; i += tgs) {
                float val = float(groupVals[i]);

                // Find nearest centroid (k=16 is small, brute force)
                uint best = 0;
                float bestDist = INFINITY;
                for (uint c = 0; c < K; c++) {
                    float d = (val - centroids[c]) * (val - centroids[c]);
                    if (d < bestDist) {
                        bestDist = d;
                        best = c;
                    }
                }

                groupAssign[i] = uint8_t(best);

                // Atomic accumulate for centroid update
                // Use atomic_uint for count, manual float accumulation via atomic
                atomic_fetch_add_explicit(&counts[best], 1, memory_order_relaxed);

                // For float sum: use simd reduce + threadgroup atomic
                // Simplified: each thread accumulates locally, then reduce
                // (For k=16 with 256 threads, contention is manageable)
            }
            BARRIER_ALL;  // fence device writes to groupAssign

            // Parallel sum: all threads accumulate, SIMD reduce, then cross-SIMD reduce
            float localSums[K];
            for (uint c = 0; c < K; c++) { localSums[c] = 0.0; }
            for (uint i = tid; i < groupElements; i += tgs) {
                localSums[groupAssign[i]] += float(groupVals[i]);
            }
            // SIMD reduce each centroid's sum across lanes
            for (uint c = 0; c < K; c++) {
                localSums[c] = simd_sum(localSums[c]);
            }
            // Cross-SIMD reduce: lane 0 of each SIMD group writes to threadgroup memory
            uint numSimdGroups = tgs / 32;
            threadgroup float partialSums[K * 32];  // K centroids × up to 32 SIMD groups (1024 threads)
            if (simdLane == 0) {
                for (uint c = 0; c < K; c++) {
                    partialSums[c * numSimdGroups + simdGroup] = localSums[c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < K) {
                float total = 0.0;
                for (uint sg = 0; sg < numSimdGroups; sg++) {
                    total += partialSums[tid * numSimdGroups + sg];
                }
                sums[tid] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Update centroids: normal update for non-empty clusters
            threadgroup float shifts[K];
            threadgroup atomic_uint emptyCount;
            if (tid == 0) atomic_store_explicit(&emptyCount, 0, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < K) {
                uint cnt = atomic_load_explicit(&counts[tid], memory_order_relaxed);
                float oldC = centroids[tid];
                if (cnt > 0) {
                    centroids[tid] = sums[tid] / float(cnt);
                } else {
                    // Mark that we need parallel empty cluster repair
                    atomic_fetch_add_explicit(&emptyCount, 1, memory_order_relaxed);
                    // Temporary: keep old centroid (will be replaced below)
                }
                shifts[tid] = (centroids[tid] - oldC) * (centroids[tid] - oldC);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Parallel empty cluster repair: ALL threads find max-error element
            if (atomic_load_explicit(&emptyCount, memory_order_relaxed) > 0) {
                // Phase 1: each thread finds its local max-error element
                float myMaxErr = 0.0;
                uint myMaxIdx = 0;
                for (uint i = tid; i < groupElements; i += tgs) {
                    float v = float(groupVals[i]);
                    uint a = groupAssign[i];
                    float e = (v - centroids[a]) * (v - centroids[a]);
                    if (e > myMaxErr) {
                        myMaxErr = e;
                        myMaxIdx = i;
                    }
                }

                // Phase 2: SIMD reduce to find max across lanes
                for (uint offset = 16; offset >= 1; offset >>= 1) {
                    float otherErr = simd_shuffle_down(myMaxErr, offset);
                    uint otherIdx = simd_shuffle_down(myMaxIdx, offset);
                    if (otherErr > myMaxErr) {
                        myMaxErr = otherErr;
                        myMaxIdx = otherIdx;
                    }
                }

                // Phase 3: cross-SIMD reduce via threadgroup memory
                // Reuse tgMin/tgMax arrays for error/index
                threadgroup float tgErr[32];
                threadgroup uint tgIdx[32];
                if (simdLane == 0) {
                    tgErr[simdGroup] = myMaxErr;
                    tgIdx[simdGroup] = myMaxIdx;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Phase 4: thread 0 finds global max and repairs empty clusters
                if (tid == 0) {
                    float globalMaxErr = tgErr[0];
                    uint globalMaxIdx = tgIdx[0];
                    for (uint sg = 1; sg < numSimdGroups; sg++) {
                        if (tgErr[sg] > globalMaxErr) {
                            globalMaxErr = tgErr[sg];
                            globalMaxIdx = tgIdx[sg];
                        }
                    }
                    // Assign the max-error element's value to each empty cluster
                    for (uint c = 0; c < K; c++) {
                        uint cnt = atomic_load_explicit(&counts[c], memory_order_relaxed);
                        if (cnt == 0) {
                            float oldC = centroids[c];
                            centroids[c] = float(groupVals[globalMaxIdx]);
                            shifts[c] = (centroids[c] - oldC) * (centroids[c] - oldC);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Convergence check: break if all centroids stopped moving
            threadgroup bool converged = false;
            if (tid == 0) {
                float maxShift = 0.0;
                for (uint c = 0; c < K; c++) {
                    if (shifts[c] > maxShift) maxShift = shifts[c];
                }
                converged = (maxShift < EPSILON * EPSILON);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (converged) break;
        }

        // Compute SSE for this initialization (parallel reduction)
        float localSSE = 0.0;
        for (uint i = tid; i < groupElements; i += tgs) {
            float val = float(groupVals[i]);
            float cen = centroids[groupAssign[i]];
            localSSE += (val - cen) * (val - cen);
        }
        localSSE = simd_sum(localSSE);
        threadgroup float tgSSE[32];
        if (simdLane == 0) { tgSSE[simdGroup] = localSSE; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float totalSSE = 0.0;
            for (uint sg = 0; sg < tgs / 32; sg++) { totalSSE += tgSSE[sg]; }
            current_sse = totalSSE;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Keep best initialization
        if (tid == 0 && current_sse < best_sse) {
            best_sse = current_sse;
            for (uint c = 0; c < K; c++) {
                best_centroids[c] = centroids[c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Restore best centroids and recompute assignments to match
    if (tid < K) {
        centroids[tid] = best_centroids[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Recompute assignments from best centroids (ensures LUT↔index consistency)
    for (uint i = tid; i < groupElements; i += tgs) {
        float val = float(groupVals[i]);
        uint best = 0;
        float bestDist = INFINITY;
        for (uint c = 0; c < K; c++) {
            float d = (val - centroids[c]) * (val - centroids[c]);
            if (d < bestDist) { bestDist = d; best = c; }
        }
        groupAssign[i] = uint8_t(best);
    }
    BARRIER_ALL;  // fence device writes to groupAssign before remap reads

    // --- Sort centroids by value (insertion sort, k=16) ---
    // Sort + LUT write on thread 0; remap parallelized across all threads.
    threadgroup uint invPerm_tg[K];
    if (tid == 0) {
        uint perm[K];
        for (uint i = 0; i < K; i++) perm[i] = i;

        // Insertion sort on centroids (k=16, trivial)
        for (uint i = 1; i < K; i++) {
            float key = centroids[perm[i]];
            uint keyIdx = perm[i];
            int j = int(i) - 1;
            while (j >= 0 && centroids[perm[j]] > key) {
                perm[j + 1] = perm[j];
                j--;
            }
            perm[j + 1] = keyIdx;
        }

        // Build inverse permutation and write sorted LUT
        for (uint i = 0; i < K; i++) {
            invPerm_tg[perm[i]] = i;
            groupLUT[i] = half(centroids[perm[i]]);
        }
    }
    BARRIER_ALL;  // fence invPerm_tg (TG) + groupLUT (device) before remap

    // Parallel remap assignments to sorted order
    for (uint i = tid; i < groupElements; i += tgs) {
        groupAssign[i] = uint8_t(invPerm_tg[groupAssign[i]]);
    }
}

// ─── Pack u4 indices ───
// Two uint8 assignments → one packed byte (low nibble first).
// Dispatch with threads = total_packed_bytes.

kernel void kmeans_pack_u4(
    device const uint8_t* assignments [[buffer(0)]],   // [N] uint8 indices (0-15)
    device uint8_t*       packed      [[buffer(1)]],   // [N/2] packed bytes
    constant uint&        totalElements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint halfN = totalElements / 2;
    if (tid >= halfN) return;

    uint8_t low  = assignments[tid * 2]     & 0x0F;
    uint8_t high = assignments[tid * 2 + 1] & 0x0F;
    packed[tid] = (high << 4) | low;
}
