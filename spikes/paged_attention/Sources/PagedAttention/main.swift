import Foundation
import Metal

// =============================================================================
// Paged-attention spike
//
// Compares two single-head decode-step attention kernels:
//   1. flat_attention   — K, V are contiguous (N × HEAD_DIM)
//   2. paged_attention  — K, V are stored as fixed-size blocks; a block table
//                         maps logical position → physical block index
//
// Question: does block-table indirection add measurable overhead to the
// attention hot path? If paged ≤ ~1.1x flat, the paged-KV architecture is
// cheap enough to ship.
// =============================================================================

let HEAD_DIM = 128
let BLOCK_SIZE = 16
let testContexts = [256, 1024, 4096]
let benchIters = 200

// MARK: - Metal kernels

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM = 128;
constant uint BLOCK_SIZE = 16;

// One threadgroup per attention head, HEAD_DIM threads each (= 4 simdgroups
// of 32 on Apple Silicon). Sequential over positions; parallel over the
// head_dim within each position. Streaming softmax (FlashAttention-1 style)
// to avoid an O(N) scratchpad.
//
// scratch[]: 4 floats (per-simdgroup partial sums for the K · Q reduction).

inline float reduce_dot(float partial, threadgroup float *scratch,
                        uint tid, uint simd_id, uint simd_lane) {
    float simd_part = simd_sum(partial);
    if (simd_lane == 0) scratch[simd_id] = simd_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void flat_attention(
    device const half  *Q       [[buffer(0)]],
    device const half  *K       [[buffer(1)]],
    device const half  *V       [[buffer(2)]],
    device       float *out     [[buffer(3)]],
    constant     uint  &N       [[buffer(4)]],
    threadgroup  float *scratch [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    const float scale = 1.0f / sqrt((float)HEAD_DIM);
    float q_t = (float)Q[tid];

    // Streaming softmax state (per-thread accumulator over V).
    float m = -INFINITY;   // running max
    float l = 0.0f;        // running sum of exp
    float acc = 0.0f;      // running V accumulator (one head_dim slot per thread)

    for (uint i = 0; i < N; i++) {
        // 1. Score s_i = Q · K_i / sqrt(d)
        float partial = q_t * (float)K[i * HEAD_DIM + tid];
        float s = reduce_dot(partial, scratch, tid, simd_id, simd_lane) * scale;

        // 2. Streaming-softmax update.
        float m_new = max(m, s);
        float exp_old = exp(m - m_new);
        float w = exp(s - m_new);
        l = l * exp_old + w;
        acc = acc * exp_old + w * (float)V[i * HEAD_DIM + tid];
        m = m_new;
    }

    out[tid] = acc / l;
}

kernel void paged_attention(
    device const half   *Q          [[buffer(0)]],
    device const half   *K_blocks   [[buffer(1)]],
    device const half   *V_blocks   [[buffer(2)]],
    device const uint   *block_tab  [[buffer(3)]],
    device       float  *out        [[buffer(4)]],
    constant     uint   &N          [[buffer(5)]],
    threadgroup  float  *scratch    [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    const float scale = 1.0f / sqrt((float)HEAD_DIM);
    float q_t = (float)Q[tid];

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    uint cached_block_idx = UINT_MAX;
    uint physical_block = 0;

    for (uint i = 0; i < N; i++) {
        uint block_idx = i / BLOCK_SIZE;
        uint within = i - block_idx * BLOCK_SIZE;

        if (block_idx != cached_block_idx) {
            physical_block = block_tab[block_idx];
            cached_block_idx = block_idx;
        }
        uint k_off = (physical_block * BLOCK_SIZE + within) * HEAD_DIM + tid;

        float partial = q_t * (float)K_blocks[k_off];
        float s = reduce_dot(partial, scratch, tid, simd_id, simd_lane) * scale;

        float m_new = max(m, s);
        float exp_old = exp(m - m_new);
        float w = exp(s - m_new);
        l = l * exp_old + w;
        acc = acc * exp_old + w * (float)V_blocks[k_off];
        m = m_new;
    }

    out[tid] = acc / l;
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: kernelSource, options: nil)
let flatPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "flat_attention")!)
let pagedPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "paged_attention")!)

print("device:    \(device.name)")
print("HEAD_DIM:  \(HEAD_DIM)")
print("BLOCK_SIZE: \(BLOCK_SIZE) positions per block")
print("threadgroup width: \(HEAD_DIM) threads (\(HEAD_DIM / 32) simdgroups of 32)")

// MARK: - Helpers

extension Float {
    var halfBitPattern: UInt16 {
        // Naive fp32 → fp16 (round-to-nearest-even, no denormals).
        let bits = self.bitPattern
        let sign = UInt16((bits >> 31) & 0x1) << 15
        var exp = Int32((bits >> 23) & 0xFF) - 127 + 15
        var mant = (bits >> 13) & 0x3FF
        if exp <= 0 { return sign }                       // underflow → zero
        if exp >= 31 { return sign | 0x7C00 }             // overflow → inf
        let roundBit = (bits >> 12) & 0x1
        let stickyBits = bits & 0xFFF
        if roundBit == 1 && (stickyBits != 0 || (mant & 0x1) == 1) {
            mant += 1
            if mant == 0x400 { mant = 0; exp += 1 }
            if exp >= 31 { return sign | 0x7C00 }
        }
        return sign | (UInt16(exp) << 10) | UInt16(mant)
    }
}

func toHalfBuffer(_ values: [Float]) -> MTLBuffer {
    let halves = values.map { $0.halfBitPattern }
    return halves.withUnsafeBufferPointer { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * 2, options: .storageModeShared)!
    }
}

func uintBuffer(_ values: [UInt32]) -> MTLBuffer {
    return values.withUnsafeBufferPointer { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * 4, options: .storageModeShared)!
    }
}

func emptyFloatBuffer(_ count: Int) -> MTLBuffer {
    return device.makeBuffer(length: count * 4, options: .storageModeShared)!
}

// CPU reference attention (single head, full fp32).
func cpuReferenceAttention(Q: [Float], K: [Float], V: [Float], N: Int) -> [Float] {
    var scores = [Float](repeating: 0, count: N)
    let scale = 1.0 / sqrt(Float(HEAD_DIM))
    for i in 0..<N {
        var s: Float = 0
        for d in 0..<HEAD_DIM { s += Q[d] * K[i * HEAD_DIM + d] }
        scores[i] = s * scale
    }
    let m = scores.max()!
    var sumExp: Float = 0
    for i in 0..<N { scores[i] = exp(scores[i] - m); sumExp += scores[i] }
    for i in 0..<N { scores[i] /= sumExp }
    var out = [Float](repeating: 0, count: HEAD_DIM)
    for i in 0..<N {
        for d in 0..<HEAD_DIM { out[d] += scores[i] * V[i * HEAD_DIM + d] }
    }
    return out
}

func runFlat(Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, out: MTLBuffer, N: UInt32) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(flatPipe)
    enc.setBuffer(Q, offset: 0, index: 0)
    enc.setBuffer(K, offset: 0, index: 1)
    enc.setBuffer(V, offset: 0, index: 2)
    enc.setBuffer(out, offset: 0, index: 3)
    var n = N
    enc.setBytes(&n, length: 4, index: 4)
    enc.setThreadgroupMemoryLength(16, index: 0)  // 4 floats for simdgroup partials
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func runPaged(Q: MTLBuffer, Kb: MTLBuffer, Vb: MTLBuffer, table: MTLBuffer, out: MTLBuffer, N: UInt32) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pagedPipe)
    enc.setBuffer(Q, offset: 0, index: 0)
    enc.setBuffer(Kb, offset: 0, index: 1)
    enc.setBuffer(Vb, offset: 0, index: 2)
    enc.setBuffer(table, offset: 0, index: 3)
    enc.setBuffer(out, offset: 0, index: 4)
    var n = N
    enc.setBytes(&n, length: 4, index: 5)
    enc.setThreadgroupMemoryLength(16, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
    let p = buf.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    return zip(a, b).map { abs($0 - $1) }.max() ?? 0
}

// MARK: - Test runner

struct ContextResult {
    let N: Int
    let flatMaxDiff: Float
    let pagedMaxDiff: Float
    let flatMs: Double
    let pagedMs: Double
}

var results: [ContextResult] = []

for N in testContexts {
    print("")
    print("===== N = \(N) positions =====")

    // 1. Generate random data.
    var rng = SystemRandomNumberGenerator()
    func rand(_ n: Int) -> [Float] {
        (0..<n).map { _ in (Float(UInt32.random(in: 0..<10_000, using: &rng)) / 10_000.0) - 0.5 }
    }
    let Q = rand(HEAD_DIM)
    let K = rand(N * HEAD_DIM)
    let V = rand(N * HEAD_DIM)

    // 2. CPU reference.
    let ref = cpuReferenceAttention(Q: Q, K: K, V: V, N: N)

    // 3. Flat path.
    let qBuf = toHalfBuffer(Q)
    let kBuf = toHalfBuffer(K)
    let vBuf = toHalfBuffer(V)
    let outFlat = emptyFloatBuffer(HEAD_DIM)
    runFlat(Q: qBuf, K: kBuf, V: vBuf, out: outFlat, N: UInt32(N))
    let flatOut = readFloats(outFlat, count: HEAD_DIM)
    let flatDiff = maxAbsDiff(flatOut, ref)

    // 4. Paged path. Lay out K/V in blocks AND shuffle the block table to make
    //    sure the indirection is real (not a degenerate identity table).
    let numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    var physicalOrder = Array(0..<numBlocks)
    physicalOrder.shuffle()
    var logicalToPhysical = [UInt32](repeating: 0, count: numBlocks)
    for (logical, physical) in physicalOrder.enumerated() {
        logicalToPhysical[logical] = UInt32(physical)
    }

    // Reorder K and V so that block at physical index p contains the data for
    // logical block findIndex(of: p) in logicalToPhysical.
    var Kpaged = [Float](repeating: 0, count: numBlocks * BLOCK_SIZE * HEAD_DIM)
    var Vpaged = [Float](repeating: 0, count: numBlocks * BLOCK_SIZE * HEAD_DIM)
    for logical in 0..<numBlocks {
        let physical = Int(logicalToPhysical[logical])
        for w in 0..<BLOCK_SIZE {
            let pos = logical * BLOCK_SIZE + w
            for d in 0..<HEAD_DIM {
                let dst = (physical * BLOCK_SIZE + w) * HEAD_DIM + d
                let srcIdx = pos * HEAD_DIM + d
                if pos < N {
                    Kpaged[dst] = K[srcIdx]
                    Vpaged[dst] = V[srcIdx]
                }
            }
        }
    }

    let kbBuf = toHalfBuffer(Kpaged)
    let vbBuf = toHalfBuffer(Vpaged)
    let tableBuf = uintBuffer(logicalToPhysical)
    let outPaged = emptyFloatBuffer(HEAD_DIM)
    runPaged(Q: qBuf, Kb: kbBuf, Vb: vbBuf, table: tableBuf, out: outPaged, N: UInt32(N))
    let pagedOut = readFloats(outPaged, count: HEAD_DIM)
    let pagedDiff = maxAbsDiff(pagedOut, ref)

    print("correctness vs CPU reference (max abs diff):")
    print(String(format: "  flat:  %.5f", flatDiff))
    print(String(format: "  paged: %.5f", pagedDiff))
    let okFlat = flatDiff < 0.01
    let okPaged = pagedDiff < 0.01
    print("  flat:  \(okFlat ? "[PASS]" : "[FAIL]")")
    print("  paged: \(okPaged ? "[PASS]" : "[FAIL]")")

    // 5. Benchmark.
    // Warm-up.
    for _ in 0..<10 { runFlat(Q: qBuf, K: kBuf, V: vBuf, out: outFlat, N: UInt32(N)) }
    for _ in 0..<10 { runPaged(Q: qBuf, Kb: kbBuf, Vb: vbBuf, table: tableBuf, out: outPaged, N: UInt32(N)) }

    let tFlat0 = Date()
    for _ in 0..<benchIters { runFlat(Q: qBuf, K: kBuf, V: vBuf, out: outFlat, N: UInt32(N)) }
    let flatMs = Date().timeIntervalSince(tFlat0) / Double(benchIters) * 1000.0

    let tPaged0 = Date()
    for _ in 0..<benchIters { runPaged(Q: qBuf, Kb: kbBuf, Vb: vbBuf, table: tableBuf, out: outPaged, N: UInt32(N)) }
    let pagedMs = Date().timeIntervalSince(tPaged0) / Double(benchIters) * 1000.0

    print("timing (μ over \(benchIters) iters):")
    print(String(format: "  flat:  %.3f ms", flatMs))
    print(String(format: "  paged: %.3f ms  (%.2fx vs flat)", pagedMs, pagedMs / flatMs))

    results.append(ContextResult(N: N, flatMaxDiff: flatDiff, pagedMaxDiff: pagedDiff, flatMs: flatMs, pagedMs: pagedMs))
}

// MARK: - Summary

print("")
print("=================================================================")
print(" Summary")
print("=================================================================")
print("  N       flat-diff    paged-diff    flat ms    paged ms      ratio")
for r in results {
    let nStr = String(format: "%-4d", r.N)
    let fd = String(format: "%.5f", r.flatMaxDiff)
    let pd = String(format: "%.5f", r.pagedMaxDiff)
    let fms = String(format: "%.3f", r.flatMs)
    let pms = String(format: "%.3f", r.pagedMs)
    let ratio = String(format: "%.2f", r.pagedMs / r.flatMs)
    print("  \(nStr)    \(fd)       \(pd)    \(fms)       \(pms)       \(ratio)x")
}

let avgRatio = results.map { $0.pagedMs / $0.flatMs }.reduce(0, +) / Double(results.count)
print("")
print(String(format: "  paged/flat average ratio: %.2fx", avgRatio))

print("")
if avgRatio < 1.10 {
    print("  [PASS] Paged attention is within 10% of flat. **Block-table indirection")
    print("         is cheap enough to ship — paged-KV architecture viable for the hot path.**")
} else if avgRatio < 1.30 {
    print("  [INFO] Paged attention is 10-30% slower than flat. Usable but warrants")
    print("         optimization (cache the block-table walk, larger BLOCK_SIZE, etc).")
} else {
    print("  [FAIL] Paged attention is >30% slower than flat. Block-table indirection")
    print("         is a real cost on the hot path; rethink the architecture.")
}
