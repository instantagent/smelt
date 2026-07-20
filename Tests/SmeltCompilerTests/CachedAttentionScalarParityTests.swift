// CachedAttentionScalarParityTests — Phase 4 U1: validates the uncapped scalar streaming
// cached attention (causal_gqa_attn_cached_scalar_f32) that lets the compiled trunk prefill
// prompts of ANY length (the SIMD causal_gqa_attn_cached_f32 caps at a 2048-row cache via
// its threadgroup sc[2048] stage).
//
// Two proofs:
//   1) <=2048 cache: scalar-cached == SIMD-cached within fp-reassociation tolerance (~1e-6,
//      <=~14 ULP — the FMA-contraction class the existing scalar-vs-SIMD pair already has;
//      NOT byte-identical, so the <=2048 path is NEVER routed to scalar and keeps running
//      the unchanged SIMD kernel under every existing byte-exact gate).
//   2) >2048 cache: scalar-cached (at startPos 0) == the PROVEN uncapped non-cached scalar
//      kernel causal_gqa_attn_f32 (source-identical at startPos 0). This is the reliable
//      correctness oracle for the over-cap regime the SIMD kernel cannot represent at all.

import Metal
import XCTest

@testable import SmeltCompiler

final class CachedAttentionScalarParityTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device?.makeCommandQueue()
    }

    private func pso(_ file: String, _ fn: String) -> MTLComputePipelineState? {
        guard let src = loadMetalShaderSource(file),
              let lib = try? device.makeLibrary(source: src, options: nil),
              let f = lib.makeFunction(name: fn) else { return nil }
        return try? device.makeComputePipelineState(function: f)
    }

    // Deterministic LCG → ~[-2, 2), reproducible without touching global RNG.
    private func fill(_ n: Int, _ seed: inout UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            out[i] = Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        return out
    }

    private let heads = 16, kvHeads = 8, headDim = 128
    private var qDim: Int { heads * headDim }
    private var kvDim: Int { kvHeads * headDim }

    /// Dispatch a cached attention kernel (q,k,v,out,frames,heads,kvHeads,headDim,startPos).
    /// `simdGrid` uses the frames*32 SIMD grid; else the one-thread-per-(t,head) scalar grid.
    private func runCached(_ p: MTLComputePipelineState, simdGrid: Bool,
                           q: [Float], kC: [Float], vC: [Float], frames: Int, startPos: Int) -> [Float] {
        let qb = device.makeBuffer(bytes: q, length: q.count * 4, options: .storageModeShared)!
        let kb = device.makeBuffer(bytes: kC, length: kC.count * 4, options: .storageModeShared)!
        let vb = device.makeBuffer(bytes: vC, length: vC.count * 4, options: .storageModeShared)!
        let ob = device.makeBuffer(length: frames * qDim * 4, options: .storageModeShared)!
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(qb, offset: 0, index: 0); enc.setBuffer(kb, offset: 0, index: 1)
        enc.setBuffer(vb, offset: 0, index: 2); enc.setBuffer(ob, offset: 0, index: 3)
        var fr = UInt32(frames), hh = UInt32(heads), kvh = UInt32(kvHeads), hdd = UInt32(headDim)
        var sp = UInt32(startPos)
        enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
        enc.setBytes(&kvh, length: 4, index: 6); enc.setBytes(&hdd, length: 4, index: 7)
        enc.setBytes(&sp, length: 4, index: 8)
        if simdGrid {
            enc.dispatchThreads(MTLSize(width: frames * 32, height: heads, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            enc.dispatchThreads(MTLSize(width: frames, height: heads, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(frames, 32), height: 1, depth: 1))
        }
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return Array(UnsafeBufferPointer(
            start: ob.contents().bindMemory(to: Float.self, capacity: frames * qDim), count: frames * qDim))
    }

    private func tol(_ a: [Float], _ b: [Float]) -> (maxAbs: Float, maxRel: Float) {
        var maxAbs: Float = 0, maxRel: Float = 0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i]); maxAbs = max(maxAbs, d)
            maxRel = max(maxRel, d / max(abs(a[i]), abs(b[i]), 1e-30))
        }
        return (maxAbs, maxRel)
    }

    /// Proof 1: scalar-cached == SIMD-cached within tolerance, startPos+frames <= 2048.
    func testScalarCachedMatchesSimdCachedWithinTolerance() throws {
        guard let simd = pso("causal_gqa_attn_simd_f32.metal", "causal_gqa_attn_cached_f32"),
              let scalar = pso("causal_gqa_attn_simd_f32.metal", "causal_gqa_attn_cached_scalar_f32") else {
            throw XCTSkip("cached attention pipelines unavailable")
        }
        let cases: [(Int, Int)] = [(0, 24), (0, 64), (0, 1024), (100, 32), (1024, 200), (2008, 40)]
        var seed: UInt64 = 0x9E3779B97F4A7C15
        for (startPos, frames) in cases {
            let rows = startPos + frames
            let q = fill(frames * qDim, &seed), kC = fill(rows * kvDim, &seed), vC = fill(rows * kvDim, &seed)
            let a = runCached(simd, simdGrid: true, q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
            let b = runCached(scalar, simdGrid: false, q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
            let (maxAbs, maxRel) = tol(a, b)
            XCTAssertLessThan(maxAbs, 1e-4, "scalar vs simd maxAbs at sp=\(startPos) f=\(frames)")
            XCTAssertLessThan(maxRel, 1e-4, "scalar vs simd maxRel at sp=\(startPos) f=\(frames)")
        }
    }

    /// CPU reference: full-causal GQA softmax for `frames` chunk-local query rows at absolute
    /// base `startPos`, attending cache rows [0, startPos+t]. Same algorithm as both cached
    /// kernels (the GPU/CPU difference is only fp reassociation).
    private func cpuAttn(q: [Float], kC: [Float], vC: [Float], frames: Int, startPos: Int) -> [Float] {
        let scaling = 1 / Float(headDim).squareRoot(), group = heads / kvHeads
        var out = [Float](repeating: 0, count: frames * qDim)
        for t in 0..<frames {
            let absT = startPos + t
            for qh in 0..<heads {
                let kvh = qh / group, qb = t * qDim + qh * headDim
                var mx = -Float.infinity
                for s in 0...absT {
                    let kb = s * kvDim + kvh * headDim
                    var dot: Float = 0; for d in 0..<headDim { dot += q[qb + d] * kC[kb + d] }
                    mx = max(mx, dot * scaling)
                }
                var denom: Float = 0, acc = [Float](repeating: 0, count: headDim)
                for s in 0...absT {
                    let kb = s * kvDim + kvh * headDim
                    var dot: Float = 0; for d in 0..<headDim { dot += q[qb + d] * kC[kb + d] }
                    let e = exp(dot * scaling - mx); denom += e
                    let vb = s * kvDim + kvh * headDim
                    for d in 0..<headDim { acc[d] += e * vC[vb + d] }
                }
                let ob = t * qDim + qh * headDim
                for d in 0..<headDim { out[ob + d] = acc[d] / denom }
            }
        }
        return out
    }

    /// Proof 2: the scalar-cached kernel is correct for a cache BEYOND 2048 — the regime the
    /// SIMD kernel cannot represent. Mirrors PRODUCTION usage: a SMALL chunk of query rows
    /// (what `prefillTrunkChunked` dispatches per over-cap chunk, ≤ max_prefill_batch) at a high
    /// startPos, attending a >2048-row cache, vs a CPU reference. A large single-dispatch over
    /// ALL >2048 rows would hit the GPU watchdog (scalar is one thread per (t,head)); the
    /// chunked runtime never does that, so neither does this test.
    func testScalarCachedOverCapMatchesCPU() throws {
        guard let scalarCached = pso("causal_gqa_attn_simd_f32.metal", "causal_gqa_attn_cached_scalar_f32") else {
            throw XCTSkip("scalar cached pipeline unavailable")
        }
        var seed: UInt64 = 0x1357_9bdf_2468_ace0
        // (startPos, frames): a small chunk attending a cache of startPos+frames > 2048.
        for (startPos, frames) in [(2048, 8), (2200, 16), (3000, 4)] {
            let rows = startPos + frames
            let q = fill(frames * qDim, &seed), kC = fill(rows * kvDim, &seed), vC = fill(rows * kvDim, &seed)
            let gpu = runCached(scalarCached, simdGrid: false, q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
            let cpu = cpuAttn(q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
            XCTAssertTrue(gpu.allSatisfy { $0.isFinite } && gpu.contains { $0 != 0 },
                          "scalar-cached output not finite+nonzero at sp=\(startPos) f=\(frames)")
            let (maxAbs, maxRel) = tol(gpu, cpu)
            XCTAssertLessThan(maxAbs, 1e-4, "scalar-cached vs CPU maxAbs at sp=\(startPos) f=\(frames) (cache \(rows))")
            XCTAssertLessThan(maxRel, 1e-4, "scalar-cached vs CPU maxRel at sp=\(startPos) f=\(frames) (cache \(rows))")
        }
    }

    /// Proof 3: the runtime substitutes the scalar kernel onto the BAKED record's SIMD grid
    /// (frames*32 threads; lanes with t>=frames return early), NOT the scalar kernel's natural
    /// (frames,heads) grid. Prove they are byte-identical — each (t,head) is computed by exactly
    /// one thread either way — for an over-cap chunk. So the production over-launch dispatch
    /// produces exactly the CPU-validated (Proof 2) numbers.
    func testScalarCachedOverLaunchGridMatchesNaturalGrid() throws {
        guard let scalarCached = pso("causal_gqa_attn_simd_f32.metal", "causal_gqa_attn_cached_scalar_f32") else {
            throw XCTSkip("scalar cached pipeline unavailable")
        }
        var seed: UInt64 = 0x2468_ace0_1357_9bdf
        let (startPos, frames) = (2048, 8)   // over-cap cache 2056, small chunk
        let rows = startPos + frames
        let q = fill(frames * qDim, &seed), kC = fill(rows * kvDim, &seed), vC = fill(rows * kvDim, &seed)
        let natural = runCached(scalarCached, simdGrid: false, q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
        let overLaunch = runCached(scalarCached, simdGrid: true, q: q, kC: kC, vC: vC, frames: frames, startPos: startPos)
        let same = natural.withUnsafeBytes { a in overLaunch.withUnsafeBytes { b in a.elementsEqual(b) } }
        XCTAssertTrue(same, "scalar-cached over-launch grid != natural grid (over-cap)")
    }
}
