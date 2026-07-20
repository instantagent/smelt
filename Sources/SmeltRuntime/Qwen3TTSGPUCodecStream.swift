// Qwen3TTSCodecStream — exact chunked streaming decode of the RVQ codec (RVQ codes →
// 24 kHz waveform, incrementally). Plan: docs/qwen3-tts-streaming-plan.md.
//
// Contract: concat(decode(chunk₀), decode(chunk₁), …) == decodeCodec(all codes) BIT-EXACT
// for ANY chunking (gate: testTTSCodecStreamMatchesOfflineDecode). The codec is causal
// end-to-end, so each block only needs the trailing columns of its own input:
//   - stride-1 convs: cache kEff−1 = (k−1)·dilation input cols, zero-init (== the offline
//     causal zero left-pad byte-for-byte); chunk ops run conv1dValid over [cache ‖ new].
//   - preTransformer (8 layers, causal sliding window 72): per-layer cache of ≤71 input
//     rows; norm/qkv/RoPE/attn recomputed over [hist ‖ new] (values identical — every
//     kernel is position-independent with fixed within-row reduction order), trailing m
//     rows kept. RoPE rows sliced at the slice's GLOBAL positions [t0−hist, t0+m).
//   - transposed convs k=2r,s=r: output p sums inputs {⌊p/r⌋−1, ⌊p/r⌋} → cache 1 input
//     col; continuation chunks emit raw [r, (m+1)·r) (skip the already-emitted leading r,
//     drop the trailing r awaiting the next frame). The k=2,s=2 upsample pair is
//     frame-local (single contributor) — no cache.
//   - everything else (RVQ gather, k=1 convs, norms, snake, gelu/swiglu, γ-residuals,
//     clamp) is frame-local.
// All weights are resident (bound once at init — mapped package MTLBuffers, plus the few
// CPU-derived constants), so a chunk pays no per-call weight materialization.
//
// Matmul routing invariant (bit-exactness): ONLY the resident matmulDispatch (M==1→gemv,
// M>1→gemm; identical per-row reduction) — the naive matmul_f32 fallback has a different
// reduction order and is banned here (init preconditions the gemv/gemm psos + K%4==0).

import Foundation
import Metal

public final class Qwen3TTSCodecStream {
    private let gpu: Qwen3TTSGPU
    private let maxFrames: Int

    /// CPU-derived constants shared across streams on one Qwen3TTSGPU (cached on the gpu object):
    /// RVQ codebook embeddings, the output_proj concat, and the preTransformer RoPE tables. Pure
    /// functions of the package weights — deriving them was ~20ms of every stream init (a direct
    /// TTFA tax: generateStreaming builds a fresh stream per utterance).
    struct Shared {
        let fBuf: MTLBuffer, rBuf: MTLBuffer
        let firstN: Int, restN: Int
        let projConcatW: MTLBuffer, projZeroB: MTLBuffer
        let onesBuf: MTLBuffer
        var cosT: [Float], sinT: [Float], ropeFrames: Int
    }

    // MARK: resident derived constants
    // The compiled record table resolves every RAW weight by name via `gpu.codecStreamWeight` at realize
    // time (f32 raw, bf16/f16 widened once), so only the CPU-DERIVED constants (codebooks / output_proj
    // concat / RoPE tables / ones) and the
    // persistent caches live on the stream.

    private let fBuf: MTLBuffer, rBuf: MTLBuffer          // RVQ codebook embeddings (CPU-derived)
    private let firstN: Int, restN: Int
    private let projConcatW: MTLBuffer, projZeroB: MTLBuffer
    private let cosT: [Float], sinT: [Float]              // [maxFrames × 64] preTransformer RoPE
    private let onesBuf: MTLBuffer                        // residual-unit unscaled adds (≤1536)

    // MARK: streaming state

    private(set) public var framesDecoded = 0             // t0: global frames consumed
    private let cachePreConv: MTLBuffer                   // [512, 2]
    private let ptInputCache: [MTLBuffer]                 // 8 × [71 rows × 512], frame-major
    private let cacheCN: [MTLBuffer]                      // 2 × [1024, 6]
    private let cacheD0: MTLBuffer                        // [1024, 6]
    private let cacheTC: [MTLBuffer]                      // 4 × [dim_i, 1] (valid iff t0 > 0)
    private let cacheRU: [[MTLBuffer]]                    // 4 × 3 × [outDim_i, 6·dilation]
    private let cacheC6: MTLBuffer                        // [96, 6]

    private static let window = 72
    private static let dilations = [1, 3, 9]

    public init(gpu: Qwen3TTSGPU, maxFrames: Int = 256) throws {
        self.gpu = gpu
        self.maxFrames = maxFrames

        let rvqDim = 256
        // CPU-derived constants come from the per-GPU shared cache (built on first use; the rope
        // tables are extended in place if a later stream asks for a larger maxFrames).
        var shared: Shared
        if let s = gpu.codecStreamShared {
            shared = s
            if shared.ropeFrames < maxFrames {
                (shared.cosT, shared.sinT) = gpu.ropeTables(frames: maxFrames, headDim: 64)
                shared.ropeFrames = maxFrames
                gpu.codecStreamShared = shared
            }
        } else {
            let firstEmb = Qwen3TTSCodec.codebookEmbedding(
                embeddingSum: gpu.f32("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
                clusterUsage: gpu.f32("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"), dim: rvqDim)
            var restFlat: [Float] = []
            for k in 0..<15 {
                restFlat += Qwen3TTSCodec.codebookEmbedding(
                    embeddingSum: gpu.f32("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.embedding_sum"),
                    clusterUsage: gpu.f32("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.cluster_usage"), dim: rvqDim)
            }
            // output_proj: conv1d k=1, weight = concat(firstProj, restProj) [512, 512, 1] — same
            // CPU concat as decodeCodec, materialized once.
            let firstProj = gpu.f32("decoder.quantizer.rvq_first.output_proj.weight")
            let restProj = gpu.f32("decoder.quantizer.rvq_rest.output_proj.weight")
            let outDim = firstProj.count / rvqDim
            var projConcat = [Float](repeating: 0, count: outDim * 2 * rvqDim)
            for o in 0..<outDim {
                for d in 0..<rvqDim { projConcat[o * 2 * rvqDim + d] = firstProj[o * rvqDim + d] }
                for d in 0..<rvqDim { projConcat[o * 2 * rvqDim + rvqDim + d] = restProj[o * rvqDim + d] }
            }
            let (cosFull, sinFull) = gpu.ropeTables(frames: maxFrames, headDim: 64)
            shared = Shared(fBuf: gpu.bufF32(firstEmb), rBuf: gpu.bufF32(restFlat),
                            firstN: firstEmb.count / rvqDim, restN: (restFlat.count / 15) / rvqDim,
                            projConcatW: gpu.bufF32(projConcat),
                            projZeroB: gpu.bufF32([Float](repeating: 0, count: outDim)),
                            onesBuf: gpu.bufF32([Float](repeating: 1, count: 1536)),
                            cosT: cosFull, sinT: sinFull, ropeFrames: maxFrames)
            gpu.codecStreamShared = shared
        }
        fBuf = shared.fBuf; rBuf = shared.rBuf
        firstN = shared.firstN; restN = shared.restN
        projConcatW = shared.projConcatW; projZeroB = shared.projZeroB
        (cosT, sinT) = (shared.cosT, shared.sinT)
        onesBuf = shared.onesBuf

        // Caches. Conv caches are ALWAYS-FULL and zero-init (== offline causal zero pad);
        // the preTransformer row caches and transConv 1-col caches grow from empty.
        cachePreConv = gpu.outF32(512 * 2)
        ptInputCache = (0..<8).map { _ in gpu.outF32((Self.window - 1) * 512) }
        cacheCN = (0..<2).map { _ in gpu.outF32(1024 * 6) }
        cacheD0 = gpu.outF32(1024 * 6)
        cacheTC = (0..<4).map { i in gpu.outF32((i == 0 ? 1536 : 1536 >> i) * 1) }
        cacheRU = (0..<4).map { i in Self.dilations.map { d in gpu.outF32((1536 >> (i + 1)) * 6 * d) } }
        cacheC6 = gpu.outF32(96 * 6)

        // Validate + warm the compiled record table's full PSO set at construction (gemv/gemm plus
        // gemm_tn_f32 / transpose_f32 / sliding_attn_simd_f32), so an older package missing any kernel
        // fails loud here, not at first decode.
        _ = try streamPipelines()
    }

    // MARK: chunk decode

    /// Decode the next chunk of frames (each `[16]` codebook ids, as generateCodes emits) into
    /// frames.count × 1920 samples — bit-exact with the offline `decodeCodec` for the same total code
    /// sequence (gate: testCompiledStreamMatchesOfflineDecode). The codec compiles to a per-chunk record
    /// table (`Qwen3TTSCodecStreamEmitter`, docs/codec-c3-streaming-plan.md): once (m, t0) are fixed the
    /// chunk is "offline-static", so a FRESH literal table runs through the unchanged literal-only
    /// `SmeltCodecRecordRunner`. The persistent caches (`preConv`/per-layer `ptInput`/`cn`/`d0`/`tc`/`ru`/
    /// `c6`) are `.cache` slots bound to THIS stream's own buffers — concat (read old ‖ new → scratch) and
    /// update (scratch → cache) are hazard-tracked copy records within one serial encoder, so state carries
    /// across calls. Scratch comes from `gpu.outF32`'s size-keyed pool (recycled across chunks AND streams).
    public func decode(_ frames: [[Int]]) throws -> [Float] {
        let m = frames.count
        guard m > 0 else { return [] }
        let t0 = framesDecoded
        precondition(t0 + m <= maxFrames, "stream exceeds maxFrames=\(maxFrames) (t0=\(t0), chunk=\(m))")
        precondition(frames.allSatisfy { $0.count == 16 }, "each frame must carry 16 codebook ids")

        var codes = [Int32](repeating: 0, count: 16 * m)
        for f in 0..<m { for c in 0..<16 { codes[c * m + f] = Int32(frames[f][c]) } }
        // Preserve the gather's OOB guard (rvqGatherSumDispatch relied on this semantic/acoustic check).
        for t in 0..<m { precondition(codes[t] >= 0 && Int(codes[t]) < firstN, "semantic code OOB") }
        for k in 0..<15 { for t in 0..<m {
            precondition(codes[(k + 1) * m + t] >= 0 && Int(codes[(k + 1) * m + t]) < restN, "acoustic code OOB")
        } }

        let config = gpu.makeCodecConfig()
        let plan = Qwen3TTSCodecStreamEmitter.plan(m: m, t0: t0, config: config)
        let pipelines = try streamPipelines()
        // Run the chunk's record table in a `batched` scope so the ~250 per-chunk scratch buffers come
        // from `gpu.outF32`'s SIZE-KEYED pool (recycled across chunks AND streams — a fresh per-utterance
        // stream reuses the prior utterance's buffers, warming the TTFA first chunk), and the whole table
        // encodes into ONE serial encoder (hazard-tracked concat→update, == the offline decode's per-stage
        // batched semantics). Under capture/per-kernel-profiling (`!willRunBatched`) there is no shared
        // encoder, so fall back to an own command buffer with fresh scratch.
        let outBuf: MTLBuffer, outLen: Int
        if gpu.willRunBatched {
            // `batched` is rethrows and this closure does no throwing work, so no `try`.
            (outBuf, outLen) = gpu.batched {
                let buffers = realize(plan, codes: codes, config: config)
                SmeltCodecRecordRunner.encode(plan.records, pipelines: pipelines, buffers: buffers, into: gpu.batchEnc!)
                return (buffers[plan.outputSlot], plan.outputLength)
            }
        } else {
            let buffers = realize(plan, codes: codes, config: config)
            let cmd = gpu.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            SmeltCodecRecordRunner.encode(plan.records, pipelines: pipelines, buffers: buffers, into: enc)
            enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            (outBuf, outLen) = (buffers[plan.outputSlot], plan.outputLength)
        }
        framesDecoded = t0 + m
        return gpu.readF32(outBuf, outLen)
    }

    /// Materialize each slot → MTLBuffer: codec weights via `gpu.codecStreamWeight` (f32 raw / bf16-f16
    /// widened-once), the pre-derived constants + persistent caches from this stream's own buffers,
    /// the RoPE slice for the chunk's global positions, and `.scratch` via `gpu.outF32` (zeroed; pooled
    /// inside the `batched` scope). The persistent CACHES are the stream's own init-time buffers, never
    /// the pool, so the pool can never recycle a live cache as this-chunk scratch.
    private func realize(_ plan: Qwen3TTSCodecStreamPlan, codes: [Int32],
                         config c: Qwen3TTSCodecConfig) -> [MTLBuffer] {
        let hd = c.ptHeadDim
        return plan.slots.map { desc in
            switch desc {
            case .weight(let name): return gpu.codecStreamWeight(name)   // f32 raw; bf16/f16 (u4 pkg) widened once
            case .codes: return gpu.device.makeBuffer(bytes: codes, length: codes.count * 4, options: .storageModeShared)!
            case .codebookFirst: return fBuf
            case .codebookRestFlat: return rBuf
            case .projConcat: return projConcatW
            case .projZeroBias: return projZeroB
            case .ones: return onesBuf
            case .ropeCosSlice(let s, let len): return gpu.bufF32(Array(cosT[(s * hd)..<((s + len) * hd)]))
            case .ropeSinSlice(let s, let len): return gpu.bufF32(Array(sinT[(s * hd)..<((s + len) * hd)]))
            case .cache(let id): return cacheBuffer(id)
            case .scratch(let count): return gpu.outF32(count)
            }
        }
    }

    /// Resolve + cache the compiled-stream PSOs — the FULL `Qwen3TTSCodecStreamEmitter.pipelineNames`
    /// the record table dispatches, a SUPERSET of init's gemv/gemm guard (it also needs
    /// `gemm_tn_f32` / `transpose_f32` / `sliding_attn_simd_f32`). Fails loud on the first
    /// `decodeCompiled` if the package is missing any, and avoids re-resolving 17 PSOs per chunk on the
    /// streaming TTFA path.
    private var compiledPipelines: [MTLComputePipelineState]?
    private func streamPipelines() throws -> [MTLComputePipelineState] {
        if let p = compiledPipelines { return p }
        let p = try Qwen3TTSCodecStreamEmitter.pipelineNames.map { try gpu.pso($0) }
        compiledPipelines = p
        return p
    }

    private func cacheBuffer(_ id: Qwen3TTSCodecCacheID) -> MTLBuffer {
        switch id {
        case .preConv: return cachePreConv
        case .d0: return cacheD0
        case .c6: return cacheC6
        case .ptInput(let i): return ptInputCache[i]
        case .cn(let b): return cacheCN[b]
        case .tc(let i): return cacheTC[i]
        case .ru(let i, let ri): return cacheRU[i][ri]
        }
    }
}
