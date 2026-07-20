// TalkerSession — one talker generation call's state: resident weights,
// persistent buffers, KV cache, rope tables, position/history bookkeeping.
// Extracted from generateCodes' preamble so the monolithic `generateCodes`
// loop (the offline path + the tap/teacher gates) and the scheduled
// blocks (Qwen3TTSBlocks, driven by SmeltScheduledLoop) compose the SAME
// encode/host halves — the two paths cannot drift.
//
// Method discipline mirrors the scheduled-loop contract:
//   encode*  — GPU-encode only, must run inside an open batched scope
//   take/accept/advance — host halves, run after the scope committed

import Foundation
import Metal

final class TalkerSession {
    let gpu: Qwen3TTSGPU
    let shape: Qwen3TTSGPU.TalkerShape
    let sampling: Qwen3TTSSampler.Params?
    let seqLen: Int
    let maxFrames: Int
    let eos: Int

    let codecHeadWBuf: MTLBuffer
    let codecHeadDType: Qwen3TTSGPU.WeightPackDType
    let mtpProj: (w: MTLBuffer, b: MTLBuffer, dtype: Qwen3TTSGPU.WeightPackDType)?
    let lmHeadBufs: [MTLBuffer]
    let lmHeadDType: Qwen3TTSGPU.WeightPackDType
    let mtpEmbBufs: [MTLBuffer]
    let mtpEmbDType: Qwen3TTSGPU.WeightPackDType
    let mtpEmbRows: Int
    let talkerEmbRows: Int
    let talkerCodecEmbBuf: MTLBuffer

    // Persistent per-call buffers (reused each frame, no per-frame alloc).
    let inputsBuf: MTLBuffer
    let ttsPadBuf: MTLBuffer
    let codesBuf: MTLBuffer
    let historyBuf: MTLBuffer
    let cb0Buf: MTLBuffer
    let cb0LogitsBuf: MTLBuffer
    let cb0UniformBuf: MTLBuffer
    let mtpUniformsBuf: MTLBuffer
    let lastHiddenBuf: MTLBuffer

    let cosFull: [Float]
    let sinFull: [Float]

    // Compiled-trunk integration (B3.1/B3.2c): the talker trunk runs through the generic
    // compiler's decode/prefill tables (the package's own trunk/ sidecar), bridged into THIS
    // session's batched scope (gather→hiddenA port, encodeTrunk*, blit normOutBuf→lastHiddenBuf).
    // Resolved ONCE here (whole-request, never per-step), so the trunk's KV is the single source
    // across prefill + every decode. Any prompt length runs compiled — over-cap chunks route to
    // the uncapped scalar attention (Phase 4 U1).
    let compiledTrunk: SmeltRuntime
    private let trunkHiddenA: MTLBuffer   // port buffers, fetched AFTER capacity sizing
    private let trunkNormOut: MTLBuffer

    // Compiled-MTP integration (B3.2d): the MTP/code_predictor transformer runs through its OWN
    // co-resident compiled trunk (trunk-mtp/), bridged into encodeMTPSubPasses (blit seed/gather
    // rows → hiddenA, encodeTrunk*, blit normOutBuf → the lm_head input). Set up ONCE here
    // (frame-invariant rope + 16-row capacity); each frame re-prefills from pos 0, resetting its KV.
    private let compiledMTPTrunk: Qwen3TTSGPU.CompiledMTPTrunk

    // Loop bookkeeping (host halves mutate these).
    private(set) var pos: Int
    private(set) var historyLen = 0
    /// The cb0 the codec head selected for the CURRENT frame (consumed by
    /// the MTP encode) — the graph's codec-head → mtp-head tokens edge.
    var currentCb0 = -1
    /// The full 16-codebook frame the MTP produced (consumed by the talker's
    /// feedback build) — the mtp-head → talker feedback edge.
    var lastCodes: [Int] = []

    init(gpu: Qwen3TTSGPU, inputsBuf: MTLBuffer, seqLen: Int, ttsPadId: Int,
         maxFrames: Int, sampling: Qwen3TTSSampler.Params?) throws {
        self.gpu = gpu
        self.sampling = sampling
        self.seqLen = seqLen
        self.maxFrames = maxFrames
        let shape = try gpu.talkerShape()
        self.shape = shape
        // The [seqLen, hidden] prompt embeds — host-uploaded floats (generate(inputsEmbeds:)) or the
        // GPU front-end output (frontEndPrefillHiddenA, the generate(text:) path); the session is source-
        // agnostic from here.
        precondition(inputsBuf.length >= seqLen * shape.hidden * 4,
                     "inputsBuf \(inputsBuf.length)B < seqLen \(seqLen) × hidden \(shape.hidden) × 4")
        // Resident codec_head (fp32 — left unpacked by the fp16 builder) bound directly, so the
        // per-frame cb0 logits matmul reads the mapped buffer instead of re-uploading ~25 MB each frame.
        codecHeadWBuf = gpu.weight("talker.codec_head.weight")!
        codecHeadDType = gpu.weightDType("talker.codec_head.weight")
        // small_to_mtp_projection is absent on sizes where talker hidden == MTP hidden (0.6B);
        // talkerShape() already asserted that equality, so nil = identity, never a silent skip.
        if let w = gpu.weight("talker.code_predictor.small_to_mtp_projection.weight") {
            mtpProj = (w, gpu.weight("talker.code_predictor.small_to_mtp_projection.bias")!,
                       gpu.weightDType("talker.code_predictor.small_to_mtp_projection.weight"))
        } else {
            mtpProj = nil
        }
        lmHeadBufs = (0..<15).map { gpu.weight("talker.code_predictor.lm_head.\($0).weight")! }
        lmHeadDType = gpu.weightDType("talker.code_predictor.lm_head.0.weight")
        mtpEmbBufs = (0..<15).map { gpu.weight("talker.code_predictor.model.codec_embedding.\($0).weight")! }
        // The 15 MTP codec_embedding tables may be bf16 (footprint) — gathered via the bf16 widen
        // path. Enforce a uniform dtype across all 15 so a mixed/malformed package can't route some
        // tables through the wrong-dtype kernel (OOB / wrong values).
        let embDType = gpu.weightDType("talker.code_predictor.model.codec_embedding.0.weight")
        for i in 1..<15 {
            precondition(gpu.weightDType("talker.code_predictor.model.codec_embedding.\(i).weight") == embDType,
                         "MTP codec_embedding table \(i) dtype != table 0 (\(embDType))")
        }
        mtpEmbDType = embDType
        mtpEmbRows = gpu.weightShape("talker.code_predictor.model.codec_embedding.0.weight")?[0] ?? 0
        talkerEmbRows = gpu.weightShape("talker.model.codec_embedding.weight")?[0] ?? 0
        // Resident cb0 (talker codec_embedding) table for the on-GPU next-frame-input gather-sum.
        talkerCodecEmbBuf = gpu.weight("talker.model.codec_embedding.weight")!
        eos = Int(gpu.manifest.eosTokens.first ?? 2150)
        // Gather just the tts_pad row instead of copying the 1.2 GB text_embedding table (memoized).
        ttsPadBuf = gpu.bufF32(gpu.ttsPadEmbed(ttsPadId, shape: shape))
        self.inputsBuf = inputsBuf
        let device = gpu.device
        codesBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        historyBuf = device.makeBuffer(length: max(1, maxFrames) * 4, options: .storageModeShared)!
        cb0Buf = device.makeBuffer(length: 4, options: .storageModeShared)!
        cb0LogitsBuf = device.makeBuffer(length: shape.vocab * 4, options: .storageModeShared)!
        cb0UniformBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        mtpUniformsBuf = device.makeBuffer(length: 15 * 4, options: .storageModeShared)!
        // Persistent GPU hidden state: the prefill's last row, then each frame's decode output,
        // blitted here inside each batched scope so cb0/MTP/decode all read it resident.
        lastHiddenBuf = device.makeBuffer(length: shape.hidden * 4, options: .storageModeShared)!
        // The full RoPE table covers every absolute position (prefill + each decode), pinned into
        // the compiled trunk so its attention matches the Float-math reference numerics.
        let maxPos = seqLen + maxFrames
        (cosFull, sinFull) = gpu.ropeTables(frames: maxPos, headDim: shape.headDim, theta: shape.ropeTheta)
        pos = seqLen

        // Compiled trunks (B3.1): resolved once at init (whole-request), so the
        // trunk KV is the single source across prefill + every decode.
        let ports = try Self.makeCompiledTrunk(
            gpu: gpu, cosFull: cosFull, sinFull: sinFull,
            inputsBuf: inputsBuf, seqLen: seqLen, hidden: shape.hidden, maxPos: maxPos)
        compiledTrunk = ports.trunk
        trunkHiddenA = ports.hiddenA
        trunkNormOut = ports.normOut
        compiledMTPTrunk = try Self.makeCompiledMTPTrunk(gpu: gpu, shape: shape)
    }

    private struct CompiledTrunkPorts {
        let trunk: SmeltRuntime
        let hiddenA: MTLBuffer
        let normOut: MTLBuffer
    }

    /// Resolve + set up the compiled MTP trunk (the GPU's cached `<pkg>/trunk-mtp` runtime).
    /// THROWS if the package ships no trunk-mtp/ (cross-network dtype mix) or the scope won't batch (the hand MTP
    /// forward is retired — Phase 4 U4). The MTP runs a FIXED 16-position frame (2 seed rows +
    /// decodes at 2..15), so capacity + the
    /// Float-math RoPE (mtpHeadDim, θ — the SAME source prepareMTPFrame uses) are frame-
    /// invariant and set up ONCE; ports are fetched LAST (a grow swaps them). Each frame's
    /// re-prefill from pos 0 resets the KV, so one runtime is reused across all frames.
    private static func makeCompiledMTPTrunk(
        gpu: Qwen3TTSGPU, shape: Qwen3TTSGPU.TalkerShape
    ) throws -> Qwen3TTSGPU.CompiledMTPTrunk {
        // Phase 4 U4: the hand MTP forward is retired, so a missing trunk-mtp/ (cross-network dtype mix) or a
        // non-batched scope is a LOUD throw, not a silent hand fallback.
        guard gpu.willRunBatched else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "TTS generation requires batched mode (SMELT_DECODE_PROFILE must not be unbatched)")
        }
        guard let trunk = try gpu.resolveCompiledMTPTrunk() else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "package ships no trunk-mtp/ sidecar — talker and code_predictor must share one trunk dtype (f32/f16/bf16/u4)")
        }
        try trunk.ensurePrefillCapacity(seqLen: 2)
        try trunk.ensureContextCapacity(16)
        func trunkSlot(_ n: String) -> Int? { trunk.manifest.buffers.slots.first { $0.name == n }?.index }
        guard let cosI = trunkSlot("ropeCos"), let sinI = trunkSlot("ropeSin") else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "MTP trunk package missing ropeCos/ropeSin slots — cannot pin RoPE")
        }
        let (cosMtp, sinMtp) = gpu.ropeTables(frames: 16, headDim: shape.mtpHeadDim, theta: shape.ropeTheta)
        trunk.writeSlot(cosI, f32: cosMtp)
        trunk.writeSlot(sinI, f32: sinMtp)
        return Qwen3TTSGPU.CompiledMTPTrunk(
            trunk: trunk,
            hiddenA: try trunk.portSlotBuffer("hiddenA"),
            normOut: try trunk.portSlotBuffer("normOutBuf"))
    }

    /// Resolve the in-package compiled trunk (the GPU's cached `<pkg>/trunk` runtime).
    /// THROWS if the package ships no trunk/ (cross-network dtype mix) or `batched {}` won't batch (no encoder to
    /// ride — profiling). Variable-length: prompts of any length run compiled — the
    /// cross-chunk prefill routes over-cap chunks to the uncapped scalar attention (Phase 4
    /// U1), so there is no 2048 prompt cap. Order: capacity FIRST (grow-only), then pin the
    /// Float-math RoPE (a grow repopulates it), write the prompt embeds for the
    /// SINGLE-chunk fast path only, and fetch the port buffers LAST (a grow swaps them).
    private static func makeCompiledTrunk(
        gpu: Qwen3TTSGPU, cosFull: [Float], sinFull: [Float],
        inputsBuf: MTLBuffer, seqLen: Int, hidden: Int, maxPos: Int
    ) throws -> CompiledTrunkPorts {
        // The compiled trunk ships in the bf16 OR u4 package's trunk/ sidecar (B3.2c / Phase 3),
        // loaded ONCE on the GPU (resolveCompiledTrunk) and reused per request. Phase 4 U4: the hand
        // talker forward is retired, so a missing trunk/ (cross-network dtype mix) or a non-batched scope is a LOUD
        // throw. The guard precedes resolve, so it never triggers the trunk's ~400-pipeline load.
        guard gpu.willRunBatched else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "TTS generation requires batched mode (SMELT_DECODE_PROFILE must not be unbatched)")
        }
        guard let trunk = try gpu.resolveCompiledTrunk() else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "package ships no trunk/ sidecar — talker and code_predictor must share one trunk dtype (f32/f16/bf16/u4)")
        }
        try trunk.ensurePrefillCapacity(seqLen: seqLen)
        try trunk.ensureContextCapacity(maxPos)
        func trunkSlot(_ n: String) -> Int? {
            trunk.manifest.buffers.slots.first { $0.name == n }?.index
        }
        // Bit-exactness depends on pinning the Float-math RoPE after growth —
        // a trunk package without these slots cannot match, so fail closed rather
        // than run with the production (Double-math) fill.
        guard let cosI = trunkSlot("ropeCos"), let sinI = trunkSlot("ropeSin") else {
            throw Qwen3TTSGPU.GenerateError.unsupportedShape(
                "compiled trunk package missing ropeCos/ropeSin slots — cannot pin RoPE")
        }
        trunk.writeSlot(cosI, f32: cosFull)
        trunk.writeSlot(sinI, f32: sinFull)
        // Single-chunk fast path pre-fills the whole prompt into hiddenA by blitting inputsBuf (the
        // host-uploaded floats OR the GPU front-end output — both already populated + waited before
        // this init-time blit). The multi-chunk path (seqLen > maxBatch) would OVERFLOW the batch-sized
        // (maxBatch-row) hiddenA slab — it blits each chunk from inputsBuf in prefillTrunkChunked
        // instead, so skip the pre-fill there.
        let hiddenA = try trunk.portSlotBuffer("hiddenA")
        if seqLen <= trunk.maxPrefillBatchSize {
            gpu.blitCopy(inputsBuf, 0, hiddenA, 0, seqLen * hidden * 4)
        }
        return CompiledTrunkPorts(
            trunk: trunk,
            hiddenA: hiddenA,
            normOut: try trunk.portSlotBuffer("normOutBuf"))
    }

    /// The session's active batched compute encoder. Required wherever the compiled
    /// trunk runs — it was only resolved when `batched {}` will batch (see
    /// `makeCompiledTrunk`), so a nil here is a logic error, not a runtime case.
    private func requireBatchEnc() -> MTLComputeCommandEncoder {
        guard let enc = gpu.batchEnc else {
            preconditionFailure("compiled trunk needs a batched scope "
                + "(unbatched mode must have disabled it at session init)")
        }
        return enc
    }

    // MARK: - Encode halves (inside an open batched scope)

    /// Prompt prefill: full-sequence forward, last row blitted resident.
    func encodePrefill() throws {
        if seqLen <= compiledTrunk.maxPrefillBatchSize {
            // Single-chunk fast path: hiddenA pre-written at init; the compiled
            // prefill rides this batched scope, last row blitted resident.
            let enc = requireBatchEnc()
            try compiledTrunk.encodeTrunkPrefill(into: enc, seqLen: seqLen)
            gpu.blitCopy(trunkNormOut, (seqLen - 1) * shape.hidden * 4,
                         lastHiddenBuf, 0, shape.hidden * 4)
        } else {
            // Cross-chunk (B3.2b): own command buffers on the trunk queue (hiddenA
            // is rewritten between chunks, so chunks can't share this scope's
            // encoder). It blits the final hidden row into lastHiddenBuf and waits
            // on the last buffer, so the still-open batched scope's cb0 reads the
            // populated lastHiddenBuf.
            _ = try compiledTrunk.prefillTrunkChunked(
                source: inputsBuf, dest: lastHiddenBuf, seqLen: seqLen, hidden: shape.hidden)
        }
    }

    /// cb0 head (codec_head logits → processors → selection), folded into
    /// whatever scope just wrote lastHiddenBuf. Both greedy (cb0_argmax) and
    /// sampling (cb0_sample_topk) select on-GPU into cb0Buf — only `wantLogits`
    /// (a raw-logits tap, e.g. the teacher-forced gate) blits the logits back.
    /// `historyLen` is the prior-cb0 count BEFORE this frame (GPU repetition
    /// penalty / min_new_tokens).
    func encodeCb0(frame: Int, wantLogits: Bool = false) throws {
        let logitsBuf = try gpu.matmulDispatch(lastHiddenBuf, 1, shape.hidden, codecHeadWBuf, nil,
                                               shape.vocab, dtype: codecHeadDType)
        if wantLogits {
            gpu.blitCopy(logitsBuf, 0, cb0LogitsBuf, 0, shape.vocab * 4)
        }
        if let s = sampling {
            // Host-write this frame's cb0 uniform draw (codebook 0); the buffer write is visible to the
            // scope's dispatches at execution, parallel to encodeMTP's mtpUniformsBuf.
            cb0UniformBuf.contents().bindMemory(to: Float.self, capacity: 1)[0] =
                Qwen3TTSSampler.uniform(seed: s.seed, frame: frame, codebook: 0)
            try gpu.cb0SampleTopKDispatch(logits: logitsBuf, history: historyBuf, historyLen: historyLen,
                                          frame: frame, n: shape.vocab, uniforms: cb0UniformBuf, out: cb0Buf,
                                          eos: eos, temperature: s.talkerTemperature, topK: s.talkerTopK)
        } else {
            try gpu.cb0ArgmaxDispatch(logits: logitsBuf, history: historyBuf, historyLen: historyLen,
                                      frame: frame, n: shape.vocab, out: cb0Buf, eos: eos)
        }
    }

    /// Feedback build + trunk decode: the 16-code gather-sum (same
    /// accumulation order as the CPU reference, bit-exact) chains straight
    /// into the next decode; output blitted resident. Host-writes codesBuf
    /// from `lastCodes` first (buffer writes are visible to the scope's
    /// dispatches at execution).
    func encodeFeedbackAndDecode() throws {
        precondition(lastCodes.count == 16, "feedback needs the previous frame's 16 codes")
        let cptr = codesBuf.contents().bindMemory(to: UInt32.self, capacity: 16)
        for i in 0..<16 { cptr[i] = UInt32(lastCodes[i]) }

        // Gather the feedback embedding STRAIGHT into the trunk's hiddenA port (bind, no
        // blit), run the compiled decode at this absolute position, blit the hidden out. The
        // gather and the trunk dispatches share the one batched compute encoder, so the
        // gather's write to hiddenA is visible to the trunk's first read (serial order).
        let enc = requireBatchEnc()
        try gpu.nextFrameInputDispatch(codes: codesBuf, ttsPad: ttsPadBuf, talkerEmb: talkerCodecEmbBuf,
                                       mtpTables: mtpEmbBufs, out: trunkHiddenA, dim: shape.hidden,
                                       mtpDType: mtpEmbDType, talkerRows: talkerEmbRows, mtpRows: mtpEmbRows)
        try compiledTrunk.encodeTrunkDecode(into: enc, tokenId: 0, position: Int32(pos))
        gpu.blitCopy(trunkNormOut, 0, lastHiddenBuf, 0, shape.hidden * 4)
    }

    /// MTP sub-pass fan-out for the current frame (`currentCb0`). Sampling
    /// paths host-precompute the 15 sub-pass uniforms first so the GPU chain
    /// draws each codebook without a per-sub-pass round-trip.
    func encodeMTP(_ state: Qwen3TTSGPU.MTPFrameState, frame: Int) throws {
        var mtpSampling: (uniforms: MTLBuffer, temperature: Float, topK: Int)? = nil
        if let s = sampling {
            let up = mtpUniformsBuf.contents().bindMemory(to: Float.self, capacity: 15)
            for gs in 0..<15 { up[gs] = Qwen3TTSSampler.uniform(seed: s.seed, frame: frame, codebook: gs + 1) }
            mtpSampling = (mtpUniformsBuf, s.subtalkerTemperature, s.subtalkerTopK)
        }
        try gpu.encodeMTPSubPasses(state, talkerHiddenBuf: lastHiddenBuf, cb0: currentCb0,
                                   talkerCodecEmbBuf: talkerCodecEmbBuf, mtpEmbBufs: mtpEmbBufs,
                                   proj: mtpProj, lmHeadDType: lmHeadDType,
                                   lmHeadBufs: lmHeadBufs, shape: shape,
                                   mtpEmbDType: mtpEmbDType, mtpSampling: mtpSampling,
                                   mtpTrunk: compiledMTPTrunk)
    }

    func prepareMTPFrame(teacherCodes: [Int]? = nil, wantTaps: Bool = false) -> Qwen3TTSGPU.MTPFrameState {
        gpu.prepareMTPFrame(shape: shape, teacherCodes: teacherCodes, wantTaps: wantTaps)
    }

    // MARK: - Host halves (after the scope committed)

    /// Read this frame's cb0 — the GPU selection (argmax greedy / sample_topk sampling) written into
    /// cb0Buf by encodeCb0's scope, which has committed by the time this is called.
    func takeCb0() -> Int {
        Int(cb0Buf.contents().bindMemory(to: UInt32.self, capacity: 1)[0])
    }

    /// Append a non-EOS cb0 to the GPU-resident history (repetition penalty).
    func acceptCb0(_ cb0: Int) {
        historyBuf.contents().bindMemory(to: UInt32.self, capacity: maxFrames)[historyLen] = UInt32(cb0)
        historyLen += 1
        currentCb0 = cb0
    }

    func readMTPCodes(_ state: Qwen3TTSGPU.MTPFrameState) -> [Int] {
        gpu.readMTPFrame(state)
    }

    func advancePosition() {
        pos += 1
    }
}
