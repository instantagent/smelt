// Qwen3TTSGPU — talker + MTP code generation (P6-U4c) and the full text-embeds→waveform
// driver entry. The talker and the MTP/code_predictor transformers run through their OWN
// compiled trunk sidecars (trunk/, trunk-mtp/) via TalkerSession — KV-cached, O(T) not O(T²),
// weights bound resident from the package. The orchestration here (cb0/MTP fan-out, next-frame
// gather-sum, sampling) wraps those compiled forwards. Gated by codes == gen_codes vs PyTorch.

import Foundation
import Metal

extension Qwen3TTSGPU {

    public enum GenerateError: Error, CustomStringConvertible {
        case noFramesGenerated
        case noTokenizer
        case missingWeight(String)
        case unsupportedShape(String)
        case malformedSchedule(String)
        public var description: String {
            switch self {
            case .noFramesGenerated: return "generate produced no frames (EOS before any frame or maxFrames=0)"
            case .noTokenizer: return "generate(text:) needs a package with bundled tokenizer files"
            case let .missingWeight(n): return "package is missing required weight \(n)"
            case let .unsupportedShape(m): return "unsupported model shape: \(m)"
            case let .malformedSchedule(m): return "malformed loop schedule: \(m)"
            }
        }
    }

    /// Copy `bytes` from `src`@srcOff → `dst`@dstOff (e.g. seed/gather rows → the compiled trunk's
    /// hiddenA port, or its normOut → lastHiddenBuf). Inside a `batched`
    /// scope this folds into the shared command buffer (end the compute encoder, blit, resume one) so
    /// it costs no extra commit/sync; otherwise it's a standalone synchronous blit.
    func blitCopy(_ src: MTLBuffer, _ srcOff: Int, _ dst: MTLBuffer, _ dstOff: Int, _ bytes: Int) {
        if let cmd = batchCmd {
            batchEnc?.endEncoding()
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(from: src, sourceOffset: srcOff, to: dst, destinationOffset: dstOff, size: bytes)
            blit.endEncoding()
            batchEnc = cmd.makeComputeCommandEncoder()!
            return
        }
        let cmd = queue.makeCommandBuffer()!, blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: src, sourceOffset: srcOff, to: dst, destinationOffset: dstOff, size: bytes)
        blit.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    }
    /// One frame's MTP working state: the on-GPU codebook ids (read back once at the end).
    /// The MTP transformer's KV + RoPE live in the compiled trunk-mtp/ sidecar (pinned at
    /// session init), so this state is just the per-frame argmax/sampling scratch.
    struct MTPFrameState {
        let idxBuf: MTLBuffer
        let tapBufs: [MTLBuffer]?
        let teacherForced: Bool
    }

    func prepareMTPFrame(shape: TalkerShape,
                         teacherCodes: [Int]? = nil, wantTaps: Bool = false) -> MTPFrameState {
        // The 15 codebook ids live on-GPU in idxBuf for the whole frame; read back ONCE at the end.
        let idxBuf = device.makeBuffer(length: 15 * 4, options: .storageModeShared)!
        if let tc = teacherCodes {
            let p = idxBuf.contents().bindMemory(to: UInt32.self, capacity: 15)
            for i in 0..<15 { p[i] = UInt32(tc[i]) }
        } else {
            memset(idxBuf.contents(), 0, 15 * 4)
        }
        // Diagnostic logits tap (fidelity gate): persistent per-sub-pass buffers, blitted inside the
        // batch and read AFTER it commits (a host read of the transient logits would see pre-commit
        // garbage — same reason as the cb0 path).
        let tapBufs: [MTLBuffer]? = !wantTaps ? nil
            : (0..<15).map { _ in device.makeBuffer(length: shape.mtpVocab * 4, options: .storageModeShared)! }
        return MTPFrameState(idxBuf: idxBuf, tapBufs: tapBufs, teacherForced: teacherCodes != nil)
    }

    func readMTPFrame(_ state: MTPFrameState) -> [Int] {
        let idxPtr = state.idxBuf.contents().bindMemory(to: UInt32.self, capacity: 15)
        return (0..<15).map { Int(idxPtr[$0]) }
    }

    /// The compiled MTP/code_predictor trunk (B3.2d, <pkg>/trunk-mtp) + its embeddings-in /
    /// hidden-out ports — the transformer forward the MTP sub-passes run.
    struct CompiledMTPTrunk {
        let trunk: SmeltRuntime
        let hiddenA: MTLBuffer
        let normOut: MTLBuffer
    }

    /// The 15 MTP sub-passes, encode-only — must run inside an open batched scope (the
    /// scheduled loop's phase scope). The internal transformer forward runs through the
    /// compiled trunk-mtp/ sidecar (`mtpTrunk`); the seed/gather/lm_head/sampling
    /// orchestration wraps it. The MTP's fixed 16-position frame is never over the prefill
    /// cap; it just consumes lastHiddenBuf, however the talker made it.
    func encodeMTPSubPasses(_ state: MTPFrameState, talkerHiddenBuf: MTLBuffer, cb0: Int,
                            talkerCodecEmbBuf: MTLBuffer, mtpEmbBufs: [MTLBuffer],
                            proj: (w: MTLBuffer, b: MTLBuffer, dtype: WeightPackDType)?,
                            lmHeadDType: WeightPackDType, lmHeadBufs: [MTLBuffer],
                            shape: TalkerShape, mtpEmbDType: WeightPackDType,
                            mtpSampling: (uniforms: MTLBuffer, temperature: Float, topK: Int)?,
                            mtpTrunk: CompiledMTPTrunk) throws {
        let talkerDim = shape.hidden, mtpHidden = shape.mtpHidden, vocab = shape.mtpVocab
        let idxBuf = state.idxBuf
        // Seed rows 0,1 = [talker hidden, cb0 embed], assembled GPU-side by two blits (exact byte
        // copies of the resident hidden + the cb0 codec_embedding row) — no CPU round-trip / upload.
        let seedBuf = outF32(2 * talkerDim)
        blitCopy(talkerHiddenBuf, 0, seedBuf, 0, talkerDim * 4)
        blitCopy(talkerCodecEmbBuf, cb0 * talkerDim * 4, seedBuf, talkerDim * 4, talkerDim * 4)
        let seedProj = proj == nil ? seedBuf
            : try matmulDispatch(seedBuf, 2, talkerDim, proj!.w, proj!.b, mtpHidden, dtype: proj!.dtype)
        // Compiled MTP trunk: blit the 2 seed rows → hiddenA, encode the prefill into the
        // batched scope (rope/capacity pinned once at session init); the trunk's normOutBuf
        // IS the 2-row hidden. The prefill-from-pos-0 resets the trunk's KV for this frame.
        // blitCopy ends+recreates batchEnc, so fetch it AFTER the blit.
        blitCopy(seedProj, 0, mtpTrunk.hiddenA, 0, 2 * mtpHidden * 4)
        guard let enc = batchEnc else { preconditionFailure("compiled MTP trunk needs a batched scope") }
        try mtpTrunk.trunk.encodeTrunkPrefill(into: enc, seqLen: 2)
        let prefillHidden = mtpTrunk.normOut
        // gs==0 runs lm_head[0] on prefill ROW 1 (no gather/decode). Blit it (matmul binds x at offset 0).
        let row1 = outF32(mtpHidden)
        blitCopy(prefillHidden, mtpHidden * 4, row1, 0, mtpHidden * 4)
        for gs in 0..<15 {
            let lastBuf: MTLBuffer
            if gs == 0 {
                lastBuf = row1
            } else {
                // Gather mtpCodecEmbs[gs-1] row = idxBuf[gs-1] (the prior sub-pass's argmax), on-GPU.
                let newRow = try gatherRowDispatch(mtpEmbBufs[gs - 1], idxBuf, slot: gs - 1, dim: talkerDim, rows: vocab, dtype: mtpEmbDType)
                let projRow = proj == nil ? newRow
                    : try matmulDispatch(newRow, 1, talkerDim, proj!.w, proj!.b, mtpHidden, dtype: proj!.dtype)
                let pos = gs + 1
                // Compiled: blit the projected row → hiddenA, decode at this absolute pos
                // (continuing the prefill KV); the trunk's normOutBuf is the hidden lm_head
                // reads (consumed before the next sub-pass's blit ends the encoder).
                blitCopy(projRow, 0, mtpTrunk.hiddenA, 0, mtpHidden * 4)
                guard let enc = batchEnc else { preconditionFailure("compiled MTP trunk needs a batched scope") }
                try mtpTrunk.trunk.encodeTrunkDecode(into: enc, tokenId: 0, position: Int32(pos))
                lastBuf = mtpTrunk.normOut
            }
            let logits = try matmulDispatch(lastBuf, 1, mtpHidden, lmHeadBufs[gs], nil, vocab, dtype: lmHeadDType)
            if let tb = state.tapBufs { blitCopy(logits, 0, tb[gs], 0, vocab * 4) }
            if state.teacherForced {
                // idxBuf already holds the forced codes; no selection dispatch.
            } else if let s = mtpSampling {
                try sampleTopKDispatch(logits, uniforms: s.uniforms, idxBuf, slot: gs, n: vocab,
                                       temperature: s.temperature, topK: s.topK)
            } else {
                try argmaxDispatch(logits, idxBuf, slot: gs, n: vocab)
            }
        }
    }

    /// Autoregressive talker+MTP greedy generation from prefill embeds. Returns per-frame
    /// [16] codebook ids; stops on codec EOS or after maxFrames. Mirrors the P5 generate gate.
    /// On-GPU next-frame talker input: `out = talkerEmb[codes0] + ttsPad + Σ mtp_i[codes_{i+1}]`
    /// (next_frame_input_f32) — the 16-codebook gather-sum, same accumulation order as the CPU
    /// Qwen3TTSGenerator.nextFrameInput (bit-exact), so the frame's codes don't round-trip to build the
    /// next decode input. `codes` is a 16-uint buffer; `mtpTables` the 15 resident MTP codec_embedding
    /// buffers (cb1..cb15); `out` feeds the next decode directly.
    func nextFrameInputDispatch(codes: MTLBuffer, ttsPad: MTLBuffer, talkerEmb: MTLBuffer,
                                mtpTables: [MTLBuffer], out: MTLBuffer, dim: Int,
                                mtpDType: WeightPackDType = .f32, talkerRows: Int = 0, mtpRows: Int = 0) throws {
        precondition(mtpTables.count == 15, "next_frame_input needs 15 MTP tables, got \(mtpTables.count)")
        precondition(codes.length >= 16 * 4, "codes buffer \(codes.length)B < 16 uint")
        precondition(out.length >= dim * 4 && ttsPad.length >= dim * 4, "out/ttsPad < \(dim*4)B")
        // bf16 MTP tables (BF16-source codec embeddings) widen in-kernel → bit-identical to f32. The
        // talkerEmb (cb0 table) stays f32 in both variants (it's also blitted raw for the MTP seed).
        precondition(mtpDType == .f32 || mtpDType == .bf16, "next_frame_input supports f32/bf16 MTP tables, not \(mtpDType)")
        // The codes index rows of each table on the GPU; trap a too-small table (dtype/shape regression)
        // on the host rather than a silent GPU OOB read (parity with gatherRowDispatch).
        let mtpElem = mtpDType == .f32 ? 4 : 2
        precondition(talkerEmb.length >= talkerRows * dim * 4, "talkerEmb \(talkerEmb.length)B < \(talkerRows*dim*4)B")
        for (i, t) in mtpTables.enumerated() {
            precondition(t.length >= mtpRows * dim * mtpElem, "MTP table \(i) \(t.length)B < \(mtpRows*dim*mtpElem)B")
        }
        let pipe = try pso(mtpDType == .bf16 ? "next_frame_input_bf16w_f32" : "next_frame_input_f32")
        encode(pipe, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: min(dim, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(codes, offset: 0, index: 0); enc.setBuffer(ttsPad, offset: 0, index: 1)
            enc.setBuffer(talkerEmb, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
            for i in 0..<15 { enc.setBuffer(mtpTables[i], offset: 0, index: 4 + i) }
            var d = UInt32(dim); enc.setBytes(&d, length: 4, index: 19)
        }
    }

    /// On-GPU codebook-0 selection (cb0_argmax_f32 = applyCb0Processors + argmax) → `out[0]` uint, so
    /// the 3072 codecHead logits + CPU argmax never round-trip; only cb0 (for the EOS check) comes back.
    /// `history` holds the prior cb0s (length `historyLen`) for the repetition penalty. Defaults match
    /// Qwen3TTSGenerator.applyCb0Processors so the selection is bit-exact.
    func cb0ArgmaxDispatch(logits: MTLBuffer, history: MTLBuffer, historyLen: Int, frame: Int, n: Int,
                           out: MTLBuffer, eos: Int,
                           repetitionPenalty: Float = Qwen3TTSGenerator.Cb0Config.repetitionPenalty,
                           suppressFrom: Int = Qwen3TTSGenerator.Cb0Config.suppressFrom,
                           minNewTokens: Int = Qwen3TTSGenerator.Cb0Config.minNewTokens) throws {
        precondition(logits.length >= n * 4, "cb0 logits \(logits.length)B < \(n*4)B")
        precondition(history.length >= historyLen * 4, "cb0 history < \(historyLen*4)B")
        precondition(out.length >= 4, "cb0 out < 4B")
        let pipe = try pso("cb0_argmax_f32")
        precondition(pipe.threadExecutionWidth == 32, "cb0_argmax assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        encode(pipe, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(history, offset: 0, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            var nU = UInt32(n), hl = UInt32(historyLen), fr = UInt32(frame), pen = repetitionPenalty
            var sf = UInt32(suppressFrom), e = UInt32(eos), mnt = UInt32(minNewTokens)
            enc.setBytes(&nU, length: 4, index: 3); enc.setBytes(&hl, length: 4, index: 4)
            enc.setBytes(&fr, length: 4, index: 5); enc.setBytes(&pen, length: 4, index: 6)
            enc.setBytes(&sf, length: 4, index: 7); enc.setBytes(&e, length: 4, index: 8)
            enc.setBytes(&mnt, length: 4, index: 9)
        }
    }

    /// On-GPU codebook-0 SAMPLED selection (cb0_sample_topk_f32 = applyCb0Processors + temperature/top-k
    /// sampling) → `out[0]` uint — the sampled twin of cb0ArgmaxDispatch. `uniforms[0]` is the host
    /// `Qwen3TTSSampler.uniform(seed, frame, 0)` draw for this frame's cb0; the processors are recomputed
    /// in-kernel from `history`/`frame`, so the 3072 logits + host sampleTopK never round-trip. Defaults
    /// match Qwen3TTSGenerator.applyCb0Processors (bit-exact processors; distribution-equivalent CDF).
    func cb0SampleTopKDispatch(logits: MTLBuffer, history: MTLBuffer, historyLen: Int, frame: Int, n: Int,
                               uniforms: MTLBuffer, out: MTLBuffer, eos: Int, temperature: Float, topK: Int,
                               repetitionPenalty: Float = Qwen3TTSGenerator.Cb0Config.repetitionPenalty,
                               suppressFrom: Int = Qwen3TTSGenerator.Cb0Config.suppressFrom,
                               minNewTokens: Int = Qwen3TTSGenerator.Cb0Config.minNewTokens) throws {
        precondition(logits.length >= n * 4, "cb0 sample logits \(logits.length)B < \(n*4)B")
        precondition(history.length >= historyLen * 4, "cb0 sample history < \(historyLen*4)B")
        precondition(uniforms.length >= 4 && out.length >= 4, "cb0 sample uniforms/out < 4B")
        precondition(temperature > 0, "cb0SampleTopK needs temperature > 0; greedy is cb0ArgmaxDispatch")
        let pipe = try pso("cb0_sample_topk_f32")
        precondition(pipe.threadExecutionWidth == 32, "cb0_sample_topk assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        encode(pipe, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(history, offset: 0, index: 1)
            enc.setBuffer(uniforms, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
            var nU = UInt32(n), hl = UInt32(historyLen), fr = UInt32(frame), pen = repetitionPenalty
            var sf = UInt32(suppressFrom), e = UInt32(eos), mnt = UInt32(minNewTokens)
            var temp = temperature, kk = UInt32(max(topK, 1))
            enc.setBytes(&nU, length: 4, index: 4); enc.setBytes(&hl, length: 4, index: 5)
            enc.setBytes(&fr, length: 4, index: 6); enc.setBytes(&pen, length: 4, index: 7)
            enc.setBytes(&sf, length: 4, index: 8); enc.setBytes(&e, length: 4, index: 9)
            enc.setBytes(&mnt, length: 4, index: 10); enc.setBytes(&temp, length: 4, index: 11)
            enc.setBytes(&kk, length: 4, index: 12)
        }
    }

    /// Model dims derived from manifest tensor shapes + the bundled config.json
    /// (docs/qwen3-tts-variants-plan.md U2). Rules: headDim comes from q_norm length (never
    /// proj-shape ratios — attn width ≠ hidden on the 0.6B), textEmbedDim is kept separate from
    /// hidden (they coincide only on the 1.7B), the MTP codec_embedding/table width follows the
    /// TALKER hidden, and small_to_mtp_projection is optional (absent on the 0.6B where
    /// hidden == mtpHidden — asserted, not silently skipped).
    struct TalkerShape {
        let textEmbedDim, projInter, hidden, heads, kvHeads, headDim, inter, vocab, layers: Int
        let mtpHidden, mtpHeads, mtpKvHeads, mtpHeadDim, mtpInter, mtpVocab, mtpLayers: Int
        let hasSmallToMtp: Bool
        let ropeTheta: Float
    }

    func talkerShape() throws -> TalkerShape {
        if let cached = talkerShapeCache { return cached }
        let computed = try computeTalkerShape()
        talkerShapeCache = computed
        return computed
    }

    private func computeTalkerShape() throws -> TalkerShape {
        func shape(_ name: String) throws -> [Int] {
            guard let s = weightShape(name) else { throw GenerateError.missingWeight(name) }
            return s
        }
        func layerCount(_ prefix: String) -> Int {
            var n = 0
            while weightShape("\(prefix)layers.\(n).input_layernorm.weight") != nil { n += 1 }
            return n
        }
        let headDim = try shape("talker.model.layers.0.self_attn.q_norm.weight")[0]
        let heads = try shape("talker.model.layers.0.self_attn.q_proj.weight")[0] / headDim
        let kvHeads = try shape("talker.model.layers.0.self_attn.k_proj.weight")[0] / headDim
        let hidden = try shape("talker.text_projection.linear_fc2.weight")[0]
        guard try shape("talker.model.layers.0.self_attn.q_proj.weight")[1] == hidden else {
            throw GenerateError.unsupportedShape("q_proj cols != text_projection fc2 rows")
        }
        let mtpHeadDim = try shape("talker.code_predictor.model.layers.0.self_attn.q_norm.weight")[0]
        let hasSmallToMtp = weightShape("talker.code_predictor.small_to_mtp_projection.weight") != nil
        let mtpHidden = try shape("talker.code_predictor.model.layers.0.self_attn.q_proj.weight")[1]
        if !hasSmallToMtp, mtpHidden != hidden {
            throw GenerateError.unsupportedShape(
                "small_to_mtp_projection absent but talker hidden \(hidden) != MTP hidden \(mtpHidden)")
        }
        // The MTP codec_embedding tables embed at TALKER width (their rows feed the next-frame
        // gather-sum and the small_to_mtp seed), not at MTP width.
        guard try shape("talker.code_predictor.model.codec_embedding.0.weight")[1] == hidden else {
            throw GenerateError.unsupportedShape("MTP codec_embedding width != talker hidden")
        }
        // rope is a CONFIG value, not a tensor shape: read the bundled config.json and refuse
        // unsupported rope modes loudly rather than defaulting.
        let cfgData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/config.json"))
        guard let root = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
              let talkerCfg = root["talker_config"] as? [String: Any],
              let theta = (talkerCfg["rope_theta"] as? NSNumber)?.floatValue else {
            throw GenerateError.unsupportedShape("config.json missing talker_config.rope_theta")
        }
        // Qwen3-TTS checkpoints ship a rope_scaling block with type "default" (an inert
        // mrope_section annotation — the talker path is plain 1D RoPE, proven by the shipped
        // parity gates). Accept exactly that; any ACTIVE scaling type is unsupported.
        if let scaling = talkerCfg["rope_scaling"] as? [String: Any] {
            let t = (scaling["rope_type"] as? String) ?? (scaling["type"] as? String) ?? "default"
            guard t == "default" else {
                throw GenerateError.unsupportedShape("rope_scaling type '\(t)'; only plain/default RoPE is supported")
            }
        } else if let scaling = talkerCfg["rope_scaling"], !(scaling is NSNull) {
            throw GenerateError.unsupportedShape("malformed rope_scaling: \(scaling)")
        }
        if let cp = talkerCfg["code_predictor_config"] as? [String: Any],
           let cpTheta = (cp["rope_theta"] as? NSNumber)?.floatValue, cpTheta != theta {
            throw GenerateError.unsupportedShape("MTP rope_theta \(cpTheta) != talker \(theta)")
        }
        return TalkerShape(
            textEmbedDim: try shape("talker.model.text_embedding.weight")[1],
            projInter: try shape("talker.text_projection.linear_fc1.weight")[0],
            hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim,
            inter: try shape("talker.model.layers.0.mlp.gate_proj.weight")[0],
            vocab: try shape("talker.codec_head.weight")[0],
            layers: layerCount("talker.model."),
            mtpHidden: mtpHidden,
            mtpHeads: try shape("talker.code_predictor.model.layers.0.self_attn.q_proj.weight")[0] / mtpHeadDim,
            mtpKvHeads: try shape("talker.code_predictor.model.layers.0.self_attn.k_proj.weight")[0] / mtpHeadDim,
            mtpHeadDim: mtpHeadDim,
            mtpInter: try shape("talker.code_predictor.model.layers.0.mlp.gate_proj.weight")[0],
            mtpVocab: try shape("talker.code_predictor.lm_head.0.weight")[0],
            mtpLayers: layerCount("talker.code_predictor.model."),
            hasSmallToMtp: hasSmallToMtp,
            ropeTheta: theta)
    }

    /// The session's prompt-embeds buffer: the GPU front-end output (`prebuilt`) when present, else the
    /// host-uploaded `inputsEmbeds` floats — which must be EXACTLY seqLen×hidden (the public array
    /// contract; a prebuilt buffer is bounds-checked in the session instead).
    private func resolveInputsBuf(_ inputsEmbeds: [Float], seqLen: Int, prebuilt: MTLBuffer?) throws -> MTLBuffer {
        if let prebuilt = prebuilt { return prebuilt }
        let hidden = try talkerShape().hidden
        precondition(inputsEmbeds.count == seqLen * hidden,
                     "inputsEmbeds \(inputsEmbeds.count) != seqLen \(seqLen) × hidden \(hidden)")
        return bufF32(inputsEmbeds)
    }

    /// `mtpTeacherCodes` (gate-only, pair with maxFrames=1 + mtpLogitsTap): force frame-0's 15
    /// MTP sub-codes to the reference's so every sub-pass's logits are reference-comparable.
    /// `prebuiltInputsBuf` (the GPU front-end output, generate(text:)) overrides the host-uploaded
    /// `inputsEmbeds` floats as the prompt source; when nil, the floats are uploaded as before (the
    /// direct-embed callers + bit-exact gates).
    public func generateCodes(inputsEmbeds: [Float], seqLen: Int, ttsPadId: Int,
                              maxFrames: Int, sampling: Qwen3TTSSampler.Params? = nil,
                              cb0LogitsTap: ((Int, [Float]) -> Void)? = nil,
                              mtpLogitsTap: (([[Float]]) -> Void)? = nil,
                              talkerHiddenTap: (([Float]) -> Void)? = nil,
                              onFrame: ((Int, [Int]) throws -> Bool)? = nil,
                              mtpTeacherCodes: [Int]? = nil,
                              prebuiltInputsBuf: MTLBuffer? = nil) throws -> [[Int]] {
        SmeltDecodeProfile.setStage("setup")
        // All per-call state (resident weights, persistent buffers, KV cache, rope) lives in the
        // session — the SAME encode/host halves the scheduled blocks compose, so this monolithic loop
        // (the offline path + the tap/teacher gates) cannot drift from the orchestrated one.
        let s = try TalkerSession(
            gpu: self, inputsBuf: try resolveInputsBuf(inputsEmbeds, seqLen: seqLen, prebuilt: prebuiltInputsBuf),
            seqLen: seqLen, ttsPadId: ttsPadId, maxFrames: maxFrames, sampling: sampling)
        SmeltDecodeProfile.setStage("talker")
        // cb0 for frame F is folded into whatever scope just wrote lastHiddenBuf (prefill for frame 0,
        // the previous frame's decode scope otherwise) — gap-free EOS gating, no extra submission.
        try profTimeThrows("prefillWall") { try batched {
            try s.encodePrefill()
            try s.encodeCb0(frame: 0, wantLogits: cb0LogitsTap != nil)
        } }
        if let tap = cb0LogitsTap { tap(0, readF32(s.cb0LogitsBuf, s.shape.vocab)) }   // post-commit read
        var gen: [[Int]] = []
        for frame in 0..<maxFrames {
            let cb0 = s.takeCb0()
            if cb0 == s.eos { break }
            s.acceptCb0(cb0)
            // Phase 4 U2 calibration: lastHiddenBuf now holds this frame's talker hidden (the
            // input the live MTP consumes) — tapped post-scope, paired by order with `gen`.
            if let tap = talkerHiddenTap { tap(readF32(s.lastHiddenBuf, s.shape.hidden)) }
            SmeltDecodeProfile.setStage("mtp")
            let state = profTime("mtpSetup") {
                s.prepareMTPFrame(teacherCodes: frame == 0 ? mtpTeacherCodes : nil,   // frame-0 only
                                  wantTaps: frame == 0 && mtpLogitsTap != nil)
            }
            try profTimeThrows("frameWall") { try batched { try s.encodeMTP(state, frame: frame) } }
            if let tap = mtpLogitsTap, let tb = state.tapBufs {
                tap(tb.map { readF32($0, s.shape.mtpVocab) })   // post-commit read
            }
            let codes16 = [cb0] + s.readMTPCodes(state)
            gen.append(codes16)
            s.lastCodes = codes16
            // Streaming tap: the frame's codes are final here. A `false` return cancels the
            // remaining generation (the caller has everything it asked for, e.g. barge-in).
            if let onFrame, try !onFrame(gen.count - 1, codes16) { break }
            guard frame < maxFrames - 1 else { break }
            SmeltDecodeProfile.setStage("talker")
            try profTimeThrows("decodeWall") { try batched {
                try s.encodeFeedbackAndDecode()
                try s.encodeCb0(frame: frame + 1, wantLogits: cb0LogitsTap != nil)
            } }
            if let tap = cb0LogitsTap { tap(frame + 1, readF32(s.cb0LogitsBuf, s.shape.vocab)) }   // post-commit read
            s.advancePosition()
        }
        return gen
    }

    /// Reconstruct the talker DECODE inputs for a generation's `codes` (Phase 4 U2 calibration):
    /// for each frame F in 0..<N-1, the next-frame gather-sum `nextFrameInput(codes[F])` the real
    /// decode fed at position seqLen+F to produce frame F+1's hidden. Returns N-1 rows of [hidden]
    /// floats — BIT-EXACT to the live decode input (the same next_frame_input_f32 kernel) — so the
    /// concatenated [prompt embeds ++ these] prefill exposes every projection to the real
    /// autoregressive distribution (teacher-forced), captured through the compiled trunk.
    /// Pin a capture trunk's RoPE (the Float-math source the compiled trunk matches) + grow its
    /// capacity for a prefill of `seqLen` (Phase 4 U2 calibration). `mtp` selects the MTP head-dim/
    /// hidden. Returns the trunk's hidden dim (for the caller's source-buffer sizing). Mirrors the
    /// session's makeCompiledTrunk setup so the captured activations match the live generation.
    public func prepareTrunkForCapture(_ trunk: SmeltRuntime, seqLen: Int, mtp: Bool) throws -> Int {
        let shape = try talkerShape()
        let hd = mtp ? shape.mtpHeadDim : shape.headDim
        let (cos, sin) = ropeTables(frames: seqLen, headDim: hd, theta: shape.ropeTheta)
        try trunk.ensurePrefillCapacity(seqLen: seqLen)   // clamps the batch slab; grows context to seqLen
        try trunk.ensureContextCapacity(seqLen)
        func slot(_ n: String) -> Int? { trunk.manifest.buffers.slots.first { $0.name == n }?.index }
        guard let cosI = slot("ropeCos"), let sinI = slot("ropeSin") else {
            throw GenerateError.unsupportedShape("capture trunk missing ropeCos/ropeSin slots")
        }
        trunk.writeSlot(cosI, f32: cos)
        trunk.writeSlot(sinI, f32: sin)
        return mtp ? shape.mtpHidden : shape.hidden
    }

    public func decodeFrameInputs(codes: [[Int]], ttsPadId: Int) throws -> [[Float]] {
        let shape = try talkerShape()
        let hidden = shape.hidden
        guard codes.count >= 2 else { return [] }
        let talkerEmb = weight("talker.model.codec_embedding.weight")!
        let mtpTables = (0..<15).map { weight("talker.code_predictor.model.codec_embedding.\($0).weight")! }
        let mtpDType = weightDType("talker.code_predictor.model.codec_embedding.0.weight")
        let talkerRows = weightShape("talker.model.codec_embedding.weight")?[0] ?? 0
        let mtpRows = weightShape("talker.code_predictor.model.codec_embedding.0.weight")?[0] ?? 0
        let ttsPad = bufF32(ttsPadEmbed(ttsPadId, shape: shape))
        let codesBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: hidden * 4, options: .storageModeShared)!
        var inputs: [[Float]] = []
        inputs.reserveCapacity(codes.count - 1)
        for f in 0..<(codes.count - 1) {
            precondition(codes[f].count == 16, "frame \(f) needs 16 codes, got \(codes[f].count)")
            let cptr = codesBuf.contents().bindMemory(to: UInt32.self, capacity: 16)
            for i in 0..<16 { cptr[i] = UInt32(codes[f][i]) }
            try batched {
                try nextFrameInputDispatch(codes: codesBuf, ttsPad: ttsPad, talkerEmb: talkerEmb,
                                           mtpTables: mtpTables, out: outBuf, dim: hidden,
                                           mtpDType: mtpDType, talkerRows: talkerRows, mtpRows: mtpRows)
            }
            inputs.append(readF32(outBuf, hidden))
        }
        return inputs
    }

    /// Assemble each frame's 16-row teacher-forced MTP seed (Phase 4 U2 calibration), mtpHidden-dim:
    /// row0 = the frame's talker hidden, row1 = talkerCodecEmb[cb0], rows r=2..15 =
    /// mtpCodecEmb[r-2][cb_{r-1}] (so codes cb0..cb14; cb15 is predicted, never fed) — then EACH row
    /// small_to_mtp-projected (identity on the 0.6B where talkerHidden==mtpHidden). This is exactly the
    /// seed encodeMTPSubPasses builds, so a 16-row prefill of the MTP trunk over it exposes the same
    /// per-projection inputs as the live prefill(2)+decode(14) (the MTP attention is causal, KV per-frame).
    public func mtpCalibrationSeeds(codes: [[Int]], talkerHiddens: [[Float]]) throws -> [[Float]] {
        let shape = try talkerShape()
        precondition(codes.count == talkerHiddens.count, "codes/hiddens count mismatch")
        let talkerDim = shape.hidden, mtpHidden = shape.mtpHidden
        let proj: (w: MTLBuffer, b: MTLBuffer, dtype: WeightPackDType)?
        if let w = weight("talker.code_predictor.small_to_mtp_projection.weight") {
            proj = (w, weight("talker.code_predictor.small_to_mtp_projection.bias")!,
                    weightDType("talker.code_predictor.small_to_mtp_projection.weight"))
        } else {
            precondition(talkerDim == mtpHidden, "no small_to_mtp but talkerDim != mtpHidden")
            proj = nil
        }
        var seeds: [[Float]] = []
        seeds.reserveCapacity(codes.count)
        for (f, hidden) in talkerHiddens.enumerated() {
            let c = codes[f]
            precondition(c.count == 16 && hidden.count == talkerDim,
                         "frame \(f): codes \(c.count)/16, hidden \(hidden.count)/\(talkerDim)")
            var rows = [Float](repeating: 0, count: 16 * talkerDim)
            rows.replaceSubrange(0..<talkerDim, with: hidden)
            rows.replaceSubrange(talkerDim..<(2 * talkerDim),
                with: weightRow("talker.model.codec_embedding.weight", c[0], talkerDim))
            for r in 2..<16 {
                rows.replaceSubrange((r * talkerDim)..<((r + 1) * talkerDim),
                    with: weightRow("talker.code_predictor.model.codec_embedding.\(r - 2).weight",
                                    c[r - 1], talkerDim))
            }
            if let proj {
                var out: MTLBuffer!
                try batched {
                    out = try matmulDispatch(bufF32(rows), 16, talkerDim, proj.w, proj.b, mtpHidden,
                                             dtype: proj.dtype)
                }
                seeds.append(readF32(out, 16 * mtpHidden))
            } else {
                seeds.append(rows)
            }
        }
        return seeds
    }

    /// The package's configured sampling params at `seed`, or nil (greedy) when the package isn't a
    /// sampling model. The seed source is the only thing that differs between .packageDefault (random)
    /// and .sampleSeeded (fixed), so both route through here.
    private func packageSampling(seed: UInt64) -> Qwen3TTSSampler.Params? {
        guard let d = manifest.decode, d.doSample else { return nil }
        return Qwen3TTSSampler.Params(
            seed: seed,
            talkerTemperature: d.temperature, talkerTopK: d.topK,
            subtalkerTemperature: d.subtalkerTemperature, subtalkerTopK: d.subtalkerTopK)
    }

    /// Resolve a DecodeMode to the low-level `sampling: Params?` generateCodes consumes. nil ⇒ greedy.
    func resolveDecode(_ mode: Qwen3TTSSampler.DecodeMode) -> Qwen3TTSSampler.Params? {
        switch mode {
        case .greedy: return nil
        case .sample(let p): return p
        case .packageDefault: return packageSampling(seed: .random(in: .min ... .max))
        case .sampleSeeded(let seed): return packageSampling(seed: seed)
        }
    }

    /// The text_projection fc weights as [Float], memoized on first use (pure package weights;
    /// shared by the front-end's per-token projection and generateCodes' tts_pad row).
    func textProjWeights() -> (fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float]) {
        if let cached = textProjWeightsCache { return cached }
        let w = (fc1W: f32("talker.text_projection.linear_fc1.weight"),
                 fc1B: f32("talker.text_projection.linear_fc1.bias"),
                 fc2W: f32("talker.text_projection.linear_fc2.weight"),
                 fc2B: f32("talker.text_projection.linear_fc2.bias"))
        textProjWeightsCache = w
        return w
    }

    /// The projected tts_pad row (text_embedding[id] → text_projection), memoized by token id —
    /// generateCodes needs it every call for the on-GPU next-frame-input gather-sum.
    func ttsPadEmbed(_ ttsPadId: Int, shape: TalkerShape) -> [Float] {
        if let cached = ttsPadEmbedCache[ttsPadId] { return cached }
        let w = textProjWeights()
        let row = Qwen3TTSTalker.textProjection(
            weightRow("talker.model.text_embedding.weight", ttsPadId, shape.textEmbedDim), rows: 1,
            fc1W: w.fc1W, fc1B: w.fc1B, fc2W: w.fc2W, fc2B: w.fc2B,
            inDim: shape.textEmbedDim, interDim: shape.projInter, outDim: shape.hidden)
        ttsPadEmbedCache[ttsPadId] = row
        return row
    }

    /// Run the front-end (tokenize + text/codec row-gather + text_projection) to the talker's
    /// `inputsEmbeds` — the input `generate(inputsEmbeds:…)`/`generateCodes` consume. Exposed so a
    /// caller can derive embeds ONCE from a fixed package and feed the identical array to several
    /// packages (e.g. GPTQ calibration, and the bf16-vs-quant fidelity gate — a u4 package would
    /// dequantize its own text_embedding, giving different embeds).
    public func inputsEmbeds(text: String, instruct: String?, language: String,
                             speaker: String? = nil)
        throws -> (embeds: [Float], seqLen: Int, ttsPadId: Int) {
        guard let tokenizer = tokenizer, let config = frontEndConfig else {
            throw GenerateError.noTokenizer
        }
        let shape = try talkerShape()
        let proj = textProjWeights()
        let (embeds, frames) = try Qwen3TTSFrontEnd.textToInputsEmbeds(
            text: text, instruct: instruct, language: language, speaker: speaker,
            tokenizer: tokenizer, config: config,
            textRow: { weightRow("talker.model.text_embedding.weight", $0, shape.textEmbedDim) },
            codecRow: { weightRow("talker.model.codec_embedding.weight", $0, shape.hidden) },
            fc1W: proj.fc1W, fc1B: proj.fc1B, fc2W: proj.fc2W, fc2B: proj.fc2B,
            dim: shape.textEmbedDim, projInter: shape.projInter, hidden: shape.hidden)
        return (embeds, frames, config.ttsPad)
    }

    /// On-GPU TTS front-end (1a-i): assemble the dual-track prefill inputs-embeds [seqLen, hidden] —
    /// the port the compiled trunk consumes as `hiddenA` — on the GPU instead of via the CPU/BLAS
    /// `voiceDesignPrefill`. The row layout + the per-row text/codec gather stay host (cheap index
    /// logic + ~seqLen mapped-buffer reads, u4 dequant included via weightRow); the heavy numerics
    /// move to the GPU: text_projection = fc2(silu(fc1)) over the gathered rows (the gemm_tn biased
    /// matmul + a silu dispatch), then the dual-track merge out = projected + codec
    /// (scale_residual_tc add-only). Row-aligned (one gathered text row per output row, specials
    /// projected per-use), so the projection comes out in output order and the merge is one dispatch.
    /// Tolerance-equivalent to voiceDesignPrefill (GPU matmul K-reduction order != cblas_sgemm), not
    /// bit-exact. Runs in the caller's batched scope if one is open, else commits per dispatch.
    public func frontEndPrefillHiddenA(text: String, instruct: String?, language: String,
                                       speaker: String? = nil)
        throws -> (buf: MTLBuffer, seqLen: Int, ttsPadId: Int) {
        guard let tokenizer = tokenizer, let config = frontEndConfig else {
            throw GenerateError.noTokenizer
        }
        let shape = try talkerShape()
        let (instructIds, inputIds) = Qwen3TTSFrontEnd.wrap(text: text, instruct: instruct, tokenizer: tokenizer)
        let ids = try config.ids(language: language, speaker: speaker)
        let rows = Qwen3TTSTalkerPrefill.layout(instructIds: instructIds, inputIds: inputIds, ids: ids)
        let buf: MTLBuffer
        if compiledFrontEndSupported {
            // Compiled front-end (1a-ii): the Lego record table, byte-identical to the host-issued path.
            buf = try compiledFrontEndHiddenA(rows: rows, shape: shape)
        } else if weightDType("talker.model.text_embedding.weight") == .bf16 {
            // A bf16 build IS the compiled-front-end class — the graph stamps tts-frontend .compiled. If
            // its pipelines are absent the package is corrupt; fail CLOSED rather than silently run the
            // host front-end while the manifest claims compiled (codex adversarial review).
            throw GenerateError.unsupportedShape(
                "bf16 package declares a compiled front-end but is missing its front-end pipelines (corrupt package)")
        } else {
            // f32/f16/u4: the honest .native host path (the graph keeps tts-frontend .native).
            buf = try frontEndPrefillHiddenA(rows: rows, shape: shape)
        }
        return (buf, rows.count, config.ttsPad)
    }

    /// Layout-driven core of the GPU front-end (above): host-gather → GPU project → GPU merge into a
    /// fresh [rows, hidden] buffer. Split out so a caller that already has the wrapped layout (the
    /// session prefill) reuses it without re-tokenizing.
    func frontEndPrefillHiddenA(rows: [Qwen3TTSTalkerPrefill.PrefillRow], shape: TalkerShape) throws -> MTLBuffer {
        let T = rows.count, dim = shape.textEmbedDim, hidden = shape.hidden, inter = shape.projInter
        precondition(T > 0, "empty prefill layout")
        // Host gather: one text_embedding row per output row + the codec addend (zero for the
        // projection-only instruct/role rows). weightRow handles f32/bf16/u4 (u4 dequantized).
        var textRows = [Float](repeating: 0, count: T * dim)
        var codecRows = [Float](repeating: 0, count: T * hidden)
        for (r, row) in rows.enumerated() {
            textRows.replaceSubrange((r * dim)..<((r + 1) * dim),
                                     with: weightRow("talker.model.text_embedding.weight", row.textId, dim))
            if let cid = row.codecId {
                codecRows.replaceSubrange((r * hidden)..<((r + 1) * hidden),
                                          with: weightRow("talker.model.codec_embedding.weight", cid, hidden))
            }
        }
        let w = textProjWeights()
        // GPU: text_projection fc2(silu(fc1)) → projected [T, hidden], then the dual-track merge
        // out = projected + codec via scale_residual_tc add-only (scaled: false ⇒ the scale buffer is
        // unread; codecBuf is passed in its slot only to satisfy the non-optional parameter).
        let h1 = try matmulDispatch(bufF32(textRows), T, dim, w.fc1W, w.fc1B, inter)
        let act = try siluDispatch(h1, T * inter)
        let projected = try matmulDispatch(act, T, inter, w.fc2W, w.fc2B, hidden)
        let codecBuf = bufF32(codecRows)
        return try scaleResidualTCDispatch(codecBuf, projected, codecBuf, hidden, T, scaled: false)
    }

    /// Whether the COMPILED front-end (1a-ii record table) runs for this package. bf16 ONLY: the compiled
    /// front-end ships with the compiled-talker (bf16) class, matching the trunk sidecar, so the block
    /// graph's `tts-frontend` → .compiled flip lives only in the bf16 graph (qwen3TTSCompiledTalker). The
    /// emitter handles f32 text_embedding too, but a static graph can't honestly mark the f32/u4-shared
    /// qwen3TTS graph compiled-for-f32-only — so the runtime gate aligns with the graph flip (f32/f16/u4
    /// keep the host-issued weightRow path).
    var compiledFrontEndSupported: Bool {
        weightDType("talker.model.text_embedding.weight") == .bf16
            // A stale/mutated bf16 package may predate the gather_rows pipelines — fall back to the
            // host-issued path instead of failing at the pso lookup (codex 1a-ii review).
            && Qwen3TTSFrontEndEmitter.pipelineNames.allSatisfy { pipeline($0) != nil }
    }

    /// Compiled front-end (1a-ii): the dual-track prefill assembled by the Qwen3TTSFrontEndEmitter record
    /// table run through the generic SmeltCodecRecordRunner — the Lego form of frontEndPrefillHiddenA
    /// (host-issued dispatches), BYTE-IDENTICAL to it (same kernels/weights/grids). Host-validates +
    /// writes the layout's text/codec ids into fresh Int32 buffers, binds the resident package weights +
    /// `hiddenA` + fresh scratch, and encodes into the caller's `enc` (the prefill batched scope).
    func encodeCompiledFrontEnd(rows: [Qwen3TTSTalkerPrefill.PrefillRow], shape: TalkerShape,
                                hiddenA: MTLBuffer, into enc: MTLComputeCommandEncoder) throws {
        precondition(compiledFrontEndSupported, "compiled front-end needs bf16/f32 text_embedding (u4 → host path)")
        let seqLen = rows.count
        let plan = Qwen3TTSFrontEndEmitter.plan(
            seqLen: seqLen, textEmbedDim: shape.textEmbedDim, projInter: shape.projInter,
            hidden: shape.hidden, textEmbedBF16: weightDType("talker.model.text_embedding.weight") == .bf16)
        let pipelines = try plan.pipelineNames.map { try pso($0) }

        // Host-write + VALIDATE the ids (a baked gather can't precheck): textId in [0, textRows);
        // codecId in [0, codecRows) or -1 (the projection-only zero-row sentinel).
        let textRows = weightShape("talker.model.text_embedding.weight")?[0] ?? 0
        let codecRows = weightShape("talker.model.codec_embedding.weight")?[0] ?? 0
        let textIdsBuf = device.makeBuffer(length: seqLen * 4, options: .storageModeShared)!
        let codecIdsBuf = device.makeBuffer(length: seqLen * 4, options: .storageModeShared)!
        let tp = textIdsBuf.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let cp = codecIdsBuf.contents().bindMemory(to: Int32.self, capacity: seqLen)
        for (i, row) in rows.enumerated() {
            precondition(row.textId >= 0 && row.textId < textRows, "front-end textId \(row.textId) out of [0,\(textRows))")
            tp[i] = Int32(row.textId)
            if let cid = row.codecId {
                precondition(cid >= 0 && cid < codecRows, "front-end codecId \(cid) out of [0,\(codecRows))")
                cp[i] = Int32(cid)
            } else {
                cp[i] = -1
            }
        }
        let buffers: [MTLBuffer] = plan.slots.map { desc in
            switch desc {
            case .textEmbed: return wbuf("talker.model.text_embedding.weight")
            case .codecEmbed: return wbuf("talker.model.codec_embedding.weight")
            case .fc1W: return wbuf("talker.text_projection.linear_fc1.weight")
            case .fc1B: return wbuf("talker.text_projection.linear_fc1.bias")
            case .fc2W: return wbuf("talker.text_projection.linear_fc2.weight")
            case .fc2B: return wbuf("talker.text_projection.linear_fc2.bias")
            case .textIds: return textIdsBuf
            case .codecIds: return codecIdsBuf
            case .hiddenA: return hiddenA
            case .scratch(let count): return outF32(count, zero: false)
            }
        }
        SmeltCodecRecordRunner.encode(plan.records, pipelines: pipelines, buffers: buffers, into: enc)
    }

    /// Standalone compiled front-end (gate oracle): run the record table in its own command buffer →
    /// a fresh [seqLen, hidden] buffer. Byte-identical to frontEndPrefillHiddenA(rows:shape:).
    func compiledFrontEndHiddenA(rows: [Qwen3TTSTalkerPrefill.PrefillRow], shape: TalkerShape) throws -> MTLBuffer {
        let hiddenA = outF32(rows.count * shape.hidden, zero: false)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        try encodeCompiledFrontEnd(rows: rows, shape: shape, hiddenA: hiddenA, into: enc)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return hiddenA
    }

    /// Self-contained text→24 kHz: front-end (`inputsEmbeds`) then the proven embeds→waveform pipeline.
    /// `instruct` designs the voice (VoiceDesign) or styles a named `speaker` (CustomVoice 1.7B);
    /// nil = no instruct rows. `speaker` is a named voice from the package's spk_id table.
    public func generate(text: String, instruct: String?, language: String,
                         speaker: String? = nil,
                         maxFrames: Int, decode: Qwen3TTSSampler.DecodeMode = .packageDefault) throws -> [Float] {
        let wallStart = SmeltDecodeProfile.enabled ? CFAbsoluteTimeGetCurrent() : 0
        if SmeltDecodeProfile.enabled { SmeltDecodeProfile.reset() }
        SmeltDecodeProfile.setStage("front")
        // GPU front-end: assemble the prompt embeds (hiddenA) on the GPU instead of the CPU/BLAS
        // voiceDesignPrefill + float upload, then feed the same downstream pipeline.
        let fe = try frontEndPrefillHiddenA(text: text, instruct: instruct, language: language, speaker: speaker)
        let wav = try generate(inputsEmbeds: [], seqLen: fe.seqLen, ttsPadId: fe.ttsPadId,
                               maxFrames: maxFrames, decode: decode, prebuiltInputsBuf: fe.buf)
        if SmeltDecodeProfile.enabled {
            let wall = CFAbsoluteTimeGetCurrent() - wallStart
            let audio = Double(wav.count) / 24000
            let report = SmeltDecodeProfile.report(totalWallS: wall)
                + String(format: "\n  wall/audio = %.1f× realtime (audio %.2f s)\n", wall / audio, audio)
            FileHandle.standardError.write(Data(report.utf8))
        }
        return wav
    }

    /// Full packaged pipeline: prefill embeds → talker+MTP codes → codec → 24 kHz waveform.
    /// `prebuiltInputsBuf` (the GPU front-end output) overrides the `inputsEmbeds` floats when set.
    public func generate(inputsEmbeds: [Float], seqLen: Int, ttsPadId: Int,
                         maxFrames: Int, decode: Qwen3TTSSampler.DecodeMode = .packageDefault,
                         prebuiltInputsBuf: MTLBuffer? = nil) throws -> [Float] {
        let prof = SmeltDecodeProfile.enabled
        let t0 = prof ? CFAbsoluteTimeGetCurrent() : 0
        let gen = try generateCodes(inputsEmbeds: inputsEmbeds, seqLen: seqLen,
                                    ttsPadId: ttsPadId, maxFrames: maxFrames, sampling: resolveDecode(decode),
                                    prebuiltInputsBuf: prebuiltInputsBuf)
        let t1 = prof ? CFAbsoluteTimeGetCurrent() : 0
        let nFrames = gen.count
        guard nFrames > 0 else { throw GenerateError.noFramesGenerated }
        // Per-frame [16] ids → [16, frames] row-major (codes[codebook*frames + frame]) for decodeCodec.
        var codes = [Int32](repeating: 0, count: 16 * nFrames)
        for f in 0..<nFrames { for c in 0..<16 { codes[c * nFrames + f] = Int32(gen[f][c]) } }
        let wav = try decodeCodec(codes: codes, frames: nFrames)
        if prof {
            FileHandle.standardError.write(Data(String(
                format: "  [phase] generateCodes=%.2fs decodeCodec=%.2fs\n",
                t1 - t0, CFAbsoluteTimeGetCurrent() - t1).utf8))
        }
        return wav
    }

    // MARK: - streaming generate

    /// One emitted audio chunk. The concatenation of all chunks' `samples` is BIT-EXACT equal
    /// to the offline `generate` output for the same inputs and decode mode. The stream always
    /// terminates with an `isFinal` chunk (possibly empty when the last boundary landed on EOS) —
    /// unless the consumer cancelled by returning `false`.
    public struct StreamChunk {
        public let samples: [Float]      // 24 kHz mono, frameCount × 1920 samples
        public let frameOffset: Int      // first frame index of this chunk
        public let frameCount: Int
        public let isFinal: Bool
    }

    /// Streaming text→24 kHz: audio chunks are emitted DURING generation (the codec decodes
    /// incrementally via Qwen3TTSCodecStream), so time-to-first-audio is the front-end + prefill +
    /// `firstChunkFrames` frames + one small codec chunk instead of the full utterance wall.
    /// Chunk sizes double from `firstChunkFrames` up to `maxChunkFrames` (low TTFA first, fewer
    /// boundaries later — each emitted chunk banks playback runway). Return `false` from `onChunk`
    /// to cancel the remaining generation (barge-in).
    ///
    /// Schedule choice (nil = the package's declared CAM loop default):
    /// 1/1 is zero-buffer minimum TTFA — the codec chunk cost scales with
    /// its frame count, so each 1-frame chunk (~45ms work on the 0.6B)
    /// banks 80ms of audio (warm effTTFA 0.083s 0.6B / 0.127s 1.7B, vs
    /// 0.195s/0.253s at 4/4; testTTSGPUStreamingTTFAProfile). 4/4 is
    /// zero-buffer at the best long-form wall. Intermediate caps (1/2, 1/4)
    /// need a small (≤15ms 0.6B / ≤45ms 1.7B) consumer buffer: the 2-frame
    /// second chunk outruns the single banked frame.
    public func generateStreaming(text: String, instruct: String?, language: String,
                                  speaker: String? = nil, maxFrames: Int,
                                  decode: Qwen3TTSSampler.DecodeMode = .packageDefault,
                                  firstChunkFrames: Int? = nil, maxChunkFrames: Int? = nil,
                                  trace: SmeltRuntimeTraceRecorder? = nil,
                                  onChunk: (StreamChunk) throws -> Bool) throws {
        let fe = try frontEndPrefillHiddenA(text: text, instruct: instruct, language: language, speaker: speaker)
        try generateStreaming(inputsEmbeds: [], seqLen: fe.seqLen, ttsPadId: fe.ttsPadId,
                              maxFrames: maxFrames, decode: decode, firstChunkFrames: firstChunkFrames,
                              maxChunkFrames: maxChunkFrames, trace: trace,
                              onChunk: onChunk, prebuiltInputsBuf: fe.buf)
    }

    /// Resolved chunk-schedule defaults: the manifest's declared CAM loop
    /// schedule is the one truth. Packages without that declaration fail closed.
    public func defaultChunkSchedule() throws -> (first: Int, max: Int) {
        guard let loop = manifest.loop else {
            throw GenerateError.malformedSchedule(
                "package manifest must declare a CAM loop schedule")
        }
        if case .chunked(let first, let max, _, _) = loop.emission {
            return (first, max)
        }
        throw GenerateError.malformedSchedule(
            "the TTS streaming path requires chunked emission")
    }

    public func generateStreaming(inputsEmbeds: [Float], seqLen: Int, ttsPadId: Int, maxFrames: Int,
                                  decode: Qwen3TTSSampler.DecodeMode = .packageDefault,
                                  firstChunkFrames: Int? = nil, maxChunkFrames: Int? = nil,
                                  trace: SmeltRuntimeTraceRecorder? = nil,
                                  onChunk: (StreamChunk) throws -> Bool,
                                  prebuiltInputsBuf: MTLBuffer? = nil) throws {
        let declared = try defaultChunkSchedule()
        let first = firstChunkFrames ?? declared.first
        // The max fallback couples to the RESOLVED first — an explicit
        // firstChunkFrames: 4 on a declared-1/1 package must not trap on
        // the precondition (same rule as the CLI/serve surfaces).
        let maxChunk = maxChunkFrames ?? max(first, declared.max)
        precondition(first >= 1 && maxChunk >= first,
                     "chunk schedule: need 1 <= firstChunkFrames <= maxChunkFrames")
        // B2.2: the declared schedule drives the loop — the generic
        // orchestrator owns phase order, command-buffer scopes, emission
        // chunking, and stops; the blocks compose the SAME session halves
        // the monolithic `generateCodes` loop (offline + gates) drives.
        SmeltDecodeProfile.setStage("setup")
        let session = try TalkerSession(
            gpu: self, inputsBuf: try resolveInputsBuf(inputsEmbeds, seqLen: seqLen, prebuilt: prebuiltInputsBuf),
            seqLen: seqLen, ttsPadId: ttsPadId, maxFrames: maxFrames, sampling: resolveDecode(decode))
        let stream = try Qwen3TTSCodecStream(gpu: self, maxFrames: maxFrames)
        guard let schedule = manifest.loop else {
            throw GenerateError.malformedSchedule(
                "package manifest must declare a CAM loop schedule")
        }
        // A declared schedule must be structurally valid against the package's
        // block graph, and the audio path is chunked-emission-only: per-step
        // emission carries no samples (runPerStepEmission emits samples: []), so
        // a package whose loop declares .perStep would stream SILENT audio rather
        // than fail. Reject it (and any structural mismatch) at the seam.
        guard let graph = manifest.blocks else {
            throw GenerateError.malformedSchedule(
                "package manifest must declare a CAM block graph")
        }
        try schedule.validate(against: graph)
        guard case .chunked = schedule.emission else {
            throw GenerateError.malformedSchedule(
                "the TTS streaming path requires chunked emission (audio is decoded "
                + "in chunks); this package's loop declares per-step emission")
        }
        let routesByBlock = Dictionary(graph.blocks.map {
            ($0.name, "\($0.impl.rawValue):\($0.compiledDelivery?.rawValue ?? "native")")
        }, uniquingKeysWith: { first, _ in first })
        let loop = try SmeltScheduledLoop<[Int]>(
            schedule: schedule,
            blocks: [Qwen3TTSTalkerBlock(session),
                     Qwen3TTSCb0HeadBlock(session),
                     Qwen3TTSMTPHeadBlock(session)],
            scope: { [self] body in try batched(body) },
            trace: trace,
            routesByBlock: routesByBlock)
        do {
            try loop.run(maxSteps: maxFrames,
                         chunkOverride: (first, maxChunk),
                         decoder: Qwen3TTSCodecStreamDecoder(stream)) { chunk in
                try onChunk(StreamChunk(samples: chunk.samples, frameOffset: chunk.offset,
                                        frameCount: chunk.count, isFinal: chunk.isFinal))
            }
        } catch is SmeltScheduledLoop<[Int]>.LoopError {
            // Keep the public error contract callers already handle.
            throw GenerateError.noFramesGenerated
        }
    }
}
