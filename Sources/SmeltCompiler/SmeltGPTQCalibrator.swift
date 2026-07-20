// SmeltGPTQCalibrator — produces the GPTQ u4 blocks Qwen3TTSPackageBuilder.build(gptqBlocks:) consumes,
// by capturing each projection's activation Hessian on the bf16 package and GPTQ-quantizing it.
//
// Streaming, peak-bounded by design. GPTQ needs a weight's COMPLETE Hessian (all calibration tokens)
// before it can quantize that weight, so one calibration pass finalizes every captured weight's H at
// once — capturing all ~196 proj at once peaks at ~7-8 GiB (down_proj K=6144 → 151 MB each × 28). So we
// partition the proj weights into groups of ≤ `layersPerPass` layers and run ONE calibration pass per
// group, capturing only that group's H (via captureHessianNames), quantizing it to blocks, then freeing
// the group's H before the next pass. Peak H RAM = one group; the emitted blocks are tiny (~0.84 GB for
// all proj). Cost = (#groups) × one calibration pass.

import Foundation
import Metal
import SmeltRuntime
import SmeltSchema

public enum SmeltGPTQCalibrator {

    public struct Prompt: Sendable {
        public let text: String, instruct: String, language: String
        public init(text: String, instruct: String, language: String) {
            self.text = text; self.instruct = instruct; self.language = language
        }
    }

    public enum CalibrationError: Error, CustomStringConvertible {
        case noProjWeights(String)
        case missingHessian(String)
        case badWeight(String, String)
        public var description: String {
            switch self {
            case let .noProjWeights(dir): return "no GPTQ-scope projection weights in checkpoint: \(dir)"
            case let .missingHessian(name): return "no Hessian captured for \(name) (weight never dispatched during calibration)"
            case let .badWeight(name, why): return "\(name): \(why)"
            }
        }
    }

    /// GPTQ-quantize every in-scope projection weight (`isFP16Candidate`) of `checkpointDir`, calibrating
    /// activations on the bf16 package at `bf16PackagePath`. Returns the u4 blocks (feed to
    /// `Qwen3TTSPackageBuilder.build(gptqBlocks:)`) and the achieved Hessian rank per weight (= min(calib
    /// tokens, K); the GPTQ rank-deficiency signal). `layersPerPass` bounds peak H RAM (one layer-group);
    /// `Int.max` = single-pass capture-all.
    public static func calibrate(
        bf16PackagePath: String,
        checkpointDir: String,
        device: MTLDevice,
        prompts: [Prompt],
        layersPerPass: Int,
        groupSize: Int = 64,
        damping: Float = 0.01,
        maxFrames: Int = 400
    ) throws -> (blocks: [String: SmeltAffineU4.Packed], ranks: [String: Int]) {
        precondition(layersPerPass >= 1, "layersPerPass must be ≥ 1")
        precondition(!prompts.isEmpty, "calibration needs ≥ 1 prompt")

        // The GPTQ scope = the projection matmuls (isFP16Candidate). text_embedding is u4 but a gather
        // (no activation Hessian) — out of scope, stays affine in the builder.
        let loader = try SafetensorsLoader(directory: checkpointDir)
        let infoByName = Dictionary(loader.tensors.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        let projNames = loader.tensors.map(\.name).filter(Qwen3TTSPackageBuilder.isFP16Candidate)
        guard !projNames.isEmpty else { throw CalibrationError.noProjWeights(checkpointDir) }
        let groups = layerGroups(projNames, layersPerPass: layersPerPass)

        // Phase 4 U2: capture rides the COMPILED trunk sidecars (the hand talker is retired), so
        // generateCodes runs the compiled trunk. We derive each prompt's teacher-forced capture
        // sources ONCE, then capture per layer-group through the trunk/ + trunk-mtp/ runtimes.
        let gpu = try Qwen3TTSGPU(packagePath: bf16PackagePath, device: device)
        precondition(gpu.willRunBatched, "calibration capture needs batched mode (not SMELT_DECODE_PROFILE unbatched)")
        let talkerTrunk = try SmeltRuntime(packagePath: "\(bf16PackagePath)/trunk")
        let mtpTrunk = try SmeltRuntime(packagePath: "\(bf16PackagePath)/trunk-mtp")
        func loadPoints(_ sub: String) throws -> [SmeltGPTQCapturePoint] {
            let url = URL(fileURLWithPath: "\(bf16PackagePath)/\(sub)/gptq_capture_points.json")
            let data = try Data(contentsOf: url)
            let points = try JSONDecoder().decode(SmeltGPTQCapturePoints.self, from: data)
            return points.prefill
        }
        let talkerPoints = try loadPoints("trunk")
        let mtpPoints = try loadPoints("trunk-mtp")

        // Per prompt: compiled generateCodes → codes + per-frame talker hiddens; reconstruct the
        // teacher-forced talker prefill source [prompt embeds ++ decode-frame inputs] and the 16-row
        // MTP seeds. This is the only expensive step; the per-group captures reuse these sources.
        struct PromptSources { let talker: [Float]; let talkerLen: Int; let mtpSeeds: [[Float]] }
        var sources: [PromptSources] = []
        for p in prompts {
            let e = try gpu.inputsEmbeds(text: p.text, instruct: p.instruct, language: p.language)
            var hiddens: [[Float]] = []
            let codes: [[Int]] = try autoreleasepool {
                try gpu.generateCodes(
                    inputsEmbeds: e.embeds, seqLen: e.seqLen, ttsPadId: e.ttsPadId,
                    maxFrames: maxFrames, talkerHiddenTap: { hiddens.append($0) })
            }
            guard !codes.isEmpty else { continue }
            let decodeInputs = try gpu.decodeFrameInputs(codes: codes, ttsPadId: e.ttsPadId)
            sources.append(PromptSources(
                talker: e.embeds + decodeInputs.flatMap { $0 },
                talkerLen: e.seqLen + decodeInputs.count,
                mtpSeeds: try gpu.mtpCalibrationSeeds(codes: codes, talkerHiddens: hiddens)))
        }
        guard !sources.isEmpty else { throw CalibrationError.noProjWeights("no frames generated for calibration") }

        // Pin RoPE + grow capacity ONCE (the rope table for the max length covers shorter prompts).
        let maxTalkerLen = sources.map(\.talkerLen).max()!
        let talkerHidden = try gpu.prepareTrunkForCapture(talkerTrunk, seqLen: maxTalkerLen, mtp: false)
        let mtpHidden = try gpu.prepareTrunkForCapture(mtpTrunk, seqLen: 16, mtp: true)

        var blocks: [String: SmeltAffineU4.Packed] = [:]
        var ranks: [String: Int] = [:]
        for group in groups {
            // Talker (net 0) and MTP (net 1) trunks emit IDENTICAL canonical names, so capture each
            // through its OWN SmeltActivationCapture (codex U2c #1). Capture ONLY this group's weights.
            let talkerCanon = Set(group.compactMap(trunkCanonical).filter { $0.net == 0 }.map(\.canonical))
            let mtpCanon = Set(group.compactMap(trunkCanonical).filter { $0.net == 1 }.map(\.canonical))
            let talkerCap = SmeltActivationCapture(); talkerCap.captureHessian = true
            talkerCap.captureHessianNames = talkerCanon
            let mtpCap = SmeltActivationCapture(); mtpCap.captureHessian = true
            mtpCap.captureHessianNames = mtpCanon
            for s in sources {
                try autoreleasepool {
                    if !talkerCanon.isEmpty {
                        let src = device.makeBuffer(bytes: s.talker, length: s.talker.count * 4,
                                                    options: .storageModeShared)!
                        try talkerTrunk.captureGPTQActivationsFromHidden(
                            source: src, seqLen: s.talkerLen, hidden: talkerHidden,
                            capturePoints: talkerPoints, into: talkerCap)
                    }
                    if !mtpCanon.isEmpty {
                        for seed in s.mtpSeeds {   // each frame's 16-row MTP frame (KV resets per frame)
                            let src = device.makeBuffer(bytes: seed, length: seed.count * 4,
                                                        options: .storageModeShared)!
                            try mtpTrunk.captureGPTQActivationsFromHidden(
                                source: src, seqLen: 16, hidden: mtpHidden,
                                capturePoints: mtpPoints, into: mtpCap)
                        }
                    }
                }
            }

            for name in group {
                guard let (net, canon) = trunkCanonical(name) else {
                    throw CalibrationError.badWeight(name, "not an in-scope talker/MTP projection")
                }
                guard let info = infoByName[name] else { throw CalibrationError.missingHessian(name) }
                let cap = net == 0 ? talkerCap : mtpCap
                guard let hessian = cap.hessian(canon) else { throw CalibrationError.missingHessian(name) }
                guard info.shape.count == 2 else {
                    throw CalibrationError.badWeight(name, "expected a 2-D [out,in] proj weight, got shape \(info.shape)")
                }
                let (rows, cols) = (info.shape[0], info.shape[1])
                let weights = try widenToF32(loader.tensorData(info), dtype: info.dtype, count: rows * cols, name: name)
                ranks[name] = min(cap.calibrationRows(canon), cols)
                blocks[name] = SmeltGPTQ.quantize(weights: weights, rows: rows, cols: cols,
                                                  groupSize: groupSize, hessian: hessian, damping: damping)
            }
        }
        return (blocks, ranks)
    }

    /// `(net, canonical)` for an in-scope HF projection name; nil otherwise. The talker (net 0,
    /// `talker.model.layers.N.*`) and MTP (net 1, `talker.code_predictor.model.layers.N.*`) trunks
    /// emit the SAME canonical `layers_N_*` names, so the net disambiguates which capture/trunk a
    /// weight came from. Allowlists the seven per-layer projection suffixes (codex U2c #5) — so
    /// small_to_mtp_projection / lm_head / codec_embedding (NOT GPTQ scope) map to nil.
    public static func trunkCanonical(_ hfName: String) -> (net: Int, canonical: String)? {
        let net: Int
        let rest: Substring
        if hfName.hasPrefix("talker.code_predictor.model.layers.") {
            net = 1; rest = hfName.dropFirst("talker.code_predictor.model.".count)
        } else if hfName.hasPrefix("talker.model.layers.") {
            net = 0; rest = hfName.dropFirst("talker.model.".count)
        } else {
            return nil
        }
        let suffixes = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                        "mlp.down_proj.weight"]
        guard suffixes.contains(where: { rest.hasSuffix($0) }) else { return nil }
        return (net, rest.replacingOccurrences(of: ".", with: "_"))
    }

    /// A layer's identity: MTP (`code_predictor`, net 1) sorts after talker (net 0); within a network, by
    /// ascending layer index. A name without a `.layers.N.` segment (none expected in scope) sorts last.
    private struct LayerKey: Hashable, Comparable {
        let net: Int, idx: Int
        static func < (a: LayerKey, b: LayerKey) -> Bool { (a.net, a.idx) < (b.net, b.idx) }
    }

    /// Partition `items` into groups of ≤ `perPass` distinct keys (sorted), so each group's combined
    /// Hessian fits the RAM budget. What matters is the per-pass key count, not the key's meaning.
    private static func grouped<T, K: Comparable & Hashable>(
        _ items: [T], by key: (T) -> K, perPass: Int
    ) -> [[T]] {
        var byKey: [K: [T]] = [:]
        for item in items { byKey[key(item), default: []].append(item) }
        let keys = byKey.keys.sorted()
        return stride(from: 0, to: keys.count, by: perPass).map { start in
            keys[start..<min(start + perPass, keys.count)].flatMap { byKey[$0]! }
        }
    }

    /// Partition `names` into groups of ≤ `layersPerPass` distinct layers.
    static func layerGroups(_ names: [String], layersPerPass: Int) -> [[String]] {
        func key(_ name: String) -> LayerKey {
            let net = name.contains("code_predictor") ? 1 : 0
            guard let r = name.range(of: ".layers.") else { return LayerKey(net: net, idx: .max) }
            return LayerKey(net: net, idx: Int(name[r.upperBound...].prefix { $0.isNumber }) ?? .max)
        }
        return grouped(names, by: key, perPass: layersPerPass)
    }

    /// Widen a row-major weight tensor (F16/BF16/F32 source) to fp32 — the values the builder's u4 path
    /// quantizes (widened bit-exact, the same as the gemv kernels), so a block built here matches.
    static func widenToF32(_ src: UnsafeRawPointer, dtype: String, count: Int, name: String) throws -> [Float] {
        var out = [Float](repeating: 0, count: count)
        switch dtype {
        case "F32":
            for i in 0..<count { out[i] = src.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
        case "BF16":
            for i in 0..<count {
                let bits = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                out[i] = Float(bitPattern: UInt32(bits) << 16)
            }
        case "F16":
            for i in 0..<count {
                let bits = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                out[i] = Float(Float16(bitPattern: bits))
            }
        default:
            throw CalibrationError.badWeight(name, "unsupported source dtype \(dtype) (expected F16, BF16, or F32)")
        }
        return out
    }

    /// Partition capture points into groups of ≤ `layersPerPass` distinct layers, so each group's
    /// combined Hessian fits the RAM budget. The layer id is read from the canonical Smelt name
    /// (`layers_N_…`) the prefill emitter constructs — stable, unlike the TTS dotted-name parse.
    static func capturePointGroups(
        _ points: [SmeltGPTQCapturePoint], layersPerPass: Int
    ) -> [[SmeltGPTQCapturePoint]] {
        func layerIndex(_ name: String) -> Int {
            guard let r = name.range(of: "layers_") else { return .max }
            return Int(name[r.upperBound...].prefix { $0.isNumber }) ?? .max
        }
        return grouped(points, by: { layerIndex($0.weightName) }, perPass: layersPerPass)
    }

    /// GPTQ-quantize every in-scope projection of a built **metal-prefill** package, calibrating
    /// activations through the general `SmeltRuntime` capture hook. Returns the u4 blocks (feed to
    /// `SmeltCompiler.build(gptqBlocks:)`) and the achieved Hessian rank per weight. `layersPerPass`
    /// bounds peak Hessian RAM to one layer-group. The package must carry `gptq_capture_points.json`
    /// (a full-trace build); fp32 weights are loaded from the HF `checkpointDir`.
    public static func calibrateRuntime(
        packagePath: String,
        checkpointDir: String,
        ir: SmeltModelIR,
        calibrationTokens: [[Int32]],
        layersPerPass: Int,
        capturePointsPath: String = "gptq_capture_points.json",
        groupSize: Int? = nil,
        damping: Float = 0.01
    ) throws -> (blocks: [String: SmeltAffineU4.Packed], ranks: [String: Int]) {
        precondition(layersPerPass >= 1, "layersPerPass must be ≥ 1")
        precondition(!calibrationTokens.isEmpty, "calibration needs ≥ 1 token sequence")
        // The layout assigns one group size globally (SmeltWeightPacker), so the IR value matches
        // every entry's; a future per-entry-group layout would need to read entry.groupSize here.
        let gs = groupSize ?? ir.quantization.groupSize

        let capPath = "\(packagePath)/\(capturePointsPath)"
        let capturePoints = try JSONDecoder().decode(
            SmeltGPTQCapturePoints.self,
            from: Data(contentsOf: URL(fileURLWithPath: capPath))
        ).prefill
        guard !capturePoints.isEmpty else { throw CalibrationError.noProjWeights(packagePath) }

        // Map each in-scope Smelt projection name to its fp32 checkpoint weight.
        let adapter: SmeltCheckpointAdapter
        do {
            adapter = try SmeltCheckpointAdapter.authored(for: ir)
        } catch {
            throw CalibrationError.badWeight(ir.modelName, "\(error)")
        }
        let loader = try SafetensorsLoader(directory: checkpointDir)
        var hfByAgent: [String: SafetensorInfo] = [:]
        for info in loader.tensors where adapter.isTextModelTensor(info.name) {
            hfByAgent[adapter.mapName(info.name)] = info
        }

        let runtime = try SmeltRuntime(packagePath: packagePath)

        var blocks: [String: SmeltAffineU4.Packed] = [:]
        var ranks: [String: Int] = [:]
        for group in capturePointGroups(capturePoints, layersPerPass: layersPerPass) {
            // Capture ONLY this group's Hessians, bounding peak [K,K] RAM to one layer-group.
            let cap = SmeltActivationCapture()
            cap.captureHessian = true
            cap.captureHessianNames = Set(group.map(\.weightName))
            for tokens in calibrationTokens {
                // captureGPTQActivations issues one command buffer per boundary; drain per
                // sequence so completed buffers don't accumulate across the whole calibration.
                try autoreleasepool {
                    try runtime.captureGPTQActivations(tokenIds: tokens, capturePoints: group, into: cap)
                }
            }
            for point in group {
                let name = point.weightName
                guard let info = hfByAgent[name] else {
                    throw CalibrationError.badWeight(name, "no checkpoint tensor maps to this projection")
                }
                guard let hessian = cap.hessian(name) else { throw CalibrationError.missingHessian(name) }
                guard info.shape.count == 2 else {
                    throw CalibrationError.badWeight(name, "expected a 2-D [out,in] proj weight, got shape \(info.shape)")
                }
                let (rows, cols) = (info.shape[0], info.shape[1])
                // The checkpoint weight's input dim must match the captured activation
                // width, or a mismatched checkpoint/package/IR would crash GPTQ's
                // hessian.count == cols*cols precondition instead of failing cleanly.
                guard cols == point.k else {
                    throw CalibrationError.badWeight(
                        name, "checkpoint input dim \(cols) != captured activation width \(point.k)")
                }
                let weights = try widenToF32(loader.tensorData(info), dtype: info.dtype, count: rows * cols, name: name)
                ranks[name] = min(cap.calibrationRows(name), cols)
                blocks[name] = SmeltGPTQ.quantize(weights: weights, rows: rows, cols: cols,
                                                  groupSize: gs, hessian: hessian, damping: damping)
            }
        }
        return (blocks, ranks)
    }

    public static func calibrateRuntime(
        packagePath: String,
        checkpointDir: String,
        ir: SmeltModelIR,
        calibrationPolicyJSONPath: String,
        defaultLayersPerPass: Int,
        groupSize: Int? = nil,
        damping: Float = 0.01
    ) throws -> (blocks: [String: SmeltAffineU4.Packed], ranks: [String: Int]) {
        let inputs = try SmeltRuntimeGPTQCalibrationPolicy.tokenIDLinesFromPackageSpecJSON(
            at: calibrationPolicyJSONPath,
            defaultLayersPerPass: defaultLayersPerPass
        )
        return try calibrateRuntime(
            packagePath: packagePath,
            checkpointDir: checkpointDir,
            ir: ir,
            calibrationTokens: inputs.tokenIDs,
            layersPerPass: inputs.policy.layersPerPass,
            capturePointsPath: inputs.policy.capturePointsPath,
            groupSize: groupSize,
            damping: damping
        )
    }
}
