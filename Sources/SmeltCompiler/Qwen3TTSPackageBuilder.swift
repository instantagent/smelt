// Qwen3TTSPackageBuilder — hand-written `.smeltpkg` builder for the multi-network
// Qwen3-TTS model (P6-U3). Bypasses SmeltWeightLayout / quantizer / emitter (all
// LLM-single-stack); reuses only compileMetalLibrary and the package directory
// layout. Emits: model.metallib (all fp32 kernels), weights.bin (every weight
// fp32, page-aligned), manifest.json (Qwen3TTSManifest). The GPU driver (U4)
// loads it with its own per-weight MTLBuffer(bytesNoCopy:) at the page-aligned
// offsets recorded here.

import Foundation
import SmeltRuntime
import SmeltSchema

public enum Qwen3TTSPackageBuilder {

    /// Packed weight element type. fp16/bf16 halve the weight-read bandwidth for the M=1 decode
    /// matmuls; `u4` is group-wise affine int4 (~0.56 B/param at g=64) for the largest matmul/embedding
    /// weights; fp32 is the default / exact path.
    public enum WeightDType: String { case f32, f16, bf16, u4 }

    /// One weight to pack (element count = product(shape)). `groupSize` is required for `u4` (the
    /// column group for the affine scale/bias) and nil otherwise.
    public struct WeightSpec {
        public let name: String
        public let shape: [Int]
        public let dtype: WeightDType
        public let groupSize: Int?
        public init(name: String, shape: [Int], dtype: WeightDType = .f32, groupSize: Int? = nil) {
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.groupSize = groupSize
        }
        public var elementCount: Int { shape.reduce(1, *) }
        /// The weight's primary data bytes (for u4: the packed nibbles; scales/biases live in sibling
        /// regions of the same block — see `u4Layout`). Used for non-u4 region sizing + size checks.
        public var dataBytes: Int {
            switch dtype {
            case .f32: return elementCount * 4
            case .f16, .bf16: return elementCount * 2
            case .u4:
                let cols = shape.count > 1 ? shape[1] : 1
                return shape[0] * SmeltAffineU4.packedRowStride(cols: cols)
            }
        }
    }

    /// Byte layout of one u4 weight block: page-aligned nibbles, then page-aligned fp16 scales, then
    /// page-aligned fp16 biases. `scaleOffset`/`biasOffset` are RELATIVE to the block start, so the
    /// runtime can bind a single mmap'd buffer at three offsets.
    struct U4BlockLayout {
        let scaleBytes: Int, biasBytes: Int           // unrounded region sizes (the manifest byteLengths)
        let scaleOffset: UInt64, biasOffset: UInt64, blockBytes: UInt64
    }

    static func u4Layout(shape: [Int], groupSize: Int, pageSize: Int) -> U4BlockLayout {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
        let nibbleBytes = rows * SmeltAffineU4.packedRowStride(cols: cols)
        let scaleBytes = rows * groups * 2
        let biasBytes = rows * groups * 2
        let p = UInt64(pageSize)
        let nibbleLen = pageAlign(UInt64(nibbleBytes), p)
        let scaleLen = pageAlign(UInt64(scaleBytes), p)
        let biasLen = pageAlign(UInt64(biasBytes), p)
        return U4BlockLayout(
            scaleBytes: scaleBytes, biasBytes: biasBytes,
            scaleOffset: nibbleLen, biasOffset: nibbleLen + scaleLen,
            blockBytes: nibbleLen + scaleLen + biasLen)
    }

    /// The u4 GEMV kernel's contract (see gemv_u4_f32.metal): K and group_size multiples of 4,
    /// group_size ≥ 4, and ceil(K/group_size) ≤ U4_MAX_GROUPS. Enforced loudly here at the build
    /// chokepoint so a misconfigured group never reaches the kernel as silent threadgroup OOB / wrong
    /// scale indexing. `U4_MAX_GROUPS` mirrors the kernel constant.
    public static let u4MaxGroups = 256
    public static func validateU4(_ spec: WeightSpec) throws {
        guard let g = spec.groupSize else { throw BuilderError.invalidU4(spec.name, "missing groupSize") }
        let cols = spec.shape.count > 1 ? spec.shape[1] : 1
        guard spec.shape.count == 2 else { throw BuilderError.invalidU4(spec.name, "u4 needs a 2-D weight, got shape \(spec.shape)") }
        guard g >= 4, g % 4 == 0 else { throw BuilderError.invalidU4(spec.name, "group_size \(g) must be a multiple of 4 ≥ 4") }
        guard cols % 4 == 0 else { throw BuilderError.invalidU4(spec.name, "cols \(cols) must be a multiple of 4") }
        let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: g)
        guard groups <= u4MaxGroups else { throw BuilderError.invalidU4(spec.name, "groups \(groups) exceeds U4_MAX_GROUPS \(u4MaxGroups)") }
    }

    /// The kernels the package carries — the catalog entry-point names for the fp32 TTS
    /// pipelines, kept DRY with SmeltKernelCatalog.
    public static let ttsPipelines: [SmeltPipeline] = [
        .snakeBetaF32, .conv1dForwardF32, .convTranspose1dF32, .layerNormCTF32,
        .matmulF32, .geluF32, .siluF32, .swigluF32, .rmsNormCodecF32, .rmsNormHeadF32,
        .ropeApplyF32, .slidingAttnF32, .causalGQAAttnF32, .rvqGatherSumF32,
        .scaleResidualF32, .scaleResidualTCF32, .clampF32, .decodeGQAAttnF32, .matmulF16WF32,
        .gemvF32, .gemvF16WF32, .argmaxF32, .gatherRowF32, .gemmF32, .gemmF16WF32, .channelCopyF32,
        .nextFrameInputF32, .cb0ArgmaxF32, .gemvBF16WF32, .gemmBF16WF32, .sampleTopKF32, .cb0SampleTopKF32,
        .gatherRowsF32, .gatherRowsBF16WF32,
        .gemvU4F32, .gemmU4F32, .gatherRowBF16WF32, .nextFrameInputBF16WF32,
        .gemmTNF32, .gemmTNF16WF32, .gemmTNBF16WF32, .causalGQAAttnSimdF32, .slidingAttnSimdF32,
        .gemvQKVBF16WF32, .gemvGateUpSwigluBF16WF32,
        .headNormRopeF32, .gemvAddBF16WF32, .transposeF32,
    ]

    public static var ttsPipelineNames: [String] {
        ttsPipelines.map { SmeltKernelCatalog.signature(for: $0).metalFunctionName }
    }

    static func pageAlign(_ x: UInt64, _ page: UInt64) -> UInt64 {
        (x + page - 1) & ~(page - 1)
    }

    static func isPowerOfTwo(_ n: Int) -> Bool { n > 0 && (n & (n - 1)) == 0 }

    /// Plan page-aligned, page-rounded offsets for the weight list. Pure — the
    /// builder gate exercises this directly. Each weight starts page-aligned and
    /// occupies a page-rounded byteLength so U4 can wrap it with
    /// MTLBuffer(bytesNoCopy:) (which requires a page-aligned pointer).
    public static func planLayout(
        _ specs: [WeightSpec],
        pageSize: Int
    ) -> (entries: [Qwen3TTSManifest.Entry], totalBytes: UInt64) {
        // pageAlign's bit-mask only aligns to a power-of-two page; bytesNoCopy needs
        // exactly getpagesize() alignment, so reject anything else up front.
        precondition(isPowerOfTwo(pageSize), "pageSize \(pageSize) must be a power of two")
        let page = UInt64(pageSize)
        var cursor: UInt64 = 0
        var entries: [Qwen3TTSManifest.Entry] = []
        entries.reserveCapacity(specs.count)
        for s in specs {
            let off = pageAlign(cursor, page)
            if s.dtype == .u4, let g = s.groupSize {
                let lay = u4Layout(shape: s.shape, groupSize: g, pageSize: pageSize)
                entries.append(.init(name: s.name, offset: off, byteLength: lay.blockBytes, shape: s.shape,
                                     dtype: "u4", groupSize: g,
                                     scaleOffset: lay.scaleOffset, scaleByteLength: UInt64(lay.scaleBytes),
                                     biasOffset: lay.biasOffset, biasByteLength: UInt64(lay.biasBytes)))
                cursor = off + lay.blockBytes
            } else {
                let len = pageAlign(UInt64(s.dataBytes), page)
                entries.append(.init(name: s.name, offset: off, byteLength: len, shape: s.shape,
                                     dtype: s.dtype == .f32 ? nil : s.dtype.rawValue))
                cursor = off + len
            }
        }
        return (entries, cursor)   // cursor is already page-aligned (last len page-rounded)
    }

    /// Core build shared by the real and synthetic paths. `fill` is handed the planned `Entry` and a
    /// slice covering the weight's whole page-aligned block; it writes the weight's bytes (for u4, the
    /// packed nibbles at slice[0] and the scale/bias regions at `entry.scaleOffset`/`biasOffset`). The
    /// page-padding between weights stays zero.
    public static func build(
        specs: [WeightSpec],
        fill: (WeightSpec, Qwen3TTSManifest.Entry, UnsafeMutableRawBufferPointer) throws -> Void,
        pipelines: [String],
        modelName: String,
        eosTokens: [Int32],
        shaderDir: String,
        outputPath: String,
        tokenizerSourceDir: String? = nil,
        tokenizerFiles: [String] = [],
        decode: Qwen3TTSManifest.Decode? = nil,
        pageSize: Int = Int(getpagesize())
    ) throws {
        // An empty set would ftruncate to 0 and mmap(0) → EINVAL, surfacing as an
        // opaque "mmap failed"; reject it up front.
        guard !specs.isEmpty else { throw BuilderError.noCarriedTensors(outputPath) }
        // Enforce the u4 kernel contract at the build chokepoint (both real + synthetic paths).
        for s in specs where s.dtype == .u4 { try validateU4(s) }
        let fm = FileManager.default
        if !fm.fileExists(atPath: outputPath) {
            try fm.createDirectory(atPath: outputPath, withIntermediateDirectories: true)
        }

        try SmeltCompiler.compileMetalLibrary(
            shaderDir: shaderDir, outputPath: "\(outputPath)/model.metallib")

        let (entries, totalBytes) = planLayout(specs, pageSize: pageSize)
        try writeWeights(
            path: "\(outputPath)/weights.bin",
            totalBytes: totalBytes, specs: specs, entries: entries, fill: fill)

        // Bundle the tokenizer/config files so the driver loads them from the package itself.
        var bundled: [String]? = nil
        if let srcDir = tokenizerSourceDir, !tokenizerFiles.isEmpty {
            for f in tokenizerFiles {
                // A bare basename only — reject `/` or `..` so a tokenizerFiles entry can't
                // removeItem/copyItem outside the package via path traversal.
                guard !f.isEmpty, !f.contains("/"), !f.contains("..") else {
                    throw BuilderError.tokenizerPathInvalid(f)
                }
                // Resolve symlinks before copying: HF-cache snapshots are directories of relative
                // symlinks into ../../blobs, and copyItem preserves a symlink as-is — the bundled
                // copy would dangle once inside the package.
                let src = URL(fileURLWithPath: "\(srcDir)/\(f)").resolvingSymlinksInPath().path
                let dst = "\(outputPath)/\(f)"
                guard fm.fileExists(atPath: src) else { throw BuilderError.tokenizerFileMissing(src) }
                guard src != dst else { continue }  // in-place package: removeItem(dst) would delete src
                if fm.fileExists(atPath: dst) { try fm.removeItem(atPath: dst) }
                try fm.copyItem(atPath: src, toPath: dst)
            }
            bundled = tokenizerFiles
        }

        // Codec-only packages carry their own block contract: codec frames in,
        // audio out. A full text→audio graph would be a lie for them, but a
        // graphless root is not a CAM package.
        let isFullPipeline = entries.contains { $0.name.hasPrefix("talker.") }
        // The compiled trunks ship all-or-none (see shouldShipTrunks): every emittable dtype
        // (f32/f16/bf16/u4) emits BOTH the trunk/ and trunk-mtp/ sidecars. The hand talker is retired,
        // so a full-pipeline package can only run if it ships trunks; the only non-shipping case left
        // is a cross-network dtype MIX (talker dtype != MTP dtype) or an unknown dtype — refuse to emit
        // that rather than stamp a graph the runtime would reject (block-graph honesty).
        let shipsTrunk = Self.shouldShipTrunks(entries)
        if isFullPipeline && !shipsTrunk {
            throw BuilderError.unrunnableFullPipeline(
                "talker and code_predictor must share one emittable trunk dtype (f32/f16/bf16/u4); "
                + "a cross-network dtype mix can't ship trunks and so can't run generation")
        }
        func makeManifest(_ blocks: SmeltBlockGraph) -> Qwen3TTSManifest {
            let structureProfile: SmeltPackageSpec.Validation.StructureProfile?
            if isFullPipeline, let bundled {
                structureProfile = SmeltPackageStructureProfiles.qwen3TTSRunnable(
                    pipelines: pipelines,
                    tokenizerFiles: bundled,
                    graph: blocks
                )
            } else {
                structureProfile = nil
            }
            let validation = isFullPipeline && bundled != nil
                ? SmeltPackagePerformanceProfiles.validation(
                    parityFixture: modelName,
                    performanceGate: SmeltQwen3TTSPackageProfiles.runnable.performanceGate,
                    structureProfile: structureProfile)
                : nil
            let loop = blocks == .qwen3TTSCodecDecoder
                ? SmeltLoopSchedule.qwen3TTSCodecDecoder
                : SmeltQwen3TTSPackageProfiles.runnable.loop
            return Qwen3TTSManifest(
                version: 1, blocks: blocks,
                loop: loop,
                modelName: modelName, pageSize: pageSize,
                pipelines: pipelines, eosTokens: eosTokens,
                totalBytes: totalBytes, weights: entries, tokenizerFiles: bundled, decode: decode,
                validation: validation)
        }
        // Emit the trunk sidecar FIRST (atomic temp+rename, full shape/dtype validation).
        // The parent manifest claims `.qwen3TTSCompiledTalker` ONLY after the trunk/ is
        // fully written, so a failed emit aborts the build rather than leaving a package
        // that advertises a compiled talker it doesn't carry.
        var blocks: SmeltBlockGraph = isFullPipeline ? .qwen3TTS : .qwen3TTSCodecDecoder
        if shipsTrunk {
            // Two co-resident compiled transformers, both sharing weights.bin: the MAIN talker
            // trunk (B3.2c, trunk/) and the MTP/code_predictor transformer (B3.2d, trunk-mtp/).
            // ALL-OR-NONE: prepare (fully resolve, no FS writes) BOTH specs first — proving every
            // tensor at every layer is present + emittable for each — then commit both. So a deep
            // failure in one trunk (a non-q0 u4 projection, a missing lm_head.0) can't half-build
            // the package by committing the other first; shouldShipTrunks' layer-0 sentinel only
            // decides INTENT, prepare proves full emitability.
            let trunkManifest = makeManifest(SmeltQwen3TTSPackageProfiles.runnable.baseGraph)
            let talkerTrunk = try Qwen3TTSTrunkSidecar.prepare(manifest: trunkManifest, spec: .talker)
            let mtpTrunk = try Qwen3TTSTrunkSidecar.prepare(manifest: trunkManifest, spec: .mtp)
            try Qwen3TTSTrunkSidecar.commit(talkerTrunk, intoPackage: outputPath)
            try Qwen3TTSTrunkSidecar.commit(mtpTrunk, intoPackage: outputPath)
            // The trunks are compiled (shipsTrunk gated that); the FRONT-END is compiled only
            // when the runtime can run it — and the runtime selects the compiled front-end by
            // text_embedding dtype (it gathers bf16; a u4 text_embedding stays the native host
            // gather, Qwen3TTSGPUTalker). So base the front-end-compiled stamp on text_embedding,
            // NOT the trunk q0 — keeping the graph honest even for a custom build(specs:) that
            // mixes a u4 trunk with a bf16 text_embedding (or vice-versa).
            let textEmbedIsBF16 = entries.first {
                $0.name == "talker.model.text_embedding.weight"
            }?.dtype == "bf16"
            blocks = SmeltQwen3TTSPackageProfiles.runnable.graph(
                textEmbeddingIsBF16: textEmbedIsBF16
            )
        }
        try makeManifest(blocks).encoded().write(to: URL(fileURLWithPath: "\(outputPath)/manifest.json"))
    }

    /// The projection dtype STRINGS the compiled trunk can EMIT (dtype picks a kernel lego): f32/f16 via
    /// the unfused dense route, bf16 fused, u4 unfused affine. Single source of truth for the manifest-
    /// string ship/share gates (`shouldShipTrunks` + `Qwen3TTSTrunkSidecar.canShareWeights`). The
    /// emitters' per-projection dispatch keys off the resolved `WeightLocator.Kind` exhaustive switch,
    /// NOT this string set — so the enum-level dispatch stays compile-checked while the two string-level
    /// gates can't drift on which dtypes ship.
    public static let emittableTrunkDtypes: Set<String> = ["f32", "f16", "bf16", "u4"]

    /// The cheap INTENT sentinel for shipping the co-resident dense trunks (talker `trunk/` +
    /// MTP `trunk-mtp/`): a full text→audio pipeline, the talker.model.codec_embedding
    /// vocab-ref, AND bf16 layer-0 q_proj in BOTH the talker and the code_predictor. It must
    /// stay total + FS-free because it also feeds the parent manifest's block-graph choice.
    /// The actual all-or-none safety is the builder's prepare-both-then-commit-both preflight
    /// (a deep MTP failure can't follow a committed talker sidecar); this sentinel only avoids
    /// even attempting trunks for the common non-bf16/talker-only build, and checking both
    /// networks here makes a mixed bf16-talker/u4-MTP build skip cheaply rather than fail the
    /// preflight loudly.
    public static func shouldShipTrunks(_ entries: [Qwen3TTSManifest.Entry]) -> Bool {
        // nil ⇒ the network's q_proj is ABSENT (can't ship); else its dtype ("f32" if the string is
        // omitted, which planLayout does for f32). Presence is distinct from dtype — both an absent
        // entry and a present-f32 entry have a nil `.dtype`, so the existence check can't be skipped.
        func q0Dtype(_ prefix: String) -> String? {
            guard let e = entries.first(where: { $0.name == "\(prefix)layers.0.self_attn.q_proj.weight" })
            else { return nil }
            return e.dtype ?? "f32"
        }
        let talker = q0Dtype("talker.model."), mtp = q0Dtype("talker.code_predictor.model.")
        // Both networks must be PRESENT and share ONE emittable trunk dtype — ALL of f32/f16 (unfused
        // dense), bf16 (fused), u4 (unfused affine). Dtype picks a kernel lego, never gates shipping.
        // A cross-network mix (e.g. bf16 talker + u4 MTP) is not a real build and ships NEITHER.
        return entries.contains { $0.name.hasPrefix("talker.") }
            && entries.contains { $0.name == "talker.model.codec_embedding.weight" }
            && talker != nil && talker == mtp
            && emittableTrunkDtypes.contains(talker!)
    }

    /// Real build: read every carried tensor from the checkpoint, upcasting to fp32
    /// (codec stored F32 → raw copy; talker/MTP stored BF16 → bits << 16).
    /// True for the talker/MTP internal projection weights (q/k/v/o/gate/up/down_proj) — the
    /// matmul weights the M=1 decode reads. Excludes lm_head/codec_head (argmax-sensitive), norms,
    /// embeddings, text_projection, and the codec — those stay fp32.
    public static func isFP16Candidate(_ name: String) -> Bool {
        guard name.hasPrefix("talker.model.layers.")
                || name.hasPrefix("talker.code_predictor.model.layers.") else { return false }
        return name.hasSuffix("_proj.weight")
    }

    /// bf16-packable weights (BF16 source + a consumer that widens bf16→fp32, bit-exact): the proj
    /// weights PLUS the gathered text_embedding (consumed via weightRow). Norms stay fp32 (the rmsnorm
    /// kernels read fp32); the codec is F32-source. fp16 keeps the proj-only set to avoid flipping a code.
    public static func isBF16Candidate(_ name: String) -> Bool {
        isFP16Candidate(name)
            || name == "talker.model.text_embedding.weight"
            || name == "talker.codec_head.weight"
            || name == "talker.code_predictor.small_to_mtp_projection.weight"
            || (name.hasPrefix("talker.code_predictor.lm_head.") && name.hasSuffix(".weight"))
    }

    /// u4-packable weights: the internal layer projections (matmul, the M=1 decode bandwidth floor)
    /// PLUS the gathered text_embedding (the single biggest tensor, 1.2 GB fp32; dequantized row-wise on
    /// the host in weightRow). Excludes the argmax-sensitive heads (codec_head/lm_head/small_to_mtp) and
    /// the GPU-gathered codec_embeddings (those need a u4 gather kernel) — they stay full-width.
    public static func isU4Candidate(_ name: String) -> Bool {
        isFP16Candidate(name) || name == "talker.model.text_embedding.weight"
    }

    /// Which dtype to pack `name` as, given the requested packed dtype for this build. u4 (lossy) covers
    /// the projections + text_embedding (isU4Candidate); the heads and codec_embeddings stay full-width.
    static func packDType(_ name: String, _ projDType: WeightDType) -> WeightDType {
        switch projDType {
        case .f32: return .f32
        case .f16: return isFP16Candidate(name) ? .f16 : .f32
        case .bf16: return isBF16Candidate(name) ? .bf16 : .f32
        case .u4: return isU4Candidate(name) ? .u4 : .f32
        }
    }

    public static func build(
        checkpointDir: String,
        checkpointPolicy: Qwen3TTSCheckpointTensorPolicy,
        shaderDir: String,
        outputPath: String,
        modelName: String = SmeltQwen3TTSPackageProfiles.runnable.modelName,
        // Single source of truth for the codec EOS — don't re-hardcode 2150 here.
        eosTokens: [Int32] = SmeltQwen3TTSPackageProfiles.runnable.eosTokens,
        // Opt-in packed dtype for the talker/MTP projection weights: .f16 (lossy ×0.5), .bf16
        // (BIT-EXACT ×0.5 — the source checkpoint's native dtype, widened to fp32 in-kernel), or .u4
        // (lossy group-wise affine int4, ~×0.28). Default .f32 leaves the existing exact gates unaffected.
        projDType: WeightDType = .f32,
        // Column group size for u4 packing (ignored otherwise). 64 ≈ 0.5625 B/param.
        u4GroupSize: Int = 64,
        // u4 clip endpoints. Default .minMax: although .mseOptimal lowers WEIGHT reconstruction error,
        // it REGRESSES end-to-end logit fidelity (measured 0.992→0.989) — clamping outlier weights hurts
        // when they sit on high-activation channels. Activation-AWARE clipping (imatrix-weighted) is the
        // real lever; plain weight-MSE is opt-in only. Same footprint either way.
        u4Clip: SmeltAffineU4.ClipMode = .minMax,
        // Per-weight activation importance (E[x²] per input channel, captured via Qwen3TTSActivationCapture)
        // for activation-aware u4 clipping. Only used with u4Clip == .mseOptimal; absent names fall back
        // to unweighted. Same footprint.
        imatrix: [String: [Float]]? = nil,
        // Precomputed GPTQ u4 blocks (from SmeltGPTQCalibrator), keyed by weight name. When present, each
        // is written VERBATIM into its u4 slice instead of the affine quantizer — same block format, so
        // the runtime gemv_u4 path is unchanged. To avoid a silent affine/GPTQ hybrid: every in-scope
        // projection weight (isFP16Candidate, the matmuls GPTQ targets) MUST have a block, and every key
        // must match such a weight — both are enforced below. text_embedding (u4 but a gather, no Hessian)
        // is out of scope and stays affine.
        gptqBlocks: [String: SmeltAffineU4.Packed]? = nil
    ) throws {
        // The talker + code_predictor live in model.safetensors (BF16); the codec decoder
        // lives in speech_tokenizer/model.safetensors (F32) — separate networks in separate
        // files, not shards of one. Each dir may itself be sharded (model-0000N-of-M).
        let fm = FileManager.default
        func safetensorPaths(in dir: String) -> [String] {
            let single = "\(dir)/model.safetensors"
            if fm.fileExists(atPath: single) { return [single] }
            // Sharded: model.safetensors.index.json's weight_map gives the exact shard set —
            // use it rather than globbing model-*.safetensors (which could miss a shard or
            // pick up a stale file, silently producing a partial/duplicate-keyed package).
            let indexPath = "\(dir)/model.safetensors.index.json"
            if let data = try? Data(contentsOf: URL(fileURLWithPath: indexPath)),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = json["weight_map"] as? [String: String] {
                return Set(weightMap.values).sorted().map { "\(dir)/\($0)" }
            }
            return []
        }
        let paths = safetensorPaths(in: checkpointDir)
            + safetensorPaths(in: "\(checkpointDir)/speech_tokenizer")
        guard !paths.isEmpty else { throw BuilderError.noCarriedTensors(checkpointDir) }
        let loader = try SafetensorsLoader(paths: paths)
        var tensorPlans: [String: Qwen3TTSCheckpointTensorPolicy.PlannedTensor] = [:]
        let carried = try loader.tensors.compactMap { tensor -> SafetensorInfo? in
            guard let plan = try checkpointPolicy.tensor(named: tensor.name) else { return nil }
            tensorPlans[tensor.name] = plan
            return tensor
        }.sorted { $0.name < $1.name }
        guard !carried.isEmpty else { throw BuilderError.noCarriedTensors(checkpointDir) }
        let carriedBlocks = Set(tensorPlans.values.map(\.block))
        for required in checkpointPolicy.requiredBlocks.sorted() where !carriedBlocks.contains(required) {
            throw BuilderError.missingTensorBlock(required)
        }
        for required in checkpointPolicy.unmatchedRequiredPatterns(in: carried.map(\.name)) {
            throw BuilderError.missingTensorPattern(
                block: required.block,
                selector: required.selector,
                reason: required.reason
            )
        }
        // EVERY dtype (f32/f16/bf16/u4) ships compiled talker + MTP trunks whose IR HARDCODES
        // rms_eps 1e-6 + rope_theta 1e6 (Qwen3TTSTrunkSidecar.synthesizeIR). The current model's
        // config matches those constants (the parity gates prove it) — but a FUTURE checkpoint whose
        // config disagrees would silently diverge. Fail loud at build (config.json is available) for
        // the talker AND the code_predictor, regardless of projection dtype.
        try validateTrunkConfigConstants(checkpointDir)
        let specs = carried.map { t -> WeightSpec in
            let plan = tensorPlans[t.name]!
            var dt = packDType(t.name, projDType)
            // In a u4 build, also halve the weights u4 doesn't cover (same footprint goal):
            //  - the argmax-sensitive heads (codec_head/lm_head/small_to_mtp) → bf16, BIT-EXACT (BF16
            //    source), matmul'd through the existing gemv_bf16w path.
            //  - the F32 codec decoder's large (rank ≥ 2) conv/transformer weights → bf16 (read back via
            //    f32(), widened — no per-conv-kernel change; bf16 keeps the full exponent so convs tolerate
            //    the narrowing, gated on wav). 1-D norms/biases stay f32.
            if projDType == .u4, dt == .f32 {
                if isBF16Candidate(t.name) { dt = .bf16 }
                else if plan.block == "codec-decoder", t.shape.count >= 2 { dt = .bf16 }
                // The 15 MTP codec_embedding tables (BF16 source) → bf16, BIT-EXACT: gathered via the
                // bf16 gather_row / next_frame_input variants. The talker codec_embedding stays f32 — it
                // is blitted raw (the MTP seed row), which would copy bf16 bytes into an f32 slot.
                else if t.name.hasPrefix("talker.code_predictor.model.codec_embedding.") { dt = .bf16 }
            }
            return WeightSpec(name: t.name, shape: t.shape, dtype: dt,
                              groupSize: dt == .u4 ? u4GroupSize : nil)
        }

        // GPTQ coverage: no silent affine/GPTQ hybrid. Every in-scope proj weight (isFP16Candidate,
        // packed u4) must have a block, and every supplied key must match such a weight.
        if let gptqBlocks {
            let scope = Set(specs.filter { $0.dtype == .u4 && isFP16Candidate($0.name) }.map(\.name))
            for name in scope where gptqBlocks[name] == nil {
                throw BuilderError.invalidGPTQBlock(name, "no GPTQ block for in-scope projection weight")
            }
            for name in gptqBlocks.keys where !scope.contains(name) {
                throw BuilderError.invalidGPTQBlock(name, "GPTQ block for a weight outside the u4 projection scope")
            }
        }

        try build(
            specs: specs,
            fill: { spec, entry, slice in
                let info = loader.tensor(named: spec.name)!
                let src = loader.tensorData(info)
                let source = tensorPlans[spec.name]!.sourceDType
                if spec.dtype == .u4 {
                    // Precomputed GPTQ block → write it verbatim (see fillU4FromBlock).
                    if let block = gptqBlocks?[spec.name] {
                        try fillU4FromBlock(spec: spec, entry: entry, block: block, slice: slice)
                        return
                    }
                    // Unweighted mseOptimal regresses fidelity, so it's only safe WITH activation
                    // importance; a u4 weight lacking an imatrix entry (e.g. text_embedding today) uses
                    // min/max. With importance present, honor the requested clip (mseOptimal = the win).
                    let imp = imatrix?[spec.name]
                    let wClip: SmeltAffineU4.ClipMode = imp != nil ? u4Clip : .minMax
                    try fillU4(spec: spec, entry: entry, srcDtype: info.dtype, srcByteCount: info.byteCount,
                               src: src, source: source, clip: wClip, importance: imp, slice: slice)
                    return
                }
                if spec.dtype == .f16 {
                    // Convert the source (F32 or BF16) → Float16, writing elementCount × 2 bytes.
                    let dst = slice.baseAddress!.assumingMemoryBound(to: Float16.self)
                    switch source {
                    case .f32:
                        guard info.dtype == "F32" else { throw BuilderError.dtypeMismatch(spec.name, expected: "F32", got: info.dtype) }
                        for i in 0..<spec.elementCount { dst[i] = Float16(src.loadUnaligned(fromByteOffset: i * 4, as: Float.self)) }
                    case .bf16:
                        guard info.dtype == "BF16" else { throw BuilderError.dtypeMismatch(spec.name, expected: "BF16", got: info.dtype) }
                        for i in 0..<spec.elementCount {
                            let bits = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                            dst[i] = Float16(Float(bitPattern: UInt32(bits) << 16))
                        }
                    }
                    return
                }
                if spec.dtype == .bf16 {
                    let dst = slice.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    switch source {
                    case .bf16:
                        // BF16 source → raw 2-byte copy, bit-exact (the kernel widens bf16→fp32 exactly).
                        guard info.dtype == "BF16" else { throw BuilderError.dtypeMismatch(spec.name, expected: "BF16", got: info.dtype) }
                        guard info.byteCount == spec.elementCount * 2 else {
                            throw BuilderError.sizeMismatch(spec.name, expected: spec.elementCount * 2, got: info.byteCount)
                        }
                        memcpy(dst, src, spec.elementCount * 2)
                    case .f32:
                        // F32 source → bf16 round-to-nearest-even (lossy ×0.5; for the F32 codec weights,
                        // bf16 keeps the full 8-bit exponent so conv weights tolerate it — gated on wav).
                        guard info.dtype == "F32" else { throw BuilderError.dtypeMismatch(spec.name, expected: "F32", got: info.dtype) }
                        guard info.byteCount == spec.elementCount * 4 else {
                            throw BuilderError.sizeMismatch(spec.name, expected: spec.elementCount * 4, got: info.byteCount)
                        }
                        for i in 0..<spec.elementCount {
                            let b = src.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)
                            let rounded = b &+ (0x7FFF &+ ((b >> 16) & 1))   // round-to-nearest-even
                            dst[i] = UInt16(truncatingIfNeeded: rounded >> 16)
                        }
                    }
                    return
                }
                switch source {
                case .f32:
                    guard info.dtype == "F32" else {
                        throw BuilderError.dtypeMismatch(spec.name, expected: "F32", got: info.dtype)
                    }
                    guard info.byteCount == spec.dataBytes else {
                        throw BuilderError.sizeMismatch(spec.name, expected: spec.dataBytes, got: info.byteCount)
                    }
                    memcpy(slice.baseAddress!, src, spec.dataBytes)
                case .bf16:
                    guard info.dtype == "BF16" else {
                        throw BuilderError.dtypeMismatch(spec.name, expected: "BF16", got: info.dtype)
                    }
                    guard info.byteCount == spec.elementCount * 2 else {
                        throw BuilderError.sizeMismatch(spec.name, expected: spec.elementCount * 2, got: info.byteCount)
                    }
                    // Unaligned BF16 loads: safetensors gives byte offsets, so the source
                    // may not be 2-byte aligned. The fp32 destination is page-aligned.
                    let d = slice.baseAddress!.assumingMemoryBound(to: UInt32.self)
                    for i in 0..<spec.elementCount {
                        let bits = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                        d[i] = UInt32(bits) << 16
                    }
                }
            },
            pipelines: ttsPipelineNames,
            modelName: modelName, eosTokens: eosTokens,
            shaderDir: shaderDir, outputPath: outputPath,
            tokenizerSourceDir: checkpointDir,
            tokenizerFiles: SmeltQwen3TTSPackageProfiles.runnable.tokenizerFiles,
            decode: try readDecodeConfig(checkpointDir))
    }

    /// Validate that the checkpoint's talker + code_predictor `rms_norm_eps` / `rope_theta`
    /// match the constants the compiled trunks HARDCODE (1e-6 / 1e6 — Qwen3TTSTrunkSidecar's
    /// synthesised IR). The current model's config matches them (the parity gates prove it); a
    /// future checkpoint whose config disagrees would diverge silently, so fail loud at build.
    /// Absent config.json / keys → skip (synthetic
    /// or non-HF build).
    static func validateTrunkConfigConstants(_ checkpointDir: String) throws {
        let trunkEps = 1e-6, trunkTheta = 1_000_000.0
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: "\(checkpointDir)/config.json")),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let talker = root["talker_config"] as? [String: Any] else { return }
        func check(_ cfg: [String: Any], _ scope: String) throws {
            if let eps = (cfg["rms_norm_eps"] as? NSNumber)?.doubleValue, eps != trunkEps {
                throw BuilderError.unsupportedTrunkConfig("\(scope).rms_norm_eps=\(eps) != the compiled trunk's \(trunkEps)")
            }
            if let theta = (cfg["rope_theta"] as? NSNumber)?.doubleValue, theta != trunkTheta {
                throw BuilderError.unsupportedTrunkConfig("\(scope).rope_theta=\(theta) != the compiled trunk's \(trunkTheta)")
            }
        }
        try check(talker, "talker_config")
        if let cp = talker["code_predictor_config"] as? [String: Any] {
            try check(cp, "code_predictor_config")
        }
    }

    /// Parse generation_config.json into the manifest decode block. Absent file → nil (the driver
    /// falls back to greedy). When do_sample, top_p/subtalker_top_p must be 1.0 — we implement only
    /// temperature + top-k, so a non-identity nucleus must fail loudly here rather than be silently
    /// ignored at decode time. (Greedy ignores top_p, so it isn't validated then.)
    static func readDecodeConfig(_ checkpointDir: String) throws -> Qwen3TTSManifest.Decode? {
        let path = "\(checkpointDir)/generation_config.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        func num(_ k: String, _ fallback: Double) -> Double { (json[k] as? NSNumber)?.doubleValue ?? fallback }
        let doSample = (json["do_sample"] as? Bool) ?? false
        let temperature = num("temperature", 1.0), topK = num("top_k", 50)
        let subTemp = num("subtalker_temperature", temperature), subTopK = num("subtalker_top_k", topK)
        if doSample {
            for k in ["top_p", "subtalker_top_p"] {
                let p = num(k, 1.0)
                if p != 1.0 { throw BuilderError.unsupportedGenerationConfig("\(k)=\(p); only top_p=1.0 (no nucleus) is implemented") }
            }
            // Catch a config the sampler can't honor at BUILD (the chokepoint) rather than trapping a
            // precondition mid-decode: the sampler needs temperature > 0 and top_k a whole number >= 1.
            for (k, t) in [("temperature", temperature), ("subtalker_temperature", subTemp)] where t <= 0 {
                throw BuilderError.unsupportedGenerationConfig("\(k)=\(t); sampling needs temperature > 0")
            }
            for (k, v) in [("top_k", topK), ("subtalker_top_k", subTopK)] where v < 1 || v != v.rounded() {
                throw BuilderError.unsupportedGenerationConfig("\(k)=\(v); top_k must be a whole number >= 1")
            }
        }
        return Qwen3TTSManifest.Decode(
            doSample: doSample,
            temperature: Float(temperature), topK: Int(topK),
            subtalkerTemperature: Float(subTemp), subtalkerTopK: Int(subTopK))
    }

    /// Quantize a 2-D weight to group-wise affine int4, writing the nibble + scale + bias regions into
    /// `slice` (the weight's whole block; scale/bias at entry.scaleOffset/biasOffset, relative to it).
    /// Source is F32 or BF16, materialized one row at a time as fp32 — bounded memory regardless of size.
    private static func fillU4(
        spec: WeightSpec, entry: Qwen3TTSManifest.Entry,
        srcDtype: String, srcByteCount: Int,
        src: UnsafeRawPointer, source: Qwen3TTSCheckpointTensorPolicy.SourceDType,
        clip: SmeltAffineU4.ClipMode, importance: [Float]?,
        slice: UnsafeMutableRawBufferPointer
    ) throws {
        let rows = spec.shape[0]
        let cols = spec.shape.count > 1 ? spec.shape[1] : 1
        // External input (caller-supplied imatrix): throw, don't trap, on a wrong length; and reject
        // non-finite / negative weights (a negative weight would invert the weighted-MSE objective,
        // letting the clip search "improve" by raising real reconstruction error).
        if let importance {
            guard importance.count == cols else { throw BuilderError.invalidImatrix(spec.name, "length \(importance.count) != cols \(cols)") }
            guard importance.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw BuilderError.invalidImatrix(spec.name, "non-finite or negative importance value") }
        }
        let groupSize = spec.groupSize!
        let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
        let nibbleRowStride = SmeltAffineU4.packedRowStride(cols: cols)
        let srcElemBytes: Int
        switch source {
        case .f32:
            guard srcDtype == "F32" else { throw BuilderError.dtypeMismatch(spec.name, expected: "F32", got: srcDtype) }
            srcElemBytes = 4
        case .bf16:
            guard srcDtype == "BF16" else { throw BuilderError.dtypeMismatch(spec.name, expected: "BF16", got: srcDtype) }
            srcElemBytes = 2
        }
        guard srcByteCount == rows * cols * srcElemBytes else {
            throw BuilderError.sizeMismatch(spec.name, expected: rows * cols * srcElemBytes, got: srcByteCount)
        }
        let (packed, scales, biases) = u4Destinations(slice: slice, entry: entry)
        var row = [Float](repeating: 0, count: cols)
        SmeltAffineU4.withOptionalImportance(importance) { imp in
            row.withUnsafeMutableBufferPointer { rb in
                for r in 0..<rows {
                    let rowBase = r * cols
                    switch source {
                    case .f32:
                        for c in 0..<cols { rb[c] = src.loadUnaligned(fromByteOffset: (rowBase + c) * 4, as: Float.self) }
                    case .bf16:
                        for c in 0..<cols {
                            let bits = src.loadUnaligned(fromByteOffset: (rowBase + c) * 2, as: UInt16.self)
                            rb[c] = Float(bitPattern: UInt32(bits) << 16)
                        }
                    }
                    SmeltAffineU4.quantizeRow(
                        values: rb.baseAddress!, cols: cols, groupSize: groupSize, clip: clip, importance: imp,
                        packed: packed + r * nibbleRowStride,
                        scales: scales + r * groups,
                        biases: biases + r * groups)
                }
            }
        }
    }

    /// The three destination pointers into a u4 weight's slice: packed nibbles at the base, fp16
    /// scales/biases at the manifest entry's relative offsets. The single home for the u4 physical
    /// layout, shared by `fillU4` (affine) and `fillU4FromBlock` (precomputed GPTQ) so they can't drift.
    private static func u4Destinations(slice: UnsafeMutableRawBufferPointer, entry: Qwen3TTSManifest.Entry)
        -> (packed: UnsafeMutablePointer<UInt8>, scales: UnsafeMutablePointer<UInt16>, biases: UnsafeMutablePointer<UInt16>) {
        let base = slice.baseAddress!
        return (base.assumingMemoryBound(to: UInt8.self),
                base.advanced(by: Int(entry.scaleOffset!)).assumingMemoryBound(to: UInt16.self),
                base.advanced(by: Int(entry.biasOffset!)).assumingMemoryBound(to: UInt16.self))
    }

    /// Write a precomputed GPTQ `block` verbatim into the weight's u4 slice (same layout `fillU4`
    /// produces). External input, so the block's dimensions are checked against the weight AND the
    /// block's array lengths against its own dimensions (a Packed can be built with mismatched arrays);
    /// a mismatch throws — a wrong block must fail the build, not silently mis-shape a weight the
    /// gemv_u4 kernel then misreads.
    private static func fillU4FromBlock(
        spec: WeightSpec, entry: Qwen3TTSManifest.Entry,
        block: SmeltAffineU4.Packed, slice: UnsafeMutableRawBufferPointer
    ) throws {
        let rows = spec.shape[0]
        let cols = spec.shape.count > 1 ? spec.shape[1] : 1
        guard block.rows == rows, block.cols == cols, block.groupSize == spec.groupSize! else {
            throw BuilderError.invalidGPTQBlock(spec.name,
                "block \(block.rows)×\(block.cols) g\(block.groupSize) != weight \(rows)×\(cols) g\(spec.groupSize!)")
        }
        guard block.nibbles.count == block.rows * block.rowStride,
              block.scales.count == block.rows * block.groups,
              block.biases.count == block.rows * block.groups else {
            throw BuilderError.invalidGPTQBlock(spec.name,
                "block array lengths (nibbles \(block.nibbles.count), scales \(block.scales.count), biases \(block.biases.count)) inconsistent with \(block.rows)×\(block.cols) g\(block.groupSize)")
        }
        let (packed, scales, biases) = u4Destinations(slice: slice, entry: entry)
        _ = block.nibbles.withUnsafeBufferPointer { memcpy(packed, $0.baseAddress!, $0.count) }
        block.scales.withUnsafeBufferPointer { scales.update(from: $0.baseAddress!, count: $0.count) }
        block.biases.withUnsafeBufferPointer { biases.update(from: $0.baseAddress!, count: $0.count) }
    }

    private static func writeWeights(
        path: String,
        totalBytes: UInt64,
        specs: [WeightSpec],
        entries: [Qwen3TTSManifest.Entry],
        fill: (WeightSpec, Qwen3TTSManifest.Entry, UnsafeMutableRawBufferPointer) throws -> Void
    ) throws {
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else { throw BuilderError.weightsOpenFailed(path) }
        defer { close(fd) }
        guard ftruncate(fd, off_t(totalBytes)) == 0 else {
            throw BuilderError.weightsTruncateFailed(path)
        }
        let size = Int(totalBytes)
        guard let base = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0),
              base != MAP_FAILED else {
            throw BuilderError.weightsMmapFailed(path)
        }
        // Unmap on every exit, including a fill() throw mid-loop (else the mapping —
        // multi-GB for the real pack — leaks for the process lifetime).
        defer { munmap(base, size) }
        // A freshly truncated file reads as zeros, so the page-padding needs no
        // explicit init — fill writes only each weight's exact data bytes.
        for (spec, entry) in zip(specs, entries) {
            // The slice covers the weight's whole reserved block (page-rounded), so a u4 fill can reach
            // its scale/bias regions at entry.scaleOffset/biasOffset.
            let slice = UnsafeMutableRawBufferPointer(
                start: base.advanced(by: Int(entry.offset)), count: Int(entry.byteLength))
            try fill(spec, entry, slice)
        }
        // Check the flush — an EIO/ENOSPC here means weights.bin is corrupt, so we must
        // not go on to emit a manifest that vouches for it.
        guard msync(base, size, MS_SYNC) == 0 else { throw BuilderError.weightsSyncFailed(path) }
    }

    public enum BuilderError: Error, CustomStringConvertible {
        case noCarriedTensors(String)
        case missingTensorBlock(String)
        case missingTensorPattern(block: String, selector: String, reason: String)
        case dtypeMismatch(String, expected: String, got: String)
        case sizeMismatch(String, expected: Int, got: Int)
        case invalidU4(String, String)
        case invalidImatrix(String, String)
        case invalidGPTQBlock(String, String)
        case weightsOpenFailed(String)
        case weightsTruncateFailed(String)
        case weightsMmapFailed(String)
        case weightsSyncFailed(String)
        case tokenizerFileMissing(String)
        case tokenizerPathInvalid(String)
        case unsupportedGenerationConfig(String)
        case unsupportedTrunkConfig(String)
        case unrunnableFullPipeline(String)

        public var description: String {
            switch self {
            case let .noCarriedTensors(p): return "no CAM-carried Qwen3-TTS tensors in checkpoint: \(p)"
            case let .missingTensorBlock(block): return "checkpoint missing tensors for CAM block \(block)"
            case let .missingTensorPattern(block, selector, reason):
                return "checkpoint missing CAM \(reason) tensor pattern \(selector) for block \(block)"
            case let .dtypeMismatch(name, expected, got): return "\(name): expected \(expected), is \(got)"
            case let .sizeMismatch(name, expected, got): return "\(name): expected \(expected) bytes, is \(got)"
            case let .invalidU4(name, why): return "\(name): invalid u4 weight — \(why)"
            case let .invalidImatrix(name, why): return "\(name): invalid imatrix — \(why)"
            case let .invalidGPTQBlock(name, why): return "\(name): invalid GPTQ block — \(why)"
            case let .weightsOpenFailed(p): return "weights.bin open failed: \(p)"
            case let .weightsTruncateFailed(p): return "weights.bin ftruncate failed: \(p)"
            case let .weightsMmapFailed(p): return "weights.bin mmap failed: \(p)"
            case let .weightsSyncFailed(p): return "weights.bin msync failed: \(p)"
            case let .tokenizerFileMissing(p): return "tokenizer file to bundle not found: \(p)"
            case let .tokenizerPathInvalid(p): return "tokenizer file must be a bare basename: \(p)"
            case let .unsupportedGenerationConfig(m): return "unsupported generation_config.json: \(m)"
            case let .unsupportedTrunkConfig(m): return "compiled-trunk config mismatch: \(m)"
            case let .unrunnableFullPipeline(m): return "full-pipeline TTS package can't run generation: \(m)"
            }
        }
    }
}
