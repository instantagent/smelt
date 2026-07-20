// Qwen3TTSTrunkSidecar — B3.2c/B3.2d (docs/talker-trunk-in-package-plan.md,
// docs/talker-mtp-trunk-plan.md): emit a compiled fp32 talker-class trunk INTO the TTS
// package as a sidecar dir that SHARES the package's weights.bin + model.metallib (zero
// weight duplication — footprint is a hard project goal). The trunk's dispatch tables bake
// ABSOLUTE byte offsets into the shared weights blob, so re-emitting them against the TTS
// layout's offsets makes the compiled trunk read the very bytes the hand path reads.
//
// BLOCK-AGNOSTIC (B3.2d §7 generalization): parameterized by a `TrunkSidecarSpec`
// (weight prefix / sidecar dir / model name / vocab-ref tensor), so the SAME primitive
// emits the MAIN talker trunk (`trunk/`, `talker.model.*`) AND the MTP/code_predictor
// transformer (`trunk-mtp/`, `talker.code_predictor.model.*`) — two co-resident compiled
// blocks sharing one weights.bin. `.talker` is the default; `.mtp` is the second block.
//
// The trunk IR is SYNTHESISED from the package's own tensor shapes (the checkpoint is the
// source of truth — a unit test asserts the talker synthesis equals the frozen talker-trunk
// golden) via the `SmeltModelIR.denseTrunk` factory, generalising across sizes
// (0.6B/1.7B) by dim substitution.
//
// bf16 or u4: DenseTrunkEmitter emits bf16 (fused) or u4 (unfused, Phase 3) projections + fp32 norms,
// so a sidecar ships for a bf16 OR u4 build. f32/f16 full-pipeline builds are rejected at build
// (Phase 4 — the hand talker is retired), so no shipped package carries the trunk natively. The
// builder gates the emit on shippable (bf16/u4) projections being present (shouldShipTrunks).

import Foundation
import SmeltSchema

public enum Qwen3TTSTrunkSidecar {

    /// Which co-resident trunk block to emit. The talker-class transformer machinery
    /// (DenseTrunkEmitter, the fp32 ABI, the canonical `layers_N_…` names) is identical
    /// across blocks; only the SHARED-blob lookup prefix, the sidecar dir, the model name,
    /// and the `[vocab, hidden]` reference tensor (codec_embedding for the talker, lm_head
    /// for the MTP — its width follows the block's OWN hidden, not the talker's) differ.
    public struct TrunkSidecarSpec: Sendable {
        public let weightPrefix: String     // "talker.model." / "talker.code_predictor.model."
        public let sidecarDir: String       // "trunk" / "trunk-mtp"
        public let modelName: String        // the synthesised IR's model name
        public let vocabRefTensor: String   // a [vocab, hidden] tensor: rows = vocab, cols = this block's hidden

        public init(weightPrefix: String, sidecarDir: String, modelName: String, vocabRefTensor: String) {
            self.weightPrefix = weightPrefix
            self.sidecarDir = sidecarDir
            self.modelName = modelName
            self.vocabRefTensor = vocabRefTensor
        }

        /// The MAIN talker trunk (Part A) — the default; byte-identical to the pre-B3.2d emit.
        public static let talker = TrunkSidecarSpec(
            weightPrefix: "talker.model.",
            sidecarDir: SmeltPackageSidecarProfiles.qwen3TTSTalkerTrunk.path,
            modelName: SmeltPackageSidecarProfiles.qwen3TTSTalkerTrunk.modelName,
            vocabRefTensor: "talker.model.codec_embedding.weight")

        /// The MTP / code_predictor transformer (B3.2d) — a SECOND co-resident block.
        /// vocab-ref is lm_head.0 (its [mtpVocab, mtpHidden] gives the MTP's OWN hidden;
        /// codec_embedding width would be the TALKER hidden — codex trap #4).
        public static let mtp = TrunkSidecarSpec(
            weightPrefix: "talker.code_predictor.model.",
            sidecarDir: SmeltPackageSidecarProfiles.qwen3TTSMTPTrunk.path,
            modelName: SmeltPackageSidecarProfiles.qwen3TTSMTPTrunk.modelName,
            vocabRefTensor: "talker.code_predictor.lm_head.0.weight")
    }

    public enum SidecarError: Error, CustomStringConvertible {
        case notBF16Trunk
        case missingTensor(String)
        case malformedShape(name: String, got: [Int], why: String)
        case shapeMismatch(name: String, got: [Int], want: [Int])
        case dtypeMismatch(name: String, want: String, got: String)
        case u4MissingMetadata(String)
        case u4MalformedRegions(name: String, why: String)
        case badSymlink(String)

        public var description: String {
            switch self {
            case .notBF16Trunk:
                return "trunk sidecar: this block's layer-0 q_proj is not an emittable trunk dtype "
                    + "(f32/f16/bf16/u4) — canShareWeights rejected it"
            case .missingTensor(let n):
                return "trunk sidecar: TTS manifest has no tensor '\(n)'"
            case let .malformedShape(name, got, why):
                return "trunk sidecar: '\(name)' shape \(got) is malformed — \(why)"
            case let .shapeMismatch(name, got, want):
                return "trunk sidecar: '\(name)' TTS shape \(got) != the IR-expected \(want) "
                    + "(the emitter bakes IR-sized dispatches at this absolute offset; a "
                    + "mismatch would read the wrong bytes — fail loud at build)"
            case let .dtypeMismatch(name, want, got):
                return "trunk sidecar: '\(name)' must be \(want) in the shared weights, is \(got)"
            case .u4MissingMetadata(let n):
                return "trunk sidecar: u4 projection '\(n)' is missing its group size or "
                    + "scale/bias offsets — a u4 weight must carry the affine quant metadata"
            case let .u4MalformedRegions(name, why):
                return "trunk sidecar: u4 projection '\(name)' has malformed affine regions — "
                    + "\(why) (a re-pointed entry must bind the SAME bytes the hand u4 matvec "
                    + "reads; fail loud at build, not as silent wrong-bytes at runtime)"
            case .badSymlink(let why):
                return "trunk sidecar: \(why)"
            }
        }
    }

    /// True when this block's layer-0 q_proj has a registered dense-trunk cell — all of
    /// f32/f16 (unfused dense gemv), bf16 (fused), u4 (unfused affine). The builder gates the emit on
    /// this; the emitter's per-layer uniformity guard + the prepare preflight then prove the WHOLE
    /// spec is emittable. Distinct from `weightLayout`'s OFFSET-MAPPING question. Dtype picks a kernel
    /// lego, never gates emittability — an unknown q_proj dtype string is the only rejection.
    public static func canShareWeights(
        _ manifest: Qwen3TTSManifest, spec: TrunkSidecarSpec = .talker
    ) -> Bool {
        // The q_proj must be PRESENT (this spec's network exists); only then is its dtype the
        // emittability question. An ABSENT entry (wrong spec / partial manifest) can't ship — and
        // since planLayout omits the dtype string for f32, "absent" and "present-f32" both read as a
        // nil `.dtype`, so presence is checked separately from the dtype.
        guard let entry = manifest.weights.first(where: {
            $0.name == "\(spec.weightPrefix)layers.0.self_attn.q_proj.weight"
        }) else { return false }
        let dtype = entry.dtype ?? "f32"   // present; nil string ⇒ f32
        return Qwen3TTSPackageBuilder.emittableTrunkDtypes.contains(dtype)
    }

    // MARK: - A1: IR synthesis (from the package's own talker shapes)

    /// Synthesise the trunk `SmeltModelIR` from `manifest`'s talker.* shapes by feeding
    /// the derived dims to the `SmeltModelIR.denseTrunk` factory (equal to the frozen
    /// talker-trunk golden). Throws if a required talker tensor is absent or mis-shaped.
    static func synthesizeIR(
        from manifest: Qwen3TTSManifest, spec: TrunkSidecarSpec = .talker, maxPrefillBatch: Int = 256
    ) throws -> SmeltModelIR {
        let byName = Dictionary(manifest.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        func shape(_ name: String) throws -> [Int] {
            guard let e = byName[name] else { throw SidecarError.missingTensor(name) }
            return e.shape
        }
        let layers = "\(spec.weightPrefix)layers."
        // headDim from the per-head qk-norm length; the projections give hidden + head counts.
        let qNorm = try shape("\(layers)0.self_attn.q_norm.weight")
        guard qNorm.count == 1, qNorm[0] > 0 else {
            throw SidecarError.malformedShape(name: "q_norm.weight", got: qNorm, why: "expected [headDim>0]")
        }
        let headDim = qNorm[0]
        let qProj = try shape("\(layers)0.self_attn.q_proj.weight")   // [qDim, hidden]
        let kProj = try shape("\(layers)0.self_attn.k_proj.weight")   // [kvDim, hidden]
        let gate = try shape("\(layers)0.mlp.gate_proj.weight")       // [inter, hidden]
        // A [vocab, hidden] reference for THIS block's hidden (cols) + vocab (rows). Must be a
        // tensor whose width is the block's OWN hidden (talker: codec_embedding; MTP: lm_head —
        // codec_embedding width would be the TALKER hidden, the codex trap).
        let vocabRef = try shape(spec.vocabRefTensor)
        guard qProj.count == 2, kProj.count == 2, gate.count == 2, vocabRef.count == 2,
              qProj[0] > 0, kProj[0] > 0, gate[0] > 0,
              qProj[0] % headDim == 0, kProj[0] % headDim == 0 else {
            throw SidecarError.malformedShape(
                name: "q/k/gate/\(spec.vocabRefTensor)", got: qProj,
                why: "expected rank-2 projections with head-divisible rows")
        }
        let hidden = qProj[1]
        // The projections + vocab-ref must agree on the hidden dim, else a derived IR would
        // silently mis-size every downstream dispatch read from these offsets.
        guard hidden > 0, kProj[1] == hidden, gate[1] == hidden, vocabRef[1] == hidden else {
            throw SidecarError.malformedShape(
                name: "hidden", got: [hidden, kProj[1], gate[1], vocabRef[1]],
                why: "q/k/gate/vocab-ref disagree on the hidden column dim")
        }
        let heads = qProj[0] / headDim
        let kvHeads = kProj[0] / headDim
        let inter = gate[0]
        let vocab = vocabRef[0]          // head-side vocab (rows)

        // Count this block's decoder layers by their input_layernorm presence.
        var maxLayer = -1
        for e in manifest.weights where e.name.hasPrefix(layers) {
            let rest = e.name.dropFirst(layers.count)
            if let dot = rest.firstIndex(of: "."), let n = Int(rest[..<dot]) {
                maxLayer = max(maxLayer, n)
            }
        }
        let numLayers = maxLayer + 1
        guard numLayers > 0 else { throw SidecarError.missingTensor("\(layers)*") }

        // Build the trunk IR directly from the derived dims. RoPE θ=1e6, eps=1e-6, split-half
        // RoPE and weight-direct qk-norm are source-architecture constants (not
        // in the tensor shapes) — they live in the `SmeltModelIR.denseTrunk` factory. This was
        // formerly a generated spec string parsed at build time; the direct factory is
        // byte-identical (proven, then frozen as the Qwen3TTSTrunkSidecarTests golden) and let
        // the legacy model-spec parser be retired.
        let ir = SmeltModelIR.denseTrunk(
            modelName: spec.modelName,
            hidden: hidden,
            numLayers: numLayers,
            vocab: vocab,
            heads: heads,
            kvHeads: kvHeads,
            headDim: headDim,
            inter: inter,
            maxPrefillBatch: maxPrefillBatch)
        try validateSmeltIR(ir)
        return ir
    }

    // MARK: - A1: weight layout adapter (TTS offsets → canonical trunk names)

    /// Map the trunk's expected weights onto the SHARED TTS weights.bin: each entry keeps
    /// the canonical name + dtype the emitter requires, but takes the TTS layout's OFFSET
    /// (into the shared blob) and a LOGICAL size (shape × dtype bytes — NOT the TTS entry's
    /// page-rounded byteLength). EVERY weight's dtype AND shape is validated against the IR
    /// dims the emitter bakes into its grid/constant values: a drift on any layer (not just
    /// layer 0, which the IR was derived from) would otherwise read the wrong bytes from a
    /// correct absolute offset — fail loud at build, not as a silent OOB at runtime.
    static func weightLayout(
        from manifest: Qwen3TTSManifest, ir: SmeltModelIR, spec: TrunkSidecarSpec = .talker
    ) throws -> [SmeltWeightEntry] {
        let byName = Dictionary(manifest.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        guard let attn = ir.config.attention else {
            throw SidecarError.malformedShape(name: "ir.attention", got: [], why: "no attention config")
        }
        let h = ir.config.hiddenSize
        let qDim = attn.qHeads * attn.headDim, kvDim = attn.kvHeads * attn.headDim
        let inter = ir.config.ffn.dim, hd = attn.headDim
        // One per-layer table: (canonical suffix, talker.* suffix, IR-expected shape, role).
        // A `.projection` is a matmul weight (bf16 OR u4 in the shared blob); a `.norm` is a
        // norm weight (always fp32). The emitter looks each up by canonical name.
        let perLayer: [(String, String, [Int], WeightRole)] = [
            ("_self_attn_q_proj_weight", ".self_attn.q_proj.weight", [qDim, h], .projection),
            ("_self_attn_k_proj_weight", ".self_attn.k_proj.weight", [kvDim, h], .projection),
            ("_self_attn_v_proj_weight", ".self_attn.v_proj.weight", [kvDim, h], .projection),
            ("_self_attn_o_proj_weight", ".self_attn.o_proj.weight", [h, qDim], .projection),
            ("_mlp_gate_proj_weight", ".mlp.gate_proj.weight", [inter, h], .projection),
            ("_mlp_up_proj_weight", ".mlp.up_proj.weight", [inter, h], .projection),
            ("_mlp_down_proj_weight", ".mlp.down_proj.weight", [h, inter], .projection),
            ("_input_layernorm_weight", ".input_layernorm.weight", [h], .norm),
            ("_post_attention_layernorm_weight", ".post_attention_layernorm.weight", [h], .norm),
            ("_self_attn_q_norm_weight", ".self_attn.q_norm.weight", [hd], .norm),
            ("_self_attn_k_norm_weight", ".self_attn.k_norm.weight", [hd], .norm),
        ]
        var specs: [(canonical: String, ttsName: String, shape: [Int], role: WeightRole)] = []
        for layer in 0..<ir.config.numLayers {
            for (cS, tS, shape, role) in perLayer {
                specs.append(("layers_\(layer)\(cS)", "\(spec.weightPrefix)layers.\(layer)\(tS)", shape, role))
            }
        }
        specs.append(("norm_weight", "\(spec.weightPrefix)norm.weight", [h], .norm))

        return try specs.map { s in
            guard let e = byName[s.ttsName] else { throw SidecarError.missingTensor(s.ttsName) }
            guard e.shape == s.shape else {
                throw SidecarError.shapeMismatch(name: s.ttsName, got: e.shape, want: s.shape)
            }
            return try mapEntry(canonical: s.canonical, ttsEntry: e, shape: s.shape, role: s.role)
        }
    }

    /// A weight's structural role in the trunk: a matmul projection (dense bf16 or affine u4
    /// in the shared blob) or a norm (always fp32). Drives the dtype-region mapping.
    private enum WeightRole { case projection, norm }

    /// Re-point one TTS entry onto a canonical trunk `SmeltWeightEntry` against the SHARED
    /// blob — same bytes, no conversion, only offsets are carried. Norms must be fp32; a
    /// projection is bf16 (one dense region) or u4 (three regions: nibbles + per-group
    /// fp16 scales + biases). For u4 the TTS native convention stores scale/bias offsets
    /// RELATIVE to the block start, so they are translated to the ABSOLUTE-into-weights.bin
    /// convention the schema entry (and `WeightLocator`) use.
    private static func mapEntry(
        canonical: String, ttsEntry e: Qwen3TTSManifest.Entry, shape: [Int], role: WeightRole
    ) throws -> SmeltWeightEntry {
        let ttsDtype = e.dtype ?? "f32"
        switch role {
        case .norm:
            guard ttsDtype == "f32" else {
                throw SidecarError.dtypeMismatch(name: e.name, want: "f32", got: ttsDtype)
            }
            return SmeltWeightEntry(
                name: canonical, offset: e.offset,
                sizeBytes: UInt64(shape.reduce(1, *) * 4), shape: e.shape, dtype: .fp32)
        case .projection:
            switch ttsDtype {
            case "f32":
                return SmeltWeightEntry(
                    name: canonical, offset: e.offset,
                    sizeBytes: UInt64(shape.reduce(1, *) * 4), shape: e.shape, dtype: .fp32)
            case "f16":
                return SmeltWeightEntry(
                    name: canonical, offset: e.offset,
                    sizeBytes: UInt64(shape.reduce(1, *) * 2), shape: e.shape, dtype: .fp16)
            case "bf16":
                return SmeltWeightEntry(
                    name: canonical, offset: e.offset,
                    sizeBytes: UInt64(shape.reduce(1, *) * 2), shape: e.shape, dtype: .bf16)
            case "u4":
                guard let groupSize = e.groupSize,
                      let scaleOff = e.scaleOffset, let scaleLen = e.scaleByteLength,
                      let biasOff = e.biasOffset, let biasLen = e.biasByteLength else {
                    throw SidecarError.u4MissingMetadata(e.name)
                }
                let cols = shape.count > 1 ? shape[1] : 1
                let nibbleBytes = UInt64(shape[0] * SmeltAffineU4.packedRowStride(cols: cols))
                // Validate the affine metadata before re-pointing: a corrupt/mismatched
                // manifest must fail loud here, not produce a plausible .affineU4 entry that
                // binds wrong-but-in-range bytes (the dispatch-safety / checkpoint-coverage
                // invariant). Group contract mirrors Qwen3TTSPackageBuilder.validateU4 IN FULL
                // (incl. the groups ≤ u4MaxGroups kernel bound below); the scale/bias regions
                // must be the exact fp16 [rows, ceil(cols/group)] sizes and sit, ordered, after
                // the nibbles within the block (byteLength).
                guard groupSize >= 4, groupSize % 4 == 0, cols % 4 == 0 else {
                    throw SidecarError.u4MalformedRegions(
                        name: e.name, why: "group_size \(groupSize) / cols \(cols) violate the "
                            + "u4 contract (group_size a multiple of 4 ≥ 4, cols a multiple of 4)")
                }
                let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
                // The kernel's U4_MAX_GROUPS bound (validateU4 enforces it at build; re-checked
                // here so a re-pointed entry can't smuggle a groups>max weight past the sidecar
                // into a Phase-3 threadgroup/scale-index OOB).
                guard groups <= Qwen3TTSPackageBuilder.u4MaxGroups else {
                    throw SidecarError.u4MalformedRegions(
                        name: e.name, why: "groups \(groups) exceeds U4_MAX_GROUPS "
                            + "\(Qwen3TTSPackageBuilder.u4MaxGroups)")
                }
                let expectRegion = UInt64(shape[0] * groups * 2)   // fp16 [rows, groups]
                guard scaleLen == expectRegion, biasLen == expectRegion else {
                    throw SidecarError.u4MalformedRegions(
                        name: e.name, why: "scale/bias region bytes (\(scaleLen)/\(biasLen)) != the "
                            + "expected \(expectRegion) for \(shape[0])×\(groups) fp16")
                }
                // Ordered, in-block: nibbles ≤ scaleOff ≤ scaleOff+scaleLen ≤ biasOff ≤
                // biasOff+biasLen ≤ byteLength. Subtractive form (every subtraction guarded by
                // the preceding ≥) so a hostile manifest with huge UInt64 offsets throws
                // .u4MalformedRegions instead of trapping on an addition overflow.
                guard scaleOff >= nibbleBytes,
                      biasOff >= scaleOff, biasOff - scaleOff >= scaleLen,
                      e.byteLength >= biasOff, e.byteLength - biasOff >= biasLen else {
                    throw SidecarError.u4MalformedRegions(
                        name: e.name, why: "relative regions (nibbles \(nibbleBytes), scale@\(scaleOff), "
                            + "bias@\(biasOff)) overlap or exceed the block (\(e.byteLength) bytes)")
                }
                // The whole block must fit in the address space so the absolute translations
                // below (e.offset + scaleOff / + biasOff, both < e.offset + byteLength) can't
                // overflow UInt64 and trap on a hostile huge offset — throw instead.
                guard e.offset <= UInt64.max - e.byteLength else {
                    throw SidecarError.u4MalformedRegions(
                        name: e.name, why: "block offset \(e.offset) + length \(e.byteLength) "
                            + "overflows the address space")
                }
                return SmeltWeightEntry(
                    name: canonical, offset: e.offset, sizeBytes: nibbleBytes,
                    shape: e.shape, dtype: .affineU4, groupSize: groupSize,
                    // RELATIVE-to-block (TTS native) → ABSOLUTE-into-weights.bin (schema).
                    scalesOffset: e.offset + scaleOff, scalesSizeBytes: scaleLen,
                    biasesOffset: e.offset + biasOff, biasesSizeBytes: biasLen)
            default:
                // f32/f16/bf16/u4 are all emittable (dtype picks a kernel lego); an unknown dtype
                // string is a malformed manifest — fail loud, never silently mis-bind.
                throw SidecarError.dtypeMismatch(name: e.name, want: "f32/f16/bf16/u4", got: ttsDtype)
            }
        }
    }

    // MARK: - A2: emit the sidecar into <ttsPkg>/trunk

    /// The fully-resolved, in-memory trunk artifacts — everything `emit` produces BEFORE it
    /// touches the filesystem. Building this proves the spec is FULLY emittable (every tensor
    /// present + right shape/dtype/regions at every layer, both dispatch tables generate), so
    /// the builder can preflight ALL co-resident specs before committing any sidecar.
    public struct PreparedTrunk {
        let ir: SmeltModelIR
        let plan: SmeltBufferPlan
        let layout: [SmeltWeightEntry]
        let compilationPlan: SmeltCompilationPlan
        let decode: TopLevelEmitter.GenerateResult
        let prefill: PrefillEmitter.GenerateResult
        let spec: TrunkSidecarSpec
        /// The FULL shared blob size — the sidecar manifest's totalBytes (the symlink IS the
        /// whole parent weights.bin; the trunk's offsets index anywhere in it).
        let totalBytes: UInt64
    }

    /// Resolve a spec to its in-memory artifacts WITHOUT writing anything — the pure prefix of
    /// `emit`. Throws on any emitability failure (non-bf16 layer-0 q_proj, a missing/mis-shaped
    /// tensor at ANY layer, a non-bf16/u4 projection anywhere, malformed u4 regions, a u4
    /// projection that defers to Phase 3, a missing vocab-ref like lm_head.0). The builder calls
    /// this for EVERY co-resident spec before emitting any, so a deep failure in one trunk can
    /// never half-build a package by committing another first.
    public static func prepare(
        manifest: Qwen3TTSManifest, spec: TrunkSidecarSpec = .talker, maxPrefillBatch: Int = 256
    ) throws -> PreparedTrunk {
        guard canShareWeights(manifest, spec: spec) else { throw SidecarError.notBF16Trunk }
        let ir = try synthesizeIR(from: manifest, spec: spec, maxPrefillBatch: maxPrefillBatch)
        let layout = try weightLayout(from: manifest, ir: ir, spec: spec)
        let compilationPlan = try SmeltCompiler.planCompilation(
            ir: ir,
            weightLayout: layout
        )
        let plan = compilationPlan.bufferPlan
        let decode = try TopLevelEmitter.generate(
            ir: ir,
            compilationPlan: compilationPlan
        )
        let prefill = try PrefillEmitter.generate(
            ir: ir,
            compilationPlan: compilationPlan
        )
        return PreparedTrunk(ir: ir, plan: plan, layout: layout, compilationPlan: compilationPlan,
                             decode: decode, prefill: prefill, spec: spec,
                             totalBytes: manifest.totalBytes)
    }

    /// Emit the compiled trunk into `<pkgPath>/trunk` sharing the parent's weights.bin +
    /// model.metallib via relative symlinks. Precondition: `manifest` is a full-pipeline
    /// bf16 build (`canShareWeights`); the builder gates on it. Standalone callers get the
    /// prepare→commit pipeline in one call; the builder preflights `prepare` for all specs
    /// first, then `commit`s each, for all-or-none safety.
    public static func emit(
        intoPackage pkgPath: String, manifest: Qwen3TTSManifest,
        spec: TrunkSidecarSpec = .talker, maxPrefillBatch: Int = 256
    ) throws {
        try commit(try prepare(manifest: manifest, spec: spec, maxPrefillBatch: maxPrefillBatch),
                   intoPackage: pkgPath)
    }

    /// Write a prepared trunk into `<pkgPath>/<sidecarDir>` (temp + atomic rename). Splitting
    /// this from `prepare` lets the builder validate every co-resident spec before the first
    /// filesystem mutation.
    public static func commit(_ prepared: PreparedTrunk, intoPackage pkgPath: String) throws {
        let spec = prepared.spec
        let ir = prepared.ir, plan = prepared.plan, layout = prepared.layout
        let decode = prepared.decode, prefill = prepared.prefill

        let fm = FileManager.default
        let trunkDir = "\(pkgPath)/\(spec.sidecarDir)"
        // Build into a temp sibling and rename into place: a failure mid-emit must NOT
        // leave a half-written sidecar that the parent manifest already claims is .compiled.
        // The temp dir sits directly under pkgPath, so its relative ../weights.bin links
        // resolve identically before and after the rename.
        let tmpDir = "\(pkgPath)/.\(spec.sidecarDir).tmp"
        if fm.fileExists(atPath: tmpDir) { try fm.removeItem(atPath: tmpDir) }
        try fm.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
        var committed = false
        defer { if !committed { try? fm.removeItem(atPath: tmpDir) } }

        try SmeltCompiler.writeDispatchTable(decode.dispatchRecords, to: "\(tmpDir)/dispatches.bin")
        try SmeltCompiler.writeDispatchTable(prefill.dispatchRecords, to: "\(tmpDir)/prefill_dispatches.bin")
        // GPTQ capture points (Phase 4 U2): where calibration reads each projection's [seqLen, k]
        // FP32 activation input through THIS sidecar's own SmeltRuntime.captureGPTQActivations, so
        // u4 calibration rides the compiled trunk.
        if !prefill.gptqCapturePoints.isEmpty {
            let capData = try SmeltGPTQCapturePoints(
                prefill: prefill.gptqCapturePoints
            ).canonicalJSONData()
            try capData.write(to: URL(fileURLWithPath: "\(tmpDir)/gptq_capture_points.json"))
        }
        // DenseTrunkEmitter's generated source is a documented stub (record-interpreted
        // only); SmeltPackageIntegrity verifies SmeltGenerated.swift by that exact name
        // when the manifest carries its checksum, so write it under the canonical name.
        try decode.source.write(toFile: "\(tmpDir)/SmeltGenerated.swift", atomically: true, encoding: .utf8)

        // Share the parent's weights.bin + metallib — RELATIVE, package-internal symlinks.
        try linkShared(name: "weights.bin", trunkDir: tmpDir, pkgPath: pkgPath)
        try linkShared(name: "model.metallib", trunkDir: tmpDir, pkgPath: pkgPath)

        let handoff = SmeltHandoffResolver.resolve(
            families: ["key_cache", "value_cache"], ir: ir, plan: plan)
        let checksums = try SmeltCompiler.computeManifestChecksums(
            pkgPath: tmpDir,
            weightsPath: "\(tmpDir)/weights.bin",
            metallibPath: "\(tmpDir)/model.metallib",
            generatedSwiftPath: "\(tmpDir)/SmeltGenerated.swift",
            dispatchesPath: "\(tmpDir)/dispatches.bin",
            prefillDispatchesPath: "\(tmpDir)/prefill_dispatches.bin",
            prefillVerifyArgmaxDispatchesPath: nil)

        let sidecar = SmeltManifest(
            kind: nil,
            headlessTrunkABI: true,  // explicit headless-trunk marker (U3 activation axis)
            blocks: nil, loop: nil,
            modelName: ir.modelName,
            config: SmeltCompiler.manifestConfigSnapshot(from: ir),
            context: nil,
            checksums: checksums,
            buildProvenance: nil,
            device: SmeltDeviceRequirements(
                metalFamily: .apple7, minMemoryBytes: UInt64(plan.totalActivationBytes)),
            // totalBytes = the FULL shared blob (the symlink IS the whole TTS weights.bin;
            // the runtime mmaps `totalBytes` and the trunk's offsets index anywhere in it).
            weights: SmeltWeightManifest(totalBytes: prepared.totalBytes, entries: layout),
            buffers: plan.toBufferTable(),
            pipelines: SmeltKernelCatalog.pipelineNames,
            slotLayout: plan.toSlotLayout(),
            prefill: SmeltPrefillManifest(
                engine: "metal", modelPath: "prefill.mlmodelc",
                maxBatchSize: ir.prefill?.maxBatchSize ?? 256,
                handoff: handoff, inputContract: SmeltPrefillInputContract()),
            inference: nil,
            optimizationReport: nil)
        try sidecar.encodePrettyJSON().write(to: URL(fileURLWithPath: "\(tmpDir)/manifest.json"))

        // Atomic-ish swap into place (rename within one directory).
        if fm.fileExists(atPath: trunkDir) { try fm.removeItem(atPath: trunkDir) }
        try fm.moveItem(atPath: tmpDir, toPath: trunkDir)
        committed = true
    }

    /// Create `trunkDir/name` → `../name`, asserting it resolves to the parent package's
    /// own file (relative + internal — footgun #9: an external/dangling link would mmap
    /// or makeLibrary off a path outside the package).
    private static func linkShared(name: String, trunkDir: String, pkgPath: String) throws {
        let fm = FileManager.default
        let target = "\(pkgPath)/\(name)"
        guard fm.fileExists(atPath: target) else {
            throw SidecarError.badSymlink("shared \(name) missing at \(target)")
        }
        let link = "\(trunkDir)/\(name)"
        if fm.fileExists(atPath: link) { try fm.removeItem(atPath: link) }
        try fm.createSymbolicLink(atPath: link, withDestinationPath: "../\(name)")
        let resolved = URL(fileURLWithPath: link).resolvingSymlinksInPath().path
        guard resolved == URL(fileURLWithPath: target).resolvingSymlinksInPath().path else {
            throw SidecarError.badSymlink("\(name) link resolves to \(resolved), not \(target)")
        }
    }
}
