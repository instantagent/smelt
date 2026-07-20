// Qwen3TTSManifest — minimal package manifest for the hand-written Qwen3-TTS
// `.smeltpkg`. The multi-network TTS topology does not fit the LLM-shaped
// SmeltManifest (buffers / slotLayout / dispatch table), and the GPU driver
// loads weights itself (per-weight MTLBuffer(bytesNoCopy:) at page-aligned
// offsets) rather than through SmeltRuntime.init. So this carries only what the
// driver needs: the kernels to build, the codec EOS, and a page-aligned fp32
// weight table. Every weight is fp32.

import Foundation

public struct Qwen3TTSManifest: Codable, Sendable {
    /// One fp32 weight slice in weights.bin.
    public struct Entry: Codable, Sendable {
        public let name: String         // checkpoint name == canonical lookup key
        public let offset: UInt64       // page-aligned byte offset into weights.bin
        public let byteLength: UInt64   // page-rounded length for MTLBuffer(bytesNoCopy:)
        public let shape: [Int]         // true tensor shape; element count = product
        /// Packed element type: "f32" (default/absent — back-compat), "f16", "bf16", or "u4".
        /// Determines the element byte size and which matmul kernel binds it.
        public let dtype: String?

        // --- group-wise affine int4 ("u4") metadata; nil for every other dtype ---
        // A u4 weight is ONE page-aligned block (byteLength spans the whole block): packed nibbles at
        // `offset`, then per-group fp16 scale/bias at `scaleOffset`/`biasOffset` RELATIVE to `offset`
        // (both page-aligned within the block, so the runtime binds one buffer at three offsets).
        // groupSize is the column group; scale/bias are each [rows, ceil(cols/groupSize)] half.
        public let groupSize: Int?
        public let scaleOffset: UInt64?
        public let scaleByteLength: UInt64?
        public let biasOffset: UInt64?
        public let biasByteLength: UInt64?

        public init(name: String, offset: UInt64, byteLength: UInt64, shape: [Int], dtype: String? = nil,
                    groupSize: Int? = nil,
                    scaleOffset: UInt64? = nil, scaleByteLength: UInt64? = nil,
                    biasOffset: UInt64? = nil, biasByteLength: UInt64? = nil) {
            self.name = name
            self.offset = offset
            self.byteLength = byteLength
            self.shape = shape
            self.dtype = dtype
            self.groupSize = groupSize
            self.scaleOffset = scaleOffset
            self.scaleByteLength = scaleByteLength
            self.biasOffset = biasOffset
            self.biasByteLength = biasByteLength
        }
    }

    /// The checkpoint's generation defaults, read from generation_config.json at build time, so the
    /// package self-describes its decode policy. `doSample == false` ⇒ greedy. top_p is required to be 1.0 (identity) at build,
    /// so it isn't carried. repetition_penalty is NOT here — it's a cb0 logits processor applied
    /// identically in both greedy and sampling modes, not a mode-selection input.
    /// Required when a package claims the Qwen3-TTS TTFA CAM gate.
    public struct Decode: Codable, Sendable {
        public let doSample: Bool
        public let temperature: Float           // talker (cb0)
        public let topK: Int
        public let subtalkerTemperature: Float  // MTP (cb1..15)
        public let subtalkerTopK: Int

        public init(doSample: Bool, temperature: Float, topK: Int,
                    subtalkerTemperature: Float, subtalkerTopK: Int) {
            self.doSample = doSample
            self.temperature = temperature
            self.topK = topK
            self.subtalkerTemperature = subtalkerTemperature
            self.subtalkerTopK = subtalkerTopK
        }
    }

    public let version: Int
    /// Declared block composition (docs/block-spec-plan.md). Decoded as optional
    /// so incomplete packages can report a validation error.
    public let blocks: SmeltBlockGraph?
    /// Declared drive loop. Decoded as optional so incomplete packages can report
    /// a validation error.
    public let loop: SmeltLoopSchedule?
    public let modelName: String
    public let pageSize: Int             // alignment for every offset and byteLength
    public let pipelines: [String]       // metalFunctionNames carried in model.metallib
    public let eosTokens: [Int32]
    public let totalBytes: UInt64        // page-rounded size of weights.bin
    public let weights: [Entry]
    /// HF tokenizer files copied into the package dir so the driver is a self-contained
    /// text→24 kHz pipeline (vocab.json, merges.txt, tokenizer_config.json, config.json).
    /// Optional: nil/absent for a weights-only package (a missing key decodes to nil).
    public let tokenizerFiles: [String]?
    public let decode: Decode?
    /// Package-owned correctness/performance policy for runnable Qwen3-TTS packages.
    public let validation: SmeltPackageSpec.Validation?

    public init(
        version: Int,
        blocks: SmeltBlockGraph? = nil,
        loop: SmeltLoopSchedule? = nil,
        modelName: String,
        pageSize: Int,
        pipelines: [String],
        eosTokens: [Int32],
        totalBytes: UInt64,
        weights: [Entry],
        tokenizerFiles: [String]? = nil,
        decode: Decode? = nil,
        validation: SmeltPackageSpec.Validation? = nil
    ) {
        self.version = version
        self.blocks = blocks
        self.loop = loop
        self.modelName = modelName
        self.pageSize = pageSize
        self.pipelines = pipelines
        self.eosTokens = eosTokens
        self.totalBytes = totalBytes
        self.weights = weights
        self.tokenizerFiles = tokenizerFiles
        self.decode = decode
        self.validation = validation
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case blocks
        case loop
        case modelName
        case pageSize
        case pipelines
        case eosTokens
        case totalBytes
        case weights
        case tokenizerFiles
        case decode
        case validation
    }

    private enum RemovedCodingKeys: String, CodingKey {
        case architecture
        case kind
    }

    public init(from decoder: Decoder) throws {
        let removed = try decoder.container(keyedBy: RemovedCodingKeys.self)
        if removed.contains(.kind) {
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: removed,
                debugDescription: "qwen3-tts manifest kind is no longer supported"
            )
        }
        if removed.contains(.architecture) {
            throw DecodingError.dataCorruptedError(
                forKey: .architecture,
                in: removed,
                debugDescription: "qwen3-tts manifest architecture is no longer supported"
            )
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try container.decode(Int.self, forKey: .version)
        self.blocks = try container.decodeIfPresent(SmeltBlockGraph.self, forKey: .blocks)
        self.loop = try container.decodeIfPresent(SmeltLoopSchedule.self, forKey: .loop)
        self.modelName = try container.decode(String.self, forKey: .modelName)
        self.pageSize = try container.decode(Int.self, forKey: .pageSize)
        self.pipelines = try container.decode([String].self, forKey: .pipelines)
        self.eosTokens = try container.decode([Int32].self, forKey: .eosTokens)
        self.totalBytes = try container.decode(UInt64.self, forKey: .totalBytes)
        self.weights = try container.decode([Entry].self, forKey: .weights)
        self.tokenizerFiles = try container.decodeIfPresent([String].self, forKey: .tokenizerFiles)
        self.decode = try container.decodeIfPresent(Decode.self, forKey: .decode)
        self.validation = try container.decodeIfPresent(
            SmeltPackageSpec.Validation.self,
            forKey: .validation
        )
    }

    /// The HF tokenizer/config files a text→24 kHz package bundles — the single source of truth
    /// shared by the builder (what it copies) and the driver (what it requires), so the two
    /// can't drift.
    public static let requiredTokenizerFiles =
        ["vocab.json", "merges.txt", "tokenizer_config.json", "config.json"]

    public func encoded() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    public static func decode(from data: Data) throws -> Qwen3TTSManifest {
        try JSONDecoder().decode(Qwen3TTSManifest.self, from: data)
    }

    public func validateQwen3TTSValidation(packagePath: String? = nil) throws {
        try validateQwen3TTSGraphAndLoop()
        guard let validation else { return }
        if let gate = validation.performanceGate,
           gate != SmeltQwen3TTSPackageProfiles.runnable.performanceGate {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_gate must be "
                    + "'\(SmeltQwen3TTSPackageProfiles.runnable.performanceGate)', got '\(gate)'"
            )
        }
        guard let profile = validation.performanceProfile else {
            if validation.performanceGate != nil {
                throw SmeltPackageSpecError.malformed(
                    "qwen3-tts validation.performance_gate requires performance_profile"
                )
            }
            if let structureProfile = validation.structureProfile {
                try validateQwen3TTSStructureProfile(structureProfile, packagePath: packagePath)
            }
            return
        }
        guard profile.gate == SmeltQwen3TTSPackageProfiles.runnable.performanceGate else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_profile.gate must be "
                    + "'\(SmeltQwen3TTSPackageProfiles.runnable.performanceGate)', got '\(profile.gate)'"
            )
        }
        guard profile.command == .bench else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_profile.command must be 'bench', got "
                    + "'\(profile.command.rawValue)'"
            )
        }
        for bound in profile.minBounds where !bound.min.isFinite || bound.min <= 0 {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_profile min-bound '\(bound.metric)' must be positive and finite"
            )
        }
        for bound in profile.maxBounds where !bound.max.isFinite || bound.max <= 0 {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_profile max-bound '\(bound.metric)' must be positive and finite"
            )
        }
        try validateCanonicalQwen3TTSPerformanceProfile(profile)
        if claimsQwen3TTFAGate {
            try validatePackageOwnedDecodePolicy()
        }
        if let structureProfile = validation.structureProfile {
            try validateQwen3TTSStructureProfile(structureProfile, packagePath: packagePath)
        }
    }

    public func validateQwen3TTSGraphAndLoop() throws {
        guard let blocks, let loop else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts manifest blocks and loop must be declared"
            )
        }
        do {
            try blocks.validate()
        } catch let err as SmeltBlockGraph.GraphError {
            throw SmeltPackageSpecError.malformed(err.description)
        }
        let packageProfile = SmeltQwen3TTSPackageProfiles.runnable
        let supportedGraphs = packageProfile.supportedBlockGraphs + [.qwen3TTSCodecDecoder]
        guard supportedGraphs.contains(blocks) else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts manifest blocks must match a supported Qwen3-TTS graph"
            )
        }
        do {
            try loop.validate(against: blocks)
        } catch let err as SmeltLoopSchedule.ScheduleError {
            throw SmeltPackageSpecError.malformed(err.description)
        }
        let expectedLoop: SmeltLoopSchedule = blocks == .qwen3TTSCodecDecoder
            ? .qwen3TTSCodecDecoder
            : packageProfile.loop
        guard loop == expectedLoop else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts manifest loop must match the declared Qwen3-TTS graph"
            )
        }
    }

    private var claimsQwen3TTFAGate: Bool {
        validation?.performanceGate == SmeltQwen3TTSPackageProfiles.runnable.performanceGate
            || validation?.performanceProfile?.gate == SmeltQwen3TTSPackageProfiles.runnable.performanceGate
    }

    private func validateCanonicalQwen3TTSPerformanceProfile(
        _ profile: SmeltPackageSpec.Validation.PerformanceProfile
    ) throws {
        let canonical = SmeltPackagePerformanceProfiles.profile(
            for: SmeltQwen3TTSPackageProfiles.runnable.performanceGate
        )
        let metricSet = Set(profile.requiredOutputMetrics)
        for metric in canonical.requiredOutputMetrics where !metricSet.contains(metric) {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.performance_profile missing canonical required output metric: \(metric)"
            )
        }
        for bound in canonical.maxBounds {
            let hasBound = profile.maxBounds.contains {
                $0.metric == bound.metric && $0.unit == bound.unit && $0.max <= bound.max
            }
            guard hasBound else {
                throw SmeltPackageSpecError.malformed(
                    "qwen3-tts validation.performance_profile missing canonical max-bound: \(bound.metric)"
                )
            }
        }
    }

    private func validatePackageOwnedDecodePolicy() throws {
        guard let decode else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts CAM validation requires package-owned decode policy"
            )
        }
        guard decode.temperature.isFinite, decode.temperature > 0 else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts CAM validation requires decode.temperature > 0"
            )
        }
        guard decode.topK > 0 else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts CAM validation requires decode.topK > 0"
            )
        }
        guard decode.subtalkerTemperature.isFinite, decode.subtalkerTemperature > 0 else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts CAM validation requires decode.subtalkerTemperature > 0"
            )
        }
        guard decode.subtalkerTopK > 0 else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts CAM validation requires decode.subtalkerTopK > 0"
            )
        }
    }

    private func validateQwen3TTSStructureProfile(
        _ profile: SmeltPackageSpec.Validation.StructureProfile,
        packagePath: String?
    ) throws {
        guard profile.id == SmeltPackageStructureProfileID.qwen3TTSRunnable else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile.id must be "
                    + "'\(SmeltPackageStructureProfileID.qwen3TTSRunnable)', got "
                    + "'\(profile.id)'"
            )
        }

        let pipelineSet = Set(pipelines)
        for pipeline in profile.requiredPipelines where !pipelineSet.contains(pipeline) {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile required pipeline missing: \(pipeline)"
            )
        }
        for pipeline in profile.forbiddenPipelines where pipelineSet.contains(pipeline) {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile forbidden pipeline present: \(pipeline)"
            )
        }

        for path in profile.requiredFiles {
            try validateQwen3TTSPackageRelativeProfilePath(path, label: "required file")
            if let packagePath,
               !FileManager.default.fileExists(atPath: "\(packagePath)/\(path)") {
                throw SmeltPackageSpecError.malformed(
                    "qwen3-tts validation.structure_profile required file missing: \(path)"
                )
            }
        }
        for path in profile.forbiddenFiles {
            try validateQwen3TTSPackageRelativeProfilePath(path, label: "forbidden file")
            if let packagePath,
               FileManager.default.fileExists(atPath: "\(packagePath)/\(path)") {
                throw SmeltPackageSpecError.malformed(
                    "qwen3-tts validation.structure_profile forbidden file present: \(path)"
                )
            }
        }

        guard profile.requiredRoutes.isEmpty || blocks != nil else {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile required routes need a block graph"
            )
        }
        let routeSet: Set<String>
        if let blocks {
            routeSet = Set(
                blocks.runtimeRouteSignatures
            )
        } else {
            routeSet = []
        }
        for route in profile.requiredRoutes where !routeSet.contains(route) {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile required route missing: \(route)"
            )
        }
    }

    private func validateQwen3TTSPackageRelativeProfilePath(
        _ path: String,
        label: String
    ) throws {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        if path.isEmpty
            || path.hasPrefix("/")
            || path.hasPrefix("~")
            || path.contains("\\")
            || path.contains("\0")
            || parts.contains(where: { $0 == "" || $0 == "." || $0 == ".." }) {
            throw SmeltPackageSpecError.malformed(
                "qwen3-tts validation.structure_profile \(label) must be package-relative: \(path)"
            )
        }
    }
}

public struct SmeltQwen3TTSPackageProfile: Sendable, Equatable {
    public let packageName: String
    public let modelName: String
    public let runtimeArchitecture: String
    public let baseGraph: SmeltBlockGraph
    public let compiledTalkerGraph: SmeltBlockGraph
    public let compiledTrunkNativeFrontEndGraph: SmeltBlockGraph
    public let loop: SmeltLoopSchedule
    public let tokenizerFiles: [String]
    public let sidecars: [SmeltPackageSpec.Sidecar]
    public let sidecarPaths: [String]
    public let pageSize: Int
    public let maxTokens: Int
    public let eosTokens: [Int32]
    public let performanceGate: String
    public let structureProfileID: String

    public init(
        packageName: String,
        modelName: String,
        runtimeArchitecture: String,
        baseGraph: SmeltBlockGraph,
        compiledTalkerGraph: SmeltBlockGraph,
        compiledTrunkNativeFrontEndGraph: SmeltBlockGraph,
        loop: SmeltLoopSchedule,
        tokenizerFiles: [String],
        sidecars: [SmeltPackageSpec.Sidecar],
        sidecarPaths: [String],
        pageSize: Int,
        maxTokens: Int,
        eosTokens: [Int32],
        performanceGate: String,
        structureProfileID: String
    ) {
        self.packageName = packageName
        self.modelName = modelName
        self.runtimeArchitecture = runtimeArchitecture
        self.baseGraph = baseGraph
        self.compiledTalkerGraph = compiledTalkerGraph
        self.compiledTrunkNativeFrontEndGraph = compiledTrunkNativeFrontEndGraph
        self.loop = loop
        self.tokenizerFiles = tokenizerFiles
        self.sidecars = sidecars
        self.sidecarPaths = sidecarPaths
        self.pageSize = pageSize
        self.maxTokens = maxTokens
        self.eosTokens = eosTokens
        self.performanceGate = performanceGate
        self.structureProfileID = structureProfileID
    }

    public var supportedBlockGraphs: [SmeltBlockGraph] {
        [baseGraph, compiledTalkerGraph, compiledTrunkNativeFrontEndGraph]
    }

    public func graph(textEmbeddingIsBF16: Bool) -> SmeltBlockGraph {
        textEmbeddingIsBF16 ? compiledTalkerGraph : compiledTrunkNativeFrontEndGraph
    }

    public func structureProfile(
        pipelines: [String],
        graph: SmeltBlockGraph
    ) -> SmeltPackageSpec.Validation.StructureProfile {
        SmeltPackageStructureProfiles.qwen3TTSRunnable(
            pipelines: pipelines,
            tokenizerFiles: tokenizerFiles,
            graph: graph
        )
    }

    public static func == (
        lhs: SmeltQwen3TTSPackageProfile,
        rhs: SmeltQwen3TTSPackageProfile
    ) -> Bool {
        lhs.packageName == rhs.packageName
            && lhs.modelName == rhs.modelName
            && lhs.runtimeArchitecture == rhs.runtimeArchitecture
            && lhs.baseGraph == rhs.baseGraph
            && lhs.compiledTalkerGraph == rhs.compiledTalkerGraph
            && lhs.compiledTrunkNativeFrontEndGraph == rhs.compiledTrunkNativeFrontEndGraph
            && lhs.loop == rhs.loop
            && lhs.tokenizerFiles == rhs.tokenizerFiles
            && lhs.sidecarSignatures == rhs.sidecarSignatures
            && lhs.sidecarPaths == rhs.sidecarPaths
            && lhs.pageSize == rhs.pageSize
            && lhs.maxTokens == rhs.maxTokens
            && lhs.eosTokens == rhs.eosTokens
            && lhs.performanceGate == rhs.performanceGate
            && lhs.structureProfileID == rhs.structureProfileID
    }

    private var sidecarSignatures: [String] {
        sidecars.map { "\($0.id)|\($0.path)|\($0.kind)" }
    }
}

public enum SmeltQwen3TTSPackageProfiles {
    public static let runnable = SmeltQwen3TTSPackageProfile(
        packageName: "qwen3-tts-12hz.smeltpkg",
        modelName: "qwen3-tts-12hz",
        runtimeArchitecture: SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue,
        baseGraph: .qwen3TTS,
        compiledTalkerGraph: .qwen3TTSCompiledTalker,
        compiledTrunkNativeFrontEndGraph: .qwen3TTSCompiledTrunkNativeFrontEnd,
        loop: .qwen3TTS,
        tokenizerFiles: Qwen3TTSManifest.requiredTokenizerFiles,
        sidecars: SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks.map(\.sidecar),
        sidecarPaths: SmeltPackageSidecarProfiles.qwen3TTSRunnableSidecarPaths,
        pageSize: 16_384,
        maxTokens: 2048,
        eosTokens: [2150],
        performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
        structureProfileID: SmeltPackageStructureProfileID.qwen3TTSRunnable
    )

    public static let all = [runnable]
}
