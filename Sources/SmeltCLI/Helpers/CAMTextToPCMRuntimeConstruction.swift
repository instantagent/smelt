import Darwin
import Foundation
import SmeltRuntime
import SmeltSchema
import SmeltServe

struct CAMTextToPCMRuntimeConstruction: Sendable {
    let packagePath: String
    let runtimeRoute: CAMRuntimeRoute
    let decision: SmeltCAMPackageCapabilities.Decision
    let featureContract: SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract
    let artifactRoles: [String]
    let gateIDs: [String]
    let camSemanticSHA256: String
    let exportABISHA256: String

    init(
        packagePath: String,
        runtimeRoute: CAMRuntimeRoute,
        capabilities: SmeltCAMPackageCapabilities,
        decision: SmeltCAMPackageCapabilities.Decision,
        verb: String
    ) throws {
        requireCAMPackageInventoryOrExit(
            capabilities,
            packagePath: packagePath,
            verb: verb,
            requireAuthoredInventory: true
        )
        try self.init(
            admittedPackagePath: packagePath,
            runtimeRoute: runtimeRoute,
            capabilities: capabilities,
            decision: decision
        )
    }

    init(serveAdmission: SmeltTextToPCMServeAdmission) throws {
        try self.init(
            admittedPackagePath: serveAdmission.packagePath,
            runtimeRoute: serveAdmission.runtimeRoute,
            capabilities: serveAdmission.capabilities,
            decision: serveAdmission.decision
        )
    }

    init(runtimeAdmission: SmeltRuntimeAdmission) throws {
        try self.init(
            admittedPackagePath: runtimeAdmission.packagePath,
            runtimeRoute: runtimeAdmission.runtimeRoute,
            capabilities: runtimeAdmission.capabilities,
            decision: runtimeAdmission.decision
        )
    }

    private init(
        admittedPackagePath packagePath: String,
        runtimeRoute: CAMRuntimeRoute,
        capabilities: SmeltCAMPackageCapabilities,
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws {
        self.packagePath = packagePath
        self.runtimeRoute = runtimeRoute
        self.decision = decision
        featureContract = try capabilities.runtimeAssemblyFeatureContract(for: decision)
        guard featureContract.schema
            == SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract.currentSchema,
              !featureContract.featureSet.isEmpty
        else {
            throw CLIError(
                "CAM text-to-PCM construction feature contract is empty or unsupported"
            )
        }
        artifactRoles = capabilities.artifactRequirements.map(\.role).sorted()
        gateIDs = decision.matchedGateIDs.sorted()
        camSemanticSHA256 = capabilities.camSemanticSHA256
        exportABISHA256 = capabilities.exportABISHA256
    }

    func requirePackagePath(_ path: String) throws {
        guard URL(fileURLWithPath: path).standardizedFileURL.path
            == URL(fileURLWithPath: packagePath).standardizedFileURL.path
        else {
            throw CLIError("CAM text-to-PCM construction package path mismatch")
        }
    }

    func requireFeatureSet(
        _ requiredFeatures: Set<String>,
        description: String
    ) throws {
        let missing = requiredFeatures
            .subtracting(Set(featureContract.featureSet))
            .sorted()
        guard missing.isEmpty else {
            throw CLIError(
                "CAM text-to-PCM construction missing \(description) features: "
                    + missing.joined(separator: ", ")
            )
        }
    }

    func make24KRuntime(verb: String) throws -> CAMTextToPCM24KRuntime {
        try requireCAMTextToPCM24KAudioConstructionFeatures(
            self,
            description: "24khz \(verb) runtime"
        )
        let manifest = try preflightCAMTextToPCM24KPackage(packagePath: packagePath)
        return try openCAMTextToPCM24KRuntime(
            packagePath: packagePath,
            manifest: manifest
        )
    }

    func load24KVoiceDefaults() throws -> CAMTextToPCMVoiceDefaults? {
        try loadCAMTextToPCM24KVoiceDefaults(packagePath: packagePath)
    }

    func declared24KChunkSchedule() throws -> (first: Int, max: Int)? {
        try loadCAMTextToPCM24KDeclaredChunkSchedule(packagePath: packagePath)
    }

    func packageInterfaceContext() throws -> SmeltPackageInterface.InterfaceValidationContext {
        switch runtimeRoute {
        case .textToPCM(let outputRate) where outputRate == "24khz":
            return SmeltPackageInterface.packageValidationContext(
                graphPolicy: .sidecarTextToCodecAudio
            ).interfaceContext
        default:
            throw CLIError(
                "CAM text-to-PCM construction cannot validate declared args for route"
            )
        }
    }

    func validate24KVoice(language: String, speaker: String?) throws {
        try validateCAMTextToPCM24KVoice(
            packagePath: packagePath,
            language: language,
            speaker: speaker
        )
    }

    func makeServeRuntime(verb: String) throws -> CAMTextToPCMServeRuntime {
        switch runtimeRoute {
        case .textToPCM(let outputRate) where outputRate == "24khz":
            return .audio24k(try make24KRuntime(verb: verb))
        default:
            throw CLIError("CAM text-to-PCM construction cannot make serve runtime for route")
        }
    }
}

enum CAMTextToPCMDecodeMode: Sendable, Equatable {
    case greedy
    case sampleSeeded(UInt64)
    case packageDefault
}

struct CAMTextToPCMStreamChunk: Sendable {
    let samples: [Float]
    let frameOffset: Int
    let frameCount: Int
    let isFinal: Bool
}

struct CAMTextToPCMVoiceDefaults: Sendable, Equatable {
    let speaker: String?
    let language: String?
    let instruct: String?
    let firstChunkFrames: Int?
    let maxChunkFrames: Int?
    let maxFrames: Int?
}

struct CAMTextToPCMVoiceCatalog: Sendable, Equatable {
    let speakers: [String]
    let languages: [String]
}

struct CAMTextToPCMServeRequest: Sendable {
    let text: String
    let speaker: String?
    let language: String?
    let instruct: String?
    let maxFrames: Int?
    let firstChunkFrames: Int?
    let maxChunkFrames: Int?
    let decode: CAMTextToPCMDecodeMode
    let seed: UInt64?
}

enum CAMTextToPCMServeRuntime: @unchecked Sendable {
    case audio24k(CAMTextToPCM24KRuntime)

    var modelID: String {
        switch self {
        case .audio24k(let runtime):
            return runtime.modelName
        }
    }

    var sampleRate: Int {
        switch self {
        case .audio24k:
            return CAMTextToPCM24KAudio.sampleRate
        }
    }

    var audioChannels: Int {
        switch self {
        case .audio24k:
            return 1
        }
    }

    var voiceDefaults: CAMTextToPCMVoiceDefaults? {
        switch self {
        case .audio24k(let runtime):
            return runtime.voiceDefaults
        }
    }

    var voiceCatalog: CAMTextToPCMVoiceCatalog {
        switch self {
        case .audio24k(let runtime):
            return runtime.voiceCatalog
        }
    }

    var declaredChunkSchedule: (first: Int, max: Int)? {
        switch self {
        case .audio24k(let runtime):
            return runtime.declaredChunkSchedule
        }
    }

    func prewarmForServe() throws {
        switch self {
        case .audio24k(let runtime):
            try runtime.prewarmCompiledTrunk()
            guard runtime.tokenizerAvailable else {
                throw CLIError(
                    "text-to-PCM package has no bundled tokenizer/config "
                        + "(weights-only build?) — rebuild from a full checkpoint"
                )
            }
        }
    }

    func generateStreaming(
        request: CAMTextToPCMServeRequest,
        onSamples: ([Float]) throws -> Bool
    ) throws {
        switch self {
        case .audio24k(let runtime):
            guard let maxFrames = request.maxFrames else {
                throw CLIError("24khz serve request missing max frame count")
            }
            try runtime.generateStreaming(
                text: request.text,
                instruct: request.instruct,
                language: request.language ?? "Auto",
                speaker: request.speaker,
                maxFrames: maxFrames,
                decode: request.decode,
                firstChunkFrames: request.firstChunkFrames,
                maxChunkFrames: request.maxChunkFrames
            ) { chunk in
                guard !chunk.samples.isEmpty else { return true }
                return try onSamples(chunk.samples)
            }
        }
    }
}

private protocol CAMTextToPCM24KRuntimeBackend: AnyObject {
    var tokenizerAvailable: Bool { get }
    var modelName: String { get }
    var voiceDefaults: CAMTextToPCMVoiceDefaults? { get }
    var voiceCatalog: CAMTextToPCMVoiceCatalog { get }
    var declaredChunkSchedule: (first: Int, max: Int)? { get }

    func validateVoice(language: String, speaker: String?) throws
    func prewarmCompiledTrunk() throws
    func generateStreaming(
        text: String,
        instruct: String?,
        language: String,
        speaker: String?,
        maxFrames: Int,
        decode: CAMTextToPCMDecodeMode,
        firstChunkFrames: Int?,
        maxChunkFrames: Int?,
        trace: SmeltRuntimeTraceRecorder?,
        onChunk: (CAMTextToPCMStreamChunk) throws -> Bool
    ) throws
}

struct CAMTextToPCM24KRuntime: @unchecked Sendable {
    private let backend: any CAMTextToPCM24KRuntimeBackend

    fileprivate init(backend: any CAMTextToPCM24KRuntimeBackend) {
        self.backend = backend
    }

    var tokenizerAvailable: Bool {
        backend.tokenizerAvailable
    }

    var modelName: String {
        backend.modelName
    }

    var voiceDefaults: CAMTextToPCMVoiceDefaults? {
        backend.voiceDefaults
    }

    var voiceCatalog: CAMTextToPCMVoiceCatalog {
        backend.voiceCatalog
    }

    var declaredChunkSchedule: (first: Int, max: Int)? {
        backend.declaredChunkSchedule
    }

    func validateVoice(language: String, speaker: String?) throws {
        try backend.validateVoice(language: language, speaker: speaker)
    }

    func prewarmCompiledTrunk() throws {
        try backend.prewarmCompiledTrunk()
    }

    func generateStreaming(
        text: String,
        instruct: String?,
        language: String,
        speaker: String?,
        maxFrames: Int,
        decode: CAMTextToPCMDecodeMode,
        firstChunkFrames: Int?,
        maxChunkFrames: Int?,
        trace: SmeltRuntimeTraceRecorder? = nil,
        onChunk: (CAMTextToPCMStreamChunk) throws -> Bool
    ) throws {
        try backend.generateStreaming(
            text: text,
            instruct: instruct,
            language: language,
            speaker: speaker,
            maxFrames: maxFrames,
            decode: decode,
            firstChunkFrames: firstChunkFrames,
            maxChunkFrames: maxChunkFrames,
            trace: trace,
            onChunk: onChunk
        )
    }
}

private let camTextToPCM24KAudioConstructionFeatures: Set<String> = [
    "block.codec-decoder",
    "block.codec-decoder.streaming",
    "block.frontend",
    "block.frontend.speaker-conditioning",
    "block.requirement.audio-format",
    "block.requirement.audio-format.pcm-f32",
    "block.requirement.audio-rate",
    "block.requirement.audio-rate.24khz",
    "block.requirement.streaming",
    "block.transformer",
    "graph.edge.pcm",
    "graph.edge.text",
    "graph.impl.compiled",
    "graph.impl.native",
    "io.pcm",
    "io.text",
]

func requireCAMTextToPCM24KAudioConstructionFeatures(
    _ construction: CAMTextToPCMRuntimeConstruction,
    description: String
) throws {
    guard construction.runtimeRoute.textToPCMOutputRate == "24khz" else {
        throw CLIError("CAM text-to-PCM construction route mismatch")
    }
    try construction.requireFeatureSet(
        camTextToPCM24KAudioConstructionFeatures,
        description: description
    )
}

private final class Qwen3TTSTextToPCM24KRuntimeBackend: CAMTextToPCM24KRuntimeBackend {
    private let manifest: Qwen3TTSManifest
    private let gpu: Qwen3TTSGPU

    init(packagePath: String, manifest: Qwen3TTSManifest) throws {
        self.manifest = manifest
        gpu = try Qwen3TTSGPU(packagePath: packagePath)
    }

    var tokenizerAvailable: Bool {
        gpu.tokenizer != nil && gpu.frontEndConfig != nil
    }

    var modelName: String {
        manifest.modelName
    }

    var voiceDefaults: CAMTextToPCMVoiceDefaults? {
        gpu.voice.map(CAMTextToPCMVoiceDefaults.init)
    }

    var voiceCatalog: CAMTextToPCMVoiceCatalog {
        CAMTextToPCMVoiceCatalog(
            speakers: gpu.frontEndConfig.map { Array($0.speakerIds.keys).sorted() } ?? [],
            languages: gpu.frontEndConfig.map { Array($0.languageIds.keys).sorted() } ?? []
        )
    }

    var declaredChunkSchedule: (first: Int, max: Int)? {
        if case .chunked(let first, let max, _, _)? = manifest.loop?.emission {
            return (first, max)
        }
        return nil
    }

    func validateVoice(language: String, speaker: String?) throws {
        guard let config = gpu.frontEndConfig else { return }
        _ = try config.ids(language: language, speaker: speaker)
    }

    func prewarmCompiledTrunk() throws {
        try gpu.prewarmCompiledTrunk()
    }

    func generateStreaming(
        text: String,
        instruct: String?,
        language: String,
        speaker: String?,
        maxFrames: Int,
        decode: CAMTextToPCMDecodeMode,
        firstChunkFrames: Int?,
        maxChunkFrames: Int?,
        trace: SmeltRuntimeTraceRecorder?,
        onChunk: (CAMTextToPCMStreamChunk) throws -> Bool
    ) throws {
        try gpu.generateStreaming(
            text: text,
            instruct: instruct,
            language: language,
            speaker: speaker,
            maxFrames: maxFrames,
            decode: decode.qwen3TTSDecodeMode,
            firstChunkFrames: firstChunkFrames,
            maxChunkFrames: maxChunkFrames,
            trace: trace
        ) { chunk in
            try onChunk(CAMTextToPCMStreamChunk(chunk))
        }
    }
}

private extension CAMTextToPCMDecodeMode {
    var qwen3TTSDecodeMode: Qwen3TTSSampler.DecodeMode {
        switch self {
        case .greedy:
            return .greedy
        case .packageDefault:
            return .packageDefault
        case .sampleSeeded(let seed):
            return .sampleSeeded(seed)
        }
    }
}

private extension CAMTextToPCMStreamChunk {
    init(_ chunk: Qwen3TTSGPU.StreamChunk) {
        self.init(
            samples: chunk.samples,
            frameOffset: chunk.frameOffset,
            frameCount: chunk.frameCount,
            isFinal: chunk.isFinal
        )
    }
}

private extension CAMTextToPCMVoiceDefaults {
    init(_ voice: Qwen3TTSVoice) {
        self.init(
            speaker: voice.speaker,
            language: voice.language,
            instruct: voice.instruct,
            firstChunkFrames: voice.firstChunkFrames,
            maxChunkFrames: voice.maxChunkFrames,
            maxFrames: voice.maxFrames
        )
    }
}

enum CAMTextToPCM24KAudio {
    static let sampleRate = Qwen3TTSAudio.sampleRate

    static func pcm16(_ samples: [Float]) -> Data {
        Qwen3TTSAudio.pcm16(samples)
    }

    static func wav(_ samples: [Float]) -> Data {
        Qwen3TTSAudio.wav(samples)
    }

    static func wavHeader(pcmBytes: Int?) -> Data {
        Qwen3TTSAudio.wavHeader(pcmBytes: pcmBytes)
    }

    static func wavSizePatches(pcmBytes: Int) -> [(offset: Int, value: UInt32)] {
        Qwen3TTSAudio.wavSizePatches(pcmBytes: pcmBytes)
    }
}

enum CAMTextToPCM24KPackageLayout {
    static var lingerIdentityFiles: [String] {
        SmeltPackageStructureProfiles.qwen3TTSRunnableBaseFiles
            + Qwen3TTSManifest.requiredTokenizerFiles
            + [Qwen3TTSVoice.fileName, SmeltPackageInterface.fileName]
    }
}

private func openCAMTextToPCM24KRuntime(
    packagePath: String,
    manifest: Qwen3TTSManifest
) throws -> CAMTextToPCM24KRuntime {
    try CAMTextToPCM24KRuntime(
        backend: Qwen3TTSTextToPCM24KRuntimeBackend(
            packagePath: packagePath,
            manifest: manifest
        )
    )
}

private func loadCAMTextToPCM24KVoiceDefaults(
    packagePath: String
) throws -> CAMTextToPCMVoiceDefaults? {
    try Qwen3TTSVoice.load(packagePath: packagePath).map(CAMTextToPCMVoiceDefaults.init)
}

private func loadCAMTextToPCM24KDeclaredChunkSchedule(
    packagePath: String
) throws -> (first: Int, max: Int)? {
    let manifestURL = URL(fileURLWithPath: packagePath, isDirectory: true)
        .appendingPathComponent("manifest.json")
    let manifest = try Qwen3TTSManifest.decode(from: Data(contentsOf: manifestURL))
    try manifest.validateQwen3TTSValidation(packagePath: packagePath)
    try requireCAMTextToPCM24KGraph(manifest)
    if case .chunked(let first, let max, _, _)? = manifest.loop?.emission {
        return (first, max)
    }
    return nil
}

private func validateCAMTextToPCM24KVoice(
    packagePath: String,
    language: String,
    speaker: String?
) throws {
    let config = try Qwen3TTSFrontEnd.Config.load(
        configJSONPath: "\(packagePath)/config.json"
    )
    _ = try config.ids(language: language, speaker: speaker)
}

private func preflightCAMTextToPCM24KPackage(
    packagePath: String
) throws -> Qwen3TTSManifest {
    let manifestURL = URL(fileURLWithPath: packagePath, isDirectory: true)
        .appendingPathComponent("manifest.json")
    let manifest = try Qwen3TTSManifest.decode(from: Data(contentsOf: manifestURL))
    guard manifest.version == 1 else {
        throw CAMTextToPCM24KPreflightError.unsupportedVersion(manifest.version)
    }
    try manifest.validateQwen3TTSValidation(packagePath: packagePath)
    try requireCAMTextToPCM24KGraph(manifest)
    guard !manifest.eosTokens.isEmpty else {
        throw CAMTextToPCM24KPreflightError.noEosTokens
    }

    let hostPage = Int(getpagesize())
    guard manifest.pageSize > 0, manifest.pageSize % hostPage == 0 else {
        throw CAMTextToPCM24KPreflightError.pageSizeMismatch(
            manifest: manifest.pageSize,
            host: hostPage
        )
    }
    guard let tokenizerFiles = manifest.tokenizerFiles,
          Set(Qwen3TTSManifest.requiredTokenizerFiles).isSubset(of: Set(tokenizerFiles))
    else {
        throw CAMTextToPCM24KPreflightError.tokenizerIncomplete(
            manifest.tokenizerFiles ?? []
        )
    }

    try requireCAMTextToPCM24KFiles(
        packagePath: packagePath,
        files: Array(Set(
            SmeltPackageStructureProfiles.qwen3TTSRunnableBaseFiles
                + Qwen3TTSManifest.requiredTokenizerFiles
        )).sorted()
    )
    guard manifest.totalBytes > 0, manifest.totalBytes <= UInt64(Int.max) else {
        throw CAMTextToPCM24KPreflightError.weightsSizeMismatch("weights.bin")
    }
    let weightsURL = URL(fileURLWithPath: packagePath, isDirectory: true)
        .appendingPathComponent("weights.bin")
    let fileSize = try FileManager.default.attributesOfItem(atPath: weightsURL.path)[.size]
        .flatMap { $0 as? NSNumber }?
        .uint64Value ?? 0
    guard fileSize >= manifest.totalBytes else {
        throw CAMTextToPCM24KPreflightError.weightsSizeMismatch("weights.bin")
    }

    let pageSize = UInt64(manifest.pageSize)
    for entry in manifest.weights {
        guard entry.offset % pageSize == 0, entry.byteLength % pageSize == 0 else {
            throw CAMTextToPCM24KPreflightError.weightMisaligned(entry.name)
        }
        guard entry.offset <= manifest.totalBytes,
              entry.byteLength <= manifest.totalBytes - entry.offset
        else {
            throw CAMTextToPCM24KPreflightError.weightOutOfBounds(entry.name)
        }
    }
    return manifest
}

private func requireCAMTextToPCM24KFiles(
    packagePath: String,
    files: [String]
) throws {
    let packageURL = URL(fileURLWithPath: packagePath, isDirectory: true)
    let missing = files.filter {
        !FileManager.default.fileExists(
            atPath: packageURL.appendingPathComponent($0).path
        )
    }
    guard missing.isEmpty else {
        throw CAMTextToPCM24KPreflightError.missingFiles(missing)
    }
}

private func requireCAMTextToPCM24KGraph(_ manifest: Qwen3TTSManifest) throws {
    guard let blocks = manifest.blocks,
          let policy = try? SmeltRuntimeGraphPolicy.resolve(blocks: blocks)
    else {
        throw CAMTextToPCM24KPreflightError.graphPolicyMismatch("unresolved")
    }
    guard policy == .sidecarTextToCodecAudio else {
        throw CAMTextToPCM24KPreflightError.graphPolicyMismatch(policy.rawValue)
    }
}

private enum CAMTextToPCM24KPreflightError: Error, CustomStringConvertible {
    case unsupportedVersion(Int)
    case graphPolicyMismatch(String)
    case noEosTokens
    case pageSizeMismatch(manifest: Int, host: Int)
    case tokenizerIncomplete([String])
    case missingFiles([String])
    case weightsSizeMismatch(String)
    case weightOutOfBounds(String)
    case weightMisaligned(String)

    var description: String {
        switch self {
        case let .unsupportedVersion(version):
            return "unsupported package version \(version)"
        case let .graphPolicyMismatch(policy):
            return "Qwen3-TTS CAM package must declare a text-to-PCM graph, got \(policy)"
        case .noEosTokens:
            return "manifest has no eosTokens"
        case let .pageSizeMismatch(manifest, host):
            return "package pageSize \(manifest) not a multiple of host page \(host)"
        case let .tokenizerIncomplete(files):
            return "package tokenizerFiles incomplete: \(files)"
        case let .missingFiles(files):
            return "missing files: \(files.joined(separator: ","))"
        case let .weightsSizeMismatch(path):
            return "weights.bin smaller than manifest totalBytes: \(path)"
        case let .weightOutOfBounds(name):
            return "weight \(name) slice extends past weights.bin"
        case let .weightMisaligned(name):
            return "weight \(name) offset/length not page-aligned"
        }
    }
}

func makeCAMTextToPCMRuntimeConstructionOrExit(
    packagePath: String,
    capabilities: SmeltCAMPackageCapabilities,
    decision: SmeltCAMPackageCapabilities.Decision,
    verb: String
) -> CAMTextToPCMRuntimeConstruction {
    makeCAMTextToPCMRuntimeConstructionOrExit(
        packagePath: packagePath,
        capabilities: capabilities,
        decision: decision,
        verb: verb,
        expectedRoute: .textToPCM(outputRate: "24khz"),
        expectedRouteDescription: "text-to-PCM 24khz"
    )
}

private func makeCAMTextToPCMRuntimeConstructionOrExit(
    packagePath: String,
    capabilities: SmeltCAMPackageCapabilities,
    decision: SmeltCAMPackageCapabilities.Decision,
    verb: String,
    expectedRoute: CAMRuntimeRoute,
    expectedRouteDescription: String
) -> CAMTextToPCMRuntimeConstruction {
    let runtimeRoute = resolveCAMRuntimeRouteOrExit(
        capabilities: capabilities,
        decision: decision,
        verb: verb
    )
    guard runtimeRoute == expectedRoute else {
        fputs("smelt \(verb): CAM route is not \(expectedRouteDescription)\n", stderr)
        exit(1)
    }
    do {
        return try CAMTextToPCMRuntimeConstruction(
            packagePath: packagePath,
            runtimeRoute: runtimeRoute,
            capabilities: capabilities,
            decision: decision,
            verb: verb
        )
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
}
