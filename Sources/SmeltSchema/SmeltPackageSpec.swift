// SmeltPackageSpec — CAM's pure assembly contract.
//
// This is the config-side shape before any package directory is written. It
// binds the existing graph/schedule vocabulary to source weights, tensor maps,
// runtime policy, package files, and resolved inference policy. Builders can
// lower model-specific inputs into this value first, validate it, then hand a
// resolved plan to the package writer.

import Foundation

public enum SmeltPackageSidecarKind {
    public static let compiledTrunk = "compiled-trunk"

    public static let known: Set<String> = [
        compiledTrunk,
    ]
}

public struct SmeltHeadlessTrunkSidecarProfile: Sendable, Equatable {
    public let id: String
    public let path: String
    public let kind: String
    public let modelName: String

    public init(id: String, path: String, kind: String, modelName: String) {
        self.id = id
        self.path = path
        self.kind = kind
        self.modelName = modelName
    }

    public var sidecar: SmeltPackageSpec.Sidecar {
        .init(id: id, path: path, kind: kind)
    }
}

public enum SmeltPackageSidecarProfiles {
    public static let qwen3TTSTalkerTrunk = SmeltHeadlessTrunkSidecarProfile(
        id: "talker-trunk",
        path: "trunk",
        kind: SmeltPackageSidecarKind.compiledTrunk,
        modelName: "qwen3-tts-12hz-talker-trunk"
    )

    public static let qwen3TTSMTPTrunk = SmeltHeadlessTrunkSidecarProfile(
        id: "mtp-trunk",
        path: "trunk-mtp",
        kind: SmeltPackageSidecarKind.compiledTrunk,
        modelName: "qwen3-tts-12hz-mtp-trunk"
    )

    public static let qwen3TTSRunnableHeadlessTrunks = [
        qwen3TTSTalkerTrunk,
        qwen3TTSMTPTrunk,
    ]

    public static let qwen3TTSRunnableSidecarPaths =
        qwen3TTSRunnableHeadlessTrunks.map(\.path)
}

public enum SmeltPackagePerformanceGateID {
    public static let textDecodePrefillStartup = "text.decode-prefill-startup"
    public static let qwen3TTSTTFA = "qwen3-tts.ttfa"

    public static let known: Set<String> = [
        textDecodePrefillStartup,
        qwen3TTSTTFA,
    ]
}

public enum SmeltPackageStructureProfileID {
    public static let qwen3TTSRunnable = "qwen3-tts.runnable"

    public static let known: Set<String> = [
        qwen3TTSRunnable,
    ]
}

public enum SmeltPackageStructureProfiles {
    public static let qwen3TTSRunnableBaseFiles = [
        "manifest.json",
        "model.metallib",
        "weights.bin",
    ] + SmeltPackageSidecarProfiles.qwen3TTSRunnableSidecarPaths

    public static func qwen3TTSRunnable(
        pipelines: [String],
        tokenizerFiles: [String],
        graph: SmeltBlockGraph
    ) -> SmeltPackageSpec.Validation.StructureProfile {
        .init(
            id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
            requiredPipelines: pipelines,
            requiredFiles: qwen3TTSRunnableBaseFiles + tokenizerFiles,
            requiredRoutes: graph.runtimeRouteSignatures
        )
    }
}

public enum SmeltPackagePerformanceTraceLabel {
    public static let execToMain = "exec -> main"
    public static let execToMainDyld = "exec -> main (dyld)"
    public static let tokenizerLoad = "tokenizer load"
    public static let smeltModelInitTotal = "SmeltModel init (total)"

    public static let textDecodePrefillStartupRequired = [
        execToMainDyld,
        tokenizerLoad,
        smeltModelInitTotal,
    ]
}

public enum SmeltPackagePerformanceMetricName {
    public static let decodeMedianMSPerToken = "decode_median_ms_per_token"
    public static let decodeTokensPerSecond = "decode_tokens_per_second"
    public static let decodeP95MSPerToken = "decode_p95_ms_per_token"
    public static let prefill64WallMS = "prefill64_wall_ms"
    public static let prefill64TokensPerSecond = "prefill64_tokens_per_second"
    public static let prefill64P95MS = "prefill64_p95_ms"
    public static let prefill256WallMS = "prefill256_wall_ms"
    public static let prefill256TokensPerSecond = "prefill256_tokens_per_second"
    public static let prefill256P95MS = "prefill256_p95_ms"
    public static let traceFirstTokenMS = "trace_first_token_ms"

    public static let bootTimeSeconds = "boot_time_sec"
    public static let cacheState = "cache_state"
    public static let coldFirstTTFASeconds = "cold_first_ttfa_s"
    public static let coldTTFAMedianSeconds = "cold_ttfa_median_s"
    public static let command = "command"
    public static let hostColdStampPath = "host_cold_stamp_path"
    public static let hostGlobalColdCandidate = "host_global_cold_candidate"
    public static let manifestSHA256 = "manifest_sha256"
    public static let mpsLinearRowTile = "mps_linear_row_tile"
    public static let overlapLaterSameWorkspaceDuringFirstAudio =
        "overlap_later_same_workspace_during_first_audio"
    public static let package = "package"
    public static let primeStreamingSetupBeforeFirstAudio =
        "prime_streaming_setup_before_first_audio"
    public static let processColdFirstTTFASeconds = "process_cold_first_ttfa_s"
    public static let provenancePath = "provenance_path"
    public static let startupExecToMainSeconds = "startup_exec_to_main_s"
    public static let tracePath = "trace_path"
    public static let ttfaMedianSeconds = "ttfa_median_s"
    public static let workspaceSeconds = "workspace_s"

    public static let firstAudioSeconds = "first_audio_s"
    public static let ttfaSeconds = "ttfa_s"
    // Real-time factor: generation wall / produced-audio duration. < 1 => faster
    // than real time. Derived per-sample from the same `smelt run` measurement that
    // yields TTFA (wall and audio duration are already emitted alongside it).
    public static let realTimeFactor = "rtf"

    public static let textDecodePrefillStartupRequired = [
        decodeMedianMSPerToken,
        decodeTokensPerSecond,
        decodeP95MSPerToken,
        prefill64WallMS,
        prefill64TokensPerSecond,
        prefill64P95MS,
        prefill256WallMS,
        prefill256TokensPerSecond,
        prefill256P95MS,
        traceFirstTokenMS,
    ]

    public static let qwen3TTSTTFARequired = [
        firstAudioSeconds,
        ttfaSeconds,
        realTimeFactor,
    ]

}

public enum SmeltPackagePerformanceUnit {
    public static let milliseconds = "ms"
    public static let seconds = "s"
    public static let tokensPerSecond = "tok/s"
    // Dimensionless wall/audio ratio ("x real time"). Kept distinct from `seconds`
    // so every unit guard stays exhaustive: a metric never silently borrows the
    // wrong unit.
    public static let realTimeFactor = "x"
}

public enum SmeltPackagePerformanceBudget {
    // Package trace-first-token budgets are intentionally independent of the
    // module startup gate's elapsed bound (the two may coincide, as qwen35_fast does at 100 ms).
    public static let textTraceFirstTokenMaxMS: Double = 100
    public static let qwen35TextTraceFirstTokenMaxMS: Double = 115
    public static let qwen3TTSFirstAudioMaxSeconds: Double = 0.50
    public static let qwen3TTSTTFAMaxSeconds: Double = 0.50

    // Decode / prefill throughput floors (min bounds) for the two shipped text
    // SKUs. Set to worst-observed tok/s x (1 - margin), margin ~15% — the same
    // conservative envelope the startup ceilings use. Model-keyed because the two
    // SKUs differ materially in throughput; non-shipped models get no floor (only
    // these two run the release gate). Derived from the quiescent bench pass in
    // docs/receipts/bounds-2026-07-11.md (measured at load<3.5 under the bench
    // lock); these are canonical-gate floors, not load-invariant promises.
    public static let qwen35FastDecodeMinTPS: Double = 280
    public static let qwen35FastPrefill64MinTPS: Double = 2280
    public static let qwen35FastPrefill256MinTPS: Double = 2600
    public static let qwen35TextDecodeMinTPS: Double = 170
    public static let qwen35TextPrefill64MinTPS: Double = 820
    public static let qwen35TextPrefill256MinTPS: Double = 850

    // TTS real-time-factor ceiling (generation wall / produced-audio duration).
    // Set to worst-observed x (1 + margin); see docs/receipts/bounds-2026-07-11.md.
    public static let qwen3TTSRealTimeFactorMax: Double = 1.20

    public static func textTraceFirstTokenMaxMS(forModelName modelName: String?) -> Double {
        switch modelName {
        case "Qwen/Qwen3.5-2B":
            qwen35TextTraceFirstTokenMaxMS
        default:
            textTraceFirstTokenMaxMS
        }
    }

    public static func textDecodeMinTPS(forModelName modelName: String?) -> Double? {
        switch modelName {
        case "Qwen/Qwen3.5-0.8B": qwen35FastDecodeMinTPS
        case "Qwen/Qwen3.5-2B": qwen35TextDecodeMinTPS
        default: nil
        }
    }

    public static func textPrefill64MinTPS(forModelName modelName: String?) -> Double? {
        switch modelName {
        case "Qwen/Qwen3.5-0.8B": qwen35FastPrefill64MinTPS
        case "Qwen/Qwen3.5-2B": qwen35TextPrefill64MinTPS
        default: nil
        }
    }

    public static func textPrefill256MinTPS(forModelName modelName: String?) -> Double? {
        switch modelName {
        case "Qwen/Qwen3.5-0.8B": qwen35FastPrefill256MinTPS
        case "Qwen/Qwen3.5-2B": qwen35TextPrefill256MinTPS
        default: nil
        }
    }
}

public enum SmeltPackagePerformanceProfiles {
    public static func validation(
        parityFixture: String?,
        performanceGate: String,
        modelName: String? = nil,
        structureProfile: SmeltPackageSpec.Validation.StructureProfile? = nil
    ) -> SmeltPackageSpec.Validation {
        .init(
            parityFixture: parityFixture,
            performanceGate: performanceGate,
            performanceProfile: profile(for: performanceGate, modelName: modelName),
            structureProfile: structureProfile
        )
    }

    public static func profile(
        for gate: String,
        modelName: String? = nil
    ) -> SmeltPackageSpec.Validation.PerformanceProfile {
        switch gate {
        case SmeltPackagePerformanceGateID.textDecodePrefillStartup:
            return .init(
                gate: gate,
                command: .run,
                requiredTraceLabels: SmeltPackagePerformanceTraceLabel
                    .textDecodePrefillStartupRequired,
                requiredOutputMetrics: SmeltPackagePerformanceMetricName
                    .textDecodePrefillStartupRequired,
                minBounds: textMinBounds(forModelName: modelName),
                maxBounds: [
                    .init(
                        metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                        max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS(
                            forModelName: modelName
                        ),
                        unit: SmeltPackagePerformanceUnit.milliseconds
                    ),
                ]
            )
        case SmeltPackagePerformanceGateID.qwen3TTSTTFA:
            return .init(
                gate: gate,
                command: .bench,
                requiredOutputMetrics: SmeltPackagePerformanceMetricName
                    .qwen3TTSTTFARequired,
                maxBounds: [
                    .init(
                        metric: SmeltPackagePerformanceMetricName.firstAudioSeconds,
                        max: SmeltPackagePerformanceBudget.qwen3TTSFirstAudioMaxSeconds,
                        unit: SmeltPackagePerformanceUnit.seconds
                    ),
                    .init(
                        metric: SmeltPackagePerformanceMetricName.ttfaSeconds,
                        max: SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds,
                        unit: SmeltPackagePerformanceUnit.seconds
                    ),
                    .init(
                        metric: SmeltPackagePerformanceMetricName.realTimeFactor,
                        max: SmeltPackagePerformanceBudget.qwen3TTSRealTimeFactorMax,
                        unit: SmeltPackagePerformanceUnit.realTimeFactor
                    ),
                ]
            )
        default:
            return .init(gate: gate, command: .bench)
        }
    }

    static func textMinBounds(
        forModelName modelName: String?
    ) -> [SmeltPackageSpec.Validation.PerformanceProfile.MinBound] {
        var bounds: [SmeltPackageSpec.Validation.PerformanceProfile.MinBound] = []
        if let tps = SmeltPackagePerformanceBudget.textDecodeMinTPS(forModelName: modelName) {
            bounds.append(.init(
                metric: SmeltPackagePerformanceMetricName.decodeTokensPerSecond,
                min: tps,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ))
        }
        if let tps = SmeltPackagePerformanceBudget.textPrefill64MinTPS(forModelName: modelName) {
            bounds.append(.init(
                metric: SmeltPackagePerformanceMetricName.prefill64TokensPerSecond,
                min: tps,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ))
        }
        if let tps = SmeltPackagePerformanceBudget.textPrefill256MinTPS(forModelName: modelName) {
            bounds.append(.init(
                metric: SmeltPackagePerformanceMetricName.prefill256TokensPerSecond,
                min: tps,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ))
        }
        return bounds
    }
}

public enum SmeltPackageSpecError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "package spec: \(why)"
        }
    }
}

public indirect enum SmeltPackageSpecValue: Codable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case number(Double)
    case bool(Bool)
    case array([SmeltPackageSpecValue])
    case object([String: SmeltPackageSpecValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([SmeltPackageSpecValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: SmeltPackageSpecValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    fileprivate var hasObjectPayload: Bool {
        if case .object(let object) = self { return !object.isEmpty }
        return false
    }
}

public struct SmeltPackageSpec: Codable, Sendable {
    public static let currentVersion = 1

    public struct Source: Codable, Sendable {
        public enum Kind: String, Codable, Sendable {
            case localFile = "local-file"
            case localDirectory = "local-directory"
            case huggingFace = "hugging-face"
        }

        public let id: String
        public let kind: Kind
        public let path: String?
        public let repo: String?
        public let revision: String?

        public init(
            id: String,
            kind: Kind,
            path: String? = nil,
            repo: String? = nil,
            revision: String? = nil
        ) {
            self.id = id
            self.kind = kind
            self.path = path
            self.repo = repo
            self.revision = revision
        }
    }

    public struct RuntimeDescriptor: Codable, Sendable {
        public enum Command: String, Codable, Sendable, CaseIterable {
            case load
            case run
            case prepare
            case serve
            case bench
            case trace
            case lingerWorker = "linger-worker"
        }

        public struct BlockRoute: Codable, Sendable, Equatable {
            public let block: String
            public let impl: SmeltBlockGraph.Impl
            public let delivery: SmeltBlockGraph.CompiledDelivery?

            public init(
                block: String,
                impl: SmeltBlockGraph.Impl,
                delivery: SmeltBlockGraph.CompiledDelivery? = nil
            ) {
                self.block = block
                self.impl = impl
                self.delivery = delivery
            }

            public var signature: String {
                "\(block):\(impl.rawValue):\(delivery?.rawValue ?? "none")"
            }
        }

        public let architecture: String
        public let commands: [Command]
        public let routes: [BlockRoute]

        public init(
            architecture: String,
            commands: [Command],
            routes: [BlockRoute]
        ) {
            self.architecture = architecture
            self.commands = commands
            self.routes = routes
        }

        public static func forGraph(
            architecture: String,
            commands: [Command],
            graph: SmeltBlockGraph
        ) -> RuntimeDescriptor {
            .init(
                architecture: architecture,
                commands: commands,
                routes: graph.runtimeRoutes
            )
        }

    }

    public enum TensorDType: String, Codable, Sendable {
        case f32
        case f16
        case bf16
        case u4
        case binary1
        case ternary2
        case gptq
        case turboQuantH = "turbo_quant_h"
        case raw
    }

    public struct TensorMap: Codable, Sendable {
        public let source: String
        public let name: String
        public let canonicalName: String
        public let block: String
        public let sourceDType: TensorDType
        public let storedDType: TensorDType
        public let shape: [Int]
        public let required: Bool

        public init(
            source: String,
            name: String,
            canonicalName: String,
            block: String,
            sourceDType: TensorDType,
            storedDType: TensorDType,
            shape: [Int],
            required: Bool = true
        ) {
            self.source = source
            self.name = name
            self.canonicalName = canonicalName
            self.block = block
            self.sourceDType = sourceDType
            self.storedDType = storedDType
            self.shape = shape
            self.required = required
        }

        enum CodingKeys: String, CodingKey {
            case source
            case name
            case canonicalName = "canonical_name"
            case block
            case sourceDType = "source_dtype"
            case storedDType = "stored_dtype"
            case shape
            case required
        }
    }

    public struct QuantizationPlan: Codable, Sendable {
        public struct Calibration: Codable, Sendable {
            public struct Corpus: Codable, Sendable {
                public let source: String
                public let path: String
                public let renderPolicy: String
                public let maxSamples: Int?
                public let maxTokens: Int?

                public init(
                    source: String,
                    path: String,
                    renderPolicy: String,
                    maxSamples: Int? = nil,
                    maxTokens: Int? = nil
                ) {
                    self.source = source
                    self.path = path
                    self.renderPolicy = renderPolicy
                    self.maxSamples = maxSamples
                    self.maxTokens = maxTokens
                }

                enum CodingKeys: String, CodingKey {
                    case source
                    case path
                    case renderPolicy = "render_policy"
                    case maxSamples = "max_samples"
                    case maxTokens = "max_tokens"
                }
            }

            public struct Artifact: Codable, Sendable {
                public let id: String
                public let path: String
                public let role: String

                public init(id: String, path: String, role: String) {
                    self.id = id
                    self.path = path
                    self.role = role
                }
            }

            public struct ResourceBounds: Codable, Sendable {
                public let maxSamples: Int?
                public let maxTokens: Int?
                public let maxLayersPerPass: Int?
                public let maxBytes: UInt64?

                public init(
                    maxSamples: Int? = nil,
                    maxTokens: Int? = nil,
                    maxLayersPerPass: Int? = nil,
                    maxBytes: UInt64? = nil
                ) {
                    self.maxSamples = maxSamples
                    self.maxTokens = maxTokens
                    self.maxLayersPerPass = maxLayersPerPass
                    self.maxBytes = maxBytes
                }

                enum CodingKeys: String, CodingKey {
                    case maxSamples = "max_samples"
                    case maxTokens = "max_tokens"
                    case maxLayersPerPass = "max_layers_per_pass"
                    case maxBytes = "max_bytes"
                }
            }

            public struct EqualityGate: Codable, Sendable {
                public let id: String
                public let candidate: String
                public let reference: String
                public let metric: String

                public init(
                    id: String,
                    candidate: String,
                    reference: String,
                    metric: String
                ) {
                    self.id = id
                    self.candidate = candidate
                    self.reference = reference
                    self.metric = metric
                }
            }

            public let corpus: Corpus?
            public let captureArtifacts: [Artifact]
            public let sideInputs: [Artifact]
            public let stagedPackages: [Artifact]
            public let resourceBounds: ResourceBounds?
            public let equalityGates: [EqualityGate]

            public init(
                corpus: Corpus? = nil,
                captureArtifacts: [Artifact] = [],
                sideInputs: [Artifact] = [],
                stagedPackages: [Artifact] = [],
                resourceBounds: ResourceBounds? = nil,
                equalityGates: [EqualityGate] = []
            ) {
                self.corpus = corpus
                self.captureArtifacts = captureArtifacts
                self.sideInputs = sideInputs
                self.stagedPackages = stagedPackages
                self.resourceBounds = resourceBounds
                self.equalityGates = equalityGates
            }

            enum CodingKeys: String, CodingKey {
                case corpus
                case captureArtifacts = "capture_artifacts"
                case sideInputs = "side_inputs"
                case stagedPackages = "staged_packages"
                case resourceBounds = "resource_bounds"
                case equalityGates = "equality_gates"
            }
        }

        public let format: TensorDType
        public let groupSize: Int?
        public let calibration: Calibration?

        public init(
            format: TensorDType,
            groupSize: Int? = nil,
            calibration: Calibration? = nil
        ) {
            self.format = format
            self.groupSize = groupSize
            self.calibration = calibration
        }

        enum CodingKeys: String, CodingKey {
            case format
            case groupSize = "group_size"
            case calibration
        }
    }

    public struct PackageFileSet: Codable, Sendable {
        public let manifest: String
        public let files: [String]

        public init(
            manifest: String = "manifest.json",
            files: [String]
        ) {
            self.manifest = manifest
            self.files = files
        }

        enum CodingKeys: String, CodingKey {
            case manifest
            case files
        }
    }

    public struct Artifact: Codable, Sendable {
        public let id: String
        public let path: String
        public let role: String

        public init(id: String, path: String, role: String) {
            self.id = id
            self.path = path
            self.role = role
        }
    }

    public struct Sidecar: Codable, Sendable {
        public let id: String
        public let path: String
        public let kind: String

        public init(id: String, path: String, kind: String) {
            self.id = id
            self.path = path
            self.kind = kind
        }
    }

    public struct Tokenizer: Codable, Sendable {
        public let format: String
        public let files: [String]

        public init(format: String, files: [String]) {
            self.format = format
            self.files = files
        }
    }

    public struct DecodePolicy: Codable, Sendable {
        public enum SamplerMode: String, Codable, Sendable {
            case greedy
            case sample
        }

        public struct Sampler: Codable, Sendable {
            public let mode: SamplerMode
            public let temperature: Double?
            public let topK: Int?
            public let topP: Double?

            public init(
                mode: SamplerMode,
                temperature: Double? = nil,
                topK: Int? = nil,
                topP: Double? = nil
            ) {
                self.mode = mode
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
            }

            enum CodingKeys: String, CodingKey {
                case mode
                case temperature
                case topK = "top_k"
                case topP = "top_p"
            }
        }

        public let sampler: Sampler
        public let subSampler: Sampler?
        public let maxSteps: Int?
        public let durationSeconds: Double?

        public init(
            sampler: Sampler,
            subSampler: Sampler? = nil,
            maxSteps: Int? = nil,
            durationSeconds: Double? = nil
        ) {
            self.sampler = sampler
            self.subSampler = subSampler
            self.maxSteps = maxSteps
            self.durationSeconds = durationSeconds
        }

        enum CodingKeys: String, CodingKey {
            case sampler
            case subSampler = "sub_sampler"
            case maxSteps = "max_steps"
            case durationSeconds = "duration_seconds"
        }
    }

    public struct Validation: Codable, Sendable, Equatable {
        public struct PerformanceProfile: Codable, Sendable, Equatable {
            public struct MinBound: Codable, Sendable, Equatable {
                public let metric: String
                public let min: Double
                public let unit: String

                public init(metric: String, min: Double, unit: String) {
                    self.metric = metric
                    self.min = min
                    self.unit = unit
                }
            }

            public struct Bound: Codable, Sendable, Equatable {
                public let metric: String
                public let max: Double
                public let unit: String

                public init(metric: String, max: Double, unit: String) {
                    self.metric = metric
                    self.max = max
                    self.unit = unit
                }
            }

            public let gate: String
            public let command: RuntimeDescriptor.Command
            public let requiredTraceLabels: [String]
            public let requiredOutputMetrics: [String]
            public let minBounds: [MinBound]
            public let maxBounds: [Bound]

            public init(
                gate: String,
                command: RuntimeDescriptor.Command,
                requiredTraceLabels: [String] = [],
                requiredOutputMetrics: [String] = [],
                minBounds: [MinBound] = [],
                maxBounds: [Bound] = []
            ) {
                self.gate = gate
                self.command = command
                self.requiredTraceLabels = requiredTraceLabels
                self.requiredOutputMetrics = requiredOutputMetrics
                self.minBounds = minBounds
                self.maxBounds = maxBounds
            }

            public init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                gate = try container.decode(String.self, forKey: .gate)
                command = try container.decode(RuntimeDescriptor.Command.self, forKey: .command)
                requiredTraceLabels = try container.decodeIfPresent(
                    [String].self,
                    forKey: .requiredTraceLabels
                ) ?? []
                requiredOutputMetrics = try container.decodeIfPresent(
                    [String].self,
                    forKey: .requiredOutputMetrics
                ) ?? []
                minBounds = try container.decodeIfPresent([MinBound].self, forKey: .minBounds) ?? []
                maxBounds = try container.decodeIfPresent([Bound].self, forKey: .maxBounds) ?? []
            }

            enum CodingKeys: String, CodingKey {
                case gate
                case command
                case requiredTraceLabels = "required_trace_labels"
                case requiredOutputMetrics = "required_output_metrics"
                case minBounds = "min_bounds"
                case maxBounds = "max_bounds"
            }
        }

        public struct StructureProfile: Codable, Sendable, Equatable {
            public let id: String
            public let requiredPipelines: [String]
            public let forbiddenPipelines: [String]
            public let requiredFiles: [String]
            public let forbiddenFiles: [String]
            public let requiredRoutes: [String]

            public init(
                id: String,
                requiredPipelines: [String] = [],
                forbiddenPipelines: [String] = [],
                requiredFiles: [String] = [],
                forbiddenFiles: [String] = [],
                requiredRoutes: [String] = []
            ) {
                self.id = id
                self.requiredPipelines = requiredPipelines
                self.forbiddenPipelines = forbiddenPipelines
                self.requiredFiles = requiredFiles
                self.forbiddenFiles = forbiddenFiles
                self.requiredRoutes = requiredRoutes
            }

            enum CodingKeys: String, CodingKey {
                case id
                case requiredPipelines = "required_pipelines"
                case forbiddenPipelines = "forbidden_pipelines"
                case requiredFiles = "required_files"
                case forbiddenFiles = "forbidden_files"
                case requiredRoutes = "required_routes"
            }
        }

        public let parityFixture: String?
        public let performanceGate: String?
        public let performanceProfile: PerformanceProfile?
        public let structureProfile: StructureProfile?

        public init(
            parityFixture: String? = nil,
            performanceGate: String? = nil,
            performanceProfile: PerformanceProfile? = nil,
            structureProfile: StructureProfile? = nil
        ) {
            self.parityFixture = parityFixture
            self.performanceGate = performanceGate
            self.performanceProfile = performanceProfile
            self.structureProfile = structureProfile
        }

        enum CodingKeys: String, CodingKey {
            case parityFixture = "parity_fixture"
            case performanceGate = "performance_gate"
            case performanceProfile = "performance_profile"
            case structureProfile = "structure_profile"
        }
    }

    public let version: Int
    public let packageName: String
    public let modelName: String
    public let sources: [Source]
    public let blocks: SmeltBlockGraph
    public let loop: SmeltLoopSchedule
    public let runtime: RuntimeDescriptor
    public let architectureConfig: SmeltPackageSpecValue
    public let tensors: [TensorMap]
    public let quantization: QuantizationPlan?
    public let sidecars: [Sidecar]
    public let artifacts: [Artifact]
    public let outputFiles: PackageFileSet
    public let tokenizer: Tokenizer?
    public let inference: SmeltInferenceManifest?
    public let decode: DecodePolicy?
    public let validation: Validation

    public init(
        version: Int = currentVersion,
        packageName: String,
        modelName: String,
        sources: [Source],
        blocks: SmeltBlockGraph,
        loop: SmeltLoopSchedule,
        runtime: RuntimeDescriptor,
        architectureConfig: SmeltPackageSpecValue,
        tensors: [TensorMap],
        quantization: QuantizationPlan? = nil,
        sidecars: [Sidecar] = [],
        artifacts: [Artifact],
        outputFiles: PackageFileSet,
        tokenizer: Tokenizer? = nil,
        inference: SmeltInferenceManifest? = nil,
        decode: DecodePolicy? = nil,
        validation: Validation = Validation()
    ) {
        self.version = version
        self.packageName = packageName
        self.modelName = modelName
        self.sources = sources
        self.blocks = blocks
        self.loop = loop
        self.runtime = runtime
        self.architectureConfig = architectureConfig
        self.tensors = tensors
        self.quantization = quantization
        self.sidecars = sidecars
        self.artifacts = artifacts
        self.outputFiles = outputFiles
        self.tokenizer = tokenizer
        self.inference = inference
        self.decode = decode
        self.validation = validation
    }

    enum CodingKeys: String, CodingKey {
        case version
        case packageName = "package_name"
        case modelName = "model_name"
        case sources
        case blocks
        case loop
        case runtime
        case architectureConfig = "architecture_config"
        case tensors
        case quantization
        case sidecars
        case artifacts
        case outputFiles = "output_files"
        case tokenizer
        case inference
        case decode
        case validation
    }

    public static func decode(from data: Data) throws -> SmeltPackageSpec {
        do {
            return try JSONDecoder().decode(SmeltPackageSpec.self, from: data)
        } catch {
            throw SmeltPackageSpecError.malformed("invalid JSON: \(error)")
        }
    }

    public func validate() throws {
        guard version == Self.currentVersion else {
            throw error("unsupported version \(version)")
        }
        try Self.validateName(packageName, field: "package_name")
        try Self.validateNonEmpty(modelName, field: "model_name")
        guard architectureConfig.hasObjectPayload else {
            throw error("architecture_config must be a non-empty object")
        }

        try validateSources()
        try blocks.validate()
        try loop.validate(against: blocks)
        try validateRuntime()
        try validateTensors()
        try validateQuantization()
        try validateArtifacts()
        try validatePolicy()
        try validateValidation()
    }

    // MARK: - Validation

    private func validateSources() throws {
        guard !sources.isEmpty else { throw error("declares no sources") }
        var seen = Set<String>()
        for source in sources {
            try Self.validateName(source.id, field: "source id")
            guard seen.insert(source.id).inserted else {
                throw error("source '\(source.id)' declared twice")
            }
            switch source.kind {
            case .localFile, .localDirectory:
                guard let path = source.path else {
                    throw error("source '\(source.id)' needs a path")
                }
                try Self.validateInputPath(path, field: "source '\(source.id)' path")
                guard source.repo == nil else {
                    throw error("local source '\(source.id)' must not declare repo")
                }
            case .huggingFace:
                guard let repo = source.repo, !repo.isEmpty else {
                    throw error("hugging-face source '\(source.id)' needs a repo")
                }
                guard source.path == nil else {
                    throw error("hugging-face source '\(source.id)' must not declare path")
                }
                if repo.contains("..") || repo.hasPrefix("/") {
                    throw error("hugging-face source '\(source.id)' has unsafe repo '\(repo)'")
                }
            }
        }
    }

    private func validateRuntime() throws {
        try Self.validateName(runtime.architecture, field: "runtime architecture")
        _ = try resolvedRuntimeGraphPolicy()

        guard !runtime.commands.isEmpty else {
            throw error("runtime declares no command capabilities")
        }
        var seenCommands = Set<RuntimeDescriptor.Command>()
        for command in runtime.commands {
            guard seenCommands.insert(command).inserted else {
                throw error("runtime command '\(command.rawValue)' declared twice")
            }
        }

        var routesByBlock: [String: RuntimeDescriptor.BlockRoute] = [:]
        for route in runtime.routes {
            guard routesByBlock[route.block] == nil else {
                throw error("runtime route for block '\(route.block)' declared twice")
            }
            routesByBlock[route.block] = route
        }
        for block in blocks.blocks {
            guard let route = routesByBlock[block.name] else {
                throw error("runtime route missing for block '\(block.name)'")
            }
            guard route.impl == block.impl, route.delivery == block.compiledDelivery else {
                throw error("runtime route for block '\(block.name)' disagrees with graph")
            }
        }
        for route in runtime.routes where !blocks.blocks.contains(where: { $0.name == route.block }) {
            throw error("runtime route references unknown block '\(route.block)'")
        }
    }

    private func validateTensors() throws {
        guard !tensors.isEmpty else { throw error("declares no tensor maps") }
        let sourceIDs = Set(sources.map(\.id))
        let blockNames = Set(blocks.blocks.map(\.name))
        var seenCanonical = Set<String>()
        for tensor in tensors {
            guard sourceIDs.contains(tensor.source) else {
                throw error("tensor '\(tensor.name)' references unknown source '\(tensor.source)'")
            }
            guard blockNames.contains(tensor.block) else {
                throw error("tensor '\(tensor.name)' references unknown block '\(tensor.block)'")
            }
            try Self.validateNonEmpty(tensor.name, field: "tensor name")
            try Self.validateNonEmpty(tensor.canonicalName, field: "tensor canonical_name")
            guard seenCanonical.insert(tensor.canonicalName).inserted else {
                throw error("tensor canonical_name '\(tensor.canonicalName)' declared twice")
            }
            guard !tensor.shape.isEmpty, tensor.shape.allSatisfy({ $0 > 0 }) else {
                throw error("tensor '\(tensor.name)' has invalid shape")
            }
        }
    }

    private func validateQuantization() throws {
        guard let quantization else { return }
        if quantization.format == .u4 || quantization.format == .gptq {
            guard let groupSize = quantization.groupSize, groupSize > 0 else {
                throw error("\(quantization.format.rawValue) quantization needs positive group_size")
            }
        }
        guard let calibration = quantization.calibration else { return }
        guard [.u4, .gptq, .turboQuantH].contains(quantization.format) else {
            throw error(
                "\(quantization.format.rawValue) quantization must not declare calibration"
            )
        }
        let hasCalibrationWork = calibration.corpus != nil
            || !calibration.captureArtifacts.isEmpty
            || !calibration.sideInputs.isEmpty
            || !calibration.stagedPackages.isEmpty
            || !calibration.equalityGates.isEmpty
        guard hasCalibrationWork else {
            throw error("quantization calibration must declare calibration work")
        }

        let sourceIDs = Set(sources.map(\.id))
        if let corpus = calibration.corpus {
            guard sourceIDs.contains(corpus.source) else {
                throw error(
                    "quantization calibration corpus references unknown source "
                        + "'\(corpus.source)'"
                )
            }
            try Self.validateInputPath(corpus.path, field: "quantization calibration corpus path")
            try Self.validateNonEmpty(
                corpus.renderPolicy,
                field: "quantization calibration corpus render_policy"
            )
            if let maxSamples = corpus.maxSamples, maxSamples <= 0 {
                throw error("quantization calibration corpus max_samples must be positive")
            }
            if let maxTokens = corpus.maxTokens, maxTokens <= 0 {
                throw error("quantization calibration corpus max_tokens must be positive")
            }
        }

        try validateCalibrationArtifacts(
            calibration.captureArtifacts,
            kind: "capture artifact",
            packageRelative: true
        )
        try validateCalibrationArtifacts(
            calibration.sideInputs,
            kind: "side input",
            packageRelative: false
        )
        try validateCalibrationArtifacts(
            calibration.stagedPackages,
            kind: "staged package",
            packageRelative: false
        )

        if let resourceBounds = calibration.resourceBounds {
            if let maxSamples = resourceBounds.maxSamples, maxSamples <= 0 {
                throw error("quantization calibration resource max_samples must be positive")
            }
            if let maxTokens = resourceBounds.maxTokens, maxTokens <= 0 {
                throw error("quantization calibration resource max_tokens must be positive")
            }
            if let maxLayers = resourceBounds.maxLayersPerPass, maxLayers <= 0 {
                throw error(
                    "quantization calibration resource max_layers_per_pass must be positive"
                )
            }
            if let maxBytes = resourceBounds.maxBytes, maxBytes == 0 {
                throw error("quantization calibration resource max_bytes must be positive")
            }
        }

        var seenGateIDs = Set<String>()
        for gate in calibration.equalityGates {
            try Self.validateName(gate.id, field: "quantization calibration equality gate id")
            try Self.validateInputPath(
                gate.candidate,
                field: "quantization calibration equality gate candidate"
            )
            try Self.validateInputPath(
                gate.reference,
                field: "quantization calibration equality gate reference"
            )
            try Self.validateName(
                gate.metric,
                field: "quantization calibration equality gate metric"
            )
            guard seenGateIDs.insert(gate.id).inserted else {
                throw error("quantization calibration equality gate '\(gate.id)' declared twice")
            }
        }
    }

    private func validateCalibrationArtifacts(
        _ artifacts: [QuantizationPlan.Calibration.Artifact],
        kind: String,
        packageRelative: Bool
    ) throws {
        var seen = Set<String>()
        for artifact in artifacts {
            try Self.validateName(artifact.id, field: "quantization calibration \(kind) id")
            if packageRelative {
                try Self.validatePackageRelativePath(
                    artifact.path,
                    field: "quantization calibration \(kind) path"
                )
            } else {
                try Self.validateInputPath(
                    artifact.path,
                    field: "quantization calibration \(kind) path"
                )
            }
            try Self.validateNonEmpty(artifact.role, field: "quantization calibration \(kind) role")
            guard seen.insert(artifact.id).inserted else {
                throw error("quantization calibration \(kind) '\(artifact.id)' declared twice")
            }
        }
    }

    private func validateArtifacts() throws {
        guard !artifacts.isEmpty else { throw error("declares no artifacts") }
        var ids = Set<String>()
        var paths = Set<String>()
        var sidecarIDs = Set<String>()
        var sidecarPaths = Set<String>()
        var packagePaths = Set<String>()
        var fileLikePackagePaths = Set<String>()
        for artifact in artifacts {
            try Self.validateName(artifact.id, field: "artifact id")
            try Self.validateNonEmpty(artifact.role, field: "artifact role")
            guard ids.insert(artifact.id).inserted else {
                throw error("artifact '\(artifact.id)' declared twice")
            }
            try Self.validatePackageRelativePath(artifact.path, field: "artifact '\(artifact.id)' path")
            guard paths.insert(artifact.path).inserted else {
                throw error("package file '\(artifact.path)' declared twice")
            }
            packagePaths.insert(artifact.path)
            fileLikePackagePaths.insert(artifact.path)
        }
        for sidecar in sidecars {
            try Self.validateName(sidecar.id, field: "sidecar id")
            try Self.validateName(sidecar.kind, field: "sidecar kind")
            guard SmeltPackageSidecarKind.known.contains(sidecar.kind) else {
                throw error("sidecar '\(sidecar.id)' has unknown kind '\(sidecar.kind)'")
            }
            guard sidecarIDs.insert(sidecar.id).inserted else {
                throw error("sidecar '\(sidecar.id)' declared twice")
            }
            try Self.validatePackageRelativePath(sidecar.path, field: "sidecar '\(sidecar.id)' path")
            guard paths.insert(sidecar.path).inserted else {
                throw error("package file '\(sidecar.path)' declared twice")
            }
            sidecarPaths.insert(sidecar.path)
            packagePaths.insert(sidecar.path)
        }
        try validateSidecarAssemblyPolicy()
        try Self.validatePackageRelativePath(outputFiles.manifest, field: "output manifest")
        guard outputFiles.manifest == "manifest.json" else {
            throw error("output manifest must be manifest.json")
        }
        packagePaths.insert(outputFiles.manifest)
        fileLikePackagePaths.insert(outputFiles.manifest)
        var outputSeen = Set<String>()
        for file in outputFiles.files {
            try Self.validatePackageRelativePath(file, field: "output file")
            guard outputSeen.insert(file).inserted else {
                throw error("output file '\(file)' declared twice")
            }
            packagePaths.insert(file)
            if !sidecarPaths.contains(file) {
                fileLikePackagePaths.insert(file)
            }
        }
        guard outputSeen.contains(outputFiles.manifest) else {
            throw error("output files must include \(outputFiles.manifest)")
        }
        for path in paths where !outputSeen.contains(path) {
            throw error("package artifact '\(path)' is missing from output files")
        }
        if let tokenizer {
            for file in tokenizer.files where !outputSeen.contains(file) {
                throw error("tokenizer file '\(file)' is missing from output files")
            }
            for file in tokenizer.files {
                packagePaths.insert(file)
                fileLikePackagePaths.insert(file)
            }
        }
        if let calibration = quantization?.calibration {
            for artifact in calibration.captureArtifacts {
                try Self.validatePackageRelativePath(
                    artifact.path,
                    field: "quantization calibration capture artifact path"
                )
                packagePaths.insert(artifact.path)
                fileLikePackagePaths.insert(artifact.path)
            }
        }
        try validateNoFileDirectoryPackagePathConflicts(
            fileLikePackagePaths: fileLikePackagePaths,
            sidecarPaths: sidecarPaths,
            packagePaths: packagePaths
        )
    }

    private func validateSidecarAssemblyPolicy() throws {
        switch try resolvedRuntimeGraphPolicy() {
        case .sidecarTextToCodecAudio:
            try validateGraphSidecarAssemblyPolicy()
        case .textGeneration, .codecAudio:
            return
        }
    }

    private func validateGraphSidecarAssemblyPolicy() throws {
        let sidecarBlocks = blocks.blocks.filter {
            $0.compiledDelivery == .compiledSidecar
                || $0.compiledDelivery == .internalSidecar
        }
        guard sidecars.count == sidecarBlocks.count else {
            let blockNames = sidecarBlocks.map(\.name).sorted().joined(separator: ",")
            throw error(
                "graph declares \(sidecarBlocks.count) sidecar-delivered block(s) "
                    + "(\(blockNames)); spec declares \(sidecars.count) sidecar(s)"
            )
        }
    }

    private func validatePolicy() throws {
        switch try resolvedRuntimeGraphPolicy() {
        case .textGeneration:
            guard let tokenizer else { throw error("text generation spec needs tokenizer policy") }
            try validateTokenizer(tokenizer)
            guard let inference else { throw error("text generation spec needs inference policy") }
            guard !inference.eosTokens.isEmpty else {
                throw error("text generation inference policy needs eos_tokens")
            }
            guard let chatTemplate = inference.chatTemplate, !chatTemplate.isEmpty else {
                throw error("text generation inference policy needs chat_template")
            }
            guard SmeltPromptTemplateName.isKnownPromptTemplate(chatTemplate) else {
                throw error(
                    "text generation inference policy chat_template "
                        + "'\(chatTemplate)' is not supported "
                        + "(available: \(SmeltPromptTemplateName.availablePromptTemplates))"
                )
            }
            guard inference.thinkingPolicy != nil else {
                throw error("text generation inference policy needs thinking_policy")
            }
            if let codec = inference.toolTranscriptCodec,
               !SmeltToolTranscriptCodecName.isKnown(codec) {
                throw error(
                    "text generation inference policy tool_transcript_codec "
                        + "'\(codec)' is not supported (available: "
                        + SmeltToolTranscriptCodecName.availableCodecs + ")"
                )
            }
            try validateDecode(requiredFor: "text generation")
        case .sidecarTextToCodecAudio:
            guard let tokenizer else { throw error("codec audio spec needs tokenizer policy") }
            try validateTokenizer(tokenizer)
            guard let inference, !inference.eosTokens.isEmpty else {
                throw error("codec audio spec needs eos_tokens")
            }
            try validateDecode(requiredFor: "codec audio")
        case .codecAudio:
            guard tokenizer == nil else {
                throw error("codec audio block spec must not declare tokenizer policy")
            }
            guard inference == nil else {
                throw error("codec audio block spec must not declare inference policy")
            }
            guard decode == nil else {
                throw error("codec audio block spec must not declare decode policy")
            }
        }
    }

    private func resolvedRuntimeGraphPolicy() throws -> SmeltRuntimeGraphPolicy {
        do {
            return try SmeltRuntimeGraphPolicy.resolve(blocks: blocks)
        } catch let policyError as SmeltRuntimeGraphPolicy.ResolveError {
            throw error(policyError.description)
        } catch let resolveError {
            throw error("runtime graph policy could not be resolved: \(resolveError)")
        }
    }

    private func validateValidation() throws {
        if let parityFixture = validation.parityFixture {
            try Self.validateNonEmpty(parityFixture, field: "validation parity_fixture")
        }
        if let performanceGate = validation.performanceGate {
            try validatePerformanceGate(performanceGate, field: "validation performance_gate")
        }
        if validation.performanceGate != nil, validation.performanceProfile == nil {
            throw error("validation performance_gate needs performance_profile")
        }
        if let profile = validation.performanceProfile {
            try validatePerformanceGate(profile.gate, field: "validation performance_profile gate")
            if let performanceGate = validation.performanceGate, performanceGate != profile.gate {
                throw error(
                    "validation performance_profile gate '\(profile.gate)' disagrees with "
                        + "performance_gate '\(performanceGate)'"
                )
            }
            guard runtime.commands.contains(profile.command) else {
                throw error(
                    "validation performance_profile command '\(profile.command.rawValue)' "
                        + "is not declared by runtime"
                )
            }
            for label in profile.requiredTraceLabels {
                try Self.validateNonEmpty(label, field: "validation required trace label")
            }
            for metric in profile.requiredOutputMetrics {
                try Self.validateName(metric, field: "validation required output metric")
            }
            var seenMinBounds = Set<String>()
            for bound in profile.minBounds {
                try Self.validateName(bound.metric, field: "validation min-bound metric")
                try Self.validateNonEmpty(bound.unit, field: "validation min-bound unit")
                guard bound.min.isFinite, bound.min > 0 else {
                    throw error("validation min-bound '\(bound.metric)' must be positive and finite")
                }
                guard seenMinBounds.insert(bound.metric).inserted else {
                    throw error("validation min-bound '\(bound.metric)' declared twice")
                }
            }
            var seenMaxBounds = Set<String>()
            for bound in profile.maxBounds {
                try Self.validateName(bound.metric, field: "validation max-bound metric")
                try Self.validateNonEmpty(bound.unit, field: "validation max-bound unit")
                guard bound.max.isFinite, bound.max > 0 else {
                    throw error("validation max-bound '\(bound.metric)' must be positive and finite")
                }
                guard seenMaxBounds.insert(bound.metric).inserted else {
                    throw error("validation max-bound '\(bound.metric)' declared twice")
                }
            }
            try validateCanonicalPerformanceProfile(profile)
        }
        if let structureProfile = validation.structureProfile {
            try validateStructureProfile(structureProfile)
        }
    }

    private func validatePerformanceGate(_ gate: String, field: String) throws {
        try Self.validateName(gate, field: field)
        guard SmeltPackagePerformanceGateID.known.contains(gate) else {
            throw error("unknown \(field) '\(gate)'")
        }
    }

    private func validateCanonicalPerformanceProfile(
        _ profile: Validation.PerformanceProfile
    ) throws {
        let canonical = SmeltPackagePerformanceProfiles.profile(
            for: profile.gate,
            modelName: modelName
        )
        guard profile.command == canonical.command else {
            throw error(
                "validation performance_profile command '\(profile.command.rawValue)' "
                    + "is not canonical '\(canonical.command.rawValue)' for gate '\(profile.gate)'"
            )
        }
        try validateRequiredStrings(
            profile.requiredTraceLabels,
            include: canonical.requiredTraceLabels,
            field: "trace label",
            gate: profile.gate
        )
        try validateRequiredStrings(
            profile.requiredOutputMetrics,
            include: canonical.requiredOutputMetrics,
            field: "output metric",
            gate: profile.gate
        )
        try validateCanonicalMinBounds(
            profile.minBounds,
            include: canonical.minBounds,
            gate: profile.gate
        )
        try validateCanonicalMaxBounds(
            profile.maxBounds,
            include: canonical.maxBounds,
            gate: profile.gate
        )
    }

    private func validateRequiredStrings(
        _ actual: [String],
        include expected: [String],
        field: String,
        gate: String
    ) throws {
        let actualSet = Set(actual)
        for value in expected where !actualSet.contains(value) {
            throw error(
                "validation performance_profile for gate '\(gate)' is missing canonical "
                    + "\(field) '\(value)'"
            )
        }
    }

    private func validateCanonicalMinBounds(
        _ actual: [Validation.PerformanceProfile.MinBound],
        include expected: [Validation.PerformanceProfile.MinBound],
        gate: String
    ) throws {
        for bound in expected {
            guard actual.contains(where: {
                $0.metric == bound.metric && $0.unit == bound.unit && $0.min >= bound.min
            }) else {
                throw error(
                    "validation performance_profile for gate '\(gate)' is missing canonical "
                        + "min-bound '\(bound.metric)'"
                )
            }
        }
    }

    private func validateCanonicalMaxBounds(
        _ actual: [Validation.PerformanceProfile.Bound],
        include expected: [Validation.PerformanceProfile.Bound],
        gate: String
    ) throws {
        for bound in expected {
            guard actual.contains(where: {
                $0.metric == bound.metric && $0.unit == bound.unit && $0.max <= bound.max
            }) else {
                throw error(
                    "validation performance_profile for gate '\(gate)' is missing canonical "
                        + "max-bound '\(bound.metric)'"
                )
            }
        }
    }

    private func validateStructureProfile(_ profile: Validation.StructureProfile) throws {
        try Self.validateName(profile.id, field: "validation structure_profile id")
        guard SmeltPackageStructureProfileID.known.contains(profile.id) else {
            throw error("unknown validation structure_profile id '\(profile.id)'")
        }
        let hasEvidence = !profile.requiredPipelines.isEmpty
            || !profile.forbiddenPipelines.isEmpty
            || !profile.requiredFiles.isEmpty
            || !profile.forbiddenFiles.isEmpty
            || !profile.requiredRoutes.isEmpty
        guard hasEvidence else {
            throw error("validation structure_profile '\(profile.id)' declares no evidence")
        }
        try validateUniqueStrings(
            profile.requiredPipelines,
            field: "validation structure_profile required pipeline",
            nameLike: true
        )
        try validateUniqueStrings(
            profile.forbiddenPipelines,
            field: "validation structure_profile forbidden pipeline",
            nameLike: true
        )
        try validateUniqueStrings(
            profile.requiredRoutes,
            field: "validation structure_profile required route",
            nameLike: false
        )
        try validateUniquePackagePaths(
            profile.requiredFiles,
            field: "validation structure_profile required file"
        )
        try validateUniquePackagePaths(
            profile.forbiddenFiles,
            field: "validation structure_profile forbidden file"
        )
    }

    private func validateUniqueStrings(
        _ values: [String],
        field: String,
        nameLike: Bool
    ) throws {
        var seen = Set<String>()
        for value in values {
            if nameLike {
                try Self.validateName(value, field: field)
            } else {
                try Self.validateNonEmpty(value, field: field)
            }
            guard seen.insert(value).inserted else {
                throw error("\(field) '\(value)' declared twice")
            }
        }
    }

    private func validateUniquePackagePaths(_ paths: [String], field: String) throws {
        var seen = Set<String>()
        for path in paths {
            try Self.validatePackageRelativePath(path, field: field)
            guard seen.insert(path).inserted else {
                throw error("\(field) '\(path)' declared twice")
            }
        }
    }

    private func validateNoFileDirectoryPackagePathConflicts(
        fileLikePackagePaths: Set<String>,
        sidecarPaths: Set<String>,
        packagePaths: Set<String>
    ) throws {
        if let conflict = fileLikePackagePaths.intersection(sidecarPaths).sorted().first {
            throw error(
                "package file '\(conflict)' conflicts with sidecar directory '\(conflict)'"
            )
        }

        let sortedPackagePaths = packagePaths.sorted()
        for filePath in fileLikePackagePaths.sorted() {
            let childPrefix = filePath + "/"
            if let nested = sortedPackagePaths.first(where: { $0.hasPrefix(childPrefix) }) {
                throw error(
                    "package file '\(filePath)' conflicts with nested package path '\(nested)'"
                )
            }
        }
    }

    private func validateTokenizer(_ tokenizer: Tokenizer) throws {
        try Self.validateNonEmpty(tokenizer.format, field: "tokenizer format")
        guard !tokenizer.files.isEmpty else {
            throw error("tokenizer declares no files")
        }
        for file in tokenizer.files {
            try Self.validatePackageRelativePath(file, field: "tokenizer file")
        }
    }

    private func validateDecode(requiredFor name: String) throws {
        guard let decode else { throw error("\(name) spec needs decode policy") }
        try validateSampler(decode.sampler)
        if let subSampler = decode.subSampler {
            try validateSampler(subSampler)
        }
        if let maxSteps = decode.maxSteps, maxSteps <= 0 {
            throw error("\(name) decode policy has non-positive max_steps")
        }
    }

    private func validateSampler(_ sampler: DecodePolicy.Sampler) throws {
        if let temperature = sampler.temperature, temperature <= 0 {
            throw error("sampler temperature must be positive")
        }
        if let topK = sampler.topK, topK <= 0 {
            throw error("sampler top_k must be positive")
        }
        if let topP = sampler.topP, topP <= 0 || topP > 1 {
            throw error("sampler top_p must be in (0, 1]")
        }
    }

    private func error(_ message: String) -> SmeltPackageSpecError {
        .malformed(message)
    }

    private static func validateNonEmpty(_ value: String, field: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SmeltPackageSpecError.malformed("\(field) must be non-empty")
        }
    }

    private static func validateName(_ value: String, field: String) throws {
        try validateNonEmpty(value, field: field)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        guard value.unicodeScalars.allSatisfy({ allowed.contains($0) }),
              !value.contains(".."),
              !value.hasPrefix("."),
              !value.hasSuffix(".")
        else {
            throw SmeltPackageSpecError.malformed(
                "\(field) '\(value)' must use only letters, digits, '.', '_', '-' "
                    + "and must not be dot-relative"
            )
        }
    }

    private static func validateInputPath(_ path: String, field: String) throws {
        try validateNonEmpty(path, field: field)
        guard !path.contains("\0"), !path.split(separator: "/").contains("..") else {
            throw SmeltPackageSpecError.malformed("\(field) is unsafe: \(path)")
        }
    }

    private static func validatePackageRelativePath(_ path: String, field: String) throws {
        try validateNonEmpty(path, field: field)
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.hasPrefix("/"),
              !path.hasPrefix("~"),
              !path.contains("\\"),
              !path.contains("\0"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw SmeltPackageSpecError.malformed("\(field) must be package-relative: \(path)")
        }
    }
}
