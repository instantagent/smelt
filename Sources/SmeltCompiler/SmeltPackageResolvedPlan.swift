// SmeltPackageResolvedPlan — CAM's pure plan between config validation and writes.
//
// A package writer should receive this object, not a loose family-specific pile
// of inferred defaults. Resolving a CAM spec validates the contract and produces
// deterministic package inventory, runtime routing, tensor, bake, and policy
// summaries without creating directories or touching package files.

import SmeltSchema

public struct SmeltPackageResolvedPlan: Sendable, Equatable {

    public struct SourcePlan: Sendable, Equatable {
        public let id: String
        public let kind: SmeltPackageSpec.Source.Kind
        public let locator: String
        public let revision: String?
    }

    public struct PackageFile: Sendable, Equatable {
        public let path: String
        public let roles: [String]
    }

    public struct RuntimePlan: Sendable, Equatable {
        public struct Route: Sendable, Equatable {
            public let block: String
            public let impl: SmeltBlockGraph.Impl
            public let delivery: SmeltBlockGraph.CompiledDelivery?
            public let signature: String
        }

        public let architecture: String
        public let commands: [SmeltPackageSpec.RuntimeDescriptor.Command]
        public let routes: [Route]
        public let setupPhases: [String]
        public let perStepPhases: [String]
        public let emission: String
    }

    public struct TensorPlan: Sendable, Equatable {
        public let canonicalName: String
        public let source: String
        public let sourceName: String
        public let block: String
        public let sourceDType: SmeltPackageSpec.TensorDType
        public let storedDType: SmeltPackageSpec.TensorDType
        public let shape: [Int]
        public let required: Bool
    }

    public struct QuantizationPlan: Sendable, Equatable {
        public struct CalibrationPlan: Sendable, Equatable {
            public struct Corpus: Sendable, Equatable {
                public let source: String
                public let path: String
                public let renderPolicy: String
                public let maxSamples: Int?
                public let maxTokens: Int?
            }

            public struct Artifact: Sendable, Equatable {
                public let id: String
                public let path: String
                public let role: String
            }

            public struct ResourceBounds: Sendable, Equatable {
                public let maxSamples: Int?
                public let maxTokens: Int?
                public let maxLayersPerPass: Int?
                public let maxBytes: UInt64?
            }

            public struct EqualityGate: Sendable, Equatable {
                public let id: String
                public let candidate: String
                public let reference: String
                public let metric: String
            }

            public let corpus: Corpus?
            public let captureArtifacts: [Artifact]
            public let sideInputs: [Artifact]
            public let stagedPackages: [Artifact]
            public let resourceBounds: ResourceBounds?
            public let equalityGates: [EqualityGate]
        }

        public let format: SmeltPackageSpec.TensorDType
        public let groupSize: Int?
        public let calibration: CalibrationPlan?
    }

    public struct BakePlan: Sendable, Equatable {
        public struct SealedComponent: Sendable, Equatable {
            public let kind: SmeltBakeManifest.Component
            public let required: [String]
            public let perf: [String]
        }

        public let path: String
        public let version: Int
        public let sealed: [SealedComponent]
    }

    public struct TokenizerPlan: Sendable, Equatable {
        public let format: String
        public let files: [String]
    }

    public struct InferencePlan: Sendable, Equatable {
        public let maxTokens: Int
        public let eosTokens: [Int32]
        public let thinkToken: Int32?
        public let thinkEndToken: Int32?
        public let thinkSkipSuffix: Int32?
        public let chatTemplate: String?
        public let thinkingPolicy: SmeltThinkingPolicy?
        public let toolTranscriptCodec: String?
        public let promptStateRestoreMode: SmeltPromptStateRestoreMode?
    }

    public struct SamplerPlan: Sendable, Equatable {
        public let mode: SmeltPackageSpec.DecodePolicy.SamplerMode
        public let temperature: Double?
        public let topK: Int?
        public let topP: Double?
    }

    public struct DecodePlan: Sendable, Equatable {
        public let sampler: SamplerPlan
        public let subSampler: SamplerPlan?
        public let maxSteps: Int?
        public let durationSeconds: Double?
    }

    public struct PerformanceProfilePlan: Sendable, Equatable {
        public struct MinBound: Sendable, Equatable {
            public let metric: String
            public let min: Double
            public let unit: String
        }

        public struct Bound: Sendable, Equatable {
            public let metric: String
            public let max: Double
            public let unit: String
        }

        public let gate: String
        public let command: SmeltPackageSpec.RuntimeDescriptor.Command
        public let requiredTraceLabels: [String]
        public let requiredOutputMetrics: [String]
        public let minBounds: [MinBound]
        public let maxBounds: [Bound]
    }

    public struct StructureProfilePlan: Sendable, Equatable {
        public let id: String
        public let requiredPipelines: [String]
        public let forbiddenPipelines: [String]
        public let requiredFiles: [String]
        public let forbiddenFiles: [String]
        public let requiredRoutes: [String]
    }

    public struct PolicyPlan: Sendable, Equatable {
        public typealias Mode = SmeltRuntimeGraphPolicy

        public let mode: Mode
        public let tokenizer: TokenizerPlan?
        public let inference: InferencePlan?
        public let decode: DecodePlan?
    }

    public struct Signature: Sendable, Equatable {
        public let lines: [String]
    }

    public let version: Int
    public let packageName: String
    public let modelName: String
    public let sources: [SourcePlan]
    public let runtime: RuntimePlan
    public let architectureConfigSignature: String
    public let tensors: [TensorPlan]
    public let quantization: QuantizationPlan?
    public let packageFiles: [PackageFile]
    public let bake: BakePlan?
    public let policy: PolicyPlan
    public let validationParityFixture: String?
    public let validationPerformanceGate: String?
    public let validationPerformanceProfile: PerformanceProfilePlan?
    public let validationStructureProfile: StructureProfilePlan?

    public static func resolve(_ spec: SmeltPackageSpec) throws -> SmeltPackageResolvedPlan {
        try spec.validate()

        let runtime = RuntimePlan(
            architecture: spec.runtime.architecture,
            commands: spec.runtime.commands.sorted { $0.rawValue < $1.rawValue },
            routes: spec.blocks.blocks.map { block in
                let route = spec.runtime.routes.first { $0.block == block.name }!
                return RuntimePlan.Route(
                    block: route.block,
                    impl: route.impl,
                    delivery: route.delivery,
                    signature: route.signature
                )
            },
            setupPhases: spec.loop.setupSignatures,
            perStepPhases: spec.loop.perStepSignatures,
            emission: spec.loop.emissionSignature
        )

        return SmeltPackageResolvedPlan(
            version: spec.version,
            packageName: spec.packageName,
            modelName: spec.modelName,
            sources: sourcePlans(from: spec.sources),
            runtime: runtime,
            architectureConfigSignature: valueSignature(spec.architectureConfig),
            tensors: tensorPlans(from: spec.tensors),
            quantization: spec.quantization.map {
                QuantizationPlan(
                    format: $0.format,
                    groupSize: $0.groupSize,
                    calibration: calibrationPlan($0.calibration)
                )
            },
            packageFiles: packageFiles(from: spec),
            bake: bakePlan(spec.outputFiles.bakeManifest),
            policy: try policyPlan(from: spec),
            validationParityFixture: spec.validation.parityFixture,
            validationPerformanceGate: spec.validation.performanceGate,
            validationPerformanceProfile: performanceProfilePlan(
                spec.validation.performanceProfile
            ),
            validationStructureProfile: structureProfilePlan(
                spec.validation.structureProfile
            )
        )
    }

    public var signature: Signature {
        var lines: [String] = [
            "version:\(version)",
            "package:\(packageName)",
            "model:\(modelName)",
            "architecture:\(runtime.architecture)",
            "architecture-config:\(architectureConfigSignature)",
            "emission:\(runtime.emission)",
        ]

        lines += sources.map {
            "source:\($0.id):\($0.kind.rawValue):\($0.locator):\($0.revision ?? "none")"
        }
        lines += runtime.commands.map { "command:\($0.rawValue)" }
        lines += runtime.routes.map { "route:\($0.signature)" }
        lines += runtime.setupPhases.map { "setup:\($0)" }
        lines += runtime.perStepPhases.map { "per-step:\($0)" }
        lines += packageFiles.map { "file:\($0.path):\($0.roles.joined(separator: ","))" }
        lines += tensors.map {
            "tensor:\($0.canonicalName):\($0.source):\($0.sourceName):\($0.block):"
                + "\($0.sourceDType.rawValue)->\($0.storedDType.rawValue):"
                + "\($0.shape.map(String.init).joined(separator: "x")):\($0.required)"
        }

        if let quantization {
            lines.append(
                "quantization:\(quantization.format.rawValue):"
                    + "\(quantization.groupSize.map(String.init) ?? "none")"
            )
            if let calibration = quantization.calibration {
                lines += calibration.signatureLines
            }
        } else {
            lines.append("quantization:none")
        }

        if let bake {
            lines.append("bake:\(bake.path):v\(bake.version)")
            lines += bake.sealed.map {
                "bake-component:\($0.kind.rawValue):"
                    + "required=\($0.required.joined(separator: ",")):"
                    + "perf=\($0.perf.joined(separator: ","))"
            }
        }

        lines += policy.signatureLines

        if let validationParityFixture {
            lines.append("validation:parity:\(validationParityFixture)")
        }
        if let validationPerformanceGate {
            lines.append("validation:performance:\(validationPerformanceGate)")
        }
        if let validationPerformanceProfile {
            lines.append(
                "validation-performance-profile:"
                    + "\(validationPerformanceProfile.gate):"
                    + "\(validationPerformanceProfile.command.rawValue)"
            )
            lines += validationPerformanceProfile.requiredTraceLabels.map {
                "validation-performance-trace-label:\($0)"
            }
            lines += validationPerformanceProfile.requiredOutputMetrics.map {
                "validation-performance-output-metric:\($0)"
            }
            lines += validationPerformanceProfile.minBounds.map {
                "validation-performance-min:\($0.metric):\($0.min):\($0.unit)"
            }
            lines += validationPerformanceProfile.maxBounds.map {
                "validation-performance-max:\($0.metric):\($0.max):\($0.unit)"
            }
        }
        if let validationStructureProfile {
            lines += validationStructureProfile.signatureLines
        }

        return Signature(lines: lines)
    }

    private static func sourcePlans(
        from sources: [SmeltPackageSpec.Source]
    ) -> [SourcePlan] {
        sources
            .map {
                SourcePlan(
                    id: $0.id,
                    kind: $0.kind,
                    locator: $0.path ?? $0.repo ?? "",
                    revision: $0.revision
                )
            }
            .sorted { $0.id < $1.id }
    }

    private static func tensorPlans(
        from tensors: [SmeltPackageSpec.TensorMap]
    ) -> [TensorPlan] {
        tensors
            .map {
                TensorPlan(
                    canonicalName: $0.canonicalName,
                    source: $0.source,
                    sourceName: $0.name,
                    block: $0.block,
                    sourceDType: $0.sourceDType,
                    storedDType: $0.storedDType,
                    shape: $0.shape,
                    required: $0.required
                )
            }
            .sorted { $0.canonicalName < $1.canonicalName }
    }

    private static func calibrationPlan(
        _ calibration: SmeltPackageSpec.QuantizationPlan.Calibration?
    ) -> QuantizationPlan.CalibrationPlan? {
        calibration.map { calibration in
            QuantizationPlan.CalibrationPlan(
                corpus: calibration.corpus.map {
                    QuantizationPlan.CalibrationPlan.Corpus(
                        source: $0.source,
                        path: $0.path,
                        renderPolicy: $0.renderPolicy,
                        maxSamples: $0.maxSamples,
                        maxTokens: $0.maxTokens
                    )
                },
                captureArtifacts: calibration.captureArtifacts
                    .map {
                        QuantizationPlan.CalibrationPlan.Artifact(
                            id: $0.id,
                            path: $0.path,
                            role: $0.role
                        )
                    }
                    .sorted { $0.id < $1.id },
                sideInputs: calibration.sideInputs
                    .map {
                        QuantizationPlan.CalibrationPlan.Artifact(
                            id: $0.id,
                            path: $0.path,
                            role: $0.role
                        )
                    }
                    .sorted { $0.id < $1.id },
                stagedPackages: calibration.stagedPackages
                    .map {
                        QuantizationPlan.CalibrationPlan.Artifact(
                            id: $0.id,
                            path: $0.path,
                            role: $0.role
                        )
                    }
                    .sorted { $0.id < $1.id },
                resourceBounds: calibration.resourceBounds.map {
                    QuantizationPlan.CalibrationPlan.ResourceBounds(
                        maxSamples: $0.maxSamples,
                        maxTokens: $0.maxTokens,
                        maxLayersPerPass: $0.maxLayersPerPass,
                        maxBytes: $0.maxBytes
                    )
                },
                equalityGates: calibration.equalityGates
                    .map {
                        QuantizationPlan.CalibrationPlan.EqualityGate(
                            id: $0.id,
                            candidate: $0.candidate,
                            reference: $0.reference,
                            metric: $0.metric
                        )
                    }
                    .sorted { $0.id < $1.id }
            )
        }
    }

    private static func packageFiles(from spec: SmeltPackageSpec) -> [PackageFile] {
        var rolesByPath: [String: Set<String>] = [:]

        func add(_ role: String, to path: String) {
            rolesByPath[path, default: []].insert(role)
        }

        for file in spec.outputFiles.files {
            add("declared-output", to: file)
        }
        add("manifest", to: spec.outputFiles.manifest)

        for artifact in spec.artifacts {
            add("artifact:\(artifact.id):\(artifact.role)", to: artifact.path)
        }
        if let calibration = spec.quantization?.calibration {
            for artifact in calibration.captureArtifacts {
                add(
                    "quant-calibration:capture:\(artifact.id):\(artifact.role)",
                    to: artifact.path
                )
            }
        }
        for sidecar in spec.sidecars {
            add("sidecar:\(sidecar.id):\(sidecar.kind)", to: sidecar.path)
        }
        if let tokenizer = spec.tokenizer {
            for file in tokenizer.files {
                add("tokenizer:\(tokenizer.format)", to: file)
            }
        }
        if let bake = spec.outputFiles.bakeManifest {
            add("bake-manifest", to: SmeltBakeManifest.fileName)
            for sealed in bake.sealed {
                for file in sealed.required {
                    add("bake:\(sealed.kind.rawValue):required", to: file)
                }
                for file in sealed.perf {
                    add("bake:\(sealed.kind.rawValue):perf", to: file)
                }
            }
        }

        return rolesByPath.keys.sorted().map {
            PackageFile(path: $0, roles: rolesByPath[$0]!.sorted())
        }
    }

    private static func bakePlan(_ bake: SmeltBakeManifest?) -> BakePlan? {
        bake.map {
            BakePlan(
                path: SmeltBakeManifest.fileName,
                version: $0.version,
                sealed: $0.sealed
                    .map {
                        BakePlan.SealedComponent(
                            kind: $0.kind,
                            required: $0.required.sorted(),
                            perf: $0.perf.sorted()
                        )
                    }
                    .sorted { $0.kind.rawValue < $1.kind.rawValue }
            )
        }
    }

    private static func performanceProfilePlan(
        _ profile: SmeltPackageSpec.Validation.PerformanceProfile?
    ) -> PerformanceProfilePlan? {
        profile.map {
            PerformanceProfilePlan(
                gate: $0.gate,
                command: $0.command,
                requiredTraceLabels: $0.requiredTraceLabels.sorted(),
                requiredOutputMetrics: $0.requiredOutputMetrics.sorted(),
                minBounds: $0.minBounds
                    .map {
                        PerformanceProfilePlan.MinBound(
                            metric: $0.metric,
                            min: $0.min,
                            unit: $0.unit
                        )
                    }
                    .sorted { $0.metric < $1.metric },
                maxBounds: $0.maxBounds
                    .map {
                        PerformanceProfilePlan.Bound(
                            metric: $0.metric,
                            max: $0.max,
                            unit: $0.unit
                        )
                    }
                    .sorted { $0.metric < $1.metric }
            )
        }
    }

    private static func structureProfilePlan(
        _ profile: SmeltPackageSpec.Validation.StructureProfile?
    ) -> StructureProfilePlan? {
        profile.map {
            StructureProfilePlan(
                id: $0.id,
                requiredPipelines: $0.requiredPipelines.sorted(),
                forbiddenPipelines: $0.forbiddenPipelines.sorted(),
                requiredFiles: $0.requiredFiles.sorted(),
                forbiddenFiles: $0.forbiddenFiles.sorted(),
                requiredRoutes: $0.requiredRoutes.sorted()
            )
        }
    }

    private static func policyPlan(from spec: SmeltPackageSpec) throws -> PolicyPlan {
        let tokenizer = spec.tokenizer.map {
            TokenizerPlan(format: $0.format, files: $0.files.sorted())
        }
        let inference = spec.inference.map {
            InferencePlan(
                maxTokens: $0.maxTokens,
                eosTokens: $0.eosTokens,
                thinkToken: $0.thinkToken,
                thinkEndToken: $0.thinkEndToken,
                thinkSkipSuffix: $0.thinkSkipSuffix,
                chatTemplate: $0.chatTemplate,
                thinkingPolicy: $0.thinkingPolicy,
                toolTranscriptCodec: $0.toolTranscriptCodec,
                promptStateRestoreMode: $0.promptStateRestoreMode
            )
        }
        let decode = spec.decode.map {
            DecodePlan(
                sampler: SamplerPlan(
                    mode: $0.sampler.mode,
                    temperature: $0.sampler.temperature,
                    topK: $0.sampler.topK,
                    topP: $0.sampler.topP
                ),
                subSampler: $0.subSampler.map {
                    SamplerPlan(
                        mode: $0.mode,
                        temperature: $0.temperature,
                        topK: $0.topK,
                        topP: $0.topP
                    )
                },
                maxSteps: $0.maxSteps,
                durationSeconds: $0.durationSeconds
            )
        }
        return PolicyPlan(
            mode: try graphPolicyMode(for: spec.blocks),
            tokenizer: tokenizer,
            inference: inference,
            decode: decode
        )
    }

    private static func graphPolicyMode(
        for blocks: SmeltBlockGraph
    ) throws -> PolicyPlan.Mode {
        let policy: SmeltRuntimeGraphPolicy
        do {
            policy = try SmeltRuntimeGraphPolicy.resolve(blocks: blocks)
        } catch let policyError as SmeltRuntimeGraphPolicy.ResolveError {
            throw SmeltPackageSpecError.malformed(
                policyError.description
            )
        } catch {
            throw SmeltPackageSpecError.malformed(
                "runtime graph policy could not be resolved: \(error)"
            )
        }
        return policy
    }
}

private extension SmeltPackageResolvedPlan.QuantizationPlan.CalibrationPlan {
    var signatureLines: [String] {
        var lines = ["quantization-calibration"]
        if let corpus {
            lines.append(
                "quantization-calibration-corpus:"
                    + "\(corpus.source):\(corpus.path):\(corpus.renderPolicy):"
                    + "\(optional(corpus.maxSamples)):\(optional(corpus.maxTokens))"
            )
        }
        lines += captureArtifacts.map {
            "quantization-calibration-capture:\($0.id):\($0.path):\($0.role)"
        }
        lines += sideInputs.map {
            "quantization-calibration-side-input:\($0.id):\($0.path):\($0.role)"
        }
        lines += stagedPackages.map {
            "quantization-calibration-staged-package:\($0.id):\($0.path):\($0.role)"
        }
        if let resourceBounds {
            lines.append(
                "quantization-calibration-resources:"
                    + "\(optional(resourceBounds.maxSamples)):"
                    + "\(optional(resourceBounds.maxTokens)):"
                    + "\(optional(resourceBounds.maxLayersPerPass)):"
                    + "\(optional(resourceBounds.maxBytes))"
            )
        }
        lines += equalityGates.map {
            "quantization-calibration-equality:"
                + "\($0.id):\($0.candidate):\($0.reference):\($0.metric)"
        }
        return lines
    }
}

private extension SmeltPackageResolvedPlan.StructureProfilePlan {
    var signatureLines: [String] {
        var lines = ["validation-structure-profile:\(id)"]
        lines += requiredPipelines.map { "validation-structure-required-pipeline:\($0)" }
        lines += forbiddenPipelines.map { "validation-structure-forbidden-pipeline:\($0)" }
        lines += requiredFiles.map { "validation-structure-required-file:\($0)" }
        lines += forbiddenFiles.map { "validation-structure-forbidden-file:\($0)" }
        lines += requiredRoutes.map { "validation-structure-required-route:\($0)" }
        return lines
    }
}

private extension SmeltPackageResolvedPlan.PolicyPlan {
    var signatureLines: [String] {
        var lines = ["policy:\(mode.rawValue)"]
        if let tokenizer {
            lines.append(
                "policy-tokenizer:\(tokenizer.format):"
                    + tokenizer.files.joined(separator: ",")
            )
        }
        if let inference {
            lines.append("policy-inference:max-tokens:\(inference.maxTokens)")
            lines.append(
                "policy-inference:eos:"
                    + inference.eosTokens.map(String.init).joined(separator: ",")
            )
            lines.append("policy-inference:think-token:\(optional(inference.thinkToken))")
            lines.append("policy-inference:think-end-token:\(optional(inference.thinkEndToken))")
            lines.append(
                "policy-inference:think-skip-suffix:\(optional(inference.thinkSkipSuffix))"
            )
            lines.append("policy-inference:chat-template:\(inference.chatTemplate ?? "none")")
            lines.append(
                "policy-inference:thinking-policy:"
                    + (inference.thinkingPolicy?.rawValue ?? "none")
            )
            lines.append(
                "policy-inference:tool-transcript-codec:"
                    + (inference.toolTranscriptCodec ?? "none")
            )
            lines.append(
                "policy-inference:prompt-state-restore-mode:"
                    + (inference.promptStateRestoreMode?.rawValue ?? "none")
            )
        }
        if let decode {
            lines.append("policy-decode:sampler:\(decode.sampler.mode.rawValue)")
            lines.append("policy-decode:temperature:\(optional(decode.sampler.temperature))")
            lines.append("policy-decode:top-k:\(optional(decode.sampler.topK))")
            lines.append("policy-decode:top-p:\(optional(decode.sampler.topP))")
            if let subSampler = decode.subSampler {
                lines.append("policy-decode:sub-sampler:\(subSampler.mode.rawValue)")
                lines.append("policy-decode:sub-temperature:\(optional(subSampler.temperature))")
                lines.append("policy-decode:sub-top-k:\(optional(subSampler.topK))")
                lines.append("policy-decode:sub-top-p:\(optional(subSampler.topP))")
            }
            lines.append("policy-decode:max-steps:\(optional(decode.maxSteps))")
            lines.append("policy-decode:duration-seconds:\(optional(decode.durationSeconds))")
        }
        return lines
    }
}

private func optional<T>(_ value: T?) -> String {
    value.map { "\($0)" } ?? "none"
}

private func valueSignature(_ value: SmeltPackageSpecValue) -> String {
    switch value {
    case .string(let string):
        return quoted(string)
    case .int(let int):
        return String(int)
    case .number(let double):
        return String(double)
    case .bool(let bool):
        return bool ? "true" : "false"
    case .array(let values):
        return "[" + values.map(valueSignature).joined(separator: ",") + "]"
    case .object(let object):
        return "{"
            + object.keys.sorted().map { key in
                "\(quoted(key)):\(valueSignature(object[key]!))"
            }.joined(separator: ",")
            + "}"
    case .null:
        return "null"
    }
}

private func quoted(_ string: String) -> String {
    var out = "\""
    for scalar in string.unicodeScalars {
        switch scalar {
        case "\"":
            out += "\\\""
        case "\\":
            out += "\\\\"
        case "\n":
            out += "\\n"
        case "\r":
            out += "\\r"
        case "\t":
            out += "\\t"
        default:
            out.unicodeScalars.append(scalar)
        }
    }
    out += "\""
    return out
}
