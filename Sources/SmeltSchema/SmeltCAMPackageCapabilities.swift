import CryptoKit
import Foundation

public enum SmeltCAMPackageCapabilitiesError: Error, CustomStringConvertible, Equatable {
    case nonRegularDescriptor(String)
    case descriptorTooLarge(Int, max: Int)
    case invalidDescriptor(String, String)
    case invalidRequest(String, String)
    case noMatchingExport(String)
    case ambiguousExport(String, [String])

    public var description: String {
        switch self {
        case .nonRegularDescriptor(let path):
            return "CAM package descriptor is not a regular file: \(path)"
        case .descriptorTooLarge(let bytes, let max):
            return "CAM package descriptor is \(bytes) bytes; maximum is \(max)"
        case .invalidDescriptor(let path, let reason):
            return "invalid CAM package descriptor \(path): \(reason)"
        case .invalidRequest(let request, let reason):
            return "invalid CAM capability request '\(request)': \(reason)"
        case .noMatchingExport(let request):
            return "no CAM export satisfies request '\(request)'"
        case .ambiguousExport(let request, let exports):
            return "CAM request '\(request)' is ambiguous: \(exports.joined(separator: ", "))"
        }
    }
}

public struct SmeltCAMCapabilityRequest: Sendable, Equatable {
    public struct PortShape: Sendable, Equatable, Hashable {
        public let typeName: String
        public let attributes: [String: String]

        public init(typeName: String, attributes: [String: String] = [:]) {
            self.typeName = typeName
            self.attributes = attributes
        }

        public static let textUTF8 = PortShape(
            typeName: "text",
            attributes: ["encoding": "utf8"]
        )

        public static let pcmF32AnyRate = PortShape(
            typeName: "pcm",
            attributes: ["dtype": "f32"]
        )

        public static func artifact(mediaType: String) -> PortShape {
            PortShape(
                typeName: "artifact",
                attributes: ["media-type": mediaType]
            )
        }

        public static func pcmF32(rate: String) -> PortShape {
            PortShape(typeName: "pcm", attributes: ["dtype": "f32", "rate": rate])
        }

        public func matches(_ port: SmeltCAMPackageDescriptor.Port) -> Bool {
            guard port.type.typeName == typeName else { return false }
            return attributes.allSatisfy { attribute in
                port.type.attributes[attribute.key] == attribute.value
            }
        }
    }

    public struct ComparisonShape: Sendable, Equatable, Hashable {
        public let subject: String
        public let relation: String
        public let value: String
        public let unit: String?

        public init(subject: String, relation: String, value: String, unit: String? = nil) {
            self.subject = subject
            self.relation = relation
            self.value = value
            self.unit = unit
        }

        fileprivate func matches(_ comparison: SmeltCAMPackageDescriptor.Comparison) -> Bool {
            comparison.subject == subject
                && comparison.relation == relation
                && comparison.value == value
                && comparison.unit == unit
        }
    }

    public struct RequirementShape: Sendable, Equatable, Hashable {
        public enum ValueShape: Sendable, Equatable, Hashable {
            case any
            case containsTerm(String)
            case prefillGPUBatch
            case prefillAllLogitsGPUBatch
        }

        public let key: String
        public let valueShape: ValueShape

        public init(key: String, valueContains: String? = nil) {
            self.key = key
            if let valueContains {
                valueShape = .containsTerm(valueContains)
            } else {
                valueShape = .any
            }
        }

        public init(key: String, valueShape: ValueShape) {
            self.key = key
            self.valueShape = valueShape
        }

        public static func prefillAllLogitsGPUBatchSize(_ value: String) -> Int? {
            let parts = value.split { $0 == " " || $0 == "\t" || $0 == "\n" }.map(String.init)
            guard parts.count == 4,
                  parts[0] == "metal",
                  parts[1] == "all-logits",
                  parts[2] == "batch"
            else {
                return nil
            }
            return Int(parts[3])
        }

        public static func prefillGPUBatchSize(_ value: String) -> Int? {
            let parts = value.split { $0 == " " || $0 == "\t" || $0 == "\n" }.map(String.init)
            if parts.count == 3,
               parts[0] == "metal",
               parts[1] == "batch" {
                return Int(parts[2])
            }
            if parts.count == 4,
               parts[0] == "metal",
               parts[1] == "verify-argmax",
               parts[2] == "batch" {
                return Int(parts[3])
            }
            return nil
        }
    }

    public struct GateObservation: Sendable, Equatable, Hashable {
        public enum FlowScope: Sendable, Equatable, Hashable {
            case selectedExportFlow
        }

        public let requirementComparisons: [ComparisonShape]
        public let requirementSubjects: Set<String>
        public let fromEventType: String?
        public let fromFlow: FlowScope?
        public let toEventType: String
        public let toFlow: FlowScope
        public let outputShape: PortShape
        public let outputPortName: String?
        public let predicateComparisons: [ComparisonShape]
        public let predicateSubjects: Set<String>
        public let requireFormatPredicateMatchingOutput: Bool

        public init(
            requirementComparisons: [ComparisonShape] = [],
            requirementSubjects: Set<String> = [],
            fromEventType: String? = nil,
            fromFlow: FlowScope? = nil,
            toEventType: String,
            toFlow: FlowScope = .selectedExportFlow,
            outputShape: PortShape,
            outputPortName: String? = nil,
            predicateComparisons: [ComparisonShape],
            predicateSubjects: Set<String> = [],
            requireFormatPredicateMatchingOutput: Bool = false
        ) {
            self.requirementComparisons = requirementComparisons
            self.requirementSubjects = requirementSubjects
            self.fromEventType = fromEventType
            self.fromFlow = fromFlow
            self.toEventType = toEventType
            self.toFlow = toFlow
            self.outputShape = outputShape
            self.outputPortName = outputPortName
            self.predicateComparisons = predicateComparisons
            self.predicateSubjects = predicateSubjects
            self.requireFormatPredicateMatchingOutput = requireFormatPredicateMatchingOutput
        }
    }

    public let name: String
    public let requiredInputShapes: [PortShape]
    public let requiredOutputShapes: [PortShape]
    public let requiredInputNames: [String]
    public let requiredOutputNames: [String]
    public let allowAdditionalRequiredInputs: Bool
    public let allowAdditionalOutputs: Bool
    public let requiredAnyExportFacts: Set<String>
    public let requiredAllExportFacts: Set<String>
    public let requiredPackageFiles: Set<String>
    public let requiredCompileRequirementKeys: Set<String>
    public let requiredCompileRequirements: [RequirementShape]
    public let forbiddenRouteTransformerLayerRoles: Set<String>
    public let requiredGateSubjects: Set<String>
    public let requiredGateObservations: [GateObservation]
    public let requireAllGateObservations: Bool

    public init(
        name: String,
        requiredInputShapes: [PortShape],
        requiredOutputShapes: [PortShape],
        requiredInputNames: [String] = [],
        requiredOutputNames: [String] = [],
        allowAdditionalRequiredInputs: Bool = false,
        allowAdditionalOutputs: Bool = true,
        requiredAnyExportFacts: Set<String> = [],
        requiredAllExportFacts: Set<String> = [],
        requiredPackageFiles: Set<String> = [],
        requiredCompileRequirementKeys: Set<String> = [],
        requiredCompileRequirements: [RequirementShape] = [],
        forbiddenRouteTransformerLayerRoles: Set<String> = [],
        requiredGateSubjects: Set<String> = [],
        requiredGateObservations: [GateObservation] = [],
        requireAllGateObservations: Bool = false
    ) {
        self.name = name
        self.requiredInputShapes = requiredInputShapes
        self.requiredOutputShapes = requiredOutputShapes
        self.requiredInputNames = requiredInputNames
        self.requiredOutputNames = requiredOutputNames
        self.allowAdditionalRequiredInputs = allowAdditionalRequiredInputs
        self.allowAdditionalOutputs = allowAdditionalOutputs
        self.requiredAnyExportFacts = requiredAnyExportFacts
        self.requiredAllExportFacts = requiredAllExportFacts
        self.requiredPackageFiles = requiredPackageFiles
        self.requiredCompileRequirementKeys = requiredCompileRequirementKeys
        self.requiredCompileRequirements = requiredCompileRequirements
        self.forbiddenRouteTransformerLayerRoles = forbiddenRouteTransformerLayerRoles
        self.requiredGateSubjects = requiredGateSubjects
        self.requiredGateObservations = requiredGateObservations
        self.requireAllGateObservations = requireAllGateObservations
    }

    private static let streamingExportFact = "run.stream"

    private static func firstAudioObservation(outputShape: PortShape) -> GateObservation {
        GateObservation(
            requirementSubjects: ["elapsed"],
            fromEventType: "flow.accepted",
            fromFlow: .selectedExportFlow,
            toEventType: "emit",
            outputShape: outputShape,
            predicateComparisons: [],
            predicateSubjects: ["duration", "format"],
            requireFormatPredicateMatchingOutput: true
        )
    }

    private static let genericFirstAudioObservation = firstAudioObservation(
        outputShape: .pcmF32AnyRate
    )

    private static let firstAudio24KObservation = firstAudioObservation(
        outputShape: .pcmF32(rate: "24khz")
    )

    private static let realtimeObservation = GateObservation(
        requirementComparisons: [
            .init(subject: "realtime-x", relation: ">=", value: "2"),
        ],
        toEventType: "gate",
        outputShape: .pcmF32AnyRate,
        predicateComparisons: []
    )

    private static let voiceDefaultsOutputShape = PortShape.pcmF32(rate: "24khz")

    public static func exactTextToText(
        name: String,
        requiredTextInputCount: Int,
        requiredInputNames: [String] = [],
        requiredAnyExportFacts: Set<String>,
        requiredGateObservations: [GateObservation] = []
    ) -> SmeltCAMCapabilityRequest {
        precondition(requiredTextInputCount > 0, "text input count must be positive")
        return SmeltCAMCapabilityRequest(
            name: name,
            requiredInputShapes: Array(
                repeating: .textUTF8,
                count: requiredTextInputCount
            ),
            requiredOutputShapes: [.textUTF8],
            requiredInputNames: requiredInputNames,
            allowAdditionalRequiredInputs: false,
            allowAdditionalOutputs: false,
            requiredAnyExportFacts: requiredAnyExportFacts,
            requiredGateObservations: requiredGateObservations
        )
    }

    public static func firstTextOutputObservation(
        elapsedSubjectRequired: Bool = true
    ) -> GateObservation {
        GateObservation(
            requirementSubjects: elapsedSubjectRequired ? ["elapsed"] : [],
            toEventType: "emit",
            outputShape: .textUTF8,
            predicateComparisons: [
                .init(subject: "tokens", relation: ">=", value: "1"),
            ]
        )
    }

    public static let runText = SmeltCAMCapabilityRequest(
        name: "run text",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"]
    )

    public static let runMultimodalText = SmeltCAMCapabilityRequest(
        name: "run multimodal text",
        requiredInputShapes: [
            .textUTF8,
            // Concrete media kinds belong to the selected export/component.
            // Admission asks only for a media port so future modalities use
            // the same graph path rather than acquiring a family switch.
            .init(typeName: "media"),
        ],
        requiredOutputShapes: [.textUTF8],
        allowAdditionalRequiredInputs: false,
        allowAdditionalOutputs: false,
        requiredAllExportFacts: ["run.generate.multimodal"]
    )

    public static let runAudio = SmeltCAMCapabilityRequest(
        name: "run audio",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.pcmF32AnyRate],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredGateObservations: [SmeltCAMCapabilityRequest.genericFirstAudioObservation]
    )

    public static func runFileTransform(
        inputName: String,
        inputMediaType: String,
        outputName: String,
        outputMediaType: String
    ) -> SmeltCAMCapabilityRequest {
        SmeltCAMCapabilityRequest(
            name: "run file transform",
            requiredInputShapes: [.artifact(mediaType: inputMediaType)],
            requiredOutputShapes: [.artifact(mediaType: outputMediaType)],
            requiredInputNames: [inputName],
            requiredOutputNames: [outputName],
            allowAdditionalRequiredInputs: false,
            allowAdditionalOutputs: false,
            requiredAllExportFacts: ["run.transform"]
        )
    }

    public static let benchDecode = SmeltCAMCapabilityRequest(
        name: "bench decode",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredGateObservations: [
            SmeltCAMCapabilityRequest.firstTextOutputObservation(),
        ]
    )

    public static let benchAudio = SmeltCAMCapabilityRequest(
        name: "bench audio",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.pcmF32AnyRate],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredGateObservations: [
            SmeltCAMCapabilityRequest.genericFirstAudioObservation,
            SmeltCAMCapabilityRequest.realtimeObservation,
        ],
        requireAllGateObservations: true
    )

    public static let benchPrefill = SmeltCAMCapabilityRequest(
        name: "bench prefill",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: ["prefill_dispatches.bin"],
        requiredCompileRequirements: [
            .init(key: "prefill", valueShape: .prefillGPUBatch),
        ]
    )

    public static let mtpBenchTarget = SmeltCAMCapabilityRequest(
        name: "mtp bench target",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "SmeltGenerated.swift",
            "dispatches.bin",
            "tokenizer.json",
        ]
    )

    public static let prefillParity = SmeltCAMCapabilityRequest(
        name: "prefill parity",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: ["dispatches.bin", "prefill_dispatches.bin"],
        requiredCompileRequirements: [
            .init(key: "prefill", valueShape: .prefillGPUBatch),
        ]
    )

    public static let profilePrefillKernels = SmeltCAMCapabilityRequest(
        name: "profile prefill kernels",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: ["prefill_dispatches.bin"],
        requiredCompileRequirements: [
            .init(key: "prefill", valueShape: .prefillGPUBatch),
        ]
    )

    public static let profileDecodeKernels = SmeltCAMCapabilityRequest(
        name: "profile decode kernels",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: ["dispatches.bin"]
    )

    public static let kernelLabPackage = SmeltCAMCapabilityRequest(
        name: "kernel lab package",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "SmeltGenerated.swift",
            "dispatches.bin",
        ]
    )

    public static let serveText = SmeltCAMCapabilityRequest(
        name: "serve text",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"]
    )

    public static let serveAudioStream = SmeltCAMCapabilityRequest(
        name: "serve audio stream",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.pcmF32AnyRate],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredAllExportFacts: [SmeltCAMCapabilityRequest.streamingExportFact],
        requiredGateObservations: [SmeltCAMCapabilityRequest.genericFirstAudioObservation]
    )

    public static let serveAudio = SmeltCAMCapabilityRequest(
        name: "serve audio",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [SmeltCAMCapabilityRequest.voiceDefaultsOutputShape],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredAllExportFacts: [SmeltCAMCapabilityRequest.streamingExportFact],
        requiredGateObservations: [SmeltCAMCapabilityRequest.firstAudio24KObservation]
    )

    public static let bakeTextPromptPrefix = SmeltCAMCapabilityRequest(
        name: "bake text prompt-prefix",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["bake.prompt-prefix"]
    )

    public static let bakeVoiceDefaults = SmeltCAMCapabilityRequest(
        name: "bake voice defaults",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [SmeltCAMCapabilityRequest.voiceDefaultsOutputShape],
        requiredAnyExportFacts: ["bake.voice-defaults"]
    )

    public static let traceTextGenerate = SmeltCAMCapabilityRequest(
        name: "trace text generation",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"]
    )

    public static let traceAudioSynthesis = SmeltCAMCapabilityRequest(
        name: "trace audio synthesis",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.pcmF32AnyRate],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredGateObservations: [SmeltCAMCapabilityRequest.genericFirstAudioObservation]
    )

    public static let traceTextSynthesize = SmeltCAMCapabilityRequest(
        name: "trace text synthesis",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [SmeltCAMCapabilityRequest.voiceDefaultsOutputShape],
        requiredAnyExportFacts: ["run.synthesize"],
        requiredGateObservations: [SmeltCAMCapabilityRequest.firstAudio24KObservation]
    )

    public static let inspectDecodeDispatchTable = SmeltCAMCapabilityRequest(
        name: "inspect decode dispatch table",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: ["dispatches.bin"]
    )

    public static let optimizerReport = SmeltCAMCapabilityRequest(
        name: "optimizer report",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAnyExportFacts: ["run.generate"],
        requiredPackageFiles: [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "SmeltGenerated.swift",
            "dispatches.bin",
            "tokenizer.json",
        ]
    )

    public static let inspectPrefillDispatchTable = SmeltCAMCapabilityRequest(
        name: "inspect prefill dispatch table",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAllExportFacts: ["inspect.prefill-dispatch-table"],
        requiredPackageFiles: ["prefill_dispatches.bin"],
        requiredCompileRequirementKeys: ["prefill"]
    )

    public static let benchPrefillLogprobs = SmeltCAMCapabilityRequest(
        name: "bench prefill logprobs",
        requiredInputShapes: [.textUTF8],
        requiredOutputShapes: [.textUTF8],
        requiredAllExportFacts: ["bench.prefill-logprobs"],
        requiredPackageFiles: ["prefill_dispatches.bin"],
        requiredCompileRequirements: [
            .init(key: "prefill", valueShape: .prefillAllLogitsGPUBatch),
        ],
        forbiddenRouteTransformerLayerRoles: ["delta"]
    )

}

public struct SmeltCAMPackageCapabilities: Sendable, Equatable {
    public static let maxDescriptorBytes = 1_048_576

    public struct ExportRecord: Sendable, Equatable {
        public let exportID: String
        public let capabilities: [String]
        public let boundFlowID: String
        public let inputPorts: [SmeltCAMPackageDescriptor.Port]
        public let outputPorts: [SmeltCAMPackageDescriptor.Port]
        public let gateIDs: [String]
    }

    public struct FlowRecord: Sendable, Equatable {
        public let flowID: String
        public let phases: [SmeltCAMPackageDescriptor.FlowPhase]
        public let emittedEndpoints: [SmeltCAMPackageDescriptor.EndpointRef]
        public let stopConditions: [SmeltCAMPackageDescriptor.StopCondition]
    }

    public struct GateContractRecord: Sendable, Equatable {
        public let gateID: String
        public let exportID: String
        public let flowID: String
        public let contract: SmeltCAMPackageDescriptor.GateContract
    }

    public struct RuntimeContractSignature: Sendable, Equatable {
        public let provenance: [String]
        public let body: [String]

        public var lines: [String] {
            ["cam-runtime-contract:v1"]
                + provenance.map { "provenance:\($0)" }
                + body.map { "body:\($0)" }
        }
    }

    public struct RuntimeAssemblyFeatureContract: Sendable, Equatable {
        public static let currentSchema = "smelt.module.runtime_assembly_feature_contract.v1"

        public let schema: String
        public let configuredGraphFeatureSet: [String]
        public let featureSet: [String]

        public init(
            schema: String = Self.currentSchema,
            configuredGraphFeatureSet: [String],
            featureSet: [String]
        ) {
            self.schema = schema
            self.configuredGraphFeatureSet = configuredGraphFeatureSet
            self.featureSet = featureSet
        }
    }

    /// Lossless export-selected graph region consumed by generic component
    /// binders. Backends may replace a structurally validated region without
    /// erasing the surrounding authored graph or selecting by model name.
    public struct ExecutionGraph: Sendable, Equatable {
        public let exportID: String
        public let flowID: String
        public let inputPorts: [SmeltCAMPackageDescriptor.Port]
        public let outputPorts: [SmeltCAMPackageDescriptor.Port]
        public let selectedInputPorts: [SmeltCAMPackageDescriptor.Port]
        public let selectedOutputPorts: [SmeltCAMPackageDescriptor.Port]
        public let authoredCapabilities: [String]
        public let nodes: [SmeltCAMPackageDescriptor.GraphNode]
        public let blocks: [SmeltCAMPackageDescriptor.Block]
        public let dataEdges: [SmeltCAMPackageDescriptor.GraphEdge]
        public let feedbackEdges: [SmeltCAMPackageDescriptor.FeedbackEdge]
        public let phases: [SmeltCAMPackageDescriptor.FlowPhase]
        public let emissions: [SmeltCAMPackageDescriptor.EndpointRef]
        public let stopConditions: [SmeltCAMPackageDescriptor.StopCondition]
        public let tensorBindings: [SmeltCAMPackageDescriptor.TensorBinding]
        public let imports: [SmeltCAMPackageDescriptor.Import]
        public let sourceReferences: [SmeltCAMPackageDescriptor.SourceReference]
        public let backendRequirements: [SmeltCAMPackageDescriptor.Requirement]
        public let compileRequirements: [SmeltCAMPackageDescriptor.Requirement]
        public let quantization: [SmeltCAMPackageDescriptor.QuantizationRule]
        public let sourceQuantization: [SmeltCAMPackageDescriptor.SourceQuantizationRule]
        public let artifactRequirements: [SmeltCAMPackageDescriptor.ArtifactRequirement]
        public let gateContracts: [SmeltCAMPackageDescriptor.GateContract]
    }

    public struct Decision: Sendable, Equatable {
        public let exportID: String
        public let flowID: String
        public let inputPorts: [SmeltCAMPackageDescriptor.Port]
        public let outputPorts: [SmeltCAMPackageDescriptor.Port]
        public let selectedInputPorts: [SmeltCAMPackageDescriptor.Port]
        public let selectedOutputPorts: [SmeltCAMPackageDescriptor.Port]
        public let matchedGateIDs: [String]
        public let matchedGateContracts: [GateContractRecord]
        public let authoredCapabilities: [String]

        public func traceRouteWitnessV6(
            camSemanticSHA256: String,
            exportABISHA256: String
        ) -> String {
            return (
                ["cam-route:v6"]
                    + traceRouteWitnessV6Fields(
                        camSemanticSHA256: camSemanticSHA256,
                        exportABISHA256: exportABISHA256
                    )
            ).joined(separator: ";")
        }

        private func traceRouteWitnessV6Fields(
            camSemanticSHA256: String,
            exportABISHA256: String
        ) -> [String] {
            [
                Self.traceRouteWitnessField("cam", camSemanticSHA256, encodeValue: true),
                Self.traceRouteWitnessField("exportABI", exportABISHA256, encodeValue: true),
                Self.traceRouteWitnessField("export", exportID, encodeValue: true),
                Self.traceRouteWitnessField("flow", flowID, encodeValue: true),
                Self.traceRouteWitnessField("gates", matchedGateIDs.joined(separator: ","), encodeValue: true),
                Self.traceRouteWitnessField(
                    "capabilities",
                    authoredCapabilities.sorted().joined(separator: ","),
                    encodeValue: true
                ),
                Self.traceRouteWitnessField(
                    "inputs",
                    inputPorts.map(Self.portWitnessSignature).sorted().joined(separator: ","),
                    encodeValue: true
                ),
                Self.traceRouteWitnessField(
                    "outputs",
                    outputPorts.map(Self.portWitnessSignature).sorted().joined(separator: ","),
                    encodeValue: true
                ),
            ]
        }

        private static func traceRouteWitnessField(
            _ name: String,
            _ value: String,
            encodeValue: Bool
        ) -> String {
            "\(name)=\(encodeValue ? percentEncodeTraceRouteWitnessValue(value) : value)"
        }

        private static func percentEncodeTraceRouteWitnessValue(_ value: String) -> String {
            value.utf8.map { byte in
                if isTraceRouteWitnessUnreserved(byte) {
                    return String(UnicodeScalar(byte))
                }
                return String(format: "%%%02X", byte)
            }.joined()
        }

        private static func isTraceRouteWitnessUnreserved(_ byte: UInt8) -> Bool {
            (byte >= 0x30 && byte <= 0x39)
                || (byte >= 0x41 && byte <= 0x5A)
                || (byte >= 0x61 && byte <= 0x7A)
                || byte == 0x2D
                || byte == 0x2E
                || byte == 0x5F
                || byte == 0x7E
        }

        private static func portWitnessSignature(_ port: SmeltCAMPackageDescriptor.Port) -> String {
            let attributes = port.type.attributes
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            return "\(port.portName):\(port.type.typeName){\(attributes)}"
                + ":\(port.optional ? "optional" : "required")"
        }
    }

    private struct ResolvedRouteFacts {
        let flow: FlowRecord
        let calledNodeIDs: Set<String>
        let calledNodes: [SmeltCAMPackageDescriptor.GraphNode]
        let nodeOrdinals: [String: String]
        let calledBlocks: [SmeltCAMPackageDescriptor.Block]
        let blockOrdinals: [String: String]
        let routeEdges: [SmeltCAMPackageDescriptor.GraphEdge]
        let routeFeedbackEdges: [SmeltCAMPackageDescriptor.FeedbackEdge]
    }

    private struct ExecutionMembership {
        var nodeIDs: Set<String> = []
        var graphValueNames: Set<String> = []
        var importedExportKeys: Set<String> = []
    }

    public let packagePath: String?
    public let descriptorIdentityToken: String?
    public let featureAdmission: SmeltCAMFeatureAdmission
    public let descriptorSchema: String
    public let descriptorVersion: Int
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let descriptorGraphSignatureSHA256: String
    public let exports: [ExportRecord]
    public let flows: [FlowRecord]
    public let imports: [SmeltCAMPackageDescriptor.Import]
    public let sourceReferences: [SmeltCAMPackageDescriptor.SourceReference]
    public let backendRequirements: [SmeltCAMPackageDescriptor.Requirement]
    public let compileRequirements: [SmeltCAMPackageDescriptor.Requirement]
    public let blocks: [SmeltCAMPackageDescriptor.Block]
    public let graphNodes: [SmeltCAMPackageDescriptor.GraphNode]
    public let graphEdges: [SmeltCAMPackageDescriptor.GraphEdge]
    public let feedbackEdges: [SmeltCAMPackageDescriptor.FeedbackEdge]
    public let tensorBindings: [SmeltCAMPackageDescriptor.TensorBinding]
    public let quantization: [SmeltCAMPackageDescriptor.QuantizationRule]
    public let sourceQuantization: [SmeltCAMPackageDescriptor.SourceQuantizationRule]
    public let artifactRequirements: [SmeltCAMPackageDescriptor.ArtifactRequirement]
    public let gateContracts: [SmeltCAMPackageDescriptor.GateContract]
    private let gateContractsByID: [String: SmeltCAMPackageDescriptor.GateContract]

    public init(
        descriptor: SmeltCAMPackageDescriptor,
        consumedFeatureSet: [String] = [],
        consumedObligationIDs: [String] = [],
        packagePath: String? = nil,
        descriptorIdentityToken: String? = nil
    ) throws {
        try descriptor.validateDecoded()
        let bindingsByExport = Dictionary(
            uniqueKeysWithValues: descriptor.exportFlowBindings.map { ($0.exportID, $0.flowID) }
        )
        featureAdmission = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: consumedFeatureSet,
            consumedObligationIDs: consumedObligationIDs
        )
        descriptorSchema = descriptor.descriptorSchema
        descriptorVersion = descriptor.descriptorVersion
        camSemanticSHA256 = descriptor.camSemanticSHA256
        exportABISHA256 = descriptor.exportABISHA256
        descriptorGraphSignatureSHA256 = Self.sha256Hex(
            Data(descriptor.graphSignature.joined(separator: "\n").utf8)
        )
        self.packagePath = packagePath
        self.descriptorIdentityToken = descriptorIdentityToken
        exports = descriptor.exports.map {
            ExportRecord(
                exportID: $0.exportID,
                capabilities: $0.capabilities,
                boundFlowID: bindingsByExport[$0.exportID] ?? "",
                inputPorts: $0.inputs,
                outputPorts: $0.outputs,
                gateIDs: $0.gates
            )
        }
        flows = descriptor.flows.map {
            FlowRecord(
                flowID: $0.flowID,
                phases: $0.phases,
                emittedEndpoints: $0.emit,
                stopConditions: $0.stop
            )
        }
        imports = descriptor.imports
        sourceReferences = descriptor.sourceReferences
        backendRequirements = descriptor.backendRequirements
        compileRequirements = descriptor.compileRequirements
        blocks = descriptor.blocks
        graphNodes = descriptor.graphNodes
        graphEdges = descriptor.graphEdges
        feedbackEdges = descriptor.feedbackEdges
        tensorBindings = descriptor.tensorBindings
        quantization = descriptor.quantization
        sourceQuantization = descriptor.sourceQuantization
        artifactRequirements = descriptor.artifactRequirements
        gateContracts = descriptor.gateContracts
        gateContractsByID = Dictionary(
            uniqueKeysWithValues: descriptor.gateContracts.map { ($0.gateID, $0) }
        )
    }

    public static func loadIfPresent(
        packageURL: URL,
        consumedFeatureSet: [String] = [],
        consumedObligationIDs: [String] = []
    ) throws -> SmeltCAMPackageCapabilities? {
        let descriptorURL = packageURL.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        let path = descriptorURL.path
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: path)) != nil {
            throw SmeltCAMPackageCapabilitiesError.nonRegularDescriptor(path)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: path)
        guard attributes[.type] as? FileAttributeType == .typeRegular else {
            throw SmeltCAMPackageCapabilitiesError.nonRegularDescriptor(path)
        }
        let size = (attributes[.size] as? NSNumber)?.intValue ?? 0
        guard size <= maxDescriptorBytes else {
            throw SmeltCAMPackageCapabilitiesError.descriptorTooLarge(
                size,
                max: maxDescriptorBytes
            )
        }
        let data = try Data(contentsOf: descriptorURL)
        guard data.count <= maxDescriptorBytes else {
            throw SmeltCAMPackageCapabilitiesError.descriptorTooLarge(
                data.count,
                max: maxDescriptorBytes
            )
        }
        let descriptor: SmeltCAMPackageDescriptor
        do {
            descriptor = try JSONDecoder().decode(SmeltCAMPackageDescriptor.self, from: data)
        } catch {
            throw SmeltCAMPackageCapabilitiesError.invalidDescriptor(path, "\(error)")
        }
        return try SmeltCAMPackageCapabilities(
            descriptor: descriptor,
            consumedFeatureSet: consumedFeatureSet,
            consumedObligationIDs: consumedObligationIDs,
            packagePath: packageURL.path,
            descriptorIdentityToken: descriptorIdentityToken(attributes: attributes, size: data.count)
        )
    }

    public func resolve(_ request: SmeltCAMCapabilityRequest) throws -> Decision {
        try validatePortBindings(request)
        let matched = exports.filter { export in
            exportMatches(export, request: request)
        }
        guard !matched.isEmpty else {
            throw SmeltCAMPackageCapabilitiesError.noMatchingExport(request.name)
        }
        guard matched.count == 1, let export = matched.first else {
            throw SmeltCAMPackageCapabilitiesError.ambiguousExport(
                request.name,
                matched.map(\.exportID).sorted()
            )
        }
        let gateRecords = matchedGateContracts(export: export, request: request)
        let selectedInputPorts = selectedPorts(
            export.inputPorts.filter { !$0.optional },
            satisfy: request.requiredInputShapes,
            names: request.requiredInputNames
        ) ?? []
        let selectedOutputPorts = selectedPorts(
            export.outputPorts.filter { !$0.optional },
            satisfy: request.requiredOutputShapes,
            names: request.requiredOutputNames
        ) ?? []
        return Decision(
            exportID: export.exportID,
            flowID: export.boundFlowID,
            inputPorts: export.inputPorts,
            outputPorts: export.outputPorts,
            selectedInputPorts: selectedInputPorts,
            selectedOutputPorts: selectedOutputPorts,
            matchedGateIDs: gateRecords.map(\.gateID),
            matchedGateContracts: gateRecords,
            authoredCapabilities: export.capabilities
        )
    }

    public func executionGraph(for decision: Decision) throws -> ExecutionGraph {
        guard let export = exports.first(where: { $0.exportID == decision.exportID }),
              export.boundFlowID == decision.flowID,
              let flow = flows.first(where: { $0.flowID == decision.flowID })
        else {
            throw SmeltCAMPackageCapabilitiesError.invalidDescriptor(
                packagePath ?? "<memory>",
                "resolved module decision does not name a bound export and flow"
            )
        }

        let inputNames = Set(decision.inputPorts.map(\.portName))
        let outputNames = Set(decision.outputPorts.map(\.portName))
        var membership = ExecutionMembership(nodeIDs: Self.calledNodeIDs(in: flow))
        for endpoint in flow.emittedEndpoints {
            _ = Self.activate(
                endpoint,
                membership: &membership,
                inputNames: inputNames,
                outputNames: outputNames
            )
        }
        for call in flow.phases.flatMap(\.calls) {
            if let imported = call.imported {
                membership.importedExportKeys.insert(
                    Self.importedExportKey(alias: imported.alias, exportID: imported.exportID)
                )
            }
        }

        let endpointPairs = graphEdges.map { ($0.from, $0.to) }
            + feedbackEdges.map { ($0.from, $0.to) }
        let seedFromInterface = membership.nodeIDs.isEmpty
            && membership.graphValueNames.isEmpty
            && membership.importedExportKeys.isEmpty
        var changed = true
        while changed {
            changed = false
            for (from, to) in endpointPairs
            where Self.isActive(
                from,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames,
                includeInterfaceEndpoints: seedFromInterface
            ) || Self.isActive(
                to,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames,
                includeInterfaceEndpoints: seedFromInterface
            ) {
                changed = Self.activate(
                    from,
                    membership: &membership,
                    inputNames: inputNames,
                    outputNames: outputNames
                ) || changed
                changed = Self.activate(
                    to,
                    membership: &membership,
                    inputNames: inputNames,
                    outputNames: outputNames
                ) || changed
            }
        }

        let selectedNodes = graphNodes.filter { membership.nodeIDs.contains($0.nodeID) }
        let declaredBlockIDs = Set(blocks.map(\.blockID))
        let blockIDs = Set(
            selectedNodes.compactMap(\.blockID)
                + selectedNodes.map(\.nodeID).filter { declaredBlockIDs.contains($0) }
        )
        let selectedBlocks = blocks.filter { blockIDs.contains($0.blockID) }
        let selectedDataEdges = graphEdges.filter {
            Self.isActive(
                $0.from,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames
            ) && Self.isActive(
                $0.to,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames
            )
        }
        let selectedFeedbackEdges = feedbackEdges.filter {
            Self.isActive(
                $0.from,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames
            ) && Self.isActive(
                $0.to,
                membership: membership,
                inputNames: inputNames,
                outputNames: outputNames
            )
        }
        let selectedTensorBindings = tensorBindings.filter {
            blockIDs.contains($0.ownerBlockID) || blockIDs.contains($0.target.blockID)
        }
        let importedAliases = Set(
            selectedNodes.compactMap { $0.imported?.alias }
                + flow.phases.flatMap(\.calls).compactMap { $0.imported?.alias }
                + membership.importedExportKeys.compactMap {
                    $0.split(separator: "\u{1F}").first.map(String.init)
                }
        )
        let declaredArtifactIDs = Set(artifactRequirements.map(\.artifactID))
        let allNodeArtifactIDs = Set(
            graphNodes.flatMap(\.annotations).compactMap { annotation in
                annotation.key == "artifact"
                    && declaredArtifactIDs.contains(annotation.value)
                    ? annotation.value
                    : nil
            }
        )
        let selectedNodeArtifactIDs = Set(
            selectedNodes.flatMap(\.annotations).compactMap { annotation in
                annotation.key == "artifact"
                    && declaredArtifactIDs.contains(annotation.value)
                    ? annotation.value
                    : nil
            }
        )
        let selectedArtifactRequirements = artifactRequirements.filter {
            !allNodeArtifactIDs.contains($0.artifactID)
                || selectedNodeArtifactIDs.contains($0.artifactID)
        }

        return ExecutionGraph(
            exportID: decision.exportID,
            flowID: decision.flowID,
            inputPorts: decision.inputPorts,
            outputPorts: decision.outputPorts,
            selectedInputPorts: decision.selectedInputPorts,
            selectedOutputPorts: decision.selectedOutputPorts,
            authoredCapabilities: decision.authoredCapabilities,
            nodes: selectedNodes,
            blocks: selectedBlocks,
            dataEdges: selectedDataEdges,
            feedbackEdges: selectedFeedbackEdges,
            phases: flow.phases,
            emissions: flow.emittedEndpoints,
            stopConditions: flow.stopConditions,
            tensorBindings: selectedTensorBindings,
            imports: imports.filter { importedAliases.contains($0.alias) },
            sourceReferences: sourceReferences,
            backendRequirements: backendRequirements,
            compileRequirements: compileRequirements,
            quantization: quantization,
            sourceQuantization: sourceQuantization,
            artifactRequirements: selectedArtifactRequirements,
            gateContracts: gateContracts
        )
    }

    public func runtimeContractSignature(
        for decision: Decision
    ) throws -> RuntimeContractSignature {
        let facts = try resolvedRouteFacts(for: decision)
        let gateLines = decision.matchedGateContracts
            .sorted { $0.gateID < $1.gateID }
            .flatMap { Self.gateContractSignature($0, nodeOrdinals: facts.nodeOrdinals) }

        let capabilityLines = decision.authoredCapabilities.sorted().map {
            "capability:\($0)"
        }
        let inputLines = decision.inputPorts.map {
            "input:\(Self.portSignature($0))"
        }.sorted()
        let outputLines = decision.outputPorts.map {
            "output:\(Self.portSignature($0))"
        }.sorted()
        let selectedInputLines = decision.selectedInputPorts.map {
            "selected-input:\(Self.portSignature($0))"
        }.sorted()
        let selectedOutputLines = decision.selectedOutputPorts.map {
            "selected-output:\(Self.portSignature($0))"
        }.sorted()
        let phaseLines = facts.flow.phases.enumerated().map {
            "phase:\($0.offset):\(Self.flowPhaseSignature($0.element, nodeOrdinals: facts.nodeOrdinals))"
        }
        let emitLines = facts.flow.emittedEndpoints.enumerated().map {
            "emit:\($0.offset):\(Self.endpointSignature($0.element, nodeOrdinals: facts.nodeOrdinals))"
        }
        let stopLines = facts.flow.stopConditions.map {
            "stop:\(Self.stopConditionSignature($0))"
        }.sorted()
        let blockLines = facts.calledBlocks.map {
            "block:\(Self.blockSignature($0, blockOrdinals: facts.blockOrdinals))"
        }
        let nodeLines = facts.calledNodes.map { node in
            let signature = Self.graphNodeSignature(
                node,
                nodeOrdinals: facts.nodeOrdinals,
                blockOrdinals: facts.blockOrdinals
            )
            return "node:\(signature)"
        }
        let edgeLines = facts.routeEdges.map {
            "edge:\(Self.graphEdgeSignature($0, nodeOrdinals: facts.nodeOrdinals))"
        }.sorted()
        let feedbackLines = facts.routeFeedbackEdges.map {
            "feedback:\(Self.feedbackEdgeSignature($0, nodeOrdinals: facts.nodeOrdinals))"
        }.sorted()
        let backendLines = backendRequirements.map {
            "backend:\(Self.requirementSignature($0))"
        }.sorted()
        let compileLines = compileRequirements.map {
            "compile:\(Self.requirementSignature($0))"
        }.sorted()
        let artifactLines = artifactRequirements.map {
            "artifact:\(Self.artifactSignature($0))"
        }.sorted()
        let quantLines = quantization.map {
            "quant:\(Self.quantizationSignature($0))"
        }.sorted()
        let sourceQuantLines = sourceQuantization.map {
            "source-quant:\(Self.sourceQuantizationSignature($0))"
        }.sorted()

        var body: [String] = []
        body.append(contentsOf: capabilityLines)
        body.append(contentsOf: inputLines)
        body.append(contentsOf: outputLines)
        body.append(contentsOf: selectedInputLines)
        body.append(contentsOf: selectedOutputLines)
        body.append(contentsOf: phaseLines)
        body.append(contentsOf: emitLines)
        body.append(contentsOf: stopLines)
        body.append(contentsOf: blockLines)
        body.append(contentsOf: nodeLines)
        body.append(contentsOf: edgeLines)
        body.append(contentsOf: feedbackLines)
        body.append(contentsOf: backendLines)
        body.append(contentsOf: compileLines)
        body.append(contentsOf: artifactLines)
        body.append(contentsOf: quantLines)
        body.append(contentsOf: sourceQuantLines)
        body.append(contentsOf: gateLines)

        return RuntimeContractSignature(
            provenance: [
                "descriptor-schema:\(descriptorSchema)",
                "descriptor-version:\(descriptorVersion)",
                "cam:\(camSemanticSHA256)",
                "export-abi:\(exportABISHA256)",
            ],
            body: body
        )
    }

    public func runtimeAssemblyFeatureContract(
        for decision: Decision
    ) throws -> RuntimeAssemblyFeatureContract {
        let facts = try resolvedRouteFacts(for: decision)
        var graphFeatures = RuntimeAssemblyFeatureSetBuilder()

        for phase in facts.flow.phases {
            try graphFeatures.add(["flow", "phase", phase.phaseType])
            for call in phase.calls {
                try graphFeatures.add(["flow", "call", call.callType])
            }
        }
        for node in facts.calledNodes {
            try graphFeatures.add(["graph", "impl", node.implementation])
            try addGraphAnnotationFeatures(node.annotations, to: &graphFeatures)
        }
        if !facts.routeFeedbackEdges.isEmpty {
            try graphFeatures.add(["graph", "feedback"])
        }
        for edge in facts.routeEdges {
            if let valueType = edge.valueType {
                try graphFeatures.add(["graph", "edge", valueType.typeName])
            }
        }
        for block in facts.calledBlocks {
            try addBlockFeatureContract(block, to: &graphFeatures)
        }
        for requirement in backendRequirements {
            try addBackendFeature(requirement, to: &graphFeatures)
        }
        for requirement in compileRequirements {
            try addCompileFeature(requirement, to: &graphFeatures)
        }
        for gate in gateContracts {
            try addDescriptorGateGraphFeatures(gate, to: &graphFeatures)
        }
        for artifact in artifactRequirements {
            try graphFeatures.add(["artifact", "role", artifact.role])
        }
        for rule in quantization {
            try addQuantizationFeature(rule, prefix: "quant", to: &graphFeatures)
        }
        for rule in sourceQuantization {
            try graphFeatures.add(["source-quant", "action", rule.action])
            if let storage = rule.storage {
                try addQuantStorageFeatures(storage, prefix: "source-quant", to: &graphFeatures)
            }
        }

        let configuredGraphFeatureSet = graphFeatures.sortedFeatures
        var allFeatures = RuntimeAssemblyFeatureSetBuilder()
        allFeatures.insert(configuredGraphFeatureSet)
        for port in decision.selectedInputPorts + decision.selectedOutputPorts {
            try allFeatures.add(["io", port.type.typeName])
        }
        for gateID in decision.matchedGateIDs {
            try allFeatures.add(["gate", gateID])
        }
        for record in decision.matchedGateContracts {
            try addGateSubjectFeatures(record.contract, to: &allFeatures)
        }

        return RuntimeAssemblyFeatureContract(
            configuredGraphFeatureSet: configuredGraphFeatureSet,
            featureSet: allFeatures.sortedFeatures
        )
    }

    private func resolvedRouteFacts(for decision: Decision) throws -> ResolvedRouteFacts {
        guard let flow = flows.first(where: { $0.flowID == decision.flowID }) else {
            throw SmeltCAMPackageCapabilitiesError.invalidDescriptor(
                packagePath ?? "<memory>",
                "resolved CAM decision references unknown flow '\(decision.flowID)'"
            )
        }
        let calledNodeIDs = Self.calledNodeIDs(in: flow)
        let calledNodes = graphNodes
            .filter { calledNodeIDs.contains($0.nodeID) }
            .sorted { $0.nodeID < $1.nodeID }
        let nodeOrdinals = Dictionary(uniqueKeysWithValues: calledNodes.enumerated().map {
            ($0.element.nodeID, "n\($0.offset)")
        })
        let calledBlockIDs = Set(calledNodes.compactMap(\.blockID))
        let calledBlocks = blocks
            .filter { calledBlockIDs.contains($0.blockID) }
            .sorted { $0.blockID < $1.blockID }
        let blockOrdinals = Dictionary(uniqueKeysWithValues: calledBlocks.enumerated().map {
            ($0.element.blockID, "b\($0.offset)")
        })
        let routeEdges = Self.routeEdges(
            graphEdges,
            decision: decision,
            flow: flow,
            calledNodeIDs: calledNodeIDs
        )
        let routeFeedbackEdges = Self.routeFeedbackEdges(
            feedbackEdges,
            decision: decision,
            calledNodeIDs: calledNodeIDs
        )
        return ResolvedRouteFacts(
            flow: flow,
            calledNodeIDs: calledNodeIDs,
            calledNodes: calledNodes,
            nodeOrdinals: nodeOrdinals,
            calledBlocks: calledBlocks,
            blockOrdinals: blockOrdinals,
            routeEdges: routeEdges,
            routeFeedbackEdges: routeFeedbackEdges
        )
    }

    private struct RuntimeAssemblyFeatureSetBuilder {
        private var features = Set<String>()

        var sortedFeatures: [String] {
            features.sorted()
        }

        mutating func insert(_ featureSet: [String]) {
            features.formUnion(featureSet)
        }

        mutating func add(_ components: [String]) throws {
            features.insert(try runtimeAssemblyFeatureCode(components))
        }
    }

    private static let runtimeAssemblyFeatureBannedTerms: Set<String> = [
        "llm",
        "tts",
        "asr",
        "family",
        "kind",
        "arch",
        "architecture",
        "profile",
        "bucket",
        "target",
        "handler",
        "registry",
        "policy",
        "manifest",
        "bridge",
        "locator",
        "model",
        "modelname",
        "module" + "id",
    ]

    private static let blockRequirementValueFeatureKeys: Set<String> = [
        "artifact-block",
        "audio-format",
        "audio-rate",
        "sampler",
        "sampler-steps",
    ]

    private static func runtimeAssemblyFeatureCode(_ components: [String]) throws -> String {
        let normalized = components.map { normalizeFeatureComponent($0) }
        guard !normalized.contains(where: \.isEmpty) else {
            throw SmeltCAMPackageCapabilitiesError.invalidDescriptor(
                "runtime-assembly-feature-contract",
                "runtime assembly feature code contains an empty component"
            )
        }
        let code = normalized.joined(separator: ".")
        let selectorTerms = selectorTermSet(code)
        if let banned = selectorTerms.first(where: { runtimeAssemblyFeatureBannedTerms.contains($0) }) {
            throw SmeltCAMPackageCapabilitiesError.invalidDescriptor(
                "runtime-assembly-feature-contract",
                "runtime assembly feature code '\(code)' contains selector-shaped term '\(banned)'"
            )
        }
        return code
    }

    private static func normalizeFeatureComponent(_ value: String) -> String {
        var scalars: [Character] = []
        var previousWasAlnum = false
        for character in value {
            if character.isUppercase {
                if previousWasAlnum {
                    scalars.append("-")
                }
                scalars.append(Character(character.lowercased()))
                previousWasAlnum = true
            } else if character.isLetter || character.isNumber {
                scalars.append(Character(character.lowercased()))
                previousWasAlnum = true
            } else {
                if previousWasAlnum {
                    scalars.append("-")
                }
                previousWasAlnum = false
            }
        }
        return String(scalars)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
    }

    private static func selectorTermSet(_ value: String) -> Set<String> {
        let tokens = value
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .flatMap(splitCamelCaseToken)
            .map { $0.lowercased() }
            .filter { !$0.isEmpty }
        var terms = Set(tokens)
        terms.insert(tokens.joined())
        for index in tokens.indices.dropLast() {
            terms.insert(tokens[index] + tokens[tokens.index(after: index)])
        }
        return terms
    }

    private static func splitCamelCaseToken(_ value: String) -> [String] {
        var out: [String] = []
        var current = ""
        for character in value {
            if character.isUppercase && !current.isEmpty {
                out.append(current)
                current = ""
            }
            current.append(character)
        }
        if !current.isEmpty {
            out.append(current)
        }
        return out
    }

    private func addGraphAnnotationFeatures(
        _ annotations: [SmeltCAMPackageDescriptor.Requirement],
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        for annotation in annotations {
            switch annotation.key {
            case "artifact":
                try features.add(["artifact", "role", annotation.value])
            case "artifact-block":
                try features.add(["artifact", "role", "block"])
            case "state":
                for state in annotation.value.split(separator: ",").map(String.init) {
                    try features.add(["graph", "state", state])
                }
            case "streaming":
                if annotation.value == "true" {
                    try features.add(["graph", "streaming"])
                }
            case "codebooks":
                try features.add(["graph", "codebooks"])
            case "prompt-format":
                try features.add(["graph", "prompt-format", annotation.value])
            case "assistant-prelude":
                try features.add(["graph", "assistant-prelude", annotation.value])
            case "thinking-policy":
                try features.add(["graph", "thinking", annotation.value])
            case "speaker":
                try features.add(["block", "frontend"])
                try features.add(["block", "frontend", "speaker-conditioning"])
            case "tag":
                try addGraphTagFeature(annotation.value, to: &features)
            default:
                break
            }
        }
    }

    private func addGraphTagFeature(
        _ value: String,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        switch value {
        case "text-tokenizer":
            try features.add(["graph", "role", "tokenizer"])
        case "text-detokenizer":
            try features.add(["graph", "role", "detokenizer"])
        case "sampler":
            try features.add(["graph", "role", "sampler"])
        case "duration-plan":
            try features.add(["graph", "role", "duration-plan"])
        default:
            break
        }
    }

    private func addBlockFeatureContract(
        _ block: SmeltCAMPackageDescriptor.Block,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        if let transformer = block.shape.transformer {
            try features.add(["block", "transformer"])
            try addTransformerFeatureContract(transformer, to: &features)
        }
        if let codecDecoder = block.shape.codecDecoder {
            try features.add(["block", "codec-decoder"])
            if codecDecoder.streaming {
                try features.add(["block", "codec-decoder", "streaming"])
            }
            if codecDecoder.codebooks != nil {
                try features.add(["block", "codec-decoder", "codebooks"])
            }
        }
        if let frontend = block.shape.frontend {
            try features.add(["block", "frontend"])
            if frontend.speakerConditioning {
                try features.add(["block", "frontend", "speaker-conditioning"])
            }
        }
        for requirement in block.shape.requirements {
            try addBlockRequirementFeature(requirement, to: &features)
        }
    }

    private func addTransformerFeatureContract(
        _ transformer: SmeltCAMPackageDescriptor.TransformerShape,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        if let layers = transformer.layers {
            for role in layers.roles {
                try features.add(["block", "transformer", "layer-role", role])
            }
        }
        if transformer.delta != nil {
            try features.add(["block", "transformer", "delta"])
        }
        if let attention = transformer.attention {
            try features.add(["block", "transformer", "attention"])
            try addAttentionFeatureContract(attention, role: nil, to: &features)
        }
        for roleAttention in transformer.attentionByRole ?? [] {
            try features.add(["block", "transformer", "attention"])
            try features.add(["block", "transformer", "attention-role", roleAttention.role])
            try addAttentionFeatureContract(
                roleAttention.attention,
                role: roleAttention.role,
                to: &features
            )
        }
        if let ffn = transformer.ffn {
            try features.add(["block", "transformer", "ffn", ffn.activation])
        }
        if let router = transformer.router {
            try features.add(["block", "transformer", "router"])
            try features.add(["block", "transformer", "moe"])
            if let activation = router.activation {
                try features.add(["block", "transformer", "router", "activation", activation])
            }
            if let normalization = router.normalization {
                try features.add([
                    "block", "transformer", "router", "normalization", normalization,
                ])
            }
            if let sharedExperts = router.sharedExperts, sharedExperts > 0 {
                try features.add(["block", "transformer", "router", "shared-experts"])
            }
            if router.scoreCorrectionBias == true {
                try features.add(["block", "transformer", "router", "score-correction-bias"])
            }
            if router.globalScale == true {
                try features.add(["block", "transformer", "router", "global-scale"])
            }
            if router.sharedExpertSink == true {
                try features.add(["block", "transformer", "router", "shared-expert-sink"])
            }
        }
        if transformer.expert != nil {
            try features.add(["block", "transformer", "expert"])
            try features.add(["block", "transformer", "moe"])
        }
        if let norm = transformer.norm {
            try features.add(["block", "transformer", "norm", norm.normType])
            if let mode = norm.mode {
                try features.add(["block", "transformer", "norm-mode", mode])
            }
        }
        if let vocab = transformer.vocab {
            try features.add([
                "block",
                "transformer",
                "vocab",
                vocab.tiedHead ? "tied-head" : "untied-head",
            ])
        }
        if transformer.perLayerInput != nil {
            try features.add(["block", "transformer", "per-layer-input"])
        }
        if transformer.denseLayerCount != nil {
            try features.add(["block", "transformer", "leading-dense-layers"])
        }
        for convolution in transformer.shortConvolutions ?? [] {
            try features.add([
                "block", "transformer", "short-convolution", convolution.site,
            ])
            if convolution.residual != "none" {
                try features.add([
                    "block", "transformer", "short-convolution-residual",
                    convolution.residual,
                ])
            }
        }
        if transformer.sharedKVLayers != nil {
            try features.add(["block", "transformer", "shared-kv"])
        }
        if transformer.logitCap != nil {
            try features.add(["block", "transformer", "logit-cap"])
        }
    }

    private func addAttentionFeatureContract(
        _ attention: SmeltCAMPackageDescriptor.AttentionShape,
        role: String?,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        if let role {
            try features.add(["block", "transformer", "attention-role", role])
        }
        if let rope = attention.rope {
            try features.add(["block", "transformer", "rope", rope.ropeType])
        }
        if let relativePosition = attention.relativePosition {
            try features.add(["block", "transformer", "relative-position"])
            if relativePosition.contentConditioned {
                try features.add([
                    "block", "transformer", "relative-position", "content-conditioned",
                ])
            }
            if relativePosition.logScalingFloor != nil {
                try features.add([
                    "block", "transformer", "relative-position", "log-scaled",
                ])
            }
        }
        if let scaling = attention.scaling {
            try features.add(["block", "transformer", "attention-scaling", scaling])
        }
        if let qkNormType = attention.qkNormType {
            try features.add(["block", "transformer", "qk-norm", qkNormType])
        }
        if let vNormType = attention.vNormType {
            try features.add(["block", "transformer", "v-norm", vNormType])
        }
        if attention.window != nil {
            try features.add(["block", "transformer", "attention", "window"])
        }
    }

    private func addBlockRequirementFeature(
        _ requirement: SmeltCAMPackageDescriptor.BlockRequirement,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        guard requirement.key != "name" else { return }
        try features.add(["block", "requirement", requirement.key])
        let normalizedKey = Self.normalizeFeatureComponent(requirement.key)
        if let value = requirement.value,
           Self.blockRequirementValueFeatureKeys.contains(normalizedKey) {
            try features.add(["block", "requirement", requirement.key, value])
        }
    }

    private func addBackendFeature(
        _ requirement: SmeltCAMPackageDescriptor.Requirement,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        switch requirement.key {
        case "target":
            try features.add(["compile", "backend", requirement.value])
        default:
            try features.add(["backend", requirement.key])
        }
    }

    private func addCompileFeature(
        _ requirement: SmeltCAMPackageDescriptor.Requirement,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        switch requirement.key {
        case "target":
            try features.add(["compile", "backend", requirement.value])
        case "prefill":
            try features.add(["compile", "prefill"])
        case "layout":
            try features.add(["compile", "layout"])
        case "startup-warmup":
            try features.add(["compile", "startup-warmup"])
        case "generated-kernels":
            try features.add(["compile", "generated-kernels"])
        case "memory":
            if requirement.value.contains("peak") {
                try features.add(["compile", "memory-bound"])
            } else {
                try features.add(["compile", "memory"])
            }
        default:
            try features.add(["compile", requirement.key])
        }
    }

    private func addQuantizationFeature(
        _ rule: SmeltCAMPackageDescriptor.QuantizationRule,
        prefix: String,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        try features.add([prefix, "action", rule.action])
        if let storage = rule.storage {
            try addQuantStorageFeatures(storage, prefix: prefix, to: &features)
        }
        if let calibration = rule.calibration {
            try features.add([prefix, "calibration", calibration.method])
        }
    }

    private func addQuantStorageFeatures(
        _ storage: SmeltCAMPackageDescriptor.QuantStorage,
        prefix: String,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        try features.add([prefix, "storage", storage.storageFormat])
        if let groupSize = storage.groupSize {
            try features.add([prefix, "group", String(groupSize)])
        }
    }

    private func addDescriptorGateGraphFeatures(
        _ gate: SmeltCAMPackageDescriptor.GateContract,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        for subject in gate.requirements.map(\.subject) {
            switch subject {
            case "calibration.gptq.rank":
                try features.add(["quant", "calibration", "rank-gate"])
            case "perplexity.delta":
                try features.add(["quant", "calibration", "perplexity-gate"])
            default:
                break
            }
        }
    }

    private func addGateSubjectFeatures(
        _ gate: SmeltCAMPackageDescriptor.GateContract,
        to features: inout RuntimeAssemblyFeatureSetBuilder
    ) throws {
        for comparison in gate.requirements {
            try features.add(["gate", "subject", comparison.subject])
        }
        for event in [gate.from, gate.to].compactMap({ $0 }) {
            for predicate in event.predicates {
                try features.add(["gate", "subject", predicate.subject])
            }
        }
    }

    private static func calledNodeIDs(in flow: FlowRecord) -> Set<String> {
        Set(flow.phases.flatMap { phase in
            phase.calls.compactMap(\.nodeID)
        })
    }

    private static func importedExportKey(alias: String, exportID: String) -> String {
        "\(alias)\u{1F}\(exportID)"
    }

    private static func isActive(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        membership: ExecutionMembership,
        inputNames: Set<String>,
        outputNames: Set<String>,
        includeInterfaceEndpoints: Bool = true
    ) -> Bool {
        switch endpoint.endpointType {
        case "nodePort":
            return endpoint.nodeID.map { membership.nodeIDs.contains($0) } ?? false
        case "graphValue":
            return endpoint.name.map { membership.graphValueNames.contains($0) } ?? false
        case "moduleInput":
            return includeInterfaceEndpoints
                && (endpoint.name.map { inputNames.contains($0) } ?? false)
        case "moduleOutput":
            return includeInterfaceEndpoints
                && (endpoint.name.map { outputNames.contains($0) } ?? false)
        case "importedPort":
            guard let alias = endpoint.importAlias, let exportID = endpoint.exportID else {
                return false
            }
            return membership.importedExportKeys.contains(
                importedExportKey(alias: alias, exportID: exportID)
            )
        default:
            return false
        }
    }

    @discardableResult
    private static func activate(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        membership: inout ExecutionMembership,
        inputNames: Set<String>,
        outputNames: Set<String>
    ) -> Bool {
        switch endpoint.endpointType {
        case "nodePort":
            guard let nodeID = endpoint.nodeID else { return false }
            return membership.nodeIDs.insert(nodeID).inserted
        case "graphValue":
            guard let name = endpoint.name else { return false }
            return membership.graphValueNames.insert(name).inserted
        case "importedPort":
            guard let alias = endpoint.importAlias, let exportID = endpoint.exportID else {
                return false
            }
            return membership.importedExportKeys.insert(
                importedExportKey(alias: alias, exportID: exportID)
            ).inserted
        case "moduleInput", "moduleOutput":
            return false
        default:
            return false
        }
    }

    private static func routeEdges(
        _ edges: [SmeltCAMPackageDescriptor.GraphEdge],
        decision: Decision,
        flow: FlowRecord,
        calledNodeIDs: Set<String>
    ) -> [SmeltCAMPackageDescriptor.GraphEdge] {
        edges.filter {
            routeEndpoint($0.from, decision: decision, flow: flow, calledNodeIDs: calledNodeIDs)
                || routeEndpoint($0.to, decision: decision, flow: flow, calledNodeIDs: calledNodeIDs)
        }
    }

    private static func routeFeedbackEdges(
        _ edges: [SmeltCAMPackageDescriptor.FeedbackEdge],
        decision: Decision,
        calledNodeIDs: Set<String>
    ) -> [SmeltCAMPackageDescriptor.FeedbackEdge] {
        edges.filter {
            routeEndpoint($0.from, decision: decision, flow: nil, calledNodeIDs: calledNodeIDs)
                || routeEndpoint($0.to, decision: decision, flow: nil, calledNodeIDs: calledNodeIDs)
        }
    }

    private static func routeEndpoint(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        decision: Decision,
        flow: FlowRecord?,
        calledNodeIDs: Set<String>
    ) -> Bool {
        switch endpoint.endpointType {
        case "nodePort":
            return endpoint.nodeID.map { calledNodeIDs.contains($0) } ?? false
        case "moduleInput":
            return endpoint.name.map { name in
                decision.inputPorts.contains { $0.portName == name }
            } ?? false
        case "moduleOutput":
            if endpoint.name.map({ name in
                decision.outputPorts.contains { $0.portName == name }
            }) == true {
                return true
            }
            return flow?.emittedEndpoints.contains {
                endpointSignature($0, nodeOrdinals: [:]) == endpointSignature(endpoint, nodeOrdinals: [:])
            } ?? false
        default:
            return false
        }
    }

    private static func portSignature(_ port: SmeltCAMPackageDescriptor.Port) -> String {
        "\(port.portName):\(valueTypeSignature(port.type)):\(port.optional ? "optional" : "required")"
    }

    private static func valueTypeSignature(_ type: SmeltCAMPackageDescriptor.ValueType) -> String {
        guard !type.attributes.isEmpty else { return type.typeName }
        let attributes = type.attributes
            .map { "\($0.key)=\($0.value)" }
            .sorted()
            .joined(separator: ",")
        return "\(type.typeName){\(attributes)}"
    }

    private static func requirementSignature(
        _ requirement: SmeltCAMPackageDescriptor.Requirement
    ) -> String {
        "\(requirement.key)=\(requirement.value)"
    }

    private static func blockRequirementSignature(
        _ requirement: SmeltCAMPackageDescriptor.BlockRequirement
    ) -> String {
        "\(requirement.key)=\(requirement.value ?? "none"):\(requirement.optional ? "optional" : "required")"
    }

    private static func comparisonSignature(
        _ comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> String {
        let value: String
        if comparison.subject == "package-files" {
            let count = comparison.value
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .count
            value = "count=\(count)"
        } else {
            value = comparison.value
        }
        return "\(comparison.subject):\(comparison.relation):\(value):\(comparison.unit ?? "none")"
    }

    private static func endpointSignature(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        nodeOrdinals: [String: String]
    ) -> String {
        switch endpoint.endpointType {
        case "nodePort":
            return "node:\(endpoint.nodeID.flatMap { nodeOrdinals[$0] } ?? "external"):\(endpoint.portName ?? "none")"
        case "moduleInput", "moduleOutput":
            return "\(endpoint.endpointType):\(endpoint.name ?? "none")"
        case "importedExport":
            return "imported:\(endpoint.exportID ?? "none"):\(endpoint.portName ?? "none")"
        default:
            return [
                endpoint.endpointType,
                endpoint.name ?? "none",
                endpoint.portName ?? "none",
            ].joined(separator: ":")
        }
    }

    private static func flowPhaseSignature(
        _ phase: SmeltCAMPackageDescriptor.FlowPhase,
        nodeOrdinals: [String: String]
    ) -> String {
        let calls = phase.calls.enumerated().map {
            "\($0.offset):\(flowCallSignature($0.element, nodeOrdinals: nodeOrdinals))"
        }.joined(separator: ",")
        return "\(phase.phaseType):\(phase.label ?? "none"):\(calls)"
    }

    private static func flowCallSignature(
        _ call: SmeltCAMPackageDescriptor.FlowCall,
        nodeOrdinals: [String: String]
    ) -> String {
        let target: String
        if let nodeID = call.nodeID {
            target = nodeOrdinals[nodeID] ?? "external"
        } else if let imported = call.imported {
            target = "imported:\(imported.exportID)"
        } else {
            target = "none"
        }
        return "\(call.callType):\(target):\(call.entrypoint ?? "none")"
    }

    private static func stopConditionSignature(
        _ stop: SmeltCAMPackageDescriptor.StopCondition
    ) -> String {
        "\(stop.stopType):\(stop.value.map(String.init) ?? "none")"
    }

    private static func blockSignature(
        _ block: SmeltCAMPackageDescriptor.Block,
        blockOrdinals: [String: String]
    ) -> String {
        let ordinal = blockOrdinals[block.blockID] ?? "external"
        let annotations = block.annotations.map(requirementSignature).sorted().joined(separator: ",")
        let shape = blockShapeSignature(block.shape)
        return "\(ordinal):\(block.operatorName):shape=\(shape):annotations(\(annotations))"
    }

    private static func blockShapeSignature(
        _ shape: SmeltCAMPackageDescriptor.BlockShape
    ) -> String {
        var parts: [String] = []
        if let derivation = shape.derivation {
            parts.append("derivation:authority=\(derivation.authority ?? "none")")
        }
        if let transformer = shape.transformer {
            parts.append("transformer(\(transformerSignature(transformer)))")
        }
        if let codecDecoder = shape.codecDecoder {
            parts.append(
                "codec-decoder(codebooks=\(codecDecoder.codebooks.map(String.init) ?? "none"),"
                    + "streaming=\(codecDecoder.streaming))"
            )
        }
        if let frontend = shape.frontend {
            parts.append("frontend(speaker-conditioning=\(frontend.speakerConditioning))")
        }
        let requirements = shape.requirements
            .map(blockRequirementSignature)
            .sorted()
            .joined(separator: ",")
        parts.append("requirements(\(requirements))")
        return parts.joined(separator: ";")
    }

    private static func transformerSignature(
        _ transformer: SmeltCAMPackageDescriptor.TransformerShape
    ) -> String {
        var parts: [String] = []
        if let hiddenSize = transformer.hiddenSize {
            parts.append("hidden=\(hiddenSize)")
        }
        if let layers = transformer.layers {
            parts.append(
                "layers(count=\(layers.count.map(String.init) ?? "none"),"
                    + "roles=\(layers.roles.joined(separator: "+")),"
                    + "repeat=\(layers.repeatCount.map(String.init) ?? "none"))"
            )
        }
        if let delta = transformer.delta {
            let projections = delta.projections
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            parts.append(
                "delta(heads=\(delta.heads),head-dim=\(delta.headDim),"
                    + "conv=\(delta.convKernel.map(String.init) ?? "none"),"
                    + "projections(\(projections)))"
            )
        }
        if let attention = transformer.attention {
            parts.append("attention(\(attentionSignature(attention)))")
        }
        let roleAttention = transformer.attentionByRole ?? []
        if !roleAttention.isEmpty {
            let roles = roleAttention
                .map { "\($0.role)(\(attentionSignature($0.attention)))" }
                .sorted()
                .joined(separator: ",")
            parts.append("attention-by-role(\(roles))")
        }
        if let ffn = transformer.ffn {
            parts.append("ffn(dim=\(ffn.dim),activation=\(ffn.activation))")
        }
        if let router = transformer.router {
            parts.append("router(top-k=\(router.topK),experts=\(router.experts))")
        }
        if let expert = transformer.expert {
            parts.append("expert(ffn(dim=\(expert.ffn.dim),activation=\(expert.ffn.activation)))")
        }
        if let norm = transformer.norm {
            parts.append(
                "norm(type=\(norm.normType),eps=\(norm.eps ?? "none"),"
                    + "mode=\(norm.mode ?? "none"))"
            )
        }
        if let vocab = transformer.vocab {
            parts.append("vocab(size=\(vocab.size),tied=\(vocab.tiedHead))")
        }
        if let perLayerInput = transformer.perLayerInput {
            parts.append(
                "per-layer-input(hidden=\(perLayerInput.hiddenSize),"
                    + "vocab=\(perLayerInput.vocabSize))"
            )
        }
        if let sharedKVLayers = transformer.sharedKVLayers {
            parts.append("shared-kv-layers=\(sharedKVLayers)")
        }
        if let logitCap = transformer.logitCap {
            parts.append("logit-cap=\(logitCap)")
        }
        return parts.joined(separator: ";")
    }

    private static func attentionSignature(
        _ attention: SmeltCAMPackageDescriptor.AttentionShape
    ) -> String {
        var parts = [
            "q-heads=\(attention.qHeads)",
            "kv-heads=\(attention.kvHeads)",
            "head-dim=\(attention.headDim)",
        ]
        if let rope = attention.rope {
            parts.append("rope=\(rope.ropeType):theta=\(rope.theta.map(String.init) ?? "none")")
        }
        if let qkNormType = attention.qkNormType {
            parts.append("qk-norm=\(qkNormType)")
        }
        if let vNormType = attention.vNormType {
            parts.append("v-norm=\(vNormType)")
        }
        if let window = attention.window {
            parts.append("window=\(window)")
        }
        return parts.joined(separator: ",")
    }

    private static func graphNodeSignature(
        _ node: SmeltCAMPackageDescriptor.GraphNode,
        nodeOrdinals: [String: String],
        blockOrdinals: [String: String]
    ) -> String {
        let ordinal = nodeOrdinals[node.nodeID] ?? "external"
        let blockOrdinal = node.blockID.flatMap { blockOrdinals[$0] } ?? "none"
        let inputs = node.inputs.map(portSignature).sorted().joined(separator: ",")
        let outputs = node.outputs.map(portSignature).sorted().joined(separator: ",")
        let annotations = node.annotations.map(requirementSignature).sorted().joined(separator: ",")
        return "\(ordinal):\(node.implementation):block=\(blockOrdinal):in(\(inputs)):out(\(outputs)):annotations(\(annotations))"
    }

    private static func graphEdgeSignature(
        _ edge: SmeltCAMPackageDescriptor.GraphEdge,
        nodeOrdinals: [String: String]
    ) -> String {
        "\(endpointSignature(edge.from, nodeOrdinals: nodeOrdinals))->"
            + "\(endpointSignature(edge.to, nodeOrdinals: nodeOrdinals)):"
            + "\(edge.valueType.map(valueTypeSignature) ?? "none")"
    }

    private static func feedbackEdgeSignature(
        _ edge: SmeltCAMPackageDescriptor.FeedbackEdge,
        nodeOrdinals: [String: String]
    ) -> String {
        "\(endpointSignature(edge.from, nodeOrdinals: nodeOrdinals))->"
            + "\(endpointSignature(edge.to, nodeOrdinals: nodeOrdinals))"
    }

    private static func artifactSignature(
        _ artifact: SmeltCAMPackageDescriptor.ArtifactRequirement
    ) -> String {
        "\(artifact.role):\(artifact.required ? "required" : "optional")"
    }

    private static func gateContractSignature(
        _ record: GateContractRecord,
        nodeOrdinals: [String: String]
    ) -> [String] {
        let gate = record.contract
        var lines: [String] = []
        if let from = gate.from {
            lines.append("gate:from:\(gateEventSignature(from, nodeOrdinals: nodeOrdinals))")
        }
        if let to = gate.to {
            lines.append("gate:to:\(gateEventSignature(to, nodeOrdinals: nodeOrdinals))")
        }
        lines += gate.requirements.map {
            "gate:require:\(comparisonSignature($0))"
        }.sorted()
        lines += gate.evidence.map {
            "gate:evidence:\(evidenceSignature($0))"
        }.sorted()
        return lines
    }

    private static func gateEventSignature(
        _ event: SmeltCAMPackageDescriptor.GateEvent,
        nodeOrdinals: [String: String]
    ) -> String {
        let endpoint = event.endpoint.map {
            endpointSignature($0, nodeOrdinals: nodeOrdinals)
        } ?? "none"
        let predicates = event.predicates
            .map(comparisonSignature)
            .sorted()
            .joined(separator: ",")
        return "\(event.eventType):flow=\(event.flowID == nil ? "none" : "selected")"
            + ":export=\(event.exportID == nil ? "none" : "selected")"
            + ":endpoint=\(endpoint):signal=\(event.signal ?? "none"):predicates(\(predicates))"
    }

    private static func evidenceSignature(
        _ evidence: SmeltCAMPackageDescriptor.EvidenceRequirement
    ) -> String {
        let storage = evidence.storage.map(quantStorageSignature) ?? "none"
        return "\(evidence.evidenceType):source-dtypes=\(evidence.sourceDTypes.count):storage=\(storage)"
    }

    private static func quantStorageSignature(
        _ storage: SmeltCAMPackageDescriptor.QuantStorage
    ) -> String {
        "\(storage.storageFormat):group=\(storage.groupSize.map(String.init) ?? "none"):compute=\(storage.computeDType ?? "none")"
    }

    private static func quantizationSignature(
        _ rule: SmeltCAMPackageDescriptor.QuantizationRule
    ) -> String {
        let storage = rule.storage.map(quantStorageSignature) ?? "none"
        let calibration = rule.calibration == nil ? "none" : "present"
        return "\(rule.action):storage=\(storage):priority=\(rule.priority.map(String.init) ?? "none")"
            + ":calibration=\(calibration):resolution=\(rule.resolution)"
    }

    private static func sourceQuantizationSignature(
        _ rule: SmeltCAMPackageDescriptor.SourceQuantizationRule
    ) -> String {
        let storage = rule.storage.map(quantStorageSignature) ?? "none"
        let evidence = rule.evidence.map(evidenceSignature).sorted().joined(separator: ",")
        return "\(rule.action):source-dtypes=\(rule.sourceDTypes.sorted().joined(separator: ","))"
            + ":storage=\(storage):target=\(rule.targetDType ?? "none"):evidence(\(evidence))"
    }

    private func exportMatches(
        _ export: ExportRecord,
        request: SmeltCAMCapabilityRequest
    ) -> Bool {
        let requiredInputs = export.inputPorts.filter { !$0.optional }
        guard ports(
            requiredInputs,
            satisfy: request.requiredInputShapes,
            names: request.requiredInputNames,
            allowAdditionalPorts: request.allowAdditionalRequiredInputs
        ) else {
            return false
        }
        guard ports(
            export.outputPorts.filter { !$0.optional },
            satisfy: request.requiredOutputShapes,
            names: request.requiredOutputNames,
            allowAdditionalPorts: request.allowAdditionalOutputs
        ) else {
            return false
        }
        if !request.requiredAnyExportFacts.isEmpty || !request.requiredAllExportFacts.isEmpty {
            let exportFacts = Set(export.capabilities)
            if !request.requiredAnyExportFacts.isEmpty {
                guard !exportFacts.isDisjoint(with: request.requiredAnyExportFacts) else {
                    return false
                }
            }
            if !request.requiredAllExportFacts.isEmpty {
                guard request.requiredAllExportFacts.isSubset(of: exportFacts) else {
                    return false
                }
            }
        }
        guard packageInventoryIncludes(request.requiredPackageFiles) else {
            return false
        }
        guard compileRequirementsInclude(request.requiredCompileRequirementKeys) else {
            return false
        }
        guard compileRequirementsSatisfy(request.requiredCompileRequirements) else {
            return false
        }
        guard routeTransformerLayerRolesAvoid(
            export: export,
            forbiddenRoles: request.forbiddenRouteTransformerLayerRoles
        ) else {
            return false
        }
        if !request.requiredGateSubjects.isEmpty || !request.requiredGateObservations.isEmpty {
            let gateContracts = matchedGateContracts(export: export, request: request)
            guard !gateContracts.isEmpty else {
                return false
            }
            guard gateContractsCoverRequiredObservations(
                gateContracts,
                request: request,
                export: export
            ) else { return false }
        }
        return true
    }

    private func selectedPorts(
        _ ports: [SmeltCAMPackageDescriptor.Port],
        satisfy shapes: [SmeltCAMCapabilityRequest.PortShape],
        names: [String] = []
    ) -> [SmeltCAMPackageDescriptor.Port]? {
        guard names.isEmpty || names.count == shapes.count else { return nil }
        guard let shape = shapes.first else { return [] }
        let requestedName = names.first
        for (index, port) in ports.enumerated()
            where shape.matches(port)
                && (requestedName.map { port.portName == $0 } ?? true) {
            var remainingPorts = ports
            remainingPorts.remove(at: index)
            if let rest = selectedPorts(
                remainingPorts,
                satisfy: Array(shapes.dropFirst()),
                names: names.isEmpty ? [] : Array(names.dropFirst())
            ) {
                return [port] + rest
            }
        }
        return nil
    }

    private func ports(
        _ ports: [SmeltCAMPackageDescriptor.Port],
        satisfy shapes: [SmeltCAMCapabilityRequest.PortShape],
        names: [String],
        allowAdditionalPorts: Bool
    ) -> Bool {
        guard allowAdditionalPorts || ports.count == shapes.count else {
            return false
        }
        return selectedPorts(ports, satisfy: shapes, names: names) != nil
    }

    private func validatePortBindings(_ request: SmeltCAMCapabilityRequest) throws {
        try validatePortBindingNames(
            request.requiredInputNames,
            shapes: request.requiredInputShapes,
            requestName: request.name,
            role: "input"
        )
        try validatePortBindingNames(
            request.requiredOutputNames,
            shapes: request.requiredOutputShapes,
            requestName: request.name,
            role: "output"
        )
    }

    private func validatePortBindingNames(
        _ names: [String],
        shapes: [SmeltCAMCapabilityRequest.PortShape],
        requestName: String,
        role: String
    ) throws {
        if names.isEmpty {
            guard Set(shapes).count == shapes.count else {
                throw SmeltCAMPackageCapabilitiesError.invalidRequest(
                    requestName,
                    "duplicate \(role) shapes require explicit \(role) port names"
                )
            }
            return
        }
        guard names.count == shapes.count else {
            throw SmeltCAMPackageCapabilitiesError.invalidRequest(
                requestName,
                "\(role) port name count must match \(role) shape count"
            )
        }
        guard !names.contains(where: \.isEmpty) else {
            throw SmeltCAMPackageCapabilitiesError.invalidRequest(
                requestName,
                "\(role) port names must be non-empty"
            )
        }
        guard Set(names).count == names.count else {
            throw SmeltCAMPackageCapabilitiesError.invalidRequest(
                requestName,
                "\(role) port names must be unique"
            )
        }
    }

    private func matchedGateContracts(
        export: ExportRecord,
        request: SmeltCAMCapabilityRequest
    ) -> [GateContractRecord] {
        let inventoryGateIDs = packageInventoryGateIDs(containing: request.requiredPackageFiles)
        let gateIDs = (export.gateIDs + inventoryGateIDs).reduce(into: [String]()) {
            result, gateID in
            if !result.contains(gateID) {
                result.append(gateID)
            }
        }
        return gateIDs.compactMap { gateID in
            guard let gate = gateContractsByID[gateID] else { return nil }
            if !request.requiredGateSubjects.isEmpty {
                let subjects = Set(gate.requirements.map(\.subject))
                guard request.requiredGateSubjects.isSubset(of: subjects) else { return nil }
            }
            if !request.requiredGateObservations.isEmpty {
                guard request.requiredGateObservations.contains(where: {
                    gateSatisfiesObservation($0, gate: gate, export: export)
                }) else {
                    return nil
                }
            }
            return GateContractRecord(
                gateID: gateID,
                exportID: export.exportID,
                flowID: export.boundFlowID,
                contract: gate
            )
        }
    }

    private func compileRequirementsInclude(_ requiredKeys: Set<String>) -> Bool {
        guard !requiredKeys.isEmpty else { return true }
        let keys = Set(compileRequirements.map(\.key))
        return requiredKeys.isSubset(of: keys)
    }

    private func compileRequirementsSatisfy(
        _ required: [SmeltCAMCapabilityRequest.RequirementShape]
    ) -> Bool {
        required.allSatisfy { shape in
            compileRequirements.contains { requirement in
                requirementSatisfies(requirement, shape: shape)
            }
        }
    }

    private func requirementSatisfies(
        _ requirement: SmeltCAMPackageDescriptor.Requirement,
        shape: SmeltCAMCapabilityRequest.RequirementShape
    ) -> Bool {
        guard requirement.key == shape.key else { return false }
        switch shape.valueShape {
        case .any:
            return true
        case let .containsTerm(term):
            return requirementValueContainsTerm(requirement.value, term)
        case .prefillGPUBatch:
            return requirementValueMatchesPrefillGPUBatch(requirement.value)
        case .prefillAllLogitsGPUBatch:
            return requirementValueMatchesPrefillAllLogitsGPUBatch(requirement.value)
        }
    }

    private func requirementValueContainsTerm(_ value: String, _ term: String) -> Bool {
        value
            .split { $0 == "," || $0 == " " || $0 == "\t" || $0 == "\n" }
            .contains { $0 == term }
    }

    private func requirementValueMatchesPrefillGPUBatch(_ value: String) -> Bool {
        SmeltCAMCapabilityRequest.RequirementShape.prefillGPUBatchSize(value) != nil
    }

    private func requirementValueMatchesPrefillAllLogitsGPUBatch(_ value: String) -> Bool {
        SmeltCAMCapabilityRequest.RequirementShape.prefillAllLogitsGPUBatchSize(value) != nil
    }

    private func routeTransformerLayerRolesAvoid(
        export: ExportRecord,
        forbiddenRoles: Set<String>
    ) -> Bool {
        guard !forbiddenRoles.isEmpty else { return true }
        guard let calledBlocks = localCalledBlocks(export: export) else { return false }
        for block in calledBlocks {
            guard let transformer = block.shape.transformer,
                  let roles = transformer.layers?.roles,
                  !roles.isEmpty
            else {
                return false
            }
            guard Set(roles).isDisjoint(with: forbiddenRoles) else {
                return false
            }
            if forbiddenRoles.contains("delta"), transformer.delta != nil {
                return false
            }
            guard roles.allSatisfy({ transformerSupportsLayerRole($0, transformer: transformer) }) else {
                return false
            }
        }
        return true
    }

    private func transformerSupportsLayerRole(
        _ role: String,
        transformer: SmeltCAMPackageDescriptor.TransformerShape
    ) -> Bool {
        if role == "delta" {
            return transformer.delta != nil
        }
        if role == "attention" {
            return transformer.attention != nil
        }
        return transformer.attentionByRole?.contains { $0.role == role } ?? false
    }

    private func localCalledBlocks(export: ExportRecord) -> [SmeltCAMPackageDescriptor.Block]? {
        guard let flow = flows.first(where: { $0.flowID == export.boundFlowID }) else {
            return nil
        }
        let nodesByID = Dictionary(uniqueKeysWithValues: graphNodes.map { ($0.nodeID, $0) })
        let blocksByID = Dictionary(uniqueKeysWithValues: blocks.map { ($0.blockID, $0) })
        var calledBlocks: [SmeltCAMPackageDescriptor.Block] = []
        for phase in flow.phases {
            for call in phase.calls where call.callType == "node" {
                guard let nodeID = call.nodeID,
                      let node = nodesByID[nodeID]
                else { continue }
                guard node.implementation == "compiled" else { continue }
                guard let blockID = node.blockID else { return nil }
                guard let block = blocksByID[blockID] else { return nil }
                calledBlocks.append(block)
            }
        }
        return calledBlocks
    }

    private func packageInventoryIncludes(_ requiredFiles: Set<String>) -> Bool {
        guard !requiredFiles.isEmpty else { return true }
        return !packageInventoryGateIDs(containing: requiredFiles).isEmpty
    }

    private func packageInventoryGateIDs(containing requiredFiles: Set<String>) -> [String] {
        guard !requiredFiles.isEmpty else { return [] }
        return gateContracts.compactMap { gate in
            for requirement in gate.requirements {
                guard let files = packageFiles(from: requirement),
                      requiredFiles.isSubset(of: files) else {
                    continue
                }
                return gate.gateID
            }
            return nil
        }
    }

    private func packageFiles(
        from comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> Set<String>? {
        guard comparison.subject == "package-files",
              comparison.relation == "include" else {
            return nil
        }
        let files = comparison.value
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return Set(files)
    }

    private func gateSatisfiesObservation(
        _ observation: SmeltCAMCapabilityRequest.GateObservation,
        gate: SmeltCAMPackageDescriptor.GateContract,
        export: ExportRecord
    ) -> Bool {
        if !observation.requirementSubjects.isEmpty {
            let subjects = Set(gate.requirements.map(\.subject))
            guard observation.requirementSubjects.isSubset(of: subjects) else { return false }
        }
        guard comparisons(gate.requirements, satisfy: observation.requirementComparisons) else {
            return false
        }
        if observation.toEventType == "gate" {
            return true
        }
        if observation.fromEventType != nil || observation.fromFlow != nil {
            guard let from = gate.from else { return false }
            if let fromEventType = observation.fromEventType {
                guard from.eventType == fromEventType else { return false }
            }
            if let fromFlow = observation.fromFlow {
                switch fromFlow {
                case .selectedExportFlow:
                    guard from.flowID == export.boundFlowID else { return false }
                }
            }
        }
        guard let to = gate.to else { return false }
        guard to.eventType == observation.toEventType else { return false }
        switch observation.toFlow {
        case .selectedExportFlow:
            guard to.flowID == export.boundFlowID else { return false }
        }
        guard let endpoint = to.endpoint,
              endpoint.endpointType == "moduleOutput",
              let name = endpoint.name else {
            return false
        }
        let matchingOutputs = export.outputPorts.filter { observation.outputShape.matches($0) }
        let output: SmeltCAMPackageDescriptor.Port
        if let outputPortName = observation.outputPortName {
            guard let named = matchingOutputs.first(where: { $0.portName == outputPortName }) else {
                return false
            }
            output = named
        } else {
            guard matchingOutputs.count == 1, let single = matchingOutputs.first else {
                return false
            }
            output = single
        }
        guard output.portName == name else { return false }
        if !observation.predicateSubjects.isEmpty {
            let subjects = Set(to.predicates.map(\.subject))
            guard observation.predicateSubjects.isSubset(of: subjects) else { return false }
        }
        if observation.requireFormatPredicateMatchingOutput {
            guard let expectedFormat = formatPredicateValue(for: output),
                  to.predicates.contains(where: {
                      $0.subject == "format"
                          && $0.relation == "=="
                          && $0.value == expectedFormat
                          && $0.unit == nil
                  }) else {
                return false
            }
        }
        return comparisons(to.predicates, satisfy: observation.predicateComparisons)
    }

    private func gateContractsCoverRequiredObservations(
        _ gateContracts: [GateContractRecord],
        request: SmeltCAMCapabilityRequest,
        export: ExportRecord
    ) -> Bool {
        guard request.requireAllGateObservations else { return true }
        return request.requiredGateObservations.allSatisfy { observation in
            gateContracts.contains {
                gateSatisfiesObservation(observation, gate: $0.contract, export: export)
            }
        }
    }

    private func formatPredicateValue(
        for port: SmeltCAMPackageDescriptor.Port
    ) -> String? {
        guard port.type.typeName == "pcm",
              let dtype = port.type.attributes["dtype"],
              let rate = port.type.attributes["rate"] else {
            return nil
        }
        return "pcm \(dtype) \(rate)"
    }

    private func comparisons(
        _ comparisons: [SmeltCAMPackageDescriptor.Comparison],
        satisfy required: [SmeltCAMCapabilityRequest.ComparisonShape]
    ) -> Bool {
        required.allSatisfy { shape in
            comparisons.contains { shape.matches($0) }
        }
    }

    private static func descriptorIdentityToken(
        attributes: [FileAttributeKey: Any],
        size: Int
    ) -> String {
        let inode = (attributes[.systemFileNumber] as? NSNumber)?.stringValue ?? "unknown"
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "inode:\(inode):size:\(size):mtime:\(String(format: "%.6f", modified))"
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
