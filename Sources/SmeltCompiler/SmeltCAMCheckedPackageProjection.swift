import CryptoKit
import Foundation
import SmeltSchema

public enum SmeltCAMCheckedPackageProjectionError: Error, CustomStringConvertible, Equatable {
    case unsupported(String)

    public var description: String {
        switch self {
        case .unsupported(let reason):
            return "checked CAM package projection: \(reason)"
        }
    }
}

public struct SmeltCAMCheckedPackageProjection: Sendable {
    public let spec: SmeltPackageSpec
    public let packageProjectionID: String
    public let packageProjectionVersion: Int
    public let buildCommandCovered: Bool
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let descriptorVersion: Int
    public let descriptorGraphSignatureSHA256: String
    public let projectedPackageSpecSHA256: String
    public let packageFiles: [String]
}

struct CheckedPrefillCompileConstraint: Sendable, Equatable {
    let batch: Int
    let emitAllLogits: Bool
    let verifyArgmax: Bool
    let verifyTokenCapacity: Int
}

private struct CheckedPackageProjectionProfile: Sendable {
    let id: String
    let version: Int
    let descriptorVersion: Int
    let semanticSHA256: String
    let exportABISHA256: String
    let descriptorGraphSignatureSHA256: String
    let buildCommandCovered: Bool
    let flowID: String
    let packageFiles: [String]
    let sources: [CheckedProjectionSource]
    let modelSourceID: String
    let assembly: CheckedProjectionAssembly
    let textPolicyBridge: CheckedProjectionTextPolicyBridge
    let validation: SmeltPackageSpec.Validation?
    let quantization: CheckedProjectionQuantization
    let prefill: CheckedProjectionPrefillPolicy
}

private struct CheckedResolvedPackageProjectionProfile: Sendable {
    let id: String
    let version: Int
    let descriptorVersion: Int
    let semanticSHA256: String
    let exportABISHA256: String
    let descriptorGraphSignatureSHA256: String
    let buildCommandCovered: Bool
    let packageFiles: [String]
    let materializer: CheckedProjectionMaterializer
}

private enum CheckedProjectionMaterializer: Sendable {
    case loweredTransformer(CheckedPackageProjectionProfile)
    case sidecarManifest(CheckedDerivedAudioPackageProjectionProfile)
}

private struct CheckedDerivedAudioPackageProjectionProfile: Sendable {
    let id: String
    let version: Int
    let descriptorVersion: Int
    let semanticSHA256: String
    let exportABISHA256: String
    let descriptorGraphSignatureSHA256: String
    let buildCommandCovered: Bool
    let flowID: String
    let packageFiles: [String]
    let sources: [CheckedProjectionSource]
    let modelSourceID: String
    let manifest: CheckedProjectionAudioManifestBridge
    let graph: CheckedProjectionAudioGraph
    let tensorMaps: [SmeltCAMIR.TensorMap]
    let quantization: CheckedProjectionAudioQuantization
}

private struct CheckedProjectionAudioManifestBridge: Sendable {
    let modelName: String
    let tokenizerFiles: [String]
    let sidecarPaths: [String]
    let pageSize: Int
    let maxTokens: Int
    let eosTokens: [Int32]
    let performanceGate: String
    let structureProfileID: String
}

private struct CheckedProjectionAudioGraph: Sendable {
    let blocks: [CheckedProjectionAudioBlock]
    let graphNodes: [CheckedProjectionAudioNode]
    let graphEdges: [CheckedProjectionAudioEdge]
    let feedbackEdge: CheckedProjectionAudioFeedbackEdge
    let setupCalls: [String]
    let stepCallsByLabel: [String: [String]]
    let emitNode: String
    let emitPort: String
    let stops: [CheckedProjectionAudioStop]
}

private struct CheckedProjectionAudioBlock: Sendable {
    let id: String
    let operatorName: SmeltCAMIR.BlockOperator
    let derivation: SmeltCAMIR.ShapeDerivation
    let requirements: [CheckedProjectionBlockRequirement]
}

private struct CheckedProjectionAudioNode: Sendable {
    let id: String
    let implementation: SmeltCAMIR.GraphImplementation
    let block: String?
    let inputs: [SmeltCAMIR.Port]
    let outputs: [SmeltCAMIR.Port]
    let annotations: [SmeltCAMIR.Constraint]
}

private struct CheckedProjectionAudioEdge: Sendable {
    let from: SmeltCAMIR.EndpointRef
    let to: SmeltCAMIR.EndpointRef
    let type: SmeltCAMIR.TypeRef
}

private struct CheckedProjectionAudioFeedbackEdge: Sendable {
    let from: SmeltCAMIR.EndpointRef
    let to: SmeltCAMIR.EndpointRef
}

private enum CheckedProjectionAudioStop: Sendable, Equatable {
    case codecEOS
    case maxFrames(Int)
    case hostCancel
}

private struct CheckedProjectionAudioQuantization: Sendable {
    let defaultStorage: CheckedProjectionQuantStorage
    let preservedPatterns: [String]
}

private enum CheckedDerivedAudioRuntimeContract {
    static let codebooks = 16
    static let residualSlots = codebooks - 1
}

private struct CheckedProjectionSource: Sendable {
    let id: String
    let kind: String
    let locator: String
    let revision: String?
    let checkpointMap: String?

    init(
        id: String,
        kind: String,
        locator: String,
        revision: String?,
        checkpointMap: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.locator = locator
        self.revision = revision
        self.checkpointMap = checkpointMap
    }
}

private enum CheckedProjectionAssembly: Sendable {
    case explicitCAMShape(CheckedProjectionTransformer)
    case roleAttentionCAMShape
}

private struct CheckedProjectionTransformer: Sendable {
    let blockID: String
    let operatorName: SmeltCAMIR.BlockOperator
    let staticSeqCapacity: Int
    let ropeDim: Int
    let ropeKind: SmeltCAMIR.RopeKind
    let gatedQ: Bool
    let qkvBias: Bool
    let attentionQKNorm: SmeltCAMIR.NormKind?
    let attentionQKNormMode: SmeltCAMIR.NormMode
    let normKind: SmeltCAMIR.NormKind
    let normMode: SmeltCAMIR.NormMode
    let normEpsilon: String
    let rmsEpsilon: Float
    let modelNormMode: SmeltNormMode
    let ffnActivation: SmeltCAMIR.Activation
    let requirements: [CheckedProjectionBlockRequirement]
}

private struct CheckedProjectionBlockRequirement: Sendable, Equatable {
    let key: String
    let value: String?
    let optional: Bool

    init(key: String, value: String? = nil, optional: Bool = false) {
        self.key = key
        self.value = value
        self.optional = optional
    }
}

private struct CheckedProjectionQuantStorage: Sendable, Equatable {
    let format: SmeltCAMIR.QuantStorageFormat
    let groupSize: Int?
}

private struct CheckedProjectionQuantization: Sendable {
    let strategy: SmeltQuantStrategy
    let defaultStorage: CheckedProjectionQuantStorage
    let quantizedPatterns: [String]
    let preservedPatterns: [String]
    let storedPatterns: [CheckedProjectionStoredPattern]
    let quantizesEmbedding: Bool
}

private struct CheckedProjectionStoredPattern: Sendable, Equatable {
    let pattern: String
    let storage: CheckedProjectionQuantStorage
}

private enum CheckedProjectionPrefillPolicy: Sendable {
    case required(CheckedProjectionPrefill)
    case none(rejectedCompileKeys: [String], rejectedSourceIDs: [String], rejectedGraphNodeIDs: [String])
}

private struct CheckedProjectionPrefill: Sendable {
    let compileKey: String
    let engine: String
    let modelPath: String
    let cachePath: String
    let maxBatchSize: Int
    let handoffNames: [String]
    let rejectedSourceIDs: [String]
    let rejectedGraphNodeIDs: [String]
}

// Temporary text lowerer adapter: copied into SmeltModelIR for today's
// lowerer/manifest contract, not CAM identity or checked profile selection.
private struct CheckedProjectionTextPolicyBridge: Sendable {
    let tokenizerInputName: String
    let tokenizerInputType: SmeltCAMIR.TypeRef
    let capabilityRequests: CheckedProjectionTextCapabilityRequests
    let setup: CheckedProjectionSetup
}

private struct CheckedProjectionResolvedTextPolicyBridge: Sendable {
    let chatTemplate: String
    let thinkingPolicy: SmeltThinkingPolicy
    let toolTranscriptCodec: String?
    let promptStateRestoreMode: SmeltPromptStateRestoreMode
}

private struct CheckedProjectionTextCapabilityRequests: Sendable {
    let run: SmeltCAMCapabilityRequest
    let benchmark: SmeltCAMCapabilityRequest

    static let singleText = CheckedProjectionTextCapabilityRequests(
        run: .runText,
        benchmark: .benchDecode
    )

    static func exactTwoText(inputNames: [String]) -> CheckedProjectionTextCapabilityRequests {
        CheckedProjectionTextCapabilityRequests(
            run: .exactTextToText(
                name: "run text with two required text inputs",
                requiredTextInputCount: 2,
                requiredInputNames: inputNames,
                requiredAnyExportFacts: ["run.generate"]
            ),
            benchmark: .exactTextToText(
                name: "bench text with two required text inputs",
                requiredTextInputCount: 2,
                requiredInputNames: inputNames,
                requiredAnyExportFacts: ["run.generate"],
                requiredGateObservations: [SmeltCAMCapabilityRequest.firstTextOutputObservation()]
            )
        )
    }
}

private enum CheckedProjectionSetup: Sendable {
    case tokenizerOnly
    case promptBuilder(CheckedProjectionPromptBuilder)
}

private struct CheckedProjectionPromptBuilder: Sendable {
    let id: String
    let template: String
    let inputNames: [String]
    let outputName: String
}

public enum SmeltCAMCheckedPackageProjector {
    private static let checkedProjectionConsumedFeatureSet: [String] = []

    static func parsePrefillCompileConstraint(
        _ value: String,
        engine: String
    ) throws -> CheckedPrefillCompileConstraint {
        let parts = value.split(separator: " ").map(String.init)
        try require(parts.first == engine, "compile prefill constraint drifted")
        var batchIndex = 1
        let mode = parts.indices.contains(batchIndex) ? parts[batchIndex] : ""
        let emitAllLogits = mode == "all-logits"
        let verifyArgmax = mode == "verify-argmax"
        if emitAllLogits || verifyArgmax { batchIndex += 1 }
        try require(
            parts.indices.contains(batchIndex),
            "prefill batch missing"
        )
        try require(parts[batchIndex] == "batch", "compile prefill constraint drifted")
        let batchText = try requireValue(
            parts.indices.contains(batchIndex + 1) ? parts[batchIndex + 1] : nil,
            "prefill batch missing"
        )
        guard let batch = Int(batchText) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "prefill batch is not an integer"
            )
        }
        let trailing = Array(parts.dropFirst(batchIndex + 2))
        let verifyTokenCapacity: Int
        if trailing.isEmpty {
            verifyTokenCapacity = verifyArgmax ? min(batch, 8) : 0
        } else {
            try require(
                verifyArgmax && trailing.count == 2
                    && trailing[0] == "transaction",
                "compile prefill constraint drifted"
            )
            guard let capacity = Int(trailing[1]),
                  capacity > 0,
                  capacity <= batch
            else {
                throw SmeltCAMCheckedPackageProjectionError.unsupported(
                    "prefill transaction capacity must be in [1, batch]"
                )
            }
            verifyTokenCapacity = capacity
        }
        return CheckedPrefillCompileConstraint(
            batch: batch,
            emitAllLogits: emitAllLogits,
            verifyArgmax: verifyArgmax,
            verifyTokenCapacity: verifyTokenCapacity
        )
    }

    private static func textValidation(
        parityFixture: String,
        modelName: String
    ) -> SmeltPackageSpec.Validation {
        SmeltPackagePerformanceProfiles.validation(
            parityFixture: parityFixture,
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: modelName
        )
    }

    private static let checkedPackageProjectionProfiles: [CheckedPackageProjectionProfile] = []

    private static let checkedDerivedAudioPackageProjectionProfiles: [CheckedDerivedAudioPackageProjectionProfile] = []

    public static func project(
        cam rawCAM: SmeltCAMIR,
        artifactRoot: String
    ) throws -> SmeltCAMCheckedPackageProjection {
        guard !artifactRoot.isEmpty else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "artifact root is required"
            )
        }

        let cam = try rawCAM.validated()
        let semanticSHA256 = try cam.semanticSHA256()
        let exportABISHA256 = try cam.exportABISHA256()
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let graphSignatureSHA256 = sha256Hex(
            Data(descriptor.graphSignature.joined(separator: "\n").utf8)
        )
        let admission = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: checkedProjectionConsumedFeatureSet
        )
        if admission.hasUnsupportedFeatures {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "unsupported CAM package projection features: "
                    + admission.unsupportedDiagnostic
            )
        }
        guard let profile = try checkedPackageProjectionProfile(
            cam: cam,
            descriptor: descriptor,
            descriptorVersion: descriptor.descriptorVersion,
            semanticSHA256: semanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorGraphSignatureSHA256: graphSignatureSHA256
        ) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                profileMissDiagnostic(
                    descriptorVersion: descriptor.descriptorVersion,
                    semanticSHA256: semanticSHA256,
                    exportABISHA256: exportABISHA256,
                    descriptorGraphSignatureSHA256: graphSignatureSHA256,
                    descriptor: descriptor
                )
            )
        }

        let spec = try projectSpec(
            from: cam,
            artifactRoot: artifactRoot,
            profile: profile
        )
        let plan = try SmeltPackageResolvedPlan.resolve(spec)
        let packageFiles = plan.packageFiles.map(\.path).sorted()
        let inventoryFiles = try packageInventoryFiles(from: cam)
        try require(
            inventoryFiles == profile.packageFiles,
            "\(profile.id) package inventory drifted"
        )
        try require(
            packageFiles == profile.packageFiles,
            "projected files \(packageFiles) do not match checked package projection profile \(profile.packageFiles)"
        )

        return SmeltCAMCheckedPackageProjection(
            spec: spec,
            packageProjectionID: profile.id,
            packageProjectionVersion: profile.version,
            buildCommandCovered: profile.buildCommandCovered,
            camSemanticSHA256: semanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorVersion: descriptor.descriptorVersion,
            descriptorGraphSignatureSHA256: graphSignatureSHA256,
            projectedPackageSpecSHA256: try sha256Hex(canonicalJSONData(spec)),
            packageFiles: packageFiles
        )
    }

    public static func sourceModelIR(cam rawCAM: SmeltCAMIR) throws -> SmeltModelIR {
        let cam = try rawCAM.validated()
        let semanticSHA256 = try cam.semanticSHA256()
        let exportABISHA256 = try cam.exportABISHA256()
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let graphSignatureSHA256 = sha256Hex(
            Data(descriptor.graphSignature.joined(separator: "\n").utf8)
        )
        let admission = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: checkedProjectionConsumedFeatureSet
        )
        if admission.hasUnsupportedFeatures {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "unsupported CAM source model features: "
                    + admission.unsupportedDiagnostic
            )
        }
        guard let profile = try checkedPackageProjectionProfile(
            cam: cam,
            descriptor: descriptor,
            descriptorVersion: descriptor.descriptorVersion,
            semanticSHA256: semanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorGraphSignatureSHA256: graphSignatureSHA256
        ) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                profileMissDiagnostic(
                    descriptorVersion: descriptor.descriptorVersion,
                    semanticSHA256: semanticSHA256,
                    exportABISHA256: exportABISHA256,
                    descriptorGraphSignatureSHA256: graphSignatureSHA256,
                    descriptor: descriptor
                )
            )
        }
        guard profile.buildCommandCovered else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "CAM projection '\(profile.id)' is not source-build covered"
            )
        }
        guard case .loweredTransformer(let materializerProfile) = profile.materializer else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "CAM projection '\(profile.id)' does not lower to a compiler source model"
            )
        }
        return try modelIR(from: cam, profile: materializerProfile)
    }

    private static func profileMissDiagnostic(
        descriptorVersion: Int,
        semanticSHA256: String,
        exportABISHA256: String,
        descriptorGraphSignatureSHA256: String,
        descriptor: SmeltCAMPackageDescriptor
    ) -> String {
        let admission = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: checkedProjectionConsumedFeatureSet
        )
        if admission.hasUnsupportedFeatures {
            return "unsupported CAM package projection features: "
                + admission.unsupportedDiagnostic
        }
        return "no checked package projection profile for descriptor v\(descriptorVersion), "
            + "CAM semantic hash '\(semanticSHA256)', export ABI hash '\(exportABISHA256)', "
            + "descriptor graph signature hash '\(descriptorGraphSignatureSHA256)'"
    }

    private static func derivedTextProjectionProfile(
        from cam: SmeltCAMIR,
        descriptorVersion: Int,
        semanticSHA256: String,
        exportABISHA256: String,
        descriptorGraphSignatureSHA256: String
    ) throws -> CheckedPackageProjectionProfile? {
        guard let export = try textProjectionExport(from: cam) else { return nil }
        guard cam.blocks.count == 1,
              let block = cam.blocks.first,
              block.operatorName == .transformer,
              block.shape.transformer != nil
        else {
            return nil
        }

        let flow = try exactlyOne(cam.flows, "text projection flow")
        let modelSource = try primaryModelSource(from: cam)
        let packageFiles = try packageInventoryFiles(from: cam)
        let textPolicyBridge = try derivedTextPolicyBridge(from: cam, export: export, flow: flow)
        try validateTextProjectionSurface(
            cam: cam,
            export: export,
            block: block,
            textPolicyBridge: textPolicyBridge,
            modelSource: modelSource
        )
        let quantization = try derivedTextQuantization(from: cam)
        let prefill = try derivedTextPrefillPolicy(from: cam, block: block, flow: flow)
        let assembly = try derivedTextAssembly(from: block, prefill: prefill, flow: flow)
        let projectionID = try derivedTextProjectionID(
            export: export,
            block: block,
            quantization: quantization,
            prefill: prefill,
            textPolicyBridge: textPolicyBridge
        )

        return CheckedPackageProjectionProfile(
            id: projectionID,
            version: 1,
            descriptorVersion: descriptorVersion,
            semanticSHA256: semanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256,
            buildCommandCovered: true,
            flowID: flow.id,
            packageFiles: packageFiles,
            sources: cam.sources.map {
                CheckedProjectionSource(
                    id: $0.id,
                    kind: $0.kind,
                    locator: $0.locator,
                    revision: $0.revision,
                    checkpointMap: $0.checkpointMap
                )
            },
            modelSourceID: modelSource.id,
            assembly: assembly,
            textPolicyBridge: textPolicyBridge,
            validation: textValidation(
                parityFixture: modelSource.locator,
                modelName: modelSource.locator
            ),
            quantization: quantization,
            prefill: prefill
        )
    }

    private static func textProjectionExport(
        from cam: SmeltCAMIR
    ) throws -> SmeltCAMIR.Export? {
        guard cam.exports.count == 1 else { return nil }
        let export = try exactlyOne(cam.exports, "text projection export")
        let outputs = export.outputs
        guard outputs.count == 1,
              outputs.first.map(isTextUTF8) == true
        else {
            return nil
        }
        guard export.inputs.allSatisfy({ !$0.optional }),
              export.outputs.allSatisfy({ !$0.optional })
        else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "text projection exports do not support optional ports"
            )
        }
        guard [1, 2].contains(export.inputs.count),
              export.inputs.allSatisfy(isTextUTF8)
        else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "text projection exports require one or two required utf8 text inputs"
            )
        }
        return export
    }

    private static func isTextUTF8(_ port: SmeltCAMIR.Port) -> Bool {
        port.type == .init("text", attributes: ["encoding": "utf8"])
    }

    private static func validateTextProjectionSurface(
        cam: SmeltCAMIR,
        export: SmeltCAMIR.Export,
        block: SmeltCAMIR.Block,
        textPolicyBridge: CheckedProjectionTextPolicyBridge,
        modelSource: SmeltCAMIR.Source
    ) throws {
        let output = try exactlyOne(export.outputs, "text projection output")
        let expectedNodeIDs: Set<String>
        var expectedEdges: [SmeltCAMIR.GraphEdge] = []
        let text = SmeltCAMIR.TypeRef("text", attributes: ["encoding": "utf8"])
        let tokens = SmeltCAMIR.TypeRef("tokens")
        let hidden = SmeltCAMIR.TypeRef("hidden")

        switch textPolicyBridge.setup {
        case .tokenizerOnly:
            expectedNodeIDs = ["tokenizer", block.id, "sampler", "detokenizer"]
            let input = try exactlyOne(export.inputs, "single text projection input")
            expectedEdges.append(
                .init(
                    from: .moduleInput(input.name),
                    to: .node("tokenizer", textPolicyBridge.tokenizerInputName),
                    type: text
                )
            )

        case .promptBuilder(let promptBuilder):
            expectedNodeIDs = [promptBuilder.id, "tokenizer", block.id, "sampler", "detokenizer"]
            let inputNames = Set(export.inputs.map(\.name))
            try require(
                inputNames == Set(promptBuilder.inputNames),
                "prompt adapter export inputs drifted"
            )
            for inputName in promptBuilder.inputNames {
                expectedEdges.append(
                    .init(
                        from: .moduleInput(inputName),
                        to: .node(promptBuilder.id, inputName),
                        type: text
                    )
                )
            }
            expectedEdges += [
                .init(
                    from: .node(promptBuilder.id, promptBuilder.outputName),
                    to: .graphValue(promptBuilder.outputName),
                    type: .init(promptBuilder.outputName)
                ),
                .init(
                    from: .graphValue(promptBuilder.outputName),
                    to: .node("tokenizer", textPolicyBridge.tokenizerInputName),
                    type: .init(promptBuilder.outputName)
                ),
            ]
        }

        try require(
            Set(cam.graphNodes.map(\.id)) == expectedNodeIDs,
            "text graph node set drifted"
        )

        expectedEdges += [
            .init(from: .node("tokenizer", "tokens"), to: .graphValue("tokens"), type: tokens),
            .init(from: .graphValue("tokens"), to: .node(block.id, "tokens"), type: tokens),
            .init(from: .node(block.id, "hidden"), to: .graphValue("hidden"), type: hidden),
            .init(from: .graphValue("hidden"), to: .node("sampler", "hidden"), type: hidden),
            .init(from: .node("sampler", "tokens"), to: .graphValue("tokens_2"), type: tokens),
            .init(from: .graphValue("tokens_2"), to: .node("detokenizer", "tokens"), type: tokens),
            .init(from: .node("detokenizer", "text"), to: .moduleOutput(output.name), type: text),
        ]
        try require(
            graphEdgeSignatures(cam.graphEdges) == graphEdgeSignatures(expectedEdges),
            "text graph edges drifted"
        )
        try require(
            cam.feedbackEdges == [.init(from: .node("sampler", "tokens"), to: .node(block.id, "tokens"))],
            "text graph feedback drifted"
        )
        try require(
            cam.tensors == [
                .init(
                    source: modelSource.id,
                    selector: .init("*", source: modelSource.id),
                    target: .init(block: block.id, selector: "*"),
                    owner: block.id
                ),
            ],
            "text tensor bindings drifted"
        )
        try require(cam.sourceQuantization.isEmpty, "text source quantization drifted")
    }

    private static func derivedTextPolicyBridge(
        from cam: SmeltCAMIR,
        export: SmeltCAMIR.Export,
        flow: SmeltCAMIR.Flow
    ) throws -> CheckedProjectionTextPolicyBridge {
        let setup = try exactlyOne(
            flow.phases.filter { $0.role == .setup },
            "text projection setup phase"
        )
        let setupCallIDs = try nodeCallIDs(setup)
        let tokenizer = try graphNode(cam, "tokenizer")
        let tokenizerInput = try exactlyOne(tokenizer.inputs, "tokenizer input")
        let requiredInputNames = export.inputs.filter { !$0.optional }.map(\.name).sorted()

        if setupCallIDs == ["tokenizer"] {
            try require(requiredInputNames.count == 1, "single text route input count drifted")
            return checkedTextPolicyBridge(
                tokenizerInputName: tokenizerInput.name,
                tokenizerInputType: tokenizerInput.type
            )
        }

        guard setupCallIDs.count == 2,
              setupCallIDs.last == "tokenizer",
              let promptBuilderID = setupCallIDs.first
        else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "text projection setup must call tokenizer or prompt adapter then tokenizer"
            )
        }
        try require(requiredInputNames.count == 2, "two text route input count drifted")
        let promptBuilder = try graphNode(cam, promptBuilderID)
        try require(promptBuilder.implementation == .adapter, "prompt adapter implementation drifted")
        let output = try exactlyOne(promptBuilder.outputs, "prompt adapter output")
        return checkedTextPolicyBridge(
            tokenizerInputName: tokenizerInput.name,
            tokenizerInputType: tokenizerInput.type,
            capabilityRequests: .exactTwoText(inputNames: requiredInputNames),
            setup: .promptBuilder(
                .init(
                    id: promptBuilderID,
                    template: try requireValue(
                        annotation("template", in: promptBuilder),
                        "prompt adapter template missing"
                    ),
                    inputNames: promptBuilder.inputs.map(\.name).sorted(),
                    outputName: output.name
                )
            )
        )
    }

    private static func derivedTextAssembly(
        from block: SmeltCAMIR.Block,
        prefill: CheckedProjectionPrefillPolicy,
        flow: SmeltCAMIR.Flow
    ) throws -> CheckedProjectionAssembly {
        let shape = try requireValue(block.shape.transformer, "text transformer shape missing")
        if shape.attentionByRole?.isEmpty == false
            || shape.perLayerInput != nil
            || shape.sharedKVLayers != nil
            || shape.logitCap != nil {
            return .roleAttentionCAMShape
        }
        let attention = try requireValue(shape.attention, "text attention shape missing")
        let norm = try requireValue(shape.norm, "text norm shape missing")
        let ffn = try requireValue(shape.ffn, "text ffn shape missing")
        let requirements = try blockRequirementMap(block.shape.requirements)
        let modelNormMode = try agentNormMode(
            from: try requireValue(norm.mode, "text norm mode missing")
        )
        let staticSeqCapacity = try textStaticSeqCapacity(
            requirements: requirements,
            prefill: prefill,
            flow: flow
        )
        return .explicitCAMShape(
            .init(
                blockID: block.id,
                operatorName: block.operatorName,
                staticSeqCapacity: staticSeqCapacity,
                ropeDim: try textRopeDim(
                    attention: attention,
                    shape: shape,
                    requirements: requirements
                ),
                ropeKind: try requireValue(attention.rope, "text attention rope missing").kind,
                gatedQ: try boolRequirement(
                    "gated-q",
                    in: requirements,
                    default: shape.delta != nil
                ),
                qkvBias: try boolRequirement("qkv-bias", in: requirements, default: false),
                attentionQKNorm: attention.qkNorm,
                attentionQKNormMode: try attention.qkNormMode
                    ?? requireValue(norm.mode, "text norm mode missing"),
                normKind: norm.kind,
                normMode: try requireValue(norm.mode, "text norm mode missing"),
                normEpsilon: try requireValue(norm.eps, "text norm epsilon missing"),
                rmsEpsilon: try floatValue(
                    try requireValue(norm.eps, "text norm epsilon missing"),
                    label: "text norm epsilon"
                ),
                modelNormMode: modelNormMode,
                ffnActivation: ffn.activation,
                requirements: blockRequirementList(block.shape.requirements)
            )
        )
    }

    private static func textStaticSeqCapacity(
        requirements: [String: String],
        prefill: CheckedProjectionPrefillPolicy,
        flow: SmeltCAMIR.Flow
    ) throws -> Int {
        if let value = requirements["static-seq-capacity"] {
            return try intValue(value, label: "static sequence capacity")
        }
        if case .required(let prefill) = prefill {
            return prefill.maxBatchSize
        }
        return min(try requireValue(
            flow.stop.first { $0.kind == .maxSteps }?.value,
            "flow max-steps missing"
        ), 256)
    }

    private static func textRopeDim(
        attention: SmeltCAMIR.AttentionShape,
        shape: SmeltCAMIR.TransformerShape,
        requirements: [String: String]
    ) throws -> Int {
        if let value = requirements["rope-dim"] {
            return try intValue(value, label: "rope dim")
        }
        if shape.delta != nil {
            return min(attention.headDim, 64)
        }
        return attention.headDim
    }

    private static func boolRequirement(
        _ key: String,
        in requirements: [String: String],
        default defaultValue: Bool
    ) throws -> Bool {
        guard let value = requirements[key] else { return defaultValue }
        switch value {
        case "true": return true
        case "false": return false
        default:
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(key) requirement is not a boolean"
            )
        }
    }

    /// Lower the module-authored, model-agnostic fused-feature brick into the
    /// compiler IR. A package either declares the brick completely or omits it;
    /// checkpoint families do not participate in this decision.
    private static func inputFusionConfig(
        from requirements: [String: String]
    ) throws -> SmeltInputFusionConfig? {
        let prefix = "input-fusion-"
        guard requirements.keys.contains(where: { $0.hasPrefix(prefix) }) else {
            return nil
        }

        let sourceWidth = try intRequirement(
            "input-fusion-source-width",
            in: requirements,
            label: "input fusion source width"
        )
        let sourceCount = try intRequirement(
            "input-fusion-source-count",
            in: requirements,
            label: "input fusion source count"
        )
        let normalizeSources = try boolRequirement(
            "input-fusion-normalize-sources",
            in: requirements,
            default: false
        )
        let postProjectionWidth = try requirements["input-fusion-post-projection-width"]
            .map { try intValue($0, label: "input fusion post projection width") }

        let supported = Set([
            "input-fusion-source-width",
            "input-fusion-source-count",
            "input-fusion-normalize-sources",
            "input-fusion-post-projection-width",
        ])
        let unknown = requirements.keys
            .filter { $0.hasPrefix(prefix) && !supported.contains($0) }
            .sorted()
        try require(
            unknown.isEmpty,
            "unsupported input fusion requirements: \(unknown.joined(separator: ", "))"
        )

        return SmeltInputFusionConfig(
            sourceWidth: sourceWidth,
            sourceCount: sourceCount,
            normalizeSources: normalizeSources,
            postProjectionWidth: postProjectionWidth
        )
    }

    private static func derivedTextQuantization(
        from cam: SmeltCAMIR
    ) throws -> CheckedProjectionQuantization {
        let defaultRule = try exactlyOne(
            cam.quantization.filter { $0.action == .default },
            "default quant rule"
        )
        let storage = try requireValue(defaultRule.storage, "default quant storage missing")
        let groupSize = try requireValue(storage.groupSize, "default quant group missing")
        let strategy: SmeltQuantStrategy
        switch storage.format {
        case .affineU4:
            strategy = .affineU4
        case .lutU4:
            strategy = .lutU4
        case .binary1:
            strategy = .binary1
        case .ternary2:
            strategy = .ternary2
        case .fp16, .bf16, .gptq, .turboQuantH:
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "default text quant storage '\(storage.format.rawValue)' does not lower to text package projection"
            )
        }
        let quantizedPatterns = cam.quantization
            .filter { $0.action == .quantize }
            .map(\.selector.pattern)
        if !quantizedPatterns.isEmpty && quantizedPatterns != ["embeddings"] {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "text projection only supports CAM quantize embeddings"
            )
        }
        let storedPatterns = try cam.quantization
            .filter { $0.action == .store }
            .map { rule in
                let storage = try requireValue(rule.storage, "stored quant storage missing")
                return CheckedProjectionStoredPattern(
                    pattern: rule.selector.pattern,
                    storage: .init(format: storage.format, groupSize: storage.groupSize)
                )
            }
        return CheckedProjectionQuantization(
            strategy: strategy,
            defaultStorage: .init(format: storage.format, groupSize: groupSize),
            quantizedPatterns: quantizedPatterns,
            preservedPatterns: cam.quantization
                .filter { $0.action == .preserve }
                .map(\.selector.pattern),
            storedPatterns: storedPatterns,
            quantizesEmbedding: quantizedPatterns == ["embeddings"]
        )
    }

    private static func derivedTextPrefillPolicy(
        from cam: SmeltCAMIR,
        block: SmeltCAMIR.Block,
        flow: SmeltCAMIR.Flow
    ) throws -> CheckedProjectionPrefillPolicy {
        let prefillConstraints = cam.compile.filter { $0.key == "prefill" }
        guard !prefillConstraints.isEmpty else {
            return .none(
                rejectedCompileKeys: ["prefill"],
                rejectedSourceIDs: ["prefill", "prefill_cache"],
                rejectedGraphNodeIDs: ["prefill"]
            )
        }
        let value = try exactlyOne(prefillConstraints.map(\.value), "prefill constraint")
        let parsed = try parsePrefillCompileConstraint(value, engine: "metal")
        let trunk = try graphNode(cam, block.id)
        return .required(
            .init(
                compileKey: "prefill",
                engine: "metal",
                modelPath: "",
                cachePath: "",
                maxBatchSize: parsed.batch,
                handoffNames: textPrefillHandoffNames(from: trunk),
                rejectedSourceIDs: ["prefill", "prefill_cache"],
                rejectedGraphNodeIDs: ["prefill"]
            )
        )
    }

    private static func textPrefillHandoffNames(
        from trunk: SmeltCAMIR.GraphNode
    ) -> [String] {
        let state = Set(
            (annotation("state", in: trunk) ?? "")
                .split(separator: ",")
                .map(String.init)
        )
        var names: [String] = []
        if state.contains("conv-state") {
            names.append("conv_state")
        }
        if state.contains("rec-state") {
            names.append("rec_state")
        }
        names += ["key_cache", "value_cache", "rope"]
        return names
    }

    private static func derivedTextProjectionID(
        export: SmeltCAMIR.Export,
        block: SmeltCAMIR.Block,
        quantization: CheckedProjectionQuantization,
        prefill: CheckedProjectionPrefillPolicy,
        textPolicyBridge: CheckedProjectionTextPolicyBridge
    ) throws -> String {
        let inputCount = export.inputs.filter { !$0.optional }.count
        let prefix = inputCount == 2
            ? "two-text-transformer"
            : "text-to-text-transformer"
        let prefillPart: String
        switch prefill {
        case .required:
            prefillPart = "prefill-decode"
        case .none:
            prefillPart = "decode-only"
        }
        var parts = [
            prefix,
            prefillPart,
            try quantizationProjectionID(quantization),
        ]
        if quantization.storedPatterns.contains(where: { $0.storage.format == .turboQuantH }) {
            parts.append("tqh-embeddings")
        }
        let shape = try requireValue(block.shape.transformer, "text transformer shape missing")
        if shape.attentionByRole?.isEmpty == false
            || shape.perLayerInput != nil
            || shape.sharedKVLayers != nil
            || shape.logitCap != nil {
            parts.append("altup")
        }
        let requirements = try blockRequirementMap(block.shape.requirements)
        if try boolRequirement("qkv-bias", in: requirements, default: false) {
            parts.append("projection-bias")
        }
        if case .promptBuilder = textPolicyBridge.setup {
            parts.append("prompt-adapter")
        }
        return parts.joined(separator: "-")
    }

    private static func quantizationProjectionID(
        _ quantization: CheckedProjectionQuantization
    ) throws -> String {
        let group = try requireValue(
            quantization.defaultStorage.groupSize,
            "default quant group missing"
        )
        return "\(quantization.defaultStorage.format.rawValue)-g\(group)"
    }

    private static func checkedPackageProjectionProfile(
        cam: SmeltCAMIR,
        descriptor: SmeltCAMPackageDescriptor,
        descriptorVersion: Int,
        semanticSHA256: String,
        exportABISHA256: String,
        descriptorGraphSignatureSHA256: String
    ) throws -> CheckedResolvedPackageProjectionProfile? {
        let admission = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: checkedProjectionConsumedFeatureSet
        )
        if !admission.hasUnsupportedFeatures,
           let textProfile = try derivedTextProjectionProfile(
                from: cam,
                descriptorVersion: descriptorVersion,
                semanticSHA256: semanticSHA256,
                exportABISHA256: exportABISHA256,
                descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256
           ) {
            return resolvedTransformerProjectionProfile(textProfile)
        }
        if !admission.hasUnsupportedFeatures,
           let audioProfile = try derivedAudioProjectionProfile(
                from: cam,
                descriptorVersion: descriptorVersion,
                semanticSHA256: semanticSHA256,
                exportABISHA256: exportABISHA256,
                descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256
           ) {
            return resolvedSidecarProjectionProfile(audioProfile)
        }

        let matches = checkedPackageProjectionRegistry.filter {
            $0.descriptorVersion == descriptorVersion
                && $0.semanticSHA256 == semanticSHA256
                && $0.exportABISHA256 == exportABISHA256
                && $0.descriptorGraphSignatureSHA256 == descriptorGraphSignatureSHA256
        }
        guard matches.count <= 1 else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "ambiguous checked package projection profiles for descriptor/hash tuple"
            )
        }
        return matches.first
    }

    private static var checkedPackageProjectionRegistry: [CheckedResolvedPackageProjectionProfile] {
        checkedPackageProjectionProfiles.map(resolvedTransformerProjectionProfile)
            + checkedDerivedAudioPackageProjectionProfiles.map(resolvedSidecarProjectionProfile)
    }

    private static func resolvedTransformerProjectionProfile(
        _ profile: CheckedPackageProjectionProfile
    ) -> CheckedResolvedPackageProjectionProfile {
        CheckedResolvedPackageProjectionProfile(
            id: profile.id,
            version: profile.version,
            descriptorVersion: profile.descriptorVersion,
            semanticSHA256: profile.semanticSHA256,
            exportABISHA256: profile.exportABISHA256,
            descriptorGraphSignatureSHA256: profile.descriptorGraphSignatureSHA256,
            buildCommandCovered: profile.buildCommandCovered,
            packageFiles: profile.packageFiles,
            materializer: .loweredTransformer(profile)
        )
    }

    private static func resolvedSidecarProjectionProfile(
        _ profile: CheckedDerivedAudioPackageProjectionProfile
    ) -> CheckedResolvedPackageProjectionProfile {
        CheckedResolvedPackageProjectionProfile(
            id: profile.id,
            version: profile.version,
            descriptorVersion: profile.descriptorVersion,
            semanticSHA256: profile.semanticSHA256,
            exportABISHA256: profile.exportABISHA256,
            descriptorGraphSignatureSHA256: profile.descriptorGraphSignatureSHA256,
            buildCommandCovered: profile.buildCommandCovered,
            packageFiles: profile.packageFiles,
            materializer: .sidecarManifest(profile)
        )
    }

    private static func projectSpec(
        from cam: SmeltCAMIR,
        artifactRoot: String,
        profile: CheckedResolvedPackageProjectionProfile
    ) throws -> SmeltPackageSpec {
        switch profile.materializer {
        case .loweredTransformer(let materializerProfile):
            let model = try modelIR(from: cam, profile: materializerProfile)
            let lowered = try SmeltPackageSpecLowering.textGeneration(
                from: model,
                source: .init(id: "package", kind: .localDirectory, path: artifactRoot)
            )
            return try compactAssemblySpec(
                from: lowered,
                model: model,
                cam: cam,
                profile: materializerProfile
            )
        case .sidecarManifest(let materializerProfile):
            return try derivedAudioPackageSpec(
                from: cam,
                artifactRoot: artifactRoot,
                profile: materializerProfile
            )
        }
    }

    private static func derivedAudioProjectionProfile(
        from cam: SmeltCAMIR,
        descriptorVersion: Int,
        semanticSHA256: String,
        exportABISHA256: String,
        descriptorGraphSignatureSHA256: String
    ) throws -> CheckedDerivedAudioPackageProjectionProfile? {
        guard let export = try audioProjectionExport(from: cam) else { return nil }
        guard cam.blocks.count == 5 else { return nil }

        let flow = try exactlyOne(cam.flows, "audio projection flow")
        let binding = try exactlyOne(cam.exportBindings, "audio projection export binding")
        try require(binding.export == export.id, "audio projection export binding drifted")
        try require(binding.flow == flow.id, "audio projection flow binding drifted")

        let modelSource = try primaryModelSource(from: cam)
        let graph = try derivedAudioGraph(
            from: cam,
            export: export,
            flow: flow,
            modelSourceID: modelSource.id
        )
        let tensorMaps = try derivedAudioTensorMaps(
            from: cam,
            graph: graph,
            modelSourceID: modelSource.id
        )
        let quantization = try derivedAudioQuantizationProfile(
            from: cam,
            graph: graph,
            modelSourceID: modelSource.id
        )
        let maxFrames = try derivedAudioMaxFrames(from: graph.stops)
        let eosTokens = try derivedAudioCodecEOSTokens(from: cam, graph: graph)
        let packageProfile = SmeltQwen3TTSPackageProfiles.runnable

        return CheckedDerivedAudioPackageProjectionProfile(
            id: try derivedAudioProjectionID(export: export, quantization: quantization),
            version: 1,
            descriptorVersion: descriptorVersion,
            semanticSHA256: semanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256,
            buildCommandCovered: true,
            flowID: flow.id,
            packageFiles: try packageInventoryFiles(from: cam),
            sources: cam.sources.map {
                CheckedProjectionSource(
                    id: $0.id,
                    kind: $0.kind,
                    locator: $0.locator,
                    revision: $0.revision,
                    checkpointMap: $0.checkpointMap
                )
            },
            modelSourceID: modelSource.id,
            manifest: .init(
                modelName: packageProfile.modelName,
                tokenizerFiles: packageProfile.tokenizerFiles,
                sidecarPaths: packageProfile.sidecarPaths,
                pageSize: packageProfile.pageSize,
                maxTokens: maxFrames,
                eosTokens: eosTokens,
                performanceGate: packageProfile.performanceGate,
                structureProfileID: packageProfile.structureProfileID
            ),
            graph: graph,
            tensorMaps: tensorMaps,
            quantization: quantization
        )
    }

    private static func audioProjectionExport(from cam: SmeltCAMIR) throws -> SmeltCAMIR.Export? {
        guard cam.exports.count == 1 else { return nil }
        let export = try exactlyOne(cam.exports, "audio projection export")
        guard export.outputs.count == 1,
              let output = export.outputs.first,
              isPCM(output, rate: "24khz")
        else {
            return nil
        }
        guard !output.optional else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "audio projection exports do not support optional outputs"
            )
        }

        let requiredTextInputs = export.inputs.filter { !$0.optional && isTextUTF8($0) }
        let optionalVoiceInputs = export.inputs.filter {
            $0.optional && $0.type == .init("voice-id")
        }
        guard requiredTextInputs.count == 1,
              optionalVoiceInputs.count == 1,
              export.inputs.count == 2
        else {
            return nil
        }
        return export
    }

    private static func isPCM(_ port: SmeltCAMIR.Port, rate: String) -> Bool {
        port.type == .init("pcm", attributes: ["dtype": "f32", "rate": rate])
    }

    private static func derivedAudioGraph(
        from cam: SmeltCAMIR,
        export: SmeltCAMIR.Export,
        flow: SmeltCAMIR.Flow,
        modelSourceID: String
    ) throws -> CheckedProjectionAudioGraph {
        let textInput = try exactlyOne(
            export.inputs.filter { !$0.optional && isTextUTF8($0) },
            "audio text input"
        )
        let audioOutput = try exactlyOne(export.outputs, "audio output")
        let frontendBlock = try exactlyOne(
            cam.blocks.filter { $0.operatorName == .ttsFrontend },
            "audio frontend block"
        )
        let codecBlock = try exactlyOne(
            cam.blocks.filter { $0.operatorName == .codecDecoder },
            "audio codec block"
        )
        let codecHeadBlock = try exactlyOne(
            cam.blocks.filter { $0.operatorName == .codecHead },
            "audio codec-head block"
        )
        let transformerBlocks = cam.blocks.filter { $0.operatorName == .transformer }
        try require(transformerBlocks.count == 2, "audio transformer block count drifted")

        let feedbackEdge = try exactlyOne(cam.feedbackEdges, "audio feedback edge")
        let feedbackFrom = try nodePort(feedbackEdge.from, label: "audio feedback source")
        let feedbackTo = try nodePort(feedbackEdge.to, label: "audio feedback target")
        let mtpNode = try graphNode(cam, feedbackFrom.node)
        let talkerNode = try graphNode(cam, feedbackTo.node)
        let codecNode = try exactlyOne(
            cam.graphNodes.filter {
                $0.implementation == .compiled && $0.block == codecBlock.id
            },
            "audio codec graph node"
        )
        let frontendNode = try exactlyOne(
            cam.graphNodes.filter {
                $0.implementation == .native
                    && $0.block == nil
                    && $0.source == nil
                    && $0.imported == nil
                    && $0.inputs.contains(where: isTextUTF8)
            },
            "audio frontend graph node"
        )
        try require(frontendNode.id == frontendBlock.id, "audio frontend node id drifted")
        let codecHeadNode = try exactlyOne(
            cam.graphNodes.filter {
                $0.implementation == .native
                    && $0.id == codecHeadBlock.id
                    && $0.block == nil
                    && $0.source == nil
                    && $0.imported == nil
            },
            "audio codec-head graph node"
        )

        let talkerBlockID = try requireValue(talkerNode.block, "audio talker block missing")
        let mtpBlockID = try requireValue(mtpNode.block, "audio mtp block missing")
        try require(
            Set(transformerBlocks.map(\.id)) == Set([talkerBlockID, mtpBlockID]),
            "audio transformer node bindings drifted"
        )
        try validateDerivedAudioBlockShapes(
            frontend: frontendBlock,
            talker: try block(cam, talkerBlockID),
            codecHead: codecHeadBlock,
            mtp: try block(cam, mtpBlockID),
            codec: codecBlock,
            modelSourceID: modelSourceID
        )
        try validateDerivedAudioNodeShapes(
            frontend: frontendNode,
            talker: talkerNode,
            codecHead: codecHeadNode,
            mtp: mtpNode,
            codec: codecNode,
            feedbackPort: feedbackTo.port
        )

        let frontendInput = try exactlyOne(frontendNode.inputs.filter(isTextUTF8), "audio frontend input")
        let frontendOutput = try exactlyOne(frontendNode.outputs, "audio frontend output")
        let talkerPromptInput = try exactlyOne(
            talkerNode.inputs.filter { $0.type == frontendOutput.type },
            "audio talker prompt input"
        )
        let talkerFeedbackInput = try exactlyOne(
            talkerNode.inputs.filter { $0.name == feedbackTo.port },
            "audio talker feedback input"
        )
        let talkerOutput = try exactlyOne(talkerNode.outputs, "audio talker output")
        let codecHeadInput = try exactlyOne(
            codecHeadNode.inputs.filter { $0.type == talkerOutput.type },
            "audio codec-head input"
        )
        let codecHeadOutput = try exactlyOne(codecHeadNode.outputs, "audio codec-head output")
        let mtpHiddenInput = try exactlyOne(
            mtpNode.inputs.filter { $0.type == talkerOutput.type },
            "audio mtp hidden input"
        )
        let mtpCodecInput = try exactlyOne(
            mtpNode.inputs.filter { $0.type == codecHeadOutput.type },
            "audio mtp codec-head input"
        )
        let mtpOutput = try exactlyOne(
            mtpNode.outputs.filter { $0.name == feedbackFrom.port },
            "audio mtp output"
        )
        let codecInput = try exactlyOne(
            codecNode.inputs.filter { $0.type == mtpOutput.type },
            "audio codec input"
        )
        let codecOutput = try exactlyOne(
            codecNode.outputs.filter { $0.type == audioOutput.type },
            "audio codec output"
        )
        try require(
            talkerFeedbackInput.type == mtpOutput.type,
            "audio feedback token type drifted"
        )

        let expectedEdges: [SmeltCAMIR.GraphEdge] = [
            .init(from: .moduleInput(textInput.name), to: .node(frontendNode.id, frontendInput.name), type: textInput.type),
            .init(from: .node(frontendNode.id, frontendOutput.name), to: .graphValue(frontendOutput.name), type: frontendOutput.type),
            .init(from: .graphValue(frontendOutput.name), to: .node(talkerNode.id, talkerPromptInput.name), type: frontendOutput.type),
            .init(from: .node(talkerNode.id, talkerOutput.name), to: .graphValue(talkerOutput.name), type: talkerOutput.type),
            .init(from: .graphValue(talkerOutput.name), to: .node(codecHeadNode.id, codecHeadInput.name), type: talkerOutput.type),
            .init(from: .node(codecHeadNode.id, codecHeadOutput.name), to: .graphValue(codecHeadOutput.name), type: codecHeadOutput.type),
            .init(from: .graphValue(codecHeadOutput.name), to: .node(mtpNode.id, mtpCodecInput.name), type: codecHeadOutput.type),
            .init(from: .graphValue(talkerOutput.name), to: .node(mtpNode.id, mtpHiddenInput.name), type: talkerOutput.type),
            .init(from: .node(mtpNode.id, mtpOutput.name), to: .graphValue(mtpOutput.name), type: mtpOutput.type),
            .init(from: .graphValue(mtpOutput.name), to: .node(codecNode.id, codecInput.name), type: mtpOutput.type),
            .init(from: .node(codecNode.id, codecOutput.name), to: .moduleOutput(audioOutput.name), type: audioOutput.type),
        ]
        try require(
            graphEdgeSignatures(cam.graphEdges) == graphEdgeSignatures(expectedEdges),
            "audio graph edges drifted"
        )
        try require(
            feedbackEdge == .init(
                from: .node(mtpNode.id, mtpOutput.name),
                to: .node(talkerNode.id, talkerFeedbackInput.name)
            ),
            "audio feedback edge drifted"
        )

        let setup = try exactlyOne(flow.phases.filter { $0.role == .setup }, "audio setup phase")
        try require(
            try nodeCallIDs(setup) == [frontendNode.id, talkerNode.id, codecHeadNode.id],
            "audio setup calls drifted"
        )
        let stepPhases = flow.phases.filter { $0.role == .step }
        try require(stepPhases.count == 2, "audio step phase count drifted")
        try require(
            try nodeCallIDs(stepPhases[0]) == [mtpNode.id, talkerNode.id, codecHeadNode.id],
            "audio generate calls drifted"
        )
        try require(
            try nodeCallIDs(stepPhases[1]) == [codecNode.id],
            "audio render calls drifted"
        )
        try require(
            flow.emit == [.node(codecNode.id, codecOutput.name)],
            "audio emit endpoint drifted"
        )
        let stops = try checkedAudioStops(from: flow.stop)
        let maxFrames = try derivedAudioMaxFrames(from: stops)
        let talkerMaxFrames = try intBlockRequirement(
            "max-frames",
            in: try block(cam, talkerBlockID).shape.requirements,
            label: "audio talker max frames"
        )
        let mtpMaxFrames = try intBlockRequirement(
            "max-frames",
            in: try block(cam, mtpBlockID).shape.requirements,
            label: "audio mtp max frames"
        )
        try require(
            talkerMaxFrames == maxFrames && mtpMaxFrames == maxFrames,
            "audio max frames drifted"
        )
        let mtpCodebooks = try intBlockRequirement(
            "codebooks",
            in: try block(cam, mtpBlockID).shape.requirements,
            label: "audio codebooks"
        )
        let mtpResidualCodebooks = try intBlockRequirement(
            "residual-codebooks",
            in: try block(cam, mtpBlockID).shape.requirements,
            label: "audio residual codebooks"
        )
        try require(mtpCodebooks == mtpResidualCodebooks, "audio residual codebooks drifted")
        try require(
            mtpCodebooks == CheckedDerivedAudioRuntimeContract.codebooks,
            "audio codebooks \(mtpCodebooks) drifted from runtime "
                + "\(CheckedDerivedAudioRuntimeContract.codebooks)"
        )
        try require(
            mtpResidualCodebooks == CheckedDerivedAudioRuntimeContract.codebooks,
            "audio residual codebooks \(mtpResidualCodebooks) drifted from runtime "
                + "\(CheckedDerivedAudioRuntimeContract.codebooks)"
        )
        try require(
            annotation("codebooks", in: mtpNode) == String(mtpCodebooks),
            "audio mtp codebooks annotation drifted"
        )

        let expectedNodeIDs = Set([frontendNode.id, talkerNode.id, codecHeadNode.id, mtpNode.id, codecNode.id])
        try require(Set(cam.graphNodes.map(\.id)) == expectedNodeIDs, "audio graph node set drifted")

        return CheckedProjectionAudioGraph(
            blocks: try cam.blocks.map {
                CheckedProjectionAudioBlock(
                    id: $0.id,
                    operatorName: $0.operatorName,
                    derivation: try requireValue($0.shape.derivation, "\($0.id) derivation missing"),
                    requirements: blockRequirementList($0.shape.requirements)
                )
            }.sorted { $0.id < $1.id },
            graphNodes: [frontendNode, talkerNode, codecHeadNode, mtpNode, codecNode].map {
                CheckedProjectionAudioNode(
                    id: $0.id,
                    implementation: $0.implementation,
                    block: $0.block,
                    inputs: $0.inputs,
                    outputs: $0.outputs,
                    annotations: $0.annotations
                )
            }.sorted { $0.id < $1.id },
            graphEdges: try expectedEdges.map {
                CheckedProjectionAudioEdge(
                    from: $0.from,
                    to: $0.to,
                    type: try requireValue($0.type, "audio graph edge type missing")
                )
            },
            feedbackEdge: .init(
                from: .node(mtpNode.id, mtpOutput.name),
                to: .node(talkerNode.id, talkerFeedbackInput.name)
            ),
            setupCalls: [frontendNode.id, talkerNode.id, codecHeadNode.id],
            stepCallsByLabel: [
                try requireValue(stepPhases[0].label, "audio generate label missing"):
                    [mtpNode.id, talkerNode.id, codecHeadNode.id],
                try requireValue(stepPhases[1].label, "audio render label missing"):
                    [codecNode.id],
            ],
            emitNode: codecNode.id,
            emitPort: codecOutput.name,
            stops: stops
        )
    }

    private static func validateDerivedAudioBlockShapes(
        frontend: SmeltCAMIR.Block,
        talker: SmeltCAMIR.Block,
        codecHead: SmeltCAMIR.Block,
        mtp: SmeltCAMIR.Block,
        codec: SmeltCAMIR.Block,
        modelSourceID: String
    ) throws {
        for block in [frontend, talker, codecHead, mtp, codec] {
            try require(
                block.shape.derivation == .init(source: modelSourceID),
                "\(block.id) derivation drifted"
            )
            try require(block.annotations.isEmpty, "\(block.id) annotations drifted")
        }
        try require(frontend.shape.frontend?.speakerConditioning == true, "audio frontend shape drifted")
        try require(frontend.shape.transformer == nil, "audio frontend transformer drifted")
        try require(frontend.shape.codecDecoder == nil, "audio frontend codec drifted")
        try require(talker.shape.transformer != nil, "audio talker transformer missing")
        try require(talker.shape.frontend == nil, "audio talker frontend drifted")
        try require(talker.shape.codecDecoder == nil, "audio talker codec drifted")
        try require(codecHead.shape.transformer == nil, "audio codec-head transformer drifted")
        try require(codecHead.shape.frontend == nil, "audio codec-head frontend drifted")
        try require(codecHead.shape.codecDecoder == nil, "audio codec-head codec drifted")
        try require(mtp.shape.transformer != nil, "audio mtp transformer missing")
        try require(mtp.shape.frontend == nil, "audio mtp frontend drifted")
        try require(mtp.shape.codecDecoder == nil, "audio mtp codec drifted")
        try require(codec.shape.codecDecoder?.streaming == true, "audio codec streaming drifted")
        try require(codec.shape.frontend == nil, "audio codec frontend drifted")
        try require(codec.shape.transformer == nil, "audio codec transformer drifted")

        let frontendRequirements = blockRequirementList(frontend.shape.requirements)
        try require(
            frontendRequirements.contains(.init(key: "speaker-conditioning", optional: true)),
            "audio frontend speaker requirement drifted"
        )
        try require(
            try blockRequirementValue(
                "source-dtype",
                in: frontend.shape.requirements,
                label: "audio frontend source dtype"
            ) == "bf16",
            "audio frontend source dtype drifted"
        )
        try require(
            talker.shape.requirements.contains {
                $0.key == "codec-feedback" && $0.value == nil && !$0.optional
            },
            "audio talker feedback missing"
        )
        try require(
            try blockRequirementValue(
                "source-dtype",
                in: talker.shape.requirements,
                label: "audio talker source dtype"
            ) == "bf16",
            "audio talker source dtype drifted"
        )
        _ = try intBlockRequirement("codec-eos", in: talker.shape.requirements, label: "audio codec eos")
        _ = try intBlockRequirement("max-frames", in: talker.shape.requirements, label: "audio talker max frames")
        try require(
            codecHead.shape.requirements.contains {
                $0.key == "sampler" && $0.value == nil && !$0.optional
            },
            "audio codec-head sampler missing"
        )
        try require(
            try blockRequirementValue(
                "source-dtype",
                in: codecHead.shape.requirements,
                label: "audio codec-head source dtype"
            ) == "bf16",
            "audio codec-head source dtype drifted"
        )
        _ = try intBlockRequirement("codebooks", in: mtp.shape.requirements, label: "audio codebooks")
        _ = try intBlockRequirement(
            "residual-codebooks",
            in: mtp.shape.requirements,
            label: "audio residual codebooks"
        )
        _ = try intBlockRequirement("max-frames", in: mtp.shape.requirements, label: "audio mtp max frames")
        try require(
            try blockRequirementValue(
                "source-dtype",
                in: mtp.shape.requirements,
                label: "audio mtp source dtype"
            ) == "bf16",
            "audio mtp source dtype drifted"
        )
        try require(
            codec.shape.requirements.contains {
                $0.key == "streaming" && $0.value == nil && !$0.optional
            },
            "audio codec streaming requirement missing"
        )
        try require(
            try blockRequirementValue("audio-format", in: codec.shape.requirements, label: "audio codec format")
                == "pcm-f32",
            "audio codec format drifted"
        )
        try require(
            try blockRequirementValue("audio-rate", in: codec.shape.requirements, label: "audio codec rate")
                == "24khz",
            "audio codec rate drifted"
        )
        try require(
            try blockRequirementValue(
                "source-dtype",
                in: codec.shape.requirements,
                label: "audio codec source dtype"
            ) == "f32",
            "audio codec source dtype drifted"
        )
    }

    private static func validateDerivedAudioNodeShapes(
        frontend: SmeltCAMIR.GraphNode,
        talker: SmeltCAMIR.GraphNode,
        codecHead: SmeltCAMIR.GraphNode,
        mtp: SmeltCAMIR.GraphNode,
        codec: SmeltCAMIR.GraphNode,
        feedbackPort: String
    ) throws {
        try require(frontend.annotations == [.init("speaker", "optional")], "audio frontend annotations drifted")
        try require(codecHead.annotations == [.init("state", "sampler")], "audio codec-head annotations drifted")
        try require(codecHead.block == nil && codecHead.source == nil && codecHead.imported == nil, "audio codec-head source drifted")
        try require(talker.implementation == .compiled, "audio talker implementation drifted")
        try require(codecHead.implementation == .native, "audio codec-head implementation drifted")
        try require(mtp.implementation == .compiled, "audio mtp implementation drifted")
        try require(codec.implementation == .compiled, "audio codec implementation drifted")
        try require(talker.source == nil && talker.imported == nil, "audio talker source drifted")
        try require(mtp.source == nil && mtp.imported == nil, "audio mtp source drifted")
        try require(codec.source == nil && codec.imported == nil, "audio codec source drifted")
        try require(annotation("artifact", in: talker) == "sidecar", "audio talker artifact drifted")
        try require(annotation("feedback", in: talker) == feedbackPort, "audio talker feedback drifted")
        try require(annotation("state", in: talker) == "kv-cache", "audio talker state drifted")
        try require(talker.annotations.count == 3, "audio talker annotations drifted")
        try require(annotation("artifact", in: mtp) == "sidecar", "audio mtp artifact drifted")
        _ = try requireValue(annotation("codebooks", in: mtp), "audio mtp codebooks missing")
        try require(mtp.annotations.count == 2, "audio mtp annotations drifted")
        try require(annotation("artifact", in: codec) == "baked-inline", "audio codec artifact drifted")
        try require(annotation("streaming", in: codec) == "true", "audio codec streaming annotation drifted")
        try require(codec.annotations.count == 2, "audio codec annotations drifted")
    }

    private static func derivedAudioTensorMaps(
        from cam: SmeltCAMIR,
        graph: CheckedProjectionAudioGraph,
        modelSourceID: String
    ) throws -> [SmeltCAMIR.TensorMap] {
        let blockIDs = Set(graph.blocks.map(\.id))
        try require(!cam.tensors.isEmpty, "audio tensor partitions missing")
        var owners = Set<String>()
        var signatures = Set<String>()
        for tensor in cam.tensors {
            try require(tensor.source == modelSourceID, "audio tensor source drifted")
            try require(tensor.selector.source == modelSourceID, "audio tensor selector source drifted")
            try require(blockIDs.contains(tensor.owner), "audio tensor owner drifted")
            try require(tensor.target.block == tensor.owner, "audio tensor target owner drifted")
            try require(!tensor.selector.pattern.isEmpty, "audio tensor selector missing")
            try require(!tensor.target.selector.isEmpty, "audio tensor target selector missing")
            owners.insert(tensor.owner)
            let signature = tensorMapSignatures([tensor]).joined(separator: "")
            try require(signatures.insert(signature).inserted, "audio tensor partition duplicated")
        }
        try require(
            owners == blockIDs,
            "audio tensor partitions do not cover every block"
        )
        return cam.tensors.sorted { lhs, rhs in
            tensorMapSignatures([lhs]).lexicographicallyPrecedes(tensorMapSignatures([rhs]))
        }
    }

    private static func derivedAudioQuantizationProfile(
        from cam: SmeltCAMIR,
        graph: CheckedProjectionAudioGraph,
        modelSourceID: String
    ) throws -> CheckedProjectionAudioQuantization {
        let defaultRule = try exactlyOne(
            cam.quantization.filter { $0.action == .default },
            "audio default quant rule"
        )
        let storage = try requireValue(defaultRule.storage, "audio default quant storage missing")
        try require(defaultRule.selector == .init("*", source: modelSourceID), "audio default quant selector drifted")
        try require(defaultRule.resolution == .declaredTensor, "audio default quant resolution drifted")
        try require(defaultRule.source == nil, "audio default quant source drifted")
        try require(defaultRule.priority == nil, "audio default quant priority drifted")
        try require(defaultRule.calibration == nil, "audio default quant calibration drifted")
        try require(defaultRule.storage?.computeDType == nil, "audio default quant compute dtype drifted")
        try require(
            storage.format == .affineU4,
            "audio default quant storage '\(storage.format.rawValue)' does not lower to sidecar audio"
        )
        _ = try requireValue(storage.groupSize, "audio default quant group missing")

        let preserves = cam.quantization.filter { $0.action == .preserve }
        let preservedPatterns = preserves.map(\.selector.pattern).sorted()
        try require(!preservedPatterns.isEmpty, "audio preserve quant patterns missing")
        for rule in preserves {
            try require(rule.selector.source == modelSourceID, "audio preserve source drifted")
            try require(rule.source == modelSourceID, "audio preserve owner source drifted")
            try require(rule.storage == nil, "audio preserve storage drifted")
            try require(rule.calibration == nil, "audio preserve calibration drifted")
            try require(
                rule.priority == patternSpecificity(rule.selector.pattern),
                "audio preserve priority drifted"
            )
            try require(
                rule.resolution == .declaredTensor || rule.resolution == .sourceDeferred,
                "audio preserve resolution drifted"
            )
        }
        try require(
            cam.quantization.filter { $0.action == .store || $0.action == .quantize }.isEmpty,
            "audio quantization action drifted"
        )
        return CheckedProjectionAudioQuantization(
            defaultStorage: .init(format: storage.format, groupSize: storage.groupSize),
            preservedPatterns: preservedPatterns
        )
    }

    private static func derivedAudioProjectionID(
        export: SmeltCAMIR.Export,
        quantization: CheckedProjectionAudioQuantization
    ) throws -> String {
        let output = try exactlyOne(export.outputs, "audio projection output")
        let rate = try requireValue(output.type.attributes["rate"], "audio output rate missing")
        let group = try requireValue(
            quantization.defaultStorage.groupSize,
            "audio default quant group missing"
        )
        return "streaming-text-to-\(rate)-audio-derived-manifest-"
            + "\(quantization.defaultStorage.format.rawValue)-g\(group)-sidecars"
    }

    private static func checkedAudioStops(
        from stops: [SmeltCAMIR.StopCondition]
    ) throws -> [CheckedProjectionAudioStop] {
        try stops.map { stop in
            switch stop.kind {
            case .codecEOS:
                try require(stop.value == nil, "audio codec-eos stop value drifted")
                return .codecEOS
            case .hostCancel:
                try require(stop.value == nil, "audio host-cancel stop value drifted")
                return .hostCancel
            case .maxFrames:
                return .maxFrames(try requireValue(stop.value, "audio max-frames value missing"))
            case .eosToken, .maxSteps:
                throw SmeltCAMCheckedPackageProjectionError.unsupported(
                    "audio stop condition \(stop.kind.rawValue) drifted"
                )
            }
        }
    }

    private static func derivedAudioMaxFrames(from stops: [CheckedProjectionAudioStop]) throws -> Int {
        try exactlyOne(
            stops.compactMap { stop in
                if case .maxFrames(let value) = stop { return value }
                return nil
            },
            "audio max-frames stop"
        )
    }

    private static func derivedAudioCodecEOSTokens(
        from cam: SmeltCAMIR,
        graph: CheckedProjectionAudioGraph
    ) throws -> [Int32] {
        let feedbackTarget = try nodePort(graph.feedbackEdge.to, label: "audio feedback target")
        let talkerNode = try graphNode(cam, feedbackTarget.node)
        let talkerBlockID = try requireValue(talkerNode.block, "audio talker block missing")
        let eos = try intBlockRequirement(
            "codec-eos",
            in: try block(cam, talkerBlockID).shape.requirements,
            label: "audio codec eos"
        )
        return [Int32(eos)]
    }

    private static func derivedAudioCodecBlockID(
        from graph: CheckedProjectionAudioGraph
    ) throws -> String {
        try exactlyOne(
            graph.blocks.filter { $0.operatorName == .codecDecoder }.map(\.id),
            "audio codec block"
        )
    }

    private static func derivedAudioResidualSlots(
        from graph: CheckedProjectionAudioGraph
    ) throws -> Int {
        let mtp = try exactlyOne(
            graph.blocks.filter { block in
                block.requirements.contains { $0.key == "residual-codebooks" }
            },
            "audio residual codebook block"
        )
        let codebooks = try intValue(
            try requireValue(
                mtp.requirements.first { $0.key == "codebooks" }?.value,
                "audio codebooks missing"
            ),
            label: "audio codebooks"
        )
        let residualCodebooks = try intValue(
            try requireValue(
                mtp.requirements.first { $0.key == "residual-codebooks" }?.value,
                "audio residual codebooks missing"
            ),
            label: "audio residual codebooks"
        )
        try require(codebooks == residualCodebooks, "audio residual codebooks drifted")
        try require(
            codebooks == CheckedDerivedAudioRuntimeContract.codebooks,
            "audio codebooks \(codebooks) drifted from runtime "
                + "\(CheckedDerivedAudioRuntimeContract.codebooks)"
        )
        try require(
            residualCodebooks == CheckedDerivedAudioRuntimeContract.codebooks,
            "audio residual codebooks \(residualCodebooks) drifted from runtime "
                + "\(CheckedDerivedAudioRuntimeContract.codebooks)"
        )
        return CheckedDerivedAudioRuntimeContract.residualSlots
    }

    private static func derivedAudioPackageSpec(
        from cam: SmeltCAMIR,
        artifactRoot: String,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws -> SmeltPackageSpec {
        try validateDerivedAudioCAM(cam, profile: profile)
        let manifest = try derivedAudioManifest(at: artifactRoot)
        try validateDerivedAudioManifest(manifest, packagePath: artifactRoot, profile: profile)
        try validateDerivedAudioCheckpointMaps(manifest: manifest, profile: profile)
        let specs = try manifest.weights.map {
            try derivedAudioWeightSpec(from: $0, profile: profile)
        }
        let tensorBlocks = try derivedAudioTensorBlocks(for: specs, profile: profile)
        let tensorSourceDTypes = try derivedAudioTensorSourceDTypes(for: specs, profile: profile)
        try validateDerivedAudioWeightLayout(manifest: manifest, specs: specs)

        let lowered = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            packageName: SmeltQwen3TTSPackageProfiles.runnable.packageName,
            sourcePath: artifactRoot,
            modelName: manifest.modelName,
            eosTokens: manifest.eosTokens,
            tokenizerFiles: try requireValue(
                manifest.tokenizerFiles,
                "derived audio manifest tokenizer files missing"
            ),
            decode: try requireValue(
                manifest.decode,
                "derived audio manifest decode policy missing"
            ),
            tensorBlocks: tensorBlocks,
            tensorSourceDTypes: tensorSourceDTypes,
            pipelines: manifest.pipelines,
            pageSize: manifest.pageSize
        )
        try validateDerivedAudioLoweredSpec(lowered, profile: profile)

        return SmeltPackageSpec(
            version: lowered.version,
            packageName: lowered.packageName,
            modelName: lowered.modelName,
            sources: lowered.sources,
            blocks: lowered.blocks,
            loop: lowered.loop,
            runtime: lowered.runtime,
            architectureConfig: lowered.architectureConfig,
            tensors: lowered.tensors,
            quantization: lowered.quantization,
            sidecars: lowered.sidecars,
            artifacts: packageDescriptorArtifacts(appendedTo: lowered.artifacts),
            outputFiles: packageDescriptorOutputFiles(appendedTo: lowered.outputFiles),
            tokenizer: lowered.tokenizer,
            inference: lowered.inference,
            decode: lowered.decode,
            validation: lowered.validation
        )
    }

    private static func validateDerivedAudioCAM(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        try require(cam.imports.isEmpty, "\(profile.id) does not project imported modules")
        for expectedSource in profile.sources {
            try requireSource(try source(cam, expectedSource.id), matches: expectedSource)
        }
        try validateDerivedAudioCapabilities(cam, profile: profile)
        try validateDerivedAudioBlocks(cam, profile: profile)
        try validateDerivedAudioGraph(cam, profile: profile)
        try validateDerivedAudioFlow(cam, profile: profile)
        try validateDerivedAudioTensors(cam, profile: profile)
        try validateDerivedAudioQuantization(cam, profile: profile)
        try validateDerivedAudioCompile(cam)
        try require(cam.sourceQuantization.isEmpty, "audio source quantization drifted")
    }

    private static func validateDerivedAudioCapabilities(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let decisions = try [
            capabilities.resolve(.runAudio),
            capabilities.resolve(.serveAudio),
            capabilities.resolve(.traceTextSynthesize),
            capabilities.resolve(.bakeVoiceDefaults),
        ]
        let exportID = try requireValue(decisions.first?.exportID, "audio capability export missing")
        for decision in decisions {
            try require(decision.flowID == profile.flowID, "audio capability flow drifted")
            try require(decision.exportID == exportID, "audio capability export drifted")
        }
    }

    private static func validateDerivedAudioBlocks(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        let actualBlocks = cam.blocks
        try require(actualBlocks.count == profile.graph.blocks.count, "audio block count drifted")
        for expected in profile.graph.blocks {
            let actual = try requireValue(
                actualBlocks.first { $0.id == expected.id },
                "audio block \(expected.id) missing"
            )
            try require(actual.operatorName == expected.operatorName, "\(expected.id) operator drifted")
            try require(actual.shape.derivation == expected.derivation, "\(expected.id) derivation drifted")
            try require(actual.annotations.isEmpty, "\(expected.id) annotations drifted")
            try require(
                blockRequirementList(actual.shape.requirements) == expected.requirements,
                "\(expected.id) requirements drifted"
            )
            switch expected.operatorName {
            case .codecDecoder:
                try require(actual.shape.codecDecoder?.streaming == true, "\(expected.id) streaming drifted")
                try require(actual.shape.transformer == nil, "\(expected.id) transformer drifted")
                try require(actual.shape.frontend == nil, "\(expected.id) frontend drifted")
            case .ttsFrontend:
                try require(
                    actual.shape.frontend?.speakerConditioning == true,
                    "\(expected.id) speaker conditioning drifted"
                )
                try require(actual.shape.transformer == nil, "\(expected.id) transformer drifted")
                try require(actual.shape.codecDecoder == nil, "\(expected.id) codec decoder drifted")
            case .codecHead:
                try require(actual.shape.transformer == nil, "\(expected.id) transformer drifted")
                try require(actual.shape.frontend == nil, "\(expected.id) frontend drifted")
                try require(actual.shape.codecDecoder == nil, "\(expected.id) codec decoder drifted")
            case .transformer:
                try require(actual.shape.transformer != nil, "\(expected.id) transformer shape missing")
                try require(actual.shape.codecDecoder == nil, "\(expected.id) codec decoder drifted")
                try require(actual.shape.frontend == nil, "\(expected.id) frontend drifted")
            case .transformerEncoder, .transformerDecoder, .patchEncoder,
                 .discreteAudioEncoder, .adapter:
                throw SmeltCAMCheckedPackageProjectionError.unsupported(
                    "\(expected.id) operator does not lower as derived audio"
                )
            }
        }
    }

    private static func validateDerivedAudioGraph(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        try require(
            cam.graphNodes.count == profile.graph.graphNodes.count,
            "audio graph node count drifted"
        )
        for expected in profile.graph.graphNodes {
            let actual = try graphNode(cam, expected.id)
            try require(actual.implementation == expected.implementation, "\(expected.id) implementation drifted")
            try require(actual.block == expected.block, "\(expected.id) block binding drifted")
            try require(actual.source == nil, "\(expected.id) source drifted")
            try require(actual.imported == nil, "\(expected.id) import drifted")
            try require(actual.inputs == expected.inputs, "\(expected.id) inputs drifted")
            try require(actual.outputs == expected.outputs, "\(expected.id) outputs drifted")
            try require(actual.annotations == expected.annotations, "\(expected.id) annotations drifted")
        }

        let expectedEdges = profile.graph.graphEdges.map {
            SmeltCAMIR.GraphEdge(from: $0.from, to: $0.to, type: $0.type)
        }
        try require(
            graphEdgeSignatures(cam.graphEdges) == graphEdgeSignatures(expectedEdges),
            "audio graph edges drifted"
        )
        try require(
            cam.feedbackEdges == [
                .init(
                    from: profile.graph.feedbackEdge.from,
                    to: profile.graph.feedbackEdge.to
                ),
            ],
            "audio feedback edge drifted"
        )
    }

    private static func validateDerivedAudioFlow(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        let flow = try checkedAudioFlow(from: cam, profile: profile)
        try require(
            flow.emit == [.node(profile.graph.emitNode, profile.graph.emitPort)],
            "audio flow emit endpoint drifted"
        )
        let setup = try exactlyOne(
            flow.phases.filter { $0.role == .setup },
            "audio setup phase"
        )
        try require(
            try nodeCallIDs(setup) == profile.graph.setupCalls,
            "audio setup calls drifted"
        )
        let stepPhases = flow.phases.filter { $0.role == .step }
        try require(
            stepPhases.count == profile.graph.stepCallsByLabel.count,
            "audio step phase count drifted"
        )
        for phase in stepPhases {
            let label = try requireValue(phase.label, "audio step label missing")
            let expectedCalls = try requireValue(
                profile.graph.stepCallsByLabel[label],
                "audio step label \(label) drifted"
            )
            try require(try nodeCallIDs(phase) == expectedCalls, "audio \(label) calls drifted")
        }
        try require(
            audioStops(flow.stop) == audioStops(profile.graph.stops),
            "audio stop conditions drifted"
        )
    }

    private static func validateDerivedAudioTensors(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        try require(
            tensorMapSignatures(cam.tensors) == tensorMapSignatures(profile.tensorMaps),
            "audio tensor partitions drifted"
        )
    }

    private static func validateDerivedAudioQuantization(
        _ cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        let defaultRule = try exactlyOne(
            cam.quantization.filter { $0.action == .default },
            "audio default quant rule"
        )
        try require(
            defaultRule.selector == .init("*", source: profile.modelSourceID),
            "audio default quant selector drifted"
        )
        try require(defaultRule.resolution == .declaredTensor, "audio default quant resolution drifted")
        try require(defaultRule.source == nil, "audio default quant source drifted")
        try require(defaultRule.priority == nil, "audio default quant priority drifted")
        try require(
            defaultRule.storage?.format == profile.quantization.defaultStorage.format,
            "audio default quant storage drifted"
        )
        try require(
            defaultRule.storage?.groupSize == profile.quantization.defaultStorage.groupSize,
            "audio default quant group drifted"
        )
        try require(defaultRule.storage?.computeDType == nil, "audio default quant compute dtype drifted")

        let preserves = cam.quantization.filter { $0.action == .preserve }
        try require(
            preserves.map(\.selector.pattern).sorted() == profile.quantization.preservedPatterns.sorted(),
            "audio preserve quant patterns drifted"
        )
        for rule in preserves {
            try require(rule.selector.source == profile.modelSourceID, "audio preserve source drifted")
            try require(rule.source == profile.modelSourceID, "audio preserve owner source drifted")
            try require(rule.storage == nil, "audio preserve storage drifted")
            try require(rule.calibration == nil, "audio preserve calibration drifted")
            try require(
                rule.priority == patternSpecificity(rule.selector.pattern),
                "audio preserve priority drifted"
            )
            try require(
                rule.resolution == .declaredTensor || rule.resolution == .sourceDeferred,
                "audio preserve resolution drifted"
            )
        }
        try require(
            cam.quantization.filter { $0.action == .store || $0.action == .quantize }.isEmpty,
            "audio quantization action drifted"
        )
    }

    private static func validateDerivedAudioCompile(_ cam: SmeltCAMIR) throws {
        try require(
            try uniqueConstraintValue(cam.compile, key: "target") == "metal",
            "audio compile target drifted"
        )
        try require(
            try uniqueConstraintValue(cam.compile, key: "layout") == "memory-neutral",
            "audio compile layout drifted"
        )
        try require(cam.compile.count == 2, "audio compile constraints drifted")
    }

    private static func derivedAudioManifest(at artifactRoot: String) throws -> Qwen3TTSManifest {
        let manifestURL = URL(fileURLWithPath: artifactRoot, isDirectory: true)
            .appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "derived audio manifest.json missing in artifact root"
            )
        }
        return try Qwen3TTSManifest.decode(from: Data(contentsOf: manifestURL))
    }

    private static func validateDerivedAudioManifest(
        _ manifest: Qwen3TTSManifest,
        packagePath: String,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        try manifest.validateQwen3TTSValidation(packagePath: packagePath)
        try require(manifest.modelName == profile.manifest.modelName, "derived audio model name drifted")
        try require(manifest.pageSize == profile.manifest.pageSize, "derived audio page size drifted")
        try require(
            manifest.tokenizerFiles == profile.manifest.tokenizerFiles,
            "derived audio tokenizer files drifted"
        )
        try require(manifest.eosTokens == profile.manifest.eosTokens, "derived audio EOS tokens drifted")
        let blocks = try requireValue(manifest.blocks, "derived audio manifest block graph missing")
        try require(
            try SmeltRuntimeGraphPolicy.resolve(blocks: blocks) == .sidecarTextToCodecAudio,
            "derived audio manifest runtime graph policy drifted"
        )
        try require(manifest.loop != nil, "derived audio manifest loop missing")
        try require(manifest.decode != nil, "derived audio manifest decode policy missing")
        let validation = try requireValue(
            manifest.validation,
            "derived audio manifest validation missing"
        )
        try require(
            validation.performanceGate == profile.manifest.performanceGate,
            "derived audio manifest performance gate drifted"
        )
        try require(
            validation.performanceProfile?.gate == profile.manifest.performanceGate,
            "derived audio manifest performance profile drifted"
        )
        let structureProfile = try requireValue(
            validation.structureProfile,
            "derived audio manifest structure profile missing"
        )
        try require(
            structureProfile.id == profile.manifest.structureProfileID,
            "derived audio manifest structure profile drifted"
        )
        let requiredFiles = Set(structureProfile.requiredFiles)
        for path in profile.manifest.sidecarPaths + profile.manifest.tokenizerFiles {
            try require(requiredFiles.contains(path), "derived audio structure file \(path) missing")
        }
    }

    private static func validateDerivedAudioCheckpointMaps(
        manifest: Qwen3TTSManifest,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        let weightNames = manifest.weights.map(\.name)
        try require(!weightNames.isEmpty, "derived audio manifest weights missing")

        for tensorMap in profile.tensorMaps {
            let matches = weightNames.filter {
                globPattern(tensorMap.selector.pattern, matches: $0)
            }
            try require(
                !matches.isEmpty,
                "audio tensor map \(tensorMap.selector.pattern) matched no manifest weights"
            )
        }

        for weightName in weightNames {
            let matches = profile.tensorMaps.filter {
                globPattern($0.selector.pattern, matches: weightName)
            }
            try require(
                matches.count == 1,
                "audio manifest weight \(weightName) must match exactly one CAM tensor map"
            )
        }

        let evidencePatternsByBlock = Dictionary(uniqueKeysWithValues: profile.graph.blocks.map { block in
            let patterns = block.requirements.compactMap { requirement -> String? in
                guard requirement.key == "tensor-evidence" else { return nil }
                return requirement.value
            }
            return (block.id, patterns)
        })

        for block in profile.graph.blocks {
            let evidencePatterns = evidencePatternsByBlock[block.id] ?? []
            for pattern in evidencePatterns {
                let evidenceWeights = weightNames.filter { globPattern(pattern, matches: $0) }
                try require(
                    !evidenceWeights.isEmpty,
                    "audio \(block.id) tensor evidence \(pattern) missing from manifest"
                )
                for weightName in evidenceWeights {
                    let tensorMap = try exactlyOne(
                        profile.tensorMaps.filter {
                            globPattern($0.selector.pattern, matches: weightName)
                        },
                        "audio tensor map for \(weightName)"
                    )
                    try require(
                        tensorMap.target.block == block.id,
                        "audio \(block.id) tensor evidence \(weightName) maps to \(tensorMap.target.block)"
                    )
                }
            }
        }

        for weightName in weightNames {
            let tensorMap = try exactlyOne(
                profile.tensorMaps.filter {
                    globPattern($0.selector.pattern, matches: weightName)
                },
                "audio tensor map for \(weightName)"
            )
            let evidencePatterns = evidencePatternsByBlock[tensorMap.target.block] ?? []
            let evidenceMatches = evidencePatterns.filter {
                globPattern($0, matches: weightName)
            }
            try require(
                !evidenceMatches.isEmpty,
                "audio manifest weight \(weightName) has no tensor evidence for \(tensorMap.target.block)"
            )
        }

        let residualSlots = try derivedAudioResidualSlots(from: profile.graph)
        let lmHeadCount = weightNames.filter {
            globPattern("talker.code_predictor.lm_head.*.weight", matches: $0)
        }.count
        let codecEmbeddingCount = weightNames.filter {
            globPattern("talker.code_predictor.model.codec_embedding.*.weight", matches: $0)
        }.count
        for index in 0..<residualSlots {
            try require(
                weightNames.contains("talker.code_predictor.lm_head.\(index).weight"),
                "audio residual lm_head.\(index) missing from manifest"
            )
            try require(
                weightNames.contains("talker.code_predictor.model.codec_embedding.\(index).weight"),
                "audio residual codec_embedding.\(index) missing from manifest"
            )
        }
        try require(
            lmHeadCount == residualSlots,
            "audio residual lm_head count \(lmHeadCount) drifted from \(residualSlots)"
        )
        try require(
            codecEmbeddingCount == residualSlots,
            "audio residual codec_embedding count \(codecEmbeddingCount) drifted from \(residualSlots)"
        )

        var hasDefaultQuantizedWeight = false
        for entry in manifest.weights {
            let dtypeString = entry.dtype ?? Qwen3TTSPackageBuilder.WeightDType.f32.rawValue
            guard let dtype = Qwen3TTSPackageBuilder.WeightDType(rawValue: dtypeString) else {
                throw SmeltCAMCheckedPackageProjectionError.unsupported(
                    "derived audio manifest weight \(entry.name) dtype '\(dtypeString)' is unsupported"
                )
            }
            let preserved = profile.quantization.preservedPatterns.contains {
                globPattern($0, matches: entry.name)
            }
            if dtype == .u4 {
                try require(
                    !preserved,
                    "audio preserved weight \(entry.name) is stored u4"
                )
                hasDefaultQuantizedWeight = true
            } else {
                try require(
                    preserved,
                    "audio non-u4 weight \(entry.name) is not covered by CAM preserve policy"
                )
            }
        }
        try require(hasDefaultQuantizedWeight, "audio manifest has no default affine-u4 weights")
    }

    private static func derivedAudioWeightSpec(
        from entry: Qwen3TTSManifest.Entry,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws -> Qwen3TTSPackageBuilder.WeightSpec {
        try require(!entry.name.isEmpty, "derived audio manifest weight name missing")
        try require(
            !entry.shape.isEmpty && entry.shape.allSatisfy { $0 > 0 },
            "derived audio manifest weight \(entry.name) shape drifted"
        )
        let dtypeString = entry.dtype ?? Qwen3TTSPackageBuilder.WeightDType.f32.rawValue
        guard let dtype = Qwen3TTSPackageBuilder.WeightDType(rawValue: dtypeString) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "derived audio manifest weight \(entry.name) dtype '\(dtypeString)' is unsupported"
            )
        }
        if dtype == .u4 {
            try require(
                entry.groupSize == profile.quantization.defaultStorage.groupSize,
                "derived audio manifest weight \(entry.name) u4 group drifted"
            )
            try require(entry.scaleOffset != nil, "derived audio manifest weight \(entry.name) scale offset missing")
            try require(
                entry.scaleByteLength != nil,
                "derived audio manifest weight \(entry.name) scale byte length missing"
            )
            try require(entry.biasOffset != nil, "derived audio manifest weight \(entry.name) bias offset missing")
            try require(
                entry.biasByteLength != nil,
                "derived audio manifest weight \(entry.name) bias byte length missing"
            )
        } else {
            try require(entry.groupSize == nil, "derived audio manifest weight \(entry.name) group drifted")
            try require(entry.scaleOffset == nil, "derived audio manifest weight \(entry.name) scale offset drifted")
            try require(
                entry.scaleByteLength == nil,
                "derived audio manifest weight \(entry.name) scale byte length drifted"
            )
            try require(entry.biasOffset == nil, "derived audio manifest weight \(entry.name) bias offset drifted")
            try require(
                entry.biasByteLength == nil,
                "derived audio manifest weight \(entry.name) bias byte length drifted"
            )
        }
        return Qwen3TTSPackageBuilder.WeightSpec(
            name: entry.name,
            shape: entry.shape,
            dtype: dtype,
            groupSize: entry.groupSize
        )
    }

    private static func derivedAudioTensorBlocks(
        for specs: [Qwen3TTSPackageBuilder.WeightSpec],
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws -> [String: String] {
        var result: [String: String] = [:]
        for spec in specs {
            let tensorMap = try exactlyOne(
                profile.tensorMaps.filter {
                    globPattern($0.selector.pattern, matches: spec.name)
                },
                "audio CAM tensor block for \(spec.name)"
            )
            result[spec.name] = tensorMap.target.block
        }
        return result
    }

    private static func derivedAudioTensorSourceDTypes(
        for specs: [Qwen3TTSPackageBuilder.WeightSpec],
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws -> [String: SmeltPackageSpec.TensorDType] {
        let sourceDTypeByBlock = Dictionary(uniqueKeysWithValues: try profile.graph.blocks.map { block in
            let requirement = try exactlyOne(
                block.requirements.filter { $0.key == "source-dtype" },
                "audio \(block.id) source dtype"
            )
            let raw = try requireValue(
                requirement.value,
                "audio \(block.id) source dtype missing"
            )
            return (block.id, try derivedAudioTensorSourceDType(raw, blockID: block.id))
        })
        var result: [String: SmeltPackageSpec.TensorDType] = [:]
        for spec in specs {
            let tensorMap = try exactlyOne(
                profile.tensorMaps.filter {
                    globPattern($0.selector.pattern, matches: spec.name)
                },
                "audio CAM tensor source dtype for \(spec.name)"
            )
            result[spec.name] = try requireValue(
                sourceDTypeByBlock[tensorMap.target.block],
                "audio CAM source dtype for \(spec.name) missing"
            )
        }
        return result
    }

    private static func derivedAudioTensorSourceDType(
        _ raw: String,
        blockID: String
    ) throws -> SmeltPackageSpec.TensorDType {
        switch raw {
        case "f32":
            return .f32
        case "bf16":
            return .bf16
        default:
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "audio \(blockID) source dtype \(raw) is unsupported"
            )
        }
    }

    private static func validateDerivedAudioWeightLayout(
        manifest: Qwen3TTSManifest,
        specs: [Qwen3TTSPackageBuilder.WeightSpec]
    ) throws {
        let planned = Qwen3TTSPackageBuilder.planLayout(
            specs.sorted { $0.name < $1.name },
            pageSize: manifest.pageSize
        )
        try require(planned.totalBytes == manifest.totalBytes, "derived audio total bytes drifted")
        try require(
            try canonicalJSONData(planned.entries) == canonicalJSONData(manifest.weights),
            "derived audio weight layout drifted"
        )
    }

    private static func validateDerivedAudioLoweredSpec(
        _ spec: SmeltPackageSpec,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws {
        try require(
            try SmeltRuntimeGraphPolicy.resolve(blocks: spec.blocks) == .sidecarTextToCodecAudio,
            "derived audio lowered runtime graph policy drifted"
        )
        try require(spec.loop == SmeltQwen3TTSPackageProfiles.runnable.loop, "derived audio package loop drifted")
        try require(spec.inference?.maxTokens == profile.manifest.maxTokens, "derived audio max tokens drifted")
        try require(spec.inference?.eosTokens == profile.manifest.eosTokens, "derived audio inference EOS drifted")
        try require(
            spec.sidecars.map(\.path).sorted() == profile.manifest.sidecarPaths.sorted(),
            "derived audio sidecar paths drifted"
        )
        let tensorBlocks = Set(spec.tensors.map(\.block))
        let expectedTensorBlocks = Set(profile.graph.blocks.map(\.id))
        try require(
            tensorBlocks == expectedTensorBlocks,
            "derived audio tensor blocks drifted from CAM graph"
        )
        let quantization = try requireValue(
            spec.quantization,
            "derived audio lowered quantization missing"
        )
        try require(quantization.format == .u4, "derived audio lowered quant format drifted")
        try require(
            quantization.groupSize == profile.quantization.defaultStorage.groupSize,
            "derived audio lowered quant group drifted"
        )
    }

    private static func modelIR(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltModelIR {
        if case .roleAttentionCAMShape = profile.assembly {
            return try roleAttentionModelIR(from: cam, profile: profile)
        }

        try require(cam.imports.isEmpty, "\(profile.id) does not project imported modules")
        let (transformer, shape, shapeRequirements) = try transformerShape(from: cam, profile: profile)
        let layers = try requireValue(shape.layers, "layer pattern missing")
        let delta = shape.delta
        let attention = try requireValue(shape.attention, "attention shape missing")
        let ffn = try requireValue(shape.ffn, "ffn shape missing")
        let norm = try requireValue(shape.norm, "norm shape missing")
        let vocab = try requireValue(shape.vocab, "vocab shape missing")
        let rope = try requireValue(attention.rope, "attention rope missing")
        let maxSteps = try flowMaxSteps(from: cam, profile: profile)
        let repeatCount = try requireValue(layers.repeatCount, "layer repeat count missing")
        let modelSource = try source(cam, profile.modelSourceID)
        let checkpointMap = try authoredCheckpointMap(from: modelSource)
        let textPolicyBridge = try textPolicyBridge(from: cam, profile: profile)
        let thinkingTokens = try thinkingTokens(from: shapeRequirements)

        for expectedSource in profile.sources {
            try requireSource(try source(cam, expectedSource.id), matches: expectedSource)
        }
        try require(rope.kind == transformer.ropeKind, "rope layout drifted")
        try require(
            attention.qkNorm == transformer.attentionQKNorm,
            "attention qk norm drifted"
        )
        try require(
            (attention.qkNormMode ?? norm.mode) == transformer.attentionQKNormMode,
            "attention qk norm mode drifted"
        )
        try require(norm.kind == transformer.normKind, "norm type drifted")
        try require(norm.mode == transformer.normMode, "norm mode drifted")
        try require(norm.eps == transformer.normEpsilon, "norm epsilon drifted")
        try require(ffn.activation == transformer.ffnActivation, "ffn activation drifted")

        return SmeltModelIR(
            modelName: modelSource.locator,
            modelRevision: modelSource.revision,
            config: SmeltConfig(
                hiddenSize: try requireValue(shape.hiddenSize, "hidden size missing"),
                numLayers: layers.roles.count * repeatCount,
                vocabSize: vocab.size,
                staticSeqCapacity: transformer.staticSeqCapacity,
                ropeDim: transformer.ropeDim,
                rmsEps: transformer.rmsEpsilon,
                normMode: transformer.modelNormMode,
                delta: try delta.map(deltaConfig),
                attention: SmeltAttentionConfig(
                    qHeads: attention.qHeads,
                    kvHeads: attention.kvHeads,
                    headDim: attention.headDim,
                    gatedQ: transformer.gatedQ,
                    qkvBias: transformer.qkvBias,
                    qkNorm: transformer.attentionQKNorm != nil,
                    qkNormMode: try agentNormMode(from: transformer.attentionQKNormMode),
                    vNorm: attention.vNorm != nil,
                    ropeTheta: Float(try requireValue(rope.theta, "rope theta missing")),
                    ropeDim: transformer.ropeDim,
                    ropeLayout: agentRoPELayout(from: rope.kind)
                ),
                ffn: SmeltFFNConfig(dim: ffn.dim, activation: try ffnActivation(ffn.activation)),
                tiedLMHead: vocab.tiedHead,
                inputFusion: try inputFusionConfig(from: shapeRequirements),
                projectionBanks: projectionBanks(from: shape)
            ),
            layerPattern: SmeltLayerPattern(
                unit: try layers.roles.map(layerType),
                repeats: repeatCount
            ),
            quantization: try quantization(from: cam, profile: profile),
            loading: SmeltLoadingConfig(
                strategy: .mmapPrefault,
                packing: .monolithic,
                checkpointMap: checkpointMap
            ),
            prefill: try prefillConfig(from: cam, profile: profile),
            decode: SmeltDecodeConfig(
                policy: .init(sampler: .init(mode: .greedy), maxSteps: maxSteps),
                policySource: .explicit
            ),
            inference: SmeltInferenceConfig(
                maxTokens: maxSteps,
                maxTokensSource: .explicit,
                eosTokens: eosTokens(from: cam),
                eosTokensSource: .explicit,
                thinkToken: thinkingTokens.token,
                thinkEndToken: thinkingTokens.endToken,
                thinkSkipSuffix: thinkingTokens.skipSuffix,
                chatTemplate: textPolicyBridge.chatTemplate,
                thinkingPolicy: textPolicyBridge.thinkingPolicy,
                toolTranscriptCodec: textPolicyBridge.toolTranscriptCodec,
                promptStateRestoreMode: textPolicyBridge.promptStateRestoreMode
            )
        )
    }

    private static func roleAttentionModelIR(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltModelIR {
        try require(cam.imports.isEmpty, "\(profile.id) does not project imported modules")
        let block = try exactlyOne(cam.blocks, "block")
        try require(block.operatorName == .transformer, "block operator drifted")
        try require(block.shape.derivation == nil, "explicit transformer derivation drifted")
        let requirements = try blockRequirementMap(block.shape.requirements)

        let shape = try requireValue(block.shape.transformer, "trunk transformer shape missing")
        let layers = try requireValue(shape.layers, "layer pattern missing")
        let ffn = try requireValue(shape.ffn, "ffn shape missing")
        let norm = try requireValue(shape.norm, "norm shape missing")
        let vocab = try requireValue(shape.vocab, "vocab shape missing")
        let maxSteps = try flowMaxSteps(from: cam, profile: profile)
        let repeatCount = try requireValue(layers.repeatCount, "layer repeat count missing")
        let modelSource = try primaryModelSource(from: cam)
        let checkpointMap = try authoredCheckpointMap(from: modelSource)
        let textPolicyBridge = try textPolicyBridge(from: cam, profile: profile)

        try require(shape.attention == nil, "single attention shape drifted")
        try require(shape.attentionByRole != nil, "role attention shapes missing")
        try require(shape.delta == nil, "delta shape drifted")
        try require(shape.router == nil, "router shape drifted")
        try require(shape.expert == nil, "expert shape drifted")
        let perLayerInput = try requireValue(shape.perLayerInput, "per-layer input missing")
        let sharedKVLayers = try requireValue(shape.sharedKVLayers, "shared KV layers missing")
        let logitCap = try requireValue(
            shape.logitCap.flatMap(Float.init),
            "logit cap missing"
        )
        let staticSeqCapacity = try intRequirement(
            "static-seq-capacity",
            in: requirements,
            label: "static sequence capacity"
        )
        let modelNormMode = try agentNormMode(
            from: try requireValue(norm.mode, "norm mode missing")
        )
        let normEpsilon = try requireValue(norm.eps, "norm epsilon missing")
        let hiddenActivation = try hiddenActivation(requirements["hidden-activation"])
        let topology = try blockTopology(from: shape)

        return SmeltModelIR(
            modelName: modelSource.locator,
            modelRevision: modelSource.revision,
            config: SmeltConfig(
                hiddenSize: try requireValue(shape.hiddenSize, "hidden size missing"),
                numLayers: layers.roles.count * repeatCount,
                vocabSize: vocab.size,
                vocabSizePerLayerInput: perLayerInput.vocabSize,
                hiddenSizePerLayerInput: perLayerInput.hiddenSize,
                hiddenActivation: hiddenActivation,
                staticSeqCapacity: staticSeqCapacity,
                ropeDim: try ropeDim(for: .global, shape: shape, requirements: requirements),
                rmsEps: try floatValue(normEpsilon, label: "norm epsilon"),
                normMode: modelNormMode,
                blockTopology: topology,
                logitCap: logitCap,
                sharedKVLayers: sharedKVLayers,
                attentionConfigs: try roleAttentionConfigs(
                    from: shape,
                    modelNormMode: modelNormMode,
                    requirements: requirements
                ),
                ffn: SmeltFFNConfig(dim: ffn.dim, activation: try ffnActivation(ffn.activation)),
                tiedLMHead: vocab.tiedHead,
                inputFusion: try inputFusionConfig(from: requirements),
                projectionBanks: projectionBanks(from: shape)
            ),
            layerPattern: SmeltLayerPattern(
                unit: try layers.roles.map(layerType),
                repeats: repeatCount
            ),
            quantization: try quantization(from: cam, profile: profile),
            loading: SmeltLoadingConfig(
                strategy: .mmapPrefault,
                packing: .monolithic,
                checkpointMap: checkpointMap
            ),
            prefill: try prefillConfig(from: cam, profile: profile),
            decode: SmeltDecodeConfig(
                policy: .init(sampler: .init(mode: .greedy), maxSteps: maxSteps),
                policySource: .explicit
            ),
            inference: SmeltInferenceConfig(
                maxTokens: maxSteps,
                maxTokensSource: .explicit,
                eosTokens: eosTokens(from: cam),
                eosTokensSource: .explicit,
                chatTemplate: textPolicyBridge.chatTemplate,
                thinkingPolicy: textPolicyBridge.thinkingPolicy,
                toolTranscriptCodec: textPolicyBridge.toolTranscriptCodec,
                promptStateRestoreMode: textPolicyBridge.promptStateRestoreMode
            )
        )
    }

    private static func projectionBanks(
        from shape: SmeltCAMIR.TransformerShape
    ) -> [SmeltProjectionBankConfig] {
        (shape.projectionBanks ?? []).map {
            SmeltProjectionBankConfig(
                id: $0.id,
                source: $0.source,
                outputs: $0.outputs,
                activationView: $0.activationView,
                activationViewLayerSpans: $0.activationViewLayerSpans
            )
        }
    }

    private static func transformerShape(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> (CheckedProjectionTransformer, SmeltCAMIR.TransformerShape, [String: String]) {
        let block = try exactlyOne(cam.blocks, "block")
        switch profile.assembly {
        case .explicitCAMShape(let transformer):
            try require(block.id == transformer.blockID, "block id drifted")
            try require(block.operatorName == transformer.operatorName, "block operator drifted")
            try require(block.shape.derivation == nil, "explicit transformer derivation drifted")
            let requirements = try blockRequirementMap(block.shape.requirements)
            try require(
                requirements == Dictionary(
                    uniqueKeysWithValues: transformer.requirements.map { ($0.key, $0.value) }
                ),
                "explicit transformer requirements drifted"
            )
            return (
                transformer,
                try requireValue(block.shape.transformer, "trunk transformer shape missing"),
                requirements
            )
        case .roleAttentionCAMShape:
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(profile.id) does not use single-attention transformer projection"
            )
        }
    }

    private static func compactAssemblySpec(
        from spec: SmeltPackageSpec,
        model: SmeltModelIR,
        cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltPackageSpec {
        SmeltPackageSpec(
            version: spec.version,
            packageName: spec.packageName,
            modelName: spec.modelName,
            sources: spec.sources,
            blocks: spec.blocks,
            loop: try textLowererCompatibilityLoop(from: cam, profile: profile),
            runtime: spec.runtime,
            architectureConfig: compactArchitectureConfig(model),
            tensors: spec.tensors,
            quantization: spec.quantization,
            sidecars: spec.sidecars,
            artifacts: packageDescriptorArtifacts(appendedTo: spec.artifacts),
            outputFiles: packageDescriptorOutputFiles(appendedTo: spec.outputFiles),
            tokenizer: spec.tokenizer,
            inference: spec.inference,
            decode: spec.decode,
            validation: profile.validation ?? spec.validation
        )
    }

    // Current text-lowerer compatibility: derive the package loop from the
    // checked CAM flow, mapping CAM sampler/detokenizer nodes onto today's
    // package text-head block without making that bridge CAM identity.
    private static func textLowererCompatibilityLoop(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltLoopSchedule {
        let flow = try checkedFlow(from: cam, profile: profile)
        let setupPhase = try exactlyOne(
            flow.phases.filter { $0.role == .setup },
            "package loop setup phase"
        )
        try validateSetupPhase(try nodeCallIDs(setupPhase), in: cam, profile: profile)
        try validateTokenizerNode(in: cam, profile: profile)

        let stepPhase = try exactlyOne(
            flow.phases.filter { $0.role == .step },
            "package loop step phase"
        )
        try require(stepPhase.label == "decode", "package loop step label drifted")
        let perStepBlocks = try textLowererCompatibilityBlocks(
            from: try nodeCallIDs(stepPhase),
            cam: cam
        )

        try require(
            flow.emit == [.node("detokenizer", "text")],
            "package loop emit endpoint drifted"
        )
        try validateDetokenizerNode(in: cam)

        let setupBlocks: [SmeltLoopSchedule.Phase]
        switch profile.prefill {
        case .required(let prefill):
            _ = try requiredPrefillConfig(from: cam, prefill: prefill)
            setupBlocks = [
                SmeltLoopSchedule.Phase(name: prefill.compileKey, blocks: perStepBlocks),
            ]
        case .none:
            _ = try prefillConfig(from: cam, profile: profile)
            setupBlocks = []
        }

        return SmeltLoopSchedule(
            setup: setupBlocks,
            perStep: [
                SmeltLoopSchedule.Phase(
                    name: try requireValue(stepPhase.label, "package loop step label missing"),
                    blocks: perStepBlocks
                ),
            ],
            emission: .perStep,
            stop: try textLowererCompatibilityStops(from: flow.stop)
        )
    }

    private static func validateSetupPhase(
        _ callIDs: [String],
        in cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws {
        switch profile.textPolicyBridge.setup {
        case .tokenizerOnly:
            try require(callIDs == ["tokenizer"], "package loop setup calls drifted")

        case .promptBuilder(let promptBuilder):
            try require(
                callIDs == [promptBuilder.id, "tokenizer"],
                "package loop setup calls drifted"
            )
            try validatePromptBuilderNode(promptBuilder, in: cam, profile: profile)
        }
    }

    private static func validatePromptBuilderNode(
        _ promptBuilder: CheckedProjectionPromptBuilder,
        in cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws {
        let node = try graphNode(cam, promptBuilder.id)
        try require(node.implementation == .adapter, "prompt builder implementation drifted")
        try require(
            node.annotations == [.init("template", promptBuilder.template)],
            "prompt builder template drifted"
        )
        try require(
            node.inputs == promptBuilder.inputNames.map {
                .init(name: $0, type: .init("text", attributes: ["encoding": "utf8"]))
            },
            "prompt builder inputs drifted"
        )
        try require(
            node.outputs == [
                .init(
                    name: promptBuilder.outputName,
                    type: .init(promptBuilder.outputName)
                ),
            ],
            "prompt builder outputs drifted"
        )

        for input in promptBuilder.inputNames {
            try require(
                cam.graphEdges.contains {
                    $0.from == .moduleInput(input)
                        && $0.to == .node(promptBuilder.id, input)
                },
                "prompt builder \(input) edge drifted"
            )
        }
        try require(
            cam.graphEdges.contains {
                $0.from == .node(promptBuilder.id, promptBuilder.outputName)
                    && $0.to == .graphValue(promptBuilder.outputName)
            },
            "prompt builder output edge drifted"
        )
        try require(
            cam.graphEdges.contains {
                $0.from == .graphValue(promptBuilder.outputName)
                    && $0.to == .node("tokenizer", profile.textPolicyBridge.tokenizerInputName)
            },
            "prompt builder tokenizer edge drifted"
        )
    }

    private static func textLowererCompatibilityBlocks(
        from callIDs: [String],
        cam: SmeltCAMIR
    ) throws -> [String] {
        try require(callIDs == ["trunk", "sampler"], "package loop step calls drifted")
        try validateTrunkNode(in: cam)
        try validateSamplerNode(in: cam)
        return ["trunk", "text-head"]
    }

    private static func textLowererCompatibilityStops(
        from stops: [SmeltCAMIR.StopCondition]
    ) throws -> [SmeltLoopSchedule.Stop] {
        let kinds = Set(stops.map(\.kind))
        try require(
            kinds == [.eosToken, .maxSteps, .hostCancel],
            "package loop stop condition drifted"
        )
        try require(
            stops.contains { $0.kind == .maxSteps && $0.value != nil },
            "package loop max-steps stop missing value"
        )
        try require(
            stops.contains { $0.kind == .eosToken },
            "package loop eos-token stop missing"
        )
        return [.eosToken, .maxSteps, .hostCancel]
    }

    private static func nodeCallIDs(_ phase: SmeltCAMIR.FlowPhase) throws -> [String] {
        try phase.calls.map { call in
            guard call.kind == .node, let node = call.node, call.imported == nil else {
                throw SmeltCAMCheckedPackageProjectionError.unsupported(
                    "package loop imports are not supported"
                )
            }
            try require(call.entrypoint == nil, "package loop entrypoint drifted")
            return node
        }
    }

    private static func validateTokenizerNode(
        in cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws {
        let node = try graphNode(cam, "tokenizer")
        try require(node.implementation == .native, "tokenizer implementation drifted")
        try require(annotation("tag", in: node) == "text-tokenizer", "tokenizer tag drifted")
        _ = try tokenizerThinkingPolicy(node)
        let promptTemplate = try promptTemplate(from: node)
        try require(
            annotation("assistant-prelude", in: node)
                == expectedAssistantPrelude(for: promptTemplate),
            "tokenizer assistant prelude drifted"
        )
        try require(
            node.inputs == [
                .init(
                    name: profile.textPolicyBridge.tokenizerInputName,
                    type: profile.textPolicyBridge.tokenizerInputType
                ),
            ],
            "tokenizer inputs drifted"
        )
        try require(
            node.outputs == [.init(name: "tokens", type: .init("tokens"))],
            "tokenizer outputs drifted"
        )
    }

    private static func tokenizerThinkingPolicy(
        _ node: SmeltCAMIR.GraphNode
    ) throws -> SmeltThinkingPolicy {
        let rawValue = try requireValue(
            annotation("thinking-policy", in: node),
            "tokenizer thinking policy missing"
        )
        guard let policy = SmeltThinkingPolicy(rawValue: rawValue) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "tokenizer thinking policy '\(rawValue)' is unsupported"
            )
        }
        return policy
    }

    private static func validateTrunkNode(in cam: SmeltCAMIR) throws {
        let node = try graphNode(cam, "trunk")
        try require(node.implementation == .compiled, "trunk implementation drifted")
        try require(node.block == "trunk", "trunk block binding drifted")
        try require(annotation("artifact", in: node) == "baked-inline", "trunk artifact drifted")
        let state = try requireValue(annotation("state", in: node), "trunk state missing")
        try require(state.split(separator: ",").contains("kv-cache"), "trunk kv-cache state drifted")
        try require(annotation("feedback", in: node) == "tokens", "trunk feedback drifted")
        try require(
            node.inputs == [.init(name: "tokens", type: .init("tokens"))],
            "trunk inputs drifted"
        )
        try require(
            node.outputs == [.init(name: "hidden", type: .init("hidden"))],
            "trunk outputs drifted"
        )
    }

    private static func validateSamplerNode(in cam: SmeltCAMIR) throws {
        let node = try graphNode(cam, "sampler")
        try require(node.implementation == .native, "sampler implementation drifted")
        try require(node.annotations.count == 2, "sampler annotations drifted")
        try require(annotation("tag", in: node) == "sampler", "sampler tag drifted")
        try require(annotation("state", in: node) == "sampler", "sampler state drifted")
        try require(
            node.inputs == [.init(name: "hidden", type: .init("hidden"))],
            "sampler inputs drifted"
        )
        try require(
            node.outputs == [.init(name: "tokens", type: .init("tokens"))],
            "sampler outputs drifted"
        )
        try require(
            cam.feedbackEdges == [.init(from: .node("sampler", "tokens"), to: .node("trunk", "tokens"))],
            "sampler feedback drifted"
        )
    }

    private static func validateDetokenizerNode(in cam: SmeltCAMIR) throws {
        let node = try graphNode(cam, "detokenizer")
        try require(node.implementation == .native, "detokenizer implementation drifted")
        try require(
            node.annotations == [.init("tag", "text-detokenizer")],
            "detokenizer annotations drifted"
        )
        try require(
            node.inputs == [.init(name: "tokens", type: .init("tokens"))],
            "detokenizer inputs drifted"
        )
        try require(
            node.outputs == [.init(name: "text", type: .init("text", attributes: ["encoding": "utf8"]))],
            "detokenizer outputs drifted"
        )
    }

    private static func graphNode(_ cam: SmeltCAMIR, _ id: String) throws -> SmeltCAMIR.GraphNode {
        try requireValue(cam.graphNodes.first { $0.id == id }, "graph node \(id) missing")
    }

    private static func graphEdgeSignatures(_ edges: [SmeltCAMIR.GraphEdge]) -> [String] {
        edges.map { edge in
            "\(endpointSignature(edge.from))->\(endpointSignature(edge.to)):\(typeSignature(edge.type))"
        }.sorted()
    }

    private static func tensorMapSignatures(_ tensors: [SmeltCAMIR.TensorMap]) -> [String] {
        tensors.map { tensor in
            "\(tensor.source):\(tensor.selector.source ?? "").\(tensor.selector.pattern)"
                + "->\(tensor.target.block).\(tensor.target.selector):\(tensor.owner)"
        }.sorted()
    }

    private static func nodePort(
        _ endpoint: SmeltCAMIR.EndpointRef,
        label: String
    ) throws -> (node: String, port: String) {
        guard endpoint.kind == .nodePort,
              let node = endpoint.node,
              let port = endpoint.port
        else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported("\(label) is not a node port")
        }
        return (node, port)
    }

    private static func endpointSignature(_ endpoint: SmeltCAMIR.EndpointRef) -> String {
        switch endpoint.kind {
        case .moduleInput:
            return "input:\(endpoint.name ?? "")"
        case .moduleOutput:
            return "output:\(endpoint.name ?? "")"
        case .graphValue:
            return "value:\(endpoint.name ?? "")"
        case .nodePort:
            return "node:\(endpoint.node ?? "").\(endpoint.port ?? "")"
        case .importedPort:
            return "import:\(endpoint.importAlias ?? "").\(endpoint.export ?? "").\(endpoint.port ?? "")"
        }
    }

    private static func typeSignature(_ type: SmeltCAMIR.TypeRef?) -> String {
        guard let type else { return "" }
        let attributes = type.attributes
            .map { "\($0.key)=\($0.value)" }
            .sorted()
            .joined(separator: ",")
        guard !attributes.isEmpty else { return type.name }
        return "\(type.name)[\(attributes)]"
    }

    private static func block(_ cam: SmeltCAMIR, _ id: String) throws -> SmeltCAMIR.Block {
        try requireValue(cam.blocks.first { $0.id == id }, "block \(id) missing")
    }

    private static func gate(_ cam: SmeltCAMIR, _ id: String) throws -> SmeltCAMIR.Gate {
        try requireValue(cam.gates.first { $0.id == id }, "gate \(id) missing")
    }

    private static func annotation(_ key: String, in node: SmeltCAMIR.GraphNode) -> String? {
        node.annotations.first { $0.key == key }?.value
    }

    private static func packageDescriptorArtifacts(
        appendedTo artifacts: [SmeltPackageSpec.Artifact]
    ) -> [SmeltPackageSpec.Artifact] {
        guard !artifacts.contains(where: { $0.path == SmeltCAMPackageDescriptor.packageFileName })
        else {
            return artifacts
        }
        return artifacts + [
            .init(
                id: "cam-descriptor",
                path: SmeltCAMPackageDescriptor.packageFileName,
                role: "cam-descriptor"
            ),
        ]
    }

    private static func packageDescriptorOutputFiles(
        appendedTo outputFiles: SmeltPackageSpec.PackageFileSet
    ) -> SmeltPackageSpec.PackageFileSet {
        var files = outputFiles.files
        if !files.contains(SmeltCAMPackageDescriptor.packageFileName) {
            files.append(SmeltCAMPackageDescriptor.packageFileName)
        }
        return SmeltPackageSpec.PackageFileSet(
            manifest: outputFiles.manifest,
            files: files.sorted(),
            bakeManifest: outputFiles.bakeManifest
        )
    }

    private static func compactArchitectureConfig(_ model: SmeltModelIR) -> SmeltPackageSpecValue {
        let weightTotalBytes = SmeltWeightManifestLoader.totalBytes(
            from: SmeltWeightLayout.computeLayout(from: model)
        )
        var object: [String: SmeltPackageSpecValue] = [
            "hidden_size": .int(model.config.hiddenSize),
            "num_layers": .int(model.config.numLayers),
            "vocab_size": .int(model.config.vocabSize),
            "static_seq_capacity": .int(model.config.compiledSeqCapacity),
            "rope_dim": .int(model.config.ropeDim),
            "num_delta_layers": .int(model.numDeltaLayers),
            "num_attn_layers": .int(model.numAttnLayers),
            "ffn_dim": .int(model.config.ffn.dim),
            "pipelines": .array([]),
            "weight_total_bytes": .int(Int(weightTotalBytes)),
        ]
        if let hiddenActivation = model.config.hiddenActivation {
            object["hidden_activation"] = .string(hiddenActivation.rawValue)
        }
        if model.config.blockTopology != .standard {
            object["block_topology"] = .string(model.config.blockTopology.rawValue)
            object["layers"] = .object([
                "pattern": .array(model.layerPattern.unit.map { .string($0.rawValue) }),
                "repeats": .int(model.layerPattern.repeats),
            ])
            object["attention"] = .object(
                Dictionary(uniqueKeysWithValues: model.config.attentionConfigs.map { role, attention in
                    (
                        role.rawValue,
                        .object([
                            "q_heads": .int(attention.qHeads),
                            "kv_heads": .int(attention.kvHeads),
                            "head_dim": .int(attention.headDim),
                            "qk_norm": .bool(attention.qkNorm),
                            "v_norm": .bool(attention.vNorm),
                            "rope_theta": .number(Double(attention.ropeTheta)),
                            "rope_dim": .int(attention.effectiveRopeDim(default: model.config.ropeDim)),
                            "sliding_window": .int(attention.slidingWindow),
                        ])
                    )
                })
            )
        }
        if model.config.vocabSizePerLayerInput > 0 || model.config.hiddenSizePerLayerInput > 0 {
            object["per_layer_input"] = .object([
                "hidden_size": .int(model.config.hiddenSizePerLayerInput),
                "vocab_size": .int(model.config.vocabSizePerLayerInput),
            ])
        }
        if model.config.sharedKVLayers > 0 {
            object["shared_kv_layers"] = .int(model.config.sharedKVLayers)
        }
        if let logitCap = model.config.logitCap {
            object["logit_cap"] = .number(Double(logitCap))
        }
        if !model.quantization.turboQuantHPatterns.isEmpty {
            object["turbo_quant_h"] = .array(model.quantization.turboQuantHPatterns.map { .string($0) })
        }
        var loading: [String: SmeltPackageSpecValue] = [
            "strategy": .string(model.loading.strategy.rawValue),
            "packing": .string(model.loading.packing.rawValue),
        ]
        if let checkpointMap = model.loading.checkpointMap {
            loading["checkpoint_map"] = .string(checkpointMap.rawValue)
        }
        object["loading"] = .object(loading)
        if let prefill = model.prefill {
            let modelPath = prefill.engine == "metal" ? "prefill.mlmodelc" : prefill.modelPath
            let handoffEntries = SmeltHandoffResolver.resolve(
                families: prefill.handoffFamilies,
                ir: model,
                plan: buildBufferPlan(from: model)
            ).entries.count
            object["prefill"] = .object([
                "engine": .string(prefill.engine),
                "model": .string(modelPath),
                "max_batch_size": .int(prefill.maxBatchSize),
                "handoff_entries": .int(handoffEntries),
            ])
        }
        return .object(object)
    }

    private static func quantization(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltQuantizationConfig {
        let defaultRule = try exactlyOne(
            cam.quantization.filter { $0.action == .default },
            "default quant rule"
        )
        try require(
            defaultRule.storage?.format == profile.quantization.defaultStorage.format,
            "default quant storage drifted"
        )
        try require(
            defaultRule.storage?.groupSize == profile.quantization.defaultStorage.groupSize,
            "default quant group drifted"
        )

        let quantized = cam.quantization.filter { $0.action == .quantize }.map(\.selector.pattern)
        try require(
            quantized == profile.quantization.quantizedPatterns,
            "embedding quantization drifted"
        )
        let excludes = cam.quantization.filter { $0.action == .preserve }.map(\.selector.pattern).sorted()
        try require(
            excludes == profile.quantization.preservedPatterns.sorted(),
            "preserve quant patterns drifted"
        )
        let stored = try cam.quantization.filter { $0.action == .store }.map { rule in
            let storage = try requireValue(rule.storage, "stored quant storage missing")
            try require(storage.computeDType == nil, "stored quant compute dtype drifted")
            return CheckedProjectionStoredPattern(
                pattern: rule.selector.pattern,
                storage: .init(format: storage.format, groupSize: storage.groupSize)
            )
        }
        try require(stored == profile.quantization.storedPatterns, "stored quant patterns drifted")

        return SmeltQuantizationConfig(
            strategy: profile.quantization.strategy,
            groupSize: try requireValue(
                profile.quantization.defaultStorage.groupSize,
                "default quant group missing"
            ),
            excludePatterns: excludes,
            quantizeEmbedding: profile.quantization.quantizesEmbedding,
            turboQuantHPatterns: profile.quantization.storedPatterns
                .filter { $0.storage.format == .turboQuantH }
                .map(\.pattern)
        )
    }

    // CAM owns route selection through export capability, flow, ports, and gate
    // contracts. This function only projects that checked text route to the
    // prompt renderer string required by today's package manifest.
    private static func textPolicyBridge(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> CheckedProjectionResolvedTextPolicyBridge {
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let runDecision = try capabilities.resolve(profile.textPolicyBridge.capabilityRequests.run)
        let benchDecision = try capabilities.resolve(profile.textPolicyBridge.capabilityRequests.benchmark)
        try require(runDecision.flowID == profile.flowID, "run text capability flow drifted")
        try require(benchDecision.flowID == profile.flowID, "bench decode capability flow drifted")
        try require(runDecision.exportID == benchDecision.exportID, "text capability export drifted")

        let tokenizer = try graphNode(cam, "tokenizer")
        let promptTemplate = try promptTemplate(from: tokenizer)
        let thinkingPolicy = try tokenizerThinkingPolicy(tokenizer)
        let toolTranscriptCodec = try toolTranscriptCodec(from: tokenizer)
        let promptStateRestoreMode = try promptStateRestoreMode(
            from: graphNode(cam, "trunk")
        )

        return CheckedProjectionResolvedTextPolicyBridge(
            chatTemplate: promptTemplate,
            thinkingPolicy: thinkingPolicy,
            toolTranscriptCodec: toolTranscriptCodec,
            promptStateRestoreMode: promptStateRestoreMode
        )
    }

    private static func promptStateRestoreMode(
        from trunk: SmeltCAMIR.GraphNode
    ) throws -> SmeltPromptStateRestoreMode {
        let declared = try requireValue(
            annotation("state", in: trunk),
            "trunk state missing"
        )
        let state = Set(declared.split(separator: ",").map(String.init))
        try require(state.contains("kv-cache"), "trunk kv-cache state drifted")
        return SmeltPromptStateRestoreMode.derive(
            fromPersistentStateNames: state
        )
    }

    private static func toolTranscriptCodec(
        from node: SmeltCAMIR.GraphNode
    ) throws -> String? {
        guard let codec = annotation("tool-format", in: node) else {
            // Compatibility bridge for packages authored before tool framing
            // became independent from ChatML role framing.
            return SmeltToolTranscriptCodecName.inferredFromLegacyPromptTemplate(
                annotation("prompt-format", in: node) ?? ""
            )
        }
        guard SmeltToolTranscriptCodecName.isKnown(codec) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "tokenizer tool format '\(codec)' is unsupported"
            )
        }
        return codec
    }

    private static func promptTemplate(
        from node: SmeltCAMIR.GraphNode
    ) throws -> String {
        let promptFormat = try requireValue(
            annotation("prompt-format", in: node),
            "tokenizer prompt format missing"
        )
        guard SmeltPromptTemplateName.isKnownPromptTemplate(promptFormat) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "tokenizer prompt format '\(promptFormat)' is unsupported"
            )
        }
        return SmeltPromptTemplateName.canonicalRoleTemplate(for: promptFormat)
    }

    private static func expectedAssistantPrelude(for promptTemplate: String) -> String? {
        switch promptTemplate {
        case "chatml", "chatml-xml-tools":
            return "preclosed-think"
        case "channel-turns":
            return "thought-channel"
        case "":
            return nil
        default:
            return nil
        }
    }

    private static func checkedTextPolicyBridge(
        tokenizerInputName: String = "prompt",
        tokenizerInputType: SmeltCAMIR.TypeRef = .init("text", attributes: ["encoding": "utf8"]),
        capabilityRequests: CheckedProjectionTextCapabilityRequests = .singleText,
        setup: CheckedProjectionSetup = .tokenizerOnly
    ) -> CheckedProjectionTextPolicyBridge {
        CheckedProjectionTextPolicyBridge(
            tokenizerInputName: tokenizerInputName,
            tokenizerInputType: tokenizerInputType,
            capabilityRequests: capabilityRequests,
            setup: setup
        )
    }

    private static func prefillConfig(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltPrefillConfig? {
        switch profile.prefill {
        case .required(let prefill):
            return try requiredPrefillConfig(from: cam, prefill: prefill)

        case .none(let rejectedCompileKeys, let rejectedSourceIDs, let rejectedGraphNodeIDs):
            for key in rejectedCompileKeys {
                try require(!cam.compile.contains { $0.key == key }, "\(key) compile constraint drifted")
            }
            for sourceID in rejectedSourceIDs {
                try require(!cam.sources.contains { $0.id == sourceID }, "\(sourceID) sidecar source drifted")
            }
            for nodeID in rejectedGraphNodeIDs {
                try require(!cam.graphNodes.contains { $0.id == nodeID }, "\(nodeID) graph node drifted")
            }
            return nil
        }
    }

    private static func requiredPrefillConfig(
        from cam: SmeltCAMIR,
        prefill: CheckedProjectionPrefill
    ) throws -> SmeltPrefillConfig {
        let value = try uniqueConstraintValue(cam.compile, key: prefill.compileKey)
        for sourceID in prefill.rejectedSourceIDs {
            try require(!cam.sources.contains { $0.id == sourceID }, "\(sourceID) sidecar source drifted")
        }
        for nodeID in prefill.rejectedGraphNodeIDs {
            try require(!cam.graphNodes.contains { $0.id == nodeID }, "\(nodeID) graph node drifted")
        }
        let compile = try parsePrefillCompileConstraint(value, engine: prefill.engine)
        try require(compile.batch == prefill.maxBatchSize, "prefill batch drifted")
        return SmeltPrefillConfig(
            engine: prefill.engine,
            modelPath: prefill.modelPath,
            cachePath: prefill.cachePath,
            maxBatchSize: compile.batch,
            handoffFamilies: prefill.handoffNames,
            emitAllLogits: compile.emitAllLogits,
            verifyArgmax: compile.verifyArgmax,
            verifyTokenCapacity: compile.verifyTokenCapacity
        )
    }

    private static func packageInventoryFiles(from cam: SmeltCAMIR) throws -> [String] {
        let requirements = cam.gates.flatMap(\.requirements).filter {
            $0.subject == "package-files"
        }
        guard !requirements.isEmpty else {
            _ = try exactlyOne(requirements, "package inventory requirement")
            return []
        }
        let inventories = try requirements.map { requirement in
            try require(requirement.subject == "package-files", "inventory subject drifted")
            try require(requirement.relation == .include, "inventory relation drifted")
            return requirement.value.split(separator: ",").map(String.init).sorted()
        }
        let uniqueInventories = Dictionary(grouping: inventories, by: { $0 })
            .keys
            .sorted { $0.joined(separator: ",") < $1.joined(separator: ",") }
        guard uniqueInventories.count == 1, let inventory = uniqueInventories.first else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "package inventory requirement expected one unique value, found "
                    + "\(uniqueInventories.count)"
            )
        }
        return inventory
    }

    private static func flowMaxSteps(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> Int {
        let flow = try checkedFlow(from: cam, profile: profile)
        return try requireValue(
            flow.stop.first { $0.kind == .maxSteps }?.value,
            "flow max-steps missing"
        )
    }

    private static func checkedFlow(
        from cam: SmeltCAMIR,
        profile: CheckedPackageProjectionProfile
    ) throws -> SmeltCAMIR.Flow {
        let flow = try exactlyOne(cam.flows, "flow")
        try require(flow.id == profile.flowID, "flow id drifted")
        return flow
    }

    private static func checkedAudioFlow(
        from cam: SmeltCAMIR,
        profile: CheckedDerivedAudioPackageProjectionProfile
    ) throws -> SmeltCAMIR.Flow {
        let flow = try exactlyOne(cam.flows, "audio flow")
        try require(flow.id == profile.flowID, "audio flow id drifted")
        return flow
    }

    private static func eosTokens(from cam: SmeltCAMIR) -> [Int32] {
        cam.flows.flatMap(\.stop)
            .compactMap { stop in stop.kind == .eosToken ? stop.value.map(Int32.init) : nil }
            .sorted()
    }

    private static func audioStops(_ stops: [SmeltCAMIR.StopCondition]) -> [String] {
        stops.map { stop in
            switch stop.kind {
            case .codecEOS:
                return "codec-eos"
            case .hostCancel:
                return "host-cancel"
            case .maxFrames:
                return "max-frames:\(stop.value ?? -1)"
            case .eosToken:
                return "eos-token:\(stop.value ?? -1)"
            case .maxSteps:
                return "max-steps:\(stop.value ?? -1)"
            }
        }.sorted()
    }

    private static func audioStops(_ stops: [CheckedProjectionAudioStop]) -> [String] {
        stops.map { stop in
            switch stop {
            case .codecEOS:
                return "codec-eos"
            case .hostCancel:
                return "host-cancel"
            case .maxFrames(let value):
                return "max-frames:\(value)"
            }
        }.sorted()
    }

    private static func layerType(_ role: SmeltCAMIR.LayerRole) throws -> SmeltLayerType {
        switch role {
        case .delta: return .delta
        case .attention: return .attention
        case .sliding: return .sliding
        case .global: return .global
        }
    }

    private static func ffnActivation(_ activation: SmeltCAMIR.Activation) throws -> SmeltActivation {
        switch activation {
        case .swiglu:
            return .swiglu
        case .geglu:
            return .geglu
        case .gelu, .silu:
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "ffn activation '\(activation.rawValue)' does not lower to the text package compiler"
            )
        }
    }

    private static func roleAttentionConfigs(
        from shape: SmeltCAMIR.TransformerShape,
        modelNormMode: SmeltNormMode,
        requirements: [String: String]
    ) throws -> [SmeltLayerType: SmeltAttentionConfig] {
        let variants = try requireValue(shape.attentionByRole, "role attention shapes missing")

        var result: [SmeltLayerType: SmeltAttentionConfig] = [:]
        for variant in variants {
            let actual = variant.attention
            let rope = try requireValue(actual.rope, "\(variant.role.rawValue) rope missing")
            let role = try layerType(variant.role)
            try require(result[role] == nil, "\(variant.role.rawValue) role drifted")
            result[role] = SmeltAttentionConfig(
                qHeads: actual.qHeads,
                kvHeads: actual.kvHeads,
                headDim: actual.headDim,
                gatedQ: false,
                qkvBias: false,
                qkNorm: actual.qkNorm != nil,
                qkNormMode: try actual.qkNormMode.map(agentNormMode(from:))
                    ?? modelNormMode,
                vNorm: actual.vNorm != nil,
                attnScale: 1,
                ropeTheta: Float(try requireValue(rope.theta, "\(variant.role.rawValue) rope theta missing")),
                ropeDim: try ropeDim(for: variant.role, shape: shape, requirements: requirements),
                ropeLayout: agentRoPELayout(from: rope.kind),
                slidingWindow: actual.window ?? 0
            )
        }
        return result
    }

    private static func ropeDim(
        for role: SmeltCAMIR.LayerRole,
        shape: SmeltCAMIR.TransformerShape,
        requirements: [String: String]
    ) throws -> Int {
        if let value = requirements["\(role.rawValue)-rope-dim"] {
            return try intValue(value, label: "\(role.rawValue) rope dim")
        }
        let variants = try requireValue(shape.attentionByRole, "role attention shapes missing")
        let variant = try requireValue(
            variants.first { $0.role == role },
            "role attention \(role.rawValue) missing"
        )
        return variant.attention.headDim
    }

    private static func deltaConfig(_ delta: SmeltCAMIR.DeltaShape) throws -> SmeltDeltaConfig {
        SmeltDeltaConfig(
            numHeads: delta.heads,
            headDim: delta.headDim,
            convKernel: try requireValue(delta.convKernel, "delta conv kernel missing"),
            qkvDim: try projection("qkv", in: delta),
            zDim: try projection("z", in: delta),
            aDim: try projection("a", in: delta),
            bDim: try projection("b", in: delta)
        )
    }

    private static func projection(_ key: String, in delta: SmeltCAMIR.DeltaShape) throws -> Int {
        try requireValue(delta.projections[key], "delta projection \(key) missing")
    }

    private static func blockRequirementMap(
        _ requirements: [SmeltCAMIR.BlockRequirement]
    ) throws -> [String: String] {
        var result: [String: String] = [:]
        for requirement in requirements {
            try require(result[requirement.key] == nil, "\(requirement.key) requirement drifted")
            result[requirement.key] = requirement.value
        }
        return result
    }

    private static func intRequirement(
        _ key: String,
        in requirements: [String: String],
        label: String
    ) throws -> Int {
        try intValue(try requireValue(requirements[key], "\(label) missing"), label: label)
    }

    private static func intBlockRequirement(
        _ key: String,
        in requirements: [SmeltCAMIR.BlockRequirement],
        label: String
    ) throws -> Int {
        try intValue(
            try blockRequirementValue(key, in: requirements, label: label),
            label: label
        )
    }

    private static func blockRequirementValue(
        _ key: String,
        in requirements: [SmeltCAMIR.BlockRequirement],
        label: String
    ) throws -> String {
        let matches = requirements.filter { $0.key == key }
        let requirement = try exactlyOne(matches, label)
        try require(!requirement.optional, "\(label) is optional")
        return try requireValue(requirement.value, "\(label) missing")
    }

    private static func intValue(_ value: String, label: String) throws -> Int {
        guard let int = Int(value) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(label) is not an integer"
            )
        }
        return int
    }

    private static func floatValue(_ value: String, label: String) throws -> Float {
        guard let float = Float(value) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(label) is not a float"
            )
        }
        return float
    }

    private static func hiddenActivation(_ value: String?) throws -> SmeltHiddenActivation? {
        guard let value else { return nil }
        let raw = value.replacingOccurrences(of: "-", with: "_")
        guard let activation = SmeltHiddenActivation(rawValue: raw) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "hidden activation '\(value)' does not lower to the text package compiler"
            )
        }
        return activation
    }

    private static func agentNormMode(from mode: SmeltCAMIR.NormMode) throws -> SmeltNormMode {
        switch mode {
        case .weight:
            return .weight
        case .onePlusWeight:
            return .onePlusWeight
        }
    }

    private static func agentRoPELayout(
        from kind: SmeltCAMIR.RopeKind
    ) -> SmeltRoPELayout {
        // CAM's current RoPE kinds are NeoX-coordinate layouts. YaRN changes
        // frequency scaling, not the coordinate pairing, so both pair the two
        // halves of the authored rotary span. Never fall back to the runtime's
        // adjacent-pair default after CAM has declared a layout.
        switch kind {
        case .neox, .yarn:
            return .splitHalf
        }
    }

    private static func blockTopology(from shape: SmeltCAMIR.TransformerShape) throws -> SmeltBlockTopology {
        let hasRoleAttention = shape.attentionByRole?.isEmpty == false
        let hasAltUpMarkers = shape.perLayerInput != nil
            || shape.sharedKVLayers != nil
            || shape.logitCap != nil
        try require(
            !hasRoleAttention && !hasAltUpMarkers,
            "non-standard block topology markers are not supported"
        )
        return .standard
    }

    private static func patternSpecificity(_ pattern: String) -> Int {
        pattern.filter { $0 != "*" }.count
    }

    private static func globPattern(_ pattern: String, matches value: String) -> Bool {
        if pattern == "*" { return true }
        if !pattern.contains("*") { return pattern == value }
        let parts = pattern.split(separator: "*", omittingEmptySubsequences: false).map(String.init)
        var remainder = value[...]
        if let first = parts.first, !first.isEmpty {
            guard remainder.hasPrefix(first) else { return false }
            remainder.removeFirst(first.count)
        }
        for part in parts.dropFirst().dropLast() where !part.isEmpty {
            guard let range = remainder.range(of: part) else { return false }
            remainder = remainder[range.upperBound...]
        }
        if let last = parts.last, !last.isEmpty {
            return remainder.hasSuffix(last)
        }
        return true
    }

    private static func blockRequirementList(
        _ requirements: [SmeltCAMIR.BlockRequirement]
    ) -> [CheckedProjectionBlockRequirement] {
        requirements.map {
            CheckedProjectionBlockRequirement(
                key: $0.key,
                value: $0.value,
                optional: $0.optional
            )
        }
    }

    private static func thinkingTokens(
        from requirements: [String: String]
    ) throws -> (token: Int32?, endToken: Int32?, skipSuffix: Int32?) {
        (
            token: try optionalInt32Requirement("think-token", from: requirements),
            endToken: try optionalInt32Requirement("think-end-token", from: requirements),
            skipSuffix: try optionalInt32Requirement("think-skip-suffix", from: requirements)
        )
    }

    private static func optionalInt32Requirement(
        _ key: String,
        from requirements: [String: String]
    ) throws -> Int32? {
        guard let value = requirements[key] else { return nil }
        guard let int = Int32(value) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(key) requirement is not an Int32"
            )
        }
        return int
    }

    private static func source(_ cam: SmeltCAMIR, _ id: String) throws -> SmeltCAMIR.Source {
        try requireValue(cam.sources.first { $0.id == id }, "source \(id) missing")
    }

    private static func primaryModelSource(from cam: SmeltCAMIR) throws -> SmeltCAMIR.Source {
        let tensorSourceIDs = Set(cam.tensors.map(\.source)).sorted()
        let sourceID = try exactlyOne(tensorSourceIDs, "model tensor source")
        let modelSource = try source(cam, sourceID)
        try require(modelSource.kind == "hf", "model tensor source kind drifted")
        return modelSource
    }

    private static func authoredCheckpointMap(
        from source: SmeltCAMIR.Source
    ) throws -> SmeltCheckpointMap {
        guard let rawMap = source.checkpointMap, !rawMap.isEmpty else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(source.id) source checkpoint map missing"
            )
        }
        guard let checkpointMap = SmeltCheckpointMap(rawValue: rawMap) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(source.id) source checkpoint map '\(rawMap)' is unsupported"
            )
        }
        return checkpointMap
    }

    private static func requireSource(
        _ source: SmeltCAMIR.Source,
        matches expected: CheckedProjectionSource
    ) throws {
        try require(source.kind == expected.kind, "\(expected.id) source kind drifted")
        try require(source.locator == expected.locator, "\(expected.id) source drifted")
        try require(source.revision == expected.revision, "\(expected.id) source revision drifted")
        try require(
            source.checkpointMap == expected.checkpointMap,
            "\(expected.id) source checkpoint map drifted"
        )
    }

    private static func uniqueConstraintValue(
        _ constraints: [SmeltCAMIR.Constraint],
        key: String
    ) throws -> String {
        let values = constraints.filter { $0.key == key }.map(\.value)
        return try exactlyOne(values, "\(key) constraint")
    }

    private static func exactlyOne<T>(_ values: [T], _ label: String) throws -> T {
        guard values.count == 1, let value = values.first else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(label) expected exactly one, found \(values.count)"
            )
        }
        return value
    }

    private static func requireValue<T>(_ value: T?, _ message: String) throws -> T {
        guard let value else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(message)
        }
        return value
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(message)
        }
    }

    private static func canonicalJSONData<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
