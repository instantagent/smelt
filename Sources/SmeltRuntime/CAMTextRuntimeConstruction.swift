import Foundation
import SmeltSchema

private enum CAMTextConstructionError: Error, CustomStringConvertible {
    case missingFlow(String)
    case missingNode(String)
    case unsupported(String)
    case invalid(String)

    var description: String {
        switch self {
        case .missingFlow(let flowID):
            return "CAM text flow '\(flowID)' is missing"
        case .missingNode(let nodeID):
            return "CAM text graph node '\(nodeID)' is missing"
        case .unsupported(let detail):
            return "unsupported CAM text graph: \(detail)"
        case .invalid(let detail):
            return "invalid CAM text graph: \(detail)"
        }
    }
}

package struct CAMTextExecutionPlan: Sendable, Equatable {
    package enum ExecutionMode: String, Sendable {
        case decodeOnly = "decode-only"
        case prefillDecode = "prefill-decode"
    }

    package struct StopPolicy: Sendable, Equatable {
        package let eosTokens: [Int32]
        package let maxSteps: Int
        package let honorsHostCancel: Bool
    }

    package struct PromptAdapter: Sendable, Equatable {
        package let nodeID: String
        package let template: String
        package let inputPortNames: [String]
        package let outputName: String
    }

    package let exportID: String
    package let flowID: String
    package let inputPortNames: [String]
    package let outputPortNames: [String]
    package let setupNodeIDs: [String]
    package let stepNodeIDs: [String]
    package let emitEndpoints: [String]
    package let tokenizerNodeID: String
    package let trunkNodeID: String
    package let samplerNodeID: String
    package let detokenizerNodeID: String
    package let promptTemplate: String
    package let thinkingPolicy: SmeltThinkingPolicy
    package let toolTranscriptCodec: String?
    package let promptStateRestoreMode: SmeltPromptStateRestoreMode
    package let assistantPrelude: String?
    package let promptAdapter: PromptAdapter?
    package let executionMode: ExecutionMode
    package let stopPolicy: StopPolicy

    package func inference() -> SmeltInferenceManifest {
        SmeltInferenceManifest(
            maxTokens: stopPolicy.maxSteps,
            eosTokens: stopPolicy.eosTokens,
            thinkToken: nil,
            thinkEndToken: nil,
            thinkSkipSuffix: nil,
            chatTemplate: promptTemplate,
            thinkingPolicy: thinkingPolicy,
            toolTranscriptCodec: toolTranscriptCodec,
            promptStateRestoreMode: promptStateRestoreMode
        )
    }
}

package struct CAMTextRuntimeConstruction: Sendable {
    package let packagePath: String
    package let decision: SmeltCAMPackageCapabilities.Decision
    package let featureContract: SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract
    package let executionPlan: CAMTextExecutionPlan
    package let artifactRoles: [String]
    package let gateIDs: [String]
    package let camSemanticSHA256: String
    package let exportABISHA256: String
    package let prefillBatchSize: Int?
    package let prefillAllLogitsBatchSize: Int?

    package init(
        packagePath: String,
        capabilities: SmeltCAMPackageCapabilities,
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws {
        self.packagePath = packagePath
        self.decision = decision
        featureContract = try capabilities.runtimeAssemblyFeatureContract(for: decision)
        let declaredPrefillBatchSize = Self.prefillBatchSize(
            from: capabilities.compileRequirements
        )
        executionPlan = try Self.makeExecutionPlan(
            capabilities: capabilities,
            decision: decision,
            hasDeclaredPrefill: declaredPrefillBatchSize != nil
        )
        artifactRoles = capabilities.artifactRequirements.map(\.role).sorted()
        gateIDs = decision.matchedGateIDs.sorted()
        camSemanticSHA256 = capabilities.camSemanticSHA256
        exportABISHA256 = capabilities.exportABISHA256
        prefillBatchSize = declaredPrefillBatchSize
        prefillAllLogitsBatchSize = Self.prefillAllLogitsBatchSize(
            from: capabilities.compileRequirements
        )
    }

    package func requirePackagePath(_ path: String) throws {
        guard URL(fileURLWithPath: path).standardizedFileURL.path
            == URL(fileURLWithPath: packagePath).standardizedFileURL.path
        else {
            throw CAMTextConstructionError.invalid("package path mismatch")
        }
    }

    package func makeRuntime(contextLimit: Int?) throws -> SmeltRuntime {
        try SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)
    }

    package func requirePrefillCapacity(tokenCount: Int) throws {
        guard tokenCount > 0 else {
            throw CAMTextConstructionError.invalid("prompt token count must be positive")
        }
        guard let prefillBatchSize else {
            throw CAMTextConstructionError.invalid("CAM route does not declare prefill metal batch")
        }
        guard tokenCount <= prefillBatchSize else {
            throw CAMTextConstructionError.invalid(
                "prompt token count \(tokenCount) exceeds CAM prefill batch "
                + "\(prefillBatchSize)"
            )
        }
    }

    package func requirePrefillAllLogitsCapacity(tokenCount: Int) throws {
        guard tokenCount > 0 else {
            throw CAMTextConstructionError.invalid("prompt token count must be positive")
        }
        guard let prefillAllLogitsBatchSize else {
            throw CAMTextConstructionError.invalid("CAM route does not declare prefill metal all-logits batch")
        }
        guard tokenCount <= prefillAllLogitsBatchSize else {
            throw CAMTextConstructionError.invalid(
                "prompt token count \(tokenCount) exceeds CAM prefill all-logits batch "
                + "\(prefillAllLogitsBatchSize)"
            )
        }
    }

    package func makeModel(contextLimit: Int?, manifest: SmeltManifest) throws -> SmeltModel {
        try SmeltModel(
            package: packagePath,
            contextLimit: contextLimit,
            manifest: manifest.withCAMTextInference(executionPlan.inference())
        )
    }

    package func makeTokenizer() throws -> SmeltTokenizer {
        try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
    }

    package func makeTextRuntime(
        traceDirectory: String?,
        contextLimit: Int?,
        limits: SmeltLimits = SmeltLimits()
    ) throws -> SmeltTextRuntime {
        let (manifest, inference) = try loadManifestConfig()
        return try SmeltTextRuntime(
            model: makeModel(contextLimit: contextLimit, manifest: manifest),
            tokenizer: makeTokenizer(),
            inferenceEOSTokens: inference.eosTokens,
            packagePath: packagePath,
            camSemanticSHA256: camSemanticSHA256,
            exportABISHA256: exportABISHA256,
            traceDirectory: traceDirectory,
            limits: limits
        )
    }

    package func loadManifestConfig() throws -> (manifest: SmeltManifest, inference: SmeltInferenceManifest) {
        let manifestPath = "\(packagePath)/manifest.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try SmeltManifest.decode(from: data)
        let inference = executionPlan.inference()
        return (manifest.withCAMTextInference(inference), inference)
    }

    package func resolveTemplate(cliOverride: String?) throws -> String {
        let packaged = executionPlan.promptTemplate
        if let cliOverride, !cliOverride.isEmpty, cliOverride != packaged {
            throw CAMTextConstructionError.unsupported(
                "CAM text routes do not accept template overrides"
            )
        }
        return packaged
    }

    package func effectiveMaxTokens(_ requested: Int) -> Int {
        min(requested, executionPlan.stopPolicy.maxSteps)
    }

    package func renderPrompt(prompt: String, systemPrompt: String) throws -> (
        prompt: String,
        systemPrompt: String
    ) {
        guard let adapter = executionPlan.promptAdapter else {
            return (prompt, systemPrompt)
        }
        let values = try promptAdapterInputs(
            adapter: adapter,
            prompt: prompt,
            systemPrompt: systemPrompt
        )
        switch adapter.template {
        case "draft-judge":
            return (
                """
                Context:
                \(values["context"] ?? "")

                Candidate:
                \(values["candidate"] ?? "")

                Review the candidate against the context.
                """,
                ""
            )
        case "raw-review":
            return (
                """
                \(values["context"] ?? "")

                \(values["candidate"] ?? "")
                """,
                ""
            )
        default:
            throw CAMTextConstructionError.unsupported(
                "prompt adapter template \(adapter.template)"
            )
        }
    }

    package func runtimeModeDescription(manifest: SmeltManifest) throws -> (
        useModelGenerate: Bool,
        mode: String
    ) {
        switch executionPlan.executionMode {
        case .decodeOnly:
            return (false, "cam-flow decode-only")
        case .prefillDecode:
            try requireManifestMetalPrefill(manifest)
            return (true, "metal-prefill")
        }
    }

    package func debugRequiresDecodeOnly() -> Bool {
        switch executionPlan.executionMode {
        case .decodeOnly:
            return true
        case .prefillDecode:
            return false
        }
    }

    private func promptAdapterInputs(
        adapter: CAMTextExecutionPlan.PromptAdapter,
        prompt: String,
        systemPrompt: String
    ) throws -> [String: String] {
        let supportedInputs: Set<String> = ["candidate", "context"]
        guard Set(adapter.inputPortNames) == supportedInputs else {
            throw CAMTextConstructionError.unsupported(
                "prompt adapter \(adapter.nodeID) inputs "
                    + adapter.inputPortNames.joined(separator: ",")
            )
        }
        return [
            "candidate": prompt,
            "context": systemPrompt,
        ]
    }

    private func requireManifestMetalPrefill(_ manifest: SmeltManifest) throws {
        guard let prefill = manifest.prefill else {
            throw CAMTextConstructionError.invalid("CAM route declares metal prefill but manifest has no prefill section")
        }
        guard prefill.engine == "metal" else {
            throw CAMTextConstructionError.invalid(
                "CAM route declares metal prefill but manifest prefill engine is "
                    + prefill.engine
            )
        }
        if let prefillBatchSize, prefill.maxBatchSize != prefillBatchSize {
            throw CAMTextConstructionError.invalid(
                "CAM route declares prefill batch \(prefillBatchSize) but manifest "
                    + "declares \(prefill.maxBatchSize)"
            )
        }
        guard manifest.checksums.prefillDispatchesBin != nil else {
            throw CAMTextConstructionError.invalid("CAM route declares metal prefill but manifest lacks prefill checksum")
        }
        guard FileManager.default.fileExists(atPath: "\(packagePath)/prefill_dispatches.bin") else {
            throw CAMTextConstructionError.invalid("CAM route declares metal prefill but package is missing prefill_dispatches.bin")
        }
    }

    private struct TextSetup {
        let tokenizer: SmeltCAMPackageDescriptor.GraphNode
        let promptAdapter: CAMTextExecutionPlan.PromptAdapter?
    }

    private static func makeExecutionPlan(
        capabilities: SmeltCAMPackageCapabilities,
        decision: SmeltCAMPackageCapabilities.Decision,
        hasDeclaredPrefill: Bool
    ) throws -> CAMTextExecutionPlan {
        guard let flow = capabilities.flows.first(where: { $0.flowID == decision.flowID }) else {
            throw CAMTextConstructionError.missingFlow(decision.flowID)
        }
        let nodesByID = Dictionary(uniqueKeysWithValues: capabilities.graphNodes.map {
            ($0.nodeID, $0)
        })
        let setupNodeIDs = try flow.phases
            .filter { $0.phaseType == "setup" }
            .flatMap { phase in try nodeIDs(from: phase, role: "setup") }
        let stepNodeIDs = try flow.phases
            .filter { $0.phaseType == "step" }
            .flatMap { phase in try nodeIDs(from: phase, role: "step") }
        let setup = try resolveTextSetup(
            setupNodeIDs: setupNodeIDs,
            nodesByID: nodesByID,
            graphEdges: capabilities.graphEdges,
            decision: decision
        )
        guard stepNodeIDs.count == 2 else {
            throw CAMTextConstructionError.unsupported("text decode step must call trunk and sampler")
        }
        guard flow.emittedEndpoints.count == 1, let emit = flow.emittedEndpoints.first else {
            throw CAMTextConstructionError.unsupported("text flow must emit exactly one endpoint")
        }
        guard let detokenizerNodeID = emit.nodeID, emit.portName == "text" else {
            throw CAMTextConstructionError.unsupported("text flow must emit detokenizer.text")
        }
        let tokenizer = setup.tokenizer
        let stepNodes = try stepNodeIDs.map { try requiredNode($0, in: nodesByID) }
        let detokenizer = try requiredNode(detokenizerNodeID, in: nodesByID)
        guard hasAnnotation("tag", "text-tokenizer", on: tokenizer) else {
            throw CAMTextConstructionError.unsupported("setup node must be tagged text-tokenizer")
        }
        guard hasAnnotation("tag", "text-detokenizer", on: detokenizer) else {
            throw CAMTextConstructionError.unsupported("emit node must be tagged text-detokenizer")
        }
        guard tokenizer.implementation == "native" else {
            throw CAMTextConstructionError.unsupported("tokenizer node must be native")
        }
        guard detokenizer.implementation == "native" else {
            throw CAMTextConstructionError.unsupported("detokenizer node must be native")
        }
        guard let trunk = stepNodes.first(where: { $0.implementation == "compiled" }) else {
            throw CAMTextConstructionError.unsupported("decode step must call a compiled trunk")
        }
        guard let sampler = stepNodes.first(where: { hasAnnotation("tag", "sampler", on: $0) }) else {
            throw CAMTextConstructionError.unsupported("decode step must call a sampler")
        }
        guard sampler.implementation == "native" else {
            throw CAMTextConstructionError.unsupported("sampler node must be native")
        }
        guard hasAnnotation("state", containing: "kv-cache", on: trunk) else {
            throw CAMTextConstructionError.unsupported("compiled trunk must declare kv-cache state")
        }
        guard hasAnnotation("feedback", containing: "tokens", on: trunk) else {
            throw CAMTextConstructionError.unsupported("compiled trunk must declare token feedback")
        }
        let promptFormat = annotationValue("prompt-format", on: tokenizer)
        let promptTemplate = try promptTemplateName(for: promptFormat)
        let toolTranscriptCodec = try toolTranscriptCodecName(
            for: annotationValue("tool-format", on: tokenizer),
            legacyPromptFormat: promptFormat
        )
        let thinkingPolicy = try thinkingPolicyValue(annotationValue("thinking-policy", on: tokenizer))
        let assistantPrelude = annotationValue("assistant-prelude", on: tokenizer)
        let promptStateRestoreMode = Self.promptStateRestoreMode(from: trunk)
        try validatePromptRendering(
            promptTemplate: promptTemplate,
            thinkingPolicy: thinkingPolicy,
            assistantPrelude: assistantPrelude
        )

        return CAMTextExecutionPlan(
            exportID: decision.exportID,
            flowID: decision.flowID,
            inputPortNames: decision.selectedInputPorts.map(\.portName).sorted(),
            outputPortNames: decision.selectedOutputPorts.map(\.portName).sorted(),
            setupNodeIDs: setupNodeIDs,
            stepNodeIDs: stepNodeIDs,
            emitEndpoints: flow.emittedEndpoints.map(endpointSignature).sorted(),
            tokenizerNodeID: tokenizer.nodeID,
            trunkNodeID: trunk.nodeID,
            samplerNodeID: sampler.nodeID,
            detokenizerNodeID: detokenizer.nodeID,
            promptTemplate: promptTemplate,
            thinkingPolicy: thinkingPolicy,
            toolTranscriptCodec: toolTranscriptCodec,
            promptStateRestoreMode: promptStateRestoreMode,
            assistantPrelude: assistantPrelude,
            promptAdapter: setup.promptAdapter,
            executionMode: try executionMode(from: flow, hasDeclaredPrefill: hasDeclaredPrefill),
            stopPolicy: try stopPolicy(from: flow.stopConditions)
        )
    }

    private static func resolveTextSetup(
        setupNodeIDs: [String],
        nodesByID: [String: SmeltCAMPackageDescriptor.GraphNode],
        graphEdges: [SmeltCAMPackageDescriptor.GraphEdge],
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws -> TextSetup {
        switch setupNodeIDs.count {
        case 1:
            let tokenizer = try requiredNode(setupNodeIDs[0], in: nodesByID)
            try validateTokenizerOnlySetup(
                tokenizer: tokenizer,
                graphEdges: graphEdges,
                decision: decision
            )
            return TextSetup(tokenizer: tokenizer, promptAdapter: nil)
        case 2:
            let promptBuilder = try requiredNode(setupNodeIDs[0], in: nodesByID)
            let tokenizer = try requiredNode(setupNodeIDs[1], in: nodesByID)
            let adapter = try validatePromptBuilderSetup(
                promptBuilder: promptBuilder,
                tokenizer: tokenizer,
                graphEdges: graphEdges,
                decision: decision
            )
            return TextSetup(tokenizer: tokenizer, promptAdapter: adapter)
        default:
            throw CAMTextConstructionError.unsupported(
                "text setup must call tokenizer or prompt adapter then tokenizer"
            )
        }
    }

    private static func validateTokenizerOnlySetup(
        tokenizer: SmeltCAMPackageDescriptor.GraphNode,
        graphEdges: [SmeltCAMPackageDescriptor.GraphEdge],
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws {
        let inputPorts = decision.selectedInputPorts.map(\.portName).sorted()
        guard inputPorts.count == 1, let inputName = inputPorts.first else {
            throw CAMTextConstructionError.unsupported(
                "tokenizer-only setup requires exactly one selected input"
            )
        }
        guard tokenizer.inputs.count == 1, let tokenizerInput = tokenizer.inputs.first else {
            throw CAMTextConstructionError.unsupported(
                "tokenizer-only setup requires one tokenizer input"
            )
        }
        guard isRequiredTextUTF8(tokenizerInput) else {
            throw CAMTextConstructionError.unsupported(
                "tokenizer input must be required text utf8"
            )
        }
        guard hasModuleInputEdge(
            inputName,
            toNodeID: tokenizer.nodeID,
            toPortName: tokenizerInput.portName,
            in: graphEdges
        ) else {
            throw CAMTextConstructionError.invalid(
                "selected input \(inputName) is not wired to tokenizer.\(tokenizerInput.portName)"
            )
        }
    }

    private static func validatePromptBuilderSetup(
        promptBuilder: SmeltCAMPackageDescriptor.GraphNode,
        tokenizer: SmeltCAMPackageDescriptor.GraphNode,
        graphEdges: [SmeltCAMPackageDescriptor.GraphEdge],
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws -> CAMTextExecutionPlan.PromptAdapter {
        guard promptBuilder.implementation == "adapter" else {
            throw CAMTextConstructionError.unsupported(
                "prompt builder setup node must be an adapter"
            )
        }
        guard let template = annotationValue("template", on: promptBuilder),
              !template.isEmpty
        else {
            throw CAMTextConstructionError.invalid("prompt adapter lacks template")
        }
        try validatePromptAdapterTemplate(template)
        let selectedInputs = decision.selectedInputPorts.map(\.portName).sorted()
        let adapterInputs = promptBuilder.inputs.map(\.portName).sorted()
        guard adapterInputs == selectedInputs else {
            throw CAMTextConstructionError.invalid(
                "prompt adapter inputs do not match selected export inputs"
            )
        }
        for input in promptBuilder.inputs {
            guard isRequiredTextUTF8(input) else {
                throw CAMTextConstructionError.unsupported(
                    "prompt adapter input \(input.portName) must be required text utf8"
                )
            }
            guard hasModuleInputEdge(
                input.portName,
                toNodeID: promptBuilder.nodeID,
                toPortName: input.portName,
                in: graphEdges
            ) else {
                throw CAMTextConstructionError.invalid(
                    "selected input \(input.portName) is not wired to prompt adapter"
                )
            }
        }
        guard promptBuilder.outputs.count == 1, let adapterOutput = promptBuilder.outputs.first else {
            throw CAMTextConstructionError.unsupported(
                "prompt adapter must have exactly one output"
            )
        }
        guard tokenizer.inputs.count == 1, let tokenizerInput = tokenizer.inputs.first else {
            throw CAMTextConstructionError.unsupported(
                "prompt-adapter setup requires one tokenizer input"
            )
        }
        guard adapterOutput.type == tokenizerInput.type else {
            throw CAMTextConstructionError.invalid(
                "prompt adapter output type must match tokenizer input type"
            )
        }
        guard hasNodeToGraphValueEdge(
            nodeID: promptBuilder.nodeID,
            portName: adapterOutput.portName,
            graphValueName: adapterOutput.portName,
            in: graphEdges
        ) else {
            throw CAMTextConstructionError.invalid(
                "prompt adapter output is not wired to its graph value"
            )
        }
        guard hasGraphValueToNodeEdge(
            graphValueName: adapterOutput.portName,
            toNodeID: tokenizer.nodeID,
            toPortName: tokenizerInput.portName,
            in: graphEdges
        ) else {
            throw CAMTextConstructionError.invalid(
                "prompt adapter graph value is not wired to tokenizer"
            )
        }
        return CAMTextExecutionPlan.PromptAdapter(
            nodeID: promptBuilder.nodeID,
            template: template,
            inputPortNames: adapterInputs,
            outputName: adapterOutput.portName
        )
    }

    private static func validatePromptAdapterTemplate(_ template: String) throws {
        switch template {
        case "draft-judge", "raw-review":
            return
        default:
            throw CAMTextConstructionError.unsupported(
                "prompt adapter template \(template)"
            )
        }
    }

    private static func prefillBatchSize(
        from requirements: [SmeltCAMPackageDescriptor.Requirement]
    ) -> Int? {
        requirements
            .lazy
            .filter { $0.key == "prefill" }
            .compactMap {
                SmeltCAMCapabilityRequest.RequirementShape
                    .prefillGPUBatchSize($0.value)
            }
            .first
    }

    private static func prefillAllLogitsBatchSize(
        from requirements: [SmeltCAMPackageDescriptor.Requirement]
    ) -> Int? {
        requirements
            .lazy
            .filter { $0.key == "prefill" }
            .compactMap {
                SmeltCAMCapabilityRequest.RequirementShape
                    .prefillAllLogitsGPUBatchSize($0.value)
            }
            .first
    }

    private static func executionMode(
        from flow: SmeltCAMPackageCapabilities.FlowRecord,
        hasDeclaredPrefill: Bool
    ) throws -> CAMTextExecutionPlan.ExecutionMode {
        let unsupportedPhases = flow.phases.filter {
            $0.phaseType != "setup" && $0.phaseType != "step"
        }
        guard unsupportedPhases.isEmpty else {
            throw CAMTextConstructionError.unsupported(
                "text flow contains unsupported phases: "
                    + unsupportedPhases.map(\.phaseType).sorted().joined(separator: ",")
            )
        }
        let setupCount = flow.phases.filter { $0.phaseType == "setup" }.count
        let stepPhases = flow.phases.filter { $0.phaseType == "step" }
        guard setupCount == 1, stepPhases.count == 1 else {
            throw CAMTextConstructionError.unsupported(
                "text flow must have one setup phase and one step phase"
            )
        }
        guard stepPhases.first?.label == "decode" else {
            throw CAMTextConstructionError.unsupported("text step phase must be labeled decode")
        }
        return hasDeclaredPrefill ? .prefillDecode : .decodeOnly
    }

    private static func nodeIDs(
        from phase: SmeltCAMPackageDescriptor.FlowPhase,
        role: String
    ) throws -> [String] {
        try phase.calls.map { call in
            guard call.callType == "node", let nodeID = call.nodeID else {
                throw CAMTextConstructionError.unsupported(
                    "\(role) phase may only call graph nodes"
                )
            }
            return nodeID
        }
    }

    private static func requiredNode(
        _ nodeID: String,
        in nodesByID: [String: SmeltCAMPackageDescriptor.GraphNode]
    ) throws -> SmeltCAMPackageDescriptor.GraphNode {
        guard let node = nodesByID[nodeID] else {
            throw CAMTextConstructionError.missingNode(nodeID)
        }
        return node
    }

    private static func isRequiredTextUTF8(
        _ port: SmeltCAMPackageDescriptor.Port
    ) -> Bool {
        !port.optional
            && port.type.typeName == "text"
            && port.type.attributes["encoding"] == "utf8"
    }

    private static func hasModuleInputEdge(
        _ inputName: String,
        toNodeID nodeID: String,
        toPortName portName: String,
        in edges: [SmeltCAMPackageDescriptor.GraphEdge]
    ) -> Bool {
        edges.contains { edge in
            edge.from.endpointType == "moduleInput"
                && edge.from.name == inputName
                && edge.to.endpointType == "nodePort"
                && edge.to.nodeID == nodeID
                && edge.to.portName == portName
        }
    }

    private static func hasNodeToGraphValueEdge(
        nodeID: String,
        portName: String,
        graphValueName: String,
        in edges: [SmeltCAMPackageDescriptor.GraphEdge]
    ) -> Bool {
        edges.contains { edge in
            edge.from.endpointType == "nodePort"
                && edge.from.nodeID == nodeID
                && edge.from.portName == portName
                && edge.to.endpointType == "graphValue"
                && edge.to.name == graphValueName
        }
    }

    private static func hasGraphValueToNodeEdge(
        graphValueName: String,
        toNodeID nodeID: String,
        toPortName portName: String,
        in edges: [SmeltCAMPackageDescriptor.GraphEdge]
    ) -> Bool {
        edges.contains { edge in
            edge.from.endpointType == "graphValue"
                && edge.from.name == graphValueName
                && edge.to.endpointType == "nodePort"
                && edge.to.nodeID == nodeID
                && edge.to.portName == portName
        }
    }

    private static func hasAnnotation(
        _ key: String,
        _ value: String,
        on node: SmeltCAMPackageDescriptor.GraphNode
    ) -> Bool {
        annotationValue(key, on: node) == value
    }

    private static func hasAnnotation(
        _ key: String,
        containing value: String,
        on node: SmeltCAMPackageDescriptor.GraphNode
    ) -> Bool {
        annotationValue(key, on: node)?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .contains(value) == true
    }

    private static func annotationValue(
        _ key: String,
        on node: SmeltCAMPackageDescriptor.GraphNode
    ) -> String? {
        node.annotations.first(where: { $0.key == key })?.value
    }

    private static func promptTemplateName(for promptFormat: String?) throws -> String {
        switch promptFormat {
        case "chatml":
            return SmeltPromptTemplateName.chatML
        case "chatml-xml-tools":
            return SmeltPromptTemplateName.chatML
        case "channel-turns":
            return SmeltPromptTemplateName.channelTurns
        case "raw":
            return SmeltPromptTemplateName.raw
        case .some(let value):
            throw CAMTextConstructionError.unsupported("prompt-format \(value)")
        case .none:
            throw CAMTextConstructionError.invalid("tokenizer node lacks prompt-format")
        }
    }

    private static func toolTranscriptCodecName(
        for toolFormat: String?,
        legacyPromptFormat: String?
    ) throws -> String? {
        if let toolFormat {
            guard SmeltToolTranscriptCodecName.isKnown(toolFormat) else {
                throw CAMTextConstructionError.unsupported(
                    "tool-format \(toolFormat)"
                )
            }
            return toolFormat
        }
        return legacyPromptFormat == SmeltPromptTemplateName.chatMLXMLTools
            ? SmeltToolTranscriptCodecName.xmlFunctionParameters
            : nil
    }

    private static func promptStateRestoreMode(
        from trunk: SmeltCAMPackageDescriptor.GraphNode
    ) -> SmeltPromptStateRestoreMode {
        let state = Set(
            (annotationValue("state", on: trunk) ?? "")
                .split(separator: ",")
                .map(String.init)
        )
        return SmeltPromptStateRestoreMode.derive(
            fromPersistentStateNames: state
        )
    }

    private static func validatePromptRendering(
        promptTemplate: String,
        thinkingPolicy: SmeltThinkingPolicy,
        assistantPrelude: String?
    ) throws {
        switch promptTemplate {
        case SmeltPromptTemplateName.chatML,
             SmeltPromptTemplateName.chatMLXMLTools:
            if thinkingPolicy == .disabled, assistantPrelude != "preclosed-think" {
                throw CAMTextConstructionError.unsupported(
                    "disabled ChatML thinking requires assistant-prelude preclosed-think"
                )
            }
        case SmeltPromptTemplateName.channelTurns:
            if thinkingPolicy == .disabled, assistantPrelude != "thought-channel" {
                throw CAMTextConstructionError.unsupported(
                    "disabled channel turns require assistant-prelude thought-channel"
                )
            }
        case SmeltPromptTemplateName.raw:
            if assistantPrelude != nil {
                throw CAMTextConstructionError.unsupported(
                    "raw prompt format does not accept assistant-prelude"
                )
            }
        default:
            throw CAMTextConstructionError.unsupported("prompt format \(promptTemplate)")
        }
    }

    private static func thinkingPolicyValue(_ raw: String?) throws -> SmeltThinkingPolicy {
        switch raw {
        case "disabled":
            return .disabled
        case "enabled":
            return .enabled
        case .some(let value):
            throw CAMTextConstructionError.unsupported("thinking-policy \(value)")
        case .none:
            throw CAMTextConstructionError.invalid("tokenizer node lacks thinking-policy")
        }
    }

    private static func stopPolicy(
        from conditions: [SmeltCAMPackageDescriptor.StopCondition]
    ) throws -> CAMTextExecutionPlan.StopPolicy {
        var eosTokens: [Int32] = []
        var maxSteps: Int?
        var honorsHostCancel = false
        for condition in conditions {
            switch condition.stopType {
            case "eos-token":
                guard let value = condition.value else {
                    throw CAMTextConstructionError.invalid("eos-token stop lacks a value")
                }
                eosTokens.append(Int32(value))
            case "max-steps":
                guard let value = condition.value, value > 0 else {
                    throw CAMTextConstructionError.invalid("max-steps stop must be positive")
                }
                maxSteps = value
            case "host-cancel":
                honorsHostCancel = true
            default:
                throw CAMTextConstructionError.unsupported("stop \(condition.stopType)")
            }
        }
        guard !eosTokens.isEmpty else {
            throw CAMTextConstructionError.invalid("text flow lacks eos-token stop")
        }
        guard let maxSteps else {
            throw CAMTextConstructionError.invalid("text flow lacks max-steps stop")
        }
        return CAMTextExecutionPlan.StopPolicy(
            eosTokens: Array(Set(eosTokens)).sorted(),
            maxSteps: maxSteps,
            honorsHostCancel: honorsHostCancel
        )
    }

    private static func endpointSignature(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef
    ) -> String {
        [
            endpoint.endpointType,
            endpoint.name ?? "",
            endpoint.nodeID ?? "",
            endpoint.portName ?? "",
            endpoint.importAlias ?? "",
            endpoint.exportID ?? "",
        ].joined(separator: ":")
    }
}

private extension SmeltManifest {
    func withCAMTextInference(_ inference: SmeltInferenceManifest) -> SmeltManifest {
        SmeltManifest(
            version: version,
            headlessTrunkABI: headlessTrunkABI,
            blocks: blocks,
            loop: loop,
            modelName: modelName,
            config: config,
            context: context,
            checksums: checksums,
            buildProvenance: buildProvenance,
            device: device,
            weights: weights,
            buffers: buffers,
            pipelines: pipelines,
            slotLayout: slotLayout,
            prefill: prefill,
            inference: inference,
            decode: decode,
            validation: validation,
            optimizationReport: optimizationReport
        )
    }
}
