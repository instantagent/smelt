import Foundation

public struct SmeltCAMPackageDescriptor: Codable, Sendable, Equatable {
    public static let currentDescriptorSchema = "smelt.module.package_descriptor.v2"
    public static let currentDescriptorVersion = 2
    public static let packageFileName = "module.json"

    public let descriptorSchema: String
    public let descriptorVersion: Int
    public let camSchemaVersion: Int
    public let moduleID: String
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let imports: [Import]
    public let exports: [Export]
    public let exportFlowBindings: [ExportFlowBinding]
    public let capabilities: [String]
    public let backendRequirements: [Requirement]
    public let compileRequirements: [Requirement]
    public let sourceReferences: [SourceReference]
    public let blocks: [Block]
    public let graphNodes: [GraphNode]
    public let graphEdges: [GraphEdge]
    public let feedbackEdges: [FeedbackEdge]
    public let flows: [Flow]
    public let tensorBindings: [TensorBinding]
    public let quantization: [QuantizationRule]
    public let sourceQuantization: [SourceQuantizationRule]
    public let artifactRequirements: [ArtifactRequirement]
    public let gateContracts: [GateContract]
    public let interfaceSignature: [String]
    public let graphSignature: [String]

    public init(from ir: SmeltCAMIR) throws {
        let cam = try ir.validated()
        let exports = cam.exports.map(Export.init(export:))
        let graphNodes = cam.graphNodes.map(GraphNode.init(node:))
        let graphEdges = cam.graphEdges.map(GraphEdge.init(edge:))
        let feedbackEdges = cam.feedbackEdges.map(FeedbackEdge.init(edge:))
        let flows = cam.flows.map(Flow.init(flow:))
        let blocks = cam.blocks.map(Block.init(block:))

        descriptorSchema = Self.currentDescriptorSchema
        descriptorVersion = Self.currentDescriptorVersion
        camSchemaVersion = cam.schemaVersion
        moduleID = cam.module.id
        camSemanticSHA256 = try cam.semanticSHA256()
        exportABISHA256 = try cam.exportABISHA256()
        imports = cam.imports.map(Import.init(import:))
        self.exports = exports
        exportFlowBindings = cam.exportBindings.map(ExportFlowBinding.init(binding:))
        capabilities = cam.capabilities
        backendRequirements = cam.backendConstraints.map(Requirement.init(requirement:))
        compileRequirements = cam.compile.map(Requirement.init(requirement:))
        sourceReferences = cam.sources.map(SourceReference.init(source:))
        self.blocks = blocks
        self.graphNodes = graphNodes
        self.graphEdges = graphEdges
        self.feedbackEdges = feedbackEdges
        self.flows = flows
        tensorBindings = cam.tensors.map(TensorBinding.init(tensor:))
        quantization = cam.quantization.map(QuantizationRule.init(rule:))
        sourceQuantization = cam.sourceQuantization.map(SourceQuantizationRule.init(rule:))
        artifactRequirements = cam.artifacts.map(ArtifactRequirement.init(artifact:))
        gateContracts = cam.gates.map(GateContract.init(gate:))
        interfaceSignature = exports.map(\.signature)
        graphSignature = Self.graphSignature(
            blocks: blocks,
            nodes: graphNodes,
            graphEdges: graphEdges,
            feedbackEdges: feedbackEdges,
            flows: flows
        )
    }

    public func canonicalJSONData(prettyPrinted: Bool = false) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    private static func graphSignature(
        blocks: [Block],
        nodes: [GraphNode],
        graphEdges: [GraphEdge],
        feedbackEdges: [FeedbackEdge],
        flows: [Flow]
    ) -> [String] {
        (
            blocks.map { "block:\($0.signature)" }
                + nodes.map { "node:\($0.signature)" }
                + graphEdges.map { "edge:\($0.signature)" }
                + feedbackEdges.map { "feedback:\($0.signature)" }
                + flows.map { "flow:\($0.signature)" }
        ).sorted()
    }
}

public extension SmeltCAMPackageDescriptor {
    struct Import: Codable, Sendable, Equatable {
        public let alias: String
        public let moduleID: String
        public let camSemanticSHA256: String
        public let exportABISHA256: String
        public let exportABI: [Export]
        public let parameters: [String: String]

        fileprivate init(import value: SmeltCAMIR.Import) {
            alias = value.alias
            moduleID = value.moduleID
            camSemanticSHA256 = value.irSHA256
            exportABISHA256 = value.exportABISHA256
            exportABI = value.exportABI.map(Export.init(export:))
            parameters = value.parameters
        }
    }

    struct Export: Codable, Sendable, Equatable {
        public let exportID: String
        public let inputs: [Port]
        public let outputs: [Port]
        public let capabilities: [String]
        public let gates: [String]

        fileprivate var signature: String {
            let input = inputs.map(\.signature).joined(separator: ",")
            let output = outputs.map(\.signature).joined(separator: ",")
            return "\(exportID)(\(input))->(\(output))"
        }

        fileprivate init(export value: SmeltCAMIR.Export) {
            exportID = value.id
            inputs = value.inputs.map(Port.init(port:))
            outputs = value.outputs.map(Port.init(port:))
            capabilities = value.capabilities
            gates = value.gates
        }
    }

    struct ExportFlowBinding: Codable, Sendable, Equatable {
        public let exportID: String
        public let flowID: String

        fileprivate init(binding value: SmeltCAMIR.ExportBinding) {
            exportID = value.export
            flowID = value.flow
        }
    }

    struct Port: Codable, Sendable, Equatable {
        public let portName: String
        public let type: ValueType
        public let optional: Bool

        fileprivate var signature: String {
            "\(portName):\(type.signature)\(optional ? "?" : "")"
        }

        fileprivate init(port value: SmeltCAMIR.Port) {
            portName = value.name
            type = ValueType(type: value.type)
            optional = value.optional
        }
    }

    struct ValueType: Codable, Sendable, Equatable {
        public let typeName: String
        public let attributes: [String: String]

        fileprivate var signature: String {
            guard !attributes.isEmpty else { return typeName }
            let attrs = attributes.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ",")
            return "\(typeName)[\(attrs)]"
        }

        fileprivate init(type value: SmeltCAMIR.TypeRef) {
            typeName = value.name
            attributes = value.attributes
        }

        fileprivate init(typeName: String, attributes: [String: String]) {
            self.typeName = typeName
            self.attributes = attributes
        }
    }

    struct Requirement: Codable, Sendable, Equatable {
        public let key: String
        public let value: String

        fileprivate init(requirement value: SmeltCAMIR.Constraint) {
            key = value.key
            self.value = value.value
        }
    }

    struct SourceReference: Codable, Sendable, Equatable {
        public let sourceID: String
        public let sourceType: String
        public let locator: String
        public let revision: String?
        public let checkpointMap: String?
        public let sha256: String?
        public let byteCount: UInt64?

        fileprivate init(source value: SmeltCAMIR.Source) {
            sourceID = value.id
            sourceType = value.kind
            locator = value.locator
            revision = value.revision
            checkpointMap = value.checkpointMap
            sha256 = value.sha256
            byteCount = value.byteCount
        }
    }

    struct Block: Codable, Sendable, Equatable {
        public let blockID: String
        public let operatorName: String
        public let shape: BlockShape
        public let annotations: [Requirement]

        fileprivate var signature: String {
            "\(blockID):\(operatorName)"
        }

        fileprivate init(block value: SmeltCAMIR.Block) {
            blockID = value.id
            operatorName = value.operatorName.rawValue
            shape = BlockShape(shape: value.shape)
            annotations = value.annotations.map(Requirement.init(requirement:))
        }
    }

    struct BlockShape: Codable, Sendable, Equatable {
        public let derivation: ShapeDerivation?
        public let transformer: TransformerShape?
        public let codecDecoder: CodecDecoderShape?
        public let frontend: FrontendShape?
        public let requirements: [BlockRequirement]

        fileprivate init(shape value: SmeltCAMIR.BlockShape) {
            derivation = value.derivation.map(ShapeDerivation.init(derivation:))
            transformer = value.transformer.map(TransformerShape.init(transformer:))
            codecDecoder = value.codecDecoder.map(CodecDecoderShape.init(codecDecoder:))
            frontend = value.frontend.map(FrontendShape.init(frontend:))
            requirements = value.requirements.map(BlockRequirement.init(requirement:))
        }
    }

    struct ShapeDerivation: Codable, Sendable, Equatable {
        public let sourceID: String
        public let authority: String?

        fileprivate init(derivation value: SmeltCAMIR.ShapeDerivation) {
            sourceID = value.source
            authority = value.authority
        }
    }

    struct BlockRequirement: Codable, Sendable, Equatable {
        public let key: String
        public let value: String?
        public let optional: Bool

        fileprivate init(requirement value: SmeltCAMIR.BlockRequirement) {
            key = value.key
            self.value = value.value
            optional = value.optional
        }
    }

    struct TransformerShape: Codable, Sendable, Equatable {
        public let hiddenSize: Int?
        public let layers: LayerPattern?
        public let delta: DeltaShape?
        public let attention: AttentionShape?
        public let attentionByRole: [RoleAttentionShape]?
        public let ffn: FFNShape?
        public let router: RouterShape?
        public let expert: ExpertShape?
        public let norm: NormShape?
        public let vocab: VocabShape?
        public let perLayerInput: PerLayerInputShape?
        public let projectionBanks: [ProjectionBank]?
        public let denseLayerCount: Int?
        public let shortConvolutions: [ShortConvolutionShape]?
        public let sharedKVLayers: Int?
        public let logitCap: String?

        fileprivate init(transformer value: SmeltCAMIR.TransformerShape) {
            hiddenSize = value.hiddenSize
            layers = value.layers.map(LayerPattern.init(pattern:))
            delta = value.delta.map(DeltaShape.init(delta:))
            attention = value.attention.map(AttentionShape.init(attention:))
            attentionByRole = value.attentionByRole?.map(RoleAttentionShape.init(roleAttention:))
            ffn = value.ffn.map(FFNShape.init(ffn:))
            router = value.router.map(RouterShape.init(router:))
            expert = value.expert.map(ExpertShape.init(expert:))
            norm = value.norm.map(NormShape.init(norm:))
            vocab = value.vocab.map(VocabShape.init(vocab:))
            perLayerInput = value.perLayerInput.map(PerLayerInputShape.init(input:))
            projectionBanks = value.projectionBanks?.map(ProjectionBank.init(bank:))
            denseLayerCount = value.denseLayerCount
            shortConvolutions = value.shortConvolutions?.map(
                ShortConvolutionShape.init(convolution:)
            )
            sharedKVLayers = value.sharedKVLayers
            logitCap = value.logitCap
        }
    }

    struct ProjectionBank: Codable, Sendable, Equatable {
        public let id: String
        public let source: String
        public let outputs: [String]
        public let activationView: String?

        fileprivate init(bank value: SmeltCAMIR.ProjectionBank) {
            id = value.id
            source = value.source.rawValue
            outputs = value.outputs.map(\.rawValue)
            activationView = value.activationView?.rawValue
        }
    }

    struct LayerPattern: Codable, Sendable, Equatable {
        public let count: Int?
        public let roles: [String]
        public let repeatCount: Int?

        fileprivate init(pattern value: SmeltCAMIR.LayerPattern) {
            count = value.count
            roles = value.roles.map(\.rawValue)
            repeatCount = value.repeatCount
        }
    }

    struct DeltaShape: Codable, Sendable, Equatable {
        public let heads: Int
        public let headDim: Int
        public let convKernel: Int?
        public let projections: [String: Int]

        fileprivate init(delta value: SmeltCAMIR.DeltaShape) {
            heads = value.heads
            headDim = value.headDim
            convKernel = value.convKernel
            projections = value.projections
        }
    }

    struct AttentionShape: Codable, Sendable, Equatable {
        public let qHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rope: RopeShape?
        public let relativePosition: RelativePositionShape?
        public let scaling: String?
        public let qkNormType: String?
        public let qkNormMode: String?
        public let vNormType: String?
        public let window: Int?

        fileprivate init(attention value: SmeltCAMIR.AttentionShape) {
            qHeads = value.qHeads
            kvHeads = value.kvHeads
            headDim = value.headDim
            rope = value.rope.map(RopeShape.init(rope:))
            relativePosition = value.relativePosition.map(
                RelativePositionShape.init(relativePosition:)
            )
            scaling = value.scaling?.rawValue
            qkNormType = value.qkNorm?.rawValue
            qkNormMode = value.qkNormMode?.rawValue
            vNormType = value.vNorm?.rawValue
            window = value.window
        }
    }

    struct RelativePositionShape: Codable, Sendable, Equatable {
        public let projectionDim: Int
        public let extent: Int
        public let contentConditioned: Bool
        public let logScalingFloor: Int?
        public let logScalingAlpha: String?

        fileprivate init(relativePosition value: SmeltCAMIR.RelativePositionShape) {
            projectionDim = value.projectionDim
            extent = value.extent
            contentConditioned = value.contentConditioned
            logScalingFloor = value.logScalingFloor
            logScalingAlpha = value.logScalingAlpha
        }
    }

    struct RoleAttentionShape: Codable, Sendable, Equatable {
        public let role: String
        public let attention: AttentionShape

        fileprivate init(roleAttention value: SmeltCAMIR.RoleAttentionShape) {
            role = value.role.rawValue
            attention = AttentionShape(attention: value.attention)
        }
    }

    struct RopeShape: Codable, Sendable, Equatable {
        public let ropeType: String
        public let theta: Int?

        fileprivate init(rope value: SmeltCAMIR.RopeShape) {
            ropeType = value.kind.rawValue
            theta = value.theta
        }
    }

    struct FFNShape: Codable, Sendable, Equatable {
        public let dim: Int
        public let activation: String

        fileprivate init(ffn value: SmeltCAMIR.FFNShape) {
            dim = value.dim
            activation = value.activation.rawValue
        }
    }

    struct RouterShape: Codable, Sendable, Equatable {
        public let topK: Int
        public let experts: Int
        public let sharedExperts: Int?
        public let activation: String?
        public let normalization: String?
        public let scoreCorrectionBias: Bool?
        public let routeScale: String?
        public let globalScale: Bool?
        public let sharedExpertSink: Bool?

        fileprivate init(router value: SmeltCAMIR.RouterShape) {
            topK = value.topK
            experts = value.experts
            sharedExperts = value.sharedExperts
            activation = value.activation?.rawValue
            normalization = value.normalization?.rawValue
            scoreCorrectionBias = value.scoreCorrectionBias
            routeScale = value.routeScale
            globalScale = value.globalScale
            sharedExpertSink = value.sharedExpertSink
        }
    }

    struct ShortConvolutionShape: Codable, Sendable, Equatable {
        public let site: String
        public let kernelSize: Int
        public let residual: String

        fileprivate init(convolution value: SmeltCAMIR.ShortConvolutionShape) {
            site = value.site.rawValue
            kernelSize = value.kernelSize
            residual = value.residual.rawValue
        }
    }

    struct ExpertShape: Codable, Sendable, Equatable {
        public let ffn: FFNShape

        fileprivate init(expert value: SmeltCAMIR.ExpertShape) {
            ffn = FFNShape(ffn: value.ffn)
        }
    }

    struct NormShape: Codable, Sendable, Equatable {
        public let normType: String
        public let eps: String?
        public let mode: String?

        fileprivate init(norm value: SmeltCAMIR.NormShape) {
            normType = value.kind.rawValue
            eps = value.eps
            mode = value.mode?.rawValue
        }
    }

    struct VocabShape: Codable, Sendable, Equatable {
        public let size: Int
        public let tiedHead: Bool

        fileprivate init(vocab value: SmeltCAMIR.VocabShape) {
            size = value.size
            tiedHead = value.tiedHead
        }
    }

    struct PerLayerInputShape: Codable, Sendable, Equatable {
        public let hiddenSize: Int
        public let vocabSize: Int

        fileprivate init(input value: SmeltCAMIR.PerLayerInputShape) {
            hiddenSize = value.hiddenSize
            vocabSize = value.vocabSize
        }
    }

    struct CodecDecoderShape: Codable, Sendable, Equatable {
        public let codebooks: Int?
        public let streaming: Bool

        fileprivate init(codecDecoder value: SmeltCAMIR.CodecDecoderShape) {
            codebooks = value.codebooks
            streaming = value.streaming
        }
    }

    struct FrontendShape: Codable, Sendable, Equatable {
        public let speakerConditioning: Bool

        fileprivate init(frontend value: SmeltCAMIR.FrontendShape) {
            speakerConditioning = value.speakerConditioning
        }
    }

    struct GraphNode: Codable, Sendable, Equatable {
        public let nodeID: String
        public let implementation: String
        public let blockID: String?
        public let sourceID: String?
        public let imported: ImportedExportRef?
        public let inputs: [Port]
        public let outputs: [Port]
        public let annotations: [Requirement]

        fileprivate var signature: String {
            let base = "\(nodeID):\(implementation):\(blockID ?? "none"):\(sourceID ?? "none")"
            guard implementation == "imported" || imported != nil else { return base }
            let importedSignature = imported.map { "\($0.alias).\($0.exportID)" } ?? "none"
            let input = inputs.sorted { $0.portName < $1.portName }
                .map(\.signature)
                .joined(separator: ",")
            let output = outputs.sorted { $0.portName < $1.portName }
                .map(\.signature)
                .joined(separator: ",")
            return "\(base):\(importedSignature):in(\(input)):out(\(output))"
        }

        fileprivate init(node value: SmeltCAMIR.GraphNode) {
            nodeID = value.id
            implementation = value.implementation.rawValue
            blockID = value.block
            sourceID = value.source
            imported = value.imported.map(ImportedExportRef.init(imported:))
            inputs = value.inputs.map(Port.init(port:))
            outputs = value.outputs.map(Port.init(port:))
            annotations = value.annotations.map(Requirement.init(requirement:))
        }
    }

    struct ImportedExportRef: Codable, Sendable, Equatable {
        public let alias: String
        public let exportID: String

        fileprivate init(imported value: SmeltCAMIR.ImportedExportRef) {
            alias = value.alias
            exportID = value.export
        }
    }

    struct EndpointRef: Codable, Sendable, Equatable {
        public let endpointType: String
        public let name: String?
        public let nodeID: String?
        public let portName: String?
        public let importAlias: String?
        public let exportID: String?

        fileprivate var signature: String {
            [
                endpointType,
                name ?? "",
                nodeID ?? "",
                portName ?? "",
                importAlias ?? "",
                exportID ?? "",
            ].joined(separator: ":")
        }

        fileprivate init(endpoint value: SmeltCAMIR.EndpointRef) {
            endpointType = value.kind.rawValue
            name = value.name
            nodeID = value.node
            portName = value.port
            importAlias = value.importAlias
            exportID = value.export
        }
    }

    struct GraphEdge: Codable, Sendable, Equatable {
        public let from: EndpointRef
        public let to: EndpointRef
        public let valueType: ValueType?

        fileprivate var signature: String {
            "\(from.signature)->\(to.signature):\(valueType?.signature ?? "none")"
        }

        fileprivate init(edge value: SmeltCAMIR.GraphEdge) {
            from = EndpointRef(endpoint: value.from)
            to = EndpointRef(endpoint: value.to)
            valueType = value.type.map(ValueType.init(type:))
        }
    }

    struct FeedbackEdge: Codable, Sendable, Equatable {
        public let from: EndpointRef
        public let to: EndpointRef

        fileprivate var signature: String {
            "\(from.signature)->\(to.signature)"
        }

        fileprivate init(edge value: SmeltCAMIR.FeedbackEdge) {
            from = EndpointRef(endpoint: value.from)
            to = EndpointRef(endpoint: value.to)
        }
    }

    struct Flow: Codable, Sendable, Equatable {
        public let flowID: String
        public let phases: [FlowPhase]
        public let emit: [EndpointRef]
        public let stop: [StopCondition]

        fileprivate var signature: String {
            let phases = phases.map(\.signature).joined(separator: "|")
            let emit = emit.map(\.signature).joined(separator: ",")
            let stop = stop.map(\.signature).joined(separator: ",")
            return "\(flowID):\(phases):\(emit):\(stop)"
        }

        fileprivate init(flow value: SmeltCAMIR.Flow) {
            flowID = value.id
            phases = value.phases.map(FlowPhase.init(phase:))
            emit = value.emit.map(EndpointRef.init(endpoint:))
            stop = value.stop.map(StopCondition.init(stop:))
        }
    }

    struct FlowPhase: Codable, Sendable, Equatable {
        public let phaseType: String
        public let label: String?
        public let calls: [FlowCall]

        fileprivate var signature: String {
            "\(phaseType):\(label ?? "none"):\(calls.map(\.signature).joined(separator: ","))"
        }

        fileprivate init(phase value: SmeltCAMIR.FlowPhase) {
            phaseType = value.role.rawValue
            label = value.label
            calls = value.calls.map(FlowCall.init(call:))
        }
    }

    struct FlowCall: Codable, Sendable, Equatable {
        public let callType: String
        public let nodeID: String?
        public let imported: ImportedExportRef?
        public let entrypoint: String?

        fileprivate var signature: String {
            let target = nodeID
                ?? imported.map { "\($0.alias).\($0.exportID)" }
                ?? "none"
            return "\(callType):\(target):\(entrypoint ?? "none")"
        }

        fileprivate init(call value: SmeltCAMIR.FlowCall) {
            callType = value.kind.rawValue
            nodeID = value.node
            imported = value.imported.map(ImportedExportRef.init(imported:))
            entrypoint = value.entrypoint
        }
    }

    struct StopCondition: Codable, Sendable, Equatable {
        public let stopType: String
        public let value: Int?

        fileprivate var signature: String {
            "\(stopType):\(value.map(String.init) ?? "none")"
        }

        fileprivate init(stop value: SmeltCAMIR.StopCondition) {
            stopType = value.kind.rawValue
            self.value = value.value
        }
    }

    struct TensorPatternRef: Codable, Sendable, Equatable {
        public let sourceID: String?
        public let pattern: String

        fileprivate init(pattern value: SmeltCAMIR.TensorSelector) {
            sourceID = value.source
            pattern = value.pattern
        }
    }

    struct TensorTarget: Codable, Sendable, Equatable {
        public let blockID: String
        public let pattern: String

        fileprivate init(target value: SmeltCAMIR.TensorTarget) {
            blockID = value.block
            pattern = value.selector
        }
    }

    struct TensorBinding: Codable, Sendable, Equatable {
        public let sourceID: String
        public let tensorPattern: TensorPatternRef
        public let target: TensorTarget
        public let ownerBlockID: String

        fileprivate init(tensor value: SmeltCAMIR.TensorMap) {
            sourceID = value.source
            tensorPattern = TensorPatternRef(pattern: value.selector)
            target = TensorTarget(target: value.target)
            ownerBlockID = value.owner
        }
    }

    struct QuantStorage: Codable, Sendable, Equatable {
        public let storageFormat: String
        public let groupSize: Int?
        public let computeDType: String?

        fileprivate init(storage value: SmeltCAMIR.QuantStorage) {
            storageFormat = value.format.rawValue
            groupSize = value.groupSize
            computeDType = value.computeDType
        }
    }

    struct QuantizationRule: Codable, Sendable, Equatable {
        public let tensorPattern: TensorPatternRef
        public let action: String
        public let storage: QuantStorage?
        public let sourceID: String?
        public let priority: Int?
        public let calibration: QuantCalibration?
        public let resolution: String

        fileprivate init(rule value: SmeltCAMIR.QuantRule) {
            tensorPattern = TensorPatternRef(pattern: value.selector)
            action = value.action.rawValue
            storage = value.storage.map(QuantStorage.init(storage:))
            sourceID = value.source
            priority = value.priority
            calibration = value.calibration.map(QuantCalibration.init(calibration:))
            resolution = value.resolution.rawValue
        }
    }

    struct QuantCalibrationCorpus: Codable, Sendable, Equatable {
        public let sourceID: String
        public let path: String?
        public let maxTokens: Int?

        fileprivate init(corpus value: SmeltCAMIR.QuantCalibrationCorpus) {
            sourceID = value.source
            path = value.path
            maxTokens = value.maxTokens
        }
    }

    struct QuantCalibration: Codable, Sendable, Equatable {
        public let method: String
        public let corpus: QuantCalibrationCorpus
        public let captures: [String]
        public let layersPerPass: Int?
        public let requirements: [Comparison]

        fileprivate init(calibration value: SmeltCAMIR.QuantCalibration) {
            method = value.method.rawValue
            corpus = QuantCalibrationCorpus(corpus: value.corpus)
            captures = value.captures
            layersPerPass = value.layersPerPass
            requirements = value.requirements.map(Comparison.init(comparison:))
        }
    }

    struct SourceQuantizationRule: Codable, Sendable, Equatable {
        public let sourceID: String
        public let sourceDTypes: [String]
        public let action: String
        public let storage: QuantStorage?
        public let targetDType: String?
        public let evidence: [EvidenceRequirement]

        fileprivate init(rule value: SmeltCAMIR.SourceQuantizationRule) {
            sourceID = value.source
            sourceDTypes = value.sourceDTypes
            action = value.action.rawValue
            storage = value.storage.map(QuantStorage.init(storage:))
            targetDType = value.targetDType
            evidence = value.evidence.map(EvidenceRequirement.init(evidence:))
        }
    }

    struct ArtifactRequirement: Codable, Sendable, Equatable {
        public let artifactID: String
        public let role: String
        public let required: Bool

        fileprivate init(artifact value: SmeltCAMIR.ArtifactRole) {
            artifactID = value.id
            role = value.role
            required = value.required
        }
    }

    struct GateContract: Codable, Sendable, Equatable {
        public let gateID: String
        public let from: GateEvent?
        public let to: GateEvent?
        public let requirements: [Comparison]
        public let measurements: [GateMeasurement]
        public let evidence: [EvidenceRequirement]

        fileprivate init(gate value: SmeltCAMIR.Gate) {
            gateID = value.id
            from = value.from.map(GateEvent.init(event:))
            to = value.to.map(GateEvent.init(event:))
            requirements = value.requirements.map(Comparison.init(comparison:))
            measurements = value.measurements.map(GateMeasurement.init(measurement:))
            evidence = value.evidence.map(EvidenceRequirement.init(evidence:))
        }
    }

    struct GateMeasurement: Codable, Sendable, Equatable {
        public let subject: String
        public let processMode: String
        public let cacheState: String
        public let occurrence: String

        fileprivate init(measurement value: SmeltCAMIR.GateMeasurement) {
            subject = value.subject
            processMode = value.processMode.rawValue
            cacheState = value.cacheState.rawValue
            occurrence = value.occurrence.rawValue
        }
    }

    struct GateEvent: Codable, Sendable, Equatable {
        public let eventType: String
        public let flowID: String?
        public let exportID: String?
        public let endpoint: EndpointRef?
        public let signal: String?
        public let predicates: [Comparison]

        fileprivate init(event value: SmeltCAMIR.GateEvent) {
            eventType = value.kind.rawValue
            flowID = value.flow
            exportID = value.export
            endpoint = value.endpoint.map(EndpointRef.init(endpoint:))
            signal = value.signal
            predicates = value.predicates.map(Comparison.init(comparison:))
        }
    }

    struct Comparison: Codable, Sendable, Equatable {
        public let subject: String
        public let relation: String
        public let value: String
        public let unit: String?

        fileprivate init(comparison value: SmeltCAMIR.Comparison) {
            subject = value.subject
            relation = value.relation.rawValue
            self.value = value.value
            unit = value.unit
        }
    }

    struct EvidenceRequirement: Codable, Sendable, Equatable {
        public let evidenceType: String
        public let tensor: String?
        public let sourceID: String?
        public let sourceDTypes: [String]
        public let storage: QuantStorage?

        fileprivate init(evidence value: SmeltCAMIR.EvidenceRequirement) {
            evidenceType = value.kind.rawValue
            tensor = value.tensor
            sourceID = value.source
            sourceDTypes = value.sourceDTypes
            storage = value.storage.map(QuantStorage.init(storage:))
        }
    }
}

public extension SmeltCAMPackageDescriptor {
    func validateDecoded() throws {
        guard descriptorSchema == Self.currentDescriptorSchema else {
            throw SmeltCAMIRError.malformed(
                "unsupported package descriptor schema '\(descriptorSchema)'"
            )
        }
        guard descriptorVersion == Self.currentDescriptorVersion else {
            throw SmeltCAMIRError.malformed(
                "unsupported package descriptor version \(descriptorVersion)"
            )
        }
        guard camSchemaVersion == SmeltCAMIR.currentSchemaVersion else {
            throw SmeltCAMIRError.malformed(
                "unsupported CAM schema version \(camSchemaVersion)"
            )
        }

        try requireDecodedUnique([moduleID], label: "module id")
        try requireDecodedUnique(imports.map(\.alias), label: "import alias")
        try requireDecodedUnique(exports.map(\.exportID), label: "export id")
        try requireDecodedUnique(exportFlowBindings.map(\.exportID), label: "export binding")
        try requireDecodedUnique(sourceReferences.map(\.sourceID), label: "source id")
        try requireDecodedUnique(blocks.map(\.blockID), label: "block id")
        try requireDecodedUnique(graphNodes.map(\.nodeID), label: "graph node id")
        try requireDecodedUnique(flows.map(\.flowID), label: "flow id")
        try requireDecodedUnique(artifactRequirements.map(\.artifactID), label: "artifact id")
        try requireDecodedUnique(gateContracts.map(\.gateID), label: "gate id")

        let exportIDs = Set(exports.map(\.exportID))
        let flowIDs = Set(flows.map(\.flowID))
        let sourceIDs = Set(sourceReferences.map(\.sourceID))
        let blockIDs = Set(blocks.map(\.blockID))
        let nodeIDs = Set(graphNodes.map(\.nodeID))
        let gateIDs = Set(gateContracts.map(\.gateID))
        let importsByAlias = Dictionary(uniqueKeysWithValues: imports.map { ($0.alias, $0) })
        let nodesByID = Dictionary(uniqueKeysWithValues: graphNodes.map { ($0.nodeID, $0) })
        let exportInputPortsByName = Dictionary(grouping: exports.flatMap { $0.inputs }, by: \.portName)
        let exportOutputPortsByName = Dictionary(grouping: exports.flatMap { $0.outputs }, by: \.portName)

        for importedModule in imports {
            try requireDecodedUnique(
                importedModule.exportABI.map(\.exportID),
                label: "export ABI for import \(importedModule.alias)"
            )
            for abi in importedModule.exportABI {
                try validateDecodedPorts(abi.inputs, label: "input port for imported export \(abi.exportID)")
                try validateDecodedPorts(abi.outputs, label: "output port for imported export \(abi.exportID)")
            }
            let computed = try decodedExportABISHA256(for: importedModule.exportABI)
            guard computed == importedModule.exportABISHA256 else {
                throw SmeltCAMIRError.malformed(
                    "import '\(importedModule.alias)' export ABI hash mismatch"
                )
            }
        }

        for export in exports {
            try validateDecodedPorts(export.inputs, label: "input port for export \(export.exportID)")
            try validateDecodedPorts(export.outputs, label: "output port for export \(export.exportID)")
            try requireDecodedUnique(export.capabilities, label: "capability for export \(export.exportID)")
            try requireDecodedUnique(export.gates, label: "gate for export \(export.exportID)")
            for gate in export.gates where !gateIDs.contains(gate) {
                throw SmeltCAMIRError.malformed(
                    "export '\(export.exportID)' references unknown gate '\(gate)'"
                )
            }
        }

        for binding in exportFlowBindings {
            guard exportIDs.contains(binding.exportID) else {
                throw SmeltCAMIRError.malformed(
                    "export binding references unknown export '\(binding.exportID)'"
                )
            }
            guard flowIDs.contains(binding.flowID) else {
                throw SmeltCAMIRError.malformed(
                    "export binding references unknown flow '\(binding.flowID)'"
                )
            }
        }
        for export in exports where !exportFlowBindings.contains(where: { $0.exportID == export.exportID }) {
            throw SmeltCAMIRError.malformed("export '\(export.exportID)' has no flow binding")
        }

        for block in blocks {
            if let sourceID = block.shape.derivation?.sourceID, !sourceIDs.contains(sourceID) {
                throw SmeltCAMIRError.malformed(
                    "block '\(block.blockID)' derives shape from unknown source '\(sourceID)'"
                )
            }
        }

        for node in graphNodes {
            try validateDecodedPorts(node.inputs, label: "input port for node \(node.nodeID)")
            try validateDecodedPorts(node.outputs, label: "output port for node \(node.nodeID)")
            try validateDecodedGraphNode(
                node,
                sourceIDs: sourceIDs,
                blockIDs: blockIDs,
                importsByAlias: importsByAlias
            )
        }

        var graphValueProducers: [String: Int] = [:]
        var graphValueTypes: [String: ValueType] = [:]
        for edge in graphEdges {
            let fromType = try validateDecodedEndpoint(
                edge.from,
                use: .source,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            let toType = try validateDecodedEndpoint(
                edge.to,
                use: .sink,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            if edge.to.endpointType == "graphValue", let name = edge.to.name {
                graphValueProducers[name, default: 0] += 1
                graphValueTypes[name] = try resolvedDecodedGraphValueType(
                    edge: edge,
                    fromType: fromType,
                    toType: toType
                )
            }
        }
        for (value, count) in graphValueProducers where count > 1 {
            throw SmeltCAMIRError.malformed(
                "graph value '\(value)' has multiple non-feedback producers"
            )
        }
        for edge in graphEdges {
            let fromType = try decodedGraphEndpointType(
                edge.from,
                use: .source,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName,
                graphValueTypes: graphValueTypes
            )
            let toType = try decodedGraphEndpointType(
                edge.to,
                use: .sink,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName,
                graphValueTypes: graphValueTypes
            )
            try validateDecodedGraphEdgeValueType(edge, fromType: fromType, toType: toType)
        }
        for edge in feedbackEdges {
            try validateDecodedEndpoint(
                edge.from,
                use: .source,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            try validateDecodedEndpoint(
                edge.to,
                use: .sink,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
        }

        for flow in flows {
            for phase in flow.phases {
                for call in phase.calls {
                    try validateDecodedFlowCall(
                        call,
                        nodeIDs: nodeIDs,
                        importsByAlias: importsByAlias
                    )
                }
            }
            for endpoint in flow.emit {
                try validateDecodedEndpoint(
                    endpoint,
                    use: .emit,
                    nodesByID: nodesByID,
                    importsByAlias: importsByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
        }

        try validateDecodedGateContracts(
            flowIDs: flowIDs,
            exportIDs: exportIDs,
            nodesByID: nodesByID,
            importsByAlias: importsByAlias,
            exportInputPortsByName: exportInputPortsByName,
            exportOutputPortsByName: exportOutputPortsByName
        )
    }

    private enum DecodedEndpointUse: Equatable {
        case source
        case sink
        case emit
    }

    private func validateDecodedGraphNode(
        _ node: GraphNode,
        sourceIDs: Set<String>,
        blockIDs: Set<String>,
        importsByAlias: [String: Import]
    ) throws {
        let hasBlock = node.blockID != nil
        let hasSource = node.sourceID != nil
        let hasImport = node.imported != nil
        switch node.implementation {
        case "compiled":
            guard hasBlock != hasSource, !hasImport else {
                throw SmeltCAMIRError.malformed(
                    "compiled graph node '\(node.nodeID)' must reference exactly one block or source"
                )
            }
            if let blockID = node.blockID, !blockIDs.contains(blockID) {
                throw SmeltCAMIRError.malformed(
                    "compiled graph node '\(node.nodeID)' must reference a declared block"
                )
            }
            if let sourceID = node.sourceID, !sourceIDs.contains(sourceID) {
                throw SmeltCAMIRError.malformed(
                    "compiled graph node '\(node.nodeID)' must reference a declared source"
                )
            }
        case "imported":
            guard !hasBlock, !hasSource,
                  let imported = node.imported,
                  let importedExport = decodedImportedExport(
                    alias: imported.alias,
                    exportID: imported.exportID,
                    importsByAlias: importsByAlias
                  ) else {
                throw SmeltCAMIRError.malformed(
                    "imported graph node '\(node.nodeID)' must reference an imported export"
                )
            }
            try validateDecodedImportedGraphNode(
                node,
                mirrors: importedExport,
                alias: imported.alias,
                exportID: imported.exportID
            )
        case "native", "adapter":
            guard !hasBlock, !hasSource, !hasImport else {
                throw SmeltCAMIRError.malformed(
                    "\(node.implementation) graph node '\(node.nodeID)' must not reference block, source, or import"
                )
            }
        default:
            throw SmeltCAMIRError.malformed(
                "graph node '\(node.nodeID)' has unknown implementation '\(node.implementation)'"
            )
        }
    }

    private func validateDecodedFlowCall(
        _ call: FlowCall,
        nodeIDs: Set<String>,
        importsByAlias: [String: Import]
    ) throws {
        switch call.callType {
        case "node":
            guard let nodeID = call.nodeID, nodeIDs.contains(nodeID) else {
                throw SmeltCAMIRError.malformed(
                    "flow call references unknown graph node '\(call.nodeID ?? "")'"
                )
            }
        case "imported":
            guard let imported = call.imported,
                  decodedImportedExport(
                    alias: imported.alias,
                    exportID: imported.exportID,
                    importsByAlias: importsByAlias
                  ) != nil else {
                throw SmeltCAMIRError.malformed("flow call references unknown imported export")
            }
        default:
            throw SmeltCAMIRError.malformed("flow call has unknown type '\(call.callType)'")
        }
    }

    private func validateDecodedImportedGraphNode(
        _ node: GraphNode,
        mirrors export: Export,
        alias: String,
        exportID: String
    ) throws {
        try validateDecodedImportedGraphNodePorts(
            node.inputs,
            expected: export.inputs,
            nodeID: node.nodeID,
            abi: "\(alias).\(exportID)",
            direction: "input"
        )
        try validateDecodedImportedGraphNodePorts(
            node.outputs,
            expected: export.outputs,
            nodeID: node.nodeID,
            abi: "\(alias).\(exportID)",
            direction: "output"
        )
    }

    private func validateDecodedImportedGraphNodePorts(
        _ projected: [Port],
        expected: [Port],
        nodeID: String,
        abi: String,
        direction: String
    ) throws {
        let projectedByName = Dictionary(uniqueKeysWithValues: projected.map { ($0.portName, $0) })
        let expectedByName = Dictionary(uniqueKeysWithValues: expected.map { ($0.portName, $0) })
        let projectedNames = Set(projectedByName.keys)
        let expectedNames = Set(expectedByName.keys)

        if let missing = expectedNames.subtracting(projectedNames).sorted().first {
            throw SmeltCAMIRError.malformed(
                "imported graph node '\(nodeID)' is missing \(direction) port "
                    + "'\(missing)' from \(abi) ABI"
            )
        }
        if let extra = projectedNames.subtracting(expectedNames).sorted().first {
            throw SmeltCAMIRError.malformed(
                "imported graph node '\(nodeID)' declares extra \(direction) port "
                    + "'\(extra)' not in \(abi) ABI"
            )
        }

        for name in expectedNames.sorted() {
            guard let projectedPort = projectedByName[name],
                  let expectedPort = expectedByName[name] else {
                continue
            }
            if projectedPort.optional != expectedPort.optional {
                throw SmeltCAMIRError.malformed(
                    "imported graph node '\(nodeID)' \(direction) port '\(name)' "
                        + "does not match \(abi) ABI: expected optional="
                        + "\(expectedPort.optional), got optional=\(projectedPort.optional)"
                )
            }
            if projectedPort.type != expectedPort.type {
                throw SmeltCAMIRError.malformed(
                    "imported graph node '\(nodeID)' \(direction) port '\(name)' "
                        + "does not match \(abi) ABI: expected "
                        + "\(expectedPort.type.signature), got \(projectedPort.type.signature)"
                )
            }
        }
    }

    private func validateDecodedGateContracts(
        flowIDs: Set<String>,
        exportIDs: Set<String>,
        nodesByID: [String: GraphNode],
        importsByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws {
        for gate in gateContracts {
            if gate.requirements.contains(where: { $0.subject == "elapsed" }) {
                guard gate.from != nil, gate.to != nil else {
                    throw SmeltCAMIRError.malformed(
                        "timing gate '\(gate.gateID)' requires from and to events"
                    )
                }
            }
            try requireDecodedUnique(
                gate.measurements.map(\.subject),
                label: "measurement subject for gate \(gate.gateID)"
            )
            let requirementSubjects = Set(gate.requirements.map(\.subject))
            for measurement in gate.measurements {
                guard !measurement.subject.isEmpty,
                      ["cold", "warm"].contains(measurement.processMode),
                      [
                        "cold",
                        "host-global-cold-candidate",
                        "unknown",
                        "warm",
                      ].contains(measurement.cacheState),
                      ["all", "first", "median", "p95"].contains(measurement.occurrence)
                else {
                    throw SmeltCAMIRError.malformed(
                        "gate '\(gate.gateID)' has malformed measurement"
                    )
                }
                guard requirementSubjects.contains(measurement.subject) else {
                    throw SmeltCAMIRError.malformed(
                        "gate '\(gate.gateID)' measurement '\(measurement.subject)' has no matching requirement"
                    )
                }
            }
            try gate.from.map {
                try validateDecodedGateEvent(
                    $0,
                    flowIDs: flowIDs,
                    exportIDs: exportIDs,
                    nodesByID: nodesByID,
                    importsByAlias: importsByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
            try gate.to.map {
                try validateDecodedGateEvent(
                    $0,
                    flowIDs: flowIDs,
                    exportIDs: exportIDs,
                    nodesByID: nodesByID,
                    importsByAlias: importsByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
        }
    }

    private func validateDecodedGateEvent(
        _ event: GateEvent,
        flowIDs: Set<String>,
        exportIDs: Set<String>,
        nodesByID: [String: GraphNode],
        importsByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws {
        if let flowID = event.flowID, !flowIDs.contains(flowID) {
            throw SmeltCAMIRError.malformed(
                "gate event references unknown flow '\(flowID)'"
            )
        }
        if let exportID = event.exportID, !exportIDs.contains(exportID) {
            throw SmeltCAMIRError.malformed(
                "gate event references unknown export '\(exportID)'"
            )
        }
        if let endpoint = event.endpoint {
            let use: DecodedEndpointUse = event.eventType == "input" ? .source : .emit
            try validateDecodedEndpoint(
                endpoint,
                use: use,
                nodesByID: nodesByID,
                importsByAlias: importsByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
        }
        switch event.eventType {
        case "flow.accepted", "emit", "input":
            break
        default:
            throw SmeltCAMIRError.malformed(
                "gate event has unknown type '\(event.eventType)'"
            )
        }
    }

    @discardableResult
    private func validateDecodedEndpoint(
        _ endpoint: EndpointRef,
        use: DecodedEndpointUse,
        nodesByID: [String: GraphNode],
        importsByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws -> ValueType? {
        try validateDecodedEndpointShape(endpoint)
        switch endpoint.endpointType {
        case "moduleInput":
            guard use == .source,
                  let name = endpoint.name,
                  exportInputPortsByName[name] != nil else {
                throw SmeltCAMIRError.malformed(
                    "module input endpoint '\(decodedEndpointKey(endpoint))' is not available as a source"
                )
            }
            return try mergedDecodedModulePortType(
                named: name,
                portsByName: exportInputPortsByName,
                endpoint: endpoint,
                role: "input"
            )
        case "moduleOutput":
            guard use != .source,
                  let name = endpoint.name,
                  exportOutputPortsByName[name] != nil else {
                throw SmeltCAMIRError.malformed(
                    "module output endpoint '\(decodedEndpointKey(endpoint))' is not available as a sink"
                )
            }
            return try mergedDecodedModulePortType(
                named: name,
                portsByName: exportOutputPortsByName,
                endpoint: endpoint,
                role: "output"
            )
        case "graphValue":
            return nil
        case "nodePort":
            guard let nodeID = endpoint.nodeID,
                  let portName = endpoint.portName,
                  let node = nodesByID[nodeID] else {
                throw SmeltCAMIRError.malformed(
                    "unknown node endpoint '\(decodedEndpointKey(endpoint))'"
                )
            }
            let ports = use == .sink ? node.inputs : node.outputs
            guard let resolvedPort = ports.first(where: { $0.portName == portName }) else {
                throw SmeltCAMIRError.malformed(
                    "node endpoint '\(decodedEndpointKey(endpoint))' has no "
                        + "\(use == .sink ? "input" : "output") port"
                )
            }
            return resolvedPort.type
        case "importedPort":
            guard let importAlias = endpoint.importAlias,
                  let exportID = endpoint.exportID,
                  let portName = endpoint.portName,
                  let importedExport = decodedImportedExport(
                    alias: importAlias,
                    exportID: exportID,
                    importsByAlias: importsByAlias
                  ) else {
                throw SmeltCAMIRError.malformed(
                    "unknown imported endpoint '\(decodedEndpointKey(endpoint))'"
                )
            }
            let ports = use == .sink ? importedExport.inputs : importedExport.outputs
            guard let resolvedPort = ports.first(where: { $0.portName == portName }) else {
                throw SmeltCAMIRError.malformed(
                    "imported endpoint '\(decodedEndpointKey(endpoint))' has no "
                        + "\(use == .sink ? "input" : "output") port"
                )
            }
            return resolvedPort.type
        default:
            throw SmeltCAMIRError.malformed(
                "endpoint has unknown type '\(endpoint.endpointType)'"
            )
        }
    }

    private func validateDecodedGraphEdgeValueType(
        _ edge: GraphEdge,
        fromType: ValueType?,
        toType: ValueType?
    ) throws {
        if let declared = edge.valueType {
            try validateDecodedDeclaredValueType(
                declared,
                matches: fromType,
                edge: edge,
                endpoint: edge.from,
                role: "source"
            )
            try validateDecodedDeclaredValueType(
                declared,
                matches: toType,
                edge: edge,
                endpoint: edge.to,
                role: "sink"
            )
        }
        if let fromType, let toType, !decodedTypesCanConnect(fromType, toType) {
            throw SmeltCAMIRError.malformed(
                "graph edge \(edge.signature) connects source port "
                    + "\(fromType.signature) to sink port \(toType.signature)"
            )
        }
    }

    private func decodedGraphEndpointType(
        _ endpoint: EndpointRef,
        use: DecodedEndpointUse,
        nodesByID: [String: GraphNode],
        importsByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]],
        graphValueTypes: [String: ValueType]
    ) throws -> ValueType? {
        let staticType = try validateDecodedEndpoint(
            endpoint,
            use: use,
            nodesByID: nodesByID,
            importsByAlias: importsByAlias,
            exportInputPortsByName: exportInputPortsByName,
            exportOutputPortsByName: exportOutputPortsByName
        )
        guard endpoint.endpointType == "graphValue", let name = endpoint.name else {
            return staticType
        }
        return graphValueTypes[name]
    }

    private func resolvedDecodedGraphValueType(
        edge: GraphEdge,
        fromType: ValueType?,
        toType: ValueType?
    ) throws -> ValueType? {
        guard edge.to.endpointType == "graphValue" else { return nil }
        var resolved = fromType
        if let toType {
            resolved = try resolved.map {
                try mergedDecodedCompatibleType($0, toType, context: "graph edge \(edge.signature)")
            } ?? toType
        }
        if let declared = edge.valueType {
            resolved = try resolved.map {
                try mergedDecodedCompatibleType($0, declared, context: "graph edge \(edge.signature)")
            } ?? declared
        }
        return resolved
    }

    private func validateDecodedDeclaredValueType(
        _ declared: ValueType,
        matches resolved: ValueType?,
        edge: GraphEdge,
        endpoint: EndpointRef,
        role: String
    ) throws {
        guard let resolved, decodedDeclaredDescribes(declared, resolved) else {
            if let resolved {
                throw SmeltCAMIRError.malformed(
                    "graph edge \(edge.signature) declares value type \(declared.signature), "
                        + "but \(role) endpoint \(decodedEndpointKey(endpoint)) resolves to "
                        + "\(resolved.signature)"
                )
            }
            return
        }
    }

    private func mergedDecodedModulePortType(
        named name: String,
        portsByName: [String: [Port]],
        endpoint: EndpointRef,
        role: String
    ) throws -> ValueType? {
        guard let ports = portsByName[name], var merged = ports.first?.type else {
            return nil
        }
        for port in ports.dropFirst() {
            do {
                merged = try mergedDecodedCompatibleType(
                    merged,
                    port.type,
                    context: "module \(role) endpoint '\(decodedEndpointKey(endpoint))'"
                )
            } catch {
                throw SmeltCAMIRError.malformed(
                    "module \(role) endpoint '\(decodedEndpointKey(endpoint))' "
                        + "has ambiguous declared types \(merged.signature) and \(port.type.signature)"
                )
            }
        }
        return merged
    }

    private func mergedDecodedCompatibleType(
        _ lhs: ValueType,
        _ rhs: ValueType,
        context: String
    ) throws -> ValueType {
        guard lhs.typeName == rhs.typeName else {
            throw SmeltCAMIRError.malformed(
                "\(context) connects incompatible value types \(lhs.signature) and \(rhs.signature)"
            )
        }
        var attributes = lhs.attributes
        for (key, value) in rhs.attributes {
            if let existing = attributes[key], existing != value {
                throw SmeltCAMIRError.malformed(
                    "\(context) connects incompatible value types \(lhs.signature) and \(rhs.signature)"
                )
            }
            attributes[key] = value
        }
        return ValueType(typeName: lhs.typeName, attributes: attributes)
    }

    private func decodedDeclaredDescribes(_ declared: ValueType, _ resolved: ValueType) -> Bool {
        guard declared.typeName == resolved.typeName else { return false }
        for (key, value) in declared.attributes {
            if let resolvedValue = resolved.attributes[key], resolvedValue != value {
                return false
            }
        }
        return true
    }

    private func decodedTypesCanConnect(_ lhs: ValueType, _ rhs: ValueType) -> Bool {
        guard lhs.typeName == rhs.typeName else { return false }
        for (key, value) in lhs.attributes {
            if let rhsValue = rhs.attributes[key], rhsValue != value {
                return false
            }
        }
        return true
    }

    private func validateDecodedEndpointShape(_ endpoint: EndpointRef) throws {
        switch endpoint.endpointType {
        case "moduleInput", "moduleOutput", "graphValue":
            guard endpoint.name?.isEmpty == false,
                  endpoint.nodeID == nil,
                  endpoint.portName == nil,
                  endpoint.importAlias == nil,
                  endpoint.exportID == nil else {
                throw SmeltCAMIRError.malformed(
                    "malformed endpoint '\(decodedEndpointKey(endpoint))'"
                )
            }
        case "nodePort":
            guard endpoint.nodeID?.isEmpty == false,
                  endpoint.portName?.isEmpty == false,
                  endpoint.name == nil,
                  endpoint.importAlias == nil,
                  endpoint.exportID == nil else {
                throw SmeltCAMIRError.malformed(
                    "malformed endpoint '\(decodedEndpointKey(endpoint))'"
                )
            }
        case "importedPort":
            guard endpoint.importAlias?.isEmpty == false,
                  endpoint.exportID?.isEmpty == false,
                  endpoint.portName?.isEmpty == false,
                  endpoint.name == nil,
                  endpoint.nodeID == nil else {
                throw SmeltCAMIRError.malformed(
                    "malformed endpoint '\(decodedEndpointKey(endpoint))'"
                )
            }
        default:
            throw SmeltCAMIRError.malformed(
                "endpoint has unknown type '\(endpoint.endpointType)'"
            )
        }
    }

    private func decodedImportedExport(
        alias: String,
        exportID: String,
        importsByAlias: [String: Import]
    ) -> Export? {
        importsByAlias[alias]?.exportABI.first {
            $0.exportID == exportID
        }
    }

    private func decodedExportABISHA256(for exports: [Export]) throws -> String {
        try SmeltCAMIR.exportABISHA256(for: exports.map(camExport))
    }

    private func camExport(_ export: Export) -> SmeltCAMIR.Export {
        SmeltCAMIR.Export(
            id: export.exportID,
            inputs: export.inputs.map(camPort),
            outputs: export.outputs.map(camPort),
            capabilities: export.capabilities,
            gates: export.gates
        )
    }

    private func camPort(_ port: Port) -> SmeltCAMIR.Port {
        SmeltCAMIR.Port(
            name: port.portName,
            type: SmeltCAMIR.TypeRef(port.type.typeName, attributes: port.type.attributes),
            optional: port.optional
        )
    }

    private func validateDecodedPorts(_ ports: [Port], label: String) throws {
        try requireDecodedUnique(ports.map(\.portName), label: label)
        for port in ports {
            guard !port.type.typeName.isEmpty else {
                throw SmeltCAMIRError.malformed("\(label) '\(port.portName)' has empty type")
            }
        }
    }

    private func requireDecodedUnique(_ values: [String], label: String) throws {
        var seen = Set<String>()
        for value in values {
            guard !value.isEmpty, seen.insert(value).inserted else {
                throw SmeltCAMIRError.malformed("duplicate or empty \(label) '\(value)'")
            }
        }
    }

    private func decodedEndpointKey(_ endpoint: EndpointRef) -> String {
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
