import CryptoKit
import Foundation

public struct SmeltCAMIR: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 4

    public let schemaVersion: Int
    public let module: Module
    public let imports: [Import]
    public let exports: [Export]
    public let exportBindings: [ExportBinding]
    public let sources: [Source]
    public let blocks: [Block]
    public let graphNodes: [GraphNode]
    public let graphEdges: [GraphEdge]
    public let feedbackEdges: [FeedbackEdge]
    public let flows: [Flow]
    public let capabilities: [String]
    public let backendConstraints: [Constraint]
    public let tensors: [TensorMap]
    public let quantization: [QuantRule]
    public let sourceQuantization: [SourceQuantizationRule]
    public let compile: [Constraint]
    public let artifacts: [ArtifactRole]
    public let gates: [Gate]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        module: Module,
        imports: [Import] = [],
        exports: [Export],
        exportBindings: [ExportBinding],
        sources: [Source] = [],
        blocks: [Block],
        graphNodes: [GraphNode],
        graphEdges: [GraphEdge],
        feedbackEdges: [FeedbackEdge] = [],
        flows: [Flow],
        capabilities: [String] = [],
        backendConstraints: [Constraint] = [],
        tensors: [TensorMap] = [],
        quantization: [QuantRule] = [],
        sourceQuantization: [SourceQuantizationRule] = [],
        compile: [Constraint] = [],
        artifacts: [ArtifactRole] = [],
        gates: [Gate] = []
    ) {
        self.schemaVersion = schemaVersion
        self.module = module
        self.imports = imports
        self.exports = exports
        self.exportBindings = exportBindings
        self.sources = sources
        self.blocks = blocks
        self.graphNodes = graphNodes
        self.graphEdges = graphEdges
        self.feedbackEdges = feedbackEdges
        self.flows = flows
        self.capabilities = capabilities
        self.backendConstraints = backendConstraints
        self.tensors = tensors
        self.quantization = quantization
        self.sourceQuantization = sourceQuantization
        self.compile = compile
        self.artifacts = artifacts
        self.gates = gates
    }

    /// Decode a canonical module IR JSON file (`.module.json`) and validate it.
    /// This is the production authoring-artifact entry point: consumers decode
    /// the checked-in / packaged IR JSON rather than parsing grammar text.
    public static func decodeModule(at url: URL) throws -> SmeltCAMIR {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(SmeltCAMIR.self, from: data).validated()
    }

    public func validated() throws -> SmeltCAMIR {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw SmeltCAMIRError.malformed("unsupported schema version \(schemaVersion)")
        }
        let context = try validationContext()
        try validateGraphAndPolicy(
            exportIDs: context.exportIDs,
            flowIDs: context.flowIDs,
            sourceIDs: context.sourceIDs,
            blockIDs: context.blockIDs,
            nodeIDs: context.nodeIDs,
            importByAlias: context.importByAlias,
            nodeByID: context.nodeByID,
            exportInputPortsByName: context.exportInputPortsByName,
            exportOutputPortsByName: context.exportOutputPortsByName
        )
        return canonicalized()
    }

    private struct ValidationContext {
        let exportIDs: Set<String>
        let flowIDs: Set<String>
        let sourceIDs: Set<String>
        let blockIDs: Set<String>
        let nodeIDs: Set<String>
        let importByAlias: [String: Import]
        let nodeByID: [String: GraphNode]
        let exportInputPortsByName: [String: [Port]]
        let exportOutputPortsByName: [String: [Port]]
    }

    private func validationContext() throws -> ValidationContext {
        try requireUnique([module.id], label: "module id")
        try requireUnique(imports.map(\.alias), label: "import alias")
        try requireUnique(exports.map(\.id), label: "export id")
        try validateSources()
        try requireUnique(blocks.map(\.id), label: "block id")
        try requireUnique(graphNodes.map(\.id), label: "graph node id")
        try requireUnique(flows.map(\.id), label: "flow id")
        try requireUnique(exportBindings.map(\.export), label: "export binding")
        try requireUnique(gates.map(\.id), label: "gate id")

        let exportIDs = Set(exports.map(\.id))
        let flowIDs = Set(flows.map(\.id))
        let gateIDs = Set(gates.map(\.id))
        let sourceIDs = Set(sources.map(\.id))
        let blockIDs = Set(blocks.map(\.id))
        let nodeIDs = Set(graphNodes.map(\.id))
        let importByAlias = Dictionary(uniqueKeysWithValues: imports.map { ($0.alias, $0) })
        let nodeByID = Dictionary(uniqueKeysWithValues: graphNodes.map { ($0.id, $0) })
        let exportInputPortsByName = Dictionary(grouping: exports.flatMap { $0.inputs }, by: \.name)
        let exportOutputPortsByName = Dictionary(grouping: exports.flatMap { $0.outputs }, by: \.name)

        for `import` in imports {
            try `import`.validateExportABI()
        }
        for export in exports {
            try requireUnique(export.inputs.map(\.name), label: "input port for export \(export.id)")
            try requireUnique(export.outputs.map(\.name), label: "output port for export \(export.id)")
            for gate in export.gates where !gateIDs.contains(gate) {
                throw SmeltCAMIRError.malformed(
                    "export '\(export.id)' references unknown gate '\(gate)'"
                )
            }
        }
        for binding in exportBindings {
            guard exportIDs.contains(binding.export) else {
                throw SmeltCAMIRError.malformed(
                    "export binding references unknown export '\(binding.export)'"
                )
            }
            guard flowIDs.contains(binding.flow) else {
                throw SmeltCAMIRError.malformed(
                    "export binding references unknown flow '\(binding.flow)'"
                )
            }
        }
        for export in exports where !exportBindings.contains(where: { $0.export == export.id }) {
            throw SmeltCAMIRError.malformed("export '\(export.id)' has no flow binding")
        }

        for block in blocks {
            try block.validate(sourceIDs: sourceIDs)
        }
        return ValidationContext(
            exportIDs: exportIDs,
            flowIDs: flowIDs,
            sourceIDs: sourceIDs,
            blockIDs: blockIDs,
            nodeIDs: nodeIDs,
            importByAlias: importByAlias,
            nodeByID: nodeByID,
            exportInputPortsByName: exportInputPortsByName,
            exportOutputPortsByName: exportOutputPortsByName
        )
    }

    private func validateGraphAndPolicy(
        exportIDs: Set<String>,
        flowIDs: Set<String>,
        sourceIDs: Set<String>,
        blockIDs: Set<String>,
        nodeIDs: Set<String>,
        importByAlias: [String: Import],
        nodeByID: [String: GraphNode],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws {
        for node in graphNodes {
            try requireUnique(node.inputs.map(\.name), label: "input port for node \(node.id)")
            try requireUnique(node.outputs.map(\.name), label: "output port for node \(node.id)")
            let hasBlock = node.block != nil
            let hasSource = node.source != nil
            let hasImport = node.imported != nil
            switch node.implementation {
            case .compiled:
                guard hasBlock != hasSource, !hasImport else {
                    throw SmeltCAMIRError.malformed(
                        "compiled graph node '\(node.id)' must reference exactly one block or source"
                    )
                }
                if let block = node.block, !blockIDs.contains(block) {
                    throw SmeltCAMIRError.malformed(
                        "compiled graph node '\(node.id)' must reference a declared block"
                    )
                }
                if let source = node.source, !sourceIDs.contains(source) {
                    throw SmeltCAMIRError.malformed(
                        "compiled graph node '\(node.id)' must reference a declared source"
                    )
                }
            case .imported:
                guard !hasBlock, !hasSource else {
                    throw SmeltCAMIRError.malformed(
                        "imported graph node '\(node.id)' must not reference block or source"
                    )
                }
                guard let imported = node.imported,
                      let importedModule = importByAlias[imported.alias],
                      let importedExport = importedModule.export(named: imported.export) else {
                    throw SmeltCAMIRError.malformed(
                        "imported graph node '\(node.id)' must reference an imported export"
                    )
                }
                try validateImportedGraphNode(
                    node,
                    mirrors: importedExport,
                    alias: imported.alias,
                    export: imported.export
                )
            case .native, .adapter:
                guard !hasBlock, !hasSource, !hasImport else {
                    throw SmeltCAMIRError.malformed(
                        "\(node.implementation.rawValue) graph node '\(node.id)' must not reference block, source, or import"
                    )
                }
            }
        }

        var graphValueProducers: [String: Int] = [:]
        var graphValueTypes: [String: TypeRef] = [:]
        for edge in graphEdges {
            let fromType = try validateEndpoint(
                edge.from,
                use: .source,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            let toType = try validateEndpoint(
                edge.to,
                use: .sink,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            if case .graphValue = edge.to.kind, let name = edge.to.name {
                graphValueProducers[name, default: 0] += 1
                graphValueTypes[name] = try resolvedGraphValueType(
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
            let fromType = try graphEndpointType(
                edge.from,
                use: .source,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName,
                graphValueTypes: graphValueTypes
            )
            let toType = try graphEndpointType(
                edge.to,
                use: .sink,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName,
                graphValueTypes: graphValueTypes
            )
            try validateGraphEdgeValueType(edge, fromType: fromType, toType: toType)
        }
        for edge in feedbackEdges {
            try validateEndpoint(
                edge.from,
                use: .source,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
            try validateEndpoint(
                edge.to,
                use: .sink,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
        }

        for flow in flows {
            for phase in flow.phases {
                for call in phase.calls {
                    try validateFlowCall(
                        call,
                        nodeIDs: nodeIDs,
                        importByAlias: importByAlias
                    )
                }
            }
            for endpoint in flow.emit {
                try validateEndpoint(
                    endpoint,
                    use: .emit,
                    nodeByID: nodeByID,
                    importByAlias: importByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
        }

        for tensor in tensors {
            guard sourceIDs.contains(tensor.source) else {
                throw SmeltCAMIRError.malformed(
                    "tensor map references unknown source '\(tensor.source)'"
                )
            }
            guard blockIDs.contains(tensor.owner) else {
                throw SmeltCAMIRError.malformed(
                    "tensor map references unknown owner block '\(tensor.owner)'"
                )
            }
            guard blockIDs.contains(tensor.target.block) else {
                throw SmeltCAMIRError.malformed(
                    "tensor map targets unknown block '\(tensor.target.block)'"
                )
            }
        }

        try validateGates(
            flowIDs: flowIDs,
            exportIDs: exportIDs,
            nodeByID: nodeByID,
            importByAlias: importByAlias,
            exportInputPortsByName: exportInputPortsByName,
            exportOutputPortsByName: exportOutputPortsByName
        )
        try validateQuantization(sourceIDs: sourceIDs, blockIDs: blockIDs)
        try validateSourceQuantization(sourceIDs: sourceIDs)
    }

    private func validateSources() throws {
        var seen = Set<String>()
        for index in sources.indices {
            let source = sources[index]
            guard !source.id.isEmpty, seen.insert(source.id).inserted else {
                throw SmeltCAMIRError.malformed(
                    "duplicate or empty source id '\(source.id)'"
                )
            }
            try source.validate()
        }
    }

    public func canonicalized() -> SmeltCAMIR {
        SmeltCAMIR(
            schemaVersion: schemaVersion,
            module: module,
            imports: imports.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            exports: exports.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            exportBindings: exportBindings.sorted { $0.sortKey < $1.sortKey },
            sources: sources.sorted { $0.sortKey < $1.sortKey },
            blocks: blocks.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            graphNodes: graphNodes.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            graphEdges: graphEdges.sorted { $0.sortKey < $1.sortKey },
            feedbackEdges: feedbackEdges.sorted { $0.sortKey < $1.sortKey },
            flows: flows.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            capabilities: capabilities.sorted(),
            backendConstraints: backendConstraints.sorted { $0.sortKey < $1.sortKey },
            tensors: tensors.sorted { $0.sortKey < $1.sortKey },
            quantization: quantization.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            sourceQuantization: sourceQuantization.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
            compile: compile.sorted { $0.sortKey < $1.sortKey },
            artifacts: artifacts.sorted { $0.sortKey < $1.sortKey },
            gates: gates.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey }
        )
    }

    public func canonicalJSONData(prettyPrinted: Bool = false) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(try validated())
    }

    public func semanticSHA256() throws -> String {
        try Self.sha256Hex(canonicalJSONData())
    }

    public func exportABISHA256() throws -> String {
        try Self.exportABISHA256(for: exports)
    }

    public static func exportABISHA256(for exports: [Export]) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let canonical = exports.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey }
        return try sha256Hex(encoder.encode(canonical))
    }

    private static func sha256Hex(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func validateGates(
        flowIDs: Set<String>,
        exportIDs: Set<String>,
        nodeByID: [String: GraphNode],
        importByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws {
        for gate in gates {
            if gate.requirements.contains(where: { $0.subject == "elapsed" }) {
                guard gate.from != nil, gate.to != nil else {
                    throw SmeltCAMIRError.malformed(
                        "timing gate '\(gate.id)' requires from and to events"
                    )
                }
            }
            try requireUnique(gate.measurements.map(\.subject), label: "measurement subject for gate \(gate.id)")
            let requirementSubjects = Set(gate.requirements.map(\.subject))
            for measurement in gate.measurements {
                try measurement.validate(label: "gate \(gate.id) measurement")
                guard requirementSubjects.contains(measurement.subject) else {
                    throw SmeltCAMIRError.malformed(
                        "gate '\(gate.id)' measurement '\(measurement.subject)' has no matching requirement"
                    )
                }
                if measurement.subject == "elapsed" {
                    guard gate.from != nil, gate.to != nil else {
                        throw SmeltCAMIRError.malformed(
                            "elapsed measurement gate '\(gate.id)' requires from and to events"
                        )
                    }
                }
            }
            try gate.from.map {
                try validateGateEvent(
                    $0,
                    flowIDs: flowIDs,
                    exportIDs: exportIDs,
                    nodeByID: nodeByID,
                    importByAlias: importByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
            try gate.to.map {
                try validateGateEvent(
                    $0,
                    flowIDs: flowIDs,
                    exportIDs: exportIDs,
                    nodeByID: nodeByID,
                    importByAlias: importByAlias,
                    exportInputPortsByName: exportInputPortsByName,
                    exportOutputPortsByName: exportOutputPortsByName
                )
            }
            for requirement in gate.requirements {
                try requirement.validate(label: "gate \(gate.id) requirement")
            }
            for requirement in gate.evidence {
                try requirement.validate(label: "gate \(gate.id) evidence")
            }
        }
    }

    private func validateGateEvent(
        _ event: GateEvent,
        flowIDs: Set<String>,
        exportIDs: Set<String>,
        nodeByID: [String: GraphNode],
        importByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws {
        if let flow = event.flow, !flowIDs.contains(flow) {
            throw SmeltCAMIRError.malformed(
                "gate event references unknown flow '\(flow)'"
            )
        }
        if let export = event.export, !exportIDs.contains(export) {
            throw SmeltCAMIRError.malformed(
                "gate event references unknown export '\(export)'"
            )
        }
        if let endpoint = event.endpoint {
            let use: EndpointUse = event.kind == .input ? .source : .emit
            try validateEndpoint(
                endpoint,
                use: use,
                nodeByID: nodeByID,
                importByAlias: importByAlias,
                exportInputPortsByName: exportInputPortsByName,
                exportOutputPortsByName: exportOutputPortsByName
            )
        }
        for predicate in event.predicates {
            try predicate.validate(label: "gate event predicate")
        }
    }

    private func validateQuantization(
        sourceIDs: Set<String>,
        blockIDs: Set<String>
    ) throws {
        for rule in quantization {
            if let source = rule.source, !sourceIDs.contains(source) {
                throw SmeltCAMIRError.malformed(
                    "quant rule references unknown source '\(source)'"
                )
            }
            if rule.action != .default, !quantRuleTargetsAnyTensor(rule) {
                guard rule.resolution == .sourceDeferred,
                      let source = rule.source,
                      sourceIDs.contains(source) else {
                    throw SmeltCAMIRError.malformed(
                        "quant rule '\(rule.selector.pattern)' targets no tensors"
                    )
                }
            }
            if let calibration = rule.calibration {
                guard sourceIDs.contains(calibration.corpus.source) else {
                    throw SmeltCAMIRError.malformed(
                        "quant calibration references unknown source '\(calibration.corpus.source)'"
                    )
                }
                if let layersPerPass = calibration.layersPerPass, layersPerPass <= 0 {
                    throw SmeltCAMIRError.malformed(
                        "quant calibration layers-per-pass must be positive"
                    )
                }
                for capture in calibration.captures {
                    let block = capture.split(separator: ".", maxSplits: 1).first.map(String.init) ?? capture
                    guard blockIDs.contains(block) else {
                        throw SmeltCAMIRError.malformed(
                            "quant calibration captures unknown block '\(block)'"
                        )
                    }
                }
                for requirement in calibration.requirements {
                    try requirement.validate(label: "quant calibration requirement")
                }
            }
        }
        for (index, lhs) in quantization.enumerated() {
            for rhs in quantization.dropFirst(index + 1)
            where lhs.overlaps(rhs)
                && lhs.action != .default
                && rhs.action != .default {
                guard let lhsPriority = lhs.priority, let rhsPriority = rhs.priority else {
                    throw SmeltCAMIRError.malformed(
                        "overlapping quant rules '\(lhs.selector.pattern)' and "
                            + "'\(rhs.selector.pattern)' need priority"
                    )
                }
                guard lhsPriority != rhsPriority || lhs.resolvedStorageKey == rhs.resolvedStorageKey else {
                    throw SmeltCAMIRError.malformed(
                        "overlapping quant rules '\(lhs.selector.pattern)' and "
                            + "'\(rhs.selector.pattern)' conflict at priority \(lhsPriority)"
                    )
                }
            }
        }
    }

    private func validateSourceQuantization(sourceIDs: Set<String>) throws {
        for rule in sourceQuantization {
            guard sourceIDs.contains(rule.source) else {
                throw SmeltCAMIRError.malformed(
                    "source quant rule references unknown source '\(rule.source)'"
                )
            }
            guard !rule.sourceDTypes.isEmpty else {
                throw SmeltCAMIRError.malformed(
                    "source quant rule for '\(rule.source)' has no source dtypes"
                )
            }
            switch rule.action {
            case .preserve:
                guard rule.storage != nil, rule.targetDType == nil else {
                    throw SmeltCAMIRError.malformed(
                        "source preserve rule for '\(rule.source)' requires storage only"
                    )
                }
            case .dequant:
                guard rule.storage == nil, rule.targetDType?.isEmpty == false else {
                    throw SmeltCAMIRError.malformed(
                        "source dequant rule for '\(rule.source)' requires target dtype"
                    )
                }
            }
            for requirement in rule.evidence {
                try requirement.validate(label: "source quant rule \(rule.source)")
            }
        }
    }

    private func quantRuleTargetsAnyTensor(_ rule: QuantRule) -> Bool {
        tensors.contains {
            rule.selector.targets($0.selector) || rule.selector.targets($0.target)
        }
    }

    private func validateFlowCall(
        _ call: FlowCall,
        nodeIDs: Set<String>,
        importByAlias: [String: Import]
    ) throws {
        switch call.kind {
        case .node:
            guard let node = call.node, nodeIDs.contains(node) else {
                throw SmeltCAMIRError.malformed(
                    "flow call references unknown graph node '\(call.node ?? "")'"
                )
            }
        case .imported:
            guard let imported = call.imported,
                  let importedModule = importByAlias[imported.alias],
                  importedModule.export(named: imported.export) != nil else {
                throw SmeltCAMIRError.malformed("flow call references unknown imported export")
            }
        }
    }

    private func validateImportedGraphNode(
        _ node: GraphNode,
        mirrors export: Export,
        alias: String,
        export exportID: String
    ) throws {
        try validateImportedGraphNodePorts(
            node.inputs,
            expected: export.inputs,
            nodeID: node.id,
            abi: "\(alias).\(exportID)",
            direction: "input"
        )
        try validateImportedGraphNodePorts(
            node.outputs,
            expected: export.outputs,
            nodeID: node.id,
            abi: "\(alias).\(exportID)",
            direction: "output"
        )
    }

    private func validateImportedGraphNodePorts(
        _ projected: [Port],
        expected: [Port],
        nodeID: String,
        abi: String,
        direction: String
    ) throws {
        let projectedByName = Dictionary(uniqueKeysWithValues: projected.map { ($0.name, $0) })
        let expectedByName = Dictionary(uniqueKeysWithValues: expected.map { ($0.name, $0) })
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
                        + "\(formatType(expectedPort.type)), got \(formatType(projectedPort.type))"
                )
            }
        }
    }

    private enum EndpointUse {
        case source
        case sink
        case emit
    }

    @discardableResult
    private func validateEndpoint(
        _ endpoint: EndpointRef,
        use: EndpointUse,
        nodeByID: [String: GraphNode],
        importByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]]
    ) throws -> TypeRef? {
        try endpoint.validateShape()
        switch endpoint.kind {
        case .moduleInput:
            guard use == .source,
                  let name = endpoint.name,
                  exportInputPortsByName[name] != nil else {
                throw SmeltCAMIRError.malformed(
                    "module input endpoint '\(endpoint.sortKey)' is not available as a source"
                )
            }
            return try mergedModulePortType(
                named: name,
                portsByName: exportInputPortsByName,
                endpoint: endpoint,
                role: "input"
            )
        case .moduleOutput:
            guard use != .source,
                  let name = endpoint.name,
                  exportOutputPortsByName[name] != nil else {
                throw SmeltCAMIRError.malformed(
                    "module output endpoint '\(endpoint.sortKey)' is not available as a sink"
                )
            }
            return try mergedModulePortType(
                named: name,
                portsByName: exportOutputPortsByName,
                endpoint: endpoint,
                role: "output"
            )
        case .graphValue:
            return nil
        case .nodePort:
            guard let nodeID = endpoint.node, let port = endpoint.port, let node = nodeByID[nodeID] else {
                throw SmeltCAMIRError.malformed("unknown node endpoint '\(endpoint.sortKey)'")
            }
            let ports = use == .sink ? node.inputs : node.outputs
            guard let resolvedPort = ports.first(where: { $0.name == port }) else {
                throw SmeltCAMIRError.malformed(
                    "node endpoint '\(endpoint.sortKey)' has no \(use == .sink ? "input" : "output") port"
                )
            }
            return resolvedPort.type
        case .importedPort:
            guard let alias = endpoint.importAlias,
                  let export = endpoint.export,
                  let port = endpoint.port,
                  let imported = importByAlias[alias],
                  let abi = imported.export(named: export) else {
                throw SmeltCAMIRError.malformed(
                    "unknown imported endpoint '\(endpoint.sortKey)'"
                )
            }
            let ports = use == .sink ? abi.inputs : abi.outputs
            guard let resolvedPort = ports.first(where: { $0.name == port }) else {
                throw SmeltCAMIRError.malformed(
                    "imported endpoint '\(endpoint.sortKey)' has no "
                        + "\(use == .sink ? "input" : "output") port"
                )
            }
            return resolvedPort.type
        }
    }

    private func graphEndpointType(
        _ endpoint: EndpointRef,
        use: EndpointUse,
        nodeByID: [String: GraphNode],
        importByAlias: [String: Import],
        exportInputPortsByName: [String: [Port]],
        exportOutputPortsByName: [String: [Port]],
        graphValueTypes: [String: TypeRef]
    ) throws -> TypeRef? {
        let staticType = try validateEndpoint(
            endpoint,
            use: use,
            nodeByID: nodeByID,
            importByAlias: importByAlias,
            exportInputPortsByName: exportInputPortsByName,
            exportOutputPortsByName: exportOutputPortsByName
        )
        guard case .graphValue = endpoint.kind, let name = endpoint.name else {
            return staticType
        }
        return graphValueTypes[name]
    }

    private func resolvedGraphValueType(
        edge: GraphEdge,
        fromType: TypeRef?,
        toType: TypeRef?
    ) throws -> TypeRef? {
        guard case .graphValue = edge.to.kind else { return nil }
        var resolved = fromType
        if let toType {
            resolved = try resolved.map {
                try mergedCompatibleType($0, toType, context: "graph edge \(edge.sortKey)")
            } ?? toType
        }
        if let declared = edge.type {
            resolved = try resolved.map {
                try mergedCompatibleType($0, declared, context: "graph edge \(edge.sortKey)")
            } ?? declared
        }
        return resolved
    }

    private func validateGraphEdgeValueType(
        _ edge: GraphEdge,
        fromType: TypeRef?,
        toType: TypeRef?
    ) throws {
        if let declared = edge.type {
            try validateDeclaredValueType(
                declared,
                matches: fromType,
                edge: edge,
                endpoint: edge.from,
                role: "source"
            )
            try validateDeclaredValueType(
                declared,
                matches: toType,
                edge: edge,
                endpoint: edge.to,
                role: "sink"
            )
        }
        if let fromType, let toType, !typesCanConnect(fromType, toType) {
            throw SmeltCAMIRError.malformed(
                "graph edge \(edge.sortKey) connects source port "
                    + "\(formatType(fromType)) to sink port \(formatType(toType))"
            )
        }
    }

    private func validateDeclaredValueType(
        _ declared: TypeRef,
        matches resolved: TypeRef?,
        edge: GraphEdge,
        endpoint: EndpointRef,
        role: String
    ) throws {
        guard let resolved, declaredDescribes(declared, resolved) else {
            if let resolved {
                throw SmeltCAMIRError.malformed(
                    "graph edge \(edge.sortKey) declares value type \(formatType(declared)), "
                        + "but \(role) endpoint \(endpoint.sortKey) resolves to \(formatType(resolved))"
                )
            }
            return
        }
    }

    private func mergedModulePortType(
        named name: String,
        portsByName: [String: [Port]],
        endpoint: EndpointRef,
        role: String
    ) throws -> TypeRef? {
        guard let ports = portsByName[name], var merged = ports.first?.type else {
            return nil
        }
        for port in ports.dropFirst() {
            do {
                merged = try mergedCompatibleType(
                    merged,
                    port.type,
                    context: "module \(role) endpoint '\(endpoint.sortKey)'"
                )
            } catch {
                throw SmeltCAMIRError.malformed(
                    "module \(role) endpoint '\(endpoint.sortKey)' has ambiguous declared types "
                        + "\(formatType(merged)) and \(formatType(port.type))"
                )
            }
        }
        return merged
    }

    private func mergedCompatibleType(_ lhs: TypeRef, _ rhs: TypeRef, context: String) throws -> TypeRef {
        guard lhs.name == rhs.name else {
            throw SmeltCAMIRError.malformed(
                "\(context) connects incompatible value types \(formatType(lhs)) and \(formatType(rhs))"
            )
        }
        var attributes = lhs.attributes
        for (key, value) in rhs.attributes {
            if let existing = attributes[key], existing != value {
                throw SmeltCAMIRError.malformed(
                    "\(context) connects incompatible value types \(formatType(lhs)) and \(formatType(rhs))"
                )
            }
            attributes[key] = value
        }
        return TypeRef(lhs.name, attributes: attributes)
    }

    private func declaredDescribes(_ declared: TypeRef, _ resolved: TypeRef) -> Bool {
        guard declared.name == resolved.name else { return false }
        for (key, value) in declared.attributes {
            if let resolvedValue = resolved.attributes[key], resolvedValue != value {
                return false
            }
        }
        return true
    }

    private func typesCanConnect(_ lhs: TypeRef, _ rhs: TypeRef) -> Bool {
        guard lhs.name == rhs.name else { return false }
        for (key, value) in lhs.attributes {
            if let rhsValue = rhs.attributes[key], rhsValue != value {
                return false
            }
        }
        return true
    }

    private func formatType(_ type: TypeRef) -> String {
        guard !type.attributes.isEmpty else { return type.name }
        let attributes = type.attributes
            .map { "\($0.key)=\($0.value)" }
            .sorted()
            .joined(separator: ",")
        return "\(type.name)[\(attributes)]"
    }

    private func requireUnique(_ values: [String], label: String) throws {
        var seen = Set<String>()
        for value in values {
            guard !value.isEmpty, seen.insert(value).inserted else {
                throw SmeltCAMIRError.malformed("duplicate or empty \(label) '\(value)'")
            }
        }
    }
}

public enum SmeltCAMIRError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "cam ir: \(why)"
        }
    }
}

public extension SmeltCAMIR {
    struct Module: Codable, Sendable, Equatable {
        public let id: String
        public let version: String?

        public init(id: String, version: String? = nil) {
            self.id = id
            self.version = version
        }
    }

    struct Import: Codable, Sendable, Equatable {
        public let alias: String
        public let moduleID: String
        public let canonicalURI: String?
        public let irSHA256: String
        public let exportABISHA256: String
        public let exportABI: [Export]
        public let parameters: [String: String]

        enum CodingKeys: String, CodingKey {
            case alias
            case moduleID
            case irSHA256
            case exportABISHA256
            case exportABI
            case parameters
        }

        public init(
            alias: String,
            moduleID: String,
            canonicalURI: String? = nil,
            irSHA256: String,
            exportABISHA256: String,
            exportABI: [Export],
            parameters: [String: String] = [:]
        ) {
            self.alias = alias
            self.moduleID = moduleID
            self.canonicalURI = canonicalURI
            self.irSHA256 = irSHA256
            self.exportABISHA256 = exportABISHA256
            self.exportABI = exportABI
            self.parameters = parameters
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            alias = try container.decode(String.self, forKey: .alias)
            moduleID = try container.decode(String.self, forKey: .moduleID)
            canonicalURI = nil
            irSHA256 = try container.decode(String.self, forKey: .irSHA256)
            exportABISHA256 = try container.decode(String.self, forKey: .exportABISHA256)
            exportABI = try container.decode([Export].self, forKey: .exportABI)
            parameters = try container.decodeIfPresent([String: String].self, forKey: .parameters) ?? [:]
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(alias, forKey: .alias)
            try container.encode(moduleID, forKey: .moduleID)
            try container.encode(irSHA256, forKey: .irSHA256)
            try container.encode(exportABISHA256, forKey: .exportABISHA256)
            try container.encode(exportABI.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey }, forKey: .exportABI)
            if !parameters.isEmpty {
                try container.encode(parameters, forKey: .parameters)
            }
        }

        fileprivate var sortKey: String { alias }

        fileprivate func canonicalized() -> Import {
            Import(
                alias: alias,
                moduleID: moduleID,
                canonicalURI: canonicalURI,
                irSHA256: irSHA256,
                exportABISHA256: exportABISHA256,
                exportABI: exportABI.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey },
                parameters: parameters
            )
        }

        fileprivate func export(named name: String) -> Export? {
            exportABI.first { $0.id == name }
        }

        fileprivate func validateExportABI() throws {
            try validateImportedExports(exportABI)
            let computed = try SmeltCAMIR.exportABISHA256(for: exportABI)
            guard computed == exportABISHA256 else {
                throw SmeltCAMIRError.malformed(
                    "import '\(alias)' export ABI hash mismatch"
                )
            }
        }

        private func validateImportedExports(_ exports: [Export]) throws {
            try requireUniqueExportABIValues(exports.map(\.id), label: "export ABI for import \(alias)")
            for export in exports {
                try validateImportedPorts(export.inputs, label: "input port for imported export \(export.id)")
                try validateImportedPorts(export.outputs, label: "output port for imported export \(export.id)")
            }
        }

        private func validateImportedPorts(_ ports: [Port], label: String) throws {
            try requireUniqueExportABIValues(ports.map(\.name), label: label)
            for port in ports {
                guard !port.type.name.isEmpty else {
                    throw SmeltCAMIRError.malformed("\(label) '\(port.name)' has empty type")
                }
            }
        }

        private func requireUniqueExportABIValues(_ values: [String], label: String) throws {
            var seen = Set<String>()
            for value in values {
                guard !value.isEmpty, seen.insert(value).inserted else {
                    throw SmeltCAMIRError.malformed("duplicate or empty \(label) '\(value)'")
                }
            }
        }
    }

    struct TypeRef: Codable, Sendable, Equatable {
        public let name: String
        public let attributes: [String: String]

        public init(_ name: String, attributes: [String: String] = [:]) {
            self.name = name
            self.attributes = attributes
        }
    }

    struct Port: Codable, Sendable, Equatable {
        public let name: String
        public let type: TypeRef
        public let optional: Bool

        public init(name: String, type: TypeRef, optional: Bool = false) {
            self.name = name
            self.type = type
            self.optional = optional
        }

        fileprivate var sortKey: String { name }
    }

    struct Export: Codable, Sendable, Equatable {
        public let id: String
        public let inputs: [Port]
        public let outputs: [Port]
        public let capabilities: [String]
        public let gates: [String]

        public init(
            id: String,
            inputs: [Port],
            outputs: [Port],
            capabilities: [String] = [],
            gates: [String] = []
        ) {
            self.id = id
            self.inputs = inputs
            self.outputs = outputs
            self.capabilities = capabilities
            self.gates = gates
        }

        fileprivate var sortKey: String { id }

        fileprivate func canonicalized() -> Export {
            Export(
                id: id,
                inputs: inputs.sorted { $0.sortKey < $1.sortKey },
                outputs: outputs.sorted { $0.sortKey < $1.sortKey },
                capabilities: capabilities.sorted(),
                gates: gates.sorted()
            )
        }
    }

    struct ExportBinding: Codable, Sendable, Equatable {
        public let export: String
        public let flow: String

        public init(export: String, flow: String) {
            self.export = export
            self.flow = flow
        }

        fileprivate var sortKey: String { export }
    }

    struct Source: Codable, Sendable, Equatable {
        public let id: String
        public let kind: String
        public let locator: String
        public let revision: String?
        public let checkpointMap: String?
        /// Integrity of the exact fetched source body, when the locator names a
        /// single file. Repository/checkpoint collections remain revision-pinned
        /// and can carry their inventory digest as a separate source.
        public let sha256: String?
        public let byteCount: UInt64?

        public init(
            id: String,
            kind: String,
            locator: String,
            revision: String? = nil,
            checkpointMap: String? = nil,
            sha256: String? = nil,
            byteCount: UInt64? = nil
        ) {
            self.id = id
            self.kind = kind
            self.locator = locator
            self.revision = revision
            self.checkpointMap = checkpointMap
            self.sha256 = sha256
            self.byteCount = byteCount
        }

        fileprivate var sortKey: String { id }

        fileprivate func validate() throws {
            if let sha256 {
                let isHex = sha256.utf8.allSatisfy { byte in
                    (48...57).contains(byte)
                        || (65...70).contains(byte)
                        || (97...102).contains(byte)
                }
                guard sha256.count == 64, isHex else {
                    throw SmeltCAMIRError.malformed(
                        "source '\(id)' has an invalid SHA-256 digest"
                    )
                }
            }
            if let byteCount, byteCount == 0 {
                throw SmeltCAMIRError.malformed(
                    "source '\(id)' byte count must be positive"
                )
            }
        }
    }

    enum BlockOperator: String, Codable, Sendable {
        case transformer
        case codecDecoder = "codec-decoder"
        case codecHead = "codec-head"
        case ttsFrontend = "tts-frontend"
        case transformerEncoder = "transformer-encoder"
        case transformerDecoder = "transformer-decoder"
        case patchEncoder = "patch-encoder"
        case discreteAudioEncoder = "discrete-audio-encoder"
        case adapter
    }

    struct Block: Codable, Sendable, Equatable {
        public let id: String
        public let operatorName: BlockOperator
        public let shape: BlockShape
        public let annotations: [Constraint]

        public init(
            id: String,
            operatorName: BlockOperator,
            shape: BlockShape,
            annotations: [Constraint] = []
        ) {
            self.id = id
            self.operatorName = operatorName
            self.shape = shape
            self.annotations = annotations
        }

        fileprivate var sortKey: String { id }

        fileprivate func canonicalized() -> Block {
            Block(
                id: id,
                operatorName: operatorName,
                shape: shape.canonicalized(),
                annotations: annotations.sorted { $0.sortKey < $1.sortKey }
            )
        }

        fileprivate func validate(sourceIDs: Set<String>) throws {
            if let derivation = shape.derivation, !sourceIDs.contains(derivation.source) {
                throw SmeltCAMIRError.malformed(
                    "block '\(id)' derives shape from unknown source '\(derivation.source)'"
                )
            }
            try shape.validate(blockID: id)
        }
    }

    struct BlockShape: Codable, Sendable, Equatable {
        public let derivation: ShapeDerivation?
        public let transformer: TransformerShape?
        public let codecDecoder: CodecDecoderShape?
        public let frontend: FrontendShape?
        public let requirements: [BlockRequirement]

        public init(
            derivation: ShapeDerivation? = nil,
            transformer: TransformerShape? = nil,
            codecDecoder: CodecDecoderShape? = nil,
            frontend: FrontendShape? = nil,
            requirements: [BlockRequirement] = []
        ) {
            self.derivation = derivation
            self.transformer = transformer
            self.codecDecoder = codecDecoder
            self.frontend = frontend
            self.requirements = requirements
        }

        fileprivate func canonicalized() -> BlockShape {
            BlockShape(
                derivation: derivation,
                transformer: transformer?.canonicalized(),
                codecDecoder: codecDecoder,
                frontend: frontend,
                requirements: requirements.sorted { $0.sortKey < $1.sortKey }
            )
        }

        fileprivate func validate(blockID: String) throws {
            try transformer?.validate(blockID: blockID)
        }
    }

    struct ShapeDerivation: Codable, Sendable, Equatable {
        public let source: String
        public let authority: String?

        public init(source: String, authority: String? = nil) {
            self.source = source
            self.authority = authority
        }
    }

    struct BlockRequirement: Codable, Sendable, Equatable {
        public let key: String
        public let value: String?
        public let optional: Bool

        public init(_ key: String, _ value: String? = nil, optional: Bool = false) {
            self.key = key
            self.value = value
            self.optional = optional
        }

        fileprivate var sortKey: String { "\(key)=\(value ?? "")=\(optional)" }
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
        /// Logical projections that consume the same activation. The module
        /// owns this graph fact; storage layout and kernel selection remain
        /// compiler decisions.
        public let projectionBanks: [ProjectionBank]?
        /// Number of leading transformer layers that use `ffn` instead of
        /// `expert`. The remaining layers use the sparse expert path.
        public let denseLayerCount: Int?
        /// Stateful depthwise short convolutions placed at semantic sites in
        /// each transformer layer. Their implementation is backend-owned.
        public let shortConvolutions: [ShortConvolutionShape]?
        public let sharedKVLayers: Int?
        public let logitCap: String?

        public init(
            hiddenSize: Int? = nil,
            layers: LayerPattern? = nil,
            delta: DeltaShape? = nil,
            attention: AttentionShape? = nil,
            attentionByRole: [RoleAttentionShape]? = nil,
            ffn: FFNShape? = nil,
            router: RouterShape? = nil,
            expert: ExpertShape? = nil,
            norm: NormShape? = nil,
            vocab: VocabShape? = nil,
            perLayerInput: PerLayerInputShape? = nil,
            projectionBanks: [ProjectionBank]? = nil,
            denseLayerCount: Int? = nil,
            shortConvolutions: [ShortConvolutionShape]? = nil,
            sharedKVLayers: Int? = nil,
            logitCap: String? = nil
        ) {
            self.hiddenSize = hiddenSize
            self.layers = layers
            self.delta = delta
            self.attention = attention
            self.attentionByRole = attentionByRole
            self.ffn = ffn
            self.router = router
            self.expert = expert
            self.norm = norm
            self.vocab = vocab
            self.perLayerInput = perLayerInput
            self.projectionBanks = projectionBanks
            self.denseLayerCount = denseLayerCount
            self.shortConvolutions = shortConvolutions
            self.sharedKVLayers = sharedKVLayers
            self.logitCap = logitCap
        }

        fileprivate func canonicalized() -> TransformerShape {
            TransformerShape(
                hiddenSize: hiddenSize,
                layers: layers,
                delta: delta?.canonicalized(),
                attention: attention,
                attentionByRole: attentionByRole?.sorted { $0.sortKey < $1.sortKey },
                ffn: ffn,
                router: router,
                expert: expert,
                norm: norm,
                vocab: vocab,
                perLayerInput: perLayerInput,
                projectionBanks: projectionBanks?
                    .map { $0.canonicalized() }
                    .sorted { $0.id < $1.id },
                denseLayerCount: denseLayerCount,
                shortConvolutions: shortConvolutions?.sorted { $0.site.rawValue < $1.site.rawValue },
                sharedKVLayers: sharedKVLayers,
                logitCap: logitCap
            )
        }

        fileprivate func validate(blockID: String) throws {
            if let attention, attention.qkNormMode != nil, attention.qkNorm == nil {
                throw SmeltCAMIRError.malformed(
                    "block '\(blockID)' declares qk-norm-mode without qk-norm"
                )
            }
            if let attentionByRole {
                var roles = Set<LayerRole>()
                for variant in attentionByRole {
                    guard roles.insert(variant.role).inserted else {
                        throw SmeltCAMIRError.malformed(
                            "block '\(blockID)' has duplicate attention shape for role '\(variant.role.rawValue)'"
                        )
                    }
                    if variant.attention.qkNormMode != nil,
                       variant.attention.qkNorm == nil {
                        throw SmeltCAMIRError.malformed(
                            "block '\(blockID)' role '\(variant.role.rawValue)' declares qk-norm-mode without qk-norm"
                        )
                    }
                }
            }
            if let sharedKVLayers, sharedKVLayers < 0 {
                throw SmeltCAMIRError.malformed(
                    "block '\(blockID)' shared-kv-layers must be non-negative"
                )
            }
            if let denseLayerCount {
                let layerCount = layers?.count ?? layers?.repeatCount.map {
                    $0 * (layers?.roles.count ?? 0)
                }
                guard denseLayerCount >= 0,
                      layerCount.map({ denseLayerCount <= $0 }) ?? true else {
                    throw SmeltCAMIRError.malformed(
                        "block '\(blockID)' dense-layer-count is outside its layer pattern"
                    )
                }
            }
            if let shortConvolutions {
                var sites = Set<ShortConvolutionSite>()
                for convolution in shortConvolutions {
                    guard convolution.kernelSize > 0,
                          sites.insert(convolution.site).inserted else {
                        throw SmeltCAMIRError.malformed(
                            "block '\(blockID)' has an invalid or duplicate short-convolution site"
                        )
                    }
                }
            }
            if let projectionBanks {
                var bankIDs = Set<String>()
                var bankedEndpoints = Set<ProjectionEndpoint>()
                for bank in projectionBanks {
                    guard !bank.id.isEmpty, bankIDs.insert(bank.id).inserted else {
                        throw SmeltCAMIRError.malformed(
                            "block '\(blockID)' has an empty or duplicate projection bank id '\(bank.id)'"
                        )
                    }
                    guard !bank.outputs.isEmpty else {
                        throw SmeltCAMIRError.malformed(
                            "block '\(blockID)' projection bank '\(bank.id)' needs at least one output"
                        )
                    }
                    if let spans = bank.activationViewLayerSpans {
                        guard bank.activationView != nil else {
                            throw SmeltCAMIRError.malformed(
                                "block '\(blockID)' projection bank '\(bank.id)' has activation-view spans without an activation view"
                            )
                        }
                        let sortedSpans = spans.sorted { $0.start < $1.start }
                        var previousEnd = 0
                        for span in sortedSpans {
                            guard span.start >= 0, span.count > 0 else {
                                throw SmeltCAMIRError.malformed(
                                    "block '\(blockID)' projection bank '\(bank.id)' has an invalid activation-view layer span"
                                )
                            }
                            let (end, overflow) = span.start.addingReportingOverflow(span.count)
                            guard !overflow else {
                                throw SmeltCAMIRError.malformed(
                                    "block '\(blockID)' projection bank '\(bank.id)' has an invalid activation-view layer span"
                                )
                            }
                            guard span.start >= previousEnd else {
                                throw SmeltCAMIRError.malformed(
                                    "block '\(blockID)' projection bank '\(bank.id)' has overlapping activation-view layer spans"
                                )
                            }
                            previousEnd = end
                        }
                        let layerCount = layers?.count ?? layers?.repeatCount.map {
                            $0 * (layers?.roles.count ?? 0)
                        }
                        if let layerCount,
                           previousEnd > layerCount {
                            throw SmeltCAMIRError.malformed(
                                "block '\(blockID)' projection bank '\(bank.id)' activation-view span exceeds \(layerCount) layers"
                            )
                        }
                    }
                    var localEndpoints = Set<ProjectionEndpoint>()
                    for endpoint in bank.outputs {
                        guard localEndpoints.insert(endpoint).inserted,
                              bankedEndpoints.insert(endpoint).inserted else {
                            throw SmeltCAMIRError.malformed(
                                "block '\(blockID)' projection endpoint '\(endpoint.rawValue)' is banked more than once"
                            )
                        }
                        guard endpoint.source == bank.source else {
                            throw SmeltCAMIRError.malformed(
                                "block '\(blockID)' projection bank '\(bank.id)' mixes source '\(bank.source.rawValue)' with endpoint '\(endpoint.rawValue)'"
                            )
                        }
                    }
                }
            }
        }
    }

    enum ProjectionSource: String, Codable, Sendable, Hashable {
        case deltaInput
        case deltaOutput
        case attentionInput
        case attentionOutput
        case ffnInput
        case ffnIntermediate
        case lmHeadInput
    }

    enum ProjectionEndpoint: String, Codable, Sendable, Hashable {
        case deltaQKV
        case deltaZ
        case deltaA
        case deltaB
        case deltaOut
        case attentionQ
        case attentionK
        case attentionV
        case attentionRelative
        case attentionOut
        case ffnGate
        case ffnUp
        case ffnDown
        case lmHead

        fileprivate var source: ProjectionSource {
            switch self {
            case .deltaQKV, .deltaZ, .deltaA, .deltaB:
                return .deltaInput
            case .deltaOut:
                return .deltaOutput
            case .attentionQ, .attentionK, .attentionV, .attentionRelative:
                return .attentionInput
            case .attentionOut:
                return .attentionOutput
            case .ffnGate, .ffnUp:
                return .ffnInput
            case .ffnDown:
                return .ffnIntermediate
            case .lmHead:
                return .lmHeadInput
            }
        }
    }

    enum ProjectionActivationView: String, Codable, Sendable, Hashable {
        case signedBitplanesI2
        case signedBitplanesI3
        case signedBitplanesI4
        case signedBitplanesI5
        case signedBitplanesI6

        public var bitCount: Int {
            switch self {
            case .signedBitplanesI2: return 2
            case .signedBitplanesI3: return 3
            case .signedBitplanesI4: return 4
            case .signedBitplanesI5: return 5
            case .signedBitplanesI6: return 6
            }
        }
    }

    struct ProjectionBank: Codable, Sendable, Equatable {
        public let id: String
        public let source: ProjectionSource
        /// Authored order defines the physical row order when a backend packs
        /// the bank. Each endpoint remains independently addressable.
        public let outputs: [ProjectionEndpoint]
        /// Optional approximate producer view shared by compatible consumers.
        /// Nil preserves the trunk's native activation representation.
        public let activationView: ProjectionActivationView?
        /// Optional repeated-layer spans where the approximate view applies.
        /// Nil means every layer; an empty list means no layer.
        public let activationViewLayerSpans: [ActivationViewLayerSpan]?

        public init(
            id: String,
            source: ProjectionSource,
            outputs: [ProjectionEndpoint],
            activationView: ProjectionActivationView? = nil,
            activationViewLayerSpans: [ActivationViewLayerSpan]? = nil
        ) {
            self.id = id
            self.source = source
            self.outputs = outputs
            self.activationView = activationView
            self.activationViewLayerSpans = activationViewLayerSpans
        }

        fileprivate func canonicalized() -> ProjectionBank {
            ProjectionBank(
                id: id,
                source: source,
                outputs: outputs,
                activationView: activationView,
                activationViewLayerSpans: activationViewLayerSpans?.sorted {
                    ($0.start, $0.count) < ($1.start, $1.count)
                }
            )
        }
    }

    struct ActivationViewLayerSpan: Codable, Sendable, Equatable {
        public let start: Int
        public let count: Int

        public init(start: Int, count: Int) {
            self.start = start
            self.count = count
        }
    }

    struct LayerPattern: Codable, Sendable, Equatable {
        public let count: Int?
        public let roles: [LayerRole]
        public let repeatCount: Int?

        public init(count: Int? = nil, roles: [LayerRole] = [], repeatCount: Int? = nil) {
            self.count = count
            self.roles = roles
            self.repeatCount = repeatCount
        }
    }

    enum LayerRole: String, Codable, Sendable, Hashable {
        case delta
        case attention
        case sliding
        case global
    }

    struct DeltaShape: Codable, Sendable, Equatable {
        public let heads: Int
        public let headDim: Int
        public let convKernel: Int?
        public let projections: [String: Int]

        public init(
            heads: Int,
            headDim: Int,
            convKernel: Int? = nil,
            projections: [String: Int] = [:]
        ) {
            self.heads = heads
            self.headDim = headDim
            self.convKernel = convKernel
            self.projections = projections
        }

        fileprivate func canonicalized() -> DeltaShape {
            DeltaShape(
                heads: heads,
                headDim: headDim,
                convKernel: convKernel,
                projections: projections
            )
        }
    }

    struct AttentionShape: Codable, Sendable, Equatable {
        public let qHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rope: RopeShape?
        public let relativePosition: RelativePositionShape?
        public let scaling: AttentionScaling?
        public let qkNorm: NormKind?
        /// Weight semantics for the learned Q/K norm scale. This is independent
        /// of the transformer residual norm: a graph may use one-plus-weight
        /// residual norms and direct learned weights for its per-head Q/K norms.
        public let qkNormMode: NormMode?
        public let vNorm: NormKind?
        public let window: Int?

        public init(
            qHeads: Int,
            kvHeads: Int,
            headDim: Int,
            rope: RopeShape? = nil,
            relativePosition: RelativePositionShape? = nil,
            scaling: AttentionScaling? = nil,
            qkNorm: NormKind? = nil,
            qkNormMode: NormMode? = nil,
            vNorm: NormKind? = nil,
            window: Int? = nil
        ) {
            self.qHeads = qHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.rope = rope
            self.relativePosition = relativePosition
            self.scaling = scaling
            self.qkNorm = qkNorm
            self.qkNormMode = qkNormMode
            self.vNorm = vNorm
            self.window = window
        }
    }

    /// A learned content-conditioned relative-position bias. The projected
    /// query state selects a bias value by backward distance; distances beyond
    /// `extent` contribute no bias.
    struct RelativePositionShape: Codable, Sendable, Equatable {
        public let projectionDim: Int
        public let extent: Int
        public let contentConditioned: Bool
        public let logScalingFloor: Int?
        public let logScalingAlpha: String?

        public init(
            projectionDim: Int,
            extent: Int,
            contentConditioned: Bool = true,
            logScalingFloor: Int? = nil,
            logScalingAlpha: String? = nil
        ) {
            self.projectionDim = projectionDim
            self.extent = extent
            self.contentConditioned = contentConditioned
            self.logScalingFloor = logScalingFloor
            self.logScalingAlpha = logScalingAlpha
        }
    }

    enum AttentionScaling: String, Codable, Sendable {
        case inverseSquareRootHeadDim = "inverse-sqrt-head-dim"
        case inverseHeadDim = "inverse-head-dim"
    }

    struct RoleAttentionShape: Codable, Sendable, Equatable {
        public let role: LayerRole
        public let attention: AttentionShape

        public init(role: LayerRole, attention: AttentionShape) {
            self.role = role
            self.attention = attention
        }

        fileprivate var sortKey: String { role.rawValue }
    }

    struct RopeShape: Codable, Sendable, Equatable {
        public let kind: RopeKind
        public let theta: Int?

        public init(kind: RopeKind, theta: Int? = nil) {
            self.kind = kind
            self.theta = theta
        }
    }

    enum RopeKind: String, Codable, Sendable {
        case neox
        case yarn
    }

    enum NormKind: String, Codable, Sendable {
        case rms
        case layer
    }

    struct FFNShape: Codable, Sendable, Equatable {
        public let dim: Int
        public let activation: Activation

        public init(dim: Int, activation: Activation) {
            self.dim = dim
            self.activation = activation
        }
    }

    enum Activation: String, Codable, Sendable {
        case swiglu
        case geglu
        case gelu
        case silu
    }

    struct RouterShape: Codable, Sendable, Equatable {
        public let topK: Int
        public let experts: Int
        public let sharedExperts: Int?
        public let activation: RouterActivation?
        public let normalization: RouterNormalization?
        public let scoreCorrectionBias: Bool?
        public let routeScale: String?
        public let globalScale: Bool?
        public let sharedExpertSink: Bool?

        public init(
            topK: Int,
            experts: Int,
            sharedExperts: Int? = nil,
            activation: RouterActivation? = nil,
            normalization: RouterNormalization? = nil,
            scoreCorrectionBias: Bool? = nil,
            routeScale: String? = nil,
            globalScale: Bool? = nil,
            sharedExpertSink: Bool? = nil
        ) {
            self.topK = topK
            self.experts = experts
            self.sharedExperts = sharedExperts
            self.activation = activation
            self.normalization = normalization
            self.scoreCorrectionBias = scoreCorrectionBias
            self.routeScale = routeScale
            self.globalScale = globalScale
            self.sharedExpertSink = sharedExpertSink
        }
    }

    enum RouterActivation: String, Codable, Sendable {
        case sigmoid
        case softmax
    }

    enum RouterNormalization: String, Codable, Sendable {
        case selected
        case selectedAndShared = "selected-and-shared"
    }

    struct ExpertShape: Codable, Sendable, Equatable {
        public let ffn: FFNShape

        public init(ffn: FFNShape) {
            self.ffn = ffn
        }
    }

    enum ShortConvolutionSite: String, Codable, Sendable, Hashable {
        case attentionKey = "attention-key"
        case attentionValue = "attention-value"
        case attentionBranchOutput = "attention-branch-output"
        case ffnBranchOutput = "ffn-branch-output"
    }

    enum ShortConvolutionResidual: String, Codable, Sendable {
        case none
        case addInput = "add-input"
    }

    struct ShortConvolutionShape: Codable, Sendable, Equatable {
        public let site: ShortConvolutionSite
        public let kernelSize: Int
        public let residual: ShortConvolutionResidual

        public init(
            site: ShortConvolutionSite,
            kernelSize: Int,
            residual: ShortConvolutionResidual = .none
        ) {
            self.site = site
            self.kernelSize = kernelSize
            self.residual = residual
        }
    }

    struct NormShape: Codable, Sendable, Equatable {
        public let kind: NormKind
        public let eps: String?
        public let mode: NormMode?

        public init(kind: NormKind, eps: String? = nil, mode: NormMode? = nil) {
            self.kind = kind
            self.eps = eps
            self.mode = mode
        }
    }

    enum NormMode: String, Codable, Sendable {
        case weight
        case onePlusWeight = "one-plus-weight"
    }

    struct VocabShape: Codable, Sendable, Equatable {
        public let size: Int
        public let tiedHead: Bool

        public init(size: Int, tiedHead: Bool) {
            self.size = size
            self.tiedHead = tiedHead
        }
    }

    struct PerLayerInputShape: Codable, Sendable, Equatable {
        public let hiddenSize: Int
        public let vocabSize: Int

        public init(hiddenSize: Int, vocabSize: Int) {
            self.hiddenSize = hiddenSize
            self.vocabSize = vocabSize
        }
    }

    struct CodecDecoderShape: Codable, Sendable, Equatable {
        public let codebooks: Int?
        public let streaming: Bool

        public init(codebooks: Int? = nil, streaming: Bool = false) {
            self.codebooks = codebooks
            self.streaming = streaming
        }
    }

    struct FrontendShape: Codable, Sendable, Equatable {
        public let speakerConditioning: Bool

        public init(speakerConditioning: Bool = false) {
            self.speakerConditioning = speakerConditioning
        }
    }

    enum GraphImplementation: String, Codable, Sendable {
        case native
        case compiled
        case imported
        case adapter
    }

    struct ImportedExportRef: Codable, Sendable, Equatable {
        public let alias: String
        public let export: String

        public init(alias: String, export: String) {
            self.alias = alias
            self.export = export
        }

        fileprivate var sortKey: String { "\(alias).\(export)" }
    }

    struct GraphNode: Codable, Sendable, Equatable {
        public let id: String
        public let implementation: GraphImplementation
        public let block: String?
        public let source: String?
        public let imported: ImportedExportRef?
        public let inputs: [Port]
        public let outputs: [Port]
        public let annotations: [Constraint]

        public init(
            id: String,
            implementation: GraphImplementation,
            block: String? = nil,
            source: String? = nil,
            imported: ImportedExportRef? = nil,
            inputs: [Port],
            outputs: [Port],
            annotations: [Constraint] = []
        ) {
            self.id = id
            self.implementation = implementation
            self.block = block
            self.source = source
            self.imported = imported
            self.inputs = inputs
            self.outputs = outputs
            self.annotations = annotations
        }

        fileprivate var sortKey: String { id }

        fileprivate func canonicalized() -> GraphNode {
            GraphNode(
                id: id,
                implementation: implementation,
                block: block,
                source: source,
                imported: imported,
                inputs: inputs.sorted { $0.sortKey < $1.sortKey },
                outputs: outputs.sorted { $0.sortKey < $1.sortKey },
                annotations: annotations.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }

    struct EndpointRef: Codable, Sendable, Equatable {
        public enum Kind: String, Codable, Sendable {
            case moduleInput
            case moduleOutput
            case graphValue
            case nodePort
            case importedPort
        }

        public let kind: Kind
        public let name: String?
        public let node: String?
        public let port: String?
        public let importAlias: String?
        public let export: String?

        public init(
            kind: Kind,
            name: String? = nil,
            node: String? = nil,
            port: String? = nil,
            importAlias: String? = nil,
            export: String? = nil
        ) {
            self.kind = kind
            self.name = name
            self.node = node
            self.port = port
            self.importAlias = importAlias
            self.export = export
        }

        public static func moduleInput(_ name: String) -> EndpointRef {
            EndpointRef(kind: .moduleInput, name: name)
        }

        public static func moduleOutput(_ name: String) -> EndpointRef {
            EndpointRef(kind: .moduleOutput, name: name)
        }

        public static func graphValue(_ name: String) -> EndpointRef {
            EndpointRef(kind: .graphValue, name: name)
        }

        public static func node(_ node: String, _ port: String) -> EndpointRef {
            EndpointRef(kind: .nodePort, node: node, port: port)
        }

        public static func imported(alias: String, export: String, port: String) -> EndpointRef {
            EndpointRef(kind: .importedPort, port: port, importAlias: alias, export: export)
        }

        fileprivate var sortKey: String {
            switch kind {
            case .moduleInput: return "moduleInput:\(name ?? "")"
            case .moduleOutput: return "moduleOutput:\(name ?? "")"
            case .graphValue: return "graphValue:\(name ?? "")"
            case .nodePort: return "nodePort:\(node ?? "").\(port ?? "")"
            case .importedPort:
                return "importedPort:\(importAlias ?? "").\(export ?? "").\(port ?? "")"
            }
        }

        fileprivate func validateShape() throws {
            switch kind {
            case .moduleInput, .moduleOutput, .graphValue:
                guard name?.isEmpty == false, node == nil, port == nil,
                      importAlias == nil, export == nil else {
                    throw SmeltCAMIRError.malformed("malformed endpoint '\(sortKey)'")
                }
            case .nodePort:
                guard node?.isEmpty == false, port?.isEmpty == false,
                      name == nil, importAlias == nil, export == nil else {
                    throw SmeltCAMIRError.malformed("malformed endpoint '\(sortKey)'")
                }
            case .importedPort:
                guard importAlias?.isEmpty == false, export?.isEmpty == false,
                      port?.isEmpty == false, name == nil, node == nil else {
                    throw SmeltCAMIRError.malformed("malformed endpoint '\(sortKey)'")
                }
            }
        }
    }

    struct GraphEdge: Codable, Sendable, Equatable {
        public let from: EndpointRef
        public let to: EndpointRef
        public let type: TypeRef?

        public init(from: EndpointRef, to: EndpointRef, type: TypeRef? = nil) {
            self.from = from
            self.to = to
            self.type = type
        }

        fileprivate var sortKey: String { "\(from.sortKey)->\(to.sortKey)" }
    }

    struct FeedbackEdge: Codable, Sendable, Equatable {
        public let from: EndpointRef
        public let to: EndpointRef

        public init(from: EndpointRef, to: EndpointRef) {
            self.from = from
            self.to = to
        }

        fileprivate var sortKey: String { "\(from.sortKey)->\(to.sortKey)" }
    }

    enum FlowPhaseRole: String, Codable, Sendable {
        case setup
        case step
    }

    enum FlowCallKind: String, Codable, Sendable {
        case node
        case imported
    }

    struct FlowCall: Codable, Sendable, Equatable {
        public let kind: FlowCallKind
        public let node: String?
        public let imported: ImportedExportRef?
        public let entrypoint: String?

        public init(
            kind: FlowCallKind,
            node: String? = nil,
            imported: ImportedExportRef? = nil,
            entrypoint: String? = nil
        ) {
            self.kind = kind
            self.node = node
            self.imported = imported
            self.entrypoint = entrypoint
        }

        public static func node(_ id: String, entrypoint: String? = nil) -> FlowCall {
            FlowCall(kind: .node, node: id, entrypoint: entrypoint)
        }

        public static func imported(
            alias: String,
            export: String,
            entrypoint: String? = nil
        ) -> FlowCall {
            FlowCall(
                kind: .imported,
                imported: ImportedExportRef(alias: alias, export: export),
                entrypoint: entrypoint
            )
        }
    }

    struct Flow: Codable, Sendable, Equatable {
        public let id: String
        public let phases: [FlowPhase]
        public let emit: [EndpointRef]
        public let stop: [StopCondition]

        public init(
            id: String,
            phases: [FlowPhase],
            emit: [EndpointRef],
            stop: [StopCondition]
        ) {
            self.id = id
            self.phases = phases
            self.emit = emit
            self.stop = stop
        }

        fileprivate var sortKey: String { id }

        fileprivate func canonicalized() -> Flow {
            Flow(
                id: id,
                phases: phases,
                emit: emit.sorted { $0.sortKey < $1.sortKey },
                stop: stop.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }

    struct FlowPhase: Codable, Sendable, Equatable {
        public let role: FlowPhaseRole
        public let label: String?
        public let calls: [FlowCall]

        public init(role: FlowPhaseRole, label: String? = nil, calls: [FlowCall]) {
            self.role = role
            self.label = label
            self.calls = calls
        }
    }

    enum StopKind: String, Codable, Sendable {
        case eosToken = "eos-token"
        case maxSteps = "max-steps"
        case maxFrames = "max-frames"
        case hostCancel = "host-cancel"
        case codecEOS = "codec-eos"
    }

    struct StopCondition: Codable, Sendable, Equatable {
        public let kind: StopKind
        public let value: Int?

        public init(kind: StopKind, value: Int? = nil) {
            self.kind = kind
            self.value = value
        }

        fileprivate var sortKey: String { "\(kind.rawValue)=\(value ?? -1)" }
    }

    struct Constraint: Codable, Sendable, Equatable {
        public let key: String
        public let value: String

        public init(_ key: String, _ value: String) {
            self.key = key
            self.value = value
        }

        fileprivate var sortKey: String { "\(key)=\(value)" }
    }

    struct TensorSelector: Codable, Sendable, Equatable {
        public let source: String?
        public let pattern: String

        public init(_ pattern: String, source: String? = nil) {
            self.source = source
            self.pattern = pattern
        }

        fileprivate var sortKey: String { "\(source ?? ""):\(pattern)" }

        fileprivate func matches(_ other: TensorSelector) -> Bool {
            if let source, let otherSource = other.source, source != otherSource {
                return false
            }
            return glob(pattern, matches: other.pattern) || glob(other.pattern, matches: pattern)
        }

        fileprivate func matches(_ target: TensorTarget) -> Bool {
            glob(pattern, matches: target.selector)
        }

        fileprivate func targets(_ other: TensorSelector) -> Bool {
            if let source, let otherSource = other.source, source != otherSource {
                return false
            }
            return glob(pattern, matches: other.pattern)
        }

        fileprivate func targets(_ target: TensorTarget) -> Bool {
            glob(pattern, matches: target.selector)
        }

        private func glob(_ pattern: String, matches value: String) -> Bool {
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
    }

    struct TensorTarget: Codable, Sendable, Equatable {
        public let block: String
        public let selector: String

        public init(block: String, selector: String) {
            self.block = block
            self.selector = selector
        }

        fileprivate var sortKey: String { "\(block).\(selector)" }
    }

    struct TensorMap: Codable, Sendable, Equatable {
        public let source: String
        public let selector: TensorSelector
        public let target: TensorTarget
        public let owner: String

        public init(source: String, selector: TensorSelector, target: TensorTarget, owner: String) {
            self.source = source
            self.selector = selector
            self.target = target
            self.owner = owner
        }

        fileprivate var sortKey: String { "\(target.sortKey)|\(source)|\(selector.sortKey)" }
    }

    enum QuantAction: String, Codable, Sendable {
        case `default`
        case preserve
        case store
        case quantize
    }

    enum QuantResolution: String, Codable, Sendable {
        case declaredTensor = "declared-tensor"
        case sourceDeferred = "source-deferred"
    }

    enum QuantStorageFormat: String, Codable, Sendable {
        case affineU4 = "affine-u4"
        case lutU4 = "lut-u4"
        case binary1 = "binary-1"
        case ternary2 = "ternary-2"
        case gptq
        case turboQuantH = "turbo-quant-h"
        case bf16
        case fp16
    }

    struct QuantStorage: Codable, Sendable, Equatable {
        public let format: QuantStorageFormat
        public let groupSize: Int?
        public let computeDType: String?

        public init(
            format: QuantStorageFormat,
            groupSize: Int? = nil,
            computeDType: String? = nil
        ) {
            self.format = format
            self.groupSize = groupSize
            self.computeDType = computeDType
        }

        fileprivate var sortKey: String {
            "\(format.rawValue)|\(groupSize ?? -1)|\(computeDType ?? "")"
        }
    }

    enum QuantMethod: String, Codable, Sendable {
        case gptq
        case imatrix
    }

    struct QuantCalibrationCorpus: Codable, Sendable, Equatable {
        public let source: String
        public let path: String?
        public let maxTokens: Int?

        public init(source: String, path: String? = nil, maxTokens: Int? = nil) {
            self.source = source
            self.path = path
            self.maxTokens = maxTokens
        }
    }

    struct QuantCalibration: Codable, Sendable, Equatable {
        public let method: QuantMethod
        public let corpus: QuantCalibrationCorpus
        public let captures: [String]
        public let layersPerPass: Int?
        public let requirements: [Comparison]

        public init(
            method: QuantMethod,
            corpus: QuantCalibrationCorpus,
            captures: [String],
            layersPerPass: Int? = nil,
            requirements: [Comparison] = []
        ) {
            self.method = method
            self.corpus = corpus
            self.captures = captures
            self.layersPerPass = layersPerPass
            self.requirements = requirements
        }

        fileprivate func canonicalized() -> QuantCalibration {
            QuantCalibration(
                method: method,
                corpus: corpus,
                captures: captures.sorted(),
                layersPerPass: layersPerPass,
                requirements: requirements.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }

    struct QuantRule: Codable, Sendable, Equatable {
        public let selector: TensorSelector
        public let action: QuantAction
        public let storage: QuantStorage?
        public let source: String?
        public let priority: Int?
        public let calibration: QuantCalibration?
        public let resolution: QuantResolution

        public init(
            selector: TensorSelector,
            action: QuantAction,
            storage: QuantStorage? = nil,
            source: String? = nil,
            priority: Int? = nil,
            calibration: QuantCalibration? = nil,
            resolution: QuantResolution = .declaredTensor
        ) {
            self.selector = selector
            self.action = action
            self.storage = storage
            self.source = source
            self.priority = priority
            self.calibration = calibration
            self.resolution = resolution
        }

        fileprivate var sortKey: String {
            "\(priority ?? Int.max)|\(selector.sortKey)|\(action.rawValue)|"
                + "\(storage?.sortKey ?? "")|\(source ?? "")|\(resolution.rawValue)"
        }

        fileprivate var resolvedStorageKey: String {
            "\(action.rawValue)|\(storage?.sortKey ?? "")"
        }

        fileprivate func canonicalized() -> QuantRule {
            QuantRule(
                selector: selector,
                action: action,
                storage: storage,
                source: source,
                priority: priority,
                calibration: calibration?.canonicalized(),
                resolution: resolution
            )
        }

        fileprivate func overlaps(_ other: QuantRule) -> Bool {
            selector.matches(other.selector)
        }
    }

    enum SourceQuantizationAction: String, Codable, Sendable {
        case preserve
        case dequant
    }

    struct SourceQuantizationRule: Codable, Sendable, Equatable {
        public let source: String
        public let sourceDTypes: [String]
        public let action: SourceQuantizationAction
        public let storage: QuantStorage?
        public let targetDType: String?
        public let evidence: [EvidenceRequirement]

        public init(
            source: String,
            sourceDTypes: [String],
            action: SourceQuantizationAction,
            storage: QuantStorage? = nil,
            targetDType: String? = nil,
            evidence: [EvidenceRequirement] = []
        ) {
            self.source = source
            self.sourceDTypes = sourceDTypes
            self.action = action
            self.storage = storage
            self.targetDType = targetDType
            self.evidence = evidence
        }

        fileprivate var sortKey: String {
            "\(source)|\(action.rawValue)|\(sourceDTypes.sorted().joined(separator: ","))"
        }

        fileprivate func canonicalized() -> SourceQuantizationRule {
            SourceQuantizationRule(
                source: source,
                sourceDTypes: sourceDTypes.sorted(),
                action: action,
                storage: storage,
                targetDType: targetDType,
                evidence: evidence.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }

    struct ArtifactRole: Codable, Sendable, Equatable {
        public let id: String
        public let role: String
        public let required: Bool

        public init(id: String, role: String, required: Bool = true) {
            self.id = id
            self.role = role
            self.required = required
        }

        fileprivate var sortKey: String { id }
    }

    enum GateEventKind: String, Codable, Sendable {
        case flowAccepted = "flow.accepted"
        case emit
        case input
    }

    struct GateEvent: Codable, Sendable, Equatable {
        public let kind: GateEventKind
        public let flow: String?
        public let export: String?
        public let endpoint: EndpointRef?
        public let signal: String?
        public let predicates: [Comparison]

        public init(
            kind: GateEventKind,
            flow: String? = nil,
            export: String? = nil,
            endpoint: EndpointRef? = nil,
            signal: String? = nil,
            predicates: [Comparison] = []
        ) {
            self.kind = kind
            self.flow = flow
            self.export = export
            self.endpoint = endpoint
            self.signal = signal
            self.predicates = predicates
        }

        fileprivate func canonicalized() -> GateEvent {
            GateEvent(
                kind: kind,
                flow: flow,
                export: export,
                endpoint: endpoint,
                signal: signal,
                predicates: predicates.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }

    enum ComparisonRelation: String, Codable, Sendable {
        case lessThanOrEqual = "<="
        case greaterThanOrEqual = ">="
        case equal = "=="
        case include
    }

    struct Comparison: Codable, Sendable, Equatable {
        public let subject: String
        public let relation: ComparisonRelation
        public let value: String
        public let unit: String?

        public init(
            subject: String,
            relation: ComparisonRelation,
            value: String,
            unit: String? = nil
        ) {
            self.subject = subject
            self.relation = relation
            self.value = value
            self.unit = unit
        }

        fileprivate var sortKey: String {
            "\(subject)|\(relation.rawValue)|\(value)|\(unit ?? "")"
        }

        fileprivate func validate(label: String) throws {
            guard !subject.isEmpty, !value.isEmpty else {
                throw SmeltCAMIRError.malformed("\(label) is empty")
            }
        }
    }

    enum EvidenceRequirementKind: String, Codable, Sendable {
        case tensorStoredAs = "tensor-stored-as"
        case sourceSHA256Recorded = "source-sha256-recorded"
        case dequantIdentity = "dequant-identity"
    }

    struct EvidenceRequirement: Codable, Sendable, Equatable {
        public let kind: EvidenceRequirementKind
        public let tensor: String?
        public let source: String?
        public let sourceDTypes: [String]
        public let storage: QuantStorage?

        public init(
            kind: EvidenceRequirementKind,
            tensor: String? = nil,
            source: String? = nil,
            sourceDTypes: [String] = [],
            storage: QuantStorage? = nil
        ) {
            self.kind = kind
            self.tensor = tensor
            self.source = source
            self.sourceDTypes = sourceDTypes
            self.storage = storage
        }

        fileprivate var sortKey: String {
            [
                kind.rawValue,
                tensor ?? "",
                source ?? "",
                sourceDTypes.sorted().joined(separator: ","),
                storage?.sortKey ?? "",
            ].joined(separator: "|")
        }

        fileprivate func canonicalized() -> EvidenceRequirement {
            EvidenceRequirement(
                kind: kind,
                tensor: tensor,
                source: source,
                sourceDTypes: sourceDTypes.sorted(),
                storage: storage
            )
        }

        fileprivate func validate(label: String) throws {
            switch kind {
            case .tensorStoredAs:
                guard tensor?.isEmpty == false, storage != nil,
                      source == nil, sourceDTypes.isEmpty else {
                    throw SmeltCAMIRError.malformed("\(label) tensor storage evidence is malformed")
                }
            case .sourceSHA256Recorded:
                guard source?.isEmpty == false, tensor == nil,
                      sourceDTypes.isEmpty, storage == nil else {
                    throw SmeltCAMIRError.malformed("\(label) source hash evidence is malformed")
                }
            case .dequantIdentity:
                guard source?.isEmpty == false, !sourceDTypes.isEmpty,
                      tensor == nil, storage == nil else {
                    throw SmeltCAMIRError.malformed("\(label) dequant identity evidence is malformed")
                }
            }
        }
    }

    enum GateMeasurementProcessMode: String, Codable, Sendable {
        case cold
        case warm
    }

    enum GateMeasurementCacheState: String, Codable, Sendable {
        case cold
        case warm
        case hostGlobalColdCandidate = "host-global-cold-candidate"
        case unknown
    }

    enum GateMeasurementOccurrence: String, Codable, Sendable {
        case first
        case median
        case p95
        case all
    }

    struct GateMeasurement: Codable, Sendable, Equatable {
        public let subject: String
        public let processMode: GateMeasurementProcessMode
        public let cacheState: GateMeasurementCacheState
        public let occurrence: GateMeasurementOccurrence

        public init(
            subject: String,
            processMode: GateMeasurementProcessMode,
            cacheState: GateMeasurementCacheState,
            occurrence: GateMeasurementOccurrence
        ) {
            self.subject = subject
            self.processMode = processMode
            self.cacheState = cacheState
            self.occurrence = occurrence
        }

        fileprivate var sortKey: String {
            [
                subject,
                processMode.rawValue,
                cacheState.rawValue,
                occurrence.rawValue,
            ].joined(separator: "|")
        }

        fileprivate func validate(label: String) throws {
            guard !subject.isEmpty else {
                throw SmeltCAMIRError.malformed("\(label) subject is empty")
            }
        }
    }

    struct Gate: Codable, Sendable, Equatable {
        public let id: String
        public let from: GateEvent?
        public let to: GateEvent?
        public let requirements: [Comparison]
        public let measurements: [GateMeasurement]
        public let evidence: [EvidenceRequirement]

        public init(
            id: String,
            from: GateEvent? = nil,
            to: GateEvent? = nil,
            requirements: [Comparison],
            measurements: [GateMeasurement] = [],
            evidence: [EvidenceRequirement] = []
        ) {
            self.id = id
            self.from = from
            self.to = to
            self.requirements = requirements
            self.measurements = measurements
            self.evidence = evidence
        }

        fileprivate var sortKey: String { id }

        fileprivate func canonicalized() -> Gate {
            Gate(
                id: id,
                from: from?.canonicalized(),
                to: to?.canonicalized(),
                requirements: requirements.sorted { $0.sortKey < $1.sortKey },
                measurements: measurements.sorted { $0.sortKey < $1.sortKey },
                evidence: evidence.map { $0.canonicalized() }.sorted { $0.sortKey < $1.sortKey }
            )
        }
    }
}

public struct SmeltCAMIRDiagnostics: Codable, Sendable, Equatable {
    public struct SourceSpan: Codable, Sendable, Equatable {
        public let file: String
        public let line: Int
        public let column: Int

        public init(file: String, line: Int, column: Int) {
            self.file = file
            self.line = line
            self.column = column
        }
    }

    public let spans: [String: SourceSpan]

    public init(spans: [String: SourceSpan]) {
        self.spans = spans
    }
}
