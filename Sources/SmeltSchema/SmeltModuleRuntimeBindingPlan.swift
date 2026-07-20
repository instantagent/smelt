import Foundation

/// The model-independent result of binding a selected module export to
/// runtime brick kinds. Concrete backends register against `bindingKey`;
/// orchestration consumes phase roles and edges without switching on a model
/// id, output sample rate, or semantic hash.
public struct SmeltModuleRuntimeBindingPlan: Sendable, Equatable {
    public enum ExecutionShape: String, Sendable, Equatable {
        case iterative
        case iterativeWithFinalization = "iterative-with-finalization"
    }

    public struct Invocation: Sendable, Equatable {
        public let nodeID: String
        public let implementation: String
        public let blockID: String?
        public let bindingKey: String

        public init(
            nodeID: String,
            implementation: String,
            blockID: String?,
            bindingKey: String
        ) {
            self.nodeID = nodeID
            self.implementation = implementation
            self.blockID = blockID
            self.bindingKey = bindingKey
        }
    }

    public struct Phase: Sendable, Equatable {
        public let role: String
        public let label: String?
        public let invocations: [Invocation]

        public init(role: String, label: String?, invocations: [Invocation]) {
            self.role = role
            self.label = label
            self.invocations = invocations
        }
    }

    public struct Edge: Sendable, Equatable {
        public let fromNodeID: String
        public let fromPort: String
        public let toNodeID: String
        public let toPort: String

        public init(fromNodeID: String, fromPort: String, toNodeID: String, toPort: String) {
            self.fromNodeID = fromNodeID
            self.fromPort = fromPort
            self.toNodeID = toNodeID
            self.toPort = toPort
        }
    }

    public struct RegionConnection: Sendable, Equatable, Hashable {
        public let from: String
        public let to: String

        public init(from: String, to: String) {
            self.from = from
            self.to = to
        }
    }

    /// Node ids and module hashes are deliberately absent. Phase/call
    /// addresses identify scheduled nodes; emission addresses identify
    /// output-only adapters. This makes optimized-region admission exact on
    /// wiring while remaining independent of model naming.
    public struct RegionContract: Sendable, Equatable {
        public let phaseRoles: [String]
        public let phaseBindingKeys: [[String]]
        public let nodeBindings: [String]
        public let dataConnections: [RegionConnection]
        public let feedbackConnections: [RegionConnection]
        public let emissions: [String]
        public let stopConditions: [String]

        public init(
            phaseRoles: [String],
            phaseBindingKeys: [[String]],
            nodeBindings: [String],
            dataConnections: [RegionConnection],
            feedbackConnections: [RegionConnection],
            emissions: [String],
            stopConditions: [String]
        ) {
            self.phaseRoles = phaseRoles
            self.phaseBindingKeys = phaseBindingKeys
            self.nodeBindings = nodeBindings
            self.dataConnections = dataConnections
            self.feedbackConnections = feedbackConnections
            self.emissions = emissions
            self.stopConditions = stopConditions
        }
    }

    public let exportID: String
    public let flowID: String
    public let graph: SmeltCAMPackageCapabilities.ExecutionGraph
    public let executionShape: ExecutionShape
    public let nodes: [Invocation]
    public let phases: [Phase]
    public let feedbackEdges: [Edge]
    public let emittedNodeIDs: [String]
    public let regionContract: RegionContract

    public init(
        capabilities: SmeltCAMPackageCapabilities,
        decision: SmeltCAMPackageCapabilities.Decision
    ) throws {
        let graph = try capabilities.executionGraph(for: decision)
        let nodesByID = Dictionary(
            uniqueKeysWithValues: graph.nodes.map { ($0.nodeID, $0) }
        )
        let blocksByID = Dictionary(
            uniqueKeysWithValues: graph.blocks.map { ($0.blockID, $0) }
        )
        let boundNodes = try graph.nodes.map {
            try Self.bind(node: $0, blocksByID: blocksByID)
        }
        let bindingsByNodeID = Dictionary(
            uniqueKeysWithValues: boundNodes.map { ($0.nodeID, $0) }
        )

        var boundPhases: [Phase] = []
        var calledNodeIDs = Set<String>()
        for phase in graph.phases {
            guard phase.phaseType == "setup"
                    || phase.phaseType == "step"
                    || phase.phaseType == "finalize"
            else {
                throw SmeltModuleRuntimeBindingError.unsupported(
                    "flow phase role '\(phase.phaseType)' is unsupported"
                )
            }
            let invocations = try phase.calls.map { call -> Invocation in
                guard call.callType == "node",
                      let nodeID = call.nodeID,
                      call.imported == nil,
                      call.entrypoint == nil
                else {
                    throw SmeltModuleRuntimeBindingError.unsupported(
                        "only local node calls bind to runtime bricks"
                    )
                }
                guard nodesByID[nodeID] != nil, let invocation = bindingsByNodeID[nodeID] else {
                    throw SmeltModuleRuntimeBindingError.malformed(
                        "flow calls missing node '\(nodeID)'"
                    )
                }
                calledNodeIDs.insert(nodeID)
                return invocation
            }
            boundPhases.append(
                Phase(role: phase.phaseType, label: phase.label, invocations: invocations)
            )
        }

        let finalizePhases = boundPhases.filter { $0.role == "finalize" }
        if finalizePhases.isEmpty {
            executionShape = .iterative
        } else {
            guard finalizePhases.count == 1,
                  finalizePhases.first?.invocations.isEmpty == false
            else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "final emission needs exactly one non-empty finalize phase"
                )
            }
            executionShape = .iterativeWithFinalization
        }

        feedbackEdges = try graph.feedbackEdges.map { edge in
            guard let fromNodeID = edge.from.nodeID,
                  let fromPort = edge.from.portName,
                  let toNodeID = edge.to.nodeID,
                  let toPort = edge.to.portName,
                  bindingsByNodeID[fromNodeID] != nil,
                  bindingsByNodeID[toNodeID] != nil
            else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "feedback edge must connect executable node ports"
                )
            }
            return Edge(
                fromNodeID: fromNodeID,
                fromPort: fromPort,
                toNodeID: toNodeID,
                toPort: toPort
            )
        }

        emittedNodeIDs = try graph.emissions.compactMap { endpoint in
            guard let nodeID = endpoint.nodeID else { return nil }
            guard bindingsByNodeID[nodeID] != nil else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "flow emission must originate at an executable node"
                )
            }
            return nodeID
        }
        guard !graph.emissions.isEmpty else {
            throw SmeltModuleRuntimeBindingError.malformed("selected flow emits nothing")
        }

        exportID = decision.exportID
        flowID = decision.flowID
        self.graph = graph
        nodes = boundNodes
        phases = boundPhases
        regionContract = try Self.makeRegionContract(
            graph: graph,
            nodes: boundNodes,
            phases: boundPhases
        )
    }

    public var bindingKeys: [String] {
        nodes.map(\.bindingKey)
    }

    private static func bind(
        node: SmeltCAMPackageDescriptor.GraphNode,
        blocksByID: [String: SmeltCAMPackageDescriptor.Block]
    ) throws -> Invocation {
        switch node.implementation {
        case "compiled":
            guard let blockID = node.blockID, let block = blocksByID[blockID] else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "compiled node '\(node.nodeID)' has no declared block"
                )
            }
            return Invocation(
                nodeID: node.nodeID,
                implementation: node.implementation,
                blockID: blockID,
                bindingKey: "compiled:\(block.operatorName)"
            )
        case "native":
            let tags = node.annotations.filter { $0.key == "tag" }.map(\.value)
            let tag: String
            if tags.count == 1, let declaredTag = tags.first {
                tag = declaredTag
            } else if tags.isEmpty, let roleBlock = blocksByID[node.nodeID] {
                // Grammar-era native bricks declare their structural operator
                // as a same-id block. Prefer an explicit tag for new modules;
                // preserve this lossless compatibility convention while they
                // migrate.
                tag = roleBlock.operatorName
            } else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "native node '\(node.nodeID)' needs one tag or a same-id role block"
                )
            }
            return Invocation(
                nodeID: node.nodeID,
                implementation: node.implementation,
                blockID: nil,
                bindingKey: "native:\(tag)"
            )
        default:
            throw SmeltModuleRuntimeBindingError.unsupported(
                "node implementation '\(node.implementation)' cannot bind locally"
            )
        }
    }

    private static func makeRegionContract(
        graph: SmeltCAMPackageCapabilities.ExecutionGraph,
        nodes: [Invocation],
        phases: [Phase]
    ) throws -> RegionContract {
        let bindingsByNodeID = Dictionary(
            uniqueKeysWithValues: nodes.map { ($0.nodeID, $0.bindingKey) }
        )
        var addressesByNodeID: [String: String] = [:]
        for (phaseIndex, phase) in phases.enumerated() {
            for (callIndex, invocation) in phase.invocations.enumerated()
            where addressesByNodeID[invocation.nodeID] == nil {
                addressesByNodeID[invocation.nodeID] = "p\(phaseIndex)c\(callIndex)"
            }
        }
        for (emitIndex, endpoint) in graph.emissions.enumerated() {
            if let nodeID = endpoint.nodeID, addressesByNodeID[nodeID] == nil {
                addressesByNodeID[nodeID] = "e\(emitIndex)"
            }
        }
        for (nodeIndex, node) in graph.nodes.enumerated()
        where addressesByNodeID[node.nodeID] == nil {
            addressesByNodeID[node.nodeID] = "n\(nodeIndex)"
        }

        func normalizedEndpoint(_ endpoint: SmeltCAMPackageDescriptor.EndpointRef) throws -> String {
            switch endpoint.endpointType {
            case "nodePort":
                guard let nodeID = endpoint.nodeID,
                      let address = addressesByNodeID[nodeID],
                      let port = endpoint.portName
                else {
                    throw SmeltModuleRuntimeBindingError.malformed(
                        "region endpoint references an unbound node port"
                    )
                }
                return "\(address).\(port)"
            case "moduleInput":
                guard let name = endpoint.name else {
                    throw SmeltModuleRuntimeBindingError.malformed("unnamed module input")
                }
                return "input.\(name)"
            case "moduleOutput":
                guard let name = endpoint.name else {
                    throw SmeltModuleRuntimeBindingError.malformed("unnamed module output")
                }
                return "output.\(name)"
            case "importedPort":
                guard let alias = endpoint.importAlias,
                      let exportID = endpoint.exportID,
                      let port = endpoint.portName
                else {
                    throw SmeltModuleRuntimeBindingError.malformed("incomplete imported port")
                }
                return "import.\(alias).\(exportID).\(port)"
            default:
                throw SmeltModuleRuntimeBindingError.unsupported(
                    "region endpoint '\(endpoint.endpointType)' is not material"
                )
            }
        }

        let graphValueEdges = Dictionary(grouping: graph.dataEdges.filter {
            $0.from.endpointType == "graphValue"
        }) { $0.from.name ?? "" }
        var dataConnections: [RegionConnection] = []
        func appendDestinations(
            source: String,
            endpoint: SmeltCAMPackageDescriptor.EndpointRef,
            visitedGraphValues: Set<String>
        ) throws {
            guard endpoint.endpointType == "graphValue" else {
                dataConnections.append(
                    RegionConnection(from: source, to: try normalizedEndpoint(endpoint))
                )
                return
            }
            guard let name = endpoint.name, !visitedGraphValues.contains(name) else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "data graph contains an unnamed or cyclic graph value"
                )
            }
            var visited = visitedGraphValues
            visited.insert(name)
            for edge in graphValueEdges[name] ?? [] {
                try appendDestinations(source: source, endpoint: edge.to, visitedGraphValues: visited)
            }
        }
        for edge in graph.dataEdges where edge.from.endpointType != "graphValue" {
            try appendDestinations(
                source: normalizedEndpoint(edge.from),
                endpoint: edge.to,
                visitedGraphValues: []
            )
        }

        let feedbackConnections = try graph.feedbackEdges.map {
            RegionConnection(
                from: try normalizedEndpoint($0.from),
                to: try normalizedEndpoint($0.to)
            )
        }
        let emissions = try graph.emissions.map(normalizedEndpoint)
        let nodeBindings = try graph.nodes.map { node -> String in
            guard let address = addressesByNodeID[node.nodeID],
                  let binding = bindingsByNodeID[node.nodeID]
            else {
                throw SmeltModuleRuntimeBindingError.malformed(
                    "execution node '\(node.nodeID)' has no region binding"
                )
            }
            return "\(address)=\(binding)"
        }.sorted()

        return RegionContract(
            phaseRoles: phases.map(\.role),
            phaseBindingKeys: phases.map { $0.invocations.map(\.bindingKey) },
            nodeBindings: nodeBindings,
            dataConnections: dataConnections.sorted(by: Self.connectionOrder),
            feedbackConnections: feedbackConnections.sorted(by: Self.connectionOrder),
            emissions: emissions,
            stopConditions: graph.stopConditions.map {
                "\($0.stopType)\($0.value.map { ":\($0)" } ?? "")"
            }
        )
    }

    private static func connectionOrder(_ lhs: RegionConnection, _ rhs: RegionConnection) -> Bool {
        lhs.from == rhs.from ? lhs.to < rhs.to : lhs.from < rhs.from
    }

    private static func exactlyOne<T>(_ values: [T], _ label: String) throws -> T {
        guard values.count == 1, let value = values.first else {
            throw SmeltModuleRuntimeBindingError.malformed(
                "\(label) expected exactly one, found \(values.count)"
            )
        }
        return value
    }
}
public enum SmeltModuleRuntimeBindingError: Error, CustomStringConvertible, Equatable {
    case malformed(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .malformed(let reason):
            return "module runtime binding: \(reason)"
        case .unsupported(let reason):
            return "module runtime binding unsupported: \(reason)"
        }
    }
}
