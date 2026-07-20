import Foundation
import SmeltSchema

package enum CAMRuntimeRoute: Sendable, Equatable {
    case textToText
    case textToPCM(outputRate: String)

    package var textToPCMOutputRate: String? {
        guard case let .textToPCM(outputRate) = self else { return nil }
        return outputRate
    }
}

package enum CAMRuntimeRouteResolver {
    package static func resolve(
        decision: SmeltCAMPackageCapabilities.Decision,
        capabilities: SmeltCAMPackageCapabilities
    ) -> CAMRuntimeRoute? {
        guard hasCurrentTarget(capabilities),
              hasCurrentQuantPipeline(capabilities),
              let flow = capabilities.flows.first(where: { $0.flowID == decision.flowID }),
              let calledNodes = localCalledNodes(in: flow, capabilities: capabilities),
              calledNodes.contains(where: { $0.implementation == "compiled" }),
              flowEmitsSelectedOutput(flow, decision: decision, capabilities: capabilities)
        else {
            return nil
        }

        let textIn = selectedInputs(decision, contain: .textUTF8)
        let textOut = selectedOutputs(decision, contain: .textUTF8)
        let pcmOutputRate = selectedPCMRate(in: decision.selectedOutputPorts)

        if textIn && textOut {
            return .textToText
        }
        if textIn,
           let outputRate = pcmOutputRate,
           selectedOutputsAreCompiledEmitters(
                flow,
                decision: decision,
                calledNodes: calledNodes,
                capabilities: capabilities
        ) {
            return .textToPCM(outputRate: outputRate)
        }
        return nil
    }

    private static func hasCurrentTarget(_ capabilities: SmeltCAMPackageCapabilities) -> Bool {
        let requirements = capabilities.backendRequirements + capabilities.compileRequirements
        let targets = requirements.filter { $0.key == "target" }
        return targets.isEmpty || targets.allSatisfy { $0.value == "metal" }
    }

    private static func hasCurrentQuantPipeline(_ capabilities: SmeltCAMPackageCapabilities) -> Bool {
        guard capabilities.sourceQuantization.isEmpty else { return false }
        return capabilities.quantization.allSatisfy { rule in
            guard rule.calibration == nil else { return false }
            guard let storage = rule.storage else { return true }
            switch storage.storageFormat {
            case "affine-u4", "lut-u4", "binary-1", "ternary-2":
                return (storage.groupSize ?? 0) > 0 && storage.computeDType == nil
            case "fp16":
                return storage.groupSize == nil && storage.computeDType == nil
            default:
                return false
            }
        }
    }

    private static func localCalledNodes(
        in flow: SmeltCAMPackageCapabilities.FlowRecord,
        capabilities: SmeltCAMPackageCapabilities
    ) -> [SmeltCAMPackageDescriptor.GraphNode]? {
        let nodesByID = Dictionary(uniqueKeysWithValues: capabilities.graphNodes.map {
            ($0.nodeID, $0)
        })
        var nodes: [SmeltCAMPackageDescriptor.GraphNode] = []
        for phase in flow.phases {
            for call in phase.calls {
                guard call.callType == "node",
                      let nodeID = call.nodeID,
                      let node = nodesByID[nodeID]
                else {
                    return nil
                }
                nodes.append(node)
            }
        }
        return nodes
    }

    private static func flowEmitsSelectedOutput(
        _ flow: SmeltCAMPackageCapabilities.FlowRecord,
        decision: SmeltCAMPackageCapabilities.Decision,
        capabilities: SmeltCAMPackageCapabilities
    ) -> Bool {
        guard !decision.selectedOutputPorts.isEmpty else { return false }
        return decision.selectedOutputPorts.allSatisfy { selected in
            flow.emittedEndpoints.contains { endpoint in
                endpointDeliversSelectedOutput(endpoint, selected: selected, capabilities: capabilities)
            }
        }
    }

    private static func endpointDeliversSelectedOutput(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        selected: SmeltCAMPackageDescriptor.Port,
        capabilities: SmeltCAMPackageCapabilities
    ) -> Bool {
        switch endpoint.endpointType {
        case "moduleOutput":
            guard let name = endpoint.name else { return false }
            return name == selected.portName
        case "nodePort":
            return capabilities.graphEdges.contains { edge in
                sameEndpoint(edge.from, endpoint)
                    && edge.to.endpointType == "moduleOutput"
                    && edge.to.name == selected.portName
                    && (edge.valueType.map { sameType($0, selected.type) } ?? true)
            }
        default:
            return false
        }
    }

    private static func selectedOutputsAreCompiledEmitters(
        _ flow: SmeltCAMPackageCapabilities.FlowRecord,
        decision: SmeltCAMPackageCapabilities.Decision,
        calledNodes: [SmeltCAMPackageDescriptor.GraphNode],
        capabilities: SmeltCAMPackageCapabilities
    ) -> Bool {
        guard !decision.selectedOutputPorts.isEmpty else { return false }
        var calledNodesByID: [String: SmeltCAMPackageDescriptor.GraphNode] = [:]
        for node in calledNodes {
            calledNodesByID[node.nodeID] = node
        }
        return decision.selectedOutputPorts.allSatisfy { selected in
            flow.emittedEndpoints.contains { endpoint in
                endpointDeliversSelectedOutput(
                    endpoint,
                    selected: selected,
                    capabilities: capabilities
                )
                    && emittedEndpointIsBackedByCompiledOutput(
                        endpoint,
                        selected: selected,
                        calledNodesByID: calledNodesByID,
                        capabilities: capabilities
                    )
            }
        }
    }

    private static func emittedEndpointIsBackedByCompiledOutput(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        selected: SmeltCAMPackageDescriptor.Port,
        calledNodesByID: [String: SmeltCAMPackageDescriptor.GraphNode],
        capabilities: SmeltCAMPackageCapabilities
    ) -> Bool {
        switch endpoint.endpointType {
        case "nodePort":
            return compiledOutputEndpoint(
                endpoint,
                hasType: selected.type,
                calledNodesByID: calledNodesByID
            )
        case "moduleOutput":
            guard endpoint.name == selected.portName else { return false }
            return capabilities.graphEdges.contains { edge in
                sameEndpoint(edge.to, endpoint)
                    && (edge.valueType.map { sameType($0, selected.type) } ?? true)
                    && compiledOutputEndpoint(
                        edge.from,
                        hasType: selected.type,
                        calledNodesByID: calledNodesByID
                    )
            }
        default:
            return false
        }
    }

    private static func compiledOutputEndpoint(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        hasType selectedType: SmeltCAMPackageDescriptor.ValueType,
        calledNodesByID: [String: SmeltCAMPackageDescriptor.GraphNode]
    ) -> Bool {
        guard endpoint.endpointType == "nodePort",
              let nodeID = endpoint.nodeID,
              let portName = endpoint.portName,
              let node = calledNodesByID[nodeID],
              node.implementation == "compiled",
              let output = node.outputs.first(where: { $0.portName == portName })
        else {
            return false
        }
        return sameType(output.type, selectedType)
    }

    private static func selectedInputs(
        _ decision: SmeltCAMPackageCapabilities.Decision,
        contain shape: SmeltCAMCapabilityRequest.PortShape
    ) -> Bool {
        decision.selectedInputPorts.contains { shape.matches($0) }
    }

    private static func selectedOutputs(
        _ decision: SmeltCAMPackageCapabilities.Decision,
        contain shape: SmeltCAMCapabilityRequest.PortShape
    ) -> Bool {
        decision.selectedOutputPorts.contains { shape.matches($0) }
    }

    private static func selectedPCMRate(
        in ports: [SmeltCAMPackageDescriptor.Port]
    ) -> String? {
        let rates = Set(ports.compactMap { port -> String? in
            guard port.type.typeName == "pcm",
                  port.type.attributes["dtype"] == "f32"
            else { return nil }
            return port.type.attributes["rate"]
        })
        guard rates.count == 1 else { return nil }
        return rates.first
    }

    private static func sameType(
        _ lhs: SmeltCAMPackageDescriptor.ValueType,
        _ rhs: SmeltCAMPackageDescriptor.ValueType
    ) -> Bool {
        lhs.typeName == rhs.typeName
            && lhs.attributes == rhs.attributes
    }

    private static func sameEndpoint(
        _ lhs: SmeltCAMPackageDescriptor.EndpointRef,
        _ rhs: SmeltCAMPackageDescriptor.EndpointRef
    ) -> Bool {
        lhs.endpointType == rhs.endpointType
            && lhs.name == rhs.name
            && lhs.nodeID == rhs.nodeID
            && lhs.portName == rhs.portName
            && lhs.importAlias == rhs.importAlias
            && lhs.exportID == rhs.exportID
    }
}
