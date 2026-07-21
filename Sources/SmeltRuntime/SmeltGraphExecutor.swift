import CryptoKit
import Foundation
import SmeltSchema

final class SmeltGraphExecutor {
    private struct StageReceipt: Encodable {
        let schema: String
        let stages: [SmeltGraphExecutionContext.StageTiming]
        let evidence: [String: String]
        let outputSHA256: String
    }

    private let packagePath: String
    private let graph: SmeltCAMPackageCapabilities.ExecutionGraph
    private let registry: SmeltGraphNodeRegistry

    init(
        packagePath: String,
        graph: SmeltCAMPackageCapabilities.ExecutionGraph,
        registry: SmeltGraphNodeRegistry
    ) {
        self.packagePath = packagePath
        self.graph = graph
        self.registry = registry
    }

    func run(
        inputURL: URL,
        outputURL: URL,
        options: [String: String]
    ) throws -> SmeltPackageRunResult {
        let context = SmeltGraphExecutionContext(
            packagePath: packagePath,
            inputURL: inputURL,
            outputURL: outputURL,
            options: options
        )
        guard let selectedInput = graph.selectedInputPorts.first else {
            throw SmeltGraphExecutorError.missingSelectedInput
        }
        var values: [String: SmeltGraphValue] = [
            moduleInputKey(selectedInput.portName): .init(inputURL),
        ]
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.nodeID, $0) })
        for phase in graph.phases {
            for call in phase.calls {
                guard call.callType == "node",
                      let nodeID = call.nodeID,
                      let entrypoint = call.entrypoint,
                      let node = nodes[nodeID]
                else {
                    throw SmeltGraphExecutorError.unsupportedCall
                }
                var inputs: [String: SmeltGraphValue] = [:]
                for edge in graph.dataEdges where
                    edge.to.endpointType == "nodePort" && edge.to.nodeID == nodeID
                {
                    guard let port = edge.to.portName else {
                        throw SmeltGraphExecutorError.malformedEndpoint
                    }
                    inputs[port] = try resolve(edge.from, values: values)
                }
                for port in node.inputs where !port.optional && inputs[port.portName] == nil {
                    throw SmeltGraphValueError.missing(node: nodeID, port: port.portName)
                }
                let started = ProcessInfo.processInfo.systemUptime
                let outputs = try registry.make(entrypoint: entrypoint).run(
                    inputs: inputs,
                    context: context
                )
                context.appendTiming(
                    node: nodeID,
                    entrypoint: entrypoint,
                    wallMilliseconds: (ProcessInfo.processInfo.systemUptime - started) * 1_000
                )
                for port in node.outputs where !port.optional && outputs[port.portName] == nil {
                    throw SmeltGraphValueError.missing(node: nodeID, port: port.portName)
                }
                for (port, value) in outputs {
                    values[nodePortKey(nodeID, port)] = value
                    for edge in graph.dataEdges where
                        edge.from.endpointType == "nodePort"
                            && edge.from.nodeID == nodeID
                            && edge.from.portName == port
                            && edge.to.endpointType == "graphValue"
                    {
                        guard let name = edge.to.name else {
                            throw SmeltGraphExecutorError.malformedEndpoint
                        }
                        values[graphValueKey(name)] = value
                    }
                }
            }
        }
        guard graph.emissions.contains(where: { endpoint in
            guard let value = try? resolve(endpoint, values: values),
                  let emittedURL = value.cast(URL.self)
            else { return false }
            return emittedURL.standardizedFileURL == outputURL.standardizedFileURL
        }) else {
            throw SmeltGraphExecutorError.outputNotEmitted
        }
        try writeReceiptIfRequested(context: context, outputURL: outputURL)
        return SmeltPackageRunResult(
            outputURL: outputURL,
            summary: context.summary.isEmpty ? "Wrote \(outputURL.path)" : context.summary
        )
    }

    private func resolve(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef,
        values: [String: SmeltGraphValue]
    ) throws -> SmeltGraphValue {
        let key: String
        switch endpoint.endpointType {
        case "moduleInput":
            guard let name = endpoint.name else {
                throw SmeltGraphExecutorError.malformedEndpoint
            }
            key = moduleInputKey(name)
        case "nodePort":
            guard let node = endpoint.nodeID, let port = endpoint.portName else {
                throw SmeltGraphExecutorError.malformedEndpoint
            }
            key = nodePortKey(node, port)
        case "graphValue":
            guard let name = endpoint.name else {
                throw SmeltGraphExecutorError.malformedEndpoint
            }
            key = graphValueKey(name)
        default:
            throw SmeltGraphExecutorError.unsupportedEndpoint(endpoint.endpointType)
        }
        guard let value = values[key] else {
            throw SmeltGraphExecutorError.unresolvedValue(key)
        }
        return value
    }

    private func writeReceiptIfRequested(
        context: SmeltGraphExecutionContext,
        outputURL: URL
    ) throws {
        guard let path = ProcessInfo.processInfo.environment["SMELT_STAGE_RECEIPT"],
              !path.isEmpty
        else { return }
        let data = try Data(contentsOf: outputURL)
        let receipt = StageReceipt(
            schema: "smelt.graph.stage-receipt.v1",
            stages: context.timings,
            evidence: context.evidence,
            outputSHA256: SHA256.hash(data: data).map {
                String(format: "%02x", $0)
            }.joined()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(
            to: URL(fileURLWithPath: path),
            options: .atomic
        )
    }

    private func moduleInputKey(_ name: String) -> String { "module-input:\(name)" }
    private func nodePortKey(_ node: String, _ port: String) -> String { "node:\(node):\(port)" }
    private func graphValueKey(_ name: String) -> String { "value:\(name)" }
}

enum SmeltGraphExecutorError: Error, Equatable {
    case unsupportedCall
    case missingSelectedInput
    case malformedEndpoint
    case unsupportedEndpoint(String)
    case unresolvedValue(String)
    case outputNotEmitted
}
