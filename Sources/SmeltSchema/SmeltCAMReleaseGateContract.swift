import CryptoKit
import Foundation

public enum SmeltCAMReleaseGateContractError: Error, CustomStringConvertible, Equatable {
    case duplicateRequiredGate(String)
    case missingRequiredGate(String)
    case missingReleaseSurface(String)
    case missingReleaseExport(gateID: String)
    case ambiguousReleaseExport(gateID: String, exports: [String])
    case missingReleaseFlow(exportID: String)
    case missingReleaseOutput(gateID: String, outputName: String)
    case ambiguousReleaseOutput(gateID: String, outputs: [String])
    case missingReleaseRequirement(String)
    case missingReleaseGateContract(String)
    case invalidSurfaceBinding(surfaceID: String, reason: String)

    public var description: String {
        switch self {
        case .duplicateRequiredGate(let gateID):
            return "duplicate CAM release gate '\(gateID)'"
        case .missingRequiredGate(let gateID):
            return "missing CAM release gate '\(gateID)'"
        case .missingReleaseSurface(let gateID):
            return "missing release surface for CAM gate '\(gateID)'"
        case .missingReleaseExport(let gateID):
            return "missing release export for CAM gate '\(gateID)'"
        case .ambiguousReleaseExport(let gateID, let exports):
            return "CAM release gate '\(gateID)' is attached to multiple exports: \(exports.joined(separator: ", "))"
        case .missingReleaseFlow(let exportID):
            return "missing release flow for CAM export '\(exportID)'"
        case .missingReleaseOutput(let gateID, let outputName):
            return "missing release output '\(outputName)' for CAM gate '\(gateID)'"
        case .ambiguousReleaseOutput(let gateID, let outputs):
            return "CAM release gate '\(gateID)' has ambiguous selected outputs: \(outputs.joined(separator: ", "))"
        case .missingReleaseRequirement(let gateID):
            return "missing release requirement for CAM gate '\(gateID)'"
        case .missingReleaseGateContract(let gateID):
            return "missing release gate contract for CAM gate '\(gateID)'"
        case .invalidSurfaceBinding(let surfaceID, let reason):
            return "invalid release surface binding '\(surfaceID)': \(reason)"
        }
    }
}

public struct SmeltCAMReleaseSurfaceBinding: Sendable, Equatable {
    public let surfaceID: String
    public let exportID: String?
    public let flowID: String?
    public let selectedInputNames: [String]
    public let selectedInputs: [String]
    public let selectedOutputNames: [String]
    public let selectedOutputs: [String]
    public let gateIDs: [String]
    public let requiresReleaseEvidence: Bool

    public init(
        surfaceID: String,
        exportID: String?,
        flowID: String?,
        selectedInputNames: [String] = [],
        selectedInputs: [String] = [],
        selectedOutputNames: [String] = [],
        selectedOutputs: [String] = [],
        gateIDs: [String],
        requiresReleaseEvidence: Bool
    ) {
        self.surfaceID = surfaceID
        self.exportID = exportID
        self.flowID = flowID
        self.selectedInputNames = selectedInputNames
        self.selectedInputs = selectedInputs
        self.selectedOutputNames = selectedOutputNames
        self.selectedOutputs = selectedOutputs
        self.gateIDs = gateIDs
        self.requiresReleaseEvidence = requiresReleaseEvidence
    }
}

public struct SmeltCAMReleaseGateContract: Codable, Sendable, Equatable {
    public static let currentSchema = "smelt.module.release_gate_contract.v1"

    public struct Measurement: Codable, Sendable, Equatable {
        public let subject: String
        public let processMode: String
        public let cacheState: String
        public let occurrence: String

        public init(subject: String, processMode: String, cacheState: String, occurrence: String) {
            self.subject = subject
            self.processMode = processMode
            self.cacheState = cacheState
            self.occurrence = occurrence
        }

        enum CodingKeys: String, CodingKey {
            case subject
            case processMode = "process_mode"
            case cacheState = "cache_state"
            case occurrence
        }
    }

    public let schema: String
    public let contractID: String
    public let contractSHA256: String
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let descriptorGraphSignatureSHA256: String
    public let kind: String
    public let exportID: String
    public let flowID: String
    public let selectedCapabilities: [String]
    public let selectedInputNames: [String]
    public let selectedInputs: [String]
    public let selectedOutputName: String
    public let selectedOutput: String
    public let gateID: String
    public let gateRequirements: [String]
    public let gatePredicates: [String]
    public let gateEvidence: [String]
    public let measurements: [Measurement]
    public let metricSubject: String
    public let metricPath: String
    public let comparator: String
    public let bound: String
    public let unit: String?
    public let fromEventID: String?
    public let fromEventType: String?
    public let fromFlowID: String?
    public let fromEndpoint: String?
    public let toEventID: String?
    public let toEventType: String?
    public let toFlowID: String?
    public let toEndpoint: String?

    enum CodingKeys: String, CodingKey {
        case schema
        case contractID = "contract_id"
        case contractSHA256 = "contract_sha256"
        case camSemanticSHA256 = "module_semantic_sha256"
        case exportABISHA256 = "export_abi_sha256"
        case descriptorGraphSignatureSHA256 = "descriptor_graph_signature_sha256"
        case kind
        case exportID = "export_id"
        case flowID = "flow_id"
        case selectedCapabilities = "selected_capabilities"
        case selectedInputNames = "selected_input_names"
        case selectedInputs = "selected_inputs"
        case selectedOutputName = "selected_output_name"
        case selectedOutput = "selected_output"
        case gateID = "gate_id"
        case gateRequirements = "gate_requirements"
        case gatePredicates = "gate_predicates"
        case gateEvidence = "gate_evidence"
        case measurements
        case metricSubject = "metric_subject"
        case metricPath = "metric_path"
        case comparator
        case bound
        case unit
        case fromEventID = "from_event_id"
        case fromEventType = "from_event_type"
        case fromFlowID = "from_flow_id"
        case fromEndpoint = "from_endpoint"
        case toEventID = "to_event_id"
        case toEventType = "to_event_type"
        case toFlowID = "to_flow_id"
        case toEndpoint = "to_endpoint"
    }

    fileprivate init(payload: SmeltCAMReleaseGateContractPayload) throws {
        let contractSHA256 = try Self.sha256Hex(Self.canonicalJSONData(payload))
        schema = payload.schema
        contractID = "contract-\(contractSHA256.prefix(16))"
        self.contractSHA256 = contractSHA256
        camSemanticSHA256 = payload.camSemanticSHA256
        exportABISHA256 = payload.exportABISHA256
        descriptorGraphSignatureSHA256 = payload.descriptorGraphSignatureSHA256
        kind = payload.kind
        exportID = payload.exportID
        flowID = payload.flowID
        selectedCapabilities = payload.selectedCapabilities
        selectedInputNames = payload.selectedInputNames
        selectedInputs = payload.selectedInputs
        selectedOutputName = payload.selectedOutputName
        selectedOutput = payload.selectedOutput
        gateID = payload.gateID
        gateRequirements = payload.gateRequirements
        gatePredicates = payload.gatePredicates
        gateEvidence = payload.gateEvidence
        measurements = payload.measurements
        metricSubject = payload.metricSubject
        metricPath = payload.metricPath
        comparator = payload.comparator
        bound = payload.bound
        unit = payload.unit
        fromEventID = payload.fromEventID
        fromEventType = payload.fromEventType
        fromFlowID = payload.fromFlowID
        fromEndpoint = payload.fromEndpoint
        toEventID = payload.toEventID
        toEventType = payload.toEventType
        toFlowID = payload.toFlowID
        toEndpoint = payload.toEndpoint
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decode(String.self, forKey: .schema)
        contractID = try container.decode(String.self, forKey: .contractID)
        contractSHA256 = try container.decode(String.self, forKey: .contractSHA256)
        camSemanticSHA256 = try container.decode(String.self, forKey: .camSemanticSHA256)
        exportABISHA256 = try container.decode(String.self, forKey: .exportABISHA256)
        descriptorGraphSignatureSHA256 = try container.decode(
            String.self,
            forKey: .descriptorGraphSignatureSHA256
        )
        kind = try container.decode(String.self, forKey: .kind)
        exportID = try container.decode(String.self, forKey: .exportID)
        flowID = try container.decode(String.self, forKey: .flowID)
        selectedCapabilities = try container.decode([String].self, forKey: .selectedCapabilities)
        selectedInputNames = try container.decode([String].self, forKey: .selectedInputNames)
        selectedInputs = try container.decode([String].self, forKey: .selectedInputs)
        selectedOutputName = try container.decode(String.self, forKey: .selectedOutputName)
        selectedOutput = try container.decode(String.self, forKey: .selectedOutput)
        gateID = try container.decode(String.self, forKey: .gateID)
        gateRequirements = try container.decode([String].self, forKey: .gateRequirements)
        gatePredicates = try container.decode([String].self, forKey: .gatePredicates)
        gateEvidence = try container.decode([String].self, forKey: .gateEvidence)
        measurements = try container.decode([Measurement].self, forKey: .measurements)
        metricSubject = try container.decode(String.self, forKey: .metricSubject)
        metricPath = try container.decode(String.self, forKey: .metricPath)
        comparator = try container.decode(String.self, forKey: .comparator)
        bound = try container.decode(String.self, forKey: .bound)
        unit = try container.decodeIfPresent(String.self, forKey: .unit)
        fromEventID = try container.decodeIfPresent(String.self, forKey: .fromEventID)
        fromEventType = try container.decodeIfPresent(String.self, forKey: .fromEventType)
        fromFlowID = try container.decodeIfPresent(String.self, forKey: .fromFlowID)
        fromEndpoint = try container.decodeIfPresent(String.self, forKey: .fromEndpoint)
        toEventID = try container.decodeIfPresent(String.self, forKey: .toEventID)
        toEventType = try container.decodeIfPresent(String.self, forKey: .toEventType)
        toFlowID = try container.decodeIfPresent(String.self, forKey: .toFlowID)
        toEndpoint = try container.decodeIfPresent(String.self, forKey: .toEndpoint)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try encodePayloadFields(to: &container)
        try container.encode(contractID, forKey: .contractID)
        try container.encode(contractSHA256, forKey: .contractSHA256)
    }

    public func canonicalJSONData(prettyPrinted: Bool = false) throws -> Data {
        try Self.canonicalJSONData(self, prettyPrinted: prettyPrinted)
    }

    fileprivate static func canonicalJSONData<T: Encodable>(
        _ value: T,
        prettyPrinted: Bool = false
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    fileprivate static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func encodePayloadFields(
        to container: inout KeyedEncodingContainer<CodingKeys>
    ) throws {
        try container.encode(schema, forKey: .schema)
        try container.encode(camSemanticSHA256, forKey: .camSemanticSHA256)
        try container.encode(exportABISHA256, forKey: .exportABISHA256)
        try container.encode(
            descriptorGraphSignatureSHA256,
            forKey: .descriptorGraphSignatureSHA256
        )
        try container.encode(kind, forKey: .kind)
        try container.encode(exportID, forKey: .exportID)
        try container.encode(flowID, forKey: .flowID)
        try container.encode(selectedCapabilities, forKey: .selectedCapabilities)
        try container.encode(selectedInputNames, forKey: .selectedInputNames)
        try container.encode(selectedInputs, forKey: .selectedInputs)
        try container.encode(selectedOutputName, forKey: .selectedOutputName)
        try container.encode(selectedOutput, forKey: .selectedOutput)
        try container.encode(gateID, forKey: .gateID)
        try container.encode(gateRequirements, forKey: .gateRequirements)
        try container.encode(gatePredicates, forKey: .gatePredicates)
        try container.encode(gateEvidence, forKey: .gateEvidence)
        try container.encode(measurements, forKey: .measurements)
        try container.encode(metricSubject, forKey: .metricSubject)
        try container.encode(metricPath, forKey: .metricPath)
        try container.encode(comparator, forKey: .comparator)
        try container.encode(bound, forKey: .bound)
        try container.encode(unit, forKey: .unit)
        try container.encode(fromEventID, forKey: .fromEventID)
        try container.encode(fromEventType, forKey: .fromEventType)
        try container.encode(fromFlowID, forKey: .fromFlowID)
        try container.encode(fromEndpoint, forKey: .fromEndpoint)
        try container.encode(toEventID, forKey: .toEventID)
        try container.encode(toEventType, forKey: .toEventType)
        try container.encode(toFlowID, forKey: .toFlowID)
        try container.encode(toEndpoint, forKey: .toEndpoint)
    }
}

private struct SmeltCAMReleaseGateContractPayload: Encodable, Sendable, Equatable {
    let schema: String
    let camSemanticSHA256: String
    let exportABISHA256: String
    let descriptorGraphSignatureSHA256: String
    let kind: String
    let exportID: String
    let flowID: String
    let selectedCapabilities: [String]
    let selectedInputNames: [String]
    let selectedInputs: [String]
    let selectedOutputName: String
    let selectedOutput: String
    let gateID: String
    let gateRequirements: [String]
    let gatePredicates: [String]
    let gateEvidence: [String]
    let measurements: [SmeltCAMReleaseGateContract.Measurement]
    let metricSubject: String
    let metricPath: String
    let comparator: String
    let bound: String
    let unit: String?
    let fromEventID: String?
    let fromEventType: String?
    let fromFlowID: String?
    let fromEndpoint: String?
    let toEventID: String?
    let toEventType: String?
    let toFlowID: String?
    let toEndpoint: String?

    enum CodingKeys: String, CodingKey {
        case schema
        case camSemanticSHA256 = "module_semantic_sha256"
        case exportABISHA256 = "export_abi_sha256"
        case descriptorGraphSignatureSHA256 = "descriptor_graph_signature_sha256"
        case kind
        case exportID = "export_id"
        case flowID = "flow_id"
        case selectedCapabilities = "selected_capabilities"
        case selectedInputNames = "selected_input_names"
        case selectedInputs = "selected_inputs"
        case selectedOutputName = "selected_output_name"
        case selectedOutput = "selected_output"
        case gateID = "gate_id"
        case gateRequirements = "gate_requirements"
        case gatePredicates = "gate_predicates"
        case gateEvidence = "gate_evidence"
        case measurements
        case metricSubject = "metric_subject"
        case metricPath = "metric_path"
        case comparator
        case bound
        case unit
        case fromEventID = "from_event_id"
        case fromEventType = "from_event_type"
        case fromFlowID = "from_flow_id"
        case fromEndpoint = "from_endpoint"
        case toEventID = "to_event_id"
        case toEventType = "to_event_type"
        case toFlowID = "to_flow_id"
        case toEndpoint = "to_endpoint"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schema, forKey: .schema)
        try container.encode(camSemanticSHA256, forKey: .camSemanticSHA256)
        try container.encode(exportABISHA256, forKey: .exportABISHA256)
        try container.encode(
            descriptorGraphSignatureSHA256,
            forKey: .descriptorGraphSignatureSHA256
        )
        try container.encode(kind, forKey: .kind)
        try container.encode(exportID, forKey: .exportID)
        try container.encode(flowID, forKey: .flowID)
        try container.encode(selectedCapabilities, forKey: .selectedCapabilities)
        try container.encode(selectedInputNames, forKey: .selectedInputNames)
        try container.encode(selectedInputs, forKey: .selectedInputs)
        try container.encode(selectedOutputName, forKey: .selectedOutputName)
        try container.encode(selectedOutput, forKey: .selectedOutput)
        try container.encode(gateID, forKey: .gateID)
        try container.encode(gateRequirements, forKey: .gateRequirements)
        try container.encode(gatePredicates, forKey: .gatePredicates)
        try container.encode(gateEvidence, forKey: .gateEvidence)
        try container.encode(measurements, forKey: .measurements)
        try container.encode(metricSubject, forKey: .metricSubject)
        try container.encode(metricPath, forKey: .metricPath)
        try container.encode(comparator, forKey: .comparator)
        try container.encode(bound, forKey: .bound)
        try container.encode(unit, forKey: .unit)
        try container.encode(fromEventID, forKey: .fromEventID)
        try container.encode(fromEventType, forKey: .fromEventType)
        try container.encode(fromFlowID, forKey: .fromFlowID)
        try container.encode(fromEndpoint, forKey: .fromEndpoint)
        try container.encode(toEventID, forKey: .toEventID)
        try container.encode(toEventType, forKey: .toEventType)
        try container.encode(toFlowID, forKey: .toFlowID)
        try container.encode(toEndpoint, forKey: .toEndpoint)
    }
}

public extension SmeltCAMPackageCapabilities {
    func releaseGateContracts(
        requiredGateIDs: [String],
        releaseSurfaces: [SmeltCAMReleaseSurfaceBinding]
    ) throws -> [SmeltCAMReleaseGateContract] {
        try validateReleaseSurfaceBindingShapes(releaseSurfaces)

        var seenRequiredGateIDs: Set<String> = []
        for gateID in requiredGateIDs where !seenRequiredGateIDs.insert(gateID).inserted {
            throw SmeltCAMReleaseGateContractError.duplicateRequiredGate(gateID)
        }

        let contracts = try requiredGateIDs.map { gateID in
            try releaseGateContract(gateID: gateID, releaseSurfaces: releaseSurfaces)
        }
        try validateReleaseSurfaceBindings(releaseSurfaces, contracts: contracts)
        return contracts
    }

    func releaseContractIDs(
        for surface: SmeltCAMReleaseSurfaceBinding,
        contracts: [SmeltCAMReleaseGateContract]
    ) throws -> [String] {
        guard surface.requiresReleaseEvidence else { return [] }
        let contractsByGate = Dictionary(uniqueKeysWithValues: contracts.map {
            ($0.gateID, $0)
        })
        return try surface.gateIDs.map { gateID in
            guard let contract = contractsByGate[gateID] else {
                throw SmeltCAMReleaseGateContractError.missingReleaseGateContract(gateID)
            }
            try validate(surface: surface, references: contract)
            return contract.contractID
        }
    }

    private func releaseGateContract(
        gateID: String,
        releaseSurfaces: [SmeltCAMReleaseSurfaceBinding]
    ) throws -> SmeltCAMReleaseGateContract {
        guard let gate = gateContracts.first(where: { $0.gateID == gateID }) else {
            throw SmeltCAMReleaseGateContractError.missingRequiredGate(gateID)
        }
        let export = try releaseExport(gateID: gateID, releaseSurfaces: releaseSurfaces)
        guard !export.boundFlowID.isEmpty else {
            throw SmeltCAMReleaseGateContractError.missingReleaseFlow(exportID: export.exportID)
        }
        let outputPort = try releaseSelectedOutputPort(gate: gate, export: export)
        let selectedInputs = export.inputPorts
            .filter { !$0.optional }
            .sorted { $0.portName < $1.portName }
        let measurements = gate.measurements.map {
            SmeltCAMReleaseGateContract.Measurement(
                subject: $0.subject,
                processMode: $0.processMode,
                cacheState: $0.cacheState,
                occurrence: $0.occurrence
            )
        }
        let primaryRequirement = try releasePrimaryRequirement(gate)
        let primaryMeasurement = gate.measurements.first {
            $0.subject == primaryRequirement?.subject
        } ?? gate.measurements.first
        let kind = releaseContractKind(gate)
        let fromEventID = releaseEventID(gate.from)
        let toEventID = releaseEventID(gate.to)
        let metricSubject = primaryMeasurement?.subject ?? primaryRequirement?.subject ?? "evidence"
        let gateRequirements = gate.requirements.map(Self.comparisonSignature).sorted()
        let gatePredicates = ([gate.from, gate.to].compactMap { $0 })
            .flatMap(\.predicates)
            .map(Self.comparisonSignature)
            .sorted()
        let gateEvidence = gate.evidence.map(Self.evidenceSignature).sorted()
        let payload = SmeltCAMReleaseGateContractPayload(
            schema: SmeltCAMReleaseGateContract.currentSchema,
            camSemanticSHA256: camSemanticSHA256,
            exportABISHA256: exportABISHA256,
            descriptorGraphSignatureSHA256: descriptorGraphSignatureSHA256,
            kind: kind,
            exportID: export.exportID,
            flowID: export.boundFlowID,
            selectedCapabilities: export.capabilities.sorted(),
            selectedInputNames: selectedInputs.map(\.portName),
            selectedInputs: selectedInputs.map(Self.portShape),
            selectedOutputName: outputPort.portName,
            selectedOutput: Self.portShape(outputPort),
            gateID: gate.gateID,
            gateRequirements: gateRequirements,
            gatePredicates: gatePredicates,
            gateEvidence: gateEvidence,
            measurements: measurements,
            metricSubject: metricSubject,
            metricPath: releaseMetricPath(
                kind: kind,
                gateID: gate.gateID,
                metricSubject: metricSubject,
                fromEventID: fromEventID,
                toEventID: toEventID
            ),
            comparator: primaryRequirement?.relation ?? "present",
            bound: primaryRequirement?.value ?? "\(gateEvidence.count)",
            unit: primaryRequirement?.unit,
            fromEventID: fromEventID,
            fromEventType: gate.from?.eventType,
            fromFlowID: gate.from?.flowID,
            fromEndpoint: Self.endpointSignature(gate.from?.endpoint),
            toEventID: toEventID,
            toEventType: gate.to?.eventType,
            toFlowID: gate.to?.flowID,
            toEndpoint: Self.endpointSignature(gate.to?.endpoint)
        )
        return try SmeltCAMReleaseGateContract(payload: payload)
    }

    private func releaseExport(
        gateID: String,
        releaseSurfaces: [SmeltCAMReleaseSurfaceBinding]
    ) throws -> ExportRecord {
        let matchingExports = exports.filter { $0.gateIDs.contains(gateID) }
        let matchingSurfaces = releaseSurfaces.filter {
            $0.requiresReleaseEvidence && $0.gateIDs.contains(gateID)
        }
        if !matchingSurfaces.isEmpty {
            let export = try releaseSurfaceExport(
                gateID: gateID,
                matchingSurfaces: matchingSurfaces,
                matchingExports: matchingExports
            )
            return export
        }
        if matchingExports.count == 1, let export = matchingExports.first {
            return export
        }
        if matchingExports.count > 1 {
            throw SmeltCAMReleaseGateContractError.ambiguousReleaseExport(
                gateID: gateID,
                exports: matchingExports.map(\.exportID).sorted()
            )
        }

        throw SmeltCAMReleaseGateContractError.missingReleaseSurface(gateID)
    }

    private func releaseSurfaceExport(
        gateID: String,
        matchingSurfaces: [SmeltCAMReleaseSurfaceBinding],
        matchingExports: [ExportRecord]
    ) throws -> ExportRecord {
        let boundPairs = Set(matchingSurfaces.compactMap { surface -> String? in
            guard let exportID = surface.exportID, let flowID = surface.flowID else {
                return nil
            }
            return "\(exportID)\u{1f}\(flowID)"
        })
        guard boundPairs.count == 1,
              let boundPair = boundPairs.first else {
            let exports = matchingSurfaces.compactMap(\.exportID).sorted()
            if exports.isEmpty {
                throw SmeltCAMReleaseGateContractError.missingReleaseExport(gateID: gateID)
            }
            throw SmeltCAMReleaseGateContractError.ambiguousReleaseExport(
                gateID: gateID,
                exports: exports
            )
        }
        let parts = boundPair.split(separator: "\u{1f}", omittingEmptySubsequences: false)
        let exportID = String(parts[0])
        let flowID = String(parts[1])
        guard let export = exports.first(where: { $0.exportID == exportID }) else {
            throw SmeltCAMReleaseGateContractError.missingReleaseExport(gateID: gateID)
        }
        if !matchingExports.isEmpty && !export.gateIDs.contains(gateID) {
            let surfaceID = matchingSurfaces.first?.surfaceID ?? gateID
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surfaceID,
                reason: "export \(exportID) does not expose gate \(gateID)"
            )
        }
        guard export.boundFlowID == flowID else {
            let surfaceID = matchingSurfaces.first?.surfaceID ?? gateID
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surfaceID,
                reason: "flow \(flowID) does not match contract flow \(export.boundFlowID)"
            )
        }
        return export
    }

    private func validateReleaseSurfaceBindingShapes(
        _ surfaces: [SmeltCAMReleaseSurfaceBinding]
    ) throws {
        for surface in surfaces where surface.requiresReleaseEvidence {
            var seenSurfaceGateIDs: Set<String> = []
            for gateID in surface.gateIDs where !seenSurfaceGateIDs.insert(gateID).inserted {
                throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                    surfaceID: surface.surfaceID,
                    reason: "release evidence surface repeats gate \(gateID)"
                )
            }
            guard !surface.gateIDs.isEmpty else {
                throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                    surfaceID: surface.surfaceID,
                    reason: "release evidence surface has no gate ids"
                )
            }
            guard let exportID = surface.exportID,
                  !exportID.isEmpty,
                  let flowID = surface.flowID,
                  !flowID.isEmpty else {
                throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                    surfaceID: surface.surfaceID,
                    reason: "release evidence surface must name export and flow"
                )
            }
        }
        let duplicateSurfaceIDs = duplicateValues(
            surfaces.filter(\.requiresReleaseEvidence).map(\.surfaceID)
        )
        if let duplicateSurfaceID = duplicateSurfaceIDs.first {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: duplicateSurfaceID,
                reason: "duplicate release evidence surface id"
            )
        }
    }

    private func validateReleaseSurfaceBindings(
        _ surfaces: [SmeltCAMReleaseSurfaceBinding],
        contracts: [SmeltCAMReleaseGateContract]
    ) throws {
        let releaseSurfaceGateIDs = Set(surfaces.filter(\.requiresReleaseEvidence).flatMap(\.gateIDs))
        for contract in contracts where !releaseSurfaceGateIDs.contains(contract.gateID) {
            throw SmeltCAMReleaseGateContractError.missingReleaseSurface(contract.gateID)
        }
        for surface in surfaces where surface.requiresReleaseEvidence {
            _ = try releaseContractIDs(for: surface, contracts: contracts)
        }
    }

    private func validate(
        surface: SmeltCAMReleaseSurfaceBinding,
        references contract: SmeltCAMReleaseGateContract
    ) throws {
        if let exportID = surface.exportID, exportID != contract.exportID {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "export \(exportID) does not match contract export \(contract.exportID)"
            )
        }
        if let flowID = surface.flowID, flowID != contract.flowID {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "flow \(flowID) does not match contract flow \(contract.flowID)"
            )
        }
        if !surface.selectedInputNames.isEmpty,
           surface.selectedInputNames != contract.selectedInputNames {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "selected input names do not match contract inputs"
            )
        }
        if !surface.selectedInputs.isEmpty,
           surface.selectedInputs != contract.selectedInputs {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "selected inputs do not match contract inputs"
            )
        }
        if !surface.selectedOutputNames.isEmpty,
           !surface.selectedOutputNames.contains(contract.selectedOutputName) {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "selected output names do not include \(contract.selectedOutputName)"
            )
        }
        if !surface.selectedOutputs.isEmpty,
           !surface.selectedOutputs.contains(contract.selectedOutput) {
            throw SmeltCAMReleaseGateContractError.invalidSurfaceBinding(
                surfaceID: surface.surfaceID,
                reason: "selected outputs do not include \(contract.selectedOutput)"
            )
        }
    }

    private func releaseSelectedOutputPort(
        gate: SmeltCAMPackageDescriptor.GateContract,
        export: ExportRecord
    ) throws -> SmeltCAMPackageDescriptor.Port {
        if gate.to?.endpoint?.endpointType == "moduleOutput",
           let output = gate.to?.endpoint?.name {
            guard let port = export.outputPorts.first(where: { $0.portName == output }) else {
                throw SmeltCAMReleaseGateContractError.missingReleaseOutput(
                    gateID: gate.gateID,
                    outputName: output
                )
            }
            return port
        }
        let requiredOutputs = export.outputPorts.filter { !$0.optional }
        if requiredOutputs.count == 1, let output = requiredOutputs.first {
            return output
        }
        let outputs = requiredOutputs.isEmpty ? export.outputPorts : requiredOutputs
        guard outputs.count == 1, let output = outputs.first else {
            throw SmeltCAMReleaseGateContractError.ambiguousReleaseOutput(
                gateID: gate.gateID,
                outputs: outputs.map(\.portName).sorted()
            )
        }
        return output
    }

    private func releasePrimaryRequirement(
        _ gate: SmeltCAMPackageDescriptor.GateContract
    ) throws -> SmeltCAMPackageDescriptor.Comparison? {
        if let measured = gate.measurements.first,
           let requirement = gate.requirements.first(where: { $0.subject == measured.subject }) {
            return requirement
        }
        if let packageFiles = gate.requirements.first(where: { $0.subject == "package-files" }) {
            return packageFiles
        }
        if let releaseSurfaceIDs = gate.requirements.first(where: { $0.subject == "release-surface-ids" }) {
            return releaseSurfaceIDs
        }
        if let first = gate.requirements.first {
            return first
        }
        if !gate.evidence.isEmpty {
            return nil
        }
        throw SmeltCAMReleaseGateContractError.missingReleaseRequirement(gate.gateID)
    }

    private func releaseContractKind(_ gate: SmeltCAMPackageDescriptor.GateContract) -> String {
        if !gate.measurements.isEmpty {
            return "event-latency"
        }
        if gate.requirements.isEmpty && !gate.evidence.isEmpty {
            return "evidence"
        }
        let subjects = Set(gate.requirements.map(\.subject))
        if subjects.contains("package-files") || subjects.contains("release-surface-ids") {
            return "inventory"
        }
        return "scalar-metric"
    }

    private func releaseMetricPath(
        kind: String,
        gateID: String,
        metricSubject: String,
        fromEventID: String?,
        toEventID: String?
    ) -> String {
        if kind == "event-latency", let fromEventID, let toEventID {
            return "\(metricSubject):\(fromEventID)->\(toEventID)"
        }
        return "\(kind):gate.\(gateID):\(metricSubject)"
    }

    private func releaseEventID(_ event: SmeltCAMPackageDescriptor.GateEvent?) -> String? {
        guard let event else { return nil }
        switch event.eventType {
        case "flow.accepted":
            return ["flow.accepted", event.flowID ?? ""].filter { !$0.isEmpty }
                .joined(separator: ":")
        case "emit":
            if event.endpoint?.endpointType == "moduleOutput",
               let flow = event.flowID,
               let output = event.endpoint?.name {
                return "emit:\(flow).\(output)"
            }
            return ["emit", event.flowID ?? "", Self.endpointSignature(event.endpoint) ?? ""]
                .filter { !$0.isEmpty }
                .joined(separator: ":")
        case "input":
            return ["input", event.flowID ?? "", Self.endpointSignature(event.endpoint) ?? ""]
                .filter { !$0.isEmpty }
                .joined(separator: ":")
        default:
            return [event.eventType, event.flowID ?? "", event.signal ?? ""]
                .filter { !$0.isEmpty }
                .joined(separator: ":")
        }
    }

    private static func endpointSignature(
        _ endpoint: SmeltCAMPackageDescriptor.EndpointRef?
    ) -> String? {
        guard let endpoint else { return nil }
        switch endpoint.endpointType {
        case "moduleInput", "moduleOutput", "graphValue":
            return [endpoint.endpointType, endpoint.name ?? ""].joined(separator: ":")
        case "nodePort":
            return [endpoint.endpointType, endpoint.nodeID ?? "", endpoint.portName ?? ""]
                .joined(separator: ":")
        case "importedPort":
            return [
                endpoint.endpointType,
                endpoint.importAlias ?? "",
                endpoint.exportID ?? "",
                endpoint.portName ?? "",
            ].joined(separator: ":")
        default:
            return endpoint.endpointType
        }
    }

    private static func comparisonSignature(
        _ comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> String {
        [
            comparison.subject,
            comparison.relation,
            comparison.value,
            comparison.unit ?? "none",
        ].joined(separator: ":")
    }

    private static func evidenceSignature(
        _ evidence: SmeltCAMPackageDescriptor.EvidenceRequirement
    ) -> String {
        let storageFormat = evidence.storage?.storageFormat ?? ""
        let groupSize = evidence.storage?.groupSize.map(String.init) ?? ""
        let computeDType = evidence.storage?.computeDType ?? ""
        let fields: [String] = [
            evidence.evidenceType,
            evidence.tensor ?? "",
            evidence.sourceID ?? "",
            evidence.sourceDTypes.joined(separator: ","),
            storageFormat,
            groupSize,
            computeDType,
        ]
        return fields.joined(separator: ":")
    }

    private static func portShape(_ port: SmeltCAMPackageDescriptor.Port) -> String {
        guard !port.type.attributes.isEmpty else { return port.type.typeName }
        let attributes = port.type.attributes
            .map { "\($0.key)=\($0.value)" }
            .sorted()
            .joined(separator: ",")
        return "\(port.type.typeName)[\(attributes)]"
    }

    private func duplicateValues(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var duplicates: Set<String> = []
        for value in values where !seen.insert(value).inserted {
            duplicates.insert(value)
        }
        return duplicates.sorted()
    }
}
