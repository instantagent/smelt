import Foundation
import SmeltRuntime
import SmeltSchema

/// A source-neutral tensor inventory projected directly from authored CAM.
public struct SmeltCheckpointTensorPlan: Sendable {
    public struct Tensor: Sendable, Equatable {
        public let sourceID: String
        public let descriptorIndex: Int
        public let name: String
        public let targetName: String
        public let dtype: String
        public let shape: [Int]
        public let byteCount: Int
        public let storageKey: String
        public let storageOffset: Int
        public let owner: String
        public let disposition: SmeltCAMIR.TensorMap.Disposition
        public let layoutOrdinal: Int
    }

    public let tensors: [Tensor]

    public var carriedTensors: [Tensor] {
        tensors.filter { $0.disposition == .carried }
    }

    public var omittedTensors: [Tensor] {
        tensors.filter { $0.disposition == .trainingOnly }
    }

    /// Validates exact source coverage, dtype, shape, byte count, and any CAM
    /// alias groups before exposing tensors to the generic package layout.
    public init(
        module: SmeltCAMIR,
        checkpoints: [String: PyTorchCheckpointLoader]
    ) throws {
        let declared = module.tensors.filter { $0.disposition != nil }
        guard !declared.isEmpty else {
            throw SmeltCheckpointTensorPlanError.emptyInventory
        }
        let declaredBySource = Dictionary(grouping: declared, by: \.source)
        var planned: [Tensor] = []
        planned.reserveCapacity(declared.count)
        var aliases: [String: (sourceID: String, storageKey: String, storageOffset: Int)] = [:]

        for sourceID in declaredBySource.keys.sorted() {
            guard let checkpoint = checkpoints[sourceID],
                  let mappings = declaredBySource[sourceID]
            else {
                throw SmeltCheckpointTensorPlanError.missingSource(sourceID)
            }
            let descriptorsByName = Dictionary(
                grouping: checkpoint.checkpointTensors,
                by: \.name
            )
            let duplicates = descriptorsByName.filter { $0.value.count != 1 }.map(\.key).sorted()
            guard duplicates.isEmpty else {
                throw SmeltCheckpointTensorPlanError.duplicateTensors(duplicates)
            }
            let mappingByName = Dictionary(grouping: mappings, by: { $0.selector.pattern })
            let duplicateMappings = mappingByName.filter { $0.value.count != 1 }.map(\.key).sorted()
            guard duplicateMappings.isEmpty else {
                throw SmeltCheckpointTensorPlanError.duplicateMappings(duplicateMappings)
            }
            let expectedNames = Set(mappingByName.keys)
            let sourceNames = Set(descriptorsByName.keys)
            let missing = expectedNames.subtracting(sourceNames).sorted()
            let unexpected = sourceNames.subtracting(expectedNames).sorted()
            guard missing.isEmpty, unexpected.isEmpty else {
                throw SmeltCheckpointTensorPlanError.coverage(
                    sourceID: sourceID,
                    missing: missing,
                    unexpected: unexpected
                )
            }
            let sourceInfo = Dictionary(
                uniqueKeysWithValues: checkpoint.tensors.map { ($0.name, $0) }
            )
            for mapping in mappings.sorted(by: {
                ($0.layoutOrdinal ?? Int.max, $0.selector.pattern)
                    < ($1.layoutOrdinal ?? Int.max, $1.selector.pattern)
            }) {
                let name = mapping.selector.pattern
                guard !name.contains("*"),
                      !mapping.target.selector.contains("*"),
                      let expectedShape = mapping.shape,
                      let expectedDType = mapping.sourceDType,
                      let disposition = mapping.disposition,
                      let layoutOrdinal = mapping.layoutOrdinal,
                      let descriptor = descriptorsByName[name]?.first,
                      let info = sourceInfo[name]
                else {
                    throw SmeltCheckpointTensorPlanError.incompleteMapping(name)
                }
                guard descriptor.dtype == expectedDType else {
                    throw SmeltCheckpointTensorPlanError.dtype(
                        name,
                        expected: expectedDType,
                        got: descriptor.dtype
                    )
                }
                guard descriptor.shape == expectedShape else {
                    throw SmeltCheckpointTensorPlanError.shape(
                        name,
                        expected: expectedShape,
                        got: descriptor.shape
                    )
                }
                let elementBytes = try Self.elementByteCount(expectedDType)
                let expectedBytes = expectedShape.reduce(elementBytes, *)
                guard descriptor.byteCount == expectedBytes else {
                    throw SmeltCheckpointTensorPlanError.byteCount(
                        name,
                        expected: expectedBytes,
                        got: descriptor.byteCount
                    )
                }
                if let alias = mapping.storageAlias {
                    let identity = (
                        sourceID: sourceID,
                        storageKey: info.storageKey,
                        storageOffset: info.storageOffset
                    )
                    if let existing = aliases[alias],
                       existing.sourceID != identity.sourceID
                        || existing.storageKey != identity.storageKey
                        || existing.storageOffset != identity.storageOffset
                    {
                        throw SmeltCheckpointTensorPlanError.aliasMismatch(alias)
                    }
                    aliases[alias] = identity
                }
                planned.append(
                    Tensor(
                        sourceID: sourceID,
                        descriptorIndex: descriptor.index,
                        name: name,
                        targetName: mapping.target.selector,
                        dtype: expectedDType.uppercased(),
                        shape: expectedShape,
                        byteCount: descriptor.byteCount,
                        storageKey: info.storageKey,
                        storageOffset: info.storageOffset,
                        owner: mapping.owner,
                        disposition: disposition,
                        layoutOrdinal: layoutOrdinal
                    )
                )
            }
        }
        let ordinals = Dictionary(grouping: planned, by: \.layoutOrdinal)
            .filter { $0.value.count != 1 }
            .map(\.key)
            .sorted()
        guard ordinals.isEmpty else {
            throw SmeltCheckpointTensorPlanError.duplicateLayoutOrdinals(ordinals)
        }
        tensors = planned.sorted { $0.layoutOrdinal < $1.layoutOrdinal }
    }

    private static func elementByteCount(_ dtype: String) throws -> Int {
        switch dtype.uppercased() {
        case "BF16", "FP16": 2
        case "FP32": 4
        default: throw SmeltCheckpointTensorPlanError.unsupportedDType(dtype)
        }
    }
}

public enum SmeltCheckpointTensorPlanError: Error, Equatable {
    case emptyInventory
    case missingSource(String)
    case duplicateTensors([String])
    case duplicateMappings([String])
    case coverage(sourceID: String, missing: [String], unexpected: [String])
    case incompleteMapping(String)
    case duplicateLayoutOrdinals([Int])
    case unsupportedDType(String)
    case dtype(String, expected: String, got: String)
    case shape(String, expected: [Int], got: [Int])
    case byteCount(String, expected: Int, got: Int)
    case aliasMismatch(String)
}
