import CryptoKit
import Foundation
import SmeltRuntime
import SmeltSchema

/// Lossless builder for CAM graphs composed from native runtime nodes.
public enum SmeltComponentPackageBuilder {
    public struct BuildResult: Sendable, Equatable {
        public let packagePath: String
        public let manifestPath: String
        public let metallibPath: String
    }

    private struct SourceCheckpoint {
        let source: SmeltCAMIR.Source
        let loader: PyTorchCheckpointLoader
        let descriptors: [CheckpointTensorDescriptor]
    }

    private struct PlannedStorage {
        let storageID: String
        let tensor: SmeltCheckpointTensorPlan.Tensor
        let offset: UInt64
        let allocationByteCount: UInt64
    }

    /// Builds and atomically publishes a package solely from CAM plus generic
    /// source-id overrides. No model identifier participates in construction.
    @discardableResult
    public static func build(
        module: SmeltCAMIR,
        sourceOverrides: [String: String],
        shaderDirectory: String,
        outputDirectory: String,
        pageSize: Int = Int(getpagesize())
    ) throws -> BuildResult {
        let module = try module.validated()
        guard let run = module.run else {
            throw SmeltComponentPackageBuilderError.missingRunContract
        }
        guard pageSize > 0, pageSize & (pageSize - 1) == 0 else {
            throw SmeltComponentPackageBuilderError.invalidPageSize(pageSize)
        }
        let usedSourceIDs = Set(module.tensors.compactMap {
            $0.disposition == nil ? nil : $0.source
        })
        var checkpoints: [String: SourceCheckpoint] = [:]
        for source in module.sources where usedSourceIDs.contains(source.id) {
            guard source.kind == "pytorch-checkpoint" else {
                throw SmeltComponentPackageBuilderError.unsupportedSourceKind(
                    source.id,
                    source.kind
                )
            }
            guard let path = sourceOverrides[source.id] else {
                throw SmeltComponentPackageBuilderError.missingSourceOverride(source.id)
            }
            let actualSHA256 = try sha256(URL(fileURLWithPath: path))
            guard let expectedSHA256 = source.sha256,
                  actualSHA256 == expectedSHA256
            else {
                throw SmeltComponentPackageBuilderError.sourceChecksumMismatch(
                    sourceID: source.id,
                    expected: source.sha256 ?? "<missing>",
                    got: actualSHA256
                )
            }
            let loader = try PyTorchCheckpointLoader(path: path)
            checkpoints[source.id] = SourceCheckpoint(
                source: source,
                loader: loader,
                descriptors: loader.checkpointTensors
            )
        }
        let plan = try SmeltCheckpointTensorPlan(
            module: module,
            checkpoints: checkpoints.mapValues(\.loader)
        )
        let layout = planLayout(plan: plan, pageSize: pageSize)
        let shaders = requirementValues("shader", in: module.compile)
        let pipelines = requirementValues("pipeline", in: module.compile)
        guard !shaders.isEmpty, !pipelines.isEmpty else {
            throw SmeltComponentPackageBuilderError.missingCompileClosure
        }

        let fileManager = FileManager.default
        let parent = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        let destination = parent.appendingPathComponent(
            "\(module.module.id).smeltpkg",
            isDirectory: true
        )
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporary) }

        let sidecarBlocks = module.blocks.compactMap { block -> (String, String)? in
            guard let path = block.shape.requirements.first(where: { $0.key == "sidecar" })?.value
            else { return nil }
            return (block.id, path)
        }
        let files = SmeltComponentPackageManifest.Files(
            sidecars: Dictionary(uniqueKeysWithValues: sidecarBlocks)
        )
        let metallibURL = temporary.appendingPathComponent(files.metallib)
        try SmeltCompiler.compileMetalLibrary(
            shaderDir: shaderDirectory,
            shaderFiles: shaders,
            outputPath: metallibURL.path
        )
        let weightsURL = temporary.appendingPathComponent(files.weights)
        try writeWeights(
            url: weightsURL,
            totalBytes: layout.totalBytes,
            storages: layout.storages,
            checkpoints: checkpoints
        )

        let manifest = SmeltComponentPackageManifest(
            moduleID: module.module.id,
            camSemanticSHA256: try module.semanticSHA256(),
            run: run,
            sources: checkpoints.values.map {
                .init(
                    id: $0.source.id,
                    kind: $0.source.kind,
                    locator: $0.source.locator,
                    revision: $0.source.revision,
                    sha256: $0.source.sha256 ?? ""
                )
            }.sorted { $0.id < $1.id },
            files: files,
            configuration: configuration(module.blocks),
            pageSize: pageSize,
            totalBytes: layout.totalBytes,
            pipelines: pipelines,
            tensors: layout.entries,
            omittedTensors: plan.omittedTensors.map(\.name),
            checksums: .init(
                weightsSHA256: try sha256(weightsURL),
                metallibSHA256: try sha256(metallibURL)
            )
        )
        try manifest.validate()
        let preparedSidecars = try sidecarBlocks.map { blockID, path in
            guard let block = module.blocks.first(where: { $0.id == blockID }) else {
                throw SmeltComponentPackageBuilderError.missingSidecarBlock(blockID)
            }
            return try SmeltDenseTransformerSidecar.prepare(
                manifest: manifest,
                block: block,
                directoryName: path
            )
        }
        let manifestURL = temporary.appendingPathComponent("manifest.json")
        try manifest.encoded().write(to: manifestURL, options: .atomic)
        try SmeltCAMPackageDescriptor(from: module)
            .canonicalJSONData(prettyPrinted: true)
            .write(
                to: temporary.appendingPathComponent(
                    SmeltCAMPackageDescriptor.packageFileName
                ),
                options: .atomic
            )
        for sidecar in preparedSidecars {
            try SmeltDenseTransformerSidecar.commit(sidecar, intoPackage: temporary.path)
        }
        _ = try SmeltComponentArtifact(path: temporary.path, verify: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
        return BuildResult(
            packagePath: destination.path,
            manifestPath: destination.appendingPathComponent("manifest.json").path,
            metallibPath: destination.appendingPathComponent(files.metallib).path
        )
    }

    private static func requirementValues(
        _ key: String,
        in requirements: [SmeltCAMIR.Constraint]
    ) -> [String] {
        requirements.compactMap { requirement -> (Int, String)? in
            if requirement.key == key { return (Int.min, requirement.value) }
            let prefix = "\(key)."
            guard requirement.key.hasPrefix(prefix),
                  let ordinal = Int(requirement.key.dropFirst(prefix.count))
            else { return nil }
            return (ordinal, requirement.value)
        }.sorted { $0.0 < $1.0 }.map(\.1)
    }

    private static func configuration(_ blocks: [SmeltCAMIR.Block]) -> [String: String] {
        Dictionary(uniqueKeysWithValues: blocks.flatMap { block -> [(String, String)] in
            var values = block.shape.requirements.compactMap { requirement in
                requirement.value.map { ("\(block.id).\(requirement.key)", $0) }
            }
            if let transformer = block.shape.transformer {
                func append(_ key: String, _ value: Int?) {
                    if let value { values.append(("\(block.id).\(key)", String(value))) }
                }
                append("hidden-size", transformer.hiddenSize)
                append(
                    "layer-count",
                    transformer.layers.flatMap { layers in
                        layers.repeatCount.map { layers.roles.count * $0 }
                    }
                )
                append("intermediate-size", transformer.ffn?.dim)
                append("query-heads", transformer.attention?.qHeads)
                append("key-value-heads", transformer.attention?.kvHeads)
                append("head-dimension", transformer.attention?.headDim)
                append("vocabulary-size", transformer.vocab?.size)
                append("rope-theta", transformer.attention?.rope?.theta)
                if let epsilon = transformer.norm?.eps {
                    values.append(("\(block.id).rms-norm-epsilon", epsilon))
                }
            }
            return values
        })
    }

    private static func planLayout(
        plan: SmeltCheckpointTensorPlan,
        pageSize: Int
    ) -> (
        entries: [SmeltComponentPackageManifest.Tensor],
        storages: [PlannedStorage],
        totalBytes: UInt64
    ) {
        let page = UInt64(pageSize)
        var cursor: UInt64 = 0
        var byStorage: [String: PlannedStorage] = [:]
        var entries: [SmeltComponentPackageManifest.Tensor] = []
        var storages: [PlannedStorage] = []
        entries.reserveCapacity(plan.carriedTensors.count)
        for tensor in plan.carriedTensors {
            let storageID = "\(tensor.sourceID):\(tensor.storageKey):\(tensor.storageOffset):\(tensor.byteCount)"
            let storage: PlannedStorage
            if let existing = byStorage[storageID] {
                storage = existing
            } else {
                cursor = aligned(cursor, to: page)
                let allocation = aligned(UInt64(tensor.byteCount), to: page)
                storage = PlannedStorage(
                    storageID: storageID,
                    tensor: tensor,
                    offset: cursor,
                    allocationByteCount: allocation
                )
                byStorage[storageID] = storage
                storages.append(storage)
                cursor += allocation
            }
            entries.append(
                .init(
                    name: tensor.name,
                    target: tensor.targetName,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    offset: storage.offset,
                    byteCount: UInt64(tensor.byteCount),
                    allocationByteCount: storage.allocationByteCount,
                    storageID: storageID,
                    owner: tensor.owner
                )
            )
        }
        return (entries, storages, cursor)
    }

    private static func writeWeights(
        url: URL,
        totalBytes: UInt64,
        storages: [PlannedStorage],
        checkpoints: [String: SourceCheckpoint]
    ) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw SmeltComponentPackageBuilderError.cannotCreate(url.path)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: totalBytes)
        for storage in storages {
            guard let checkpoint = checkpoints[storage.tensor.sourceID] else {
                throw SmeltComponentPackageBuilderError.missingSourceOverride(
                    storage.tensor.sourceID
                )
            }
            let descriptor = checkpoint.descriptors[storage.tensor.descriptorIndex]
            let source = checkpoint.loader.checkpointTensorData(descriptor)
            try handle.seek(toOffset: storage.offset)
            var copied = 0
            while copied < descriptor.byteCount {
                let count = min(8 * 1_024 * 1_024, descriptor.byteCount - copied)
                try handle.write(
                    contentsOf: Data(bytes: source.advanced(by: copied), count: count)
                )
                copied += count
            }
        }
        try handle.synchronize()
        try handle.close()
    }

    private static func aligned(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        (value + alignment - 1) & ~(alignment - 1)
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 8 * 1_024 * 1_024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

public enum SmeltComponentPackageBuilderError: Error, Equatable {
    case invalidPageSize(Int)
    case missingRunContract
    case missingSourceOverride(String)
    case unsupportedSourceKind(String, String)
    case sourceChecksumMismatch(sourceID: String, expected: String, got: String)
    case missingCompileClosure
    case missingSidecarBlock(String)
    case cannotCreate(String)
}
