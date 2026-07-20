import Foundation

/// Self-contained package contract for the pure-Smelt Rig graph.
public struct SmeltRigPackageManifest: Codable, Sendable, Equatable {
    public static let currentSchema = "smelt.rig.v1"
    public static let runExport = "transform"
    public static let runEntrypoint = "mesh.auto-rig"

    public static let runContract = SmeltPackageRunContract(
        export: runExport,
        entrypoint: runEntrypoint,
        input: .init(
            flag: "input",
            mediaTypes: ["model/gltf-binary"],
            fileExtensions: ["glb"],
            help: "Source triangle mesh"
        ),
        output: .init(
            flag: "output",
            mediaTypes: ["model/gltf-binary"],
            fileExtensions: ["glb"],
            help: "Generated skinned mesh"
        ),
        options: [
            .init(
                flag: "skeleton-tokens",
                value: .string,
                help: "Comma-separated skeleton prefix tokens"
            ),
            .init(
                flag: "sampling-seed",
                value: .unsignedInteger,
                defaultValue: "0",
                help: "Surface point-sampling seed"
            ),
            .init(
                flag: "sample-seed",
                value: .unsignedInteger,
                help: "Enable sampled skeleton decoding with this seed"
            ),
            .init(
                flag: "beam-count",
                value: .positiveInteger,
                defaultValue: "10",
                help: "Sampled decoding beam width"
            ),
        ]
    )

    public static let camModule = SmeltFileTransformCAM.module(
        moduleID: "mesh_auto_rig",
        exportID: runExport,
        entrypoint: runEntrypoint,
        inputName: runContract.input.flag,
        inputMediaType: runContract.input.mediaTypes[0],
        outputName: runContract.output.flag,
        outputMediaType: runContract.output.mediaTypes[0]
    )

    public struct Source: Codable, Sendable, Equatable {
        public let repository: String
        public let commit: String
        public let huggingFaceRevision: String
        public let checkpointSHA256: String

        public init(
            repository: String,
            commit: String,
            huggingFaceRevision: String,
            checkpointSHA256: String
        ) {
            self.repository = repository
            self.commit = commit
            self.huggingFaceRevision = huggingFaceRevision
            self.checkpointSHA256 = checkpointSHA256
        }
    }

    public struct Files: Codable, Sendable, Equatable {
        public let weights: String
        public let metallib: String
        public let languageTrunk: String

        public init(
            weights: String = "weights.bin",
            metallib: String = "model.metallib",
            languageTrunk: String = "language-trunk"
        ) {
            self.weights = weights
            self.metallib = metallib
            self.languageTrunk = languageTrunk
        }
    }

    public struct Configuration: Codable, Sendable, Equatable {
        public let sampledPointCount: Int
        public let meshTokenCount: Int
        public let conditionTokenCount: Int
        public let skinTokensPerJoint: Int
        public let meshWidth: Int
        public let vaeWidth: Int
        public let languageHiddenSize: Int
        public let languageLayerCount: Int
        public let languageIntermediateSize: Int
        public let languageQueryHeads: Int
        public let languageKeyValueHeads: Int
        public let languageHeadDim: Int
        public let languageVocabularySize: Int
        public let languageMaximumPositions: Int
        public let ropeTheta: Float
        public let languageRMSNormEpsilon: Float
        public let fsqLevels: [Int]

        public init(
            sampledPointCount: Int,
            meshTokenCount: Int,
            conditionTokenCount: Int,
            skinTokensPerJoint: Int,
            meshWidth: Int,
            vaeWidth: Int,
            languageHiddenSize: Int,
            languageLayerCount: Int,
            languageIntermediateSize: Int,
            languageQueryHeads: Int,
            languageKeyValueHeads: Int,
            languageHeadDim: Int,
            languageVocabularySize: Int,
            languageMaximumPositions: Int,
            ropeTheta: Float,
            languageRMSNormEpsilon: Float,
            fsqLevels: [Int]
        ) {
            self.sampledPointCount = sampledPointCount
            self.meshTokenCount = meshTokenCount
            self.conditionTokenCount = conditionTokenCount
            self.skinTokensPerJoint = skinTokensPerJoint
            self.meshWidth = meshWidth
            self.vaeWidth = vaeWidth
            self.languageHiddenSize = languageHiddenSize
            self.languageLayerCount = languageLayerCount
            self.languageIntermediateSize = languageIntermediateSize
            self.languageQueryHeads = languageQueryHeads
            self.languageKeyValueHeads = languageKeyValueHeads
            self.languageHeadDim = languageHeadDim
            self.languageVocabularySize = languageVocabularySize
            self.languageMaximumPositions = languageMaximumPositions
            self.ropeTheta = ropeTheta
            self.languageRMSNormEpsilon = languageRMSNormEpsilon
            self.fsqLevels = fsqLevels
        }

        public static let pinned = Configuration(
            sampledPointCount: 54_000,
            meshTokenCount: 512,
            conditionTokenCount: 384,
            skinTokensPerJoint: 4,
            meshWidth: 512,
            vaeWidth: 768,
            languageHiddenSize: 896,
            languageLayerCount: 28,
            languageIntermediateSize: 3_072,
            languageQueryHeads: 16,
            languageKeyValueHeads: 8,
            languageHeadDim: 128,
            languageVocabularySize: 33_036,
            languageMaximumPositions: 3_192,
            ropeTheta: 1_000_000,
            languageRMSNormEpsilon: 1e-6,
            fsqLevels: [8, 8, 8, 8, 8]
        )
    }

    public struct Tensor: Codable, Sendable, Equatable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let offset: UInt64
        public let byteCount: UInt64
        public let allocationByteCount: UInt64
        public let storageID: String
        public let component: String

        public init(
            name: String,
            dtype: String,
            shape: [Int],
            offset: UInt64,
            byteCount: UInt64,
            allocationByteCount: UInt64,
            storageID: String,
            component: String
        ) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.offset = offset
            self.byteCount = byteCount
            self.allocationByteCount = allocationByteCount
            self.storageID = storageID
            self.component = component
        }
    }

    public struct Checksums: Codable, Sendable, Equatable {
        public let weightsSHA256: String
        public let metallibSHA256: String

        public init(weightsSHA256: String, metallibSHA256: String) {
            self.weightsSHA256 = weightsSHA256
            self.metallibSHA256 = metallibSHA256
        }
    }

    public let schema: String
    public let run: SmeltPackageRunContract
    public let source: Source
    public let files: Files
    public let configuration: Configuration
    public let pageSize: Int
    public let totalBytes: UInt64
    public let pipelines: [String]
    public let tensors: [Tensor]
    public let omittedTrainingTensors: [String]
    public let checksums: Checksums

    public init(
        run: SmeltPackageRunContract = Self.runContract,
        source: Source,
        files: Files = .init(),
        configuration: Configuration = .pinned,
        pageSize: Int,
        totalBytes: UInt64,
        pipelines: [String],
        tensors: [Tensor],
        omittedTrainingTensors: [String],
        checksums: Checksums
    ) {
        schema = Self.currentSchema
        self.run = run
        self.source = source
        self.files = files
        self.configuration = configuration
        self.pageSize = pageSize
        self.totalBytes = totalBytes
        self.pipelines = pipelines
        self.tensors = tensors
        self.omittedTrainingTensors = omittedTrainingTensors
        self.checksums = checksums
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(from data: Data) throws -> SmeltRigPackageManifest {
        try JSONDecoder().decode(SmeltRigPackageManifest.self, from: data)
    }

    public func validate() throws {
        guard schema == Self.currentSchema else {
            throw SmeltRigPackageManifestError.invalid("unsupported schema: \(schema)")
        }
        guard configuration == .pinned else {
            throw SmeltRigPackageManifestError.invalid("configuration differs from pinned rig model")
        }
        do {
            try run.validate()
        } catch {
            throw SmeltRigPackageManifestError.invalid("invalid run contract: \(error)")
        }
        guard run == Self.runContract else {
            throw SmeltRigPackageManifestError.invalid("run contract differs from pinned interface")
        }
        guard pageSize > 0, pageSize & (pageSize - 1) == 0 else {
            throw SmeltRigPackageManifestError.invalid("pageSize is not a power of two")
        }
        guard tensors.count == 622, omittedTrainingTensors.count == 50 else {
            throw SmeltRigPackageManifestError.invalid("tensor inventory is not 622 carried / 50 omitted")
        }
        guard Set(tensors.map(\.name)).count == tensors.count,
              Set(omittedTrainingTensors).count == omittedTrainingTensors.count,
              Set(pipelines).count == pipelines.count,
              !pipelines.isEmpty
        else {
            throw SmeltRigPackageManifestError.invalid("names or pipelines are empty/duplicated")
        }
        guard files.weights == "weights.bin",
              files.metallib == "model.metallib",
              files.languageTrunk == "language-trunk"
        else {
            throw SmeltRigPackageManifestError.invalid("unexpected package filenames")
        }
        guard source.checkpointSHA256.count == 64,
              checksums.weightsSHA256.count == 64,
              checksums.metallibSHA256.count == 64
        else {
            throw SmeltRigPackageManifestError.invalid("checksum length is invalid")
        }

        let page = UInt64(pageSize)
        var storageRanges: [String: (offset: UInt64, allocation: UInt64, bytes: UInt64)] = [:]
        for tensor in tensors {
            guard tensor.dtype == "BF16",
                  !tensor.name.isEmpty,
                  !tensor.storageID.isEmpty,
                  !tensor.component.isEmpty,
                  !tensor.shape.isEmpty,
                  tensor.shape.allSatisfy({ $0 > 0 }),
                  tensor.offset % page == 0,
                  tensor.allocationByteCount % page == 0,
                  tensor.byteCount > 0,
                  tensor.byteCount <= tensor.allocationByteCount,
                  tensor.offset <= totalBytes,
                  tensor.allocationByteCount <= totalBytes - tensor.offset
            else {
                throw SmeltRigPackageManifestError.invalid(
                    "invalid tensor metadata for \(tensor.name)"
                )
            }
            let elementCount = try tensor.shape.reduce(1) { partial, dimension in
                let product = partial.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw SmeltRigPackageManifestError.invalid(
                        "shape overflow for \(tensor.name)"
                    )
                }
                return product.partialValue
            }
            guard tensor.byteCount == UInt64(elementCount * 2) else {
                throw SmeltRigPackageManifestError.invalid(
                    "byte count does not match BF16 shape for \(tensor.name)"
                )
            }
            let range = (tensor.offset, tensor.allocationByteCount, tensor.byteCount)
            if let existing = storageRanges[tensor.storageID] {
                guard existing == range else {
                    throw SmeltRigPackageManifestError.invalid(
                        "aliased storage metadata differs for \(tensor.name)"
                    )
                }
            } else {
                storageRanges[tensor.storageID] = range
            }
        }
        let sorted = storageRanges.map { (id: $0.key, start: $0.value.offset,
                                          end: $0.value.offset + $0.value.allocation) }
            .sorted { $0.start < $1.start }
        for index in 1..<sorted.count where sorted[index].start < sorted[index - 1].end {
            throw SmeltRigPackageManifestError.invalid(
                "storage ranges overlap: \(sorted[index - 1].id), \(sorted[index].id)"
            )
        }
        let byName = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
        guard let embedding = byName["transformer.model.embed_tokens.weight"],
              let head = byName["transformer.lm_head.weight"],
              embedding.storageID == head.storageID,
              embedding.offset == head.offset,
              embedding.byteCount == head.byteCount
        else {
            throw SmeltRigPackageManifestError.invalid("language-model embedding and head are not tied")
        }
    }
}

public enum SmeltRigPackageManifestError: Error, Equatable {
    case invalid(String)
}
