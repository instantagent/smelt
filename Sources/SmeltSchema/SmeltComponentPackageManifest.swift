import Foundation

/// Self-contained package contract for a CAM-authored native component graph.
public struct SmeltComponentPackageManifest: Codable, Sendable, Equatable {
    public static let currentSchema = "smelt.component-package.v1"

    public struct Source: Codable, Sendable, Equatable {
        public let id: String
        public let kind: String
        public let locator: String
        public let revision: String?
        public let sha256: String

        public init(
            id: String,
            kind: String,
            locator: String,
            revision: String? = nil,
            sha256: String
        ) {
            self.id = id
            self.kind = kind
            self.locator = locator
            self.revision = revision
            self.sha256 = sha256
        }
    }

    public struct Files: Codable, Sendable, Equatable {
        public let weights: String
        public let metallib: String
        public let sidecars: [String: String]

        public init(
            weights: String = "weights.bin",
            metallib: String = "model.metallib",
            sidecars: [String: String] = [:]
        ) {
            self.weights = weights
            self.metallib = metallib
            self.sidecars = sidecars
        }
    }

    public struct Tensor: Codable, Sendable, Equatable {
        public let name: String
        public let target: String
        public let dtype: String
        public let shape: [Int]
        public let offset: UInt64
        public let byteCount: UInt64
        public let allocationByteCount: UInt64
        public let storageID: String
        public let owner: String

        public init(
            name: String,
            target: String,
            dtype: String,
            shape: [Int],
            offset: UInt64,
            byteCount: UInt64,
            allocationByteCount: UInt64,
            storageID: String,
            owner: String
        ) {
            self.name = name
            self.target = target
            self.dtype = dtype
            self.shape = shape
            self.offset = offset
            self.byteCount = byteCount
            self.allocationByteCount = allocationByteCount
            self.storageID = storageID
            self.owner = owner
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
    public let moduleID: String
    public let camSemanticSHA256: String
    public let run: SmeltPackageRunContract
    public let sources: [Source]
    public let files: Files
    public let configuration: [String: String]
    public let pageSize: Int
    public let totalBytes: UInt64
    public let pipelines: [String]
    public let tensors: [Tensor]
    public let omittedTensors: [String]
    public let checksums: Checksums

    public init(
        moduleID: String,
        camSemanticSHA256: String,
        run: SmeltPackageRunContract,
        sources: [Source],
        files: Files = .init(),
        configuration: [String: String] = [:],
        pageSize: Int,
        totalBytes: UInt64,
        pipelines: [String],
        tensors: [Tensor],
        omittedTensors: [String],
        checksums: Checksums
    ) {
        schema = Self.currentSchema
        self.moduleID = moduleID
        self.camSemanticSHA256 = camSemanticSHA256
        self.run = run
        self.sources = sources
        self.files = files
        self.configuration = configuration
        self.pageSize = pageSize
        self.totalBytes = totalBytes
        self.pipelines = pipelines
        self.tensors = tensors
        self.omittedTensors = omittedTensors
        self.checksums = checksums
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(from data: Data) throws -> SmeltComponentPackageManifest {
        try JSONDecoder().decode(SmeltComponentPackageManifest.self, from: data)
    }

    public func validate() throws {
        guard schema == Self.currentSchema else {
            throw SmeltComponentPackageManifestError.invalid("unsupported schema: \(schema)")
        }
        guard !moduleID.isEmpty, camSemanticSHA256.count == 64 else {
            throw SmeltComponentPackageManifestError.invalid("module identity is invalid")
        }
        do {
            try run.validate()
        } catch {
            throw SmeltComponentPackageManifestError.invalid("invalid run contract: \(error)")
        }
        guard pageSize > 0, pageSize & (pageSize - 1) == 0 else {
            throw SmeltComponentPackageManifestError.invalid("pageSize is not a power of two")
        }
        guard !sources.isEmpty,
              Set(sources.map(\.id)).count == sources.count,
              Set(tensors.map(\.name)).count == tensors.count,
              Set(omittedTensors).count == omittedTensors.count,
              Set(pipelines).count == pipelines.count,
              !pipelines.isEmpty
        else {
            throw SmeltComponentPackageManifestError.invalid("names or pipelines are empty/duplicated")
        }
        guard files.weights == "weights.bin",
              files.metallib == "model.metallib",
              files.sidecars.values.allSatisfy({ !$0.isEmpty })
        else {
            throw SmeltComponentPackageManifestError.invalid("unexpected package filenames")
        }
        guard sources.allSatisfy({
                  !$0.id.isEmpty && !$0.kind.isEmpty && !$0.locator.isEmpty && $0.sha256.count == 64
              }),
              checksums.weightsSHA256.count == 64,
              checksums.metallibSHA256.count == 64
        else {
            throw SmeltComponentPackageManifestError.invalid("checksum length is invalid")
        }

        let page = UInt64(pageSize)
        var storageRanges: [String: (offset: UInt64, allocation: UInt64, bytes: UInt64)] = [:]
        for tensor in tensors {
            guard let elementByteCount = Self.elementByteCount(dtype: tensor.dtype),
                  !tensor.name.isEmpty,
                  !tensor.target.isEmpty,
                  !tensor.storageID.isEmpty,
                  !tensor.owner.isEmpty,
                  !tensor.shape.isEmpty,
                  tensor.shape.allSatisfy({ $0 > 0 }),
                  tensor.offset % page == 0,
                  tensor.allocationByteCount % page == 0,
                  tensor.byteCount > 0,
                  tensor.byteCount <= tensor.allocationByteCount,
                  tensor.offset <= totalBytes,
                  tensor.allocationByteCount <= totalBytes - tensor.offset
            else {
                throw SmeltComponentPackageManifestError.invalid(
                    "invalid tensor metadata for \(tensor.name)"
                )
            }
            let elementCount = try tensor.shape.reduce(1) { partial, dimension in
                let product = partial.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw SmeltComponentPackageManifestError.invalid(
                        "shape overflow for \(tensor.name)"
                    )
                }
                return product.partialValue
            }
            guard tensor.byteCount == UInt64(elementCount * elementByteCount) else {
                throw SmeltComponentPackageManifestError.invalid(
                    "byte count does not match \(tensor.dtype) shape for \(tensor.name)"
                )
            }
            let range = (tensor.offset, tensor.allocationByteCount, tensor.byteCount)
            if let existing = storageRanges[tensor.storageID] {
                guard existing == range else {
                    throw SmeltComponentPackageManifestError.invalid(
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
            throw SmeltComponentPackageManifestError.invalid(
                "storage ranges overlap: \(sorted[index - 1].id), \(sorted[index].id)"
            )
        }
    }

    private static func elementByteCount(dtype: String) -> Int? {
        switch dtype.uppercased() {
        case "BF16", "FP16": 2
        case "FP32": 4
        default: nil
        }
    }
}

public enum SmeltComponentPackageManifestError: Error, Equatable {
    case invalid(String)
}
