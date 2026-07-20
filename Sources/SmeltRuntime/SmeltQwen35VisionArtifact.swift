import CryptoKit
import Foundation
import Metal
import SmeltSchema

/// Self-contained compiled-component artifact for the selected Qwen3.5 vision
/// subgraph. It is intentionally smaller than a runnable text manifest: the
/// parent execution graph owns scheduling, while this component owns exact
/// tensor bytes and its Metal implementation.
public struct SmeltQwen35VisionArtifactManifest: Codable, Sendable, Equatable {
    public static let currentSchema = "smelt.compiled-component.qwen35-vision.v1"

    public struct Files: Codable, Sendable, Equatable {
        public let weights: String
        public let metallib: String
        public let cam: String

        public init(
            weights: String = "weights.bin",
            metallib: String = "model.metallib",
            cam: String = "cam.json"
        ) {
            self.weights = weights
            self.metallib = metallib
            self.cam = cam
        }
    }

    public struct Tensor: Codable, Sendable, Equatable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let offset: Int
        public let byteCount: Int

        public init(
            name: String,
            dtype: String,
            shape: [Int],
            offset: Int,
            byteCount: Int
        ) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.offset = offset
            self.byteCount = byteCount
        }
    }

    public struct Checksums: Codable, Sendable, Equatable {
        public let weightsSHA256: String
        public let metallibSHA256: String
        public let camSHA256: String

        public init(
            weightsSHA256: String,
            metallibSHA256: String,
            camSHA256: String
        ) {
            self.weightsSHA256 = weightsSHA256
            self.metallibSHA256 = metallibSHA256
            self.camSHA256 = camSHA256
        }
    }

    public let schema: String
    public let componentRole: String
    public let sourceID: String
    public let config: SmeltQwen35VisionConfig
    public let files: Files
    public let totalBytes: Int
    public let tensors: [Tensor]
    public let checksums: Checksums

    public init(
        sourceID: String,
        config: SmeltQwen35VisionConfig,
        files: Files = .init(),
        totalBytes: Int,
        tensors: [Tensor],
        checksums: Checksums
    ) {
        schema = Self.currentSchema
        componentRole = "vision-encoder-merger"
        self.sourceID = sourceID
        self.config = config
        self.files = files
        self.totalBytes = totalBytes
        self.tensors = tensors
        self.checksums = checksums
    }

    public func validate() throws {
        guard schema == Self.currentSchema,
              componentRole == "vision-encoder-merger",
              !sourceID.isEmpty,
              totalBytes > 0,
              !tensors.isEmpty
        else {
            throw SmeltQwen35VisionArtifactError.invalidManifest(
                "schema, role, source, size, or tensor inventory is invalid"
            )
        }
        let names = Set(tensors.map(\.name))
        guard names.count == tensors.count else {
            throw SmeltQwen35VisionArtifactError.invalidManifest(
                "tensor names are not unique"
            )
        }
        var ranges: [(start: Int, end: Int, name: String)] = []
        for tensor in tensors {
            guard (tensor.dtype == "BF16" || tensor.dtype == "F16"),
                  !tensor.shape.isEmpty,
                  tensor.shape.allSatisfy({ $0 > 0 }),
                  tensor.offset >= 0,
                  tensor.byteCount > 0,
                  tensor.byteCount <= totalBytes,
                  tensor.offset <= totalBytes - tensor.byteCount
            else {
                throw SmeltQwen35VisionArtifactError.invalidManifest(
                    "tensor '\(tensor.name)' has invalid dtype, shape, or byte range"
                )
            }
            var elementCount = 1
            for dimension in tensor.shape {
                let product = elementCount.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw SmeltQwen35VisionArtifactError.invalidManifest(
                        "tensor '\(tensor.name)' shape overflows Int"
                    )
                }
                elementCount = product.partialValue
            }
            let byteProduct = elementCount.multipliedReportingOverflow(by: 2)
            guard !byteProduct.overflow else {
                throw SmeltQwen35VisionArtifactError.invalidManifest(
                    "tensor '\(tensor.name)' byte count overflows Int"
                )
            }
            let expectedBytes = byteProduct.partialValue
            guard tensor.byteCount == expectedBytes else {
                throw SmeltQwen35VisionArtifactError.invalidManifest(
                    "tensor '\(tensor.name)' byte count \(tensor.byteCount) "
                        + "does not match shape \(tensor.shape)"
                )
            }
            ranges.append((
                start: tensor.offset,
                end: tensor.offset + tensor.byteCount,
                name: tensor.name
            ))
        }
        let sortedRanges = ranges.sorted { $0.start < $1.start }
        for index in 1..<sortedRanges.count where
            sortedRanges[index].start < sortedRanges[index - 1].end {
            throw SmeltQwen35VisionArtifactError.invalidManifest(
                "tensor byte ranges overlap: '\(sortedRanges[index - 1].name)' "
                    + "and '\(sortedRanges[index].name)'"
            )
        }
    }
}
public final class SmeltQwen35VisionArtifact: CheckpointTensorSource {
    public let path: String
    public let manifest: SmeltQwen35VisionArtifactManifest
    public let module: SmeltCAMIR
    private let mappedWeights: NSData

    public init(path: String, verify: Bool = true) throws {
        self.path = path
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifestURL = root.appendingPathComponent("manifest.json")
        let manifest = try JSONDecoder().decode(
            SmeltQwen35VisionArtifactManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        try manifest.validate()
        self.manifest = manifest

        let weightsURL = root.appendingPathComponent(manifest.files.weights)
        let metallibURL = root.appendingPathComponent(manifest.files.metallib)
        let camURL = root.appendingPathComponent(manifest.files.cam)
        for url in [weightsURL, metallibURL, camURL] where
            !FileManager.default.fileExists(atPath: url.path) {
            throw SmeltQwen35VisionArtifactError.missingFile(url.path)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        guard let fileBytes = attributes[.size] as? NSNumber,
              fileBytes.intValue == manifest.totalBytes
        else {
            throw SmeltQwen35VisionArtifactError.invalidManifest(
                "weights.bin size does not match totalBytes"
            )
        }
        if verify {
            try Self.verify(weightsURL, expected: manifest.checksums.weightsSHA256)
            try Self.verify(metallibURL, expected: manifest.checksums.metallibSHA256)
            try Self.verify(camURL, expected: manifest.checksums.camSHA256)
        }
        module = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: Data(contentsOf: camURL)
        ).validated()
        mappedWeights = try NSData(
            contentsOf: weightsURL,
            options: [.mappedIfSafe]
        )
    }

    public var checkpointTensors: [CheckpointTensorDescriptor] {
        manifest.tensors.enumerated().map { index, tensor in
            CheckpointTensorDescriptor(
                index: index,
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                byteCount: tensor.byteCount
            )
        }
    }

    public func checkpointTensorData(
        _ descriptor: CheckpointTensorDescriptor
    ) -> UnsafeRawPointer {
        let tensor = manifest.tensors[descriptor.index]
        return mappedWeights.bytes.advanced(by: tensor.offset)
    }

    public func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        try device.makeLibrary(
            URL: URL(fileURLWithPath: path, isDirectory: true)
                .appendingPathComponent(manifest.files.metallib)
        )
    }

    private static func verify(_ url: URL, expected: String) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 8 * 1_024 * 1_024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == expected else {
            throw SmeltQwen35VisionArtifactError.checksumMismatch(
                url.lastPathComponent,
                expected: expected,
                got: actual
            )
        }
    }
}

public enum SmeltQwen35VisionArtifactError: Error, CustomStringConvertible {
    case invalidManifest(String)
    case missingFile(String)
    case checksumMismatch(String, expected: String, got: String)

    public var description: String {
        switch self {
        case .invalidManifest(let message):
            return "invalid Qwen3.5 vision artifact: \(message)"
        case .missingFile(let path):
            return "Qwen3.5 vision artifact is missing '\(path)'"
        case .checksumMismatch(let file, let expected, let got):
            return "Qwen3.5 vision artifact checksum mismatch for \(file): "
                + "expected \(expected), got \(got)"
        }
    }
}
