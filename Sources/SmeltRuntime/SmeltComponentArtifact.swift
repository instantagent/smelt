import SmeltSchema
import CryptoKit
import Foundation
import Metal

/// Validated, mmap-backed CAM component package.
public final class SmeltComponentArtifact: CheckpointTensorSource {
    public let path: String
    public let manifest: SmeltComponentPackageManifest
    public let normWeightSemantics: CheckpointNormWeightSemantics = .modelDeclared
    private let mappedWeights: NSData

    /// Loads the package and optionally verifies the large-file checksums.
    public init(path: String, verify: Bool = true) throws {
        self.path = path
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifestURL = root.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw SmeltComponentArtifactError.missingFile(manifestURL.path)
        }
        let manifest = try SmeltComponentPackageManifest.decode(
            from: Data(contentsOf: manifestURL)
        )
        try manifest.validate()
        self.manifest = manifest

        let weightsURL = root.appendingPathComponent(manifest.files.weights)
        let metallibURL = root.appendingPathComponent(manifest.files.metallib)
        for url in [weightsURL, metallibURL] where
            !FileManager.default.fileExists(atPath: url.path)
        {
            throw SmeltComponentArtifactError.missingFile(url.path)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        guard let size = attributes[.size] as? NSNumber,
              size.uint64Value == manifest.totalBytes
        else {
            throw SmeltComponentArtifactError.invalidPackage(
                "weights.bin size does not match manifest.totalBytes"
            )
        }
        if verify {
            try Self.verify(weightsURL, expected: manifest.checksums.weightsSHA256)
            try Self.verify(metallibURL, expected: manifest.checksums.metallibSHA256)
        }
        mappedWeights = try NSData(contentsOf: weightsURL, options: [.mappedIfSafe])
    }

    public var checkpointTensors: [CheckpointTensorDescriptor] {
        manifest.tensors.enumerated().map { index, tensor in
            CheckpointTensorDescriptor(
                index: index,
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                byteCount: Int(tensor.byteCount)
            )
        }
    }

    /// Returns a zero-copy pointer to a manifest-validated tensor slice.
    public func checkpointTensorData(
        _ descriptor: CheckpointTensorDescriptor
    ) -> UnsafeRawPointer {
        precondition(
            descriptor.index >= 0 && descriptor.index < manifest.tensors.count,
            "descriptor does not belong to this component artifact"
        )
        let tensor = manifest.tensors[descriptor.index]
        precondition(
            tensor.name == descriptor.name,
            "descriptor name does not match component manifest index"
        )
        return mappedWeights.bytes.advanced(by: Int(tensor.offset))
    }

    /// Loads the package's frozen Metal function closure.
    public func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        try device.makeLibrary(
            URL: URL(fileURLWithPath: path, isDirectory: true)
                .appendingPathComponent(manifest.files.metallib)
        )
    }

    /// Creates a page-aligned zero-copy Metal view of one packaged storage.
    /// The artifact must outlive the returned buffer.
    public func makeWeightBuffer(device: MTLDevice, tensorNamed name: String) throws -> MTLBuffer {
        guard let tensor = manifest.tensors.first(where: { $0.name == name }) else {
            throw SmeltComponentArtifactError.missingTensor(name)
        }
        let pointer = UnsafeMutableRawPointer(
            mutating: mappedWeights.bytes.advanced(by: Int(tensor.offset))
        )
        guard let buffer = device.makeBuffer(
            bytesNoCopy: pointer,
            length: Int(tensor.allocationByteCount),
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw SmeltComponentArtifactError.cannotCreateWeightBuffer(name)
        }
        buffer.label = "component.weight.\(name)"
        return buffer
    }

    private static func verify(_ url: URL, expected: String) throws {
        let actual = try sha256(url)
        guard actual == expected else {
            throw SmeltComponentArtifactError.checksumMismatch(
                url.lastPathComponent,
                expected: expected,
                got: actual
            )
        }
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

public enum SmeltComponentArtifactError: Error, Equatable {
    case missingFile(String)
    case invalidPackage(String)
    case checksumMismatch(String, expected: String, got: String)
    case missingTensor(String)
    case cannotCreateWeightBuffer(String)
}
