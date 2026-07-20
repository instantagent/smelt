import CryptoKit
import Foundation
import SmeltRuntime
import SmeltSchema

public enum SmeltQwen35VisionArtifactBuilder {
    public static func build(
        modulePath: String,
        checkpointPath: String,
        shaderDirectory: String,
        outputPath: String
    ) throws {
        let module = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: Data(contentsOf: URL(fileURLWithPath: modulePath))
        ).validated()
        let checkpoint = try SafetensorsLoader(directory: checkpointPath)
        let plan = try SmeltQwen35VisionCheckpointPlan(
            module: module,
            checkpoint: checkpoint
        )
        let descriptors = checkpoint.checkpointTensors

        let fileManager = FileManager.default
        let destination = URL(fileURLWithPath: outputPath, isDirectory: true)
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporary) }

        let files = SmeltQwen35VisionArtifactManifest.Files()
        let camURL = temporary.appendingPathComponent(files.cam)
        try module.canonicalJSONData(prettyPrinted: true).write(to: camURL)

        let compileShaders = temporary.appendingPathComponent("shaders", isDirectory: true)
        try fileManager.createDirectory(at: compileShaders, withIntermediateDirectories: true)
        for file in [
            "qwen35_vision.metal",
            "gemm_bf16w_f32.metal",
            "gemm_f16w_f32.metal",
        ] {
            let source = URL(fileURLWithPath: shaderDirectory, isDirectory: true)
                .appendingPathComponent(file)
            guard fileManager.fileExists(atPath: source.path) else {
                throw SmeltQwen35VisionArtifactBuilderError.missingShader(source.path)
            }
            try fileManager.copyItem(
                at: source,
                to: compileShaders.appendingPathComponent(file)
            )
        }
        let metallibURL = temporary.appendingPathComponent(files.metallib)
        try SmeltCompiler.compileMetalLibrary(
            shaderDir: compileShaders.path,
            outputPath: metallibURL.path
        )
        try fileManager.removeItem(at: compileShaders)

        var offset = 0
        var entries: [SmeltQwen35VisionArtifactManifest.Tensor] = []
        entries.reserveCapacity(plan.tensors.count)
        for tensor in plan.tensors {
            let descriptor = descriptors[tensor.descriptorIndex]
            offset = aligned(offset, to: 64)
            entries.append(
                .init(
                    name: tensor.name,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    offset: offset,
                    byteCount: descriptor.byteCount
                )
            )
            offset += descriptor.byteCount
        }
        let totalBytes = aligned(offset, to: 4_096)
        let weightsURL = temporary.appendingPathComponent(files.weights)
        guard fileManager.createFile(atPath: weightsURL.path, contents: nil) else {
            throw SmeltQwen35VisionArtifactBuilderError.cannotCreate(weightsURL.path)
        }
        let handle = try FileHandle(forWritingTo: weightsURL)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(totalBytes))
        for (tensor, entry) in zip(plan.tensors, entries) {
            let descriptor = descriptors[tensor.descriptorIndex]
            let source = checkpoint.checkpointTensorData(descriptor)
            try handle.seek(toOffset: UInt64(entry.offset))
            var copied = 0
            while copied < entry.byteCount {
                let count = min(8 * 1_024 * 1_024, entry.byteCount - copied)
                try handle.write(
                    contentsOf: Data(bytes: source.advanced(by: copied), count: count)
                )
                copied += count
            }
        }
        try handle.synchronize()
        try handle.close()

        let checksums = SmeltQwen35VisionArtifactManifest.Checksums(
            weightsSHA256: try sha256(weightsURL),
            metallibSHA256: try sha256(metallibURL),
            camSHA256: try sha256(camURL)
        )
        let manifest = SmeltQwen35VisionArtifactManifest(
            sourceID: plan.sourceID,
            config: plan.config,
            files: files,
            totalBytes: totalBytes,
            tensors: entries,
            checksums: checksums
        )
        try manifest.validate()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(
            to: temporary.appendingPathComponent("manifest.json")
        )

        // Validate the finished bytes through the production loader before the
        // atomic publish. A malformed component never replaces a working one.
        _ = try SmeltQwen35VisionArtifact(path: temporary.path, verify: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
    }

    private static func aligned(_ value: Int, to alignment: Int) -> Int {
        ((value + alignment - 1) / alignment) * alignment
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
public enum SmeltQwen35VisionArtifactBuilderError: Error, CustomStringConvertible {
    case missingShader(String)
    case cannotCreate(String)

    public var description: String {
        switch self {
        case .missingShader(let path):
            return "Qwen3.5 vision artifact shader is missing: \(path)"
        case .cannotCreate(let path):
            return "cannot create Qwen3.5 vision artifact file: \(path)"
        }
    }
}
