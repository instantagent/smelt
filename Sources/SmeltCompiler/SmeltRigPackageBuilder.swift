import SmeltRuntime
import SmeltSchema
import CryptoKit
import Foundation

/// Lossless package builder for the pinned pure-Smelt rig model graph.
public enum SmeltRigPackageBuilder {
    public static let checkpointSHA256 =
        "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"

    public static let shaderFiles = [
        "activations_f32.metal",
        "causal_gqa_attn_simd_f32.metal",
        "decode_gqa_attn_f32.metal",
        "dense_trunk_bf16_precise.metal",
        "gemm_bf16w_f32.metal",
        "gemv_add_bf16w_f32.metal",
        "gemv_gateup_swiglu_bf16w_f32.metal",
        "gemv_qkv_bf16w_f32.metal",
        "gather_row_bf16w_f32.metal",
        "head_norm_rope_f32.metal",
        "rms_norm_codec_f32.metal",
        "rms_norm_head_f32.metal",
        "rope_apply_f32.metal",
        "scale_residual_f32.metal",
        "neural_primitives_f32.metal",
    ]

    public static let pipelines: [SmeltPipeline] = [
        .noncausalAttentionF32,
        .noncausalAttentionQ8F32,
        .noncausalAttentionUpdateF32,
        .layerNormRowsF32,
        .fourierPositionEmbeddingF32,
        .pmpeBF16SemanticsF32,
        .appendStridedFeaturesF32,
        .fsqBase8x5DecodeF32,
        .addRowsF32,
        .sigmoidF32,
        .denseBF16WF32,
        .denseBF16WF32Rows4,
        .denseBF16WF32Rows8,
        .denseBF16WF32Rows8Epilogue,
        .geluF32,
        .layerNormRowsBF16WF32,
        .extractInterleavedHeadPartF32,
        .rmsNormRowsBF16WF32,
        .repackConcatenatedHeadPartsF32,
        .rmsNormCodecBF16WF32,
        .rmsNormHeadBF16WF32,
        .headNormRopeBF16WF32,
        .gatherRowBF16WF32,
        .gemvQKVBF16WF32,
        .decodeGQAAttnF32,
        .gemvAddBF16WF32,
        .gemvGateUpSwigluBF16WF32,
        .gemmBF16WF32,
        .ropeApplyF32,
        .causalGQAAttnCachedF32,
        .scaleResidualTCF32,
        .swigluF32,
        .rmsNormCodecBF16,
        .rmsNormHeadBF16,
        .headNormRopeBF16,
        .gemvQKVBF16,
        .gemvAddBF16,
        .gemvGateUpSwigluBF16,
        .decodeGQAAttnBF16,
        .gemmBF16,
        .ropeApplyBF16,
        .causalGQAAttnCachedBF16,
        .scaleResidualTCBF16,
        .swigluBF16,
        .gatherRowBF16,
        .denseBF16,
    ]

    private struct PlannedStorage {
        let storageID: String
        let tensor: SmeltRigCheckpointPlan.Tensor
        let offset: UInt64
        let allocationByteCount: UInt64
    }

    /// Builds, validates, and atomically publishes a package. Checkpoint BF16
    /// bytes are copied verbatim; tied storages are written once and represented
    /// by aliased manifest entries.
    public static func build(
        checkpointPath: String,
        shaderDirectory: String,
        outputPath: String,
        expectedCheckpointSHA256: String = checkpointSHA256,
        pageSize: Int = Int(getpagesize())
    ) throws {
        guard pageSize > 0, pageSize & (pageSize - 1) == 0 else {
            throw SmeltRigPackageBuilderError.invalidPageSize(pageSize)
        }
        let checkpointURL = URL(fileURLWithPath: checkpointPath)
        let actualCheckpointSHA256 = try sha256(checkpointURL)
        guard actualCheckpointSHA256 == expectedCheckpointSHA256 else {
            throw SmeltRigPackageBuilderError.checkpointChecksumMismatch(
                expected: expectedCheckpointSHA256,
                got: actualCheckpointSHA256
            )
        }
        let checkpoint = try PyTorchCheckpointLoader(path: checkpointPath)
        let plan = try SmeltRigCheckpointPlan(checkpoint: checkpoint)
        let descriptors = checkpoint.checkpointTensors
        let layout = planLayout(plan: plan, pageSize: pageSize)

        let fileManager = FileManager.default
        let destination = URL(fileURLWithPath: outputPath, isDirectory: true)
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporary) }

        let files = SmeltRigPackageManifest.Files()
        let metallibURL = temporary.appendingPathComponent(files.metallib)
        try SmeltCompiler.compileMetalLibrary(
            shaderDir: shaderDirectory,
            shaderFiles: shaderFiles,
            outputPath: metallibURL.path
        )
        let weightsURL = temporary.appendingPathComponent(files.weights)
        try writeWeights(
            url: weightsURL,
            totalBytes: layout.totalBytes,
            storages: layout.storages,
            checkpoint: checkpoint,
            descriptors: descriptors
        )

        let pipelineNames = pipelines.map {
            SmeltKernelCatalog.signature(for: $0).metalFunctionName
        }
        let checksums = SmeltRigPackageManifest.Checksums(
            weightsSHA256: try sha256(weightsURL),
            metallibSHA256: try sha256(metallibURL)
        )
        let manifest = SmeltRigPackageManifest(
            source: .init(
                repository: "VAST-AI-Research/SkinTokens",
                commit: "273b691d35989d71cd17ff2895fdc735097b92d1",
                huggingFaceRevision: "79736cad0fd84de384d5eede659b4ebd24effe33",
                checkpointSHA256: actualCheckpointSHA256
            ),
            files: files,
            pageSize: pageSize,
            totalBytes: layout.totalBytes,
            pipelines: pipelineNames,
            tensors: layout.entries,
            omittedTrainingTensors: plan.trainingOnlyTensors.map(\.name),
            checksums: checksums
        )
        try manifest.validate()
        let languageTrunk = try SmeltQwenDenseTrunkSidecar.prepare(manifest: manifest)
        try manifest.encoded().write(
            to: temporary.appendingPathComponent("manifest.json"),
            options: .atomic
        )
        let moduleDescriptor = try SmeltCAMPackageDescriptor(
            from: SmeltRigPackageManifest.camModule
        )
        try moduleDescriptor.canonicalJSONData(prettyPrinted: true).write(
            to: temporary.appendingPathComponent(
                SmeltCAMPackageDescriptor.packageFileName
            ),
            options: .atomic
        )
        try SmeltQwenDenseTrunkSidecar.commit(
            languageTrunk,
            intoPackage: temporary.path
        )

        // Production-loader verification is the package truth gate before
        // publish. It rehashes both artifacts and revalidates every slice.
        _ = try SmeltRigArtifact(path: temporary.path, verify: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
    }

    private static func planLayout(
        plan: SmeltRigCheckpointPlan,
        pageSize: Int
    ) -> (
        entries: [SmeltRigPackageManifest.Tensor],
        storages: [PlannedStorage],
        totalBytes: UInt64
    ) {
        let page = UInt64(pageSize)
        var cursor: UInt64 = 0
        var byStorage: [String: PlannedStorage] = [:]
        var entries: [SmeltRigPackageManifest.Tensor] = []
        var storages: [PlannedStorage] = []
        entries.reserveCapacity(plan.carriedTensors.count)

        for tensor in plan.carriedTensors {
            let storageID = "\(tensor.storageKey):\(tensor.storageOffset):\(tensor.byteCount)"
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
                SmeltRigPackageManifest.Tensor(
                    name: tensor.name,
                    dtype: "BF16",
                    shape: tensor.shape,
                    offset: storage.offset,
                    byteCount: UInt64(tensor.byteCount),
                    allocationByteCount: storage.allocationByteCount,
                    storageID: storageID,
                    component: tensor.component.rawValue
                )
            )
        }
        return (entries, storages, cursor)
    }

    private static func writeWeights(
        url: URL,
        totalBytes: UInt64,
        storages: [PlannedStorage],
        checkpoint: PyTorchCheckpointLoader,
        descriptors: [CheckpointTensorDescriptor]
    ) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw SmeltRigPackageBuilderError.cannotCreate(url.path)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: totalBytes)
        for storage in storages {
            let descriptor = descriptors[storage.tensor.descriptorIndex]
            let source = checkpoint.checkpointTensorData(descriptor)
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

public enum SmeltRigPackageBuilderError: Error, Equatable {
    case invalidPageSize(Int)
    case checkpointChecksumMismatch(expected: String, got: String)
    case cannotCreate(String)
}
