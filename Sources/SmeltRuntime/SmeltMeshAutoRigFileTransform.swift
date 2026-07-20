import SmeltSchema
import CryptoKit
import Foundation

struct SmeltMeshAutoRigFileTransform: SmeltFileTransformRuntime {
    private struct StageReceipt: Encodable {
        let schema: String
        let samplingPointNormalsSHA256: String
        let samplingSourceVertexIndicesSHA256: String
        let samplingSourceTriangleIndicesSHA256: String
        let samplingBarycentricUVSHA256: String
        let meshQuerySourceIndicesSHA256: String
        let conditionQuerySourceIndicesSHA256: String
        let tokenIDsSHA256: String
        let generatedTokenIDsSHA256: String
        let decodedSkinWeightsSHA256: String
        let vertexJointIndicesSHA256: String
        let vertexWeightsSHA256: String
        let outputSHA256: String
        let generatedTokenCount: Int
        let jointCount: Int
        let decodedSkinVertexCount: Int
        let vertexCount: Int
    }

    static let registration = SmeltFileTransformRegistration(
        entrypoint: SmeltRigPackageManifest.runEntrypoint,
        make: { try SmeltMeshAutoRigFileTransform(packagePath: $0) }
    )

    private let runtime: SmeltAutoRigRuntime

    init(packagePath: String) throws {
        runtime = try SmeltAutoRigRuntime(packagePath: packagePath)
    }

    func run(
        inputURL: URL,
        outputURL: URL,
        options: [String: String]
    ) throws -> SmeltPackageRunResult {
        let startTokens = try skeletonTokens(options["skeleton-tokens"])
        let samplingSeed = try unsignedOption("sampling-seed", options: options)
        let beamCount = try positiveOption("beam-count", options: options)
        let decodeMode: SmeltRigDecodeMode
        if let sampleSeedText = options["sample-seed"] {
            let sampleSeed = try unsignedValue(sampleSeedText, flag: "sample-seed")
            decodeMode =
                beamCount > 1
                ? .beamSampled(seed: sampleSeed, width: beamCount)
                : .sampled(seed: sampleSeed)
        } else {
            decodeMode = .greedy
        }
        let result = try runtime.rigGLB(
            inputURL: inputURL,
            outputURL: outputURL,
            samplingSeed: samplingSeed,
            startTokenIDs: startTokens,
            generationConfiguration: .init(
                policyMode: .validated,
                decodeMode: decodeMode
            )
        )
        try writeDecodedSkinWeightsIfRequested(result.neural.skinWeights.values)
        try writeStageReceiptIfRequested(result: result, outputURL: outputURL)
        return SmeltPackageRunResult(
            outputURL: outputURL,
            summary: "Wrote \(result.vertexSkin.vertexCount) vertices / "
                + "\(result.vertexSkin.jointCount) joints: \(outputURL.path)"
        )
    }

    private func writeDecodedSkinWeightsIfRequested(_ values: [Float]) throws {
        guard
            let path = ProcessInfo.processInfo.environment[
                "SMELT_RIG_DECODED_SKIN_WEIGHTS"
            ], !path.isEmpty
        else {
            return
        }
        try values.withUnsafeBytes { bytes in
            try Data(bytes).write(
                to: URL(fileURLWithPath: path),
                options: .atomic
            )
        }
    }

    private func skeletonTokens(_ value: String?) throws -> [Int] {
        guard let value else {
            return [
                SmeltRigVocabulary.skeletonBOS,
                SmeltRigVocabulary.noClass,
            ]
        }
        let fields = value.split(separator: ",", omittingEmptySubsequences: false)
        let tokens = fields.compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        guard tokens.count == fields.count else {
            throw SmeltPackageRunnerError.invalidOption("skeleton-tokens", value)
        }
        return tokens
    }

    private func unsignedOption(
        _ flag: String,
        options: [String: String]
    ) throws -> UInt64 {
        guard let value = options[flag] else {
            throw SmeltPackageRunnerError.missingOptionDefault(flag)
        }
        return try unsignedValue(value, flag: flag)
    }

    private func positiveOption(
        _ flag: String,
        options: [String: String]
    ) throws -> Int {
        guard let value = options[flag], let parsed = Int(value), parsed > 0 else {
            throw SmeltPackageRunnerError.missingOptionDefault(flag)
        }
        return parsed
    }

    private func unsignedValue(_ value: String, flag: String) throws -> UInt64 {
        guard let parsed = UInt64(value) else {
            throw SmeltPackageRunnerError.invalidOption(flag, value)
        }
        return parsed
    }

    private func writeStageReceiptIfRequested(
        result: SmeltGLBRiggingResult,
        outputURL: URL
    ) throws {
        guard
            let path = ProcessInfo.processInfo.environment[
                "SMELT_RIG_STAGE_RECEIPT"
            ], !path.isEmpty
        else {
            return
        }
        let receipt = StageReceipt(
            schema: "smelt.rig.stage-receipt.v1",
            samplingPointNormalsSHA256: sha256(result.sampling.pointNormals),
            samplingSourceVertexIndicesSHA256: sha256(
                result.sampling.receipt.sourceVertexIndices
            ),
            samplingSourceTriangleIndicesSHA256: sha256(
                result.sampling.receipt.sourceTriangleIndices
            ),
            samplingBarycentricUVSHA256: sha256(
                result.sampling.receipt.barycentricUV
            ),
            meshQuerySourceIndicesSHA256: sha256(
                result.neural.meshQuerySourceIndices
            ),
            conditionQuerySourceIndicesSHA256: sha256(
                result.neural.conditionQuerySourceIndices
            ),
            tokenIDsSHA256: sha256(result.neural.generation.tokenIDs),
            generatedTokenIDsSHA256: sha256(
                result.neural.generation.generatedTokenIDs
            ),
            decodedSkinWeightsSHA256: sha256(result.neural.skinWeights.values),
            vertexJointIndicesSHA256: sha256(result.vertexSkin.jointIndices),
            vertexWeightsSHA256: sha256(result.vertexSkin.weights),
            outputSHA256: try sha256(Data(contentsOf: outputURL)),
            generatedTokenCount: result.neural.generation.generatedTokenIDs.count,
            jointCount: result.vertexSkin.jointCount,
            decodedSkinVertexCount: result.neural.skinWeights.vertexCount,
            vertexCount: result.vertexSkin.vertexCount
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(receipt).write(
            to: URL(fileURLWithPath: path),
            options: .atomic
        )
    }

    private func sha256<T>(_ values: [T]) -> String {
        values.withUnsafeBytes { bytes in
            sha256(Data(bytes))
        }
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map {
            String(format: "%02x", $0)
        }.joined()
    }
}
