import Darwin
import CryptoKit
import Foundation
import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class PackageIntegrityTests: XCTestCase {

    private static func makeManagedTempRoot(_ name: String) -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(getpid())", isDirectory: true)
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        atexit_b {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }

    private static let tempRoot: URL = {
        makeManagedTempRoot("smelt-package-integrity-tests")
    }()

    private func makeTempPackage() throws -> String {
        let root = Self.tempRoot
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root.path
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func writePackageFiles(at packagePath: String) throws -> SmeltManifest {
        let files: [(String, Data)] = [
            ("weights.bin", Data("weights".utf8)),
            ("model.metallib", Data("metallib".utf8)),
            ("SmeltGenerated.swift", Data("generated".utf8)),
            ("dispatches.bin", Data("dispatches".utf8)),
            ("prefill_dispatches.bin", Data("prefill".utf8)),
            ("prefill_verify_argmax_dispatches.bin", Data("verify_argmax".utf8)),
            ("tokenizer.json", Data("{\"tok\":1}".utf8)),
        ]

        for (name, data) in files {
            try data.write(to: URL(fileURLWithPath: "\(packagePath)/\(name)"))
        }

        return SmeltManifest(
            blocks: .tokenFeedbackText,
            loop: .tokenFeedbackText,
            modelName: "test/model",
            config: SmeltManifestConfig(
                hiddenSize: 1,
                numLayers: 1,
                vocabSize: 1,
                staticSeqCapacity: 1,
                ropeDim: 1,
                numDeltaLayers: 0,
                numAttnLayers: 0,
                ffnDim: 1
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: sha256Hex(Data("weights".utf8)),
                metallib: sha256Hex(Data("metallib".utf8)),
                generatedSwift: sha256Hex(Data("generated".utf8)),
                dispatchesBin: sha256Hex(Data("dispatches".utf8)),
                prefillDispatchesBin: sha256Hex(Data("prefill".utf8)),
                prefillVerifyArgmaxDispatchesBin: sha256Hex(Data("verify_argmax".utf8)),
                tokenizerJSON: sha256Hex(Data("{\"tok\":1}".utf8))
            ),
            buildProvenance: SmeltBuildProvenance(
                buildFingerprint: "build123",
                weightsFingerprint: "weights123",
                specSHA256: "spec123",
                compilerSourcesSHA256: "compiler123",
                shaderSourcesSHA256: "shader123",
                resolvedOptions: SmeltResolvedBuildOptions(
                    layerPatternUnit: ["delta"],
                    layerPatternRepeats: 1,
                    quantizationStrategy: "affine_u4",
                    groupSize: 64,
                    excludePatterns: [],
                    quantizeEmbedding: true,
                    loadingStrategy: "mmap_prefault",
                    packing: "monolithic",
                    prefillEngine: "metal",
                    maxPrefillBatch: 256,
                    prefillHandoffFamilies: [],
                    inferenceMaxTokens: 8,
                    eosTokens: [],
                    thinkToken: nil,
                    thinkEndToken: nil,
                    thinkSkipSuffix: nil,
                    tiedLMHead: true,
                    traceMode: "full"
                )
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 7, entries: []),
            buffers: SmeltBufferTable(slots: []),
            pipelines: [],
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 0,
                recStateBaseSlot: 0,
                keyCacheBaseSlot: 0,
                valCacheBaseSlot: 0,
                ropeCosSlot: 0,
                ropeSinSlot: 0,
                tokenIdSlot: 0,
                positionSlot: 0,
                weightsSlot: 0
            ),
            inference: SmeltInferenceManifest(
                maxTokens: 8,
                eosTokens: [1],
                chatTemplate: "chatml",
                thinkingPolicy: .disabled
            ),
            decode: SmeltPackageSpec.DecodePolicy(
                sampler: .init(mode: .greedy),
                maxSteps: 8
            ),
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "package-integrity",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )
    }

    func testPackageIntegrityAcceptsMatchingChecksums() throws {
        let packagePath = try makeTempPackage()
        let manifest = try writePackageFiles(at: packagePath)
        let manifestData = try manifest.encodePrettyJSON()
        try manifestData.write(to: URL(fileURLWithPath: "\(packagePath)/manifest.json"))

        let report = try SmeltPackageIntegrity.verify(
            packagePath: packagePath,
            includeWeights: true
        )
        XCTAssertEqual(report.verifiedFiles.count, 7)
        XCTAssertEqual(report.skippedFiles.count, 0)
        XCTAssertEqual(report.buildProvenance?.buildFingerprint, "build123")
    }

    func testPackageIntegrityRejectsChecksumMismatch() throws {
        let packagePath = try makeTempPackage()
        let manifest = try writePackageFiles(at: packagePath)
        let manifestData = try manifest.encodePrettyJSON()
        try manifestData.write(to: URL(fileURLWithPath: "\(packagePath)/manifest.json"))

        try Data("tampered".utf8).write(
            to: URL(fileURLWithPath: "\(packagePath)/weights.bin")
        )

        XCTAssertThrowsError(
            try SmeltPackageIntegrity.verify(packagePath: packagePath, includeWeights: true)
        ) { error in
            guard let runtimeError = error as? SmeltRuntimeError else {
                return XCTFail("Expected SmeltRuntimeError, got \(error)")
            }
            guard case .checksumMismatch(let detail) = runtimeError else {
                return XCTFail("Expected checksumMismatch, got \(runtimeError)")
            }
            XCTAssertTrue(detail.contains("weights.bin"))
        }
    }
}
