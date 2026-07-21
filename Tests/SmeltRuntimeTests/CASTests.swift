import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class CASTests: XCTestCase {

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
        makeManagedTempRoot("smelt-cas-tests")
    }()

    override func setUp() {
        super.setUp()
        // Fresh store per test, inside the managed temp root: tests never
        // touch the real Application Support store or each other's
        // entries (the corruption test poisons its store deliberately).
        setenv(
            "SMELT_CAS_DIR",
            Self.tempRoot
                .appendingPathComponent("store-\(UUID().uuidString)").path,
            1
        )
    }

    override func tearDown() {
        unsetenv("SMELT_CAS_DIR")
        super.tearDown()
    }

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

    private static let fileContents: [(String, Data)] = [
        ("weights.bin", Data("weights".utf8)),
        ("model.metallib", Data("metallib".utf8)),
        ("SmeltGenerated.swift", Data("generated".utf8)),
        ("dispatches.bin", Data("dispatches".utf8)),
        ("prefill_dispatches.bin", Data("prefill".utf8)),
        ("prefill_verify_argmax_dispatches.bin", Data("verify_argmax".utf8)),
        ("tokenizer.json", Data("{\"tok\":1}".utf8)),
    ]

    /// Prepared artifacts: no manifest checksum; CAS keys them by
    /// content hash at adopt time.
    private static let unchecksummedContents: [(String, Data)] = [
        ("compiled_grammar.trie", Data("trie-bytes".utf8)),
        ("prepared_prefix.snapshot", Data("snapshot-bytes".utf8)),
        ("model.metalarchive", Data("metalarchive-bytes".utf8)),
    ]

    private static var allContents: [(String, Data)] {
        fileContents + unchecksummedContents
    }

    @discardableResult
    private func writePackage(at packagePath: String) throws -> SmeltManifest {
        for (name, data) in Self.allContents {
            try data.write(to: URL(fileURLWithPath: "\(packagePath)/\(name)"))
        }
        let manifest = SmeltManifest(
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
                parityFixture: "cas",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )
        let manifestData = try manifest.encodePrettyJSON()
        try manifestData.write(
            to: URL(fileURLWithPath: "\(packagePath)/manifest.json")
        )
        return manifest
    }

    private func isSymlink(_ path: String) -> Bool {
        var st = stat()
        guard lstat(path, &st) == 0 else { return false }
        return (st.st_mode & S_IFMT) == S_IFLNK
    }

    func testAdoptSwapsToSymlinkAndPreservesContent() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        let report = try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)

        for file in report.files {
            XCTAssertEqual(file.state, .adopted, file.name)
        }
        XCTAssertEqual(report.files.count, Self.allContents.count)
        for (name, data) in Self.allContents {
            let path = "\(packagePath)/\(name)"
            XCTAssertTrue(isSymlink(path), name)
            // open() follows the symlink — the runtime's view is unchanged.
            XCTAssertEqual(
                try Data(contentsOf: URL(fileURLWithPath: path)), data, name
            )
        }
        // manifest.json is never adopted: it is the package's identity.
        XCTAssertFalse(isSymlink("\(packagePath)/manifest.json"))
        // The real integrity verifier passes through the symlinks.
        let integrity = try SmeltPackageIntegrity.verify(
            packagePath: packagePath, includeWeights: true
        )
        XCTAssertEqual(integrity.verifiedFiles.count, 7)
    }

    func testSecondPackageDedupsToSameEntry() throws {
        let packageA = try makeTempPackage()
        let packageB = try makeTempPackage()
        try writePackage(at: packageA)
        try writePackage(at: packageB)

        try SmeltCAS.adopt(packagePath: packageA, minBytes: 0)
        let reportB = try SmeltCAS.adopt(packagePath: packageB, minBytes: 0)

        for file in reportB.files {
            XCTAssertEqual(file.state, .adopted, file.name)
        }
        // Checksummed and unchecksummed artifacts alike land on the same
        // store entry when their bytes match.
        for name in ["weights.bin", "compiled_grammar.trie"] {
            let targetA = try FileManager.default.destinationOfSymbolicLink(
                atPath: "\(packageA)/\(name)"
            )
            let targetB = try FileManager.default.destinationOfSymbolicLink(
                atPath: "\(packageB)/\(name)"
            )
            XCTAssertEqual(targetA, targetB, name)
        }
    }

    func testDivergentPreparedArtifactsDoNotShare() throws {
        let packageA = try makeTempPackage()
        let packageB = try makeTempPackage()
        try writePackage(at: packageA)
        try writePackage(at: packageB)
        // Different prepared grammar in B: same model, different package state.
        try Data("other-trie".utf8).write(
            to: URL(fileURLWithPath: "\(packageB)/compiled_grammar.trie")
        )

        try SmeltCAS.adopt(packagePath: packageA, minBytes: 0)
        try SmeltCAS.adopt(packagePath: packageB, minBytes: 0)

        let targetA = try FileManager.default.destinationOfSymbolicLink(
            atPath: "\(packageA)/compiled_grammar.trie"
        )
        let targetB = try FileManager.default.destinationOfSymbolicLink(
            atPath: "\(packageB)/compiled_grammar.trie"
        )
        XCTAssertNotEqual(targetA, targetB)
        XCTAssertEqual(
            try Data(contentsOf: URL(
                fileURLWithPath: "\(packageB)/compiled_grammar.trie"
            )),
            Data("other-trie".utf8)
        )
    }

    func testAdoptIsIdempotent() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
        let second = try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)

        for file in second.files {
            XCTAssertEqual(file.state, .alreadyAdopted, file.name)
        }
    }

    func testCorruptedFileRefusesAdoption() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)
        try Data("tampered".utf8).write(
            to: URL(fileURLWithPath: "\(packagePath)/weights.bin")
        )

        XCTAssertThrowsError(
            try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
        )
        // The tampered file was refused, not adopted.
        XCTAssertFalse(isSymlink("\(packagePath)/weights.bin"))
        // The untampered files were still deduplicated.
        XCTAssertTrue(isSymlink("\(packagePath)/model.metallib"))
        XCTAssertTrue(isSymlink("\(packagePath)/compiled_grammar.trie"))
    }

    func testRestoreRoundTrips() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
        let report = try SmeltCAS.restore(packagePath: packagePath)

        for file in report.files {
            XCTAssertEqual(file.state, .restored, file.name)
        }
        for (name, data) in Self.allContents {
            let path = "\(packagePath)/\(name)"
            XCTAssertFalse(isSymlink(path), name)
            XCTAssertEqual(
                try Data(contentsOf: URL(fileURLWithPath: path)), data, name
            )
        }
        let integrity = try SmeltPackageIntegrity.verify(
            packagePath: packagePath, includeWeights: true
        )
        XCTAssertEqual(integrity.verifiedFiles.count, 7)
    }

    func testBelowThresholdFilesStayInPackage() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        // Default threshold (8 MiB) leaves these tiny fixtures alone.
        let report = try SmeltCAS.adopt(packagePath: packagePath)

        for file in report.files {
            XCTAssertEqual(file.state, .belowThreshold, file.name)
        }
        for (name, _) in Self.allContents {
            XCTAssertFalse(isSymlink("\(packagePath)/\(name)"), name)
        }
    }

    func testStatusReportsEligibilityWithoutChanges() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        let report = try SmeltCAS.status(packagePath: packagePath, minBytes: 0)

        XCTAssertEqual(report.adoptable.count, Self.allContents.count)
        for (name, _) in Self.allContents {
            XCTAssertFalse(isSymlink("\(packagePath)/\(name)"), name)
        }
    }

    func testCorruptSameSizeStoreEntryIsRefused() throws {
        let packageA = try makeTempPackage()
        let packageB = try makeTempPackage()
        try writePackage(at: packageA)
        try writePackage(at: packageB)

        try SmeltCAS.adopt(packagePath: packageA, minBytes: 0)
        // Corrupt the store entry in place: same size, different bytes.
        let entry = try FileManager.default.destinationOfSymbolicLink(
            atPath: "\(packageA)/weights.bin"
        )
        chmod(entry, 0o644)
        try Data("weighXs".utf8).write(to: URL(fileURLWithPath: entry))

        // B must not be pointed at an entry whose contents no longer
        // hash to its name, despite the size matching.
        XCTAssertThrowsError(
            try SmeltCAS.adopt(packagePath: packageB, minBytes: 0)
        )
        XCTAssertFalse(isSymlink("\(packageB)/weights.bin"))
    }

    func testRestorePreflightLeavesPackageUntouchedOnBrokenLink() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)
        try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)

        // Break one link by deleting its store entry.
        let entry = try FileManager.default.destinationOfSymbolicLink(
            atPath: "\(packagePath)/tokenizer.json"
        )
        try FileManager.default.removeItem(atPath: entry)

        XCTAssertThrowsError(
            try SmeltCAS.restore(packagePath: packagePath)
        )
        // Preflight aborted before restoring anything.
        for (name, _) in Self.allContents where name != "tokenizer.json" {
            XCTAssertTrue(isSymlink("\(packagePath)/\(name)"), name)
        }
    }

    func testRestorePreservesSnapshotOwnerOnlyPermissions() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)
        try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
        try SmeltCAS.restore(packagePath: packagePath)

        var st = stat()
        XCTAssertEqual(stat("\(packagePath)/prepared_prefix.snapshot", &st), 0)
        XCTAssertEqual(st.st_mode & 0o777, 0o600)
        XCTAssertEqual(stat("\(packagePath)/weights.bin", &st), 0)
        XCTAssertEqual(st.st_mode & 0o777, 0o644)
    }

    func testConcurrentAdoptersOnSamePackageBothSucceed() throws {
        let packagePath = try makeTempPackage()
        try writePackage(at: packagePath)

        let group = DispatchGroup()
        nonisolated(unsafe) var errors: [Error] = []   // guarded by `lock` below
        let lock = NSLock()
        for _ in 0..<2 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                do {
                    try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
                } catch {
                    lock.lock()
                    errors.append(error)
                    lock.unlock()
                }
            }
        }
        group.wait()

        // The loser of the swap race must report success (already
        // adopted), never a spurious failure.
        XCTAssertTrue(errors.isEmpty, "\(errors)")
        for (name, data) in Self.allContents {
            let path = "\(packagePath)/\(name)"
            XCTAssertTrue(isSymlink(path), name)
            XCTAssertEqual(
                try Data(contentsOf: URL(fileURLWithPath: path)), data, name
            )
        }
    }

    func testConcurrentAdoptersBothSucceed() throws {
        let packageA = try makeTempPackage()
        let packageB = try makeTempPackage()
        try writePackage(at: packageA)
        try writePackage(at: packageB)

        let group = DispatchGroup()
        nonisolated(unsafe) var errors: [Error] = []   // guarded by `lock` below
        let lock = NSLock()
        for packagePath in [packageA, packageB] {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                do {
                    try SmeltCAS.adopt(packagePath: packagePath, minBytes: 0)
                } catch {
                    lock.lock()
                    errors.append(error)
                    lock.unlock()
                }
            }
        }
        group.wait()

        XCTAssertTrue(errors.isEmpty, "\(errors)")
        XCTAssertTrue(isSymlink("\(packageA)/weights.bin"))
        XCTAssertTrue(isSymlink("\(packageB)/weights.bin"))
        for packagePath in [packageA, packageB] {
            XCTAssertEqual(
                try Data(contentsOf: URL(
                    fileURLWithPath: "\(packagePath)/weights.bin"
                )),
                Data("weights".utf8)
            )
        }
    }
}
