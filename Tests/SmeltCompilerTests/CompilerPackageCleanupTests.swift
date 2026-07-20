import XCTest

@testable import SmeltCompiler
import SmeltSchema

final class CompilerPackageCleanupTests: XCTestCase {
    func testPackageFileMaterializationResolvesRelativeSourceSymlink() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("smelt-package-materialize-\(UUID().uuidString)", isDirectory: true)
        let snapshot = root.appendingPathComponent("snapshots/revision", isDirectory: true)
        let blobs = root.appendingPathComponent("blobs", isDirectory: true)
        let package = root.appendingPathComponent("Model.smeltpkg", isDirectory: true)
        defer { try? fm.removeItem(at: root) }

        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try fm.createDirectory(at: package, withIntermediateDirectories: true)
        let payload = Data(#"{"version":"1.0"}"#.utf8)
        let blob = blobs.appendingPathComponent("tokenizer-payload")
        try payload.write(to: blob)
        let source = snapshot.appendingPathComponent("tokenizer.json")
        try fm.createSymbolicLink(
            atPath: source.path,
            withDestinationPath: "../../blobs/tokenizer-payload"
        )
        let destination = package.appendingPathComponent("tokenizer.json")

        try SmeltCompiler.materializePackageFile(
            sourcePath: source.path,
            destinationPath: destination.path
        )

        XCTAssertEqual(try Data(contentsOf: destination), payload)
        XCTAssertNil(try? fm.destinationOfSymbolicLink(atPath: destination.path))
    }

    func testGraphCostModelCannotInvalidatePackedWeights() {
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltCompiler/SmeltGraphCostModel.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltCompiler/SmeltFrozenIRCostModel.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltCompiler/DeltaNetPlugin.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltCompiler/SmeltMetalLibraryCompiler.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltOptimizerReport.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltFrozenComponentPlanBuilder.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltMetalFrozenOperationProfiler.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltQwen35VisionFrozenPlan.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltSpeculativeDecode.swift"
        ))
        XCTAssertFalse(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltRuntime/SmeltSuffixLookupDrafter.swift"
        ))
        XCTAssertTrue(SmeltCompiler.sourceCanAffectPackedWeightBytes(
            "Sources/SmeltCompiler/SmeltNativeWeightWriter.swift"
        ))
    }

    func testGeneratedPackageCleanupKeepsWeightsAndRemovesStaleSidecarsAndSymlinks() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("smelt-package-cleanup-\(UUID().uuidString)", isDirectory: true)
        let pkg = root.appendingPathComponent("Model.smeltpkg", isDirectory: true)
        defer { try? fm.removeItem(at: root) }

        try fm.createDirectory(at: pkg, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: pkg.appendingPathComponent("weights.bin"))
        try Data([4]).write(to: pkg.appendingPathComponent("weights.bin.progress"))
        try Data([5]).write(to: pkg.appendingPathComponent("custom.keep"))

        for name in [
            "manifest.json",
            "dispatches.bin",
            "prefill_dispatches.bin",
            SmeltBakeArtifacts.prefixMeta,
            SmeltBakeArtifacts.grammarMeta,
            SmeltPackageInterface.fileName,
            "tokenizer.bin",
        ] {
            try Data(name.utf8).write(to: pkg.appendingPathComponent(name))
        }

        let cache = pkg.appendingPathComponent("cache", isDirectory: true)
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        try Data([6]).write(to: cache.appendingPathComponent("entry.bin"))

        let metallibTarget = root.appendingPathComponent("target.metallib")
        try Data([7]).write(to: metallibTarget)
        try fm.createSymbolicLink(
            atPath: pkg.appendingPathComponent("model.metallib").path,
            withDestinationPath: metallibTarget.path
        )
        try fm.createSymbolicLink(
            atPath: pkg.appendingPathComponent(SmeltBakeArtifacts.grammarTrie).path,
            withDestinationPath: root.appendingPathComponent("missing.trie").path
        )

        try SmeltCompiler.removeStaleGeneratedPackageArtifacts(pkgPath: pkg.path)

        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("weights.bin").path))
        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("weights.bin.progress").path))
        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("custom.keep").path))

        for name in [
            "manifest.json",
            "dispatches.bin",
            "prefill_dispatches.bin",
            SmeltBakeArtifacts.prefixMeta,
            SmeltBakeArtifacts.grammarMeta,
            SmeltBakeArtifacts.grammarTrie,
            SmeltPackageInterface.fileName,
            "tokenizer.bin",
            "model.metallib",
            "cache",
        ] {
            XCTAssertFalse(
                packagePathExistsOrIsSymlink(pkg.appendingPathComponent(name).path),
                "\(name) should be removed before package rebuild output is written"
            )
        }
    }

    func testGeneratedPackageCleanupCanPreserveTokenizerAssetsForWeightReuse() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("smelt-package-cleanup-\(UUID().uuidString)", isDirectory: true)
        let pkg = root.appendingPathComponent("Model.smeltpkg", isDirectory: true)
        defer { try? fm.removeItem(at: root) }

        try fm.createDirectory(at: pkg, withIntermediateDirectories: true)
        try Data([1]).write(to: pkg.appendingPathComponent("weights.bin"))
        try Data([2]).write(to: pkg.appendingPathComponent("tokenizer.json"))
        try Data([3]).write(to: pkg.appendingPathComponent("tokenizer.bin"))
        try Data([4]).write(to: pkg.appendingPathComponent("special_tokens.json"))
        try Data([5]).write(to: pkg.appendingPathComponent(SmeltBakeArtifacts.grammarMeta))

        try SmeltCompiler.removeStaleGeneratedPackageArtifacts(
            pkgPath: pkg.path,
            preserveTokenizerAssets: true
        )

        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("weights.bin").path))
        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("tokenizer.json").path))
        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("tokenizer.bin").path))
        XCTAssertTrue(fm.fileExists(atPath: pkg.appendingPathComponent("special_tokens.json").path))
        XCTAssertFalse(fm.fileExists(
            atPath: pkg.appendingPathComponent(SmeltBakeArtifacts.grammarMeta).path))
    }

    private func packagePathExistsOrIsSymlink(_ path: String) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: path) {
            return true
        }
        return (try? fm.destinationOfSymbolicLink(atPath: path)) != nil
    }
}
