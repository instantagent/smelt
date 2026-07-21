import Darwin
import CryptoKit
import Foundation
import Testing

@testable import SmeltRuntime

@Suite(.serialized)
struct SmeltPackageStoreTests {
    @Test
    func installsByIdentityAndSharesPackageFileInodes() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-package-store-\(UUID().uuidString)",
            isDirectory: true
        )
        let source = temporary.appendingPathComponent("source.smeltpkg", isDirectory: true)
        let store = temporary.appendingPathComponent("store", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(#"{"version":1,"checksums":{"weights":"abc"}}"#.utf8).write(
            to: source.appendingPathComponent("manifest.json")
        )
        try Data([1, 2, 3, 4]).write(to: source.appendingPathComponent("weights.bin"))
        setenv("SMELT_PACKAGE_STORE_DIR", store.path, 1)
        defer {
            unsetenv("SMELT_PACKAGE_STORE_DIR")
            try? FileManager.default.removeItem(at: temporary)
        }

        let first = try SmeltPackageStore.install(packagePath: source.path)
        let second = try SmeltPackageStore.install(packagePath: source.path)

        #expect(first == second)
        #expect(try SmeltPackageStore.installedPackages() == [first])
        #expect(first.packageURL.lastPathComponent == "\(first.identity).smeltpkg")
        #expect(first.identity.count == 64)

        var firstMetadata = stat()
        var secondMetadata = stat()
        #expect(stat(first.packageURL.appendingPathComponent("weights.bin").path, &firstMetadata) == 0)
        #expect(stat(second.packageURL.appendingPathComponent("weights.bin").path, &secondMetadata) == 0)
        #expect(firstMetadata.st_dev == secondMetadata.st_dev)
        #expect(firstMetadata.st_ino == secondMetadata.st_ino)
    }

    @Test
    func distinctPackagesWithIdenticalWeightsShareBlobInodeOnInstall() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-package-dedup-\(UUID().uuidString)",
            isDirectory: true
        )
        let sourceA = temporary.appendingPathComponent("a.smeltpkg", isDirectory: true)
        let sourceB = temporary.appendingPathComponent("b.smeltpkg", isDirectory: true)
        let store = temporary.appendingPathComponent("store", isDirectory: true)
        let cas = temporary.appendingPathComponent("cas", isDirectory: true)
        for source in [sourceA, sourceB] {
            try FileManager.default.createDirectory(
                at: source,
                withIntermediateDirectories: true
            )
        }
        let weights = Data(repeating: 0x5a, count: 1024 * 1024 + 1)
        let checksum = SHA256.hash(data: weights)
            .map { String(format: "%02x", $0) }
            .joined()
        try weights.write(to: sourceA.appendingPathComponent("weights.bin"))
        try weights.write(to: sourceB.appendingPathComponent("weights.bin"))
        for (source, variant) in [(sourceA, "a"), (sourceB, "b")] {
            let manifest = """
            {"version":1,"package_variant":"\(variant)","checksums":{"weights_bin":"\(checksum)"}}
            """
            try Data(manifest.utf8).write(
                to: source.appendingPathComponent("manifest.json")
            )
        }
        setenv("SMELT_PACKAGE_STORE_DIR", store.path, 1)
        setenv("SMELT_CAS_DIR", cas.path, 1)
        defer {
            unsetenv("SMELT_PACKAGE_STORE_DIR")
            unsetenv("SMELT_CAS_DIR")
            try? FileManager.default.removeItem(at: temporary)
        }

        let installedA = try SmeltPackageStore.install(packagePath: sourceA.path)
        let installedB = try SmeltPackageStore.install(packagePath: sourceB.path)

        #expect(installedA.identity != installedB.identity)
        let weightsA = installedA.packageURL.appendingPathComponent("weights.bin")
        let weightsB = installedB.packageURL.appendingPathComponent("weights.bin")
        var linkA = stat()
        var linkB = stat()
        #expect(lstat(weightsA.path, &linkA) == 0)
        #expect(lstat(weightsB.path, &linkB) == 0)
        #expect(linkA.st_mode & S_IFMT == S_IFLNK)
        #expect(linkB.st_mode & S_IFMT == S_IFLNK)
        var blobA = stat()
        var blobB = stat()
        #expect(stat(weightsA.path, &blobA) == 0)
        #expect(stat(weightsB.path, &blobB) == 0)
        #expect(blobA.st_dev == blobB.st_dev)
        #expect(blobA.st_ino == blobB.st_ino)

        var sourceMetadata = stat()
        #expect(lstat(sourceA.appendingPathComponent("weights.bin").path, &sourceMetadata) == 0)
        #expect(sourceMetadata.st_mode & S_IFMT == S_IFREG)
    }

    @Test
    func rejectsNonDigestLookup() throws {
        #expect(throws: SmeltPackageStoreError.self) {
            _ = try SmeltPackageStore.locate(identity: "not-a-digest")
        }
    }

    @Test
    func materializationDereferencesStoreSymlinksForPortableConsumers() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-package-materialize-\(UUID().uuidString)",
            isDirectory: true
        )
        let source = temporary.appendingPathComponent("source.smeltpkg", isDirectory: true)
        let cas = temporary.appendingPathComponent("cas", isDirectory: true)
        let blobs = cas.appendingPathComponent("sha256", isDirectory: true)
        let store = temporary.appendingPathComponent("store", isDirectory: true)
        let exported = temporary.appendingPathComponent("export.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try Data(#"{"version":1,"checksums":{"weights":"abc"}}"#.utf8).write(
            to: source.appendingPathComponent("manifest.json")
        )
        let blob = blobs.appendingPathComponent(String(repeating: "a", count: 64))
        try Data([5, 4, 3, 2, 1]).write(to: blob)
        try FileManager.default.createSymbolicLink(
            at: source.appendingPathComponent("weights.bin"),
            withDestinationURL: blob
        )
        setenv("SMELT_PACKAGE_STORE_DIR", store.path, 1)
        setenv("SMELT_CAS_DIR", cas.path, 1)
        defer {
            unsetenv("SMELT_PACKAGE_STORE_DIR")
            unsetenv("SMELT_CAS_DIR")
            try? FileManager.default.removeItem(at: temporary)
        }

        let installed = try SmeltPackageStore.install(packagePath: source.path)
        let installedWeights = installed.packageURL.appendingPathComponent("weights.bin")
        var installedMetadata = stat()
        #expect(lstat(installedWeights.path, &installedMetadata) == 0)
        #expect(installedMetadata.st_mode & S_IFMT == S_IFLNK)

        try SmeltPackageStore.materialize(identity: installed.identity, at: exported)
        let exportedWeights = exported.appendingPathComponent("weights.bin")
        var exportedMetadata = stat()
        #expect(lstat(exportedWeights.path, &exportedMetadata) == 0)
        #expect(exportedMetadata.st_mode & S_IFMT == S_IFREG)
        #expect(try Data(contentsOf: exportedWeights) == Data([5, 4, 3, 2, 1]))
        #expect(try SmeltPackageIdentity.compute(packagePath: exported.path) == installed.identity)
    }
}
