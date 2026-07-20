import CryptoKit
import Foundation
import XCTest
@testable import SmeltRuntime

final class SmeltPackageDerivationTests: XCTestCase {
    func testSuccessfulDerivationReplacesOutputWithoutChangingBase() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let output = fixture.root.appendingPathComponent("triage.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data("old output".utf8).write(to: output.appendingPathComponent("state.txt"))
        let baseBefore = try treeDigest(fixture.base)

        try SmeltPackageDerivation.derive(
            base: fixture.base,
            output: output,
            mutate: { temporary in
                try Data("new agent".utf8).write(
                    to: temporary.appendingPathComponent("smelt.txt")
                )
            },
            validate: { temporary in
                XCTAssertTrue(FileManager.default.fileExists(
                    atPath: temporary.appendingPathComponent("weights.bin").path
                ))
                XCTAssertEqual(
                    try String(
                        contentsOf: temporary.appendingPathComponent("smelt.txt"),
                        encoding: .utf8
                    ),
                    "new agent"
                )
            }
        )

        XCTAssertEqual(try treeDigest(fixture.base), baseBefore)
        XCTAssertEqual(
            try String(
                contentsOf: output.appendingPathComponent("smelt.txt"),
                encoding: .utf8
            ),
            "new agent"
        )
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("state.txt").path
        ))
        XCTAssertTrue(try createTemporaries(in: fixture.root).isEmpty)
    }

    func testMutationFailurePreservesBaseAndPreviousOutput() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let output = fixture.root.appendingPathComponent("triage.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data("old output".utf8).write(to: output.appendingPathComponent("state.txt"))
        let baseBefore = try treeDigest(fixture.base)
        let outputBefore = try treeDigest(output)

        XCTAssertThrowsError(try SmeltPackageDerivation.derive(
            base: fixture.base,
            output: output,
            mutate: { temporary in
                try Data("partial".utf8).write(
                    to: temporary.appendingPathComponent("smelt.txt")
                )
                throw FixtureError.injected
            },
            validate: { _ in XCTFail("validation must not run") }
        ))

        XCTAssertEqual(try treeDigest(fixture.base), baseBefore)
        XCTAssertEqual(try treeDigest(output), outputBefore)
        XCTAssertTrue(try createTemporaries(in: fixture.root).isEmpty)
    }

    func testValidationFailureDoesNotExposeNewOutput() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let output = fixture.root.appendingPathComponent("triage.smeltpkg", isDirectory: true)
        let baseBefore = try treeDigest(fixture.base)

        XCTAssertThrowsError(try SmeltPackageDerivation.derive(
            base: fixture.base,
            output: output,
            mutate: { temporary in
                try Data("partial".utf8).write(
                    to: temporary.appendingPathComponent("smelt.txt")
                )
            },
            validate: { _ in throw FixtureError.injected }
        ))

        XCTAssertEqual(try treeDigest(fixture.base), baseBefore)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        XCTAssertTrue(try createTemporaries(in: fixture.root).isEmpty)
    }

    func testRefusesToOverwriteBasePackage() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        XCTAssertThrowsError(try SmeltPackageDerivation.derive(
            base: fixture.base,
            output: fixture.base,
            mutate: { _ in },
            validate: { _ in }
        )) { error in
            guard case SmeltPackageDerivationError.baseAndOutputMatch = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    private enum FixtureError: Error {
        case injected
    }

    private struct Fixture {
        let root: URL
        let base: URL

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent(
                "agent-derive-tests-\(UUID().uuidString)",
                isDirectory: true
            )
            base = root.appendingPathComponent("base.smeltpkg", isDirectory: true)
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            try Data(repeating: 0xA5, count: 1_048_576).write(
                to: base.appendingPathComponent("weights.bin")
            )
            try Data("base metadata".utf8).write(
                to: base.appendingPathComponent("manifest.json")
            )
        }

        func cleanup() {
            try? FileManager.default.removeItem(at: root)
        }
    }

    private func createTemporaries(in directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.contains(".create-") }
    }

    private func treeDigest(_ directory: URL) throws -> String {
        let fm = FileManager.default
        let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        var rows: [(String, Data)] = []
        while let url = enumerator?.nextObject() as? URL {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            let relative = String(url.path.dropFirst(directory.path.count + 1))
            rows.append((relative, try Data(contentsOf: url)))
        }
        var hash = SHA256()
        for (relative, data) in rows.sorted(by: { $0.0 < $1.0 }) {
            hash.update(data: Data(relative.utf8))
            hash.update(data: Data([0]))
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
