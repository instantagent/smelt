import Darwin
import Foundation
import XCTest
@testable import SmeltAgent
import SmeltRuntime
import SmeltServe

final class SmeltAgentTests: XCTestCase {
    func testAgentResponseOwnsTheEmbeddedSmeltResult() {
        let embedded = SmeltTextGenerator.Response(
            text: "hello",
            tokenIDs: [1, 2, 3],
            promptTokenCount: 7,
            prefillTime: 0.25,
            generateTime: 0.5
        )

        XCTAssertEqual(AgentResponse(smeltResponse: embedded), AgentResponse(
            text: "hello",
            tokenIDs: [1, 2, 3],
            promptTokenCount: 7,
            prefillTime: 0.25,
            generateTime: 0.5
        ))
    }

    func testAgentIsAThinPortableReferenceToOneStoredSmeltPackage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-agent-extraction-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(#"{"format":"smelt-test-v1"}"#.utf8).write(
            to: model.appendingPathComponent("manifest.json")
        )
        let cas = root.appendingPathComponent("cas", isDirectory: true)
        let blobs = cas.appendingPathComponent("sha256", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let blob = blobs.appendingPathComponent(String(repeating: "a", count: 64))
        try Data(repeating: 7, count: 1024).write(to: blob)
        try FileManager.default.createSymbolicLink(
            at: model.appendingPathComponent("weights.bin"),
            withDestinationURL: blob
        )
        let packageStore = root.appendingPathComponent("smelt-store", isDirectory: true)
        setenv("SMELT_PACKAGE_STORE_DIR", packageStore.path, 1)
        setenv("SMELT_CAS_DIR", cas.path, 1)
        defer {
            unsetenv("SMELT_PACKAGE_STORE_DIR")
            unsetenv("SMELT_CAS_DIR")
        }

        let first = try AgentArtifact.create(
            at: root.appendingPathComponent("first.agent", isDirectory: true),
            name: "first",
            modelPackagePath: model.path,
            instructions: "Be concise.",
            tools: ["read"]
        )
        let second = try AgentArtifact.create(
            at: root.appendingPathComponent("second.agent", isDirectory: true),
            name: "second",
            modelPackagePath: model.path
        )

        XCTAssertEqual(first.manifest.model, second.manifest.model)
        XCTAssertEqual(
            try first.manifest.resolveModel().packageURL,
            try second.manifest.resolveModel().packageURL
        )
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: first.url.path),
            [AgentManifest.fileName]
        )
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: first.url.appendingPathComponent("weights.bin").path
        ))
        XCTAssertFalse(String(data: try Data(
            contentsOf: first.url.appendingPathComponent(AgentManifest.fileName)
        ), encoding: .utf8)!.contains(model.path))

        let registry = root.appendingPathComponent("registry", isDirectory: true)
        _ = try AgentRegistry.publish(first, name: "first", registry: registry)
        _ = try AgentRegistry.publish(second, name: "second", registry: registry)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(
            atPath: registry.appendingPathComponent("models").path
        ), ["\(first.manifest.model.smeltPackageIdentity).smeltpkg"])
        let publishedWeights = registry
            .appendingPathComponent("models")
            .appendingPathComponent("\(first.manifest.model.smeltPackageIdentity).smeltpkg")
            .appendingPathComponent("weights.bin")
        var metadata = stat()
        XCTAssertEqual(lstat(publishedWeights.path, &metadata), 0)
        XCTAssertEqual(metadata.st_mode & S_IFMT, S_IFREG)
        XCTAssertEqual(
            try Data(contentsOf: publishedWeights),
            Data(repeating: 7, count: 1024)
        )

        XCTAssertThrowsError(try AgentRegistry.publish(
            first,
            name: "../../outside",
            registry: registry
        )) { error in
            XCTAssertEqual(
                error as? AgentManifestError,
                AgentManifestError.invalidName("../../outside")
            )
        }
        XCTAssertThrowsError(try AgentRegistry.install(
            name: "../../outside",
            registry: registry
        )) { error in
            XCTAssertEqual(
                error as? AgentManifestError,
                AgentManifestError.invalidName("../../outside")
            )
        }
    }

    func testRejectsUnknownAgentVersion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-agent-version-\(UUID().uuidString).agent",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let identity = String(repeating: "a", count: 64)
        try Data(
            """
            {"version":2,"name":"future","model":{"smelt_package_identity":"\(identity)"},"tools":[],"defaultMode":"once"}
            """.utf8
        ).write(to: root.appendingPathComponent(AgentManifest.fileName))
        XCTAssertThrowsError(try AgentArtifact.load(at: root)) { error in
            XCTAssertEqual(
                error as? AgentManifestError,
                AgentManifestError.unsupportedVersion(2)
            )
        }
    }
}
