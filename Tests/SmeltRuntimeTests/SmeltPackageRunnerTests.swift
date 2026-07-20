import Foundation
import Testing

@testable import SmeltRuntime
@testable import SmeltSchema

@Suite("CAM-selected file transforms", .serialized)
struct SmeltPackageRunnerTests {
    private struct FixtureManifest: Encodable {
        let schema = "smelt.unrelated.fixture.v99"
        let run: SmeltPackageRunContract
    }

    @Test("A second-client artifact copy runs without a package-family switch")
    func copyTransform() throws {
        let fixture = try makeCopyPackage()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let input = fixture.root.appendingPathComponent("input.bin")
        let output = fixture.root.appendingPathComponent("output.bin")
        let expected = Data("lego-bricks".utf8)
        try expected.write(to: input)

        let runner = try SmeltPackageRunner(packagePath: fixture.package.path)
        let result = try runner.run(
            inputURL: input,
            outputURL: output,
            options: [:]
        )

        #expect(try Data(contentsOf: output) == expected)
        #expect(result.summary.contains("11 bytes"))
        #expect(runner.contract.export == "transform")
        #expect(runner.contract.entrypoint == "artifact.copy")
    }

    @Test("The public smelt run command uses the same second-client CAM route")
    func copyTransformThroughCLI() throws {
        let fixture = try makeCopyPackage()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let input = fixture.root.appendingPathComponent("input.bin")
        let output = fixture.root.appendingPathComponent("output.bin")
        let expected = Data("public-lego-route".utf8)
        try expected.write(to: input)

        let process = Process()
        process.executableURL = try iaExecutable()
        process.arguments = [
            "run",
            fixture.package.path,
            "--input",
            input.path,
            "--output",
            output.path,
        ]
        let standardError = Pipe()
        process.standardError = standardError
        process.standardOutput = Pipe()
        try process.run()
        process.waitUntilExit()
        let errorText = String(
            decoding: standardError.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self
        )

        #expect(process.terminationStatus == 0, Comment(rawValue: errorText))
        #expect(try Data(contentsOf: output) == expected)
        #expect(errorText.contains("Wrote 17 bytes"))
    }

    @Test("A run contract cannot redirect the CAM-selected export")
    func exportMismatchFails() throws {
        let fixture = try makeCopyPackage(contractExport: "different")
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        #expect(throws: SmeltPackageRunnerError.self) {
            _ = try SmeltPackageRunner(packagePath: fixture.package.path)
        }
    }

    private func makeCopyPackage(
        contractExport: String = "transform"
    ) throws -> (root: URL, package: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "agent-file-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        let package = root.appendingPathComponent("copy.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(
            at: package,
            withIntermediateDirectories: true
        )
        let contract = SmeltPackageRunContract(
            export: contractExport,
            entrypoint: "artifact.copy",
            input: .init(
                flag: "input",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Source bytes"
            ),
            output: .init(
                flag: "output",
                mediaTypes: ["application/octet-stream"],
                fileExtensions: ["bin"],
                help: "Copied bytes"
            )
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(FixtureManifest(run: contract)).write(
            to: package.appendingPathComponent("manifest.json")
        )
        let module = SmeltFileTransformCAM.module(
            moduleID: "fixture_copy",
            exportID: "transform",
            entrypoint: "artifact.copy",
            inputName: "input",
            inputMediaType: "application/octet-stream",
            outputName: "output",
            outputMediaType: "application/octet-stream"
        )
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        try descriptor.canonicalJSONData().write(
            to: package.appendingPathComponent(
                SmeltCAMPackageDescriptor.packageFileName
            )
        )
        return (root, package)
    }

    private func iaExecutable() throws -> URL {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            Bundle(for: SmeltPackageRunnerTestBundleMarker.self).bundleURL
                .deletingLastPathComponent().appendingPathComponent("smelt"),
            repoRoot.appendingPathComponent(".build/debug/smelt"),
            repoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/smelt"),
            repoRoot.appendingPathComponent(".build/release/smelt"),
            repoRoot.appendingPathComponent(".build/arm64-apple-macosx/release/smelt"),
        ]
        for candidate in candidates
            where FileManager.default.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
        throw SmeltPackageRunnerTestError.missingIA(candidates.map(\.path))
    }
}

private enum SmeltPackageRunnerTestError: Error {
    case missingIA([String])
}

private final class SmeltPackageRunnerTestBundleMarker: NSObject {}
