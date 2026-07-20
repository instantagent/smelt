import Foundation
import SmeltCompiler
import SmeltSchema
import XCTest

final class CAMCommandTests: XCTestCase {
    func testCamCheckAcceptsAllExamples() throws {
        for name in Self.exampleNames {
            let ir = registryModuleIR(name)
            let result = try Self.runAgent(["module", "check", Self.moduleURL(name).path])

            XCTAssertEqual(result.status, 0, "\(name): \(result.stderr)")
            XCTAssertTrue(result.stdout.contains("module\t\(ir.module.id)\n"), result.stdout)
            XCTAssertTrue(result.stdout.contains("semantic_sha256\t\(try ir.semanticSHA256())\n"), result.stdout)
            XCTAssertTrue(result.stdout.contains("export_abi_sha256\t\(try ir.exportABISHA256())\n"), result.stdout)
            XCTAssertEqual(result.stderr, "")
        }
    }

    func testCamCheckReportsTypedDS4ObligationsWithoutFailing() throws {
        let result = try Self.runAgent(["module", "check", Self.moduleURL("ds4_heavy_quant.cam").path])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stderr, "")
        for fragment in [
            "required_feature_codes\t",
            "unsupported_feature_codes\t",
            "required_obligation\t",
            "unsupported_obligation\t",
            "transformer.rope.yarn",
            "theta=1000000",
            "quant.storage.gptq",
            "pattern=model.layers.*.mlp.experts.*",
            "corpus_path_sha256=",
            "gate.quant-quality",
        ] {
            XCTAssertTrue(result.stdout.contains(fragment), "\(fragment): \(result.stdout)")
        }

        let obligationLines = result.stdout
            .split(separator: "\n")
            .filter { $0.contains("_obligation\t") }
            .joined(separator: "\n")
        for banned in [
            "DS4",
            "deepseek",
            "calibration/ds4-prompts.jsonl",
            ".cam",
        ] {
            XCTAssertFalse(obligationLines.contains(banned), "\(banned): \(obligationLines)")
        }
    }

    func testCamAdmissionJSONUsesSnakeCaseContract() throws {
        let admission = SmeltCAMFeatureAdmission(
            descriptor: try SmeltCAMPackageDescriptor(from: registryModuleIR("ds4_heavy_quant"))
        )
        let result = try Self.runAgent(["module", "admission", Self.moduleURL("ds4_heavy_quant.cam").path, "--json"])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stderr, "")
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(result.stdout.utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["schema"] as? String, SmeltCAMFeatureAdmission.currentSchema)
        XCTAssertEqual(object["stage"] as? String, SmeltCAMFeatureAdmission.preBridgeStage)
        XCTAssertEqual(object["required_obligation_ids"] as? [String], admission.requiredObligationIDs)
        XCTAssertEqual(object["unsupported_obligation_ids"] as? [String], admission.requiredObligationIDs)
        XCTAssertNil(object["requiredObligationIDs"])
        XCTAssertNil(object["unsupportedFeatures"])
        let obligations = try XCTUnwrap(object["required_obligations"] as? [[String: Any]])
        XCTAssertFalse(obligations.isEmpty)
        for key in ["canonical_id", "code", "scope", "parameters", "evidence"] {
            XCTAssertNotNil(obligations[0][key], "\(key): \(obligations[0])")
        }
    }

    func testCamIRJSONMatchesCanonicalBytes() throws {
        let ir = registryModuleIR("qwen35_text")
        let result = try Self.runAgent(["module", "ir", Self.moduleURL("qwen35_text.cam").path])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stdout, String(decoding: try ir.canonicalJSONData(), as: UTF8.self) + "\n")
        XCTAssertEqual(result.stderr, "")
    }

    func testCamUsageAndArgumentErrors() throws {
        var result = try Self.runAgent(["module"])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("Usage:"), result.stderr)

        result = try Self.runAgent(["module", "--help"])
        XCTAssertEqual(result.status, 0)
        XCTAssertTrue(result.stdout.contains("smelt module check <model.module.json>"), result.stdout)
        XCTAssertTrue(result.stdout.contains("smelt module admission <model.module.json> --json"), result.stdout)

        result = try Self.runAgent(["module", "profile"])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("unknown subcommand"), result.stderr)

        result = try Self.runAgent(["module", "ir", Self.moduleURL("qwen35_text.cam").path, "--runtime"])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("unknown option '--runtime'"), result.stderr)

        result = try Self.runAgent(["module", "ir", Self.moduleURL("qwen35_text.cam").path, "--json", "--hashes"])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("conflicting output options"), result.stderr)

        result = try Self.runAgent(["module", "admission", Self.moduleURL("qwen35_text.cam").path])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("Usage:"), result.stderr)
    }

    func testCamCheckReportsInvalidSyntax() throws {
        // `smelt module check` decodes `.module.json` IR; a malformed descriptor is
        // rejected as invalid JSON rather than parsed as grammar text.
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let bad = root.appendingPathComponent("bad.module.json")
        try Data("""
        { "module": "bad", banana }
        """.utf8).write(to: bad)

        let result = try Self.runAgent(["module", "check", bad.path])
        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("smelt module check failed"), result.stderr)
    }

    private static let exampleNames = [
        "ds4_heavy_quant.cam",
        "qwen35_fast.cam",
        "qwen35_reasoner.cam",
        "qwen35_text.cam",
        "qwen3_tts.cam",
    ]

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testCamGrammarInputFailsClosedAtCommandBoundary() throws {
        // Phase C removed the .cam authoring grammar. A .cam input path must be
        // rejected END TO END with the deleted-format diagnostic — a real
        // invocation, not just a source scan. Covers the commands that take a
        // bare path (so the suffix guard is reached without other required args).
        // `module admission --json` reaches the same shared guard once its args
        // validate. runAgent skips when
        // the binary isn't built, and runs in CI/release where it matters.
        let camPath = "legacy-model.cam"
        for argv in [
            ["build", camPath],
            ["module", "check", camPath],
            ["module", "ir", camPath],
        ] {
            let result = try Self.runAgent(argv)
            XCTAssertNotEqual(
                result.status, 0,
                "smelt \(argv.joined(separator: " ")) should fail closed; stdout: \(result.stdout)"
            )
            XCTAssertTrue(
                result.stderr.contains("the .cam authoring grammar was removed"),
                "smelt \(argv.joined(separator: " ")) stderr: \(result.stderr)"
            )
        }
    }

    func testTtsBuildCommandIsRemoved() throws {
        // The specialized TTS front door was removed with the module migration;
        // TTS packages now use the same generic `smelt build` path as every module.
        let result = try Self.runAgent([
            "tts-build",
            "some-checkpoint-dir",
            "--module", "Models/qwen3_tts.module.json",
            "--cam", "Models/qwen3_tts.module.json",
            "--output", "out.smeltpkg",
        ])
        XCTAssertNotEqual(result.status, 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("Unknown command: tts-build"), "stderr: \(result.stderr)")
    }

    func testLingerWorkerRejectsLegacyCamLingerIdentityFlag() throws {
        // F1c: `--cam-linger-identity` was renamed to `--module-linger-identity`.
        // A stray legacy flag must be loud-rejected before the package capability
        // load.
        let result = try Self.runAgent([
            "linger-worker",
            "some-package.smeltpkg",
            "--socket", "/tmp/smelt-linger-legacy-flag.sock",
            "--cam-linger-identity", "abc123",
        ])
        XCTAssertNotEqual(result.status, 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("--cam-linger-identity"), "stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("--module-linger-identity"), "stderr: \(result.stderr)")
    }

    /// The checked-in authored IR artifact (`Models/<id>.module.json`) that the
    /// `smelt module` commands decode, derived from the `<id>.cam` oracle name.
    private static func moduleURL(_ name: String) -> URL {
        let id = name.hasSuffix(".cam") ? String(name.dropLast(4)) : name
        return repoRoot
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("\(id).module.json")
    }

    private static func runAgent(
        _ arguments: [String]
    ) throws -> (status: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = try iaExecutable()
        process.arguments = arguments

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        return (
            process.terminationStatus,
            String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self),
            String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        )
    }

    private static func iaExecutable() throws -> URL {
        let candidates = [
            Bundle(for: CAMCommandTests.self).bundleURL
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
        throw XCTSkip("smelt executable has not been built")
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-cam-command-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }
}
