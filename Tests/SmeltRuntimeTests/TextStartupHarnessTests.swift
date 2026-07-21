import Foundation
import SmeltSchema
import XCTest

final class TextStartupHarnessTests: XCTestCase {
    func testPackageProfileSuppliesTraceFirstTokenThreshold() throws {
        let fixture = try HarnessFixture(fakeSmeltBody: Self.fakeSmeltBody(prefillMS: "40.0"))
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 2)
        XCTAssertTrue(result.stdout.contains("text-profile-gate text.decode-prefill-startup"), result.stdout)
        XCTAssertTrue(
            result.stdout.contains("text-profile-max-bound trace_first_token_ms 100ms"),
            result.stdout
        )
        XCTAssertTrue(result.stderr.contains("text-startup-gates failed"), result.stderr)
        XCTAssertTrue(
            result.stderr.contains("first trace first token 105.000ms > 100.000ms"),
            result.stderr
        )
        XCTAssertTrue(
            result.stdout.contains("text-profile-required-output-metric trace_first_token_ms"),
            result.stdout
        )
    }

    func testPackageProfileRequiresDeclaredTraceLabels() throws {
        let fixture = try HarnessFixture(fakeSmeltBody: Self.fakeSmeltBody(includeTokenizerLabel: false))
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("missing startup trace labels required by --max-trace-first-ms: tokenizer load"),
            result.stderr
        )
    }

    func testPackageProfileRequiresTraceFirstTokenMetric() throws {
        let fixture = try HarnessFixture(
            fakeSmeltBody: Self.fakeSmeltBody(),
            manifest: Self.packageProfileManifest(
                requiredOutputMetrics: Self.canonicalRequiredMetrics.filter {
                    $0 != "trace_first_token_ms"
                }
            )
        )
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "text package performance_profile missing canonical required output metric: trace_first_token_ms"
            ),
            result.stderr
        )
    }

    func testPackageProfileRequiresCanonicalTraceFirstBoundBeforeBenchmark() throws {
        let fixture = try HarnessFixture(
            fakeSmeltBody: Self.fakeSmeltBody(includeTimingLine: false),
            manifest: Self.packageProfileManifest(
                includeMaxBound: false
            )
        )
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "text package performance_profile missing canonical max-bound: trace_first_token_ms"
            ),
            result.stderr
        )
    }

    func testPackageProfileRejectsStaleCanonicalTraceLabelsBeforeBenchmark() throws {
        let fixture = try HarnessFixture(
            fakeSmeltBody: Self.fakeSmeltBody(),
            manifest: Self.packageProfileManifest(requiredTraceLabels: [
                "exec -> main (dyld)",
                "SmeltModel init (total)",
            ])
        )
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "text package performance_profile missing canonical required trace label: tokenizer load"
            ),
            result.stderr
        )
        XCTAssertFalse(result.stdout.contains("ok"), result.stdout)
    }

    func testPackageProfileRequiresGeneratedTokenEvidence() throws {
        let fixture = try HarnessFixture(
            fakeSmeltBody: Self.fakeSmeltBody(includeGeneratedTokens: false)
        )
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("missing generated token IDs required by package profile"),
            result.stderr
        )
    }

    func testPackageProfileDoesNotForceTimedPackageIntegrityVerification() throws {
        let fixture = try HarnessFixture(
            fakeSmeltBody: Self.fakeSmeltBody(forbidVerifyPackageEnvironment: true)
        )
        defer { fixture.cleanup() }

        let result = try fixture.runPackageProfile()

        XCTAssertEqual(result.status, 0, result.stderr)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: fixture.outputJSON)) as? [String: Any]
        )
        let packageRealpath = try Self.realpath(fixture.package.path)
        XCTAssertEqual(json["kind"] as? String, "smelt.module.text_startup_evidence")
        XCTAssertEqual(json["package"] as? String, packageRealpath)
        XCTAssertEqual(json["package_realpath"] as? String, packageRealpath)
        XCTAssertEqual(try XCTUnwrap(json["manifest_sha256"] as? String).count, 64)
        XCTAssertEqual(json["package_profile"] as? Bool, true)
        XCTAssertEqual(json["profile_gate"] as? String, "text.decode-prefill-startup")
    }

    private static func packageProfileManifest() -> String {
        packageProfileManifest(requiredOutputMetrics: canonicalRequiredMetrics)
    }

    private static let canonicalRequiredMetrics =
        SmeltPackagePerformanceMetricName.textDecodePrefillStartupRequired

    private static let canonicalRequiredTraceLabels =
        SmeltPackagePerformanceTraceLabel.textDecodePrefillStartupRequired

    private static func packageProfileManifest(
        requiredOutputMetrics: [String] = canonicalRequiredMetrics,
        requiredTraceLabels: [String] = canonicalRequiredTraceLabels,
        includeMaxBound: Bool = true
    ) -> String {
        let metrics = requiredOutputMetrics
            .map { #"                "\#($0)""# }
            .joined(separator: ",\n")
        let labels = requiredTraceLabels
            .map { #"                "\#($0)""# }
            .joined(separator: ",\n")
        let maxBounds = includeMaxBound
            ? """
                      "max_bounds": [
                        {"metric": "trace_first_token_ms", "max": \(SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS), "unit": "ms"}
                      ]
              """
            : #"              "max_bounds": []"#
        return """
        {
          "validation": {
            "performance_gate": "text.decode-prefill-startup",
            "performance_profile": {
              "gate": "text.decode-prefill-startup",
              "command": "run",
              "required_output_metrics": [
        \(metrics)
              ],
              "required_trace_labels": [
        \(labels)
              ],
        \(maxBounds)
            }
          }
        }
        """
    }

    private static func fakeSmeltBody(
        prefillMS: String = "10.0",
        includeTokenizerLabel: Bool = true,
        includeTimingLine: Bool = true,
        includeGeneratedTokens: Bool = true,
        requireVerifyPackageEnvironment: Bool = false,
        forbidVerifyPackageEnvironment: Bool = false
    ) -> String {
        let profileScript = shellPrintLines(canonicalProfileTSV)
        let verifyPackageCheck = requireVerifyPackageEnvironment
            ? """
              if [[ "${SMELT_VERIFY_PACKAGE:-}" != "1" ]]; then
                echo missing SMELT_VERIFY_PACKAGE >&2
                exit 88
              fi
              """
            : forbidVerifyPackageEnvironment
            ? """
              if [[ "${SMELT_VERIFY_PACKAGE:-}" == "1" ]]; then
                echo unexpected SMELT_VERIFY_PACKAGE >&2
                exit 88
              fi
              """
            : ""
        var stderrLines = [
            "startup:   +10.0ms  exec -> main (dyld)",
            "startup:   +55.0ms  SmeltModel init (total)",
            "Prompt tokens: 1",
        ]
        if includeGeneratedTokens {
            stderrLines.append("Generated token IDs: [1]")
        }
        if includeTokenizerLabel {
            stderrLines.insert("startup:   +0.0ms  tokenizer load", at: 1)
        }
        if includeTimingLine {
            stderrLines.append("Timing: prefill \(prefillMS)ms, generate 1.0ms, 1.0 tok/s")
        }
        let body = stderrLines
            .map { "echo '\($0)' >&2" }
            .joined(separator: "\n")
        return """
        #!/usr/bin/env bash
        if [[ "$1 $2 $3" == "lab package-profile \(SmeltPackagePerformanceGateID.textDecodePrefillStartup)" ]]; then
          [[ "$3" == "\(SmeltPackagePerformanceGateID.textDecodePrefillStartup)" ]] || {
            echo "unexpected cam profile: $*" >&2
            exit 97
          }
        \(profileScript)
          exit 0
        fi
        \(verifyPackageCheck)
        \(body)
        echo ok
        exit 0
        """
    }

    private static var canonicalProfileTSV: String {
        let profile = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        var lines = [
            "gate\t\(profile.gate)",
            "command\t\(profile.command.rawValue)",
        ]
        for label in profile.requiredTraceLabels {
            lines.append("required_trace_label\t\(label)")
        }
        for metric in profile.requiredOutputMetrics {
            lines.append("required_output_metric\t\(metric)")
        }
        for bound in profile.maxBounds {
            lines.append("max_bound\t\(bound.metric)\t\(formatted(bound.max))\t\(bound.unit)")
        }
        return lines.joined(separator: "\n")
    }

    private static func formatted(_ value: Double) -> String {
        String(format: "%g", value)
    }

    private static func shellPrintLines(_ value: String) -> String {
        value.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "          printf '%s\\n' '\(shellSingleQuote(String($0)))'" }
            .joined(separator: "\n")
    }

    private static func shellSingleQuote(_ value: String) -> String {
        value.replacingOccurrences(of: "'", with: "'\"'\"'")
    }

    private static func run(
        _ executable: String,
        _ arguments: [String],
        environment additions: [String: String] = [:]
    ) throws -> (status: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        additions.forEach { environment[$0.key] = $0.value }
        process.environment = environment

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: stdoutData, as: UTF8.self),
            String(decoding: stderrData, as: UTF8.self)
        )
    }

    private static func realpath(_ path: String) throws -> String {
        let result = try run(
            "/usr/bin/env",
            ["python3", "-c", "import os, sys; print(os.path.realpath(sys.argv[1]))", path]
        )
        XCTAssertEqual(result.status, 0, result.stderr)
        return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private struct HarnessFixture {
        let root: URL
        let script: URL
        let package: URL
        let tmp: URL
        let outputJSON: URL

        init(
            fakeSmeltBody: String,
            manifest: String = TextStartupHarnessTests.packageProfileManifest()
        ) throws {
            let fm = FileManager.default
            root = fm.temporaryDirectory
                .appendingPathComponent("smelt-text-startup-harness-\(UUID().uuidString)", isDirectory: true)
            let tools = root.appendingPathComponent("tools", isDirectory: true)
            let build = root
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
            package = root.appendingPathComponent("fake-text.smeltpkg", isDirectory: true)
            tmp = root.appendingPathComponent("tmp", isDirectory: true)
            outputJSON = root.appendingPathComponent("out.json")
            try [tools, build, package, tmp].forEach {
                try fm.createDirectory(at: $0, withIntermediateDirectories: true)
            }

            let repoRoot = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            let sourceScript = repoRoot
                .appendingPathComponent("tools/benchmark-text-startup.sh")
            script = tools.appendingPathComponent("benchmark-text-startup.sh")
            try fm.copyItem(at: sourceScript, to: script)

            try Data(manifest.utf8)
                .write(to: package.appendingPathComponent("manifest.json"))
            let fakeSmelt = build.appendingPathComponent("smelt")
            try Data(fakeSmeltBody.utf8).write(to: fakeSmelt)
            try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeSmelt.path)
        }

        func cleanup() {
            try? FileManager.default.removeItem(at: root)
        }

        func runPackageProfile() throws -> (status: Int32, stdout: String, stderr: String) {
            try TextStartupHarnessTests.run(
                "/bin/bash",
                [
                    script.path,
                    package.path,
                    "--skip-build",
                    "--iterations", "1",
                    "--warmup", "0",
                    "--use-package-profile",
                    "--output-json", outputJSON.path,
                ],
                environment: ["TMPDIR": tmp.path + "/"]
            )
        }
    }
}
