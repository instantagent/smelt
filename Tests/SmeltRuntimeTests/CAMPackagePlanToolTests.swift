import Foundation
import XCTest

final class CAMPackagePlanToolTests: XCTestCase {
    func testBuildCamPackagePlanUsesRealFiveColumnPlanForAllRows() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        let result = try fixture.runBuildPlan(["--skip-tool-build", "--skip-source-build", "--all"])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            result.stdout,
            """
            module build qwen35_fast
            module build qwen35_text
            module build qwen3_tts

            """
        )

        let log = try fixture.readLog()
        XCTAssertTrue(
            log.contains("build Models/qwen35_fast.module.json --output artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg --module-artifact-root artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B-build-source.smeltpkg --module-build-evidence-json .build/module-build-evidence/qwen35-fast-build.json"),
            log
        )
        XCTAssertTrue(
            log.contains("build Models/qwen3_tts.module.json --output qwen3-tts.smeltpkg --module-artifact-root qwen3-tts-build-source.smeltpkg --module-build-evidence-json .build/module-build-evidence/qwen3-tts-build.json"),
            log
        )
        XCTAssertFalse(log.contains("qwen35-cam-packages"), log)
    }

    func testBuildCamPackagePlanSelectsRowsByIDOnly() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        let result = try fixture.runBuildPlan([
            "--skip-tool-build",
            "--skip-source-build",
            "--id", "qwen35_text",
            "--id", "qwen3_tts",
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            result.stdout,
            """
            module build qwen35_text
            module build qwen3_tts

            """
        )
        let log = try fixture.readLog()
        XCTAssertTrue(log.contains("build Models/qwen35_text.module.json"), log)
        XCTAssertTrue(log.contains("build Models/qwen3_tts.module.json"), log)
        XCTAssertFalse(log.contains("build Models/qwen35_fast.module.json"), log)
        XCTAssertFalse(log.contains("build Models/qwen35_reasoner.module.json"), log)
    }

    func testBuildCamPackagePlanMaterializesSourceBeforePackageBuildByDefault() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = try fixture.writeSourceInput("qwen35-text-source")

        let result = try fixture.runBuildPlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            result.stdout,
            """
            module source build qwen35_text
            module build qwen35_text

            """
        )
        let log = try fixture.readLog()
        XCTAssertTrue(
            log.contains("build Models/qwen35_text.module.json --module-source-package --weights-dir \(source.path) --shader-dir Resources/Shaders --output .build/module-source-build/qwen35_text --trace-mode stripped-markers"),
            log
        )
        XCTAssertTrue(
            log.contains("build Models/qwen35_text.module.json --output artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B.smeltpkg --module-artifact-root artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B-build-source.smeltpkg --module-build-evidence-json .build/module-build-evidence/qwen35-text-build.json"),
            log
        )
    }

    func testBuildCamSourcePlanUsesCamSourceInputEnvAndMovesTextSourcePackage() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = try fixture.writeSourceInput("qwen35-text-source")

        let result = try fixture.runSourcePlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stdout, "module source build qwen35_text\n")
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: fixture.root
                    .appendingPathComponent("artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B-build-source.smeltpkg")
                    .path
            )
        )
    }

    func testBuildCamSourcePlanMissingInputNamesCamLocatorAndEnv() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        let result = try fixture.runSourcePlan(["--skip-tool-build", "--id", "qwen35_text"])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("Missing module source input for qwen35_text (Qwen/Qwen3.5-2B)"),
            result.stderr
        )
        XCTAssertTrue(
            result.stderr.contains("Set SMELT_MODULE_SOURCE_QWEN35_TEXT to a local checkpoint or optimized artifact path."),
            result.stderr
        )
        XCTAssertTrue(
            result.stderr.contains("bash tools/fetch-module-source-inputs.sh --id qwen35_text"),
            result.stderr
        )
    }

    func testBuildCamSourcePlanRejectsEmptySourceInput() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = fixture.root.appendingPathComponent("empty-source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let result = try fixture.runSourcePlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("module source input for qwen35_text has no carried tensor artifact"),
            result.stderr
        )
        XCTAssertTrue(
            result.stderr.contains("Expected *.safetensors, *.safetensors.index.json, *.npz, or *.bin."),
            result.stderr
        )
        XCTAssertEqual(try fixture.readLog(), "")
    }

    func testBuildCamSourcePlanRejectsIncompleteIndexedCheckpoint() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = fixture.root.appendingPathComponent(
            "partial-indexed-source",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data("one shard only".utf8).write(
            to: source.appendingPathComponent("model-00001-of-00002.safetensors")
        )
        try Data(#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#.utf8).write(
            to: source.appendingPathComponent("model.safetensors.index.json")
        )

        let result = try fixture.runSourcePlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "Indexed safetensors checkpoint is incomplete: 1 missing/empty shard(s) "
                    + "from model.safetensors.index.json: model-00002-of-00002.safetensors"
            ),
            result.stderr
        )
        XCTAssertEqual(try fixture.readLog(), "")
    }

    func testBuildCamSourcePlanRequiresCheckpointMapForEveryQwenRow() throws {
        let rows = [
            ("qwen35_fast", "qwen35_fast.module.json", "SMELT_MODULE_SOURCE_QWEN35_FAST"),
            ("qwen35_text", "qwen35_text.module.json", "SMELT_MODULE_SOURCE_QWEN35_TEXT"),
        ]

        for (rowID, moduleFile, envName) in rows {
            let fixture = try Fixture()
            defer { fixture.cleanup() }
            let source = try fixture.writeSourceInput("\(rowID)-source")
            try fixture.replaceText(
                in: "Models/\(moduleFile)",
                "\"checkpointMap\" : \"hf.qwen\",",
                with: ""
            )

            let result = try fixture.runSourcePlan(
                ["--skip-tool-build", "--id", rowID],
                environment: [envName: source.path]
            )

            XCTAssertEqual(result.status, 1, rowID)
            XCTAssertTrue(
                result.stderr.contains(
                    "module source row \(rowID) uses module source package build but module source weights has no checkpoint-map."
                ),
                "\(rowID): \(result.stderr)"
            )
            XCTAssertEqual(try fixture.readLog(), "", rowID)
        }
    }

    func testBuildCamSourcePlanUsesCamSourcePackageForEveryModuleRow() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        let result = try fixture.runSourcePlan([
            "--skip-tool-build",
            "--id", "qwen35_fast",
            "--id", "qwen35_text",
            "--id", "qwen3_tts",
        ], environment: [
            "SMELT_MODULE_SOURCE_QWEN35_FAST": try fixture.writeSourceInput("qwen35-fast-source").path,
            "SMELT_MODULE_SOURCE_QWEN35_TEXT": try fixture.writeSourceInput("qwen35-text-source").path,
            "SMELT_MODULE_SOURCE_QWEN3_TTS": try fixture.writeSourceInput("qwen3-tts-source").path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        let log = try fixture.readLog()
        XCTAssertTrue(log.contains("build Models/qwen35_fast.module.json --module-source-package"), log)
        XCTAssertTrue(log.contains("build Models/qwen35_text.module.json --module-source-package"), log)
        XCTAssertTrue(log.contains("build Models/qwen3_tts.module.json --module-source-package"), log)
    }

    func testBuildCamSourcePlanRejectsCamSourcePackageFlagOnNonCAMBuildInput() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = try fixture.writeSourceInput("qwen35-text-source")
        try fixture.replaceText(
            in: "Models/source-build-plan.tsv",
            "build|{module}|--module-source-package",
            with: "build|tools/qwen35-2b-not-the-cam-row.txt|--module-source-package"
        )

        let result = try fixture.runSourcePlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "module source row qwen35_text uses --module-source-package but build input is not the module row path."
            ),
            result.stderr
        )
        XCTAssertEqual(try fixture.readLog(), "")
    }

    func testBuildCamSourcePlanReadsCheckpointMapFromWeightsSourceNotOtherSources() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = try fixture.writeSourceInput("qwen35-text-source")
        // The tool decodes the module descriptor structurally: the checkpoint-map
        // must be read off the `weights` source. A checkpointMap on a different
        // source (here the tokenizer) must not count.
        try fixture.replaceText(
            in: "Models/qwen35_text.module.json",
            "\"checkpointMap\" : \"hf.qwen\",",
            with: ""
        )
        try fixture.replaceText(
            in: "Models/qwen35_text.module.json",
            "\"id\" : \"tokenizer\",",
            with: "\"checkpointMap\" : \"hf.qwen\",\n      \"id\" : \"tokenizer\","
        )

        let result = try fixture.runSourcePlan(
            ["--skip-tool-build", "--id", "qwen35_text"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "module source row qwen35_text uses module source package build but module source weights has no checkpoint-map."
            ),
            result.stderr
        )
        XCTAssertEqual(try fixture.readLog(), "")
    }

    func testFetchCamSourceInputsUsesCamWeightsDeclarationAndSourcePlanTarget() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = fixture.root.appendingPathComponent("download-target", isDirectory: true)

        let result = try fixture.runFetchPlan(
            ["--id", "qwen35_text", "--dry-run"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stdout, "module source fetch qwen35_text -> \(source.path)\n")
        let log = try fixture.readLog()
        XCTAssertTrue(
            log.contains("download Qwen/Qwen3.5-2B --revision main --local-dir \(source.path) --dry-run"),
            log
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: source.path),
            "dry-run fetch must not leave a new source target behind"
        )
    }

    func testFetchCamSourceInputsDryRunKeepsExistingTargetDirectory() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let source = fixture.root.appendingPathComponent("existing-download-target", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let result = try fixture.runFetchPlan(
            ["--id", "qwen35_text", "--dry-run"],
            environment: ["SMELT_MODULE_SOURCE_QWEN35_TEXT": source.path]
        )

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: source.path),
            "dry-run fetch must not remove an existing source target"
        )
    }

    func testBuildCamPackagePlanRejectsGroupSelection() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }

        let result = try fixture.runBuildPlan([
            "--skip-tool-build",
            "--skip-source-build",
            "--group", "qwen35",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("Usage: bash tools/build-module-package-plan.sh"), result.stderr)
        XCTAssertEqual(try fixture.readLog(), "")
    }

    func testBuildCamPackagePlanRejectsOldSixColumnRows() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let plan = fixture.root.appendingPathComponent("old-plan.tsv")
        try Data(
            """
            # group\tid\tmodule_path\tartifact_root\toutput_package\tevidence_json
            qwen\tqwen35_fast\tModels/qwen35_fast.module.json\tartifact-root\toutput.smeltpkg\tevidence.json

            """.utf8
        ).write(to: plan)

        let result = try fixture.runBuildPlan([
            "--skip-tool-build",
            "--skip-source-build",
            "--plan", plan.path,
            "--all",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("Invalid module package build plan row for qwen: too many columns"),
            result.stderr
        )
        XCTAssertEqual(try fixture.readLog(), "")
    }

    private struct Fixture {
        let root: URL
        let log: URL

        init() throws {
            let fm = FileManager.default
            root = fm.temporaryDirectory
                .appendingPathComponent("smelt-cam-package-plan-\(UUID().uuidString)", isDirectory: true)
            log = root.appendingPathComponent("calls.log")

            let repoRoot = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            let tools = root.appendingPathComponent("tools", isDirectory: true)
            let models = root
                .appendingPathComponent("Models", isDirectory: true)
            let releaseBuild = root
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("release", isDirectory: true)
            try fm.createDirectory(at: tools, withIntermediateDirectories: true)
            try fm.createDirectory(at: models, withIntermediateDirectories: true)
            try fm.createDirectory(at: releaseBuild, withIntermediateDirectories: true)
            try fm.copyItem(
                at: repoRoot.appendingPathComponent("tools/build-module-package-plan.sh"),
                to: tools.appendingPathComponent("build-module-package-plan.sh")
            )
            try fm.copyItem(
                at: repoRoot.appendingPathComponent("tools/build-module-source-plan.sh"),
                to: tools.appendingPathComponent("build-module-source-plan.sh")
            )
            try fm.copyItem(
                at: repoRoot.appendingPathComponent("tools/fetch-module-source-inputs.sh"),
                to: tools.appendingPathComponent("fetch-module-source-inputs.sh")
            )
            try fm.copyItem(
                at: repoRoot.appendingPathComponent("Models/package-build-plan.tsv"),
                to: models.appendingPathComponent("package-build-plan.tsv")
            )
            try fm.copyItem(
                at: repoRoot.appendingPathComponent("Models/source-build-plan.tsv"),
                to: models.appendingPathComponent("source-build-plan.tsv")
            )
            for module in [
                "qwen35_fast.module.json",
                "qwen35_text.module.json",
                "qwen3_tts.module.json",
            ] {
                try fm.copyItem(
                    at: repoRoot.appendingPathComponent("Models/\(module)"),
                    to: models.appendingPathComponent(module)
                )
            }
            try Self.writeExecutable(releaseBuild.appendingPathComponent("smelt"), body: """
                #!/usr/bin/env bash
                printf '%s\\n' "$*" >> "$SMELT_TEST_LOG"
                out=""
                prev=""
                for arg in "$@"; do
                  if [[ "$prev" == "--output" ]]; then
                    out="$arg"
                    break
                  fi
                  prev="$arg"
                done
                if [[ -n "$out" ]]; then
                  if [[ "$1" == "build" && "$2" == Models/*.module.json && " $* " == *" --module-source-package "* ]]; then
                    mkdir -p "$out/staged.smeltpkg"
                  fi
                fi
                """)
            let bin = root.appendingPathComponent("bin", isDirectory: true)
            try fm.createDirectory(at: bin, withIntermediateDirectories: true)
            try Self.writeExecutable(bin.appendingPathComponent("hf"), body: """
                #!/usr/bin/env bash
                printf '%s\\n' "$*" >> "$SMELT_TEST_LOG"
                target=""
                prev=""
                dry_run=0
                for arg in "$@"; do
                  if [[ "$prev" == "--local-dir" ]]; then
                    target="$arg"
                  fi
                  if [[ "$arg" == "--dry-run" ]]; then
                    dry_run=1
                  fi
                  prev="$arg"
                done
                if [[ "$dry_run" == 1 && -n "$target" ]]; then
                  mkdir -p "$target/.cache/huggingface/download"
                  touch "$target/.cache/huggingface/.gitignore"
                fi
                """)
        }

        func cleanup() {
            try? FileManager.default.removeItem(at: root)
        }

        func readLog() throws -> String {
            guard FileManager.default.fileExists(atPath: log.path) else { return "" }
            return String(decoding: try Data(contentsOf: log), as: UTF8.self)
        }

        func writeSourceInput(_ name: String) throws -> URL {
            let url = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try Data("stub tensor artifact".utf8)
                .write(to: url.appendingPathComponent("model.safetensors"))
            return url
        }

        func replaceText(in relativePath: String, _ needle: String, with replacement: String) throws {
            let url = root.appendingPathComponent(relativePath)
            let text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
            XCTAssertTrue(text.contains(needle), "Fixture file \(relativePath) did not contain \(needle)")
            try Data(text.replacingOccurrences(of: needle, with: replacement).utf8).write(to: url)
        }

        func runBuildPlan(
            _ arguments: [String],
            environment: [String: String] = [:]
        ) throws -> (status: Int32, stdout: String, stderr: String) {
            try Self.run(
                "/bin/bash",
                [root.appendingPathComponent("tools/build-module-package-plan.sh").path] + arguments,
                environment: ["HOME": root.path, "SMELT_TEST_LOG": log.path]
                    .merging(environment) { _, new in new },
                currentDirectory: root
            )
        }

        func runSourcePlan(
            _ arguments: [String],
            environment: [String: String] = [:]
        ) throws -> (status: Int32, stdout: String, stderr: String) {
            try Self.run(
                "/bin/bash",
                [root.appendingPathComponent("tools/build-module-source-plan.sh").path] + arguments,
                environment: ["HOME": root.path, "SMELT_TEST_LOG": log.path]
                    .merging(environment) { _, new in new },
                currentDirectory: root
            )
        }

        func runFetchPlan(
            _ arguments: [String],
            environment: [String: String] = [:]
        ) throws -> (status: Int32, stdout: String, stderr: String) {
            let bin = root.appendingPathComponent("bin").path
            let path = "\(bin):\(ProcessInfo.processInfo.environment["PATH"] ?? "")"
            return try Self.run(
                "/bin/bash",
                [root.appendingPathComponent("tools/fetch-module-source-inputs.sh").path] + arguments,
                environment: ["HOME": root.path, "SMELT_TEST_LOG": log.path, "PATH": path]
                    .merging(environment) { _, new in new },
                currentDirectory: root
            )
        }

        private static func run(
            _ executable: String,
            _ arguments: [String],
            environment: [String: String],
            currentDirectory: URL
        ) throws -> (status: Int32, stdout: String, stderr: String) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.currentDirectoryURL = currentDirectory
            var processEnvironment = ProcessInfo.processInfo.environment
            environment.forEach { processEnvironment[$0.key] = $0.value }
            process.environment = processEnvironment

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

        private static func writeExecutable(_ url: URL, body: String) throws {
            try Data(body.utf8).write(to: url)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: url.path
            )
        }
    }
}
