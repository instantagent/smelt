import Foundation
import Metal
@testable import SmeltCompiler
@testable import SmeltSchema
import XCTest

final class ServeDescriptorDispatchTests: XCTestCase {
    func testPerformanceCommandsHaveCleanBreakToLabNamespace() throws {
        let retired = try Self.runRawSmelt(["verify", "/tmp/model.smeltpkg"])
        XCTAssertEqual(retired.status, 1)
        XCTAssertTrue(retired.stderr.contains("Unknown command: verify"), retired.stderr)

        let lab = try Self.runRawSmelt(["lab", "--help"])
        XCTAssertEqual(lab.status, 0, lab.stderr)
        XCTAssertTrue(lab.stdout.contains("smelt lab verify"), lab.stdout)
        XCTAssertTrue(lab.stdout.contains("smelt lab bench verify"), lab.stdout)
        XCTAssertTrue(lab.stdout.contains("smelt lab profile verify"), lab.stdout)
        XCTAssertTrue(lab.stdout.contains("smelt lab sweep qmm"), lab.stdout)
    }

    func testRunCAMTextExportPreflightsInventoryWithoutManifestBridge() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-text-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary-text-to-text-run-runtime"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testRunCAMTextDeclaredValueConsumesFlagShapedTokenBeforeLingerPreflight() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-text-declared-flag-value-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Data(Self.textStyleArgsJSON.utf8)
            .write(to: package.appendingPathComponent(SmeltPackageInterface.fileName))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "run",
            package.path,
            "--style",
            "--linger",
            "5",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no CAM export satisfies linger request"), result.stderr)
        XCTAssertFalse(result.stderr.contains("--linger must be"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testRunOnceCAMTextExportPreflightsInventoryWithoutLLMDescriptor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-once-cam-text-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-agent-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "run",
            package.path,
            "--once",
            "--prompt",
            "hello",
            "--max-tokens",
            "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected llm package"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testVerifyCAMTextExportPreflightsInventoryWithoutLLMDescriptor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-verify-cam-text-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-verify-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "verify",
            package.path,
            "--decode-iterations",
            "1",
            "--prefill-iterations",
            "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected llm package"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Verify failed"), result.stderr)
    }

    func testVerifyRejectsRetiredModelGateFlagsBeforeInventory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-verify-retired-gates-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-verify-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        for flag in ["--gate-qwen35", "--gate-llama32"] {
            let result = try Self.runSmelt([
                "verify",
                package.path,
                flag,
                "--decode-iterations",
                "1",
                "--prefill-iterations",
                "1",
            ])

            XCTAssertEqual(result.status, 1, flag)
            XCTAssertTrue(
                result.stderr.contains("\(flag) was removed"),
                "\(flag): \(result.stderr)"
            )
            XCTAssertFalse(result.stderr.contains("module package inventory missing files"), flag)
            XCTAssertFalse(result.stderr.contains("Verify failed"), flag)
        }
    }

    func testRunCAMAudioExportFailsClosedAgainstLLMManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-audio-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary-text-to-pcm-run-runtime"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-pcm-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testRunCAMAudioRejectsCodecOnlyGraphBeforeFrontendLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-codec-only-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-codec-only.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.qwen3TTSCodecOnlyManifest().encoded()
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)
        try Self.writeQwen3TTSFacadeInventoryFiles(to: package)

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("Qwen3-TTS CAM package must declare a text-to-PCM graph"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("config.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testCAMTextRouteIgnoresModuleIDAndGraphTagsForInventoryPreflight() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-ignores-tags-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen35_text.cam") { object in
            object["moduleID"] = "renamed_graph"
            var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
            for index in nodes.indices {
                nodes[index]["annotations"] = [
                    ["key": "tag", "value": "do-not-route-on-this"],
                ]
            }
            object["graphNodes"] = nodes
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no manifest bridge route"), result.stderr)
    }

    func testCAMTextRouteAcceptsNativeSignedStorageForInventoryPreflight() throws {
        for storageFormat in ["binary-1", "ternary-2"] {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "agent-module-contract-signed-storage-\(storageFormat)-\(UUID().uuidString)",
                    isDirectory: true
                )
            let package = root.appendingPathComponent("fake-module-text.smeltpkg", isDirectory: true)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }

            try Data(Self.qwen3TTSManifest.utf8)
                .write(to: package.appendingPathComponent("manifest.json"))
            try Self.mutatedCAMDescriptorData("qwen35_text.cam") { object in
                var rules = try XCTUnwrap(object["quantization"] as? [[String: Any]])
                let defaultRule = try XCTUnwrap(rules.indices.first {
                    rules[$0]["action"] as? String == "default"
                })
                rules[defaultRule]["storage"] = [
                    "storageFormat": storageFormat,
                    "groupSize": 128,
                ]
                object["quantization"] = rules
            }
            .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

            let result = try Self.runSmelt([
                "run",
                package.path,
                "hello",
            ])

            XCTAssertEqual(result.status, 1, storageFormat)
            XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
            XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
            XCTAssertFalse(result.stderr.contains("no CAM runtime route"), result.stderr)
        }
    }

    func testCAMManifestBridgeRouteIgnoresAudioGraphTags() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-audio-tags-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
            for index in nodes.indices {
                var annotations = nodes[index]["annotations"] as? [[String: Any]] ?? []
                annotations.removeAll { $0["key"] as? String == "tag" }
                annotations.append(["key": "tag", "value": "mutated-routing-metadata"])
                nodes[index]["annotations"] = annotations
            }
            object["graphNodes"] = nodes
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-pcm-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no manifest bridge route"), result.stderr)
    }

    func testCAMManifestBridgeRouteRejectsCompileTargetDriftBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-compile-drift-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen35_text.cam") { object in
            var requirements = try XCTUnwrap(object["compileRequirements"] as? [[String: Any]])
            let index = try XCTUnwrap(requirements.indices.first {
                requirements[$0]["key"] as? String == "target"
            })
            requirements[index]["value"] = "cpu"
            object["compileRequirements"] = requirements
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("no CAM runtime route"), result.stderr)
        XCTAssertTrue(result.stderr.contains("export 'generate' flow 'generate'"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testCAMFeatureAdmissionRejectsHeavyQuantBeforeRuntimeBridge() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-admission-heavy-quant-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-heavy.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("ds4_heavy_quant.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let tracePath = root.appendingPathComponent("fake.smttrace")
        try Data("not a trace\n".utf8).write(to: tracePath)
        let suitePath = root.appendingPathComponent("suite.json")
        try Data(
            """
            {
              "schemaVersion": 1,
              "cases": [
                {
                  "name": "ds4-heavy-quant",
                  "golden": "goldens/ds4.smttrace"
                }
              ]
            }
            """.utf8
        ).write(to: suitePath)
        let commandCases: [(name: String, verb: String, arguments: [String])] = [
            ("run", "run", ["run", package.path, "hello"]),
            (
                "bench",
                "lab bench decode",
                ["bench", package.path, "--iterations", "1", "--warmup", "0"]
            ),
            ("serve", "serve", ["serve", package.path, "--transport", "invalid"]),
            ("trace inspect", "lab trace", ["trace", "inspect", package.path]),
            (
                "trace record",
                "lab trace",
                ["trace", "record", package.path, "--case-text", "hello"]
            ),
            (
                "trace verify",
                "lab trace",
                ["trace", "verify", package.path, "--golden", tracePath.path]
            ),
            ("trace replay", "lab trace", ["trace", "replay", package.path, tracePath.path]),
            (
                "trace suite",
                "lab trace",
                ["trace", "suite", suitePath.path, "--package", package.path]
            ),
            (
                "linger-worker",
                "linger-worker",
                [
                    "linger-worker",
                    package.path,
                    "--socket",
                    socket.path,
                    "--idle",
                    "1",
                ]
            ),
        ]

        for commandCase in commandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            Self.assertDS4FeatureAdmissionFailure(
                result,
                verb: commandCase.verb,
                label: commandCase.name
            )
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testCAMManifestBridgeRouteRejectsAudioGraphDeliveryDriftBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-audio-drift-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
            let index = try XCTUnwrap(nodes.indices.first {
                nodes[$0]["nodeID"] as? String == "codec-decoder"
            })
            nodes[index]["implementation"] = "native"
            nodes[index]["blockID"] = NSNull()
            nodes[index]["annotations"] = []
            object["graphNodes"] = nodes
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("no CAM runtime route"), result.stderr)
        XCTAssertTrue(result.stderr.contains("export 'synth' flow 'synth'"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testCAMRuntimeRouteRejectsShapeOnlyCompiledAudioEmitterBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-shape-only-emitter-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
            let decoderIndex = try XCTUnwrap(nodes.indices.first {
                nodes[$0]["nodeID"] as? String == "codec-decoder"
            })
            nodes[decoderIndex]["implementation"] = "native"
            nodes[decoderIndex]["blockID"] = NSNull()
            nodes[decoderIndex]["annotations"] = []
            nodes.append([
                "nodeID": "shape-only-audio-emitter",
                "implementation": "compiled",
                "blockID": "codec-decoder",
                "inputs": [],
                "outputs": [[
                    "portName": "audio",
                    "optional": false,
                    "type": [
                        "typeName": "pcm",
                        "attributes": ["dtype": "f32", "rate": "24khz"],
                    ],
                ]],
                "annotations": [],
            ])
            object["graphNodes"] = nodes

            var flows = try XCTUnwrap(object["flows"] as? [[String: Any]])
            let flowIndex = try XCTUnwrap(flows.indices.first {
                flows[$0]["flowID"] as? String == "synth"
            })
            var phases = try XCTUnwrap(flows[flowIndex]["phases"] as? [[String: Any]])
            var renderCalls = try XCTUnwrap(phases[2]["calls"] as? [[String: Any]])
            renderCalls.append([
                "callType": "node",
                "nodeID": "shape-only-audio-emitter",
            ])
            phases[2]["calls"] = renderCalls
            flows[flowIndex]["phases"] = phases
            flows[flowIndex]["emit"] = [[
                "endpointType": "moduleOutput",
                "name": "audio",
            ]]
            object["flows"] = flows
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("no CAM runtime route"), result.stderr)
        XCTAssertTrue(result.stderr.contains("export 'synth' flow 'synth'"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testCAMManifestBridgeRouteRejectsSameShapeInternalEmitBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-internal-emit-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
            let nodeIndex = try XCTUnwrap(nodes.indices.first {
                nodes[$0]["nodeID"] as? String == "codec-decoder"
            })
            var codec = nodes[nodeIndex]
            var outputs = try XCTUnwrap(codec["outputs"] as? [[String: Any]])
            outputs.append([
                "portName": "debug_audio",
                "optional": false,
                "type": [
                    "typeName": "pcm",
                    "attributes": ["dtype": "f32", "rate": "24khz"],
                ],
            ])
            codec["outputs"] = outputs
            nodes[nodeIndex] = codec
            object["graphNodes"] = nodes

            var flows = try XCTUnwrap(object["flows"] as? [[String: Any]])
            let flowIndex = try XCTUnwrap(flows.indices.first {
                flows[$0]["flowID"] as? String == "synth"
            })
            flows[flowIndex]["emit"] = [[
                "endpointType": "nodePort",
                "nodeID": "codec-decoder",
                "portName": "debug_audio",
            ]]
            object["flows"] = flows
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("no CAM runtime route"), result.stderr)
        XCTAssertTrue(result.stderr.contains("export 'synth' flow 'synth'"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testCAMManifestBridgeRouteUsesSelectedOutputForDualOutputAudioExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-contract-dual-output-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "synth",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no manifest bridge route"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testRunMalformedCAMDescriptorFailsBeforeManifestFallback() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-malformed-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-malformed.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Data("not json".utf8)
            .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("invalid CAM package descriptor"), result.stderr)
        XCTAssertTrue(result.stderr.contains("module.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no run capability"), result.stderr)
    }

    func testRunSymlinkedCAMDescriptorFailsBeforeManifestFallback() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-cam-symlink-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-symlink.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        let external = root.appendingPathComponent("external-cam.json")
        try Self.camDescriptorData("qwen35_text.cam").write(to: external)
        try FileManager.default.createSymbolicLink(
            at: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName),
            withDestinationURL: external
        )

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("CAM package descriptor is not a regular file"), result.stderr)
        XCTAssertTrue(result.stderr.contains("module.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no run capability"), result.stderr)
    }

    func testBenchCAMTextExportPreflightsInventoryWithoutManifestBridge() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-cam-text-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "bench",
            package.path,
            "--iterations", "1",
            "--warmup", "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Bench failed"), result.stderr)
    }

    func testBenchRejectsRetiredQwenGateBeforeInventory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-retired-qwen-gate-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-bench-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "bench",
            package.path,
            "--gate-qwen35",
            "--iterations",
            "1",
            "--warmup",
            "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("--gate-qwen35 was removed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Bench failed"), result.stderr)
    }

    func testBenchCAMAudioExportFailsClosedAgainstLLMManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-cam-audio-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let result = try Self.runSmelt([
            "bench",
            package.path,
            "--iterations", "1",
            "--warmup", "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("module audio exports use a specialized audio benchmark harness"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Bench failed"), result.stderr)
    }

    func testBenchCAMAudioExportUsesSpecializedHarnessDiagnosticWithQwen3TTSManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-cam-audio-qwen-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let result = try Self.runSmelt([
            "bench",
            package.path,
            "--iterations", "1",
            "--warmup", "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("module audio exports use a specialized audio benchmark harness"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Bench failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no bench capability"), result.stderr)
    }

    func testServeCAMTextExportPreflightsInventoryWithoutManifestBridge() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-cam-text-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("smelt serve failed"), result.stderr)
    }

    func testServeCAMAudioExportFailsClosedAgainstLLMManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-cam-audio-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("smelt serve failed"), result.stderr)
    }

    func testServeCAMManifestBridgeRouteUsesSelectedOutputForDualOutputAudioExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-cam-dual-output-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "synth",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-pcm-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no manifest bridge route"), result.stderr)
        XCTAssertFalse(result.stderr.contains("smelt serve failed"), result.stderr)
    }

    func testServeMalformedCAMDescriptorFailsBeforeManifestFallback() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-cam-malformed-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-malformed.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Data("not json".utf8)
            .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("invalid CAM package descriptor"), result.stderr)
        XCTAssertTrue(result.stderr.contains("module.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("smelt serve failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no serve capability"), result.stderr)
    }

    func testServeSymlinkedCAMDescriptorFailsBeforeManifestFallback() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-cam-symlink-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-symlink.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        let external = root.appendingPathComponent("external-cam.json")
        try Self.camDescriptorData("qwen35_text.cam").write(to: external)
        try FileManager.default.createSymbolicLink(
            at: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName),
            withDestinationURL: external
        )

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("CAM package descriptor is not a regular file"), result.stderr)
        XCTAssertTrue(result.stderr.contains("module.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("smelt serve failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("no serve capability"), result.stderr)
    }


    func testTraceRecordCAMTextExportFailsClosedAgainstTTSManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-cam-text-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary-text-to-text-trace-runtime"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-text manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
    }

    func testTraceRecordCAMTextManifestBridgeRouteUsesSelectedOutputForDualOutputTextExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-cam-text-dual-output-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen35_text.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "generate",
                portName: "debug_audio",
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "24khz"]
            )
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-text manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-pcm-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM --case-text is ambiguous"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
    }

    func testTraceRecordCAMAudioExportFailsClosedAgainstLLMManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-cam-audio-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
    }

    func testTraceRecordCAMAudioManifestBridgeRouteUsesSelectedOutputForDualOutputAudioExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-cam-audio-dual-output-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "synth",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-pcm-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM --case-text is ambiguous"), result.stderr)
    }

    func testTraceRecordCAMTextReportsAmbiguousWhenBothTextTraceRequestsMatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-cam-text-ambiguous-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            try Self.appendExportOutput(
                in: &object,
                exportID: "synth",
                portName: "debug_text",
                typeName: "text",
                attributes: ["encoding": "utf8"]
            )
            var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
            let index = try XCTUnwrap(exports.indices.first {
                exports[$0]["exportID"] as? String == "synth"
            })
            var capabilities = try XCTUnwrap(exports[index]["capabilities"] as? [String])
            capabilities.append("run.generate")
            exports[index]["capabilities"] = capabilities
            object["exports"] = exports
        }
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("CAM --case-text is ambiguous"), result.stderr)
        XCTAssertTrue(result.stderr.contains("debug_text:text"), result.stderr)
        XCTAssertTrue(result.stderr.contains("audio:pcm"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
    }

    func testLingerWorkerCAMTextExportPreflightsInventoryWithoutManifestBridge() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-text-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("text-to-text-manifest-bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary runtime witness"), result.stderr)
        XCTAssertFalse(result.stderr.contains("temporary-text-to-text-warm-runtime"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
    }

    func testLingerWorkerCAMAudioExportFailsClosedAgainstLLMManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-audio-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
    }

    func testLingerWorkerRejectsMismatchedCAMIdentityBeforeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-identity-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", Self.fakeCAMLingerIdentity,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module linger identity mismatch"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testLingerWorkerRejectsLegacyAdapterCAMIdentityBeforeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-legacy-identity-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", Self.legacyAdapterCAMLingerIdentity,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("invalid module linger identity"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerRejectsProtocolStampedCAMIdentityBeforeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-protocol-stamped-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", Self.protocolStampedCAMLingerIdentity,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("invalid module linger identity"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerRejectsLegacyRouteFieldBeforeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-contract-mismatch-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let identity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .runText
        )
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(identity.json.utf8)) as? [String: Any]
        )
        object["manifestBridgeRoute"] = "text-to-pcm-manifest-bridge"
        let legacyData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", legacyData.base64EncodedString(),
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("invalid module linger identity"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("module linger identity mismatch"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerAcceptsRunTextCAMIdentityForSameWorkerExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-run-text-identity-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let runIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .runText
        )
        let workerIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .serveText
        )
        let traceIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .traceTextGenerate
        )
        XCTAssertFalse(runIdentity.json.contains("\"adapter\""), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("protocolVersion"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("manifestBridgeRoute"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("temporary-"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requestName"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("run text"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("serve text"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("trace text generation"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("descriptorIdentityToken"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requiredInputShapes"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requiredOutputShapes"), runIdentity.json)
        XCTAssertEqual(runIdentity.decoded, workerIdentity.decoded)
        XCTAssertEqual(runIdentity.decoded, traceIdentity.decoded)
        XCTAssertFalse(runIdentity.decoded.camSemanticSHA256.isEmpty)
        XCTAssertFalse(runIdentity.decoded.exportABISHA256.isEmpty)
        XCTAssertEqual(runIdentity.decoded.exportID, workerIdentity.decoded.exportID)
        XCTAssertEqual(runIdentity.decoded.flowID, workerIdentity.decoded.flowID)
        XCTAssertEqual(runIdentity.decoded.inputPorts, workerIdentity.decoded.inputPorts)
        XCTAssertEqual(runIdentity.decoded.outputPorts, workerIdentity.decoded.outputPorts)
        XCTAssertEqual(
            runIdentity.decoded.authoredCapabilities,
            workerIdentity.decoded.authoredCapabilities
        )
        XCTAssertEqual(runIdentity.decoded.matchedGateIDs, workerIdentity.decoded.matchedGateIDs)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", runIdentity.encoded,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("module linger identity mismatch"), result.stderr)
        XCTAssertFalse(result.stderr.contains("invalid module linger identity"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerAcceptsRunAudioCAMIdentityForSameWorkerExport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-run-audio-identity-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMDescriptor("qwen3_tts.cam", to: package)

        let runIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .runAudio
        )
        let workerIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .serveAudio
        )
        let traceIdentity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .traceTextSynthesize
        )
        XCTAssertFalse(runIdentity.json.contains("\"adapter\""), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("protocolVersion"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("manifestBridgeRoute"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("temporary-"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requestName"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("run audio"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("serve audio"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("trace text synthesis"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("descriptorIdentityToken"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requiredInputShapes"), runIdentity.json)
        XCTAssertFalse(runIdentity.json.contains("requiredOutputShapes"), runIdentity.json)
        XCTAssertEqual(runIdentity.decoded, workerIdentity.decoded)
        XCTAssertEqual(runIdentity.decoded, traceIdentity.decoded)
        XCTAssertFalse(runIdentity.decoded.camSemanticSHA256.isEmpty)
        XCTAssertFalse(runIdentity.decoded.exportABISHA256.isEmpty)
        XCTAssertEqual(runIdentity.decoded.exportID, workerIdentity.decoded.exportID)
        XCTAssertEqual(runIdentity.decoded.flowID, workerIdentity.decoded.flowID)
        XCTAssertEqual(runIdentity.decoded.inputPorts, workerIdentity.decoded.inputPorts)
        XCTAssertEqual(runIdentity.decoded.outputPorts, workerIdentity.decoded.outputPorts)
        XCTAssertEqual(
            runIdentity.decoded.authoredCapabilities,
            workerIdentity.decoded.authoredCapabilities
        )
        XCTAssertEqual(runIdentity.decoded.matchedGateIDs, workerIdentity.decoded.matchedGateIDs)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", runIdentity.encoded,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("weights.bin"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("module linger identity mismatch"), result.stderr)
        XCTAssertFalse(result.stderr.contains("invalid module linger identity"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerAcceptsCopiedDescriptorWarmIdentityButStillChecksInventory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-copied-identity-\(UUID().uuidString)", isDirectory: true)
        let first = root.appendingPathComponent("first.smeltpkg", isDirectory: true)
        let second = root.appendingPathComponent("second.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: first, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for package in [first, second] {
            try Data(Self.qwen3TTSManifest.utf8)
                .write(to: package.appendingPathComponent("manifest.json"))
            try Self.writeCAMDescriptor("qwen35_text.cam", to: package)
        }
        let identity = try Self.encodedCAMLingerIdentity(
            package: first,
            request: .runText
        )
        let copiedIdentity = try Self.encodedCAMLingerIdentity(
            package: second,
            request: .serveText
        )
        XCTAssertEqual(identity.decoded, copiedIdentity.decoded)
        XCTAssertNotEqual(first.path, second.path)

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            second.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", identity.encoded,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertFalse(result.stderr.contains("module linger identity mismatch"), result.stderr)
        XCTAssertTrue(result.stderr.contains("module package inventory missing files"), result.stderr)
        XCTAssertTrue(result.stderr.contains("tokenizer.json"), result.stderr)
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    func testLingerWorkerRejectsExpectedCAMIdentityWhenDescriptorMissing() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-cam-identity-missing-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-llm.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", Self.fakeCAMLingerIdentity,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
    }

    func testBridgeBoundPackageOpeningCommandsRejectPackagesWithoutCAMDescriptor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bridge-bound-cam-missing-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-llm.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let socket = root.appendingPathComponent("worker.sock")
        let commandCases: [(name: String, arguments: [String], expected: String)] = [
            (
                "run",
                ["run", package.path, "hello"],
                "smelt run: expected module descriptor but package has no module.json"
            ),
            (
                "bench",
                ["bench", package.path, "--iterations", "1", "--warmup", "0"],
                "smelt lab bench decode: expected module descriptor but package has no module.json"
            ),
            (
                "serve",
                ["serve", package.path, "--transport", "http", "--port", "65535"],
                "smelt serve: expected module descriptor but package has no module.json"
            ),
            (
                "trace-inspect",
                ["trace", "inspect", package.path],
                "smelt lab trace: expected module descriptor but package has no module.json"
            ),
            (
                "trace-record",
                ["trace", "record", package.path, "--case-text", "hello"],
                "smelt lab trace: expected module descriptor but package has no module.json"
            ),
            (
                "linger-worker",
                ["linger-worker", package.path, "--socket", socket.path, "--idle", "1"],
                "smelt linger-worker: expected module descriptor but package has no module.json"
            ),
        ]

        for commandCase in commandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            XCTAssertEqual(result.status, 1, commandCase.name)
            XCTAssertTrue(
                result.stderr.contains(commandCase.expected),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(result.stderr.contains("no run capability"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("no bench capability"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Qwen"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Load failed"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Run failed"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Bench failed"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Create failed"), commandCase.name)
        }
    }

    func testRunRejectsPackageWithoutCAMBeforeManifestArchitectureDispatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-run-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-unknown-tts.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.unknownTTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt run: expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("Qwen"),
            result.stderr
        )
    }


    func testServeRejectsPackageWithoutCAMBeforeManifestArchitectureDispatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-serve-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-tts.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let result = try Self.runSmelt([
            "serve",
            package.path,
            "--transport", "http",
            "--port", "65535",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt serve: expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("package detected as tts but its manifest is unreadable"),
            result.stderr
        )
    }

    func testLingerWorkerRejectsPackageWithoutCAMBeforeManifestArchitectureDispatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-linger-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-tts.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt linger-worker: expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("Qwen"),
            result.stderr
        )
    }

    func testTraceRecordRejectsPackageWithoutCAMBeforeManifestArchitectureDispatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-trace-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-tts.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let result = try Self.runSmelt([
            "trace",
            "record",
            package.path,
            "--case-text", "test prompt",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt lab trace: expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("runtime recording is not implemented"), result.stderr)
        XCTAssertFalse(
            result.stderr.contains("Qwen"),
            result.stderr
        )
    }

    func testGenericBenchRejectsPackageWithoutCAMBeforeManifestArchitectureDispatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-tts.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let result = try Self.runSmelt([
            "bench",
            package.path,
            "--iterations", "1",
            "--warmup", "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt lab bench decode: expected module descriptor but package has no module.json"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("Bench failed"),
            result.stderr
        )
    }

    func testPackageOpeningCommandsRejectUnknownLLMWithoutCAMBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-unknown-llm-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-unknown-llm.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.unknownLLMManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let socket = root.appendingPathComponent("worker.sock")
        let commandCases: [(name: String, arguments: [String], expected: String)] = [
            (
                "run",
                ["run", package.path, "hello"],
                "smelt run: expected module descriptor but package has no module.json"
            ),
            (
                "bench",
                ["bench", package.path, "--iterations", "1", "--warmup", "0"],
                "smelt lab bench decode: expected module descriptor but package has no module.json"
            ),
            (
                "serve",
                ["serve", package.path, "--transport", "http", "--port", "65535"],
                "smelt serve: expected module descriptor but package has no module.json"
            ),
            (
                "trace",
                ["trace", "record", package.path, "--case-text", "hello"],
                "smelt lab trace: expected module descriptor but package has no module.json"
            ),
            (
                "linger-worker",
                ["linger-worker", package.path, "--socket", socket.path, "--idle", "1"],
                "smelt linger-worker: expected module descriptor but package has no module.json"
            ),
        ]

        for commandCase in commandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            XCTAssertEqual(result.status, 1, commandCase.name)
            XCTAssertTrue(
                result.stderr.contains(commandCase.expected),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("failed to load"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("Load failed"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("no run capability"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("no bench capability"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("no serve capability"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("no trace capability"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains("no linger-worker capability"),
                "\(commandCase.name): \(result.stderr)"
            )
        }
    }

    func testDispatchesCAMDecodeTableUsesStaticCapabilityWithoutLLMDescriptor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-dispatches-cam-decode-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-dispatches.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)
        try Self.writeDispatchTable("dispatches.bin", to: package)

        let result = try Self.runSmelt([
            "dispatches",
            package.path,
            "--sequence",
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertTrue(result.stdout.contains("records=1"), result.stdout)
        XCTAssertTrue(result.stdout.contains("pipeline_0"), result.stdout)
        XCTAssertFalse(result.stderr.contains("expected llm package"), result.stderr)
    }

    func testDispatchesCAMPrefillTableUsesRuntimeContractAndExactInventory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-dispatches-cam-prefill-\(UUID().uuidString)", isDirectory: true)
        let fast = root.appendingPathComponent("fake-cam-fast.smeltpkg", isDirectory: true)
        let text = root.appendingPathComponent("fake-cam-text.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: fast, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: text, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: fast)
        try Self.writeDispatchTable("dispatches.bin", to: fast)

        let missingFastTable = try Self.runSmelt([
            "dispatches",
            fast.path,
            "--table", "prefill",
        ])

        XCTAssertEqual(missingFastTable.status, 1)
        XCTAssertTrue(
            missingFastTable.stderr.contains("module capability files missing: prefill_dispatches.bin"),
            missingFastTable.stderr
        )
        XCTAssertFalse(missingFastTable.stderr.contains("Failed to load"), missingFastTable.stderr)
        XCTAssertFalse(missingFastTable.stderr.contains("expected llm package"), missingFastTable.stderr)

        try Self.writeCAMDescriptor("qwen35_text.cam", to: text)

        let missing = try Self.runSmelt([
            "dispatches",
            text.path,
            "--table", "prefill",
        ])

        XCTAssertEqual(missing.status, 1)
        XCTAssertTrue(
            missing.stderr.contains("module capability files missing: prefill_dispatches.bin"),
            missing.stderr
        )
        XCTAssertFalse(missing.stderr.contains("Failed to load"), missing.stderr)

        try Self.writeDispatchTable("prefill_dispatches.bin", to: text)

        let accepted = try Self.runSmelt([
            "dispatches",
            text.path,
            "--table", "prefill",
            "--sequence",
        ])

        XCTAssertEqual(accepted.status, 0, accepted.stderr)
        XCTAssertTrue(accepted.stdout.contains("records=1"), accepted.stdout)
        XCTAssertTrue(accepted.stdout.contains("pipeline_0"), accepted.stdout)
    }

    func testLabInspectDispatchesReadsVerifyArgmaxTable() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-lab-dispatches-verify-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-verify.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("bonsai_27b_ternary.cam", to: package)
        try Self.writeDispatchTable("prefill_verify_argmax_dispatches.bin", to: package)

        let result = try Self.runRawSmelt([
            "lab", "inspect", "dispatches", package.path,
            "--table", "verify",
            "--sequence-length", "4",
            "--sequence",
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertTrue(result.stdout.contains("prefill_verify_argmax_dispatches.bin"), result.stdout)
        XCTAssertTrue(result.stdout.contains("records=1"), result.stdout)
        XCTAssertTrue(result.stdout.contains("threadgroups=1"), result.stdout)
    }

    func testBenchLogprobsCAMRequiresAuthoredPrefillCapabilityBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-bench-logprobs-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-logprobs.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedWithoutAuthoredCapability = try Self.runSmelt([
            "bench-logprobs",
            "--package",
            package.path,
            "--iters",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutAuthoredCapability.status, 1)
        XCTAssertTrue(
            rejectedWithoutAuthoredCapability.stderr.contains(
                "no CAM export satisfies lab bench logprobs text request"
            ),
            rejectedWithoutAuthoredCapability.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredCapability.stderr.contains("bench-logprobs failed"),
            rejectedWithoutAuthoredCapability.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredCapability.stderr.contains("expected llm package"),
            rejectedWithoutAuthoredCapability.stderr
        )

        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.appendExportCapability(
                in: &object,
                exportID: "generate",
                capability: "bench.prefill-logprobs"
            )
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal all-logits batch 256"
            )
            try Self.replaceTransformerLayerRoles(
                in: &object,
                blockID: "trunk",
                roles: ["attention"],
                removeDeltaShape: true
            )
        }

        let rejectedWithoutInventory = try Self.runSmelt([
            "bench-logprobs",
            "--package",
            package.path,
            "--iters",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("bench-logprobs failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Failed to load"),
            rejectedWithoutInventory.stderr
        )

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.appendExportCapability(
                in: &object,
                exportID: "generate",
                capability: "bench.prefill-logprobs"
            )
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal all-logits batch 64"
            )
            try Self.replaceTransformerLayerRoles(
                in: &object,
                blockID: "trunk",
                roles: ["attention"],
                removeDeltaShape: true
            )
        }
        try Self.writeCAMTextInventoryFiles(to: package)

        let rejectedOverCAMBatch = try Self.runSmelt([
            "bench-logprobs",
            "--package",
            package.path,
            "--prompt-tokens",
            "65",
            "--iters",
            "1",
        ])

        XCTAssertEqual(rejectedOverCAMBatch.status, 1)
        XCTAssertTrue(
            rejectedOverCAMBatch.stderr.contains(
                "prompt token count 65 exceeds CAM prefill all-logits batch 64"
            ),
            rejectedOverCAMBatch.stderr
        )
        XCTAssertFalse(
            rejectedOverCAMBatch.stderr.contains("Failed to load"),
            rejectedOverCAMBatch.stderr
        )
        XCTAssertFalse(
            rejectedOverCAMBatch.stderr.contains("Package does not support prefillAllLogits"),
            rejectedOverCAMBatch.stderr
        )
    }

    func testPrefillBenchCAMRequiresPrefillInventoryBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-prefill-bench-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-prefill-bench.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedRemovedGate = try Self.runSmelt([
            "prefill-bench",
            package.path,
            "--gate-qwen35",
            "--iterations",
            "1",
            "--warmup",
            "0",
        ])

        XCTAssertEqual(rejectedRemovedGate.status, 1)
        XCTAssertTrue(
            rejectedRemovedGate.stderr.contains(
                "--gate-qwen35 was removed"
            ),
            rejectedRemovedGate.stderr
        )
        XCTAssertFalse(
            rejectedRemovedGate.stderr.contains("Prefill bench failed"),
            rejectedRemovedGate.stderr
        )
        XCTAssertFalse(
            rejectedRemovedGate.stderr.contains("module package inventory missing files"),
            rejectedRemovedGate.stderr
        )

        let rejectedWithoutInventory = try Self.runSmelt([
            "prefill-bench",
            package.path,
            "--iterations",
            "1",
            "--warmup",
            "0",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Prefill bench failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Failed to load"),
            rejectedWithoutInventory.stderr
        )
    }

    func testPrefillCAMRequiresPrefillInventoryBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-prefill-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-prefill.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "prefill",
            package.path,
            "--tokens",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Prefill failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Failed to load"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMTextInventoryFiles(to: package)
        try FileManager.default.removeItem(
            at: package.appendingPathComponent("dispatches.bin")
        )
        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,prefill_dispatches.bin,tokenizer.json,tokenizer.bin,module.json",
                ]]
            )
        }

        let rejectedWithoutDecodeInventory = try Self.runSmelt([
            "prefill",
            package.path,
            "--tokens",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutDecodeInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutDecodeInventory.stderr.contains(
                "no CAM export satisfies lab prefill text request"
            ),
            rejectedWithoutDecodeInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutDecodeInventory.stderr.contains("Prefill failed"),
            rejectedWithoutDecodeInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutDecodeInventory.stderr.contains("Failed to load"),
            rejectedWithoutDecodeInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutDecodeInventory.stderr.contains("no dispatches.bin"),
            rejectedWithoutDecodeInventory.stderr
        )
    }

    func testPrefillKernelsCAMRequiresPrefillInventoryBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-prefill-kernels-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-prefill-kernels.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "prefill-kernels",
            package.path,
            "--tokens",
            "1",
            "--iterations",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Prefill kernel profile failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Failed to load"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMTextInventoryFiles(to: package)
        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal all-logits batch 256"
            )
        }

        let rejectedWithoutPlainPrefillCompile = try Self.runSmelt([
            "prefill-kernels",
            package.path,
            "--tokens",
            "1",
            "--iterations",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutPlainPrefillCompile.status, 1)
        XCTAssertTrue(
            rejectedWithoutPlainPrefillCompile.stderr.contains(
                "no CAM export satisfies lab profile prefill text request"
            ),
            rejectedWithoutPlainPrefillCompile.stderr
        )
        XCTAssertFalse(
            rejectedWithoutPlainPrefillCompile.stderr.contains("Prefill kernel profile failed"),
            rejectedWithoutPlainPrefillCompile.stderr
        )
        XCTAssertFalse(
            rejectedWithoutPlainPrefillCompile.stderr.contains("Failed to load"),
            rejectedWithoutPlainPrefillCompile.stderr
        )
        XCTAssertFalse(
            rejectedWithoutPlainPrefillCompile.stderr.contains(
                "Package does not have Metal prefill"
            ),
            rejectedWithoutPlainPrefillCompile.stderr
        )

        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.setCompileRequirementValue(
                in: &object,
                key: "prefill",
                value: "metal batch 1"
            )
        }

        let rejectedOverCAMBatch = try Self.runSmelt([
            "prefill-kernels",
            package.path,
            "--tokens",
            "2",
            "--iterations",
            "1",
        ])

        XCTAssertEqual(rejectedOverCAMBatch.status, 1)
        XCTAssertTrue(
            rejectedOverCAMBatch.stderr.contains(
                "prompt token count 2 exceeds CAM prefill batch 1"
            ),
            rejectedOverCAMBatch.stderr
        )
        XCTAssertFalse(
            rejectedOverCAMBatch.stderr.contains("Prefill kernel profile failed"),
            rejectedOverCAMBatch.stderr
        )
        XCTAssertFalse(
            rejectedOverCAMBatch.stderr.contains("Failed to load"),
            rejectedOverCAMBatch.stderr
        )
        XCTAssertFalse(
            rejectedOverCAMBatch.stderr.contains("Package does not have Metal prefill"),
            rejectedOverCAMBatch.stderr
        )
    }

    func testKernelsCAMRequiresDecodeInventoryBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-kernels-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-kernels.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "kernels",
            package.path,
            "--iterations",
            "1",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Kernel profile failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Failed to load"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.writeCAMTextInventoryFiles(to: package)
        try FileManager.default.removeItem(
            at: package.appendingPathComponent("prefill_dispatches.bin")
        )

        let acceptedWithoutPrefillInventory = try Self.runSmelt([
            "kernels",
            package.path,
            "--iterations",
            "1",
        ])

        XCTAssertEqual(acceptedWithoutPrefillInventory.status, 1)
        XCTAssertTrue(
            acceptedWithoutPrefillInventory.stderr.contains("Kernel profile failed"),
            acceptedWithoutPrefillInventory.stderr
        )
        XCTAssertFalse(
            acceptedWithoutPrefillInventory.stderr.contains("module package inventory missing files"),
            acceptedWithoutPrefillInventory.stderr
        )
        XCTAssertFalse(
            acceptedWithoutPrefillInventory.stderr.contains("prefill_dispatches.bin"),
            acceptedWithoutPrefillInventory.stderr
        )
        XCTAssertFalse(
            acceptedWithoutPrefillInventory.stderr.contains("no CAM export satisfies"),
            acceptedWithoutPrefillInventory.stderr
        )
    }

    func testReplayCAMRequiresInventoryBeforeTraceLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-replay-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-replay.smeltpkg", isDirectory: true)
        let missingTrace = root.appendingPathComponent("missing-trace.jsonl")
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_text.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "replay",
            missingTrace.path,
            "--package",
            package.path,
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Replay failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("No such file"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )

        try Self.writeMutatedCAMDescriptor("qwen35_text.cam", to: package) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: []
            )
        }

        let rejectedWithoutAuthoredInventory = try Self.runSmelt([
            "replay",
            missingTrace.path,
            "--package",
            package.path,
        ])

        XCTAssertEqual(rejectedWithoutAuthoredInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutAuthoredInventory.stderr.contains(
                "module package inventory requirement missing"
            ),
            rejectedWithoutAuthoredInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredInventory.stderr.contains("Replay failed"),
            rejectedWithoutAuthoredInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredInventory.stderr.contains("No such file"),
            rejectedWithoutAuthoredInventory.stderr
        )

        try Self.writeMutatedCAMDescriptor("qwen35_fast.cam", to: package) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: [[
                    "subject": "package-files",
                    "relation": "include",
                    "value": "dispatches.bin",
                ]]
            )
        }
        try Self.writeDispatchTable("dispatches.bin", to: package)

        let rejectedWithTooSmallInventory = try Self.runSmelt([
            "optimizer-report",
            package.path,
        ])

        XCTAssertEqual(rejectedWithTooSmallInventory.status, 1)
        XCTAssertTrue(
            rejectedWithTooSmallInventory.stderr.contains(
                "no CAM export satisfies optimizer report request"
            ),
            rejectedWithTooSmallInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithTooSmallInventory.stderr.contains("Optimizer report failed"),
            rejectedWithTooSmallInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithTooSmallInventory.stderr.contains("No such file"),
            rejectedWithTooSmallInventory.stderr
        )
    }

    func testOptimizerReportCAMRequiresInventoryBeforeReportLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-optimizer-report-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-optimizer-report.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "optimizer-report",
            package.path,
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Optimizer report failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("No such file"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )

        try Self.writeMutatedCAMDescriptor("qwen35_fast.cam", to: package) { object in
            try Self.replaceGateRequirements(
                in: &object,
                gateID: "inventory",
                requirements: []
            )
        }

        let rejectedWithoutAuthoredInventory = try Self.runSmelt([
            "optimizer-report",
            package.path,
        ])

        XCTAssertEqual(rejectedWithoutAuthoredInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutAuthoredInventory.stderr.contains(
                "module package inventory requirement missing"
            ),
            rejectedWithoutAuthoredInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredInventory.stderr.contains("Optimizer report failed"),
            rejectedWithoutAuthoredInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutAuthoredInventory.stderr.contains("No such file"),
            rejectedWithoutAuthoredInventory.stderr
        )
    }

    func testOptimizeNextCAMRequiresInventoryBeforeGitWorktree() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-optimize-next-cam-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-optimize-next.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)

        let rejectedWithoutInventory = try Self.runSmelt([
            "optimize-next",
            package.path,
            "--verifier",
            "true",
        ])

        XCTAssertEqual(rejectedWithoutInventory.status, 1)
        XCTAssertTrue(
            rejectedWithoutInventory.stderr.contains("module package inventory missing files"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("Optimize-next failed"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("git"),
            rejectedWithoutInventory.stderr
        )
        XCTAssertFalse(
            rejectedWithoutInventory.stderr.contains("expected llm package"),
            rejectedWithoutInventory.stderr
        )
    }

    func testOptimizeNextRequiresBuildCommandBeforeReportLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-optimize-next-build-command-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-optimize-next.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)
        try Self.writeOptimizerReportInventoryFiles(to: package)

        let rejectedWithoutBuildCommand = try Self.runSmelt([
            "optimize-next",
            package.path,
            "--verifier",
            "true",
        ])

        XCTAssertEqual(rejectedWithoutBuildCommand.status, 1)
        XCTAssertTrue(
            rejectedWithoutBuildCommand.stderr.contains(
                "Optimize-next failed: optimize-next requires --build-command CMD"
            ),
            rejectedWithoutBuildCommand.stderr
        )
        XCTAssertFalse(
            rejectedWithoutBuildCommand.stderr.contains("module package inventory missing files"),
            rejectedWithoutBuildCommand.stderr
        )
        XCTAssertFalse(
            rejectedWithoutBuildCommand.stderr.contains("No such file"),
            rejectedWithoutBuildCommand.stderr
        )
        XCTAssertFalse(
            rejectedWithoutBuildCommand.stderr.contains("no known build script could be inferred"),
            rejectedWithoutBuildCommand.stderr
        )
    }

    func testOptimizeNextUsesCWDRelativePackagePathForAdmissionAndReportLoad() throws {
        let root = Self.repoRoot
            .appendingPathComponent(".smeltpkg-optimize-next-path-\(UUID().uuidString)", isDirectory: true)
        let invocationDirectory = root.appendingPathComponent("invocation", isDirectory: true)
        let package = invocationDirectory.appendingPathComponent("relative.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)
        try Self.writeOptimizerReportInventoryFiles(to: package)

        let result = try Self.runSmelt(
            [
                "optimize-next",
                "relative.smeltpkg",
                "--verifier",
                "true",
                "--build-command",
                "true",
            ],
            currentDirectory: invocationDirectory
        )

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stderr.contains("Optimize-next failed"), result.stderr)
        XCTAssertFalse(
            result.stderr.contains("module package inventory missing files"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("No such file"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("expected llm package"),
            result.stderr
        )
    }

    func testLLMOnlyUtilitiesRejectUnknownLLMArchitectureBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-llm-only-unknown-descriptor-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-unknown-llm.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.unknownLLMManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))

        let camCommandCases: [(name: String, arguments: [String], failurePrefix: String)] = [
            ("profile", ["profile", package.path, "--iterations", "1"], "Profile failed"),
            ("dispatches", ["dispatches", package.path], "Failed to load"),
            ("bench-logprobs", ["bench-logprobs", package.path, "--iters", "1"], "bench-logprobs failed"),
            ("prefill-bench", ["prefill-bench", package.path, "--iterations", "1"], "Prefill bench failed"),
            ("prefill", ["prefill", package.path, "--tokens", "1"], "Prefill failed"),
            ("prefill-kernels", ["prefill-kernels", package.path, "--iterations", "1"], "Prefill kernel profile failed"),
            ("kernels", ["kernels", package.path, "--iterations", "1"], "Kernel profile failed"),
            ("optimizer-report", ["optimizer-report", package.path], "Optimizer report failed"),
            ("optimize-next", ["optimize-next", package.path, "--verifier", "true"], "Optimize-next failed"),
            (
                "kernel-lab",
                [
                    "kernel-lab",
                    package.path,
                    "--case", "package-replay",
                    "--iterations", "1",
                    "--warmup", "0",
                ],
                "Kernel lab failed"
            ),
            (
                "replay",
                [
                    "replay",
                    root.appendingPathComponent("missing-trace.jsonl").path,
                    "--package",
                    package.path,
                ],
                "Replay failed"
            ),
        ]
        let legacyCommandCases: [(name: String, arguments: [String], failurePrefix: String)] = []

        for commandCase in camCommandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            XCTAssertEqual(result.status, 1, commandCase.name)
            XCTAssertTrue(
                result.stderr.contains("expected module descriptor but package has no module.json"),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains(commandCase.failurePrefix),
                "\(commandCase.name): \(result.stderr)"
            )
        }

        for commandCase in legacyCommandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            XCTAssertEqual(result.status, 1, commandCase.name)
            XCTAssertTrue(
                result.stderr.contains(
                    "expected llm package, got llm architecture 'mystery-llm'"
                ),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(
                result.stderr.contains(commandCase.failurePrefix),
                "\(commandCase.name): \(result.stderr)"
            )
        }
    }

    func testKernelLabCAMRequiresInventoryBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-kernel-lab-cam-inventory-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-kernel-lab.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCAMDescriptor("qwen35_fast.cam", to: package)

        let result = try Self.runSmelt([
            "kernel-lab",
            package.path,
            "--case", "package-replay",
            "--iterations", "1",
            "--warmup", "0",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("module package inventory missing files"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("Kernel lab failed"),
            result.stderr
        )
        XCTAssertFalse(
            result.stderr.contains("expected llm"),
            result.stderr
        )
    }

    func testCAMAudioShapeDriftRejectsPackageOpeningCommandsBeforeHandlers() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-audio-shape-drift-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio-drift.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
            try Self.mutatePortAttributes(
                in: &exports[0],
                portListKey: "outputs",
                index: 0,
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            object["exports"] = exports
            try Self.mutateGraphNodePortAttributes(
                in: &object,
                nodeID: "codec-decoder",
                portListKey: "outputs",
                portName: "audio",
                attributes: ["dtype": "f32", "rate": "16khz"]
            )
            try Self.mutateGraphEdgeValueType(
                in: &object,
                typeName: "pcm",
                attributes: ["dtype": "f32", "rate": "16khz"]
            ) { from, to in
                from["endpointType"] as? String == "nodePort"
                    && from["nodeID"] as? String == "codec-decoder"
                    && from["portName"] as? String == "audio"
                    && to["endpointType"] as? String == "moduleOutput"
                    && to["name"] as? String == "audio"
            }
        }.write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let socket = root.appendingPathComponent("worker.sock")
        let commandCases: [(name: String, arguments: [String], expected: String)] = [
            (
                "run",
                ["run", package.path, "hello"],
                "smelt run: no CAM export satisfies run request"
            ),
            (
                "bench",
                ["bench", package.path, "--iterations", "1", "--warmup", "0"],
                "smelt lab bench decode: no CAM export satisfies decode benchmark request"
            ),
            (
                "serve",
                ["serve", package.path, "--transport", "http", "--port", "65535"],
                "smelt serve: no CAM export satisfies serve request"
            ),
            (
                "trace",
                ["trace", "record", package.path, "--case-text", "hello"],
                "smelt lab trace: no CAM export satisfies trace text request"
            ),
            (
                "linger-worker",
                ["linger-worker", package.path, "--socket", socket.path, "--idle", "1"],
                "smelt linger-worker: no CAM export satisfies linger request"
            ),
        ]

        for commandCase in commandCases {
            let result = try Self.runSmelt(commandCase.arguments)
            XCTAssertEqual(result.status, 1, commandCase.name)
            XCTAssertTrue(
                result.stderr.contains(commandCase.expected),
                "\(commandCase.name): \(result.stderr)"
            )
            XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Qwen"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Run failed"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Bench failed"), commandCase.name)
            XCTAssertFalse(result.stderr.contains("Create failed"), commandCase.name)
        }
    }

    func testRunCAMAudioLingerPreflightsWorkerCapabilityBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-audio-linger-preflight-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio-no-stream.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
            exports[0]["capabilities"] = ["run.synthesize", "prepare.voice-defaults"]
            object["exports"] = exports
        }.write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
            "--linger", "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt run: no CAM export satisfies linger request"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testRunCAMAudioLingerPreflightWinsBeforeManifestConflict() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-audio-linger-preflight-conflict-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio-no-stream.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.llmManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
            exports[0]["capabilities"] = ["run.synthesize", "prepare.voice-defaults"]
            object["exports"] = exports
        }.write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))

        let result = try Self.runSmelt([
            "run",
            package.path,
            "hello",
            "--linger", "1",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt run: no CAM export satisfies linger request"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("expected text-to-PCM manifest bridge"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
    }

    func testLingerWorkerCAMAudioPreflightsRunStreamBeforeRuntimeLoad() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-cam-audio-linger-worker-preflight-\(UUID().uuidString)", isDirectory: true)
        let package = root.appendingPathComponent("fake-cam-audio-no-stream.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(Self.qwen3TTSManifest.utf8)
            .write(to: package.appendingPathComponent("manifest.json"))
        try Self.mutatedCAMDescriptorData("qwen3_tts.cam") { object in
            var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
            exports[0]["capabilities"] = ["run.synthesize", "prepare.voice-defaults"]
            object["exports"] = exports
        }.write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))
        let identity = try Self.encodedCAMLingerIdentity(
            package: package,
            request: .runAudio
        )

        let socket = root.appendingPathComponent("worker.sock")
        let result = try Self.runSmelt([
            "linger-worker",
            package.path,
            "--socket", socket.path,
            "--idle", "1",
            "--module-linger-identity", identity.encoded,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("smelt linger-worker: no CAM export satisfies linger request"),
            result.stderr
        )
        XCTAssertFalse(result.stderr.contains("CAM/manifest conflict"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Qwen"), result.stderr)
        XCTAssertFalse(result.stderr.contains("Run failed"), result.stderr)
        XCTAssertFalse(result.stderr.contains("tts run failed"), result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socket.path))
    }

    private static let qwen3TTSManifest = """
    {}
    """

    private static let textStyleArgsJSON = """
    {
      "version": 1,
      "args": [
        {
          "flag": "style",
          "type": "string",
          "required": true
        }
      ],
      "prompt": "{style} {input}"
    }
    """

    private static let unknownLLMManifest = """
    {
      "kind": "llm",
      "architecture": "mystery-llm"
    }
    """

    private static let unknownTTSManifest = """
    {
      "kind": "tts",
      "architecture": "mystery-audio"
    }
    """

    private static let llmManifest = """
    {
      "kind": "llm"
    }
    """

    private static func qwen3TTSCodecOnlyManifest() -> Qwen3TTSManifest {
        Qwen3TTSManifest(
            version: 1,
            blocks: .qwen3TTSCodecDecoder,
            loop: .qwen3TTSCodecDecoder,
            modelName: "qwen3-tts-codec-only",
            pageSize: 16,
            pipelines: [],
            eosTokens: [],
            totalBytes: 0,
            weights: []
        )
    }

    private static func writeQwen3TTSFacadeInventoryFiles(to package: URL) throws {
        for file in ["config.json", "merges.txt", "model.metallib", "tokenizer_config.json", "vocab.json", "weights.bin"] {
            try Data().write(to: package.appendingPathComponent(file))
        }
        for directory in ["trunk", "trunk-mtp"] {
            try FileManager.default.createDirectory(
                at: package.appendingPathComponent(directory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
    }

    private static var fakeCAMLingerIdentity: String {
        let json = """
        {
          "camSemanticSHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "exportABISHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "exportID": "other",
          "flowID": "other",
          "inputPorts": [],
          "outputPorts": [],
          "authoredCapabilities": [],
          "matchedGateIDs": []
        }
        """
        return Data(json.utf8).base64EncodedString()
    }

    private static var legacyAdapterCAMLingerIdentity: String {
        let json = """
        {
          "protocolVersion": 1,
          "descriptorIdentityToken": null,
          "camSemanticSHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "exportABISHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "requestName": "linger text",
          "exportID": "generate",
          "flowID": "generate",
          "delivery": "worker",
          "inputPorts": [],
          "outputPorts": [],
          "authoredCapabilities": [],
          "matchedGateIDs": [],
          "adapter": "temporary-text-to-text-warm-runtime"
        }
        """
        return Data(json.utf8).base64EncodedString()
    }

    private static var protocolStampedCAMLingerIdentity: String {
        let json = """
        {
          "camSemanticSHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "exportABISHA256": "0000000000000000000000000000000000000000000000000000000000000000",
          "exportID": "generate",
          "flowID": "generate",
          "inputPorts": [],
          "outputPorts": [],
          "authoredCapabilities": [],
          "matchedGateIDs": [],
          "protocolVersion": 4
        }
        """
        return Data(json.utf8).base64EncodedString()
    }

    private struct TestLingerCAMIdentity: Codable, Equatable {
        let camSemanticSHA256: String
        let exportABISHA256: String
        let exportID: String
        let flowID: String
        let inputPorts: [TestLingerCAMPortRecord]
        let outputPorts: [TestLingerCAMPortRecord]
        let authoredCapabilities: [String]
        let matchedGateIDs: [String]
    }

    private struct TestLingerCAMPortRecord: Codable, Equatable, Comparable {
        let name: String
        let typeName: String
        let attributes: [String: String]
        let optional: Bool

        static func < (lhs: TestLingerCAMPortRecord, rhs: TestLingerCAMPortRecord) -> Bool {
            if lhs.name != rhs.name { return lhs.name < rhs.name }
            if lhs.typeName != rhs.typeName { return lhs.typeName < rhs.typeName }
            if lhs.optional != rhs.optional { return !lhs.optional && rhs.optional }
            let lhsAttributes = lhs.attributes.sorted { left, right in
                left.key == right.key ? left.value < right.value : left.key < right.key
            }
            let rhsAttributes = rhs.attributes.sorted { left, right in
                left.key == right.key ? left.value < right.value : left.key < right.key
            }
            for (left, right) in zip(lhsAttributes, rhsAttributes) {
                if left.key != right.key { return left.key < right.key }
                if left.value != right.value { return left.value < right.value }
            }
            return lhsAttributes.count < rhsAttributes.count
        }
    }

    private static func encodedCAMLingerIdentity(
        package: URL,
        request: SmeltCAMCapabilityRequest
    ) throws -> (encoded: String, decoded: TestLingerCAMIdentity, json: String) {
        let capabilities = try XCTUnwrap(
            SmeltCAMPackageCapabilities.loadIfPresent(packageURL: package)
        )
        let decision = try capabilities.resolve(request)
        let identity = TestLingerCAMIdentity(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256,
            exportID: decision.exportID,
            flowID: decision.flowID,
            inputPorts: decision.inputPorts.map(portRecord).sorted(),
            outputPorts: decision.outputPorts.map(portRecord).sorted(),
            authoredCapabilities: decision.authoredCapabilities.sorted(),
            matchedGateIDs: decision.matchedGateIDs.sorted()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(identity)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        return (data.base64EncodedString(), identity, json)
    }

    private static func portRecord(_ port: SmeltCAMPackageDescriptor.Port) -> TestLingerCAMPortRecord {
        TestLingerCAMPortRecord(
            name: port.portName,
            typeName: port.type.typeName,
            attributes: port.type.attributes,
            optional: port.optional
        )
    }

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func writeCAMDescriptor(_ name: String, to package: URL) throws {
        try camDescriptorData(name)
            .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))
    }

    private static func writeOptimizerReportInventoryFiles(to package: URL) throws {
        for relativePath in [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "SmeltGenerated.swift",
            "dispatches.bin",
            "prefill_dispatches.bin",
            "tokenizer.json",
            "tokenizer.bin",
        ] {
            try Data().write(to: package.appendingPathComponent(relativePath))
        }
    }

    private static func writeMutatedCAMDescriptor(
        _ name: String,
        to package: URL,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        try mutatedCAMDescriptorData(name, mutate: mutate)
            .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))
    }

    private static func appendExportCapability(
        in object: inout [String: Any],
        exportID: String,
        capability: String
    ) throws {
        var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
        let index = try XCTUnwrap(exports.indices.first { exports[$0]["exportID"] as? String == exportID })
        var capabilities = try XCTUnwrap(exports[index]["capabilities"] as? [String])
        capabilities.append(capability)
        exports[index]["capabilities"] = capabilities
        object["exports"] = exports
    }

    private static func setCompileRequirementValue(
        in object: inout [String: Any],
        key: String,
        value: String
    ) throws {
        var compile = try XCTUnwrap(object["compileRequirements"] as? [[String: Any]])
        let index = try XCTUnwrap(
            compile.indices.first { (compile[$0]["key"] as? String) == key }
        )
        compile[index]["value"] = value
        object["compileRequirements"] = compile
    }

    private static func replaceGateRequirements(
        in object: inout [String: Any],
        gateID: String,
        requirements: [[String: Any]]
    ) throws {
        var gates = try XCTUnwrap(object["gateContracts"] as? [[String: Any]])
        let index = try XCTUnwrap(gates.indices.first { gates[$0]["gateID"] as? String == gateID })
        gates[index]["requirements"] = requirements
        let requirementSubjects = Set(requirements.compactMap { $0["subject"] as? String })
        let measurements = (gates[index]["measurements"] as? [[String: Any]]) ?? []
        gates[index]["measurements"] = measurements.filter {
            guard let subject = $0["subject"] as? String else { return false }
            return requirementSubjects.contains(subject)
        }
        object["gateContracts"] = gates
    }

    private static func replaceTransformerLayerRoles(
        in object: inout [String: Any],
        blockID: String,
        roles: [String],
        removeDeltaShape: Bool = false
    ) throws {
        var blocks = try XCTUnwrap(object["blocks"] as? [[String: Any]])
        let index = try XCTUnwrap(blocks.indices.first {
            blocks[$0]["blockID"] as? String == blockID
        })
        var block = blocks[index]
        var shape = try XCTUnwrap(block["shape"] as? [String: Any])
        var transformer = try XCTUnwrap(shape["transformer"] as? [String: Any])
        var layers = try XCTUnwrap(transformer["layers"] as? [String: Any])
        layers["roles"] = roles
        transformer["layers"] = layers
        if removeDeltaShape {
            transformer.removeValue(forKey: "delta")
        }
        shape["transformer"] = transformer
        block["shape"] = shape
        blocks[index] = block
        object["blocks"] = blocks
    }

    private static func writeDispatchTable(_ name: String, to package: URL) throws {
        var record = SmeltDispatchRecord.empty()
        record.pipeline = 0
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 1
        record.tgH = 1
        record.tgD = 1
        let records = [record]
        let data = records.withUnsafeBytes { Data($0) }
        try data.write(to: package.appendingPathComponent(name))
    }

    private static func writeCAMTextInventoryFiles(to package: URL) throws {
        for file in [
            "SmeltGenerated.swift",
            "dispatches.bin",
            "model.metallib",
            "prefill_dispatches.bin",
            "tokenizer.bin",
            "tokenizer.json",
            "weights.bin",
        ] {
            try Data("fixture\n".utf8)
                .write(to: package.appendingPathComponent(file))
        }
    }

    private static func camDescriptorData(_ name: String) throws -> Data {
        let descriptor = try SmeltCAMPackageDescriptor(from: registryModuleIR(name))
        return try descriptor.canonicalJSONData()
    }

    private static func mutatedCAMDescriptorData(
        _ name: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws -> Data {
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: camDescriptorData(name)) as? [String: Any]
        )
        try mutate(&object)
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private static func mutatePortAttributes(
        in export: inout [String: Any],
        portListKey: String,
        index: Int,
        attributes: [String: String]
    ) throws {
        var ports = try XCTUnwrap(export[portListKey] as? [[String: Any]])
        var port = ports[index]
        var type = try XCTUnwrap(port["type"] as? [String: Any])
        type["attributes"] = attributes
        port["type"] = type
        ports[index] = port
        export[portListKey] = ports
    }

    private static func appendExportOutput(
        in object: inout [String: Any],
        exportID: String,
        portName: String,
        typeName: String,
        attributes: [String: String]
    ) throws {
        var exports = try XCTUnwrap(object["exports"] as? [[String: Any]])
        let index = try XCTUnwrap(exports.indices.first {
            exports[$0]["exportID"] as? String == exportID
        })
        var outputs = try XCTUnwrap(exports[index]["outputs"] as? [[String: Any]])
        outputs.append([
            "portName": portName,
            "optional": false,
            "type": ["typeName": typeName, "attributes": attributes],
        ])
        exports[index]["outputs"] = outputs
        object["exports"] = exports
    }

    private static func mutateGraphNodePortAttributes(
        in object: inout [String: Any],
        nodeID: String,
        portListKey: String,
        portName: String,
        attributes: [String: String]
    ) throws {
        var nodes = try XCTUnwrap(object["graphNodes"] as? [[String: Any]])
        let nodeIndex = try XCTUnwrap(nodes.indices.first { index in
            nodes[index]["nodeID"] as? String == nodeID
        })
        var node = nodes[nodeIndex]
        var ports = try XCTUnwrap(node[portListKey] as? [[String: Any]])
        let portIndex = try XCTUnwrap(ports.indices.first { index in
            ports[index]["portName"] as? String == portName
        })
        var port = ports[portIndex]
        var type = try XCTUnwrap(port["type"] as? [String: Any])
        type["attributes"] = attributes
        port["type"] = type
        ports[portIndex] = port
        node[portListKey] = ports
        nodes[nodeIndex] = node
        object["graphNodes"] = nodes
    }

    private static func mutateGraphEdgeValueType(
        in object: inout [String: Any],
        typeName: String,
        attributes: [String: String],
        where matches: ([String: Any], [String: Any]) -> Bool
    ) throws {
        var edges = try XCTUnwrap(object["graphEdges"] as? [[String: Any]])
        let edgeIndex = try XCTUnwrap(edges.indices.first { index in
            guard let from = edges[index]["from"] as? [String: Any],
                  let to = edges[index]["to"] as? [String: Any] else {
                return false
            }
            return matches(from, to)
        })
        edges[edgeIndex]["valueType"] = [
            "typeName": typeName,
            "attributes": attributes,
        ]
        object["graphEdges"] = edges
    }

    private static func assertDS4FeatureAdmissionFailure(
        _ result: (status: Int32, stdout: String, stderr: String),
        verb: String,
        label: String
    ) {
        XCTAssertEqual(result.status, 1, label)
        XCTAssertTrue(
            result.stderr.contains(
                "smelt \(verb): module feature admission failed at "
                    + "\(SmeltCAMFeatureAdmission.preBridgeStage)"
            ),
            "\(label): \(result.stderr)"
        )
        for fragment in [
            "transformer.rope.yarn",
            "transformer.moe.router",
            "transformer.moe.expert",
            "quant.storage.gptq",
            "quant.calibration.gptq",
            "compile.generated-kernels",
            "compile.memory-bound",
            "gate.quant-quality",
        ] {
            XCTAssertTrue(result.stderr.contains(fragment), "\(label): \(fragment): \(result.stderr)")
        }
        for forbidden in [
            "no manifest bridge route",
            "no CAM runtime route",
            "export 'generate' flow 'generate'",
            "CAM/manifest conflict",
            "Run failed",
            "tts run failed",
            "Create failed",
            "smelt serve ready",
            "trace replay",
            "unsupported trace suite",
            "invalid trace",
        ] {
            XCTAssertFalse(result.stderr.contains(forbidden), "\(label): \(forbidden): \(result.stderr)")
        }
    }


    private static func runSmelt(
        _ arguments: [String],
        currentDirectory: URL? = nil
    ) throws -> (status: Int32, stdout: String, stderr: String) {
        try runRawSmelt(
            labRoutedArguments(arguments),
            currentDirectory: currentDirectory
        )
    }

    private static func runRawSmelt(
        _ arguments: [String],
        currentDirectory: URL? = nil
    ) throws -> (status: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = try smeltExecutable()
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory

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

    private static func labRoutedArguments(_ arguments: [String]) -> [String] {
        guard let command = arguments.first else { return arguments }
        let tail = Array(arguments.dropFirst())
        switch command {
        case "verify": return ["lab", "verify"] + tail
        case "bench": return ["lab", "bench", "decode"] + tail
        case "prefill-bench": return ["lab", "bench", "prefill"] + tail
        case "mtp-bench": return ["lab", "bench", "speculative"] + tail
        case "bench-logprobs": return ["lab", "bench", "logprobs"] + tail
        case "profile": return ["lab", "profile", "decode"] + tail
        case "kernels": return ["lab", "profile", "decode"] + tail + ["--kernels"]
        case "prefill-kernels": return ["lab", "profile", "prefill"] + tail
        case "dispatches": return ["lab", "inspect", "dispatches"] + tail
        case "optimizer-report": return ["lab", "inspect", "cost"] + tail
        case "kernel-lab": return ["lab", "kernel"] + tail
        case "trace": return ["lab", "trace"] + tail
        case "replay": return ["lab", "replay"] + tail
        case "prefill": return ["lab", "prefill"] + tail
        case "optimize-next": return ["lab", "optimize"] + tail
        case "module-profile": return ["lab", "package-profile"] + tail
        default: return arguments
        }
    }

    private static func smeltExecutable() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["SMELT_TEST_EXECUTABLE"],
           !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw XCTSkip("SMELT_TEST_EXECUTABLE is not executable: \(override)")
            }
            return url
        }
        let candidates = [
            Bundle(for: ServeDescriptorDispatchTests.self).bundleURL
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
}
