import Foundation
import CryptoKit
import SmeltCompiler
import SmeltSchema
import XCTest

final class BuildCAMFrontDoorTests: XCTestCase {
    private enum FixtureError: Error, CustomStringConvertible {
        case malformed(String)

        var description: String {
            switch self {
            case .malformed(let reason):
                return "malformed CAM front-door fixture: \(reason)"
            }
        }
    }

    private func assertSameResolvedPath(
        _ actual: String?,
        _ expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.map { URL(fileURLWithPath: $0).resolvingSymlinksInPath().path },
            URL(fileURLWithPath: expected).resolvingSymlinksInPath().path,
            file: file,
            line: line
        )
    }

    func testSmeltBuildRejectsPackageSpecJSON() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let specURL = try writePackageSpecHeaderFixture(in: root)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            specURL.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("package spec JSON is internal; build a .module.json module instead"),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path))
    }

    func testSmeltBuildRejectsPackageSpecJSONBeforeLLMFallback() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let specURL = try writePackageSpecHeaderFixture(in: root)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try Self.runAgent([
            "build",
            specURL.path,
            "--output",
            package.path,
            "--shader-dir",
            root.path,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("package spec JSON is internal; build a .module.json module instead"),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    func testSmeltBuildRejectsCamSourcePackageFlagForAgentInput() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let agentURL = root.appendingPathComponent("qwen.txt")
        try Data("model \"demo\"\n".utf8).write(to: agentURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try Self.runAgent([
            "build",
            agentURL.path,
            "--module-source-package",
            "--output",
            package.path,
            "--shader-dir",
            root.path,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("--module-source-package requires a .module.json input"),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    func testSmeltBuildAcceptsCheckedCAMModule() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let camURL = Self.camURL("qwen35_text.cam")
        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: registryModuleIR(camURL.lastPathComponent),
            artifactRoot: source.path
        )
        try writeCAMArtifactRoot(to: source, projection: projection)

        let package = root.appendingPathComponent("qwen-cam.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen-cam-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--module-artifact-root",
            source.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertTrue(result.stderr.contains("Built: \(package.path)"), result.stderr)
        let manifestData = try Data(contentsOf: package.appendingPathComponent("manifest.json"))
        XCTAssertEqual(try SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData), .textGeneration)
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        XCTAssertNil(manifest.kind)
        XCTAssertNil(manifest.architecture)
        XCTAssertEqual(try packageFiles(at: package), projection.packageFiles)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: source.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName).path
            )
        )

        let evidenceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidence)) as? [String: Any]
        )
        XCTAssertEqual(evidenceJSON["evidence_schema"] as? String, "smelt.module.build_evidence.v3")
        XCTAssertNil(evidenceJSON["contract_id"])
        XCTAssertNil(evidenceJSON["contract_version"])
        XCTAssertNil(evidenceJSON["projected_package_contract_sha256"])
        XCTAssertNil(evidenceJSON["kind"])
        XCTAssertNil(evidenceJSON["runtime"])
        XCTAssertNil(evidenceJSON["spec_path"])
        XCTAssertNil(evidenceJSON["spec_sha256"])
        XCTAssertNil(evidenceJSON["source_manifest_sha256"])
        XCTAssertEqual(evidenceJSON["package_path"] as? String, package.path)
        assertSameResolvedPath(evidenceJSON["module_path"] as? String, moduleURL.path)
        XCTAssertEqual(evidenceJSON["package_projection_id"] as? String, projection.packageProjectionID)
        XCTAssertEqual(evidenceJSON["package_projection_version"] as? Int, projection.packageProjectionVersion)
        XCTAssertEqual(evidenceJSON["module_semantic_sha256"] as? String, projection.camSemanticSHA256)
        XCTAssertEqual(evidenceJSON["export_abi_sha256"] as? String, projection.exportABISHA256)
        XCTAssertEqual(evidenceJSON["descriptor_version"] as? Int, projection.descriptorVersion)
        XCTAssertEqual(
            evidenceJSON["descriptor_graph_signature_sha256"] as? String,
            projection.descriptorGraphSignatureSHA256
        )
        XCTAssertEqual(
            evidenceJSON["projected_package_spec_sha256"] as? String,
            projection.projectedPackageSpecSHA256
        )
        XCTAssertEqual(
            evidenceJSON["source_package_files"] as? [String],
            projection.packageFiles.filter { $0 != SmeltCAMPackageDescriptor.packageFileName }
        )
        XCTAssertEqual(
            evidenceJSON["generated_package_files"] as? [String],
            [SmeltCAMPackageDescriptor.packageFileName]
        )
        XCTAssertEqual((evidenceJSON["generated_payload_sha256"] as? String)?.count, 64)
        let descriptorData = try Data(
            contentsOf: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        )
        let expectedDescriptorData = try SmeltCAMPackageDescriptor(
            from: registryModuleIR(camURL.lastPathComponent)
        ).canonicalJSONData()
        XCTAssertEqual(descriptorData, expectedDescriptorData)
        XCTAssertEqual(
            evidenceJSON["module_descriptor_sha256"] as? String,
            Self.sha256Hex(descriptorData)
        )
        XCTAssertEqual((evidenceJSON["source_payload_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["package_payload_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["package_files"] as? [String])?.sorted(), projection.packageFiles)
        XCTAssertGreaterThanOrEqual(evidenceJSON["build_elapsed_ms"] as? Int ?? -1, 0)
    }

    func testSmeltBuildAcceptsCheckedFastPrefillCAMModule() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let camURL = Self.camURL("qwen35_fast.cam")
        let moduleURL = Self.moduleURL("qwen35_fast.cam")
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: registryModuleIR(camURL.lastPathComponent),
            artifactRoot: source.path
        )
        try writeCAMArtifactRoot(to: source, projection: projection)

        let package = root.appendingPathComponent("qwen-fast-cam.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen-fast-cam-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--module-artifact-root",
            source.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(projection.packageProjectionID, "text-to-text-transformer-prefill-decode-affine-u4-g64")
        XCTAssertTrue(projection.packageFiles.contains("prefill_dispatches.bin"))
        XCTAssertEqual(try packageFiles(at: package), projection.packageFiles)

        let manifest = try SmeltManifest.decode(
            from: Data(contentsOf: package.appendingPathComponent("manifest.json"))
        )
        XCTAssertNotNil(manifest.prefill)
        XCTAssertEqual(manifest.loop?.setupSignatures, ["prefill:trunk,text-head"])
        XCTAssertEqual(manifest.loop?.perStepSignatures, ["decode:trunk,text-head"])
        XCTAssertEqual(
            manifest.validation?.performanceGate,
            SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        XCTAssertEqual(
            manifest.validation?.performanceProfile?.gate,
            SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )

        let evidenceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidence)) as? [String: Any]
        )
        XCTAssertEqual(evidenceJSON["evidence_schema"] as? String, "smelt.module.build_evidence.v3")
        XCTAssertNil(evidenceJSON["contract_id"])
        XCTAssertNil(evidenceJSON["contract_version"])
        XCTAssertNil(evidenceJSON["projected_package_contract_sha256"])
        assertSameResolvedPath(evidenceJSON["module_path"] as? String, moduleURL.path)
        XCTAssertEqual(evidenceJSON["package_projection_id"] as? String, projection.packageProjectionID)
        XCTAssertEqual(evidenceJSON["package_projection_version"] as? Int, projection.packageProjectionVersion)
        XCTAssertEqual(evidenceJSON["module_semantic_sha256"] as? String, projection.camSemanticSHA256)
        XCTAssertEqual(evidenceJSON["export_abi_sha256"] as? String, projection.exportABISHA256)
        XCTAssertEqual(evidenceJSON["descriptor_version"] as? Int, projection.descriptorVersion)
        XCTAssertEqual(
            evidenceJSON["descriptor_graph_signature_sha256"] as? String,
            projection.descriptorGraphSignatureSHA256
        )
        XCTAssertEqual(
            evidenceJSON["projected_package_spec_sha256"] as? String,
            projection.projectedPackageSpecSHA256
        )
        XCTAssertEqual(
            evidenceJSON["source_package_files"] as? [String],
            projection.packageFiles.filter { $0 != "module.json" }
        )
        XCTAssertEqual(evidenceJSON["generated_package_files"] as? [String], ["module.json"])
        XCTAssertEqual((evidenceJSON["module_descriptor_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["package_files"] as? [String])?.sorted(), projection.packageFiles)
    }

    func testSmeltBuildAcceptsReasonerCheckedCAMModule() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let camURL = Self.camURL("qwen35_reasoner.cam")
        let moduleURL = Self.moduleURL("qwen35_reasoner.cam")
        let source = root.appendingPathComponent("qwen-reasoner-payloads", isDirectory: true)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: registryModuleIR(camURL.lastPathComponent),
            artifactRoot: source.path
        )
        try writeCAMArtifactRoot(to: source, projection: projection)

        let package = root.appendingPathComponent("qwen-reasoner-cam.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen-reasoner-cam-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--module-artifact-root",
            source.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            projection.packageProjectionID,
            "two-text-transformer-prefill-decode-affine-u4-g64-prompt-adapter"
        )
        XCTAssertTrue(projection.packageFiles.contains("prefill_dispatches.bin"))
        XCTAssertEqual(try packageFiles(at: package), projection.packageFiles)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: source.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName).path
            )
        )

        let manifest = try SmeltManifest.decode(
            from: Data(contentsOf: package.appendingPathComponent("manifest.json"))
        )
        XCTAssertNil(manifest.architecture)
        XCTAssertNotNil(manifest.prefill)
        XCTAssertEqual(manifest.loop?.setupSignatures, ["prefill:trunk,text-head"])
        XCTAssertEqual(manifest.loop?.perStepSignatures, ["decode:trunk,text-head"])
        XCTAssertEqual(
            manifest.validation?.performanceGate,
            SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        XCTAssertEqual(
            manifest.validation?.performanceProfile?.gate,
            SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )

        let evidenceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidence)) as? [String: Any]
        )
        XCTAssertEqual(evidenceJSON["evidence_schema"] as? String, "smelt.module.build_evidence.v3")
        XCTAssertNil(evidenceJSON["contract_id"])
        XCTAssertNil(evidenceJSON["contract_version"])
        XCTAssertNil(evidenceJSON["projected_package_contract_sha256"])
        XCTAssertNil(evidenceJSON["kind"])
        XCTAssertNil(evidenceJSON["runtime"])
        XCTAssertNil(evidenceJSON["spec_path"])
        XCTAssertNil(evidenceJSON["spec_sha256"])
        XCTAssertNil(evidenceJSON["source_manifest_sha256"])
        XCTAssertEqual(evidenceJSON["package_path"] as? String, package.path)
        assertSameResolvedPath(evidenceJSON["module_path"] as? String, moduleURL.path)
        XCTAssertEqual(evidenceJSON["package_projection_id"] as? String, projection.packageProjectionID)
        XCTAssertEqual(evidenceJSON["package_projection_version"] as? Int, projection.packageProjectionVersion)
        XCTAssertEqual(evidenceJSON["module_semantic_sha256"] as? String, projection.camSemanticSHA256)
        XCTAssertEqual(evidenceJSON["export_abi_sha256"] as? String, projection.exportABISHA256)
        XCTAssertEqual(evidenceJSON["descriptor_version"] as? Int, projection.descriptorVersion)
        XCTAssertEqual(
            evidenceJSON["descriptor_graph_signature_sha256"] as? String,
            projection.descriptorGraphSignatureSHA256
        )
        XCTAssertEqual(
            evidenceJSON["projected_package_spec_sha256"] as? String,
            projection.projectedPackageSpecSHA256
        )
        XCTAssertEqual(
            evidenceJSON["source_package_files"] as? [String],
            projection.packageFiles.filter { $0 != SmeltCAMPackageDescriptor.packageFileName }
        )
        XCTAssertEqual(
            evidenceJSON["generated_package_files"] as? [String],
            [SmeltCAMPackageDescriptor.packageFileName]
        )

        let descriptorData = try Data(
            contentsOf: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        )
        let expectedDescriptorData = try SmeltCAMPackageDescriptor(
            from: registryModuleIR(camURL.lastPathComponent)
        ).canonicalJSONData()
        XCTAssertEqual(descriptorData, expectedDescriptorData)
        XCTAssertEqual(
            evidenceJSON["module_descriptor_sha256"] as? String,
            Self.sha256Hex(descriptorData)
        )
        XCTAssertEqual((evidenceJSON["source_payload_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["generated_payload_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["package_payload_sha256"] as? String)?.count, 64)
        XCTAssertEqual((evidenceJSON["package_files"] as? [String])?.sorted(), projection.packageFiles)
    }

    func testSmeltBuildRejectsInPlaceCAMModuleStamping() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let camURL = Self.camURL("qwen35_text.cam")
        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let sourcePackage = root.appendingPathComponent("qwen-source.smeltpkg", isDirectory: true)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: registryModuleIR(camURL.lastPathComponent),
            artifactRoot: sourcePackage.path
        )
        try writeCAMArtifactRoot(to: sourcePackage, projection: projection)
        let evidence = root.appendingPathComponent("qwen-cam-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--module-artifact-root",
            sourcePackage.path,
            "--output",
            sourcePackage.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("--module-artifact-root and --output must be distinct paths"),
            result.stderr
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: sourcePackage
                    .appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
                    .path
            )
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path))
    }

    func testSmeltBuildRejectsRawCAMModuleWithoutArtifactRoot() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let package = root.appendingPathComponent("raw-cam.smeltpkg", isDirectory: true)

        let camResult = try Self.runAgent([
            "build",
            moduleURL.path,
            "--output",
            package.path,
        ])

        XCTAssertEqual(camResult.status, 1)
        XCTAssertTrue(
            camResult.stderr.contains(
                "missing required option for module build: --module-artifact-root"
            ),
            camResult.stderr
        )
        XCTAssertEqual(try packageFiles(at: package), [])
    }

    func testSmeltBuildRejectsLegacyOptionsForCAMInput() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let moduleURL = Self.moduleURL("qwen35_text.cam")
        for legacyOption in ["--weights-dir", "--optimizer-report"] {
            let package = root.appendingPathComponent(
                "\(legacyOption.dropFirst(2))-qwen-cam.smeltpkg",
                isDirectory: true
            )
            let evidence = root.appendingPathComponent(
                "\(legacyOption.dropFirst(2))-qwen-cam-build-evidence.json"
            )

            var command = [
                "build",
                moduleURL.path,
                "--output",
                package.path,
                "--module-artifact-root",
                root.path,
                "--module-build-evidence-json",
                evidence.path,
                legacyOption,
            ]
            if legacyOption == "--weights-dir" {
                command.append(root.path)
            }

            let result = try Self.runAgent(command)

            XCTAssertEqual(result.status, 1, legacyOption)
            XCTAssertTrue(
                result.stderr.contains("unsupported option for module build: \(legacyOption)"),
                result.stderr
            )
            XCTAssertFalse(FileManager.default.fileExists(atPath: package.path), legacyOption)
            XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path), legacyOption)
        }
    }

    func testSmeltBuildProjectsTokenizerLocatorChangesThroughCAMGraph() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        // Drift the tokenizer source locator directly on the authored IR value
        // (the lowered `hf-file` locator the `.cam` `"tokenizer.json"` token
        // produced), the grammar-free equivalent of the old text edit.
        let base = registryModuleIR("qwen35_text")
        let driftedIR = try mutatedModuleIR(base) { object in
            guard var sources = object["sources"] as? [[String: Any]],
                  let index = sources.firstIndex(where: { $0["id"] as? String == "tokenizer" })
            else {
                XCTFail("qwen35_text is missing its tokenizer source")
                return
            }
            sources[index]["locator"] = "Qwen/Qwen3.5-2B/tokenizer.model"
            object["sources"] = sources
        }
        XCTAssertNotEqual(try base.semanticSHA256(), try driftedIR.semanticSHA256())
        // `smelt build` consumes the authored IR (`.module.json`); write the
        // drifted IR out as the module artifact the command reads.
        let driftedModule = root.appendingPathComponent("qwen35_text_drifted.module.json")
        try driftedIR.canonicalJSONData(prettyPrinted: true).write(to: driftedModule)
        let originalProjection = try SmeltCAMCheckedPackageProjector.project(
            cam: base,
            artifactRoot: root.appendingPathComponent("original-payloads", isDirectory: true).path
        )
        let driftedProjection = try SmeltCAMCheckedPackageProjector.project(
            cam: driftedIR,
            artifactRoot: root.appendingPathComponent("qwen-payloads", isDirectory: true).path
        )
        XCTAssertNotEqual(
            originalProjection.camSemanticSHA256,
            driftedProjection.camSemanticSHA256
        )
        XCTAssertNotEqual(
            originalProjection.projectedPackageSpecSHA256,
            driftedProjection.projectedPackageSpecSHA256
        )

        let source = root.appendingPathComponent("qwen-payloads", isDirectory: true)
        try writeCAMArtifactRoot(to: source, projection: driftedProjection)
        let package = root.appendingPathComponent("qwen-drifted.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen-drifted-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            driftedModule.path,
            "--output",
            package.path,
            "--module-artifact-root",
            source.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertTrue(result.stderr.contains("Built: \(package.path)"), result.stderr)
        XCTAssertEqual(try packageFiles(at: package), driftedProjection.packageFiles)
        let evidenceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidence)) as? [String: Any]
        )
        assertSameResolvedPath(evidenceJSON["module_path"] as? String, driftedModule.path)
        XCTAssertEqual(
            evidenceJSON["module_semantic_sha256"] as? String,
            driftedProjection.camSemanticSHA256
        )
        XCTAssertEqual(
            evidenceJSON["projected_package_spec_sha256"] as? String,
            driftedProjection.projectedPackageSpecSHA256
        )
        let descriptorData = try Data(
            contentsOf: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        )
        let expectedDescriptorData = try SmeltCAMPackageDescriptor(
            from: driftedIR
        ).canonicalJSONData()
        XCTAssertEqual(descriptorData, expectedDescriptorData)
    }

    func testSmeltBuildAcceptsQwen3TTSCheckedCAMModule() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("qwen3-tts-payloads", isDirectory: true)
        let camURL = Self.camURL("qwen3_tts.cam")
        let moduleURL = Self.moduleURL("qwen3_tts.cam")
        let cam = registryModuleIR(camURL.lastPathComponent)
        try writeQwen3TTSPayloads(to: source, cam: cam)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: source.path
        )

        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("qwen3-tts-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--output",
            package.path,
            "--module-artifact-root",
            source.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            projection.packageProjectionID,
            "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g128-sidecars"
        )
        for path in projection.packageFiles {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: package.appendingPathComponent(path).path),
                path
            )
        }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: source.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName).path
            )
        )

        let manifest = try Qwen3TTSManifest.decode(
            from: Data(contentsOf: package.appendingPathComponent("manifest.json"))
        )
        XCTAssertEqual(manifest.validation?.performanceGate, SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        XCTAssertEqual(projection.spec.validation.performanceGate, SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        XCTAssertEqual(
            projection.spec.validation.performanceProfile?.gate,
            SmeltPackagePerformanceGateID.qwen3TTSTTFA
        )

        let evidenceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidence)) as? [String: Any]
        )
        XCTAssertEqual(evidenceJSON["evidence_schema"] as? String, "smelt.module.build_evidence.v3")
        XCTAssertEqual(evidenceJSON["package_path"] as? String, package.path)
        assertSameResolvedPath(evidenceJSON["module_path"] as? String, moduleURL.path)
        XCTAssertEqual(evidenceJSON["package_projection_id"] as? String, projection.packageProjectionID)
        XCTAssertEqual(evidenceJSON["package_projection_version"] as? Int, projection.packageProjectionVersion)
        XCTAssertEqual(evidenceJSON["module_semantic_sha256"] as? String, projection.camSemanticSHA256)
        XCTAssertEqual(evidenceJSON["export_abi_sha256"] as? String, projection.exportABISHA256)
        XCTAssertEqual(
            evidenceJSON["descriptor_graph_signature_sha256"] as? String,
            projection.descriptorGraphSignatureSHA256
        )
        XCTAssertEqual(
            evidenceJSON["projected_package_spec_sha256"] as? String,
            projection.projectedPackageSpecSHA256
        )
        XCTAssertEqual(
            evidenceJSON["source_package_files"] as? [String],
            projection.packageFiles.filter { $0 != SmeltCAMPackageDescriptor.packageFileName }
        )
        XCTAssertEqual(
            evidenceJSON["generated_package_files"] as? [String],
            [SmeltCAMPackageDescriptor.packageFileName]
        )
        XCTAssertEqual((evidenceJSON["package_files"] as? [String])?.sorted(), projection.packageFiles)

        let descriptorData = try Data(
            contentsOf: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        )
        let expectedDescriptorData = try SmeltCAMPackageDescriptor(
            from: cam
        ).canonicalJSONData()
        XCTAssertEqual(descriptorData, expectedDescriptorData)
        XCTAssertEqual(
            evidenceJSON["module_descriptor_sha256"] as? String,
            Self.sha256Hex(descriptorData)
        )
    }

    func testSmeltBuildKeepsProjectedButNotBuildCoveredGuard() throws {
        let sourceURL = Self.repoRoot
            .appendingPathComponent("Sources", isDirectory: true)
            .appendingPathComponent("SmeltCompiler", isDirectory: true)
            .appendingPathComponent("SmeltCAMPackageBuilder.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        XCTAssertTrue(source.contains("guard projection.buildCommandCovered else"))
        XCTAssertTrue(source.contains("checked-package-projected"))
        XCTAssertTrue(source.contains("not build-command-covered"))
    }

    func testSmeltBuildRejectsDS4HeavyQuantBeforePackageWrites() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let package = root.appendingPathComponent("ds4-cam.smeltpkg", isDirectory: true)
        let evidence = root.appendingPathComponent("ds4-cam-build-evidence.json")

        let result = try Self.runAgent([
            "build",
            Self.moduleURL("ds4_heavy_quant.cam").path,
            "--output",
            package.path,
            "--module-artifact-root",
            source.path,
            "--module-build-evidence-json",
            evidence.path,
        ])

        XCTAssertEqual(result.status, 1)
        for fragment in [
            "unsupported CAM package projection features",
            "transformer.rope.yarn",
            "transformer.moe.router",
            "transformer.moe.expert",
            "quant.storage.gptq",
            "quant.calibration.gptq",
            "compile.generated-kernels",
            "compile.memory-bound",
            "gate.quant-quality",
        ] {
            XCTAssertTrue(result.stderr.contains(fragment), "\(fragment): \(result.stderr)")
        }
        XCTAssertFalse(result.stderr.contains("no checked package projection profile"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path))
    }

    func testSmeltBuildRejectsUnexpectedCAMArgument() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--output",
            package.path,
            "--module-artifact-root",
            root.path,
            "--module-build-evidence-json",
            root.appendingPathComponent("evidence.json").path,
            "stray",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("unexpected argument for module build: stray"),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    func testSmeltBuildRejectsDuplicateCAMOption() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--output",
            package.path,
            "--module-artifact-root",
            root.path,
            "--module-build-evidence-json",
            root.appendingPathComponent("evidence.json").path,
            "--output",
            root.appendingPathComponent("other.smeltpkg").path,
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains("duplicate option for module build: --output"),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    func testSmeltBuildRejectsMissingCAMOptionValue() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let moduleURL = Self.moduleURL("qwen35_text.cam")
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try Self.runAgent([
            "build",
            moduleURL.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
        ])

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(
            result.stderr.contains(
                "missing value for module build option: --module-build-evidence-json"
            ),
            result.stderr
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    private func writePackageSpecHeaderFixture(in root: URL) throws -> URL {
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try Data(#"{"version":1,"package_name":"qwen.cam"}"#.utf8).write(to: specURL)
        return specURL
    }

    private func writeCAMArtifactRoot(
        to directory: URL,
        projection: SmeltCAMCheckedPackageProjection
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for path in projection.packageFiles
            where path != "manifest.json"
                && path != SmeltCAMPackageDescriptor.packageFileName {
            let url = directory.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("cam payload: \(path)\n".utf8).write(to: url)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifest = try camManifestFixture(for: projection)
        let manifestData = try encoder.encode(manifest)
        _ = try SmeltManifest.decode(from: manifestData)
        try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
    }

    private func camManifestFixture(
        for projection: SmeltCAMCheckedPackageProjection
    ) throws -> SmeltManifest {
        let spec = projection.spec
        let architecture = try object(spec.architectureConfig, label: "architecture_config")
        let prefill = try architecture["prefill"].map {
            try object($0, label: "architecture_config.prefill")
        }
        let prefillManifest = try prefill.map {
            SmeltPrefillManifest(
                engine: try string("engine", in: $0),
                modelPath: try string("model", in: $0),
                maxBatchSize: try int("max_batch_size", in: $0),
                handoff: SmeltHandoffTable(
                    entries: dummyHandoffEntries(
                        count: try int("handoff_entries", in: $0)
                    ),
                    ropeCosSlot: 0,
                    ropeSinSlot: 1
                ),
                inputContract: SmeltPrefillInputContract()
            )
        }
        let buildProvenance = try textBuildProvenanceFixture(
            spec: spec,
            architecture: architecture
        )

        return SmeltManifest(
            blocks: spec.blocks,
            loop: spec.loop,
            modelName: spec.modelName,
            config: SmeltManifestConfig(
                hiddenSize: try int("hidden_size", in: architecture),
                numLayers: try int("num_layers", in: architecture),
                vocabSize: try int("vocab_size", in: architecture),
                hiddenActivation: try optionalString("hidden_activation", in: architecture),
                staticSeqCapacity: try int("static_seq_capacity", in: architecture),
                ropeDim: try int("rope_dim", in: architecture),
                numDeltaLayers: try int("num_delta_layers", in: architecture),
                numAttnLayers: try int("num_attn_layers", in: architecture),
                ffnDim: try int("ffn_dim", in: architecture),
                blockTopology: try optionalString("block_topology", in: architecture),
                layerPattern: try manifestLayerPattern(in: architecture),
                attentionByRole: try manifestAttentionByRole(in: architecture),
                perLayerInput: try manifestPerLayerInput(in: architecture),
                logitCap: try optionalNumber("logit_cap", in: architecture).map(Float.init),
                sharedKVLayers: try optionalInt("shared_kv_layers", in: architecture),
                turboQuantHPatterns: try optionalStringArray("turbo_quant_h", in: architecture)
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: "",
                prefillDispatchesBin: prefill == nil ? nil : "",
                tokenizerJSON: nil
            ),
            buildProvenance: buildProvenance,
            device: SmeltDeviceRequirements(metalFamily: .apple7, minMemoryBytes: 1),
            weights: SmeltWeightManifest(
                totalBytes: UInt64(try int("weight_total_bytes", in: architecture)),
                entries: []
            ),
            buffers: SmeltBufferTable(slots: []),
            pipelines: try stringArray("pipelines", in: architecture),
            slotLayout: emptySlotLayout(),
            prefill: prefillManifest,
            inference: try required(spec.inference, label: "spec.inference"),
            decode: try required(spec.decode, label: "spec.decode"),
            validation: spec.validation
        )
    }

    private func textBuildProvenanceFixture(
        spec: SmeltPackageSpec,
        architecture: [String: SmeltPackageSpecValue]
    ) throws -> SmeltBuildProvenance {
        let loading = try object(
            try value("loading", in: architecture),
            label: "architecture_config.loading"
        )
        let layers = try architecture["layers"].map {
            try object($0, label: "architecture_config.layers")
        }
        let prefill = try architecture["prefill"].map {
            try object($0, label: "architecture_config.prefill")
        }
        let quantization = try required(spec.quantization, label: "spec.quantization")
        let groupSize = try required(quantization.groupSize, label: "spec.quantization.groupSize")
        let layerPatternUnit = try layers.map { try stringArray("pattern", in: $0) } ?? ["attention"]
        let layerPatternRepeats = try layers.map { try int("repeats", in: $0) }
            ?? int("num_layers", in: architecture)
        return SmeltBuildProvenance(
            buildFingerprint: Self.sha256Hex(Data("fixture-build".utf8)),
            weightsFingerprint: Self.sha256Hex(Data("fixture-weights".utf8)),
            specSHA256: Self.sha256Hex(Data("fixture-spec".utf8)),
            compilerSourcesSHA256: Self.sha256Hex(Data("fixture-compiler".utf8)),
            shaderSourcesSHA256: Self.sha256Hex(Data("fixture-shader".utf8)),
            resolvedOptions: SmeltResolvedBuildOptions(
                layerPatternUnit: layerPatternUnit,
                layerPatternRepeats: layerPatternRepeats,
                quantizationStrategy: quantization.format.rawValue,
                groupSize: groupSize,
                excludePatterns: [],
                quantizeEmbedding: true,
                loadingStrategy: try string("strategy", in: loading),
                packing: try string("packing", in: loading),
                checkpointMap: try optionalString("checkpoint_map", in: loading),
                prefillEngine: try prefill.map { try string("engine", in: $0) },
                maxPrefillBatch: try prefill.map { try int("max_batch_size", in: $0) },
                prefillHandoffFamilies: [],
                prefillEmitAllLogits: true,
                prefillVerifyArgmax: false,
                inferenceMaxTokens: try required(spec.inference, label: "spec.inference").maxTokens,
                eosTokens: try required(spec.inference, label: "spec.inference").eosTokens,
                thinkToken: try required(spec.inference, label: "spec.inference").thinkToken,
                thinkEndToken: try required(spec.inference, label: "spec.inference").thinkEndToken,
                thinkSkipSuffix: try required(spec.inference, label: "spec.inference").thinkSkipSuffix,
                tiedLMHead: true,
                normMode: "one-plus-weight",
                traceMode: "stripped-markers",
                turboQuantHPatterns: try optionalStringArray("turbo_quant_h", in: architecture),
                preserveNativePatterns: [],
                compilationGeneratedKernels: "auto",
                compilationWeightLayout: "memory_neutral"
            )
        )
    }

    private func writeQwen3TTSPayloads(to directory: URL, cam: SmeltCAMIR) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        let specs = try qwen3TTSRunnableSpecs(cam: cam)
        let orderedSpecs = specs.sorted { $0.name < $1.name }
        let layout = Qwen3TTSPackageBuilder.planLayout(orderedSpecs, pageSize: profile.pageSize)
        let pipelines = ["qwen3_tts_test_kernel"]
        let graph = profile.graph(textEmbeddingIsBF16: false)
        let validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: profile.modelName,
            performanceGate: profile.performanceGate,
            structureProfile: profile.structureProfile(pipelines: pipelines, graph: graph)
        )
        let manifest = Qwen3TTSManifest(
            version: 1,
            blocks: graph,
            loop: profile.loop,
            modelName: profile.modelName,
            pageSize: profile.pageSize,
            pipelines: pipelines,
            eosTokens: profile.eosTokens,
            totalBytes: layout.totalBytes,
            weights: layout.entries,
            tokenizerFiles: profile.tokenizerFiles,
            decode: .init(
                doSample: false,
                temperature: 1,
                topK: 50,
                subtalkerTemperature: 1,
                subtalkerTopK: 50
            ),
            validation: validation
        )
        try manifest.encoded().write(to: directory.appendingPathComponent("manifest.json"))
        try Data(repeating: 0, count: Int(layout.totalBytes)).write(
            to: directory.appendingPathComponent("weights.bin")
        )
        try Data("metallib".utf8).write(to: directory.appendingPathComponent("model.metallib"))
        for tokenizerFile in profile.tokenizerFiles {
            try Data("tokenizer".utf8).write(to: directory.appendingPathComponent(tokenizerFile))
        }
        for sidecar in SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks {
            try writeQwen3TTSSidecarPayload(
                to: directory.appendingPathComponent(sidecar.path, isDirectory: true),
                profile: sidecar,
                totalBytes: layout.totalBytes
            )
        }
    }

    private func writeQwen3TTSSidecarPayload(
        to directory: URL,
        profile: SmeltHeadlessTrunkSidecarProfile,
        totalBytes: UInt64
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("dispatches".utf8).write(to: directory.appendingPathComponent("dispatches.bin"))
        try Data("prefill-dispatches".utf8).write(
            to: directory.appendingPathComponent("prefill_dispatches.bin")
        )
        try Data("generated".utf8).write(
            to: directory.appendingPathComponent("SmeltGenerated.swift")
        )
        try FileManager.default.createSymbolicLink(
            atPath: directory.appendingPathComponent("weights.bin").path,
            withDestinationPath: "../weights.bin"
        )
        try FileManager.default.createSymbolicLink(
            atPath: directory.appendingPathComponent("model.metallib").path,
            withDestinationPath: "../model.metallib"
        )
        let manifest = SmeltManifest(
            kind: nil,
            headlessTrunkABI: true,
            blocks: nil,
            loop: nil,
            modelName: profile.modelName,
            config: SmeltManifestConfig(
                hiddenSize: 64,
                numLayers: 1,
                vocabSize: 128,
                staticSeqCapacity: 32,
                ropeDim: 32,
                numDeltaLayers: 0,
                numAttnLayers: 1,
                ffnDim: 128
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: "",
                prefillDispatchesBin: ""
            ),
            device: SmeltDeviceRequirements(metalFamily: .apple7, minMemoryBytes: 1),
            weights: SmeltWeightManifest(totalBytes: totalBytes, entries: []),
            buffers: SmeltBufferTable(slots: []),
            pipelines: ["qwen_trunk_kernel"],
            slotLayout: emptySlotLayout(),
            prefill: SmeltPrefillManifest(
                engine: "metal",
                modelPath: "prefill.mlmodelc",
                maxBatchSize: 256,
                handoff: SmeltHandoffTable(entries: [], ropeCosSlot: 0, ropeSinSlot: 1),
                inputContract: SmeltPrefillInputContract()
            ),
            inference: nil,
            decode: nil,
            validation: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
    }

    private func qwen3TTSRunnableSpecs(cam: SmeltCAMIR) throws -> [Qwen3TTSPackageBuilder.WeightSpec] {
        let hidden = 64
        let headDim = 32
        let qDim = 64
        let kvDim = 32
        let inter = 128
        let vocab = 128
        let groupSize = 128
        let residualSlots = try qwen3TTSResidualMTPWeightSlots(from: cam)

        var specs: [Qwen3TTSPackageBuilder.WeightSpec] = [
            qwen3TTSU4Weight(name: "talker.model.text_embedding.weight", shape: [vocab, hidden]),
            .init(name: "talker.text_projection.linear_fc1.weight", shape: [hidden, hidden]),
            .init(name: "talker.text_projection.linear_fc1.bias", shape: [hidden]),
            .init(name: "talker.text_projection.linear_fc2.weight", shape: [hidden, hidden]),
            .init(name: "talker.text_projection.linear_fc2.bias", shape: [hidden]),
            .init(name: "talker.model.codec_embedding.weight", shape: [vocab, hidden]),
            .init(name: "talker.codec_head.weight", shape: [vocab, hidden], dtype: .bf16),
            .init(
                name: "talker.code_predictor.small_to_mtp_projection.weight",
                shape: [hidden, hidden],
                dtype: .bf16
            ),
            .init(name: "talker.code_predictor.small_to_mtp_projection.bias", shape: [hidden]),
            .init(name: "decoder.pre_conv.conv.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.quantizer.rvq_first.output_proj.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.pre_transformer.input_proj.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.upsample.0.0.conv.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.decoder.6.conv.weight", shape: [2, 2], dtype: .bf16),
        ]

        for index in 0..<residualSlots {
            specs.append(
                .init(
                    name: "talker.code_predictor.lm_head.\(index).weight",
                    shape: [vocab, hidden],
                    dtype: .bf16
                )
            )
            specs.append(
                .init(
                    name: "talker.code_predictor.model.codec_embedding.\(index).weight",
                    shape: [vocab, hidden],
                    dtype: .bf16
                )
            )
        }

        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter,
            groupSize: groupSize
        )
        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.code_predictor.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter,
            groupSize: groupSize
        )
        return specs
    }

    private func qwen3TTSResidualMTPWeightSlots(from cam: SmeltCAMIR) throws -> Int {
        let blocks = cam.blocks.filter { block in
            block.shape.requirements.contains { $0.key == "residual-codebooks" }
        }
        guard blocks.count == 1, let block = blocks.first else {
            throw FixtureError.malformed(
                "expected one residual-codebooks block, found \(blocks.count)"
            )
        }
        let totalCodebooks = try qwen3TTSIntRequirement("codebooks", in: block)
        let residualCodebooks = try qwen3TTSIntRequirement("residual-codebooks", in: block)
        guard totalCodebooks == residualCodebooks else {
            throw FixtureError.malformed(
                "codebooks \(totalCodebooks) != residual-codebooks \(residualCodebooks)"
            )
        }
        guard totalCodebooks > 1 else {
            throw FixtureError.malformed("codebooks must include codebook 0 and residual slots")
        }
        return totalCodebooks - 1
    }

    private func qwen3TTSIntRequirement(
        _ key: String,
        in block: SmeltCAMIR.Block
    ) throws -> Int {
        let values = block.shape.requirements.filter { $0.key == key }.compactMap(\.value)
        guard values.count == 1, let raw = values.first, let value = Int(raw) else {
            throw FixtureError.malformed(
                "expected one integer \(key) requirement on block \(block.id)"
            )
        }
        return value
    }

    private func appendQwen3TTSTrunkLayerSpecs(
        to specs: inout [Qwen3TTSPackageBuilder.WeightSpec],
        prefix: String,
        hidden: Int,
        headDim: Int,
        qDim: Int,
        kvDim: Int,
        inter: Int,
        groupSize: Int
    ) {
        let layer = "\(prefix)layers.0"
        specs.append(contentsOf: [
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.q_proj.weight",
                shape: [qDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.k_proj.weight",
                shape: [kvDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.v_proj.weight",
                shape: [kvDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.o_proj.weight",
                shape: [hidden, qDim],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.gate_proj.weight",
                shape: [inter, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.up_proj.weight",
                shape: [inter, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.down_proj.weight",
                shape: [hidden, inter],
                groupSize: groupSize
            ),
            .init(name: "\(layer).input_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).post_attention_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).self_attn.q_norm.weight", shape: [headDim]),
            .init(name: "\(layer).self_attn.k_norm.weight", shape: [headDim]),
            .init(name: "\(prefix)norm.weight", shape: [hidden]),
        ])
    }

    private func qwen3TTSU4Weight(
        name: String,
        shape: [Int],
        groupSize: Int = 128
    ) -> Qwen3TTSPackageBuilder.WeightSpec {
        .init(name: name, shape: shape, dtype: .u4, groupSize: groupSize)
    }

    private func dummyHandoffEntries(count: Int) -> [SmeltResolvedHandoff] {
        (0..<count).map {
            SmeltResolvedHandoff(
                tensorName: "fixture_handoff_\($0)",
                slotIndex: $0,
                expectedElements: 1
            )
        }
    }

    private func object(
        _ value: SmeltPackageSpecValue,
        label: String
    ) throws -> [String: SmeltPackageSpecValue] {
        guard case .object(let object) = value else {
            throw FixtureError.malformed("\(label) is not an object")
        }
        return object
    }

    private func value(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> SmeltPackageSpecValue {
        guard let value = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is missing")
        }
        return value
    }

    private func int(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Int {
        guard case .int(let value)? = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is not an integer")
        }
        return value
    }

    private func optionalInt(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Int? {
        guard let value = object[key] else { return nil }
        guard case .int(let int) = value else {
            throw FixtureError.malformed("architecture_config.\(key) is not an integer")
        }
        return int
    }

    private func string(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> String {
        guard case .string(let value)? = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is not a string")
        }
        return value
    }

    private func optionalString(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> String? {
        guard let value = object[key] else { return nil }
        guard case .string(let string) = value else {
            throw FixtureError.malformed("architecture_config.\(key) is not a string")
        }
        return string
    }

    private func bool(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Bool {
        guard case .bool(let value)? = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is not a bool")
        }
        return value
    }

    private func number(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Double {
        guard let value = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is missing")
        }
        if case .number(let number) = value {
            return number
        }
        if case .int(let int) = value {
            return Double(int)
        }
        throw FixtureError.malformed("architecture_config.\(key) is not a number")
    }

    private func optionalNumber(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Double? {
        guard object[key] != nil else { return nil }
        return try number(key, in: object)
    }

    private func stringArray(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> [String] {
        guard case .array(let values)? = object[key] else {
            throw FixtureError.malformed("architecture_config.\(key) is not an array")
        }
        return try values.map { value in
            guard case .string(let string) = value else {
                throw FixtureError.malformed("architecture_config.\(key) has non-string item")
            }
            return string
        }
    }

    private func optionalStringArray(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> [String] {
        guard object[key] != nil else { return [] }
        return try stringArray(key, in: object)
    }

    private func manifestLayerPattern(
        in architecture: [String: SmeltPackageSpecValue]
    ) throws -> SmeltManifestLayerPattern? {
        guard let value = architecture["layers"] else { return nil }
        let layers = try object(value, label: "architecture_config.layers")
        return SmeltManifestLayerPattern(
            pattern: try stringArray("pattern", in: layers),
            repeats: try int("repeats", in: layers)
        )
    }

    private func manifestAttentionByRole(
        in architecture: [String: SmeltPackageSpecValue]
    ) throws -> [String: SmeltManifestRoleAttentionConfig] {
        guard let value = architecture["attention"] else { return [:] }
        let attention = try object(value, label: "architecture_config.attention")
        return try Dictionary(uniqueKeysWithValues: attention.map { role, value in
            let config = try object(value, label: "architecture_config.attention.\(role)")
            return (
                role,
                SmeltManifestRoleAttentionConfig(
                    qHeads: try int("q_heads", in: config),
                    kvHeads: try int("kv_heads", in: config),
                    headDim: try int("head_dim", in: config),
                    qkNorm: try bool("qk_norm", in: config),
                    vNorm: try bool("v_norm", in: config),
                    ropeTheta: try number("rope_theta", in: config),
                    ropeDim: try int("rope_dim", in: config),
                    slidingWindow: try int("sliding_window", in: config)
                )
            )
        })
    }

    private func manifestPerLayerInput(
        in architecture: [String: SmeltPackageSpecValue]
    ) throws -> SmeltManifestPerLayerInputConfig? {
        guard let value = architecture["per_layer_input"] else { return nil }
        let input = try object(value, label: "architecture_config.per_layer_input")
        return SmeltManifestPerLayerInputConfig(
            hiddenSize: try int("hidden_size", in: input),
            vocabSize: try int("vocab_size", in: input)
        )
    }

    private func required<T>(_ value: T?, label: String) throws -> T {
        guard let value else {
            throw FixtureError.malformed("\(label) is missing")
        }
        return value
    }

    private func emptySlotLayout() -> SmeltSlotLayout {
        SmeltSlotLayout(
            convStateBaseSlot: 0,
            recStateBaseSlot: 0,
            keyCacheBaseSlot: 0,
            valCacheBaseSlot: 0,
            ropeCosSlot: 0,
            ropeSinSlot: 0,
            tokenIdSlot: 0,
            positionSlot: 0,
            weightsSlot: 0
        )
    }

    private func packageFiles(at package: URL) throws -> [String] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: package.path) else { return [] }
        return enumerator.compactMap { item -> String? in
            guard let path = item as? String else { return nil }
            var isDirectory = ObjCBool(false)
            fm.fileExists(
                atPath: package.appendingPathComponent(path).path,
                isDirectory: &isDirectory
            )
            return isDirectory.boolValue ? nil : path
        }.sorted()
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-build-cam-front-door-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
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

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func iaExecutable() throws -> URL {
        let candidates = [
            Bundle(for: BuildCAMFrontDoorTests.self).bundleURL
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

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func camURL(_ name: String) -> URL {
        repoRoot
            .appendingPathComponent("Examples", isDirectory: true)
            .appendingPathComponent("CAM", isDirectory: true)
            .appendingPathComponent(name)
    }

    /// The checked-in authored IR artifact (`Models/<id>.module.json`) that
    /// `smelt build` consumes, derived from the `<id>.cam` oracle name.
    private static func moduleURL(_ name: String) -> URL {
        let id = name.hasSuffix(".cam") ? String(name.dropLast(4)) : name
        return repoRoot
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("\(id).module.json")
    }
}
