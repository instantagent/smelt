import Foundation
import CryptoKit
import Testing
@testable import SmeltCompiler
import SmeltSchema

@Suite struct SmeltPackageSpecBuilderTests {
    @Test func packageSpecBuilderAssemblesPrematerializedTextPackage() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        #expect(try packageFiles(at: package) == [
            "SmeltGenerated.swift",
            "manifest.json",
            "tokenizer.json",
            "weights.bin",
        ])
        let manifestData = try Data(contentsOf: package.appendingPathComponent("manifest.json"))
        #expect(try SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData) == .textGeneration)
        #expect(try string(at: package.appendingPathComponent("weights.bin")) == "weights")
    }

    @Test func packageSpecBuilderPreservesManifestValidationPolicy() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "fixtures/qwen35",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, validation: validation)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", validation: validation), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        let manifest = try JSONDecoder().decode(
            StubManifest.self,
            from: Data(contentsOf: package.appendingPathComponent("manifest.json"))
        )
        #expect(manifest.kind == nil)
        #expect(manifest.validation == validation)
    }

    @Test func packageSpecBuilderWritesBuildEvidence() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let evidenceURL = root.appendingPathComponent("build-evidence.json")
        let tool = URL(fileURLWithPath: "/bin/echo")

        let result = try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path,
            evidencePath: evidenceURL.path,
            command: [
                tool.path,
                "build",
                specURL.path,
                "--output",
                package.path,
                "--module-build-evidence-json",
                evidenceURL.path,
            ]
        )

        #expect(result.packagePath == package.path)
        #expect(result.specSHA256 == Self.sha256Hex(try Data(contentsOf: specURL)))
        #expect(result.sourceManifestSHA256.count == 64)
        #expect(result.sourcePayloadSHA256.count == 64)
        #expect(result.packagePayloadSHA256.count == 64)
        #expect(result.resolvedPlanSignature.contains {
            $0.hasPrefix("file:manifest.json:") && $0.contains("manifest")
        })
        #expect(result.resolvedPlanSignatureSHA256.count == 64)
        let evidence = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidenceURL)) as? [String: Any]
        )
        #expect(evidence["evidence_schema"] as? String == "smelt.package_spec.build_evidence.v1")
        #expect(evidence["kind"] as? String == "smelt.module.package_spec_build_evidence")
        #expect(evidence["command"] as? [String] == [
            tool.path,
            "build",
            specURL.path,
            "--output",
            package.path,
            "--module-build-evidence-json",
            evidenceURL.path,
        ])
        #expect(evidence["tool_sha256"] as? String == Self.sha256Hex(try Data(contentsOf: tool)))
        #expect(evidence["spec_sha256"] as? String == result.specSHA256)
        #expect(evidence["source_manifest_sha256"] as? String == result.sourceManifestSHA256)
        #expect(evidence["source_payload_sha256"] as? String == result.sourcePayloadSHA256)
        #expect(evidence["package_payload_sha256"] as? String == result.packagePayloadSHA256)
        #expect(evidence["resolved_plan_signature"] as? [String] == result.resolvedPlanSignature)
        #expect(
            evidence["resolved_plan_signature_sha256"] as? String
                == result.resolvedPlanSignatureSHA256
        )
        let resolvedPlanFiles = try #require(
            evidence["resolved_plan_package_files"] as? [[String: Any]]
        )
        #expect(resolvedPlanFiles.map { $0["path"] as? String } == result.packageFiles)
        #expect(evidence["package_files"] as? [String] == result.packageFiles)
        let runtime = try #require(evidence["runtime"] as? [String: Any])
        #expect(runtime["kind"] == nil)
        #expect(runtime["architecture"] as? String == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
    }

    @Test func packageSpecBuilderBuildsProjectedCAMSpecWithCAMEvidence() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let spec = withCAMDescriptorArtifact(textSpec(sourcePath: "payloads"))
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let evidenceURL = root.appendingPathComponent("cam-build-evidence.json")
        let camURL = root.appendingPathComponent("qwen35_text.module.json")
        try Data("cam".utf8).write(to: camURL)
        let descriptorData = Data("""
        {"descriptorSchema":"smelt.module.package_descriptor.v2","descriptorVersion":2}
        """.utf8)
        let tool = URL(fileURLWithPath: "/bin/echo")
        let identity = SmeltPackageSpecBuilder.CAMBuildIdentity(
            camPath: camURL.path,
            packageProjectionID: "text-to-text-transformer-prefill-decode-affine-u4-g64",
            packageProjectionVersion: 1,
            camSemanticSHA256: Self.sha256Hex(Data("cam-semantic".utf8)),
            exportABISHA256: Self.sha256Hex(Data("export-abi".utf8)),
            descriptorVersion: 2,
            descriptorGraphSignatureSHA256: Self.sha256Hex(Data("graph".utf8)),
            projectedPackageSpecSHA256: Self.sha256Hex(Data("projected-package-spec".utf8))
        )

        let result = try SmeltPackageSpecBuilder.build(
            spec: spec,
            camIdentity: identity,
            sourceBaseDirectory: root.path,
            outputDirectory: package.path,
            generatedFiles: [
                .init(
                    path: SmeltCAMPackageDescriptor.packageFileName,
                    data: descriptorData
                ),
            ],
            evidencePath: evidenceURL.path,
            command: [
                tool.path,
                "build",
                camURL.path,
                "--module-artifact-root",
                source.path,
                "--output",
                package.path,
                "--module-build-evidence-json",
                evidenceURL.path,
            ]
        )

        #expect(result.specPath == camURL.path)
        #expect(result.specSHA256 == identity.projectedPackageSpecSHA256)
        #expect(try packageFiles(at: package) == [
            "SmeltGenerated.swift",
            "manifest.json",
            "module.json",
            "tokenizer.json",
            "weights.bin",
        ])
        #expect(!FileManager.default.fileExists(
            atPath: source.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName).path
        ))
        #expect(result.generatedPackageFiles == [SmeltCAMPackageDescriptor.packageFileName])
        #expect(result.generatedPayloadSHA256?.count == 64)
        #expect(result.sourcePackageFiles == [
            "SmeltGenerated.swift",
            "manifest.json",
            "tokenizer.json",
            "weights.bin",
        ])
        let evidence = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: evidenceURL)) as? [String: Any]
        )
        #expect(evidence["kind"] == nil)
        #expect(evidence["runtime"] == nil)
        #expect(evidence["spec_path"] == nil)
        #expect(evidence["spec_sha256"] == nil)
        #expect(evidence["source_manifest_sha256"] == nil)
        #expect(evidence["evidence_schema"] as? String == "smelt.module.build_evidence.v3")
        #expect(evidence["contract_id"] == nil)
        #expect(evidence["contract_version"] == nil)
        #expect(evidence["projected_package_contract_sha256"] == nil)
        #expect(evidence["module_path"] as? String == camURL.path)
        #expect(evidence["package_projection_id"] as? String == identity.packageProjectionID)
        #expect(evidence["package_projection_version"] as? Int == identity.packageProjectionVersion)
        #expect(evidence["module_semantic_sha256"] as? String == identity.camSemanticSHA256)
        #expect(evidence["export_abi_sha256"] as? String == identity.exportABISHA256)
        #expect(evidence["descriptor_version"] as? Int == identity.descriptorVersion)
        #expect(
            evidence["descriptor_graph_signature_sha256"] as? String
                == identity.descriptorGraphSignatureSHA256
        )
        #expect(
            evidence["projected_package_spec_sha256"] as? String
                == identity.projectedPackageSpecSHA256
        )
        #expect(evidence["source_package_files"] as? [String] == result.sourcePackageFiles)
        #expect(evidence["source_payload_sha256"] as? String == result.sourcePayloadSHA256)
        #expect(
            evidence["generated_package_files"] as? [String]
                == [SmeltCAMPackageDescriptor.packageFileName]
        )
        #expect(evidence["generated_payload_sha256"] as? String == result.generatedPayloadSHA256)
        #expect(evidence["module_descriptor_sha256"] as? String == Self.sha256Hex(descriptorData))
        #expect(evidence["package_payload_sha256"] as? String == result.packagePayloadSHA256)
        #expect(evidence["package_files"] as? [String] == result.packageFiles)
        #expect((evidence["build_elapsed_ms"] as? Int ?? -1) >= 0)
    }

    @Test func packageSpecBuilderAllowsGeneratedTextPipelineInventoryForEmptySpecPolicy() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, pipelines: ["qwen_kernel"])
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        #expect(result.packagePath == package.path)
        #expect(try packageFiles(at: package) == [
            "SmeltGenerated.swift",
            "manifest.json",
            "tokenizer.json",
            "weights.bin",
        ])
    }

    @Test func packageSpecBuilderRejectsDeclaredTextPipelineInventoryDrift() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, pipelines: ["manifest_kernel"])
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(
            textSpec(
                sourcePath: "payloads",
                architectureConfig: textArchitectureConfig(pipelines: ["spec_kernel"])
            ),
            to: specURL
        )
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("pipelines manifest=[1 items] spec=[1 items]"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderIgnoresKnownCompilerByproductsInSourceRoot() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        for path in [
            "SmeltGeneratedKernels.metal",
            "gptq_capture_points.json",
            "model.metalarchive",
            "trace_markers.json",
        ] {
            try Data("compiler byproduct: \(path)\n".utf8)
                .write(to: source.appendingPathComponent(path))
        }
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let result = try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        #expect(result.sourcePackageFiles == [
            "SmeltGenerated.swift",
            "manifest.json",
            "tokenizer.json",
            "weights.bin",
        ])
        #expect(try packageFiles(at: package) == [
            "SmeltGenerated.swift",
            "manifest.json",
            "tokenizer.json",
            "weights.bin",
        ])
    }

    @Test func packageSpecBuilderRejectsGeneratedFileSourceOverlap() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try Data("stale descriptor".utf8).write(
            to: source.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        )
        let spec = withCAMDescriptorArtifact(textSpec(sourcePath: "payloads"))
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let camURL = root.appendingPathComponent("qwen35_text.module.json")
        try Data("cam".utf8).write(to: camURL)
        let identity = SmeltPackageSpecBuilder.CAMBuildIdentity(
            camPath: camURL.path,
            packageProjectionID: "text-to-text-transformer-prefill-decode-affine-u4-g64",
            packageProjectionVersion: 1,
            camSemanticSHA256: Self.sha256Hex(Data("cam-semantic".utf8)),
            exportABISHA256: Self.sha256Hex(Data("export-abi".utf8)),
            descriptorVersion: 1,
            descriptorGraphSignatureSHA256: Self.sha256Hex(Data("graph".utf8)),
            projectedPackageSpecSHA256: Self.sha256Hex(Data("projected-package-spec".utf8))
        )

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                spec: spec,
                camIdentity: identity,
                sourceBaseDirectory: root.path,
                outputDirectory: package.path,
                generatedFiles: [
                    .init(
                        path: SmeltCAMPackageDescriptor.packageFileName,
                        data: Data("descriptor".utf8)
                    ),
                ]
            )
        }

        #expect(String(describing: error).contains("already exists in source root"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsInvalidEvidenceCommandBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let evidenceURL = root.appendingPathComponent("build-evidence.json")

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path,
                evidencePath: evidenceURL.path,
                command: [
                    "/bin/echo",
                    "build",
                    specURL.path,
                    "--output",
                    package.path,
                    "--weights-dir",
                    root.path,
                ]
            )
        }

        #expect(String(describing: error).contains(
            "invalid build evidence command: unsupported option for module package spec JSON: --weights-dir"
        ))
        #expect(!FileManager.default.fileExists(atPath: package.path))
        #expect(!FileManager.default.fileExists(atPath: evidenceURL.path))
    }

    @Test func packageSpecBuildCommandPolicySeparatesBuildFromEvidenceRequirements() {
        let command = [
            "/bin/smelt",
            "build",
            "qwen.cam.json",
            "--output",
            "qwen.smeltpkg",
        ]

        #expect(SmeltPackageSpecBuilder.validateBuildCommandArguments(command) == nil)
        #expect(
            SmeltPackageSpecBuilder.validateBuildCommandArguments(
                command,
                requireEvidenceFlag: true
            ) == "missing required option for module package spec JSON: --module-build-evidence-json"
        )
    }

    @Test func packageSpecBuilderAcceptsMatchingDeclaredChecksums() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, declareChecksums: true)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        #expect(try string(at: package.appendingPathComponent("weights.bin")) == "weights")
    }

    @Test func packageSpecBuilderRejectsInvalidDeclaredArgsBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let bake = SmeltBakeManifest(sealed: [SmeltBakeManifest.args()])
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try Data("""
        {"version":1,"args":[{"flag":"prompt","type":"string","default":"kernels"}],
         "prompt":"Explain {prompt}: {input}"}
        """.utf8).write(to: source.appendingPathComponent(SmeltPackageInterface.fileName))
        try bake.write(packagePath: source.path)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", bakeManifest: bake), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("shadows"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsBakeMarkerDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let expected = SmeltBakeManifest(sealed: [SmeltBakeManifest.args()])
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try Data("""
        {"version":1,"args":[{"flag":"topic","type":"string","default":"kernels"}],
         "prompt":"Explain {topic}: {input}"}
        """.utf8).write(to: source.appendingPathComponent(SmeltPackageInterface.fileName))
        try SmeltBakeManifest(sealed: [
            SmeltBakeManifest.Sealed(
                kind: .args,
                required: [SmeltPackageInterface.fileName],
                perf: ["ghost-perf-accelerator.bin"]
            ),
        ]).write(packagePath: source.path)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", bakeManifest: expected), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("baked.json disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsMissingDeclaredPayloadBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try FileManager.default.removeItem(at: source.appendingPathComponent("tokenizer.json"))
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsUndeclaredSourcePayloadBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try Data("stale".utf8).write(to: source.appendingPathComponent("old-bake.json"))
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderCopiesDeclaredSidecarDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try writeSidecarPayload(to: source.appendingPathComponent("trunk", isDirectory: true))
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", sidecarPath: "trunk"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        #expect(try string(at: package.appendingPathComponent("trunk/manifest.json"))
            == "sidecar-manifest")
    }

    @Test func packageSpecBuilderPreservesInternalSidecarSymlink() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let sidecar = source.appendingPathComponent("trunk", isDirectory: true)
        try writeSidecarPayload(to: sidecar)
        try FileManager.default.createSymbolicLink(
            atPath: sidecar.appendingPathComponent("weights-link.bin").path,
            withDestinationPath: "../weights.bin"
        )
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", sidecarPath: "trunk"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        try SmeltPackageSpecBuilder.build(
            specPath: specURL.path,
            outputDirectory: package.path
        )

        let linkPath = package.appendingPathComponent("trunk/weights-link.bin").path
        let destination = try FileManager.default.destinationOfSymbolicLink(atPath: linkPath)
        #expect(destination == "../weights.bin")
        #expect(try string(at: URL(fileURLWithPath: linkPath)) == "weights")
    }

    @Test func packageSpecBuilderRejectsEscapingSidecarSymlinkBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let sidecar = source.appendingPathComponent("trunk", isDirectory: true)
        try writeSidecarPayload(to: sidecar)
        try Data("outside".utf8).write(to: root.appendingPathComponent("outside.bin"))
        try FileManager.default.createSymbolicLink(
            atPath: sidecar.appendingPathComponent("outside-link.bin").path,
            withDestinationPath: "../../outside.bin"
        )
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", sidecarPath: "trunk"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsManifestValidationDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "fixtures/qwen35",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads", validation: validation), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsStaleChecksummedPayloadBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, declareChecksums: true)
        try Data("stale-weights".utf8).write(to: source.appendingPathComponent("weights.bin"))
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsSymlinkedFilePayloadBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        try FileManager.default.removeItem(at: source.appendingPathComponent("weights.bin"))
        try Data("outside".utf8).write(to: root.appendingPathComponent("outside.bin"))
        try FileManager.default.createSymbolicLink(
            atPath: source.appendingPathComponent("weights.bin").path,
            withDestinationPath: "../outside.bin"
        )
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsTextManifestRootArchitectureBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, architecture: "qwen")
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("must not declare root architecture"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsManifestBlockDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, blocks: .qwen3TTSCompiledTrunkNativeFrontEnd)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsTextInferencePolicyDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(
            to: source,
            inference: .init(
                maxTokens: 64,
                eosTokens: [1],
                chatTemplate: "chatml",
                thinkingPolicy: .disabled
            ),
            decode: .init(sampler: .init(mode: .greedy), maxSteps: 64)
        )
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("text inference policy disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsTextArchitectureConfigDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(
            textSpec(
                sourcePath: "payloads",
                architectureConfig: textArchitectureConfig(hiddenSize: 3)
            ),
            to: specURL
        )
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("text architecture_config policy disagrees"))
        #expect(String(describing: error).contains("hidden_size manifest=2 spec=3"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsTextDecodePolicyDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(
            to: source,
            decode: .init(
                sampler: .init(mode: .sample, temperature: 0.7, topK: 12, topP: 0.95),
                maxSteps: 32
            )
        )
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("text decode policy disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsMissingTextDecodePolicyBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeTextPayloads(to: source, decode: nil)
        let specURL = root.appendingPathComponent("qwen.cam.json")
        try writeSpec(textSpec(sourcePath: "payloads"), to: specURL)
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("requires package-owned decode policy"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsQwen3TTSManifestValidationWhenSpecIsEmptyBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let decode = Qwen3TTSManifest.Decode(
            doSample: false,
            temperature: 1.0,
            topK: 1,
            subtalkerTemperature: 1.0,
            subtalkerTopK: 1
        )
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "payloads",
            eosTokens: [2150],
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeQwen3TTSPayloads(
            to: source,
            specs: specs,
            graph: spec.blocks,
            decode: decode,
            validation: spec.validation
        )
        let specURL = root.appendingPathComponent("qwen3-tts.cam.json")
        try writeSpec(copySpec(spec, validation: .init()), to: specURL)
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("validation policy disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsQwen3TTSDecodePolicyDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let decode = Qwen3TTSManifest.Decode(
            doSample: true,
            temperature: 0.7,
            topK: 12,
            subtalkerTemperature: 0.8,
            subtalkerTopK: 8
        )
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "payloads",
            eosTokens: [2150],
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeQwen3TTSPayloads(
            to: source,
            specs: specs,
            graph: spec.blocks,
            decode: Qwen3TTSManifest.Decode(
                doSample: true,
                temperature: 0.7,
                topK: 13,
                subtalkerTemperature: 0.8,
                subtalkerTopK: 8
            ),
            validation: spec.validation
        )
        let specURL = root.appendingPathComponent("qwen3-tts.cam.json")
        try writeSpec(spec, to: specURL)
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("Qwen3-TTS architecture_config policy disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsQwen3TTSTopLevelDecodeDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let decode = Qwen3TTSManifest.Decode(
            doSample: true,
            temperature: 0.7,
            topK: 12,
            subtalkerTemperature: 0.8,
            subtalkerTopK: 8
        )
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "payloads",
            eosTokens: [2150],
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeQwen3TTSPayloads(
            to: source,
            specs: specs,
            graph: spec.blocks,
            decode: decode,
            validation: spec.validation
        )
        let specURL = root.appendingPathComponent("qwen3-tts.cam.json")
        try writeSpec(
            copySpec(spec, decode: .init(sampler: .init(mode: .greedy))),
            to: specURL
        )
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains("Qwen3-TTS decode policy disagrees"))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsQwen3TTSSidecarManifestDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let decode = Qwen3TTSManifest.Decode(
            doSample: false,
            temperature: 1.0,
            topK: 1,
            subtalkerTemperature: 1.0,
            subtalkerTopK: 1
        )
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "payloads",
            eosTokens: [2150],
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeQwen3TTSPayloads(
            to: source,
            specs: specs,
            graph: spec.blocks,
            decode: decode,
            validation: spec.validation
        )
        try Data("sidecar-manifest".utf8).write(
            to: source.appendingPathComponent("trunk/manifest.json")
        )
        let specURL = root.appendingPathComponent("qwen3-tts.cam.json")
        try writeSpec(spec, to: specURL)
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains(
            "Qwen3-TTS sidecar 'trunk' manifest.json cannot be validated"
        ))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func packageSpecBuilderRejectsQwen3TTSSidecarSharedLinkDriftBeforePackageCreation() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let decode = Qwen3TTSManifest.Decode(
            doSample: false,
            temperature: 1.0,
            topK: 1,
            subtalkerTemperature: 1.0,
            subtalkerTopK: 1
        )
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "payloads",
            eosTokens: [2150],
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )
        let source = root.appendingPathComponent("payloads", isDirectory: true)
        try writeQwen3TTSPayloads(
            to: source,
            specs: specs,
            graph: spec.blocks,
            decode: decode,
            validation: spec.validation
        )
        let weightsLink = source.appendingPathComponent("trunk/weights.bin")
        try FileManager.default.removeItem(at: weightsLink)
        try Data("copied-weights".utf8).write(to: weightsLink)
        let specURL = root.appendingPathComponent("qwen3-tts.cam.json")
        try writeSpec(spec, to: specURL)
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)

        let error = #expect(throws: SmeltPackageSpecBuilderError.self) {
            try SmeltPackageSpecBuilder.build(
                specPath: specURL.path,
                outputDirectory: package.path
            )
        }
        #expect(String(describing: error).contains(
            "Qwen3-TTS sidecar 'trunk' shared weights.bin must be a package-internal symlink"
        ))
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    private func textSpec(
        sourcePath: String,
        sidecarPath: String? = nil,
        bakeManifest: SmeltBakeManifest? = nil,
        validation: SmeltPackageSpec.Validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
    ) -> SmeltPackageSpec {
        textSpec(
            sourcePath: sourcePath,
            sidecarPath: sidecarPath,
            bakeManifest: bakeManifest,
            architectureConfig: textArchitectureConfig(),
            validation: validation
        )
    }

    private func textSpec(
        sourcePath: String,
        sidecarPath: String? = nil,
        bakeManifest: SmeltBakeManifest? = nil,
        architectureConfig: SmeltPackageSpecValue,
        validation: SmeltPackageSpec.Validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
    ) -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.tokenFeedbackText
        var files = [
            "manifest.json",
            "SmeltGenerated.swift",
            "tokenizer.json",
            "weights.bin",
        ]
        let sidecars: [SmeltPackageSpec.Sidecar]
        if let sidecarPath {
            files.append(sidecarPath)
            sidecars = [
                .init(
                    id: "trunk-sidecar",
                    path: sidecarPath,
                    kind: SmeltPackageSidecarKind.compiledTrunk
                ),
            ]
        } else {
            sidecars = []
        }
        if let bakeManifest {
            files.append(SmeltBakeManifest.fileName)
            for sealed in bakeManifest.sealed {
                files.append(contentsOf: sealed.required)
                files.append(contentsOf: sealed.perf)
            }
        }

        return SmeltPackageSpec(
            packageName: "qwen.cam",
            modelName: "qwen",
            sources: [
                .init(id: "package", kind: .localDirectory, path: sourcePath)
            ],
            blocks: graph,
            loop: .tokenFeedbackText,
            runtime: .forGraph(
                architecture: "text-generation",
                commands: SmeltPackageSpec.RuntimeDescriptor.Command.allCases,
                graph: graph
            ),
            architectureConfig: architectureConfig,
            tensors: [
                .init(
                    source: "package",
                    name: "layers.0.self_attn.q_proj.weight",
                    canonicalName: "layers_0_self_attn_q_proj_weight",
                    block: "trunk",
                    sourceDType: .bf16,
                    storedDType: .u4,
                    shape: [2, 2]
                )
            ],
            quantization: .init(format: .u4, groupSize: 128),
            sidecars: sidecars,
            artifacts: [
                .init(id: "generated", path: "SmeltGenerated.swift", role: "generated-swift"),
                .init(id: "weights", path: "weights.bin", role: "weights"),
            ],
            outputFiles: .init(files: files, bakeManifest: bakeManifest),
            tokenizer: .init(format: "tokenizer-json", files: ["tokenizer.json"]),
            inference: .init(
                maxTokens: 32,
                eosTokens: [1],
                chatTemplate: "chatml",
                thinkingPolicy: .disabled
            ),
            decode: .init(sampler: .init(mode: .greedy), maxSteps: 32),
            validation: validation
        )
    }

    private func textArchitectureConfig(
        hiddenSize: Int = 2,
        pipelines: [String] = []
    ) -> SmeltPackageSpecValue {
        .object([
            "hidden_size": .int(hiddenSize),
            "num_layers": .int(1),
            "vocab_size": .int(8),
            "static_seq_capacity": .int(32),
            "rope_dim": .int(2),
            "num_delta_layers": .int(0),
            "num_attn_layers": .int(1),
            "ffn_dim": .int(4),
            "pipelines": .array(pipelines.map { .string($0) }),
            "weight_total_bytes": .int(7),
        ])
    }

    private func withCAMDescriptorArtifact(_ spec: SmeltPackageSpec) -> SmeltPackageSpec {
        var files = spec.outputFiles.files
        if !files.contains(SmeltCAMPackageDescriptor.packageFileName) {
            files.append(SmeltCAMPackageDescriptor.packageFileName)
        }
        return SmeltPackageSpec(
            version: spec.version,
            packageName: spec.packageName,
            modelName: spec.modelName,
            sources: spec.sources,
            blocks: spec.blocks,
            loop: spec.loop,
            runtime: spec.runtime,
            architectureConfig: spec.architectureConfig,
            tensors: spec.tensors,
            quantization: spec.quantization,
            sidecars: spec.sidecars,
            artifacts: spec.artifacts + [
                .init(
                    id: "cam-descriptor",
                    path: SmeltCAMPackageDescriptor.packageFileName,
                    role: "cam-descriptor"
                ),
            ],
            outputFiles: .init(
                manifest: spec.outputFiles.manifest,
                files: files.sorted(),
                bakeManifest: spec.outputFiles.bakeManifest
            ),
            tokenizer: spec.tokenizer,
            inference: spec.inference,
            decode: spec.decode,
            validation: spec.validation
        )
    }

    private func copySpec(
        _ spec: SmeltPackageSpec,
        decode: SmeltPackageSpec.DecodePolicy
    ) -> SmeltPackageSpec {
        copySpec(spec, decode: decode, validation: nil)
    }

    private func copySpec(
        _ spec: SmeltPackageSpec,
        validation: SmeltPackageSpec.Validation
    ) -> SmeltPackageSpec {
        copySpec(spec, decode: spec.decode, validation: validation)
    }

    private func copySpec(
        _ spec: SmeltPackageSpec,
        decode: SmeltPackageSpec.DecodePolicy?,
        validation: SmeltPackageSpec.Validation? = nil
    ) -> SmeltPackageSpec {
        SmeltPackageSpec(
            version: spec.version,
            packageName: spec.packageName,
            modelName: spec.modelName,
            sources: spec.sources,
            blocks: spec.blocks,
            loop: spec.loop,
            runtime: spec.runtime,
            architectureConfig: spec.architectureConfig,
            tensors: spec.tensors,
            quantization: spec.quantization,
            sidecars: spec.sidecars,
            artifacts: spec.artifacts,
            outputFiles: spec.outputFiles,
            tokenizer: spec.tokenizer,
            inference: spec.inference,
            decode: decode,
            validation: validation ?? spec.validation
        )
    }

    private struct StubManifest: Codable {
        let kind: String?
        let architecture: String?
        let modelName: String
        let blocks: SmeltBlockGraph
        let loop: SmeltLoopSchedule
        let inference: SmeltInferenceManifest?
        let decode: SmeltPackageSpec.DecodePolicy?
        let validation: SmeltPackageSpec.Validation?
        let checksums: SmeltManifestChecksums?
    }

    private func writeTextPayloads(
        to directory: URL,
        architecture: String? = nil,
        blocks: SmeltBlockGraph = .tokenFeedbackText,
        loop: SmeltLoopSchedule = .tokenFeedbackText,
        inference: SmeltInferenceManifest? = SmeltInferenceManifest(
            maxTokens: 32,
            eosTokens: [1],
            chatTemplate: "chatml",
            thinkingPolicy: .disabled
        ),
        decode: SmeltPackageSpec.DecodePolicy? = SmeltPackageSpec.DecodePolicy(
            sampler: .init(mode: .greedy),
            maxSteps: 32
        ),
        validation: SmeltPackageSpec.Validation? = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        ),
        pipelines: [String] = [],
        declareChecksums: Bool = false
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let generated = Data("generated".utf8)
        let tokenizer = Data("tokenizer".utf8)
        let weights = Data("weights".utf8)
        try generated.write(to: directory.appendingPathComponent("SmeltGenerated.swift"))
        try tokenizer.write(to: directory.appendingPathComponent("tokenizer.json"))
        try weights.write(to: directory.appendingPathComponent("weights.bin"))
        let checksums = declareChecksums
            ? SmeltManifestChecksums(
                weightsBin: Self.sha256Hex(weights),
                metallib: Self.sha256Hex(Data("unplanned-metallib".utf8)),
                generatedSwift: Self.sha256Hex(generated),
                dispatchesBin: Self.sha256Hex(Data("unplanned-dispatches".utf8)),
                tokenizerJSON: Self.sha256Hex(tokenizer)
            )
            : SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: "",
                tokenizerJSON: nil
            )
        let manifest = SmeltManifest(
            architecture: architecture,
            blocks: blocks,
            loop: loop,
            modelName: "qwen",
            config: SmeltManifestConfig(
                hiddenSize: 2,
                numLayers: 1,
                vocabSize: 8,
                staticSeqCapacity: 32,
                ropeDim: 2,
                numDeltaLayers: 0,
                numAttnLayers: 1,
                ffnDim: 4
            ),
            context: nil,
            checksums: checksums,
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: UInt64(weights.count), entries: []),
            buffers: SmeltBufferTable(slots: []),
            pipelines: pipelines,
            slotLayout: emptySlotLayout(),
            inference: inference,
            decode: decode,
            validation: validation
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: directory.appendingPathComponent("manifest.json")
        )
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

    private func qwen3TTSRunnableSpecs(
        textEmbeddingDType: Qwen3TTSPackageBuilder.WeightDType
    ) -> [Qwen3TTSPackageBuilder.WeightSpec] {
        let hidden = 64
        let headDim = 32
        let qDim = 64
        let kvDim = 32
        let inter = 128
        let vocab = 128

        var specs: [Qwen3TTSPackageBuilder.WeightSpec] = [
            .init(name: "talker.model.text_embedding.weight", shape: [vocab, hidden], dtype: textEmbeddingDType),
            .init(name: "talker.text_projection.linear_fc1.weight", shape: [hidden, hidden]),
            .init(name: "talker.text_projection.linear_fc1.bias", shape: [hidden]),
            .init(name: "talker.text_projection.linear_fc2.weight", shape: [hidden, hidden]),
            .init(name: "talker.text_projection.linear_fc2.bias", shape: [hidden]),
            .init(name: "talker.model.codec_embedding.weight", shape: [vocab, hidden]),
            .init(name: "talker.codec_head.weight", shape: [vocab, hidden], dtype: .bf16),
            .init(name: "talker.code_predictor.lm_head.0.weight", shape: [vocab, hidden], dtype: .bf16),
            .init(name: "talker.code_predictor.model.codec_embedding.0.weight", shape: [vocab, hidden], dtype: .bf16),
            .init(name: "talker.code_predictor.small_to_mtp_projection.weight", shape: [hidden, hidden], dtype: .bf16),
            .init(name: "talker.code_predictor.small_to_mtp_projection.bias", shape: [hidden]),
            .init(name: "decoder.pre_conv.conv.weight", shape: [2, 2]),
        ]

        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter
        )
        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.code_predictor.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter
        )
        return specs
    }

    private func appendQwen3TTSTrunkLayerSpecs(
        to specs: inout [Qwen3TTSPackageBuilder.WeightSpec],
        prefix: String,
        hidden: Int,
        headDim: Int,
        qDim: Int,
        kvDim: Int,
        inter: Int
    ) {
        let layer = "\(prefix)layers.0"
        specs.append(contentsOf: [
            .init(name: "\(layer).self_attn.q_proj.weight", shape: [qDim, hidden], dtype: .bf16),
            .init(name: "\(layer).self_attn.k_proj.weight", shape: [kvDim, hidden], dtype: .bf16),
            .init(name: "\(layer).self_attn.v_proj.weight", shape: [kvDim, hidden], dtype: .bf16),
            .init(name: "\(layer).self_attn.o_proj.weight", shape: [hidden, qDim], dtype: .bf16),
            .init(name: "\(layer).mlp.gate_proj.weight", shape: [inter, hidden], dtype: .bf16),
            .init(name: "\(layer).mlp.up_proj.weight", shape: [inter, hidden], dtype: .bf16),
            .init(name: "\(layer).mlp.down_proj.weight", shape: [hidden, inter], dtype: .bf16),
            .init(name: "\(layer).input_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).post_attention_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).self_attn.q_norm.weight", shape: [headDim]),
            .init(name: "\(layer).self_attn.k_norm.weight", shape: [headDim]),
            .init(name: "\(prefix)norm.weight", shape: [hidden]),
        ])
    }

    private func writeQwen3TTSPayloads(
        to directory: URL,
        specs: [Qwen3TTSPackageBuilder.WeightSpec],
        graph: SmeltBlockGraph,
        decode: Qwen3TTSManifest.Decode,
        validation: SmeltPackageSpec.Validation,
        pageSize: Int = 16
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let orderedSpecs = specs.sorted { $0.name < $1.name }
        let layout = Qwen3TTSPackageBuilder.planLayout(orderedSpecs, pageSize: pageSize)
        let manifest = Qwen3TTSManifest(
            version: 1,
            blocks: graph,
            loop: .qwen3TTS,
            modelName: "qwen3-tts-12hz",
            pageSize: pageSize,
            pipelines: ["qwen_kernel"],
            eosTokens: [2150],
            totalBytes: layout.totalBytes,
            weights: layout.entries,
            tokenizerFiles: Qwen3TTSManifest.requiredTokenizerFiles,
            decode: decode,
            validation: validation
        )
        try manifest.encoded().write(
            to: directory.appendingPathComponent("manifest.json")
        )
        try Data(repeating: 0, count: Int(layout.totalBytes)).write(
            to: directory.appendingPathComponent("weights.bin")
        )
        try Data("metallib".utf8).write(to: directory.appendingPathComponent("model.metallib"))
        for tokenizerFile in Qwen3TTSManifest.requiredTokenizerFiles {
            try Data("tokenizer".utf8).write(to: directory.appendingPathComponent(tokenizerFile))
        }
        try writeQwen3TTSSidecarPayload(
            to: directory.appendingPathComponent("trunk", isDirectory: true),
            spec: .talker,
            totalBytes: layout.totalBytes
        )
        try writeQwen3TTSSidecarPayload(
            to: directory.appendingPathComponent("trunk-mtp", isDirectory: true),
            spec: .mtp,
            totalBytes: layout.totalBytes
        )
    }

    private func writeQwen3TTSSidecarPayload(
        to directory: URL,
        spec: Qwen3TTSTrunkSidecar.TrunkSidecarSpec,
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
            modelName: spec.modelName,
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
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
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
            inference: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))
    }

    private func writeSidecarPayload(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("sidecar-manifest".utf8).write(
            to: directory.appendingPathComponent("manifest.json")
        )
        let nested = directory.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data("sidecar-generated".utf8).write(
            to: nested.appendingPathComponent("SmeltGenerated.swift")
        )
    }

    private func writeSpec(_ spec: SmeltPackageSpec, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(spec).write(to: url)
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
            "smelt-package-spec-builder-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    private func string(at url: URL) throws -> String {
        String(decoding: try Data(contentsOf: url), as: UTF8.self)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
