import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

@Suite struct SmeltPackageAssemblerTests {

    @Test func assembleRejectsMissingPayloadBeforeCreatingPackage() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let payloads = dataPayloads(for: plan).filter { $0.path != "tokenizer.bin" }

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: payloads
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func assembleRejectsPayloadOutsideResolvedPlanBeforeCreatingPackage() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        var payloads = dataPayloads(for: plan)
        payloads.append(.init(path: "bonus.bin", body: .data(Data("extra".utf8))))

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: payloads
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func assembleRejectsUnsafePayloadPathBeforeCreatingPackage() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        var payloads = dataPayloads(for: plan)
        payloads[0] = .init(path: "../escape.bin", body: .data(Data("escape".utf8)))

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: payloads
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func assembleRejectsMissingCopySourceBeforeCreatingPackage() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        var payloads = dataPayloads(for: plan)
        payloads[0] = .init(
            path: payloads[0].path,
            body: .copyFile(root.appendingPathComponent("missing.bin").path)
        )

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: payloads
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
    }

    @Test func assembleRollsBackNewPackageWhenCommitFails() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try conflictingFilesystemPlan()
        let package = root.appendingPathComponent("broken.smeltpkg", isDirectory: true)

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: dataPayloads(for: plan)
            )
        }
        #expect(!FileManager.default.fileExists(atPath: package.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).filter {
            $0.hasPrefix(".broken.smeltpkg.tmp-")
        }.isEmpty)
    }

    @Test func assembleWritesExactlyResolvedInventory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        let prepared = try SmeltPackageAssembler.assemble(
            plan: plan,
            packagePath: package.path,
            payloads: dataPayloads(for: plan)
        )

        #expect(prepared.plan == plan)
        let written = try packageFiles(at: package)
        #expect(written == plan.packageFiles.map(\.path).sorted())
        for file in written {
            let data = try Data(contentsOf: package.appendingPathComponent(file))
            #expect(String(decoding: data, as: UTF8.self) == "payload:\(file)")
        }
    }

    @Test func assembleRejectsStaleExistingPackageInventoryBeforeWriting() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try Data("old-manifest".utf8).write(
            to: package.appendingPathComponent("manifest.json")
        )
        try Data("stale".utf8).write(
            to: package.appendingPathComponent("stale.bin")
        )

        #expect(throws: SmeltPackageAssemblerError.self) {
            try SmeltPackageAssembler.assemble(
                plan: plan,
                packagePath: package.path,
                payloads: dataPayloads(for: plan)
            )
        }
        let manifest = try Data(contentsOf: package.appendingPathComponent("manifest.json"))
        #expect(String(decoding: manifest, as: UTF8.self) == "old-manifest")
    }

    @Test func assembleReplacesExistingSymlinkPayloadWithRegularFile() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try resolvedTextPlan()
        let package = root.appendingPathComponent("qwen.smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        let casTarget = root.appendingPathComponent("cas-model.metallib")
        try Data("old-model".utf8).write(to: casTarget)
        let modelPath = package.appendingPathComponent("model.metallib")
        try FileManager.default.createSymbolicLink(
            atPath: modelPath.path,
            withDestinationPath: casTarget.path
        )
        let replacement = root.appendingPathComponent("replacement-model.metallib")
        try Data("new-model".utf8).write(to: replacement)

        var payloads = dataPayloads(for: plan)
        let modelIndex = try #require(payloads.firstIndex { $0.path == "model.metallib" })
        payloads[modelIndex] = .init(
            path: "model.metallib",
            body: .copyFile(replacement.path)
        )

        try SmeltPackageAssembler.assemble(
            plan: plan,
            packagePath: package.path,
            payloads: payloads
        )

        let model = try Data(contentsOf: modelPath)
        #expect(String(decoding: model, as: UTF8.self) == "new-model")
        #expect((try? FileManager.default.destinationOfSymbolicLink(
            atPath: modelPath.path
        )) == nil)
        let originalTarget = try Data(contentsOf: casTarget)
        #expect(String(decoding: originalTarget, as: UTF8.self) == "old-model")
    }

    @Test func assembleCopiesSourceFilesAndSidecarDirectories() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let plan = try SmeltPackageResolvedPlan.resolve(qwenTTSSidecarSpec())
        let source = root.appendingPathComponent("tokenizer-source.json")
        try Data("copied-tokenizer".utf8).write(to: source)
        let sidecar = root.appendingPathComponent("trunk-source", isDirectory: true)
        try FileManager.default.createDirectory(at: sidecar, withIntermediateDirectories: true)
        try Data("sidecar-manifest".utf8).write(to: sidecar.appendingPathComponent("manifest.json"))
        let mtpSidecar = root.appendingPathComponent("trunk-mtp-source", isDirectory: true)
        try FileManager.default.createDirectory(at: mtpSidecar, withIntermediateDirectories: true)
        try Data("mtp-sidecar-manifest".utf8).write(
            to: mtpSidecar.appendingPathComponent("manifest.json")
        )
        let package = root.appendingPathComponent("qwen3-tts.smeltpkg", isDirectory: true)
        let payloads = plan.packageFiles.map { file in
            if file.path == "tokenizer.json" {
                return SmeltPackageAssembler.FilePayload(
                    path: file.path,
                    body: .copyFile(source.path)
                )
            }
            if file.path == "trunk" {
                return SmeltPackageAssembler.FilePayload(
                    path: file.path,
                    body: .copyDirectory(sidecar.path)
                )
            }
            if file.path == "trunk-mtp" {
                return SmeltPackageAssembler.FilePayload(
                    path: file.path,
                    body: .copyDirectory(mtpSidecar.path)
                )
            }
            return SmeltPackageAssembler.FilePayload(
                path: file.path,
                body: .data(Data("payload:\(file.path)".utf8))
            )
        }

        try SmeltPackageAssembler.assemble(
            plan: plan,
            packagePath: package.path,
            payloads: payloads
        )

        var isDirectory = ObjCBool(false)
        #expect(FileManager.default.fileExists(
            atPath: package.appendingPathComponent("trunk").path,
            isDirectory: &isDirectory
        ))
        #expect(isDirectory.boolValue)
        let sidecarManifest = try Data(
            contentsOf: package.appendingPathComponent("trunk/manifest.json")
        )
        #expect(String(decoding: sidecarManifest, as: UTF8.self) == "sidecar-manifest")
        let mtpSidecarManifest = try Data(
            contentsOf: package.appendingPathComponent("trunk-mtp/manifest.json")
        )
        #expect(String(decoding: mtpSidecarManifest, as: UTF8.self) == "mtp-sidecar-manifest")
        let tokenizer = try Data(
            contentsOf: package.appendingPathComponent("tokenizer.json")
        )
        #expect(String(decoding: tokenizer, as: UTF8.self) == "copied-tokenizer")
    }

    private func resolvedTextPlan() throws -> SmeltPackageResolvedPlan {
        let spec = try SmeltPackageSpecLowering.textGeneration(from: FixtureModelIRs.qwen35_2B)
        return try SmeltPackageResolvedPlan.resolve(spec)
    }

    private func dataPayloads(
        for plan: SmeltPackageResolvedPlan
    ) -> [SmeltPackageAssembler.FilePayload] {
        plan.packageFiles.map {
            .init(path: $0.path, body: .data(Data("payload:\($0.path)".utf8)))
        }
    }

    private func conflictingFilesystemPlan() throws -> SmeltPackageResolvedPlan {
        let base = try resolvedTextPlan()
        return SmeltPackageResolvedPlan(
            version: base.version,
            packageName: "broken.cam",
            modelName: base.modelName,
            sources: base.sources,
            runtime: base.runtime,
            architectureConfigSignature: base.architectureConfigSignature,
            tensors: base.tensors,
            quantization: base.quantization,
            packageFiles: [
                .init(path: "manifest.json", roles: ["declared-output", "manifest"]),
                .init(path: "manifest.json/child", roles: ["declared-output"]),
                .init(
                    path: "tokenizer.json",
                    roles: ["declared-output", "tokenizer:tokenizer-json"]
                ),
                .init(path: "weights.bin", roles: ["artifact:weights:weights", "declared-output"]),
            ],
            policy: base.policy,
            validationParityFixture: base.validationParityFixture,
            validationPerformanceGate: base.validationPerformanceGate,
            validationPerformanceProfile: base.validationPerformanceProfile,
            validationStructureProfile: base.validationStructureProfile
        )
    }

    private func packageFiles(at package: URL) throws -> [String] {
        let fm = FileManager.default
        let rootPath = package.path
        guard let enumerator = fm.enumerator(atPath: rootPath) else { return [] }
        return enumerator.compactMap { item -> String? in
            guard let path = item as? String else { return nil }
            var isDirectory = ObjCBool(false)
            fm.fileExists(
                atPath: URL(fileURLWithPath: rootPath).appendingPathComponent(path).path,
                isDirectory: &isDirectory
            )
            return isDirectory.boolValue ? nil : path
        }.sorted()
    }

    private func qwenTTSSidecarSpec() -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.qwen3TTSCompiledTrunkNativeFrontEnd
        let source = SmeltPackageSpec.Source(
            id: "weights",
            kind: .huggingFace,
            repo: "Qwen/Qwen3-TTS"
        )
        return SmeltPackageSpec(
            packageName: "qwen3-tts.cam",
            modelName: "qwen3-tts",
            sources: [source],
            blocks: graph,
            loop: .qwen3TTS,
            runtime: .forGraph(
                architecture: SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue,
                commands: SmeltPackageSpec.RuntimeDescriptor.Command.allCases,
                graph: graph
            ),
            architectureConfig: .object(["page_size": .int(16_384)]),
            tensors: [
                .init(
                    source: source.id,
                    name: "talker.model.layers.0.self_attn.q_proj.weight",
                    canonicalName: "talker.model.layers.0.self_attn.q_proj.weight",
                    block: "talker",
                    sourceDType: .bf16,
                    storedDType: .bf16,
                    shape: [2, 2]
                )
            ],
            quantization: .init(format: .u4, groupSize: 64),
            sidecars: SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks
                .map(\.sidecar),
            artifacts: [
                .init(id: "weights", path: "weights.bin", role: "weights"),
                .init(id: "metallib", path: "model.metallib", role: "metallib"),
            ],
            outputFiles: .init(files: [
                "manifest.json",
                "weights.bin",
                "model.metallib",
                "tokenizer.json",
                "trunk",
                "trunk-mtp",
            ]),
            tokenizer: .init(format: "byte-bpe", files: ["tokenizer.json"]),
            inference: .init(maxTokens: 2048, eosTokens: [2150]),
            decode: .init(sampler: .init(mode: .sample, temperature: 0.8), maxSteps: 2048)
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-cam-assembler-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }
}
