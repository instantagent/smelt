import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

@Suite struct SmeltPackageResolvedPlanTests {

    @Test func resolvesDeterministicTextGenerationPlanWithoutWriting() throws {
        let temp = try emptyTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let ir = FixtureModelIRs.qwen35_2b_affine_metalprefill
        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)

        let first = try SmeltPackageResolvedPlan.resolve(spec)
        let second = try SmeltPackageResolvedPlan.resolve(spec)

        #expect(first == second)
        #expect(first.signature == second.signature)
        #expect(first.runtime.architecture == "text-generation")
        #expect(first.runtime.routes.map(\.signature) == [
            "tokenizer:native:none",
            "trunk:compiled:compiled-inline",
            "text-head:native:none",
        ])
        #expect(first.runtime.commands.map(\.rawValue) == [
            "bench", "linger-worker", "load", "prepare", "run", "serve", "trace",
        ])
        #expect(first.policy.mode == .textGeneration)
        #expect(first.policy.inference?.chatTemplate == "chatml")
        #expect(first.validationPerformanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
        #expect(first.validationPerformanceProfile?.gate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
        #expect(first.validationPerformanceProfile?.command == .run)
        #expect(first.validationPerformanceProfile?.requiredTraceLabels
            == SmeltPackagePerformanceTraceLabel.textDecodePrefillStartupRequired.sorted())
        #expect(first.validationPerformanceProfile?.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS(forModelName: ir.modelName),
                unit: SmeltPackagePerformanceUnit.milliseconds
            ),
        ])
        #expect(first.packageFiles.contains(.init(
            path: "prefill_dispatches.bin",
            roles: ["artifact:prefill-dispatches:dispatch-table", "declared-output"]
        )))
        #expect(first.packageFiles.contains(.init(
            path: "tokenizer.bin",
            roles: ["artifact:compiled-tokenizer:tokenizer-cache", "declared-output"]
        )))
        #expect(try directoryContents(temp).isEmpty)
    }

    @Test func quantizationMutationChangesExpectedPlanFieldOnly() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
        let base = try SmeltPackageResolvedPlan.resolve(spec)
        let baseQuantization = try #require(spec.quantization)
        let baseGroup = try #require(baseQuantization.groupSize)

        let mutatedSpec = copy(
            spec,
            quantization: .init(format: baseQuantization.format, groupSize: baseGroup + 1)
        )
        let mutated = try SmeltPackageResolvedPlan.resolve(mutatedSpec)

        #expect(mutated.quantization?.groupSize == baseGroup + 1)
        #expect(mutated.signature != base.signature)
        #expect(mutated.packageFiles == base.packageFiles)
        #expect(mutated.runtime == base.runtime)
        #expect(mutated.tensors == base.tensors)
    }

    @Test func quantizationCalibrationChangesPlanSignatureAndInventory() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
        let base = try SmeltPackageResolvedPlan.resolve(spec)
        let baseQuantization = try #require(spec.quantization)
        let sourceID = try #require(spec.sources.first?.id)
        let calibration = Self.quantizationCalibration(source: sourceID)

        let mutatedSpec = copy(
            spec,
            quantization: .init(
                format: baseQuantization.format,
                groupSize: baseQuantization.groupSize,
                calibration: calibration
            )
        )
        let mutated = try SmeltPackageResolvedPlan.resolve(mutatedSpec)

        #expect(mutated.quantization?.calibration?.corpus?.path == "calibration/prompts.jsonl")
        #expect(mutated.signature != base.signature)
        #expect(mutated.signature.lines.contains(
            "quantization-calibration-corpus:"
                + "\(sourceID):calibration/prompts.jsonl:chat-template:128:2048"
        ))
        #expect(mutated.signature.lines.contains(
            "quantization-calibration-equality:"
                + "packed-regions:weights.bin:calibration/reference.weights.bin:sha256"
        ))
        #expect(mutated.packageFiles.contains(.init(
            path: "calibration/gptq_capture_points.json",
            roles: ["quant-calibration:capture:prefill-captures:activation-capture"]
        )))
        #expect(mutated.runtime == base.runtime)
        #expect(mutated.tensors == base.tensors)
    }

    @Test func resolveRejectsInvalidSpecBeforeWriting() throws {
        let temp = try emptyTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let spec = try SmeltPackageSpecLowering.textGeneration(from: FixtureModelIRs.qwen35_2B)
        let badSource = SmeltPackageSpec.Source(
            id: spec.sources[0].id,
            kind: .localDirectory,
            path: "../escape"
        )
        let invalid = copy(spec, sources: [badSource])

        #expect(throws: SmeltPackageSpecError.self) {
            try SmeltPackageResolvedPlan.resolve(invalid)
        }
        #expect(try directoryContents(temp).isEmpty)
    }

    private func copy(
        _ spec: SmeltPackageSpec,
        sources: [SmeltPackageSpec.Source]? = nil,
        runtime: SmeltPackageSpec.RuntimeDescriptor? = nil,
        quantization: SmeltPackageSpec.QuantizationPlan? = nil
    ) -> SmeltPackageSpec {
        SmeltPackageSpec(
            version: spec.version,
            packageName: spec.packageName,
            modelName: spec.modelName,
            sources: sources ?? spec.sources,
            blocks: spec.blocks,
            loop: spec.loop,
            runtime: runtime ?? spec.runtime,
            architectureConfig: spec.architectureConfig,
            tensors: spec.tensors,
            quantization: quantization ?? spec.quantization,
            sidecars: spec.sidecars,
            artifacts: spec.artifacts,
            outputFiles: spec.outputFiles,
            tokenizer: spec.tokenizer,
            inference: spec.inference,
            decode: spec.decode,
            validation: spec.validation
        )
    }

    private static func quantizationCalibration(
        source: String
    ) -> SmeltPackageSpec.QuantizationPlan.Calibration {
        .init(
            corpus: .init(
                source: source,
                path: "calibration/prompts.jsonl",
                renderPolicy: "chat-template",
                maxSamples: 128,
                maxTokens: 2048
            ),
            captureArtifacts: [
                .init(
                    id: "prefill-captures",
                    path: "calibration/gptq_capture_points.json",
                    role: "activation-capture"
                ),
            ],
            sideInputs: [
                .init(id: "imatrix", path: "calibration/imatrix.dat", role: "imatrix"),
            ],
            stagedPackages: [
                .init(
                    id: "bf16-prefill",
                    path: "calibration/bf16-prefill.smeltpkg",
                    role: "activation-source"
                ),
            ],
            resourceBounds: .init(
                maxSamples: 128,
                maxTokens: 2048,
                maxLayersPerPass: 4,
                maxBytes: 1_000_000_000
            ),
            equalityGates: [
                .init(
                    id: "packed-regions",
                    candidate: "weights.bin",
                    reference: "calibration/reference.weights.bin",
                    metric: "sha256"
                ),
            ]
        )
    }

    private func emptyTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-cam-plan-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    private func directoryContents(_ url: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: url.path).sorted()
    }
}
