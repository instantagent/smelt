import CryptoKit
import Foundation
import XCTest
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

final class SmeltTraceReportTests: XCTestCase {
    private func makePackage(_ name: String) throws -> String {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString).smeltpkg", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        return root.path
    }

    private func writeFile(
        _ packagePath: String,
        _ relativePath: String,
        data: Data = Data([0])
    ) throws {
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }

    private func writeDispatchTable(
        _ packagePath: String,
        _ fileName: String,
        pipeline: UInt16 = 0
    ) throws {
        // Build the two records directly into a zero-filled buffer and set only the
        // meaningful fields in place. SmeltDispatchRecord has interior alignment
        // padding (bytes 1, 7, 10-11) that the memberwise `.empty()`/`.swap()`
        // initializers leave uninitialized; copying such a value into the buffer
        // would re-import that garbage padding and make dispatches.bin's SHA
        // nondeterministic run-to-run (and between the two records in one process).
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        var data = Data(count: 2 * stride)
        data.withUnsafeMutableBytes { raw in
            let table = raw.bindMemory(to: SmeltDispatchRecord.self)
            table[0].opKind = SmeltDispatchRecord.opDispatch
            table[0].pipeline = pipeline
            table[0].dispatchStyle = SmeltDispatchRecord.styleThreadgroups
            table[0].gridW = 1
            table[0].gridH = 1
            table[0].gridD = 1
            table[0].tgW = 1
            table[0].tgH = 1
            table[0].tgD = 1
            table[1].opKind = SmeltDispatchRecord.opSwap
        }
        try writeFile(packagePath, fileName, data: data)
    }

    private func writeJSON<T: Encodable>(
        _ value: T,
        to packagePath: String,
        fileName: String = "manifest.json"
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try writeFile(packagePath, fileName, data: try encoder.encode(value))
    }

    private func canonicalSHA256<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func fullTextRuntimeEvents(report: SmeltTraceReport) -> [SmeltTraceEvent] {
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        return [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "setup:prefill",
                block: "trunk",
                route: routes["trunk"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "setup:prefill",
                block: "text-head",
                route: routes["text-head"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 4,
                phase: "per-step:decode",
                block: "trunk",
                route: routes["trunk"],
                step: 0,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 5,
                phase: "per-step:decode",
                block: "text-head",
                route: routes["text-head"],
                step: 0,
                witness: "ok"
            ),
        ]
    }

    private func copyContract(
        _ contract: SmeltTraceContract,
        graph: SmeltTraceGraph? = nil,
        loop: SmeltTraceLoop? = nil,
        blocks: [SmeltTraceBlockRoute]? = nil,
        dispatchTables: [SmeltTraceDispatchTableContract]? = nil,
        sidecars: [SmeltTraceSidecarContract]? = nil,
        artifacts: [SmeltTraceArtifactContract]? = nil,
        events: [SmeltTraceEvent]? = nil,
        issues: [SmeltTraceIssue]? = nil
    ) -> SmeltTraceContract {
        SmeltTraceContract(
            package: contract.package,
            graph: graph ?? contract.graph,
            loop: loop ?? contract.loop,
            blocks: blocks ?? contract.blocks,
            dispatchTables: dispatchTables ?? contract.dispatchTables,
            sidecars: sidecars ?? contract.sidecars,
            artifacts: artifacts ?? contract.artifacts,
            dtypeSummary: contract.dtypeSummary,
            events: events ?? contract.events,
            issues: issues ?? contract.issues
        )
    }

    private func assertTraceSuiteValidationError(
        _ suite: SmeltTraceSuiteSpec,
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let suiteRoot = try makePackage("smelt-trace-invalid-suite")
        let suitePath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("suite.json").path
        try writeJSON(suite, to: suiteRoot, fileName: "suite.json")

        XCTAssertThrowsError(
            try SmeltTrace.verifySuite(path: suitePath),
            file: file,
            line: line
        ) { error in
            XCTAssertTrue(
                (error as NSError).localizedDescription.contains(expected),
                "expected '\(expected)' in \(error)",
                file: file,
                line: line
            )
        }
    }

    private func minimalLLMManifest(
        kind: String? = nil,
        headlessTrunkABI: Bool? = nil,
        blocks: SmeltBlockGraph? = .tokenFeedbackText,
        loop: SmeltLoopSchedule? = .tokenFeedbackText,
        pipelines: [String] = ["alpha"],
        prefill: Bool = false,
        inference: SmeltInferenceManifest? = SmeltInferenceManifest(
            maxTokens: 8,
            eosTokens: [1],
            chatTemplate: "chatml",
            thinkingPolicy: .disabled
        ),
        decode: SmeltPackageSpec.DecodePolicy? = SmeltPackageSpec.DecodePolicy(
            sampler: .init(mode: .greedy),
            maxSteps: 8
        ),
        validation: SmeltPackageSpec.Validation? = SmeltPackagePerformanceProfiles.validation(
            parityFixture: "test/model",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
    ) -> SmeltManifest {
        SmeltManifest(
            kind: kind,
            headlessTrunkABI: headlessTrunkABI,
            blocks: blocks,
            loop: loop,
            modelName: "test/model",
            config: SmeltManifestConfig(
                hiddenSize: 1,
                numLayers: 1,
                vocabSize: 1,
                staticSeqCapacity: 1,
                ropeDim: 1,
                numDeltaLayers: 0,
                numAttnLayers: 0,
                ffnDim: 1
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: ""
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 4, entries: [
                SmeltWeightEntry(
                    name: "weight",
                    offset: 0,
                    sizeBytes: 4,
                    shape: [1],
                    dtype: .fp32
                )
            ]),
            buffers: SmeltBufferTable(slots: []),
            pipelines: pipelines,
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 0,
                recStateBaseSlot: 0,
                keyCacheBaseSlot: 0,
                valCacheBaseSlot: 0,
                ropeCosSlot: 0,
                ropeSinSlot: 0,
                tokenIdSlot: 0,
                positionSlot: 0,
                weightsSlot: 0
            ),
            prefill: prefill ? SmeltPrefillManifest(
                engine: "metal",
                modelPath: "",
                maxBatchSize: 4,
                handoff: SmeltHandoffTable(
                    entries: [],
                    ropeCosSlot: 0,
                    ropeSinSlot: 0
                ),
                inputContract: SmeltPrefillInputContract()
            ) : nil,
            inference: inference,
            decode: decode,
            validation: validation,
            optimizationReport: nil
        )
    }

    func testLLMTraceReportsGraphAndDispatchTable() throws {
        let packagePath = try makePackage("smelt-trace-llm")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "text-generation")
        XCTAssertEqual(report.graph?.signature, "text->text")
        XCTAssertEqual(report.graph?.source, "declared")
        XCTAssertEqual(report.loop?.source, "declared")
        XCTAssertEqual(report.loop?.perStep.first?.name, "decode")
        XCTAssertEqual(report.blocks.map(\.name), ["tokenizer", "trunk", "text-head"])
        XCTAssertEqual(
            report.dispatchTables.first { $0.name == "dispatches.bin" }?.dispatchCount,
            1
        )
        XCTAssertFalse(report.hasErrors, report.issues.map(\.message).joined(separator: "\n"))
    }

    func testTextTraceRejectsMissingLoopInsteadOfInferringSchedule() throws {
        let packagePath = try makePackage("smelt-trace-text-missing-loop")
        try writeJSON(minimalLLMManifest(loop: nil), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "text-generation")
        XCTAssertEqual(report.graph?.source, "declared")
        XCTAssertNil(report.loop)
        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "missingTextLoop"
                && $0.message.contains("declared loop schedule")
        })
    }

    func testLLMTraceFlagsCAMPartialStampedTopology() throws {
        let packagePath = try makePackage("smelt-trace-llm-partial-topology")
        try writeLLMTracePackage(
            packagePath,
            manifest: minimalLLMManifest(
                loop: nil,
                inference: llmCAMInference(),
                validation: llmCAMValidation()
            )
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidTextCAMPolicy"
                && $0.message.contains("blocks and loop")
        })
    }

    func testLLMTraceFlagsCAMLoopDrift() throws {
        let packagePath = try makePackage("smelt-trace-llm-loop-drift")
        let driftedLoop = SmeltLoopSchedule(
            setup: [SmeltLoopSchedule.Phase(name: "prefill", blocks: ["trunk", "text-head"])],
            perStep: [SmeltLoopSchedule.Phase(name: "decode", blocks: ["missing-block"])],
            emission: .perStep,
            stop: [.eosToken, .maxSteps, .hostCancel]
        )
        try writeLLMTracePackage(
            packagePath,
            manifest: minimalLLMManifest(
                loop: driftedLoop,
                inference: llmCAMInference(),
                validation: llmCAMValidation()
            )
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidTextCAMPolicy"
                && $0.message.contains("phase 'decode' drives unknown block 'missing-block'")
        })
    }

    func testTTSTraceReportsCompiledRoutesAndSidecars() throws {
        let packagePath = try makePackage("smelt-trace-tts")
        try writeQwenTracePackage(packagePath, manifest: qwenTraceManifest())

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "tts")
        XCTAssertEqual(report.graph?.signature, "text->audio")
        XCTAssertEqual(report.loop?.emission, "chunked:first=1:max=1:growth=double:via=codec-decoder")
        XCTAssertEqual(report.sidecars.map(\.name), ["trunk", "trunk-mtp"])
        XCTAssertTrue(report.sidecars.allSatisfy(\.exists))
        XCTAssertEqual(report.blocks.first { $0.name == "talker" }?.route, "compiled:baked-sidecar")
        XCTAssertEqual(report.blocks.first { $0.name == "mtp-head" }?.route, "native:internal-sidecar")
        XCTAssertFalse(report.hasErrors, report.issues.map(\.message).joined(separator: "\n"))
    }

    func testTTSTraceFlagsCompiledFrontendDTypeMismatch() throws {
        let packagePath = try makePackage("smelt-trace-tts-mismatch")
        try writeQwenTracePackage(
            packagePath,
            manifest: qwenTraceManifest(weightDType: "u4"),
            sidecars: false
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "blockRouteMismatch" && $0.message.contains("tts-frontend")
        })
        XCTAssertEqual(report.blocks.first { $0.name == "tts-frontend" }?.status, .error)
    }

    func testTTSTraceFlagsPartialStampedTopology() throws {
        let packagePath = try makePackage("smelt-trace-tts-partial-topology")
        try writeQwenTracePackage(
            packagePath,
            manifest: qwenTraceManifest(loop: nil)
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidQwen3TTSTopology"
                && $0.message.contains("blocks and loop")
        })
    }

    func testTTSTraceRejectsKnownQwen3TTSMissingBlocks() throws {
        let packagePath = try makePackage("smelt-trace-tts-missing-blocks")
        try writeQwenTracePackage(
            packagePath,
            manifest: qwenTraceManifest(blocks: nil)
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "tts")
        XCTAssertNil(report.graph)
        XCTAssertNotNil(report.loop)
        XCTAssertTrue(report.blocks.isEmpty)
        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidQwen3TTSTopology"
                && $0.message.contains("blocks and loop must be declared")
        })
        XCTAssertTrue(report.issues.contains { $0.code == "missingTTSGraph" })
        XCTAssertFalse(report.issues.contains { $0.code == "genericTrace" })
    }

    func testTTSTraceFlagsStampedLoopDrift() throws {
        let packagePath = try makePackage("smelt-trace-tts-loop-drift")
        let driftedLoop = SmeltLoopSchedule(
            setup: [SmeltLoopSchedule.Phase(name: "prefill", blocks: ["talker"])],
            perStep: [SmeltLoopSchedule.Phase(name: "decode", blocks: ["talker"])],
            emission: .chunked(first: 1, max: 1, growth: .double, via: "codec-decoder"),
            stop: [.eosToken, .maxSteps]
        )
        try writeQwenTracePackage(
            packagePath,
            manifest: qwenTraceManifest(loop: driftedLoop)
        )

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidQwen3TTSTopology"
                && $0.message.contains("declared Qwen3-TTS graph")
        })
    }

    func testTTSTraceGraphlessPackageUsesGenericArtifactTrace() throws {
        let packagePath = try makePackage("smelt-trace-tts-unknown")
        let manifest = """
        {
          "version": 1,
          "kind": "tts",
          "architecture": "mystery-audio",
          "modelName": "test/future-audio",
          "files": {
            "weights": "weights.bin"
          }
        }
        """
        try writeFile(packagePath, "manifest.json", data: Data(manifest.utf8))
        try writeFile(packagePath, "weights.bin")

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "tts")
        XCTAssertEqual(report.modelName, "test/future-audio")
        XCTAssertNil(report.graph)
        XCTAssertNil(report.loop)
        XCTAssertTrue(report.blocks.isEmpty)
        XCTAssertTrue(report.sidecars.isEmpty)
        XCTAssertEqual(Set(report.artifacts.map(\.name)), ["manifest.json", "weights.bin"])
        XCTAssertTrue(report.issues.contains {
            $0.code == "genericTrace"
                && $0.message.contains("tts")
        })
    }

    func testTraceMalformedBlocksDoesNotFallBackAsGraphless() throws {
        let packagePath = try makePackage("smelt-trace-malformed-blocks")
        let manifest = """
        {
          "kind": "tts",
          "architecture": "mystery-audio",
          "modelName": "test/bad-graph",
          "blocks": 7,
          "files": {
            "weights": "weights.bin"
          }
        }
        """
        try writeFile(packagePath, "manifest.json", data: Data(manifest.utf8))
        try writeFile(packagePath, "weights.bin")

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "tts")
        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidRuntimeGraphPolicy"
                && $0.message.contains("manifest blocks")
        })
        XCTAssertFalse(report.issues.contains { $0.code == "genericTrace" })
    }

    func testTraceInvalidDecodableBlocksDoesNotFallBackAsGraphless() throws {
        let packagePath = try makePackage("smelt-trace-invalid-blocks")
        let manifest = """
        {
          "kind": "tts",
          "architecture": "mystery-audio",
          "modelName": "test/invalid-graph",
          "blocks": {
            "version": 1,
            "blocks": []
          },
          "files": {
            "weights": "weights.bin"
          }
        }
        """
        try writeFile(packagePath, "manifest.json", data: Data(manifest.utf8))
        try writeFile(packagePath, "weights.bin")

        let report = try SmeltTrace.inspect(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "tts")
        XCTAssertTrue(report.hasErrors)
        XCTAssertTrue(report.issues.contains {
            $0.code == "invalidRuntimeGraphPolicy"
                && $0.message.contains("runtime graph is invalid")
        })
        XCTAssertFalse(report.issues.contains { $0.code == "genericTrace" })
    }

    func testGenericTraceReportsFuturePackageKindWithoutLLMFallback() throws {
        let packagePath = try makePackage("smelt-trace-future-kind")
        let manifest = """
        {
          "schemaVersion": 1,
          "kind": "embedding",
          "modelName": "test/embedder",
          "files": {
            "weights": "model.bin"
          }
        }
        """
        try writeFile(packagePath, "manifest.json", data: Data(manifest.utf8))
        try writeFile(packagePath, "model.bin", data: Data([1, 2, 3]))
        try writeFile(packagePath, "config/runtime.json", data: Data(#"{"batch":1}"#.utf8))

        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let witness = try SmeltTrace.record(packagePath: packagePath)

        XCTAssertEqual(report.packageKind, "embedding")
        XCTAssertEqual(report.modelName, "test/embedder")
        XCTAssertNil(report.graph)
        XCTAssertNil(report.loop)
        XCTAssertEqual(report.blocks.count, 0)
        XCTAssertEqual(report.artifacts.map(\.name), [
            "config/runtime.json",
            "manifest.json",
            "model.bin",
        ])
        XCTAssertFalse(report.hasErrors, report.issues.map(\.message).joined(separator: "\n"))
        XCTAssertTrue(report.issues.contains { $0.code == "genericTrace" })
        XCTAssertEqual(witness.contract.package.packageKind, "embedding")
        XCTAssertEqual(witness.capture.mode, "package-contract+loop")
    }

    func testGenericTraceWitnessIsStableAcrossPackageDirectories() throws {
        let first = try makePackage("smelt-trace-future-a")
        let second = try makePackage("smelt-trace-future-b")
        let manifest = """
        {
          "schemaVersion": 1,
          "kind": "embedding",
          "modelName": "test/embedder"
        }
        """
        for packagePath in [first, second] {
            try writeFile(packagePath, "manifest.json", data: Data(manifest.utf8))
            try writeFile(packagePath, "model.bin", data: Data([1, 2, 3]))
        }

        let expected = try SmeltTrace.record(packagePath: first)
        let actual = try SmeltTrace.record(packagePath: second)
        let comparison = SmeltTrace.compare(expected: expected, actual: actual)

        XCTAssertTrue(comparison.matches, comparison.differences.map(\.path).joined(separator: ", "))
        XCTAssertEqual(expected.contract.artifacts.map(\.name), ["manifest.json", "model.bin"])
    }

    func testTraceWitnessIsStableAcrossPackageDirectories() throws {
        let first = try makePackage("smelt-trace-witness-a")
        let second = try makePackage("smelt-trace-witness-b")
        for packagePath in [first, second] {
            try writeJSON(minimalLLMManifest(), to: packagePath)
            try writeFile(packagePath, "weights.bin")
            try writeFile(packagePath, "model.metallib")
            try writeFile(packagePath, "SmeltGenerated.swift")
            try writeDispatchTable(packagePath, "dispatches.bin")
        }

        let expected = try SmeltTrace.record(packagePath: first)
        let actual = try SmeltTrace.record(packagePath: second)
        let comparison = SmeltTrace.compare(expected: expected, actual: actual)

        XCTAssertTrue(comparison.matches, comparison.differences.map(\.path).joined(separator: ", "))
        XCTAssertEqual(expected.contractSHA256, actual.contractSHA256)
        XCTAssertFalse(expected.contract.events.isEmpty)
        XCTAssertEqual(expected.capture.mode, "package-contract+loop")
        XCTAssertTrue(expected.capture.eventsCaptured)
        XCTAssertTrue(expected.contract.events.contains {
            $0.kind == "block" && $0.phase == "per-step:decode" && $0.block == "trunk"
                && $0.route == "compiled:baked-inline"
        })
    }

    func testTraceWitnessCompareCatchesPackageContractDrift() throws {
        let expectedPackage = try makePackage("smelt-trace-witness-expected")
        try writeJSON(minimalLLMManifest(), to: expectedPackage)
        try writeFile(expectedPackage, "weights.bin")
        try writeFile(expectedPackage, "model.metallib")
        try writeFile(expectedPackage, "SmeltGenerated.swift")
        try writeDispatchTable(expectedPackage, "dispatches.bin")

        let actualPackage = try makePackage("smelt-trace-witness-actual")
        try writeJSON(minimalLLMManifest(), to: actualPackage)
        try writeFile(actualPackage, "weights.bin")
        try writeFile(actualPackage, "model.metallib")
        try writeFile(actualPackage, "SmeltGenerated.swift")

        let expected = try SmeltTrace.record(packagePath: expectedPackage)
        let actual = try SmeltTrace.record(packagePath: actualPackage)
        let comparison = SmeltTrace.compare(expected: expected, actual: actual)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains { $0.path == "contractSHA256" })
        XCTAssertTrue(comparison.differences.contains { $0.path == "dispatchTables" })
        XCTAssertTrue(comparison.differences.contains { $0.path == "issues" })
    }

    func testTraceWitnessCompareCatchesLoopDrift() throws {
        let expectedPackage = try makePackage("smelt-trace-loop-expected")
        try writeJSON(minimalLLMManifest(loop: .tokenFeedbackText), to: expectedPackage)
        try writeFile(expectedPackage, "weights.bin")
        try writeFile(expectedPackage, "model.metallib")
        try writeFile(expectedPackage, "SmeltGenerated.swift")
        try writeDispatchTable(expectedPackage, "dispatches.bin")

        let actualPackage = try makePackage("smelt-trace-loop-actual")
        let driftedLoop = SmeltLoopSchedule(
            setup: [SmeltLoopSchedule.Phase(name: "prefill", blocks: ["trunk", "text-head"])],
            perStep: [SmeltLoopSchedule.Phase(name: "decode-alt", blocks: ["trunk", "text-head"])],
            emission: .perStep,
            stop: [.eosToken, .maxSteps, .hostCancel]
        )
        try writeJSON(minimalLLMManifest(loop: driftedLoop), to: actualPackage)
        try writeFile(actualPackage, "weights.bin")
        try writeFile(actualPackage, "model.metallib")
        try writeFile(actualPackage, "SmeltGenerated.swift")
        try writeDispatchTable(actualPackage, "dispatches.bin")

        let expected = try SmeltTrace.record(packagePath: expectedPackage)
        let actual = try SmeltTrace.record(packagePath: actualPackage)
        let comparison = SmeltTrace.compare(expected: expected, actual: actual)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains { $0.path == "loop" })
        XCTAssertTrue(comparison.differences.contains { $0.path == "events" })
    }

    func testTraceWitnessWriteLoadAndVerifyRoundTrip() throws {
        let packagePath = try makePackage("smelt-trace-witness-roundtrip")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let witnessPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("golden.smttrace").path

        try SmeltTrace.writeWitness(witness, to: witnessPath)
        let loaded = try SmeltTrace.loadWitness(from: witnessPath)
        let comparison = try SmeltTrace.verify(packagePath: packagePath, against: witnessPath)

        XCTAssertEqual(loaded.contractSHA256, witness.contractSHA256)
        XCTAssertTrue(comparison.matches, comparison.differences.map(\.path).joined(separator: ", "))
    }

    func testTraceReplayUsesEmbeddedRuntimeEvents() throws {
        let packagePath = try makePackage("smelt-trace-runtime-replay")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "setup:prefill",
                block: "trunk",
                route: routes["trunk"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "setup:prefill",
                block: "text-head",
                route: routes["text-head"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 4,
                phase: "per-step:decode",
                block: "trunk",
                route: routes["trunk"],
                step: 0,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 5,
                phase: "per-step:decode",
                block: "text-head",
                route: routes["text-head"],
                step: 0,
                witness: "ok"
            ),
        ]
        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )
        XCTAssertFalse(witness.contract.issues.contains { $0.severity == .error })
        let witnessPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("runtime.smttrace").path
        try SmeltTrace.writeWitness(witness, to: witnessPath)

        let strictVerify = try SmeltTrace.verify(packagePath: packagePath, against: witnessPath)
        let replay = try SmeltTrace.replay(packagePath: packagePath, from: witnessPath)

        XCTAssertFalse(strictVerify.matches)
        XCTAssertTrue(strictVerify.differences.contains { $0.path == "capture" })
        XCTAssertTrue(replay.matches, replay.differences.map(\.path).joined(separator: ", "))
    }

    func testTraceVerifyRejectsMatchingInvalidWitness() throws {
        let packagePath = try makePackage("smelt-trace-invalid-witness-verify")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let trunkRoute = report.blocks.first { $0.name == "trunk" }!.route
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "per-step:decode",
                block: "trunk",
                route: trunkRoute,
                step: 0,
                witness: "ok"
            ),
        ]
        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )
        let witnessPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("bad-golden.smttrace").path
        try SmeltTrace.writeWitness(witness, to: witnessPath)

        let comparison = try SmeltTrace.verify(
            packagePath: packagePath,
            against: witnessPath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.issues.errors"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.issues.errors"
        })
    }

    func testTraceCompareRejectsStaleContractDigest() throws {
        let packagePath = try makePackage("smelt-trace-stale-digest")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let stale = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: "stale",
            contract: witness.contract
        )

        let comparison = SmeltTrace.compare(expected: stale, actual: stale)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contractSHA256.validity"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contractSHA256.validity"
        })
    }

    func testTraceCompareRejectsStaticEventDriftWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-static-event-drift")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let driftedContract = copyContract(
            witness.contract,
            events: Array(witness.contract.events.dropLast())
        )
        let drifted = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(driftedContract),
            contract: driftedContract
        )

        let comparison = SmeltTrace.compare(expected: drifted, actual: drifted)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.events.staticValidity"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.events.staticValidity"
        })
    }

    func testTraceCompareRejectsRuntimeWitnessWithOmittedSemanticIssues() throws {
        let packagePath = try makePackage("smelt-trace-runtime-omitted-issues")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let trunkRoute = report.blocks.first { $0.name == "trunk" }!.route
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "per-step:decode",
                block: "trunk",
                route: trunkRoute,
                step: 0,
                witness: "ok"
            ),
        ]
        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )
        let forgedContract = copyContract(witness.contract, issues: [])
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertFalse(comparison.differences.contains {
            $0.path == "expected.contract.issues.errors"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.issues.runtimeValidity"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.issues.runtimeValidity"
        })
    }

    func testTraceCompareRejectsDuplicateContractBlocksWithoutCrashing() throws {
        let packagePath = try makePackage("smelt-trace-duplicate-blocks")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let duplicateBlocks = witness.contract.blocks + [witness.contract.blocks[1]]
        let forgedContract = copyContract(witness.contract, blocks: duplicateBlocks)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.blocks.names"
                && ($0.actual ?? "").contains("trunk")
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.blocks.names"
                && ($0.actual ?? "").contains("trunk")
        })
    }

    func testTraceCompareRejectsGraphBlockCountDriftWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-graph-count-drift")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let graph = SmeltTraceGraph(
            source: witness.contract.graph!.source,
            signature: witness.contract.graph!.signature,
            blockCount: witness.contract.blocks.count + 1
        )
        let forgedContract = copyContract(witness.contract, graph: graph)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.graph.blockCount"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.graph.blockCount"
        })
    }

    func testTraceCompareRejectsLoopUnknownBlockWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-loop-unknown-block")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let loop = SmeltTraceLoop(
            source: witness.contract.loop!.source,
            setup: [
                SmeltTraceLoopPhase(
                    name: "prefill",
                    blocks: ["trunk", "missing-head"],
                    feedsNextStep: false
                )
            ],
            perStep: witness.contract.loop!.perStep,
            emission: witness.contract.loop!.emission,
            stop: witness.contract.loop!.stop
        )
        let forgedContract = copyContract(
            witness.contract,
            loop: loop,
            events: witness.contract.events
        )
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.loop.blocks"
                && ($0.actual ?? "").contains("missing-head")
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.loop.blocks"
                && ($0.actual ?? "").contains("missing-head")
        })
    }

    func testTraceCompareRejectsDuplicateArtifactsWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-duplicate-artifacts")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let duplicateArtifacts = witness.contract.artifacts + [witness.contract.artifacts[0]]
        let forgedContract = copyContract(witness.contract, artifacts: duplicateArtifacts)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.artifacts.names"
                && ($0.actual ?? "").contains("manifest.json")
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.artifacts.manifest"
        })
    }

    func testTraceCompareRejectsManifestHashDriftWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-manifest-hash-drift")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let artifacts = witness.contract.artifacts.map { artifact in
            guard artifact.name == "manifest.json" else { return artifact }
            return SmeltTraceArtifactContract(
                name: artifact.name,
                kind: artifact.kind,
                exists: artifact.exists,
                bytes: artifact.bytes,
                declaredSHA256: artifact.declaredSHA256,
                actualSHA256: String(repeating: "0", count: 64)
            )
        }
        let forgedContract = copyContract(witness.contract, artifacts: artifacts)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.package.manifestSHA256"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.package.manifestSHA256"
        })
    }

    func testTraceCompareRejectsArtifactChecksumMismatchWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-artifact-checksum-mismatch")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let artifacts = witness.contract.artifacts.map { artifact in
            guard artifact.name == "weights.bin" else { return artifact }
            return SmeltTraceArtifactContract(
                name: artifact.name,
                kind: artifact.kind,
                exists: artifact.exists,
                bytes: artifact.bytes,
                declaredSHA256: String(repeating: "f", count: 64),
                actualSHA256: artifact.actualSHA256 ?? String(repeating: "0", count: 64)
            )
        }
        let forgedContract = copyContract(witness.contract, artifacts: artifacts)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.artifacts.declaredSHA256"
                && ($0.actual ?? "").contains("weights.bin")
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.artifacts.declaredSHA256"
                && ($0.actual ?? "").contains("weights.bin")
        })
    }

    func testTraceCompareRejectsDispatchTableParseErrorWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-dispatch-parse-error")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let dispatchTables = witness.contract.dispatchTables.map { table in
            guard table.name == "dispatches.bin" else { return table }
            return SmeltTraceDispatchTableContract(
                name: table.name,
                exists: table.exists,
                sha256: table.sha256,
                parseError: "bad record",
                totalRecords: table.totalRecords,
                dispatchCount: table.dispatchCount,
                swapCount: table.swapCount,
                topPipelines: table.topPipelines
            )
        }
        let forgedContract = copyContract(witness.contract, dispatchTables: dispatchTables)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.dispatchTables.parseError"
                && ($0.actual ?? "").contains("dispatches.bin")
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "actual.contract.dispatchTables.parseError"
                && ($0.actual ?? "").contains("dispatches.bin")
        })
    }

    func testTraceCompareRejectsDuplicateDispatchTablesWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-duplicate-dispatches")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let forgedTables = witness.contract.dispatchTables + [witness.contract.dispatchTables[0]]
        let forgedContract = copyContract(witness.contract, dispatchTables: forgedTables)
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.dispatchTables.names"
                && ($0.actual ?? "").contains("dispatches.bin")
        })
    }

    func testTraceCompareRejectsSidecarErrorsWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-sidecar-error")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let sidecar = SmeltTraceSidecarContract(
            name: "trunk",
            exists: true,
            packageKind: "headless-trunk",
            modelName: "test/sidecar",
            headlessTrunkABI: true,
            manifestSHA256: String(repeating: "a", count: 64),
            dispatchTables: [],
            issues: [
                SmeltTraceIssue(
                    severity: .error,
                    code: "sidecarInvalidManifest",
                    message: "bad sidecar"
                )
            ]
        )
        let forgedContract = copyContract(witness.contract, sidecars: [sidecar], issues: [])
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertFalse(comparison.differences.contains {
            $0.path == "expected.contract.issues.errors"
        })
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.sidecars.issues.errors"
                && ($0.actual ?? "").contains("sidecarInvalidManifest")
        })
    }

    func testTraceCompareRejectsSidecarDispatchParseErrorWithFreshDigest() throws {
        let packagePath = try makePackage("smelt-trace-sidecar-dispatch-error")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let witness = try SmeltTrace.record(packagePath: packagePath)
        let table = SmeltTraceDispatchTableContract(
            name: "dispatches.bin",
            exists: true,
            sha256: String(repeating: "b", count: 64),
            parseError: "bad sidecar dispatch",
            totalRecords: nil,
            dispatchCount: nil,
            swapCount: nil,
            topPipelines: []
        )
        let sidecar = SmeltTraceSidecarContract(
            name: "trunk",
            exists: true,
            packageKind: "headless-trunk",
            modelName: "test/sidecar",
            headlessTrunkABI: true,
            manifestSHA256: String(repeating: "a", count: 64),
            dispatchTables: [table],
            issues: []
        )
        let forgedContract = copyContract(witness.contract, sidecars: [sidecar])
        let forged = SmeltTraceWitness(
            schemaVersion: witness.schemaVersion,
            format: witness.format,
            capture: witness.capture,
            contractSHA256: try canonicalSHA256(forgedContract),
            contract: forgedContract
        )

        let comparison = SmeltTrace.compare(expected: forged, actual: forged)

        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains {
            $0.path == "expected.contract.sidecars.dispatchTables.parseError"
                && ($0.actual ?? "").contains("trunk:dispatches.bin")
        })
    }

    func testTraceWitnessCanCarryRuntimeEvents() throws {
        let packagePath = try makePackage("smelt-trace-runtime-events")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let runtimeEvents = [
            SmeltTraceEvent(
                kind: "phase-begin",
                index: 0,
                phase: "per-step:decode",
                step: 0,
                witness: "feedsNextStep=false"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "per-step:decode",
                block: "trunk",
                route: "compiled:baked-inline",
                step: 0,
                witness: "ok"
            ),
        ]

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertEqual(witness.capture.mode, "package-contract+runtime")
        XCTAssertEqual(witness.contract.events, runtimeEvents)
        XCTAssertTrue(witness.capture.eventsCaptured)
    }

    func testCAMRouteEventPrefixesRuntimeEventsAndPreservesIndices() throws {
        let packagePath = try makePackage("smelt-trace-cam-route-events")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let capabilities = try SmeltCAMPackageCapabilities(
            descriptor: SmeltCAMPackageDescriptor(
                from: registryModuleIR("qwen35_text")
            )
        )
        let decision = try capabilities.resolve(.traceTextGenerate)
        let runtimeEvents = [
            SmeltTraceEvent(
                kind: "case",
                index: 0,
                witness: "io=text->text"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "per-step:decode",
                block: "trunk",
                route: "compiled:baked-inline",
                step: 0,
                witness: "ok"
            ),
        ]

        let routeWitness = decision.traceRouteWitnessV6(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
        let events = SmeltTraceCAMRoute.events(
            witness: routeWitness,
            followedBy: runtimeEvents
        )
        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: events)
        )

        XCTAssertEqual(events.map(\.index), [0, 1, 2])
        XCTAssertEqual(witness.capture.mode, "package-contract+runtime")
        XCTAssertEqual(witness.contract.events.first?.kind, SmeltTraceCAMRoute.eventKind)
        XCTAssertEqual(witness.contract.events.first?.witness, routeWitness)
        XCTAssertTrue(routeWitness.hasPrefix("cam-route:v6;"))
        XCTAssertFalse(routeWitness.contains("manifestBridgeRoute"))
        XCTAssertTrue(routeWitness.contains("export=generate"))
        XCTAssertFalse(routeWitness.contains("request="))
        XCTAssertFalse(routeWitness.contains("requiredInputs="))
        XCTAssertFalse(routeWitness.contains("requiredOutputs="))
        XCTAssertFalse(routeWitness.contains("temporaryRuntimeWitness"))
        XCTAssertFalse(routeWitness.contains("temporary-"))
        XCTAssertEqual(witness.contract.events[1].kind, "case")
        XCTAssertEqual(witness.contract.events[1].index, 1)
        XCTAssertEqual(witness.contract.events[2].kind, "block-finish")
        XCTAssertEqual(witness.contract.events[2].index, 2)
        XCTAssertFalse(witness.contract.issues.contains { $0.code == "runtimeEventIndexMismatch" })

        let withoutRoute = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )
        let comparison = SmeltTrace.compare(expected: witness, actual: withoutRoute)
        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains { $0.path == "events" })
    }

    func testCAMRouteRejectsOldTemporaryRuntimeWitnessRoute() throws {
        let packagePath = try makePackage("smelt-trace-cam-route-v6-drift")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let capabilities = try SmeltCAMPackageCapabilities(
            descriptor: SmeltCAMPackageDescriptor(
                from: registryModuleIR("qwen35_text")
            )
        )
        let decision = try capabilities.resolve(.traceTextGenerate)
        let freshRouteWitness = decision.traceRouteWitnessV6(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
        let oldRouteWitness = freshRouteWitness.replacingOccurrences(
            of: "cam-route:v6;",
            with: "cam-route:v3;"
        ) + ";temporaryRuntimeWitness=temporary-text-to-text-trace-runtime"
        let runtimeEvents = fullTextRuntimeEvents(report: report)

        let old = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: oldRouteWitness,
                    followedBy: runtimeEvents
                )
            )
        )
        let fresh = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: freshRouteWitness,
                    followedBy: runtimeEvents
                )
            )
        )
        let comparison = SmeltTrace.compare(expected: old, actual: fresh)

        XCTAssertTrue(old.contract.issues.contains { $0.code == "runtimeCAMRouteInvalid" })
        XCTAssertFalse(fresh.contract.issues.contains { $0.severity == .error })
        XCTAssertNotEqual(old.contractSHA256, fresh.contractSHA256)
        XCTAssertFalse(comparison.matches)
        XCTAssertTrue(comparison.differences.contains { $0.path == "contractSHA256" })
        XCTAssertTrue(comparison.differences.contains { $0.path == "events" })
    }

    func testCAMRouteRejectsMalformedV6WithBackendOrRetiredFields() throws {
        let packagePath = try makePackage("smelt-trace-cam-route-v6-retired-fields")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let capabilities = try SmeltCAMPackageCapabilities(
            descriptor: SmeltCAMPackageDescriptor(
                from: registryModuleIR("qwen35_text")
            )
        )
        let decision = try capabilities.resolve(.traceTextGenerate)
        let routeWitness = decision.traceRouteWitnessV6(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
        let backendFieldWitness = routeWitness + ";manifestBridgeRoute=text-to-text-manifest-bridge"

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: backendFieldWitness,
                    followedBy: fullTextRuntimeEvents(report: report)
                )
            )
        )

        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeCAMRouteInvalid"
                && $0.message.contains("unknown witness field 'manifestBridgeRoute'")
        })

        let extraFieldWitness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: routeWitness + ";surprise=bad",
                    followedBy: fullTextRuntimeEvents(report: report)
                )
            )
        )
        XCTAssertTrue(extraFieldWitness.contract.issues.contains {
            $0.code == "runtimeCAMRouteInvalid"
                && $0.message.contains("unknown witness field 'surprise'")
        })

        let emptyExportWitness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: routeWitness.replacingOccurrences(
                        of: ";export=generate;",
                        with: ";export=;"
                    ),
                    followedBy: fullTextRuntimeEvents(report: report)
                )
            )
        )
        XCTAssertTrue(emptyExportWitness.contract.issues.contains {
            $0.code == "runtimeCAMRouteInvalid"
                && $0.message.contains("empty witness field 'export'")
        })

        for retiredField in [
            "request",
            "requiredInputs",
            "requiredOutputs",
            "temporaryRuntimeWitness",
            "currentBackendContract",
        ] {
            let retiredFieldWitness = try SmeltTrace.record(
                packagePath: packagePath,
                options: SmeltTraceRecordOptions(
                    runtimeEvents: SmeltTraceCAMRoute.events(
                        witness: routeWitness + ";\(retiredField)=bad",
                        followedBy: fullTextRuntimeEvents(report: report)
                    )
                )
            )
            XCTAssertTrue(retiredFieldWitness.contract.issues.contains {
                $0.code == "runtimeCAMRouteInvalid"
                    && $0.message.contains("unknown witness field '\(retiredField)'")
            })
        }

        let v4RequestWitness = routeWitness.replacingOccurrences(
            of: "cam-route:v6;",
            with: "cam-route:v4;"
        ) + ";request=trace%20text%20generation"
        let v4Witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: v4RequestWitness,
                    followedBy: fullTextRuntimeEvents(report: report)
                )
            )
        )
        XCTAssertTrue(v4Witness.contract.issues.contains {
            $0.code == "runtimeCAMRouteInvalid"
                && $0.message.contains("must use cam-route:v6")
        })
    }

    func testCAMRouteCompareAndReplayRejectStaleOldRouteWitness() throws {
        let packagePath = try makePackage("smelt-trace-cam-route-v6-stale-replay")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let capabilities = try SmeltCAMPackageCapabilities(
            descriptor: SmeltCAMPackageDescriptor(
                from: registryModuleIR("qwen35_text")
            )
        )
        let decision = try capabilities.resolve(.traceTextGenerate)
        let freshRouteWitness = decision.traceRouteWitnessV6(
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
        let oldRouteWitness = freshRouteWitness.replacingOccurrences(
            of: "cam-route:v6;",
            with: "cam-route:v5;"
        ) + ";manifestBridgeRoute=text-to-text-manifest-bridge"
        let oldEvents = SmeltTraceCAMRoute.events(
            witness: oldRouteWitness,
            followedBy: fullTextRuntimeEvents(report: report)
        )
        let fresh = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                runtimeEvents: SmeltTraceCAMRoute.events(
                    witness: freshRouteWitness,
                    followedBy: fullTextRuntimeEvents(report: report)
                )
            )
        )
        let staleContract = copyContract(fresh.contract, events: oldEvents, issues: [])
        let stale = SmeltTraceWitness(
            schemaVersion: fresh.schemaVersion,
            format: fresh.format,
            capture: fresh.capture,
            contractSHA256: try canonicalSHA256(staleContract),
            contract: staleContract
        )
        let witnessPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("stale.smttrace").path
        try SmeltTrace.writeWitness(stale, to: witnessPath)

        let compare = SmeltTrace.compare(expected: stale, actual: stale)
        let replay = try SmeltTrace.replay(packagePath: packagePath, from: witnessPath)

        XCTAssertFalse(compare.matches)
        XCTAssertTrue(compare.differences.contains { $0.path == "expected.contract.issues.runtimeValidity" })
        XCTAssertFalse(replay.matches)
        XCTAssertTrue(replay.differences.contains { $0.path == "expected.contract.issues.runtimeValidity" })
        XCTAssertTrue(replay.differences.contains { $0.path == "actual.contract.issues.errors" })
    }

    func testTraceWitnessFlagsMissingRuntimeBlockCoverage() throws {
        let packagePath = try makePackage("smelt-trace-runtime-missing-blocks")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let trunkRoute = report.blocks.first { $0.name == "trunk" }!.route
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "per-step:decode",
                block: "trunk",
                route: trunkRoute,
                step: 0,
                witness: "ok"
            ),
        ]

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingBlocks"
                && $0.message.contains("tokenizer")
                && $0.message.contains("text-head")
        })
        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingLoopBlocks"
                && $0.message.contains("text-head")
        })
    }

    func testTraceWitnessFlagsMissingRuntimePhaseBlockCoverage() throws {
        let packagePath = try makePackage("smelt-trace-runtime-missing-phase-blocks")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "setup:prefill",
                block: "trunk",
                route: routes["trunk"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "setup:prefill",
                block: "text-head",
                route: routes["text-head"],
                step: -1,
                witness: "ok"
            ),
        ]

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertFalse(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingBlocks"
        })
        XCTAssertFalse(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingLoopBlocks"
        })
        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingLoopPhaseBlocks"
                && $0.message.contains("per-step:decode")
                && $0.message.contains("trunk")
                && $0.message.contains("text-head")
        })
    }

    func testTraceWitnessFlagsRuntimeRouteMismatch() throws {
        let packagePath = try makePackage("smelt-trace-runtime-route-mismatch")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "per-step:decode",
                block: "trunk",
                route: "compiled:wrong",
                step: 0,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "per-step:decode",
                block: "text-head",
                route: routes["text-head"],
                step: 0,
                witness: "ok"
            ),
        ]

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeRouteMismatch" && $0.message.contains("trunk")
        })
        XCTAssertFalse(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingBlocks"
        })
    }

    func testTraceWitnessFlagsRuntimeUnknownLoopPhase() throws {
        let packagePath = try makePackage("smelt-trace-runtime-unknown-loop-phase")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "per-step:bogus",
                block: "trunk",
                route: routes["trunk"],
                step: 0,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "per-step:decode",
                block: "text-head",
                route: routes["text-head"],
                step: 0,
                witness: "ok"
            ),
        ]

        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )

        XCTAssertTrue(witness.contract.issues.contains {
            $0.code == "runtimeUnknownLoopPhase"
                && $0.message.contains("per-step:bogus")
        })
        XCTAssertFalse(witness.contract.issues.contains {
            $0.code == "runtimeCoverageMissingBlocks"
        })
    }

    func testTraceSuiteUpdateAndVerifyRoundTrip() throws {
        let packagePath = try makePackage("smelt-trace-suite-package")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")

        let suiteRoot = try makePackage("smelt-trace-suite-root")
        let suite = SmeltTraceSuiteSpec(
            package: packagePath,
            cases: [
                SmeltTraceSuiteCaseSpec(
                    name: "static-contract",
                    golden: "goldens/static-contract.smttrace"
                )
            ]
        )
        let suitePath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("suite.json").path
        try writeJSON(suite, to: suiteRoot, fileName: "suite.json")

        let update = try SmeltTrace.verifySuite(
            path: suitePath,
            options: SmeltTraceSuiteOptions(updateGoldens: true)
        )
        XCTAssertTrue(update.matches)
        XCTAssertEqual(update.cases.first?.updated, true)

        let goldenPath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("goldens/static-contract.smttrace").path
        XCTAssertTrue(FileManager.default.fileExists(atPath: goldenPath))

        let verify = try SmeltTrace.verifySuite(path: suitePath)
        XCTAssertTrue(verify.matches, verify.cases.first?.error ?? "")
        XCTAssertEqual(verify.cases.first?.updated, false)
    }

    func testTraceSuitePackageOverrideCatchesDrift() throws {
        let expectedPackage = try makePackage("smelt-trace-suite-expected")
        try writeJSON(minimalLLMManifest(), to: expectedPackage)
        try writeFile(expectedPackage, "weights.bin")
        try writeFile(expectedPackage, "model.metallib")
        try writeFile(expectedPackage, "SmeltGenerated.swift")
        try writeDispatchTable(expectedPackage, "dispatches.bin")

        let driftedPackage = try makePackage("smelt-trace-suite-drifted")
        try writeJSON(minimalLLMManifest(), to: driftedPackage)
        try writeFile(driftedPackage, "weights.bin")
        try writeFile(driftedPackage, "model.metallib")
        try writeFile(driftedPackage, "SmeltGenerated.swift")

        let suiteRoot = try makePackage("smelt-trace-suite-drift-root")
        let suite = SmeltTraceSuiteSpec(
            package: expectedPackage,
            cases: [
                SmeltTraceSuiteCaseSpec(
                    name: "static-contract",
                    golden: "goldens/static-contract.smttrace"
                )
            ]
        )
        let suitePath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("suite.json").path
        try writeJSON(suite, to: suiteRoot, fileName: "suite.json")

        let update = try SmeltTrace.verifySuite(
            path: suitePath,
            options: SmeltTraceSuiteOptions(updateGoldens: true)
        )
        XCTAssertTrue(update.matches)

        let verify = try SmeltTrace.verifySuite(
            path: suitePath,
            options: SmeltTraceSuiteOptions(packageOverride: driftedPackage)
        )

        XCTAssertFalse(verify.matches)
        XCTAssertTrue(verify.cases.first?.differences.contains { $0.path == "dispatchTables" } == true)
    }

    func testTraceSuiteRejectsUpdateWithPackageOverride() throws {
        let suitePackage = try makePackage("smelt-trace-suite-update-override-suite")
        try writeJSON(minimalLLMManifest(), to: suitePackage)
        try writeFile(suitePackage, "weights.bin")
        try writeFile(suitePackage, "model.metallib")
        try writeFile(suitePackage, "SmeltGenerated.swift")
        try writeDispatchTable(suitePackage, "dispatches.bin")

        let overridePackage = try makePackage("smelt-trace-suite-update-override-package")
        try writeJSON(minimalLLMManifest(), to: overridePackage)
        try writeFile(overridePackage, "weights.bin")
        try writeFile(overridePackage, "model.metallib")
        try writeFile(overridePackage, "SmeltGenerated.swift")
        try writeDispatchTable(overridePackage, "dispatches.bin")

        let suiteRoot = try makePackage("smelt-trace-suite-update-override-root")
        let suite = SmeltTraceSuiteSpec(
            package: suitePackage,
            cases: [
                SmeltTraceSuiteCaseSpec(
                    name: "static-contract",
                    golden: "goldens/static-contract.smttrace"
                )
            ]
        )
        let suitePath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("suite.json").path
        let goldenPath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("goldens/static-contract.smttrace").path
        try writeJSON(suite, to: suiteRoot, fileName: "suite.json")

        XCTAssertThrowsError(
            try SmeltTrace.verifySuite(
                path: suitePath,
                options: SmeltTraceSuiteOptions(
                    packageOverride: overridePackage,
                    updateGoldens: true
                )
            )
        ) { error in
            XCTAssertTrue(
                (error as NSError).localizedDescription.contains("package override"),
                "\(error)"
            )
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: goldenPath))
    }

    func testTraceSuiteReplaysRuntimeGoldenWithoutEventsFile() throws {
        let packagePath = try makePackage("smelt-trace-suite-runtime-replay-package")
        try writeJSON(minimalLLMManifest(), to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
        let report = try SmeltTrace.inspect(packagePath: packagePath)
        let routes = Dictionary(uniqueKeysWithValues: report.blocks.map { ($0.name, $0.route) })
        let runtimeEvents = [
            SmeltTraceEvent(kind: "case", index: 0, witness: "io=text->text"),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 1,
                phase: "input:tokenize",
                block: "tokenizer",
                route: routes["tokenizer"],
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 2,
                phase: "setup:prefill",
                block: "trunk",
                route: routes["trunk"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 3,
                phase: "setup:prefill",
                block: "text-head",
                route: routes["text-head"],
                step: -1,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 4,
                phase: "per-step:decode",
                block: "trunk",
                route: routes["trunk"],
                step: 0,
                witness: "ok"
            ),
            SmeltTraceEvent(
                kind: "block-finish",
                index: 5,
                phase: "per-step:decode",
                block: "text-head",
                route: routes["text-head"],
                step: 0,
                witness: "ok"
            ),
        ]

        let suiteRoot = try makePackage("smelt-trace-suite-runtime-replay-root")
        let goldenPath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("goldens/runtime.smttrace").path
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: goldenPath).deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let witness = try SmeltTrace.record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(runtimeEvents: runtimeEvents)
        )
        try SmeltTrace.writeWitness(witness, to: goldenPath)
        let suite = SmeltTraceSuiteSpec(
            package: packagePath,
            cases: [
                SmeltTraceSuiteCaseSpec(
                    name: "runtime-replay",
                    golden: "goldens/runtime.smttrace"
                )
            ]
        )
        let suitePath = URL(fileURLWithPath: suiteRoot)
            .appendingPathComponent("suite.json").path
        try writeJSON(suite, to: suiteRoot, fileName: "suite.json")

        let result = try SmeltTrace.verifySuite(path: suitePath)

        XCTAssertTrue(result.matches, result.cases.first?.error ?? "")
        XCTAssertEqual(result.cases.first?.updated, false)
    }

    func testTraceSuiteRejectsUnsupportedSchemaVersion() throws {
        try assertTraceSuiteValidationError(
            SmeltTraceSuiteSpec(
                schemaVersion: 2,
                cases: [
                    SmeltTraceSuiteCaseSpec(
                        name: "static-contract",
                        golden: "goldens/static-contract.smttrace"
                    )
                ]
            ),
            contains: "unsupported trace suite schemaVersion"
        )
    }

    func testTraceSuiteRejectsEmptyCases() throws {
        try assertTraceSuiteValidationError(
            SmeltTraceSuiteSpec(cases: []),
            contains: "at least one case"
        )
    }

    func testTraceSuiteRejectsDuplicateCaseNames() throws {
        try assertTraceSuiteValidationError(
            SmeltTraceSuiteSpec(
                cases: [
                    SmeltTraceSuiteCaseSpec(
                        name: "static-contract",
                        golden: "goldens/a.smttrace"
                    ),
                    SmeltTraceSuiteCaseSpec(
                        name: " static-contract ",
                        golden: "goldens/b.smttrace"
                    ),
                ]
            ),
            contains: "duplicate trace suite case name 'static-contract'"
        )
    }

    func testTraceSuiteRejectsBlankCaseName() throws {
        try assertTraceSuiteValidationError(
            SmeltTraceSuiteSpec(
                cases: [
                    SmeltTraceSuiteCaseSpec(
                        name: "  ",
                        golden: "goldens/static-contract.smttrace"
                    )
                ]
            ),
            contains: "blank name"
        )
    }

    func testTraceSuiteRejectsBlankGolden() throws {
        try assertTraceSuiteValidationError(
            SmeltTraceSuiteSpec(
                cases: [
                    SmeltTraceSuiteCaseSpec(
                        name: "static-contract",
                        golden: "  "
                    )
                ]
            ),
            contains: "blank golden"
        )
    }

    private func ttsEntry(_ name: String, dtype: String) -> Qwen3TTSManifest.Entry {
        Qwen3TTSManifest.Entry(
            name: name,
            offset: 0,
            byteLength: 4,
            shape: [1],
            dtype: dtype
        )
    }

    private func llmCAMInference() -> SmeltInferenceManifest {
        SmeltInferenceManifest(
            maxTokens: 8,
            eosTokens: [0],
            chatTemplate: "chatml",
            thinkingPolicy: .disabled
        )
    }

    private func llmCAMValidation() -> SmeltPackageSpec.Validation {
        SmeltPackagePerformanceProfiles.validation(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
    }

    private func writeLLMTracePackage(
        _ packagePath: String,
        manifest: SmeltManifest
    ) throws {
        try writeJSON(manifest, to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        try writeFile(packagePath, "SmeltGenerated.swift")
        try writeDispatchTable(packagePath, "dispatches.bin")
    }

    private func qwenTraceManifest(
        blocks: SmeltBlockGraph? = .qwen3TTSCompiledTalker,
        loop: SmeltLoopSchedule? = .qwen3TTS,
        weightDType: String = "bf16"
    ) -> Qwen3TTSManifest {
        Qwen3TTSManifest(
            version: 1,
            blocks: blocks,
            loop: loop,
            modelName: "test/qwen3-tts",
            pageSize: 4096,
            pipelines: Qwen3TTSCodecEmitter.pipelineNames,
            eosTokens: [0],
            totalBytes: 4,
            weights: [
                ttsEntry("talker.model.text_embedding.weight", dtype: weightDType),
                ttsEntry("talker.model.codec_embedding.weight", dtype: weightDType),
                ttsEntry("talker.model.layers.0.self_attn.q_proj.weight", dtype: weightDType),
                ttsEntry("talker.code_predictor.model.layers.0.self_attn.q_proj.weight", dtype: weightDType),
            ]
        )
    }

    private func writeQwenTracePackage(
        _ packagePath: String,
        manifest: Qwen3TTSManifest,
        sidecars: Bool = true
    ) throws {
        try writeJSON(manifest, to: packagePath)
        try writeFile(packagePath, "weights.bin")
        try writeFile(packagePath, "model.metallib")
        guard sidecars else { return }
        try writeHeadlessSidecar(packagePath, "trunk")
        try writeHeadlessSidecar(packagePath, "trunk-mtp")
    }

    private func writeHeadlessSidecar(_ packagePath: String, _ name: String) throws {
        let sidecarPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent(name, isDirectory: true).path
        try FileManager.default.createDirectory(
            atPath: sidecarPath,
            withIntermediateDirectories: true
        )
        let manifest = minimalLLMManifest(
            kind: nil,
            headlessTrunkABI: true,
            blocks: nil,
            loop: nil,
            prefill: true,
            inference: nil,
            decode: nil,
            validation: nil
        )
        try writeJSON(manifest, to: sidecarPath)
        try writeFile(sidecarPath, "weights.bin")
        try writeFile(sidecarPath, "model.metallib")
        try writeFile(sidecarPath, "SmeltGenerated.swift")
        try writeDispatchTable(sidecarPath, "dispatches.bin")
        try writeDispatchTable(sidecarPath, "prefill_dispatches.bin")
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
}
