import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class SmeltRuntimePipelineMaterializationTests: XCTestCase {
    private func makeManifest(pipelines: [String]) -> SmeltManifest {
        SmeltManifest(
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
            weights: SmeltWeightManifest(totalBytes: 0, entries: []),
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
            prefill: SmeltPrefillManifest(
                engine: "metal",
                modelPath: "",
                maxBatchSize: 64,
                handoff: SmeltHandoffTable(
                    entries: [],
                    ropeCosSlot: 0,
                    ropeSinSlot: 0
                ),
                inputContract: SmeltPrefillInputContract()
            )
        )
    }

    private func dispatchRecord(pipeline: UInt16) -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = pipeline
        return record
    }

    func testRequiredPipelineIndicesFollowDispatchTablesAndRuntimeFallbacks() throws {
        let manifest = makeManifest(pipelines: [
            "seed",
            "causal_gqa_attn_cached_f32",
            "causal_gqa_attn_cached_scalar_f32",
            "other",
            "unused",
        ])
        let decode = [dispatchRecord(pipeline: 3)]
        let prefill = [dispatchRecord(pipeline: 1), SmeltDispatchRecord.swap()]

        try decode.withUnsafeBufferPointer { decodeTable in
            try prefill.withUnsafeBufferPointer { prefillTable in
                let required = try SmeltRuntime.requiredPipelineIndices(
                    manifest: manifest,
                    dispatchTables: [
                        ("dispatches.bin", decodeTable),
                        ("prefill_dispatches.bin", prefillTable),
                    ],
                    eager: false
                )

                XCTAssertEqual(required, [1, 2, 3])
            }
        }
    }

    func testRequiredPipelineIndicesFallBackToAllWhenTablesAreAbsent() throws {
        let manifest = makeManifest(pipelines: ["a", "b", "c"])
        let required = try SmeltRuntime.requiredPipelineIndices(
            manifest: manifest,
            dispatchTables: [],
            eager: false
        )
        XCTAssertEqual(required, [0, 1, 2])
    }

    func testRequiredPipelineIndicesEagerModeKeepsAllPipelines() throws {
        let manifest = makeManifest(pipelines: ["a", "b", "c"])
        let table = [dispatchRecord(pipeline: 1)]

        try table.withUnsafeBufferPointer { dispatchTable in
            let required = try SmeltRuntime.requiredPipelineIndices(
                manifest: manifest,
                dispatchTables: [("dispatches.bin", dispatchTable)],
                eager: true
            )
            XCTAssertEqual(required, [0, 1, 2])
        }
    }

    func testRequiredPipelineIndicesRejectOutOfRangeRecords() throws {
        let manifest = makeManifest(pipelines: ["a", "b"])
        let table = [dispatchRecord(pipeline: 9)]

        try table.withUnsafeBufferPointer { dispatchTable in
            XCTAssertThrowsError(
                try SmeltRuntime.requiredPipelineIndices(
                    manifest: manifest,
                    dispatchTables: [("dispatches.bin", dispatchTable)],
                    eager: false
                )
            )
        }
    }
}
