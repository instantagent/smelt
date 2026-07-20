import Foundation
import Testing

@testable import SmeltCompiler
import SmeltRuntime
import SmeltSchema

@Suite("Frozen IR cost model")
struct FrozenIRCostModelTests {
    private let context = SmeltCostModelContext(
        mode: .decode,
        sequenceLength: 1,
        position: 0
    )

    @Test("Component spans reconcile to the clean frozen-plan command buffer")
    func componentSpanCalibration() throws {
        var builder = SmeltFrozenComponentPlanBuilder(
            planID: "component-fixture",
            provenanceKey: "fixture:component",
            context: SmeltCostModelContext(mode: .prefill, sequenceLength: 4)
        )
        for index in 0..<3 {
            builder.append(
                pipeline: "fixture_kernel_\(index)",
                operationGroup: index == 2 ? "normalization" : "elementwise",
                logicalShape: [4, index + 1],
                resources: [SmeltFrozenComponentPlanBuilder.resource(
                    bindingIndex: 0,
                    name: "activation",
                    storageClass: .hotActivation,
                    access: .readWrite,
                    bytes: UInt64((index + 1) * 16)
                )],
                operations: [SmeltFrozenComponentPlanBuilder.operation(
                    index == 2 ? .reduction : .fp32Arithmetic,
                    count: UInt64((index + 1) * 8)
                )]
            )
        }
        let plan = builder.build()
        let profile = SmeltFrozenIRExecutionProfile(
            provenanceKey: plan.provenanceKey,
            context: plan.context,
            deviceName: "fixture-device",
            measurementMethod: "fixture-stage-spans",
            wholePlanGPUUs: 220,
            spans: [
                SmeltFrozenIRExecutionSpan(recordIndices: [0, 1], gpuUs: 150),
                SmeltFrozenIRExecutionSpan(recordIndices: [2], gpuUs: 50),
            ]
        )

        let calibration = try SmeltFrozenIRCostModel.calibration(
            plan: plan,
            profiles: [profile],
            cleanWholePlanGPUUs: [100]
        )
        let report = SmeltFrozenIRCostModel.report(
            plan: plan,
            calibration: calibration
        )

        #expect(calibration.dispatchSamples.count == 3)
        #expect(calibration.instrumentedWholePlanMedianGPUUs == 220)
        #expect(calibration.instrumentedSpanMedianGPUUs == 200)
        #expect(report.summary.calibrationStatus == .exactArtifactMatch)
        #expect(report.summary.calibratedDispatchCount == 3)
        #expect(abs((report.summary.calibratedMedianGPUUs ?? 0) - 100) < 0.000_001)
        #expect(abs((report.summary.additiveCalibrationErrorFraction ?? 1)) < 0.000_001)
        #expect(Set(calibration.dispatchSamples.map(\.key.stableID)).count == 3)
    }

    @Test("Signed matvec bill resolves weight regions and logical spans")
    func signedMatvecResourceBill() throws {
        let fixture = makeFixture(includeUnknown: false)
        let report = try fixture.model.report(context: context)
        let bill = try #require(report.records.first)

        #expect(bill.operationGroup == "signed.bitplane_matvec")
        #expect(bill.isDescribed)
        #expect(bill.resources.map(\.resourceName) == [
            "projection", "projection.scales", "activationPlanes",
            "activationScales", "hiddenA",
        ])
        #expect(bill.resources.map(\.logicalBytes) == [128, 16, 64, 2, 16])
        #expect(report.summary.descriptorCoverageFraction == 1)

        let streaming = try #require(report.summary.storageTotals.first {
            $0.storageClass == .streamingWeight
        })
        #expect(streaming.readBytes == 144)
        #expect(streaming.writeBytes == 0)
    }

    @Test("Unknown descriptors and spans remain explicit")
    func unknownDescriptorIsExplicit() throws {
        let fixture = makeFixture(includeUnknown: true)
        let report = try fixture.model.report(context: context)
        let unknown = try #require(report.records.first { $0.pipeline == "new_kernel" })

        #expect(unknown.operationGroup == nil)
        #expect(!unknown.isDescribed)
        #expect(unknown.unknowns.contains("missing operation-group descriptor"))
        #expect(unknown.resources.first?.access == .unknown)
        #expect(unknown.resources.first?.logicalBytes == nil)
        #expect(report.summary.unknownDispatches == ["new_kernel"])
        #expect(report.summary.intermediateMaterializationBytes == nil)
        #expect(report.summary.calibrationStatus == .absent)
        #expect(report.summary.calibratedMedianGPUUs == nil)
    }

    @Test("Runtime swaps resolve cur and alt before resource accounting")
    func swapsResolveVariableSlots() throws {
        let fixture = makeFixture(includeUnknown: true)
        let report = try fixture.model.report(context: context)
        let first = try #require(report.records.first { $0.pipeline.contains("matvec") })
        let unknown = try #require(report.records.first { $0.pipeline == "new_kernel" })

        #expect(first.resources.last?.slotIndex == 0)
        // After swap: cur=1, alt=0. The unknown record binds alt.
        #expect(unknown.resources.first?.slotIndex == 0)
        #expect(report.summary.swapCount == 1)
    }

    @Test("Exact calibration produces measured-time descriptor coverage")
    func calibrationCoverage() throws {
        let base = makeFixture(includeUnknown: true)
        let signedKey = SmeltFrozenIRCostModel.dispatchCostKey(
            record: base.records[0],
            pipeline: "signed_binary_bitplane_i4_matvec_g128_rows8",
            context: context
        )
        let unknownKey = SmeltFrozenIRCostModel.dispatchCostKey(
            record: base.records[2],
            pipeline: "new_kernel",
            context: context
        )
        let calibration = SmeltDeviceCostCalibration(
            provenanceKey: "m2-max:test-package",
            deviceName: "Apple M2 Max",
            measurementMethod: "fixture",
            context: context,
            dispatchesSHA256: "dispatches",
            metallibSHA256: "metal",
            weightsSHA256: "weights",
            hostRecordUs: 0.25,
            wholePlanMedianGPUUs: 10,
            wholePlanP95GPUUs: 11,
            dispatchSamples: [
                SmeltDispatchCostSample(
                    key: signedKey,
                    medianGPUUs: 19,
                    p95GPUUs: 20,
                    sampleCount: 20
                ),
                SmeltDispatchCostSample(
                    key: unknownKey,
                    medianGPUUs: 1,
                    p95GPUUs: 2,
                    sampleCount: 20
                ),
            ]
        )
        let model = SmeltFrozenIRCostModel(
            manifest: base.manifest,
            records: base.records,
            calibration: calibration
        )
        let report = try model.report(context: context)

        #expect(report.provenanceKey
            == "device=Apple M2 Max:build=unknown-build:table=dispatches.bin:dispatches=dispatches:metallib=metal:weights=weights")
        #expect(report.summary.calibratedDispatchCount == 2)
        #expect(report.summary.calibratedMedianGPUUs == 20)
        #expect(report.summary.describedCalibratedMedianGPUUs == 19)
        #expect(report.summary.measuredGPUCoverageFraction == 0.95)
        #expect(report.summary.measuredWholePlanMedianGPUUs == 10)
        #expect(report.summary.additiveCalibrationErrorFraction == 1)
        #expect(report.summary.predictedHostRecordUs == 0.75)
        #expect(report.summary.calibrationStatus == .exactArtifactMatch)
    }

    @Test("Mismatched artifact calibration is rejected explicitly")
    func mismatchedCalibrationIsRejected() throws {
        let fixture = makeFixture(includeUnknown: false)
        let key = SmeltFrozenIRCostModel.dispatchCostKey(
            record: fixture.records[0],
            pipeline: "signed_binary_bitplane_i4_matvec_g128_rows8",
            context: context
        )
        let calibration = SmeltDeviceCostCalibration(
            provenanceKey: "wrong-package",
            deviceName: "Apple M2 Max",
            measurementMethod: "fixture",
            context: context,
            buildFingerprint: "wrong-build",
            dispatchesSHA256: "wrong-dispatches",
            metallibSHA256: "metal",
            weightsSHA256: "weights",
            dispatchSamples: [SmeltDispatchCostSample(
                key: key,
                medianGPUUs: 1,
                p95GPUUs: 2,
                sampleCount: 20
            )]
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: fixture.records,
            calibration: calibration
        )

        let report = try model.report(context: context)
        #expect(report.summary.calibrationStatus == .artifactMismatch)
        #expect(report.summary.calibratedDispatchCount == 0)
        #expect(report.summary.calibratedMedianGPUUs == nil)
        #expect(report.summary.predictedHostRecordUs == nil)
        #expect(report.provenanceKey.contains("package:build="))
        #expect(report.provenanceKey.contains(":dispatches=dispatches"))
        #expect(report.provenanceKey.contains(":weights=weights"))
    }

    @Test("Calibration from another execution context is rejected explicitly")
    func mismatchedCalibrationContextIsRejected() throws {
        let fixture = makeFixture(includeUnknown: false)
        let calibration = SmeltDeviceCostCalibration(
            provenanceKey: "wrong-context",
            deviceName: "Apple M2 Max",
            measurementMethod: "fixture",
            context: SmeltCostModelContext(
                mode: .decode,
                sequenceLength: 1,
                position: 16
            ),
            dispatchesSHA256: "dispatches",
            metallibSHA256: "metal",
            weightsSHA256: "weights",
            hostRecordUs: 1,
            wholePlanMedianGPUUs: 10,
            dispatchSamples: []
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: fixture.records,
            calibration: calibration
        )

        let report = try model.report(context: context)
        #expect(report.summary.calibrationStatus == .contextMismatch)
        #expect(report.summary.measuredWholePlanMedianGPUUs == nil)
        #expect(report.summary.predictedHostRecordUs == nil)
        #expect(report.provenanceKey.contains("package:build="))
    }

    @Test("Guarded records keep raw one-based runtime ordinals")
    func guardedRecordsKeepRuntimeOrdinals() throws {
        let fixture = makeFixture(includeUnknown: false)
        var guarded = signedMatvecRecord()
        guarded.constantCount = 3
        var guardConstant = constant(binding: .max, value: 1)
        guardConstant.kind =
            SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
        guarded.con2 = guardConstant
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [guarded, signedMatvecRecord()]
        )

        let report = try model.report(context: context)
        #expect(report.records[0].dispatchOrdinal == 1)
        #expect(!report.records[0].executesGPU)
        #expect(report.records[1].dispatchOrdinal == 2)
        #expect(report.records[1].executesGPU)
        #expect(report.summary.dispatchCount == 1)
        #expect(report.summary.skippedDispatchCount == 1)
    }

    @Test("Deltas are deterministic and incomplete bills stay unknown")
    func deterministicDeltaPreservesUnknowns() throws {
        let complete = makeFixture(includeUnknown: false)
        let incomplete = makeFixture(includeUnknown: true)
        let baseline = try complete.model.report(context: context, planID: "baseline")
        let candidate = try incomplete.model.report(context: context, planID: "candidate")
        let delta = SmeltFrozenIRCostModel.delta(
            baseline: baseline,
            candidate: candidate
        )

        #expect(delta.dispatchCountDelta == 1)
        #expect(delta.hostRecordCountDelta == 2)
        #expect(delta.logicalReadBytesDelta == nil)
        #expect(delta.logicalWriteBytesDelta == nil)
        #expect(delta.intermediateMaterializationBytesDelta == nil)
        #expect(delta.unknowns == ["candidate has undescribed dispatches"])
        #expect(try delta.encodeJSON() == delta.encodeJSON())
        #expect(try delta.encodeJSON(prettyPrinted: false)
            == delta.encodeJSON(prettyPrinted: false))
    }

    @Test("Dynamic context extents validate position-relative buffers")
    func dynamicContextBufferBounds() throws {
        let fixture = makeFixture(includeUnknown: false)
        let runtimeContext = SmeltCostModelContext(
            mode: .decode,
            sequenceLength: 1,
            position: 16
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [applyRoPERecord(), specializedAttentionRecord()]
        )

        let report = try model.report(context: runtimeContext)
        let rope = report.records[0]
        let attention = report.records[1]
        #expect(rope.isDescribed)
        #expect(rope.resources[1].byteOffset == 256)
        #expect(rope.resources[1].logicalBytes == 16)
        #expect(attention.isDescribed)
        #expect(attention.resources[1].logicalBytes == 272)
        #expect(report.summary.descriptorCoverageFraction == 1)
        #expect(report.summary.unknownDispatches.isEmpty)
    }

    @Test("Capability names resolve specialized recurrence head topology")
    func specializedRecurrenceTopology() throws {
        let fixture = makeFixture(includeUnknown: false)
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [specializedRecurrenceRecord()]
        )

        let report = try model.report(context: context)
        let bill = try #require(report.records.first)
        #expect(bill.operationGroup == "state.deltanet_recurrence")
        #expect(bill.isDescribed)
        #expect(bill.resources[0].logicalBytes == 48 * 128 * 128 * 2)
        #expect(bill.resources[1].logicalBytes == (2 * 16 + 48) * 128 * 2)
        #expect(bill.resources[6].logicalBytes == 48 * 128 * 2)
    }

    @Test("Specialized prefill recurrence separates shape from sequence constant")
    func specializedPrefillRecurrenceTopology() throws {
        let fixture = makeFixture(includeUnknown: false)
        let prefillContext = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 1,
            position: 0
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [specializedPrefillRecurrenceRecord()]
        )

        let report = try model.report(context: prefillContext)
        let bill = try #require(report.records.first)
        #expect(bill.operationGroup == "state.deltanet_recurrence")
        #expect(bill.isDescribed)
        #expect(bill.resources[0].logicalBytes == 48 * 128 * 128 * 2)
        #expect(bill.resources[1].logicalBytes == (2 * 16 + 48) * 128 * 2)
        #expect(bill.resources[6].logicalBytes == 48 * 128 * 2)
        #expect(report.summary.descriptorCoverageFraction == 1)
        #expect(report.summary.unknownDispatches.isEmpty)
    }

    @Test("Machine-readable report is byte deterministic")
    func deterministicJSON() throws {
        let fixture = makeFixture(includeUnknown: true)
        let report = try fixture.model.report(context: context)

        #expect(try report.encodeJSON() == report.encodeJSON())
        #expect(try report.encodeJSON(prettyPrinted: false)
            == report.encodeJSON(prettyPrinted: false))
    }

    @Test("Prefill table bills B4 weight replay at exact sequence geometry")
    func prefillB4Traffic() throws {
        let fixture = makeFixture(includeUnknown: false)
        let context = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 256,
            position: 0
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [signedDirectBatchedRecord()],
            dispatchTable: .prefill
        )

        let report = try model.report(context: context)
        let bill = try #require(report.records.first)
        #expect(report.planID == "package-prefill")
        #expect(report.provenanceKey.contains("table=prefill_dispatches.bin"))
        #expect(bill.operationGroup == "signed.direct_matvec_batched")
        #expect(bill.isDescribed)
        #expect(bill.geometry?.grid == [1, 64, 1])
        #expect(bill.resources.map(\.logicalBytes) == [
            8_192, 1_024, 65_536, 4_096,
        ])
        #expect(bill.operations.first {
            $0.operationClass == .fp16Arithmetic
        }?.count == 524_288)
        let streaming = try #require(report.summary.storageTotals.first {
            $0.storageClass == .streamingWeight
        })
        #expect(streaming.readBytes == 9_216)
        #expect(report.summary.descriptorCoverageFraction == 1)
    }

    @Test("Precise ternary affine decode descriptors bill the complete ABI")
    func preciseTernaryAffineDecodeTraffic() throws {
        let fixture = makeFixture(includeUnknown: false)
        let records = [
            ternaryAffineMatvecRecord(),
            ternaryAffineMatvecAddRecord(),
            ternaryAffineBank4Record(),
            ternaryAffineGateUpRecord(),
        ]
        let report = try SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: records
        ).report(context: context)

        #expect(report.summary.descriptorCoverageFraction == 1)
        #expect(report.records.map(\.operationGroup) == [
            "signed.ternary_affine_matvec",
            "signed.ternary_affine_matvec",
            "signed.ternary_affine_bank4_matvec",
            "signed.ternary_affine_gate_up_swiglu",
        ])
        #expect(report.records[0].resources.map(\.logicalBytes)
            == [256, 16, 16, 256, 16])
        #expect(report.records[1].resources.map(\.logicalBytes)
            == [256, 16, 16, 256, 16, 16])
        #expect(report.records[2].resources.map(\.logicalBytes)
            == [256, 16, 16, 256, 4, 4, 4, 4])
        #expect(report.records[3].resources.map(\.logicalBytes)
            == [256, 16, 16, 256, 16, 16, 256, 16])
    }

    @Test("Precise ternary affine QMM bills one weight replay per BM32 batch tile")
    func preciseTernaryAffineQMMTraffic() throws {
        let fixture = makeFixture(includeUnknown: false)
        let context = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 64,
            position: 0
        )
        let report = try SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [ternaryAffineQMMRecord()],
            dispatchTable: .prefill
        ).report(context: context)
        let bill = try #require(report.records.first)

        #expect(bill.operationGroup == "signed.ternary_affine_qmm")
        #expect(bill.geometry?.grid == [1, 2, 1])
        #expect(bill.resources.map(\.logicalBytes)
            == [512, 32, 32, 16_384, 1_024])
        #expect(report.summary.descriptorCoverageFraction == 1)
    }

    @Test("Precise ternary affine QMV bills one physical weight replay per batch row")
    func preciseTernaryAffineQMVBatchTraffic() throws {
        let fixture = makeFixture(includeUnknown: false)
        let context = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 4,
            position: 0
        )
        let report = try SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [ternaryAffineMatvecBatchedRecord()],
            dispatchTable: .prefill
        ).report(context: context)
        let bill = try #require(report.records.first)

        #expect(bill.operationGroup == "signed.ternary_affine_matvec_batched")
        #expect(bill.geometry?.grid == [1, 4, 1])
        #expect(bill.resources.map(\.logicalBytes)
            == [1_024, 64, 64, 1_024, 64])
        #expect(report.summary.descriptorCoverageFraction == 1)
    }

    @Test("Batched signed view and bit-GEMM bill every prompt row")
    func batchedSignedViewAndBitGEMMTraffic() throws {
        let fixture = makeFixture(includeUnknown: false)
        let context = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 256,
            position: 0
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [batchedActivationRecord(), batchedBitGEMMRecord()],
            dispatchTable: .prefill
        )

        let report = try model.report(context: context)
        #expect(report.summary.descriptorCoverageFraction == 1)
        #expect(report.records[0].operationGroup
            == "signed.activation_bitplanes_batched")
        #expect(report.records[0].resources.map(\.logicalBytes)
            == [65_536, 16_384, 512])
        #expect(report.records[1].operationGroup
            == "signed.bitplane_matvec_batched_b4")
        #expect(report.records[1].resources.map(\.logicalBytes)
            == [8_192, 1_024, 16_384, 512, 4_096])
        let streaming = try #require(report.summary.storageTotals.first {
            $0.storageClass == .streamingWeight
        })
        #expect(streaming.readBytes == 9_216)
    }

    @Test("Fused SwiGLU has a complete traffic and operation descriptor")
    func fusedSwiGLUDescriptor() throws {
        let fixture = makeFixture(includeUnknown: false)
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [swigluRecord()]
        )
        let report = try model.report(context: SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 256,
            position: 0
        ))
        let bill = try #require(report.records.first)
        #expect(bill.operationGroup == "activation.swiglu")
        #expect(bill.resources.map(\.logicalBytes) == [4_096, 4_096, 4_096])
        #expect(bill.operations.first {
            $0.operationClass == .transcendental
        }?.count == 2_048)
        #expect(report.summary.descriptorCoverageFraction == 1)
    }

    @Test("Calibration table identity prevents cross-phase artifact matches")
    func calibrationTableIdentity() throws {
        let fixture = makeFixture(includeUnknown: false)
        let prefillContext = SmeltCostModelContext(
            mode: .prefill,
            sequenceLength: 256,
            position: 0
        )
        let calibration = SmeltDeviceCostCalibration(
            provenanceKey: "decode-only",
            deviceName: "Apple M2 Max",
            measurementMethod: "fixture",
            context: prefillContext,
            dispatchTableName: "dispatches.bin",
            dispatchesSHA256: "prefill",
            metallibSHA256: "metal",
            weightsSHA256: "weights",
            dispatchSamples: []
        )
        let model = SmeltFrozenIRCostModel(
            manifest: fixture.manifest,
            records: [signedDirectBatchedRecord()],
            calibration: calibration,
            dispatchTable: .prefill
        )
        let report = try model.report(context: prefillContext)
        #expect(report.summary.calibrationStatus == .artifactMismatch)
        #expect(report.summary.measuredWholePlanMedianGPUUs == nil)
    }

    private struct Fixture {
        let manifest: SmeltManifest
        let records: [SmeltDispatchRecord]
        let model: SmeltFrozenIRCostModel
    }

    private func makeFixture(includeUnknown: Bool) -> Fixture {
        let manifest = SmeltManifest(
            modelName: "fixture/frozen-cost",
            config: SmeltManifestConfig(
                hiddenSize: 8,
                numLayers: 1,
                vocabSize: 16,
                staticSeqCapacity: 16,
                ropeDim: 8,
                numDeltaLayers: 0,
                numAttnLayers: 1
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "weights",
                metallib: "metal",
                generatedSwift: "swift",
                dispatchesBin: "dispatches",
                prefillDispatchesBin: "prefill"
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple8,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 416, entries: [
                SmeltWeightEntry(
                    name: "projection",
                    offset: 0,
                    sizeBytes: 128,
                    shape: [8, 128],
                    dtype: .binary1,
                    groupSize: 128,
                    packedRowStride: 16,
                    paddedCols: 128,
                    scalesOffset: 128,
                    scalesSizeBytes: 16
                ),
                SmeltWeightEntry(
                    name: "ternary_projection",
                    offset: 144,
                    sizeBytes: 256,
                    shape: [8, 128],
                    dtype: .ternary2,
                    groupSize: 128,
                    packedRowStride: 32,
                    paddedCols: 128,
                    scalesOffset: 400,
                    scalesSizeBytes: 16
                ),
            ]),
            buffers: SmeltBufferTable(slots: [
                SmeltBufferSlot(
                    index: 0, name: "hiddenA", sizeBytes: 65_536,
                    dtype: .fp16, shape: [0, 128], category: .activation
                ),
                SmeltBufferSlot(
                    index: 1, name: "hiddenB", sizeBytes: 65_536,
                    dtype: .fp16, shape: [0, 8], category: .activation
                ),
                SmeltBufferSlot(
                    index: 30, name: "weights", sizeBytes: 416,
                    dtype: .raw, category: .weight
                ),
                SmeltBufferSlot(
                    index: 33, name: "activationPlanes", sizeBytes: 16_384,
                    dtype: .raw, category: .activation
                ),
                SmeltBufferSlot(
                    index: 34, name: "activationScales", sizeBytes: 512,
                    dtype: .fp16, category: .activation
                ),
                SmeltBufferSlot(
                    index: 40, name: "keyCache_0", sizeBytes: 16,
                    dtype: .fp16, shape: [1, 0, 8], category: .state
                ),
                SmeltBufferSlot(
                    index: 41, name: "valCache_0", sizeBytes: 16,
                    dtype: .fp16, shape: [1, 0, 8], category: .state
                ),
                SmeltBufferSlot(
                    index: 42, name: "ropeCos", sizeBytes: 16,
                    dtype: .fp16, shape: [0, 8], category: .table
                ),
                SmeltBufferSlot(
                    index: 43, name: "ropeSin", sizeBytes: 16,
                    dtype: .fp16, shape: [0, 8], category: .table
                ),
                SmeltBufferSlot(
                    index: 44, name: "recState_0", sizeBytes: 48 * 128 * 128 * 2,
                    dtype: .fp16, category: .state
                ),
                SmeltBufferSlot(
                    index: 45, name: "deltaQKV", sizeBytes: (2 * 16 + 48) * 128 * 2,
                    dtype: .fp16, category: .activation
                ),
                SmeltBufferSlot(
                    index: 46, name: "deltaB", sizeBytes: 48 * 2,
                    dtype: .fp16, category: .activation
                ),
                SmeltBufferSlot(
                    index: 47, name: "deltaA", sizeBytes: 48 * 2,
                    dtype: .fp16, category: .activation
                ),
                SmeltBufferSlot(
                    index: 48, name: "aLog", sizeBytes: 48 * 2,
                    dtype: .fp16, category: .weight
                ),
                SmeltBufferSlot(
                    index: 49, name: "dtBias", sizeBytes: 48 * 2,
                    dtype: .fp16, category: .weight
                ),
                SmeltBufferSlot(
                    index: 50, name: "recOut", sizeBytes: 48 * 128 * 2,
                    dtype: .fp16, category: .activation
                ),
            ]),
            pipelines: [
                "signed_binary_bitplane_i4_matvec_g128_rows8",
                "new_kernel",
                "apply_rope",
                "attention_decode_d8_h1_kv1",
                "deltanet_recurrence_mlx_decode_d128_h48_qk16",
                "signed_binary_matvec_g128_rows8_batched_b4",
                "signed_activation_bitplanes_i4_g128_batched",
                "signed_binary_bitplane_i4_matvec_g128_rows8_batched_b4",
                "swiglu_fused",
                "deltanet_recurrence_mlx_prefill_d128_h48_qk16",
                "signed_ternary_affine_matvec_g128_rows8",
                "signed_ternary_affine_matvec_add_g128_rows8",
                "signed_ternary_affine_bank4_matvec_g128_rows8",
                "signed_ternary_affine_gate_up_swiglu_g128_rows8",
                "signed_ternary_affine_qmm_g128_bm32_bn32_bk32",
            ],
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 40,
                recStateBaseSlot: 40,
                keyCacheBaseSlot: 40,
                valCacheBaseSlot: 40,
                ropeCosSlot: 40,
                ropeSinSlot: 40,
                tokenIdSlot: 41,
                positionSlot: 42,
                weightsSlot: 30
            )
        )
        var records = [signedMatvecRecord()]
        if includeUnknown {
            records.append(.swap())
            records.append(unknownRecord())
        }
        return Fixture(
            manifest: manifest,
            records: records,
            model: SmeltFrozenIRCostModel(manifest: manifest, records: records)
        )
    }

    private func signedMatvecRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 0
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 5
        record.buf0 = buffer(slot: 30, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 128)
        record.buf2 = buffer(slot: 33, binding: 2, offset: 0)
        record.buf3 = buffer(slot: 34, binding: 3, offset: 0)
        record.buf4 = buffer(
            slot: SmeltBufferRecord.slotCur,
            binding: 4,
            offset: 0
        )
        record.constantCount = 2
        record.con0 = constant(binding: 5, value: 8)
        record.con1 = constant(binding: 6, value: 128)
        return record
    }

    private func unknownRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 1
        record.dispatchStyle = SmeltDispatchRecord.styleThreads
        record.gridW = 8
        record.gridH = 1
        record.gridD = 1
        record.tgW = 8
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 1
        record.buf0 = buffer(
            slot: SmeltBufferRecord.slotAlt,
            binding: 0,
            offset: 0
        )
        return record
    }

    private func signedDirectBatchedRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 5
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 4
        record.gridHKind = SmeltDispatchRecord.gridSeqLenCeilDivLiteral
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 4
        record.buf0 = buffer(slot: 30, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 128)
        record.buf2 = buffer(slot: 0, binding: 2, offset: 0)
        record.buf3 = buffer(slot: 1, binding: 3, offset: 0)
        record.constantCount = 3
        record.con0 = constant(binding: 4, value: 8)
        record.con1 = constant(binding: 5, value: 128)
        var batch = constant(binding: 6, value: 0)
        batch.kind = SmeltConstantRecord.kindSeqLen
        record.con2 = batch
        return record
    }

    private func batchedActivationRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 6
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridHKind = SmeltDispatchRecord.gridSeqLen
        record.gridD = 1
        record.tgW = 32
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 3
        record.buf0 = buffer(slot: 0, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 33, binding: 1, offset: 0)
        record.buf2 = buffer(slot: 34, binding: 2, offset: 0)
        record.constantCount = 1
        record.con0 = constant(binding: 3, value: 128)
        return record
    }

    private func batchedBitGEMMRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 7
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 4
        record.gridHKind = SmeltDispatchRecord.gridSeqLenCeilDivLiteral
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 5
        record.buf0 = buffer(slot: 30, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 128)
        record.buf2 = buffer(slot: 33, binding: 2, offset: 0)
        record.buf3 = buffer(slot: 34, binding: 3, offset: 0)
        record.buf4 = buffer(slot: 1, binding: 4, offset: 0)
        record.constantCount = 3
        record.con0 = constant(binding: 5, value: 8)
        record.con1 = constant(binding: 6, value: 128)
        var batch = constant(binding: 7, value: 0)
        batch.kind = SmeltConstantRecord.kindSeqLen
        record.con2 = batch
        return record
    }

    private func swigluRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 8
        record.dispatchStyle = SmeltDispatchRecord.styleThreads
        record.gridW = 2_048
        record.gridH = 1
        record.gridD = 1
        record.tgW = 256
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 3
        record.buf0 = buffer(slot: 0, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 1, binding: 1, offset: 0)
        record.buf2 = buffer(slot: 0, binding: 2, offset: 0)
        record.constantCount = 1
        record.con0 = constant(binding: 3, value: 2_048)
        return record
    }

    private func applyRoPERecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 2
        record.dispatchStyle = SmeltDispatchRecord.styleThreads
        record.gridW = 8
        record.gridH = 1
        record.gridD = 1
        record.tgW = 8
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 3
        record.buf0 = buffer(slot: 0, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 42, binding: 1, offset: 16)
        record.buf1.offsetKind = 1
        record.buf2 = buffer(slot: 43, binding: 2, offset: 16)
        record.buf2.offsetKind = 1
        record.constantCount = 3
        record.con0 = constant(binding: 3, value: 8)
        record.con1 = constant(binding: 4, value: 8)
        record.con2 = constant(binding: 5, value: 1)
        return record
    }

    private func specializedAttentionRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 3
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 32
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 3
        record.buf0 = buffer(slot: 0, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 40, binding: 1, offset: 0)
        record.buf2 = buffer(slot: 41, binding: 2, offset: 0)
        record.constantCount = 1
        var sequence = constant(binding: 3, value: 0)
        sequence.kind = SmeltConstantRecord.kindPositionPlus1
        record.con0 = sequence
        return record
    }

    private func specializedRecurrenceRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 4
        record.dispatchStyle = SmeltDispatchRecord.styleThreads
        record.gridW = 32
        record.gridH = 128
        record.gridD = 48
        record.tgW = 32
        record.tgH = 4
        record.tgD = 1
        record.bufferCount = 7
        record.buf0 = buffer(slot: 44, binding: 0, offset: 0)
        record.buf1 = buffer(slot: 45, binding: 1, offset: 0)
        record.buf2 = buffer(slot: 46, binding: 2, offset: 0)
        record.buf3 = buffer(slot: 47, binding: 3, offset: 0)
        record.buf4 = buffer(slot: 48, binding: 4, offset: 0)
        record.buf5 = buffer(slot: 49, binding: 5, offset: 0)
        record.buf6 = buffer(slot: 50, binding: 6, offset: 0)
        return record
    }

    private func specializedPrefillRecurrenceRecord() -> SmeltDispatchRecord {
        var record = specializedRecurrenceRecord()
        record.pipeline = 9
        record.constantCount = 1
        record.con0 = constant(binding: 7, value: 1)
        return record
    }

    private func ternaryAffineMatvecRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 10
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 5
        record.buf0 = buffer(slot: 30, binding: 0, offset: 144)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 400)
        record.buf2 = buffer(slot: 30, binding: 2, offset: 400)
        record.buf3 = buffer(slot: 0, binding: 3, offset: 0)
        record.buf4 = buffer(slot: 1, binding: 4, offset: 0)
        record.constantCount = 2
        record.con0 = constant(binding: 5, value: 8)
        record.con1 = constant(binding: 6, value: 128)
        return record
    }

    private func ternaryAffineMatvecAddRecord() -> SmeltDispatchRecord {
        var record = ternaryAffineMatvecRecord()
        record.pipeline = 11
        record.bufferCount = 6
        record.buf5 = buffer(slot: 0, binding: 5, offset: 0)
        record.con0 = constant(binding: 6, value: 8)
        record.con1 = constant(binding: 7, value: 128)
        return record
    }

    private func ternaryAffineBank4Record() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 12
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 8
        record.buf0 = buffer(slot: 30, binding: 0, offset: 144)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 400)
        record.buf2 = buffer(slot: 30, binding: 2, offset: 400)
        record.buf3 = buffer(slot: 0, binding: 3, offset: 0)
        record.buf4 = buffer(slot: 1, binding: 4, offset: 0)
        record.buf5 = buffer(slot: 1, binding: 5, offset: 0)
        record.buf6 = buffer(slot: 1, binding: 6, offset: 0)
        record.buf7 = buffer(slot: 1, binding: 7, offset: 0)
        record.constantCount = 5
        record.con0 = constant(binding: 8, value: 2)
        record.con1 = constant(binding: 9, value: 2)
        record.con2 = constant(binding: 10, value: 2)
        record.con3 = constant(binding: 11, value: 2)
        record.con4 = constant(binding: 12, value: 128)
        return record
    }

    private func ternaryAffineGateUpRecord() -> SmeltDispatchRecord {
        var record = SmeltDispatchRecord.empty()
        record.opKind = SmeltDispatchRecord.opDispatch
        record.pipeline = 13
        record.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        record.gridW = 1
        record.gridH = 1
        record.gridD = 1
        record.tgW = 64
        record.tgH = 1
        record.tgD = 1
        record.bufferCount = 8
        record.buf0 = buffer(slot: 30, binding: 0, offset: 144)
        record.buf1 = buffer(slot: 30, binding: 1, offset: 400)
        record.buf2 = buffer(slot: 30, binding: 2, offset: 400)
        record.buf3 = buffer(slot: 30, binding: 3, offset: 144)
        record.buf4 = buffer(slot: 30, binding: 4, offset: 400)
        record.buf5 = buffer(slot: 30, binding: 5, offset: 400)
        record.buf6 = buffer(slot: 0, binding: 6, offset: 0)
        record.buf7 = buffer(slot: 1, binding: 7, offset: 0)
        record.constantCount = 2
        record.con0 = constant(binding: 8, value: 8)
        record.con1 = constant(binding: 9, value: 128)
        return record
    }

    private func ternaryAffineQMMRecord() -> SmeltDispatchRecord {
        var record = ternaryAffineMatvecRecord()
        record.pipeline = 14
        record.gridH = 32
        record.gridHKind = SmeltDispatchRecord.gridSeqLenCeilDivLiteral
        record.tgW = 128
        record.constantCount = 3
        var batch = constant(binding: 7, value: 0)
        batch.kind = SmeltConstantRecord.kindSeqLen
        record.con2 = batch
        return record
    }

    private func ternaryAffineMatvecBatchedRecord() -> SmeltDispatchRecord {
        var record = ternaryAffineMatvecRecord()
        record.gridHKind = SmeltDispatchRecord.gridSeqLen
        return record
    }

    private func buffer(
        slot: Int16,
        binding: UInt8,
        offset: UInt64
    ) -> SmeltBufferRecord {
        var record = SmeltBufferRecord.empty()
        record.slot = slot
        record.bindingIndex = binding
        record.offsetKind = 0
        record.offset = offset
        return record
    }

    private func constant(binding: UInt8, value: UInt32) -> SmeltConstantRecord {
        var record = SmeltConstantRecord.empty()
        record.kind = SmeltConstantRecord.kindLiteralU32
        record.bindingIndex = binding
        record.value = value
        return record
    }
}
