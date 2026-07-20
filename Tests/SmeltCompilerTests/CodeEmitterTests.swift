import Foundation
import Testing
@testable import SmeltCompiler

// MARK: - Binding validation

@Test func emitRejectsWrongBufferCount() {
    var emitter = SmeltCodeEmitter()
    // rmsNorm1PW expects 3 buffers, 2 constants
    let badDispatch = SmeltDispatch(
        pipeline: .rmsNorm1PW,
        buffers: [
            SmeltBufferBinding(slot: 0, index: 0),
            SmeltBufferBinding(slot: 1, index: 1),
            // missing third buffer
        ],
        constants: [
            SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
            SmeltConstantBinding(expression: "1e-6", type: .float32, index: 4),
        ],
        dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
    )
    #expect(throws: SmeltEmitError.self) {
        try emitter.emit(badDispatch)
    }
}

@Test func emitRejectsWrongConstantCount() {
    var emitter = SmeltCodeEmitter()
    let badDispatch = SmeltDispatch(
        pipeline: .elementwiseAdd,
        buffers: [
            SmeltBufferBinding(slot: 0, index: 0),
            SmeltBufferBinding(slot: 1, index: 1),
            SmeltBufferBinding(slot: 2, index: 2),
        ],
        constants: [],  // expects 1
        dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
    )
    #expect(throws: SmeltEmitError.self) {
        try emitter.emit(badDispatch)
    }
}

@Test func emitSignedMatvecAutomaticallyTilesBatchShape() throws {
    let entry = SmeltWeightEntry(
        name: "signed.weight",
        offset: 0,
        sizeBytes: 9 * 256 / 8,
        shape: [9, 256],
        dtype: .binary1,
        groupSize: 128,
        packedRowStride: 32,
        paddedCols: 256,
        scalesOffset: 512,
        scalesSizeBytes: 9 * 2 * 2
    )
    var decode = SmeltCodeEmitter()
    _ = try decode.emitSignedMatvec(
        weightEntry: entry,
        weightsSlot: 30,
        inputBinding: SmeltBufferBinding(slot: 7, index: 2),
        outputBinding: SmeltBufferBinding(slot: 8, index: 3),
        rows: 9,
        cols: 256
    )
    #expect(
        decode.dispatchRecords.last?.pipeline
            == UInt16(SmeltPipeline.signedBinaryMatvecG128Rows8.rawValue)
    )

    var prefill = SmeltCodeEmitter()
    _ = try prefill.emitSignedMatvec(
        weightEntry: entry,
        weightsSlot: 30,
        inputBinding: SmeltBufferBinding(slot: 7, index: 2),
        outputBinding: SmeltBufferBinding(slot: 8, index: 3),
        rows: 9,
        cols: 256,
        batchSize: 8,
        dynamicGridH: .seqLen
    )
    let record = try #require(prefill.dispatchRecords.last)
    #expect(
        record.pipeline
            == UInt16(SmeltPipeline.signedBinaryMatvecG128Rows8BatchedB4.rawValue)
    )
    #expect(record.constantCount == 3)
    #expect(record.gridHKind == SmeltDispatchRecord.gridSeqLenCeilDivLiteral)
    #expect(record.gridH == 4)
}

@Test func emitTernaryDecodeUsesReferenceCompatibleAffineRoute() throws {
    let entry = SmeltWeightEntry(
        name: "signed.weight",
        offset: 0,
        sizeBytes: 9 * 256 / 4,
        shape: [9, 256],
        dtype: .ternary2,
        groupSize: 128,
        packedRowStride: 64,
        paddedCols: 256,
        scalesOffset: 640,
        scalesSizeBytes: 9 * 2 * 2
    )
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitSignedMatvec(
        weightEntry: entry,
        weightsSlot: 30,
        inputBinding: SmeltBufferBinding(slot: 7, index: 2),
        outputBinding: SmeltBufferBinding(slot: 8, index: 3),
        rows: 9,
        cols: 256
    )
    let record = try #require(emitter.dispatchRecords.last)
    #expect(
        record.pipeline
            == UInt16(SmeltPipeline.signedTernaryAffineMatvecG128Rows8.rawValue)
    )
    #expect(record.bufferCount == 5)
    #expect(record.constantCount == 2)
    #expect(record.gridW == 2)
}

@Test func emitTernaryPrefillSelectsMLXQMVOrQMMByBatchShape() throws {
    let entry = SmeltWeightEntry(
        name: "signed.weight",
        offset: 0,
        sizeBytes: 9 * 512 / 4,
        shape: [9, 512],
        dtype: .ternary2,
        groupSize: 128,
        packedRowStride: 128,
        paddedCols: 512,
        scalesOffset: 1_152,
        scalesSizeBytes: 9 * 4 * 2
    )
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitSignedMatvec(
        weightEntry: entry,
        weightsSlot: 30,
        inputBinding: SmeltBufferBinding(slot: 7, index: 2),
        outputBinding: SmeltBufferBinding(slot: 8, index: 3),
        rows: 9,
        cols: 512,
        batchSize: 16,
        dynamicGridH: .seqLen
    )
    let records = Array(emitter.dispatchRecords.suffix(2))
    #expect(records.count == 2)
    let qmv = records[0]
    #expect(
        qmv.pipeline
            == UInt16(SmeltPipeline.signedTernaryAffineMatvecG128Rows8.rawValue)
    )
    #expect(qmv.bufferCount == 5)
    #expect(qmv.constantCount == 3)
    #expect(qmv.gridHKind == SmeltDispatchRecord.gridSeqLen)
    #expect(qmv.minSeqLen == 0)
    #expect(
        qmv.con2.kind
            == SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse
    )
    #expect(qmv.con2.value == 6)

    let qmm = records[1]
    #expect(
        qmm.pipeline
            == UInt16(
                SmeltPipeline.signedTernaryAffineQMMG128BM32BN32BK32.rawValue
            )
    )
    #expect(qmm.bufferCount == 5)
    #expect(qmm.constantCount == 3)
    #expect(qmm.gridHKind == SmeltDispatchRecord.gridSeqLenCeilDivLiteral)
    #expect(qmm.gridH == 32)
    #expect(qmm.minSeqLen == 6)
}

@Test func emitTernaryGateUpUsesExactGenericFusedBrick() throws {
    func entry(_ name: String, offset: UInt64) -> SmeltWeightEntry {
        SmeltWeightEntry(
            name: name,
            offset: offset,
            sizeBytes: 9 * 512 / 4,
            shape: [9, 512],
            dtype: .ternary2,
            groupSize: 128,
            packedRowStride: 128,
            paddedCols: 512,
            scalesOffset: offset + UInt64(9 * 512 / 4),
            scalesSizeBytes: 9 * 4 * 2
        )
    }
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitSignedTernaryAffineGateUpSwiglu(
        gateEntry: entry("gate", offset: 0),
        upEntry: entry("up", offset: 4_096),
        weightsSlot: 30,
        inputBinding: SmeltBufferBinding(slot: 8, index: 6),
        outputBinding: SmeltBufferBinding(slot: 11, index: 7),
        rows: 9,
        cols: 512
    )
    let record = try #require(emitter.dispatchRecords.last)
    #expect(
        record.pipeline == UInt16(
            SmeltPipeline.signedTernaryAffineGateUpSwigluG128Rows8.rawValue)
    )
    #expect(record.bufferCount == 8)
    #expect(record.constantCount == 2)
    #expect(record.gridW == 2)
}

@Test func emitSignedProjectionBankBatchedOwnsOneReusableActivationView() throws {
    func entry(_ name: String, rows: Int, offset: UInt64) -> SmeltWeightEntry {
        SmeltWeightEntry(
            name: name,
            offset: offset,
            sizeBytes: UInt64(rows * 256 / 8),
            shape: [rows, 256],
            dtype: .binary1,
            groupSize: 128,
            packedRowStride: 32,
            paddedCols: 256,
            scalesOffset: offset + UInt64(rows * 256 / 8),
            scalesSizeBytes: UInt64(rows * 2 * 2)
        )
    }
    var emitter = SmeltCodeEmitter()
    let emitted = try emitter.emitSignedBitplaneProjectionBankBatchedIfPossible(
        view: .signedBitplanesI4,
        members: [
            (entry("gate", rows: 9, offset: 0), 11, 9),
            (entry("up", rows: 13, offset: 1_024), 12, 13),
        ],
        weightsSlot: 30,
        inputSlot: 8,
        planesSlot: 33,
        activationScalesSlot: 34,
        cols: 256,
        batchSize: 256
    )
    let lines = try #require(emitted)

    #expect(!lines.isEmpty)
    #expect(emitter.dispatchRecords.count == 3)
    #expect(
        emitter.dispatchRecords[0].pipeline
            == UInt16(SmeltPipeline.signedActivationBitplanesI4G128Batched.rawValue)
    )
    #expect(emitter.dispatchRecords[0].gridHKind == SmeltDispatchRecord.gridSeqLen)
    for record in emitter.dispatchRecords.dropFirst() {
        #expect(
            record.pipeline
                == UInt16(
                    SmeltPipeline.signedBinaryBitplaneI4MatvecG128Rows8BatchedB4.rawValue
                )
        )
        #expect(record.gridHKind == SmeltDispatchRecord.gridSeqLenCeilDivLiteral)
        #expect(record.gridH == 4)
    }
}

// MARK: - Correct emission

@Test func emitRMSNormProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let lines = try emitter.emitRMSNorm1PW(
        inputSlot: 0,
        weightSlot: 30, weightOffset: 1_048_576,
        outputSlot: 8,
        dim: 2_048, eps: 1e-6,
        comment: "Input layernorm"
    )

    // Specialized d2048 path: comment(1) + setPipeline(1) + 3 setBuffers(3) + dispatch(1) = 6
    #expect(lines.count == 6)
    #expect(lines[0].contains("// Input layernorm"))
    #expect(lines[1].contains("setComputePipelineState(p[\(SmeltPipeline.rmsNorm1PWD2048.rawValue)])"))
    #expect(lines[2].contains("b[0]"))
    #expect(lines[3].contains("b[30]") && lines[3].contains("1048576"))
    #expect(lines[4].contains("b[8]"))
    #expect(lines[5].contains("dispatchThreadgroups"))
    #expect(!lines.joined(separator: "\n").contains("setBytes"))
}

@Test func emitGeneratedPipelineNameAppendsManifestEntry() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let generatedName = "generated_elementwise_add_test"
    let lines = try emitter.emit(SmeltDispatch(
        pipeline: .elementwiseAdd,
        pipelineNameOverride: generatedName,
        buffers: [
            SmeltBufferBinding(slot: 0, index: 0),
            SmeltBufferBinding(slot: 1, index: 1),
            SmeltBufferBinding(slot: 2, index: 2),
        ],
        constants: [
            SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
        ],
        dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
    ))

    let generatedIndex = SmeltKernelCatalog.pipelineNames.count
    #expect(emitter.buildPipelineNames().last == generatedName)
    #expect(lines.contains { $0.contains("setComputePipelineState(p[\(generatedIndex)])") })
    #expect(emitter.dispatchRecords.last?.pipeline == UInt16(generatedIndex))
    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(name: generatedName, plannedKernelCandidate: nil),
    ])

    emitter.optimizeAndRebuildRecords()
    #expect(emitter.buildPipelineNames().last == generatedName)
    #expect(emitter.dispatchRecords.last?.pipeline == UInt16(generatedIndex))
    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(name: generatedName, plannedKernelCandidate: nil),
    ])
}

@Test func emitCatalogPipelineNameOverrideUsesCatalogEntry() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let catalogName = SmeltKernelCatalog.signature(for: .elementwiseAdd).metalFunctionName
    let catalogIndex = try #require(SmeltKernelCatalog.pipelineIndex(named: catalogName))

    let lines = try emitter.emit(SmeltDispatch(
        pipeline: .elementwiseAdd,
        pipelineNameOverride: catalogName,
        buffers: [
            SmeltBufferBinding(slot: 0, index: 0),
            SmeltBufferBinding(slot: 1, index: 1),
            SmeltBufferBinding(slot: 2, index: 2),
        ],
        constants: [
            SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
        ],
        dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
    ))

    #expect(emitter.namedPipelines.isEmpty)
    #expect(emitter.buildPipelineNames() == SmeltKernelCatalog.pipelineNames)
    #expect(lines.contains { $0.contains("setComputePipelineState(p[\(catalogIndex)])") })
    #expect(emitter.dispatchRecords.last?.pipeline == UInt16(catalogIndex))
    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(name: catalogName, plannedKernelCandidate: nil),
    ])

    emitter.optimizeAndRebuildRecords()
    #expect(emitter.namedPipelines.isEmpty)
    #expect(emitter.buildPipelineNames() == SmeltKernelCatalog.pipelineNames)
    #expect(emitter.dispatchRecords.last?.pipeline == UInt16(catalogIndex))
    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(name: catalogName, plannedKernelCandidate: nil),
    ])
}

@Test func emitPlannedKernelRouteRecordsPipelineNameAndCandidateTogether() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let candidate = SmeltPlannedKernelCandidate(
        consumerID: "test.down.prefill",
        operation: .affineMatvecPrefillFull,
        shape: SmeltKernelShape(rows: 256, cols: 1024, groupSize: 128),
        weights: [
            SmeltPlannedKernelWeight(
                weightName: "test_down_weight",
                role: .affine
            ),
        ],
        kind: .ffnDownPrefill
    )
    let capability = try #require(SmeltKernelCapabilityRegistry.generatedCapability(
        operation: candidate.operation,
        shape: candidate.shape
    ))
    let route = SmeltPlannedKernelRoute(candidate: candidate, capability: capability)

    _ = try emitter.emit(SmeltDispatch(
        pipeline: .affineMatvec,
        plannedKernelRoute: route,
        buffers: [
            SmeltBufferBinding(slot: 0, index: 0),
            SmeltBufferBinding(slot: 0, index: 1),
            SmeltBufferBinding(slot: 0, index: 2),
            SmeltBufferBinding(slot: 1, index: 3),
            SmeltBufferBinding(slot: 2, index: 4),
        ],
        constants: [
            SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
        ],
        dispatch: .threadgroups(
            width: 8,
            height: 1,
            depth: 1,
            tgWidth: 128,
            tgHeight: 1,
            tgDepth: 1
        )
    ))

    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(
            name: capability.id,
            plannedKernelCandidate: candidate
        ),
    ])

    emitter.optimizeAndRebuildRecords()
    #expect(emitter.namedPipelineUses == [
        SmeltNamedPipelineUse(
            name: capability.id,
            plannedKernelCandidate: candidate
        ),
    ])
}

@Test func emitFusedLUTMatvecProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let lines = try emitter.emitFusedLUTMatvec(
        indicesSlot: 30, indicesOffset: 2_097_152,
        lutSlot: 30, lutOffset: 8_388_608,
        inputSlot: 8, outputSlot: 2,
        rows: 6_144, cols: 2_048, groupSize: 16,
        comment: "QKV projection"
    )

    // comment(1) + setPipeline(1) + 4 setBuffers(4) + 1 constant×2(2) + dispatch(1) = 9
    #expect(lines.count == 9)
    #expect(lines[0].contains("// QKV projection"))
    // Pipeline index is now a specialized FC variant (first specialized = base count)
    #expect(lines[1].contains("setComputePipelineState"))
    #expect(lines[2].contains("offset: 2097152"))  // indices offset
    #expect(lines[3].contains("offset: 8388608"))  // LUT offset
    // Only 1 constant now (numRows at index 4), cols/groupSize are function constants
    #expect(lines[6].contains("UInt32 = 6144"))  // numRows
    // Dispatch threadgroups: width=ceil(rows/8)
    // Tiled dispatch: (rows + 7) / 8 threadgroups, 64 threads
    #expect(lines.last?.contains("width: \((6144 + 7) / 8)") == true)

    // Verify specialized pipeline was registered
    let pipelineNames = emitter.buildPipelineNames()
    #expect(pipelineNames.contains("fused_lut_matvec:2048:16"))
}

@Test func emitElementwiseAddProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitElementwiseAdd(
        inputASlot: 0, inputBSlot: 8, outputSlot: 1,
        count: 2_048
    )

    #expect(lines[0].contains("p[12]"))  // elementwiseAdd
    // dispatchThreads (not threadgroups)
    #expect(lines.last?.contains("dispatchThreads") == true)
}

@Test func emitElementwiseAddWithOffsetsProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitElementwiseAddWithOffsets(
        inputASlot: 4, inputAOffset: 512,
        inputBSlot: 30, inputBOffset: 4_096,
        outputSlot: 4, outputOffset: 512,
        count: 256,
        comment: "Projection bias",
        minSeqLen: 2
    )

    let joined = lines.joined(separator: "\n")
    #expect(lines[0].contains("// Projection bias"))
    #expect(joined.contains("b[4]") && joined.contains("offset: 512"))
    #expect(joined.contains("b[30]") && joined.contains("offset: 4096"))
    #expect(lines.last?.contains("dispatchThreads") == true)
    #expect(emitter.dispatchRecords.last?.minSeqLen == 2)
}

@Test func emitBatchedBiasAddUsesDynamicSeqLenHeight() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitBatchedBiasAdd(
        inputSlot: 4,
        biasSlot: 30, biasOffset: 4_096,
        outputSlot: 4,
        rows: 256,
        batchSize: 64,
        comment: "Projection bias batched"
    )

    let joined = lines.joined(separator: "\n")
    #expect(lines[0].contains("// Projection bias batched"))
    #expect(joined.contains("p[\(SmeltPipeline.projectionBiasAddBatched.rawValue)]"))
    #expect(joined.contains("b[30]") && joined.contains("offset: 4096"))
    #expect(lines.last?.contains("dispatchThreads") == true)

    let record = try #require(emitter.dispatchRecords.last)
    #expect(record.pipeline == UInt16(SmeltPipeline.projectionBiasAddBatched.rawValue))
    #expect(record.gridH == 0)
    #expect(record.gridHKind == SmeltDispatchRecord.gridSeqLen)
    #expect(record.minSeqLen == 0)
}

@Test func emitEmbeddingGatherProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitEmbeddingGather(
        weightSlot: 30, weightOffset: 0,
        tokenIdSlot: 95, outputSlot: 0,
        hiddenSize: 2_048,
        comment: "Embedding lookup"
    )

    #expect(lines[0].contains("// Embedding lookup"))
    #expect(lines[1].contains("p[13]"))  // embeddingGather
}

@Test func emitArgmaxUsesSplitReductionForLargeVocab() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitArgmax(
        inputSlot: 16, outputSlot: 17,
        count: 248_320
    )

    #expect(lines.contains { $0.contains("p[410]") })
    #expect(lines.contains { $0.contains("p[411]") })
    #expect(lines.contains { $0.contains("b[28]") })
    #expect(lines.last?.contains("dispatchThreadgroups") == true)
}

@Test func emitArgmaxKeepsSingleThreadgroupForSmallVocab() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitArgmax(
        inputSlot: 16, outputSlot: 17,
        count: 4_096
    )

    #expect(lines.contains { $0.contains("p[14]") })
    #expect(!lines.contains { $0.contains("p[410]") })
    #expect(lines.last?.contains("width: 1") == true)
}

@Test func emitGeGLUProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitGeGLU(
        gateSlot: 11,
        upSlot: 12,
        outputSlot: 13,
        count: 24_576
    )

    #expect(lines[0].contains("p[\(SmeltPipeline.gegluFused.rawValue)]"))
    #expect(lines.last?.contains("dispatchThreads") == true)
}

@Test func emitLogitCapProducesCorrectLines() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitLogitCap(
        inputSlot: 16,
        outputSlot: 16,
        count: 262_144,
        cap: 30,
        comment: "Logit capping"
    )

    #expect(lines[0].contains("// Logit capping"))
    #expect(lines[1].contains("p[\(SmeltPipeline.logitCap.rawValue)]"))
    #expect(lines.joined(separator: "\n").contains("Float = 30.0"))
}

// MARK: - Generated code quality

@Test func emittedCodeUsesOnlyIntegerIndices() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitRMSNorm1PW(
        inputSlot: 0, weightSlot: 30, weightOffset: 0,
        outputSlot: 8, dim: 2_048, eps: 1e-6
    )

    for line in lines {
        // No string lookups, no dictionary access, no quotes (except in var declarations)
        #expect(!line.contains("[\""))
        #expect(!line.contains("dictionary"))
        #expect(!line.contains("Dictionary"))
    }
}

@Test func emitterIndentIsRespected() throws {
    var emitter8 = SmeltCodeEmitter(indent: 8)
    let lines = try emitter8.emitElementwiseAdd(
        inputASlot: 0, inputBSlot: 1, outputSlot: 2, count: 128
    )
    for line in lines where !line.isEmpty {
        #expect(line.hasPrefix("        "))  // 8 spaces
    }
}

@Test func multiDispatchUniqueVarNames() throws {
    var emitter = SmeltCodeEmitter()
    // Emit two dispatches that both use constant index 3
    let lines1 = try emitter.emitRMSNorm1PW(
        inputSlot: 0, weightSlot: 30, weightOffset: 0,
        outputSlot: 8, dim: 2_048, eps: 1e-6
    )
    let lines2 = try emitter.emitRMSNorm1PW(
        inputSlot: 1, weightSlot: 30, weightOffset: 4_096,
        outputSlot: 8, dim: 2_048, eps: 1e-6
    )
    // First dispatch uses _d0c3, _d0c4; second uses _d1c3, _d1c4
    let allLines = lines1 + lines2
    let varDecls = allLines.filter { $0.contains("var _d") }
    let varNames = varDecls.compactMap { line -> String? in
        guard let range = line.range(of: "_d\\d+c\\d+", options: .regularExpression) else {
            return nil
        }
        return String(line[range])
    }
    // All var names must be unique
    #expect(Set(varNames).count == varNames.count, "Duplicate var names: \(varNames)")
    // Should have _d0c3, _d0c4, _d1c3, _d1c4
    #expect(varNames.isEmpty)
}

@Test func emitGenericRMSNormStillUsesDynamicConstants() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let lines = try emitter.emitRMSNorm1PW(
        inputSlot: 0,
        weightSlot: 30, weightOffset: 0,
        outputSlot: 8,
        dim: 4096, eps: 1e-5
    )

    #expect(lines[0].contains("setComputePipelineState(p[\(SmeltPipeline.rmsNorm1PW.rawValue)])"))
    #expect(lines.joined(separator: "\n").contains("setBytes"))
}

@Test func emitQwenSpecializedAffineMatvecsProduceNoSetBytes() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let cases: [(rows: Int, cols: Int, pipeline: SmeltPipeline)] = [
        (2_048, 2_048, .affineMatvecC2048R2048G64),
        (6_144, 2_048, .affineMatvecC2048R6144G64),
        (4_096, 2_048, .affineMatvecC2048R4096G64),
        (512, 2_048, .affineMatvecC2048R512G64),
        (2_048, 6_144, .affineMatvecC6144R2048G64),
        (248_320, 2_048, .affineMatvecC2048R248320G64),
    ]

    for (rows, cols, pipeline) in cases {
        let lines = try emitter.emitAffineMatvec(
            weightsSlot: 30, weightsOffset: 0,
            scalesSlot: 30, scalesOffset: 1_024,
            biasesSlot: 30, biasesOffset: 2_048,
            inputSlot: 8, outputSlot: 2,
            rows: rows, cols: cols, groupSize: 64
        )

        #expect(lines[0].contains("setComputePipelineState(p[\(pipeline.rawValue)])"))
        #expect(!lines.joined(separator: "\n").contains("setBytes"))
        let expectedWidth = pipeline == .affineMatvecC2048R2048G64 ? (rows + 3) / 4 : (rows + 7) / 8
        #expect(lines.joined(separator: "\n").contains("dispatchThreadgroups(MTLSize(width: \(expectedWidth)"))
    }
}

@Test func emitQwen0808DecodeAffineMatvecsUseSpecializedGeometry() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let cases: [(rows: Int, cols: Int, pipeline: SmeltPipeline)] = [
        (2_048, 1_024, .affineMatvecC1024R2048G64Rows4),
        (6_144, 1_024, .affineMatvecC1024R6144G64Rows4),
        (1_024, 2_048, .affineMatvecC2048R1024G64Rows4),
        (1_024, 3_584, .affineMatvecC3584R1024G64Rows4),
        (248_320, 1_024, .affineMatvecC1024R248320G64Rows4),
    ]

    for (rows, cols, pipeline) in cases {
        let lines = try emitter.emitAffineMatvec(
            weightsSlot: 30, weightsOffset: 0,
            scalesSlot: 30, scalesOffset: 1_024,
            biasesSlot: 30, biasesOffset: 2_048,
            inputSlot: 8, outputSlot: 2,
            rows: rows, cols: cols, groupSize: 64
        )

        #expect(lines[0].contains("setComputePipelineState(p[\(pipeline.rawValue)])"))
        #expect(!lines.joined(separator: "\n").contains("setBytes"))
        let expectedWidth = switch pipeline {
        case .affineMatvecC1024R2048G64Rows4,
             .affineMatvecC1024R6144G64Rows4,
             .affineMatvecC2048R1024G64Rows4,
             .affineMatvecC3584R1024G64Rows4,
             .affineMatvecC1024R248320G64Rows4:
            (rows + 3) / 4
        default:
            (rows + 7) / 8
        }
        #expect(lines.last?.contains("width: \(expectedWidth)") == true)
    }
}

@Test func emitQwenSpecializedFusedDualAffineMatvecProducesNoSetBytes() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let lines = try emitter.emitFusedDualAffineMatvec(
        w1WeightsSlot: 30, w1WeightsOffset: 0,
        w1ScalesSlot: 30, w1ScalesOffset: 1_024,
        w1BiasesSlot: 30, w1BiasesOffset: 2_048,
        w2WeightsSlot: 30, w2WeightsOffset: 3_072,
        w2ScalesSlot: 30, w2ScalesOffset: 4_096,
        w2BiasesSlot: 30, w2BiasesOffset: 5_120,
        inputSlot: 8, output1Slot: 13, output2Slot: 14,
        rows: 16, cols: 2_048, groupSize: 64
    )

    #expect(lines[0].contains("setComputePipelineState(p[\(SmeltPipeline.fusedDualAffineMatvecC2048R16G64.rawValue)])"))
    #expect(!lines.joined(separator: "\n").contains("setBytes"))

    let qwen08Lines = try emitter.emitFusedDualAffineMatvec(
        w1WeightsSlot: 30, w1WeightsOffset: 0,
        w1ScalesSlot: 30, w1ScalesOffset: 1_024,
        w1BiasesSlot: 30, w1BiasesOffset: 2_048,
        w2WeightsSlot: 30, w2WeightsOffset: 3_072,
        w2ScalesSlot: 30, w2ScalesOffset: 4_096,
        w2BiasesSlot: 30, w2BiasesOffset: 5_120,
        inputSlot: 8, output1Slot: 13, output2Slot: 14,
        rows: 16, cols: 1_024, groupSize: 64
    )
    #expect(qwen08Lines[0].contains("setComputePipelineState(p[\(SmeltPipeline.fusedDualAffineMatvecC1024R16G64Rows4.rawValue)])"))
    #expect(qwen08Lines.last?.contains("width: 4") == true)
    #expect(!qwen08Lines.joined(separator: "\n").contains("setBytes"))
}

@Test func emitQwenSpecializedFusedAffineGateUpSwigluProducesNoSetBytes() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let cases: [(rows: Int, cols: Int, pipeline: SmeltPipeline, tgWidth: Int)] = [
        (3_584, 1_024, .fusedAffineGateUpSwigluC1024R3584G64Rows4, 64),
        (6_144, 2_048, .fusedAffineGateUpSwigluC2048R6144G64, 64),
        (11_008, 2_048, .fusedAffineGateUpSwigluC2048R11008G64, 64),
    ]

    for (rows, cols, pipeline, tgWidth) in cases {
        let lines = try emitter.emitFusedAffineGateUpSwiglu(
            gateWeightsSlot: 30, gateWeightsOffset: 0,
            gateScalesSlot: 30, gateScalesOffset: 1_024,
            gateBiasesSlot: 30, gateBiasesOffset: 2_048,
            upWeightsSlot: 30, upWeightsOffset: 3_072,
            upScalesSlot: 30, upScalesOffset: 4_096,
            upBiasesSlot: 30, upBiasesOffset: 5_120,
            inputSlot: 8, outputSlot: 13,
            rows: rows, cols: cols, groupSize: 64
        )

        #expect(lines[0].contains("setComputePipelineState(p[\(pipeline.rawValue)])"))
        #expect(!lines.joined(separator: "\n").contains("setBytes"))
        #expect(lines.joined(separator: "\n").contains("threadsPerThreadgroup: MTLSize(width: \(tgWidth)"))
        let expectedWidth = (rows + 3) / 4
        #expect(lines.joined(separator: "\n").contains("dispatchThreadgroups(MTLSize(width: \(expectedWidth)"))
    }
}

@Test func emitSpecializedFusedAffineGateUpGeGLUProducesNoSetBytes() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let cases: [(rows: Int, cols: Int, pipeline: SmeltPipeline)] = [
        (6_144, 1_536, .fusedAffineGateUpGeGLUC1536R6144G128Rows4),
        (12_288, 1_536, .fusedAffineGateUpGeGLUC1536R12288G128Rows4),
    ]

    for (rows, cols, pipeline) in cases {
        let lines = try emitter.emitFusedAffineGateUpGeGLU(
            gateWeightsSlot: 30, gateWeightsOffset: 0,
            gateScalesSlot: 30, gateScalesOffset: 1_024,
            gateBiasesSlot: 30, gateBiasesOffset: 2_048,
            upWeightsSlot: 30, upWeightsOffset: 3_072,
            upScalesSlot: 30, upScalesOffset: 4_096,
            upBiasesSlot: 30, upBiasesOffset: 5_120,
            inputSlot: 8, outputSlot: 13,
            rows: rows, cols: cols, groupSize: 128
        )

        #expect(lines[0].contains("setComputePipelineState(p[\(pipeline.rawValue)])"))
        #expect(!lines.joined(separator: "\n").contains("setBytes"))
        let expectedWidth = (rows + 3) / 4
        #expect(lines.joined(separator: "\n").contains("dispatchThreadgroups(MTLSize(width: \(expectedWidth)"))
    }
}

@Test func emitSpecializedNormScaleAffineGateUpGeGLUProducesNoSetBytes() throws {
    var emitter = SmeltCodeEmitter(indent: 4)
    let cases: [(rows: Int, cols: Int, pipeline: SmeltPipeline)] = [
        (6_144, 1_536, .normScaleAffineGateUpGeGLUC1536R6144G128Rows4),
        (12_288, 1_536, .normScaleAffineGateUpGeGLUC1536R12288G128Rows4),
    ]
    let gateEntry = SmeltWeightEntry(
        name: "gate",
        offset: 0,
        sizeBytes: 6_144 * 1_536 / 2,
        shape: [6_144, 1_536],
        dtype: .affineU4,
        groupSize: 128,
        scalesOffset: 1_024,
        scalesSizeBytes: UInt64(6_144 * (1_536 / 128) * 2),
        biasesOffset: 2_048,
        biasesSizeBytes: UInt64(6_144 * (1_536 / 128) * 2)
    )
    let upEntry = SmeltWeightEntry(
        name: "up",
        offset: 3_072,
        sizeBytes: 6_144 * 1_536 / 2,
        shape: [6_144, 1_536],
        dtype: .affineU4,
        groupSize: 128,
        scalesOffset: 4_096,
        scalesSizeBytes: UInt64(6_144 * (1_536 / 128) * 2),
        biasesOffset: 5_120,
        biasesSizeBytes: UInt64(6_144 * (1_536 / 128) * 2)
    )

    for (rows, cols, pipeline) in cases {
        let lines = try emitter.emitNormScaleAffineGateUpGeGLUIfPossible(
            normInputSlotVar: "cur",
            normWeightSlot: 30,
            normWeightOffset: 6_144,
            gateEntry: gateEntry,
            upEntry: upEntry,
            weightsSlot: 30,
            outputSlot: 13,
            rows: rows,
            cols: cols,
            groupSize: 128
        )

        #expect(lines != nil)
        let emitted = lines!.joined(separator: "\n")
        #expect(emitted.contains("setComputePipelineState(p[\(SmeltPipeline.rmsNormScaleOnlyD1536.rawValue)])"))
        #expect(emitted.contains("setComputePipelineState(p[\(pipeline.rawValue)])"))
        #expect(!emitted.contains("setBytes"))
        let expectedWidth = (rows + 3) / 4
        #expect(emitted.contains("dispatchThreadgroups(MTLSize(width: \(expectedWidth)"))
    }
}

@Test func emitRawAndComment() {
    let emitter = SmeltCodeEmitter(indent: 4)
    #expect(emitter.emitComment("swap buffers") == "    // swap buffers")
    #expect(emitter.emitRaw("swap(&cur, &alt)") == "    swap(&cur, &alt)")
    #expect(emitter.emitBlank() == "")
}
