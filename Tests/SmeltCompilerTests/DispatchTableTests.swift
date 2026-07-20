// DispatchTableTests — Verify dispatch table record layout and serialization.

import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class DispatchTableTests: XCTestCase {

    /// Pin exact sizes and strides. If these change, dispatches.bin from
    /// older packages becomes unreadable. Fail loudly so we bump a version.
    func testRecordSizesArePinned() {
        XCTAssertEqual(MemoryLayout<SmeltBufferRecord>.stride, 16)
        XCTAssertEqual(MemoryLayout<SmeltConstantRecord>.stride, 8)
        XCTAssertEqual(MemoryLayout<SmeltDispatchRecord>.stride, 360)

        // Strides must equal sizes (no inter-element padding)
        XCTAssertEqual(
            MemoryLayout<SmeltBufferRecord>.size,
            MemoryLayout<SmeltBufferRecord>.stride
        )
        XCTAssertEqual(
            MemoryLayout<SmeltConstantRecord>.size,
            MemoryLayout<SmeltConstantRecord>.stride
        )
        XCTAssertEqual(
            MemoryLayout<SmeltDispatchRecord>.size,
            360
        )
    }

    func testSwapRecord() {
        let rec = SmeltDispatchRecord.swap()
        XCTAssertEqual(rec.opKind, SmeltDispatchRecord.opSwap)
        XCTAssertEqual(rec.bufferCount, 0)
        XCTAssertEqual(rec.constantCount, 0)
    }

    func testEmptyRecord() {
        let rec = SmeltDispatchRecord.empty()
        XCTAssertEqual(rec.opKind, SmeltDispatchRecord.opDispatch)
        XCTAssertEqual(rec.pipeline, 0)
        XCTAssertEqual(rec.bufferCount, 0)
        XCTAssertEqual(rec.constantCount, 0)
    }

    func testBufferSlotSentinels() {
        XCTAssertEqual(SmeltBufferRecord.slotCur, -1)
        XCTAssertEqual(SmeltBufferRecord.slotAlt, -2)
    }

    func testConstantKinds() {
        XCTAssertEqual(SmeltConstantRecord.kindLiteralU32, 0)
        XCTAssertEqual(SmeltConstantRecord.kindLiteralF32, 1)
        XCTAssertEqual(SmeltConstantRecord.kindPosition, 2)
        XCTAssertEqual(SmeltConstantRecord.kindPositionPlus1, 3)
        XCTAssertEqual(SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse, 10)
        XCTAssertEqual(SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse, 11)
        XCTAssertEqual(SmeltConstantRecord.kindCacheSeqCapacity, 12)
        XCTAssertEqual(SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse, 13)
    }

    // MARK: - SmeltDispatch → SmeltDispatchRecord conversion

    func testDispatchToRecord_StaticConstants() {
        let dispatch = SmeltDispatch(
            pipeline: .fusedLutMatvec,
            buffers: [
                SmeltBufferBinding(slot: 30, offset: 12345, index: 0),
                SmeltBufferBinding(slot: 30, offset: 67890, index: 1),
                SmeltBufferBinding(slot: 8, index: 2),
                SmeltBufferBinding(slot: 2, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "2048", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "16", type: .uint32, index: 5),
            ],
            dispatch: .threadgroups(
                width: 6144, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: "QKV projection"
        )

        let rec = dispatch.toRecord()
        XCTAssertEqual(rec.opKind, SmeltDispatchRecord.opDispatch)
        XCTAssertEqual(rec.pipeline, UInt16(SmeltPipeline.fusedLutMatvec.rawValue))
        XCTAssertEqual(rec.style, SmeltDispatchRecord.styleThreadgroups)
        XCTAssertEqual(rec.bufferCount, 4)
        XCTAssertEqual(rec.constantCount, 2)
        XCTAssertEqual(rec.gridW, 6144)
        XCTAssertEqual(rec.tgW, 256)

        XCTAssertEqual(rec.buf0.slot, 30)
        XCTAssertEqual(rec.buf0.offset, 12345)
        XCTAssertEqual(rec.buf0.bindingIndex, 0)
        XCTAssertEqual(rec.buf1.offset, 67890)

        XCTAssertEqual(rec.con0.kind, SmeltConstantRecord.kindLiteralU32)
        XCTAssertEqual(rec.con0.value, 2048)
        XCTAssertEqual(rec.con0.bindingIndex, 4)
        XCTAssertEqual(rec.con1.value, 16)
    }

    func testDispatchToRecord_DynamicPosition() {
        let dispatch = SmeltDispatch(
            pipeline: .kvCacheUpdate,
            buffers: [
                SmeltBufferBinding(slot: 67, index: 0),
                SmeltBufferBinding(slot: 21, index: 1),
            ],
            constants: [
                SmeltConstantBinding(expression: "256", type: .uint32, index: 2),
                SmeltConstantBinding(expression: "UInt32(position)", type: .uint32, index: 4),
                SmeltConstantBinding(
                    expression: "UInt32(position + 1)", type: .uint32, index: 7
                ),
            ],
            dispatch: .threads(
                width: 512, height: 1, depth: 1,
                tgWidth: 512, tgHeight: 1, tgDepth: 1
            ),
            comment: nil
        )

        let rec = dispatch.toRecord()
        XCTAssertEqual(rec.style, SmeltDispatchRecord.styleThreads)
        XCTAssertEqual(rec.constantCount, 3)

        XCTAssertEqual(rec.con0.kind, SmeltConstantRecord.kindLiteralU32)
        XCTAssertEqual(rec.con0.value, 256)
        XCTAssertEqual(rec.con1.kind, SmeltConstantRecord.kindPosition)
        XCTAssertEqual(rec.con2.kind, SmeltConstantRecord.kindPositionPlus1)
    }

    func testDispatchToRecord_RuntimeCacheSeqCapacity() {
        let dispatch = SmeltDispatch(
            pipeline: .attentionDecode,
            buffers: [
                SmeltBufferBinding(slot: 20, index: 0),
                SmeltBufferBinding(slot: 67, index: 1),
                SmeltBufferBinding(slot: 73, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 6),
            ],
            dispatch: .threadgroups(
                width: 8, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: nil
        )

        let rec = dispatch.toRecord()
        XCTAssertEqual(rec.constantCount, 1)
        XCTAssertEqual(rec.con0.kind, SmeltConstantRecord.kindCacheSeqCapacity)
        XCTAssertEqual(rec.con0.bindingIndex, 6)
    }

    func testDispatchToRecord_DoubleBuffer() {
        let dispatch = SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(variableSlot: "cur", index: 0),
                SmeltBufferBinding(slot: 14, index: 1),
                SmeltBufferBinding(variableSlot: "alt", index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: 2048, height: 1, depth: 1,
                tgWidth: 1024, tgHeight: 1, tgDepth: 1
            ),
            comment: nil
        )

        let rec = dispatch.toRecord()
        XCTAssertEqual(rec.buf0.slot, SmeltBufferRecord.slotCur)
        XCTAssertEqual(rec.buf2.slot, SmeltBufferRecord.slotAlt)
        XCTAssertEqual(rec.buf1.slot, 14)
    }

    // MARK: - Full model dispatch table

    func testQwen35DispatchTableSize() throws {
        let ir = SmeltModelIR.qwen35_2B
        try validateSmeltIR(ir)
        let plan = buildBufferPlan(from: ir)
        let layout = SmeltWeightLayout.computeLayout(from: ir)

        let result = try TopLevelEmitter.generate(
            ir: ir, plan: plan, weightLayout: layout
        )

        let records = result.dispatchRecords
        let dispatches = records.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
        let swaps = records.filter { $0.opKind == SmeltDispatchRecord.opSwap }

        fputs(
            "  Dispatch table: \(records.count) total"
                + " (\(dispatches.count) dispatches + \(swaps.count) swaps)\n",
            stderr
        )

        // 389 dispatches after decode-side optimization and removal of the
        // unused speculative-decode surface:
        // - split DeltaNet conv+SiLU and joint Q/K RMS scaling
        // - fused matvec+residual projections
        // - mutually exclusive compact/vector/two-pass D256 attention routes
        // - per-head Q/K RMS norms on decode attention
        XCTAssertEqual(dispatches.count, 389)
        // 48 swaps (24 layers × 2 swaps per layer)
        XCTAssertEqual(swaps.count, 48)
        // Total
        XCTAssertEqual(records.count, 437)
    }

    func testDispatchToRecord_FloatConstant() {
        let dispatch = SmeltDispatch(
            pipeline: .rmsNorm1PW,
            buffers: [SmeltBufferBinding(slot: 0, index: 0)],
            constants: [
                SmeltConstantBinding(expression: "1e-06", type: .float32, index: 4),
            ],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: 1024, tgHeight: 1, tgDepth: 1
            ),
            comment: nil
        )

        let rec = dispatch.toRecord()
        XCTAssertEqual(rec.con0.kind, SmeltConstantRecord.kindLiteralF32)
        let reconstituted = Float(bitPattern: rec.con0.value)
        XCTAssertEqual(reconstituted, 1e-06, accuracy: 1e-10)
    }

    func testRoundTripAsBytes() {
        // Create a record, write to bytes, read back, verify identical
        var rec = SmeltDispatchRecord.empty()
        rec.opKind = SmeltDispatchRecord.opDispatch
        rec.pipeline = 7
        rec.style = SmeltDispatchRecord.styleThreads
        rec.bufferCount = 2
        rec.constantCount = 1
        rec.gridW = 2048
        rec.gridH = 1
        rec.gridD = 1
        rec.tgW = 256
        rec.tgH = 1
        rec.tgD = 1
        rec.buf0 = SmeltBufferRecord(slot: 30, bindingIndex: 0, offsetKind: 0, offset: 12345)
        rec.buf1 = SmeltBufferRecord(slot: SmeltBufferRecord.slotCur, bindingIndex: 1, offsetKind: 0, offset: 0)
        rec.con0 = SmeltConstantRecord(
            kind: SmeltConstantRecord.kindLiteralU32, bindingIndex: 4, pad: 0, value: 2048
        )

        // Write to bytes
        let size = MemoryLayout<SmeltDispatchRecord>.stride
        var data = Data(count: size)
        data.withUnsafeMutableBytes { ptr in
            ptr.storeBytes(of: rec, as: SmeltDispatchRecord.self)
        }

        // Read back
        let read = data.withUnsafeBytes { ptr in
            ptr.load(as: SmeltDispatchRecord.self)
        }

        XCTAssertEqual(read.opKind, SmeltDispatchRecord.opDispatch)
        XCTAssertEqual(read.pipeline, 7)
        XCTAssertEqual(read.bufferCount, 2)
        XCTAssertEqual(read.constantCount, 1)
        XCTAssertEqual(read.gridW, 2048)
        XCTAssertEqual(read.tgW, 256)
        XCTAssertEqual(read.buf0.slot, 30)
        XCTAssertEqual(read.buf0.offset, 12345)
        XCTAssertEqual(read.buf1.slot, SmeltBufferRecord.slotCur)
        XCTAssertEqual(read.con0.value, 2048)
    }

    // MARK: - Prefill schema extensions

    /// Pin prefill constant kinds — changing these breaks prefill_dispatches.bin.
    func testPrefillConstantKindsArePinned() {
        XCTAssertEqual(SmeltConstantRecord.kindSeqLen, 4)
        XCTAssertEqual(SmeltConstantRecord.kindStartPos, 5)
        XCTAssertEqual(SmeltConstantRecord.kindStartPosPlusLiteral, 6)
        XCTAssertEqual(SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse, 13)
    }

    func testDispatchToRecord_MaxSeqLenGuardUsesNonBindingConstant() {
        let record = SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [],
            constants: [],
            dispatch: .threads(
                width: 1, height: 1, depth: 1,
                tgWidth: 1, tgHeight: 1, tgDepth: 1
            ),
            maxSeqLenExclusive: 6
        ).toRecord()

        XCTAssertEqual(record.constantCount, 1)
        XCTAssertEqual(
            record.con0.kind,
            SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse
        )
        XCTAssertEqual(record.con0.bindingIndex, UInt8.max)
        XCTAssertEqual(record.con0.value, 6)
    }

    /// Verify constant kinds don't overlap with existing decode kinds.
    func testPrefillConstantKindsNoOverlap() {
        let decodeKinds: Set<UInt8> = [
            SmeltConstantRecord.kindLiteralU32,
            SmeltConstantRecord.kindLiteralF32,
            SmeltConstantRecord.kindPosition,
            SmeltConstantRecord.kindPositionPlus1,
        ]
        let prefillKinds: Set<UInt8> = [
            SmeltConstantRecord.kindSeqLen,
            SmeltConstantRecord.kindStartPos,
            SmeltConstantRecord.kindStartPosPlusLiteral,
        ]
        let decodeGuardKinds: Set<UInt8> = [
            SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse,
            SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse,
        ]
        XCTAssertTrue(decodeKinds.isDisjoint(with: prefillKinds))
        XCTAssertTrue(decodeGuardKinds.isDisjoint(with: prefillKinds))
    }

    /// Prefill offset kind 2: pack stride + addend into UInt64.
    /// Low 32 bits = stride, high 32 bits = literal addend.
    func testOffsetPackingRoundTrip() {
        let stride: UInt32 = 128   // ropeDim * fp16
        let addend: UInt32 = 1024  // position_in_batch * ropeDim * fp16
        let packed: UInt64 = UInt64(stride) | (UInt64(addend) << 32)

        // Unpack
        let unpackedStride = UInt32(packed & 0xFFFF_FFFF)
        let unpackedAddend = UInt32(packed >> 32)
        XCTAssertEqual(unpackedStride, stride)
        XCTAssertEqual(unpackedAddend, addend)
    }

    /// Offset packing with large real-world values.
    func testOffsetPackingLargeValues() {
        // RoPE stride for headDim=256: 256 * 2 = 512
        // Addend for position 63: 63 * 512 = 32256
        let stride: UInt32 = 512
        let addend: UInt32 = 32256
        let packed: UInt64 = UInt64(stride) | (UInt64(addend) << 32)

        let buf = SmeltBufferRecord(
            slot: 79, bindingIndex: 1, offsetKind: 2, offset: packed
        )
        XCTAssertEqual(UInt32(buf.offset & 0xFFFF_FFFF), stride)
        XCTAssertEqual(UInt32(buf.offset >> 32), addend)
    }

    /// kindStartPosPlusLiteral stores a literal in the value field.
    func testStartPosPlusLiteralRecord() {
        let con = SmeltConstantRecord(
            kind: SmeltConstantRecord.kindStartPosPlusLiteral,
            bindingIndex: 4,
            pad: 0,
            value: 7  // position 7 within the batch
        )
        // Runtime would resolve: UInt32(startPos) + con.value = startPos + 7
        XCTAssertEqual(con.kind, 6)
        XCTAssertEqual(con.value, 7)
    }

    /// Record stride unchanged after adding prefill constants.
    func testRecordStrideUnchangedForPrefill() {
        // This is the existing pinning test, repeated here to be explicit:
        // adding new constant kind VALUES must not change the record LAYOUT.
        XCTAssertEqual(MemoryLayout<SmeltConstantRecord>.stride, 8)
        XCTAssertEqual(MemoryLayout<SmeltDispatchRecord>.stride, 360)
    }
}
