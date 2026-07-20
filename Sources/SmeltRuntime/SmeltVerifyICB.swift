// SmeltVerifyICB — MTLIndirectCommandBuffer build path for the verify pass.
//
// The verify dispatch table is a serialized description of a Metal compute
// pass: ~886 records each producing one dispatch. Most records have
// per-round-invariant inputs (seqLen is constant at maxPrefillBatchSize,
// buffer slots/offsets are static, grid+threadgroup sizes are literal).
// Only records touching startPos — RoPE, KV cache writes, attention —
// vary per round.
//
// ICB lets us encode the static-record subset once at runtime init and
// replay it each round, eliminating the per-dispatch CPU encoder overhead
// for the static portion. With ~80% static records and ~22 µs encode/record,
// the theoretical CPU savings is ~16 ms per round (realistic ~5–10 ms after
// CPU/GPU overlap masks part of the encode time).

import Foundation
import Metal
import SmeltSchema

extension SmeltRuntime {

    enum VerifyICBClassification {
        case staticRecord
        case skipForBuild
        case dynamic
    }

    enum VerifyICBState {
        case pending
        case failed
        case built(VerifyICBResult)
    }

    static func classifyVerifyRecord(
        _ rec: SmeltDispatchRecord, verifySeqLen: Int32
    ) -> VerifyICBClassification {
        if rec.opKind == SmeltDispatchRecord.opSwap {
            return .staticRecord
        }
        for bidx in 0..<Int(rec.bufferCount) {
            let buf = getBuffer(rec, index: bidx)
            if buf.offsetKind == 1 || buf.offsetKind == 2 {
                return .dynamic
            }
        }
        for cidx in 0..<Int(rec.constantCount) {
            let con = getConstant(rec, index: cidx)
            switch con.kind {
            case SmeltConstantRecord.kindPosition,
                 SmeltConstantRecord.kindPositionPlus1,
                 SmeltConstantRecord.kindStartPos,
                 SmeltConstantRecord.kindStartPosPlusLiteral,
                 SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse,
                 SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
                return .dynamic
            case SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
                let val = con.value == 0
                    ? 0
                    : UInt32(bitPattern: verifySeqLen) % con.value
                if val == 0 { return .skipForBuild }
                continue
            case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
                if UInt32(bitPattern: verifySeqLen) >= con.value {
                    return .skipForBuild
                }
                continue
            default:
                continue
            }
        }
        return .staticRecord
    }

    struct VerifyICBStats {
        let totalRecords: Int
        let staticDispatchCount: Int
        let dynamicDispatchCount: Int
        let swapCount: Int
        let skippedByMinSeqLen: Int
        let skippedByZeroGrid: Int
        let skippedBySkipIfZero: Int
        let constantsBufferBytes: Int
        let verifySeqLen: Int32
    }

    /// Interleaved execution plan that preserves original dispatch order.
    /// Each step is either a run of ICB commands or a single dynamic
    /// record that must be encoded manually with its build-time cur/alt.
    enum ExecutionStep {
        case icbRange(Range<Int>)
        case dynamic(record: SmeltDispatchRecord, cur: Int, alt: Int)
    }

    struct VerifyICBResult {
        let icb: MTLIndirectCommandBuffer
        let constantsBuffer: MTLBuffer
        /// Pre-materialized [MTLBuffer] array for useResources. Stable
        /// order so Metal can hash/dedup deterministically across runs.
        let referencedBuffers: [MTLBuffer]
        let executionScript: [ExecutionStep]
        let stats: VerifyICBStats
    }

    /// Run the verify pass via the built ICB. Walks `executionScript` to
    /// interleave `executeCommandsInBuffer` ranges with manually-encoded
    /// dynamic records (RoPE / KV write / attention) at their original
    /// positions in the dispatch order. Caller drives the command-buffer
    /// commit; this only fills the encoder.
    func executeVerifyICB(
        _ result: VerifyICBResult,
        enc: MTLComputeCommandEncoder,
        seqLen: Int32, startPos: Int32
    ) {
        enc.useResources(result.referencedBuffers, usage: [.read, .write])
        enc.useResource(result.constantsBuffer, usage: .read)

        // .concurrentDispatch ICBs run commands concurrently among
        // themselves inside a single executeCommandsInBuffer call.
        // Verify has layer-to-layer serial deps, so we submit each
        // command as its own range — the parent encoder's hazard
        // tracking then serializes them via useResource declarations.
        for step in result.executionScript {
            switch step {
            case .icbRange(let range):
                for i in range {
                    enc.executeCommandsInBuffer(result.icb, range: i..<i+1)
                }
            case .dynamic(let record, let cur, let alt):
                emitDispatchRecord(
                    record,
                    enc: enc, cur: cur, alt: alt,
                    seqLen: seqLen, startPos: startPos
                )
            }
        }
    }

    func buildVerifyICB(
        table: UnsafeBufferPointer<SmeltDispatchRecord>,
        verifySeqLen: Int32
    ) -> VerifyICBResult? {
        let plan = planVerifyICB(table: table, verifySeqLen: verifySeqLen)
        guard !plan.staticSlots.isEmpty else { return nil }

        let constsPerRecord = agentMaxConstantsPerDispatch
        let constantsBytes = plan.staticSlots.count * constsPerRecord * 4
        guard let constantsBuf = device.makeBuffer(
            length: constantsBytes, options: .storageModeShared
        ) else { return nil }
        let constsPtr = constantsBuf.contents()
            .bindMemory(to: UInt32.self, capacity: constantsBytes / 4)

        let icbDesc = MTLIndirectCommandBufferDescriptor()
        icbDesc.commandTypes = .concurrentDispatch
        icbDesc.inheritBuffers = false
        icbDesc.inheritPipelineState = false
        icbDesc.maxKernelBufferBindCount = agentMaxBuffersPerDispatch
        guard let icb = device.makeIndirectCommandBuffer(
            descriptor: icbDesc,
            maxCommandCount: plan.staticSlots.count,
            options: []
        ) else { return nil }

        var referencedSlots: Set<Int> = []

        for (icbIdx, slot) in plan.staticSlots.enumerated() {
            let rec = table[slot.recordPosition]
            let cmd = icb.indirectComputeCommandAt(icbIdx)
            cmd.setComputePipelineState(pipelineState(for: Int(rec.pipeline)))

            for bidx in 0..<Int(rec.bufferCount) {
                let buf = getBuffer(rec, index: bidx)
                let bufSlot = resolveDispatchBufferSlot(
                    buf, cur: slot.cur, alt: slot.alt
                )
                // Classifier already excluded offsetKind 1/2 (dynamic);
                // startPos=0 is unreachable for the remaining kinds.
                let offset = resolveDispatchBufferOffset(
                    buf, seqLen: verifySeqLen, startPos: 0
                )
                cmd.setKernelBuffer(
                    buffers[bufSlot], offset: offset, at: Int(buf.bindingIndex)
                )
                referencedSlots.insert(bufSlot)
            }

            let recordConstOffset = icbIdx * constsPerRecord
            var unusedSkip = false
            for cidx in 0..<Int(rec.constantCount) {
                let con = getConstant(rec, index: cidx)
                let resolved = resolvePrefillConstant(
                    con,
                    seqLen: verifySeqLen,
                    startPos: 0,
                    skipDispatch: &unusedSkip
                )
                constsPtr[recordConstOffset + cidx] = resolved
                if con.bindingIndex != UInt8.max {
                    cmd.setKernelBuffer(
                        constantsBuf,
                        offset: (recordConstOffset + cidx) * 4,
                        at: Int(con.bindingIndex)
                    )
                }
            }

            let tgSize = MTLSize(
                width: Int(rec.tgW), height: Int(rec.tgH), depth: Int(rec.tgD)
            )
            let gridSize = MTLSize(
                width: slot.gridW, height: slot.gridH, depth: slot.gridD
            )
            if rec.dispatchStyle == SmeltDispatchRecord.styleThreadgroups {
                cmd.concurrentDispatchThreadgroups(
                    gridSize, threadsPerThreadgroup: tgSize
                )
            } else {
                cmd.concurrentDispatchThreads(
                    gridSize, threadsPerThreadgroup: tgSize
                )
            }
        }

        let stats = VerifyICBStats(
            totalRecords: table.count,
            staticDispatchCount: plan.staticSlots.count,
            dynamicDispatchCount: plan.dynamicCount,
            swapCount: plan.swapCount,
            skippedByMinSeqLen: plan.skippedByMinSeqLen,
            skippedByZeroGrid: plan.skippedByZeroGrid,
            skippedBySkipIfZero: plan.skippedBySkipIfZero,
            constantsBufferBytes: constantsBytes,
            verifySeqLen: verifySeqLen
        )

        let referencedBuffers = referencedSlots.sorted().map { buffers[$0] }

        return VerifyICBResult(
            icb: icb,
            constantsBuffer: constantsBuf,
            referencedBuffers: referencedBuffers,
            executionScript: plan.executionScript,
            stats: stats
        )
    }

    private struct VerifyICBPlan {
        struct StaticSlot {
            let recordPosition: Int
            let cur: Int
            let alt: Int
            let gridW: Int
            let gridH: Int
            let gridD: Int
        }
        var staticSlots: [StaticSlot] = []
        var dynamicCount: Int = 0
        var swapCount: Int = 0
        var skippedByMinSeqLen: Int = 0
        var skippedByZeroGrid: Int = 0
        var skippedBySkipIfZero: Int = 0
        var executionScript: [ExecutionStep] = []
    }

    private func planVerifyICB(
        table: UnsafeBufferPointer<SmeltDispatchRecord>,
        verifySeqLen: Int32
    ) -> VerifyICBPlan {
        var plan = VerifyICBPlan()
        var cur = 0
        var alt = 1
        var pendingIcbStart: Int? = nil
        var nextIcbSlot = 0

        func flushPending() {
            if let start = pendingIcbStart, nextIcbSlot > start {
                plan.executionScript.append(.icbRange(start..<nextIcbSlot))
            }
            pendingIcbStart = nil
        }

        for i in 0..<table.count {
            let rec = table[i]
            if rec.opKind == SmeltDispatchRecord.opSwap {
                swap(&cur, &alt)
                plan.swapCount += 1
                continue
            }
            if rec.minSeqLen > 0 && UInt32(verifySeqLen) < UInt32(rec.minSeqLen) {
                plan.skippedByMinSeqLen += 1
                continue
            }
            switch SmeltRuntime.classifyVerifyRecord(rec, verifySeqLen: verifySeqLen) {
            case .dynamic:
                flushPending()
                plan.executionScript.append(
                    .dynamic(record: rec, cur: cur, alt: alt)
                )
                plan.dynamicCount += 1
                continue
            case .skipForBuild:
                plan.skippedBySkipIfZero += 1
                continue
            case .staticRecord:
                break
            }

            let gridW = resolvePrefillGrid(
                rec.gridW, kind: rec.gridWKind, seqLen: verifySeqLen
            )
            let gridH = resolvePrefillGrid(
                rec.gridH, kind: rec.gridHKind, seqLen: verifySeqLen
            )
            let gridD = resolvePrefillGrid(
                rec.gridD, kind: rec.gridDKind, seqLen: verifySeqLen
            )
            if gridW == 0 || gridH == 0 || gridD == 0 {
                plan.skippedByZeroGrid += 1
                continue
            }

            plan.staticSlots.append(VerifyICBPlan.StaticSlot(
                recordPosition: i,
                cur: cur, alt: alt,
                gridW: gridW, gridH: gridH, gridD: gridD
            ))
            if pendingIcbStart == nil { pendingIcbStart = nextIcbSlot }
            nextIcbSlot += 1
        }
        flushPending()
        return plan
    }
}
