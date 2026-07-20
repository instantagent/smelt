// SmeltCodecRecordBuilder — the runtime-local, literal-only dispatch-record builder for the compiled
// codec (C2-FULL, docs/codec-c2-full-plan.md §2.0). The ergonomic `SmeltDispatch`/`SmeltPipeline`
// builder lives in `SmeltCompiler`, which depends on `SmeltRuntime` (not the reverse), so the runtime
// constructs records directly over the public `SmeltSchema` POD. The codec table is emit-time-static,
// so this only ever needs LITERAL kinds (literal grid, literal byte offset 0, literal u32/f32
// constants) — exactly the subset `SmeltCodecRecordRunner` allows. The schema `SmeltDispatchRecord`
// field layout is the single source of truth; the packing here mirrors `SmeltDispatch.toRecord()`'s
// literal path. Pipelines are referenced by a DENSE LOCAL index into `Qwen3TTSCodecEmitter`'s
// `pipelineNames` (not `SmeltPipeline.rawValue`), matching how the runtime resolves `record.pipeline`
// positionally.

import SmeltSchema

enum SmeltCodecRecordBuilder {

    /// One literal constant binding: u32 (kind 0, value = bits) or f32 (kind 1, value = IEEE bits).
    struct Const {
        let kind: UInt8
        let value: UInt32
        let index: Int
        static func u32(_ v: UInt32, _ i: Int) -> Const { .init(kind: SmeltConstantRecord.kindLiteralU32, value: v, index: i) }
        static func f32(_ v: Float, _ i: Int) -> Const { .init(kind: SmeltConstantRecord.kindLiteralF32, value: v.bitPattern, index: i) }
    }

    /// Build a `dispatchThreads`-style record (every codec kernel uses `dispatchThreads`, confirmed
    /// against `Qwen3TTSGPUCodec.encode`). `pipeline` is the dense local index; `buffers` are
    /// (slot, bindingIndex) with literal offset 0; `constants` are literal u32/f32.
    static func threads(pipeline: Int,
                        buffers: [(slot: Int, index: Int)],
                        constants: [Const],
                        grid: (Int, Int, Int),
                        tg: (Int, Int, Int)) -> SmeltDispatchRecord {
        precondition(buffers.count <= agentMaxBuffersPerDispatch, "codec record: >\(agentMaxBuffersPerDispatch) buffers")
        precondition(constants.count <= agentMaxConstantsPerDispatch, "codec record: >\(agentMaxConstantsPerDispatch) constants")
        var r = SmeltDispatchRecord.empty()
        r.opKind = SmeltDispatchRecord.opDispatch
        r.pipeline = UInt16(pipeline)
        r.dispatchStyle = SmeltDispatchRecord.styleThreads
        r.gridW = UInt32(grid.0); r.gridH = UInt32(grid.1); r.gridD = UInt32(grid.2)
        r.tgW = UInt32(tg.0); r.tgH = UInt32(tg.1); r.tgD = UInt32(tg.2)
        r.bufferCount = UInt8(buffers.count)
        for (i, b) in buffers.enumerated() {
            // The schema memberwise init is internal; build via the public `.empty()` + field set.
            var rec = SmeltBufferRecord.empty()
            rec.slot = Int16(b.slot); rec.bindingIndex = UInt8(b.index); rec.offsetKind = 0; rec.offset = 0
            setBuffer(&r, i, rec)
        }
        r.constantCount = UInt8(constants.count)
        for (i, c) in constants.enumerated() {
            var rec = SmeltConstantRecord.empty()
            rec.kind = c.kind; rec.bindingIndex = UInt8(c.index); rec.pad = 0; rec.value = c.value
            setConstant(&r, i, rec)
        }
        return r
    }

    // The schema POD stores buffers/constants as fixed-name fields (buf0..15, con0..7); mirror the
    // schema `getBuffer`/`getConstant` readers with switch-based setters.
    private static func setBuffer(_ r: inout SmeltDispatchRecord, _ i: Int, _ b: SmeltBufferRecord) {
        switch i {
        case 0: r.buf0 = b;  case 1: r.buf1 = b;  case 2: r.buf2 = b;  case 3: r.buf3 = b
        case 4: r.buf4 = b;  case 5: r.buf5 = b;  case 6: r.buf6 = b;  case 7: r.buf7 = b
        case 8: r.buf8 = b;  case 9: r.buf9 = b;  case 10: r.buf10 = b; case 11: r.buf11 = b
        case 12: r.buf12 = b; case 13: r.buf13 = b; case 14: r.buf14 = b; case 15: r.buf15 = b
        default: preconditionFailure("codec record: buffer index \(i) out of range")
        }
    }
    private static func setConstant(_ r: inout SmeltDispatchRecord, _ i: Int, _ c: SmeltConstantRecord) {
        switch i {
        case 0: r.con0 = c; case 1: r.con1 = c; case 2: r.con2 = c; case 3: r.con3 = c
        case 4: r.con4 = c; case 5: r.con5 = c; case 6: r.con6 = c; case 7: r.con7 = c
        default: preconditionFailure("codec record: constant index \(i) out of range")
        }
    }
}
