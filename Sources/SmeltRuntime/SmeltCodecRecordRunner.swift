// SmeltCodecRecordRunner — the codec interpret seam (C1, docs/codec-c1-plan.md). Runs a flat
// list of LITERAL-kind dispatch records over CALLER-SUPPLIED pipelines + buffers, with no
// package / IR / SmeltRuntime load. The conv-vocoder codec is emitted at a fixed shape, so every
// record is emit-time-static — literal grid dims, literal byte offsets, literal u32/f32
// constants. A record using a position / seqLen / cache-capacity / cur-alt kind is a programming
// error here (the transformer trunks use the full `interpretDispatchTable`, which resolves those
// against the runtime's request state). This is the reusable record-runner core C2/C3 extend
// (the codec analog of the trunk's `encodeTrunk*` external-encoder seam).

import Metal
import SmeltSchema

public enum SmeltCodecRecordRunner {

    /// Encode `records` into `enc` over `pipelines` (indexed by `record.pipeline`) and `buffers`
    /// (indexed by each binding's slot). Literal kinds only — asserts loud on a dynamic kind, so a
    /// mis-emitted record fails at the boundary rather than reading the wrong bytes.
    public static func encode(
        _ records: [SmeltDispatchRecord],
        pipelines: [MTLComputePipelineState],
        buffers: [MTLBuffer],
        into enc: MTLComputeCommandEncoder
    ) {
        for rec in records {
            // Fail loud on a non-dispatch op (e.g. a double-buffer swap): the codec table is
            // literal+static, so a swap/other op is a mis-emit, NOT a record to silently drop —
            // a dropped dispatch would corrupt the chain yet could still pass an unlucky memcmp.
            precondition(rec.opKind == SmeltDispatchRecord.opDispatch,
                         "codec record runner: only dispatch records (opKind \(SmeltDispatchRecord.opDispatch)), got \(rec.opKind)")
            precondition(Int(rec.pipeline) < pipelines.count,
                         "codec record runner: pipeline index \(rec.pipeline) out of range (\(pipelines.count))")
            enc.setComputePipelineState(pipelines[Int(rec.pipeline)])
            for b in 0..<Int(rec.bufferCount) {
                let buf = getBuffer(rec, index: b)
                precondition(buf.offsetKind == 0,
                             "codec record runner: only literal buffer offsets (kind 0), got \(buf.offsetKind)")
                precondition(Int(buf.slot) < buffers.count,
                             "codec record runner: buffer slot \(buf.slot) out of range (\(buffers.count))")
                enc.setBuffer(buffers[Int(buf.slot)], offset: Int(buf.offset), index: Int(buf.bindingIndex))
            }
            for c in 0..<Int(rec.constantCount) {
                let con = getConstant(rec, index: c)
                precondition(con.kind <= 1,
                             "codec record runner: only literal u32/f32 constants (kind 0/1), got \(con.kind)")
                var v = con.value
                enc.setBytes(&v, length: 4, index: Int(con.bindingIndex))
            }
            precondition(rec.gridWKind == 0 && rec.gridHKind == 0 && rec.gridDKind == 0,
                         "codec record runner: only literal grid dims")
            let grid = MTLSize(width: Int(rec.gridW), height: Int(rec.gridH), depth: Int(rec.gridD))
            let tg = MTLSize(width: Int(rec.tgW), height: Int(rec.tgH), depth: Int(rec.tgD))
            if rec.dispatchStyle == SmeltDispatchRecord.styleThreadgroups {
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            } else {
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
        }
    }
}
