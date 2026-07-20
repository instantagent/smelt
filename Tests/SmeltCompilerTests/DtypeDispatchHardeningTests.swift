// DtypeDispatchHardeningTests — Trap #3 hardening.
//
// Every dtype dispatcher must throw on dtypes it has no kernel/path for,
// never fall through silently (emitMatvec's old fp16 fallthrough, the affine
// quantizer's old zero-fill/warn-and-skip).
//
// As of dtype-building-blocks U2, fp16-act × {bf16,fp32} DENSE now HAS a kernel
// (fp16_matvec_{bf16,fp32}w) and emits — so the "truly unsupported" matvec dtypes
// are only int32/raw; bf16/fp32 emit (asserted positively below).

import Foundation
import Testing

@testable import SmeltCompiler
@testable import SmeltSchema

private func entry(
    dtype: SmeltDType,
    lutOffset: UInt64? = nil,
    scalesOffset: UInt64? = nil,
    biasesOffset: UInt64? = nil
) -> SmeltWeightEntry {
    SmeltWeightEntry(
        name: "test_weight",
        offset: 0,
        sizeBytes: 64 * 64,
        shape: [64, 64],
        dtype: dtype,
        groupSize: 32,
        lutOffset: lutOffset,
        scalesOffset: scalesOffset,
        biasesOffset: biasesOffset
    )
}

@Test func emitMatvecThrowsOnUnsupportedDtypes() {
    // int32/raw are not matvec weights (no dense kernel) — still LOUD. bf16/fp32 are NO LONGER here:
    // U2 gave them a dense kernel; see emitMatvecAndVarNowEmitBF16FP32Dense.
    for dtype in [SmeltDType.int32, .raw] {
        var emitter = SmeltCodeEmitter()
        #expect(throws: SmeltEmitError.self) {
            try emitter.emitMatvec(
                weightEntry: entry(dtype: dtype),
                weightsSlot: 30, inputSlot: 0, outputSlot: 1,
                rows: 64, cols: 64, groupSize: 32
            )
        }
    }
}

@Test func emitMatvecAndVarNowEmitBF16FP32Dense() throws {
    // The U2 capability: fp16-act × {bf16,fp32} dense emits (no metadata needed — dense weight).
    // Non-vacuous counterpart to dropping bf16/fp32 from the throw lists above.
    for dtype in [SmeltDType.bf16, .fp32] {
        var fixed = SmeltCodeEmitter()
        let fixedLines = try fixed.emitMatvec(
            weightEntry: entry(dtype: dtype),
            weightsSlot: 30, inputSlot: 0, outputSlot: 1,
            rows: 64, cols: 64, groupSize: 32
        )
        #expect(!fixedLines.isEmpty, "emitMatvec should emit for \(dtype)")

        var variable = SmeltCodeEmitter()
        let varLines = try variable.emitMatvecVar(
            weightEntry: entry(dtype: dtype),
            weightsSlot: 30, inputSlotVar: "cur", outputSlot: 1,
            rows: 64, cols: 64, groupSize: 32
        )
        #expect(!varLines.isEmpty, "emitMatvecVar should emit for \(dtype)")
    }
}

@Test func emitMatvecThrowsOnQuantEntryMissingMetadata() {
    // u4Lut without lutOffset / affineU4 without scales+biases used to fall
    // through to the fp16 kernel and read packed nibbles as fp16.
    for dtype in [SmeltDType.u4Lut, .affineU4, .turboQuantH] {
        var emitter = SmeltCodeEmitter()
        #expect(throws: SmeltEmitError.self) {
            try emitter.emitMatvec(
                weightEntry: entry(dtype: dtype),
                weightsSlot: 30, inputSlot: 0, outputSlot: 1,
                rows: 64, cols: 64, groupSize: 32
            )
        }
    }
}

@Test func emitMatvecVarThrowsOnUnsupportedDtypes() {
    // int32/raw: not matvec weights. u4Lut/affineU4: registered families but this entry() carries
    // no lut/scales/biases offsets, so the metadata guard throws (the silent-nibbles-as-fp16 trap).
    // bf16/fp32 are NO LONGER here — they emit (emitMatvecAndVarNowEmitBF16FP32Dense).
    for dtype in [SmeltDType.int32, .raw, .u4Lut, .affineU4] {
        var emitter = SmeltCodeEmitter()
        #expect(throws: SmeltEmitError.self) {
            try emitter.emitMatvecVar(
                weightEntry: entry(dtype: dtype),
                weightsSlot: 30, inputSlotVar: "cur", outputSlot: 1,
                rows: 64, cols: 64, groupSize: 32
            )
        }
    }
}

@Test func emitMatvecStillEmitsForSupportedEntries() throws {
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitMatvec(
        weightEntry: entry(
            dtype: .affineU4, scalesOffset: 4_096, biasesOffset: 8_192
        ),
        weightsSlot: 30, inputSlot: 0, outputSlot: 1,
        rows: 64, cols: 64, groupSize: 32
    )
    #expect(!lines.isEmpty)
}

@Test func affineQuantizerThrowsOnUnsupportedSourceDtype() {
    let data = [UInt8](repeating: 0, count: 64 * 64 * 8)
    let outputPath = NSTemporaryDirectory()
        + "dtype_hardening_\(ProcessInfo.processInfo.globallyUniqueString).bin"
    defer { try? FileManager.default.removeItem(atPath: outputPath) }

    let config = SmeltQuantizationConfig(
        strategy: .affineU4, groupSize: 32, excludePatterns: []
    )

    data.withUnsafeBytes { raw in
        let tensor = (
            runtimeName: "test_weight",
            data: raw.baseAddress! as UnsafeRawPointer,
            byteCount: data.count,
            shape: [64, 64],
            dtype: "F64"
        )
        #expect(throws: SmeltAffineQuantizerError.self) {
            try SmeltAffineQuantizer.quantize(
                tensors: [tensor], config: config, outputPath: outputPath
            )
        }
    }
}
