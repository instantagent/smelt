import Foundation
import Testing
@testable import SmeltCompiler

// MARK: - TTS emit helpers (extracted from parked Kokoro work)

@Test func emitSnakeActivationProducesExpectedDispatch() throws {
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitSnakeActivation(
        inputSlot: 0, outputSlot: 1,
        alphaSlot: 30, alphaOffset: 0,
        channels: 256, length: 200
    )
    let dispatches = emitter.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    #expect(dispatches.count == 1)
    let rec = dispatches[0]
    #expect(rec.pipeline == UInt16(SmeltPipeline.snakeActivation.rawValue))
    #expect(rec.bufferCount == 3)
    #expect(rec.constantCount == 2)
}

@Test func emitConv1dProducesExpectedDispatch() throws {
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitConv1d(
        inputSlot: 0, outputSlot: 1,
        weightSlot: 30, weightOffset: 0,
        biasSlot: 30, biasOffset: 1000,
        cIn: 512, cOut: 512, kernel: 5,
        stride: 1, padding: 2, lengthIn: 100
    )
    let dispatches = emitter.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    #expect(dispatches.count == 1)
    let rec = dispatches[0]
    #expect(rec.pipeline == UInt16(SmeltPipeline.conv1dForward.rawValue))
    #expect(rec.bufferCount == 4)
    #expect(rec.constantCount == 8)
    #expect(rec.gridW > 0)
}

@Test func emitConvTranspose1dProducesExpectedDispatch() throws {
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitConvTranspose1d(
        inputSlot: 0, outputSlot: 1,
        weightSlot: 30, weightOffset: 0,
        biasSlot: 30, biasOffset: 500,
        cIn: 512, cOut: 256, kernel: 20,
        stride: 10, padding: 5, lengthIn: 50
    )
    let dispatches = emitter.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    #expect(dispatches.count == 1)
    let rec = dispatches[0]
    #expect(rec.pipeline == UInt16(SmeltPipeline.convTranspose1d.rawValue))
    #expect(rec.bufferCount == 4)
    #expect(rec.constantCount == 6)
}

@Test func emitLayerNormProducesExpectedDispatch() throws {
    var emitter = SmeltCodeEmitter()
    _ = try emitter.emitLayerNorm(
        inputSlot: 0, outputSlot: 1,
        weightSlot: 30, weightOffset: 0,
        biasSlot: 30, biasOffset: 200,
        dim: 768, rows: 4
    )
    let dispatches = emitter.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    #expect(dispatches.count == 1)
    let rec = dispatches[0]
    #expect(rec.pipeline == UInt16(SmeltPipeline.layerNorm.rawValue))
    #expect(rec.bufferCount == 4)
    #expect(rec.constantCount == 2)
    // One threadgroup per row — multi-row must dispatch `rows` groups, not 1.
    #expect(rec.gridW == 4)
}
