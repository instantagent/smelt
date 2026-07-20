import XCTest
import Metal
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// C1 micro (docs/codec-c1-plan.md): the FIRST conv-vocoder kernel emitted as a dispatch RECORD
/// (via the generic SmeltDispatch → toRecord path) and run through the standalone
/// SmeltCodecRecordRunner produces output BYTE-IDENTICAL to a MANUAL setBuffer+dispatch of the
/// same conv1d_forward_f32 kernel. Proves the EMIT (record fields: 2-D grid, literal constants
/// incl. the packed lengthIn, weight bindings) AND the RUNNER (slot/offset/constant/grid
/// interpretation) drive the kernel identically — no package, no IR, no weights.bin; random
/// buffers, bit-identical by construction (same kernel, same bindings, two encode paths).
final class CodecRecordRunnerTests: XCTestCase {

    func testConv1dRecordRunMatchesManualDispatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("needs Metal") }
        // conv1d_forward_f32 lives in a combined shader (no standalone file), so compile the
        // FULL codec metallib and fetch the kernel by name — the same lib the package ships.
        let metallib = FileManager.default.temporaryDirectory
            .appendingPathComponent("codec-c1-\(getpid()).metallib")
        try SmeltCompiler.compileMetalLibrary(shaderDir: "Resources/Shaders", outputPath: metallib.path)
        defer { try? FileManager.default.removeItem(at: metallib) }
        guard let fn = try device.makeLibrary(URL: metallib).makeFunction(name: "conv1d_forward_f32") else {
            throw XCTSkip("no conv1d_forward_f32 function in the metallib")
        }
        let pipe = try device.makeComputePipelineState(function: fn)
        // A fixed conv shape (every constant emit-time literal). Feed an already-"padded" input so
        // the micro tests the conv alone (the channel_copy pad is the next slice).
        let (cIn, cOut, k, stride, dilation, groups) = (8, 8, 3, 1, 1, 1)
        let paddedLen = 16
        let kEff = (k - 1) * dilation + 1
        let lengthOut = (paddedLen - kEff) / stride + 1
        let outCount = cOut * lengthOut

        var rng: UInt64 = 0xC0DEC1
        func rnd(_ n: Int) -> [Float] {
            (0..<n).map { _ in
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                return Float((rng >> 33) & 0xFFFF) / Float(0x10000) * 0.2 - 0.1
            }
        }
        let pBuf = try makeSharedBuffer(device: device, rnd(cIn * paddedLen))
        let wBuf = try makeSharedBuffer(device: device, rnd(cOut * cIn * k))
        let bBuf = try makeSharedBuffer(device: device, rnd(cOut))
        let outManual = device.makeBuffer(length: outCount * 4, options: .storageModeShared)!
        let outRecord = device.makeBuffer(length: outCount * 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let grid = MTLSize(width: lengthOut, height: cOut, depth: 1)
        let tg = MTLSize(width: min(lengthOut, 32), height: 1, depth: 1)
        // The conv1d_forward_f32 constant block (mirrors conv1dForwardEncode): cIn,cOut,k,stride,
        // pad=0,dilation,groups, lengthIn packed with the high bit.
        let consts: [UInt32] = [
            UInt32(cIn), UInt32(cOut), UInt32(k), UInt32(stride),
            0, UInt32(dilation), UInt32(groups), UInt32(paddedLen) | 0x8000_0000,
        ]

        // (1) Manual dispatch — the hand way.
        let cb1 = queue.makeCommandBuffer()!, enc1 = cb1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(pipe)
        enc1.setBuffer(pBuf, offset: 0, index: 0); enc1.setBuffer(wBuf, offset: 0, index: 1)
        enc1.setBuffer(bBuf, offset: 0, index: 2); enc1.setBuffer(outManual, offset: 0, index: 3)
        for (i, c) in consts.enumerated() { var v = c; enc1.setBytes(&v, length: 4, index: 4 + i) }
        enc1.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

        // (2) Record-driven dispatch — emit a record (generic SmeltDispatch), run via the seam.
        // buffers slots: 0=pBuf, 1=wBuf, 2=bBuf, 3=outRecord.
        let rec = SmeltDispatch(
            pipeline: .conv1dForwardF32,
            buffers: [
                SmeltBufferBinding(slot: 0, index: 0),
                SmeltBufferBinding(slot: 1, index: 1),
                SmeltBufferBinding(slot: 2, index: 2),
                SmeltBufferBinding(slot: 3, index: 3),
            ],
            constants: consts.enumerated().map {
                SmeltConstantBinding(expression: "\($0.element)", type: .uint32, index: 4 + $0.offset)
            },
            dispatch: .threads(width: lengthOut, height: cOut, depth: 1,
                               tgWidth: min(lengthOut, 32), tgHeight: 1, tgDepth: 1),
            comment: "codec conv1d micro"
        ).toRecord()

        // Pipelines indexed by record.pipeline (= conv1dForwardF32.rawValue). Pad with the same
        // pipe at lower indices (never dispatched — only the conv index is referenced).
        let convIdx = Int(SmeltPipeline.conv1dForwardF32.rawValue)
        let pipelines = [MTLComputePipelineState](repeating: pipe, count: convIdx + 1)

        let cb2 = queue.makeCommandBuffer()!, enc2 = cb2.makeComputeCommandEncoder()!
        SmeltCodecRecordRunner.encode(
            [rec], pipelines: pipelines, buffers: [pBuf, wBuf, bBuf, outRecord], into: enc2)
        enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

        let m = Array(UnsafeBufferPointer(
            start: outManual.contents().bindMemory(to: Float.self, capacity: outCount), count: outCount))
        let r = Array(UnsafeBufferPointer(
            start: outRecord.contents().bindMemory(to: Float.self, capacity: outCount), count: outCount))
        XCTAssertTrue(m.allSatisfy { $0.isFinite } && m.contains { $0 != 0 },
                      "conv1d output not finite+nonzero — gate vacuous")
        // Byte-identical: compare bit patterns (Float == treats +0.0 == -0.0, hiding a sign-of-zero split).
        XCTAssertEqual(m.map(\.bitPattern), r.map(\.bitPattern),
                       "record-driven conv1d != manual dispatch of the same kernel (bit-level)")
    }
}
