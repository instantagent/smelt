import Metal
import XCTest

@testable import SmeltCompiler

final class RecurrentVerifyCheckpointTests: XCTestCase {
    func testDeltaNetCheckpointFeedsRoundedStateBetweenVerifiedTokens() throws {
        let headDim = 32
        let valueHeads = 1
        let qkHeads = 1
        let tokenCount = 4
        let stateCount = valueHeads * headDim * headDim
        let hiddenStride = (2 * qkHeads + valueHeads) * headDim

        let initialState = (0..<stateCount).map {
            Float16(sin(Float($0 * 17 + 3) * 0.013) * 0.35)
        }
        let qkv = (0..<(tokenCount * hiddenStride)).map {
            Float16(cos(Float($0 * 29 + 11) * 0.017) * 0.55)
        }
        let bProjection = (0..<(tokenCount * valueHeads)).map {
            Float16(sin(Float($0 * 7 + 5) * 0.19) * 0.4)
        }
        let aProjection = (0..<(tokenCount * valueHeads)).map {
            Float16(cos(Float($0 * 5 + 2) * 0.23) * 0.35)
        }
        let aLog = (0..<valueHeads).map { Float16(-1.5 - Float($0) * 0.125) }
        let dtBias = (0..<valueHeads).map { Float16(0.15 + Float($0) * 0.03125) }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "prefill_recurrence.metal",
            functionName: "deltanet_recurrence_mlx_prefill_checkpoint"
        ))

        func execute(
            state: MTLBuffer,
            qkv: [Float16],
            bProjection: [Float16],
            aProjection: [Float16],
            tokenCount: Int
        ) throws -> (output: MTLBuffer, history: MTLBuffer) {
            let qkvBuffer = try makeSharedBuffer(device: device, qkv)
            let bBuffer = try makeSharedBuffer(device: device, bProjection)
            let aBuffer = try makeSharedBuffer(device: device, aProjection)
            let aLogBuffer = try makeSharedBuffer(device: device, aLog)
            let dtBiasBuffer = try makeSharedBuffer(device: device, dtBias)
            let output = try makeSharedBuffer(
                device: device,
                count: tokenCount * valueHeads * headDim,
                of: Float16.self
            )
            let history = try makeSharedBuffer(
                device: device,
                count: (tokenCount + 1) * stateCount,
                of: Float16.self
            )
            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(state, offset: 0, index: 0)
                encoder.setBuffer(qkvBuffer, offset: 0, index: 1)
                encoder.setBuffer(bBuffer, offset: 0, index: 2)
                encoder.setBuffer(aBuffer, offset: 0, index: 3)
                encoder.setBuffer(aLogBuffer, offset: 0, index: 4)
                encoder.setBuffer(dtBiasBuffer, offset: 0, index: 5)
                encoder.setBuffer(output, offset: 0, index: 6)
                var d = UInt32(headDim)
                var hv = UInt32(valueHeads)
                var hqk = UInt32(qkHeads)
                var count = UInt32(tokenCount)
                encoder.setBytes(&d, length: 4, index: 7)
                encoder.setBytes(&hv, length: 4, index: 8)
                encoder.setBytes(&hqk, length: 4, index: 9)
                encoder.setBytes(&count, length: 4, index: 10)
                encoder.setBuffer(history, offset: 0, index: 11)
                encoder.dispatchThreads(
                    MTLSize(width: 32, height: headDim, depth: valueHeads),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
                )
            }
            return (output, history)
        }

        func bits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
            Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: UInt16.self),
                count: count
            ))
        }

        let batchedState = try makeSharedBuffer(device: device, initialState)
        let batched = try execute(
            state: batchedState,
            qkv: qkv,
            bProjection: bProjection,
            aProjection: aProjection,
            tokenCount: tokenCount
        )

        let sequentialState = try makeSharedBuffer(device: device, initialState)
        var sequentialOutput: [UInt16] = []
        var sequentialHistory = initialState.map(\.bitPattern)
        for token in 0..<tokenCount {
            let qkvStart = token * hiddenStride
            let row = try execute(
                state: sequentialState,
                qkv: Array(qkv[qkvStart..<(qkvStart + hiddenStride)]),
                bProjection: Array(
                    bProjection[(token * valueHeads)..<((token + 1) * valueHeads)]),
                aProjection: Array(
                    aProjection[(token * valueHeads)..<((token + 1) * valueHeads)]),
                tokenCount: 1
            )
            sequentialOutput += bits(
                row.output, count: valueHeads * headDim)
            sequentialHistory += bits(sequentialState, count: stateCount)
        }

        XCTAssertEqual(
            bits(batched.output, count: tokenCount * valueHeads * headDim),
            sequentialOutput
        )
        XCTAssertEqual(
            bits(batched.history, count: (tokenCount + 1) * stateCount),
            sequentialHistory
        )
        XCTAssertEqual(
            bits(batchedState, count: stateCount),
            bits(sequentialState, count: stateCount)
        )
    }
}
