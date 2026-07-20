import Metal
import XCTest

final class RMSScaleQKKernelTests: XCTestCase {
    func testRMSScaleQKMatchesExplicitGraphEdgesAndPreservesValueRegion() throws {
        let headDim = 128
        let qkHeads = 2
        let positions = 3
        let qkWidth = qkHeads * headDim
        let qkvDim = qkWidth * 3
        let epsilon: Float = 1e-6
        let input = (0..<(positions * qkvDim)).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms.metal",
            functionName: "rms_scale_qk"
        ))
        let values = try makeSharedBuffer(device: device, input)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(values, offset: 0, index: 0)
            var headDimValue = UInt32(headDim)
            var epsilonValue = epsilon
            var qkvDimValue = UInt32(qkvDim)
            var qkHeadsValue = UInt32(qkHeads)
            encoder.setBytes(&headDimValue, length: 4, index: 1)
            encoder.setBytes(&epsilonValue, length: 4, index: 2)
            encoder.setBytes(&qkvDimValue, length: 4, index: 3)
            encoder.setBytes(&qkHeadsValue, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 2 * qkHeads, height: positions, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let output = UnsafeBufferPointer(
            start: values.contents().assumingMemoryBound(to: Float16.self),
            count: input.count
        )
        for position in 0..<positions {
            let positionBase = position * qkvDim
            for region in 0..<2 {
                let edgeScale = region == 0
                    ? Float16(1.0 / Float(headDim))
                    : Float16(1.0 / sqrt(Float(headDim)))
                for head in 0..<qkHeads {
                    let base = positionBase + region * qkWidth + head * headDim
                    var sumSquares: Float = 0
                    for index in 0..<headDim {
                        let value = Float(input[base + index])
                        sumSquares += value * value
                    }
                    let rmsScale = 1.0 / sqrt(sumSquares / Float(headDim) + epsilon)
                    for index in 0..<headDim {
                        let normalized = Float16(Float(input[base + index]) * rmsScale)
                        let expected = normalized * edgeScale
                        XCTAssertEqual(
                            Float(output[base + index]),
                            Float(expected),
                            accuracy: 0.000_02
                        )
                    }
                }
            }

            let valueBase = positionBase + 2 * qkWidth
            for index in 0..<qkWidth {
                XCTAssertEqual(
                    output[valueBase + index].bitPattern,
                    input[valueBase + index].bitPattern
                )
            }
        }
    }
}
