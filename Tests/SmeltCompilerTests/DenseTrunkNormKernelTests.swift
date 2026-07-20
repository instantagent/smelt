// DenseTrunkNormKernelTests — raw-byte gates proving that authoritative
// BF16 norm scales take the same trunk math path as their exact FP32 widening.

import Metal
import XCTest

@testable import SmeltCompiler

final class DenseTrunkNormKernelTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device?.makeCommandQueue()
    }

    private func pipelines(
        shaderFile: String,
        fp32Name: String,
        bf16Name: String
    ) throws -> (MTLComputePipelineState, MTLComputePipelineState) {
        guard let source = loadMetalShaderSource(shaderFile) else {
            throw XCTSkip("Shader source not found: \(shaderFile)")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        let fp32 = try XCTUnwrap(library.makeFunction(name: fp32Name))
        let bf16 = try XCTUnwrap(library.makeFunction(name: bf16Name))
        return (
            try device.makeComputePipelineState(function: fp32),
            try device.makeComputePipelineState(function: bf16)
        )
    }

    private func values(count: Int, seed: Int) -> [Float] {
        (0..<count).map { index in
            let x = Float(index &* 73 &+ seed &* 109) * 0.0073
            return sin(x) * 0.71 + cos(x * 0.37) * 0.19
        }
    }

    private func bf16Scale(count: Int, seed: Int) -> (bits: [UInt16], widened: [Float]) {
        let source = values(count: count, seed: seed).map { $0 + 1.25 }
        let bits = source.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        return (bits, bits.map { Float(bitPattern: UInt32($0) << 16) })
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
    }

    private func assertRawEqual(
        _ actual: MTLBuffer,
        _ expected: MTLBuffer,
        count: Int,
        label: String
    ) {
        let actualValues = read(actual, count: count)
        let expectedValues = read(expected, count: count)
        XCTAssertTrue(actualValues.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(actualValues.map(abs).max() ?? 0, 1e-3)
        for index in 0..<count {
            XCTAssertEqual(
                actualValues[index].bitPattern,
                expectedValues[index].bitPattern,
                "\(label) byte divergence at scalar \(index)"
            )
        }
    }

    func testRowRMSNormBF16ScaleMatchesExactFP32Widening() throws {
        let (fp32, bf16) = try pipelines(
            shaderFile: "rms_norm_codec_f32.metal",
            fp32Name: "rms_norm_codec_f32",
            bf16Name: "rms_norm_codec_bf16w_f32"
        )
        let frames = 3
        let dim = 896
        let count = frames * dim
        let input = try makeSharedBuffer(device: device, values(count: count, seed: 17))
        let scale = bf16Scale(count: dim, seed: 31)
        let fp32Scale = try makeSharedBuffer(device: device, scale.widened)
        let bf16Scale = try makeSharedBuffer(device: device, scale.bits)
        let expected = try makeSharedBuffer(device: device, count: count, of: Float.self)
        let actual = try makeSharedBuffer(device: device, count: count, of: Float.self)

        try runOnGPU(queue: queue) { encoder in
            func encode(
                _ pipeline: MTLComputePipelineState,
                scale: MTLBuffer,
                output: MTLBuffer
            ) {
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setBuffer(scale, offset: 0, index: 1)
                encoder.setBuffer(output, offset: 0, index: 2)
                var frameCount = UInt32(frames)
                var width = UInt32(dim)
                var epsilon: Float = 1e-6
                encoder.setBytes(&frameCount, length: 4, index: 3)
                encoder.setBytes(&width, length: 4, index: 4)
                encoder.setBytes(&epsilon, length: 4, index: 5)
                encoder.dispatchThreadgroups(
                    MTLSize(width: frames, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
            encode(fp32, scale: fp32Scale, output: expected)
            encode(bf16, scale: bf16Scale, output: actual)
        }
        assertRawEqual(actual, expected, count: count, label: "row RMSNorm")
    }

    func testHeadRMSNormBF16ScaleMatchesExactFP32Widening() throws {
        let (fp32, bf16) = try pipelines(
            shaderFile: "rms_norm_head_f32.metal",
            fp32Name: "rms_norm_head_f32",
            bf16Name: "rms_norm_head_bf16w_f32"
        )
        let frames = 3
        let heads = 8
        let headDim = 128
        let count = frames * heads * headDim
        let input = try makeSharedBuffer(device: device, values(count: count, seed: 47))
        let scale = bf16Scale(count: headDim, seed: 59)
        let fp32Scale = try makeSharedBuffer(device: device, scale.widened)
        let bf16Scale = try makeSharedBuffer(device: device, scale.bits)
        let expected = try makeSharedBuffer(device: device, count: count, of: Float.self)
        let actual = try makeSharedBuffer(device: device, count: count, of: Float.self)

        try runOnGPU(queue: queue) { encoder in
            func encode(
                _ pipeline: MTLComputePipelineState,
                scale: MTLBuffer,
                output: MTLBuffer
            ) {
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setBuffer(scale, offset: 0, index: 1)
                encoder.setBuffer(output, offset: 0, index: 2)
                var frameCount = UInt32(frames)
                var headCount = UInt32(heads)
                var width = UInt32(headDim)
                var epsilon: Float = 1e-6
                encoder.setBytes(&frameCount, length: 4, index: 3)
                encoder.setBytes(&headCount, length: 4, index: 4)
                encoder.setBytes(&width, length: 4, index: 5)
                encoder.setBytes(&epsilon, length: 4, index: 6)
                encoder.dispatchThreadgroups(
                    MTLSize(width: frames * heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
            encode(fp32, scale: fp32Scale, output: expected)
            encode(bf16, scale: bf16Scale, output: actual)
        }
        assertRawEqual(actual, expected, count: count, label: "head RMSNorm")
    }

    func testFusedHeadNormRoPEBF16ScaleMatchesExactFP32Widening() throws {
        let (fp32, bf16) = try pipelines(
            shaderFile: "head_norm_rope_f32.metal",
            fp32Name: "head_norm_rope_f32",
            bf16Name: "head_norm_rope_bf16w_f32"
        )
        let heads = 16
        let headDim = 128
        let count = heads * headDim
        let input = try makeSharedBuffer(device: device, values(count: count, seed: 71))
        let scale = bf16Scale(count: headDim, seed: 83)
        let fp32Scale = try makeSharedBuffer(device: device, scale.widened)
        let bf16Scale = try makeSharedBuffer(device: device, scale.bits)
        let angles = (0..<headDim).map { Float($0) * 0.0037 }
        let cosine = try makeSharedBuffer(device: device, angles.map(cos))
        let sine = try makeSharedBuffer(device: device, angles.map(sin))
        let expected = try makeSharedBuffer(device: device, count: count, of: Float.self)
        let actual = try makeSharedBuffer(device: device, count: count, of: Float.self)

        try runOnGPU(queue: queue) { encoder in
            func encode(
                _ pipeline: MTLComputePipelineState,
                scale: MTLBuffer,
                output: MTLBuffer
            ) {
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setBuffer(scale, offset: 0, index: 1)
                encoder.setBuffer(cosine, offset: 0, index: 2)
                encoder.setBuffer(sine, offset: 0, index: 3)
                encoder.setBuffer(output, offset: 0, index: 4)
                var headCount = UInt32(heads)
                var width = UInt32(headDim)
                var epsilon: Float = 1e-6
                encoder.setBytes(&headCount, length: 4, index: 5)
                encoder.setBytes(&width, length: 4, index: 6)
                encoder.setBytes(&epsilon, length: 4, index: 7)
                encoder.dispatchThreadgroups(
                    MTLSize(width: heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
            encode(fp32, scale: fp32Scale, output: expected)
            encode(bf16, scale: bf16Scale, output: actual)
        }
        assertRawEqual(actual, expected, count: count, label: "fused head norm+RoPE")
    }
}
