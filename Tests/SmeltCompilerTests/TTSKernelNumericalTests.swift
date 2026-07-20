// TTSKernelNumericalTests — GPU-vs-CPU numerical gates for the extracted Kokoro
// codec kernels (snake, layer_norm, conv1d_forward, conv_transpose1d).
//
// Each test builds known FP16 inputs, computes a plain-Swift CPU reference, runs
// the Metal kernel on the GPU, and asserts they agree. Self-contained — no Python
// or ONNX dependency (the kernels were ONNX-gated upstream; these gates prove the
// extraction onto current main is numerically faithful, and in conv1d's case that
// the out_stride=L_out modification is correct for strided convs).

import Metal
import XCTest

@testable import SmeltCompiler

final class TTSKernelNumericalTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        device = d
        queue = device.makeCommandQueue()
    }

    private func pipeline(_ shaderFile: String, _ fn: String) throws -> MTLComputePipelineState {
        guard let source = loadMetalShaderSource(shaderFile) else {
            throw XCTSkip("Shader not found: \(shaderFile)")
        }
        let lib = try device.makeLibrary(source: source, options: nil)
        let function = try XCTUnwrap(lib.makeFunction(name: fn))
        return try device.makeComputePipelineState(function: function)
    }

    private func buffer(_ values: [Float]) -> MTLBuffer {
        let h = values.map { Float16($0) }
        return device.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared)!
    }

    private func outBuffer(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: count * 2, options: .storageModeShared)!
        memset(b.contents(), 0, count * 2)
        return b
    }

    private func read(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private func assertClose(_ gpu: [Float], _ ref: [Float], tol: Float = 0.03,
                             _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(gpu.count, ref.count, "\(label): count", file: file, line: line)
        var maxDiff: Float = 0
        for i in 0..<gpu.count {
            // Guard explicitly: Swift.max(finite, .nan) returns the finite value, so
            // a NaN/Inf kernel output would otherwise slip past the tolerance check —
            // the exact "NaN audio" failure mode this gate exists to catch.
            XCTAssertTrue(gpu[i].isFinite, "\(label): non-finite gpu[\(i)]=\(gpu[i])",
                          file: file, line: line)
            maxDiff = max(maxDiff, abs(gpu[i] - ref[i]))
        }
        XCTAssertLessThan(maxDiff, tol, "\(label): maxDiff \(maxDiff)\n gpu=\(gpu)\n ref=\(ref)",
                          file: file, line: line)
    }

    // MARK: - snake

    func testSnakeActivationNumerical() throws {
        let pso = try pipeline("snake.metal", "snake_activation")
        let channels = 4, length = 8
        let alpha = (0..<channels).map { 0.5 + Float($0) * 0.3 }
        let input = (0..<channels * length).map { sin(Float($0) * 0.3) * 1.5 }

        var ref = [Float](repeating: 0, count: channels * length)
        for c in 0..<channels {
            let a = alpha[c]
            for l in 0..<length {
                let x = input[c * length + l]
                let s = sin(a * x)
                ref[c * length + l] = x + (s * s) / a
            }
        }

        let inBuf = buffer(input), aBuf = buffer(alpha), outBuf = outBuffer(channels * length)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var ch = UInt32(channels), ln = UInt32(length)
        enc.setBytes(&ch, length: 4, index: 3)
        enc.setBytes(&ln, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: length, height: channels, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: length, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertClose(read(outBuf, channels * length), ref, "snake")
    }

    // MARK: - layer_norm

    func testLayerNormNumerical() throws {
        let pso = try pipeline("layer_norm.metal", "layer_norm")
        let rows = 3, dim = 16
        let eps: Float = 1e-5
        let weight = (0..<dim).map { 0.8 + Float($0) * 0.02 }
        let bias = (0..<dim).map { Float($0) * 0.01 - 0.05 }
        let input = (0..<rows * dim).map { cos(Float($0) * 0.2) * 2.0 }

        var ref = [Float](repeating: 0, count: rows * dim)
        for r in 0..<rows {
            let base = r * dim
            var mean: Float = 0
            for i in 0..<dim { mean += input[base + i] }
            mean /= Float(dim)
            var varr: Float = 0
            for i in 0..<dim { let d = input[base + i] - mean; varr += d * d }
            varr /= Float(dim)
            let invStd = 1.0 / (varr + eps).squareRoot()
            for i in 0..<dim {
                ref[base + i] = (input[base + i] - mean) * invStd * weight[i] + bias[i]
            }
        }

        let inBuf = buffer(input), wBuf = buffer(weight), bBuf = buffer(bias)
        let outBuf = outBuffer(rows * dim)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var dimV = UInt32(dim), epsV = eps
        enc.setBytes(&dimV, length: 4, index: 4)
        enc.setBytes(&epsV, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: dim, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertClose(read(outBuf, rows * dim), ref, "layer_norm")
    }

    // MARK: - conv1d_forward (strided: L_out != L_in, validates out_stride=L_out)

    func testConv1dForwardNumericalStrided() throws {
        let pso = try pipeline("conv1d_spatial.metal", "conv1d_forward")
        let cIn = 2, cOut = 2, k = 3, stride = 2, padding = 1, dilation = 1, groups = 1
        let lIn = 6
        let lOut = (lIn + 2 * padding - dilation * (k - 1) - 1) / stride + 1  // = 3
        XCTAssertEqual(lOut, 3)

        let input = (0..<cIn * lIn).map { sin(Float($0) * 0.4) }
        let weight = (0..<cOut * cIn * k).map { cos(Float($0) * 0.3) * 0.5 }
        let bias = (0..<cOut).map { Float($0) * 0.1 + 0.05 }

        // CPU reference: standard grouped conv, output row stride = L_out.
        var ref = [Float](repeating: 0, count: cOut * lOut)
        let gsIn = cIn / groups, gsOut = cOut / groups
        for oc in 0..<cOut {
            let g = oc / gsOut
            for ol in 0..<lOut {
                var acc = bias[oc]
                for ic in 0..<gsIn {
                    let icGlobal = g * gsIn + ic
                    for kk in 0..<k {
                        let il = ol * stride - padding + kk * dilation
                        if il >= 0 && il < lIn {
                            acc += weight[oc * gsIn * k + ic * k + kk] * input[icGlobal * lIn + il]
                        }
                    }
                }
                ref[oc * lOut + ol] = acc
            }
        }

        let inBuf = buffer(input), wBuf = buffer(weight), bBuf = buffer(bias)
        let outBuf = outBuffer(cOut * lOut)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var cInV = UInt32(cIn), cOutV = UInt32(cOut), kV = UInt32(k), strideV = UInt32(stride)
        var padV = UInt32(padding), dilV = UInt32(dilation), grpV = UInt32(groups)
        var lInPacked = UInt32(lIn) | 0x8000_0000  // has_bias = true
        enc.setBytes(&cInV, length: 4, index: 4)
        enc.setBytes(&cOutV, length: 4, index: 5)
        enc.setBytes(&kV, length: 4, index: 6)
        enc.setBytes(&strideV, length: 4, index: 7)
        enc.setBytes(&padV, length: 4, index: 8)
        enc.setBytes(&dilV, length: 4, index: 9)
        enc.setBytes(&grpV, length: 4, index: 10)
        enc.setBytes(&lInPacked, length: 4, index: 11)
        enc.dispatchThreads(MTLSize(width: lOut, height: cOut, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: lOut, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertClose(read(outBuf, cOut * lOut), ref, "conv1d_forward strided")
    }

    // MARK: - conv_transpose1d

    func testConvTranspose1dNumerical() throws {
        let pso = try pipeline("conv_transpose1d.metal", "conv_transpose1d")
        let cIn = 2, cOut = 2, k = 4, stride = 2, padding = 1, lIn = 3
        let lOut = (lIn - 1) * stride - 2 * padding + k  // = 6
        XCTAssertEqual(lOut, 6)

        let input = (0..<cIn * lIn).map { sin(Float($0) * 0.5) }
        let weight = (0..<cIn * cOut * k).map { cos(Float($0) * 0.25) * 0.5 }  // [C_in, C_out, K]
        let bias = (0..<cOut).map { Float($0) * 0.1 }

        var ref = [Float](repeating: 0, count: cOut * lOut)
        for oc in 0..<cOut {
            for ol in 0..<lOut {
                var acc = bias[oc]
                for ic in 0..<cIn {
                    for kk in 0..<k {
                        let numerator = ol + padding - kk
                        if numerator >= 0 && numerator % stride == 0 {
                            let il = numerator / stride
                            if il < lIn {
                                acc += weight[ic * cOut * k + oc * k + kk] * input[ic * lIn + il]
                            }
                        }
                    }
                }
                ref[oc * lOut + ol] = acc
            }
        }

        let inBuf = buffer(input), wBuf = buffer(weight), bBuf = buffer(bias)
        let outBuf = outBuffer(cOut * lOut)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var cInV = UInt32(cIn), cOutV = UInt32(cOut), kV = UInt32(k)
        var strideV = UInt32(stride), padV = UInt32(padding), lInV = UInt32(lIn)
        enc.setBytes(&cInV, length: 4, index: 4)
        enc.setBytes(&cOutV, length: 4, index: 5)
        enc.setBytes(&kV, length: 4, index: 6)
        enc.setBytes(&strideV, length: 4, index: 7)
        enc.setBytes(&padV, length: 4, index: 8)
        enc.setBytes(&lInV, length: 4, index: 9)
        enc.dispatchThreads(MTLSize(width: lOut, height: cOut, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: lOut, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertClose(read(outBuf, cOut * lOut), ref, "conv_transpose1d")
    }
}
