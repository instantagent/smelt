// Qwen3TTSCodecGPUTests — fp32 GPU-kernel-vs-CPU-reference parity gates for the codec's
// fp32 leaf kernels. Each test dispatches the fp32
// Metal kernel and asserts it matches the corresponding Qwen3TTSCodec CPU op (which is
// itself gated against the real model) at cosine >= 0.999 AND relL2 <= 0.02 + finiteness.

import Foundation
import Metal
import Accelerate
import XCTest
@testable import SmeltRuntime
import SmeltCompiler
import SmeltSchema

final class Qwen3TTSCodecGPUTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    private func env(_ k: String) -> String? {
        guard let v = ProcessInfo.processInfo.environment[k], !v.isEmpty else { return nil }
        return v
    }
    private func readBinF32(_ path: String) -> [Float] {
        let d = try! Data(contentsOf: URL(fileURLWithPath: path))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    private func readBinI32(_ path: String) -> [Int32] {
        let d = try! Data(contentsOf: URL(fileURLWithPath: path))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let d = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal device available") }
        device = d
        queue = device.makeCommandQueue()
    }

    private func pipeline(_ shaderFile: String, _ fn: String) throws -> MTLComputePipelineState {
        // Only a genuinely-absent shader source is a skip; a compile/link failure of a
        // present shader is a hard failure (a silent skip there masks a real kernel bug —
        // e.g. an unsupported Metal builtin like erf()).
        guard loadMetalShaderSource(shaderFile) != nil else { throw XCTSkip("Shader not found: \(shaderFile)") }
        return try XCTUnwrap(makeComputePipeline(device: device, shaderFile: shaderFile, functionName: fn),
                             "pipeline build failed for \(shaderFile):\(fn) (compile/link error?)")
    }

    private func bufF32(_ v: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared)!
    }
    private func outF32(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        memset(b.contents(), 0, count * 4)
        return b
    }
    private func readF32(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { ptr[$0] }
    }
    private func bufI32(_ v: [Int32]) -> MTLBuffer {
        device.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared)!
    }

    private func assertMatch(_ gpu: [Float], _ ref: [Float], _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(gpu.count, ref.count, "\(label): count", file: file, line: line)
        var dot: Float = 0, na: Float = 0, nb: Float = 0, diff: Float = 0
        for i in 0..<gpu.count {
            XCTAssertTrue(gpu[i].isFinite, "\(label): non-finite gpu[\(i)]=\(gpu[i])", file: file, line: line)
            dot += gpu[i] * ref[i]; na += gpu[i] * gpu[i]; nb += ref[i] * ref[i]
            let d = gpu[i] - ref[i]; diff += d * d
        }
        let cos = dot / (na.squareRoot() * nb.squareRoot())
        let relL2 = diff.squareRoot() / nb.squareRoot()
        XCTAssertGreaterThanOrEqual(cos, 0.999, "\(label): cosine \(cos)", file: file, line: line)
        XCTAssertLessThanOrEqual(relL2, 0.02, "\(label): relL2 \(relL2)", file: file, line: line)
    }

    // MARK: - U1: SnakeBeta fp32

    func testSnakeBetaF32MatchesCPU() throws {
        let pso = try pipeline("snake_beta_f32.metal", "snake_beta_f32")
        let dim = 8, frames = 16
        // exp'd in the kernel, so keep alpha/beta in a realistic small range.
        let alpha = (0..<dim).map { -0.4 + Float($0) * 0.1 }
        let beta = (0..<dim).map { 0.6 - Float($0) * 0.15 }
        let input = (0..<dim * frames).map { sin(Float($0) * 0.37) * 1.8 }

        let ref = Qwen3TTSCodec.snakeBeta(input, dim: dim, frames: frames,
                                          .init(alpha: alpha, beta: beta))

        let inBuf = bufF32(input), aBuf = bufF32(alpha), bBuf = bufF32(beta)
        let outBuf = outF32(dim * frames)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var ch = UInt32(dim), ln = UInt32(frames)
        enc.setBytes(&ch, length: 4, index: 4)
        enc.setBytes(&ln, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: frames, height: dim, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: frames, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertMatch(readF32(outBuf, dim * frames), ref, "snake_beta_f32")
    }

    // MARK: - U2: conv1d fp32

    /// Run conv1d_forward_f32 with the CPU's exact causal pre-pad (padding=0 in-kernel) and
    /// compare to the full Qwen3TTSCodec.causalConv1d output.
    private func runConv1dGate(cIn: Int, cOut: Int, lengthIn: Int, kernel: Int,
                               stride: Int, dilation: Int, groups: Int, label: String) throws {
        let pso = try pipeline("conv1d_spatial_f32.metal", "conv1d_forward_f32")
        let gsIn = cIn / groups
        let input = (0..<cIn * lengthIn).map { sin(Float($0) * 0.21) * 1.3 }
        let weight = (0..<cOut * gsIn * kernel).map { cos(Float($0) * 0.17) * 0.4 }
        let bias = (0..<cOut).map { Float($0) * 0.03 - 0.1 }
        let ref = Qwen3TTSCodec.causalConv1d(
            input: input, inChannels: cIn, lengthIn: lengthIn,
            weight: weight, bias: bias, outChannels: cOut,
            kernel: kernel, stride: stride, dilation: dilation, groups: groups)

        // Replicate the CPU causal pad/extra into a [cIn, paddedLen] buffer; dispatch padding=0.
        let kEff = (kernel - 1) * dilation + 1
        let padding = kEff - stride
        let nFrames = Int((Double(lengthIn - kEff + padding) / Double(stride)).rounded(.up)) + 1
        let idealLength = (nFrames - 1) * stride + (kEff - padding)
        let extra = max(0, idealLength - lengthIn)
        let paddedLen = lengthIn + padding + extra
        var padded = [Float](repeating: 0, count: cIn * paddedLen)
        for c in 0..<cIn {
            for t in 0..<lengthIn { padded[c * paddedLen + padding + t] = input[c * lengthIn + t] }
        }
        let lengthOut = (paddedLen - kEff) / stride + 1
        XCTAssertEqual(lengthOut, ref.lengthOut, "\(label): lengthOut")

        let inBuf = bufF32(padded), wBuf = bufF32(weight), bBuf = bufF32(bias)
        let outBuf = outF32(cOut * lengthOut)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var cInU = UInt32(cIn), cOutU = UInt32(cOut), kU = UInt32(kernel), sU = UInt32(stride)
        var padU = UInt32(0), dilU = UInt32(dilation), gU = UInt32(groups)
        var lInPacked = UInt32(paddedLen) | 0x80000000  // has_bias high bit set
        enc.setBytes(&cInU, length: 4, index: 4); enc.setBytes(&cOutU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&sU, length: 4, index: 7)
        enc.setBytes(&padU, length: 4, index: 8); enc.setBytes(&dilU, length: 4, index: 9)
        enc.setBytes(&gU, length: 4, index: 10); enc.setBytes(&lInPacked, length: 4, index: 11)
        enc.dispatchThreads(MTLSize(width: lengthOut, height: cOut, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(lengthOut, 32), height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertMatch(readF32(outBuf, cOut * lengthOut), ref.out, label)
    }

    func testConv1dF32MatchesCPU() throws {
        // Dense groups=1: pre_conv-like (k3 s1), DAC final (k7 s1), residual-unit dilations.
        try runConv1dGate(cIn: 12, cOut: 16, lengthIn: 24, kernel: 3, stride: 1, dilation: 1, groups: 1, label: "conv dense k3")
        try runConv1dGate(cIn: 8, cOut: 8, lengthIn: 24, kernel: 7, stride: 1, dilation: 1, groups: 1, label: "conv dense k7")
        try runConv1dGate(cIn: 6, cOut: 6, lengthIn: 30, kernel: 3, stride: 1, dilation: 3, groups: 1, label: "conv dense dil3")
        try runConv1dGate(cIn: 6, cOut: 6, lengthIn: 40, kernel: 3, stride: 1, dilation: 9, groups: 1, label: "conv dense dil9")
        // Depthwise groups=dim (ConvNeXt dwconv).
        try runConv1dGate(cIn: 10, cOut: 10, lengthIn: 24, kernel: 7, stride: 1, dilation: 1, groups: 10, label: "conv depthwise k7")
    }

    // MARK: - U3: convTranspose1d fp32

    /// Dispatch conv_transpose1d_f32 (full output), right-trim by (kernel-stride), compare
    /// to Qwen3TTSCodec.causalTransConv1d.
    private func runTransConvGate(cIn: Int, cOut: Int, lengthIn: Int, kernel: Int,
                                  stride: Int, label: String) throws {
        let pso = try pipeline("conv_transpose1d_f32.metal", "conv_transpose1d_f32")
        let input = (0..<cIn * lengthIn).map { sin(Float($0) * 0.29) * 1.1 }
        let weight = (0..<cIn * cOut * kernel).map { cos(Float($0) * 0.13) * 0.5 }
        let bias = (0..<cOut).map { Float($0) * 0.02 - 0.05 }
        let ref = Qwen3TTSCodec.causalTransConv1d(
            input: input, cIn: cIn, lengthIn: lengthIn, weight: weight, bias: bias,
            cOut: cOut, kernel: kernel, stride: stride)

        let fullLen = (lengthIn - 1) * stride + kernel
        let inBuf = bufF32(input), wBuf = bufF32(weight), bBuf = bufF32(bias)
        let outBuf = outF32(cOut * fullLen)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
        var cInU = UInt32(cIn), cOutU = UInt32(cOut), kU = UInt32(kernel)
        var sU = UInt32(stride), padU = UInt32(0), lInU = UInt32(lengthIn)
        enc.setBytes(&cInU, length: 4, index: 4); enc.setBytes(&cOutU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&sU, length: 4, index: 7)
        enc.setBytes(&padU, length: 4, index: 8); enc.setBytes(&lInU, length: 4, index: 9)
        enc.dispatchThreads(MTLSize(width: fullLen, height: cOut, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(fullLen, 32), height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        // Right-trim by (kernel - stride) to match the CPU causal trim.
        let full = readF32(outBuf, cOut * fullLen)
        let trim = max(0, kernel - stride)
        let lengthOut = fullLen - trim
        XCTAssertEqual(lengthOut, ref.lengthOut, "\(label): lengthOut")
        var trimmed = [Float](repeating: 0, count: cOut * lengthOut)
        for oc in 0..<cOut { for j in 0..<lengthOut { trimmed[oc * lengthOut + j] = full[oc * fullLen + j] } }
        assertMatch(trimmed, ref.out, label)
    }

    func testConvTranspose1dF32MatchesCPU() throws {
        // DAC upsample rates: kernel = 2*rate, stride = rate (trim = rate).
        for rate in [8, 5, 4, 3] {
            try runTransConvGate(cIn: 6, cOut: 4, lengthIn: 8, kernel: 2 * rate, stride: rate, label: "transconv DAC r\(rate)")
        }
        // ConvNeXt upsample x2: kernel = 2, stride = 2 (trim = 0).
        try runTransConvGate(cIn: 8, cOut: 8, lengthIn: 12, kernel: 2, stride: 2, label: "transconv x2")
    }

    // MARK: - U4: layerNorm fp32 (ConvNeXt LN over channels per frame, [C,T])

    func testLayerNormCTF32MatchesCPU() throws {
        let pso = try pipeline("layer_norm_ct_f32.metal", "layer_norm_ct_f32")
        let dim = 12, frames = 20
        let eps: Float = 1e-6
        let normW = (0..<dim).map { 0.8 + Float($0) * 0.02 }
        let normB = (0..<dim).map { Float($0) * 0.01 - 0.05 }
        let input = (0..<dim * frames).map { cos(Float($0) * 0.23) * 1.7 }

        let ref = Qwen3TTSCodec.layerNormCT(input, dim: dim, frames: frames, normW: normW, normB: normB, eps: eps)

        let inBuf = bufF32(input), wBuf = bufF32(normW), bBuf = bufF32(normB)
        let outBuf = outF32(dim * frames)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
        var ch = UInt32(dim), fr = UInt32(frames), e = eps
        enc.setBytes(&ch, length: 4, index: 4); enc.setBytes(&fr, length: 4, index: 5)
        enc.setBytes(&e, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: frames, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(frames, 32), height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertMatch(readF32(outBuf, dim * frames), ref, "layer_norm_ct_f32")
    }

    // MARK: - P2-U1a: fp32 matmul (y[M,N] = x[M,K]·Wᵀ + b)

    func testMatmulF32MatchesCPU() throws {
        let pso = try pipeline("matmul_f32.metal", "matmul_f32")
        let M = 5, N = 7, K = 11
        let x = (0..<M * K).map { sin(Float($0) * 0.31) * 1.2 }
        let w = (0..<N * K).map { cos(Float($0) * 0.19) * 0.6 }
        let bias = (0..<N).map { Float($0) * 0.04 - 0.1 }
        let ref = Qwen3TTSTalker.linearBias(x, rows: M, inF: K, outF: N, w, bias)

        let xBuf = bufF32(x), wBuf = bufF32(w), bBuf = bufF32(bias), outBuf = outF32(M * N)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
        var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(1)
        enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: N, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(N, 32), height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        assertMatch(readF32(outBuf, M * N), ref, "matmul_f32")
    }

    // MARK: - P2-U1b: pointwise activations (gelu, silu)

    private func runActivationGate(_ fn: String, _ ref: (Float) -> Float, _ label: String) throws {
        let pso = try pipeline("activations_f32.metal", fn)
        let n = 64
        let input = (0..<n).map { sin(Float($0) * 0.41) * 3.0 - 1.0 }
        let refOut = input.map(ref)
        let inBuf = bufF32(input), outBuf = outF32(n)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
        var cnt = UInt32(n); enc.setBytes(&cnt, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        assertMatch(readF32(outBuf, n), refOut, label)
    }

    func testGeluF32MatchesCPU() throws {
        try runActivationGate("gelu_f32", { Qwen3TTSCodec.gelu($0) }, "gelu_f32")
    }
    func testSiluF32MatchesCPU() throws {
        try runActivationGate("silu_f32", { $0 / (1 + exp(-$0)) }, "silu_f32")
    }

    // MARK: - P2-U1b: codec RMSNorm fp32 ([frames, dim] over feature dim, weight direct)

    func testRMSNormCodecF32MatchesCPU() throws {
        let pso = try pipeline("rms_norm_codec_f32.metal", "rms_norm_codec_f32")
        let frames = 18, dim = 32
        let eps: Float = 1e-5
        let w = (0..<dim).map { 0.9 + Float($0) * 0.01 }
        let input = (0..<frames * dim).map { cos(Float($0) * 0.27) * 1.4 }
        let ref = Qwen3TTSCodec.rmsNorm(input, frames: frames, dim: dim, w, eps: eps)

        let inBuf = bufF32(input), wBuf = bufF32(w), outBuf = outF32(frames * dim)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var fr = UInt32(frames), dm = UInt32(dim), e = eps
        enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&dm, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: frames * 32, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        assertMatch(readF32(outBuf, frames * dim), ref, "rms_norm_codec_f32")
    }

    // MARK: - P2-U1b: RVQ dequant gather+sum (q = [qFirst; qRest], [2*dim, T])

    func testRVQGatherSumF32MatchesCPU() throws {
        let pso = try pipeline("rvq_gather_sum_f32.metal", "rvq_gather_sum_f32")
        let dim = 8, frames = 10, restCount = 15, firstN = 12, restN = 16
        let firstEmb = (0..<firstN * dim).map { sin(Float($0) * 0.11) }
        let restEmb = (0..<restCount * restN * dim).map { cos(Float($0) * 0.07) * 0.5 }
        // codes [K, frames], K = 1 + restCount. Row 0 in [0,firstN); rows 1.. in [0,restN).
        var codes = [Int32](repeating: 0, count: (1 + restCount) * frames)
        for t in 0..<frames { codes[t] = Int32((t * 7 + 1) % firstN) }
        for k in 0..<restCount { for t in 0..<frames { codes[(k + 1) * frames + t] = Int32((t * 3 + k) % restN) } }

        // CPU reference q = [qFirst (dim) ; qRest (dim)] stacked into [2*dim, frames].
        var ref = [Float](repeating: 0, count: 2 * dim * frames)
        for t in 0..<frames {
            let c0 = Int(codes[t])
            for d in 0..<dim { ref[d * frames + t] = firstEmb[c0 * dim + d] }
            for k in 0..<restCount {
                let ck = Int(codes[(k + 1) * frames + t])
                for d in 0..<dim { ref[(dim + d) * frames + t] += restEmb[(k * restN + ck) * dim + d] }
            }
        }

        let cBuf = bufI32(codes), fBuf = bufF32(firstEmb), rBuf = bufF32(restEmb)
        let qBuf = outF32(2 * dim * frames)
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(cBuf, offset: 0, index: 0); enc.setBuffer(fBuf, offset: 0, index: 1)
        enc.setBuffer(rBuf, offset: 0, index: 2); enc.setBuffer(qBuf, offset: 0, index: 3)
        var dm = UInt32(dim), fr = UInt32(frames), rc = UInt32(restCount), rn = UInt32(restN)
        enc.setBytes(&dm, length: 4, index: 4); enc.setBytes(&fr, length: 4, index: 5)
        enc.setBytes(&rc, length: 4, index: 6); enc.setBytes(&rn, length: 4, index: 7)
        enc.dispatchThreads(MTLSize(width: frames, height: dim, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(frames, 32), height: 1, depth: 1))
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        assertMatch(readF32(qBuf, 2 * dim * frames), ref, "rvq_gather_sum_f32")
    }

    // MARK: - P2-U3: ConvNeXt block composed on GPU (chain of gated kernels)

    private func encode(_ pso: MTLComputePipelineState, _ grid: MTLSize, _ tg: MTLSize,
                        _ body: (MTLComputeCommandEncoder) -> Void) {
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso); body(enc)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        XCTAssertNil(cmd.error)
    }

    /// C2-B0 (docs/codec-c2b-plan.md): the NEW catalog `transpose_f32` GPU kernel ([rows,cols] →
    /// [cols,rows], dst[c,r]=src[r,c]) matches the hand CPU transpose (Qwen3TTSGPUCodec.swift:965)
    /// bit-for-bit, NON-SQUARE + BIDIRECTIONAL (the codec's CT↔TC layout bridge: [C,T]↔[T,C]). The
    /// §6 leaf oracle before C2-B composes it into the pre_transformer record table. The one Sources
    /// change in C2-B (catalog kernel registered, appended rawValue 400, no renumber).
    func testTransposeF32MatchesCPU() throws {
        let pso = try pipeline("transpose_f32.metal", "transpose_f32")
        func cpuT(_ src: [Float], _ rows: Int, _ cols: Int) -> [Float] {
            var t = [Float](repeating: 0, count: rows * cols)
            for r in 0..<rows { for c in 0..<cols { t[c * rows + r] = src[r * cols + c] } }
            return t
        }
        func gpuT(_ src: [Float], _ rows: Int, _ cols: Int) -> [Float] {
            let inBuf = bufF32(src), outBuf = outF32(rows * cols)
            encode(pso, MTLSize(width: cols, height: rows, depth: 1), MTLSize(width: min(cols, 32), height: 1, depth: 1)) { enc in
                enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
                var rr = UInt32(rows), cc = UInt32(cols)
                enc.setBytes(&rr, length: 4, index: 2); enc.setBytes(&cc, length: 4, index: 3)
            }
            return readF32(outBuf, rows * cols)
        }
        for (rows, cols) in [(12, 20), (20, 12), (16, 100), (100, 16)] {
            let src = (0..<rows * cols).map { Float($0) * 0.013 - 0.7 }
            let g = gpuT(src, rows, cols)
            XCTAssertTrue(g.contains { $0 != 0 }, "transpose [\(rows),\(cols)] vacuous")
            XCTAssertEqual(g.map(\.bitPattern), cpuT(src, rows, cols).map(\.bitPattern),
                           "transpose_f32 [\(rows),\(cols)] != CPU")
        }
    }

    // MARK: - P2-U4a: sliding-window-72 attention (frames > 72 to exercise the window)

    func testSlidingAttnF32MatchesCPU() throws {
        let pso = try pipeline("sliding_attn_f32.metal", "sliding_attn_f32")
        let frames = 100, heads = 4, headDim = 16, window = 72  // frames > window
        let attnDim = heads * headDim
        let q = (0..<frames * attnDim).map { sin(Float($0) * 0.013) * 1.1 }
        let k = (0..<frames * attnDim).map { cos(Float($0) * 0.017) * 1.0 }
        let v = (0..<frames * attnDim).map { sin(Float($0) * 0.019 + 0.5) * 0.9 }

        // CPU reference: sliding-window causal attention.
        let scaling = 1.0 / Float(headDim).squareRoot()
        var ref = [Float](repeating: 0, count: frames * attnDim)
        for hd in 0..<heads {
            for t in 0..<frames {
                let lo = max(0, t - window + 1)
                var sc = [Float](repeating: 0, count: t - lo + 1)
                var mx = -Float.greatestFiniteMagnitude
                for s in lo...t {
                    var dot: Float = 0
                    let qb = t * attnDim + hd * headDim, kb = s * attnDim + hd * headDim
                    for d in 0..<headDim { dot += q[qb + d] * k[kb + d] }
                    sc[s - lo] = dot * scaling; if sc[s - lo] > mx { mx = sc[s - lo] }
                }
                var denom: Float = 0
                for i in 0..<sc.count { sc[i] = expf(sc[i] - mx); denom += sc[i] }
                let ob = t * attnDim + hd * headDim
                for s in lo...t {
                    let wgt = sc[s - lo] / denom, vb = s * attnDim + hd * headDim
                    for d in 0..<headDim { ref[ob + d] += wgt * v[vb + d] }
                }
            }
        }

        let qBuf = bufF32(q), kBuf = bufF32(k), vBuf = bufF32(v), outBuf = outF32(frames * attnDim)
        encode(pso, MTLSize(width: frames, height: heads, depth: 1),
               MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(qBuf, offset: 0, index: 0); enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim), win = UInt32(window)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&hdd, length: 4, index: 6); enc.setBytes(&win, length: 4, index: 7)
        }
        assertMatch(readF32(outBuf, frames * attnDim), ref, "sliding_attn_f32 (frames>window)")
    }

    // MARK: - P2-U4b: RoPE apply (rotate_half, theta 10000)

    /// cos/sin tables [frames, headDim] in cat(freqs,freqs) layout, theta 10000.
    private func ropeTables(frames: Int, headDim: Int, theta: Float = 10000) -> ([Float], [Float]) {
        let half = headDim / 2
        var cosT = [Float](repeating: 0, count: frames * headDim)
        var sinT = [Float](repeating: 0, count: frames * headDim)
        for t in 0..<frames {
            for i in 0..<half {
                let invFreq = 1.0 / powf(theta, Float(2 * i) / Float(headDim))
                let ang = Float(t) * invFreq
                let c = cosf(ang), s = sinf(ang)
                cosT[t * headDim + i] = c; cosT[t * headDim + i + half] = c
                sinT[t * headDim + i] = s; sinT[t * headDim + i + half] = s
            }
        }
        return (cosT, sinT)
    }

    func testRopeApplyF32MatchesCPU() throws {
        let pso = try pipeline("rope_apply_f32.metal", "rope_apply_f32")
        let frames = 20, heads = 4, headDim = 16
        let attnDim = heads * headDim, half = headDim / 2
        let x = (0..<frames * attnDim).map { sin(Float($0) * 0.023) * 1.3 }
        let (cosT, sinT) = ropeTables(frames: frames, headDim: headDim)

        var ref = x
        for t in 0..<frames {
            for hd in 0..<heads {
                let base = t * attnDim + hd * headDim, cb = t * headDim
                for j in 0..<half {
                    let a = x[base + j], b = x[base + j + half]
                    ref[base + j] = a * cosT[cb + j] - b * sinT[cb + j]
                    ref[base + j + half] = b * cosT[cb + j + half] + a * sinT[cb + j + half]
                }
            }
        }

        let xBuf = bufF32(x), cBuf = bufF32(cosT), sBuf = bufF32(sinT), outBuf = outF32(frames * attnDim)
        let total = frames * heads * half
        encode(pso, MTLSize(width: total, height: 1, depth: 1),
               MTLSize(width: min(total, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(cBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&hdd, length: 4, index: 6)
        }
        assertMatch(readF32(outBuf, frames * attnDim), ref, "rope_apply_f32")
    }

    // MARK: - P2-U4c: pre_transformer (multi-layer) composed on GPU

    private func matmulDispatch(_ xBuf: MTLBuffer, _ M: Int, _ K: Int, _ weight: [Float],
                                _ bias: [Float]?, _ N: Int) throws -> MTLBuffer {
        let pso = try pipeline("matmul_f32.metal", "matmul_f32")
        let wBuf = bufF32(weight), bBuf = bufF32(bias ?? [Float](repeating: 0, count: N)), outBuf = outF32(M * N)
        encode(pso, MTLSize(width: N, height: M, depth: 1), MTLSize(width: min(N, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(bias == nil ? 0 : 1)
            enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
            enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
        }
        return outBuf
    }

    private func rmsNormCodecDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ dim: Int, _ w: [Float],
                                      eps: Float = 1e-5) throws -> MTLBuffer {
        let pso = try pipeline("rms_norm_codec_f32.metal", "rms_norm_codec_f32")
        let wBuf = bufF32(w), outBuf = outF32(frames * dim)
        encode(pso, MTLSize(width: frames * 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var fr = UInt32(frames), dm = UInt32(dim), e = eps
            enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&dm, length: 4, index: 4); enc.setBytes(&e, length: 4, index: 5)
        }
        return outBuf
    }

    private func ropeDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ heads: Int, _ headDim: Int,
                              _ cBuf: MTLBuffer, _ sBuf: MTLBuffer) throws -> MTLBuffer {
        precondition(headDim % 2 == 0, "rope_apply_f32 rotates pairs; headDim must be even")
        let pso = try pipeline("rope_apply_f32.metal", "rope_apply_f32")
        let outBuf = outF32(frames * heads * headDim)
        let total = frames * heads * (headDim / 2)
        encode(pso, MTLSize(width: total, height: 1, depth: 1), MTLSize(width: min(total, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(cBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&hdd, length: 4, index: 6)
        }
        return outBuf
    }

    private func swigluDispatch(_ gateBuf: MTLBuffer, _ upBuf: MTLBuffer, _ count: Int) throws -> MTLBuffer {
        let pso = try pipeline("activations_f32.metal", "swiglu_f32")
        let outBuf = outF32(count)
        encode(pso, MTLSize(width: count, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(gateBuf, offset: 0, index: 0); enc.setBuffer(upBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var c = UInt32(count); enc.setBytes(&c, length: 4, index: 3)
        }
        return outBuf
    }

    private func scaleResidualTCDispatch(_ xBuf: MTLBuffer, _ resBuf: MTLBuffer, _ scale: [Float],
                                         _ channels: Int, _ frames: Int) throws -> MTLBuffer {
        let pso = try pipeline("scale_residual_f32.metal", "scale_residual_tc_f32")
        let sBuf = bufF32(scale), outBuf = outF32(frames * channels)
        encode(pso, MTLSize(width: frames, height: channels, depth: 1), MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(resBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var ch = UInt32(channels), fr = UInt32(frames), hs = UInt32(1)
            enc.setBytes(&ch, length: 4, index: 4); enc.setBytes(&fr, length: 4, index: 5)
            enc.setBytes(&hs, length: 4, index: 6)
        }
        return outBuf
    }

    // MARK: - P3-U1: talker per-head RMSNorm (q_norm/k_norm)

    private func rmsNormHeadDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ heads: Int, _ headDim: Int,
                                     _ w: [Float], _ eps: Float) throws -> MTLBuffer {
        let pso = try pipeline("rms_norm_head_f32.metal", "rms_norm_head_f32")
        let wBuf = bufF32(w), outBuf = outF32(frames * heads * headDim)
        encode(pso, MTLSize(width: frames * heads * 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim), e = eps
            enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&hh, length: 4, index: 4)
            enc.setBytes(&hdd, length: 4, index: 5); enc.setBytes(&e, length: 4, index: 6)
        }
        return outBuf
    }

    func testRMSNormHeadF32MatchesCPU() throws {
        let frames = 20, heads = 4, headDim = 8
        let eps: Float = 1e-6
        let w = (0..<headDim).map { 0.9 + Float($0) * 0.02 }
        let x = (0..<frames * heads * headDim).map { sin(Float($0) * 0.019) * 1.4 }
        var ref = x
        Qwen3TTSTalker.headRMSNorm(&ref, frames: frames, heads: heads, headDim: headDim, w, eps: eps)
        let out = try rmsNormHeadDispatch(bufF32(x), frames, heads, headDim, w, eps)
        assertMatch(readF32(out, frames * heads * headDim), ref, "rms_norm_head_f32")
    }

    // MARK: - P3-U2: full-causal GQA attention

    private func causalGQAAttnDispatch(_ qBuf: MTLBuffer, _ kBuf: MTLBuffer, _ vBuf: MTLBuffer,
                                       _ frames: Int, _ heads: Int, _ kvHeads: Int, _ headDim: Int) throws -> MTLBuffer {
        precondition(headDim > 0 && headDim <= 128, "causal_gqa_attn_f32 acc[] is sized for 0<headDim<=128")
        precondition(kvHeads > 0 && heads % kvHeads == 0, "GQA needs heads divisible by kvHeads>0")
        let pso = try pipeline("causal_gqa_attn_f32.metal", "causal_gqa_attn_f32")
        let outBuf = outF32(frames * heads * headDim)
        encode(pso, MTLSize(width: frames, height: heads, depth: 1), MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(qBuf, offset: 0, index: 0); enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), kvh = UInt32(kvHeads), hdd = UInt32(headDim)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&kvh, length: 4, index: 6); enc.setBytes(&hdd, length: 4, index: 7)
        }
        return outBuf
    }

    private func decodeGQAAttnDispatch(_ qBuf: MTLBuffer, _ kBuf: MTLBuffer, _ vBuf: MTLBuffer,
                                       _ cacheLen: Int, _ heads: Int, _ kvHeads: Int, _ headDim: Int) throws -> MTLBuffer {
        precondition(headDim > 0 && headDim <= 128, "decode_gqa_attn_f32 acc[] is sized for 0<headDim<=128")
        precondition(kvHeads > 0 && heads % kvHeads == 0, "GQA needs heads divisible by kvHeads>0")
        let pso = try pipeline("decode_gqa_attn_f32.metal", "decode_gqa_attn_f32")
        let outBuf = outF32(heads * headDim)
        encode(pso, MTLSize(width: heads * 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(qBuf, offset: 0, index: 0); enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var cl = UInt32(cacheLen), hh = UInt32(heads), kvh = UInt32(kvHeads), hdd = UInt32(headDim)
            enc.setBytes(&cl, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&kvh, length: 4, index: 6); enc.setBytes(&hdd, length: 4, index: 7)
        }
        return outBuf
    }

    // Phase-2-U1 parity: the decode kernel (1 query attends the full cache) reproduces the last
    // query row of the full causal kernel — the invariant the KV-cache decode path relies on.
    func testDecodeGQAAttnF32MatchesCausalLastRow() throws {
        let frames = 30, heads = 4, kvHeads = 2, headDim = 8
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        let q = (0..<frames * qDim).map { sin(Float($0) * 0.011) * 1.1 }
        let k = (0..<frames * kvDim).map { cos(Float($0) * 0.013) }
        let v = (0..<frames * kvDim).map { sin(Float($0) * 0.017 + 0.4) * 0.9 }
        let full = readF32(try causalGQAAttnDispatch(bufF32(q), bufF32(k), bufF32(v), frames, heads, kvHeads, headDim), frames * qDim)
        let lastRow = Array(full[((frames - 1) * qDim)..<(frames * qDim)])
        let qLast = Array(q[((frames - 1) * qDim)..<(frames * qDim)])
        let decoded = readF32(try decodeGQAAttnDispatch(bufF32(qLast), bufF32(k), bufF32(v), frames, heads, kvHeads, headDim), qDim)
        assertMatch(decoded, lastRow, "decode_gqa_attn == causal_gqa_attn last row")
    }

    // Phase-6-U1: fp16-weight matmul. (1) the kernel reproduces fp32-with-fp16-rounded-weights
    // (proves the kernel is correct), and (2) the fp16-weight numeric impact vs true fp32 is small
    // and does NOT flip the argmax — on an lm_head-shaped M=1 decode case.
    func testMatmulF16WF32ParityAndError() throws {
        let M = 1, N = 2048, K = 1024
        let x = (0..<M * K).map { sin(Float($0) * 0.013) * 1.2 }
        let wF32 = (0..<N * K).map { cos(Float($0) * 0.0017) * 0.5 }
        let wF16 = wF32.map { Float16($0) }
        var refRounded = [Float](repeating: 0, count: M * N)  // what the kernel SHOULD produce
        var refTrue = [Float](repeating: 0, count: M * N)     // true fp32
        for m in 0..<M { for n in 0..<N {
            var aR: Float = 0, aT: Float = 0
            for k in 0..<K { aR += x[m * K + k] * Float(wF16[n * K + k]); aT += x[m * K + k] * wF32[n * K + k] }
            refRounded[m * N + n] = aR; refTrue[m * N + n] = aT
        } }
        let pso = try pipeline("matmul_f16w_f32.metal", "matmul_f16w_f32")
        let xBuf = bufF32(x)
        let wBuf = device.makeBuffer(bytes: wF16, length: wF16.count * 2, options: .storageModeShared)!
        let bBuf = bufF32([Float](repeating: 0, count: N)), outBuf = outF32(M * N)
        encode(pso, MTLSize(width: N, height: M, depth: 1), MTLSize(width: min(N, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(0)
            enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
            enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
        }
        let got = readF32(outBuf, M * N)
        // (1) kernel correctness vs fp32-with-fp16-rounded weights (reassociation tolerance only).
        var maxRelR: Float = 0
        for i in 0..<M * N { maxRelR = max(maxRelR, abs(got[i] - refRounded[i]) / (abs(refRounded[i]) + 1e-6)) }
        XCTAssertLessThan(maxRelR, 1e-3, "fp16w kernel vs fp32-with-fp16-weights relErr \(maxRelR)")
        // (2) fp16-weight numeric impact vs true fp32: small max abs/rel.
        var maxAbsT: Float = 0, maxRelT: Float = 0
        for i in 0..<M * N { let d = abs(got[i] - refTrue[i]); maxAbsT = max(maxAbsT, d); maxRelT = max(maxRelT, d / (abs(refTrue[i]) + 1e-3)) }
        XCTAssertLessThan(maxRelT, 0.05, "fp16 weight impact relErr \(maxRelT) (maxAbs \(maxAbsT))")
        // (3) margin-aware: fp16 must not flip the argmax across a REAL margin. A flip is only
        // acceptable when the two logits were within the fp16 perturbation (a genuine near-tie —
        // which this near-uniform synthetic case is; real lm_head logits have a clear winner).
        func argmax(_ v: [Float]) -> Int { var b = 0, mx = -Float.greatestFiniteMagnitude; for i in v.indices where v[i] > mx { mx = v[i]; b = i }; return b }
        let amTrue = argmax(refTrue), amGot = argmax(got)
        XCTAssertLessThan(abs(refTrue[amTrue] - refTrue[amGot]), 2 * maxAbsT + 1e-4,
                          "fp16 flipped the argmax across a real margin (\(amTrue) vs \(amGot))")
    }

    // BF16 storage: gemv_bf16w_f32(bf16 W) must equal gemv_f32(the SAME weights widened bf16→fp32)
    // BIT-EXACTLY — bf16→fp32 is a lossless widen (top 16 bits, mantissa zero-padded), so the kernel
    // feeds the gemv the identical fp32 value the old fp32-storage path did. Not a tolerance test.
    func testGemvBF16WMatchesF32WidenedBitExact() throws {
        let N = 2048, K = 2048, tg = 32
        let x = (0..<K).map { sin(Float($0) * 0.011) * 1.1 }
        let bias = (0..<N).map { Float($0 % 7) * 0.01 }
        // Arbitrary bf16 weight bit patterns (truncate random fp32 to its top 16 bits) + their EXACT
        // fp32 widen. A bf16 SOURCE checkpoint stores exactly these 2-byte values; the widen is lossless.
        let raw = (0..<N * K).map { cos(Float($0) * 0.0013) * 0.4 }
        let bf16: [UInt16] = raw.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let widened: [Float] = bf16.map { Float(bitPattern: UInt32($0) << 16) }

        func runGemv(_ fn: String, _ wBuf: MTLBuffer) throws -> [Float] {
            let pso = try pipeline("\(fn).metal", fn)
            let xBuf = bufF32(x), bBuf = bufF32(bias), outBuf = outF32(N)
            encode(pso, MTLSize(width: N * tg, height: 1, depth: 1), MTLSize(width: tg, height: 1, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(1), nU = UInt32(N), kU = UInt32(K), hb = UInt32(1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return readF32(outBuf, N)
        }
        let bf16Buf = device.makeBuffer(bytes: bf16, length: bf16.count * 2, options: .storageModeShared)!
        let got = try runGemv("gemv_bf16w_f32", bf16Buf)
        let ref = try runGemv("gemv_f32", bufF32(widened))
        for i in 0..<N {
            XCTAssertEqual(got[i].bitPattern, ref[i].bitPattern,
                           "bf16w GEMV not bit-exact to fp32-widened at \(i): \(got[i]) vs \(ref[i])")
        }
    }

    // sample_topk_f32 GPU kernel: draws indices that match the host Qwen3TTSSampler distribution.
    // Runs the whole uniform sweep in ONE command buffer (one dispatch per slot, independent out
    // slots), then checks: top_k=1 is argmax for every u; and the empirical frequency over a uniform
    // u-sweep matches the analytic softmax (exact to the sweep step, since both are inverse-CDF).
    private func sampleTopKSweep(_ logits: [Float], uniforms: [Float], temperature: Float, topK: Int) throws -> [Int] {
        let pso = try pipeline("sample_topk_f32.metal", "sample_topk_f32")
        let n = logits.count, trials = uniforms.count
        let lBuf = bufF32(logits), uBuf = bufF32(uniforms)
        let oBuf = device.makeBuffer(length: trials * 4, options: .storageModeShared)!
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for slot in 0..<trials {
            enc.setBuffer(lBuf, offset: 0, index: 0); enc.setBuffer(uBuf, offset: 0, index: 1)
            enc.setBuffer(oBuf, offset: 0, index: 2)
            var nU = UInt32(n), s = UInt32(slot), temp = temperature, kk = UInt32(topK)
            enc.setBytes(&nU, length: 4, index: 3); enc.setBytes(&s, length: 4, index: 4)
            enc.setBytes(&temp, length: 4, index: 5); enc.setBytes(&kk, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        let op = oBuf.contents().bindMemory(to: UInt32.self, capacity: trials)
        return (0..<trials).map { Int(op[$0]) }
    }

    func testSampleTopKKernelTopK1IsArgmax() throws {
        let logits: [Float] = [0.1, 3.5, -2.0, 3.4, 1.0, 3.49]
        let us = (0..<64).map { Float($0) / 64 }
        let got = try sampleTopKSweep(logits, uniforms: us, temperature: 0.9, topK: 1)
        for idx in got { XCTAssertEqual(idx, 1, "top_k=1 must be argmax for every u") }
    }

    func testSampleTopKKernelMatchesSoftmaxFrequency() throws {
        // top_k=2 keeps indices 0,1; P(0) = softmax(logit/temp). Sweep u uniformly → frequency is the
        // analytic probability to the sweep step (inverse-CDF), the same property the CPU sampler gate
        // checks. The GPU runs fp32 (host is fp64), so allow a small margin beyond pure discretization.
        let logits: [Float] = [2.0, 1.0, -5.0, -6.0]
        let temp: Float = 0.8
        let w0 = exp(Double(logits[0]) / Double(temp)), w1 = exp(Double(logits[1]) / Double(temp))
        let p0 = w0 / (w0 + w1)
        let trials = 4096
        let us = (0..<trials).map { Float($0) / Float(trials) }
        let got = try sampleTopKSweep(logits, uniforms: us, temperature: temp, topK: 2)
        XCTAssertTrue(got.allSatisfy { $0 == 0 || $0 == 1 }, "top_k=2 keeps only indices 0,1")
        let freq0 = Double(got.filter { $0 == 0 }.count) / Double(trials)
        XCTAssertEqual(freq0, p0, accuracy: 0.01, "GPU sampler frequency must match softmax")
    }

    // Phase-7-perf: coalesced GEMV (M=1) reproduces matmul_f32, and its fp16 variant matches
    // fp32-with-fp16-weights — on a talker-shaped decode case (M=1, N=2048, K=2048).
    func testGemvMatchesMatmulM1() throws {
        let N = 2048, K = 2048, tg = 32   // one SIMD per output (matches the driver dispatch)
        let x = (0..<K).map { sin(Float($0) * 0.011) * 1.1 }
        let wF32 = (0..<N * K).map { cos(Float($0) * 0.0013) * 0.4 }
        let bias = (0..<N).map { Float($0 % 7) * 0.01 }
        var ref = [Float](repeating: 0, count: N)
        var refF16 = [Float](repeating: 0, count: N)
        let wF16 = wF32.map { Float16($0) }
        for n in 0..<N {
            var a: Float = 0, aH: Float = 0
            for k in 0..<K { a += x[k] * wF32[n * K + k]; aH += x[k] * Float(wF16[n * K + k]) }
            ref[n] = a + bias[n]; refF16[n] = aH + bias[n]
        }
        func runGemv(_ fn: String, _ wBuf: MTLBuffer) throws -> [Float] {
            let pso = try pipeline(fn == "gemv_f16w_f32" ? "gemv_f16w_f32.metal" : "gemv_f32.metal", fn)
            let outBuf = outF32(N)
            encode(pso, MTLSize(width: N * tg, height: 1, depth: 1), MTLSize(width: tg, height: 1, depth: 1)) { enc in
                enc.setBuffer(bufF32(x), offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(bufF32(bias), offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(1), nU = UInt32(N), kU = UInt32(K), hb = UInt32(1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return readF32(outBuf, N)
        }
        // Whole-vector relL2 (robust to near-zero outputs). The GEMV uses a tree reduction vs the
        // reference's sequential sum, so it's fp32-equivalent (reassociation), not bit-exact.
        func relL2(_ got: [Float], _ r: [Float]) -> Float {
            var diff: Float = 0, nb: Float = 0
            for i in 0..<r.count { let d = got[i] - r[i]; diff += d * d; nb += r[i] * r[i] }
            return diff.squareRoot() / nb.squareRoot()
        }
        let gotF32 = try runGemv("gemv_f32", bufF32(wF32))
        XCTAssertLessThan(relL2(gotF32, ref), 0.02, "gemv_f32 vs matmul reference (reassociation)")
        let wF16Buf = device.makeBuffer(bytes: wF16, length: wF16.count * 2, options: .storageModeShared)!
        let gotF16 = try runGemv("gemv_f16w_f32", wF16Buf)
        XCTAssertLessThan(relL2(gotF16, refF16), 0.02, "gemv_f16w vs fp32-with-fp16-weights")
    }

    // gemv_u4_f32 / gemm_u4_f32: group-wise affine int4 weights. The reference is a CPU dot of the
    // SAME packed bytes dequantized by SmeltAffineU4 — so this validates the kernel's nibble unpacking,
    // threadgroup-staged scale/bias, and the per-chunk dequant identity, NOT the quantization loss
    // (that's the end-to-end gate's job). Reassociated vs the sequential reference → relL2 small, not
    // bit-exact. group=64 over K=2048/6144 covers the talker's projection shapes.
    // Weights AND activations are pseudo-RANDOM (not smooth): with smooth data a low/high-nibble swap or
    // 4-column reversal stays ~1e-5 relL2 and slips under the threshold — random data makes any
    // nibble-ordering bug blow relL2 up to O(1) while a correct unpack stays at the reassociation floor.
    func testGemvGemmU4MatchCPUDequant() throws {
        func relL2(_ got: [Float], _ r: [Float]) -> Float {
            var diff: Float = 0, nb: Float = 0
            for i in 0..<r.count { let d = got[i] - r[i]; diff += d * d; nb += r[i] * r[i] }
            return diff.squareRoot() / nb.squareRoot()
        }
        var rng: UInt64 = 0x2545F4914F6CDD1D
        func rand() -> Float {   // xorshift64* → [-1, 1)
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return Float(Int32(truncatingIfNeeded: rng &* 0x2545F4914F6CDD1D)) / Float(Int32.max)
        }
        for (N, K, g) in [(1024, 2048, 64), (512, 6144, 64), (256, 1024, 32)] {
            let wF32 = (0..<N * K).map { _ in rand() * 0.4 }
            let bias = (0..<N).map { Float($0 % 7) * 0.01 }
            let packed = SmeltAffineU4.quantize(wF32, rows: N, cols: K, groupSize: g)
            let wDeq = SmeltAffineU4.dequantize(packed)   // exactly what the kernel reconstructs
            let wqBuf = device.makeBuffer(bytes: packed.nibbles, length: packed.nibbles.count, options: .storageModeShared)!
            let scBuf = device.makeBuffer(bytes: packed.scales, length: packed.scales.count * 2, options: .storageModeShared)!
            let biBuf = device.makeBuffer(bytes: packed.biases, length: packed.biases.count * 2, options: .storageModeShared)!

            // CPU reference: dot of x against the SAME dequantized weights.
            func cpuRef(_ x: [Float], _ M: Int) -> [Float] {
                var r = [Float](repeating: 0, count: M * N)
                for m in 0..<M { for n in 0..<N { var a = bias[n]; for k in 0..<K { a += x[m * K + k] * wDeq[n * K + k] }; r[m * N + n] = a } }
                return r
            }
            func runU4(_ pso: MTLComputePipelineState, _ M: Int, _ x: [Float]) -> [Float] {
                let out = outF32(M * N)
                encode(pso, MTLSize(width: N * 32, height: M, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                    enc.setBuffer(bufF32(x), offset: 0, index: 0); enc.setBuffer(wqBuf, offset: 0, index: 1)
                    enc.setBuffer(scBuf, offset: 0, index: 2); enc.setBuffer(biBuf, offset: 0, index: 3)
                    enc.setBuffer(bufF32(bias), offset: 0, index: 4); enc.setBuffer(out, offset: 0, index: 5)
                    var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(1), gs = UInt32(g)
                    enc.setBytes(&mU, length: 4, index: 6); enc.setBytes(&nU, length: 4, index: 7)
                    enc.setBytes(&kU, length: 4, index: 8); enc.setBytes(&hb, length: 4, index: 9)
                    enc.setBytes(&gs, length: 4, index: 10)
                }
                return readF32(out, M * N)
            }
            // M=1 decode (gemv) and M=3 prefill (gemm), same row reduction.
            let xV = (0..<K).map { _ in rand() * 1.1 }
            let gotV = try runU4(pipeline("gemv_u4_f32.metal", "gemv_u4_f32"), 1, xV)
            XCTAssertLessThan(relL2(gotV, cpuRef(xV, 1)), 2e-3, "gemv_u4 N=\(N) K=\(K) g=\(g)")
            let M = 3
            let xM = (0..<M * K).map { _ in rand() * 0.9 }
            let gotM = try runU4(pipeline("gemm_u4_f32.metal", "gemm_u4_f32"), M, xM)
            XCTAssertLessThan(relL2(gotM, cpuRef(xM, M)), 2e-3, "gemm_u4 N=\(N) K=\(K) g=\(g)")
        }
    }

    // channel_copy_f32: per-channel strided copy for on-GPU causal pad (dst longer, dstOff=leftPad,
    // zeros from the pre-memset) and transposed-conv trim (dst shorter). Both must match the CPU
    // pad/trim byte-for-byte (it replaces a CPU readback → bit-exact codec).
    func testChannelCopyF32() throws {
        let pso = try pipeline("channel_copy_f32.metal", "channel_copy_f32")
        func run(channels: Int, srcLen: Int, dstLen: Int, srcOff: Int, dstOff: Int, copyLen: Int,
                 _ src: [Float]) -> [Float] {
            let srcBuf = bufF32(src), dstBuf = outF32(channels * dstLen)   // outF32 zeroes
            encode(pso, MTLSize(width: copyLen, height: channels, depth: 1),
                   MTLSize(width: min(copyLen, 32), height: 1, depth: 1)) { enc in
                enc.setBuffer(srcBuf, offset: 0, index: 0); enc.setBuffer(dstBuf, offset: 0, index: 1)
                var ch = UInt32(channels), sl = UInt32(srcLen), dl = UInt32(dstLen)
                var so = UInt32(srcOff), dof = UInt32(dstOff), cl = UInt32(copyLen)
                enc.setBytes(&ch, length: 4, index: 2); enc.setBytes(&sl, length: 4, index: 3)
                enc.setBytes(&dl, length: 4, index: 4); enc.setBytes(&so, length: 4, index: 5)
                enc.setBytes(&dof, length: 4, index: 6); enc.setBytes(&cl, length: 4, index: 7)
            }
            return readF32(dstBuf, channels * dstLen)
        }
        // Pad: 3 channels, src len 5 → dst len 8 with leftPad 2 (zeros at [0,2) and [7,8)).
        let src = (0..<15).map { Float($0) + 1 }   // [c=0: 1..5, c=1: 6..10, c=2: 11..15]
        let padded = run(channels: 3, srcLen: 5, dstLen: 8, srcOff: 0, dstOff: 2, copyLen: 5, src)
        for c in 0..<3 {
            for l in 0..<8 {
                let exp: Float = (l >= 2 && l < 7) ? Float(c * 5 + (l - 2)) + 1 : 0
                XCTAssertEqual(padded[c * 8 + l], exp, "pad c\(c) l\(l)")
            }
        }
        // Trim: src len 8 → dst len 5 (keep prefix), e.g. transConv right-trim.
        let trimmed = run(channels: 3, srcLen: 8, dstLen: 5, srcOff: 0, dstOff: 0, copyLen: 5, padded)
        for c in 0..<3 { for l in 0..<5 { XCTAssertEqual(trimmed[c * 5 + l], padded[c * 8 + l], "trim c\(c) l\(l)") } }
    }

    // Qwen3TTSActivationCapture full-Hessian path (GPTQ): the handed-off H must equal Σ XᵀX of the
    // committed inputs across calls (FULL symmetric — ssyrk fills only the upper triangle while
    // accumulating, the accessor mirrors it). The diagonal must match the imatrix sum-of-sq, and the H
    // must drive SmeltGPTQ identically to a hand-built full-symmetric H of the same data.
    func testActivationCaptureHessianMatchesXtX() throws {
        let m = 6, k = 16
        var rng: UInt64 = 0x9911AABB
        func rand() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) }
        let cap = Qwen3TTSActivationCapture(); cap.captureHessian = true
        var ref = [Float](repeating: 0, count: k * k)   // Σ XᵀX (full, reference)
        for _ in 0..<2 {   // two dispatches → accumulation
            let X = (0..<m * k).map { _ in rand() }
            let buf = device.makeBuffer(bytes: X, length: X.count * 4, options: .storageModeShared)!
            cap.accumulate("w", buf, m: m, k: k)
            for a in 0..<k { for b in 0..<k { var s: Float = 0; for t in 0..<m { s += X[t * k + a] * X[t * k + b] }; ref[a * k + b] += s } }
        }
        let H = try XCTUnwrap(cap.hessian("w"))
        // FULL symmetric — both triangles must match the reference (the accessor mirrors upper→lower, so
        // GPTQ's column-major spotrf_('U') reads real off-diagonals, not zeros).
        for a in 0..<k { for b in 0..<k { XCTAssertEqual(H[a * k + b], ref[a * k + b], accuracy: max(1e-3, abs(ref[a * k + b]) * 1e-4), "H[\(a),\(b)]") } }
        // Diagonal must also equal the imatrix sum-of-squares (× rowCount = 2m).
        let imp = try XCTUnwrap(cap.importance("w"))
        for a in 0..<k { XCTAssertEqual(imp[a] * Float(2 * m), ref[a * k + a], accuracy: max(1e-3, abs(ref[a * k + a]) * 1e-4), "diag \(a)") }
        // The captured H must drive GPTQ identically to a hand-built full-symmetric H — i.e. the handoff
        // really is fully symmetric (a half-filled H would quantize differently, the bug this guards).
        let W = (0..<8 * k).map { _ in rand() * 0.05 }
        let fromCap = SmeltAffineU4.dequantize(SmeltGPTQ.quantize(weights: W, rows: 8, cols: k, groupSize: 8, hessian: H))
        let fromRef = SmeltAffineU4.dequantize(SmeltGPTQ.quantize(weights: W, rows: 8, cols: k, groupSize: 8, hessian: ref))
        for i in 0..<fromCap.count { XCTAssertEqual(fromCap[i], fromRef[i], "GPTQ output differs at \(i): captured H not fully symmetric") }
    }

    // G4-U1: build(gptqBlocks:) writes precomputed u4 blocks VERBATIM. Given blocks that ARE the affine
    // min/max quantization, the package must be byte-identical to the plain affine build (proving the
    // verbatim write reproduces fillU4 exactly). Plus the no-silent-hybrid coverage: a missing in-scope
    // block, an out-of-scope key, and a wrong-shape block must all throw.
    func testBuildConsumesGPTQBlocks() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        let g = 64
        // Affine-minMax block for every in-scope proj weight (isFP16Candidate u4), from the same
        // bf16→fp32 widening the builder uses → identical to what fillU4 writes.
        let loader = try SafetensorsLoader(directory: model)
        var blocks: [String: SmeltAffineU4.Packed] = [:]
        var shape: [String: (Int, Int)] = [:]
        for info in loader.tensors where Qwen3TTSPackageBuilder.isFP16Candidate(info.name) {
            let (n, k) = (info.shape[0], info.shape[1])
            shape[info.name] = (n, k)
            blocks[info.name] = SmeltAffineU4.quantize(loadTensorBF16(loader, info.name), rows: n, cols: k, groupSize: g)
        }
        XCTAssertFalse(blocks.isEmpty, "no in-scope proj weights found")

        var built: [String] = []
        defer { for p in built { try? FileManager.default.removeItem(atPath: p) } }
        let checkpointPolicy = try qwen3TTSTestCheckpointPolicy()
        @discardableResult
        func build(_ tag: String, _ gptqBlocks: [String: SmeltAffineU4.Packed]?) throws -> String {
            let pkg = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen3tts-g4u1-\(tag)-\(UUID().uuidString).smeltpkg")
            built.append(pkg)
            try Qwen3TTSPackageBuilder.build(checkpointDir: model, checkpointPolicy: checkpointPolicy,
                                             shaderDir: shaderDir, outputPath: pkg,
                                             projDType: .u4, u4GroupSize: g, gptqBlocks: gptqBlocks)
            return pkg
        }
        func weightsBin(_ pkg: String) -> Data { try! Data(contentsOf: URL(fileURLWithPath: "\(pkg)/weights.bin")) }
        let aff = try build("aff", nil)        // plain affine (minMax)
        let blk = try build("blk", blocks)     // verbatim blocks that ARE the affine quant
        XCTAssertEqual(weightsBin(aff), weightsBin(blk),
                       "gptqBlocks build (affine blocks) must equal the plain affine build byte-for-byte")

        // No silent hybrid: missing an in-scope block → throw.
        let oneName = blocks.keys.sorted().first!
        var missing = blocks; missing.removeValue(forKey: oneName)
        XCTAssertThrowsError(try build("miss", missing)) { XCTAssertTrue("\($0)".contains("GPTQ block"), "\($0)") }
        // An out-of-scope key (text_embedding is u4 but not a GPTQ target) → throw.
        var extra = blocks; extra["talker.model.text_embedding.weight"] = blocks[oneName]
        XCTAssertThrowsError(try build("extra", extra)) { XCTAssertTrue("\($0)".contains("GPTQ block"), "\($0)") }
        // A wrong-shape block (cols off by one group) → throw (a valid Packed of the wrong dimensions).
        let (n, k) = shape[oneName]!
        var wrong = blocks
        wrong[oneName] = SmeltAffineU4.quantize([Float](repeating: 0, count: n * (k + g)), rows: n, cols: k + g, groupSize: g)
        XCTAssertThrowsError(try build("wrong", wrong)) { XCTAssertTrue("\($0)".contains("GPTQ block"), "\($0)") }
    }

    // G4-U2: SmeltGPTQCalibrator streams (grouped multi-pass) GPTQ blocks for every in-scope proj weight.
    // Asserts: (a) coverage — a block for every isFP16Candidate proj weight, so build(gptqBlocks:) accepts
    // them (the real builder-scope cross-check); (b) ranks populated; (c) streaming invariance — a block
    // is identical whether captured in a multi-weight group or alone (grouping changes nothing numeric).
    func testGPTQCalibratorStreamsBuildableBlocks() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        let g = 64, maxFrames = 48
        let prompts = [SmeltGPTQCalibrator.Prompt(text: "The quick brown fox.", instruct: "Calm.", language: "English"),
                       SmeltGPTQCalibrator.Prompt(text: "Counting by sevens is easy.", instruct: "Calm.", language: "English")]
        // bf16 calibration package (bundles the tokenizer for inputsEmbeds).
        let bf16Pkg = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen3tts-g4u2-bf16-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: bf16Pkg) }
        let checkpointPolicy = try qwen3TTSTestCheckpointPolicy()
        try Qwen3TTSPackageBuilder.build(checkpointDir: model, checkpointPolicy: checkpointPolicy,
                                         shaderDir: shaderDir, outputPath: bf16Pkg, projDType: .bf16)

        // Multi-pass (layersPerPass < #layers forces ≥2 groups → exercises capture→quantize→free).
        let (blocks, ranks) = try SmeltGPTQCalibrator.calibrate(
            bf16PackagePath: bf16Pkg, checkpointDir: model, device: device, prompts: prompts,
            layersPerPass: 16, groupSize: g, maxFrames: maxFrames)

        // (a) coverage == exactly the in-scope proj weights.
        let loader = try SafetensorsLoader(directory: model)
        let expected = Set(loader.tensors.map(\.name).filter(Qwen3TTSPackageBuilder.isFP16Candidate))
        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(Set(blocks.keys), expected, "calibrator must produce a block for every in-scope proj weight")
        // (b) ranks populated (= min(calib tokens, K) > 0).
        for name in expected { XCTAssertGreaterThan(ranks[name] ?? 0, 0, "no rank for \(name)") }

        // (a, cont.) the blocks actually build (builder coverage aligns with the calibrator's scope).
        let gptqPkg = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen3tts-g4u2-gptq-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: gptqPkg) }
        try Qwen3TTSPackageBuilder.build(checkpointDir: model, checkpointPolicy: checkpointPolicy,
                                         shaderDir: shaderDir, outputPath: gptqPkg,
                                         projDType: .u4, u4GroupSize: g, gptqBlocks: blocks)

        // (c) grouping invariance: every block is non-vacuous (a u4 weight quantizes to a non-uniform
        // nibble field). With the compiled capture (Phase 4 U2), grouping invariance is STRUCTURAL —
        // captureHessianNames is a per-name filter, so a weight's Hessian (Σ XᵀX over its own input
        // rows from the deterministic greedy generation) is identical regardless of which other names
        // share its group. (The old hand-path single-weight recapture is retired with the hand talker.)
        let oneName = "talker.model.layers.0.mlp.down_proj.weight"
        let streamed = try XCTUnwrap(blocks[oneName])
        XCTAssertEqual(streamed.nibbles.count, loader.tensor(named: oneName)!.shape.reduce(1, *) / 2,
                       "down_proj nibble field is half the element count (4-bit packed)")
        XCTAssertTrue(streamed.nibbles.contains { $0 != streamed.nibbles[0] },
                      "block nibbles are uniform — capture was vacuous")
    }

    // gather_row_bf16w_f32 / next_frame_input_bf16w_f32: the bf16-table variants must be BIT-IDENTICAL to
    // the f32 kernels when the bf16 table holds the same BF16-source values (widen = exact). This is the
    // gate for storing the MTP codec_embedding tables bf16 (halves their footprint, no quality change).
    func testBF16GatherKernelsBitExactVsF32() throws {
        let dim = 2048, rows = 64
        // BF16-representable values (round f32→bf16) so f32-stored and bf16-stored encode the SAME number.
        var rng: UInt64 = 0x51ED2701A3C9F4B7
        func bf16bits(_ f: Float) -> UInt16 { let b = f.bitPattern; return UInt16(truncatingIfNeeded: (b &+ (0x7FFF &+ ((b >> 16) & 1))) >> 16) }
        func widen(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
        func mkTable() -> (f32: [Float], bf16: [UInt16]) {
            var f = [Float](repeating: 0, count: rows * dim), h = [UInt16](repeating: 0, count: rows * dim)
            for i in 0..<f.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let v = Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) * 0.1
                h[i] = bf16bits(v); f[i] = widen(h[i])   // identical value, two storages
            }
            return (f, h)
        }
        func bf16Buf(_ h: [UInt16]) -> MTLBuffer { device.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared)! }

        // gather_row: f32 vs bf16w, same row index.
        let t = mkTable()
        let idxBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        idxBuf.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 37
        func gather(_ fn: String, _ table: MTLBuffer) throws -> [Float] {
            let pso = try pipeline("\(fn).metal", fn), out = outF32(dim)
            encode(pso, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: min(dim, 256), height: 1, depth: 1)) { enc in
                enc.setBuffer(table, offset: 0, index: 0); enc.setBuffer(idxBuf, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
                var d = UInt32(dim), s = UInt32(0); enc.setBytes(&d, length: 4, index: 3); enc.setBytes(&s, length: 4, index: 4)
            }
            return readF32(out, dim)
        }
        let gF32 = try gather("gather_row_f32", bufF32(t.f32))
        let gBF16 = try gather("gather_row_bf16w_f32", bf16Buf(t.bf16))
        for d in 0..<dim { XCTAssertEqual(gBF16[d], gF32[d], "gather_row bf16 vs f32 d=\(d)") }

        // next_frame_input: f32 vs bf16w MTP tables (talkerEmb + ttsPad stay f32 in both).
        let talker = mkTable().f32, ttsPad = (0..<dim).map { Float($0) * 1e-4 }
        let mtps = (0..<15).map { _ in mkTable() }
        let codes = [3] + (0..<15).map { ($0 * 7 + 2) % rows }
        let codesBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        let cp = codesBuf.contents().bindMemory(to: UInt32.self, capacity: 16)
        for i in 0..<16 { cp[i] = UInt32(codes[i]) }
        func nfi(_ fn: String, _ tables: [MTLBuffer]) throws -> [Float] {
            let pso = try pipeline("\(fn).metal", fn), out = outF32(dim)
            encode(pso, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: min(dim, 256), height: 1, depth: 1)) { enc in
                enc.setBuffer(codesBuf, offset: 0, index: 0); enc.setBuffer(bufF32(ttsPad), offset: 0, index: 1)
                enc.setBuffer(bufF32(talker), offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
                for i in 0..<15 { enc.setBuffer(tables[i], offset: 0, index: 4 + i) }
                var d = UInt32(dim); enc.setBytes(&d, length: 4, index: 19)
            }
            return readF32(out, dim)
        }
        let nF32 = try nfi("next_frame_input_f32", mtps.map { bufF32($0.f32) })
        let nBF16 = try nfi("next_frame_input_bf16w_f32", mtps.map { bf16Buf($0.bf16) })
        for d in 0..<dim { XCTAssertEqual(nBF16[d], nF32[d], "next_frame_input bf16 vs f32 d=\(d)") }
    }

    // cb0_argmax_f32: on-GPU cb0 selection must EXACTLY match applyCb0Processors + CPU argmax (lowest
    // index wins ties), across the stateful repetition penalty, suppress range, and min_new_tokens —
    // a flipped cb0 is a flipped frame (codes==gen_codes). Exercises ties, repeats, eos-suppress.
    func testCb0ArgmaxF32MatchesCPU() throws {
        let pso = try pipeline("cb0_argmax_f32.metal", "cb0_argmax_f32")
        let n = 3072, suppressFrom = 2048, eos = 2150, penalty: Float = 1.05, minNew = 2
        func cpuCb0(_ logits: [Float], history: [Int], frame: Int) -> Int {
            let l = Qwen3TTSGenerator.applyCb0Processors(logits, history: history, frame: frame,
                repetitionPenalty: penalty, suppressFrom: suppressFrom, vocab: n, codecEosId: eos, minNewTokens: minNew)
            var best = 0, mx = -Float.greatestFiniteMagnitude
            for v in 0..<n where l[v] > mx { mx = l[v]; best = v }
            return best
        }
        func gpuCb0(_ logits: [Float], history: [Int], frame: Int) -> Int {
            let lBuf = bufF32(logits)
            let hBuf = device.makeBuffer(length: max(1, history.count) * 4, options: .storageModeShared)!
            let hp = hBuf.contents().bindMemory(to: UInt32.self, capacity: max(1, history.count))
            for (i, h) in history.enumerated() { hp[i] = UInt32(h) }
            let oBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
            encode(pso, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(lBuf, offset: 0, index: 0); enc.setBuffer(hBuf, offset: 0, index: 1); enc.setBuffer(oBuf, offset: 0, index: 2)
                var nU = UInt32(n), hl = UInt32(history.count), fr = UInt32(frame), pen = penalty
                var sf = UInt32(suppressFrom), e = UInt32(eos), mnt = UInt32(minNew)
                enc.setBytes(&nU, length: 4, index: 3); enc.setBytes(&hl, length: 4, index: 4)
                enc.setBytes(&fr, length: 4, index: 5); enc.setBytes(&pen, length: 4, index: 6)
                enc.setBytes(&sf, length: 4, index: 7); enc.setBytes(&e, length: 4, index: 8); enc.setBytes(&mnt, length: 4, index: 9)
            }
            return Int(oBuf.contents().bindMemory(to: UInt32.self, capacity: 1)[0])
        }
        var rng: UInt64 = 0x243F6A8885A308D3
        func nf() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) * 6 }
        for frame in [0, 1, 2, 5] {
            for trial in 0..<8 {
                var logits = (0..<n).map { _ in nf() }
                // history: some prior audio codes incl one that's (pre-penalty) near the top → penalty can flip it.
                let history = trial % 3 == 0 ? [] : (0..<(frame + trial)).map { ($0 * 37 + 11) % suppressFrom }
                if trial % 2 == 0, !history.isEmpty { logits[history[0]] = 5.9 }   // make a penalized token contend
                XCTAssertEqual(gpuCb0(logits, history: history, frame: frame), cpuCb0(logits, history: history, frame: frame),
                               "cb0 frame=\(frame) trial=\(trial)")
            }
        }
        // Exact tie at two audio indices → lowest wins (both paths).
        var ties = (0..<n).map { _ in Float(-10) }; ties[100] = 5; ties[700] = 5
        XCTAssertEqual(gpuCb0(ties, history: [], frame: 5), cpuCb0(ties, history: [], frame: 5), "cb0 tie")
        // frame < minNew with eos as the raw max → eos suppressed, an audio code wins.
        var eosTop = (0..<n).map { _ in Float(-10) }; eosTop[eos] = 9; eosTop[42] = 1
        XCTAssertEqual(gpuCb0(eosTop, history: [], frame: 0), cpuCb0(eosTop, history: [], frame: 0), "cb0 eos-suppressed")
        XCTAssertEqual(gpuCb0(eosTop, history: [], frame: 5), cpuCb0(eosTop, history: [], frame: 5), "cb0 eos-allowed")
    }

    private let cb0N = 3072, cb0SuppressFrom = 2048, cb0Eos = 2150
    private let cb0Penalty: Float = 1.05, cb0MinNew = 2

    /// Run the whole uniform sweep in ONE command buffer. cb0_sample_topk_f32 always reads uniforms[0]
    /// / writes out[0] (cb0 is slot 0 in production), so each trial binds the shared uniform/out buffers
    /// at a per-slot byte offset — the slot-free analog of sampleTopKSweep's `slot` constant.
    private func cb0SampleSweep(_ logits: [Float], history: [Int], frame: Int, uniforms: [Float],
                                temperature: Float, topK: Int) throws -> [Int] {
        let pso = try pipeline("cb0_sample_topk_f32.metal", "cb0_sample_topk_f32")
        let trials = uniforms.count
        let lBuf = bufF32(logits), uBuf = bufF32(uniforms)
        let hBuf = device.makeBuffer(length: max(1, history.count) * 4, options: .storageModeShared)!
        let hp = hBuf.contents().bindMemory(to: UInt32.self, capacity: max(1, history.count))
        for (i, h) in history.enumerated() { hp[i] = UInt32(h) }
        let oBuf = device.makeBuffer(length: trials * 4, options: .storageModeShared)!
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for slot in 0..<trials {
            enc.setBuffer(lBuf, offset: 0, index: 0); enc.setBuffer(hBuf, offset: 0, index: 1)
            enc.setBuffer(uBuf, offset: slot * 4, index: 2); enc.setBuffer(oBuf, offset: slot * 4, index: 3)
            var nU = UInt32(cb0N), hl = UInt32(history.count), fr = UInt32(frame), pen = cb0Penalty
            var sf = UInt32(cb0SuppressFrom), e = UInt32(cb0Eos), mnt = UInt32(cb0MinNew)
            var temp = temperature, kk = UInt32(topK)
            enc.setBytes(&nU, length: 4, index: 4); enc.setBytes(&hl, length: 4, index: 5)
            enc.setBytes(&fr, length: 4, index: 6); enc.setBytes(&pen, length: 4, index: 7)
            enc.setBytes(&sf, length: 4, index: 8); enc.setBytes(&e, length: 4, index: 9)
            enc.setBytes(&mnt, length: 4, index: 10); enc.setBytes(&temp, length: 4, index: 11)
            enc.setBytes(&kk, length: 4, index: 12)
            enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        let op = oBuf.contents().bindMemory(to: UInt32.self, capacity: trials)
        return (0..<trials).map { Int(op[$0]) }
    }

    private func cb0Processed(_ logits: [Float], history: [Int], frame: Int) -> [Float] {
        Qwen3TTSGenerator.applyCb0Processors(logits, history: history, frame: frame,
            repetitionPenalty: cb0Penalty, suppressFrom: cb0SuppressFrom, vocab: cb0N,
            codecEosId: cb0Eos, minNewTokens: cb0MinNew)
    }

    // cb0_sample_topk_f32 (top_k=1): collapses to the processed argmax for EVERY u, so the fused-processor
    // prelude must run inside the GPU sampler exactly as in cb0_argmax. Exercises the repetition penalty
    // flipping the raw winner and the min_new_tokens eos gate (the processors that, if dropped on the
    // sampling path, would silently corrupt sampled audio while the greedy gate stayed green).
    func testCb0SampleTopKTopK1MatchesProcessedArgmax() throws {
        func argmax(_ l: [Float]) -> Int { var b = 0, m = -Float.greatestFiniteMagnitude; for v in 0..<l.count where l[v] > m { m = l[v]; b = v }; return b }
        let us = (0..<256).map { Float($0) / 256 }
        // Penalty flips the winner: raw max is the history token 10 (2.0 → 1.905), but 20 (1.95) is unpenalized.
        var flip = (0..<cb0N).map { _ in Float(-30) }; flip[10] = 2.0; flip[20] = 1.95
        let want = argmax(cb0Processed(flip, history: [10], frame: 5))
        XCTAssertEqual(want, 20, "penalty must flip the winner to 20 (else the test is vacuous)")
        for idx in try cb0SampleSweep(flip, history: [10], frame: 5, uniforms: us, temperature: 0.9, topK: 1) {
            XCTAssertEqual(idx, 20, "top_k=1 must draw the processed argmax for every u")
        }
        // min_new_tokens: eos is the raw max but suppressed at frame 0 (< minNew), allowed at frame 5.
        var eosTop = (0..<cb0N).map { _ in Float(-30) }; eosTop[cb0Eos] = 9; eosTop[42] = 1
        for idx in try cb0SampleSweep(eosTop, history: [], frame: 0, uniforms: us, temperature: 0.9, topK: 1) {
            XCTAssertEqual(idx, 42, "frame<minNew must suppress eos on the sampling path")
        }
        for idx in try cb0SampleSweep(eosTop, history: [], frame: 5, uniforms: us, temperature: 0.9, topK: 1) {
            XCTAssertEqual(idx, cb0Eos, "frame>=minNew must allow eos")
        }
    }

    // cb0_sample_topk_f32 distribution-equivalence: over a uniform u-sweep the GPU draws must match the
    // host (applyCb0Processors → sampleTopK) reference — per-u identical except for fp32-vs-fp64 CDF
    // boundary rounding (the documented sample_topk standard). Non-vacuity: the penalty pushes a token
    // OUT of the kept set (never drawn) and a suppressed-range token is never drawn, so the GPU threshold
    // pass demonstrably applies the processors, not just the bare logits; and both kept indices appear.
    func testCb0SampleTopKMatchesHostFrequency() throws {
        let temp: Float = 0.9, topK = 2, frame = 5, history = [40]
        var logits = (0..<cb0N).map { _ in Float(-30) }
        logits[10] = 2.0      // kept (top)
        logits[20] = 1.85     // kept (2nd → threshold)
        logits[40] = 1.9      // history: penalized to 1.9/1.05 ≈ 1.81 < threshold ⇒ dropped from top-2
        logits[2100] = 5.0    // >= suppressFrom, non-eos ⇒ -inf, must never be drawn
        // Sanity: without the penalty 40 would be in the top-2; with it, 20 is.
        let processed = cb0Processed(logits, history: history, frame: frame)
        XCTAssertGreaterThan(processed[20], processed[40], "penalty must drop 40 below 20 (else vacuous)")
        XCTAssertEqual(processed[2100], -.infinity, "range-suppress must -inf token 2100")

        let trials = 4096
        let us = (0..<trials).map { Float($0) / Float(trials) }
        let gpu = try cb0SampleSweep(logits, history: history, frame: frame, uniforms: us, temperature: temp, topK: topK)
        let host = us.map { Qwen3TTSSampler.sampleTopK(processed, temperature: temp, topK: topK, u: $0) }

        XCTAssertTrue(gpu.allSatisfy { $0 == 10 || $0 == 20 }, "GPU must keep only {10,20}: 40 penalized out, 2100 suppressed")
        XCTAssertTrue(gpu.contains(10) && gpu.contains(20), "both kept indices must appear (u is read)")
        let mismatches = zip(gpu, host).filter { $0 != $1 }.count
        XCTAssertLessThanOrEqual(mismatches, 4, "GPU vs host draws differ only at CDF boundary rounding (got \(mismatches)/\(trials))")
        let f10g = Double(gpu.filter { $0 == 10 }.count) / Double(trials)
        let f10h = Double(host.filter { $0 == 10 }.count) / Double(trials)
        XCTAssertEqual(f10g, f10h, accuracy: 0.01, "GPU sampled cb0 frequency must match the host distribution")
    }

    // Two structural cases the {10,20}-frequency gate above can't reach (codex 1b review):
    //  A) the k-th kept (THRESHOLD) token is itself history-penalized — the threshold must use the
    //     PROCESSED value (metal line 80 re-reads through cb0_effective_logit, not the raw logit). A raw
    //     re-read would set the keep cutoff too high and drop the penalized token entirely.
    //  B) topK reaches the suppressed -inf tail — exercises the processed -inf threshold + the
    //     `eff > -INFINITY` keep filter; no suppressed-range index may ever be drawn.
    func testCb0SampleTopKProcessedThresholdAndSuppressedTail() throws {
        let temp: Float = 0.9
        let trials = 4096
        let us = (0..<trials).map { Float($0) / Float(trials) }
        func mism(_ a: [Int], _ b: [Int]) -> Int { zip(a, b).filter { $0 != $1 }.count }

        // A) threshold token is the history-penalized one.
        var la = (0..<cb0N).map { _ in Float(-30) }
        la[10] = 3.0     // top, unpenalized
        la[40] = 2.0     // history → processed 2.0/1.05 ≈ 1.905, the 2nd-largest PROCESSED ⇒ threshold token
        la[20] = 1.5     // below the processed threshold ⇒ excluded
        let pa = cb0Processed(la, history: [40], frame: 5)
        XCTAssertEqual(pa[40], 2.0 / 1.05, accuracy: 1e-5, "token 40 must carry the penalized value")
        XCTAssertGreaterThan(pa[40], pa[20], "the penalized token must still be the kept threshold (else vacuous)")
        let gpuA = try cb0SampleSweep(la, history: [40], frame: 5, uniforms: us, temperature: temp, topK: 2)
        let hostA = us.map { Qwen3TTSSampler.sampleTopK(pa, temperature: temp, topK: 2, u: $0) }
        XCTAssertTrue(gpuA.allSatisfy { $0 == 10 || $0 == 40 }, "kept set is {10,40}; 20 excluded")
        XCTAssertTrue(gpuA.contains(40), "the penalized threshold token must be drawable — a raw threshold re-read would drop it")
        XCTAssertLessThanOrEqual(mism(gpuA, hostA), 4, "GPU vs host where the threshold token is penalized")

        // B) topK reaches past the ~2048 finite audio tokens into the suppressed -inf tail. frame 0 keeps
        // eos (>=suppressFrom) suppressed too, so every index >= suppressFrom is -inf. Fewer trials: each
        // dispatch runs ~topK threshold rounds. A suppressed index carries a high RAW logit, so a raw
        // threshold re-read would also corrupt the cutoff here.
        let tailTrials = 256
        let tailUs = (0..<tailTrials).map { Float($0) / Float(tailTrials) }
        var lb = (0..<cb0N).map { _ in Float(-30) }
        lb[5] = 2.0; lb[6] = 1.0; lb[7] = 0.0
        lb[2100] = 9.0     // suppressed (>= suppressFrom, != eos) but high raw ⇒ processed -inf
        let pb = cb0Processed(lb, history: [], frame: 0)
        XCTAssertEqual(pb[2100], -.infinity, "range-suppress must -inf the high-raw tail token")
        let topKtail = cb0SuppressFrom + 50        // > the 2048 finite tokens ⇒ threshold is -inf
        let gpuB = try cb0SampleSweep(lb, history: [], frame: 0, uniforms: tailUs, temperature: temp, topK: topKtail)
        let hostB = tailUs.map { Qwen3TTSSampler.sampleTopK(pb, temperature: temp, topK: topKtail, u: $0) }
        XCTAssertTrue(gpuB.allSatisfy { $0 < cb0SuppressFrom }, "no suppressed-tail index may be drawn even when topK reaches it")
        XCTAssertLessThanOrEqual(mism(gpuB, hostB), 8, "GPU vs host with topK into the suppressed -inf tail")
    }

    // next_frame_input_f32: the on-GPU 16-codebook gather-sum must match the CPU
    // Qwen3TTSGenerator.nextFrameInput byte-for-byte (same accumulation order → bit-exact), incl. the
    // 15 MTP-table bindings (a swapped/missed buffer index shows here, not just end-to-end).
    func testNextFrameInputF32() throws {
        let pso = try pipeline("next_frame_input_f32.metal", "next_frame_input_f32")
        let dim = 8, vocab0 = 40, vocab = 24
        var rng: UInt64 = 0xD1B54A32D192ED03
        func nf() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) }
        let talkerEmb = (0..<vocab0 * dim).map { _ in nf() }
        let mtpEmbs = (0..<15).map { _ in (0..<vocab * dim).map { _ in nf() } }
        let ttsPad = (0..<dim).map { _ in nf() }
        let codes = [7] + (0..<15).map { ($0 * 3 + 1) % vocab }   // cb0 in talker table, cb1..15 in MTP
        let ref = Qwen3TTSGenerator.nextFrameInput(codes16: codes, talkerCodecEmb: talkerEmb,
                                                   mtpCodecEmbs: mtpEmbs, ttsPadEmbed: ttsPad, dim: dim)
        let codesBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        let cp = codesBuf.contents().bindMemory(to: UInt32.self, capacity: 16)
        for i in 0..<16 { cp[i] = UInt32(codes[i]) }
        let outBuf = outF32(dim)
        encode(pso, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: dim, height: 1, depth: 1)) { enc in
            enc.setBuffer(codesBuf, offset: 0, index: 0); enc.setBuffer(bufF32(ttsPad), offset: 0, index: 1)
            enc.setBuffer(bufF32(talkerEmb), offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            for i in 0..<15 { enc.setBuffer(bufF32(mtpEmbs[i]), offset: 0, index: 4 + i) }
            var d = UInt32(dim); enc.setBytes(&d, length: 4, index: 19)
        }
        let got = readF32(outBuf, dim)
        // Exact (not tolerance): identical accumulation order + pure adds (no FMA) → bit-for-bit.
        for d in 0..<dim { XCTAssertEqual(got[d], ref[d], "nextFrameInput d=\(d)") }
    }

    // gather_rows_f32 / gather_rows_bf16w_f32 (Phase 1a-ii): the ids-from-slot MULTI-row gather the baked
    // TTS front-end uses to gather its text/codec rows. out[t,:] = table[ids[t],:], with a SIGNED id < 0 ⇒
    // a zero row (the projection-only codec sentinel — bit-identical to the host front-end leaving a
    // nil-codec row at zero). The bf16 variant widens bf16→fp32 (bits<<16) bit-identically to the f32
    // table read, matching the host weightRow widen.
    func testGatherRowsMatchesReferenceAndBF16Parity() throws {
        let rows = 40, dim = 16, n = 12
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func nf() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) }
        // bf16-representable table (top 16 bits), so the f32 reference == the bf16-widened read exactly.
        let bits = (0..<rows * dim).map { _ in UInt16(truncatingIfNeeded: nf().bitPattern >> 16) }
        let table = bits.map { Float(bitPattern: UInt32($0) << 16) }
        let ids: [Int32] = [3, -1, 0, 39, 7, -1, 15, 22, -1, 1, 38, 5]   // valid rows + the -1 zero sentinel
        XCTAssertEqual(ids.count, n)
        func gather(_ fn: String, _ tableBuf: MTLBuffer) throws -> [Float] {
            let pso = try pipeline("\(fn).metal", fn)
            let idsBuf = device.makeBuffer(bytes: ids, length: n * 4, options: .storageModeShared)!
            let out = outF32(n * dim)
            encode(pso, MTLSize(width: n * dim, height: 1, depth: 1), MTLSize(width: min(n * dim, 256), height: 1, depth: 1)) { enc in
                enc.setBuffer(tableBuf, offset: 0, index: 0); enc.setBuffer(idsBuf, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
                var nU = UInt32(n), d = UInt32(dim); enc.setBytes(&nU, length: 4, index: 3); enc.setBytes(&d, length: 4, index: 4)
            }
            return readF32(out, n * dim)
        }
        // CPU reference: id < 0 ⇒ zero row, else table[id].
        var ref = [Float](repeating: 0, count: n * dim)
        for t in 0..<n where ids[t] >= 0 { let id = Int(ids[t]); for d in 0..<dim { ref[t * dim + d] = table[id * dim + d] } }
        let bf16Buf = device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared)!

        let gotF32 = try gather("gather_rows_f32", bufF32(table))
        let gotBF16 = try gather("gather_rows_bf16w_f32", bf16Buf)
        for i in 0..<n * dim {
            XCTAssertEqual(gotF32[i], ref[i], "gather_rows_f32 [\(i)]")
            XCTAssertEqual(gotBF16[i], ref[i], "gather_rows_bf16w_f32 [\(i)] (bf16==f32 widen)")
        }
        // Non-vacuity: a -1 row is exactly zero AND a valid row is nonzero (the gather actually ran).
        XCTAssertTrue((0..<dim).allSatisfy { gotF32[1 * dim + $0] == 0 }, "id=-1 must produce a zero row")
        XCTAssertTrue((0..<dim).contains { gotF32[0 * dim + $0] != 0 }, "id=3 must produce a nonzero row")
    }

    // Prefill GEMM (coalesced batched GEMV): gemm_f32/f16 must match the matmul reference at M>1. Each
    // (n,m) output reduces exactly like gemv_f32 on that row, so this is fp32-equivalent (reassociated
    // vs the sequential reference) — a parity break here flips a prefill code before codes==gen_codes does.
    func testGemmMatchesMatmul() throws {
        let N = 1024, K = 2048
        let wF32 = (0..<N * K).map { cos(Float($0) * 0.0013) * 0.4 }
        let bias = (0..<N).map { Float($0 % 7) * 0.01 }
        let wF16 = wF32.map { Float16($0) }
        func refRow(_ x: [Float], _ m: Int, half: Bool) -> [Float] {
            (0..<N).map { n in var a: Float = bias[n]; for k in 0..<K { a += x[m * K + k] * (half ? Float(wF16[n * K + k]) : wF32[n * K + k]) }; return a }
        }
        func runGemm(_ fn: String, _ wBuf: MTLBuffer, M: Int, x: [Float]) throws -> [Float] {
            let pso = try pipeline(fn == "gemm_f16w_f32" ? "gemm_f16w_f32.metal" : "gemm_f32.metal", fn)
            let xBuf = bufF32(x), bBuf = bufF32(bias), outBuf = outF32(M * N)
            encode(pso, MTLSize(width: N * 32, height: M, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return readF32(outBuf, M * N)
        }
        func relL2(_ got: [Float], _ r: [Float]) -> Float {
            var diff: Float = 0, nb: Float = 0
            for i in 0..<r.count { let d = got[i] - r[i]; diff += d * d; nb += r[i] * r[i] }
            return diff.squareRoot() / nb.squareRoot()
        }
        let wF16Buf = device.makeBuffer(bytes: wF16, length: wF16.count * 2, options: .storageModeShared)!
        for M in [1, 7, 30] {   // 30 = the real prefill; 7/1 exercise small-M and the M=1 boundary
            let x = (0..<M * K).map { sin(Float($0) * 0.011) * 1.1 }
            var ref = [Float](), refH = [Float]()
            for m in 0..<M { ref += refRow(x, m, half: false); refH += refRow(x, m, half: true) }
            XCTAssertLessThan(relL2(try runGemm("gemm_f32", bufF32(wF32), M: M, x: x), ref), 0.02, "gemm_f32 M=\(M)")
            XCTAssertLessThan(relL2(try runGemm("gemm_f16w_f32", wF16Buf, M: M, x: x), refH), 0.02, "gemm_f16w M=\(M)")
        }
    }

    // Phase-7-perf PROFILE (opt-in, SMELT_TTS_GEMV_PROFILE=1): pin whether the talker/MTP M=1 decode
    // GEMV is dispatch-launch-bound (→ qkv/gate-up FUSION is the lever) or per-GEMV execution/bandwidth-
    // bound (→ fusion won't help). The decode is a serial chain of ~5000 tiny dependent GEMVs each
    // ~1ms vs ~80µs at bandwidth (~12-20× off) — this distinguishes WHICH latency before the refactor.
    // Two decisive measurements over the shipped coalesced gemv_f32 kernel:
    //   (1) per-GEMV GPU time vs K — a fixed floor at small K that does NOT scale with work = launch/
    //       scheduling overhead (amortizable by fewer dispatches); linear-in-K = weight-read bandwidth.
    //   (2) fused vs split — 1 GEMV (N=6144) vs 3 GEMVs (N=2048), SAME total weight bytes (the qkv
    //       fusion candidate). speedup≈3 → launch-bound (fusion wins); ≈1 → bandwidth-bound (it won't).
    func testGemvDispatchProfile() throws {
        guard env("SMELT_TTS_GEMV_PROFILE") == "1" else { throw XCTSkip("opt-in profile: set SMELT_TTS_GEMV_PROFILE=1") }
        let pso = try pipeline("gemv_f32.metal", "gemv_f32")
        let tg = 32

        // One command buffer of `reps` GEMV dispatches all writing the SAME out buffer: the write-after-
        // write hazard serializes them (default encoder, tracked hazards) = the dependent decode chain,
        // back-to-back with no overlap. `cold`: each rep reads a DISTINCT weight slice cycling a large
        // pool (>L2) so every read misses cache — the production case where each GEMV reads a different
        // layer/proj weight. `cold=false` reuses one weight buffer (cache-assisted; inflates bandwidth).
        // Returns mean per-GEMV GPU µs (gpuEnd-gpuStart, valid after wait) and host wall µs / reps.
        let poolBytes = 512 * 1024 * 1024   // 512 MB ≫ any Apple-GPU L2 → defeats weight caching
        let pool = device.makeBuffer(length: poolBytes, options: .storageModeShared)!
        do {  // fill the pool once with non-trivial values
            let p = pool.contents().bindMemory(to: Float.self, capacity: poolBytes / 4)
            for i in 0..<(poolBytes / 4) { p[i] = cos(Float(i) * 0.0009) * 0.4 }
        }
        func chainGemv(N: Int, K: Int, reps: Int, cold: Bool) -> (gpuUs: Double, hostUs: Double) {
            let wBytes = N * K * 4
            let slices = max(1, poolBytes / wBytes)   // distinct cold slices available in the pool
            let xBuf = bufF32((0..<K).map { sin(Float($0) * 0.01) })
            let bBuf = outF32(N), outBuf = outF32(N)
            let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            for r in 0..<reps {
                // Page-aligned offset into the pool; cycling distinct slices forces cold DRAM reads.
                let off = cold ? ((r % slices) * wBytes) & ~0x3FFF : 0
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(pool, offset: off, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(1), nU = UInt32(N), kU = UInt32(K), hb = UInt32(0)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: N * tg, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cmd.commit(); cmd.waitUntilCompleted()
            let host = (CFAbsoluteTimeGetCurrent() - t0) * 1e6 / Double(reps)
            let gpu = (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6 / Double(reps)
            return (gpu, host)
        }

        _ = chainGemv(N: 2048, K: 2048, reps: 64, cold: true)   // warm pipeline + first-allocation cost

        print("=== GEMV per-dispatch profile: per-GEMV GPU time vs K (N=2048, reps=256, COLD weights) ===")
        print("      K   gpu_us/gemv   host_us/gemv   eff_GB/s   hot_gpu_us")
        for K in [64, 128, 256, 512, 1024, 2048, 4096, 8192] {
            let c = chainGemv(N: 2048, K: K, reps: 256, cold: true)
            let h = chainGemv(N: 2048, K: K, reps: 256, cold: false)
            let bw = Double(2048 * K * 4) / (c.gpuUs * 1e-6) / 1e9
            print(String(format: "  %5d   %9.3f     %9.3f     %6.1f     %9.3f", K, c.gpuUs, c.hostUs, bw, h.gpuUs))
        }

        // RAW dependent chain: out[r] feeds x[r+1] (square N=K, ping-pong buffers), each reading a cold
        // weight slice — the TRUE serial decode chain. The WAW chain above lets the scheduler overlap
        // dispatch r+1's weight prefetch with r's compute (only the tiny output serializes); a RAW chain
        // forbids that. If RAW ≫ WAW, the decode exposes per-dispatch LATENCY (chain length / fusion is
        // the lever); if RAW ≈ WAW, it's purely weight bandwidth (fusion won't help). This is the crux —
        // production attributes ~1ms/GEMV but the isolated WAW kernel is ~50µs (a ~20× gap to explain).
        func rawChainGemv(NK: Int, reps: Int) -> Double {
            let bufs = [bufF32((0..<NK).map { sin(Float($0) * 0.01) }), outF32(NK)]
            let bBuf = outF32(NK)
            let wBytes = NK * NK * 4, slices = max(1, poolBytes / wBytes)
            let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            for r in 0..<reps {
                let off = ((r % slices) * wBytes) & ~0x3FFF
                enc.setBuffer(bufs[r % 2], offset: 0, index: 0); enc.setBuffer(pool, offset: off, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(bufs[(r + 1) % 2], offset: 0, index: 3)
                var mU = UInt32(1), nU = UInt32(NK), kU = UInt32(NK), hb = UInt32(0)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: NK * tg, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
            enc.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
            return (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6 / Double(reps)
        }
        let rawUs = rawChainGemv(NK: 2048, reps: 256)
        let wawUs = chainGemv(N: 2048, K: 2048, reps: 256, cold: true).gpuUs
        print("=== RAW dependent chain vs WAW chain (NK=2048, COLD) ===")
        print(String(format: "  RAW (out→next x): %8.3f us/gemv", rawUs))
        print(String(format: "  WAW (shared out): %8.3f us/gemv   ratio %.2f× (≫1 → latency-exposed serial chain)",
                     wawUs, rawUs / wawUs))

        // (2) qkv fusion test at the talker decode shape (K=2048), COLD.
        let a2048 = chainGemv(N: 2048, K: 2048, reps: 256, cold: true).gpuUs
        let a6144 = chainGemv(N: 6144, K: 2048, reps: 256, cold: true).gpuUs
        let splitStep = 3 * a2048, fusedStep = a6144
        print("=== fused vs split (qkv shape, K=2048, COLD) ===")
        print(String(format: "  split 3×(N=2048): %8.3f us/step", splitStep))
        print(String(format: "  fused 1×(N=6144): %8.3f us/step", fusedStep))
        print(String(format: "  fusion speedup:   %6.2f×   (≈3 → launch-bound/fusion-wins; ≈1 → bandwidth-bound)",
                     splitStep / fusedStep))
    }

    // Phase-7-perf PROFILE (opt-in): per-dispatch GPU cost of the OTHER decode-path kernels at their
    // real decode shapes (talker: frames=1, heads=16, kvHeads=8, headDim=128, cacheLen~40), to pin the
    // per-frame budget after the rms_norm_codec rewrite and decide which kernel to optimize next. Each
    // is chained reps× into one command buffer (gpuEnd-gpuStart / reps = per-dispatch GPU µs).
    func testDecodeKernelProfile() throws {
        guard env("SMELT_TTS_GEMV_PROFILE") == "1" else { throw XCTSkip("opt-in profile: set SMELT_TTS_GEMV_PROFILE=1") }
        let heads = 16, kvHeads = 8, headDim = 128, cacheLen = 40
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        func chain(_ pso: MTLComputePipelineState, grid: MTLSize, tg: MTLSize, reps: Int,
                   _ bind: (MTLComputeCommandEncoder) -> Void) -> Double {
            let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            for _ in 0..<reps { bind(enc); enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
            enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            return (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6 / Double(reps)
        }
        // rms_norm_head_f32 — production grid: one 32-lane SIMD per (frame,head); frames=1.
        let headPso = try pipeline("rms_norm_head_f32.metal", "rms_norm_head_f32")
        let xH = bufF32((0..<qDim).map { sin(Float($0) * 0.01) }), wH = bufF32((0..<headDim).map { 0.9 + Float($0 % 5) * 0.01 })
        let oH = outF32(qDim)
        let headUs = chain(headPso, grid: MTLSize(width: heads * 32, height: 1, depth: 1),
                           tg: MTLSize(width: 32, height: 1, depth: 1), reps: 256) { enc in
            enc.setBuffer(xH, offset: 0, index: 0); enc.setBuffer(wH, offset: 0, index: 1); enc.setBuffer(oH, offset: 0, index: 2)
            var fr = UInt32(1), hh = UInt32(heads), hdd = UInt32(headDim), e = Float(1e-6)
            enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&hh, length: 4, index: 4)
            enc.setBytes(&hdd, length: 4, index: 5); enc.setBytes(&e, length: 4, index: 6)
        }
        // rope_apply_f32 — production grid: one thread per (frame,head,pair); frames=1.
        let ropePso = try pipeline("rope_apply_f32.metal", "rope_apply_f32")
        let xR = bufF32((0..<qDim).map { sin(Float($0) * 0.01) })
        let cR = bufF32((0..<headDim).map { cos(Float($0) * 0.02) }), sR = bufF32((0..<headDim).map { sin(Float($0) * 0.02) })
        let oR = outF32(qDim)
        let ropeTotal = heads * (headDim / 2)
        let ropeUs = chain(ropePso, grid: MTLSize(width: ropeTotal, height: 1, depth: 1),
                           tg: MTLSize(width: min(ropeTotal, 256), height: 1, depth: 1), reps: 256) { enc in
            enc.setBuffer(xR, offset: 0, index: 0); enc.setBuffer(cR, offset: 0, index: 1)
            enc.setBuffer(sR, offset: 0, index: 2); enc.setBuffer(oR, offset: 0, index: 3)
            var fr = UInt32(1), hh = UInt32(heads), hdd = UInt32(headDim)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&hdd, length: 4, index: 6)
        }
        // decode_gqa_attn_f32 — production grid: one 32-lane SIMD per query head (lanes split keys).
        let gqaPso = try pipeline("decode_gqa_attn_f32.metal", "decode_gqa_attn_f32")
        let qG = bufF32((0..<qDim).map { sin(Float($0) * 0.01) })
        let kG = bufF32((0..<cacheLen * kvDim).map { cos(Float($0) * 0.003) }), vG = bufF32((0..<cacheLen * kvDim).map { sin(Float($0) * 0.004) })
        let oG = outF32(qDim)
        let gqaUs = chain(gqaPso, grid: MTLSize(width: heads * 32, height: 1, depth: 1),
                          tg: MTLSize(width: 32, height: 1, depth: 1), reps: 256) { enc in
            enc.setBuffer(qG, offset: 0, index: 0); enc.setBuffer(kG, offset: 0, index: 1)
            enc.setBuffer(vG, offset: 0, index: 2); enc.setBuffer(oG, offset: 0, index: 3)
            var cl = UInt32(cacheLen), hh = UInt32(heads), kvh = UInt32(kvHeads), hdd = UInt32(headDim)
            enc.setBytes(&cl, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&kvh, length: 4, index: 6); enc.setBytes(&hdd, length: 4, index: 7)
        }
        print("=== decode-kernel per-dispatch GPU cost (talker shape, frames=1) ===")
        print(String(format: "  rms_norm_head:  %8.3f us  (×2/layer ×28 = %.2f ms/frame)", headUs, headUs * 2 * 28 / 1000))
        print(String(format: "  rope_apply:     %8.3f us  (×2/layer ×28 = %.2f ms/frame)", ropeUs, ropeUs * 2 * 28 / 1000))
        print(String(format: "  decode_gqa:     %8.3f us  (×1/layer ×28 = %.2f ms/frame)", gqaUs, gqaUs * 28 / 1000))
    }

    // MTP-fused U1: GPU argmax must match the CPU `for v where lg[v] > mx` argmax EXACTLY (lowest
    // index wins ties), including adversarial cases — a flipped index is a flipped code (codes==gen_codes).
    func testArgmaxF32MatchesCPU() throws {
        let pso = try pipeline("argmax_f32.metal", "argmax_f32")
        func cpuArgmax(_ v: [Float]) -> Int { var best = 0, mx = -Float.greatestFiniteMagnitude; for i in v.indices where v[i] > mx { mx = v[i]; best = i }; return best }
        func gpuArgmax(_ v: [Float], slot: Int = 0, slots: Int = 1) -> Int {
            let inBuf = bufF32(v)
            let outBuf = device.makeBuffer(length: slots * 4, options: .storageModeShared)!
            memset(outBuf.contents(), 0, slots * 4)
            encode(pso, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(inBuf, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
                var n = UInt32(v.count), s = UInt32(slot)
                enc.setBytes(&n, length: 4, index: 2); enc.setBytes(&s, length: 4, index: 3)
            }
            return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: slots)[slot])
        }
        // Random vectors at the MTP vocab size.
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func next() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) }
        for _ in 0..<20 {
            let v = (0..<2048).map { _ in next() * 4 }
            XCTAssertEqual(gpuArgmax(v), cpuArgmax(v), "random argmax")
        }
        // Adversarial: exact ties → lowest index wins; ±0; monotone asc/desc; a +inf; a NaN (CPU skips it).
        var ties = [Float](repeating: 0, count: 2048); ties[100] = 5; ties[700] = 5; ties[700 - 1] = 4
        XCTAssertEqual(gpuArgmax(ties), 100, "tie → lowest index")
        var withZeroSign = (0..<2048).map { _ in Float(-1) }; withZeroSign[42] = -0.0; withZeroSign[900] = 0.0  // -0==+0, lowest index 42
        XCTAssertEqual(gpuArgmax(withZeroSign), cpuArgmax(withZeroSign), "signed-zero tie")
        let asc = (0..<2048).map { Float($0) }; XCTAssertEqual(gpuArgmax(asc), 2047, "ascending")
        let desc = (0..<2048).map { Float(2048 - $0) }; XCTAssertEqual(gpuArgmax(desc), 0, "descending")
        var withInf = (0..<2048).map { _ in next() }; withInf[1234] = .infinity; XCTAssertEqual(gpuArgmax(withInf), 1234, "+inf wins")
        var withNaN = (0..<2048).map { Float($0) }; withNaN[5] = .nan  // CPU `>` skips NaN → still picks 2047
        XCTAssertEqual(gpuArgmax(withNaN), cpuArgmax(withNaN), "NaN skipped")
        let allNaN = [Float](repeating: .nan, count: 2048)  // nothing wins → CPU best=0; GPU sentinel → 0 (not OOB)
        XCTAssertEqual(gpuArgmax(allNaN), 0, "all-NaN → index 0 (no OOB sentinel)")
        XCTAssertEqual(cpuArgmax(allNaN), 0, "CPU all-NaN → 0")
        // -FLT_MAX boundary: CPU mx starts at -greatestFiniteMagnitude, so values <= -FLT_MAX (-inf,
        // -FLT_MAX itself) never win — GPU must agree (eligibility is `v > -FLT_MAX`).
        var negBoundary = [Float](repeating: -.infinity, count: 2048)
        negBoundary[0] = -.infinity; negBoundary[1] = -.greatestFiniteMagnitude  // CPU → 0 (nothing > -FLT_MAX)
        XCTAssertEqual(gpuArgmax(negBoundary), cpuArgmax(negBoundary), "-inf/-FLT_MAX never selected → 0")
        var oneAboveFloor = [Float](repeating: -.greatestFiniteMagnitude, count: 2048)
        oneAboveFloor[1000] = -1e30  // the only value > -FLT_MAX → CPU & GPU pick 1000
        XCTAssertEqual(gpuArgmax(oneAboveFloor), cpuArgmax(oneAboveFloor), "single value above -FLT_MAX")
        // Slot write: argmax writes only out[slot].
        let v = (0..<2048).map { _ in next() * 4 }
        XCTAssertEqual(gpuArgmax(v, slot: 3, slots: 8), cpuArgmax(v), "writes the requested slot")
    }

    // MTP-fused U2: gather one table row by an index READ FROM A BUFFER (the chaining path — the
    // index comes from the prior sub-pass's GPU argmax, never the CPU).
    func testGatherRowF32() throws {
        let pso = try pipeline("gather_row_f32.metal", "gather_row_f32")
        let rows = 50, dim = 2048
        let table = (0..<rows * dim).map { Float($0) * 0.001 }
        let tableBuf = bufF32(table)
        for (slot, idx) in [(0, 7), (3, 49), (1, 0)] {
            let idxBuf = bufI32((0..<8).map { _ in Int32(0) })
            idxBuf.contents().bindMemory(to: UInt32.self, capacity: 8)[slot] = UInt32(idx)
            let outBuf = outF32(dim)
            encode(pso, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: min(dim, 256), height: 1, depth: 1)) { enc in
                enc.setBuffer(tableBuf, offset: 0, index: 0); enc.setBuffer(idxBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                var d = UInt32(dim), s = UInt32(slot)
                enc.setBytes(&d, length: 4, index: 3); enc.setBytes(&s, length: 4, index: 4)
            }
            let got = readF32(outBuf, dim)
            let want = Array(table[(idx * dim)..<((idx + 1) * dim)])
            XCTAssertEqual(got, want, "gather row \(idx) via slot \(slot)")
        }
    }

    func testCausalGQAAttnF32MatchesCPU() throws {
        let frames = 30, heads = 4, kvHeads = 2, headDim = 8
        let qDim = heads * headDim, kvDim = kvHeads * headDim, group = heads / kvHeads
        let q = (0..<frames * qDim).map { sin(Float($0) * 0.011) * 1.1 }
        let k = (0..<frames * kvDim).map { cos(Float($0) * 0.013) }
        let v = (0..<frames * kvDim).map { sin(Float($0) * 0.017 + 0.4) * 0.9 }
        let scaling = 1.0 / Float(headDim).squareRoot()
        var ref = [Float](repeating: 0, count: frames * qDim)
        for qh in 0..<heads {
            let kvh = qh / group
            for t in 0..<frames {
                var sc = [Float](repeating: 0, count: t + 1)
                var mx = -Float.greatestFiniteMagnitude
                let qb = t * qDim + qh * headDim
                for s in 0...t {
                    var dot: Float = 0
                    let kb = s * kvDim + kvh * headDim
                    for d in 0..<headDim { dot += q[qb + d] * k[kb + d] }
                    sc[s] = dot * scaling; if sc[s] > mx { mx = sc[s] }
                }
                var denom: Float = 0
                for s in 0...t { sc[s] = expf(sc[s] - mx); denom += sc[s] }
                let ob = t * qDim + qh * headDim
                for s in 0...t {
                    let wgt = sc[s] / denom, vb = s * kvDim + kvh * headDim
                    for d in 0..<headDim { ref[ob + d] += wgt * v[vb + d] }
                }
            }
        }
        let out = try causalGQAAttnDispatch(bufF32(q), bufF32(k), bufF32(v), frames, heads, kvHeads, headDim)
        assertMatch(readF32(out, frames * qDim), ref, "causal_gqa_attn_f32")
    }

    // MARK: - P3-U3: talker forward composed on GPU

    private func loadTensorBF16(_ loader: SafetensorsLoader, _ name: String) -> [Float] {
        let info = loader.tensor(named: name)!
        let count = info.shape.reduce(1, *)
        let src = loader.tensorData(info)
        switch info.dtype {
        case "F32": return Array(UnsafeBufferPointer(start: src.assumingMemoryBound(to: Float.self), count: count))
        case "BF16":
            let bf = UnsafeBufferPointer(start: src.assumingMemoryBound(to: UInt16.self), count: count)
            return bf.map { Float(bitPattern: UInt32($0) << 16) }
        default: fatalError("unsupported dtype \(info.dtype) for \(name)")
        }
    }

    /// Talker 28L forward on GPU (works in [frames, hidden] row-major). cos/sin are the
    /// mRoPE-collapsed 1D tables (theta 1e6). Returns final-norm hidden [frames, hidden].
    private func talkerForwardGPU(_ inputBuf: MTLBuffer, frames: Int, w: Qwen3TTSTalker.Weights,
                                  hidden: Int, heads: Int, kvHeads: Int, headDim: Int, inter: Int,
                                  cBuf: MTLBuffer, sBuf: MTLBuffer, eps: Float = 1e-6) throws -> MTLBuffer {
        let qDim = heads * headDim, ones = [Float](repeating: 1, count: hidden)
        var h = inputBuf
        for layer in w.layers {
            let normed = try rmsNormCodecDispatch(h, frames, hidden, layer.inputNorm, eps: eps)
            var q = try matmulDispatch(normed, frames, hidden, layer.qProj, nil, qDim)
            var k = try matmulDispatch(normed, frames, hidden, layer.kProj, nil, kvHeads * headDim)
            let v = try matmulDispatch(normed, frames, hidden, layer.vProj, nil, kvHeads * headDim)
            q = try rmsNormHeadDispatch(q, frames, heads, headDim, layer.qNorm, eps)
            k = try rmsNormHeadDispatch(k, frames, kvHeads, headDim, layer.kNorm, eps)
            q = try ropeDispatch(q, frames, heads, headDim, cBuf, sBuf)
            k = try ropeDispatch(k, frames, kvHeads, headDim, cBuf, sBuf)
            let attn = try causalGQAAttnDispatch(q, k, v, frames, heads, kvHeads, headDim)
            let proj = try matmulDispatch(attn, frames, qDim, layer.oProj, nil, hidden)
            h = try scaleResidualTCDispatch(proj, h, ones, hidden, frames)  // h + proj
            let normed2 = try rmsNormCodecDispatch(h, frames, hidden, layer.postAttnNorm, eps: eps)
            let gate = try matmulDispatch(normed2, frames, hidden, layer.gateProj, nil, inter)
            let up = try matmulDispatch(normed2, frames, hidden, layer.upProj, nil, inter)
            let act = try swigluDispatch(gate, up, frames * inter)
            let down = try matmulDispatch(act, frames, inter, layer.downProj, nil, hidden)
            h = try scaleResidualTCDispatch(down, h, ones, hidden, frames)  // h + down
        }
        return try rmsNormCodecDispatch(h, frames, hidden, w.normW, eps: eps)
    }

    func testTalkerForwardGPUMatchesCPU() throws {
        let hidden = 32, heads = 4, kvHeads = 2, headDim = 8, inter = 40, frames = 12
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        func rnd(_ n: Int, _ s: Float, _ a: Float = 0.25) -> [Float] { (0..<n).map { sin(Float($0) * s) * a } }
        func nrm(_ n: Int) -> [Float] { (0..<n).map { 0.9 + Float($0 % 5) * 0.01 } }
        var layers: [Qwen3TTSTalker.Layer] = []
        for l in 0..<2 {
            let f = Float(l + 1)
            layers.append(.init(
                inputNorm: nrm(hidden), postAttnNorm: nrm(hidden),
                qProj: rnd(qDim * hidden, 0.011 * f), kProj: rnd(kvDim * hidden, 0.013 * f),
                vProj: rnd(kvDim * hidden, 0.017 * f), oProj: rnd(hidden * qDim, 0.019 * f),
                qNorm: nrm(headDim), kNorm: nrm(headDim),
                gateProj: rnd(inter * hidden, 0.007 * f), upProj: rnd(inter * hidden, 0.009 * f),
                downProj: rnd(hidden * inter, 0.005 * f)))
        }
        let w = Qwen3TTSTalker.Weights(normW: nrm(hidden), layers: layers)
        let input = (0..<frames * hidden).map { cos(Float($0) * 0.021) * 1.0 }
        let (cosT, sinT) = ropeTables(frames: frames, headDim: headDim, theta: 1e6)
        let refCos = cosT, refSin = sinT
        let ref = Qwen3TTSTalker.forward(inputsEmbeds: input, frames: frames, cos: refCos, sin: refSin, w: w,
                                         hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim, inter: inter)
        let out = try talkerForwardGPU(bufF32(input), frames: frames, w: w, hidden: hidden, heads: heads,
                                       kvHeads: kvHeads, headDim: headDim, inter: inter, cBuf: bufF32(cosT), sBuf: bufF32(sinT))
        assertMatch(readF32(out, frames * hidden), ref, "talker forward GPU")
    }

    func testTalkerForwardGPURealWeightsMatchesReference() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
        func t(_ n: String) -> [Float] { loadTensorBF16(loader, "talker.model.\(n)") }
        var layers: [Qwen3TTSTalker.Layer] = []
        for i in 0..<28 {
            let p = "layers.\(i)."
            layers.append(.init(
                inputNorm: t("\(p)input_layernorm.weight"), postAttnNorm: t("\(p)post_attention_layernorm.weight"),
                qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
                vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
                qNorm: t("\(p)self_attn.q_norm.weight"), kNorm: t("\(p)self_attn.k_norm.weight"),
                gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"), downProj: t("\(p)mlp.down_proj.weight")))
        }
        let w = Qwen3TTSTalker.Weights(normW: t("norm.weight"), layers: layers)

        let ie = readBinF32("\(refs)/talker_inputs_embeds.bin")  // [1,T,2048]
        let frames = ie.count / 2048
        let (cosT, sinT) = ropeTables(frames: frames, headDim: 128, theta: 1e6)
        let hiddenBuf = try talkerForwardGPU(bufF32(ie), frames: frames, w: w, hidden: 2048, heads: 16,
                                             kvHeads: 8, headDim: 128, inter: 6144, cBuf: bufF32(cosT), sBuf: bufF32(sinT))
        let refHidden = readBinF32("\(refs)/talker_last_hidden.bin")  // [1,T,2048]
        assertMatch(readF32(hiddenBuf, frames * 2048), refHidden, "talker forward GPU (real weights)")

        // codecHead: [3072,2048] -> logits; assert argmax == reference cb0 codes per frame.
        let headW = loadTensorBF16(loader, "talker.codec_head.weight")
        let logits = try matmulDispatch(hiddenBuf, frames, 2048, headW, nil, 3072)
        let lg = readF32(logits, frames * 3072)
        let codes = readBinI32("\(refs)/gen_codes.bin").map { Int($0) }  // [16, nFrames]
        // The captured prefill's last position predicts gen frame 0's cb0 (greedy = raw argmax here).
        let nFrames = codes.count / 16
        var am = 0, mx = -Float.greatestFiniteMagnitude
        for v in 0..<3072 { let x = lg[(frames - 1) * 3072 + v]; if x > mx { mx = x; am = v } }
        XCTAssertEqual(am, codes[0], "talker codecHead argmax at prefill end != generated cb0[0]")
        _ = nFrames
    }

    // MARK: - P4-U1: MTP transformer (= talkerForwardGPU at MTP dims) + projection + lm_head

    func testMTPTransformerGPUMatchesCPU() throws {
        let hidden = 1024, heads = 16, kvHeads = 8, headDim = 128, inter = 3072, frames = 4
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        func rnd(_ n: Int, _ s: Float, _ a: Float = 0.06) -> [Float] { (0..<n).map { sin(Float($0) * s) * a } }
        func nrm(_ n: Int) -> [Float] { (0..<n).map { 0.95 + Float($0 % 5) * 0.005 } }
        var layers: [Qwen3TTSTalker.Layer] = []
        for l in 0..<5 {
            let f = Float(l + 1)
            layers.append(.init(
                inputNorm: nrm(hidden), postAttnNorm: nrm(hidden),
                qProj: rnd(qDim * hidden, 0.0007 * f), kProj: rnd(kvDim * hidden, 0.0009 * f),
                vProj: rnd(kvDim * hidden, 0.0011 * f), oProj: rnd(hidden * qDim, 0.0013 * f),
                qNorm: nrm(headDim), kNorm: nrm(headDim),
                gateProj: rnd(inter * hidden, 0.0005 * f), upProj: rnd(inter * hidden, 0.0006 * f),
                downProj: rnd(hidden * inter, 0.0004 * f)))
        }
        let w = Qwen3TTSTalker.Weights(normW: nrm(hidden), layers: layers)
        let input = (0..<frames * hidden).map { cos(Float($0) * 0.013) * 0.5 }
        let ref = Qwen3TTSMTP.transformer(inputsEmbeds: input, frames: frames, w: w)
        let (cosT, sinT) = ropeTables(frames: frames, headDim: headDim, theta: 1e6)
        let out = try talkerForwardGPU(bufF32(input), frames: frames, w: w, hidden: hidden, heads: heads,
                                       kvHeads: kvHeads, headDim: headDim, inter: inter, cBuf: bufF32(cosT), sBuf: bufF32(sinT))
        assertMatch(readF32(out, frames * hidden), ref, "MTP transformer GPU (= talkerForwardGPU)")
    }

    func testMTPProjectionAndLmHeadGPUMatchesCPU() throws {
        let rows = 6
        let projW = (0..<1024 * 2048).map { sin(Float($0) * 0.0007) * 0.05 }
        let projB = (0..<1024).map { Float($0) * 0.0001 - 0.05 }
        let projIn = (0..<rows * 2048).map { cos(Float($0) * 0.011) }
        let projRef = Qwen3TTSMTP.projection(projIn, rows: rows, weight: projW, bias: projB)
        let projOut = try matmulDispatch(bufF32(projIn), rows, 2048, projW, projB, 1024)
        assertMatch(readF32(projOut, rows * 1024), projRef, "MTP projection GPU")

        let headW = (0..<2048 * 1024).map { cos(Float($0) * 0.0009) * 0.04 }
        let headIn = (0..<rows * 1024).map { sin(Float($0) * 0.013) }
        let headRef = Qwen3TTSMTP.lmHead(headIn, rows: rows, weight: headW)
        let headOut = try matmulDispatch(bufF32(headIn), rows, 1024, headW, nil, 2048)
        assertMatch(readF32(headOut, rows * 2048), headRef, "MTP lm_head GPU")
    }

    // MARK: - P4-U2: MTP per-frame composition (subTalkerLogits analog) on GPU

    /// Synthetic MTP weights (5 layers at real MTP dims) for the composition gates.
    private func syntheticMTPWeights() -> Qwen3TTSTalker.Weights {
        let hidden = 1024, heads = 16, kvHeads = 8, headDim = 128, inter = 3072
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        func rnd(_ n: Int, _ s: Float, _ a: Float = 0.06) -> [Float] { (0..<n).map { sin(Float($0) * s) * a } }
        func nrm(_ n: Int) -> [Float] { (0..<n).map { 0.95 + Float($0 % 5) * 0.005 } }
        var layers: [Qwen3TTSTalker.Layer] = []
        for l in 0..<5 {
            let f = Float(l + 1)
            layers.append(.init(
                inputNorm: nrm(hidden), postAttnNorm: nrm(hidden),
                qProj: rnd(qDim * hidden, 0.0007 * f), kProj: rnd(kvDim * hidden, 0.0009 * f),
                vProj: rnd(kvDim * hidden, 0.0011 * f), oProj: rnd(hidden * qDim, 0.0013 * f),
                qNorm: nrm(headDim), kNorm: nrm(headDim),
                gateProj: rnd(inter * hidden, 0.0005 * f), upProj: rnd(inter * hidden, 0.0006 * f),
                downProj: rnd(hidden * inter, 0.0004 * f)))
        }
        return Qwen3TTSTalker.Weights(normW: nrm(hidden), layers: layers)
    }

    func testMTPSubTalkerLogitsGPUMatchesCPU() throws {
        let talkerDim = 2048, mtpHidden = 1024, vocab = 2048, groups = 16
        func rnd(_ n: Int, _ s: Float, _ a: Float = 0.05) -> [Float] { (0..<n).map { sin(Float($0) * s) * a } }
        let talkerHidden = (0..<talkerDim).map { cos(Float($0) * 0.011) * 0.4 }
        // cb0 indexes the talker table [3072,*]: exercise a special row (2150). Residuals < 2048.
        var codecIds = (0..<groups).map { (($0 * 13 + 1) % vocab) }
        codecIds[0] = 2150
        let talkerCodecEmb = rnd(3072 * talkerDim, 0.0003)  // [3072,2048]
        // Index-distinct tables/heads so a wrong i-1/i selection can't pass.
        var mtpCodecEmbs: [[Float]] = []
        for i in 0..<15 { mtpCodecEmbs.append(rnd(2048 * talkerDim, 0.00031 + Float(i) * 0.00002)) }
        let projW = rnd(mtpHidden * talkerDim, 0.0004), projB = rnd(mtpHidden, 0.01)
        let mtpW = syntheticMTPWeights()
        var lmHeads: [[Float]] = []
        for i in 0..<15 { lmHeads.append(rnd(vocab * mtpHidden, 0.0006 + Float(i) * 0.00003)) }

        let ref = Qwen3TTSMTP.subTalkerLogits(
            talkerHidden: talkerHidden, codecIds: codecIds, talkerCodecEmb: talkerCodecEmb,
            mtpCodecEmbs: Array(mtpCodecEmbs[0..<14]), projW: projW, projB: projB,
            transformerW: mtpW, lmHeads: lmHeads)

        // Host-assemble the 16-position [16,2048] sequence (pos0 talker hidden, pos1 talker
        // codec_emb[cb0], pos1+i mtp codec_emb[i-1][code_i] for i=1..14).
        var ie = [Float](repeating: 0, count: groups * talkerDim)
        for d in 0..<talkerDim { ie[d] = talkerHidden[d] }
        let c0 = codecIds[0]
        for d in 0..<talkerDim { ie[talkerDim + d] = talkerCodecEmb[c0 * talkerDim + d] }
        for i in 1...(groups - 2) {
            let ci = codecIds[i], table = mtpCodecEmbs[i - 1], base = (1 + i) * talkerDim
            for d in 0..<talkerDim { ie[base + d] = table[ci * talkerDim + d] }
        }
        let proj = try matmulDispatch(bufF32(ie), groups, talkerDim, projW, projB, mtpHidden)
        let (cosT, sinT) = ropeTables(frames: groups, headDim: 128, theta: 1e6)
        let hidden = try talkerForwardGPU(proj, frames: groups, w: mtpW, hidden: mtpHidden, heads: 16,
                                          kvHeads: 8, headDim: 128, inter: 3072, cBuf: bufF32(cosT), sBuf: bufF32(sinT))
        let hid = readF32(hidden, groups * mtpHidden)
        var out = [Float](repeating: 0, count: 15 * vocab)
        for i in 1...(groups - 1) {
            let row = Array(hid[(i * mtpHidden)..<((i + 1) * mtpHidden)])
            let lg = try matmulDispatch(bufF32(row), 1, mtpHidden, lmHeads[i - 1], nil, vocab)
            let lgArr = readF32(lg, vocab)
            for v in 0..<vocab { out[(i - 1) * vocab + v] = lgArr[v] }
        }
        assertMatch(out, ref, "MTP subTalkerLogits GPU")
    }

    // MARK: - P4-U3: free-running greedy mtpGenerate on GPU (real weights, decisive margins)

    /// GPU MTP free-running generation for one frame: given the talker hidden + cb0, run the 15
    /// sequential argmax sub-passes and return cb1..15. Reuses talkerForwardGPU at MTP dims.
    private func mtpGenerateGPU(_ talkerHidden: [Float], cb0: Int, talkerCodecEmb: [Float],
                                mtpCodecEmbs: [[Float]], projW: [Float], projB: [Float],
                                mtpW: Qwen3TTSTalker.Weights, lmHeads: [[Float]]) throws -> [Int] {
        let talkerDim = 2048, mtpHidden = 1024, vocab = 2048
        var seq = talkerHidden
        seq += Array(talkerCodecEmb[(cb0 * talkerDim)..<((cb0 + 1) * talkerDim)])
        var residuals: [Int] = []
        for gs in 0..<15 {
            let rows = seq.count / talkerDim
            let proj = try matmulDispatch(bufF32(seq), rows, talkerDim, projW, projB, mtpHidden)
            let (cosT, sinT) = ropeTables(frames: rows, headDim: 128, theta: 1e6)
            let hidden = try talkerForwardGPU(proj, frames: rows, w: mtpW, hidden: mtpHidden, heads: 16,
                                              kvHeads: 8, headDim: 128, inter: 3072, cBuf: bufF32(cosT), sBuf: bufF32(sinT))
            let hid = readF32(hidden, rows * mtpHidden)
            let last = Array(hid[((rows - 1) * mtpHidden)..<(rows * mtpHidden)])
            let lg = readF32(try matmulDispatch(bufF32(last), 1, mtpHidden, lmHeads[gs], nil, vocab), vocab)
            var am = 0, mx = -Float.greatestFiniteMagnitude
            for v in 0..<vocab where lg[v] > mx { mx = lg[v]; am = v }
            residuals.append(am)
            if gs < 14 { seq += Array(mtpCodecEmbs[gs][(am * talkerDim)..<((am + 1) * talkerDim)]) }
        }
        return residuals
    }

    func testMTPGenerateGPURealWeightsMatchesReference() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
        func t(_ n: String) -> [Float] { loadTensorBF16(loader, n) }
        // Talker (for frame-0 prefill hidden) and MTP weights.
        var tLayers: [Qwen3TTSTalker.Layer] = []
        for i in 0..<28 {
            let p = "talker.model.layers.\(i)."
            tLayers.append(.init(
                inputNorm: t("\(p)input_layernorm.weight"), postAttnNorm: t("\(p)post_attention_layernorm.weight"),
                qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
                vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
                qNorm: t("\(p)self_attn.q_norm.weight"), kNorm: t("\(p)self_attn.k_norm.weight"),
                gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"), downProj: t("\(p)mlp.down_proj.weight")))
        }
        let talkerW = Qwen3TTSTalker.Weights(normW: t("talker.model.norm.weight"), layers: tLayers)
        var mLayers: [Qwen3TTSTalker.Layer] = []
        for i in 0..<5 {
            let p = "talker.code_predictor.model.layers.\(i)."
            mLayers.append(.init(
                inputNorm: t("\(p)input_layernorm.weight"), postAttnNorm: t("\(p)post_attention_layernorm.weight"),
                qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
                vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
                qNorm: t("\(p)self_attn.q_norm.weight"), kNorm: t("\(p)self_attn.k_norm.weight"),
                gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"), downProj: t("\(p)mlp.down_proj.weight")))
        }
        let mtpW = Qwen3TTSTalker.Weights(normW: t("talker.code_predictor.model.norm.weight"), layers: mLayers)
        var lmHeads: [[Float]] = [], mtpEmbs: [[Float]] = []
        for i in 0..<15 { lmHeads.append(t("talker.code_predictor.lm_head.\(i).weight")) }
        for i in 0..<15 { mtpEmbs.append(t("talker.code_predictor.model.codec_embedding.\(i).weight")) }

        // Frame-0 talker hidden = last position of the prefill forward.
        let ie = readBinF32("\(refs)/talker_inputs_embeds.bin")
        let pf = ie.count / 2048
        let (cosT, sinT) = ropeTables(frames: pf, headDim: 128, theta: 1e6)
        let th = readF32(try talkerForwardGPU(bufF32(ie), frames: pf, w: talkerW, hidden: 2048, heads: 16,
                                              kvHeads: 8, headDim: 128, inter: 6144, cBuf: bufF32(cosT), sBuf: bufF32(sinT)), pf * 2048)
        let frame0Hidden = Array(th[((pf - 1) * 2048)..<(pf * 2048)])

        let codes = readBinI32("\(refs)/gen_codes.bin").map { Int($0) }  // [16, nFrames]
        let nFrames = codes.count / 16
        let cb0 = codes[0]  // frame 0, cb0
        let residuals = try mtpGenerateGPU(frame0Hidden, cb0: cb0, talkerCodecEmb: t("talker.model.codec_embedding.weight"),
                                           mtpCodecEmbs: mtpEmbs, projW: t("talker.code_predictor.small_to_mtp_projection.weight"),
                                           projB: t("talker.code_predictor.small_to_mtp_projection.bias"), mtpW: mtpW, lmHeads: lmHeads)
        // Expected residuals = gen_codes[cb1..15, frame 0].
        let expected = (1...15).map { codes[$0 * nFrames + 0] }
        XCTAssertEqual(residuals, expected, "GPU MTP residual codes (frame 0) != reference cb1..15")
    }

    // MARK: - P5-U1: full talker+MTP greedy generation loop on GPU == real model

    func testGenerateGPURealWeightsMatchesReference() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let loader = try SafetensorsLoader(paths: ["\(model)/model.safetensors"])
        func t(_ n: String) -> [Float] { loadTensorBF16(loader, n) }
        func layers(_ prefix: String, _ n: Int) -> [Qwen3TTSTalker.Layer] {
            (0..<n).map { i in let p = "\(prefix)layers.\(i)."
                return .init(inputNorm: t("\(p)input_layernorm.weight"), postAttnNorm: t("\(p)post_attention_layernorm.weight"),
                    qProj: t("\(p)self_attn.q_proj.weight"), kProj: t("\(p)self_attn.k_proj.weight"),
                    vProj: t("\(p)self_attn.v_proj.weight"), oProj: t("\(p)self_attn.o_proj.weight"),
                    qNorm: t("\(p)self_attn.q_norm.weight"), kNorm: t("\(p)self_attn.k_norm.weight"),
                    gateProj: t("\(p)mlp.gate_proj.weight"), upProj: t("\(p)mlp.up_proj.weight"), downProj: t("\(p)mlp.down_proj.weight")) }
        }
        let talkerW = Qwen3TTSTalker.Weights(normW: t("talker.model.norm.weight"), layers: layers("talker.model.", 28))
        let mtpW = Qwen3TTSTalker.Weights(normW: t("talker.code_predictor.model.norm.weight"), layers: layers("talker.code_predictor.model.", 5))
        let codecHeadW = t("talker.codec_head.weight"), talkerCodecEmb = t("talker.model.codec_embedding.weight")
        let mtpProjW = t("talker.code_predictor.small_to_mtp_projection.weight"), mtpProjB = t("talker.code_predictor.small_to_mtp_projection.bias")
        var lmHeads: [[Float]] = [], mtpEmbs: [[Float]] = []
        for i in 0..<15 { lmHeads.append(t("talker.code_predictor.lm_head.\(i).weight")) }
        for i in 0..<15 { mtpEmbs.append(t("talker.code_predictor.model.codec_embedding.\(i).weight")) }
        // tts_pad_embed = text_projection(text_embedding(tts_pad_id)).
        let ttsPadId = readBinI32("\(refs)/prefill_ids.bin").map { Int($0) }[2]
        let textEmb = t("talker.model.text_embedding.weight")
        let ttsPadEmbed = Qwen3TTSTalker.textProjection(
            Array(textEmb[(ttsPadId * 2048)..<((ttsPadId + 1) * 2048)]), rows: 1,
            fc1W: t("talker.text_projection.linear_fc1.weight"), fc1B: t("talker.text_projection.linear_fc1.bias"),
            fc2W: t("talker.text_projection.linear_fc2.weight"), fc2B: t("talker.text_projection.linear_fc2.bias"))

        let codes = readBinI32("\(refs)/gen_codes.bin").map { Int($0) }
        let nFrames = codes.count / 16
        var ie = readBinF32("\(refs)/talker_inputs_embeds.bin")
        var seqLen = ie.count / 2048
        var cb0History: [Int] = []
        var gen: [[Int]] = []
        let nGen = 4  // bounded: full-recompute talker forward is O(T^2)/step; a GPU KV cache is P6/perf.
        for frame in 0..<nGen {
            let (cosT, sinT) = ropeTables(frames: seqLen, headDim: 128, theta: 1e6)
            let hidden = readF32(try talkerForwardGPU(bufF32(ie), frames: seqLen, w: talkerW, hidden: 2048, heads: 16,
                                                      kvHeads: 8, headDim: 128, inter: 6144, cBuf: bufF32(cosT), sBuf: bufF32(sinT)), seqLen * 2048)
            let lastHidden = Array(hidden[((seqLen - 1) * 2048)..<(seqLen * 2048)])
            let rawLogits = readF32(try matmulDispatch(bufF32(lastHidden), 1, 2048, codecHeadW, nil, 3072), 3072)
            let logits = Qwen3TTSGenerator.applyCb0Processors(rawLogits, history: cb0History, frame: frame)
            var cb0 = 0, mx = -Float.greatestFiniteMagnitude
            for v in 0..<3072 where logits[v] > mx { mx = logits[v]; cb0 = v }
            XCTAssertNotEqual(cb0, 2150, "unexpected eos within bounded frames")
            cb0History.append(cb0)
            let cb1_15 = try mtpGenerateGPU(lastHidden, cb0: cb0, talkerCodecEmb: talkerCodecEmb, mtpCodecEmbs: mtpEmbs,
                                            projW: mtpProjW, projB: mtpProjB, mtpW: mtpW, lmHeads: lmHeads)
            let codes16 = [cb0] + cb1_15
            gen.append(codes16)
            ie += Qwen3TTSGenerator.nextFrameInput(codes16: codes16, talkerCodecEmb: talkerCodecEmb,
                                                   mtpCodecEmbs: mtpEmbs, ttsPadEmbed: ttsPadEmbed)
            seqLen += 1
        }
        // gen[frame][codebook] == gen_codes[codebook*nFrames + frame].
        for f in 0..<nGen { for c in 0..<16 {
            XCTAssertEqual(gen[f][c], codes[c * nFrames + f], "GPU generated frame \(f) cb \(c) != reference")
        } }
    }

    // MARK: - P6-U1: catalog registration of the fp32 TTS kernels

    // (pipeline case, shader file, exact entry-point, bufferBindingCount, constantCount).
    // Counts mirror the per-kernel test dispatches above.
    private static let ttsCatalogKernels: [(SmeltPipeline, String, String, Int, Int)] = [
        (.snakeBetaF32,        "snake_beta_f32.metal",        "snake_beta_f32",        4, 2),
        (.conv1dForwardF32,    "conv1d_spatial_f32.metal",    "conv1d_forward_f32",    4, 8),
        (.convTranspose1dF32,  "conv_transpose1d_f32.metal",  "conv_transpose1d_f32",  4, 6),
        (.layerNormCTF32,      "layer_norm_ct_f32.metal",     "layer_norm_ct_f32",     4, 3),
        (.matmulF32,           "matmul_f32.metal",            "matmul_f32",            4, 4),
        (.geluF32,             "activations_f32.metal",       "gelu_f32",              2, 1),
        (.siluF32,             "activations_f32.metal",       "silu_f32",              2, 1),
        (.swigluF32,           "activations_f32.metal",       "swiglu_f32",            3, 1),
        (.rmsNormCodecF32,     "rms_norm_codec_f32.metal",    "rms_norm_codec_f32",    3, 3),
        (.rmsNormHeadF32,      "rms_norm_head_f32.metal",     "rms_norm_head_f32",     3, 4),
        (.ropeApplyF32,        "rope_apply_f32.metal",        "rope_apply_f32",        4, 3),
        (.slidingAttnF32,      "sliding_attn_f32.metal",      "sliding_attn_f32",      4, 4),
        (.causalGQAAttnF32,    "causal_gqa_attn_f32.metal",   "causal_gqa_attn_f32",   4, 4),
        (.rvqGatherSumF32,     "rvq_gather_sum_f32.metal",    "rvq_gather_sum_f32",    4, 4),
        (.scaleResidualF32,    "scale_residual_f32.metal",    "scale_residual_f32",    4, 2),
        (.scaleResidualTCF32,  "scale_residual_f32.metal",    "scale_residual_tc_f32", 4, 3),
        (.clampF32,            "clamp_f32.metal",             "clamp_f32",             2, 3),
        (.decodeGQAAttnF32,    "decode_gqa_attn_f32.metal",   "decode_gqa_attn_f32",   4, 4),
        (.matmulF16WF32,       "matmul_f16w_f32.metal",       "matmul_f16w_f32",       4, 4),
        (.gemvF32,             "gemv_f32.metal",              "gemv_f32",              4, 4),
        (.gemvF16WF32,         "gemv_f16w_f32.metal",         "gemv_f16w_f32",         4, 4),
        (.gemvBF16WF32,        "gemv_bf16w_f32.metal",        "gemv_bf16w_f32",        4, 4),
        (.gemmBF16WF32,        "gemm_bf16w_f32.metal",        "gemm_bf16w_f32",        4, 4),
        (.argmaxF32,           "argmax_f32.metal",            "argmax_f32",            2, 2),
        (.gatherRowF32,        "gather_row_f32.metal",        "gather_row_f32",        3, 2),
        (.gatherRowsF32,       "gather_rows_f32.metal",       "gather_rows_f32",       3, 2),
        (.gatherRowsBF16WF32,  "gather_rows_bf16w_f32.metal", "gather_rows_bf16w_f32", 3, 2),
    ]

    func testTTSKernelsRegisteredInCatalog() throws {
        for (pipe, file, fn, bufCount, constCount) in Self.ttsCatalogKernels {
            let sig = SmeltKernelCatalog.signature(for: pipe)
            // signatures[] is rawValue-indexed, so a misordered append surfaces as a
            // pipeline/name mismatch here.
            XCTAssertEqual(sig.pipeline, pipe, "catalog index drift for \(fn)")
            XCTAssertEqual(sig.metalFunctionName, fn, "catalog name for \(pipe)")
            XCTAssertEqual(sig.bufferBindingCount, bufCount, "\(fn) buffer count")
            XCTAssertEqual(sig.constantCount, constCount, "\(fn) constant count")
            // Exact function presence: the runtime substitutes a placeholder for a
            // missing/misnamed function, so a catalog entry that names a nonexistent
            // entry-point must hard-fail here, not silently no-op at run time.
            _ = try pipeline(file, sig.metalFunctionName)
        }
    }

    // MARK: - P6-U2: CAM checkpoint tensor policy

    func testTTSCheckpointPolicyMapsRealNamesFromCAM() throws {
        typealias SourceDType = Qwen3TTSCheckpointTensorPolicy.SourceDType
        let policy = try qwen3TTSTestCheckpointPolicy()
        XCTAssertEqual(policy.requiredBlocks, [
            "codec-decoder",
            "codec-head",
            "mtp-head",
            "talker",
            "tts-frontend",
        ])

        let cases: [(String, String, SourceDType)] = [
            ("talker.model.layers.0.self_attn.q_proj.weight", "talker", .bf16),
            ("talker.model.norm.weight", "talker", .bf16),
            ("talker.model.text_embedding.weight", "tts-frontend", .bf16),
            ("talker.model.codec_embedding.weight", "talker", .bf16),
            ("talker.codec_head.weight", "codec-head", .bf16),
            ("talker.text_projection.linear_fc1.weight", "tts-frontend", .bf16),
            ("talker.code_predictor.model.layers.0.self_attn.q_proj.weight", "mtp-head", .bf16),
            ("talker.code_predictor.model.norm.weight", "mtp-head", .bf16),
            ("talker.code_predictor.lm_head.3.weight", "mtp-head", .bf16),
            ("talker.code_predictor.model.codec_embedding.2.weight", "mtp-head", .bf16),
            ("talker.code_predictor.small_to_mtp_projection.weight", "mtp-head", .bf16),
            ("decoder.pre_conv.conv.weight", "codec-decoder", .f32),
            ("decoder.pre_transformer.layers.0.input_proj.weight", "codec-decoder", .f32),
            ("decoder.upsample.0.0.conv.weight", "codec-decoder", .f32),
            ("decoder.decoder.6.conv.weight", "codec-decoder", .f32),
            ("decoder.quantizer.rvq_first.output_proj.weight", "codec-decoder", .f32),
            ("decoder.quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum", "codec-decoder", .f32),
        ]
        for (name, block, dtype) in cases {
            let tensor = try XCTUnwrap(try policy.tensor(named: name), "policy tensor \(name)")
            XCTAssertEqual(tensor.name, name)
            XCTAssertEqual(tensor.block, block, "block(\(name))")
            XCTAssertEqual(tensor.sourceDType, dtype, "sourceDType(\(name))")
        }
        XCTAssertEqual(policy.unmatchedRequiredPatterns(in: cases.map(\.0)), [])

        let missingTextEmbedding = cases.map(\.0).filter { $0 != "talker.model.text_embedding.weight" }
        XCTAssertTrue(
            policy.unmatchedRequiredPatterns(in: missingTextEmbedding).contains {
                $0.block == "tts-frontend"
                    && $0.selector == "talker.model.text_embedding.weight"
                    && $0.reason == "tensor-map"
            },
            "partial frontend checkpoint must fail before package build"
        )

        for junk in ["lm_head.weight", "model.embed_tokens.weight", "talkerX.foo", ""] {
            XCTAssertNil(try policy.tensor(named: junk), "policy should skip \(junk)")
        }
    }

    // MARK: - P6-U3: package builder (synthetic, checkpoint-free)

    // Phase-6-U2: fp16 weight packing — half byteLength, dtype tag, and the right predicate.
    func testTTSFP16WeightPackingLayout() {
        typealias B = Qwen3TTSPackageBuilder
        let f32 = B.WeightSpec(name: "talker.model.layers.0.input_layernorm.weight", shape: [2048])
        let f16 = B.WeightSpec(name: "talker.model.layers.0.self_attn.q_proj.weight", shape: [2048, 2048], dtype: .f16)
        XCTAssertEqual(f32.dataBytes, 2048 * 4)
        XCTAssertEqual(f16.dataBytes, 2048 * 2048 * 2)  // fp16 = half of fp32
        let page = 16384
        let (entries, _) = B.planLayout([f32, f16], pageSize: page)
        XCTAssertNil(entries[0].dtype)                 // f32 → absent (back-compat)
        XCTAssertEqual(entries[1].dtype, "f16")
        XCTAssertEqual(Int(entries[1].byteLength), (2048 * 2048 * 2 + page - 1) / page * page)
        // Predicate: only talker/MTP layer projection weights become fp16.
        XCTAssertTrue(B.isFP16Candidate("talker.model.layers.5.self_attn.q_proj.weight"))
        XCTAssertTrue(B.isFP16Candidate("talker.code_predictor.model.layers.2.mlp.down_proj.weight"))
        XCTAssertFalse(B.isFP16Candidate("talker.model.layers.0.input_layernorm.weight"))
        XCTAssertFalse(B.isFP16Candidate("talker.model.layers.0.self_attn.q_norm.weight"))
        XCTAssertFalse(B.isFP16Candidate("talker.codec_head.weight"))
        XCTAssertFalse(B.isFP16Candidate("talker.code_predictor.lm_head.0.weight"))
        XCTAssertFalse(B.isFP16Candidate("talker.text_projection.linear_fc1.weight"))
        XCTAssertFalse(B.isFP16Candidate("decoder.pre_conv.conv.weight"))
    }

    func testTTSPackageBuilderSyntheticRoundTrip() throws {
        typealias B = Qwen3TTSPackageBuilder
        // Small codec-only weights exercise layout, weights.bin write, the real
        // metallib compile, and manifest without claiming a runnable talker graph.
        let specs: [B.WeightSpec] = [
            .init(name: "decoder.pre_conv.conv.bias", shape: [10]),
            .init(name: "decoder.ups.0.bias", shape: [16]),
            .init(name: "decoder.post_conv.conv.bias", shape: [8]),
        ]
        let page = Int(getpagesize())
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-synth-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"

        // Deterministic fill: f[i] = name.count + i, so each slice is independently checkable.
        try B.build(
            specs: specs,
            fill: { spec, _, slice in
                let f = slice.bindMemory(to: Float.self)
                for i in 0..<spec.elementCount { f[i] = Float(spec.name.count + i) }
            },
            pipelines: B.ttsPipelineNames,
            modelName: "qwen3-tts-synth", eosTokens: [2150],
            shaderDir: shaderDir, outputPath: pkg)

        // 1. Manifest: page-aligned offsets, page-rounded byteLengths, monotonic + non-overlapping.
        let mdata = try Data(contentsOf: URL(fileURLWithPath: "\(pkg)/manifest.json"))
        let m = try Qwen3TTSManifest.decode(from: mdata)
        XCTAssertEqual(m.eosTokens, [2150])
        XCTAssertEqual(m.pipelines, B.ttsPipelineNames)
        XCTAssertNil(m.validation, "codec-only synthetic packages must not claim a TTS TTFA gate")
        XCTAssertEqual(m.blocks, .qwen3TTSCodecDecoder)
        XCTAssertEqual(m.loop, .qwen3TTSCodecDecoder)
        try m.validateQwen3TTSValidation()
        XCTAssertEqual(m.weights.count, specs.count)
        XCTAssertEqual(Int(m.totalBytes) % page, 0, "totalBytes page-rounded")
        for (e, s) in zip(m.weights, specs) {
            XCTAssertEqual(e.name, s.name)
            XCTAssertEqual(e.shape, s.shape)
            XCTAssertEqual(Int(e.offset) % page, 0, "\(s.name) offset page-aligned")
            XCTAssertEqual(Int(e.byteLength) % page, 0, "\(s.name) byteLength page-rounded")
            XCTAssertGreaterThanOrEqual(Int(e.byteLength), s.dataBytes)
        }
        for i in 1..<m.weights.count {
            XCTAssertGreaterThanOrEqual(
                m.weights[i].offset, m.weights[i - 1].offset + m.weights[i - 1].byteLength,
                "weights overlap at \(i)")
        }
        XCTAssertLessThanOrEqual(
            m.weights.last!.offset + m.weights.last!.byteLength, m.totalBytes)

        // 2. weights.bin is exactly totalBytes and each slice round-trips.
        let wsize = (try FileManager.default.attributesOfItem(atPath: "\(pkg)/weights.bin")[.size] as! NSNumber).uint64Value
        XCTAssertEqual(wsize, m.totalBytes)
        let wdata = try Data(contentsOf: URL(fileURLWithPath: "\(pkg)/weights.bin"))
        wdata.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for (e, s) in zip(m.weights, specs) {
                let f = raw.baseAddress!.advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)
                for i in 0..<s.elementCount {
                    XCTAssertEqual(f[i], Float(s.name.count + i), "\(s.name)[\(i)]")
                }
            }
        }

        // 3. Every manifest pipeline builds from the package's own model.metallib
        // (exact function presence — the runtime would placeholder-substitute a miss).
        let lib = try device.makeLibrary(URL: URL(fileURLWithPath: "\(pkg)/model.metallib"))
        for fn in m.pipelines {
            let f = try XCTUnwrap(lib.makeFunction(name: fn), "metallib missing \(fn)")
            _ = try device.makeComputePipelineState(function: f)
        }
    }

    // u4 packing: planLayout lays out the nibble/scale/bias block (page-aligned, relative offsets on
    // the Entry), validateU4 enforces the kernel contract, and a build round-trips the exact packed
    // bytes at those offsets. (The end-to-end cosine-vs-bf16 gate exercises fillU4's BF16 source path.)
    func testU4PackLayoutAndRoundTrip() throws {
        typealias B = Qwen3TTSPackageBuilder
        let N = 128, K = 256, g = 64, page = Int(getpagesize())
        let spec = B.WeightSpec(name: "decoder.u4_pack_layout.weight",
                                shape: [N, K], dtype: .u4, groupSize: g)
        let groups = (K + g - 1) / g

        // 1. planLayout: u4 metadata + page-aligned block sub-regions.
        let (entries, total) = B.planLayout([spec], pageSize: page)
        let e = entries[0]
        XCTAssertEqual(e.dtype, "u4")
        XCTAssertEqual(e.groupSize, g)
        XCTAssertEqual(Int(e.offset) % page, 0, "block page-aligned")
        XCTAssertEqual(Int(e.scaleOffset!) % page, 0, "scale offset (relative) page-aligned")
        XCTAssertEqual(Int(e.biasOffset!) % page, 0, "bias offset (relative) page-aligned")
        XCTAssertEqual(Int(e.scaleByteLength!), N * groups * 2)
        XCTAssertEqual(Int(e.biasByteLength!), N * groups * 2)
        XCTAssertLessThan(e.scaleOffset!, e.biasOffset!)
        let biasRegionEnd = e.biasOffset! + ((e.biasByteLength! + UInt64(page) - 1) & ~(UInt64(page) - 1))
        XCTAssertLessThanOrEqual(biasRegionEnd, e.byteLength)
        XCTAssertEqual(Int(e.byteLength) % page, 0)
        XCTAssertLessThanOrEqual(e.offset + e.byteLength, total)

        // 2. validateU4 rejects malformed u4 (odd-ish cols, bad group_size, too many groups).
        XCTAssertThrowsError(try B.validateU4(B.WeightSpec(name: "w", shape: [4, 6], dtype: .u4, groupSize: 64)), "cols%4!=0")
        XCTAssertThrowsError(try B.validateU4(B.WeightSpec(name: "w", shape: [4, 64], dtype: .u4, groupSize: 6)), "group_size%4!=0")
        XCTAssertThrowsError(try B.validateU4(B.WeightSpec(name: "w", shape: [4, 64], dtype: .u4, groupSize: nil)), "missing group_size")
        XCTAssertThrowsError(try B.validateU4(B.WeightSpec(name: "w", shape: [4, 256 * 32], dtype: .u4, groupSize: 4)), "groups>256")
        XCTAssertNoThrow(try B.validateU4(spec))

        // 3. Round-trip: build a package whose u4 fill writes precomputed packed bytes at the planned
        // offsets; weights.bin must hold those exact nibble/scale/bias bytes (the relative-offset contract).
        var rng: UInt64 = 0xA5A5_1234_DEAD_BEEF
        let w = (0..<N * K).map { _ -> Float in
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27
            return Float(Int32(truncatingIfNeeded: rng)) / Float(Int32.max) * 0.3
        }
        let packed = SmeltAffineU4.quantize(w, rows: N, cols: K, groupSize: g)
        let pkg = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen3tts-u4-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        try B.build(
            specs: [spec],
            fill: { _, entry, slice in
                memcpy(slice.baseAddress!, packed.nibbles, packed.nibbles.count)
                _ = packed.scales.withUnsafeBytes { memcpy(slice.baseAddress! + Int(entry.scaleOffset!), $0.baseAddress!, $0.count) }
                _ = packed.biases.withUnsafeBytes { memcpy(slice.baseAddress! + Int(entry.biasOffset!), $0.baseAddress!, $0.count) }
            },
            pipelines: B.ttsPipelineNames, modelName: "qwen3-tts-u4", eosTokens: [2150],
            shaderDir: shaderDir, outputPath: pkg)

        let m = try Qwen3TTSManifest.decode(from: Data(contentsOf: URL(fileURLWithPath: "\(pkg)/manifest.json")))
        let me = m.weights[0]
        let wdata = try Data(contentsOf: URL(fileURLWithPath: "\(pkg)/weights.bin"))
        wdata.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let blockBase = raw.baseAddress!.advanced(by: Int(me.offset))
            let nib = blockBase.assumingMemoryBound(to: UInt8.self)
            for i in 0..<packed.nibbles.count { XCTAssertEqual(nib[i], packed.nibbles[i], "nibble \(i)") }
            let sc = blockBase.advanced(by: Int(me.scaleOffset!)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<packed.scales.count { XCTAssertEqual(sc[i], packed.scales[i], "scale \(i)") }
            let bi = blockBase.advanced(by: Int(me.biasOffset!)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<packed.biases.count { XCTAssertEqual(bi[i], packed.biases[i], "bias \(i)") }
        }
    }

    // MARK: - P6-U4a: GPU driver package load (per-weight bytesNoCopy buffers)

    func testTTSGPULoaderRoundTrip() throws {
        typealias B = Qwen3TTSPackageBuilder
        let specs: [B.WeightSpec] = [
            .init(name: "decoder.pre_conv.conv.bias", shape: [10]),
            .init(name: "decoder.post_conv.conv.bias", shape: [16]),
        ]
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-load-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        try B.build(
            specs: specs,
            fill: { spec, _, slice in
                let f = slice.bindMemory(to: Float.self)
                for i in 0..<spec.elementCount { f[i] = Float(spec.name.count * 100 + i) }
            },
            pipelines: B.ttsPipelineNames,
            modelName: "qwen3-tts-loadtest", eosTokens: [2150],
            shaderDir: shaderDir, outputPath: pkg)

        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        for fn in B.ttsPipelineNames {
            XCTAssertNotNil(gpu.pipeline(fn), "pipeline \(fn)")
        }
        // Each weight is a bytesNoCopy slice over the live mmap; reading its contents
        // proves the slice maps the right bytes at its page-aligned offset.
        for s in specs {
            let buf = try XCTUnwrap(gpu.weight(s.name), "weight \(s.name)")
            XCTAssertGreaterThanOrEqual(buf.length, s.dataBytes)
            XCTAssertEqual(gpu.weightShape(s.name), s.shape)
            let f = buf.contents().bindMemory(to: Float.self, capacity: s.elementCount)
            for i in 0..<s.elementCount {
                XCTAssertEqual(f[i], Float(s.name.count * 100 + i), "\(s.name)[\(i)]")
            }
        }
        XCTAssertNil(gpu.weight("does.not.exist"))
    }

    // The real checkpoint's generation_config.json (do_sample=true) lands in manifest.decode.
    func testTTSPackageCarriesDecodePolicy() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-decode-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
            checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: FileManager.default.currentDirectoryPath + "/Resources/Shaders",
            outputPath: pkg, projDType: .bf16)   // bf16: the decode policy is dtype-agnostic, and f32 full-pipeline is no longer buildable
        let manifest = try Qwen3TTSManifest.decode(
            from: Data(contentsOf: URL(fileURLWithPath: "\(pkg)/manifest.json")))
        let d = try XCTUnwrap(manifest.decode, "manifest must carry the decode block from generation_config.json")
        XCTAssertEqual(manifest.validation?.performanceGate, SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        XCTAssertEqual(
            manifest.validation?.performanceProfile,
            SmeltPackagePerformanceProfiles.profile(for: SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        )
        XCTAssertEqual(
            manifest.validation?.structureProfile?.id,
            SmeltPackageStructureProfileID.qwen3TTSRunnable
        )
        XCTAssertEqual(
            manifest.validation?.structureProfile?.requiredFiles.sorted(),
            ([
                "manifest.json",
                "model.metallib",
                "weights.bin",
                "trunk",
                "trunk-mtp",
            ] + Qwen3TTSManifest.requiredTokenizerFiles).sorted()
        )
        XCTAssertEqual(
            manifest.validation?.structureProfile?.requiredPipelines.sorted(),
            Qwen3TTSPackageBuilder.ttsPipelineNames.sorted()
        )
        let graph = try XCTUnwrap(manifest.blocks)
        XCTAssertEqual(
            manifest.validation?.structureProfile?.requiredRoutes.sorted(),
            graph.runtimeRouteSignatures.sorted()
        )
        XCTAssertTrue(d.doSample, "checkpoint generation_config has do_sample=true")
        XCTAssertEqual(d.temperature, 0.9, accuracy: 1e-6)
        XCTAssertEqual(d.topK, 50)
        XCTAssertEqual(d.subtalkerTemperature, 0.9, accuracy: 1e-6)
        XCTAssertEqual(d.subtalkerTopK, 50)

        // The driver resolves .packageDefault → sampling (do_sample=true) with the manifest's params.
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let resolved = try XCTUnwrap(gpu.resolveDecode(.packageDefault), ".packageDefault must sample when do_sample")
        XCTAssertEqual(resolved.talkerTemperature, 0.9, accuracy: 1e-6)
        XCTAssertEqual(resolved.talkerTopK, 50)
        XCTAssertEqual(resolved.subtalkerTemperature, 0.9, accuracy: 1e-6)
        XCTAssertEqual(resolved.subtalkerTopK, 50)
        // .packageDefault draws a fresh seed per call, so repeated calls vary (unseeded model behavior).
        XCTAssertNotEqual(resolved.seed, gpu.resolveDecode(.packageDefault)?.seed, ".packageDefault seed varies per call")
        XCTAssertNil(gpu.resolveDecode(.greedy), ".greedy must resolve to nil (deterministic argmax)")
        XCTAssertEqual(gpu.resolveDecode(.sample(.init(seed: 42)))?.seed, 42, ".sample passes through")
        // .sampleSeeded: reproducible (fixed seed) with the PACKAGE's params, not Config's.
        let seeded = try XCTUnwrap(gpu.resolveDecode(.sampleSeeded(7)), ".sampleSeeded must sample when do_sample")
        XCTAssertEqual(seeded.seed, 7)
        XCTAssertEqual(gpu.resolveDecode(.sampleSeeded(7))?.seed, 7, ".sampleSeeded is deterministic")
        XCTAssertEqual(seeded.talkerTemperature, resolved.talkerTemperature, "same package temp as .packageDefault")
        XCTAssertEqual(seeded.talkerTopK, resolved.talkerTopK)
    }

    // MARK: - P6-U4b: packaged codec graph (decodeCodec) == real model

    func testTTSGPUDecodeCodecRealWeightsMatchesReference() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        // Build a codec-only .smeltpkg from speech_tokenizer/model.safetensors (codec
        // weights are all F32), then decode through the package's own driver + metallib.
        let loader = try SafetensorsLoader(paths: ["\(model)/speech_tokenizer/model.safetensors"])
        let policy = try qwen3TTSTestCheckpointPolicy()
        let codec = try loader.tensors
            .filter { try policy.tensor(named: $0.name)?.block == "codec-decoder" }
            .sorted { $0.name < $1.name }
        XCTAssertFalse(codec.isEmpty, "no decoder.* tensors found")
        let specs = codec.map { Qwen3TTSPackageBuilder.WeightSpec(name: $0.name, shape: $0.shape) }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-codec-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        try Qwen3TTSPackageBuilder.build(
            specs: specs,
            fill: { spec, _, slice in
                let info = loader.tensor(named: spec.name)!
                precondition(info.dtype == "F32", "\(spec.name) is \(info.dtype)")
                memcpy(slice.baseAddress!, loader.tensorData(info), spec.dataBytes)
            },
            pipelines: Qwen3TTSPackageBuilder.ttsPipelineNames,
            modelName: "qwen3-tts-codec", eosTokens: [2150],
            shaderDir: shaderDir, outputPath: pkg)

        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let codes = readBinI32("\(refs)/input_codes.bin")  // [16, frames]
        let frames = codes.count / 16
        let wav = try gpu.decodeCodec(codes: codes, frames: frames)
        let ref = readBinF32("\(refs)/stage_decoder_wav.bin")
        assertMatch(wav, ref, "decodeCodec packaged == real model")
    }

    // MARK: - P6-U4c/U5: full packaged pipeline (embeds -> codes -> 24kHz wav) == real model

    func testTTSGPUPackagedGenerateMatchesRealModel() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        // Build the FULL .smeltpkg from the real checkpoint (both safetensors) via the real
        // builder path, then load it through the driver.
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-full-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        // bf16 ships the compiled trunks (the only path that runs generation post-Phase-4) and is
        // bit-exact to the f32 reference for these BF16-source weights.
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
                                         checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
                                         shaderDir: shaderDir, outputPath: pkg, projDType: .bf16)

        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)

        // Prefill embeds + tts_pad id, exactly as the P5 generate reference.
        let ie = readBinF32("\(refs)/talker_inputs_embeds.bin")
        let seqLen = ie.count / 2048
        let ttsPadId = readBinI32("\(refs)/prefill_ids.bin").map { Int($0) }[2]
        let refCodes = readBinI32("\(refs)/gen_codes.bin").map { Int($0) }
        let nFrames = refCodes.count / 16

        // (U4c) talker+MTP packaged generation == the real model's codes, all frames.
        let gen = try gpu.generateCodes(inputsEmbeds: ie, seqLen: seqLen, ttsPadId: ttsPadId, maxFrames: nFrames)
        XCTAssertEqual(gen.count, nFrames, "generated frame count")
        for f in 0..<nFrames { for c in 0..<16 {
            XCTAssertEqual(gen[f][c], refCodes[c * nFrames + f], "packaged code frame \(f) cb \(c)")
        } }

        // (U5) decode the generated codes through the packaged codec == real model's wav.
        var codes = [Int32](repeating: 0, count: 16 * nFrames)
        for f in 0..<nFrames { for c in 0..<16 { codes[c * nFrames + f] = Int32(gen[f][c]) } }
        let wav = try gpu.decodeCodec(codes: codes, frames: nFrames)
        assertMatch(wav, readBinF32("\(refs)/stage_decoder_wav.bin"), "packaged generate == real model wav")
    }

    // BF16 weight storage: the proj weights packed bf16 (the source's native dtype, widened to fp32
    // in-kernel) must produce BIT-IDENTICAL codes + wav to the fp32 package — fp32 storage was just the
    // bf16 values zero-padded — at half their footprint.
    func testTTSGPUPackagedBF16GenerateMatchesRealModel() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-bf16-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
                                         checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
                                         shaderDir: shaderDir, outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        XCTAssertEqual(gpu.weightDType("talker.model.layers.0.self_attn.q_proj.weight"), .bf16, "proj weight should be bf16")
        XCTAssertEqual(gpu.weightDType("talker.model.layers.0.input_layernorm.weight"), .f32, "norms stay fp32")
        XCTAssertEqual(gpu.weightDType("talker.code_predictor.model.layers.0.mlp.down_proj.weight"), .bf16, "MTP proj bf16")
        XCTAssertEqual(gpu.weightDType("talker.model.text_embedding.weight"), .bf16, "text_embedding bf16 (gathered via weightRow)")
        XCTAssertEqual(gpu.weightDType("talker.codec_head.weight"), .bf16, "codec_head bf16")
        XCTAssertEqual(gpu.weightDType("talker.code_predictor.lm_head.0.weight"), .bf16, "lm_head bf16")
        XCTAssertEqual(gpu.weightDType("talker.code_predictor.small_to_mtp_projection.weight"), .bf16, "small_to_mtp bf16")

        let ie = readBinF32("\(refs)/talker_inputs_embeds.bin")
        let seqLen = ie.count / 2048
        let ttsPadId = readBinI32("\(refs)/prefill_ids.bin").map { Int($0) }[2]
        let refCodes = readBinI32("\(refs)/gen_codes.bin").map { Int($0) }
        let nFrames = refCodes.count / 16

        let gen = try gpu.generateCodes(inputsEmbeds: ie, seqLen: seqLen, ttsPadId: ttsPadId, maxFrames: nFrames)
        XCTAssertEqual(gen.count, nFrames, "generated frame count")
        for f in 0..<nFrames { for c in 0..<16 {
            XCTAssertEqual(gen[f][c], refCodes[c * nFrames + f], "bf16 code frame \(f) cb \(c)")
        } }
        var codes = [Int32](repeating: 0, count: 16 * nFrames)
        for f in 0..<nFrames { for c in 0..<16 { codes[c * nFrames + f] = Int32(gen[f][c]) } }
        let wav = try gpu.decodeCodec(codes: codes, frames: nFrames)
        assertMatch(wav, readBinF32("\(refs)/stage_decoder_wav.bin"), "bf16 packaged generate == real model wav")
    }

    // U1 CustomVoice gate (docs/qwen3-tts-variants-plan.md): named-speaker / Auto-nothink /
    // no-instruct / dialect-remap parity against the real model, per case dir captured by
    // tools/dump-voice-reference.py. Checks per case: front-end inputsEmbeds match,
    // frame-0 codes exactly equal across all 16 codebooks (frame-0 argmax margins are 8-19
    // logits — robust; free-run greedy parity is brittle-by-construction on this
    // do_sample model: from frame 1 the margins collapse to 0.01-0.5 and CPU-vs-Metal fp
    // noise flips them — the established frame-0 methodology from the int4 work), and the
    // full REF code sequence decoded through our codec == the reference wav.
    func testTTSGPUCustomVoiceMatchesRealModel() throws {
        guard let pkgPath = env("SMELT_VOICE_VARIANT_PACKAGE"), let refs = env("SMELT_VOICE_VARIANT_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_VARIANT_PACKAGE + SMELT_VOICE_VARIANT_REFS")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkgPath, device: device)
        let fm = FileManager.default
        let caseDirs = try fm.contentsOfDirectory(atPath: refs).sorted()
            .filter { fm.fileExists(atPath: "\(refs)/\($0)/case.json") }
        XCTAssertFalse(caseDirs.isEmpty, "no case dirs under \(refs)")
        for name in caseDirs {
            let dir = "\(refs)/\(name)"
            let caseJSON = try JSONSerialization.jsonObject(
                with: Data(contentsOf: URL(fileURLWithPath: "\(dir)/case.json"))) as! [String: Any]
            let text = caseJSON["text"] as! String
            let language = caseJSON["language"] as! String
            let speaker = caseJSON["speaker"] as? String
            let instruct = caseJSON["instruct"] as? String   // JSON null → nil
            let dim = caseJSON["embeds_dim"] as! Int

            let refIE = readBinF32("\(dir)/talker_inputs_embeds.bin")
            let ie = try gpu.inputsEmbeds(text: text, instruct: instruct,
                                          language: language, speaker: speaker)
            XCTAssertEqual(ie.embeds.count, refIE.count,
                           "\(name): prefill length (T×\(dim)) — assembly branch mismatch?")
            assertMatch(ie.embeds, refIE, "\(name): front-end inputsEmbeds")

            let seqLen = refIE.count / dim
            let ttsPadId = readBinI32("\(dir)/prefill_ids.bin").map { Int($0) }[2]
            let refCodes = readBinI32("\(dir)/gen_codes.bin").map { Int($0) }
            let nFrames = refCodes.count / 16

            // Frame-0, driven from the REF embeds (isolates talker+MTP from front-end rounding):
            // cb0 identity (its margins are decisive on every size), then the 15 MTP sub-passes
            // TEACHER-FORCED to the reference codes so each sub-pass's logits are comparable —
            // identity is additionally asserted only where the reference's own top-2 margin is
            // ≥ 0.5 logits (the 0.6B's sub-chain margins are intrinsically thin; framework §6).
            let margins = (caseJSON["frame0_margins"] as! [Any]).map { ($0 as! NSNumber).doubleValue }
            let mtpVocab = caseJSON["mtp_vocab"] as! Int
            let refCb0Logits = readBinF32("\(dir)/frame0_cb0_logits.bin")
            let refMtpLogits = readBinF32("\(dir)/frame0_mtp_logits.bin")
            let refFrame0 = (0..<16).map { refCodes[$0 * nFrames] }
            var ourCb0Logits: [Float] = []
            var ourMtpLogits: [[Float]] = []
            let gen = try gpu.generateCodes(inputsEmbeds: refIE, seqLen: seqLen,
                                            ttsPadId: ttsPadId, maxFrames: 1,
                                            cb0LogitsTap: { if $0 == 0 { ourCb0Logits = $1 } },
                                            mtpLogitsTap: { ourMtpLogits = $0 },
                                            mtpTeacherCodes: Array(refFrame0[1...]))
            XCTAssertEqual(gen.count, 1, "\(name): frame-0 generated")
            XCTAssertEqual(gen[0][0], refFrame0[0], "\(name): frame-0 cb0 identity")
            assertMatch(ourCb0Logits, refCb0Logits, "\(name): frame-0 cb0 logits")
            XCTAssertEqual(ourMtpLogits.count, 15, "\(name): MTP logits tapped")
            for i in 0..<15 {
                let refSlice = Array(refMtpLogits[(i * mtpVocab)..<((i + 1) * mtpVocab)])
                if ProcessInfo.processInfo.environment["SMELT_TTS_CV_GATE_DEBUG"] == "1" {
                    var dot: Float = 0, na: Float = 0, nb: Float = 0
                    for k in 0..<mtpVocab { dot += ourMtpLogits[i][k] * refSlice[k]; na += ourMtpLogits[i][k] * ourMtpLogits[i][k]; nb += refSlice[k] * refSlice[k] }
                    print(String(format: "[cv-gate] %@ mtp%d cosine=%.6f", name, i, dot / (na.squareRoot() * nb.squareRoot())))
                }
                assertMatch(ourMtpLogits[i], refSlice, "\(name): frame-0 mtp\(i) logits")
                if margins[i + 1] >= 0.5 {
                    var best = 0
                    for v in 1..<mtpVocab where ourMtpLogits[i][v] > ourMtpLogits[i][best] { best = v }
                    XCTAssertEqual(best, refFrame0[i + 1],
                                   "\(name): frame-0 cb\(i + 1) identity (ref margin \(margins[i + 1]))")
                }
            }

            // Full reference code sequence through our codec == reference wav.
            var codes = [Int32](repeating: 0, count: 16 * nFrames)
            for i in 0..<codes.count { codes[i] = Int32(refCodes[i]) }
            let wav = try gpu.decodeCodec(codes: codes, frames: nFrames)
            assertMatch(wav, readBinF32("\(dir)/stage_decoder_wav.bin"), "\(name): wav from ref codes")
        }
    }

    // MARK: - P7-U3: literal text → 24 kHz from the .smeltpkg alone (front-end + bundled tokenizer)

    func testTTSGPUPackagedTextToWavMatchesRealModel() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-text-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        // bf16 ships the compiled trunks (the only generation path post-Phase-4); bit-exact to the
        // f32 reference for these BF16-source weights, so the wav still matches the real model.
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
                                         checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
                                         shaderDir: shaderDir, outputPath: pkg, projDType: .bf16)

        // The package bundles its own tokenizer/config — nothing but the source strings goes in.
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        XCTAssertNotNil(gpu.tokenizer, "full package must bundle the tokenizer")

        // maxFrames well above the 26-frame utterance so the run stops on the model's own codec
        // EOS, not the cap — proving the full text→wav path including natural termination.
        let wav = try gpu.generate(
            text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.",
            language: "English", maxFrames: 64, decode: .greedy)   // bit-exact gate forces greedy
        assertMatch(wav, readBinF32("\(refs)/stage_decoder_wav.bin"), "packaged text→wav == real model")
    }

    // cb0 temperature/top-k sampling must break the greedy degeneration: on the pangram + "speak
    // slowly" prompt, greedy argmax collapses into stuck single-token cb0 runs (observed 72 then 103
    // repeats), drops the second sentence, and never emits EOS (runs to the frame cap). Sampling
    // must instead terminate on EOS well under the cap with no long stuck run.
    func testTTSGPUSampledGenerateBreaksGreedyLoops() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-sampled-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
            checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: FileManager.default.currentDirectoryPath + "/Resources/Shaders",
            outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let cfg = gpu.frontEndConfig!; let dim = 2048
        let (embeds, frames) = try Qwen3TTSFrontEnd.textToInputsEmbeds(
            text: "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
            instruct: "Speak slowly and clearly, like an audiobook narrator.",
            language: "English", tokenizer: gpu.tokenizer!, config: cfg,
            textRow: { gpu.weightRow("talker.model.text_embedding.weight", $0, dim) },
            codecRow: { gpu.weightRow("talker.model.codec_embedding.weight", $0, dim) },
            fc1W: gpu.f32("talker.text_projection.linear_fc1.weight"),
            fc1B: gpu.f32("talker.text_projection.linear_fc1.bias"),
            fc2W: gpu.f32("talker.text_projection.linear_fc2.weight"),
            fc2B: gpu.f32("talker.text_projection.linear_fc2.bias"), dim: dim)
        let cap = 256
        let gen = try gpu.generateCodes(inputsEmbeds: embeds, seqLen: frames, ttsPadId: cfg.ttsPad,
                                        maxFrames: cap, sampling: .init(seed: 1))
        XCTAssertLessThan(gen.count, cap, "sampling must terminate on EOS, not run to the frame cap")
        var maxRun = 0, run = 0, prev = -1
        for cb0 in gen.map({ $0[0] }) { run = (cb0 == prev) ? run + 1 : 1; prev = cb0; maxRun = max(maxRun, run) }
        XCTAssertLessThan(maxRun, 16, "no stuck single-token cb0 run (greedy collapsed to 72/103)")
    }

    // Phase-7-perf DIAGNOSTIC (opt-in, needs SMELT_DECODE_PROFILE=1 for the attribution): the
    // testTTSGPUPackagedTextToWavMatchesRealModel one-shot wall is COLD — first touch faults the whole
    // multi-GB fp32 package from SSD, and those major faults stall the GPU mid-kernel, inflating the
    // per-frame gpuCompute by ~2s on frame 1. Steady-state (page-cache warm) decode is ~30 ms/frame
    // (testTalkerForwardBackToBack). This runs generate() TWICE in one process: the first warms the
    // page cache, the SECOND is the true WARM end-to-end wall + attribution — the missing ground truth
    // that quantifies the per-frame cb0/MTP/lastHidden readback-sync wall the back-to-back test hides.
    // Asserts the standard wav==reference numeric gate on BOTH runs (catches a cold/warm-state
    // regression); the exact codes==gen_codes gate lives in testTTSGPUPackagedGenerateMatchesRealModel.
    func testTTSGPUWarmEndToEndProfile() throws {
        guard let model = env("SMELT_VOICE_MODEL"), let refs = env("SMELT_VOICE_REFS") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL + SMELT_VOICE_REFS")
        }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-warm-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        // Pack dtype is env-selectable so the same warm harness benchmarks the shipped dtypes
        // (bf16 / u4 — f32/f16 full-pipeline can't run generation post-Phase-4).
        let projDType: Qwen3TTSPackageBuilder.WeightDType =
            env("SMELT_TTS_PACK_DTYPE") == "u4" ? .u4 : .bf16
        FileHandle.standardError.write(Data("  [pack dtype: \(projDType)]\n".utf8))
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
            checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: FileManager.default.currentDirectoryPath + "/Resources/Shaders", outputPath: pkg,
            projDType: projDType)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let ref = readBinF32("\(refs)/stage_decoder_wav.bin")
        func run(_ tag: String) throws {
            FileHandle.standardError.write(Data("=== \(tag) end-to-end run ===\n".utf8))
            let wav = try gpu.generate(text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.",
                                       language: "English", maxFrames: 64, decode: .greedy)
            assertMatch(wav, ref, "\(tag) packaged text→wav == real model")
        }
        try run("COLD (1st)")
        try run("WARM (2nd)")
    }

    // Dtype-completeness GOLD gate: an f32-weight package generates a wav BYTE-IDENTICAL to the
    // bf16-weight package. The model is bf16-native, so the f32 build stores each weight as bits<<16
    // (exact) and the bf16 build stores the raw bf16; every f32-vs-bf16 kernel pair (gemv_f32 vs
    // gemv_bf16w_f32, etc.) widens identically AND lane-strides in the same reduction order, the codec
    // is f32 in BOTH builds, and the text_projection stays f32 in both — so the WHOLE pipeline is
    // byte-identical. This proves the f32/f16-dense trunk wiring binds correctly at runtime (a wrong
    // binding would diverge or crash). No refs needed (it's a self-consistency gate, f32 vs bf16).
    func testF32TrunkGeneratesByteIdenticalToBF16() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        let checkpointPolicy = try qwen3TTSTestCheckpointPolicy()
        func wav(_ dt: Qwen3TTSPackageBuilder.WeightDType) throws -> [Float] {
            let pkg = (NSTemporaryDirectory() as NSString)
                .appendingPathComponent("qwen3tts-\(dt)-\(UUID().uuidString).smeltpkg")
            defer { try? FileManager.default.removeItem(atPath: pkg) }
            try Qwen3TTSPackageBuilder.build(checkpointDir: model, checkpointPolicy: checkpointPolicy,
                                             shaderDir: shaderDir, outputPath: pkg, projDType: dt)
            let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
            XCTAssertEqual(gpu.weightDType("talker.model.layers.0.self_attn.q_proj.weight"), dt == .f32 ? .f32 : .bf16,
                           "\(dt) build q_proj dtype")
            return try gpu.generate(text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.",
                                    language: "English", maxFrames: 64, decode: .greedy)
        }
        let bf16wav = try wav(.bf16)
        let f32wav = try wav(.f32)
        XCTAssertEqual(f32wav.count, bf16wav.count, "f32 and bf16 wav lengths differ")
        XCTAssertTrue(!f32wav.isEmpty && f32wav.contains { $0 != 0 }, "f32 wav is empty/all-zero (gate vacuous)")
        let same = f32wav.withUnsafeBytes { a in bf16wav.withUnsafeBytes { b in a.elementsEqual(b) } }
        XCTAssertTrue(same, "f32-weight trunk wav != bf16-weight trunk wav — dtype-completeness byte-identity broken")
    }

    // Streaming-U0 BASELINE (opt-in, refs-free): warm wall + stage attribution from a PREBUILT
    // package (SMELT_VOICE_PKG). No wav==reference assert — the refs live in /tmp and don't
    // survive reboots; this harness only needs the wall decomposition (TTFA today == the full wall,
    // since decodeCodec runs after ALL frames). Run with SMELT_DECODE_PROFILE=1 for attribution.
    func testTTSGPUWarmProfileFromPackage() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        for tag in ["COLD (1st)", "WARM (2nd)", "WARM (3rd)"] {
            FileHandle.standardError.write(Data("=== \(tag) end-to-end run ===\n".utf8))
            let t0 = CFAbsoluteTimeGetCurrent()
            let wav = try gpu.generate(text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.",
                                       language: "English", maxFrames: 64, decode: .greedy)
            let wall = CFAbsoluteTimeGetCurrent() - t0
            let audio = Double(wav.count) / 24000
            FileHandle.standardError.write(Data(String(
                format: "  [%@] wall %.2fs, audio %.2fs, %.2fx realtime\n", tag, wall, audio, wall / audio).utf8))
            XCTAssertGreaterThan(wav.count, 0, "\(tag) produced audio")
        }
    }

    // Streaming-U1 GATE (opt-in, refs-free): chunked Qwen3TTSCodecStream decode == offline
    // decodeCodec BIT-EXACT (memcmp of the float buffers) across chunkings — all-at-once,
    // all-singles, the production doubling schedule, and a ragged mix with a partial final
    // chunk. Codes come from a greedy generate on the prebuilt package (deterministic), so
    // the fixture is the real production distribution, not synthetic noise.
    func testTTSCodecStreamMatchesOfflineDecode() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let ie = try gpu.inputsEmbeds(text: "Hello, this is a test.",
                                      instruct: "Speak in a calm, clear voice.", language: "English")
        let gen = try gpu.generateCodes(inputsEmbeds: ie.embeds, seqLen: ie.seqLen,
                                        ttsPadId: ie.ttsPadId, maxFrames: 64)
        let frames = gen.count
        XCTAssertGreaterThan(frames, 8, "fixture must span multiple chunks")
        var codes = [Int32](repeating: 0, count: 16 * frames)
        for f in 0..<frames { for c in 0..<16 { codes[c * frames + f] = Int32(gen[f][c]) } }
        let offline = try gpu.decodeCodec(codes: codes, frames: frames)
        XCTAssertEqual(offline.count, frames * 1920, "1920 samples/frame")

        func splitSizes(first: Int, cap: Int) -> [Int] {
            var sizes: [Int] = [], c = first, left = frames
            while left > 0 { let take = min(c, left); sizes.append(take); left -= take; c = min(c * 2, cap) }
            return sizes
        }
        var ragged: [Int] = []
        do {
            let pattern = [3, 1, 7, 2, 5]
            var left = frames, i = 0
            while left > 0 { let take = min(pattern[i % pattern.count], left); ragged.append(take); left -= take; i += 1 }
        }
        let chunkings: [[Int]] = [[frames], Array(repeating: 1, count: frames),
                                  splitSizes(first: 4, cap: 16), ragged]
        for chunking in chunkings {
            let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: 64)
            var streamed: [Float] = []
            var idx = 0
            for size in chunking {
                let out = try stream.decode(Array(gen[idx..<(idx + size)]))
                XCTAssertEqual(out.count, size * 1920, "chunk sample count, chunking \(chunking)")
                streamed += out
                idx += size
            }
            XCTAssertEqual(streamed.count, offline.count, "total length, chunking \(chunking)")
            let equal = streamed.withUnsafeBytes { a in offline.withUnsafeBytes { b in
                a.count == b.count && memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0
            } }
            if !equal {
                let firstBad = (0..<min(streamed.count, offline.count)).first { streamed[$0] != offline[$0] }
                XCTFail("streamed != offline for chunking \(chunking); first mismatch at sample "
                        + "\(firstBad.map(String.init) ?? "?") (frame \(firstBad.map { $0 / 1920 } ?? -1))")
            }
        }
    }

    // Streaming-U1b GATE (opt-in, refs-free): LONG-sequence bit-exactness. The short fixture
    // (26 frames) never exercises the preTransformer cache EVICTION path (history saturates at
    // 71 rows only past 72 frames) nor sliding-window clamping deep into the sequence. Codes
    // come from seeded sampling on a longer prompt (greedy degenerates on long prompts; seeded
    // sampling is deterministic and model-faithful).
    func testTTSCodecStreamLongSequenceMatchesOffline() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let text = "A gentle rain fell on the cobblestones, and the cafe windows glowed amber in the "
                 + "evening. The astronaut described the curve of the Earth, a thin blue line in the "
                 + "black of space, while the small fishing boat found its way back through the fog."
        let ie = try gpu.inputsEmbeds(text: text, instruct: "Speak in a calm, clear voice.", language: "English")
        let sampling = gpu.resolveDecode(.sampleSeeded(7))
        let gen = try gpu.generateCodes(inputsEmbeds: ie.embeds, seqLen: ie.seqLen,
                                        ttsPadId: ie.ttsPadId, maxFrames: 256, sampling: sampling)
        let frames = gen.count
        XCTAssertGreaterThan(frames, 90, "long fixture must saturate the 71-row preTransformer history")
        var codes = [Int32](repeating: 0, count: 16 * frames)
        for f in 0..<frames { for c in 0..<16 { codes[c * frames + f] = Int32(gen[f][c]) } }
        let offline = try gpu.decodeCodec(codes: codes, frames: frames)

        var doubling: [Int] = []
        var c = 4, left = frames
        while left > 0 { let take = min(c, left); doubling.append(take); left -= take; c = min(c * 2, 16) }
        for chunking in [Array(repeating: 1, count: frames), doubling] {
            let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: 256)
            var streamed: [Float] = []
            var idx = 0
            for size in chunking { streamed += try stream.decode(Array(gen[idx..<(idx + size)])); idx += size }
            XCTAssertEqual(streamed.count, offline.count)
            let equal = streamed.withUnsafeBytes { a in offline.withUnsafeBytes { b in
                a.count == b.count && memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0
            } }
            if !equal {
                let firstBad = (0..<min(streamed.count, offline.count)).first { streamed[$0] != offline[$0] }
                XCTFail("long-sequence streamed != offline; first mismatch at sample "
                        + "\(firstBad.map(String.init) ?? "?") (frame \(firstBad.map { $0 / 1920 } ?? -1) of \(frames))")
            }
        }
    }

    // Kernel-change bit-exactness harness (opt-in, refs-free): with SMELT_VOICE_WAV_BASELINE=<dir>,
    // the FIRST run writes the greedy + seeded end-to-end wavs as raw f32; subsequent runs memcmp
    // against them. Run before a kernel change to capture, after to verify the change is bit-exact
    // (same codes, same samples). Delete the dir to re-baseline after an intentional numeric change.
    func testTTSGPUWavMatchesBaseline() throws {
        guard let pkg = env("SMELT_VOICE_PKG"), let dir = env("SMELT_VOICE_WAV_BASELINE") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG + SMELT_VOICE_WAV_BASELINE")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let shortText = "Hello, this is a test.", instruct = "Speak in a calm, clear voice."
        let longText = "A gentle rain fell on the cobblestones, and the cafe windows glowed amber in the "
                     + "evening. The astronaut described the curve of the Earth, a thin blue line in the "
                     + "black of space, while the small fishing boat found its way back through the fog."
        let cases: [(String, String, Qwen3TTSSampler.DecodeMode, Int)] = [
            ("short-greedy", shortText, .greedy, 64),
            ("long-seeded7", longText, .sampleSeeded(7), 256)]
        for (name, text, mode, maxFrames) in cases {
            let wav = try gpu.generate(text: text, instruct: instruct, language: "English",
                                       maxFrames: maxFrames, decode: mode)
            let path = "\(dir)/\(name).f32"
            if FileManager.default.fileExists(atPath: path) {
                let ref = readBinF32(path)
                XCTAssertEqual(wav.count, ref.count, "\(name): sample count vs baseline")
                let equal = wav.count == ref.count && wav.withUnsafeBytes { a in
                    ref.withUnsafeBytes { b in memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0 } }
                if !equal {
                    let firstBad = (0..<min(wav.count, ref.count)).first { wav[$0] != ref[$0] }
                    XCTFail("\(name): wav != baseline; first mismatch at sample "
                            + "\(firstBad.map(String.init) ?? "?") (frame \(firstBad.map { $0 / 1920 } ?? -1))")
                }
            } else {
                try Data(bytes: wav, count: wav.count * 4).write(to: URL(fileURLWithPath: path))
                FileHandle.standardError.write(Data("  [wav-baseline] wrote \(path) (\(wav.count) samples)\n".utf8))
            }
        }
    }

    // TTFA micro-component PROFILE (opt-in, refs-free): isolates the pieces the breakdown
    // profile reports as aggregates — tokenizer wall, full front-end wall, and the
    // FIRST-chunk codec decode wall as a function of chunk size (fresh stream each, since
    // TTFA only ever sees a first chunk). Sizes 1/2/4 tell us whether a first=1 schedule
    // pays ~full fixed chunk cost or scales with frames.
    func testTTSGPUTTFAMicroProfile() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let text = "Hello, this is a test.", instruct = "Speak in a calm, clear voice."
        _ = try gpu.generate(text: text, instruct: instruct, language: "English",
                             maxFrames: 16, decode: .greedy)   // warmup (cold faults, pipeline JIT)
        var frames8: [[Int]] = []
        let ie0 = try gpu.inputsEmbeds(text: text, instruct: instruct, language: "English")
        _ = try gpu.generateCodes(inputsEmbeds: ie0.embeds, seqLen: ie0.seqLen, ttsPadId: ie0.ttsPadId,
                                  maxFrames: 16, onFrame: { i, codes in
            frames8.append(codes); return i < 7
        })
        for round in 1...3 {
            var lines = ["  [TTFA micro round \(round)]"]
            var t = CFAbsoluteTimeGetCurrent()
            let tok = gpu.tokenizer!
            _ = Qwen3TTSFrontEnd.wrap(text: text, instruct: instruct, tokenizer: tok)
            lines.append(String(format: "    tokenize(wrap)        %.4fs", CFAbsoluteTimeGetCurrent() - t))
            t = CFAbsoluteTimeGetCurrent()
            _ = try gpu.inputsEmbeds(text: text, instruct: instruct, language: "English")
            lines.append(String(format: "    inputsEmbeds (full)   %.4fs", CFAbsoluteTimeGetCurrent() - t))
            // Same work via the static row-accessor core with prefetched weights: isolates the
            // gathers + projections + assembly from the (memoized) wrapper lookups around them.
            let shape = try gpu.talkerShape()
            let proj = gpu.textProjWeights()
            t = CFAbsoluteTimeGetCurrent()
            _ = try Qwen3TTSFrontEnd.textToInputsEmbeds(
                text: text, instruct: instruct, language: "English",
                tokenizer: tok, config: gpu.frontEndConfig!,
                textRow: { gpu.weightRow("talker.model.text_embedding.weight", $0, shape.textEmbedDim) },
                codecRow: { gpu.weightRow("talker.model.codec_embedding.weight", $0, shape.hidden) },
                fc1W: proj.fc1W, fc1B: proj.fc1B, fc2W: proj.fc2W, fc2B: proj.fc2B,
                dim: shape.textEmbedDim, projInter: shape.projInter, hidden: shape.hidden)
            lines.append(String(format: "    textToInputsEmbeds    %.4fs", CFAbsoluteTimeGetCurrent() - t))
            for size in [1, 2, 4, 8] {
                let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: 16)
                t = CFAbsoluteTimeGetCurrent()
                let out = try stream.decode(Array(frames8[0..<size]))
                lines.append(String(format: "    codec first-chunk(%d)  %.4fs", size,
                                    CFAbsoluteTimeGetCurrent() - t))
                XCTAssertEqual(out.count, size * 1920)
            }
            FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
        }
    }

    // TTFA-window breakdown PROFILE (opt-in, refs-free): times EXACTLY the time-to-first-audio
    // critical path — front-end, codec-stream init, prefill+frame0, three more decode frames
    // (talker+MTP each), first 4-frame codec chunk — by cancelling generation at the first chunk
    // boundary. Run with SMELT_DECODE_PROFILE=1 for stage.category kernel attribution on top.
    func testTTSGPUTTFABreakdownProfile() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let text = "Hello, this is a test.", instruct = "Speak in a calm, clear voice."
        _ = try gpu.generate(text: text, instruct: instruct, language: "English",
                             maxFrames: 16, decode: .greedy)   // warmup (cold faults, pipeline JIT)
        for round in 1...3 {
            if SmeltDecodeProfile.enabled { SmeltDecodeProfile.reset(); SmeltDecodeProfile.setStage("front") }
            let t0 = CFAbsoluteTimeGetCurrent()
            let ie = try gpu.inputsEmbeds(text: text, instruct: instruct, language: "English")
            let tFront = CFAbsoluteTimeGetCurrent()
            let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: 64)
            let tInit = CFAbsoluteTimeGetCurrent()
            var ts: [Double] = []
            var first4: [[Int]] = []
            _ = try gpu.generateCodes(inputsEmbeds: ie.embeds, seqLen: ie.seqLen, ttsPadId: ie.ttsPadId,
                                      maxFrames: 64, onFrame: { i, codes in
                ts.append(CFAbsoluteTimeGetCurrent())
                first4.append(codes)
                return i < 3
            })
            let tGen = CFAbsoluteTimeGetCurrent()
            SmeltDecodeProfile.setStage("codecChunk")   // separate the chunk's kernels in the report
            let samples = try stream.decode(first4)
            let tEnd = CFAbsoluteTimeGetCurrent()
            XCTAssertEqual(samples.count, 4 * 1920)
            var lines = [String(format: "  [TTFA breakdown, round %d] total %.3fs (seqLen %d)",
                                round, tEnd - t0, ie.seqLen),
                         String(format: "    front(inputsEmbeds)    %.3fs", tFront - t0),
                         String(format: "    codecStream init       %.3fs", tInit - tFront),
                         String(format: "    setup+prefill+frame0   %.3fs", ts[0] - tInit)]
            for f in 1..<ts.count {
                lines.append(String(format: "    frame%d (decode+MTP)    %.3fs", f, ts[f] - ts[f - 1]))
            }
            lines.append(String(format: "    codec chunk (4 frames) %.3fs", tEnd - tGen))
            FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
            if SmeltDecodeProfile.enabled {
                FileHandle.standardError.write(Data((SmeltDecodeProfile.report(totalWallS: tEnd - t0) + "\n").utf8))
            }
        }
    }

    // Streaming-U3 PROFILE (opt-in, refs-free): TTFA + playback-deadline timeline. Playback
    // starts when chunk 0 lands (at TTFA); chunk k must land before the playhead exhausts the
    // audio already emitted: margin_k = (TTFA + sum of audio before k) - emitWall_k. Reports
    // TTFA and min margin per chunk schedule (warm), plus a long-utterance sustained run.
    func testTTSGPUStreamingTTFAProfile() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let shortText = "Hello, this is a test.", instruct = "Speak in a calm, clear voice."
        let longText = "A gentle rain fell on the cobblestones, and the cafe windows glowed amber in the "
                     + "evening. The astronaut described the curve of the Earth, a thin blue line in the "
                     + "black of space, while the small fishing boat found its way back through the fog."

        struct RunStats { let ttfa: Double; let minMargin: Double; let wall: Double; let audio: Double }
        func run(_ text: String, decode: Qwen3TTSSampler.DecodeMode, first: Int, cap: Int,
                 maxFrames: Int) throws -> RunStats {
            var emits: [(wall: Double, seconds: Double)] = []
            let t0 = CFAbsoluteTimeGetCurrent()
            try gpu.generateStreaming(text: text, instruct: instruct, language: "English",
                                      maxFrames: maxFrames, decode: decode,
                                      firstChunkFrames: first, maxChunkFrames: cap) { chunk in
                if !chunk.samples.isEmpty {
                    emits.append((CFAbsoluteTimeGetCurrent() - t0, Double(chunk.samples.count) / 24000))
                }
                return true
            }
            let wall = CFAbsoluteTimeGetCurrent() - t0
            let ttfa = emits[0].wall
            var playhead = ttfa, minMargin = Double.infinity
            for e in emits {
                minMargin = min(minMargin, playhead - e.wall)   // emitted at e.wall, needed by playhead
                playhead += e.seconds
            }
            return RunStats(ttfa: ttfa, minMargin: minMargin, wall: wall,
                            audio: emits.reduce(0) { $0 + $1.seconds })
        }

        // Effective TTFA = first-chunk wall + the initial jitter buffer a consumer must hold
        // before starting playback to never underrun (= max(0, -minMargin)). Generation is
        // <1x realtime in aggregate, so a bounded initial buffer always exists; the schedule
        // controls how much (smaller caps = smoother arrival = smaller buffer, but more
        // chunk-boundary overhead).
        func effTTFA(_ s: RunStats) -> Double { s.ttfa + max(0, -s.minMargin) }

        _ = try run(shortText, decode: .greedy, first: 4, cap: 4, maxFrames: 64)   // warmup (cold faults)
        var lines = ["  [streaming TTFA profile, warm]"]
        var defaultStats: RunStats? = nil
        for (first, cap) in [(1, 1), (1, 2), (1, 4), (2, 4), (2, 8), (4, 4), (4, 8), (4, 16), (8, 16)] {
            let s = try run(shortText, decode: .greedy, first: first, cap: cap, maxFrames: 64)
            if first == 4 && cap == 4 { defaultStats = s }   // the shipped default schedule
            lines.append(String(format: "    first=%d cap=%-2d  TTFA %.3fs  minMargin %+.3fs  effTTFA %.3fs  wall %.2fs  audio %.2fs",
                                first, cap, s.ttfa, s.minMargin, effTTFA(s), s.wall, s.audio))
        }
        let long = try run(longText, decode: .sampleSeeded(7), first: 4, cap: 4, maxFrames: 256)
        lines.append(String(format: "    LONG (sampled, 4/4)  TTFA %.3fs  minMargin %+.3fs  effTTFA %.3fs  wall %.2fs  audio %.2fs",
                            long.ttfa, long.minMargin, effTTFA(long), long.wall, long.audio))
        FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))

        let d = try XCTUnwrap(defaultStats)
        XCTAssertLessThan(effTTFA(d), 1.0, "warm effective TTFA (default 4/4 schedule) regressed past 1s")
        XCTAssertLessThan(effTTFA(long), 1.6, "long-utterance effective TTFA regressed past 1.6s")
    }

    // Streaming-U2 GATE (opt-in, refs-free): generateStreaming's concatenated chunks == offline
    // generate BIT-EXACT under BOTH deterministic decode modes (greedy + seeded sampling).
    // Anti-vacuity (codex): the fixture spans multiple chunks (totalFrames > firstChunkFrames)
    // and the first chunk must arrive while generation is still in flight — asserted both
    // structurally (first chunk == firstChunkFrames < total) and by wall time (first-chunk
    // latency well under the stream's total wall; a generate-everything-then-replay
    // implementation would land at ~1.0× of it).
    func testTTSGPUStreamingGenerateMatchesOffline() throws {
        guard let pkg = env("SMELT_VOICE_PKG") else {
            throw XCTSkip("opt-in: needs SMELT_VOICE_PKG (prebuilt .smeltpkg)")
        }
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let text = "Hello, this is a test.", instruct = "Speak in a calm, clear voice.", lang = "English"
        for (tag, mode) in [("greedy", Qwen3TTSSampler.DecodeMode.greedy),
                            ("sampleSeeded", .sampleSeeded(42))] {
            let offline = try gpu.generate(text: text, instruct: instruct, language: lang,
                                           maxFrames: 64, decode: mode)
            let totalFrames = offline.count / 1920
            XCTAssertGreaterThan(totalFrames, 4, "[\(tag)] fixture must outlast the first chunk")

            var streamed: [Float] = []
            var nonFinalFrameCounts: [Int] = []
            var firstChunkWall = 0.0, sawFinal = false
            let t0 = CFAbsoluteTimeGetCurrent()
            try gpu.generateStreaming(text: text, instruct: instruct, language: lang,
                                      maxFrames: 64, decode: mode,
                                      firstChunkFrames: 4, maxChunkFrames: 4) { chunk in
                XCTAssertFalse(sawFinal, "[\(tag)] no chunks after the final marker")
                XCTAssertEqual(chunk.frameOffset, streamed.count / 1920, "[\(tag)] contiguous chunks")
                XCTAssertEqual(chunk.samples.count, chunk.frameCount * 1920, "[\(tag)] 1920 samples/frame")
                streamed += chunk.samples
                if chunk.isFinal { sawFinal = true } else {
                    if nonFinalFrameCounts.isEmpty { firstChunkWall = CFAbsoluteTimeGetCurrent() - t0 }
                    nonFinalFrameCounts.append(chunk.frameCount)
                }
                return true
            }
            let streamWall = CFAbsoluteTimeGetCurrent() - t0

            XCTAssertTrue(sawFinal, "[\(tag)] stream must terminate with an isFinal chunk")
            XCTAssertEqual(streamed.count, offline.count, "[\(tag)] total samples")
            let equal = streamed.withUnsafeBytes { a in offline.withUnsafeBytes { b in
                a.count == b.count && memcmp(a.baseAddress!, b.baseAddress!, a.count) == 0
            } }
            if !equal {
                let firstBad = (0..<min(streamed.count, offline.count)).first { streamed[$0] != offline[$0] }
                XCTFail("[\(tag)] streamed != offline; first mismatch at sample \(firstBad.map(String.init) ?? "?")")
            }
            XCTAssertEqual(nonFinalFrameCounts.first, 4, "[\(tag)] first chunk == firstChunkFrames")
            XCTAssertGreaterThanOrEqual(nonFinalFrameCounts.count, 2, "[\(tag)] fixture spans multiple chunks")
            XCTAssertLessThan(firstChunkWall, 0.6 * streamWall,
                              "[\(tag)] first chunk must arrive while generation is in flight "
                              + "(TTFA \(firstChunkWall)s vs stream wall \(streamWall)s)")
            FileHandle.standardError.write(Data(String(
                format: "  [stream-%@] TTFA %.3fs, stream wall %.2fs, %d chunks %@\n",
                tag, firstChunkWall, streamWall, nonFinalFrameCounts.count,
                "\(nonFinalFrameCounts)").utf8))
        }
        // Cancellation smoke: stop after the first chunk — the stream returns early, no final
        // marker. No chunk params: this also pins the nil-default resolution to the package's
        // declared CAM loop schedule.
        var chunks = 0
        var defaultFirstCount = -1
        try gpu.generateStreaming(text: text, instruct: instruct, language: lang,
                                  maxFrames: 64, decode: .greedy) { chunk in
            chunks += 1
            defaultFirstCount = chunk.frameCount
            return false
        }
        XCTAssertEqual(chunks, 1, "cancelled stream emits exactly the one chunk")
        XCTAssertEqual(defaultFirstCount, try gpu.defaultChunkSchedule().first,
                       "nil chunk params resolve to the package's declared schedule")
    }

    // Dtype-completeness f16 RUNTIME gate (codex follow-up): the f32 gold gate proves the dense binding
    // via f32==bf16 byte-identity, but f16 uses .gemvF16WF32/.gemmF16WF32 — a DIFFERENT rounding of the
    // bf16-native weights, so NOT byte-identical to bf16. This proves the f16 dense route SHIPS + RUNS
    // end-to-end on the real checkpoint: BOTH trunk sidecars present, and a finite, non-empty,
    // EOS-terminating wav (a wrong f16 binding → NaN/garbage/crash). SUPERSEDES the Phase-4 f16 reject
    // (the hand talker was an unwired f32/f16 emit, not a real limitation).
    func testFP16TrunkShipsAndGeneratesFiniteWav() throws {
        guard let model = env("SMELT_VOICE_MODEL") else { throw XCTSkip("opt-in: needs SMELT_VOICE_MODEL") }
        let pkg = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("qwen3tts-fp16-run-\(UUID().uuidString).smeltpkg")
        defer { try? FileManager.default.removeItem(atPath: pkg) }
        let shaderDir = FileManager.default.currentDirectoryPath + "/Resources/Shaders"
        try Qwen3TTSPackageBuilder.build(checkpointDir: model,
                                         checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
                                         shaderDir: shaderDir, outputPath: pkg, projDType: .f16)
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: "\(pkg)/trunk") && fm.fileExists(atPath: "\(pkg)/trunk-mtp"),
                      "f16 full-pipeline must ship BOTH compiled trunk sidecars (trunk/ + trunk-mtp/)")
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        XCTAssertEqual(gpu.weightDType("talker.model.layers.0.self_attn.q_proj.weight"), .f16,
                       "f16 build must pack projections as fp16")
        let wav = try gpu.generate(text: "Hello, this is a test.", instruct: "Speak in a calm, clear voice.",
                                   language: "English", maxFrames: 64, decode: .greedy)
        XCTAssertFalse(wav.isEmpty, "f16 generated an empty wav")
        XCTAssertTrue(wav.allSatisfy { $0.isFinite }, "f16 wav has non-finite samples — f16 binding broken")
        XCTAssertTrue(wav.contains { $0 != 0 }, "f16 wav is all-zero — f16 binding broken")
    }
}
