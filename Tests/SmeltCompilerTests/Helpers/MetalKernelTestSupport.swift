// MetalKernelTestSupport — Shared scaffolding for Metal kernel tests.
//
// Pipeline construction, synthetic packed-quantized weight fixtures, the
// affine-matvec CPU reference, and small Metal-encode/buffer-build
// helpers used by region exactness tests and any kernel parity tests
// that share these primitives.

import Foundation
import Metal
import XCTest

/// Default trip-wire floor for "kernel wrote nothing" detection.
/// Matches the test class's `tripwireMinAbs`. Helpers below default
/// to this; regions with diluted output magnitudes (e.g. long-context
/// attention) pass a tighter threshold.
let regionTripwireMinAbs: Float = 1e-2

/// Per-(batch, head) trip-wire over a `[B, heads, headDim]` slice.
/// Catches dispatch-grid bugs that skip a (b, h) cell — unwritten
/// cells stay at the buffer's zero-initialized state, so the per-cell
/// max-abs check fires before the equality assertion does. Returns
/// the global max-abs across the slice.
func assertPerHeadTripwire(
    slice: [Float16],
    heads: Int, headDim: Int, batches: Int,
    label: String,
    minAbs: Float = regionTripwireMinAbs
) -> Float {
    var maxAbs: Float = 0
    for b in 0..<batches {
        for h in 0..<heads {
            let start = (b * heads + h) * headDim
            var cellMax: Float = 0
            for i in start..<start + headDim {
                let v = abs(Float(slice[i]))
                if v > cellMax { cellMax = v }
            }
            XCTAssertGreaterThan(
                cellMax, minAbs,
                "\(label) [b\(b), h\(h)] is suspiciously close to zero"
            )
            if cellMax > maxAbs { maxAbs = cellMax }
        }
    }
    return maxAbs
}

/// Per-touched-row trip-wire on a KV cache of shape
/// `[kvHeads, cacheSeqCapacity, headDim]`. Only the rows the kernel
/// writes (`absPos = startPos..startPos+batchSize`) are checked;
/// rows outside that window keep their initial fixture values, and
/// the equality assertion against `cpuExpected` covers those.
func assertPerCacheTripwire(
    cache: [Float16],
    kvHeads: Int, cacheSeqCapacity: Int, headDim: Int,
    startPos: Int, batchSize: Int,
    label: String,
    minAbs: Float = regionTripwireMinAbs
) -> Float {
    var maxAbs: Float = 0
    for h in 0..<kvHeads {
        for pos in 0..<batchSize {
            let absPos = startPos + pos
            let start = h * cacheSeqCapacity * headDim + absPos * headDim
            var cellMax: Float = 0
            for i in start..<start + headDim {
                let v = abs(Float(cache[i]))
                if v > cellMax { cellMax = v }
            }
            XCTAssertGreaterThan(
                cellMax, minAbs,
                "\(label) [h\(h), absPos\(absPos)] is suspiciously close to zero"
            )
            if cellMax > maxAbs { maxAbs = cellMax }
        }
    }
    return maxAbs
}

/// Coprime prime multipliers for synthetic test fixtures. Each region
/// uses these in role-specific ways — e.g., the primary multiplier
/// seeds the input vector in matvec regions, the Q vector in
/// attention_decode, and the qk vector in apply_rope. Centralizing the
/// primes keeps regions on the same coprime set; a new region picking
/// a near-collision prime would silently correlate its fixtures with
/// existing tests.
enum RegionSeed {
    static let primary: Int = 6151
    static let secondary: Int = 7919
    static let tertiary: Int = 4817
}

/// Process-lifetime cache of compiled compute pipelines keyed by source
/// file + entry point + function-constant values. The shader source
/// cache lives behind `loadMetalShaderSource`; this lifts the
/// `MTLLibrary` and `MTLComputePipelineState` build cost as well so
/// each test class stops re-compiling the same Metal kernels.
private struct PipelineKey: Hashable {
    let shaderFile: String
    let functionName: String
    let cols: UInt32?
    let groupSize: UInt32?
}

private nonisolated(unsafe) var pipelineCache: [PipelineKey: MTLComputePipelineState] = [:]
private let pipelineCacheLock = NSLock()

/// Compile and instantiate a compute pipeline from a shader file.
/// `cols` and `groupSize` are the FC_COLS / FC_GROUP_SIZE function
/// constants required by `affine_matvec`-family kernels. Pipelines are
/// cached on `(shaderFile, functionName, cols, groupSize)` so repeated
/// test methods compile each kernel once per process.
func makeComputePipeline(
    device: MTLDevice,
    shaderFile: String,
    functionName: String,
    cols: UInt32? = nil,
    groupSize: UInt32? = nil
) -> MTLComputePipelineState? {
    let key = PipelineKey(
        shaderFile: shaderFile,
        functionName: functionName,
        cols: cols,
        groupSize: groupSize
    )
    pipelineCacheLock.lock()
    defer { pipelineCacheLock.unlock() }
    if let cached = pipelineCache[key] {
        return cached
    }
    let options: MTLCompileOptions?
    if shaderFile.hasSuffix("_precise.metal")
        || shaderFile == "attention.metal"
        || shaderFile == "conv1d.metal"
        || shaderFile == "norms.metal"
        || shaderFile == "prefill_attention.metal"
        || shaderFile == "prefill_recurrence.metal"
        || shaderFile == "recurrence.metal"
        || shaderFile == "signed_quant.metal"
    {
        let precise = MTLCompileOptions()
        precise.fastMathEnabled = false
        options = precise
    } else {
        options = nil
    }
    guard let source = loadMetalShaderSource(shaderFile),
          let lib = try? device.makeLibrary(source: source, options: options)
    else { return nil }

    let fn: MTLFunction?
    if let cols, let groupSize {
        let constants = MTLFunctionConstantValues()
        var colsVal = cols
        var gsVal = groupSize
        constants.setConstantValue(&colsVal, type: .uint, index: 0)
        constants.setConstantValue(&gsVal, type: .uint, index: 1)
        fn = try? lib.makeFunction(name: functionName, constantValues: constants)
    } else {
        fn = lib.makeFunction(name: functionName)
    }

    guard let fn,
          let pipeline = try? device.makeComputePipelineState(function: fn)
    else { return nil }
    pipelineCache[key] = pipeline
    return pipeline
}

/// Wrap a Swift array in a shared-storage MTLBuffer using the type's
/// natural stride. Replaces hand-written `array.count * 2` (Float16) /
/// `array.count` (UInt8) length calculations and the off-by-stride
/// class of bugs that comes with them.
func makeSharedBuffer<T>(device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length = array.count * MemoryLayout<T>.stride
    // Scope the raw pointer to the array's storage so it can't outlive the
    // backing buffer; makeBuffer copies the bytes before withUnsafeBytes returns.
    return try array.withUnsafeBytes { rawBuf in
        try XCTUnwrap(
            device.makeBuffer(bytes: rawBuf.baseAddress!, length: length, options: .storageModeShared)
        )
    }
}

/// Allocate a zero-initialized shared-storage MTLBuffer for `count`
/// elements of type `T`. Metal initializes shared/managed storage to
/// zero by default, so no explicit memset is needed at the call site.
func makeSharedBuffer<T>(device: MTLDevice, count: Int, of: T.Type) throws -> MTLBuffer {
    let length = count * MemoryLayout<T>.stride
    return try XCTUnwrap(
        device.makeBuffer(length: length, options: .storageModeShared)
    )
}

/// Encode one `rms_norm_1pw` dispatch on the supplied compute encoder.
/// The kernel uses one threadgroup × 256 threads — one TG covers the
/// entire hidden dimension. Buffer indices match the kernel signature
/// at `Resources/Shaders/norms.metal:9-18`.
func encodeRmsNorm1pw(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    input: MTLBuffer,
    normWeight: MTLBuffer,
    output: MTLBuffer,
    cols: Int,
    eps: Float
) {
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(normWeight, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    var dimVal = UInt32(cols)
    encoder.setBytes(&dimVal, length: 4, index: 3)
    var epsVal = eps
    encoder.setBytes(&epsVal, length: 4, index: 4)
    encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
}

/// CPU mirror of `rms_norm_1pw`: `output[i] = x * rsqrt(mean+eps) * (1+w)`.
/// Returns the FP16-rounded normalized vector that downstream matvecs
/// would read from the GPU norm-output buffer. The norm weight is a
/// delta from 1.0, not a scale factor — every region's CPU reference
/// has to start with this exact formula.
func cpuRmsNorm1pw(
    input: [Float16],
    normWeight: [Float16],
    cols: Int,
    eps: Float
) -> [Float] {
    var sumSq: Float = 0
    for v in input { sumSq += Float(v) * Float(v) }
    let invRms = 1 / (sumSq / Float(cols) + eps).squareRoot()

    var normalized = [Float](repeating: 0, count: cols)
    for i in 0..<cols {
        let scaled = Float(input[i]) * invRms * (1 + Float(normWeight[i]))
        normalized[i] = Float(Float16(scaled))
    }
    return normalized
}

/// Run a closure on a single command-buffer + compute-encoder pair,
/// then commit, wait, and rethrow any GPU error. Replaces the seven
/// lines of identical boilerplate (`makeCommandBuffer`, `endEncoding`,
/// `commit`, `waitUntilCompleted`, error check) at every region's
/// GPU executor.
func runOnGPU(
    queue: MTLCommandQueue,
    _ encode: (MTLComputeCommandEncoder) throws -> Void
) throws {
    let cmdBuf = try XCTUnwrap(queue.makeCommandBuffer())
    let enc = try XCTUnwrap(cmdBuf.makeComputeCommandEncoder())
    try encode(enc)
    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()
    if let err = cmdBuf.error {
        throw err
    }
}

/// Single-line region-exactness log. Used by every region test to
/// surface max diff and GPU output magnitude for triage.
func logRegionExactness(
    name: String,
    shape: String,
    maxDiff: Float,
    gpuMaxAbs: Float
) {
    fputs(
        "  \(name) \(shape):"
            + " max diff = \(String(format: "%.6f", maxDiff)),"
            + " gpu max abs = \(String(format: "%.3f", gpuMaxAbs))\n",
        stderr
    )
}

/// Standard tail of every region exactness test: trip-wire on max-abs,
/// element-wise max-diff vs reference, log, and assert tolerance.
/// Returns the observed max diff so callers can do extra assertions
/// (e.g., index equality on argmax).
@discardableResult
func assertRegionExactness(
    name: String,
    shape: String,
    gpu: [Float16],
    cpu: [Float],
    tolerance: Float,
    minAbs: Float = regionTripwireMinAbs,
    file: StaticString = #filePath,
    line: UInt = #line
) -> Float {
    let gpuMaxAbs = gpu.map { abs(Float($0)) }.max() ?? 0
    XCTAssertGreaterThan(
        gpuMaxAbs, minAbs,
        "\(name): GPU output is suspiciously close to zero \(shape)",
        file: file, line: line
    )
    XCTAssertEqual(gpu.count, cpu.count, file: file, line: line)
    var maxDiff: Float = 0
    for i in 0..<gpu.count {
        let diff = abs(Float(gpu[i]) - cpu[i])
        if diff > maxDiff { maxDiff = diff }
    }
    logRegionExactness(name: name, shape: shape, maxDiff: maxDiff, gpuMaxAbs: gpuMaxAbs)
    XCTAssertLessThan(
        maxDiff, tolerance,
        "\(name): GPU diverged from CPU reference \(shape)",
        file: file, line: line
    )
    return maxDiff
}

/// Synthetic Float16 vector with deterministic content. Used for inputs,
/// norm weights, residuals — anything that needs a smooth distribution
/// without external fixtures.
func makeFloat16Vector(
    count: Int,
    seedOffset: Float,
    multiplier: Float,
    bias: Float = 0,
    phase: Float = 0.0137
) -> [Float16] {
    var out = [Float16](repeating: 0, count: count)
    for i in 0..<count {
        out[i] = Float16(bias + multiplier * sin((Float(i) + seedOffset) * phase))
    }
    return out
}

/// Build the (input, normWeight) pair every region whose first stage is
/// `rms_norm_1pw` shares. `inputBias` controls the input distribution's
/// mean — region tests using nonlinear downstream activations (e.g.,
/// silu) need a positive bias to keep activations away from FP16
/// underflow; linear matvec tests can use mean-zero inputs.
func makeRmsNormInputs(
    cols: Int,
    seed: Int,
    inputBias: Float = 0
) -> (input: [Float16], normWeight: [Float16]) {
    let inputSeed = Float(seed &* RegionSeed.primary)
    let normSeed = Float(seed &* RegionSeed.tertiary)

    let input = makeFloat16Vector(
        count: cols, seedOffset: inputSeed, multiplier: 0.5, bias: inputBias
    )
    let normWeight = makeFloat16Vector(
        count: cols, seedOffset: normSeed, multiplier: 0.5, bias: 0.5, phase: 0.0091
    )
    return (input, normWeight)
}

/// Synthetic packed-u4 quantized weight fixture. Returns the packed
/// `[R, C/2]` weight bytes plus per-group `[R, C/groupSize]` scales and
/// biases. Identical statistical distribution to the existing
/// `PrefillKernelTests` `makeAffineTestData` so tests can be ported one
/// by one without changing observable values.
func makeQuantizedAffineFixture(
    rows: Int,
    cols: Int,
    groupSize: Int,
    seed: Int
) -> (weights: [UInt8], scales: [Float16], biases: [Float16]) {
    let weightBytes = rows * cols / 2
    let sbCount = rows * (cols / groupSize)

    let weightSeed = seed &* 131
    let scaleSeed = Float(seed &* RegionSeed.secondary)
    let biasSeed = Float(seed &* 6869)

    var weights = [UInt8](repeating: 0, count: weightBytes)
    var scales = [Float16](repeating: 0, count: sbCount)
    var biases = [Float16](repeating: 0, count: sbCount)

    for i in 0..<weightBytes {
        weights[i] = UInt8(truncatingIfNeeded: i &* 17 &+ 23 &+ weightSeed)
    }
    for i in 0..<sbCount {
        scales[i] = Float16(
            0.01 + 0.04 * (0.5 + 0.5 * sin((Float(i) + scaleSeed) * 0.0071))
        )
        biases[i] = Float16(cos((Float(i) + biasSeed) * 0.0113) * 0.03)
    }
    return (weights, scales, biases)
}

/// Single-row affine_matvec CPU reference, FP32 accumulation. Mirrors
/// the per-row math the GPU `affine_matvec` kernel performs (group-wise
/// `scale * sum(nibble * x) + bias * sum(x)`). When `roundToFP16` is
/// true (the default), the result matches a standalone matvec kernel
/// that writes `half(acc)`. Pass false for fused chains where the
/// matvec output is consumed in FP32 by a downstream activation.
func referenceAffineMatvecRow(
    weights: [UInt8],
    scales: [Float16],
    biases: [Float16],
    input: [Float],
    row: Int,
    cols: Int,
    groupSize: Int,
    roundToFP16: Bool = true
) -> Float {
    let groupsPerRow = cols / groupSize
    let rowWeightBase = row * (cols / 2)
    let rowSBBase = row * groupsPerRow
    var acc: Float = 0

    for g in 0..<groupsPerRow {
        let scale = Float(scales[rowSBBase + g])
        let bias = Float(biases[rowSBBase + g])
        let colBase = g * groupSize
        var dot: Float = 0
        var xsum: Float = 0
        for i in 0..<groupSize {
            let col = colBase + i
            let byte = weights[rowWeightBase + (col / 2)]
            let nibble = (col & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
            let x = input[col]
            dot += Float(nibble) * x
            xsum += x
        }
        acc += scale * dot + bias * xsum
    }
    return roundToFP16 ? Float(Float16(acc)) : acc
}
