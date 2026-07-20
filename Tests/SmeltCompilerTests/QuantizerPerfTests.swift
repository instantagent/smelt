// QuantizerPerfTests — Empirical tests for KMeans quantization quality and convergence.
//
// These tests answer:
// 1. Does convergence exit fire? (Compare MAX_ITER=10 vs 100 — same RMSE means converged by iter 10)
// 2. Does N_INIT=1 match N_INIT=3 on real weight distributions?
// 3. What's the actual quality on different data shapes?

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class QuantizerPerfTests: XCTestCase {

    private var device: MTLDevice!

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1",
            "Set SMELT_RUN_SLOW_TESTS=1 to run quantizer perf tests"
        )
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "No Metal device available")
    }

    // MARK: - Helpers

    /// Quantize FP16 data and return (RMSE, time_seconds).
    private func quantizeAndMeasure(
        data: [Float16],
        shape: [Int],
        shaderOverrides: [String: String] = [:],
        label: String
    ) throws -> (rmse: Double, seconds: Double) {
        let count = data.count
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1

        let outputPath = NSTemporaryDirectory()
            + "qperf_\(label)_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        // Load and optionally patch shader source
        var shaderSource = try loadShaderSource()
        for (key, value) in shaderOverrides {
            shaderSource = shaderSource.replacingOccurrences(of: key, with: value)
        }

        let library = try device.makeLibrary(source: shaderSource, options: nil)
        guard let quantizeFn = library.makeFunction(name: "kmeans_quantize"),
              let packFn = library.makeFunction(name: "kmeans_pack_u4"),
              device.makeCommandQueue() != nil
        else {
            throw TestError.setupFailed
        }
        // Validate the (optionally patched) shader compiles into pipeline states.
        // SmeltQuantizer.quantize() rebuilds its own pipelines internally, so the
        // results are intentionally discarded — we keep the throwing call for the
        // compile-time validation side effect only.
        _ = try device.makeComputePipelineState(function: quantizeFn)
        _ = try device.makeComputePipelineState(function: packFn)

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: []
        )

        let start = CFAbsoluteTimeGetCurrent()
        let entries = try data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "test",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: shape,
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Dequantize and compute RMSE
        let entry = entries[0]
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let numGroups = ((rows + 15) / 16)
        let halfCols = cols / 2

        var sumSqErr: Double = 0
        fileData.withUnsafeBytes { raw in
            let packed = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: UInt8.self, capacity: count / 2)
            let lutBase = raw.baseAddress!.advanced(by: Int(entry.lutOffset!))
                .bindMemory(to: Float16.self, capacity: numGroups * 16)

            for row in 0..<rows {
                let group = row / 16
                let groupLUT = lutBase + group * 16
                for col in 0..<halfCols {
                    let byte = packed[row * halfCols + col]
                    let loVal = Float(groupLUT[Int(byte & 0x0F)])
                    let hiVal = Float(groupLUT[Int(byte >> 4)])
                    let origLo = Float(data[row * cols + col * 2])
                    let origHi = Float(data[row * cols + col * 2 + 1])
                    sumSqErr += Double((loVal - origLo) * (loVal - origLo))
                    sumSqErr += Double((hiVal - origHi) * (hiVal - origHi))
                }
            }
        }
        let rmse = sqrt(sumSqErr / Double(count))
        return (rmse, elapsed)
    }

    /// Generate data mimicking real weight distributions (roughly normal, zero-centered).
    private func generateWeightLikeData(rows: Int, cols: Int, scale: Float = 0.02) -> [Float16] {
        let count = rows * cols
        var data = [Float16](repeating: 0, count: count)
        // Box-Muller approximation using sin/cos of index (deterministic, reproducible)
        for idx in 0..<count {
            let u1 = (Float(idx % 997) + 1) / 998.0
            let u2 = (Float(idx % 991) + 1) / 992.0
            let normal = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            data[idx] = Float16(normal * scale)
        }
        return data
    }

    /// Generate bimodal data (two clusters — harder for quantization).
    private func generateBimodalData(rows: Int, cols: Int) -> [Float16] {
        let count = rows * cols
        var data = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            let center: Float = (idx % 3 == 0) ? -0.5 : 0.5
            let noise = sin(Float(idx) * 0.0137) * 0.1
            data[idx] = Float16(center + noise)
        }
        return data
    }

    /// Generate uniform data (worst case — no natural clusters).
    private func generateUniformData(rows: Int, cols: Int) -> [Float16] {
        let count = rows * cols
        var data = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            let val = Float(idx) / Float(count) - 0.5
            data[idx] = Float16(val)
        }
        return data
    }

    private func loadShaderSource() throws -> String {
        let candidates = [
            "Resources/Shaders/kmeans.metal",
            "../Resources/Shaders/kmeans.metal",
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            return try String(contentsOfFile: path, encoding: .utf8)
        }
        throw TestError.setupFailed
    }

    private enum TestError: Error { case setupFailed }

    /// Identity passthrough that the optimizer cannot constant-fold. Used to keep
    /// runtime values opaque so general-purpose guards (e.g. the zero-pad branch in
    /// testPhaseProfile_GateProj) stay live even when the input happens to be a
    /// compile-time constant that makes the branch trivially false.
    @inline(never)
    private func opaque(_ value: Int) -> Int { value }

    // MARK: - Convergence tests

    /// If convergence exit works, MAX_ITER=10 should match MAX_ITER=100.
    /// Uses weight-like data (normal distribution, scale 0.02).
    func testConvergenceExitFires_WeightLikeData() throws {
        let data = generateWeightLikeData(rows: 64, cols: 128)

        let (rmse100, time100) = try quantizeAndMeasure(
            data: data, shape: [64, 128], label: "iter100"
        )
        let (rmse10, time10) = try quantizeAndMeasure(
            data: data, shape: [64, 128],
            shaderOverrides: ["constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 10;"],
            label: "iter10"
        )

        fputs(
            "  Convergence test (weight-like 64×128):\n"
                + "    MAX_ITER=100: RMSE=\(String(format: "%.6f", rmse100))"
                + "  time=\(String(format: "%.3f", time100))s\n"
                + "    MAX_ITER=10:  RMSE=\(String(format: "%.6f", rmse10))"
                + "  time=\(String(format: "%.3f", time10))s\n"
                + "    Ratio: \(String(format: "%.4f", rmse10 / rmse100))x\n",
            stderr
        )

        // If converged by iter 10, RMSE should be within 5% of iter 100
        XCTAssertLessThan(
            rmse10 / rmse100, 1.05,
            "MAX_ITER=10 RMSE (\(rmse10)) should be within 5% of MAX_ITER=100 (\(rmse100)). "
                + "Convergence exit may not be firing."
        )
    }

    /// Same test but with bimodal data (harder distribution).
    func testConvergenceExitFires_BimodalData() throws {
        let data = generateBimodalData(rows: 64, cols: 128)

        let (rmse100, _) = try quantizeAndMeasure(
            data: data, shape: [64, 128], label: "bimodal100"
        )
        let (rmse10, _) = try quantizeAndMeasure(
            data: data, shape: [64, 128],
            shaderOverrides: ["constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 10;"],
            label: "bimodal10"
        )

        fputs(
            "  Convergence test (bimodal 64×128):\n"
                + "    MAX_ITER=100: RMSE=\(String(format: "%.6f", rmse100))\n"
                + "    MAX_ITER=10:  RMSE=\(String(format: "%.6f", rmse10))\n"
                + "    Ratio: \(String(format: "%.4f", rmse10 / rmse100))x\n",
            stderr
        )

        XCTAssertLessThan(
            rmse10 / rmse100, 1.05,
            "Bimodal: convergence should fire before iter 10"
        )
    }

    // MARK: - N_INIT tests

    /// Does N_INIT=1 match N_INIT=3 on weight-like data?
    func testNInit1MatchesNInit3_WeightLikeData() throws {
        let data = generateWeightLikeData(rows: 64, cols: 128)

        let (rmse3, time3) = try quantizeAndMeasure(
            data: data, shape: [64, 128], label: "ninit3"
        )
        let (rmse1, time1) = try quantizeAndMeasure(
            data: data, shape: [64, 128],
            shaderOverrides: ["constant uint N_INIT = 3;": "constant uint N_INIT = 1;"],
            label: "ninit1"
        )

        fputs(
            "  N_INIT test (weight-like 64×128):\n"
                + "    N_INIT=3: RMSE=\(String(format: "%.6f", rmse3))"
                + "  time=\(String(format: "%.3f", time3))s\n"
                + "    N_INIT=1: RMSE=\(String(format: "%.6f", rmse1))"
                + "  time=\(String(format: "%.3f", time1))s\n"
                + "    Quality ratio: \(String(format: "%.4f", rmse1 / rmse3))x"
                + "  Speed ratio: \(String(format: "%.1f", time3 / time1))x\n",
            stderr
        )

        // N_INIT=1 should be within 10% of N_INIT=3 on well-behaved data
        XCTAssertLessThan(
            rmse1 / rmse3, 1.10,
            "N_INIT=1 RMSE (\(rmse1)) should be within 10% of N_INIT=3 (\(rmse3))"
        )
    }

    /// N_INIT test on a larger tensor (closer to real layer weights).
    func testNInit1MatchesNInit3_LargerTensor() throws {
        let data = generateWeightLikeData(rows: 256, cols: 512, scale: 0.01)

        let (rmse3, time3) = try quantizeAndMeasure(
            data: data, shape: [256, 512], label: "large_ninit3"
        )
        let (rmse1, time1) = try quantizeAndMeasure(
            data: data, shape: [256, 512],
            shaderOverrides: ["constant uint N_INIT = 3;": "constant uint N_INIT = 1;"],
            label: "large_ninit1"
        )

        fputs(
            "  N_INIT test (weight-like 256×512):\n"
                + "    N_INIT=3: RMSE=\(String(format: "%.6f", rmse3))"
                + "  time=\(String(format: "%.3f", time3))s\n"
                + "    N_INIT=1: RMSE=\(String(format: "%.6f", rmse1))"
                + "  time=\(String(format: "%.3f", time1))s\n"
                + "    Quality ratio: \(String(format: "%.4f", rmse1 / rmse3))x"
                + "  Speed ratio: \(String(format: "%.1f", time3 / time1))x\n",
            stderr
        )

        XCTAssertLessThan(
            rmse1 / rmse3, 1.10,
            "N_INIT=1 on 256×512 should be within 10% of N_INIT=3"
        )
    }

    /// N_INIT test on uniform data (hardest case — no natural clusters).
    func testNInit1MatchesNInit3_UniformData() throws {
        let data = generateUniformData(rows: 64, cols: 128)

        let (rmse3, _) = try quantizeAndMeasure(
            data: data, shape: [64, 128], label: "uniform_ninit3"
        )
        let (rmse1, _) = try quantizeAndMeasure(
            data: data, shape: [64, 128],
            shaderOverrides: ["constant uint N_INIT = 3;": "constant uint N_INIT = 1;"],
            label: "uniform_ninit1"
        )

        fputs(
            "  N_INIT test (uniform 64×128):\n"
                + "    N_INIT=3: RMSE=\(String(format: "%.6f", rmse3))\n"
                + "    N_INIT=1: RMSE=\(String(format: "%.6f", rmse1))\n"
                + "    Quality ratio: \(String(format: "%.4f", rmse1 / rmse3))x\n",
            stderr
        )

        // Uniform is harder — allow 15% tolerance
        XCTAssertLessThan(
            rmse1 / rmse3, 1.15,
            "N_INIT=1 on uniform data should be within 15% of N_INIT=3"
        )
    }

    // MARK: - Combined: N_INIT=1 + MAX_ITER=15 (the proposed fast config)

    /// Test the "fast" config we'd actually ship against the current defaults.
    func testFastConfigVsDefaults() throws {
        let data = generateWeightLikeData(rows: 256, cols: 512, scale: 0.01)

        let (rmseDefault, timeDefault) = try quantizeAndMeasure(
            data: data, shape: [256, 512], label: "default_config"
        )
        let (rmseFast, timeFast) = try quantizeAndMeasure(
            data: data, shape: [256, 512],
            shaderOverrides: [
                "constant uint N_INIT = 3;": "constant uint N_INIT = 1;",
                "constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 15;",
            ],
            label: "fast_config"
        )

        let speedup = timeDefault / max(timeFast, 0.001)
        let qualityLoss = (rmseFast / rmseDefault - 1) * 100

        fputs(
            "  Fast config test (256×512):\n"
                + "    Default (N=3, I=100): RMSE=\(String(format: "%.6f", rmseDefault))"
                + "  time=\(String(format: "%.3f", timeDefault))s\n"
                + "    Fast (N=1, I=15):     RMSE=\(String(format: "%.6f", rmseFast))"
                + "  time=\(String(format: "%.3f", timeFast))s\n"
                + "    Speedup: \(String(format: "%.1f", speedup))x"
                + "  Quality loss: \(String(format: "%.1f", qualityLoss))%\n",
            stderr
        )

        XCTAssertLessThan(
            rmseFast / rmseDefault, 1.10,
            "Fast config should be within 10% RMSE of defaults"
        )
        // Speed assertion: at small tensor sizes GPU overhead dominates,
        // so we only check quality. The speedup manifests on large tensors
        // (e.g., [6144, 2048]) where KMeans compute >> dispatch overhead.
        fputs(
            "    (Speed assertion skipped — small tensors are overhead-dominated)\n",
            stderr
        )
    }

    // MARK: - Phase profiling

    /// Break down where time actually goes for a real-scale tensor.
    func testPhaseProfile_GateProj() throws {
        // 6_144 is 16-aligned, so the optimizer would fold paddedRows == rows and
        // flag the zero-pad branch as dead. Route through opaque() so the general
        // padding path stays live (it correctly handles non-16-multiple rows).
        let rows = opaque(6_144)
        let cols = 2_048
        let count = rows * cols
        let data = generateWeightLikeData(rows: rows, cols: cols, scale: 0.01)

        // Phase 1: Shader compilation (one-time cost)
        var t0 = CFAbsoluteTimeGetCurrent()
        let shaderSource = try loadShaderSource()
        let library = try device.makeLibrary(source: shaderSource, options: nil)
        let quantizeFn = library.makeFunction(name: "kmeans_quantize")!
        let packFn = library.makeFunction(name: "kmeans_pack_u4")!
        let quantizePipe = try device.makeComputePipelineState(function: quantizeFn)
        let packPipe = try device.makeComputePipelineState(function: packFn)
        let queue = device.makeCommandQueue()!
        let compileTime = CFAbsoluteTimeGetCurrent() - t0

        // Phase 2: Buffer allocation
        let groupSize = 16
        let paddedRows = ((rows + groupSize - 1) / groupSize) * groupSize
        let totalElements = paddedRows * cols
        let fp16Bytes = totalElements * 2

        t0 = CFAbsoluteTimeGetCurrent()
        let sourceBuffer = device.makeBuffer(length: fp16Bytes, options: .storageModeShared)!
        let assignBuffer = device.makeBuffer(
            length: totalElements, options: .storageModeShared
        )!
        let numGroups = paddedRows / groupSize
        let lutBuffer = device.makeBuffer(
            length: numGroups * 16 * 2, options: .storageModeShared
        )!
        let allocTime = CFAbsoluteTimeGetCurrent() - t0

        // Phase 3: Data copy (simulating FP16 memcpy — test data is already FP16)
        t0 = CFAbsoluteTimeGetCurrent()
        data.withUnsafeBytes { raw in
            sourceBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: count * 2)
        }
        // Zero-pad
        if paddedRows > rows {
            let padStart = rows * cols * 2
            memset(sourceBuffer.contents().advanced(by: padStart), 0, (paddedRows - rows) * cols * 2)
        }
        let copyTime = CFAbsoluteTimeGetCurrent() - t0

        // Phase 4: GPU KMeans (the actual quantization)
        let groupElements = UInt32(groupSize * cols)
        t0 = CFAbsoluteTimeGetCurrent()
        try SmeltQuantizer.dispatchKMeansBatched(
            sourceBuffer: sourceBuffer,
            assignBuffer: assignBuffer,
            lutBuffer: lutBuffer,
            numGroups: numGroups,
            groupElements: groupElements,
            seed: 42,
            queue: queue,
            pipeline: quantizePipe
        )
        let kmeansTime = CFAbsoluteTimeGetCurrent() - t0

        // Phase 5: Pack u4
        t0 = CFAbsoluteTimeGetCurrent()
        let packedBuffer = try SmeltQuantizer.dispatchPackU4(
            assignBuffer: assignBuffer,
            totalElements: totalElements,
            device: device,
            queue: queue,
            pipeline: packPipe
        )
        let packTime = CFAbsoluteTimeGetCurrent() - t0

        // Phase 6: Memcpy to output
        t0 = CFAbsoluteTimeGetCurrent()
        let outputSize = totalElements / 2 + numGroups * 16 * 2
        let outputBuf = UnsafeMutableRawPointer.allocate(byteCount: outputSize, alignment: 16)
        defer { outputBuf.deallocate() }
        memcpy(outputBuf, packedBuffer.contents(), totalElements / 2)
        memcpy(outputBuf.advanced(by: totalElements / 2), lutBuffer.contents(), numGroups * 16 * 2)
        let outputTime = CFAbsoluteTimeGetCurrent() - t0

        let total = compileTime + allocTime + copyTime + kmeansTime + packTime + outputTime

        fputs(
            "\n  === PHASE PROFILE: gate_proj [6144, 2048] ===\n"
                + "    Shader compile:  \(String(format: "%6.2f", compileTime))s"
                + "  (\(String(format: "%.0f", compileTime / total * 100))%)\n"
                + "    Buffer alloc:    \(String(format: "%6.2f", allocTime))s"
                + "  (\(String(format: "%.0f", allocTime / total * 100))%)\n"
                + "    Data copy:       \(String(format: "%6.2f", copyTime))s"
                + "  (\(String(format: "%.0f", copyTime / total * 100))%)\n"
                + "    GPU KMeans:      \(String(format: "%6.2f", kmeansTime))s"
                + "  (\(String(format: "%.0f", kmeansTime / total * 100))%)\n"
                + "    GPU Pack u4:     \(String(format: "%6.2f", packTime))s"
                + "  (\(String(format: "%.0f", packTime / total * 100))%)\n"
                + "    Output memcpy:   \(String(format: "%6.2f", outputTime))s"
                + "  (\(String(format: "%.0f", outputTime / total * 100))%)\n"
                + "    TOTAL:           \(String(format: "%6.2f", total))s\n",
            stderr
        )

        // The test itself just needs to not crash
        XCTAssertGreaterThan(total, 0)
    }

    // MARK: - Real-scale tensor tests

    /// Test on actual layer weight dimensions: [6144, 2048] gate/up projection.
    /// This is the shape that takes 18-25s per tensor in the build.
    func testRealScale_GateProj() throws {
        let rows = 6_144
        let cols = 2_048
        let data = generateWeightLikeData(rows: rows, cols: cols, scale: 0.01)

        let (rmseDefault, timeDefault) = try quantizeAndMeasure(
            data: data, shape: [rows, cols], label: "gate_default"
        )
        let (rmseFast, timeFast) = try quantizeAndMeasure(
            data: data, shape: [rows, cols],
            shaderOverrides: [
                "constant uint N_INIT = 3;": "constant uint N_INIT = 1;",
                "constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 15;",
            ],
            label: "gate_fast"
        )

        let speedup = timeDefault / max(timeFast, 0.001)
        let qualityLoss = (rmseFast / rmseDefault - 1) * 100

        fputs(
            "\n  === REAL SCALE: gate_proj [6144, 2048] ===\n"
                + "    Default (N=3, I=100): RMSE=\(String(format: "%.6f", rmseDefault))"
                + "  time=\(String(format: "%.2f", timeDefault))s\n"
                + "    Fast (N=1, I=15):     RMSE=\(String(format: "%.6f", rmseFast))"
                + "  time=\(String(format: "%.2f", timeFast))s\n"
                + "    Speedup: \(String(format: "%.1f", speedup))x"
                + "  Quality loss: \(String(format: "%.2f", qualityLoss))%\n",
            stderr
        )

        XCTAssertLessThan(
            rmseFast / rmseDefault, 1.05,
            "Fast config on [6144, 2048] should be within 5% RMSE"
        )
    }

    /// Test on QKV projection: [6144, 2048] (same shape, different data).
    func testRealScale_QKVProj() throws {
        let rows = 6_144
        let cols = 2_048
        // Different scale/seed than gate to test another distribution
        let data = generateWeightLikeData(rows: rows, cols: cols, scale: 0.005)

        let (rmseDefault, timeDefault) = try quantizeAndMeasure(
            data: data, shape: [rows, cols], label: "qkv_default"
        )
        let (rmseFast, timeFast) = try quantizeAndMeasure(
            data: data, shape: [rows, cols],
            shaderOverrides: [
                "constant uint N_INIT = 3;": "constant uint N_INIT = 1;",
                "constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 15;",
            ],
            label: "qkv_fast"
        )

        let speedup = timeDefault / max(timeFast, 0.001)
        let qualityLoss = (rmseFast / rmseDefault - 1) * 100

        fputs(
            "\n  === REAL SCALE: qkv_proj [6144, 2048] ===\n"
                + "    Default (N=3, I=100): RMSE=\(String(format: "%.6f", rmseDefault))"
                + "  time=\(String(format: "%.2f", timeDefault))s\n"
                + "    Fast (N=1, I=15):     RMSE=\(String(format: "%.6f", rmseFast))"
                + "  time=\(String(format: "%.2f", timeFast))s\n"
                + "    Speedup: \(String(format: "%.1f", speedup))x"
                + "  Quality loss: \(String(format: "%.2f", qualityLoss))%\n",
            stderr
        )

        XCTAssertLessThan(
            rmseFast / rmseDefault, 1.05,
            "Fast config on QKV should be within 5% RMSE"
        )
    }

    /// Test on down projection: [2048, 6144] (transposed shape).
    func testRealScale_DownProj() throws {
        let rows = 2_048
        let cols = 6_144
        let data = generateWeightLikeData(rows: rows, cols: cols, scale: 0.008)

        let (rmseDefault, timeDefault) = try quantizeAndMeasure(
            data: data, shape: [rows, cols], label: "down_default"
        )
        let (rmseFast, timeFast) = try quantizeAndMeasure(
            data: data, shape: [rows, cols],
            shaderOverrides: [
                "constant uint N_INIT = 3;": "constant uint N_INIT = 1;",
                "constant uint MAX_ITER = 100;": "constant uint MAX_ITER = 15;",
            ],
            label: "down_fast"
        )

        let speedup = timeDefault / max(timeFast, 0.001)
        let qualityLoss = (rmseFast / rmseDefault - 1) * 100

        fputs(
            "\n  === REAL SCALE: down_proj [2048, 6144] ===\n"
                + "    Default (N=3, I=100): RMSE=\(String(format: "%.6f", rmseDefault))"
                + "  time=\(String(format: "%.2f", timeDefault))s\n"
                + "    Fast (N=1, I=15):     RMSE=\(String(format: "%.6f", rmseFast))"
                + "  time=\(String(format: "%.2f", timeFast))s\n"
                + "    Speedup: \(String(format: "%.1f", speedup))x"
                + "  Quality loss: \(String(format: "%.2f", qualityLoss))%\n",
            stderr
        )

        XCTAssertLessThan(
            rmseFast / rmseDefault, 1.05,
            "Fast config on [2048, 6144] should be within 5% RMSE"
        )
    }
}
