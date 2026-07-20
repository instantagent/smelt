// QuantizerQualityTests — Compare Agent GPU quantizer vs sklearn-style quantization.
//
// Measures the quality gap between linear-spread init (current) and
// quantile-based init (what sklearn effectively does on 1D data).

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class QuantizerQualityTests: XCTestCase {

    private var device: MTLDevice!

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1",
            "Set SMELT_RUN_SLOW_TESTS=1 to run quantizer quality tests"
        )
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "No Metal device available")
    }

    // MARK: - Helpers

    /// Generate weight-like data: peaked near zero, roughly normal.
    private func generateNormalWeights(count: Int, scale: Float = 0.02) -> [Float] {
        var data = [Float](repeating: 0, count: count)
        for idx in 0..<count {
            let u1 = (Float(idx % 997) + 1) / 998.0
            let u2 = (Float(idx % 991) + 1) / 992.0
            let normal = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            data[idx] = normal * scale
        }
        return data
    }

    /// CPU reference quantizer using optimal 1D quantization (quantile-based).
    /// For 1D data, optimal k-means centroids are close to the quantile midpoints.
    private func referenceQuantize(
        _ values: [Float], numCentroids: Int
    ) -> (centroids: [Float], rmse: Double) {
        // Sort values
        let sorted = values.sorted()
        let count = sorted.count

        // Initialize centroids at quantile midpoints
        var centroids = [Float](repeating: 0, count: numCentroids)
        for idx in 0..<numCentroids {
            let lo = count * idx / numCentroids
            let hi = count * (idx + 1) / numCentroids
            let mid = (lo + hi) / 2
            centroids[idx] = sorted[mid]
        }

        // Run Lloyd's algorithm
        for _ in 0..<100 {
            // Assign
            var sums = [Double](repeating: 0, count: numCentroids)
            var counts = [Int](repeating: 0, count: numCentroids)
            for val in values {
                var best = 0
                var bestDist = Float.infinity
                for cidx in 0..<numCentroids {
                    let dist = (val - centroids[cidx]) * (val - centroids[cidx])
                    if dist < bestDist { bestDist = dist; best = cidx }
                }
                sums[best] += Double(val)
                counts[best] += 1
            }
            // Update
            var maxShift: Float = 0
            for cidx in 0..<numCentroids {
                if counts[cidx] > 0 {
                    let newC = Float(sums[cidx] / Double(counts[cidx]))
                    let shift = (newC - centroids[cidx]) * (newC - centroids[cidx])
                    if shift > maxShift { maxShift = shift }
                    centroids[cidx] = newC
                }
            }
            if maxShift < 1e-12 { break }
        }

        // Compute RMSE
        var sumSqErr: Double = 0
        for val in values {
            var bestDist = Float.infinity
            for cen in centroids {
                let dist = (val - cen) * (val - cen)
                if dist < bestDist { bestDist = dist }
            }
            sumSqErr += Double(bestDist)
        }
        let rmse = sqrt(sumSqErr / Double(values.count))
        return (centroids, rmse)
    }

    /// Run Agent GPU quantizer on FP16 data and return RMSE.
    private func agentQuantizeRMSE(data: [Float16], shape: [Int]) throws -> Double {
        let count = data.count
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1

        let outputPath = NSTemporaryDirectory()
            + "qqual_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: []
        )

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
        return sqrt(sumSqErr / Double(count))
    }

    // MARK: - Quality comparison

    /// Compare Agent GPU quantizer vs quantile-initialized reference on weight-like data.
    func testQualityGap_WeightDistribution() throws {
        let rows = 64
        let cols = 128
        let f32Data = generateNormalWeights(count: rows * cols, scale: 0.02)
        let fp16Data = f32Data.map { Float16($0) }

        // Reference: quantile-initialized KMeans on each group
        var refSumSqErr: Double = 0
        let groupSize = 16
        for group in 0..<(rows / groupSize) {
            let start = group * groupSize * cols
            let end = start + groupSize * cols
            let groupVals = Array(f32Data[start..<end])
            let (_, groupRMSE) = referenceQuantize(groupVals, numCentroids: 16)
            refSumSqErr += groupRMSE * groupRMSE * Double(groupVals.count)
        }
        let refRMSE = sqrt(refSumSqErr / Double(f32Data.count))

        // Agent GPU
        let agentRMSE = try agentQuantizeRMSE(data: fp16Data, shape: [rows, cols])

        let ratio = agentRMSE / refRMSE

        fputs(
            "\n  === QUALITY GAP: weight-like [64, 128] ===\n"
                + "    Reference (quantile init): RMSE=\(String(format: "%.6f", refRMSE))\n"
                + "    Agent GPU (linear init):   RMSE=\(String(format: "%.6f", agentRMSE))\n"
                + "    Ratio (Agent/Ref):         \(String(format: "%.3f", ratio))x\n"
                + "    Gap:                       \(String(format: "%.1f", (ratio - 1) * 100))%%\n",
            stderr
        )

        // Agent should be within 20% of the reference
        XCTAssertLessThan(ratio, 1.20, "Smelt RMSE should be within 20% of quantile reference")
    }

    /// Same comparison on a larger, real-scale tensor.
    func testQualityGap_RealScale() throws {
        let rows = 256
        let cols = 512
        let f32Data = generateNormalWeights(count: rows * cols, scale: 0.01)
        let fp16Data = f32Data.map { Float16($0) }

        var refSumSqErr: Double = 0
        let groupSize = 16
        for group in 0..<(rows / groupSize) {
            let start = group * groupSize * cols
            let end = start + groupSize * cols
            let groupVals = Array(f32Data[start..<end])
            let (_, groupRMSE) = referenceQuantize(groupVals, numCentroids: 16)
            refSumSqErr += groupRMSE * groupRMSE * Double(groupVals.count)
        }
        let refRMSE = sqrt(refSumSqErr / Double(f32Data.count))

        let agentRMSE = try agentQuantizeRMSE(data: fp16Data, shape: [rows, cols])

        let ratio = agentRMSE / refRMSE

        fputs(
            "\n  === QUALITY GAP: weight-like [256, 512] ===\n"
                + "    Reference (quantile init): RMSE=\(String(format: "%.6f", refRMSE))\n"
                + "    Agent GPU (linear init):   RMSE=\(String(format: "%.6f", agentRMSE))\n"
                + "    Ratio (Agent/Ref):         \(String(format: "%.3f", ratio))x\n"
                + "    Gap:                       \(String(format: "%.1f", (ratio - 1) * 100))%%\n",
            stderr
        )

        XCTAssertLessThan(ratio, 1.20)
    }
}
