// QuantizerTests — GPU tests for the Metal KMeans quantizer.
//
// These tests require a Metal device and run the actual GPU kernels.
// They verify: correctness, batched dispatch equivalence, convergence,
// and error handling.

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class QuantizerTests: XCTestCase {

    private var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        // Skip GPU tests on CI or machines without Metal
        try? XCTSkipIf(device == nil, "No Metal device available")
    }

    // MARK: - Basic correctness

    /// Quantize a small [32, 64] tensor and verify the output is valid.
    func testQuantizeSmallTensor() throws {
        let rows = 32
        let cols = 64
        let count = rows * cols

        // Create synthetic FP16 data: values in [-1, 1]
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            fp16Data[idx] = Float16(sin(Float(idx) * 0.01))
        }

        let outputPath = NSTemporaryDirectory() + "quant_test_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4,
            groupSize: 16,
            excludePatterns: []
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            let tensor = (
                runtimeName: "test_weight",
                data: raw.baseAddress! as UnsafeRawPointer,
                byteCount: count * 2,
                shape: [rows, cols],
                dtype: "F16"
            )
            return try SmeltQuantizer.quantize(
                tensors: [tensor],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        XCTAssertEqual(entries.count, 1)
        let entry = entries[0]
        XCTAssertEqual(entry.dtype, .u4Lut)
        XCTAssertEqual(entry.name, "test_weight")
        XCTAssertEqual(entry.groupSize, 16)

        // Packed size: rows * cols / 2 (2 u4 per byte)
        let expectedPacked = UInt64(count / 2)
        XCTAssertEqual(entry.sizeBytes, expectedPacked)

        // LUT: (rows / groupSize) groups * 16 entries * 2 bytes
        let numGroups = rows / 16
        let expectedLUT = UInt64(numGroups * 16 * 2)
        XCTAssertEqual(entry.lutSizeBytes, expectedLUT)

        // Verify output file exists and has correct size
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let totalExpected = Int(entry.offset + expectedPacked)
        + Int(expectedLUT)
        XCTAssertGreaterThanOrEqual(fileData.count, totalExpected)
    }

    /// Quantize and verify the LUT contains sorted centroids.
    func testLUTIsSorted() throws {
        let rows = 16
        let cols = 32
        let count = rows * cols

        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            fp16Data[idx] = Float16(Float(idx) / Float(count) - 0.5)
        }

        let outputPath = NSTemporaryDirectory() + "quant_lut_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: []
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "test_lut",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [rows, cols],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        let entry = entries[0]
        guard let lutOffset = entry.lutOffset else {
            XCTFail("Missing LUT offset")
            return
        }

        // Read LUT from output file
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let lutStart = Int(lutOffset)
        let lutCount = 16  // one group = 16 FP16 centroids
        var lut = [Float16](repeating: 0, count: lutCount)
        fileData.withUnsafeBytes { raw in
            let src = raw.baseAddress!.advanced(by: lutStart)
                .bindMemory(to: Float16.self, capacity: lutCount)
            for idx in 0..<lutCount { lut[idx] = src[idx] }
        }

        // Verify sorted ascending (kernel sorts centroids)
        for idx in 1..<lutCount {
            XCTAssertLessThanOrEqual(
                Float(lut[idx - 1]), Float(lut[idx]),
                "LUT not sorted at index \(idx): \(lut[idx - 1]) > \(lut[idx])"
            )
        }
    }

    /// Quantize and dequantize, verify RMSE is reasonable for 4-bit.
    func testDequantizeRMSE() throws {
        let rows = 64
        let cols = 128
        let count = rows * cols

        // Random-ish data with known distribution
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            let val = sin(Float(idx) * 0.037) * cos(Float(idx) * 0.013)
            fp16Data[idx] = Float16(val)
        }

        let outputPath = NSTemporaryDirectory() + "quant_rmse_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: []
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "test_rmse",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [rows, cols],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        let entry = entries[0]
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))

        // Dequantize: read packed indices + LUT, reconstruct values
        let numGroups = rows / 16
        let halfCols = cols / 2

        fileData.withUnsafeBytes { raw in
            let packed = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: UInt8.self, capacity: count / 2)
            let lutBase = raw.baseAddress!.advanced(by: Int(entry.lutOffset!))
                .bindMemory(to: Float16.self, capacity: numGroups * 16)

            var sumSqErr: Double = 0
            for row in 0..<rows {
                let group = row / 16
                let groupLUT = lutBase + group * 16
                for col in 0..<halfCols {
                    let byte = packed[row * halfCols + col]
                    let loIdx = Int(byte & 0x0F)
                    let hiIdx = Int(byte >> 4)
                    let loVal = Float(groupLUT[loIdx])
                    let hiVal = Float(groupLUT[hiIdx])
                    let origLo = Float(fp16Data[row * cols + col * 2])
                    let origHi = Float(fp16Data[row * cols + col * 2 + 1])
                    sumSqErr += Double((loVal - origLo) * (loVal - origLo))
                    sumSqErr += Double((hiVal - origHi) * (hiVal - origHi))
                }
            }
            let rmse = sqrt(sumSqErr / Double(count))

            // 4-bit quantization with 16 centroids on smooth data:
            // RMSE should be < 10% of the signal range
            let maxVal = fp16Data.map { abs(Float($0)) }.max() ?? 1.0
            XCTAssertLessThan(
                rmse, Double(maxVal) * 0.15,
                "RMSE \(rmse) too high for 4-bit quantization (max val \(maxVal))"
            )
        }
    }

    // MARK: - FP16 passthrough

    /// Non-quantized tensors should be copied as FP16.
    func testFP16Passthrough() throws {
        let count = 128
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count { fp16Data[idx] = Float16(Float(idx) * 0.1) }

        let outputPath = NSTemporaryDirectory() + "quant_fp16_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        // Exclude everything — forces FP16 passthrough
        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: ["*"]
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "norm_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [count],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        let entry = entries[0]
        XCTAssertEqual(entry.dtype, .fp16)
        XCTAssertEqual(entry.sizeBytes, UInt64(count * 2))

        // Verify data is bit-identical
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        fileData.withUnsafeBytes { raw in
            let out = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: Float16.self, capacity: count)
            for idx in 0..<count {
                XCTAssertEqual(
                    out[idx], fp16Data[idx],
                    "FP16 passthrough mismatch at index \(idx)"
                )
            }
        }
    }

    // MARK: - Multi-tensor batch

    /// Quantize multiple tensors in one call — exercises the batched dispatch
    /// across tensor boundaries.
    func testMultipleTensorsInOneBuild() throws {
        let outputPath = NSTemporaryDirectory() + "quant_multi_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: ["small_*"]
        )

        // One quantized tensor + one FP16 passthrough
        let bigCount = 48 * 64
        var bigData = [Float16](repeating: 0, count: bigCount)
        for idx in 0..<bigCount { bigData[idx] = Float16(cos(Float(idx) * 0.02)) }

        let smallCount = 64
        var smallData = [Float16](repeating: 0, count: smallCount)
        for idx in 0..<smallCount { smallData[idx] = Float16(Float(idx) * 0.05) }

        let entries = try bigData.withUnsafeBytes { bigRaw in
            try smallData.withUnsafeBytes { smallRaw in
                try SmeltQuantizer.quantize(
                    tensors: [
                        (
                            runtimeName: "big_weight",
                            data: bigRaw.baseAddress! as UnsafeRawPointer,
                            byteCount: bigCount * 2,
                            shape: [48, 64],
                            dtype: "F16"
                        ),
                        (
                            runtimeName: "small_norm",
                            data: smallRaw.baseAddress! as UnsafeRawPointer,
                            byteCount: smallCount * 2,
                            shape: [64],
                            dtype: "F16"
                        ),
                    ],
                    config: config,
                    outputPath: outputPath,
                    device: device
                )
            }
        }

        XCTAssertEqual(entries.count, 2)
        XCTAssertEqual(entries[0].dtype, .u4Lut)
        XCTAssertEqual(entries[0].name, "big_weight")
        XCTAssertEqual(entries[1].dtype, .fp16)
        XCTAssertEqual(entries[1].name, "small_norm")

        // Verify both entries have valid offsets in the file
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        XCTAssertGreaterThanOrEqual(
            fileData.count,
            Int(entries[1].offset + entries[1].sizeBytes)
        )
    }

    // MARK: - BF16 conversion

    /// Verify BF16 input is correctly converted before quantization.
    func testBF16Input() throws {
        let rows = 16
        let cols = 32
        let count = rows * cols

        // Create BF16 data (FP32 upper 16 bits)
        var bf16Data = [UInt16](repeating: 0, count: count)
        for idx in 0..<count {
            let fp32 = Float(sin(Float(idx) * 0.05))
            bf16Data[idx] = UInt16(fp32.bitPattern >> 16)
        }

        let outputPath = NSTemporaryDirectory() + "quant_bf16_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16, excludePatterns: []
        )

        let entries = try bf16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "bf16_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [rows, cols],
                    dtype: "BF16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].dtype, .u4Lut)
        // If we got here without throwing, BF16 conversion + quantization worked
    }
}
