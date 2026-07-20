// PreserveNativeQuantizerTests — both weight quantizers must honor the
// `preserve_native` storage policy: a matched matvec projection is written as a
// RAW bf16 entry (dtype .bf16, 2 bytes/element, raw bf16 bytes, no fp16 downcast,
// not quantized), with a PREFLIGHT that the source is BF16. The layout
// (SmeltWeightPacker.appendWeightEntry) already preserves; these tests pin that
// SmeltQuantizer (GPU/LUT) and SmeltAffineQuantizer (CPU/affine) MATCH it exactly
// so layout/quantizer can't drift.

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class PreserveNativeQuantizerTests: XCTestCase {

    private var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
    }

    /// BF16 source bytes for `[rows, cols]` = FP32 upper 16 bits.
    private func makeBF16Tensor(rows: Int, cols: Int) -> [UInt16] {
        let count = rows * cols
        var data = [UInt16](repeating: 0, count: count)
        for idx in 0..<count {
            let fp32 = Float(sin(Float(idx) * 0.013) * cos(Float(idx) * 0.007))
            data[idx] = UInt16(fp32.bitPattern >> 16)
        }
        return data
    }

    // MARK: - SmeltQuantizer (GPU / LUT path)

    /// A matched bf16 projection is preserved: dtype .bf16, raw bf16 bytes.
    func testGPUQuantizerPreservesMatchedBF16Projection() throws {
        let rows = 64
        let cols = 64
        let bf16Data = makeBF16Tensor(rows: rows, cols: cols)

        let outputPath = NSTemporaryDirectory()
            + "preserve_gpu_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["*_q_proj_weight"]
        )

        let entries = try bf16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "layers_0_self_attn_q_proj_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: rows * cols * 2,
                    shape: [rows, cols],
                    dtype: "BF16"
                )],
                config: config,
                outputPath: outputPath,
                device: device,
                activationDtype: .fp16
            )
        }

        XCTAssertEqual(entries.count, 1)
        let entry = entries[0]
        XCTAssertEqual(entry.dtype, .bf16, "preserved projection must be tagged .bf16")
        XCTAssertEqual(entry.sizeBytes, UInt64(rows * cols * 2))
        XCTAssertNil(entry.groupSize)
        XCTAssertNil(entry.lutOffset)

        // Written bytes must EQUAL the source bf16 bytes (raw, not converted).
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        fileData.withUnsafeBytes { raw in
            let out = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: UInt16.self, capacity: rows * cols)
            for idx in 0..<(rows * cols) {
                XCTAssertEqual(
                    out[idx], bf16Data[idx],
                    "preserved bf16 byte mismatch at index \(idx) (must be raw, not downcast)")
            }
        }
    }

    /// A non-bf16 source for a matched projection must THROW the preflight error.
    func testGPUQuantizerThrowsOnNonBF16PreserveSource() throws {
        let rows = 64
        let cols = 64
        let count = rows * cols
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count { fp16Data[idx] = Float16(sin(Float(idx) * 0.01)) }

        let outputPath = NSTemporaryDirectory()
            + "preserve_gpu_throw_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["*_q_proj_weight"]
        )

        XCTAssertThrowsError(
            try fp16Data.withUnsafeBytes { raw in
                _ = try SmeltQuantizer.quantize(
                    tensors: [(
                        runtimeName: "layers_0_self_attn_q_proj_weight",
                        data: raw.baseAddress! as UnsafeRawPointer,
                        byteCount: count * 2,
                        shape: [rows, cols],
                        dtype: "F16"
                    )],
                    config: config,
                    outputPath: outputPath,
                    device: device,
                    activationDtype: .fp16
                )
            }
        ) { error in
            guard case SmeltQuantizerError.preserveNativeRequiresBF16Source = error else {
                XCTFail("expected preserveNativeRequiresBF16Source, got \(error)")
                return
            }
        }
    }

    /// A non-projection role (norm) matching the same pattern is NOT preserved —
    /// the role gate excludes it, so it stays fp16.
    func testGPUQuantizerDoesNotPreserveNonProjection() throws {
        let count = 64
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count { fp16Data[idx] = Float16(Float(idx) * 0.1) }

        let outputPath = NSTemporaryDirectory()
            + "preserve_gpu_norm_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        // Pattern would match by name, but the norm role is not preserve-eligible.
        let config = SmeltQuantizationConfig(
            strategy: .lutU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["input_layernorm_weight"]
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "input_layernorm_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [count],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device,
                activationDtype: .fp16
            )
        }

        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].dtype, .fp16, "a norm role must NOT be preserved")
    }

    // MARK: - SmeltAffineQuantizer (CPU / affine path)

    /// The affine quantizer must preserve the same way: dtype .bf16, raw bytes,
    /// NOT affine-quantized.
    func testAffineQuantizerPreservesMatchedBF16Projection() throws {
        let rows = 64
        let cols = 64
        let bf16Data = makeBF16Tensor(rows: rows, cols: cols)

        let outputPath = NSTemporaryDirectory()
            + "preserve_affine_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["*_q_proj_weight"]
        )

        let entries = try bf16Data.withUnsafeBytes { raw in
            try SmeltAffineQuantizer.quantize(
                tensors: [(
                    runtimeName: "layers_0_self_attn_q_proj_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: rows * cols * 2,
                    shape: [rows, cols],
                    dtype: "BF16"
                )],
                config: config,
                outputPath: outputPath,
                activationDtype: .fp16
            )
        }

        XCTAssertEqual(entries.count, 1)
        let entry = entries[0]
        XCTAssertEqual(entry.dtype, .bf16, "preserved projection must be tagged .bf16 (not affine_u4)")
        XCTAssertEqual(entry.sizeBytes, UInt64(rows * cols * 2))
        XCTAssertNil(entry.groupSize)
        XCTAssertNil(entry.scalesOffset)

        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        fileData.withUnsafeBytes { raw in
            let out = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: UInt16.self, capacity: rows * cols)
            for idx in 0..<(rows * cols) {
                XCTAssertEqual(
                    out[idx], bf16Data[idx],
                    "preserved bf16 byte mismatch at index \(idx) (must be raw, not affine-packed)")
            }
        }
    }

    /// Affine quantizer preflight: a non-bf16 source for a matched projection throws.
    func testAffineQuantizerThrowsOnNonBF16PreserveSource() throws {
        let rows = 64
        let cols = 64
        let count = rows * cols
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count { fp16Data[idx] = Float16(sin(Float(idx) * 0.01)) }

        let outputPath = NSTemporaryDirectory()
            + "preserve_affine_throw_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["*_q_proj_weight"]
        )

        XCTAssertThrowsError(
            try fp16Data.withUnsafeBytes { raw in
                _ = try SmeltAffineQuantizer.quantize(
                    tensors: [(
                        runtimeName: "layers_0_self_attn_q_proj_weight",
                        data: raw.baseAddress! as UnsafeRawPointer,
                        byteCount: count * 2,
                        shape: [rows, cols],
                        dtype: "F16"
                    )],
                    config: config,
                    outputPath: outputPath,
                    activationDtype: .fp16
                )
            }
        ) { error in
            guard case SmeltAffineQuantizerError.preserveNativeRequiresBF16Source = error else {
                XCTFail("expected preserveNativeRequiresBF16Source, got \(error)")
                return
            }
        }
    }

    /// Affine quantizer: a non-projection role matching the pattern is NOT preserved.
    func testAffineQuantizerDoesNotPreserveNonProjection() throws {
        let count = 64
        var fp16Data = [Float16](repeating: 0, count: count)
        for idx in 0..<count { fp16Data[idx] = Float16(Float(idx) * 0.1) }

        let outputPath = NSTemporaryDirectory()
            + "preserve_affine_norm_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 16,
            excludePatterns: [],
            preserveNativePatterns: ["input_layernorm_weight"]
        )

        let entries = try fp16Data.withUnsafeBytes { raw in
            try SmeltAffineQuantizer.quantize(
                tensors: [(
                    runtimeName: "input_layernorm_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [count],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                activationDtype: .fp16
            )
        }

        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].dtype, .fp16, "a norm role must NOT be preserved")
    }
}
