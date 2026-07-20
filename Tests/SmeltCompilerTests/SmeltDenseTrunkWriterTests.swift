import XCTest
@testable import SmeltCompiler
@testable import SmeltSchema

/// The dense-trunk real-checkpoint weight writer. This byte-level gate is
/// independent of any checkpoint adapter and verifies that
/// the writer produces bf16 projections RAW and fp32 norms via the canonical
/// widen, at the offsets `computeLayout` assigns, with no fp16 conversion and
/// no norm w−1 shift. (The real-talker end-to-end parity is W5, once the
/// generic checkpoint adapter maps talker.* names.)
final class SmeltFP32TrunkWriterTests: XCTestCase {

    func testWritesBf16RawAndFp32WidenedNoShift() throws {
        let ir = FixtureModelIRs.f32_trunk
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        XCTAssertFalse(layout.contains { $0.name == "embed_tokens" || $0.name == "lm_head_weight" },
                       "dense trunk layout must carry no embed/head")
        XCTAssertTrue(layout.allSatisfy { $0.dtype == .bf16 || $0.dtype == .fp32 })

        // A deterministic source tensor per layout entry. bf16 entries get
        // BF16 source (raw-copied). fp32 entries alternate BF16 source
        // (widened) and F32 source (raw-copied) to exercise both fp32 routes.
        var owned: [UnsafeMutableRawPointer] = []
        defer { owned.forEach { $0.deallocate() } }
        var tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = []
        var expectBf16: [String: [UInt16]] = [:]
        var expectF32: [String: [Float]] = [:]

        func bf16Buf(_ vals: [UInt16]) -> UnsafeRawPointer {
            let buf = UnsafeMutableRawPointer.allocate(byteCount: vals.count * 2, alignment: 2)
            vals.withUnsafeBytes { buf.copyMemory(from: $0.baseAddress!, byteCount: vals.count * 2) }
            owned.append(buf); return UnsafeRawPointer(buf)
        }
        func f32Buf(_ vals: [Float]) -> UnsafeRawPointer {
            let buf = UnsafeMutableRawPointer.allocate(byteCount: vals.count * 4, alignment: 4)
            vals.withUnsafeBytes { buf.copyMemory(from: $0.baseAddress!, byteCount: vals.count * 4) }
            owned.append(buf); return UnsafeRawPointer(buf)
        }

        for (idx, entry) in layout.enumerated() {
            let n = entry.shape.reduce(1, *)
            switch entry.dtype {
            case .bf16:
                let vals = (0..<n).map { UInt16(truncatingIfNeeded: idx * 131 + $0 * 7 + 1) }
                tensors.append((entry.name, bf16Buf(vals), n * 2, entry.shape, "BF16"))
                expectBf16[entry.name] = vals
            case .fp32 where idx % 2 == 0:
                let vals = (0..<n).map { UInt16(truncatingIfNeeded: idx * 97 + $0 * 5 + 3) }
                tensors.append((entry.name, bf16Buf(vals), n * 2, entry.shape, "BF16"))
                expectF32[entry.name] = vals.map { Float(bitPattern: UInt32($0) << 16) }
            case .fp32:
                let vals = (0..<n).map { Float(idx) * 0.5 + Float($0) * 0.25 }
                tensors.append((entry.name, f32Buf(vals), n * 4, entry.shape, "F32"))
                expectF32[entry.name] = vals
            default:
                XCTFail("unexpected layout dtype \(entry.dtype) for \(entry.name)")
            }
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fp32-writer-\(getpid())", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = dir.appendingPathComponent("weights.bin").path

        let returned = try SmeltFP32TrunkWriter.write(
            tensors: tensors, expectedLayout: layout, outputPath: out)
        XCTAssertEqual(returned.map(\.name), layout.map(\.name))

        let blob = try Data(contentsOf: URL(fileURLWithPath: out))
        let totalSize = layout.reduce(UInt64(0)) { max($0, $1.offset + $1.sizeBytes) }
        XCTAssertEqual(UInt64(blob.count), totalSize)

        for entry in layout {
            let off = Int(entry.offset)
            let n = entry.shape.reduce(1, *)
            switch entry.dtype {
            case .bf16:
                let got = blob.subdata(in: off..<off + n * 2).withUnsafeBytes {
                    Array($0.bindMemory(to: UInt16.self))
                }
                XCTAssertEqual(got, expectBf16[entry.name], "bf16 raw mismatch: \(entry.name)")
            case .fp32:
                let got = blob.subdata(in: off..<off + n * 4).withUnsafeBytes {
                    Array($0.bindMemory(to: Float.self))
                }
                XCTAssertEqual(got, expectF32[entry.name], "fp32 value mismatch: \(entry.name)")
            default:
                break
            }
        }
    }

    func testBf16EntryRejectsNonBf16Source() throws {
        // Parity with the hand path: bf16 specs require bf16 source — a F32
        // projection is a loud error, not a silent narrow.
        let entry = SmeltWeightEntry(
            name: "layers_0_self_attn_q_proj_weight", offset: 0,
            sizeBytes: 8, shape: [2, 2], dtype: .bf16)
        let vals = [Float](repeating: 1, count: 4)
        let buf = UnsafeMutableRawPointer.allocate(byteCount: 16, alignment: 4)
        defer { buf.deallocate() }
        vals.withUnsafeBytes { buf.copyMemory(from: $0.baseAddress!, byteCount: 16) }
        let dir = FileManager.default.temporaryDirectory
        let out = dir.appendingPathComponent("fp32-writer-reject-\(getpid()).bin").path
        defer { try? FileManager.default.removeItem(atPath: out) }
        XCTAssertThrowsError(try SmeltFP32TrunkWriter.write(
            tensors: [(entry.name, UnsafeRawPointer(buf), 16, [2, 2], "F32")],
            expectedLayout: [entry], outputPath: out))
    }

    func testWidensUnalignedBf16NormSource() throws {
        // Safetensors packs tensors back-to-back, so a BF16 norm's byte offset
        // is often odd. The widen must read unaligned — place the source at a
        // deliberately misaligned (odd) address and verify the fp32 output.
        let n = 5
        let bf16Bits: [UInt16] = [0x3F80, 0x4040, 0xBF00, 0x0001, 0x7F7F]
        let entry = SmeltWeightEntry(
            name: "norm_weight", offset: 0, sizeBytes: UInt64(n * 4),
            shape: [n], dtype: .fp32)

        let backing = UnsafeMutableRawPointer.allocate(byteCount: n * 2 + 1, alignment: 2)
        defer { backing.deallocate() }
        let misaligned = backing.advanced(by: 1)   // odd address
        XCTAssertEqual(Int(bitPattern: misaligned) % 2, 1, "source must be misaligned")
        _ = bf16Bits.withUnsafeBytes { memcpy(misaligned, $0.baseAddress!, n * 2) }

        let dir = FileManager.default.temporaryDirectory
        let out = dir.appendingPathComponent("fp32-writer-unaligned-\(getpid()).bin").path
        defer { try? FileManager.default.removeItem(atPath: out) }
        _ = try SmeltFP32TrunkWriter.write(
            tensors: [(entry.name, UnsafeRawPointer(misaligned), n * 2, [n], "BF16")],
            expectedLayout: [entry], outputPath: out)

        let got = try Data(contentsOf: URL(fileURLWithPath: out))
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertEqual(got, bf16Bits.map { Float(bitPattern: UInt32($0) << 16) })
    }

    func testNormShiftSkippedForDenseTrunks() throws {
        // The fixture is fp32 + norm_mode weight, so WITHOUT the fp32 guard the
        // norm-compat pass would shift these by w−1. The guard skips it (f32
        // norm kernels are weight-direct).
        let cfg = FixtureModelIRs.f32_trunk.config
        XCTAssertEqual(cfg.activationDtype, .fp32)
        XCTAssertEqual(cfg.normMode, .weight)
        XCTAssertFalse(SmeltCompiler.shouldShiftNormWeightForCompatibility(
            runtimeName: "norm_weight", config: cfg))
        XCTAssertFalse(SmeltCompiler.shouldShiftNormWeightForCompatibility(
            runtimeName: "layers_0_input_layernorm_weight", config: cfg))
    }
}
