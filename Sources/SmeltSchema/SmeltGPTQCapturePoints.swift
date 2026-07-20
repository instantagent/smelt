import Foundation

/// Where GPTQ calibration reads each in-scope projection's activation input.
///
/// Emitted into a .smeltpkg (`gptq_capture_points.json`) by the prefill build and
/// consumed by the calibration capture interpreter. Each point names the boundary
/// in the FINAL (post-optimization) prefill dispatch stream after which the
/// projection's input buffer holds `[seqLen, K]` and can be read to accumulate the
/// activation Hessian.
public struct SmeltGPTQCapturePoint: Codable, Sendable {
    /// The projection weight whose Hessian this input feeds.
    public let weightName: String
    /// Fixed buffer slot holding the `[seqLen, K]` activation input (FP16 unless
    /// `inputIsFloat16` is false — the F32 talker-trunk ports are FP32).
    public let inputSlot: Int
    /// Input dimension (the matmul's `cols`); the width of each captured row.
    public let k: Int
    /// Number of dispatch ops to execute before the input is readable — the same
    /// dispatch-only count `SmeltTraceMarker` uses, resolved post-optimization.
    public let dispatchCount: Int
    /// Whether the captured input slot is FP16 (the text affine-u4 prefill ABI) or FP32
    /// (the F32 talker-trunk ABI). The runtime passes this to `accumulate(inputIsFloat16:)`,
    /// so an FP32 trunk input is read as fp32, not mis-decoded as packed fp16. Defaults to
    /// true (absent in pre-Phase-4 packages = the text fp16 path).
    public let inputIsFloat16: Bool

    public init(weightName: String, inputSlot: Int, k: Int, dispatchCount: Int,
                inputIsFloat16: Bool = true) {
        self.weightName = weightName
        self.inputSlot = inputSlot
        self.k = k
        self.dispatchCount = dispatchCount
        self.inputIsFloat16 = inputIsFloat16
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        weightName = try c.decode(String.self, forKey: .weightName)
        inputSlot = try c.decode(Int.self, forKey: .inputSlot)
        k = try c.decode(Int.self, forKey: .k)
        dispatchCount = try c.decode(Int.self, forKey: .dispatchCount)
        // Absent (pre-Phase-4 text packages) -> the fp16 path.
        inputIsFloat16 = try c.decodeIfPresent(Bool.self, forKey: .inputIsFloat16) ?? true
    }
}

/// All GPTQ capture points in a package, by stage. Only prefill is populated
/// today (calibration drives the Metal prefill).
public struct SmeltGPTQCapturePoints: Codable, Sendable {
    public let prefill: [SmeltGPTQCapturePoint]

    public init(prefill: [SmeltGPTQCapturePoint] = []) {
        self.prefill = prefill
    }

    /// Encodes package metadata with a stable object-key order so independent
    /// builds produce byte-identical artifacts.
    public func canonicalJSONData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }
}
