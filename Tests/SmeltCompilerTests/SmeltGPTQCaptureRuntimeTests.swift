import Foundation
import Testing
@testable import SmeltRuntime
import SmeltSchema

// Integration test for the runtime GPTQ capture interpreter against the in-repo
// metal-prefill vehicle. Gated on the built package existing;
// skips cleanly when absent, like the
// other real-checkpoint tests in this suite.

private let qwen35PkgPath = "/tmp/qwen35-0.8b-pkg/Qwen_Qwen3.5-0.8B.smeltpkg"

@Test func gptqCaptureReadsTheRightInputSlotAtEachBoundary() throws {
    let capPath = "\(qwen35PkgPath)/gptq_capture_points.json"
    guard FileManager.default.fileExists(atPath: capPath) else { return }

    let points = try JSONDecoder().decode(
        SmeltGPTQCapturePoints.self,
        from: Data(contentsOf: URL(fileURLWithPath: capPath))
    ).prefill
    #expect(!points.isEmpty)

    let runtime = try SmeltRuntime(packagePath: qwen35PkgPath)
    let capture = SmeltActivationCapture()
    capture.captureHessian = true

    // One attention layer (3 = first attn in the delta/attn pattern) + its MLP.
    let layer = 3
    let q = "layers_\(layer)_self_attn_q_proj_weight"
    let k = "layers_\(layer)_self_attn_k_proj_weight"
    let v = "layers_\(layer)_self_attn_v_proj_weight"
    let o = "layers_\(layer)_self_attn_o_proj_weight"
    let gate = "layers_\(layer)_mlp_gate_proj_weight"
    let up = "layers_\(layer)_mlp_up_proj_weight"
    let down = "layers_\(layer)_mlp_down_proj_weight"
    let names = Set([q, k, v, o, gate, up, down])
    capture.captureHessianNames = names

    let seqLen = 8
    let tokens = (0..<seqLen).map { Int32(100 + $0) }
    try runtime.captureGPTQActivations(
        tokenIds: tokens,
        capturePoints: points.filter { names.contains($0.weightName) },
        into: capture
    )

    // Every captured weight saw exactly one row per token.
    for name in names { #expect(capture.calibrationRows(name) == seqLen, "\(name)") }

    let hq = try #require(capture.hessian(q))
    let hk = try #require(capture.hessian(k))
    let hv = try #require(capture.hessian(v))
    let ho = try #require(capture.hessian(o))
    let hGate = try #require(capture.hessian(gate))
    let hUp = try #require(capture.hessian(up))
    let hDown = try #require(capture.hessian(down))

    // Q/K/V read the SAME post-input-norm activation, so their Hessians are
    // bit-identical (same ssyrk over the same input bytes).
    #expect(hq == hk)
    #expect(hq == hv)
    // Gate/Up read the SAME post-attention-norm activation.
    #expect(hGate == hUp)

    // Q and Gate read the SAME fixed slot (normOutBuf) but at DIFFERENT boundaries
    // (pre- vs post-attention norm). Distinct Hessians prove the interpreter re-reads
    // the slot at each boundary rather than capturing once and reusing stale data.
    #expect(hq != hGate)

    // O (attention-output input) and Down (SwiGLU-intermediate input) read other
    // slots entirely — non-trivial and distinct from the norm-fed projections.
    #expect(!ho.allSatisfy { $0 == 0 })
    #expect(!hDown.allSatisfy { $0 == 0 })
    #expect(ho.count != hq.count)   // o input dim (2048) ≠ hidden (1024)
}

@Test func gptqCaptureIsIndependentAcrossCalls() throws {
    // Two separate captures of the SAME prompt on the SAME runtime must produce
    // identical Hessians. This fails if a prior prefill's leftover state (e.g.
    // DeltaNet conv/recurrent) contaminates the next — captureGPTQActivations
    // resets the working buffers per call to prevent exactly that.
    let capPath = "\(qwen35PkgPath)/gptq_capture_points.json"
    guard FileManager.default.fileExists(atPath: capPath) else { return }

    let points = try JSONDecoder().decode(
        SmeltGPTQCapturePoints.self,
        from: Data(contentsOf: URL(fileURLWithPath: capPath))
    ).prefill
    // A projection in a LATE attention layer (23), so any leftover recurrent state
    // from earlier delta layers would perturb its input on a contaminated second call.
    let name = "layers_23_self_attn_q_proj_weight"
    let pts = points.filter { $0.weightName == name }
    #expect(!pts.isEmpty)

    let runtime = try SmeltRuntime(packagePath: qwen35PkgPath)
    let tokens = (0..<10).map { Int32(500 + $0) }

    func captureOnce() throws -> [Float] {
        let cap = SmeltActivationCapture()
        cap.captureHessian = true
        cap.captureHessianNames = [name]
        try runtime.captureGPTQActivations(tokenIds: tokens, capturePoints: pts, into: cap)
        return try #require(cap.hessian(name))
    }

    let first = try captureOnce()
    let second = try captureOnce()
    #expect(first == second)
}
