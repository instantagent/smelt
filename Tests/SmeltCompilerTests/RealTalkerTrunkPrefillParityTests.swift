import XCTest
import Metal
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// W2b — THE real-weights PREFILL parity gate (docs/talker-trunk-fit-audit.md).
/// The real Qwen3-TTS talker trunk, built from `/tmp/qwen3-tts-12hz` through the
/// generic compiler (the `.talker` adapter → SmeltFP32TrunkWriter → the W2
/// DenseTrunkPrefillEmitter), prefilled over a batch of embeddings via the runtime
/// `prefillTrunk` entry, must produce hidden states BIT-EXACT to the hand
/// Qwen3TTSGPU.talkerForwardPrefillGPU consuming the SAME weight bytes.
///
/// W5c proved the DECODE table bit-exact on the real checkpoint; this proves the
/// PREFILL table. Same harness shape: the hand side is built from the compiled
/// package's own weights.bin (mapped to `talker.*`), so both paths consume
/// byte-identical weights — the only variable is the compute path. RoPE tables
/// are pinned to the hand Float-math source on both sides (single chunk,
/// startPos 0).
///
/// Opt-in + heavy: loads the 1.7B-class checkpoint. Auto-skips without the local
/// checkpoint, Metal, or the package-build env.
final class RealTalkerTrunkPrefillParityTests: XCTestCase {

    private let frames = 24   // a single prefill chunk (≤ max_prefill_batch and ≤ 2048)
    private let checkpoint = "/tmp/qwen3-tts-12hz"

    /// Phase 4 U2b — the compiled-trunk GPTQ capture seam. The small-batch trunk (max_prefill_batch
    /// 8) carries its capture points; a 40-row hiddenA seed captures every projection's Hessian
    /// through `captureGPTQActivationsFromHidden` — CHUNKED (40/8 = 5 chunks, exercising the
    /// cross-chunk capture accumulation) — and each Hessian must be [k×k], finite, non-vacuous. This
    /// proves calibration can read activations through the COMPILED trunk (no hand talkerForward*GPU).
    func testCaptureGPTQActivationsFromHiddenProducesFiniteHessians() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/config.json"),
              fm.fileExists(atPath: "\(checkpoint)/model.safetensors") else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint")
        }
        let ir = FixtureModelIRs.talkerTrunkSmallBatch
        let hidden = ir.config.hiddenSize, layers = ir.config.numLayers
        let headDim = ir.config.attention!.headDim

        let root = fm.temporaryDirectory.appendingPathComponent("capture-seam-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let built = try managedBuild(
            ir: ir, inputName: "talker-trunk-smallbatch",
            outputDir: try makeTempDir(under: root).path,
            weightsDir: checkpoint, shaderDir: "Resources/Shaders")
        let runtime = try SmeltRuntime(packagePath: built.packagePath)
        XCTAssertEqual(runtime.maxPrefillBatchSize, 8, "fixture must be a max_prefill_batch 8 trunk")

        // The trunk carries gptq_capture_points.json (Phase 4 U2a).
        let points = try JSONDecoder().decode(
            SmeltGPTQCapturePoints.self,
            from: Data(contentsOf: URL(fileURLWithPath: "\(built.packagePath)/gptq_capture_points.json"))
        ).prefill
        XCTAssertEqual(points.count, 7 * layers, "one capture point per projection per layer")
        XCTAssertTrue(points.allSatisfy { !$0.inputIsFloat16 }, "trunk capture is FP32")

        let seqLen = 40   // 40 / max_prefill_batch 8 = 5 chunks → exercises chunked capture
        var seed: UInt64 = 0xa5a5_5a5a_c3c3_3c3c
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 33) & 0xFFFF) / Float(0x10000) * 0.2 - 0.1
        }
        let embeds = (0..<(seqLen * hidden)).map { _ in nextFloat() }
        let source = runtime.device.makeBuffer(
            bytes: embeds, length: seqLen * hidden * 4, options: .storageModeShared)!

        func slot(_ n: String) -> Int { runtime.manifest.buffers.slots.first { $0.name == n }!.index }
        try runtime.ensurePrefillCapacity(seqLen: seqLen)
        try runtime.ensureContextCapacity(seqLen)
        let cosT = (0..<(seqLen * headDim)).map { Float(cos(Double($0) * 1e-4)) }
        let sinT = (0..<(seqLen * headDim)).map { Float(sin(Double($0) * 1e-4)) }
        runtime.writeSlot(slot("ropeCos"), f32: cosT)
        runtime.writeSlot(slot("ropeSin"), f32: sinT)

        let cap = SmeltActivationCapture()
        cap.captureHessian = true
        cap.captureHessianNames = Set(points.map(\.weightName))
        try runtime.captureGPTQActivationsFromHidden(
            source: source, seqLen: seqLen, hidden: hidden, capturePoints: points, into: cap)

        for p in points {
            guard let h = cap.hessian(p.weightName) else {
                return XCTFail("no Hessian for \(p.weightName)")
            }
            XCTAssertEqual(h.count, p.k * p.k, "\(p.weightName) Hessian shape != k×k")
            XCTAssertTrue(h.allSatisfy { $0.isFinite }, "\(p.weightName) Hessian not finite")
            XCTAssertTrue(h.contains { $0 != 0 }, "\(p.weightName) Hessian all-zero (vacuous capture)")
            // Each chunk contributes its rows; 5 chunks of 8 → 40 calibration rows.
            XCTAssertEqual(cap.calibrationRows(p.weightName), seqLen,
                           "\(p.weightName) calibration rows \(cap.calibrationRows(p.weightName)) != \(seqLen)")
        }
    }

    /// Phase 4 U1 — THE variable-length (>2048) prefill gate. A 2080-frame prompt (over the
    /// SIMD cached attention's 2048 sc[] cap) prefilled through `prefillTrunkChunked` runs
    /// COMPILED end-to-end: the over-cap chunk (startPos 2048, cache 2080 > 2048) routes to
    /// the uncapped scalar attention. This is the INTEGRATION gate (capacity sizing + chunking
    /// + substitution all work at >2048): the final hidden is finite+nonzero, the substitution
    /// counter advanced (non-vacuity — the SIMD kernel would read past its sc[2048] stage), and
    /// the result is deterministic across runs. The scalar kernel's NUMERICAL correctness at
    /// >2048 is proven separately + reliably by CachedAttentionScalarParityTests (vs the proven
    /// uncapped non-cached scalar kernel); the chunking MATH was proven byte-exact at <=2048 during
    /// bring-up (the hand oracle is now retired). The legacy hand prefill was NOT a usable >2048
    /// oracle (its scratch path doesn't survive long sequences — exactly why it was retired).
    func testOverCapPrefillRunsCompiledWithScalarSubstitution() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/config.json"),
              fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        guard !fm.fileExists(atPath: "\(checkpoint)/weights.json") else {
            return XCTFail("stray weights.json at \(checkpoint) would bypass the .talker adapter")
        }

        // The normal batch-256 trunk; 2080 frames → 9 chunks, the last (startPos 2048,
        // cache 2080) over the 2048 cap → scalar substitution.
        let ir = FixtureModelIRs.talkerTrunk
        let hidden = ir.config.hiddenSize
        let headDim = ir.config.attention!.headDim
        let frames = 2080

        let root = fm.temporaryDirectory
            .appendingPathComponent("overcap-prefill-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let built = try managedBuild(
            ir: ir, inputName: "talker-trunk",
            outputDir: try makeTempDir(under: root).path,
            weightsDir: checkpoint, shaderDir: "Resources/Shaders")
        let runtime = try SmeltRuntime(packagePath: built.packagePath)

        var seed: UInt64 = 0x0fed_cba9_8765_4321
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 33) & 0xFFFF) / Float(0x10000) * 0.2 - 0.1
        }
        let embeds = (0..<(frames * hidden)).map { _ in nextFloat() }
        // RoPE pinned to the hand Float-math source, sized to the full prompt.
        let cosT = (0..<(frames * headDim)).map { Float(cos(Double($0) * 1e-4)) }
        let sinT = (0..<(frames * headDim)).map { Float(sin(Double($0) * 1e-4)) }

        func slot(_ n: String) -> Int { runtime.manifest.buffers.slots.first { $0.name == n }!.index }
        try runtime.ensurePrefillCapacity(seqLen: frames)
        try runtime.ensureContextCapacity(frames)
        runtime.writeSlot(slot("ropeCos"), f32: cosT)
        runtime.writeSlot(slot("ropeSin"), f32: sinT)
        let source = runtime.device.makeBuffer(
            bytes: embeds, length: frames * hidden * 4, options: .storageModeShared)!

        // Run twice (fresh dest each time) → finite, nonzero, deterministic.
        func runOnce() throws -> (chunks: Int, subs: Int, last: [Float]) {
            try runtime.ensureContextCapacity(frames)   // grow-only; KV must hold the full prompt
            let dest = runtime.device.makeBuffer(length: hidden * 4, options: .storageModeShared)!
            let dptr = dest.contents().bindMemory(to: Float.self, capacity: hidden)
            for i in 0..<hidden { dptr[i] = .nan }      // poison: the blit-out must overwrite
            SmeltRuntime.cachedAttnScalarSubstitutions = 0
            let chunks = try runtime.prefillTrunkChunked(
                source: source, dest: dest, seqLen: frames, hidden: hidden)
            return (chunks, SmeltRuntime.cachedAttnScalarSubstitutions,
                    Array(UnsafeBufferPointer(start: dptr, count: hidden)))
        }

        let a = try runOnce()
        let b = try runOnce()

        XCTAssertGreaterThan(a.chunks, 1, "2080 frames / batch 256 must chunk")
        XCTAssertGreaterThan(a.subs, 0,
            "the >2048 chunk must have routed to the scalar attention — gate is vacuous otherwise")
        XCTAssertTrue(a.last.allSatisfy { $0.isFinite } && a.last.contains { $0 != 0 },
                      "over-cap last-row hidden is not finite+nonzero — the compiled >2048 path is broken")
        // Determinism: byte-identical across runs (catches uninitialized/race in the >2048 path).
        let same = a.last.withUnsafeBytes { x in b.last.withUnsafeBytes { y in x.elementsEqual(y) } }
        XCTAssertTrue(same, "compiled >2048 prefill is not deterministic across runs")
    }
}
