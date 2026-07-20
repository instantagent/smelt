import XCTest
import Metal
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// Codec default-flip (docs/codec-default-flip-plan.md): `decodeCodec` is now SOLE-compiled — the conv+
/// transformer vocoder runs only as the `Qwen3TTSCodecEmitter` record table through
/// `SmeltCodecRecordRunner` (the hand dispatch-helper decode is retired). This gate is the
/// checkpoint-only correctness anchor that replaces the deleted compiled-vs-hand bit-exact gate.
///
/// Oracle = the STREAMING codec (`Qwen3TTSCodecStream`), an INDEPENDENT compiled reimplementation of the
/// decode (its own per-chunk `Qwen3TTSCodecStreamEmitter` record table — valid-conv + persistent caches,
/// vs this offline emitter's causal-pad single-shot). A single-chunk stream decode must be BYTE-IDENTICAL
/// to the compiled offline decode (both were proven byte-exact to the now-retired hand paths, so they
/// agree by construction). This is a fast GPU-vs-GPU cross-check (the pure-CPU `Qwen3TTSCodec.decodeCodec`
/// reference is intractable at production frame counts — the DAC does ~tens of billions of naive Swift
/// MACs). `frames = 84` exercises the pre_transformer sliding window (72) and gemm_tn row-rounding (84 not
/// a multiple of 16). Multi-chunk cache coverage is testCompiledStreamMatchesOfflineDecode (below).
final class Qwen3TTSCompiledCodecTests: XCTestCase {

    private let checkpoint = "/tmp/qwen3-tts-12hz"

    func testCompiledCodecMatchesStreamingDecode() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("flip-codec-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        // A bf16 full-pipeline package: its model.metallib compiles ALL Resources/Shaders (incl.
        // transpose_f32 / gemm_tn_f32 / sliding_attn_simd_f32 the sole-compiled codec hard-requires).
        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders",
            outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)

        // frames = 84: > window 72 AND not a multiple of 16. Deterministic in-range codes.
        let frames = 84
        let cfg = gpu.makeCodecConfig()
        var gen: [[Int]] = []                      // frame-major [16] per frame (streaming input)
        for t in 0..<frames {
            var row = [Int](repeating: 0, count: 16)
            row[0] = (t * 7 + 1) % cfg.firstN      // codebook 0 = semantic (< firstN)
            for k in 0..<15 { row[k + 1] = (t * 3 + k) % cfg.restN }  // codebooks 1..15 = acoustic (< restN)
            gen.append(row)
        }
        var codes = [Int32](repeating: 0, count: 16 * frames)   // codebook-major [16, frames] (offline input)
        for t in 0..<frames { for c in 0..<16 { codes[c * frames + t] = Int32(gen[t][c]) } }

        // Compiled offline path (the sole production path) + absolute wall (codec is once-per-utterance;
        // a 15-20% codec regression is visible on short utterances even when end-to-end stays talker-dominated).
        let t0 = CFAbsoluteTimeGetCurrent()
        let offline = try gpu.decodeCodec(codes: codes, frames: frames)
        let wallMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        FileHandle.standardError.write(Data("  [compiled decodeCodec wall] \(String(format: "%.1f", wallMs)) ms (\(frames) frames)\n".utf8))

        // Streaming path (independent compiled emitter — per-chunk record table), one chunk == offline.
        let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: frames)
        let t1 = CFAbsoluteTimeGetCurrent()
        let streamed = try stream.decode(gen)
        let streamMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        FileHandle.standardError.write(Data("  [streaming (compiled) decode wall] \(String(format: "%.1f", streamMs)) ms (\(frames) frames)\n".utf8))

        XCTAssertEqual(offline.count, 1920 * frames, "offline wav length != 1920*frames")
        XCTAssertEqual(streamed.count, offline.count, "streaming wav length != offline")
        XCTAssertTrue(offline.allSatisfy { $0.isFinite } && offline.contains { $0 != 0 },
                      "compiled wav not finite+nonzero — gate vacuous")
        XCTAssertEqual(offline.map(\.bitPattern), streamed.map(\.bitPattern),
                       "compiled offline decodeCodec != streaming decode (bit-level)")
    }

    /// C3 re-anchor gate (docs/codec-c3-streaming-plan.md): now that `Qwen3TTSCodecStream.decode` IS the
    /// compiled per-chunk record table (`Qwen3TTSCodecStreamEmitter`; the hand stream is retired), the
    /// anchor is compiled-STREAMING == compiled-OFFLINE `decodeCodec` byte-identical across chunkings —
    /// two INDEPENDENT compiled emitters (streaming valid-conv + persistent caches vs offline causal-pad
    /// single-shot) must agree. `frames = 100` saturates the 71-row pre_transformer history (eviction)
    /// AND the sliding window (72); the chunkings span all-at-once, all-singles (the M=1 gemv path +
    /// growing history), the production doubling schedule, and a ragged partial-final mix — exercising
    /// the persistent caches (concat→update across chunks) under every shape. Self-contained (builds its
    /// own package), so it covers multi-chunk cache behavior in CI without the SMELT_VOICE_PKG fixture
    /// that testTTSCodecStreamMatchesOfflineDecode needs.
    func testCompiledStreamMatchesOfflineDecode() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("c3-codec-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders",
            outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)

        let frames = 100                               // > window 72 → cache eviction + window clamp
        let cfg = gpu.makeCodecConfig()
        var gen: [[Int]] = []
        for t in 0..<frames {
            var row = [Int](repeating: 0, count: 16)
            row[0] = (t * 7 + 1) % cfg.firstN
            for k in 0..<15 { row[k + 1] = (t * 3 + k) % cfg.restN }
            gen.append(row)
        }
        // Compiled-offline reference (single-shot decodeCodec): codebook-major [16, frames].
        var codes = [Int32](repeating: 0, count: 16 * frames)
        for t in 0..<frames { for c in 0..<16 { codes[c * frames + t] = Int32(gen[t][c]) } }
        let offline = try gpu.decodeCodec(codes: codes, frames: frames)
        XCTAssertEqual(offline.count, frames * 1920, "offline wav length != 1920*frames")
        XCTAssertTrue(offline.allSatisfy { $0.isFinite } && offline.contains { $0 != 0 },
                      "offline wav not finite+nonzero — gate vacuous")

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
            let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: frames)   // fresh state per chunking
            var streamed: [Float] = []
            var idx = 0
            for size in chunking {
                streamed += try stream.decode(Array(gen[idx..<(idx + size)]))
                idx += size
            }
            XCTAssertEqual(streamed.count, offline.count, "length mismatch, chunking \(chunking)")
            if streamed.map(\.bitPattern) != offline.map(\.bitPattern) {
                let firstBad = (0..<min(streamed.count, offline.count)).first { streamed[$0] != offline[$0] }
                XCTFail("compiled stream != compiled offline for chunking \(chunking); first mismatch at sample "
                        + "\(firstBad.map(String.init) ?? "?") (frame \(firstBad.map { $0 / 1920 } ?? -1))")
            }
        }
    }

    /// C3 dtype gate (codex RED): a u4 build stores the rank≥2 `decoder.*` conv/transformer weights as
    /// bf16 ("read back via f32(), widened" — Qwen3TTSPackageBuilder). The compiled streaming codec runs
    /// f32 record kernels, so its realizer MUST widen those bf16 weights (gpu.codecStreamWeight), not bind
    /// raw bytes — else it feeds bf16 bytes to an f32 conv/gemm and produces garbage. This builds a u4
    /// package (codec decoder ⇒ bf16) and asserts compiled-stream == compiled-offline byte-identical:
    /// offline decodeCodec is dtype-aware (bufF32(f32(name))), so agreement proves the stream widens the
    /// same way. WITHOUT the fix this test fails (or NaNs); the bf16 gate above can't catch it (bf16
    /// packages keep codec weights f32, so the widen path never runs there).
    func testCompiledStreamMatchesOfflineDecodeU4() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("c3-u4-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(checkpointDir: checkpoint,
                                         checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
                                         shaderDir: "Resources/Shaders",
                                         outputPath: pkg, projDType: .u4, u4GroupSize: 64)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        // Sanity: the codec decoder weights really are non-f32 in this build (else the gate is vacuous).
        XCTAssertNotEqual(gpu.weightDType("decoder.pre_conv.conv.weight"), .f32,
                          "u4 build should store rank≥2 decoder weights as bf16 — gate vacuous otherwise")

        let frames = 80                                // > window 72 → cache eviction too
        let cfg = gpu.makeCodecConfig()
        var gen: [[Int]] = []
        for t in 0..<frames {
            var row = [Int](repeating: 0, count: 16)
            row[0] = (t * 7 + 1) % cfg.firstN
            for k in 0..<15 { row[k + 1] = (t * 3 + k) % cfg.restN }
            gen.append(row)
        }
        var codes = [Int32](repeating: 0, count: 16 * frames)
        for t in 0..<frames { for c in 0..<16 { codes[c * frames + t] = Int32(gen[t][c]) } }
        let offline = try gpu.decodeCodec(codes: codes, frames: frames)
        XCTAssertTrue(offline.allSatisfy { $0.isFinite } && offline.contains { $0 != 0 }, "offline wav vacuous")

        for chunking in [[frames], Array(repeating: 1, count: frames)] {   // all-at-once + all-singles
            let stream = try Qwen3TTSCodecStream(gpu: gpu, maxFrames: frames)
            var streamed: [Float] = []
            var idx = 0
            for size in chunking { streamed += try stream.decode(Array(gen[idx..<(idx + size)])); idx += size }
            XCTAssertEqual(streamed.count, offline.count, "length, chunking \(chunking)")
            if streamed.map(\.bitPattern) != offline.map(\.bitPattern) {
                let firstBad = (0..<min(streamed.count, offline.count)).first { streamed[$0] != offline[$0] }
                XCTFail("u4 compiled stream != compiled offline, chunking \(chunking); first mismatch at sample "
                        + "\(firstBad.map(String.init) ?? "?") (frame \(firstBad.map { $0 / 1920 } ?? -1))")
            }
        }
    }

    // Phase 1a-i: the on-GPU TTS front-end (frontEndPrefillHiddenA = layout-driven gather → GPU
    // fc2(silu(fc1)) → scale_residual_tc merge) must reproduce the CPU/BLAS voiceDesignPrefill on the
    // real weights. GPU matmul K-reduction order != cblas_sgemm, so this is the planned tolerance gate
    // (cosine >= 0.999, relL2 <= 0.02), not bit-exact. Both paths gather via the SAME weightRow accessor,
    // so the comparison isolates the projection+merge numerics. Driven from an explicit in-bounds layout
    // (no tokenizer dependency); exercises instruct + think prefix + speaker + text + the specials.
    func testGPUFrontEndMatchesHostVoiceDesignPrefill() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("frontend-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders", outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let shape = try gpu.talkerShape()

        // Explicit valid in-bounds layout (no tokenize/config): text-vocab ids for the
        // text/role/instruct/special rows, codec-vocab ids for the tags.
        let instructIds = [11, 12]
        let inputIds = [21, 22, 23, 31, 32, 33, 90, 91, 92, 93, 94]   // role[3] + text[3] + trailing[5]
        let ids = Qwen3TTSTalkerPrefill.Ids(
            ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
            codecPad: 7, codecBos: 8, languageId: 50, codecNothink: nil, speakerId: 60)

        let w = gpu.textProjWeights()
        let ref = Qwen3TTSTalkerPrefill.voiceDesignPrefill(
            instructIds: instructIds, inputIds: inputIds, ids: ids,
            textRow: { gpu.weightRow("talker.model.text_embedding.weight", $0, shape.textEmbedDim) },
            codecRow: { gpu.weightRow("talker.model.codec_embedding.weight", $0, shape.hidden) },
            fc1W: w.fc1W, fc1B: w.fc1B, fc2W: w.fc2W, fc2B: w.fc2B,
            dim: shape.textEmbedDim, projInter: shape.projInter, hidden: shape.hidden).embeds

        let rows = Qwen3TTSTalkerPrefill.layout(instructIds: instructIds, inputIds: inputIds, ids: ids)
        let gpuBuf = try gpu.frontEndPrefillHiddenA(rows: rows, shape: shape)
        let got = gpu.readF32(gpuBuf, rows.count * shape.hidden)

        XCTAssertEqual(got.count, ref.count, "GPU front-end produced \(got.count) != host \(ref.count)")
        XCTAssertTrue(got.allSatisfy { $0.isFinite } && got.contains { $0 != 0 }, "GPU front-end not finite+nonzero — vacuous")
        var dot = 0.0, na = 0.0, nb = 0.0, diff = 0.0
        for i in 0..<got.count { let a = Double(got[i]), b = Double(ref[i]); dot += a * b; na += a * a; nb += b * b; diff += (a - b) * (a - b) }
        let cos = dot / (na.squareRoot() * nb.squareRoot()), rl2 = diff.squareRoot() / nb.squareRoot()
        XCTAssertGreaterThanOrEqual(cos, 0.999, "GPU front-end vs host voiceDesignPrefill cosine \(cos)")
        XCTAssertLessThanOrEqual(rl2, 0.02, "GPU front-end vs host voiceDesignPrefill relL2 \(rl2)")
    }

    // Phase 1a-ii: the COMPILED front-end (Qwen3TTSFrontEndEmitter record table run through the generic
    // SmeltCodecRecordRunner) must be BYTE-IDENTICAL to the host-issued frontEndPrefillHiddenA (1a-i) —
    // same kernels (gemm_tn_f32/silu_f32/scale_residual_tc + gather_rows), same resident weights (fc1/fc2
    // fp32; text_embedding bf16 widened in-kernel like weightRow; codec fp32), same grids. Not tolerance:
    // the Lego record path reproduces the hand path bit-for-bit, so the host front-end stays the oracle.
    func testCompiledFrontEndMatchesHostIssuedByteIdentical() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("frontend-compiled-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders", outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let shape = try gpu.talkerShape()
        XCTAssertTrue(gpu.compiledFrontEndSupported, "bf16 package must support the compiled front-end")

        let instructIds = [11, 12]
        let ids = Qwen3TTSTalkerPrefill.Ids(
            ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
            codecPad: 7, codecBos: 8, languageId: 50, codecNothink: nil, speakerId: 60)
        // SHORT + LONG (seqLen > maxPrefillBatchSize, where the LIVE path would chunk the prefill). This
        // gate compares the FE BUFFERS directly (not the chunked decode — prefillTrunkChunked is covered
        // by the cross-chunk session gate): the compiled FE emits one record table for the whole seqLen,
        // so byte-identity at large seqLen guards the length-dependent records the short fixture can't
        // reach (codex adversarial review #4).
        let shortInput = [21, 22, 23, 31, 32, 33, 90, 91, 92, 93, 94]
        let longInput = [21, 22, 23] + (0..<320).map { ($0 * 7 + 13) % 1000 } + [90, 91, 92, 93, 94]
        for inputIds in [shortInput, longInput] {
            let rows = Qwen3TTSTalkerPrefill.layout(instructIds: instructIds, inputIds: inputIds, ids: ids)
            let host = gpu.readF32(try gpu.frontEndPrefillHiddenA(rows: rows, shape: shape), rows.count * shape.hidden)
            let compiled = gpu.readF32(try gpu.compiledFrontEndHiddenA(rows: rows, shape: shape), rows.count * shape.hidden)
            XCTAssertEqual(compiled.count, host.count, "seqLen \(rows.count): compiled \(compiled.count) != host \(host.count)")
            XCTAssertTrue(compiled.contains { $0 != 0 } && compiled.allSatisfy { $0.isFinite }, "seqLen \(rows.count): compiled front-end vacuous")
            XCTAssertEqual(compiled.map(\.bitPattern), host.map(\.bitPattern), "seqLen \(rows.count): compiled record table != host-issued (byte-level)")
        }
    }

    // Phase 1a-ii (codex adversarial review #1): the SAMPLED live decode through the compiled front-end
    // had no distinct oracle (the amplification gate is greedy-only). Decode the SAME prompt under a
    // fixed seed through both the compiled FE and the host-issued GPU FE: their hiddenA is byte-identical
    // (the gate above), so the sampled codes must be byte-identical too — this directly exercises the
    // live cb0_sample_topk + MTP sample_topk through the compiled FE (a broken FE would shift a CDF
    // boundary and diverge here even though greedy stayed identical).
    func testCompiledFrontEndSampledDecodeMatchesHostIssued() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("frontend-sampled-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders", outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)
        let shape = try gpu.talkerShape()

        let ids = Qwen3TTSTalkerPrefill.Ids(
            ttsBos: 1, ttsEos: 2, ttsPad: 3, codecThink: 4, codecThinkBos: 5, codecThinkEos: 6,
            codecPad: 7, codecBos: 8, languageId: 50, codecNothink: nil, speakerId: 60)
        let rows = Qwen3TTSTalkerPrefill.layout(
            instructIds: [11, 12], inputIds: [21, 22, 23, 31, 32, 33, 90, 91, 92, 93, 94], ids: ids)
        let ttsPad = 3   // the synthetic layout's ttsPad (feedback embed row); both paths use it identically
        let sampling = Qwen3TTSSampler.Params(seed: 7)
        // Self-contained: the two FE buffers are byte-identical (so the sampled codes MUST match); assert
        // that here too rather than relying on the separate byte-identical gate when run alone.
        let hostBuf = try gpu.frontEndPrefillHiddenA(rows: rows, shape: shape)
        let compBuf = try gpu.compiledFrontEndHiddenA(rows: rows, shape: shape)
        XCTAssertEqual(gpu.readF32(compBuf, rows.count * shape.hidden).map(\.bitPattern),
                       gpu.readF32(hostBuf, rows.count * shape.hidden).map(\.bitPattern),
                       "compiled vs host-issued FE hiddenA must be byte-identical (sampled gate precondition)")
        func sampledDecode(_ buf: MTLBuffer) throws -> [[Int]] {
            try gpu.generateCodes(inputsEmbeds: [], seqLen: rows.count, ttsPadId: ttsPad,
                                  maxFrames: 64, sampling: sampling, prebuiltInputsBuf: buf)
        }
        let hostCodes = try sampledDecode(hostBuf)
        let compCodes = try sampledDecode(compBuf)
        XCTAssertGreaterThanOrEqual(hostCodes.count, 2, "sampled decode must run >= 2 frames (min_new_tokens) — vacuous")
        XCTAssertEqual(compCodes.count, hostCodes.count, "sampled frame count: compiled \(compCodes.count) vs host-issued \(hostCodes.count)")
        XCTAssertEqual(compCodes, hostCodes, "compiled-FE sampled codes != host-issued-FE (same hiddenA + seed → identical cb0_sample_topk + MTP)")
    }

    // Phase 1a-i-B amplification gate (codex: hiddenA tolerance ALONE is insufficient — a 28-layer
    // trunk can amplify prefill drift into a flipped greedy code). The LIVE generate(text:) path now
    // assembles hiddenA via the GPU front-end; decode the SAME text through both the GPU front-end
    // (prebuiltInputsBuf) and the host CPU/BLAS front-end (inputsEmbeds), greedy, and confirm the
    // front-end swap does not change the produced codes — i.e. the >= 0.999 hiddenA cosine survives the
    // full trunk + MTP to identical discrete codes. Both gather via weightRow, so the only difference is
    // the GPU-matmul vs cblas projection. The package bundles the tokenizer (build(checkpointDir:)).
    func testGPUFrontEndLiveGenerateMatchesHostFrontEnd() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(checkpoint)/model.safetensors"),
              let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("needs the local \(checkpoint) checkpoint + Metal")
        }
        let root = fm.temporaryDirectory.appendingPathComponent("frontend-live-\(getpid())", isDirectory: true)
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let pkg = root.appendingPathComponent("tts", isDirectory: true).path
        try Qwen3TTSPackageBuilder.build(
            checkpointDir: checkpoint, checkpointPolicy: try qwen3TTSTestCheckpointPolicy(),
            shaderDir: "Resources/Shaders", outputPath: pkg, projDType: .bf16)
        let gpu = try Qwen3TTSGPU(packagePath: pkg, device: device)

        let text = "Hello, this is a test.", instruct = "Speak in a calm, clear voice.", lang = "English"
        let maxFrames = 48
        // Host front-end (CPU/BLAS voiceDesignPrefill → upload), greedy.
        let ie = try gpu.inputsEmbeds(text: text, instruct: instruct, language: lang)
        let hostCodes = try gpu.generateCodes(inputsEmbeds: ie.embeds, seqLen: ie.seqLen,
                                              ttsPadId: ie.ttsPadId, maxFrames: maxFrames, sampling: nil)
        // GPU front-end (the live generate(text:) path), greedy.
        let fe = try gpu.frontEndPrefillHiddenA(text: text, instruct: instruct, language: lang)
        let gpuCodes = try gpu.generateCodes(inputsEmbeds: [], seqLen: fe.seqLen, ttsPadId: fe.ttsPadId,
                                             maxFrames: maxFrames, sampling: nil, prebuiltInputsBuf: fe.buf)

        XCTAssertEqual(fe.seqLen, ie.seqLen, "GPU vs host front-end seqLen")
        XCTAssertFalse(hostCodes.isEmpty, "host front-end produced no frames — vacuous")
        let n = min(hostCodes.count, gpuCodes.count)
        let firstDiff = (0..<n).first { hostCodes[$0] != gpuCodes[$0] }
        let msg = "  [front-end amplification] host \(hostCodes.count) / gpu \(gpuCodes.count) frames; "
            + "first code divergence: \(firstDiff.map(String.init) ?? "none")\n"
        FileHandle.standardError.write(Data(msg.utf8))
        XCTAssertEqual(gpuCodes.count, hostCodes.count, "GPU vs host front-end frame count")
        XCTAssertEqual(gpuCodes, hostCodes, "GPU front-end codes must match host front-end (greedy)")
    }
}
