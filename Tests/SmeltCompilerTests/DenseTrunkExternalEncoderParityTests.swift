import XCTest
import Metal
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// W3 (docs/talker-trunk-fit-audit.md): the compiled trunk encodes into a
/// CALLER-OWNED command encoder identically to the self-contained decodeStep /
/// prefillTrunk entries. The talker session composes the trunk into its
/// advance/prefill command-buffer scope (feedback-gather → trunk → cb0/MTP in one
/// buffer) via encodeTrunkDecode / encodeTrunkPrefill; this pins that those
/// produce BYTE-IDENTICAL hidden states to the wrapped entries.
///
/// Non-vacuity: the wrapped and external paths run on INDEPENDENT runtimes of the
/// same package (separate KV caches + normOutBuf), and normOutBuf is poisoned
/// (NaN) before the external encode — so a no-op external, an external that
/// secretly called decodeStep, or a missing/wrong external KV write cannot pass by
/// reading the wrapped path's leftover state.
final class DenseTrunkExternalEncoderParityTests: XCTestCase {

    private static let suiteRoot = makeManagedTempRoot("f32-extenc-\(getpid())")
    private let hidden = 256, steps = 4

    private func buildTwoRuntimes() throws -> (wrapped: SmeltRuntime, external: SmeltRuntime) {
        let built = try buildSyntheticPackageFromSpec(
            ir: FixtureModelIRs.f32_trunk, inputName: "f32-trunk",
            weights: .random, into: Self.suiteRoot)
        // Same package, two independent runtimes: identical weights + deterministic
        // production rope fill, but separate request-scoped state.
        return (try SmeltRuntime(packagePath: built.packagePath),
                try SmeltRuntime(packagePath: built.packagePath))
    }

    private func seededEmbeds(_ count: Int, seed: inout UInt64) -> [Float] {
        (0..<count).map { _ in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 33) & 0xFFFF) / Float(0x10000) * 0.2 - 0.1
        }
    }

    private func isLive(_ v: [Float]) -> Bool {
        v.allSatisfy { $0.isFinite } && v.contains { $0 != 0 }
    }

    /// The non-vacuity ritual shared by the decode and prefill contracts: on the
    /// independent `external` runtime, poison normOutBuf (NaN), write the
    /// embeddings, encode the trunk into a CALLER-OWNED encoder, assert the result
    /// is STILL poison after endEncoding but BEFORE our commit (the entry must only
    /// ENCODE — a secret internal commit would have produced a live result), then
    /// commit and assert byte-equality with the wrapped entry's `expected`.
    /// Capacity must already be ensured on `external` (the encode can't resize).
    private func assertEncodeMatchesWrapped(
        external: SmeltRuntime, embeds: [Float], count: Int, expected: [Float],
        queue: MTLCommandQueue, label: String = "",
        file: StaticString = #filePath, line: UInt = #line,
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let hiddenA = SmeltFixedSlot.hiddenA.rawValue
        let normOut = SmeltFixedSlot.normOutBuf.rawValue
        external.writeSlot(normOut, f32: [Float](repeating: .nan, count: count))
        external.writeSlot(hiddenA, f32: embeds)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        try encode(enc)
        enc.endEncoding()
        XCTAssertFalse(isLive(external.dumpSlot(normOut, count: count, asFP32: true)),
                       "\(label)the entry produced a result before the caller committed",
                       file: file, line: line)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let result = external.dumpSlot(normOut, count: count, asFP32: true)

        XCTAssertTrue(isLive(expected) && isLive(result),
                      "\(label)hidden state not finite+nonzero — contract is vacuous",
                      file: file, line: line)
        XCTAssertEqual(result, expected,
                       "\(label)external-encoder result != wrapped entry", file: file, line: line)
    }

    func testEncodeTrunkDecodeMatchesDecodeStep() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("needs Metal") }
        let (wrappedRT, externalRT) = try buildTwoRuntimes()
        try wrappedRT.ensureContextCapacity(64)
        try externalRT.ensureContextCapacity(64)

        let hiddenA = SmeltFixedSlot.hiddenA.rawValue
        let normOut = SmeltFixedSlot.normOutBuf.rawValue
        let queue = externalRT.device.makeCommandQueue()!
        var seed: UInt64 = 0xC0FFEE

        for pos in 0..<steps {
            let embeds = seededEmbeds(hidden, seed: &seed)
            wrappedRT.writeSlot(hiddenA, f32: embeds)
            _ = try wrappedRT.decodeStep(tokenId: 0, position: Int32(pos))
            let wrapped = wrappedRT.dumpSlot(normOut, count: hidden, asFP32: true)

            try assertEncodeMatchesWrapped(
                external: externalRT, embeds: embeds, count: hidden, expected: wrapped,
                queue: queue, label: "pos \(pos): "
            ) { enc in
                try externalRT.encodeTrunkDecode(into: enc, tokenId: 0, position: Int32(pos))
            }
        }
    }

    func testEncodeTrunkPrefillMatchesPrefillTrunk() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("needs Metal") }
        let (wrappedRT, externalRT) = try buildTwoRuntimes()

        let frames = 8
        var seed: UInt64 = 0xBEEF
        let embeds = seededEmbeds(frames * hidden, seed: &seed)

        // Self-contained: prefillTrunk ensures capacity (production rope fill),
        // writes embeddings, owns the command buffer.
        let wrapped = try wrappedRT.prefillTrunk(embeddings: embeds, seqLen: frames)

        // External: ensure capacity BEFORE opening the encoder (a grow reallocates
        // buffers), then encode into a caller-owned encoder.
        try externalRT.ensurePrefillCapacity(seqLen: frames)
        try assertEncodeMatchesWrapped(
            external: externalRT, embeds: embeds, count: frames * hidden, expected: wrapped,
            queue: externalRT.device.makeCommandQueue()!
        ) { enc in
            try externalRT.encodeTrunkPrefill(into: enc, seqLen: frames)
        }

        // KV-usability cross-check (W3 review): the prefill-hidden match above
        // proves the last row, but the session's real use is prefill-then-decode.
        // Run ONE decode at position=frames over each runtime's own prefilled
        // cache (decodeStep grows frames→frames+1, preserving the K/V prefix) and
        // require byte-identical hidden. A no-op or wrong external K/V write would
        // diverge here even though the prefill output matched.
        let hiddenA = SmeltFixedSlot.hiddenA.rawValue
        let normOut = SmeltFixedSlot.normOutBuf.rawValue
        var dseed: UInt64 = 0x5EED
        let decEmbeds = seededEmbeds(hidden, seed: &dseed)
        wrappedRT.writeSlot(hiddenA, f32: decEmbeds)
        _ = try wrappedRT.decodeStep(tokenId: 0, position: Int32(frames))
        let wDecode = wrappedRT.dumpSlot(normOut, count: hidden, asFP32: true)
        externalRT.writeSlot(hiddenA, f32: decEmbeds)
        _ = try externalRT.decodeStep(tokenId: 0, position: Int32(frames))
        let eDecode = externalRT.dumpSlot(normOut, count: hidden, asFP32: true)
        XCTAssertTrue(isLive(wDecode) && isLive(eDecode),
                      "post-prefill decode hidden not finite+nonzero — cross-check vacuous")
        XCTAssertEqual(eDecode, wDecode,
                       "post-prefill decode differs — external prefill's K/V cache is not usable")
    }

    /// Non-vacuity of the ABI guard: the fp32 headless-trunk entries must FAIL
    /// CLOSED on a normal fp16 package — they drive the embeddings-in/hidden-out
    /// ports and would otherwise silently run the full token-in/LM-head table
    /// under a trunk-API name. A real fp16 LLM package (a normal Qwen LLM) has hiddenA
    /// in fp16, so all three entries throw before encoding anything.
    func testTrunkEntriesRejectNonFP32Package() throws {
        try XCTSkipIf(skipIntegrationTests, "package-build integration test")
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("needs Metal") }
        let built = try buildSyntheticPackageFromSpec(
            ir: FixtureModelIRs.qwen35_0_8B, inputName: "qwen35-0.8b",
            weights: .local, into: Self.suiteRoot)
        let rt = try SmeltRuntime(packagePath: built.packagePath)
        try rt.ensureContextCapacity(8)

        // Self-contained prefill: the ABI guard runs before the size precondition,
        // so empty embeddings still reach (and trip) the guard.
        XCTAssertThrowsError(try rt.prefillTrunk(embeddings: [], seqLen: 1))

        // The encode-into-external-encoder entries: the guard throws before any
        // dispatch is encoded into the caller's encoder.
        let cmdBuf = rt.device.makeCommandQueue()!.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        XCTAssertThrowsError(try rt.encodeTrunkDecode(into: enc, tokenId: 0, position: 0))
        XCTAssertThrowsError(try rt.encodeTrunkPrefill(into: enc, seqLen: 1))
        enc.endEncoding()
    }
}
