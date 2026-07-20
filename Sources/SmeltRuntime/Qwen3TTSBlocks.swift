// Qwen3TTSBlocks — the TTS pipeline's blocks behind the scheduled-loop
// hooks (B2.2). Each block composes TalkerSession's encode/host halves —
// the same halves the monolithic `generateCodes` loop drives — so the
// orchestrated and monolithic paths share one implementation and cannot drift.
//
// The graph's data edges are realized through the shared session
// (codec-head → mtp-head cb0 tokens via `currentCb0`; mtp-head → talker
// feedback via `lastCodes`), per the B2.1 contract: the schedule says when,
// the edges say what flows, blocks own how.

import Foundation

/// A block was driven in a phase/step it doesn't implement — a malformed
/// schedule (the loop vocabulary failed). Fail closed rather than silently run
/// the wrong work in the wrong command-buffer scope.
enum Qwen3TTSBlockError: Error, CustomStringConvertible {
    case unexpectedPhase(block: String, phase: String)
    var description: String {
        switch self {
        case let .unexpectedPhase(block, phase):
            return "block '\(block)' driven in unsupported phase '\(phase)' — malformed schedule"
        }
    }
}

/// Trunk: prompt prefill in setup; feedback-embedding build + single-token
/// decode in `advance` (its own emissions re-enter as embedding sums — the
/// declared feedback edge).
final class Qwen3TTSTalkerBlock: SmeltScheduledBlock {
    typealias Element = [Int]
    let blockName = "talker"
    private let s: TalkerSession

    init(_ session: TalkerSession) { s = session }

    func encode(phase: String, step: Int) throws {
        SmeltDecodeProfile.setStage("talker")
        // prefill is a setup phase (step < 0); advance is a per-step phase.
        switch (phase, step < 0) {
        case ("prefill", true): try s.encodePrefill()
        case ("advance", false): try s.encodeFeedbackAndDecode()
        default: throw Qwen3TTSBlockError.unexpectedPhase(block: blockName, phase: phase)
        }
    }

    func finish(phase: String, step: Int) throws -> SmeltBlockSignal<[Int]> {
        if phase == "advance" { s.advancePosition() }
        return .none
    }
}

/// cb0 head: codec_head logits + selection ride in the talker's scope
/// (prefill computes frame 0's cb0; `advance` at step N computes frame
/// N+1's — the pipelined shape). The host half samples/reads, gates EOS,
/// and appends the repetition-penalty history.
final class Qwen3TTSCb0HeadBlock: SmeltScheduledBlock {
    typealias Element = [Int]
    let blockName = "codec-head"
    private let s: TalkerSession

    init(_ session: TalkerSession) { s = session }

    private func frame(for step: Int) -> Int { step < 0 ? 0 : step + 1 }

    func encode(phase: String, step: Int) throws {
        // cb0 rides the talker's scope: prefill (setup, frame 0) and advance
        // (per-step, frame N+1) — the phase implies the step sign.
        switch (phase, step < 0) {
        case ("prefill", true), ("advance", false): try s.encodeCb0(frame: frame(for: step))
        default: throw Qwen3TTSBlockError.unexpectedPhase(block: blockName, phase: phase)
        }
    }

    func finish(phase: String, step: Int) throws -> SmeltBlockSignal<[Int]> {
        let cb0 = s.takeCb0()
        if cb0 == s.eos { return .init(stop: true) }
        s.acceptCb0(cb0)
        return .none
    }
}

/// MTP head: the 15-sub-pass fan-out across the residual codebooks,
/// consuming the cb0 token + the talker's resident hidden. Emits the
/// completed 16-codebook frame (the loop's per-step product).
final class Qwen3TTSMTPHeadBlock: SmeltScheduledBlock {
    typealias Element = [Int]
    let blockName = "mtp-head"
    private let s: TalkerSession
    private var frameState: Qwen3TTSGPU.MTPFrameState?

    init(_ session: TalkerSession) { s = session }

    func encode(phase: String, step: Int) throws {
        // MTP fan-out runs only in the per-step "mtp" phase (never setup).
        guard (phase, step < 0) == ("mtp", false) else {
            throw Qwen3TTSBlockError.unexpectedPhase(block: blockName, phase: phase)
        }
        SmeltDecodeProfile.setStage("mtp")
        let state = s.prepareMTPFrame()
        frameState = state
        try s.encodeMTP(state, frame: step)
    }

    func finish(phase: String, step: Int) throws -> SmeltBlockSignal<[Int]> {
        guard let state = frameState else { return .none }
        frameState = nil
        let codes16 = [s.currentCb0] + s.readMTPCodes(state)
        s.lastCodes = codes16
        return .init(emitted: codes16)
    }
}

/// Emission decoder: the streaming codec (its three-stage chunk pipeline
/// and stream carry are its own — the contract's block-internal scopes).
final class Qwen3TTSCodecStreamDecoder: SmeltChunkDecoder {
    typealias Element = [Int]
    private let stream: Qwen3TTSCodecStream

    init(_ stream: Qwen3TTSCodecStream) { self.stream = stream }

    var unitsDecoded: Int { stream.framesDecoded }

    func decodeChunk(_ items: [[Int]]) throws -> [Float] {
        try stream.decode(items)
    }
}
