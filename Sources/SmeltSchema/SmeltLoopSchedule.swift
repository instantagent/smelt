// SmeltLoopSchedule — the drive loop as declared data (B2 of
// docs/block-spec-plan.md): who steps whom, per emission, and which steps
// share a command buffer.
//
// The central mapping, taken from how the hand loops actually work: a
// declared PHASE is a command-buffer scope. The TTS per-frame loop is two
// GPU phases (the MTP sub-step fan-out | feedback-build + trunk decode +
// cb0 head); prefill + first cb0 is one setup phase. Text generation is
// the degenerate one-phase-per-step case. This selective batching IS
// the TTFA-critical shape (single-submission batching measurably
// regressed), so it is declared, never implied.
//
// Division of labor (the B2.1 vocabulary review's line): the schedule says
// WHEN blocks run and WHAT SHARES A BUFFER; the block graph's edges say
// WHAT FLOWS between them; everything else belongs to block
// implementations, reached through lifecycle hooks at phase boundaries —
// host-side work (the cb0 readback + sampler processors + history, the
// text grammar-mask compute overlapped between commit and wait) is a
// block's host half, and a block driven by `emission` owns its INTERNAL
// scope structure (the codec decoder's three-stage chunk pipeline is the
// codec block's business, not schedule data). B2.2's orchestrator must be
// generic against exactly this contract — if it needs per-model glue the
// schedule didn't say, the vocabulary failed, not the orchestrator.
//
// B2.1 declared the vocabulary; B2.2 made it load-bearing: the TTS
// streaming path runs on SmeltScheduledLoop consuming this schedule, and
// the chunk defaults here are the ONE truth — API nil-params and the
// CLI/serve surfaces all resolve request > voice.json > this declaration
// (pre-stamping packages keep their legacy defaults).

import Foundation

public struct SmeltLoopSchedule: Codable, Sendable, Equatable {

    /// One phase = one command-buffer scope. `blocks` are graph block names
    /// whose GPU work is encoded into that scope, in order. Host-side work
    /// (readbacks, CPU sampling, bookkeeping) happens between phases by
    /// construction.
    public struct Phase: Codable, Sendable, Equatable {
        public let name: String
        public let blocks: [String]
        /// True when this phase only prepares the NEXT step (the TTS
        /// `advance` phase: feedback build + trunk decode + next cb0).
        /// The orchestrator skips such phases when no next step will run —
        /// the hand loop's `guard frame < maxFrames - 1`. Optional so
        /// already-stamped manifests decode (nil = false).
        public let feedsNextStep: Bool?

        public init(name: String, blocks: [String], feedsNextStep: Bool? = nil) {
            self.name = name
            self.blocks = blocks
            self.feedsNextStep = feedsNextStep
        }

        public var signature: String {
            let base = "\(name):\(blocks.joined(separator: ","))"
            guard feedsNextStep == true else { return base }
            return "\(base):feeds-next-step"
        }
    }

    /// How the chunk target evolves after the first chunk.
    public enum ChunkGrowth: String, Codable, Sendable {
        /// Double toward `max` (the measured TTS schedule: low TTFA first
        /// chunk, throughput-sized steady state).
        case double
        /// Stay at `first` (max must equal first).
        case fixed
    }

    /// How generated steps become output the caller sees.
    public enum Emission: Codable, Sendable, Equatable {
        /// Every step emits.
        case perStep
        /// Steps buffer into chunks decoded by `via` (a head block whose
        /// output is the graph's output): `first` steps for the first
        /// chunk, then `growth` toward `max`. These are package DEFAULTS;
        /// per-request parameters (CLI flags, request fields) override.
        case chunked(first: Int, max: Int, growth: ChunkGrowth, via: String)
        /// Run the output block once, after all iterative per-step blocks complete.
        case final(via: String)

        public var signature: String {
            switch self {
            case .perStep:
                return "per-step"
            case .chunked(let first, let max, let growth, let via):
                return "chunked:\(first):\(max):\(growth.rawValue):\(via)"
            case .final(let via):
                return "final:\(via)"
            }
        }
    }

    /// Why the loop ends. Declared so an orchestrator knows what it must
    /// check (and a reader knows what the package can do).
    public enum Stop: String, Codable, Sendable {
        case eosToken = "eos-token"
        case maxSteps = "max-steps"
        case hostCancel = "host-cancel"
    }

    public let version: Int
    /// Run once per call (prompt prefill + first head selection).
    public let setup: [Phase]
    /// Run per generated step (token or frame).
    public let perStep: [Phase]
    public let emission: Emission
    public let stop: [Stop]

    public init(
        version: Int = 1,
        setup: [Phase],
        perStep: [Phase],
        emission: Emission,
        stop: [Stop]
    ) {
        self.version = version
        self.setup = setup
        self.perStep = perStep
        self.emission = emission
        self.stop = stop
    }

    public var setupSignatures: [String] {
        setup.map(\.signature)
    }

    public var perStepSignatures: [String] {
        perStep.map(\.signature)
    }

    public var emissionSignature: String {
        emission.signature
    }

    public enum ScheduleError: Error, CustomStringConvertible, Equatable {
        case malformed(String)

        public var description: String {
            switch self {
            case .malformed(let why): return "loop schedule: \(why)"
            }
        }
    }

    /// Structural validation against the graph the schedule drives.
    public func validate(against graph: SmeltBlockGraph) throws {
        guard version == 1 else {
            throw ScheduleError.malformed("unsupported version \(version)")
        }
        if perStep.isEmpty {
            guard case .final = emission else {
                throw ScheduleError.malformed("no per-step phases")
            }
        }
        guard !stop.isEmpty else {
            throw ScheduleError.malformed("no stop conditions")
        }
        let known = Set(graph.blocks.map(\.name))
        var seenPhases = Set<String>()
        for phase in setup + perStep {
            guard !phase.name.isEmpty, seenPhases.insert(phase.name).inserted else {
                throw ScheduleError.malformed(
                    "duplicate or empty phase name '\(phase.name)'")
            }
            guard !phase.blocks.isEmpty else {
                throw ScheduleError.malformed("phase '\(phase.name)' drives no blocks")
            }
            for name in phase.blocks where !known.contains(name) {
                throw ScheduleError.malformed(
                    "phase '\(phase.name)' drives unknown block '\(name)'")
            }
        }
        if case .chunked(let first, let max, let growth, let via) = emission {
            guard first >= 1, max >= first else {
                throw ScheduleError.malformed(
                    "chunked emission needs 1 <= first (\(first)) <= max (\(max))")
            }
            if growth == .fixed, max != first {
                throw ScheduleError.malformed(
                    "fixed chunk growth requires max == first (got \(first)/\(max))")
            }
            guard let target = graph.blocks.first(where: { $0.name == via }) else {
                throw ScheduleError.malformed(
                    "chunked emission via unknown block '\(via)'")
            }
            // The emission decoder is what the caller hears/sees: its
            // output must be the graph's output.
            guard target.output == graph.blocks.last?.output else {
                throw ScheduleError.malformed(
                    "emission block '\(via)' produces \(target.output.rawValue), "
                        + "not the graph output")
            }
        }
        if case .final(let via) = emission {
            guard let target = graph.blocks.first(where: { $0.name == via }) else {
                throw ScheduleError.malformed(
                    "final emission via unknown block '\(via)'")
            }
            guard target.output == graph.blocks.last?.output else {
                throw ScheduleError.malformed(
                    "final emission block '\(via)' produces \(target.output.rawValue), "
                        + "not the graph output")
            }
        }
    }

    // MARK: - Canonical schedules (what the hand loops do today)

    /// Text token-feedback loop: one command buffer per token (selection kernel rides in the
    /// trunk's buffer), every step emits, stops on EOS/max/callback.
    public static let tokenFeedbackText = SmeltLoopSchedule(
        setup: [Phase(name: "prefill", blocks: ["trunk", "text-head"])],
        perStep: [Phase(name: "decode", blocks: ["trunk", "text-head"])],
        emission: .perStep,
        stop: [.eosToken, .maxSteps, .hostCancel]
    )

    /// Qwen3-TTS: the measured selective-batching shape. Setup = prompt
    /// prefill + first cb0 in one buffer. Per frame: the MTP fan-out is one
    /// buffer; feedback-embedding build + trunk decode + next cb0 is one
    /// buffer; cb0 readback/sampling/history run on the host between them.
    /// Frames buffer into codec chunks on the 1/1 zero-buffer default
    /// (per-request schedule parameters override).
    public static let qwen3TTS = SmeltLoopSchedule(
        setup: [Phase(name: "prefill", blocks: ["talker", "codec-head"])],
        perStep: [
            Phase(name: "mtp", blocks: ["mtp-head"]),
            Phase(name: "advance", blocks: ["talker", "codec-head"],
                  feedsNextStep: true),
        ],
        emission: .chunked(first: 1, max: 1, growth: .double, via: "codec-decoder"),
        stop: [.eosToken, .maxSteps, .hostCancel]
    )

    /// Standalone Qwen3-TTS codec decoder block package: one decode phase
    /// produces the final audio artifact from provided codec frames.
    public static let qwen3TTSCodecDecoder = SmeltLoopSchedule(
        setup: [Phase(name: "decode", blocks: ["codec-decoder"])],
        perStep: [],
        emission: .final(via: "codec-decoder"),
        stop: [.maxSteps, .hostCancel]
    )
}
