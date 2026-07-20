// SmeltScheduledLoop — the generic drive loop (B2.2 of
// docs/block-spec-plan.md): runs a package's declared SmeltLoopSchedule over
// its block instances. The orchestrator owns exactly what the schedule
// declares — phase order, command-buffer scopes, emission chunking, stop
// conditions — and nothing else: data flow between blocks is construction-
// time wiring (the graph's edges, realized by the blocks sharing state),
// and each block's host half runs in its `finish` hook at the phase
// boundary. If driving a model needs per-model glue HERE, the schedule
// vocabulary failed (the B2.1 review contract).
//
// Scope discipline: the orchestrator opens one scope per declared phase
// (`scope` is the package's batched-command-buffer runner, which forbids
// nesting) — block `encode` hooks must only encode into it. `finish` hooks
// run after the scope committed, so host reads see real data.

import Foundation
import SmeltSchema

/// One step-driven block in a scheduled loop. `Element` is the per-step
/// product the loop emits (TTS: a 16-codebook frame).
public protocol SmeltScheduledBlock<Element>: AnyObject {
    associatedtype Element

    var blockName: String { get }
    /// Encode this block's GPU work for `phase` into the currently open
    /// scope. `step` is -1 during setup phases.
    func encode(phase: String, step: Int) throws
    /// Host half, after the phase's scope committed.
    func finish(phase: String, step: Int) throws -> SmeltBlockSignal<Element>
}

public struct SmeltBlockSignal<Element> {
    /// A completed per-step product (drives emission).
    public var emitted: Element?
    /// Stop the loop (EOS and friends). Emission still flushes.
    public var stop: Bool

    public init(emitted: Element? = nil, stop: Bool = false) {
        self.emitted = emitted
        self.stop = stop
    }

    public static var none: SmeltBlockSignal<Element> { .init() }
}

/// The emission decoder for `.chunked` schedules (the schedule's `via`
/// block): turns buffered elements into output samples, carrying its own
/// stream state — and its own internal command-buffer scopes (the contract:
/// emission-driven blocks own their internals).
public protocol SmeltChunkDecoder<Element>: AnyObject {
    associatedtype Element
    func decodeChunk(_ items: [Element]) throws -> [Float]
    /// Units already decoded (chunk offsets in the output stream).
    var unitsDecoded: Int { get }
}

public final class SmeltScheduledLoop<Element> {

    public struct Chunk {
        public let samples: [Float]
        public let offset: Int
        public let count: Int
        public let isFinal: Bool
    }

    public enum LoopError: Error, CustomStringConvertible {
        case unknownBlock(String)
        case missingDecoder(String)
        case nothingGenerated
        case undeclaredStop(String)

        public var description: String {
            switch self {
            case .unknownBlock(let name):
                return "schedule drives block '\(name)' but no instance was provided"
            case .missingDecoder(let via):
                return "emission via '\(via)' needs a SmeltChunkDecoder instance"
            case .nothingGenerated:
                return "loop produced no output"
            case .undeclaredStop(let which):
                return "loop stopped by '\(which)' but the schedule does not declare it"
            }
        }
    }

    private let schedule: SmeltLoopSchedule
    private let scope: (() throws -> Void) throws -> Void
    private let setupPhases: [(phase: SmeltLoopSchedule.Phase, blocks: [any SmeltScheduledBlock<Element>])]
    private let stepPhases: [(phase: SmeltLoopSchedule.Phase, blocks: [any SmeltScheduledBlock<Element>])]
    private let trace: SmeltRuntimeTraceRecorder?
    private let routesByBlock: [String: String]

    /// `blocks`: every instance the schedule's phases reference. `scope`:
    /// the package's batched command-buffer runner (one scope per phase).
    public init(
        schedule: SmeltLoopSchedule,
        blocks: [any SmeltScheduledBlock<Element>],
        scope: @escaping (() throws -> Void) throws -> Void,
        trace: SmeltRuntimeTraceRecorder? = nil,
        routesByBlock: [String: String] = [:]
    ) throws {
        // The declared stop set is load-bearing, not descriptive: the
        // runner always enforces its step cap, so the schedule must say so.
        guard schedule.stop.contains(.maxSteps) else {
            throw LoopError.undeclaredStop("max-steps")
        }
        self.schedule = schedule
        self.scope = scope
        self.trace = trace
        self.routesByBlock = routesByBlock
        var byName: [String: any SmeltScheduledBlock<Element>] = [:]
        for block in blocks { byName[block.blockName] = block }
        func resolve(_ phase: SmeltLoopSchedule.Phase) throws
            -> (SmeltLoopSchedule.Phase, [any SmeltScheduledBlock<Element>]) {
            let instances = try phase.blocks.map { name in
                guard let block = byName[name] else {
                    throw LoopError.unknownBlock(name)
                }
                return block
            }
            return (phase, instances)
        }
        setupPhases = try schedule.setup.map(resolve)
        stepPhases = try schedule.perStep.map(resolve)
    }

    /// Drive the loop. `maxSteps` is the caller's cap (the schedule's
    /// `maxSteps` stop). `chunkOverride` replaces the schedule's declared
    /// chunk defaults (per-request parameters win). The concatenation of
    /// all chunks' samples must equal the unchunked output — the decoder's
    /// contract, gated by the streaming-vs-offline tests.
    public func run(
        maxSteps: Int,
        chunkOverride: (first: Int, max: Int)? = nil,
        decoder: (any SmeltChunkDecoder<Element>)? = nil,
        onChunk: (Chunk) throws -> Bool
    ) throws {
        switch schedule.emission {
        case .perStep:
            try runPerStepEmission(maxSteps: maxSteps, onChunk: onChunk)
        case .chunked(let first, let max, let growth, let via):
            guard let decoder else { throw LoopError.missingDecoder(via) }
            let bounds = chunkOverride ?? (first, max)
            precondition(bounds.first >= 1 && bounds.max >= bounds.first,
                         "chunk schedule: need 1 <= first <= max")
            try runChunked(
                maxSteps: maxSteps, first: bounds.first, max: bounds.max,
                growth: growth, decoder: decoder, onChunk: onChunk)
        case .final(let via):
            guard let decoder else { throw LoopError.missingDecoder(via) }
            try runFinalEmission(maxSteps: maxSteps, decoder: decoder, onChunk: onChunk)
        }
    }

    // MARK: - The loop core

    /// Runs phases and feeds every emitted element to `consume` at the
    /// phase boundary where it appeared (so a cancelling consumer skips the
    /// rest of the step — the hand loops' `onFrame` semantics). `consume`
    /// returns false to stop.
    private func drive(
        maxSteps: Int,
        consume: (Element) throws -> Bool
    ) throws {
        var stopped = false

        func runPhase(
            _ entry: (phase: SmeltLoopSchedule.Phase, blocks: [any SmeltScheduledBlock<Element>]),
            step: Int
        ) throws {
            let phaseLabel = step < 0
                ? "setup:\(entry.phase.name)"
                : "per-step:\(entry.phase.name)"
            trace?.record(
                kind: "phase-begin",
                phase: phaseLabel,
                step: step,
                witness: "feedsNextStep=\(entry.phase.feedsNextStep ?? false)"
            )
            try scope {
                for block in entry.blocks {
                    try block.encode(phase: entry.phase.name, step: step)
                    trace?.record(
                        kind: "block-encode",
                        phase: phaseLabel,
                        block: block.blockName,
                        route: routesByBlock[block.blockName],
                        step: step,
                        witness: "ok"
                    )
                }
            }
            for block in entry.blocks {
                let signal = try block.finish(phase: entry.phase.name, step: step)
                var witness = "ok"
                if signal.emitted != nil { witness += ":emitted" }
                if signal.stop { witness += ":stop" }
                trace?.record(
                    kind: "block-finish",
                    phase: phaseLabel,
                    block: block.blockName,
                    route: routesByBlock[block.blockName],
                    step: step,
                    witness: witness
                )
                if signal.stop {
                    trace?.record(
                        kind: "stop",
                        phase: phaseLabel,
                        block: block.blockName,
                        route: routesByBlock[block.blockName],
                        step: step,
                        witness: "eos-token"
                    )
                    stopped = true
                }
                if let element = signal.emitted {
                    if try !consume(element) {
                        // A cancelling consumer on a schedule that never
                        // declared host-cancel is a contract violation —
                        // loud, never silently ignored or silently honored.
                        guard schedule.stop.contains(.hostCancel) else {
                            throw LoopError.undeclaredStop("host-cancel")
                        }
                        trace?.record(
                            kind: "stop",
                            phase: phaseLabel,
                            block: block.blockName,
                            route: routesByBlock[block.blockName],
                            step: step,
                            witness: "host-cancel"
                        )
                        stopped = true
                    }
                }
            }
            trace?.record(
                kind: "phase-end",
                phase: phaseLabel,
                step: step,
                witness: stopped ? "stopped" : "ok"
            )
        }

        for entry in setupPhases {
            try runPhase(entry, step: -1)
            if stopped { return }
        }
        var step = 0
        while !stopped && step < maxSteps {
            for entry in stepPhases {
                // A phase that only prepares the next step is dead work
                // when no next step will run.
                if entry.phase.feedsNextStep == true, step == maxSteps - 1 {
                    trace?.record(
                        kind: "phase-skip",
                        phase: "per-step:\(entry.phase.name)",
                        step: step,
                        witness: "feedsNextStep=true:last-step"
                    )
                    continue
                }
                try runPhase(entry, step: step)
                if stopped { break }
            }
            step += 1
        }
        if !stopped, step >= maxSteps {
            trace?.record(kind: "stop", step: step, witness: "max-steps")
        }
    }

    private func runChunked(
        maxSteps: Int, first: Int, max maxChunk: Int,
        growth: SmeltLoopSchedule.ChunkGrowth,
        decoder: any SmeltChunkDecoder<Element>,
        onChunk: (Chunk) throws -> Bool
    ) throws {
        var pending: [Element] = []
        var chunkTarget = first
        var cancelled = false

        try drive(maxSteps: maxSteps) { element in
            pending.append(element)
            guard pending.count >= chunkTarget else { return true }
            let offset = decoder.unitsDecoded
            let samples = try decoder.decodeChunk(pending)
            let count = pending.count
            pending.removeAll(keepingCapacity: true)
            if growth == .double {
                chunkTarget = Swift.min(chunkTarget * 2, maxChunk)
            }
            trace?.record(
                kind: "emission",
                witness: "chunk:offset=\(offset):count=\(count):final=false"
            )
            let keepGoing = try onChunk(Chunk(
                samples: samples, offset: offset, count: count, isFinal: false))
            cancelled = !keepGoing
            return keepGoing
        }
        guard !cancelled else { return }
        guard decoder.unitsDecoded > 0 || !pending.isEmpty else {
            throw LoopError.nothingGenerated
        }
        // Tail flush + end-of-stream marker.
        let offset = decoder.unitsDecoded
        let count = pending.count
        let samples = pending.isEmpty ? [] : try decoder.decodeChunk(pending)
        trace?.record(
            kind: "emission",
            witness: "chunk:offset=\(offset):count=\(count):final=true"
        )
        _ = try onChunk(Chunk(samples: samples, offset: offset, count: count, isFinal: true))
    }

    private func runPerStepEmission(
        maxSteps: Int, onChunk: (Chunk) throws -> Bool
    ) throws {
        // Per-step emission carries no samples (text-element loops decode
        // host-side); the chunk marks step boundaries for the caller.
        var emitted = 0
        try drive(maxSteps: maxSteps) { _ in
            emitted += 1
            trace?.record(
                kind: "emission",
                step: emitted - 1,
                witness: "per-step:offset=\(emitted - 1):count=1:final=false"
            )
            return try onChunk(Chunk(samples: [], offset: emitted - 1, count: 1, isFinal: false))
        }
        guard emitted > 0 else { throw LoopError.nothingGenerated }
        trace?.record(
            kind: "emission",
            step: emitted,
            witness: "per-step:offset=\(emitted):count=0:final=true"
        )
        _ = try onChunk(Chunk(samples: [], offset: emitted, count: 0, isFinal: true))
    }

    private func runFinalEmission(
        maxSteps: Int,
        decoder: any SmeltChunkDecoder<Element>,
        onChunk: (Chunk) throws -> Bool
    ) throws {
        var latest: Element?
        try drive(maxSteps: maxSteps) { element in
            latest = element
            return true
        }
        guard let latest else { throw LoopError.nothingGenerated }
        let offset = decoder.unitsDecoded
        let samples = try decoder.decodeChunk([latest])
        trace?.record(
            kind: "emission",
            witness: "final:offset=\(offset):count=1:final=true"
        )
        _ = try onChunk(Chunk(samples: samples, offset: offset, count: 1, isFinal: true))
    }
}
