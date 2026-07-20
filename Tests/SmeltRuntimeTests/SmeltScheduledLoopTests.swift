import Foundation
import Testing
@testable import SmeltRuntime
import SmeltSchema

// SmeltScheduledLoop — the generic orchestrator, exercised with fake blocks
// (no GPU): phase/scope ordering, the pipelined stop semantics, the
// feedsNextStep skip, chunk growth, cancellation, and the tail flush.

private final class FakeBlock: SmeltScheduledBlock {
    typealias Element = Int
    let blockName: String
    let log: Log
    /// step → behavior overrides
    var stopAtStep: Int?
    var emitsPerStep: Bool

    final class Log {
        var entries: [String] = []
    }

    init(_ name: String, log: Log, emitsPerStep: Bool = false) {
        blockName = name
        self.log = log
        self.emitsPerStep = emitsPerStep
    }

    func encode(phase: String, step: Int) throws {
        log.entries.append("enc \(blockName) \(phase)@\(step)")
    }

    func finish(phase: String, step: Int) throws -> SmeltBlockSignal<Int> {
        log.entries.append("fin \(blockName) \(phase)@\(step)")
        if let stopAt = stopAtStep, step == stopAt {
            return .init(stop: true)
        }
        return emitsPerStep && step >= 0 ? .init(emitted: step) : .none
    }
}

private final class FakeDecoder: SmeltChunkDecoder {
    typealias Element = Int
    var decoded: [[Int]] = []
    var unitsDecoded = 0

    func decodeChunk(_ items: [Int]) throws -> [Float] {
        decoded.append(items)
        unitsDecoded += items.count
        return items.map(Float.init)
    }
}

private func ttsLikeSchedule() -> SmeltLoopSchedule {
    SmeltLoopSchedule(
        setup: [.init(name: "prefill", blocks: ["trunk", "head"])],
        perStep: [
            .init(name: "work", blocks: ["worker"]),
            .init(name: "advance", blocks: ["trunk", "head"], feedsNextStep: true),
        ],
        emission: .chunked(first: 1, max: 4, growth: .double, via: "decoder"),
        stop: [.eosToken, .maxSteps, .hostCancel]
    )
}

@Suite struct SmeltScheduledLoopTests {

    private func makeLoop(
        schedule: SmeltLoopSchedule, log: FakeBlock.Log,
        trace: SmeltRuntimeTraceRecorder? = nil,
        routesByBlock: [String: String] = [:],
        configure: (FakeBlock, FakeBlock, FakeBlock) -> Void = { _, _, _ in }
    ) throws -> SmeltScheduledLoop<Int> {
        let trunk = FakeBlock("trunk", log: log)
        let head = FakeBlock("head", log: log)
        let worker = FakeBlock("worker", log: log, emitsPerStep: true)
        configure(trunk, head, worker)
        return try SmeltScheduledLoop<Int>(
            schedule: schedule,
            blocks: [trunk, head, worker],
            scope: { body in
                log.entries.append("scope{")
                try body()
                log.entries.append("}scope")
            },
            trace: trace,
            routesByBlock: routesByBlock
        )
    }

    @Test func phaseOrderAndScopeDiscipline() throws {
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: ttsLikeSchedule(), log: log)
        var chunks: [SmeltScheduledLoop<Int>.Chunk] = []
        try loop.run(maxSteps: 2, decoder: decoder) { chunks.append($0); return true }

        // Setup: one scope wrapping BOTH phase blocks' encodes, finishes after.
        #expect(Array(log.entries.prefix(5)) == [
            "scope{", "enc trunk prefill@-1", "enc head prefill@-1", "}scope",
            "fin trunk prefill@-1",
        ])
        // The final step (step 1 of 2) skips the feedsNextStep advance phase.
        #expect(!log.entries.contains("enc trunk advance@1"))
        #expect(log.entries.contains("enc trunk advance@0"))
        // Two emitted steps → chunk growth 1 then 2-cap... first=1 emits
        // immediately; second buffered then tail-flushed as the final chunk.
        #expect(decoder.decoded == [[0], [1]])
        #expect(chunks.map(\.isFinal) == [false, true])
        #expect(chunks.map(\.offset) == [0, 1])
    }

    @Test func runtimeTraceRecordsActualPhaseBlockAndEmissionEvents() throws {
        let log = FakeBlock.Log()
        let trace = SmeltRuntimeTraceRecorder()
        let decoder = FakeDecoder()
        let loop = try makeLoop(
            schedule: ttsLikeSchedule(),
            log: log,
            trace: trace,
            routesByBlock: ["worker": "native:runtime-test"]
        )

        try loop.run(maxSteps: 2, decoder: decoder) { _ in true }

        #expect(trace.events.first == SmeltTraceEvent(
            kind: "phase-begin",
            index: 0,
            phase: "setup:prefill",
            step: -1,
            witness: "feedsNextStep=false"
        ))
        #expect(trace.events.contains {
            $0.kind == "block-encode" && $0.phase == "per-step:work"
                && $0.block == "worker" && $0.step == 0
                && $0.route == "native:runtime-test"
        })
        #expect(trace.events.contains {
            $0.kind == "block-finish" && $0.phase == "per-step:work"
                && $0.block == "worker" && $0.step == 0
                && $0.witness == "ok:emitted"
        })
        #expect(trace.events.contains {
            $0.kind == "phase-skip" && $0.phase == "per-step:advance"
                && $0.step == 1
        })
        #expect(trace.events.contains {
            $0.kind == "emission" && $0.witness == "chunk:offset=1:count=1:final=true"
        })
        #expect(trace.events.last?.kind == "emission")
        #expect(trace.events.map(\.index) == Array(trace.events.indices))
    }

    @Test func stopFromSetupSkipsLoop() throws {
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: ttsLikeSchedule(), log: log) { _, head, _ in
            head.stopAtStep = -1   // EOS detected at the setup boundary
        }
        #expect(throws: SmeltScheduledLoop<Int>.LoopError.self) {
            try loop.run(maxSteps: 4, decoder: decoder) { _ in true }
        }
        #expect(!log.entries.contains("enc worker work@0"))
    }

    @Test func stopAfterAdvanceSkipsNextStepsWork() throws {
        // EOS lands in the head's finish of advance@0 (the pipelined cb0
        // shape) — step 1's work phase must never run, but step 0's emission
        // still flushes.
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: ttsLikeSchedule(), log: log) { _, head, _ in
            head.stopAtStep = 0
        }
        var finals = 0
        try loop.run(maxSteps: 8, decoder: decoder) { if $0.isFinal { finals += 1 }; return true }
        #expect(log.entries.contains("enc worker work@0"))
        #expect(!log.entries.contains("enc worker work@1"))
        #expect(finals == 1)
        #expect(decoder.decoded == [[0]])
    }

    @Test func cancellationStopsBeforeAdvance() throws {
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: ttsLikeSchedule(), log: log)
        try loop.run(maxSteps: 8, decoder: decoder) { _ in false }   // barge-in at first chunk
        #expect(log.entries.contains("enc worker work@0"))
        // Cancelled at the emission boundary: the same step's advance never encodes.
        #expect(!log.entries.contains("enc trunk advance@0"))
    }

    @Test func chunkGrowthDoublesToCap() throws {
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: ttsLikeSchedule(), log: log)
        try loop.run(maxSteps: 12, decoder: decoder) { _ in true }
        // 1 → 2 → 4 → 4 → tail(1): 1+2+4+4 = 11 emitted in non-final chunks.
        #expect(decoder.decoded.map(\.count) == [1, 2, 4, 4, 1])
    }

    @Test func fixedGrowthKeepsChunkSize() throws {
        var schedule = ttsLikeSchedule()
        schedule = SmeltLoopSchedule(
            setup: schedule.setup, perStep: schedule.perStep,
            emission: .chunked(first: 2, max: 2, growth: .fixed, via: "decoder"),
            stop: schedule.stop)
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: schedule, log: log)
        try loop.run(maxSteps: 6, decoder: decoder) { _ in true }
        #expect(decoder.decoded.map(\.count) == [2, 2, 2])
    }

    @Test func finalEmissionDecodesOnlyTheLastElement() throws {
        let schedule = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "denoise", blocks: ["worker"])],
            emission: .final(via: "decoder"),
            stop: [.maxSteps, .hostCancel]
        )
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: schedule, log: log)
        var chunks: [SmeltScheduledLoop<Int>.Chunk] = []
        try loop.run(maxSteps: 3, decoder: decoder) { chunks.append($0); return true }

        #expect(decoder.decoded == [[2]])
        #expect(chunks.map(\.samples) == [[2]])
        #expect(chunks.map(\.isFinal) == [true])
        #expect(log.entries.contains("enc worker denoise@0"))
        #expect(log.entries.contains("enc worker denoise@2"))
    }

    @Test func undeclaredHostCancelIsLoud() throws {
        // A schedule without host-cancel meeting a cancelling consumer is a
        // contract violation — thrown, never silently ignored or honored.
        var schedule = ttsLikeSchedule()
        schedule = SmeltLoopSchedule(
            setup: schedule.setup, perStep: schedule.perStep,
            emission: schedule.emission, stop: [.eosToken, .maxSteps])
        let log = FakeBlock.Log()
        let decoder = FakeDecoder()
        let loop = try makeLoop(schedule: schedule, log: log)
        #expect(throws: SmeltScheduledLoop<Int>.LoopError.self) {
            try loop.run(maxSteps: 8, decoder: decoder) { _ in false }
        }
    }

    @Test func undeclaredMaxStepsRejectedAtInit() {
        let schedule = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "x", blocks: ["worker"])],
            emission: .perStep,
            stop: [.eosToken])
        let log = FakeBlock.Log()
        #expect(throws: SmeltScheduledLoop<Int>.LoopError.self) {
            _ = try SmeltScheduledLoop<Int>(
                schedule: schedule,
                blocks: [FakeBlock("worker", log: log)],
                scope: { try $0() })
        }
    }

    @Test func unknownBlockNameThrowsAtInit() {
        let schedule = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "x", blocks: ["nobody"])],
            emission: .perStep,
            stop: [.maxSteps])
        #expect(throws: SmeltScheduledLoop<Int>.LoopError.self) {
            _ = try SmeltScheduledLoop<Int>(schedule: schedule, blocks: [], scope: { try $0() })
        }
    }
}
