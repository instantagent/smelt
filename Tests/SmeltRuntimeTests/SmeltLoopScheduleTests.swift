import Foundation
import Testing
@testable import SmeltSchema

// SmeltLoopSchedule — the drive loop declared as data (B2.1 of
// docs/block-spec-plan.md). Phases are command-buffer scopes; the canonical
// schedules must describe what the hand loops measurably do.

@Suite struct SmeltLoopScheduleTests {

    @Test func canonicalSchedulesValidateAgainstTheirGraphs() throws {
        try SmeltLoopSchedule.tokenFeedbackText.validate(against: .tokenFeedbackText)
        try SmeltLoopSchedule.qwen3TTS.validate(against: .qwen3TTS)
    }

    @Test func ttsScheduleDeclaresTheSelectiveBatchingShape() {
        // The TTFA-critical shape: two command buffers per frame (MTP
        // fan-out | feedback+trunk+cb0), zero-buffer 1/1 chunk default.
        let schedule = SmeltLoopSchedule.qwen3TTS
        #expect(schedule.perStep.map(\.name) == ["mtp", "advance"])
        #expect(schedule.perStep[1].blocks == ["talker", "codec-head"])
        #expect(schedule.emission == .chunked(first: 1, max: 1, growth: .double, via: "codec-decoder"))
    }

    @Test func roundTripsThroughJSON() throws {
        for schedule in [SmeltLoopSchedule.tokenFeedbackText, .qwen3TTS] {
            let data = try JSONEncoder().encode(schedule)
            let back = try JSONDecoder().decode(SmeltLoopSchedule.self, from: data)
            #expect(back == schedule)
        }
    }

    @Test func unknownBlockNamesRejected() {
        let schedule = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "decode", blocks: ["no-such-block"])],
            emission: .perStep,
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try schedule.validate(against: .tokenFeedbackText)
        }
    }

    @Test func chunkedEmissionBoundsAndTargetChecked() {
        let badBounds = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "step", blocks: ["talker"])],
            emission: .chunked(first: 4, max: 2, growth: .double, via: "codec-decoder"),
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try badBounds.validate(against: .qwen3TTS)
        }
        let badTarget = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "step", blocks: ["talker"])],
            emission: .chunked(first: 1, max: 1, growth: .double, via: "no-such-block"),
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try badTarget.validate(against: .qwen3TTS)
        }
    }

    @Test func emptyPerStepOrStopsRejected() {
        let noSteps = SmeltLoopSchedule(
            setup: [], perStep: [], emission: .perStep, stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try noSteps.validate(against: .tokenFeedbackText)
        }
        let noStops = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "decode", blocks: ["trunk"])],
            emission: .perStep,
            stop: []
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try noStops.validate(against: .tokenFeedbackText)
        }
    }

    @Test func fixedGrowthRequiresEqualBounds() {
        let bad = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "step", blocks: ["talker"])],
            emission: .chunked(first: 1, max: 4, growth: .fixed, via: "codec-decoder"),
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try bad.validate(against: .qwen3TTS)
        }
    }

    @Test func emissionBlockMustProduceGraphOutput() {
        // Emitting "via" a block that doesn't produce the package's output
        // would be a lying schedule — the codec head emits tokens, not audio.
        let bad = SmeltLoopSchedule(
            setup: [],
            perStep: [.init(name: "step", blocks: ["talker"])],
            emission: .chunked(first: 1, max: 1, growth: .double, via: "codec-head"),
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try bad.validate(against: .qwen3TTS)
        }
    }

    @Test func duplicatePhaseNamesRejected() {
        let schedule = SmeltLoopSchedule(
            setup: [.init(name: "x", blocks: ["trunk"])],
            perStep: [.init(name: "x", blocks: ["trunk"])],
            emission: .perStep,
            stop: [.eosToken]
        )
        #expect(throws: SmeltLoopSchedule.ScheduleError.self) {
            try schedule.validate(against: .tokenFeedbackText)
        }
    }
}
