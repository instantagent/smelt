// Baked-prefix equivalence gates: generation through a baked prompt prefix
// (restore + suffix prefill) must produce the same tokens as a full prefill.
// Gated on a locally built canonical package; skips cleanly when absent.

import Darwin
import Foundation
import Testing
import SmeltSchema
@testable import SmeltRuntime

private let qwen08bPackage =
    "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"

private let bakedPrefixTempRoot: URL = {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("smelt-baked-prefix-tests-\(getpid())", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    atexit_b {
        try? FileManager.default.removeItem(at: root)
    }
    return root
}()

/// Clone a package directory via hardlinks (weights.bin stays shared on disk)
/// so the test can bake into it without mutating the canonical artifact.
func hardlinkClone(of packagePath: String) throws -> String {
    let fm = FileManager.default
    let clone = bakedPrefixTempRoot
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try fm.createDirectory(at: clone, withIntermediateDirectories: true)
    for entry in try fm.contentsOfDirectory(atPath: packagePath) {
        // Start from an unbaked package even if the source has a baked prefix.
        if entry.hasPrefix("baked_prefix") { continue }
        let src = "\(packagePath)/\(entry)"
        let dst = clone.appendingPathComponent(entry).path
        var isDir: ObjCBool = false
        fm.fileExists(atPath: src, isDirectory: &isDir)
        if isDir.boolValue {
            try fm.copyItem(atPath: src, toPath: dst)
        } else {
            try fm.linkItem(atPath: src, toPath: dst)
        }
    }

    let manifestURL = clone.appendingPathComponent("manifest.json")
    let manifestData = try Data(contentsOf: manifestURL)
    guard var manifest = try JSONSerialization.jsonObject(with: manifestData)
        as? [String: Any],
        var validation = manifest["validation"] as? [String: Any],
        let gate = validation["performance_gate"] as? String
    else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let profile = SmeltPackagePerformanceProfiles.profile(
        for: gate,
        modelName: manifest["modelName"] as? String
    )
    validation["performance_profile"] = try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(profile)
    )
    manifest["validation"] = validation
    try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        .write(to: manifestURL, options: .atomic)
    return clone.path
}

private func generateTokens(
    _ model: SmeltModel, ids: [Int32], maxTokens: Int
) throws -> [Int32] {
    var count = 0
    let result = try model.generate(tokenIds: ids) { _ in
        count += 1
        return count < maxTokens
    }
    return result.tokens
}

private func writeBakedContinuationTokenizerFixture() throws -> String {
    let vocab: [String: Int] = [
        "▁": 0,
        "\n": 1,
        "a": 2,
        "b": 3,
        "d": 4,
        "e": 5,
        "i": 6,
        "m": 7,
        "n": 8,
        "r": 9,
        "s": 10,
        "t": 11,
        "u": 12,
        "y": 13,
    ]
    let fixture: [String: Any] = [
        "model": [
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        ] as [String: Any],
        "added_tokens": [
            ["content": "<|im_start|>", "id": 100, "special": true],
            ["content": "<|im_end|>", "id": 101, "special": true],
            ["content": "<think>", "id": 102, "special": true],
            ["content": "</think>", "id": 103, "special": true],
        ] as [[String: Any]],
    ]
    let data = try JSONSerialization.data(withJSONObject: fixture)
    let path = bakedPrefixTempRoot
        .appendingPathComponent("qwen-continuation-tokenizer-\(UUID().uuidString).json")
        .path
    try data.write(to: URL(fileURLWithPath: path))
    return path
}

@Suite struct BakedPrefixTests {
    @Test func namedPreparedPromptsRoundTripAndMatchByContract() throws {
        let pkgURL = bakedPrefixTempRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: pkgURL, withIntermediateDirectories: true
        )
        func snapshot(_ ids: [Int32]) -> SmeltPromptSnapshot {
            SmeltPromptSnapshot(
                promptLength: ids.count,
                nextToken: ids.last ?? 0,
                byteCount: 0,
                capturedLength: ids.count,
                replayTokenIds: [],
                convStates: [], recStates: [], keyCaches: [], valueCaches: []
            )
        }
        let defaultIDs: [Int32] = [1, 2]
        let interactiveIDs: [Int32] = [1, 2, 3]
        _ = try SmeltPreparedPromptSet.write(
            packagePath: pkgURL.path,
            state: SmeltPreparedPromptState(
                id: "run/default",
                tokenIds: defaultIDs,
                snapshot: snapshot(defaultIDs)
            )
        )
        _ = try SmeltPreparedPromptSet.write(
            packagePath: pkgURL.path,
            state: SmeltPreparedPromptState(
                id: "interactive/pi-v1",
                tokenIds: interactiveIDs,
                sampling: SmeltPreparedPromptSampling(
                    temperature: 0.7, topK: 20, topP: 0.95
                ),
                snapshot: snapshot(interactiveIDs)
            )
        )

        let loaded = try #require(try SmeltPreparedPromptSet.load(
            packagePath: pkgURL.path
        ))
        #expect(loaded.states.map(\.id) == [
            "interactive/pi-v1", "run/default",
        ])
        #expect(loaded.longestMatch(tokenIds: [1, 2, 3, 4])?.id
            == "interactive/pi-v1")
        #expect(loaded.longestMatch(
            tokenIds: [1, 2, 3, 4], contract: "run/default"
        )?.id == "run/default")
        #expect(loaded.longestMatch(
            tokenIds: [1, 9], contract: "interactive/pi-v1"
        ) == nil)
        #expect(loaded.longestMatch(
            tokenIds: [1, 2, 3, 4], contract: "interactive/pi-v1"
        )?.sampling == SmeltPreparedPromptSampling(
            temperature: 0.7, topK: 20, topP: 0.95
        ))
        #expect(try SmeltPreparedPromptSet.declaredFiles(
            packagePath: pkgURL.path
        ).count == 3)
    }

    @Test func namedPreparedPromptsRejectInvalidPackagedSamplingPolicy() throws {
        let pkgURL = bakedPrefixTempRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: pkgURL, withIntermediateDirectories: true
        )
        let metadata = #"{"version":1,"entries":[{"id":"interactive/test","token_ids":[1],"snapshot_file":"prepared_prompt_deadbeef.snapshot","sampling":{"temperature":0.7,"top_k":20,"top_p":1.5}}]}"#
        try Data(metadata.utf8).write(
            to: pkgURL.appendingPathComponent(SmeltPreparedPromptSet.fileName)
        )

        #expect(throws: SmeltPreparedPromptError.self) {
            _ = try SmeltPreparedPromptSet.load(packagePath: pkgURL.path)
        }
    }

    @Test func namedPreparedPromptsRequireExactAutomaticCaptureReplayTail() throws {
        let pkgURL = bakedPrefixTempRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: pkgURL, withIntermediateDirectories: true
        )
        let ids: [Int32] = [11, 12, 13, 14]
        let valid = SmeltPromptSnapshot(
            promptLength: ids.count,
            nextToken: 0,
            byteCount: 0,
            capturedLength: 2,
            replayTokenIds: [13, 14],
            convStates: [], recStates: [], keyCaches: [], valueCaches: []
        )
        _ = try SmeltPreparedPromptSet.write(
            packagePath: pkgURL.path,
            state: SmeltPreparedPromptState(
                id: "interactive/pi-v1",
                tokenIds: ids,
                snapshot: valid
            )
        )
        let loaded = try #require(try SmeltPreparedPromptSet.load(
            packagePath: pkgURL.path
        ))
        #expect(loaded.states.first?.snapshot.capturedLength == 2)
        #expect(loaded.states.first?.snapshot.replayTokenIds == [13, 14])

        let invalid = SmeltPromptSnapshot(
            promptLength: ids.count,
            nextToken: 0,
            byteCount: 0,
            capturedLength: 2,
            replayTokenIds: [13, 99],
            convStates: [], recStates: [], keyCaches: [], valueCaches: []
        )
        #expect(throws: SmeltPreparedPromptError.self) {
            _ = try SmeltPreparedPromptSet.write(
                packagePath: pkgURL.path,
                state: SmeltPreparedPromptState(
                    id: "interactive/bad",
                    tokenIds: ids,
                    snapshot: invalid
                )
            )
        }
    }

    @Test func bakedPrefixContinuationRoundTripsAsOptionalMetadata() throws {
        let pkgURL = bakedPrefixTempRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: pkgURL,
            withIntermediateDirectories: true
        )
        let prefixIds: [Int32] = [1, 2, 3]
        let continuation = SmeltBakedPromptContinuation(
            template: "chatml",
            thinkingPolicy: .disabled,
            promptSuffixTokenIds: [4, 5, 6]
        )
        let snapshot = SmeltPromptSnapshot(
            promptLength: prefixIds.count,
            nextToken: 3,
            byteCount: 0,
            capturedLength: prefixIds.count,
            replayTokenIds: [],
            convStates: [],
            recStates: [],
            keyCaches: [],
            valueCaches: []
        )

        _ = try SmeltBakedPrefix.write(
            packagePath: pkgURL.path,
            tokenIds: prefixIds,
            snapshot: snapshot,
            continuation: continuation
        )

        let loaded = try SmeltBakedPrefix.loadStrict(packagePath: pkgURL.path)
        #expect(loaded.tokenIds == prefixIds)
        #expect(loaded.continuation == continuation)
        #expect(loaded.continuation?.matches(
            template: "chatml",
            thinkingPolicy: .disabled
        ) == true)
        #expect(loaded.continuation?.matches(
            template: "chatml",
            thinkingPolicy: .enabled
        ) == false)
    }

    @Test func bakedContinuationReconstructsUnbakedQwenChatTokens() throws {
        let tokenizer = try SmeltTokenizer(jsonPath: writeBakedContinuationTokenizerFixture())
        let imStart: Int32 = 100
        let imEnd: Int32 = 101
        let think: Int32 = 102
        let thinkEnd: Int32 = 103

        var systemPrefix: [Int32] = [imStart]
        systemPrefix += tokenizer.encode("system")
        systemPrefix += tokenizer.encode("\n")
        systemPrefix += tokenizer.encode("be")
        systemPrefix += [imEnd]
        systemPrefix += tokenizer.encode("\n")

        let plan = try #require(try SmeltBakedPromptContinuationBuilder.systemPromptPlan(
            tokenizer: tokenizer,
            template: "chatml",
            thinkingPolicy: .disabled
        ))
        let bakedPrefixTokenIds = systemPrefix + plan.prefixTailTokenIds
        let rebuilt = try #require(SmeltBakedPromptContinuationBuilder.inputIds(
            prompt: "red",
            tokenizer: tokenizer,
            bakedPrefixTokenIds: bakedPrefixTokenIds,
            continuation: plan.continuation,
            template: "chatml",
            thinkingPolicy: .disabled
        ))

        var expected = systemPrefix
        expected += [imStart]
        expected += tokenizer.encode("user")
        expected += tokenizer.encode("\n")
        expected += tokenizer.encode("red")
        expected += [imEnd]
        expected += tokenizer.encode("\n")
        expected += [imStart]
        expected += tokenizer.encode("assistant")
        expected += tokenizer.encode("\n")
        expected += [think]
        expected += tokenizer.encode("\n\n")
        expected += [thinkEnd]
        expected += tokenizer.encode("\n\n")

        #expect(rebuilt == expected)
        #expect(SmeltBakedPromptContinuationBuilder.inputIds(
            prompt: "red",
            tokenizer: tokenizer,
            bakedPrefixTokenIds: bakedPrefixTokenIds,
            continuation: plan.continuation,
            template: "chatml",
            thinkingPolicy: .enabled
        ) == nil)
    }

    @Test func staleBakedContinuationFallsBackToFullPrompt() throws {
        let tokenizer = try SmeltTokenizer(jsonPath: writeBakedContinuationTokenizerFixture())
        let staleContinuation = SmeltBakedPromptContinuation(
            template: "chatml",
            thinkingPolicy: .disabled,
            promptSuffixTokenIds: [101]
        )
        let unbaked = [Int32(11), 12, 13]
        let baked = [Int32(100), 12, 10, 5, 9]

        let mismatched = SmeltBakedPromptContinuationBuilder.inputIds(
            prompt: "red",
            tokenizer: tokenizer,
            bakedPrefixTokenIds: baked,
            continuation: staleContinuation,
            template: "chatml",
            thinkingPolicy: .enabled,
            unbakedInputIds: unbaked
        )
        #expect(mismatched == unbaked)

        let legacy = SmeltBakedPromptContinuationBuilder.inputIds(
            prompt: "red",
            tokenizer: tokenizer,
            bakedPrefixTokenIds: baked,
            continuation: nil,
            template: "chatml",
            thinkingPolicy: .enabled,
            unbakedInputIds: unbaked
        )
        #expect(legacy == baked + unbaked)
    }

    @Test func bakedGenerationMatchesFullPrefill() throws {
        guard FileManager.default.fileExists(atPath: qwen08bPackage) else { return }
        let pkg = try hardlinkClone(of: qwen08bPackage)
        let tokenizer = try SmeltTokenizer(path: "\(pkg)/tokenizer.json")

        let prefixIds = tokenizer.encodeWithSpecials(
            "<|im_start|>system\nYou answer in exactly one short sentence."
                + "<|im_end|>\n"
        )
        let suffixIds = tokenizer.encodeWithSpecials(
            "<|im_start|>user\nName one primary color.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        let fullIds = prefixIds + suffixIds

        let freshModel = try SmeltModel(package: pkg)
        #expect(freshModel.bakedPrefixTokenIds == nil)
        let baseline = try generateTokens(freshModel, ids: fullIds, maxTokens: 16)
        #expect(!baseline.isEmpty)

        try freshModel.bakePromptPrefix(tokenIds: prefixIds)

        let bakedModel = try SmeltModel(package: pkg)
        #expect(bakedModel.bakedPrefixTokenIds == prefixIds)
        let baked = try generateTokens(bakedModel, ids: fullIds, maxTokens: 16)
        #expect(baked == baseline)

        // A request that does NOT start with the baked prefix must take the
        // full-prefill path and still match a fresh model's output.
        let other = suffixIds
        let bakedOther = try generateTokens(bakedModel, ids: other, maxTokens: 16)
        let freshOther = try generateTokens(freshModel, ids: other, maxTokens: 16)
        #expect(bakedOther == freshOther)
    }

    @Test func corruptBakedPrefixFallsBackToFullPrefill() throws {
        guard FileManager.default.fileExists(atPath: qwen08bPackage) else { return }
        let pkg = try hardlinkClone(of: qwen08bPackage)
        let tokenizer = try SmeltTokenizer(path: "\(pkg)/tokenizer.json")
        let prefixIds = tokenizer.encodeWithSpecials(
            "<|im_start|>system\nBe terse.<|im_end|>\n"
        )
        let fullIds = prefixIds + tokenizer.encodeWithSpecials(
            "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n"
        )

        let model = try SmeltModel(package: pkg)
        let baseline = try generateTokens(model, ids: fullIds, maxTokens: 8)
        try model.bakePromptPrefix(tokenIds: prefixIds)
        try Data("garbage".utf8).write(
            to: URL(fileURLWithPath: "\(pkg)/\(SmeltBakedPrefix.snapshotFileName)")
        )

        let corrupted = try SmeltModel(package: pkg)
        #expect(corrupted.bakedPrefixTokenIds == nil)
        let tokens = try generateTokens(corrupted, ids: fullIds, maxTokens: 8)
        #expect(tokens == baseline)
    }
}
