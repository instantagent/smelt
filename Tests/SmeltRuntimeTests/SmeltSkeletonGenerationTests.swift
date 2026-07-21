import Foundation
import Testing

@testable import SmeltRuntime

@Suite("Smelt skeleton generation policy")
struct SmeltSkeletonGenerationTests {
    @Test("Vocabulary constants match the pinned checkpoint")
    func vocabulary() {
        #expect(SmeltSkeletonVocabulary.skinTokenRange.count == 32_768)
        #expect(SmeltSkeletonVocabulary.skinTokenRange.first == 267)
        #expect(SmeltSkeletonVocabulary.skinTokenRange.last == 33_034)
        #expect(SmeltSkeletonVocabulary.modelEOS == 33_035)
        #expect(SmeltSkeletonVocabulary.vocabularySize == 33_036)
    }

    @Test("Default generation budget uses all artifact context")
    func defaultGenerationBudget() {
        #expect(SmeltSkeletonGenerationConfiguration().maximumGeneratedTokens == nil)
    }

    @Test("Validated skeleton choices preserve completion context")
    func completionBudget() throws {
        let oneJoint = [257, 263, 10, 20, 30]
        let closingOnly = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: oneJoint,
            remainingTokenBudget: 12
        )
        #expect(closingOnly == [SmeltSkeletonVocabulary.skeletonEOS])

        let oneMoreJoint = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: oneJoint,
            remainingTokenBudget: 13
        )
        #expect(oneMoreJoint.contains(SmeltSkeletonVocabulary.skeletonEOS))
        #expect(oneMoreJoint.contains(0))
        #expect(!oneMoreJoint.contains(SmeltSkeletonVocabulary.spring))
        #expect(!oneMoreJoint.contains(SmeltSkeletonVocabulary.branch))

        let partRecord = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: oneJoint,
            remainingTokenBudget: 14
        )
        #expect(partRecord.contains(SmeltSkeletonVocabulary.spring))

        let branchRecord = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: oneJoint,
            remainingTokenBudget: 17
        )
        #expect(branchRecord.contains(SmeltSkeletonVocabulary.branch))
    }

    @Test("Skeleton FSM covers every authored transition")
    func skeletonTransitions() throws {
        #expect(
            try SmeltSkeletonTokenizer.allowedSkeletonTokens(
                after: [],
                mode: .sourceCompatible
            ) == [257]
        )
        let afterBOS = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: [257],
            mode: .sourceCompatible
        )
        #expect(afterBOS.prefix(4) == [263, 264, 265, 266])
        #expect(afterBOS.contains(260))
        #expect(afterBOS.contains(0))
        #expect(!afterBOS.contains(258))

        let afterClass = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: [257, 263],
            mode: .sourceCompatible
        )
        #expect(afterClass.contains(258))
        #expect(afterClass.contains(260))
        #expect(afterClass.contains(255))

        let oneJoint = [257, 263, 10, 20, 30]
        let afterJoint = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: oneJoint,
            mode: .sourceCompatible
        )
        #expect(afterJoint.contains(256))
        #expect(afterJoint.contains(258))
        #expect(afterJoint.contains(261))
        #expect(afterJoint.contains(42))

        let branchParent = oneJoint + [256, 40, 50, 60]
        let sourceAfterParent = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: branchParent,
            mode: .sourceCompatible
        )
        #expect(sourceAfterParent.contains(258))
        #expect(sourceAfterParent.contains(256))
        let validatedAfterParent = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: branchParent,
            mode: .validated
        )
        #expect(validatedAfterParent == Array(0..<256))
        let validatedComplete = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
            after: branchParent + [70, 80, 90],
            mode: .validated
        )
        #expect(validatedComplete.contains(258))
        #expect(validatedComplete.contains(256))
    }

    @Test("Detokenization preserves branch hierarchy and coordinate centers")
    func detokenize() throws {
        let tokens = [
            257, 263,
            128, 128, 128,
            256,
            128, 128, 128,
            192, 128, 128,
            258,
        ]
        let skeleton = try SmeltSkeletonTokenizer.detokenize(tokens)
        #expect(skeleton.joints.count == 2)
        #expect(skeleton.joints[0].parent == -1)
        #expect(skeleton.joints[1].parent == 0)
        #expect(skeleton.joints[0].position.x.bitPattern == Float(0.00390625).bitPattern)
        #expect(skeleton.joints[1].position.x.bitPattern == Float(0.50390625).bitPattern)
        #expect(try SmeltSkeletonTokenizer.sourceCompatibleBoneCount(tokens) == 2)
    }

    @Test("Validated skin phase emits exactly four codes per joint")
    func validatedSkinPhase() throws {
        let skeleton = [
            257, 263,
            128, 128, 128,
            128, 128, 192,
            258,
        ]
        let skin = Array(repeating: 267, count: 8)
        #expect(
            try SmeltSkeletonGenerationPolicy.allowedNextTokens(
                sequence: skeleton,
                mode: .validated
            ) == Array(267..<33_035)
        )
        #expect(
            try SmeltSkeletonGenerationPolicy.allowedNextTokens(
                sequence: skeleton + skin,
                mode: .validated
            ) == [33_035]
        )
        #expect(
            try SmeltSkeletonGenerationPolicy.allowedNextTokens(
                sequence: skeleton + skin + [33_035],
                mode: .validated
            ).isEmpty
        )
    }

    @Test("Source-compatible mode pins upstream mask and count quirks")
    func sourceCompatibleSkinPhase() throws {
        let skeleton = [257, 263, 128, 128, 128, 258]
        let initial = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: skeleton,
            mode: .sourceCompatible
        )
        #expect(initial.first == 258)
        #expect(initial.last == 33_034)
        #expect(initial.contains(259))
        #expect(initial.contains(266))
        #expect(
            try SmeltSkeletonGenerationPolicy.allowedNextTokens(
                sequence: skeleton + [267, 268, 269],
                mode: .sourceCompatible
            ) == [33_035]
        )
    }

    @Test("Sampling composes repetition, top-k, top-p, and grammar")
    func sampling() throws {
        var logits = [Float](
            repeating: -.infinity,
            count: SmeltSkeletonVocabulary.vocabularySize
        )
        logits[267] = 5
        logits[268] = 4.9
        logits[269] = 4.8
        let topOne = try SmeltSkeletonGenerationPolicy.sample(
            logits: logits,
            history: [],
            allowedTokens: [267, 268, 269],
            repetitionPenalty: 2,
            temperature: 1.5,
            topK: 1,
            topP: 0.95,
            uniform: 0.99
        )
        #expect(topOne.token == 267)

        let penalized = try SmeltSkeletonGenerationPolicy.sample(
            logits: logits,
            history: [267],
            allowedTokens: [267, 268, 269],
            repetitionPenalty: 2,
            temperature: 1.5,
            topK: 2,
            topP: 0.95,
            uniform: 0
        )
        #expect(penalized.token == 268)

        let nucleus = try SmeltSkeletonGenerationPolicy.sample(
            logits: logits,
            history: [],
            allowedTokens: [267, 268, 269],
            repetitionPenalty: 1,
            temperature: 0.01,
            topK: 3,
            topP: 0.5,
            uniform: 0.99
        )
        #expect(nucleus.token == 267)

        let distribution = try SmeltSkeletonGenerationPolicy.filteredDistribution(
            logits: logits,
            history: [],
            allowedTokens: [267, 268, 269],
            repetitionPenalty: 1,
            temperature: 1.5,
            topK: 2,
            topP: 1
        )
        #expect(distribution.map(\.token) == [267, 268])
        #expect(
            abs(distribution.map { exp($0.logProbability) }.reduce(0, +) - 1)
                < 1e-12
        )
    }

    @Test("Real Qwen loop completes a supplied one-joint skeleton")
    func realQwenLoop() throws {
        guard let package = ProcessInfo.processInfo.environment[
            "SMELT_SKINNING_PACKAGE"
        ] else {
            return
        }
        let languageModel = try SmeltSkeletonLanguageRuntime(packagePath: package)
        let generator = SmeltSkeletonGenerator(languageModel: languageModel)
        let meshEmbeddings = [Float](repeating: 0, count: 512 * 896)
        let start = [257, 263, 128, 128, 128, 258]
        let result = try generator.generate(
            meshEmbeddings: meshEmbeddings,
            startTokenIDs: start,
            configuration: .init(
                policyMode: .validated,
                decodeMode: .greedy,
                maximumGeneratedTokens: 5
            )
        )
        #expect(Array(result.tokenIDs.prefix(start.count)) == start)
        #expect(result.generatedTokenIDs.count == 5)
        #expect(result.generatedTokenIDs.last == 33_035)
        #expect(result.skeleton.joints.count == 1)
        #expect(result.skinCodeIndices.count == 1)
        #expect(result.skinCodeIndices[0].count == 4)
        #expect(result.skinCodeIndices[0].allSatisfy { (0..<32_768).contains($0) })
    }

    @Test("Real Qwen beam sampler snapshots and restores per-beam KV state")
    func realQwenBeamLoop() throws {
        guard let package = ProcessInfo.processInfo.environment[
            "SMELT_SKINNING_PACKAGE"
        ] else {
            return
        }
        let languageModel = try SmeltSkeletonLanguageRuntime(packagePath: package)
        let generator = SmeltSkeletonGenerator(languageModel: languageModel)
        let meshEmbeddings = [Float](repeating: 0, count: 512 * 896)
        let start = [257, 263, 128, 128, 128, 258]
        let result = try generator.generateTokenSequence(
            meshEmbeddings: meshEmbeddings,
            startTokenIDs: start,
            configuration: .init(
                policyMode: .sourceCompatible,
                decodeMode: .beamSampled(seed: 17, width: 2),
                maximumGeneratedTokens: 4,
                topK: 2
            )
        )
        #expect(Array(result.tokenIDs.prefix(start.count)) == start)
        #expect(result.generatedTokenIDs.count == 4)
        #expect(result.generatedTokenIDs.last == 33_035)
    }
}
