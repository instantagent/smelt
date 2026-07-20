import SmeltCompiler
import SmeltModels
import SmeltSchema
import Foundation
import Testing

@Suite("CAM static cost model")
struct CAMStaticCostModelTests {
    private let memoryLimit = UInt64(96 * 1_073_741_824)

    @Test("Typed topology derives Inkling resident, active, and KV bills")
    func inklingCostBill() throws {
        let module = try #require(SmeltModels.definition(id: "inkling"))
        let report = try SmeltCAMStaticCostModel.report(
            module: module,
            scenario: SmeltCAMStaticCostScenario(
                storage: .bf16,
                contextLength: 65_536,
                exactCheckpointBytes: 1_904_604_285_204,
                memoryLimitBytes: memoryLimit,
                sustainedMemoryBandwidthBytesPerSecond: 400_000_000_000
            )
        )

        #expect(report.declaredResidentParameters == 975_000_000_000)
        #expect(report.declaredActiveParameters == 41_000_000_000)
        #expect(report.derivedResidentParameters == 947_021_970_496)
        #expect(report.derivedActiveParameters == 41_052_306_496)
        #expect(report.residentWeightBytes == 1_904_604_285_204)
        #expect(report.kvStateBytes == 3_183_476_736)
        #expect(report.shortConvolutionStateBytes == 6_352_896)
        #expect(report.totalPersistentBytes == 1_907_794_114_836)
        #expect(report.fitsMemoryLimit == false)
        #expect(report.oneTokenKVReadBytes == report.kvStateBytes)
        #expect(report.oneTokenKVWriteBytes == 495_616)
        #expect(report.oneTokenArithmeticOperations == 106_545_067_008)
        #expect((report.bandwidthUpperBoundTokensPerSecond ?? 0) < 5)
        #expect(report.attentionRoles.map(\.role) == ["global", "sliding"])
    }

    @Test("Global KV scales with context while sliding KV remains capped")
    func contextScaling() throws {
        let module = try #require(SmeltModels.definition(id: "inkling"))
        let contexts = [65_536, 262_144, 1_048_576]
        let reports = try contexts.map { context in
            try SmeltCAMStaticCostModel.report(
                module: module,
                scenario: SmeltCAMStaticCostScenario(
                    storage: .affineU4G64,
                    contextLength: context
                )
            )
        }
        let global = try reports.map { report in
            try #require(report.attentionRoles.first { $0.role == "global" })
        }
        let sliding = try reports.map { report in
            try #require(report.attentionRoles.first { $0.role == "sliding" })
        }

        #expect(global[1].kvStateBytes == global[0].kvStateBytes * 4)
        #expect(global[2].kvStateBytes == global[1].kvStateBytes * 4)
        #expect(Set(sliding.map(\.kvStateBytes)).count == 1)
        #expect(Set(sliding.map(\.attendedTokensPerLayer)) == [512])
        #expect(reports.allSatisfy { $0.bytesPerStoredParameter == 0.5625 })
    }

    @Test("Cost derives from graph shape rather than model identity")
    func identityIndependent() throws {
        let original = try #require(SmeltModels.definition(id: "inkling"))
        let originalData = try original.canonicalJSONData(prettyPrinted: true)
        let originalSource = String(decoding: originalData, as: UTF8.self)
        let renamedSource = originalSource.replacingOccurrences(
            of: "\"id\" : \"inkling\"",
            with: "\"id\" : \"cost_identity_must_not_matter\""
        )
        try #require(renamedSource != originalSource)
        let renamed = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: Data(renamedSource.utf8)
        )
        let scenario = SmeltCAMStaticCostScenario(
            storage: .nvfp4,
            contextLength: 262_144,
            exactCheckpointBytes: 591_854_374_368
        )
        let baseline = try SmeltCAMStaticCostModel.report(module: original, scenario: scenario)
        let drifted = try SmeltCAMStaticCostModel.report(module: renamed, scenario: scenario)

        #expect(baseline.moduleID != drifted.moduleID)
        #expect(baseline.moduleSemanticSHA256 != drifted.moduleSemanticSHA256)
        #expect(baseline.derivedResidentParameters == drifted.derivedResidentParameters)
        #expect(baseline.derivedActiveParameters == drifted.derivedActiveParameters)
        #expect(baseline.residentWeightBytes == drifted.residentWeightBytes)
        #expect(baseline.kvStateBytes == drifted.kvStateBytes)
        #expect(baseline.oneTokenArithmeticOperations == drifted.oneTokenArithmeticOperations)
    }
}
