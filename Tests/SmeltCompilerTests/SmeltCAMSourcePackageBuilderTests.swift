import Foundation
import SmeltModels
import SmeltSchema
import Testing

@testable import SmeltCompiler

@Suite struct SmeltCAMSourcePackageBuilderTests {
    @Test func validatesCAMSourcePackageBuildArguments() {
        let command = [
            "smelt",
            "build",
            "Models/qwen35_text.module.json",
            "--module-source-package",
            "--weights-dir",
            "weights",
            "--shader-dir",
            "Resources/Shaders",
            "--output",
            "out",
            "--trace-mode",
            "stripped-markers",
            "--optimizer-report",
        ]

        #expect(SmeltCAMSourcePackageBuilder.validateBuildCommandArguments(command) == nil)
    }

    @Test func rejectsPackageBuildFlagsForCAMSourcePackageBuild() {
        let command = [
            "smelt",
            "build",
            "Models/qwen35_text.module.json",
            "--module-source-package",
            "--weights-dir",
            "weights",
            "--shader-dir",
            "Resources/Shaders",
            "--output",
            "out",
            "--module-artifact-root",
            "source.smeltpkg",
        ]

        #expect(
            SmeltCAMSourcePackageBuilder.validateBuildCommandArguments(command)
                == "unsupported option for module source build: --module-artifact-root"
        )
    }

  @Test func requiresSourceInputForModuleSourcePackageBuild() {
        let command = [
            "smelt",
            "build",
            "Models/qwen35_text.module.json",
            "--module-source-package",
            "--shader-dir",
            "Resources/Shaders",
            "--output",
            "out",
        ]

        #expect(
            SmeltCAMSourcePackageBuilder.validateBuildCommandArguments(command)
        == "missing source input for module source build: pass --weights-dir or --source ID=PATH"
        )
    }

    @Test func sourceBuildRejectsFinalSmeltPackagePathWhereOutputRequiresParentDirectory() {
        let sourceCommand = [
            "smelt", "build", "Models/qwen35_text.module.json",
            "--module-source-package",
            "--weights-dir", "weights",
            "--shader-dir", "Resources/Shaders",
            "--output", "artifacts/final.smeltpkg",
        ]
        #expect(
            SmeltCAMSourcePackageBuilder.validateBuildCommandArguments(
                sourceCommand
            ) == "--output is a parent directory; do not pass a final .smeltpkg package path"
        )

    }

    @Test func sourcePackageWritesCanonicalModuleDescriptor() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    let moduleURL =
      repositoryRoot
            .appendingPathComponent("Models/qwen35_text.module.json")
        let module = try SmeltCAMIR.decodeModule(at: moduleURL)
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: packageURL,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try SmeltCAMSourcePackageBuilder.writePackageDescriptor(
            cam: module,
            packagePath: packageURL.path
        )

        let actual = try Data(
            contentsOf: packageURL.appendingPathComponent(
                SmeltCAMPackageDescriptor.packageFileName
            )
        )
        let expected = try SmeltCAMPackageDescriptor(from: module).canonicalJSONData()
        #expect(actual == expected)
    }

    @Test func qwenMTPModuleLowersToGenericInputFusionIR() throws {
        let module = try #require(SmeltModels.definition(id: "qwen36_27b_mtp"))
        let ir = try SmeltCAMCheckedPackageProjector.sourceModelIR(cam: module)
        let fusion = try #require(ir.config.inputFusion)

        #expect(ir.modelName == "Qwen/Qwen3.6-27B")
        #expect(ir.config.numLayers == 1)
        #expect(ir.layerPattern.expanded == [.attention])
        #expect(fusion.sourceWidth == 5_120)
        #expect(fusion.sourceCount == 2)
        #expect(fusion.normalizeSources)
        #expect(fusion.postProjectionWidth == nil)
        #expect(ir.loading.checkpointMap == .qwenMTPHF)
        #expect(ir.prefill == nil)
        try validateSmeltIR(ir)
    }

    @Test func bonsaiProjectionBanksOwnTopologyLayoutAndDecodeLowering() throws {
        let module = try #require(SmeltModels.definition(id: "bonsai_27b_binary"))
        let ir = try SmeltCAMCheckedPackageProjector.sourceModelIR(cam: module)
    #expect(
      ir.config.projectionBanks.map(\.id) == [
            "attention-input", "attention-output", "delta-input", "delta-output",
            "ffn-input", "ffn-intermediate",
        ])

        let base = SmeltWeightLayout.computeLayout(from: ir)
        let compilationPlan = try SmeltCompiler.planCompilation(ir: ir)
        let planned = compilationPlan.plannedWeightEntries
        let baseByName = Dictionary(uniqueKeysWithValues: base.map { ($0.name, $0) })
        let plannedByName = Dictionary(uniqueKeysWithValues: planned.map { ($0.name, $0) })

        func requirePackedBank(_ names: [String]) throws {
            let members = try names.map { try #require(plannedByName[$0]) }
            for pair in zip(members, members.dropFirst()) {
                #expect(pair.1.offset == pair.0.offset + pair.0.sizeBytes)
                #expect(
                    pair.1.scalesOffset
                        == pair.0.scalesOffset! + pair.0.scalesSizeBytes!
                )
            }
            let originalLast = try #require(baseByName[names.last!])
            #expect(
                members.last!.scalesOffset! + members.last!.scalesSizeBytes!
                    <= originalLast.scalesOffset! + originalLast.scalesSizeBytes!
            )
        }

        try requirePackedBank([
            "layers_0_linear_attn_in_proj_qkv_weight",
            "layers_0_linear_attn_in_proj_z_weight",
            "layers_0_linear_attn_in_proj_a_weight",
            "layers_0_linear_attn_in_proj_b_weight",
        ])
        try requirePackedBank([
            "layers_3_self_attn_q_proj_weight",
            "layers_3_self_attn_k_proj_weight",
            "layers_3_self_attn_v_proj_weight",
        ])
        try requirePackedBank([
            "layers_0_mlp_gate_proj_weight",
            "layers_0_mlp_up_proj_weight",
        ])

        let decode = try TopLevelEmitter.generate(
            ir: ir,
            compilationPlan: compilationPlan,
            traceMode: .stripped
        )
        let bankPipeline = UInt16(
            SmeltPipeline.signedBinaryBitplaneI3Bank4MatvecG128Rows8.rawValue)
        let activationPipeline = UInt16(
            SmeltPipeline.signedActivationBitplanesI3G128.rawValue)
        let normI3Pipeline = UInt16(
            SmeltPipeline.normScaleSignedActivationBitplanesI3G128.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == bankPipeline
            }.count == ir.config.numLayers
        )
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == normI3Pipeline
            }.count == ir.config.numLayers
        )
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == activationPipeline
            }.isEmpty
        )
        let selectedFFNLayers = 58
        let i4GateUpPipeline = UInt16(
            SmeltPipeline.signedBinaryBitplaneI4GateUpSwigluG128Rows8.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == i4GateUpPipeline
            }.count == selectedFFNLayers
        )
        let directGateUpPipeline = UInt16(
            SmeltPipeline.signedBinaryGateUpSwigluG128Rows8.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == directGateUpPipeline
            }.count == ir.config.numLayers - selectedFFNLayers
        )
        let downActivationPipeline = UInt16(
            SmeltPipeline.signedActivationBitplanesI4G128.rawValue)
        let normI4Pipeline = UInt16(
            SmeltPipeline.normScaleSignedActivationBitplanesI4G128.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == downActivationPipeline
            }.count == ir.config.numLayers
        )
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == normI4Pipeline
            }.count == selectedFFNLayers
        )
        let preciseNormScale = UInt16(
            SmeltPipeline.rmsNormScaleOnlyPrecise.rawValue)
        let residualNormScale = UInt16(
            SmeltPipeline.residualAddRMSNormScaleOnlyPrecise.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == preciseNormScale
            }.count == 1
        )
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == residualNormScale
            }.count == ir.config.numLayers + selectedFFNLayers - 1
        )
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.normActivationView.rawValue
            ] == 1
        )
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.residualAddNormActivationView.rawValue
            ] == ir.config.numLayers + selectedFFNLayers - 1
        )
        let downPipeline = UInt16(
            SmeltPipeline.signedBinaryBitplaneI4MatvecG128Rows8.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == downPipeline
            }.count == ir.config.numLayers
        )
        let outputActivationPipeline = UInt16(
            SmeltPipeline.signedActivationBitplanesI6G128.rawValue)
        let outputActivationIndices = decode.dispatchRecords.enumerated().compactMap {
            index, record in
            record.opKind == SmeltDispatchRecord.opDispatch
                && record.pipeline == outputActivationPipeline ? index : nil
        }
        let fusedDeltaProducerPipeline = UInt16(
            SmeltPipeline.rmsNormGatedD128SignedActivationBitplanesI6G128.rawValue)
        let fusedDeltaProducerIndices = decode.dispatchRecords.enumerated().compactMap {
            index, record in
            record.opKind == SmeltDispatchRecord.opDispatch
                && record.pipeline == fusedDeltaProducerPipeline ? index : nil
        }
        #expect(outputActivationIndices.isEmpty)
        #expect(fusedDeltaProducerIndices.count == 48)
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.gatedNormActivationView.rawValue
            ] == 48
        )
        let fusedAttentionProducerPipeline = UInt16(
            SmeltPipeline.sigmoidMulSignedActivationBitplanesI6G128.rawValue)
        let fusedAttentionProducerIndices = decode.dispatchRecords.enumerated().compactMap {
            index, record in
            record.opKind == SmeltDispatchRecord.opDispatch
                && record.pipeline == fusedAttentionProducerPipeline ? index : nil
        }
        #expect(fusedAttentionProducerIndices.count == 16)
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.sigmoidMulActivationView.rawValue
            ] == 16
        )
        let outputPipeline = UInt16(
            SmeltPipeline.signedBinaryBitplaneI6MatvecG128Rows8.rawValue)
        let outputIndices = decode.dispatchRecords.enumerated().compactMap {
            index, record in
            record.opKind == SmeltDispatchRecord.opDispatch
                && record.pipeline == outputPipeline ? index : nil
        }
        #expect(
            outputIndices.count == ir.config.numLayers
        )
        let recurrencePipeline = UInt16(
            SmeltPipeline.deltanetRecurrenceMlxDecodeD128H48QK16.rawValue)
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline == recurrencePipeline
            }.count == 48
        )
        #expect(
            decode.dispatchRecords.filter {
                $0.opKind == SmeltDispatchRecord.opDispatch
                    && $0.pipeline
                        == UInt16(SmeltPipeline.deltanetRecurrenceMlxDecode.rawValue)
            }.isEmpty
        )
    let producerIndices =
      (fusedDeltaProducerIndices
      + fusedAttentionProducerIndices).sorted()
        #expect(zip(producerIndices, outputIndices).allSatisfy { pair in pair.0 < pair.1 })

        let bufferPlan = buildBufferPlan(from: ir)
    let planes = try #require(
      bufferPlan.slots.first {
            $0.index == bufferPlan.projectionActivationPlanesSlot
        })
    let scales = try #require(
      bufferPlan.slots.first {
            $0.index == bufferPlan.projectionActivationScalesSlot
        })
        #expect(planes.shape == [256, 136, 6, 4])
        #expect(scales.shape == [256, 136])

        let prefill = try PrefillEmitter.generate(
            ir: ir, compilationPlan: compilationPlan, traceMode: .stripped
        )
    let batchedBuilders = Set(
      [
            SmeltPipeline.signedActivationBitplanesI2G128Batched,
            .signedActivationBitplanesI3G128Batched,
            .signedActivationBitplanesI4G128Batched,
            .signedActivationBitplanesI5G128Batched,
            .signedActivationBitplanesI6G128Batched,
        ].map { UInt16($0.rawValue) })
    let batchedConsumers = Set(
      [
            SmeltPipeline.signedBinaryBitplaneI2MatvecG128Rows8BatchedB4,
            .signedBinaryBitplaneI3MatvecG128Rows8BatchedB4,
            .signedBinaryBitplaneI4MatvecG128Rows8BatchedB4,
            .signedBinaryBitplaneI5MatvecG128Rows8BatchedB4,
            .signedBinaryBitplaneI6MatvecG128Rows8BatchedB4,
        ].map { UInt16($0.rawValue) })
    #expect(
      prefill.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
                && batchedBuilders.contains($0.pipeline)
        }.count == 250)
    #expect(
      prefill.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
                && batchedConsumers.contains($0.pipeline)
        }.count == 484)
        let prefillRecurrencePipeline = UInt16(
            SmeltPipeline.deltanetRecurrenceMlxPrefillD128H48QK16.rawValue)
    #expect(
      prefill.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
                && $0.pipeline == prefillRecurrencePipeline
        }.count == 48)
    #expect(
      prefill.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
                && $0.pipeline
                    == UInt16(SmeltPipeline.deltanetRecurrenceMlxPrefill.rawValue)
        }.isEmpty)
    }

    @Test func bonsaiTernaryUsesReferenceAffineProjectionBricksWithoutPinnedViews() throws {
        let module = try #require(SmeltModels.definition(id: "bonsai_27b_ternary"))
        let ir = try SmeltCAMCheckedPackageProjector.sourceModelIR(cam: module)
        let compilationPlan = try SmeltCompiler.planCompilation(ir: ir)
        let decode = try TopLevelEmitter.generate(
            ir: ir,
            compilationPlan: compilationPlan,
            traceMode: .stripped
        )

        let affine = UInt16(
            SmeltPipeline.signedTernaryAffineMatvecG128Rows8.rawValue
        )
        let fusedGateUp = UInt16(
            SmeltPipeline.signedTernaryAffineGateUpSwigluG128Rows8.rawValue
        )
        let fusedResidual = UInt16(
            SmeltPipeline.signedTernaryAffineMatvecAddG128Rows8.rawValue
        )
        let projectionBank = UInt16(
            SmeltPipeline.signedTernaryAffineBank4MatvecG128Rows8.rawValue
        )
        let bitplanePipelineCases: [SmeltPipeline] = [
            .signedActivationBitplanesI5G128,
            .signedActivationBitplanesI6G128,
            .signedTernaryBitplaneI5MatvecG128Rows8,
            .signedTernaryBitplaneI5MatvecG128Rows2Wide,
            .signedTernaryBitplaneI5Bank4MatvecG128Rows8,
            .signedTernaryBitplaneI5GateUpSwigluG128Rows8,
            .signedTernaryBitplaneI6MatvecG128Rows8,
        ]
        let bitplanePipelines = Set(
            bitplanePipelineCases.map { UInt16($0.rawValue) }
        )
        let dispatches = decode.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
        }

        // Exact MLX projection arithmetic is the default ternary brick. The
        // model descriptor no longer pins an experimental activation-view
        // strategy; a future measured planner may select one through the same
        // generic compiler path.
        #expect(dispatches.filter { $0.pipeline == affine }.count == 1)
        #expect(dispatches.filter { $0.pipeline == fusedGateUp }.count == 0)
        #expect(dispatches.filter { $0.pipeline == fusedResidual }.count == 128)
        #expect(dispatches.filter { $0.pipeline == projectionBank }.count == 128)
        #expect(
            dispatches.filter {
                $0.pipeline == UInt16(SmeltPipeline.swigluFused.rawValue)
            }.count == 64
        )
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.matvecResidualAdd.rawValue
            ] == 128
        )
        #expect(dispatches.allSatisfy { !bitplanePipelines.contains($0.pipeline) })
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.normActivationView.rawValue
            ] == nil
        )
        #expect(
            decode.optimizationStats.rewriteCounts[
                SmeltFusionRule.residualAddNormActivationView.rawValue
            ] == nil
        )
    }
}
