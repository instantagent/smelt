import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class OptimizerReportTests: XCTestCase {
    private func makeManifest(
        modelName: String = "test/optimizer-report-model",
        hiddenSize: Int = 2560,
        ffnDim: Int = 10240,
        vocabSize: Int = 262144,
        optimizationReport: SmeltOptimizationReport? = nil
    ) -> SmeltManifest {
        SmeltManifest(
            modelName: modelName,
            config: SmeltManifestConfig(
                hiddenSize: hiddenSize,
                numLayers: 42,
                vocabSize: vocabSize,
                staticSeqCapacity: 131072,
                ropeDim: 256,
                numDeltaLayers: 0,
                numAttnLayers: 0,
                ffnDim: ffnDim
            ),
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: ""
            ),
            buildProvenance: SmeltBuildProvenance(
                buildFingerprint: "build123",
                weightsFingerprint: "weights123",
                specSHA256: "spec123",
                compilerSourcesSHA256: "compiler123",
                shaderSourcesSHA256: "shader123",
                resolvedOptions: SmeltResolvedBuildOptions(
                    layerPatternUnit: ["attention"],
                    layerPatternRepeats: 42,
                    quantizationStrategy: "affine_int4",
                    groupSize: 128,
                    excludePatterns: [],
                    quantizeEmbedding: false,
                    loadingStrategy: "download",
                    packing: "qmm16x128",
                    prefillEngine: "metal",
                    maxPrefillBatch: 256,
                    prefillHandoffFamilies: [],
                    inferenceMaxTokens: 128,
                    eosTokens: [],
                    thinkToken: nil,
                    thinkEndToken: nil,
                    thinkSkipSuffix: nil,
                    tiedLMHead: false,
                    traceMode: "full"
                )
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 0, entries: []),
            buffers: SmeltBufferTable(slots: []),
            pipelines: [],
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 0,
                recStateBaseSlot: 0,
                keyCacheBaseSlot: 0,
                valCacheBaseSlot: 0,
                ropeCosSlot: 0,
                ropeSinSlot: 0,
                tokenIdSlot: 0,
                positionSlot: 0,
                weightsSlot: 0
            ),
            optimizationReport: optimizationReport
        )
    }

    private func structureReport(
        kind: String,
        names: [String]
    ) -> SmeltDispatchStructureReport {
        SmeltDispatchStructureReport(
            packagePath: "/tmp/test.smeltpkg",
            tableName: kind == "prefill" ? "prefill_dispatches.bin" : "dispatches.bin",
            totalRecords: names.count,
            dispatchCount: names.count,
            swapCount: 0,
            pipelineUsages: names.enumerated().map { index, name in
                SmeltDispatchPipelineUsage(
                    pipelineIndex: UInt16(index),
                    name: name,
                    dispatchCount: 1
                )
            }
        )
    }

    // Representative decode/prefill pipeline-name lists used only as
    // structure-report inputs; the optimizer-report assertions here do not
    // depend on the specific names.
    private var sampleDecodePipelineNames: [String] {
        [
            "rms_norm_1pw_d2560",
            "rms_norm_1pw_d2560_add",
            "attention_decode",
            "affine_matvec_c2560_r2048_g128_rows4",
        ]
    }

    private var samplePrefillPipelineNames: [String] {
        [
            "affine_embedding_gather_batched",
            "rms_norm_1pw_d2560_batched",
            "attention_prefill",
            "affine_matvec_c2560_r2048_g128_batched",
        ]
    }

    private func camContext() -> SmeltOptimizerCAMReportContext {
        SmeltOptimizerCAMReportContext(
            camSemanticSHA256: "cam123",
            exportABISHA256: "abi123",
            descriptorGraphSignatureSHA256: "graph123",
            exportID: "optimizer",
            flowID: "report",
            matchedGateIDs: ["gate.optimizer-report"],
            authoredCapabilities: ["optimizer.report"]
        )
    }

    func testOptimizationReportCompilationPlanCodablePolicyShape() throws {
        let legacy = try JSONDecoder().decode(
            SmeltOptimizationReport.self,
            from: Data("""
            {
              "decodeRewriteCounts": {"rmsNormResidualAdd": 1},
              "prefillRewriteCounts": {}
            }
            """.utf8)
        )
        XCTAssertNil(legacy.compilationPlan)
        XCTAssertEqual(legacy.decodeRewriteCounts["rmsNormResidualAdd"], 1)

        XCTAssertThrowsError(try JSONDecoder().decode(
            SmeltCompilationPlanReport.self,
            from: Data("""
            {
              "plannedKernelUses": 78,
              "generatedKernels": 4,
              "plannedWeightUses": 102,
              "plannedWeightStorageDecisions": 78,
              "duplicateWeightLayouts": 0,
              "weightStorageIssues": 0,
              "memoryNeutralWeightStorage": true
            }
            """.utf8)
        ))

        let report = SmeltOptimizationReport(
            compilationPlan: SmeltCompilationPlanReport(
                plannedKernelUses: 78,
                plannedKernelConsumers: [
                    SmeltPlannedKernelConsumerReport(
                        consumerID: "layers_0_mlp.down_proj.residual.decode",
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        phase: "decode",
                        operation: "affineMatvecResidualAdd",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 64,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_down_proj_weight",
                                role: "affine"
                            ),
                        ]
                    ),
                    SmeltPlannedKernelConsumerReport(
                        consumerID: "layers_0_mlp.gate_up.prefill",
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 64,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_gate_proj_weight",
                                role: "gate"
                            ),
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_up_proj_weight",
                                role: "up"
                            ),
                        ]
                    ),
                ],
                plannedKernelCandidates: 80,
                unsupportedKernelCandidates: 2,
                unsupportedKernelCandidateRecords: [
                    SmeltUnsupportedKernelCandidateReport(
                        consumerID: "layers_1_mlp.down_proj.prefill",
                        phase: "prefill",
                        operation: "affineMatvecPrefillFull",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 48,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_down_proj_weight",
                                role: "affine"
                            ),
                        ],
                        reason: "no_generated_capability"
                    ),
                    SmeltUnsupportedKernelCandidateReport(
                        consumerID: "layers_1_mlp.gate_up.prefill",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 48,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_gate_proj_weight",
                                role: "gate"
                            ),
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_up_proj_weight",
                                role: "up"
                            ),
                        ],
                        reason: "no_generated_capability"
                    ),
                ],
                plannedBufferSlots: 128,
                plannedActivationBytes: 4_194_304,
                generatedKernels: 4,
                emittedGeneratedKernels: 2,
                plannedGeneratedKernelCapabilities: [
                    SmeltGeneratedKernelCapabilityReport(
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        phase: "decode",
                        operation: "affineMatvecResidualAdd",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 64,
                        sourceKind: "package_local_generated",
                        emittedGeneratedSource: true,
                        weightRequirements: [
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "affine",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                        ],
                        rowTile: 8,
                        batchTile: nil,
                        threadgroupWidth: 64
                    ),
                    SmeltGeneratedKernelCapabilityReport(
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 64,
                        sourceKind: "built_in_catalog",
                        emittedGeneratedSource: false,
                        weightRequirements: [
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "gate",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "up",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                        ],
                        rowTile: 32,
                        batchTile: 16,
                        threadgroupWidth: 128
                    ),
                ],
                plannedGeneratedKernelNames: [
                    "fused_affine_matvec_add_c3584_r1024_g64",
                    "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                    "affine_matvec_c3584_r1024_g64_batched_full",
                    "fused_affine_matvec_add_c2048_r1024_g64",
                ],
                emittedGeneratedKernelNames: [
                    "fused_affine_matvec_add_c3584_r1024_g64",
                    "fused_affine_matvec_add_c2048_r1024_g64",
                ],
                plannedWeightUses: 102,
                plannedWeightNames: [
                    "layers_0_mlp_down_proj_weight",
                    "layers_0_mlp_gate_proj_weight",
                ],
                plannedWeightConsumerIDs: [
                    "layers_0_mlp.down_proj.residual.decode",
                    "layers_0_mlp.gate_up.prefill",
                ],
                plannedWeightConsumers: [
                    SmeltPlannedWeightConsumerReport(
                        weightName: "layers_0_mlp_down_proj_weight",
                        consumerID: "layers_0_mlp.down_proj.residual.decode",
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        weightRole: "affine",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                    SmeltPlannedWeightConsumerReport(
                        weightName: "layers_0_mlp_gate_proj_weight",
                        consumerID: "layers_0_mlp.gate_up.prefill",
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        weightRole: "gate",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                ],
                plannedWeightStorageDecisions: 78,
                plannedWeightStorageDecisionNames: [
                    "layers_0_mlp_down_proj_weight",
                    "layers_0_mlp_gate_proj_weight",
                ],
                plannedWeightStorageDecisionRecords: [
                    SmeltPlannedWeightStorageDecisionReport(
                        weightName: "layers_0_mlp_down_proj_weight",
                        currentLayout: "affine_u4_row_major_g64",
                        selectedLayout: "affine_u4_row_major_g64",
                        consumers: [
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.down_proj.residual.decode",
                                consumerKind: "ffnDownResidualDecode"
                            ),
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.down_proj.prefill",
                                consumerKind: "ffnDownPrefill"
                            ),
                        ],
                        requiresDuplicateLayout: false
                    ),
                    SmeltPlannedWeightStorageDecisionReport(
                        weightName: "layers_0_mlp_gate_proj_weight",
                        currentLayout: "affine_u4_row_major_g64",
                        selectedLayout: "affine_u4_row_major_g64",
                        consumers: [
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.gate_up.prefill",
                                consumerKind: "ffnGateUpPrefill"
                            ),
                        ],
                        requiresDuplicateLayout: false
                    ),
                ],
                duplicateWeightLayouts: 0,
                weightStorageIssues: 0,
                weightStorageIssueNames: [],
                memoryNeutralWeightStorage: true,
                generatedKernelConsumerKinds: [
                    "ffnDownResidualDecode",
                    "ffnGateUpPrefill",
                ]
            )
        )
        let decoded = try JSONDecoder().decode(
            SmeltOptimizationReport.self,
            from: JSONEncoder().encode(report)
        )
        XCTAssertEqual(decoded.compilationPlan, report.compilationPlan)
    }

    func testMarkdownIncludesCompilationPlanSummary() {
        let optimizationReport = SmeltOptimizationReport(
            compilationPlan: SmeltCompilationPlanReport(
                plannedKernelUses: 78,
                plannedKernelConsumers: [
                    SmeltPlannedKernelConsumerReport(
                        consumerID: "layers_0_mlp.down_proj.residual.decode",
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        phase: "decode",
                        operation: "affineMatvecResidualAdd",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 64,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_down_proj_weight",
                                role: "affine"
                            ),
                        ]
                    ),
                    SmeltPlannedKernelConsumerReport(
                        consumerID: "layers_0_mlp.gate_up.prefill",
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 64,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_gate_proj_weight",
                                role: "gate"
                            ),
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_0_mlp_up_proj_weight",
                                role: "up"
                            ),
                        ]
                    ),
                ],
                plannedKernelCandidates: 80,
                unsupportedKernelCandidates: 2,
                unsupportedKernelCandidateRecords: [
                    SmeltUnsupportedKernelCandidateReport(
                        consumerID: "layers_1_mlp.down_proj.prefill",
                        phase: "prefill",
                        operation: "affineMatvecPrefillFull",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 48,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_down_proj_weight",
                                role: "affine"
                            ),
                        ],
                        reason: "no_generated_capability"
                    ),
                    SmeltUnsupportedKernelCandidateReport(
                        consumerID: "layers_1_mlp.gate_up.prefill",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 48,
                        weights: [
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_gate_proj_weight",
                                role: "gate"
                            ),
                            SmeltPlannedKernelWeightReport(
                                weightName: "layers_1_mlp_up_proj_weight",
                                role: "up"
                            ),
                        ],
                        reason: "no_generated_capability"
                    ),
                ],
                plannedBufferSlots: 128,
                plannedActivationBytes: 4_194_304,
                generatedKernels: 4,
                emittedGeneratedKernels: 2,
                plannedGeneratedKernelCapabilities: [
                    SmeltGeneratedKernelCapabilityReport(
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        phase: "decode",
                        operation: "affineMatvecResidualAdd",
                        rows: 1_024,
                        cols: 3_584,
                        groupSize: 64,
                        sourceKind: "package_local_generated",
                        emittedGeneratedSource: true,
                        weightRequirements: [
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "affine",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                        ],
                        rowTile: 8,
                        batchTile: nil,
                        threadgroupWidth: 64
                    ),
                    SmeltGeneratedKernelCapabilityReport(
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        phase: "prefill",
                        operation: "fusedGateUpSwigluPrefillFull",
                        rows: 3_584,
                        cols: 1_024,
                        groupSize: 64,
                        sourceKind: "built_in_catalog",
                        emittedGeneratedSource: false,
                        weightRequirements: [
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "gate",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                            SmeltGeneratedKernelWeightRequirementReport(
                                role: "up",
                                acceptedLayouts: ["affine_u4_row_major_g64"]
                            ),
                        ],
                        rowTile: 32,
                        batchTile: 16,
                        threadgroupWidth: 128
                    ),
                ],
                plannedGeneratedKernelNames: [
                    "fused_affine_matvec_add_c3584_r1024_g64",
                    "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                    "affine_matvec_c3584_r1024_g64_batched_full",
                    "fused_affine_matvec_add_c2048_r1024_g64",
                ],
                emittedGeneratedKernelNames: [
                    "fused_affine_matvec_add_c3584_r1024_g64",
                    "fused_affine_matvec_add_c2048_r1024_g64",
                ],
                plannedWeightUses: 102,
                plannedWeightNames: [
                    "layers_0_mlp_down_proj_weight",
                    "layers_0_mlp_gate_proj_weight",
                ],
                plannedWeightConsumerIDs: [
                    "layers_0_mlp.down_proj.residual.decode",
                    "layers_0_mlp.gate_up.prefill",
                ],
                plannedWeightConsumers: [
                    SmeltPlannedWeightConsumerReport(
                        weightName: "layers_0_mlp_down_proj_weight",
                        consumerID: "layers_0_mlp.down_proj.residual.decode",
                        capabilityName: "fused_affine_matvec_add_c3584_r1024_g64",
                        weightRole: "affine",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                    SmeltPlannedWeightConsumerReport(
                        weightName: "layers_0_mlp_gate_proj_weight",
                        consumerID: "layers_0_mlp.gate_up.prefill",
                        capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                        weightRole: "gate",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                ],
                plannedWeightStorageDecisions: 78,
                plannedWeightStorageDecisionNames: [
                    "layers_0_mlp_down_proj_weight",
                    "layers_0_mlp_gate_proj_weight",
                ],
                plannedWeightStorageDecisionRecords: [
                    SmeltPlannedWeightStorageDecisionReport(
                        weightName: "layers_0_mlp_down_proj_weight",
                        currentLayout: "affine_u4_row_major_g64",
                        selectedLayout: "affine_u4_row_major_g64",
                        consumers: [
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.down_proj.residual.decode",
                                consumerKind: "ffnDownResidualDecode"
                            ),
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.down_proj.prefill",
                                consumerKind: "ffnDownPrefill"
                            ),
                        ],
                        requiresDuplicateLayout: false
                    ),
                    SmeltPlannedWeightStorageDecisionReport(
                        weightName: "layers_0_mlp_gate_proj_weight",
                        currentLayout: "affine_u4_row_major_g64",
                        selectedLayout: "affine_u4_row_major_g64",
                        consumers: [
                            SmeltPlannedWeightStorageDecisionConsumerReport(
                                consumerID: "layers_0_mlp.gate_up.prefill",
                                consumerKind: "ffnGateUpPrefill"
                            ),
                        ],
                        requiresDuplicateLayout: false
                    ),
                ],
                duplicateWeightLayouts: 0,
                weightStorageIssues: 0,
                weightStorageIssueNames: [],
                memoryNeutralWeightStorage: true,
                generatedKernelConsumerKinds: [
                    "ffnDownResidualDecode",
                    "ffnGateUpPrefill",
                ]
            )
        )
        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: optimizationReport),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: sampleDecodePipelineNames
            ),
            prefillReport: structureReport(
                kind: "prefill",
                names: samplePrefillPipelineNames
            ),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("Capability-planned compilation:"))
        XCTAssertTrue(markdown.contains("- Kernel generation policy: `auto`"))
        XCTAssertTrue(markdown.contains(
            "- Generated kernel consumer kinds: `ffnDownResidualDecode`, `ffnGateUpPrefill`"
        ))
        XCTAssertTrue(markdown.contains("- Weight layout policy: `memory_neutral`"))
        XCTAssertTrue(markdown.contains("- Planned buffer slots: 128"))
        XCTAssertTrue(markdown.contains("- Planned activation bytes: 4194304"))
        XCTAssertTrue(markdown.contains(
            "- Generated kernel candidates: 78 selected / 80 total (2 unsupported)"
        ))
        XCTAssertTrue(markdown.contains(
            "- Unsupported kernel candidate records: `layers_1_mlp.down_proj.prefill` prefill/affineMatvecPrefillFull c3584r1024g48 no_generated_capability via `affine:layers_1_mlp_down_proj_weight`, `layers_1_mlp.gate_up.prefill` prefill/fusedGateUpSwigluPrefillFull c1024r3584g48 no_generated_capability via `gate:layers_1_mlp_gate_proj_weight`, `up:layers_1_mlp_up_proj_weight`"
        ))
        XCTAssertTrue(markdown.contains("- Planned kernel uses: 78"))
        XCTAssertTrue(markdown.contains(
            "- Planned kernel consumer records: `layers_0_mlp.down_proj.residual.decode`->`fused_affine_matvec_add_c3584_r1024_g64` decode/affineMatvecResidualAdd c3584r1024g64, `layers_0_mlp.gate_up.prefill`->`fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full` prefill/fusedGateUpSwigluPrefillFull c1024r3584g64"
        ))
        XCTAssertTrue(markdown.contains("- Planned generated capabilities: 4"))
        XCTAssertTrue(markdown.contains("- Emitted generated kernels: 2"))
        XCTAssertTrue(markdown.contains(
            "- Planned generated capability records: `fused_affine_matvec_add_c3584_r1024_g64` decode/affineMatvecResidualAdd c3584r1024g64 package_local_generated, `fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full` prefill/fusedGateUpSwigluPrefillFull c1024r3584g64 built_in_catalog"
        ))
        XCTAssertTrue(markdown.contains(
            "- Planned generated capability names: `fused_affine_matvec_add_c3584_r1024_g64`, `fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full`, `affine_matvec_c3584_r1024_g64_batched_full`, `fused_affine_matvec_add_c2048_r1024_g64`"
        ))
        XCTAssertTrue(markdown.contains(
            "- Emitted generated kernel names: `fused_affine_matvec_add_c3584_r1024_g64`, `fused_affine_matvec_add_c2048_r1024_g64`"
        ))
        XCTAssertTrue(markdown.contains("- Planned weight consumers: 102"))
        XCTAssertTrue(markdown.contains(
            "- Planned weight names: `layers_0_mlp_down_proj_weight`, `layers_0_mlp_gate_proj_weight`"
        ))
        XCTAssertTrue(markdown.contains(
            "- Planned weight consumer IDs: `layers_0_mlp.down_proj.residual.decode`, `layers_0_mlp.gate_up.prefill`"
        ))
        XCTAssertTrue(markdown.contains(
            "- Planned weight consumer records: `layers_0_mlp_down_proj_weight`->`layers_0_mlp.down_proj.residual.decode` via `fused_affine_matvec_add_c3584_r1024_g64`/affine, `layers_0_mlp_gate_proj_weight`->`layers_0_mlp.gate_up.prefill` via `fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full`/gate"
        ))
        XCTAssertTrue(markdown.contains("- Planned weight storage decisions: 78"))
        XCTAssertTrue(markdown.contains(
            "- Planned weight storage decision names: `layers_0_mlp_down_proj_weight`, `layers_0_mlp_gate_proj_weight`"
        ))
        XCTAssertTrue(markdown.contains(
            "- Planned weight storage decision records: `layers_0_mlp_down_proj_weight` `affine_u4_row_major_g64`->`affine_u4_row_major_g64` single for `layers_0_mlp.down_proj.residual.decode` (ffnDownResidualDecode), `layers_0_mlp.down_proj.prefill` (ffnDownPrefill), `layers_0_mlp_gate_proj_weight` `affine_u4_row_major_g64`->`affine_u4_row_major_g64` single for `layers_0_mlp.gate_up.prefill` (ffnGateUpPrefill)"
        ))
        XCTAssertTrue(markdown.contains("- Weight storage: memory-neutral, 0 duplicate layouts, 0 issues"))
    }

    func testMarkdownTurnsMissingCompilerOpportunitiesIntoAgentTasks() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: ["rmsNormResidualAdd": 42],
            prefillRewriteCounts: ["rmsNormResidualAdd": 2772],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560->elementwise_add",
                    count: 42,
                    fusedKernelAvailable: true
                ),
            ],
            prefillOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "dualMatvecActivation",
                    shape: "affine_matvec_c2560_r10240_g128_batched+affine_matvec_c2560_r10240_g128_batched->geglu_fused",
                    count: 42,
                    fusedKernelAvailable: false
                ),
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560_batched->affine_matvec_c2560_r10240_g128_batched",
                    count: 42,
                    fusedKernelAvailable: false
                ),
            ]
        )
        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: optimizationReport),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: sampleDecodePipelineNames
            ),
            prefillReport: structureReport(
                kind: "prefill",
                names: samplePrefillPipelineNames
            ),
            camContext: camContext(),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("**Rebuild package?** No for CAM admission"))
        XCTAssertTrue(markdown.contains("- CAM optimizer route: export `optimizer`, flow `report`"))
        XCTAssertTrue(markdown.contains("**Add kernels?** Yes"))
        XCTAssertTrue(markdown.contains("## Compiler Strategy Diagnosis"))
        XCTAssertTrue(markdown.contains("Optimization goal: maximize throughput. No target is required"))
        XCTAssertTrue(markdown.contains("ordered by Apple Silicon impact score"))
        XCTAssertTrue(markdown.contains("Apple Silicon priority score:"))
        XCTAssertTrue(markdown.contains("Scoring signals:"))
        XCTAssertTrue(markdown.contains("Task ID: `fusion-prefill-dualmatvecactivation"))
        XCTAssertTrue(markdown.contains("norm_scale_affine_matvec_c2560_r10240_g128_batched"))
        XCTAssertTrue(markdown.contains("Done when: this task ID disappears from the optimizer report"))

        let tasks = SmeltOptimizerReportGenerator.agentTasks(
            from: makeManifest(optimizationReport: optimizationReport)
        )
        XCTAssertEqual(tasks.count, 2)
        XCTAssertEqual(tasks[0].mode, "prefill")
        XCTAssertTrue(tasks[0].id.hasPrefix("fusion-prefill-dualmatvecactivation"))
        XCTAssertGreaterThan(tasks[0].appleSiliconImpactScore, tasks[1].appleSiliconImpactScore)
        XCTAssertTrue(tasks[0].markdownCard(priority: 1).contains("Task ID: `\(tasks[0].id)`"))
    }

    func testAgentTasksPreferAppleSiliconImpactOverRawSiteCount() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: [:],
            prefillRewriteCounts: [:],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560->affine_matvec_c2560_r2048_g128_rows4",
                    count: 80,
                    fusedKernelAvailable: false
                ),
            ],
            prefillOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "dualMatvecActivation",
                    shape: "affine_matvec_c2560_r10240_g128_batched+affine_matvec_c2560_r10240_g128_batched->geglu_fused",
                    count: 42,
                    fusedKernelAvailable: false
                ),
            ]
        )

        let tasks = SmeltOptimizerReportGenerator.agentTasks(
            from: makeManifest(optimizationReport: optimizationReport)
        )

        XCTAssertEqual(tasks.count, 2)
        XCTAssertEqual(tasks[0].mode, "prefill")
        XCTAssertEqual(tasks[0].pattern, "dualMatvecActivation")
        XCTAssertGreaterThan(tasks[0].appleSiliconImpactScore, tasks[1].appleSiliconImpactScore)
        XCTAssertTrue(tasks[0].scoringSignals.contains("prefill batches stress unified-memory bandwidth"))
    }

    func testStrategyDiagnosisEscalatesDecodeWhenVocabProjectionDominates() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: [:],
            prefillRewriteCounts: [:],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560->affine_matvec_c2560_r2048_g128_rows4",
                    count: 35,
                    fusedKernelAvailable: false
                ),
            ],
            prefillOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560_batched->affine_matvec_c2560_r10240_g128_batched",
                    count: 42,
                    fusedKernelAvailable: false
                ),
            ]
        )

        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: optimizationReport),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: [
                    "affine_matvec_c2560_r262144_g128_rows8",
                    "affine_matvec_c2560_r2048_g128_rows4",
                ]
            ),
            prefillReport: structureReport(
                kind: "prefill",
                names: [
                    "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full",
                    "affine_matvec_c10240_r2560_g128_batched_tile4",
                ]
            ),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("vocab-scale final projection"))
        XCTAssertTrue(markdown.contains("not a credible decode plan by themselves"))
        XCTAssertTrue(markdown.contains("emit a structural decode work item"))
    }

    func testStrategyDiagnosisEscalatesDecodeWithoutUserTargets() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: [:],
            prefillRewriteCounts: [:],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560->affine_matvec_c2560_r2048_g128_rows4",
                    count: 35,
                    fusedKernelAvailable: false
                ),
            ],
            prefillOpportunities: []
        )

        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: optimizationReport),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: [
                    "norm_scale_affine_gate_up_geglu_c2560_r10240_g128_rows4",
                    "affine_matvec_c10240_r2560_g128_rows4",
                    "affine_matvec_c2560_r2048_g128_rows4",
                ]
            ),
            prefillReport: structureReport(kind: "prefill", names: []),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("Optimization goal: maximize throughput. No target is required"))
        XCTAssertTrue(markdown.contains("active heavyweight QMM hotspot dominates"))
        XCTAssertTrue(markdown.contains("emit a structural decode work item for `norm_scale_affine_gate_up_geglu_c2560_r10240_g128_rows4`"))
    }

    func testMarkdownUsesMeasuredTimingEvidenceForStrategy() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: [:],
            prefillRewriteCounts: [:],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560->affine_matvec_c2560_r2048_g128_rows4",
                    count: 35,
                    fusedKernelAvailable: false
                ),
            ],
            prefillOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2560_batched->affine_matvec_c2560_r2048_g128_batched",
                    count: 35,
                    fusedKernelAvailable: false
                ),
            ]
        )
        let timingEvidence = SmeltOptimizerTimingEvidence(
            decode: SmeltOptimizerModeTiming(
                mode: "decode",
                iterations: 2,
                tokenCount: 1,
                position: 0,
                totalGpuUs: 1000,
                kernels: [
                    SmeltOptimizerKernelTiming(
                        name: "affine_matvec_c2560_r262144_g128_rows8",
                        dispatchCount: 1,
                        avgGpuUs: 650,
                        pctOfTotal: 65
                    ),
                    SmeltOptimizerKernelTiming(
                        name: "rms_norm_1pw_d2560",
                        dispatchCount: 43,
                        avgGpuUs: 50,
                        pctOfTotal: 5
                    ),
                ]
            ),
            prefill: [
                SmeltOptimizerModeTiming(
                    mode: "prefill",
                    iterations: 2,
                    tokenCount: 64,
                    position: nil,
                    totalGpuUs: 2000,
                    kernels: [
                        SmeltOptimizerKernelTiming(
                            name: "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full",
                            dispatchCount: 42,
                            avgGpuUs: 1200,
                            pctOfTotal: 60
                        ),
                    ]
                ),
            ]
        )

        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: optimizationReport),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: [
                    "affine_matvec_c2560_r2048_g128_rows4",
                    "affine_matvec_c2560_r262144_g128_rows8",
                ]
            ),
            prefillReport: structureReport(
                kind: "prefill",
                names: [
                    "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full",
                    "affine_matvec_c2560_r2048_g128_batched",
                ]
            ),
            timingEvidence: timingEvidence,
            options: SmeltOptimizerReportOptions(includeTimings: true, timingIterations: 2)
        )

        XCTAssertTrue(markdown.contains("## Measured Timing Evidence"))
        XCTAssertTrue(markdown.contains("Top measured kernels:"))
        XCTAssertTrue(markdown.contains("affine_matvec_c2560_r262144_g128_rows8"))
        XCTAssertTrue(markdown.contains("fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full"))
        XCTAssertTrue(markdown.contains("trust the measured profile first"))
        XCTAssertTrue(markdown.contains("emit a timing-backed structural work item"))
        XCTAssertTrue(markdown.contains("## Timing-Backed Structural Work Queue"))
        XCTAssertTrue(markdown.contains("Structural Priority 1"))
        XCTAssertTrue(markdown.contains("measured hot-path work"))
    }

    func testStrategyDiagnosisUsesManifestVocabForUnknownFamilies() {
        let optimizationReport = SmeltOptimizationReport(
            decodeRewriteCounts: [:],
            prefillRewriteCounts: [:],
            decodeOpportunities: [
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2048->affine_matvec_c2048_r248320_g64_rows4",
                    count: 1,
                    fusedKernelAvailable: false
                ),
                SmeltFusionOpportunitySummary(
                    pattern: "normConsumer",
                    shape: "rms_norm_1pw_d2048->affine_matvec_c2048_r6144_g64_rows4",
                    count: 24,
                    fusedKernelAvailable: false
                ),
            ],
            prefillOpportunities: []
        )

        let manifest = makeManifest(
            modelName: "example/unknown-family",
            hiddenSize: 2048,
            ffnDim: 6144,
            vocabSize: 248320,
            optimizationReport: optimizationReport
        )
        let tasks = SmeltOptimizerReportGenerator.agentTasks(from: manifest)
        XCTAssertEqual(tasks.first?.shape, "rms_norm_1pw_d2048->affine_matvec_c2048_r248320_g64_rows4")
        XCTAssertEqual(tasks.first?.touchesVocabScale, true)

        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: manifest,
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: [
                    "affine_matvec_c2048_r248320_g64_rows4",
                    "affine_matvec_c2048_r6144_g64_rows4",
                ]
            ),
            prefillReport: structureReport(kind: "prefill", names: []),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("- CAM optimizer route: unavailable"))
        XCTAssertTrue(markdown.contains("vocab-scale final projection"))
        XCTAssertTrue(markdown.contains("affine_matvec_c2048_r248320_g64_rows4"))
    }

    func testMarkdownRecommendsRebuildWhenCAMContextHasNoCompilerMetadata() {
        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(optimizationReport: nil),
            integrity: nil,
            decodeReport: structureReport(
                kind: "decode",
                names: ["swiglu_fused"]
            ),
            prefillReport: structureReport(
                kind: "prefill",
                names: ["affine_embedding_gather_batched"]
            ),
            camContext: camContext(),
            options: SmeltOptimizerReportOptions()
        )

        XCTAssertTrue(markdown.contains("**Rebuild package?** Probably"))
        XCTAssertTrue(markdown.contains("CAM optimizer-report admission"))
        XCTAssertTrue(markdown.contains("without manifest-family profile inference"))
        XCTAssertFalse(markdown.contains("Gate failures:"))
        XCTAssertFalse(markdown.contains("missing required decode pipeline"))
    }

    func testMarkdownReportsDeterministicGraphPlanDecision() {
        let context = SmeltCostModelContext(
            mode: .decode,
            sequenceLength: 17,
            position: 16
        )
        let samples = (0..<10).map { index in
            SmeltPairedPlanSample(
                order: index.isMultiple(of: 2) ? .baselineFirst : .candidateFirst,
                baselineGPUUs: 25_500,
                candidateGPUUs: 25_100,
                baselineWallUs: 26_000,
                candidateWallUs: 25_600
            )
        }
        let measurement = SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max-package-key",
            context: context,
            baselinePlanID: "stable",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: samples,
            baselineStructure: SmeltPlanStructuralCost(
                recordCount: 1_062,
                dispatchCount: 934,
                swapCount: 128,
                totalThreadgroups: 9_800,
                singleThreadgroupDispatches: 128,
                maxThreadgroupsPerDispatch: 40,
                distinctPipelines: 474
            ),
            candidateStructure: SmeltPlanStructuralCost(
                recordCount: 934,
                dispatchCount: 806,
                swapCount: 128,
                totalThreadgroups: 4_680,
                singleThreadgroupDispatches: 256,
                maxThreadgroupsPerDispatch: 1,
                distinctPipelines: 475
            ),
            parity: SmeltPlanParityEvidence(
                checkedSteps: 17,
                valuesPerStep: 248_320
            )
        )
        let policy = SmeltPlanDecisionPolicy()
        let decision = SmeltPlanCostDecider(
            provenanceKey: measurement.provenanceKey,
            policy: policy
        ).decide(measurement: measurement)
        let comparison = SmeltPlanComparisonReport(
            measurement: measurement,
            decisionPolicy: policy,
            decision: decision
        )

        let markdown = SmeltOptimizerReportGenerator.markdown(
            packagePath: "/tmp/test.smeltpkg",
            manifest: makeManifest(),
            integrity: nil,
            decodeReport: structureReport(kind: "decode", names: []),
            prefillReport: structureReport(kind: "prefill", names: []),
            options: SmeltOptimizerReportOptions(planComparisons: [comparison])
        )

        XCTAssertTrue(markdown.contains("## Graph Plan Cost Evidence"))
        XCTAssertTrue(markdown.contains("Exact FP16 logits: yes"))
        XCTAssertTrue(markdown.contains("10 interleaved pairs"))
        XCTAssertTrue(markdown.contains("records 1062 → 934"))
        XCTAssertTrue(markdown.contains("max dispatch fan-out 40 → 1"))
        XCTAssertTrue(markdown.contains("`candidate` (`candidateFaster`)"))
    }
}
