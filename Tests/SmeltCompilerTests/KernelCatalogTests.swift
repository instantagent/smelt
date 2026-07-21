import Testing

@testable import SmeltCompiler

// MARK: - Kernel catalog tests

@Suite("SmeltKernelCatalog")
struct KernelCatalogTests {

    @Test("All pipelines have catalog entries")
    func allPipelinesHaveEntries() {
        let allCases = SmeltPipeline.allCases
        #expect(SmeltKernelCatalog.signatures.count == allCases.count)

        // Every pipeline rawValue maps to the correct entry
        for pipeline in allCases {
            let sig = SmeltKernelCatalog.signature(for: pipeline)
            #expect(sig.pipeline == pipeline)
        }
    }

    @Test("Signatures are indexed by rawValue")
    func signaturesIndexedByRawValue() {
        for (index, sig) in SmeltKernelCatalog.signatures.enumerated() {
            #expect(sig.pipeline.rawValue == index)
        }
    }

    @Test("Metal function names match .metal kernel names")
    func metalFunctionNamesMatchKernels() {
        // Verify each pipeline maps to the expected Metal function name
        let expectedNames: [(SmeltPipeline, String)] = [
            (.fusedLutMatvec, "fused_lut_matvec"),
            (.rmsNorm1PW, "rms_norm_1pw"),
            (.rmsNormGated, "rms_norm_gated"),
            (.conv1dUpdateSilu, "conv1d_update_silu"),
            (.l2Normalize, "l2_normalize"),
            (.computeGates, "compute_gates"),
            (.stateDecay, "state_decay"),
            (.kvMemReadout, "kv_mem_readout"),
            (.computeDelta, "compute_delta"),
            (.outerProductUpdate, "outer_product_update"),
            (.queryReadout, "query_readout"),
            (.swigluFused, "swiglu_fused"),
            (.elementwiseAdd, "elementwise_add"),
            (.embeddingGather, "embedding_gather"),
            (.argmaxFP16, "argmax_fp16"),
            (.applyRope, "apply_rope"),
            (.kvCacheUpdate, "kv_cache_update"),
            (.ropeKVCacheUpdate, "rope_kv_cache_update"),
            (.fusedAffineMatvecAddC2048R1024G64Rows4, "fused_affine_matvec_add_c2048_r1024_g64_rows4"),
            (.attentionDecode, "attention_decode"),
            (.attentionDecodeSoftcap, "attention_decode_softcap"),
            (.attentionDecodeD256H8KV2, "attention_decode_d256_h8_kv2"),
            (.attentionDecodeD256H8KV2SDPA, "attention_decode_d256_h8_kv2_sdpa"),
            (.attentionDecodeD256H16KV4, "attention_decode_d256_h16_kv4"),
            (.attentionDecodeD256H16KV4SDPA, "attention_decode_d256_h16_kv4_sdpa"),
            (.attentionDecodeD256H24KV4, "attention_decode_d256_h24_kv4"),
            (.attentionDecodeD256H24KV4SDPA, "attention_decode_d256_h24_kv4_sdpa"),
            (.attentionPrefillSDPAVectorD256, "attention_prefill_sdpa_vector_d256"),
            (.signedTernaryAffineQMMG128BM32BN32BK32, "signed_ternary_affine_qmm_g128_bm32_bn32_bk32"),
            (.attentionPrefillMLXFallbackD256, "attention_prefill_mlx_fallback_d256"),
            (.attentionDecodeMLXVectorD256, "attention_decode_mlx_vector_d256"),
            (.attentionDecodeMLXVector2Pass1D256B128, "attention_decode_mlx_vector_2pass_1_d256_b128"),
            (.attentionDecodeMLXVector2Pass2D256B128, "attention_decode_mlx_vector_2pass_2_d256_b128"),
            (.attentionDecodeD256H8KV1, "attention_decode_d256_h8_kv1"),
            (.attentionDecodeD256H8KV1SDPA, "attention_decode_d256_h8_kv1_sdpa"),
            (.attentionDecodeD512H8KV1, "attention_decode_d512_h8_kv1"),
            (.attentionDecodeD512H8KV1SDPA, "attention_decode_d512_h8_kv1_sdpa"),
            (.attentionDecodeD256H8KV1Fused, "attention_decode_d256_h8_kv1_fused"),
            (.attentionDecodeD256H8KV1FusedSoftcap, "attention_decode_d256_h8_kv1_fused_softcap"),
            (.attentionDecodeD256H8KV1FusedShared, "attention_decode_d256_h8_kv1_fused_shared"),
            (.attentionDecodeD256H8KV1FusedSharedSoftcap, "attention_decode_d256_h8_kv1_fused_shared_softcap"),
            (.attentionDecodeD512H8KV1Fused, "attention_decode_d512_h8_kv1_fused"),
            (.attentionDecodeD512H8KV1FusedSoftcap, "attention_decode_d512_h8_kv1_fused_softcap"),
            (.attentionDecodeD512H8KV1FusedShared, "attention_decode_d512_h8_kv1_fused_shared"),
            (.attentionDecodeD512H8KV1FusedSharedSoftcap, "attention_decode_d512_h8_kv1_fused_shared_softcap"),
            (.affineMatvecC1536R2048G128Rows4, "affine_matvec_c1536_r2048_g128_rows4"),
            (.affineMatvecC1536R256G128Rows4, "affine_matvec_c1536_r256_g128_rows4"),
            (.affineMatvecC1536R6144G128Rows4, "affine_matvec_c1536_r6144_g128_rows4"),
            (.affineMatvecC1536R12288G128Rows4, "affine_matvec_c1536_r12288_g128_rows4"),
            (.affineMatvecC1536R262144G128Rows4, "affine_matvec_c1536_r262144_g128_rows4"),
            (.affineMatvecC1536R262144G128Rows8, "affine_matvec_c1536_r262144_g128_rows8"),
            (.normScaleAffineMatvecC1536R2048G128Rows4, "norm_scale_affine_matvec_c1536_r2048_g128_rows4"),
            (.normScaleAffineMatvecC1536R12288G128Rows4, "norm_scale_affine_matvec_c1536_r12288_g128_rows4"),
            (.normScaleAffineMatvecC1536R262144G128Rows4, "norm_scale_affine_matvec_c1536_r262144_g128_rows4"),
            (.normScaleAffineMatvecC1536R262144G128Rows8, "norm_scale_affine_matvec_c1536_r262144_g128_rows8"),
            (.affineMatvecC256R1536G128Rows4, "affine_matvec_c256_r1536_g128_rows4"),
            (.affineMatvecC2048R1536G128Rows4, "affine_matvec_c2048_r1536_g128_rows4"),
            (.affineMatvecC6144R1536G128Rows4, "affine_matvec_c6144_r1536_g128_rows4"),
            (.affineMatvecC12288R1536G128Rows4, "affine_matvec_c12288_r1536_g128_rows4"),
            (.affineMatvecC1536R256G128Rows8, "affine_matvec_c1536_r256_g128_rows8"),
            (.fusedAffineGateUpGeGLU, "fused_affine_gate_up_geglu"),
            (.fusedAffineGateUpGeGLUC1536R6144G128Rows4, "fused_affine_gate_up_geglu_c1536_r6144_g128_rows4"),
            (.fusedAffineGateUpGeGLUC1536R12288G128Rows4, "fused_affine_gate_up_geglu_c1536_r12288_g128_rows4"),
            (.fusedAffineGateUpGeGLUC1536R6144G128Rows8, "fused_affine_gate_up_geglu_c1536_r6144_g128_rows8"),
            (.fusedAffineGateUpGeGLUC1536R12288G128Rows8, "fused_affine_gate_up_geglu_c1536_r12288_g128_rows8"),
            (.normScaleAffineGateUpGeGLUC1536R6144G128Rows4, "norm_scale_affine_gate_up_geglu_c1536_r6144_g128_rows4"),
            (.normScaleAffineGateUpGeGLUC1536R12288G128Rows4, "norm_scale_affine_gate_up_geglu_c1536_r12288_g128_rows4"),
            (.fusedAffineMatvecAddC2048R1536G128Rows4, "fused_affine_matvec_add_c2048_r1536_g128_rows4"),
            (.fusedAffineMatvecAddC4096R1536G128, "fused_affine_matvec_add_c4096_r1536_g128"),
            (.fusedAffineMatvecAddC6144R1536G128Rows4, "fused_affine_matvec_add_c6144_r1536_g128_rows4"),
            (.fusedAffineMatvecAddC12288R1536G128Rows4, "fused_affine_matvec_add_c12288_r1536_g128_rows4"),
            (.affineMatvecC1024R2048G64Rows4, "affine_matvec_c1024_r2048_g64_rows4"),
            (.affineMatvecC1024R6144G64Rows4, "affine_matvec_c1024_r6144_g64_rows4"),
            (.affineMatvecC2048R1024G64Rows4, "affine_matvec_c2048_r1024_g64_rows4"),
            (.affineMatvecC3584R1024G64Rows4, "affine_matvec_c3584_r1024_g64_rows4"),
            (.affineMatvecC1024R248320G64Rows4, "affine_matvec_c1024_r248320_g64_rows4"),
            (.fusedAffineGateUpSwigluC1024R3584G64Rows4, "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4"),
            (.rmsNorm1PWD256Add, "rms_norm_1pw_d256_add"),
            (.rmsNorm1PWD256AddScalarWeight, "rms_norm_1pw_d256_add_scalar_weight"),
            (.normScaleAffineMatvecC256R1024G128Rows4, "norm_scale_affine_matvec_c256_r1024_g128_rows4"),
            (.normScaleAffineMatvecC256R2048G128Rows4, "norm_scale_affine_matvec_c256_r2048_g128_rows4"),
            (.rmsNorm1PWD1536Add, "rms_norm_1pw_d1536_add"),
            (.rmsNorm1PWD1536AddBatched, "rms_norm_1pw_d1536_add_batched"),
            (.rmsNormScaleOnlyD1536, "rms_norm_scale_only_d1536"),
            (.gegluFused, "geglu_fused"),
            (.logitCap, "logit_cap"),
            (.scalarMul, "scalar_mul"),
            (.rmsNorm1PWD2048, "rms_norm_1pw_d2048"),
            (.fusedAffineGateUpSwigluC2048R6144G64, "fused_affine_gate_up_swiglu_c2048_r6144_g64"),
            (.fusedAffineGateUpSwigluC2048R11008G64, "fused_affine_gate_up_swiglu_c2048_r11008_g64"),
            (.fusedAffineMatvecAddC2048R2048G64, "fused_affine_matvec_add_c2048_r2048_g64"),
            (.fusedAffineMatvecAddC6144R2048G64, "fused_affine_matvec_add_c6144_r2048_g64"),
            (.affineMatvecC2048R2048G64, "affine_matvec_c2048_r2048_g64"),
            (.affineMatvecC2048R6144G64, "affine_matvec_c2048_r6144_g64"),
            (.affineMatvecC2048R4096G64, "affine_matvec_c2048_r4096_g64"),
            (.affineMatvecC2048R512G64, "affine_matvec_c2048_r512_g64"),
            (.affineMatvecC6144R2048G64, "affine_matvec_c6144_r2048_g64"),
            (.affineMatvecC2048R151936G64Rows8, "affine_matvec_c2048_r151936_g64_rows8"),
            (.affineMatvecC2048R248320G64, "affine_matvec_c2048_r248320_g64"),
            (.fusedDualAffineMatvecC2048R16G64, "fused_dual_affine_matvec_c2048_r16_g64"),
            (.fusedDualAffineMatvecC1024R16G64Rows4, "fused_dual_affine_matvec_c1024_r16_g64_rows4"),
            (.fusedDualAffineMatvecC2048R16G64Batched, "fused_dual_affine_matvec_c2048_r16_g64_batched"),
            (.fusedAffineGateUpSwigluC2048R6144G64Batched, "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched"),
            (.affineMatvecC2048R2048G64Batched, "affine_matvec_c2048_r2048_g64_batched"),
            (.affineMatvecC2048R6144G64Batched, "affine_matvec_c2048_r6144_g64_batched"),
            (.affineMatvecC2048R4096G64Batched, "affine_matvec_c2048_r4096_g64_batched"),
            (.affineMatvecC2048R512G64Batched, "affine_matvec_c2048_r512_g64_batched"),
            (.affineMatvecC6144R2048G64Batched, "affine_matvec_c6144_r2048_g64_batched"),
            (.fusedDualAffineMatvecC2048R512G64Batched, "fused_dual_affine_matvec_c2048_r512_g64_batched"),
            (.conv1dUpdateSilu6144x4Prefill, "conv1d_update_silu_c6144_k4_prefill"),
            (.l2NormalizeQD128C6144H16Prefill, "l2_normalize_q_d128_c6144_h16_prefill"),
            (.l2NormalizeKD128C6144H16Prefill, "l2_normalize_k_d128_c6144_h16_prefill"),
            (.deltanetRecurrenceMlxPrefillD128H16, "deltanet_recurrence_mlx_prefill_d128_h16"),
            (.deltanetRecurrenceMlxDecodeD128H32QK16, "deltanet_recurrence_mlx_decode_d128_h32_qk16"),
            (.deltanetRecurrenceMlxPrefillD128H32QK16, "deltanet_recurrence_mlx_prefill_d128_h32_qk16"),
            (.deltanetRecurrenceMlxDecodeD128H48QK16, "deltanet_recurrence_mlx_decode_d128_h48_qk16"),
            (.deltanetRecurrenceMlxPrefillD128H48QK16, "deltanet_recurrence_mlx_prefill_d128_h48_qk16"),
            (.conv1dUpdateSiluPrefill, "conv1d_update_silu_prefill"),
            (.l2NormalizeQPrefill, "l2_normalize_q_prefill"),
            (.l2NormalizeKPrefill, "l2_normalize_k_prefill"),
            (.deltanetRecurrenceMlxPrefill, "deltanet_recurrence_mlx_prefill"),
            (.conv1dUpdateSiluPrefillCheckpoint, "conv1d_update_silu_prefill_checkpoint"),
            (.deltanetRecurrenceMlxPrefillCheckpoint, "deltanet_recurrence_mlx_prefill_checkpoint"),
            (.rmsNormGatedD128Batched, "rms_norm_gated_d128_batched"),
            (.rmsNorm1PWD1024Batched, "rms_norm_1pw_d1024_batched"),
            (.fusedAffineGateUpSwigluC2048R6144G64BatchedFull, "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"),
            (.affineMatvecC2048R2048G64BatchedFull, "affine_matvec_c2048_r2048_g64_batched_full"),
            (.affineMatvecC2048R6144G64BatchedFull, "affine_matvec_c2048_r6144_g64_batched_full"),
            (.affineMatvecC2048R4096G64BatchedFull, "affine_matvec_c2048_r4096_g64_batched_full"),
            (.affineMatvecC2048R512G64BatchedFull, "affine_matvec_c2048_r512_g64_batched_full"),
            (.affineMatvecC6144R2048G64BatchedFull, "affine_matvec_c6144_r2048_g64_batched_full"),
            (.fusedAffineGateUpSwigluC2048R8192G64BatchedFull, "fused_affine_gate_up_swiglu_c2048_r8192_g64_batched_full"),
            (.affineMatvecC8192R2048G64BatchedFull, "affine_matvec_c8192_r2048_g64_batched_full"),
            (.fusedAffineMatvecAddC2048R1024G64BatchedFull, "fused_affine_matvec_add_c2048_r1024_g64_batched_full"),
            (.fusedAffineMatvecAddC3584R1024G64BatchedFull, "fused_affine_matvec_add_c3584_r1024_g64_batched_full"),
            (.fusedAffineMatvecAddC2048R1024G64BatchedFullB8, "fused_affine_matvec_add_c2048_r1024_g64_batched_full_b8"),
            (.fusedAffineMatvecAddC3584R1024G64BatchedFullB8, "fused_affine_matvec_add_c3584_r1024_g64_batched_full_b8"),
            (.fusedAffineMatvecAddC2048R2048G64BatchedFull, "fused_affine_matvec_add_c2048_r2048_g64_batched_full"),
            (.fusedAffineMatvecAddC6144R2048G64BatchedFull, "fused_affine_matvec_add_c6144_r2048_g64_batched_full"),
            (.fusedAffineMatvecAddC8192R2048G64BatchedFull, "fused_affine_matvec_add_c8192_r2048_g64_batched_full"),
            (.fusedAffineMatvecAddC4096R2560G64BatchedFull, "fused_affine_matvec_add_c4096_r2560_g64_batched_full"),
            (.fusedAffineMatvecAddC9216R2560G64BatchedFull, "fused_affine_matvec_add_c9216_r2560_g64_batched_full"),
            (.rmsNormScaleOnlyD1024Batched, "rms_norm_scale_only_d1024_batched"),
            (.rmsNormScaleOnlyD2048Batched, "rms_norm_scale_only_d2048_batched"),
            (.rmsNormScaleOnlyD2560Batched, "rms_norm_scale_only_d2560_batched"),
            (.normScaleAffineMatvecC1024R6144G64BatchedFull, "norm_scale_affine_matvec_c1024_r6144_g64_batched_full"),
            (.normScaleAffineMatvecC1024R4096G64BatchedFull, "norm_scale_affine_matvec_c1024_r4096_g64_batched_full"),
            (.normScaleAffineMatvecC2048R2048G64BatchedFull, "norm_scale_affine_matvec_c2048_r2048_g64_batched_full"),
            (.normScaleAffineMatvecC2048R6144G64BatchedFull, "norm_scale_affine_matvec_c2048_r6144_g64_batched_full"),
            (.normScaleAffineMatvecC2048R4096G64BatchedFull, "norm_scale_affine_matvec_c2048_r4096_g64_batched_full"),
            (.normScaleAffineMatvecC2560R8192G64BatchedFull, "norm_scale_affine_matvec_c2560_r8192_g64_batched_full"),
            (.normScaleAffineMatvecC2560R10240G128Batched, "norm_scale_affine_matvec_c2560_r10240_g128_batched"),
            (.normScaleFusedDualAffineMatvecC2560R512G128Batched, "norm_scale_fused_dual_affine_matvec_c2560_r512_g128_batched"),
            (.normScaleFusedDualAffineMatvecC2560R1024G128Batched, "norm_scale_fused_dual_affine_matvec_c2560_r1024_g128_batched"),
            (.fusedNormRopeAndKvCachePrefill, "fused_norm_rope_and_kv_cache_prefill"),
            (.affineMatvecC10240R2560G128BatchedTile4, "affine_matvec_c10240_r2560_g128_batched_tile4"),
            (.fusedAffineGateUpGeGLUC2560R10240G128BatchedFull, "fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full"),
            (.affineMatvecC10240R2560G128Rows4SG1, "affine_matvec_c10240_r2560_g128_rows4_sg1"),
            (.normScaleAffineGateUpSwigluC1024R3584G64BatchedFull, "norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"),
            (.normScaleAffineGateUpSwigluC2048R6144G64BatchedFull, "norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"),
            (.normScaleAffineGateUpSwigluC2048R8192G64BatchedFull, "norm_scale_affine_gate_up_swiglu_c2048_r8192_g64_batched_full"),
            (.normScaleAffineGateUpSwigluC2560R9216G64BatchedFull, "norm_scale_affine_gate_up_swiglu_c2560_r9216_g64_batched_full"),
            (.rmsNormScaleOnlyD2048Eps1e5Batched, "rms_norm_scale_only_d2048_eps1e5_batched"),
            (.fusedAffineGateUpSwigluC3072R8192G64, "fused_affine_gate_up_swiglu_c3072_r8192_g64"),
            (.affineMatvecC3072R3072G64, "affine_matvec_c3072_r3072_g64"),
            (.affineMatvecC3072R1024G64, "affine_matvec_c3072_r1024_g64"),
            (.affineMatvecC3072R8192G64, "affine_matvec_c3072_r8192_g64"),
            (.affineMatvecC8192R3072G64, "affine_matvec_c8192_r3072_g64"),
            (.fusedDualAffineMatvecC3072R1024G64Batched, "fused_dual_affine_matvec_c3072_r1024_g64_batched"),
            (.fusedAffineGateUpSwigluC3072R8192G64BatchedFull, "fused_affine_gate_up_swiglu_c3072_r8192_g64_batched_full"),
            (.affineMatvecC3072R3072G64BatchedFull, "affine_matvec_c3072_r3072_g64_batched_full"),
            (.affineMatvecC3072R1024G64BatchedFull, "affine_matvec_c3072_r1024_g64_batched_full"),
            (.affineMatvecC8192R3072G64BatchedFull, "affine_matvec_c8192_r3072_g64_batched_full"),
            (.fusedAffineMatvecAddC3072R3072G64BatchedFull, "fused_affine_matvec_add_c3072_r3072_g64_batched_full"),
            (.fusedAffineMatvecAddC8192R3072G64BatchedFull, "fused_affine_matvec_add_c8192_r3072_g64_batched_full"),
            (.normScaleAffineMatvecC3072R3072G64BatchedFull, "norm_scale_affine_matvec_c3072_r3072_g64_batched_full"),
            (.normScaleAffineGateUpSwigluC3072R8192G64BatchedFull, "norm_scale_affine_gate_up_swiglu_c3072_r8192_g64_batched_full"),
            (.rmsNormScaleOnlyD3072Eps1e5Batched, "rms_norm_scale_only_d3072_eps1e5_batched"),
            (.sigmoidKernel, "sigmoid_kernel"),
            (.elementwiseMul, "elementwise_mul"),
            (.bufferCopy, "buffer_copy"),
            (.fp16Matvec, "fp16_matvec"),
            (.gateSplit, "gate_split"),
            (.perHeadRmsNorm1PW, "per_head_rms_norm_1pw"),
            (.perHeadRmsNorm1PWBatched, "per_head_rms_norm_1pw_batched"),
            (.perHeadRmsNorm, "per_head_rms_norm"),
            (.perHeadRmsNormBatched, "per_head_rms_norm_batched"),
            (.perHeadRmsNormNoScale, "per_head_rms_norm_noscale"),
            (.perHeadRmsNormNoScaleBatched, "per_head_rms_norm_noscale_batched"),
            (.lutEmbeddingGather, "lut_embedding_gather"),
            (.affineEmbeddingGatherBatched, "affine_embedding_gather_batched"),
            (.fusedLutMatmul, "fused_lut_matmul"),
            (.rmsNorm1PWBatched, "rms_norm_1pw_batched"),
            (.embeddingGatherBatched, "embedding_gather_batched"),
            (.attentionPrefill, "attention_prefill"),
            (.attentionPrefillSoftcap, "attention_prefill_softcap"),
            (.sigmoidMul, "sigmoid_mul"),
            (.scalarMulWeight, "scalar_mul_weight"),
            (.rmsNorm1PWFromFP32Batched, "rms_norm_1pw_from_fp32_batched"),
            (.projectionBiasAddBatched, "projection_bias_add_batched"),
            (.attentionDecodeD128H16KV2SDPA, "attention_decode_d128_h16_kv2_sdpa"),
            (.argmaxFP16Partials, "argmax_fp16_partials"),
            (.argmaxKeyReduce, "argmax_key_reduce"),
        ]

        for (pipeline, expectedName) in expectedNames {
            let sig = SmeltKernelCatalog.signature(for: pipeline)
            #expect(sig.metalFunctionName == expectedName)
        }
    }

    @Test("Validation passes for correct counts")
    func validationPassesForCorrectCounts() throws {
        for sig in SmeltKernelCatalog.signatures {
            try SmeltKernelCatalog.validate(
                pipeline: sig.pipeline,
                bufferCount: sig.bufferBindingCount,
                constantCount: sig.constantCount
            )
        }
    }

    @Test("Validation fails for wrong buffer count")
    func validationFailsForWrongBufferCount() {
        #expect(throws: SmeltCatalogError.self) {
            try SmeltKernelCatalog.validate(
                pipeline: .fusedLutMatvec,
                bufferCount: 3,
                constantCount: 2
            )
        }
    }

    @Test("Validation fails for wrong constant count")
    func validationFailsForWrongConstantCount() {
        #expect(throws: SmeltCatalogError.self) {
            try SmeltKernelCatalog.validate(
                pipeline: .attentionDecode,
                bufferCount: 5,
                constantCount: 5
            )
        }
    }

    @Test("Pipeline names list matches signatures order")
    func pipelineNamesMatchOrder() {
        let names = SmeltKernelCatalog.pipelineNames
        #expect(names.count == SmeltKernelCatalog.signatures.count)
        for (index, name) in names.enumerated() {
            #expect(name == SmeltKernelCatalog.signatures[index].metalFunctionName)
        }
    }

    @Test("Pipeline name lookup matches signatures order")
    func pipelineNameLookupMatchesOrder() {
        for (index, signature) in SmeltKernelCatalog.signatures.enumerated() {
            #expect(SmeltKernelCatalog.pipelineIndex(named: signature.metalFunctionName) == index)
        }
        #expect(SmeltKernelCatalog.pipelineIndex(named: "__not_a_catalog_pipeline__") == nil)
    }

    @Test("Known binding counts for key kernels")
    func knownBindingCounts() {
        // attentionDecode is the most complex generic decode kernel: 5 buffers + 6 constants
        let attn = SmeltKernelCatalog.signature(for: .attentionDecode)
        #expect(attn.bufferBindingCount == 5)
        #expect(attn.constantCount == 6)

        let attnSoftcap = SmeltKernelCatalog.signature(for: .attentionDecodeSoftcap)
        #expect(attnSoftcap.bufferBindingCount == 5)
        #expect(attnSoftcap.constantCount == 7)

        let fusedSliding = SmeltKernelCatalog.signature(for: .attentionDecodeD256H8KV1Fused)
        #expect(fusedSliding.bufferBindingCount == 7)
        #expect(fusedSliding.constantCount == 3)

        let fusedSlidingSoftcap = SmeltKernelCatalog.signature(
            for: .attentionDecodeD256H8KV1FusedSoftcap
        )
        #expect(fusedSlidingSoftcap.bufferBindingCount == 7)
        #expect(fusedSlidingSoftcap.constantCount == 4)

        let fusedGlobal = SmeltKernelCatalog.signature(for: .attentionDecodeD512H8KV1Fused)
        #expect(fusedGlobal.bufferBindingCount == 7)
        #expect(fusedGlobal.constantCount == 3)

        let fusedGlobalSoftcap = SmeltKernelCatalog.signature(
            for: .attentionDecodeD512H8KV1FusedSoftcap
        )
        #expect(fusedGlobalSoftcap.bufferBindingCount == 7)
        #expect(fusedGlobalSoftcap.constantCount == 4)

        let attnPrefillSoftcap = SmeltKernelCatalog.signature(for: .attentionPrefillSoftcap)
        #expect(attnPrefillSoftcap.bufferBindingCount == 4)
        #expect(attnPrefillSoftcap.constantCount == 8)

        let perHeadNormBatched = SmeltKernelCatalog.signature(for: .perHeadRmsNormBatched)
        #expect(perHeadNormBatched.bufferBindingCount == 3)
        #expect(perHeadNormBatched.constantCount == 3)

        let perHeadNormNoScaleBatched = SmeltKernelCatalog.signature(for: .perHeadRmsNormNoScaleBatched)
        #expect(perHeadNormNoScaleBatched.bufferBindingCount == 1)
        #expect(perHeadNormNoScaleBatched.constantCount == 3)

        let scalarMul = SmeltKernelCatalog.signature(for: .scalarMul)
        #expect(scalarMul.bufferBindingCount == 2)
        #expect(scalarMul.constantCount == 2)

        let scalarMulWeight = SmeltKernelCatalog.signature(for: .scalarMulWeight)
        #expect(scalarMulWeight.bufferBindingCount == 3)
        #expect(scalarMulWeight.constantCount == 1)

        let fp32NormBatched = SmeltKernelCatalog.signature(for: .rmsNorm1PWFromFP32Batched)
        #expect(fp32NormBatched.bufferBindingCount == 3)
        #expect(fp32NormBatched.constantCount == 2)

        let projectionBiasAdd = SmeltKernelCatalog.signature(for: .projectionBiasAddBatched)
        #expect(projectionBiasAdd.bufferBindingCount == 3)
        #expect(projectionBiasAdd.constantCount == 1)

        let d128H16KV2SDPA = SmeltKernelCatalog.signature(for: .attentionDecodeD128H16KV2SDPA)
        #expect(d128H16KV2SDPA.bufferBindingCount == 3)
        #expect(d128H16KV2SDPA.constantCount == 2)

        let ropeKV = SmeltKernelCatalog.signature(for: .ropeKVCacheUpdate)
        #expect(ropeKV.bufferBindingCount == 4)
        #expect(ropeKV.constantCount == 8)

        let vibeGateUp = SmeltKernelCatalog.signature(for: .fusedAffineGateUpSwigluC2048R11008G64)
        #expect(vibeGateUp.bufferBindingCount == 8)
        #expect(vibeGateUp.constantCount == 0)

        let vibeLMHead = SmeltKernelCatalog.signature(for: .affineMatvecC2048R151936G64Rows8)
        #expect(vibeLMHead.bufferBindingCount == 5)
        #expect(vibeLMHead.constantCount == 0)

        let argmaxPartials = SmeltKernelCatalog.signature(for: .argmaxFP16Partials)
        #expect(argmaxPartials.bufferBindingCount == 2)
        #expect(argmaxPartials.constantCount == 2)

        let argmaxReduce = SmeltKernelCatalog.signature(for: .argmaxKeyReduce)
        #expect(argmaxReduce.bufferBindingCount == 2)
        #expect(argmaxReduce.constantCount == 1)

        // l2Normalize is in-place: 1 buffer + 2 constants
        let l2 = SmeltKernelCatalog.signature(for: .l2Normalize)
        #expect(l2.bufferBindingCount == 1)
        #expect(l2.constantCount == 2)

        // computeGates has the most buffer bindings: 6 buffers + 1 constant
        let gates = SmeltKernelCatalog.signature(for: .computeGates)
        #expect(gates.bufferBindingCount == 6)
        #expect(gates.constantCount == 1)
    }
}
