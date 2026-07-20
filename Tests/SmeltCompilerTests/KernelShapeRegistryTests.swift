import Testing

@testable import SmeltCompiler

@Suite("Kernel Shape Registry")
struct KernelShapeRegistryTests {

    @Test("Decode affine shapes select tuned kernels")
    func decodeAffineShapesSelectTunedKernels() {
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 2_048,
                cols: 1_536,
                groupSize: 128
            ) == .affineMatvecC1536R2048G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 256,
                cols: 1_536,
                groupSize: 128
            ) == .affineMatvecC1536R256G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 1_536,
                cols: 256,
                groupSize: 128
            ) == .affineMatvecC256R1536G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 1_536,
                cols: 12_288,
                groupSize: 128
            ) == .affineMatvecC12288R1536G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 262_144,
                cols: 1_536,
                groupSize: 128
            ) == .affineMatvecC1536R262144G128Rows8
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 1_536,
                cols: 4_096,
                groupSize: 128
            ) == .affineMatvecC4096R1536G128
        )
        #expect(
            SmeltKernelShapeRegistry.decodeFusedGeGLUPipeline(
                rows: 6_144,
                cols: 1_536,
                groupSize: 128
            ) == .fusedAffineGateUpGeGLUC1536R6144G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeFusedGeGLUPipeline(
                rows: 12_288,
                cols: 1_536,
                groupSize: 128
            ) == .fusedAffineGateUpGeGLUC1536R12288G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 151_936,
                cols: 2_048,
                groupSize: 64
            ) == .affineMatvecC2048R151936G64Rows8
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 10_240,
                cols: 2_560,
                groupSize: 128
            ) == .affineMatvecC2560R10240G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 10_752,
                cols: 2_560,
                groupSize: 128
            ) == .affineMatvecC2560R10752G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 262_144,
                cols: 2_560,
                groupSize: 128
            ) == .affineMatvecC2560R262144G128Rows8
        )
        #expect(
            SmeltKernelShapeRegistry.decodeFusedGeGLUPipeline(
                rows: 10_240,
                cols: 2_560,
                groupSize: 128
            ) == .fusedAffineGateUpGeGLUC2560R10240G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
                rows: 2_560,
                cols: 10_240,
                groupSize: 128
            ) == .affineMatvecC10240R2560G128Rows4SG1
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormScaleAffinePipeline(
                rows: 262_144,
                cols: 1_536,
                groupSize: 128
            ) == .normScaleAffineMatvecC1536R262144G128Rows8
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormScaleAffinePipeline(
                rows: 1_024,
                cols: 256,
                groupSize: 128
            ) == .normScaleAffineMatvecC256R1024G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormScaleAffinePipeline(
                rows: 2_048,
                cols: 256,
                groupSize: 128
            ) == .normScaleAffineMatvecC256R2048G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormAddScaleAffinePipeline(
                rows: 256,
                cols: 2_560,
                groupSize: 128
            ) == .normAddScaleAffineMatvecC2560R256G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormScaleGeGLUPipeline(
                rows: 6_144,
                cols: 1_536,
                groupSize: 128
            ) == .normScaleAffineGateUpGeGLUC1536R6144G128Rows4
        )
        #expect(
            SmeltKernelShapeRegistry.decodeNormScaleGeGLUPipeline(
                rows: 12_288,
                cols: 1_536,
                groupSize: 128
            ) == .normScaleAffineGateUpGeGLUC1536R12288G128Rows4
        )
    }

    @Test("Prefill affine and KV shapes select fixed kernels")
    func prefillShapesSelectFixedKernels() {
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
                rows: 4_096,
                cols: 1_536,
                groupSize: 128
            ) == .affineMatvecC1536R4096G128Batched
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
                rows: 1_536,
                cols: 6_144,
                groupSize: 128
            ) == .affineMatvecC6144R1536G128Batched
        )
        #expect(
            SmeltKernelShapeRegistry.prefillDualAffinePipeline(
                rows: 512,
                cols: 1_536,
                groupSize: 128
            ) == .fusedDualAffineMatvecC1536R512G128Batched
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
                rows: 10_240,
                cols: 2_560,
                groupSize: 128
            ) == .affineMatvecC2560R10240G128Batched
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
                rows: 10_752,
                cols: 2_560,
                groupSize: 128
            ) == .affineMatvecC2560R10752G128Batched
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
                rows: 2_560,
                cols: 10_240,
                groupSize: 128
            ) == .affineMatvecC10240R2560G128BatchedTile4
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineBatchedBatchTile(
                .affineMatvecC10240R2560G128BatchedTile4
            ) == 4
        )
        #expect(
            SmeltKernelShapeRegistry.prefillDualAffinePipeline(
                rows: 1_024,
                cols: 2_560,
                groupSize: 128
            ) == .fusedDualAffineMatvecC2560R1024G128Batched
        )
    }

    @Test("Llama prefill FFN shapes select QMM kernels")
    func llamaPrefillFFNShapesSelectQMMKernels() {
        #expect(
            SmeltKernelShapeRegistry.prefillFusedGateUpFullPipeline(
                rows: 8_192,
                cols: 2_048,
                groupSize: 64,
                activation: .swiglu
            ) == .fusedAffineGateUpSwigluC2048R8192G64BatchedFull
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineFullPipeline(
                rows: 2_048,
                cols: 8_192,
                groupSize: 64
            ) == .affineMatvecC8192R2048G64BatchedFull
        )
        #expect(
            SmeltKernelShapeRegistry.decodeFusedFFNPipeline(
                rows: 8_192,
                cols: 3_072,
                groupSize: 64
            ) == .fusedAffineGateUpSwigluC3072R8192G64
        )
        #expect(
            SmeltKernelShapeRegistry.prefillFusedGateUpFullPipeline(
                rows: 8_192,
                cols: 3_072,
                groupSize: 64,
                activation: .swiglu
            ) == .fusedAffineGateUpSwigluC3072R8192G64BatchedFull
        )
        #expect(
            SmeltKernelShapeRegistry.prefillAffineFullPipeline(
                rows: 3_072,
                cols: 8_192,
                groupSize: 64
            ) == .affineMatvecC8192R3072G64BatchedFull
        )
        #expect(
            SmeltKernelShapeRegistry.prefillDualAffinePipeline(
                rows: 1_024,
                cols: 3_072,
                groupSize: 64
            ) == .fusedDualAffineMatvecC3072R1024G64Batched
        )
    }

    @Test("Decode FFN gate-up shape selects fixed QMM kernel")
    func decodeFFNGateUpShapeSelectsFixedQMMKernel() {
        #expect(
            SmeltKernelShapeRegistry.decodeFusedFFNPipeline(
                rows: 11_008,
                cols: 2_048,
                groupSize: 64
            ) == .fusedAffineGateUpSwigluC2048R11008G64
        )
        #expect(SmeltKernelShapeRegistry.decodeFusedFFNUsesRows4(.fusedAffineGateUpSwigluC2048R11008G64))
        #expect(SmeltKernelShapeRegistry.decodeFusedFFNThreads(.fusedAffineGateUpSwigluC2048R11008G64) == 64)
    }

    @Test("GeGLU prefill FFN shape selects generic full-QMM gate-up route")
    func geGLUPrefillFFNShapeSelectsFullQMMRoute() {
        #expect(
            SmeltKernelShapeRegistry.prefillFusedGateUpFullPipeline(
                rows: 10_240,
                cols: 2_560,
                groupSize: 128,
                activation: .geglu
            ) == .fusedAffineGateUpGeGLUC2560R10240G128BatchedFull
        )
        #expect(
            SmeltKernelShapeRegistry.prefillFusedGateUpFullPipeline(
                rows: 10_240,
                cols: 2_560,
                groupSize: 128,
                activation: .swiglu
            ) == nil
        )
    }

    @Test("RMS norm shapes select tuned pipelines")
    func rmsNormShapesSelectTunedPipelines() {
        #expect(
            SmeltKernelShapeRegistry.decodeRmsNormPipeline(dim: 1_536, eps: 1e-6)
                == .rmsNorm1PWD1536
        )
        #expect(
            SmeltKernelShapeRegistry.batchedRmsNormPipeline(dim: 1_536, eps: 1e-6)
                == .rmsNorm1PWD1536Batched
        )
        #expect(
            SmeltKernelShapeRegistry.rmsNormThreads(.rmsNorm1PWD1536) == 192
        )
        #expect(
            SmeltKernelShapeRegistry.batchedRmsNormPipeline(dim: 2_560, eps: 1e-6)
                == .rmsNorm1PWD2560Batched
        )
        #expect(
            SmeltKernelShapeRegistry.rmsNormThreads(.rmsNorm1PWD2560Batched) == 320
        )
    }
}
