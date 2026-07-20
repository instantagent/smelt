// SmeltKernelShapeRegistry - Centralized hot-shape lookup for tuned kernels.
//
// This registry covers tuned model shape routes without
// duplicating shape switches across the emitters.

import Foundation

enum SmeltKernelShapeRegistry {
    static func decodeAffinePipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        // Qwen 3.5 0.8B
        case (2_048, 1_024, 64): .affineMatvecC1024R2048G64
        case (3_584, 1_024, 64): .affineMatvecC1024R3584G64
        case (4_096, 1_024, 64): .affineMatvecC1024R4096G64
        case (512, 1_024, 64): .affineMatvecC1024R512G64
        case (6_144, 1_024, 64): .affineMatvecC1024R6144G64
        case (1_024, 2_048, 64): .affineMatvecC2048R1024G64
        case (1_024, 3_584, 64): .affineMatvecC3584R1024G64
        case (248_320, 1_024, 64): .affineMatvecC1024R248320G64

        // Qwen 3.5 2B
        case (2_048, 2_048, 64): .affineMatvecC2048R2048G64
        case (6_144, 2_048, 64): .affineMatvecC2048R6144G64
        case (4_096, 2_048, 64): .affineMatvecC2048R4096G64
        case (512, 2_048, 64): .affineMatvecC2048R512G64
        case (2_048, 6_144, 64): .affineMatvecC6144R2048G64
        case (151_936, 2_048, 64): .affineMatvecC2048R151936G64Rows8
        case (248_320, 2_048, 64): .affineMatvecC2048R248320G64

        // Qwen 3.5 4B
        case (8_192, 2_560, 64): .affineMatvecC2560R8192G64
        case (4_096, 2_560, 64): .affineMatvecC2560R4096G64
        case (1_024, 2_560, 64): .affineMatvecC2560R1024G64
        case (2_560, 4_096, 64): .affineMatvecC4096R2560G64
        case (2_560, 9_216, 64): .affineMatvecC9216R2560G64
        case (248_320, 2_560, 64): .affineMatvecC2560R248320G64

        // 1536-col shapes
        case (2_048, 1_536, 128): .affineMatvecC1536R2048G128
        case (4_096, 1_536, 128): .affineMatvecC1536R4096G128
        case (256, 1_536, 128): .affineMatvecC1536R256G128
        case (512, 1_536, 128): .affineMatvecC1536R512G128
        case (6_144, 1_536, 128): .affineMatvecC1536R6144G128
        case (12_288, 1_536, 128): .affineMatvecC1536R12288G128
        case (1_536, 2_048, 128): .affineMatvecC2048R1536G128
        case (1_536, 4_096, 128): .affineMatvecC4096R1536G128
        case (1_536, 6_144, 128): .affineMatvecC6144R1536G128
        case (1_536, 12_288, 128): .affineMatvecC12288R1536G128

        // hidden 3072, group 64
        case (3_072, 3_072, 64): .affineMatvecC3072R3072G64
        case (1_024, 3_072, 64): .affineMatvecC3072R1024G64
        case (8_192, 3_072, 64): .affineMatvecC3072R8192G64
        case (3_072, 8_192, 64): .affineMatvecC8192R3072G64

        default:
            nil
        }
    }

    static func decodeAffineDecodePipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (2_048, 1_024, 64): .affineMatvecC1024R2048G64Rows4
        case (6_144, 1_024, 64): .affineMatvecC1024R6144G64Rows4
        case (1_024, 2_048, 64): .affineMatvecC2048R1024G64Rows4
        case (1_024, 3_584, 64): .affineMatvecC3584R1024G64Rows4
        case (2_048, 1_536, 128): .affineMatvecC1536R2048G128Rows4
        case (256, 1_536, 128): .affineMatvecC1536R256G128Rows4
        case (6_144, 1_536, 128): .affineMatvecC1536R6144G128Rows4
        case (12_288, 1_536, 128): .affineMatvecC1536R12288G128Rows4
        case (262_144, 1_536, 128): .affineMatvecC1536R262144G128Rows8
        case (1_536, 256, 128): .affineMatvecC256R1536G128Rows4
        case (1_536, 2_048, 128): .affineMatvecC2048R1536G128Rows4
        case (1_536, 6_144, 128): .affineMatvecC6144R1536G128Rows4
        case (1_536, 12_288, 128): .affineMatvecC12288R1536G128Rows4
        case (248_320, 1_024, 64): .affineMatvecC1024R248320G64Rows4
        case (3_072, 3_072, 64): .affineMatvecC3072R3072G64
        case (2_048, 2_560, 128): .affineMatvecC2560R2048G128Rows4
        case (4_096, 2_560, 128): .affineMatvecC2560R4096G128Rows4
        case (512, 2_560, 128): .affineMatvecC2560R512G128Rows4
        case (1_024, 2_560, 128): .affineMatvecC2560R1024G128Rows4
        case (10_240, 2_560, 128): .affineMatvecC2560R10240G128Rows4
        case (10_752, 2_560, 128): .affineMatvecC2560R10752G128Rows4
        case (2_560, 2_048, 128): .affineMatvecC2048R2560G128Rows4
        case (2_560, 4_096, 128): .affineMatvecC4096R2560G128Rows4
        case (2_560, 10_240, 128): .affineMatvecC10240R2560G128Rows4SG1
        case (256, 2_560, 128): .affineMatvecC2560R256G128Rows4
        case (2_560, 256, 128): .affineMatvecC256R2560G128Rows4
        case (262_144, 2_560, 128): .affineMatvecC2560R262144G128Rows8

        // Draft-model shapes
        case (1_024, 256, 128): .affineMatvecC256R1024G128Rows4
        case (2_048, 256, 128): .affineMatvecC256R2048G128Rows4
        case (256, 1_024, 128): .affineMatvecC1024R256G128Rows4
        case (256, 2_048, 128): .affineMatvecC2048R256G128Rows4

        default:
            decodeAffinePipeline(rows: rows, cols: cols, groupSize: groupSize)
        }
    }

    static func decodeNormScaleAffinePipeline(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (2_048, 1_536, 128): .normScaleAffineMatvecC1536R2048G128Rows4
        case (12_288, 1_536, 128): .normScaleAffineMatvecC1536R12288G128Rows4
        case (262_144, 1_536, 128): .normScaleAffineMatvecC1536R262144G128Rows8
        case (1_024, 256, 128): .normScaleAffineMatvecC256R1024G128Rows4
        case (2_048, 256, 128): .normScaleAffineMatvecC256R2048G128Rows4
        default: nil
        }
    }

    static func decodeNormAddScaleAffinePipeline(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (256, 2_560, 128): .normAddScaleAffineMatvecC2560R256G128Rows4
        default: nil
        }
    }

    static func decodeNormScaleAffineUsesRows4(_ pipeline: SmeltPipeline?) -> Bool {
        switch pipeline {
        case .normScaleAffineMatvecC1536R2048G128Rows4,
             .normScaleAffineMatvecC1536R12288G128Rows4,
             .normScaleAffineMatvecC1536R262144G128Rows4,
             .normScaleAffineMatvecC256R1024G128Rows4,
             .normScaleAffineMatvecC256R2048G128Rows4:
            true
        default:
            false
        }
    }

    static func decodeDualAffinePipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (16, 1_024, 64): .fusedDualAffineMatvecC1024R16G64Rows4
        case (16, 2_048, 64): .fusedDualAffineMatvecC2048R16G64
        case (32, 2_560, 64): .fusedDualAffineMatvecC2560R32G64
        default: nil
        }
    }

    static func decodeDualAffineRowTile(_ pipeline: SmeltPipeline?) -> Int {
        pipeline == .fusedDualAffineMatvecC1024R16G64Rows4 ? 4 : 8
    }

    static func decodeFusedFFNPipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (3_584, 1_024, 64): .fusedAffineGateUpSwigluC1024R3584G64Rows4
        case (6_144, 2_048, 64): .fusedAffineGateUpSwigluC2048R6144G64
        case (11_008, 2_048, 64): .fusedAffineGateUpSwigluC2048R11008G64
        case (9_216, 2_560, 64): .fusedAffineGateUpSwigluC2560R9216G64
        case (8_192, 3_072, 64): .fusedAffineGateUpSwigluC3072R8192G64
        default: nil
        }
    }

    static func decodeFusedGeGLUPipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (6_144, 1_536, 128): .fusedAffineGateUpGeGLUC1536R6144G128Rows4
        case (12_288, 1_536, 128): .fusedAffineGateUpGeGLUC1536R12288G128Rows4
        case (10_240, 2_560, 128): .fusedAffineGateUpGeGLUC2560R10240G128Rows4
        case (2_048, 256, 128): .fusedAffineGateUpGeGLUC256R2048G128Rows4
        default: nil
        }
    }

    static func decodeNormScaleGeGLUPipeline(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (6_144, 1_536, 128): .normScaleAffineGateUpGeGLUC1536R6144G128Rows4
        case (12_288, 1_536, 128): .normScaleAffineGateUpGeGLUC1536R12288G128Rows4
        case (10_240, 2_560, 128): .normScaleAffineGateUpGeGLUC2560R10240G128Rows4
        case (2_048, 256, 128): .normScaleAffineGateUpGeGLUC256R2048G128Rows4
        default: nil
        }
    }

    static func prefillAffineFullPipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        // Qwen 3.5 0.8B
        case (2_048, 1_024, 64): .affineMatvecC1024R2048G64BatchedFull
        case (3_584, 1_024, 64): .affineMatvecC1024R3584G64BatchedFull
        case (4_096, 1_024, 64): .affineMatvecC1024R4096G64BatchedFull
        case (512, 1_024, 64): .affineMatvecC1024R512G64BatchedFull
        case (6_144, 1_024, 64): .affineMatvecC1024R6144G64BatchedFull
        case (1_024, 2_048, 64): .affineMatvecC2048R1024G64BatchedFull
        case (1_024, 3_584, 64): .affineMatvecC3584R1024G64BatchedFull

        // Qwen 3.5 2B
        case (2_048, 2_048, 64): .affineMatvecC2048R2048G64BatchedFull
        case (6_144, 2_048, 64): .affineMatvecC2048R6144G64BatchedFull
        case (4_096, 2_048, 64): .affineMatvecC2048R4096G64BatchedFull
        case (512, 2_048, 64): .affineMatvecC2048R512G64BatchedFull
        case (2_048, 6_144, 64): .affineMatvecC6144R2048G64BatchedFull
        case (2_048, 8_192, 64): .affineMatvecC8192R2048G64BatchedFull

        // Qwen 3.5 4B
        case (8_192, 2_560, 64): .affineMatvecC2560R8192G64BatchedFull
        case (4_096, 2_560, 64): .affineMatvecC2560R4096G64BatchedFull
        case (1_024, 2_560, 64): .affineMatvecC2560R1024G64BatchedFull
        case (2_560, 4_096, 64): .affineMatvecC4096R2560G64BatchedFull
        case (2_560, 9_216, 64): .affineMatvecC9216R2560G64BatchedFull

        // hidden 3072, group 64
        case (3_072, 3_072, 64): .affineMatvecC3072R3072G64BatchedFull
        case (1_024, 3_072, 64): .affineMatvecC3072R1024G64BatchedFull
        case (3_072, 8_192, 64): .affineMatvecC8192R3072G64BatchedFull

        default:
            nil
        }
    }

    static func prefillAffineFullUsesQMM(_ pipeline: SmeltPipeline?) -> Bool {
        switch pipeline {
        case .affineMatvecC1024R2048G64BatchedFull,
             .affineMatvecC1024R3584G64BatchedFull,
             .affineMatvecC1024R4096G64BatchedFull,
             .affineMatvecC1024R512G64BatchedFull,
             .affineMatvecC1024R6144G64BatchedFull,
             .affineMatvecC2048R1024G64BatchedFull,
             .affineMatvecC3584R1024G64BatchedFull,
             .affineMatvecC2048R2048G64BatchedFull,
             .affineMatvecC2048R6144G64BatchedFull,
             .affineMatvecC2048R4096G64BatchedFull,
             .affineMatvecC2048R512G64BatchedFull,
             .affineMatvecC6144R2048G64BatchedFull,
             .affineMatvecC8192R2048G64BatchedFull,
             .affineMatvecC2560R8192G64BatchedFull,
             .affineMatvecC2560R4096G64BatchedFull,
             .affineMatvecC2560R1024G64BatchedFull,
             .affineMatvecC4096R2560G64BatchedFull,
             .affineMatvecC9216R2560G64BatchedFull,
             .affineMatvecC3072R3072G64BatchedFull,
             .affineMatvecC3072R1024G64BatchedFull,
             .affineMatvecC8192R3072G64BatchedFull,
             .affineMatvecC1536R2048G128BatchedFull,
             .affineMatvecC1536R4096G128BatchedFull,
             .affineMatvecC1536R6144G128BatchedFull,
             .affineMatvecC1536R12288G128BatchedFull,
             .affineMatvecC2048R1536G128BatchedFull,
             .affineMatvecC4096R1536G128BatchedFull,
             .affineMatvecC6144R1536G128BatchedFull,
             .affineMatvecC12288R1536G128BatchedFull:
            true
        default:
            false
        }
    }

    static func prefillAffineFullRowTile(_ pipeline: SmeltPipeline?) -> Int {
        prefillAffineFullUsesQMM(pipeline) ? 32 : 8
    }

    static func prefillAffineFullBatchTile(_ pipeline: SmeltPipeline?) -> Int {
        prefillAffineFullUsesQMM(pipeline) ? 16 : 8
    }

    static func prefillAffineFullThreads(_ pipeline: SmeltPipeline?) -> Int {
        prefillAffineFullUsesQMM(pipeline) ? 128 : 64
    }

    static func prefillDualAffinePipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        // Delta A/B
        case (16, 1_024, 64): .fusedDualAffineMatvecC1024R16G64Batched
        case (16, 2_048, 64): .fusedDualAffineMatvecC2048R16G64Batched
        case (32, 2_560, 64): .fusedDualAffineMatvecC2560R32G64Batched

        // Attention K/V
        case (512, 1_024, 64): .fusedDualAffineMatvecC1024R512G64Batched
        case (512, 2_048, 64): .fusedDualAffineMatvecC2048R512G64Batched
        case (1_024, 2_560, 64): .fusedDualAffineMatvecC2560R1024G64Batched
        case (1_024, 3_072, 64): .fusedDualAffineMatvecC3072R1024G64Batched
        case (256, 1_536, 128): .fusedDualAffineMatvecC1536R256G128Batched
        case (512, 1_536, 128): .fusedDualAffineMatvecC1536R512G128Batched
        case (512, 2_560, 128): .fusedDualAffineMatvecC2560R512G128Batched
        case (1_024, 2_560, 128): .fusedDualAffineMatvecC2560R1024G128Batched

        default:
            nil
        }
    }

    static func prefillAffineBatchedPipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        // 1536-col shapes
        case (2_048, 1_536, 128): .affineMatvecC1536R2048G128Batched
        case (4_096, 1_536, 128): .affineMatvecC1536R4096G128Batched
        case (6_144, 1_536, 128): .affineMatvecC1536R6144G128Batched
        case (12_288, 1_536, 128): .affineMatvecC1536R12288G128Batched
        case (1_536, 2_048, 128): .affineMatvecC2048R1536G128Batched
        case (1_536, 4_096, 128): .affineMatvecC4096R1536G128Batched
        case (1_536, 6_144, 128): .affineMatvecC6144R1536G128Batched
        case (1_536, 12_288, 128): .affineMatvecC12288R1536G128Batched
        // 2560-col shapes
        case (2_048, 2_560, 128): .affineMatvecC2560R2048G128Batched
        case (4_096, 2_560, 128): .affineMatvecC2560R4096G128Batched
        case (512, 2_560, 128): .affineMatvecC2560R512G128Batched
        case (1_024, 2_560, 128): .affineMatvecC2560R1024G128Batched
        case (10_240, 2_560, 128): .affineMatvecC2560R10240G128Batched
        case (10_752, 2_560, 128): .affineMatvecC2560R10752G128Batched
        case (2_560, 2_048, 128): .affineMatvecC2048R2560G128Batched
        case (2_560, 4_096, 128): .affineMatvecC4096R2560G128Batched
        case (2_560, 10_240, 128): .affineMatvecC10240R2560G128BatchedTile4
        case (256, 2_560, 128): .affineMatvecC2560R256G128Batched
        case (2_560, 256, 128): .affineMatvecC256R2560G128Batched
        case (262_144, 2_560, 128): .affineMatvecC2560R262144G128Batched
        default:
            nil
        }
    }

    static func prefillAffineSmallBatchPipeline(rows: Int, cols: Int, groupSize: Int) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (2_048, 2_560, 128):
            .affineMatvecC2560R2048G128BatchedTile3
        case (2_560, 2_048, 128):
            .affineMatvecC2048R2560G128BatchedTile3
        case (10_240, 2_560, 128):
            .affineMatvecC2560R10240G128BatchedExtB4
        case (2_560, 10_240, 128):
            .affineMatvecC10240R2560G128BatchedTile3
        case (4_096, 2_560, 128):
            .affineMatvecC2560R4096G128BatchedTile3
        case (2_560, 4_096, 128):
            .affineMatvecC4096R2560G128BatchedTile3
        default:
            nil
        }
    }

    static func prefillAffineSmallBatchRowTile(_ pipeline: SmeltPipeline?) -> Int {
        switch pipeline {
        case .affineMatvecC2560R2048G128BatchedSG4BT5,
             .affineMatvecC2048R2560G128BatchedSG4BT5,
             .affineMatvecC2560R10240G128BatchedSG4BT5,
             .affineMatvecC10240R2560G128BatchedSG4BT5:
            16
        case .affineMatvecC2560R10240G128BatchedExtB5,
             .affineMatvecC10240R2560G128BatchedExtB5,
             .affineMatvecC2560R2048G128BatchedExtB5,
             .affineMatvecC2048R2560G128BatchedExtB5,
             .affineMatvecC2560R10240G128BatchedExtB4,
             .affineMatvecC10240R2560G128BatchedExtB4,
             .affineMatvecC2560R2048G128BatchedExtB4,
             .affineMatvecC2048R2560G128BatchedExtB4:
            8
        case .affineMatvecC10240R2560G128BatchedTile3,
             .affineMatvecC2560R2048G128BatchedTile3,
             .affineMatvecC2048R2560G128BatchedTile3,
             .affineMatvecC2560R4096G128BatchedTile3,
             .affineMatvecC4096R2560G128BatchedTile3:
            8
        default:
            8
        }
    }

    static func prefillAffineSmallBatchBatchTile(_ pipeline: SmeltPipeline?) -> Int {
        switch pipeline {
        case .affineMatvecC2560R2048G128BatchedSG4BT5,
             .affineMatvecC2048R2560G128BatchedSG4BT5,
             .affineMatvecC2560R10240G128BatchedSG4BT5,
             .affineMatvecC10240R2560G128BatchedSG4BT5,
             .affineMatvecC2560R10240G128BatchedExtB5,
             .affineMatvecC10240R2560G128BatchedExtB5,
             .affineMatvecC2560R2048G128BatchedExtB5,
             .affineMatvecC2048R2560G128BatchedExtB5:
            5
        case .affineMatvecC2560R10240G128BatchedExtB4,
             .affineMatvecC10240R2560G128BatchedExtB4,
             .affineMatvecC2560R2048G128BatchedExtB4,
             .affineMatvecC2048R2560G128BatchedExtB4:
            4
        case .affineMatvecC10240R2560G128BatchedTile3,
             .affineMatvecC2560R2048G128BatchedTile3,
             .affineMatvecC2048R2560G128BatchedTile3,
             .affineMatvecC2560R4096G128BatchedTile3,
             .affineMatvecC4096R2560G128BatchedTile3:
            3
        default:
            8
        }
    }

    static func prefillAffineSmallBatchThreads(_ pipeline: SmeltPipeline?) -> Int {
        switch pipeline {
        case .affineMatvecC2560R2048G128BatchedSG4BT5,
             .affineMatvecC2048R2560G128BatchedSG4BT5,
             .affineMatvecC2560R10240G128BatchedSG4BT5,
             .affineMatvecC10240R2560G128BatchedSG4BT5:
            128
        case .affineMatvecC2560R10240G128BatchedExtB5,
             .affineMatvecC10240R2560G128BatchedExtB5,
             .affineMatvecC2560R2048G128BatchedExtB5,
             .affineMatvecC2048R2560G128BatchedExtB5,
             .affineMatvecC2560R10240G128BatchedExtB4,
             .affineMatvecC10240R2560G128BatchedExtB4,
             .affineMatvecC2560R2048G128BatchedExtB4,
             .affineMatvecC2048R2560G128BatchedExtB4:
            64
        default:
            64
        }
    }

    static func prefillAffineBatchedBatchTile(_ pipeline: SmeltPipeline?) -> Int {
        switch pipeline {
        case .affineMatvecC10240R2560G128BatchedTile4:
            4
        default:
            8
        }
    }

    static func prefillVerifyArgmaxPipeline(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPipeline? {
        switch (rows, cols, groupSize) {
        case (262_144, 1_536, 128):
            .affineMatvecArgmaxC1536R262144G128Batched
        case (262_144, 2_560, 128):
            .affineMatvecArgmaxC2560R262144G128Batched
        default:
            nil
        }
    }

    static func prefillVerifyArgmaxReducePipeline(rows: Int) -> SmeltPipeline? {
        switch rows {
        case 262_144:
            .lmHeadArgmaxReduceR262144
        default:
            nil
        }
    }

    static func prefillFusedGateUpFullPipeline(
        rows: Int,
        cols: Int,
        groupSize: Int,
        activation: SmeltActivation,
        preferVerifySmallBatch: Bool = false
    ) -> SmeltPipeline? {
        switch activation {
        case .swiglu:
            switch (rows, cols, groupSize) {
            case (3_584, 1_024, 64): .fusedAffineGateUpSwigluC1024R3584G64BatchedFull
            case (6_144, 2_048, 64): .fusedAffineGateUpSwigluC2048R6144G64BatchedFull
            case (8_192, 2_048, 64): .fusedAffineGateUpSwigluC2048R8192G64BatchedFull
            case (9_216, 2_560, 64): .fusedAffineGateUpSwigluC2560R9216G64BatchedFull
            case (8_192, 3_072, 64): .fusedAffineGateUpSwigluC3072R8192G64BatchedFull
            default: nil
            }
        case .geglu:
            switch (rows, cols, groupSize) {
            case (10_240, 2_560, 128):
                preferVerifySmallBatch
                    ? .fusedAffineGateUpGeGLUC2560R10240G128Batched
                    : .fusedAffineGateUpGeGLUC2560R10240G128BatchedFull
            default: nil
            }
        }
    }

    static func prefillDualMatvecActivationPipeline(
        first: SmeltPipeline,
        second: SmeltPipeline,
        activation: SmeltPipeline
    ) -> SmeltPipeline? {
        switch (first, second, activation) {
        case (
            .affineMatvecC2560R10240G128Batched,
            .affineMatvecC2560R10240G128Batched,
            .gegluFused
        ):
            .fusedAffineGateUpGeGLUC2560R10240G128Batched
        default:
            nil
        }
    }

    static func prefillFusedGateUpFullUsesQMM(_ pipeline: SmeltPipeline?) -> Bool {
        switch pipeline {
        case .fusedAffineGateUpSwigluC1024R3584G64BatchedFull,
             .fusedAffineGateUpSwigluC2048R6144G64BatchedFull,
             .fusedAffineGateUpSwigluC2048R8192G64BatchedFull,
             .fusedAffineGateUpSwigluC2560R9216G64BatchedFull,
             .fusedAffineGateUpSwigluC3072R8192G64BatchedFull,
             .fusedAffineGateUpGeGLUC2560R10240G128BatchedFull:
            true
        default:
            false
        }
    }

    static func prefillFusedGateUpFullUsesSG4(_ pipeline: SmeltPipeline?) -> Bool {
        pipeline == .fusedAffineGateUpGeGLUC2560R10240G128BatchedBT4SG4
    }

    static func prefillFusedGateUpFullRowTile(_ pipeline: SmeltPipeline?) -> Int {
        if prefillFusedGateUpFullUsesQMM(pipeline) { return 32 }
        return prefillFusedGateUpFullUsesSG4(pipeline) ? 16 : 8
    }

    static func prefillFusedGateUpFullBatchTile(_ pipeline: SmeltPipeline?) -> Int {
        if prefillFusedGateUpFullUsesSG4(pipeline) { return 4 }
        if pipeline == .fusedAffineGateUpGeGLUC2560R10240G128Batched { return 3 }
        return prefillFusedGateUpFullUsesQMM(pipeline) ? 16 : 8
    }

    static func prefillFusedGateUpFullThreads(_ pipeline: SmeltPipeline?) -> Int {
        prefillFusedGateUpFullUsesQMM(pipeline)
            || prefillFusedGateUpFullUsesSG4(pipeline) ? 128 : 64
    }

    static func decodeFusedFFNUsesRows4(_ pipeline: SmeltPipeline?) -> Bool {
        switch pipeline {
        case .fusedAffineGateUpSwigluC1024R3584G64Rows4,
             .fusedAffineGateUpSwigluC2048R6144G64,
             .fusedAffineGateUpSwigluC2048R11008G64,
             .fusedAffineGateUpSwigluC2560R9216G64,
             .fusedAffineGateUpSwigluC3072R8192G64,
             .fusedAffineGateUpGeGLUC1536R6144G128Rows4,
             .fusedAffineGateUpGeGLUC1536R12288G128Rows4,
             .fusedAffineGateUpGeGLUC2560R10240G128Rows4,
             .fusedAffineGateUpGeGLUC256R2048G128Rows4,
             .normScaleAffineGateUpGeGLUC1536R6144G128Rows4,
             .normScaleAffineGateUpGeGLUC1536R12288G128Rows4,
             .normScaleAffineGateUpGeGLUC2560R10240G128Rows4,
             .normScaleAffineGateUpGeGLUC256R2048G128Rows4:
            true
        default:
            false
        }
    }

    static func decodeFusedFFNThreads(_: SmeltPipeline?) -> Int {
        64
    }

    static func decodeAffineUsesRows4(_ pipeline: SmeltPipeline?) -> Bool {
        switch pipeline {
        case .affineMatvecC2048R2048G64,
             .affineMatvecC3072R3072G64,
             .affineMatvecC1024R2048G64Rows4,
             .affineMatvecC1024R6144G64Rows4,
             .affineMatvecC2048R1024G64Rows4,
             .affineMatvecC3584R1024G64Rows4,
             .affineMatvecC1024R248320G64Rows4,
             .affineMatvecC1536R2048G128Rows4,
             .affineMatvecC1536R256G128Rows4,
             .affineMatvecC1536R6144G128Rows4,
             .affineMatvecC1536R12288G128Rows4,
             .affineMatvecC1536R262144G128Rows4,
             .affineMatvecC256R1536G128Rows4,
             .affineMatvecC2048R1536G128Rows4,
             .affineMatvecC6144R1536G128Rows4,
             .affineMatvecC12288R1536G128Rows4,
             .affineMatvecC2560R2048G128Rows4,
             .affineMatvecC2560R4096G128Rows4,
             .affineMatvecC2560R512G128Rows4,
             .affineMatvecC2560R1024G128Rows4,
             .affineMatvecC2560R10240G128Rows4,
             .affineMatvecC2560R10752G128Rows4,
             .affineMatvecC2048R2560G128Rows4,
             .affineMatvecC4096R2560G128Rows4,
             .affineMatvecC10240R2560G128Rows4,
             .affineMatvecC10240R2560G128Rows4SG1,
             .affineMatvecC2560R256G128Rows4,
             .affineMatvecC256R2560G128Rows4,
             .affineMatvecC256R1024G128Rows4,
             .affineMatvecC256R2048G128Rows4,
             .affineMatvecC1024R256G128Rows4,
             .affineMatvecC2048R256G128Rows4:
            true
        default:
            false
        }
    }

    static func decodeRmsNormPipeline(dim: Int, eps: Float) -> SmeltPipeline? {
        guard eps == 1e-6 else { return nil }
        switch dim {
        case 1_024:
            return .rmsNorm1PWD1024
        case 1_536:
            return .rmsNorm1PWD1536
        case 2_048:
            return .rmsNorm1PWD2048
        case 2_560:
            return .rmsNorm1PWD2560
        default:
            return nil
        }
    }

    static func batchedRmsNormPipeline(dim: Int, eps: Float) -> SmeltPipeline? {
        guard eps == 1e-6 else { return nil }
        switch dim {
        case 1_024:
            return .rmsNorm1PWD1024Batched
        case 1_536:
            return .rmsNorm1PWD1536Batched
        case 2_560:
            return .rmsNorm1PWD2560Batched
        default:
            return nil
        }
    }

    static func rmsNormThreads(_ pipeline: SmeltPipeline?) -> Int? {
        switch pipeline {
        case .rmsNorm1PWD1024, .rmsNorm1PWD1024Batched:
            128
        case .rmsNorm1PWD1536, .rmsNorm1PWD1536Batched:
            192
        case .rmsNorm1PWD2048:
            256
        case .rmsNorm1PWD2560, .rmsNorm1PWD2560Batched:
            320
        default:
            nil
        }
    }
}
