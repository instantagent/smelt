import Foundation

/// Generates model-derived Metal wrapper kernels for reusable templated families.
///
/// The templates still need concrete rows/cols/groupSize for the fastest QMM paths,
/// but those concrete wrappers should come from model shape discovery instead of
/// handwritten catalog cases.
enum SmeltGeneratedKernelVariants {
    static func prefillAffineFullName(rows: Int, cols: Int, groupSize: Int) -> String {
        "affine_matvec_c\(cols)_r\(rows)_g\(groupSize)_batched_full"
    }

    static func prefillFusedGateUpSwigluFullName(rows: Int, cols: Int, groupSize: Int) -> String {
        "fused_affine_gate_up_swiglu_c\(cols)_r\(rows)_g\(groupSize)_batched_full"
    }

    static func prefillAffineSmallBatchName(rows: Int, cols: Int, groupSize: Int) -> String {
        "affine_matvec_c\(cols)_r\(rows)_g\(groupSize)_batched_verify_b4"
    }

    static func prefillFusedGateUpSwigluSmallBatchName(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> String {
        "fused_affine_gate_up_swiglu_c\(cols)_r\(rows)_g\(groupSize)_batched_verify_b4"
    }

    static func prefillFusedDualAffineSmallBatchName(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> String {
        "fused_dual_affine_matvec_c\(cols)_r\(rows)_g\(groupSize)_batched_verify_b4"
    }

    /// Long reductions expose more instruction-level latency than row
    /// parallelism. Measurements on the canonical large-model shapes show
    /// that assigning 16 lanes per row wins once K reaches 8K, while the
    /// 8-lane team remains better below that point. Keeping this policy beside
    /// generated naming/source makes the choice automatic for every model.
    static func prefillAffineSmallBatchRowTeamWidth(cols: Int) -> Int {
        cols >= 8_192 ? 16 : 8
    }

    static func prefillAffineSmallBatchRowTile(cols: Int) -> Int {
        64 / prefillAffineSmallBatchRowTeamWidth(cols: cols)
    }

    static func prefillVerifyArgmaxName(rows: Int, cols: Int, groupSize: Int) -> String {
        "affine_matvec_argmax_c\(cols)_r\(rows)_g\(groupSize)_batched"
    }

    static func prefillVerifyArgmaxReduceName(rows: Int) -> String {
        "lm_head_argmax_reduce_r\(rows)"
    }

    static func decodeFusedAffineMatvecAddName(rows: Int, cols: Int, groupSize: Int) -> String {
        "fused_affine_matvec_add_c\(cols)_r\(rows)_g\(groupSize)"
    }

    static func decodeFusedAffineMatvecAddRows4Name(rows: Int, cols: Int, groupSize: Int) -> String {
        "fused_affine_matvec_add_c\(cols)_r\(rows)_g\(groupSize)_rows4"
    }

    static func decodeFusedDualAffineMatvecAddRows4Name(rows: Int, cols: Int, groupSize: Int) -> String {
        "fused_dual_affine_matvec_add_c\(cols)_r\(rows)_g\(groupSize)_rows4"
    }

    static func canGenerateAffineU4Full(groupSize: Int) -> Bool {
        groupSize > 0 && groupSize % 32 == 0
    }

    static func canGenerateAffineU4Fixed(rows: Int, cols: Int, groupSize: Int) -> Bool {
        rows > 0
            && rows % 8 == 0
            && cols > 0
            && cols % 16 == 0
            && groupSize > 0
            && cols % groupSize == 0
    }

    static func canGenerateAffineU4FixedRows4(rows: Int, cols: Int, groupSize: Int) -> Bool {
        rows > 0
            && rows % 4 == 0
            && cols > 0
            && cols % 16 == 0
            && groupSize > 0
            && cols % groupSize == 0
    }

    static func lutMatvecSuffix(kernelPlan: SmeltKernelPlan) -> String {
        let blocks = generatedSourceBlocks(kernelPlan: kernelPlan)
        guard !blocks.isEmpty else { return "" }
        return "\n\n// MARK: - Generated model-shape QMM wrappers\n\n"
            + blocks.map(\.source).joined(separator: "\n\n")
    }

    static func generatedSourceFunctionNames(kernelPlan: SmeltKernelPlan) -> Set<String> {
        Set(generatedSourceBlocks(kernelPlan: kernelPlan).map(\.name))
    }

    private static func generatedSourceBlocks(
        kernelPlan: SmeltKernelPlan
    ) -> [(name: String, source: String)] {
        var blocks: [(name: String, source: String)] = []

        func append(_ name: String, _ block: String) {
            blocks.append((name, block))
        }

        for capability in kernelPlan.emittedGeneratedCapabilities {
            let shape = capability.shape
            guard let template = capability.sourceTemplate else {
                preconditionFailure(
                    "package-local generated capability missing source template: \(capability.id)"
                )
            }
            switch template {
            case .affineMatvecResidualAddFixed:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedAffineMatvecAddKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .affineMatvecResidualAddFixedRows4:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedAffineMatvecAddRows4Kernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .fusedDualAffineMatvecResidualAddFixedRows4:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedDualAffineMatvecAddRows4Kernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .affineMatvecPrefillFull:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    affineFullKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .fusedGateUpSwigluPrefillFull:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedGateUpSwigluFullKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .affineMatvecPrefillSmallBatch:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    affineSmallBatchKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .fusedGateUpSwigluPrefillSmallBatch:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedGateUpSwigluSmallBatchKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .fusedDualAffineMatvecPrefillSmallBatch:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    fusedDualAffineSmallBatchKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .affineVerifyArgmaxPrefill:
                let geometry = requireGeneratedSourceGeometry(capability)
                append(
                    capability.id,
                    verifyArgmaxKernel(
                        name: capability.id,
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize,
                        geometry: geometry
                    )
                )
            case .verifyArgmaxReduce:
                append(
                    capability.id,
                    verifyArgmaxReduceKernel(
                        name: capability.id,
                        rows: shape.rows
                    )
                )
            }
        }

        return blocks
    }

    private struct GeneratedSourceGeometry {
        let rowTile: Int
        let batchTile: Int?
        let threadgroupWidth: Int

        var comment: String {
            let batch = batchTile.map(String.init) ?? "nil"
            return "// planned geometry: row_tile=\(rowTile) batch_tile=\(batch) threadgroup_width=\(threadgroupWidth)"
        }
    }

    private static func requireGeneratedSourceGeometry(
        _ capability: SmeltKernelCapability
    ) -> GeneratedSourceGeometry {
        guard let rowTile = capability.rowTile,
              let threadgroupWidth = capability.threadgroupWidth else {
            preconditionFailure(
                "generated source geometry missing for \(capability.id): "
                    + "planned rowTile=\(String(describing: capability.rowTile)), "
                    + "batchTile=\(String(describing: capability.batchTile)), "
                    + "threadgroupWidth=\(String(describing: capability.threadgroupWidth))"
            )
        }
        return GeneratedSourceGeometry(
            rowTile: rowTile,
            batchTile: capability.batchTile,
            threadgroupWidth: threadgroupWidth
        )
    }

    private static func fusedAffineMatvecAddKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* weights  [[buffer(0)]],
            device const half*    scales   [[buffer(1)]],
            device const half*    biases   [[buffer(2)]],
            device const half*    input    [[buffer(3)]],
            device half*          output   [[buffer(4)]],
            device const half*    residual [[buffer(5)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            fused_affine_matvec_add_fixed<\(rows), \(cols), \(groupSize)>(
                weights, scales, biases, input, output, residual,
                tgid, simd_lane, simd_group
            );
        }
        """
    }

    private static func fusedAffineMatvecAddRows4Kernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* weights  [[buffer(0)]],
            device const half*    scales   [[buffer(1)]],
            device const half*    biases   [[buffer(2)]],
            device const half*    input    [[buffer(3)]],
            device half*          output   [[buffer(4)]],
            device const half*    residual [[buffer(5)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            fused_affine_matvec_add_fixed_rows4<\(rows), \(cols), \(groupSize)>(
                weights, scales, biases, input, output, residual,
                tgid, simd_lane, simd_group
            );
        }
        """
    }

    private static func fusedDualAffineMatvecAddRows4Kernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* w1_weights [[buffer(0)]],
            device const half*    w1_scales  [[buffer(1)]],
            device const half*    w1_biases  [[buffer(2)]],
            device const uint8_t* w2_weights [[buffer(3)]],
            device const half*    w2_scales  [[buffer(4)]],
            device const half*    w2_biases  [[buffer(5)]],
            device const half*    input      [[buffer(6)]],
            device half*          output1    [[buffer(7)]],
            device half*          output2    [[buffer(8)]],
            device const half*    residual1  [[buffer(9)]],
            device const half*    residual2  [[buffer(10)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            fused_dual_affine_matvec_add_fixed_rows4<\(rows), \(cols), \(groupSize)>(
                w1_weights, w1_scales, w1_biases,
                w2_weights, w2_scales, w2_biases,
                input, output1, output2,
                residual1, residual2,
                tgid, simd_lane, simd_group
            );
        }
        """
    }

    private static func affineFullKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* weights  [[buffer(0)]],
            device const half*    scales   [[buffer(1)]],
            device const half*    biases   [[buffer(2)]],
            device const half*    input    [[buffer(3)]],
            device half*          output   [[buffer(4)]],
            constant uint&        actualBatch [[buffer(5)]],
            uint2 tgid       [[threadgroup_position_in_grid]],
            uint tid         [[thread_index_in_threadgroup]],
            uint simd_lane   [[thread_index_in_simdgroup]],
            uint simd_group  [[simdgroup_index_in_threadgroup]]
        ) {
            threadgroup half Xs[\(batchTile) * (\(geometry.rowTile) + 8)];
            threadgroup float Ws[\(geometry.rowTile) * (\(geometry.rowTile) + 8)];
            affine_matvec_fixed_batched_full<\(rows), \(cols), \(groupSize), \(batchTile)>(
                weights, scales, biases, input, output,
                actualBatch,
                Xs, Ws,
                tgid, tid, simd_lane, simd_group
            );
        }
        """
    }

    private static func fusedGateUpSwigluFullKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* gate_weights [[buffer(0)]],
            device const half*    gate_scales  [[buffer(1)]],
            device const half*    gate_biases  [[buffer(2)]],
            device const uint8_t* up_weights   [[buffer(3)]],
            device const half*    up_scales    [[buffer(4)]],
            device const half*    up_biases    [[buffer(5)]],
            device const half*    input        [[buffer(6)]],
            device half*          output       [[buffer(7)]],
            constant uint&        actualBatch  [[buffer(8)]],
            uint2 tgid       [[threadgroup_position_in_grid]],
            uint tid         [[thread_index_in_threadgroup]],
            uint simd_lane   [[thread_index_in_simdgroup]],
            uint simd_group  [[simdgroup_index_in_threadgroup]]
        ) {
            threadgroup half Xs[\(batchTile) * (\(geometry.rowTile) + 8)];
            threadgroup float Wg[\(geometry.rowTile) * (\(geometry.rowTile) + 8)];
            threadgroup float Wu[\(geometry.rowTile) * (\(geometry.rowTile) + 8)];
            agent_fused_affine_gate_up_qmm_fixed_batched_full<\(rows), \(cols), \(groupSize), \(batchTile)>(
                gate_weights, gate_scales, gate_biases,
                up_weights, up_scales, up_biases,
                input, output,
                actualBatch,
                Xs, Wg, Wu,
                tgid, tid, simd_lane, simd_group
            );
        }
        """
    }

    private static func affineSmallBatchKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        precondition(batchTile == 4, "vec4 small-batch affine requires batch tile 4")
        let rowTeamWidth = prefillAffineSmallBatchRowTeamWidth(cols: cols)
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* weights  [[buffer(0)]],
            device const half*    scales   [[buffer(1)]],
            device const half*    biases   [[buffer(2)]],
            device const half*    input    [[buffer(3)]],
            device half*          output   [[buffer(4)]],
            constant uint&        actualBatch [[buffer(5)]],
            uint2 tgid [[threadgroup_position_in_grid]],
            uint tiisg [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]]
        ) {
            affine_matvec_fixed_ext_vec4<
                \(rows), \(cols), \(groupSize), \(rowTeamWidth), 2, 2
            >(
                weights, scales, biases, input, output,
                actualBatch, tgid, tiisg, sgitg
            );
        }
        """
    }

    private static func fusedGateUpSwigluSmallBatchKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        precondition(batchTile == 4, "vec4 small-batch gate/up requires batch tile 4")
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* gate_weights [[buffer(0)]],
            device const half*    gate_scales  [[buffer(1)]],
            device const half*    gate_biases  [[buffer(2)]],
            device const uint8_t* up_weights   [[buffer(3)]],
            device const half*    up_scales    [[buffer(4)]],
            device const half*    up_biases    [[buffer(5)]],
            device const half*    input        [[buffer(6)]],
            device half*          output       [[buffer(7)]],
            constant uint&        actualBatch  [[buffer(8)]],
            uint2 tgid [[threadgroup_position_in_grid]],
            uint tiisg [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]]
        ) {
            fused_affine_gate_up_swiglu_fixed_ext_vec4<\(rows), \(cols), \(groupSize)>(
                gate_weights, gate_scales, gate_biases,
                up_weights, up_scales, up_biases,
                input, output, actualBatch,
                tgid, tiisg, sgitg
            );
        }
        """
    }

    private static func fusedDualAffineSmallBatchKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* first_weights [[buffer(0)]],
            device const half* first_scales [[buffer(1)]],
            device const half* first_biases [[buffer(2)]],
            device const uint8_t* second_weights [[buffer(3)]],
            device const half* second_scales [[buffer(4)]],
            device const half* second_biases [[buffer(5)]],
            device const half* input [[buffer(6)]],
            device half* first_output [[buffer(7)]],
            device half* second_output [[buffer(8)]],
            constant uint& actualBatch [[buffer(9)]],
            uint2 tgid [[threadgroup_position_in_grid]],
            uint tiisg [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]]
        ) {
            fused_dual_affine_matvec_fixed_ext<
                \(rows), \(cols), \(groupSize), \(batchTile)
            >(
                first_weights, first_scales, first_biases,
                second_weights, second_scales, second_biases,
                input, first_output, second_output, actualBatch,
                tgid, tiisg, sgitg
            );
        }
        """
    }

    private static func verifyArgmaxKernel(
        name: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        geometry: GeneratedSourceGeometry
    ) -> String {
        let batchTile = requireBatchTile(geometry, name: name)
        return """
        \(geometry.comment)
        kernel void \(name)(
            device const uint8_t* weights [[buffer(0)]],
            device const half* scales [[buffer(1)]],
            device const half* biases [[buffer(2)]],
            device const half* input [[buffer(3)]],
            device uint2* partialKeys [[buffer(4)]],
            constant uint& actualBatch [[buffer(5)]],
            constant float& logitCap [[buffer(6)]],
            uint2 tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            threadgroup uint2 localKeys[\(batchTile) * 2];
            affine_matvec_argmax_fixed_batched<\(rows), \(cols), \(groupSize), \(batchTile)>(
                weights, scales, biases, input, partialKeys,
                actualBatch, logitCap, localKeys,
                tgid, simd_lane, simd_group
            );
        }
        """
    }

    private static func verifyArgmaxReduceKernel(
        name: String,
        rows: Int
    ) -> String {
        """
        kernel void \(name)(
            device const uint2* partialKeys [[buffer(0)]],
            device int* output [[buffer(1)]],
            constant uint& actualBatch [[buffer(2)]],
            uint batch [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            threadgroup uint2 scratch[256];
            lm_head_argmax_reduce_fixed<\(rows)>(
                partialKeys, output, actualBatch, batch, tid, scratch
            );
        }
        """
    }

    private static func requireBatchTile(
        _ geometry: GeneratedSourceGeometry,
        name: String
    ) -> Int {
        guard let batchTile = geometry.batchTile else {
            preconditionFailure("generated full source geometry missing batch tile for \(name)")
        }
        return batchTile
    }
}
