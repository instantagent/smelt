import Foundation
import SmeltSchema

extension SmeltQwen35VisionRuntime {
    /// Freeze the exact component route selected by `encode`. The returned
    /// plan contains only generic MPS/Metal operations and logical resources;
    /// downstream costing has no Qwen-specific behavior.
    public static func frozenPlan(
        config: SmeltQwen35VisionConfig,
        grids: [Grid],
        provenanceKey: String,
        gemmBackend: SmeltQwen35VisionGEMMBackend = .mps,
        attentionBackend: SmeltQwen35VisionAttentionBackend = .mpsStaged
    ) throws -> SmeltFrozenIRPlan {
        guard gemmBackend == .mps, attentionBackend == .mpsStaged else {
            throw SmeltQwen35VisionRuntimeError.unsupportedFrozenPlanRoute(
                gemm: gemmBackend,
                attention: attentionBackend
            )
        }
        guard !grids.isEmpty,
              grids.allSatisfy({
                  $0.temporal > 0 && $0.height > 0 && $0.width > 0
                    && $0.height.isMultiple(of: config.spatialMergeSize)
                    && $0.width.isMultiple(of: config.spatialMergeSize)
              })
        else {
            throw SmeltQwen35VisionRuntimeError.invalidGrid(grids)
        }
        let tokens = grids.reduce(0) { $0 + $1.patchCount }
        let mergeUnit = config.spatialMergeSize * config.spatialMergeSize
        guard tokens.isMultiple(of: mergeUnit) else {
            throw SmeltQwen35VisionRuntimeError.invalidGrid(grids)
        }
        let mergedTokens = tokens / mergeUnit
        let hidden = config.hiddenSize
        let intermediate = config.intermediateSize
        let mergedHidden = hidden * mergeUnit
        let patchWidth = config.inChannels
            * config.temporalPatchSize
            * config.patchSize
            * config.patchSize
        var plan = SmeltFrozenComponentPlanBuilder(
            planID: "component-dense-vision",
            provenanceKey: provenanceKey,
            context: SmeltCostModelContext(mode: .prefill, sequenceLength: tokens)
        )

        func elementProduct(_ values: [Int]) -> UInt64 {
            values.reduce(UInt64(1)) { partial, value in
                let (next, overflow) = partial.multipliedReportingOverflow(
                    by: UInt64(max(value, 0))
                )
                return overflow ? UInt64.max : next
            }
        }
        func elements(_ values: Int...) -> UInt64 {
            elementProduct(values)
        }
        func bytes(_ values: Int...) -> UInt64 {
            SmeltFrozenComponentPlanBuilder.bytes(elements: elementProduct(values))
        }
        func resource(
            _ binding: Int,
            _ name: String,
            _ storage: SmeltFrozenIRStorageClass,
            _ access: SmeltFrozenIRAccessKind,
            _ byteCount: UInt64
        ) -> SmeltFrozenIRResourceAccess {
            SmeltFrozenComponentPlanBuilder.resource(
                bindingIndex: binding,
                name: name,
                storageClass: storage,
                access: access,
                bytes: byteCount
            )
        }
        func operation(
            _ kind: SmeltFrozenIROperationClass,
            _ count: UInt64
        ) -> SmeltFrozenIROperationBill {
            SmeltFrozenComponentPlanBuilder.operation(kind, count: count)
        }
        func multiplied(_ value: UInt64, by multiplier: Int) -> UInt64 {
            let (result, overflow) = value.multipliedReportingOverflow(
                by: UInt64(max(multiplier, 0))
            )
            return overflow ? UInt64.max : result
        }

        func appendGEMM(
            _ label: String,
            rows: Int,
            inputSize: Int,
            outputSize: Int,
            input: String,
            weight: String,
            output: String
        ) {
            let inputElements = elements(rows, inputSize)
            let weightElements = elements(outputSize, inputSize)
            let outputElements = elements(rows, outputSize)
            plan.append(
                pipeline: "mps.matrix_multiplication.f32",
                operationGroup: "dense.matmul",
                logicalShape: [rows, outputSize, inputSize],
                resources: [
                    resource(0, input, .hotActivation, .read,
                             SmeltFrozenComponentPlanBuilder.bytes(elements: inputElements)),
                    resource(1, weight, .streamingWeight, .read,
                             SmeltFrozenComponentPlanBuilder.bytes(elements: weightElements)),
                    resource(2, output, .hotActivation, .write,
                             SmeltFrozenComponentPlanBuilder.bytes(elements: outputElements)),
                ],
                operations: [operation(
                    .fp32Arithmetic,
                    multiplied(multiplied(outputElements, by: inputSize), by: 2)
                )],
                intermediateMaterializationBytes: SmeltFrozenComponentPlanBuilder.bytes(
                    elements: outputElements
                )
            )
        }
        func appendBias(
            _ label: String,
            rows: Int,
            columns: Int,
            values: String,
            bias: String
        ) {
            let valueBytes = bytes(rows, columns)
            plan.append(
                pipeline: "qwen35_vision_add_bias_rows_f32",
                operationGroup: "elementwise.bias",
                logicalShape: [rows, columns],
                resources: [
                    resource(0, values, .hotActivation, .readWrite, valueBytes),
                    resource(1, bias, .streamingWeight, .read, bytes(columns)),
                ],
                operations: [operation(.fp32Arithmetic, elements(rows, columns))]
            )
        }
        func appendAdd(
            _ label: String,
            rows: Int,
            columns: Int,
            lhs: String,
            rhs: String,
            output: String
        ) {
            let valueBytes = bytes(rows, columns)
            plan.append(
                pipeline: "qwen35_vision_add_f32",
                operationGroup: "elementwise.add",
                logicalShape: [rows, columns],
                resources: [
                    resource(0, lhs, .hotActivation, .read, valueBytes),
                    resource(1, rhs, .hotActivation, .read, valueBytes),
                    resource(2, output, .hotActivation, .write, valueBytes),
                ],
                operations: [operation(.fp32Arithmetic, elements(rows, columns))],
                intermediateMaterializationBytes: valueBytes
            )
        }
        func appendLayerNorm(
            _ label: String,
            rows: Int,
            dimension: Int,
            input: String,
            weight: String,
            bias: String,
            output: String
        ) {
            let valueElements = elements(rows, dimension)
            let valueBytes = SmeltFrozenComponentPlanBuilder.bytes(elements: valueElements)
            plan.append(
                pipeline: "qwen35_vision_layer_norm_f32",
                operationGroup: "normalization.layer",
                logicalShape: [rows, dimension],
                resources: [
                    resource(0, input, .hotActivation, .read, valueBytes),
                    resource(1, weight, .streamingWeight, .read, bytes(dimension)),
                    resource(2, bias, .streamingWeight, .read, bytes(dimension)),
                    resource(3, output, .hotActivation, .write, valueBytes),
                ],
                operations: [
                    operation(.fp32Arithmetic, multiplied(valueElements, by: 8)),
                    operation(.reduction, multiplied(valueElements, by: 2)),
                ],
                synchronization: .threadgroup,
                intermediateMaterializationBytes: valueBytes
            )
        }
        func appendGELU(
            _ label: String,
            rows: Int,
            columns: Int,
            values: String
        ) {
            let valueElements = elements(rows, columns)
            let valueBytes = SmeltFrozenComponentPlanBuilder.bytes(elements: valueElements)
            plan.append(
                pipeline: "qwen35_vision_gelu_tanh_f32",
                operationGroup: "activation.gelu_tanh",
                logicalShape: [rows, columns],
                resources: [resource(0, values, .hotActivation, .readWrite, valueBytes)],
                operations: [
                    operation(.fp32Arithmetic, multiplied(valueElements, by: 10)),
                    operation(.transcendental, valueElements),
                ]
            )
        }

        appendGEMM(
            "patch-embed", rows: tokens, inputSize: patchWidth, outputSize: hidden,
            input: "patches", weight: "patch_embed.weight", output: "hidden-a"
        )
        appendBias(
            "patch-embed.bias", rows: tokens, columns: hidden,
            values: "hidden-a", bias: "patch_embed.bias"
        )
        appendAdd(
            "position-add", rows: tokens, columns: hidden,
            lhs: "hidden-a", rhs: "position", output: "hidden-b"
        )

        var residual = "hidden-b"
        var nextResidual = "hidden-a"
        for layer in 0..<config.layerCount {
            let prefix = "blocks.\(layer)"
            appendLayerNorm(
                "\(prefix).norm1", rows: tokens, dimension: hidden,
                input: residual, weight: "\(prefix).norm1.weight",
                bias: "\(prefix).norm1.bias", output: "normed"
            )
            appendGEMM(
                "\(prefix).qkv", rows: tokens, inputSize: hidden, outputSize: 3 * hidden,
                input: "normed", weight: "\(prefix).qkv.weight", output: "qkv"
            )
            appendBias(
                "\(prefix).qkv.bias", rows: tokens, columns: 3 * hidden,
                values: "qkv", bias: "\(prefix).qkv.bias"
            )
            let hiddenBytes = bytes(tokens, hidden)
            plan.append(
                pipeline: "qwen35_vision_split_rope_f32",
                operationGroup: "attention.qkv_rope_split",
                logicalShape: [tokens, config.headCount, config.headDim],
                resources: [
                    resource(0, "qkv", .hotActivation, .read, bytes(tokens, 3 * hidden)),
                    resource(1, "cosines", .lookupTable, .read, bytes(tokens, config.headDim)),
                    resource(2, "sines", .lookupTable, .read, bytes(tokens, config.headDim)),
                    resource(3, "q", .hotActivation, .write, hiddenBytes),
                    resource(4, "k", .hotActivation, .write, hiddenBytes),
                    resource(5, "v", .hotActivation, .write, hiddenBytes),
                ],
                operations: [operation(.fp32Arithmetic, multiplied(elements(tokens, hidden), by: 6))],
                intermediateMaterializationBytes: multiplied(hiddenBytes, by: 3)
            )

            for (segment, grid) in grids.enumerated() {
                let chunkTokens = grid.height * grid.width
                for frame in 0..<grid.temporal {
                    for head in 0..<config.headCount {
                        let stem = "\(prefix).attention.s\(segment).f\(frame).h\(head)"
                        let activationBytes = bytes(chunkTokens, config.headDim)
                        let scoreElements = elements(chunkTokens, chunkTokens)
                        let scoreBytes = SmeltFrozenComponentPlanBuilder.bytes(
                            elements: scoreElements
                        )
                        let mmaOperations = multiplied(
                            multiplied(scoreElements, by: config.headDim),
                            by: 2
                        )
                        plan.append(
                            pipeline: "mps.matrix_multiplication.f32.qk",
                            operationGroup: "attention.matmul",
                            logicalShape: [chunkTokens, chunkTokens, config.headDim],
                            resources: [
                                resource(0, "\(stem).q", .hotActivation, .read, activationBytes),
                                resource(1, "\(stem).k", .hotActivation, .read, activationBytes),
                                resource(2, "attention-scores", .hotActivation, .write, scoreBytes),
                            ],
                            operations: [operation(.fp32Arithmetic, mmaOperations)],
                            intermediateMaterializationBytes: scoreBytes
                        )
                        plan.append(
                            pipeline: "qwen35_vision_softmax_rows_f32",
                            operationGroup: "attention.softmax",
                            logicalShape: [chunkTokens, chunkTokens],
                            resources: [resource(
                                0, "attention-scores", .hotActivation, .readWrite, scoreBytes
                            )],
                            operations: [
                                operation(.fp32Arithmetic, multiplied(scoreElements, by: 5)),
                                operation(.transcendental, scoreElements),
                                operation(.reduction, multiplied(scoreElements, by: 2)),
                            ],
                            synchronization: .threadgroup
                        )
                        plan.append(
                            pipeline: "mps.matrix_multiplication.f32.pv",
                            operationGroup: "attention.matmul",
                            logicalShape: [chunkTokens, config.headDim, chunkTokens],
                            resources: [
                                resource(0, "attention-scores", .hotActivation, .read, scoreBytes),
                                resource(1, "\(stem).v", .hotActivation, .read, activationBytes),
                                resource(2, "\(stem).output", .hotActivation, .write, activationBytes),
                            ],
                            operations: [operation(.fp32Arithmetic, mmaOperations)],
                            intermediateMaterializationBytes: activationBytes
                        )
                    }
                }
            }

            appendGEMM(
                "\(prefix).attention.proj", rows: tokens, inputSize: hidden, outputSize: hidden,
                input: "attention", weight: "\(prefix).proj.weight", output: "projection"
            )
            appendBias(
                "\(prefix).attention.proj.bias", rows: tokens, columns: hidden,
                values: "projection", bias: "\(prefix).proj.bias"
            )
            appendAdd(
                "\(prefix).attention.residual", rows: tokens, columns: hidden,
                lhs: residual, rhs: "projection", output: nextResidual
            )
            swap(&residual, &nextResidual)
            appendLayerNorm(
                "\(prefix).norm2", rows: tokens, dimension: hidden,
                input: residual, weight: "\(prefix).norm2.weight",
                bias: "\(prefix).norm2.bias", output: "normed"
            )
            appendGEMM(
                "\(prefix).fc1", rows: tokens, inputSize: hidden, outputSize: intermediate,
                input: "normed", weight: "\(prefix).fc1.weight", output: "intermediate"
            )
            appendBias(
                "\(prefix).fc1.bias", rows: tokens, columns: intermediate,
                values: "intermediate", bias: "\(prefix).fc1.bias"
            )
            appendGELU(
                "\(prefix).gelu", rows: tokens, columns: intermediate,
                values: "intermediate"
            )
            appendGEMM(
                "\(prefix).fc2", rows: tokens, inputSize: intermediate, outputSize: hidden,
                input: "intermediate", weight: "\(prefix).fc2.weight", output: "projection"
            )
            appendBias(
                "\(prefix).fc2.bias", rows: tokens, columns: hidden,
                values: "projection", bias: "\(prefix).fc2.bias"
            )
            appendAdd(
                "\(prefix).ffn.residual", rows: tokens, columns: hidden,
                lhs: residual, rhs: "projection", output: nextResidual
            )
            swap(&residual, &nextResidual)
        }

        appendLayerNorm(
            "merger.norm", rows: tokens, dimension: hidden,
            input: residual, weight: "merger.norm.weight",
            bias: "merger.norm.bias", output: "normed"
        )
        appendGEMM(
            "merger.fc1", rows: mergedTokens, inputSize: mergedHidden,
            outputSize: mergedHidden, input: "normed-merged",
            weight: "merger.fc1.weight", output: "merger-intermediate"
        )
        appendBias(
            "merger.fc1.bias", rows: mergedTokens, columns: mergedHidden,
            values: "merger-intermediate", bias: "merger.fc1.bias"
        )
        appendGELU(
            "merger.gelu", rows: mergedTokens, columns: mergedHidden,
            values: "merger-intermediate"
        )
        appendGEMM(
            "merger.fc2", rows: mergedTokens, inputSize: mergedHidden,
            outputSize: config.outputHiddenSize, input: "merger-intermediate",
            weight: "merger.fc2.weight", output: "vision-output"
        )
        appendBias(
            "merger.fc2.bias", rows: mergedTokens, columns: config.outputHiddenSize,
            values: "vision-output", bias: "merger.fc2.bias"
        )
        return plan.build()
    }
}
