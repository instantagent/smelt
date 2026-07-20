// SmeltWeightPacker — Packs quantized weight files into a monolithic weights.bin.
//
// Build-time only. Reads individual weight files (packed u4 indices + LUTs,
// or raw FP16), concatenates them into a single file with per-format alignment,
// computes SHA-256 checksum, and returns a SmeltWeightManifest with offsets.
//
// The runtime mmaps this single file and uses byte offsets to bind Metal buffers.

import CryptoKit
import Foundation
import SmeltSchema

// MARK: - Weight packer errors

/// Errors thrown during weight packing.
public enum SmeltWeightPackerError: Error, CustomStringConvertible {
    case sourceFileNotFound(name: String, path: String)
    case outputDirectoryCreationFailed(path: String)
    case fileWriteFailed(path: String, underlying: Error)
    case sizeMismatch(name: String, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .sourceFileNotFound(name, path):
            return "weight file not found: '\(name)' at \(path)"
        case let .outputDirectoryCreationFailed(path):
            return "failed to create output directory: \(path)"
        case let .fileWriteFailed(path, underlying):
            return "failed to write \(path): \(underlying)"
        case let .sizeMismatch(name, expected, actual):
            return "weight '\(name)' size mismatch: expected \(expected) bytes, got \(actual)"
        }
    }
}

// MARK: - Weight layout (compile-time offset computation)

/// Computes the weight manifest layout from IR alone, without reading files.
/// Useful for code-gen which needs offsets before weights are actually packed.
public struct SmeltWeightLayout {

    /// Compute the full weight layout from the model IR.
    /// Returns entries in canonical order: embed_tokens, per-layer weights,
    /// final norm, lm_head.
    public static func computeLayout(from ir: SmeltModelIR) -> [SmeltWeightEntry] {
        let cfg = ir.config
        let quant = ir.quantization
        var entries: [SmeltWeightEntry] = []
        var offset: UInt64 = 0

        // --- embed_tokens ---
        // Dense trunks expose an embeddings-IN port (the front-end native block
        // owns the embed table and writes hidden directly; DenseTrunkEmitter does
        // no token gather). Emit no embed table — it would be dead weight, and a
        // embeddings-in package must not carry one.
        let quantizesPacked = ir.quantization.strategy == .lutU4
            || ir.quantization.strategy == .affineU4
            || ir.quantization.strategy == .binary1
            || ir.quantization.strategy == .ternary2
        if cfg.portTopology == .embeddingsInHiddenOut || cfg.inputFusion != nil {
            // skip — embeddings-in port, no gather
        } else if ir.quantization.quantizeEmbedding && quantizesPacked {
            // Quantized embedding: u4 format (LUT or affine, per strategy)
            appendWeightEntry(
                name: SmeltCanonicalTensorNames.embedTokens,
                shape: [cfg.vocabSize, cfg.hiddenSize],
                quant: ir.quantization,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
        } else {
            // FP16 embedding (default)
            entries.append(makeFP16Entry(
                name: SmeltCanonicalTensorNames.embedTokens,
                shape: [cfg.vocabSize, cfg.hiddenSize],
                offset: &offset
            ))
        }

        if cfg.hiddenSizePerLayerInput > 0 && cfg.vocabSizePerLayerInput > 0 {
            let perLayerTotalDim = cfg.numLayers * cfg.hiddenSizePerLayerInput
            if ir.quantization.quantizeEmbedding
                && (ir.quantization.strategy == .affineU4
                    || ir.quantization.strategy == .binary1
                    || ir.quantization.strategy == .ternary2)
            {
                appendWeightEntry(
                    name: SmeltCanonicalTensorNames.embedTokensPerLayer,
                    shape: [cfg.vocabSizePerLayerInput, perLayerTotalDim],
                    quant: ir.quantization,
                    activationDtype: cfg.activationDtype,
                    portTopology: cfg.portTopology,
                    entries: &entries,
                    offset: &offset
                )
            } else {
                // Keep per-layer residual-input embeddings in fp16 on the LUT
                // path. Official checkpoints flatten one token's
                // per-layer inputs into a very wide table, and the current LUT
                // quantizer is not a good fit for that representation.
                entries.append(makeFP16Entry(
                    name: SmeltCanonicalTensorNames.embedTokensPerLayer,
                    shape: [cfg.vocabSizePerLayerInput, perLayerTotalDim],
                    offset: &offset
                ))
            }

            appendWeightEntry(
                name: "per_layer_model_projection_weight",
                shape: [perLayerTotalDim, cfg.hiddenSize],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )

            entries.append(makeFP16Entry(
                name: "per_layer_projection_norm_weight",
                shape: [cfg.hiddenSizePerLayerInput],
                offset: &offset
            ))
        }

        // Runtime-supplied feature fusion. Each source may carry its own norm,
        // then the concatenated slab is projected into the ordinary stack.
        // Legacy EAGLE `backboneHiddenSize` resolves to this same primitive.
        if let fusion = cfg.resolvedInputFusion {
            if fusion.normalizeSources {
                for source in 0 ..< fusion.sourceCount {
                    entries.append(makeNormEntry(
                        cfg: cfg,
                        name: "input_fusion_norm_\(source)_weight",
                        shape: [fusion.sourceWidth],
                        offset: &offset
                    ))
                }
            }
            appendWeightEntry(
                name: "pre_projection_weight",
                shape: [cfg.hiddenSize, fusion.concatenatedWidth],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
            if let postWidth = fusion.postProjectionWidth {
                appendWeightEntry(
                    name: "post_projection_weight",
                    shape: [postWidth, cfg.hiddenSize],
                    quant: quant,
                    activationDtype: cfg.activationDtype,
                    portTopology: cfg.portTopology,
                    entries: &entries,
                    offset: &offset
                )
            }
        }

        // EAGLE-class draft-model cluster embedder. Replaces the dense
        // lm_head matvec with a top-k centroid gather over the vocab.
        // Two tensors:
        //   - centroids: [num_centroids, hidden_size] — learned linear
        //     classifier mapping the draft model's last hidden state to
        //     centroid logits. Routes through appendWeightEntry so it
        //     picks up the spec's quantization strategy (fp16 / u4_lut /
        //     affine_u4).
        //   - token_ordering: [vocab_size] — permutation buffer that
        //     groups vocab tokens into contiguous clusters of size
        //     vocab_size / num_centroids. Stored as int32 even though
        //     the upstream HF tensor is int64; vocab_size fits well
        //     under 2^31 and the down-cast happens at HF ingest time
        //     (Phase 2c2). Sized as raw bytes because no current entry
        //     dtype handles 4-byte indices semantically.
        if let cluster = cfg.clusterEmbedder {
            appendWeightEntry(
                name: "masked_embedding_centroids_weight",
                shape: [cluster.numCentroids, cfg.hiddenSize],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
            // 16-byte align before the int32 buffer because the centroid
            // entry above does not self-align its trailing edge —
            // appendWeightEntry's u4/affine paths leave `offset` at
            // header+lut+data-end which can be sub-16-byte for arbitrary
            // shapes. The per-layer loop's first call self-aligns again,
            // so this is the only boundary that needs an explicit guard.
            offset = alignTo16(offset)
            let tokenOrderingBytes =
                UInt64(cfg.vocabSize) * UInt64(MemoryLayout<Int32>.stride)
            entries.append(SmeltWeightEntry(
                name: SmeltCanonicalTensorNames.maskedEmbeddingTokenOrdering,
                offset: offset,
                sizeBytes: tokenOrderingBytes,
                shape: [cfg.vocabSize],
                dtype: .int32
            ))
            offset += tokenOrderingBytes
        }

        // --- Per-layer weights ---
        let layers = ir.layerPattern.expanded
        for (layerIdx, layerType) in layers.enumerated() {
            // input_layernorm_weight: [hiddenSize], dtype per trunk ABI
            entries.append(makeNormEntry(
                cfg: cfg,
                name: "layers_\(layerIdx)_input_layernorm_weight",
                shape: [cfg.hiddenSize],
                offset: &offset
            ))

            switch layerType {
            case .delta:
                appendDeltaWeights(
                    layerIdx: layerIdx,
                    cfg: cfg,
                    quant: quant,
                    entries: &entries,
                    offset: &offset
                )
            case .attention:
                appendAttentionWeights(
                    layerIdx: layerIdx,
                    layerType: layerType,
                    cfg: cfg,
                    quant: quant,
                    isKVShared: ir.isKVSharedLayer(layerIdx),
                    entries: &entries,
                    offset: &offset
                )

            case .sliding, .global:
                appendAttentionWeights(
                    layerIdx: layerIdx,
                    layerType: layerType,
                    cfg: cfg,
                    quant: quant,
                    isKVShared: ir.isKVSharedLayer(layerIdx),
                    entries: &entries,
                    offset: &offset
                )
            }

            appendBlockNormWeights(
                layerIdx: layerIdx,
                cfg: cfg,
                entries: &entries,
                offset: &offset
            )

            // FFN weights (same for both layer types)
            appendFFNWeights(
                layerIdx: layerIdx,
                cfg: cfg,
                quant: quant,
                entries: &entries,
                offset: &offset
            )

            if cfg.hiddenSizePerLayerInput > 0 {
                appendPerLayerResidualInputWeights(
                    layerIdx: layerIdx,
                    cfg: cfg,
                    quant: quant,
                    entries: &entries,
                    offset: &offset
                )
            }

        }

        // --- final norm: norm_weight [hiddenSize], dtype per trunk ABI ---
        entries.append(makeNormEntry(
            cfg: cfg,
            name: "norm_weight",
            shape: [cfg.hiddenSize],
            offset: &offset
        ))

        // --- lm_head ---
        // Dense trunks expose a hidden-out port (no LM head / argmax — cb0 + MTP
        // consume the hidden state); emit no head weight at all.
        if !cfg.tiedLMHead && cfg.portTopology == .tokenInLogitsOut {
            // Separate lm_head_weight — may be quantized or FP16
            appendWeightEntry(
                name: "lm_head_weight",
                shape: [cfg.vocabSize, cfg.hiddenSize],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
        }
        // When tied, generated code references embed_tokens offset directly.

        return entries
    }

    private static func appendBlockNormWeights(
        layerIdx: Int,
        cfg: SmeltConfig,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        entries.append(makeNormEntry(
            cfg: cfg,
            name: "layers_\(layerIdx)_post_attention_layernorm_weight",
            shape: [cfg.hiddenSize],
            offset: &offset
        ))
    }

    // MARK: - DeltaNet layer weights

    private static func appendDeltaWeights(
        layerIdx: Int,
        cfg: SmeltConfig,
        quant: SmeltQuantizationConfig,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        guard let delta = cfg.delta else { return }
        let prefix = "layers_\(layerIdx)_linear_attn"
        let hidden = cfg.hiddenSize
        let valueDim = delta.zDim

        // in_proj_qkv: [qkvDim, hidden]
        appendWeightEntry(
            name: "\(prefix)_in_proj_qkv_weight",
            shape: [delta.qkvDim, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // in_proj_z: [valueDim, hidden]
        appendWeightEntry(
            name: "\(prefix)_in_proj_z_weight",
            shape: [valueDim, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // in_proj_a: [numHeads, hidden]
        appendWeightEntry(
            name: "\(prefix)_in_proj_a_weight",
            shape: [delta.numHeads, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // in_proj_b: [numHeads, hidden]
        appendWeightEntry(
            name: "\(prefix)_in_proj_b_weight",
            shape: [delta.numHeads, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // conv1d_weight: [qkvDim, convKernel] FP16 (excluded)
        entries.append(makeFP16Entry(
            name: "\(prefix)_conv1d_weight",
            shape: [delta.qkvDim, delta.convKernel],
            offset: &offset
        ))

        // A_log: [numHeads] FP16 (excluded)
        entries.append(makeFP16Entry(
            name: "\(prefix)_A_log",
            shape: [delta.numHeads],
            offset: &offset
        ))

        // dt_bias: [numHeads] FP16 (excluded)
        entries.append(makeFP16Entry(
            name: "\(prefix)_dt_bias",
            shape: [delta.numHeads],
            offset: &offset
        ))

        // norm_weight: [headDim] FP16 (excluded) — gated RMS norm is per Delta head width
        entries.append(makeFP16Entry(
            name: "\(prefix)_norm_weight",
            shape: [delta.headDim],
            offset: &offset
        ))

        // out_proj: [hidden, valueDim]
        appendWeightEntry(
            name: "\(prefix)_out_proj_weight",
            shape: [hidden, valueDim],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )
    }

    // MARK: - Attention layer weights

    private static func appendAttentionWeights(
        layerIdx: Int,
        layerType: SmeltLayerType,
        cfg: SmeltConfig,
        quant: SmeltQuantizationConfig,
        isKVShared: Bool,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        guard let attn = cfg.attentionConfig(for: layerType) else { return }
        let prefix = "layers_\(layerIdx)_self_attn"
        let hidden = cfg.hiddenSize

        // q_proj: [qProjDim, hidden]. Always emitted — even external-KV
        // layers compute their own Q from the layer's hidden state.
        appendWeightEntry(
            name: "\(prefix)_q_proj_weight",
            shape: [attn.qProjDim, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )
        if attn.qkvBias {
            entries.append(makeFP16Entry(
                name: "\(prefix)_q_proj_bias",
                shape: [attn.qProjDim],
                offset: &offset
            ))
        }

        // k_proj / v_proj: skipped when externalKV is set OR this is a shared-KV
        // layer. External-KV (EAGLE-class draft model) takes K/V from a separately-loaded
        // target package's cache; a shared-KV layer (cross-layer KV sharing) reuses an
        // earlier layer's K/V. In both cases the per-layer K/V projections are absent
        // from the checkpoint and must NOT be declared here — the emitter already skips
        // their dispatches — or `SmeltWeightManifestLoader` would demand a missing tensor.
        if !attn.externalKV && !isKVShared {
            appendWeightEntry(
                name: "\(prefix)_k_proj_weight",
                shape: [attn.kProjDim, hidden],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
            if attn.qkvBias {
                entries.append(makeFP16Entry(
                    name: "\(prefix)_k_proj_bias",
                    shape: [attn.kProjDim],
                    offset: &offset
                ))
            }

            appendWeightEntry(
                name: "\(prefix)_v_proj_weight",
                shape: [attn.vProjDim, hidden],
                quant: quant,
                activationDtype: cfg.activationDtype,
                portTopology: cfg.portTopology,
                entries: &entries,
                offset: &offset
            )
            if attn.qkvBias {
                entries.append(makeFP16Entry(
                    name: "\(prefix)_v_proj_bias",
                    shape: [attn.vProjDim],
                    offset: &offset
                ))
            }
        }

        // o_proj: [hidden, qHeads * headDim]. Always emitted — draft models
        // still project the attention output back to the hidden dim.
        appendWeightEntry(
            name: "\(prefix)_o_proj_weight",
            shape: [hidden, attn.qHeads * attn.headDim],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // q_norm_weight / k_norm_weight: optional per-head norms. External-KV and
        // shared-KV layers ship q_norm only — the matching k_norm belongs to whoever
        // computes K (the target, or the shared source layer).
        if attn.qkNorm {
            entries.append(makeNormEntry(
                cfg: cfg,
                name: "\(prefix)_q_norm_weight",
                shape: [attn.headDim],
                offset: &offset
            ))
            if !attn.externalKV && !isKVShared {
                entries.append(makeNormEntry(
                    cfg: cfg,
                    name: "\(prefix)_k_norm_weight",
                    shape: [attn.headDim],
                    offset: &offset
                ))
            }
        }
    }

    // MARK: - FFN weights

    private static func appendFFNWeights(
        layerIdx: Int,
        cfg: SmeltConfig,
        quant: SmeltQuantizationConfig,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        let prefix = "layers_\(layerIdx)_mlp"
        let hidden = cfg.hiddenSize
        let ffnDim = cfg.ffnDim(for: layerIdx)

        // gate_proj: [ffnDim, hidden]
        appendWeightEntry(
            name: "\(prefix)_gate_proj_weight",
            shape: [ffnDim, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // up_proj: [ffnDim, hidden]
        appendWeightEntry(
            name: "\(prefix)_up_proj_weight",
            shape: [ffnDim, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )

        // down_proj: [hidden, ffnDim]
        appendWeightEntry(
            name: "\(prefix)_down_proj_weight",
            shape: [hidden, ffnDim],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )
    }

    private static func appendPerLayerResidualInputWeights(
        layerIdx: Int,
        cfg: SmeltConfig,
        quant: SmeltQuantizationConfig,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        let prefix = "layers_\(layerIdx)"
        let hidden = cfg.hiddenSize
        let perLayerHidden = cfg.hiddenSizePerLayerInput

        appendWeightEntry(
            name: "\(prefix)_per_layer_input_gate_weight",
            shape: [perLayerHidden, hidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )
        appendWeightEntry(
            name: "\(prefix)_per_layer_projection_weight",
            shape: [hidden, perLayerHidden],
            quant: quant,
            activationDtype: cfg.activationDtype,
            portTopology: cfg.portTopology,
            entries: &entries,
            offset: &offset
        )
        entries.append(makeFP16Entry(
            name: "\(prefix)_post_per_layer_input_norm_weight",
            shape: [hidden],
            offset: &offset
        ))
    }

    // MARK: - Entry builders

    /// Decide whether to create an FP16, LUT, or affine entry based on strategy
    /// and exclusion patterns.
    private static func appendWeightEntry(
        name: String,
        shape: [Int],
        quant: SmeltQuantizationConfig,
        activationDtype: SmeltDType,
        portTopology: SmeltPortTopology,
        entries: inout [SmeltWeightEntry],
        offset: inout UInt64
    ) {
        // Embeddings-in/hidden-out trunks retain authoritative BF16 projection
        // storage. Port topology, not activation dtype, selects this policy.
        if portTopology == .embeddingsInHiddenOut {
            entries.append(makeRawEntry(name: name, shape: shape, dtype: .bf16,
                                        bytesPerElement: 2, offset: &offset))
            return
        }
        // preserve_native (fp16-act LLM): an eligible projection matching a preserve_native glob is
        // kept at native bf16 (the U2 fp16_matvec_bf16w kernel) instead of fp16-downcast/quantized.
        // BEFORE the exclude/fp16/affine/lut branches so a preserved tensor leaves quant/GPTQ scope.
        // Optimistic .bf16 tag (2 bytes, same size as fp16 — offset-neutral vs the downcast path);
        // the quantizer validates the real source is BF16. dtype-building-blocks plan U2c.
        if SmeltWeightRole.preservesNativeBF16(name: name, activationDtype: activationDtype,
                                               config: quant) {
            entries.append(makeRawEntry(name: name, shape: shape, dtype: .bf16,
                                        bytesPerElement: 2, offset: &offset))
            return
        }
        let excluded = isExcludedFromQuantization(
            name: name,
            patterns: quant.excludePatterns
        )
        if excluded || quant.strategy == .fp16 {
            entries.append(makeFP16Entry(
                name: name,
                shape: shape,
                offset: &offset
            ))
        } else if quant.strategy == .affineU4 {
            entries.append(makeAffineEntry(
                name: name,
                shape: shape,
                groupSize: quant.groupSize,
                offset: &offset
            ))
        } else if quant.strategy == .binary1 || quant.strategy == .ternary2 {
            entries.append(makeSignedEntry(
                name: name,
                shape: shape,
                dtype: quant.strategy == .binary1 ? .binary1 : .ternary2,
                groupSize: quant.groupSize,
                offset: &offset
            ))
        } else {
            entries.append(makeLUTEntry(
                name: name,
                shape: shape,
                groupSize: quant.groupSize,
                offset: &offset
            ))
        }
    }

    /// dense-trunk entries (W0.3, docs/talker-trunk-fit-audit.md): raw
    /// unquantized storage — bf16 projections (the fused *_bf16w_f32
    /// kernels read the checkpoint's bf16 bits directly) and fp32 norms
    /// (the f32 norm kernels read float weights).
    private static func makeRawEntry(
        name: String,
        shape: [Int],
        dtype: SmeltDType,
        bytesPerElement: Int,
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        let elements = shape.reduce(1, *)
        let sizeBytes = UInt64(elements * bytesPerElement)
        let alignedOffset = alignTo16(offset)
        offset = alignedOffset + sizeBytes
        return SmeltWeightEntry(
            name: name,
            offset: alignedOffset,
            sizeBytes: sizeBytes,
            shape: shape,
            dtype: dtype
        )
    }

    /// Norm storage is the operation cell's declared activation family:
    /// FP16, BF16, and FP32 each retain their corresponding raw dtype.
    private static func makeNormEntry(
        cfg: SmeltConfig,
        name: String,
        shape: [Int],
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        if cfg.activationDtype == .fp32 {
            return makeRawEntry(name: name, shape: shape, dtype: .fp32,
                                bytesPerElement: 4, offset: &offset)
        }
        if cfg.activationDtype == .bf16 {
            return makeRawEntry(name: name, shape: shape, dtype: .bf16,
                                bytesPerElement: 2, offset: &offset)
        }
        return makeFP16Entry(name: name, shape: shape, offset: &offset)
    }

    /// Create an FP16 weight entry, advancing offset with 16-byte alignment.
    private static func makeFP16Entry(
        name: String,
        shape: [Int],
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        let elements = shape.reduce(1, *)
        let sizeBytes = UInt64(elements * 2)  // FP16 = 2 bytes
        let alignedOffset = alignTo16(offset)
        offset = alignedOffset + sizeBytes

        return SmeltWeightEntry(
            name: name,
            offset: alignedOffset,
            sizeBytes: sizeBytes,
            shape: shape,
            dtype: .fp16
        )
    }

    /// Create a u4_lut weight entry (indices + LUT), advancing offset.
    private static func makeLUTEntry(
        name: String,
        shape: [Int],
        groupSize: Int,
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        let paddedRows = ((rows + groupSize - 1) / groupSize) * groupSize
        let numGroups = paddedRows / groupSize
        let totalElements = paddedRows * cols

        // Packed u4: 2 indices per byte
        let packedSize = UInt64((totalElements + 1) / 2)
        // LUT: numGroups * 16 entries * 2 bytes (FP16)
        let lutSize = UInt64(numGroups * 16 * 2)

        let packedRowStride = (cols + 1) / 2  // bytes per row of packed indices

        // Indices block
        let indicesOffset = alignTo16(offset)
        // LUT block (immediately after indices, also aligned)
        let lutOffset = alignTo16(indicesOffset + packedSize)
        offset = lutOffset + lutSize

        return SmeltWeightEntry(
            name: name,
            offset: indicesOffset,
            sizeBytes: packedSize,
            shape: shape,
            dtype: .u4Lut,
            groupSize: groupSize,
            lutOffset: lutOffset,
            lutSizeBytes: lutSize,
            packedRowStride: packedRowStride,
            paddedCols: cols
        )
    }

    /// Create an affine_u4 weight entry (packed nibbles + scales + biases),
    /// advancing offset. Groups along columns with the specified group size.
    /// These blocks are 128-byte aligned to keep the decode hot path from
    /// landing on 64-byte-shifted starts for the affine matvec kernels.
    ///
    /// Layout: [packed data] [scales] [biases]
    /// - Packed: [R, C/2] bytes (2 nibbles per byte, low nibble first)
    /// - Scales: [R, numColGroups] float16
    /// - Biases: [R, numColGroups] float16
    private static func makeAffineEntry(
        name: String,
        shape: [Int],
        groupSize: Int,
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1

        // Packed u4: 2 values per byte, row-major
        let packedSize = UInt64(rows * ((cols + 1) / 2))
        let packedRowStride = (cols + 1) / 2

        // Column groups for scale/bias
        let numColGroups = (cols + groupSize - 1) / groupSize
        let scalesBiasElements = rows * numColGroups
        let scalesSize = UInt64(scalesBiasElements * 2)  // FP16 = 2 bytes
        let biasesSize = UInt64(scalesBiasElements * 2)

        let packedOffset = alignTo128(offset)
        let scalesOffset = alignTo128(packedOffset + packedSize)
        let biasesOffset = alignTo128(scalesOffset + scalesSize)
        offset = biasesOffset + biasesSize

        return SmeltWeightEntry(
            name: name,
            offset: packedOffset,
            sizeBytes: packedSize,
            shape: shape,
            dtype: .affineU4,
            groupSize: groupSize,
            packedRowStride: packedRowStride,
            paddedCols: cols,
            scalesOffset: scalesOffset,
            scalesSizeBytes: scalesSize,
            biasesOffset: biasesOffset,
            biasesSizeBytes: biasesSize
        )
    }

    /// Create a canonical signed binary/ternary entry: row-major LSB-first
    /// semantic codes plus one fp16 multiplier per column group. There is no
    /// bias plane and no package-specific ternary transpose.
    private static func makeSignedEntry(
        name: String,
        shape: [Int],
        dtype: SmeltDType,
        groupSize: Int,
        offset: inout UInt64
    ) -> SmeltWeightEntry {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        let paddedCols = ((cols + groupSize - 1) / groupSize) * groupSize
        let bits = dtype == .binary1 ? 1 : 2
        let packedRowStride = (paddedCols * bits + 7) / 8
        let packedSize = UInt64(rows * packedRowStride)
        let scalesSize = UInt64(rows * (paddedCols / groupSize) * 2)

        let packedOffset = alignTo128(offset)
        let scalesOffset = alignTo128(packedOffset + packedSize)
        offset = scalesOffset + scalesSize

        return SmeltWeightEntry(
            name: name,
            offset: packedOffset,
            sizeBytes: packedSize,
            shape: shape,
            dtype: dtype,
            groupSize: groupSize,
            packedRowStride: packedRowStride,
            paddedCols: paddedCols,
            scalesOffset: scalesOffset,
            scalesSizeBytes: scalesSize
        )
    }

    /// Round up to the next 16-byte boundary.
    private static func alignTo16(_ value: UInt64) -> UInt64 {
        (value + 15) & ~15
    }

    /// Round up to the next 128-byte boundary.
    private static func alignTo128(_ value: UInt64) -> UInt64 {
        (value + 127) & ~127
    }
}

// MARK: - Glob pattern matching

/// Check if a weight name matches any of the exclusion patterns.
/// Supports simple glob: `*` matches any substring, otherwise substring match.
internal func isExcludedFromQuantization(
    name: String,
    patterns: [String]
) -> Bool {
    for pattern in patterns {
        if matchesGlob(name: name, pattern: pattern) {
            return true
        }
    }
    return false
}

/// Match `name` against `patterns` using the SAFE exact-or-glob convention shared by
/// turbo_quant_h (SmeltAffineQuantizer) and preserve_native (SmeltWeightRole): a pattern
/// WITHOUT `*` must match the name EXACTLY — plain substring-glob is too loose here, e.g.
/// `embed_tokens` would also catch `embed_tokens_per_layer` and silently route it through a
/// destructive quant/preserve path. A pattern WITH `*` uses the glob matcher. Per-layer tensors
/// are layer-prefixed, so a pattern meant to span layers must use a wildcard (e.g. `*_q_proj_weight`).
internal func matchesExactOrGlob(name: String, patterns: [String]) -> Bool {
    for pattern in patterns {
        if pattern.contains("*") {
            if matchesGlob(name: name, pattern: pattern) { return true }
        } else if name == pattern {
            return true
        }
    }
    return false
}

/// Simple glob matching: `*` matches zero or more characters.
/// Without `*`, checks for substring containment (matches Python behavior).
private func matchesGlob(name: String, pattern: String) -> Bool {
    if pattern.contains("*") {
        // Split on * and check all literal segments appear in order.
        let segments = pattern.split(separator: "*", omittingEmptySubsequences: false)
            .map(String.init)
        var cursor = name.startIndex

        for (idx, segment) in segments.enumerated() {
            if segment.isEmpty { continue }

            if idx == 0 && !pattern.hasPrefix("*") {
                // First segment must be a prefix
                guard name.hasPrefix(segment) else { return false }
                cursor = name.index(name.startIndex, offsetBy: segment.count)
            } else if idx == segments.count - 1 && !pattern.hasSuffix("*") {
                // Last segment must be a suffix
                guard name.hasSuffix(segment) else { return false }
                // Also verify it's after cursor
                let suffixStart = name.index(name.endIndex, offsetBy: -segment.count)
                guard suffixStart >= cursor else { return false }
                cursor = name.endIndex
            } else {
                // Middle segment: find next occurrence after cursor
                guard let range = name.range(
                    of: segment, range: cursor..<name.endIndex
                ) else { return false }
                cursor = range.upperBound
            }
        }
        return true
    } else {
        // No wildcard: substring containment (matches Python's `pattern in name`)
        return name.contains(pattern)
    }
}

// MARK: - Weight packer

/// Packs individual weight files into a monolithic weights.bin.
public struct SmeltWeightPacker {

    private enum LayoutSource {
        case ir(SmeltModelIR)
        case compilationPlan(SmeltCompilationPlan)
    }

    private let layoutSource: LayoutSource

    /// SHA-256 hex digest of the packed weights.bin, computed after packing.
    public private(set) var weightsChecksum: String = ""

    /// Initialize with the model IR to know expected weight shapes.
    public init(ir: SmeltModelIR) {
        self.layoutSource = .ir(ir)
    }

    init(compilationPlan: SmeltCompilationPlan) {
        self.layoutSource = .compilationPlan(compilationPlan)
    }

    /// Pack weight files from sourceDir into a single weights.bin at outputDir.
    ///
    /// Source file naming convention:
    ///   - `{name}.bin` for packed u4 indices or FP16 data
    ///   - `{name}_lut.bin` for per-group FP16 LUTs
    ///
    /// - Returns: SmeltWeightManifest describing the packed layout.
    public mutating func packWeights(
        sourceDir: String,
        outputDir: String
    ) throws -> SmeltWeightManifest {
        let fm = FileManager.default

        // Ensure output directory exists
        if !fm.fileExists(atPath: outputDir) {
            do {
                try fm.createDirectory(
                    atPath: outputDir,
                    withIntermediateDirectories: true
                )
            } catch {
                throw SmeltWeightPackerError
                    .outputDirectoryCreationFailed(path: outputDir)
            }
        }

        let plannedLayout = try plannedWeightLayout()
        let layout = plannedLayout.entries
        let outputPath = "\(outputDir)/weights.bin"

        // Compute total file size from layout
        var totalSize: UInt64 = 0
        for entry in layout {
            var entryEnd = entry.offset + entry.sizeBytes
            if let lutOff = entry.lutOffset, let lutSize = entry.lutSizeBytes {
                let lutEnd = lutOff + lutSize
                if lutEnd > entryEnd { entryEnd = lutEnd }
            }
            if let biasOff = entry.biasesOffset, let biasSize = entry.biasesSizeBytes {
                let biasEnd = biasOff + biasSize
                if biasEnd > entryEnd { entryEnd = biasEnd }
            }
            if entryEnd > totalSize {
                totalSize = entryEnd
            }
        }

        // Allocate zero-filled buffer
        var buffer = Data(count: Int(totalSize))

        // Pack each weight
        for entry in layout {
            let dataPath = "\(sourceDir)/\(entry.name).bin"
            guard fm.fileExists(atPath: dataPath) else {
                throw SmeltWeightPackerError.sourceFileNotFound(
                    name: entry.name,
                    path: dataPath
                )
            }

            let data = try Data(contentsOf: URL(fileURLWithPath: dataPath))
            if data.count != Int(entry.sizeBytes) {
                throw SmeltWeightPackerError.sizeMismatch(
                    name: entry.name,
                    expected: Int(entry.sizeBytes),
                    actual: data.count
                )
            }
            buffer.replaceSubrange(
                Int(entry.offset)..<(Int(entry.offset) + data.count),
                with: data
            )

            // For u4_lut, also pack the LUT file
            if entry.dtype == .u4Lut,
               let lutOffset = entry.lutOffset,
               let lutSize = entry.lutSizeBytes
            {
                let lutPath = "\(sourceDir)/\(entry.name)_lut.bin"
                guard fm.fileExists(atPath: lutPath) else {
                    throw SmeltWeightPackerError.sourceFileNotFound(
                        name: "\(entry.name)_lut",
                        path: lutPath
                    )
                }

                let lutData = try Data(
                    contentsOf: URL(fileURLWithPath: lutPath)
                )
                if lutData.count != Int(lutSize) {
                    throw SmeltWeightPackerError.sizeMismatch(
                        name: "\(entry.name)_lut",
                        expected: Int(lutSize),
                        actual: lutData.count
                    )
                }
                buffer.replaceSubrange(
                    Int(lutOffset)..<(Int(lutOffset) + lutData.count),
                    with: lutData
                )
            }

            // For affine_u4, pack scales and biases files
            if entry.dtype == .affineU4 {
                if let scOff = entry.scalesOffset, let scSize = entry.scalesSizeBytes {
                    let scPath = "\(sourceDir)/\(entry.name)_scales.bin"
                    guard fm.fileExists(atPath: scPath) else {
                        throw SmeltWeightPackerError.sourceFileNotFound(
                            name: "\(entry.name)_scales",
                            path: scPath
                        )
                    }
                    let scData = try Data(contentsOf: URL(fileURLWithPath: scPath))
                    if scData.count != Int(scSize) {
                        throw SmeltWeightPackerError.sizeMismatch(
                            name: "\(entry.name)_scales",
                            expected: Int(scSize),
                            actual: scData.count
                        )
                    }
                    buffer.replaceSubrange(
                        Int(scOff)..<(Int(scOff) + scData.count),
                        with: scData
                    )
                }
                if let biOff = entry.biasesOffset, let biSize = entry.biasesSizeBytes {
                    let biPath = "\(sourceDir)/\(entry.name)_biases.bin"
                    guard fm.fileExists(atPath: biPath) else {
                        throw SmeltWeightPackerError.sourceFileNotFound(
                            name: "\(entry.name)_biases",
                            path: biPath
                        )
                    }
                    let biData = try Data(contentsOf: URL(fileURLWithPath: biPath))
                    if biData.count != Int(biSize) {
                        throw SmeltWeightPackerError.sizeMismatch(
                            name: "\(entry.name)_biases",
                            expected: Int(biSize),
                            actual: biData.count
                        )
                    }
                    buffer.replaceSubrange(
                        Int(biOff)..<(Int(biOff) + biData.count),
                        with: biData
                    )
                }
            }
        }

        // Write the packed file
        do {
            try buffer.write(to: URL(fileURLWithPath: outputPath))
        } catch {
            throw SmeltWeightPackerError.fileWriteFailed(
                path: outputPath,
                underlying: error
            )
        }

        // Compute SHA-256 for manifest checksums
        let digest = SHA256.hash(data: buffer)
        self.weightsChecksum = digest.map {
            String(format: "%02x", $0)
        }.joined()

        return SmeltWeightManifest(
            totalBytes: totalSize,
            entries: layout
        )
    }

    private func plannedWeightLayout() throws -> SmeltPlannedWeightLayout {
        switch layoutSource {
        case .ir(let ir):
            let compilationPlan = try SmeltCompiler.planCompilation(ir: ir)
            return SmeltPlannedWeightLayout(
                entries: compilationPlan.plannedWeightEntries,
                storagePlan: compilationPlan.weightStoragePlan
            )
        case .compilationPlan(let compilationPlan):
            try SmeltCompiler.validateWeightStoragePlan(
                compilationPlan.weightStoragePlan,
                policy: compilationPlan.policy.weightLayout
            )
            return SmeltPlannedWeightLayout(
                entries: compilationPlan.plannedWeightEntries,
                storagePlan: compilationPlan.weightStoragePlan
            )
        }
    }

    /// Compute SHA-256 hex digest of a file at the given path.
    public static func sha256(ofFileAt path: String) throws -> String {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
