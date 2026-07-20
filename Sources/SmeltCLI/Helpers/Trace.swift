import Foundation
import SmeltRuntime
import SmeltSchema

func traceLayerIndex(from label: String) -> Int? {
    guard label.first == "L" else { return nil }
    let digits = label.dropFirst().prefix { $0.isNumber }
    return Int(digits)
}

enum TraceOccurrenceSelection {
    case first
    case last
    case index(Int)
}

func parseTraceOccurrenceSelection(_ raw: String) throws -> TraceOccurrenceSelection {
    switch raw {
    case "", "last":
        return .last
    case "first":
        return .first
    default:
        guard let index = Int(raw), index >= 0 else {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "--dump-trace-occurrence must be 'first', 'last', or a non-negative integer"
                ]
            )
        }
        return .index(index)
    }
}

func selectTraceMarker(
    label: String,
    markers: [SmeltTraceMarker],
    occurrence: TraceOccurrenceSelection
) -> SmeltTraceMarker? {
    let matches = markers.filter { $0.label == label }
    switch occurrence {
    case .first:
        return matches.first
    case .last:
        return matches.last
    case .index(let index):
        guard index < matches.count else { return nil }
        return matches[index]
    }
}

func traceMarkerMatches(
    label: String,
    markers: [SmeltTraceMarker]
) -> [SmeltTraceMarker] {
    markers.filter { $0.label == label }
}

enum DebugTraceMode {
    case decode
    case prefill
}

func captureTraceSample(
    packagePath: String,
    tokenIds: [Int32],
    marker: SmeltTraceMarker,
    mode: DebugTraceMode,
    usesBatchedRowOffset: Bool,
    contextLimit: Int?,
    count: Int = 32,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> (token: Int32, values: [Float]) {
    try construction?.requirePackagePath(packagePath)
    let runtime = try construction?.makeRuntime(contextLimit: contextLimit)
        ?? SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)
    let manifest = try loadManifest(packagePath: packagePath)
    guard !tokenIds.isEmpty else {
        return (0, [])
    }

    let token: Int32
    switch mode {
    case .prefill:
        token = try runtime.debugPrefillStep(
            tokenIds: tokenIds,
            startPos: 0,
            maxDispatches: marker.dispatchCount
        )
    case .decode:
        runtime.resetWorkingBuffers()
        if tokenIds.count > 1 {
            for position in 0..<(tokenIds.count - 1) {
                _ = try runtime.decodeStep(
                    tokenId: tokenIds[position],
                    position: Int32(position)
                )
            }
        }
        token = try runtime.debugDecodeStep(
            tokenId: tokenIds[tokenIds.count - 1],
            position: Int32(tokenIds.count - 1),
            maxDispatches: marker.dispatchCount
        )
    }

    let isBatchedSlot = mode == .prefill && tokenIds.count > 1 && usesBatchedRowOffset
    let rowStride = batchedRowStrideElements(
        for: marker.bufferSlot,
        manifest: manifest,
        compiledMaxBatch: max(runtime.maxPrefillBatchSize, 1),
        marker: marker
    )
    let elementOffset = isBatchedSlot ? (tokenIds.count - 1) * rowStride : 0
    let asFP32 = manifest.buffers.slots.first(
        where: { $0.index == marker.bufferSlot }
    )?.dtype == .fp32
    return (
        token,
        runtime.dumpSlot(
            marker.bufferSlot,
            elementOffset: elementOffset,
            count: max(count, 0),
            asFP32: asFP32
        )
    )
}

func runLayerTraceDebug(
    packagePath: String,
    prompt: String,
    inputIds: [Int32],
    tokenizer: SmeltTokenizer,
    contextLimit: Int?,
    construction: CAMTextRuntimeConstruction? = nil
) throws {
    let markers = try loadTraceMarkers(packagePath: packagePath)
    guard !markers.decode.isEmpty, !markers.prefill.isEmpty else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Package has no trace_markers.json or it is missing decode/prefill markers"
            ]
        )
    }

    let decodeByLabel = Dictionary(uniqueKeysWithValues: markers.decode.map { ($0.label, $0) })
    let commonMarkers = markers.prefill
        .filter { decodeByLabel[$0.label] != nil }
        .sorted { a, b in
            if a.dispatchCount != b.dispatchCount { return a.dispatchCount < b.dispatchCount }
            return a.label < b.label
        }

    guard !commonMarkers.isEmpty else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Package trace markers do not share any decode/prefill labels"
            ]
        )
    }

    fputs("Loaded: \(packagePath)\n", stderr)
    fputs("Prompt: \(prompt)\n", stderr)
    fputs("Prompt tokens: \(inputIds.count)\n", stderr)
    fputs("Tracing \(commonMarkers.count) shared decode/prefill markers\n", stderr)

    var firstBadLabel: String?
    for prefillMarker in commonMarkers {
        let decodeMarker = decodeByLabel[prefillMarker.label]!
        let prefillMatches = traceMarkerMatches(
            label: prefillMarker.label,
            markers: markers.prefill
        )
        let prefillSample = try captureTraceSample(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: prefillMarker,
            mode: .prefill,
            usesBatchedRowOffset: prefillMatches.count <= 1,
            contextLimit: contextLimit,
            construction: construction
        )
        let decodeSample = try captureTraceSample(
            packagePath: packagePath,
            tokenIds: inputIds,
            marker: decodeMarker,
            mode: .decode,
            usesBatchedRowOffset: false,
            contextLimit: contextLimit,
            construction: construction
        )

        let count = min(prefillSample.values.count, decodeSample.values.count)
        var maxDiff: Float = 0
        for idx in 0..<count {
            maxDiff = max(maxDiff, abs(prefillSample.values[idx] - decodeSample.values[idx]))
        }

        fputs(
            "  \(prefillMarker.label):"
                + " slot=\(prefillMarker.bufferSlot)"
                + " prefill@\(prefillMarker.dispatchCount)"
                + " decode@\(decodeMarker.dispatchCount)"
                + " maxDiff=\(String(format: "%.6f", maxDiff))\n",
            stderr
        )

        if maxDiff > 0.01, firstBadLabel == nil {
            firstBadLabel = prefillMarker.label
            fputs("  first divergence: \(prefillMarker.label)\n", stderr)
            fputs(
                "    prefill[0:12]=\(prefillSample.values.prefix(12).map { String(format: "%.4f", $0) })\n",
                stderr
            )
            fputs(
                "    decode [0:12]=\(decodeSample.values.prefix(12).map { String(format: "%.4f", $0) })\n",
                stderr
            )
            fputs(
                "    prefill token=\(prefillSample.token) (\(tokenizer.decode([prefillSample.token])))\n",
                stderr
            )
            fputs(
                "    decode  token=\(decodeSample.token) (\(tokenizer.decode([decodeSample.token])))\n",
                stderr
            )
            break
        }
    }

    if firstBadLabel == nil {
        fputs("  trace matched across all markers\n", stderr)
    }
}

private func loadManifest(packagePath: String) throws -> SmeltManifest {
    let path = "\(packagePath)/manifest.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try SmeltManifest.decode(from: data)
}

private func batchedRowStrideElements(
    for slot: Int,
    manifest: SmeltManifest,
    compiledMaxBatch: Int,
    marker: SmeltTraceMarker? = nil
) -> Int {
    guard let slotInfo = manifest.buffers.slots.first(where: { $0.index == slot }),
          compiledMaxBatch > 0
    else {
        return manifest.config.hiddenSize
    }

    if slotInfo.name == "ffnIntBuf",
       let marker,
       let layerIndex = traceLayerIndex(from: marker.label),
       let ffnDim = manifest.weights.entries.first(
           where: { $0.name == "layers_\(layerIndex)_mlp_gate_proj_weight" }
       )?.shape.first
    {
        return ffnDim
    }

    if let marker,
       let layerIndex = traceLayerIndex(from: marker.label)
    {
        switch slotInfo.name {
        case "attnQBuf":
            if let dim = manifest.weights.entries.first(
                where: { $0.name == "layers_\(layerIndex)_self_attn_q_proj_weight" }
            )?.shape.first {
                return dim
            }
        case "attnKBuf":
            if let dim = manifest.weights.entries.first(
                where: { $0.name == "layers_\(layerIndex)_self_attn_k_proj_weight" }
            )?.shape.first {
                return dim
            }
        case "attnVBuf":
            if let dim = manifest.weights.entries.first(
                where: { $0.name == "layers_\(layerIndex)_self_attn_v_proj_weight" }
            )?.shape.first {
                return dim
            }
        case "attnOutBuf":
            if let dim = manifest.weights.entries.first(
                where: { $0.name == "layers_\(layerIndex)_self_attn_o_proj_weight" }
            )?.shape.dropFirst().first {
                return dim
            }
        case "ffnGateBuf", "ffnUpBuf":
            if let dim = manifest.weights.entries.first(
                where: { $0.name == "layers_\(layerIndex)_mlp_gate_proj_weight" }
            )?.shape.first {
                return dim
            }
        default:
            break
        }
    }

    guard let bytesPerElement = slotInfo.dtype.bytesPerElement else {
        return manifest.config.hiddenSize
    }
    let totalElements = slotInfo.sizeBytes / bytesPerElement
    guard totalElements > 0, totalElements % compiledMaxBatch == 0 else {
        return manifest.config.hiddenSize
    }
    return totalElements / compiledMaxBatch
}
