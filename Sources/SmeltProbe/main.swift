import Foundation
import CryptoKit
import Metal
import SmeltCompiler
import SmeltRuntime
import SmeltSchema

private struct UsageError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

private func parseArg(
    _ name: String,
    from args: [String],
    default defaultValue: String? = nil
) throws -> String {
    guard let idx = args.firstIndex(of: name) else {
        if let defaultValue { return defaultValue }
        throw UsageError(message: "Missing required argument \(name)")
    }
    let valueIdx = idx + 1
    guard valueIdx < args.count else {
        throw UsageError(message: "Missing value for \(name)")
    }
    return args[valueIdx]
}

private func parseOptionalPositiveIntArg(_ name: String, from args: [String]) throws -> Int? {
    guard let raw = try? parseArg(name, from: args) else { return nil }
    guard let parsed = Int(raw), parsed > 0 else {
        throw UsageError(message: "\(name) must be a positive integer")
    }
    return parsed
}

private func parseCSVInt32(_ text: String) throws -> [Int32] {
    let values = text
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    guard !values.isEmpty else {
        throw UsageError(message: "Expected at least one token id")
    }
    return try values.map {
        guard let v = Int32($0) else {
            throw UsageError(message: "Invalid Int32 token id '\($0)'")
        }
        return v
    }
}

private func format(_ value: Float) -> String {
    String(format: "%.4f", Double(value))
}

private func formatExact(_ value: Float) -> String {
    String(format: "%.8g", Double(value))
}

private func diffMagnitude(_ lhs: Float, _ rhs: Float) -> Float {
    if lhs.isFinite && rhs.isFinite {
        return abs(lhs - rhs)
    }
    return lhs == rhs ? 0 : .infinity
}

private enum TraceOccurrenceSelection {
    case first
    case last
    case index(Int)
}

private struct PrefillChunkExecution {
    let tokenIds: [Int32]
    let startPos: Int
}

private struct ResolvedTraceSample {
    let marker: SmeltTraceMarker
    let dispatchRecordIndex: Int
    let dispatchOrdinal: Int
    let pipelineName: String
    let slot: Int
    let elementOffset: Int
    let bindingIndex: Int
    let asFP32: Bool
    let rowWidth: Int
    let values: [Float]
}

private func parseTraceOccurrenceSelection(_ raw: String) throws -> TraceOccurrenceSelection {
    switch raw {
    case "", "last":
        return .last
    case "first":
        return .first
    default:
        guard let index = Int(raw), index >= 0 else {
            throw UsageError(
                message: "--occurrence must be 'first', 'last', or a non-negative integer"
            )
        }
        return .index(index)
    }
}

private func loadManifest(packagePath: String) throws -> SmeltManifest {
    let path = "\(packagePath)/manifest.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try SmeltManifest.decode(from: data)
}

private func loadTraceMarkers(packagePath: String) throws -> SmeltTraceMarkers {
    let path = "\(packagePath)/trace_markers.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(SmeltTraceMarkers.self, from: data)
}

private func loadDispatchTable(
    packagePath: String,
    prefill: Bool
) throws -> [SmeltDispatchRecord] {
    let fileName = prefill ? "prefill_dispatches.bin" : "dispatches.bin"
    let path = "\(packagePath)/\(fileName)"
    return try loadDispatchTable(path: path)
}

private func loadDispatchTable(path: String) throws -> [SmeltDispatchRecord] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let stride = MemoryLayout<SmeltDispatchRecord>.stride
    guard data.count % stride == 0 else {
        throw UsageError(message: "\(path) has invalid dispatch-table size")
    }
    return data.withUnsafeBytes { raw in
        let count = data.count / stride
        let ptr = raw.bindMemory(to: SmeltDispatchRecord.self)
        return Array(UnsafeBufferPointer(start: ptr.baseAddress, count: count))
    }
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func sha256Hex(values: [Float], dtype: SmeltDType) -> String? {
    switch dtype {
    case .fp16:
        let typed = values.map(Float16.init)
        return typed.withUnsafeBytes { sha256Hex(Data($0)) }
    case .fp32:
        return values.withUnsafeBytes { sha256Hex(Data($0)) }
    default:
        return nil
    }
}

private func selectTraceMarker(
    label: String,
    markers: [SmeltTraceMarker],
    occurrence: TraceOccurrenceSelection
) throws -> SmeltTraceMarker {
    let matches = markers.filter { $0.label == label }
    guard !matches.isEmpty else {
        throw UsageError(message: "No trace marker named '\(label)'")
    }
    switch occurrence {
    case .first:
        return matches[0]
    case .last:
        return matches[matches.count - 1]
    case .index(let index):
        guard index < matches.count else {
            throw UsageError(
                message: "Trace marker '\(label)' occurrence \(index) is out of range (\(matches.count) matches)"
            )
        }
        return matches[index]
    }
}

private func traceLayerIndex(from label: String) -> Int? {
    guard label.first == "L" else { return nil }
    let digits = label.dropFirst().prefix { $0.isNumber }
    return Int(digits)
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
            if marker.label.hasSuffix(".attn_raw"),
               let dim = manifest.weights.entries.first(
                   where: { $0.name == "layers_\(layerIndex)_self_attn_o_proj_weight" }
               )?.shape.dropFirst().first {
                return dim
            }
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
    if slotInfo.category == .state {
        return totalElements
    }
    guard totalElements > 0, totalElements % compiledMaxBatch == 0 else {
        return manifest.config.hiddenSize
    }
    return totalElements / compiledMaxBatch
}

private func isBatchedSlot(
    slot: Int,
    manifest: SmeltManifest,
    compiledMaxBatch: Int
) -> Bool {
    guard let slotInfo = manifest.buffers.slots.first(where: { $0.index == slot }),
          compiledMaxBatch > 1
    else {
        return false
    }

    guard let bytesPerElement = slotInfo.dtype.bytesPerElement else {
        return false
    }
    guard slotInfo.category != .state else { return false }
    let totalElements = slotInfo.sizeBytes / bytesPerElement
    return totalElements > 0 && totalElements % compiledMaxBatch == 0
}

private func compiledMaxBatch(manifest: SmeltManifest) -> Int {
    if let prefill = manifest.prefill?.maxBatchSize, prefill > 0 {
        return prefill
    }
    return 1
}

private func resolveSlot(_ slot: Int16, cur: Int, alt: Int) -> Int {
    switch slot {
    case SmeltBufferRecord.slotCur:
        return cur
    case SmeltBufferRecord.slotAlt:
        return alt
    default:
        return Int(slot)
    }
}

private enum TraceMode {
    case prefill
    case decode
}

private enum RowSelection {
    case base
    case last
    case index(Int)
}

private struct ResolvedDispatchContext {
    let manifest: SmeltManifest
    let marker: SmeltTraceMarker?
    let runtime: SmeltRuntime
    let mode: TraceMode
    let ids: [Int32]
    let startPos: Int
    let position: Int
    let dispatchOrdinal: Int
    let recordIndex: Int
    let pipelineName: String
    let record: SmeltDispatchRecord
    let cur: Int
    let alt: Int
}

private struct ResolvedBinding {
    let bufferIndex: Int
    let metalBindingIndex: Int
    let rawSlot: Int
    let resolvedSlot: Int
    let slotName: String
    let dtype: SmeltDType
    let offsetKind: Int
    let offsetBytes: Int
    let baseElementOffset: Int
    let elementOffset: Int
    let rowWidth: Int
    let values: [Float]
}

private func roundToHalf(_ value: Float) -> Float {
    Float(Float16(value))
}

private func parseRowSelection(_ raw: String) throws -> RowSelection {
    switch raw {
    case "", "base":
        return .base
    case "last":
        return .last
    default:
        guard let index = Int(raw), index >= 0 else {
            throw UsageError(
                message: "--row-index must be 'base', 'last', or a non-negative integer"
            )
        }
        return .index(index)
    }
}

private func parseOptionalDType(_ raw: String?) throws -> SmeltDType? {
    guard let raw, !raw.isEmpty else { return nil }
    guard let dtype = SmeltDType(rawValue: raw) else {
        throw UsageError(message: "--force-dtype must be one of: fp16, fp32, int32, raw, u4Lut, affineU4")
    }
    return dtype
}

private func resolveOffsetBytes(
    _ record: SmeltBufferRecord,
    mode: TraceMode,
    position: Int,
    seqLen: Int,
    startPos: Int
) -> Int {
    switch mode {
    case .prefill:
        switch record.offsetKind {
        case 0:
            return Int(record.offset)
        case 1:
            return startPos * Int(record.offset)
        case 2:
            let stride = Int(record.offset & 0xFFFF_FFFF)
            let addend = Int(record.offset >> 32)
            return startPos * stride + addend
        case 3:
            return max(seqLen - 1, 0) * Int(record.offset)
        case 4:
            let stride = Int(record.offset & 0xFFFF_FFFF)
            let divisor = max(Int(record.offset >> 32), 1)
            return (seqLen / divisor) * stride
        default:
            return Int(record.offset)
        }
    case .decode:
        switch record.offsetKind {
        case 1:
            return position * Int(record.offset)
        default:
            return Int(record.offset)
        }
    }
}

private func decodeDispatchWouldExecute(
    _ record: SmeltDispatchRecord,
    position: Int32
) -> Bool {
    var skipDispatch = false
    for cidx in 0..<Int(record.constantCount) {
        let con = getConstant(record, index: cidx)
        switch con.kind {
        case SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: position + 1) < con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
        case SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: position + 1) >= con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
        default:
            break
        }
        if skipDispatch { return false }
    }
    return true
}

private func prefillDispatchWouldExecute(
    _ record: SmeltDispatchRecord,
    seqLen: Int32
) -> Bool {
    if record.minSeqLen > 0 && UInt32(bitPattern: seqLen) < UInt32(record.minSeqLen) {
        return false
    }

    var skipDispatch = false
    for cidx in 0..<Int(record.constantCount) {
        let con = getConstant(record, index: cidx)
        switch con.kind {
        case SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
            let val = con.value == 0 ? 0 : UInt32(bitPattern: seqLen) % con.value
            if val == 0 { skipDispatch = true }
        case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: seqLen) < con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
        default:
            break
        }
        if skipDispatch { return false }
    }
    return true
}

private func dispatchWouldExecute(
    _ record: SmeltDispatchRecord,
    mode: TraceMode,
    position: Int,
    seqLen: Int
) -> Bool {
    switch mode {
    case .decode:
        return decodeDispatchWouldExecute(record, position: Int32(position))
    case .prefill:
        return prefillDispatchWouldExecute(record, seqLen: Int32(seqLen))
    }
}

private func prefillChunks(
    runtime: SmeltRuntime,
    requestIds: [Int32]
) throws -> [PrefillChunkExecution] {
    guard runtime.hasMetalPrefill, runtime.maxPrefillBatchSize > 0 else {
        throw UsageError(message: "Package has no Metal prefill path")
    }
    guard !requestIds.isEmpty else { return [] }

    let chunkSize = max(runtime.maxPrefillBatchSize, 1)
    if requestIds.count <= chunkSize {
        return [PrefillChunkExecution(tokenIds: requestIds, startPos: 0)]
    }

    var chunks: [PrefillChunkExecution] = []
    chunks.reserveCapacity((requestIds.count + chunkSize - 1) / chunkSize)
    var start = 0
    while start < requestIds.count {
        let end = min(start + chunkSize, requestIds.count)
        chunks.append(
            PrefillChunkExecution(
                tokenIds: Array(requestIds[start..<end]),
                startPos: start
            )
        )
        start = end
    }
    return chunks
}

@discardableResult
private func executePrefillThroughChunk(
    runtime: SmeltRuntime,
    requestIds: [Int32],
    targetChunkIndex: Int? = nil,
    maxDispatches: Int? = nil
) throws -> PrefillChunkExecution {
    let chunks = try prefillChunks(runtime: runtime, requestIds: requestIds)
    guard let lastIndex = chunks.indices.last else {
        throw UsageError(message: "Expected at least one token id")
    }
    let targetIndex = targetChunkIndex ?? lastIndex
    guard chunks.indices.contains(targetIndex) else {
        throw UsageError(message: "Chunk index \(targetIndex) is out of range")
    }

    runtime.resetWorkingBuffers()
    for (index, chunk) in chunks.enumerated() {
        if index == targetIndex, let maxDispatches {
            _ = try runtime.debugPrefillStep(
                tokenIds: chunk.tokenIds,
                startPos: Int32(chunk.startPos),
                maxDispatches: maxDispatches
            )
            return chunk
        }

        _ = try runtime.prefillStep(
            tokenIds: chunk.tokenIds,
            startPos: Int32(chunk.startPos)
        )

        if index == targetIndex {
            return chunk
        }
    }

    return chunks[lastIndex]
}

private func bytesPerElement(for dtype: SmeltDType) -> Int {
    dtype.bytesPerElement ?? 0
}

private func slotInfo(
    _ slot: Int,
    manifest: SmeltManifest
) throws -> SmeltBufferSlot {
    guard let info = manifest.buffers.slots.first(where: { $0.index == slot }) else {
        throw UsageError(message: "Buffer slot \(slot) is missing from manifest")
    }
    return info
}

private func resolveDispatchContext(
    packagePath: String,
    ids: [Int32],
    mode: TraceMode,
    label: String? = nil,
    occurrence: TraceOccurrenceSelection = .last,
    dispatchOrdinal explicitDispatchOrdinal: Int? = nil,
    executeDispatch: Bool = true,
    contextLimit: Int? = nil,
    prefillCount: Int = 0
) throws -> ResolvedDispatchContext {
    let manifest = try loadManifest(packagePath: packagePath)
    let trace = try loadTraceMarkers(packagePath: packagePath)
    let markers = mode == .prefill ? trace.prefill : trace.decode
    let marker = try label.map { try selectTraceMarker(label: $0, markers: markers, occurrence: occurrence) }
    let dispatchOrdinal = explicitDispatchOrdinal ?? marker?.dispatchCount ?? 0
    guard dispatchOrdinal > 0 else {
        throw UsageError(message: "Resolved dispatch ordinal must be positive")
    }

    let table = try loadDispatchTable(packagePath: packagePath, prefill: mode == .prefill)
    let runtime = try SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)

    let activeIds: [Int32]
    let startPos: Int
    let position: Int
    switch mode {
    case .prefill:
        let execution = try executePrefillThroughChunk(
            runtime: runtime,
            requestIds: ids,
            maxDispatches: dispatchOrdinal
        )
        activeIds = execution.tokenIds
        startPos = execution.startPos
        position = startPos + max(activeIds.count - 1, 0)
    case .decode:
        guard prefillCount >= 0,
              ids.isEmpty ? prefillCount == 0 : prefillCount < ids.count else {
            throw UsageError(message: "--prefill-count must leave one decode token to inspect")
        }
        activeIds = ids
        startPos = 0
        position = max(ids.count - 1, 0)
        if executeDispatch {
            runtime.resetWorkingBuffers()
            if !ids.isEmpty {
                if prefillCount > 0 {
                    let prefix = Array(ids.prefix(prefillCount))
                    for chunk in try prefillChunks(runtime: runtime, requestIds: prefix) {
                        _ = try runtime.prefillStep(
                            tokenIds: chunk.tokenIds,
                            startPos: Int32(chunk.startPos)
                        )
                    }
                }
                for pos in prefillCount..<position {
                    _ = try runtime.decodeStep(tokenId: ids[pos], position: Int32(pos))
                }
                _ = try runtime.debugDecodeStep(
                    tokenId: ids[position],
                    position: Int32(position),
                    maxDispatches: dispatchOrdinal
                )
            }
        } else {
            try primeDecodeStateForDispatch(
                runtime: runtime,
                ids: ids,
                dispatchOrdinal: dispatchOrdinal
            )
        }
    }

    var cur = 0
    var alt = 1
    var seenDispatches = 0
    var selectedRecord: SmeltDispatchRecord?
    var selectedRecordIndex = -1
    var selectedCur = cur
    var selectedAlt = alt
    var lastExecutedRecord: SmeltDispatchRecord?
    var lastExecutedRecordIndex = -1
    var lastExecutedCur = cur
    var lastExecutedAlt = alt
    for (recordIndex, record) in table.enumerated() {
        if record.opKind == SmeltDispatchRecord.opSwap {
            let tmp = cur
            cur = alt
            alt = tmp
            continue
        }

        // Compiler trace markers count Metal dispatches only. Buffer and
        // constant records are part of the binary stream but must not advance
        // the marker ordinal, or large unrolled matmuls resolve hundreds of
        // records too early and the probe reads mostly unwritten zeros.
        guard record.opKind == SmeltDispatchRecord.opDispatch else { continue }

        seenDispatches += 1
        if dispatchWouldExecute(
            record,
            mode: mode,
            position: position,
            seqLen: activeIds.count
        ) {
            lastExecutedRecord = record
            lastExecutedRecordIndex = recordIndex
            lastExecutedCur = cur
            lastExecutedAlt = alt
        }
        if seenDispatches == dispatchOrdinal {
            if marker != nil {
                selectedRecord = lastExecutedRecord
                selectedRecordIndex = lastExecutedRecordIndex
                selectedCur = lastExecutedCur
                selectedAlt = lastExecutedAlt
            } else {
                selectedRecord = record
                selectedRecordIndex = recordIndex
                selectedCur = cur
                selectedAlt = alt
            }
            break
        }
    }

    guard let record = selectedRecord else {
        throw UsageError(
            message: "Could not resolve dispatch record @ dispatch \(dispatchOrdinal)"
        )
    }

    let pipelineName = Int(record.pipeline) < manifest.pipelines.count
        ? manifest.pipelines[Int(record.pipeline)]
        : "pipeline_\(record.pipeline)"

    return ResolvedDispatchContext(
        manifest: manifest,
        marker: marker,
        runtime: runtime,
        mode: mode,
        ids: activeIds,
        startPos: startPos,
        position: position,
        dispatchOrdinal: dispatchOrdinal,
        recordIndex: selectedRecordIndex,
        pipelineName: pipelineName,
        record: record,
        cur: selectedCur,
        alt: selectedAlt
    )
}

private func resolveBinding(
    _ bindingIndex: Int,
    context: ResolvedDispatchContext,
    rowSelection: RowSelection,
    forcedDType: SmeltDType?,
    valueOffset: Int,
    valueCount: Int
) throws -> ResolvedBinding {
    guard bindingIndex >= 0, bindingIndex < Int(context.record.bufferCount) else {
        throw UsageError(
            message: "--buffer-index \(bindingIndex) is out of range for bufferCount \(context.record.bufferCount)"
        )
    }

    let buffer = getBuffer(context.record, index: bindingIndex)
    let resolvedSlot = resolveSlot(buffer.slot, cur: context.cur, alt: context.alt)
    let info = try slotInfo(resolvedSlot, manifest: context.manifest)
    let dumpDType = forcedDType ?? info.dtype
    let byteWidth = bytesPerElement(for: dumpDType)
    let offsetBytes = resolveOffsetBytes(
        buffer,
        mode: context.mode,
        position: context.position,
        seqLen: context.ids.count,
        startPos: context.startPos
    )
    let baseElementOffset = byteWidth > 0 ? offsetBytes / byteWidth : 0
    let maxBatch = compiledMaxBatch(manifest: context.manifest)
    let rowWidth = batchedRowStrideElements(
        for: resolvedSlot,
        manifest: context.manifest,
        compiledMaxBatch: maxBatch,
        marker: context.marker
    )

    var elementOffset = baseElementOffset
    if context.mode == .prefill,
       isBatchedSlot(slot: resolvedSlot, manifest: context.manifest, compiledMaxBatch: maxBatch),
       rowWidth > 0
    {
        let rowIndex: Int
        switch rowSelection {
        case .base:
            rowIndex = 0
        case .last:
            rowIndex = max(context.ids.count - 1, 0)
        case .index(let index):
            rowIndex = index
        }
        elementOffset += rowIndex * rowWidth
    }

    let count = max(valueCount, 0) > 0 ? valueCount : max(rowWidth - valueOffset, 0)
    let asFP32 = dumpDType == .fp32 || dumpDType == .int32
    let values = count > 0
        ? context.runtime.dumpSlot(
            resolvedSlot,
            elementOffset: elementOffset + valueOffset,
            count: count,
            asFP32: asFP32
        )
        : []

    return ResolvedBinding(
        bufferIndex: bindingIndex,
        metalBindingIndex: Int(buffer.bindingIndex),
        rawSlot: Int(buffer.slot),
        resolvedSlot: resolvedSlot,
        slotName: info.name,
        dtype: dumpDType,
        offsetKind: Int(buffer.offsetKind),
        offsetBytes: offsetBytes,
        baseElementOffset: baseElementOffset,
        elementOffset: elementOffset + valueOffset,
        rowWidth: rowWidth,
        values: values
    )
}

private func resolveTraceSample(
    packagePath: String,
    ids: [Int32],
    label: String,
    mode: TraceMode,
    occurrence: TraceOccurrenceSelection,
    contextLimit: Int? = nil,
    rowSelection: RowSelection = .last,
    prefillCount: Int = 0
) throws -> ResolvedTraceSample {
    let manifest = try loadManifest(packagePath: packagePath)
    let trace = try loadTraceMarkers(packagePath: packagePath)
    let markers = mode == .prefill ? trace.prefill : trace.decode
    let marker = try selectTraceMarker(label: label, markers: markers, occurrence: occurrence)
    let table = try loadDispatchTable(packagePath: packagePath, prefill: mode == .prefill)
    let runtime = try SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)

    let activeIds: [Int32]
    let startPos: Int
    let seqLen: Int
    let position: Int
    switch mode {
    case .prefill:
        let execution = try executePrefillThroughChunk(
            runtime: runtime,
            requestIds: ids,
            maxDispatches: marker.dispatchCount
        )
        activeIds = execution.tokenIds
        startPos = execution.startPos
        seqLen = activeIds.count
        position = startPos + max(activeIds.count - 1, 0)
    case .decode:
        guard prefillCount >= 0,
              ids.isEmpty ? prefillCount == 0 : prefillCount < ids.count else {
            throw UsageError(message: "--prefill-count must leave one decode token to inspect")
        }
        activeIds = ids
        startPos = 0
        seqLen = ids.count
        position = max(ids.count - 1, 0)
        runtime.resetWorkingBuffers()
        if !ids.isEmpty {
            if prefillCount > 0 {
                let prefix = Array(ids.prefix(prefillCount))
                for chunk in try prefillChunks(runtime: runtime, requestIds: prefix) {
                    _ = try runtime.prefillStep(
                        tokenIds: chunk.tokenIds,
                        startPos: Int32(chunk.startPos)
                    )
                }
            }
            for pos in prefillCount..<position {
                _ = try runtime.decodeStep(tokenId: ids[pos], position: Int32(pos))
            }
            _ = try runtime.debugDecodeStep(
                tokenId: ids[position],
                position: Int32(position),
                maxDispatches: marker.dispatchCount
            )
        }
    }

    var cur = 0
    var alt = 1
    var seenDispatches = 0
    var selectedRecord: SmeltDispatchRecord?
    var selectedRecordIndex = -1
    var selectedBindingIndex = -1
    var selectedOffsetBytes = 0
    var lastExecutedRecord: SmeltDispatchRecord?
    var lastExecutedRecordIndex = -1
    var lastExecutedCur = cur
    var lastExecutedAlt = alt
    for (recordIndex, record) in table.enumerated() {
        if record.opKind == SmeltDispatchRecord.opSwap {
            let tmp = cur
            cur = alt
            alt = tmp
            continue
        }

        guard record.opKind == SmeltDispatchRecord.opDispatch else { continue }

        seenDispatches += 1
        if dispatchWouldExecute(
            record,
            mode: mode,
            position: position,
            seqLen: seqLen
        ) {
            lastExecutedRecord = record
            lastExecutedRecordIndex = recordIndex
            lastExecutedCur = cur
            lastExecutedAlt = alt
        }
        if seenDispatches == marker.dispatchCount {
            let resolvedRecord = lastExecutedRecord ?? record
            let resolvedRecordIndex = lastExecutedRecord != nil ? lastExecutedRecordIndex : recordIndex
            let resolvedCur = lastExecutedRecord != nil ? lastExecutedCur : cur
            let resolvedAlt = lastExecutedRecord != nil ? lastExecutedAlt : alt
            var bestBindingIndex = -1
            var bestOffsetBytes = 0
            for binding in 0..<Int(resolvedRecord.bufferCount) {
                let buf = getBuffer(resolvedRecord, index: binding)
                let resolvedSlot = resolveSlot(buf.slot, cur: resolvedCur, alt: resolvedAlt)
                guard resolvedSlot == marker.bufferSlot else { continue }
                let offsetBytes = resolveOffsetBytes(
                    buf,
                    mode: mode,
                    position: position,
                    seqLen: seqLen,
                    startPos: startPos
                )
                if Int(buf.bindingIndex) >= bestBindingIndex {
                    bestBindingIndex = Int(buf.bindingIndex)
                    bestOffsetBytes = offsetBytes
                }
            }

            selectedRecord = resolvedRecord
            selectedRecordIndex = resolvedRecordIndex
            selectedBindingIndex = bestBindingIndex
            selectedOffsetBytes = bestOffsetBytes
            break
        }
    }

    guard let record = selectedRecord else {
        throw UsageError(
            message: "Could not resolve dispatch record for '\(label)' @ dispatch \(marker.dispatchCount)"
        )
    }

    guard let slotInfo = manifest.buffers.slots.first(where: { $0.index == marker.bufferSlot }) else {
        throw UsageError(message: "Buffer slot \(marker.bufferSlot) is missing from manifest")
    }

    let asFP32 = slotInfo.dtype == .fp32
    let bytesPerElement = asFP32 || slotInfo.dtype == .int32 ? 4 : 2
    var elementOffset = selectedOffsetBytes / max(bytesPerElement, 1)

    let markerMatchCount = markers.filter { $0.label == label }.count
    let maxBatch = compiledMaxBatch(manifest: manifest)
    let rowWidth = batchedRowStrideElements(
        for: marker.bufferSlot,
        manifest: manifest,
        compiledMaxBatch: maxBatch,
        marker: marker
    )
    if mode == .prefill,
       markerMatchCount == 1,
       isBatchedSlot(slot: marker.bufferSlot, manifest: manifest, compiledMaxBatch: maxBatch),
       selectedOffsetBytes == 0 {
        // Batched kernels bind the whole output at byte zero, so select the
        // requested row here. Unrolled per-position kernels bind each row at
        // its literal byte offset already; adding the row stride again would
        // read the next unwritten row.
        let rowIndex: Int
        switch rowSelection {
        case .base:
            rowIndex = 0
        case .last:
            rowIndex = max(seqLen - 1, 0)
        case .index(let index):
            rowIndex = index
        }
        elementOffset += rowIndex * rowWidth
    }

    let values = runtime.dumpSlot(
        marker.bufferSlot,
        elementOffset: elementOffset,
        count: rowWidth,
        asFP32: asFP32
    )
    let pipelineName = Int(record.pipeline) < manifest.pipelines.count
        ? manifest.pipelines[Int(record.pipeline)]
        : "pipeline_\(record.pipeline)"

    return ResolvedTraceSample(
        marker: marker,
        dispatchRecordIndex: selectedRecordIndex,
        dispatchOrdinal: marker.dispatchCount,
        pipelineName: pipelineName,
        slot: marker.bufferSlot,
        elementOffset: elementOffset,
        bindingIndex: selectedBindingIndex,
        asFP32: asFP32,
        rowWidth: rowWidth,
        values: values
    )
}

private func compareLabel(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let label = try parseArg("--label", from: args)
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let occurrence = try parseTraceOccurrenceSelection(
        try parseArg("--occurrence", from: args, default: "last")
    )

    let prefill = try resolveTraceSample(
        packagePath: package,
        ids: ids,
        label: label,
        mode: .prefill,
        occurrence: occurrence,
        contextLimit: contextLimit
    )
    let decode = try resolveTraceSample(
        packagePath: package,
        ids: ids,
        label: label,
        mode: .decode,
        occurrence: .last,
        contextLimit: contextLimit
    )

    let count = min(prefill.rowWidth, decode.rowWidth, prefill.values.count, decode.values.count)
    var maxDiff: Float = 0
    var maxIdx = 0
    var above001 = 0
    for idx in 0..<count {
        let diff = diffMagnitude(prefill.values[idx], decode.values[idx])
        if diff > maxDiff {
            maxDiff = diff
            maxIdx = idx
        }
        if diff > 0.01 {
            above001 += 1
        }
    }

    print(
        "label=\(label)"
            + " slot=\(prefill.slot)"
            + " prefillDispatch=\(prefill.dispatchOrdinal)"
            + " decodeDispatch=\(decode.dispatchOrdinal)"
            + " prefillRecord=\(prefill.dispatchRecordIndex)"
            + " decodeRecord=\(decode.dispatchRecordIndex)"
            + " prefillBinding=\(prefill.bindingIndex)"
            + " decodeBinding=\(decode.bindingIndex)"
            + " prefillOffset=\(prefill.elementOffset)"
            + " decodeOffset=\(decode.elementOffset)"
            + " rowWidth=\(count)"
            + " maxDiff=\(formatExact(maxDiff))"
            + " idx=\(maxIdx)"
            + " prefill=\(formatExact(prefill.values[maxIdx]))"
            + " decode=\(formatExact(decode.values[maxIdx]))"
            + " above0.01=\(above001)"
    )
    print("prefillPipeline=\(prefill.pipelineName)")
    print("decodePipeline=\(decode.pipelineName)")
    print(
        "prefillHead=\(prefill.values.prefix(12).map(formatExact))"
    )
    print(
        "decodeHead=\(decode.values.prefix(12).map(formatExact))"
    )
}

private func summarizeDiff(label: String, reference: [Float], actual: [Float]) {
    let count = min(reference.count, actual.count)
    guard count > 0 else {
        print("\(label): no values")
        return
    }
    var maxDiff: Float = 0
    var maxIdx = 0
    var above001 = 0
    for idx in 0..<count {
        let diff = diffMagnitude(reference[idx], actual[idx])
        if diff > maxDiff {
            maxDiff = diff
            maxIdx = idx
        }
        if diff > 0.01 {
            above001 += 1
        }
    }
    print(
        "\(label):"
            + " maxDiff=\(formatExact(maxDiff))"
            + " idx=\(maxIdx)"
            + " reference=\(formatExact(reference[maxIdx]))"
            + " actual=\(formatExact(actual[maxIdx]))"
            + " above0.01=\(above001)"
    )
}

private func rawDispatchOrdinal(
    forRecordIndex recordIndex: Int,
    table: [SmeltDispatchRecord]
) throws -> Int {
    guard recordIndex >= 0 else {
        throw UsageError(message: "Dispatch record index must be non-negative")
    }
    var ordinal = 0
    for (idx, record) in table.enumerated() where record.opKind == SmeltDispatchRecord.opDispatch {
        ordinal += 1
        if idx == recordIndex {
            return ordinal
        }
    }
    throw UsageError(message: "No raw dispatch ordinal found for record \(recordIndex)")
}

private func stagedRMSNormAddReference(
    input: [Float],
    weight: [Float],
    residual: [Float],
    eps: Float
) -> (norm: [Float], output: [Float]) {
    precondition(input.count == weight.count && input.count == residual.count)
    let dim = Float(input.count)
    var sumSq: Float = 0
    for value in input {
        sumSq += value * value
    }
    let rs = 1.0 / sqrt(sumSq / dim + eps)

    var norm: [Float] = []
    var output: [Float] = []
    norm.reserveCapacity(input.count)
    output.reserveCapacity(input.count)
    for idx in input.indices {
        let scale = rs * (1 + weight[idx])
        let roundedNorm = Float(Float16(input[idx] * scale))
        norm.append(roundedNorm)
        output.append(Float(Float16(roundedNorm + residual[idx])))
    }
    return (norm, output)
}

private func analyzeRMSNormAdd(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let prefillNormDispatch = Int(try parseArg("--prefill-norm-dispatch", from: args)) ?? 0
    let prefillAddDispatch = Int(try parseArg("--prefill-add-dispatch", from: args)) ?? 0
    let decodeDispatch = Int(try parseArg("--decode-dispatch", from: args)) ?? 0
    let rowSelection = try parseRowSelection(
        try parseArg("--prefill-row-index", from: args, default: "last")
    )
    let eps = Float(try parseArg("--eps", from: args, default: "1e-6")) ?? 1e-6
    let dim = Int(try parseArg("--dim", from: args, default: "1536")) ?? 1536

    let prefillNormContext = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: .prefill,
        dispatchOrdinal: prefillNormDispatch
    )
    let prefillAddContext = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: .prefill,
        dispatchOrdinal: prefillAddDispatch
    )
    let decodeContext = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: .decode,
        dispatchOrdinal: decodeDispatch
    )

    let prefillInput = try resolveBinding(
        0,
        context: prefillNormContext,
        rowSelection: rowSelection,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values
    let prefillWeight = try resolveBinding(
        1,
        context: prefillNormContext,
        rowSelection: .base,
        forcedDType: .fp16,
        valueOffset: 0,
        valueCount: dim
    ).values
    let prefillNorm = try resolveBinding(
        2,
        context: prefillNormContext,
        rowSelection: rowSelection,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values
    let prefillResidual = try resolveBinding(
        0,
        context: prefillAddContext,
        rowSelection: rowSelection,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values
    let prefillOutput = try resolveBinding(
        2,
        context: prefillAddContext,
        rowSelection: rowSelection,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values

    let decodeInput = try resolveBinding(
        0,
        context: decodeContext,
        rowSelection: .base,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values
    let decodeWeight = try resolveBinding(
        1,
        context: decodeContext,
        rowSelection: .base,
        forcedDType: .fp16,
        valueOffset: 0,
        valueCount: dim
    ).values
    let decodeResidual = try resolveBinding(
        2,
        context: decodeContext,
        rowSelection: .base,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values
    let decodeOutput = try resolveBinding(
        3,
        context: decodeContext,
        rowSelection: .base,
        forcedDType: nil,
        valueOffset: 0,
        valueCount: dim
    ).values

    let prefillReference = stagedRMSNormAddReference(
        input: prefillInput,
        weight: prefillWeight,
        residual: prefillResidual,
        eps: eps
    )
    let decodeReference = stagedRMSNormAddReference(
        input: decodeInput,
        weight: decodeWeight,
        residual: decodeResidual,
        eps: eps
    )

    print("prefillNormPipeline=\(prefillNormContext.pipelineName)")
    print("prefillAddPipeline=\(prefillAddContext.pipelineName)")
    print("decodePipeline=\(decodeContext.pipelineName)")
    summarizeDiff(label: "prefill input vs decode input", reference: prefillInput, actual: decodeInput)
    summarizeDiff(label: "prefill weight vs decode weight", reference: prefillWeight, actual: decodeWeight)
    summarizeDiff(label: "prefill residual vs decode residual", reference: prefillResidual, actual: decodeResidual)
    summarizeDiff(label: "prefill norm vs staged(prefill inputs)", reference: prefillReference.norm, actual: prefillNorm)
    summarizeDiff(label: "prefill output vs staged(prefill inputs)", reference: prefillReference.output, actual: prefillOutput)
    summarizeDiff(label: "decode output vs staged(decode inputs)", reference: decodeReference.output, actual: decodeOutput)
    summarizeDiff(label: "prefill output vs decode output", reference: prefillOutput, actual: decodeOutput)
}

private func inspectLabel(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let label = try parseArg("--label", from: args)
    let occurrence = try parseTraceOccurrenceSelection(
        try parseArg("--occurrence", from: args, default: "last")
    )
    let modeRaw = try parseArg("--mode", from: args, default: "both")
    let rowSelection = try parseRowSelection(
        try parseArg("--row-index", from: args, default: "base")
    )
    let forcedDType = try parseOptionalDType(
        try parseArg("--force-dtype", from: args, default: "")
    )
    let valueOffset = Int(try parseArg("--value-offset", from: args, default: "0")) ?? 0
    let valueCount = Int(try parseArg("--values", from: args, default: "8")) ?? 8
    let selectedBufferIndex = Int(
        try parseArg("--buffer-index", from: args, default: "-1")
    ) ?? -1
    guard selectedBufferIndex >= -1 else {
        throw UsageError(message: "--buffer-index must be nonnegative")
    }
    let prefillCount = Int(
        try parseArg("--prefill-count", from: args, default: "0")
    ) ?? 0

    let modes: [TraceMode]
    switch modeRaw {
    case "prefill":
        modes = [.prefill]
    case "decode":
        modes = [.decode]
    case "both":
        modes = [.prefill, .decode]
    default:
        throw UsageError(message: "--mode must be prefill, decode, or both")
    }

    for mode in modes {
        if selectedBufferIndex < 0 {
            let sample = try resolveTraceSample(
                packagePath: package,
                ids: ids,
                label: label,
                mode: mode,
                occurrence: occurrence,
                contextLimit: contextLimit,
                rowSelection: rowSelection,
                prefillCount: mode == .decode ? prefillCount : 0
            )
            let modeName = mode == .prefill ? "prefill" : "decode"
            let slotInfo = try slotInfo(
                sample.slot,
                manifest: loadManifest(packagePath: package)
            )
            let end = min(sample.values.count, valueOffset + max(valueCount, 0))
            let head = valueOffset < end
                ? Array(sample.values[valueOffset..<end])
                : []
            let rowHashInfo = sha256Hex(values: sample.values, dtype: slotInfo.dtype)
                .map { " rowSHA256=\($0)" } ?? ""
            print(
                "\(modeName) label=\(label)"
                    + " dispatch=\(sample.dispatchOrdinal)"
                    + " record=\(sample.dispatchRecordIndex)"
                    + " pipeline=\(sample.pipelineName)"
            )
            print(
                "  markerSlot=\(sample.slot)"
                    + " slotName=\(slotInfo.name)"
                    + " dtype=\(slotInfo.dtype.rawValue)"
                    + " elementOffset=\(sample.elementOffset + valueOffset)"
                    + " rowWidth=\(sample.rowWidth)"
                    + rowHashInfo
                    + " head=\(head.map(formatExact))"
            )
            continue
        }
        let context = try resolveDispatchContext(
            packagePath: package,
            ids: ids,
            mode: mode,
            label: label,
            occurrence: mode == .prefill ? occurrence : .last,
            contextLimit: contextLimit,
            prefillCount: mode == .decode ? prefillCount : 0
        )
        let modeName = mode == .prefill ? "prefill" : "decode"
        print(
            "\(modeName) label=\(label)"
                + " dispatch=\(context.dispatchOrdinal)"
                + " record=\(context.recordIndex)"
                + " cur=\(context.cur)"
                + " alt=\(context.alt)"
                + " pipeline=\(context.pipelineName)"
                + " bufferCount=\(context.record.bufferCount)"
        )
        let bufferIndices = [selectedBufferIndex]
        guard bufferIndices.allSatisfy({ $0 < Int(context.record.bufferCount) }) else {
            throw UsageError(message: "--buffer-index is outside the selected dispatch")
        }
        for bufferIndex in bufferIndices {
            let binding = try resolveBinding(
                bufferIndex,
                context: context,
                rowSelection: rowSelection,
                forcedDType: forcedDType,
                valueOffset: valueOffset,
                valueCount: valueCount
            )
            print(
                "  bufferIndex=\(binding.bufferIndex)"
                    + " metalBinding=\(binding.metalBindingIndex)"
                    + " rawSlot=\(binding.rawSlot)"
                    + " resolvedSlot=\(binding.resolvedSlot)"
                    + " slotName=\(binding.slotName)"
                    + " dtype=\(binding.dtype.rawValue)"
                    + " offsetKind=\(binding.offsetKind)"
                    + " offsetBytes=\(binding.offsetBytes)"
                    + " baseElementOffset=\(binding.baseElementOffset)"
                    + " elementOffset=\(binding.elementOffset)"
                    + " rowWidth=\(binding.rowWidth)"
                    + " head=\(binding.values.prefix(valueCount).map(formatExact))"
            )
        }
    }
}

private func inspectDispatch(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let dispatchOrdinal = Int(try parseArg("--dispatch", from: args)) ?? 0
    let modeRaw = try parseArg("--mode", from: args)
    let rowSelection = try parseRowSelection(
        try parseArg("--row-index", from: args, default: "base")
    )
    let forcedDType = try parseOptionalDType(
        try parseArg("--force-dtype", from: args, default: "")
    )
    let valueOffset = Int(try parseArg("--value-offset", from: args, default: "0")) ?? 0
    let valueCount = Int(try parseArg("--values", from: args, default: "8")) ?? 8
    let selectedBufferIndex = Int(
        try parseArg("--buffer-index", from: args, default: "-1")
    ) ?? -1
    guard selectedBufferIndex >= -1 else {
        throw UsageError(message: "--buffer-index must be nonnegative")
    }
    let prefillCount = Int(
        try parseArg("--prefill-count", from: args, default: "0")
    ) ?? 0

    let mode: TraceMode
    switch modeRaw {
    case "prefill":
        mode = .prefill
    case "decode":
        mode = .decode
    default:
        throw UsageError(message: "--mode must be prefill or decode")
    }

    let context = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: mode,
        dispatchOrdinal: dispatchOrdinal,
        contextLimit: contextLimit,
        prefillCount: mode == .decode ? prefillCount : 0
    )
    let modeName = mode == .prefill ? "prefill" : "decode"
    print(
        "\(modeName) dispatch=\(context.dispatchOrdinal)"
            + " record=\(context.recordIndex)"
            + " cur=\(context.cur)"
            + " alt=\(context.alt)"
            + " pipeline=\(context.pipelineName)"
            + " bufferCount=\(context.record.bufferCount)"
            + " tg=(\(context.record.tgW),\(context.record.tgH),\(context.record.tgD))"
            + " grid=(\(context.record.gridW),\(context.record.gridH),\(context.record.gridD))"
            + " gridKind=(\(context.record.gridWKind),\(context.record.gridHKind),\(context.record.gridDKind))"
    )
    let bufferIndices = selectedBufferIndex >= 0
        ? [selectedBufferIndex]
        : Array(0..<Int(context.record.bufferCount))
    guard bufferIndices.allSatisfy({ $0 < Int(context.record.bufferCount) }) else {
        throw UsageError(message: "--buffer-index is outside the selected dispatch")
    }
    for bufferIndex in bufferIndices {
        let binding = try resolveBinding(
            bufferIndex,
            context: context,
            rowSelection: rowSelection,
            forcedDType: forcedDType,
            valueOffset: valueOffset,
            valueCount: valueCount
        )
        print(
            "  bufferIndex=\(binding.bufferIndex)"
                + " metalBinding=\(binding.metalBindingIndex)"
                + " rawSlot=\(binding.rawSlot)"
                + " resolvedSlot=\(binding.resolvedSlot)"
                + " slotName=\(binding.slotName)"
                + " dtype=\(binding.dtype.rawValue)"
                + " offsetKind=\(binding.offsetKind)"
                + " offsetBytes=\(binding.offsetBytes)"
                + " baseElementOffset=\(binding.baseElementOffset)"
                + " elementOffset=\(binding.elementOffset)"
                + " rowWidth=\(binding.rowWidth)"
                + " head=\(binding.values.prefix(valueCount).map(formatExact))"
        )
    }
}

private func compareLabelBinding(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let label = try parseArg("--label", from: args)
    let bufferIndex = Int(try parseArg("--buffer-index", from: args)) ?? -1
    let occurrence = try parseTraceOccurrenceSelection(
        try parseArg("--occurrence", from: args, default: "last")
    )
    let prefillRowSelection = try parseRowSelection(
        try parseArg("--prefill-row-index", from: args, default: "last")
    )
    let decodeRowSelection = try parseRowSelection(
        try parseArg("--decode-row-index", from: args, default: "base")
    )
    let forcedDType = try parseOptionalDType(
        try parseArg("--force-dtype", from: args, default: "")
    )
    let valueOffset = Int(try parseArg("--value-offset", from: args, default: "0")) ?? 0
    let count = Int(try parseArg("--count", from: args, default: "0")) ?? 0

    let prefillContext = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: .prefill,
        label: label,
        occurrence: occurrence,
        contextLimit: contextLimit
    )
    let decodeContext = try resolveDispatchContext(
        packagePath: package,
        ids: ids,
        mode: .decode,
        label: label,
        occurrence: .last,
        contextLimit: contextLimit
    )
    let prefillBinding = try resolveBinding(
        bufferIndex,
        context: prefillContext,
        rowSelection: prefillRowSelection,
        forcedDType: forcedDType,
        valueOffset: valueOffset,
        valueCount: count
    )
    let decodeBinding = try resolveBinding(
        bufferIndex,
        context: decodeContext,
        rowSelection: decodeRowSelection,
        forcedDType: forcedDType,
        valueOffset: valueOffset,
        valueCount: count > 0 ? count : prefillBinding.rowWidth
    )

    let compareCount = min(prefillBinding.values.count, decodeBinding.values.count)
    guard compareCount > 0 else {
        throw UsageError(message: "Resolved binding comparison count is zero")
    }
    var maxDiff: Float = 0
    var maxIdx = 0
    var above001 = 0
    for idx in 0..<compareCount {
        let diff = diffMagnitude(prefillBinding.values[idx], decodeBinding.values[idx])
        if diff > maxDiff {
            maxDiff = diff
            maxIdx = idx
        }
        if diff > 0.01 {
            above001 += 1
        }
    }

    print(
        "label=\(label)"
            + " bufferIndex=\(bufferIndex)"
            + " prefillDispatch=\(prefillContext.dispatchOrdinal)"
            + " decodeDispatch=\(decodeContext.dispatchOrdinal)"
            + " prefillRecord=\(prefillContext.recordIndex)"
            + " decodeRecord=\(decodeContext.recordIndex)"
            + " prefillSlot=\(prefillBinding.resolvedSlot)"
            + " decodeSlot=\(decodeBinding.resolvedSlot)"
            + " prefillOffset=\(prefillBinding.elementOffset)"
            + " decodeOffset=\(decodeBinding.elementOffset)"
            + " compareCount=\(compareCount)"
            + " maxDiff=\(formatExact(maxDiff))"
            + " idx=\(maxIdx)"
            + " prefill=\(formatExact(prefillBinding.values[maxIdx]))"
            + " decode=\(formatExact(decodeBinding.values[maxIdx]))"
            + " above0.01=\(above001)"
    )
    print("prefillPipeline=\(prefillContext.pipelineName)")
    print("decodePipeline=\(decodeContext.pipelineName)")
    print(
        "prefillHead=\(prefillBinding.values.prefix(12).map(formatExact))"
    )
    print(
        "decodeHead=\(decodeBinding.values.prefix(12).map(formatExact))"
    )
}

private func compareRows(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let sharedSlot = Int(try parseArg("--slot", from: args, default: "-1")) ?? -1
    let prefillSlot = Int(
        try parseArg("--prefill-slot", from: args, default: "\(sharedSlot)")
    ) ?? -1
    let decodeSlot = Int(
        try parseArg("--decode-slot", from: args, default: "\(sharedSlot)")
    ) ?? -1
    let rowWidth = Int(try parseArg("--row-width", from: args)) ?? 0
    let prefillDispatch = Int(try parseArg("--prefill-dispatch", from: args)) ?? 0
    let decodeDispatch = Int(try parseArg("--decode-dispatch", from: args)) ?? 0
    let decodeRowSelection = try parseRowSelection(
        try parseArg("--decode-row-index", from: args, default: "base")
    )

    guard prefillSlot >= 0 else {
        throw UsageError(message: "--prefill-slot must be non-negative")
    }
    guard decodeSlot >= 0 else {
        throw UsageError(message: "--decode-slot must be non-negative")
    }
    guard rowWidth > 0 else {
        throw UsageError(message: "--row-width must be positive")
    }
    guard prefillDispatch > 0 else {
        throw UsageError(message: "--prefill-dispatch must be positive")
    }
    guard decodeDispatch > 0 else {
        throw UsageError(message: "--decode-dispatch must be positive")
    }

    let prefill = try SmeltRuntime(packagePath: package)
    let decode = try SmeltRuntime(packagePath: package)

    let chunks = try prefillChunks(runtime: prefill, requestIds: ids)
    for (chunkIndex, _) in chunks.enumerated() {
        let execution = try executePrefillThroughChunk(
            runtime: prefill,
            requestIds: ids,
            targetChunkIndex: chunkIndex,
            maxDispatches: prefillDispatch
        )
        let prefillValues = prefill.dumpSlot(
            prefillSlot,
            count: execution.tokenIds.count * rowWidth
        )

        for localPos in 0..<execution.tokenIds.count {
            let pos = execution.startPos + localPos
            decode.resetWorkingBuffers()
            if pos > 0 {
                for i in 0..<pos {
                    _ = try decode.decodeStep(
                        tokenId: ids[i],
                        position: Int32(i)
                    )
                }
            }
            _ = try decode.debugDecodeStep(
                tokenId: ids[pos],
                position: Int32(pos),
                maxDispatches: decodeDispatch
            )
            let decodeRowIndex: Int
            switch decodeRowSelection {
            case .base:
                decodeRowIndex = 0
            case .last:
                decodeRowIndex = pos
            case .index(let index):
                decodeRowIndex = index
            }
            let decodeValues = decode.dumpSlot(
                decodeSlot,
                elementOffset: decodeRowIndex * rowWidth,
                count: rowWidth
            )

            var maxDiff: Float = 0
            var maxIdx = 0
            var above001 = 0
            let base = localPos * rowWidth
            for idx in 0..<rowWidth {
                let diff = diffMagnitude(prefillValues[base + idx], decodeValues[idx])
                if diff > maxDiff {
                    maxDiff = diff
                    maxIdx = idx
                }
                if diff > 0.01 {
                    above001 += 1
                }
            }

    print(
        "pos=\(pos)"
            + " decodeRow=\(decodeRowIndex)"
            + " maxDiff=\(formatExact(maxDiff))"
            + " idx=\(maxIdx)"
            + " prefill=\(formatExact(prefillValues[base + maxIdx]))"
            + " decode=\(formatExact(decodeValues[maxIdx]))"
            + " above0.01=\(above001)"
    )
            fflush(stdout)
        }
    }
}

private func compareFinalSlot(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let sharedSlot = Int(try parseArg("--slot", from: args, default: "-1")) ?? -1
    let prefillSlot = Int(
        try parseArg("--prefill-slot", from: args, default: "\(sharedSlot)")
    ) ?? -1
    let decodeSlot = Int(
        try parseArg("--decode-slot", from: args, default: "\(sharedSlot)")
    ) ?? -1
    let prefillRowSelection = try parseRowSelection(
        try parseArg("--prefill-row-index", from: args, default: "last")
    )
    let count = Int(try parseArg("--count", from: args)) ?? 0

    guard prefillSlot >= 0 else {
        throw UsageError(message: "--prefill-slot must be non-negative")
    }
    guard decodeSlot >= 0 else {
        throw UsageError(message: "--decode-slot must be non-negative")
    }
    guard count > 0 else {
        throw UsageError(message: "--count must be positive")
    }

    let manifest = try loadManifest(packagePath: package)
    let prefill = try SmeltRuntime(packagePath: package, contextLimit: contextLimit)
    let execution = try executePrefillThroughChunk(
        runtime: prefill,
        requestIds: ids
    )
    let prefillRowIndex: Int
    switch prefillRowSelection {
    case .base:
        prefillRowIndex = 0
    case .last:
        prefillRowIndex = max(execution.tokenIds.count - 1, 0)
    case .index(let index):
        guard index < execution.tokenIds.count else {
            throw UsageError(
                message: "--prefill-row-index \(index) is out of range for "
                    + "\(execution.tokenIds.count) rows in the final prefill chunk"
            )
        }
        prefillRowIndex = index
    }
    let prefillIsBatched = isBatchedSlot(
        slot: prefillSlot,
        manifest: manifest,
        compiledMaxBatch: compiledMaxBatch(manifest: manifest)
    )
    let prefillRowStride = batchedRowStrideElements(
        for: prefillSlot,
        manifest: manifest,
        compiledMaxBatch: compiledMaxBatch(manifest: manifest)
    )
    let prefillElementOffset = prefillIsBatched ? prefillRowIndex * prefillRowStride : 0
    let prefillValues = prefill.dumpSlot(
        prefillSlot,
        elementOffset: prefillElementOffset,
        count: count
    )

    let decode = try SmeltRuntime(packagePath: package, contextLimit: contextLimit)
    decode.resetWorkingBuffers()
    for (position, tokenId) in ids.enumerated() {
        _ = try decode.decodeStep(tokenId: tokenId, position: Int32(position))
    }
    let decodeValues = decode.dumpSlot(decodeSlot, count: count)

    var maxDiff: Float = 0
    var maxIdx = 0
    var above001 = 0
    for idx in 0..<count {
        let diff = diffMagnitude(prefillValues[idx], decodeValues[idx])
        if diff > maxDiff {
            maxDiff = diff
            maxIdx = idx
        }
        if diff > 0.01 {
            above001 += 1
        }
    }

    print(
        "prefillRow=\(prefillIsBatched ? prefillRowIndex : 0)"
            + " maxDiff=\(formatExact(maxDiff))"
            + " idx=\(maxIdx)"
            + " prefill=\(formatExact(prefillValues[maxIdx]))"
            + " decode=\(formatExact(decodeValues[maxIdx]))"
            + " above0.01=\(above001)"
    )
}

private func generateFromTokenIDs(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contextLimit = try parseOptionalPositiveIntArg("--context-limit", from: args)
    let maxTokens = Int(try parseArg("--max-tokens", from: args, default: "4")) ?? 0
    guard maxTokens > 0 else {
        throw UsageError(message: "--max-tokens must be positive")
    }
    let prefixMode = try parseArg("--prefix-mode", from: args, default: "prefill")
    let topK = Int(try parseArg("--top-k", from: args, default: "0")) ?? -1
    guard topK >= 0 else {
        throw UsageError(message: "--top-k must be non-negative")
    }

    let runtime = try SmeltRuntime(packagePath: package, contextLimit: contextLimit)
    runtime.resetWorkingBuffers()
    var current: Int32
    switch prefixMode {
    case "prefill":
        let chunks = try prefillChunks(runtime: runtime, requestIds: ids)
        guard !chunks.isEmpty else {
            throw UsageError(message: "Expected at least one token id")
        }
        current = 0
        for chunk in chunks {
            current = try runtime.prefillStep(
                tokenIds: chunk.tokenIds,
                startPos: Int32(chunk.startPos)
            )
        }
    case "decode":
        current = 0
        for (position, tokenID) in ids.enumerated() {
            current = try runtime.decodeStep(tokenId: tokenID, position: Int32(position))
        }
    default:
        throw UsageError(message: "--prefix-mode must be 'prefill' or 'decode'")
    }

    var generated: [Int32] = []
    generated.reserveCapacity(maxTokens)
    for step in 0..<maxTokens {
        if topK > 0 {
            let ranked = runtime.topKLogits(k: topK)
                .map { "\($0.0):\(formatExact($0.1))" }
                .joined(separator: ",")
            print("step=\(step) selected=\(current) top=[\(ranked)]")
        }
        generated.append(current)
        if step + 1 < maxTokens {
            current = try runtime.decodeStep(
                tokenId: current,
                position: Int32(ids.count + step)
            )
        }
    }

    print(
        "prefixMode=\(prefixMode)"
            + " inputCount=\(ids.count)"
            + " generated=[\(generated.map(String.init).joined(separator: ","))]"
    )
}

private func listDispatches(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let modeRaw = try parseArg("--mode", from: args, default: "decode")
    let contains = try parseArg("--contains", from: args, default: "")
    let prefill: Bool
    switch modeRaw {
    case "decode":
        prefill = false
    case "prefill":
        prefill = true
    default:
        throw UsageError(message: "--mode must be decode or prefill")
    }

    let manifest = try loadManifest(packagePath: package)
    let table = try loadDispatchTable(packagePath: package, prefill: prefill)
    var ordinal = 0
    for (recordIndex, record) in table.enumerated() where record.opKind == SmeltDispatchRecord.opDispatch {
        ordinal += 1
        let name = Int(record.pipeline) < manifest.pipelines.count
            ? manifest.pipelines[Int(record.pipeline)]
            : "pipeline_\(record.pipeline)"
        if !contains.isEmpty && !name.contains(contains) {
            continue
        }
        print(
            "dispatch=\(ordinal)"
                + " record=\(recordIndex)"
                + " pipeline=\(name)"
        )
    }
}

private func primeDecodeStateForDispatch(
    runtime: SmeltRuntime,
    ids: [Int32],
    dispatchOrdinal: Int
) throws {
    try runtime.prepareForRequest(
        batchCapacity: max(ids.count, 1),
        contextCapacity: max(ids.count, 1)
    )
    runtime.resetWorkingBuffers()

    let position = ids.count - 1
    guard position >= 0 else { return }

    if position > 0 {
        for idx in 0..<position {
            _ = try runtime.decodeStep(tokenId: ids[idx], position: Int32(idx))
        }
    }

    if dispatchOrdinal > 1 {
        _ = try runtime.debugDecodeStep(
            tokenId: ids[position],
            position: Int32(position),
            maxDispatches: dispatchOrdinal - 1
        )
    }
}

private func benchDispatch(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let dispatchOrdinal = Int(try parseArg("--dispatch", from: args)) ?? 0
    let warmup = Int(try parseArg("--warmup", from: args, default: "3")) ?? 3
    let iterations = Int(try parseArg("--iterations", from: args, default: "20")) ?? 20

    guard !ids.isEmpty else {
        throw UsageError(message: "--ids must not be empty")
    }
    guard dispatchOrdinal > 0 else {
        throw UsageError(message: "--dispatch must be positive")
    }

    let runtime = try SmeltRuntime(packagePath: package)
    try primeDecodeStateForDispatch(runtime: runtime, ids: ids, dispatchOrdinal: dispatchOrdinal)
    guard let result = try runtime.benchmarkDecodeDispatch(
        tokenId: ids[ids.count - 1],
        position: Int32(ids.count - 1),
        dispatchOrdinal: dispatchOrdinal,
        warmup: warmup,
        iterations: iterations
    ) else {
        throw UsageError(message: "Dispatch \(dispatchOrdinal) did not execute at this position")
    }

    print(
        "dispatch=\(result.dispatchOrdinal)"
            + " record=\(result.recordIndex)"
            + " pipeline=\(result.name)"
            + " medianUs=\(String(format: "%.2f", result.medianGpuUs))"
            + " p95Us=\(String(format: "%.2f", result.p95GpuUs))"
            + " avgUs=\(String(format: "%.2f", result.avgGpuUs))"
            + " minUs=\(String(format: "%.2f", result.minGpuUs))"
            + " maxUs=\(String(format: "%.2f", result.maxGpuUs))"
            + " samples=\(result.samplesGpuUs.count)"
    )
}

private func benchPipeline(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let contains = try parseArg("--contains", from: args)
    let warmup = Int(try parseArg("--warmup", from: args, default: "3")) ?? 3
    let iterations = Int(try parseArg("--iterations", from: args, default: "20")) ?? 20

    guard !contains.isEmpty else {
        throw UsageError(message: "--contains must not be empty")
    }
    guard !ids.isEmpty else {
        throw UsageError(message: "--ids must not be empty")
    }

    let manifest = try loadManifest(packagePath: package)
    let table = try loadDispatchTable(packagePath: package, prefill: false)
    var matches: [(dispatchOrdinal: Int, recordIndex: Int, name: String)] = []
    var dispatchOrdinal = 0
    for (recordIndex, record) in table.enumerated() where record.opKind == SmeltDispatchRecord.opDispatch {
        dispatchOrdinal += 1
        let name = Int(record.pipeline) < manifest.pipelines.count
            ? manifest.pipelines[Int(record.pipeline)]
            : "pipeline_\(record.pipeline)"
        if name.contains(contains) {
            matches.append((dispatchOrdinal, recordIndex, name))
        }
    }

    guard !matches.isEmpty else {
        throw UsageError(message: "No decode dispatch matched '\(contains)'")
    }

    let runtime = try SmeltRuntime(packagePath: package)
    var kept: [SmeltRuntime.DispatchBenchmarkProfile] = []
    for match in matches {
        try primeDecodeStateForDispatch(
            runtime: runtime,
            ids: ids,
            dispatchOrdinal: match.dispatchOrdinal
        )
        guard let result = try runtime.benchmarkDecodeDispatch(
            tokenId: ids[ids.count - 1],
            position: Int32(ids.count - 1),
            dispatchOrdinal: match.dispatchOrdinal,
            warmup: warmup,
            iterations: iterations
        ) else {
            print(
                "dispatch=\(match.dispatchOrdinal)"
                    + " record=\(match.recordIndex)"
                    + " pipeline=\(match.name)"
                    + " skipped=true"
            )
            continue
        }
        kept.append(result)
        print(
            "dispatch=\(result.dispatchOrdinal)"
                + " record=\(result.recordIndex)"
                + " pipeline=\(result.name)"
                + " medianUs=\(String(format: "%.2f", result.medianGpuUs))"
                + " p95Us=\(String(format: "%.2f", result.p95GpuUs))"
                + " avgUs=\(String(format: "%.2f", result.avgGpuUs))"
                + " minUs=\(String(format: "%.2f", result.minGpuUs))"
                + " maxUs=\(String(format: "%.2f", result.maxGpuUs))"
        )
    }

    guard !kept.isEmpty else {
        throw UsageError(message: "All matched dispatches were skipped at this position")
    }

    let medianOfMedians = kept.map(\.medianGpuUs).reduce(0, +) / Double(kept.count)
    let medianOfP95 = kept.map(\.p95GpuUs).reduce(0, +) / Double(kept.count)
    let avgOfAvg = kept.map(\.avgGpuUs).reduce(0, +) / Double(kept.count)
    print(
        "summary"
            + " matches=\(kept.count)"
            + " contains=\(contains)"
            + " meanMedianUs=\(String(format: "%.2f", medianOfMedians))"
            + " meanP95Us=\(String(format: "%.2f", medianOfP95))"
            + " meanAvgUs=\(String(format: "%.2f", avgOfAvg))"
    )
}

private struct DecodePlanRun {
    let pureGPUUs: Double
    let wallUs: Double
    let logitsByStep: [[Float16]]
}

private func executeDecodePlan(
    runtime: SmeltRuntime,
    records: [SmeltDispatchRecord],
    ids: [Int32],
    captureLogits: Bool
) throws -> DecodePlanRun {
    runtime.resetWorkingBuffers()
    var logitsByStep: [[Float16]] = []
    if captureLogits {
        logitsByStep.reserveCapacity(ids.count)
    }
    var finalGPUUs = 0.0
    var finalWallUs = 0.0
    for (index, id) in ids.enumerated() {
        let timing = try runtime.profileDecodeStep(
            tokenId: id,
            position: Int32(index),
            dispatchRecords: records
        )
        if captureLogits {
            logitsByStep.append(runtime.allLogitsHalf())
        }
        if index == ids.count - 1 {
            finalGPUUs = timing.pureGpuMs * 1_000
            finalWallUs = (
                timing.cpuMs + timing.gpuMs + timing.readMs
            ) * 1_000
        }
    }
    return DecodePlanRun(
        pureGPUUs: finalGPUUs,
        wallUs: finalWallUs,
        logitsByStep: logitsByStep
    )
}

private func comparePlanLogits(
    baseline: [[Float16]],
    candidate: [[Float16]]
) -> (exact: Bool, evidence: SmeltPlanParityEvidence) {
    let checkedSteps = min(baseline.count, candidate.count)
    let valuesPerStep = min(
        baseline.first?.count ?? 0,
        candidate.first?.count ?? 0
    )
    var firstStep: Int?
    var firstValue: Int?
    var maximumDifference = 0.0

    for step in 0..<checkedSteps {
        let valueCount = min(baseline[step].count, candidate[step].count)
        for value in 0..<valueCount {
            let lhs = baseline[step][value]
            let rhs = candidate[step][value]
            guard lhs.bitPattern != rhs.bitPattern else { continue }
            if firstStep == nil {
                firstStep = step
                firstValue = value
            }
            let lhsFloat = Double(Float(lhs))
            let rhsFloat = Double(Float(rhs))
            let difference = lhsFloat.isFinite && rhsFloat.isFinite
                ? abs(lhsFloat - rhsFloat)
                : .infinity
            maximumDifference = max(maximumDifference, difference)
        }
        if baseline[step].count != candidate[step].count, firstStep == nil {
            firstStep = step
            firstValue = valueCount
            maximumDifference = .infinity
        }
    }
    if baseline.count != candidate.count, firstStep == nil {
        firstStep = checkedSteps
        firstValue = 0
        maximumDifference = .infinity
    }

    let evidence = SmeltPlanParityEvidence(
        checkedSteps: checkedSteps,
        valuesPerStep: valuesPerStep,
        firstDivergenceStep: firstStep,
        firstDivergenceValue: firstValue,
        maximumAbsoluteDifference: maximumDifference
    )
    return (firstStep == nil, evidence)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
    return sorted[middle]
}

private func decodePlanStructure(
    records: [SmeltDispatchRecord],
    context: SmeltCostModelContext
) -> SmeltPlanStructuralCost {
    var recordCount = 0
    var dispatchCount = 0
    var swapCount = 0
    var totalThreadgroups = 0
    var singleThreadgroupDispatches = 0
    var maxThreadgroupsPerDispatch = 0
    var pipelines: Set<UInt16> = []

    for record in records {
        if record.opKind == SmeltDispatchRecord.opSwap {
            recordCount += 1
            swapCount += 1
            continue
        }
        guard record.opKind == SmeltDispatchRecord.opDispatch,
              decodeRecordExecutes(record, position: context.position)
        else {
            continue
        }
        let gridWidth = resolveGridDimension(
            kind: record.gridWKind,
            literal: Int(record.gridW),
            sequenceLength: context.sequenceLength
        )
        let gridHeight = resolveGridDimension(
            kind: record.gridHKind,
            literal: Int(record.gridH),
            sequenceLength: context.sequenceLength
        )
        let gridDepth = resolveGridDimension(
            kind: record.gridDKind,
            literal: Int(record.gridD),
            sequenceLength: context.sequenceLength
        )
        guard gridWidth > 0, gridHeight > 0, gridDepth > 0 else { continue }

        let threadgroups: Int
        if record.dispatchStyle == SmeltDispatchRecord.styleThreads {
            threadgroups = ceilDiv(gridWidth, Int(record.tgW))
                * ceilDiv(gridHeight, Int(record.tgH))
                * ceilDiv(gridDepth, Int(record.tgD))
        } else {
            threadgroups = gridWidth * gridHeight * gridDepth
        }
        recordCount += 1
        dispatchCount += 1
        totalThreadgroups += threadgroups
        if threadgroups == 1 {
            singleThreadgroupDispatches += 1
        }
        maxThreadgroupsPerDispatch = max(
            maxThreadgroupsPerDispatch,
            threadgroups
        )
        pipelines.insert(record.pipeline)
    }

    return SmeltPlanStructuralCost(
        recordCount: recordCount,
        dispatchCount: dispatchCount,
        swapCount: swapCount,
        totalThreadgroups: totalThreadgroups,
        singleThreadgroupDispatches: singleThreadgroupDispatches,
        maxThreadgroupsPerDispatch: maxThreadgroupsPerDispatch,
        distinctPipelines: pipelines.count
    )
}

private func decodeRecordExecutes(
    _ record: SmeltDispatchRecord,
    position: Int
) -> Bool {
    let positionPlusOne = UInt32(clamping: position + 1)
    for index in 0..<Int(record.constantCount) {
        let constant = getConstant(record, index: index)
        switch constant.kind {
        case SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse:
            if positionPlusOne >= constant.value { return false }
        case SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
            if positionPlusOne < constant.value { return false }
        default:
            continue
        }
    }
    return true
}

private func resolveGridDimension(
    kind: UInt8,
    literal: Int,
    sequenceLength: Int
) -> Int {
    switch kind {
    case SmeltDispatchRecord.gridSeqLen:
        return sequenceLength
    case SmeltDispatchRecord.gridSeqLenMulLiteral:
        return sequenceLength * literal
    case SmeltDispatchRecord.gridSeqLenCeilDivLiteral:
        let floorDivision = literal & Int(UInt32(1) << 31) != 0
        let divisor = max(literal & 0x7fff_ffff, 1)
        return floorDivision
            ? sequenceLength / divisor
            : ceilDiv(sequenceLength, divisor)
    default:
        return literal
    }
}

private func ceilDiv(_ value: Int, _ divisor: Int) -> Int {
    let safeDivisor = max(divisor, 1)
    return (value + safeDivisor - 1) / safeDivisor
}

private func compareDecodePlans(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let baselinePath = try parseArg(
        "--baseline-table",
        from: args,
        default: "\(package)/dispatches.bin"
    )
    let candidatePath = try parseArg("--candidate-table", from: args)
    let baselineID = try parseArg(
        "--baseline-id", from: args, default: "baseline"
    )
    let candidateID = try parseArg(
        "--candidate-id", from: args, default: "candidate"
    )
    let outputPath = try parseArg("--output", from: args, default: "")
    let warmup = Int(try parseArg("--warmup", from: args, default: "2")) ?? -1
    let iterations = Int(
        try parseArg("--iterations", from: args, default: "12")
    ) ?? 0
    guard !ids.isEmpty else {
        throw UsageError(message: "--ids must not be empty")
    }
    guard warmup >= 0 else {
        throw UsageError(message: "--warmup must be non-negative")
    }
    guard iterations > 0 else {
        throw UsageError(message: "--iterations must be positive")
    }

    let manifest = try loadManifest(packagePath: package)
    let baselineData = try Data(contentsOf: URL(fileURLWithPath: baselinePath))
    let candidateData = try Data(contentsOf: URL(fileURLWithPath: candidatePath))
    let baselineRecords = try loadDispatchTable(path: baselinePath)
    let candidateRecords = try loadDispatchTable(path: candidatePath)
    let runtime = try SmeltRuntime(packagePath: package)
    try runtime.prepareForRequest(
        batchCapacity: max(ids.count, 1),
        contextCapacity: max(ids.count, 1)
    )

    // Parity is checked independently of timing. Each plan starts from zeroed
    // request state and replays the same token prefix through its own graph.
    let baselineParityRun = try executeDecodePlan(
        runtime: runtime,
        records: baselineRecords,
        ids: ids,
        captureLogits: true
    )
    let candidateParityRun = try executeDecodePlan(
        runtime: runtime,
        records: candidateRecords,
        ids: ids,
        captureLogits: true
    )
    let parity = comparePlanLogits(
        baseline: baselineParityRun.logitsByStep,
        candidate: candidateParityRun.logitsByStep
    )

    var samples: [SmeltPairedPlanSample] = []
    samples.reserveCapacity(iterations)
    for round in 0..<(warmup + iterations) {
        let baselineFirst = round.isMultiple(of: 2)
        let baselineRun: DecodePlanRun
        let candidateRun: DecodePlanRun
        if baselineFirst {
            baselineRun = try executeDecodePlan(
                runtime: runtime,
                records: baselineRecords,
                ids: ids,
                captureLogits: false
            )
            candidateRun = try executeDecodePlan(
                runtime: runtime,
                records: candidateRecords,
                ids: ids,
                captureLogits: false
            )
        } else {
            candidateRun = try executeDecodePlan(
                runtime: runtime,
                records: candidateRecords,
                ids: ids,
                captureLogits: false
            )
            baselineRun = try executeDecodePlan(
                runtime: runtime,
                records: baselineRecords,
                ids: ids,
                captureLogits: false
            )
        }
        guard round >= warmup else { continue }
        samples.append(SmeltPairedPlanSample(
            order: baselineFirst ? .baselineFirst : .candidateFirst,
            baselineGPUUs: baselineRun.pureGPUUs,
            candidateGPUUs: candidateRun.pureGPUUs,
            baselineWallUs: baselineRun.wallUs,
            candidateWallUs: candidateRun.wallUs
        ))
    }

    let provenanceText = [
        String(runtime.metalDevice.registryID),
        runtime.metalDevice.name,
        ProcessInfo.processInfo.operatingSystemVersionString,
        manifest.checksums.weightsBin,
        manifest.checksums.metallib,
        manifest.buildProvenance?.compilerSourcesSHA256 ?? "",
    ].joined(separator: "|")
    let provenanceKey = sha256Hex(Data(provenanceText.utf8))
    let context = SmeltCostModelContext(
        mode: .decode,
        sequenceLength: ids.count,
        position: ids.count - 1
    )
    let measurement = SmeltPairedPlanMeasurement(
        provenanceKey: provenanceKey,
        context: context,
        baselinePlanID: baselineID,
        candidatePlanID: candidateID,
        exactOutputMatch: parity.exact,
        samples: samples,
        baselineTableSHA256: sha256Hex(baselineData),
        candidateTableSHA256: sha256Hex(candidateData),
        baselineStructure: decodePlanStructure(
            records: baselineRecords,
            context: context
        ),
        candidateStructure: decodePlanStructure(
            records: candidateRecords,
            context: context
        ),
        parity: parity.evidence
    )
    let decisionPolicy = SmeltPlanDecisionPolicy()
    let decision = SmeltPlanCostDecider(
        provenanceKey: provenanceKey,
        policy: decisionPolicy
    ).decide(measurement: measurement)
    let report = SmeltPlanComparisonReport(
        measurement: measurement,
        decisionPolicy: decisionPolicy,
        decision: decision
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let measurementData = try encoder.encode(report)
    if outputPath.isEmpty {
        print(String(decoding: measurementData, as: UTF8.self))
    } else {
        try measurementData.write(
            to: URL(fileURLWithPath: outputPath),
            options: .atomic
        )
    }

    let baselineMedian = median(samples.map(\.baselineGPUUs))
    let candidateMedian = median(samples.map(\.candidateGPUUs))
    let improvement = baselineMedian - candidateMedian
    let winFraction = Double(samples.filter {
        $0.candidateGPUUs < $0.baselineGPUUs
    }.count) / Double(samples.count)
    fputs(
        "compare-decode-plans exact=\(parity.exact)"
            + " baselineMedianGPUUs=\(String(format: "%.2f", baselineMedian))"
            + " candidateMedianGPUUs=\(String(format: "%.2f", candidateMedian))"
            + " candidateImprovementUs=\(String(format: "%.2f", improvement))"
            + " candidateWinFraction=\(String(format: "%.3f", winFraction))"
            + " pairs=\(samples.count)"
            + " selected=\(decision.selectedPlanID)"
            + " reason=\(decision.reason.rawValue)"
            + (outputPath.isEmpty ? "\n" : " output=\(outputPath)\n"),
        stderr
    )
}

private func tokenizePrompt(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let prompt = try parseArg("--prompt", from: args, default: "")
    let promptFile = try parseArg("--prompt-file", from: args, default: "")
    let prependBOS = try parseArg("--prepend-bos", from: args, default: "auto")

    let text: String
    if !prompt.isEmpty {
        text = prompt
    } else if !promptFile.isEmpty {
        text = try String(contentsOfFile: promptFile, encoding: .utf8)
    } else {
        throw UsageError(message: "Provide --prompt or --prompt-file")
    }

    let tokenizer = try SmeltTokenizer(path: "\(package)/tokenizer.json")
    var ids = tokenizer.encode(text)
    switch prependBOS {
    case "auto":
        if let bos = tokenizer.bosTokenId {
            ids.insert(Int32(bos), at: 0)
        }
    case "always":
        guard let bos = tokenizer.bosTokenId else {
            throw UsageError(message: "Tokenizer has no BOS token for --prepend-bos always")
        }
        ids.insert(Int32(bos), at: 0)
    case "never":
        break
    default:
        throw UsageError(message: "--prepend-bos must be auto, always, or never")
    }

    print(ids.map(String.init).joined(separator: ","))
    print("count=\(ids.count)")
}

private func textRuntimeStats(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let batchCapacity = try parseOptionalPositiveIntArg(
        "--batch-capacity", from: args) ?? 1
    let contextCapacity = try parseOptionalPositiveIntArg(
        "--context-capacity", from: args)

    let runtime = try SmeltRuntime(
        packagePath: package,
        contextLimit: contextCapacity
    )
    try runtime.prepareForRequest(
        batchCapacity: batchCapacity,
        contextCapacity: contextCapacity ?? 1
    )
    let stats = runtime.memoryStats()

    print("totalAllocatedBytes=\(stats.totalAllocatedBytes)")
    print("weightBytes=\(stats.weightBytes)")
    print("persistentBytes=\(stats.persistentBytes)")
    print("batchScopedBytes=\(stats.batchScopedBytes)")
    print("contextScopedBytes=\(stats.contextScopedBytes)")
    print("batchCapacity=\(stats.currentBatchCapacity)")
    print("contextCapacity=\(stats.currentContextCapacity)")
    print(
        "weightBufferBytes="
            + runtime.metalWeightBufferLengths.map(String.init).joined(separator: ",")
    )
}

private func benchVerifyArgmax(args: [String]) throws {
    let package = try parseArg("--package", from: args)
    let ids = try parseCSVInt32(try parseArg("--ids", from: args))
    let parsedBatches = try parseCSVInt32(
        try parseArg("--batches", from: args, default: "1,2,4")
    ).map(Int.init)
    let batches = Array(Set(parsedBatches)).sorted()
    let warmup = Int(try parseArg("--warmup", from: args, default: "2")) ?? -1
    let iterations = Int(
        try parseArg("--iterations", from: args, default: "10")
    ) ?? -1
    guard warmup >= 0, iterations > 0 else {
        throw UsageError(message: "--warmup must be non-negative and --iterations positive")
    }
    guard !batches.isEmpty, batches.allSatisfy({ $0 > 0 && $0 <= ids.count }) else {
        throw UsageError(
            message: "--batches must contain positive sizes no larger than --ids count"
        )
    }

    let maximumBatch = batches.max()!
    let verify = try SmeltRuntime(packagePath: package, contextLimit: maximumBatch + 1)
    guard verify.canChunkedPrefillVerifyArgmax(tokenCount: maximumBatch) else {
        throw UsageError(
            message: "package cannot run transactional verify-argmax at batch \(maximumBatch)"
        )
    }
    try verify.prepareForRequest(
        batchCapacity: maximumBatch,
        contextCapacity: maximumBatch + 1
    )

    let sequential = try SmeltRuntime(
        packagePath: package, contextLimit: maximumBatch + 1
    )
    try sequential.prepareForRequest(
        batchCapacity: 1,
        contextCapacity: maximumBatch + 1
    )
    var expected: [Int32] = []
    expected.reserveCapacity(maximumBatch)
    for index in 0..<maximumBatch {
        expected.append(
            try sequential.decodeStep(
                tokenId: ids[index], position: Int32(index), selectionMode: .argmax
            )
        )
    }

    var baselinePerRowMs: Double?
    print(
        "verify-argmax package=\(package) batches="
            + batches.map(String.init).joined(separator: ",")
            + " warmup=\(warmup) iterations=\(iterations)"
    )
    for batch in batches {
        let tokens = Array(ids.prefix(batch))
        let reference = Array(expected.prefix(batch))
        verify.resetWorkingBuffers()
        let actual = try verify.prefillVerifyArgmax(tokens: tokens, startPos: 0)
        guard actual == reference else {
            let mismatch = zip(actual, reference).enumerated().first {
                $0.element.0 != $0.element.1
            }
            let detail = mismatch.map {
                "row \($0.offset): verify=\($0.element.0) sequential=\($0.element.1)"
            } ?? "row-count mismatch verify=\(actual.count) sequential=\(reference.count)"
            throw UsageError(message: "verify-argmax parity failed at B\(batch), \(detail)")
        }

        for _ in 0..<warmup {
            verify.resetWorkingBuffers()
            _ = try verify.prefillVerifyArgmax(tokens: tokens, startPos: 0)
        }
        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            verify.resetWorkingBuffers()
            let start = CFAbsoluteTimeGetCurrent()
            _ = try verify.prefillVerifyArgmax(tokens: tokens, startPos: 0)
            samples.append((CFAbsoluteTimeGetCurrent() - start) * 1_000)
        }
        let totalMs = median(samples)
        let perRowMs = totalMs / Double(batch)
        if batch == 1 { baselinePerRowMs = perRowMs }
        let base = String(
            format: "B%d parity=exact total_median_ms=%.3f per_row_ms=%.3f rows_per_s=%.2f",
            batch, totalMs, perRowMs, 1_000 / perRowMs
        )
        let speedup = baselinePerRowMs.map {
            String(format: " speedup_vs_B1=%.3fx", $0 / perRowMs)
        } ?? ""
        let outputs = " outputs=" + actual.map(String.init).joined(separator: ",")
        print(base + speedup + outputs)
    }
}

private func checkQwen35Vision(args: [String]) throws {
    let modulePath = try parseArg("--module", from: args)
    let checkpointPath = try parseArg("--checkpoint", from: args)
    let moduleData = try Data(contentsOf: URL(fileURLWithPath: modulePath))
    let module = try JSONDecoder().decode(SmeltCAMIR.self, from: moduleData).validated()
    let checkpoint = try SafetensorsLoader(directory: checkpointPath)
    let plan = try SmeltQwen35VisionCheckpointPlan(
        module: module,
        checkpoint: checkpoint
    )
    let c = plan.config
    print(
        "vision-check passed"
            + " source=\(plan.sourceID)"
            + " tensors=\(plan.tensors.count)"
            + " layers=\(c.layerCount)"
            + " hidden=\(c.hiddenSize)"
            + " heads=\(c.headCount)"
            + " ffn=\(c.intermediateSize)"
            + " patch=\(c.temporalPatchSize)x\(c.patchSize)x\(c.patchSize)"
            + " merge=\(c.spatialMergeSize)"
            + " output=\(c.outputHiddenSize)"
    )
}

private func runQwen35Vision(args: [String]) throws {
    let componentPath = try parseArg("--component", from: args, default: "")
    let modulePath = try parseArg("--module", from: args, default: "")
    let checkpointPath = try parseArg("--checkpoint", from: args, default: "")
    let shaderDirectory = try parseArg(
        "--shader-dir",
        from: args,
        default: "Resources/Shaders"
    )
    if componentPath.isEmpty && (modulePath.isEmpty || checkpointPath.isEmpty) {
        throw UsageError(
            message: "vision run requires --component or both --module and --checkpoint"
        )
    }
    let height = try parseOptionalPositiveIntArg("--height", from: args) ?? 2
    let width = try parseOptionalPositiveIntArg("--width", from: args) ?? 2
    let temporal = try parseOptionalPositiveIntArg("--temporal", from: args) ?? 1
    let packedSegments = try parseOptionalPositiveIntArg(
        "--packed-segments", from: args
    ) ?? 1
    let warmupIterations = try parseOptionalPositiveIntArg("--warmup", from: args) ?? 0
    let measuredIterations = try parseOptionalPositiveIntArg("--iterations", from: args) ?? 1
    let diagnoseNonFinite = args.contains("--diagnose")
    let referencePath = try parseArg("--reference-f32", from: args, default: "")
    let outputPath = try parseArg("--output-f32", from: args, default: "")
    let costJSONPath = try parseArg("--cost-json", from: args, default: "")
    let costCalibrationJSONPath = try parseArg(
        "--cost-calibration-json",
        from: args,
        default: ""
    )
    let profileCostIterations = try parseOptionalPositiveIntArg(
        "--profile-cost-iterations",
        from: args
    ) ?? (args.contains("--profile-cost") ? 1 : 0)
    let imagePath = try parseArg("--image", from: args, default: "")
    let gemmBackendText = try parseArg("--gemm", from: args, default: "mps")
    let gemmBackend: SmeltQwen35VisionGEMMBackend
    switch gemmBackendText {
    case "mps": gemmBackend = .mps
    case "reference-metal": gemmBackend = .referenceMetal
    default:
        throw UsageError(message: "--gemm must be mps or reference-metal")
    }
    let attentionBackendText = try parseArg(
        "--attention", from: args, default: "mps-staged"
    )
    let attentionBackend: SmeltQwen35VisionAttentionBackend
    switch attentionBackendText {
    case "reference": attentionBackend = .reference
    case "mps-staged": attentionBackend = .mpsStaged
    default:
        throw UsageError(
            message: "--attention must be reference or mps-staged"
        )
    }

    let artifact: SmeltQwen35VisionArtifact?
    let moduleData: Data
    let module: SmeltCAMIR
    let checkpoint: any CheckpointTensorSource
    let modelSource: String
    if componentPath.isEmpty {
        artifact = nil
        moduleData = try Data(contentsOf: URL(fileURLWithPath: modulePath))
        module = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: moduleData
        ).validated()
        checkpoint = try SafetensorsLoader(directory: checkpointPath)
        modelSource = "source-checkpoint"
    } else {
        let loaded = try SmeltQwen35VisionArtifact(
            path: componentPath,
            verify: true
        )
        artifact = loaded
        module = loaded.module
        moduleData = try module.canonicalJSONData(prettyPrinted: false)
        checkpoint = loaded
        modelSource = "verified-component"
    }
    let plan = try SmeltQwen35VisionCheckpointPlan(
        module: module,
        checkpoint: checkpoint
    )
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw UsageError(message: "Metal device/queue unavailable")
    }

    let loadStart = ContinuousClock.now
    let weights = try SmeltQwen35VisionWeights(
        device: device,
        checkpoint: checkpoint,
        plan: plan
    )
    let pipelines: SmeltQwen35VisionPipelines
    if let artifact {
        pipelines = try SmeltQwen35VisionPipelines(
            device: device,
            library: artifact.makeLibrary(device: device)
        )
    } else {
        pipelines = try SmeltQwen35VisionPipelines(
            device: device,
            shaderDirectory: shaderDirectory
        )
    }
    let runtime = SmeltQwen35VisionRuntime(
        device: device,
        queue: queue,
        config: plan.config,
        weights: weights,
        pipelines: pipelines,
        gemmBackend: gemmBackend,
        attentionBackend: attentionBackend
    )
    let loadSeconds = ContinuousClock.now - loadStart

    let preprocessStart = ContinuousClock.now
    let grid: SmeltQwen35VisionRuntime.Grid
    let patches: [Float]
    let inputDescription: String
    if imagePath.isEmpty {
        grid = SmeltQwen35VisionRuntime.Grid(
            temporal: temporal,
            height: height,
            width: width
        )
        let patchWidth = plan.config.inChannels
            * plan.config.temporalPatchSize
            * plan.config.patchSize
            * plan.config.patchSize
        patches = [Float](repeating: 0, count: grid.patchCount * patchWidth)
        inputDescription = "zero-grid"
    } else {
        let preprocessor = SmeltQwen35ImagePreprocessor(
            config: try SmeltQwen35ImagePreprocessorConfig(module: module)
        )
        let result = try preprocessor.preprocess(imageAt: URL(fileURLWithPath: imagePath))
        grid = result.grid
        patches = result.patches
        inputDescription = "image:\(result.resizedWidth)x\(result.resizedHeight)"
    }
    let basePatches = patches
    let packedPatches = packedSegments == 1
        ? basePatches
        : (0..<packedSegments).flatMap { segment in
            let offset = Float(segment) * 0.03125
            return basePatches.map { $0 + offset }
        }
    let grids = [SmeltQwen35VisionRuntime.Grid](
        repeating: grid,
        count: packedSegments
    )
    let preprocessSeconds = ContinuousClock.now - preprocessStart
    let supportsFrozenCost = gemmBackend == .mps && attentionBackend == .mpsStaged
    let requestedFrozenCost = profileCostIterations > 0
        || !costJSONPath.isEmpty
        || !costCalibrationJSONPath.isEmpty
    if requestedFrozenCost && !supportsFrozenCost {
        throw UsageError(
            message: "frozen vision costing requires --gemm mps --attention mps-staged"
        )
    }
    let frozenPlanProvenance: String
    if let artifact {
        let checksums = artifact.manifest.checksums
        frozenPlanProvenance = "compiled-component"
            + ":schema=\(artifact.manifest.schema)"
            + ":cam=\(checksums.camSHA256)"
            + ":weights=\(checksums.weightsSHA256)"
            + ":metallib=\(checksums.metallibSHA256)"
    } else {
        frozenPlanProvenance = "source-component"
            + ":module=\(sha256Hex(moduleData))"
            + ":source=\(plan.sourceID)"
    }
    let frozenPlan = supportsFrozenCost ? try SmeltQwen35VisionRuntime.frozenPlan(
        config: plan.config,
        grids: grids,
        provenanceKey: frozenPlanProvenance,
        gemmBackend: gemmBackend,
        attentionBackend: attentionBackend
    ) : nil
    var costReport = frozenPlan.map { SmeltFrozenIRCostModel.report(plan: $0) }
    for _ in 0..<warmupIterations {
        _ = try runtime.encode(
            patches: packedPatches,
            grids: grids,
            diagnoseNonFinite: diagnoseNonFinite
        )
    }
    var encodeSamples: [Double] = []
    var workspaceSamples: [Double] = []
    var patchCopySamples: [Double] = []
    var positionSamples: [Double] = []
    var setupSamples: [Double] = []
    var commandEncodingSamples: [Double] = []
    var commandExecutionSamples: [Double] = []
    var gpuSamples: [Double] = []
    encodeSamples.reserveCapacity(measuredIterations)
    var output: SmeltQwen35VisionRuntime.Output?
    for _ in 0..<measuredIterations {
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let measuredOutput = try runtime.encode(
            patches: packedPatches,
            grids: grids,
            diagnoseNonFinite: diagnoseNonFinite
        )
        output = measuredOutput
        encodeSamples.append(CFAbsoluteTimeGetCurrent() - encodeStart)
        workspaceSamples.append(measuredOutput.timing.workspaceSeconds)
        patchCopySamples.append(measuredOutput.timing.patchCopySeconds)
        positionSamples.append(measuredOutput.timing.positionSeconds)
        setupSamples.append(measuredOutput.timing.setupSeconds)
        commandEncodingSamples.append(measuredOutput.timing.commandEncodingSeconds)
        commandExecutionSamples.append(measuredOutput.timing.commandExecutionSeconds)
        gpuSamples.append(measuredOutput.timing.gpuSeconds)
    }
    guard let output else {
        throw UsageError(message: "vision benchmark produced no output")
    }
    let encodeSeconds = median(encodeSamples)
    let values = output.values()
    guard values.allSatisfy(\.isFinite) else {
        throw UsageError(message: "vision output contains non-finite values")
    }
    var costCalibration: SmeltDeviceCostCalibration?
    if profileCostIterations > 0 {
        guard let frozenPlan else {
            throw UsageError(message: "vision route has no frozen cost plan")
        }
        var profiles: [SmeltFrozenIRExecutionProfile] = []
        profiles.reserveCapacity(profileCostIterations)
        for profileIndex in 0..<profileCostIterations {
            let profiledOutput = try runtime.encode(
                patches: packedPatches,
                grids: grids,
                profileFrozenOperations: true
            )
            guard let operationProfile = profiledOutput.operationProfile else {
                throw UsageError(
                    message: "vision cost profile \(profileIndex) produced no operation samples"
                )
            }
            let profiledValues = profiledOutput.values()
            guard profiledValues.count == values.count,
                  zip(profiledValues, values).allSatisfy({
                      $0.0.bitPattern == $0.1.bitPattern
                  })
            else {
                throw UsageError(
                    message: "vision cost profile \(profileIndex) changed output bits"
                )
            }
            profiles.append(SmeltFrozenIRExecutionProfile(
                provenanceKey: frozenPlan.provenanceKey,
                context: frozenPlan.context,
                deviceName: device.name,
                measurementMethod: operationProfile.measurementMethod,
                wholePlanGPUUs: operationProfile.wholePlanGPUUs,
                spans: operationProfile.spans.map {
                    SmeltFrozenIRExecutionSpan(
                        recordIndices: $0.recordIndices,
                        gpuUs: $0.gpuUs
                    )
                }
            ))
        }
        let calibration = try SmeltFrozenIRCostModel.calibration(
            plan: frozenPlan,
            profiles: profiles,
            cleanWholePlanGPUUs: gpuSamples.map { $0 * 1_000_000 },
            hostRecordUs: median(commandEncodingSamples) * 1_000_000
                / Double(max(frozenPlan.records.count, 1))
        )
        costCalibration = calibration
        costReport = SmeltFrozenIRCostModel.report(
            plan: frozenPlan,
            calibration: calibration
        )
    }
    if !costJSONPath.isEmpty {
        guard let costReport else {
            throw UsageError(message: "vision route has no frozen cost report")
        }
        try costReport.encodeJSON().write(
            to: URL(fileURLWithPath: costJSONPath),
            options: .atomic
        )
    }
    if !costCalibrationJSONPath.isEmpty {
        guard let costCalibration else {
            throw UsageError(
                message: "--cost-calibration-json requires --profile-cost"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(costCalibration).write(
            to: URL(fileURLWithPath: costCalibrationJSONPath),
            options: .atomic
        )
    }
    if !outputPath.isEmpty {
        try values.withUnsafeBytes { raw in
            try Data(raw).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        }
    }
    let minimum = values.min() ?? 0
    let maximum = values.max() ?? 0
    let checksum = values.reduce(Float(0), +)
    let prefix = values.prefix(8).map(formatExact).joined(separator: ",")
    var parity = ""
    if !referencePath.isEmpty {
        let data = try Data(contentsOf: URL(fileURLWithPath: referencePath))
        guard data.count == values.count * MemoryLayout<Float>.stride else {
            throw UsageError(
                message: "reference has \(data.count) bytes; expected \(values.count * 4)"
            )
        }
        let reference = data.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        var dot: Double = 0, actualNorm: Double = 0, referenceNorm: Double = 0
        var errorNorm: Double = 0
        var maximumAbsolute: Float = 0
        for (actual, expected) in zip(values, reference) {
            let a = Double(actual), e = Double(expected), d = a - e
            dot += a * e
            actualNorm += a * a
            referenceNorm += e * e
            errorNorm += d * d
            maximumAbsolute = max(maximumAbsolute, abs(actual - expected))
        }
        let cosine = dot / (actualNorm.squareRoot() * referenceNorm.squareRoot())
        let relativeL2 = (errorNorm / referenceNorm).squareRoot()
        parity = " max_abs=\(formatExact(maximumAbsolute))"
            + " rel_l2=\(String(format: "%.8g", relativeL2))"
            + " cosine=\(String(format: "%.8g", cosine))"
    }
    let encodeMedianText = String(format: "%.3f", encodeSeconds)
    let encodeSamplesText = encodeSamples
        .map { String(format: "%.3f", $0) }
        .joined(separator: ",")
    let logicalReadBytes = costReport?.summary.storageTotals.reduce(UInt64(0)) {
        $0 &+ $1.readBytes
    } ?? 0
    let logicalWriteBytes = costReport?.summary.storageTotals.reduce(UInt64(0)) {
        $0 &+ $1.writeBytes
    } ?? 0
    let fp32Operations = costReport?.summary.operationTotals.first {
        $0.operationClass == .fp32Arithmetic
    }?.count ?? 0
    var measuredGroupGPUUs: [String: Double] = [:]
    for record in costReport?.records ?? [] {
        guard let group = record.operationGroup,
              let gpuUs = record.calibratedMedianGPUUs
        else { continue }
        measuredGroupGPUUs[group, default: 0] += gpuUs
    }
    let measuredGroups = measuredGroupGPUUs.sorted {
        if $0.value != $1.value { return $0.value > $1.value }
        return $0.key < $1.key
    }.map {
        "\($0.key):\(String(format: "%.3f", $0.value / 1_000))ms"
    }.joined(separator: ",")
    let instrumentedSpanGPUUs = costCalibration?.instrumentedSpanMedianGPUUs ?? 0
    let reconciliationScale = instrumentedSpanGPUUs > 0
        ? (costCalibration?.wholePlanMedianGPUUs ?? 0) / instrumentedSpanGPUUs
        : 0
    print(
        "vision-run passed"
            + " source=\(modelSource)"
            + " input=\(inputDescription)"
            + " gemm=\(gemmBackendText)"
            + " attention=\(attentionBackendText)"
            + " packed_segments=\(packedSegments)"
            + " grid=\(grid.temporal)x\(grid.height)x\(grid.width)"
            + " output=\(output.tokenCount)x\(output.hiddenSize)"
            + " load_s=\(loadSeconds.components.seconds).\(String(format: "%03d", loadSeconds.components.attoseconds / 1_000_000_000_000_000))"
            + " preprocess_s=\(preprocessSeconds.components.seconds).\(String(format: "%03d", preprocessSeconds.components.attoseconds / 1_000_000_000_000_000))"
            + " encode_s=\(encodeMedianText)"
            + " encode_samples_s=\(encodeSamplesText)"
            + " workspace_s=\(String(format: "%.3f", median(workspaceSamples)))"
            + " patch_copy_s=\(String(format: "%.3f", median(patchCopySamples)))"
            + " position_s=\(String(format: "%.3f", median(positionSamples)))"
            + " setup_s=\(String(format: "%.3f", median(setupSamples)))"
            + " command_encode_s=\(String(format: "%.3f", median(commandEncodingSamples)))"
            + " command_execute_s=\(String(format: "%.3f", median(commandExecutionSamples)))"
            + " gpu_s=\(String(format: "%.3f", median(gpuSamples)))"
            + " cost_dispatches=\(costReport?.summary.dispatchCount ?? 0)"
            + " cost_read_gib=\(String(format: "%.3f", Double(logicalReadBytes) / 1_073_741_824))"
            + " cost_write_gib=\(String(format: "%.3f", Double(logicalWriteBytes) / 1_073_741_824))"
            + " cost_fp32_tops=\(String(format: "%.3f", Double(fp32Operations) / 1.0e12))"
            + " cost_materialized_gib=\(String(format: "%.3f", Double(costReport?.summary.intermediateMaterializationBytes ?? 0) / 1_073_741_824))"
            + " cost_profile_iterations=\(profileCostIterations)"
            + " cost_calibrated_gpu_s=\(String(format: "%.3f", (costReport?.summary.calibratedMedianGPUUs ?? 0) / 1_000_000))"
            + " cost_profiled_gpu_s=\(String(format: "%.3f", (costCalibration?.instrumentedWholePlanMedianGPUUs ?? 0) / 1_000_000))"
            + " cost_reconcile_scale=\(String(format: "%.6f", reconciliationScale))"
            + " cost_additive_error=\(String(format: "%.4f", costReport?.summary.additiveCalibrationErrorFraction ?? 0))"
            + (measuredGroups.isEmpty ? "" : " cost_groups=\(measuredGroups)")
            + " min=\(formatExact(minimum))"
            + " max=\(formatExact(maximum))"
            + " sum=\(formatExact(checksum))"
            + " first=\(prefix)"
            + parity
    )
}



private func usage() -> Never {
    fputs(
        """
        Usage:
          smelt-probe tokenize --package <pkg> (--prompt <text> | --prompt-file <path>) [--prepend-bos auto|always|never]
          smelt-probe text-runtime-stats --package <pkg> [--batch-capacity N] [--context-capacity N]
          smelt-probe bench-verify-argmax --package <pkg> --ids <csv> [--batches 1,2,4] [--warmup N] [--iterations N]
          smelt-probe check-qwen35-vision --module <model.module.json> --checkpoint <hf-dir>
          smelt-probe run-qwen35-vision (--component <agentcomponent> | --module <model.module.json> --checkpoint <hf-dir> [--shader-dir <dir>]) [--image <file> | --temporal N --height N --width N] [--packed-segments N] [--gemm mps|reference-metal] [--attention reference|mps-staged] [--reference-f32 <file>] [--output-f32 <file>] [--cost-json <file>] [--profile-cost | --profile-cost-iterations N] [--cost-calibration-json <file>]
          smelt-probe compare-rows --package <pkg> [--slot <n> | --prefill-slot <n> --decode-slot <n>] --ids <csv> --row-width <n> --prefill-dispatch <n> --decode-dispatch <n> [--decode-row-index base|last|N]
          smelt-probe compare-final-slot --package <pkg> [--context-limit N] [--slot <n> | --prefill-slot <n> --decode-slot <n>] --ids <csv> --count <n>
          smelt-probe generate-ids --package <pkg> [--context-limit N] --ids <csv> [--prefix-mode prefill|decode] [--max-tokens N] [--top-k N]
          smelt-probe compare-label --package <pkg> [--context-limit N] --ids <csv> --label <name> [--occurrence first|last|N]
          smelt-probe analyze-rmsnorm-add --package <pkg> --ids <csv> --prefill-norm-dispatch <n> --prefill-add-dispatch <n> --decode-dispatch <n> [--prefill-row-index base|last|N] [--dim <n>] [--eps <f>]
          smelt-probe list-dispatches --package <pkg> [--mode decode|prefill] [--contains <substring>]
          smelt-probe bench-dispatch --package <pkg> --ids <csv> --dispatch <n> [--warmup <n>] [--iterations <n>]
          smelt-probe bench-pipeline --package <pkg> --ids <csv> --contains <substring> [--warmup <n>] [--iterations <n>]
          smelt-probe compare-decode-plans --package <pkg> --ids <csv> --candidate-table <path> [--baseline-table <path>] [--baseline-id <id>] [--candidate-id <id>] [--warmup <n>] [--iterations <n>] [--output <json>]
          smelt-probe inspect-label --package <pkg> [--context-limit N] --ids <csv> --label <name> [--mode prefill|decode|both] [--occurrence first|last|N] [--prefill-count N] [--buffer-index N] [--row-index base|last|N] [--force-dtype fp16|fp32|int32] [--value-offset <n>] [--values <n>]
          smelt-probe inspect-dispatch --package <pkg> [--context-limit N] --ids <csv> --mode prefill|decode --dispatch <n> [--prefill-count N] [--buffer-index N] [--row-index base|last|N] [--force-dtype fp16|fp32|int32] [--value-offset <n>] [--values <n>]
          smelt-probe compare-label-binding --package <pkg> [--context-limit N] --ids <csv> --label <name> --buffer-index <n> [--occurrence first|last|N] [--prefill-row-index base|last|N] [--decode-row-index base|last|N] [--force-dtype fp16|fp32|int32] [--value-offset <n>] [--count <n>]

        Example:
          smelt-probe compare-rows --package <pkg> --ids 2,818,5279,529,7001,563 --slot 22 --row-width 512 --prefill-dispatch 3392 --decode-dispatch 133
        """,
        stderr
    )
    exit(2)
}

do {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let command = args.first else { usage() }

    switch command {
    case "tokenize":
        try tokenizePrompt(args: Array(args.dropFirst()))
    case "text-runtime-stats":
        try textRuntimeStats(args: Array(args.dropFirst()))
    case "bench-verify-argmax":
        try benchVerifyArgmax(args: Array(args.dropFirst()))
    case "check-qwen35-vision":
        try checkQwen35Vision(args: Array(args.dropFirst()))
    case "run-qwen35-vision":
        try runQwen35Vision(args: Array(args.dropFirst()))
    case "compare-rows":
        try compareRows(args: Array(args.dropFirst()))
    case "compare-final-slot":
        try compareFinalSlot(args: Array(args.dropFirst()))
    case "generate-ids":
        try generateFromTokenIDs(args: Array(args.dropFirst()))
    case "compare-label":
        try compareLabel(args: Array(args.dropFirst()))
    case "analyze-rmsnorm-add":
        try analyzeRMSNormAdd(args: Array(args.dropFirst()))
    case "list-dispatches":
        try listDispatches(args: Array(args.dropFirst()))
    case "bench-dispatch":
        try benchDispatch(args: Array(args.dropFirst()))
    case "bench-pipeline":
        try benchPipeline(args: Array(args.dropFirst()))
    case "compare-decode-plans":
        try compareDecodePlans(args: Array(args.dropFirst()))
    case "inspect-label":
        try inspectLabel(args: Array(args.dropFirst()))
    case "inspect-dispatch":
        try inspectDispatch(args: Array(args.dropFirst()))
    case "compare-label-binding":
        try compareLabelBinding(args: Array(args.dropFirst()))
    case "--help", "-h", "help":
        usage()
    default:
        throw UsageError(message: "Unknown command '\(command)'")
    }
} catch let error as UsageError {
    fputs("error: \(error.description)\n", stderr)
    usage()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
