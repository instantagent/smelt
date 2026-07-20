// SmeltAffineQuantizer — CPU-only affine (scale+bias) weight quantization.
//
// Quantizes model weights to u4 affine format: per-group min/max scaling.
// No GPU needed — computes min/max per column group on CPU, packs nibbles.
// Produces weights.bin via mmap'd streaming write — no monolithic RAM allocation.
//
// Storage layout per tensor:
//   Packed weights: [R, C/2] bytes   (2 nibbles per byte, low nibble first)
//   Scales:         [R, G]   float16 (G = ceil(C / groupSize))
//   Biases:         [R, G]   float16
//
// Dequantization: value = nibble * scale + bias
// Where: scale = (max - min) / 15, bias = min

import Foundation
import SmeltSchema

/// CPU-only affine weight quantizer.
public struct SmeltAffineQuantizer {

    /// Quantize all weights using affine (min/max) scaling and write to output file.
    ///
    /// - Parameters:
    ///   - tensors: Named tensor data from safetensors (host memory, may be mmap'd).
    ///   - config: Quantization config from IR (must have strategy == .affineU4).
    ///   - outputPath: Path to write weights.bin.
    /// - Returns: Weight entries matching the output file layout.
    public static func quantize(
        tensors: [(runtimeName: String, data: UnsafeRawPointer, byteCount: Int,
                    shape: [Int], dtype: String)],
        config: SmeltQuantizationConfig,
        outputPath: String,
        expectedLayout: [SmeltWeightEntry]? = nil,
        imatrix: [String: [Float]]? = nil,
        gptqBlocks: [String: SmeltAffineU4.Packed]? = nil,
        activationDtype: SmeltDType = .fp16
    ) throws -> [SmeltWeightEntry] {
        // Validate every source dtype up front — the write loop below handles
        // exactly these; anything else used to zero-fill or warn-and-skip
        // silently (Trap #3, same class as the emitMatvec fp16 fallthrough).
        let handledSourceDtypes: Set<String> = ["F32", "F16", "BF16", "I32"]
        for tensor in tensors where !tensor.shape.isEmpty && tensor.shape[0] > 0 {
            guard handledSourceDtypes.contains(tensor.dtype) else {
                throw SmeltAffineQuantizerError.unsupportedSourceDtype(
                    tensor: tensor.runtimeName, dtype: tensor.dtype
                )
            }
        }

        // Pre-compute layout (offsets for each weight in output file)
        var entries: [SmeltWeightEntry] = []
        var totalSize: UInt64 = 0

        for tensor in tensors {
            // Skip rank-0 or empty tensors
            guard !tensor.shape.isEmpty, tensor.shape[0] > 0 else {
                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName, offset: 0, sizeBytes: 0,
                    shape: [], dtype: .raw, groupSize: nil
                ))
                continue
            }

            let isQuantized = shouldQuantizeAffine(
                name: tensor.runtimeName, config: config
            )
            let useTurboQuantH = shouldUseTurboQuantH(
                name: tensor.runtimeName, config: config
            )
            let rows = tensor.shape[0]
            let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1

            // Guard against Int overflow
            let (elemCount, overflow) = rows.multipliedReportingOverflow(by: cols)
            guard !overflow, elemCount <= Int.max / 4 else {
                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName, offset: 0, sizeBytes: 0,
                    shape: [], dtype: .raw, groupSize: nil
                ))
                continue
            }

            // Odd columns can't be u4-packed (2 elements per byte)
            let canQuantize = isQuantized && cols % 2 == 0
            let canTurboQuantH = useTurboQuantH && tensor.shape.count == 2

            // Index buffers (I32 source) take precedence over the
            // affine path even when shape-compatible — quantizing a
            // permutation table as FP weights is silent corruption.
            if tensor.dtype == "I32" {
                let int32Size = UInt64(rows * cols * 4)
                let offset = align16(totalSize)
                totalSize = offset + int32Size

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: offset,
                    sizeBytes: int32Size,
                    shape: tensor.shape,
                    dtype: .int32,
                    groupSize: nil
                ))
            } else if SmeltWeightRole.preservesNativeBF16(
                name: tensor.runtimeName, activationDtype: activationDtype, config: config
            ) {
                // preserve_native: keep the projection at native bf16 (U2c), NOT
                // affine-quantized and NOT in GPTQ scope. BEFORE the TQH/affine/
                // shouldQuantize decisions so a preserved tensor leaves quant scope.
                // PREFLIGHT — bf16 source only (single-sourced in SmeltWeightRole).
                if let reason = SmeltWeightRole.preserveNativeSourceRejection(
                    dtype: tensor.dtype, byteCount: tensor.byteCount, rows: rows, cols: cols) {
                    throw SmeltAffineQuantizerError.preserveNativeRequiresBF16Source(
                        tensor: tensor.runtimeName, dtype: reason)
                }
                let bf16Size = UInt64(rows * cols * 2)
                let offset = align16(totalSize)
                totalSize = offset + bf16Size

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: offset,
                    sizeBytes: bf16Size,
                    shape: tensor.shape,
                    dtype: .bf16,
                    groupSize: nil
                ))
            } else if canTurboQuantH {
                // TurboQuant-H uses a fixed 128-element group internally;
                // see SmeltTurboQuantHQuantizer. The runtime accepts any
                // cols, padding the partial final group during encode.
                let plan = SmeltTurboQuantHQuantizer.plan(
                    rows: rows, cols: cols, groupSize: 128
                )

                let codesOffset = align128(totalSize)
                let codebookOffset = align128(codesOffset + plan.codesSizeBytes)
                totalSize = codebookOffset + plan.codebookSizeBytes

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: codesOffset,
                    sizeBytes: plan.codesSizeBytes,
                    shape: tensor.shape,
                    dtype: .turboQuantH,
                    groupSize: 128,
                    packedRowStride: (plan.paddedToGroups + 3) / 4,
                    paddedCols: plan.paddedToGroups,
                    codebookOffset: codebookOffset,
                    codebookSizeBytes: plan.codebookSizeBytes
                ))
            } else if canQuantize {
                let groupSize = config.groupSize
                let numColGroups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
                let packedSize = UInt64(rows * ((cols + 1) / 2))
                let scalesSize = UInt64(rows * numColGroups * 2)  // FP16
                let biasesSize = UInt64(rows * numColGroups * 2)  // FP16

                let packedOffset = align128(totalSize)
                let scalesOffset = align128(packedOffset + packedSize)
                let biasesOffset = align128(scalesOffset + scalesSize)
                totalSize = biasesOffset + biasesSize

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: packedOffset,
                    sizeBytes: packedSize,
                    shape: tensor.shape,
                    dtype: .affineU4,
                    groupSize: groupSize,
                    packedRowStride: (cols + 1) / 2,
                    paddedCols: cols,
                    scalesOffset: scalesOffset,
                    scalesSizeBytes: scalesSize,
                    biasesOffset: biasesOffset,
                    biasesSizeBytes: biasesSize
                ))
            } else {
                let fp16Size = UInt64(rows * cols * 2)
                let offset = align16(totalSize)
                totalSize = offset + fp16Size

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: offset,
                    sizeBytes: fp16Size,
                    shape: tensor.shape,
                    dtype: .fp16,
                    groupSize: nil
                ))
            }
        }

        if let expectedLayout {
            entries = try entriesUsingExpectedLayout(entries, expectedLayout: expectedLayout)
            totalSize = totalFileSize(entries)
        }

        // GPTQ coverage: no silent affine/GPTQ hybrid. Every in-scope projection
        // weight (resolved to affine_u4 here, after TQH routing) must have a
        // block, and every supplied block must map to such a weight.
        if let gptqBlocks {
            let scope = Set(SmeltGPTQScope.inResolvedScope(entries).map(\.name))
            for name in scope where gptqBlocks[name] == nil {
                throw SmeltAffineQuantizerError.gptqCoverageMissing(tensor: name)
            }
            for name in gptqBlocks.keys where !scope.contains(name) {
                throw SmeltAffineQuantizerError.gptqBlockOutOfScope(tensor: name)
            }
        }

        // Resume support: check for existing partial weights.bin + progress file
        let progressPath = outputPath + ".progress"
        var resumeFromIdx = 0
        let fm = FileManager.default

        // A GPTQ build forces a fresh write (below) and never resumes, so it must
        // neither trust nor leave a .progress marker: a crash mid-GPTQ-build leaves
        // full-size GPTQ bytes that a later plain-affine build would otherwise
        // resume over as if affine. Clear any stale marker up front.
        if gptqBlocks != nil { try? fm.removeItem(atPath: progressPath) }

        let existingSize = (try? fm.attributesOfItem(atPath: outputPath))?[.size] as? UInt64
        // A GPTQ build must never resume: resume keys only on weights.bin size +
        // .progress, but the GPTQ blocks live in memory and can't be matched
        // against the partial file. Resuming could keep stale bytes (a prior
        // plain-affine partial, or different blocks at the same layout size) and
        // silently produce an affine/GPTQ hybrid the coverage check can't catch.
        if gptqBlocks == nil,
           existingSize == totalSize,
           let progressData = try? String(contentsOfFile: progressPath, encoding: .utf8),
           let savedIdx = Int(progressData.trimmingCharacters(in: .whitespacesAndNewlines))
        {
            resumeFromIdx = savedIdx
            fputs("  Resuming affine quantization from tensor \(resumeFromIdx + 1)/\(tensors.count)\n", stderr)
        }

        // Create or open output file for writing
        let flags: Int32 = resumeFromIdx > 0
            ? (O_RDWR)
            : (O_RDWR | O_CREAT | O_TRUNC)
        let fd = open(outputPath, flags, 0o644)
        guard fd >= 0 else {
            throw SmeltAffineQuantizerError.fileCreateFailed(outputPath)
        }
        defer { close(fd) }

        if resumeFromIdx == 0 {
            ftruncate(fd, off_t(totalSize))
        }
        guard let mmapPtr = mmap(nil, Int(totalSize), PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, 0),
              mmapPtr != MAP_FAILED
        else {
            throw SmeltAffineQuantizerError.mmapFailed
        }
        defer {
            msync(mmapPtr, Int(totalSize), MS_SYNC)
            munmap(mmapPtr, Int(totalSize))
        }

        // Quantize each tensor
        fputs("  Output file: \(Int(totalSize)) bytes mmap'd (affine u4)\n", stderr)
        for (idx, tensor) in tensors.enumerated() {
            let entry = entries[idx]
            guard entry.sizeBytes > 0 else { continue }

            // Skip already-quantized tensors on resume
            if idx < resumeFromIdx {
                fputs("  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName) -> skipped (resume)\n", stderr)
                continue
            }
            let tensorStart = CFAbsoluteTimeGetCurrent()

            if entry.dtype == .affineU4 {
                let label: String
                if let block = gptqBlocks?[entry.name] {
                    // Precomputed GPTQ block → write it verbatim (coverage
                    // already checked above).
                    try fillAffineFromBlock(block: block, entry: entry, output: mmapPtr)
                    label = "affine_u4 (gptq)"
                } else {
                    quantizeTensorAffine(
                        source: tensor.data,
                        sourceBytes: tensor.byteCount,
                        sourceDtype: tensor.dtype,
                        shape: tensor.shape,
                        entry: entry,
                        config: config,
                        output: mmapPtr
                    )
                    label = "affine_u4"
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) -> \(label)  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else if entry.dtype == .turboQuantH {
                // Surface a name mismatch loudly: a provided imatrix that misses
                // a TQH target silently quantizes it unweighted, which would
                // invalidate a TQH-vs-TQH+imatrix comparison.
                if imatrix != nil, imatrix?[entry.name] == nil {
                    fputs(
                        "  warning: --imatrix has no entry for \(entry.name);"
                            + " quantizing unweighted\n",
                        stderr
                    )
                }
                try quantizeTensorTurboQuantH(
                    source: tensor.data,
                    sourceDtype: tensor.dtype,
                    shape: tensor.shape,
                    entry: entry,
                    importance: imatrix?[entry.name],
                    output: mmapPtr
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) -> turbo_quant_h  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else if entry.dtype == .int32 {
                let dst = mmapPtr.advanced(by: Int(entry.offset))
                let rows = tensor.shape[0]
                let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1
                let elementCount = rows * cols
                memcpy(dst, tensor.data, elementCount * 4)
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) -> int32  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else if entry.dtype == .bf16 {
                // preserved native bf16 — raw copy, no affine packing, no fp16 downcast.
                let dst = mmapPtr.advanced(by: Int(entry.offset))
                let rows = tensor.shape[0]
                let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1
                let elementCount = rows * cols
                memcpy(dst, tensor.data, elementCount * 2)
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) -> bf16 (preserved)  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else {
                // FP16: copy with dtype conversion if needed
                let dst = mmapPtr.advanced(by: Int(entry.offset))
                let rows = tensor.shape[0]
                let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1
                let elementCount = rows * cols
                switch tensor.dtype {
                case "BF16":
                    convertBF16toFP16(source: tensor.data, dest: dst, elementCount: elementCount)
                case "F32":
                    convertF32toFP16(source: tensor.data, dest: dst, elementCount: elementCount)
                case "F16":
                    memcpy(dst, tensor.data, elementCount * 2)
                default:
                    throw SmeltAffineQuantizerError.unsupportedSourceDtype(
                        tensor: tensor.runtimeName, dtype: tensor.dtype
                    )
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) -> fp16  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            }

            // Checkpoint progress after each tensor (enables resume on crash).
            // Not for GPTQ builds — they never resume, and a leftover marker would
            // let a later plain-affine build resume over the stale GPTQ bytes.
            if gptqBlocks == nil {
                try? "\(idx + 1)".write(toFile: progressPath, atomically: true, encoding: .utf8)
            }
            msync(mmapPtr, Int(totalSize), MS_ASYNC)
        }

        // All done — remove progress file
        try? fm.removeItem(atPath: progressPath)

        return entries
    }

    // MARK: - Per-tensor affine quantization

    /// Quantize a single tensor using affine (min/max) scaling.
    /// Groups along COLUMNS with the specified group size.
    ///
    /// For each (row, group):
    ///   scale = (max - min) / 15.0
    ///   bias = min
    ///   nibble = round((value - bias) / scale), clamped to [0, 15]
    private static func quantizeTensorAffine(
        source: UnsafeRawPointer,
        sourceBytes: Int,
        sourceDtype: String,
        shape: [Int],
        entry: SmeltWeightEntry,
        config: SmeltQuantizationConfig,
        output: UnsafeMutableRawPointer
    ) {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        let groupSize = config.groupSize
        let numColGroups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
        let sourceElementBytes: Int
        switch sourceDtype {
        case "F32":
            sourceElementBytes = 4
        case "F16", "BF16":
            sourceElementBytes = 2
        default:
            // Unreachable: quantize() validates source dtypes up front.
            preconditionFailure(
                "unsupported source dtype '\(sourceDtype)' reached "
                    + "quantizeTensorAffine for '\(entry.name)'"
            )
        }
        // Output pointers (shared layout derivation; see affineU4Destinations).
        let (packedDst, scalesDst, biasesDst) = affineU4Destinations(
            output: output, entry: entry, groupCount: numColGroups)

        let packedRowStride = (cols + 1) / 2

        final class QuantizeBuffers: @unchecked Sendable {
            let source: UnsafeRawPointer
            let packedDst: UnsafeMutablePointer<UInt8>
            let scalesDst: UnsafeMutablePointer<UInt16>
            let biasesDst: UnsafeMutablePointer<UInt16>

            init(
                source: UnsafeRawPointer,
                packedDst: UnsafeMutablePointer<UInt8>,
                scalesDst: UnsafeMutablePointer<UInt16>,
                biasesDst: UnsafeMutablePointer<UInt16>
            ) {
                self.source = source
                self.packedDst = packedDst
                self.scalesDst = scalesDst
                self.biasesDst = biasesDst
            }
        }
        let buffers = QuantizeBuffers(
            source: source,
            packedDst: packedDst,
            scalesDst: scalesDst,
            biasesDst: biasesDst
        )

        let quantizeRows: @Sendable (Range<Int>) -> Void = { rowRange in
            let rowValues = UnsafeMutablePointer<Float>.allocate(capacity: cols)
            defer { rowValues.deallocate() }

            for row in rowRange {
                let rowSource = sourceElementBytes > 0
                    ? buffers.source.advanced(by: row * cols * sourceElementBytes)
                    : buffers.source
                switch sourceDtype {
                case "F32":
                    for c in 0..<cols {
                        var val: Float = 0
                        memcpy(&val, rowSource.advanced(by: c * 4), 4)
                        rowValues[c] = val
                    }
                case "F16":
                    for c in 0..<cols {
                        var bits: UInt16 = 0
                        memcpy(&bits, rowSource.advanced(by: c * 2), 2)
                        rowValues[c] = Float(Float16(bitPattern: bits))
                    }
                case "BF16":
                    for c in 0..<cols {
                        var bf16Bits: UInt16 = 0
                        memcpy(&bf16Bits, rowSource.advanced(by: c * 2), 2)
                        let fp32Bits = UInt32(bf16Bits) << 16
                        rowValues[c] = Float(bitPattern: fp32Bits)
                    }
                default:
                    // Unreachable: quantize() validates source dtypes up front.
                    preconditionFailure(
                        "unsupported source dtype '\(sourceDtype)' reached "
                            + "affine row quantization"
                    )
                }

                SmeltAffineU4.quantizeRow(
                    values: rowValues, cols: cols, groupSize: groupSize,
                    packed: buffers.packedDst + row * packedRowStride,
                    scales: buffers.scalesDst + row * numColGroups,
                    biases: buffers.biasesDst + row * numColGroups)
            }
        }

        let shouldParallelize = rows >= 64 && cols >= 1024
        if shouldParallelize {
            let workerCount = min(ProcessInfo.processInfo.activeProcessorCount, rows)
            let rowsPerWorker = (rows + workerCount - 1) / workerCount
            DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
                let start = worker * rowsPerWorker
                let end = min(start + rowsPerWorker, rows)
                guard start < end else { return }
                quantizeRows(start..<end)
            }
        } else {
            quantizeRows(0..<rows)
        }
    }

    /// Convert the source tensor to fp16 row-major and feed it through
    /// SmeltTurboQuantHQuantizer. Write codes (2-bpw packed) at
    /// entry.offset, codebook (P × 4 fp16) at entry.codebookOffset.
    private static func quantizeTensorTurboQuantH(
        source: UnsafeRawPointer,
        sourceDtype: String,
        shape: [Int],
        entry: SmeltWeightEntry,
        importance: [Float]?,
        output: UnsafeMutableRawPointer
    ) throws {
        let rows = shape[0]
        let cols = shape[1]
        let elementCount = rows * cols
        guard ["BF16", "F32", "F16"].contains(sourceDtype) else {
            throw SmeltAffineQuantizerError.unsupportedSourceDtype(
                tensor: entry.name, dtype: sourceDtype
            )
        }
        var fp16Source = [Float16](repeating: 0, count: elementCount)
        fp16Source.withUnsafeMutableBytes { dst in
            switch sourceDtype {
            case "BF16":
                convertBF16toFP16(
                    source: source, dest: dst.baseAddress!,
                    elementCount: elementCount
                )
            case "F32":
                convertF32toFP16(
                    source: source, dest: dst.baseAddress!,
                    elementCount: elementCount
                )
            case "F16":
                memcpy(dst.baseAddress!, source, elementCount * 2)
            default:
                break  // unreachable: gated above
            }
        }

        if let importance {
            // Fail with an actionable error rather than the quantizer's opaque
            // length precondition trap when the imatrix doesn't fit this tensor.
            let padded = SmeltTurboQuantHQuantizer.plan(
                rows: rows, cols: cols, groupSize: 128
            ).paddedToGroups
            guard importance.count == padded else {
                throw SmeltAffineQuantizerError.imatrixLengthMismatch(
                    tensor: entry.name, expected: padded, actual: importance.count
                )
            }
            // External file input: convert the quantizer's value precondition
            // trap into an actionable error (a corrupt/edited .smeltim).
            guard importance.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw SmeltAffineQuantizerError.imatrixInvalidValues(tensor: entry.name)
            }
        }

        let (codes, codebook) = fp16Source.withUnsafeBufferPointer { buf in
            SmeltTurboQuantHQuantizer.quantize(
                weights: buf.baseAddress!,
                rows: rows, cols: cols, groupSize: 128,
                importance: importance
            )
        }

        // Discard withUnsafeBufferPointer's result: the closure's last expression
        // is memcpy, which returns the dest pointer; the copy is pure side effect.
        _ = codes.withUnsafeBufferPointer { cbuf in
            memcpy(
                output.advanced(by: Int(entry.offset)),
                cbuf.baseAddress!,
                codes.count
            )
        }
        if let codebookOffset = entry.codebookOffset {
            _ = codebook.withUnsafeBufferPointer { cbuf in
                memcpy(
                    output.advanced(by: Int(codebookOffset)),
                    cbuf.baseAddress!,
                    codebook.count * 2
                )
            }
        }
    }

    // MARK: - Helpers

    /// The authoritative "this weight resolves to affine_u4" decision — affine
    /// strategy, not TQH-routed, not an unquantized embedding, not excluded.
    /// `SmeltGPTQScope` reuses it so GPTQ scope tracks the real quant decision
    /// (the weight layout's dtype is assigned before TQH routing, so it can't).
    static func shouldQuantizeAffine(
        name: String,
        config: SmeltQuantizationConfig
    ) -> Bool {
        guard config.strategy == .affineU4 else { return false }
        if shouldUseTurboQuantH(name: name, config: config) { return false }
        if !config.quantizeEmbedding,
           name == SmeltCanonicalTensorNames.embedTokens
        {
            return false
        }
        return !isExcludedFromQuantization(name: name, patterns: config.excludePatterns)
    }

    private static func shouldUseTurboQuantH(
        name: String,
        config: SmeltQuantizationConfig
    ) -> Bool {
        // Exact-or-glob convention (matchesExactOrGlob): patterns without `*` match exactly. The
        // plain substring-glob is too loose — `embed_tokens` would also match `embed_tokens_per_layer`
        // and silently route the per-layer tensor through TQH, whose consumers don't yet dequant
        // turboQuantH entries, producing a package that reads 2-bit codes as fp16 garbage.
        matchesExactOrGlob(name: name, patterns: config.turboQuantHPatterns)
    }

    private static func align16(_ offset: UInt64) -> UInt64 {
        (offset + 15) & ~15
    }

    private static func align128(_ offset: UInt64) -> UInt64 {
        (offset + 127) & ~127
    }

    private static func convertF32toFP16(
        source: UnsafeRawPointer, dest: UnsafeMutableRawPointer, elementCount: Int
    ) {
        let dst = dest.bindMemory(to: UInt16.self, capacity: elementCount)
        for idx in 0..<elementCount {
            var fp32Val: Float = 0
            memcpy(&fp32Val, source.advanced(by: idx * 4), 4)
            dst[idx] = Float16(fp32Val).bitPattern
        }
    }

    private static func convertBF16toFP16(
        source: UnsafeRawPointer, dest: UnsafeMutableRawPointer, elementCount: Int
    ) {
        let dst = dest.bindMemory(to: UInt16.self, capacity: elementCount)
        for idx in 0..<elementCount {
            var bf16Bits: UInt16 = 0
            memcpy(&bf16Bits, source.advanced(by: idx * 2), 2)
            let fp32Bits = UInt32(bf16Bits) << 16
            let fp32 = Float(bitPattern: fp32Bits)
            dst[idx] = Float16(fp32).bitPattern
        }
    }

    /// Write a precomputed GPTQ affine_u4 block verbatim into `entry`'s layout
    /// regions (nibbles at `entry.offset`, scales/biases at their offsets),
    /// validating dimensions and array lengths first.
    private static func fillAffineFromBlock(
        block: SmeltAffineU4.Packed,
        entry: SmeltWeightEntry,
        output: UnsafeMutableRawPointer
    ) throws {
        let rows = entry.shape[0]
        let cols = entry.shape.count > 1 ? entry.shape[1] : 1
        guard block.rows == rows, block.cols == cols, block.groupSize == entry.groupSize else {
            throw SmeltAffineQuantizerError.gptqBlockShapeMismatch(
                tensor: entry.name,
                detail: "block \(block.rows)×\(block.cols) g\(block.groupSize) != weight "
                    + "\(rows)×\(cols) g\(entry.groupSize.map(String.init) ?? "nil")")
        }
        let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: block.groupSize)
        guard block.nibbles.count == Int(entry.sizeBytes),
              block.scales.count == rows * groups,
              block.biases.count == rows * groups
        else {
            throw SmeltAffineQuantizerError.gptqBlockShapeMismatch(
                tensor: entry.name,
                detail: "array lengths (nibbles \(block.nibbles.count)/\(entry.sizeBytes), "
                    + "scales \(block.scales.count), biases \(block.biases.count)) inconsistent "
                    + "with \(rows)×\(cols) g\(block.groupSize)")
        }
        let (packedDst, scalesDst, biasesDst) = affineU4Destinations(
            output: output, entry: entry, groupCount: groups)
        block.nibbles.withUnsafeBufferPointer { _ = memcpy(packedDst, $0.baseAddress!, $0.count) }
        block.scales.withUnsafeBufferPointer { scalesDst.update(from: $0.baseAddress!, count: $0.count) }
        block.biases.withUnsafeBufferPointer { biasesDst.update(from: $0.baseAddress!, count: $0.count) }
    }

    private static func entriesUsingExpectedLayout(
        _ computedEntries: [SmeltWeightEntry],
        expectedLayout: [SmeltWeightEntry]
    ) throws -> [SmeltWeightEntry] {
        let expectedByName = Dictionary(uniqueKeysWithValues: expectedLayout.map {
            ($0.name, $0)
        })
        return try computedEntries.map { computed in
            guard let expected = expectedByName[computed.name] else {
                throw SmeltAffineQuantizerError.expectedLayoutMissing(
                    tensor: computed.name
                )
            }
            try validateExpectedLayoutEntry(expected, matches: computed)
            return expected
        }
    }

    private static func validateExpectedLayoutEntry(
        _ expected: SmeltWeightEntry,
        matches computed: SmeltWeightEntry
    ) throws {
        func require(_ condition: Bool, _ detail: String) throws {
            guard condition else {
                throw SmeltAffineQuantizerError.expectedLayoutMismatch(
                    tensor: computed.name,
                    detail: detail
                )
            }
        }

        try require(expected.shape == computed.shape, "shape \(expected.shape) != \(computed.shape)")
        try require(expected.dtype == computed.dtype, "dtype \(expected.dtype) != \(computed.dtype)")
        try require(
            expected.groupSize == computed.groupSize,
            "group size \(String(describing: expected.groupSize))"
                + " != \(String(describing: computed.groupSize))"
        )
        try require(
            expected.sizeBytes == computed.sizeBytes,
            "data size \(expected.sizeBytes) != \(computed.sizeBytes)"
        )
        try require(
            expected.lutSizeBytes == computed.lutSizeBytes,
            "lut size \(String(describing: expected.lutSizeBytes))"
                + " != \(String(describing: computed.lutSizeBytes))"
        )
        try require(
            expected.packedRowStride == computed.packedRowStride,
            "packed row stride \(String(describing: expected.packedRowStride))"
                + " != \(String(describing: computed.packedRowStride))"
        )
        try require(
            expected.paddedCols == computed.paddedCols,
            "padded cols \(String(describing: expected.paddedCols))"
                + " != \(String(describing: computed.paddedCols))"
        )
        try require(
            expected.scalesSizeBytes == computed.scalesSizeBytes,
            "scales size \(String(describing: expected.scalesSizeBytes))"
                + " != \(String(describing: computed.scalesSizeBytes))"
        )
        try require(
            expected.biasesSizeBytes == computed.biasesSizeBytes,
            "biases size \(String(describing: expected.biasesSizeBytes))"
                + " != \(String(describing: computed.biasesSizeBytes))"
        )
        try require(
            expected.codebookSizeBytes == computed.codebookSizeBytes,
            "codebook size \(String(describing: expected.codebookSizeBytes))"
                + " != \(String(describing: computed.codebookSizeBytes))"
        )
    }

    private static func totalFileSize(_ entries: [SmeltWeightEntry]) -> UInt64 {
        entries.reduce(UInt64(0)) { total, entry in
            max(total, entryEndOffset(entry))
        }
    }

    private static func entryEndOffset(_ entry: SmeltWeightEntry) -> UInt64 {
        var end = entry.offset + entry.sizeBytes
        if let offset = entry.lutOffset, let size = entry.lutSizeBytes {
            end = max(end, offset + size)
        }
        if let offset = entry.scalesOffset, let size = entry.scalesSizeBytes {
            end = max(end, offset + size)
        }
        if let offset = entry.biasesOffset, let size = entry.biasesSizeBytes {
            end = max(end, offset + size)
        }
        if let offset = entry.codebookOffset, let size = entry.codebookSizeBytes {
            end = max(end, offset + size)
        }
        return end
    }

    /// The single home for the affine_u4 write layout — packed nibbles at
    /// `entry.offset`, fp16 scales/biases at their offsets — shared by
    /// `quantizeTensorAffine` and `fillAffineFromBlock` so they can't drift.
    /// Offsets are always set for an affine_u4 entry.
    private static func affineU4Destinations(
        output: UnsafeMutableRawPointer, entry: SmeltWeightEntry, groupCount: Int
    ) -> (
        packed: UnsafeMutablePointer<UInt8>,
        scales: UnsafeMutablePointer<UInt16>,
        biases: UnsafeMutablePointer<UInt16>
    ) {
        let rows = entry.shape[0]
        let packed = output.advanced(by: Int(entry.offset))
            .bindMemory(to: UInt8.self, capacity: Int(entry.sizeBytes))
        let scales = output.advanced(by: Int(entry.scalesOffset!))
            .bindMemory(to: UInt16.self, capacity: rows * groupCount)
        let biases = output.advanced(by: Int(entry.biasesOffset!))
            .bindMemory(to: UInt16.self, capacity: rows * groupCount)
        return (packed, scales, biases)
    }
}

// MARK: - Errors

public enum SmeltAffineQuantizerError: Error, CustomStringConvertible {
    case fileCreateFailed(String)
    case mmapFailed
    case unsupportedSourceDtype(tensor: String, dtype: String)
    case imatrixLengthMismatch(tensor: String, expected: Int, actual: Int)
    case imatrixInvalidValues(tensor: String)
    case gptqCoverageMissing(tensor: String)
    case gptqBlockOutOfScope(tensor: String)
    case gptqBlockShapeMismatch(tensor: String, detail: String)
    case preserveNativeRequiresBF16Source(tensor: String, dtype: String)
    case expectedLayoutMissing(tensor: String)
    case expectedLayoutMismatch(tensor: String, detail: String)

    public var description: String {
        switch self {
        case let .fileCreateFailed(path): return "Failed to create output file: \(path)"
        case .mmapFailed: return "Failed to mmap output file"
        case let .unsupportedSourceDtype(tensor, dtype):
            return "Unsupported source dtype '\(dtype)' for tensor '\(tensor)'; "
                + "quantization requires BF16/F32/F16 (or I32 index) source"
        case let .imatrixLengthMismatch(tensor, expected, actual):
            return "imatrix entry for '\(tensor)' has \(actual) lanes but the "
                + "tensor needs \(expected) (paddedToGroups) — wrong imatrix "
                + "for this model?"
        case let .imatrixInvalidValues(tensor):
            return "imatrix entry for '\(tensor)' has non-finite or negative "
                + "values; expected non-negative squared moments"
        case let .gptqCoverageMissing(tensor):
            return "GPTQ build is missing a block for in-scope projection '\(tensor)' "
                + "(no silent affine/GPTQ hybrid)"
        case let .gptqBlockOutOfScope(tensor):
            return "GPTQ block supplied for '\(tensor)', which is not an in-scope "
                + "affine_u4 projection weight"
        case let .gptqBlockShapeMismatch(tensor, detail):
            return "GPTQ block for '\(tensor)' does not match the weight: \(detail)"
        case let .preserveNativeRequiresBF16Source(tensor, dtype):
            return "preserve_native matched '\(tensor)' but its source dtype is \(dtype) — "
                + "preserve_native requires a BF16 source (fp16 is a no-op; "
                + "fp32-source is a deferred unit)"
        case let .expectedLayoutMissing(tensor):
            return "planned affine layout is missing tensor '\(tensor)'"
        case let .expectedLayoutMismatch(tensor, detail):
            return "planned affine layout for '\(tensor)' does not match writer "
                + "requirements: \(detail)"
        }
    }
}
