// SmeltQuantizer — GPU-accelerated weight quantization via Metal KMeans.
//
// Quantizes model weights to u4+LUT format using fused KMeans on Metal GPU.
// Full KMeans (not mini-batch), 100 iterations, 3 inits, empty cluster repair.
// Produces weights.bin via mmap'd streaming write — no monolithic RAM allocation.

import Foundation
import Metal
import SmeltSchema

/// GPU-accelerated weight quantizer.
public struct SmeltQuantizer {

    /// Quantize all weights and write to output file.
    ///
    /// - Parameters:
    ///   - tensors: Named tensor data from safetensors (host memory, may be mmap'd).
    ///   - adapter: Checkpoint adapter for name mapping.
    ///   - config: Quantization config from IR.
    ///   - modelConfig: Model config from IR (for shape validation).
    ///   - outputPath: Path to write weights.bin.
    ///   - device: Metal device.
    /// - Returns: Weight entries matching the output file layout.
    public static func quantize(
        tensors: [(runtimeName: String, data: UnsafeRawPointer, byteCount: Int,
                    shape: [Int], dtype: String)],
        config: SmeltQuantizationConfig,
        outputPath: String,
        device: MTLDevice,
        activationDtype: SmeltDType = .fp16
    ) throws -> [SmeltWeightEntry] {
        guard let queue = device.makeCommandQueue() else {
            throw SmeltQuantizerError.noCommandQueue
        }

        // Load Metal quantization kernels
        let shaderSource = try loadKMeansShaderSource()
        let library = try device.makeLibrary(source: shaderSource, options: nil)
        guard let quantizeFn = library.makeFunction(name: "kmeans_quantize"),
              let packFn = library.makeFunction(name: "kmeans_pack_u4")
        else {
            throw SmeltQuantizerError.kernelNotFound
        }
        let quantizePipeline = try device.makeComputePipelineState(function: quantizeFn)
        let packPipeline = try device.makeComputePipelineState(function: packFn)

        // Pre-compute layout (offsets for each weight in output file)
        var entries: [SmeltWeightEntry] = []
        var totalSize: UInt64 = 0

        for tensor in tensors {
            // Skip rank-0 or empty tensors (no entry created, not written)
            guard !tensor.shape.isEmpty, tensor.shape[0] > 0 else {
                // Append nil marker so indices stay aligned
                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName, offset: 0, sizeBytes: 0,
                    shape: [], dtype: .raw, groupSize: nil,
                    lutOffset: nil, lutSizeBytes: nil,
                    packedRowStride: nil, paddedCols: nil
                ))
                continue
            }

            let isQuantized = shouldQuantize(
                name: tensor.runtimeName, config: config
            )
            let rows = tensor.shape[0]
            let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1

            // Guard against Int overflow on size computations
            let (elemCount, overflow) = rows.multipliedReportingOverflow(by: cols)
            guard !overflow, elemCount <= Int.max / 4 else {
                // Tensor too large for safe arithmetic — skip
                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName, offset: 0, sizeBytes: 0,
                    shape: [], dtype: .raw, groupSize: nil,
                    lutOffset: nil, lutSizeBytes: nil,
                    packedRowStride: nil, paddedCols: nil
                ))
                continue
            }

            // Odd columns can't be u4-packed (2 elements per byte)
            let canQuantize = isQuantized && cols % 2 == 0

            // Index buffers (I32 source) take precedence over the
            // quantizer's u4 path even when their shape would otherwise
            // satisfy `canQuantize`, since quantizing a permutation
            // table as if it were FP weights is silent corruption.
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
                    groupSize: nil,
                    lutOffset: nil,
                    lutSizeBytes: nil,
                    packedRowStride: nil,
                    paddedCols: nil
                ))
            } else if SmeltWeightRole.preservesNativeBF16(
                name: tensor.runtimeName, activationDtype: activationDtype, config: config
            ) {
                // preserve_native: keep the projection at native bf16 (U2c).
                // PREFLIGHT — bf16 source only (single-sourced in SmeltWeightRole).
                if let reason = SmeltWeightRole.preserveNativeSourceRejection(
                    dtype: tensor.dtype, byteCount: tensor.byteCount, rows: rows, cols: cols) {
                    throw SmeltQuantizerError.preserveNativeRequiresBF16Source(
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
                    groupSize: nil,
                    lutOffset: nil,
                    lutSizeBytes: nil,
                    packedRowStride: nil,
                    paddedCols: nil
                ))
            } else if canQuantize {
                let groupSize = config.groupSize
                let paddedRows = ((rows + groupSize - 1) / groupSize) * groupSize
                let numGroups = paddedRows / groupSize
                let totalElements = paddedRows * cols
                let packedSize = UInt64(totalElements / 2)
                let lutSize = UInt64(numGroups * 16 * 2)  // 16 FP16 centroids per group

                let indicesOffset = align16(totalSize)
                let lutOffset = align16(indicesOffset + packedSize)
                totalSize = lutOffset + lutSize

                entries.append(SmeltWeightEntry(
                    name: tensor.runtimeName,
                    offset: indicesOffset,
                    sizeBytes: packedSize,
                    shape: tensor.shape,
                    dtype: .u4Lut,
                    groupSize: groupSize,
                    lutOffset: lutOffset,
                    lutSizeBytes: lutSize,
                    packedRowStride: cols / 2,
                    paddedCols: cols
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
                    groupSize: nil,
                    lutOffset: nil,
                    lutSizeBytes: nil,
                    packedRowStride: nil,
                    paddedCols: nil
                ))
            }
        }

        // Resume support: check for existing partial weights.bin + progress file
        let progressPath = outputPath + ".progress"
        var resumeFromIdx = 0
        let fm = FileManager.default

        let existingSize = (try? fm.attributesOfItem(atPath: outputPath))?[.size] as? UInt64
        if existingSize == totalSize,
           let progressData = try? String(contentsOfFile: progressPath, encoding: .utf8),
           let savedIdx = Int(progressData.trimmingCharacters(in: .whitespacesAndNewlines))
        {
            resumeFromIdx = savedIdx
            fputs("  Resuming from tensor \(resumeFromIdx + 1)/\(tensors.count)\n", stderr)
        }

        // Create or open output file for writing
        let flags: Int32 = resumeFromIdx > 0
            ? (O_RDWR)                       // resume: open existing
            : (O_RDWR | O_CREAT | O_TRUNC)   // fresh: truncate
        let fd = open(outputPath, flags, 0o644)
        guard fd >= 0 else {
            throw SmeltQuantizerError.fileCreateFailed(outputPath)
        }
        defer { close(fd) }

        if resumeFromIdx == 0 {
            ftruncate(fd, off_t(totalSize))
        }
        guard let mmapPtr = mmap(nil, Int(totalSize), PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, 0),
              mmapPtr != MAP_FAILED
        else {
            throw SmeltQuantizerError.mmapFailed
        }
        defer {
            msync(mmapPtr, Int(totalSize), MS_SYNC)
            munmap(mmapPtr, Int(totalSize))
        }

        // Quantize each tensor
        fputs("  Output file: \(Int(totalSize)) bytes mmap'd\n", stderr)
        for (idx, tensor) in tensors.enumerated() {
            let entry = entries[idx]
            guard entry.sizeBytes > 0 else { continue }  // skip placeholder entries

            // Skip already-quantized tensors on resume
            if idx < resumeFromIdx {
                fputs("  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName) → skipped (resume)\n", stderr)
                continue
            }
            let tensorStart = CFAbsoluteTimeGetCurrent()

            if entry.dtype == .u4Lut {
                try quantizeTensor(
                    source: tensor.data,
                    sourceBytes: tensor.byteCount,
                    sourceDtype: tensor.dtype,
                    shape: tensor.shape,
                    entry: entry,
                    config: config,
                    output: mmapPtr,
                    device: device,
                    queue: queue,
                    quantizePipeline: quantizePipeline,
                    packPipeline: packPipeline
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) → u4  \(String(format: "%.1f", elapsed))s\n",
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
                        + " \(tensor.shape) → int32  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else if entry.dtype == .bf16 {
                // preserved native bf16 — raw copy, no fp16 downcast.
                let dst = mmapPtr.advanced(by: Int(entry.offset))
                let rows = tensor.shape[0]
                let cols = tensor.shape.count > 1 ? tensor.shape[1] : 1
                let elementCount = rows * cols
                memcpy(dst, tensor.data, elementCount * 2)
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) → bf16 (preserved)  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            } else {
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
                    fputs("Warning: unsupported dtype '\(tensor.dtype)' for \(tensor.runtimeName)\n", stderr)
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - tensorStart
                fputs(
                    "  [\(idx + 1)/\(tensors.count)] \(tensor.runtimeName)"
                        + " \(tensor.shape) → fp16  \(String(format: "%.1f", elapsed))s\n",
                    stderr
                )
            }

            // Checkpoint progress after each tensor (enables resume on crash)
            try? "\(idx + 1)".write(toFile: progressPath, atomically: true, encoding: .utf8)
            msync(mmapPtr, Int(totalSize), MS_ASYNC)
        }

        // All done — remove progress file
        try? fm.removeItem(atPath: progressPath)

        return entries
    }

    // MARK: - Per-tensor quantization

    private static func quantizeTensor(
        source: UnsafeRawPointer,
        sourceBytes: Int,
        sourceDtype: String,
        shape: [Int],
        entry: SmeltWeightEntry,
        config: SmeltQuantizationConfig,
        output: UnsafeMutableRawPointer,
        device: MTLDevice,
        queue: MTLCommandQueue,
        quantizePipeline: MTLComputePipelineState,
        packPipeline: MTLComputePipelineState
    ) throws {
        let rows = shape[0]
        let cols = shape.count > 1 ? shape[1] : 1
        let groupSize = config.groupSize
        let paddedRows = ((rows + groupSize - 1) / groupSize) * groupSize
        let numGroups = paddedRows / groupSize
        let groupElements = UInt32(groupSize * cols)
        let totalElements = paddedRows * cols

        // Upload source to GPU (convert BF16→FP16 if needed)
        let fp16Bytes = totalElements * 2
        guard let sourceBuffer = device.makeBuffer(length: fp16Bytes, options: .storageModeShared)
        else {
            throw SmeltQuantizerError.bufferAllocationFailed
        }

        let srcDst = sourceBuffer.contents().bindMemory(to: UInt8.self, capacity: fp16Bytes)
        let elementCount = rows * cols
        switch sourceDtype {
        case "BF16":
            convertBF16toFP16(source: source, dest: srcDst, elementCount: elementCount)
        case "F32":
            convertF32toFP16(source: source, dest: srcDst, elementCount: elementCount)
        case "F16":
            memcpy(srcDst, source, elementCount * 2)
        default:
            throw SmeltQuantizerError.unsupportedDtype(sourceDtype)
        }
        // Zero-pad if rows were padded
        if paddedRows > rows {
            let padStart = rows * cols * 2
            memset(srcDst.advanced(by: padStart), 0, (paddedRows - rows) * cols * 2)
        }

        // Allocate GPU output buffers
        guard let assignBuffer = device.makeBuffer(
            length: totalElements, options: .storageModeShared
        ),
              let lutBuffer = device.makeBuffer(
                length: numGroups * 16 * 2, options: .storageModeShared
              )
        else {
            throw SmeltQuantizerError.bufferAllocationFailed
        }

        // Dispatch kmeans_quantize (batched to avoid GPU timeout)
        try dispatchKMeansBatched(
            sourceBuffer: sourceBuffer,
            assignBuffer: assignBuffer,
            lutBuffer: lutBuffer,
            numGroups: numGroups,
            groupElements: groupElements,
            seed: 42,
            queue: queue,
            pipeline: quantizePipeline
        )

        // Pack u4 indices
        let packedBuffer = try dispatchPackU4(
            assignBuffer: assignBuffer,
            totalElements: totalElements,
            device: device,
            queue: queue,
            pipeline: packPipeline
        )

        // Copy packed indices to mmap'd output
        let indicesDst = output.advanced(by: Int(entry.offset))
        memcpy(indicesDst, packedBuffer.contents(), totalElements / 2)

        // Copy LUT to mmap'd output
        if let lutOffset = entry.lutOffset {
            let lutDst = output.advanced(by: Int(lutOffset))
            memcpy(lutDst, lutBuffer.contents(), numGroups * 16 * 2)
        }
    }

    // MARK: - Batched dispatch

    /// Dispatch kmeans_quantize in chunks to avoid GPU execution timeout.
    /// Each batch gets its own command buffer. 64 groups per batch keeps
    /// each dispatch well within the ~2s GPU timeout.
    internal static func dispatchKMeansBatched(
        sourceBuffer: MTLBuffer,
        assignBuffer: MTLBuffer,
        lutBuffer: MTLBuffer,
        numGroups: Int,
        groupElements: UInt32,
        seed: UInt32,
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState
    ) throws {
        // Very wide groups, especially large embeddings, are more likely to trip
        // Metal internal errors if we queue multiple heavy kmeans batches in
        // flight. In that regime, prefer a conservative schedule over peak
        // build-time throughput.
        let useConservativeScheduling = groupElements >= 32_768
        let maxPerBatch = useConservativeScheduling ? 1 : 8
        let maxInFlight = useConservativeScheduling ? 1 : 4
        // kmeans.metal is tuned around a 256-threadgroup. Keep that cap for
        // very wide groups instead of scaling all the way to 1024 threads.
        let simdWidth = pipeline.threadExecutionWidth
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadgroupCap = useConservativeScheduling ? 256 : 1024
        let threadsPerGroup = max(
            simdWidth,
            min(maxThreads, threadgroupCap) / simdWidth * simdWidth
        )
        var batchStart = 0
        var inFlight: [(MTLCommandBuffer, Int)] = []

        // Sliding window: keep up to maxInFlight batches on the GPU
        while batchStart < numGroups {
            let batchCount = min(maxPerBatch, numGroups - batchStart)
            let elemBytes = Int(groupElements)

            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder()
            else {
                throw SmeltQuantizerError.noCommandQueue
            }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(sourceBuffer, offset: batchStart * elemBytes * 2, index: 0)
            enc.setBuffer(assignBuffer, offset: batchStart * elemBytes, index: 1)
            enc.setBuffer(lutBuffer, offset: batchStart * 16 * 2, index: 2)
            var ge = groupElements
            enc.setBytes(&ge, length: 4, index: 3)
            var sd = seed
            enc.setBytes(&sd, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: batchCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            inFlight.append((cmdBuf, batchStart))
            batchStart += batchCount

            // Drain oldest when window is full
            if inFlight.count >= maxInFlight {
                let (oldest, start) = inFlight.removeFirst()
                oldest.waitUntilCompleted()
                if let error = oldest.error {
                    throw SmeltQuantizerError.metalExecutionFailed(
                        "kmeans_quantize batch \(start): \(error.localizedDescription)"
                    )
                }
            }
        }

        // Drain remaining
        for (cmdBuf, start) in inFlight {
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw SmeltQuantizerError.metalExecutionFailed(
                    "kmeans_quantize batch \(start): \(error.localizedDescription)"
                )
            }
        }
    }

    /// Dispatch kmeans_pack_u4 and return the packed buffer.
    internal static func dispatchPackU4(
        assignBuffer: MTLBuffer,
        totalElements: Int,
        device: MTLDevice,
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState
    ) throws -> MTLBuffer {
        let packedBytes = totalElements / 2
        guard let packedBuffer = device.makeBuffer(
            length: packedBytes, options: .storageModeShared
        ) else {
            throw SmeltQuantizerError.bufferAllocationFailed
        }

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            throw SmeltQuantizerError.noCommandQueue
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(assignBuffer, offset: 0, index: 0)
        enc.setBuffer(packedBuffer, offset: 0, index: 1)
        var te = UInt32(totalElements)
        enc.setBytes(&te, length: 4, index: 2)
        enc.dispatchThreads(
            MTLSize(width: packedBytes, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(packedBytes, 1024), height: 1, depth: 1
            )
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw SmeltQuantizerError.metalExecutionFailed(
                "kmeans_pack_u4: \(error.localizedDescription)"
            )
        }
        return packedBuffer
    }

    // MARK: - Helpers

    private static func shouldQuantize(
        name: String,
        config: SmeltQuantizationConfig
    ) -> Bool {
        guard config.strategy == .lutU4 else { return false }
        if !config.quantizeEmbedding,
           name == SmeltCanonicalTensorNames.embedTokens
        {
            return false
        }
        return !isExcludedFromQuantization(name: name, patterns: config.excludePatterns)
    }

    private static func align16(_ offset: UInt64) -> UInt64 {
        (offset + 15) & ~15
    }

    private static func convertF32toFP16(
        source: UnsafeRawPointer, dest: UnsafeMutableRawPointer, elementCount: Int
    ) {
        // Use unaligned loads — safetensors data may not be 4-byte aligned
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
        // BF16→FP16 via FP32 intermediate. Unaligned reads for safetensors compat.
        for idx in 0..<elementCount {
            var bf16Bits: UInt16 = 0
            memcpy(&bf16Bits, source.advanced(by: idx * 2), 2)
            // BF16 → FP32: shift left 16 bits
            let fp32Bits = UInt32(bf16Bits) << 16
            let fp32 = Float(bitPattern: fp32Bits)
            // FP32 → FP16
            dst[idx] = Float16(fp32).bitPattern
        }
    }

    private static func loadKMeansShaderSource() throws -> String {
        // Look for kmeans.metal in known locations
        let candidates = [
            "Resources/Shaders/kmeans.metal",
            "../Resources/Shaders/kmeans.metal",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: path) {
                return try String(contentsOfFile: path, encoding: .utf8)
            }
        }
        // Try Bundle
        // Bundle.module not available in non-resource targets
        throw SmeltQuantizerError.shaderNotFound
    }
}

// MARK: - Errors

public enum SmeltQuantizerError: Error, CustomStringConvertible {
    case noCommandQueue
    case kernelNotFound
    case bufferAllocationFailed
    case fileCreateFailed(String)
    case mmapFailed
    case shaderNotFound
    case unsupportedDtype(String)
    case metalExecutionFailed(String)
    case preserveNativeRequiresBF16Source(tensor: String, dtype: String)

    public var description: String {
        switch self {
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .kernelNotFound: return "KMeans Metal kernel not found"
        case .bufferAllocationFailed: return "Failed to allocate Metal buffer"
        case let .fileCreateFailed(path): return "Failed to create output file: \(path)"
        case .mmapFailed: return "Failed to mmap output file"
        case .shaderNotFound: return "kmeans.metal shader source not found"
        case let .unsupportedDtype(dtype): return "Unsupported tensor dtype: \(dtype)"
        case let .metalExecutionFailed(msg): return "Metal execution failed: \(msg)"
        case let .preserveNativeRequiresBF16Source(tensor, dtype):
            return "preserve_native matched '\(tensor)' but its source dtype is \(dtype) — "
                + "preserve_native requires a BF16 source (fp16 is a no-op; "
                + "fp32-source is a deferred unit)"
        }
    }
}
