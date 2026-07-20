// SmeltWeightManifestLoader — Reads an existing weights.json manifest
// and converts it to [SmeltWeightEntry] for use in code generation.
//
// This bridges the gap between the Python quantizer's output format
// and Smelt's typed weight system. The Python quantizer uses packed
// column counts (cols/2 for u4) and different naming conventions.

import Foundation

/// Loads an existing weights.json and converts to SmeltWeightEntry array.
public struct SmeltWeightManifestLoader {

    /// Load weights.json and return typed entries.
    ///
    /// - Parameter path: Path to weights.json file.
    /// - Returns: Array of SmeltWeightEntry in the order they appear in the file.
    public static func load(from path: String) throws -> [SmeltWeightEntry] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SmeltCompilerError.noShaders("Invalid weights.json format")
        }

        // Preserve insertion order by reading the raw JSON
        // JSONSerialization doesn't guarantee order, so we sort by offset
        var entries: [SmeltWeightEntry] = []

        for (name, value) in dict {
            guard let meta = value as? [String: Any] else { continue }

            func intField(_ camel: String, _ snake: String? = nil) -> Int? {
                if let value = meta[camel] as? Int { return value }
                if let value = meta[camel] as? NSNumber { return value.intValue }
                if let snake {
                    if let value = meta[snake] as? Int { return value }
                    if let value = meta[snake] as? NSNumber { return value.intValue }
                }
                return nil
            }

            func uint64Field(_ camel: String, _ snake: String? = nil) -> UInt64? {
                intField(camel, snake).map(UInt64.init)
            }

            let offset = uint64Field("offset") ?? 0
            let sizeBytes = uint64Field("sizeBytes", "size_bytes") ?? 0
            let shape = meta["shape"] as? [Int] ?? []
            let isQuantized = meta["quantized"] as? Bool ?? false
            let groupSize = intField("groupSize", "group_size")
            let lutOffset = uint64Field("lutOffset", "lut_offset")
            let lutSizeBytes = uint64Field("lutSizeBytes", "lut_size_bytes")
            let explicitPackedRowStride = intField("packedRowStride", "packed_row_stride")
            let explicitPaddedCols = intField("paddedCols", "padded_cols")
            let scalesOffset = uint64Field("scalesOffset", "scales_offset")
            let scalesSizeBytes = uint64Field("scalesSizeBytes", "scales_size_bytes")
            let biasesOffset = uint64Field("biasesOffset", "biases_offset")
            let biasesSizeBytes = uint64Field("biasesSizeBytes", "biases_size_bytes")
            let codebookOffset = uint64Field("codebookOffset", "codebook_offset")
            let codebookSizeBytes = uint64Field("codebookSizeBytes", "codebook_size_bytes")

            // Explicit dtype wins when present. Legacy local manifests may
            // only have `quantized: true`, which historically meant u4_lut.
            let explicit = (meta["dtype"] as? String).flatMap(SmeltDType.init(rawValue:))
            let dtype: SmeltDType
            if let explicit {
                dtype = explicit
            } else {
                dtype = isQuantized ? .u4Lut : .fp16
            }

            // Compute packed row stride for quantized weights
            let packedRowStride: Int?
            let paddedCols: Int?
            if let explicitPackedRowStride, let explicitPaddedCols {
                packedRowStride = explicitPackedRowStride
                paddedCols = explicitPaddedCols
            } else if (isQuantized || dtype == .u4Lut || dtype == .affineU4
                        || dtype == .binary1 || dtype == .ternary2
                        || dtype == .turboQuantH),
                      shape.count >= 2 {
                paddedCols = shape[1]
                if dtype == .affineU4 {
                    packedRowStride = (shape[1] + 1) / 2
                } else if dtype == .binary1 {
                    packedRowStride = (shape[1] + 7) / 8
                } else if dtype == .ternary2 {
                    packedRowStride = (shape[1] + 3) / 4
                } else {
                    // Legacy LUT manifests store padded columns in shape[1].
                    packedRowStride = shape[1] / 2
                }
            } else {
                packedRowStride = nil
                paddedCols = nil
            }

            entries.append(SmeltWeightEntry(
                name: name,
                offset: offset,
                sizeBytes: sizeBytes,
                shape: shape,
                dtype: dtype,
                groupSize: groupSize,
                lutOffset: lutOffset,
                lutSizeBytes: lutSizeBytes,
                packedRowStride: packedRowStride,
                paddedCols: paddedCols,
                scalesOffset: scalesOffset,
                scalesSizeBytes: scalesSizeBytes,
                biasesOffset: biasesOffset,
                biasesSizeBytes: biasesSizeBytes,
                codebookOffset: codebookOffset,
                codebookSizeBytes: codebookSizeBytes
            ))
        }

        // Sort by offset for deterministic ordering
        entries.sort { $0.offset < $1.offset }

        return entries
    }

    /// Compute total file size from entries (max offset + size + LUT).
    public static func totalBytes(from entries: [SmeltWeightEntry]) -> UInt64 {
        var maxEnd: UInt64 = 0
        for entry in entries {
            let end = entry.offset + entry.sizeBytes
            if end > maxEnd { maxEnd = end }
            if let lutOff = entry.lutOffset, let lutSize = entry.lutSizeBytes {
                let lutEnd = lutOff + lutSize
                if lutEnd > maxEnd { maxEnd = lutEnd }
            }
            // Affine_u4 keeps per-group scales+biases at offsets
            // BEYOND the packed code region; TurboQuant-H keeps the
            // per-group codebook at a similar trailing offset.
            // Without these, the highest-offset tensor's auxiliary
            // bytes fall off the end of the runtime's mmap.
            if let scalesOff = entry.scalesOffset,
               let scalesSize = entry.scalesSizeBytes {
                let scalesEnd = scalesOff + scalesSize
                if scalesEnd > maxEnd { maxEnd = scalesEnd }
            }
            if let biasesOff = entry.biasesOffset,
               let biasesSize = entry.biasesSizeBytes {
                let biasesEnd = biasesOff + biasesSize
                if biasesEnd > maxEnd { maxEnd = biasesEnd }
            }
            if let codebookOff = entry.codebookOffset,
               let codebookSize = entry.codebookSizeBytes {
                let codebookEnd = codebookOff + codebookSize
                if codebookEnd > maxEnd { maxEnd = codebookEnd }
            }
        }
        return maxEnd
    }
}
