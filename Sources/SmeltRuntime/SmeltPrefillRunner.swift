// SmeltPrefillRunner — Runs CoreML batch prefill and hands off states to Metal.
//
// Responsibilities:
// 1. Load CoreML prefill model ONCE at init (cached for all requests)
// 2. Load cached prompt states from .npy files at init
// 3. Run batch prefill with token IDs + cached states
// 4. Copy CoreML outputs to Metal buffers at resolved slot indices
// 5. Copy recurrent and cache states into Metal buffers
//
// All tensor names come from SmeltPrefillInputContract — no hardcoded names.
// This runs once per request (~40ms), not per token.

import CoreML
import Foundation
import Metal
import SmeltSchema

/// Runs CoreML batch prefill and copies states to Metal buffers.
public final class SmeltPrefillRunner {

    /// Result of a prefill run.
    public struct PrefillResult {
        public let firstToken: Int32
        public let position: Int32
    }

    private let model: MLModel
    private let contract: SmeltPrefillInputContract
    private let maxBatchSize: Int
    private let handoff: SmeltHandoffTable
    private let buffers: [MTLBuffer]

    // Cached prompt state
    private let cachedStates: [String: MLMultiArray]
    private let ropeCos: NpyTensor?
    private let ropeSin: NpyTensor?
    private let promptLength: Int
    private let maxKV: Int
    private let ropeDim: Int

    /// Initialize with a compiled CoreML model and cached prompt states.
    ///
    /// - Parameters:
    ///   - prefillManifest: Prefill config from the package manifest.
    ///   - packagePath: Path to the .smeltpkg directory.
    ///   - cacheDir: Path to cached prompt states (.npy files + meta.json).
    ///   - buffers: SmeltRuntime's internal buffer array.
    public init(
        prefillManifest: SmeltPrefillManifest,
        packagePath: String,
        cacheDir: String,
        buffers: [MTLBuffer]
    ) throws {
        self.contract = prefillManifest.inputContract
        self.maxBatchSize = prefillManifest.maxBatchSize
        self.handoff = prefillManifest.handoff
        self.buffers = buffers

        // Load CoreML model ONCE (cached for all requests)
        let modelPath = "\(packagePath)/\(prefillManifest.modelPath)"
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        self.model = try MLModel(
            contentsOf: URL(fileURLWithPath: modelPath),
            configuration: mlConfig
        )

        // Load cache metadata
        let metaPath = "\(cacheDir)/meta.json"
        let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
        guard let meta = try JSONSerialization.jsonObject(with: metaData)
            as? [String: Any]
        else {
            throw SmeltPrefillError.missingOutput("invalid meta.json")
        }
        self.promptLength = meta["prompt_len"] as? Int ?? 0
        self.maxKV = meta["max_kv"] as? Int ?? 256
        self.ropeDim = meta["rope_dim"] as? Int ?? 64

        // Load cached state arrays
        var states: [String: MLMultiArray] = [:]
        // Build input names from contract patterns using family + index
        let inputPattern = prefillManifest.inputContract.stateInputPattern
        let outputPattern = prefillManifest.inputContract.stateOutputPattern
        for entry in handoff.entries {
            // Extract family and index from output tensor name using the output pattern
            let (family, index) = Self.extractFamilyIndex(
                tensorName: entry.tensorName, pattern: outputPattern
            )
            // Build the input name and cache filename using the input pattern
            let baseName = inputPattern
                .replacingOccurrences(of: "{family}", with: family)
                .replacingOccurrences(of: "{index}", with: index)
            let npyPath = "\(cacheDir)/\(baseName).npy"
            if FileManager.default.fileExists(atPath: npyPath) {
                let npy = try NpyLoader.load(path: npyPath)
                states[baseName] = try Self.npyToMLMultiArray(npy)
            }
        }
        self.cachedStates = states

        // Load RoPE tables
        let cosPath = "\(cacheDir)/rope_cos.npy"
        let sinPath = "\(cacheDir)/rope_sin.npy"
        self.ropeCos = FileManager.default.fileExists(atPath: cosPath)
            ? try NpyLoader.load(path: cosPath) : nil
        self.ropeSin = FileManager.default.fileExists(atPath: sinPath)
            ? try NpyLoader.load(path: sinPath) : nil

        // Load RoPE into Metal buffers once (not per request)
        // Metal expects FP16. Cache files may be FP32 — convert if needed.
        if let cos = ropeCos {
            let cosSlot = handoff.ropeCosSlot
            if cosSlot >= 0, cosSlot < buffers.count {
                Self.copyRoPEToMetal(npy: cos, dst: buffers[cosSlot])
            }
        }
        if let sin = ropeSin {
            let sinSlot = handoff.ropeSinSlot
            if sinSlot >= 0, sinSlot < buffers.count {
                Self.copyRoPEToMetal(npy: sin, dst: buffers[sinSlot])
            }
        }
    }

    // MARK: - Run prefill

    /// Run batch prefill and hand off states to Metal buffers.
    public func runPrefill(tokenIds: [Int32]) throws -> PrefillResult {
        guard !tokenIds.isEmpty else {
            throw SmeltPrefillError.missingOutput("tokenIds is empty")
        }
        guard tokenIds.count <= maxBatchSize else {
            throw SmeltPrefillError.missingOutput(
                "tokenIds.count \(tokenIds.count) exceeds maxBatchSize \(maxBatchSize)"
            )
        }
        guard promptLength + tokenIds.count < maxKV else {
            throw SmeltPrefillError.missingOutput(
                "position \(promptLength + tokenIds.count) would exceed maxKV \(maxKV)"
            )
        }
        // RoPE table needs promptLength + maxBatchSize rows (padded input)
        guard promptLength + maxBatchSize <= maxKV else {
            throw SmeltPrefillError.missingOutput(
                "RoPE slice \(promptLength)..<\(promptLength + maxBatchSize) exceeds maxKV \(maxKV)"
            )
        }

        let actualLen = tokenIds.count

        // Build inputs using contract (no hardcoded names)
        let inputs = try buildInputs(tokenIds: tokenIds)

        // Run CoreML prediction
        let output = try model.prediction(from: inputs)

        // Extract first token from logits
        guard let logits = output.featureValue(for: contract.logitsOutputName)?
            .multiArrayValue
        else {
            throw SmeltPrefillError.missingOutput(contract.logitsOutputName)
        }
        guard let vocabSize = logits.shape.last?.intValue, vocabSize > 0 else {
            throw SmeltPrefillError.missingOutput("logits has zero vocab dimension")
        }
        guard logits.count >= actualLen * vocabSize else {
            throw SmeltPrefillError.missingOutput("logits too small")
        }
        let firstToken = Self.argmax(logits, offset: (actualLen - 1) * vocabSize, count: vocabSize)

        // Hand off states to Metal buffers
        try handOffStates(output: output)

        return PrefillResult(
            firstToken: Int32(firstToken),
            position: Int32(promptLength + actualLen)
        )
    }

    // MARK: - Input construction

    private func buildInputs(tokenIds: [Int32]) throws -> MLDictionaryFeatureProvider {
        let actualLen = tokenIds.count

        // Pad token IDs to maxBatchSize
        let ids = try MLMultiArray(shape: [1, maxBatchSize as NSNumber], dataType: .int32)
        let idPtr = ids.dataPointer.bindMemory(to: Int32.self, capacity: maxBatchSize)
        for idx in 0..<maxBatchSize {
            idPtr[idx] = idx < actualLen ? tokenIds[idx] : 0
        }

        var inputs: [String: MLFeatureValue] = [
            contract.tokenInputName: .init(multiArray: ids),
            contract.seqLenInputName: .init(multiArray: try int32Scalar(Int32(actualLen))),
            contract.startPosInputName: .init(multiArray: try int32Scalar(Int32(promptLength))),
        ]

        // RoPE slices (if contract specifies them as inputs)
        if let cosName = contract.ropeCosInputName, let cos = ropeCos {
            let cosSlice = try sliceRoPE(cos, start: promptLength, count: maxBatchSize)
            inputs[cosName] = .init(multiArray: cosSlice)
        }
        if let sinName = contract.ropeSinInputName, let sin = ropeSin {
            let sinSlice = try sliceRoPE(sin, start: promptLength, count: maxBatchSize)
            inputs[sinName] = .init(multiArray: sinSlice)
        }

        // Attention mask (if contract specifies it)
        if let maskName = contract.maskInputName {
            let mask = try buildCausalMask(
                seqLen: maxBatchSize, maxKV: maxKV, actualLen: actualLen
            )
            inputs[maskName] = .init(multiArray: mask)
        }

        // Cached state inputs (using contract patterns)
        for (baseName, array) in cachedStates {
            inputs[baseName] = .init(multiArray: array)
        }

        return try MLDictionaryFeatureProvider(dictionary: inputs)
    }

    // MARK: - State handoff

    private func handOffStates(output: MLFeatureProvider) throws {
        for entry in handoff.entries {
            guard let featureValue = output.featureValue(for: entry.tensorName),
                  let array = featureValue.multiArrayValue
            else {
                throw SmeltPrefillError.missingOutput(entry.tensorName)
            }

            let slot = entry.slotIndex
            guard slot >= 0, slot < buffers.count else {
                throw SmeltPrefillError.invalidSlot(entry.tensorName, slot)
            }

            let copyCount = entry.expectedElements > 0 ? entry.expectedElements : array.count
            if entry.expectedElements > 0, array.count != entry.expectedElements {
                throw SmeltPrefillError.missingOutput(
                    "\(entry.tensorName): expected \(entry.expectedElements), got \(array.count)"
                )
            }

            let dst = buffers[slot]
            if entry.convertFP16toFP32 {
                Self.copyFP16toFP32(src: array, dst: dst, count: copyCount)
            } else {
                Self.copyToBuffer(src: array, dst: dst, count: copyCount)
            }
        }
    }

    // MARK: - Helpers

    private func sliceRoPE(_ npy: NpyTensor, start: Int, count: Int) throws -> MLMultiArray {
        // Determine element size from .npy dtype
        let bytesPerElement: Int
        let mlDtype: MLMultiArrayDataType
        switch npy.dtype {
        case "f2": bytesPerElement = 2; mlDtype = .float16
        case "f4": bytesPerElement = 4; mlDtype = .float32
        default: throw SmeltPrefillError.missingOutput("unsupported RoPE dtype: \(npy.dtype)")
        }

        // Bounds check: ensure we don't read past the table
        let maxRows = npy.shape.first ?? 0
        guard start + count <= maxRows else {
            throw SmeltPrefillError.missingOutput(
                "RoPE slice \(start)..<\(start + count) exceeds table size \(maxRows)"
            )
        }

        let result = try MLMultiArray(
            shape: [1, count as NSNumber, ropeDim as NSNumber],
            dataType: mlDtype
        )
        let rowBytes = ropeDim * bytesPerElement
        let srcBase = npy.data.bindMemory(to: UInt8.self, capacity: npy.byteCount)
        let dstBase = result.dataPointer.bindMemory(to: UInt8.self, capacity: count * rowBytes)
        for idx in 0..<count {
            memcpy(
                dstBase.advanced(by: idx * rowBytes),
                srcBase.advanced(by: (start + idx) * rowBytes),
                rowBytes
            )
        }
        return result
    }

    private func buildCausalMask(seqLen: Int, maxKV: Int, actualLen: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, seqLen as NSNumber, maxKV as NSNumber],
            dataType: .float16
        )
        let ptr = mask.dataPointer.bindMemory(to: Float16.self, capacity: seqLen * maxKV)
        // Fill with -10000 (masked)
        for idx in 0..<(seqLen * maxKV) { ptr[idx] = Float16(-10_000.0) }
        // Unmask visible positions
        for idx in 0..<seqLen {
            if idx < actualLen {
                for j in 0..<(promptLength + idx + 1) {
                    ptr[idx * maxKV + j] = 0
                }
            } else {
                ptr[idx * maxKV] = 0  // padding attends to position 0
            }
        }
        return mask
    }

    private func int32Scalar(_ value: Int32) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1], dataType: .int32)
        arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = value
        return arr
    }

    private static func copyToBuffer(src: MLMultiArray, dst: MTLBuffer, count: Int) {
        let byteCount = count * 2
        precondition(dst.length >= byteCount)
        if src.dataType == .float16 {
            dst.contents().copyMemory(
                from: src.dataPointer.bindMemory(to: UInt8.self, capacity: byteCount),
                byteCount: byteCount
            )
        } else {
            let srcPtr = src.dataPointer.bindMemory(to: Float.self, capacity: count)
            let dstPtr = dst.contents().bindMemory(to: Float16.self, capacity: count)
            for idx in 0..<count { dstPtr[idx] = Float16(srcPtr[idx]) }
        }
    }

    private static func copyFP16toFP32(src: MLMultiArray, dst: MTLBuffer, count: Int) {
        precondition(dst.length >= count * 4)
        let dstPtr = dst.contents().bindMemory(to: Float.self, capacity: count)
        if src.dataType == .float16 {
            let srcPtr = src.dataPointer.bindMemory(to: Float16.self, capacity: count)
            for idx in 0..<count { dstPtr[idx] = Float(srcPtr[idx]) }
        } else {
            dst.contents().copyMemory(
                from: src.dataPointer.bindMemory(to: Float.self, capacity: count),
                byteCount: count * 4
            )
        }
    }

    private static func argmax(_ array: MLMultiArray, offset: Int, count: Int) -> Int {
        var bestIdx = 0
        if array.dataType == .float16 {
            let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: offset + count)
            var bestVal = ptr[offset]
            for idx in 1..<count {
                if ptr[offset + idx] > bestVal { bestVal = ptr[offset + idx]; bestIdx = idx }
            }
        } else {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: offset + count)
            var bestVal = ptr[offset]
            for idx in 1..<count {
                if ptr[offset + idx] > bestVal { bestVal = ptr[offset + idx]; bestIdx = idx }
            }
        }
        return bestIdx
    }

    /// Extract family and index from a tensor name using a pattern like "{family}_{index}_out".
    /// Example: extractFamilyIndex("conv_state_5_out", pattern: "{family}_{index}_out")
    ///   → ("conv_state", "5")
    private static func extractFamilyIndex(
        tensorName: String, pattern: String
    ) -> (family: String, index: String) {
        // Split pattern on {family} and {index} to get prefix/middle/suffix
        // For pattern "{family}_{index}_out":
        //   parts before {family} = ""
        //   between {family} and {index} = "_"
        //   after {index} = "_out"
        let parts = pattern.components(separatedBy: "{family}")
        guard parts.count == 2 else { return (tensorName, "0") }
        let prefix = parts[0]
        let afterFamily = parts[1]

        let parts2 = afterFamily.components(separatedBy: "{index}")
        guard parts2.count == 2 else { return (tensorName, "0") }
        let separator = parts2[0]
        let suffix = parts2[1]

        // Strip prefix and suffix from tensor name
        var name = tensorName
        if !prefix.isEmpty, name.hasPrefix(prefix) { name = String(name.dropFirst(prefix.count)) }
        if !suffix.isEmpty, name.hasSuffix(suffix) { name = String(name.dropLast(suffix.count)) }

        // Split on separator to get family and index
        if let sepRange = name.range(of: separator, options: .backwards) {
            let family = String(name[..<sepRange.lowerBound])
            let index = String(name[sepRange.upperBound...])
            return (family, index)
        }

        return (name, "0")
    }

    private static func copyRoPEToMetal(npy: NpyTensor, dst: MTLBuffer) {
        if npy.dtype == "f2" {
            // FP16 → direct copy
            let copyBytes = min(npy.byteCount, dst.length)
            dst.contents().copyMemory(from: npy.data, byteCount: copyBytes)
        } else if npy.dtype == "f4" {
            // FP32 → FP16 conversion
            let srcPtr = npy.data.bindMemory(to: Float.self, capacity: npy.elementCount)
            let dstPtr = dst.contents().bindMemory(to: Float16.self, capacity: npy.elementCount)
            let count = min(npy.elementCount, dst.length / 2)
            for idx in 0..<count {
                dstPtr[idx] = Float16(srcPtr[idx])
            }
        }
    }

    private static func npyToMLMultiArray(_ npy: NpyTensor) throws -> MLMultiArray {
        let nsShape = npy.shape.map { $0 as NSNumber }
        let dataType: MLMultiArrayDataType
        let bytesPerElement: Int
        switch npy.dtype {
        case "f2": dataType = .float16; bytesPerElement = 2
        case "f4": dataType = .float32; bytesPerElement = 4
        case "i4": dataType = .int32; bytesPerElement = 4
        default: throw SmeltPrefillError.missingOutput("unsupported npy dtype: \(npy.dtype)")
        }
        let result = try MLMultiArray(shape: nsShape, dataType: dataType)
        let copyBytes = npy.elementCount * bytesPerElement
        result.dataPointer.copyMemory(from: npy.data, byteCount: copyBytes)
        return result
    }
}

// MARK: - Errors

public enum SmeltPrefillError: Error, CustomStringConvertible {
    case missingOutput(String)
    case invalidSlot(String, Int)

    public var description: String {
        switch self {
        case let .missingOutput(name):
            return "Prefill error: \(name)"
        case let .invalidSlot(name, slot):
            return "Invalid slot \(slot) for '\(name)'"
        }
    }
}
