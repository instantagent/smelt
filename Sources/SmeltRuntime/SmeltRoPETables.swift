import Foundation
import Metal
import SmeltSchema

enum SmeltRoPETables {
    struct TablePair: Sendable {
        let cos: [Float16]
        let sin: [Float16]
    }

    static func resolvedPairs(from manifest: SmeltManifest) -> [SmeltRoPETablePairManifest] {
        if !manifest.slotLayout.ropeTablePairs.isEmpty {
            return manifest.slotLayout.ropeTablePairs
        }
        return [
            SmeltRoPETablePairManifest(
                theta: 10_000,
                dim: manifest.config.ropeDim,
                freqDim: nil,
                layout: "interleaved",
                cosSlot: manifest.slotLayout.ropeCosSlot,
                sinSlot: manifest.slotLayout.ropeSinSlot
            )
        ]
    }

    static func build(
        rowCount: Int,
        dim: Int,
        theta: Float,
        freqDim: Int?,
        scaling: SmeltRoPEScaling? = nil,
        layout: String = "interleaved"
    ) -> TablePair {
        let (cos, sin): ([Float16], [Float16]) = buildTables(
            rowCount: rowCount, dim: dim, theta: theta,
            freqDim: freqDim, scaling: scaling, layout: layout)
        return TablePair(cos: cos, sin: sin)
    }

    /// fp32 tables for the f32 trunk ABI (W0): same math, the Float values
    /// BEFORE the half cast — the f32 RoPE kernels take `const float*`.
    static func buildF32(
        rowCount: Int,
        dim: Int,
        theta: Float,
        freqDim: Int?,
        scaling: SmeltRoPEScaling? = nil,
        layout: String = "interleaved"
    ) -> (cos: [Float], sin: [Float]) {
        buildTables(rowCount: rowCount, dim: dim, theta: theta,
                    freqDim: freqDim, scaling: scaling, layout: layout)
    }

    /// One source of truth for the table math; T picks the storage cast.
    private static func buildTables<T: BinaryFloatingPoint>(
        rowCount: Int,
        dim: Int,
        theta: Float,
        freqDim: Int?,
        scaling: SmeltRoPEScaling?,
        layout: String,
        materializeFrequency: (Float) -> Float = { $0 }
    ) -> ([T], [T]) {
        precondition(rowCount >= 0)
        precondition(dim >= 0)
        precondition(dim % 2 == 0, "RoPE dim must be even")

        let elementCount = rowCount * dim
        var cosTable = Array(repeating: T(0), count: elementCount)
        var sinTable = Array(repeating: T(0), count: elementCount)

        guard dim > 0, rowCount > 0 else {
            return (cosTable, sinTable)
        }

        let baseDim = Float(max(freqDim ?? dim, 1))
        let halfDim = dim / 2

        for position in 0..<rowCount {
            let rowBase = position * dim
            for pair in 0..<halfDim {
                let invFreq = inverseFrequency(
                    pair: pair,
                    baseDim: baseDim,
                    theta: theta,
                    scaling: scaling
                )
                let angle = Float(position) * materializeFrequency(invFreq)
                let cosVal = T(Float(cos(Double(angle))))
                let sinVal = T(Float(sin(Double(angle))))
                let d0: Int
                let d1: Int
                if layout == "split_half" {
                    d0 = rowBase + pair
                    d1 = rowBase + pair + halfDim
                } else {
                    d0 = rowBase + pair * 2
                    d1 = d0 + 1
                }
                cosTable[d0] = cosVal
                cosTable[d1] = cosVal
                sinTable[d0] = sinVal
                sinTable[d1] = sinVal
            }
        }

        return (cosTable, sinTable)
    }

    private static func inverseFrequency(
        pair: Int,
        baseDim: Float,
        theta: Float,
        scaling: SmeltRoPEScaling?
    ) -> Float {
        let exponent = Float(2 * pair) / baseDim
        let invFreq = Float(pow(Double(theta), Double(-exponent)))
        guard let scaling else { return invFreq }

        switch scaling.type {
        case .llama3:
            let waveLength = (2.0 * Float.pi) / invFreq
            let lowFreqWaveLength = Float(scaling.originalMaxPositionEmbeddings)
                / scaling.lowFreqFactor
            let highFreqWaveLength = Float(scaling.originalMaxPositionEmbeddings)
                / scaling.highFreqFactor

            if waveLength > lowFreqWaveLength {
                return invFreq / scaling.factor
            }
            if waveLength < highFreqWaveLength {
                return invFreq
            }

            let smooth = (Float(scaling.originalMaxPositionEmbeddings) / waveLength
                - scaling.lowFreqFactor)
                / (scaling.highFreqFactor - scaling.lowFreqFactor)
            return (1.0 - smooth) * invFreq / scaling.factor + smooth * invFreq
        }
    }

    static func populate(
        cosBuffer: MTLBuffer,
        sinBuffer: MTLBuffer,
        rowCount: Int,
        dim: Int,
        theta: Float,
        freqDim: Int?,
        scaling: SmeltRoPEScaling? = nil,
        layout: String = "interleaved",
        dtype: SmeltDType = .fp16
    ) {
        precondition([.fp16, .bf16, .fp32].contains(dtype),
                     "RoPE tables are fp16, bf16, or fp32 (got \(dtype))")
        if dtype == .fp32 {
            let (cos, sin) = buildF32(
                rowCount: rowCount, dim: dim, theta: theta,
                freqDim: freqDim, scaling: scaling, layout: layout)
            let expectedBytes = cos.count * MemoryLayout<Float>.stride
            precondition(cosBuffer.length >= expectedBytes)
            precondition(sinBuffer.length >= expectedBytes)
            _ = cos.withUnsafeBytes { memcpy(cosBuffer.contents(), $0.baseAddress!, expectedBytes) }
            _ = sin.withUnsafeBytes { memcpy(sinBuffer.contents(), $0.baseAddress!, expectedBytes) }
            return
        }
        if dtype == .bf16 {
            func roundedBF16(_ value: Float) -> Float {
                let raw = value.bitPattern
                let rounded = raw &+ 0x7FFF &+ ((raw >> 16) & 1)
                return Float(bitPattern: rounded & 0xFFFF_0000)
            }
            let (cos, sin): ([Float], [Float]) = buildTables(
                rowCount: rowCount,
                dim: dim,
                theta: theta,
                freqDim: freqDim,
                scaling: scaling,
                layout: layout,
                materializeFrequency: roundedBF16
            )
            func bits(_ value: Float) -> UInt16 {
                let raw = value.bitPattern
                let rounded = raw &+ 0x7FFF &+ ((raw >> 16) & 1)
                return UInt16(truncatingIfNeeded: rounded >> 16)
            }
            let cosBits = cos.map(bits)
            let sinBits = sin.map(bits)
            let expectedBytes = cosBits.count * MemoryLayout<UInt16>.stride
            precondition(cosBuffer.length >= expectedBytes)
            precondition(sinBuffer.length >= expectedBytes)
            _ = cosBits.withUnsafeBytes {
                memcpy(cosBuffer.contents(), $0.baseAddress!, expectedBytes)
            }
            _ = sinBits.withUnsafeBytes {
                memcpy(sinBuffer.contents(), $0.baseAddress!, expectedBytes)
            }
            return
        }
        let (cos, sin): ([Float16], [Float16]) = buildTables(
            rowCount: rowCount,
            dim: dim,
            theta: theta,
            freqDim: freqDim,
            scaling: scaling,
            layout: layout,
            materializeFrequency: { Float(Float16($0)) }
        )
        let expectedBytes = cos.count * MemoryLayout<Float16>.stride
        precondition(cosBuffer.length >= expectedBytes)
        precondition(sinBuffer.length >= expectedBytes)

        _ = cos.withUnsafeBytes { src in
            memcpy(cosBuffer.contents(), src.baseAddress!, expectedBytes)
        }
        _ = sin.withUnsafeBytes { src in
            memcpy(sinBuffer.contents(), src.baseAddress!, expectedBytes)
        }
    }
}
