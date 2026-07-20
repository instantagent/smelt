// SmeltSignedQuantCodec — source-neutral binary/ternary packed-weight semantics.
//
// Import adapters normalize MLX/GGUF containers into this contract. Runtime
// kernels and CPU references consume only the semantic format, logical shape,
// group size, packed row stride, code bytes, and fp16 scale bytes.

import Foundation

public enum SmeltSignedQuantFormat: String, Codable, CaseIterable, Hashable, Sendable {
    /// One bit per value: 0 -> -scale, 1 -> +scale.
    case binary1
    /// Two bits per value: 0 -> -scale, 1 -> 0, 2 -> +scale. Code 3 is invalid.
    case ternary2

    public var bitsPerCode: Int {
        switch self {
        case .binary1: return 1
        case .ternary2: return 2
        }
    }

    public func semanticCode(_ packedCode: UInt8) throws -> Int8 {
        switch self {
        case .binary1:
            guard packedCode < 2 else {
                throw SmeltSignedQuantError.invalidCode(format: self, code: packedCode, index: nil)
            }
            return packedCode == 0 ? -1 : 1
        case .ternary2:
            guard packedCode < 3 else {
                throw SmeltSignedQuantError.invalidCode(format: self, code: packedCode, index: nil)
            }
            return Int8(packedCode) - 1
        }
    }
}

public enum SmeltSignedQuantError: Error, CustomStringConvertible, Equatable {
    case invalidGeometry(String)
    case bufferTooSmall(name: String, need: Int, got: Int)
    case invalidCode(format: SmeltSignedQuantFormat, code: UInt8, index: Int?)
    case invalidMLXScale(format: SmeltSignedQuantFormat, group: Int, bits: UInt16)
    case invalidMLXBias(
        format: SmeltSignedQuantFormat,
        group: Int,
        expectedBits: UInt16,
        actualBits: UInt16
    )

    public var description: String {
        switch self {
        case .invalidGeometry(let detail):
            return "invalid signed-quant geometry: \(detail)"
        case let .bufferTooSmall(name, need, got):
            return "signed-quant \(name) buffer needs \(need) bytes, got \(got)"
        case let .invalidCode(format, code, index):
            let suffix = index.map { " at logical index \($0)" } ?? ""
            return "invalid \(format.rawValue) code \(code)\(suffix)"
        case let .invalidMLXScale(format, group, bits):
            return "invalid \(format.rawValue) MLX scale at group \(group): "
                + String(format: "0x%04x", bits)
        case let .invalidMLXBias(format, group, expected, actual):
            return "invalid \(format.rawValue) MLX bias at group \(group): expected "
                + String(format: "0x%04x", expected) + ", got "
                + String(format: "0x%04x", actual)
        }
    }
}

public enum SmeltSignedQuantCodec {
    /// Packed bytes for `logicalCount` codes, rounded up to a whole byte.
    public static func packedByteCount(
        logicalCount: Int,
        format: SmeltSignedQuantFormat
    ) throws -> Int {
        guard logicalCount >= 0 else {
            throw SmeltSignedQuantError.invalidGeometry("negative logical count \(logicalCount)")
        }
        let (bits, overflow) = logicalCount.multipliedReportingOverflow(by: format.bitsPerCode)
        guard !overflow, bits <= Int.max - 7 else {
            throw SmeltSignedQuantError.invalidGeometry("packed byte count overflows")
        }
        return (bits + 7) / 8
    }

    /// Number of scale groups in one row. Padded columns must be whole groups so
    /// a kernel never reads a partial, differently-defined tail group.
    public static func groupsPerRow(paddedCols: Int, groupSize: Int) throws -> Int {
        guard paddedCols > 0 else {
            throw SmeltSignedQuantError.invalidGeometry("paddedCols must be positive")
        }
        guard groupSize > 0, paddedCols % groupSize == 0 else {
            throw SmeltSignedQuantError.invalidGeometry(
                "paddedCols \(paddedCols) is not a multiple of groupSize \(groupSize)"
            )
        }
        return paddedCols / groupSize
    }

    /// Extract one LSB-first code. This matches MLX's packed U32 words and the
    /// Q1_0/Q2_0 byte representation after container headers/scales are removed.
    public static func code(
        at logicalIndex: Int,
        from bytes: UnsafeRawPointer,
        byteCount: Int,
        format: SmeltSignedQuantFormat
    ) throws -> UInt8 {
        guard logicalIndex >= 0 else {
            throw SmeltSignedQuantError.invalidGeometry("negative logical index")
        }
        let (bitOffset, overflow) = logicalIndex.multipliedReportingOverflow(
            by: format.bitsPerCode)
        guard !overflow else {
            throw SmeltSignedQuantError.invalidGeometry("logical bit offset overflows")
        }
        let byteIndex = bitOffset >> 3
        guard byteIndex < byteCount else {
            throw SmeltSignedQuantError.bufferTooSmall(
                name: "code", need: byteIndex + 1, got: byteCount)
        }
        let shift = bitOffset & 7
        let mask = UInt8((1 << format.bitsPerCode) - 1)
        return (bytes.load(fromByteOffset: byteIndex, as: UInt8.self) >> shift) & mask
    }

    /// Validate the semantic code domain. Binary's entire bit domain is valid;
    /// ternary reserves packed value 3 and rejects it at import.
    public static func validateCodes(
        _ bytes: UnsafeRawPointer,
        byteCount: Int,
        logicalCount: Int,
        format: SmeltSignedQuantFormat
    ) throws {
        let need = try packedByteCount(logicalCount: logicalCount, format: format)
        guard byteCount >= need else {
            throw SmeltSignedQuantError.bufferTooSmall(name: "code", need: need, got: byteCount)
        }
        guard format == .ternary2 else { return }
        for i in 0..<logicalCount {
            let value = try code(at: i, from: bytes, byteCount: byteCount, format: format)
            if value == 3 {
                throw SmeltSignedQuantError.invalidCode(format: format, code: value, index: i)
            }
        }
    }

    /// Normalize MLX affine groups into Smelt's signed-code scale-only form.
    ///
    /// MLX binary uses q*scale+bias with bias=-scale/2, yielding {-d,+d}.
    /// MLX ternary uses bias=-scale, yielding {-d,0,+d}. Relationships are
    /// checked before the bias plane can be discarded. Exporters can round the
    /// scale and bias relationship independently to fp16; accept at most one
    /// fp16 ULP of drift, then keep the symmetric signed-code ABI.
    public static func normalizeMLXScales(
        format: SmeltSignedQuantFormat,
        scales: UnsafeRawPointer,
        biases: UnsafeRawPointer,
        groupCount: Int,
        into output: UnsafeMutablePointer<UInt16>
    ) throws {
        guard groupCount >= 0 else {
            throw SmeltSignedQuantError.invalidGeometry("negative group count")
        }
        for group in 0..<groupCount {
            let sourceScaleBits = scales.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            let biasBits = biases.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            output[group] = try canonicalMLXScaleBits(
                format: format,
                sourceScaleBits: sourceScaleBits,
                biasBits: biasBits,
                group: group
            )
        }
    }

    /// Validate and normalize one MLX affine group without allocating a scale
    /// plane. Streaming package writers use this while copying large tensors.
    public static func canonicalMLXScaleBits(
        format: SmeltSignedQuantFormat,
        sourceScaleBits: UInt16,
        biasBits: UInt16,
        group: Int
    ) throws -> UInt16 {
        let sourceScale = Float16(bitPattern: sourceScaleBits)
        guard sourceScale.isFinite, sourceScale >= 0 else {
            throw SmeltSignedQuantError.invalidMLXScale(
                format: format, group: group, bits: sourceScaleBits)
        }

        let canonicalScale: Float16
        switch format {
        case .binary1:
            canonicalScale = Float16(Float(sourceScale) * 0.5)
        case .ternary2:
            canonicalScale = sourceScale
        }
        let expectedBiasValue = Float16(-Float(canonicalScale))
        let expectedBias = expectedBiasValue.bitPattern
        let actualBias = Float16(bitPattern: biasBits)
        let independentRoundingIsEquivalent = actualBias.isFinite
            && actualBias <= 0
            && abs(Float(actualBias) - Float(expectedBiasValue))
                <= Float(canonicalScale.ulp)
        guard biasBits == expectedBias || independentRoundingIsEquivalent else {
            throw SmeltSignedQuantError.invalidMLXBias(
                format: format,
                group: group,
                expectedBits: expectedBias,
                actualBits: biasBits
            )
        }
        return canonicalScale.bitPattern
    }

    /// CPU oracle for one logical row. Codes may include group-aligned padded
    /// columns; only `cols` values are returned.
    public static func dequantizeRow(
        format: SmeltSignedQuantFormat,
        codes: UnsafeRawPointer,
        codeByteCount: Int,
        scales: UnsafeRawPointer,
        scaleByteCount: Int,
        cols: Int,
        paddedCols: Int,
        groupSize: Int,
        into output: UnsafeMutablePointer<Float>
    ) throws {
        guard cols > 0, cols <= paddedCols else {
            throw SmeltSignedQuantError.invalidGeometry(
                "cols \(cols) must be in 1...paddedCols \(paddedCols)"
            )
        }
        let groups = try groupsPerRow(paddedCols: paddedCols, groupSize: groupSize)
        let codeNeed = try packedByteCount(logicalCount: paddedCols, format: format)
        guard codeByteCount >= codeNeed else {
            throw SmeltSignedQuantError.bufferTooSmall(
                name: "code", need: codeNeed, got: codeByteCount)
        }
        let scaleNeed = groups * 2
        guard scaleByteCount >= scaleNeed else {
            throw SmeltSignedQuantError.bufferTooSmall(
                name: "scale", need: scaleNeed, got: scaleByteCount)
        }
        try validateCodes(
            codes, byteCount: codeByteCount, logicalCount: paddedCols, format: format)

        for col in 0..<cols {
            let packed = try code(
                at: col, from: codes, byteCount: codeByteCount, format: format)
            let semantic = try format.semanticCode(packed)
            let scaleBits = scales.loadUnaligned(
                fromByteOffset: (col / groupSize) * 2, as: UInt16.self)
            output[col] = Float(semantic) * Float(Float16(bitPattern: scaleBits))
        }
    }

    /// Runtime composition form of `dequantizeRow`: write the canonical fp16
    /// row ABI directly, avoiding a temporary fp32 row when one module exposes
    /// a signed embedding to another module.
    public static func dequantizeRowToFloat16Bits(
        format: SmeltSignedQuantFormat,
        codes: UnsafeRawPointer,
        codeByteCount: Int,
        scales: UnsafeRawPointer,
        scaleByteCount: Int,
        cols: Int,
        paddedCols: Int,
        groupSize: Int,
        into output: UnsafeMutablePointer<UInt16>
    ) throws {
        guard cols > 0, cols <= paddedCols else {
            throw SmeltSignedQuantError.invalidGeometry(
                "cols \(cols) must be in 1...paddedCols \(paddedCols)"
            )
        }
        let groups = try groupsPerRow(
            paddedCols: paddedCols,
            groupSize: groupSize
        )
        let codeNeed = try packedByteCount(
            logicalCount: paddedCols,
            format: format
        )
        guard codeByteCount >= codeNeed else {
            throw SmeltSignedQuantError.bufferTooSmall(
                name: "code",
                need: codeNeed,
                got: codeByteCount
            )
        }
        let scaleNeed = groups * MemoryLayout<UInt16>.stride
        guard scaleByteCount >= scaleNeed else {
            throw SmeltSignedQuantError.bufferTooSmall(
                name: "scale",
                need: scaleNeed,
                got: scaleByteCount
            )
        }
        try validateCodes(
            codes,
            byteCount: codeByteCount,
            logicalCount: paddedCols,
            format: format
        )

        for col in 0..<cols {
            let packed = try code(
                at: col,
                from: codes,
                byteCount: codeByteCount,
                format: format
            )
            let semantic = try format.semanticCode(packed)
            let scaleBits = scales.loadUnaligned(
                fromByteOffset: (col / groupSize) * MemoryLayout<UInt16>.stride,
                as: UInt16.self
            )
            output[col] = Float16(
                Float(semantic) * Float(Float16(bitPattern: scaleBits))
            ).bitPattern
        }
    }
}
