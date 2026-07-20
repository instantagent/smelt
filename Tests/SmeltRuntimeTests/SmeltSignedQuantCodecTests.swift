import Foundation
import Testing

@testable import SmeltRuntime

@Test func signedQuantCodecCoversSemanticDomainsAndGeometry() throws {
    #expect(try SmeltSignedQuantCodec.packedByteCount(
        logicalCount: 9, format: .binary1) == 2)
    #expect(try SmeltSignedQuantCodec.packedByteCount(
        logicalCount: 5, format: .ternary2) == 2)
    #expect(try SmeltSignedQuantCodec.groupsPerRow(paddedCols: 8, groupSize: 4) == 2)
    #expect(throws: SmeltSignedQuantError.self) {
        try SmeltSignedQuantCodec.groupsPerRow(paddedCols: 7, groupSize: 4)
    }

    // binary codes [-1,+1,-1,+1 | +1,-1,+1,-1], LSB first
    let binaryCodes: [UInt8] = [0b0101_1010]
    let binaryScales: [UInt16] = [Float16(0.5).bitPattern, Float16(2).bitPattern]
    var binaryOut = [Float](repeating: 0, count: 8)
    try binaryCodes.withUnsafeBytes { codeBytes in
        try binaryScales.withUnsafeBytes { scaleBytes in
            try binaryOut.withUnsafeMutableBufferPointer { out in
                try SmeltSignedQuantCodec.dequantizeRow(
                    format: .binary1,
                    codes: codeBytes.baseAddress!, codeByteCount: codeBytes.count,
                    scales: scaleBytes.baseAddress!, scaleByteCount: scaleBytes.count,
                    cols: 8, paddedCols: 8, groupSize: 4,
                    into: out.baseAddress!
                )
            }
        }
    }
    #expect(binaryOut == [-0.5, 0.5, -0.5, 0.5, 2, -2, 2, -2])

    var binaryFP16 = [UInt16](repeating: 0, count: 8)
    try binaryCodes.withUnsafeBytes { codeBytes in
        try binaryScales.withUnsafeBytes { scaleBytes in
            try binaryFP16.withUnsafeMutableBufferPointer { out in
                try SmeltSignedQuantCodec.dequantizeRowToFloat16Bits(
                    format: .binary1,
                    codes: codeBytes.baseAddress!,
                    codeByteCount: codeBytes.count,
                    scales: scaleBytes.baseAddress!,
                    scaleByteCount: scaleBytes.count,
                    cols: 8,
                    paddedCols: 8,
                    groupSize: 4,
                    into: out.baseAddress!
                )
            }
        }
    }
    #expect(binaryFP16.map { Float(Float16(bitPattern: $0)) } == binaryOut)

    // ternary codes [-1,0,+1,0 | +1,0,-1,+1], packed 4 codes/byte
    let ternaryCodes: [UInt8] = [0b01_10_01_00, 0b10_00_01_10]
    let ternaryScales: [UInt16] = [Float16(0.25).bitPattern, Float16(4).bitPattern]
    var ternaryOut = [Float](repeating: 0, count: 8)
    try ternaryCodes.withUnsafeBytes { codeBytes in
        try ternaryScales.withUnsafeBytes { scaleBytes in
            try ternaryOut.withUnsafeMutableBufferPointer { out in
                try SmeltSignedQuantCodec.dequantizeRow(
                    format: .ternary2,
                    codes: codeBytes.baseAddress!, codeByteCount: codeBytes.count,
                    scales: scaleBytes.baseAddress!, scaleByteCount: scaleBytes.count,
                    cols: 8, paddedCols: 8, groupSize: 4,
                    into: out.baseAddress!
                )
            }
        }
    }
    #expect(ternaryOut == [-0.25, 0, 0.25, 0, 4, 0, -4, 4])

    var ternaryFP16 = [UInt16](repeating: 0, count: 8)
    try ternaryCodes.withUnsafeBytes { codeBytes in
        try ternaryScales.withUnsafeBytes { scaleBytes in
            try ternaryFP16.withUnsafeMutableBufferPointer { out in
                try SmeltSignedQuantCodec.dequantizeRowToFloat16Bits(
                    format: .ternary2,
                    codes: codeBytes.baseAddress!,
                    codeByteCount: codeBytes.count,
                    scales: scaleBytes.baseAddress!,
                    scaleByteCount: scaleBytes.count,
                    cols: 8,
                    paddedCols: 8,
                    groupSize: 4,
                    into: out.baseAddress!
                )
            }
        }
    }
    #expect(ternaryFP16.map { Float(Float16(bitPattern: $0)) } == ternaryOut)
}

@Test func signedQuantCodecRejectsTernaryCodeThree() throws {
    let codes: [UInt8] = [0b00_01_10_11]
    do {
        try codes.withUnsafeBytes { bytes in
            try SmeltSignedQuantCodec.validateCodes(
                bytes.baseAddress!, byteCount: bytes.count,
                logicalCount: 4, format: .ternary2)
        }
        Issue.record("expected reserved ternary code to fail")
    } catch let error as SmeltSignedQuantError {
        #expect(error == .invalidCode(format: .ternary2, code: 3, index: 0))
    }
}

@Test func signedQuantCodecNormalizesMLXAffineTripletsExactly() throws {
    let binarySourceScales: [UInt16] = [
        Float16(0.5).bitPattern,
        Float16(0.125).bitPattern,
    ]
    let binaryBiases: [UInt16] = [
        Float16(-0.25).bitPattern,
        Float16(-0.0625).bitPattern,
    ]
    var binaryCanonical = [UInt16](repeating: 0, count: 2)
    try binarySourceScales.withUnsafeBytes { scales in
        try binaryBiases.withUnsafeBytes { biases in
            try binaryCanonical.withUnsafeMutableBufferPointer { output in
                try SmeltSignedQuantCodec.normalizeMLXScales(
                    format: .binary1,
                    scales: scales.baseAddress!, biases: biases.baseAddress!,
                    groupCount: 2, into: output.baseAddress!
                )
            }
        }
    }
    #expect(binaryCanonical == [Float16(0.25).bitPattern, Float16(0.0625).bitPattern])

    let ternaryScales: [UInt16] = [Float16(0.25).bitPattern]
    let ternaryBiases: [UInt16] = [Float16(-0.25).bitPattern]
    var ternaryCanonical = [UInt16](repeating: 0, count: 1)
    try ternaryScales.withUnsafeBytes { scales in
        try ternaryBiases.withUnsafeBytes { biases in
            try ternaryCanonical.withUnsafeMutableBufferPointer { output in
                try SmeltSignedQuantCodec.normalizeMLXScales(
                    format: .ternary2,
                    scales: scales.baseAddress!, biases: biases.baseAddress!,
                    groupCount: 1, into: output.baseAddress!
                )
            }
        }
    }
    #expect(ternaryCanonical == ternaryScales)

    let badBias: [UInt16] = [Float16(-0.5).bitPattern]
    #expect(throws: SmeltSignedQuantError.self) {
        try ternaryScales.withUnsafeBytes { scales in
            try badBias.withUnsafeBytes { biases in
                var output: UInt16 = 0
                try SmeltSignedQuantCodec.normalizeMLXScales(
                    format: .ternary2,
                    scales: scales.baseAddress!, biases: biases.baseAddress!,
                    groupCount: 1, into: &output
                )
            }
        }
    }
}

@Test func binaryMLXNormalizationAcceptsOnlyOneULPIndependentRounding() throws {
    // Stored scale = 2 * min-subnormal; the ideal half-scale is one
    // min-subnormal, while an independently rounded negative bias may be -0.
    #expect(
        try SmeltSignedQuantCodec.canonicalMLXScaleBits(
            format: .binary1,
            sourceScaleBits: UInt16(0x0002),
            biasBits: UInt16(0x8000),
            group: 0
        ) == UInt16(0x0001)
    )
    #expect(throws: SmeltSignedQuantError.self) {
        try SmeltSignedQuantCodec.canonicalMLXScaleBits(
            format: .binary1,
            sourceScaleBits: Float16(0.5).bitPattern,
            biasBits: Float16(-0.249).bitPattern,
            group: 0
        )
    }
}

@Test func ternaryMLXNormalizationAcceptsOnlyOneULPIndependentRounding() throws {
    // The released ternary embedding contains 40 groups at this subnormal
    // boundary: scale is two min-subnormals while bias independently rounded
    // to one negative min-subnormal. Preserve the symmetric ternary scale ABI.
    #expect(
        try SmeltSignedQuantCodec.canonicalMLXScaleBits(
            format: .ternary2,
            sourceScaleBits: UInt16(0x0002),
            biasBits: UInt16(0x8001),
            group: 0
        ) == UInt16(0x0002)
    )
    #expect(throws: SmeltSignedQuantError.self) {
        try SmeltSignedQuantCodec.canonicalMLXScaleBits(
            format: .ternary2,
            sourceScaleBits: UInt16(0x0002),
            biasBits: UInt16(0x8000),
            group: 0
        )
    }
}
