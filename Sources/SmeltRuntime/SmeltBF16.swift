import Foundation

/// Shared BF16 storage conversion used at runtime ABI boundaries.
enum SmeltBF16 {
    /// Narrows FP32 to BF16 with round-to-nearest, ties-to-even.
    static func encode(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    /// Widens BF16 storage to its exact FP32 representation.
    static func decode(_ value: UInt16) -> Float {
        Float(bitPattern: UInt32(value) << 16)
    }

    static func encode(_ values: [Float]) -> [UInt16] {
        values.map(encode)
    }

    static func decode(
        _ values: UnsafePointer<UInt16>,
        count: Int
    ) -> [Float] {
        (0..<count).map { decode(values[$0]) }
    }
}
