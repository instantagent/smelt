import Foundation

/// Reader for the `.smeltim` importance-matrix artifact produced by
/// tools/compute_imatrix.py. Binary layout (little-endian; the project is
/// arm64-only so the host matches the file byte order):
///   magic "SMIM", u32 version=1, u32 numTensors;
///   per tensor: u32 nameLen, name utf8, u32 cols, u32 paddedToGroups,
///   then paddedToGroups × f32 — the per-rotated-lane importance λ that
///   SmeltTurboQuantHQuantizer.quantize weights its codebook fit by.
public enum SmeltImatrix {
    public enum Error: Swift.Error, CustomStringConvertible {
        case badMagic
        case unsupportedVersion(UInt32)
        case truncated

        public var description: String {
            switch self {
            case .badMagic: return "imatrix: bad magic (expected SMIM)"
            case .unsupportedVersion(let v): return "imatrix: unsupported version \(v)"
            case .truncated: return "imatrix: truncated file"
            }
        }
    }

    /// Parse a `.smeltim` file into `{ manifest weight name -> importance }`.
    public static func read(path: String) throws -> [String: [Float]] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        var off = 0

        func u32() throws -> Int {
            guard off + 4 <= data.count else { throw Error.truncated }
            let v = data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: off, as: UInt32.self)
            }
            off += 4
            return Int(UInt32(littleEndian: v))
        }

        guard Array(data.prefix(4)) == Array("SMIM".utf8) else { throw Error.badMagic }
        off = 4
        let version = try u32()
        guard version == 1 else { throw Error.unsupportedVersion(UInt32(version)) }
        let count = try u32()

        var out: [String: [Float]] = [:]
        out.reserveCapacity(count)
        for _ in 0 ..< count {
            let nameLen = try u32()
            guard off + nameLen <= data.count else { throw Error.truncated }
            let name = String(decoding: data[(data.startIndex + off)..<(data.startIndex + off + nameLen)],
                              as: UTF8.self)
            off += nameLen
            _ = try u32()  // cols — informational; quantize() validates the length
            let pg = try u32()
            guard off + pg * 4 <= data.count else { throw Error.truncated }
            var vec = [Float](repeating: 0, count: pg)
            data.withUnsafeBytes { raw in
                _ = memcpy(&vec, raw.baseAddress!.advanced(by: off), pg * 4)
            }
            off += pg * 4
            out[name] = vec
        }
        return out
    }
}
