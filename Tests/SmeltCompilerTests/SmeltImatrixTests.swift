import Foundation
import Testing
@testable import SmeltCompiler

@Test func agentImatrixReadsWriterFormat() throws {
    // Mirror tools/compute_imatrix.py write_smeltim byte layout exactly, then
    // read it back. Includes a non-128-aligned cols (200 -> paddedToGroups 256)
    // to exercise the cols != paddedToGroups distinction in the format.
    func u32(_ v: UInt32) -> [UInt8] {
        var le = v.littleEndian
        return withUnsafeBytes(of: &le) { Array($0) }
    }
    func f32le(_ v: Float) -> [UInt8] {
        var le = v.bitPattern.littleEndian
        return withUnsafeBytes(of: &le) { Array($0) }
    }

    let tensors: [(name: String, cols: UInt32, vec: [Float])] = [
        ("layers_0_mlp_gate_proj_weight", 2560, (0 ..< 2560).map { Float($0) * 0.5 }),
        ("layers_0_mlp_down_proj_weight", 200, (0 ..< 256).map { Float($0) + 0.25 }),
    ]

    var bytes = Array("SMIM".utf8)
    bytes += u32(1)
    bytes += u32(UInt32(tensors.count))
    for t in tensors {
        let nb = Array(t.name.utf8)
        bytes += u32(UInt32(nb.count))
        bytes += nb
        bytes += u32(t.cols)
        bytes += u32(UInt32(t.vec.count))
        for f in t.vec { bytes += f32le(f) }
    }

    let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("imatrix-\(UUID().uuidString).smeltim")
    try Data(bytes).write(to: tmp)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let parsed = try SmeltImatrix.read(path: tmp.path)
    #expect(parsed.count == 2)
    for t in tensors {
        let got = try #require(parsed[t.name])
        #expect(got == t.vec, "values mismatch for \(t.name)")
    }
}

@Test func agentImatrixRejectsBadMagic() {
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("bad-\(UUID().uuidString).smeltim")
    try? Data(Array("NOPE".utf8) + [0, 0, 0, 0]).write(to: tmp)
    defer { try? FileManager.default.removeItem(at: tmp) }
    #expect(throws: SmeltImatrix.Error.self) {
        _ = try SmeltImatrix.read(path: tmp.path)
    }
}
