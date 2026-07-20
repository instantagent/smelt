import Foundation
import Testing
@testable import SmeltCompiler

// The GPTQ build seam fingerprints injected blocks into weights provenance so a
// different set of blocks rebuilds rather than reusing stale weights. These check
// the fingerprint is order-independent over the dict and sensitive to every field.

private func block(nibbles: [UInt8], scales: [UInt16], biases: [UInt16]) -> SmeltAffineU4.Packed {
    // Shape fields are not part of the fingerprint (the byte lengths already encode
    // shape); fixed self-consistent values keep the constructor happy.
    SmeltAffineU4.Packed(
        nibbles: nibbles, scales: scales, biases: biases,
        rows: 1, cols: nibbles.count * 2, groupSize: max(nibbles.count * 2, 1))
}

@Test func gptqFingerprintIsDeterministicAndOrderIndependent() {
    let a = block(nibbles: [0x12, 0x34], scales: [0x3C00, 0x3800], biases: [0, 1])
    let b = block(nibbles: [0xAB], scales: [0x4000], biases: [2])
    // Same blocks inserted in different dict-construction orders hash identically.
    let f1 = SmeltCompiler.gptqBlocksFingerprint(["w_q": a, "w_k": b])
    let f2 = SmeltCompiler.gptqBlocksFingerprint(["w_k": b, "w_q": a])
    #expect(f1 == f2)
    #expect(f1.count == 64)  // hex sha256
}

@Test func gptqFingerprintIsSensitiveToEveryField() {
    let base = ["w": block(nibbles: [1, 2], scales: [0x3C00], biases: [0])]
    let ref = SmeltCompiler.gptqBlocksFingerprint(base)

    #expect(SmeltCompiler.gptqBlocksFingerprint(["w": block(nibbles: [1, 3], scales: [0x3C00], biases: [0])]) != ref)
    #expect(SmeltCompiler.gptqBlocksFingerprint(["w": block(nibbles: [1, 2], scales: [0x3800], biases: [0])]) != ref)
    #expect(SmeltCompiler.gptqBlocksFingerprint(["w": block(nibbles: [1, 2], scales: [0x3C00], biases: [1])]) != ref)
    // A different weight name must change the fingerprint too.
    #expect(SmeltCompiler.gptqBlocksFingerprint(["x": block(nibbles: [1, 2], scales: [0x3C00], biases: [0])]) != ref)
}

@Test func gptqFingerprintIsInjectiveAcrossFieldBoundaries() {
    // Without length prefixes these collide: name "A" (0x41) ‖ nibbles [0x10] streams
    // the same bytes as name "" ‖ nibbles [0x41, 0x10] when scales/biases match.
    let a = ["A": block(nibbles: [0x10], scales: [0x3C00], biases: [0])]
    let b = ["": block(nibbles: [0x41, 0x10], scales: [0x3C00], biases: [0])]
    #expect(SmeltCompiler.gptqBlocksFingerprint(a) != SmeltCompiler.gptqBlocksFingerprint(b))
}
