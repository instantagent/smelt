// SmeltTurboQuantHCodec — shared TurboQuant-H decode primitives
// reachable from both SmeltCompiler (writer/test) and SmeltRuntime
// (read path) without breaking the module dep direction
// (SmeltCompiler -> SmeltRuntime; both -> SmeltSchema).
//
// Reference algorithm: cactus-compute/cactus/blog/turboquant-h.md.
// The encode side (Hadamard rotation + k-means codebook fit + 2-bit
// codes) lives in SmeltCompiler/SmeltTurboQuantHQuantizer.swift;
// only the decode primitives that the runtime needs are hoisted
// here.

public enum SmeltTurboQuantHCodec {

    /// Matches the writer-side convention: the final group is
    /// padded with zeros if cols is not a multiple of groupSize,
    /// but it still counts.
    @inline(__always)
    public static func numGroups(cols: Int, groupSize: Int) -> Int {
        (cols + groupSize - 1) / groupSize
    }

    /// In-place fast Walsh-Hadamard transform on a power-of-two
    /// buffer. UN-normalized at this step — caller
    /// scales by 1/√n. The transform is self-inverse under the
    /// 1/√n normalization, so the same call serves both encode
    /// rotation and decode inverse.
    public static func fastWalshHadamard(
        buffer: inout [Float], size n: Int
    ) {
        precondition(n > 0 && (n & (n - 1)) == 0,
                     "FWHT size must be a power of two")
        var h = 1
        while h < n {
            var i = 0
            while i < n {
                for j in i ..< (i + h) {
                    let a = buffer[j]
                    let b = buffer[j + h]
                    buffer[j] = a + b
                    buffer[j + h] = a - b
                }
                i += h * 2
            }
            h *= 2
        }
    }

    /// Decode one TurboQuant-H row from raw pointer inputs into a
    /// caller-owned fp16 buffer slot. The caller passes:
    ///
    /// - `codes`: pointer at the START of this row's code bytes
    ///   (writer-side stride = `codesPerRow`).
    /// - `codebook`: pointer at the [num_groups, 4] fp16 codebook
    ///   (UInt16 bit patterns of fp16 values).
    /// - `output`: pointer at this row's UInt16 output slot
    ///   (caller materializes the bit patterns as fp16 by
    ///   reinterpretation; output[i] = Float16(...).bitPattern).
    /// - `cols`: logical column count (output written 0..<cols).
    /// - `groupSize`: must be a power of two; matches the writer's
    ///   group size (128 today).
    ///
    /// Algorithm: per group, read G codes from the packed stream,
    /// scatter through the codebook into a working buffer, apply
    /// the (self-inverse) Walsh-Hadamard butterfly, scale by 1/√G,
    /// trim to `cols`. Pad slots in the partial final group
    /// participate in the inverse (their codes encode the writer's
    /// zero pads and are load-bearing on the inverse) but are
    /// dropped on the output write.
    public static func dequantizeRowInto(
        codes: UnsafePointer<UInt8>,
        codebook: UnsafePointer<UInt16>,
        output: UnsafeMutablePointer<UInt16>,
        cols: Int,
        groupSize: Int = 128
    ) {
        precondition(groupSize > 0 && (groupSize & (groupSize - 1)) == 0,
                     "TurboQuant-H groupSize must be a power of two")
        let numGroups = Self.numGroups(cols: cols, groupSize: groupSize)
        let hadamardScale = Float(1.0 / Double(groupSize).squareRoot())
        var groupBuf = [Float](repeating: 0, count: groupSize)
        for gIdx in 0 ..< numGroups {
            for j in 0 ..< groupSize {
                let pos = gIdx * groupSize + j
                let byte = codes[pos / 4]
                let shift = (pos % 4) * 2
                let code = Int((byte >> shift) & 0x3)
                let cbBits = codebook[gIdx * 4 + code]
                groupBuf[j] = Float(Float16(bitPattern: cbBits))
            }
            fastWalshHadamard(buffer: &groupBuf, size: groupSize)
            for j in 0 ..< groupSize {
                let pos = gIdx * groupSize + j
                if pos < cols {
                    output[pos] = Float16(
                        groupBuf[j] * hadamardScale
                    ).bitPattern
                }
            }
        }
    }
}
