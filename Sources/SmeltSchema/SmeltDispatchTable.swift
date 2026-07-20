// SmeltDispatchTable — Binary dispatch table format for the decode hot path.
//
// Fixed-width POD records. mmap'd at runtime, iterated directly.
// No parsing, no allocations, no heap. The compiler writes these
// as raw bytes to dispatches.bin; the runtime casts the mmap'd
// memory to UnsafeBufferPointer<SmeltDispatchRecord>.
//
// Record layout is stable within a package — the compiler and runtime
// agree on the format. Different model specs produce different tables.

// MARK: - Record types

/// Maximum buffer bindings per dispatch. Metal supports up to 31.
/// Raised from 8 to 16 to support fused prefill kernels.
public let agentMaxBuffersPerDispatch = 16

/// Maximum constant bindings per dispatch.
public let agentMaxConstantsPerDispatch = 8

/// One buffer binding in a dispatch record.
public struct SmeltBufferRecord {
    /// Buffer slot index. >= 0 for literal slots.
    /// -1 = resolve to `cur` (double-buffer A/B).
    /// -2 = resolve to `alt` (double-buffer B/A).
    public var slot: Int16

    /// Metal argument buffer index.
    public var bindingIndex: UInt8

    /// Offset kind:
    /// 0 = literal byte offset
    /// 1 = position * stride (decode)
    /// 2 = startPos * low32(offset) + high32(offset) (prefill RoPE/KV)
    /// 3 = (seqLen - 1) * offset (prefill LM head)
    /// 4 = floor(seqLen / high32(offset)) * low32(offset) (prefill tail offset)
    public var offsetKind: UInt8

    /// Byte offset into the buffer (literal), or stride for dynamic offsets.
    public var offset: UInt64

    /// Sentinel slot values for double-buffer resolution.
    public static let slotCur: Int16 = -1
    public static let slotAlt: Int16 = -2
}

/// One constant binding in a dispatch record.
public struct SmeltConstantRecord {
    /// Constant kind:
    /// 0 = literal UInt32 (value is the bits)
    /// 1 = literal Float32 (value is the IEEE 754 bits)
    /// 2 = UInt32(position) — resolved at runtime
    /// 3 = UInt32(position + 1) — resolved at runtime
    public var kind: UInt8

    /// Metal argument buffer index.
    public var bindingIndex: UInt8

    /// Padding for alignment.
    public var pad: UInt16

    /// Literal value bits (ignored for kind 2/3).
    public var value: UInt32

    public static let kindLiteralU32: UInt8 = 0
    public static let kindLiteralF32: UInt8 = 1
    public static let kindPosition: UInt8 = 2
    public static let kindPositionPlus1: UInt8 = 3
    /// Prefill: resolved to UInt32(seqLen) at runtime.
    public static let kindSeqLen: UInt8 = 4
    /// Prefill: resolved to UInt32(startPos) at runtime.
    public static let kindStartPos: UInt8 = 5
    /// Prefill: resolved to UInt32(startPos) + value at runtime.
    public static let kindStartPosPlusLiteral: UInt8 = 6
    /// Prefill: resolved to UInt32(seqLen) * value at runtime.
    public static let kindSeqLenMulLiteral: UInt8 = 7
    /// Prefill: resolved to UInt32(seqLen % value) at runtime.
    public static let kindSeqLenModLiteral: UInt8 = 8
    /// Prefill: resolved to UInt32(seqLen % value); dispatch is skipped when zero.
    public static let kindSeqLenModLiteralSkipIfZero: UInt8 = 9
    /// Decode: dispatch is skipped unless UInt32(position + 1) < value.
    public static let kindPositionPlus1LessThanLiteralSkipIfFalse: UInt8 = 10
    /// Decode: dispatch is skipped unless UInt32(position + 1) >= value.
    public static let kindPositionPlus1GreaterEqualLiteralSkipIfFalse: UInt8 = 11
    /// Runtime physical KV cache sequence capacity.
    public static let kindCacheSeqCapacity: UInt8 = 12
    /// Prefill: dispatch is skipped unless UInt32(seqLen) < value.
    public static let kindSeqLenLessThanLiteralSkipIfFalse: UInt8 = 13
}

/// One entry in the dispatch table.
public struct SmeltDispatchRecord {
    /// Operation kind: 0 = Metal dispatch, 1 = swap double-buffer.
    public var opKind: UInt8

    /// Pipeline index (into the pipelines array). Ignored for swap ops.
    public var pipeline: UInt16

    /// Packed dispatch metadata. Low 2 bits = style (0 threadgroups, 1 threads).
    /// Remaining bits pack dynamic prefill grid kinds.
    public var style: UInt8

    /// Number of valid entries in the buffers array (0-16).
    public var bufferCount: UInt8

    /// Number of valid entries in the constants array (0-8).
    public var constantCount: UInt8

    /// Packed prefill minSeqLen guard (0 = always execute).
    public var pad: UInt16

    /// Grid dimensions (width, height, depth).
    public var gridW: UInt32
    public var gridH: UInt32
    public var gridD: UInt32

    /// Threadgroup dimensions (width, height, depth).
    public var tgW: UInt32
    public var tgH: UInt32
    public var tgD: UInt32

    /// Buffer bindings (fixed-size array, up to 16).
    public var buf0: SmeltBufferRecord
    public var buf1: SmeltBufferRecord
    public var buf2: SmeltBufferRecord
    public var buf3: SmeltBufferRecord
    public var buf4: SmeltBufferRecord
    public var buf5: SmeltBufferRecord
    public var buf6: SmeltBufferRecord
    public var buf7: SmeltBufferRecord
    public var buf8: SmeltBufferRecord
    public var buf9: SmeltBufferRecord
    public var buf10: SmeltBufferRecord
    public var buf11: SmeltBufferRecord
    public var buf12: SmeltBufferRecord
    public var buf13: SmeltBufferRecord
    public var buf14: SmeltBufferRecord
    public var buf15: SmeltBufferRecord

    /// Constant bindings (fixed-size array).
    public var con0: SmeltConstantRecord
    public var con1: SmeltConstantRecord
    public var con2: SmeltConstantRecord
    public var con3: SmeltConstantRecord
    public var con4: SmeltConstantRecord
    public var con5: SmeltConstantRecord
    public var con6: SmeltConstantRecord
    public var con7: SmeltConstantRecord

    /// Operation kind constants.
    public static let opDispatch: UInt8 = 0
    public static let opSwap: UInt8 = 1

    /// Dispatch style constants.
    public static let styleThreadgroups: UInt8 = 0
    public static let styleThreads: UInt8 = 1

    /// Grid dimension kind constants.
    public static let gridLiteral: UInt8 = 0
    public static let gridSeqLen: UInt8 = 1
    public static let gridSeqLenMulLiteral: UInt8 = 2
    /// Kind 3 uses the high bit of the literal to distinguish ceil vs floor:
    /// low31 = divisor, high bit clear => ceil(seqLen / divisor),
    /// high bit set => floor(seqLen / divisor).
    public static let gridSeqLenCeilDivLiteral: UInt8 = 3
}

// MARK: - Convenience

extension SmeltDispatchRecord {
    private static let styleMask: UInt8 = 0x03
    private static let gridWShift: UInt8 = 2
    private static let gridHShift: UInt8 = 4
    private static let gridDShift: UInt8 = 6

    /// Low 2 bits of the packed style byte.
    public var dispatchStyle: UInt8 {
        get { style & Self.styleMask }
        set { style = (style & ~Self.styleMask) | (newValue & Self.styleMask) }
    }

    public var gridWKind: UInt8 {
        get { (style >> Self.gridWShift) & 0x03 }
        set {
            style = (style & ~(0x03 << Self.gridWShift))
                | ((newValue & 0x03) << Self.gridWShift)
        }
    }

    public var gridHKind: UInt8 {
        get { (style >> Self.gridHShift) & 0x03 }
        set {
            style = (style & ~(0x03 << Self.gridHShift))
                | ((newValue & 0x03) << Self.gridHShift)
        }
    }

    public var gridDKind: UInt8 {
        get { (style >> Self.gridDShift) & 0x03 }
        set {
            style = (style & ~(0x03 << Self.gridDShift))
                | ((newValue & 0x03) << Self.gridDShift)
        }
    }

    /// Skip this dispatch when seqLen < minSeqLen (0 = always execute).
    public var minSeqLen: UInt16 {
        get { pad }
        set { pad = newValue }
    }

    /// Create a swap operation (no dispatch, just swap cur/alt).
    public static func swap() -> SmeltDispatchRecord {
        var rec = SmeltDispatchRecord.empty()
        rec.opKind = SmeltDispatchRecord.opSwap
        return rec
    }

    /// Create a zero-initialized record.
    public static func empty() -> SmeltDispatchRecord {
        SmeltDispatchRecord(
            opKind: 0, pipeline: 0, style: 0,
            bufferCount: 0, constantCount: 0, pad: 0,
            gridW: 0, gridH: 0, gridD: 0,
            tgW: 0, tgH: 0, tgD: 0,
            buf0: .empty(), buf1: .empty(), buf2: .empty(), buf3: .empty(),
            buf4: .empty(), buf5: .empty(), buf6: .empty(), buf7: .empty(),
            buf8: .empty(), buf9: .empty(), buf10: .empty(), buf11: .empty(),
            buf12: .empty(), buf13: .empty(), buf14: .empty(), buf15: .empty(),
            con0: .empty(), con1: .empty(), con2: .empty(), con3: .empty(),
            con4: .empty(), con5: .empty(), con6: .empty(), con7: .empty()
        )
    }
}

extension SmeltBufferRecord {
    public static func empty() -> SmeltBufferRecord {
        SmeltBufferRecord(slot: 0, bindingIndex: 0, offsetKind: 0, offset: 0)
    }
}

extension SmeltConstantRecord {
    public static func empty() -> SmeltConstantRecord {
        SmeltConstantRecord(kind: 0, bindingIndex: 0, pad: 0, value: 0)
    }
}

// MARK: - Record field accessors

/// Read a buffer record by index from a dispatch record.
public func getBuffer(_ rec: SmeltDispatchRecord, index: Int) -> SmeltBufferRecord {
    switch index {
    case 0: return rec.buf0
    case 1: return rec.buf1
    case 2: return rec.buf2
    case 3: return rec.buf3
    case 4: return rec.buf4
    case 5: return rec.buf5
    case 6: return rec.buf6
    case 7: return rec.buf7
    case 8: return rec.buf8
    case 9: return rec.buf9
    case 10: return rec.buf10
    case 11: return rec.buf11
    case 12: return rec.buf12
    case 13: return rec.buf13
    case 14: return rec.buf14
    case 15: return rec.buf15
    default: return SmeltBufferRecord.empty()
    }
}

/// Read a constant record by index from a dispatch record.
public func getConstant(_ rec: SmeltDispatchRecord, index: Int) -> SmeltConstantRecord {
    switch index {
    case 0: return rec.con0
    case 1: return rec.con1
    case 2: return rec.con2
    case 3: return rec.con3
    case 4: return rec.con4
    case 5: return rec.con5
    case 6: return rec.con6
    case 7: return rec.con7
    default: return SmeltConstantRecord.empty()
    }
}
