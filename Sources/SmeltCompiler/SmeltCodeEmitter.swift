// SmeltCodeEmitter — Generates Swift dispatch code from structured descriptions.
//
// The emitter is the bridge between the IR/buffer plan/weight layout and the
// generated Swift source file. It outputs [String] of Swift source lines that
// compile into the hot-path decode function.
//
// Every emitted line uses only integer indices and literal constants:
// - Pipeline indices from SmeltPipeline.rawValue
// - Buffer slot indices from SmeltFixedSlot.rawValue or SmeltBufferPlan dynamic bases
// - Weight byte offsets from SmeltWeightLayout
// - Constant values as literal UInt32/Float
//
// No strings, no dictionaries, no ARC in the generated code.

// MARK: - Weight lookup

/// Shared weight entry lookup — throws instead of crashing.
/// Used by plugins and top-level emitter for safe weight resolution.
public func requireWeight(
    _ name: String,
    from entries: [String: SmeltWeightEntry]
) throws -> SmeltWeightEntry {
    guard let entry = entries[name] else {
        throw SmeltEmitError.missingWeight(name: name)
    }
    return entry
}

// MARK: - Dispatch geometry

/// How a kernel is dispatched on the GPU.
public enum SmeltDispatchStyle: Sendable {
    /// dispatchThreads(MTLSize, threadsPerThreadgroup: MTLSize)
    case threads(width: Int, height: Int, depth: Int, tgWidth: Int, tgHeight: Int, tgDepth: Int)
    /// dispatchThreadgroups(MTLSize, threadsPerThreadgroup: MTLSize)
    case threadgroups(width: Int, height: Int, depth: Int, tgWidth: Int, tgHeight: Int, tgDepth: Int)
}

public enum SmeltDynamicGridDimension: Sendable, Equatable {
    case seqLen
    case seqLenMul(Int)
    case seqLenCeilDiv(Int)
    case seqLenFloorDiv(Int)
}

// MARK: - Slot reference

/// A buffer slot reference — either a compile-time integer or a runtime variable name.
public enum SmeltSlotRef: Sendable, Equatable {
    /// Fixed integer slot index (e.g. SmeltFixedSlot.qkvBuf.rawValue → 2).
    case fixed(Int)
    /// Runtime variable name (e.g. "cur" for double-buffer tracking).
    case variable(String)

    /// The Swift expression to emit inside `b[...]`.
    var expression: String {
        switch self {
        case let .fixed(idx): return "\(idx)"
        case let .variable(name): return name
        }
    }
}

// MARK: - Buffer binding

/// One buffer binding in a kernel dispatch.
public struct SmeltBufferBinding: Sendable {
    /// Buffer slot reference (fixed integer or runtime variable).
    public let slot: SmeltSlotRef
    /// Byte offset expression. If nil, uses literal `byteOffset`.
    /// For runtime offsets (e.g. RoPE position-dependent), set this to the expression string.
    public let offsetExpression: String?
    /// Literal byte offset (used when offsetExpression is nil).
    public let byteOffset: UInt64
    /// Offset kind for the binary dispatch record.
    /// 0 = literal, 1 = position*stride, 2 = startPos*stride+addend, 3 = (seqLen-1)*stride
    public let offsetKind: UInt8
    /// Metal binding index (the `index:` parameter in setBuffer).
    public let bindingIndex: Int

    /// Create a binding with a fixed slot and literal offset.
    public init(slot: Int, offset: UInt64 = 0, index: Int) {
        self.slot = .fixed(slot)
        self.byteOffset = offset
        self.offsetExpression = nil
        self.offsetKind = 0
        self.bindingIndex = index
    }

    /// Create a binding with a variable slot (runtime double-buffer).
    public init(variableSlot name: String, offset: UInt64 = 0, index: Int) {
        self.slot = .variable(name)
        self.byteOffset = offset
        self.offsetExpression = nil
        self.offsetKind = 0
        self.bindingIndex = index
    }

    /// Create a binding with a variable slot reference (SmeltSlotRef).
    public init(slot: SmeltSlotRef, index: Int) {
        self.slot = slot
        self.byteOffset = 0
        self.offsetExpression = nil
        self.offsetKind = 0
        self.bindingIndex = index
    }

    /// Create a binding with a fixed slot and runtime offset expression.
    public init(slot: Int, offsetExpression: String, index: Int) {
        self.slot = .fixed(slot)
        self.byteOffset = 0
        self.offsetExpression = offsetExpression
        self.offsetKind = 1
        self.bindingIndex = index
    }

    /// Create a binding with explicit offset kind (for prefill packed offsets).
    public init(slot: Int, offset: UInt64, offsetKind: UInt8, index: Int) {
        self.slot = .fixed(slot)
        self.byteOffset = offset
        self.offsetExpression = nil
        self.offsetKind = offsetKind
        self.bindingIndex = index
    }

    /// Preserve a graph-owned slot/offset expression while changing only its
    /// Metal argument index for an alternate kernel ABI.
    public func rebound(to index: Int) -> SmeltBufferBinding {
        SmeltBufferBinding(
            slot: slot,
            offsetExpression: offsetExpression,
            byteOffset: byteOffset,
            offsetKind: offsetKind,
            bindingIndex: index
        )
    }

    private init(
        slot: SmeltSlotRef,
        offsetExpression: String?,
        byteOffset: UInt64,
        offsetKind: UInt8,
        bindingIndex: Int
    ) {
        self.slot = slot
        self.offsetExpression = offsetExpression
        self.byteOffset = byteOffset
        self.offsetKind = offsetKind
        self.bindingIndex = bindingIndex
    }

}

/// One constant binding (setBytes) in a kernel dispatch.
public struct SmeltConstantBinding: Sendable {
    /// The Swift expression to emit (e.g. "2048" or "1e-6").
    public let expression: String
    /// The Swift type for the var declaration.
    public let type: SmeltConstantType
    /// Metal binding index.
    public let bindingIndex: Int

    public init(expression: String, type: SmeltConstantType, index: Int) {
        self.expression = expression
        self.type = type
        self.bindingIndex = index
    }
}

/// Supported constant types in kernel dispatches.
/// Byte size is derived — no caller-supplied length that could mismatch.
public enum SmeltConstantType: Sendable {
    case uint32
    case float32

    public var typeName: String {
        switch self {
        case .uint32: return "UInt32"
        case .float32: return "Float"
        }
    }

    public var byteSize: Int { 4 }
}

// MARK: - Dispatch description

/// Complete description of one kernel dispatch, ready to emit as Swift source.
public struct SmeltDispatch: Sendable {
    public let pipeline: SmeltPipeline
    /// Optional concrete Metal function name appended to the package manifest.
    /// The base `pipeline` still supplies the binding contract and fallback family.
    public let pipelineNameOverride: String?
    public let buffers: [SmeltBufferBinding]
    public let constants: [SmeltConstantBinding]
    public let dispatch: SmeltDispatchStyle
    /// Optional comment emitted above the dispatch block.
    public let comment: String?
    /// Typed planned kernel candidate for generated-route authorization.
    let plannedKernelCandidate: SmeltPlannedKernelCandidate?
    fileprivate let generatedBufferBindingCount: Int?
    fileprivate let generatedConstantCount: Int?
    /// Function constant: column count (FC_COLS). Nil for non-specialized pipelines.
    public let fcCols: Int?
    /// Function constant: group size (FC_GROUP_SIZE). Nil for non-specialized pipelines.
    public let fcGroupSize: Int?
    /// Runtime-resolved prefill grid dimensions.
    public let dynamicGridW: SmeltDynamicGridDimension?
    public let dynamicGridH: SmeltDynamicGridDimension?
    public let dynamicGridD: SmeltDynamicGridDimension?
    /// Skip this dispatch when seqLen is smaller than the threshold.
    public let minSeqLen: Int?
    /// Prefill-only: execute only when seqLen is smaller than this threshold.
    public let maxSeqLenExclusive: Int?
    /// Decode-only: execute only when position + 1 is at least this threshold.
    public let minPositionPlus1: Int?
    /// Decode-only: execute only when position + 1 is smaller than this threshold.
    public let maxPositionPlus1Exclusive: Int?

    public init(
        pipeline: SmeltPipeline,
        pipelineNameOverride: String? = nil,
        buffers: [SmeltBufferBinding],
        constants: [SmeltConstantBinding],
        dispatch: SmeltDispatchStyle,
        comment: String? = nil,
        fcCols: Int? = nil,
        fcGroupSize: Int? = nil,
        dynamicGridW: SmeltDynamicGridDimension? = nil,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        dynamicGridD: SmeltDynamicGridDimension? = nil,
        minSeqLen: Int? = nil,
        maxSeqLenExclusive: Int? = nil,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil,
        generatedBufferBindingCount: Int? = nil,
        generatedConstantCount: Int? = nil
    ) {
        self.pipeline = pipeline
        self.pipelineNameOverride = pipelineNameOverride
        self.buffers = buffers
        self.constants = constants
        self.dispatch = dispatch
        self.comment = comment
        self.plannedKernelCandidate = nil
        self.generatedBufferBindingCount = nil
        self.generatedConstantCount = nil
        self.fcCols = fcCols
        self.fcGroupSize = fcGroupSize
        self.dynamicGridW = dynamicGridW
        self.dynamicGridH = dynamicGridH
        self.dynamicGridD = dynamicGridD
        self.minSeqLen = minSeqLen
        self.maxSeqLenExclusive = maxSeqLenExclusive
        self.minPositionPlus1 = minPositionPlus1
        self.maxPositionPlus1Exclusive = maxPositionPlus1Exclusive
    }

    init(
        pipeline: SmeltPipeline,
        pipelineNameOverride: String? = nil,
        buffers: [SmeltBufferBinding],
        constants: [SmeltConstantBinding],
        dispatch: SmeltDispatchStyle,
        comment: String? = nil,
        plannedKernelCandidate: SmeltPlannedKernelCandidate,
        fcCols: Int? = nil,
        fcGroupSize: Int? = nil,
        dynamicGridW: SmeltDynamicGridDimension? = nil,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        dynamicGridD: SmeltDynamicGridDimension? = nil,
        minSeqLen: Int? = nil,
        maxSeqLenExclusive: Int? = nil,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil,
        generatedBufferBindingCount: Int? = nil,
        generatedConstantCount: Int? = nil
    ) {
        self.pipeline = pipeline
        self.pipelineNameOverride = pipelineNameOverride
        self.buffers = buffers
        self.constants = constants
        self.dispatch = dispatch
        self.comment = comment
        self.plannedKernelCandidate = plannedKernelCandidate
        self.generatedBufferBindingCount = generatedBufferBindingCount
        self.generatedConstantCount = generatedConstantCount
        self.fcCols = fcCols
        self.fcGroupSize = fcGroupSize
        self.dynamicGridW = dynamicGridW
        self.dynamicGridH = dynamicGridH
        self.dynamicGridD = dynamicGridD
        self.minSeqLen = minSeqLen
        self.maxSeqLenExclusive = maxSeqLenExclusive
        self.minPositionPlus1 = minPositionPlus1
        self.maxPositionPlus1Exclusive = maxPositionPlus1Exclusive
    }

    init(
        pipeline: SmeltPipeline,
        plannedKernelRoute: SmeltPlannedKernelRoute,
        buffers: [SmeltBufferBinding],
        constants: [SmeltConstantBinding],
        dispatch: SmeltDispatchStyle,
        comment: String? = nil,
        fcCols: Int? = nil,
        fcGroupSize: Int? = nil,
        dynamicGridW: SmeltDynamicGridDimension? = nil,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        dynamicGridD: SmeltDynamicGridDimension? = nil,
        minSeqLen: Int? = nil,
        maxSeqLenExclusive: Int? = nil,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil
    ) {
        self.init(
            pipeline: pipeline,
            pipelineNameOverride: plannedKernelRoute.pipelineNameOverride,
            buffers: buffers,
            constants: constants,
            dispatch: dispatch,
            comment: comment,
            plannedKernelCandidate: plannedKernelRoute.candidate,
            fcCols: fcCols,
            fcGroupSize: fcGroupSize,
            dynamicGridW: dynamicGridW,
            dynamicGridH: dynamicGridH,
            dynamicGridD: dynamicGridD,
            minSeqLen: minSeqLen,
            maxSeqLenExclusive: maxSeqLenExclusive,
            minPositionPlus1: minPositionPlus1,
            maxPositionPlus1Exclusive: maxPositionPlus1Exclusive,
            generatedBufferBindingCount: plannedKernelRoute.generatedBufferBindingCount,
            generatedConstantCount: plannedKernelRoute.generatedConstantCount
        )
    }
}

struct SmeltNamedPipelineUse: Sendable, Equatable {
    let name: String
    let plannedKernelCandidate: SmeltPlannedKernelCandidate?
}

// MARK: - Dispatch table conversion

extension SmeltDispatch {
    /// Convert to a binary dispatch record for dispatches.bin.
    public func toRecord() -> SmeltDispatchRecord {
        var rec = SmeltDispatchRecord.empty()
        rec.opKind = SmeltDispatchRecord.opDispatch
        rec.pipeline = UInt16(pipeline.rawValue)

        switch dispatch {
        case let .threadgroups(w, h, d, tw, th, td):
            rec.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
            rec.gridW = UInt32(w)
            rec.gridH = UInt32(h)
            rec.gridD = UInt32(d)
            rec.tgW = UInt32(tw)
            rec.tgH = UInt32(th)
            rec.tgD = UInt32(td)
        case let .threads(w, h, d, tw, th, td):
            rec.dispatchStyle = SmeltDispatchRecord.styleThreads
            rec.gridW = UInt32(w)
            rec.gridH = UInt32(h)
            rec.gridD = UInt32(d)
            rec.tgW = UInt32(tw)
            rec.tgH = UInt32(th)
            rec.tgD = UInt32(td)
        }

        var gridWKind = rec.gridWKind
        var gridW = rec.gridW
        applyDynamicGrid(dynamicGridW, toKind: &gridWKind, value: &gridW)
        rec.gridWKind = gridWKind
        rec.gridW = gridW

        var gridHKind = rec.gridHKind
        var gridH = rec.gridH
        applyDynamicGrid(dynamicGridH, toKind: &gridHKind, value: &gridH)
        rec.gridHKind = gridHKind
        rec.gridH = gridH

        var gridDKind = rec.gridDKind
        var gridD = rec.gridD
        applyDynamicGrid(dynamicGridD, toKind: &gridDKind, value: &gridD)
        rec.gridDKind = gridDKind
        rec.gridD = gridD
        rec.minSeqLen = UInt16(minSeqLen ?? 0)

        rec.bufferCount = UInt8(min(buffers.count, agentMaxBuffersPerDispatch))
        for idx in 0..<Int(rec.bufferCount) {
            let binding = buffers[idx]
            var buf = SmeltBufferRecord.empty()

            switch binding.slot {
            case let .fixed(slot): buf.slot = Int16(slot)
            case let .variable(name):
                buf.slot = name == "cur" ? SmeltBufferRecord.slotCur
                    : SmeltBufferRecord.slotAlt
            }

            buf.bindingIndex = UInt8(binding.bindingIndex)

            if binding.offsetKind > 0 && binding.offsetExpression == nil {
                // Explicit offset kind (prefill packed offsets)
                buf.offsetKind = binding.offsetKind
                buf.offset = binding.byteOffset
            } else if let expr = binding.offsetExpression {
                if expr.contains("__seqLenMinus1__") {
                    // Prefill: (seqLen-1) * stride
                    buf.offsetKind = 3
                    buf.offset = UInt64(parseOffsetStride(expr))
                } else {
                    // Decode: position * stride
                    buf.offsetKind = 1
                    buf.offset = UInt64(parseOffsetStride(expr))
                }
            } else {
                buf.offsetKind = 0
                buf.offset = binding.byteOffset
            }

            setBuffer(&rec, index: idx, value: buf)
        }

        rec.constantCount = UInt8(min(constants.count, agentMaxConstantsPerDispatch))
        for idx in 0..<Int(rec.constantCount) {
            let binding = constants[idx]
            var con = SmeltConstantRecord.empty()
            con.bindingIndex = UInt8(binding.bindingIndex)

            if binding.expression == "__seqLen__" {
                con.kind = SmeltConstantRecord.kindSeqLen
            } else if binding.expression.hasPrefix("__seqLen__*") {
                con.kind = SmeltConstantRecord.kindSeqLenMulLiteral
                let literal = binding.expression.dropFirst("__seqLen__*".count)
                con.value = UInt32(literal) ?? 0
            } else if binding.expression.hasPrefix("__seqLenModSkipIfZero__*") {
                con.kind = SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero
                let literal = binding.expression.dropFirst("__seqLenModSkipIfZero__*".count)
                con.value = UInt32(literal) ?? 0
            } else if binding.expression.hasPrefix("__seqLenMod__*") {
                con.kind = SmeltConstantRecord.kindSeqLenModLiteral
                let literal = binding.expression.dropFirst("__seqLenMod__*".count)
                con.value = UInt32(literal) ?? 0
            } else if binding.expression.hasPrefix("__startPos__+") {
                // startPos + literal: extract the literal value
                con.kind = SmeltConstantRecord.kindStartPosPlusLiteral
                let literal = binding.expression.dropFirst("__startPos__+".count)
                con.value = UInt32(literal) ?? 0
            } else if binding.expression == "__startPos__" {
                con.kind = SmeltConstantRecord.kindStartPos
            } else if binding.expression == "cacheSeqCapacity" {
                con.kind = SmeltConstantRecord.kindCacheSeqCapacity
            } else if binding.expression.contains("position + 1") {
                con.kind = SmeltConstantRecord.kindPositionPlus1
            } else if binding.expression.contains("position") {
                con.kind = SmeltConstantRecord.kindPosition
            } else {
                switch binding.type {
                case .uint32:
                    con.kind = SmeltConstantRecord.kindLiteralU32
                    if let val = UInt32(binding.expression) {
                        con.value = val
                    }
                case .float32:
                    con.kind = SmeltConstantRecord.kindLiteralF32
                    if let val = Float(binding.expression) {
                        con.value = val.bitPattern
                    }
                }
            }

            setConstant(&rec, index: idx, value: con)
        }

        var nextConstantIndex = Int(rec.constantCount)
        if let minPositionPlus1, nextConstantIndex < agentMaxConstantsPerDispatch {
            var guardCon = SmeltConstantRecord.empty()
            guardCon.kind = SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
            guardCon.bindingIndex = UInt8.max
            guardCon.value = UInt32(minPositionPlus1)
            setConstant(&rec, index: nextConstantIndex, value: guardCon)
            nextConstantIndex += 1
        }
        if let maxPositionPlus1Exclusive, nextConstantIndex < agentMaxConstantsPerDispatch {
            var guardCon = SmeltConstantRecord.empty()
            guardCon.kind = SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
            guardCon.bindingIndex = UInt8.max
            guardCon.value = UInt32(maxPositionPlus1Exclusive)
            setConstant(&rec, index: nextConstantIndex, value: guardCon)
            nextConstantIndex += 1
        }
        if let maxSeqLenExclusive, nextConstantIndex < agentMaxConstantsPerDispatch {
            var guardCon = SmeltConstantRecord.empty()
            guardCon.kind = SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse
            guardCon.bindingIndex = UInt8.max
            guardCon.value = UInt32(maxSeqLenExclusive)
            setConstant(&rec, index: nextConstantIndex, value: guardCon)
            nextConstantIndex += 1
        }
        rec.constantCount = UInt8(nextConstantIndex)

        return rec
    }
}

private func applyDynamicGrid(
    _ dim: SmeltDynamicGridDimension?,
    toKind kind: inout UInt8,
    value: inout UInt32
) {
    guard let dim else {
        kind = SmeltDispatchRecord.gridLiteral
        return
    }

    switch dim {
    case .seqLen:
        kind = SmeltDispatchRecord.gridSeqLen
        value = 0
    case .seqLenMul(let scale):
        kind = SmeltDispatchRecord.gridSeqLenMulLiteral
        value = UInt32(scale)
    case .seqLenCeilDiv(let divisor):
        kind = SmeltDispatchRecord.gridSeqLenCeilDivLiteral
        value = UInt32(divisor)
    case .seqLenFloorDiv(let divisor):
        kind = SmeltDispatchRecord.gridSeqLenCeilDivLiteral
        value = 0x8000_0000 | UInt32(divisor)
    }
}

/// Parse a position-dependent offset stride from an expression like
/// "Int(position) * 128" or "let ropeXOff = Int(position) * 128".
private func parseOffsetStride(_ expression: String) -> Int {
    // Look for "* N" pattern
    if let range = expression.range(of: #"\*\s*(\d+)"#, options: .regularExpression) {
        let match = expression[range]
        let numStr = match.drop(while: { $0 == "*" || $0 == " " })
        return Int(numStr) ?? 0
    }
    return 0
}

// MARK: - Record field accessors (fixed-size arrays as individual fields)

private func setBuffer(
    _ rec: inout SmeltDispatchRecord, index: Int, value: SmeltBufferRecord
) {
    switch index {
    case 0: rec.buf0 = value
    case 1: rec.buf1 = value
    case 2: rec.buf2 = value
    case 3: rec.buf3 = value
    case 4: rec.buf4 = value
    case 5: rec.buf5 = value
    case 6: rec.buf6 = value
    case 7: rec.buf7 = value
    case 8: rec.buf8 = value
    case 9: rec.buf9 = value
    case 10: rec.buf10 = value
    case 11: rec.buf11 = value
    case 12: rec.buf12 = value
    case 13: rec.buf13 = value
    case 14: rec.buf14 = value
    case 15: rec.buf15 = value
    default: break
    }
}

private func setConstant(
    _ rec: inout SmeltDispatchRecord, index: Int, value: SmeltConstantRecord
) {
    switch index {
    case 0: rec.con0 = value
    case 1: rec.con1 = value
    case 2: rec.con2 = value
    case 3: rec.con3 = value
    case 4: rec.con4 = value
    case 5: rec.con5 = value
    case 6: rec.con6 = value
    case 7: rec.con7 = value
    default: break
    }
}

// MARK: - Emitter errors

/// Errors from code emission — these are compiler bugs, not user errors.
public enum SmeltEmitError: Error, CustomStringConvertible {
    case bindingCountMismatch(
        pipeline: SmeltPipeline, expectedBuffers: Int, gotBuffers: Int,
        expectedConstants: Int, gotConstants: Int
    )
    case missingWeight(name: String)
    case missingConfig(detail: String)
    case unsupported(detail: String)

    public var description: String {
        switch self {
        case let .bindingCountMismatch(pipeline, eb, gb, ec, gc):
            return "binding mismatch for \(pipeline): "
                + "expected \(eb) buffers/\(ec) constants, "
                + "got \(gb) buffers/\(gc) constants"
        case let .missingWeight(name):
            return "weight '\(name)' not found in layout"
        case let .missingConfig(detail):
            return "missing config: \(detail)"
        case let .unsupported(detail):
            return "unsupported emit path: \(detail)"
        }
    }
}

// MARK: - Code emitter

/// Emits Swift source lines for Metal compute dispatches.
/// Tracks a dispatch counter to generate unique constant variable names
/// across the entire generated function body. The caller (plugin or top-level
/// emitter) is responsible for ordering dispatches and tracking buffer state.
public struct SmeltCodeEmitter {

    /// Current indentation level (number of spaces).
    public let indent: Int

    /// Pre-computed indentation string.
    private let pad: String

    /// Monotonically increasing dispatch counter — ensures unique constant var names.
    private var dispatchIndex: Int = 0

    /// Collected binary dispatch records (for dispatches.bin).
    public private(set) var dispatchRecords: [SmeltDispatchRecord] = []

    /// Collected high-level IR ops for optimization passes.
    public private(set) var emittedOps: [SmeltIROp] = []

    /// Rewrites applied during the latest optimize-and-rebuild step.
    public private(set) var optimizationStats = SmeltOptimizationStats()

    /// Key for function-constant-specialized pipeline deduplication.
    private struct FCPipelineKey: Hashable {
        let basePipeline: SmeltPipeline
        let cols: Int
        let groupSize: Int
    }

    /// Maps (base pipeline, cols, groupSize) to the specialized pipeline index.
    private var fcPipelineMap: [FCPipelineKey: Int] = [:]

    /// Ordered list of specialized pipeline entries (appended after base pipelines).
    /// Each entry is (metalFunctionName, cols, groupSize).
    public private(set) var specializedPipelines: [(name: String, cols: Int, groupSize: Int)] = []

    /// Extra pipeline manifest entries in the exact order their package-local indices were assigned.
    private var extraPipelineNames: [String] = []

    /// Maps generated Metal function names to package-local pipeline indices.
    private var namedPipelineMap: [String: Int] = [:]

    /// Ordered generated Metal function names appended after function-constant entries.
    public private(set) var namedPipelines: [String] = []

    /// Named pipeline dispatches, including planned route metadata when available.
    private(set) var namedPipelineUses: [SmeltNamedPipelineUse] = []

    /// Whether debug trace markers should be retained in emitted IR.
    private let traceMarkersEnabled: Bool

    /// Fused-kernel route planner used by helper emitters.
    let fusionPlanner: SmeltFusionPlanner

    /// Total number of pipelines (base + specialized).
    /// Base pipelines come from SmeltKernelCatalog.pipelineNames.
    public var totalPipelineCount: Int {
        SmeltKernelCatalog.pipelineNames.count + extraPipelineNames.count
    }

    /// Build the full pipeline name list for the manifest.
    /// Base pipeline names first, then package-local entries in assigned-index order.
    public func buildPipelineNames() -> [String] {
        var names = SmeltKernelCatalog.pipelineNames
        names.append(contentsOf: extraPipelineNames)
        return names
    }

    /// Resolve or create a specialized pipeline index for a function-constant dispatch.
    private mutating func resolveSpecializedPipeline(
        basePipeline: SmeltPipeline,
        cols: Int,
        groupSize: Int
    ) -> Int {
        let key = FCPipelineKey(basePipeline: basePipeline, cols: cols, groupSize: groupSize)
        if let existing = fcPipelineMap[key] {
            return existing
        }
        let sig = SmeltKernelCatalog.signatures[basePipeline.rawValue]
        let idx = SmeltKernelCatalog.pipelineNames.count + extraPipelineNames.count
        specializedPipelines.append((name: sig.metalFunctionName, cols: cols, groupSize: groupSize))
        extraPipelineNames.append("\(sig.metalFunctionName):\(cols):\(groupSize)")
        fcPipelineMap[key] = idx
        return idx
    }

    /// Resolve or create a package-local pipeline index for generated Metal wrappers.
    private mutating func resolveNamedPipeline(_ name: String) -> Int {
        if let catalogIndex = SmeltKernelCatalog.pipelineIndex(named: name) {
            return catalogIndex
        }
        if let existing = namedPipelineMap[name] {
            return existing
        }
        let idx = SmeltKernelCatalog.pipelineNames.count + extraPipelineNames.count
        namedPipelines.append(name)
        extraPipelineNames.append(name)
        namedPipelineMap[name] = idx
        return idx
    }

    private mutating func recordNamedPipelineUse(for dispatch: SmeltDispatch) {
        guard let name = dispatch.pipelineNameOverride else {
            return
        }
        namedPipelineUses.append(SmeltNamedPipelineUse(
            name: name,
            plannedKernelCandidate: dispatch.plannedKernelCandidate
        ))
    }

    public init(indent: Int = 4, traceMarkersEnabled: Bool = true) {
        self.init(
            indent: indent,
            traceMarkersEnabled: traceMarkersEnabled,
            fusionPlanner: .auto
        )
    }

    init(
        indent: Int = 4,
        traceMarkersEnabled: Bool = true,
        fusionPlanner: SmeltFusionPlanner
    ) {
        self.indent = indent
        self.pad = String(repeating: " ", count: indent)
        self.traceMarkersEnabled = traceMarkersEnabled
        self.fusionPlanner = fusionPlanner
    }

    /// Record a swap operation in the dispatch table.
    public mutating func recordSwap() {
        dispatchRecords.append(SmeltDispatchRecord.swap())
        emittedOps.append(.swap)
    }

    /// Record a named trace boundary without emitting a dispatch.
    public mutating func recordTraceMarker(label: String, bufferSlot: Int) {
        guard traceMarkersEnabled else { return }
        // precondition (not assert): the namespace invariant must hold in release
        // builds too, or a colliding debug label would be silently reclassified as
        // a capture point. This is build-time codegen, so fail-fast is correct.
        precondition(!label.hasPrefix(Self.capturePointLabelPrefix),
                     "trace-marker label collides with the GPTQ capture-point namespace: \(label)")
        emittedOps.append(.traceMarker(label: label, bufferSlot: bufferSlot))
    }

    /// Synthetic prefix for trace markers that are actually GPTQ capture points.
    /// They ride the optimizer-aware marker stream so their dispatch boundary is
    /// resolved post-optimization, but they carry capture data (weight name, K)
    /// in a side map and are excluded from the debug trace markers.
    private static let capturePointLabelPrefix = "__gptq_cap__"
    /// Capture data carried alongside the marker op; the input slot lives in the
    /// marker's own `bufferSlot` (resolved by the walk), so it isn't duplicated here.
    private var capturePointInfo: [String: (weightName: String, k: Int)] = [:]

    /// Record a GPTQ capture point: a marker after this projection's dispatch(es)
    /// at which `inputSlot` holds the `[seqLen, K]` activation the calibrator reads.
    /// Gated like trace markers, so a calibration build must run in full trace mode.
    public mutating func recordCapturePoint(weightName: String, inputSlot: Int, k: Int) {
        guard traceMarkersEnabled else { return }
        let label = "\(Self.capturePointLabelPrefix)\(capturePointInfo.count)"
        capturePointInfo[label] = (weightName, k)
        emittedOps.append(.traceMarker(label: label, bufferSlot: inputSlot))
    }

    /// Walk the emitted op stream, invoking `body` for each trace-marker op with
    /// the dispatch-only count that precedes it (swaps don't count). Both debug
    /// markers and GPTQ capture points — which share the marker op type — resolve
    /// their post-optimization boundary this way.
    private func forEachMarkerBoundary(
        _ body: (_ label: String, _ bufferSlot: Int, _ dispatchCount: Int) -> Void
    ) {
        var opCount = 0
        for op in emittedOps {
            switch op {
            case .dispatch: opCount += 1
            case .swap: break
            case let .traceMarker(label, bufferSlot): body(label, bufferSlot, opCount)
            }
        }
    }

    /// Build GPTQ capture points against the current (post-optimization) op stream.
    public func buildCapturePoints() -> [SmeltGPTQCapturePoint] {
        guard traceMarkersEnabled else { return [] }
        var points: [SmeltGPTQCapturePoint] = []
        forEachMarkerBoundary { label, bufferSlot, dispatchCount in
            guard let info = capturePointInfo[label] else { return }
            points.append(SmeltGPTQCapturePoint(
                weightName: info.weightName, inputSlot: bufferSlot,
                k: info.k, dispatchCount: dispatchCount))
        }
        return points
    }

    /// Build trace markers against the current emitted op stream.
    ///
    /// Markers resolve to the number of Metal dispatches that must execute
    /// before the named boundary is reached. Runtime debug truncation uses the
    /// same dispatch-only count and includes intervening swaps automatically.
    /// GPTQ capture points ride the same op type but are not debug boundaries.
    public func buildTraceMarkers() -> [SmeltTraceMarker] {
        guard traceMarkersEnabled else { return [] }
        var markers: [SmeltTraceMarker] = []
        forEachMarkerBoundary { label, bufferSlot, dispatchCount in
            guard capturePointInfo[label] == nil else { return }
            markers.append(SmeltTraceMarker(
                label: label, dispatchCount: dispatchCount, bufferSlot: bufferSlot))
        }
        return markers
    }

    /// Run optimization passes on the collected dispatch IR and rebuild binary records.
    /// Call this after all dispatches have been emitted, before reading dispatchRecords.
    public mutating func optimizeAndRebuildRecords() {
        optimizationStats = SmeltDispatchOptimizer.optimize(
            &emittedOps,
            planner: fusionPlanner
        )

        // Rebuild binary records from optimized IR
        dispatchRecords = []
        fcPipelineMap = [:]
        specializedPipelines = []
        extraPipelineNames = []
        namedPipelineMap = [:]
        namedPipelines = []
        namedPipelineUses = []

        for op in emittedOps {
            switch op {
            case .dispatch(let dispatch):
                let pipelineIndex: Int
                if let pipelineNameOverride = dispatch.pipelineNameOverride {
                    pipelineIndex = resolveNamedPipeline(pipelineNameOverride)
                } else if let fcCols = dispatch.fcCols,
                   let fcGroupSize = dispatch.fcGroupSize
                {
                    pipelineIndex = resolveSpecializedPipeline(
                        basePipeline: dispatch.pipeline,
                        cols: fcCols, groupSize: fcGroupSize
                    )
                } else {
                    pipelineIndex = dispatch.pipeline.rawValue
                }
                recordNamedPipelineUse(for: dispatch)
                var record = dispatch.toRecord()
                record.pipeline = UInt16(pipelineIndex)
                dispatchRecords.append(record)
            case .swap:
                dispatchRecords.append(SmeltDispatchRecord.swap())
            case .traceMarker:
                continue
            }
        }
    }

    /// Compute threadgroup width: clamped to [32, 1024] and rounded down to multiple of 32.
    /// SIMD reduction kernels require tgs to be a multiple of 32.
    private func tgWidth(_ maxThreads: Int) -> Int {
        let clamped = min(max(maxThreads, 32), 1024)
        return (clamped / 32) * 32
    }

    // MARK: - Primary emit

    /// Emit Swift source lines for a single kernel dispatch.
    /// Validates binding counts against the kernel catalog.
    public mutating func emit(_ dispatch: SmeltDispatch) throws -> [String] {
        let dIdx = dispatchIndex
        dispatchIndex += 1
        // Validate bindings against catalog
        let sig = SmeltKernelCatalog.signatures[dispatch.pipeline.rawValue]
        let expectedBufferCount = dispatch.generatedBufferBindingCount
            ?? sig.bufferBindingCount
        let expectedConstantCount = dispatch.generatedConstantCount
            ?? sig.constantCount
        if dispatch.buffers.count != expectedBufferCount
            || dispatch.constants.count != expectedConstantCount
        {
            throw SmeltEmitError.bindingCountMismatch(
                pipeline: dispatch.pipeline,
                expectedBuffers: expectedBufferCount,
                gotBuffers: dispatch.buffers.count,
                expectedConstants: expectedConstantCount,
                gotConstants: dispatch.constants.count
            )
        }

        var lines: [String] = []

        // Validate binding counts fit dispatch table format
        if dispatch.buffers.count > agentMaxBuffersPerDispatch {
            throw SmeltEmitError.missingConfig(
                detail: "Dispatch '\(dispatch.comment ?? "")' has \(dispatch.buffers.count)"
                    + " buffer bindings (max \(agentMaxBuffersPerDispatch))"
            )
        }
        if dispatch.constants.count > agentMaxConstantsPerDispatch {
            throw SmeltEmitError.missingConfig(
                detail: "Dispatch '\(dispatch.comment ?? "")' has \(dispatch.constants.count)"
                    + " constant bindings (max \(agentMaxConstantsPerDispatch))"
            )
        }

        if dispatch.pipelineNameOverride != nil
            && (dispatch.fcCols != nil || dispatch.fcGroupSize != nil)
        {
            throw SmeltEmitError.missingConfig(
                detail: "Dispatch '\(dispatch.comment ?? "")' cannot combine generated pipeline names with function constants"
            )
        }

        // Resolve pipeline index: generated name, function constants, or base.
        let pipelineIndex: Int
        if let pipelineNameOverride = dispatch.pipelineNameOverride {
            pipelineIndex = resolveNamedPipeline(pipelineNameOverride)
        } else if let fcCols = dispatch.fcCols, let fcGroupSize = dispatch.fcGroupSize {
            pipelineIndex = resolveSpecializedPipeline(
                basePipeline: dispatch.pipeline,
                cols: fcCols, groupSize: fcGroupSize
            )
        } else {
            pipelineIndex = dispatch.pipeline.rawValue
        }

        // Track high-level op for optimization passes
        emittedOps.append(.dispatch(dispatch))
        recordNamedPipelineUse(for: dispatch)

        // Collect binary dispatch record (with resolved pipeline index)
        var record = dispatch.toRecord()
        record.pipeline = UInt16(pipelineIndex)
        dispatchRecords.append(record)

        // Comment
        if let comment = dispatch.comment {
            lines.append("\(pad)// \(comment)")
        }

        // Set pipeline
        lines.append("\(pad)enc.setComputePipelineState(p[\(pipelineIndex)])")

        // Set buffers
        for binding in dispatch.buffers {
            let slotExpr = binding.slot.expression
            let offsetExpr = binding.offsetExpression ?? "\(binding.byteOffset)"
            lines.append(
                "\(pad)enc.setBuffer(b[\(slotExpr)], "
                    + "offset: \(offsetExpr), "
                    + "index: \(binding.bindingIndex))"
            )
        }

        // Set constants (each needs a var declaration + setBytes)
        // Variable names include dispatch index to avoid redeclaration in single scope
        for constant in dispatch.constants {
            let varName = "_d\(dIdx)c\(constant.bindingIndex)"
            lines.append(
                "\(pad)var \(varName): \(constant.type.typeName) = \(constant.expression)"
            )
            lines.append(
                "\(pad)enc.setBytes(&\(varName), "
                    + "length: \(constant.type.byteSize), "
                    + "index: \(constant.bindingIndex))"
            )
        }

        // Dispatch
        switch dispatch.dispatch {
        case let .threads(w, h, d, tw, th, td):
            lines.append(
                "\(pad)enc.dispatchThreads("
                    + "MTLSize(width: \(w), height: \(h), depth: \(d)), "
                    + "threadsPerThreadgroup: "
                    + "MTLSize(width: \(tw), height: \(th), depth: \(td)))"
            )
        case let .threadgroups(w, h, d, tw, th, td):
            lines.append(
                "\(pad)enc.dispatchThreadgroups("
                    + "MTLSize(width: \(w), height: \(h), depth: \(d)), "
                    + "threadsPerThreadgroup: "
                    + "MTLSize(width: \(tw), height: \(th), depth: \(td)))"
            )
        }

        var guardClauses: [String] = []
        if let minPositionPlus1 = dispatch.minPositionPlus1 {
            guardClauses.append("position + 1 >= \(minPositionPlus1)")
        }
        if let maxPositionPlus1Exclusive = dispatch.maxPositionPlus1Exclusive {
            guardClauses.append("position + 1 < \(maxPositionPlus1Exclusive)")
        }
        if !guardClauses.isEmpty {
            let condition = guardClauses.joined(separator: " && ")
            lines = ["\(pad)if \(condition) {"] + lines.map { "\(pad)\($0)" } + ["\(pad)}"]
        }

        return lines
    }

    // MARK: - Convenience emitters

    /// Emit a fused LUT matvec dispatch (quantized weight × input → output).
    /// cols and groupSize are baked into the pipeline via function constants.
    public mutating func emitFusedLUTMatvec(
        indicesSlot: Int, indicesOffset: UInt64,
        lutSlot: Int, lutOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        return try emit(SmeltDispatch(
            pipeline: .fusedLutMatvec,
            buffers: [
                SmeltBufferBinding(slot: indicesSlot, offset: indicesOffset, index: 0),
                SmeltBufferBinding(slot: lutSlot, offset: lutOffset, index: 1),
                SmeltBufferBinding(slot: inputSlot, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 4),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: cols,
            fcGroupSize: groupSize
        ))
    }

    /// Emit a fused LUT matvec dispatch with a variable input slot.
    public mutating func emitFusedLUTMatvecVar(
        indicesSlot: Int, indicesOffset: UInt64,
        lutSlot: Int, lutOffset: UInt64,
        inputSlotVar: String, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .fusedLutMatvec,
            buffers: [
                SmeltBufferBinding(slot: indicesSlot, offset: indicesOffset, index: 0),
                SmeltBufferBinding(slot: lutSlot, offset: lutOffset, index: 1),
                SmeltBufferBinding(variableSlot: inputSlotVar, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 4),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: cols,
            fcGroupSize: groupSize
        ))
    }

    /// Native signed binary/ternary matvec. Binary batches use a B4 kernel that
    /// amortizes each packed weight/scale tile across four activation rows.
    /// Ternary projections preserve MLX affine-QMV arithmetic for every row;
    /// a later matrix route must be selected explicitly rather than silently
    /// changing numerical topology merely because the activation is batched.
    /// The first optimized contract is intentionally g128, matching the
    /// released signed checkpoints. Other group geometries are first-class
    /// storage but fail here until a kernel advertises them.
    mutating func emitSignedMatvec(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputBinding: SmeltBufferBinding,
        outputBinding: SmeltBufferBinding,
        rows: Int,
        cols: Int,
        batchSize: Int = 1,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        minSeqLen: Int? = nil,
        comment: String? = nil
    ) throws -> [String] {
        guard let scalesOffset = weightEntry.scalesOffset,
              weightEntry.groupSize == 128,
              weightEntry.paddedCols == cols,
              cols % 128 == 0,
              weightEntry.shape == [rows, cols]
        else {
            throw SmeltEmitError.unsupported(
                detail: "signed matvec '\(weightEntry.name)' requires exact rank-2 g128 geometry"
            )
        }
        guard inputBinding.bindingIndex == 2, outputBinding.bindingIndex == 3 else {
            throw SmeltEmitError.missingConfig(
                detail: "signed matvec input/output bindings must use Metal indices 2/3"
            )
        }
        let isBatched = batchSize > 1 || dynamicGridH != nil
        if weightEntry.dtype == .ternary2, isBatched {
            // MLX's affine projection dispatcher on the canonical Apple GPU
            // profile switches large (D>4096 or O>4096) matrices from QMV to
            // its BM32/BN32/BK32 QMM topology at M=6. Keep both arithmetic
            // bricks in the frozen table and guard them by sequence shape;
            // no model/family identifier participates in the decision.
            let qmmMinBatch = 6
            let affineBuffers = [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 2),
                inputBinding.rebound(to: 3),
                outputBinding.rebound(to: 4),
            ]
            var result = try emit(SmeltDispatch(
                pipeline: .signedTernaryAffineMatvecG128Rows8,
                buffers: affineBuffers,
                constants: [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
                    SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 6),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8,
                    height: batchSize,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: comment.map { "\($0) (MLX QMV topology)" },
                dynamicGridH: dynamicGridH,
                minSeqLen: minSeqLen,
                maxSeqLenExclusive: qmmMinBatch
            ))
            result += try emit(SmeltDispatch(
                pipeline: .signedTernaryAffineQMMG128BM32BN32BK32,
                buffers: affineBuffers,
                constants: [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
                    SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 6),
                    SmeltConstantBinding(
                        expression: dynamicGridH == nil ? "\(batchSize)" : "__seqLen__",
                        type: .uint32,
                        index: 7
                    ),
                ],
                dispatch: .threadgroups(
                    width: (rows + 31) / 32,
                    height: (batchSize + 31) / 32,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: comment.map { "\($0) (MLX QMM topology)" },
                dynamicGridH: dynamicGridH == nil ? nil : .seqLenCeilDiv(32),
                minSeqLen: max(minSeqLen ?? 0, qmmMinBatch)
            ))
            return result
        }
        let rowTile: Int
        let threadgroupWidth: Int
        let pipeline: SmeltPipeline
        switch weightEntry.dtype {
        case .binary1:
            rowTile = 8
            threadgroupWidth = 64
            pipeline = isBatched
                ? .signedBinaryMatvecG128Rows8BatchedB4
                : .signedBinaryMatvecG128Rows8
        case .ternary2:
            rowTile = 8
            threadgroupWidth = 64
            pipeline = .signedTernaryAffineMatvecG128Rows8
        case .fp16, .fp32, .bf16, .int32, .u4Lut, .affineU4, .turboQuantH, .raw:
            throw SmeltEmitError.unsupported(
                detail: "signed matvec received \(weightEntry.dtype.rawValue) weight"
            )
        }
        let buffers: [SmeltBufferBinding]
        let constants: [SmeltConstantBinding]
        if pipeline == .signedTernaryAffineMatvecG128Rows8 {
            buffers = [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                // MLX affine projections in signed checkpoints prove
                // bias == -scale. Bind the scale region independently so the
                // Metal compiler retains the source affine expression tree
                // without adding a duplicate bias plane to the package.
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 2),
                inputBinding.rebound(to: 3),
                outputBinding.rebound(to: 4),
            ]
            constants = [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 6),
            ]
        } else {
            buffers = [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                inputBinding,
                outputBinding,
            ]
            constants = [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 5),
            ] + (isBatched ? [
                SmeltConstantBinding(
                    expression: dynamicGridH == nil ? "\(batchSize)" : "__seqLen__",
                    type: .uint32,
                    index: 6
                ),
            ] : [])
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: buffers,
            constants: constants,
            dispatch: .threadgroups(
                width: (rows + rowTile - 1) / rowTile,
                height: pipeline == .signedTernaryAffineMatvecG128Rows8
                    ? batchSize
                    : (isBatched ? (batchSize + 3) / 4 : batchSize),
                depth: 1,
                tgWidth: threadgroupWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: isBatched && dynamicGridH != nil
                ? (pipeline == .signedTernaryAffineMatvecG128Rows8
                    ? .seqLen
                    : .seqLenCeilDiv(4))
                : nil,
            minSeqLen: minSeqLen
        ))
    }

    /// Lower a CAM-authored common-input signed projection bank when its
    /// planned weights form one code slab and one scale slab. Returning nil is
    /// an ordinary backend decision: callers retain independent matvecs.
    mutating func emitSignedPackedProjectionBankIfPossible(
        members: [(entry: SmeltWeightEntry, outputSlot: Int, rows: Int)],
        weightsSlot: Int,
        inputSlot: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String]? {
        guard (2...4).contains(members.count),
              cols % 128 == 0,
              let first = members.first,
              let firstScalesOffset = first.entry.scalesOffset,
              members.allSatisfy({ member in
                  member.entry.dtype == first.entry.dtype
                      && (member.entry.dtype == .binary1
                          || member.entry.dtype == .ternary2)
                      && member.entry.groupSize == 128
                      && member.entry.paddedCols == cols
                      && member.entry.shape == [member.rows, cols]
                      && member.entry.scalesOffset != nil
                      && member.entry.scalesSizeBytes != nil
              })
        else { return nil }

        var expectedCodeOffset = first.entry.offset
        var expectedScaleOffset = firstScalesOffset
        for member in members {
            guard member.entry.offset == expectedCodeOffset,
                  member.entry.scalesOffset == expectedScaleOffset
            else { return nil }
            expectedCodeOffset += member.entry.sizeBytes
            expectedScaleOffset += member.entry.scalesSizeBytes!
        }

        let paddedMembers = members + Array(
            repeating: members.last!, count: 4 - members.count)
        let rowCounts = members.map(\.rows) + Array(
            repeating: 0, count: 4 - members.count)
        let totalRows = rowCounts.reduce(0, +)
        let isTernary = first.entry.dtype == .ternary2
        let prefixBuffers = [
            SmeltBufferBinding(
                slot: weightsSlot, offset: first.entry.offset, index: 0),
            SmeltBufferBinding(
                slot: weightsSlot, offset: firstScalesOffset, index: 1),
        ] + (isTernary ? [
            SmeltBufferBinding(
                slot: weightsSlot, offset: firstScalesOffset, index: 2),
            SmeltBufferBinding(slot: inputSlot, index: 3),
        ] : [
            SmeltBufferBinding(slot: inputSlot, index: 2),
        ])
        let outputStart = isTernary ? 4 : 3
        let constantStart = isTernary ? 8 : 7
        return try emit(SmeltDispatch(
            pipeline: isTernary
                ? .signedTernaryAffineBank4MatvecG128Rows8
                : .signedBinaryPackedBank4MatvecG128Rows8,
            buffers: prefixBuffers + paddedMembers.enumerated().map { index, member in
                SmeltBufferBinding(slot: member.outputSlot, index: index + outputStart)
            },
            constants: rowCounts.enumerated().map { index, rows in
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32,
                    index: index + constantStart)
            } + [
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32,
                    index: constantStart + 4),
            ],
            dispatch: .threadgroups(
                width: (totalRows + 7) / 8,
                height: 1,
                depth: 1,
                tgWidth: 64,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    mutating func emitSignedActivationView(
        _ view: SmeltCAMIR.ProjectionActivationView,
        inputSlot: Int,
        planesSlot: Int,
        scalesSlot: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String] {
        let pipeline: SmeltPipeline
        switch view {
        case .signedBitplanesI2:
            pipeline = .signedActivationBitplanesI2G128
        case .signedBitplanesI3:
            pipeline = .signedActivationBitplanesI3G128
        case .signedBitplanesI4:
            pipeline = .signedActivationBitplanesI4G128
        case .signedBitplanesI5:
            pipeline = .signedActivationBitplanesI5G128
        case .signedBitplanesI6:
            pipeline = .signedActivationBitplanesI6G128
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: planesSlot, index: 1),
                SmeltBufferBinding(slot: scalesSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 3),
            ],
            dispatch: .threadgroups(
                width: (cols + 127) / 128,
                height: 1,
                depth: 1,
                tgWidth: 32,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Materialize the same CAM-owned signed activation view for every active
    /// prompt row. The layout is token-major so a consumer can tile prompt
    /// rows while decode continues to use the first single-token slice.
    mutating func emitSignedActivationViewBatched(
        _ view: SmeltCAMIR.ProjectionActivationView,
        inputSlot: Int,
        planesSlot: Int,
        scalesSlot: Int,
        cols: Int,
        batchSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        let pipeline: SmeltPipeline
        switch view {
        case .signedBitplanesI2:
            pipeline = .signedActivationBitplanesI2G128Batched
        case .signedBitplanesI3:
            pipeline = .signedActivationBitplanesI3G128Batched
        case .signedBitplanesI4:
            pipeline = .signedActivationBitplanesI4G128Batched
        case .signedBitplanesI5:
            pipeline = .signedActivationBitplanesI5G128Batched
        case .signedBitplanesI6:
            pipeline = .signedActivationBitplanesI6G128Batched
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: planesSlot, index: 1),
                SmeltBufferBinding(slot: scalesSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 3),
            ],
            dispatch: .threadgroups(
                width: (cols + 127) / 128,
                height: batchSize,
                depth: 1,
                tgWidth: 32,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLen
        ))
    }

    mutating func emitSignedBitplaneMatmulBatchedIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        batchSize: Int,
        comment: String? = nil
    ) throws -> [String]? {
        guard weightEntry.dtype == .binary1,
              weightEntry.groupSize == 128,
              weightEntry.paddedCols == cols,
              weightEntry.shape == [rows, cols],
              let scalesOffset = weightEntry.scalesOffset
        else { return nil }

        let pipeline: SmeltPipeline
        switch view {
        case .signedBitplanesI2:
            pipeline = .signedBinaryBitplaneI2MatvecG128Rows8BatchedB4
        case .signedBitplanesI3:
            pipeline = .signedBinaryBitplaneI3MatvecG128Rows8BatchedB4
        case .signedBitplanesI4:
            pipeline = .signedBinaryBitplaneI4MatvecG128Rows8BatchedB4
        case .signedBitplanesI5:
            pipeline = .signedBinaryBitplaneI5MatvecG128Rows8BatchedB4
        case .signedBitplanesI6:
            pipeline = .signedBinaryBitplaneI6MatvecG128Rows8BatchedB4
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(slot: planesSlot, index: 2),
                SmeltBufferBinding(slot: activationScalesSlot, index: 3),
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 5),
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 6),
                SmeltConstantBinding(
                    expression: "__seqLen__", type: .uint32, index: 7),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8,
                height: (batchSize + 3) / 4,
                depth: 1,
                tgWidth: 64,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLenCeilDiv(4)
        ))
    }

    /// Lower a common-input projection bank through one graph-owned batched
    /// activation view and independently addressable tiled consumers. Physical
    /// weight contiguity is optional; it can be exploited by a later packed
    /// bank kernel without changing the graph contract.
    mutating func emitSignedBitplaneProjectionBankBatchedIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        members: [(entry: SmeltWeightEntry, outputSlot: Int, rows: Int)],
        weightsSlot: Int,
        inputSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        cols: Int,
        batchSize: Int,
        producerComment: String? = nil,
        projectionComment: String? = nil
    ) throws -> [String]? {
        guard !members.isEmpty,
              members.allSatisfy({ member in
                  member.entry.dtype == .binary1
                      && member.entry.groupSize == 128
                      && member.entry.paddedCols == cols
                      && member.entry.shape == [member.rows, cols]
                      && member.entry.scalesOffset != nil
              })
        else { return nil }

        var lines = try emitSignedActivationViewBatched(
            view,
            inputSlot: inputSlot,
            planesSlot: planesSlot,
            scalesSlot: activationScalesSlot,
            cols: cols,
            batchSize: batchSize,
            comment: producerComment
        )
        for member in members {
            guard let projection = try emitSignedBitplaneMatmulBatchedIfPossible(
                view: view,
                weightEntry: member.entry,
                weightsSlot: weightsSlot,
                planesSlot: planesSlot,
                activationScalesSlot: activationScalesSlot,
                outputSlot: member.outputSlot,
                rows: member.rows,
                cols: cols,
                batchSize: batchSize,
                comment: projectionComment
            ) else {
                preconditionFailure(
                    "validated batched activation-view bank member failed to emit")
            }
            lines += projection
        }
        return lines
    }

    mutating func emitSignedBitplaneProjectionBankIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        members: [(entry: SmeltWeightEntry, outputSlot: Int, rows: Int)],
        weightsSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String]? {
        let ternaryViewSupported: Bool
        switch view {
        case .signedBitplanesI4, .signedBitplanesI5, .signedBitplanesI6:
            ternaryViewSupported = true
        case .signedBitplanesI2, .signedBitplanesI3:
            ternaryViewSupported = false
        }
        if ternaryViewSupported,
           !members.isEmpty,
           cols.isMultiple(of: 128),
           members.allSatisfy({ member in
               member.entry.dtype == .ternary2
                   && member.entry.groupSize == 128
                   && member.entry.paddedCols == cols
                   && member.entry.shape == [member.rows, cols]
                   && member.entry.scalesOffset != nil
           }) {
            let ternaryBankPipeline: SmeltPipeline
            switch view {
            case .signedBitplanesI4:
                ternaryBankPipeline = .signedTernaryBitplaneI4Bank4MatvecG128Rows8
            case .signedBitplanesI5:
                ternaryBankPipeline = .signedTernaryBitplaneI5Bank4MatvecG128Rows8
            case .signedBitplanesI6:
                ternaryBankPipeline = .signedTernaryBitplaneI6Bank4MatvecG128Rows8
            case .signedBitplanesI2, .signedBitplanesI3:
                preconditionFailure("unsupported ternary activation view")
            }

            if (2...4).contains(members.count),
               let first = members.first,
               let firstScalesOffset = first.entry.scalesOffset {
                var expectedCodeOffset = first.entry.offset
                var expectedScaleOffset = firstScalesOffset
                var isContiguous = true
                for member in members {
                    guard member.entry.offset == expectedCodeOffset,
                          member.entry.scalesOffset == expectedScaleOffset,
                          let scalesSizeBytes = member.entry.scalesSizeBytes
                    else {
                        isContiguous = false
                        break
                    }
                    expectedCodeOffset += member.entry.sizeBytes
                    expectedScaleOffset += scalesSizeBytes
                }
                if isContiguous {
                    let paddedMembers = members + Array(
                        repeating: members.last!, count: 4 - members.count)
                    let rowCounts = members.map(\.rows) + Array(
                        repeating: 0, count: 4 - members.count)
                    return try emit(SmeltDispatch(
                        pipeline: ternaryBankPipeline,
                        buffers: [
                            SmeltBufferBinding(
                                slot: weightsSlot,
                                offset: first.entry.offset,
                                index: 0
                            ),
                            SmeltBufferBinding(
                                slot: weightsSlot,
                                offset: firstScalesOffset,
                                index: 1
                            ),
                            SmeltBufferBinding(slot: planesSlot, index: 2),
                            SmeltBufferBinding(
                                slot: activationScalesSlot,
                                index: 3
                            ),
                        ] + paddedMembers.enumerated().map { index, member in
                            SmeltBufferBinding(
                                slot: member.outputSlot,
                                index: index + 4
                            )
                        },
                        constants: rowCounts.enumerated().map { index, rows in
                            SmeltConstantBinding(
                                expression: "\(rows)",
                                type: .uint32,
                                index: index + 8
                            )
                        } + [
                            SmeltConstantBinding(
                                expression: "\(cols)",
                                type: .uint32,
                                index: 12
                            ),
                        ],
                        dispatch: .threadgroups(
                            width: (rowCounts.reduce(0, +) + 3) / 4,
                            height: 1,
                            depth: 1,
                            tgWidth: 64,
                            tgHeight: 1,
                            tgDepth: 1
                        ),
                        comment: comment
                    ))
                }
            }

            // Contiguity is an optimization contract, not a model validity
            // requirement. Preserve the same graph as independent consumers
            // when a package/backend layout cannot expose a packed bank.
            var lines: [String] = []
            for member in members {
                guard let memberLines = try emitSignedBitplaneMatvecIfPossible(
                    view: view,
                    weightEntry: member.entry,
                    weightsSlot: weightsSlot,
                    planesSlot: planesSlot,
                    activationScalesSlot: activationScalesSlot,
                    outputSlot: member.outputSlot,
                    rows: member.rows,
                    cols: cols,
                    comment: comment
                ) else {
                    preconditionFailure(
                        "validated ternary activation-view bank member failed to emit")
                }
                lines += memberLines
            }
            return lines
        }

        let pipeline: SmeltPipeline
        switch view {
        case .signedBitplanesI2:
            pipeline = .signedBinaryBitplaneI2Bank4MatvecG128Rows8
        case .signedBitplanesI3:
            pipeline = .signedBinaryBitplaneI3Bank4MatvecG128Rows8
        case .signedBitplanesI4, .signedBitplanesI5, .signedBitplanesI6:
            return nil
        }
        guard (2...4).contains(members.count),
              cols % 128 == 0,
              let first = members.first,
              let firstScalesOffset = first.entry.scalesOffset,
              members.allSatisfy({ member in
                  member.entry.dtype == .binary1
                      && member.entry.groupSize == 128
                      && member.entry.paddedCols == cols
                      && member.entry.shape == [member.rows, cols]
                      && member.entry.scalesOffset != nil
                      && member.entry.scalesSizeBytes != nil
              })
        else { return nil }
        var expectedCodeOffset = first.entry.offset
        var expectedScaleOffset = firstScalesOffset
        for member in members {
            guard member.entry.offset == expectedCodeOffset,
                  member.entry.scalesOffset == expectedScaleOffset
            else { return nil }
            expectedCodeOffset += member.entry.sizeBytes
            expectedScaleOffset += member.entry.scalesSizeBytes!
        }

        let paddedMembers = members + Array(
            repeating: members.last!, count: 4 - members.count)
        let rowCounts = members.map(\.rows) + Array(
            repeating: 0, count: 4 - members.count)
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: first.entry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: firstScalesOffset, index: 1),
                SmeltBufferBinding(slot: planesSlot, index: 2),
                SmeltBufferBinding(slot: activationScalesSlot, index: 3),
            ] + paddedMembers.enumerated().map { index, member in
                SmeltBufferBinding(slot: member.outputSlot, index: index + 4)
            },
            constants: rowCounts.enumerated().map { index, rows in
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: index + 8)
            } + [
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 12),
            ],
            dispatch: .threadgroups(
                width: (rowCounts.reduce(0, +) + 7) / 8,
                height: 1,
                depth: 1,
                tgWidth: 64,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    mutating func emitSignedBitplaneMatvecIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String]? {
        let useWideTernaryRows = weightEntry.dtype == .ternary2
            && cols >= 8_192 && rows >= 1_024
        let pipeline: SmeltPipeline
        switch (weightEntry.dtype, view) {
        case (.binary1, .signedBitplanesI2):
            pipeline = .signedBinaryBitplaneI2MatvecG128Rows8
        case (.binary1, .signedBitplanesI3):
            pipeline = .signedBinaryBitplaneI3MatvecG128Rows8
        case (.binary1, .signedBitplanesI4):
            pipeline = .signedBinaryBitplaneI4MatvecG128Rows8
        case (.binary1, .signedBitplanesI5):
            pipeline = .signedBinaryBitplaneI5MatvecG128Rows8
        case (.binary1, .signedBitplanesI6):
            pipeline = .signedBinaryBitplaneI6MatvecG128Rows8
        case (.ternary2, .signedBitplanesI4):
            pipeline = useWideTernaryRows
                ? .signedTernaryBitplaneI4MatvecG128Rows2Wide
                : .signedTernaryBitplaneI4MatvecG128Rows8
        case (.ternary2, .signedBitplanesI5):
            pipeline = useWideTernaryRows
                ? .signedTernaryBitplaneI5MatvecG128Rows2Wide
                : .signedTernaryBitplaneI5MatvecG128Rows8
        case (.ternary2, .signedBitplanesI6):
            pipeline = useWideTernaryRows
                ? .signedTernaryBitplaneI6MatvecG128Rows2Wide
                : .signedTernaryBitplaneI6MatvecG128Rows8
        default:
            return nil
        }
        guard weightEntry.groupSize == 128,
              weightEntry.paddedCols == cols,
              weightEntry.shape == [rows, cols],
              let scalesOffset = weightEntry.scalesOffset
        else { return nil }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(slot: planesSlot, index: 2),
                SmeltBufferBinding(slot: activationScalesSlot, index: 3),
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 5),
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 6),
            ],
            dispatch: .threadgroups(
                width: weightEntry.dtype == .ternary2
                    ? (rows + 3) / 4 : (rows + 7) / 8,
                height: 1,
                depth: 1,
                tgWidth: useWideTernaryRows
                    ? 64 : (weightEntry.dtype == .ternary2 ? 128 : 64),
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a complete lossy activation-view producer followed by its single
    /// binary projection consumer. Keeping the pair in one side-effecting
    /// emitter guarantees dispatch-record order matches graph order; callers
    /// must not probe support by emitting the consumer before the producer.
    mutating func emitSignedBitplaneProjectionIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        producerComment: String? = nil,
        projectionComment: String? = nil
    ) throws -> [String]? {
        guard weightEntry.dtype == .binary1 || weightEntry.dtype == .ternary2,
              weightEntry.groupSize == 128,
              weightEntry.paddedCols == cols,
              weightEntry.shape == [rows, cols],
              weightEntry.scalesOffset != nil
        else { return nil }

        var lines = try emitSignedActivationView(
            view,
            inputSlot: inputSlot,
            planesSlot: planesSlot,
            scalesSlot: activationScalesSlot,
            cols: cols,
            comment: producerComment
        )
        guard let projectionLines = try emitSignedBitplaneMatvecIfPossible(
            view: view,
            weightEntry: weightEntry,
            weightsSlot: weightsSlot,
            planesSlot: planesSlot,
            activationScalesSlot: activationScalesSlot,
            outputSlot: outputSlot,
            rows: rows,
            cols: cols,
            comment: projectionComment
        ) else {
            preconditionFailure("validated signed bitplane projection failed to emit")
        }
        lines += projectionLines
        return lines
    }

    /// Native binary gate+up projection with the SwiGLU consumer folded into
    /// the same dispatch. The geometry is authorized by the matvec gateway;
    /// model emitters only select it when both bricks have identical binary
    /// g128 storage.
    mutating func emitSignedBinaryGateUpSwiglu(
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputBinding: SmeltBufferBinding,
        outputBinding: SmeltBufferBinding,
        rows: Int,
        cols: Int,
        batchSize: Int = 1,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        comment: String? = nil
    ) throws -> [String] {
        guard gateEntry.dtype == .binary1,
              upEntry.dtype == .binary1,
              let gateScales = gateEntry.scalesOffset,
              let upScales = upEntry.scalesOffset,
              gateEntry.groupSize == 128,
              upEntry.groupSize == 128,
              gateEntry.paddedCols == cols,
              upEntry.paddedCols == cols,
              gateEntry.shape == [rows, cols],
              upEntry.shape == [rows, cols],
              cols % 128 == 0
        else {
            throw SmeltEmitError.unsupported(
                detail: "signed binary gate/up requires matching rank-2 g128 geometry"
            )
        }
        guard inputBinding.bindingIndex == 4, outputBinding.bindingIndex == 5 else {
            throw SmeltEmitError.missingConfig(
                detail: "signed binary gate/up input/output bindings must use Metal indices 4/5"
            )
        }
        let isBatched = batchSize > 1 || dynamicGridH != nil
        return try emit(SmeltDispatch(
            pipeline: isBatched
                ? .signedBinaryGateUpSwigluG128Rows8BatchedB4
                : .signedBinaryGateUpSwigluG128Rows8,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateScales, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upEntry.offset, index: 2),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upScales, index: 3),
                inputBinding,
                outputBinding,
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 6),
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 7),
            ] + (isBatched ? [
                SmeltConstantBinding(
                    expression: dynamicGridH == nil ? "\(batchSize)" : "__seqLen__",
                    type: .uint32,
                    index: 8
                ),
            ] : []),
            dispatch: .threadgroups(
                width: (rows + 7) / 8,
                height: isBatched ? (batchSize + 3) / 4 : batchSize,
                depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: isBatched && dynamicGridH != nil
                ? .seqLenCeilDiv(4)
                : nil
        ))
    }

    /// Native ternary gate+up projection with the staged fp16 SwiGLU consumer
    /// folded into the same dispatch. This is authorized by dtype and exact
    /// g128 geometry only; no model or family identifier participates.
    mutating func emitSignedTernaryAffineGateUpSwiglu(
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputBinding: SmeltBufferBinding,
        outputBinding: SmeltBufferBinding,
        rows: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String] {
        guard gateEntry.dtype == .ternary2,
              upEntry.dtype == .ternary2,
              let gateScales = gateEntry.scalesOffset,
              let upScales = upEntry.scalesOffset,
              gateEntry.groupSize == 128,
              upEntry.groupSize == 128,
              gateEntry.paddedCols == cols,
              upEntry.paddedCols == cols,
              gateEntry.shape == [rows, cols],
              upEntry.shape == [rows, cols],
              cols % 128 == 0
        else {
            throw SmeltEmitError.unsupported(
                detail: "signed ternary gate/up requires matching rank-2 g128 geometry"
            )
        }
        guard inputBinding.bindingIndex == 6, outputBinding.bindingIndex == 7 else {
            throw SmeltEmitError.missingConfig(
                detail: "signed ternary gate/up input/output bindings must use Metal indices 6/7"
            )
        }
        return try emit(SmeltDispatch(
            pipeline: .signedTernaryAffineGateUpSwigluG128Rows8,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateScales, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateScales, index: 2),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upEntry.offset, index: 3),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upScales, index: 4),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upScales, index: 5),
                inputBinding,
                outputBinding,
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 8),
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 9),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8,
                height: 1,
                depth: 1,
                tgWidth: 64,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Prefer the CAM-authored contiguous projection-bank row space for exact
    /// ternary gate/up when available, then apply the graph-owned precise
    /// SwiGLU as a second dispatch. This keeps four affine accumulators live
    /// instead of eight; the direct one-dispatch fusion remains the fallback
    /// for compatible but non-contiguous weight regions.
    mutating func emitSignedTernaryAffineBankGateUpSwigluIfPossible(
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        gateOutputSlot: Int,
        upOutputSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        comment: String? = nil
    ) throws -> [String]? {
        guard let bankLines = try emitSignedPackedProjectionBankIfPossible(
            members: [
                (entry: gateEntry, outputSlot: gateOutputSlot, rows: rows),
                (entry: upEntry, outputSlot: upOutputSlot, rows: rows),
            ],
            weightsSlot: weightsSlot,
            inputSlot: inputSlot,
            cols: cols,
            comment: comment
        ) else { return nil }
        var lines = bankLines
        lines += try emit(SmeltDispatch(
            pipeline: .swigluFused,
            buffers: [
                SmeltBufferBinding(slot: gateOutputSlot, index: 0),
                SmeltBufferBinding(slot: upOutputSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: rows,
                height: 1,
                depth: 1,
                tgWidth: min(rows, 1_024),
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: "SwiGLU fused"
        ))
        return lines
    }

    /// Use a CAM-owned low-bit activation view for paired gate/up weights.
    /// The measured cost rule keeps the fp16-dot route for small cells where
    /// producing bitplanes costs more than the popcount consumer saves.
    mutating func emitSignedBitplaneGateUpSwigluIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        producerComment: String? = nil,
        consumerComment: String? = nil
    ) throws -> [String]? {
        guard gateEntry.dtype == upEntry.dtype else { return nil }
        let pipeline: SmeltPipeline
        switch (gateEntry.dtype, view) {
        case (.binary1, .signedBitplanesI3):
            pipeline = .signedBinaryBitplaneI3GateUpSwigluG128Rows8
        case (.binary1, .signedBitplanesI4):
            pipeline = .signedBinaryBitplaneI4GateUpSwigluG128Rows8
        case (.binary1, .signedBitplanesI5):
            pipeline = .signedBinaryBitplaneI5GateUpSwigluG128Rows8
        case (.binary1, .signedBitplanesI6):
            pipeline = .signedBinaryBitplaneI6GateUpSwigluG128Rows8
        case (.ternary2, .signedBitplanesI4):
            pipeline = .signedTernaryBitplaneI4GateUpSwigluG128Rows8
        case (.ternary2, .signedBitplanesI5):
            pipeline = .signedTernaryBitplaneI5GateUpSwigluG128Rows8
        case (.ternary2, .signedBitplanesI6):
            pipeline = .signedTernaryBitplaneI6GateUpSwigluG128Rows8
        default:
            return nil
        }
        let (cellElements, cellSizeOverflow) = rows.multipliedReportingOverflow(by: cols)
        guard !cellSizeOverflow,
              cellElements >= 8_000_000,
              gateEntry.groupSize == 128,
              upEntry.groupSize == 128,
              gateEntry.paddedCols == cols,
              upEntry.paddedCols == cols,
              gateEntry.shape == [rows, cols],
              upEntry.shape == [rows, cols],
              let gateScales = gateEntry.scalesOffset,
              let upScales = upEntry.scalesOffset,
              cols.isMultiple(of: 128)
        else { return nil }

        var lines = try emitSignedActivationView(
            view,
            inputSlot: inputSlot,
            planesSlot: planesSlot,
            scalesSlot: activationScalesSlot,
            cols: cols,
            comment: producerComment
        )
        let ternary = gateEntry.dtype == .ternary2
        lines += try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: gateScales, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upEntry.offset, index: 2),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: upScales, index: 3),
                SmeltBufferBinding(slot: planesSlot, index: 4),
                SmeltBufferBinding(slot: activationScalesSlot, index: 5),
                SmeltBufferBinding(slot: outputSlot, index: 6),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 7),
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 8),
            ],
            dispatch: .threadgroups(
                width: (rows + (ternary ? 3 : 7)) / (ternary ? 4 : 8),
                height: 1,
                depth: 1,
                tgWidth: ternary ? 128 : 64,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: consumerComment
        ))
        return lines
    }

    /// Lower a gate/up bank through one shared activation view, two ordinary
    /// projection consumers, and the existing SwiGLU node. This is the generic
    /// occupancy-friendly fallback when a storage/view family has no winning
    /// fused dual-projection consumer.
    mutating func emitSignedBitplaneStagedGateUpSwigluIfPossible(
        view: SmeltCAMIR.ProjectionActivationView,
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        planesSlot: Int,
        activationScalesSlot: Int,
        gateOutputSlot: Int,
        upOutputSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        producerComment: String? = nil,
        consumerComment: String? = nil
    ) throws -> [String]? {
        let supportedView: Bool
        switch view {
        case .signedBitplanesI4, .signedBitplanesI5, .signedBitplanesI6:
            supportedView = true
        case .signedBitplanesI2, .signedBitplanesI3:
            supportedView = false
        }
        guard supportedView,
              gateEntry.dtype == .ternary2,
              upEntry.dtype == .ternary2,
              gateEntry.groupSize == 128,
              upEntry.groupSize == 128,
              gateEntry.paddedCols == cols,
              upEntry.paddedCols == cols,
              gateEntry.shape == [rows, cols],
              upEntry.shape == [rows, cols],
              gateEntry.scalesOffset != nil,
              upEntry.scalesOffset != nil,
              cols.isMultiple(of: 128)
        else { return nil }

        var lines = try emitSignedActivationView(
            view,
            inputSlot: inputSlot,
            planesSlot: planesSlot,
            scalesSlot: activationScalesSlot,
            cols: cols,
            comment: producerComment
        )
        guard let projectionLines = try emitSignedBitplaneProjectionBankIfPossible(
            view: view,
            members: [
                (entry: gateEntry, outputSlot: gateOutputSlot, rows: rows),
                (entry: upEntry, outputSlot: upOutputSlot, rows: rows),
            ],
            weightsSlot: weightsSlot,
            planesSlot: planesSlot,
            activationScalesSlot: activationScalesSlot,
            cols: cols,
            comment: consumerComment
        ) else {
            preconditionFailure("validated staged signed gate/up bank failed to emit")
        }
        lines += projectionLines
        lines += try emit(SmeltDispatch(
            pipeline: .swigluFused,
            buffers: [
                SmeltBufferBinding(slot: gateOutputSlot, index: 0),
                SmeltBufferBinding(slot: upOutputSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: rows, height: 1, depth: 1,
                tgWidth: min(rows, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: "Staged signed gate/up SwiGLU"
        ))
        return lines
    }

    /// Emit a fused gate+up+SwiGLU dispatch: two LUT matvecs + SwiGLU activation in one kernel.
    /// Replaces 3 separate dispatches (gate_proj, up_proj, swiglu_fused).
    /// cols and groupSize are baked into the pipeline via function constants.
    public mutating func emitFusedGateUpSwiglu(
        gateIndicesSlot: Int, gateIndicesOffset: UInt64,
        gateLutSlot: Int, gateLutOffset: UInt64,
        upIndicesSlot: Int, upIndicesOffset: UInt64,
        upLutSlot: Int, upLutOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        return try emit(SmeltDispatch(
            pipeline: .fusedGateUpSwiglu,
            buffers: [
                SmeltBufferBinding(slot: gateIndicesSlot, offset: gateIndicesOffset, index: 0),
                SmeltBufferBinding(slot: gateLutSlot, offset: gateLutOffset, index: 1),
                SmeltBufferBinding(slot: upIndicesSlot, offset: upIndicesOffset, index: 2),
                SmeltBufferBinding(slot: upLutSlot, offset: upLutOffset, index: 3),
                SmeltBufferBinding(slot: inputSlot, index: 4),
                SmeltBufferBinding(slot: outputSlot, index: 5),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 6),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: cols,
            fcGroupSize: groupSize
        ))
    }

    /// Emit a fused dual LUT matvec dispatch: two weight matrices, same input, two outputs.
    /// Replaces 2 separate fused_lut_matvec dispatches that share the same input.
    /// cols and groupSize are baked into the pipeline via function constants.
    public mutating func emitFusedDualLutMatvec(
        w1IndicesSlot: Int, w1IndicesOffset: UInt64,
        w1LutSlot: Int, w1LutOffset: UInt64,
        w2IndicesSlot: Int, w2IndicesOffset: UInt64,
        w2LutSlot: Int, w2LutOffset: UInt64,
        inputSlot: Int, output1Slot: Int, output2Slot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        return try emit(SmeltDispatch(
            pipeline: .fusedDualLutMatvec,
            buffers: [
                SmeltBufferBinding(slot: w1IndicesSlot, offset: w1IndicesOffset, index: 0),
                SmeltBufferBinding(slot: w1LutSlot, offset: w1LutOffset, index: 1),
                SmeltBufferBinding(slot: w2IndicesSlot, offset: w2IndicesOffset, index: 2),
                SmeltBufferBinding(slot: w2LutSlot, offset: w2LutOffset, index: 3),
                SmeltBufferBinding(slot: inputSlot, index: 4),
                SmeltBufferBinding(slot: output1Slot, index: 5),
                SmeltBufferBinding(slot: output2Slot, index: 6),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 7),
            ],
            dispatch: .threadgroups(
                width: (rows + 7) / 8, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: cols,
            fcGroupSize: groupSize
        ))
    }

    /// Emit a fused dual affine matvec: two weight matrices, same input, two outputs.
    public mutating func emitFusedDualAffineMatvec(
        w1WeightsSlot: Int, w1WeightsOffset: UInt64,
        w1ScalesSlot: Int, w1ScalesOffset: UInt64,
        w1BiasesSlot: Int, w1BiasesOffset: UInt64,
        w2WeightsSlot: Int, w2WeightsOffset: UInt64,
        w2ScalesSlot: Int, w2ScalesOffset: UInt64,
        w2BiasesSlot: Int, w2BiasesOffset: UInt64,
        inputSlot: Int, output1Slot: Int, output2Slot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeDualAffineMatvec(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let specializedPipeline = route.pipeline
        let rowTile = SmeltKernelShapeRegistry.decodeDualAffineRowTile(specializedPipeline)
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .fusedDualAffineMatvec,
            buffers: [
                SmeltBufferBinding(slot: w1WeightsSlot, offset: w1WeightsOffset, index: 0),
                SmeltBufferBinding(slot: w1ScalesSlot, offset: w1ScalesOffset, index: 1),
                SmeltBufferBinding(slot: w1BiasesSlot, offset: w1BiasesOffset, index: 2),
                SmeltBufferBinding(slot: w2WeightsSlot, offset: w2WeightsOffset, index: 3),
                SmeltBufferBinding(slot: w2ScalesSlot, offset: w2ScalesOffset, index: 4),
                SmeltBufferBinding(slot: w2BiasesSlot, offset: w2BiasesOffset, index: 5),
                SmeltBufferBinding(slot: inputSlot, index: 6),
                SmeltBufferBinding(slot: output1Slot, index: 7),
                SmeltBufferBinding(slot: output2Slot, index: 8),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 9),
            ] : [],
            dispatch: .threadgroups(
                width: (rows + rowTile - 1) / rowTile, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: specializedPipeline == nil ? cols : nil,
            fcGroupSize: specializedPipeline == nil ? groupSize : nil
        ))
    }

    /// Emit a fused affine gate+up+SwiGLU: two affine matvecs + SwiGLU activation in one kernel.
    public mutating func emitFusedAffineGateUpSwiglu(
        gateWeightsSlot: Int, gateWeightsOffset: UInt64,
        gateScalesSlot: Int, gateScalesOffset: UInt64,
        gateBiasesSlot: Int, gateBiasesOffset: UInt64,
        upWeightsSlot: Int, upWeightsOffset: UInt64,
        upScalesSlot: Int, upScalesOffset: UInt64,
        upBiasesSlot: Int, upBiasesOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeFusedSwiGLU(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let specializedPipeline = route.pipeline
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .fusedAffineGateUpSwiglu,
            buffers: [
                SmeltBufferBinding(slot: gateWeightsSlot, offset: gateWeightsOffset, index: 0),
                SmeltBufferBinding(slot: gateScalesSlot, offset: gateScalesOffset, index: 1),
                SmeltBufferBinding(slot: gateBiasesSlot, offset: gateBiasesOffset, index: 2),
                SmeltBufferBinding(slot: upWeightsSlot, offset: upWeightsOffset, index: 3),
                SmeltBufferBinding(slot: upScalesSlot, offset: upScalesOffset, index: 4),
                SmeltBufferBinding(slot: upBiasesSlot, offset: upBiasesOffset, index: 5),
                SmeltBufferBinding(slot: inputSlot, index: 6),
                SmeltBufferBinding(slot: outputSlot, index: 7),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 8),
            ] : [],
            dispatch: {
                return .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1,
                    depth: 1,
                    tgWidth: route.threadgroupWidth,
                    tgHeight: 1,
                    tgDepth: 1
                )
            }(),
            comment: comment,
            fcCols: specializedPipeline == nil ? cols : nil,
            fcGroupSize: specializedPipeline == nil ? groupSize : nil
        ))
    }

    /// Emit a fused affine gate+up+GeGLU: two affine matvecs + GeGLU activation in one kernel.
    public mutating func emitFusedAffineGateUpGeGLU(
        gateWeightsSlot: Int, gateWeightsOffset: UInt64,
        gateScalesSlot: Int, gateScalesOffset: UInt64,
        gateBiasesSlot: Int, gateBiasesOffset: UInt64,
        upWeightsSlot: Int, upWeightsOffset: UInt64,
        upScalesSlot: Int, upScalesOffset: UInt64,
        upBiasesSlot: Int, upBiasesOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeFusedGeGLU(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let specializedPipeline = route.pipeline
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .fusedAffineGateUpGeGLU,
            buffers: [
                SmeltBufferBinding(slot: gateWeightsSlot, offset: gateWeightsOffset, index: 0),
                SmeltBufferBinding(slot: gateScalesSlot, offset: gateScalesOffset, index: 1),
                SmeltBufferBinding(slot: gateBiasesSlot, offset: gateBiasesOffset, index: 2),
                SmeltBufferBinding(slot: upWeightsSlot, offset: upWeightsOffset, index: 3),
                SmeltBufferBinding(slot: upScalesSlot, offset: upScalesOffset, index: 4),
                SmeltBufferBinding(slot: upBiasesSlot, offset: upBiasesOffset, index: 5),
                SmeltBufferBinding(slot: inputSlot, index: 6),
                SmeltBufferBinding(slot: outputSlot, index: 7),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 8),
            ] : [],
            dispatch: {
                return .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1,
                    depth: 1,
                    tgWidth: route.threadgroupWidth,
                    tgHeight: 1,
                    tgDepth: 1
                )
            }(),
            comment: comment,
            fcCols: specializedPipeline == nil ? cols : nil,
            fcGroupSize: specializedPipeline == nil ? groupSize : nil
        ))
    }

    /// Emit an affine u4 matvec dispatch (scale+bias dequant × input → output).
    /// cols and groupSize are baked into the pipeline via function constants.
    public mutating func emitAffineMatvec(
        weightsSlot: Int, weightsOffset: UInt64,
        scalesSlot: Int, scalesOffset: UInt64,
        biasesSlot: Int, biasesOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emitAffineMatvecImpl(
            weightsSlot: weightsSlot, weightsOffset: weightsOffset,
            scalesSlot: scalesSlot, scalesOffset: scalesOffset,
            biasesSlot: biasesSlot, biasesOffset: biasesOffset,
            inputSlot: inputSlot, outputSlot: outputSlot,
            rows: rows, cols: cols, groupSize: groupSize,
            comment: comment,
            plannedKernelCandidate: nil
        )
    }

    mutating func emitAffineMatvec(
        weightsSlot: Int, weightsOffset: UInt64,
        scalesSlot: Int, scalesOffset: UInt64,
        biasesSlot: Int, biasesOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil,
        plannedKernelCandidate: SmeltPlannedKernelCandidate
    ) throws -> [String] {
        try emitAffineMatvecImpl(
            weightsSlot: weightsSlot, weightsOffset: weightsOffset,
            scalesSlot: scalesSlot, scalesOffset: scalesOffset,
            biasesSlot: biasesSlot, biasesOffset: biasesOffset,
            inputSlot: inputSlot, outputSlot: outputSlot,
            rows: rows, cols: cols, groupSize: groupSize,
            comment: comment,
            plannedKernelCandidate: plannedKernelCandidate
        )
    }

    private mutating func emitAffineMatvecImpl(
        weightsSlot: Int, weightsOffset: UInt64,
        scalesSlot: Int, scalesOffset: UInt64,
        biasesSlot: Int, biasesOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String?,
        plannedKernelCandidate: SmeltPlannedKernelCandidate?
    ) throws -> [String] {
        let route = fusionPlanner.decodeAffineMatvec(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let specializedPipeline = route.pipeline
        let threadgroupWidth = specializedPipeline == .affineMatvecC10240R2560G128Rows4SG1 ? 128 : 64
        let pipeline = specializedPipeline ?? .affineMatvec
        let buffers = [
            SmeltBufferBinding(slot: weightsSlot, offset: weightsOffset, index: 0),
            SmeltBufferBinding(slot: scalesSlot, offset: scalesOffset, index: 1),
            SmeltBufferBinding(slot: biasesSlot, offset: biasesOffset, index: 2),
            SmeltBufferBinding(slot: inputSlot, index: 3),
            SmeltBufferBinding(slot: outputSlot, index: 4),
        ]
        let constants = specializedPipeline == nil ? [
            SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
        ] : []
        let dispatchStyle = SmeltDispatchStyle.threadgroups(
            width: (rows + route.rowTile - 1) / route.rowTile,
            height: 1, depth: 1,
            tgWidth: threadgroupWidth, tgHeight: 1, tgDepth: 1
        )
        let fcCols = specializedPipeline == nil ? cols : nil
        let fcGroupSize = specializedPipeline == nil ? groupSize : nil
        let dispatch: SmeltDispatch
        if let plannedKernelCandidate {
            dispatch = SmeltDispatch(
                pipeline: pipeline,
                buffers: buffers,
                constants: constants,
                dispatch: dispatchStyle,
                comment: comment,
                plannedKernelCandidate: plannedKernelCandidate,
                fcCols: fcCols,
                fcGroupSize: fcGroupSize
            )
        } else {
            dispatch = SmeltDispatch(
                pipeline: pipeline,
                buffers: buffers,
                constants: constants,
                dispatch: dispatchStyle,
                comment: comment,
                fcCols: fcCols,
                fcGroupSize: fcGroupSize
            )
        }
        return try emit(dispatch)
    }

    mutating func emitFusedDualAffineMatvecAdd(
        firstEntry: SmeltWeightEntry,
        secondEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        firstOutputSlot: Int,
        secondOutputSlot: Int,
        firstResidualOffset: UInt64,
        secondResidualOffset: UInt64,
        rows: Int,
        cols: Int,
        groupSize: Int,
        comment: String?,
        plannedKernelRoute: SmeltPlannedKernelRoute
    ) throws -> [String] {
        guard let firstScales = firstEntry.scalesOffset,
              let firstBiases = firstEntry.biasesOffset,
              let secondScales = secondEntry.scalesOffset,
              let secondBiases = secondEntry.biasesOffset
        else {
            throw SmeltEmitError.unsupported(
                detail: "fused dual affine matvec-add requires affineU4 scales/biases"
            )
        }
        let shape = SmeltKernelShape(rows: rows, cols: cols, groupSize: groupSize)
        guard let geometry = plannedKernelRoute
            .fusedDualAffineMatvecResidualAddLaunchGeometry(expectedShape: shape)
        else {
            throw SmeltEmitError.unsupported(
                detail: "planned fused dual affine matvec-add route has incompatible geometry"
            )
        }
        return try emit(SmeltDispatch(
            pipeline: .fusedAffineMatvecAdd,
            plannedKernelRoute: plannedKernelRoute,
            buffers: [
                SmeltBufferBinding(slot: weightsSlot, offset: firstEntry.offset, index: 0),
                SmeltBufferBinding(slot: weightsSlot, offset: firstScales, index: 1),
                SmeltBufferBinding(slot: weightsSlot, offset: firstBiases, index: 2),
                SmeltBufferBinding(slot: weightsSlot, offset: secondEntry.offset, index: 3),
                SmeltBufferBinding(slot: weightsSlot, offset: secondScales, index: 4),
                SmeltBufferBinding(slot: weightsSlot, offset: secondBiases, index: 5),
                SmeltBufferBinding(slot: inputSlot, index: 6),
                SmeltBufferBinding(slot: firstOutputSlot, index: 7),
                SmeltBufferBinding(slot: secondOutputSlot, index: 8),
                SmeltBufferBinding(slot: weightsSlot, offset: firstResidualOffset, index: 9),
                SmeltBufferBinding(slot: weightsSlot, offset: secondResidualOffset, index: 10),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: geometry.gridWidth(rows: rows),
                height: 1,
                depth: 1,
                tgWidth: geometry.threadgroupWidth,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an affine u4 matvec dispatch with a variable input slot.
    public mutating func emitAffineMatvecVar(
        weightsSlot: Int, weightsOffset: UInt64,
        scalesSlot: Int, scalesOffset: UInt64,
        biasesSlot: Int, biasesOffset: UInt64,
        inputSlotVar: String, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeAffineMatvec(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let specializedPipeline = route.pipeline
        let threadgroupWidth = specializedPipeline == .affineMatvecC10240R2560G128Rows4SG1 ? 128 : 64
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .affineMatvec,
            buffers: [
                SmeltBufferBinding(slot: weightsSlot, offset: weightsOffset, index: 0),
                SmeltBufferBinding(slot: scalesSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(slot: biasesSlot, offset: biasesOffset, index: 2),
                SmeltBufferBinding(variableSlot: inputSlotVar, index: 3),
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
            ] : [],
            dispatch: .threadgroups(
                width: (rows + route.rowTile - 1) / route.rowTile,
                height: 1, depth: 1,
                tgWidth: threadgroupWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            fcCols: specializedPipeline == nil ? cols : nil,
            fcGroupSize: specializedPipeline == nil ? groupSize : nil
        ))
    }

    /// Emit a stripped-mode exact RMS-norm scale-only + affine matvec chain when
    /// a specialized norm-scale affine kernel exists for the target shape.
    public mutating func emitNormScaleAffineMatvecIfPossible(
        normInputSlotVar: String,
        normWeightSlot: Int, normWeightOffset: UInt64,
        normOutputSlot: Int,
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize: Int,
        comment: String? = nil
    ) throws -> [String]? {
        guard cols == 1_536 else { return nil }
        guard weightEntry.dtype == .affineU4,
              let scalesOff = weightEntry.scalesOffset,
              let biasesOff = weightEntry.biasesOffset
        else {
            return nil
        }

        // Specialized routes must see the ENTRY's group size (per-tensor entries may be g32
        // while the spec global may differ) — the planner
        // declines unknown combinations, falling back to generic dispatch.
        let resolvedGroupSize = weightEntry.groupSize ?? groupSize
        guard let route = fusionPlanner.decodeNormScaleAffine(
            rows: rows,
            cols: cols,
            groupSize: resolvedGroupSize
        ) else {
            return nil
        }
        let specializedPipeline = route.pipeline

        let scaleComment = comment.map { "\($0) [scale-only]" } ?? "RMS norm scale only"
        let fusedComment = comment.map { "\($0) [norm-scaled affine]" } ?? "Norm-scaled affine"

        var lines = try emit(SmeltDispatch(
            pipeline: .rmsNormScaleOnlyD1536,
            buffers: [
                SmeltBufferBinding(variableSlot: normInputSlotVar, index: 0),
                SmeltBufferBinding(slot: SmeltFixedSlot.normScaleScratch.rawValue, index: 1),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: 192, tgHeight: 1, tgDepth: 1
            ),
            comment: scaleComment
        ))

        lines += try emit(SmeltDispatch(
            pipeline: specializedPipeline,
            buffers: [
                SmeltBufferBinding(slot: SmeltFixedSlot.normScaleScratch.rawValue, index: 0),
                SmeltBufferBinding(variableSlot: normInputSlotVar, index: 1),
                SmeltBufferBinding(slot: normWeightSlot, offset: normWeightOffset, index: 2),
                SmeltBufferBinding(slot: normOutputSlot, index: 3),
                SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 4),
                SmeltBufferBinding(slot: weightsSlot, offset: scalesOff, index: 5),
                SmeltBufferBinding(slot: weightsSlot, offset: biasesOff, index: 6),
                SmeltBufferBinding(slot: outputSlot, index: 7),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: (rows + route.rowTile - 1) / route.rowTile,
                height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: fusedComment
        ))

        return lines
    }

    /// Emit a stripped-mode exact RMS-norm scale-only + affine gate/up + GeGLU
    /// chain when a tuned norm-scale GeGLU kernel exists for the target shape.
    public mutating func emitNormScaleAffineGateUpGeGLUIfPossible(
        normInputSlotVar: String,
        normWeightSlot: Int, normWeightOffset: UInt64,
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize: Int,
        comment: String? = nil
    ) throws -> [String]? {
        guard cols == 1_536 else { return nil }
        guard gateEntry.dtype == .affineU4,
              let gateScalesOff = gateEntry.scalesOffset,
              let gateBiasesOff = gateEntry.biasesOffset,
              upEntry.dtype == .affineU4,
              let upScalesOff = upEntry.scalesOffset,
              let upBiasesOff = upEntry.biasesOffset
        else {
            return nil
        }

        // Entry-resolved group size; both entries must agree (see
        // emitNormScaleAffineMatvecIfPossible).
        let resolvedGroupSize = gateEntry.groupSize ?? groupSize
        guard (upEntry.groupSize ?? groupSize) == resolvedGroupSize else {
            return nil
        }
        guard let route = fusionPlanner.decodeNormScaleGeGLU(
            rows: rows,
            cols: cols,
            groupSize: resolvedGroupSize
        ) else {
            return nil
        }
        let specializedPipeline = route.pipeline

        let scaleComment = comment.map { "\($0) [scale-only]" } ?? "RMS norm scale only"
        let fusedComment = comment.map { "\($0) [norm-scaled GeGLU]" } ?? "Norm-scaled GeGLU"

        var lines = try emit(SmeltDispatch(
            pipeline: .rmsNormScaleOnlyD1536,
            buffers: [
                SmeltBufferBinding(variableSlot: normInputSlotVar, index: 0),
                SmeltBufferBinding(slot: SmeltFixedSlot.normScaleScratch.rawValue, index: 1),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: 192, tgHeight: 1, tgDepth: 1
            ),
            comment: scaleComment
        ))

        lines += try emit(SmeltDispatch(
            pipeline: specializedPipeline,
            buffers: [
                SmeltBufferBinding(slot: SmeltFixedSlot.normScaleScratch.rawValue, index: 0),
                SmeltBufferBinding(variableSlot: normInputSlotVar, index: 1),
                SmeltBufferBinding(slot: normWeightSlot, offset: normWeightOffset, index: 2),
                SmeltBufferBinding(slot: weightsSlot, offset: gateEntry.offset, index: 3),
                SmeltBufferBinding(slot: weightsSlot, offset: gateScalesOff, index: 4),
                SmeltBufferBinding(slot: weightsSlot, offset: gateBiasesOff, index: 5),
                SmeltBufferBinding(slot: weightsSlot, offset: upEntry.offset, index: 6),
                SmeltBufferBinding(slot: weightsSlot, offset: upScalesOff, index: 7),
                SmeltBufferBinding(slot: weightsSlot, offset: upBiasesOff, index: 8),
                SmeltBufferBinding(slot: outputSlot, index: 9),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: (rows + 3) / 4, height: 1, depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: fusedComment
        ))

        return lines
    }

    /// Emit an RMS norm (1-pass weighted) dispatch.
    public mutating func emitRMSNorm1PW(
        inputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        outputSlot: Int,
        dim: Int, eps: Float,
        inputOffset: UInt64 = 0,
        outputOffset: UInt64 = 0,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeRMSNorm(
            dim: dim,
            eps: eps
        )
        let specializedPipeline = route.pipeline
        let specializedThreads = route.threadgroupWidth
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .rmsNorm1PW,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, offset: inputOffset, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, offset: outputOffset, index: 2),
            ],
            constants: specializedPipeline == nil
                ? [
                    SmeltConstantBinding(expression: "\(dim)", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 4),
                ]
                : [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: specializedThreads ?? tgWidth(dim),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    public mutating func emitRMSNorm1PWAddVarIfPossible(
        inputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        residualSlotVar: String,
        outputSlotVar: String,
        dim: Int,
        eps: Float,
        comment: String? = nil
    ) throws -> [String]? {
        guard eps == 1e-6 else { return nil }
        let pipeline: SmeltPipeline
        let tgWidth: Int
        switch dim {
        case 256:
            pipeline = .rmsNorm1PWD256Add
            tgWidth = 256
        case 1_536:
            pipeline = .rmsNorm1PWD1536Add
            tgWidth = 192
        case 2_560:
            pipeline = .rmsNorm1PWD2560Add
            tgWidth = 320
        default:
            return nil
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(variableSlot: residualSlotVar, index: 2),
                SmeltBufferBinding(variableSlot: outputSlotVar, index: 3),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: tgWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit RMSNorm from an FP32 input buffer back to a half output buffer.
    public mutating func emitRMSNorm1PWFromFP32(
        inputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        outputSlot: Int,
        dim: Int, eps: Float,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .rmsNorm1PWFromFP32,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(dim)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 4),
            ],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: tgWidth(dim), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an elementwise add dispatch (a + b → output).
    public mutating func emitElementwiseAdd(
        inputASlot: Int, inputBSlot: Int, outputSlot: Int,
        count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(slot: inputASlot, index: 0),
                SmeltBufferBinding(slot: inputBSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit elementwise add for fixed slots with literal byte offsets.
    public mutating func emitElementwiseAddWithOffsets(
        inputASlot: Int, inputAOffset: UInt64,
        inputBSlot: Int, inputBOffset: UInt64,
        outputSlot: Int, outputOffset: UInt64,
        count: Int,
        comment: String? = nil,
        minSeqLen: Int? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(slot: inputASlot, offset: inputAOffset, index: 0),
                SmeltBufferBinding(slot: inputBSlot, offset: inputBOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, offset: outputOffset, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minSeqLen: minSeqLen
        ))
    }

    /// Emit a sequence-batched bias add: output[b, row] = input[b, row] + bias[row].
    public mutating func emitBatchedBiasAdd(
        inputSlot: Int,
        biasSlot: Int, biasOffset: UInt64,
        outputSlot: Int,
        rows: Int,
        batchSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .projectionBiasAddBatched,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: biasSlot, offset: biasOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: rows, height: batchSize, depth: 1,
                tgWidth: min(rows, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLen
        ))
    }

    /// Snake activation: y = x + sin(alpha·x)² / alpha, per-channel learnable alpha.
    /// Extracted from the parked Kokoro TTS work (ONNX gate 14).
    public mutating func emitSnakeActivation(
        inputSlot: Int, outputSlot: Int,
        alphaSlot: Int, alphaOffset: UInt64,
        channels: Int, length: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .snakeActivation,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: alphaSlot, offset: alphaOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(channels)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(length)", type: .uint32, index: 4),
            ],
            dispatch: .threads(
                width: length, height: channels, depth: 1,
                tgWidth: min(length, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a general 1D convolution (padding/stride/dilation/groups) dispatch. FP16,
    /// contiguous [C, L] buffers (input row stride L_in, output row stride L_out).
    /// `has_bias` is packed into the high bit of the L_in constant.
    /// Extracted from the parked Kokoro TTS work (ONNX gate 18);
    /// the optional buf_stride override was dropped to fit the 8-constant dispatch cap.
    public mutating func emitConv1d(
        inputSlot: Int, outputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        biasSlot: Int, biasOffset: UInt64,
        cIn: Int, cOut: Int, kernel: Int,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1,
        lengthIn: Int, hasBias: Bool = true,
        comment: String? = nil
    ) throws -> [String] {
        let lOut = (lengthIn + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        let lInPacked = hasBias ? (lengthIn | 0x8000_0000) : lengthIn
        return try emit(SmeltDispatch(
            pipeline: .conv1dForward,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: biasSlot, offset: biasOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(cIn)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(cOut)", type: .uint32, index: 5),
                SmeltConstantBinding(expression: "\(kernel)", type: .uint32, index: 6),
                SmeltConstantBinding(expression: "\(stride)", type: .uint32, index: 7),
                SmeltConstantBinding(expression: "\(padding)", type: .uint32, index: 8),
                SmeltConstantBinding(expression: "\(dilation)", type: .uint32, index: 9),
                SmeltConstantBinding(expression: "\(groups)", type: .uint32, index: 10),
                SmeltConstantBinding(expression: "\(lInPacked)", type: .uint32, index: 11),
            ],
            dispatch: .threads(
                width: lOut, height: cOut, depth: 1,
                tgWidth: min(lOut, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a transposed conv1d (upsampling) dispatch. FP16.
    /// Extracted from the parked Kokoro TTS work (ONNX gate 15).
    public mutating func emitConvTranspose1d(
        inputSlot: Int, outputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        biasSlot: Int, biasOffset: UInt64,
        cIn: Int, cOut: Int, kernel: Int,
        stride: Int = 1, padding: Int = 0,
        lengthIn: Int,
        comment: String? = nil
    ) throws -> [String] {
        let lOut = (lengthIn - 1) * stride - 2 * padding + kernel
        return try emit(SmeltDispatch(
            pipeline: .convTranspose1d,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: biasSlot, offset: biasOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(cIn)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(cOut)", type: .uint32, index: 5),
                SmeltConstantBinding(expression: "\(kernel)", type: .uint32, index: 6),
                SmeltConstantBinding(expression: "\(stride)", type: .uint32, index: 7),
                SmeltConstantBinding(expression: "\(padding)", type: .uint32, index: 8),
                SmeltConstantBinding(expression: "\(lengthIn)", type: .uint32, index: 9),
            ],
            dispatch: .threads(
                width: lOut, height: cOut, depth: 1,
                tgWidth: min(lOut, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a LayerNorm dispatch (standard, not RMS): normalize each of `rows`
    /// rows of [rows, dim] using weight and bias. One threadgroup per row — the
    /// kernel takes its row from `threadgroup_position_in_grid` (stride `dim`), so
    /// `width: rows` normalizes every row, not just row 0.
    /// Extracted from the parked Kokoro TTS work (ONNX gate 17).
    public mutating func emitLayerNorm(
        inputSlot: Int, outputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        biasSlot: Int, biasOffset: UInt64,
        dim: Int, rows: Int = 1, eps: Float = 1e-5,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .layerNorm,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: biasSlot, offset: biasOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(dim)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 5),
            ],
            dispatch: .threadgroups(
                width: rows, height: 1, depth: 1,
                tgWidth: min(dim, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// fp16-ACTIVATION dense matvec PIPELINE for a weight dtype: .fp16 → fp16_matvec; .bf16/.fp32 →
    /// the U2 hand-written fp16_matvec_{bf16,fp32}w kernels (same threadgroup-reduction shape, the
    /// weight load widened to float). The gateway (MatvecKernelTable.select) only ever yields
    /// `.dense(.fp16/.bf16/.fp32)`, so any other dtype is a caller bug — exhaustive, no silent
    /// default (the dtype-dispatch-safety invariant). Shared by the fixed-slot (emitFP16Matvec),
    /// variable-slot (emitFP16MatvecVar), and per-batch prefill (emitBatchedMatmul) lowerings.
    static func fp16DenseMatvecPipeline(_ weight: SmeltDType) throws -> SmeltPipeline {
        switch weight {
        case .fp16: return .fp16Matvec
        case .bf16: return .fp16MatvecBF16W
        case .fp32: return .fp16MatvecFP32W
        case .affineU4, .binary1, .ternary2, .u4Lut, .turboQuantH, .int32, .raw:
            throw SmeltEmitError.unsupported(
                detail: "fp16DenseMatvecPipeline: \(weight.rawValue) is not a dense matvec weight")
        }
    }

    /// Emit an fp16-ACTIVATION dense matvec dispatch (dense weight × fp16 input → fp16 output).
    /// `weight` selects the kernel via `fp16DenseMatvecPipeline` (fp16 / bf16 / fp32 dense).
    public mutating func emitFP16Matvec(
        weightSlot: Int, weightOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int,
        weight: SmeltDType = .fp16,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: try Self.fp16DenseMatvecPipeline(weight),
            buffers: [
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 0),
                SmeltBufferBinding(slot: inputSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 3),
            ],
            dispatch: .threadgroups(
                width: rows, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an FP16 dense matvec that keeps the output accumulator in FP32.
    public mutating func emitFP16MatvecFP32Out(
        weightSlot: Int, weightOffset: UInt64,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .fp16MatvecFP32Out,
            buffers: [
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 0),
                SmeltBufferBinding(slot: inputSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 3),
            ],
            dispatch: .threadgroups(
                width: rows, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an FP16 matvec dispatch with a variable input slot.
    /// Variable-input-slot form of `emitFP16Matvec` (same kernels; the input binds a named slot).
    public mutating func emitFP16MatvecVar(
        weightSlot: Int, weightOffset: UInt64,
        inputSlotVar: String, outputSlot: Int,
        rows: Int, cols: Int,
        weight: SmeltDType = .fp16,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: try Self.fp16DenseMatvecPipeline(weight),
            buffers: [
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 0),
                SmeltBufferBinding(variableSlot: inputSlotVar, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 3),
            ],
            dispatch: .threadgroups(
                width: rows, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an embedding gather dispatch.
    public mutating func emitEmbeddingGather(
        weightSlot: Int, weightOffset: UInt64,
        tokenIdSlot: Int, outputSlot: Int,
        hiddenSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .embeddingGather,
            buffers: [
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 0),
                SmeltBufferBinding(slot: tokenIdSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(hiddenSize)", type: .uint32, index: 3
                ),
            ],
            dispatch: .threads(
                width: hiddenSize, height: 1, depth: 1,
                tgWidth: min(hiddenSize, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit an argmax dispatch (single threadgroup).
    public mutating func emitArgmax(
        inputSlot: Int, outputSlot: Int,
        count: Int,
        inputOffset: UInt64 = 0,
        outputOffset: UInt64 = 0,
        minSeqLen: Int? = nil,
        comment: String? = nil
    ) throws -> [String] {
        if count >= 65_536 && count <= 262_144 {
            let chunkSize = 2_048
            let partialCount = (count + chunkSize - 1) / chunkSize
            var lines: [String] = []
            lines += try emit(SmeltDispatch(
                pipeline: .argmaxFP16Partials,
                buffers: [
                    SmeltBufferBinding(
                        slot: inputSlot,
                        offset: inputOffset,
                        index: 0
                    ),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.normScaleScratch.rawValue,
                        index: 1
                    ),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 2),
                    SmeltConstantBinding(expression: "\(chunkSize)", type: .uint32, index: 3),
                ],
                dispatch: .threadgroups(
                    width: partialCount, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment.map { "\($0) partials" },
                minSeqLen: minSeqLen
            ))
            lines += try emit(SmeltDispatch(
                pipeline: .argmaxKeyReduce,
                buffers: [
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.normScaleScratch.rawValue,
                        index: 0
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot,
                        offset: outputOffset,
                        index: 1
                    ),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(partialCount)", type: .uint32, index: 2),
                ],
                dispatch: .threadgroups(
                    width: 1, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment.map { "\($0) reduce" },
                minSeqLen: minSeqLen
            ))
            return lines
        }

        return try emit(SmeltDispatch(
            pipeline: .argmaxFP16,
            buffers: [
                SmeltBufferBinding(
                    slot: inputSlot,
                    offset: inputOffset,
                    index: 0
                ),
                SmeltBufferBinding(
                    slot: outputSlot,
                    offset: outputOffset,
                    index: 1
                ),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 2),
            ],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: tgWidth(count), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minSeqLen: minSeqLen
        ))
    }

    public mutating func emitGeGLU(
        gateSlot: Int,
        upSlot: Int,
        outputSlot: Int,
        count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .gegluFused,
            buffers: [
                SmeltBufferBinding(slot: gateSlot, index: 0),
                SmeltBufferBinding(slot: upSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    public mutating func emitGeGLU(
        gateSlot: Int,
        upSlot: Int,
        upOffset: UInt64,
        outputSlot: Int,
        count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .gegluFused,
            buffers: [
                SmeltBufferBinding(slot: gateSlot, index: 0),
                SmeltBufferBinding(slot: upSlot, offset: upOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    public mutating func emitScalarMul(
        inputSlot: Int,
        outputSlot: Int,
        scalar: Float,
        count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .scalarMul,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: outputSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(scalar)", type: .float32, index: 2),
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    public mutating func emitScalarMulWeightVar(
        inputSlotVar: String,
        weightSlot: Int,
        weightOffset: UInt64,
        outputSlotVar: String,
        count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .scalarMulWeight,
            buffers: [
                SmeltBufferBinding(variableSlot: inputSlotVar, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(variableSlot: outputSlotVar, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    public mutating func emitLogitCap(
        inputSlot: Int,
        outputSlot: Int,
        count: Int,
        cap: Float,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .logitCap,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: outputSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 2),
                SmeltConstantBinding(expression: "\(cap)", type: .float32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a quantized embedding gather (u4+LUT dequant on the fly).
    public mutating func emitLUTEmbeddingGather(
        indicesSlot: Int, indicesOffset: UInt64,
        lutSlot: Int, lutOffset: UInt64,
        tokenIdSlot: Int, outputSlot: Int,
        hiddenSize: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .lutEmbeddingGather,
            buffers: [
                SmeltBufferBinding(slot: indicesSlot, offset: indicesOffset, index: 0),
                SmeltBufferBinding(slot: lutSlot, offset: lutOffset, index: 1),
                SmeltBufferBinding(slot: tokenIdSlot, index: 2),
                SmeltBufferBinding(slot: outputSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(hiddenSize)", type: .uint32, index: 4
                ),
                SmeltConstantBinding(
                    expression: "\(groupSize)", type: .uint32, index: 5
                ),
            ],
            dispatch: .threads(
                width: hiddenSize / 2, height: 1, depth: 1,
                tgWidth: min(hiddenSize / 2, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit a quantized embedding gather (u4+affine dequant on the fly).
    public mutating func emitAffineEmbeddingGather(
        weightsSlot: Int, weightsOffset: UInt64,
        scalesSlot: Int, scalesOffset: UInt64,
        biasesSlot: Int, biasesOffset: UInt64,
        tokenIdSlot: Int, outputSlot: Int,
        hiddenSize: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .affineEmbeddingGather,
            buffers: [
                SmeltBufferBinding(slot: weightsSlot, offset: weightsOffset, index: 0),
                SmeltBufferBinding(slot: scalesSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(slot: biasesSlot, offset: biasesOffset, index: 2),
                SmeltBufferBinding(slot: tokenIdSlot, index: 3),
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(hiddenSize)", type: .uint32, index: 5
                ),
                SmeltConstantBinding(
                    expression: "\(groupSize)", type: .uint32, index: 6
                ),
            ],
            dispatch: .threads(
                width: hiddenSize / 2, height: 1, depth: 1,
                tgWidth: min(hiddenSize / 2, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Gather one or more rows from canonical signed g128 storage. A grid-y
    /// batch uses one token id and one output row per y coordinate.
    public mutating func emitSignedEmbeddingGather(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        tokenIdSlot: Int,
        tokenIdOffset: UInt64 = 0,
        outputSlot: Int,
        outputOffset: UInt64 = 0,
        hiddenSize: Int,
        batchSize: Int = 1,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        minSeqLen: Int? = nil,
        comment: String? = nil
    ) throws -> [String] {
        guard let scalesOffset = weightEntry.scalesOffset,
              weightEntry.groupSize == 128,
              weightEntry.paddedCols == hiddenSize,
              hiddenSize % 128 == 0
        else {
            throw SmeltEmitError.unsupported(
                detail: "signed embedding '\(weightEntry.name)' requires exact g128 geometry"
            )
        }
        let pipeline: SmeltPipeline
        switch weightEntry.dtype {
        case .binary1: pipeline = .signedBinaryEmbeddingGatherG128
        case .ternary2: pipeline = .signedTernaryEmbeddingGatherG128
        case .fp16, .fp32, .bf16, .int32, .u4Lut, .affineU4, .turboQuantH, .raw:
            throw SmeltEmitError.unsupported(
                detail: "signed embedding gather received \(weightEntry.dtype.rawValue) weight"
            )
        }
        return try emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOffset, index: 1),
                SmeltBufferBinding(slot: tokenIdSlot, offset: tokenIdOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, offset: outputOffset, index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(hiddenSize)", type: .uint32, index: 4),
            ],
            dispatch: .threads(
                width: hiddenSize, height: batchSize, depth: 1,
                tgWidth: min(hiddenSize, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: dynamicGridH,
            minSeqLen: minSeqLen
        ))
    }

    /// Emit `tqh_matvec_prepare_input` + `tqh_matvec` as a pair.
    /// The X_hat scratch buffer (numGroups × G fp32, i.e.
    /// `ceil(cols/128) * 128 * 4` bytes for decode; multiplied by
    /// batchSize for the prefill-batched variant) must be
    /// pre-allocated by the caller and passed via `xHatScratchSlot`.
    /// The caller is also responsible for ensuring no other dispatch
    /// reads/writes the scratch slot between these two emits. See
    /// the metal kernel source for the two-kernel math decomposition.
    ///
    /// `batchSize > 1` OR `dynamicGridH != nil` selects the
    /// batched kernel pair (`tqh_matvec_prepare_input_batched` /
    /// `tqh_matvec_batched`); both axes are b-strided via `gid.y`.
    /// `inputOffsetExpression` is decode-only — the batched path
    /// computes its own b * cols offset internally.
    public mutating func emitTQHMatvec(
        codesSlot: Int, codesOffset: UInt64,
        codebookSlot: Int, codebookOffset: UInt64,
        inputSlot: Int, inputOffset: UInt64 = 0,
        inputOffsetExpression: String? = nil,
        xHatScratchSlot: Int, xHatScratchOffset: UInt64 = 0,
        outputSlot: Int, outputOffset: UInt64 = 0,
        rows: Int, cols: Int, codesPerRow: Int,
        batchSize: Int = 1,
        minSeqLen: Int? = nil,
        dynamicGridH: SmeltDynamicGridDimension? = nil,
        comment: String? = nil
    ) throws -> [String] {
        let isBatched = batchSize > 1 || dynamicGridH != nil
        if isBatched, inputOffsetExpression != nil {
            throw SmeltEmitError.unsupported(
                detail: "emitTQHMatvec: inputOffsetExpression is "
                + "incompatible with batched dispatch (the batched "
                + "kernel computes its own b * cols offset via gid.y)"
            )
        }
        let preparePipeline: SmeltPipeline = isBatched ? .tqhMatvecPrepareInputBatched : .tqhMatvecPrepareInput
        let matvecPipeline: SmeltPipeline = isBatched ? .tqhMatvecBatched : .tqhMatvec

        let g = 128
        let numGroups = SmeltTurboQuantHCodec.numGroups(cols: cols, groupSize: g)
        let prepareWidth = numGroups * g
        let inputBinding: SmeltBufferBinding =
            if let inputOffsetExpression {
                SmeltBufferBinding(
                    slot: inputSlot,
                    offsetExpression: inputOffsetExpression,
                    index: 0
                )
            } else {
                SmeltBufferBinding(
                    slot: inputSlot, offset: inputOffset, index: 0
                )
            }
        let prepareCommentTag = isBatched ? "prepare X_hat batched" : "prepare X_hat"
        var lines = try emit(SmeltDispatch(
            pipeline: preparePipeline,
            buffers: [
                inputBinding,
                SmeltBufferBinding(slot: xHatScratchSlot, offset: xHatScratchOffset, index: 1),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 2
                ),
            ],
            dispatch: .threads(
                width: prepareWidth, height: batchSize, depth: 1,
                tgWidth: g, tgHeight: 1, tgDepth: 1
            ),
            comment: comment.map { "\($0) (\(prepareCommentTag))" },
            dynamicGridH: dynamicGridH,
            minSeqLen: minSeqLen
        ))
        // SG4 design — 4 simdgroups per TG × 4 rows per SG = 16 rows
        // per TG, 128 threads per TG. Coalesced reads within each
        // row. Same layout for both batched (prefill verify) and
        // unbatched (decode/refresh) paths.
        let rowsPerTG = 16
        let threadsPerTG = 128
        let matvecTGWidth = threadsPerTG
        let numTGs = (rows + rowsPerTG - 1) / rowsPerTG
        let matvecWidth = numTGs * threadsPerTG
        lines += try emit(SmeltDispatch(
            pipeline: matvecPipeline,
            buffers: [
                SmeltBufferBinding(slot: codesSlot, offset: codesOffset, index: 0),
                SmeltBufferBinding(slot: codebookSlot, offset: codebookOffset, index: 1),
                SmeltBufferBinding(slot: xHatScratchSlot, offset: xHatScratchOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, offset: outputOffset, index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(rows)", type: .uint32, index: 4
                ),
                SmeltConstantBinding(
                    expression: "\(cols)", type: .uint32, index: 5
                ),
                SmeltConstantBinding(
                    expression: "\(codesPerRow)", type: .uint32, index: 6
                ),
            ],
            dispatch: .threads(
                width: matvecWidth, height: batchSize, depth: 1,
                tgWidth: matvecTGWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: dynamicGridH,
            minSeqLen: minSeqLen
        ))
        return lines
    }

    /// TurboQuant-H embedding gather. The kernel is per-thread
    /// independent (no threadgroup memory), one thread per output
    /// column. Grid is rounded up to num_groups * G so partial
    /// final groups still dispatch a full group of threads; the
    /// kernel's `pos < cols` guard suppresses out-of-bounds writes.
    /// `tokenIdOffset` and `outputOffset` let prefill-time callers
    /// unroll B per-token dispatches against shared
    /// tokenIdsBatch / hiddenA buffers without inlining a
    /// SmeltDispatch literal.
    public mutating func emitTurboQuantHEmbeddingGather(
        codesSlot: Int, codesOffset: UInt64,
        codebookSlot: Int, codebookOffset: UInt64,
        tokenIdSlot: Int, tokenIdOffset: UInt64 = 0,
        outputSlot: Int, outputOffset: UInt64 = 0,
        hiddenSize: Int, codesPerRow: Int,
        minSeqLen: Int? = nil,
        comment: String? = nil
    ) throws -> [String] {
        let g = 128
        let numGroups = SmeltTurboQuantHCodec.numGroups(
            cols: hiddenSize, groupSize: g
        )
        let dispatchWidth = numGroups * g
        return try emit(SmeltDispatch(
            pipeline: .tqhEmbeddingGather,
            buffers: [
                SmeltBufferBinding(slot: codesSlot, offset: codesOffset, index: 0),
                SmeltBufferBinding(slot: codebookSlot, offset: codebookOffset, index: 1),
                SmeltBufferBinding(slot: tokenIdSlot, offset: tokenIdOffset, index: 2),
                SmeltBufferBinding(slot: outputSlot, offset: outputOffset, index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(codesPerRow)", type: .uint32, index: 4
                ),
                SmeltConstantBinding(
                    expression: "\(hiddenSize)", type: .uint32, index: 5
                ),
            ],
            dispatch: .threads(
                width: dispatchWidth, height: 1, depth: 1,
                tgWidth: min(dispatchWidth, 128), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minSeqLen: minSeqLen
        ))
    }

    // MARK: - Variable-slot helpers (double-buffer pattern)

    /// Emit RMS norm with a variable input slot (e.g. "cur" for double-buffer).
    public mutating func emitRMSNorm1PWVar(
        inputSlotVar: String,
        weightSlot: Int, weightOffset: UInt64,
        outputSlot: Int,
        dim: Int, eps: Float,
        comment: String? = nil
    ) throws -> [String] {
        let route = fusionPlanner.decodeRMSNormVariableInput(
            dim: dim,
            eps: eps
        )
        let specializedPipeline = route.pipeline
        return try emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .rmsNorm1PW,
            buffers: [
                SmeltBufferBinding(variableSlot: inputSlotVar, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: specializedPipeline == nil
                ? [
                    SmeltConstantBinding(expression: "\(dim)", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 4),
                ]
                : [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: route.threadgroupWidth ?? tgWidth(dim),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// Emit elementwise add with variable input/output slots (double-buffer).
    public mutating func emitElementwiseAddVar(
        inputAVar: String, inputBSlot: Int,
        outputVar: String, count: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emit(SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(variableSlot: inputAVar, index: 0),
                SmeltBufferBinding(slot: inputBSlot, index: 1),
                SmeltBufferBinding(variableSlot: outputVar, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(count)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: count, height: 1, depth: 1,
                tgWidth: min(count, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    // MARK: - Weight-aware matvec

    /// Consult the ONE matvec gateway for an fp16-activation site's kernel FAMILY, mapping the
    /// gateway's throw onto this emitter's public error type (`SmeltEmitError`). This is the
    /// shared fp16-act family-selection entry point: SmeltCodeEmitter's own switches
    /// (emitMatvec / emitMatvecVar) AND the PrefillEmitter helpers that hold an `inout
    /// SmeltCodeEmitter` route their dtype→family decision through here, so no fp16-act consumer
    /// can hand-pick fp16 or drift from the fp32-act talker path. TopLevelEmitter/DeltaNet
    /// bypasses route in later units of docs/dtype-building-blocks-plan.md, and a SwiftSyntax
    /// no-bypass lint then locks "only `select()` may pick a matvec family" as the enforced
    /// invariant. The binding, constants, grids and specialization stay local (byte-identical).
    func matvecFamily(
        _ entry: SmeltWeightEntry,
        shape: MatvecKernelTable.Shape,
        output: MatvecKernelTable.Output,
        slot: MatvecKernelTable.Slot
    ) throws -> MatvecKernelTable.Family {
        do {
            return try MatvecKernelTable.select(MatvecKernelTable.Cell(
                activation: .fp16, weight: entry.dtype,
                shape: shape, fusion: .none, output: output, slot: slot
            ))
        } catch let e as MatvecKernelTable.SelectError {
            throw SmeltEmitError.unsupported(detail: String(describing: e))
        }
    }

    /// Non-throwing family PROBE for OPTIONAL-FUSION sites only. A fused emitter (DeltaNet A+B
    /// dual, the gate+up FFN) uses its fused kernel iff BOTH weights authorize the SAME fused
    /// family, and otherwise emits separate (gateway-routed) matvecs. This probes the FUSED cell
    /// (fusion-aware — so "a generic affine GEMM exists" cannot stand in for "a fused gate-up
    /// kernel exists", the plan's authorization invariant) and returns nil when the gateway
    /// REJECTS it: the caller's `else` then emits the unfused path, whose own matvec emit
    /// re-raises any loud `.missing` (e.g. a bf16/fp32 weight). It catches ONLY
    /// `MatvecKernelTable.SelectError` — never masking another failure. NOT for sites already
    /// committed to fused lowering (those stay loud/assertive). Still routes the dtype→family
    /// decision through the ONE gateway, so it is no-bypass-lint-clean.
    func optionalFusedFamily(
        _ entry: SmeltWeightEntry,
        shape: MatvecKernelTable.Shape,
        fusion: MatvecKernelTable.Fusion,
        output: MatvecKernelTable.Output = .fp16,
        slot: MatvecKernelTable.Slot = .fixed
    ) -> MatvecKernelTable.Family? {
        do {
            return try MatvecKernelTable.select(MatvecKernelTable.Cell(
                activation: .fp16, weight: entry.dtype,
                shape: shape, fusion: fusion, output: output, slot: slot
            ))
        } catch is MatvecKernelTable.SelectError {
            // The (dtype, shape, fusion) has no registered fused kernel — not fusable here. The
            // caller emits the unfused path; loudness for real holes surfaces there.
            return nil
        } catch {
            // select() only throws SelectError; any other error is a programming fault, not a
            // "not fusable" signal — trap loudly rather than silently disabling fusion.
            preconditionFailure(
                "optionalFusedFamily: unexpected non-SelectError from select(): \(error)")
        }
    }

    /// Non-throwing GROUP probe for optional-fusion sites with TWO weights (DeltaNet A+B, fused
    /// K+V, gate+up). Returns the SHARED fused family iff BOTH weights authorize the SAME one for
    /// (shape, fusion), else nil — so a fused site checks ONE family ("both agree") and can never
    /// half-express the dual-weight invariant (fuse with a mismatched pair). Built on
    /// optionalFusedFamily, so it routes through the ONE gateway and stays no-bypass-lint-clean.
    func bothFusedFamily(
        _ a: SmeltWeightEntry, _ b: SmeltWeightEntry,
        shape: MatvecKernelTable.Shape,
        fusion: MatvecKernelTable.Fusion,
        output: MatvecKernelTable.Output = .fp16,
        slot: MatvecKernelTable.Slot = .fixed
    ) -> MatvecKernelTable.Family? {
        let fa = optionalFusedFamily(a, shape: shape, fusion: fusion, output: output, slot: slot)
        let fb = optionalFusedFamily(b, shape: shape, fusion: fusion, output: output, slot: slot)
        guard let shared = fa, fb == fa else { return nil }
        return shared
    }

    /// Emit the correct matvec kernel based on weight dtype (LUT vs FP16).
    public mutating func emitMatvec(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        try emitMatvecImpl(
            weightEntry: weightEntry,
            weightsSlot: weightsSlot,
            inputSlot: inputSlot,
            outputSlot: outputSlot,
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            comment: comment,
            plannedKernelCandidate: nil
        )
    }

    mutating func emitMatvec(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil,
        plannedKernelCandidate: SmeltPlannedKernelCandidate
    ) throws -> [String] {
        try emitMatvecImpl(
            weightEntry: weightEntry,
            weightsSlot: weightsSlot,
            inputSlot: inputSlot,
            outputSlot: outputSlot,
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            comment: comment,
            plannedKernelCandidate: plannedKernelCandidate
        )
    }

    mutating func emitMatvec(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil,
        plannedKernelRoute: SmeltPlannedKernelRoute?
    ) throws -> [String] {
        if let plannedKernelRoute {
            return try emitMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputSlot: inputSlot,
                outputSlot: outputSlot,
                rows: rows,
                cols: cols,
                groupSize: groupSize,
                comment: comment,
                plannedKernelCandidate: plannedKernelRoute.candidate
            )
        }
        return try emitMatvec(
            weightEntry: weightEntry,
            weightsSlot: weightsSlot,
            inputSlot: inputSlot,
            outputSlot: outputSlot,
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            comment: comment
        )
    }

    private mutating func emitMatvecImpl(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String?,
        plannedKernelCandidate: SmeltPlannedKernelCandidate?
    ) throws -> [String] {
        // Family is chosen by the ONE gateway (MatvecKernelTable.select), not an inline dtype
        // switch — so this fp16-act path and the fp32-act talker path can never drift on which
        // dtypes compile, and an unfilled cell is a LOUD throw, never a silent fp16 fallback.
        // The binding/group-size logic below stays local and verbatim (byte-identical).
        // Group size resolves from the ENTRY when it carries one — some entries mix per-tensor
        // group sizes (e.g. g32) with the spec's global.
        let resolvedGroupSize = weightEntry.groupSize ?? groupSize
        switch try matvecFamily(weightEntry, shape: .gemv, output: .fp16, slot: .fixed) {
        case .lutU4:
            guard let lutOff = weightEntry.lutOffset else {
                throw SmeltEmitError.unsupported(
                    detail: "u4Lut weight '\(weightEntry.name)' has no lutOffset"
                )
            }
            return try emitFusedLUTMatvec(
                indicesSlot: weightsSlot, indicesOffset: weightEntry.offset,
                lutSlot: weightsSlot, lutOffset: lutOff,
                inputSlot: inputSlot, outputSlot: outputSlot,
                rows: rows, cols: cols, groupSize: resolvedGroupSize,
                comment: comment
            )
        case .affineU4:
            guard let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset
            else {
                throw SmeltEmitError.unsupported(
                    detail: "affineU4 weight '\(weightEntry.name)' has no "
                        + "scales/biases offsets"
                )
            }
            return try emitAffineMatvecImpl(
                weightsSlot: weightsSlot, weightsOffset: weightEntry.offset,
                scalesSlot: weightsSlot, scalesOffset: scalesOff,
                biasesSlot: weightsSlot, biasesOffset: biasesOff,
                inputSlot: inputSlot, outputSlot: outputSlot,
                rows: rows, cols: cols, groupSize: resolvedGroupSize,
                comment: comment,
                plannedKernelCandidate: plannedKernelCandidate
            )
        case .binary1, .ternary2:
            return try emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(slot: inputSlot, index: 2),
                outputBinding: SmeltBufferBinding(slot: outputSlot, index: 3),
                rows: rows,
                cols: cols,
                comment: comment
            )
        case .tqh:
            guard let codebookOff = weightEntry.codebookOffset,
                  let codesPerRow = weightEntry.packedRowStride
            else {
                throw SmeltEmitError.unsupported(
                    detail: "turboQuantH weight '\(weightEntry.name)' has no "
                        + "codebook offset / packed row stride"
                )
            }
            return try emitTQHMatvec(
                codesSlot: weightsSlot, codesOffset: weightEntry.offset,
                codebookSlot: weightsSlot, codebookOffset: codebookOff,
                inputSlot: inputSlot,
                xHatScratchSlot: SmeltFixedSlot.tqhMatvecXHatBuf.rawValue,
                outputSlot: outputSlot,
                rows: rows, cols: cols, codesPerRow: codesPerRow,
                comment: comment
            )
        case .dense(let dt):
            // fp16-act dense: .fp16 → fp16_matvec, .bf16/.fp32 → the U2 fp16_matvec_{bf16,fp32}w
            // kernels (emitFP16Matvec picks the pipeline; the gateway authorizes only these three).
            return try emitFP16Matvec(
                weightSlot: weightsSlot, weightOffset: weightEntry.offset,
                inputSlot: inputSlot, outputSlot: outputSlot,
                rows: rows, cols: cols, weight: dt,
                comment: comment
            )
        }
    }

    public mutating func emitMatvecVar(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlotVar: String, outputSlot: Int,
        rows: Int, cols: Int, groupSize: Int,
        comment: String? = nil
    ) throws -> [String] {
        // Family via the ONE gateway; see emitMatvec. Variable-input-slot lowering, so a TQH
        // weight selects the .tqh family but has no variable-slot kernel — that lowering gap is
        // handled locally below (the family exists; only this slot's emit is unwired).
        let resolvedGroupSize = weightEntry.groupSize ?? groupSize
        switch try matvecFamily(weightEntry, shape: .gemv, output: .fp16, slot: .variable) {
        case .lutU4:
            guard let lutOff = weightEntry.lutOffset else {
                throw SmeltEmitError.unsupported(
                    detail: "u4Lut weight '\(weightEntry.name)' has no lutOffset"
                )
            }
            return try emitFusedLUTMatvecVar(
                indicesSlot: weightsSlot, indicesOffset: weightEntry.offset,
                lutSlot: weightsSlot, lutOffset: lutOff,
                inputSlotVar: inputSlotVar, outputSlot: outputSlot,
                rows: rows, cols: cols, groupSize: resolvedGroupSize,
                comment: comment
            )
        case .affineU4:
            guard let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset
            else {
                throw SmeltEmitError.unsupported(
                    detail: "affineU4 weight '\(weightEntry.name)' has no "
                        + "scales/biases offsets"
                )
            }
            return try emitAffineMatvecVar(
                weightsSlot: weightsSlot, weightsOffset: weightEntry.offset,
                scalesSlot: weightsSlot, scalesOffset: scalesOff,
                biasesSlot: weightsSlot, biasesOffset: biasesOff,
                inputSlotVar: inputSlotVar, outputSlot: outputSlot,
                rows: rows, cols: cols, groupSize: resolvedGroupSize,
                comment: comment
            )
        case .binary1, .ternary2:
            return try emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(variableSlot: inputSlotVar, index: 2),
                outputBinding: SmeltBufferBinding(slot: outputSlot, index: 3),
                rows: rows,
                cols: cols,
                comment: comment
            )
        case .tqh:
            // Unreachable: variable-slot tqh is notMeaningful (no emitMatvecVar consumer packs
            // tqh — see MatvecKernelTable.meaningfulCells), so the gateway already threw
            // .notMeaningful above. Kept only for Family exhaustiveness. A TQH-packed emitMatvecVar
            // weight would add the cell + an emitTQHMatvecVar kernel together (§0.5).
            throw SmeltEmitError.unsupported(
                detail: "emitMatvecVar has no .turboQuantH branch; "
                    + "variable-slot tqh is notMeaningful (no consumer)"
            )
        case .dense(let dt):
            // fp16-act dense, variable input slot: .fp16 → fp16_matvec, .bf16/.fp32 → the U2
            // fp16_matvec_{bf16,fp32}w kernels (gateway authorizes only these three dense dtypes).
            return try emitFP16MatvecVar(
                weightSlot: weightsSlot, weightOffset: weightEntry.offset,
                inputSlotVar: inputSlotVar, outputSlot: outputSlot,
                rows: rows, cols: cols, weight: dt,
                comment: comment
            )
        }
    }

    // MARK: - Source line helpers

    /// Emit a blank line.
    public func emitBlank() -> String { "" }

    /// Emit a comment line.
    public func emitComment(_ text: String) -> String { "\(pad)// \(text)" }

    /// Emit a raw Swift source line (for custom code like swap).
    public func emitRaw(_ line: String) -> String { "\(pad)\(line)" }
}
