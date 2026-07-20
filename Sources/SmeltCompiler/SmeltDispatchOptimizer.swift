// SmeltDispatchOptimizer — Peephole optimization passes over the dispatch IR.
//
// Runs after all dispatches are collected and before binary lowering.
// Each pass pattern-matches on [SmeltIROp] and rewrites in place.
// The optimizer reduces dispatch count without changing runtime behavior.

// MARK: - IR operation

/// A single operation in the dispatch sequence — either a kernel dispatch or a swap.
public enum SmeltIROp: Sendable {
    /// A Metal kernel dispatch.
    case dispatch(SmeltDispatch)
    /// A double-buffer swap (cur <-> alt).
    case swap
    /// A non-dispatched debug boundary marker.
    case traceMarker(label: String, bufferSlot: Int)
}

// MARK: - Pass protocol

/// An optimization pass over the dispatch IR.
public protocol SmeltOptimizationPass: Sendable {
    var name: String { get }
    /// Transform the operation sequence in place.
    func run(_ ops: inout [SmeltIROp])
}

// MARK: - Optimization Stats

public struct SmeltOptimizationStats: Sendable, Equatable {
    public private(set) var rewriteCounts: [String: Int]
    public private(set) var opportunities: [SmeltFusionOpportunitySummary]

    public init(
        rewriteCounts: [String: Int] = [:],
        opportunities: [SmeltFusionOpportunitySummary] = []
    ) {
        self.rewriteCounts = rewriteCounts
        self.opportunities = opportunities
    }

    public var isEmpty: Bool {
        rewriteCounts.isEmpty && opportunities.isEmpty
    }

    public var logSummary: String {
        guard !rewriteCounts.isEmpty else { return "none" }
        return rewriteCounts
            .sorted { lhs, rhs in lhs.key < rhs.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ", ")
    }

    public var opportunityLogSummary: String {
        guard !opportunities.isEmpty else { return "none" }
        return opportunities
            .sorted {
                if $0.count != $1.count { return $0.count > $1.count }
                if $0.pattern != $1.pattern { return $0.pattern < $1.pattern }
                return $0.shape < $1.shape
            }
            .prefix(8)
            .map {
                let status = $0.fusedKernelAvailable ? "available" : "missing"
                return "\($0.pattern):\($0.shape)=\($0.count)(\(status))"
            }
            .joined(separator: ", ")
    }

    mutating func record(_ rule: SmeltFusionRule) {
        rewriteCounts[rule.rawValue, default: 0] += 1
    }
}

// MARK: - Optimizer

/// Runs optimization passes over the collected dispatch sequence.
public struct SmeltDispatchOptimizer {

    /// Default pass pipeline, ordered by priority.
    public static let defaultPasses: [any SmeltOptimizationPass] = [
        ApplyFusionPlannerPass(),
        NormScaleDualAffineConsumerPass(),
    ]

    /// Run all passes on the dispatch sequence.
    public static func optimize(
        _ ops: inout [SmeltIROp],
        passes: [any SmeltOptimizationPass]? = nil
    ) {
        _ = optimize(&ops, planner: .auto, passes: passes)
    }

    @discardableResult
    static func optimize(
        _ ops: inout [SmeltIROp],
        planner: SmeltFusionPlanner,
        passes: [any SmeltOptimizationPass]? = nil
    ) -> SmeltOptimizationStats {
        var stats = SmeltOptimizationStats(
            opportunities: SmeltFusionOpportunityScanner.scan(
                ops,
                planner: planner
            )
        )
        for pass in (passes ?? defaultPasses(planner: planner)) {
            let before = ops.count
            if let fusionPass = pass as? ApplyFusionPlannerPass {
                fusionPass.run(&ops, stats: &stats)
            } else if let dualNormPass = pass as? NormScaleDualAffineConsumerPass {
                dualNormPass.run(&ops, stats: &stats)
            } else {
                pass.run(&ops)
            }
            let after = ops.count
            if after != before {
                // Pass changed something — could log here for debugging
            }
        }
        return stats
    }

    private static func defaultPasses(
        planner: SmeltFusionPlanner
    ) -> [any SmeltOptimizationPass] {
        [
            ApplyFusionPlannerPass(planner: planner),
            NormScaleDualAffineConsumerPass(),
        ]
    }
}

// MARK: - Fusion Opportunity Scanner

private enum SmeltFusionOpportunityScanner {
    private struct Key: Hashable {
        let pattern: String
        let shape: String
        let fusedKernelAvailable: Bool
    }

    static func scan(
        _ ops: [SmeltIROp],
        planner: SmeltFusionPlanner
    ) -> [SmeltFusionOpportunitySummary] {
        var counts: [Key: Int] = [:]

        func record(
            pattern: String,
            shape: String,
            fusedKernelAvailable: Bool
        ) {
            counts[
                Key(
                    pattern: pattern,
                    shape: shape,
                    fusedKernelAvailable: fusedKernelAvailable
                ),
                default: 0
            ] += 1
        }

        for index in ops.indices {
            guard case .dispatch(let current) = ops[index] else { continue }

            let rewrite = planner.rewrite(window: ops[index...])
            // Trace markers can make an output-eliding rewrite inapplicable,
            // but they do not make its Metal kernel disappear. Opportunity
            // reporting classifies kernel availability independently from
            // trace preservation; the optimizer still uses the original ops.
            let sourceVisibleOps = ops[index...].filter { op in
                if case .traceMarker = op { return false }
                return true
            }
            let sourceVisibleRewrite = planner.sourceVisibleRewrite(
                window: sourceVisibleOps[...]
            )
            if rewrite?.rule == .contiguousL2Normalize {
                record(
                    pattern: SmeltFusionRule.contiguousL2Normalize.rawValue,
                    shape: pipelineName(current),
                    fusedKernelAvailable: true
                )
            }

            guard let nextIndex = nextDispatchIndex(after: index, in: ops),
                  case .dispatch(let next) = ops[nextIndex]
            else {
                continue
            }

            if isNormLike(current),
               let normOutput = current.buffers.last,
               next.buffers.contains(where: { sameBinding($0, normOutput) })
            {
                record(
                    pattern: "normConsumer",
                    shape: "\(pipelineName(current))->\(pipelineName(next))",
                    fusedKernelAvailable: sourceVisibleRewrite != nil
                )
            }

            if isMatvecLike(current),
               next.pipeline == .elementwiseAdd,
               let matvecOutput = current.buffers.last,
               next.buffers.prefix(2).contains(where: {
                   sameBinding($0, matvecOutput)
               })
            {
                record(
                    pattern: SmeltFusionRule.matvecResidualAdd.rawValue,
                    shape: "\(pipelineName(current))->\(pipelineName(next))",
                    fusedKernelAvailable: rewrite?.rule == .matvecResidualAdd
                )
            }

            guard let thirdIndex = nextDispatchIndex(after: nextIndex, in: ops),
                  case .dispatch(let third) = ops[thirdIndex],
                  isMatvecLike(current),
                  isMatvecLike(next),
                  isActivationConsumer(third),
                  let firstOutput = current.buffers.last,
                  let secondOutput = next.buffers.last
            else {
                continue
            }

            let activationInputs = third.buffers.prefix(2)
            if activationInputs.contains(where: { sameBinding($0, firstOutput) })
                && activationInputs.contains(where: { sameBinding($0, secondOutput) })
            {
                record(
                    pattern: "dualMatvecActivation",
                    shape: "\(pipelineName(current))+\(pipelineName(next))->\(pipelineName(third))",
                    fusedKernelAvailable: rewrite?.rule == .dualMatvecActivation
                )
            }
        }

        return counts.map { key, count in
            SmeltFusionOpportunitySummary(
                pattern: key.pattern,
                shape: key.shape,
                count: count,
                fusedKernelAvailable: key.fusedKernelAvailable
            )
        }
        .sorted {
            if $0.count != $1.count { return $0.count > $1.count }
            if $0.pattern != $1.pattern { return $0.pattern < $1.pattern }
            return $0.shape < $1.shape
        }
    }

    private static func nextDispatchIndex(
        after index: Int,
        in ops: [SmeltIROp]
    ) -> Int? {
        var cursor = ops.index(after: index)
        while cursor < ops.endIndex {
            switch ops[cursor] {
            case .dispatch:
                return cursor
            case .traceMarker:
                cursor = ops.index(after: cursor)
            case .swap:
                return nil
            }
        }
        return nil
    }

    private static func sameBinding(
        _ lhs: SmeltBufferBinding,
        _ rhs: SmeltBufferBinding
    ) -> Bool {
        lhs.slot == rhs.slot
            && lhs.byteOffset == rhs.byteOffset
            && lhs.offsetKind == rhs.offsetKind
            && lhs.offsetExpression == rhs.offsetExpression
    }

    private static func pipelineName(_ dispatch: SmeltDispatch) -> String {
        var name = SmeltKernelCatalog.signatures[
            dispatch.pipeline.rawValue
        ].metalFunctionName
        if let cols = dispatch.fcCols,
           let groupSize = dispatch.fcGroupSize
        {
            name += ":\(cols):\(groupSize)"
        }
        return name
    }

    private static func isNormLike(_ dispatch: SmeltDispatch) -> Bool {
        pipelineName(dispatch).contains("rms_norm")
    }

    private static func isActivationConsumer(_ dispatch: SmeltDispatch) -> Bool {
        switch dispatch.pipeline {
        case .gegluFused, .swigluFused, .fusedGateUpSwiglu:
            return true
        default:
            return false
        }
    }

    private static func isMatvecLike(_ dispatch: SmeltDispatch) -> Bool {
        let name = pipelineName(dispatch)
        return name.contains("matvec")
            || name.contains("matmul")
            || name.contains("fp16_matvec")
    }
}

// MARK: - ApplyFusionPlannerPass

/// Applies the planner's registered fusion capabilities greedily over the IR.
///
/// This is the generic window matcher for policy-backed fusions: at each op it
/// asks the planner for the best legal rewrite starting there, applies it, and
/// keeps reduced rewrites at the same index so newly adjacent ops can be
/// considered too.
public struct ApplyFusionPlannerPass: SmeltOptimizationPass {
    public var name: String { "apply-fusion-planner" }

    let planner: SmeltFusionPlanner
    let allowedRules: Set<SmeltFusionRule>?

    init(
        planner: SmeltFusionPlanner = .auto,
        allowedRules: Set<SmeltFusionRule>? = nil
    ) {
        self.planner = planner
        self.allowedRules = allowedRules
    }

    public func run(_ ops: inout [SmeltIROp]) {
        var stats = SmeltOptimizationStats()
        run(&ops, stats: &stats)
    }

    func run(_ ops: inout [SmeltIROp], stats: inout SmeltOptimizationStats) {
        var i = 0
        while i < ops.count {
            guard let rewrite = planner.rewrite(window: ops[i...]),
                  allowedRules?.contains(rewrite.rule) ?? true
            else {
                i += 1
                continue
            }

            if shouldDeferToOverlappingDualRewrite(rewrite, at: i, in: ops) {
                i += 1
                continue
            }

            let end = i + rewrite.consumedOpCount
            guard rewrite.consumedOpCount > 0, end <= ops.count else {
                i += 1
                continue
            }
            let producedOpCount = rewrite.producedOps.count
            ops.replaceSubrange(i..<end, with: rewrite.producedOps)
            stats.record(rewrite.rule)
            if producedOpCount >= rewrite.consumedOpCount {
                i += max(producedOpCount, 1)
            }
        }
    }

    private func shouldDeferToOverlappingDualRewrite(
        _ rewrite: SmeltFusionRewrite,
        at index: Int,
        in ops: [SmeltIROp]
    ) -> Bool {
        guard rewrite.rule == .cooperativeNormScaleConsumer,
              allowedRules?.contains(.dualMatvecActivation) ?? true,
              index + 1 < ops.count,
              let overlapping = planner.rewrite(window: ops[(index + 1)...])
        else {
            return false
        }
        return overlapping.rule == .dualMatvecActivation
            && (allowedRules?.contains(overlapping.rule) ?? true)
    }
}

// MARK: - NormScaleDualAffineConsumerPass

/// Rewrites remaining E4B prefill K+V dual projections to consume the
/// cooperative norm-scale scratch plus raw norm input instead of re-reading
/// the materialized norm output.
///
/// Why this is a separate pass instead of another `CooperativeNormScaleRule`:
/// the standard cooperative rewrite matches a 2-op window `[norm, consumer]`,
/// but on the E4B prefill the Q affine sits between the norm and the K+V
/// dual matvec, so the planner can't reach the dual from the norm with a
/// 2-op window. This pass runs after the cooperative planner and either
/// rebases the dual onto an already-emitted scale-only scratch
/// (`findExistingScaleContext`) or emits a fresh scale-only dispatch directly
/// before the dual (`findDirectNormContext`).
public struct NormScaleDualAffineConsumerPass: SmeltOptimizationPass {
    public var name: String { "norm-scale-dual-affine-consumer" }

    private struct DirectNormContext {
        let normIndex: Int
        let normInput: SmeltBufferBinding
        let normWeight: SmeltBufferBinding
        let normOutput: SmeltBufferBinding
        let scalePipeline: SmeltPipeline
        let scaleTgWidth: Int
        let dynamicGridW: SmeltDynamicGridDimension?
        let dynamicGridH: SmeltDynamicGridDimension?
        let dynamicGridD: SmeltDynamicGridDimension?
        let minSeqLen: Int?
        let maxSeqLenExclusive: Int?
        let minPositionPlus1: Int?
        let maxPositionPlus1Exclusive: Int?
    }

    private struct ExistingScaleContext {
        let writerIndex: Int
        let scale: SmeltBufferBinding
        let normInput: SmeltBufferBinding
        let normWeight: SmeltBufferBinding
        let normOutput: SmeltBufferBinding
    }

    private static let dualRules: [SmeltPipeline: SmeltPipeline] = [
        .fusedDualAffineMatvecC2560R512G128Batched:
            .normScaleFusedDualAffineMatvecC2560R512G128Batched,
        .fusedDualAffineMatvecC2560R1024G128Batched:
            .normScaleFusedDualAffineMatvecC2560R1024G128Batched,
    ]

    public init() {}

    public func run(_ ops: inout [SmeltIROp]) {
        var stats = SmeltOptimizationStats()
        run(&ops, stats: &stats)
    }

    func run(_ ops: inout [SmeltIROp], stats: inout SmeltOptimizationStats) {
        var i = 0
        while i < ops.count {
            guard case .dispatch(let dual) = ops[i],
                  let fusedPipeline = Self.dualRules[dual.pipeline],
                  dual.buffers.indices.contains(8),
                  dual.constants.isEmpty
            else {
                i += 1
                continue
            }

            let normOutput = dual.buffers[6]

            if let existing = findExistingScaleContext(
                for: normOutput,
                before: i,
                in: ops
            ),
               normOutputReaderIndices(
                   normOutput,
                   in: ops,
                   after: existing.writerIndex
               ) == [i]
            {
                ops[i] = .dispatch(makeNormScaleDual(
                    from: dual,
                    fusedPipeline: fusedPipeline,
                    scale: existing.scale,
                    normInput: existing.normInput,
                    normWeight: existing.normWeight
                ))
                stats.record(.normScaleDualAffineConsumer)
                i += 1
                continue
            }

            if let direct = findDirectNormContext(
                for: normOutput,
                before: i,
                in: ops
            ),
               canRewriteDirectNormContext(direct, candidateIndex: i, in: ops)
            {
                let scale = SmeltBufferBinding(
                    slot: SmeltFixedSlot.normScaleScratch.rawValue,
                    index: 0
                )
                let scaleDispatch = makeScaleOnlyDispatch(from: direct)
                let fusedDual = makeNormScaleDual(
                    from: dual,
                    fusedPipeline: fusedPipeline,
                    scale: scale,
                    normInput: direct.normInput,
                    normWeight: direct.normWeight
                )
                ops.replaceSubrange(i...i, with: [
                    .dispatch(scaleDispatch),
                    .dispatch(fusedDual),
                ])
                stats.record(.normScaleDualAffineConsumer)
                i += 2
                continue
            }

            i += 1
        }
    }

    private func findExistingScaleContext(
        for normOutput: SmeltBufferBinding,
        before candidateIndex: Int,
        in ops: [SmeltIROp]
    ) -> ExistingScaleContext? {
        var cursor = candidateIndex - 1
        while cursor >= 0 {
            switch ops[cursor] {
            case .swap:
                return nil
            case .traceMarker:
                break
            case .dispatch(let dispatch):
                if isNormScaleMaterializer(dispatch, normOutput: normOutput) {
                    return ExistingScaleContext(
                        writerIndex: cursor,
                        scale: dispatch.buffers[0],
                        normInput: dispatch.buffers[1],
                        normWeight: dispatch.buffers[2],
                        normOutput: dispatch.buffers[3]
                    )
                }
                if dispatchWrites(dispatch, binding: normOutput)
                    || referencesNormScaleScratch(dispatch)
                {
                    return nil
                }
            }
            cursor -= 1
        }
        return nil
    }

    private func findDirectNormContext(
        for normOutput: SmeltBufferBinding,
        before candidateIndex: Int,
        in ops: [SmeltIROp]
    ) -> DirectNormContext? {
        var cursor = candidateIndex - 1
        while cursor >= 0 {
            switch ops[cursor] {
            case .swap:
                return nil
            case .traceMarker:
                break
            case .dispatch(let dispatch):
                if referencesNormScaleScratch(dispatch) {
                    return nil
                }
                if dispatchWrites(dispatch, binding: normOutput) {
                    return directNormContext(from: dispatch, at: cursor)
                }
            }
            cursor -= 1
        }
        return nil
    }

    private func directNormContext(
        from norm: SmeltDispatch,
        at index: Int
    ) -> DirectNormContext? {
        guard norm.buffers.indices.contains(2) else {
            return nil
        }

        let scalePipeline: SmeltPipeline
        let scaleTgWidth: Int
        switch norm.pipeline {
        case .rmsNorm1PWD2560Batched:
            scalePipeline = .rmsNormScaleOnlyD2560Batched
            scaleTgWidth = 1024
        case .rmsNorm1PWBatched
            where norm.constants.count >= 2
                && norm.constants[0].expression == "2560"
                && Float(norm.constants[1].expression) == 1e-6:
            scalePipeline = .rmsNormScaleOnlyD2560Batched
            scaleTgWidth = 1024
        default:
            return nil
        }

        return DirectNormContext(
            normIndex: index,
            normInput: norm.buffers[0],
            normWeight: norm.buffers[1],
            normOutput: norm.buffers[2],
            scalePipeline: scalePipeline,
            scaleTgWidth: scaleTgWidth,
            dynamicGridW: norm.dynamicGridW,
            dynamicGridH: norm.dynamicGridH,
            dynamicGridD: norm.dynamicGridD,
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )
    }

    private func canRewriteDirectNormContext(
        _ context: DirectNormContext,
        candidateIndex: Int,
        in ops: [SmeltIROp]
    ) -> Bool {
        let readers = normOutputReaderIndices(
            context.normOutput,
            in: ops,
            after: context.normIndex
        )
        guard readers.contains(candidateIndex) else {
            return false
        }

        for readerIndex in readers where readerIndex != candidateIndex {
            guard case .dispatch(let reader) = ops[readerIndex],
                  isAllowedE4BQProjection(reader, reading: context.normOutput)
            else {
                return false
            }
        }

        // The rewritten dual reads raw normInput + scratch instead of the
        // materialized normOutput. If anything between the norm producer and
        // the dual writes normInput, the fused kernel sees the mutated value
        // while the unfused dual would have consumed the pre-mutation
        // normOutput. Bail conservatively when that happens.
        for index in (context.normIndex + 1) ..< candidateIndex {
            guard case .dispatch(let intermediate) = ops[index] else { continue }
            if dispatchWrites(intermediate, binding: context.normInput) {
                return false
            }
        }

        return true
    }

    private func normOutputReaderIndices(
        _ normOutput: SmeltBufferBinding,
        in ops: [SmeltIROp],
        after writerIndex: Int
    ) -> [Int] {
        var readers: [Int] = []
        var cursor = writerIndex + 1
        while cursor < ops.count {
            switch ops[cursor] {
            case .swap:
                return readers
            case .traceMarker:
                break
            case .dispatch(let dispatch):
                if dispatchWrites(dispatch, binding: normOutput) {
                    return readers
                }
                if dispatch.buffers.contains(where: { sameBinding($0, normOutput) }) {
                    readers.append(cursor)
                }
            }
            cursor += 1
        }
        return readers
    }

    private func isAllowedE4BQProjection(
        _ dispatch: SmeltDispatch,
        reading normOutput: SmeltBufferBinding
    ) -> Bool {
        switch dispatch.pipeline {
        case .affineMatvecC2560R2048G128Batched,
             .affineMatvecC2560R4096G128Batched:
            return dispatch.buffers.indices.contains(3)
                && sameBinding(dispatch.buffers[3], normOutput)
        default:
            return false
        }
    }

    private func isNormScaleMaterializer(
        _ dispatch: SmeltDispatch,
        normOutput: SmeltBufferBinding
    ) -> Bool {
        guard dispatch.buffers.count >= 4,
              sameBinding(
                  dispatch.buffers[0],
                  SmeltBufferBinding(
                      slot: SmeltFixedSlot.normScaleScratch.rawValue,
                      index: dispatch.buffers[0].bindingIndex
                  )
              ),
              sameBinding(dispatch.buffers[3], normOutput)
        else {
            return false
        }

        let name = SmeltKernelCatalog.signature(
            for: dispatch.pipeline
        ).metalFunctionName
        return name.hasPrefix("norm_scale_")
            && !name.hasPrefix("norm_scale_fused_dual_affine_matvec")
    }

    private func makeScaleOnlyDispatch(
        from context: DirectNormContext
    ) -> SmeltDispatch {
        SmeltDispatch(
            pipeline: context.scalePipeline,
            buffers: [
                rebind(context.normInput, index: 0),
                SmeltBufferBinding(
                    slot: SmeltFixedSlot.normScaleScratch.rawValue,
                    index: 1
                ),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: context.scaleTgWidth,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: "RMS norm scale only for K+V",
            dynamicGridW: context.dynamicGridW,
            dynamicGridH: context.dynamicGridH,
            dynamicGridD: context.dynamicGridD,
            minSeqLen: context.minSeqLen,
            maxSeqLenExclusive: context.maxSeqLenExclusive,
            minPositionPlus1: context.minPositionPlus1,
            maxPositionPlus1Exclusive: context.maxPositionPlus1Exclusive
        )
    }

    private func makeNormScaleDual(
        from dual: SmeltDispatch,
        fusedPipeline: SmeltPipeline,
        scale: SmeltBufferBinding,
        normInput: SmeltBufferBinding,
        normWeight: SmeltBufferBinding
    ) -> SmeltDispatch {
        var buffers: [SmeltBufferBinding] = [
            rebind(scale, index: 0),
            rebind(normInput, index: 1),
            rebind(normWeight, index: 2),
        ]
        var nextIndex = 3
        for (bufferIndex, buffer) in dual.buffers.enumerated() {
            if bufferIndex == 6 { continue }
            buffers.append(rebind(buffer, index: nextIndex))
            nextIndex += 1
        }

        return SmeltDispatch(
            pipeline: fusedPipeline,
            buffers: buffers,
            constants: [],
            dispatch: dual.dispatch,
            comment: "Norm-scaled " + (dual.comment ?? "K+V dual affine"),
            dynamicGridW: dual.dynamicGridW,
            dynamicGridH: dual.dynamicGridH,
            dynamicGridD: dual.dynamicGridD,
            minSeqLen: dual.minSeqLen,
            maxSeqLenExclusive: dual.maxSeqLenExclusive,
            minPositionPlus1: dual.minPositionPlus1,
            maxPositionPlus1Exclusive: dual.maxPositionPlus1Exclusive
        )
    }

    private func dispatchWrites(
        _ dispatch: SmeltDispatch,
        binding: SmeltBufferBinding
    ) -> Bool {
        if dispatch.buffers.indices.contains(2),
           isSupportedNormProducer(dispatch),
           sameBinding(dispatch.buffers[2], binding)
        {
            return true
        }
        if isNormScaleMaterializer(dispatch, normOutput: binding) {
            return true
        }
        return dispatch.buffers.last.map { sameBinding($0, binding) } ?? false
    }

    private func isSupportedNormProducer(_ dispatch: SmeltDispatch) -> Bool {
        switch dispatch.pipeline {
        case .rmsNorm1PWD2560Batched:
            return true
        case .rmsNorm1PWBatched:
            return dispatch.constants.count >= 2
                && dispatch.constants[0].expression == "2560"
                && Float(dispatch.constants[1].expression) == 1e-6
        default:
            return false
        }
    }

    private func referencesNormScaleScratch(_ dispatch: SmeltDispatch) -> Bool {
        dispatch.buffers.contains {
            if case .fixed(let slot) = $0.slot {
                return slot == SmeltFixedSlot.normScaleScratch.rawValue
            }
            return false
        }
    }

    private func rebind(_ binding: SmeltBufferBinding, index: Int) -> SmeltBufferBinding {
        switch binding.slot {
        case .fixed(let slot):
            if let offsetExpression = binding.offsetExpression {
                return SmeltBufferBinding(
                    slot: slot,
                    offsetExpression: offsetExpression,
                    index: index
                )
            }
            if binding.offsetKind != 0 {
                return SmeltBufferBinding(
                    slot: slot,
                    offset: binding.byteOffset,
                    offsetKind: binding.offsetKind,
                    index: index
                )
            }
            return SmeltBufferBinding(
                slot: slot,
                offset: binding.byteOffset,
                index: index
            )
        case .variable(let name):
            return SmeltBufferBinding(
                variableSlot: name,
                offset: binding.byteOffset,
                index: index
            )
        }
    }

    private func sameBinding(
        _ lhs: SmeltBufferBinding,
        _ rhs: SmeltBufferBinding
    ) -> Bool {
        lhs.slot == rhs.slot
            && lhs.byteOffset == rhs.byteOffset
            && lhs.offsetKind == rhs.offsetKind
            && lhs.offsetExpression == rhs.offsetExpression
    }
}

// MARK: - FuseL2NormPass

/// Fuse consecutive L2 normalize dispatches on contiguous buffer sub-ranges.
///
/// Pattern (per DeltaNet layer):
///   l2Normalize(qkvBuf, offset=0,         gridW=numHeads, headDim)  // Q
///   l2Normalize(qkvBuf, offset=kByteOff,  gridW=numHeads, headDim)  // K
///
/// Q and K are contiguous in qkvBuf: K starts at Q_offset + numHeads * headDim * 2.
/// The kernel addresses data[head_id * headDim], so doubling gridW naturally
/// covers both Q (heads 0..<N) and K (heads N..<2N) in one dispatch.
///
/// Saves 1 dispatch per DeltaNet layer (18 for Qwen 3.5 2B).
public struct FuseL2NormPass: SmeltOptimizationPass {
    public var name: String { "fuse-l2-norm" }

    public func run(_ ops: inout [SmeltIROp]) {
        ApplyFusionPlannerPass(
            allowedRules: [.contiguousL2Normalize]
        ).run(&ops)
    }
}

// MARK: - FuseMatvecResidualAddPass

/// Fuse a matvec immediately followed by an elementwiseAdd on its output
/// into a single fused matvec+add dispatch.
///
/// Supports both quantization formats:
///   fusedLutMatvec (4 bufs, output@3) → fusedLutMatvecAdd (5 bufs)
///   affineMatvec   (5 bufs, output@4) → fusedAffineMatvecAdd (6 bufs)
///
/// Pattern (2x per layer — attention/delta residual + FFN residual):
///   matvec(..., output=X, numRows)
///   elementwiseAdd(residual=cur, intermediate=X, output=alt, count)
///
/// The intermediate buffer (X) is eliminated — matvec result goes straight to
/// the output with the residual added in-kernel.
///
/// Saves 1 dispatch per fusion site (up to 48 per model = 24 layers × 2).
public struct FuseMatvecResidualAddPass: SmeltOptimizationPass {
    public var name: String { "fuse-matvec-residual-add" }

    let planner: SmeltFusionPlanner

    init(planner: SmeltFusionPlanner = .auto) {
        self.planner = planner
    }

    public func run(_ ops: inout [SmeltIROp]) {
        ApplyFusionPlannerPass(
            planner: planner,
            allowedRules: [.matvecResidualAdd]
        ).run(&ops)
    }
}

// MARK: - AtomicNormFusionPass

/// Fuse norm+consumer into a single dispatch via device-scope atomic signaling.
/// TG 0 computes the norm scale, atomically signals "ready", other TGs spin then proceed.
/// Eliminates one dispatch per fusion site.
public struct AtomicNormFusionPass: SmeltOptimizationPass {
    public var name: String { "atomic-norm-fusion" }

    private struct Rule {
        let consumerPipeline: SmeltPipeline
        let fusedPipeline: SmeltPipeline
        let consumerInputIndex: Int
        let writesNormOutput: Bool
    }

    private static let rules: [Rule] = [
        Rule(consumerPipeline: .fusedAffineGateUpSwiglu,
             fusedPipeline: .atomicNormAffineGateUpSwiglu,
             consumerInputIndex: 6, writesNormOutput: false),
        Rule(consumerPipeline: .affineMatvec,
             fusedPipeline: .atomicNormAffineMatvec,
             consumerInputIndex: 3, writesNormOutput: true),
    ]

    public func run(_ ops: inout [SmeltIROp]) {
        let scratchSlot = SmeltFixedSlot.normOutBuf.rawValue  // reuse for scratch[2]

        var i = 0
        while i < ops.count - 1 {
            guard case .dispatch(let norm) = ops[i],
                  case .dispatch(let consumer) = ops[i + 1],
                  norm.pipeline == .rmsNorm1PW,
                  norm.buffers.count == 3,
                  norm.constants.count == 2
            else { i += 1; continue }

            let normInput = norm.buffers[0]
            let normWeight = norm.buffers[1]
            let normOutput = norm.buffers[2]

            guard let rule = Self.rules.first(where: { $0.consumerPipeline == consumer.pipeline }),
                  consumer.buffers[rule.consumerInputIndex].slot == normOutput.slot
                    && consumer.buffers[rule.consumerInputIndex].byteOffset == normOutput.byteOffset
            else { i += 1; continue }

            // Build fused dispatch
            var fusedBuffers: [SmeltBufferBinding] = [
                rebind(normInput, index: 0),
                rebind(normWeight, index: 1),
            ]
            var nextIdx = 2

            if rule.writesNormOutput {
                fusedBuffers.append(rebind(normOutput, index: nextIdx))
                nextIdx += 1
            }

            // Consumer buffers (skip the input that gets replaced by norm)
            for (ci, cb) in consumer.buffers.enumerated() {
                if ci == rule.consumerInputIndex { continue }
                fusedBuffers.append(rebind(cb, index: nextIdx))
                nextIdx += 1
            }

            // Scratch buffer for scale + ready flag
            fusedBuffers.append(SmeltBufferBinding(slot: scratchSlot, index: nextIdx))
            nextIdx += 1

            // Constants: numRows, eps
            let fusedConstants = [
                SmeltConstantBinding(expression: consumer.constants[0].expression, type: .uint32, index: nextIdx),
                SmeltConstantBinding(expression: norm.constants[1].expression, type: .float32, index: nextIdx + 1),
            ]

            let fused = SmeltDispatch(
                pipeline: rule.fusedPipeline,
                buffers: fusedBuffers,
                constants: fusedConstants,
                dispatch: consumer.dispatch,
                comment: "Atomic norm + " + (consumer.comment ?? "consumer"),
                fcCols: consumer.fcCols,
                fcGroupSize: consumer.fcGroupSize
            )

            ops[i] = .dispatch(fused)
            ops.remove(at: i + 1)
            // Don't advance — check if next pair also matches
        }
    }

    private func rebind(_ b: SmeltBufferBinding, index: Int) -> SmeltBufferBinding {
        switch b.slot {
        case .fixed(let slot): return SmeltBufferBinding(slot: slot, offset: b.byteOffset, index: index)
        case .variable(let name): return SmeltBufferBinding(variableSlot: name, offset: b.byteOffset, index: index)
        }
    }
}

// MARK: - CooperativeNormFusionPass

/// Cooperative norm fusion: replaces rmsNorm1PW with a tiny scale-only kernel (1 TG, 4 bytes output)
/// and rewrites the consumer to normalize inline using the pre-computed scale.
///
/// Pattern:
///   rmsNorm1PW(input, weight, output=X)
///   consumer(input=X, ...)
///
/// Becomes:
///   rmsNormScaleOnly(input, scale_scratch)          — 1 TG, ~10µs
///   normScaleConsumer(scale_scratch, input, weight, ...)  — normalizes inline
///
/// Saves: 4KB norm output write + 4KB consumer input read. Consumer reads raw input (L2-cached)
/// + 4-byte scale + norm_weight (L2-cached). Net: eliminates the normalized intermediate buffer.
public struct CooperativeNormFusionPass: SmeltOptimizationPass {
    public var name: String { "cooperative-norm-fusion" }

    let planner: SmeltFusionPlanner

    init(planner: SmeltFusionPlanner = .auto) {
        self.planner = planner
    }

    public func run(_ ops: inout [SmeltIROp]) {
        ApplyFusionPlannerPass(
            planner: planner,
            allowedRules: [.cooperativeNormScaleConsumer]
        ).run(&ops)
    }
}

// MARK: - FuseNormConsumerPass (DISABLED — replaced by CooperativeNormFusionPass)

/// Generic pass: fuse rmsNorm1PW into its sole consumer (matvec or gate_up_swiglu).
/// Eliminates inter-kernel gap + device memory round-trip for norm output.
/// Works across any LLM that does norm → projection.
///
/// Rule table maps (norm_pipeline, consumer_pipeline) → fused_pipeline.
/// The fused kernel computes the norm inline (each TG redundantly, input L2-cached).
///
/// Two variants:
///   - Sole consumer: norm output NOT written to device memory (e.g., post_attn_norm → FFN)
///   - Multi consumer: norm output written as side effect (e.g., input_norm → QKV, with Z/A/B also reading)
public struct FuseNormConsumerPass: SmeltOptimizationPass {
    public var name: String { "fuse-norm-consumer" }

    private struct FusionRule {
        let fusedPipeline: SmeltPipeline
        let normOutputBufferIndex: Int   // norm output buffer index in the norm dispatch
        let consumerInputIndex: Int      // which consumer buffer reads the norm output
        let writeNormOutput: Bool        // whether the fused kernel should write norm output (multi-consumer)
    }

    // Map: (consumer pipeline) → fusion rule. Norm is always rmsNorm1PW.
    private static let rules: [SmeltPipeline: FusionRule] = [
        .fusedAffineGateUpSwiglu: FusionRule(
            fusedPipeline: .fusedRmsNormAffineGateUpSwiglu,
            normOutputBufferIndex: 2,  // norm output slot
            consumerInputIndex: 6,     // gate_up_swiglu input slot
            writeNormOutput: false     // sole consumer, no side-effect write
        ),
        .affineMatvec: FusionRule(
            fusedPipeline: .fusedRmsNormAffineMatvec,
            normOutputBufferIndex: 2,
            consumerInputIndex: 3,     // affine_matvec input slot
            writeNormOutput: true      // may have other consumers
        ),
    ]

    public func run(_ ops: inout [SmeltIROp]) {
        var i = 0
        while i < ops.count - 1 {
            guard case .dispatch(let norm) = ops[i],
                  case .dispatch(let consumer) = ops[i + 1],
                  norm.pipeline == .rmsNorm1PW,
                  let rule = Self.rules[consumer.pipeline],
                  norm.buffers.count == 3,      // input, weight, output
                  norm.constants.count == 2      // dim, eps
            else {
                i += 1
                continue
            }

            let normInput = norm.buffers[0]    // input (variable: "cur")
            let normWeight = norm.buffers[1]   // weight
            let normOutput = norm.buffers[2]   // output (normOutSlot)

            // Verify: consumer reads from norm output
            guard consumer.buffers[rule.consumerInputIndex].slot == normOutput.slot
                && consumer.buffers[rule.consumerInputIndex].byteOffset == normOutput.byteOffset
            else {
                i += 1; continue
            }

            // Build fused dispatch
            var fusedBuffers: [SmeltBufferBinding] = []

            if rule.writeNormOutput {
                // Fused norm+matvec: norm_input(0), norm_weight(1), norm_output(2), then matvec buffers(3+)
                fusedBuffers.append(rebind(normInput, index: 0))
                fusedBuffers.append(rebind(normWeight, index: 1))
                fusedBuffers.append(rebind(normOutput, index: 2))  // side-effect write
                // Append consumer buffers, skipping the input (already provided as norm_input)
                var nextIdx = 3
                for (ci, cb) in consumer.buffers.enumerated() {
                    if ci == rule.consumerInputIndex { continue }  // skip, replaced by norm_input
                    fusedBuffers.append(rebind(cb, index: nextIdx))
                    nextIdx += 1
                }
            } else {
                // Fused norm+gate_up_swiglu: norm_input(0), norm_weight(1), then consumer buffers(2+)
                fusedBuffers.append(rebind(normInput, index: 0))
                fusedBuffers.append(rebind(normWeight, index: 1))
                var nextIdx = 2
                for (ci, cb) in consumer.buffers.enumerated() {
                    if ci == rule.consumerInputIndex { continue }
                    fusedBuffers.append(rebind(cb, index: nextIdx))
                    nextIdx += 1
                }
            }

            // Constants: consumer's constants + eps from norm
            var fusedConstants = consumer.constants.enumerated().map { (ci, cc) in
                SmeltConstantBinding(
                    expression: cc.expression, type: cc.type,
                    index: rule.writeNormOutput ? ci + 7 : ci + 9  // offset past fused buffer bindings
                )
            }
            // Add eps constant
            let epsBindingIndex = fusedConstants.last.map { $0.bindingIndex + 1 }
                ?? (rule.writeNormOutput ? 7 : 9)
            fusedConstants.append(SmeltConstantBinding(
                expression: norm.constants[1].expression,  // eps
                type: .float32,
                index: epsBindingIndex
            ))

            let fused = SmeltDispatch(
                pipeline: rule.fusedPipeline,
                buffers: fusedBuffers,
                constants: fusedConstants,
                dispatch: consumer.dispatch,  // same grid as consumer
                comment: "Fused norm + " + (consumer.comment ?? "consumer"),
                fcCols: consumer.fcCols,
                fcGroupSize: consumer.fcGroupSize
            )

            ops[i] = .dispatch(fused)
            ops.remove(at: i + 1)
        }
    }

    private func rebind(_ b: SmeltBufferBinding, index: Int) -> SmeltBufferBinding {
        switch b.slot {
        case .fixed(let slot):
            return SmeltBufferBinding(slot: slot, offset: b.byteOffset, index: index)
        case .variable(let name):
            return SmeltBufferBinding(variableSlot: name, offset: b.byteOffset, index: index)
        }
    }
}
