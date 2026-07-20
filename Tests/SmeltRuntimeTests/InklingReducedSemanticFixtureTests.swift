import SmeltSchema
import Foundation
import Testing

@Suite("Reduced CAM semantic fixture")
struct InklingReducedSemanticFixtureTests {
    @Test("CAM bricks recompose bit-exactly against an independent literal topology")
    func exactTopologyRecomposition() throws {
        let module = try SmeltCAMIR.decodeModule(at: Self.moduleURL("inkling"))
        let mtp = try SmeltCAMIR.decodeModule(at: Self.moduleURL("inkling_mtp"))

        let camTrace = try ReducedSemanticExecutor.camDriven(module: module, mtp: mtp)
        let referenceTrace = ReducedSemanticExecutor.literalReference()

        #expect(camTrace.map(\.id) == referenceTrace.map(\.id))
        #expect(camTrace.map { $0.value.bitPattern } == referenceTrace.map { $0.value.bitPattern })
        #expect(Self.firstDivergence(camTrace, referenceTrace) == nil)
        #expect(camTrace.filter { $0.id.contains(".ffn.dense") }.count == 10)
        #expect(camTrace.filter { $0.id.contains(".ffn.sparse") }.count == 64)
        #expect(camTrace.filter { $0.id.contains(".router.selected") }.count == 64)
        #expect(camTrace.filter { $0.id.contains(".sconv.") }.count == 74 * 4)
        #expect(camTrace.filter { $0.id.contains("attention.global") }.count == 13)
        #expect(camTrace.filter { $0.id.contains("attention.sliding") }.count == 61)
    }

    @Test("First-divergence evidence points at the first changed brick")
    func firstDivergenceEvidence() {
        let reference = ReducedSemanticExecutor.literalReference()
        var changed = reference
        let changedIndex = try! #require(changed.firstIndex { $0.id == "trunk.layer.5.attention.global" })
        changed[changedIndex].value = changed[changedIndex].value.nextUp

        let divergence = Self.firstDivergence(reference, changed)
        #expect(divergence?.index == changedIndex)
        #expect(divergence?.expectedID == "trunk.layer.5.attention.global")
        #expect(divergence?.actualID == "trunk.layer.5.attention.global")
        #expect(divergence?.expectedBits != divergence?.actualBits)
    }

    @Test("Routing correction changes selection but not joint normalization logits")
    func routerSemantics() {
        let routed = ReducedSemanticExecutor.route(x: 0.375, experts: 256, topK: 6, shared: 2)
        #expect(routed.selected.count == 6)
        #expect(routed.shared.count == 2)
        #expect(abs(routed.weights.reduce(0, +) - 1) < 0.000_001)

        let selectedScores = routed.selected.map { ReducedSemanticExecutor.routerLogit(x: 0.375, expert: $0) }
        let sharedScores = routed.shared.map { ReducedSemanticExecutor.routerLogit(x: 0.375, expert: $0) }
        let expected = (selectedScores + sharedScores).map(ReducedSemanticExecutor.sigmoid)
        let denominator = expected.reduce(0, +)
        #expect(routed.weights.map(\.bitPattern)
            == expected.map { ($0 / denominator).bitPattern })
    }

    private struct Divergence {
        let index: Int
        let expectedID: String?
        let actualID: String?
        let expectedBits: UInt32?
        let actualBits: UInt32?
    }

    private static func firstDivergence(
        _ expected: [ReducedSemanticExecutor.Stage],
        _ actual: [ReducedSemanticExecutor.Stage]
    ) -> Divergence? {
        let shared = min(expected.count, actual.count)
        for index in 0..<shared where
            expected[index].id != actual[index].id
                || expected[index].value.bitPattern != actual[index].value.bitPattern
        {
            return Divergence(
                index: index,
                expectedID: expected[index].id,
                actualID: actual[index].id,
                expectedBits: expected[index].value.bitPattern,
                actualBits: actual[index].value.bitPattern
            )
        }
        guard expected.count != actual.count else { return nil }
        return Divergence(
            index: shared,
            expectedID: expected.indices.contains(shared) ? expected[shared].id : nil,
            actualID: actual.indices.contains(shared) ? actual[shared].id : nil,
            expectedBits: expected.indices.contains(shared) ? expected[shared].value.bitPattern : nil,
            actualBits: actual.indices.contains(shared) ? actual[shared].value.bitPattern : nil
        )
    }

    private static func moduleURL(_ id: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("\(id).module.json")
    }
}

/// A deliberately tiny numerical execution whose topology is full-size. The
/// tensor widths are scalar, but every authored semantic branch executes. This
/// makes topology/parity failures cheap to localize without pretending the
/// fixture benchmarks or numerically substitutes for the 975B checkpoint.
private enum ReducedSemanticExecutor {
    struct Stage: Equatable {
        let id: String
        var value: Float
    }

    struct RouteResult {
        let selected: [Int]
        let shared: [Int]
        let weights: [Float]
        let output: Float
    }

    private struct Topology {
        let trunkRoles: [SmeltCAMIR.LayerRole]
        let trunkDenseLayers: Int
        let trunkAttention: [SmeltCAMIR.LayerRole: SmeltCAMIR.AttentionShape]
        let trunkRouter: SmeltCAMIR.RouterShape
        let convolutions: [SmeltCAMIR.ShortConvolutionShape]
        let mtpRoles: [SmeltCAMIR.LayerRole]
        let patchLayers: Int
        let audioValuesPerToken: Int
    }

    static func camDriven(module: SmeltCAMIR, mtp: SmeltCAMIR) throws -> [Stage] {
        let trunk = try #require(module.blocks.first { $0.id == "trunk" }?.shape.transformer)
        let mtpShape = try #require(mtp.blocks.first { $0.id == "mtp" }?.shape.transformer)
        let patch = try #require(module.blocks.first { $0.id == "image-patch-encoder" })
        let audio = try #require(module.blocks.first { $0.id == "audio-encoder" })
        let attention = Dictionary(uniqueKeysWithValues: try #require(trunk.attentionByRole).map {
            ($0.role, $0.attention)
        })
        return execute(Topology(
            trunkRoles: try expand(trunk.layers),
            trunkDenseLayers: try #require(trunk.denseLayerCount),
            trunkAttention: attention,
            trunkRouter: try #require(trunk.router),
            convolutions: try #require(trunk.shortConvolutions),
            mtpRoles: try expand(mtpShape.layers),
            patchLayers: try integerRequirement("layers", block: patch),
            audioValuesPerToken: try integerRequirement("values-per-token", block: audio)
        ))
    }

    /// Independent topology oracle: no module id, source metadata, descriptor,
    /// or CAM field is consulted. The math path is shared intentionally so a
    /// mismatch means wiring/order, and its first divergent brick is exact.
    static func literalReference() -> [Stage] {
        let sliding = SmeltCAMIR.AttentionShape(
            qHeads: 64,
            kvHeads: 16,
            headDim: 128,
            relativePosition: .init(projectionDim: 16, extent: 512),
            scaling: .inverseHeadDim,
            qkNorm: .rms,
            window: 512
        )
        let global = SmeltCAMIR.AttentionShape(
            qHeads: 64,
            kvHeads: 8,
            headDim: 128,
            relativePosition: .init(
                projectionDim: 16,
                extent: 1_024,
                logScalingFloor: 128_000,
                logScalingAlpha: "0.1"
            ),
            scaling: .inverseHeadDim,
            qkNorm: .rms
        )
        let repeated = Array(
            repeating: [
                SmeltCAMIR.LayerRole.sliding, .sliding, .sliding,
                .sliding, .sliding, .global,
            ],
            count: 11
        ).flatMap { $0 }
        return execute(Topology(
            trunkRoles: repeated,
            trunkDenseLayers: 2,
            trunkAttention: [.sliding: sliding, .global: global],
            trunkRouter: SmeltCAMIR.RouterShape(
                topK: 6,
                experts: 256,
                sharedExperts: 2,
                activation: .sigmoid,
                normalization: .selectedAndShared,
                scoreCorrectionBias: true,
                routeScale: "8",
                globalScale: true,
                sharedExpertSink: true
            ),
            convolutions: [
                .init(site: .attentionKey, kernelSize: 4, residual: .addInput),
                .init(site: .attentionValue, kernelSize: 4, residual: .addInput),
                .init(site: .attentionBranchOutput, kernelSize: 4, residual: .addInput),
                .init(site: .ffnBranchOutput, kernelSize: 4, residual: .addInput),
            ],
            mtpRoles: [
                .sliding, .global, .sliding, .global,
                .sliding, .sliding, .sliding, .sliding,
            ],
            patchLayers: 4,
            audioValuesPerToken: 16
        ))
    }

    private static func execute(_ topology: Topology) -> [Stage] {
        var trace: [Stage] = []
        var image: Float = 0.125
        for layer in 0..<topology.patchLayers {
            let projected = image * (0.75 + Float(layer) * 0.0625) + Float(layer + 1) * 0.0078125
            image = projected / sqrt(projected * projected + 0.000_001)
            trace.append(Stage(id: "image.patch.\(layer)", value: image))
        }

        var audio: Float = 0
        for value in 0..<topology.audioValuesPerToken {
            audio += Float((value * 37) % 1_280) * 0.000_031_25
        }
        audio /= Float(topology.audioValuesPerToken)
        trace.append(Stage(id: "audio.dmel.embedding", value: audio))

        var x = 0.25 + image * 0.125 + audio
        trace.append(Stage(id: "multimodal.token-fusion", value: x))
        for (layer, role) in topology.trunkRoles.enumerated() {
            x = applyLayer(
                x: x,
                layer: layer,
                namespace: "trunk",
                role: role,
                attention: topology.trunkAttention[role]!,
                dense: layer < topology.trunkDenseLayers,
                router: topology.trunkRouter,
                convolutions: topology.convolutions,
                trace: &trace
            )
        }

        // Two-source MTP fusion. Its eight layers remain dense and use the
        // base unembedding outside this fixture, matching module composition.
        x = x * 0.625 + 0.1875 * 0.375
        trace.append(Stage(id: "mtp.input-fusion", value: x))
        for (layer, role) in topology.mtpRoles.enumerated() {
            x = applyLayer(
                x: x,
                layer: layer,
                namespace: "mtp",
                role: role,
                attention: topology.trunkAttention[role]!,
                dense: true,
                router: topology.trunkRouter,
                convolutions: topology.convolutions,
                trace: &trace
            )
        }
        trace.append(Stage(id: "base.unembedding", value: x * 0.875 - 0.03125))
        return trace
    }

    private static func applyLayer(
        x: Float,
        layer: Int,
        namespace: String,
        role: SmeltCAMIR.LayerRole,
        attention: SmeltCAMIR.AttentionShape,
        dense: Bool,
        router: SmeltCAMIR.RouterShape,
        convolutions: [SmeltCAMIR.ShortConvolutionShape],
        trace: inout [Stage]
    ) -> Float {
        var q = rms(x * (0.75 + Float(layer % 7) * 0.015625))
        let rawK = x * (0.625 + Float(layer % 5) * 0.0078125)
        let rawV = x * (0.5 + Float(layer % 3) * 0.015625)
        let key = convolve(rawK, layer: layer, site: .attentionKey, convolutions: convolutions)
        let value = convolve(rawV, layer: layer, site: .attentionValue, convolutions: convolutions)
        trace.append(Stage(id: "\(namespace).layer.\(layer).sconv.attention-key", value: key))
        trace.append(Stage(id: "\(namespace).layer.\(layer).sconv.attention-value", value: value))

        let relative = attention.relativePosition!
        let distance = min(3, relative.extent - 1)
        var bias = x * Float(relative.projectionDim) * Float(distance + 1) * 0.000_122_070_312_5
        if let floor = relative.logScalingFloor,
           let alphaText = relative.logScalingAlpha,
           let alpha = Float(alphaText)
        {
            let absolutePosition = 256_000 + layer
            let tau = 1 + alpha * log(max(Float(absolutePosition + 1) / Float(floor), 1))
            q *= tau
            bias *= tau
        }
        let scale: Float = attention.scaling == .inverseHeadDim
            ? 1 / Float(attention.headDim)
            : 1 / sqrt(Float(attention.headDim))
        let attended = sigmoid(q * key * scale + bias) * value
        trace.append(Stage(id: "\(namespace).layer.\(layer).attention.\(role.rawValue)", value: attended))
        let attentionBranch = convolve(
            attended * 0.75,
            layer: layer,
            site: .attentionBranchOutput,
            convolutions: convolutions
        )
        trace.append(Stage(id: "\(namespace).layer.\(layer).sconv.attention-branch-output", value: attentionBranch))
        var output = x + attentionBranch

        let ffn: Float
        if dense {
            ffn = silu(output * 0.1875) * (output * -0.15625) * 0.125
            trace.append(Stage(id: "\(namespace).layer.\(layer).ffn.dense", value: ffn))
        } else {
            let routed = route(
                x: output,
                experts: router.experts,
                topK: router.topK,
                shared: router.sharedExperts ?? 0
            )
            let selectionDigest = routed.selected.reduce(Float(0)) { $0 + Float($1) * 0.000_001 }
            trace.append(Stage(id: "\(namespace).layer.\(layer).router.selected", value: selectionDigest))
            ffn = routed.output * Float(router.routeScale ?? "1")! * 1.0078125
            trace.append(Stage(id: "\(namespace).layer.\(layer).ffn.sparse", value: ffn))
        }
        let ffnBranch = convolve(
            ffn,
            layer: layer,
            site: .ffnBranchOutput,
            convolutions: convolutions
        )
        trace.append(Stage(id: "\(namespace).layer.\(layer).sconv.ffn-branch-output", value: ffnBranch))
        output += ffnBranch
        // Keep the synthetic scalar bounded without changing branch ordering.
        return tanh(output * 0.25)
    }

    static func route(x: Float, experts: Int, topK: Int, shared: Int) -> RouteResult {
        var candidates: [(expert: Int, score: Float)] = []
        candidates.reserveCapacity(experts)
        for expert in 0..<experts {
            let corrected = sigmoid(routerLogit(x: x, expert: expert))
                + Float((expert * 17) % 11 - 5) * 0.000_05
            candidates.append((expert: expert, score: corrected))
        }
        candidates.sort { lhs, rhs in
            lhs.score == rhs.score ? lhs.expert < rhs.expert : lhs.score > rhs.score
        }
        let selected = candidates.prefix(topK).map { $0.expert }
        let sharedIDs = Array(experts..<(experts + shared))
        // Correction bias is deliberately absent here: it affects selection,
        // never the jointly normalized routed/shared logits.
        let normalizationScores = (selected + sharedIDs).map {
            sigmoid(routerLogit(x: x, expert: $0))
        }
        let denominator = normalizationScores.reduce(0, +)
        let weights = normalizationScores.map { $0 / denominator }
        let activeExperts = selected + sharedIDs
        var output = Float(0)
        for index in activeExperts.indices {
            output += weights[index] * expertOutput(x: x, expert: activeExperts[index])
        }
        return RouteResult(
            selected: selected,
            shared: sharedIDs,
            weights: weights,
            output: output
        )
    }

    static func routerLogit(x: Float, expert: Int) -> Float {
        x * (0.015625 + Float((expert * 13) % 29) * 0.000_244_140_625)
            + Float(expert % 7 - 3) * 0.000_976_562_5
    }

    static func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func expertOutput(x: Float, expert: Int) -> Float {
        let gate = x * (0.03125 + Float(expert % 17) * 0.000_488_281_25)
        let up = x * (-0.0234375 + Float(expert % 11) * 0.000_244_140_625)
        let down = 0.0625 + Float(expert % 5) * 0.00390625
        return silu(gate) * up * down
    }

    private static func convolve(
        _ input: Float,
        layer: Int,
        site: SmeltCAMIR.ShortConvolutionSite,
        convolutions: [SmeltCAMIR.ShortConvolutionShape]
    ) -> Float {
        guard let convolution = convolutions.first(where: { $0.site == site }) else {
            return input
        }
        return shortConvolution(
            input,
            layer: layer,
            site: site,
            kernelSize: convolution.kernelSize,
            residual: convolution.residual
        )
    }

    private static func shortConvolution(
        _ input: Float,
        layer: Int,
        site: SmeltCAMIR.ShortConvolutionSite,
        kernelSize: Int,
        residual: SmeltCAMIR.ShortConvolutionResidual
    ) -> Float {
        let siteIndex: Int
        switch site {
        case .attentionKey: siteIndex = 0
        case .attentionValue: siteIndex = 1
        case .attentionBranchOutput: siteIndex = 2
        case .ffnBranchOutput: siteIndex = 3
        }
        var convolved: Float = residual == .addInput ? input : 0
        for tap in 0..<max(kernelSize - 1, 0) {
            let history = Float((layer + 1) * (siteIndex + 1) * (tap + 1)) * 0.000_030_517_578_125
            convolved += history * Float(tap + 1) * 0.03125
        }
        convolved += input * 0.125
        return convolved
    }

    private static func rms(_ value: Float) -> Float {
        value / sqrt(value * value + 0.000_001)
    }

    private static func silu(_ value: Float) -> Float {
        value * sigmoid(value)
    }

    private static func expand(
        _ pattern: SmeltCAMIR.LayerPattern?
    ) throws -> [SmeltCAMIR.LayerRole] {
        let pattern = try #require(pattern)
        if let repeatCount = pattern.repeatCount {
            return Array(repeating: pattern.roles, count: repeatCount).flatMap { $0 }
        }
        if let count = pattern.count, pattern.roles.count == 1 {
            return Array(repeating: pattern.roles[0], count: count)
        }
        throw FixtureError.unsupportedLayerPattern
    }

    private static func integerRequirement(
        _ key: String,
        block: SmeltCAMIR.Block
    ) throws -> Int {
        guard let text = block.shape.requirements.first(where: { $0.key == key })?.value,
              let value = Int(text) else {
            throw FixtureError.missingRequirement(key)
        }
        return value
    }

    private enum FixtureError: Error {
        case missingRequirement(String)
        case unsupportedLayerPattern
    }
}
