// SmeltBlockGraph — the package's declared composition: front-end blocks →
// transformer trunk(s) → head blocks, with typed ports, state, and side
// outputs. The Lego thesis made literal: a model is wiring between blocks,
// and the package manifest says what the wiring is.
//
// B1 of docs/block-spec-plan.md: this is the DECLARATION layer. Builders
// stamp the graph that describes what they already build (hand-built or
// compiled — `impl` says which); nothing executes differently yet. The
// graph's endpoints declare the package IO contract. Runtime behavior is
// selected by CAM flows and capabilities, not by mapping those endpoints back
// into package kinds.
//
// Honesty rules (from the codex design review): state (KV caches, the
// talker's GPU sampler history, codec stream
// carry) and side outputs (alignment timings) are first-class — a block
// vocabulary that can't say these things would describe the existing stacks
// dishonestly. Loop schedules (who steps whom, command-buffer grouping) are
// B2 and deliberately NOT declared here.

import Foundation

public struct SmeltBlockGraph: Codable, Sendable, Equatable {

    /// What flows across a block boundary.
    public enum PortType: String, Codable, Sendable, CaseIterable {
        case text
        case tokens
        case embeddings
        case hidden
        case logits
        case codecFrames = "codec-frames"
        case audio

        /// Types that can cross the package boundary (a graph's endpoints).
        public var isExternal: Bool { self == .text || self == .audio || self == .codecFrames }
    }

    /// Functional position in the pipeline. Orthogonal to `impl`: an
    /// audio encoder is a transformer by construction but a front-end by
    /// function (it produces the decoder's conditioning).
    public enum Role: String, Codable, Sendable {
        case frontend
        case trunk
        case head

        var rank: Int {
            switch self {
            case .frontend: return 0
            case .trunk: return 1
            case .head: return 2
            }
        }
    }

    /// How the block is realized. `compiled` = dispatch tables from the
    /// emitter; `native` = a named runtime implementation. Promotion from
    /// native to compiled changes this field and nothing else.
    public enum Impl: String, Codable, Sendable {
        case compiled
        case native
    }

    /// HOW a block's compiled compute is delivered — orthogonal to `impl`, so the
    /// graph is an honest map of what actually runs (and, for distribution, what is a
    /// SEPARABLE shippable artifact). `bakedInline` = a baked dispatch table in the
    /// MAIN package (the text trunk's `dispatches.bin`) — part of the whole package, not
    /// separable. `bakedSidecar` = a baked table in its OWN sidecar SUBDIR with its own
    /// manifest, sharing the parent `weights.bin`/`model.metallib` (the talker `trunk/`)
    /// — independently shippable. `runtimeEmit` = the emitter runs IN the runtime per
    /// request, a shape-variant block (the codec), NOT a baked artifact. `internalSidecar`
    /// = a `.native` head wrapper that drives a compiled internal transformer in a sidecar
    /// subdir (the MTP head → `trunk-mtp/`). nil = no compiled artifact (a plain hand
    /// `.native` block). Like `impl`, DECLARATIVE today — it makes the graph honest,
    /// nothing branches on it yet.
    public enum CompiledDelivery: String, Codable, Sendable {
        case bakedInline = "baked-inline"
        case bakedSidecar = "baked-sidecar"
        case runtimeEmit = "runtime-emit"
        case internalSidecar = "internal-sidecar"
    }

    /// Persistent per-block state the runtime carries across steps.
    public enum StateKind: String, Codable, Sendable {
        case kvCache = "kv-cache"
        case sampler
        case stream
    }

    /// Declared side outputs. A closed vocabulary on purpose: an unknown
    /// label written by a future smelt fails the typed decode and reports
    /// as "newer smelt", never as silently-ignored data.
    public enum SideOutput: String, Codable, Sendable {
        case alignment
    }

    public struct Block: Codable, Sendable, Equatable {
        public let name: String
        public let role: Role
        public let impl: Impl
        /// First input is the primary port (fed by the previous block);
        /// extra inputs must be the output of some earlier block (e.g. the
        /// MTP head consumes the codec head's cb0 token AND the talker's
        /// hidden state).
        public let inputs: [PortType]
        public let output: PortType
        /// Autoregressive self-feedback: what this block's own emissions
        /// re-enter as on the next step (LLM trunk: sampled token; talker:
        /// codec embedding sums). A loop edge, not
        /// wiring — no upstream producer required; the loop schedule (B2)
        /// drives it.
        public let feedback: PortType?
        public let state: [StateKind]?
        public let sideOutputs: [SideOutput]?
        /// Compiled-artifact delivery this block carries (orthogonal to `impl`);
        /// nil = plain native. See `CompiledDelivery`.
        public let compiledDelivery: CompiledDelivery?

        public init(
            name: String, role: Role, impl: Impl,
            inputs: [PortType], output: PortType,
            feedback: PortType? = nil,
            state: [StateKind]? = nil, sideOutputs: [SideOutput]? = nil,
            compiledDelivery: CompiledDelivery? = nil
        ) {
            self.name = name
            self.role = role
            self.impl = impl
            self.inputs = inputs
            self.output = output
            self.feedback = feedback
            self.state = state
            self.sideOutputs = sideOutputs
            self.compiledDelivery = compiledDelivery
        }

        /// Copy with a new `impl` + `compiledDelivery` — derives a compiled graph
        /// variant honestly without restating the other fields.
        public func realized(_ impl: Impl, delivery: CompiledDelivery?) -> Block {
            Block(name: name, role: role, impl: impl, inputs: inputs, output: output,
                  feedback: feedback, state: state, sideOutputs: sideOutputs,
                  compiledDelivery: delivery)
        }
    }

    public let version: Int
    public let blocks: [Block]

    public init(version: Int = 1, blocks: [Block]) {
        self.version = version
        self.blocks = blocks
    }

    public var runtimeRoutes: [SmeltPackageSpec.RuntimeDescriptor.BlockRoute] {
        blocks.map {
            .init(block: $0.name, impl: $0.impl, delivery: $0.compiledDelivery)
        }
    }

    public var runtimeRouteSignatures: [String] {
        runtimeRoutes.map(\.signature)
    }

    public enum GraphError: Error, CustomStringConvertible, Equatable {
        case malformed(String)

        public var description: String {
            switch self {
            case .malformed(let why): return "block graph: \(why)"
            }
        }
    }

    /// The package's modality signature: what it consumes and emits.
    public var signature: (input: PortType, output: PortType)? {
        guard let first = blocks.first?.inputs.first,
              let last = blocks.last?.output else { return nil }
        return (first, last)
    }

    /// Structural validation: a linear pipeline whose wiring is satisfiable.
    public func validate() throws {
        guard version == 1 else {
            throw GraphError.malformed("unsupported version \(version)")
        }
        guard !blocks.isEmpty else {
            throw GraphError.malformed("no blocks")
        }
        var seenNames = Set<String>()
        for block in blocks {
            guard !block.name.isEmpty, seenNames.insert(block.name).inserted else {
                throw GraphError.malformed("duplicate or empty block name '\(block.name)'")
            }
            guard !block.inputs.isEmpty else {
                throw GraphError.malformed("block '\(block.name)' has no inputs")
            }
        }

        // Endpoints must be externally visible types.
        guard let sig = signature, sig.input.isExternal, sig.output.isExternal else {
            throw GraphError.malformed(
                "graph endpoints must be external types (text/audio); got "
                    + "\(blocks.first!.inputs.first!.rawValue) → \(blocks.last!.output.rawValue)"
            )
        }

        // Roles run frontend → trunk → head: a front-end first (something
        // must adapt the external input), a head last, a trunk between.
        if blocks.count == 1 {
            guard sig.input == .codecFrames, sig.output == .audio else {
                throw GraphError.malformed(
                    "single-block package graphs are only supported for codec-frames → audio"
                )
            }
            return
        }
        var lastRank = -1
        for block in blocks {
            guard block.role.rank >= lastRank else {
                throw GraphError.malformed(
                    "block '\(block.name)' (\(block.role.rawValue)) appears after a later role"
                )
            }
            lastRank = block.role.rank
        }
        guard blocks.first!.role == .frontend, blocks.last!.role == .head else {
            throw GraphError.malformed(
                "graph must start with a frontend and end with a head"
            )
        }
        guard blocks.contains(where: { $0.role == .trunk }) else {
            throw GraphError.malformed("graph declares no trunk")
        }

        // Wiring: the first block takes exactly the external input; every
        // later primary input is the previous block's output; extra inputs
        // must be produced by SOME earlier block.
        guard blocks.first!.inputs.count == 1 else {
            throw GraphError.malformed(
                "block '\(blocks.first!.name)' is first and must have exactly "
                    + "one input (nothing upstream can feed extras)"
            )
        }
        var produced: [PortType] = []
        for (index, block) in blocks.enumerated() {
            if index > 0 {
                let upstream = blocks[index - 1].output
                guard block.inputs.first == upstream else {
                    throw GraphError.malformed(
                        "block '\(block.name)' expects \(block.inputs.first!.rawValue) "
                            + "but '\(blocks[index - 1].name)' produces \(upstream.rawValue)"
                    )
                }
                for extra in block.inputs.dropFirst() {
                    guard produced.contains(extra) else {
                        throw GraphError.malformed(
                            "block '\(block.name)' input \(extra.rawValue) is not "
                                + "produced by any earlier block"
                        )
                    }
                }
            }
            produced.append(block.output)
        }
    }

    // MARK: - Canonical graphs (single source of truth for builders + tests)

    /// Text token-feedback graph: tokenizer front-end, a compiled
    /// trunk whose dispatch table runs embedding gather → layers → LM head
    /// → logits, and selection + detokenize as the text head.
    public static let tokenFeedbackText = SmeltBlockGraph(blocks: [
        Block(name: "tokenizer", role: .frontend, impl: .native,
              inputs: [.text], output: .tokens),
        Block(name: "trunk", role: .trunk, impl: .compiled,
              inputs: [.tokens], output: .logits,
              feedback: .tokens,
              state: [.kvCache],
              compiledDelivery: .bakedInline),
        Block(name: "text-head", role: .head, impl: .native,
              inputs: [.logits], output: .text,
              state: [.sampler]),
    ])

    /// The Qwen3-TTS pipeline TEMPLATE: text front-end (tokenizer + embedding sums +
    /// speaker/language/instruct conditioning), the talker trunk whose own emissions
    /// re-enter as codec embedding sums, the cb0 codec head (GPU sampler — repetition
    /// penalty, suppress ranges, min-new-token EOS), the MTP head autoregressing across
    /// the residual codebooks, and the codec decoder. NOTE: no shipped package stamps
    /// this native-talker form anymore — the hand talker is retired (Phase 4). EVERY
    /// emittable dtype (f32/f16/bf16/u4) promotes the talker + MTP to a compiled trunk
    /// (`qwen3TTSCompiledTalker` for a bf16 text_embedding, else
    /// `qwen3TTSCompiledTrunkNativeFrontEnd`). This graph survives only as the builder's
    /// internal base for trunk-sidecar preparation.
    public static let qwen3TTS = SmeltBlockGraph(blocks: [
        Block(name: "tts-frontend", role: .frontend, impl: .native,
              inputs: [.text], output: .embeddings),
        Block(name: "talker", role: .trunk, impl: .native,
              inputs: [.embeddings], output: .hidden,
              feedback: .embeddings,
              state: [.kvCache]),
        Block(name: "codec-head", role: .head, impl: .native,
              inputs: [.hidden], output: .tokens,
              state: [.sampler]),
        Block(name: "mtp-head", role: .head, impl: .native,
              inputs: [.tokens, .hidden], output: .codecFrames,
              feedback: .embeddings,
              state: [.kvCache, .sampler]),
        Block(name: "codec-decoder", role: .head, impl: .compiled,
              inputs: [.codecFrames], output: .audio,
              state: [.stream],
              compiledDelivery: .runtimeEmit),
    ])

    /// The Qwen3-TTS pipeline for a bf16 build (B3.2c/d, 1a-ii): `qwen3TTS` with the
    /// tts-frontend promoted to `.compiled`/`.runtimeEmit` (its gather→project→merge is a
    /// Qwen3TTSFrontEndEmitter record table run through the generic SmeltCodecRecordRunner,
    /// byte-identical to the host path), the talker promoted to `.compiled`/`.bakedSidecar`
    /// (its embeddings→hidden dispatch tables live in the `trunk/` sidecar, sharing package
    /// weights), and the MTP head annotated `.internalSidecar` (it stays a `.native` head
    /// wrapping a compiled internal transformer in `trunk-mtp/`). The codec-decoder is
    /// already compiled in both variants. DERIVED from the native graph via `replacing` so
    /// the other blocks can't silently diverge. A bf16 build carries the trunk sidecars AND a
    /// compiled front-end (the front-end gathers bf16 text_embedding), so the builder stamps
    /// THIS graph then; a u4 build compiles the trunks but keeps the front-end native (see
    /// `qwen3TTSCompiledTrunkNativeFrontEnd`); f32/f16 keep `qwen3TTS` (all native).
    public static let qwen3TTSCompiledTalker = qwen3TTS
        .replacing("tts-frontend") { $0.realized(.compiled, delivery: .runtimeEmit) }
        .replacing("talker") { $0.realized(.compiled, delivery: .bakedSidecar) }
        .replacing("mtp-head") { $0.realized(.native, delivery: .internalSidecar) }

    /// A u4 build (Phase 3): the talker + MTP trunks compile (their unfused u4 route lives in
    /// the `trunk/` + `trunk-mtp/` sidecars, sharing the package's u4 weights), but the
    /// FRONT-END stays NATIVE — text_embedding is u4 and there is no compiled u4 gather path
    /// yet (the compiled front-end gathers bf16). So this is `qwen3TTSCompiledTalker` WITHOUT
    /// the front-end promotion: an HONEST graph for u4 (compiled trunks, native front-end),
    /// not the bf16 graph's compiled-front-end claim.
    public static let qwen3TTSCompiledTrunkNativeFrontEnd = qwen3TTS
        .replacing("talker") { $0.realized(.compiled, delivery: .bakedSidecar) }
        .replacing("mtp-head") { $0.realized(.native, delivery: .internalSidecar) }

    /// Standalone Qwen3-TTS codec decoder block package: RVQ codec frames in,
    /// waveform audio out. This is the graph for codec-only packages built by
    /// `Qwen3TTSPackageBuilder.build(specs:)`; it is a single block because the
    /// package boundary is the block boundary.
    public static let qwen3TTSCodecDecoder = SmeltBlockGraph(blocks: [
        Block(name: "codec-decoder", role: .head, impl: .compiled,
              inputs: [.codecFrames], output: .audio,
              state: [.stream],
              compiledDelivery: .runtimeEmit),
    ])

    /// This graph with the named block transformed — derive a compiled variant from a
    /// canonical graph without restating the other blocks (whose drift would otherwise
    /// go uncaught).
    public func replacing(_ name: String, _ transform: (Block) -> Block) -> SmeltBlockGraph {
        SmeltBlockGraph(version: version, blocks: blocks.map { $0.name == name ? transform($0) : $0 })
    }
}
