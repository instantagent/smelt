// SmeltModuleAuthoring — a thin sugar layer over the module IR member structs
// in SmeltSchema. It re-exports the schema so model definitions author values
// with a single `import SmeltModuleAuthoring`.
//
// Authority invariant: this layer adds NO semantics. It does not reimplement
// any parser lowering (export-binding inference, graph-value synthesis, quant
// priority assignment). Model definitions author the fully-lowered IR directly;
// canonical-JSON byte parity against the grammar parser is the correctness gate
// during the migration, and `validated()`/`canonicalized()` on the IR remain
// the authority thereafter. Everything here bottoms out in existing IR
// initializers.
//
// Containment: depends ONLY on SmeltSchema (enforced by Package.swift and the
// containment lint). It physically cannot import the compiler, runtime, or
// Metal, so authored code can only ever produce a `SmeltCAMIR` value.

@_exported import SmeltSchema

/// Short alias for the IR namespace used throughout model definitions.
public typealias IR = SmeltCAMIR

// MARK: - Type vocabulary

/// `text` values carry a utf8 encoding attribute.
public func textType() -> IR.TypeRef { IR.TypeRef("text", attributes: ["encoding": "utf8"]) }

/// A bare value type with no attributes (`tokens`, `hidden`, ...).
public func bareType(_ name: String) -> IR.TypeRef { IR.TypeRef(name) }

// MARK: - Small constructors

public func port(_ name: String, _ type: IR.TypeRef, optional: Bool = false) -> IR.Port {
    IR.Port(name: name, type: type, optional: optional)
}

public func annot(_ key: String, _ value: String) -> IR.Constraint { IR.Constraint(key, value) }

// MARK: - Standard single-prompt text-generation graph
//
// tokenizer(native) -> trunk(compiled) -> sampler(native) -> detokenizer(native)
// with a sampler->trunk feedback edge. Shared by every text-gen LLM fixture;
// the tokenizer annotations and the trunk `state` value are the only things
// that vary. The trunk always references a block named "trunk" and carries
// `artifact compiled-inline` + `feedback tokens`. Reproduces exactly the graph the
// grammar parser lowers from the canonical text-gen `graph:` stanza.

/// Tokenizer annotations for a ChatML prompt format with a preclosed think
/// prelude and disabled thinking policy (Qwen family).
public let chatmlTokenizerAnnotations: [IR.Constraint] = [
    annot("assistant-prelude", "preclosed-think"),
    annot("prompt-format", "chatml"),
    annot("tag", "text-tokenizer"),
    annot("thinking-policy", "disabled"),
]

/// The minimal tokenizer annotation: just the `text-tokenizer` tag.
public let plainTokenizerAnnotations: [IR.Constraint] = [annot("tag", "text-tokenizer")]

/// The tokenizer/trunk/sampler/detokenizer node set for a single-prompt LLM.
/// `tokenizerInput` defaults to the module's `prompt` port; a composed front end
/// (e.g. a prompt-builder adapter) passes the intermediate value it tokenizes.
public func llmTextGenNodes(
    tokenizer: [IR.Constraint], trunkState: String, tokenizerInput: IR.Port = port("prompt", textType())
) -> [IR.GraphNode] {
    [
        IR.GraphNode(
            id: "tokenizer",
            implementation: .native,
            inputs: [tokenizerInput],
            outputs: [port("tokens", bareType("tokens"))],
            annotations: tokenizer
        ),
        IR.GraphNode(
            id: "trunk",
            implementation: .compiled,
            block: "trunk",
            inputs: [port("tokens", bareType("tokens"))],
            outputs: [port("hidden", bareType("hidden"))],
            annotations: [
                annot("artifact", "compiled-inline"),
                annot("feedback", "tokens"),
                annot("state", trunkState),
            ]
        ),
        IR.GraphNode(
            id: "sampler",
            implementation: .native,
            inputs: [port("hidden", bareType("hidden"))],
            outputs: [port("tokens", bareType("tokens"))],
            annotations: [annot("state", "sampler"), annot("tag", "sampler")]
        ),
        IR.GraphNode(
            id: "detokenizer",
            implementation: .native,
            inputs: [port("tokens", bareType("tokens"))],
            outputs: [port("text", textType())],
            annotations: [annot("tag", "text-detokenizer")]
        ),
    ]
}

/// The prompt→tokenizer front edge plus the tokenizer→…→output back edges. The
/// sampler's `tokens` output flows to a deduped graph value `tokens_2` because
/// the tokenizer already produced a graph value named `tokens`.
public func llmTextGenEdges() -> [IR.GraphEdge] {
    [IR.GraphEdge(from: .moduleInput("prompt"), to: .node("tokenizer", "prompt"), type: textType())]
        + llmTextGenBackEdges()
}

/// The tokenizer→trunk→sampler→detokenizer→output edges, shared by every text-gen
/// LLM. A composed front end supplies its own `prompt→…→tokenizer` edges and
/// appends this back half.
public func llmTextGenBackEdges() -> [IR.GraphEdge] {
    [
        IR.GraphEdge(from: .node("tokenizer", "tokens"), to: .graphValue("tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .graphValue("tokens"), to: .node("trunk", "tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .node("trunk", "hidden"), to: .graphValue("hidden"), type: bareType("hidden")),
        IR.GraphEdge(from: .graphValue("hidden"), to: .node("sampler", "hidden"), type: bareType("hidden")),
        IR.GraphEdge(from: .node("sampler", "tokens"), to: .graphValue("tokens_2"), type: bareType("tokens")),
        IR.GraphEdge(from: .graphValue("tokens_2"), to: .node("detokenizer", "tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .node("detokenizer", "text"), to: .moduleOutput("text"), type: textType()),
    ]
}

/// The sampler→trunk token feedback edge.
public func llmTextGenFeedback() -> IR.FeedbackEdge {
    IR.FeedbackEdge(from: .node("sampler", "tokens"), to: .node("trunk", "tokens"))
}

/// The standard `generate` flow: setup tokenizer, step decode {trunk, sampler},
/// emit detokenizer.text, with the given stop conditions. `flowID`/`setupCalls`
/// default to the single-prompt shape; a composed front end overrides them (e.g.
/// a `review` flow whose setup runs the prompt-builder before the tokenizer).
public func llmGenerateFlow(
    flowID: String = "generate", setupCalls: [IR.FlowCall] = [.node("tokenizer")], stop: [IR.StopCondition]
) -> IR.Flow {
    IR.Flow(
        id: flowID,
        phases: [
            IR.FlowPhase(role: .setup, calls: setupCalls),
            IR.FlowPhase(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
        ],
        emit: [.node("detokenizer", "text")],
        stop: stop
    )
}

/// A `<source>."<pattern>" -> <block>.<targetSelector>` tensor map. The owner is
/// the target block (as the parser derives it).
public func tensorMap(source: String = "weights", pattern: String, block: String, target targetSelector: String) -> IR.TensorMap {
    IR.TensorMap(
        source: source,
        selector: IR.TensorSelector(pattern, source: source),
        target: IR.TensorTarget(block: block, selector: targetSelector),
        owner: block
    )
}

/// The `weights.* -> <block>.*` whole-trunk tensor map.
public func wholeTensorMap(source: String = "weights", block: String) -> IR.TensorMap {
    tensorMap(source: source, pattern: "*", block: block, target: "*")
}

// MARK: - Common gate shapes

/// A `startup` timing gate: `from flow.accepted` → `to emit text where tokens
/// >= 1`, requiring `elapsed <= <ms> ms`. `measured` adds the cold/cold/first
/// elapsed measurement block (present when the text declares `measure elapsed`).
public func startupTimingGate(flow: String = "generate", elapsedMs: String, measured: Bool) -> IR.Gate {
    IR.Gate(
        id: "startup",
        from: IR.GateEvent(kind: .flowAccepted, flow: flow),
        to: IR.GateEvent(
            kind: .emit,
            flow: flow,
            endpoint: .moduleOutput("text"),
            predicates: [IR.Comparison(subject: "tokens", relation: .greaterThanOrEqual, value: "1")]
        ),
        requirements: [IR.Comparison(subject: "elapsed", relation: .lessThanOrEqual, value: elapsedMs, unit: "ms")],
        measurements: measured
            ? [IR.GateMeasurement(subject: "elapsed", processMode: .cold, cacheState: .cold, occurrence: .first)]
            : []
    )
}

/// `gate prefill: require prefill-batch <= <n>`.
public func prefillGate(batch: String) -> IR.Gate {
    IR.Gate(id: "prefill", requirements: [IR.Comparison(subject: "prefill-batch", relation: .lessThanOrEqual, value: batch)])
}

/// `gate decode: require decode-output tokens >= 1`.
public func decodeGate() -> IR.Gate {
    IR.Gate(id: "decode", requirements: [IR.Comparison(subject: "decode-output.tokens", relation: .greaterThanOrEqual, value: "1")])
}

/// The nine files every compiled text-generation package ships, in the order
/// the inventory gate lists them. Shared by every text-gen LLM fixture's
/// `package-files include` gate so the list is authored once. The packaged
/// module descriptor is `module.json`.
public let standardTextGenPackageFiles =
    "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin,prefill_dispatches.bin,tokenizer.json,tokenizer.bin,module.json"

/// A gate whose only requirements are `package-files include <list>` and,
/// optionally, `release-surface-ids include <list>`.
public func inventoryGate(id: String, packageFiles: String, releaseSurfaceIDs: String? = nil) -> IR.Gate {
    var requirements = [IR.Comparison(subject: "package-files", relation: .include, value: packageFiles)]
    if let releaseSurfaceIDs {
        requirements.append(IR.Comparison(subject: "release-surface-ids", relation: .include, value: releaseSurfaceIDs))
    }
    return IR.Gate(id: id, requirements: requirements)
}
