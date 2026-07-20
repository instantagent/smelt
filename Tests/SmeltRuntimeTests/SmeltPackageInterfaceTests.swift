import Foundation
import Testing
@testable import SmeltSchema

// SmeltPackageInterface — args.json decode / validate / argv scan / resolve /
// template fill. Pure logic; the CLI is a thin wrapper over these.

private func decode(_ json: String) throws -> SmeltPackageInterface {
    try SmeltPackageInterface.decode(from: Data(json.utf8))
}

private let ttsBuiltins: Set<String> = [
    "speaker", "language", "instruct", "seed", "max-frames",
    "first-chunk", "max-chunk", "play", "greedy", "prompt", "package",
]

private let codecAudioInterfaceContext = SmeltPackageInterface.InterfaceValidationContext.targeted(
    interfaceName: "targeted audio",
    policy: .codecAudio,
    builtinFlags: ttsBuiltins
)

private let textGenerationInterfaceContext = SmeltPackageInterface.InterfaceValidationContext
    .textGeneration(builtinFlags: [])

@Suite struct SmeltPackageInterfaceDecodeTests {

    @Test func decodesTypedDefaults() throws {
        let spec = try decode(
            """
            {"version": 1, "args": [
              {"flag": "voice", "type": "enum", "values": ["Ryan", "Katie"],
               "default": "Ryan", "target": "speaker"},
              {"flag": "pace", "type": "int", "default": 4, "target": "max-chunk"},
              {"flag": "loud", "type": "bool", "default": true, "target": "instruct"}
            ]}
            """
        )
        #expect(spec.args[0].defaultValue == "Ryan")
        #expect(spec.args[1].defaultValue == "4")
        #expect(spec.args[2].defaultValue == "true")
    }

    @Test func malformedJSONThrows() {
        #expect(throws: SmeltPackageInterfaceError.self) {
            try decode(#"{"version": 1, "args": [{"flag": 7}]}"#)
        }
    }

}

@Suite struct SmeltPackageInterfaceValidateTests {

    private func ttsArg(
        flag: String = "voice", type: SmeltPackageInterface.ArgType = .string,
        values: [String]? = nil, defaultValue: String? = nil,
        required: Bool = false, target: String? = "speaker"
    ) -> SmeltPackageInterface.Arg {
        .init(flag: flag, type: type, values: values, defaultValue: defaultValue,
              required: required, target: target)
    }

    @Test func validTTSInterfacePasses() throws {
        let spec = SmeltPackageInterface(args: [
            ttsArg(flag: "voice", type: .enum, values: ["Ryan", "Katie"],
                   defaultValue: "Ryan"),
            ttsArg(flag: "frames", type: .int, target: "max-frames"),
        ])
        try spec.validate(interfaceContext: codecAudioInterfaceContext)
    }

    @Test func packageValidationContextFollowsGraphPolicy() throws {
        let qwen = SmeltPackageInterface.packageValidationContext(
            graphPolicy: .sidecarTextToCodecAudio
        )
        #expect(qwen.interfaceContext.policy.targets["speaker"] != nil)
        #expect(qwen.interfaceContext.policy.targets["duration"] == nil)
        #expect(qwen.builtins.bareNames.contains("speaker"))
    }

    @Test func builtinShadowRejected() {
        let spec = SmeltPackageInterface(args: [ttsArg(flag: "speaker")])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func duplicateFlagRejected() {
        let spec = SmeltPackageInterface(args: [
            ttsArg(flag: "voice", target: "speaker"),
            ttsArg(flag: "voice", target: "language"),
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func duplicateTargetRejected() {
        let spec = SmeltPackageInterface(args: [
            ttsArg(flag: "voice"), ttsArg(flag: "persona"),
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func requiredWithDefaultRejected() {
        let spec = SmeltPackageInterface(args: [
            ttsArg(defaultValue: "Ryan", required: true)
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func uppercaseFlagNameRejected() {
        let spec = SmeltPackageInterface(args: [ttsArg(flag: "Voice")])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func digitLeadingFlagNameRejected() {
        // Flag grammar must stay a subset of the placeholder grammar — a
        // digit-leading flag would be undeclarable in a prompt template.
        let spec = SmeltPackageInterface(args: [ttsArg(flag: "1st")])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func bindTargetedPlaceholderInTemplateRejected() {
        let spec = SmeltPackageInterface(
            args: [
                .init(flag: "component", type: .string, required: true),
                .init(flag: "routes", type: .stringList, target: "bind:routes"),
            ],
            prompt: "Routes {routes} for {component}: {input}"
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func numericTTSTargetRequiresInt() {
        let spec = SmeltPackageInterface(args: [
            ttsArg(flag: "frames", type: .string, target: "max-frames")
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func ttsTargetlessArgRejected() {
        let spec = SmeltPackageInterface(args: [ttsArg(required: true, target: nil)])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func enumDefaultOutsideValuesRejected() {
        let spec = SmeltPackageInterface(args: [
            ttsArg(type: .enum, values: ["Ryan"], defaultValue: "Katie")
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }

    @Test func bindRequiresStringListAndViceVersa() {
        let bindAsString = SmeltPackageInterface(args: [
            .init(flag: "routes", type: .string, target: "bind:routes")
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try bindAsString.validate(interfaceContext: textGenerationInterfaceContext)
        }
        let listWithoutBind = SmeltPackageInterface(args: [
            .init(flag: "routes", type: .stringList, required: true)
        ])
        #expect(throws: SmeltPackageInterfaceError.self) {
            try listWithoutBind.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func bindWithStringListPasses() throws {
        let spec = SmeltPackageInterface(args: [
            .init(flag: "routes", type: .stringList, target: "bind:routes")
        ])
        try spec.validate(interfaceContext: textGenerationInterfaceContext)
    }

    @Test func textPromptArgsMustBeRequiredOrDefaulted() {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "component", type: .string)],
            prompt: "Component: {component}\n{input}"
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func undeclaredPlaceholderRejected() {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "component", type: .string, required: true)],
            prompt: "Component: {compnent}\n{input}"
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func unusedPromptArgRejected() {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "component", type: .string, required: true)],
            prompt: "Just {input}"
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func promptArgWithoutTemplateRejected() {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "component", type: .string, required: true)]
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: textGenerationInterfaceContext)
        }
    }

    @Test func jsonBracesPassValidation() throws {
        // A schema example in the template must not trip the placeholder
        // audit: {"severity": ...} is not an identifier-like placeholder.
        let spec = SmeltPackageInterface(
            args: [.init(flag: "component", type: .string, required: true)],
            prompt: #"Emit {"severity": "low"} style JSON for {component}: {input}"#
        )
        try spec.validate(interfaceContext: textGenerationInterfaceContext)
    }

    @Test func ttsPromptTemplateRejected() {
        let spec = SmeltPackageInterface(
            args: [ttsArg()], prompt: "{input}"
        )
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.validate(interfaceContext: codecAudioInterfaceContext)
        }
    }
}

@Suite struct SmeltPackageInterfaceScanTests {

    private let declared = SmeltPackageInterface(args: [
        .init(flag: "voice", type: .enum, values: ["Ryan", "Katie"],
              defaultValue: "Ryan", target: "speaker"),
        .init(flag: "fast", type: .bool, defaultValue: "false", target: "instruct"),
    ])

    private func scan(_ argv: [String]) throws -> SmeltPackageInterface.Scan {
        try SmeltPackageInterface.scan(
            argv: ["smelt", "run", "pkg"] + argv,
            startIndex: 3,
            builtinsWithValues: ["--seed", "--prompt", "--bind"],
            builtinBools: ["--play", "--greedy"],
            declared: declared
        )
    }

    @Test func declaredAndPositionalsSeparate() throws {
        let s = try scan(["hello", "--voice", "Katie", "world", "--play"])
        #expect(s.declaredRaw == ["voice": "Katie"])
        #expect(s.positionals == ["hello", "world"])
    }

    @Test func unknownFlagRejected() {
        #expect(throws: SmeltPackageInterfaceError.unknownFlag("--speakr")) {
            try scan(["--speakr", "Ryan"])
        }
    }

    @Test func doubleDashPassesThrough() throws {
        let s = try scan(["--voice", "Katie", "--", "--not-a-flag", "text"])
        #expect(s.positionals == ["--not-a-flag", "text"])
        #expect(s.declaredRaw == ["voice": "Katie"])
    }

    @Test func terminatorIndexReportedForCLITruncation() throws {
        // Escaped built-ins after -- must be invisible to flag parsing; the
        // CLI truncates its argv view at this index.
        let s = try scan(["hello", "--", "--seed", "9"])
        #expect(s.terminatorIndex == 4)  // ["smelt","run","pkg","hello","--",...]
        #expect(s.positionals == ["hello", "--seed", "9"])
        let none = try scan(["hello"])
        #expect(none.terminatorIndex == nil)
    }

    @Test func duplicateDeclaredFlagsInScanThrowNotTrap() {
        // A hand-edited args.json that skipped validation must fail as
        // malformed, not crash on dictionary construction.
        let dup = SmeltPackageInterface(args: [
            .init(flag: "voice", type: .string, target: "speaker"),
            .init(flag: "voice", type: .string, target: "language"),
        ])
        #expect(throws: SmeltPackageInterfaceError.malformed("flag 'voice' declared twice")) {
            _ = try SmeltPackageInterface.scan(
                argv: ["smelt", "run", "pkg"], startIndex: 3,
                builtinsWithValues: [], builtinBools: [], declared: dup
            )
        }
    }

    @Test func singleDashIsPositional() throws {
        let s = try scan(["-5", "degrees"])
        #expect(s.positionals == ["-5", "degrees"])
    }

    @Test func interactiveShortFlagCanBeDeclaredAsBuiltin() throws {
        let s = try SmeltPackageInterface.scan(
            argv: ["smelt", "run", "pkg", "-i", "hello"],
            startIndex: 3,
            builtinsWithValues: [],
            builtinBools: ["-i"],
            declared: nil
        )
        #expect(s.positionals == ["hello"])
    }

    @Test func equalsSyntaxGetsHint() {
        #expect(throws: SmeltPackageInterfaceError.self) {
            try scan(["--voice=Katie"])
        }
    }

    @Test func missingValueRejected() {
        #expect(throws: SmeltPackageInterfaceError.missingValue(flag: "--voice")) {
            try scan(["--voice"])
        }
        #expect(throws: SmeltPackageInterfaceError.missingValue(flag: "--seed")) {
            try scan(["--seed"])
        }
    }

    @Test func repeatedDeclaredFlagRejected() {
        #expect(throws: SmeltPackageInterfaceError.repeatedFlag("--voice")) {
            try scan(["--voice", "Ryan", "--voice", "Katie"])
        }
    }

    @Test func repeatedBuiltinRejectedButBindRepeatable() throws {
        #expect(throws: SmeltPackageInterfaceError.repeatedFlag("--seed")) {
            try scan(["--seed", "1", "--seed", "2"])
        }
        _ = try scan(["--bind", "a=1", "--bind", "b=2"])
    }

    @Test func boolDeclaredFlagTakesNoValue() throws {
        let s = try scan(["--fast", "hello"])
        #expect(s.declaredRaw == ["fast": "true"])
        #expect(s.positionals == ["hello"])
    }

    @Test func helpDetected() throws {
        let s = try scan(["--help"])
        #expect(s.helpRequested)
    }

    @Test func noDeclaredInterfaceStillStrict() {
        #expect(throws: SmeltPackageInterfaceError.unknownFlag("--voice")) {
            _ = try SmeltPackageInterface.scan(
                argv: ["smelt", "run", "pkg", "--voice", "Katie"],
                startIndex: 3,
                builtinsWithValues: ["--seed"],
                builtinBools: [],
                declared: nil
            )
        }
    }
}

@Suite struct SmeltPackageInterfaceResolveTests {

    private let spec = SmeltPackageInterface(args: [
        .init(flag: "voice", type: .enum, values: ["Ryan", "Katie"],
              defaultValue: "Ryan", target: "speaker"),
        .init(flag: "frames", type: .int, target: "max-frames"),
        .init(flag: "temp", type: .number, defaultValue: "0.5"),
        .init(flag: "component", type: .string, required: true),
        .init(flag: "routes", type: .stringList, target: "bind:routes"),
    ])

    @Test func defaultsAndExplicitValues() throws {
        let r = try spec.resolve(declaredRaw: [
            "component": "kernel", "frames": "128",
        ])
        #expect(r["voice"] == .string("Ryan"))
        #expect(r["frames"] == .int(128))
        #expect(r["temp"] == .number(0.5))
        #expect(r["component"] == .string("kernel"))
        #expect(r["routes"] == nil)
    }

    @Test func missingRequiredThrows() {
        #expect(throws: SmeltPackageInterfaceError.missingRequired(["component"])) {
            try spec.resolve(declaredRaw: [:])
        }
    }

    @Test func typeViolationsThrow() {
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.resolve(declaredRaw: ["component": "x", "frames": "many"])
        }
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.resolve(declaredRaw: ["component": "x", "voice": "Bob"])
        }
        #expect(throws: SmeltPackageInterfaceError.self) {
            try spec.resolve(declaredRaw: ["component": "x", "routes": "a,,b"])
        }
    }

    @Test func stringListSplits() throws {
        let r = try spec.resolve(declaredRaw: [
            "component": "x", "routes": "billing, auth,infra",
        ])
        #expect(r["routes"] == .list(["billing", "auth", "infra"]))
    }
}

@Suite struct SmeltPackageInterfaceTemplateTests {

    private let spec = SmeltPackageInterface(
        args: [
            .init(flag: "component", type: .string, required: true),
            .init(flag: "severity", type: .string, defaultValue: "low"),
        ],
        prompt: #"Component: {component} ({severity})\#nJSON like {"a": 1} and {unknown} stay.\#n{input}"#
    )

    @Test func fillsDeclaredAndInput() throws {
        let out = try spec.fillPrompt(
            resolved: ["component": .string("kernel"), "severity": .string("high")],
            input: "the report"
        )
        #expect(out == #"Component: kernel (high)\#nJSON like {"a": 1} and {unknown} stay.\#nthe report"#)
    }

    @Test func noTemplatePassesInputThrough() throws {
        let bare = SmeltPackageInterface(args: [])
        #expect(try bare.fillPrompt(resolved: [:], input: "hi") == "hi")
    }

    @Test func unterminatedBraceVerbatim() throws {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "a", type: .string, required: true)],
            prompt: "x {a} tail{unclosed"
        )
        let out = try spec.fillPrompt(resolved: ["a": .string("V")], input: "")
        #expect(out == "x V tail{unclosed")
    }

    @Test func listValueRendersCommaJoined() throws {
        let spec = SmeltPackageInterface(
            args: [.init(flag: "routes", type: .stringList, required: true)],
            prompt: "Routes: {routes}"
        )
        let out = try spec.fillPrompt(
            resolved: ["routes": .list(["a", "b"])], input: ""
        )
        #expect(out == "Routes: a, b")
    }

    @Test func targetedFlagNameStaysVerbatimInFill() throws {
        // Only targetless args are placeholders; a bind-targeted flag's
        // name in braces passes through untouched.
        let spec = SmeltPackageInterface(
            args: [.init(flag: "routes", type: .stringList, target: "bind:routes")],
            prompt: "Routes {routes}: {input}"
        )
        let out = try spec.fillPrompt(
            resolved: ["routes": .list(["a"])], input: "hi"
        )
        #expect(out == "Routes {routes}: hi")
    }
}
