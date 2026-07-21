// SmeltPackageInterface — a package-declared CLI interface (`args.json`).
// A compiled package can expose typed flags through `smelt run`; targets map to
// runtime capabilities such as audio fields, prompt placeholders, and grammar
// bindings. Presentation and agent policy belong to downstream consumers.
//
// Design: docs/run-args-unification-plan.md. Everything here is pure
// (decode / validate / argv scan / resolve / template fill) so the CLI stays
// a thin wrapper and the logic is unit-testable.

import Foundation

public enum SmeltPackageInterfaceError: Error, CustomStringConvertible, Equatable {
    case malformed(String)
    case unknownFlag(String)
    case missingValue(flag: String)
    case equalsSyntax(token: String, hint: String)
    case repeatedFlag(String)
    case badValue(flag: String, value: String, expected: String)
    case missingRequired([String])
    case conflict(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "args.json: \(why)"
        case .unknownFlag(let flag):
            return "unknown flag \(flag) (use --help for this package's flags; "
                + "`--` or --prompt passes literal text through)"
        case .missingValue(let flag): return "\(flag) expects a value"
        case .equalsSyntax(let token, let hint):
            return "\(token): use space-separated values (\(hint))"
        case .repeatedFlag(let flag): return "\(flag) given more than once"
        case .badValue(let flag, let value, let expected):
            return "\(flag): expected \(expected), got '\(value)'"
        case .missingRequired(let flags):
            return "missing required flag\(flags.count == 1 ? "" : "s"): "
                + flags.map { "--\($0)" }.joined(separator: ", ")
        case .conflict(let why): return why
        }
    }
}

/// A resolved declared-arg value, typed per the declaration.
public enum SmeltPackageArgumentValue: Equatable, Sendable {
    case string(String)
    case int(Int)
    case number(Double)
    case bool(Bool)
    case list([String])

    /// The value as prompt-template text.
    public var templateText: String {
        switch self {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .number(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        case .list(let items): return items.joined(separator: ", ")
        }
    }
}

public struct SmeltPackageInterface: Sendable, Equatable {

    public enum ArgType: String, Codable, Sendable {
        case string
        case int
        case number
        case bool
        case `enum`
        case stringList = "string-list"
    }

    public struct Arg: Sendable, Equatable {
        public let flag: String
        public let type: ArgType
        public let values: [String]?     // enum only
        public let defaultValue: String? // canonical string form; parsed per type
        public let required: Bool
        public let description: String?
        public let target: String?       // nil -> text prompt placeholder {flag}

        public init(
            flag: String, type: ArgType, values: [String]? = nil,
            defaultValue: String? = nil, required: Bool = false,
            description: String? = nil, target: String? = nil
        ) {
            self.flag = flag
            self.type = type
            self.values = values
            self.defaultValue = defaultValue
            self.required = required
            self.description = description
            self.target = target
        }
    }

    public let version: Int
    public let args: [Arg]
    /// Text prompt template; `{flag}` and `{input}` placeholders.
    public let prompt: String?
    public init(
        version: Int = 1,
        args: [Arg],
        prompt: String? = nil
    ) {
        self.version = version
        self.args = args
        self.prompt = prompt
    }

    public static let fileName = "args.json"

    // MARK: - Decode / encode

    /// JSON shape, kept separate from the public model so `default` can be a
    /// string, number, or bool in the file but canonical (string) in memory.
    private struct ArgJSON: Codable {
        let flag: String
        let type: ArgType
        let values: [String]?
        let `default`: JSONScalar?
        let required: Bool?
        let description: String?
        let target: String?
    }

    private struct FileJSON: Codable {
        let version: Int
        let args: [ArgJSON]?
        let prompt: String?
    }

    private enum JSONScalar: Codable {
        case string(String)
        case int(Int)
        case number(Double)
        case bool(Bool)

        init(from decoder: Decoder) throws {
            let single = try decoder.singleValueContainer()
            // Bool before Int: JSONDecoder bridges NSNumber, and a bare
            // `true` must not decode as 1.
            if let b = try? single.decode(Bool.self) { self = .bool(b); return }
            if let i = try? single.decode(Int.self) { self = .int(i); return }
            if let d = try? single.decode(Double.self) { self = .number(d); return }
            if let s = try? single.decode(String.self) { self = .string(s); return }
            throw DecodingError.typeMismatch(
                JSONScalar.self,
                .init(codingPath: decoder.codingPath,
                      debugDescription: "default must be a string, number, or bool")
            )
        }

        func encode(to encoder: Encoder) throws {
            var single = encoder.singleValueContainer()
            switch self {
            case .string(let s): try single.encode(s)
            case .int(let i): try single.encode(i)
            case .number(let d): try single.encode(d)
            case .bool(let b): try single.encode(b)
            }
        }

        var canonical: String {
            switch self {
            case .string(let s): return s
            case .int(let i): return String(i)
            case .number(let d): return String(d)
            case .bool(let b): return b ? "true" : "false"
            }
        }
    }

    public static func decode(from data: Data) throws -> SmeltPackageInterface {
        let file: FileJSON
        do {
            file = try JSONDecoder().decode(FileJSON.self, from: data)
        } catch {
            throw SmeltPackageInterfaceError.malformed("\(error)")
        }
        return SmeltPackageInterface(
            version: file.version,
            args: (file.args ?? []).map {
                Arg(
                    flag: $0.flag, type: $0.type, values: $0.values,
                    defaultValue: $0.default?.canonical,
                    required: $0.required ?? false,
                    description: $0.description, target: $0.target
                )
            },
            prompt: file.prompt
        )
    }

    /// The declared interface of the package at `packagePath`, or nil when
    /// none was prepared. Throws on a present-but-malformed file (a broken
    /// interface must fail loudly, not degrade to undeclared flags).
    public static func load(packagePath: String) throws -> SmeltPackageInterface? {
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try decode(from: Data(contentsOf: url))
    }

    // MARK: - Validation

    /// The allowed declared-arg targets and their types for a package, chosen from
    /// the package's graph policy. The args layer validates against this and never
    /// branches on architecture. Text generation uses prompt placeholders + bind
    /// slots structurally, independent of this policy.
    public struct ArgTargetPolicy: Sendable {
        /// target name → the arg types allowed for it.
        public let targets: [String: Set<ArgType>]
        public init(targets: [String: Set<ArgType>]) { self.targets = targets }

        public static let none = ArgTargetPolicy(targets: [:])
        public static let codecAudio = ArgTargetPolicy(targets: [
            "speaker": [.string, .enum], "language": [.string, .enum],
            "instruct": [.string, .enum],
            "first-chunk": [.int], "max-chunk": [.int], "max-frames": [.int],
        ])
    }

    public struct RunBuiltinFlags: Sendable, Equatable {
        public let value: Set<String>
        public let bool: Set<String>

        public init(value: Set<String>, bool: Set<String>) {
            self.value = value
            self.bool = bool
        }

        public var all: Set<String> { value.union(bool) }
        public var bareNames: Set<String> { Self.bareNames(all) }

        public static func bareNames(_ flags: Set<String>) -> Set<String> {
            Set(flags.map { flag in
                flag.hasPrefix("--") ? String(flag.dropFirst(2)) : flag
            })
        }

        public static let textGeneration = RunBuiltinFlags(
            value: [
                "--package", "--prompt", "--max-tokens", "--context-limit",
                "--template", "--temp", "--seed", "--system", "--system-file",
                "--prompt-file", "--linger", "--bind",
                "--stop-after-dispatch", "--debug-token-index", "--debug-prefix-tokens",
                "--trace-verify-token-count", "--dump-trace-label",
                "--dump-trace-occurrence", "--dump-slot", "--dump-offset",
                "--dump-count",
            ],
            bool: [
                "--debug", "--trace-layers", "--debug-prefill", "--force-decode",
            ]
        )

        public static let codecAudio = RunBuiltinFlags(
            value: [
                "--package", "--prompt", "--speaker", "--language", "--instruct",
                "--seed", "--max-frames", "--first-chunk", "--max-chunk",
                "--linger",
            ],
            bool: ["--greedy"]
        )
    }

    public struct InterfaceValidationContext: Sendable {
        public let policy: ArgTargetPolicy
        public let builtinFlags: Set<String>
        fileprivate let interfaceName: String
        fileprivate let promptTemplatesAllowed: Bool
        fileprivate let bindTargetsAllowed: Bool
        fileprivate let declaredTargetsRequired: Bool
        fileprivate let unsupportedArgsMessage: String?

        public init(
            interfaceName: String,
            policy: ArgTargetPolicy,
            builtinFlags: Set<String>,
            promptTemplatesAllowed: Bool,
            bindTargetsAllowed: Bool,
            declaredTargetsRequired: Bool,
            unsupportedArgsMessage: String? = nil
        ) {
            self.interfaceName = interfaceName
            self.policy = policy
            self.builtinFlags = builtinFlags
            self.promptTemplatesAllowed = promptTemplatesAllowed
            self.bindTargetsAllowed = bindTargetsAllowed
            self.declaredTargetsRequired = declaredTargetsRequired
            self.unsupportedArgsMessage = unsupportedArgsMessage
        }

        public static func textGeneration(
            builtinFlags: Set<String>
        ) -> InterfaceValidationContext {
            InterfaceValidationContext(
                interfaceName: "text-generation",
                policy: .none,
                builtinFlags: builtinFlags,
                promptTemplatesAllowed: true,
                bindTargetsAllowed: true,
                declaredTargetsRequired: false
            )
        }

        public static func targeted(
            interfaceName: String,
            policy: ArgTargetPolicy,
            builtinFlags: Set<String>
        ) -> InterfaceValidationContext {
            InterfaceValidationContext(
                interfaceName: interfaceName,
                policy: policy,
                builtinFlags: builtinFlags,
                promptTemplatesAllowed: false,
                bindTargetsAllowed: false,
                declaredTargetsRequired: true
            )
        }

        public static func noDeclaredArgs(
            interfaceName: String,
            builtinFlags: Set<String>
        ) -> InterfaceValidationContext {
            InterfaceValidationContext(
                interfaceName: interfaceName,
                policy: .none,
                builtinFlags: builtinFlags,
                promptTemplatesAllowed: false,
                bindTargetsAllowed: false,
                declaredTargetsRequired: false,
                unsupportedArgsMessage: "\(interfaceName) packages support no declared args yet"
            )
        }
    }

    public struct PackageValidationContext: Sendable {
        public let interfaceContext: InterfaceValidationContext
        public let builtins: RunBuiltinFlags

        public init(
            interfaceContext: InterfaceValidationContext,
            builtins: RunBuiltinFlags
        ) {
            self.interfaceContext = interfaceContext
            self.builtins = builtins
        }
    }

    public static func packageValidationContext(
        graphPolicy: SmeltRuntimeGraphPolicy
    ) -> PackageValidationContext {
        switch graphPolicy {
        case .textGeneration:
            return PackageValidationContext(
                interfaceContext: .textGeneration(builtinFlags: RunBuiltinFlags.textGeneration.bareNames),
                builtins: .textGeneration
            )
        case .sidecarTextToCodecAudio:
            return PackageValidationContext(
                interfaceContext: .targeted(
                    interfaceName: "targeted audio",
                    policy: .codecAudio,
                    builtinFlags: RunBuiltinFlags.codecAudio.bareNames
                ),
                builtins: .codecAudio
            )
        case .codecAudio:
            return PackageValidationContext(
                interfaceContext: .noDeclaredArgs(
                    interfaceName: "codec audio",
                    builtinFlags: []
                ),
                builtins: .init(value: [], bool: [])
            )
        }
    }

    public static func packageValidationContext(
        manifestData: Data
    ) throws -> PackageValidationContext {
        try packageValidationContext(
            graphPolicy: SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData)
        )
    }

    public func validate(packageContext context: PackageValidationContext) throws {
        try validate(interfaceContext: context.interfaceContext)
    }

    public static let bindTargetPrefix = "bind:"

    private static let flagPattern = try! NSRegularExpression(
        pattern: "^[a-z][a-z0-9-]*$"
    )
    private static let placeholderPattern = try! NSRegularExpression(
        pattern: "\\{([a-z][a-z0-9-]*)\\}"
    )

    private static func isValidFlagName(_ name: String) -> Bool {
        let range = NSRange(name.startIndex..., in: name)
        return flagPattern.firstMatch(in: name, range: range) != nil
    }

    /// Identifier-like `{placeholder}` names appearing in `template`.
    public static func placeholders(in template: String) -> [String] {
        let range = NSRange(template.startIndex..., in: template)
        return placeholderPattern.matches(in: template, range: range).compactMap {
            Range($0.range(at: 1), in: template).map { String(template[$0]) }
        }
    }

    public func validate(interfaceContext context: InterfaceValidationContext) throws {
        guard version == 1 else {
            throw SmeltPackageInterfaceError.malformed("unsupported version \(version)")
        }
        guard !args.isEmpty || prompt != nil else {
            throw SmeltPackageInterfaceError.malformed(
                "declares no args or prompt template"
            )
        }

        var seenFlags = Set<String>()
        var seenTargets = Set<String>()
        for arg in args {
            guard Self.isValidFlagName(arg.flag) else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)' must match [a-z][a-z0-9-]*"
                )
            }
            guard arg.flag != "help" else {
                throw SmeltPackageInterfaceError.malformed("flag 'help' is reserved")
            }
            guard !context.builtinFlags.contains(arg.flag) else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)' shadows the built-in --\(arg.flag)"
                )
            }
            guard seenFlags.insert(arg.flag).inserted else {
                throw SmeltPackageInterfaceError.malformed("flag '\(arg.flag)' declared twice")
            }
            if arg.required && arg.defaultValue != nil {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)' is required but has a default — pick one"
                )
            }

            switch arg.type {
            case .enum:
                guard let values = arg.values, !values.isEmpty else {
                    throw SmeltPackageInterfaceError.malformed(
                        "enum flag '\(arg.flag)' needs non-empty \"values\""
                    )
                }
                guard Set(values).count == values.count else {
                    throw SmeltPackageInterfaceError.malformed(
                        "enum flag '\(arg.flag)' has duplicate values"
                    )
                }
            default:
                guard arg.values == nil else {
                    throw SmeltPackageInterfaceError.malformed(
                        "flag '\(arg.flag)': \"values\" is only valid for enum"
                    )
                }
            }

            // Defaults must parse like a CLI value would.
            if let def = arg.defaultValue {
                _ = try Self.parseValue(def, for: arg)
            }

            try validateTarget(
                of: arg, context: context, seenTargets: &seenTargets)
        }

        try validatePromptTemplate(context: context)
    }

    private func validateTarget(
        of arg: Arg, context: InterfaceValidationContext,
        seenTargets: inout Set<String>
    ) throws {
        if let unsupportedArgsMessage = context.unsupportedArgsMessage {
            throw SmeltPackageInterfaceError.malformed(
                "flag '\(arg.flag)': \(unsupportedArgsMessage)"
            )
        }

        let isBind = arg.target?.hasPrefix(Self.bindTargetPrefix) ?? false
        if isBind || arg.type == .stringList {
            // string-list ⇔ bind: CSV lists exist to feed grammar slots.
            guard isBind && arg.type == .stringList && context.bindTargetsAllowed else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': bind targets require type string-list "
                        + "(and vice versa), on a text-generation interface"
                )
            }
            let name = String(arg.target!.dropFirst(Self.bindTargetPrefix.count))
            guard !name.isEmpty else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': empty bind slot name"
                )
            }
        }

        if context.declaredTargetsRequired {
            guard let target = arg.target else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': args need a target "
                        + "(\(context.policy.targets.keys.sorted().joined(separator: ", ")))"
                )
            }
            guard let allowedTypes = context.policy.targets[target] else {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': unknown target '\(target)'"
                )
            }
            guard allowedTypes.contains(arg.type) else {
                let expected = allowedTypes.map(\.rawValue).sorted()
                    .joined(separator: " or ")
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': target '\(target)' requires type \(expected)"
                )
            }
        } else {
            if let target = arg.target, !target.hasPrefix(Self.bindTargetPrefix) {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': unknown \(context.interfaceName) target '\(target)' "
                        + "(omit for a prompt placeholder, or use bind:NAME)"
                )
            }
            // A prompt placeholder with neither default nor required can't
            // render its template when absent — reject the package shape.
            if arg.target == nil, !arg.required, arg.defaultValue == nil {
                throw SmeltPackageInterfaceError.malformed(
                    "flag '\(arg.flag)': prompt placeholders must be required "
                        + "or carry a default"
                )
            }
        }

        if let target = arg.target {
            guard seenTargets.insert(target).inserted else {
                throw SmeltPackageInterfaceError.malformed(
                    "two flags target '\(target)'"
                )
            }
        }
    }

    private func validatePromptTemplate(context: InterfaceValidationContext) throws {
        let promptArgs = args.filter { $0.target == nil }
        if !context.promptTemplatesAllowed {
            guard prompt == nil else {
                throw SmeltPackageInterfaceError.malformed(
                    "prompt templates are for text-generation interfaces"
                )
            }
        } else {
            guard let prompt else {
                if let first = promptArgs.first {
                    throw SmeltPackageInterfaceError.malformed(
                        "flag '\(first.flag)' is a prompt placeholder but there "
                            + "is no prompt template"
                    )
                }
                return
            }
            let found = Set(Self.placeholders(in: prompt))
            // Identifier-like placeholders must resolve — a typo'd
            // {compnent} silently passing through is exactly the failure
            // mode declared interfaces exist to prevent. Only targetless
            // args are placeholders: a bind-targeted {routes} in the
            // template is a spec error, not a substitution.
            let placeholderFlags = Set(promptArgs.map(\.flag))
            let unknown = found.subtracting(placeholderFlags).subtracting(["input"])
            guard unknown.isEmpty else {
                throw SmeltPackageInterfaceError.malformed(
                    "prompt template references undeclared placeholder\(unknown.count == 1 ? "" : "s") "
                        + unknown.sorted().map { "{\($0)}" }.joined(separator: ", ")
                )
            }
            // And a declared prompt arg the template never uses is dead
            // interface — reject during package preparation, not at run.
            let unused = promptArgs.map(\.flag).filter { !found.contains($0) }
            guard unused.isEmpty else {
                throw SmeltPackageInterfaceError.malformed(
                    "prompt template never references "
                        + unused.map { "{\($0)}" }.joined(separator: ", ")
                )
            }
        }
    }

    // MARK: - Argv scan

    /// One strict pass over `argv[startIndex...]`.
    ///
    /// - `--` ends flag parsing; later tokens are positional verbatim.
    /// - Only `--`-prefixed tokens are flags; single-dash tokens are
    ///   positional (negative numbers never need escaping).
    /// - Built-ins keep their existing parse sites; this scan only needs to
    ///   know which take values (to skip them) and which are bare bools.
    /// - Declared flags are collected raw here; `resolve` types them.
    public struct Scan: Equatable, Sendable {
        public let declaredRaw: [String: String]
        public let positionals: [String]
        public let helpRequested: Bool
        /// Absolute argv index of the `--` terminator, if present. The CLI
        /// must stop its own flag parsing there too — escaped text after
        /// `--` must never reach a `--flag VALUE` lookup.
        public let terminatorIndex: Int?
    }

    public static func scan(
        argv: [String],
        startIndex: Int,
        builtinsWithValues: Set<String>,
        builtinBools: Set<String>,
        repeatableBuiltins: Set<String> = ["--bind"],
        declared: SmeltPackageInterface?
    ) throws -> Scan {
        // A hand-edited args.json can carry duplicate flags even though
        // validation rejects them — fail as malformed, never trap.
        var declaredByFlag: [String: Arg] = [:]
        for arg in declared?.args ?? [] {
            guard declaredByFlag.updateValue(arg, forKey: "--\(arg.flag)") == nil else {
                throw SmeltPackageInterfaceError.malformed("flag '\(arg.flag)' declared twice")
            }
        }
        var declaredRaw: [String: String] = [:]
        var positionals: [String] = []
        var seenBuiltins = Set<String>()
        var helpRequested = false
        var terminatorIndex: Int?

        var idx = max(0, startIndex)
        while idx < argv.count {
            let token = argv[idx]
            if token == "--" {
                terminatorIndex = idx
                positionals.append(contentsOf: argv[(idx + 1)...])
                break
            }
            let isShortBuiltinBool = token.hasPrefix("-") && builtinBools.contains(token)
            guard token.hasPrefix("--") || isShortBuiltinBool else {
                positionals.append(token)
                idx += 1
                continue
            }
            if token == "--help" {
                helpRequested = true
                idx += 1
                continue
            }
            if let eq = token.firstIndex(of: "=") {
                let name = String(token[..<eq])
                throw SmeltPackageInterfaceError.equalsSyntax(
                    token: token,
                    hint: "\(name) \(token[token.index(after: eq)...])"
                )
            }
            if let arg = declaredByFlag[token] {
                if arg.type == .bool {
                    guard declaredRaw[arg.flag] == nil else {
                        throw SmeltPackageInterfaceError.repeatedFlag(token)
                    }
                    declaredRaw[arg.flag] = "true"
                    idx += 1
                    continue
                }
                guard idx + 1 < argv.count else {
                    throw SmeltPackageInterfaceError.missingValue(flag: token)
                }
                guard declaredRaw[arg.flag] == nil else {
                    throw SmeltPackageInterfaceError.repeatedFlag(token)
                }
                declaredRaw[arg.flag] = argv[idx + 1]
                idx += 2
                continue
            }
            if builtinsWithValues.contains(token) {
                guard idx + 1 < argv.count else {
                    throw SmeltPackageInterfaceError.missingValue(flag: token)
                }
                if !repeatableBuiltins.contains(token),
                   !seenBuiltins.insert(token).inserted {
                    throw SmeltPackageInterfaceError.repeatedFlag(token)
                }
                idx += 2
                continue
            }
            if builtinBools.contains(token) {
                idx += 1
                continue
            }
            throw SmeltPackageInterfaceError.unknownFlag(token)
        }

        return Scan(
            declaredRaw: declaredRaw,
            positionals: positionals,
            helpRequested: helpRequested,
            terminatorIndex: terminatorIndex
        )
    }

    // MARK: - Resolve

    /// Type and default the scanned values. Returns only declared args
    /// (target application is the caller's job — it owns the precedence
    /// chain against prepared and built-in values).
    public func resolve(
        declaredRaw: [String: String]
    ) throws -> [String: SmeltPackageArgumentValue] {
        var resolved: [String: SmeltPackageArgumentValue] = [:]
        var missing: [String] = []
        for arg in args {
            if let raw = declaredRaw[arg.flag] {
                resolved[arg.flag] = try Self.parseValue(raw, for: arg)
            } else if let def = arg.defaultValue {
                resolved[arg.flag] = try Self.parseValue(def, for: arg)
            } else if arg.required {
                missing.append(arg.flag)
            }
        }
        guard missing.isEmpty else {
            throw SmeltPackageInterfaceError.missingRequired(missing)
        }
        return resolved
    }

    private static func parseValue(
        _ raw: String, for arg: Arg
    ) throws -> SmeltPackageArgumentValue {
        switch arg.type {
        case .string:
            return .string(raw)
        case .enum:
            guard arg.values?.contains(raw) ?? false else {
                throw SmeltPackageInterfaceError.badValue(
                    flag: "--\(arg.flag)", value: raw,
                    expected: "one of \((arg.values ?? []).joined(separator: ", "))"
                )
            }
            return .string(raw)
        case .int:
            guard let v = Int(raw) else {
                throw SmeltPackageInterfaceError.badValue(
                    flag: "--\(arg.flag)", value: raw, expected: "an integer"
                )
            }
            return .int(v)
        case .number:
            guard let v = Double(raw), v.isFinite else {
                throw SmeltPackageInterfaceError.badValue(
                    flag: "--\(arg.flag)", value: raw, expected: "a finite number"
                )
            }
            return .number(v)
        case .bool:
            guard raw == "true" || raw == "false" else {
                throw SmeltPackageInterfaceError.badValue(
                    flag: "--\(arg.flag)", value: raw, expected: "true or false"
                )
            }
            return .bool(raw == "true")
        case .stringList:
            let items = raw.split(separator: ",", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            guard !items.isEmpty, !items.contains("") else {
                throw SmeltPackageInterfaceError.badValue(
                    flag: "--\(arg.flag)", value: raw,
                    expected: "V1,V2,... with no empty items"
                )
            }
            return .list(items)
        }
    }

    // MARK: - Prompt template fill

    /// Substitute `{flag}`/`{input}` in the template. Only exact declared
    /// names (and `input`) are placeholders; all other braces — JSON
    /// examples, prose — pass through verbatim, so no escaping syntax.
    public func fillPrompt(
        resolved: [String: SmeltPackageArgumentValue], input: String
    ) throws -> String {
        guard let prompt else { return input }
        // Only targetless args substitute — a bind-targeted flag's name in
        // braces stays verbatim (and validation rejects it in the template).
        var names = Set(args.filter { $0.target == nil }.map(\.flag))
        names.insert("input")
        var out = ""
        var rest = Substring(prompt)
        while let open = rest.firstIndex(of: "{") {
            out += rest[..<open]
            let afterOpen = rest.index(after: open)
            guard let close = rest[afterOpen...].firstIndex(of: "}") else {
                out += rest[open...]
                rest = Substring("")
                break
            }
            let name = String(rest[afterOpen..<close])
            if name == "input" {
                out += input
                rest = rest[rest.index(after: close)...]
            } else if names.contains(name) {
                guard let value = resolved[name] else {
                    // Validated interfaces can still meet an optional,
                    // defaultless arg the invocation didn't pass.
                    throw SmeltPackageInterfaceError.missingRequired([name])
                }
                out += value.templateText
                rest = rest[rest.index(after: close)...]
            } else {
                // Not a placeholder — emit the "{" and rescan from the next
                // character so an inner "{name}" can still match.
                out += "{"
                rest = rest[afterOpen...]
            }
        }
        out += rest
        return out
    }

    // MARK: - Help rendering

    /// The declared-args section of `smelt run <pkg> --help`.
    public func helpLines() -> [String] {
        guard !args.isEmpty else { return [] }
        var lines = ["Package flags (declared by this package):"]
        for arg in args {
            var spec = "--\(arg.flag)"
            switch arg.type {
            case .bool: break
            case .enum: spec += " <\((arg.values ?? []).joined(separator: "|"))>"
            case .stringList: spec += " V1,V2,..."
            default: spec += " <\(arg.type.rawValue)>"
            }
            var notes: [String] = []
            if let d = arg.description { notes.append(d) }
            if arg.required { notes.append("required") }
            if let def = arg.defaultValue { notes.append("default: \(def)") }
            let pad = spec.count < 26
                ? String(repeating: " ", count: 26 - spec.count) : "  "
            lines.append("  \(spec)\(pad)\(notes.joined(separator: "; "))")
        }
        return lines
    }
}
