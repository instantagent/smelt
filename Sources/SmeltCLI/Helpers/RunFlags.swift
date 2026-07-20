import Foundation
import SmeltSchema

// The complete built-in flag registry for `smelt run`, per graph policy.
// Strict argv depends on this being exhaustive: a flag parsed anywhere on a
// run path but missing here is rejected as unknown before it can act.

enum RunFlags {
    /// Text run path (Run.swift + the RunPrompt debug/trace surface):
    /// flags that take a value.
    static let textValue = SmeltPackageInterface.RunBuiltinFlags.textGeneration.value
    /// Text run path: bare boolean flags.
    static let textBool = SmeltPackageInterface.RunBuiltinFlags.textGeneration.bool

    /// Text-to-PCM run path (TextToPCMRun.swift): flags that take a value.
    static let textToPCMValue = SmeltPackageInterface.RunBuiltinFlags.codecAudio.value
    /// Text-to-PCM run path: bare boolean flags.
    static let textToPCMBool = SmeltPackageInterface.RunBuiltinFlags.codecAudio.bool

    /// Declared-arg names may not shadow these (create-time check); built-ins
    /// are spelled without the leading dashes in args.json validation.
    static func bareNames(_ flags: Set<String>) -> Set<String> {
        SmeltPackageInterface.RunBuiltinFlags.bareNames(flags)
    }
}

/// Merge declared bind-targeted args into the explicit `--bind` map.
/// An explicitly passed declared flag and an explicit `--bind` for the same
/// slot are equal precedence — a conflict; a declared *default* yields to an
/// explicit `--bind`.
func mergeDeclaredBindings(
    interface: SmeltPackageInterface?,
    scanned: SmeltPackageInterface.Scan,
    resolved: [String: SmeltPackageArgumentValue],
    explicit: [String: [String]]
) throws -> [String: [String]] {
    var merged = explicit
    for arg in interface?.args ?? [] {
        guard let target = arg.target,
              target.hasPrefix(SmeltPackageInterface.bindTargetPrefix),
              case .list(let values)? = resolved[arg.flag]
        else { continue }
        let name = String(target.dropFirst(SmeltPackageInterface.bindTargetPrefix.count))
        if merged[name] != nil {
            guard scanned.declaredRaw[arg.flag] == nil else {
                throw CLIError(
                    "--\(arg.flag) and --bind \(name)=... both bind '\(name)'"
                )
            }
            continue
        }
        merged[name] = values
    }
    return merged
}

/// A user-facing CLI error that prints as its message, not an NSError dump.
struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

/// The package's declared interface, or a loud exit on a malformed file —
/// a broken interface must never degrade to undeclared flags. Re-validates
/// the shape at load: a hand-edited args.json that shadows a built-in or
/// mistypes a target must not alter flag semantics just because it skipped
/// `smelt create --args`.
func loadPackageInterface(
    packagePath: String,
    graphPolicy: SmeltRuntimeGraphPolicy
) -> SmeltPackageInterface? {
    loadPackageInterface(
        packagePath: packagePath,
        interfaceContext: packageInterfaceContext(graphPolicy: graphPolicy)
    )
}

func packageInterfaceContext(
    graphPolicy: SmeltRuntimeGraphPolicy
) -> SmeltPackageInterface.InterfaceValidationContext {
    SmeltPackageInterface.packageValidationContext(graphPolicy: graphPolicy).interfaceContext
}

struct RunPackageInterface {
    let declared: SmeltPackageInterface?
    let builtins: SmeltPackageInterface.RunBuiltinFlags
    let graphPolicy: SmeltRuntimeGraphPolicy
}

func loadRunPackageInterface(
    packagePath: String
) -> RunPackageInterface {
    do {
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json"))
        let context = try SmeltPackageInterface.packageValidationContext(manifestData: manifestData)
        return RunPackageInterface(
            declared: loadPackageInterface(
                packagePath: packagePath,
                interfaceContext: context.interfaceContext
            ),
            builtins: context.builtins,
            graphPolicy: try SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData)
        )
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
}

private func loadPackageInterface(
    packagePath: String,
    interfaceContext: SmeltPackageInterface.InterfaceValidationContext
) -> SmeltPackageInterface? {
    do {
        // Enforce bake honesty before the declared interface is exposed — this
        // runs at the CLI layer, ahead of runtime construction, so a stray or
        // declared-but-missing args.json can't surface on --help or be forwarded
        // to a linger worker before the runtime would have caught it.
        try SmeltBakeManifest.enforce(
            packagePath: packagePath,
            ignoring: SmeltBakeManifest.ignoredFromEnv(),
            argsValidationContext: interfaceContext
        )
        guard let interface = try SmeltPackageInterface.load(packagePath: packagePath)
        else { return nil }
        try interface.validate(interfaceContext: interfaceContext)
        return interface
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
}
