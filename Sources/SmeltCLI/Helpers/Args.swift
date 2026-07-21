import Foundation
import SmeltSchema

func parseArg(_ flag: String, default defaultValue: String = "") -> String {
    parseArg(args, flag, default: defaultValue)
}

func parseArg(
    _ arguments: [String],
    _ flag: String,
    default defaultValue: String = ""
) -> String {
    if let idx = arguments.firstIndex(of: flag), idx + 1 < arguments.count {
        return arguments[idx + 1]
    }
    return defaultValue
}

struct ScopedFlagReader {
    let argv: [String]
    let startIndex: Int
    let endIndex: Int
    let valueFlags: Set<String>
    let boolFlags: Set<String>

    init(
        argv: [String],
        startIndex: Int,
        terminatorIndex: Int?,
        valueFlags: Set<String> = [],
        boolFlags: Set<String> = []
    ) {
        self.argv = argv
        self.startIndex = max(0, startIndex)
        self.endIndex = min(terminatorIndex ?? argv.count, argv.count)
        self.valueFlags = valueFlags
        self.boolFlags = boolFlags
    }

    func value(_ flag: String, default defaultValue: String = "") -> String {
        var idx = startIndex
        while idx < endIndex {
            let token = argv[idx]
            if valueFlags.contains(token) {
                if token == flag, idx + 1 < endIndex {
                    return argv[idx + 1]
                }
                idx += 2
                continue
            }
            if boolFlags.contains(token) {
                idx += 1
                continue
            }
            if token == flag, idx + 1 < endIndex {
                return argv[idx + 1]
            }
            idx += 1
        }
        return defaultValue
    }

    func values(_ flag: String) -> [String] {
        var values: [String] = []
        var idx = startIndex
        while idx < endIndex {
            let token = argv[idx]
            if valueFlags.contains(token) {
                if token == flag, idx + 1 < endIndex {
                    values.append(argv[idx + 1])
                }
                idx += 2
                continue
            }
            if boolFlags.contains(token) {
                idx += 1
                continue
            }
            idx += 1
        }
        return values
    }

    func has(_ flag: String) -> Bool {
        var idx = startIndex
        while idx < endIndex {
            let token = argv[idx]
            if valueFlags.contains(token) {
                idx += 2
                continue
            }
            if boolFlags.contains(token) {
                if token == flag { return true }
                idx += 1
                continue
            }
            if token == flag { return true }
            idx += 1
        }
        return false
    }

    func positiveIntOrExit(_ flag: String, verb: String) -> Int? {
        let raw = value(flag)
        guard !raw.isEmpty else { return nil }
        guard let value = Int(raw), value > 0 else {
            fputs("smelt \(verb): \(flag) must be a positive integer, got '\(raw)'\n", stderr)
            exit(1)
        }
        return value
    }

    func nonNegativeIntOrExit(_ flag: String, verb: String) -> Int? {
        let raw = value(flag)
        guard !raw.isEmpty else { return nil }
        guard let value = Int(raw), value >= 0 else {
            fputs("smelt \(verb): \(flag) must be a non-negative integer, got '\(raw)'\n", stderr)
            exit(1)
        }
        return value
    }
}

func scopedDeclaredRunFlags(_ declared: SmeltPackageInterface?) -> SmeltPackageInterface.RunBuiltinFlags {
    guard let declared else { return .init(value: [], bool: []) }
    var value = Set<String>()
    var bool = Set<String>()
    for arg in declared.args {
        let flag = "--\(arg.flag)"
        if arg.type == .bool {
            bool.insert(flag)
        } else {
            value.insert(flag)
        }
    }
    return .init(value: value, bool: bool)
}

func hasArg(_ flag: String) -> Bool {
    args.contains(flag)
}

func hasArg(_ arguments: [String], _ flag: String) -> Bool {
    arguments.contains(flag)
}

func parseRepeatedArg(_ flag: String) -> [String] {
    var values: [String] = []
    var idx = 0
    while idx < args.count {
        if args[idx] == flag, idx + 1 < args.count {
            values.append(args[idx + 1])
            idx += 2
            continue
        }
        idx += 1
    }
    return values
}

/// Parse repeated `--bind NAME=V1,V2,...` flags into grammar slot bindings.
func parseGrammarBindings() throws -> [String: [String]] {
    if args.last == "--bind" {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "--bind expects NAME=V1,V2,... with at least one non-empty value (got \"\")"
            ]
        )
    }
    return try parseGrammarBindings(rawValues: parseRepeatedArg("--bind"))
}

func parseGrammarBindings(rawValues: [String]) throws -> [String: [String]] {
    func malformed(_ raw: String) -> NSError {
        NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "--bind expects NAME=V1,V2,... with at least one non-empty value (got \"\(raw)\")"
            ]
        )
    }
    var bindings: [String: [String]] = [:]
    for raw in rawValues {
        let parts = raw.split(separator: "=", maxSplits: 1)
        let name = parts.isEmpty ? "" : String(parts[0])
        let values = parts.count == 2
            ? parts[1].split(separator: ",", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            : []
        guard !name.isEmpty, !values.isEmpty, !values.contains("") else {
            throw malformed(raw)
        }
        guard bindings[name] == nil else {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "--bind \(name) given more than once"
                ]
            )
        }
        bindings[name] = values
    }
    return bindings
}

func hasAnyArg(_ flags: [String]) -> Bool {
    flags.contains { hasArg($0) }
}

func parsePositiveIntArg(_ flag: String) throws -> Int? {
    try parsePositiveIntArg(args, flag)
}

func parsePositiveIntArg(_ arguments: [String], _ flag: String) throws -> Int? {
    let raw = parseArg(arguments, flag)
    guard !raw.isEmpty else { return nil }
    guard let parsed = Int(raw), parsed > 0 else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "\(flag) must be a positive integer"
            ]
        )
    }
    return parsed
}

/// A built-in numeric run flag: absent → nil (caller supplies its default);
/// present but below `minimum` (or not an integer) → loud exit. Never silently
/// falls back to a default on a malformed value (which also hides typos parked
/// after a value-taking flag).
private func requireIntFlag(_ flag: String, minimum: Int) -> Int? {
    let raw = parseArg(flag)
    guard !raw.isEmpty else { return nil }
    guard let v = Int(raw), v >= minimum else {
        let kind = minimum > 0 ? "positive" : "non-negative"
        fputs("smelt run: \(flag) must be a \(kind) integer, got '\(raw)'\n", stderr)
        exit(1)
    }
    return v
}

/// Positive (≥ 1) built-in numeric flag; absent → nil.
func requirePositiveIntFlag(_ flag: String) -> Int? { requireIntFlag(flag, minimum: 1) }

/// Non-negative (≥ 0) built-in numeric flag (e.g. `--linger 0` = off); absent → nil.
func requireNonNegativeIntFlag(_ flag: String) -> Int? { requireIntFlag(flag, minimum: 0) }

func parseNonNegativeDoubleArg(_ flag: String) throws -> Double? {
    try parseNonNegativeDoubleArg(args, flag)
}

func parseNonNegativeDoubleArg(_ arguments: [String], _ flag: String) throws -> Double? {
    let raw = parseArg(arguments, flag)
    guard !raw.isEmpty else { return nil }
    guard let parsed = Double(raw), parsed.isFinite, parsed >= 0 else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "\(flag) must be a non-negative finite number"
            ]
        )
    }
    return parsed
}

extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

func positionalArgs(
    startingAt startIndex: Int,
    skippingFlagsWithValues flagsWithValues: Set<String>
) -> [String] {
    var result: [String] = []
    var idx = startIndex
    while idx < args.count {
        let arg = args[idx]
        if flagsWithValues.contains(arg), idx + 1 < args.count {
            idx += 2
            continue
        }
        result.append(arg)
        idx += 1
    }
    return result
}

func readPromptFromStdin() -> String? {
    guard isatty(FileHandle.standardInput.fileDescriptor) == 0 else {
        return nil
    }
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard !data.isEmpty,
          let text = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
          !text.isEmpty
    else {
        return nil
    }
    return text
}

func isDirectory(_ path: String) -> Bool {
    var isDir: ObjCBool = false
    return FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
        && isDir.boolValue
}

func isPackagePath(_ path: String) -> Bool {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue
    else {
        return false
    }
    return FileManager.default.fileExists(atPath: "\(path)/manifest.json")
}

func inferPackagePathFromCWD() -> String? {
    let fm = FileManager.default
    let cwd = fm.currentDirectoryPath
    if isPackagePath(cwd) {
        return cwd
    }
    guard let entries = try? fm.contentsOfDirectory(atPath: cwd) else {
        return nil
    }
    let packages = entries
        .filter { $0.hasSuffix(".smeltpkg") && isPackagePath("\(cwd)/\($0)") }
        .sorted()
    guard packages.count == 1 else {
        return nil
    }
    return "\(cwd)/\(packages[0])"
}

func readTextFileArg(_ path: String, label: String) throws -> String {
    do {
        return try String(contentsOfFile: path, encoding: .utf8)
    } catch {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Could not read \(label) file \(path): \(error)"
            ]
        )
    }
}

func parseCSVInts(_ value: String) -> [Int] {
    value
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 > 0 }
}

func parseCSVNonNegativeInts(_ value: String) -> [Int] {
    value
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 >= 0 }
}

func parseCSVIntDoubleMap(_ value: String) -> [Int: Double] {
    var result: [Int: Double] = [:]
    for entry in value.split(separator: ",") {
        let parts = entry.split(separator: "=", maxSplits: 1)
        guard parts.count == 2,
              let key = Int(parts[0].trimmingCharacters(in: .whitespaces)),
              key > 0,
              let threshold = Double(parts[1].trimmingCharacters(in: .whitespaces))
        else {
            continue
        }
        result[key] = threshold
    }
    return result
}
