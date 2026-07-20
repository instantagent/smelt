import Foundation
import Testing

/// Containment lint for the fixtures-as-code authoring layer (Phase C).
///
/// After the `.cam` grammar deletion, the checked-in `Models/<id>.module.json`
/// files are the SOLE production authority. `SmeltModels` (definitions as Swift
/// values) and `SmeltModuleAuthoring` (the sugar over the IR structs) exist only
/// to emit those artifacts and to feed tests. This lint keeps them quarantined
/// so fixtures-as-code can never leak into a runtime input, BOTH directions:
///
///   (a) source imports — every file under `Sources/SmeltModels/` and
///       `Sources/SmeltModuleAuthoring/` may import ONLY `SmeltSchema` /
///       `SmeltModuleAuthoring` (never the compiler, runtime, Metal, …);
///   (b) package edges — NO production target (`SmeltCompiler`, `SmeltRuntime`,
///       `SmeltCLI`, `SmeltProbe`, `SmeltSchema`) may depend on `SmeltModels`
///       or `SmeltModuleAuthoring`. Only test targets and the `smelt-models`
///       executable (`SmeltModelsCLI`) may.
///
/// Both scans are dependency-free (scanner style of `MatvecNoBypassLintTests`),
/// and each carries a negative fixture proving non-vacuity.
@Suite struct ModuleAuthoringContainmentLintTests {
    private static let authoringModules: Set<String> = ["SmeltModels", "SmeltModuleAuthoring"]
    private static let allowedSourceImports: Set<String> = ["SmeltSchema", "SmeltModuleAuthoring"]

    // MARK: - (a) source-import containment

    @Test func authoringSourcesImportOnlySchemaAndAuthoring() throws {
        for relDir in ["Sources/SmeltModels", "Sources/SmeltModuleAuthoring"] {
            let dir = try repoRelativePath(relDir)
            let swiftFiles = try FileManager.default
                .contentsOfDirectory(atPath: dir)
                .filter { $0.hasSuffix(".swift") }
            #expect(!swiftFiles.isEmpty, Comment(rawValue: "\(relDir) has no Swift sources to scan"))
            for file in swiftFiles {
                let source = try String(
                    contentsOfFile: (dir as NSString).appendingPathComponent(file), encoding: .utf8)
                let offenders = Self.importedModules(in: source)
                    .filter { !Self.allowedSourceImports.contains($0) }
                #expect(
                    offenders.isEmpty,
                    Comment(rawValue: "\(relDir)/\(file) imports forbidden module(s): \(offenders.sorted())"))
            }
        }
    }

    @Test func importScannerFlagsForbiddenImport() {
        // Negative fixture: a SmeltModels-style file reaching into the runtime.
        let rogue = """
        @_exported import SmeltSchema
        import SmeltModuleAuthoring
        import SmeltRuntime   // forbidden — pulls the runtime into fixtures-as-code
        import class Foundation.NSObject
        """
        let offenders = Self.importedModules(in: rogue)
            .filter { !Self.allowedSourceImports.contains($0) }
        #expect(offenders.contains("SmeltRuntime"))
        #expect(offenders.contains("Foundation"))
        // The legitimate imports are not flagged.
        #expect(!offenders.contains("SmeltSchema"))
        #expect(!offenders.contains("SmeltModuleAuthoring"))
    }

    // MARK: - (b) package-edge containment

    @Test func noProductionTargetDependsOnAuthoringLayer() throws {
        let manifest = try String(contentsOfFile: try repoRelativePath("Package.swift"), encoding: .utf8)
        let edges = Self.targetEdges(inManifest: manifest)
        let byName = Dictionary(uniqueKeysWithValues: edges.map { ($0.name, $0) })

        // Positive edges: the authoring cluster is a thin chain onto SmeltSchema.
        #expect(byName["SmeltModuleAuthoring"]?.dependencies == Set(["SmeltSchema"]))
        #expect(byName["SmeltModels"]?.dependencies == Set(["SmeltModuleAuthoring"]))

        // Every consumer of the authoring modules must be a test target or the
        // smelt-models executable (or the authoring chain itself).
        for edge in edges where !Self.mayConsumeAuthoring(edge) {
            #expect(
                edge.dependencies.isDisjoint(with: Self.authoringModules),
                Comment(rawValue: "production target \(edge.name) depends on the authoring layer: "
                    + "\(edge.dependencies.intersection(Self.authoringModules).sorted())"))
        }

        // Sanity: the scan actually saw the production targets it is meant to guard.
        for name in ["SmeltCompiler", "SmeltRuntime", "SmeltCLI", "SmeltProbe", "SmeltSchema"] {
            #expect(byName[name] != nil, Comment(rawValue: "manifest scan missed production target \(name)"))
        }
    }

    @Test func packageEdgeScannerFlagsForbiddenEdge() {
        // Negative fixture: a production target wired to fixtures-as-code.
        let rogue = """
        .target(
            name: "SmeltCompiler",
            dependencies: ["SmeltSchema", "SmeltRuntime", "SmeltModels"],
            path: "Sources/SmeltCompiler"
        ),
        .testTarget(
            name: "SmeltCompilerTests",
            dependencies: ["SmeltCompiler", "SmeltModels", "SmeltModuleAuthoring"]
        ),
        """
        let edges = Self.targetEdges(inManifest: rogue)
        let byName = Dictionary(uniqueKeysWithValues: edges.map { ($0.name, $0) })

        let compiler = byName["SmeltCompiler"]
        #expect(compiler != nil)
        #expect(compiler.map { !Self.mayConsumeAuthoring($0) } == true)
        #expect(compiler?.dependencies.isDisjoint(with: Self.authoringModules) == false)

        // The test target legitimately consumes the authoring layer.
        #expect(byName["SmeltCompilerTests"].map(Self.mayConsumeAuthoring) == true)
    }

    // MARK: - scanners

    private static func mayConsumeAuthoring(_ edge: TargetEdge) -> Bool {
        edge.isTest || edge.name == "SmeltModelsCLI" || authoringModules.contains(edge.name)
    }

    /// Extract the module name from every `import` in `source`, tolerating
    /// leading attributes (`@_exported`, `@testable`) and submodule imports
    /// (`import class Foundation.NSObject` → `Foundation`).
    static func importedModules(in source: String) -> [String] {
        var modules: [String] = []
        for rawLine in source.split(whereSeparator: \.isNewline) {
            var line = rawLine.trimmingCharacters(in: .whitespaces)
            if let comment = line.range(of: "//") {
                line = String(line[..<comment.lowerBound]).trimmingCharacters(in: .whitespaces)
            }
            while line.hasPrefix("@") {
                guard let space = line.firstIndex(of: " ") else { break }
                line = String(line[line.index(after: space)...]).trimmingCharacters(in: .whitespaces)
            }
            guard line.hasPrefix("import ") else { continue }
            let rest = line.dropFirst("import ".count).trimmingCharacters(in: .whitespaces)
            guard let lastToken = rest.split(separator: " ").last else { continue }
            let module = lastToken.split(separator: ".").first.map(String.init) ?? String(lastToken)
            modules.append(module)
        }
        return modules
    }

    struct TargetEdge {
        let name: String
        let dependencies: Set<String>
        let isTest: Bool
    }

    /// Line-oriented scan of a `Package.swift`-style manifest: attribute each
    /// `.target(` / `.executableTarget(` / `.testTarget(` block to its `name:`
    /// and its `dependencies:` string entries. Comments are stripped; the
    /// products section (`.library(`/`.executable(`) and `.binaryTarget(` are
    /// not target openers, so they are ignored.
    static func targetEdges(inManifest source: String) -> [TargetEdge] {
        var edges: [TargetEdge] = []
        var name: String?
        var deps: Set<String> = []
        var isTest = false
        var inside = false

        func flush() {
            if inside, let name {
                edges.append(TargetEdge(name: name, dependencies: deps, isTest: isTest))
            }
            name = nil
            deps = []
            isTest = false
            inside = false
        }

        for rawLine in source.split(whereSeparator: \.isNewline) {
            var line = String(rawLine)
            if let comment = line.range(of: "//") {
                line = String(line[..<comment.lowerBound])
            }
            let isOpener = line.contains(".target(")
                || line.contains(".executableTarget(")
                || line.contains(".testTarget(")
            if isOpener {
                flush()
                inside = true
                isTest = line.contains(".testTarget(")
            }
            guard inside else { continue }
            if name == nil, let value = firstStringLiteral(after: "name:", in: line) {
                name = value
            }
            if line.contains("dependencies:") {
                deps.formUnion(stringLiterals(in: line))
            }
        }
        flush()
        return edges
    }

    private static func firstStringLiteral(after marker: String, in line: String) -> String? {
        guard let markerRange = line.range(of: marker) else { return nil }
        return stringLiterals(in: String(line[markerRange.upperBound...])).first
    }

    private static func stringLiterals(in text: String) -> [String] {
        var out: [String] = []
        var current: String?
        for ch in text {
            if ch == "\"" {
                if let value = current {
                    out.append(value)
                    current = nil
                } else {
                    current = ""
                }
            } else if current != nil {
                current?.append(ch)
            }
        }
        return out
    }
}
