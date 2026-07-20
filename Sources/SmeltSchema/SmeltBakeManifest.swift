// SmeltBakeManifest — the uniform honesty record (`baked.json`) for generic
// prepared state sealed into a `.smeltpkg`.
//
// Before this marker, "what did bake seal?" was answered three different ways:
// some paths inferred sidecar file-presence with silent fallback, while others
// read route-specific manifest fields. `baked.json` makes the answer uniform and
// explicit: a package *declares* its sealed components, and the runtime can
// *fail loud* when a declared artifact is missing instead of silently behaving
// as if it were never baked.
//
// `baked.json` is the authority for the *declaration* only. It does NOT become a
// second runtime selection signal: the graph/configured route keeps ownership of
// the active artifact path. A consistency gate asserts the two agree by exact path
// — never a boolean proxy — so this stays an honest stamp of the runtime path,
// not a divergent mirror.

import Foundation

/// Canonical package-relative filenames for bake artifacts, owned here so every
/// layer (schema, runtime, CLI, tests) reads them from one place. The runtime's
/// `SmeltBakedPrefix` / `SmeltBakedGrammar` reference these constants rather than
/// repeating the literals, so a rename can't desync the marker's backfill.
public enum SmeltBakeArtifacts {
    public static let prefixMeta = "baked_prefix.json"
    public static let prefixSnapshot = "baked_prefix.snapshot"
    public static let preparedPromptsMeta = "prepared_prompts.json"
    public static let grammarMeta = "baked_grammar.json"
    public static let grammarTrie = "baked_grammar.trie"
    // Voice and interface filenames already live on their schema types.
}

public struct SmeltBakeManifest: Codable, Sendable, Equatable {

    /// A sealed "job" component. The set is route-spanning: a package lists only
    /// the components it actually baked.
    public enum Component: String, Codable, Sendable, CaseIterable {
        case prefix          // prompt prefill snapshot
        case preparedPrompts = "prepared-prompts" // named prompt-state set
        case grammar         // JSON-schema output constraint
        case args            // declared CLI interface
        case voice           // voice defaults
        // NOTE: startup warmups are NOT a sealed component: they live in the
        // package manifest's always-present startup policy, so there is no clean
        // marker<->manifest invariant. The marker tracks sealed *artifacts*.
    }

    /// One declared component and the package-relative files it carries.
    public struct Sealed: Codable, Sendable, Equatable {
        public let kind: Component
        /// Files that MUST be present + loadable; a declared-but-missing one is a
        /// fail-loud condition for the runtime.
        public let required: [String]
        /// Optional accelerators whose absence is a graceful degrade (e.g. the
        /// grammar trie rebuilds from the schema). Recorded for honesty.
        public let perf: [String]

        public init(kind: Component, required: [String], perf: [String] = []) {
            self.kind = kind
            self.required = required
            self.perf = perf
        }
    }

    public let version: Int
    public let sealed: [Sealed]

    public init(version: Int = 1, sealed: [Sealed]) {
        self.version = version
        self.sealed = sealed
    }

    public static let fileName = "baked.json"

    // MARK: - Component constructors (the canonical required/perf shape per kind)

    public static func prefix() -> Sealed {
        Sealed(
            kind: .prefix,
            required: [SmeltBakeArtifacts.prefixMeta, SmeltBakeArtifacts.prefixSnapshot]
        )
    }

    public static func preparedPrompts(requiredFiles: [String]) -> Sealed {
        Sealed(kind: .preparedPrompts, required: requiredFiles.sorted())
    }

    /// Grammar: the schema JSON is required; the serialized trie is a perf
    /// accelerator listed only when bake wrote it.
    public static func grammar(hasTrie: Bool) -> Sealed {
        Sealed(
            kind: .grammar,
            required: [SmeltBakeArtifacts.grammarMeta],
            perf: hasTrie ? [SmeltBakeArtifacts.grammarTrie] : []
        )
    }

    public static func args() -> Sealed {
        Sealed(kind: .args, required: [SmeltPackageInterface.fileName])
    }

    public static func voice() -> Sealed {
        Sealed(kind: .voice, required: [Qwen3TTSVoice.fileName])
    }

    // MARK: - Load / write

    /// The bake marker of the package at `packagePath`, or nil when none exists.
    /// Throws on a present-but-malformed marker — a broken declaration must fail
    /// loudly, not be ignored.
    public static func load(packagePath: String) throws -> SmeltBakeManifest? {
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(
            SmeltBakeManifest.self, from: Data(contentsOf: url)
        )
    }

    public func write(packagePath: String) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(Self.fileName)
        try enc.encode(self).write(to: url, options: .atomic)
    }

    // MARK: - Recording

    /// A copy with `component` added or replacing any existing entry of the same
    /// kind (re-baking a component updates its declaration). Deterministic order.
    public func recording(_ component: Sealed) -> SmeltBakeManifest {
        var kept = sealed.filter { $0.kind != component.kind }
        kept.append(component)
        kept.sort { $0.kind.rawValue < $1.kind.rawValue }
        return SmeltBakeManifest(version: version, sealed: kept)
    }

    /// Record `components` into the package's `baked.json`, creating it if absent.
    ///
    /// On first creation the marker is *backfilled* from artifacts already present
    /// (a legacy package baked before this marker existed), so the very first
    /// write is a COMPLETE declaration — never one that omits real sidecars and so
    /// reads as dishonest under closed-world enforcement.
    public static func record(
        _ components: [Sealed], packagePath: String
    ) throws {
        guard !components.isEmpty else { return }
        var manifest = try load(packagePath: packagePath)
            ?? SmeltBakeManifest(
                version: 1, sealed: inferExisting(packagePath: packagePath)
            )
        for component in components {
            manifest = manifest.recording(component)
        }
        try manifest.write(packagePath: packagePath)
    }

    // MARK: - Backfill

    /// Infer the sealed components of a package from artifacts already on disk —
    /// used to migrate a legacy package to a complete marker on its first bake.
    /// File-presence only (used at bake time, where a present sidecar is a real
    /// seal); strict loadability is the runtime's enforcement job.
    public static func inferExisting(packagePath: String) -> [Sealed] {
        let fm = FileManager.default
        func present(_ name: String) -> Bool {
            fm.fileExists(atPath: "\(packagePath)/\(name)")
        }

        var out: [Sealed] = []
        if present(SmeltBakeArtifacts.prefixMeta),
           present(SmeltBakeArtifacts.prefixSnapshot) {
            out.append(prefix())
        }
        if present(SmeltBakeArtifacts.preparedPromptsMeta) {
            struct PreparedMeta: Decodable {
                struct Entry: Decodable {
                    let snapshotFile: String
                    enum CodingKeys: String, CodingKey {
                        case snapshotFile = "snapshot_file"
                    }
                }
                let entries: [Entry]
            }
            let url = URL(fileURLWithPath: packagePath)
                .appendingPathComponent(SmeltBakeArtifacts.preparedPromptsMeta)
            let files = (try? JSONDecoder().decode(
                PreparedMeta.self, from: Data(contentsOf: url)
            ).entries.map(\.snapshotFile)) ?? []
            out.append(preparedPrompts(requiredFiles:
                [SmeltBakeArtifacts.preparedPromptsMeta] + files
            ))
        }
        if present(SmeltBakeArtifacts.grammarMeta) {
            out.append(grammar(hasTrie: present(SmeltBakeArtifacts.grammarTrie)))
        }
        if present(SmeltPackageInterface.fileName) {
            out.append(args())
        }
        if present(Qwen3TTSVoice.fileName) {
            out.append(voice())
        }
        return out.sorted { $0.kind.rawValue < $1.kind.rawValue }
    }

    private static func packageInterfaceValidationContext(
        packagePath: String
    ) throws -> SmeltPackageInterface.PackageValidationContext? {
        let url = URL(fileURLWithPath: "\(packagePath)/manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let manifestData = try Data(contentsOf: url)
        do {
            return try SmeltPackageInterface.packageValidationContext(manifestData: manifestData)
        } catch {
            let graphPolicy = (try? SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData))?
                .rawValue ?? "missing-or-unsupported-graph"
            throw SmeltBakeEnforcementError.argsPolicyUnsupported(
                runtimeGraphPolicy: graphPolicy
            )
        }
    }

    // MARK: - Enforcement (honesty: declared ⇒ present; present ⇒ declared)

    public func declares(_ kind: Component) -> Bool {
        sealed.contains { $0.kind == kind }
    }

    /// Recognized fixed-name bake sidecars and the component each belongs to.
    /// Used for the closed-world check.
    public static let recognizedSidecars: [(file: String, kind: Component)] = [
        (SmeltBakeArtifacts.prefixMeta, .prefix),
        (SmeltBakeArtifacts.prefixSnapshot, .prefix),
        (SmeltBakeArtifacts.preparedPromptsMeta, .preparedPrompts),
        (SmeltBakeArtifacts.grammarMeta, .grammar),
        (SmeltBakeArtifacts.grammarTrie, .grammar),  // perf accel; stray ⇒ undeclared
        (SmeltPackageInterface.fileName, .args),
        (Qwen3TTSVoice.fileName, .voice),
    ]

    /// Components the operator has opted out of via `SMELT_NO_BAKED_*` — "ignore
    /// the declaration, fall back." Must agree with the lenient loaders that read
    /// the same env vars (`SmeltBakedPrefix`/`SmeltBakedGrammar`).
    public static func ignoredFromEnv(
        _ env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Set<Component> {
        var ignored: Set<Component> = []
        if env["SMELT_NO_BAKED_PREFIX"] == "1" { ignored.insert(.prefix) }
        if env["SMELT_NO_BAKED_GRAMMAR"] == "1" { ignored.insert(.grammar) }
        if env["SMELT_NO_BAKED_VOICE"] == "1" { ignored.insert(.voice) }
        return ignored
    }

    /// Fail loud when the marker and the package disagree: a declared component
    /// whose required artifact is absent, or a recognized sidecar present but
    /// undeclared (a partial/failed bake). Loadability (corrupt files) is the
    /// caller's job via the component's strict loader. Opted-out components are
    /// skipped on both sides.
    public func validatePresence(
        packagePath: String, ignoring: Set<Component>
    ) throws {
        let fm = FileManager.default
        func present(_ name: String) -> Bool {
            fm.fileExists(atPath: "\(packagePath)/\(name)")
        }
        for component in sealed where !ignoring.contains(component.kind) {
            for file in component.required where !present(file) {
                throw SmeltBakeEnforcementError.declaredArtifactMissing(
                    kind: component.kind, file: file)
            }
        }
        // File-level closed-world: every present recognized sidecar must be in
        // some declared component's required/perf list. Kind-level alone would
        // miss a marker that declares `.grammar` with `perf: []` while the trie
        // is on disk — a present-but-undeclared perf artifact.
        let declaredFiles = Set(sealed.flatMap { $0.required + $0.perf })
        for (file, kind) in Self.recognizedSidecars
        where !ignoring.contains(kind) && !declaredFiles.contains(file) && present(file) {
            throw SmeltBakeEnforcementError.undeclaredSidecarPresent(file: file, kind: kind)
        }
    }

    /// Load the marker (if any) and enforce honesty: presence + closed-world, plus
    /// strict validation of declared package-policy sidecars whose schema loaders
    /// live in this layer (`args.json`, `voice.json`). `args.json` is validated
    /// against manifest-derived runtime policy when a manifest is present.
    /// Prefix/grammar strict loads live at their Runtime use sites. No marker ⇒ legacy.
    @discardableResult
    public static func enforce(
        packagePath: String,
        ignoring: Set<Component>,
        argsValidationContext: SmeltPackageInterface.InterfaceValidationContext? = nil
    ) throws -> SmeltBakeManifest? {
        guard let manifest = try load(packagePath: packagePath) else { return nil }
        try manifest.validatePresence(packagePath: packagePath, ignoring: ignoring)
        if manifest.declares(.args), !ignoring.contains(.args) {
            if let interface = try SmeltPackageInterface.load(packagePath: packagePath) {
                if let argsValidationContext {
                    try interface.validate(interfaceContext: argsValidationContext)
                } else if let context = try packageInterfaceValidationContext(packagePath: packagePath) {
                    try interface.validate(packageContext: context)
                }
            }
        }
        if manifest.declares(.voice), !ignoring.contains(.voice) {
            _ = try Qwen3TTSVoice.load(packagePath: packagePath, env: [:])
        }
        return manifest
    }
}

public enum SmeltBakeEnforcementError: Error, CustomStringConvertible, Equatable {
    case declaredArtifactMissing(kind: SmeltBakeManifest.Component, file: String)
    case undeclaredSidecarPresent(file: String, kind: SmeltBakeManifest.Component)
    case argsPolicyUnsupported(runtimeGraphPolicy: String)

    public var description: String {
        switch self {
        case .declaredArtifactMissing(let kind, let file):
            return "baked.json declares \(kind.rawValue) but its artifact "
                + "'\(file)' is missing — the package is dishonest. "
                + "Rebuild/re-bake, or set the matching SMELT_NO_BAKED_* to ignore it."
        case .undeclaredSidecarPresent(let file, let kind):
            return "bake sidecar '\(file)' (\(kind.rawValue)) is present but not "
                + "declared in baked.json — likely a partial/failed bake. "
                + "Re-bake to record it, or remove the stray file."
        case .argsPolicyUnsupported(let runtimeGraphPolicy):
            return "baked.json declares args but runtime graph '\(runtimeGraphPolicy)'"
                + " has no supported declared-args policy in this smelt build. Re-bake "
                + "with a supported runtime descriptor."
        }
    }
}
