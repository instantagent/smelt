import CryptoKit
import Foundation

public struct SmeltTraceWitness: Codable, Sendable {
    public let schemaVersion: Int
    public let format: String
    public let capture: SmeltTraceCapture
    public let contractSHA256: String
    public let contract: SmeltTraceContract
}

public struct SmeltTraceCapture: Codable, Sendable, Equatable {
    public let mode: String
    public let eventsCaptured: Bool
}

public struct SmeltTraceContract: Codable, Sendable, Equatable {
    public let package: SmeltTracePackageContract
    public let graph: SmeltTraceGraph?
    public let loop: SmeltTraceLoop?
    public let blocks: [SmeltTraceBlockRoute]
    public let dispatchTables: [SmeltTraceDispatchTableContract]
    public let sidecars: [SmeltTraceSidecarContract]
    public let artifacts: [SmeltTraceArtifactContract]
    public let dtypeSummary: [SmeltTraceDTypeCount]
    public let events: [SmeltTraceEvent]
    public let issues: [SmeltTraceIssue]
}

public struct SmeltTracePackageContract: Codable, Sendable, Equatable {
    public let packageKind: String
    public let modelName: String?
    public let manifestKind: String?
    public let manifestSHA256: String
    public let buildFingerprint: String?
}

public struct SmeltTraceDispatchTableContract: Codable, Sendable, Equatable {
    public let name: String
    public let exists: Bool
    public let sha256: String?
    public let parseError: String?
    public let totalRecords: Int?
    public let dispatchCount: Int?
    public let swapCount: Int?
    public let topPipelines: [SmeltTracePipelineUsage]
}

public struct SmeltTraceSidecarContract: Codable, Sendable, Equatable {
    public let name: String
    public let exists: Bool
    public let packageKind: String?
    public let modelName: String?
    public let headlessTrunkABI: Bool?
    public let manifestSHA256: String?
    public let dispatchTables: [SmeltTraceDispatchTableContract]
    public let issues: [SmeltTraceIssue]
}

public struct SmeltTraceArtifactContract: Codable, Sendable, Equatable {
    public let name: String
    public let kind: String
    public let exists: Bool
    public let bytes: UInt64?
    public let declaredSHA256: String?
    public let actualSHA256: String?
}

public struct SmeltTraceEvent: Codable, Sendable, Equatable {
    public let kind: String
    public let index: Int
    public let phase: String?
    public let block: String?
    public let route: String?
    public let step: Int?
    public let witness: String?

    public init(
        kind: String,
        index: Int,
        phase: String? = nil,
        block: String? = nil,
        route: String? = nil,
        step: Int? = nil,
        witness: String? = nil
    ) {
        self.kind = kind
        self.index = index
        self.phase = phase
        self.block = block
        self.route = route
        self.step = step
        self.witness = witness
    }
}

public enum SmeltTraceCAMRoute {
    public static let eventKind = "cam-route"

    public static func events(
        witness: String,
        followedBy events: [SmeltTraceEvent]
    ) -> [SmeltTraceEvent] {
        let route = SmeltTraceEvent(
            kind: eventKind,
            index: 0,
            witness: witness
        )
        return [route] + events.enumerated().map { offset, event in
            SmeltTraceEvent(
                kind: event.kind,
                index: offset + 1,
                phase: event.phase,
                block: event.block,
                route: event.route,
                step: event.step,
                witness: event.witness
            )
        }
    }
}

/// Mutable event collector for real loop execution. It is intentionally tiny:
/// the scheduled loop is serial, and the witness layer canonicalizes these
/// events later when they are embedded into `.smttrace`.
public final class SmeltRuntimeTraceRecorder {
    public private(set) var events: [SmeltTraceEvent] = []

    public init() {}

    public func record(
        kind: String,
        phase: String? = nil,
        block: String? = nil,
        route: String? = nil,
        step: Int? = nil,
        witness: String? = nil
    ) {
        events.append(SmeltTraceEvent(
            kind: kind,
            index: events.count,
            phase: phase,
            block: block,
            route: route,
            step: step,
            witness: witness
        ))
    }

    public func reset() {
        events.removeAll(keepingCapacity: true)
    }
}

public struct SmeltTraceComparison: Codable, Sendable {
    public let matches: Bool
    public let differences: [SmeltTraceDifference]
}

public struct SmeltTraceDifference: Codable, Sendable, Equatable {
    public let path: String
    public let expected: String?
    public let actual: String?
}

public struct SmeltTraceSuiteSpec: Codable, Sendable, Equatable {
    public let schemaVersion: Int?
    public let package: String?
    public let cases: [SmeltTraceSuiteCaseSpec]

    public init(
        schemaVersion: Int? = 1,
        package: String? = nil,
        cases: [SmeltTraceSuiteCaseSpec]
    ) {
        self.schemaVersion = schemaVersion
        self.package = package
        self.cases = cases
    }
}

public struct SmeltTraceSuiteCaseSpec: Codable, Sendable, Equatable {
    public let name: String
    public let package: String?
    public let golden: String
    public let events: String?

    public init(
        name: String,
        package: String? = nil,
        golden: String,
        events: String? = nil
    ) {
        self.name = name
        self.package = package
        self.golden = golden
        self.events = events
    }
}

public struct SmeltTraceSuiteOptions: Sendable {
    public let inspectOptions: SmeltTraceOptions
    public let packageOverride: String?
    public let updateGoldens: Bool

    public init(
        inspectOptions: SmeltTraceOptions = SmeltTraceOptions(),
        packageOverride: String? = nil,
        updateGoldens: Bool = false
    ) {
        self.inspectOptions = inspectOptions
        self.packageOverride = packageOverride
        self.updateGoldens = updateGoldens
    }
}

public struct SmeltTraceSuiteResult: Codable, Sendable {
    public let suite: String
    public let matches: Bool
    public let cases: [SmeltTraceSuiteCaseResult]
}

public struct SmeltTraceSuiteCaseResult: Codable, Sendable {
    public let name: String
    public let package: String
    public let golden: String
    public let matches: Bool
    public let updated: Bool
    public let differences: [SmeltTraceDifference]
    public let error: String?
}

public struct SmeltTraceRecordOptions: Sendable {
    public let inspectOptions: SmeltTraceOptions
    public let runtimeEvents: [SmeltTraceEvent]

    public init(
        inspectOptions: SmeltTraceOptions = SmeltTraceOptions(),
        runtimeEvents: [SmeltTraceEvent] = []
    ) {
        self.inspectOptions = inspectOptions
        self.runtimeEvents = runtimeEvents
    }
}

public struct SmeltTraceReplayOptions: Sendable {
    public let inspectOptions: SmeltTraceOptions
    public let runtimeEvents: [SmeltTraceEvent]?

    public init(
        inspectOptions: SmeltTraceOptions = SmeltTraceOptions(),
        runtimeEvents: [SmeltTraceEvent]? = nil
    ) {
        self.inspectOptions = inspectOptions
        self.runtimeEvents = runtimeEvents
    }
}

public extension SmeltTrace {
    static func record(
        packagePath: String,
        options: SmeltTraceRecordOptions = SmeltTraceRecordOptions()
    ) throws -> SmeltTraceWitness {
        let report = try inspect(packagePath: packagePath, options: options.inspectOptions)
        return try witness(from: report, runtimeEvents: options.runtimeEvents)
    }

    static func loadWitness(from path: String) throws -> SmeltTraceWitness {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(SmeltTraceWitness.self, from: data)
    }

    static func writeWitness(_ witness: SmeltTraceWitness, to path: String) throws {
        let data = try encodeCanonical(witness)
        try data.write(to: URL(fileURLWithPath: path))
    }

    static func compare(
        expected: SmeltTraceWitness,
        actual: SmeltTraceWitness
    ) -> SmeltTraceComparison {
        var differences = witnessValidityDifferences(expected, label: "expected")
            + witnessValidityDifferences(actual, label: "actual")
        diff("schemaVersion", expected.schemaVersion, actual.schemaVersion, into: &differences)
        diff("format", expected.format, actual.format, into: &differences)
        diff("capture", expected.capture, actual.capture, into: &differences)
        diff("contractSHA256", expected.contractSHA256, actual.contractSHA256, into: &differences)

        diff("package", expected.contract.package, actual.contract.package, into: &differences)
        diff("graph", expected.contract.graph, actual.contract.graph, into: &differences)
        diff("loop", expected.contract.loop, actual.contract.loop, into: &differences)
        diff("blocks", expected.contract.blocks, actual.contract.blocks, into: &differences)
        diff("dispatchTables", expected.contract.dispatchTables, actual.contract.dispatchTables, into: &differences)
        diff("sidecars", expected.contract.sidecars, actual.contract.sidecars, into: &differences)
        diff("artifacts", expected.contract.artifacts, actual.contract.artifacts, into: &differences)
        diff("dtypeSummary", expected.contract.dtypeSummary, actual.contract.dtypeSummary, into: &differences)
        diff("events", expected.contract.events, actual.contract.events, into: &differences)
        diff("issues", expected.contract.issues, actual.contract.issues, into: &differences)

        return SmeltTraceComparison(matches: differences.isEmpty, differences: differences)
    }

    static func validateSuite(_ suite: SmeltTraceSuiteSpec) throws {
        if let schemaVersion = suite.schemaVersion, schemaVersion != 1 {
            throw suiteError("unsupported trace suite schemaVersion \(schemaVersion); expected 1")
        }
        if let package = suite.package, package.trimmedSuiteValue.isEmpty {
            throw suiteError("trace suite has blank package")
        }
        guard !suite.cases.isEmpty else {
            throw suiteError("trace suite must contain at least one case")
        }

        var names = Set<String>()
        for (index, testCase) in suite.cases.enumerated() {
            let name = testCase.name.trimmedSuiteValue
            guard !name.isEmpty else {
                throw suiteError("case \(index) has blank name")
            }
            guard names.insert(name).inserted else {
                throw suiteError("duplicate trace suite case name '\(name)'")
            }
            guard !testCase.golden.trimmedSuiteValue.isEmpty else {
                throw suiteError("case '\(name)' has blank golden")
            }
            if let package = testCase.package, package.trimmedSuiteValue.isEmpty {
                throw suiteError("case '\(name)' has blank package")
            }
            if let events = testCase.events, events.trimmedSuiteValue.isEmpty {
                throw suiteError("case '\(name)' has blank events")
            }
        }
    }

    static func verify(
        packagePath: String,
        against witnessPath: String,
        options: SmeltTraceRecordOptions = SmeltTraceRecordOptions()
    ) throws -> SmeltTraceComparison {
        let expected = try loadWitness(from: witnessPath)
        let actual = try record(packagePath: packagePath, options: options)
        return compare(expected: expected, actual: actual)
    }

    static func replay(
        packagePath: String,
        from witnessPath: String,
        options: SmeltTraceReplayOptions = SmeltTraceReplayOptions()
    ) throws -> SmeltTraceComparison {
        let expected = try loadWitness(from: witnessPath)
        let runtimeEvents = options.runtimeEvents ?? replayEvents(from: expected)
        let actual = try record(
            packagePath: packagePath,
            options: SmeltTraceRecordOptions(
                inspectOptions: options.inspectOptions,
                runtimeEvents: runtimeEvents
            )
        )
        return compare(expected: expected, actual: actual)
    }

    static func verifySuite(
        path: String,
        options: SmeltTraceSuiteOptions = SmeltTraceSuiteOptions()
    ) throws -> SmeltTraceSuiteResult {
        if options.updateGoldens, options.packageOverride != nil {
            throw suiteError("trace suite cannot update goldens while using a package override")
        }
        let suiteURL = URL(fileURLWithPath: path)
        let suiteData = try Data(contentsOf: suiteURL)
        let suite = try JSONDecoder().decode(SmeltTraceSuiteSpec.self, from: suiteData)
        try validateSuite(suite)
        let baseURL = suiteURL.deletingLastPathComponent()
        let results = suite.cases.map { testCase in
            verifySuiteCase(
                testCase,
                suite: suite,
                suiteBaseURL: baseURL,
                options: options
            )
        }
        return SmeltTraceSuiteResult(
            suite: path,
            matches: results.allSatisfy(\.matches),
            cases: results
        )
    }
}

private extension SmeltTrace {
    static func witness(
        from report: SmeltTraceReport,
        runtimeEvents: [SmeltTraceEvent]
    ) throws -> SmeltTraceWitness {
        let events = runtimeEvents.isEmpty
            ? traceEvents(loop: report.loop, blocks: report.blocks)
            : runtimeEvents
        let captureMode = runtimeEvents.isEmpty
            ? "package-contract+loop"
            : "package-contract+runtime"
        let contract = SmeltTraceContract(
            package: SmeltTracePackageContract(
                packageKind: report.packageKind,
                modelName: report.modelName,
                manifestKind: report.manifestKind,
                manifestSHA256: report.manifestSHA256,
                buildFingerprint: report.buildFingerprint
            ),
            graph: report.graph,
            loop: report.loop,
            blocks: report.blocks,
            dispatchTables: report.dispatchTables.map(dispatchContract),
            sidecars: report.sidecars.map(sidecarContract),
            artifacts: report.artifacts.map(artifactContract),
            dtypeSummary: report.dtypeSummary,
            events: events,
            issues: report.issues + runtimeCoverageIssues(
                report: report,
                events: runtimeEvents
            )
        )
        let digest = try contractDigest(contract)
        return SmeltTraceWitness(
            schemaVersion: 1,
            format: "smelt.trace.witness",
            capture: SmeltTraceCapture(mode: captureMode, eventsCaptured: !events.isEmpty),
            contractSHA256: digest,
            contract: contract
        )
    }

    static func traceEvents(
        loop: SmeltTraceLoop?,
        blocks: [SmeltTraceBlockRoute]
    ) -> [SmeltTraceEvent] {
        guard let loop else { return [] }
        let routesByBlock = Dictionary(blocks.map { ($0.name, $0.route) }, uniquingKeysWith: { first, _ in first })
        var events: [SmeltTraceEvent] = []

        func appendPhase(scope: String, phase: SmeltTraceLoopPhase) {
            events.append(SmeltTraceEvent(
                kind: "phase",
                index: events.count,
                phase: "\(scope):\(phase.name)",
                block: nil,
                route: nil,
                step: nil,
                witness: "feedsNextStep=\(phase.feedsNextStep)"
            ))
            for block in phase.blocks {
                events.append(SmeltTraceEvent(
                    kind: "block",
                    index: events.count,
                    phase: "\(scope):\(phase.name)",
                    block: block,
                    route: routesByBlock[block],
                    step: nil,
                    witness: "declared-route"
                ))
            }
        }

        for phase in loop.setup {
            appendPhase(scope: "setup", phase: phase)
        }
        for phase in loop.perStep {
            appendPhase(scope: "per-step", phase: phase)
        }
        events.append(SmeltTraceEvent(
            kind: "emission",
            index: events.count,
            phase: nil,
            block: nil,
            route: nil,
            step: nil,
            witness: loop.emission
        ))
        events.append(SmeltTraceEvent(
            kind: "stop",
            index: events.count,
            phase: nil,
            block: nil,
            route: nil,
            step: nil,
            witness: loop.stop.joined(separator: ",")
        ))
        return events
    }

    static func runtimeCoverageIssues(
        report: SmeltTraceReport,
        events runtimeEvents: [SmeltTraceEvent]
    ) -> [SmeltTraceIssue] {
        runtimeCoverageIssues(blocks: report.blocks, loop: report.loop, events: runtimeEvents)
    }

    static func runtimeCoverageIssues(
        blocks: [SmeltTraceBlockRoute],
        loop: SmeltTraceLoop?,
        events runtimeEvents: [SmeltTraceEvent]
    ) -> [SmeltTraceIssue] {
        guard !runtimeEvents.isEmpty else { return [] }
        var issues: [SmeltTraceIssue] = []
        issues += camRouteRuntimeIssues(events: runtimeEvents)
        let routesByBlock = Dictionary(blocks.map { ($0.name, $0.route) }, uniquingKeysWith: { first, _ in first })
        let knownBlocks = Set(routesByBlock.keys)
        let coveredBlocks = Set(runtimeEvents.compactMap(\.block))
        let phaseBlocks = loop.map(runtimePhaseBlocks) ?? [:]
        let declaredLoopPhases = Set(phaseBlocks.keys)
        let loopBlocks = Set(phaseBlocks.values.flatMap { $0 })
        var coveredBlocksByPhase: [String: Set<String>] = [:]

        for (ordinal, event) in runtimeEvents.enumerated() where event.index != ordinal {
            issues.append(SmeltTraceIssue(
                severity: .error,
                code: "runtimeEventIndexMismatch",
                message: "event \(ordinal) has index \(event.index)"
            ))
        }

        for event in runtimeEvents {
            guard let block = event.block else { continue }
            if !knownBlocks.contains(block) {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeUnknownBlock",
                    message: "event \(event.index) references unknown block '\(block)'"
                ))
                continue
            }
            if let route = event.route,
               let expected = routesByBlock[block],
               route != expected
            {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeRouteMismatch",
                    message: "event \(event.index) block '\(block)' route '\(route)' != '\(expected)'"
                ))
            }
            guard loopBlocks.contains(block) else { continue }
            guard let phase = event.phase, isLoopPhaseLabel(phase) else {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeLoopBlockMissingPhase",
                    message: "event \(event.index) loop block '\(block)' has no declared loop phase"
                ))
                continue
            }
            guard let allowedBlocks = phaseBlocks[phase] else {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeUnknownLoopPhase",
                    message: "event \(event.index) references undeclared loop phase '\(phase)'"
                ))
                continue
            }
            if !allowedBlocks.contains(block) {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeBlockPhaseMismatch",
                    message: "event \(event.index) block '\(block)' is not declared in loop phase '\(phase)'"
                ))
            } else {
                coveredBlocksByPhase[phase, default: []].insert(block)
            }
        }

        for event in runtimeEvents {
            guard let phase = event.phase, isLoopPhaseLabel(phase) else { continue }
            if let block = event.block, loopBlocks.contains(block) { continue }
            if !declaredLoopPhases.contains(phase) {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeUnknownLoopPhase",
                    message: "event \(event.index) references undeclared loop phase '\(phase)'"
                ))
            }
        }

        let missingBlocks = knownBlocks.subtracting(coveredBlocks).sorted()
        if !missingBlocks.isEmpty {
            issues.append(SmeltTraceIssue(
                severity: .error,
                code: "runtimeCoverageMissingBlocks",
                message: "runtime witness did not cover blocks: \(missingBlocks.joined(separator: ", "))"
            ))
        }

        if loop != nil {
            let missingLoopBlocks = loopBlocks.subtracting(coveredBlocks).sorted()
            if !missingLoopBlocks.isEmpty {
                issues.append(SmeltTraceIssue(
                    severity: .error,
                    code: "runtimeCoverageMissingLoopBlocks",
                    message: "runtime witness did not cover loop blocks: \(missingLoopBlocks.joined(separator: ", "))"
                ))
            }
            for phase in declaredLoopPhases.sorted() {
                let missing = (phaseBlocks[phase] ?? [])
                    .subtracting(coveredBlocksByPhase[phase] ?? [])
                    .sorted()
                if !missing.isEmpty {
                    issues.append(SmeltTraceIssue(
                        severity: .error,
                        code: "runtimeCoverageMissingLoopPhaseBlocks",
                        message: "runtime witness did not cover loop phase '\(phase)' blocks: \(missing.joined(separator: ", "))"
                    ))
                }
            }
        }

        return issues
    }

    static func camRouteRuntimeIssues(events: [SmeltTraceEvent]) -> [SmeltTraceIssue] {
        let routeEntries = events.enumerated().filter { _, event in
            event.kind == SmeltTraceCAMRoute.eventKind
        }
        guard !routeEntries.isEmpty else { return [] }

        var issues: [SmeltTraceIssue] = []
        if routeEntries.count > 1 {
            issues.append(camRouteIssue("runtime witness contains \(routeEntries.count) CAM route events"))
        }

        for (ordinal, event) in routeEntries {
            if ordinal != 0 {
                issues.append(camRouteIssue("CAM route event \(event.index) must be the first runtime event"))
            }
            if event.index != 0 {
                issues.append(camRouteIssue("CAM route event has index \(event.index); expected 0"))
            }
            issues += camRouteWitnessIssues(event.witness, eventIndex: event.index)
        }
        return issues
    }

    static func camRouteWitnessIssues(_ witness: String?, eventIndex: Int) -> [SmeltTraceIssue] {
        guard let witness, !witness.isEmpty else {
            return [camRouteIssue("CAM route event \(eventIndex) has no witness")]
        }

        var issues: [SmeltTraceIssue] = []
        let oldRuntimeWitnessKey = ["temporary", "Runtime", "Witness"].joined()
        let oldPrefix = ["temporary", "-"].joined()
        if witness.contains(oldRuntimeWitnessKey) {
            issues.append(camRouteIssue("CAM route event \(eventIndex) carries retired runtime witness evidence"))
        }
        if witness.contains(oldPrefix) {
            issues.append(camRouteIssue("CAM route event \(eventIndex) carries retired temporary evidence"))
        }
        guard witness.hasPrefix("cam-route:v6;") else {
            issues.append(camRouteIssue("CAM route event \(eventIndex) must use cam-route:v6"))
            return issues
        }

        let parts = witness.split(separator: ";", omittingEmptySubsequences: false).map(String.init)
        guard parts.first == "cam-route:v6" else {
            issues.append(camRouteIssue("CAM route event \(eventIndex) has malformed v6 prefix"))
            return issues
        }

        var valuesByField: [String: [String]] = [:]
        for field in parts.dropFirst() {
            let pair = field.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2, !pair[0].isEmpty else {
                issues.append(camRouteIssue("CAM route event \(eventIndex) has malformed witness field"))
                continue
            }
            valuesByField[String(pair[0]), default: []].append(String(pair[1]))
        }

        for field in valuesByField.keys.sorted() where (valuesByField[field] ?? []).count > 1 {
            issues.append(camRouteIssue("CAM route event \(eventIndex) has duplicate witness field '\(field)'"))
        }

        let allowedFields = Set(camRouteV6RequiredFields)
        for field in valuesByField.keys.sorted() where !allowedFields.contains(field) {
            issues.append(camRouteIssue("CAM route event \(eventIndex) has unknown witness field '\(field)'"))
        }

        for field in camRouteV6RequiredFields where valuesByField[field] == nil {
            issues.append(camRouteIssue("CAM route event \(eventIndex) is missing witness field '\(field)'"))
        }

        for field in camRouteV6NonEmptyFields where valuesByField[field]?.first?.isEmpty == true {
            issues.append(camRouteIssue("CAM route event \(eventIndex) has empty witness field '\(field)'"))
        }
        return issues
    }

    static let camRouteV6RequiredFields = [
        "cam",
        "exportABI",
        "export",
        "flow",
        "gates",
        "capabilities",
        "inputs",
        "outputs",
    ]

    static let camRouteV6NonEmptyFields = [
        "cam",
        "exportABI",
        "export",
        "flow",
        "inputs",
        "outputs",
    ]

    static func camRouteIssue(_ message: String) -> SmeltTraceIssue {
        SmeltTraceIssue(
            severity: .error,
            code: "runtimeCAMRouteInvalid",
            message: message
        )
    }

    static func runtimePhaseBlocks(_ loop: SmeltTraceLoop) -> [String: Set<String>] {
        var result: [String: Set<String>] = [:]
        for phase in loop.setup {
            result["setup:\(phase.name)"] = Set(phase.blocks)
        }
        for phase in loop.perStep {
            result["per-step:\(phase.name)"] = Set(phase.blocks)
        }
        return result
    }

    static func isLoopPhaseLabel(_ phase: String) -> Bool {
        phase.hasPrefix("setup:") || phase.hasPrefix("per-step:")
    }

    static func dispatchContract(_ table: SmeltTraceDispatchTable) -> SmeltTraceDispatchTableContract {
        SmeltTraceDispatchTableContract(
            name: table.name,
            exists: table.exists,
            sha256: table.sha256,
            parseError: table.parseError,
            totalRecords: table.totalRecords,
            dispatchCount: table.dispatchCount,
            swapCount: table.swapCount,
            topPipelines: table.topPipelines
        )
    }

    static func sidecarContract(_ sidecar: SmeltTraceSidecar) -> SmeltTraceSidecarContract {
        SmeltTraceSidecarContract(
            name: sidecar.name,
            exists: sidecar.exists,
            packageKind: sidecar.packageKind,
            modelName: sidecar.modelName,
            headlessTrunkABI: sidecar.headlessTrunkABI,
            manifestSHA256: sidecar.manifestSHA256,
            dispatchTables: sidecar.dispatchTables.map(dispatchContract),
            issues: sidecar.issues
        )
    }

    static func artifactContract(_ artifact: SmeltTraceArtifact) -> SmeltTraceArtifactContract {
        SmeltTraceArtifactContract(
            name: artifact.name,
            kind: artifact.kind,
            exists: artifact.exists,
            bytes: artifact.bytes,
            declaredSHA256: artifact.declaredSHA256,
            actualSHA256: artifact.actualSHA256
        )
    }

    static func contractDigest(_ contract: SmeltTraceContract) throws -> String {
        sha256Hex(try encodeCanonical(contract))
    }

    static func encodeCanonical<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func describe<T: Encodable>(_ value: T) -> String {
        if let data = try? encodeCanonical(value),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        return String(describing: value)
    }

    static func diff<T: Encodable & Equatable>(
        _ path: String,
        _ expected: T,
        _ actual: T,
        into differences: inout [SmeltTraceDifference]
    ) {
        guard expected != actual else { return }
        differences.append(SmeltTraceDifference(
            path: path,
            expected: describe(expected),
            actual: describe(actual)
        ))
    }

    static func witnessValidityDifferences(
        _ witness: SmeltTraceWitness,
        label: String
    ) -> [SmeltTraceDifference] {
        var differences: [SmeltTraceDifference] = []
        if witness.schemaVersion != 1 {
            differences.append(SmeltTraceDifference(
                path: "\(label).schemaVersion.validity",
                expected: "1",
                actual: "\(witness.schemaVersion)"
            ))
        }
        if witness.format != "smelt.trace.witness" {
            differences.append(SmeltTraceDifference(
                path: "\(label).format.validity",
                expected: "smelt.trace.witness",
                actual: witness.format
            ))
        }
        do {
            let actualDigest = try contractDigest(witness.contract)
            if actualDigest != witness.contractSHA256 {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contractSHA256.validity",
                    expected: actualDigest,
                    actual: witness.contractSHA256
                ))
            }
        } catch {
            differences.append(SmeltTraceDifference(
                path: "\(label).contractSHA256.validity",
                expected: "canonical contract digest",
                actual: "\(error)"
            ))
        }

        let expectedEventsCaptured = !witness.contract.events.isEmpty
        if witness.capture.eventsCaptured != expectedEventsCaptured {
            differences.append(SmeltTraceDifference(
                path: "\(label).capture.eventsCaptured.validity",
                expected: "\(expectedEventsCaptured)",
                actual: "\(witness.capture.eventsCaptured)"
            ))
        }

        let badEventIndices = witness.contract.events
            .enumerated()
            .filter { ordinal, event in event.index != ordinal }
            .map { ordinal, event in "event \(ordinal) has index \(event.index)" }
        if !badEventIndices.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.events.indices",
                expected: "sequential indices from 0",
                actual: badEventIndices.joined(separator: "; ")
            ))
        }

        let errorIssues = witness.contract.issues.filter { $0.severity == .error }
        if !errorIssues.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.issues.errors",
                expected: "[]",
                actual: describe(errorIssues)
            ))
        }
        differences += contractStructureDifferences(witness.contract, label: label)

        switch witness.capture.mode {
        case "package-contract+loop":
            let expectedEvents = traceEvents(
                loop: witness.contract.loop,
                blocks: witness.contract.blocks
            )
            if witness.contract.events != expectedEvents {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contract.events.staticValidity",
                    expected: describe(expectedEvents),
                    actual: describe(witness.contract.events)
                ))
            }
        case "package-contract+runtime":
            let expectedIssues = runtimeCoverageIssues(
                blocks: witness.contract.blocks,
                loop: witness.contract.loop,
                events: witness.contract.events
            )
            let missingIssues = expectedIssues.filter { !witness.contract.issues.contains($0) }
            if !missingIssues.isEmpty {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contract.issues.runtimeValidity",
                    expected: describe(expectedIssues),
                    actual: describe(witness.contract.issues)
                ))
            }
        default:
            differences.append(SmeltTraceDifference(
                path: "\(label).capture.mode.validity",
                expected: "package-contract+loop or package-contract+runtime",
                actual: witness.capture.mode
            ))
        }
        return differences
    }

    static func contractStructureDifferences(
        _ contract: SmeltTraceContract,
        label: String
    ) -> [SmeltTraceDifference] {
        var differences: [SmeltTraceDifference] = []
        let blockNames = contract.blocks.map(\.name)
        let duplicateBlocks = duplicateValues(blockNames)
        if !duplicateBlocks.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.blocks.names",
                expected: "unique block names",
                actual: duplicateBlocks.joined(separator: ", ")
            ))
        }
        if let graph = contract.graph, graph.blockCount != contract.blocks.count {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.graph.blockCount",
                expected: "\(contract.blocks.count)",
                actual: "\(graph.blockCount)"
            ))
        }
        if let loop = contract.loop {
            let knownBlocks = Set(blockNames)
            let unknownLoopBlocks = (loop.setup + loop.perStep)
                .flatMap(\.blocks)
                .filter { !knownBlocks.contains($0) }
            if !unknownLoopBlocks.isEmpty {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contract.loop.blocks",
                    expected: "loop blocks declared in contract.blocks",
                    actual: Array(Set(unknownLoopBlocks)).sorted().joined(separator: ", ")
                ))
            }
        }

        let artifactNames = contract.artifacts.map(\.name)
        let duplicateArtifacts = duplicateValues(artifactNames)
        if !duplicateArtifacts.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.artifacts.names",
                expected: "unique artifact names",
                actual: duplicateArtifacts.joined(separator: ", ")
            ))
        }
        let manifestArtifacts = contract.artifacts.filter { $0.name == "manifest.json" }
        if manifestArtifacts.count != 1 {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.artifacts.manifest",
                expected: "exactly one manifest.json artifact",
                actual: "\(manifestArtifacts.count)"
            ))
        } else {
            let manifest = manifestArtifacts[0]
            if !manifest.exists {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contract.artifacts.manifest.exists",
                    expected: "true",
                    actual: "false"
                ))
            }
            if manifest.actualSHA256 != contract.package.manifestSHA256 {
                differences.append(SmeltTraceDifference(
                    path: "\(label).contract.package.manifestSHA256",
                    expected: manifest.actualSHA256,
                    actual: contract.package.manifestSHA256
                ))
            }
        }
        let checksumMismatches = contract.artifacts.compactMap { artifact -> String? in
            guard let declared = artifact.declaredSHA256,
                  let actual = artifact.actualSHA256,
                  declared != actual else { return nil }
            return artifact.name
        }
        if !checksumMismatches.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.artifacts.declaredSHA256",
                expected: "declared checksums match actual checksums",
                actual: checksumMismatches.joined(separator: ", ")
            ))
        }

        let dispatchTableNames = contract.dispatchTables.map(\.name)
        let duplicateDispatchTables = duplicateValues(dispatchTableNames)
        if !duplicateDispatchTables.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.dispatchTables.names",
                expected: "unique dispatch table names",
                actual: duplicateDispatchTables.joined(separator: ", ")
            ))
        }
        let dispatchParseErrors = contract.dispatchTables.compactMap { table -> String? in
            table.parseError == nil ? nil : table.name
        }
        if !dispatchParseErrors.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.dispatchTables.parseError",
                expected: "no dispatch table parse errors",
                actual: dispatchParseErrors.joined(separator: ", ")
            ))
        }

        let sidecarNames = contract.sidecars.map(\.name)
        let duplicateSidecars = duplicateValues(sidecarNames)
        if !duplicateSidecars.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.sidecars.names",
                expected: "unique sidecar names",
                actual: duplicateSidecars.joined(separator: ", ")
            ))
        }
        let sidecarErrors = contract.sidecars.flatMap { sidecar in
            sidecar.issues
                .filter { $0.severity == .error }
                .map { "\(sidecar.name):\($0.code)" }
        }
        if !sidecarErrors.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.sidecars.issues.errors",
                expected: "[]",
                actual: sidecarErrors.joined(separator: ", ")
            ))
        }
        let duplicateSidecarDispatchTables = contract.sidecars.compactMap { sidecar -> String? in
            let duplicates = duplicateValues(sidecar.dispatchTables.map(\.name))
            return duplicates.isEmpty ? nil : "\(sidecar.name):\(duplicates.joined(separator: ","))"
        }
        if !duplicateSidecarDispatchTables.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.sidecars.dispatchTables.names",
                expected: "unique sidecar dispatch table names",
                actual: duplicateSidecarDispatchTables.joined(separator: ", ")
            ))
        }
        let sidecarDispatchParseErrors = contract.sidecars.flatMap { sidecar in
            sidecar.dispatchTables.compactMap { table -> String? in
                table.parseError == nil ? nil : "\(sidecar.name):\(table.name)"
            }
        }
        if !sidecarDispatchParseErrors.isEmpty {
            differences.append(SmeltTraceDifference(
                path: "\(label).contract.sidecars.dispatchTables.parseError",
                expected: "no sidecar dispatch table parse errors",
                actual: sidecarDispatchParseErrors.joined(separator: ", ")
            ))
        }
        return differences
    }

    static func duplicateValues(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var duplicates = Set<String>()
        for value in values where !seen.insert(value).inserted {
            duplicates.insert(value)
        }
        return duplicates.sorted()
    }

    static func verifySuiteCase(
        _ testCase: SmeltTraceSuiteCaseSpec,
        suite: SmeltTraceSuiteSpec,
        suiteBaseURL: URL,
        options: SmeltTraceSuiteOptions
    ) -> SmeltTraceSuiteCaseResult {
        let packageRaw = options.packageOverride ?? testCase.package ?? suite.package ?? ""
        let package = packageRaw.isEmpty
            ? ""
            : (options.packageOverride != nil
                ? packageRaw
                : resolveSuitePath(packageRaw, relativeTo: suiteBaseURL))
        let golden = resolveSuitePath(testCase.golden, relativeTo: suiteBaseURL)
        do {
            guard !packageRaw.isEmpty else {
                throw suiteError("case '\(testCase.name)' has no package")
            }
            let events = try testCase.events.map {
                try loadSuiteEvents(resolveSuitePath($0, relativeTo: suiteBaseURL))
            }
            let recordOptions = SmeltTraceRecordOptions(
                inspectOptions: options.inspectOptions,
                runtimeEvents: events ?? []
            )
            if options.updateGoldens {
                let witness = try record(packagePath: package, options: recordOptions)
                let errorIssues = witness.contract.issues.filter { $0.severity == .error }
                if !errorIssues.isEmpty {
                    let codes = errorIssues.map(\.code).joined(separator: ", ")
                    throw suiteError("case '\(testCase.name)' produced trace errors: \(codes)")
                }
                let goldenURL = URL(fileURLWithPath: golden)
                try FileManager.default.createDirectory(
                    at: goldenURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try writeWitness(witness, to: golden)
                return SmeltTraceSuiteCaseResult(
                    name: testCase.name,
                    package: package,
                    golden: golden,
                    matches: true,
                    updated: true,
                    differences: [],
                    error: nil
                )
            }
            let comparison: SmeltTraceComparison
            if events == nil {
                comparison = try replay(
                    packagePath: package,
                    from: golden,
                    options: SmeltTraceReplayOptions(inspectOptions: options.inspectOptions)
                )
            } else {
                comparison = try verify(
                    packagePath: package,
                    against: golden,
                    options: recordOptions
                )
            }
            return SmeltTraceSuiteCaseResult(
                name: testCase.name,
                package: package,
                golden: golden,
                matches: comparison.matches,
                updated: false,
                differences: comparison.differences,
                error: nil
            )
        } catch {
            return SmeltTraceSuiteCaseResult(
                name: testCase.name,
                package: package,
                golden: golden,
                matches: false,
                updated: false,
                differences: [],
                error: "\(error)"
            )
        }
    }

    static func loadSuiteEvents(_ path: String) throws -> [SmeltTraceEvent] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode([SmeltTraceEvent].self, from: data)
    }

    static func replayEvents(from witness: SmeltTraceWitness) -> [SmeltTraceEvent] {
        witness.capture.mode == "package-contract+runtime"
            ? witness.contract.events
            : []
    }

    static func resolveSuitePath(_ path: String, relativeTo baseURL: URL) -> String {
        guard !(path as NSString).isAbsolutePath else { return path }
        return baseURL.appendingPathComponent(path).path
    }

    static func suiteError(_ message: String) -> NSError {
        NSError(
            domain: "SmeltTraceSuite",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

private extension String {
    var trimmedSuiteValue: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
