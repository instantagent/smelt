import Foundation
import SmeltRuntime
import SmeltSchema

private let camRuntimeConsumedFeatureSet: [String] = []

func loadCAMPackageCapabilitiesOrExit(
    packagePath: String,
    verb: String
) -> SmeltCAMPackageCapabilities? {
    do {
        return try SmeltCAMPackageCapabilities.loadIfPresent(
            packageURL: URL(fileURLWithPath: packagePath, isDirectory: true),
            consumedFeatureSet: camRuntimeConsumedFeatureSet
        )
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
}

func requireCAMPackageCapabilitiesOrExit(
    packagePath: String,
    verb: String
) -> SmeltCAMPackageCapabilities {
    guard let capabilities = loadCAMPackageCapabilitiesOrExit(
        packagePath: packagePath,
        verb: verb
    ) else {
        fputs(
            "smelt \(verb): expected module descriptor but package has no "
                + "\(SmeltCAMPackageDescriptor.packageFileName)\n",
            stderr
        )
        exit(1)
    }
    requireCAMFeatureAdmissionOrExit(capabilities, verb: verb)
    return capabilities
}

func requireCAMPackageInventoryOrExit(
    _ capabilities: SmeltCAMPackageCapabilities,
    packagePath: String,
    verb: String,
    requireAuthoredInventory: Bool = false,
    allowMissingFiles: Set<String> = []
) {
    do {
        try SmeltCAMPackageInventoryValidator.validate(
            capabilities,
            packageURL: URL(fileURLWithPath: packagePath, isDirectory: true),
            requireAuthoredInventory: requireAuthoredInventory,
            allowMissingFiles: allowMissingFiles
        )
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
}

func requireCAMCapabilityFilesOrExit(
    _ requiredFiles: Set<String>,
    packagePath: String,
    verb: String
) {
    let missing = requiredFiles.sorted().filter { relativePath in
        !FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: packagePath, isDirectory: true)
                .appendingPathComponent(relativePath)
                .path
        )
    }
    guard missing.isEmpty else {
        fputs(
            "smelt \(verb): module capability files missing: "
                + missing.joined(separator: ",")
                + "\n",
            stderr
        )
        exit(1)
    }
}

private func requireCAMFeatureAdmissionOrExit(
    _ capabilities: SmeltCAMPackageCapabilities,
    verb: String
) {
    let admission = capabilities.featureAdmission
    guard !admission.hasUnsupportedFeatures else {
        fputs(
            "smelt \(verb): module feature admission failed at \(admission.stage): "
                + "\(admission.unsupportedDiagnostic)\n",
            stderr
        )
        exit(1)
    }
}

func describeCAMDecision(_ decision: SmeltCAMPackageCapabilities.Decision) -> String {
    "export '\(decision.exportID)' flow '\(decision.flowID)' "
        + "selectedInputs '\(describeCAMPorts(decision.selectedInputPorts))' "
        + "selectedOutputs '\(describeCAMPorts(decision.selectedOutputPorts))' "
        + "gates '\(decision.matchedGateIDs.sorted().joined(separator: ","))' "
        + "capabilities '\(decision.authoredCapabilities.sorted().joined(separator: ","))'"
}

private func describeCAMPorts(_ ports: [SmeltCAMPackageDescriptor.Port]) -> String {
    ports
        .map { port in
            let attributes = port.type.attributes
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            return "\(port.portName):\(port.type.typeName){\(attributes)}"
                + ":\(port.optional ? "optional" : "required")"
        }
        .sorted()
        .joined(separator: ",")
}

/// Resolves a direct package path from `--package <path>`, the positional
/// argument at args[2], or the current working directory. Smelt deliberately
/// does not resolve installed Smelt names.
func resolvePackagePath(usage: [String]) -> (path: String, promptStartIndex: Int) {
    let packageFlag = parseArg("--package")
    if !packageFlag.isEmpty {
        return (packageFlag, 2)
    }
    if args.count >= 3, isPackagePath(args[2]) {
        return (args[2], 3)
    }
    if let inferred = inferPackagePathFromCWD() {
        return (inferred, 2)
    }
    for line in usage {
        fputs(line, stderr)
    }
    exit(1)
}

struct KernelProfileRow {
    let name: String
    let avgUs: Double
    let dispatches: Int
    let pct: Double
}

/// Prints the per-kernel GPU timing table shared by `kernels` and
/// `prefill-kernels`. `perTokenUnits` is the trailing label
/// ("ms/tok" vs "ms/prefill").
func printKernelProfileTable(
    accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)],
    iterations: Int,
    perTokenUnits: String
) {
    let grandTotal = accumulated.values.map(\.totalUs).reduce(0, +) / Double(iterations)
    var rows = accumulated.map { name, val in
        KernelProfileRow(
            name: name,
            avgUs: val.totalUs / Double(val.count),
            dispatches: val.dispatches,
            pct: grandTotal > 0 ? (val.totalUs / Double(val.count) / grandTotal) * 100 : 0
        )
    }
    rows.sort { $0.avgUs > $1.avgUs }

    let header = "  \("Kernel".padding(toLength: 28, withPad: " ", startingAt: 0)) #disp    GPU µs    % total"
    fputs(header + "\n", stderr)
    fputs("  " + String(repeating: "─", count: 62) + "\n", stderr)
    for r in rows {
        let bar = String(repeating: "█", count: max(0, Int(r.pct / 2.5)))
        let name = r.name.padding(toLength: 28, withPad: " ", startingAt: 0)
        let line = "  \(name) \(String(format: "%5d", r.dispatches)) \(String(format: "%9.0f", r.avgUs)) \(String(format: "%7.1f", r.pct))%  \(bar)"
        fputs(line + "\n", stderr)
    }
    fputs("  " + String(repeating: "─", count: 62) + "\n", stderr)
    fputs("  TOTAL\(String(repeating: " ", count: 28)) \(String(format: "%9.0f", grandTotal)) µs\n", stderr)
    fputs("  \(String(repeating: " ", count: 35)) \(String(format: "%9.1f", grandTotal / 1_000)) \(perTokenUnits)\n", stderr)
}
