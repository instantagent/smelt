import Foundation
import SmeltCompiler
import SmeltSchema

func runCAMCommand() {
    let usage = """
    Usage:
      smelt module check <model.module.json>
      smelt module admission <model.module.json> --json
      smelt module ir <model.module.json> [--json|--hashes]
      smelt module cost <model.module.json> --storage <bf16|affine-u4-g64|nvfp4> --context <tokens>
        [--block <id>] [--checkpoint-bytes <bytes>] [--memory-limit-bytes <bytes>]
        [--bandwidth-gbps <decimal-GB/s>] [--output <report.json>]
      smelt module hf-inventory <owner/model> --revision <revision> [--output <inventory.json>]
    """
    guard args.count >= 3 else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }
    if args[2] == "--help" || args[2] == "-h" {
        print(usage)
        return
    }

    switch args[2] {
    case "check":
        runCAMCheckCommand(usage: usage)
    case "admission":
        runCAMAdmissionCommand(usage: usage)
    case "ir":
        runCAMIRCommand(usage: usage)
    case "cost":
        runCAMCostCommand(usage: usage)
    case "hf-inventory":
        runCAMHFInventoryCommand(usage: usage)
    default:
        fputs("smelt module: unknown subcommand '\(args[2])'\n", stderr)
        fputs("\(usage)\n", stderr)
        exit(1)
    }
}

private func runCAMHFInventoryCommand(usage: String) {
    var modelID: String?
    var revision: String?
    var outputPath: String?
    var idx = 3
    while idx < args.count {
        switch args[idx] {
        case "--revision":
            guard idx + 1 < args.count else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            revision = args[idx + 1]
            idx += 2
        case "--output":
            guard idx + 1 < args.count else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            outputPath = args[idx + 1]
            idx += 2
        default:
            guard !args[idx].hasPrefix("--"), modelID == nil else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            modelID = args[idx]
            idx += 1
        }
    }
    guard let modelID, let revision else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }

    do {
        let inventory = try SmeltHFCheckpointInventoryProbe.probe(
            modelID: modelID,
            revision: revision
        )
        var data = try inventory.encodeJSON()
        data.append(0x0A)
        if let outputPath {
            try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        } else {
            FileHandle.standardOutput.write(data)
        }
    } catch {
        fputs("smelt module hf-inventory failed: \(error)\n", stderr)
        exit(1)
    }
}

private func runCAMCostCommand(usage: String) {
    var path: String?
    var blockID = "trunk"
    var storage: SmeltCAMWeightStorageProfile?
    var contextLength: Int?
    var checkpointBytes: UInt64?
    var memoryLimitBytes: UInt64?
    var bandwidthBytesPerSecond: Double?
    var outputPath: String?

    var idx = 3
    while idx < args.count {
        let arg = args[idx]
        func nextValue() -> String? {
            guard idx + 1 < args.count else { return nil }
            return args[idx + 1]
        }
        switch arg {
        case "--block":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            blockID = value
            idx += 2
        case "--storage":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            storage = SmeltCAMWeightStorageProfile(rawValue: value)
            idx += 2
        case "--context":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            contextLength = Int(value)
            idx += 2
        case "--checkpoint-bytes":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            checkpointBytes = UInt64(value)
            idx += 2
        case "--memory-limit-bytes":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            memoryLimitBytes = UInt64(value)
            idx += 2
        case "--bandwidth-gbps":
            guard let value = nextValue(), let gbps = Double(value), gbps > 0 else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            bandwidthBytesPerSecond = gbps * 1_000_000_000
            idx += 2
        case "--output":
            guard let value = nextValue() else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            outputPath = value
            idx += 2
        default:
            if arg.hasPrefix("--") || path != nil {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            path = arg
            idx += 1
        }
    }

    guard let path, let storage, let contextLength, contextLength > 0 else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }
    rejectCAMGrammarInput(path)

    do {
        let module = try SmeltCAMIR.decodeModule(at: URL(fileURLWithPath: path))
        let report = try SmeltCAMStaticCostModel.report(
            module: module,
            blockID: blockID,
            scenario: SmeltCAMStaticCostScenario(
                storage: storage,
                contextLength: contextLength,
                exactCheckpointBytes: checkpointBytes,
                memoryLimitBytes: memoryLimitBytes,
                sustainedMemoryBandwidthBytesPerSecond: bandwidthBytesPerSecond
            )
        )
        var data = try report.encodeJSON()
        data.append(0x0A)
        if let outputPath {
            try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        } else {
            FileHandle.standardOutput.write(data)
        }
    } catch {
        fputs("smelt module cost failed: \(error)\n", stderr)
        exit(1)
    }
}

/// Fail closed if `path` is a removed `.cam` grammar input, before any decode
/// attempt surfaces a confusing JSON error. Shared by every `smelt module` verb.
private func rejectCAMGrammarInput(_ path: String) {
    if let diagnostic = SmeltCAMGrammarRemoval.rejectionDiagnostic(forInputPath: path) {
        fputs("smelt module: \(diagnostic)\n", stderr)
        exit(1)
    }
}

private func runCAMCheckCommand(usage: String) {
    guard args.count == 4, !args[3].hasPrefix("-") else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }

    let path = args[3]
    rejectCAMGrammarInput(path)
    do {
        let ir = try SmeltCAMIR.decodeModule(at: URL(fileURLWithPath: path))
        let admission = SmeltCAMFeatureAdmission(
            descriptor: try SmeltCAMPackageDescriptor(from: ir)
        )
        print("module\t\(ir.module.id)")
        print("imports\t\(summary(ir.imports.map(\.alias)))")
        print("exports\t\(summary(ir.exports.map(\.id)))")
        print("flows\t\(summary(ir.flows.map(\.id)))")
        print("semantic_sha256\t\(try ir.semanticSHA256())")
        print("export_abi_sha256\t\(try ir.exportABISHA256())")
        print("required_feature_codes\t\(summary(admission.requiredFeatureSet))")
        print("unsupported_feature_codes\t\(summary(admission.unsupportedFeatureSet))")
        for obligation in admission.requiredObligations {
            print("required_obligation\t\(obligation.checkSummary)")
        }
        for obligation in admission.unsupportedFeatures {
            print("unsupported_obligation\t\(obligation.checkSummary)")
        }
    } catch {
        fputs("smelt module check failed: \(error)\n", stderr)
        exit(1)
    }
}

private func runCAMAdmissionCommand(usage: String) {
    guard args.count == 5, !args[3].hasPrefix("-"), args[4] == "--json" else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }

    let path = args[3]
    rejectCAMGrammarInput(path)
    do {
        let ir = try SmeltCAMIR.decodeModule(at: URL(fileURLWithPath: path))
        let admission = SmeltCAMFeatureAdmission(
            descriptor: try SmeltCAMPackageDescriptor(from: ir)
        )
        let data = try JSONSerialization.data(
            withJSONObject: admissionJSONObject(admission),
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        fputs("smelt module admission failed: \(error)\n", stderr)
        exit(1)
    }
}

private enum CAMIROutput {
    case json
    case hashes
}

private func runCAMIRCommand(usage: String) {
    var path: String?
    var output: CAMIROutput?

    var idx = 3
    while idx < args.count {
        let arg = args[idx]
        switch arg {
        case "--json":
            guard output == nil else {
                fputs("smelt module ir: conflicting output options\n", stderr)
                exit(1)
            }
            output = .json
            idx += 1
        case "--hashes":
            guard output == nil else {
                fputs("smelt module ir: conflicting output options\n", stderr)
                exit(1)
            }
            output = .hashes
            idx += 1
        default:
            if arg.hasPrefix("--") {
                fputs("smelt module ir: unknown option '\(arg)'\n", stderr)
                exit(1)
            }
            guard path == nil else {
                fputs("\(usage)\n", stderr)
                exit(1)
            }
            path = arg
            idx += 1
        }
    }

    guard let path else {
        fputs("\(usage)\n", stderr)
        exit(1)
    }
    rejectCAMGrammarInput(path)

    do {
        let ir = try SmeltCAMIR.decodeModule(at: URL(fileURLWithPath: path))
        switch output ?? .json {
        case .json:
            FileHandle.standardOutput.write(try ir.canonicalJSONData())
            FileHandle.standardOutput.write(Data("\n".utf8))
        case .hashes:
            print("semantic_sha256\t\(try ir.semanticSHA256())")
            print("export_abi_sha256\t\(try ir.exportABISHA256())")
        }
    } catch {
        fputs("smelt module ir failed: \(error)\n", stderr)
        exit(1)
    }
}

private func summary(_ values: [String]) -> String {
    values.isEmpty ? "-" : values.sorted().joined(separator: ",")
}

private func admissionJSONObject(_ admission: SmeltCAMFeatureAdmission) -> [String: Any] {
    [
        "schema": admission.schema,
        "stage": admission.stage,
        "descriptor_schema": admission.descriptorSchema,
        "descriptor_version": admission.descriptorVersion,
        "module_semantic_sha256": admission.camSemanticSHA256,
        "export_abi_sha256": admission.exportABISHA256,
        "required_feature_set": admission.requiredFeatureSet,
        "consumed_feature_set": admission.consumedFeatureSet,
        "unsupported_feature_set": admission.unsupportedFeatureSet,
        "required_obligation_ids": admission.requiredObligationIDs,
        "consumed_obligation_ids": admission.consumedObligationIDs,
        "unsupported_obligation_ids": admission.unsupportedObligationIDs,
        "required_obligations": admission.requiredObligations.map(obligationJSONObject),
        "unsupported_obligations": admission.unsupportedFeatures.map(obligationJSONObject),
    ]
}

private func obligationJSONObject(
    _ obligation: SmeltCAMFeatureAdmission.FeatureRequirement
) -> [String: Any] {
    [
        "code": obligation.code,
        "scope": obligation.scope,
        "parameters": obligation.parameters,
        "canonical_id": obligation.canonicalID,
        "evidence": obligation.evidence,
    ]
}
