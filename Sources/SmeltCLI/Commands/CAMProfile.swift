import Foundation
import SmeltSchema

func runCAMProfileCommand() {
    guard args.count == 3 || args.count == 5,
          !args[2].hasPrefix("-"),
          args.count == 3 || args[3] == "--model-name" && !args[4].isEmpty
    else {
        fputs("Usage: smelt module-profile <performance-gate> [--model-name <model-name>]\n", stderr)
        exit(1)
    }

    let gate = args[2]
    let modelName = args.count == 5 ? args[4] : nil
    guard SmeltPackagePerformanceGateID.known.contains(gate) else {
        fputs("smelt module-profile: unknown performance gate '\(gate)'\n", stderr)
        exit(1)
    }

    let profile = SmeltPackagePerformanceProfiles.profile(for: gate, modelName: modelName)
    print("gate\t\(profile.gate)")
    print("command\t\(profile.command.rawValue)")
    for label in profile.requiredTraceLabels {
        print("required_trace_label\t\(label)")
    }
    for metric in profile.requiredOutputMetrics {
        print("required_output_metric\t\(metric)")
    }
    for bound in profile.minBounds {
        print("min_bound\t\(bound.metric)\t\(formatCAMProfileNumber(bound.min))\t\(bound.unit)")
    }
    for bound in profile.maxBounds {
        print("max_bound\t\(bound.metric)\t\(formatCAMProfileNumber(bound.max))\t\(bound.unit)")
    }

    switch gate {
    case SmeltPackagePerformanceGateID.qwen3TTSTTFA:
        print("structure_id\t\(SmeltPackageStructureProfileID.qwen3TTSRunnable)")
    default:
        break
    }
}

private func formatCAMProfileNumber(_ value: Double) -> String {
    String(format: "%g", value)
}
