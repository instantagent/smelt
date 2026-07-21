import Foundation
import SmeltLab

private let labUsage = """
Smelt Lab

Correctness:
  smelt lab verify <model.smeltpkg> [verify options]

Benchmarks:
  smelt lab bench decode <model.smeltpkg> [bench options]
  smelt lab bench prefill <model.smeltpkg> [prefill bench options]
  smelt lab bench verify <model.smeltpkg> --ids <csv> [--batches 1,2,4]
  smelt lab bench speculative <target.smeltpkg> (<drafter.smeltpkg> | --suffix-only)
  smelt lab bench logprobs <model.smeltpkg> [logprob bench options]
  smelt lab bench dispatch <model.smeltpkg> --ids <csv> --dispatch <n>
  smelt lab bench pipeline <model.smeltpkg> --ids <csv> --contains <substring>

Profiling and inspection:
  smelt lab profile decode <model.smeltpkg> [--kernels] [profile options]
  smelt lab profile prefill <model.smeltpkg> [profile options]
  smelt lab profile verify <model.smeltpkg> --ids <csv> [profile options]
  smelt lab inspect dispatches <model.smeltpkg> [--table decode|prefill|verify]
  smelt lab inspect cost <model.smeltpkg> [optimizer report options]
  smelt lab inspect runtime <model.smeltpkg> [runtime sizing options]

Experiments:
  smelt lab kernel <model.smeltpkg> [kernel lab options]
  smelt lab trace <trace subcommand> [trace options]
  smelt lab replay <trace.jsonl> [--package <model.smeltpkg>]
  smelt lab prefill <model.smeltpkg> [prefill options]
  smelt lab sweep qmm [model.smeltpkg] [sweep options]
  smelt lab optimize <model.smeltpkg> [optimizer-loop options]

Package contracts:
  smelt lab package-profile <performance-gate> [--model-name <name>]

Low-level probes:
  smelt lab probe <probe command> [probe options]
"""

func runLabCommand(_ args: [String]) {
    guard args.count >= 3 else {
        fputs(labUsage + "\n", stderr)
        exit(1)
    }

    switch args[2] {
    case "help", "--help", "-h":
        print(labUsage)
    case "verify":
        forwardLabCommand(args, as: "verify", dropping: 3, runVerifyCommand)
    case "bench":
        runLabBenchCommand(args)
    case "profile":
        runLabProfileCommand(args)
    case "inspect":
        runLabInspectCommand(args)
    case "kernel":
        forwardLabCommand(args, as: "kernel-lab", dropping: 3, runKernelLabCommand)
    case "trace":
        forwardLabCommand(args, as: "trace", dropping: 3, runTraceCommand)
    case "replay":
        forwardLabCommand(args, as: "replay", dropping: 3, runReplayCommand)
    case "prefill":
        forwardLabCommand(args, as: "prefill", dropping: 3, runPrefillCommand)
    case "optimize":
        forwardLabCommand(args, as: "optimize-next", dropping: 3, runOptimizeNextCommand)
    case "package-profile":
        forwardLabCommand(args, as: "module-profile", dropping: 3, runCAMProfileCommand)
    case "sweep":
        runLabSweepCommand(args)
    case "probe":
        runSmeltLabProbe(arguments: Array(args.dropFirst(3)))
    default:
        fputs("Unknown smelt lab command: \(args[2])\n\n", stderr)
        fputs(labUsage + "\n", stderr)
        exit(1)
    }
}

private func runLabBenchCommand(_ args: [String]) {
    guard args.count >= 4 else {
        fputs(labUsage + "\n", stderr)
        exit(1)
    }
    switch args[3] {
    case "decode":
        forwardLabCommand(args, as: "bench", dropping: 4, runBenchCommand)
    case "prefill":
        forwardLabCommand(args, as: "prefill-bench", dropping: 4, runPrefillBenchCommand)
    case "verify":
        runLabProbeWithPositionalPackage(args, command: "bench-verify-argmax")
    case "speculative":
        forwardLabCommand(args, as: "mtp-bench", dropping: 4, runMtpBenchCommand)
    case "logprobs":
        forwardLabCommand(args, as: "bench-logprobs", dropping: 4, runBenchLogprobsCommand)
    case "dispatch":
        runLabProbeWithPositionalPackage(args, command: "bench-dispatch")
    case "pipeline":
        runLabProbeWithPositionalPackage(args, command: "bench-pipeline")
    default:
        fputs("Unknown smelt lab bench workload: \(args[3])\n", stderr)
        exit(1)
    }
}

private func runLabProfileCommand(_ args: [String]) {
    guard args.count >= 4 else {
        fputs(labUsage + "\n", stderr)
        exit(1)
    }
    switch args[3] {
    case "decode":
        if args.contains("--kernels") {
            forwardLabCommand(args, as: "kernels", dropping: 4, runKernelsCommand)
        } else {
            forwardLabCommand(args, as: "profile", dropping: 4, runProfileCommand)
        }
    case "prefill":
        forwardLabCommand(args, as: "prefill-kernels", dropping: 4, runPrefillKernelsCommand)
    case "verify":
        forwardLabCommand(args, as: "profile-verify", dropping: 4, runVerifyProfileCommand)
    default:
        fputs("Unknown smelt lab profile workload: \(args[3])\n", stderr)
        exit(1)
    }
}

private func runLabInspectCommand(_ args: [String]) {
    guard args.count >= 4 else {
        fputs(labUsage + "\n", stderr)
        exit(1)
    }
    switch args[3] {
    case "dispatches":
        forwardLabCommand(args, as: "dispatches", dropping: 4, runDispatchesCommand)
    case "cost":
        forwardLabCommand(args, as: "optimizer-report", dropping: 4, runOptimizerReportCommand)
    case "runtime":
        runLabProbeWithPositionalPackage(args, command: "text-runtime-stats")
    default:
        fputs("Unknown smelt lab inspect subject: \(args[3])\n", stderr)
        exit(1)
    }
}

private func runLabSweepCommand(_ args: [String]) {
    guard args.count >= 4 else {
        fputs(labUsage + "\n", stderr)
        exit(1)
    }
    switch args[3] {
    case "qmm":
        forwardLabCommand(args, as: "sweep-qmm", dropping: 4, runQMMSweepCommand)
    default:
        fputs("Unknown smelt lab sweep family: \(args[3])\n", stderr)
        exit(1)
    }
}

private func runLabProbeWithPositionalPackage(_ args: [String], command: String) {
    guard args.count >= 5 else {
        fputs("smelt lab: \(args[2]) \(args[3]) requires a package path\n", stderr)
        exit(1)
    }
    runSmeltLabProbe(
        arguments: [command, "--package", args[4]] + Array(args.dropFirst(5))
    )
}

private func forwardLabCommand(
    _ args: [String],
    as command: String,
    dropping prefixCount: Int,
    _ body: ([String]) -> Void
) {
    body([args[0], command] + Array(args.dropFirst(prefixCount)))
}
