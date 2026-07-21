// smelt — build, run, serve, and inspect compiled model packages.

import Foundation

let traceStartupMain =
    ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1"
if traceStartupMain {
    var info = kinfo_proc()
    var size = MemoryLayout<kinfo_proc>.stride
    var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
    if sysctl(&mib, 4, &info, &size, nil, 0) == 0 {
        let start = info.kp_proc.p_starttime
        let started = Double(start.tv_sec) + Double(start.tv_usec) / 1e6
        let ms = (Date().timeIntervalSince1970 - started) * 1000
        fputs(
            "startup: \(String(format: "%+7.1fms", ms))  exec -> main (dyld)\n",
            stderr
        )
    }
}

// Mutable so strict-argv commands can truncate the view at a `--`
// terminator — escaped prompt text must never reach a `--flag VALUE`
// lookup. Single mutation point, before any concurrent work starts.
nonisolated(unsafe) var args = CommandLine.arguments

let publicUsage = """
Smelt

Build a model package:
  smelt build <module-or-model-input> [build options]

Run one package request:
  smelt run <model.smeltpkg> [input] [package-declared options]

Serve a model package:
  smelt serve <model.smeltpkg> [serve options]

Create and run reusable agents:
  smelt agent <create|run|install|publish> ...

Measure and verify:
  smelt lab <subcommand> [lab options]

Inspect module contracts:
  smelt module <subcommand> [module options]
"""

guard args.count >= 2 else {
    fputs("\(publicUsage)\n", stderr)
    exit(1)
}

switch args[1] {
case "help", "--help", "-h":
    print(publicUsage)

case "run":
    runRunCommand()

case "build":
    runBuildCommand()

case "module":
    runCAMCommand()

case "lab":
    runLabCommand(args)

// Internal: warm worker spawned by `smelt run --linger`.
case "linger-worker":
    runLingerWorkerCommand()

case "serve":
    runServeCommand()

case "agent":
    runAgentCommand(Array(args.dropFirst(2)))

case "cas":
    runCasCommand()

default:
    fputs("Unknown command: \(args[1])\n", stderr)
    fputs("\n\(publicUsage)\n", stderr)
    exit(1)
}

if traceStartupMain {
    fputs("startup: command done; remaining wall time is teardown\n", stderr)
}
