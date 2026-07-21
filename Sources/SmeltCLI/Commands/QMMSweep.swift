import Foundation

func runQMMSweepCommand(_ args: [String]) {
    if args.contains("--help") || args.contains("-h") {
        fputs(
            "Usage: smelt lab sweep qmm [model.smeltpkg] [--library source|package] [--shader-dir DIR] [--batches CSV] [--iterations N] [--warmup N]\n",
            stdout
        )
        exit(0)
    }
    runKernelLabCommand(
        [args[0], "kernel-lab"]
            + Array(args.dropFirst(2))
            + ["--case", "qmm-sweep"]
    )
}
