import Foundation
import SmeltRuntime

func runKernelsCommand() {
    guard args.count >= 3 else {
        fputs("Usage: smelt kernels <model.smeltpkg> [--iterations N] [--position N] [--trace-start N --trace-length N]\n", stderr)
        exit(1)
    }
    let pkgPath = args[2]
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: pkgPath,
        request: .profileDecodeKernels,
        verb: "kernels"
    )
    let iterations = Int(parseArg("--iterations", default: "10")) ?? 10
    let fixedPosition = Int32(parseArg("--position", default: ""))
    let traceStart = Int32(parseArg("--trace-start", default: ""))
    let traceLength = Int(parseArg("--trace-length", default: "")) ?? 0

    do {
        let runtime = try construction.makeRuntime(contextLimit: nil)
        if traceStart != nil || traceLength > 0 {
            guard fixedPosition == nil else {
                fputs("Use either --position or trace mode, not both\n", stderr)
                exit(1)
            }
            guard let traceStart, traceLength > 0 else {
                fputs("Trace mode requires both --trace-start N and --trace-length N\n", stderr)
                exit(1)
            }

            let endPosition = traceStart + Int32(traceLength)
            try primeDecodeState(runtime: runtime, targetPosition: traceStart)
            var samples: [KernelTraceSample] = []
            for position in traceStart..<endPosition {
                let results = try runtime.profileKernels(tokenId: 0, position: position)
                var perKernelUs: [String: Double] = [:]
                var totalUs = 0.0
                for result in results {
                    perKernelUs[result.name] = result.totalGpuUs
                    totalUs += result.totalGpuUs
                }
                samples.append(
                    KernelTraceSample(
                        position: position,
                        totalUs: totalUs,
                        perKernelUs: perKernelUs
                    )
                )
            }
            printKernelTrace(samples, startPosition: traceStart)
            return
        }

        if let fixedPosition {
            fputs("Per-kernel GPU profile (\(iterations) iterations, fixed position \(fixedPosition))...\n\n", stderr)
        } else {
            fputs("Per-kernel GPU profile (\(iterations) iterations)...\n\n", stderr)
        }

        if let fixedPosition {
            try primeDecodeState(runtime: runtime, targetPosition: fixedPosition)
            _ = try runtime.decodeStep(tokenId: 0, position: fixedPosition)
        } else {
            _ = try warmSequentialDecode(
                runtime: runtime,
                iterations: 1
            )
        }
        var accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)] = [:]
        var nextPosition = Int32(1)
        for _ in 0..<iterations {
            let position: Int32
            if let fixedPosition {
                try primeDecodeState(runtime: runtime, targetPosition: fixedPosition)
                position = fixedPosition
            } else {
                position = nextSequentialDecodePosition(
                    runtime: runtime,
                    nextPosition: &nextPosition
                )
            }
            let results = try runtime.profileKernels(
                tokenId: 0, position: position
            )
            for r in results {
                let prev = accumulated[r.name, default: (0, 0, 0)]
                accumulated[r.name] = (
                    prev.totalUs + r.totalGpuUs,
                    prev.count + 1,
                    r.dispatchCount
                )
            }
        }

        printKernelProfileTable(
            accumulated: accumulated,
            iterations: iterations,
            perTokenUnits: "ms/tok"
        )
    } catch {
        fputs("Kernel profile failed: \(error)\n", stderr)
        exit(1)
    }
}
