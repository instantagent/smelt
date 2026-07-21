import Foundation
import SmeltRuntime

func runProfileCommand(_ args: [String]) {
    guard args.count >= 3 else {
        fputs("Usage: smelt lab profile decode <model.smeltpkg> [--iterations N] [--position N]\n", stderr)
        exit(1)
    }
    let pkgPath = args[2]
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: pkgPath,
        request: .runText,
        verb: "lab profile decode"
    )
    let iterations = Int(parseArg(args, "--iterations", default: "100")) ?? 100
    let fixedPosition = Int32(parseArg(args, "--position", default: ""))

    do {
        let runtime = try construction.makeRuntime(contextLimit: nil)
        if let fixedPosition {
            fputs("Profiling \(iterations) decode steps at fixed position \(fixedPosition)...\n", stderr)
        } else {
            fputs("Profiling \(iterations) decode steps...\n", stderr)
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

        var cpuTimes = [Double]()
        var gpuTimes = [Double]()
        var readTimes = [Double]()
        var pureGpuTimes = [Double]()
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
            let result = try runtime.profileDecodeStep(
                tokenId: 0, position: position
            )
            cpuTimes.append(result.cpuMs)
            gpuTimes.append(result.gpuMs)
            readTimes.append(result.readMs)
            pureGpuTimes.append(result.pureGpuMs)
        }

        cpuTimes.sort()
        gpuTimes.sort()
        readTimes.sort()
        pureGpuTimes.sort()

        let medCPU = cpuTimes[iterations / 2]
        let medGPU = gpuTimes[iterations / 2]
        let medRead = readTimes[iterations / 2]
        let medPureGPU = pureGpuTimes[iterations / 2]
        let medTotal = medCPU + medGPU + medRead

        fputs("Decode step profile (median of \(iterations)):\n", stderr)
        fputs(
            "  CPU encode:   \(String(format: "%5.2f", medCPU))ms"
                + "  (\(String(format: "%.0f", medCPU / medTotal * 100))%)\n",
            stderr
        )
        fputs(
            "  GPU execute:  \(String(format: "%5.2f", medGPU))ms"
                + "  (\(String(format: "%.0f", medGPU / medTotal * 100))%)\n",
            stderr
        )
        fputs(
            "  Pure GPU:     \(String(format: "%5.2f", medPureGPU))ms"
                + "  (Metal timestamps)\n",
            stderr
        )
        fputs(
            "  Submit gap:   \(String(format: "%5.2f", medGPU - medPureGPU))ms"
                + "  (CPU→GPU latency)\n",
            stderr
        )
        fputs(
            "  Readback:     \(String(format: "%5.2f", medRead))ms"
                + "  (\(String(format: "%.0f", medRead / medTotal * 100))%)\n",
            stderr
        )
        fputs(
            "  TOTAL:        \(String(format: "%5.2f", medTotal))ms/tok"
                + "  (\(String(format: "%.0f", 1_000 / medTotal)) tok/s)\n",
            stderr
        )
    } catch {
        fputs("Profile failed: \(error)\n", stderr)
        exit(1)
    }
}
