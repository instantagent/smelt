// Per-dispatch GPU timing for the verify pass, gated on
// SMELT_PROFILE_VERIFY=<csv_path>. One-shot capture on the first
// prefillVerifyArgmax call, then transitions to .completed. Optional
// SMELT_PROFILE_VERIFY_MATCH=<csv_substrings> samples only matching
// pipeline names so large generated tables fit Metal's counter limit.

import Foundation
import Metal
import SmeltSchema

extension SmeltRuntime {

    enum VerifyProfileState {
        case disabled
        case armed
        case pending(SmeltVerifyProfileContext)
        case completed
    }

    final class SmeltVerifyProfileContext {
        let buffer: MTLCounterSampleBuffer
        let outputPath: String
        let maxDispatches: Int
        let pipelineNameMatches: [String]
        var dispatchCount: Int = 0
        var kernelNames: [String] = []

        init(
            buffer: MTLCounterSampleBuffer,
            outputPath: String,
            maxDispatches: Int,
            pipelineNameMatches: [String]
        ) {
            self.buffer = buffer
            self.outputPath = outputPath
            self.maxDispatches = maxDispatches
            self.pipelineNameMatches = pipelineNameMatches
            kernelNames.reserveCapacity(maxDispatches)
        }

        func shouldSample(pipelineName: String) -> Bool {
            pipelineNameMatches.isEmpty
                || pipelineNameMatches.contains { pipelineName.contains($0) }
        }
    }

    /// If profiling is armed, allocate a sample buffer sized for the
    /// upcoming verify call and transition to .pending. Returns the
    /// context the interpret loop should sample into, or nil to skip.
    func tryArmVerifyProfile(
        recordCount: Int
    ) -> SmeltVerifyProfileContext? {
        guard case .armed = verifyProfile else { return nil }
        guard let path = verifyProfileOutputPath else { return nil }
        guard recordCount > 0 else { return nil }

        guard device.supportsCounterSampling(.atStageBoundary) else {
            print("[SmeltVerifyProfile] device does not support "
                + ".atStageBoundary sampling; skipping")
            verifyProfile = .disabled
            return nil
        }
        guard let counterSet = device.counterSets?.first(where: {
            $0.name == MTLCommonCounterSet.timestamp.rawValue
        }) else {
            print("[SmeltVerifyProfile] no timestamp counter set on device; skipping")
            verifyProfile = .disabled
            return nil
        }

        let desc = MTLCounterSampleBufferDescriptor()
        desc.counterSet = counterSet
        desc.storageMode = .shared
        // Apple GPUs cap a timestamp sample buffer at 32 KiB: two UInt64
        // samples per dispatch leaves room for 2,048 dispatches. A verify
        // table can contain tens of thousands of records, so filtering is
        // part of the profiler contract rather than a model-specific escape.
        let maxDispatches = min(recordCount, 2_048)
        desc.sampleCount = maxDispatches * 2

        let buf: MTLCounterSampleBuffer
        do {
            buf = try device.makeCounterSampleBuffer(descriptor: desc)
        } catch {
            print("[SmeltVerifyProfile] makeCounterSampleBuffer failed: \(error)")
            verifyProfile = .disabled
            return nil
        }
        let pipelineNameMatches = ProcessInfo.processInfo.environment[
            "SMELT_PROFILE_VERIFY_MATCH"
        ]?.split(separator: ",").map(String.init).filter { !$0.isEmpty } ?? []
        let ctx = SmeltVerifyProfileContext(
            buffer: buf,
            outputPath: path,
            maxDispatches: maxDispatches,
            pipelineNameMatches: pipelineNameMatches
        )
        verifyProfile = .pending(ctx)
        return ctx
    }

    /// After the command buffer completes, resolve the GPU timestamps
    /// and dump a CSV of per-dispatch timings. Calibrates GPU ticks to
    /// nanoseconds via sampleTimestamps so the data is human-readable.
    func finalizeVerifyProfile(_ ctx: SmeltVerifyProfileContext) {
        defer { verifyProfile = .completed }
        let sampledCount = ctx.dispatchCount
        guard sampledCount > 0 else {
            print("[SmeltVerifyProfile] no dispatches sampled; skipping CSV")
            return
        }

        let totalSamples = sampledCount * 2
        let data: Data
        do {
            data = try ctx.buffer.resolveCounterRange(0..<totalSamples) ?? Data()
        } catch {
            print("[SmeltVerifyProfile] resolveCounterRange failed: \(error)")
            return
        }
        guard data.count >= totalSamples * MemoryLayout<UInt64>.stride else {
            print("[SmeltVerifyProfile] resolved data too small: "
                + "\(data.count) bytes, expected \(totalSamples * 8)")
            return
        }
        let timestamps = data.withUnsafeBytes { raw -> [UInt64] in
            let count = totalSamples
            let p = raw.bindMemory(to: UInt64.self)
            return Array(p[0..<count])
        }

        let start = device.sampleTimestamps()
        // Sleep ~1ms so the second sample is meaningfully later. We're
        // using two pairs to compute the GPU tick → ns scale factor.
        var ts = timespec(tv_sec: 0, tv_nsec: 1_000_000)
        nanosleep(&ts, nil)
        let end = device.sampleTimestamps()
        let cpuDelta = Double(end.cpu > start.cpu ? end.cpu - start.cpu : 1)
        let gpuDelta = Double(end.gpu > start.gpu ? end.gpu - start.gpu : 1)
        let nsPerTick = cpuDelta / gpuDelta

        var lines: [String] = []
        lines.reserveCapacity(sampledCount + 1)
        lines.append("idx,kernel,gpu_start_ticks,gpu_end_ticks,delta_ticks,delta_ns")
        for i in 0..<sampledCount {
            let start = timestamps[2 * i]
            let end = timestamps[2 * i + 1]
            let delta = end > start ? end - start : 0
            let ns = Double(delta) * nsPerTick
            let kernel = i < ctx.kernelNames.count ? ctx.kernelNames[i] : "?"
            lines.append("\(i),\(kernel),\(start),\(end),\(delta),\(Int(ns))")
        }
        let csv = lines.joined(separator: "\n") + "\n"
        do {
            try csv.write(toFile: ctx.outputPath, atomically: true, encoding: .utf8)
            print("[SmeltVerifyProfile] dumped \(sampledCount) dispatch timings to "
                + ctx.outputPath)
        } catch {
            print("[SmeltVerifyProfile] write failed: \(error)")
        }
    }

    /// Pipeline name for a dispatch record's pipeline index. Used in CSV.
    func pipelineName(for pipelineIndex: UInt16) -> String {
        let idx = Int(pipelineIndex)
        guard idx >= 0, idx < manifest.pipelines.count else {
            return "pipeline_\(pipelineIndex)"
        }
        return manifest.pipelines[idx]
    }
}
