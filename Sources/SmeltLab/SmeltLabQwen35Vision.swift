import Foundation
import Metal
import SmeltCompiler
import SmeltRuntime
import SmeltSchema

func checkQwen35Vision(args: [String]) throws {
    let modulePath = try parseArg("--module", from: args)
    let checkpointPath = try parseArg("--checkpoint", from: args)
    let moduleData = try Data(contentsOf: URL(fileURLWithPath: modulePath))
    let module = try JSONDecoder().decode(SmeltCAMIR.self, from: moduleData).validated()
    let checkpoint = try SafetensorsLoader(directory: checkpointPath)
    let plan = try SmeltQwen35VisionCheckpointPlan(
        module: module,
        checkpoint: checkpoint
    )
    let c = plan.config
    print(
        "vision-check passed"
            + " source=\(plan.sourceID)"
            + " tensors=\(plan.tensors.count)"
            + " layers=\(c.layerCount)"
            + " hidden=\(c.hiddenSize)"
            + " heads=\(c.headCount)"
            + " ffn=\(c.intermediateSize)"
            + " patch=\(c.temporalPatchSize)x\(c.patchSize)x\(c.patchSize)"
            + " merge=\(c.spatialMergeSize)"
            + " output=\(c.outputHiddenSize)"
    )
}

func runQwen35Vision(args: [String]) throws {
    let componentPath = try parseArg("--component", from: args, default: "")
    let modulePath = try parseArg("--module", from: args, default: "")
    let checkpointPath = try parseArg("--checkpoint", from: args, default: "")
    let shaderDirectory = try parseArg(
        "--shader-dir",
        from: args,
        default: "Resources/Shaders"
    )
    if componentPath.isEmpty && (modulePath.isEmpty || checkpointPath.isEmpty) {
        throw UsageError(
            message: "vision run requires --component or both --module and --checkpoint"
        )
    }
    let height = try parseOptionalPositiveIntArg("--height", from: args) ?? 2
    let width = try parseOptionalPositiveIntArg("--width", from: args) ?? 2
    let temporal = try parseOptionalPositiveIntArg("--temporal", from: args) ?? 1
    let packedSegments = try parseOptionalPositiveIntArg(
        "--packed-segments", from: args
    ) ?? 1
    let warmupIterations = try parseOptionalPositiveIntArg("--warmup", from: args) ?? 0
    let measuredIterations = try parseOptionalPositiveIntArg("--iterations", from: args) ?? 1
    let diagnoseNonFinite = args.contains("--diagnose")
    let referencePath = try parseArg("--reference-f32", from: args, default: "")
    let outputPath = try parseArg("--output-f32", from: args, default: "")
    let costJSONPath = try parseArg("--cost-json", from: args, default: "")
    let costCalibrationJSONPath = try parseArg(
        "--cost-calibration-json",
        from: args,
        default: ""
    )
    let profileCostIterations = try parseOptionalPositiveIntArg(
        "--profile-cost-iterations",
        from: args
    ) ?? (args.contains("--profile-cost") ? 1 : 0)
    let imagePath = try parseArg("--image", from: args, default: "")
    let gemmBackendText = try parseArg("--gemm", from: args, default: "mps")
    let gemmBackend: SmeltQwen35VisionGEMMBackend
    switch gemmBackendText {
    case "mps": gemmBackend = .mps
    case "reference-metal": gemmBackend = .referenceMetal
    default:
        throw UsageError(message: "--gemm must be mps or reference-metal")
    }
    let attentionBackendText = try parseArg(
        "--attention", from: args, default: "mps-staged"
    )
    let attentionBackend: SmeltQwen35VisionAttentionBackend
    switch attentionBackendText {
    case "reference": attentionBackend = .reference
    case "mps-staged": attentionBackend = .mpsStaged
    default:
        throw UsageError(
            message: "--attention must be reference or mps-staged"
        )
    }

    let artifact: SmeltQwen35VisionArtifact?
    let moduleData: Data
    let module: SmeltCAMIR
    let checkpoint: any CheckpointTensorSource
    let modelSource: String
    if componentPath.isEmpty {
        artifact = nil
        moduleData = try Data(contentsOf: URL(fileURLWithPath: modulePath))
        module = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: moduleData
        ).validated()
        checkpoint = try SafetensorsLoader(directory: checkpointPath)
        modelSource = "source-checkpoint"
    } else {
        let loaded = try SmeltQwen35VisionArtifact(
            path: componentPath,
            verify: true
        )
        artifact = loaded
        module = loaded.module
        moduleData = try module.canonicalJSONData(prettyPrinted: false)
        checkpoint = loaded
        modelSource = "verified-component"
    }
    let plan = try SmeltQwen35VisionCheckpointPlan(
        module: module,
        checkpoint: checkpoint
    )
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw UsageError(message: "Metal device/queue unavailable")
    }

    let loadStart = ContinuousClock.now
    let weights = try SmeltQwen35VisionWeights(
        device: device,
        checkpoint: checkpoint,
        plan: plan
    )
    let pipelines: SmeltQwen35VisionPipelines
    if let artifact {
        pipelines = try SmeltQwen35VisionPipelines(
            device: device,
            library: artifact.makeLibrary(device: device)
        )
    } else {
        pipelines = try SmeltQwen35VisionPipelines(
            device: device,
            shaderDirectory: shaderDirectory
        )
    }
    let runtime = SmeltQwen35VisionRuntime(
        device: device,
        queue: queue,
        config: plan.config,
        weights: weights,
        pipelines: pipelines,
        gemmBackend: gemmBackend,
        attentionBackend: attentionBackend
    )
    let loadSeconds = ContinuousClock.now - loadStart

    let preprocessStart = ContinuousClock.now
    let grid: SmeltQwen35VisionRuntime.Grid
    let patches: [Float]
    let inputDescription: String
    if imagePath.isEmpty {
        grid = SmeltQwen35VisionRuntime.Grid(
            temporal: temporal,
            height: height,
            width: width
        )
        let patchWidth = plan.config.inChannels
            * plan.config.temporalPatchSize
            * plan.config.patchSize
            * plan.config.patchSize
        patches = [Float](repeating: 0, count: grid.patchCount * patchWidth)
        inputDescription = "zero-grid"
    } else {
        let preprocessor = SmeltQwen35ImagePreprocessor(
            config: try SmeltQwen35ImagePreprocessorConfig(module: module)
        )
        let result = try preprocessor.preprocess(imageAt: URL(fileURLWithPath: imagePath))
        grid = result.grid
        patches = result.patches
        inputDescription = "image:\(result.resizedWidth)x\(result.resizedHeight)"
    }
    let basePatches = patches
    let packedPatches = packedSegments == 1
        ? basePatches
        : (0..<packedSegments).flatMap { segment in
            let offset = Float(segment) * 0.03125
            return basePatches.map { $0 + offset }
        }
    let grids = [SmeltQwen35VisionRuntime.Grid](
        repeating: grid,
        count: packedSegments
    )
    let preprocessSeconds = ContinuousClock.now - preprocessStart
    let supportsFrozenCost = gemmBackend == .mps && attentionBackend == .mpsStaged
    let requestedFrozenCost = profileCostIterations > 0
        || !costJSONPath.isEmpty
        || !costCalibrationJSONPath.isEmpty
    if requestedFrozenCost && !supportsFrozenCost {
        throw UsageError(
            message: "frozen vision costing requires --gemm mps --attention mps-staged"
        )
    }
    let frozenPlanProvenance: String
    if let artifact {
        let checksums = artifact.manifest.checksums
        frozenPlanProvenance = "compiled-component"
            + ":schema=\(artifact.manifest.schema)"
            + ":cam=\(checksums.camSHA256)"
            + ":weights=\(checksums.weightsSHA256)"
            + ":metallib=\(checksums.metallibSHA256)"
    } else {
        frozenPlanProvenance = "source-component"
            + ":module=\(sha256Hex(moduleData))"
            + ":source=\(plan.sourceID)"
    }
    let frozenPlan = supportsFrozenCost ? try SmeltQwen35VisionRuntime.frozenPlan(
        config: plan.config,
        grids: grids,
        provenanceKey: frozenPlanProvenance,
        gemmBackend: gemmBackend,
        attentionBackend: attentionBackend
    ) : nil
    var costReport = frozenPlan.map { SmeltFrozenIRCostModel.report(plan: $0) }
    for _ in 0..<warmupIterations {
        _ = try runtime.encode(
            patches: packedPatches,
            grids: grids,
            diagnoseNonFinite: diagnoseNonFinite
        )
    }
    var encodeSamples: [Double] = []
    var workspaceSamples: [Double] = []
    var patchCopySamples: [Double] = []
    var positionSamples: [Double] = []
    var setupSamples: [Double] = []
    var commandEncodingSamples: [Double] = []
    var commandExecutionSamples: [Double] = []
    var gpuSamples: [Double] = []
    encodeSamples.reserveCapacity(measuredIterations)
    var output: SmeltQwen35VisionRuntime.Output?
    for _ in 0..<measuredIterations {
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let measuredOutput = try runtime.encode(
            patches: packedPatches,
            grids: grids,
            diagnoseNonFinite: diagnoseNonFinite
        )
        output = measuredOutput
        encodeSamples.append(CFAbsoluteTimeGetCurrent() - encodeStart)
        workspaceSamples.append(measuredOutput.timing.workspaceSeconds)
        patchCopySamples.append(measuredOutput.timing.patchCopySeconds)
        positionSamples.append(measuredOutput.timing.positionSeconds)
        setupSamples.append(measuredOutput.timing.setupSeconds)
        commandEncodingSamples.append(measuredOutput.timing.commandEncodingSeconds)
        commandExecutionSamples.append(measuredOutput.timing.commandExecutionSeconds)
        gpuSamples.append(measuredOutput.timing.gpuSeconds)
    }
    guard let output else {
        throw UsageError(message: "vision benchmark produced no output")
    }
    let encodeSeconds = median(encodeSamples)
    let values = output.values()
    guard values.allSatisfy(\.isFinite) else {
        throw UsageError(message: "vision output contains non-finite values")
    }
    var costCalibration: SmeltDeviceCostCalibration?
    if profileCostIterations > 0 {
        guard let frozenPlan else {
            throw UsageError(message: "vision route has no frozen cost plan")
        }
        var profiles: [SmeltFrozenIRExecutionProfile] = []
        profiles.reserveCapacity(profileCostIterations)
        for profileIndex in 0..<profileCostIterations {
            let profiledOutput = try runtime.encode(
                patches: packedPatches,
                grids: grids,
                profileFrozenOperations: true
            )
            guard let operationProfile = profiledOutput.operationProfile else {
                throw UsageError(
                    message: "vision cost profile \(profileIndex) produced no operation samples"
                )
            }
            let profiledValues = profiledOutput.values()
            guard profiledValues.count == values.count,
                  zip(profiledValues, values).allSatisfy({
                      $0.0.bitPattern == $0.1.bitPattern
                  })
            else {
                throw UsageError(
                    message: "vision cost profile \(profileIndex) changed output bits"
                )
            }
            profiles.append(SmeltFrozenIRExecutionProfile(
                provenanceKey: frozenPlan.provenanceKey,
                context: frozenPlan.context,
                deviceName: device.name,
                measurementMethod: operationProfile.measurementMethod,
                wholePlanGPUUs: operationProfile.wholePlanGPUUs,
                spans: operationProfile.spans.map {
                    SmeltFrozenIRExecutionSpan(
                        recordIndices: $0.recordIndices,
                        gpuUs: $0.gpuUs
                    )
                }
            ))
        }
        let calibration = try SmeltFrozenIRCostModel.calibration(
            plan: frozenPlan,
            profiles: profiles,
            cleanWholePlanGPUUs: gpuSamples.map { $0 * 1_000_000 },
            hostRecordUs: median(commandEncodingSamples) * 1_000_000
                / Double(max(frozenPlan.records.count, 1))
        )
        costCalibration = calibration
        costReport = SmeltFrozenIRCostModel.report(
            plan: frozenPlan,
            calibration: calibration
        )
    }
    if !costJSONPath.isEmpty {
        guard let costReport else {
            throw UsageError(message: "vision route has no frozen cost report")
        }
        try costReport.encodeJSON().write(
            to: URL(fileURLWithPath: costJSONPath),
            options: .atomic
        )
    }
    if !costCalibrationJSONPath.isEmpty {
        guard let costCalibration else {
            throw UsageError(
                message: "--cost-calibration-json requires --profile-cost"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(costCalibration).write(
            to: URL(fileURLWithPath: costCalibrationJSONPath),
            options: .atomic
        )
    }
    if !outputPath.isEmpty {
        try values.withUnsafeBytes { raw in
            try Data(raw).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        }
    }
    let minimum = values.min() ?? 0
    let maximum = values.max() ?? 0
    let checksum = values.reduce(Float(0), +)
    let prefix = values.prefix(8).map(formatExact).joined(separator: ",")
    var parity = ""
    if !referencePath.isEmpty {
        let data = try Data(contentsOf: URL(fileURLWithPath: referencePath))
        guard data.count == values.count * MemoryLayout<Float>.stride else {
            throw UsageError(
                message: "reference has \(data.count) bytes; expected \(values.count * 4)"
            )
        }
        let reference = data.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        var dot: Double = 0, actualNorm: Double = 0, referenceNorm: Double = 0
        var errorNorm: Double = 0
        var maximumAbsolute: Float = 0
        for (actual, expected) in zip(values, reference) {
            let a = Double(actual), e = Double(expected), d = a - e
            dot += a * e
            actualNorm += a * a
            referenceNorm += e * e
            errorNorm += d * d
            maximumAbsolute = max(maximumAbsolute, abs(actual - expected))
        }
        let cosine = dot / (actualNorm.squareRoot() * referenceNorm.squareRoot())
        let relativeL2 = (errorNorm / referenceNorm).squareRoot()
        parity = " max_abs=\(formatExact(maximumAbsolute))"
            + " rel_l2=\(String(format: "%.8g", relativeL2))"
            + " cosine=\(String(format: "%.8g", cosine))"
    }
    let encodeMedianText = String(format: "%.3f", encodeSeconds)
    let encodeSamplesText = encodeSamples
        .map { String(format: "%.3f", $0) }
        .joined(separator: ",")
    let logicalReadBytes = costReport?.summary.storageTotals.reduce(UInt64(0)) {
        $0 &+ $1.readBytes
    } ?? 0
    let logicalWriteBytes = costReport?.summary.storageTotals.reduce(UInt64(0)) {
        $0 &+ $1.writeBytes
    } ?? 0
    let fp32Operations = costReport?.summary.operationTotals.first {
        $0.operationClass == .fp32Arithmetic
    }?.count ?? 0
    var measuredGroupGPUUs: [String: Double] = [:]
    for record in costReport?.records ?? [] {
        guard let group = record.operationGroup,
              let gpuUs = record.calibratedMedianGPUUs
        else { continue }
        measuredGroupGPUUs[group, default: 0] += gpuUs
    }
    let measuredGroups = measuredGroupGPUUs.sorted {
        if $0.value != $1.value { return $0.value > $1.value }
        return $0.key < $1.key
    }.map {
        "\($0.key):\(String(format: "%.3f", $0.value / 1_000))ms"
    }.joined(separator: ",")
    let instrumentedSpanGPUUs = costCalibration?.instrumentedSpanMedianGPUUs ?? 0
    let reconciliationScale = instrumentedSpanGPUUs > 0
        ? (costCalibration?.wholePlanMedianGPUUs ?? 0) / instrumentedSpanGPUUs
        : 0
    print(
        "vision-run passed"
            + " source=\(modelSource)"
            + " input=\(inputDescription)"
            + " gemm=\(gemmBackendText)"
            + " attention=\(attentionBackendText)"
            + " packed_segments=\(packedSegments)"
            + " grid=\(grid.temporal)x\(grid.height)x\(grid.width)"
            + " output=\(output.tokenCount)x\(output.hiddenSize)"
            + " load_s=\(loadSeconds.components.seconds).\(String(format: "%03d", loadSeconds.components.attoseconds / 1_000_000_000_000_000))"
            + " preprocess_s=\(preprocessSeconds.components.seconds).\(String(format: "%03d", preprocessSeconds.components.attoseconds / 1_000_000_000_000_000))"
            + " encode_s=\(encodeMedianText)"
            + " encode_samples_s=\(encodeSamplesText)"
            + " workspace_s=\(String(format: "%.3f", median(workspaceSamples)))"
            + " patch_copy_s=\(String(format: "%.3f", median(patchCopySamples)))"
            + " position_s=\(String(format: "%.3f", median(positionSamples)))"
            + " setup_s=\(String(format: "%.3f", median(setupSamples)))"
            + " command_encode_s=\(String(format: "%.3f", median(commandEncodingSamples)))"
            + " command_execute_s=\(String(format: "%.3f", median(commandExecutionSamples)))"
            + " gpu_s=\(String(format: "%.3f", median(gpuSamples)))"
            + " cost_dispatches=\(costReport?.summary.dispatchCount ?? 0)"
            + " cost_read_gib=\(String(format: "%.3f", Double(logicalReadBytes) / 1_073_741_824))"
            + " cost_write_gib=\(String(format: "%.3f", Double(logicalWriteBytes) / 1_073_741_824))"
            + " cost_fp32_tops=\(String(format: "%.3f", Double(fp32Operations) / 1.0e12))"
            + " cost_materialized_gib=\(String(format: "%.3f", Double(costReport?.summary.intermediateMaterializationBytes ?? 0) / 1_073_741_824))"
            + " cost_profile_iterations=\(profileCostIterations)"
            + " cost_calibrated_gpu_s=\(String(format: "%.3f", (costReport?.summary.calibratedMedianGPUUs ?? 0) / 1_000_000))"
            + " cost_profiled_gpu_s=\(String(format: "%.3f", (costCalibration?.instrumentedWholePlanMedianGPUUs ?? 0) / 1_000_000))"
            + " cost_reconcile_scale=\(String(format: "%.6f", reconciliationScale))"
            + " cost_additive_error=\(String(format: "%.4f", costReport?.summary.additiveCalibrationErrorFraction ?? 0))"
            + (measuredGroups.isEmpty ? "" : " cost_groups=\(measuredGroups)")
            + " min=\(formatExact(minimum))"
            + " max=\(formatExact(maximum))"
            + " sum=\(formatExact(checksum))"
            + " first=\(prefix)"
            + parity
    )
}
