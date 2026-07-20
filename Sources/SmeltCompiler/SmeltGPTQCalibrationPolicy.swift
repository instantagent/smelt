import Foundation
import SmeltSchema

public enum SmeltGPTQCalibrationPolicyError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let message):
            "GPTQ calibration policy: \(message)"
        }
    }
}

public struct SmeltGPTQCalibrationPolicy: Sendable, Equatable {
    public static let qwen3TTSPromptLinesRenderPolicy = "qwen3-tts.prompt-lines"
    public static let activationCaptureRole = "activation-capture"

    public let corpusPath: String
    public let maxSamples: Int?
    public let layersPerPass: Int
    public let captureArtifactPaths: [String]

    public static func qwen3TTSPromptLines(
        fromPackageSpecJSON data: Data,
        defaultLayersPerPass: Int
    ) throws -> SmeltGPTQCalibrationPolicy {
        let spec = try SmeltPackageSpec.decode(from: data)
        return try qwen3TTSPromptLines(from: spec, defaultLayersPerPass: defaultLayersPerPass)
    }

    public static func qwen3TTSPromptLines(
        from spec: SmeltPackageSpec,
        defaultLayersPerPass: Int
    ) throws -> SmeltGPTQCalibrationPolicy {
        try spec.validate()
        guard defaultLayersPerPass > 0 else {
            throw error("default layers-per-pass must be positive")
        }
        guard let quantization = spec.quantization else {
            throw error("package spec has no quantization plan")
        }
        guard quantization.format == .u4 || quantization.format == .gptq else {
            throw error(
                "Qwen3-TTS GPTQ needs u4/gptq quantization, got '\(quantization.format.rawValue)'"
            )
        }
        guard let calibration = quantization.calibration else {
            throw error("package spec has no quantization calibration policy")
        }
        guard let corpus = calibration.corpus else {
            throw error("Qwen3-TTS GPTQ needs a calibration corpus")
        }
        guard corpus.renderPolicy == qwen3TTSPromptLinesRenderPolicy else {
            throw error(
                "unsupported Qwen3-TTS calibration render_policy '\(corpus.renderPolicy)'"
            )
        }
        if corpus.maxTokens != nil {
            throw error("Qwen3-TTS prompt-line calibration cannot enforce max_tokens")
        }
        if calibration.resourceBounds?.maxBytes != nil {
            throw error("Qwen3-TTS prompt-line calibration cannot enforce max_bytes")
        }
        if !calibration.sideInputs.isEmpty {
            throw error("Qwen3-TTS prompt-line calibration cannot consume side_inputs")
        }
        if !calibration.stagedPackages.isEmpty {
            throw error("Qwen3-TTS prompt-line calibration cannot consume staged_packages")
        }
        if !calibration.equalityGates.isEmpty {
            throw error("Qwen3-TTS prompt-line calibration cannot run equality_gates")
        }
        let captures = calibration.captureArtifacts.filter {
            $0.role == activationCaptureRole
        }
        guard !captures.isEmpty else {
            throw error(
                "Qwen3-TTS prompt-line calibration needs at least one "
                    + "capture_artifact with role '\(activationCaptureRole)'"
            )
        }
        let unsupportedCaptures = calibration.captureArtifacts.filter {
            $0.role != activationCaptureRole
        }
        guard unsupportedCaptures.isEmpty else {
            throw error(
                "unsupported Qwen3-TTS capture_artifact roles: "
                    + unsupportedCaptures.map(\.role).sorted().joined(separator: ", ")
            )
        }

        let maxLayers = calibration.resourceBounds?.maxLayersPerPass
        let layersPerPass = min(defaultLayersPerPass, maxLayers ?? defaultLayersPerPass)
        guard layersPerPass > 0 else {
            throw error("resolved layers-per-pass must be positive")
        }

        return SmeltGPTQCalibrationPolicy(
            corpusPath: corpus.path,
            maxSamples: corpus.maxSamples,
            layersPerPass: layersPerPass,
            captureArtifactPaths: captures.map(\.path).sorted()
        )
    }

    private static func error(_ message: String) -> SmeltGPTQCalibrationPolicyError {
        .malformed(message)
    }
}

public struct SmeltRuntimeGPTQCalibrationPolicy: Sendable, Equatable {
    public static let textTokenIDLinesRenderPolicy = "text.token-id-lines"
    public static let activationCaptureRole = "activation-capture"

    public let tokenIDsPath: String
    public let capturePointsPath: String
    public let maxSamples: Int?
    public let maxTokens: Int?
    public let layersPerPass: Int

    public static func textTokenIDLines(
        fromPackageSpecJSON data: Data,
        defaultLayersPerPass: Int
    ) throws -> SmeltRuntimeGPTQCalibrationPolicy {
        let spec = try SmeltPackageSpec.decode(from: data)
        return try textTokenIDLines(from: spec, defaultLayersPerPass: defaultLayersPerPass)
    }

    public static func textTokenIDLines(
        from spec: SmeltPackageSpec,
        defaultLayersPerPass: Int
    ) throws -> SmeltRuntimeGPTQCalibrationPolicy {
        try spec.validate()
        guard defaultLayersPerPass > 0 else {
            throw error("default layers-per-pass must be positive")
        }
        guard (try? SmeltRuntimeGraphPolicy.resolve(blocks: spec.blocks)) == .textGeneration else {
            throw error("expected text-generation graph for runtime GPTQ calibration")
        }
        guard let quantization = spec.quantization else {
            throw error("package spec has no quantization plan")
        }
        guard quantization.format == .u4 || quantization.format == .gptq else {
            throw error(
                "runtime GPTQ needs u4/gptq quantization, got '\(quantization.format.rawValue)'"
            )
        }
        guard let calibration = quantization.calibration else {
            throw error("package spec has no quantization calibration policy")
        }
        guard let corpus = calibration.corpus else {
            throw error("runtime GPTQ needs a token-id calibration corpus")
        }
        guard corpus.renderPolicy == textTokenIDLinesRenderPolicy else {
            throw error(
                "unsupported runtime GPTQ calibration render_policy '\(corpus.renderPolicy)'"
            )
        }
        if !calibration.sideInputs.isEmpty {
            throw error("runtime GPTQ token-id calibration cannot consume side_inputs")
        }
        if !calibration.stagedPackages.isEmpty {
            throw error("runtime GPTQ token-id calibration cannot consume staged_packages")
        }
        if calibration.resourceBounds?.maxBytes != nil {
            throw error("runtime GPTQ token-id calibration cannot enforce max_bytes")
        }
        if !calibration.equalityGates.isEmpty {
            throw error("runtime GPTQ token-id calibration cannot run equality_gates")
        }

        let captures = calibration.captureArtifacts.filter {
            $0.role == activationCaptureRole
        }
        guard captures.count == 1, let capture = captures.first else {
            throw error(
                "expected exactly one capture_artifact with role '\(activationCaptureRole)'"
            )
        }
        let unsupportedCaptures = calibration.captureArtifacts.filter {
            $0.role != activationCaptureRole
        }
        guard unsupportedCaptures.isEmpty else {
            throw error(
                "unsupported capture_artifact roles: "
                    + unsupportedCaptures.map(\.role).sorted().joined(separator: ", ")
            )
        }

        let maxLayers = calibration.resourceBounds?.maxLayersPerPass
        let layersPerPass = min(defaultLayersPerPass, maxLayers ?? defaultLayersPerPass)
        guard layersPerPass > 0 else {
            throw error("resolved layers-per-pass must be positive")
        }

        return SmeltRuntimeGPTQCalibrationPolicy(
            tokenIDsPath: corpus.path,
            capturePointsPath: capture.path,
            maxSamples: cap(corpus.maxSamples, calibration.resourceBounds?.maxSamples),
            maxTokens: cap(corpus.maxTokens, calibration.resourceBounds?.maxTokens),
            layersPerPass: layersPerPass
        )
    }

    public static func tokenIDLinesFromPackageSpecJSON(
        at policyPath: String,
        defaultLayersPerPass: Int
    ) throws -> (policy: SmeltRuntimeGPTQCalibrationPolicy, tokenIDs: [[Int32]]) {
        let data = try Data(contentsOf: URL(fileURLWithPath: policyPath))
        let policy = try textTokenIDLines(
            fromPackageSpecJSON: data,
            defaultLayersPerPass: defaultLayersPerPass
        )
        let corpusPath = resolvePolicyRelativePath(policy.tokenIDsPath, policyPath: policyPath)
        let text = try String(
            contentsOf: URL(fileURLWithPath: corpusPath),
            encoding: .utf8
        )
        let tokenIDs = try parseTokenIDLines(
            text,
            path: corpusPath,
            maxSamples: policy.maxSamples,
            maxTokens: policy.maxTokens
        )
        return (policy, tokenIDs)
    }

    private static func parseTokenIDLines(
        _ text: String,
        path: String,
        maxSamples: Int?,
        maxTokens: Int?
    ) throws -> [[Int32]] {
        var sequences: [[Int32]] = []
        for (lineIndex, line) in text.split(whereSeparator: \.isNewline).enumerated() {
            let fields = line.split {
                $0 == " " || $0 == "\t" || $0 == ","
            }
            if fields.isEmpty { continue }
            var ids: [Int32] = []
            for field in fields {
                guard let parsed = Int64(field),
                      parsed >= 0,
                      parsed <= Int64(Int32.max) else {
                    throw error(
                        "\(path): line \(lineIndex + 1) has invalid token id '\(field)'"
                    )
                }
                ids.append(Int32(parsed))
            }
            if let maxTokens {
                ids = Array(ids.prefix(maxTokens))
            }
            if !ids.isEmpty {
                sequences.append(ids)
            }
            if let maxSamples, sequences.count >= maxSamples {
                break
            }
        }
        guard !sequences.isEmpty else {
            throw error("\(path): token-id calibration corpus is empty")
        }
        return sequences
    }

    private static func resolvePolicyRelativePath(
        _ path: String,
        policyPath: String
    ) -> String {
        if path.hasPrefix("/") {
            return path
        }
        return URL(fileURLWithPath: policyPath)
            .deletingLastPathComponent()
            .appendingPathComponent(path)
            .standardizedFileURL
            .path
    }

    private static func cap(_ first: Int?, _ second: Int?) -> Int? {
        switch (first, second) {
        case let (.some(first), .some(second)):
            return min(first, second)
        case let (.some(first), .none):
            return first
        case let (.none, .some(second)):
            return second
        case (.none, .none):
            return nil
        }
    }

    private static func error(_ message: String) -> SmeltGPTQCalibrationPolicyError {
        .malformed(message)
    }
}

public enum SmeltLLMImatrixCalibrationPolicyError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let message):
            "LLM imatrix calibration policy: \(message)"
        }
    }
}

public struct SmeltLLMImatrixCalibrationPolicy: Sendable, Equatable {
    public static let imatrixRole = "imatrix"

    public let imatrixPath: String

    public static func fromPackageSpecJSON(_ data: Data) throws -> SmeltLLMImatrixCalibrationPolicy {
        let spec = try SmeltPackageSpec.decode(from: data)
        return try from(spec)
    }

    public static func from(_ spec: SmeltPackageSpec) throws -> SmeltLLMImatrixCalibrationPolicy {
        try spec.validate()
        guard (try? SmeltRuntimeGraphPolicy.resolve(blocks: spec.blocks)) == .textGeneration else {
            throw error("expected text-generation graph for LLM imatrix side input")
        }
        guard let quantization = spec.quantization else {
            throw error("package spec has no quantization plan")
        }
        guard quantization.format == .u4 || quantization.format == .turboQuantH else {
            throw error(
                "LLM imatrix side input needs u4/turbo_quant_h quantization, got "
                    + "'\(quantization.format.rawValue)'"
            )
        }
        guard let calibration = quantization.calibration else {
            throw error("package spec has no quantization calibration policy")
        }
        if calibration.corpus != nil {
            throw error("LLM imatrix side-input build cannot consume a calibration corpus")
        }
        if !calibration.captureArtifacts.isEmpty {
            throw error("LLM imatrix side-input build cannot consume capture_artifacts")
        }
        if !calibration.stagedPackages.isEmpty {
            throw error("LLM imatrix side-input build cannot consume staged_packages")
        }
        if calibration.resourceBounds != nil {
            throw error("LLM imatrix side-input build cannot enforce resource_bounds")
        }
        if !calibration.equalityGates.isEmpty {
            throw error("LLM imatrix side-input build cannot run equality_gates")
        }

        let imatrixInputs = calibration.sideInputs.filter { $0.role == imatrixRole }
        guard imatrixInputs.count == 1, let imatrix = imatrixInputs.first else {
            throw error("expected exactly one side_input with role '\(imatrixRole)'")
        }
        let unsupported = calibration.sideInputs.filter { $0.role != imatrixRole }
        guard unsupported.isEmpty else {
            throw error(
                "unsupported side_input roles: "
                    + unsupported.map(\.role).sorted().joined(separator: ", ")
            )
        }

        return SmeltLLMImatrixCalibrationPolicy(imatrixPath: imatrix.path)
    }

    private static func error(_ message: String) -> SmeltLLMImatrixCalibrationPolicyError {
        .malformed(message)
    }
}
