import CryptoKit
import Foundation

public struct SmeltCAMFeatureAdmission: Codable, Sendable, Equatable {
    public static let currentSchema = "smelt.module.feature_admission"
    public static let preBridgeStage = "pre-bridge-feature-admission"
    private static let packageAssemblyCompileRequirementKeys: Set<String> = [
        "layout",
        "prefill",
        "target",
    ]

    public struct FeatureRequirement: Codable, Sendable, Equatable {
        public let code: String
        public let scope: String
        public let parameters: [String: String]
        public let canonicalID: String
        public let evidence: String

        public init(
            code: String,
            scope: String,
            parameters: [String: String] = [:],
            evidence: String
        ) {
            self.code = code
            self.scope = scope
            self.parameters = parameters
            self.canonicalID = Self.canonicalID(
                code: code,
                scope: scope,
                parameters: parameters
            )
            self.evidence = evidence
        }

        public var diagnostic: String {
            let parameterText = parameters
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            let parameters = parameterText.isEmpty ? "" : " params {\(parameterText)}"
            return "\(code) id \(canonicalID) scope \(scope)\(parameters): \(evidence)"
        }

        public var checkSummary: String {
            let parameterText = parameters
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            return [
                canonicalID,
                code,
                "scope=\(scope)",
                parameterText.isEmpty ? "params=-" : "params={\(parameterText)}",
            ].joined(separator: " ")
        }

        private enum CodingKeys: String, CodingKey {
            case code
            case scope
            case parameters
            case canonicalID
            case evidence
        }

        private struct CanonicalPayload: Codable {
            let schema: String
            let code: String
            let scope: String
            let parameters: [String: String]
        }

        private static func canonicalID(
            code: String,
            scope: String,
            parameters: [String: String]
        ) -> String {
            let payload = CanonicalPayload(
                schema: "smelt.module.feature_obligation",
                code: code,
                scope: scope,
                parameters: parameters
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            guard let data = try? encoder.encode(payload) else {
                preconditionFailure("CAM feature obligation canonical payload is not encodable")
            }
            return SmeltCAMFeatureAdmission.sha256Hex(data)
        }
    }

    public let schema: String
    public let stage: String
    public let descriptorSchema: String
    public let descriptorVersion: Int
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let requiredFeatureSet: [String]
    public let consumedFeatureSet: [String]
    public let unsupportedFeatureSet: [String]
    public let requiredObligationIDs: [String]
    public let consumedObligationIDs: [String]
    public let unsupportedObligationIDs: [String]
    public let requiredObligations: [FeatureRequirement]
    public let unsupportedFeatures: [FeatureRequirement]

    public var hasUnsupportedFeatures: Bool {
        !unsupportedFeatures.isEmpty
    }

    public var unsupportedDiagnostic: String {
        unsupportedFeatures.map(\.diagnostic).joined(separator: "; ")
    }

    public init(
        descriptor: SmeltCAMPackageDescriptor,
        consumedFeatureSet: [String] = [],
        consumedObligationIDs: [String] = []
    ) {
        let consumed = Array(Set(consumedFeatureSet)).sorted()
        let detected = Self.detectRequiredFeatures(in: descriptor)
        let detectedIDSet = Set(detected.map(\.canonicalID))
        let consumedIDs = Array(Set(consumedObligationIDs).intersection(detectedIDSet)).sorted()
        let consumedIDSet = Set(consumedIDs)
        let unsupported = detected
            .filter { !consumedIDSet.contains($0.canonicalID) }
            .sorted(by: Self.featureSort)

        schema = Self.currentSchema
        stage = Self.preBridgeStage
        descriptorSchema = descriptor.descriptorSchema
        descriptorVersion = descriptor.descriptorVersion
        camSemanticSHA256 = descriptor.camSemanticSHA256
        exportABISHA256 = descriptor.exportABISHA256
        requiredFeatureSet = Self.uniqueCodes(in: detected)
        self.consumedFeatureSet = consumed
        unsupportedFeatureSet = Self.uniqueCodes(in: unsupported)
        requiredObligationIDs = detected.map(\.canonicalID).sorted()
        self.consumedObligationIDs = consumedIDs
        unsupportedObligationIDs = unsupported.map(\.canonicalID).sorted()
        requiredObligations = detected.sorted(by: Self.featureSort)
        unsupportedFeatures = unsupported
    }

    private static func detectRequiredFeatures(
        in descriptor: SmeltCAMPackageDescriptor
    ) -> [FeatureRequirement] {
        var features: [FeatureRequirement] = []

        func append(
            _ code: String,
            scope: String,
            parameters: [String: String] = [:],
            evidence: String
        ) {
            features.append(.init(
                code: code,
                scope: scope,
                parameters: parameters,
                evidence: evidence
            ))
        }

        for block in descriptor.blocks {
            guard let transformer = block.shape.transformer else { continue }
            let attentionShapes: [(scope: String, attention: SmeltCAMPackageDescriptor.AttentionShape)] =
                [transformer.attention].compactMap { attention in
                    attention.map { ("block:\(block.blockID)/attention:default", $0) }
                }
                + (transformer.attentionByRole?.map {
                    ("block:\(block.blockID)/attention:\($0.role)", $0.attention)
                } ?? [])
            for (scope, attention) in attentionShapes where attention.rope?.ropeType == "yarn" {
                var parameters = [
                    "q_heads": String(attention.qHeads),
                    "kv_heads": String(attention.kvHeads),
                    "head_dim": String(attention.headDim),
                ]
                if let theta = attention.rope?.theta {
                    parameters["theta"] = String(theta)
                }
                append(
                    "transformer.rope.yarn",
                    scope: scope,
                    parameters: parameters,
                    evidence: "uses Yarn RoPE "
                        + "(theta \(attention.rope?.theta.map(String.init) ?? "unspecified"))"
                )
            }
            if let router = transformer.router {
                append(
                    "transformer.moe.router",
                    scope: "block:\(block.blockID)",
                    parameters: [
                        "top_k": String(router.topK),
                        "experts": String(router.experts),
                    ],
                    evidence: "uses MoE router "
                        + "(top-k \(router.topK), experts \(router.experts))"
                )
            }
            if let expert = transformer.expert {
                var parameters = [
                    "ffn_dim": String(expert.ffn.dim),
                    "activation": expert.ffn.activation,
                ]
                if let experts = transformer.router?.experts {
                    parameters["experts"] = String(experts)
                }
                append(
                    "transformer.moe.expert",
                    scope: "block:\(block.blockID)",
                    parameters: parameters,
                    evidence: "uses MoE expert "
                        + "(expert ffn dim \(expert.ffn.dim), activation \(expert.ffn.activation))"
                )
            }
        }

        for rule in descriptor.quantization where rule.storage?.storageFormat == "gptq" {
            var parameters = quantRuleParameters(rule)
            parameters["storage"] = rule.storage?.storageFormat
            append(
                "quant.storage.gptq",
                scope: quantRuleScope(rule),
                parameters: parameters,
                evidence: "uses GPTQ tensor storage "
                    + "(action \(rule.action), group \(rule.storage?.groupSize.map(String.init) ?? "unspecified"))"
            )
        }

        for rule in descriptor.quantization {
            guard let calibration = rule.calibration, calibration.method == "gptq" else {
                continue
            }
            let captures = calibration.captures.sorted().joined(separator: ",")
            let layers = calibration.layersPerPass.map(String.init) ?? "unspecified"
            let maxTokens = calibration.corpus.maxTokens.map(String.init) ?? "unspecified"
            var parameters = quantRuleParameters(rule)
            parameters["method"] = calibration.method
            parameters["corpus_source"] = calibration.corpus.sourceID
            parameters["corpus_path_sha256"] = calibration.corpus.path.map(sha256Hex) ?? "none"
            parameters["max_tokens"] = maxTokens
            parameters["captures"] = captures
            parameters["layers_per_pass"] = layers
            parameters["requirements"] = calibration.requirements
                .map(comparisonSignature)
                .sorted()
                .joined(separator: ",")
            append(
                "quant.calibration.gptq",
                scope: "\(quantRuleScope(rule))/calibration:gptq",
                parameters: parameters,
                evidence: "declares GPTQ calibration artifacts "
                    + "(max-tokens \(maxTokens), captures \(captures), layers-per-pass \(layers))"
            )
        }

        for requirement in descriptor.compileRequirements where requirement.key == "generated-kernels" {
            append(
                "compile.generated-kernels",
                scope: "compile:\(requirement.key)",
                parameters: [
                    "key": requirement.key,
                    "value": requirement.value,
                ],
                evidence: "requires generated kernels (mode \(requirement.value))"
            )
        }
        for requirement in descriptor.compileRequirements where
            requirement.key == "memory" && requirement.value.contains("peak") {
            append(
                "compile.memory-bound",
                scope: "compile:\(requirement.key)",
                parameters: [
                    "key": requirement.key,
                    "value": requirement.value,
                ],
                evidence: "declares peak memory requirement (\(requirement.value))"
            )
        }
        for requirement in descriptor.compileRequirements where isUnclassifiedCompileRequirement(requirement) {
            append(
                "compile.unclassified",
                scope: "compile:\(requirement.key)",
                parameters: [
                    "key": requirement.key,
                    "value": requirement.value,
                ],
                evidence: "declares unclassified compile requirement "
                    + "(\(requirement.key) \(requirement.value))"
            )
        }

        for gate in descriptor.gateContracts {
            for requirement in gate.requirements where
                requirement.subject == "calibration.gptq.rank"
                    || requirement.subject == "perplexity.delta" {
                var parameters = [
                    "gate_id": gate.gateID,
                    "subject": requirement.subject,
                    "relation": requirement.relation,
                    "value": requirement.value,
                ]
                if let unit = requirement.unit {
                    parameters["unit"] = unit
                }
                append(
                    "gate.quant-quality",
                    scope: "gate:\(gate.gateID)/requirement:\(requirement.subject)",
                    parameters: parameters,
                    evidence: "declares quant-quality gate subjects "
                        + "(\(comparisonSignature(requirement)))"
                )
            }
        }

        return uniqueObligations(features).sorted(by: featureSort)
    }

    private static func isUnclassifiedCompileRequirement(
        _ requirement: SmeltCAMPackageDescriptor.Requirement
    ) -> Bool {
        if packageAssemblyCompileRequirementKeys.contains(requirement.key) {
            return false
        }
        if requirement.key == "generated-kernels" {
            return false
        }
        if requirement.key == "memory", requirement.value.contains("peak") {
            return false
        }
        return true
    }

    private static func quantRuleScope(
        _ rule: SmeltCAMPackageDescriptor.QuantizationRule
    ) -> String {
        "quant:\(rule.action):\(rule.tensorPattern.pattern)"
    }

    private static func quantRuleParameters(
        _ rule: SmeltCAMPackageDescriptor.QuantizationRule
    ) -> [String: String] {
        var parameters = [
            "action": rule.action,
            "pattern": rule.tensorPattern.pattern,
            "resolution": rule.resolution,
        ]
        if let sourceID = rule.sourceID {
            parameters["source_id"] = sourceID
        }
        if let priority = rule.priority {
            parameters["priority"] = String(priority)
        }
        if let storage = rule.storage {
            parameters["storage"] = storage.storageFormat
            parameters["group_size"] = storage.groupSize.map(String.init) ?? "none"
            parameters["compute_dtype"] = storage.computeDType ?? "none"
        }
        return parameters
    }

    private static func comparisonSignature(
        _ comparison: SmeltCAMPackageDescriptor.Comparison
    ) -> String {
        [
            comparison.subject,
            comparison.relation,
            comparison.value,
            comparison.unit ?? "",
        ].joined(separator: ":")
    }

    private static func uniqueCodes(in features: [FeatureRequirement]) -> [String] {
        Array(Set(features.map(\.code))).sorted()
    }

    private static func uniqueObligations(
        _ features: [FeatureRequirement]
    ) -> [FeatureRequirement] {
        var seen = Set<String>()
        var out: [FeatureRequirement] = []
        for feature in features where seen.insert(feature.canonicalID).inserted {
            out.append(feature)
        }
        return out
    }

    private static func featureSort(
        _ lhs: FeatureRequirement,
        _ rhs: FeatureRequirement
    ) -> Bool {
        if lhs.code != rhs.code { return lhs.code < rhs.code }
        if lhs.scope != rhs.scope { return lhs.scope < rhs.scope }
        return lhs.canonicalID < rhs.canonicalID
    }

    private static func sha256Hex(_ value: String) -> String {
        sha256Hex(Data(value.utf8))
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
