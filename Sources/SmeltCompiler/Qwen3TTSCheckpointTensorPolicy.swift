import Foundation
import SmeltSchema

public struct Qwen3TTSCheckpointTensorPolicy: Sendable {
    public enum SourceDType: Sendable, Equatable {
        case f32
        case bf16
    }

    public struct PlannedTensor: Sendable, Equatable {
        public let name: String
        public let block: String
        public let sourceDType: SourceDType

        public init(name: String, block: String, sourceDType: SourceDType) {
            self.name = name
            self.block = block
            self.sourceDType = sourceDType
        }
    }

    public struct TensorRule: Sendable, Equatable {
        public let source: String
        public let selector: String
        public let block: String
        public let sourceDType: SourceDType

        public init(source: String, selector: String, block: String, sourceDType: SourceDType) {
            self.source = source
            self.selector = selector
            self.block = block
            self.sourceDType = sourceDType
        }
    }

    public struct RequiredPattern: Sendable, Equatable {
        public let block: String
        public let selector: String
        public let reason: String

        public init(block: String, selector: String, reason: String) {
            self.block = block
            self.selector = selector
            self.reason = reason
        }
    }

    public let sourceID: String
    public let requiredBlocks: Set<String>
    public let rules: [TensorRule]
    public let requiredPatterns: [RequiredPattern]

    public init(cam: SmeltCAMIR, sourceID requestedSourceID: String? = nil) throws {
        let cam = try cam.validated()
        guard !cam.tensors.isEmpty else {
            throw Error.noTensorMaps(cam.module.id)
        }
        let sourceID: String
        if let requestedSourceID {
            sourceID = requestedSourceID
        } else {
            let sources = Set(cam.tensors.map(\.source))
            guard sources.count == 1, let only = sources.first else {
                throw Error.ambiguousTensorSources(sources.sorted())
            }
            sourceID = only
        }
        let blockByID = Dictionary(uniqueKeysWithValues: cam.blocks.map { ($0.id, $0) })
        let sourceDTypeByBlock = try Dictionary(uniqueKeysWithValues: cam.blocks.map { block in
            (block.id, try Self.sourceDType(for: block))
        })
        let rules = try cam.tensors
            .filter { $0.source == sourceID }
            .map { tensor -> TensorRule in
                guard tensor.selector.source == nil || tensor.selector.source == sourceID else {
                    throw Error.tensorSourceMismatch(
                        selector: tensor.selector.pattern,
                        declared: tensor.source,
                        selectorSource: tensor.selector.source!
                    )
                }
                guard blockByID[tensor.target.block] != nil else {
                    throw Error.unknownBlock(tensor.target.block)
                }
                guard let sourceDType = sourceDTypeByBlock[tensor.target.block] else {
                    throw Error.missingSourceDType(tensor.target.block)
                }
                return TensorRule(
                    source: tensor.source,
                    selector: tensor.selector.pattern,
                    block: tensor.target.block,
                    sourceDType: sourceDType
                )
            }
        guard !rules.isEmpty else {
            throw Error.noTensorMaps(sourceID)
        }
        self.sourceID = sourceID
        self.rules = rules.sorted { "\($0.block):\($0.selector)" < "\($1.block):\($1.selector)" }
        self.requiredPatterns = Self.requiredPatterns(from: cam.blocks, rules: rules)
        self.requiredBlocks = Set(requiredPatterns.map(\.block))
    }

    public init(sourceID: String, rules: [TensorRule], requiredBlocks: Set<String>) throws {
        guard !rules.isEmpty else {
            throw Error.noTensorMaps(sourceID)
        }
        let mismatchedSources = Set(rules.map(\.source)).subtracting([sourceID])
        guard mismatchedSources.isEmpty else {
            throw Error.ambiguousTensorSources(([sourceID] + mismatchedSources).sorted())
        }
        self.sourceID = sourceID
        self.rules = rules.sorted { "\($0.block):\($0.selector)" < "\($1.block):\($1.selector)" }
        self.requiredBlocks = requiredBlocks
        self.requiredPatterns = Self.uniqueRequiredPatterns(
            rules.map { RequiredPattern(block: $0.block, selector: $0.selector, reason: "tensor-map") }
        )
    }

    public func tensor(named name: String) throws -> PlannedTensor? {
        let matches = rules.filter { Self.globPattern($0.selector, matches: name) }
        guard matches.count <= 1 else {
            throw Error.ambiguousTensorMap(
                name: name,
                selectors: matches.map(\.selector).sorted()
            )
        }
        guard let match = matches.first else { return nil }
        return PlannedTensor(name: name, block: match.block, sourceDType: match.sourceDType)
    }

    public func unmatchedRequiredPatterns<S: Sequence>(
        in tensorNames: S
    ) -> [RequiredPattern] where S.Element == String {
        let names = Array(tensorNames)
        return requiredPatterns.filter { required in
            !names.contains { Self.globPattern(required.selector, matches: $0) }
        }
    }

    private static func requiredPatterns(
        from blocks: [SmeltCAMIR.Block],
        rules: [TensorRule]
    ) -> [RequiredPattern] {
        let tensorMaps = rules.map {
            RequiredPattern(block: $0.block, selector: $0.selector, reason: "tensor-map")
        }
        let evidence = blocks.flatMap { block in
            block.shape.requirements.compactMap { requirement -> RequiredPattern? in
                guard requirement.key == "tensor-evidence", let selector = requirement.value else {
                    return nil
                }
                return RequiredPattern(block: block.id, selector: selector, reason: "tensor-evidence")
            }
        }
        return uniqueRequiredPatterns(tensorMaps + evidence)
    }

    private static func uniqueRequiredPatterns(_ patterns: [RequiredPattern]) -> [RequiredPattern] {
        var seen = Set<String>()
        var result: [RequiredPattern] = []
        for pattern in patterns.sorted(by: {
            "\($0.block):\($0.selector):\($0.reason)" < "\($1.block):\($1.selector):\($1.reason)"
        }) {
            let key = "\(pattern.block):\(pattern.selector):\(pattern.reason)"
            guard seen.insert(key).inserted else { continue }
            result.append(pattern)
        }
        return result
    }

    private static func sourceDType(for block: SmeltCAMIR.Block) throws -> SourceDType {
        let requirements = block.shape.requirements.filter { $0.key == "source-dtype" }
        guard requirements.count == 1, let raw = requirements.first?.value else {
            throw Error.missingSourceDType(block.id)
        }
        switch raw {
        case "f32":
            return .f32
        case "bf16":
            return .bf16
        default:
            throw Error.unsupportedSourceDType(block: block.id, value: raw)
        }
    }

    private static func globPattern(_ pattern: String, matches value: String) -> Bool {
        if pattern == "*" { return true }
        if !pattern.contains("*") { return pattern == value }
        let parts = pattern.split(separator: "*", omittingEmptySubsequences: false).map(String.init)
        var remainder = value[...]
        if let first = parts.first, !first.isEmpty {
            guard remainder.hasPrefix(first) else { return false }
            remainder.removeFirst(first.count)
        }
        for part in parts.dropFirst().dropLast() where !part.isEmpty {
            guard let range = remainder.range(of: part) else { return false }
            remainder = remainder[range.upperBound...]
        }
        if let last = parts.last, !last.isEmpty {
            return remainder.hasSuffix(last)
        }
        return true
    }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case noTensorMaps(String)
        case ambiguousTensorSources([String])
        case tensorSourceMismatch(selector: String, declared: String, selectorSource: String)
        case unknownBlock(String)
        case missingSourceDType(String)
        case unsupportedSourceDType(block: String, value: String)
        case ambiguousTensorMap(name: String, selectors: [String])

        public var description: String {
            switch self {
            case let .noTensorMaps(source):
                return "Qwen3-TTS CAM checkpoint policy has no tensor maps for \(source)"
            case let .ambiguousTensorSources(sources):
                return "Qwen3-TTS CAM checkpoint policy has ambiguous tensor sources: \(sources.joined(separator: ","))"
            case let .tensorSourceMismatch(selector, declared, selectorSource):
                return "Qwen3-TTS CAM tensor \(selector) declares source \(declared) but selector uses \(selectorSource)"
            case let .unknownBlock(block):
                return "Qwen3-TTS CAM checkpoint policy references unknown block \(block)"
            case let .missingSourceDType(block):
                return "Qwen3-TTS CAM block \(block) must declare exactly one source-dtype"
            case let .unsupportedSourceDType(block, value):
                return "Qwen3-TTS CAM block \(block) source-dtype \(value) is unsupported"
            case let .ambiguousTensorMap(name, selectors):
                return "Qwen3-TTS checkpoint tensor \(name) matches multiple CAM tensor maps: \(selectors.joined(separator: ","))"
            }
        }
    }
}
