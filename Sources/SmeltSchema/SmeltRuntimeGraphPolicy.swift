import Foundation

public enum SmeltRuntimeGraphPolicy: String, Codable, Sendable, Equatable {
    case textGeneration = "text-generation"
    case sidecarTextToCodecAudio = "sidecar-text-to-codec-audio"
    case codecAudio = "codec-audio"

    public enum ResolveError: Error, CustomStringConvertible, Equatable {
        case missingGraph
        case invalidGraph(String)
        case unrecognizedTextToAudio
        case unsupportedSignature(String)

        public var description: String {
            switch self {
            case .missingGraph:
                return "runtime manifest has no block graph"
            case .invalidGraph(let reason):
                return "runtime graph is invalid: \(reason)"
            case .unrecognizedTextToAudio:
                return "text-to-audio graph has no recognized audio assembly policy"
            case .unsupportedSignature(let signature):
                return "runtime graph signature \(signature) has no resolved policy"
            }
        }
    }

    public static func resolve(blocks: SmeltBlockGraph) throws -> Self {
        guard blocks.version == 1 else {
            throw ResolveError.unsupportedSignature(
                "block graph version \(blocks.version) (package from a newer smelt?)"
            )
        }
        do {
            try blocks.validate()
        } catch {
            throw ResolveError.invalidGraph(String(describing: error))
        }
        guard let signature = blocks.signature else {
            throw ResolveError.invalidGraph("runtime graph has no external signature")
        }
        switch (signature.input, signature.output) {
        case (.text, .text):
            return .textGeneration
        case (.codecFrames, .audio):
            return .codecAudio
        case (.text, .audio):
            if blocks.blocks.contains(where: {
                $0.output == .codecFrames
                    || $0.feedback == .embeddings
                    || $0.compiledDelivery == .internalSidecar
            }) {
                return .sidecarTextToCodecAudio
            }
            throw ResolveError.unrecognizedTextToAudio
        default:
            throw ResolveError.unsupportedSignature(
                "\(signature.input.rawValue)->\(signature.output.rawValue)"
            )
        }
    }

    public static func resolve(manifestData: Data) throws -> Self {
        let header = try JSONDecoder().decode(ManifestHeader.self, from: manifestData)
        guard let blocks = header.blocks else {
            throw ResolveError.missingGraph
        }
        return try resolve(blocks: blocks)
    }

    private struct ManifestHeader: Decodable {
        let blocks: SmeltBlockGraph?
    }
}
