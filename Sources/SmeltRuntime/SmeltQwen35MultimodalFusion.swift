import Foundation
import SmeltSchema

/// Graph-derived contract for Qwen3.5's visual-token replacement and MRoPE
/// seam. The implementation is model-family-specific, while admission is
/// structural: no module id, semantic hash, or package filename participates.
public struct SmeltQwen35MultimodalFusionConfig: Sendable, Equatable {
    public let visionStartTokenID: Int32
    public let visionEndTokenID: Int32
    public let imageTokenID: Int32
    public let videoTokenID: Int32
    public let hiddenSize: Int
    public let spatialMergeSize: Int
    public let ropeTheta: Float
    public let mropeSections: [Int]

    public init(module: SmeltCAMIR) throws {
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let decision = try capabilities.resolve(.runMultimodalText)
        let bindingPlan = try SmeltModuleRuntimeBindingPlan(
            capabilities: capabilities,
            decision: decision
        )
        let graph = bindingPlan.graph

        let tokenizerCandidates = graph.nodes.filter {
            Self.annotation("tag", in: $0.annotations) == "text-tokenizer"
                && Self.annotation("image-token", in: $0.annotations) != nil
        }
        let fusionCandidates = graph.nodes.filter {
            Self.annotation("tag", in: $0.annotations) == "visual-token-fusion"
        }
        let preprocessorCandidates = graph.nodes.filter {
            Self.annotation("tag", in: $0.annotations) == "media-preprocessor"
        }
        guard tokenizerCandidates.count == 1,
              fusionCandidates.count == 1,
              preprocessorCandidates.count == 1,
              let tokenizer = tokenizerCandidates.first,
              let fusion = fusionCandidates.first,
              let preprocessor = preprocessorCandidates.first
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "selected region needs one tokenizer, visual-token-fusion, "
                    + "and media-preprocessor"
            )
        }
        guard Self.annotation("deepstack-visual-indexes", in: fusion.annotations) == "none",
              Self.annotation("mrope-interleaved", in: fusion.annotations) == "true"
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "only replacement fusion with interleaved MRoPE and no deepstack is supported"
            )
        }

        visionStartTokenID = try Self.int32("vision-start-token", in: tokenizer.annotations)
        visionEndTokenID = try Self.int32("vision-end-token", in: tokenizer.annotations)
        imageTokenID = try Self.int32("image-token", in: tokenizer.annotations)
        videoTokenID = try Self.int32("video-token", in: tokenizer.annotations)
        spatialMergeSize = try Self.positiveInt(
            "spatial-merge-size", in: preprocessor.annotations
        )
        guard let sectionText = Self.annotation("mrope-section", in: fusion.annotations) else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "visual-token-fusion is missing mrope-section"
            )
        }
        let sections = sectionText.split(separator: ",").compactMap { Int($0) }
        guard sections.count == 3, sections.allSatisfy({ $0 > 0 }) else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "mrope-section must contain three positive integers"
            )
        }
        mropeSections = sections

        let executableBlockIDs = Set(
            graph.nodes.compactMap { $0.implementation == "compiled" ? $0.blockID : nil }
        )
        let trunks = graph.blocks.filter {
            $0.operatorName == "transformer"
                && executableBlockIDs.contains($0.blockID)
                && $0.shape.transformer?.vocab != nil
        }
        guard trunks.count == 1,
              let transformer = trunks[0].shape.transformer,
              let hidden = transformer.hiddenSize,
              let theta = transformer.attention?.rope?.theta,
              theta > 0
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "selected region needs one executable language transformer with RoPE"
            )
        }
        hiddenSize = hidden
        ropeTheta = Float(theta)
    }

    private static func annotation(
        _ key: String,
        in annotations: [SmeltCAMPackageDescriptor.Requirement]
    ) -> String? {
        let matches = annotations.filter { $0.key == key }
        return matches.count == 1 ? matches[0].value : nil
    }

    private static func positiveInt(
        _ key: String,
        in annotations: [SmeltCAMPackageDescriptor.Requirement]
    ) throws -> Int {
        guard let text = annotation(key, in: annotations),
              let value = Int(text), value > 0
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "missing or invalid positive annotation '\(key)'"
            )
        }
        return value
    }

    private static func int32(
        _ key: String,
        in annotations: [SmeltCAMPackageDescriptor.Requirement]
    ) throws -> Int32 {
        guard let text = annotation(key, in: annotations),
              let value = Int32(text), value >= 0
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "missing or invalid token annotation '\(key)'"
            )
        }
        return value
    }
}

public struct SmeltQwen35MultimodalPositionPlan: Sendable, Equatable {
    public let temporal: [Int]
    public let height: [Int]
    public let width: [Int]
    public let ropeDelta: Int

    public init(
        tokenIDs: [Int32],
        grids: [SmeltQwen35VisionRuntime.Grid],
        config: SmeltQwen35MultimodalFusionConfig
    ) throws {
        guard !tokenIDs.isEmpty else {
            throw SmeltQwen35MultimodalFusionError.invalidInput(
                "position planning requires at least one token"
            )
        }
        var temporal: [Int] = []
        var height: [Int] = []
        var width: [Int] = []
        temporal.reserveCapacity(tokenIDs.count)
        height.reserveCapacity(tokenIDs.count)
        width.reserveCapacity(tokenIDs.count)

        func modality(_ token: Int32) -> Int {
            if token == config.imageTokenID { return 1 }
            if token == config.videoTokenID { return 2 }
            return 0
        }

        var tokenOffset = 0
        var gridOffset = 0
        var currentPosition = 0
        while tokenOffset < tokenIDs.count {
            let type = modality(tokenIDs[tokenOffset])
            var groupEnd = tokenOffset + 1
            while groupEnd < tokenIDs.count,
                  modality(tokenIDs[groupEnd]) == type {
                groupEnd += 1
            }
            let groupLength = groupEnd - tokenOffset
            if type == 0 {
                for position in currentPosition..<(currentPosition + groupLength) {
                    temporal.append(position)
                    height.append(position)
                    width.append(position)
                }
                currentPosition += groupLength
            } else {
                guard gridOffset < grids.count else {
                    throw SmeltQwen35MultimodalFusionError.invalidInput(
                        "visual placeholder group \(gridOffset) has no grid"
                    )
                }
                let grid = grids[gridOffset]
                let merge = config.spatialMergeSize
                guard grid.temporal > 0,
                      grid.height > 0, grid.height % merge == 0,
                      grid.width > 0, grid.width % merge == 0
                else {
                    throw SmeltQwen35MultimodalFusionError.invalidInput(
                        "visual grid \(gridOffset) is incompatible with merge=\(merge)"
                    )
                }
                let mergedHeight = grid.height / merge
                let mergedWidth = grid.width / merge
                let expected = grid.temporal * mergedHeight * mergedWidth
                guard groupLength == expected else {
                    throw SmeltQwen35MultimodalFusionError.invalidInput(
                        "visual placeholder group \(gridOffset) has \(groupLength) "
                            + "tokens; grid requires \(expected)"
                    )
                }

                // Exact Qwen3.5 get_vision_position_ids flattening. Video
                // callers split temporal frames before this point, as the
                // official processor does; still-image grids have temporal=1.
                for index in 0..<expected {
                    temporal.append(currentPosition)
                    height.append(currentPosition + index / (mergedWidth * grid.temporal))
                    width.append(currentPosition + index % mergedWidth)
                }
                currentPosition += max(mergedHeight, mergedWidth)
                gridOffset += 1
            }
            tokenOffset = groupEnd
        }
        guard gridOffset == grids.count else {
            throw SmeltQwen35MultimodalFusionError.invalidInput(
                "received \(grids.count) visual grids but consumed \(gridOffset)"
            )
        }
        guard temporal.count == tokenIDs.count,
              height.count == tokenIDs.count,
              width.count == tokenIDs.count
        else {
            throw SmeltQwen35MultimodalFusionError.invalidInput(
                "position plan length does not match token count"
            )
        }
        self.temporal = temporal
        self.height = height
        self.width = width
        let maximum = max(
            temporal.max() ?? 0,
            max(height.max() ?? 0, width.max() ?? 0)
        )
        ropeDelta = maximum + 1 - tokenIDs.count
    }

    /// Build adjacent-pair fp16 tables for the compiled language package.
    /// The package has already permuted Q/K from the source model's split-half
    /// layout, so each selected source frequency is duplicated adjacently.
    public func ropeTables(
        rowCount: Int,
        ropeDim: Int,
        config: SmeltQwen35MultimodalFusionConfig
    ) throws -> (cos: [Float16], sin: [Float16]) {
        guard rowCount >= temporal.count else {
            throw SmeltQwen35MultimodalFusionError.invalidInput(
                "RoPE row count \(rowCount) is shorter than prompt \(temporal.count)"
            )
        }
        guard ropeDim > 0, ropeDim % 2 == 0,
              config.mropeSections.reduce(0, +) == ropeDim / 2
        else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "mrope sections \(config.mropeSections) do not cover ropeDim=\(ropeDim)"
            )
        }

        var frequencyAxes: [Int] = []
        var remaining = config.mropeSections
        while frequencyAxes.count < ropeDim / 2 {
            var advanced = false
            for axis in 0..<3 where remaining[axis] > 0 {
                frequencyAxes.append(axis)
                remaining[axis] -= 1
                advanced = true
            }
            guard advanced else { break }
        }
        guard frequencyAxes.count == ropeDim / 2 else {
            throw SmeltQwen35MultimodalFusionError.invalidContract(
                "could not interleave all MRoPE frequency sections"
            )
        }

        var cosines = [Float16](repeating: 0, count: rowCount * ropeDim)
        var sines = [Float16](repeating: 0, count: rowCount * ropeDim)
        for row in 0..<rowCount {
            let positions: [Int]
            if row < temporal.count {
                positions = [temporal[row], height[row], width[row]]
            } else {
                let decodePosition = row + ropeDelta
                positions = [decodePosition, decodePosition, decodePosition]
            }
            let rowBase = row * ropeDim
            for frequency in 0..<(ropeDim / 2) {
                let exponent = Float(2 * frequency) / Float(ropeDim)
                let inverseFrequency = Float(
                    pow(Double(config.ropeTheta), Double(-exponent))
                )
                let angle = Float(positions[frequencyAxes[frequency]]) * inverseFrequency
                let cosine = Float16(Float(cos(Double(angle))))
                let sine = Float16(Float(sin(Double(angle))))
                let destination = rowBase + frequency * 2
                cosines[destination] = cosine
                cosines[destination + 1] = cosine
                sines[destination] = sine
                sines[destination + 1] = sine
            }
        }
        return (cosines, sines)
    }
}

public enum SmeltQwen35MultimodalFusion {
    /// Gather exact package token rows, then replace image/video placeholder
    /// rows with encoder output in encounter order.
    public static func embeddings(
        tokenIDs: [Int32],
        visualEmbeddings: [Float],
        runtime: SmeltRuntime,
        config: SmeltQwen35MultimodalFusionConfig
    ) throws -> [Float16] {
        let visualTokenCount = tokenIDs.reduce(into: 0) { count, token in
            if token == config.imageTokenID || token == config.videoTokenID {
                count += 1
            }
        }
        guard visualEmbeddings.count == visualTokenCount * config.hiddenSize else {
            throw SmeltQwen35MultimodalFusionError.invalidInput(
                "visual embeddings contain \(visualEmbeddings.count) values; "
                    + "\(visualTokenCount) placeholders require "
                    + "\(visualTokenCount * config.hiddenSize)"
            )
        }

        var result = [Float16](
            repeating: 0,
            count: tokenIDs.count * config.hiddenSize
        )
        for (row, token) in tokenIDs.enumerated() {
            let data = try runtime.embedToken(token)
            let rowBytes = config.hiddenSize * MemoryLayout<Float16>.stride
            guard data.count == rowBytes else {
                throw SmeltQwen35MultimodalFusionError.invalidInput(
                    "package embedding row has \(data.count) bytes; expected \(rowBytes)"
                )
            }
            _ = result.withUnsafeMutableBytes { destination in
                data.withUnsafeBytes { source in
                    memcpy(
                        destination.baseAddress!.advanced(by: row * rowBytes),
                        source.baseAddress!,
                        rowBytes
                    )
                }
            }
        }

        var visualRow = 0
        for (row, token) in tokenIDs.enumerated()
        where token == config.imageTokenID || token == config.videoTokenID {
            let destination = row * config.hiddenSize
            let source = visualRow * config.hiddenSize
            for column in 0..<config.hiddenSize {
                result[destination + column] = Float16(visualEmbeddings[source + column])
            }
            visualRow += 1
        }
        return result
    }
}

public enum SmeltQwen35MultimodalFusionError: Error, CustomStringConvertible {
    case invalidContract(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .invalidContract(let message):
            return "invalid Qwen3.5 multimodal fusion contract: \(message)"
        case .invalidInput(let message):
            return "invalid Qwen3.5 multimodal fusion input: \(message)"
        }
    }
}
