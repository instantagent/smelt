import CoreGraphics
import Foundation
import ImageIO
import SmeltSchema

public struct SmeltQwen35ImagePreprocessorConfig: Sendable, Equatable {
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let minimumPixels: Int
    public let maximumPixels: Int
    public let imageMean: [Float]
    public let imageStandardDeviation: [Float]
    public let resample: String

    public init(module: SmeltCAMIR) throws {
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let decision = try capabilities.resolve(.runMultimodalText)
        let plan = try SmeltModuleRuntimeBindingPlan(
            capabilities: capabilities,
            decision: decision
        )
        let candidates = plan.graph.nodes.filter { node in
            node.implementation == "native"
                && node.annotations.contains {
                    $0.key == "tag" && $0.value == "media-preprocessor"
                }
        }
        guard candidates.count == 1, let node = candidates.first else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "selected region needs one native media-preprocessor"
            )
        }
        let annotations = try Self.annotationMap(node.annotations)
        patchSize = try Self.positiveInt("patch-size", annotations)
        temporalPatchSize = try Self.positiveInt("temporal-patch-size", annotations)
        spatialMergeSize = try Self.positiveInt("spatial-merge-size", annotations)
        minimumPixels = try Self.positiveInt("min-pixels", annotations)
        maximumPixels = try Self.positiveInt("max-pixels", annotations)
        imageMean = try Self.floatList("image-mean", count: 3, annotations)
        imageStandardDeviation = try Self.floatList("image-std", count: 3, annotations)
        guard let resample = annotations["resample"], resample == "bicubic" else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "media-preprocessor must declare bicubic resampling"
            )
        }
        self.resample = resample
        guard minimumPixels <= maximumPixels,
              imageStandardDeviation.allSatisfy({ $0 > 0 })
        else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "media-preprocessor pixel bounds or standard deviation are invalid"
            )
        }
    }

    private static func annotationMap(
        _ annotations: [SmeltCAMPackageDescriptor.Requirement]
    ) throws -> [String: String] {
        var result: [String: String] = [:]
        for annotation in annotations {
            guard result[annotation.key] == nil else {
                throw SmeltQwen35ImagePreprocessorError.invalidContract(
                    "duplicate media-preprocessor annotation '\(annotation.key)'"
                )
            }
            result[annotation.key] = annotation.value
        }
        return result
    }

    private static func positiveInt(
        _ key: String,
        _ values: [String: String]
    ) throws -> Int {
        guard let text = values[key], let value = Int(text), value > 0 else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "missing or invalid media-preprocessor annotation '\(key)'"
            )
        }
        return value
    }

    private static func floatList(
        _ key: String,
        count: Int,
        _ values: [String: String]
    ) throws -> [Float] {
        guard let text = values[key] else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "missing media-preprocessor annotation '\(key)'"
            )
        }
        let result = text.split(separator: ",").compactMap { Float($0) }
        guard result.count == count, result.allSatisfy(\.isFinite) else {
            throw SmeltQwen35ImagePreprocessorError.invalidContract(
                "media-preprocessor annotation '\(key)' is invalid"
            )
        }
        return result
    }
}
public struct SmeltQwen35ImagePreprocessor {
    public struct Result {
        public let patches: [Float]
        public let grid: SmeltQwen35VisionRuntime.Grid
        public let resizedWidth: Int
        public let resizedHeight: Int
    }

    public let config: SmeltQwen35ImagePreprocessorConfig

    public init(config: SmeltQwen35ImagePreprocessorConfig) {
        self.config = config
    }

    public func preprocess(imageAt url: URL) throws -> Result {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw SmeltQwen35ImagePreprocessorError.imageDecodeFailed(url.path)
        }
        return try preprocess(image: image)
    }

    public func preprocess(image: CGImage) throws -> Result {
        let (height, width) = try smartResize(height: image.height, width: image.width)
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rgba,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else {
            throw SmeltQwen35ImagePreprocessorError.bitmapContextFailed(width, height)
        }
        context.interpolationQuality = .high
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let gridHeight = height / config.patchSize
        let gridWidth = width / config.patchSize
        let grid = SmeltQwen35VisionRuntime.Grid(
            temporal: 1,
            height: gridHeight,
            width: gridWidth
        )
        let patchWidth = 3
            * config.temporalPatchSize
            * config.patchSize
            * config.patchSize
        var patches: [Float] = []
        patches.reserveCapacity(grid.patchCount * patchWidth)
        let merge = config.spatialMergeSize
        for blockRow in 0..<(gridHeight / merge) {
            for blockColumn in 0..<(gridWidth / merge) {
                for innerRow in 0..<merge {
                    for innerColumn in 0..<merge {
                        let patchRow = blockRow * merge + innerRow
                        let patchColumn = blockColumn * merge + innerColumn
                        for channel in 0..<3 {
                            for _ in 0..<config.temporalPatchSize {
                                for row in 0..<config.patchSize {
                                    let pixelRow = patchRow * config.patchSize + row
                                    for column in 0..<config.patchSize {
                                        let pixelColumn = patchColumn * config.patchSize + column
                                        let byteIndex = pixelRow * bytesPerRow + pixelColumn * 4 + channel
                                        let scaled = Float(rgba[byteIndex]) / 255
                                        patches.append(
                                            (scaled - config.imageMean[channel])
                                                / config.imageStandardDeviation[channel]
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Result(
            patches: patches,
            grid: grid,
            resizedWidth: width,
            resizedHeight: height
        )
    }

    public func smartResize(height: Int, width: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw SmeltQwen35ImagePreprocessorError.invalidImageDimensions(width, height)
        }
        let ratio = Double(max(height, width)) / Double(min(height, width))
        guard ratio <= 200 else {
            throw SmeltQwen35ImagePreprocessorError.aspectRatioTooLarge(ratio)
        }
        let factor = config.patchSize * config.spatialMergeSize
        var resizedHeight = Int(
            (Double(height) / Double(factor)).rounded(.toNearestOrEven)
        ) * factor
        var resizedWidth = Int(
            (Double(width) / Double(factor)).rounded(.toNearestOrEven)
        ) * factor
        if resizedHeight * resizedWidth > config.maximumPixels {
            let beta = sqrt(Double(height * width) / Double(config.maximumPixels))
            resizedHeight = max(
                factor,
                Int(floor(Double(height) / beta / Double(factor))) * factor
            )
            resizedWidth = max(
                factor,
                Int(floor(Double(width) / beta / Double(factor))) * factor
            )
        } else if resizedHeight * resizedWidth < config.minimumPixels {
            let beta = sqrt(Double(config.minimumPixels) / Double(height * width))
            resizedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            resizedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }
        return (resizedHeight, resizedWidth)
    }
}

public enum SmeltQwen35ImagePreprocessorError: Error, CustomStringConvertible {
    case invalidContract(String)
    case imageDecodeFailed(String)
    case invalidImageDimensions(Int, Int)
    case aspectRatioTooLarge(Double)
    case bitmapContextFailed(Int, Int)

    public var description: String {
        switch self {
        case .invalidContract(let reason):
            return "Qwen3.5 image preprocessor contract: \(reason)"
        case .imageDecodeFailed(let path):
            return "Qwen3.5 image preprocessor could not decode '\(path)'"
        case .invalidImageDimensions(let width, let height):
            return "Qwen3.5 image preprocessor invalid dimensions \(width)x\(height)"
        case .aspectRatioTooLarge(let ratio):
            return "Qwen3.5 image preprocessor aspect ratio \(ratio) exceeds 200"
        case .bitmapContextFailed(let width, let height):
            return "Qwen3.5 image preprocessor could not allocate \(width)x\(height) RGB bitmap"
        }
    }
}
