import Foundation

public enum WhisperTinyReferenceAssets {
    public static func repoRoot(startingAt filePath: String = #filePath) throws -> String {
        var url = URL(fileURLWithPath: filePath)
        let fm = FileManager.default

        while url.path != "/" {
            url = url.deletingLastPathComponent()
            let weights = url.appendingPathComponent("tools/whisper_ref/whisper-tiny.safetensors").path
            if fm.fileExists(atPath: weights) {
                return url.path
            }
        }

        throw ReferenceAssetError.repoRootNotFound
    }

    public static func weightsPath() throws -> String {
        "\(try repoRoot())/tools/whisper_ref/whisper-tiny.safetensors"
    }

    public static func referenceDir() throws -> String {
        "\(try repoRoot())/tools/whisper_ref/dumps/passing-medium"
    }

    public static func loadReference(name: String) throws -> (data: [Float], shape: [Int]) {
        let npy = try NpyLoader.load(path: "\(try referenceDir())/\(name).npy")
        let count = npy.elementCount
        var floats = [Float](repeating: 0, count: count)

        if npy.dtype == "f4" {
            floats.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.initialize(from: npy.fp32Pointer, count: count)
            }
        } else if npy.dtype == "f2" {
            let src = npy.data.bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                floats[i] = Float(src[i])
            }
        } else {
            throw ReferenceAssetError.unsupportedNpyDType(npy.dtype)
        }

        return (floats, npy.shape)
    }
}

public enum ReferenceAssetError: Error, CustomStringConvertible {
    case repoRootNotFound
    case unsupportedNpyDType(String)

    public var description: String {
        switch self {
        case .repoRootNotFound:
            return "Could not find repo root containing tools/whisper_ref"
        case let .unsupportedNpyDType(dtype):
            return "Unsupported reference npy dtype: \(dtype)"
        }
    }
}
