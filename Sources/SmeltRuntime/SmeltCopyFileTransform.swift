import Foundation

struct SmeltCopyFileTransform: SmeltFileTransformRuntime {
    static let registration = SmeltFileTransformRegistration(
        entrypoint: "artifact.copy",
        make: { _ in SmeltCopyFileTransform() }
    )

    func run(
        inputURL: URL,
        outputURL: URL,
        options _: [String: String]
    ) throws -> SmeltPackageRunResult {
        let data = try Data(contentsOf: inputURL)
        try data.write(to: outputURL, options: .atomic)
        return SmeltPackageRunResult(
            outputURL: outputURL,
            summary: "Wrote \(data.count) bytes: \(outputURL.path)"
        )
    }
}
