import Foundation

struct SmeltCopyFileTransform: SmeltGraphNodeRuntime {
  static let registration = SmeltGraphNodeRegistration(
        entrypoint: "artifact.copy",
    make: { SmeltCopyFileTransform() }
    )

    func run(
    inputs: [String: SmeltGraphValue],
    context: SmeltGraphExecutionContext
  ) throws -> [String: SmeltGraphValue] {
    let inputURL = try inputs.required("input", as: URL.self, node: "artifact.copy")
        let data = try Data(contentsOf: inputURL)
    try data.write(to: context.outputURL, options: .atomic)
    context.setSummary("Wrote \(data.count) bytes: \(context.outputURL.path)")
    return ["output": .init(context.outputURL)]
    }
}
