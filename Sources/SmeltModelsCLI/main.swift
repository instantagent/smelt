import Foundation
import SmeltModels
import SmeltModuleAuthoring

let usage = """
Usage:
  smelt-models emit --output <dir>   Write <module-id>.module.json for every definition
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("smelt-models: \(message)\n".utf8))
    FileHandle.standardError.write(Data("\(usage)\n".utf8))
    exit(1)
}

let args = CommandLine.arguments
guard args.count >= 2 else { fail("missing subcommand") }

switch args[1] {
case "emit":
    var output: String?
    var idx = 2
    while idx < args.count {
        switch args[idx] {
        case "--output":
            guard idx + 1 < args.count else { fail("--output requires a directory") }
            output = args[idx + 1]
            idx += 2
        default:
            fail("unknown option '\(args[idx])'")
        }
    }
    guard let output else { fail("emit requires --output <dir>") }

    let dir = URL(fileURLWithPath: output, isDirectory: true)
    do {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for definition in SmeltModels.all {
            let data = try definition.canonicalJSONData(prettyPrinted: true)
            let file = dir.appendingPathComponent("\(definition.module.id).module.json")
            var bytes = data
            bytes.append(0x0A)  // trailing newline
            try bytes.write(to: file)
            print("emit\t\(definition.module.id).module.json\t\(data.count) bytes")
        }
    } catch {
        fail("emit failed: \(error)")
    }

case "--help", "-h":
    print(usage)

default:
    fail("unknown subcommand '\(args[1])'")
}
