import Foundation

private let shaderSourceRootCandidates = [
    "Resources/Shaders",
    "../Resources/Shaders",
]

func loadMetalShaderSource(_ filename: String) -> String? {
    for root in shaderSourceRootCandidates {
        let path = URL(fileURLWithPath: root).appendingPathComponent(filename).path
        guard FileManager.default.fileExists(atPath: path) else { continue }
        var seen = Set<String>()
        return try? resolveMetalShaderSource(at: path, seen: &seen)
    }
    return nil
}

private func resolveMetalShaderSource(
    at path: String,
    seen: inout Set<String>
) throws -> String {
    let normalizedPath = URL(fileURLWithPath: path).standardizedFileURL.path
    if seen.contains(normalizedPath) {
        return ""
    }
    seen.insert(normalizedPath)

    let source = try String(contentsOfFile: normalizedPath, encoding: .utf8)
    let baseDir = URL(fileURLWithPath: normalizedPath).deletingLastPathComponent()
    var expanded = ""

    for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
        let line = String(rawLine)
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed == "#pragma once" {
            continue
        }

        if let includeName = parseQuotedInclude(from: trimmed) {
            let includeURL = baseDir.appendingPathComponent(includeName)
            if FileManager.default.fileExists(atPath: includeURL.path),
               let includeSource = try? resolveMetalShaderSource(at: includeURL.path, seen: &seen)
            {
                expanded += includeSource
                if !includeSource.hasSuffix("\n") {
                    expanded += "\n"
                }
                continue
            }
        }

        expanded += line
        expanded += "\n"
    }

    return expanded
}

private func parseQuotedInclude(from line: String) -> String? {
    guard line.hasPrefix("#include \""), line.hasSuffix("\"") else { return nil }
    let start = line.index(line.startIndex, offsetBy: 10)
    let end = line.index(before: line.endIndex)
    return String(line[start..<end])
}
