import Foundation
import Testing

@Test func repositoryContainsNoMergeConflictMarkers() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let root = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let roots = ["Sources", "Tests", "Models", "tools"]
    let leftMarker = "<" + "<<<<<<" + " "
    let divider = "=" + "======"
    let rightMarker = ">" + ">>>>>>" + " "
    let extensions = Set(["swift", "metal", "h", "m", "json", "sh"])
    let fm = FileManager.default
    var violations: [String] = []

    for relativeRoot in roots {
        let directory = root.appendingPathComponent(relativeRoot)
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { continue }
        for case let file as URL in enumerator {
            guard extensions.contains(file.pathExtension),
                  let contents = try? String(contentsOf: file, encoding: .utf8)
            else { continue }
            let hasMarker = contents.split(
                separator: "\n", omittingEmptySubsequences: false
            ).contains { line in
                line.hasPrefix(leftMarker)
                    || line == Substring(divider)
                    || line.hasPrefix(rightMarker)
            }
            if hasMarker {
                violations.append(file.path.replacingOccurrences(
                    of: root.path + "/", with: ""))
            }
        }
    }

    #expect(violations.sorted().isEmpty, "merge conflict markers: \(violations.sorted())")
}
