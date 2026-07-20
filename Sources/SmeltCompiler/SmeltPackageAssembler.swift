// SmeltPackageAssembler — shared CAM package transaction skeleton.
//
// This is intentionally small: every family still owns its heavy artifact
// production, but the commit seam is now common. A caller hands the assembler a
// validated resolved plan plus concrete payloads; preflight proves the payload
// set exactly matches the CAM package inventory before any package directory is
// created.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

public enum SmeltPackageAssemblerError: Error, CustomStringConvertible, Equatable {
    case malformed(String)
    case io(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "package assembler: \(why)"
        case .io(let why): return "package assembler I/O: \(why)"
        }
    }
}

public enum SmeltPackageAssembler {

    public struct FilePayload: Sendable, Equatable {
        public enum Body: Sendable, Equatable {
            case data(Data)
            case copyFile(String)
            /// Immutable package artifact: hard-link on one filesystem so
            /// large weights occupy one inode, clone/copy only as fallback.
            case sharedFile(String)
            case copyDirectory(String)
            case directory
        }

        public let path: String
        public let body: Body

        public init(path: String, body: Body) {
            self.path = path
            self.body = body
        }
    }

    public struct PreparedAssembly: Sendable, Equatable {
        public let packagePath: String
        public let plan: SmeltPackageResolvedPlan
        public let payloads: [FilePayload]
    }

    public static func prepare(
        plan: SmeltPackageResolvedPlan,
        packagePath: String,
        payloads: [FilePayload]
    ) throws -> PreparedAssembly {
        try validatePackagePath(packagePath)
        let plannedPaths = Set(plan.packageFiles.map(\.path))
        let requiredPaths = plannedPaths

        var payloadsByPath: [String: FilePayload] = [:]
        for payload in payloads {
            try validatePackageRelativePath(payload.path, field: "payload path")
            guard plannedPaths.contains(payload.path) else {
                throw SmeltPackageAssemblerError.malformed(
                    "payload '\(payload.path)' is not declared by resolved plan"
                )
            }
            guard payloadsByPath[payload.path] == nil else {
                throw SmeltPackageAssemblerError.malformed(
                    "payload '\(payload.path)' declared twice"
                )
            }
            if case .copyFile(let source) = payload.body {
                try validateCopySource(source, payload: payload.path)
            }
            if case .sharedFile(let source) = payload.body {
                try validateCopySource(source, payload: payload.path)
            }
            if case .copyDirectory(let source) = payload.body {
                try validateCopyDirectorySource(source, payload: payload.path)
            }
            payloadsByPath[payload.path] = payload
        }

        let missing = requiredPaths.subtracting(payloadsByPath.keys).sorted()
        guard missing.isEmpty else {
            throw SmeltPackageAssemblerError.malformed(
                "missing payload(s): \(missing.joined(separator: ", "))"
            )
        }

        let ordered = plan.packageFiles.map(\.path).compactMap { payloadsByPath[$0] }
        return PreparedAssembly(packagePath: packagePath, plan: plan, payloads: ordered)
    }

    @discardableResult
    public static func assemble(
        plan: SmeltPackageResolvedPlan,
        packagePath: String,
        payloads: [FilePayload],
        fileManager: FileManager = .default
    ) throws -> PreparedAssembly {
        let prepared = try prepare(plan: plan, packagePath: packagePath, payloads: payloads)
        try commit(prepared, fileManager: fileManager)
        return prepared
    }

    public static func commit(
        _ prepared: PreparedAssembly,
        fileManager fm: FileManager = .default
    ) throws {
        let packageURL = URL(fileURLWithPath: prepared.packagePath, isDirectory: true)
        var isDirectory = ObjCBool(false)
        if fm.fileExists(atPath: prepared.packagePath, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw SmeltPackageAssemblerError.io(
                    "package path exists and is not a directory: \(prepared.packagePath)"
                )
            }
            try validateNoStalePackageEntries(prepared, fileManager: fm)
            try writePayloads(prepared.payloads, into: packageURL, fileManager: fm)
        } else {
            try commitNewPackage(prepared, packageURL: packageURL, fileManager: fm)
        }
    }

    private static func commitNewPackage(
        _ prepared: PreparedAssembly,
        packageURL: URL,
        fileManager fm: FileManager
    ) throws {
        let parent = packageURL.deletingLastPathComponent()
        let staging = parent.appendingPathComponent(
            ".\(packageURL.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try io {
                try fm.createDirectory(at: parent, withIntermediateDirectories: true)
                try fm.createDirectory(at: staging, withIntermediateDirectories: false)
            }
            try writePayloads(prepared.payloads, into: staging, fileManager: fm)
            try io {
                try fm.moveItem(at: staging, to: packageURL)
            }
        } catch {
            try? fm.removeItem(at: staging)
            throw error
        }
    }

    private static func writePayloads(
        _ payloads: [FilePayload],
        into root: URL,
        fileManager fm: FileManager
    ) throws {
        for payload in payloads {
            let destination = root.appendingPathComponent(payload.path)
            try createParentDirectory(for: destination, fileManager: fm)
            switch payload.body {
            case .data(let data):
                try io {
                    try data.write(to: destination, options: .atomic)
                }
            case .copyFile(let source):
                try copyFileAtomically(
                    from: URL(fileURLWithPath: source).resolvingSymlinksInPath(),
                    to: destination,
                    fileManager: fm
                )
            case .sharedFile(let source):
                try shareFileAtomically(
                    from: URL(fileURLWithPath: source).resolvingSymlinksInPath(),
                    to: destination,
                    fileManager: fm
                )
            case .copyDirectory(let source):
                try copyDirectoryAtomically(
                    from: URL(fileURLWithPath: source).resolvingSymlinksInPath(),
                    to: destination,
                    fileManager: fm
                )
            case .directory:
                try createDirectory(at: destination, fileManager: fm)
            }
        }
    }

    private static func validateNoStalePackageEntries(
        _ prepared: PreparedAssembly,
        fileManager fm: FileManager
    ) throws {
        let root = URL(fileURLWithPath: prepared.packagePath, isDirectory: true)
        let planned = Set(prepared.plan.packageFiles.map(\.path))
        let opaqueSubtrees = Set(prepared.payloads.compactMap { payload in
            if case .copyDirectory = payload.body { return payload.path }
            return nil
        })
        var allowed = planned
        for path in planned {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }

        guard let enumerator = fm.enumerator(atPath: root.path) else {
            throw SmeltPackageAssemblerError.io(
                "could not enumerate package path: \(prepared.packagePath)"
            )
        }

        var stale: [String] = []
        for case let relative as String in enumerator {
            if !allowed.contains(relative)
                && !opaqueSubtrees.contains(where: { relative.hasPrefix($0 + "/") }) {
                stale.append(relative)
                if stale.count >= 8 { break }
            }
        }
        guard stale.isEmpty else {
            throw SmeltPackageAssemblerError.malformed(
                "existing package has stale file(s): \(stale.sorted().joined(separator: ", "))"
            )
        }
    }

    private static func createParentDirectory(
        for destination: URL,
        fileManager fm: FileManager
    ) throws {
        let parent = destination.deletingLastPathComponent()
        try io {
            try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        }
    }

    private static func createDirectory(
        at url: URL,
        fileManager fm: FileManager
    ) throws {
        var isDirectory = ObjCBool(false)
        if fm.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw SmeltPackageAssemblerError.io(
                    "directory payload conflicts with file: \(url.path)"
                )
            }
            return
        }
        try io {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
    }

    private static func copyFileAtomically(
        from source: URL,
        to destination: URL,
        fileManager fm: FileManager
    ) throws {
        let tmp = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).tmp-\(UUID().uuidString)")
        do {
            try copyFilePreservingBytes(from: source, to: tmp, fileManager: fm)
            if isSymbolicLink(at: destination) {
                try fm.removeItem(at: destination)
                try fm.moveItem(at: tmp, to: destination)
            } else if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(
                    destination,
                    withItemAt: tmp,
                    backupItemName: nil,
                    options: []
                )
            } else {
                try fm.moveItem(at: tmp, to: destination)
            }
        } catch {
            try? fm.removeItem(at: tmp)
            throw SmeltPackageAssemblerError.io(error.localizedDescription)
        }
    }

    private static func copyDirectoryAtomically(
        from source: URL,
        to destination: URL,
        fileManager fm: FileManager
    ) throws {
        let parent = destination.deletingLastPathComponent()
        let tmp = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try fm.copyItem(at: source, to: tmp)
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: tmp, to: destination)
        } catch {
            try? fm.removeItem(at: tmp)
            throw SmeltPackageAssemblerError.io(error.localizedDescription)
        }
    }

    private static func shareFileAtomically(
        from source: URL,
        to destination: URL,
        fileManager fm: FileManager
    ) throws {
        if !isSymbolicLink(at: destination),
           fm.fileExists(atPath: destination.path),
           sameFileReference(source, destination, fileManager: fm) {
            // Incremental projection commonly starts with source and final
            // already hard-linked to the same immutable weight body. Replacing
            // one link with a third link to that inode is both unnecessary and
            // rejected by Foundation's replaceItemAt on APFS.
            return
        }
        let tmp = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).tmp-\(UUID().uuidString)")
        do {
            do {
                try fm.linkItem(at: source, to: tmp)
            } catch {
                // Cross-volume and filesystems without hard links still get
                // APFS clone-on-write where available before a byte copy.
                try copyFilePreservingBytes(from: source, to: tmp, fileManager: fm)
            }
            if isSymbolicLink(at: destination) {
                try fm.removeItem(at: destination)
                try fm.moveItem(at: tmp, to: destination)
            } else if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(
                    destination,
                    withItemAt: tmp,
                    backupItemName: nil,
                    options: []
                )
            } else {
                try fm.moveItem(at: tmp, to: destination)
            }
        } catch {
            try? fm.removeItem(at: tmp)
            throw SmeltPackageAssemblerError.io(error.localizedDescription)
        }
    }

    private static func sameFileReference(
        _ lhs: URL,
        _ rhs: URL,
        fileManager fm: FileManager
    ) -> Bool {
        guard let left = try? fm.attributesOfItem(atPath: lhs.path),
              let right = try? fm.attributesOfItem(atPath: rhs.path),
              let leftDevice = left[.systemNumber] as? NSNumber,
              let rightDevice = right[.systemNumber] as? NSNumber,
              let leftInode = left[.systemFileNumber] as? NSNumber,
              let rightInode = right[.systemFileNumber] as? NSNumber else {
            return false
        }
        return leftDevice == rightDevice && leftInode == rightInode
    }

    private static func copyFilePreservingBytes(
        from source: URL,
        to destination: URL,
        fileManager fm: FileManager
    ) throws {
        #if canImport(Darwin)
        if clonefile(source.path, destination.path, 0) == 0 {
            return
        }
        #endif
        try fm.copyItem(at: source, to: destination)
    }

    private static func isSymbolicLink(at url: URL) -> Bool {
        let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey])
        return values?.isSymbolicLink == true
    }

    private static func validatePackagePath(_ packagePath: String) throws {
        guard !packagePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SmeltPackageAssemblerError.malformed("package path must be non-empty")
        }
    }

    private static func validateCopySource(_ source: String, payload: String) throws {
        guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SmeltPackageAssemblerError.malformed(
                "copy payload '\(payload)' has empty source"
            )
        }
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: source, isDirectory: &isDirectory) else {
            throw SmeltPackageAssemblerError.malformed(
                "copy payload '\(payload)' source does not exist: \(source)"
            )
        }
        guard !isDirectory.boolValue else {
            throw SmeltPackageAssemblerError.malformed(
                "copy payload '\(payload)' source is a directory: \(source)"
            )
        }
    }

    private static func validateCopyDirectorySource(_ source: String, payload: String) throws {
        guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SmeltPackageAssemblerError.malformed(
                "copy-directory payload '\(payload)' has empty source"
            )
        }
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: source, isDirectory: &isDirectory) else {
            throw SmeltPackageAssemblerError.malformed(
                "copy-directory payload '\(payload)' source does not exist: \(source)"
            )
        }
        guard isDirectory.boolValue else {
            throw SmeltPackageAssemblerError.malformed(
                "copy-directory payload '\(payload)' source is not a directory: \(source)"
            )
        }
    }

    private static func validatePackageRelativePath(
        _ path: String,
        field: String
    ) throws {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.hasPrefix("~"),
              !path.contains("\\"),
              !path.contains("\0")
        else {
            throw SmeltPackageAssemblerError.malformed("\(field) is unsafe: \(path)")
        }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty else {
            throw SmeltPackageAssemblerError.malformed("\(field) is unsafe: \(path)")
        }
        for part in parts where part.isEmpty || part == "." || part == ".." {
            throw SmeltPackageAssemblerError.malformed("\(field) is unsafe: \(path)")
        }
    }

    private static func io(_ body: () throws -> Void) throws {
        do {
            try body()
        } catch let error as SmeltPackageAssemblerError {
            throw error
        } catch {
            throw SmeltPackageAssemblerError.io(error.localizedDescription)
        }
    }
}
