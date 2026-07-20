import CryptoKit
import Darwin
import Foundation

/// Stable content identity for a compiled Smelt package.
///
/// Package manifests carry the checksums of their runtime payloads, so hashing
/// the exact manifest bytes gives consumers a portable identifier without
/// re-reading multi-gigabyte weights during every lookup.
public enum SmeltPackageIdentity {
    public static func compute(packagePath: String) throws -> String {
        let manifestURL = URL(fileURLWithPath: packagePath, isDirectory: true)
            .appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let value = try JSONSerialization.jsonObject(with: data)
        guard value is [String: Any] else {
            throw SmeltPackageStoreError.invalidManifest(manifestURL.path)
        }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

public struct SmeltStoredPackage: Sendable, Equatable {
    public let identity: String
    public let packageURL: URL

    public init(identity: String, packageURL: URL) {
        self.identity = identity
        self.packageURL = packageURL
    }
}

/// Content-addressed package location shared by Smelt consumers.
///
/// Installation clones immutable package files into one canonical location.
/// Every consumer then opens that same stored inode for file-backed mappings
/// and no-copy Metal buffers. APFS clone-on-write avoids the first physical
/// copy where possible; other filesystems fall back to ordinary copies. This
/// is deliberately best effort: package identity, not residency, is the API.
public enum SmeltPackageStore {
    public static func rootURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["SMELT_PACKAGE_STORE_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        return applicationSupport
            .appendingPathComponent("smelt", isDirectory: true)
            .appendingPathComponent("packages", isDirectory: true)
    }

    public static func packageURL(identity: String) throws -> URL {
        guard isSHA256(identity) else {
            throw SmeltPackageStoreError.invalidIdentity(identity)
        }
        return rootURL().appendingPathComponent(
            "\(identity).smeltpkg",
            isDirectory: true
        )
    }

    public static func locate(identity: String) throws -> SmeltStoredPackage? {
        let url = try packageURL(identity: identity)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let actual = try SmeltPackageIdentity.compute(packagePath: url.path)
        guard actual == identity else {
            throw SmeltPackageStoreError.identityMismatch(
                expected: identity,
                actual: actual
            )
        }
        return SmeltStoredPackage(identity: identity, packageURL: url)
    }

    /// Enumerates canonical packages without exposing the store's directory
    /// layout to downstream consumers.
    public static func installedPackages() throws -> [SmeltStoredPackage] {
        let root = rootURL()
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        let candidates = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return try candidates
            .filter { $0.pathExtension == "smeltpkg" }
            .map { candidate in
                let identity = candidate.deletingPathExtension().lastPathComponent
                guard isSHA256(identity) else {
                    throw SmeltPackageStoreError.invalidIdentity(identity)
                }
                guard let stored = try locate(identity: identity) else {
                    throw SmeltPackageStoreError.packageNotFound(identity)
                }
                return stored
            }
            .sorted { $0.identity < $1.identity }
    }

    @discardableResult
    public static func install(packagePath: String) throws -> SmeltStoredPackage {
        let source = URL(fileURLWithPath: packagePath, isDirectory: true)
            .standardizedFileURL
        let identity = try SmeltPackageIdentity.compute(packagePath: source.path)
        if let existing = try locate(identity: identity) { return existing }

        let fileManager = FileManager.default
        let root = rootURL()
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        let destination = try packageURL(identity: identity)
        let staging = root.appendingPathComponent(
            ".\(identity).tmp-\(getpid())-\(UUID().uuidString)",
            isDirectory: true
        )
        try cloneTree(from: source, to: staging)
        do {
            try fileManager.moveItem(at: staging, to: destination)
        } catch CocoaError.fileWriteFileExists {
            try? fileManager.removeItem(at: staging)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
        guard let installed = try locate(identity: identity) else {
            throw SmeltPackageStoreError.installFailed(destination.path)
        }
        return installed
    }

    /// Writes a portable copy of one stored package.
    ///
    /// The canonical store may contain absolute symlinks into Smelt's local
    /// content-addressed blob store. Those links are an implementation detail
    /// that must not escape into registries or release archives. Materializing
    /// dereferences file symlinks while still using clone-on-write for every
    /// eligible payload, so a local export remains deduplicated on a best-effort
    /// basis without making the exported package machine-local.
    @discardableResult
    public static func materialize(
        identity: String,
        at destination: URL
    ) throws -> URL {
        guard let stored = try locate(identity: identity) else {
            throw SmeltPackageStoreError.packageNotFound(identity)
        }

        let fileManager = FileManager.default
        let destination = destination.standardizedFileURL
        guard !fileManager.fileExists(atPath: destination.path) else {
            throw SmeltPackageStoreError.destinationExists(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(getpid())-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try materializeTree(from: stored.packageURL, to: staging)
            let actual = try SmeltPackageIdentity.compute(packagePath: staging.path)
            guard actual == identity else {
                throw SmeltPackageStoreError.identityMismatch(
                    expected: identity,
                    actual: actual
                )
            }
            try fileManager.moveItem(at: staging, to: destination)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
        return destination
    }

    private static func cloneTree(from source: URL, to destination: URL) throws {
        var metadata = stat()
        guard lstat(source.path, &metadata) == 0 else {
            throw SmeltPackageStoreError.installFailed(source.path)
        }
        switch metadata.st_mode & S_IFMT {
        case S_IFLNK:
            let target = try FileManager.default.destinationOfSymbolicLink(
                atPath: source.path
            )
            try FileManager.default.createSymbolicLink(
                atPath: destination.path,
                withDestinationPath: target
            )
        case S_IFDIR:
            try FileManager.default.createDirectory(
                at: destination,
                withIntermediateDirectories: false
            )
            for child in try FileManager.default.contentsOfDirectory(
                at: source,
                includingPropertiesForKeys: nil,
                options: []
            ) {
                try cloneTree(
                    from: child,
                    to: destination.appendingPathComponent(child.lastPathComponent)
                )
            }
        case S_IFREG:
            if clonefile(source.path, destination.path, 0) != 0 {
                try FileManager.default.copyItem(at: source, to: destination)
            }
        default:
            throw SmeltPackageStoreError.unsupportedEntry(source.path)
        }
    }

    private static func materializeTree(
        from source: URL,
        to destination: URL
    ) throws {
        var metadata = stat()
        guard lstat(source.path, &metadata) == 0 else {
            throw SmeltPackageStoreError.installFailed(source.path)
        }
        switch metadata.st_mode & S_IFMT {
        case S_IFLNK:
            let target = try FileManager.default.destinationOfSymbolicLink(
                atPath: source.path
            )
            guard SmeltCAS.isStoreEntryPath(target) else {
                throw SmeltPackageStoreError.unsupportedEntry(source.path)
            }
            let resolved = source.resolvingSymlinksInPath()
            var targetMetadata = stat()
            guard stat(resolved.path, &targetMetadata) == 0,
                  targetMetadata.st_mode & S_IFMT == S_IFREG
            else {
                throw SmeltPackageStoreError.unsupportedEntry(source.path)
            }
            try cloneRegularFile(from: resolved, to: destination)
        case S_IFDIR:
            try FileManager.default.createDirectory(
                at: destination,
                withIntermediateDirectories: false
            )
            for child in try FileManager.default.contentsOfDirectory(
                at: source,
                includingPropertiesForKeys: nil,
                options: []
            ) {
                try materializeTree(
                    from: child,
                    to: destination.appendingPathComponent(child.lastPathComponent)
                )
            }
        case S_IFREG:
            try cloneRegularFile(from: source, to: destination)
        default:
            throw SmeltPackageStoreError.unsupportedEntry(source.path)
        }
    }

    private static func cloneRegularFile(
        from source: URL,
        to destination: URL
    ) throws {
        if clonefile(source.path, destination.path, 0) != 0 {
            try FileManager.default.copyItem(at: source, to: destination)
        }
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }
}

public enum SmeltPackageStoreError: Error, CustomStringConvertible, Equatable {
    case invalidManifest(String)
    case invalidIdentity(String)
    case identityMismatch(expected: String, actual: String)
    case packageNotFound(String)
    case destinationExists(String)
    case unsupportedEntry(String)
    case installFailed(String)

    public var description: String {
        switch self {
        case .invalidManifest(let path):
            return "package manifest is not a JSON object: \(path)"
        case .invalidIdentity(let identity):
            return "invalid Smelt package identity '\(identity)'"
        case .identityMismatch(let expected, let actual):
            return "stored package identity mismatch: expected \(expected), got \(actual)"
        case .packageNotFound(let identity):
            return "Smelt package \(identity) is not installed"
        case .destinationExists(let path):
            return "package materialization destination already exists: \(path)"
        case .unsupportedEntry(let path):
            return "unsupported package entry: \(path)"
        case .installFailed(let path):
            return "failed to install Smelt package at \(path)"
        }
    }
}
