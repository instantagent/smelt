import Foundation

#if canImport(Darwin)
import Darwin
#endif

public enum SmeltPackageDerivationError: Error, CustomStringConvertible {
    case invalidBase(String)
    case invalidOutput(String)
    case baseAndOutputMatch(String)
    case copyFailed(String)
    case publishFailed(String)

    public var description: String {
        switch self {
        case .invalidBase(let path):
            return "base package does not exist: \(path)"
        case .invalidOutput(let path):
            return "output must be a .smeltpkg package path: \(path)"
        case .baseAndOutputMatch(let path):
            return "output would overwrite the base package: \(path)"
        case .copyFailed(let detail):
            return "could not derive the package: \(detail)"
        case .publishFailed(let detail):
            return "could not publish the package: \(detail)"
        }
    }
}

/// A generic copy-on-write package transaction.
///
/// Mutation and validation happen in a hidden sibling directory. The final
/// path changes only after both closures return successfully. On APFS, the
/// recursive copy uses clones for large immutable model files; on other file
/// systems `copyfile` transparently falls back to a byte copy.
public enum SmeltPackageDerivation {
    @discardableResult
    public static func derive(
        base: URL,
        output: URL,
        fileManager: FileManager = .default,
        mutate: (URL) throws -> Void,
        validate: (URL) throws -> Void
    ) throws -> URL {
        let base = base.standardizedFileURL
        let output = output.standardizedFileURL
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: base.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw SmeltPackageDerivationError.invalidBase(base.path)
        }
        guard output.pathExtension == "smeltpkg" else {
            throw SmeltPackageDerivationError.invalidOutput(output.path)
        }
        guard base.path != output.path else {
            throw SmeltPackageDerivationError.baseAndOutputMatch(base.path)
        }

        let parent = output.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        let temporary = parent.appendingPathComponent(
            ".\(output.lastPathComponent).create-\(UUID().uuidString)",
            isDirectory: true
        )

        do {
            try copyPackage(from: base, to: temporary, fileManager: fileManager)
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw SmeltPackageDerivationError.copyFailed(error.localizedDescription)
        }

        do {
            try mutate(temporary)
            try validate(temporary)
            try publish(
                temporary: temporary,
                output: output,
                fileManager: fileManager
            )
            return output
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw error
        }
    }

    private static func copyPackage(
        from source: URL,
        to destination: URL,
        fileManager: FileManager
    ) throws {
        #if canImport(Darwin)
        let flags = copyfile_flags_t(
            COPYFILE_ALL | COPYFILE_RECURSIVE | COPYFILE_CLONE | COPYFILE_NOFOLLOW_SRC
        )
        if copyfile(source.path, destination.path, nil, flags) == 0 {
            return
        }
        let copyError = errno
        try? fileManager.removeItem(at: destination)
        if copyError != ENOTSUP && copyError != EXDEV {
            throw POSIXError(POSIXErrorCode(rawValue: copyError) ?? .EIO)
        }
        #endif
        try fileManager.copyItem(at: source, to: destination)
    }

    private static func publish(
        temporary: URL,
        output: URL,
        fileManager: FileManager
    ) throws {
        guard fileManager.fileExists(atPath: output.path) else {
            do {
                try fileManager.moveItem(at: temporary, to: output)
                return
            } catch {
                throw SmeltPackageDerivationError.publishFailed(
                    error.localizedDescription
                )
            }
        }

        #if canImport(Darwin)
        if renamex_np(temporary.path, output.path, UInt32(RENAME_SWAP)) == 0 {
            // `temporary` now names the replaced package. Publishing succeeded;
            // cleanup is best-effort and cannot invalidate the complete output.
            try? fileManager.removeItem(at: temporary)
            return
        }
        throw SmeltPackageDerivationError.publishFailed(
            String(cString: strerror(errno))
        )
        #else
        do {
            _ = try fileManager.replaceItemAt(
                output,
                withItemAt: temporary,
                backupItemName: nil,
                options: []
            )
        } catch {
            throw SmeltPackageDerivationError.publishFailed(
                error.localizedDescription
            )
        }
        #endif
    }
}
