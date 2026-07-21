import CryptoKit
import Darwin
import Foundation

// Content-addressed store for large package artifacts. `adopt` moves a
// package's manifest-checksummed files into the shared store and leaves
// symlinks behind, so N packages built from the same base model share one
// weights.bin inode on disk. Read-only file-backed mappings give macOS and
// Metal the best opportunity to share physical pages across consumers; exact
// residency remains an OS-controlled optimization, not an API guarantee.
//
// Adoption order is windowless on the common (same-volume) path: hardlink
// the file into the store first, then atomically replace the package path
// with a symlink via rename. A concurrent loader sees either the regular
// file or the symlink — never a missing path. A process that already
// mmap'd the old path is unaffected; its open fd pins the inode.

enum SmeltCAS {
    /// Files smaller than this stay in the package; symlink indirection
    /// only pays for itself on large artifacts.
    static let defaultMinBytes = SmeltPackageStore.defaultBlobAdoptionMinBytes

    enum CASError: Error, CustomStringConvertible {
        case hashMismatch(file: String, expected: String, actual: String)
        case storeEntryCorrupt(file: String, entry: String)
        case brokenLink(file: String, target: String)
        case io(String)

        public var description: String {
            switch self {
            case .hashMismatch(let file, let expected, let actual):
                return "\(file): contents (sha256 \(actual.prefix(12))...) "
                    + "disagree with manifest (\(expected.prefix(12))...); "
                    + "refusing to adopt"
            case .storeEntryCorrupt(let file, let entry):
                return "\(file): store entry \(entry) has wrong size; "
                    + "run `smelt cas restore` on affected packages and "
                    + "delete the entry"
            case .brokenLink(let file, let target):
                return "\(file): symlink target \(target) is missing "
                    + "(store deleted, or package moved to another machine?)"
            case .io(let message):
                return message
            }
        }
    }

    /// Store root: $SMELT_CAS_DIR or
    /// ~/Library/Application Support/smelt/cas. Entries live under
    /// `<root>/sha256/<hex>`.
    static func storeRoot() -> URL {
        if let override = ProcessInfo.processInfo.environment["SMELT_CAS_DIR"],
            !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        )[0]
        return appSupport
            .appendingPathComponent("smelt", isDirectory: true)
            .appendingPathComponent("cas", isDirectory: true)
    }

    /// Entries directory: `<root>/sha256`.
    static func entriesDirectory() -> URL {
        storeRoot().appendingPathComponent("sha256", isDirectory: true)
    }

    static func entryPath(forKey key: String) -> String {
        entriesDirectory().appendingPathComponent(key).path
    }

    /// True when a healthy (regular-file) entry exists for the key.
    static func hasEntry(key: String) -> Bool {
        var st = stat()
        return lstat(entryPath(forKey: key), &st) == 0
            && (st.st_mode & S_IFMT) == S_IFREG
    }

    /// Install a downloaded blob into the store, consuming the staging
    /// file. The staging contents must hash to `key`; the staging file
    /// should live on the store's volume (put it next to the entries)
    /// so the final rename is atomic.
    static func insertEntry(
        key: String, fromStagingFile staging: String
    ) throws {
        let actual = try sha256Hex(ofFileAt: staging)
        guard actual == key else {
            unlink(staging)
            throw CASError.hashMismatch(
                file: staging, expected: key, actual: actual
            )
        }
        try FileManager.default.createDirectory(
            at: entriesDirectory(), withIntermediateDirectories: true
        )
        let entry = entryPath(forKey: key)
        try withEntryLock(entry: entry) {
            if hasEntry(key: key) {
                unlink(staging)
                return
            }
            guard rename(staging, entry) == 0 else {
                unlink(staging)
                throw CASError.io(
                    "rename into store failed: \(errnoString())"
                )
            }
            chmod(entry, 0o400)
            fsyncDirectory(containing: entry)
        }
    }

    /// SHA-256 of a file, streamed. Exposed to package-store consumers that
    /// share the same content keying.
    static func sha256(ofFileAt path: String) throws -> String {
        try sha256Hex(ofFileAt: path)
    }

    // MARK: - Status

    /// Every regular file in the package is shareable: files with a
    /// manifest checksum are keyed (and corruption-gated) by it; the rest
    /// — prepared artifacts like compiled_grammar.trie and prepared_prefix.snapshot,
    /// model.metalarchive — are keyed by their content hash at adopt time.
    /// Identical bytes land on the same store entry either way.
    /// manifest.json itself always stays a regular file: it is the
    /// package's identity and the key source for everything else.
    static func status(
        packagePath: String,
        minBytes: Int = defaultMinBytes
    ) throws -> SmeltPackageBlobReport {
        let expectedByName = try expectedChecksums(packagePath: packagePath)

        var files: [SmeltPackageBlobFileReport] = []
        var seen: Set<String> = []
        let names = try FileManager.default
            .contentsOfDirectory(atPath: packagePath).sorted()

        for name in names {
            if name == "manifest.json" { continue }
            if name.contains(".cas-tmp-") || name.contains(".restore-tmp-") {
                continue
            }
            let path = "\(packagePath)/\(name)"
            var st = stat()
            guard lstat(path, &st) == 0 else { continue }
            let expected = expectedByName[name]

            if (st.st_mode & S_IFMT) == S_IFLNK {
                let target = (try? FileManager.default
                    .destinationOfSymbolicLink(atPath: path)) ?? ""
                guard isStoreEntryPath(target) else { continue }
                // The entry must still be a regular file — a store entry
                // replaced by a symlink/directory is as broken as a
                // missing one.
                var targetSt = stat()
                let targetOK = lstat(target, &targetSt) == 0
                    && (targetSt.st_mode & S_IFMT) == S_IFREG
                seen.insert(name)
                files.append(SmeltPackageBlobFileReport(
                    name: name,
                    state: targetOK ? .alreadyAdopted : .brokenLink,
                    bytes: targetOK ? targetSt.st_size : 0,
                    sha256: expected
                ))
                continue
            }
            guard (st.st_mode & S_IFMT) == S_IFREG else { continue }
            seen.insert(name)
            files.append(SmeltPackageBlobFileReport(
                name: name,
                state: st.st_size >= minBytes ? .eligible : .belowThreshold,
                bytes: st.st_size,
                sha256: expected
            ))
        }

        for (name, expected) in expectedByName.sorted(by: { $0.key < $1.key })
        where !seen.contains(name) {
            files.append(SmeltPackageBlobFileReport(
                name: name, state: .missing, bytes: 0, sha256: expected
            ))
        }
        return SmeltPackageBlobReport(packagePath: packagePath, files: files)
    }

    // MARK: - Adopt

    /// Move eligible files into the store, leaving symlinks behind. The
    /// store key is the file's SHA-256: for manifest-checksummed files
    /// the recorded hash doubles as a corruption gate (contents that
    /// disagree are refused, and the error is thrown after the remaining
    /// files are processed); everything else is keyed by the content
    /// hash computed here, so identical prepared artifacts dedup too.
    @discardableResult
    static func adopt(
        packagePath: String,
        minBytes: Int = defaultMinBytes
    ) throws -> SmeltPackageBlobReport {
        let before = try status(packagePath: packagePath, minBytes: minBytes)
        let entriesDir = storeRoot().appendingPathComponent(
            "sha256", isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: entriesDir, withIntermediateDirectories: true
        )

        var files: [SmeltPackageBlobFileReport] = []
        var firstError: Error?

        for report in before.files {
            let path = "\(packagePath)/\(report.name)"
            var st = stat()
            let isRegular = lstat(path, &st) == 0
                && (st.st_mode & S_IFMT) == S_IFREG
            guard report.state == .eligible, isRegular else {
                files.append(report)
                continue
            }

            do {
                let key = try adoptFile(
                    path: path,
                    name: report.name,
                    expected: report.sha256,
                    entriesDir: entriesDir
                )
                files.append(SmeltPackageBlobFileReport(
                    name: report.name, state: .adopted,
                    bytes: st.st_size, sha256: key
                ))
            } catch {
                if firstError == nil { firstError = error }
                files.append(SmeltPackageBlobFileReport(
                    name: report.name, state: .refused,
                    bytes: st.st_size, sha256: report.sha256
                ))
            }
        }

        if let firstError { throw firstError }
        return SmeltPackageBlobReport(packagePath: packagePath, files: files)
    }

    // MARK: - Restore

    /// Copy adopted files back into the package as regular files,
    /// replacing the symlinks. The escape hatch before moving a package
    /// to another machine or volume.
    @discardableResult
    static func restore(packagePath: String) throws -> SmeltPackageBlobReport {
        let before = try status(packagePath: packagePath, minBytes: 0)
        let adopted = before.files.filter {
            $0.state == .alreadyAdopted || $0.state == .brokenLink
        }

        // Preflight every target so a broken link aborts the whole
        // restore before any file is touched, not halfway through. The
        // target must be a regular file, not a directory or a further
        // symlink someone smuggled into the store.
        for report in adopted {
            let path = "\(packagePath)/\(report.name)"
            let target = try FileManager.default
                .destinationOfSymbolicLink(atPath: path)
            var st = stat()
            guard lstat(target, &st) == 0,
                (st.st_mode & S_IFMT) == S_IFREG else {
                throw CASError.brokenLink(file: report.name, target: target)
            }
        }

        var files: [SmeltPackageBlobFileReport] = []
        for report in before.files {
            guard report.state == .alreadyAdopted else {
                files.append(report)
                continue
            }
            let path = "\(packagePath)/\(report.name)"
            let target = try FileManager.default
                .destinationOfSymbolicLink(atPath: path)
            let tmp = "\(path).restore-tmp-\(getpid())"
            try? FileManager.default.removeItem(atPath: tmp)
            try FileManager.default.copyItem(atPath: target, toPath: tmp)
            // Snapshots are created owner-only (they hold prompt state);
            // give them that back. Everything else is plain shareable.
            chmod(tmp, report.name == "prepared_prefix.snapshot" ? 0o600 : 0o644)
            guard rename(tmp, path) == 0 else {
                try? FileManager.default.removeItem(atPath: tmp)
                throw CASError.io(
                    "\(report.name): rename failed: \(errnoString())"
                )
            }
            files.append(SmeltPackageBlobFileReport(
                name: report.name, state: .restored,
                bytes: report.bytes, sha256: report.sha256
            ))
        }
        return SmeltPackageBlobReport(packagePath: packagePath, files: files)
    }

    // MARK: - Internals

    /// Read only the integrity fields needed for blob adoption. Package-store
    /// consumers can install any Smelt package flavor; requiring the runnable
    /// text manifest here would silently exclude component packages from dedup.
    private static func expectedChecksums(
        packagePath: String
    ) throws -> [String: String] {
        let data = try Data(contentsOf: URL(
            fileURLWithPath: "\(packagePath)/manifest.json"
        ))
        guard let manifest = try JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            throw CASError.io("manifest.json is not an object")
        }
        guard let checksums = manifest["checksums"] as? [String: Any] else {
            return [:]
        }
        let files = manifest["files"] as? [String: Any] ?? [:]
        var result: [String: String] = [:]

        func record(_ checksumKeys: [String], file: String) {
            for key in checksumKeys {
                if let value = checksums[key] as? String, !value.isEmpty {
                    result[file] = value
                    return
                }
            }
        }

        record(["weights_bin"], file: "weights.bin")
        record(["metallib"], file: "model.metallib")
        record(["generated_swift"], file: "SmeltGenerated.swift")
        record(["dispatches_bin"], file: "dispatches.bin")
        record(["prefill_dispatches_bin"], file: "prefill_dispatches.bin")
        record(
            ["prefill_verify_argmax_dispatches_bin"],
            file: "prefill_verify_argmax_dispatches.bin"
        )
        record(["tokenizer_json"], file: "tokenizer.json")

        record(
            ["weightsSHA256", "weights_sha256"],
            file: files["weights"] as? String ?? "weights.bin"
        )
        record(
            ["metallibSHA256", "metallib_sha256"],
            file: files["metallib"] as? String ?? "model.metallib"
        )
        if let cam = files["cam"] as? String {
            record(["camSHA256", "cam_sha256"], file: cam)
        }
        return result
    }

    /// Adopt one file. The opened fd pins an inode for the whole
    /// operation: the hash, the store insert, and the final swap all
    /// refer to that inode, so a concurrent rewrite of `path` (a rebuild
    /// replacing compiled_grammar.trie, say) can never publish bytes under
    /// the wrong store key, and the swap is skipped when the recheck
    /// sees the path was rewritten. Known residual: a rewrite landing in
    /// the microseconds between that recheck and the swap's rename is
    /// still replaced by the symlink to the older content (recoverable
    /// by re-baking; the window was the full hash duration before the
    /// fd-pinning design). Returns the store key.
    private static func adoptFile(
        path: String, name: String, expected: String?, entriesDir: URL
    ) throws -> String {
        let fd = open(path, O_RDONLY | O_NOFOLLOW)
        guard fd >= 0 else {
            // A concurrent adopter of the same package may have swapped
            // the path to a symlink already (open gives ELOOP). That is
            // success, not failure.
            if errno == ELOOP, let key = adoptedStoreKey(path: path) {
                return key
            }
            throw CASError.io("\(name): open failed: \(errnoString())")
        }
        defer { close(fd) }
        var pinned = stat()
        guard fstat(fd, &pinned) == 0,
            (pinned.st_mode & S_IFMT) == S_IFREG else {
            throw CASError.io("\(name): not a regular file")
        }

        let key = try sha256Hex(fd: fd)
        if let expected, key != expected {
            throw CASError.hashMismatch(
                file: name, expected: expected, actual: key
            )
        }

        let entry = entriesDir.appendingPathComponent(key).path
        try withEntryLock(entry: entry) {
            try insertPinned(
                path: path, fd: fd, pinned: pinned, entry: entry, key: key,
                name: name
            )
            var current = stat()
            guard lstat(path, &current) == 0 else {
                throw CASError.io(
                    "\(name): disappeared during adoption"
                )
            }
            if (current.st_mode & S_IFMT) == S_IFLNK {
                // Another adopter won the race to the same entry: done.
                guard (try? FileManager.default
                    .destinationOfSymbolicLink(atPath: path)) == entry else {
                    throw CASError.io(
                        "\(name): changed during adoption; left in place"
                    )
                }
                chmod(entry, 0o400)
                return
            }
            // Swap only while the package path still names the pinned
            // inode; if it was rewritten since we hashed, leave the
            // fresh file alone (the store entry stays for next time).
            guard current.st_dev == pinned.st_dev,
                current.st_ino == pinned.st_ino else {
                throw CASError.io(
                    "\(name): changed during adoption; left in place"
                )
            }
            try replaceWithSymlink(path: path, target: entry)
            chmod(entry, 0o400)
        }
        return key
    }

    /// The store key a path is already adopted under, when it is a
    /// symlink to a healthy (regular-file) store entry.
    private static func adoptedStoreKey(path: String) -> String? {
        guard let target = try? FileManager.default
            .destinationOfSymbolicLink(atPath: path),
            isStoreEntryPath(target) else { return nil }
        var st = stat()
        guard lstat(target, &st) == 0,
            (st.st_mode & S_IFMT) == S_IFREG else { return nil }
        return (target as NSString).lastPathComponent
    }

    /// Get the pinned inode's bytes into the store at `entry`.
    /// Same-volume: hardlink to a temp name (instant, no extra disk),
    /// verified to be the pinned inode. Cross-volume, or when the path
    /// was rewritten under us: stream a copy from the hashed fd. Either
    /// way the bytes that land under `key` are exactly the bytes that
    /// were hashed.
    private static func insertPinned(
        path: String, fd: Int32, pinned: stat, entry: String, key: String,
        name: String
    ) throws {
        var entrySt = stat()
        if lstat(entry, &entrySt) == 0 {
            // Dedup hit. Never trust it blindly: require a regular file
            // whose contents still hash to its name before pointing
            // packages at it.
            guard (entrySt.st_mode & S_IFMT) == S_IFREG else {
                throw CASError.storeEntryCorrupt(file: name, entry: entry)
            }
            if entrySt.st_dev == pinned.st_dev,
                entrySt.st_ino == pinned.st_ino {
                return  // our own inode from an earlier partial adopt
            }
            guard try sha256Hex(ofFileAt: entry) == key else {
                throw CASError.storeEntryCorrupt(file: name, entry: entry)
            }
            return
        }

        let tmp = "\(entry).tmp-\(getpid())"
        unlink(tmp)
        if link(path, tmp) == 0 {
            // The hardlink names whatever inode `path` has *now* —
            // verify it is the pinned one, else fall back to copying
            // the fd we actually hashed.
            var tmpSt = stat()
            let isPinnedInode = lstat(tmp, &tmpSt) == 0
                && tmpSt.st_dev == pinned.st_dev
                && tmpSt.st_ino == pinned.st_ino
            if !isPinnedInode {
                unlink(tmp)
                try copyPinned(fd: fd, to: tmp)
            }
        } else if errno == EXDEV {
            try copyPinned(fd: fd, to: tmp)
        } else {
            throw CASError.io(
                "\(name): link into store failed: \(errnoString())"
            )
        }
        guard rename(tmp, entry) == 0 else {
            unlink(tmp)
            throw CASError.io(
                "\(name): rename into store failed: \(errnoString())"
            )
        }
        fsyncDirectory(containing: entry)
    }

    /// Stream the pinned fd's contents to `destination` and force them
    /// to disk. Reading from the fd (not the path) keeps this immune to
    /// concurrent path rewrites.
    private static func copyPinned(fd: Int32, to destination: String) throws {
        guard lseek(fd, 0, SEEK_SET) == 0 else {
            throw CASError.io("seek failed: \(errnoString())")
        }
        let out = open(
            destination, O_WRONLY | O_CREAT | O_TRUNC | O_EXCL, 0o600
        )
        guard out >= 0 else {
            throw CASError.io(
                "store temp create failed: \(errnoString())"
            )
        }
        defer { close(out) }
        var buffer = [UInt8](repeating: 0, count: 1 << 20)
        while true {
            let readCount = buffer.withUnsafeMutableBytes {
                read(fd, $0.baseAddress, $0.count)
            }
            if readCount == 0 { break }
            guard readCount > 0 else {
                unlink(destination)
                throw CASError.io("read failed: \(errnoString())")
            }
            var written = 0
            while written < readCount {
                let n = buffer.withUnsafeBytes {
                    write(out, $0.baseAddress! + written, readCount - written)
                }
                guard n > 0 else {
                    unlink(destination)
                    throw CASError.io("write failed: \(errnoString())")
                }
                written += n
            }
        }
        _ = fcntl(out, F_FULLFSYNC)
    }

    /// True when `target` is a path directly inside the store's entries
    /// directory. Compares canonicalized parents so /tmp vs /private/tmp
    /// (and other symlinked roots) classify correctly, and sibling
    /// prefixes like <root>-evil never match.
    static func isStoreEntryPath(_ target: String) -> Bool {
        let parent = canonicalPath(
            (target as NSString).deletingLastPathComponent
        )
        let entriesDir = canonicalPath(
            storeRoot().appendingPathComponent("sha256").path
        )
        return parent == entriesDir
    }

    private static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path).resolvingSymlinksInPath().path
    }

    private static func fsyncDirectory(containing path: String) {
        let dir = (path as NSString).deletingLastPathComponent
        let fd = open(dir, O_RDONLY)
        guard fd >= 0 else { return }
        _ = fcntl(fd, F_FULLFSYNC)
        close(fd)
    }

    /// Atomically replace `path` with a symlink to `target`.
    private static func replaceWithSymlink(
        path: String, target: String
    ) throws {
        let tmp = "\(path).cas-tmp-\(getpid())"
        unlink(tmp)
        guard symlink(target, tmp) == 0 else {
            throw CASError.io("symlink failed: \(errnoString())")
        }
        guard rename(tmp, path) == 0 else {
            unlink(tmp)
            throw CASError.io("symlink swap failed: \(errnoString())")
        }
    }

    /// Serialize concurrent adopters of the same entry (two first-runs
    /// racing) with an flock'd sidecar.
    private static func withEntryLock(
        entry: String, _ body: () throws -> Void
    ) throws {
        let fd = open("\(entry).lock", O_CREAT | O_WRONLY, 0o644)
        guard fd >= 0 else {
            throw CASError.io("lock open failed: \(errnoString())")
        }
        defer { close(fd) }
        guard flock(fd, LOCK_EX) == 0 else {
            throw CASError.io("flock failed: \(errnoString())")
        }
        defer { flock(fd, LOCK_UN) }
        try body()
    }

    /// Hash the fd's contents from the start. Reading the fd, not the
    /// path, ties the digest to the pinned inode.
    private static func sha256Hex(fd: Int32) throws -> String {
        guard lseek(fd, 0, SEEK_SET) == 0 else {
            throw CASError.io("seek failed: \(errnoString())")
        }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: 1 << 20)
        while true {
            let readCount = buffer.withUnsafeMutableBytes {
                read(fd, $0.baseAddress, $0.count)
            }
            if readCount == 0 { break }
            guard readCount > 0 else {
                throw CASError.io("read failed: \(errnoString())")
            }
            buffer.withUnsafeBytes {
                hasher.update(
                    bufferPointer: UnsafeRawBufferPointer(
                        rebasing: $0.prefix(readCount)
                    )
                )
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Hex(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func errnoString() -> String {
        String(cString: strerror(errno))
    }
}
