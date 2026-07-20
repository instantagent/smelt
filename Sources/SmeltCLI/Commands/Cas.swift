import Foundation
import SmeltRuntime

// smelt cas — deduplicate large package artifacts through the shared
// content-addressed store (~/Library/Application Support/smelt/cas, or
// $SMELT_CAS_DIR). `adopt` moves a package's checksummed files into the
// store and leaves symlinks; packages built from the same base model then
// share one copy of weights.bin. `smelt run` adopts automatically on
// first run unless SMELT_CAS=0.

func runCasCommand() {
    let usage =
        "Usage: smelt cas <adopt|restore|status> <model.smeltpkg> "
        + "[--min-bytes N] [--quiet]\n"
    guard args.count >= 4 else {
        fputs(usage, stderr)
        exit(1)
    }
    let subcommand = args[2]
    let packagePath = args[3]
    let quiet = args.contains("--quiet")
    let minBytes = Int(parseArg("--min-bytes", default: ""))
        ?? SmeltCAS.defaultMinBytes

    do {
        let report: SmeltCAS.Report
        switch subcommand {
        case "adopt":
            report = try SmeltCAS.adopt(
                packagePath: packagePath, minBytes: minBytes
            )
        case "restore":
            report = try SmeltCAS.restore(packagePath: packagePath)
        case "status":
            report = try SmeltCAS.status(
                packagePath: packagePath, minBytes: minBytes
            )
        default:
            fputs(usage, stderr)
            exit(1)
        }
        if !quiet {
            printCasReport(report, subcommand: subcommand)
        }
    } catch {
        if !quiet {
            fputs("cas \(subcommand) failed: \(error)\n", stderr)
        }
        exit(1)
    }
}

private func printCasReport(_ report: SmeltCAS.Report, subcommand: String) {
    fputs("Store: \(SmeltCAS.storeRoot().path)\n", stderr)
    var sharedBytes: Int64 = 0
    for file in report.files where file.state != .missing {
        fputs(
            "  \(file.name.padding(toLength: 40, withPad: " ", startingAt: 0))"
                + " \(formatBytes(file.bytes).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + " \(file.state.rawValue)\n",
            stderr
        )
        if file.state == .adopted || file.state == .alreadyAdopted {
            sharedBytes += file.bytes
        }
    }
    if sharedBytes > 0 {
        fputs(
            "\(formatBytes(sharedBytes)) shared through the store; "
                + "identical artifacts in other packages dedup to it on "
                + "their adopt.\n",
            stderr
        )
    }
    if subcommand == "status", !report.adoptable.isEmpty {
        fputs(
            "Run `smelt cas adopt \(report.packagePath)` to deduplicate.\n",
            stderr
        )
    }
}

private func formatBytes(_ bytes: Int64) -> String {
    if bytes >= 1 << 30 {
        return String(format: "%.2f GB", Double(bytes) / Double(1 << 30))
    }
    if bytes >= 1 << 20 {
        return String(format: "%.1f MB", Double(bytes) / Double(1 << 20))
    }
    if bytes >= 1 << 10 {
        return String(format: "%.1f KB", Double(bytes) / Double(1 << 10))
    }
    return "\(bytes) B"
}

/// Spawn a detached, quiet `smelt cas adopt` for the package — the
/// transparent first-run dedup behind `smelt run`. Cheap pre-check only
/// (no manifest parse): spawn when any package file is still a regular
/// file of adoptable size. The running process is unaffected by the
/// swap: its open fd pins the inode, and each package path flips
/// atomically from regular file to symlink with no missing-path window.
func spawnCasAdoptIfUseful(packagePath: String) {
    if ProcessInfo.processInfo.environment["SMELT_CAS"] == "0" { return }
    let names = (try? FileManager.default
        .contentsOfDirectory(atPath: packagePath)) ?? []
    let hasAdoptable = names.contains { name in
        guard name != "manifest.json" else { return false }
        var st = stat()
        return lstat("\(packagePath)/\(name)", &st) == 0
            && (st.st_mode & S_IFMT) == S_IFREG
            && st.st_size >= SmeltCAS.defaultMinBytes
    }
    guard hasAdoptable else { return }
    guard let exe = Bundle.main.executablePath else { return }

    let arguments = [exe, "cas", "adopt", packagePath, "--quiet"]
    var attr: posix_spawnattr_t?
    posix_spawnattr_init(&attr)
    defer { posix_spawnattr_destroy(&attr) }
    posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETSID))

    var fileActions: posix_spawn_file_actions_t?
    posix_spawn_file_actions_init(&fileActions)
    defer { posix_spawn_file_actions_destroy(&fileActions) }
    posix_spawn_file_actions_addopen(&fileActions, 0, "/dev/null", O_RDONLY, 0)
    posix_spawn_file_actions_addopen(&fileActions, 1, "/dev/null", O_WRONLY, 0)
    posix_spawn_file_actions_addopen(&fileActions, 2, "/dev/null", O_WRONLY, 0)

    let argv: [UnsafeMutablePointer<CChar>?] =
        arguments.map { strdup($0) } + [nil]
    defer { for arg in argv { free(arg) } }
    var pid: pid_t = 0
    _ = posix_spawn(&pid, exe, &fileActions, &attr, argv, environ)
}
