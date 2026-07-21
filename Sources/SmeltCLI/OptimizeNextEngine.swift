import Foundation
import SmeltRuntime

private struct SmeltOptimizeNextError: Error, CustomStringConvertible {
    let description: String
}

private struct SmeltOptimizeNextOptions {
    var packagePath: String
    var verifierCommand: String
    var buildCommand: String?
    var worktreePath: String?
    var agentCommand: String
    var taskID: String?
    var applyOnPass: Bool
    var dryRun: Bool
    var seedPackage: Bool
    var forcePackageBuild: Bool
    var skipPackageBuild: Bool
    var reportDirectory: String?
}

private struct SmeltProcessCaptureResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
}

private struct SmeltLoggedProcessResult {
    let exitCode: Int32
    let stdoutPath: String
    let stderrPath: String
}

func optimizeNextStatus(arguments args: [String]) throws -> Int32 {
    let options = try parseOptimizeNextOptions(arguments: args)
    let fm = FileManager.default
    let packagePath = absolutePath(options.packagePath, relativeTo: fm.currentDirectoryPath)
    let camContext = requireCAMOptimizerReportAdmissionOrExit(
        packagePath: packagePath,
        verb: "optimize-next"
    )
    guard let buildCommand = options.buildCommand else {
        throw SmeltOptimizeNextError(
            description: "optimize-next requires --build-command CMD"
        )
    }
    let repoRoot = try gitOutput(["rev-parse", "--show-toplevel"])
    guard let packageRelativePath = relativePath(packagePath, inside: repoRoot) else {
        throw SmeltOptimizeNextError(
            description: "optimize-next requires the package path to be inside the git repository"
        )
    }

    let beforeReport = try SmeltOptimizerReportGenerator.markdown(
        packagePath: packagePath,
        camContext: camContext
    )
    let tasks = try SmeltOptimizerReportGenerator.agentTasks(packagePath: packagePath)
    guard !tasks.isEmpty else {
        print("NO_TASKS: optimizer report contains no missing fused-kernel tasks")
        return 0
    }
    let selectedTask: SmeltOptimizerAgentTask
    let selectedPriority: Int
    if let taskID = options.taskID {
        guard let found = tasks.enumerated().first(where: { $0.element.id == taskID }) else {
            throw SmeltOptimizeNextError(
                description: "task id \(taskID) was not found in the optimizer report"
            )
        }
        selectedTask = found.element
        selectedPriority = found.offset + 1
    } else {
        selectedTask = tasks[0]
        selectedPriority = 1
    }

    let timestamp = optimizeNextTimestamp()
    let runSlug = optimizeNextSlug(selectedTask.id)
    let worktreePath = absolutePath(
        options.worktreePath
            ?? "\(NSTemporaryDirectory())smelt-optimize-\(runSlug)-\(timestamp)",
        relativeTo: repoRoot
    )
    let reportDirectory = absolutePath(
        options.reportDirectory
            ?? "\(NSTemporaryDirectory())smelt-optimize-runs/\(timestamp)-\(runSlug)",
        relativeTo: repoRoot
    )
    let branchName = "smelt-optimize/\(runSlug)-\(timestamp)"
    let prompt = makeOptimizeNextPrompt(
        packageRelativePath: packageRelativePath,
        task: selectedTask,
        taskPriority: selectedPriority,
        beforeReport: beforeReport,
        buildCommand: buildCommand,
        verifierCommand: options.verifierCommand
    )

    if options.dryRun {
        print("OPTIMIZE-NEXT DRY RUN")
        print("Task ID: \(selectedTask.id)")
        print("Task: \(selectedTask.title)")
        print("Package: \(packageRelativePath)")
        print("Worktree: \(worktreePath)")
        print("Report directory: \(reportDirectory)")
        print("Agent command: \(options.agentCommand) --cd \(shellQuote(worktreePath)) -")
        print("Build command: \(buildCommand)")
        print("Verifier command: \(options.verifierCommand)")
        print("Seed package: \(options.seedPackage ? "yes" : "no")")
        print("Package rebuild policy: \(packageBuildPolicyDescription(options))")
        return 0
    }

    if options.applyOnPass {
        let parentStatus = try gitOutput(["status", "--porcelain"], workingDirectory: repoRoot)
        guard parentStatus.isEmpty else {
            throw SmeltOptimizeNextError(
                description: "--apply-on-pass requires the current worktree to be clean"
            )
        }
    }

    guard !fm.fileExists(atPath: worktreePath) else {
        throw SmeltOptimizeNextError(description: "worktree path already exists: \(worktreePath)")
    }
    try fm.createDirectory(
        atPath: reportDirectory,
        withIntermediateDirectories: true
    )
    try beforeReport.write(
        toFile: "\(reportDirectory)/optimizer-report.before.md",
        atomically: true,
        encoding: .utf8
    )
    try selectedTask.markdownCard(priority: selectedPriority).write(
        toFile: "\(reportDirectory)/task.md",
        atomically: true,
        encoding: .utf8
    )
    try prompt.write(
        toFile: "\(reportDirectory)/prompt.md",
        atomically: true,
        encoding: .utf8
    )

    try checkedGit(["worktree", "add", "-b", branchName, worktreePath, "HEAD"], cwd: repoRoot)
    if options.seedPackage {
        let seededPath = "\(worktreePath)/\(packageRelativePath)"
        try fm.createDirectory(
            atPath: URL(fileURLWithPath: seededPath).deletingLastPathComponent().path,
            withIntermediateDirectories: true
        )
        if fm.fileExists(atPath: seededPath) {
            try fm.removeItem(atPath: seededPath)
        }
        try fm.copyItem(atPath: packagePath, toPath: seededPath)
    }

    let agent = try runShellLogged(
        "\(options.agentCommand) --cd \(shellQuote(worktreePath)) -",
        workingDirectory: repoRoot,
        stdin: prompt,
        stdoutPath: "\(reportDirectory)/agent.stdout.log",
        stderrPath: "\(reportDirectory)/agent.stderr.log"
    )
    guard agent.exitCode == 0 else {
        try writeOptimizeNextResult(
            directory: reportDirectory,
            status: "FAIL",
            task: selectedTask,
            details: "agent command failed with exit code \(agent.exitCode)"
        )
        print("FAIL: agent command failed; logs in \(reportDirectory)")
        return agent.exitCode == 0 ? 1 : agent.exitCode
    }

    let diffCheck = try runShellLogged(
        "git add -N . && git diff --check",
        workingDirectory: worktreePath,
        stdoutPath: "\(reportDirectory)/diff-check.stdout.log",
        stderrPath: "\(reportDirectory)/diff-check.stderr.log"
    )
    guard diffCheck.exitCode == 0 else {
        try writeOptimizeNextResult(
            directory: reportDirectory,
            status: "FAIL",
            task: selectedTask,
            details: "git diff --check failed"
        )
        print("FAIL: git diff --check failed; logs in \(reportDirectory)")
        return diffCheck.exitCode
    }

    let toolBuild = try runShellLogged(
        "swift build -c release",
        workingDirectory: worktreePath,
        stdoutPath: "\(reportDirectory)/swift-build-release.stdout.log",
        stderrPath: "\(reportDirectory)/swift-build-release.stderr.log"
    )
    guard toolBuild.exitCode == 0 else {
        try writeOptimizeNextResult(
            directory: reportDirectory,
            status: "FAIL",
            task: selectedTask,
            details: "release tool build failed"
        )
        print("FAIL: release tool build failed; logs in \(reportDirectory)")
        return toolBuild.exitCode
    }

    let changedFiles = try changedPaths(workingDirectory: worktreePath)
    let shouldBuildPackage = shouldRunPackageBuild(
        changedFiles: changedFiles,
        force: options.forcePackageBuild,
        skip: options.skipPackageBuild
    )
    if shouldBuildPackage {
        let packageBuild = try runShellLogged(
            buildCommand,
            workingDirectory: worktreePath,
            stdoutPath: "\(reportDirectory)/package-build.stdout.log",
            stderrPath: "\(reportDirectory)/package-build.stderr.log"
        )
        guard packageBuild.exitCode == 0 else {
            try writeOptimizeNextResult(
                directory: reportDirectory,
                status: "FAIL",
                task: selectedTask,
                details: "package build failed"
            )
            print("FAIL: package build failed; logs in \(reportDirectory)")
            return packageBuild.exitCode
        }
    } else {
        try "Skipped package build. Changed paths did not touch package-affecting inputs.\n"
            .write(
                toFile: "\(reportDirectory)/package-build.skipped.log",
                atomically: true,
                encoding: .utf8
            )
    }

    let verifier = try runShellLogged(
        options.verifierCommand,
        workingDirectory: worktreePath,
        stdoutPath: "\(reportDirectory)/verifier.stdout.log",
        stderrPath: "\(reportDirectory)/verifier.stderr.log"
    )
    guard verifier.exitCode == 0 else {
        try writeOptimizeNextResult(
            directory: reportDirectory,
            status: "FAIL",
            task: selectedTask,
            details: "verifier failed"
        )
        print("FAIL: verifier failed; logs in \(reportDirectory)")
        return verifier.exitCode
    }

    let afterReportPath = "\(reportDirectory)/optimizer-report.after.md"
    let afterReport = try runShellLogged(
        ".build/release/smelt lab inspect cost \(shellQuote(packageRelativePath)) --output \(shellQuote(afterReportPath))",
        workingDirectory: worktreePath,
        stdoutPath: "\(reportDirectory)/optimizer-report.stdout.log",
        stderrPath: "\(reportDirectory)/optimizer-report.stderr.log"
    )
    guard afterReport.exitCode == 0 else {
        try writeOptimizeNextResult(
            directory: reportDirectory,
            status: "FAIL",
            task: selectedTask,
            details: "post-run optimizer report failed"
        )
        print("FAIL: post-run optimizer report failed; logs in \(reportDirectory)")
        return afterReport.exitCode
    }

    let afterReportText = try String(contentsOfFile: afterReportPath, encoding: .utf8)
    let solved = !afterReportText.contains(selectedTask.id)
    if solved, options.applyOnPass {
        try applyPassingWorktreePatch(
            worktreePath: worktreePath,
            repoRoot: repoRoot,
            reportDirectory: reportDirectory
        )
    }

    let status = solved ? "PASS_SOLVED" : "PASS_UNSOLVED"
    try writeOptimizeNextResult(
        directory: reportDirectory,
        status: status,
        task: selectedTask,
        details: solved
            ? "verifier passed and task id disappeared"
            : "verifier passed but task id is still present"
    )
    print("\(status): \(selectedTask.id)")
    print("Worktree: \(worktreePath)")
    print("Run record: \(reportDirectory)")
    return solved ? 0 : 2
}

private func parseOptimizeNextOptions(arguments args: [String]) throws -> SmeltOptimizeNextOptions {
    guard args.count >= 3 else {
        throw SmeltOptimizeNextError(description: optimizeNextUsage())
    }
    var packagePath: String?
    var verifierCommand: String?
    var buildCommand: String?
    var worktreePath: String?
    // optimize-next already isolates the agent in a throwaway git worktree and only
    // applies a patch after verifier success. Avoid Codex's own filesystem sandbox
    // here because SwiftPM/Metal builds need user caches and may invoke sandbox-exec.
    var agentCommand = "codex exec --dangerously-bypass-approvals-and-sandbox"
    var taskID: String?
    var applyOnPass = false
    var dryRun = false
    var seedPackage = true
    var forcePackageBuild = false
    var skipPackageBuild = false
    var reportDirectory: String?

    var idx = 2
    while idx < args.count {
        let arg = args[idx]
        switch arg {
        case "--verifier":
            verifierCommand = try parseValue(args, &idx, flag: arg)
        case "--build-command":
            buildCommand = try parseValue(args, &idx, flag: arg)
        case "--worktree":
            worktreePath = try parseValue(args, &idx, flag: arg)
        case "--agent-command":
            agentCommand = try parseValue(args, &idx, flag: arg)
        case "--task-id":
            taskID = try parseValue(args, &idx, flag: arg)
        case "--report-dir":
            reportDirectory = try parseValue(args, &idx, flag: arg)
        case "--apply-on-pass":
            applyOnPass = true
            idx += 1
        case "--dry-run":
            dryRun = true
            idx += 1
        case "--no-seed-package":
            seedPackage = false
            idx += 1
        case "--force-package-build":
            forcePackageBuild = true
            idx += 1
        case "--skip-package-build":
            skipPackageBuild = true
            idx += 1
        case "--help", "-h":
            throw SmeltOptimizeNextError(description: optimizeNextUsage())
        default:
            if arg.hasPrefix("--") {
                throw SmeltOptimizeNextError(description: "unknown optimize-next option \(arg)")
            }
            guard packagePath == nil else {
                throw SmeltOptimizeNextError(description: optimizeNextUsage())
            }
            packagePath = arg
            idx += 1
        }
    }

    guard let packagePath else {
        throw SmeltOptimizeNextError(description: optimizeNextUsage())
    }
    guard let verifierCommand else {
        throw SmeltOptimizeNextError(description: "optimize-next requires --verifier CMD")
    }
    guard !(forcePackageBuild && skipPackageBuild) else {
        throw SmeltOptimizeNextError(
            description: "--force-package-build and --skip-package-build are mutually exclusive"
        )
    }

    return SmeltOptimizeNextOptions(
        packagePath: packagePath,
        verifierCommand: verifierCommand,
        buildCommand: buildCommand,
        worktreePath: worktreePath,
        agentCommand: agentCommand,
        taskID: taskID,
        applyOnPass: applyOnPass,
        dryRun: dryRun,
        seedPackage: seedPackage,
        forcePackageBuild: forcePackageBuild,
        skipPackageBuild: skipPackageBuild,
        reportDirectory: reportDirectory
    )
}

private func parseValue(_ args: [String], _ idx: inout Int, flag: String) throws -> String {
    guard idx + 1 < args.count else {
        throw SmeltOptimizeNextError(description: "\(flag) requires a value")
    }
    let value = args[idx + 1]
    idx += 2
    return value
}

private func optimizeNextUsage() -> String {
    "Usage: smelt lab optimize <model.smeltpkg> --verifier CMD --build-command CMD [--worktree DIR] [--agent-command CMD] [--task-id ID] [--apply-on-pass] [--dry-run] [--no-seed-package] [--force-package-build|--skip-package-build] [--report-dir DIR]"
}

private func makeOptimizeNextPrompt(
    packageRelativePath: String,
    task: SmeltOptimizerAgentTask,
    taskPriority: Int,
    beforeReport: String,
    buildCommand: String,
    verifierCommand: String
) -> String {
    """
    You are working in an isolated git worktree for Smelt.

    Implement exactly one compiler-emitted optimizer task. Keep the change narrowly scoped, do not commit, and do not edit unrelated files. Generated model artifacts are allowed only as outputs of the build command.

    Package under test: `\(packageRelativePath)`

    Required success condition:
    - The task ID below disappears from a regenerated `smelt lab inspect cost`.
    - The verifier command below passes.
    - Structure gates and performance gates still pass.

    Commands the harness will run after your attempt:
    - `git add -N . && git diff --check`
    - `swift build -c release`
    - package build only if package-affecting inputs changed: `\(buildCommand)`
    - `\(verifierCommand)`
    - `.build/release/smelt lab inspect cost \(packageRelativePath)`

    \(task.markdownCard(priority: taskPriority))

    Full optimizer report before your attempt:

    \(beforeReport)
    """
}

private func packageBuildPolicyDescription(_ options: SmeltOptimizeNextOptions) -> String {
    if options.forcePackageBuild { return "always" }
    if options.skipPackageBuild { return "never" }
    return "only when package-affecting paths changed"
}

private func shouldRunPackageBuild(
    changedFiles: [String],
    force: Bool,
    skip: Bool
) -> Bool {
    if force { return true }
    if skip { return false }
    return changedFiles.contains(where: packageAffectingPath)
}

private func packageAffectingPath(_ path: String) -> Bool {
    path == "Package.swift"
        || path.hasPrefix("Sources/SmeltCompiler/")
        || path.hasPrefix("Sources/SmeltSchema/")
        || path.hasPrefix("Resources/Shaders/")
        || (path.hasPrefix("tools/") && path.hasSuffix(".sh"))
}

private func changedPaths(workingDirectory: String) throws -> [String] {
    let tracked = try gitOutput(["diff", "--name-only", "HEAD"], workingDirectory: workingDirectory)
    let untracked = try gitOutput(
        ["ls-files", "--others", "--exclude-standard"],
        workingDirectory: workingDirectory
    )
    return (tracked.split(separator: "\n") + untracked.split(separator: "\n"))
        .map(String.init)
        .filter { !$0.isEmpty }
}

private func applyPassingWorktreePatch(
    worktreePath: String,
    repoRoot: String,
    reportDirectory: String
) throws {
    let patchPath = "\(reportDirectory)/passing-worktree.patch"
    let export = try runShellLogged(
        "git add -A && git diff --cached --binary > \(shellQuote(patchPath))",
        workingDirectory: worktreePath,
        stdoutPath: "\(reportDirectory)/apply-export.stdout.log",
        stderrPath: "\(reportDirectory)/apply-export.stderr.log"
    )
    guard export.exitCode == 0 else {
        throw SmeltOptimizeNextError(description: "failed to export passing worktree patch")
    }
    let attrs = try FileManager.default.attributesOfItem(atPath: patchPath)
    let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
    guard size > 0 else {
        throw SmeltOptimizeNextError(description: "passing worktree had no patch to apply")
    }
    let apply = try runShellLogged(
        "git apply \(shellQuote(patchPath))",
        workingDirectory: repoRoot,
        stdoutPath: "\(reportDirectory)/apply-parent.stdout.log",
        stderrPath: "\(reportDirectory)/apply-parent.stderr.log"
    )
    guard apply.exitCode == 0 else {
        throw SmeltOptimizeNextError(description: "failed to apply passing patch to parent worktree")
    }
}

private func writeOptimizeNextResult(
    directory: String,
    status: String,
    task: SmeltOptimizerAgentTask,
    details: String
) throws {
    let text = """
    # Smelt Optimize Next Result

    Status: \(status)
    Task ID: `\(task.id)`
    Task: \(task.title)
    Details: \(details)
    """
    try text.write(
        toFile: "\(directory)/result.md",
        atomically: true,
        encoding: .utf8
    )
}

private func runShellLogged(
    _ command: String,
    workingDirectory: String,
    stdin: String? = nil,
    stdoutPath: String,
    stderrPath: String
) throws -> SmeltLoggedProcessResult {
    try FileManager.default.createDirectory(
        atPath: URL(fileURLWithPath: stdoutPath).deletingLastPathComponent().path,
        withIntermediateDirectories: true
    )
    try "command: \(command)\n\n".write(toFile: stdoutPath, atomically: true, encoding: .utf8)
    try "".write(toFile: stderrPath, atomically: true, encoding: .utf8)
    let stdout = try FileHandle(forWritingTo: URL(fileURLWithPath: stdoutPath))
    let stderr = try FileHandle(forWritingTo: URL(fileURLWithPath: stderrPath))
    try stdout.seekToEnd()
    defer {
        try? stdout.close()
        try? stderr.close()
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/zsh")
    process.arguments = ["-lc", command]
    process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
    process.standardOutput = stdout
    process.standardError = stderr
    let inputPipe = Pipe()
    if stdin != nil {
        process.standardInput = inputPipe
    }
    try process.run()
    if let stdin {
        inputPipe.fileHandleForWriting.write(Data(stdin.utf8))
        try inputPipe.fileHandleForWriting.close()
    }
    process.waitUntilExit()
    return SmeltLoggedProcessResult(
        exitCode: process.terminationStatus,
        stdoutPath: stdoutPath,
        stderrPath: stderrPath
    )
}

private func gitOutput(
    _ arguments: [String],
    workingDirectory: String = FileManager.default.currentDirectoryPath
) throws -> String {
    let result = try runCapture(
        executable: "/usr/bin/env",
        arguments: ["git"] + arguments,
        workingDirectory: workingDirectory
    )
    guard result.exitCode == 0 else {
        throw SmeltOptimizeNextError(description: result.stderr)
    }
    return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
}

private func checkedGit(_ arguments: [String], cwd: String) throws {
    let result = try runCapture(
        executable: "/usr/bin/env",
        arguments: ["git"] + arguments,
        workingDirectory: cwd
    )
    guard result.exitCode == 0 else {
        throw SmeltOptimizeNextError(description: result.stderr)
    }
}

private func runCapture(
    executable: String,
    arguments: [String],
    workingDirectory: String
) throws -> SmeltProcessCaptureResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    let out = String(
        data: stdout.fileHandleForReading.readDataToEndOfFile(),
        encoding: .utf8
    ) ?? ""
    let err = String(
        data: stderr.fileHandleForReading.readDataToEndOfFile(),
        encoding: .utf8
    ) ?? ""
    return SmeltProcessCaptureResult(
        exitCode: process.terminationStatus,
        stdout: out,
        stderr: err
    )
}

private func absolutePath(_ path: String, relativeTo root: String) -> String {
    if path.hasPrefix("/") {
        return URL(fileURLWithPath: path).standardized.path
    }
    return URL(fileURLWithPath: root)
        .appendingPathComponent(path)
        .standardized
        .path
}

private func relativePath(_ path: String, inside root: String) -> String? {
    let standardizedPath = URL(fileURLWithPath: path).standardized.path
    let standardizedRoot = URL(fileURLWithPath: root).standardized.path
    guard standardizedPath == standardizedRoot
        || standardizedPath.hasPrefix(standardizedRoot + "/")
    else {
        return nil
    }
    if standardizedPath == standardizedRoot { return "." }
    return String(standardizedPath.dropFirst(standardizedRoot.count + 1))
}

private func shellQuote(_ value: String) -> String {
    "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
}

private func optimizeNextTimestamp() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: Date())
        .replacingOccurrences(of: ":", with: "")
        .replacingOccurrences(of: "-", with: "")
}

private func optimizeNextSlug(_ taskID: String) -> String {
    guard taskID.count > 64 else { return taskID }
    return "\(taskID.prefix(48))-\(taskID.suffix(12))"
}
