import SmeltRuntime
import SmeltSchema
import Foundation

private struct DeclaredPackageRunInvocation {
    let input: String
    let output: String
    let options: [String: String]
}

/// Dispatches a package-authored file interface through its CAM-selected export
/// before text/audio routing. The CLI understands the contract, not the model.
func dispatchDeclaredPackageRunIfPresent(
    packagePath: String,
    promptStartIndex: Int,
    fullArgv: [String]
) -> Bool {
    let contract: SmeltPackageRunContract
    do {
        guard let declared = try SmeltPackageRunner.declaredContract(packagePath: packagePath)
        else { return false }
        contract = declared
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }

    let invocation: DeclaredPackageRunInvocation
    do {
        if packageRunHelpRequested(
            promptStartIndex: promptStartIndex,
            fullArgv: fullArgv
        ) {
            printPackageRunHelp(contract)
            exit(0)
        }
        invocation = try parsePackageRunInvocation(
            contract: contract,
            promptStartIndex: promptStartIndex,
            fullArgv: fullArgv
        )
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        printPackageRunHelp(contract, stream: stderr)
        exit(1)
    }

    do {
        let runner = try SmeltPackageRunner(packagePath: packagePath)
        let result = try runner.run(
            inputURL: URL(fileURLWithPath: invocation.input),
            outputURL: URL(fileURLWithPath: invocation.output),
            options: invocation.options
        )
        fputs("\(result.summary)\n", stderr)
        return true
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
}

private func packageRunHelpRequested(
    promptStartIndex: Int,
    fullArgv: [String]
) -> Bool {
    let start = min(max(0, promptStartIndex), fullArgv.count)
    return fullArgv[start...].contains("--help")
        || fullArgv[start...].contains("-h")
}

private func parsePackageRunInvocation(
    contract: SmeltPackageRunContract,
    promptStartIndex: Int,
    fullArgv: [String]
) throws -> DeclaredPackageRunInvocation {
    let inputFlag = "--\(contract.input.flag)"
    let outputFlag = "--\(contract.output.flag)"
    let optionFlags = Dictionary(uniqueKeysWithValues: contract.options.map {
        ("--\($0.flag)", $0.flag)
    })
    var input: String?
    var output: String?
    var options: [String: String] = [:]
    var index = min(max(0, promptStartIndex), fullArgv.count)

    while index < fullArgv.count {
        let argument = fullArgv[index]
        if argument == "--package" {
            guard index + 1 < fullArgv.count else {
                throw CLIError("--package requires a value")
            }
            index += 2
            continue
        }
        if argument == "--" {
            guard index + 1 == fullArgv.count else {
                throw CLIError("file-transform packages do not accept positional input")
            }
            break
        }
        let flag: String
        if argument == inputFlag {
            flag = contract.input.flag
        } else if argument == outputFlag {
            flag = contract.output.flag
        } else if let option = optionFlags[argument] {
            flag = option
        } else if argument.hasPrefix("-") {
            throw CLIError("unknown flag '\(argument)'")
        } else {
            throw CLIError("unexpected positional input '\(argument)'")
        }
        guard index + 1 < fullArgv.count else {
            throw CLIError("--\(flag) requires a value")
        }
        let value = fullArgv[index + 1]
        guard !value.hasPrefix("--") else {
            throw CLIError("--\(flag) requires a value")
        }
        if flag == contract.input.flag {
            guard input == nil else { throw CLIError("--\(flag) was provided more than once") }
            input = value
        } else if flag == contract.output.flag {
            guard output == nil else { throw CLIError("--\(flag) was provided more than once") }
            output = value
        } else {
            guard options[flag] == nil else {
                throw CLIError("--\(flag) was provided more than once")
            }
            options[flag] = value
        }
        index += 2
    }

    guard let input else { throw CLIError("missing required --\(contract.input.flag)") }
    guard let output else { throw CLIError("missing required --\(contract.output.flag)") }
    return DeclaredPackageRunInvocation(input: input, output: output, options: options)
}

private func printPackageRunHelp(
    _ contract: SmeltPackageRunContract,
    stream: UnsafeMutablePointer<FILE> = stdout
) {
    fputs(
        "Usage: smelt run <package.smeltpkg> --\(contract.input.flag) <file> "
            + "--\(contract.output.flag) <file> [options]\n",
        stream
    )
    fputs(
        "  --\(contract.input.flag)\t\(contract.input.help) "
            + "(.\(contract.input.fileExtensions.joined(separator: ", .")))\n",
        stream
    )
    fputs(
        "  --\(contract.output.flag)\t\(contract.output.help) "
            + "(.\(contract.output.fileExtensions.joined(separator: ", .")))\n",
        stream
    )
    for option in contract.options {
        let suffix = option.defaultValue.map { " (default: \($0))" } ?? ""
        fputs("  --\(option.flag)\t\(option.help)\(suffix)\n", stream)
    }
}
