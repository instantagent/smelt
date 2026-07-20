import SmeltSchema
import Foundation

/// Result of executing a package-authored `smelt run` file transformation.
public struct SmeltPackageRunResult: Sendable, Equatable {
    public let outputURL: URL
    public let summary: String

    public init(outputURL: URL, summary: String) {
        self.outputURL = outputURL
        self.summary = summary
    }
}

/// Runtime dispatcher for packages that declare a file-transform run contract.
///
/// The public surface is model-agnostic. Concrete implementations are selected
/// by the entrypoint in the CAM-resolved flow.
public final class SmeltPackageRunner {
    private struct Header: Decodable {
        let run: SmeltPackageRunContract?
    }

    public let contract: SmeltPackageRunContract
    private let implementation: any SmeltFileTransformRuntime

    public static func declaredContract(packagePath: String) throws -> SmeltPackageRunContract? {
        let header = try loadHeader(packagePath: packagePath)
        try header.run?.validate()
        return header.run
    }

    public init(packagePath: String) throws {
        let header = try Self.loadHeader(packagePath: packagePath)
        guard let contract = header.run else {
            throw SmeltPackageRunnerError.missingRunContract
        }
        try contract.validate()
        self.contract = contract
        let packageURL = URL(fileURLWithPath: packagePath, isDirectory: true)
        guard let capabilities = try SmeltCAMPackageCapabilities.loadIfPresent(
            packageURL: packageURL
        ) else {
            throw SmeltPackageRunnerError.missingModuleDescriptor
        }
        let request = SmeltCAMCapabilityRequest.runFileTransform(
            inputName: contract.input.flag,
            inputMediaType: contract.input.mediaTypes[0],
            outputName: contract.output.flag,
            outputMediaType: contract.output.mediaTypes[0]
        )
        let decision = try capabilities.resolve(request)
        guard decision.exportID == contract.export else {
            throw SmeltPackageRunnerError.exportMismatch(
                expected: contract.export,
                selected: decision.exportID
            )
        }
        let graph = try capabilities.executionGraph(for: decision)
        let calls = graph.phases.flatMap(\.calls).filter {
            $0.entrypoint == contract.entrypoint
        }
        guard calls.count == 1,
              calls[0].callType == "node",
              let nodeID = calls[0].nodeID,
              graph.nodes.contains(where: {
                  $0.nodeID == nodeID && $0.implementation == "native"
              })
        else {
            throw SmeltPackageRunnerError.entrypointNotSelected(contract.entrypoint)
        }
        implementation = try SmeltBuiltInFileTransforms.registry.make(
            entrypoint: contract.entrypoint,
            packagePath: packagePath
        )
    }

    public func run(
        inputURL: URL,
        outputURL: URL,
        options suppliedOptions: [String: String]
    ) throws -> SmeltPackageRunResult {
        try validateFile(inputURL, port: contract.input)
        try validateFile(outputURL, port: contract.output)
        guard inputURL.standardizedFileURL != outputURL.standardizedFileURL else {
            throw SmeltPackageRunnerError.inputOverwritesOutput(inputURL.path)
        }
        let options = try resolvedOptions(suppliedOptions)

        return try implementation.run(
            inputURL: inputURL,
            outputURL: outputURL,
            options: options
        )
    }

    private static func loadHeader(packagePath: String) throws -> Header {
        let url = URL(fileURLWithPath: packagePath, isDirectory: true)
            .appendingPathComponent("manifest.json")
        do {
            return try JSONDecoder().decode(Header.self, from: Data(contentsOf: url))
        } catch {
            throw SmeltPackageRunnerError.invalidManifest("\(url.path): \(error)")
        }
    }

    private func resolvedOptions(_ supplied: [String: String]) throws -> [String: String] {
        let declarations = Dictionary(uniqueKeysWithValues: contract.options.map { ($0.flag, $0) })
        for flag in supplied.keys where declarations[flag] == nil {
            throw SmeltPackageRunnerError.unknownOption(flag)
        }
        var resolved = Dictionary(uniqueKeysWithValues: contract.options.compactMap { option in
            option.defaultValue.map { (option.flag, $0) }
        })
        for (flag, value) in supplied {
            guard !value.isEmpty else {
                throw SmeltPackageRunnerError.invalidOption(flag, value)
            }
            resolved[flag] = value
        }
        for option in contract.options {
            guard let value = resolved[option.flag] else { continue }
            switch option.value {
            case .string:
                break
            case .unsignedInteger:
                _ = try unsignedValue(value, flag: option.flag)
            case .positiveInteger:
                guard let parsed = Int(value), parsed > 0 else {
                    throw SmeltPackageRunnerError.invalidOption(option.flag, value)
                }
            }
        }
        return resolved
    }

    private func validateFile(_ url: URL, port: SmeltPackageRunContract.FilePort) throws {
        let fileExtension = url.pathExtension.lowercased()
        guard port.fileExtensions.contains(fileExtension) else {
            throw SmeltPackageRunnerError.unsupportedFileExtension(
                flag: port.flag,
                expected: port.fileExtensions,
                got: fileExtension
            )
        }
    }

    private func unsignedValue(_ value: String, flag: String) throws -> UInt64 {
        guard let parsed = UInt64(value) else {
            throw SmeltPackageRunnerError.invalidOption(flag, value)
        }
        return parsed
    }
}

public enum SmeltPackageRunnerError: Error, CustomStringConvertible, Equatable {
    case invalidManifest(String)
    case missingRunContract
    case missingModuleDescriptor
    case exportMismatch(expected: String, selected: String)
    case entrypointNotSelected(String)
    case unknownOption(String)
    case invalidOption(String, String)
    case missingOptionDefault(String)
    case unsupportedFileExtension(flag: String, expected: [String], got: String)
    case inputOverwritesOutput(String)

    public var description: String {
        switch self {
        case .invalidManifest(let message):
            return "invalid package manifest: \(message)"
        case .missingRunContract:
            return "package does not declare a run interface"
        case .missingModuleDescriptor:
            return "file-transform package does not contain module.json"
        case .exportMismatch(let expected, let selected):
            return "run contract export '\(expected)' differs from CAM-selected export "
                + "'\(selected)'"
        case .entrypointNotSelected(let entrypoint):
            return "run contract entrypoint '\(entrypoint)' is not selected by the CAM flow"
        case .unknownOption(let flag):
            return "package does not declare --\(flag)"
        case .invalidOption(let flag, let value):
            return "invalid --\(flag) value '\(value)'"
        case .missingOptionDefault(let flag):
            return "package omitted the required --\(flag) default"
        case .unsupportedFileExtension(let flag, let expected, let got):
            return "--\(flag) expects .\(expected.joined(separator: ", .")); got .\(got)"
        case .inputOverwritesOutput(let path):
            return "input and output resolve to the same file: \(path)"
        }
    }
}
