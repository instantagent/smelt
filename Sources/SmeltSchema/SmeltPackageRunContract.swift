/// Package-authored declaration for a file-in/file-out `smelt run` interface.
///
/// The CLI reads this contract without knowing the model family. The package
/// owns its accepted files and options; the runtime owns execution.
public struct SmeltPackageRunContract: Codable, Sendable, Equatable {
    public static let currentVersion = 2

    public enum Interface: String, Codable, Sendable, Equatable {
        case fileTransform = "file-transform"
    }

    public enum OptionValue: String, Codable, Sendable, Equatable {
        case string
        case unsignedInteger = "unsigned-integer"
        case positiveInteger = "positive-integer"
    }

    public struct FilePort: Codable, Sendable, Equatable {
        public let flag: String
        public let mediaTypes: [String]
        public let fileExtensions: [String]
        public let help: String

        public init(
            flag: String,
            mediaTypes: [String],
            fileExtensions: [String],
            help: String
        ) {
            self.flag = flag
            self.mediaTypes = mediaTypes
            self.fileExtensions = fileExtensions
            self.help = help
        }
    }

    public struct Option: Codable, Sendable, Equatable {
        public let flag: String
        public let value: OptionValue
        public let defaultValue: String?
        public let help: String

        public init(
            flag: String,
            value: OptionValue,
            defaultValue: String? = nil,
            help: String
        ) {
            self.flag = flag
            self.value = value
            self.defaultValue = defaultValue
            self.help = help
        }
    }

    public let version: Int
    public let interface: Interface
    public let export: String
    public let entrypoint: String
    public let input: FilePort
    public let output: FilePort
    public let options: [Option]

    public init(
        version: Int = Self.currentVersion,
        interface: Interface = .fileTransform,
        export: String,
        entrypoint: String,
        input: FilePort,
        output: FilePort,
        options: [Option] = []
    ) {
        self.version = version
        self.interface = interface
        self.export = export
        self.entrypoint = entrypoint
        self.input = input
        self.output = output
        self.options = options
    }

    public func validate() throws {
        guard version == Self.currentVersion else {
            throw SmeltPackageRunContractError.invalid("unsupported version \(version)")
        }
        guard Self.isValidIdentifier(export), Self.isValidIdentifier(entrypoint) else {
            throw SmeltPackageRunContractError.invalid("invalid export or entrypoint")
        }
        guard Self.isValidFlag(input.flag), Self.isValidFlag(output.flag),
              input.flag != output.flag
        else {
            throw SmeltPackageRunContractError.invalid("invalid or duplicated file-port flag")
        }
        for port in [input, output] {
            guard port.mediaTypes.count == 1,
                  !port.fileExtensions.isEmpty,
                  !port.help.isEmpty,
                  port.mediaTypes.allSatisfy({ !$0.isEmpty }),
                  port.fileExtensions.allSatisfy({
                      !$0.isEmpty && !$0.hasPrefix(".") && $0 == $0.lowercased()
                  })
            else {
                throw SmeltPackageRunContractError.invalid(
                    "invalid file-port declaration for --\(port.flag)"
                )
            }
        }

        var flags = Set([input.flag, output.flag])
        for option in options {
            guard Self.isValidFlag(option.flag),
                  flags.insert(option.flag).inserted,
                  !option.help.isEmpty
            else {
                throw SmeltPackageRunContractError.invalid(
                    "invalid or duplicated option flag --\(option.flag)"
                )
            }
            guard let defaultValue = option.defaultValue else { continue }
            switch option.value {
            case .string:
                break
            case .unsignedInteger:
                guard UInt64(defaultValue) != nil else {
                    throw SmeltPackageRunContractError.invalid(
                        "invalid default for --\(option.flag)"
                    )
                }
            case .positiveInteger:
                guard let value = Int(defaultValue), value > 0 else {
                    throw SmeltPackageRunContractError.invalid(
                        "invalid default for --\(option.flag)"
                    )
                }
            }
        }
    }

    private static func isValidFlag(_ flag: String) -> Bool {
        guard let first = flag.first, first.isLowercase else { return false }
        return flag.allSatisfy { $0.isLowercase || $0.isNumber || $0 == "-" }
    }

    private static func isValidIdentifier(_ value: String) -> Bool {
        guard let first = value.first, first.isLetter || first == "_" else {
            return false
        }
        return value.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" || $0 == "."
        }
    }
}

public enum SmeltPackageRunContractError: Error, Equatable {
    case invalid(String)
}
