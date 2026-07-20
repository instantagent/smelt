import Foundation

protocol SmeltFileTransformRuntime {
    func run(
        inputURL: URL,
        outputURL: URL,
        options: [String: String]
    ) throws -> SmeltPackageRunResult
}

struct SmeltFileTransformRegistration: Sendable {
    typealias Factory = @Sendable (String) throws -> any SmeltFileTransformRuntime

    let entrypoint: String
    let make: Factory
}

struct SmeltFileTransformRegistry: Sendable {
    private let factories: [String: SmeltFileTransformRegistration.Factory]

    init(registrations: [SmeltFileTransformRegistration]) {
        var factories: [String: SmeltFileTransformRegistration.Factory] = [:]
        for registration in registrations {
            precondition(
                factories[registration.entrypoint] == nil,
                "duplicate file-transform entrypoint \(registration.entrypoint)"
            )
            factories[registration.entrypoint] = registration.make
        }
        self.factories = factories
    }

    func make(
        entrypoint: String,
        packagePath: String
    ) throws -> any SmeltFileTransformRuntime {
        guard let factory = factories[entrypoint] else {
            throw SmeltFileTransformRegistryError.unknownEntrypoint(entrypoint)
        }
        return try factory(packagePath)
    }
}

enum SmeltFileTransformRegistryError: Error, CustomStringConvertible, Equatable {
    case unknownEntrypoint(String)

    var description: String {
        switch self {
        case .unknownEntrypoint(let entrypoint):
            return "no file-transform implementation is registered for '\(entrypoint)'"
        }
    }
}
