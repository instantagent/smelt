import Foundation

struct SmeltGraphValue {
    private let storage: Any

    init<T>(_ value: T) {
        storage = value
    }

    func cast<T>(_ type: T.Type = T.self) -> T? {
        storage as? T
    }
}

final class SmeltGraphExecutionContext {
    struct StageTiming: Encodable {
        let node: String
        let entrypoint: String
        let wallMilliseconds: Double
    }

    let packagePath: String
    let inputURL: URL
    let outputURL: URL
    let options: [String: String]
    let capturesEvidence: Bool
    private(set) var summary = ""
    private(set) var timings: [StageTiming] = []
    private(set) var evidence: [String: String] = [:]
    private var resources: [String: Any] = [:]

    init(
        packagePath: String,
        inputURL: URL,
        outputURL: URL,
        options: [String: String]
    ) {
        self.packagePath = packagePath
        self.inputURL = inputURL
        self.outputURL = outputURL
        self.options = options
        capturesEvidence = !(ProcessInfo.processInfo.environment["SMELT_STAGE_RECEIPT"] ?? "")
            .isEmpty
    }

    func resource<T>(
        _ key: String,
        make: () throws -> T
    ) throws -> T {
        if let existing = resources[key] {
            guard let typed = existing as? T else {
                throw SmeltGraphValueError.resourceTypeMismatch(key)
            }
            return typed
        }
        let value = try make()
        resources[key] = value
        return value
    }

    func setSummary(_ value: String) {
        summary = value
    }

    func appendTiming(node: String, entrypoint: String, wallMilliseconds: Double) {
        timings.append(
            .init(
                node: node,
                entrypoint: entrypoint,
                wallMilliseconds: wallMilliseconds
            )
        )
    }

    func recordEvidence(_ key: String, value: String) {
        if capturesEvidence {
            evidence[key] = value
        }
    }
}

protocol SmeltGraphNodeRuntime {
    func run(
        inputs: [String: SmeltGraphValue],
        context: SmeltGraphExecutionContext
    ) throws -> [String: SmeltGraphValue]
}

struct SmeltGraphNodeRegistration: Sendable {
    typealias Factory = @Sendable () -> any SmeltGraphNodeRuntime

    let entrypoint: String
    let make: Factory
}

struct SmeltGraphNodeRegistry: Sendable {
    private let factories: [String: SmeltGraphNodeRegistration.Factory]

    init(registrations: [SmeltGraphNodeRegistration]) {
        var factories: [String: SmeltGraphNodeRegistration.Factory] = [:]
        for registration in registrations {
            precondition(
                factories[registration.entrypoint] == nil,
                "duplicate graph node entrypoint \(registration.entrypoint)"
            )
            factories[registration.entrypoint] = registration.make
        }
        self.factories = factories
    }

    func make(entrypoint: String) throws -> any SmeltGraphNodeRuntime {
        guard let factory = factories[entrypoint] else {
            throw SmeltGraphNodeRegistryError.unknownEntrypoint(entrypoint)
        }
        return factory()
    }
}

enum SmeltGraphNodeRegistryError: Error, CustomStringConvertible, Equatable {
    case unknownEntrypoint(String)

    var description: String {
        switch self {
        case .unknownEntrypoint(let entrypoint):
            return "no graph node implementation is registered for '\(entrypoint)'"
        }
    }
}

enum SmeltGraphValueError: Error, Equatable {
    case missing(node: String, port: String)
    case wrongType(node: String, port: String)
    case resourceTypeMismatch(String)
}

extension Dictionary where Key == String, Value == SmeltGraphValue {
    func required<T>(_ port: String, as type: T.Type, node: String) throws -> T {
        guard let value = self[port] else {
            throw SmeltGraphValueError.missing(node: node, port: port)
        }
        guard let typed = value.cast(type) else {
            throw SmeltGraphValueError.wrongType(node: node, port: port)
        }
        return typed
    }
}
