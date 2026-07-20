import Foundation

// Process signal handling is executable-adapter policy. SmeltServe transports
// deliberately do not install handlers or terminate their embedding process.
func installFatalSignalHandlers(label: String) {
    FatalSignalHandlers.shared.install(label: label)
}

private final class FatalSignalHandlers: @unchecked Sendable {
    static let shared = FatalSignalHandlers()

    private let lock = NSLock()
    private var label: String = "smelt"
    private var sources: [DispatchSourceSignal] = []
    private var installed = false

    private init() {}

    func install(label: String) {
        lock.lock(); defer { lock.unlock() }
        self.label = label
        guard !installed else { return }
        installed = true
        sources = [
            makeSource(SIGTERM, name: "SIGTERM"),
            makeSource(SIGINT, name: "SIGINT"),
        ]
    }

    private func makeSource(_ signum: Int32, name: String) -> DispatchSourceSignal {
        signal(signum, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signum, queue: .global())
        source.setEventHandler { [weak self] in
            let label = self?.lock.withLock { self?.label } ?? "smelt"
            fputs("\(label) received \(name); exiting.\n", stderr)
            exit(128 + signum)
        }
        source.resume()
        return source
    }
}

private extension NSLock {
    func withLock<R>(_ body: () -> R) -> R {
        lock(); defer { unlock() }
        return body()
    }
}
