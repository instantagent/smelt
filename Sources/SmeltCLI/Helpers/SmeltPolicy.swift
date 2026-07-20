import Foundation
import SmeltRuntime

func resolveSelectionMode(
    tempArg: String,
    seedArg: String
) throws -> (mode: SmeltSelectionMode, description: String) {
    let temperature = Float(tempArg) ?? 0
    guard temperature >= 0 else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "--temp must be non-negative"
            ]
        )
    }

    guard temperature > 0 else {
        return (.argmax, "argmax")
    }

    let seed: UInt64
    if seedArg.isEmpty {
        seed = UInt64.random(in: UInt64.min...UInt64.max)
    } else if let parsed = UInt64(seedArg) {
        seed = parsed
    } else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: "--seed must be a uint64"
            ]
        )
    }

    return (
        .temperature(temperature, seed: seed),
        String(format: "temperature=%.3f seed=%llu", temperature, seed)
    )
}
