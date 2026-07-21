// swift-tools-version: 6.0

import PackageDescription
import Foundation

let whisperKitPath = ProcessInfo.processInfo.environment["WHISPERKIT_PATH"]

var packageDependencies: [Package.Dependency] = []
var targetDependencies: [Target.Dependency] = []

if let whisperKitPath, !whisperKitPath.isEmpty {
    packageDependencies.append(.package(name: "whisperkit", path: whisperKitPath))
    targetDependencies.append(.product(name: "WhisperKit", package: "whisperkit"))
}

let package = Package(
    name: "WhisperTinyDecoderSpike",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "WhisperTinyDecoderSpike",
            targets: ["WhisperTinyDecoderSpike"]
        )
    ],
    dependencies: packageDependencies,
    targets: [
        .target(
            name: "WhisperTinyDecoderSpike",
            dependencies: targetDependencies
        ),
        .testTarget(
            name: "WhisperTinyDecoderSpikeTests",
            dependencies: ["WhisperTinyDecoderSpike"]
        )
    ]
)
