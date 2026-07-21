// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MmapMetalCow",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "MmapMetalCow",
            path: "Sources/MmapMetalCow"
        )
    ]
)
