// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PagedAttention",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "PagedAttention",
            path: "Sources/PagedAttention"
        )
    ]
)
