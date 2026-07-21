// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SpecDecodeAcceptance",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(name: "Smelt", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "SpecDecodeAcceptance",
            dependencies: [
                .product(name: "SmeltRuntime", package: "Smelt")
            ],
            path: "Sources/SpecDecodeAcceptance"
        )
    ]
)
