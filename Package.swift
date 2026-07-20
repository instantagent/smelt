// swift-tools-version: 6.0
import Foundation
import PackageDescription

// llguidance (constrained tool-call decoding) is a prebuilt binary by default —
// no Rust toolchain needed. `bash tools/build-llguidance.sh` builds it from
// source instead (upstream tag + tools/llguidance-serialize.patch); its local
// xcframework takes precedence when present. When bumping LLG_REF or the
// patch, re-run the script with --package, pin the new checksum below, and
// upload the zip to the matching public instantagent/binaries release tag.
let localLLGuidance = "third_party/llguidance/CLLGuidance.xcframework"
let llguidanceTarget: Target = FileManager.default.fileExists(
    atPath: Context.packageDirectory + "/" + localLLGuidance)
    ? .binaryTarget(name: "CLLGuidance", path: localLLGuidance)
    : .binaryTarget(
        name: "CLLGuidance",
        url: "https://github.com/instantagent/binaries/releases/download/llguidance-v1.7.4-agent1/CLLGuidance.xcframework.zip",
        checksum: "10b2b69b9e2c88cb8cdb4a574ad141988f103221abdf75b3069c8fcee329686b"
    )

// Private release-evidence and source-migration scanners are valuable in the
// maintainer gate, but compiling them adds roughly 20K lines to every ordinary
// `swift test` invocation. Keep them out of the default target graph entirely;
// tools/verify-release.sh opts them back in for the authoritative gate.
let includeMaintainerTests =
    ProcessInfo.processInfo.environment["SMELT_INCLUDE_MAINTAINER_TESTS"] == "1"
func existingTestExcludes(target: String, names: [String]) -> [String] {
    names.filter { name in
        FileManager.default.fileExists(
            atPath: Context.packageDirectory
                + "/Tests/\(target)/\(name)"
        )
    }
}
let compilerTestExcludes = includeMaintainerTests ? [] : existingTestExcludes(
    target: "SmeltCompilerTests",
    names: ["CAMModuleCompletionMatrixTests.swift"]
)
let runtimeTestExcludes = includeMaintainerTests ? [] : existingTestExcludes(
    target: "SmeltRuntimeTests",
    names: [
        "CAMCommandInventoryLintTests.swift",
        "CAMSelectorDeletionScannerToolTests.swift",
        "ReleaseVerifyHarnessTests.swift",
        "TTSVerifyHarnessTests.swift",
        "TextPackageVerifyHarnessTests.swift",
    ]
)

let package = Package(
    name: "Smelt",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SmeltSchema", targets: ["SmeltSchema"]),
        .library(name: "SmeltCompiler", targets: ["SmeltCompiler"]),
        .library(name: "SmeltRuntime", targets: ["SmeltRuntime"]),
        .library(name: "SmeltServe", targets: ["SmeltServe"]),
        .library(name: "SmeltModuleAuthoring", targets: ["SmeltModuleAuthoring"]),
        .library(name: "SmeltModels", targets: ["SmeltModels"]),
        .executable(name: "smelt", targets: ["SmeltCLI"]),
        .executable(name: "smelt-probe", targets: ["SmeltProbe"]),
        .executable(name: "smelt-models", targets: ["SmeltModelsCLI"]),
    ],
    targets: [
        // Shared schema: manifest, buffer table, enums — used by both compiler and runtime.
        .target(
            name: "SmeltSchema",
            path: "Sources/SmeltSchema"
        ),
        .target(
            name: "SmeltCompiler",
            dependencies: ["SmeltSchema", "SmeltRuntime"],
            path: "Sources/SmeltCompiler"
        ),
        // Swift module authoring: a thin sugar layer over the module IR member
        // structs in SmeltSchema. Depends ONLY on SmeltSchema — physically
        // cannot import the compiler, runtime, or Metal (containment lint).
        .target(
            name: "SmeltModuleAuthoring",
            dependencies: ["SmeltSchema"],
            path: "Sources/SmeltModuleAuthoring"
        ),
        // Model definitions as Swift values. One public function per model;
        // the qwen35 trio is one parameterized function. Depends only on
        // SmeltModuleAuthoring (transitively SmeltSchema).
        .target(
            name: "SmeltModels",
            dependencies: ["SmeltModuleAuthoring"],
            path: "Sources/SmeltModels"
        ),
        // `smelt-models emit --output Models` writes <id>.module.json for every
        // definition using canonicalJSONData(prettyPrinted: true).
        .executableTarget(
            name: "SmeltModelsCLI",
            dependencies: ["SmeltModels", "SmeltModuleAuthoring", "SmeltSchema"],
            path: "Sources/SmeltModelsCLI"
        ),
        llguidanceTarget,
        .target(
            name: "SmeltRuntime",
            dependencies: ["SmeltSchema", "CLLGuidance"],
            path: "Sources/SmeltRuntime"
        ),
        .target(
            name: "SmeltServe",
            dependencies: ["SmeltRuntime", "SmeltSchema"],
            path: "Sources/SmeltServe"
        ),
        .executableTarget(
            name: "SmeltCLI",
            dependencies: ["SmeltCompiler", "SmeltRuntime", "SmeltServe", "SmeltSchema"],
            path: "Sources/SmeltCLI"
        ),
        .executableTarget(
            name: "SmeltProbe",
            dependencies: ["SmeltCompiler", "SmeltRuntime", "SmeltSchema"],
            path: "Sources/SmeltProbe"
        ),
        .testTarget(
            name: "SmeltSchemaTests",
            dependencies: ["SmeltSchema"],
            path: "Tests/SmeltSchemaTests"
        ),
        .testTarget(
            name: "SmeltCLITests",
            dependencies: ["SmeltCLI", "SmeltServe"],
            path: "Tests/SmeltCLITests"
        ),
        .testTarget(
            name: "SmeltCompilerTests",
            dependencies: ["SmeltCompiler", "SmeltRuntime", "SmeltModels", "SmeltModuleAuthoring"],
            path: "Tests/SmeltCompilerTests",
            exclude: compilerTestExcludes,
            resources: [.copy("Fixtures")],
            swiftSettings: [
                // Parity tests (SmeltGPTQParityTests, Qwen3TTSCodecGPUTests) call cblas_* directly;
                // the ACCELERATE_NEW_LAPACK define doesn't propagate from dependencies, so repeat it
                // here to clear the macOS 13.3 deprecation (no ILP64 → 32-bit ints, source-compatible).
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK=1"]),
            ]
        ),
        .testTarget(
            name: "SmeltRuntimeTests",
            dependencies: ["SmeltRuntime", "SmeltCompiler", "SmeltSchema", "SmeltModels"],
            path: "Tests/SmeltRuntimeTests",
            exclude: runtimeTestExcludes,
            resources: [.copy("Fixtures")],
            swiftSettings: [
                // Qwen3TTSCodecGateTests calls cblas_* directly; Swift
                // target settings don't propagate to dependent targets,
                // so the ACCELERATE_NEW_LAPACK define from SmeltRuntime
                // must be repeated here to clear the deprecation warning.
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK=1"]),
            ]
        ),
    ]
)
