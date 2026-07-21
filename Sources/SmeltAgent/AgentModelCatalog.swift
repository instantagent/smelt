import SmeltRuntime

/// Agent-facing projection of Smelt's installed package catalog. The package
/// store remains opaque to Smelt agent; only Smelt resolves its layout.
package enum AgentModelCatalog {
    package static func installedPackagePaths() throws -> [String] {
        try SmeltPackageStore.installedPackages().map(\.packageURL.path)
    }
}
