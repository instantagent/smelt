import Testing
@testable import SmeltRuntime

@Test func runtimeModuleLoads() {
    // Smoke test: the runtime module imports.
    #expect(true, "SmeltTextRuntime module loaded")
}
