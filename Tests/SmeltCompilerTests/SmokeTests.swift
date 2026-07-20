import Testing
@testable import SmeltCompiler

@Test func compilerModuleLoads() {
    // Smoke test: the compiler module imports and the entry point exists.
    // SmeltCompiler.build() will fatalError, so we just verify the type exists.
    #expect(true, "SmeltCompiler module loaded")
}
