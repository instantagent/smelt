import Testing
@testable import SmeltRuntime

// MARK: - SmeltToken

@Test func tokenConstruction() {
    let token = SmeltToken(id: 42, position: 7)
    #expect(token.id == 42)
    #expect(token.position == 7)
}

@Test func tokenIsSendable() {
    // Compile-time check: SmeltToken can cross isolation boundaries.
    let token = SmeltToken(id: 1, position: 0)
    let _: any Sendable = token
}

@Test func generateResultIsSendable() {
    let result = SmeltGenerateResult(
        tokens: [1, 2, 3],
        generateTime: 0.045,
        tokensPerSecond: 66.7,
        prefillTime: 0.040
    )
    let _: any Sendable = result
    #expect(result.tokens == [1, 2, 3])
    #expect(result.tokensPerSecond > 0)
}

// MARK: - SmeltModel Sendable conformance

@Test func modelIsSendable() {
    // Compile-time check: SmeltModel can be captured in @Sendable closures.
    // This is required for generateStream's AsyncThrowingStream.
    // We can't construct a model without a .smeltpkg, but we can
    // verify the conformance exists at the type level.
    // If SmeltModel lost Sendable conformance, this file would fail to compile:
    func acceptSendable<T: Sendable>(_ type: T.Type) {}
    acceptSendable(SmeltModel.self)
}
